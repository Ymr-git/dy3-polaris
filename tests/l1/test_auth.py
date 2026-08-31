"""L1 认证模块测试 — JWT 签发/验证/撤销 + 密码安全 + 用户生命周期管理.

测试覆盖:
1. PasswordHasher: PBKDF2-HMAC-SHA256 + pepper, 恒定时间比较
2. JWTManager: 签发/验证/撤销/刷新, Token 版本化
3. UserLifecycleManager: 注册/认证/变更/毕业/暂停/归档
4. 异常体系: AuthenticationError / TokenError / LifecycleError
5. 线程安全
6. 边界条件与异常处理
"""

import threading
import time

import pytest

from dy3_polaris.l1.auth import (
    AuthenticationError,
    DEFAULT_TOKEN_VERSION,
    JWTManager,
    L1AuthError,
    LifecycleError,
    PasswordHasher,
    TokenError,
    TokenExpiredError,
    TokenPayload,
    TokenRevokedError,
    UserLifecycleManager,
)
from dy3_polaris.l1.models import (
    ABACAttributes,
    AuditAction,
    AuditResult,
    GradeLevel,
    LabAccessTier,
    User,
    UserRole,
    UserStatus,
)


# ============================================================
# 1. PasswordHasher 测试
# ============================================================


class TestPasswordHasher:
    """密码哈希器测试."""

    def test_hash_password_returns_string(self):
        """哈希密码返回字符串."""
        h = PasswordHasher.hash_password("TestPass123!")
        assert isinstance(h, str)
        assert h.startswith("pbkdf2_sha256$")

    def test_hash_password_format(self):
        """哈希格式: pbkdf2_sha256${iterations}${salt_hex}${hash_hex}."""
        h = PasswordHasher.hash_password("TestPass123!")
        parts = h.split("$")
        assert len(parts) == 4
        assert parts[0] == "pbkdf2_sha256"
        assert int(parts[1]) >= 600000  # OWASP 推荐
        # salt 和 hash 都是 hex
        bytes.fromhex(parts[2])
        bytes.fromhex(parts[3])

    def test_verify_correct_password(self):
        """正确密码验证通过."""
        password = "MySecurePass2024!"
        stored = PasswordHasher.hash_password(password)
        assert PasswordHasher.verify_password(password, stored) is True

    def test_verify_wrong_password(self):
        """错误密码验证失败."""
        stored = PasswordHasher.hash_password("CorrectPass123!")
        assert PasswordHasher.verify_password("WrongPass456!", stored) is False

    def test_verify_empty_password(self):
        """空密码验证失败."""
        stored = PasswordHasher.hash_password("SomePass123!")
        assert PasswordHasher.verify_password("", stored) is False

    def test_verify_empty_hash(self):
        """空哈希验证失败."""
        assert PasswordHasher.verify_password("SomePass123!", "") is False

    def test_hash_empty_password_raises(self):
        """空密码哈希抛出异常."""
        with pytest.raises(ValueError):
            PasswordHasher.hash_password("")

    def test_hash_different_salts(self):
        """同一密码两次哈希结果不同 (随机盐)."""
        h1 = PasswordHasher.hash_password("SamePass123!")
        h2 = PasswordHasher.hash_password("SamePass123!")
        assert h1 != h2

    def test_verify_invalid_hash_format(self):
        """无效哈希格式验证失败."""
        assert PasswordHasher.verify_password("pass", "invalid_hash") is False
        assert PasswordHasher.verify_password("pass", "pbkdf2_sha256$abc$def") is False

    def test_needs_rehash_old_iterations(self):
        """旧迭代次数需要重新哈希."""
        # 模拟旧哈希 (低迭代次数)
        import hashlib
        import hmac
        import os

        old_iterations = 100000
        salt = os.urandom(32)
        peppered = hmac.new(
            PasswordHasher._PEPPER, b"pass", hashlib.sha256
        ).digest()
        dk = hashlib.pbkdf2_hmac("sha256", peppered, salt, old_iterations)
        old_hash = f"pbkdf2_sha256${old_iterations}${salt.hex()}${dk.hex()}"
        assert PasswordHasher.needs_rehash(old_hash) is True

    def test_needs_rehash_current_iterations(self):
        """当前迭代次数不需要重新哈希."""
        h = PasswordHasher.hash_password("pass123!")
        assert PasswordHasher.needs_rehash(h) is False

    def test_needs_rehash_invalid_format(self):
        """无效格式返回 True (需要重新哈希)."""
        assert PasswordHasher.needs_rehash("invalid") is True


# ============================================================
# 2. JWTManager 测试
# ============================================================


class TestJWTManager:
    """JWT 管理器测试."""

    def _make_user(self, role: UserRole = UserRole.UNDERGRAD) -> User:
        return User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=role,
            status=UserStatus.ACTIVE,
        )

    def test_issue_token_returns_pair(self):
        """签发 Token 返回 (access, refresh) 对."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        access, refresh = jwt.issue_token(user)
        assert isinstance(access, str)
        assert isinstance(refresh, str)
        assert access != refresh

    def test_issue_token_format(self):
        """Token 格式: header.payload.signature."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        access, _ = jwt.issue_token(user)
        parts = access.split(".")
        assert len(parts) == 3

    def test_verify_valid_token(self):
        """验证有效 Token 返回 payload."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        access, _ = jwt.issue_token(user)
        payload = jwt.verify_token(access)
        assert payload.user_id == user.user_id
        assert payload.role == user.role.value
        assert payload.token_type == "access"

    def test_verify_invalid_signature(self):
        """签名无效的 Token 验证失败."""
        jwt1 = JWTManager(secret_key="secret-1")
        jwt2 = JWTManager(secret_key="secret-2")
        user = self._make_user()
        access, _ = jwt1.issue_token(user)
        with pytest.raises(TokenError):
            jwt2.verify_token(access)

    def test_verify_tampered_token(self):
        """篡改的 Token 验证失败."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        access, _ = jwt.issue_token(user)
        # 篡改 payload 部分
        parts = access.split(".")
        tampered = f"{parts[0]}.{parts[1][:10]}x{parts[1][11:]}.{parts[2]}"
        with pytest.raises(TokenError):
            jwt.verify_token(tampered)

    def test_verify_expired_token(self):
        """过期 Token 验证失败."""
        jwt = JWTManager(secret_key="test-secret", token_ttl=-100)
        user = self._make_user()
        access, _ = jwt.issue_token(user)
        with pytest.raises(TokenExpiredError):
            jwt.verify_token(access)

    def test_revoke_token(self):
        """撤销 Token 后验证失败."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        access, _ = jwt.issue_token(user)
        jwt.revoke_token(access)
        with pytest.raises(TokenRevokedError):
            jwt.verify_token(access)

    def test_revoke_all_tokens(self):
        """版本化撤销: 所有旧 Token 失效."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        access, _ = jwt.issue_token(user)
        jwt.revoke_all_tokens(user.user_id)
        with pytest.raises(TokenRevokedError):
            jwt.verify_token(access)

    def test_refresh_token_rotation(self):
        """Refresh Token 轮换: 旧 Token 用后即废."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        _, refresh = jwt.issue_token(user)
        new_access, new_refresh = jwt.refresh_token(refresh)
        # 旧 refresh token 不可重用
        with pytest.raises(TokenRevokedError):
            jwt.refresh_token(refresh)
        # 新 access token 可验证
        payload = jwt.verify_token(new_access)
        assert payload.user_id == user.user_id

    def test_refresh_with_access_token_fails(self):
        """使用 access token 刷新失败."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        access, _ = jwt.issue_token(user)
        with pytest.raises(TokenError):
            jwt.refresh_token(access)

    def test_verify_refresh_token_directly(self):
        """直接验证 refresh token 失败 (类型不匹配)."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        _, refresh = jwt.issue_token(user)
        with pytest.raises(TokenError):
            jwt.verify_token(refresh)

    def test_issue_token_suspended_user(self):
        """已停用用户无法签发 Token."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        user.status = UserStatus.SUSPENDED
        with pytest.raises(LifecycleError):
            jwt.issue_token(user)

    def test_token_version_check(self):
        """Token 版本检查: 版本号不匹配则拒绝."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        access, _ = jwt.issue_token(user, token_version=1)
        # 升级版本
        jwt.revoke_all_tokens(user.user_id)
        # 旧版本 Token 被拒绝
        with pytest.raises(TokenRevokedError):
            jwt.verify_token(access)

    def test_get_token_version(self):
        """获取用户 Token 版本号."""
        jwt = JWTManager(secret_key="test-secret")
        user = self._make_user()
        jwt.issue_token(user, token_version=3)
        assert jwt.get_token_version(user.user_id) == 3

    def test_get_token_version_default(self):
        """未签发 Token 的用户版本号为默认值."""
        jwt = JWTManager(secret_key="test-secret")
        assert jwt.get_token_version("nonexistent") == DEFAULT_TOKEN_VERSION

    def test_token_payload_serialization(self):
        """TokenPayload 序列化/反序列化."""
        payload = TokenPayload(
            user_id="u-001",
            student_id="CS20240001",
            role="undergrad",
            institution_id="inst-001",
            token_id="jti-001",
            token_version=1,
            issued_at=1000,
            expires_at=2000,
            issuer="test-issuer",
            audience="test-aud",
            token_type="access",
        )
        d = payload.to_dict()
        restored = TokenPayload.from_dict(d)
        assert restored.user_id == payload.user_id
        assert restored.role == payload.role
        assert restored.token_type == payload.token_type

    def test_blacklist_cleanup(self):
        """黑名单过期条目自动清理."""
        jwt = JWTManager(secret_key="test-secret", token_ttl=1)
        user = self._make_user()
        access, _ = jwt.issue_token(user)
        jwt.revoke_token(access)
        assert jwt.get_blacklist_size() >= 1
        # 等待过期 (Token TTL=1秒)
        time.sleep(2)
        # 再次撤销触发清理
        jwt.revoke_all_tokens(user.user_id)
        # 过期条目应被清理
        assert jwt.get_blacklist_size() == 0


# ============================================================
# 3. UserLifecycleManager 测试
# ============================================================


class TestUserLifecycleManager:
    """用户生命周期管理器测试."""

    def _make_manager(self) -> UserLifecycleManager:
        return UserLifecycleManager(
            jwt_manager=JWTManager(secret_key="test-secret"),
        )

    def test_register_user(self):
        """注册新用户."""
        mgr = self._make_manager()
        user = mgr.register(
            student_id="CS20240001",
            institution_id="inst-001",
            password="Pass123!",
        )
        assert user.student_id == "CS20240001"
        assert user.role == UserRole.UNDERGRAD
        assert user.status == UserStatus.ACTIVE

    def test_register_duplicate_student_id(self):
        """重复学号注册失败."""
        mgr = self._make_manager()
        mgr.register("CS20240001", "inst-001", "Pass123!")
        with pytest.raises(LifecycleError):
            mgr.register("CS20240001", "inst-001", "Pass456!")

    def test_register_empty_password(self):
        """空密码注册失败."""
        mgr = self._make_manager()
        with pytest.raises(ValueError):
            mgr.register("CS20240001", "inst-001", "")

    def test_authenticate_success(self):
        """认证成功返回 user + tokens."""
        mgr = self._make_manager()
        mgr.register("CS20240001", "inst-001", "Pass123!")
        user, access, refresh = mgr.authenticate("CS20240001", "Pass123!")
        assert user.student_id == "CS20240001"
        assert isinstance(access, str)
        assert isinstance(refresh, str)

    def test_authenticate_wrong_password(self):
        """密码错误认证失败."""
        mgr = self._make_manager()
        mgr.register("CS20240001", "inst-001", "Pass123!")
        with pytest.raises(AuthenticationError):
            mgr.authenticate("CS20240001", "WrongPass!")

    def test_authenticate_nonexistent_user(self):
        """不存在的学号认证失败."""
        mgr = self._make_manager()
        with pytest.raises(AuthenticationError):
            mgr.authenticate("CS99999999", "Pass123!")

    def test_authenticate_suspended_user(self):
        """已停用用户认证失败."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        mgr.suspend(user.user_id)
        with pytest.raises(AuthenticationError):
            mgr.authenticate("CS20240001", "Pass123!")

    def test_change_role_undergrad_to_graduate(self):
        """本科生升级为研究生."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        updated = mgr.change_role(
            user.user_id, UserRole.GRADUATE, supervisor_id="teacher-001"
        )
        assert updated.role == UserRole.GRADUATE
        assert updated.abac_attributes.supervisor_id == "teacher-001"
        assert updated.abac_attributes.grade_level == GradeLevel.MASTER

    def test_change_role_graduate_without_supervisor(self):
        """本科生升级研究生未指定导师失败."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        with pytest.raises(ValueError):
            mgr.change_role(user.user_id, UserRole.GRADUATE)

    def test_change_role_nonexistent_user(self):
        """变更不存在用户的角色失败."""
        mgr = self._make_manager()
        with pytest.raises(LifecycleError):
            mgr.change_role("nonexistent", UserRole.GRADUATE)

    def test_graduate_user(self):
        """毕业降级为校友."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        graduated = mgr.graduate(user.user_id)
        assert graduated.role == UserRole.ALUMNI
        assert graduated.status == UserStatus.ALUMNI
        assert graduated.abac_attributes.lab_access_tier == LabAccessTier.TIER0

    def test_graduate_already_alumni(self):
        """校友不能再次毕业."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        mgr.graduate(user.user_id)
        with pytest.raises(LifecycleError):
            mgr.graduate(user.user_id)

    def test_suspend_user(self):
        """暂停用户."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        suspended = mgr.suspend(user.user_id)
        assert suspended.status == UserStatus.SUSPENDED

    def test_suspend_non_active_user(self):
        """非活跃用户不能暂停."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        mgr.suspend(user.user_id)
        with pytest.raises(LifecycleError):
            mgr.suspend(user.user_id)

    def test_reactivate_user(self):
        """恢复已暂停用户."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        mgr.suspend(user.user_id)
        reactivated = mgr.reactivate(user.user_id)
        assert reactivated.status == UserStatus.ACTIVE

    def test_reactivate_non_suspended(self):
        """非暂停用户不能恢复."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        with pytest.raises(LifecycleError):
            mgr.reactivate(user.user_id)

    def test_archive_user(self):
        """归档校友用户."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        mgr.graduate(user.user_id)
        archived = mgr.archive(user.user_id)
        assert archived.status == UserStatus.ALUMNI

    def test_archive_non_alumni(self):
        """非校友不能归档."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        with pytest.raises(LifecycleError):
            mgr.archive(user.user_id)

    def test_get_user(self):
        """获取用户信息."""
        mgr = self._make_manager()
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        found = mgr.get_user(user.user_id)
        assert found is not None
        assert found.student_id == "CS20240001"

    def test_get_user_by_student_id(self):
        """通过学号获取用户."""
        mgr = self._make_manager()
        mgr.register("CS20240001", "inst-001", "Pass123!")
        found = mgr.get_user_by_student_id("CS20240001")
        assert found is not None
        assert found.student_id == "CS20240001"

    def test_get_user_nonexistent(self):
        """获取不存在用户返回 None."""
        mgr = self._make_manager()
        assert mgr.get_user("nonexistent") is None

    def test_list_users(self):
        """列出所有用户."""
        mgr = self._make_manager()
        mgr.register("CS20240001", "inst-001", "Pass123!")
        mgr.register("CS20240002", "inst-001", "Pass456!")
        users = mgr.list_users()
        assert len(users) == 2

    def test_audit_logs_recorded(self):
        """审计日志正确记录."""
        mgr = self._make_manager()
        mgr.register("CS20240001", "inst-001", "Pass123!")
        mgr.authenticate("CS20240001", "Pass123!")
        logs = mgr.get_audit_logs()
        assert len(logs) >= 2  # 注册 + 登录
        assert any(log.action == AuditAction.LOGIN for log in logs)

    def test_failed_auth_logs_denied(self):
        """认证失败记录 DENIED 审计日志."""
        mgr = self._make_manager()
        mgr.register("CS20240001", "inst-001", "Pass123!")
        try:
            mgr.authenticate("CS20240001", "WrongPass!")
        except AuthenticationError:
            pass
        logs = mgr.get_audit_logs()
        denied_logs = [l for l in logs if l.result == AuditResult.DENIED]
        assert len(denied_logs) >= 1


# ============================================================
# 4. 异常体系测试
# ============================================================


class TestExceptionHierarchy:
    """异常体系测试."""

    def test_l1_auth_error_is_l6_error(self):
        """L1AuthError 继承 L6Error."""
        from dy3_polaris.l6.core.exceptions import L6Error
        assert issubclass(L1AuthError, L6Error)

    def test_authentication_error_is_l1_auth_error(self):
        """AuthenticationError 继承 L1AuthError."""
        assert issubclass(AuthenticationError, L1AuthError)

    def test_token_error_is_l1_auth_error(self):
        """TokenError 继承 L1AuthError."""
        assert issubclass(TokenError, L1AuthError)

    def test_token_expired_is_token_error(self):
        """TokenExpiredError 继承 TokenError."""
        assert issubclass(TokenExpiredError, TokenError)

    def test_token_revoked_is_token_error(self):
        """TokenRevokedError 继承 TokenError."""
        assert issubclass(TokenRevokedError, TokenError)

    def test_lifecycle_error_is_l1_auth_error(self):
        """LifecycleError 继承 L1AuthError."""
        assert issubclass(LifecycleError, L1AuthError)

    def test_authentication_error_jsonrpc_code(self):
        """AuthenticationError JSON-RPC 码."""
        err = AuthenticationError("test", user_id="u-001")
        assert err._jsonrpc_code() == -32201

    def test_token_error_jsonrpc_code(self):
        """TokenError JSON-RPC 码."""
        err = TokenError("test")
        assert err._jsonrpc_code() == -32202

    def test_token_expired_jsonrpc_code(self):
        """TokenExpiredError JSON-RPC 码."""
        err = TokenExpiredError()
        assert err._jsonrpc_code() == -32203

    def test_token_revoked_jsonrpc_code(self):
        """TokenRevokedError JSON-RPC 码."""
        err = TokenRevokedError()
        assert err._jsonrpc_code() == -32204

    def test_lifecycle_error_jsonrpc_code(self):
        """LifecycleError JSON-RPC 码."""
        err = LifecycleError("test")
        assert err._jsonrpc_code() == -32205

    def test_authentication_error_context(self):
        """AuthenticationError 包含 user_id 上下文."""
        err = AuthenticationError("认证失败", user_id="u-001")
        assert err.context["user_id"] == "u-001"

    def test_lifecycle_error_context(self):
        """LifecycleError 包含 user_id 上下文."""
        err = LifecycleError("生命周期错误", user_id="u-001")
        assert err.context["user_id"] == "u-001"


# ============================================================
# 5. 线程安全测试
# ============================================================


class TestAuthThreadSafety:
    """认证模块线程安全测试."""

    def test_concurrent_register(self):
        """并发注册不抛异常."""
        mgr = UserLifecycleManager(
            jwt_manager=JWTManager(secret_key="test-secret"),
        )
        errors = []

        def worker(idx):
            try:
                mgr.register(
                    f"CS{idx:08d}",
                    "inst-001",
                    f"Pass{idx}!",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(mgr.list_users()) == 20

    def test_concurrent_authenticate(self):
        """并发认证不抛异常."""
        mgr = UserLifecycleManager(
            jwt_manager=JWTManager(secret_key="test-secret"),
        )
        mgr.register("CS20240001", "inst-001", "Pass123!")
        errors = []

        def worker():
            try:
                for _ in range(10):
                    mgr.authenticate("CS20240001", "Pass123!")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_jwt_issue_verify(self):
        """并发 JWT 签发和验证不抛异常."""
        jwt = JWTManager(secret_key="test-secret")
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        errors = []

        def worker():
            try:
                for _ in range(20):
                    access, _ = jwt.issue_token(user)
                    jwt.verify_token(access)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================
# 6. 集成测试
# ============================================================


class TestAuthIntegration:
    """认证与权限集成测试."""

    def test_full_lifecycle(self):
        """完整用户生命周期: 注册→认证→变更→毕业→归档."""
        mgr = UserLifecycleManager(
            jwt_manager=JWTManager(secret_key="test-secret"),
        )

        # 1. 注册
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        assert user.status == UserStatus.ACTIVE

        # 2. 认证
        user, access, refresh = mgr.authenticate("CS20240001", "Pass123!")
        assert access  # Token 非空

        # 3. 角色变更 (本科→研究生)
        user = mgr.change_role(
            user.user_id, UserRole.GRADUATE, supervisor_id="teacher-001"
        )
        assert user.role == UserRole.GRADUATE

        # 4. 毕业
        user = mgr.graduate(user.user_id)
        assert user.status == UserStatus.ALUMNI

        # 5. 归档
        user = mgr.archive(user.user_id)
        assert user.status == UserStatus.ALUMNI

    def test_token_revocation_on_graduate(self):
        """毕业时所有 Token 被撤销."""
        mgr = UserLifecycleManager(
            jwt_manager=JWTManager(secret_key="test-secret"),
        )
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        _, access, _ = mgr.authenticate("CS20240001", "Pass123!")

        # 毕业前 Token 有效
        mgr._jwt_manager.verify_token(access)

        # 毕业 → Token 撤销
        mgr.graduate(user.user_id)
        with pytest.raises(TokenRevokedError):
            mgr._jwt_manager.verify_token(access)

    def test_token_revocation_on_suspend(self):
        """暂停时所有 Token 被撤销."""
        mgr = UserLifecycleManager(
            jwt_manager=JWTManager(secret_key="test-secret"),
        )
        user = mgr.register("CS20240001", "inst-001", "Pass123!")
        _, access, _ = mgr.authenticate("CS20240001", "Pass123!")

        mgr.suspend(user.user_id)
        with pytest.raises(TokenRevokedError):
            mgr._jwt_manager.verify_token(access)
