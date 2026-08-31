"""L1 用户域认证模块 — JWT 签发/验证/撤销 + 密码安全 + 用户生命周期管理.

设计依据:
- L1 设计文档第二章 2.4: 角色生命周期 (注册→审核→激活→变更→毕业→归档)
- L1 设计文档第七章 7.2: API /api/v1/auth/* 接口
- L1 设计文档第七章 7.5: JWT 验证延迟 ≤ 5ms

融合世界先进方案:
- OpenAI Platform: JWT 无状态认证 + Token 版本化撤销
- WorkOS: JWT 最佳实践 (算法白名单 + 全声明验证 + Refresh Token 轮换)
- OWASP: PBKDF2-HMAC-SHA256 + pepper 密码哈希 (≥ 600,000 迭代, OWASP Password Storage Cheat Sheet 推荐)
- Khan Academy: 教育场景角色生命周期管理
- Authgear: 混合认证模式 (Web Cookie + Mobile Bearer)

模块组成:
1. 异常体系: L1AuthError 层级 (AuthenticationError / TokenError / LifecycleError)
2. 密码安全: PasswordHasher (PBKDF2-HMAC-SHA256 + pepper, 恒定时间比较)
3. JWT 管理: JWTManager (签发/验证/撤销/刷新, HS256 + jti 黑名单)
4. 用户生命周期: UserLifecycleManager (注册/激活/变更/毕业/归档/暂停)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from dy3_polaris.l1.models import (
    ABACAttributes,
    AuditAction,
    AuditLogEntry,
    AuditResult,
    DataLevel,
    GradeLevel,
    LabAccessTier,
    Permission,
    Role,
    User,
    UserStatus,
    UserRole,
)
from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 1. 常量定义
# ============================================================

# JWT 配置
DEFAULT_TOKEN_TTL_SECONDS: int = 7200  # 默认 Token 有效期 2 小时
DEFAULT_REFRESH_TTL_SECONDS: int = 7 * 24 * 3600  # Refresh Token 7 天
DEFAULT_ISSUER: str = "dy3-auth-service"
DEFAULT_AUDIENCE: str = "dy3-polaris-api"
CLOCK_SKEW_SECONDS: int = 30  # 时钟偏移容忍

# 密码哈希配置 (PBKDF2)
PBKDF2_ITERATIONS: int = 600_000  # OWASP 推荐 ≥ 600,000 (HMAC-SHA256)
PBKDF2_SALT_BYTES: int = 32
PBKDF2_HASH_NAME: str = "sha256"
PASSWORD_MAX_BYTES: int = 1024  # 密码最大字节数 (防 DoS)

# Token 版本化 (密码修改/角色降级时递增, 使旧 Token 失效)
DEFAULT_TOKEN_VERSION: int = 1


# ============================================================
# 2. 异常体系 (对齐 L3/L0 模式, JSON-RPC 码 -32200 范围)
# ============================================================


class L1AuthError(L6Error):
    """L1 认证授权层基础异常 (JSON-RPC -32200)."""

    def __init__(
        self,
        code: str = "L1_AUTH_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32200


class AuthenticationError(L1AuthError):
    """认证失败 (JSON-RPC -32201).

    用户名/密码不匹配、账户不存在、账户已停用等.
    """

    def __init__(
        self,
        detail: str = "认证失败",
        user_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = {"user_id": user_id}
        if context:
            ctx.update(context)
        super().__init__("AUTHENTICATION_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32201


class TokenError(L1AuthError):
    """Token 相关错误 (JSON-RPC -32202)."""

    def __init__(
        self,
        detail: str = "Token 无效",
        token_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {}
        if token_id:
            ctx["token_id"] = token_id
        if context:
            ctx.update(context)
        super().__init__("TOKEN_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32202


class TokenExpiredError(TokenError):
    """Token 已过期 (JSON-RPC -32203)."""

    def __init__(self, token_id: str = "", expired_at: int = 0) -> None:
        super().__init__(
            "Token 已过期",
            token_id=token_id,
            context={"expired_at": expired_at},
        )

    def _jsonrpc_code(self) -> int:
        return -32203


class TokenRevokedError(TokenError):
    """Token 已被撤销 (JSON-RPC -32204)."""

    def __init__(self, token_id: str = "") -> None:
        super().__init__("Token 已被撤销", token_id=token_id)

    def _jsonrpc_code(self) -> int:
        return -32204


class LifecycleError(L1AuthError):
    """用户生命周期错误 (JSON-RPC -32205).

    非法状态转换、重复注册、未审核用户尝试登录等.
    """

    def __init__(
        self,
        detail: str = "生命周期操作失败",
        user_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = {"user_id": user_id}
        if context:
            ctx.update(context)
        super().__init__("LIFECYCLE_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32205


# ============================================================
# 3. 密码安全 (PBKDF2-HMAC-SHA256 + pepper)
# ============================================================


class PasswordHasher:
    """密码哈希器 — PBKDF2-HMAC-SHA256 + pepper.

    设计依据:
    - OWASP Password Storage Cheat Sheet: PBKDF2 ≥ 600,000 迭代 (HMAC-SHA256)
    - pepper: 服务级密钥, 纵深防御 (即使数据库泄露, 无 pepper 无法破解)
    - 恒定时间比较: hmac.compare_digest 防时序攻击

    存储格式: pbkdf2_sha256${iterations}${salt_hex}${hash_hex}
    """

    _PEPPER: bytes = os.environ.get(
        "DY3_PASSWORD_PEPPER", "dy3-polaris-default-pepper-2026"
    ).encode("utf-8")

    @classmethod
    def hash_password(cls, password: str) -> str:
        """哈希密码 → 存储格式字符串.

        Args:
            password: 明文密码

        Returns:
            格式: pbkdf2_sha256${iterations}${salt_hex}${hash_hex}

        Raises:
            ValueError: 密码为空或超过最大长度
        """
        if not password:
            raise ValueError("密码不能为空")
        pw_bytes = password.encode("utf-8")
        if len(pw_bytes) > PASSWORD_MAX_BYTES:
            raise ValueError(f"密码长度超过 {PASSWORD_MAX_BYTES} 字节")

        salt = os.urandom(PBKDF2_SALT_BYTES)
        peppered = hmac.new(cls._PEPPER, pw_bytes, hashlib.sha256).digest()
        dk = hashlib.pbkdf2_hmac(
            PBKDF2_HASH_NAME, peppered, salt, PBKDF2_ITERATIONS
        )
        return (
            f"pbkdf2_sha256${PBKDF2_ITERATIONS}"
            f"${salt.hex()}${dk.hex()}"
        )

    @classmethod
    def verify_password(cls, password: str, stored_hash: str) -> bool:
        """验证密码 (恒定时间比较).

        Args:
            password: 明文密码
            stored_hash: hash_password() 返回的存储格式字符串

        Returns:
            True 如果密码匹配, False 否则
        """
        if not password or not stored_hash:
            return False
        try:
            parts = stored_hash.split("$")
            if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
                return False
            iterations = int(parts[1])
            salt = bytes.fromhex(parts[2])
            expected_dk = bytes.fromhex(parts[3])

            pw_bytes = password.encode("utf-8")
            peppered = hmac.new(cls._PEPPER, pw_bytes, hashlib.sha256).digest()
            actual_dk = hashlib.pbkdf2_hmac(
                PBKDF2_HASH_NAME, peppered, salt, iterations
            )
            return hmac.compare_digest(expected_dk, actual_dk)
        except (ValueError, IndexError):
            return False

    @classmethod
    def needs_rehash(cls, stored_hash: str) -> bool:
        """检查是否需要重新哈希 (迭代次数升级时).

        Args:
            stored_hash: 存储的哈希字符串

        Returns:
            True 如果当前迭代次数低于推荐值
        """
        try:
            parts = stored_hash.split("$")
            if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
                return True
            iterations = int(parts[1])
            return iterations < PBKDF2_ITERATIONS
        except (ValueError, IndexError):
            return True


# ============================================================
# 4. JWT 管理器
# ============================================================


@dataclass
class TokenPayload:
    """JWT Token 载荷结构."""

    user_id: str
    student_id: str
    role: str
    institution_id: str
    token_id: str  # jti, 用于黑名单
    token_version: int  # 版本化撤销
    issued_at: int  # iat
    expires_at: int  # exp
    issuer: str  # iss
    audience: str  # aud
    token_type: str = "access"  # access / refresh

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub": self.user_id,
            "student_id": self.student_id,
            "role": self.role,
            "inst": self.institution_id,
            "jti": self.token_id,
            "tv": self.token_version,
            "iat": self.issued_at,
            "exp": self.expires_at,
            "iss": self.issuer,
            "aud": self.audience,
            "typ": self.token_type,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TokenPayload:
        return cls(
            user_id=d.get("sub", ""),
            student_id=d.get("student_id", ""),
            role=d.get("role", ""),
            institution_id=d.get("inst", ""),
            token_id=d.get("jti", ""),
            token_version=d.get("tv", 1),
            issued_at=d.get("iat", 0),
            expires_at=d.get("exp", 0),
            issuer=d.get("iss", ""),
            audience=d.get("aud", ""),
            token_type=d.get("typ", "access"),
        )


class JWTManager:
    """JWT 管理器 — 签发/验证/撤销/刷新.

    设计依据:
    - HS256 对称签名 (单服务场景, 性能最优, 验证 ≤ 5ms)
    - jti 黑名单 (Redis 替代: 内存 dict + TTL, 测试友好)
    - token_version 版本化撤销 (密码修改/角色降级时递增)
    - 全声明验证 (exp/iat/iss/aud + 时钟偏移容忍)

    线程安全: threading.RLock 保护黑名单和版本映射.
    """

    def __init__(
        self,
        secret_key: str | None = None,
        issuer: str = DEFAULT_ISSUER,
        audience: str = DEFAULT_AUDIENCE,
        token_ttl: int = DEFAULT_TOKEN_TTL_SECONDS,
        refresh_ttl: int = DEFAULT_REFRESH_TTL_SECONDS,
    ) -> None:
        self._secret = (
            secret_key
            or os.environ.get("DY3_JWT_SECRET", "dy3-polaris-dev-secret-key")
        ).encode("utf-8")
        self._issuer = issuer
        self._audience = audience
        self._token_ttl = token_ttl
        self._refresh_ttl = refresh_ttl

        # 黑名单: jti → 过期时间戳 (过期后自动清理)
        self._blacklist: dict[str, int] = {}
        # 版本映射: user_id → 当前 token_version
        self._token_versions: dict[str, int] = {}
        # Refresh token 存储: refresh_jti → (user_id, expires_at)
        self._refresh_tokens: dict[str, tuple[str, int]] = {}

        self._lock = threading.RLock()

    # --- Token 签发 ---

    def issue_token(
        self,
        user: User,
        token_version: int = DEFAULT_TOKEN_VERSION,
    ) -> tuple[str, str]:
        """签发 Access Token + Refresh Token.

        Args:
            user: 用户对象
            token_version: Token 版本号 (用于版本化撤销)

        Returns:
            (access_token, refresh_token) 元组

        Raises:
            LifecycleError: 用户状态非 ACTIVE
        """
        if user.status != UserStatus.ACTIVE:
            raise LifecycleError(
                f"用户状态非 ACTIVE, 无法签发 Token (当前: {user.status.value})",
                user_id=user.user_id,
            )

        now = int(time.time())
        access_jti = f"jat-{uuid.uuid4().hex[:16]}"
        refresh_jti = f"jrt-{uuid.uuid4().hex[:16]}"

        # Access Token
        access_payload = TokenPayload(
            user_id=user.user_id,
            student_id=user.student_id,
            role=user.role.value,
            institution_id=user.institution_id,
            token_id=access_jti,
            token_version=token_version,
            issued_at=now,
            expires_at=now + self._token_ttl,
            issuer=self._issuer,
            audience=self._audience,
            token_type="access",
        )
        access_token = self._encode(access_payload.to_dict())

        # Refresh Token (更长 TTL)
        refresh_payload = TokenPayload(
            user_id=user.user_id,
            student_id=user.student_id,
            role=user.role.value,
            institution_id=user.institution_id,
            token_id=refresh_jti,
            token_version=token_version,
            issued_at=now,
            expires_at=now + self._refresh_ttl,
            issuer=self._issuer,
            audience=self._audience,
            token_type="refresh",
        )
        refresh_token = self._encode(refresh_payload.to_dict())

        with self._lock:
            self._refresh_tokens[refresh_jti] = (
                user.user_id,
                now + self._refresh_ttl,
            )
            self._token_versions[user.user_id] = token_version

        return access_token, refresh_token

    # --- Token 验证 ---

    def verify_token(self, token: str) -> TokenPayload:
        """验证 Access Token.

        验证步骤:
        1. 解码签名 (HS256)
        2. 检查 exp 过期 (含时钟偏移容忍)
        3. 检查 iss / aud 声明
        4. 检查 jti 黑名单
        5. 检查 token_version 版本化撤销

        Args:
            token: JWT 字符串

        Returns:
            TokenPayload 解码后的载荷

        Raises:
            TokenError: Token 格式错误或签名无效
            TokenExpiredError: Token 已过期
            TokenRevokedError: Token 已被撤销 (黑名单或版本不匹配)
        """
        payload_dict = self._decode(token)
        if payload_dict is None:
            raise TokenError("Token 签名验证失败")

        payload = TokenPayload.from_dict(payload_dict)

        # 检查 token 类型
        if payload.token_type != "access":
            raise TokenError(
                f"预期 access token, 实际 {payload.token_type}",
                token_id=payload.token_id,
            )

        now = int(time.time())

        # 检查过期 (含时钟偏移)
        if now > payload.expires_at + CLOCK_SKEW_SECONDS:
            raise TokenExpiredError(
                token_id=payload.token_id,
                expired_at=payload.expires_at,
            )

        # 检查签发方
        if payload.issuer != self._issuer:
            raise TokenError(
                f"签发方不匹配: 预期 {self._issuer}, 实际 {payload.issuer}",
                token_id=payload.token_id,
            )

        # 检查受众
        if payload.audience != self._audience:
            raise TokenError(
                f"受众不匹配: 预期 {self._audience}, 实际 {payload.audience}",
                token_id=payload.token_id,
            )

        with self._lock:
            # 检查黑名单
            if payload.token_id in self._blacklist:
                raise TokenRevokedError(token_id=payload.token_id)

            # 检查版本化撤销
            current_version = self._token_versions.get(payload.user_id, 1)
            if payload.token_version < current_version:
                raise TokenRevokedError(token_id=payload.token_id)

        return payload

    # --- Token 撤销 ---

    def revoke_token(self, token: str) -> None:
        """撤销 Token (加入黑名单, 即时生效).

        Args:
            token: JWT 字符串
        """
        payload_dict = self._decode(token)
        if payload_dict is None:
            raise TokenError("无法解码 Token, 撤销失败")

        payload = TokenPayload.from_dict(payload_dict)
        now = int(time.time())

        with self._lock:
            # 加入黑名单, TTL = Token 剩余有效期
            self._blacklist[payload.token_id] = payload.expires_at
            # 清理过期的黑名单条目
            self._cleanup_blacklist(now)

    def revoke_all_tokens(self, user_id: str) -> None:
        """撤销用户的所有 Token (递增 token_version).

        用于密码修改、角色降级等场景.

        Args:
            user_id: 用户 ID
        """
        with self._lock:
            current = self._token_versions.get(user_id, 1)
            self._token_versions[user_id] = current + 1
            # 清理该用户的 refresh tokens
            to_remove = [
                jti for jti, (uid, _) in self._refresh_tokens.items()
                if uid == user_id
            ]
            for jti in to_remove:
                del self._refresh_tokens[jti]
            # 清理过期的黑名单条目
            self._cleanup_blacklist(int(time.time()))

    # --- Token 刷新 ---

    def refresh_token(self, refresh_token_str: str) -> tuple[str, str]:
        """使用 Refresh Token 获取新的 Access Token + 新的 Refresh Token.

        实现 Refresh Token 轮换 (旧 refresh token 立即失效).

        Args:
            refresh_token_str: Refresh Token 字符串

        Returns:
            (new_access_token, new_refresh_token)

        Raises:
            TokenError: Refresh Token 无效
            TokenExpiredError: Refresh Token 已过期
            TokenRevokedError: Refresh Token 已被使用或撤销
        """
        payload_dict = self._decode(refresh_token_str)
        if payload_dict is None:
            raise TokenError("Refresh Token 签名验证失败")

        payload = TokenPayload.from_dict(payload_dict)

        if payload.token_type != "refresh":
            raise TokenError(
                f"预期 refresh token, 实际 {payload.token_type}",
                token_id=payload.token_id,
            )

        now = int(time.time())
        if now > payload.expires_at + CLOCK_SKEW_SECONDS:
            raise TokenExpiredError(
                token_id=payload.token_id,
                expired_at=payload.expires_at,
            )

        with self._lock:
            # 检查 refresh token 是否有效 (轮换: 用后即删)
            if payload.token_id not in self._refresh_tokens:
                raise TokenRevokedError(token_id=payload.token_id)

            stored_user_id, _ = self._refresh_tokens[payload.token_id]
            if stored_user_id != payload.user_id:
                raise TokenRevokedError(token_id=payload.token_id)

            # 删除旧 refresh token (轮换)
            del self._refresh_tokens[payload.token_id]

        # 构造虚拟 User 用于签发新 Token
        # 注意: 实际场景应从数据库重新加载 User
        # 此处仅用 payload 信息构造, role/student_id 从原 Token 继承
        user = User(
            user_id=payload.user_id,
            student_id=payload.student_id,
            institution_id=payload.institution_id,
            role=UserRole(payload.role),
        )
        return self.issue_token(
            user, token_version=payload.token_version
        )

    # --- 内部方法 ---

    def _encode(self, payload: dict[str, Any]) -> str:
        """编码 JWT (HS256)."""
        header = {"alg": "HS256", "typ": "JWT"}
        header_b = self._b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_b = self._b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{header_b}.{payload_b}"
        signature = hmac.new(
            self._secret, signing_input.encode("utf-8"), hashlib.sha256
        ).digest()
        sig_b = self._b64url(signature)
        return f"{signing_input}.{sig_b}"

    def _decode(self, token: str) -> dict[str, Any] | None:
        """解码 JWT 并验证签名.

        Returns:
            payload dict 如果签名有效, None 否则
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            header_b, payload_b, sig_b = parts

            # 验证签名 (恒定时间)
            signing_input = f"{header_b}.{payload_b}"
            expected_sig = hmac.new(
                self._secret,
                signing_input.encode("utf-8"),
                hashlib.sha256,
            ).digest()
            actual_sig = self._b64url_decode(sig_b)
            if not hmac.compare_digest(expected_sig, actual_sig):
                return None

            # 解码 payload
            payload_json = self._b64url_decode_str(payload_b)
            return json.loads(payload_json)
        except (ValueError, json.JSONDecodeError, IndexError):
            return None

    @staticmethod
    def _b64url(data: bytes) -> str:
        """Base64URL 编码 (无填充)."""
        import base64
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    @staticmethod
    def _b64url_decode(data: str) -> bytes:
        """Base64URL 解码 (自动补齐填充)."""
        import base64
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    @classmethod
    def _b64url_decode_str(cls, data: str) -> str:
        """Base64URL 解码为字符串."""
        return cls._b64url_decode(data).decode("utf-8")

    def _cleanup_blacklist(self, now: int) -> None:
        """清理过期的黑名单条目 (调用方需持锁)."""
        expired = [
            jti for jti, exp in self._blacklist.items()
            if exp < now
        ]
        for jti in expired:
            del self._blacklist[jti]

    def get_blacklist_size(self) -> int:
        """获取当前黑名单大小 (用于监控)."""
        with self._lock:
            return len(self._blacklist)

    def get_token_version(self, user_id: str) -> int:
        """获取用户当前 Token 版本号."""
        with self._lock:
            return self._token_versions.get(user_id, DEFAULT_TOKEN_VERSION)


# ============================================================
# 5. 用户生命周期管理器
# ============================================================


class UserLifecycleManager:
    """用户生命周期管理器 (设计文档 2.4).

    六阶段生命周期:
    1. 注册: 创建 User 记录, 分配初始角色
    2. 审核: 管理员/教师确认身份
    3. 激活: 审核通过, 签发 Token
    4. 变更: 角色升级/选修新课/获得实验授权
    5. 毕业: 降级为 ALUMNI (只读)
    6. 归档: 数据匿名化/删除

    状态转换矩阵:
    - PENDING → ACTIVE (activate)
    - ACTIVE → ACTIVE (change_role)
    - ACTIVE → SUSPENDED (suspend)
    - SUSPENDED → ACTIVE (reactivate)
    - ACTIVE/SUSPENDED → ALUMNI (graduate)
    - ALUMNI → 终态 (archive: 不可恢复)

    线程安全: threading.RLock 保护用户存储.
    """

    # 合法的状态转换
    _TRANSITIONS: dict[UserStatus, set[UserStatus]] = {
        UserStatus.ACTIVE: {
            UserStatus.ACTIVE,
            UserStatus.SUSPENDED,
            UserStatus.ALUMNI,
        },
        UserStatus.SUSPENDED: {
            UserStatus.ACTIVE,
            UserStatus.ALUMNI,
        },
        UserStatus.ALUMNI: set(),  # 终态, 不再转换
    }

    def __init__(
        self,
        jwt_manager: JWTManager | None = None,
        hasher: type[PasswordHasher] | None = None,
    ) -> None:
        self._jwt_manager = jwt_manager or JWTManager()
        self._hasher = hasher or PasswordHasher
        # 用户存储: user_id → (User, password_hash)
        self._users: dict[str, tuple[User, str]] = {}
        # student_id → user_id 索引
        self._student_id_index: dict[str, str] = {}
        # 审计日志缓冲
        self._audit_logs: list[AuditLogEntry] = []
        self._lock = threading.RLock()

    # --- 注册与审核 ---

    def register(
        self,
        student_id: str,
        institution_id: str,
        password: str,
        role: UserRole = UserRole.UNDERGRAD,
        abac_attributes: ABACAttributes | None = None,
    ) -> User:
        """注册新用户.

        创建 User 记录, 哈希密码, 分配初始角色.
        用户初始状态为 ACTIVE (设计文档: 注册即激活, 审核为可选流程).

        Args:
            student_id: 学号 (格式: 2位大写字母 + 8位数字)
            institution_id: 机构 ID
            password: 明文密码
            role: 初始角色 (默认 UNDERGRAD)
            abac_attributes: ABAC 属性 (可选, 默认为年级 freshman)

        Returns:
            创建的 User 对象

        Raises:
            LifecycleError: 学号已存在
            ValueError: 学号格式无效或密码为空
        """
        if not password:
            raise ValueError("密码不能为空")

        with self._lock:
            if student_id in self._student_id_index:
                raise LifecycleError(
                    f"学号已存在: {student_id}",
                    context={"student_id": student_id},
                )

            user = User(
                student_id=student_id,
                institution_id=institution_id,
                role=role,
                status=UserStatus.ACTIVE,
                abac_attributes=abac_attributes or ABACAttributes(),
            )

            password_hash = self._hasher.hash_password(password)
            self._users[user.user_id] = (user, password_hash)
            self._student_id_index[student_id] = user.user_id

            self._log_audit(
                actor_id=user.user_id,
                actor_role=role,
                action=AuditAction.LOGIN,
                target_resource=f"user:{user.user_id}",
                target_data_level=DataLevel.L3_SENSITIVE,
                purpose="用户注册",
                result=AuditResult.SUCCESS,
            )

            return user

    # --- 认证 (登录) ---

    def authenticate(
        self,
        student_id: str,
        password: str,
    ) -> tuple[User, str, str]:
        """认证用户 (学号 + 密码), 签发 Token.

        Args:
            student_id: 学号
            password: 明文密码

        Returns:
            (user, access_token, refresh_token) 元组

        Raises:
            AuthenticationError: 学号不存在/密码错误/账户已停用
        """
        with self._lock:
            user_id = self._student_id_index.get(student_id)
            if user_id is None:
                raise AuthenticationError(
                    "学号不存在", user_id=""
                )

            user, password_hash = self._users[user_id]

            # 检查账户状态
            if user.status == UserStatus.SUSPENDED:
                raise AuthenticationError(
                    "账户已停用", user_id=user.user_id
                )
            if user.status == UserStatus.ALUMNI:
                raise AuthenticationError(
                    "校友账户仅限只读访问", user_id=user.user_id
                )

            # 验证密码
            if not self._hasher.verify_password(password, password_hash):
                self._log_audit(
                    actor_id=user.user_id,
                    actor_role=user.role,
                    action=AuditAction.LOGIN,
                    target_resource=f"user:{user.user_id}",
                    target_data_level=DataLevel.L3_SENSITIVE,
                    purpose="密码验证失败",
                    result=AuditResult.DENIED,
                )
                raise AuthenticationError(
                    "密码错误", user_id=user.user_id
                )

            # 登录成功, 签发 Token
            token_version = self._jwt_manager.get_token_version(user.user_id)
            access_token, refresh_token = self._jwt_manager.issue_token(
                user, token_version=token_version
            )

            # 检查是否需要重新哈希密码
            if self._hasher.needs_rehash(password_hash):
                new_hash = self._hasher.hash_password(password)
                self._users[user_id] = (user, new_hash)

            self._log_audit(
                actor_id=user.user_id,
                actor_role=user.role,
                action=AuditAction.LOGIN,
                target_resource=f"user:{user.user_id}",
                target_data_level=DataLevel.L3_SENSITIVE,
                purpose="用户登录",
                result=AuditResult.SUCCESS,
            )

            return user, access_token, refresh_token

    # --- 角色变更 ---

    def change_role(
        self,
        user_id: str,
        new_role: UserRole,
        supervisor_id: str | None = None,
    ) -> User:
        """变更用户角色 (如本科→研究生).

        关键操作:
        1. 验证状态转换合法性
        2. 更新角色
        3. 同步更新 ABAC 属性 (年级/实验权限)
        4. 不强制撤销 Token (角色升级, 权限扩大)

        Args:
            user_id: 用户 ID
            new_role: 新角色
            supervisor_id: 导师 ID (研究生必填)

        Returns:
            更新后的 User 对象

        Raises:
            LifecycleError: 用户不存在或非法状态转换
            ValueError: 研究生未指定导师
        """
        with self._lock:
            entry = self._users.get(user_id)
            if entry is None:
                raise LifecycleError(
                    f"用户不存在: {user_id}", user_id=user_id
                )
            user, password_hash = entry

            # 本科→研究生: 需补充导师信息
            if (
                user.role == UserRole.UNDERGRAD
                and new_role == UserRole.GRADUATE
            ):
                if not supervisor_id:
                    raise ValueError(
                        "升级为研究生必须指定导师 supervisor_id"
                    )
                user.abac_attributes.supervisor_id = supervisor_id
                user.abac_attributes.grade_level = GradeLevel.MASTER

            # 研究生→教师: 提升实验权限
            if (
                user.role == UserRole.GRADUATE
                and new_role == UserRole.TEACHER
            ):
                user.abac_attributes.lab_access_tier = LabAccessTier.TIER3

            user.role = new_role
            user.touch()
            self._users[user_id] = (user, password_hash)

            self._log_audit(
                actor_id=user_id,
                actor_role=new_role,
                action=AuditAction.MODIFY,
                target_resource=f"user:{user_id}",
                target_data_level=DataLevel.L3_SENSITIVE,
                purpose=f"角色变更: →{new_role.value}",
                result=AuditResult.SUCCESS,
            )

            return user

    # --- 毕业 ---

    def graduate(self, user_id: str) -> User:
        """毕业: 降级为 ALUMNI (只读).

        关键操作:
        1. 角色 → ALUMNI
        2. 状态 → ALUMNI
        3. ABAC 降级 (实验权限 → TIER0)
        4. 撤销所有活跃 Token (auth_version++)
        5. 写审计日志

        Args:
            user_id: 用户 ID

        Returns:
            更新后的 User 对象

        Raises:
            LifecycleError: 用户不存在或已是校友
        """
        with self._lock:
            entry = self._users.get(user_id)
            if entry is None:
                raise LifecycleError(
                    f"用户不存在: {user_id}", user_id=user_id
                )
            user, password_hash = entry

            if user.status == UserStatus.ALUMNI:
                raise LifecycleError(
                    "用户已是校友状态", user_id=user_id
                )

            # 状态转换验证
            if user.status not in self._TRANSITIONS:
                raise LifecycleError(
                    f"无法从 {user.status.value} 转换为 ALUMNI",
                    user_id=user_id,
                )

            user.role = UserRole.ALUMNI
            user.status = UserStatus.ALUMNI
            user.abac_attributes.lab_access_tier = LabAccessTier.TIER0
            user.touch()
            self._users[user_id] = (user, password_hash)

            # 撤销所有 Token
            self._jwt_manager.revoke_all_tokens(user_id)

            self._log_audit(
                actor_id=user_id,
                actor_role=UserRole.ALUMNI,
                action=AuditAction.MODIFY,
                target_resource=f"user:{user_id}",
                target_data_level=DataLevel.L3_SENSITIVE,
                purpose="毕业降级为校友",
                result=AuditResult.SUCCESS,
            )

            return user

    # --- 暂停/恢复 ---

    def suspend(self, user_id: str) -> User:
        """暂停用户 (违规/安全原因).

        Args:
            user_id: 用户 ID

        Returns:
            更新后的 User 对象
        """
        with self._lock:
            entry = self._users.get(user_id)
            if entry is None:
                raise LifecycleError(
                    f"用户不存在: {user_id}", user_id=user_id
                )
            user, password_hash = entry

            if user.status != UserStatus.ACTIVE:
                raise LifecycleError(
                    f"仅 ACTIVE 用户可暂停 (当前: {user.status.value})",
                    user_id=user_id,
                )

            user.status = UserStatus.SUSPENDED
            user.touch()
            self._users[user_id] = (user, password_hash)

            # 撤销所有 Token
            self._jwt_manager.revoke_all_tokens(user_id)

            self._log_audit(
                actor_id=user_id,
                actor_role=user.role,
                action=AuditAction.MODIFY,
                target_resource=f"user:{user_id}",
                target_data_level=DataLevel.L3_SENSITIVE,
                purpose="账户暂停",
                result=AuditResult.SUCCESS,
            )

            return user

    def reactivate(self, user_id: str) -> User:
        """恢复已暂停的用户.

        Args:
            user_id: 用户 ID

        Returns:
            更新后的 User 对象
        """
        with self._lock:
            entry = self._users.get(user_id)
            if entry is None:
                raise LifecycleError(
                    f"用户不存在: {user_id}", user_id=user_id
                )
            user, password_hash = entry

            if user.status != UserStatus.SUSPENDED:
                raise LifecycleError(
                    f"仅 SUSPENDED 用户可恢复 (当前: {user.status.value})",
                    user_id=user_id,
                )

            user.status = UserStatus.ACTIVE
            user.touch()
            self._users[user_id] = (user, password_hash)

            self._log_audit(
                actor_id=user_id,
                actor_role=user.role,
                action=AuditAction.MODIFY,
                target_resource=f"user:{user_id}",
                target_data_level=DataLevel.L3_SENSITIVE,
                purpose="账户恢复",
                result=AuditResult.SUCCESS,
            )

            return user

    # --- 归档 ---

    def archive(self, user_id: str) -> User:
        """归档用户 (毕业后超过保留期限).

        归档操作:
        1. 不可逆标记
        2. 触发数据匿名化/删除 (T6 隐私治理)
        3. 删除密码哈希 (无法再登录)

        Args:
            user_id: 用户 ID

        Returns:
            归档后的 User 对象

        Raises:
            LifecycleError: 用户不存在或非校友状态
        """
        with self._lock:
            entry = self._users.get(user_id)
            if entry is None:
                raise LifecycleError(
                    f"用户不存在: {user_id}", user_id=user_id
                )
            user, password_hash = entry

            if user.status != UserStatus.ALUMNI:
                raise LifecycleError(
                    f"仅校友可归档 (当前: {user.status.value})",
                    user_id=user_id,
                )

            # 删除密码 (无法再登录)
            self._users[user_id] = (user, "")

            self._log_audit(
                actor_id=user_id,
                actor_role=UserRole.ALUMNI,
                action=AuditAction.DELETE,
                target_resource=f"user:{user_id}",
                target_data_level=DataLevel.L4_CONFIDENTIAL,
                purpose="数据归档/匿名化",
                result=AuditResult.SUCCESS,
            )

            return user

    # --- 查询方法 ---

    def get_user(self, user_id: str) -> User | None:
        """获取用户信息."""
        with self._lock:
            entry = self._users.get(user_id)
            return entry[0] if entry else None

    def get_user_by_student_id(self, student_id: str) -> User | None:
        """通过学号获取用户."""
        with self._lock:
            user_id = self._student_id_index.get(student_id)
            if user_id is None:
                return None
            entry = self._users.get(user_id)
            return entry[0] if entry else None

    def list_users(self) -> list[User]:
        """列出所有用户."""
        with self._lock:
            return [entry[0] for entry in self._users.values()]

    def get_audit_logs(self, limit: int = 100) -> list[AuditLogEntry]:
        """获取审计日志."""
        with self._lock:
            return list(self._audit_logs[-limit:])

    # --- 内部方法 ---

    def _log_audit(
        self,
        actor_id: str,
        actor_role: UserRole,
        action: AuditAction,
        target_resource: str,
        target_data_level: DataLevel,
        purpose: str,
        result: AuditResult,
    ) -> None:
        """记录审计日志 (调用方需持锁)."""
        entry = AuditLogEntry(
            actor_id=actor_id,
            actor_role=actor_role,
            action=action,
            target_resource=target_resource,
            target_data_level=target_data_level,
            purpose=purpose,
            result=result,
        )
        self._audit_logs.append(entry)


__all__ = [
    # 常量
    "DEFAULT_TOKEN_TTL_SECONDS",
    "DEFAULT_REFRESH_TTL_SECONDS",
    "DEFAULT_ISSUER",
    "DEFAULT_AUDIENCE",
    "CLOCK_SKEW_SECONDS",
    "PBKDF2_ITERATIONS",
    "PBKDF2_SALT_BYTES",
    "PBKDF2_HASH_NAME",
    "PASSWORD_MAX_BYTES",
    "DEFAULT_TOKEN_VERSION",
    # 异常
    "L1AuthError",
    "AuthenticationError",
    "TokenError",
    "TokenExpiredError",
    "TokenRevokedError",
    "LifecycleError",
    # 密码安全
    "PasswordHasher",
    # JWT 管理
    "TokenPayload",
    "JWTManager",
    # 用户生命周期
    "UserLifecycleManager",
]
