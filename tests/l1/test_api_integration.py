"""T7 API 层与层间集成测试 (TDD).

测试覆盖:
1. 异常体系 (L1APIError 层级, JSON-RPC -32700 范围)
2. 统一响应格式 (_ok / _err)
3. API Key 管理 (生成/验证/撤销/限流)
4. 游标分页 (CursorPagination)
5. 幂等性管理器 (IdempotencyManager)
6. 令牌桶限流器 (TokenBucketRateLimiter)
7. Webhook 管理 (注册/签名/投递)
8. SSE 事件流 (EventStreamManager)
9. 中间件 (AuthMiddleware / ABACMiddleware / AuditMiddleware)
10. L1 REST API 路由器 (17 个核心端点)
11. 层间接口 (LayerInterfaces — L0/L2/L3/CC2)
12. 集成测试 (端到端流程)

设计依据:
- L1 设计文档第七章 7.1-7.5: API 设计、中间件、层间接口
- L1 设计文档第八章 8.1-8.4: 与 L0/L2/L3/CC2 的接口定义
- 任务拆分文档 T7: 交付物定义

融合世界先进方案:
- Stripe API: Idempotency-Key 幂等机制
- GitHub API: 游标分页 (cursor-based pagination)
- AWS API Gateway: 令牌桶限流
- Shopify Webhook: HMAC-SHA256 签名验证
- OpenAI SSE: Server-Sent Events 流式输出
"""

from __future__ import annotations

import asyncio
import json
import time
import hashlib
import hmac
import threading
from typing import Any

import pytest

from dy3_polaris.l1.models import (
    User,
    UserRole,
    UserStatus,
    ABACAttributes,
    GradeLevel,
    LabAccessTier,
    AuditAction,
    AuditResult,
    DataLevel,
    AuditLogEntry,
    Permission,
)
from dy3_polaris.l1.auth import JWTManager, PasswordHasher, TokenPayload
from dy3_polaris.l1.access_control import (
    RBACMatrix,
    ABACEvaluator,
    AccessControlManager,
    AccessRequest,
    AccessResult,
    AccessDecision,
    ActionType,
    ResourceType,
)


# ============================================================
# 辅助函数
# ============================================================

def make_user(
    user_id: str = "u-001",
    role: UserRole = UserRole.UNDERGRAD,
    status: UserStatus = UserStatus.ACTIVE,
) -> User:
    """创建测试用户."""
    return User(
        user_id=user_id,
        student_id="CS20240001",
        role=role,
        status=status,
        institution_id="inst-001",
        abac_attributes=ABACAttributes(
            grade_level=GradeLevel.FRESHMAN,
            lab_access_tier=LabAccessTier.TIER0,
        ),
    )


# ============================================================
# 1. 异常体系测试
# ============================================================

class TestExceptionHierarchy:
    """异常体系测试 (JSON-RPC -32700 范围)."""

    def test_l1_api_error_base(self):
        """L1APIError 基类存在."""
        from dy3_polaris.l1.api_integration import L1APIError

        err = L1APIError(detail="测试错误")
        assert err.code == "L1_API_ERROR"
        assert err.detail == "测试错误"

    def test_l1_api_error_jsonrpc_code(self):
        """L1APIError JSON-RPC 码为 -32700."""
        from dy3_polaris.l1.api_integration import L1APIError

        err = L1APIError()
        assert err._jsonrpc_code() == -32700

    def test_api_key_error(self):
        """APIKeyError 异常."""
        from dy3_polaris.l1.api_integration import APIKeyError

        err = APIKeyError(detail="无效的 API Key")
        assert err._jsonrpc_code() == -32701
        assert err.detail == "无效的 API Key"

    def test_rate_limit_error(self):
        """APIRateLimitError 异常."""
        from dy3_polaris.l1.api_integration import APIRateLimitError

        err = APIRateLimitError(detail="请求过于频繁", retry_after=30)
        assert err._jsonrpc_code() == -32702
        assert err.retry_after == 30

    def test_idempotency_error(self):
        """IdempotencyError 异常."""
        from dy3_polaris.l1.api_integration import IdempotencyError

        err = IdempotencyError(detail="幂等键冲突")
        assert err._jsonrpc_code() == -32703

    def test_webhook_error(self):
        """WebhookError 异常."""
        from dy3_polaris.l1.api_integration import WebhookError

        err = WebhookError(detail="签名验证失败")
        assert err._jsonrpc_code() == -32704

    def test_layer_interface_error(self):
        """LayerInterfaceError 异常."""
        from dy3_polaris.l1.api_integration import LayerInterfaceError

        err = LayerInterfaceError(detail="L2 层不可达")
        assert err._jsonrpc_code() == -32705

    def test_inheritance_from_l6_error(self):
        """所有异常继承 L6Error."""
        from dy3_polaris.l1.api_integration import L1APIError
        from dy3_polaris.l6.core.exceptions import L6Error

        assert issubclass(L1APIError, L6Error)


# ============================================================
# 2. 统一响应格式测试
# ============================================================

class TestUnifiedResponse:
    """统一响应格式测试."""

    def test_ok_response(self):
        """成功响应格式."""
        from dy3_polaris.l1.api_integration import ok_response

        resp = ok_response(data={"id": 1}, message="成功")
        assert resp["code"] == 0
        assert resp["data"] == {"id": 1}
        assert resp["message"] == "成功"

    def test_ok_response_default(self):
        """默认成功响应."""
        from dy3_polaris.l1.api_integration import ok_response

        resp = ok_response()
        assert resp["code"] == 0
        assert resp["data"] is None
        assert resp["message"] == ""

    def test_error_response(self):
        """错误响应格式."""
        from dy3_polaris.l1.api_integration import error_response

        resp = error_response(code=-32701, message="API_KEY_ERROR", detail="无效的 API Key")
        assert resp["code"] == -32701
        assert resp["message"] == "API_KEY_ERROR"
        assert resp["detail"] == "无效的 API Key"

    def test_paginated_response(self):
        """分页响应格式."""
        from dy3_polaris.l1.api_integration import paginated_response

        resp = paginated_response(
            data=[{"id": 1}, {"id": 2}],
            total=100,
            cursor="next-cursor-id",
            has_more=True,
        )
        assert resp["code"] == 0
        assert len(resp["data"]) == 2
        assert resp["pagination"]["total"] == 100
        assert resp["pagination"]["next_cursor"] == "next-cursor-id"
        assert resp["pagination"]["has_more"] is True


# ============================================================
# 3. API Key 管理测试
# ============================================================

class TestAPIKeyManager:
    """API Key 管理器测试 (设计文档 7.1 + 世界先进方案)."""

    def test_generate_api_key(self):
        """生成 API Key."""
        from dy3_polaris.l1.api_integration import APIKeyManager

        mgr = APIKeyManager()
        key, key_id = mgr.generate_key(
            owner_id="u-001",
            scopes=["read:reports", "read:sessions"],
        )
        assert key.startswith("dy3_sk_")
        assert len(key) > 40
        assert key_id.startswith("key_")

    def test_validate_api_key(self):
        """验证 API Key."""
        from dy3_polaris.l1.api_integration import APIKeyManager

        mgr = APIKeyManager()
        key, key_id = mgr.generate_key(owner_id="u-001", scopes=["read:reports"])
        payload = mgr.validate_key(key)
        assert payload is not None
        assert payload.owner_id == "u-001"
        assert "read:reports" in payload.scopes

    def test_validate_invalid_key(self):
        """无效 API Key 返回 None."""
        from dy3_polaris.l1.api_integration import APIKeyManager

        mgr = APIKeyManager()
        result = mgr.validate_key("dy3_sk_invalid_key_12345")
        assert result is None

    def test_revoke_api_key(self):
        """撤销 API Key."""
        from dy3_polaris.l1.api_integration import APIKeyManager

        mgr = APIKeyManager()
        key, key_id = mgr.generate_key(owner_id="u-001", scopes=["read:reports"])
        mgr.revoke_key(key_id)
        # 撤销后验证失败
        result = mgr.validate_key(key)
        assert result is None

    def test_api_key_scope_check(self):
        """API Key scope 检查."""
        from dy3_polaris.l1.api_integration import APIKeyManager

        mgr = APIKeyManager()
        key, _ = mgr.generate_key(owner_id="u-001", scopes=["read:reports"])
        payload = mgr.validate_key(key)
        assert payload is not None
        assert payload.has_scope("read:reports") is True
        assert payload.has_scope("write:reports") is False

    def test_api_key_expiration(self):
        """API Key 过期."""
        from dy3_polaris.l1.api_integration import APIKeyManager, APIKeyPayload

        mgr = APIKeyManager()
        key, key_id = mgr.generate_key(
            owner_id="u-001",
            scopes=["read:reports"],
            ttl_seconds=-1,  # 已过期
        )
        result = mgr.validate_key(key)
        assert result is None


# ============================================================
# 4. 游标分页测试
# ============================================================

class TestCursorPagination:
    """游标分页测试 (GitHub API 模式)."""

    def test_paginate_first_page(self):
        """第一页分页."""
        from dy3_polaris.l1.api_integration import CursorPaginator

        items = [{"id": f"item-{i}"} for i in range(50)]
        paginator = CursorPaginator(page_size=10)
        page = paginator.paginate(items, cursor=None)

        assert len(page.items) == 10
        assert page.has_more is True
        assert page.next_cursor is not None

    def test_paginate_last_page(self):
        """最后一页分页."""
        from dy3_polaris.l1.api_integration import CursorPaginator

        items = [{"id": f"item-{i}"} for i in range(8)]
        paginator = CursorPaginator(page_size=10)
        page = paginator.paginate(items, cursor=None)

        assert len(page.items) == 8
        assert page.has_more is False
        assert page.next_cursor is None

    def test_paginate_with_cursor(self):
        """使用游标获取下一页."""
        from dy3_polaris.l1.api_integration import CursorPaginator

        items = [{"id": f"item-{i}"} for i in range(25)]
        paginator = CursorPaginator(page_size=10)

        # 第一页
        page1 = paginator.paginate(items, cursor=None)
        assert len(page1.items) == 10
        assert page1.has_more is True

        # 第二页
        page2 = paginator.paginate(items, cursor=page1.next_cursor)
        assert len(page2.items) == 10
        assert page2.has_more is True

        # 第三页
        page3 = paginator.paginate(items, cursor=page2.next_cursor)
        assert len(page3.items) == 5
        assert page3.has_more is False

    def test_paginate_empty(self):
        """空列表分页."""
        from dy3_polaris.l1.api_integration import CursorPaginator

        paginator = CursorPaginator(page_size=10)
        page = paginator.paginate([], cursor=None)
        assert len(page.items) == 0
        assert page.has_more is False
        assert page.next_cursor is None


# ============================================================
# 5. 幂等性管理器测试
# ============================================================

class TestIdempotencyManager:
    """幂等性管理器测试 (Stripe Idempotency-Key 模式)."""

    def test_first_request_processed(self):
        """首次请求正常处理."""
        from dy3_polaris.l1.api_integration import IdempotencyManager

        mgr = IdempotencyManager()

        @mgr.wrap
        def create_order(order_id: str, amount: int) -> dict:
            return {"order_id": order_id, "amount": amount}

        result = create_order("ord-001", 100, _idempotency_key="key-001")
        assert result["order_id"] == "ord-001"
        assert result["amount"] == 100

    def test_duplicate_returns_cached(self):
        """重复请求返回缓存结果."""
        from dy3_polaris.l1.api_integration import IdempotencyManager

        mgr = IdempotencyManager()
        call_count = 0

        @mgr.wrap
        def create_order(order_id: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"order_id": order_id}

        # 第一次调用
        result1 = create_order("ord-001", _idempotency_key="key-001")
        assert call_count == 1

        # 第二次相同 key
        result2 = create_order("ord-002", _idempotency_key="key-001")
        assert call_count == 1  # 函数未再次执行
        assert result2 == result1  # 返回缓存结果

    def test_different_keys_processed_separately(self):
        """不同幂等键独立处理."""
        from dy3_polaris.l1.api_integration import IdempotencyManager

        mgr = IdempotencyManager()
        call_count = 0

        @mgr.wrap
        def create_order(order_id: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"order_id": order_id}

        create_order("ord-001", _idempotency_key="key-001")
        create_order("ord-002", _idempotency_key="key-002")
        assert call_count == 2

    def test_idempotency_expiry(self):
        """幂等记录过期后可重新处理."""
        from dy3_polaris.l1.api_integration import IdempotencyManager

        mgr = IdempotencyManager(ttl_seconds=0)
        call_count = 0

        @mgr.wrap
        def create_order(order_id: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"order_id": order_id}

        create_order("ord-001", _idempotency_key="key-001")
        time.sleep(0.1)
        create_order("ord-002", _idempotency_key="key-001")
        assert call_count == 2  # 过期后重新执行


# ============================================================
# 6. 令牌桶限流器测试
# ============================================================

class TestTokenBucketRateLimiter:
    """令牌桶限流器测试 (AWS API Gateway 模式)."""

    def test_allow_under_limit(self):
        """限流内允许."""
        from dy3_polaris.l1.api_integration import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=5)
        for _ in range(5):
            allowed, retry_after = limiter.acquire("client-001")
            assert allowed is True

    def test_deny_over_limit(self):
        """超限拒绝."""
        from dy3_polaris.l1.api_integration import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(capacity=3, refill_rate=1)
        for _ in range(3):
            limiter.acquire("client-002")
        allowed, retry_after = limiter.acquire("client-002")
        assert allowed is False
        assert retry_after > 0

    def test_refill_after_wait(self):
        """等待后令牌补充."""
        from dy3_polaris.l1.api_integration import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(capacity=2, refill_rate=100)
        limiter.acquire("client-003")
        limiter.acquire("client-003")
        # 消耗完
        allowed, _ = limiter.acquire("client-003")
        assert allowed is False
        # 等待补充
        time.sleep(0.05)
        allowed, _ = limiter.acquire("client-003")
        assert allowed is True

    def test_independent_clients(self):
        """不同客户端独立限流."""
        from dy3_polaris.l1.api_integration import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(capacity=2, refill_rate=1)
        # client-A 消耗完
        limiter.acquire("client-A")
        limiter.acquire("client-A")
        allowed_a, _ = limiter.acquire("client-A")
        assert allowed_a is False
        # client-B 仍有额度
        allowed_b, _ = limiter.acquire("client-B")
        assert allowed_b is True


# ============================================================
# 7. Webhook 管理测试
# ============================================================

class TestWebhookManager:
    """Webhook 管理测试 (Shopify HMAC-SHA256 模式)."""

    def test_register_webhook(self):
        """注册 Webhook."""
        from dy3_polaris.l1.api_integration import WebhookManager

        mgr = WebhookManager()
        endpoint = mgr.register(
            owner_id="u-001",
            url="https://example.com/webhook",
            events=["session.created", "context.refreshed"],
            secret="webhook-secret-123",
        )
        assert endpoint.endpoint_id.startswith("wh_")
        assert endpoint.url == "https://example.com/webhook"
        assert "session.created" in endpoint.events

    def test_sign_payload(self):
        """HMAC-SHA256 签名."""
        from dy3_polaris.l1.api_integration import WebhookManager

        mgr = WebhookManager()
        endpoint = mgr.register(
            owner_id="u-001",
            url="https://example.com/webhook",
            events=["session.created"],
            secret="my-secret",
        )
        payload = b'{"event": "test"}'
        signature = mgr.sign_payload(endpoint.endpoint_id, payload)
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA-256 hex

    def test_verify_signature(self):
        """验证签名."""
        from dy3_polaris.l1.api_integration import WebhookManager

        mgr = WebhookManager()
        endpoint = mgr.register(
            owner_id="u-001",
            url="https://example.com/webhook",
            events=["session.created"],
            secret="my-secret",
        )
        payload = b'{"event": "test"}'
        signature = mgr.sign_payload(endpoint.endpoint_id, payload)
        assert mgr.verify_signature(endpoint.endpoint_id, payload, signature) is True

    def test_verify_invalid_signature(self):
        """无效签名验证失败."""
        from dy3_polaris.l1.api_integration import WebhookManager

        mgr = WebhookManager()
        endpoint = mgr.register(
            owner_id="u-001",
            url="https://example.com/webhook",
            events=["session.created"],
            secret="my-secret",
        )
        assert mgr.verify_signature(endpoint.endpoint_id, b'{"event": "test"}', "invalid-signature") is False

    def test_get_subscribers_for_event(self):
        """获取事件订阅者."""
        from dy3_polaris.l1.api_integration import WebhookManager

        mgr = WebhookManager()
        mgr.register(
            owner_id="u-001",
            url="https://a.com/webhook",
            events=["session.created"],
            secret="s1",
        )
        mgr.register(
            owner_id="u-002",
            url="https://b.com/webhook",
            events=["session.created", "context.refreshed"],
            secret="s2",
        )
        subscribers = mgr.get_subscribers("session.created")
        assert len(subscribers) == 2

    def test_unregister_webhook(self):
        """注销 Webhook."""
        from dy3_polaris.l1.api_integration import WebhookManager

        mgr = WebhookManager()
        endpoint = mgr.register(
            owner_id="u-001",
            url="https://example.com/webhook",
            events=["session.created"],
            secret="s1",
        )
        mgr.unregister(endpoint.endpoint_id)
        subscribers = mgr.get_subscribers("session.created")
        assert len(subscribers) == 0


# ============================================================
# 8. SSE 事件流管理测试
# ============================================================

class TestEventStreamManager:
    """SSE 事件流管理测试 (OpenAI SSE 模式)."""

    def test_register_stream(self):
        """注册事件流."""
        from dy3_polaris.l1.api_integration import EventStreamManager

        mgr = EventStreamManager()
        stream_id = mgr.register_stream(user_id="u-001", session_id="sess-001")
        assert stream_id.startswith("sse_")
        assert mgr.is_active(stream_id) is True

    def test_push_event(self):
        """推送事件到流."""
        from dy3_polaris.l1.api_integration import EventStreamManager

        mgr = EventStreamManager()
        stream_id = mgr.register_stream(user_id="u-001", session_id="sess-001")
        mgr.push_event(stream_id, event_type="message", data='{"text": "hello"}')
        events = mgr.get_events(stream_id)
        assert len(events) == 1
        assert events[0].event_type == "message"
        assert events[0].data == '{"text": "hello"}'

    def test_close_stream(self):
        """关闭事件流."""
        from dy3_polaris.l1.api_integration import EventStreamManager

        mgr = EventStreamManager()
        stream_id = mgr.register_stream(user_id="u-001", session_id="sess-001")
        mgr.close_stream(stream_id)
        assert mgr.is_active(stream_id) is False

    def test_format_sse(self):
        """SSE 格式化."""
        from dy3_polaris.l1.api_integration import EventStreamManager, SSEEvent

        event = SSEEvent(event_type="message", data='{"text": "hello"}', event_id="evt-001")
        formatted = EventStreamManager.format_sse(event)
        assert "event: message" in formatted
        assert "data: {\"text\": \"hello\"}" in formatted
        assert "id: evt-001" in formatted
        assert formatted.endswith("\n\n")

    def test_multiple_streams_per_user(self):
        """单用户多流."""
        from dy3_polaris.l1.api_integration import EventStreamManager

        mgr = EventStreamManager()
        s1 = mgr.register_stream(user_id="u-001", session_id="sess-001")
        s2 = mgr.register_stream(user_id="u-001", session_id="sess-002")
        assert s1 != s2
        assert mgr.is_active(s1) is True
        assert mgr.is_active(s2) is True


# ============================================================
# 9. 中间件测试
# ============================================================

class TestAuthMiddleware:
    """JWT 认证中间件测试."""

    def test_extract_bearer_token(self):
        """从 Authorization 头提取 Bearer Token."""
        from dy3_polaris.l1.api_integration import AuthMiddleware

        mw = AuthMiddleware(JWTManager())
        token = mw.extract_bearer_token("Bearer abc.def.ghi")
        assert token == "abc.def.ghi"

    def test_extract_bearer_token_invalid(self):
        """无效 Authorization 头."""
        from dy3_polaris.l1.api_integration import AuthMiddleware

        mw = AuthMiddleware(JWTManager())
        assert mw.extract_bearer_token("Basic abc") is None
        assert mw.extract_bearer_token("") is None
        assert mw.extract_bearer_token(None) is None

    def test_authenticate_valid_token(self):
        """有效 Token 认证."""
        from dy3_polaris.l1.api_integration import AuthMiddleware

        jwt_mgr = JWTManager()
        user = make_user()
        access_token, _ = jwt_mgr.issue_token(user)
        mw = AuthMiddleware(jwt_mgr)
        payload = mw.authenticate(access_token)
        assert payload is not None
        assert payload.user_id == "u-001"

    def test_authenticate_invalid_token(self):
        """无效 Token 认证失败."""
        from dy3_polaris.l1.api_integration import AuthMiddleware

        mw = AuthMiddleware(JWTManager())
        payload = mw.authenticate("invalid.token.here")
        assert payload is None


class TestABACMiddleware:
    """ABAC 权限校验中间件测试."""

    def test_check_permission_allowed(self):
        """权限检查通过."""
        from dy3_polaris.l1.api_integration import ABACMiddleware

        rbac = RBACMatrix()
        abac = ABACEvaluator()
        acm = AccessControlManager(rbac, abac)
        mw = ABACMiddleware(acm)

        user = make_user(role=UserRole.UNDERGRAD)
        allowed = mw.check_permission(user, Permission.KB_PUBLIC_READ)
        assert allowed is True

    def test_check_permission_denied(self):
        """权限检查拒绝."""
        from dy3_polaris.l1.api_integration import ABACMiddleware

        rbac = RBACMatrix()
        abac = ABACEvaluator()
        acm = AccessControlManager(rbac, abac)
        mw = ABACMiddleware(acm)

        user = make_user(role=UserRole.UNDERGRAD)
        allowed = mw.check_permission(user, Permission.KB_WRITE_EDIT)
        assert allowed is False

    def test_require_permission_decorator(self):
        """权限装饰器."""
        from dy3_polaris.l1.api_integration import ABACMiddleware

        rbac = RBACMatrix()
        abac = ABACEvaluator()
        acm = AccessControlManager(rbac, abac)
        mw = ABACMiddleware(acm)

        user = make_user(role=UserRole.TEACHER)
        assert mw.require_permission(user, Permission.KB_WRITE_EDIT) is True

    def test_public_paths_bypass(self):
        """公开路径跳过认证."""
        from dy3_polaris.l1.api_integration import ABACMiddleware

        mw = ABACMiddleware(AccessControlManager(RBACMatrix(), ABACEvaluator()))
        assert mw.is_public_path("/api/v1/auth/login") is True
        assert mw.is_public_path("/api/v1/auth/refresh") is True
        assert mw.is_public_path("/health") is True
        assert mw.is_public_path("/api/v1/sessions") is False


class TestAuditMiddleware:
    """审计日志中间件测试."""

    def test_create_audit_entry(self):
        """创建审计条目."""
        from dy3_polaris.l1.api_integration import AuditMiddleware

        mw = AuditMiddleware()
        user = make_user()
        entry = mw.create_entry(
            user=user,
            action=AuditAction.EXPORT,
            resource="/api/v1/export/learner-data",
            data_level=DataLevel.L3_SENSITIVE,
            result=AuditResult.SUCCESS,
        )
        assert entry.actor_id == "u-001"
        assert entry.action == AuditAction.EXPORT
        assert entry.result == AuditResult.SUCCESS

    def test_audit_log_recording(self):
        """审计日志记录."""
        from dy3_polaris.l1.api_integration import AuditMiddleware

        mw = AuditMiddleware()
        user = make_user()
        mw.record(
            user=user,
            action=AuditAction.VIEW,
            resource="/api/v1/sessions/sess-001",
            data_level=DataLevel.L3_SENSITIVE,
            result=AuditResult.SUCCESS,
        )
        assert len(mw.entries) == 1

    def test_audit_query_by_action(self):
        """按操作类型查询审计."""
        from dy3_polaris.l1.api_integration import AuditMiddleware

        mw = AuditMiddleware()
        user = make_user()
        mw.record(user=user, action=AuditAction.VIEW, resource="r1", data_level=DataLevel.L2_INTERNAL, result=AuditResult.SUCCESS)
        mw.record(user=user, action=AuditAction.EXPORT, resource="r2", data_level=DataLevel.L3_SENSITIVE, result=AuditResult.SUCCESS)
        mw.record(user=user, action=AuditAction.VIEW, resource="r3", data_level=DataLevel.L2_INTERNAL, result=AuditResult.DENIED)

        views = mw.query(action=AuditAction.VIEW)
        assert len(views) == 2


# ============================================================
# 10. L1 REST API 路由器测试
# ============================================================

class TestL1APIRouter:
    """L1 REST API 路由器测试 (17 个核心端点)."""

    def _create_router(self):
        """创建测试路由器."""
        from dy3_polaris.l1.api_integration import L1APIRouter

        jwt_mgr = JWTManager()
        rbac = RBACMatrix()
        abac = ABACEvaluator()
        acm = AccessControlManager(rbac, abac)
        router = L1APIRouter(
            jwt_manager=jwt_mgr,
            access_control=acm,
        )
        return router, jwt_mgr

    def test_create_app(self):
        """创建 Starlette 应用."""
        from dy3_polaris.l1.api_integration import L1APIRouter

        router, _ = self._create_router()
        app = router.create_app()
        assert app is not None

    def test_get_routes_summary(self):
        """获取路由摘要."""
        router, _ = self._create_router()
        summary = router.get_routes_summary()
        assert len(summary) >= 17
        paths = [r["path"] for r in summary]
        assert "/api/v1/auth/login" in paths
        assert "/api/v1/sessions" in paths
        assert "/api/v1/hitl/confirm" in paths
        assert "/api/v1/audit/logs" in paths

    def test_login_endpoint(self):
        """登录端点 (POST /api/v1/auth/login)."""
        from starlette.testclient import TestClient

        router, jwt_mgr = self._create_router()
        # 预注册用户
        user = make_user()
        pw_hash = PasswordHasher.hash_password("TestPass123!")
        router.register_user(user, pw_hash)

        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/api/v1/auth/login", json={
            "student_id": "CS20240001",
            "password": "TestPass123!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "access_token" in data["data"]
        assert "refresh_token" in data["data"]

    def test_login_invalid_credentials(self):
        """无效凭证登录失败."""
        from starlette.testclient import TestClient

        router, _ = self._create_router()
        app = router.create_app()
        client = TestClient(app)

        resp = client.post("/api/v1/auth/login", json={
            "student_id": "nonexistent",
            "password": "wrong",
        })
        assert resp.status_code == 200  # 统一响应格式
        data = resp.json()
        assert data["code"] != 0

    def test_health_endpoint(self):
        """健康检查端点."""
        from starlette.testclient import TestClient

        router, _ = self._create_router()
        app = router.create_app()
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "healthy"

    def test_protected_endpoint_without_token(self):
        """未认证访问受保护端点."""
        from starlette.testclient import TestClient

        router, _ = self._create_router()
        app = router.create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/users/me")
        assert resp.status_code == 401

    def test_protected_endpoint_with_token(self):
        """认证后访问受保护端点."""
        from starlette.testclient import TestClient

        router, jwt_mgr = self._create_router()
        user = make_user()
        access_token, _ = jwt_mgr.issue_token(user)
        router.register_user(user, PasswordHasher.hash_password("TestPass123!"))

        app = router.create_app()
        client = TestClient(app)

        resp = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["user_id"] == "u-001"

    def test_create_session_endpoint(self):
        """创建会话端点."""
        from starlette.testclient import TestClient

        router, jwt_mgr = self._create_router()
        user = make_user()
        access_token, _ = jwt_mgr.issue_token(user)
        router.register_user(user, PasswordHasher.hash_password("TestPass123!"))

        app = router.create_app()
        client = TestClient(app)

        resp = client.post(
            "/api/v1/sessions",
            json={"session_type": "diagnosis"},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "session_id" in data["data"]

    def test_audit_logs_endpoint(self):
        """审计日志查询端点."""
        from starlette.testclient import TestClient

        router, jwt_mgr = self._create_router()
        user = make_user(role=UserRole.ADMIN)
        access_token, _ = jwt_mgr.issue_token(user)
        router.register_user(user, PasswordHasher.hash_password("TestPass123!"))

        app = router.create_app()
        client = TestClient(app)

        resp = client.get(
            "/api/v1/audit/logs",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "pagination" in data


# ============================================================
# 11. 层间接口测试
# ============================================================

class TestLayerInterfaces:
    """层间接口实现测试 (L0/L2/L3/CC2)."""

    def test_layer_interfaces_init(self):
        """层间接口初始化."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        assert interfaces is not None

    def test_report_audit_logs_to_l0(self):
        """L1→L0 审计日志上报."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        entries = [
            AuditLogEntry(
                actor_id="u-001",
                actor_role=UserRole.UNDERGRAD,
                action=AuditAction.VIEW,
                target_resource="report/r-001",
                target_data_level=DataLevel.L3_SENSITIVE,
                purpose="查看学情",
                result=AuditResult.SUCCESS,
            ),
        ]
        result = interfaces.report_audit_logs(entries)
        assert result is True
        assert interfaces.l0_audit_count == 1

    def test_report_privacy_event_to_l0(self):
        """L1→L0 隐私事件通知."""
        from dy3_polaris.l1.api_integration import LayerInterfaces
        from dy3_polaris.l1.models import PrivacyEvent

        interfaces = LayerInterfaces()
        event = PrivacyEvent(
            event_type="data_export",
            user_id="u-001",
            data_level=DataLevel.L3_SENSITIVE,
            detail="导出学情数据",
        )
        result = interfaces.report_privacy_event(event)
        assert result is True

    def test_write_provenance_to_l0(self):
        """L1→L0 Provenance 写入."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.write_provenance(
            session_id="sess-001",
            agent_id="agent-diagnosis-001",
            action="diagnosis",
            output_hash="sha256:abc123",
        )
        assert result is True

    def test_send_context_envelope_to_l2(self):
        """L1→L2 上下文信封传递."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.send_context_envelope(
            session_id="sess-001",
            envelope_data={"cognitive_load": 0.3, "mastery": {"CS101": 0.7}},
        )
        assert result is True

    def test_receive_learner_profile_from_l2(self):
        """L2→L1 学情画像输出."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.receive_learner_profile(
            user_id="u-001",
            profile_data={"bkt_params": {"CS101": {"P_T": 0.2, "P_S": 0.1}}},
        )
        assert result is True
        assert "u-001" in interfaces.l2_profiles

    def test_receive_bkt_update_from_l2(self):
        """L2→L1 BKT 参数更新."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.receive_bkt_update(
            user_id="u-001",
            kp_id="CS101-01",
            update_data={"P_T": 0.15, "P_S": 0.08},
        )
        assert result is True

    def test_check_access_to_l3(self):
        """L1→L3 知识访问权限校验."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.check_access(
            user_id="u-001",
            resource_id="CS101-advanced-module",
            action="read",
        )
        assert isinstance(result, bool)

    def test_request_resources_from_l3(self):
        """L1→L3 学习资源推荐请求."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.request_resources(
            user_id="u-001",
            session_id="sess-001",
            weak_kcs=["CS101-01", "CS101-02"],
        )
        assert result is True

    def test_receive_knowledge_result_from_l3(self):
        """L3→L1 知识查询结果."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.receive_knowledge_result(
            session_id="sess-001",
            resources=[{"id": "res-001", "title": "基本概念"}],
            confidence_scores=[0.9],
        )
        assert result is True

    def test_route_approval_request_to_cc2(self):
        """L1↔CC2 确认请求路由."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.route_approval_request(
            user_id="u-001",
            session_id="sess-001",
            content="需要教师审核的内容",
            confidence=0.6,
        )
        assert result is True

    def test_route_approval_response_from_cc2(self):
        """L1↔CC2 确认响应路由."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.route_approval_response(
            request_id="hitl-001",
            decision="approve",
            comment="通过",
        )
        assert result is True

    def test_report_feedback_to_cc2(self):
        """L1↔CC2 反馈数据上报."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.report_feedback(
            session_id="sess-001",
            user_id="u-001",
            feedback_type="factual",
            content="内容有误",
        )
        assert result is True

    def test_alert_emergency_to_cc2(self):
        """L1↔CC2 紧急干预通知."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.alert_emergency(
            session_id="sess-001",
            user_id="u-001",
            trigger_reason="认知负荷过高",
            trigger_value=0.96,
        )
        assert result is True

    def test_pull_compliance_policies_from_l0(self):
        """L0→L1 合规策略拉取."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        policies = interfaces.pull_compliance_policies()
        assert isinstance(policies, list)

    def test_receive_policy_update_from_l0(self):
        """L0→L1 策略变更通知."""
        from dy3_polaris.l1.api_integration import LayerInterfaces

        interfaces = LayerInterfaces()
        result = interfaces.receive_policy_update(
            policy_id="policy-001",
            version="2.0",
            diff={"retention_days": 365},
        )
        assert result is True


# ============================================================
# 12. 集成测试
# ============================================================

class TestIntegration:
    """端到端集成测试."""

    def test_full_login_and_access_flow(self):
        """完整登录→认证→权限→访问流程."""
        from starlette.testclient import TestClient
        from dy3_polaris.l1.api_integration import L1APIRouter

        jwt_mgr = JWTManager()
        rbac = RBACMatrix()
        abac = ABACEvaluator()
        acm = AccessControlManager(rbac, abac)
        router = L1APIRouter(jwt_manager=jwt_mgr, access_control=acm)

        # 注册用户
        user = make_user(role=UserRole.UNDERGRAD)
        pw_hash = PasswordHasher.hash_password("MyPassword123!")
        router.register_user(user, pw_hash)

        app = router.create_app()
        client = TestClient(app)

        # 1. 登录
        resp = client.post("/api/v1/auth/login", json={
            "student_id": "CS20240001",
            "password": "MyPassword123!",
        })
        assert resp.json()["code"] == 0
        token = resp.json()["data"]["access_token"]

        # 2. 访问 /users/me
        resp = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.json()["data"]["user_id"] == "u-001"

        # 3. 创建会话
        resp = client.post(
            "/api/v1/sessions",
            json={"session_type": "diagnosis"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.json()["code"] == 0
        session_id = resp.json()["data"]["session_id"]

        # 4. 获取会话详情
        resp = client.get(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.json()["code"] == 0

    def test_token_revocation_flow(self):
        """Token 撤销流程."""
        from starlette.testclient import TestClient
        from dy3_polaris.l1.api_integration import L1APIRouter

        jwt_mgr = JWTManager()
        router = L1APIRouter(
            jwt_manager=jwt_mgr,
            access_control=AccessControlManager(RBACMatrix(), ABACEvaluator()),
        )
        user = make_user()
        access_token, _ = jwt_mgr.issue_token(user)
        router.register_user(user, PasswordHasher.hash_password("TestPass123!"))

        app = router.create_app()
        client = TestClient(app)

        # 撤销前可以访问
        resp = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {access_token}"})
        assert resp.status_code == 200

        # 撤销 Token
        jwt_mgr.revoke_token(access_token)

        # 撤销后无法访问
        resp = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {access_token}"})
        assert resp.status_code == 401

    def test_rate_limiting_on_api(self):
        """API 限流测试."""
        from dy3_polaris.l1.api_integration import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(capacity=3, refill_rate=1)
        results = []
        for _ in range(5):
            allowed, _ = limiter.acquire("api-client-001")
            results.append(allowed)
        assert results == [True, True, True, False, False]

    def test_api_key_authentication(self):
        """API Key 认证流程."""
        from dy3_polaris.l1.api_integration import APIKeyManager

        mgr = APIKeyManager()
        key, key_id = mgr.generate_key(
            owner_id="u-001",
            scopes=["read:reports", "write:reports"],
        )

        # 验证有效 Key
        payload = mgr.validate_key(key)
        assert payload is not None
        assert payload.has_scope("write:reports") is True

        # 撤销后验证失败
        mgr.revoke_key(key_id)
        assert mgr.validate_key(key) is None

    def test_idempotent_session_creation(self):
        """幂等会话创建."""
        from dy3_polaris.l1.api_integration import IdempotencyManager

        mgr = IdempotencyManager()
        created_ids = []

        @mgr.wrap
        def create_session(user_id: str) -> str:
            sid = f"sess-{len(created_ids)}"
            created_ids.append(sid)
            return sid

        # 相同幂等键
        r1 = create_session("u-001", _idempotency_key="idem-001")
        r2 = create_session("u-001", _idempotency_key="idem-001")
        assert r1 == r2
        assert len(created_ids) == 1  # 只创建了一次
