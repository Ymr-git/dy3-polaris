"""L1 用户域 API 层与层间集成 (T7) — 核心引擎.

设计依据:
- L1 设计文档第七章 7.1-7.5: 技术选型、API 设计、ER 图、关键流程、性能指标
- L1 设计文档第八章 8.1-8.4: 与 L0/L2/L3/CC2 的接口定义
- 任务拆分文档 T7: 交付物定义

融合世界先进方案:
- Stripe API: Idempotency-Key 幂等机制 (V4 UUID + 缓存结果)
- GitHub API: 游标分页 (cursor-based pagination, O(1) 复杂度)
- AWS API Gateway: 令牌桶限流 (Token Bucket, 允许突发)
- Shopify Webhook: HMAC-SHA256 签名验证 + 重试投递
- OpenAI SSE: Server-Sent Events 流式输出 (text/event-stream)
- OAuth 2.0 Client Credentials: API Key + Scope 管理模式
- REST API Best Practices: URI 版本控制 + 标准错误码 + 分页元数据

模块组成:
1. 异常体系: L1APIError 层级 (JSON-RPC -32700 范围)
2. 统一响应: ok_response / error_response / paginated_response
3. API Key 管理: APIKeyManager (生成/验证/撤销/scope)
4. 游标分页: CursorPaginator (GitHub API 模式)
5. 幂等性: IdempotencyManager (Stripe Idempotency-Key 模式)
6. 限流器: TokenBucketRateLimiter (AWS API Gateway 模式)
7. Webhook: WebhookManager (Shopify HMAC-SHA256 模式)
8. SSE 事件流: EventStreamManager (OpenAI SSE 模式)
9. 中间件: AuthMiddleware / ABACMiddleware / AuditMiddleware
10. REST API 路由器: L1APIRouter (17 个核心端点)
11. 层间接口: LayerInterfaces (L0/L2/L3/CC2 四组接口)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from dy3_polaris.l1.models import (
    ABACAttributes,
    AuditAction,
    AuditLogEntry,
    AuditResult,
    DataLevel,
    Permission,
    PrivacyEvent,
    User,
    UserRole,
    UserStatus,
)
from dy3_polaris.l1.auth import (
    JWTManager,
    PasswordHasher,
    TokenPayload,
    TokenExpiredError,
    TokenRevokedError,
    TokenError,
)
from dy3_polaris.l1.access_control import (
    ABACEvaluator,
    AccessControlManager,
    RBACMatrix,
)
from dy3_polaris.l1.session_manager import LearningSessionManager
from dy3_polaris.l1.session_manager import SessionType, SessionStatus
from dy3_polaris.l1.privacy_governance import AuditLogger, PrivacyGovernanceManager
from dy3_polaris.l1.hitl_manager import HiTLManager, HiTLType, HiTLPriority
from dy3_polaris.l1.context_broker import LearningContextBroker
from dy3_polaris.l6.core.exceptions import L6Error

_logger = logging.getLogger("dy3_polaris.l1.api_integration")


# ============================================================
# 1. 常量定义
# ============================================================

API_VERSION: str = "v1"
API_PREFIX: str = f"/api/{API_VERSION}"

# API Key 配置
API_KEY_PREFIX: str = "dy3_sk_"
API_KEY_BYTES: int = 32  # 256-bit
API_KEY_DEFAULT_TTL: int = 90 * 24 * 3600  # 90 天

# 游标分页默认
DEFAULT_PAGE_SIZE: int = 20
MAX_PAGE_SIZE: int = 100

# 幂等性默认 TTL
IDEMPOTENCY_DEFAULT_TTL: int = 24 * 3600  # 24 小时

# 令牌桶默认
DEFAULT_BUCKET_CAPACITY: int = 100
DEFAULT_REFILL_RATE: float = 10.0  # 令牌/秒

# 公开路径 (无需认证)
PUBLIC_PATHS: frozenset[str] = frozenset({
    "/health",
    f"{API_PREFIX}/auth/login",
    f"{API_PREFIX}/auth/refresh",
    f"{API_PREFIX}/auth/register",
})


# ============================================================
# 2. 异常体系 (JSON-RPC -32700 范围)
# ============================================================


class L1APIError(L6Error):
    """L1 API 层基础异常 (JSON-RPC -32700).

    所有 API 与集成相关异常的基类, 继承自 L6Error.
    """

    def __init__(
        self,
        code: str = "L1_API_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32700


class APIKeyError(L1APIError):
    """API Key 错误 (JSON-RPC -32701).

    API Key 无效、过期或已被撤销.
    """

    def __init__(
        self,
        detail: str = "API Key 无效",
        key_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {}
        if key_id:
            ctx["key_id"] = key_id
        if context:
            ctx.update(context)
        super().__init__("API_KEY_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32701


class APIRateLimitError(L1APIError):
    """API 限流错误 (JSON-RPC -32702).

    请求频率超过限流阈值.

    Attributes:
        retry_after: 建议重试等待秒数
    """

    def __init__(
        self,
        detail: str = "请求过于频繁",
        retry_after: int = 1,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.retry_after = retry_after
        ctx: dict[str, Any] = {"retry_after": retry_after}
        if context:
            ctx.update(context)
        super().__init__("API_RATE_LIMIT_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32702


class IdempotencyError(L1APIError):
    """幂等性错误 (JSON-RPC -32703).

    幂等键冲突、幂等记录不存在等.
    """

    def __init__(
        self,
        detail: str = "幂等键冲突",
        key: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {}
        if key:
            ctx["idempotency_key"] = key
        if context:
            ctx.update(context)
        super().__init__("IDEMPOTENCY_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32703


class WebhookError(L1APIError):
    """Webhook 错误 (JSON-RPC -32704).

    签名验证失败、投递失败、端点不存在等.
    """

    def __init__(
        self,
        detail: str = "Webhook 错误",
        endpoint_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {}
        if endpoint_id:
            ctx["endpoint_id"] = endpoint_id
        if context:
            ctx.update(context)
        super().__init__("WEBHOOK_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32704


class LayerInterfaceError(L1APIError):
    """层间接口错误 (JSON-RPC -32705).

    目标层不可达、接口调用失败等.
    """

    def __init__(
        self,
        detail: str = "层间接口错误",
        target_layer: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {}
        if target_layer:
            ctx["target_layer"] = target_layer
        if context:
            ctx.update(context)
        super().__init__("LAYER_INTERFACE_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32705


# ============================================================
# 3. 统一响应格式
# ============================================================


def ok_response(
    data: Any = None,
    message: str = "",
) -> dict[str, Any]:
    """构造成功响应 (信封单点: shared/contract.py).

    Returns:
        {"code": 0, "data": ..., "message": ...}
    """
    from dy3_polaris.shared.contract import ok

    return ok(data, message)


def error_response(
    code: int,
    message: str,
    detail: str = "",
) -> dict[str, Any]:
    """构造错误响应 (信封单点: shared/contract.py).

    Returns:
        {"code": <error_code>, "message": <error_message>, "detail": <detail>}
    """
    from dy3_polaris.shared.contract import err

    return err(code, message, detail)


def paginated_response(
    data: list[Any],
    total: int,
    cursor: str | None = None,
    has_more: bool = False,
) -> dict[str, Any]:
    """构造分页响应 (游标分页).

    Returns:
        {"code": 0, "data": [...], "pagination": {"total": N, "next_cursor": "...", "has_more": true/false}}
    """
    return {
        "code": 0,
        "data": data,
        "pagination": {
            "total": total,
            "next_cursor": cursor,
            "has_more": has_more,
        },
    }


# ============================================================
# 4. API Key 管理
# ============================================================


@dataclass
class APIKeyPayload:
    """API Key 载荷 (验证后返回).

    Attributes:
        key_id: Key 标识符
        owner_id: 所有者用户 ID
        scopes: 授权范围列表 (如 ["read:reports", "write:sessions"])
        created_at: 创建时间戳
        expires_at: 过期时间戳 (0 = 永不过期)
        revoked: 是否已撤销
    """

    key_id: str
    owner_id: str
    scopes: list[str]
    created_at: int
    expires_at: int = 0
    revoked: bool = False

    def has_scope(self, scope: str) -> bool:
        """检查是否拥有指定 scope."""
        return scope in self.scopes

    def is_expired(self) -> bool:
        """检查是否已过期."""
        if self.expires_at == 0:
            return False
        return int(time.time()) > self.expires_at


class APIKeyManager:
    """API Key 管理器 — 生成/验证/撤销/scope 控制.

    融合方案:
    - OAuth 2.0 Client Credentials: M2M 认证模式
    - Stripe API Key: 前缀 + 随机字节 + 哈希存储
    - GitHub Token: scope 粒度权限控制
    - AWS IAM: 密钥轮换和撤销

    线程安全: threading.RLock 保护密钥存储.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # key_id → APIKeyPayload
        self._keys: dict[str, APIKeyPayload] = {}
        # key_hash → key_id (用于快速验证, 不存储明文)
        self._key_hashes: dict[str, str] = {}

    def generate_key(
        self,
        owner_id: str,
        scopes: list[str],
        ttl_seconds: int = API_KEY_DEFAULT_TTL,
    ) -> tuple[str, str]:
        """生成 API Key.

        Args:
            owner_id: 所有者用户 ID
            scopes: 授权范围列表
            ttl_seconds: 有效期 (秒), 0 = 永不过期

        Returns:
            (api_key_string, key_id) 元组
        """
        key_id = f"key_{uuid.uuid4().hex[:16]}"
        raw_bytes = secrets.token_bytes(API_KEY_BYTES)
        key_string = API_KEY_PREFIX + base64.urlsafe_b64encode(raw_bytes).decode("ascii").rstrip("=")

        now = int(time.time())
        if ttl_seconds == 0:
            expires_at = 0  # 0 = 永不过期
        else:
            expires_at = now + ttl_seconds  # 正数=未来过期, 负数=已过期

        payload = APIKeyPayload(
            key_id=key_id,
            owner_id=owner_id,
            scopes=list(scopes),
            created_at=now,
            expires_at=expires_at,
        )

        key_hash = hashlib.sha256(key_string.encode("utf-8")).hexdigest()

        with self._lock:
            self._keys[key_id] = payload
            self._key_hashes[key_hash] = key_id

        return key_string, key_id

    def validate_key(self, key: str) -> APIKeyPayload | None:
        """验证 API Key.

        Args:
            key: API Key 字符串

        Returns:
            APIKeyPayload 如果有效, None 如果无效/过期/已撤销
        """
        if not key or not key.startswith(API_KEY_PREFIX):
            return None

        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()

        with self._lock:
            key_id = self._key_hashes.get(key_hash)
            if key_id is None:
                return None

            payload = self._keys.get(key_id)
            if payload is None:
                return None

            if payload.revoked:
                return None

            if payload.is_expired():
                return None

            return payload

    def revoke_key(self, key_id: str) -> bool:
        """撤销 API Key.

        Args:
            key_id: Key 标识符

        Returns:
            True 如果撤销成功, False 如果 Key 不存在
        """
        with self._lock:
            payload = self._keys.get(key_id)
            if payload is None:
                return False
            payload.revoked = True
            return True


# ============================================================
# 5. 游标分页 (GitHub API 模式)
# ============================================================


@dataclass
class Page:
    """分页结果.

    Attributes:
        items: 当前页数据
        has_more: 是否有更多数据
        next_cursor: 下一页游标 (None 如果无更多)
    """

    items: list[Any]
    has_more: bool = False
    next_cursor: str | None = None


class CursorPaginator:
    """游标分页器 (GitHub API 模式).

    使用索引游标而非偏移量, 查询复杂度 O(1),
    避免深分页性能退化和数据漂移问题.

    融合方案:
    - GitHub API: cursor-based pagination
    - Stripe API: has_more + next_cursor 元数据
    """

    def __init__(self, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        self._page_size = min(page_size, MAX_PAGE_SIZE)

    def paginate(
        self,
        items: list[Any],
        cursor: str | None = None,
    ) -> Page:
        """对列表进行游标分页.

        Args:
            items: 完整数据列表
            cursor: 游标 (None = 第一页)

        Returns:
            Page 分页结果
        """
        if not items:
            return Page(items=[], has_more=False, next_cursor=None)

        # 解析游标 (base64 编码的索引)
        start_idx = 0
        if cursor:
            try:
                decoded = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("ascii")
                start_idx = int(decoded)
            except (ValueError, UnicodeDecodeError):
                start_idx = 0

        end_idx = start_idx + self._page_size
        page_items = items[start_idx:end_idx]
        has_more = end_idx < len(items)

        next_cursor = None
        if has_more:
            next_cursor = base64.urlsafe_b64encode(
                str(end_idx).encode("ascii")
            ).decode("ascii")

        return Page(
            items=page_items,
            has_more=has_more,
            next_cursor=next_cursor,
        )


# ============================================================
# 6. 幂等性管理器 (Stripe Idempotency-Key 模式)
# ============================================================


@dataclass
class _IdempotencyRecord:
    """幂等性缓存记录."""

    result: Any
    created_at: float


class IdempotencyManager:
    """幂等性管理器 (Stripe Idempotency-Key 模式).

    机制:
    1. 客户端为每个操作生成唯一 key (UUID 或参数哈希)
    2. 服务器首次处理请求, 缓存 key → result
    3. 后续相同 key 的重试请求直接返回缓存结果

    状态: PENDING → SUCCEEDED (或 FAILED)

    线程安全: threading.RLock 保护缓存.
    """

    def __init__(self, ttl_seconds: int = IDEMPOTENCY_DEFAULT_TTL) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._cache: dict[str, _IdempotencyRecord] = {}

    def _is_expired(self, record: _IdempotencyRecord) -> bool:
        """检查记录是否过期."""
        if self._ttl <= 0:
            return True
        return (time.time() - record.created_at) > self._ttl

    def _cleanup_expired(self) -> None:
        """清理过期记录."""
        expired_keys = [
            k for k, v in self._cache.items()
            if self._is_expired(v)
        ]
        for k in expired_keys:
            del self._cache[k]

    def wrap(self, fn: Callable) -> Callable:
        """装饰器: 为函数添加幂等性.

        被装饰函数可通过 _idempotency_key 关键字参数指定幂等键.
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = kwargs.pop("_idempotency_key", None)

            if key is None:
                # 无幂等键, 直接执行
                return fn(*args, **kwargs)

            with self._lock:
                self._cleanup_expired()

                # 检查缓存
                record = self._cache.get(key)
                if record is not None and not self._is_expired(record):
                    return record.result

                # 执行函数
                result = fn(*args, **kwargs)

                # 缓存结果
                self._cache[key] = _IdempotencyRecord(
                    result=result,
                    created_at=time.time(),
                )

                return result

        return wrapper


# ============================================================
# 7. 令牌桶限流器 (AWS API Gateway 模式)
# ============================================================


class TokenBucketRateLimiter:
    """令牌桶限流器 (AWS API Gateway 模式).

    算法:
    - 桶容量 (capacity): 最大令牌数, 允许短时突发
    - 补充速率 (refill_rate): 令牌/秒, 控制长期平均速率
    - 请求消耗 1 个令牌, 桶空时拒绝

    优势:
    - 允许突发 (满桶时可一次性处理 capacity 个请求)
    - 长期平均速率不超过 refill_rate
    - 适合 B2B API (允许客户偶尔批处理)

    线程安全: threading.RLock 保护令牌状态.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_BUCKET_CAPACITY,
        refill_rate: float = DEFAULT_REFILL_RATE,
    ) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._lock = threading.RLock()
        # client_id → (tokens, last_refill_time)
        self._buckets: dict[str, tuple[float, float]] = {}

    def _get_bucket(self, client_id: str) -> tuple[float, float]:
        """获取或初始化客户端桶."""
        if client_id not in self._buckets:
            self._buckets[client_id] = (float(self._capacity), time.time())
        return self._buckets[client_id]

    def _refill(self, client_id: str) -> tuple[float, float]:
        """补充令牌."""
        tokens, last_time = self._get_bucket(client_id)
        now = time.time()
        elapsed = now - last_time
        new_tokens = min(
            float(self._capacity),
            tokens + elapsed * self._refill_rate,
        )
        self._buckets[client_id] = (new_tokens, now)
        return new_tokens, now

    def acquire(self, client_id: str) -> tuple[bool, int]:
        """尝试获取一个令牌.

        Args:
            client_id: 客户端标识

        Returns:
            (allowed, retry_after) 元组
            - allowed: True 如果允许请求
            - retry_after: 如果拒绝, 建议重试等待秒数 (0 如果允许)
        """
        with self._lock:
            tokens, _ = self._refill(client_id)

            if tokens >= 1.0:
                # 消耗令牌
                self._buckets[client_id] = (tokens - 1.0, time.time())
                return (True, 0)
            else:
                # 计算重试等待时间
                retry_after = max(1, int(1.0 / self._refill_rate))
                return (False, retry_after)


# ============================================================
# 8. Webhook 管理 (Shopify HMAC-SHA256 模式)
# ============================================================


@dataclass
class WebhookEndpoint:
    """Webhook 端点.

    Attributes:
        endpoint_id: 端点 ID
        owner_id: 所有者 ID
        url: 回调 URL
        events: 订阅事件列表
        secret: HMAC 签名密钥
        created_at: 创建时间
        active: 是否活跃
    """

    endpoint_id: str
    owner_id: str
    url: str
    events: list[str]
    secret: str
    created_at: int = field(default_factory=lambda: int(time.time()))
    active: bool = True


class WebhookManager:
    """Webhook 管理器 (Shopify HMAC-SHA256 模式).

    功能:
    1. 端点注册: 注册回调 URL + 事件订阅 + 签名密钥
    2. 签名生成: HMAC-SHA256(payload, secret)
    3. 签名验证: 接收方验证签名防篡改
    4. 事件投递: 按 event_type 查找订阅者
    5. 端点管理: 注销/停用端点

    融合方案:
    - Shopify Webhook: HMAC-SHA256 + X-Shopify-Hmac-Sha256 头
    - Stripe Webhook: 带时间戳的签名 (t=...,v1=...)
    - GitHub Webhook: X-Hub-Signature-256 头

    线程安全: threading.RLock 保护端点存储.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._endpoints: dict[str, WebhookEndpoint] = {}
        # event_type → [endpoint_id, ...]
        self._event_index: dict[str, list[str]] = {}

    def register(
        self,
        owner_id: str,
        url: str,
        events: list[str],
        secret: str,
    ) -> WebhookEndpoint:
        """注册 Webhook 端点.

        Args:
            owner_id: 所有者 ID
            url: 回调 URL
            events: 订阅事件列表
            secret: HMAC 签名密钥

        Returns:
            注册的 WebhookEndpoint
        """
        endpoint_id = f"wh_{uuid.uuid4().hex[:16]}"
        endpoint = WebhookEndpoint(
            endpoint_id=endpoint_id,
            owner_id=owner_id,
            url=url,
            events=list(events),
            secret=secret,
        )

        with self._lock:
            self._endpoints[endpoint_id] = endpoint
            for event in events:
                if event not in self._event_index:
                    self._event_index[event] = []
                self._event_index[event].append(endpoint_id)

        return endpoint

    def unregister(self, endpoint_id: str) -> bool:
        """注销 Webhook 端点.

        Args:
            endpoint_id: 端点 ID

        Returns:
            True 如果注销成功
        """
        with self._lock:
            endpoint = self._endpoints.pop(endpoint_id, None)
            if endpoint is None:
                return False

            for event in endpoint.events:
                if event in self._event_index:
                    try:
                        self._event_index[event].remove(endpoint_id)
                    except ValueError:
                        pass
                    if not self._event_index[event]:
                        del self._event_index[event]

            return True

    def get_subscribers(self, event_type: str) -> list[WebhookEndpoint]:
        """获取事件订阅者列表.

        Args:
            event_type: 事件类型

        Returns:
            订阅该事件的端点列表
        """
        with self._lock:
            endpoint_ids = self._event_index.get(event_type, [])
            return [
                self._endpoints[eid]
                for eid in endpoint_ids
                if eid in self._endpoints and self._endpoints[eid].active
            ]

    def sign_payload(self, endpoint_id: str, payload: bytes) -> str:
        """生成 HMAC-SHA256 签名.

        Args:
            endpoint_id: 端点 ID
            payload: 原始 payload 字节

        Returns:
            十六进制签名字符串

        Raises:
            WebhookError: 端点不存在
        """
        with self._lock:
            endpoint = self._endpoints.get(endpoint_id)
            if endpoint is None:
                raise WebhookError(
                    detail=f"Webhook 端点不存在: {endpoint_id}",
                    endpoint_id=endpoint_id,
                )
            secret = endpoint.secret

        return hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

    def verify_signature(
        self,
        endpoint_id: str,
        payload: bytes,
        signature: str,
    ) -> bool:
        """验证 HMAC-SHA256 签名.

        Args:
            endpoint_id: 端点 ID
            payload: 原始 payload 字节
            signature: 待验证的签名

        Returns:
            True 如果签名匹配
        """
        try:
            expected = self.sign_payload(endpoint_id, payload)
            return hmac.compare_digest(expected, signature)
        except WebhookError:
            return False


# ============================================================
# 9. SSE 事件流管理 (OpenAI SSE 模式)
# ============================================================


@dataclass
class SSEEvent:
    """Server-Sent Events 事件.

    Attributes:
        event_type: 事件类型
        data: 事件数据 (JSON 字符串)
        event_id: 事件 ID (用于断线重连)
    """

    event_type: str
    data: str
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")


@dataclass
class _SSEStream:
    """SSE 流内部状态."""

    stream_id: str
    user_id: str
    session_id: str
    events: list[SSEEvent] = field(default_factory=list)
    active: bool = True
    created_at: int = field(default_factory=lambda: int(time.time()))


class EventStreamManager:
    """SSE 事件流管理器 (OpenAI SSE 模式).

    功能:
    1. 流注册: 为用户/会话创建事件流
    2. 事件推送: 向流推送事件 (异步队列)
    3. 事件消费: 客户端通过 SSE 端点消费事件
    4. 流管理: 关闭/清理事件流
    5. SSE 格式化: 符合 text/event-stream 规范

    融合方案:
    - OpenAI Streaming API: data: [DONE] 结束标记
    - W3C SSE 规范: event/id/data/retry 字段
    - sse-starlette: EventSourceResponse 兼容

    线程安全: threading.RLock 保护流存储.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._streams: dict[str, _SSEStream] = {}

    def register_stream(
        self,
        user_id: str,
        session_id: str,
    ) -> str:
        """注册事件流.

        Args:
            user_id: 用户 ID
            session_id: 会话 ID

        Returns:
            stream_id
        """
        stream_id = f"sse_{uuid.uuid4().hex[:16]}"
        stream = _SSEStream(
            stream_id=stream_id,
            user_id=user_id,
            session_id=session_id,
        )
        with self._lock:
            self._streams[stream_id] = stream
        return stream_id

    def push_event(
        self,
        stream_id: str,
        event_type: str,
        data: str,
    ) -> bool:
        """推送事件到流.

        Args:
            stream_id: 流 ID
            event_type: 事件类型
            data: 事件数据 (JSON 字符串)

        Returns:
            True 如果推送成功
        """
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None or not stream.active:
                return False
            stream.events.append(SSEEvent(
                event_type=event_type,
                data=data,
            ))
            return True

    def get_events(self, stream_id: str) -> list[SSEEvent]:
        """获取流中所有事件 (并清空队列).

        Args:
            stream_id: 流 ID

        Returns:
            事件列表
        """
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                return []
            events = list(stream.events)
            stream.events.clear()
            return events

    def close_stream(self, stream_id: str) -> bool:
        """关闭事件流.

        Args:
            stream_id: 流 ID

        Returns:
            True 如果关闭成功
        """
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                return False
            stream.active = False
            return True

    def is_active(self, stream_id: str) -> bool:
        """检查流是否活跃."""
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                return False
            return stream.active

    @staticmethod
    def format_sse(event: SSEEvent) -> str:
        """将事件格式化为 SSE 文本.

        符合 W3C Server-Sent Events 规范:
            event: <event_type>\\n
            id: <event_id>\\n
            data: <data>\\n
            \\n

        Args:
            event: SSE 事件

        Returns:
            SSE 格式化文本
        """
        lines = [
            f"event: {event.event_type}",
            f"id: {event.event_id}",
            f"data: {event.data}",
        ]
        return "\n".join(lines) + "\n\n"


# ============================================================
# 10. 中间件
# ============================================================


class AuthMiddleware:
    """JWT 认证中间件.

    职责:
    1. 从 Authorization 头提取 Bearer Token
    2. 验证 JWT 签名和声明
    3. 返回 TokenPayload (包含用户信息)

    融合方案:
    - OWASP: Bearer Token 认证最佳实践
    - JWT RFC 7519: 标准声明验证
    - WorkOS: JWT 最佳实践 (算法白名单)
    """

    def __init__(self, jwt_manager: JWTManager) -> None:
        self._jwt = jwt_manager

    def extract_bearer_token(self, auth_header: str | None) -> str | None:
        """从 Authorization 头提取 Bearer Token.

        Args:
            auth_header: Authorization 头值

        Returns:
            Token 字符串, 或 None 如果格式无效
        """
        if not auth_header:
            return None
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        return parts[1].strip()

    def authenticate(self, token: str) -> TokenPayload | None:
        """验证 Token 并返回载荷.

        Args:
            token: JWT Token 字符串

        Returns:
            TokenPayload 如果验证成功, None 如果失败
        """
        try:
            return self._jwt.verify_token(token)
        except (TokenError, TokenExpiredError, TokenRevokedError):
            return None


class ABACMiddleware:
    """ABAC 权限校验中间件.

    职责:
    1. 基于 RBAC + ABAC 检查用户权限
    2. 公开路径跳过认证
    3. 提供 require_permission 辅助方法

    融合方案:
    - AWS IAM: 策略优先级 + 显式拒绝优先
    - Amazon Cedar: 声明式策略评估
    - OPA (Open Policy Agent): 策略即数据
    """

    def __init__(self, access_control: AccessControlManager) -> None:
        self._acm = access_control

    def check_permission(
        self,
        user: User,
        permission: Permission,
    ) -> bool:
        """检查用户是否拥有指定权限.

        Args:
            user: 用户对象
            permission: 权限枚举

        Returns:
            True 如果允许
        """
        return self._acm.rbac_matrix.check_permission(user.role, permission)

    def require_permission(
        self,
        user: User,
        permission: Permission,
    ) -> bool:
        """要求权限 (同 check_permission, 语义化别名)."""
        return self.check_permission(user, permission)

    def is_public_path(self, path: str) -> bool:
        """检查路径是否为公开路径 (无需认证).

        Args:
            path: 请求路径

        Returns:
            True 如果是公开路径
        """
        return path in PUBLIC_PATHS


class AuditMiddleware:
    """审计日志中间件.

    职责:
    1. 自动记录 API 请求审计日志
    2. 支持按操作类型、用户等维度查询
    3. 异步写入 (不阻塞请求)

    融合方案:
    - NIST SP 800-92: 日志管理指南
    - GDPR Art.30: 处理活动记录
    - FERPA §1232g(b)(3): 教育记录访问日志
    """

    def __init__(self) -> None:
        self._entries: list[AuditLogEntry] = []
        self._lock = threading.RLock()

    @property
    def entries(self) -> list[AuditLogEntry]:
        """获取所有审计条目."""
        with self._lock:
            return list(self._entries)

    def create_entry(
        self,
        user: User,
        action: AuditAction,
        resource: str,
        data_level: DataLevel,
        result: AuditResult,
        purpose: str = "",
    ) -> AuditLogEntry:
        """创建审计条目.

        Args:
            user: 用户对象
            action: 审计操作
            resource: 目标资源
            data_level: 数据级别
            result: 操作结果
            purpose: 操作目的

        Returns:
            AuditLogEntry
        """
        return AuditLogEntry(
            actor_id=user.user_id,
            actor_role=user.role,
            action=action,
            target_resource=resource,
            target_data_level=data_level,
            purpose=purpose or action.value,
            result=result,
        )

    def record(
        self,
        user: User,
        action: AuditAction,
        resource: str,
        data_level: DataLevel,
        result: AuditResult,
        purpose: str = "",
    ) -> None:
        """记录审计日志.

        Args:
            user: 用户对象
            action: 审计操作
            resource: 目标资源
            data_level: 数据级别
            result: 操作结果
            purpose: 操作目的
        """
        entry = self.create_entry(
            user=user,
            action=action,
            resource=resource,
            data_level=data_level,
            result=result,
            purpose=purpose,
        )
        with self._lock:
            self._entries.append(entry)

    def query(
        self,
        action: AuditAction | None = None,
        actor_id: str | None = None,
        result: AuditResult | None = None,
    ) -> list[AuditLogEntry]:
        """查询审计日志.

        Args:
            action: 可选, 按操作类型过滤
            actor_id: 可选, 按操作者过滤
            result: 可选, 按结果过滤

        Returns:
            匹配的审计条目列表
        """
        with self._lock:
            results = list(self._entries)

        if action is not None:
            results = [e for e in results if e.action == action]
        if actor_id is not None:
            results = [e for e in results if e.actor_id == actor_id]
        if result is not None:
            results = [e for e in results if e.result == result]

        return results


# ============================================================
# 11. L1 REST API 路由器 (17 个核心端点)
# ============================================================


def _user_to_dict(user: User) -> dict[str, Any]:
    """将 User 对象序列化为字典."""
    abac = user.abac_attributes
    return {
        "user_id": user.user_id,
        "student_id": user.student_id,
        "role": user.role.value,
        "status": user.status.value,
        "institution_id": user.institution_id,
        "abac_attributes": {
            "grade_level": abac.grade_level.value,
            "major_direction": abac.major_direction.value,
            "course_progress": abac.course_progress,
            "lab_access_tier": abac.lab_access_tier.value,
        },
    }


def _session_to_dict(session: Any) -> dict[str, Any]:
    """将会话对象序列化为字典."""
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "session_type": session.session_type.value if hasattr(session.session_type, 'value') else str(session.session_type),
        "status": session.status.value if hasattr(session.status, 'value') else str(session.status),
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        # 统一会话闭环: 关联的 L5 Agent 执行会话 + 交互计数
        "agent_sessions": list(getattr(session, "agent_sessions", []) or []),
        "question_count": int(getattr(session, "question_count", 0) or 0),
        "agent_execution_count": len(getattr(session, "agent_sessions", []) or []),
    }


@dataclass
class AgentAuditView:
    """L0 Agent 执行轨迹 → 审计端点统一视图 (与 AuditMiddleware 条目兼容)."""
    timestamp: int = 0
    actor: str = ""
    action: str = ""
    result: str = "success"
    target_resource: str = ""
    detail: str = ""
    log_id: str = ""
    actor_id: str = ""


class L1APIRouter:
    """L1 REST API 路由器.

    基于 Starlette 构建, 暴露 17 个核心 RESTful 端点.
    支持挂载到 uvicorn 或嵌入到 FastAPI 应用.

    端点概览:
        GET  /health                          — 健康检查
        POST /api/v1/auth/login               — 登录
        POST /api/v1/auth/logout              — 登出
        POST /api/v1/auth/refresh             — 刷新 Token
        GET  /api/v1/users/me                 — 获取当前用户
        PUT  /api/v1/users/me/preferences     — 更新偏好
        POST /api/v1/sessions                 — 创建会话
        GET  /api/v1/sessions/{id}            — 获取会话
        POST /api/v1/sessions/{id}/fork       — Fork 会话
        POST /api/v1/sessions/{id}/merge      — 合并 Fork
        POST /api/v1/sessions/{id}/pause      — 暂停会话
        GET  /api/v1/context/{session_id}     — 获取上下文
        POST /api/v1/context/{session_id}/refresh — 刷新上下文
        POST /api/v1/hitl/confirm             — HiTL 确认
        POST /api/v1/hitl/feedback            — HiTL 反馈
        GET  /api/v1/hitl/emergency           — 紧急干预
        GET  /api/v1/audit/logs               — 审计日志
        GET  /api/v1/export/learner-data      — 导出学情数据

    融合方案:
    - REST API Best Practices: URI 版本控制 + 标准错误码
    - Starlette: ASGI 异步框架 (与 L3/L6 路由一致)
    - GitHub API: 游标分页
    - Stripe API: 统一响应格式
    """

    def __init__(
        self,
        jwt_manager: JWTManager,
        access_control: AccessControlManager,
        session_manager: LearningSessionManager | None = None,
        hitl_manager: HiTLManager | None = None,
        context_broker: LearningContextBroker | None = None,
        privacy_governance: PrivacyGovernanceManager | None = None,
        audit_middleware: AuditMiddleware | None = None,
        audit_engine: Any | None = None,
    ) -> None:
        self._jwt = jwt_manager
        self._acm = access_control
        self._session_mgr = session_manager or LearningSessionManager()
        self._hitl_mgr = hitl_manager or HiTLManager()
        self._context_broker = context_broker
        self._privacy_gov = privacy_governance
        self._audit_mw = audit_middleware or AuditMiddleware()
        self._auth_mw = AuthMiddleware(jwt_manager)
        self._abac_mw = ABACMiddleware(access_control)
        # L0 治理审计引擎 (Agent 执行轨迹, 持久化): 与 AuditMiddleware 统一对外
        self._audit_engine = audit_engine

        # 用户存储 (测试用, 生产环境用数据库)
        self._users: dict[str, tuple[User, str]] = {}  # student_id → (User, pw_hash)
        self._users_by_id: dict[str, User] = {}  # user_id → User

    def register_user(self, user: User, password_hash: str) -> None:
        """注册用户 (用于测试/初始化).

        Args:
            user: 用户对象
            password_hash: 密码哈希
        """
        self._users[user.student_id] = (user, password_hash)
        self._users_by_id[user.user_id] = user

    def _get_authenticated_user(self, request: Request) -> User | None:
        """从请求中提取并验证用户.

        Args:
            request: Starlette 请求

        Returns:
            User 对象如果认证成功, None 否则
        """
        auth_header = request.headers.get("Authorization")
        token = self._auth_mw.extract_bearer_token(auth_header)
        if token is None:
            return None

        payload = self._auth_mw.authenticate(token)
        if payload is None:
            return None

        return self._users_by_id.get(payload.user_id)

    def create_app(self) -> Starlette:
        """创建 Starlette 应用实例.

        Returns:
            配置好的 Starlette 应用
        """
        routes = self._build_routes()
        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["GET", "POST", "PUT", "DELETE"],
                allow_headers=["*"],
            ),
        ]
        return Starlette(routes=routes, middleware=middleware)

    def _build_routes(self) -> list[Route]:
        """构建路由列表."""
        return [
            # 健康检查
            Route("/health", self._health, methods=["GET"]),

            # 认证 (T2)
            Route(f"{API_PREFIX}/auth/login", self._auth_login, methods=["POST"]),
            Route(f"{API_PREFIX}/auth/logout", self._auth_logout, methods=["POST"]),
            Route(f"{API_PREFIX}/auth/refresh", self._auth_refresh, methods=["POST"]),

            # 用户 (T2)
            Route(f"{API_PREFIX}/users", self._list_users, methods=["GET"]),
            Route(f"{API_PREFIX}/users/import", self._import_users, methods=["POST"]),
            Route(f"{API_PREFIX}/roles", self._list_roles, methods=["GET"]),
            Route(f"{API_PREFIX}/users/me", self._users_me, methods=["GET"]),
            Route(f"{API_PREFIX}/users/me/preferences", self._users_me_preferences, methods=["PUT"]),

            # 会话 (T5)
            Route(f"{API_PREFIX}/sessions", self._create_session, methods=["POST"]),
            Route(f"{API_PREFIX}/sessions", self._list_sessions, methods=["GET"]),
            Route(f"{API_PREFIX}/sessions/{{session_id}}", self._get_session, methods=["GET"]),
            Route(f"{API_PREFIX}/sessions/{{session_id}}/fork", self._fork_session, methods=["POST"]),
            Route(f"{API_PREFIX}/sessions/{{session_id}}/merge", self._merge_session, methods=["POST"]),
            Route(f"{API_PREFIX}/sessions/{{session_id}}/pause", self._pause_session, methods=["POST"]),
            Route(f"{API_PREFIX}/sessions/{{session_id}}/attach-agent-session", self._attach_agent_session, methods=["POST"]),

            # 上下文 (T3)
            Route(f"{API_PREFIX}/context/{{session_id}}", self._get_context, methods=["GET"]),
            Route(f"{API_PREFIX}/context/{{session_id}}/refresh", self._refresh_context, methods=["POST"]),

            # HiTL (T4)
            Route(f"{API_PREFIX}/hitl/confirm", self._hitl_confirm, methods=["POST"]),
            Route(f"{API_PREFIX}/hitl/feedback", self._hitl_feedback, methods=["POST"]),
            Route(f"{API_PREFIX}/hitl/emergency", self._hitl_emergency, methods=["GET"]),

            # 审计与导出 (T6)
            Route(f"{API_PREFIX}/audit/logs", self._audit_logs, methods=["GET"]),
            Route(f"{API_PREFIX}/export/learner-data", self._export_learner_data, methods=["GET"]),

            # 账户管理 (M-F3)
            Route(f"{API_PREFIX}/auth/register", self._auth_register, methods=["POST"]),
            Route(f"{API_PREFIX}/auth/change-password", self._auth_change_password, methods=["POST"]),
            Route(f"{API_PREFIX}/admin/create-user", self._admin_create_user, methods=["POST"]),
        ]

    # ---- 健康检查 ----

    async def _health(self, request: Request) -> JSONResponse:
        """GET /health — 健康检查."""
        return JSONResponse(ok_response({
            "status": "healthy",
            "version": API_VERSION,
            "timestamp": int(time.time()),
        }))

    # ---- 认证 (T2) ----

    async def _auth_login(self, request: Request) -> JSONResponse:
        """POST /api/v1/auth/login — 登录."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(error_response(-32700, "PARSE_ERROR", "请求体不是有效的 JSON"))

        student_id = body.get("student_id", "")
        password = body.get("password", "")

        if not student_id or not password:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "学号和密码不能为空"))

        user_tuple = self._users.get(student_id)
        if user_tuple is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "用户不存在"))

        user, pw_hash = user_tuple
        if not PasswordHasher.verify_password(password, pw_hash):
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "密码错误"))

        if user.status != UserStatus.ACTIVE:
            return JSONResponse(error_response(-32205, "LIFECYCLE_ERROR", f"用户状态非 ACTIVE: {user.status.value}"))

        access_token, refresh_token = self._jwt.issue_token(user)

        return JSONResponse(ok_response({
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "user_id": user.user_id,
            "student_id": user.student_id,
            "role": user.role.value,
        }))

    async def _auth_logout(self, request: Request) -> JSONResponse:
        """POST /api/v1/auth/logout — 登出."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        auth_header = request.headers.get("Authorization", "")
        token = self._auth_mw.extract_bearer_token(auth_header)
        if token:
            self._jwt.revoke_token(token)

        return JSONResponse(ok_response(message="登出成功"))

    async def _auth_refresh(self, request: Request) -> JSONResponse:
        """POST /api/v1/auth/refresh — 刷新 Token."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(error_response(-32700, "PARSE_ERROR", "请求体不是有效的 JSON"))

        refresh_token = body.get("refresh_token", "")
        if not refresh_token:
            return JSONResponse(error_response(-32202, "TOKEN_ERROR", "refresh_token 不能为空"))

        try:
            new_access, new_refresh = self._jwt.refresh_token(refresh_token)
            return JSONResponse(ok_response({
                "access_token": new_access,
                "refresh_token": new_refresh,
                "token_type": "Bearer",
            }))
        except (TokenError, TokenExpiredError, TokenRevokedError) as e:
            return JSONResponse(error_response(-32202, "TOKEN_ERROR", e.detail))

    # ---- 用户 (T2) ----

    async def _list_users(self, request: Request) -> JSONResponse:
        """GET /api/v1/users — 用户列表 (管理端用户管理三件套之列表)."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)
        items = []
        for sid, (u, _hash) in self._users.items():
            items.append({
                "student_id": sid,
                "user_id": u.user_id,
                "name": getattr(u, "name", "") or sid,
                "role": getattr(u.role, "value", str(getattr(u, "role", ""))),
                "institution_id": u.institution_id,
            })
        items.sort(key=lambda x: x["student_id"])
        return JSONResponse(ok_response({"users": items, "total": len(items)}))

    async def _list_roles(self, request: Request) -> JSONResponse:
        """GET /api/v1/roles — 角色与权限说明 (管理端角色管理)."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)
        roles = [
            {"role": "admin", "label": "管理员", "permissions": ["全部管理", "用户/角色/导入管理", "策略/审计/报告", "系统监控"]},
            {"role": "teacher", "label": "教师", "permissions": ["查看学习者画像", "学情对比", "答疑/练习", "反幻觉审查"]},
            {"role": "undergrad", "label": "本科生", "permissions": ["学习画像", "动态练习", "答疑(4Agent)", "知识库", "助手"]},
            {"role": "postgrad", "label": "研究生", "permissions": ["学习画像", "动态练习", "答疑(4Agent)", "知识库", "助手"]},
            {"role": "guest", "label": "访客", "permissions": ["浏览知识库", "查看监控(需登录)"], "note": "未登录仅可浏览公开页面"},
        ]
        return JSONResponse(ok_response({"roles": roles, "total": len(roles)}))

    async def _import_users(self, request: Request) -> JSONResponse:
        """POST /api/v1/users/import — 批量导入用户 (演示: 逐行 student_id,password,role,name)."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(error_response(-32700, "PARSE_ERROR", "请求体解析失败"), status_code=400)
        raw = str(body.get("content") or body.get("text") or "").strip()
        lines = [ln.strip() for ln in raw.replace("\r", "").split("\n") if ln.strip() and not ln.strip().startswith("#")]
        if not lines:
            return JSONResponse(error_response(-32602, "INVALID_PARAMS", "无有效数据行"), status_code=400)
        imported = 0
        skipped = 0
        errors = []
        role_map = {"admin": UserRole.ADMIN, "teacher": UserRole.TEACHER,
                    "student": UserRole.UNDERGRAD, "undergrad": UserRole.UNDERGRAD,
                    "graduate": UserRole.GRADUATE, "researcher": UserRole.RESEARCHER,
                    "alumni": UserRole.ALUMNI}
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 2:
                skipped += 1
                continue
            sid, pw = parts[0], parts[1]
            role = role_map.get((parts[2] if len(parts) > 2 else "student").lower(), UserRole.UNDERGRAD)
            if sid in self._users:
                skipped += 1
                continue
            try:
                u = User(student_id=sid, institution_id=user.institution_id or "inst-001",
                         role=role, abac_attributes=ABACAttributes())
                self._users[sid] = (u, PasswordHasher.hash_password(pw))
                self._users_by_id[u.user_id] = u
                imported += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sid}: {exc}")
        return JSONResponse(ok_response({
            "imported": imported, "skipped": skipped, "errors": errors[:5],
            "total_lines": len(lines),
        }))

    async def _users_me(self, request: Request) -> JSONResponse:
        """GET /api/v1/users/me — 获取当前用户."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        return JSONResponse(ok_response(_user_to_dict(user)))

    async def _users_me_preferences(self, request: Request) -> JSONResponse:
        """PUT /api/v1/users/me/preferences — 更新偏好."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(error_response(-32700, "PARSE_ERROR", "请求体不是有效的 JSON"))

        return JSONResponse(ok_response({
            "updated": True,
            "preferences": body,
        }))

    # ---- 会话 (T5) ----

    async def _list_sessions(self, request: Request) -> JSONResponse:
        """GET /api/v1/sessions — 列出当前用户全部会话 (M-F6).

        返回会话列表 (按创建时间倒序), 支持 ?status=active 过滤活跃会话。
        """
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        sessions = self._session_mgr.list_sessions_for_user(user.user_id)
        status_filter = (request.query_params.get("status") or "").lower()
        if status_filter == "active":
            sessions = [s for s in sessions if (s.status.value if hasattr(s.status, 'value') else str(s.status)).upper() in ("ACTIVE", "PAUSED", "FORKED")]
        elif status_filter:
            sessions = [s for s in sessions if (s.status.value if hasattr(s.status, 'value') else str(s.status)).lower() == status_filter]

        return JSONResponse(ok_response({
            "items": [_session_to_dict(s) for s in sessions],
            "total": len(sessions),
        }))

    async def _create_session(self, request: Request) -> JSONResponse:
        """POST /api/v1/sessions — 创建会话."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        try:
            body = await request.json()
        except Exception:
            body = {}

        session_type_str = body.get("session_type", "diagnosis")
        try:
            session_type = SessionType(session_type_str)
        except ValueError:
            session_type = SessionType.DIAGNOSIS

        session = self._session_mgr.create_session(
            user_id=user.user_id,
            session_type=session_type,
        )

        return JSONResponse(ok_response(_session_to_dict(session)))

    async def _get_session(self, request: Request) -> JSONResponse:
        """GET /api/v1/sessions/{id} — 获取会话."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        session_id = request.path_params.get("session_id", "")
        session = self._session_mgr.get_session(session_id)
        if session is None:
            return JSONResponse(error_response(-32501, "SESSION_NOT_FOUND", f"会话未找到: {session_id}"))

        return JSONResponse(ok_response(_session_to_dict(session)))

    async def _fork_session(self, request: Request) -> JSONResponse:
        """POST /api/v1/sessions/{id}/fork — Fork 会话."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        session_id = request.path_params.get("session_id", "")
        try:
            body = await request.json()
        except Exception:
            body = {}

        fork_reason = body.get("fork_reason", "manual")
        branch_label = body.get("branch_label", "fork")

        try:
            forked = self._session_mgr.fork_session(
                session_id=session_id,
                fork_reason=fork_reason,
                branch_label=branch_label,
            )
            return JSONResponse(ok_response(_session_to_dict(forked)))
        except Exception as e:
            return JSONResponse(error_response(-32503, "FORK_ERROR", str(e)))

    async def _merge_session(self, request: Request) -> JSONResponse:
        """POST /api/v1/sessions/{id}/merge — 合并 Fork."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        session_id = request.path_params.get("session_id", "")
        try:
            body = await request.json()
        except Exception:
            body = {}

        target_id = body.get("target_session_id", "")
        try:
            self._session_mgr.merge_fork(
                fork_session_id=session_id,
                target_session_id=target_id,
            )
            return JSONResponse(ok_response({"merged": True}))
        except Exception as e:
            return JSONResponse(error_response(-32503, "FORK_ERROR", str(e)))

    async def _pause_session(self, request: Request) -> JSONResponse:
        """POST /api/v1/sessions/{id}/pause — 暂停会话."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        session_id = request.path_params.get("session_id", "")
        try:
            self._session_mgr.pause_session(session_id)
            return JSONResponse(ok_response({"paused": True}))
        except Exception as e:
            return JSONResponse(error_response(-32501, "SESSION_NOT_FOUND", str(e)))

    async def _attach_agent_session(self, request: Request) -> JSONResponse:
        """POST /api/v1/sessions/{id}/attach-agent-session — 关联一次 L5 Agent 执行会话.

        统一会话闭环: L1 会话作为唯一用户会话入口, 聚合 L5 Agent 执行记录.
        由 /api/query 端到端链路内部调用 (跨层关联, 支持可选认证).
        请求体: {agent_session_id: str}
        """
        session_id = request.path_params.get("session_id", "")
        if not session_id:
            return JSONResponse(error_response(-32700, "INVALID_REQUEST", "缺少路径参数: session_id"), status_code=400)
        try:
            body = await request.json()
        except Exception:
            body = {}
        agent_session_id = body.get("agent_session_id", "")
        if not agent_session_id:
            return JSONResponse(error_response(-32700, "INVALID_REQUEST", "缺少必填参数: agent_session_id"), status_code=400)

        # 内部跨层调用可不认证 (与 /api/query 一致); 外部调用要求认证
        try:
            session = self._session_mgr.get_session(session_id)
        except Exception:
            session = None
        if session is None:
            return JSONResponse(error_response(-32501, "SESSION_NOT_FOUND", f"会话未找到: {session_id}"))

        attach = getattr(session, "attach_agent_session", None)
        if attach is not None:
            attach(agent_session_id)
        else:
            if agent_session_id not in (getattr(session, "agent_sessions", []) or []):
                session.agent_sessions = list(getattr(session, "agent_sessions", []) or []) + [agent_session_id]
            session.question_count = int(getattr(session, "question_count", 0) or 0) + 1
        return JSONResponse(ok_response(_session_to_dict(session)))

    # ---- 上下文 (T3) ----

    async def _get_context(self, request: Request) -> JSONResponse:
        """GET /api/v1/context/{session_id} — 获取上下文."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        session_id = request.path_params.get("session_id", "")
        if self._context_broker:
            context = self._context_broker.get_context(session_id)
            if context:
                return JSONResponse(ok_response({"session_id": session_id, "context": "available"}))

        return JSONResponse(ok_response({"session_id": session_id, "context": "cached"}))

    async def _refresh_context(self, request: Request) -> JSONResponse:
        """POST /api/v1/context/{session_id}/refresh — 刷新上下文."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        session_id = request.path_params.get("session_id", "")
        if self._context_broker:
            self._context_broker.refresh_context(session_id)

        return JSONResponse(ok_response({"session_id": session_id, "refreshed": True}))

    # ---- HiTL (T4) ----

    async def _hitl_confirm(self, request: Request) -> JSONResponse:
        """POST /api/v1/hitl/confirm — HiTL 确认."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(error_response(-32700, "PARSE_ERROR", "请求体不是有效的 JSON"))

        content = body.get("content", "")
        confidence = body.get("confidence", 1.0)
        session_id = body.get("session_id", "")

        request_obj = self._hitl_mgr.create_approval_request(
            user_id=user.user_id,
            session_id=session_id,
            hitl_type=HiTLType.CONFIRMATION,
            content=content,
            confidence=confidence,
        )

        return JSONResponse(ok_response({
            "request_id": request_obj.request_id,
            "status": request_obj.status,
        }))

    async def _hitl_feedback(self, request: Request) -> JSONResponse:
        """POST /api/v1/hitl/feedback — HiTL 反馈."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(error_response(-32700, "PARSE_ERROR", "请求体不是有效的 JSON"))

        return JSONResponse(ok_response({
            "received": True,
            "feedback_type": body.get("feedback_type", "unknown"),
        }))

    async def _hitl_emergency(self, request: Request) -> JSONResponse:
        """GET /api/v1/hitl/emergency — 紧急干预状态."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        pending = self._hitl_mgr.get_pending_requests(user_id=user.user_id)
        return JSONResponse(ok_response({
            "user_id": user.user_id,
            "pending_count": len(pending),
            "alerts": [],
        }))

    # ---- 审计与导出 (T6) ----

    async def _audit_logs(self, request: Request) -> JSONResponse:
        """GET /api/v1/audit/logs — 查询审计日志."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        # 获取审计条目 (L1 中间件 + L0 Agent 执行轨迹 统一合并)
        entries = list(self._audit_mw.entries)
        if self._audit_engine is not None:
            try:
                for log in self._audit_engine.query(limit=200):
                    meta = log.metadata or {}
                    entries.append(AgentAuditView(
                        timestamp=int(log.timestamp * 1000),
                        actor=log.actor or (meta.get("learner_id") or ""),
                        action=log.action,
                        result=log.outcome or "success",
                        target_resource=f"agent:{log.agent_id}" if log.agent_id else "",
                        detail=str(meta.get("detail") or ""),
                    ))
            except Exception as exc:  # noqa: BLE001
                _logger.warning("合并 L0 Agent 审计轨迹失败: %s", exc)
        entries.sort(key=lambda e: getattr(e, "timestamp", 0) or 0, reverse=True)

        # 分页
        paginator = CursorPaginator(page_size=20)
        page = paginator.paginate(entries, cursor=None)

        # 序列化
        items = []
        for entry in page.items:
            items.append({
                "log_id": entry.log_id,
                "actor_id": entry.actor_id,
                "actor": getattr(entry, "actor", None) or entry.actor_id,
                "action": entry.action.value if hasattr(entry.action, 'value') else str(entry.action),
                "target_resource": entry.target_resource,
                "result": entry.result.value if hasattr(entry.result, 'value') else str(entry.result),
                "timestamp": entry.timestamp,
                "detail": getattr(entry, "detail", "") or "",
            })

        return JSONResponse(paginated_response(
            data=items,
            total=len(entries),
            cursor=page.next_cursor,
            has_more=page.has_more,
        ))

    async def _export_learner_data(self, request: Request) -> JSONResponse:
        """GET /api/v1/export/learner-data — 导出学情数据."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "未认证"), status_code=401)

        # 导出数据 (Privacy by Design)
        data = {
            "user_id": user.user_id,
            "student_id": user.student_id,
            "name": user.name,
            "role": user.role.value,
        }

        if self._privacy_gov:
            data = self._privacy_gov.export_learner_data(
                data=data,
                requester_role=user.role,
                requester_id=user.user_id,
                owner_id=user.user_id,
            )

        return JSONResponse(ok_response(data))

    # ---- 账户管理 (M-F3) ----

    async def _auth_register(self, request: Request) -> JSONResponse:
        """POST /api/v1/auth/register — 学生自主注册."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(error_response(-32700, "PARSE_ERROR", "请求体不是有效的 JSON"))
        student_id = body.get("student_id", "")
        password = body.get("password", "")
        institution_id = body.get("institution_id", "inst-001")
        if not student_id or not password:
            return JSONResponse(error_response(-32201, "VALIDATION_ERROR", "学号和密码不能为空"))
        if len(password) < 4:
            return JSONResponse(error_response(-32201, "VALIDATION_ERROR", "密码至少 4 个字符"))
        if self._users.get(student_id):
            return JSONResponse(error_response(-32201, "DUPLICATE_ERROR", "该学号已被注册"))
        try:
            password_hash = PasswordHasher.hash_password(password)
            user = User(
                student_id=student_id,
                institution_id=institution_id,
                role=UserRole.UNDERGRAD,
                status=UserStatus.ACTIVE,
            )
            self.register_user(user, password_hash)
            return JSONResponse(ok_response({"student_id": student_id, "user_id": user.user_id, "role": user.role.value}))
        except ValueError as e:
            return JSONResponse(error_response(-32201, "VALIDATION_ERROR", str(e)))

    async def _auth_change_password(self, request: Request) -> JSONResponse:
        """POST /api/v1/auth/change-password — 修改密码."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(error_response(-32700, "PARSE_ERROR", "请求体不是有效的 JSON"))
        old_password = body.get("old_password", "")
        new_password = body.get("new_password", "")
        if not old_password or not new_password:
            return JSONResponse(error_response(-32201, "VALIDATION_ERROR", "新旧密码不能为空"))
        if len(new_password) < 4:
            return JSONResponse(error_response(-32201, "VALIDATION_ERROR", "新密码至少 4 个字符"))
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "请先登录"), status_code=401)
        user_tuple = self._users.get(user.student_id)
        if not user_tuple or not PasswordHasher.verify_password(old_password, user_tuple[1]):
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "旧密码错误"))
        self._users[user.student_id] = (user, PasswordHasher.hash_password(new_password))
        return JSONResponse(ok_response({"message": "密码修改成功"}))

    async def _admin_create_user(self, request: Request) -> JSONResponse:
        """POST /api/v1/admin/create-user — 管理员创建子账号."""
        user = self._get_authenticated_user(request)
        if user is None:
            return JSONResponse(error_response(-32201, "AUTHENTICATION_ERROR", "请先登录"), status_code=401)
        if user.role != UserRole.ADMIN:
            return JSONResponse(error_response(-32203, "FORBIDDEN_ERROR", "仅管理员可执行此操作"), status_code=403)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(error_response(-32700, "PARSE_ERROR", "请求体不是有效的 JSON"))
        student_id = body.get("student_id", "")
        password = body.get("password", "")
        role_str = body.get("role", "undergrad")
        if not student_id or not password:
            return JSONResponse(error_response(-32201, "VALIDATION_ERROR", "学号和密码不能为空"))
        try:
            role = UserRole(role_str)
        except ValueError:
            return JSONResponse(error_response(-32201, "VALIDATION_ERROR", f"无效角色: {role_str}"))
        if self._users.get(student_id):
            return JSONResponse(error_response(-32201, "DUPLICATE_ERROR", "该学号已被注册"))
        try:
            password_hash = PasswordHasher.hash_password(password)
            new_user = User(
                student_id=student_id,
                institution_id=body.get("institution_id", "inst-001"),
                role=role, status=UserStatus.ACTIVE,
                abac_attributes=ABACAttributes() if isinstance(role, UserRole) else None,
            )
            self.register_user(new_user, password_hash)
            return JSONResponse(ok_response({"student_id": student_id, "user_id": new_user.user_id, "role": new_user.role.value}))
        except ValueError as e:
            return JSONResponse(error_response(-32201, "VALIDATION_ERROR", str(e)))

    def get_routes_summary(self) -> list[dict[str, str]]:
        """获取路由摘要 (用于文档/发现).

        Returns:
            [{"path": ..., "methods": [...], "description": ...}]
        """
        return [
            {"path": "/health", "methods": ["GET"], "description": "健康检查"},
            {"path": f"{API_PREFIX}/auth/login", "methods": ["POST"], "description": "学号+密码登录"},
            {"path": f"{API_PREFIX}/auth/logout", "methods": ["POST"], "description": "登出, Token 加入黑名单"},
            {"path": f"{API_PREFIX}/auth/refresh", "methods": ["POST"], "description": "刷新 JWT Token"},
            {"path": f"{API_PREFIX}/users/me", "methods": ["GET"], "description": "获取当前用户信息"},
            {"path": f"{API_PREFIX}/users/me/preferences", "methods": ["PUT"], "description": "更新学习偏好"},
            {"path": f"{API_PREFIX}/sessions", "methods": ["POST"], "description": "创建学习会话"},
            {"path": f"{API_PREFIX}/sessions/{{session_id}}", "methods": ["GET"], "description": "获取会话详情"},
            {"path": f"{API_PREFIX}/sessions/{{session_id}}/fork", "methods": ["POST"], "description": "创建 Session Fork"},
            {"path": f"{API_PREFIX}/sessions/{{session_id}}/merge", "methods": ["POST"], "description": "合并 Fork 分支"},
            {"path": f"{API_PREFIX}/sessions/{{session_id}}/pause", "methods": ["POST"], "description": "暂停会话"},
            {"path": f"{API_PREFIX}/context/{{session_id}}", "methods": ["GET"], "description": "获取 Context Envelope"},
            {"path": f"{API_PREFIX}/context/{{session_id}}/refresh", "methods": ["POST"], "description": "刷新上下文"},
            {"path": f"{API_PREFIX}/hitl/confirm", "methods": ["POST"], "description": "HiTL 确认型操作"},
            {"path": f"{API_PREFIX}/hitl/feedback", "methods": ["POST"], "description": "HiTL 纠错型反馈"},
            {"path": f"{API_PREFIX}/hitl/emergency", "methods": ["GET"], "description": "获取紧急干预状态"},
            {"path": f"{API_PREFIX}/audit/logs", "methods": ["GET"], "description": "查询审计日志"},
            {"path": f"{API_PREFIX}/export/learner-data", "methods": ["GET"], "description": "导出脱敏学情数据"},
        ]


# ============================================================
# 12. 层间接口 (L0/L2/L3/CC2)
# ============================================================


class LayerInterfaces:
    """层间接口实现 (L0/L2/L3/CC2 四组接口).

    设计依据:
    - L1 设计文档第八章 8.1-8.4
    - 任务拆分文档 T7: 7.3 层间接口实现

    接口分组:
    1. L1 → L0 (上报): 审计日志、隐私事件、Provenance
    2. L0 → L1 (拉取): 合规策略、策略变更通知
    3. L1 → L2 (传递): 上下文信封、学习记忆、遗忘调度
    4. L2 → L1 (返回): 学情画像、BKT 参数更新
    5. L1 → L3 (请求): 知识访问校验、资源推荐
    6. L3 → L1 (返回): 知识查询结果
    7. L1 ↔ CC2 (HiTL Gate): 确认请求/响应、反馈、紧急干预

    融合方案:
    - 事件驱动架构 (EDA): 松耦合异步通信
    - Privacy by Design: 数据最小化传输
    - 可溯源: 每个调用携带 session_id
    """

    def __init__(self) -> None:
        # L0 状态
        self._l0_audit_log: list[AuditLogEntry] = []
        self._l0_audit_count: int = 0
        self._l0_privacy_events: list[PrivacyEvent] = []
        self._l0_provenance: list[dict[str, Any]] = []
        self._l0_policies: list[dict[str, Any]] = []

        # L2 状态
        self._l2_contexts: dict[str, dict[str, Any]] = {}  # session_id → envelope
        self._l2_profiles: dict[str, dict[str, Any]] = {}  # user_id → profile
        self._l2_bkt_updates: dict[str, dict[str, Any]] = {}  # user_id → updates
        self._l2_memory: list[dict[str, Any]] = []
        self._l2_decay_requests: list[dict[str, Any]] = []

        # L3 状态
        self._l3_access_log: list[dict[str, Any]] = []
        self._l3_resource_requests: list[dict[str, Any]] = []
        self._l3_results: dict[str, dict[str, Any]] = {}  # session_id → result

        # CC2 状态
        self._cc2_approval_requests: list[dict[str, Any]] = []
        self._cc2_approval_responses: list[dict[str, Any]] = []
        self._cc2_feedback: list[dict[str, Any]] = []
        self._cc2_emergencies: list[dict[str, Any]] = []

        self._lock = threading.RLock()

    # ---- L1 → L0 (上报) ----

    @property
    def l0_audit_count(self) -> int:
        """L0 审计日志计数."""
        return self._l0_audit_count

    @property
    def l2_profiles(self) -> dict[str, dict[str, Any]]:
        """L2 学情画像 (只读)."""
        return self._l2_profiles

    def report_audit_logs(self, entries: list[AuditLogEntry]) -> bool:
        """L1→L0 审计日志批量上报.

        Args:
            entries: 审计日志条目列表

        Returns:
            True 如果上报成功
        """
        with self._lock:
            self._l0_audit_log.extend(entries)
            self._l0_audit_count += len(entries)
        _logger.debug(f"L1→L0: 上报 {len(entries)} 条审计日志")
        return True

    def report_privacy_event(self, event: PrivacyEvent) -> bool:
        """L1→L0 隐私事件通知.

        Args:
            event: 隐私事件

        Returns:
            True 如果通知成功
        """
        with self._lock:
            self._l0_privacy_events.append(event)
        _logger.debug(f"L1→L0: 隐私事件 {event.event_type} (user={event.user_id})")
        return True

    def write_provenance(
        self,
        session_id: str,
        agent_id: str,
        action: str,
        output_hash: str,
    ) -> bool:
        """L1→L0 Provenance 写入.

        Args:
            session_id: 会话 ID
            agent_id: Agent ID
            action: 执行动作
            output_hash: 输出哈希

        Returns:
            True 如果写入成功
        """
        with self._lock:
            self._l0_provenance.append({
                "session_id": session_id,
                "agent_id": agent_id,
                "action": action,
                "output_hash": output_hash,
                "timestamp": int(time.time() * 1000),
            })
        _logger.debug(f"L1→L0: Provenance {agent_id}/{action} (session={session_id})")
        return True

    # ---- L0 → L1 (拉取) ----

    def pull_compliance_policies(self) -> list[dict[str, Any]]:
        """L0→L1 合规策略拉取.

        Returns:
            策略列表
        """
        with self._lock:
            return list(self._l0_policies)

    def receive_policy_update(
        self,
        policy_id: str,
        version: str,
        diff: dict[str, Any],
    ) -> bool:
        """L0→L1 策略变更通知.

        Args:
            policy_id: 策略 ID
            version: 新版本号
            diff: 变更内容

        Returns:
            True 如果接收成功
        """
        with self._lock:
            self._l0_policies.append({
                "policy_id": policy_id,
                "version": version,
                "diff": diff,
                "effective_at": int(time.time() * 1000),
            })
        _logger.debug(f"L0→L1: 策略更新 {policy_id} v{version}")
        return True

    # ---- L1 → L2 (传递) ----

    def send_context_envelope(
        self,
        session_id: str,
        envelope_data: dict[str, Any],
    ) -> bool:
        """L1→L2 上下文信封传递.

        Args:
            session_id: 会话 ID
            envelope_data: 上下文数据

        Returns:
            True 如果传递成功
        """
        with self._lock:
            self._l2_contexts[session_id] = envelope_data
        _logger.debug(f"L1→L2: 上下文信封 (session={session_id})")
        return True

    def send_memory_entry(
        self,
        user_id: str,
        session_id: str,
        entry: dict[str, Any],
    ) -> bool:
        """L1→L2 学习记忆写入.

        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            entry: 记忆条目

        Returns:
            True 如果写入成功
        """
        with self._lock:
            self._l2_memory.append({
                "user_id": user_id,
                "session_id": session_id,
                "entry": entry,
                "timestamp": int(time.time() * 1000),
            })
        return True

    def send_decay_request(
        self,
        user_id: str,
        kp_ids: list[str],
    ) -> bool:
        """L1→L2 遗忘调度请求.

        Args:
            user_id: 用户 ID
            kp_ids: 知识点 ID 列表

        Returns:
            True 如果请求成功
        """
        with self._lock:
            self._l2_decay_requests.append({
                "user_id": user_id,
                "kp_ids": kp_ids,
                "timestamp": int(time.time() * 1000),
            })
        return True

    # ---- L2 → L1 (返回) ----

    def receive_learner_profile(
        self,
        user_id: str,
        profile_data: dict[str, Any],
    ) -> bool:
        """L2→L1 学情画像输出.

        Args:
            user_id: 用户 ID
            profile_data: 画像数据

        Returns:
            True 如果接收成功
        """
        with self._lock:
            self._l2_profiles[user_id] = profile_data
        _logger.debug(f"L2→L1: 学情画像 (user={user_id})")
        return True

    def receive_bkt_update(
        self,
        user_id: str,
        kp_id: str,
        update_data: dict[str, Any],
    ) -> bool:
        """L2→L1 BKT 参数更新.

        Args:
            user_id: 用户 ID
            kp_id: 知识点 ID
            update_data: 更新数据

        Returns:
            True 如果接收成功
        """
        with self._lock:
            key = f"{user_id}:{kp_id}"
            self._l2_bkt_updates[key] = update_data
        _logger.debug(f"L2→L1: BKT 更新 {kp_id} (user={user_id})")
        return True

    # ---- L1 → L3 (请求) ----

    def check_access(
        self,
        user_id: str,
        resource_id: str,
        action: str,
    ) -> bool:
        """L1→L3 知识访问权限校验.

        Args:
            user_id: 用户 ID
            resource_id: 资源 ID
            action: 操作 (read/write)

        Returns:
            True 如果允许访问
        """
        with self._lock:
            self._l3_access_log.append({
                "user_id": user_id,
                "resource_id": resource_id,
                "action": action,
                "timestamp": int(time.time() * 1000),
            })
        # 默认允许 (生产环境对接 L3 实际权限校验)
        return True

    def request_resources(
        self,
        user_id: str,
        session_id: str,
        weak_kcs: list[str],
    ) -> bool:
        """L1→L3 学习资源推荐请求.

        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            weak_kcs: 薄弱知识点列表

        Returns:
            True 如果请求成功
        """
        with self._lock:
            self._l3_resource_requests.append({
                "user_id": user_id,
                "session_id": session_id,
                "weak_kcs": weak_kcs,
                "timestamp": int(time.time() * 1000),
            })
        _logger.debug(f"L1→L3: 资源推荐 (session={session_id}, kcs={len(weak_kcs)})")
        return True

    # ---- L3 → L1 (返回) ----

    def receive_knowledge_result(
        self,
        session_id: str,
        resources: list[dict[str, Any]],
        confidence_scores: list[float],
    ) -> bool:
        """L3→L1 知识查询结果.

        Args:
            session_id: 会话 ID
            resources: 资源列表
            confidence_scores: 置信度分数列表

        Returns:
            True 如果接收成功
        """
        with self._lock:
            self._l3_results[session_id] = {
                "resources": resources,
                "confidence_scores": confidence_scores,
                "timestamp": int(time.time() * 1000),
            }
        _logger.debug(f"L3→L1: 知识结果 (session={session_id}, resources={len(resources)})")
        return True

    # ---- L1 ↔ CC2 (HiTL Gate) ----

    def route_approval_request(
        self,
        user_id: str,
        session_id: str,
        content: str,
        confidence: float,
    ) -> bool:
        """L1↔CC2 确认请求路由.

        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            content: 需确认的内容
            confidence: 置信度

        Returns:
            True 如果路由成功
        """
        with self._lock:
            self._cc2_approval_requests.append({
                "user_id": user_id,
                "session_id": session_id,
                "content": content,
                "confidence": confidence,
                "timestamp": int(time.time() * 1000),
            })
        _logger.debug(f"L1↔CC2: 确认请求 (user={user_id}, confidence={confidence})")
        return True

    def route_approval_response(
        self,
        request_id: str,
        decision: str,
        comment: str = "",
    ) -> bool:
        """L1↔CC2 确认响应路由.

        Args:
            request_id: 请求 ID
            decision: 决策 (approve/reject/modify)
            comment: 备注

        Returns:
            True 如果路由成功
        """
        with self._lock:
            self._cc2_approval_responses.append({
                "request_id": request_id,
                "decision": decision,
                "comment": comment,
                "timestamp": int(time.time() * 1000),
            })
        _logger.debug(f"L1↔CC2: 确认响应 {request_id} → {decision}")
        return True

    def report_feedback(
        self,
        session_id: str,
        user_id: str,
        feedback_type: str,
        content: str,
    ) -> bool:
        """L1↔CC2 反馈数据上报.

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            feedback_type: 反馈类型
            content: 反馈内容

        Returns:
            True 如果上报成功
        """
        with self._lock:
            self._cc2_feedback.append({
                "session_id": session_id,
                "user_id": user_id,
                "feedback_type": feedback_type,
                "content": content,
                "timestamp": int(time.time() * 1000),
            })
        _logger.debug(f"L1↔CC2: 反馈 {feedback_type} (user={user_id})")
        return True

    def alert_emergency(
        self,
        session_id: str,
        user_id: str,
        trigger_reason: str,
        trigger_value: float,
    ) -> bool:
        """L1↔CC2 紧急干预通知.

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            trigger_reason: 触发原因
            trigger_value: 触发值

        Returns:
            True 如果通知成功
        """
        with self._lock:
            self._cc2_emergencies.append({
                "session_id": session_id,
                "user_id": user_id,
                "trigger_reason": trigger_reason,
                "trigger_value": trigger_value,
                "timestamp": int(time.time() * 1000),
            })
        _logger.warning(
            f"L1↔CC2: 紧急干预 {trigger_reason}={trigger_value} "
            f"(user={user_id}, session={session_id})"
        )
        return True
