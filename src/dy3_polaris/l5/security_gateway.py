"""统一安全网关 — 覆盖各层写端点 (复用 L1 JWT).

机制:
- 纯 ASGI 中间件, 注册于 unified_app (挂载各层 Mount 之外)
- 写方法 (POST/PUT/DELETE) 且不在白名单 → 校验 Authorization: Bearer <token>
  (复用 L1 JWTManager.verify_token)
- GET/OPTIONS/HEAD 与白名单前缀放行
- 认证失败 → 401 JSON {code:-32201, message, trace_id} (trace 头由 TraceIDMiddleware 补)

公开白名单 (产品语义):
- /l1/api/v1/auth/*  认证通道
- /health /api/*     平台统一入口 (内部会话/反馈聚合)
- /static 静态资源
- /l2/practice/* /l2/event/collect /l2/irt/*  学生端公开学习操作
- /l3/retrieve/*     POST 语义只读检索
- /l6/jsonrpc        L6 协议端点 (工具调用内部协议)
"""
from __future__ import annotations

import json
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dy3_polaris.l5.tracing import get_trace_id

#: 公开白名单前缀 (路径前缀匹配)
PUBLIC_PREFIXES: tuple[str, ...] = (
    "/l1/api/v1/auth/",
    "/health",
    "/api/",
    "/static",
    "/l2/practice/",
    "/l2/event/collect",
    "/l2/irt/",
    "/l3/retrieve/",
    "/l6/jsonrpc",
)

#: 只读方法 (放行)
READ_METHODS: tuple[str, ...] = ("GET", "OPTIONS", "HEAD")


class SecurityGatewayMiddleware:
    """统一安全网关: 写端点 Bearer token 校验 (复用 L1 JWTManager)."""

    def __init__(self, app: ASGIApp, jwt_manager: Any | None = None) -> None:
        self.app = app
        self.jwt_manager = jwt_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = (scope.get("method") or "GET").upper()
        path = scope.get("path") or "/"
        if method in READ_METHODS or self._is_public(path):
            await self.app(scope, receive, send)
            return

        # 写方法: 校验 Bearer token
        ok_auth = False
        if self.jwt_manager is not None:
            token = self._extract_bearer(scope.get("headers") or [])
            if token:
                try:
                    self.jwt_manager.verify_token(token)
                    ok_auth = True
                except Exception:  # noqa: BLE001  (过期/篡改/黑名单)
                    ok_auth = False
        if not ok_auth:
            await self._deny(send)
            return
        await self.app(scope, receive, send)

    # ------------------------------------------------------------------
    @staticmethod
    def _is_public(path: str) -> bool:
        return any(path.startswith(p) for p in PUBLIC_PREFIXES)

    @staticmethod
    def _extract_bearer(headers: list[tuple[bytes, bytes]]) -> str | None:
        for name, value in headers:
            if name.lower() == b"authorization":
                raw = value.decode("utf-8", "replace")
                if raw.lower().startswith("bearer "):
                    return raw[7:].strip()
        return None

    @staticmethod
    async def _deny(send: Send) -> None:
        body = json.dumps({
            "code": -32201,
            "message": "Authentication required",
            "trace_id": get_trace_id() or "",
        }, ensure_ascii=False).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        })


def gateway_middleware(jwt_manager: Any | None) -> Any:
    """构造 SecurityGatewayMiddleware 的 Starlette Middleware 包装."""
    from starlette.middleware import Middleware

    return Middleware(SecurityGatewayMiddleware, jwt_manager=jwt_manager)


__all__ = ["SecurityGatewayMiddleware", "gateway_middleware"]
