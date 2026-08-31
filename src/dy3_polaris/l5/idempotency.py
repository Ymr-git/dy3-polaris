"""HTTP 幂等键中间件 — Stripe Idempotency-Key 模式 (请求级).

机制:
- 写请求 (POST/PUT) 携带 ``X-Idempotency-Key`` 头 → 以 ``method:path:key`` 为键
- 首次请求: 执行并缓存完整响应 (status/headers/body), TTL 24h
- 重复请求 (同键同路径): 直接返回缓存响应, 不重复执行副作用
- 幂等键仅做"同键同路径去重", 不校验业务体 (调用方自行保证同键语义一致)

仅内存实现 (进程内 TTL 缓存), 供跨层写用例复用.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: 幂等键请求头
IDEMPOTENCY_HEADER: bytes = b"x-idempotency-key"
#: 缓存 TTL (秒)
IDEMPOTENCY_TTL: float = 24 * 3600.0

_READ_METHODS: tuple[str, ...] = ("GET", "OPTIONS", "HEAD")


class IdempotencyMiddleware:
    """请求级幂等键中间件."""

    def __init__(self, app: ASGIApp, ttl: float = IDEMPOTENCY_TTL) -> None:
        self.app = app
        self.ttl = ttl
        self._lock = threading.RLock()
        #: key -> (expire_at, status, headers, body)
        self._cache: dict[str, tuple[float, int, list[tuple[bytes, bytes]], bytes]] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = (scope.get("method") or "GET").upper()
        if method in _READ_METHODS:
            await self.app(scope, receive, send)
            return

        key = self._extract_key(scope)
        if key is None:
            await self.app(scope, receive, send)
            return

        cached = self._get_cached(key)
        if cached is not None:
            expire, status, headers, body = cached
            await self._send_cached(send, status, headers, body)
            return

        # 首次: 缓冲完整响应
        buf: list[bytes] = []
        result: dict[str, Any] = {}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                result["status"] = message.get("status", 200)
                result["headers"] = [
                    (n, v)
                    for n, v in message.get("headers", [])
                    if n.lower() != b"content-length"
                ]
                await send(message)
            elif message["type"] == "http.response.body":
                buf.append(message.get("body") or b"")
                if not message.get("more_body", False):
                    result["body"] = b"".join(buf)
                    await send(message)

        await self.app(scope, receive, send_wrapper)
        # 幂等键只缓存"已完成"响应 (异常由上层 500 兜底处理)
        if "status" in result:
            self._store(key, result)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_key(scope: Scope) -> str | None:
        path = scope.get("path") or "/"
        for name, value in scope.get("headers") or []:
            if name.lower() == IDEMPOTENCY_HEADER:
                raw = value.decode("utf-8", "replace").strip()
                if raw:
                    return f"{(scope.get('method') or 'POST')}:{path}:{raw}"
        return None

    def _get_cached(
        self, key: str
    ) -> tuple[float, int, list[tuple[bytes, bytes]], bytes] | None:
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            expire, status, headers, body = item
            if time.time() > expire:
                self._cache.pop(key, None)
                return None
            return item

    def _store(self, key: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = (
                time.time() + self.ttl,
                int(result.get("status", 200)),
                list(result.get("headers") or []),
                bytes(result.get("body") or b""),
            )

    @staticmethod
    async def _send_cached(
        send: Send,
        status: int,
        headers: list[tuple[bytes, bytes]],
        body: bytes,
    ) -> None:
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": headers + [(b"content-length", str(len(body)).encode())],
        })
        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        })


def idempotency_middleware() -> Any:
    """构造 IdempotencyMiddleware 的 Starlette Middleware 包装."""
    from starlette.middleware import Middleware

    return Middleware(IdempotencyMiddleware)


__all__ = ["IDEMPOTENCY_HEADER", "IdempotencyMiddleware", "idempotency_middleware"]
