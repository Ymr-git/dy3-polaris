"""请求级 trace_id — contextvars 注入 (不依赖外部追踪库).

设计:
- 请求入口 (unified_app) 经 TraceIDMiddleware 生成/透传 trace_id
- 写入 contextvars, 全程同请求可见 (L1~L7 任意代码可 get_trace_id())
- 响应头 ``X-Trace-Id`` 回写; 错误响应体 (code != 0) 自动回填 trace_id
- 前缀 ``tr-`` (统一命名空间, 见 shared/ids.py); 透传请求头 X-Trace-Id 时保留原值

使用:
    from dy3_polaris.l5.tracing import get_trace_id
    tid = get_trace_id()   # 未在请求上下文时为 ""
"""
from __future__ import annotations

import contextvars
import json
import uuid
from typing import Any

from starlette.middleware import Middleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dy3_polaris.shared.ids import new_id

_TRACE_ID_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dy3_trace_id", default=""
)

#: 透传/回写请求头 (小写字节形式, 供 ASGI header 匹配)
TRACE_HEADER: bytes = b"x-trace-id"


def new_trace_id() -> str:
    """生成请求级 trace_id (tr- 前缀, 统一命名空间)."""
    return new_id("tr")


def get_trace_id() -> str:
    """读取当前请求的 trace_id (无请求上下文时返回空串)."""
    return _TRACE_ID_VAR.get()


def set_trace_id(tid: str) -> contextvars.Token[str]:
    """注入 trace_id (返回 token, 供 finally reset)."""
    return _TRACE_ID_VAR.set(tid)


def reset_trace_id(token: contextvars.Token[str]) -> None:
    """恢复 trace_id 上下文. 仅限当前 contextvars 上下文有效."""
    _TRACE_ID_VAR.reset(token)


class TraceIDMiddleware:
    """请求级 trace 中间件 (纯 ASGI).

    职责:
    1. 入口生成/透传 trace_id (X-Trace-Id 请求头优先)
    2. contextvars 注入 (同请求任意代码可读)
    3. 响应头 X-Trace-Id 回写
    4. 错误响应体 (code != 0 的 JSON) 回填 trace_id (超集, 不覆盖已有键)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 1. 透传或生成
        raw = None
        for name, value in scope.get("headers") or []:
            if name.lower() == TRACE_HEADER:
                raw = value
                break
        tid = raw.decode("utf-8", "replace") if raw else new_trace_id()

        # 2. 注入 contextvars
        token = set_trace_id(tid)
        body_buf: list[bytes] = []
        try:

            async def send_wrapper(message: Message) -> None:
                if message["type"] == "http.response.start":
                    # 剥离 content-length: body 可能被回填 trace_id 改变长度,
                    # 交由服务器以 chunked/连接定界发送, 避免长度不一致截断
                    headers = [
                        (n, v)
                        for n, v in message.get("headers", [])
                        if n.lower() not in (b"content-length", TRACE_HEADER)
                    ]
                    headers.append((TRACE_HEADER, tid.encode()))
                    message["headers"] = headers
                    await send(message)
                elif message["type"] == "http.response.body":
                    body_buf.append(message.get("body") or b"")
                    if not message.get("more_body", False):
                        full = b"".join(body_buf)
                        # 4. 错误体回填 (仅 JSON 错误 dict, 不碰成功响应)
                        full = _backfill_error_body(full, tid)
                        await send({
                            "type": "http.response.body",
                            "body": full,
                            "more_body": False,
                        })

            await self.app(scope, receive, send_wrapper)
        finally:
            reset_trace_id(token)


def _backfill_error_body(body: bytes, tid: str) -> bytes:
    """错误响应体回填 trace_id: {code!=0} 时 setdefault('trace_id', tid).

    超集操作: 不删除/修改任何现有键, 兼容现有测试断言.
    """
    try:
        data: Any = json.loads(body)
    except (ValueError, TypeError):
        return body
    if isinstance(data, dict) and data.get("code", 0) != 0:
        data.setdefault("trace_id", tid)
        return json.dumps(data, ensure_ascii=False).encode("utf-8")
    return body


# 便捷: 供 unified_app 注册 (与 CORSMiddleware 同格式)
def trace_middleware() -> Middleware:
    """构造 TraceIDMiddleware 的 Starlette Middleware 包装."""
    return Middleware(TraceIDMiddleware)


__all__ = [
    "TraceIDMiddleware",
    "TRACE_HEADER",
    "get_trace_id",
    "new_trace_id",
    "reset_trace_id",
    "set_trace_id",
    "trace_middleware",
]
