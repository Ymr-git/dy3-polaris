"""Dy3+ Polaris MCP Client 基类.

扩展 MCP SDK 的 ClientSession，提供：
- 自动重连（指数退避）
- 限流感知（自动读取 Retry-After）
- 溯源包自动附加到调用
- 连接池管理（多 Server 并发访问）
- 统一调用接口（call_tool_safe）

使用示例:
    from dy3_polaris.l6.core.client import Dy3MCPClient

    client = Dy3MCPClient(transport="stdio", command="python", args=["bkt_server.py"])
    async with client.connect() as session:
        tools = await session.list_tools()
        result = await session.call_tool_safe("bkt_compute", {"learner_id": "u001"})
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from .config import (
    L6Config,
    SSETransportConfig,
    TransportType,
    WebSocketTransportConfig,
    get_config,
)
from .exceptions import L6Error, MCPToolExecutionError, RateLimitError, TransportTimeoutError
from .models import KPA, KPAEventType, LayerTag
from .utils import snapshot_sanitize

logger = logging.getLogger(__name__)


# ============================================================
# 重连策略
# ============================================================

class ExponentialBackoff:
    """指数退避重连策略."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        max_retries: int = 5,
        jitter: bool = True,
    ) -> None:
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.jitter = jitter
        self._attempt = 0

    def reset(self) -> None:
        self._attempt = 0

    def next_delay(self) -> float | None:
        """计算下一次重连延迟.

        Returns:
            延迟秒数，None 表示重连次数已耗尽。
        """
        if self._attempt >= self.max_retries:
            return None
        delay = min(self.base_delay * (2 ** self._attempt), self.max_delay)
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        self._attempt += 1
        return delay


# ============================================================
# Dy3+ MCP Client
# ============================================================

class Dy3MCPClient:
    """Dy3+ Polaris MCP Client.

    封装 MCP SDK ClientSession，提供：
    - 三种传输方式的统一初始化
    - 自动重连
    - 限流感知
    - 溯源附加
    """

    def __init__(
        self,
        *,
        transport: TransportType | str = TransportType.STDIO,
        config: L6Config | None = None,
        # stdio 参数
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        # SSE / WebSocket 参数
        url: str | None = None,
        # 通用
        request_timeout: float | None = None,
        session_id: str | None = None,
    ) -> None:
        self._config = config or get_config()
        self._transport = TransportType(transport) if isinstance(transport, str) else transport
        self._session_id = session_id

        # 传输参数
        t_config = self._config.get_transport_config(self._transport)
        self._command = command or (self._config.stdio.command if isinstance(t_config, type(self._config.stdio)) else "python")
        self._args = args or []
        self._env = env
        self._cwd = cwd
        self._url = url
        self._timeout = request_timeout or getattr(t_config, "request_timeout", 60.0)

        # 重连
        if isinstance(t_config, (SSETransportConfig, WebSocketTransportConfig)):
            self._reconnect = ExponentialBackoff(
                base_delay=t_config.base_delay,
                max_delay=t_config.max_delay,
                max_retries=t_config.max_retries,
            )
        else:
            self._reconnect = ExponentialBackoff(max_retries=3)

        # 内部状态
        self._session: ClientSession | None = None
        self._read_stream = None
        self._write_stream = None
        self._connected = False

        # 溯源
        self._kpa_chain: list[KPA] = []

    # ---- 属性 ----

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def session(self) -> ClientSession | None:
        return self._session

    # ---- 连接管理 ----

    async def connect(self) -> Dy3MCPClient:
        """建立连接并初始化会话.

        支持作为异步上下文管理器使用:
            async with client.connect() as session:
                ...
        """
        if self._transport == TransportType.STDIO:
            return await self._connect_stdio()
        elif self._transport == TransportType.SSE:
            return await self._connect_sse()
        else:
            return await self._connect_websocket()

    async def _connect_stdio(self) -> Dy3MCPClient:
        """通过 stdio 建立连接."""
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
            cwd=self._cwd,
        )
        read_stream, write_stream = await stdio_client(server_params).__aenter__()
        self._read_stream = read_stream
        self._write_stream = write_stream

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()
        self._connected = True
        logger.info(f"[Client:{self._session_id}] Connected via stdio to {self._command}")
        return self

    async def _connect_sse(self) -> Dy3MCPClient:
        """通过 SSE 建立连接（带自动重连）."""
        from mcp.client.sse import sse_client

        url = self._url or self._config.sse.url
        attempt = 0
        last_error: Exception | None = None

        while True:
            try:
                read_stream, write_stream = await sse_client(url).__aenter__()
                self._read_stream = read_stream
                self._write_stream = write_stream

                self._session = ClientSession(read_stream, write_stream)
                await self._session.__aenter__()
                await self._session.initialize()
                self._connected = True
                logger.info(f"[Client:{self._session_id}] Connected via SSE to {url}")
                self._reconnect.reset()
                return self
            except Exception as e:
                last_error = e
                delay = self._reconnect.next_delay()
                if delay is None:
                    break
                logger.warning(f"[Client:{self._session_id}] SSE connect failed, retry in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)

        raise L6Error(
            "TRANSPORT_RECONNECT_EXHAUSTED",
            f"SSE connection to {url} failed after {self._reconnect.max_retries} retries",
            {"url": url, "last_error": str(last_error)},
        )

    async def _connect_websocket(self) -> Dy3MCPClient:
        """通过 WebSocket (streamable-http) 建立连接（带自动重连）."""
        url = self._url or self._config.websocket.url
        attempt = 0
        last_error: Exception | None = None

        # 使用 httpx 作为 streamable-http 客户端
        import httpx

        while True:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as http_client:
                    # streamable-http transport
                    from mcp.client.streamable_http import streamablehttp_client
                    read_stream, write_stream, _ = await streamablehttp_client(url, http_client).__aenter__()
                    self._read_stream = read_stream
                    self._write_stream = write_stream

                    self._session = ClientSession(read_stream, write_stream)
                    await self._session.__aenter__()
                    await self._session.initialize()
                    self._connected = True
                    logger.info(f"[Client:{self._session_id}] Connected via WebSocket/HTTP to {url}")
                    self._reconnect.reset()
                    return self
            except Exception as e:
                last_error = e
                delay = self._reconnect.next_delay()
                if delay is None:
                    break
                logger.warning(f"[Client:{self._session_id}] WebSocket connect failed, retry in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)

        raise L6Error(
            "TRANSPORT_RECONNECT_EXHAUSTED",
            f"WebSocket connection to {url} failed after {self._reconnect.max_retries} retries",
            {"url": url, "last_error": str(last_error)},
        )

    async def disconnect(self) -> None:
        """断开连接."""
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error closing session")
        self._session = None
        self._connected = False
        logger.info(f"[Client:{self._session_id}] Disconnected")

    # ---- 上下文管理器 ----

    async def __aenter__(self) -> Dy3MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()

    # ---- 安全调用（带限流感知和溯源） ----

    async def call_tool_safe(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
        attach_provenance: bool = True,
    ) -> Any:
        """安全调用工具.

        相比原始 call_tool，增加：
        1. 超时控制
        2. 限流感知（捕获 RateLimitError 并提供 retry_after）
        3. 溯源自动附加

        Args:
            tool_name: 工具名
            arguments: 参数字典
            timeout: 超时秒数，None 使用默认
            attach_provenance: 是否附加 KPA 溯源包

        Returns:
            工具执行结果

        Raises:
            RateLimitError: 限流时抛出（含 retry_after 属性）
            TransportTimeoutError: 超时时抛出
        """
        if not self._session or not self._connected:
            raise L6Error("TRANSPORT_CLOSED", "Client not connected")

        effective_timeout = timeout or self._timeout

        # 创建调用前 KPA
        call_kpa: KPA | None = None
        if attach_provenance and self._config.provenance_auto_attach:
            call_kpa = KPA(
                prev_hash=self._kpa_chain[-1].compute_hash() if self._kpa_chain else None,
                event_type=KPAEventType.TOOL_INVOKED,
                actor=f"client:{self._session_id or 'default'}",
                layer=LayerTag.L6_PROTOCOL,
                input_snapshot=snapshot_sanitize(arguments),
            )
            self._kpa_chain.append(call_kpa)

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments=arguments),
                timeout=effective_timeout,
            )

            is_error = getattr(result, "isError", False)

            # 更新 KPA 输出
            if call_kpa and hasattr(result, "content"):
                call_kpa.output_snapshot = {
                    "is_error": is_error,
                    "content_types": [c.type for c in result.content] if result.content else [],
                }

            # 处理 MCP isError=true 的响应
            if is_error:
                error_text = ""
                if hasattr(result, "content") and result.content:
                    error_text = "\n".join(
                        getattr(c, "text", str(c))
                        for c in result.content
                        if getattr(c, "type", None) == "text"
                    )
                raise MCPToolExecutionError(
                    tool_name=tool_name,
                    detail=error_text or f"Tool {tool_name} returned isError=true",
                    is_error=True,
                    context={"arguments": arguments},
                )

            return result

        except asyncio.TimeoutError:
            raise TransportTimeoutError(
                transport_type=self._transport.value,
                timeout_seconds=effective_timeout,
                detail=f"Tool {tool_name} timed out after {effective_timeout}s",
                context={"tool_name": tool_name},
            )
        except RateLimitError:
            raise
        except MCPToolExecutionError:
            raise
        except Exception as e:
            if call_kpa:
                call_kpa.output_snapshot = {"error": str(e)}
            raise

    # ---- 溯源 ----

    @property
    def kpa_chain(self) -> list[KPA]:
        return list(self._kpa_chain)

    def reset_kpa_chain(self) -> None:
        self._kpa_chain.clear()

    # ---- 便捷方法 ----

    async def list_tools(self) -> list[Any]:
        """列出远端工具."""
        if not self._session:
            raise L6Error("TRANSPORT_CLOSED", "Client not connected")
        result = await self._session.list_tools()
        return result.tools

    async def list_resources(self) -> list[Any]:
        """列出远端资源."""
        if not self._session:
            raise L6Error("TRANSPORT_CLOSED", "Client not connected")
        result = await self._session.list_resources()
        return result.resources

    async def read_resource(self, uri: str) -> Any:
        """读取远端资源."""
        if not self._session:
            raise L6Error("TRANSPORT_CLOSED", "Client not connected")
        return await self._session.read_resource(uri)


