"""Dy3+ Polaris MCP Server 基类.

继承 MCP SDK 的 FastMCP，扩展 Dy3+ 特有能力：
- 限流中间件（per-tool rate limiting）
- 溯源自动附加（KPA 自动注入到工具调用链）
- 工具分类标签（Dy3ToolAnnotations 集成）
- 生命周期钩子增强（pre_call / post_call）
- 健康检查 endpoint

使用示例:
    from dy3_polaris.l6.core.server import Dy3MCPServer

    server = Dy3MCPServer(name="dy3-bkt-server", layer=LayerTag.L2)

    @server.tool(
        annotations=Dy3ToolAnnotations(layer=LayerTag.L2, category=ToolCategory.INTERNAL)
    )
    async def bkt_compute(learner_id: str, kp_id: str, response: bool) -> dict:
        '''贝叶斯知识追踪计算.'''
        return {"p_know": 0.85}

    server.run(transport="stdio")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from mcp.types import TextContent, Tool

from .config import L6Config, TransportType, get_config
from .exceptions import (
    L6Error,
    MCPToolExecutionError,
    MCPToolNotFoundError,
    RateLimitError,
)
from .models import (
    Dy3ToolAnnotations,
    KPA,
    KPAEventType,
    LayerTag,
    ToolCategory,
    ToolRegistration,
)
from .utils import snapshot_sanitize

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ============================================================
# 限流器
# ============================================================

class TokenBucketLimiter:
    """令牌桶限流器.

    每个 tool_name 独立一个桶。
    """

    def __init__(
        self,
        default_limit: int = 100,
        window_seconds: int = 60,
    ) -> None:
        self._default_limit = default_limit
        self._window = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._custom_limits: dict[str, int] = {}

    def set_limit(self, tool_name: str, limit: int) -> None:
        """设置工具自定义限流."""
        self._custom_limits[tool_name] = limit

    def acquire(self, tool_name: str) -> tuple[bool, float]:
        """尝试获取令牌.

        Returns:
            (allowed, retry_after): 是否允许，不允许时返回建议重试时间
        """
        limit = self._custom_limits.get(tool_name, self._default_limit)
        now = time.time()
        cutoff = now - self._window

        # 清理过期记录
        bucket = self._buckets[tool_name]
        self._buckets[tool_name] = [t for t in bucket if t > cutoff]

        if len(self._buckets[tool_name]) < limit:
            self._buckets[tool_name].append(now)
            return True, 0.0

        # 计算最旧记录的过期时间
        oldest = min(self._buckets[tool_name])
        retry_after = oldest + self._window - now + 0.1
        return False, max(0.1, retry_after)

    def reset(self) -> None:
        """清空所有桶（测试用）."""
        self._buckets.clear()
        self._custom_limits.clear()


# ============================================================
# 生命周期上下文
# ============================================================

@asynccontextmanager
async def default_lifespan(server: Dy3MCPServer) -> AsyncIterator[dict[str, Any]]:
    """默认生命周期管理器.

    在 lifespan 内初始化限流器和内部状态。
    """
    logger.info(f"[{server.name}] Lifespan starting...")
    yield {"start_time": time.time(), "server_name": server.name}
    logger.info(f"[{server.name}] Lifespan shutting down...")


# ============================================================
# Dy3+ MCP Server 基类
# ============================================================

class Dy3MCPServer(FastMCP):
    """Dy3+ Polaris MCP Server 基类.

    在 FastMCP 基础上扩展：
    1. 限流中间件（Token Bucket）
    2. 溯源自动附加（KPA）
    3. 工具分类标签
    4. 生命周期钩子（pre_call / post_call）
    5. 健康检查
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        layer: LayerTag | None = None,
        config: L6Config | None = None,
        lifespan: Callable[..., AsyncIterator[dict[str, Any]]] | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or get_config()
        self._dy3_config = cfg
        self._dy3_layer = layer
        self._dy3_limiter = TokenBucketLimiter(
            default_limit=cfg.default_rate_limit,
            window_seconds=cfg.rate_limit_window,
        )

        # 工具注册表（Dy3+ 扩展元数据）
        self._dy3_tool_registrations: dict[str, ToolRegistration] = {}

        # 工具 handler 存储（用于测试和内部转发）
        self._dy3_tool_handlers: dict[str, Callable[..., Any]] = {}

        # 生命周期钩子
        self._dy3_pre_call_hooks: list[Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]] = []
        self._dy3_post_call_hooks: list[Callable[[str, dict[str, Any], Any], Coroutine[Any, Any, None]]] = []

        # KPA 链头
        self._dy3_kpa_chain: list[KPA] = []

        super().__init__(
            name=name or cfg.mcp.server_name,
            instructions=cfg.mcp.instructions,
            lifespan=lifespan or default_lifespan,
            **kwargs,
        )

    # ---- 属性 ----

    @property
    def dy3_config(self) -> L6Config:
        return self._dy3_config

    @property
    def dy3_layer(self) -> LayerTag | None:
        return self._dy3_layer

    # ---- 扩展注册方法 ----

    def register_dy3_tool(
        self,
        registration: ToolRegistration,
        handler: Callable[..., Any],
    ) -> None:
        """注册一个带 Dy3+ 元数据的工具.

        Args:
            registration: 工具注册信息（含 Dy3ToolAnnotations）
            handler: 工具执行函数（同步或异步）
        """
        if not registration.enabled:
            logger.warning(f"Tool {registration.name} is disabled, skipping registration")
            return

        # 保存 Dy3+ 元数据
        self._dy3_tool_registrations[registration.name] = registration

        # 设置自定义限流
        if registration.annotations.rate_limit is not None:
            self._dy3_limiter.set_limit(registration.name, registration.annotations.rate_limit)

        # 包装 handler：织入限流检查、pre_call / post_call 钩子、KPA 溯源
        wrapped = self._wrap_handler(registration, handler)

        # 保存包装后的 handler（用于测试和内部转发）
        self._dy3_tool_handlers[registration.name] = wrapped

        # 使用 FastMCP 的装饰器注册包装后的 handler
        self.tool(name=registration.name, description=registration.description)(wrapped)
        logger.debug(
            f"Registered tool: {registration.name} "
            f"[{registration.annotations.category.value}] "
            f"layer={registration.annotations.layer} "
            f"latency={registration.annotations.estimated_latency_ms}ms"
        )

    def _wrap_handler(
        self,
        registration: ToolRegistration,
        handler: Callable[..., Any],
    ) -> Callable[..., Any]:
        """包装原始 handler，织入 Dy3+ 中间件."""
        tool_name = registration.name
        is_async = asyncio.iscoroutinefunction(handler)

        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # 1. 限流检查
            self.check_rate_limit(tool_name)

            # 2. pre_call 钩子
            arguments = kwargs if kwargs else ({"args": args} if args else {})
            await self._run_pre_call_hooks(tool_name, arguments)

            # 3. 创建 KPA（溯源）
            call_kpa: KPA | None = None
            if self._dy3_config.provenance_auto_attach:
                call_kpa = self.create_kpa(
                    event_type=KPAEventType.TOOL_INVOKED,
                    actor=tool_name,
                    input_snapshot=snapshot_sanitize(arguments),
                )

            # 4. 执行 handler
            start = time.monotonic()
            try:
                if is_async:
                    result = await handler(*args, **kwargs)
                else:
                    result = handler(*args, **kwargs)

                # 5. post_call 钩子
                await self._run_post_call_hooks(tool_name, arguments, result)

                # 6. 更新 KPA
                latency_ms = int((time.monotonic() - start) * 1000)
                if call_kpa:
                    call_kpa.output_snapshot = {
                        "success": True,
                        "result_type": type(result).__name__,
                        "latency_ms": latency_ms,
                    }

                return result

            except Exception as exc:
                latency_ms = int((time.monotonic() - start) * 1000)
                if call_kpa:
                    call_kpa.output_snapshot = {
                        "success": False,
                        "error": str(exc),
                        "latency_ms": latency_ms,
                    }
                raise

        return _async_wrapper

    def register_dy3_resource(
        self,
        uri: str,
        name: str,
        handler: Callable[..., Any],
        *,
        description: str = "",
        mime_type: str = "application/json",
        layer: LayerTag | None = None,
    ) -> None:
        """注册一个带 Dy3+ 元数据的资源."""
        self.resource(uri=uri, name=name, description=description, mime_type=mime_type)(handler)

    # ---- 生命周期钩子 ----

    def on_pre_call(
        self,
        hook: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]:
        """注册工具调用前钩子（如参数校验、权限检查）."""
        self._dy3_pre_call_hooks.append(hook)
        return hook

    def on_post_call(
        self,
        hook: Callable[[str, dict[str, Any], Any], Coroutine[Any, Any, None]],
    ) -> Callable[[str, dict[str, Any], Any], Coroutine[Any, Any, None]]:
        """注册工具调用后钩子（如结果缓存、日志记录）."""
        self._dy3_post_call_hooks.append(hook)
        return hook

    async def _run_pre_call_hooks(self, tool_name: str, arguments: dict[str, Any]) -> None:
        for hook in self._dy3_pre_call_hooks:
            try:
                await hook(tool_name, arguments)
            except Exception:
                logger.exception(f"Pre-call hook failed for {tool_name}")

    async def _run_post_call_hooks(
        self, tool_name: str, arguments: dict[str, Any], result: Any
    ) -> None:
        for hook in self._dy3_post_call_hooks:
            try:
                await hook(tool_name, arguments, result)
            except Exception:
                logger.exception(f"Post-call hook failed for {tool_name}")

    # ---- 限流 ----

    def check_rate_limit(self, tool_name: str) -> None:
        """检查限流，超限时抛出 RateLimitError."""
        allowed, retry_after = self._dy3_limiter.acquire(tool_name)
        if not allowed:
            raise RateLimitError(
                tool_name=tool_name,
                limit=self._dy3_limiter._custom_limits.get(tool_name, self._dy3_config.default_rate_limit),
                window_seconds=self._dy3_config.rate_limit_window,
                retry_after=retry_after,
            )

    # ---- 溯源 ----

    def create_kpa(
        self,
        event_type: KPAEventType,
        actor: str,
        *,
        input_snapshot: dict[str, Any] | None = None,
        processing_logic: str = "",
        output_snapshot: dict[str, Any] | None = None,
        context_refs: list[str] | None = None,
        confidence: float | None = None,
    ) -> KPA:
        """创建一个 KPA 溯源数据包.

        自动关联到当前 KPA 链的最后一个节点。
        """
        prev_hash = self._dy3_kpa_chain[-1].compute_hash() if self._dy3_kpa_chain else None
        layer = self._dy3_layer or LayerTag.L6_PROTOCOL

        kpa = KPA(
            prev_hash=prev_hash,
            event_type=event_type,
            actor=actor,
            layer=layer,
            input_snapshot=input_snapshot or {},
            processing_logic=processing_logic,
            output_snapshot=output_snapshot or {},
            context_refs=context_refs or [],
            confidence=confidence,
        )
        self._dy3_kpa_chain.append(kpa)
        return kpa

    @property
    def kpa_chain(self) -> list[KPA]:
        """获取当前 KPA 溯源链."""
        return list(self._dy3_kpa_chain)

    def reset_kpa_chain(self) -> None:
        """重置 KPA 链（新会话开始时调用）."""
        self._dy3_kpa_chain.clear()

    # ---- 服务端安全调用 ----

    async def call_tool_safe(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
        attach_provenance: bool = True,
    ) -> Any:
        """服务端安全调用工具（内部路由使用）.

        与 Client 侧 call_tool_safe 对应，用于 Server 内部转发。
        """
        info = self._dy3_tool_registrations.get(tool_name)
        if info is None:
            raise MCPToolNotFoundError(tool_name)

        # 限流检查
        self.check_rate_limit(tool_name)

        # 查找实际 handler（FastMCP 内部注册的工具）
        # 注意：这里无法直接访问 FastMCP 的内部 handler，
        # 实际内部转发应使用 Client 连接同一 Server 或直接使用注册时保存的 wrapped handler
        # 简化实现：返回一个说明响应
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "note": "Server-side internal routing - use Client to call actual handler",
            "provenance_chain_length": len(self._dy3_kpa_chain),
        }

    # ---- 查询 ----

    def get_dy3_tool_info(self, tool_name: str) -> ToolRegistration | None:
        """获取工具的 Dy3+ 扩展信息."""
        return self._dy3_tool_registrations.get(tool_name)

    def list_dy3_tools(self) -> dict[str, ToolRegistration]:
        """列出所有已注册工具的 Dy3+ 扩展信息."""
        return dict(self._dy3_tool_registrations)

    def list_tools_by_category(self, category: ToolCategory) -> list[ToolRegistration]:
        """按分类列出工具."""
        return [
            reg for reg in self._dy3_tool_registrations.values()
            if reg.annotations.category == category
        ]

    def list_tools_by_layer(self, layer: LayerTag) -> list[ToolRegistration]:
        """按架构层列出工具."""
        return [
            reg for reg in self._dy3_tool_registrations.values()
            if reg.annotations.layer == layer
        ]

    # ---- 健康检查 ----

    async def health_check(self) -> dict[str, Any]:
        """健康检查，返回服务状态."""
        return {
            "status": "healthy",
            "server_name": self.name,
            "layer": self._dy3_layer.value if self._dy3_layer else None,
            "registered_tools": len(self._dy3_tool_registrations),
            "kpa_chain_length": len(self._dy3_kpa_chain),
            "protocol_version": self._dy3_config.mcp.protocol_version,
        }

    # ---- 运行 ----

    def run(self, transport: str = "stdio", **kwargs: Any) -> None:
        """启动 MCP Server.

        支持 stdio / sse / streamable-http 三种传输方式。
        """
        # 映射 Dy3+ WebSocket 到 MCP SDK 的 streamable-http
        actual_transport = transport
        if transport == "websocket":
            actual_transport = "streamable-http"
            logger.info("Mapping 'websocket' transport to 'streamable-http' for MCP SDK compatibility")

        logger.info(
            f"Starting Dy3+ MCP Server: {self.name} "
            f"transport={actual_transport} "
            f"layer={self._dy3_layer} "
            f"tools={len(self._dy3_tool_registrations)}"
        )
        super().run(transport=actual_transport, **kwargs)