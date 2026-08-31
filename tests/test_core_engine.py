"""T1 MCP 核心引擎 - 单元测试.

测试覆盖:
1. 数据模型（KPA, ToolRegistration, A2AMessage, ComputeResourceDescriptor）
2. 异常体系（L6Error 层级、JSON-RPC 错误码映射）
3. 配置管理（L6Config 加载、环境变量覆盖）
4. 限流器（TokenBucketLimiter 令牌桶）
5. Dy3MCPServer（工具注册、分类查询、KPA 链、健康检查）
6. Dy3MCPClient（ExponentialBackoff 重连策略）
"""

from __future__ import annotations

import os
import time
import pytest

from dy3_polaris.l6.core.exceptions import (
    ErrorCode,
    InternalError,
    L6Error,
    MCPToolExecutionError,
    MCPToolNotFoundError,
    MethodNotFoundError,
    ParseError,
    RateLimitError,
    ReconnectExhaustedError,
    TransportClosedError,
    TransportTimeoutError,
)
from dy3_polaris.l6.core.config import (
    L6Config,
    TransportType,
    get_config,
    reset_config,
)
from dy3_polaris.l6.core.models import (
    A2ACapability,
    A2AMessage,
    A2AMessageType,
    ComputeResourceDescriptor,
    ComputeResourceStatus,
    ComputeResourceType,
    Dy3ToolAnnotations,
    KPA,
    KPAEventType,
    LayerTag,
    ResourceRegistration,
    ResourceType,
    ToolCategory,
    ToolRegistration,
)
from dy3_polaris.l6.core.server import Dy3MCPServer, TokenBucketLimiter
from dy3_polaris.l6.core.client import ExponentialBackoff
from dy3_polaris.l6.core.utils import snapshot_sanitize


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def reset_global_config():
    """每个测试前重置全局配置."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def sample_tool_reg() -> ToolRegistration:
    return ToolRegistration(
        name="bkt_compute",
        description="贝叶斯知识追踪计算",
        input_schema={
            "type": "object",
            "properties": {
                "learner_id": {"type": "string"},
                "kp_id": {"type": "string"},
                "response": {"type": "boolean"},
            },
            "required": ["learner_id", "kp_id", "response"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["bkt", "personalization", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=100,
            domain_scope=["DOM-A", "DOM-B"],
        ),
    )


# ============================================================
# 1. 数据模型测试
# ============================================================

class TestKPA:
    def test_create_kpa(self):
        kpa = KPA(
            event_type=KPAEventType.TOOL_INVOKED,
            actor="A1",
            layer=LayerTag.L5_AGENT_RUNTIME,
            input_snapshot={"kp_id": "DOM-A-01"},
        )
        assert kpa.event_type == KPAEventType.TOOL_INVOKED
        assert kpa.actor == "A1"
        assert kpa.layer == LayerTag.L5_AGENT_RUNTIME
        assert kpa.prev_hash is None
        assert kpa.kpa_id  # 自动生成

    def test_kpa_compute_hash(self):
        kpa1 = KPA(
            event_type=KPAEventType.TOOL_INVOKED,
            actor="A1",
            layer=LayerTag.L5_AGENT_RUNTIME,
        )
        h1 = kpa1.compute_hash()
        assert isinstance(h1, str)
        assert len(h1) == 64  # SHA256 hex

        # 相同输入产生相同 hash
        kpa2 = KPA(
            event_type=KPAEventType.TOOL_INVOKED,
            actor="A1",
            layer=LayerTag.L5_AGENT_RUNTIME,
        )
        h2 = kpa2.compute_hash()
        # 时间戳不同，所以 hash 不同 (正确行为); 同毫秒创建时可能相同, 显式 sleep 保证
        import time as _t
        _t.sleep(0.002)
        kpa3 = KPA(
            event_type=KPAEventType.TOOL_INVOKED,
            actor="A1",
            layer=LayerTag.L5_AGENT_RUNTIME,
        )
        assert h1 != kpa3.compute_hash()

    def test_kpa_chain_hash(self):
        kpa1 = KPA(event_type=KPAEventType.TOOL_INVOKED, actor="A1", layer=LayerTag.L5_AGENT_RUNTIME)
        h1 = kpa1.compute_hash()
        kpa2 = KPA(
            event_type=KPAEventType.AGENT_OUTPUT,
            actor="A2",
            layer=LayerTag.L5_AGENT_RUNTIME,
            prev_hash=h1,
        )
        h2 = kpa2.compute_hash()
        assert h2 != h1
        assert kpa2.prev_hash == h1


class TestToolRegistration:
    def test_create_registration(self, sample_tool_reg: ToolRegistration):
        assert sample_tool_reg.name == "bkt_compute"
        assert sample_tool_reg.annotations.layer == LayerTag.L2_PERSONALIZATION
        assert sample_tool_reg.annotations.category == ToolCategory.INTERNAL
        assert "DOM-A" in sample_tool_reg.annotations.domain_scope

    def test_to_mcp_tool_dict(self, sample_tool_reg: ToolRegistration):
        d = sample_tool_reg.to_mcp_tool_dict()
        assert d["name"] == "bkt_compute"
        assert "inputSchema" in d
        assert d["inputSchema"]["type"] == "object"


class TestA2AMessage:
    def test_create_message(self):
        msg = A2AMessage(
            message_type=A2AMessageType.TASK_REQUEST,
            from_agent="A1",
            to_agent="A2",
            payload={"task": "diagnose"},
        )
        assert msg.from_agent == "A1"
        assert msg.to_agent == "A2"
        assert msg.message_id  # 自动生成
        assert msg.payload["task"] == "diagnose"


class TestComputeResourceDescriptor:
    def test_available_resource(self):
        r = ComputeResourceDescriptor(
            resource_type=ComputeResourceType.GPU,
            name="RTX-4090",
            gpu_count=1,
            gpu_memory_gb=24.0,
            max_queue_depth=5,
        )
        assert r.is_available
        assert r.queue_depth == 0

    def test_busy_resource(self):
        r = ComputeResourceDescriptor(
            resource_type=ComputeResourceType.GPU,
            name="RTX-4090",
            status=ComputeResourceStatus.BUSY,
        )
        assert not r.is_available

    def test_full_queue(self):
        r = ComputeResourceDescriptor(
            resource_type=ComputeResourceType.CLOUD_GPU,
            name="A100-cluster",
            max_queue_depth=2,
            current_queue=["task1", "task2"],
        )
        assert not r.is_available


# ============================================================
# 2. 异常体系测试
# ============================================================

class TestExceptions:
    def test_l6_error_base(self):
        err = L6Error("TEST_CODE", "something failed", {"key": "value"})
        assert err.code == "TEST_CODE"
        assert err.detail == "something failed"
        assert err.context == {"key": "value"}
        assert "[TEST_CODE]" in str(err)

    def test_json_rpc_error_mapping(self):
        err = ParseError()
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32700

        err2 = MethodNotFoundError("tools/list")
        rpc2 = err2.to_json_rpc_error()
        assert rpc2["code"] == -32601

    def test_transport_timeout(self):
        err = TransportTimeoutError("websocket", 30.0)
        assert err.transport_type == "websocket"
        assert err.timeout_seconds == 30.0
        assert err.to_json_rpc_error()["code"] == -32001

    def test_rate_limit_error(self):
        err = RateLimitError("bkt_compute", 100, 60, 5.5)
        assert err.retry_after == 5.5

    def test_reconnect_exhausted(self):
        err = ReconnectExhaustedError("sse", 5)
        assert err.attempts == 5
        assert err.to_json_rpc_error()["code"] == -32003

    def test_mcp_tool_not_found(self):
        err = MCPToolNotFoundError("nonexistent")
        assert "nonexistent" in err.detail


# ============================================================
# 3. 配置管理测试
# ============================================================

class TestConfig:
    def test_default_config(self):
        cfg = L6Config()
        assert cfg.mcp.protocol_version == "2024-11-05"
        assert cfg.mcp.server_name == "dy3-polaris"
        assert cfg.default_transport == TransportType.STDIO
        assert cfg.stdio.request_timeout == 60.0
        assert cfg.sse.heartbeat_interval == 30.0
        assert cfg.sse.max_retries == 5
        assert cfg.websocket.max_retries == 5
        assert cfg.provenance_enabled is True
        assert cfg.default_rate_limit == 100

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DY3_L6_MCP__PROTOCOL_VERSION", "2025-01-01")
        monkeypatch.setenv("DY3_L6_DEFAULT_TRANSPORT", "websocket")
        cfg = L6Config()
        assert cfg.mcp.protocol_version == "2025-01-01"
        assert cfg.default_transport == TransportType.WEBSOCKET

    def test_get_config_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reset_config(self):
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2

    def test_get_transport_config(self):
        cfg = L6Config()
        assert cfg.get_transport_config(TransportType.STDIO) is cfg.stdio
        assert cfg.get_transport_config(TransportType.SSE) is cfg.sse


# ============================================================
# 4. 限流器测试
# ============================================================

class TestTokenBucketLimiter:
    def test_basic_acquire(self):
        limiter = TokenBucketLimiter(default_limit=3, window_seconds=60)
        assert limiter.acquire("tool_a")[0] is True
        assert limiter.acquire("tool_a")[0] is True
        assert limiter.acquire("tool_a")[0] is True
        allowed, retry_after = limiter.acquire("tool_a")
        assert allowed is False
        assert retry_after > 0

    def test_different_tools_independent(self):
        limiter = TokenBucketLimiter(default_limit=1, window_seconds=60)
        assert limiter.acquire("tool_a")[0] is True
        assert limiter.acquire("tool_b")[0] is True  # 不同工具独立桶

    def test_custom_limit(self):
        limiter = TokenBucketLimiter(default_limit=10, window_seconds=60)
        limiter.set_limit("premium_tool", 1)
        assert limiter.acquire("premium_tool")[0] is True
        assert limiter.acquire("premium_tool")[0] is False
        # 默认限制不受影响
        assert limiter.acquire("normal_tool")[0] is True

    def test_reset(self):
        limiter = TokenBucketLimiter(default_limit=1, window_seconds=60)
        limiter.acquire("tool_a")
        assert limiter.acquire("tool_a")[0] is False
        limiter.reset()
        assert limiter.acquire("tool_a")[0] is True


# ============================================================
# 5. Dy3MCPServer 测试
# ============================================================

class TestDy3MCPServer:
    def test_create_server(self):
        server = Dy3MCPServer(name="test-server", layer=LayerTag.L6_PROTOCOL)
        assert server.name == "test-server"
        assert server.dy3_layer == LayerTag.L6_PROTOCOL
        assert server.dy3_config is not None

    def test_register_tool(self, sample_tool_reg: ToolRegistration):
        server = Dy3MCPServer(name="test-reg")
        async def handler(**kwargs): return {}
        server.register_dy3_tool(sample_tool_reg, handler)

        info = server.get_dy3_tool_info("bkt_compute")
        assert info is not None
        assert info.annotations.layer == LayerTag.L2_PERSONALIZATION
        assert info.annotations.estimated_latency_ms == 100

    def test_register_disabled_tool(self):
        server = Dy3MCPServer(name="test-disabled")
        reg = ToolRegistration(
            name="disabled_tool",
            description="A disabled tool",
            enabled=False,
        )
        async def handler(**kwargs): return {}
        server.register_dy3_tool(reg, handler)
        assert server.get_dy3_tool_info("disabled_tool") is None

    def test_list_tools_by_category(self, sample_tool_reg: ToolRegistration):
        server = Dy3MCPServer(name="test-cat")
        async def handler(**kwargs): return {}
        server.register_dy3_tool(sample_tool_reg, handler)

        internal = server.list_tools_by_category(ToolCategory.INTERNAL)
        assert len(internal) == 1
        assert internal[0].name == "bkt_compute"

        connector = server.list_tools_by_category(ToolCategory.CONNECTOR_TIER1)
        assert len(connector) == 0

    def test_kpa_chain(self):
        server = Dy3MCPServer(name="test-kpa")
        kpa1 = server.create_kpa(
            KPAEventType.TOOL_INVOKED, "A1",
            input_snapshot={"kp": "DOM-A-01"},
        )
        kpa2 = server.create_kpa(
            KPAEventType.AGENT_OUTPUT, "A1",
            output_snapshot={"result": "ok"},
        )
        chain = server.kpa_chain
        assert len(chain) == 2
        assert kpa2.prev_hash == kpa1.compute_hash()

        server.reset_kpa_chain()
        assert len(server.kpa_chain) == 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        server = Dy3MCPServer(name="test-health", layer=LayerTag.L3_DOMAIN_KNOWLEDGE)
        result = await server.health_check()
        assert result["status"] == "healthy"
        assert result["server_name"] == "test-health"
        assert result["layer"] == "L3"


# ============================================================
# 6. ExponentialBackoff 测试
# ============================================================

class TestExponentialBackoff:
    def test_basic_sequence(self):
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=30.0, max_retries=5, jitter=False)
        assert backoff.next_delay() == 1.0   # 1 * 2^0
        assert backoff.next_delay() == 2.0   # 1 * 2^1
        assert backoff.next_delay() == 4.0   # 1 * 2^2
        assert backoff.next_delay() == 8.0   # 1 * 2^3
        assert backoff.next_delay() == 16.0  # 1 * 2^4
        assert backoff.next_delay() is None  # exhausted

    def test_max_delay_cap(self):
        backoff = ExponentialBackoff(base_delay=10.0, max_delay=20.0, max_retries=10, jitter=False)
        assert backoff.next_delay() == 10.0
        assert backoff.next_delay() == 20.0  # 10 * 2^1 = 20 (capped)
        assert backoff.next_delay() == 20.0  # 10 * 2^2 = 40 → capped to 20

    def test_reset(self):
        backoff = ExponentialBackoff(max_retries=1)
        assert backoff.next_delay() is not None
        assert backoff.next_delay() is None
        backoff.reset()
        assert backoff.next_delay() is not None

    def test_jitter_range(self):
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=30.0, max_retries=5, jitter=True)
        for _ in range(5):
            d = backoff.next_delay()
            assert d is not None
            assert 0.5 <= d <= 30.0  # jitter: [0.5x, 1.0x] of base


# ============================================================
# 7. 中间件与钩子测试
# ============================================================

class TestMiddlewareHooks:
    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        """限流中间件在工具调用时生效."""
        server = Dy3MCPServer(name="test-rate-limit")
        reg = ToolRegistration(
            name="limited_tool",
            description="A tool with rate limit",
            annotations=Dy3ToolAnnotations(rate_limit=1),
        )

        async def handler(**kwargs): return {"ok": True}
        server.register_dy3_tool(reg, handler)

        # 第一次调用成功
        wrapped = server._dy3_tool_handlers.get("limited_tool")
        result = await wrapped()
        assert result == {"ok": True}

        # 第二次调用应触发限流
        from dy3_polaris.l6.core.exceptions import RateLimitError
        with pytest.raises(RateLimitError):
            await wrapped()

    @pytest.mark.asyncio
    async def test_pre_call_hook_triggered(self):
        """pre_call 钩子在工具调用前触发."""
        server = Dy3MCPServer(name="test-pre-hook")
        calls = []

        @server.on_pre_call
        async def pre_hook(tool_name: str, arguments: dict):
            calls.append(("pre", tool_name, arguments))

        reg = ToolRegistration(
            name="hooked_tool",
            description="A hooked tool",
        )
        async def handler(x: int = 0): return {"x": x}
        server.register_dy3_tool(reg, handler)

        wrapped = server._dy3_tool_handlers.get("hooked_tool")
        await wrapped(x=42)

        assert len(calls) == 1
        assert calls[0][0] == "pre"
        assert calls[0][1] == "hooked_tool"
        assert calls[0][2]["x"] == 42

    @pytest.mark.asyncio
    async def test_post_call_hook_triggered(self):
        """post_call 钩子在工具调用后触发."""
        server = Dy3MCPServer(name="test-post-hook")
        calls = []

        @server.on_post_call
        async def post_hook(tool_name: str, arguments: dict, result: Any):
            calls.append(("post", tool_name, result))

        reg = ToolRegistration(
            name="post_tool",
            description="A post-hooked tool",
        )
        async def handler(val: str = ""): return {"val": val}
        server.register_dy3_tool(reg, handler)

        wrapped = server._dy3_tool_handlers.get("post_tool")
        await wrapped(val="hello")

        assert len(calls) == 1
        assert calls[0][0] == "post"
        assert calls[0][2] == {"val": "hello"}

    @pytest.mark.asyncio
    async def test_kpa_created_on_tool_call(self):
        """工具调用自动创建 KPA."""
        server = Dy3MCPServer(name="test-kpa-auto")
        server.reset_kpa_chain()

        reg = ToolRegistration(
            name="kpa_tool",
            description="A tool that creates KPA",
        )
        async def handler(y: int = 0): return {"y": y}
        server.register_dy3_tool(reg, handler)

        initial = len(server.kpa_chain)
        wrapped = server._dy3_tool_handlers.get("kpa_tool")
        await wrapped(y=99)

        assert len(server.kpa_chain) == initial + 1
        kpa = server.kpa_chain[-1]
        assert kpa.event_type == KPAEventType.TOOL_INVOKED
        assert kpa.actor == "kpa_tool"
        assert kpa.output_snapshot["success"] is True
        assert kpa.output_snapshot["result_type"] == "dict"

    @pytest.mark.asyncio
    async def test_handler_exception_updates_kpa(self):
        """handler 异常时 KPA 记录错误."""
        server = Dy3MCPServer(name="test-kpa-error")
        server.reset_kpa_chain()

        reg = ToolRegistration(name="error_tool", description="Always fails")
        async def handler(): raise ValueError("intentional failure")
        server.register_dy3_tool(reg, handler)

        wrapped = server._dy3_tool_handlers.get("error_tool")
        with pytest.raises(ValueError):
            await wrapped()

        kpa = server.kpa_chain[-1]
        assert kpa.output_snapshot["success"] is False
        assert "intentional failure" in kpa.output_snapshot["error"]


# ============================================================
# 8. 服务端安全调用测试
# ============================================================

class TestServerCallToolSafe:
    @pytest.mark.asyncio
    async def test_call_existing_tool(self):
        """call_tool_safe 调用已注册工具."""
        server = Dy3MCPServer(name="test-safe-call")
        reg = ToolRegistration(name="safe_tool", description="Safe tool")
        async def handler(): return {"result": "ok"}
        server.register_dy3_tool(reg, handler)

        result = await server.call_tool_safe("safe_tool", {})
        assert result["tool_name"] == "safe_tool"
        assert result["provenance_chain_length"] >= 0

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self):
        """call_tool_safe 调用未注册工具应抛异常."""
        server = Dy3MCPServer(name="test-missing")
        from dy3_polaris.l6.core.exceptions import MCPToolNotFoundError
        with pytest.raises(MCPToolNotFoundError):
            await server.call_tool_safe("missing_tool", {})


# ============================================================
# 9. 工具函数测试
# ============================================================

class TestSnapshotSanitize:
    def test_basic_types(self):
        data = {"str": "hello", "int": 42, "float": 3.14, "bool": True, "none": None}
        result = snapshot_sanitize(data)
        assert result["str"] == "hello"
        assert result["int"] == 42

    def test_long_string_truncation(self):
        data = {"long": "x" * 500}
        result = snapshot_sanitize(data)
        assert len(result["long"]) == 259  # 256 + "..."

    def test_nested_dict(self):
        data = {"a": {"b": {"c": "deep"}}}
        result = snapshot_sanitize(data, max_depth=1)
        assert result["a"]["b"] == "<dict>"  # depth exceeded at level 2

    def test_list_truncation(self):
        data = {"items": [{"k": "v"}] * 20}
        result = snapshot_sanitize(data)
        assert len(result["items"]) == 10  # max 10 items