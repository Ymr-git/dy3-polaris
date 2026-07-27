"""T7 REST API 与 Legacy 集成测试.

基于 httpx.AsyncClient + ASGITransport 进行无端口测试。

测试覆盖:
1. 响应辅助函数 (_ok / _err / _legacy_ok / _legacy_err)
2. 健康检查端点
3. 工具管理端点
4. 算力资源端点
5. A2A 协议端点
6. 溯源端点
7. 广播端点
8. 记忆图谱端点
9. JSON-RPC 2.0 端点
10. L6Router 元信息
11. Legacy 适配器端点
12. LegacyAdapter 元信息
13. Legacy 字段映射
"""

from __future__ import annotations

import logging

logging.disable(logging.CRITICAL)

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette

from dy3_polaris.l6.api.router import L6Router, _ok, _err
from dy3_polaris.l6.api.legacy import LegacyAdapter, _legacy_ok, _legacy_err, _LegacyHandlers
from dy3_polaris.l6.core.engine import L6CoreEngine
from dy3_polaris.l6.core.config import L6Config
from dy3_polaris.l6.core.models import (
    ToolRegistration,
    Dy3ToolAnnotations,
    ToolCategory,
    ComputeResourceDescriptor,
    ComputeResourceType,
    ComputeResourceStatus,
    A2ACapability,
)
from dy3_polaris.l6.broadcast.memory_graph import MemoryNode, NodeType, EdgeType


# ============================================================
# 测试辅助
# ============================================================

def _make_engine() -> L6CoreEngine:
    """创建并初始化 L6CoreEngine, 注册示例数据."""
    engine = L6CoreEngine()
    engine.initialize()

    # 注册示例工具 (register_sync 因为 handler 是同步的)
    tool = ToolRegistration(
        name="test_tool",
        description="测试工具",
        annotations=Dy3ToolAnnotations(category=ToolCategory.INTERNAL),
    )
    engine.tool_registry.register_sync(tool, handler=lambda **args: {"echo": args})

    # 注册算力资源
    desc = ComputeResourceDescriptor(
        resource_type=ComputeResourceType.LOCAL_CPU,
        resource_id="cpu-01",
        name="test-cpu",
    )
    engine.compute_scheduler.register(desc)

    # 注册 Agent
    cap = A2ACapability(
        agent_id="agent-1",
        agent_name="测试Agent",
    )
    engine.a2a_bus.register_agent("agent-1", cap)

    # 添加记忆图谱节点
    engine.memory_graph.add_node(
        node_id="kp-1",
        node_type=NodeType.KNOWLEDGE,
        content={"title": "化学键"},
    )

    # 创建溯源链
    engine.provenance_store.create_chain()

    return engine


async def _get(client: AsyncClient, path: str) -> dict:
    r = await client.get(path)
    return r.json()


async def _post(client: AsyncClient, path: str, json: dict | None = None) -> dict:
    r = await client.post(path, json=json or {})
    return r.json()


async def _delete(client: AsyncClient, path: str) -> dict:
    r = await client.delete(path)
    return r.json()


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
async def client():
    engine = _make_engine()
    router = L6Router(engine)
    app = router.create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def legacy_client():
    engine = _make_engine()
    adapter = LegacyAdapter(engine, prefix="/api/v1")
    app = adapter.create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ============================================================
# 1. 响应辅助函数
# ============================================================

class TestResponseHelpers:
    """测试 _ok / _err / _legacy_ok / _legacy_err 辅助函数."""

    def test_ok_基本格式(self) -> None:
        r = _ok()
        assert r["code"] == 0
        assert r["data"] is None
        assert r["message"] == ""

    def test_ok_带数据(self) -> None:
        r = _ok({"k": "v"})
        assert r["code"] == 0
        assert r["data"] == {"k": "v"}

    def test_err_基本格式(self) -> None:
        r = _err(-32000, "错误")
        assert r["code"] == -32000
        assert r["message"] == "错误"

    def test_err_带_detail(self) -> None:
        r = _err(-32000, "错误", "详细信息")
        assert r["detail"] == "详细信息"

    def test_legacy_ok_基本格式(self) -> None:
        r = _legacy_ok()
        assert r["status"] == "ok"
        assert r["data"] is None
        assert r["error_msg"] == ""
        assert r["api_version"] == "legacy/v1"

    def test_legacy_err_基本格式(self) -> None:
        r = _legacy_err("出错了")
        assert r["status"] == "error"
        assert r["error_msg"] == "出错了"
        assert r["api_version"] == "legacy/v1"


# ============================================================
# 2. 健康检查
# ============================================================

class TestHealthEndpoint:
    """GET /health 健康检查."""

    async def test_health_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/health")
        assert data["code"] == 0
        assert data["data"]["initialized"] is True

    async def test_health_包含模块状态(self, client: AsyncClient) -> None:
        data = await _get(client, "/health")
        mods = data["data"]["modules"]
        assert "tool_registry" in mods
        assert "a2a_bus" in mods
        assert "compute_scheduler" in mods


# ============================================================
# 3. 工具管理
# ============================================================

class TestToolsEndpoints:
    """工具管理端点."""

    async def test_list_tools_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/tools")
        assert data["code"] == 0
        assert isinstance(data["data"], list)

    async def test_list_tools_包含注册工具(self, client: AsyncClient) -> None:
        data = await _get(client, "/tools")
        names = [t["name"] for t in data["data"]]
        assert "test_tool" in names

    async def test_get_tool_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/tools/test_tool")
        assert data["code"] == 0
        assert data["data"]["name"] == "test_tool"

    async def test_get_tool_404(self, client: AsyncClient) -> None:
        data = await _get(client, "/tools/nonexistent")
        assert data["code"] == -32601

    async def test_call_tool_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/tools/test_tool/call", {"arguments": {"x": 1}})
        assert data["code"] == 0

    async def test_call_tool_404(self, client: AsyncClient) -> None:
        data = await _post(client, "/tools/ghost/call")
        assert data["code"] != 0

    async def test_tool_stats_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/tools/stats")
        assert data["code"] == 0


# ============================================================
# 4. 算力资源
# ============================================================

class TestComputeEndpoints:
    """算力资源端点."""

    async def test_list_resources_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/compute/resources")
        assert data["code"] == 0
        assert isinstance(data["data"], list)
        ids = [r["resource_id"] for r in data["data"]]
        assert "cpu-01" in ids

    async def test_get_resource_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/compute/resources/cpu-01")
        assert data["code"] == 0
        assert data["data"]["name"] == "test-cpu"

    async def test_get_resource_404(self, client: AsyncClient) -> None:
        data = await _get(client, "/compute/resources/ghost")
        assert data["code"] == -32000

    async def test_register_resource_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/compute/resources", {
            "resource_type": "local_cpu",
            "name": "new-cpu",
        })
        assert data["code"] == 0
        assert "resource_id" in data["data"]

    async def test_delete_resource_200(self, client: AsyncClient) -> None:
        data = await _delete(client, "/compute/resources/cpu-01")
        assert data["code"] == 0
        assert data["data"]["unregistered"] == "cpu-01"

    async def test_compute_metrics_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/compute/metrics")
        assert data["code"] == 0


# ============================================================
# 5. A2A 协议
# ============================================================

class TestA2AEndpoints:
    """A2A 协议端点."""

    async def test_list_agents_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/a2a/agents")
        assert data["code"] == 0
        assert isinstance(data["data"], list)
        ids = [a.get("agent_id") for a in data["data"]]
        assert "agent-1" in ids

    async def test_get_agent_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/a2a/agents/agent-1")
        assert data["code"] == 0
        assert data["data"]["agent_name"] == "测试Agent"

    async def test_get_agent_404(self, client: AsyncClient) -> None:
        data = await _get(client, "/a2a/agents/ghost")
        assert data["code"] == -32000

    async def test_delete_agent_200(self, client: AsyncClient) -> None:
        data = await _delete(client, "/a2a/agents/agent-1")
        assert data["code"] == 0


# ============================================================
# 6. 溯源
# ============================================================

class TestProvenanceEndpoints:
    """溯源端点."""

    async def test_list_chains_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/provenance/chains")
        assert data["code"] == 0
        assert data["data"]["chain_count"] >= 1

    async def test_create_chain_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/provenance/chains")
        assert data["code"] == 0
        assert "chain_id" in data["data"]

    async def test_get_chain_404(self, client: AsyncClient) -> None:
        data = await _get(client, "/provenance/chains/ghost")
        assert data["code"] == -32000


# ============================================================
# 7. 广播
# ============================================================

class TestBroadcastEndpoints:
    """广播端点."""

    async def test_list_topics_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/broadcast/topics")
        assert data["code"] == 0
        assert isinstance(data["data"], list)

    async def test_publish_event_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/broadcast/publish", {
            "topic": "test.event",
            "payload": {"k": "v"},
        })
        assert data["code"] == 0
        assert data["data"]["topic"] == "test.event"
        assert "event_id" in data["data"]

    async def test_broadcast_metrics_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/broadcast/metrics")
        assert data["code"] == 0

    async def test_event_log_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/broadcast/events")
        assert data["code"] == 0
        assert isinstance(data["data"], list)


# ============================================================
# 8. 记忆图谱
# ============================================================

class TestMemoryEndpoints:
    """记忆图谱端点."""

    async def test_add_node_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/memory/nodes", {
            "node_id": "kp-new",
            "node_type": "knowledge",
            "content": {"title": "新知识点"},
        })
        assert data["code"] == 0
        assert data["data"]["node_id"] == "kp-new"

    async def test_get_node_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/memory/nodes/kp-1")
        assert data["code"] == 0
        assert data["data"]["content"]["title"] == "化学键"

    async def test_get_node_404(self, client: AsyncClient) -> None:
        data = await _get(client, "/memory/nodes/ghost")
        assert data["code"] == -32000

    async def test_delete_node_200(self, client: AsyncClient) -> None:
        data = await _delete(client, "/memory/nodes/kp-1")
        assert data["code"] == 0
        assert data["data"]["removed"] == "kp-1"

    async def test_add_edge_200(self, client: AsyncClient) -> None:
        """先添加两个节点, 再连边."""
        engine = _make_engine()
        engine.memory_graph.add_node("a", NodeType.KNOWLEDGE)
        engine.memory_graph.add_node("b", NodeType.KNOWLEDGE)
        router = L6Router(engine)
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            data = await _post(ac, "/memory/edges", {
                "source_id": "a",
                "target_id": "b",
                "edge_type": "prerequisite",
                "weight": 0.8,
            })
            assert data["code"] == 0
            assert data["data"]["edge_type"] == "prerequisite"

    async def test_memory_metrics_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/memory/metrics")
        assert data["code"] == 0
        assert "node_count" in data["data"]

    async def test_export_memory_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/memory/export")
        assert data["code"] == 0
        assert "nodes" in data["data"]
        assert "edges" in data["data"]

    async def test_decay_memory_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/memory/decay")
        assert data["code"] == 0
        assert "pruned" in data["data"]

    async def test_spread_activation_200(self, client: AsyncClient) -> None:
        engine = _make_engine()
        engine.memory_graph.add_node("a", NodeType.KNOWLEDGE)
        engine.memory_graph.add_node("b", NodeType.KNOWLEDGE)
        engine.memory_graph.add_edge("a", "b", weight=1.0)
        router = L6Router(engine)
        app = router.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            data = await _post(ac, "/memory/spread", {"node_id": "a"})
            assert data["code"] == 0
            assert "a" in data["data"]

    async def test_search_memory_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/memory/search?node_type=knowledge")
        assert data["code"] == 0
        assert isinstance(data["data"], list)


# ============================================================
# 9. JSON-RPC 2.0
# ============================================================

class TestJsonRpcEndpoint:
    """JSON-RPC 2.0 端点."""

    async def test_jsonrpc_get_status(self, client: AsyncClient) -> None:
        data = await _post(client, "/jsonrpc", {
            "jsonrpc": "2.0",
            "method": "get_status",
            "params": {},
            "id": 1,
        })
        assert "result" in data
        assert data["id"] == 1

    async def test_jsonrpc_method_not_found(self, client: AsyncClient) -> None:
        data = await _post(client, "/jsonrpc", {
            "jsonrpc": "2.0",
            "method": "unknown",
            "params": {},
            "id": 2,
        })
        assert "error" in data
        assert data["error"]["code"] == -32000

    async def test_jsonrpc_notification(self, client: AsyncClient) -> None:
        data = await _post(client, "/jsonrpc", {
            "jsonrpc": "2.0",
            "method": "get_status",
            "params": {},
        })
        assert data.get("result") is None
        assert data.get("id") is None

    async def test_jsonrpc_batch(self, client: AsyncClient) -> None:
        data = await _post(client, "/jsonrpc", [
            {"jsonrpc": "2.0", "method": "get_status", "params": {}, "id": 1},
            {"jsonrpc": "2.0", "method": "get_status", "params": {}, "id": 2},
        ])
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_jsonrpc_parse_error(self, client: AsyncClient) -> None:
        """发送非 JSON body."""
        r = await client.post(
            "/jsonrpc",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        data = r.json()
        assert data["code"] == -32700


# ============================================================
# 10. L6Router 元信息
# ============================================================

class TestL6Router:
    """L6Router 元信息."""

    def test_create_app_返回_starlette实例(self) -> None:
        engine = _make_engine()
        router = L6Router(engine)
        app = router.create_app()
        assert isinstance(app, Starlette)

    def test_get_routes_summary_25条路由(self) -> None:
        engine = _make_engine()
        router = L6Router(engine)
        routes = router.get_routes_summary()
        assert len(routes) == 25

    def test_routes_summary_每条有必需字段(self) -> None:
        engine = _make_engine()
        router = L6Router(engine)
        for r in router.get_routes_summary():
            assert "path" in r
            assert "methods" in r
            assert "description" in r


# ============================================================
# 11. Legacy 端点
# ============================================================

class TestLegacyEndpoints:
    """Legacy 适配器端点."""

    async def test_legacy_tool_list_200(self, legacy_client: AsyncClient) -> None:
        data = await _get(legacy_client, "/api/v1/tools")
        assert data["status"] == "ok"
        assert isinstance(data["data"], list)

    async def test_legacy_tool_call_400_缺tool(self, legacy_client: AsyncClient) -> None:
        data = await _post(legacy_client, "/api/v1/tool/call", {})
        assert data["status"] == "error"

    async def test_legacy_system_status_200(self, legacy_client: AsyncClient) -> None:
        data = await _get(legacy_client, "/api/v1/system/status")
        assert data["status"] == "ok"
        assert data["data"]["online"] is True

    async def test_legacy_learner_404(self, legacy_client: AsyncClient) -> None:
        data = await _get(legacy_client, "/api/v1/learner/ghost")
        assert data["status"] == "error"

    async def test_legacy_knowledge_404(self, legacy_client: AsyncClient) -> None:
        data = await _get(legacy_client, "/api/v1/knowledge/ghost")
        assert data["status"] == "error"

    async def test_legacy_assessment_404(self, legacy_client: AsyncClient) -> None:
        data = await _get(legacy_client, "/api/v1/assessment/ghost")
        assert data["status"] == "error"


# ============================================================
# 12. LegacyAdapter 元信息
# ============================================================

class TestLegacyAdapter:
    """LegacyAdapter 元信息."""

    def test_create_app_返回_starlette实例(self) -> None:
        engine = _make_engine()
        adapter = LegacyAdapter(engine)
        app = adapter.create_app()
        assert isinstance(app, Starlette)

    def test_get_routes_summary_6条路由(self) -> None:
        engine = _make_engine()
        adapter = LegacyAdapter(engine)
        routes = adapter.get_routes_summary()
        assert len(routes) == 6

    def test_default_prefix(self) -> None:
        engine = _make_engine()
        adapter = LegacyAdapter(engine)
        routes = adapter.get_routes_summary()
        assert all(r["path"].startswith("/api/v1") for r in routes)

    def test_custom_prefix(self) -> None:
        engine = _make_engine()
        adapter = LegacyAdapter(engine, prefix="/api/v2")
        routes = adapter.get_routes_summary()
        assert all(r["path"].startswith("/api/v2") for r in routes)

    async def test_learner_200(self, legacy_client: AsyncClient) -> None:
        """添加 learner 节点后查询."""
        engine = _make_engine()
        engine.memory_graph.add_node(
            node_id="stu-1",
            node_type=NodeType.LEARNER,
            metadata={"student_id": "stu-1"},
            content={"name": "张三", "grade": "大三"},
        )
        adapter = LegacyAdapter(engine, prefix="/api/v1")
        app = adapter.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            data = await _get(ac, "/api/v1/learner/stu-1")
            assert data["status"] == "ok"
            assert data["data"]["student_id"] == "stu-1"
            assert data["data"]["name"] == "张三"

    async def test_knowledge_200(self, legacy_client: AsyncClient) -> None:
        engine = _make_engine()
        adapter = LegacyAdapter(engine, prefix="/api/v1")
        app = adapter.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            data = await _get(ac, "/api/v1/knowledge/kp-1")
            assert data["status"] == "ok"
            assert data["data"]["kp_id"] == "kp-1"
            assert data["data"]["title"] == "化学键"

    async def test_assessment_200(self, legacy_client: AsyncClient) -> None:
        engine = _make_engine()
        engine.memory_graph.add_node(
            node_id="asmt-1",
            node_type=NodeType.ASSESSMENT,
            metadata={"student_id": "stu-1"},
            content={"score": 95, "max_score": 100, "passed": True},
        )
        adapter = LegacyAdapter(engine, prefix="/api/v1")
        app = adapter.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            data = await _get(ac, "/api/v1/assessment/asmt-1")
            assert data["status"] == "ok"
            assert data["data"]["assessment_id"] == "asmt-1"
            assert data["data"]["score"] == 95


# ============================================================
# 13. Legacy 字段映射
# ============================================================

class TestLegacyFieldMapping:
    """Legacy 字段映射."""

    def test_node_to_legacy_learner(self) -> None:
        node = MemoryNode(
            node_id="s1",
            node_type=NodeType.LEARNER,
            content={"name": "张三"},
            metadata={"student_id": "s1", "grade": "大三", "major": "化学"},
            strength=0.8,
        )
        result = _LegacyHandlers._node_to_legacy_learner(node)
        assert result["student_id"] == "s1"
        assert result["name"] == "张三"
        assert result["grade"] == "大三"
        assert result["major"] == "化学"
        assert result["strength"] == 0.8

    def test_node_to_legacy_knowledge(self) -> None:
        node = MemoryNode(
            node_id="kp-1",
            node_type=NodeType.KNOWLEDGE,
            content={"title": "化学键", "description": "离子键与共价键"},
            strength=0.9,
        )
        result = _LegacyHandlers._node_to_legacy_knowledge(node)
        assert result["kp_id"] == "kp-1"
        assert result["title"] == "化学键"
        assert result["description"] == "离子键与共价键"
        assert result["strength"] == 0.9

    def test_node_to_legacy_assessment(self) -> None:
        node = MemoryNode(
            node_id="asmt-1",
            node_type=NodeType.ASSESSMENT,
            metadata={"student_id": "s1"},
            content={"score": 95, "max_score": 100, "passed": True},
        )
        result = _LegacyHandlers._node_to_legacy_assessment(node)
        assert result["assessment_id"] == "asmt-1"
        assert result["student_id"] == "s1"
        assert result["score"] == 95
        assert result["passed"] is True
