"""全栈集成测试 — L0→L1→L2→L3→L4→L5→L6 七层统一组装.

验证 UnifiedApp 将所有七层 Router 挂载到单一 Starlette 应用,
以及 IntegrationBridge 跨七层健康聚合与 API 发现.

测试维度:
1. 七层路由挂载验证 — 每层关键端点可达
2. 统一健康检查 — 聚合七层状态
3. API 发现 — 列出所有层端点
4. IntegrationBridge — 七层健康聚合
5. 端到端治理流 — L0 governance 可通过统一入口访问
6. 端到端知识检索 — L3 retrieval 可通过统一入口访问
7. 端到端协议层 — L6 JSON-RPC 可通过统一入口访问
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from dy3_polaris.l5.unified_app import UnifiedApp


# ============================================================
# 辅助函数
# ============================================================


async def _async_login(ac: AsyncClient) -> dict[str, str]:
    """登录获取 Bearer token (安全网关: 写端点需鉴权)."""
    resp = await ac.post("/l1/api/v1/auth/login",
                         json={"student_id": "DY20240001", "password": "demo123"})
    assert resp.status_code == 200, resp.text
    return {"Authorization": "Bearer " + resp.json()["data"]["access_token"]}


def _make_full_app() -> UnifiedApp:
    """创建挂载全部七层的 UnifiedApp."""
    return UnifiedApp.create_full_app_builder()


# ============================================================
# 1. 七层路由挂载验证
# ============================================================


class TestSevenLayerMount:
    """验证全部七层路由挂载到统一应用."""

    @pytest.mark.asyncio
    async def test_l0_governance_health(self):
        """L0 governance 健康检查端点可达."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/governance/v1/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 0

    @pytest.mark.asyncio
    async def test_l1_health(self):
        """L1 用户域健康检查端点可达."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/l1/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_l2_health(self):
        """L2 个性化层健康检查端点可达."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/l2/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_l3_health(self):
        """L3 知识层健康检查端点可达."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/l3/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_l4_health(self):
        """L4 决策引擎健康检查端点可达."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/l4/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_l5_health(self):
        """L5 Agent Runtime 健康检查端点可达."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/l5/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_l6_health(self):
        """L6 协议基础设施健康检查端点可达."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/l6/health")
            assert resp.status_code == 200


# ============================================================
# 2. 统一健康检查
# ============================================================


class TestUnifiedHealth:
    """测试统一 /health 端点聚合七层状态."""

    @pytest.mark.asyncio
    async def test_unified_health_returns_all_layers(self):
        """统一健康检查返回全部七层状态."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 0
            layers = body["data"]["layers"]
            # 应包含全部八层 (M-F2 挂载 L7 后)
            expected_layers = {"l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"}
            assert set(layers.keys()) == expected_layers

    @pytest.mark.asyncio
    async def test_unified_health_has_overall_status(self):
        """统一健康检查包含 overall_status 字段."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            body = resp.json()
            assert "status" in body["data"]
            assert body["data"]["status"] in ("healthy", "degraded")

    @pytest.mark.asyncio
    async def test_full_builder_all_layers_healthy(self):
        """全栈默认组装后七层核心服务全部可用 (整体系统就绪)."""
        app_builder = _make_full_app()
        health = app_builder.bridge.get_cross_layer_health()
        for layer, info in health.items():
            assert info["status"] == "healthy", f"{layer} 未就绪: {info}"
            unavailable = [
                name
                for name, status in info.get("services", {}).items()
                if (status.get("state") if isinstance(status, dict) else status) != "available"
            ]
            assert not unavailable, f"{layer} 存在不可用服务: {unavailable}"


# ============================================================
# 3. API 发现
# ============================================================


class TestAPIDiscovery:
    """测试 /api/info 端点列出所有层端点."""

    @pytest.mark.asyncio
    async def test_api_info_lists_all_layers(self):
        """API 发现端点列出全部七层."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/info")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 0
            layers_in_response = body["data"].get("layers", [])
            # 应包含全部七层标识
            for layer in ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]:
                assert layer in layers_in_response

    @pytest.mark.asyncio
    async def test_api_info_has_endpoints(self):
        """API 发现端点返回非空端点列表."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/info")
            body = resp.json()
            endpoints = body["data"]["endpoints"]
            assert len(endpoints) > 0
            assert body["data"]["total"] == len(endpoints)

    @pytest.mark.asyncio
    async def test_api_info_endpoints_have_layer_field(self):
        """每个端点包含 layer 字段."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/info")
            endpoints = resp.json()["data"]["endpoints"]
            for ep in endpoints:
                assert "layer" in ep
                assert "path" in ep
                assert "methods" in ep


# ============================================================
# 3.5 全栈功能链路
# ============================================================


class TestFunctionalFullStack:
    """测试全栈默认组装后的真实功能链路 (会话/IRT/画像)."""

    @pytest.mark.asyncio
    async def test_l5_create_and_get_session(self):
        """L5 会话创建与查询可用."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            headers = await _async_login(ac)
            resp = await ac.post(
                "/l5/session",
                json={
                    "learner_id": "demo-learner",
                    "context": {"source": "full-stack-test"},
                },
                headers=headers,
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["code"] == 0
            session_id = body["data"]["session_id"]

            got = await ac.get(f"/l5/session/{session_id}")
            assert got.status_code == 200
            assert got.json()["data"]["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_l2_irt_ability_snapshot(self):
        """L2 IRT 能力快照查询可用."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/l2/irt/ability/demo-learner")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["code"] == 0
            assert "theta" in body["data"]

    @pytest.mark.asyncio
    async def test_full_chain_query_profile_decision_output(self):
        """端到端链路: 画像上下文 -> L4 决策 -> 输出记录."""
        app_builder = _make_full_app()
        bridge = app_builder.bridge
        result = await bridge.process_query_with_profile(
            learner_id="demo-learner",
            query="Dy3+ 的激发态波长是多少？",
        )
        assert result is not None
        assert "action_type" in result
        assert "confidence" in result
        assert "learner_profile" in result
        assert result["learner_profile"]["learner_id"] == "demo-learner"


# ============================================================
# 4. IntegrationBridge 七层健康聚合
# ============================================================


class TestBridgeFullStack:
    """测试 IntegrationBridge 七层健康聚合."""

    def test_bridge_health_includes_all_layers(self):
        """Bridge 健康聚合包含全部八层."""
        app_builder = _make_full_app()
        bridge = app_builder.bridge
        health = bridge.get_cross_layer_health()
        expected_layers = {"l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"}
        assert set(health.keys()) == expected_layers

    def test_bridge_has_l0_reference(self):
        """Bridge 持有 L0 governance_router 引用."""
        app_builder = _make_full_app()
        bridge = app_builder.bridge
        assert bridge.governance_router is not None

    def test_bridge_has_l1_reference(self):
        """Bridge 持有 L1 router 引用."""
        app_builder = _make_full_app()
        bridge = app_builder.bridge
        assert bridge.l1_router is not None

    def test_bridge_has_l3_reference(self):
        """Bridge 持有 L3 router 引用."""
        app_builder = _make_full_app()
        bridge = app_builder.bridge
        assert bridge.l3_router is not None

    def test_bridge_has_l6_reference(self):
        """Bridge 持有 L6 router 引用."""
        app_builder = _make_full_app()
        bridge = app_builder.bridge
        assert bridge.l6_router is not None


# ============================================================
# 5. 端到端治理流 (L0)
# ============================================================


class TestEndToEndGovernance:
    """通过统一入口访问 L0 治理 API."""

    @pytest.mark.asyncio
    async def test_governance_routes_discovery(self):
        """L0 governance 路由发现可通过统一入口访问."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/governance/v1/routes")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 0
            assert len(body["data"]) > 0

    @pytest.mark.asyncio
    async def test_governance_health_ready(self):
        """L0 governance 就绪探针可通过统一入口访问."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/governance/v1/health/ready")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 0


# ============================================================
# 6. 端到端知识检索 (L3)
# ============================================================


class TestEndToEndKnowledge:
    """通过统一入口访问 L3 知识层 API."""

    @pytest.mark.asyncio
    async def test_l3_stats_endpoint(self):
        """L3 stats 端点可通过统一入口访问."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/l3/stats")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_l3_ontology_endpoint(self):
        """L3 ontology 端点可通过统一入口访问."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/l3/ontology/domains")
            assert resp.status_code == 200


# ============================================================
# 7. 端到端协议层 (L6)
# ============================================================


class TestEndToEndProtocol:
    """通过统一入口访问 L6 协议层 API."""

    @pytest.mark.asyncio
    async def test_l6_tools_listing(self):
        """L6 tools 列表可通过统一入口访问."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/l6/tools")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_l6_jsonrpc(self):
        """L6 JSON-RPC 端点可通过统一入口访问."""
        app_builder = _make_full_app()
        app = app_builder.create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/l6/jsonrpc", json={
                "jsonrpc": "2.0",
                "method": "system.listMethods",
                "id": 1,
            })
            assert resp.status_code == 200


# ============================================================
# 8. 统一路由摘要
# ============================================================


class TestUnifiedRoutesSummary:
    """测试 UnifiedApp.get_routes_summary() 包含全部七层."""

    def test_routes_summary_has_all_layers(self):
        """路由摘要包含全部七层端点."""
        app_builder = _make_full_app()
        summary = app_builder.get_routes_summary()
        layers_found = {ep["layer"] for ep in summary}
        for layer in ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "Unified"]:
            assert layer in layers_found

    def test_routes_summary_has_unified_endpoints(self):
        """路由摘要包含统一端点 (/health, /api/info)."""
        app_builder = _make_full_app()
        summary = app_builder.get_routes_summary()
        paths = [ep["path"] for ep in summary]
        assert "/health" in paths
        assert "/api/info" in paths

    def test_routes_summary_total_exceeds_100(self):
        """全部七层路由总数超过 100."""
        app_builder = _make_full_app()
        summary = app_builder.get_routes_summary()
        # 7 layers × ~10+ routes each + 2 unified = 100+
        assert len(summary) > 100
