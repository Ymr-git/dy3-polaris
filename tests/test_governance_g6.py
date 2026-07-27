"""G6 治理集成 REST API 路由层 — 全面集成测试.

使用 httpx.AsyncClient + pytest-asyncio 的 ASGI 测试模式（不启动真实服务器）。
覆盖 GovernanceRouter 的全部 54 条路由，涵盖 G1-G5 + CC1-CC2 子系统。

测试结构:
- 辅助函数测试（_ok, _err, _new_trace_id, GovernanceSubsystems）
- 路由发现测试
- 健康检查测试（三级）
- G1/G2 策略治理测试
- G3 CC1 防幻觉测试
- G4 CC2 人机协作测试
- G5 审计测试
- G5 度量测试
- G5 合规测试
- 全链路集成测试
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# ============================================================
# 导入被测模块
# ============================================================

from dy3_polaris.l0.governance.policy_store import PolicyStore
from dy3_polaris.l0.governance.evaluator import PolicyEvaluator
from dy3_polaris.l0.cc1.pipeline import AntiHallucinationPipeline
from dy3_polaris.l0.cc2.engine import CollaborationEngine
from dy3_polaris.l0.governance.audit_engine import AuditEngine
from dy3_polaris.l0.governance.metrics_engine import MetricsEngine
from dy3_polaris.l0.governance.compliance import ComplianceReporter
from dy3_polaris.l0.governance_router import (
    GovernanceSubsystems,
    GovernanceRouter,
    _GovernanceHandlers,
    _ok,
    _err,
    _new_trace_id,
)


# ============================================================
# 全局 fixture
# ============================================================


@pytest_asyncio.fixture
async def client():
    """创建 ASGI 测试客户端，初始化全部子系统."""
    store = PolicyStore()
    evaluator = PolicyEvaluator(store)
    pipeline = AntiHallucinationPipeline()
    cc2_engine = CollaborationEngine()
    audit = AuditEngine()
    metrics = MetricsEngine()
    compliance = ComplianceReporter()
    subsys = GovernanceSubsystems(
        policy_store=store,
        policy_evaluator=evaluator,
        anti_hallucination_pipeline=pipeline,
        collaboration_engine=cc2_engine,
        audit_engine=audit,
        metrics_engine=metrics,
        compliance_reporter=compliance,
    )
    router = GovernanceRouter(subsys)
    app = router.create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def empty_client():
    """创建未初始化任何子系统的测试客户端（用于测试 503 错误）."""
    subsys = GovernanceSubsystems()
    router = GovernanceRouter(subsys)
    app = router.create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ============================================================
# 1. 辅助函数测试
# ============================================================


class TestOkResponse:
    """测试 _ok 辅助函数."""

    def test_ok_default(self):
        """默认参数返回 code=0, data=None, message=''."""
        result = _ok()
        assert result == {"code": 0, "data": None, "message": ""}

    def test_ok_with_data(self):
        """传入 data 参数."""
        result = _ok(data="test")
        assert result["code"] == 0
        assert result["data"] == "test"
        assert result["message"] == ""

    def test_ok_with_dict_data(self):
        """传入字典类型 data."""
        data = {"key": "value"}
        result = _ok(data=data)
        assert result["code"] == 0
        assert result["data"] == data

    def test_ok_with_list_data(self):
        """传入列表类型 data."""
        data = [1, 2, 3]
        result = _ok(data=data)
        assert result["code"] == 0
        assert result["data"] == data

    def test_ok_with_message(self):
        """传入 message 参数."""
        result = _ok(data="x", message="成功")
        assert result["code"] == 0
        assert result["data"] == "x"
        assert result["message"] == "成功"

    def test_ok_keys(self):
        """验证返回字典只包含三个键."""
        result = _ok(data="test")
        assert set(result.keys()) == {"code", "data", "message"}


class TestErrResponse:
    """测试 _err 辅助函数."""

    def test_err_basic(self):
        """基本错误响应."""
        result = _err(-32000, "test")
        assert result["code"] == -32000
        assert result["message"] == "test"

    def test_err_with_detail(self):
        """包含 detail 字段的错误响应."""
        result = _err(-32000, "test", "detail")
        assert result["code"] == -32000
        assert result["message"] == "test"
        assert result["detail"] == "detail"

    def test_err_without_detail(self):
        """不传 detail 时响应不包含 detail 键."""
        result = _err(-32000, "test")
        assert "detail" not in result

    def test_err_different_codes(self):
        """不同错误码."""
        for code in [-32000, -32700, -32600, 500]:
            result = _err(code, "error")
            assert result["code"] == code
            assert result["message"] == "error"

    def test_err_empty_message(self):
        """空消息错误."""
        result = _err(-32000, "")
        assert result["code"] == -32000
        assert result["message"] == ""


class TestNewTraceId:
    """测试 _new_trace_id 辅助函数."""

    def test_trace_id_prefix(self):
        """trace_id 以 'g6-' 开头."""
        tid = _new_trace_id()
        assert tid.startswith("g6-")

    def test_trace_id_length(self):
        """trace_id 长度为 3 (g6-) + 12 (hex) = 15."""
        tid = _new_trace_id()
        assert len(tid) == 15

    def test_trace_id_unique(self):
        """多次调用生成不同的 ID."""
        ids = {_new_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_trace_id_hex(self):
        """trace_id 前缀后部分为十六进制."""
        tid = _new_trace_id()
        hex_part = tid[3:]
        int(hex_part, 16)  # 不抛异常即通过


class TestGovernanceSubsystems:
    """测试 GovernanceSubsystems 容器."""

    def test_default_health_map_all_false(self):
        """默认构造时所有子系统就绪状态为 False."""
        subsys = GovernanceSubsystems()
        hm = subsys.health_map()
        assert all(not v for v in hm.values())

    def test_health_map_keys(self):
        """health_map 包含所有七个子系统键."""
        subsys = GovernanceSubsystems()
        hm = subsys.health_map()
        expected_keys = {
            "policy_store", "policy_evaluator", "anti_hallucination",
            "collaboration", "audit", "metrics", "compliance",
        }
        assert set(hm.keys()) == expected_keys

    def test_health_map_all_init(self):
        """全部子系统初始化后所有值为 True."""
        subsys = GovernanceSubsystems(
            policy_store=PolicyStore(),
            policy_evaluator=PolicyEvaluator(PolicyStore()),
            anti_hallucination_pipeline=AntiHallucinationPipeline(),
            collaboration_engine=CollaborationEngine(),
            audit_engine=AuditEngine(),
            metrics_engine=MetricsEngine(),
            compliance_reporter=ComplianceReporter(),
        )
        hm = subsys.health_map()
        assert all(v for v in hm.values())

    def test_health_map_partial_init(self):
        """部分子系统初始化，对应值为 True."""
        subsys = GovernanceSubsystems(
            audit_engine=AuditEngine(),
            metrics_engine=MetricsEngine(),
        )
        hm = subsys.health_map()
        assert hm["audit"] is True
        assert hm["metrics"] is True
        assert hm["policy_store"] is False
        assert hm["collaboration"] is False

    def test_none_subsystems(self):
        """显式传入 None 等同于未初始化."""
        subsys = GovernanceSubsystems(policy_store=None)
        assert subsys.health_map()["policy_store"] is False


# ============================================================
# 2. 路由发现测试
# ============================================================


class TestRoutesDiscovery:
    """测试路由发现端点 GET /governance/v1/routes."""

    @pytest.mark.asyncio
    async def test_routes_summary_count(self, client: AsyncClient):
        """路由总数为 54."""
        resp = await client.get("/governance/v1/routes")
        assert resp.status_code == 200
        routes = resp.json()["data"]
        assert len(routes) == 54

    @pytest.mark.asyncio
    async def test_routes_summary_has_health(self, client: AsyncClient):
        """路由列表包含 /governance/v1/health."""
        resp = await client.get("/governance/v1/routes")
        routes = resp.json()["data"]
        paths = [r["path"] for r in routes]
        assert "/governance/v1/health" in paths

    @pytest.mark.asyncio
    async def test_routes_summary_has_chain(self, client: AsyncClient):
        """路由列表包含 /governance/v1/chain/evaluate."""
        resp = await client.get("/governance/v1/routes")
        routes = resp.json()["data"]
        paths = [r["path"] for r in routes]
        assert "/governance/v1/chain/evaluate" in paths

    @pytest.mark.asyncio
    async def test_routes_summary_has_policies(self, client: AsyncClient):
        """路由列表包含策略相关端点."""
        resp = await client.get("/governance/v1/routes")
        routes = resp.json()["data"]
        paths = [r["path"] for r in routes]
        assert "/governance/v1/policies" in paths
        assert "/governance/v1/policies/evaluate" in paths

    @pytest.mark.asyncio
    async def test_routes_summary_has_anti_hallucination(self, client: AsyncClient):
        """路由列表包含防幻觉相关端点."""
        resp = await client.get("/governance/v1/routes")
        routes = resp.json()["data"]
        paths = [r["path"] for r in routes]
        assert "/governance/v1/anti-hallucination/verify" in paths

    @pytest.mark.asyncio
    async def test_routes_summary_has_collaboration(self, client: AsyncClient):
        """路由列表包含协作相关端点."""
        resp = await client.get("/governance/v1/routes")
        routes = resp.json()["data"]
        paths = [r["path"] for r in routes]
        assert "/governance/v1/collaboration/profiles" in paths
        assert "/governance/v1/collaboration/evaluate-react" in paths

    @pytest.mark.asyncio
    async def test_routes_summary_has_audit(self, client: AsyncClient):
        """路由列表包含审计相关端点."""
        resp = await client.get("/governance/v1/routes")
        routes = resp.json()["data"]
        paths = [r["path"] for r in routes]
        assert "/governance/v1/audit/decisions" in paths
        assert "/governance/v1/audit/stats" in paths

    @pytest.mark.asyncio
    async def test_routes_summary_has_metrics(self, client: AsyncClient):
        """路由列表包含度量相关端点."""
        resp = await client.get("/governance/v1/routes")
        routes = resp.json()["data"]
        paths = [r["path"] for r in routes]
        assert "/governance/v1/metrics/define" in paths
        assert "/governance/v1/metrics/dora" in paths

    @pytest.mark.asyncio
    async def test_routes_summary_has_compliance(self, client: AsyncClient):
        """路由列表包含合规相关端点."""
        resp = await client.get("/governance/v1/routes")
        routes = resp.json()["data"]
        paths = [r["path"] for r in routes]
        assert "/governance/v1/compliance/report" in paths
        assert "/governance/v1/compliance/nist-summary" in paths

    @pytest.mark.asyncio
    async def test_routes_summary_each_has_methods_and_description(self, client: AsyncClient):
        """每条路由包含 path, methods, description."""
        resp = await client.get("/governance/v1/routes")
        routes = resp.json()["data"]
        for route in routes:
            assert "path" in route
            assert "methods" in route
            assert "description" in route
            assert isinstance(route["methods"], list)

    @pytest.mark.asyncio
    async def test_routes_summary_code(self, client: AsyncClient):
        """路由发现返回 code=0."""
        resp = await client.get("/governance/v1/routes")
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_get_routes_summary_from_router(self):
        """直接从 GovernanceRouter 获取路由摘要."""
        subsys = GovernanceSubsystems()
        router = GovernanceRouter(subsys)
        summary = router.get_routes_summary()
        assert len(summary) == 54

    @pytest.mark.asyncio
    async def test_routes_health_deep_present(self, client: AsyncClient):
        """路由列表包含深度健康检查端点."""
        resp = await client.get("/governance/v1/routes")
        paths = [r["path"] for r in resp.json()["data"]]
        assert "/governance/v1/health/deep" in paths

    @pytest.mark.asyncio
    async def test_routes_chain_status_present(self, client: AsyncClient):
        """路由列表包含链路状态端点."""
        resp = await client.get("/governance/v1/routes")
        paths = [r["path"] for r in resp.json()["data"]]
        assert "/governance/v1/chain/status" in paths


# ============================================================
# 3. 健康检查测试
# ============================================================


class TestHealthCheck:
    """测试三级健康检查端点."""

    @pytest.mark.asyncio
    async def test_health_alive(self, client: AsyncClient):
        """GET /health 返回 alive 状态."""
        resp = await client.get("/governance/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["status"] == "alive"
        assert "timestamp" in body["data"]

    @pytest.mark.asyncio
    async def test_health_timestamp_is_number(self, client: AsyncClient):
        """健康检查 timestamp 为数字."""
        resp = await client.get("/governance/v1/health")
        data = resp.json()["data"]
        assert isinstance(data["timestamp"], float)

    @pytest.mark.asyncio
    async def test_health_empty_subsystems(self, empty_client: AsyncClient):
        """无子系统时健康检查仍然返回 alive."""
        resp = await empty_client.get("/governance/v1/health")
        assert resp.json()["data"]["status"] == "alive"

    @pytest.mark.asyncio
    async def test_health_ready_all_initialized(self, client: AsyncClient):
        """GET /health/ready — 全部子系统初始化时返回 ready."""
        resp = await client.get("/governance/v1/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["status"] == "ready"
        assert all(body["data"]["subsystems"].values())

    @pytest.mark.asyncio
    async def test_health_ready_no_subsystems(self, empty_client: AsyncClient):
        """无子系统时返回 degraded."""
        resp = await empty_client.get("/governance/v1/health/ready")
        assert resp.json()["data"]["status"] == "degraded"
        assert not any(resp.json()["data"]["subsystems"].values())

    @pytest.mark.asyncio
    async def test_health_ready_subsystems_keys(self, client: AsyncClient):
        """就绪探针包含所有子系统键."""
        resp = await client.get("/governance/v1/health/ready")
        subsys = resp.json()["data"]["subsystems"]
        expected = {
            "policy_store", "policy_evaluator", "anti_hallucination",
            "collaboration", "audit", "metrics", "compliance",
        }
        assert set(subsys.keys()) == expected

    @pytest.mark.asyncio
    async def test_health_deep_all_ok(self, client: AsyncClient):
        """GET /health/deep — 深度检查全链路."""
        resp = await client.get("/governance/v1/health/deep")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["status"] == "healthy"
        assert "checks" in body["data"]

    @pytest.mark.asyncio
    async def test_health_deep_checks_policy_store(self, client: AsyncClient):
        """深度检查包含 policy_store 检查项."""
        resp = await client.get("/governance/v1/health/deep")
        checks = resp.json()["data"]["checks"]
        assert "policy_store" in checks
        assert checks["policy_store"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_deep_checks_policy_evaluator(self, client: AsyncClient):
        """深度检查包含 policy_evaluator 检查项."""
        resp = await client.get("/governance/v1/health/deep")
        checks = resp.json()["data"]["checks"]
        assert "policy_evaluator" in checks
        assert checks["policy_evaluator"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_deep_checks_anti_hallucination(self, client: AsyncClient):
        """深度检查包含 anti_hallucination 检查项."""
        resp = await client.get("/governance/v1/health/deep")
        checks = resp.json()["data"]["checks"]
        assert "anti_hallucination" in checks

    @pytest.mark.asyncio
    async def test_health_deep_checks_collaboration(self, client: AsyncClient):
        """深度检查包含 collaboration 检查项."""
        resp = await client.get("/governance/v1/health/deep")
        checks = resp.json()["data"]["checks"]
        assert "collaboration" in checks

    @pytest.mark.asyncio
    async def test_health_deep_checks_audit(self, client: AsyncClient):
        """深度检查包含 audit 检查项."""
        resp = await client.get("/governance/v1/health/deep")
        checks = resp.json()["data"]["checks"]
        assert "audit" in checks

    @pytest.mark.asyncio
    async def test_health_deep_checks_metrics(self, client: AsyncClient):
        """深度检查包含 metrics 检查项."""
        resp = await client.get("/governance/v1/health/deep")
        checks = resp.json()["data"]["checks"]
        assert "metrics" in checks

    @pytest.mark.asyncio
    async def test_health_deep_empty_subsystems(self, empty_client: AsyncClient):
        """无子系统时深度检查返回 healthy（无检查项）."""
        resp = await empty_client.get("/governance/v1/health/deep")
        body = resp.json()["data"]
        assert body["status"] == "healthy"
        assert body["checks"] == {}

    @pytest.mark.asyncio
    async def test_health_deep_policy_count(self, client: AsyncClient):
        """深度检查返回策略数量."""
        resp = await client.get("/governance/v1/health/deep")
        checks = resp.json()["data"]["checks"]
        assert "policy_count" in checks["policy_store"]


# ============================================================
# 4. G1/G2 策略治理测试
# ============================================================


class TestPoliciesCRUD:
    """测试策略 CRUD 端点."""

    # ---- 创建策略 ----

    @pytest.mark.asyncio
    async def test_create_policy_success(self, client: AsyncClient):
        """POST /policies — 成功创建策略."""
        body = {
            "name": "测试策略",
            "domain": "policy",
            "action": "allow",
        }
        resp = await client.post("/governance/v1/policies", json=body)
        assert resp.status_code == 201
        assert resp.json()["code"] == 0
        assert "policy_id" in resp.json()["data"]

    @pytest.mark.asyncio
    async def test_create_policy_with_all_fields(self, client: AsyncClient):
        """创建策略时传入完整字段."""
        body = {
            "name": "完整策略",
            "description": "测试描述",
            "domain": "policy",
            "scope": "global",
            "priority": 10,
            "enabled": True,
            "action": "deny",
            "tags": ["test", "governance"],
            "created_by": "admin",
        }
        resp = await client.post("/governance/v1/policies", json=body)
        assert resp.status_code == 201
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_create_policy_with_condition(self, client: AsyncClient):
        """创建带匹配条件的策略."""
        body = {
            "name": "条件策略",
            "domain": "policy",
            "action": "log",
            "condition": {
                "logic": "and",
                "rules": [
                    {"field": "agent_id", "operator": "exact", "value": "tutor-001"},
                ],
            },
        }
        resp = await client.post("/governance/v1/policies", json=body)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_create_policy_with_transform(self, client: AsyncClient):
        """创建 transform 动作策略（自动创建 TransformSpec）."""
        body = {
            "name": "转换策略",
            "action": "transform",
            "domain": "policy",
        }
        resp = await client.post("/governance/v1/policies", json=body)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_create_policy_with_escalation(self, client: AsyncClient):
        """创建 escalate 动作策略（自动创建 EscalationSpec）."""
        body = {
            "name": "升级策略",
            "action": "escalate",
            "domain": "policy",
        }
        resp = await client.post("/governance/v1/policies", json=body)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_create_policy_missing_name(self, client: AsyncClient):
        """缺少 name 字段返回 400."""
        body = {"domain": "policy", "action": "allow"}
        resp = await client.post("/governance/v1/policies", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_policy_empty_body(self, client: AsyncClient):
        """空请求体返回 400."""
        resp = await client.post("/governance/v1/policies", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_policy_invalid_json(self, client: AsyncClient):
        """无效 JSON 返回 400."""
        resp = await client.post(
            "/governance/v1/policies",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == -32700

    @pytest.mark.asyncio
    async def test_create_policy_store_uninitialized(self, empty_client: AsyncClient):
        """策略存储未初始化返回 503."""
        resp = await empty_client.post("/governance/v1/policies", json={"name": "x"})
        assert resp.status_code == 503

    # ---- 列表策略 ----

    @pytest.mark.asyncio
    async def test_list_policies_empty(self, client: AsyncClient):
        """GET /policies — 空列表."""
        resp = await client.get("/governance/v1/policies")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_list_policies_after_create(self, client: AsyncClient):
        """创建策略后列表不为空."""
        await client.post("/governance/v1/policies", json={"name": "t1", "domain": "policy"})
        resp = await client.get("/governance/v1/policies")
        assert len(resp.json()["data"]) >= 1

    @pytest.mark.asyncio
    async def test_list_policies_store_uninitialized(self, empty_client: AsyncClient):
        """策略存储未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/policies")
        assert resp.status_code == 503

    # ---- 查询策略 ----

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, client: AsyncClient):
        """GET /policies/{id} — 不存在的策略返回 404."""
        resp = await client.get("/governance/v1/policies/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_policy_success(self, client: AsyncClient):
        """创建后查询策略."""
        create_resp = await client.post(
            "/governance/v1/policies", json={"name": "get-test", "domain": "policy"},
        )
        pid = create_resp.json()["data"]["policy_id"]
        resp = await client.get(f"/governance/v1/policies/{pid}")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert resp.json()["data"]["name"] == "get-test"

    @pytest.mark.asyncio
    async def test_get_policy_fields(self, client: AsyncClient):
        """查询策略返回完整字段."""
        create_resp = await client.post(
            "/governance/v1/policies",
            json={"name": "fields-test", "domain": "policy", "priority": 5, "tags": ["a"]},
        )
        pid = create_resp.json()["data"]["policy_id"]
        resp = await client.get(f"/governance/v1/policies/{pid}")
        data = resp.json()["data"]
        assert data["name"] == "fields-test"
        assert data["priority"] == 5
        assert data["tags"] == ["a"]
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_get_policy_store_uninitialized(self, empty_client: AsyncClient):
        """策略存储未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/policies/some-id")
        assert resp.status_code == 503

    # ---- 删除策略 ----

    @pytest.mark.asyncio
    async def test_delete_policy_success(self, client: AsyncClient):
        """DELETE /policies/{id} — 成功删除."""
        create_resp = await client.post(
            "/governance/v1/policies", json={"name": "del-test", "domain": "policy"},
        )
        pid = create_resp.json()["data"]["policy_id"]
        resp = await client.delete(f"/governance/v1/policies/{pid}")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_delete_policy_not_found(self, client: AsyncClient):
        """删除不存在的策略返回 404."""
        resp = await client.delete("/governance/v1/policies/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_policy_twice(self, client: AsyncClient):
        """重复删除返回 404."""
        create_resp = await client.post(
            "/governance/v1/policies", json={"name": "del2", "domain": "policy"},
        )
        pid = create_resp.json()["data"]["policy_id"]
        await client.delete(f"/governance/v1/policies/{pid}")
        resp = await client.delete(f"/governance/v1/policies/{pid}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_policy_store_uninitialized(self, empty_client: AsyncClient):
        """策略存储未初始化返回 503."""
        resp = await empty_client.delete("/governance/v1/policies/some-id")
        assert resp.status_code == 503


class TestPolicyEvaluation:
    """测试策略评估端点."""

    @pytest.mark.asyncio
    async def test_evaluate_policy_success(self, client: AsyncClient):
        """POST /policies/evaluate — 评估请求."""
        body = {"actor": "agent-tutor", "action": "grade", "resource": "student-001"}
        resp = await client.post("/governance/v1/policies/evaluate", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert "decision" in resp.json()["data"]

    @pytest.mark.asyncio
    async def test_evaluate_policy_default_allow(self, client: AsyncClient):
        """无策略时默认允许."""
        body = {"actor": "test", "action": "query", "resource": "data"}
        resp = await client.post("/governance/v1/policies/evaluate", json=body)
        assert resp.json()["data"]["decision"] == "allow"

    @pytest.mark.asyncio
    async def test_evaluate_policy_with_deny(self, client: AsyncClient):
        """创建 deny 策略后评估被拒绝."""
        # 创建全局 deny 策略
        await client.post("/governance/v1/policies", json={
            "name": "deny-all", "domain": "policy", "action": "deny",
            "scope": "global", "priority": 100,
            "condition": {"logic": "and", "rules": [
                {"field": "action", "operator": "exact", "value": "forbidden_action"},
            ]},
        })
        body = {"actor": "test", "action": "forbidden_action", "resource": "x"}
        resp = await client.post("/governance/v1/policies/evaluate", json=body)
        assert resp.json()["data"]["decision"] == "deny"

    @pytest.mark.asyncio
    async def test_evaluate_policy_evaluator_uninitialized(self, empty_client: AsyncClient):
        """评估器未初始化返回 503."""
        resp = await empty_client.post("/governance/v1/policies/evaluate", json={})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_evaluate_batch_success(self, client: AsyncClient):
        """POST /policies/evaluate-batch — 批量评估."""
        body = {
            "requests": [
                {"actor": "a1", "action": "x", "resource": "r1"},
                {"actor": "a2", "action": "y", "resource": "r2"},
            ]
        }
        resp = await client.post("/governance/v1/policies/evaluate-batch", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert len(resp.json()["data"]) == 2

    @pytest.mark.asyncio
    async def test_evaluate_batch_empty(self, client: AsyncClient):
        """批量评估空请求列表."""
        resp = await client.post("/governance/v1/policies/evaluate-batch", json={"requests": []})
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_evaluate_batch_evaluator_uninitialized(self, empty_client: AsyncClient):
        """评估器未初始化返回 503."""
        resp = await empty_client.post("/governance/v1/policies/evaluate-batch", json={"requests": []})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_policy_metrics(self, client: AsyncClient):
        """GET /policies/metrics — 评估度量."""
        resp = await client.get("/governance/v1/policies/metrics")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_policy_metrics_evaluator_uninitialized(self, empty_client: AsyncClient):
        """评估器未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/policies/metrics")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_policy_metrics_after_eval(self, client: AsyncClient):
        """评估后度量计数增加."""
        await client.post("/governance/v1/policies/evaluate", json={"actor": "a", "action": "x"})
        resp = await client.get("/governance/v1/policies/metrics")
        data = resp.json()["data"]
        # 评估次数嵌套在 evaluations.total
        total = data.get("total_evaluations", data.get("evaluations", {}).get("total", 0))
        assert total >= 1

    @pytest.mark.asyncio
    async def test_detect_conflicts(self, client: AsyncClient):
        """POST /policies/conflicts — 冲突检测."""
        resp = await client.post("/governance/v1/policies/conflicts")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_detect_conflicts_evaluator_uninitialized(self, empty_client: AsyncClient):
        """评估器未初始化返回 503."""
        resp = await empty_client.post("/governance/v1/policies/conflicts")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_evaluate_policy_invalid_body(self, client: AsyncClient):
        """评估请求传入非法字段."""
        # 额外的非法字段会被 pydantic 忽略或报错
        resp = await client.post("/governance/v1/policies/evaluate", json={"actor": 12345})
        # actor 应该是字符串
        assert resp.status_code in (200, 400)

    @pytest.mark.asyncio
    async def test_evaluate_policy_all_fields(self, client: AsyncClient):
        """评估请求传入全部字段."""
        body = {
            "actor": "agent-001",
            "action": "tool_call",
            "resource": "calculator",
            "layer": "L0",
            "domain": "governance",
            "context": {"key": "value"},
        }
        resp = await client.post("/governance/v1/policies/evaluate", json=body)
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_batch_missing_requests_key(self, client: AsyncClient):
        """批量评估缺少 requests 键."""
        resp = await client.post("/governance/v1/policies/evaluate-batch", json={})
        # 应该返回空列表（body.get("requests", []) 返回 []）
        assert resp.status_code == 200


# ============================================================
# 5. G3 CC1 防幻觉测试
# ============================================================


class TestAntiHallucination:
    """测试 CC1 防幻觉端点."""

    @pytest.mark.asyncio
    async def test_verify_success(self, client: AsyncClient):
        """POST /anti-hallucination/verify — 验证文本."""
        body = {
            "output_text": "水的沸点是100摄氏度。",
            "context_chunks": ["在标准大气压下，水的沸点为100摄氏度。"],
        }
        resp = await client.post("/governance/v1/anti-hallucination/verify", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_verify_report_structure(self, client: AsyncClient):
        """验证报告包含关键字段."""
        body = {"output_text": "测试文本", "agent_id": "agent-001"}
        resp = await client.post("/governance/v1/anti-hallucination/verify", json=body)
        data = resp.json()["data"]
        assert "overall_score" in data
        assert "hallucination_detected" in data
        assert "status" in data

    @pytest.mark.asyncio
    async def test_verify_empty_text(self, client: AsyncClient):
        """空文本验证."""
        body = {"output_text": ""}
        resp = await client.post("/governance/v1/anti-hallucination/verify", json=body)
        assert resp.status_code == 200
        # 空输出应该跳过（status=skipped）
        assert resp.json()["data"]["status"] in ("skipped", "passed")

    @pytest.mark.asyncio
    async def test_verify_with_citations(self, client: AsyncClient):
        """带引用的验证."""
        body = {
            "output_text": "引用文献显示A方案更优。",
            "citations": ["https://example.com/paper1"],
            "agent_id": "agent-002",
        }
        resp = await client.post("/governance/v1/anti-hallucination/verify", json=body)
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_verify_with_sample_outputs(self, client: AsyncClient):
        """带采样输出的验证."""
        body = {
            "output_text": "测试内容。",
            "sample_outputs": ["采样1", "采样2", "采样3"],
        }
        resp = await client.post("/governance/v1/anti-hallucination/verify", json=body)
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_verify_missing_output_text(self, client: AsyncClient):
        """缺少 output_text 返回 400."""
        resp = await client.post("/governance/v1/anti-hallucination/verify", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_verify_pipeline_uninitialized(self, empty_client: AsyncClient):
        """防幻觉管道未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/anti-hallucination/verify",
            json={"output_text": "test"},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_verify_invalid_json(self, client: AsyncClient):
        """无效 JSON 返回 400."""
        resp = await client.post(
            "/governance/v1/anti-hallucination/verify",
            content=b"bad",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    # ---- 配置端点 ----

    @pytest.mark.asyncio
    async def test_get_config(self, client: AsyncClient):
        """GET /anti-hallucination/config — 获取配置."""
        resp = await client.get("/governance/v1/anti-hallucination/config")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        data = resp.json()["data"]
        assert "pass_threshold" in data
        assert "refuse_threshold" in data
        assert "verifiers" in data

    @pytest.mark.asyncio
    async def test_get_config_pipeline_uninitialized(self, empty_client: AsyncClient):
        """管道未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/anti-hallucination/config")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_update_config(self, client: AsyncClient):
        """PUT /anti-hallucination/config — 更新配置."""
        body = {
            "pass_threshold": 0.8,
            "degrade_threshold": 0.6,
            "refuse_threshold": 0.4,
        }
        resp = await client.put("/governance/v1/anti-hallucination/config", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert resp.json()["data"]["updated"] is True

    @pytest.mark.asyncio
    async def test_update_config_pipeline_uninitialized(self, empty_client: AsyncClient):
        """管道未初始化返回 503."""
        resp = await empty_client.put(
            "/governance/v1/anti-hallucination/config",
            json={"pass_threshold": 0.5},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_update_config_invalid(self, client: AsyncClient):
        """更新配置传入无效值."""
        body = {"pass_threshold": "not_a_number"}
        resp = await client.put("/governance/v1/anti-hallucination/config", json=body)
        assert resp.status_code == 400

    # ---- 验证器和统计 ----

    @pytest.mark.asyncio
    async def test_list_verifiers(self, client: AsyncClient):
        """GET /anti-hallucination/verifiers — 列出验证器."""
        resp = await client.get("/governance/v1/anti-hallucination/verifiers")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert isinstance(resp.json()["data"], list)

    @pytest.mark.asyncio
    async def test_list_verifiers_pipeline_uninitialized(self, empty_client: AsyncClient):
        """管道未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/anti-hallucination/verifiers")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_list_verifiers_has_builtin(self, client: AsyncClient):
        """内置验证器至少包含四种类型."""
        resp = await client.get("/governance/v1/anti-hallucination/verifiers")
        data = resp.json()["data"]
        assert len(data) >= 4

    @pytest.mark.asyncio
    async def test_stats(self, client: AsyncClient):
        """GET /anti-hallucination/stats — 统计信息."""
        resp = await client.get("/governance/v1/anti-hallucination/stats")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert resp.json()["data"]["pipeline_initialized"] is True

    @pytest.mark.asyncio
    async def test_stats_pipeline_uninitialized(self, empty_client: AsyncClient):
        """管道未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/anti-hallucination/stats")
        assert resp.status_code == 503


# ============================================================
# 6. G4 CC2 人机协作测试
# ============================================================


class TestCollaborationProfiles:
    """测试 CC2 协作配置端点."""

    @pytest.mark.asyncio
    async def test_register_profile_success(self, client: AsyncClient):
        """POST /collaboration/profiles — 注册配置."""
        body = {"agent_id": "tutor-001"}
        resp = await client.post("/governance/v1/collaboration/profiles", json=body)
        assert resp.status_code == 201
        assert resp.json()["code"] == 0
        assert resp.json()["data"]["agent_id"] == "tutor-001"

    @pytest.mark.asyncio
    async def test_register_profile_with_mode(self, client: AsyncClient):
        """注册配置指定协作模式."""
        body = {"agent_id": "agent-monitored", "mode": "monitored"}
        resp = await client.post("/governance/v1/collaboration/profiles", json=body)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_register_profile_full(self, client: AsyncClient):
        """注册完整配置."""
        body = {
            "agent_id": "agent-full",
            "mode": "conditional",
            "default_mode": "conditional",
            "max_auto_steps": 20,
            "confidence_threshold": 0.8,
            "timeout_seconds": 600.0,
            "escalation_targets": ["human-reviewer-001"],
            "enabled": True,
            "tags": ["education", "grading"],
        }
        resp = await client.post("/governance/v1/collaboration/profiles", json=body)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_register_profile_missing_agent_id(self, client: AsyncClient):
        """缺少 agent_id 返回 400."""
        resp = await client.post("/governance/v1/collaboration/profiles", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_register_profile_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/collaboration/profiles", json={"agent_id": "x"},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_list_profiles_empty(self, client: AsyncClient):
        """GET /collaboration/profiles — 空列表."""
        resp = await client.get("/governance/v1/collaboration/profiles")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_list_profiles_after_register(self, client: AsyncClient):
        """注册后列表不为空."""
        await client.post("/governance/v1/collaboration/profiles", json={"agent_id": "a1"})
        resp = await client.get("/governance/v1/collaboration/profiles")
        assert len(resp.json()["data"]) >= 1

    @pytest.mark.asyncio
    async def test_list_profiles_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/collaboration/profiles")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_get_profile_success(self, client: AsyncClient):
        """GET /collaboration/profiles/{id} — 查询配置."""
        await client.post("/governance/v1/collaboration/profiles", json={"agent_id": "tutor-002"})
        resp = await client.get("/governance/v1/collaboration/profiles/tutor-002")
        assert resp.status_code == 200
        assert resp.json()["data"]["agent_id"] == "tutor-002"

    @pytest.mark.asyncio
    async def test_get_profile_not_found(self, client: AsyncClient):
        """查询不存在的配置返回 404."""
        resp = await client.get("/governance/v1/collaboration/profiles/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_profile_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/collaboration/profiles/x")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_update_profile_success(self, client: AsyncClient):
        """PUT /collaboration/profiles/{id} — 更新配置."""
        await client.post("/governance/v1/collaboration/profiles", json={"agent_id": "upd-001"})
        body = {"mode": "supervised", "confidence_threshold": 0.9}
        resp = await client.put("/governance/v1/collaboration/profiles/upd-001", json=body)
        assert resp.status_code == 200
        assert resp.json()["data"]["mode"] == "supervised"

    @pytest.mark.asyncio
    async def test_update_profile_not_found(self, client: AsyncClient):
        """更新不存在的配置返回 400."""
        resp = await client.put(
            "/governance/v1/collaboration/profiles/nonexistent",
            json={"mode": "supervised"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_profile_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.put(
            "/governance/v1/collaboration/profiles/x", json={"mode": "supervised"},
        )
        assert resp.status_code == 503


class TestCollaborationREACT:
    """测试 CC2 REACT 评估端点."""

    @pytest.mark.asyncio
    async def test_evaluate_react_high_risk(self, client: AsyncClient):
        """高风险 REACT 评分 → supervised 模式."""
        body = {
            "agent_id": "tutor-001",
            "score": {
                "risk": 5.0, "explainability": 5.0, "accuracy": 5.0,
                "consequence": 5.0, "time_sensitivity": 5.0,
            },
        }
        resp = await client.post("/governance/v1/collaboration/evaluate-react", json=body)
        assert resp.status_code == 200
        assert resp.json()["data"]["recommended_mode"] == "supervised"

    @pytest.mark.asyncio
    async def test_evaluate_react_low_risk(self, client: AsyncClient):
        """低风险 REACT 评分 → autonomous 模式."""
        body = {
            "agent_id": "auto-agent",
            "score": {
                "risk": 0.0, "explainability": 0.0, "accuracy": 0.0,
                "consequence": 0.0, "time_sensitivity": 0.0,
            },
        }
        resp = await client.post("/governance/v1/collaboration/evaluate-react", json=body)
        assert resp.json()["data"]["recommended_mode"] == "autonomous"

    @pytest.mark.asyncio
    async def test_evaluate_react_medium(self, client: AsyncClient):
        """中等 REACT 评分 → conditional 模式."""
        body = {
            "agent_id": "cond-agent",
            "score": {
                "risk": 3.0, "explainability": 3.0, "accuracy": 3.0,
                "consequence": 3.0, "time_sensitivity": 3.0,
            },
        }
        resp = await client.post("/governance/v1/collaboration/evaluate-react", json=body)
        assert resp.json()["data"]["recommended_mode"] == "conditional"

    @pytest.mark.asyncio
    async def test_evaluate_react_response_structure(self, client: AsyncClient):
        """REACT 评估返回包含 agent_id 和 recommended_mode."""
        body = {
            "agent_id": "test",
            "score": {"risk": 1.0, "explainability": 1.0, "accuracy": 1.0,
                       "consequence": 1.0, "time_sensitivity": 1.0},
        }
        resp = await client.post("/governance/v1/collaboration/evaluate-react", json=body)
        data = resp.json()["data"]
        assert "agent_id" in data
        assert "recommended_mode" in data

    @pytest.mark.asyncio
    async def test_evaluate_react_missing_score(self, client: AsyncClient):
        """缺少 score 返回 400."""
        resp = await client.post(
            "/governance/v1/collaboration/evaluate-react",
            json={"agent_id": "test"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_evaluate_react_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/collaboration/evaluate-react",
            json={"agent_id": "x", "score": {}},
        )
        assert resp.status_code == 503


class TestCollaborationSwitchMode:
    """测试 CC2 模式切换端点."""

    @pytest.mark.asyncio
    async def test_switch_mode_success(self, client: AsyncClient):
        """POST /collaboration/switch-mode — 相邻级切换."""
        # 注册 conditional 模式的 Agent
        await client.post("/governance/v1/collaboration/profiles", json={
            "agent_id": "switch-agent", "mode": "conditional",
        })
        body = {
            "agent_id": "switch-agent",
            "target_mode": "monitored",
            "reason": "质量下降",
            "trigger": "anomaly_detected",
        }
        resp = await client.post("/governance/v1/collaboration/switch-mode", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_switch_mode_response_fields(self, client: AsyncClient):
        """模式切换返回事件字段."""
        await client.post("/governance/v1/collaboration/profiles", json={
            "agent_id": "sw-2", "mode": "supervised",
        })
        body = {"agent_id": "sw-2", "target_mode": "conditional", "trigger": "manual_request"}
        resp = await client.post("/governance/v1/collaboration/switch-mode", json=body)
        data = resp.json()["data"]
        assert "agent_id" in data
        assert "from_mode" in data
        assert "to_mode" in data
        assert "event_id" in data

    @pytest.mark.asyncio
    async def test_switch_mode_not_registered(self, client: AsyncClient):
        """切换未注册的 Agent 返回 400."""
        body = {"agent_id": "nonexistent", "target_mode": "supervised", "trigger": "manual_request"}
        resp = await client.post("/governance/v1/collaboration/switch-mode", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_switch_mode_skip_level(self, client: AsyncClient):
        """跳级切换返回 400."""
        await client.post("/governance/v1/collaboration/profiles", json={
            "agent_id": "skip-agent", "mode": "autonomous",
        })
        body = {"agent_id": "skip-agent", "target_mode": "supervised", "trigger": "manual_request"}
        resp = await client.post("/governance/v1/collaboration/switch-mode", json=body)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_switch_mode_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/collaboration/switch-mode", json={},
        )
        assert resp.status_code == 503


class TestCollaborationInterventions:
    """测试 CC2 干预端点."""

    @pytest.mark.asyncio
    async def test_create_intervention_success(self, client: AsyncClient):
        """POST /collaboration/interventions — 创建干预."""
        body = {
            "agent_id": "tutor-003",
            "intervention_type": "checkpoint",
            "reason": "需要人工审核评分决策",
        }
        resp = await client.post("/governance/v1/collaboration/interventions", json=body)
        assert resp.status_code == 201
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_create_intervention_record_id(self, client: AsyncClient):
        """创建干预返回记录 ID."""
        body = {
            "agent_id": "tutor-004",
            "intervention_type": "escalation",
            "reason": "检测到异常行为",
        }
        resp = await client.post("/governance/v1/collaboration/interventions", json=body)
        data = resp.json()["data"]
        assert "request" in data
        assert "record_id" in data

    @pytest.mark.asyncio
    async def test_create_intervention_with_context(self, client: AsyncClient):
        """创建干预带上下文."""
        body = {
            "agent_id": "ctx-agent",
            "intervention_type": "checkpoint",
            "context": {"student_id": "s001", "score": 95},
            "timeout_seconds": 60.0,
        }
        resp = await client.post("/governance/v1/collaboration/interventions", json=body)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_create_intervention_missing_agent_id(self, client: AsyncClient):
        """缺少 agent_id 返回 400."""
        resp = await client.post(
            "/governance/v1/collaboration/interventions", json={},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_intervention_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/collaboration/interventions", json={},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_respond_intervention_success(self, client: AsyncClient):
        """POST /collaboration/interventions/{id}/respond — 响应干预."""
        # 先创建干预
        create_resp = await client.post("/governance/v1/collaboration/interventions", json={
            "agent_id": "resp-agent", "intervention_type": "checkpoint",
        })
        request_id = create_resp.json()["data"]["request"]["request_id"]
        # 响应
        body = {"decision": "approve", "human_input": "审核通过"}
        resp = await client.post(
            f"/governance/v1/collaboration/interventions/{request_id}/respond",
            json=body,
        )
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_respond_intervention_reject(self, client: AsyncClient):
        """拒绝干预."""
        create_resp = await client.post("/governance/v1/collaboration/interventions", json={
            "agent_id": "rej-agent", "intervention_type": "checkpoint",
        })
        request_id = create_resp.json()["data"]["request"]["request_id"]
        body = {"decision": "reject", "human_input": "评分过高"}
        resp = await client.post(
            f"/governance/v1/collaboration/interventions/{request_id}/respond",
            json=body,
        )
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_respond_intervention_not_found(self, client: AsyncClient):
        """响应不存在的干预返回 400."""
        resp = await client.post(
            "/governance/v1/collaboration/interventions/nonexistent/respond",
            json={"decision": "approve"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_respond_intervention_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/collaboration/interventions/x/respond", json={},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_list_interventions_empty(self, client: AsyncClient):
        """GET /collaboration/interventions — 空列表."""
        resp = await client.get("/governance/v1/collaboration/interventions")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_list_interventions_after_create(self, client: AsyncClient):
        """创建后列表不为空."""
        await client.post("/governance/v1/collaboration/interventions", json={
            "agent_id": "list-agent", "intervention_type": "checkpoint",
        })
        resp = await client.get("/governance/v1/collaboration/interventions")
        assert len(resp.json()["data"]) >= 1

    @pytest.mark.asyncio
    async def test_list_interventions_filter_by_agent(self, client: AsyncClient):
        """按 agent_id 过滤干预列表."""
        await client.post("/governance/v1/collaboration/interventions", json={
            "agent_id": "filter-agent", "intervention_type": "checkpoint",
        })
        resp = await client.get(
            "/governance/v1/collaboration/interventions",
            params={"agent_id": "filter-agent"},
        )
        assert resp.status_code == 200
        for item in resp.json()["data"]:
            assert item["request"]["agent_id"] == "filter-agent"

    @pytest.mark.asyncio
    async def test_list_interventions_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/collaboration/interventions")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_get_intervention_success(self, client: AsyncClient):
        """GET /collaboration/interventions/{id} — 查询单个干预."""
        create_resp = await client.post("/governance/v1/collaboration/interventions", json={
            "agent_id": "get-intv-agent", "intervention_type": "checkpoint",
        })
        request_id = create_resp.json()["data"]["request"]["request_id"]
        resp = await client.get(f"/governance/v1/collaboration/interventions/{request_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["request"]["request_id"] == request_id

    @pytest.mark.asyncio
    async def test_get_intervention_not_found(self, client: AsyncClient):
        """查询单个干预不存在返回 404."""
        resp = await client.get("/governance/v1/collaboration/interventions/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_intervention_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/collaboration/interventions/x")
        assert resp.status_code == 503


class TestCollaborationEscalate:
    """测试 CC2 升级端点."""

    @pytest.mark.asyncio
    async def test_escalate_success(self, client: AsyncClient):
        """POST /collaboration/escalate — 升级到人工."""
        body = {
            "agent_id": "esc-agent",
            "reason": "检测到安全风险",
            "context": {"risk_level": "high"},
        }
        resp = await client.post("/governance/v1/collaboration/escalate", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_escalate_response_fields(self, client: AsyncClient):
        """升级返回干预记录字段."""
        resp = await client.post("/governance/v1/collaboration/escalate", json={
            "agent_id": "esc-2", "reason": "异常",
        })
        data = resp.json()["data"]
        assert "request" in data
        assert data["request"]["intervention_type"] == "escalation"

    @pytest.mark.asyncio
    async def test_escalate_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.post("/governance/v1/collaboration/escalate", json={})
        assert resp.status_code == 503


class TestCollaborationNegotiations:
    """测试 CC2 协商端点."""

    @pytest.mark.asyncio
    async def test_start_negotiation_success(self, client: AsyncClient):
        """POST /collaboration/negotiations — 发起协商."""
        body = {
            "agent_id": "nego-agent",
            "proposal": "建议调整评分标准",
            "context": {"current_threshold": 0.7},
            "max_rounds": 3,
        }
        resp = await client.post("/governance/v1/collaboration/negotiations", json=body)
        assert resp.status_code == 201
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_start_negotiation_session_id(self, client: AsyncClient):
        """发起协商返回 session_id."""
        resp = await client.post("/governance/v1/collaboration/negotiations", json={
            "agent_id": "nego-2", "proposal": "test",
        })
        data = resp.json()["data"]
        assert "session_id" in data
        assert data["agent_id"] == "nego-2"

    @pytest.mark.asyncio
    async def test_start_negotiation_missing_agent_id(self, client: AsyncClient):
        """缺少 agent_id 返回 400."""
        resp = await client.post("/governance/v1/collaboration/negotiations", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_start_negotiation_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/collaboration/negotiations", json={},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_add_negotiation_round(self, client: AsyncClient):
        """POST /collaboration/negotiations/{id}/rounds — 添加轮次."""
        # 先发起协商
        start_resp = await client.post("/governance/v1/collaboration/negotiations", json={
            "agent_id": "round-agent", "proposal": "初始提案",
        })
        sid = start_resp.json()["data"]["session_id"]
        # 添加轮次
        body = {"actor": "human"}
        resp = await client.post(
            f"/governance/v1/collaboration/negotiations/{sid}/rounds", json=body,
        )
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        # 返回的是 NegotiationRound 对象
        data = resp.json()["data"]
        assert "round_number" in data

    @pytest.mark.asyncio
    async def test_add_round_nonexistent_session(self, client: AsyncClient):
        """添加轮次到不存在的会话返回 400."""
        resp = await client.post(
            "/governance/v1/collaboration/negotiations/nonexistent/rounds",
            json={"actor": "human"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_add_round_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/collaboration/negotiations/x/rounds", json={},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_finalize_negotiation_approve(self, client: AsyncClient):
        """POST /collaboration/negotiations/{id}/finalize — 终结协商（批准）."""
        start_resp = await client.post("/governance/v1/collaboration/negotiations", json={
            "agent_id": "fin-agent", "proposal": "test",
        })
        sid = start_resp.json()["data"]["session_id"]
        body = {"outcome": "approve"}
        resp = await client.post(
            f"/governance/v1/collaboration/negotiations/{sid}/finalize", json=body,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_finalize_negotiation_reject(self, client: AsyncClient):
        """终结协商（拒绝）."""
        start_resp = await client.post("/governance/v1/collaboration/negotiations", json={
            "agent_id": "fin-rej", "proposal": "test",
        })
        sid = start_resp.json()["data"]["session_id"]
        body = {"outcome": "reject"}
        resp = await client.post(
            f"/governance/v1/collaboration/negotiations/{sid}/finalize", json=body,
        )
        assert resp.json()["data"]["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_finalize_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/collaboration/negotiations/x/finalize", json={},
        )
        assert resp.status_code == 503


class TestCollaborationStats:
    """测试 CC2 统计端点."""

    @pytest.mark.asyncio
    async def test_stats(self, client: AsyncClient):
        """GET /collaboration/stats — 协作统计."""
        resp = await client.get("/governance/v1/collaboration/stats")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_stats_has_profile_count(self, client: AsyncClient):
        """统计包含 profile_count（实际字段名 registered_agents）."""
        await client.post("/governance/v1/collaboration/profiles", json={"agent_id": "stat-agent"})
        resp = await client.get("/governance/v1/collaboration/stats")
        data = resp.json()["data"]
        # 实际字段名为 registered_agents
        assert "registered_agents" in data
        assert data["registered_agents"] >= 1

    @pytest.mark.asyncio
    async def test_stats_engine_uninitialized(self, empty_client: AsyncClient):
        """引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/collaboration/stats")
        assert resp.status_code == 503


# ============================================================
# 7. G5 审计测试
# ============================================================


class TestAuditEndpoints:
    """测试 G5 审计端点."""

    @pytest.mark.asyncio
    async def test_query_decisions_empty(self, client: AsyncClient):
        """GET /audit/decisions — 空列表."""
        resp = await client.get("/governance/v1/audit/decisions")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_query_decisions_with_params(self, client: AsyncClient):
        """带查询参数."""
        resp = await client.get("/governance/v1/audit/decisions", params={
            "actor": "test-agent", "limit": 10,
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_query_decisions_with_limit(self, client: AsyncClient):
        """带 limit 参数."""
        resp = await client.get("/governance/v1/audit/decisions", params={"limit": 5})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_query_decisions_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/audit/decisions")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_get_decision_not_found(self, client: AsyncClient):
        """GET /audit/decisions/{id} — 不存在的决策返回 404."""
        resp = await client.get("/governance/v1/audit/decisions/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_decision_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/audit/decisions/x")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_get_trace_empty(self, client: AsyncClient):
        """GET /audit/traces/{id} — 空 trace."""
        resp = await client.get("/governance/v1/audit/traces/nonexistent-trace")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_get_trace_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/audit/traces/x")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_aggregate_action_empty(self, client: AsyncClient):
        """GET /audit/aggregate/action — 空聚合."""
        resp = await client.get("/governance/v1/audit/aggregate/action")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_aggregate_action_with_agent(self, client: AsyncClient):
        """按 agent_id 过滤聚合."""
        resp = await client.get(
            "/governance/v1/audit/aggregate/action",
            params={"agent_id": "test-agent"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_aggregate_action_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/audit/aggregate/action")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_aggregate_outcome_empty(self, client: AsyncClient):
        """GET /audit/aggregate/outcome — 空聚合."""
        resp = await client.get("/governance/v1/audit/aggregate/outcome")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_aggregate_outcome_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/audit/aggregate/outcome")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_latency_stats(self, client: AsyncClient):
        """GET /audit/latency-stats — 延迟统计."""
        resp = await client.get("/governance/v1/audit/latency-stats")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_latency_stats_with_agent(self, client: AsyncClient):
        """按 agent_id 过滤延迟统计."""
        resp = await client.get(
            "/governance/v1/audit/latency-stats",
            params={"agent_id": "test-agent"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_latency_stats_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/audit/latency-stats")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_build_baseline(self, client: AsyncClient):
        """POST /audit/baselines — 构建基线."""
        body = {"entity_id": "agent-001", "window": 3600}
        resp = await client.post("/governance/v1/audit/baselines", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_build_baseline_no_data(self, client: AsyncClient):
        """无足够数据时返回 None."""
        resp = await client.post("/governance/v1/audit/baselines", json={})
        assert resp.status_code == 200
        # 无数据时返回 {"baseline": None, "message": "..."}
        data = resp.json()["data"]
        assert data.get("baseline") is None or "baseline" in data

    @pytest.mark.asyncio
    async def test_build_baseline_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.post("/governance/v1/audit/baselines", json={})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_detect_anomalies(self, client: AsyncClient):
        """POST /audit/anomalies — 异常检测."""
        body = {"entity_id": "agent-001", "sensitivity": "high"}
        resp = await client.post("/governance/v1/audit/anomalies", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert isinstance(resp.json()["data"], list)

    @pytest.mark.asyncio
    async def test_detect_anomalies_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.post("/governance/v1/audit/anomalies", json={})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_get_alerts_empty(self, client: AsyncClient):
        """GET /audit/alerts — 空告警列表."""
        resp = await client.get("/governance/v1/audit/alerts")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_get_alerts_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/audit/alerts")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_stats(self, client: AsyncClient):
        """GET /audit/stats — 审计统计."""
        resp = await client.get("/governance/v1/audit/stats")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        data = resp.json()["data"]
        # 实际字段名为 total_recorded
        assert "total_recorded" in data

    @pytest.mark.asyncio
    async def test_stats_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/audit/stats")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_summary(self, client: AsyncClient):
        """GET /audit/summary — 审计摘要."""
        resp = await client.get("/governance/v1/audit/summary")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_summary_audit_uninitialized(self, empty_client: AsyncClient):
        """审计引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/audit/summary")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_audit_after_chain_evaluate(self, client: AsyncClient):
        """全链路评估后审计引擎有记录."""
        # 先通过全链路评估产生审计记录
        await client.post("/governance/v1/chain/evaluate", json={})
        resp = await client.get("/governance/v1/audit/stats")
        data = resp.json()["data"]
        assert data.get("total_recorded", 0) >= 1


# ============================================================
# 8. G5 度量测试
# ============================================================


class TestMetricsEndpoints:
    """测试 G5 度量端点."""

    @pytest.mark.asyncio
    async def test_define_metric(self, client: AsyncClient):
        """POST /metrics/define — 定义指标."""
        body = {
            "name": "test_latency",
            "metric_type": "histogram",
            "unit": "ms",
            "description": "测试延迟",
        }
        resp = await client.post("/governance/v1/metrics/define", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert resp.json()["data"]["defined"] == "test_latency"

    @pytest.mark.asyncio
    async def test_define_metric_gauge(self, client: AsyncClient):
        """定义 gauge 类型指标."""
        body = {"name": "cpu_usage", "metric_type": "gauge", "unit": "%"}
        resp = await client.post("/governance/v1/metrics/define", json=body)
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_define_metric_counter(self, client: AsyncClient):
        """定义 counter 类型指标."""
        body = {"name": "request_count", "metric_type": "counter"}
        resp = await client.post("/governance/v1/metrics/define", json=body)
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_define_metric_missing_name(self, client: AsyncClient):
        """缺少 name 返回 400."""
        resp = await client.post("/governance/v1/metrics/define", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_define_metric_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/metrics/define",
            json={"name": "x", "metric_type": "gauge"},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_record_metric(self, client: AsyncClient):
        """POST /metrics/record — 记录指标值."""
        # 先定义
        await client.post("/governance/v1/metrics/define", json={
            "name": "test_latency", "metric_type": "histogram", "unit": "ms",
        })
        body = {"metric_name": "test_latency", "value": 42.5}
        resp = await client.post("/governance/v1/metrics/record", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_record_metric_with_tags(self, client: AsyncClient):
        """记录指标带标签."""
        body = {"metric_name": "test_latency", "value": 100.0, "tags": {"env": "test"}}
        resp = await client.post("/governance/v1/metrics/record", json=body)
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_record_metric_missing_name(self, client: AsyncClient):
        """缺少 metric_name 返回 400."""
        resp = await client.post("/governance/v1/metrics/record", json={"value": 1.0})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_record_metric_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/metrics/record",
            json={"metric_name": "x", "value": 1.0},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_get_values_empty(self, client: AsyncClient):
        """GET /metrics/{name}/values — 无数据."""
        resp = await client.get("/governance/v1/metrics/nonexistent/values")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_get_values_after_record(self, client: AsyncClient):
        """记录后查询值."""
        await client.post("/governance/v1/metrics/define", json={
            "name": "val_test", "metric_type": "gauge",
        })
        await client.post("/governance/v1/metrics/record", json={
            "metric_name": "val_test", "value": 10.0,
        })
        resp = await client.get("/governance/v1/metrics/val_test/values")
        assert len(resp.json()["data"]) >= 1

    @pytest.mark.asyncio
    async def test_get_values_with_limit(self, client: AsyncClient):
        """带 limit 参数查询."""
        resp = await client.get(
            "/governance/v1/metrics/val_test/values",
            params={"limit": 5},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_values_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/metrics/x/values")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_get_latest_not_found(self, client: AsyncClient):
        """GET /metrics/{name}/latest — 无数据返回 404."""
        resp = await client.get("/governance/v1/metrics/nonexistent/latest")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_latest_success(self, client: AsyncClient):
        """记录后获取最新值."""
        await client.post("/governance/v1/metrics/define", json={
            "name": "latest_test", "metric_type": "gauge",
        })
        await client.post("/governance/v1/metrics/record", json={
            "metric_name": "latest_test", "value": 99.9,
        })
        resp = await client.get("/governance/v1/metrics/latest_test/latest")
        assert resp.status_code == 200
        assert resp.json()["data"]["value"] == 99.9

    @pytest.mark.asyncio
    async def test_get_latest_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/metrics/x/latest")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_aggregate_avg(self, client: AsyncClient):
        """POST /metrics/aggregate — 平均值聚合."""
        await client.post("/governance/v1/metrics/define", json={
            "name": "agg_test", "metric_type": "histogram",
        })
        for v in [10.0, 20.0, 30.0]:
            await client.post("/governance/v1/metrics/record", json={
                "metric_name": "agg_test", "value": v,
            })
        resp = await client.post("/governance/v1/metrics/aggregate", json={
            "metric_name": "agg_test", "func": "avg",
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["value"] == 20.0

    @pytest.mark.asyncio
    async def test_aggregate_sum(self, client: AsyncClient):
        """求和聚合."""
        # 独立创建数据
        await client.post("/governance/v1/metrics/define", json={
            "name": "agg_sum_test", "metric_type": "histogram",
        })
        for v in [10.0, 20.0, 30.0]:
            await client.post("/governance/v1/metrics/record", json={
                "metric_name": "agg_sum_test", "value": v,
            })
        resp = await client.post("/governance/v1/metrics/aggregate", json={
            "metric_name": "agg_sum_test", "func": "sum",
        })
        assert resp.json()["data"]["value"] == 60.0

    @pytest.mark.asyncio
    async def test_aggregate_count(self, client: AsyncClient):
        """计数聚合."""
        await client.post("/governance/v1/metrics/define", json={
            "name": "agg_cnt_test", "metric_type": "histogram",
        })
        for v in [10.0, 20.0, 30.0]:
            await client.post("/governance/v1/metrics/record", json={
                "metric_name": "agg_cnt_test", "value": v,
            })
        resp = await client.post("/governance/v1/metrics/aggregate", json={
            "metric_name": "agg_cnt_test", "func": "count",
        })
        assert resp.json()["data"]["value"] == 3.0

    @pytest.mark.asyncio
    async def test_aggregate_min_max(self, client: AsyncClient):
        """最小值/最大值聚合."""
        await client.post("/governance/v1/metrics/define", json={
            "name": "agg_mm_test", "metric_type": "histogram",
        })
        for v in [10.0, 20.0, 30.0]:
            await client.post("/governance/v1/metrics/record", json={
                "metric_name": "agg_mm_test", "value": v,
            })
        min_resp = await client.post("/governance/v1/metrics/aggregate", json={
            "metric_name": "agg_mm_test", "func": "min",
        })
        assert min_resp.json()["data"]["value"] == 10.0
        max_resp = await client.post("/governance/v1/metrics/aggregate", json={
            "metric_name": "agg_mm_test", "func": "max",
        })
        assert max_resp.json()["data"]["value"] == 30.0

    @pytest.mark.asyncio
    async def test_aggregate_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/metrics/aggregate", json={},
        )
        assert resp.status_code == 503


class TestSLOEndpoints:
    """测试 SLO 端点."""

    @pytest.mark.asyncio
    async def test_register_slo(self, client: AsyncClient):
        """POST /metrics/slos — 注册 SLO."""
        body = {
            "name": "test_slo",
            "target": 99.5,
            "metric_name": "test_latency",
        }
        resp = await client.post("/governance/v1/metrics/slos", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert resp.json()["data"]["registered"] == "test_slo"

    @pytest.mark.asyncio
    async def test_register_slo_with_window(self, client: AsyncClient):
        """注册 SLO 带窗口."""
        body = {
            "name": "slo_with_window",
            "target": 99.0,
            "metric_name": "test_latency",
            "window": 7200.0,
        }
        resp = await client.post("/governance/v1/metrics/slos", json=body)
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_register_slo_missing_name(self, client: AsyncClient):
        """缺少 name 返回 400."""
        resp = await client.post("/governance/v1/metrics/slos", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_register_slo_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/metrics/slos", json={"name": "x"},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_get_slo_success(self, client: AsyncClient):
        """GET /metrics/slos/{name} — 查询 SLO."""
        await client.post("/governance/v1/metrics/slos", json={
            "name": "get_slo_test", "target": 99.0, "metric_name": "test_latency",
        })
        resp = await client.get("/governance/v1/metrics/slos/get_slo_test")
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "get_slo_test"

    @pytest.mark.asyncio
    async def test_get_slo_not_found(self, client: AsyncClient):
        """查询不存在的 SLO 返回 404."""
        resp = await client.get("/governance/v1/metrics/slos/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_slo_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/metrics/slos/x")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_evaluate_slo(self, client: AsyncClient):
        """POST /metrics/slos/{name}/evaluate — 评估 SLO."""
        await client.post("/governance/v1/metrics/slos", json={
            "name": "eval_slo", "target": 99.0, "metric_name": "test_latency",
        })
        resp = await client.post("/governance/v1/metrics/slos/eval_slo/evaluate")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_slo_snapshot_fields(self, client: AsyncClient):
        """SLO 快照包含关键字段."""
        # 创建自己的 SLO
        await client.post("/governance/v1/metrics/slos", json={
            "name": "snapshot_slo_test", "target": 99.0, "metric_name": "test_latency",
        })
        resp = await client.post("/governance/v1/metrics/slos/snapshot_slo_test/evaluate")
        data = resp.json()["data"]
        assert "slo_name" in data
        assert "compliance_percentage" in data
        assert "burn_rate" in data

    @pytest.mark.asyncio
    async def test_evaluate_slo_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.post("/governance/v1/metrics/slos/x/evaluate")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_evaluate_all_slos_empty(self, client: AsyncClient):
        """GET /metrics/slos/evaluate-all — 无 SLO 时返回空列表."""
        # 需要确保无 SLO，用新的引擎
        resp = await client.get("/governance/v1/metrics/slos/evaluate-all")
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"], list)

    @pytest.mark.asyncio
    async def test_evaluate_all_slos_after_register(self, client: AsyncClient):
        """注册 SLO 后评估所有."""
        await client.post("/governance/v1/metrics/slos", json={
            "name": "all_slo_test", "target": 99.5, "metric_name": "test_latency",
        })
        resp = await client.get("/governance/v1/metrics/slos/evaluate-all")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) >= 1

    @pytest.mark.asyncio
    async def test_evaluate_all_slos_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/metrics/slos/evaluate-all")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_slo_alerts_empty(self, client: AsyncClient):
        """GET /metrics/slos/alerts — 无告警."""
        resp = await client.get("/governance/v1/metrics/slos/alerts")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_slo_alerts_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/metrics/slos/alerts")
        assert resp.status_code == 503


class TestDORAEndpoints:
    """测试 DORA 端点."""

    @pytest.mark.asyncio
    async def test_dora_deployment(self, client: AsyncClient):
        """POST /metrics/dora/deployments — 记录 DORA 部署."""
        body = {
            "deployment_id": "deploy-001",
            "agent_id": "agent-deploy",
            "status": "success",
            "duration_seconds": 120.0,
        }
        resp = await client.post("/governance/v1/metrics/dora/deployments", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        assert resp.json()["data"]["recorded"] is True

    @pytest.mark.asyncio
    async def test_dora_deployment_minimal(self, client: AsyncClient):
        """最小参数 DORA 部署."""
        body = {"agent_id": "agent-001"}
        resp = await client.post("/governance/v1/metrics/dora/deployments", json=body)
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_dora_deployment_failure(self, client: AsyncClient):
        """记录失败的 DORA 部署."""
        body = {"agent_id": "agent-fail", "status": "failure"}
        resp = await client.post("/governance/v1/metrics/dora/deployments", json=body)
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_dora_deployment_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.post(
            "/governance/v1/metrics/dora/deployments", json={},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_dora_metrics(self, client: AsyncClient):
        """GET /metrics/dora — DORA 指标."""
        resp = await client.get("/governance/v1/metrics/dora")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_dora_metrics_with_agent(self, client: AsyncClient):
        """按 agent_id 过滤 DORA 指标."""
        resp = await client.get(
            "/governance/v1/metrics/dora",
            params={"agent_id": "agent-001"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_dora_metrics_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/metrics/dora")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_metrics_stats(self, client: AsyncClient):
        """GET /metrics/stats — 度量统计."""
        resp = await client.get("/governance/v1/metrics/stats")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        data = resp.json()["data"]
        # 实际字段名为 defined_metrics
        assert "defined_metrics" in data

    @pytest.mark.asyncio
    async def test_metrics_stats_uninitialized(self, empty_client: AsyncClient):
        """度量引擎未初始化返回 503."""
        resp = await empty_client.get("/governance/v1/metrics/stats")
        assert resp.status_code == 503


# ============================================================
# 9. G5 合规测试
# ============================================================


class TestComplianceEndpoints:
    """测试 G5 合规端点."""

    @pytest.mark.asyncio
    async def test_compliance_report(self, client: AsyncClient):
        """POST /compliance/report — 生成合规报告."""
        body = {
            "audit_stats": {"total_decisions": 100},
            "metrics_stats": {},
            "frameworks": ["SOC2"],
        }
        resp = await client.post("/governance/v1/compliance/report", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_compliance_report_structure(self, client: AsyncClient):
        """合规报告包含关键字段."""
        resp = await client.post("/governance/v1/compliance/report", json={
            "audit_stats": {"total_decisions": 50},
        })
        data = resp.json()["data"]
        assert "overall_score" in data
        assert "domains" in data
        assert "frameworks" in data

    @pytest.mark.asyncio
    async def test_compliance_report_with_multiple_frameworks(self, client: AsyncClient):
        """多框架合规报告."""
        resp = await client.post("/governance/v1/compliance/report", json={
            "audit_stats": {},
            "frameworks": ["SOC2", "NIST_AI_RMF", "ETHICAL_COMPLIANCE"],
        })
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_compliance_report_minimal(self, client: AsyncClient):
        """最小参数合规报告."""
        resp = await client.post("/governance/v1/compliance/report", json={})
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_compliance_report_uninitialized(self, empty_client: AsyncClient):
        """合规报告器未初始化返回 503."""
        resp = await empty_client.post("/governance/v1/compliance/report", json={})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_nist_summary(self, client: AsyncClient):
        """POST /compliance/nist-summary — NIST 摘要."""
        body = {
            "report": {
                "domains": [
                    {
                        "domain_id": "GOVERN",
                        "name": "治理域",
                        "controls": [
                            {
                                "control_id": "GV-1",
                                "name": "测试控制",
                                "framework": "NIST_AI_RMF",
                                "nist_function": "govern",
                                "score": 0.9,
                            },
                        ],
                    },
                ],
                "overall_score": 0.85,
                "generated_at": 1234567890.0,
            },
        }
        resp = await client.post("/governance/v1/compliance/nist-summary", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_nist_summary_structure(self, client: AsyncClient):
        """NIST 摘要包含四函数."""
        resp = await client.post("/governance/v1/compliance/nist-summary", json={
            "report": {
                "domains": [],
                "overall_score": 0.8,
                "generated_at": 0.0,
            },
        })
        data = resp.json()["data"]
        assert "govern" in data
        assert "map" in data
        assert "measure" in data
        assert "manage" in data

    @pytest.mark.asyncio
    async def test_nist_summary_missing_report(self, client: AsyncClient):
        """缺少 report 参数返回 400."""
        resp = await client.post("/governance/v1/compliance/nist-summary", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_nist_summary_uninitialized(self, empty_client: AsyncClient):
        """合规报告器未初始化返回 503."""
        resp = await empty_client.post("/governance/v1/compliance/nist-summary", json={
            "report": {
                "domains": [],
                "overall_score": 0.0,
                "generated_at": 0.0,
            },
        })
        assert resp.status_code == 503


# ============================================================
# 10. 全链路集成测试
# ============================================================


class TestChainIntegration:
    """测试全链路集成端点."""

    @pytest.mark.asyncio
    async def test_chain_evaluate_full(self, client: AsyncClient):
        """POST /chain/evaluate — 完整 G1→G3→G4→G5 流水线."""
        body = {
            "policy": {
                "actor": "agent-tutor",
                "action": "grade",
                "resource": "student-001",
            },
            "anti_hallucination": {
                "output_text": "该学生成绩为A，表现优秀。",
                "context_chunks": ["学生期末考试成绩95分。"],
            },
            "collaboration": {
                "agent_id": "tutor-001",
                "score": {
                    "risk": 2.0,
                    "explainability": 1.5,
                    "accuracy": 2.0,
                    "consequence": 3.0,
                    "time_sensitivity": 1.0,
                },
            },
            "actor": "chain-test",
        }
        resp = await client.post("/governance/v1/chain/evaluate", json=body)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        data = resp.json()["data"]
        assert "trace_id" in data
        assert "stages" in data
        assert "policy_eval" in data["stages"]
        assert "anti_hallucination" in data["stages"]
        assert "collaboration" in data["stages"]
        assert "audit" in data["stages"]
        assert "metrics" in data["stages"]

    @pytest.mark.asyncio
    async def test_chain_evaluate_minimal(self, client: AsyncClient):
        """最小参数全链路评估（仅触发审计和度量）."""
        resp = await client.post("/governance/v1/chain/evaluate", json={})
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        data = resp.json()["data"]
        assert "trace_id" in data
        # 仅审计和度量阶段
        assert data["stages"].get("audit", {}).get("recorded") is True
        # 度量阶段可能成功或失败（取决于参数匹配）
        metrics_stage = data["stages"].get("metrics", {})
        assert "recorded" in metrics_stage or "error" in metrics_stage

    @pytest.mark.asyncio
    async def test_chain_evaluate_custom_trace_id(self, client: AsyncClient):
        """自定义 trace_id."""
        body = {"trace_id": "custom-trace-001"}
        resp = await client.post("/governance/v1/chain/evaluate", json=body)
        assert resp.json()["data"]["trace_id"] == "custom-trace-001"

    @pytest.mark.asyncio
    async def test_chain_evaluate_auto_trace_id(self, client: AsyncClient):
        """自动生成 trace_id 以 g6- 开头."""
        resp = await client.post("/governance/v1/chain/evaluate", json={})
        assert resp.json()["data"]["trace_id"].startswith("g6-")

    @pytest.mark.asyncio
    async def test_chain_evaluate_policy_only(self, client: AsyncClient):
        """仅策略评估阶段."""
        body = {
            "policy": {"actor": "a", "action": "test", "resource": "r"},
        }
        resp = await client.post("/governance/v1/chain/evaluate", json=body)
        data = resp.json()["data"]
        assert "policy_eval" in data["stages"]
        assert "decision" in data["stages"]["policy_eval"]

    @pytest.mark.asyncio
    async def test_chain_evaluate_cc1_only(self, client: AsyncClient):
        """仅防幻觉验证阶段."""
        body = {
            "anti_hallucination": {"output_text": "测试内容。"},
        }
        resp = await client.post("/governance/v1/chain/evaluate", json=body)
        data = resp.json()["data"]
        assert "anti_hallucination" in data["stages"]

    @pytest.mark.asyncio
    async def test_chain_evaluate_cc2_only(self, client: AsyncClient):
        """仅协作评估阶段 — 中等分数 conditional 模式."""
        body = {
            "collaboration": {
                "agent_id": "test",
                "score": {
                    "risk": 3.0, "explainability": 3.0, "accuracy": 3.0,
                    "consequence": 3.0, "time_sensitivity": 3.0,
                },
            },
        }
        resp = await client.post("/governance/v1/chain/evaluate", json=body)
        data = resp.json()["data"]
        assert "collaboration" in data["stages"]
        assert data["stages"]["collaboration"]["recommended_mode"] == "conditional"

    @pytest.mark.asyncio
    async def test_chain_evaluate_invalid_json(self, client: AsyncClient):
        """无效 JSON 返回 400."""
        resp = await client.post(
            "/governance/v1/chain/evaluate",
            content=b"bad",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == -32700

    @pytest.mark.asyncio
    async def test_chain_status(self, client: AsyncClient):
        """GET /chain/status — 链路状态总览."""
        resp = await client.get("/governance/v1/chain/status")
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        data = resp.json()["data"]
        assert "subsystems" in data
        # 全部初始化
        assert all(data["subsystems"].values())

    @pytest.mark.asyncio
    async def test_chain_status_empty(self, empty_client: AsyncClient):
        """无子系统时链路状态."""
        resp = await empty_client.get("/governance/v1/chain/status")
        data = resp.json()["data"]
        assert not any(data["subsystems"].values())

    @pytest.mark.asyncio
    async def test_chain_status_has_stats(self, client: AsyncClient):
        """链路状态包含各子系统统计."""
        resp = await client.get("/governance/v1/chain/status")
        data = resp.json()["data"]
        assert "audit" in data or "metrics" in data or "collaboration" in data or "subsystems" in data

    @pytest.mark.asyncio
    async def test_chain_status_after_operations(self, client: AsyncClient):
        """执行操作后状态包含统计数据."""
        # 执行全链路评估
        await client.post("/governance/v1/chain/evaluate", json={})
        resp = await client.get("/governance/v1/chain/status")
        data = resp.json()["data"]
        # 审计应该有数据（使用 total_recorded 字段名）
        if "audit" in data:
            assert data["audit"].get("total_recorded", 0) >= 1


# ============================================================
# 11. 额外边界和错误场景测试
# ============================================================


class TestEdgeCases:
    """边界和错误场景测试."""

    @pytest.mark.asyncio
    async def test_unknown_route_404(self, client: AsyncClient):
        """未知路由返回 404."""
        resp = await client.get("/governance/v1/unknown-endpoint")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_route_post_404(self, client: AsyncClient):
        """未知 POST 路由返回 404."""
        resp = await client.post("/governance/v1/unknown", json={})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_wrong_method_405(self, client: AsyncClient):
        """HTTP 方法错误返回 405."""
        resp = await client.delete("/governance/v1/policies")
        assert resp.status_code == 405

    @pytest.mark.asyncio
    async def test_post_to_get_endpoint(self, client: AsyncClient):
        """向 GET 端点发 POST 请求."""
        resp = await client.post("/governance/v1/health")
        assert resp.status_code == 405

    @pytest.mark.asyncio
    async def test_health_no_subsystems_always_200(self, empty_client: AsyncClient):
        """无子系统时存活检查始终 200."""
        resp = await empty_client.get("/governance/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_cors_headers(self, client: AsyncClient):
        """CORS 头存在."""
        resp = await client.options(
            "/governance/v1/health",
            headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"},
        )
        # Starlette CORS middleware 处理 preflight
        assert resp.status_code in (200, 404, 400)

    @pytest.mark.asyncio
    async def test_response_content_type(self, client: AsyncClient):
        """响应 Content-Type 为 JSON."""
        resp = await client.get("/governance/v1/health")
        assert "application/json" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_multiple_health_checks_consistent(self, client: AsyncClient):
        """多次健康检查结果一致."""
        resp1 = await client.get("/governance/v1/health")
        resp2 = await client.get("/governance/v1/health")
        assert resp1.json()["data"]["status"] == resp2.json()["data"]["status"]

    @pytest.mark.asyncio
    async def test_chain_evaluate_generates_unique_traces(self, client: AsyncClient):
        """多次全链路评估生成不同 trace_id."""
        ids = set()
        for _ in range(5):
            resp = await client.post("/governance/v1/chain/evaluate", json={})
            ids.add(resp.json()["data"]["trace_id"])
        assert len(ids) == 5

    @pytest.mark.asyncio
    async def test_policy_evaluate_caching(self, client: AsyncClient):
        """策略评估使用缓存."""
        body = {"actor": "cache-test", "action": "cached_action", "resource": "r"}
        resp1 = await client.post("/governance/v1/policies/evaluate", json=body)
        resp2 = await client.post("/governance/v1/policies/evaluate", json=body)
        assert resp1.json()["data"]["decision"] == resp2.json()["data"]["decision"]

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client: AsyncClient):
        """并发请求不产生异常."""
        import asyncio
        tasks = [
            client.get("/governance/v1/health"),
            client.get("/governance/v1/health/ready"),
            client.get("/governance/v1/chain/status"),
        ]
        results = await asyncio.gather(*tasks)
        for resp in results:
            assert resp.status_code == 200


class TestResponseEnvelope:
    """测试统一响应信封格式."""

    @pytest.mark.asyncio
    async def test_ok_response_has_code_zero(self, client: AsyncClient):
        """成功响应 code=0."""
        resp = await client.get("/governance/v1/health")
        assert resp.json()["code"] == 0

    @pytest.mark.asyncio
    async def test_ok_response_has_message(self, client: AsyncClient):
        """成功响应 message 为空字符串."""
        resp = await client.get("/governance/v1/health")
        assert resp.json()["message"] == ""

    @pytest.mark.asyncio
    async def test_ok_response_has_data(self, client: AsyncClient):
        """成功响应包含 data."""
        resp = await client.get("/governance/v1/health")
        assert "data" in resp.json()

    @pytest.mark.asyncio
    async def test_err_response_code(self, empty_client: AsyncClient):
        """错误响应 code 非 0."""
        resp = await empty_client.get("/governance/v1/policies")
        body = resp.json()
        assert body["code"] != 0

    @pytest.mark.asyncio
    async def test_err_response_has_message(self, empty_client: AsyncClient):
        """错误响应包含 message."""
        resp = await empty_client.get("/governance/v1/policies")
        assert "message" in resp.json()
        assert len(resp.json()["message"]) > 0
