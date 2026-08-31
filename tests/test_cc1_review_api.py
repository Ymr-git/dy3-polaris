"""CC1 四层反幻觉评审引擎 — REST API 集成测试.

测试 GovernanceRouter 暴露的 CC1 评审端点:
- POST /governance/v1/review/execute — 执行四层反幻觉评审
- GET  /governance/v1/review/layers  — 列出四层评审规则
- GET  /governance/v1/review/config  — 获取评审配置
- GET  /governance/v1/review/weights — 获取四层权重

遵循 TDD: 先写测试 (RED), 再实现 (GREEN), 最后重构 (REFACTOR).
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l0.governance_router import (
    GovernanceRouter,
    GovernanceSubsystems,
)
from dy3_polaris.l0.cc1.review_pipeline import ReviewPipeline


# ============================================================
# 测试 Fixtures
# ============================================================


@pytest.fixture
def review_pipeline() -> ReviewPipeline:
    """创建评审管道实例."""
    return ReviewPipeline()


@pytest.fixture
def subsys(review_pipeline: ReviewPipeline) -> GovernanceSubsystems:
    """创建包含评审管道的治理子系统."""
    return GovernanceSubsystems(review_pipeline=review_pipeline)


@pytest.fixture
def client(subsys: GovernanceSubsystems) -> TestClient:
    """创建测试客户端 (与 UnifiedApp 一致的 /governance 挂载)."""
    from starlette.routing import Mount
    from starlette.applications import Starlette
    router = GovernanceRouter(subsys)
    app = router.create_app()
    wrapped = Starlette(routes=[Mount("/governance", app=app)])
    return TestClient(wrapped)


def _ok_data(resp) -> dict:
    """提取成功响应的 data 字段."""
    body = resp.json()
    assert body["code"] == 0, f"Expected code=0, got {body}"
    return body["data"]


# ============================================================
# POST /governance/v1/review/execute
# ============================================================


class TestReviewExecute:
    """POST /governance/v1/review/execute — 执行四层反幻觉评审."""

    def test_execute_high_quality_output(self, client: TestClient):
        """高质量输出 → PASS, 综合分 >= 85."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "output_text": "Dy3+ 的发射主峰在 575nm, 对应 ⁴F₉/₂→⁶H₁₃/₂ 跃迁。",
                "context_chunks": [
                    "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射",
                ],
                "citations": [],
            },
        )
        assert resp.status_code == 200
        result = _ok_data(resp)
        assert result["verdict"] == "pass"
        assert result["composite_score"] >= 85.0
        assert result["report_id"].startswith("rr-")
        assert result["agent_id"] == "agent-knowledge"

    def test_execute_low_quality_output(self, client: TestClient):
        """低质量输出 → BLOCK/FLAG, 综合分 < 85."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "output_text": (
                    "Dy3+ 的发射主峰在 600nm, 掺杂浓度 20mol%, "
                    "衰减寿命 10ms, CIE 色坐标 (0.8, 0.8)。"
                ),
                "context_chunks": [
                    "Dy3+ 发射峰 600nm, 掺杂 20mol%, 衰减 10ms",
                ],
                "citations": [],
            },
        )
        assert resp.status_code == 200
        result = _ok_data(resp)
        assert result["verdict"] in ("block", "flag")
        assert result["composite_score"] < 85.0

    def test_execute_with_layer_results(self, client: TestClient):
        """评审结果包含四层评审详情."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "output_text": "Dy3+ 的发射主峰在 575nm。",
                "context_chunks": ["Dy3+ 发射峰 575nm"],
                "citations": [],
            },
        )
        data = _ok_data(resp)
        assert "layer_results" in data
        assert "layer_scores" in data
        layer_keys = list(data["layer_results"].keys())
        assert "l1_fact" in layer_keys
        assert "l2_logic" in layer_keys
        assert "l3_numerical" in layer_keys
        assert "l4_provenance" in layer_keys

    def test_execute_with_self_correction(self, client: TestClient):
        """FLAG/BLOCK 结果包含自纠回路信息."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "output_text": (
                    "Dy3+ 的发射主峰在 600nm, 属于 d 区过渡金属。"
                ),
                "context_chunks": ["Dy3+ 发射峰 600nm, d 区过渡金属"],
                "citations": [],
            },
        )
        data = _ok_data(resp)
        assert "self_correction" in data

    def test_execute_empty_output(self, client: TestClient):
        """空输出 → PASS (无声明可校验)."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "output_text": "",
                "context_chunks": [],
                "citations": [],
            },
        )
        assert resp.status_code == 200
        data = _ok_data(resp)
        assert data["verdict"] == "pass"
        assert data["composite_score"] == 100.0

    def test_execute_missing_output_text(self, client: TestClient):
        """缺少 output_text → 400."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "context_chunks": [],
                "citations": [],
            },
        )
        assert resp.status_code == 400

    def test_execute_invalid_body(self, client: TestClient):
        """无效请求体 → 400."""
        resp = client.post(
            "/governance/v1/review/execute",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_execute_with_issues_list(self, client: TestClient):
        """评审结果包含问题列表."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "output_text": (
                    "Dy3+ 发射主峰 600nm, 掺杂浓度 20mol%。"
                ),
                "context_chunks": ["Dy3+ 发射峰 600nm, 掺杂 20mol%"],
                "citations": [],
            },
        )
        data = _ok_data(resp)
        assert "issues" in data
        assert isinstance(data["issues"], list)

    def test_execute_timestamps(self, client: TestClient):
        """评审结果包含时间戳."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "output_text": "Dy3+ 发射峰 575nm。",
                "context_chunks": ["Dy3+ 发射峰 575nm"],
                "citations": [],
            },
        )
        data = _ok_data(resp)
        assert "created_at" in data
        assert "completed_at" in data
        assert data["created_at"] > 0
        assert data["completed_at"] >= data["created_at"]

    def test_execute_no_pipeline_returns_503(self):
        """评审管道未初始化 → 503."""
        from starlette.routing import Mount
        from starlette.applications import Starlette
        subsys = GovernanceSubsystems()
        router = GovernanceRouter(subsys)
        wrapped = Starlette(routes=[Mount("/governance", app=router.create_app())])
        with TestClient(wrapped) as c:
            resp = c.post(
                "/governance/v1/review/execute",
                json={
                    "agent_id": "agent-knowledge",
                    "output_text": "test",
                    "context_chunks": [],
                    "citations": [],
                },
            )
            assert resp.status_code == 503


# ============================================================
# GET /governance/v1/review/layers
# ============================================================


class TestReviewLayers:
    """GET /governance/v1/review/layers — 列出四层评审规则."""

    def test_layers_returns_four_layers(self, client: TestClient):
        """返回四层评审规则."""
        resp = client.get("/governance/v1/review/layers")
        assert resp.status_code == 200
        layers = _ok_data(resp)
        assert len(layers) == 4

    def test_layers_contain_layer_types(self, client: TestClient):
        """四层类型正确."""
        resp = client.get("/governance/v1/review/layers")
        layers = _ok_data(resp)
        layer_types = {l["layer_type"] for l in layers}
        assert "l1_fact" in layer_types
        assert "l2_logic" in layer_types
        assert "l3_numerical" in layer_types
        assert "l4_provenance" in layer_types

    def test_layers_contain_rules(self, client: TestClient):
        """每层包含规则列表."""
        resp = client.get("/governance/v1/review/layers")
        layers = _ok_data(resp)
        for layer in layers:
            assert "rules" in layer
            assert len(layer["rules"]) > 0
            assert layer["rule_count"] == len(layer["rules"])
            for rule in layer["rules"]:
                assert "rule_id" in rule
                assert "name" in rule
                assert "description" in rule
                assert "severity" in rule

    def test_layers_fact_layer_rule_count(self, client: TestClient):
        """L1 事实层包含 12 条规则."""
        resp = client.get("/governance/v1/review/layers")
        layers = _ok_data(resp)
        fact_layer = next(l for l in layers if l["layer_type"] == "l1_fact")
        assert fact_layer["rule_count"] == 12

    def test_layers_logic_layer_rule_count(self, client: TestClient):
        """L2 逻辑层包含 10 条规则."""
        resp = client.get("/governance/v1/review/layers")
        layers = _ok_data(resp)
        logic_layer = next(
            l for l in layers if l["layer_type"] == "l2_logic"
        )
        assert logic_layer["rule_count"] == 15  # enhanced rules

    def test_layers_numerical_layer_rule_count(self, client: TestClient):
        """L3 数值层包含 12 条规则."""
        resp = client.get("/governance/v1/review/layers")
        layers = _ok_data(resp)
        numerical_layer = next(
            l for l in layers if l["layer_type"] == "l3_numerical"
        )
        assert numerical_layer["rule_count"] == 18  # enhanced rules

    def test_layers_provenance_layer_rule_count(self, client: TestClient):
        """L4 溯源层包含 10 条规则."""
        resp = client.get("/governance/v1/review/layers")
        layers = _ok_data(resp)
        provenance_layer = next(
            l for l in layers if l["layer_type"] == "l4_provenance"
        )
        assert provenance_layer["rule_count"] == 15  # enhanced rules


# ============================================================
# GET /governance/v1/review/config
# ============================================================


class TestReviewConfig:
    """GET /governance/v1/review/config — 获取评审配置."""

    def test_config_returns_all_fields(self, client: TestClient):
        """返回所有配置字段."""
        resp = client.get("/governance/v1/review/config")
        assert resp.status_code == 200
        cfg = _ok_data(resp)
        assert "pass_threshold" in cfg
        assert "flag_threshold" in cfg
        assert "max_corrections" in cfg
        assert "enable_self_correction" in cfg
        assert "error_penalty" in cfg
        assert "critical_penalty" in cfg
        assert "warning_penalty" in cfg
        assert "info_penalty" in cfg

    def test_config_default_values(self, client: TestClient):
        """默认配置值正确."""
        resp = client.get("/governance/v1/review/config")
        cfg = _ok_data(resp)
        assert cfg["pass_threshold"] == 85.0
        assert cfg["flag_threshold"] == 60.0
        assert cfg["max_corrections"] == 2
        assert cfg["enable_self_correction"] is True
        assert cfg["error_penalty"] == 30.0
        assert cfg["critical_penalty"] == 50.0
        assert cfg["warning_penalty"] == 10.0
        assert cfg["info_penalty"] == 5.0


# ============================================================
# GET /governance/v1/review/weights
# ============================================================


class TestReviewWeights:
    """GET /governance/v1/review/weights — 获取四层权重."""

    def test_weights_returns_four_layers(self, client: TestClient):
        """返回四层权重."""
        resp = client.get("/governance/v1/review/weights")
        assert resp.status_code == 200
        weights = _ok_data(resp)
        assert "l1_fact" in weights
        assert "l2_logic" in weights
        assert "l3_numerical" in weights
        assert "l4_provenance" in weights

    def test_weights_sum_to_one(self, client: TestClient):
        """权重之和为 1.0."""
        resp = client.get("/governance/v1/review/weights")
        weights = _ok_data(resp)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_weights_values(self, client: TestClient):
        """权重值正确: L1=0.40, L2=0.25, L3=0.20, L4=0.15."""
        resp = client.get("/governance/v1/review/weights")
        weights = _ok_data(resp)
        assert weights["l1_fact"] == 0.40
        assert weights["l2_logic"] == 0.25
        assert weights["l3_numerical"] == 0.20
        assert weights["l4_provenance"] == 0.15


# ============================================================
# 健康检查集成
# ============================================================


class TestHealthIntegration:
    """健康检查包含 review_pipeline 状态."""

    def test_health_includes_review_pipeline(self, client: TestClient):
        """就绪检查包含 review_pipeline 字段."""
        resp = client.get("/governance/v1/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        health = body["data"]
        assert "subsystems" in health
        assert "review_pipeline" in health["subsystems"]
        assert health["subsystems"]["review_pipeline"] is True

    def test_health_ready_with_review_pipeline(self, client: TestClient):
        """就绪检查通过."""
        resp = client.get("/governance/v1/health/ready")
        assert resp.status_code == 200


# ============================================================
# 工厂函数测试
# ============================================================


class TestCreateGovernanceApp:
    """create_governance_app 工厂函数测试."""

    def test_factory_creates_app_with_review_pipeline(self):
        """工厂函数创建包含评审管道的应用."""
        from dy3_polaris.l0.governance_router import create_governance_app

        app = create_governance_app()
        assert app is not None

        with TestClient(app) as c:
            resp = c.get("/governance/v1/review/weights")
            assert resp.status_code == 200
            weights = _ok_data(resp)
            assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_factory_creates_app_without_review_pipeline(self):
        """工厂函数创建不含评审管道的应用."""
        from dy3_polaris.l0.governance_router import create_governance_app

        app = create_governance_app(include_review_pipeline=False)
        assert app is not None

        with TestClient(app) as c:
            resp = c.get("/governance/v1/review/weights")
            assert resp.status_code == 503


# ============================================================
# 路由发现
# ============================================================


class TestRouteDiscovery:
    """路由发现包含 CC1 评审端点."""

    def test_routes_include_review_endpoints(self, client: TestClient):
        """路由发现包含四层评审端点."""
        resp = client.get("/governance/v1/routes")
        assert resp.status_code == 200
        routes = _ok_data(resp)
        paths = {r["path"] for r in routes}
        assert "/governance/v1/review/execute" in paths
        assert "/governance/v1/review/layers" in paths
        assert "/governance/v1/review/config" in paths
        assert "/governance/v1/review/weights" in paths
