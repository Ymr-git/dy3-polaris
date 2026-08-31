"""M-F4 知识库管理 + 治理可视化 — 回归测试."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def client() -> TestClient:
    builder = UnifiedApp.create_full_app_builder()
    return TestClient(builder.create_app())


class TestGovernanceApi:
    """L0 治理接口 (此前双前缀 bug, 现修复)."""

    def test_governance_health(self, client):
        data = client.get("/governance/v1/health").json()["data"]
        assert any(k in data for k in ("status", "healthy", "layer", "ok"))

    def test_governance_policies(self, client):
        r = client.get("/governance/v1/policies")
        assert r.status_code == 200

    def test_governance_policy_metrics(self, client):
        data = client.get("/governance/v1/policies/metrics").json()["data"]
        assert "evaluations" in data

    def test_governance_anti_hallucination(self, client):
        data = client.get("/governance/v1/anti-hallucination/stats").json()["data"]
        assert "pipeline_initialized" in data

    def test_governance_review_statistics(self, client):
        data = client.get("/governance/v1/review/statistics").json()["data"]
        for k in ("total", "pass", "flag", "block", "pass_rate"):
            assert k in data

    def test_governance_review_layers(self, client):
        data = client.get("/governance/v1/review/layers").json()["data"]
        assert isinstance(data, list) and len(data) >= 4  # 四层评审

    def test_no_double_prefix(self, client):
        """双前缀 bug 修复: /governance/governance/v1 不再存在."""
        assert client.get("/governance/governance/v1/health").status_code == 404


class TestKnowledgeBaseApi:
    """L3 知识库接口."""

    def test_entities(self, client):
        data = client.get("/l3/entities").json()["data"]
        assert "items" in data and "total" in data

    def test_graph_stats(self, client):
        data = client.get("/l3/graph/stats").json()["data"]
        assert "entities_count" in data

    def test_quality_dashboard(self, client):
        data = client.get("/l3/quality/dashboard").json()["data"]
        assert "total_entities" in data

    def test_ontology_domains(self, client):
        data = client.get("/l3/ontology/domains").json()["data"]
        assert "domains" in data


class TestMF4Frontend:
    """M-F4 前端视图: 知识库管理 + 治理可视化."""

    def test_js_has_kb_view(self, client):
        js = client.get("/static/assets/app.js").text
        assert "/l3/entities" in js
        assert "/l3/graph/stats" in js
        assert "知识库管理" in js

    def test_js_has_governance_views(self, client):
        js = client.get("/static/assets/app.js").text
        assert "/governance/v1/review/statistics" in js
        assert "/governance/v1/review/layers" in js
        assert "/l1/api/v1/audit/logs" in js
        assert "反幻觉评审" in js

    def test_css_has_callout_and_table(self, client):
        css = client.get("/static/assets/app.css").text
        assert ".callout" in css
        assert ".table-wrap" in css
        assert ".badge.info" in css
