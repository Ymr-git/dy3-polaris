"""M-F7 缺口补齐专项测试: 演示数据播种 + 溯源链端点 + 前端视图资源."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def client() -> TestClient:
    builder = UnifiedApp.create_full_app_builder()
    return TestClient(builder.create_app())


class TestSeedDemoLearningData:
    """42 KP 画像播种."""

    def test_profile_seeded(self, client):
        r = client.get("/l2/profile/DY20240001")
        assert r.json()["code"] == 0
        d = r.json()["data"]
        assert len(d["kp_mastery"]) == 42
        assert d["theta"] is not None
        assert 0 < d["confidence"] <= 1
        assert len(d["weak_kps"]) > 0

    def test_profile_second_learner(self, client):
        r = client.get("/l2/profile/DY20240002")
        assert r.json()["code"] == 0
        assert len(r.json()["data"]["kp_mastery"]) == 42

    def test_mastery_in_range(self, client):
        d = client.get("/l2/profile/DY20240001").json()["data"]
        for v in d["kp_mastery"].values():
            assert 0.0 <= v <= 1.0


class TestSeedDemoKnowledge:
    """3 实体 + 溯源链播种."""

    def test_entities_seeded(self, client):
        d = client.get("/l3/entities").json()["data"]
        assert d["total"] >= 3

    def test_provenance_chain_verified(self, client):
        # 按名定位 3 个 demo 实体 (而非盲目取 items[:3], 因列表含层级/快照实体)
        d = client.get("/l3/entities").json()["data"]
        demo_names = {
            "NaGdF4:Eu3+ 纳米晶",
            "YPO4:Dy3+ 荧光粉",
            "BaMgAl10O17:Eu2+ (BAM)",
        }
        demos = [e for e in d["items"] if e["name"] in demo_names]
        assert len(demos) == 3, f"demo 实体未全部播种: {[e['name'] for e in demos]}"
        for e in demos:
            r = client.get(f"/l3/quality/provenance/{e['entity_id']}/chain")
            assert r.json()["code"] == 0
            body = r.json()["data"]
            assert "VERIFIED" in body["verified"].upper()
            assert len(body["chain"]) >= 1
            assert body["chain"][0]["integrity_hash"]

    def test_chain_multi_hop(self, client):
        # 链尾实体 (BAM) 链长应为 3 (三实体串联: BAM → YPO4 → NaGdF4)
        d = client.get("/l3/entities").json()["data"]
        tail = next((e for e in d["items"] if e["name"] == "BaMgAl10O17:Eu2+ (BAM)"), None)
        assert tail is not None, "链尾实体 BAM 未播种"
        r = client.get(f"/l3/quality/provenance/{tail['entity_id']}/chain")
        assert len(r.json()["data"]["chain"]) == 3

    def test_provenance_404(self, client):
        r = client.get("/l3/quality/provenance/nonexistent/chain")
        assert r.status_code == 404


class TestMF7FrontendAssets:
    """前端资源包含新视图渲染函数."""

    def test_mf6_js_has_new_views(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "renderBktHeatmap" in js
        assert "renderLearnRing" in js
        assert "injectProvenanceBadges" in js
        assert "setupResponsiveNav" in js
        assert "learn-mastery" in js
        assert "kp_mastery" in js

    def test_css_has_responsive_layout(self, client):
        css = client.get("/static/assets/app.css").text
        assert "#menuBtn" in css
        assert "#sidebarMask" in css
        assert ".sidebar.open" in css
        assert "@media(max-width:768px)" in css
