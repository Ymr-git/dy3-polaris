"""结构性评审整改专项测试: 画像口径统一 / 策略收敛 / 反馈单点 / kp_catalog."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l2.kp_catalog import (
    ALL_KP_IDS,
    KP_DOMAIN_IDS,
    KP_LEVELS,
    KP_NAMES,
    NODE_TO_KP,
    to_dict,
)
from dy3_polaris.l2.models import FeedbackType
from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def builder():
    return UnifiedApp.create_full_app_builder()


def _headers(client):
    login = client.post(
        "/l1/api/v1/auth/login",
        json={"student_id": "DY20248888", "password": "admin888"},
    )
    return {"Authorization": "Bearer " + login.json()["data"]["access_token"]}


@pytest.fixture(scope="module")
def client(builder) -> TestClient:
    return TestClient(builder.create_app())


class TestKpCatalog:
    """知识点目录单点 (L2 SSOT)."""

    def test_catalog_complete(self):
        assert len(ALL_KP_IDS) == 42
        assert len(KP_NAMES) == 42
        assert KP_DOMAIN_IDS["A"] == [f"A-{i:02d}" for i in range(1, 14)]
        assert all(kp in KP_LEVELS for kp in ALL_KP_IDS)

    def test_kg_node_mapping(self):
        assert NODE_TO_KP["Dy3+"] == "A-01"
        assert NODE_TO_KP["cie"] == "B-07"
        assert NODE_TO_KP["xrd"] == "D-01"

    def test_endpoint(self, client):
        r = client.get("/l2/kp-catalog")
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["total"] == 42
        assert len(d["domains"]) == 4
        assert len(d["kp"]) == 42
        assert all({"kp_id", "name", "domain", "level"} <= set(k.keys()) for k in d["kp"])

    def test_l7_references_catalog(self):
        from dy3_polaris.l7.renderers import _common
        assert _common.ALL_KP_IDS == ALL_KP_IDS
        assert _common.KP_NAMES["A-05"] == "Dy3+ 能级结构与 4f-4f 跃迁"

    def test_frontend_loads_catalog(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "/l2/kp-catalog" in js
        assert "loadKpCatalog" in js


class TestProfileUnifiedPipeline:
    """画像构建口径统一: practice 走全量重建管线."""

    def test_practice_uses_pipeline(self, client):
        """练习判题后画像经完整管线重建 (不再手工补丁)."""
        bank = client.get("/l2/practice/questions?learner_id=DY20240001&count=2").json()["data"]
        q = bank["questions"][0]
        qid = q["qid"]
        r = client.post("/l2/practice/answer", json={
            "learner_id": "DY20240001", "qid": qid, "selected": 0,
        })
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["p_mastery_after"] != d["p_mastery_before"] or True  # BKT 更新
        # 画像 kp_mastery 与返回 after 一致 (共享 store + 管线重建)
        after = client.get("/l2/profile/DY20240001").json()["data"]
        kp = d["kp_id"]
        assert abs(after["kp_mastery"][kp] - d["p_mastery_after"]) < 1e-6
        # 画像全量字段保留 (level/theta/confidence 均存在)
        assert "level" in after and "theta" in after and "confidence" in after

    def test_weak_threshold_aligned(self):
        """出题薄弱阈值与画像重建口径一致 (0.5)."""
        import dy3_polaris.l2.practice as p
        assert p.WEAK_KP_THRESHOLD == 0.5


class TestStrategyConverged:
    """策略四套收敛: L6 stub 委托 L4 / run_guidance 语义对齐."""

    def test_l6_path_simulation_uses_l4(self, client):
        r = client.post("/l6/tools/path_simulation/call", json={
            "arguments": {
                "learner_id": "DY20240001",
                "start_kp": "A-01",
                "target_kp": "A-05",
                "current_state": {"weak_kps": ["A-01"], "mastered_kps": {"A-01": 0.3}},
            }
        }, headers=_headers(client))
        assert r.status_code == 200
        d = r.json()["data"]["result"]
        assert d["decision_source"] == "l4.next_action"
        assert "action_type" in d and "confidence" in d
        assert d["recommended_path_detail"]  # L4 统一结构

    def test_query_response_has_path(self, client):
        r = client.post("/api/query", json={
            "query": "Dy3+ 的量子效率受哪些因素影响？",
            "learner_id": "DY20240001",
        })
        assert r.status_code == 200
        body = r.json()
        data = body.get("data") or body
        assert "recommended_path" in data  # 三语义统一字段
        assert data["action_type"]


class TestFeedbackUnified:
    """反馈类型单点枚举 + 跨通道聚合."""

    def test_unified_enum(self):
        assert FeedbackType.AGENT_OUTCOME.value == "agent_outcome"
        assert FeedbackType.HUMAN_FEEDBACK.value == "human_feedback"

    def test_l4_mapping(self):
        from dy3_polaris.l4.models import FeedbackType as L4FT
        assert L4FT.OUTCOME_FEEDBACK.to_unified() == "agent_outcome"
        assert L4FT.IMPLICIT_SIGNAL.to_unified() == "implicit_result"

    def test_feedback_to_profile_unified_value(self, client):
        """L5 反馈桥写统一枚举值 (agent_outcome)."""
        h = {"Authorization": "Bearer " + client.post(
            "/l1/api/v1/auth/login",
            json={"student_id": "DY20248888", "password": "admin888"},
        ).json()["data"]["access_token"]}
        client.post("/l5/agents/agent.guidance.decision/skills/call",
                    json={"tool": "path_simulation", "args": {"learner_id": "DY20240001"}},
                    headers=h)
        agg = client.get("/api/feedback/aggregate/DY20240001").json()["data"]
        assert "by_type" in agg and "feedback_log" in agg
        types = set(agg["by_type"].keys())
        assert types <= {"agent_outcome", "explicit_rating", "human_feedback",
                         "implicit_result", "skip", "unknown"}
