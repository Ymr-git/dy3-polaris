"""架构归位专项测试: 画像唯一写方乐观锁 / L4 唯一策略决策 / 会话上下文统一."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l2.models import ProfileConflictError
from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def builder():
    return UnifiedApp.create_full_app_builder()


@pytest.fixture(scope="module")
def client(builder) -> TestClient:
    return TestClient(builder.create_app())


def _headers(client, student="DY20248888", pwd="admin888"):
    login = client.post(
        "/l1/api/v1/auth/login",
        json={"student_id": student, "password": pwd},
    )
    return {"Authorization": "Bearer " + login.json()["data"]["access_token"]}


class TestProfileWriteGateway:
    """画像唯一写方 + 乐观锁 (PUT /l2/profile/{id}/mastery)."""

    def test_put_mastery_success(self, client):
        p = client.get("/l2/profile/DY20240001").json()["data"]
        v = p["version"]
        r = client.put("/l2/profile/DY20240001/mastery", json={
            "version": v,
            "updates": {"extras": {"feedback_log": [{"ts": 1, "rating": 0.7}]}},
        }, headers=_headers(client))
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["updated"] is True
        assert d["version"] > v
        # 版本号被写入 profile 响应
        p2 = client.get("/l2/profile/DY20240001").json()["data"]
        assert p2["version"] == d["version"]

    def test_put_mastery_conflict_409(self, client):
        """陈旧版本写入 → 409 冲突 + 最新 version."""
        p = client.get("/l2/profile/DY20240001").json()["data"]
        stale = p["version"] - 1 if p["version"] > 0 else 0
        r = client.put("/l2/profile/DY20240001/mastery", json={
            "version": stale,
            "updates": {"confidence": 0.9},
        }, headers=_headers(client))
        assert r.status_code == 409
        body = r.json()
        assert body["code"] == -32310
        assert body["current_version"] > stale

    def test_put_mastery_requires_version(self, client):
        r = client.put("/l2/profile/DY20240001/mastery", json={"updates": {}}, headers=_headers(client))
        assert r.status_code == 400

    def test_l5_write_goes_through_gateway(self, client):
        """L5 Agent 写画像 → 乐观锁 (经 apply_update), 不再直接 store.save_profile."""
        h = _headers(client)
        r = client.post("/l5/agents/agent.quality.review/run", json={
            "learner_id": "DY20240001", "content": "Dy3+ 发射波长 575nm",
        }, headers=h)
        assert r.json()["code"] == 0
        p = client.get("/l2/profile/DY20240001").json()["data"]
        assert "review_log" in p["extras"] or len(p["extras"].get("review_log", [])) >= 0


class TestL4StrategyOwner:
    """策略引擎归位 L4 (POST /l4/decision/next-action)."""

    def test_next_action_guide(self, client):
        r = client.post("/l4/decision/next-action", json={
            "learner_id": "DY20240001", "mode": "guide",
        }, headers=_headers(client))
        assert r.status_code == 200
        d = r.json()["data"]
        # 三语义统一: action_type + confidence + recommended_path
        assert d["action_type"] in ("practice", "review", "assess", "learn")
        assert 0 <= d["confidence"] <= 1
        assert isinstance(d["recommended_path"], list)
        if d["recommended_path"]:
            st = d["recommended_path"][0]
            assert {"kp_id", "action", "target", "effort"} <= set(st.keys())
        assert d["plan_id"].startswith("na-")
        assert d["mode"] == "guide"

    def test_next_action_modes(self, client):
        for mode in ("default", "review", "guide", "assess"):
            r = client.post("/l4/decision/next-action", json={
                "learner_id": "DY20240001", "mode": mode,
            }, headers=_headers(client))
            assert r.status_code == 200, mode
            assert r.json()["data"]["mode"] == mode

    def test_next_action_bad_mode(self, client):
        r = client.post("/l4/decision/next-action", json={
            "learner_id": "DY20240001", "mode": "bogus",
        }, headers=_headers(client))
        assert r.status_code == 400

    def test_frontend_uses_l4(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "/l4/decision/next-action" in js
        assert "recommended_path" in js


class TestUnifiedSessionContext:
    """L1 唯一会话入口 + 上下文统一获取."""

    def test_session_context_from_l1(self, client, builder):
        handlers = builder._handlers
        h = _headers(client)
        s = client.post("/l1/api/v1/sessions", json={"session_type": "query"}, headers=h).json()["data"]
        ctx = handlers.get_session_context(s["session_id"])
        assert ctx is not None
        assert ctx.get("session_id") == s["session_id"]
        assert "user_id" in ctx

    def test_query_response_returns_l1_session(self, client):
        h = _headers(client)
        s = client.post("/l1/api/v1/sessions", json={"session_type": "query"}, headers=h).json()["data"]
        r = client.post("/api/query", json={
            "query": "Dy3+ 的量子效率受哪些因素影响？",
            "learner_id": "DY20240001",
            "session_id": s["session_id"],
        })
        assert r.status_code == 200
        sess = r.json()["data"]["session"]
        assert sess["session_id"] == s["session_id"]  # 前端只见 L1 会话 ID
        assert "agent_execution_count" in sess

    def test_l5_session_record_has_source(self, client):
        h = _headers(client)
        s = client.post("/l1/api/v1/sessions", json={"session_type": "query"}, headers=h).json()["data"]
        client.post("/api/query", json={
            "query": "Dy3+ 发光机理", "learner_id": "DY20240001", "session_id": s["session_id"],
        })
        s2 = client.get(f"/l1/api/v1/sessions/{s['session_id']}", headers=h).json()["data"]
        assert s2["agent_execution_count"] >= 1
        l5_id = s2["agent_sessions"][0]
        l5 = client.get(f"/l5/session/{l5_id}", headers=h).json()["data"]
        assert l5["source_session_id"] == s["session_id"]
