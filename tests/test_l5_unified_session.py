"""统一会话闭环专项测试: L1 唯一入口 + L5 关联 + 前端接线."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(UnifiedApp.create_full_app_builder().create_app())


def _headers(client, student="DY20248888", pwd="admin888"):
    login = client.post(
        "/l1/api/v1/auth/login",
        json={"student_id": student, "password": pwd},
    )
    return {"Authorization": "Bearer " + login.json()["data"]["access_token"]}


class TestUnifiedSessionClosedLoop:
    """统一会话闭环: L1 用户会话聚合 L5 执行会话."""

    def test_session_type_query(self, client):
        """L1 支持 query 会话类型."""
        h = _headers(client)
        r = client.post("/l1/api/v1/sessions", json={"session_type": "query"}, headers=h)
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["session_type"] == "query"
        assert data["question_count"] == 0
        assert data["agent_sessions"] == []
        assert data["agent_execution_count"] == 0

    def test_query_attaches_agent_session(self, client):
        """实时答疑携带 L1 会话 ID → L5 执行会话自动关联到 L1."""
        h = _headers(client)
        s = client.post("/l1/api/v1/sessions", json={"session_type": "query"}, headers=h).json()["data"]
        l1_id = s["session_id"]

        r = client.post("/api/query", json={
            "query": "Dy3+ 的量子效率受哪些因素影响？",
            "learner_id": "DY20240001",
            "session_id": l1_id,
        })
        assert r.status_code == 200
        body = r.json()
        data = body.get("data") or {}
        answer = data.get("answer") or body.get("answer") or ""
        assert answer

        # L1 会话应已聚合: 提问数 +1, 关联 1 个 L5 执行会话
        s2 = client.get(f"/l1/api/v1/sessions/{l1_id}", headers=h).json()["data"]
        assert s2["question_count"] >= 1
        assert s2["agent_execution_count"] >= 1
        assert len(s2["agent_sessions"]) >= 1
        # L5 执行会话带 source_session_id
        l5_id = s2["agent_sessions"][0]
        l5 = client.get(f"/l5/session/{l5_id}", headers=h).json()["data"]
        assert l5["source_session_id"] == l1_id

    def test_l1_list_shows_attached(self, client):
        """L1 会话列表可见统一会话记录 (前端 query-history 数据源)."""
        h = _headers(client)
        items = client.get("/l1/api/v1/sessions", headers=h).json()["data"]["items"]
        query_sessions = [s for s in items if s["session_type"] == "query"]
        assert len(query_sessions) >= 1
        assert all("question_count" in s and "agent_execution_count" in s for s in query_sessions)

    def test_attach_endpoint_direct(self, client):
        """L1 attach-agent-session 端点可直接关联 (跨层内部接口)."""
        h = _headers(client)
        s = client.post("/l1/api/v1/sessions", json={"session_type": "query"}, headers=h).json()["data"]
        r = client.post(
            f"/l1/api/v1/sessions/{s['session_id']}/attach-agent-session",
            json={"agent_session_id": "sess-agent-xyz"},
            headers=h,
        )
        assert r.status_code == 200
        assert r.json()["data"]["question_count"] == 1
        assert "sess-agent-xyz" in r.json()["data"]["agent_sessions"]

    def test_frontend_wired(self, client):
        """前端已接线统一会话: query 视图创建 L1 会话 + 提问带 session_id + 会话管理显示关联."""
        js = client.get("/static/assets/mf6-features.js").text
        assert "session_type: 'query'" in js
        assert "dy3_query_session_" in js
        assert "session_id: l1id" in js
        assert "agent_execution_count" in js
        assert "实时答疑" in js
