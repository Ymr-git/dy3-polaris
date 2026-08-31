"""统一 session_id 命名空间 + 请求级 trace_id 中间件 专项测试."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.unified_app import UnifiedApp
from dy3_polaris.l5.tracing import get_trace_id, new_trace_id, set_trace_id, reset_trace_id


@pytest.fixture(scope="module")
def builder():
    return UnifiedApp.create_full_app_builder()


@pytest.fixture(scope="module")
def client(builder) -> TestClient:
    return TestClient(builder.create_app())


class TestSessionNamespace:
    """统一 session_id 命名空间 (shared/ids.py 单点)."""

    def test_shared_factory_prefixes(self):
        from dy3_polaris.shared.ids import new_session_id
        assert new_session_id("l1").startswith("sess-")
        assert new_session_id("l2").startswith("l2s-")
        assert new_session_id("l5").startswith("ag-")
        assert new_session_id("ws").startswith("ws-")
        assert new_session_id("tr").startswith("tr-")

    def test_l1_session_sess_prefix(self, client):
        login = client.post("/l1/api/v1/auth/login",
                            json={"student_id": "DY20240001", "password": "demo123"})
        h = {"Authorization": "Bearer " + login.json()["data"]["access_token"]}
        r = client.post("/l1/api/v1/sessions", json={"session_type": "query"}, headers=h)
        assert r.json()["data"]["session_id"].startswith("sess-")

    def test_l5_session_ag_prefix(self, client):
        """L5 执行会话 ag- 前缀, 经 source_session_id 关联 L1 (sess-)."""
        builder = UnifiedApp.create_full_app_builder()
        mgr = builder.bridge.session_manager
        rec = mgr.create_session(
            agent_id="agent.test.demo", learner_id="stu-001",
            source_session_id="sess-abc123",
        )
        assert rec.session_id.startswith("ag-")
        assert rec.source_session_id == "sess-abc123"
        # 反查: 按 L1 会话聚合执行会话
        by_source = mgr.get_sessions_by_source("sess-abc123")
        assert rec.session_id in {s.session_id for s in by_source}

    def test_l2_session_l2s_prefix(self, client):
        from dy3_polaris.l2.session.session_manager import SessionManager as L2SM
        sid = L2SM().start_session("learner-001")
        assert sid.startswith("l2s-")
        assert len(sid[len("l2s-"):]) == 12

    def test_l6_a2a_prefix(self):
        from dy3_polaris.l6.a2a.protocol import create_session_id
        a = create_session_id("agent.a", "agent.b")
        b = create_session_id("agent.b", "agent.a")
        assert a.startswith("a2a-")
        assert a == b  # 确定性保持

    def test_frontend_uses_l1_session(self, client):
        """前端 /api/query 传 L1 sess- 会话, 响应返回 L1 会话 (唯一入口)."""
        login = client.post("/l1/api/v1/auth/login",
                            json={"student_id": "DY20240001", "password": "demo123"})
        h = {"Authorization": "Bearer " + login.json()["data"]["access_token"]}
        s = client.post("/l1/api/v1/sessions", json={"session_type": "query"}, headers=h).json()["data"]
        assert s["session_id"].startswith("sess-")
        r = client.post("/api/query", json={
            "query": "Dy3+ 发光机理", "learner_id": "DY20240001",
            "session_id": s["session_id"],
        })
        body = r.json()
        data = body.get("data") or body
        assert data["session"]["session_id"] == s["session_id"]
        assert data["session"]["agent_execution_count"] >= 1


class TestTraceIDMiddleware:
    """请求级 trace_id: contextvars 注入 + 响应头 + 错误体回填."""

    def test_trace_header_on_response(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        tid = r.headers.get("x-trace-id")
        assert tid and tid.startswith("tr-")

    def test_error_body_backfill(self, client):
        """错误响应 (code != 0) 自动回填 trace_id."""
        r = client.post("/l2/practice/answer", json={"learner_id": "x", "qid": "no-such"})
        assert r.status_code != 200
        body = r.json()
        assert body["code"] != 0
        assert body.get("trace_id", "").startswith("tr-")
        # 与响应头一致
        assert body["trace_id"] == r.headers.get("x-trace-id")

    def test_trace_id_forwarding(self, client):
        """透传 X-Trace-Id: 传入的自定义 ID 被保留."""
        r = client.get("/l2/kp-catalog", headers={"X-Trace-Id": "tr-custom123456"})
        assert r.headers.get("x-trace-id") == "tr-custom123456"

    def test_contextvars_injection(self, client):
        """中间件注入的 trace_id 在请求内可读 (contextvars)."""
        token = set_trace_id("tr-unit-test-001")
        try:
            assert get_trace_id() == "tr-unit-test-001"
        finally:
            reset_trace_id(token)
        assert new_trace_id().startswith("tr-")

    def test_l3_retrieval_trace_wired(self, client):
        """L3 检索结果模型支持 trace_id 字段 (请求级注入契约)."""
        from dy3_polaris.l3.models import RetrievalResult
        r = RetrievalResult(query="Dy3+", results=[], scores=[], total=0,
                            retrieval_time_ms=0.0, source_type="vector", trace_id="tr-test-1")
        assert r.trace_id == "tr-test-1"

    def test_l7_error_payload_backfill(self):
        """L7 error_payload 无显式 trace_id 时回填上下文."""
        from dy3_polaris.l7.api.error_codes import error_payload
        token = set_trace_id("tr-l7-test-001")
        try:
            payload = error_payload("DASHBOARD_NO_DATA")
            assert payload["trace_id"] == "tr-l7-test-001"
        finally:
            reset_trace_id(token)
        # 无上下文时为空串 (直连场景兼容)
        assert error_payload("DASHBOARD_NO_DATA")["trace_id"] == ""
