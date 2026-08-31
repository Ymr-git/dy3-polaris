"""L7 API T6 — 错误码/认证/WebSocket/处理器单元测试."""

from __future__ import annotations

import pytest

from dy3_polaris.l7.api.artifact_api import (
    edit_artifact,
    get_artifact,
    list_artifacts,
)
from dy3_polaris.l7.api.auth import (
    ACCESS_TTL,
    REFRESH_TTL,
    AccessControl,
    TokenManager,
    extract_token,
)
from dy3_polaris.l7.api.dashboard_api import (
    bkt_dashboard,
    contribution_for_session,
    provenance_for_kp,
)
from dy3_polaris.l7.api.error_codes import (
    ERROR_CODES,
    all_error_codes,
    error_payload,
    http_status_for,
)
from dy3_polaris.l7.api.websocket import (
    CHANNELS,
    HEARTBEAT_INTERVAL,
    MAX_CONNECTIONS_PER_USER,
    RECONNECT_BACKOFF,
    ConnectionManager,
    build_message,
    reconnect_delay,
)
from dy3_polaris.l7.artifact_manager import ArtifactManager
from dy3_polaris.l7.models import Artifact, ArtifactDiff, ArtifactType


class TestErrorCodes:
    """错误码规范 (Ch.9.5)."""

    def test_all_eight_codes(self):
        assert len(all_error_codes()) == 8
        for code in ("RENDER_UNSUPPORTED_MIME", "RENDER_PAYLOAD_INVALID", "ARTIFACT_NOT_FOUND",
                     "ARTIFACT_READONLY", "EDIT_REJECTED", "SESSION_EXPIRED", "RATE_LIMITED",
                     "DASHBOARD_NO_DATA"):
            assert code in ERROR_CODES

    def test_http_status_mapping(self):
        assert http_status_for("ARTIFACT_NOT_FOUND") == 404
        assert http_status_for("ARTIFACT_READONLY") == 403
        assert http_status_for("EDIT_REJECTED") == 409
        assert http_status_for("SESSION_EXPIRED") == 401
        assert http_status_for("RATE_LIMITED") == 429
        assert http_status_for("DASHBOARD_NO_DATA") == 404
        assert http_status_for("UNKNOWN") == 500

    def test_error_payload_format(self):
        ep = error_payload("ARTIFACT_READONLY", trace_id="t-1", details={"field": "x"})
        assert ep["status"] == "error"
        assert ep["code"] == "ARTIFACT_READONLY"
        assert ep["trace_id"] == "t-1"
        assert ep["details"] == {"field": "x"}
        assert "message" in ep


class TestAuth:
    """认证与授权 (Ch.9.6)."""

    def test_issue_tokens(self):
        tm = TokenManager("secret")
        tokens = tm.issue_tokens("u1", "student")
        assert tokens["token_type"] == "Bearer"
        assert tokens["expires_in"] == ACCESS_TTL
        assert tokens["refresh_token"]

    def test_verify_access(self):
        tm = TokenManager("secret")
        tokens = tm.issue_tokens("u1", "student")
        payload = tm.verify(tokens["access_token"])
        assert payload is not None
        assert payload["sub"] == "u1"
        assert payload["role"] == "student"
        assert payload["typ"] == "access"

    def test_tampered_token_rejected(self):
        tm = TokenManager("secret")
        tokens = tm.issue_tokens("u1", "student")
        tampered = tokens["access_token"][:-4] + "AAAA"
        assert tm.verify(tampered) is None

    def test_refresh_flow(self):
        tm = TokenManager("secret")
        tokens = tm.issue_tokens("u1", "student")
        refreshed = tm.refresh(tokens["refresh_token"])
        assert refreshed is not None
        # 刷新后的 access token 必须可验证
        assert tm.verify(refreshed["access_token"]) is not None

    def test_revoke(self):
        tm = TokenManager("secret")
        tokens = tm.issue_tokens("u1", "student")
        tm.revoke(tokens["access_token"])
        assert tm.verify(tokens["access_token"]) is None

    def test_extract_token(self):
        assert extract_token("Bearer abc123") == "abc123"
        assert extract_token(None, "ws-token") == "ws-token"
        assert extract_token(None) is None
        assert extract_token("Basic xyz") is None

    def test_rbac_student(self):
        ac = AccessControl()
        assert ac.can_view_learner("student", "u1", "u1") is True
        assert ac.can_view_learner("student", "u1", "u2") is False

    def test_rbac_teacher(self):
        ac = AccessControl()
        assert ac.can_view_learner("teacher", "t1", "s1", {"s1"}) is True
        assert ac.can_view_learner("teacher", "t1", "s9", {"s1"}) is False

    def test_rbac_admin(self):
        ac = AccessControl()
        assert ac.can_view_learner("admin", "a", "anyone") is True


class TestWebSocket:
    """WebSocket 规范 (Ch.9.4)."""

    def test_channels(self):
        assert set(CHANNELS) == {"stream", "broadcast", "debate"}
        assert CHANNELS["stream"]["path"] == "/ws/stream"
        assert "artifact_created" in CHANNELS["stream"]["events"]
        assert "bkt_update" in CHANNELS["broadcast"]["events"]
        assert "speech" in CHANNELS["debate"]["events"]

    def test_reconnect_backoff(self):
        assert reconnect_delay(0) == 1.0
        assert reconnect_delay(1) == 2.0
        assert reconnect_delay(2) == 4.0
        assert reconnect_delay(5) == 30.0
        assert reconnect_delay(99) == 30.0  # 上限

    def test_connection_limit(self):
        cm = ConnectionManager()
        for i in range(3):
            assert cm.register("u1", f"c{i}") is True
        assert cm.register("u1", "c3") is False  # 第 4 个拒绝
        assert cm.count("u1") == 3

    def test_heartbeat(self):
        cm = ConnectionManager()
        cm.register("u1", "c1")
        assert cm.heartbeat_tick("u1", "c1") is True
        assert cm.heartbeat_tick("u1", "ghost") is False

    def test_unregister(self):
        cm = ConnectionManager()
        cm.register("u1", "c1")
        cm.unregister("u1", "c1")
        assert cm.count("u1") == 0

    def test_build_message(self):
        msg = build_message("artifact_created", {"id": "a1"}, channel="stream")
        assert msg["event_type"] == "artifact_created"
        assert msg["channel"] == "stream"
        assert msg["payload"]["id"] == "a1"
        assert msg["timestamp"] > 0


class TestArtifactApiHandlers:
    """Artifact API 处理器."""

    def _manager(self) -> ArtifactManager:
        mgr = ArtifactManager()
        art = Artifact(type=ArtifactType.TEXT, mime="text/vnd.dy3+markdown", payload={"content": "原始"})
        mgr.register(art)
        return mgr, art

    def test_list(self):
        mgr, _ = self._manager()
        result = list_artifacts(mgr)
        assert result["total"] == 1
        assert result["page"] == 1
        assert result["size"] == 20
        assert result["total_pages"] == 1

    def test_list_pagination(self):
        mgr = ArtifactManager()
        for i in range(5):
            mgr.register(Artifact(type=ArtifactType.TEXT, mime="text/vnd.dy3+markdown",
                                  payload={}, title=f"a{i}"))
        result = list_artifacts(mgr, page=2, size=2)
        assert len(result["items"]) == 2
        assert result["total"] == 5
        assert result["total_pages"] == 3

    def test_get_detail_versions(self):
        mgr, art = self._manager()
        mgr.update(art.artifact_id, {"content": "v2"})
        detail = get_artifact(mgr, art.artifact_id)
        assert detail["artifact_id"] == art.artifact_id
        assert len(detail["versions"]) == 2

    def test_get_missing(self):
        mgr, _ = self._manager()
        result = get_artifact(mgr, "missing")
        assert result["code"] == "ARTIFACT_NOT_FOUND"

    def test_edit(self):
        mgr, art = self._manager()
        diff = ArtifactDiff(artifact_id=art.artifact_id,
                            ops=[{"op": "replace", "path": "content", "value": "新值"}])
        edited = edit_artifact(mgr, art.artifact_id, diff)
        assert edited["payload"]["content"] == "新值"

    def test_edit_readonly(self):
        mgr = ArtifactManager()
        art = Artifact(type=ArtifactType.TEXT, mime="text/vnd.dy3+markdown", payload={}, editable=False)
        mgr.register(art)
        diff = ArtifactDiff(artifact_id=art.artifact_id, ops=[])
        result = edit_artifact(mgr, art.artifact_id, diff)
        assert result["code"] == "ARTIFACT_READONLY"

    def test_edit_missing(self):
        mgr, _ = self._manager()
        diff = ArtifactDiff(artifact_id="missing", ops=[])
        result = edit_artifact(mgr, "missing", diff)
        assert result["code"] == "ARTIFACT_NOT_FOUND"


class TestDashboardApiHandlers:
    """Dashboard API 处理器."""

    def test_bkt_full(self):
        mgr = ArtifactManager()
        bkt = {
            "A-01": {"p_l": 0.9, "p_k_l": 0.85, "p_g": 0.1, "p_s": 0.05},
            "A-04": {"p_l": 0.75, "p_k_l": 0.2, "p_g": 0.2, "p_s": 0.1},
        }
        result = bkt_dashboard(mgr, bkt)
        assert len(result["kp_states"]) == 2
        assert result["domain_summary"]["A"]["kp_count"] == 13
        assert len(result["bottleneck_kps"]) == 1  # A-04 瓶颈

    def test_bkt_from_manager(self):
        mgr = ArtifactManager()
        art = Artifact(type=ArtifactType.TEXT, mime="text/vnd.dy3+markdown", payload={},
                       learner_context={"bkt_state": {"A-01": {"p_l": 0.6}}})
        mgr.register(art)
        result = bkt_dashboard(mgr)
        assert len(result["kp_states"]) == 1
        assert result["kp_states"][0]["p_l"] == 0.6

    def test_provenance_no_data(self):
        result = provenance_for_kp("A-01", events=[])
        assert result["code"] == "DASHBOARD_NO_DATA"

    def test_provenance_sorted(self):
        events = [
            {"timestamp": 2000.0, "event_type": "decision", "agent_id": "A1", "summary": "后"},
            {"timestamp": 1000.0, "event_type": "knowledge", "agent_id": "A1", "summary": "先"},
        ]
        result = provenance_for_kp("A-01", events=events, depth="full")
        assert result["count"] == 2
        assert result["events"][0]["summary"] == "先"
        assert result["depth"] == "full"

    def test_contribution(self):
        agents = [
            {"id": "A1", "speech_count": 10, "citation_count": 5, "adopted_count": 3, "reputation_delta": 0.05},
        ]
        result = contribution_for_session("sess-1", agents)
        assert result["agents"]["A1"]["speech_count"] == 10
        assert result["agents"]["A1"]["reputation_delta"] == 0.05

    def test_contribution_no_data(self):
        result = contribution_for_session("sess-1", agents=[])
        assert result["code"] == "DASHBOARD_NO_DATA"
