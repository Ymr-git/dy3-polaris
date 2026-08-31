"""M-F5 WebSocket 实时三通道 + 辩论面板 — 集成测试."""
from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from dy3_polaris.l5.unified_app import UnifiedApp
from dy3_polaris.l7.api.websocket import HUB, build_message, reconnect_delay


@pytest.fixture(scope="module")
def client() -> TestClient:
    builder = UnifiedApp.create_full_app_builder()
    return TestClient(builder.create_app())


@pytest.fixture(scope="module")
def auth_token(client) -> str:
    r = client.post("/l1/api/v1/auth/login",
                    json={"student_id": "DY20240001", "password": "demo123"})
    assert r.json()["code"] == 0, r.text
    return r.json()["data"]["access_token"]


class TestWSRoutes:
    """三通道路由挂载."""

    def test_routes_mounted(self, client):
        app = client.app
        ws_paths = sorted(p for r in app.routes for p in [getattr(r, "path", "")]
                          if p.startswith("/ws/"))
        assert ws_paths == ["/ws/broadcast", "/ws/debate", "/ws/stream"]

    def test_unauthorized_rejected(self, client):
        for path in ("/ws/stream", "/ws/broadcast", "/ws/debate"):
            with pytest.raises(WebSocketDisconnect) as ei:
                with client.websocket_connect(path):
                    pass
            assert ei.value.code == 4401


class TestWSLive:
    """认证连接 + 心跳 + 事件推送."""

    def test_heartbeat_ping_pong(self, client, auth_token):
        with client.websocket_connect("/ws/debate?token=" + auth_token) as ws:
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"

    def test_debate_event_stream(self, client, auth_token):
        with client.websocket_connect("/ws/debate?token=" + auth_token) as ws:
            HUB.broadcast("debate", "debate_start",
                          {"topic": "量子效率", "agents": ["A1", "A2"]})
            ev = ws.receive_json()
            assert ev["event_type"] == "debate_start"
            assert ev["channel"] == "debate"
            assert ev["payload"]["topic"] == "量子效率"

            HUB.broadcast("debate", "speech",
                          {"agent": "A1", "stance": "support", "summary": "支持"})
            ev = ws.receive_json()
            assert ev["event_type"] == "speech"
            assert ev["payload"]["stance"] == "support"

            HUB.broadcast("debate", "convergence",
                          {"rounds": [1, 2], "consensus": [0.5, 0.8]})
            ev = ws.receive_json()
            assert ev["event_type"] == "convergence"
            assert ev["payload"]["consensus"] == [0.5, 0.8]

            HUB.broadcast("debate", "end", {"selected_agent": "A1"})
            ev = ws.receive_json()
            assert ev["event_type"] == "end"

    def test_broadcast_bkt_event(self, client, auth_token):
        with client.websocket_connect("/ws/broadcast?token=" + auth_token) as ws:
            HUB.broadcast("broadcast", "bkt_update",
                          {"kp_id": "KP-01", "old": 0.3, "new": 0.6})
            ev = ws.receive_json()
            assert ev["event_type"] == "bkt_update"
            assert ev["payload"]["kp_id"] == "KP-01"

    def test_stream_artifact_event(self, client, auth_token):
        with client.websocket_connect("/ws/stream?token=" + auth_token) as ws:
            HUB.broadcast("stream", "artifact_created", {"artifact_id": "a-1"})
            ev = ws.receive_json()
            assert ev["event_type"] == "artifact_created"

    def test_channel_isolation(self, client, auth_token):
        """broadcast 通道不应收到 debate 事件."""
        with client.websocket_connect("/ws/broadcast?token=" + auth_token) as ws:
            HUB.broadcast("debate", "speech", {"agent": "A1"})
            # 无事件 → 发送 ping 应只回 pong
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"


class TestResilience:
    """韧性机制: 退避 / 连接限制 / 消息格式."""

    def test_reconnect_backoff(self):
        assert reconnect_delay(0) == 1.0
        assert reconnect_delay(1) == 2.0
        assert reconnect_delay(5) == 30.0
        assert reconnect_delay(99) == 30.0  # 上限

    def test_max_connections(self, client, auth_token):
        """单用户最多 3 连接, 第 4 个被拒."""
        sockets = []
        try:
            for _ in range(3):
                ws = client.websocket_connect("/ws/debate?token=" + auth_token)
                ws.__enter__()
                sockets.append(ws)
            with pytest.raises(WebSocketDisconnect) as ei:
                ws4 = client.websocket_connect("/ws/debate?token=" + auth_token)
                ws4.__enter__()
            assert ei.value.code == 4429
        finally:
            for ws in sockets:
                try:
                    ws.__exit__(None, None, None)
                except Exception:
                    pass

    def test_build_message_format(self):
        m = build_message("bkt_update", {"kp_id": "KP-01"}, "broadcast")
        assert set(m.keys()) == {"event_type", "channel", "timestamp", "payload"}

    def test_hub_auth_with_l1_jwt(self, client, auth_token):
        """HUB 使用 L1 JWTManager 验证."""
        assert HUB._token_manager is not None
        claims = HUB.authenticate(auth_token)
        assert claims and claims.get("user_id")


class TestDebateFrontend:
    """辩论面板前端资源."""

    def test_ws_client_served(self, client):
        resp = client.get("/static/assets/ws-client.js")
        assert resp.status_code == 200
        js = resp.text
        assert "Dy3WS" in js
        assert "'/ws/' + channel" in js or "/ws/" in js
        assert "BACKOFF" in js
        assert "ping" in js

    def test_index_has_debate_panel(self, client):
        html = client.get("/").text
        assert "debatePanel" in html
        assert "ws-client.js" in html

    def test_css_has_debate_styles(self, client):
        css = client.get("/static/assets/app.css").text
        assert ".debate-timeline" in css
        assert ".debate-speech" in css
