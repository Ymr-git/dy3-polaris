"""M-F6 会话管理 + 设置 + i18n + 对比/时间旅行 + 响应式 — 集成测试."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def client() -> TestClient:
    builder = UnifiedApp.create_full_app_builder()
    return TestClient(builder.create_app())


@pytest.fixture(scope="module")
def headers(client) -> dict:
    r = client.post("/l1/api/v1/auth/login",
                    json={"student_id": "DY20240001", "password": "demo123"})
    tok = r.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {tok}"}


class TestSessionListApi:
    """GET /l1/api/v1/sessions — 列会话 (M-F6 新增)."""

    def test_list_requires_auth(self, client):
        assert client.get("/l1/api/v1/sessions").status_code == 401

    def test_list_empty_then_after_create(self, client, headers):
        r = client.get("/l1/api/v1/sessions", headers=headers)
        assert r.json()["code"] == 0
        before = r.json()["data"]["total"]

        r = client.post("/l1/api/v1/sessions", json={"session_type": "practice"},
                        headers=headers)
        assert r.json()["code"] == 0

        r = client.get("/l1/api/v1/sessions", headers=headers)
        assert r.json()["data"]["total"] == before + 1

    def test_list_active_filter(self, client, headers):
        r = client.get("/l1/api/v1/sessions?status=active", headers=headers)
        assert r.json()["code"] == 0
        for item in r.json()["data"]["items"]:
            assert item["status"] in ("active", "paused", "forked")

    def test_fork_and_pause(self, client, headers):
        r = client.get("/l1/api/v1/sessions", headers=headers)
        sid = r.json()["data"]["items"][0]["session_id"]

        r = client.post(f"/l1/api/v1/sessions/{sid}/fork",
                        json={"branch_label": "mf6"}, headers=headers)
        assert r.json()["code"] == 0

        r = client.post(f"/l1/api/v1/sessions/{sid}/pause", headers=headers)
        assert r.json()["code"] == 0


class TestSessionManagerMethod:
    """LearningSessionManager.list_sessions_for_user."""

    def test_method_exists_and_sorted(self):
        from dy3_polaris.l1.session_manager import LearningSessionManager
        mgr = LearningSessionManager()
        s1 = mgr.create_session("u-1", "diagnosis")
        s2 = mgr.create_session("u-1", "practice")
        sessions = mgr.list_sessions_for_user("u-1")
        assert [s.session_id for s in sessions] == [s2.session_id, s1.session_id]
        assert mgr.list_sessions_for_user("u-999") == []


class TestMF6Frontend:
    """M-F6 前端资源."""

    def test_mf6_js_served(self, client):
        resp = client.get("/static/assets/mf6-features.js")
        assert resp.status_code == 200
        js = resp.text
        assert "Dy3I18N" in js
        assert "/l1/api/v1/sessions" in js
        assert "MutationObserver" in js
        assert "renderCompare" in js
        assert "renderTimeTravel" in js

    def test_index_includes_mf6(self, client):
        html = client.get("/").text
        assert "mf6-features.js?v=" in html
        assert "app.js?v=" in html
        assert "ws-client.js?v=" in html

    def test_css_has_compare_grid(self, client):
        css = client.get("/static/assets/app.css").text
        assert ".grid cols-2" in css or "cols-2" in css


class TestI18NBackend:
    """L7 i18n 后端能力 (M-F6 语言切换依赖)."""

    def test_translate_supported(self):
        from dy3_polaris.l7.i18n.i18n_setup import translate, is_supported_locale
        assert is_supported_locale("zh-CN")
        assert is_supported_locale("en-US")
        zh = translate("dashboard", "zh-CN")
        en = translate("dashboard", "en-US")
        assert zh and en

    def test_i18n_config(self):
        from dy3_polaris.l7.i18n.i18n_setup import i18n_init_config
        cfg = i18n_init_config("zh-CN")
        assert "locale" in cfg or "resources" in cfg


class TestResponsiveBackend:
    """L7 响应式能力 (M-F6 响应式打磨)."""

    def test_breakpoint_for(self):
        from dy3_polaris.l7.responsive.layout_manager import breakpoint_for, layout_plan
        assert breakpoint_for(375) in ("mobile", "xs", "sm")
        assert breakpoint_for(1440) in ("desktop", "lg", "xl")
        plan = layout_plan(1280)
        assert "breakpoint" in plan
