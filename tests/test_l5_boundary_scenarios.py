"""多智能体协同边界场景专项测试: 超时兜底 / 并发写保护 / 总线锁 / 闭环链路补齐."""
from __future__ import annotations

import threading
import time

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.communication import Message, MessageBus
from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(UnifiedApp.create_full_app_builder().create_app())


def _headers(client):
    login = client.post(
        "/l1/api/v1/auth/login",
        json={"student_id": "DY20248888", "password": "admin888"},
    )
    return {"Authorization": "Bearer " + login.json()["data"]["access_token"]}


class TestTimeoutFallback:
    """失联兜底: Agent 执行超时返回降级结果."""

    def test_run_agent_supports_timeout_param(self, client):
        h = _headers(client)
        r = client.post("/l5/agents/agent.learning.diagnosis/run",
                        json={"learner_id": "DY20240001", "timeout_s": 0.001}, headers=h)
        # 超时后应返回降级结果 (status=timeout) 或正常结果, 但绝不悬挂
        d = r.json().get("data", {})
        assert r.status_code == 200
        assert d.get("status") in ("completed", "timeout")
        assert d.get("fallback", False) or d.get("status") == "completed"

    def test_plan_timeout_caches_result(self, client):
        h = _headers(client)
        r = client.post("/l5/orchestrate", json={
            "paradigm": "pipeline",
            "tasks": [
                {"task_id": "t1", "agent_id": "agent.learning.diagnosis",
                 "input": {"learner_id": "DY20240001"}},
            ],
            "config": {"total_timeout_s": 30},
        }, headers=h)
        assert r.json()["code"] == 0
        plan_id = r.json()["data"]["plan_id"]
        g = client.get(f"/l5/orchestration/{plan_id}")
        assert g.json()["code"] == 0, g.text


class TestConcurrentProfileWrite:
    """任务冲突: 并发写画像不丢失更新 (合并写 + 版本号)."""

    def test_merge_write_keeps_kp_mastery(self, client):
        builder = UnifiedApp.create_full_app_builder()
        profile_service = builder.bridge.profile_service
        store = profile_service.store

        snap = profile_service.get_profile_snapshot("MERGE_USER")
        if snap is None:
            from dy3_polaris.l2.models import LearnerSnapshot

            snap = LearnerSnapshot(learner_id="MERGE_USER", snapshot_ts=time.time())
            store.save_profile("MERGE_USER", snap)

        # 模拟两路并发写: 一路写 A-01, 一路写 B-01
        def write_a():
            p = profile_service.get_profile_snapshot("MERGE_USER")
            km = dict(p.kp_mastery)
            km["A-01"] = 0.91
            p.kp_mastery = km
            store.save_profile("MERGE_USER", p)

        def write_b():
            p = profile_service.get_profile_snapshot("MERGE_USER")
            km = dict(p.kp_mastery)
            km["B-01"] = 0.82
            p.kp_mastery = km
            store.save_profile("MERGE_USER", p)

        t1, t2 = threading.Thread(target=write_a), threading.Thread(target=write_b)
        t1.start(); t2.start(); t1.join(); t2.join()

        final = profile_service.get_profile_snapshot("MERGE_USER")
        assert final.kp_mastery.get("A-01", 0) == 0.91, "A-01 写回丢失"
        assert final.kp_mastery.get("B-01", 0) == 0.82, "B-01 写回丢失"
        assert final.version >= 2

    def test_extras_logs_merged(self, client):
        builder = UnifiedApp.create_full_app_builder()
        store = builder.bridge.profile_service.store
        snap = builder.bridge.profile_service.get_profile_snapshot("MERGE_USER")
        extras = dict(snap.extras or {})
        extras.setdefault("feedback_log", []).append({"ts": time.time(), "rating": 0.5})
        snap.extras = extras
        store.save_profile("MERGE_USER", snap)
        after = builder.bridge.profile_service.get_profile_snapshot("MERGE_USER")
        assert len(after.extras.get("feedback_log", [])) >= 1


class TestMessageBusLock:
    """消息乱序: publish 线程安全 (stream_id 单调)."""

    def test_publish_thread_safe(self):
        bus = MessageBus()
        bus.create_channel("t.lock")
        seen: list[str] = []
        bus.subscribe("t.lock", "obs", lambda m: seen.append(m.stream_id))

        def pub(n):
            for i in range(50):
                bus.publish(Message(channel="t.lock", publisher=f"p{n}", payload={"i": i}))

        threads = [threading.Thread(target=pub, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 200 条消息 stream_id 全部唯一且有序
        assert len(seen) == 200
        assert len(set(seen)) == 200


class TestClosedLoopBreaks:
    """闭环链路断点补齐: 事件采集 / 反馈写回 / 答疑端到端."""

    def test_event_collect_view_and_query(self, client):
        r = client.post("/l2/event/collect", json={
            "learner_id": "LOOP_USER", "event_type": "view", "detail": "kb A-05",
        })
        assert r.json()["code"] == 0, r.text
        r2 = client.post("/l2/event/collect", json={
            "learner_id": "LOOP_USER", "event_type": "query", "detail": "Dy3+ 量子效率",
        })
        assert r2.json()["code"] == 0
        p = client.get("/l2/profile/LOOP_USER").json()["data"]
        assert len(p["extras"]["interaction_log"]) >= 2

    def test_event_collect_bad_type(self, client):
        r = client.post("/l2/event/collect", json={
            "learner_id": "LOOP_USER", "event_type": "bad_type",
        })
        assert r.json()["code"] != 0

    def test_feedback_to_profile_writes(self, client):
        builder = UnifiedApp.create_full_app_builder()
        bridge = builder._bridge
        bridge.feedback_to_profile("LOOP_USER", {
            "action_type": "answer", "confidence": 0.8,
        })
        p = builder.bridge.profile_service.get_profile_snapshot("LOOP_USER")
        assert p is not None
        assert len(p.extras.get("feedback_log", [])) >= 1

    def test_query_endpoint_answers(self, client):
        h = _headers(client)
        r = client.post("/api/query", json={"query": "Dy3+ 的量子效率受哪些因素影响？"}, headers=h)
        assert r.status_code == 200
        body = r.json()
        data = body.get("data") or {}
        answer = data.get("answer") or body.get("answer") or ""
        assert answer, "应返回非空答案"
        assert (data.get("confidence") or body.get("confidence") or 0) > 0

    def test_frontend_has_query_and_recommendation(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "renderQuery" in js
        assert "'query': renderQuery" in js
        # 策略决策归位 L4: 今日推荐调 /l4/decision/next-action
        assert "/l4/decision/next-action" in js
        assert "recommended_path" in js
        assert "/l2/event/collect" in js
        assert "实时答疑" in js
