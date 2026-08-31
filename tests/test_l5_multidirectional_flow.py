"""多向 Agent 流程专项测试: 诊断广播 / 知识生成出题考核 / 决策审核写回画像.

覆盖用户需求: ① 诊断了解薄弱块后广播给其他 3 个 Agent;
② 知识生成调用画像针对性出题练习; ③ 知识生成按画像考核并写回画像、广播用户情况;
④ 决策→画像、审核→画像 逆向写回; ⑤ 任意方向跨频道传播。
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.communication import Message
from dy3_polaris.l5.unified_app import UnifiedApp


@pytest.fixture(scope="module")
def env():
    """共享 builder (client 与总线同实例, 可验证广播)."""
    builder = UnifiedApp.create_full_app_builder()
    client = TestClient(builder.create_app())
    events: list[tuple[str, str, str]] = []

    def on_msg(m: Message):
        events.append((m.channel, m.publisher, m.payload.get("event", "")))

    for ch in (
        "learning.knowledge.gap", "learning.diagnosis.report",
        "knowledge.generation.output", "knowledge.review.result",
        "guidance.decision.command", "learning.interaction.event",
    ):
        try:
            builder.bridge.message_bus.subscribe(ch, "test-observer", on_msg)
        except Exception:
            pass
    return {"client": client, "bus": builder.bridge.message_bus, "events": events}


def _headers(client):
    login = client.post(
        "/l1/api/v1/auth/login",
        json={"student_id": "DY20248888", "password": "admin888"},
    )
    return {"Authorization": "Bearer " + login.json()["data"]["access_token"]}


def _run(client, aid, payload):
    r = client.post(f"/l5/agents/{aid}/run", json=payload, headers=_headers(client))
    return r.json().get("data", {})


class TestDiagnosisBroadcast:
    """诊断 Agent 广播薄弱块给其他 Agent."""

    def test_diagnosis_broadcasts_gap(self, env):
        before = len(env["events"])
        d = _run(env["client"], "agent.learning.diagnosis", {"learner_id": "DY20240001"})
        assert d["status"] == "completed"
        new = [
            e for e in env["events"][before:]
            if e[0] == "learning.knowledge.gap" and e[1] == "agent.learning.diagnosis"
        ]
        assert len(new) >= 1
        assert new[0][2] == "diagnosis_report"

    def test_diagnosis_report_channel(self, env):
        before = len(env["events"])
        _run(env["client"], "agent.learning.diagnosis", {"learner_id": "DY20240001"})
        new = [
            e for e in env["events"][before:]
            if e[0] == "learning.diagnosis.report" and e[1] == "agent.learning.diagnosis"
        ]
        assert len(new) >= 1


class TestGenerationPractice:
    """知识生成 Agent 调用画像针对性出题练习."""

    def test_practice_mode_reads_profile(self, env):
        d = _run(env["client"], "agent.knowledge.generation", {
            "learner_id": "DY20240001", "mode": "practice", "count": 5,
        })
        assert d["mode"] == "practice"
        assert d["status"] == "completed"
        assert 0 < d["count"] <= 5
        # 题目不泄露答案
        for q in d["questions"]:
            assert "answer" not in q
            assert "explanation" not in q
            assert q["kp_id"]

    def test_practice_targets_weak_kps(self, env):
        p = env["client"].get("/l2/profile/DY20240001").json()["data"]
        weak = set(p["weak_kps"])
        d = _run(env["client"], "agent.knowledge.generation", {
            "learner_id": "DY20240001", "mode": "practice", "count": 5,
        })
        assert d["target_kps"]
        assert weak.intersection(d["target_kps"])

    def test_practice_issues_broadcast(self, env):
        before = len(env["events"])
        _run(env["client"], "agent.knowledge.generation", {
            "learner_id": "DY20240001", "mode": "practice", "count": 3,
        })
        new = [
            e for e in env["events"][before:]
            if e[1] == "agent.knowledge.generation" and e[2] == "practice_issued"
        ]
        assert len(new) >= 1


class TestGenerationAssess:
    """知识生成 Agent 按画像考核 → BKT → 画像写回 → 广播."""

    def test_assess_issues_questions(self, env):
        d = _run(env["client"], "agent.knowledge.generation", {
            "learner_id": "DY20240001", "mode": "assess", "count": 3,
        })
        assert d["mode"] == "assess"
        assert d["status"] == "pending_answers"
        assert len(d["questions"]) == 3
        assert d["bloom_target"]
        assert d["learning_style"]

    def test_assess_grades_and_writes_profile(self, env):
        issued = _run(env["client"], "agent.knowledge.generation", {
            "learner_id": "DY20240001", "mode": "assess", "count": 2,
        })
        kp0 = issued["questions"][0]["kp_id"]
        p0 = env["client"].get("/l2/profile/DY20240001").json()["data"]
        before = p0["kp_mastery"].get(kp0, 0.0)

        d = _run(env["client"], "agent.knowledge.generation", {
            "learner_id": "DY20240001", "mode": "assess",
            "answers": [{"qid": issued["questions"][0]["qid"], "selected": 0},
                        {"qid": issued["questions"][1]["qid"], "selected": 1}],
        })
        assert d["status"] == "completed"
        assert "score" in d
        # 画像写回: assess_log + kp_mastery 联动
        p1 = env["client"].get("/l2/profile/DY20240001").json()["data"]
        assert p1["extras"]["assess_log"]
        assert abs(p1["kp_mastery"].get(kp0, 0.0) - before) > 1e-9

    def test_assess_broadcasts_user_state(self, env):
        before = len(env["events"])
        issued = _run(env["client"], "agent.knowledge.generation", {
            "learner_id": "DY20240001", "mode": "assess", "count": 1,
        })
        qid = issued["questions"][0]["qid"]
        _run(env["client"], "agent.knowledge.generation", {
            "learner_id": "DY20240001", "mode": "assess",
            "answers": [{"qid": qid, "selected": 0}],
        })
        new = [e for e in env["events"][before:] if e[1] == "agent.knowledge.generation"]
        assert any(e[2] == "assess_report" for e in new)
        assert any(e[2] == "assess_completed" for e in new)


class TestReverseWriteBacks:
    """逆向链路: 决策→画像, 审核→画像."""

    def test_decision_writes_profile(self, env):
        _run(env["client"], "agent.guidance.decision", {
            "learner_id": "DY20240001", "query": "Dy3+ 的量子效率受哪些因素影响？",
        })
        p = env["client"].get("/l2/profile/DY20240001").json()["data"]
        assert p["extras"]["decision_log"]
        assert p["extras"]["decision_log"][-1]["decision"] in ("direct", "needs_confirmation")

    def test_review_writes_profile(self, env):
        _run(env["client"], "agent.quality.review", {
            "learner_id": "DY20240001",
            "content": "Dy3+ 离子的发射波长为 575nm，来自 4F9/2 -> 6H13/2 跃迁",
        })
        p = env["client"].get("/l2/profile/DY20240001").json()["data"]
        assert p["extras"]["review_log"]
        assert p["extras"]["review_log"][-1]["verdict"] in (
            "approved", "needs_review", "rejected", "skipped",
        )

    def test_generation_output_channel(self, env):
        before = len(env["events"])
        _run(env["client"], "agent.knowledge.generation", {
            "learner_id": "DY20240001", "query": "Dy3+ 的量子效率受哪些因素影响？",
        })
        new = [
            e for e in env["events"][before:]
            if e[0] == "knowledge.generation.output" and e[1] == "agent.knowledge.generation"
        ]
        assert len(new) >= 1

    def test_decision_command_channel(self, env):
        before = len(env["events"])
        _run(env["client"], "agent.guidance.decision", {
            "learner_id": "DY20240001", "query": "Dy3+ 的量子效率受哪些因素影响？",
        })
        new = [
            e for e in env["events"][before:]
            if e[0] == "guidance.decision.command" and e[1] == "agent.guidance.decision"
        ]
        assert len(new) >= 1


class TestFrontendMultidirectional:
    """前端多向链路与模式按钮."""

    def test_mf6_js_has_multidirectional(self, client=None):
        from starlette.testclient import TestClient

        from dy3_polaris.l5.unified_app import UnifiedApp

        c = TestClient(UnifiedApp.create_full_app_builder().create_app())
        js = c.get("/static/assets/mf6-features.js").text
        assert "多向传播" in js
        assert "考核→画像" in js
        assert "决策→画像" in js
        assert "data-mode" in js
        assert "mode: 'assess'" in js
        assert "mode: 'practice'" in js
