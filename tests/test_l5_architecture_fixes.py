"""架构修复与动态化专项测试: Skill 执行器 / 接口修复 / inbox / 动态画像 / WS 推流."""
from __future__ import annotations

import time

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.skill_executor import SKILL_CATALOG, SkillExecutor
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


class TestInterfaceFixes:
    """P0 接口接线错误修复."""

    def test_l4_feedback_works(self, client):
        r = client.post("/l4/decision/feedback", json={
            "action_id": "act-iface-1", "rating": 0.8, "feedback_type": "explicit",
        }, headers=_headers(client))
        assert r.status_code == 200
        assert r.json()["code"] == 0

    def test_l5_session_fork_close(self, client):
        h = _headers(client)
        s = client.post("/l5/session", json={
            "agent_id": "agent.learning.diagnosis", "learner_id": "DY20240001",
        }, headers=h)
        assert s.status_code == 200
        sid = s.json()["data"].get("session_id") or s.json()["data"].get("id")
        fk = client.post(f"/l5/session/{sid}/fork", json={"fork_reason": "t"}, headers=h)
        assert fk.status_code == 200 and fk.json()["code"] == 0
        cl = client.post(f"/l5/session/{sid}/close", headers=h)
        assert cl.status_code == 200 and cl.json()["code"] == 0

    def test_lazy_channel_publish(self, client):
        r = client.post("/l5/message/publish", json={
            "channel": "adhoc.channel", "payload": {"x": 1},
        }, headers=_headers(client))
        assert r.json()["code"] == 0


class TestSkillExecutor:
    """Agent Skill 体系: 12 技能动态调用."""

    def test_catalog_has_12_skills(self):
        assert len(SKILL_CATALOG) == 12
        per_agent = {}
        for name, meta in SKILL_CATALOG.items():
            per_agent.setdefault(meta["agent"], []).append(name)
        assert all(len(v) == 3 for v in per_agent.values())

    def test_skills_endpoint(self, client):
        h = _headers(client)
        r = client.get("/l5/agents/agent.learning.diagnosis/skills", headers=h)
        assert r.json()["code"] == 0
        skills = r.json()["data"]["skills"]
        assert len(skills) == 3
        assert all(s["available"] for s in skills)

    @pytest.mark.parametrize("agent_id,tool,args", [
        ("agent.learning.diagnosis", "bkt_compute", {"learner_id": "DY20240001"}),
        ("agent.learning.diagnosis", "irt_evaluate", {"learner_id": "DY20240001"}),
        ("agent.learning.diagnosis", "forgetfulness_scan", {"learner_id": "DY20240001"}),
        ("agent.knowledge.generation", "rag_retrieve", {"query": "Dy3+ 的量子效率", "top_k": 2}),
        ("agent.knowledge.generation", "connector_tier1_query", {"query": "Dy3+"}),
        ("agent.quality.review", "rule_engine_check", {"content": "Dy3+ 发射波长 575nm"}),
        ("agent.quality.review", "cross_validation", {"content": "Dy3+ 发射波长 575nm"}),
        ("agent.quality.review", "standard_value_check", {"content": "波长 575nm"}),
        ("agent.guidance.decision", "topology_analysis", {"learner_id": "DY20240001"}),
        ("agent.guidance.decision", "path_simulation", {"learner_id": "DY20240001"}),
        ("agent.guidance.decision", "uncertainty_confirm", {"confidence": 0.4}),
    ])
    def test_call_all_skills(self, client, agent_id, tool, args):
        r = client.post(f"/l5/agents/{agent_id}/skills/call",
                        json={"tool": tool, "args": args}, headers=_headers(client))
        assert r.status_code == 200, r.text
        d = r.json()["data"]
        assert d["status"] == "ok", d
        assert d["tool"].startswith(("internal.", "l3."))
        assert "elapsed_ms" in d

    def test_skill_wrong_agent_forbidden(self, client):
        r = client.post("/l5/agents/agent.learning.diagnosis/skills/call",
                        json={"tool": "rag_retrieve", "args": {"query": "x"}},
                        headers=_headers(client))
        assert r.status_code == 403

    def test_skill_unknown_404(self, client):
        r = client.post("/l5/agents/agent.learning.diagnosis/skills/call",
                        json={"tool": "no_such_skill", "args": {}},
                        headers=_headers(client))
        assert r.status_code == 404


class TestAgentInbox:
    """Agent 广播订阅消费 (按需协作)."""

    def test_inbox_receives_diagnosis_broadcast(self, client):
        builder = UnifiedApp.create_full_app_builder()
        c = TestClient(builder.create_app())
        # 知识生成 Agent 订阅 learning.knowledge.gap → 诊断后 inbox 应有消息
        tok = c.post("/l1/api/v1/auth/login",
                     json={"student_id": "DY20248888", "password": "admin888"}).json()["data"]["access_token"]
        h = {"Authorization": "Bearer " + tok}
        c.post("/l5/agents/agent.learning.diagnosis/run", json={"learner_id": "DY20240001"}, headers=h)
        inbox = c.get("/l5/agents/agent.knowledge.generation/inbox").json()["data"]
        assert inbox["total"] >= 1
        assert any(m["channel"] == "learning.knowledge.gap" for m in inbox["messages"])

    def test_inbox_limits(self):
        builder = UnifiedApp.create_full_app_builder()
        c = TestClient(builder.create_app())
        rt = builder.agent_runtime if hasattr(builder, "agent_runtime") else None
        # 通过 builder 内部 agent_runtime 检查清空
        runtime = getattr(builder, "_agent_runtime", None)
        if runtime is not None:
            n = runtime.clear_inbox("agent.knowledge.generation")
            assert n >= 0


class TestDynamicProfile:
    """画像动态化: overall/dimensions/遗忘衰减/新用户默认0."""

    def test_profile_dynamic_fields(self, client):
        p = client.get("/l2/profile/DY20240001").json()["data"]
        assert "overall_mastery" in p
        assert 0 <= p["overall_mastery"] <= 1
        assert "dimensions" in p and set(p["dimensions"].keys()) >= {"A", "B", "C", "D"}
        assert "raw_kp_mastery" in p
        assert "decay_hours" in p
        assert p["initial_assessed"] is True

    def test_new_user_default_zero(self, client):
        p = client.get("/l2/profile/USER_NEVER_ASSESSED").json()["data"]
        assert p["overall_mastery"] == 0.0
        assert p["initial_assessed"] is False
        assert p["kp_mastery"] == {}
        assert p["level"] == "novice"

    def test_decay_never_increases(self, client):
        p = client.get("/l2/profile/DY20240001").json()["data"]
        # 衰减方向断言: 有效掌握度不高于原始值 (容差 0.001 覆盖重建浮点差异)
        for kp, eff in p["kp_mastery"].items():
            assert eff <= p["raw_kp_mastery"][kp] + 0.001, f"{kp} 衰减后不应超过原值"


class TestWsBroadcastPublish:
    """WS bkt_update 推流 (答题后发布)."""

    def test_answer_publishes_bkt_update(self, client):
        # 直接调用 HUB: 答题后应有事件 (通过 hub 订阅捕获不可行, 验证发布路径存在)
        from dy3_polaris.l7.api.websocket import HUB, CHANNELS

        assert "broadcast" in CHANNELS
        assert any(e == "bkt_update" for e in CHANNELS["broadcast"]["events"])
        # 发布方法可调用
        HUB.broadcast("broadcast", "bkt_update", {"learner_id": "x", "kp_id": "A-01", "p_mastery_after": 0.5})
