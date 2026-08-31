"""L5 广播与编排链路专项测试: 频道注册 / 发布 / 订阅传播 / 编排执行.

覆盖本轮修复: ① 4 个 Agent 广播频道注册到 MessageBus;
② publish_message handler 构造 Message 对象 (原参数不匹配缺陷);
③ orchestrate handler 构造 OrchestrationPlan (原参数不匹配缺陷);
④ OrchestrationEngine 结果缓存 get_result。
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.communication import Message
from dy3_polaris.l5.default_agents import build_default_agents
from dy3_polaris.l5.unified_app import UnifiedApp

AGENT_CHANNELS = {
    "learning.diagnosis.report",
    "learning.interaction.event",
    "learning.knowledge.gap",
    "knowledge.generation.output",
    "knowledge.review.result",
    "guidance.decision.command",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    builder = UnifiedApp.create_full_app_builder()
    return TestClient(builder.create_app())


def _headers(client):
    login = client.post(
        "/l1/api/v1/auth/login",
        json={"student_id": "DY20248888", "password": "admin888"},
    )
    return {"Authorization": "Bearer " + login.json()["data"]["access_token"]}


class TestBroadcastChannels:
    """4 个 Agent 广播频道注册与发布."""

    def test_agent_defs_declare_channels(self):
        channels = set()
        for d in build_default_agents():
            for ch in d.broadcast_channels:
                channels.add(ch.channel)
        assert AGENT_CHANNELS.issubset(channels)

    def test_channels_registered_in_bus(self):
        builder = UnifiedApp.create_full_app_builder()
        bus = builder.bridge.message_bus
        registered = set(bus.channels.keys())
        assert AGENT_CHANNELS.issubset(registered), (
            f"未注册: {AGENT_CHANNELS - registered}"
        )

    def test_publish_all_agent_channels(self, client):
        for ch in sorted(AGENT_CHANNELS):
            r = client.post("/l5/message/publish", json={
                "channel": ch,
                "payload": {"probe": True},
                "publisher": "test-agent",
            }, headers=_headers(client))
            assert r.json()["code"] == 0, f"{ch}: {r.json()}"
            assert r.json()["data"]["message_id"]

    def test_publish_unknown_channel_lazily_creates(self, client):
        # 懒创建频道: 未知频道也可发布 (支持按需广播)
        r = client.post("/l5/message/publish", json={
            "channel": "no.such.channel", "payload": {"x": 1},
        }, headers=_headers(client))
        assert r.json()["code"] == 0
        assert r.json()["data"]["message_id"]

    def test_publish_missing_channel_param(self, client):
        r = client.post("/l5/message/publish", json={"payload": {}})
        assert r.json()["code"] != 0


class TestPubSubPropagation:
    """发布→订阅传播链路 (Agent 间信息流)."""

    def test_subscriber_receives_published_message(self):
        builder = UnifiedApp.create_full_app_builder()
        bus = builder.bridge.message_bus
        received: list[Message] = []

        def on_msg(m: Message):
            received.append(m)

        sub = bus.subscribe(
            "learning.diagnosis.report", "agent.knowledge.generation", on_msg
        )
        try:
            bus.publish(Message(
                channel="learning.diagnosis.report",
                publisher="agent.learning.diagnosis",
                payload={"weak_kps": ["A-01"]},
            ))
            assert len(received) == 1
            assert received[0].payload["weak_kps"] == ["A-01"]
            # 消息历史已记录
            hist = bus.get_history("learning.diagnosis.report") or []
            assert any(m.message_id == received[0].message_id for m in hist)
        finally:
            sub.unsubscribe()


class TestOrchestrate:
    """编排流程 (OrchestrationPlan + Agent worker)."""

    def _headers(self, client):
        login = client.post(
            "/l1/api/v1/auth/login",
            json={"student_id": "DY20248888", "password": "admin888"},
        )
        return {"Authorization": "Bearer " + login.json()["data"]["access_token"]}

    def test_orchestrate_pipeline_runs_three_agents(self, client):
        r = client.post("/l5/orchestrate", json={
            "paradigm": "pipeline",
            "tasks": [
                {"task_id": "t1", "agent_id": "agent.learning.diagnosis",
                 "input": {"learner_id": "DY20240001"}},
                {"task_id": "t2", "agent_id": "agent.knowledge.generation",
                 "input": {"query": "Dy3+ 的量子效率受哪些因素影响？"}},
                {"task_id": "t3", "agent_id": "agent.quality.review",
                 "input": {"content": "Dy3+ 的发射波长为 575nm"}},
            ],
        }, headers=self._headers(client))
        assert r.status_code == 200, r.text
        d = r.json()["data"]
        assert d["plan_id"]
        # get_result 缓存可取
        g = client.get(f"/l5/orchestration/{d['plan_id']}")
        assert g.json()["code"] == 0

    def test_orchestrate_bad_paradigm(self, client):
        r = client.post("/l5/orchestrate", json={
            "paradigm": "magic", "tasks": [],
        }, headers=self._headers(client))
        assert r.status_code == 400

    def test_orchestrate_missing_paradigm(self, client):
        r = client.post("/l5/orchestrate", json={"tasks": []},
                        headers=self._headers(client))
        assert r.json()["code"] != 0


class TestAgentsFrontend:
    """前端动态加载 Agent 功能."""

    def test_mf6_js_has_agent_views(self, client):
        js = client.get("/static/assets/mf6-features.js").text
        assert "renderAgentList" in js
        assert "renderAgentChain" in js
        assert "'agents-list': renderAgentList" in js
        assert "'agents-chain': renderAgentChain" in js
        assert "/l5/agents" in js
