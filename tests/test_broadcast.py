"""广播协议与记忆图谱 - 完整单元测试.

测试覆盖:
1. 主题匹配 (match_topic) — 精确/单层通配/多层通配/不匹配
2. 广播事件 (BroadcastEvent) — 创建/序列化
3. 订阅记录 (Subscription) — 创建/默认值
4. 投递模式 (DeliveryMode) — 枚举值
5. 广播总线 (BroadcastBus)
   - 订阅管理 (subscribe/unsubscribe/重复/上限)
   - 发布 (精确/通配/多订阅者/无订阅者)
   - 事件过滤 (filter_fn)
   - 投递失败 (回调异常 → BroadcastDeliveryError)
   - 订阅者激活/停用
   - 事件日志
   - 查询 (get_subscribers/list_topics/get_subscriptions)
   - 度量统计
   - 重置
6. 广播度量 (BroadcastMetrics) — 记录/导出/重置
7. 节点类型 (NodeType) / 边类型 (EdgeType)
8. 记忆节点 (MemoryNode) — 创建/touch/序列化/strength范围
9. 记忆边 (MemoryEdge) — 创建/序列化/weight范围
10. 记忆图谱 (MemoryGraph)
    - 节点管理 (add/remove/get/has/touch)
    - 边管理 (add/remove/get/has/自环/环检测)
    - 邻居查询 (out/in/both/类型过滤)
    - 路径查找 (直连/多跳/无路径/同节点)
    - 多条件搜索 (类型/强度/元数据/内容/limit)
    - 子图提取
    - 衰减 (因子/清除低强度节点)
    - 强化
    - 扩散激活 (深度/衰减/多跳)
    - 环检测 (has_cycle)
    - 导出/导入
    - 度量统计
    - 重置/计数
11. 记忆图谱度量 (MemoryGraphMetrics) — 记录/导出/重置
"""

from __future__ import annotations

import threading
import time

import pytest

from dy3_polaris.l6.broadcast.broadcast import (
    BroadcastBus,
    BroadcastEvent,
    BroadcastMetrics,
    DeliveryMode,
    Subscription,
    match_topic,
)
from dy3_polaris.l6.broadcast.memory_graph import (
    EdgeType,
    MemoryEdge,
    MemoryGraph,
    MemoryGraphMetrics,
    MemoryNode,
    NodeType,
)
from dy3_polaris.l6.core.exceptions import (
    BroadcastDeliveryError,
    BroadcastError,
    EdgeNotFoundError,
    GraphCycleError,
    MemoryGraphError,
    NodeNotFoundError,
    SubscriberNotFoundError,
)


# ============================================================
# 辅助工具
# ============================================================

def _make_bus(**kwargs) -> BroadcastBus:
    """快速创建广播总线."""
    return BroadcastBus(**kwargs)


def _make_graph(**kwargs) -> MemoryGraph:
    """快速创建记忆图谱."""
    return MemoryGraph(**kwargs)


# ============================================================
# 1. 主题匹配测试
# ============================================================

class TestMatchTopic:
    """match_topic 层级通配匹配测试."""

    def test精确匹配(self) -> None:
        assert match_topic("learner.profile", "learner.profile") is True

    def test精确不匹配(self) -> None:
        assert match_topic("learner.profile", "learner.assessment") is False

    def test单层通配匹配(self) -> None:
        assert match_topic("learner.*", "learner.profile") is True
        assert match_topic("learner.*", "learner.assessment") is True

    def test单层通配不匹配多层(self) -> None:
        """learner.* 不匹配 learner.profile.updated."""
        assert match_topic("learner.*", "learner.profile.updated") is False

    def test多层通配匹配(self) -> None:
        assert match_topic("learner.**", "learner.profile.updated") is True
        assert match_topic("learner.**", "learner.assessment.score.high") is True

    def test多层通配匹配单层(self) -> None:
        """learner.** 也匹配 learner.profile."""
        assert match_topic("learner.**", "learner.profile") is True

    def test通配符在中间(self) -> None:
        assert match_topic("agent.*.event", "agent.click.event") is True
        assert match_topic("agent.*.event", "agent.click.click.event") is False

    def test无通配符不匹配(self) -> None:
        assert match_topic("agent.click", "agent.click.event") is False

    def test空字符串(self) -> None:
        assert match_topic("", "") is True
        # "" split(".") == [""], 即一个空段, * 匹配单层 → True
        assert match_topic("*", "") is True

    def test全通配(self) -> None:
        assert match_topic("**", "any.topic.here") is True
        assert match_topic("**", "single") is True

    def test部分前缀匹配(self) -> None:
        assert match_topic("learner.profile.*", "learner.profile.updated") is True
        assert match_topic("learner.profile.*", "learner.profile") is False


# ============================================================
# 2. BroadcastEvent 测试
# ============================================================

class TestBroadcastEvent:
    """广播事件测试."""

    def test基本创建(self) -> None:
        e = BroadcastEvent(topic="learner.profile", payload={"k": "v"}, source="agent-1")
        assert e.topic == "learner.profile"
        assert e.payload == {"k": "v"}
        assert e.source == "agent-1"
        assert len(e.event_id) == 16
        assert e.timestamp > 0

    def test默认值(self) -> None:
        e = BroadcastEvent(topic="test")
        assert e.payload == {}
        assert e.source == ""
        assert e.metadata == {}

    def test指定_event_id(self) -> None:
        e = BroadcastEvent(topic="t", event_id="custom-id-12345")
        assert e.event_id == "custom-id-12345"

    def test指定_timestamp(self) -> None:
        ts = 1000000.0
        e = BroadcastEvent(topic="t", timestamp=ts)
        assert e.timestamp == ts

    def test_to_dict(self) -> None:
        e = BroadcastEvent(
            topic="learner.assessment",
            payload={"score": 95},
            source="grader-1",
            metadata={"priority": "high"},
        )
        d = e.to_dict()
        assert d["topic"] == "learner.assessment"
        assert d["payload"] == {"score": 95}
        assert d["source"] == "grader-1"
        assert d["metadata"] == {"priority": "high"}
        assert "event_id" in d
        assert "timestamp" in d


# ============================================================
# 3. Subscription 测试
# ============================================================

class TestSubscription:
    """订阅记录测试."""

    def test基本创建(self) -> None:
        cb = lambda e: None
        sub = Subscription(
            subscriber_id="agent-1",
            topic_pattern="learner.*",
            callback=cb,
        )
        assert sub.subscriber_id == "agent-1"
        assert sub.topic_pattern == "learner.*"
        assert sub.callback is cb
        assert sub.filter_fn is None
        assert sub.delivery_mode == DeliveryMode.SYNC
        assert sub.active is True
        assert sub.delivered_count == 0
        assert sub.failed_count == 0
        assert sub.created_at > 0

    def test带过滤器和异步模式(self) -> None:
        filt = lambda e: True
        sub = Subscription(
            subscriber_id="agent-2",
            topic_pattern="**",
            callback=lambda e: None,
            filter_fn=filt,
            delivery_mode=DeliveryMode.ASYNC,
        )
        assert sub.filter_fn is filt
        assert sub.delivery_mode == DeliveryMode.ASYNC


# ============================================================
# 4. DeliveryMode 枚举测试
# ============================================================

class TestDeliveryMode:
    """投递模式枚举测试."""

    def test两种模式(self) -> None:
        assert DeliveryMode.SYNC == "sync"
        assert DeliveryMode.ASYNC == "async"

    def test枚举数量(self) -> None:
        assert len(DeliveryMode) == 2

    def test是字符串枚举(self) -> None:
        assert isinstance(DeliveryMode.SYNC, str)


# ============================================================
# 5. BroadcastBus 测试
# ============================================================

class TestBusSubscribe:
    """广播总线 - 订阅管理."""

    def test_subscribe_返回订阅对象(self) -> None:
        bus = _make_bus()
        sub = bus.subscribe("s1", "topic.*", lambda e: None)
        assert sub.subscriber_id == "s1"
        assert sub.topic_pattern == "topic.*"

    def test_subscribe_后可查询主题(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "learner.*", lambda e: None)
        assert "learner.*" in bus.list_topics()

    def test_subscribe_重复抛异常(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "topic", lambda e: None)
        with pytest.raises(BroadcastError, match="DUPLICATE"):
            bus.subscribe("s1", "topic", lambda e: None)

    def test_subscribe_不同订阅者同主题(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "topic", lambda e: None)
        bus.subscribe("s2", "topic", lambda e: None)
        assert len(bus.get_subscribers("topic")) == 2

    def test_subscribe_超过上限抛异常(self) -> None:
        bus = _make_bus(max_subscribers_per_topic=2)
        bus.subscribe("s1", "topic", lambda e: None)
        bus.subscribe("s2", "topic", lambda e: None)
        with pytest.raises(BroadcastError, match="FULL"):
            bus.subscribe("s3", "topic", lambda e: None)

    def test_unsubscribe_指定主题(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "topic", lambda e: None)
        removed = bus.unsubscribe("s1", "topic")
        assert removed == 1
        assert "topic" not in bus.list_topics()

    def test_unsubscribe_所有主题(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t1", lambda e: None)
        bus.subscribe("s1", "t2", lambda e: None)
        removed = bus.unsubscribe("s1")
        assert removed == 2
        assert len(bus.list_topics()) == 0

    def test_unsubscribe_不存在抛异常(self) -> None:
        bus = _make_bus()
        with pytest.raises(SubscriberNotFoundError):
            bus.unsubscribe("ghost")

    def test_unsubscribe_指定主题不存在(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t1", lambda e: None)
        with pytest.raises(SubscriberNotFoundError):
            bus.unsubscribe("s1", "other-topic")


class TestBusPublish:
    """广播总线 - 发布."""

    def test_publish_精确匹配(self) -> None:
        received: list[BroadcastEvent] = []
        bus = _make_bus()
        bus.subscribe("s1", "learner.profile", lambda e: received.append(e))
        bus.publish("learner.profile", {"name": "张三"})
        assert len(received) == 1
        assert received[0].payload == {"name": "张三"}

    def test_publish_单层通配匹配(self) -> None:
        received: list[str] = []
        bus = _make_bus()
        bus.subscribe("s1", "learner.*", lambda e: received.append(e.topic))
        bus.publish("learner.profile", {})
        bus.publish("learner.assessment", {})
        bus.publish("learner.profile.updated", {})  # 不匹配
        assert received == ["learner.profile", "learner.assessment"]

    def test_publish_多层通配匹配(self) -> None:
        received: list[str] = []
        bus = _make_bus()
        bus.subscribe("s1", "learner.**", lambda e: received.append(e.topic))
        bus.publish("learner.profile.updated", {})
        bus.publish("learner.assessment.score.high", {})
        assert len(received) == 2

    def test_publish_多订阅者(self) -> None:
        count = [0, 0]
        bus = _make_bus()
        bus.subscribe("s1", "topic", lambda e: count.__setitem__(0, count[0] + 1))
        bus.subscribe("s2", "topic", lambda e: count.__setitem__(1, count[1] + 1))
        bus.publish("topic")
        assert count == [1, 1]

    def test_publish_无订阅者(self) -> None:
        bus = _make_bus()
        event = bus.publish("no-subscriber", {"data": 1})
        assert event.topic == "no-subscriber"

    def test_publish_返回事件对象(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t", lambda e: None)
        event = bus.publish("t", {"key": "val"}, source="src")
        assert event.topic == "t"
        assert event.payload == {"key": "val"}
        assert event.source == "src"

    def test_publish_带过滤器通过(self) -> None:
        received: list[BroadcastEvent] = []
        bus = _make_bus()
        bus.subscribe(
            "s1", "topic",
            lambda e: received.append(e),
            filter_fn=lambda e: e.payload.get("priority") == "high",
        )
        bus.publish("topic", {"priority": "high"})
        bus.publish("topic", {"priority": "low"})
        assert len(received) == 1

    def test_publish_投递失败抛异常(self) -> None:
        bus = _make_bus()
        def bad_callback(e: BroadcastEvent) -> None:
            raise RuntimeError("boom")
        bus.subscribe("s1", "topic", bad_callback)
        with pytest.raises(BroadcastDeliveryError):
            bus.publish("topic")

    def test_publish_部分失败仍投递其他(self) -> None:
        received: list[str] = []
        bus = _make_bus()
        def good_cb(e: BroadcastEvent) -> None:
            received.append(e.topic)
        def bad_cb(e: BroadcastEvent) -> None:
            raise RuntimeError("boom")
        bus.subscribe("s1", "topic", good_cb)
        bus.subscribe("s2", "topic", bad_cb)
        with pytest.raises(BroadcastDeliveryError):
            bus.publish("topic")
        # good_cb 仍然被调用
        assert received == ["topic"]

    def test_publish_过滤器异常计为失败(self) -> None:
        bus = _make_bus()
        bus.subscribe(
            "s1", "topic",
            lambda e: None,
            filter_fn=lambda e: 1 / 0,  # 抛异常
        )
        with pytest.raises(BroadcastDeliveryError):
            bus.publish("topic")


class TestBusActivation:
    """广播总线 - 订阅者激活/停用."""

    def test_deactivate_后不接收事件(self) -> None:
        received: list[str] = []
        bus = _make_bus()
        bus.subscribe("s1", "topic", lambda e: received.append(e.topic))
        bus.deactivate_subscriber("s1")
        bus.publish("topic")
        assert received == []

    def test_activate_后恢复接收(self) -> None:
        received: list[str] = []
        bus = _make_bus()
        bus.subscribe("s1", "topic", lambda e: received.append(e.topic))
        bus.deactivate_subscriber("s1")
        bus.publish("topic")  # 不接收
        bus.activate_subscriber("s1")
        bus.publish("topic")  # 接收
        assert received == ["topic"]

    def test_deactivate_返回数量(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t1", lambda e: None)
        bus.subscribe("s1", "t2", lambda e: None)
        assert bus.deactivate_subscriber("s1") == 2

    def test_activate_不存在返回零(self) -> None:
        bus = _make_bus()
        assert bus.activate_subscriber("ghost") == 0


class TestBusQueries:
    """广播总线 - 查询."""

    def test_get_subscribers_精确(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "topic", lambda e: None)
        bus.subscribe("s2", "topic", lambda e: None)
        subs = bus.get_subscribers("topic")
        assert set(subs) == {"s1", "s2"}

    def test_get_subscribers_通配匹配(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "learner.*", lambda e: None)
        bus.subscribe("s2", "learner.**", lambda e: None)
        subs = bus.get_subscribers("learner.profile")
        assert set(subs) == {"s1", "s2"}

    def test_get_subscribers_去重(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "learner.*", lambda e: None)
        bus.subscribe("s1", "learner.**", lambda e: None)
        subs = bus.get_subscribers("learner.profile")
        assert subs == ["s1"]

    def test_get_subscribers_排除停用(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "topic", lambda e: None)
        bus.subscribe("s2", "topic", lambda e: None)
        bus.deactivate_subscriber("s2")
        subs = bus.get_subscribers("topic")
        assert subs == ["s1"]

    def test_list_topics(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t1", lambda e: None)
        bus.subscribe("s2", "t2", lambda e: None)
        topics = bus.list_topics()
        assert set(topics) == {"t1", "t2"}

    def test_get_subscriptions(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t1", lambda e: None)
        bus.subscribe("s1", "t2", lambda e: None)
        bus.subscribe("s2", "t1", lambda e: None)
        subs = bus.get_subscriptions("s1")
        assert len(subs) == 2

    def test_get_subscriptions_不存在返回空(self) -> None:
        bus = _make_bus()
        assert bus.get_subscriptions("ghost") == []


class TestBusEventLog:
    """广播总线 - 事件日志."""

    def test_启用事件日志(self) -> None:
        bus = _make_bus(event_log_enabled=True, event_log_max_size=10)
        bus.subscribe("s1", "t", lambda e: None)
        bus.publish("t", {"v": 1})
        bus.publish("t", {"v": 2})
        log = bus.get_event_log()
        assert len(log) == 2
        assert log[0].payload == {"v": 1}
        assert log[1].payload == {"v": 2}

    def test_未启用事件日志(self) -> None:
        bus = _make_bus(event_log_enabled=False)
        bus.subscribe("s1", "t", lambda e: None)
        bus.publish("t")
        assert bus.get_event_log() == []

    def test_事件日志上限(self) -> None:
        bus = _make_bus(event_log_enabled=True, event_log_max_size=3)
        bus.subscribe("s1", "t", lambda e: None)
        for i in range(5):
            bus.publish("t", {"i": i})
        log = bus.get_event_log()
        assert len(log) == 3
        # 保留最后 3 条
        assert log[0].payload == {"i": 2}
        assert log[2].payload == {"i": 4}

    def test_clear_event_log(self) -> None:
        bus = _make_bus(event_log_enabled=True)
        bus.subscribe("s1", "t", lambda e: None)
        bus.publish("t")
        bus.clear_event_log()
        assert bus.get_event_log() == []


class TestBusMetrics:
    """广播总线 - 度量统计."""

    def test_publish_count(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t", lambda e: None)
        bus.publish("t")
        bus.publish("t")
        m = bus.get_metrics()
        assert m["publish_count"] == 2

    def test_delivery_count(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t", lambda e: None)
        bus.subscribe("s2", "t", lambda e: None)
        bus.publish("t")
        m = bus.get_metrics()
        assert m["delivery_count"] == 2

    def test_failure_count(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t", lambda e: (_ for _ in ()).throw(RuntimeError()))
        try:
            bus.publish("t")
        except BroadcastDeliveryError:
            pass
        m = bus.get_metrics()
        assert m["failure_count"] == 1

    def test_filter_rejected(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t", lambda e: None, filter_fn=lambda e: False)
        bus.publish("t")
        m = bus.get_metrics()
        assert m["filter_rejected"] == 1
        assert m["delivery_count"] == 0

    def test_active_subscriptions(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t", lambda e: None)
        bus.subscribe("s2", "t", lambda e: None)
        bus.deactivate_subscriber("s2")
        m = bus.get_metrics()
        assert m["active_subscriptions"] == 1
        assert m["total_subscriptions"] == 2

    def test_topics_in_metrics(self) -> None:
        bus = _make_bus()
        bus.subscribe("s1", "t1", lambda e: None)
        m = bus.get_metrics()
        assert "t1" in m["topics"]


class TestBusReset:
    """广播总线 - 重置."""

    def test_reset_清除所有(self) -> None:
        bus = _make_bus(event_log_enabled=True)
        bus.subscribe("s1", "t", lambda e: None)
        bus.publish("t")
        bus.reset()
        assert bus.list_topics() == []
        assert bus.get_event_log() == []
        m = bus.get_metrics()
        assert m["publish_count"] == 0


# ============================================================
# 6. BroadcastMetrics 测试
# ============================================================

class TestBroadcastMetrics:
    """广播度量收集器测试."""

    def test_record_publish(self) -> None:
        m = BroadcastMetrics()
        m.record_publish()
        m.record_publish()
        assert m.export()["publish_count"] == 2

    def test_record_delivery(self) -> None:
        m = BroadcastMetrics()
        m.record_delivery(True)
        m.record_delivery(True)
        m.record_delivery(False)
        e = m.export()
        assert e["delivery_count"] == 2
        assert e["failure_count"] == 1

    def test_record_filter_reject(self) -> None:
        m = BroadcastMetrics()
        m.record_filter_reject()
        assert m.export()["filter_rejected"] == 1

    def test_reset(self) -> None:
        m = BroadcastMetrics()
        m.record_publish()
        m.record_delivery(True)
        m.reset()
        e = m.export()
        assert all(v == 0 for v in e.values())

    def test线程安全(self) -> None:
        m = BroadcastMetrics()
        def worker() -> None:
            for _ in range(100):
                m.record_publish()
                m.record_delivery(True)
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        e = m.export()
        assert e["publish_count"] == 1000
        assert e["delivery_count"] == 1000


# ============================================================
# 7. NodeType / EdgeType 枚举测试
# ============================================================

class TestNodeType:
    """节点类型枚举测试."""

    def test六种类型(self) -> None:
        expected = {"learner", "knowledge", "skill", "assessment", "resource", "session"}
        actual = {t.value for t in NodeType}
        assert actual == expected

    def test是字符串枚举(self) -> None:
        assert isinstance(NodeType.KNOWLEDGE, str)
        assert NodeType.LEARNER == "learner"


class TestEdgeType:
    """边类型枚举测试."""

    def test五种类型(self) -> None:
        expected = {"prerequisite", "related", "derived", "learned", "references"}
        actual = {t.value for t in EdgeType}
        assert actual == expected

    def test是字符串枚举(self) -> None:
        assert isinstance(EdgeType.PREREQUISITE, str)


# ============================================================
# 8. MemoryNode 测试
# ============================================================

class TestMemoryNode:
    """记忆节点测试."""

    def test基本创建(self) -> None:
        n = MemoryNode(node_id="n1", node_type=NodeType.KNOWLEDGE)
        assert n.node_id == "n1"
        assert n.node_type == NodeType.KNOWLEDGE
        assert n.content == {}
        assert n.metadata == {}
        assert n.strength == 1.0
        assert n.access_count == 0

    def test自动生成_id(self) -> None:
        n = MemoryNode()
        assert len(n.node_id) == 12

    def test_strength_范围限制(self) -> None:
        n = MemoryNode(strength=1.5)
        assert n.strength == 1.0
        n2 = MemoryNode(strength=-0.5)
        assert n2.strength == 0.0

    def test_touch(self) -> None:
        n = MemoryNode()
        before = n.last_accessed_at
        time.sleep(0.001)
        n.touch()
        assert n.last_accessed_at > before
        assert n.access_count == 1

    def test_to_dict(self) -> None:
        n = MemoryNode(
            node_id="n2",
            node_type=NodeType.LEARNER,
            content={"name": "张三"},
            metadata={"grade": "大三"},
            strength=0.8,
        )
        d = n.to_dict()
        assert d["node_id"] == "n2"
        assert d["node_type"] == "learner"
        assert d["content"] == {"name": "张三"}
        assert d["metadata"] == {"grade": "大三"}
        assert d["strength"] == 0.8
        assert "created_at" in d
        assert "last_accessed_at" in d
        assert "access_count" in d


# ============================================================
# 9. MemoryEdge 测试
# ============================================================

class TestMemoryEdge:
    """记忆边测试."""

    def test基本创建(self) -> None:
        e = MemoryEdge(source_id="s1", target_id="t1")
        assert e.source_id == "s1"
        assert e.target_id == "t1"
        assert e.edge_type == EdgeType.RELATED
        assert e.weight == 1.0
        assert e.metadata == {}

    def test_weight_范围限制(self) -> None:
        e = MemoryEdge("s", "t", weight=2.0)
        assert e.weight == 1.0
        e2 = MemoryEdge("s", "t", weight=-1.0)
        assert e2.weight == 0.0

    def test_to_dict(self) -> None:
        e = MemoryEdge(
            source_id="s1",
            target_id="t1",
            edge_type=EdgeType.PREREQUISITE,
            weight=0.7,
            metadata={"order": 1},
        )
        d = e.to_dict()
        assert d["source_id"] == "s1"
        assert d["target_id"] == "t1"
        assert d["edge_type"] == "prerequisite"
        assert d["weight"] == 0.7
        assert d["metadata"] == {"order": 1}


# ============================================================
# 10. MemoryGraph 测试
# ============================================================

class TestGraphNodeManagement:
    """记忆图谱 - 节点管理."""

    def test_add_node_返回节点(self) -> None:
        g = _make_graph()
        n = g.add_node("n1", NodeType.KNOWLEDGE, {"title": "化学键"})
        assert n.node_id == "n1"
        assert n.node_type == NodeType.KNOWLEDGE
        assert n.content == {"title": "化学键"}

    def test_add_node_自动_id(self) -> None:
        g = _make_graph()
        n = g.add_node()
        assert len(n.node_id) == 12

    def test_add_node_已存在则更新(self) -> None:
        g = _make_graph()
        g.add_node("n1", NodeType.KNOWLEDGE, {"v": 1})
        n = g.add_node("n1", NodeType.SKILL, {"v": 2})
        assert n.node_type == NodeType.SKILL
        assert n.content == {"v": 2}

    def test_remove_node(self) -> None:
        g = _make_graph()
        g.add_node("n1")
        assert g.remove_node("n1") is True
        assert not g.has_node("n1")

    def test_remove_node_不存在抛异常(self) -> None:
        g = _make_graph()
        with pytest.raises(NodeNotFoundError):
            g.remove_node("ghost")

    def test_remove_node_级联删除边(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.remove_node("a")
        assert not g.has_edge("a", "b")
        # b 仍存在
        assert g.has_node("b")

    def test_get_node(self) -> None:
        g = _make_graph()
        g.add_node("n1", content={"k": "v"})
        n = g.get_node("n1")
        assert n.content == {"k": "v"}

    def test_get_node_不存在抛异常(self) -> None:
        g = _make_graph()
        with pytest.raises(NodeNotFoundError):
            g.get_node("ghost")

    def test_has_node(self) -> None:
        g = _make_graph()
        g.add_node("n1")
        assert g.has_node("n1") is True
        assert g.has_node("n2") is False

    def test_touch_node_强化(self) -> None:
        g = _make_graph()
        g.add_node("n1", strength=0.5)
        n = g.touch_node("n1")
        assert n.access_count == 1
        assert n.strength == 0.55  # +0.05

    def test_touch_node_强度上限(self) -> None:
        g = _make_graph()
        g.add_node("n1", strength=0.99)
        n = g.touch_node("n1")
        assert n.strength == 1.0  # clamp

    def test_touch_node_不存在抛异常(self) -> None:
        g = _make_graph()
        with pytest.raises(NodeNotFoundError):
            g.touch_node("ghost")


class TestGraphEdgeManagement:
    """记忆图谱 - 边管理."""

    def test_add_edge(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        e = g.add_edge("a", "b", EdgeType.PREREQUISITE, 0.8)
        assert e.source_id == "a"
        assert e.target_id == "b"
        assert e.edge_type == EdgeType.PREREQUISITE
        assert e.weight == 0.8

    def test_add_edge_源不存在抛异常(self) -> None:
        g = _make_graph()
        g.add_node("b")
        with pytest.raises(NodeNotFoundError):
            g.add_edge("ghost", "b")

    def test_add_edge_目标不存在抛异常(self) -> None:
        g = _make_graph()
        g.add_node("a")
        with pytest.raises(NodeNotFoundError):
            g.add_edge("a", "ghost")

    def test_add_edge_自环抛异常(self) -> None:
        g = _make_graph()
        g.add_node("a")
        with pytest.raises(MemoryGraphError, match="Self-loop"):
            g.add_edge("a", "a")

    def test_add_edge_环检测_阻止(self) -> None:
        """A→B, B→C, 添加 C→A 应抛 GraphCycleError."""
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b", EdgeType.PREREQUISITE)
        g.add_edge("b", "c", EdgeType.PREREQUISITE)
        with pytest.raises(GraphCycleError):
            g.add_edge("c", "a", EdgeType.PREREQUISITE)

    def test_add_edge_环检测_非前置类型不检测(self) -> None:
        """RELATED 类型不检测环."""
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", EdgeType.RELATED)
        g.add_edge("b", "a", EdgeType.RELATED)  # 不抛异常
        assert g.has_edge("b", "a")

    def test_add_edge_跳过环检测(self) -> None:
        """check_cycle=False 时不检测."""
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", EdgeType.PREREQUISITE)
        # check_cycle=False 允许环
        g.add_edge("b", "a", EdgeType.PREREQUISITE, check_cycle=False)
        assert g.has_edge("b", "a")

    def test_remove_edge(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        assert g.remove_edge("a", "b") is True
        assert not g.has_edge("a", "b")

    def test_remove_edge_不存在抛异常(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        with pytest.raises(EdgeNotFoundError):
            g.remove_edge("a", "b")

    def test_get_edge(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", EdgeType.LEARNED, 0.6)
        e = g.get_edge("a", "b")
        assert e.edge_type == EdgeType.LEARNED
        assert e.weight == 0.6

    def test_get_edge_不存在抛异常(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        with pytest.raises(EdgeNotFoundError):
            g.get_edge("a", "b")

    def test_has_edge(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        assert g.has_edge("a", "b") is False
        g.add_edge("a", "b")
        assert g.has_edge("a", "b") is True


class TestGraphNeighbors:
    """记忆图谱 - 邻居查询."""

    def test_neighbors_出边(self) -> None:
        g = _make_graph()
        for nid in ["a", "b", "c"]:
            g.add_node(nid)
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert set(g.neighbors("a", direction="out")) == {"b", "c"}

    def test_neighbors_入边(self) -> None:
        g = _make_graph()
        for nid in ["a", "b", "c"]:
            g.add_node(nid)
        g.add_edge("b", "a")
        g.add_edge("c", "a")
        assert set(g.neighbors("a", direction="in")) == {"b", "c"}

    def test_neighbors_双向(self) -> None:
        g = _make_graph()
        for nid in ["a", "b", "c"]:
            g.add_node(nid)
        g.add_edge("a", "b")  # a→b (出)
        g.add_edge("c", "a")  # c→a (入)
        assert set(g.neighbors("a", direction="both")) == {"b", "c"}

    def test_neighbors_边类型过滤(self) -> None:
        g = _make_graph()
        for nid in ["a", "b", "c"]:
            g.add_node(nid)
        g.add_edge("a", "b", EdgeType.PREREQUISITE)
        g.add_edge("a", "c", EdgeType.RELATED)
        assert g.neighbors("a", edge_type=EdgeType.PREREQUISITE) == ["b"]
        assert g.neighbors("a", edge_type=EdgeType.RELATED) == ["c"]

    def test_neighbors_无邻居(self) -> None:
        g = _make_graph()
        g.add_node("a")
        assert g.neighbors("a") == []

    def test_neighbors_不存在抛异常(self) -> None:
        g = _make_graph()
        with pytest.raises(NodeNotFoundError):
            g.neighbors("ghost")


class TestGraphFindPath:
    """记忆图谱 - 路径查找."""

    def test_直连路径(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        path = g.find_path("a", "b")
        assert path == ["a", "b"]

    def test_多跳路径(self) -> None:
        g = _make_graph()
        for nid in ["a", "b", "c", "d"]:
            g.add_node(nid)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "d")
        path = g.find_path("a", "d")
        assert path == ["a", "b", "c", "d"]

    def test_无路径返回_none(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        path = g.find_path("a", "b")
        assert path is None

    def test_同节点返回自身(self) -> None:
        g = _make_graph()
        g.add_node("a")
        path = g.find_path("a", "a")
        assert path == ["a"]

    def test_源不存在抛异常(self) -> None:
        g = _make_graph()
        g.add_node("b")
        with pytest.raises(NodeNotFoundError):
            g.find_path("ghost", "b")

    def test_bfs_最短路径(self) -> None:
        """验证 BFS 找到最短路径而非任意路径."""
        g = _make_graph()
        for nid in ["a", "b", "c", "d"]:
            g.add_node(nid)
        g.add_edge("a", "b")
        g.add_edge("a", "d")  # 捷径
        g.add_edge("b", "c")
        g.add_edge("c", "d")
        path = g.find_path("a", "d")
        assert path == ["a", "d"]  # 最短


class TestGraphSearch:
    """记忆图谱 - 多条件搜索."""

    def test_按类型搜索(self) -> None:
        g = _make_graph()
        g.add_node("k1", NodeType.KNOWLEDGE)
        g.add_node("k2", NodeType.KNOWLEDGE)
        g.add_node("s1", NodeType.SKILL)
        results = g.search(node_type=NodeType.KNOWLEDGE)
        assert len(results) == 2
        assert all(n.node_type == NodeType.KNOWLEDGE for n in results)

    def test_按强度范围搜索(self) -> None:
        g = _make_graph()
        g.add_node("n1", strength=0.9)
        g.add_node("n2", strength=0.3)
        g.add_node("n3", strength=0.6)
        results = g.search(min_strength=0.5)
        assert len(results) == 2

    def test_按元数据搜索(self) -> None:
        g = _make_graph()
        g.add_node("n1", metadata={"category": "chem"})
        g.add_node("n2", metadata={"category": "math"})
        results = g.search(metadata_key="category", metadata_value="chem")
        assert len(results) == 1
        assert results[0].node_id == "n1"

    def test_按内容搜索(self) -> None:
        g = _make_graph()
        g.add_node("n1", content={"title": "化学键"})
        g.add_node("n2", content={"title": "分子轨道"})
        results = g.search(content_key="title", content_value="化学键")
        assert len(results) == 1

    def test_组合条件(self) -> None:
        g = _make_graph()
        g.add_node("n1", NodeType.KNOWLEDGE, metadata={"tag": "a"}, strength=0.9)
        g.add_node("n2", NodeType.KNOWLEDGE, metadata={"tag": "a"}, strength=0.3)
        g.add_node("n3", NodeType.SKILL, metadata={"tag": "a"}, strength=0.9)
        results = g.search(
            node_type=NodeType.KNOWLEDGE,
            min_strength=0.5,
            metadata_key="tag",
            metadata_value="a",
        )
        assert len(results) == 1
        assert results[0].node_id == "n1"

    def test_limit_限制(self) -> None:
        g = _make_graph()
        for i in range(10):
            g.add_node(f"n{i}", NodeType.KNOWLEDGE)
        results = g.search(node_type=NodeType.KNOWLEDGE, limit=3)
        assert len(results) == 3

    def test_无匹配返回空(self) -> None:
        g = _make_graph()
        g.add_node("n1", NodeType.KNOWLEDGE)
        results = g.search(node_type=NodeType.SKILL)
        assert results == []


class TestGraphSubgraph:
    """记忆图谱 - 子图提取."""

    def test_子图提取(self) -> None:
        g = _make_graph()
        for nid in ["a", "b", "c", "d"]:
            g.add_node(nid)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "d")  # d 不在子图中
        sg = g.subgraph(["a", "b", "c"])
        assert len(sg["nodes"]) == 3
        assert len(sg["edges"]) == 2  # a→b, b→c

    def test_子图_不含外部边(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("a", "c")  # c 不在子图中
        sg = g.subgraph(["a", "b"])
        assert len(sg["edges"]) == 1  # 只有 a→b

    def test_子图_不存在节点跳过(self) -> None:
        g = _make_graph()
        g.add_node("a")
        sg = g.subgraph(["a", "ghost"])
        assert len(sg["nodes"]) == 1


class TestGraphDecay:
    """记忆图谱 - 衰减."""

    def test_decay_降低强度(self) -> None:
        g = _make_graph(decay_factor=0.9)
        g.add_node("n1", strength=1.0)
        g.add_node("n2", strength=0.5)
        pruned = g.decay()
        assert pruned == 0
        assert g.get_node("n1").strength == pytest.approx(0.9)
        assert g.get_node("n2").strength == pytest.approx(0.45)

    def test_decay_清除低强度节点(self) -> None:
        g = _make_graph(decay_factor=0.1, min_strength=0.1)
        g.add_node("n1", strength=0.5)  # 0.5 * 0.1 = 0.05 < 0.1 → 清除
        g.add_node("n2", strength=1.0)  # 1.0 * 0.1 = 0.1 = 0.1 → 保留
        pruned = g.decay()
        assert pruned == 1
        assert g.has_node("n2")
        assert not g.has_node("n1")

    def test_decay_自定义因子(self) -> None:
        g = _make_graph(decay_factor=0.9)
        g.add_node("n1", strength=1.0)
        g.decay(factor=0.5)
        assert g.get_node("n1").strength == pytest.approx(0.5)

    def test_decay_级联清除边(self) -> None:
        g = _make_graph(decay_factor=0.01, min_strength=0.1)
        g.add_node("a", strength=0.5)
        g.add_node("b", strength=1.0)
        g.add_edge("a", "b")
        g.decay()
        # a 被清除, 边也应消失
        assert not g.has_edge("a", "b")
        assert g.edge_count() == 0


class TestGraphReinforce:
    """记忆图谱 - 强化."""

    def test_reinforce_增加强度(self) -> None:
        g = _make_graph()
        g.add_node("n1", strength=0.5)
        n = g.reinforce("n1", 0.3)
        assert n.strength == pytest.approx(0.8)

    def test_reinforce_上限(self) -> None:
        g = _make_graph()
        g.add_node("n1", strength=0.9)
        n = g.reinforce("n1", 0.5)
        assert n.strength == 1.0

    def test_reinforce_更新访问时间(self) -> None:
        g = _make_graph()
        g.add_node("n1")
        before = g.get_node("n1").last_accessed_at
        time.sleep(0.001)
        n = g.reinforce("n1")
        assert n.last_accessed_at > before
        assert n.access_count == 1

    def test_reinforce_不存在抛异常(self) -> None:
        g = _make_graph()
        with pytest.raises(NodeNotFoundError):
            g.reinforce("ghost")


class TestGraphSpreadingActivation:
    """记忆图谱 - 扩散激活."""

    def test_基本扩散(self) -> None:
        g = _make_graph(spreading_depth=2, spreading_decay=0.5)
        g.add_node("a", strength=0.5)
        g.add_node("b", strength=0.3)
        g.add_node("c", strength=0.2)
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("b", "c", weight=1.0)
        activations = g.spreading_activation("a")
        assert "a" in activations
        assert activations["a"] == 1.0
        # b: 1.0 * 1.0 * 0.5 = 0.5
        assert "b" in activations
        assert activations["b"] == pytest.approx(0.5)
        # c: 0.5 * 1.0 * 0.5 = 0.25
        assert "c" in activations
        assert activations["c"] == pytest.approx(0.25)

    def test_起始节点强化(self) -> None:
        g = _make_graph()
        g.add_node("a", strength=0.5)
        g.spreading_activation("a")
        assert g.get_node("a").strength == pytest.approx(0.6)  # +0.1

    def test_邻居部分强化(self) -> None:
        g = _make_graph(spreading_decay=0.5)
        g.add_node("a", strength=0.5)
        g.add_node("b", strength=0.3)
        g.add_edge("a", "b", weight=1.0)
        g.spreading_activation("a")
        # b: activation=0.5, strength += 0.5 * 0.1 = 0.05
        assert g.get_node("b").strength == pytest.approx(0.35)

    def test_深度限制(self) -> None:
        g = _make_graph(spreading_depth=1, spreading_decay=0.5)
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("b", "c", weight=1.0)
        activations = g.spreading_activation("a", depth=1)
        assert "b" in activations
        assert "c" not in activations  # 超出深度

    def test_低激活不传播(self) -> None:
        """spread < 0.01 不传播."""
        g = _make_graph(spreading_depth=2, spreading_decay=0.001)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", weight=1.0)
        activations = g.spreading_activation("a")
        assert "a" in activations
        assert "b" not in activations  # 1.0 * 1.0 * 0.001 = 0.001 < 0.01

    def test_自定义深度和衰减(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", weight=1.0)
        activations = g.spreading_activation("a", depth=3, decay=0.8)
        assert activations["b"] == pytest.approx(0.8)

    def test_不存在抛异常(self) -> None:
        g = _make_graph()
        with pytest.raises(NodeNotFoundError):
            g.spreading_activation("ghost")


class TestGraphCycleDetection:
    """记忆图谱 - 环检测."""

    def test_has_cycle_无环(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b", EdgeType.PREREQUISITE)
        g.add_edge("b", "c", EdgeType.PREREQUISITE)
        assert g.has_cycle() is False

    def test_has_cycle_有环(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b", EdgeType.PREREQUISITE, check_cycle=False)
        g.add_edge("b", "c", EdgeType.PREREQUISITE, check_cycle=False)
        g.add_edge("c", "a", EdgeType.PREREQUISITE, check_cycle=False)
        assert g.has_cycle() is True

    def test_has_cycle_仅检测前置边(self) -> None:
        """RELATED 边不参与环检测."""
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", EdgeType.RELATED)
        g.add_edge("b", "a", EdgeType.RELATED)
        assert g.has_cycle() is False

    def test_has_cycle_空图谱(self) -> None:
        g = _make_graph()
        assert g.has_cycle() is False


class TestGraphExportImport:
    """记忆图谱 - 导出/导入."""

    def test_export(self) -> None:
        g = _make_graph()
        g.add_node("a", NodeType.KNOWLEDGE, {"title": "化学键"})
        g.add_node("b", NodeType.KNOWLEDGE, {"title": "分子轨道"})
        g.add_edge("a", "b", EdgeType.PREREQUISITE)
        data = g.export()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert "metrics" in data

    def test_import_后恢复(self) -> None:
        g1 = _make_graph()
        g1.add_node("a", NodeType.KNOWLEDGE, {"title": "化学键"}, strength=0.8)
        g1.add_node("b", NodeType.KNOWLEDGE, {"title": "分子轨道"})
        g1.add_edge("a", "b", EdgeType.PREREQUISITE, 0.9)
        data = g1.export()

        g2 = _make_graph()
        g2.import_data(data)
        assert g2.node_count() == 2
        assert g2.edge_count() == 1
        assert g2.get_node("a").content == {"title": "化学键"}
        assert g2.get_node("a").strength == pytest.approx(0.8)
        e = g2.get_edge("a", "b")
        assert e.edge_type == EdgeType.PREREQUISITE
        assert e.weight == pytest.approx(0.9)

    def test_import_追加不覆盖(self) -> None:
        g = _make_graph()
        g.add_node("existing")
        g.import_data({"nodes": [{"node_id": "new", "node_type": "knowledge"}], "edges": []})
        assert g.has_node("existing")
        assert g.has_node("new")

    def test_export_空图谱(self) -> None:
        g = _make_graph()
        data = g.export()
        assert data["nodes"] == []
        assert data["edges"] == []


class TestGraphMetrics:
    """记忆图谱 - 度量统计."""

    def test_node_count_edge_count(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        m = g.get_metrics()
        assert m["node_count"] == 3
        assert m["edge_count"] == 2

    def test_access_count(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.touch_node("a")
        g.touch_node("a")
        m = g.get_metrics()
        assert m["access_count"] == 2

    def test_decay_count_and_pruned(self) -> None:
        g = _make_graph(decay_factor=0.1, min_strength=0.06)
        g.add_node("a", strength=0.5)
        g.add_node("b", strength=1.0)
        g.decay()
        m = g.get_metrics()
        assert m["decay_count"] == 1
        assert m["pruned_count"] == 1

    def test_spreading_count(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.spreading_activation("a")
        m = g.get_metrics()
        assert m["spreading_count"] == 1

    def test_cycle_rejected(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", EdgeType.PREREQUISITE)
        try:
            g.add_edge("b", "a", EdgeType.PREREQUISITE)
        except GraphCycleError:
            pass
        m = g.get_metrics()
        assert m["cycle_rejected"] == 1

    def test_type_distribution(self) -> None:
        g = _make_graph()
        g.add_node("k1", NodeType.KNOWLEDGE)
        g.add_node("k2", NodeType.KNOWLEDGE)
        g.add_node("s1", NodeType.SKILL)
        m = g.get_metrics()
        assert m["type_distribution"]["knowledge"] == 2
        assert m["type_distribution"]["skill"] == 1


class TestGraphReset:
    """记忆图谱 - 重置."""

    def test_reset_清除所有(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.touch_node("a")
        g.reset()
        assert g.node_count() == 0
        assert g.edge_count() == 0
        m = g.get_metrics()
        assert m["access_count"] == 0


class TestGraphCountHelpers:
    """记忆图谱 - 计数辅助."""

    def test_node_count(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        assert g.node_count() == 2

    def test_edge_count(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert g.edge_count() == 2

    def test_edge_count_删除后更新(self) -> None:
        g = _make_graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        assert g.edge_count() == 1
        g.remove_edge("a", "b")
        assert g.edge_count() == 0


# ============================================================
# 11. MemoryGraphMetrics 测试
# ============================================================

class TestMemoryGraphMetrics:
    """记忆图谱度量收集器测试."""

    def test_record_access(self) -> None:
        m = MemoryGraphMetrics()
        m.record_access()
        m.record_access()
        assert m.export()["access_count"] == 2

    def test_record_decay(self) -> None:
        m = MemoryGraphMetrics()
        m.record_decay(pruned=3)
        e = m.export()
        assert e["decay_count"] == 1
        assert e["pruned_count"] == 3

    def test_record_spreading(self) -> None:
        m = MemoryGraphMetrics()
        m.record_spreading()
        assert m.export()["spreading_count"] == 1

    def test_record_cycle_rejected(self) -> None:
        m = MemoryGraphMetrics()
        m.record_cycle_rejected()
        assert m.export()["cycle_rejected"] == 1

    def test_reset(self) -> None:
        m = MemoryGraphMetrics()
        m.record_access()
        m.record_decay(1)
        m.reset()
        e = m.export()
        assert all(v == 0 for v in e.values())

    def test线程安全(self) -> None:
        m = MemoryGraphMetrics()
        def worker() -> None:
            for _ in range(100):
                m.record_access()
                m.record_decay(1)
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        e = m.export()
        assert e["access_count"] == 1000
        assert e["pruned_count"] == 1000


# ============================================================
# 12. 集成场景测试
# ============================================================

class TestIntegrationScenarios:
    """广播 + 记忆图谱集成场景测试."""

    def test_广播事件触发图谱强化(self) -> None:
        """模拟: 广播学情事件 → 触发记忆图谱强化."""
        bus = _make_bus()
        graph = _make_graph()
        graph.add_node("kp-1", NodeType.KNOWLEDGE, {"title": "化学键"}, strength=0.5)
        graph.add_node("kp-2", NodeType.KNOWLEDGE, {"title": "分子轨道"}, strength=0.5)
        graph.add_edge("kp-1", "kp-2", EdgeType.PREREQUISITE)

        # 订阅学情事件, 触发扩散激活
        def on_learner_event(event: BroadcastEvent) -> None:
            kp_id = event.payload.get("kp_id")
            if kp_id and graph.has_node(kp_id):
                graph.spreading_activation(kp_id)

        bus.subscribe("memory-agent", "learner.**", on_learner_event)

        # 发布学情事件
        before_strength = graph.get_node("kp-2").strength
        bus.publish("learner.assessment.completed", {"kp_id": "kp-1"})
        after_strength = graph.get_node("kp-2").strength

        # kp-2 应被扩散激活强化 (初始 0.5, 扩散后增加)
        assert after_strength > before_strength

    def test_衰减后低强度节点不参与扩散(self) -> None:
        """衰减清除节点后, 扩散激活不会触及已清除节点."""
        g = _make_graph(decay_factor=0.15, min_strength=0.1, spreading_depth=2)
        g.add_node("a", strength=1.0)
        g.add_node("b", strength=0.5)
        g.add_node("c", strength=0.3)
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("b", "c", weight=1.0)

        # 衰减后 b (0.075) 和 c (0.045) 被清除, a (0.15) 保留
        pruned = g.decay()
        assert pruned == 2

        # 扩散激活只有 a
        activations = g.spreading_activation("a")
        assert "a" in activations
        assert "b" not in activations

    def test_多主题多订阅者场景(self) -> None:
        """模拟: 多个 Agent 订阅不同主题模式."""
        bus = _make_bus()
        events_a: list[str] = []
        events_b: list[str] = []
        events_c: list[str] = []

        bus.subscribe("agent-a", "learner.profile.*", lambda e: events_a.append(e.topic))
        bus.subscribe("agent-b", "learner.assessment.*", lambda e: events_b.append(e.topic))
        bus.subscribe("agent-c", "learner.**", lambda e: events_c.append(e.topic))

        bus.publish("learner.profile.updated", {})
        bus.publish("learner.assessment.completed", {})

        assert events_a == ["learner.profile.updated"]
        assert events_b == ["learner.assessment.completed"]
        assert len(events_c) == 2  # 匹配两个

    def test_知识图谱前置链验证(self) -> None:
        """模拟: 化学知识图谱前置依赖链."""
        g = _make_graph()
        kp_chain = ["原子结构", "化学键", "分子几何", "分子轨道理论"]
        for kp in kp_chain:
            g.add_node(kp, NodeType.KNOWLEDGE, {"title": kp})

        for i in range(len(kp_chain) - 1):
            g.add_edge(kp_chain[i], kp_chain[i + 1], EdgeType.PREREQUISITE)

        # 验证无环
        assert g.has_cycle() is False

        # 验证路径
        path = g.find_path("原子结构", "分子轨道理论")
        assert path == kp_chain

        # 尝试添加反向前置依赖 → 应被阻止
        with pytest.raises(GraphCycleError):
            g.add_edge("分子轨道理论", "原子结构", EdgeType.PREREQUISITE)
