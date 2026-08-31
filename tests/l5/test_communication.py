"""通信与状态传递模块测试 — TDD 测试用例.

测试覆盖:
1. Message — 不可变消息载体 (统一格式)
2. MessageBus — 消息总线 (Pub/Sub 引擎 + 频道管理)
3. ChannelSubscription — 频道订阅管理
4. StatePropagator — 状态传播器 (共享状态 + Reducer 聚合)
5. StateUpdate — 状态更新 (含 Reducer 类型)
6. HandoffManager — Agent 控制权移交 (OpenAI Handoff 模式)
7. HandoffContext — 移交上下文
8. MessageRouter — 消息路由 (Fork 前缀隔离 + 频道路由)
9. 集成测试 — 与 SessionManager/OrchestrationEngine 联动
10. 错误处理 — 通信异常与恢复

融合世界先进方案:
- LangGraph: Channel + Reducer + BSP 屏障同步
- OpenAI Agents SDK: Handoff 单向移交 + 全量上下文传递
- Google ADK: Session/State/Memory 三层 + output_key + 分层作用域
- AutoGen: Topic 发布-订阅 + GroupChat Manager
- Temporal: Signal/Query + 事件历史 + 确定性重放
- Kafka/RabbitMQ: 发布-订阅 + 消费确认 + 分区日志
- CrewAI: 任务上下文链 + 委派工具
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from dy3_polaris.l5.communication import (
    ChannelSubscription,
    CommunicationError,
    HandoffContext,
    HandoffManager,
    HandoffState,
    Message,
    MessageBus,
    MessagePriority,
    MessageRouter,
    ReducerType,
    StatePropagator,
    StateScope,
    StateUpdate,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def message_bus():
    """创建消息总线实例."""
    return MessageBus()


@pytest.fixture
def state_propagator():
    """创建状态传播器实例."""
    return StatePropagator()


@pytest.fixture
def handoff_manager():
    """创建 Handoff 管理器实例."""
    return HandoffManager()


@pytest.fixture
def message_router():
    """创建消息路由器实例."""
    return MessageRouter()


# ============================================================
# 1. Message 测试
# ============================================================


class TestMessage:
    """不可变消息载体测试 (统一消息格式)."""

    def test_message_creation(self):
        """创建消息应自动生成 ID 和时间戳."""
        msg = Message(
            channel="learning.diagnosis.report",
            publisher="agent.learning.diagnosis",
            payload={"report_id": "rpt-001", "kp_gaps": ["KP-12"]},
        )
        assert msg.message_id.startswith("msg-")
        assert msg.channel == "learning.diagnosis.report"
        assert msg.publisher == "agent.learning.diagnosis"
        assert msg.payload["report_id"] == "rpt-001"
        assert msg.timestamp > 0
        assert msg.stream_id is None

    def test_message_with_stream_id(self):
        """消息可携带 Redis Streams 消费位点."""
        msg = Message(
            channel="learning.interaction.event",
            publisher="L1",
            payload={"event_type": "answer"},
            stream_id="1637356800000-0",
        )
        assert msg.stream_id == "1637356800000-0"

    def test_message_immutability(self):
        """消息应是不可变的 (frozen)."""
        msg = Message(
            channel="test",
            publisher="agent.test",
            payload={"key": "value"},
        )
        with pytest.raises((AttributeError, TypeError)):
            msg.channel = "changed"

    def test_message_priority(self):
        """消息支持优先级 (Kafka 分区优先级模式)."""
        msg = Message(
            channel="guidance.decision.command",
            publisher="agent.guidance",
            payload={"action": "intervene"},
            priority=MessagePriority.HIGH,
        )
        assert msg.priority == MessagePriority.HIGH

    def test_message_default_priority(self):
        """消息默认优先级为 NORMAL."""
        msg = Message(
            channel="test",
            publisher="agent.test",
            payload={},
        )
        assert msg.priority == MessagePriority.NORMAL

    def test_message_to_dict(self):
        """消息应可序列化为字典."""
        msg = Message(
            channel="test.channel",
            publisher="agent.test",
            payload={"data": 123},
        )
        d = msg.to_dict()
        assert d["channel"] == "test.channel"
        assert d["publisher"] == "agent.test"
        assert d["payload"]["data"] == 123
        assert "message_id" in d
        assert "timestamp" in d


# ============================================================
# 2. MessageBus 测试
# ============================================================


class TestMessageBus:
    """消息总线测试 (Pub/Sub 引擎 + 频道管理)."""

    def test_bus_creation(self, message_bus):
        """创建消息总线."""
        assert message_bus is not None
        assert len(message_bus.channels) == 0

    def test_create_channel(self, message_bus):
        """创建频道."""
        message_bus.create_channel("learning.diagnosis.report")
        assert "learning.diagnosis.report" in message_bus.channels

    def test_create_duplicate_channel_raises(self, message_bus):
        """重复创建频道应抛异常."""
        message_bus.create_channel("test.channel")
        with pytest.raises(CommunicationError, match="already exists"):
            message_bus.create_channel("test.channel")

    @pytest.mark.asyncio
    async def test_subscribe_to_channel(self, message_bus):
        """订阅频道."""
        message_bus.create_channel("test.channel")
        sub = message_bus.subscribe(
            channel="test.channel",
            subscriber_id="agent.test",
        )
        assert sub.channel == "test.channel"
        assert sub.subscriber_id == "agent.test"
        assert sub.active is True

    @pytest.mark.asyncio
    async def test_subscribe_nonexistent_channel_raises(self, message_bus):
        """订阅不存在的频道应抛异常."""
        with pytest.raises(CommunicationError, match="not found"):
            message_bus.subscribe("nonexistent", "agent.test")

    @pytest.mark.asyncio
    async def test_publish_message(self, message_bus):
        """发布消息到频道."""
        message_bus.create_channel("test.channel")
        received: list[Message] = []

        message_bus.subscribe(
            channel="test.channel",
            subscriber_id="agent.sub",
            callback=lambda msg: received.append(msg),
        )

        msg = Message(
            channel="test.channel",
            publisher="agent.pub",
            payload={"data": "hello"},
        )
        message_bus.publish(msg)

        assert len(received) == 1
        assert received[0].payload["data"] == "hello"

    @pytest.mark.asyncio
    async def test_publish_to_multiple_subscribers(self, message_bus):
        """一条消息应分发给所有订阅者 (广播模式)."""
        message_bus.create_channel("broadcast.channel")
        received_a: list[Message] = []
        received_b: list[Message] = []

        message_bus.subscribe("broadcast.channel", "agent.a",
                              callback=lambda msg: received_a.append(msg))
        message_bus.subscribe("broadcast.channel", "agent.b",
                              callback=lambda msg: received_b.append(msg))

        msg = Message(
            channel="broadcast.channel",
            publisher="agent.pub",
            payload={"x": 1},
        )
        message_bus.publish(msg)

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_publish_to_nonexistent_channel_creates(self, message_bus):
        """发布到不存在的频道应懒创建 (支持按需广播)."""
        msg = Message(
            channel="nonexistent",
            publisher="agent.test",
            payload={},
        )
        message_bus.publish(msg)
        assert "nonexistent" in message_bus.channels
        # 消息历史已记录
        hist = message_bus.get_history("nonexistent")
        assert hist and hist[0].message_id == msg.message_id

    @pytest.mark.asyncio
    async def test_unsubscribe(self, message_bus):
        """取消订阅后不再收到消息."""
        message_bus.create_channel("test.channel")
        received: list[Message] = []

        sub = message_bus.subscribe("test.channel", "agent.test",
                                    callback=lambda msg: received.append(msg))
        sub.unsubscribe()

        msg = Message("test.channel", "agent.pub", {"data": 1})
        message_bus.publish(msg)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_get_channel_subscribers(self, message_bus):
        """获取频道订阅者列表."""
        message_bus.create_channel("test.channel")
        message_bus.subscribe("test.channel", "agent.a")
        message_bus.subscribe("test.channel", "agent.b")

        subs = message_bus.get_subscribers("test.channel")
        assert len(subs) == 2
        assert "agent.a" in subs
        assert "agent.b" in subs

    @pytest.mark.asyncio
    async def test_message_history(self, message_bus):
        """消息总线应保留消息历史 (Kafka 日志模式)."""
        message_bus.create_channel("test.channel")
        message_bus.subscribe("test.channel", "agent.test")

        for i in range(5):
            msg = Message("test.channel", "agent.pub", {"index": i})
            message_bus.publish(msg)

        history = message_bus.get_history("test.channel")
        assert len(history) == 5
        assert history[0].payload["index"] == 0
        assert history[4].payload["index"] == 4

    @pytest.mark.asyncio
    async def test_message_history_with_limit(self, message_bus):
        """消息历史支持限制返回数量."""
        message_bus.create_channel("test.channel")
        message_bus.subscribe("test.channel", "agent.test")

        for i in range(10):
            message_bus.publish(Message("test.channel", "agent.pub", {"i": i}))

        recent = message_bus.get_history("test.channel", limit=3)
        assert len(recent) == 3
        assert recent[0].payload["i"] == 7
        assert recent[2].payload["i"] == 9

    @pytest.mark.asyncio
    async def test_async_callback(self, message_bus):
        """支持异步回调函数."""
        message_bus.create_channel("test.channel")
        received: list[Message] = []

        async def async_callback(msg: Message) -> None:
            received.append(msg)

        message_bus.subscribe("test.channel", "agent.test", callback=async_callback)

        message_bus.publish(Message("test.channel", "agent.pub", {"data": 1}))
        # 处理异步回调
        await asyncio.sleep(0.01)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_close_channel(self, message_bus):
        """关闭频道后无法发布消息."""
        message_bus.create_channel("test.channel")
        message_bus.subscribe("test.channel", "agent.test")

        message_bus.close_channel("test.channel")

        with pytest.raises(CommunicationError, match="closed"):
            message_bus.publish(Message("test.channel", "agent.pub", {}))

    @pytest.mark.asyncio
    async def test_stream_id_assignment(self, message_bus):
        """发布消息应自动分配 stream_id (Redis Streams 消费位点)."""
        message_bus.create_channel("test.channel")
        message_bus.subscribe("test.channel", "agent.test")

        msg = Message("test.channel", "agent.pub", {"data": 1})
        message_bus.publish(msg)

        assert msg.stream_id is not None
        assert isinstance(msg.stream_id, str)


# ============================================================
# 3. ChannelSubscription 测试
# ============================================================


class TestChannelSubscription:
    """频道订阅管理测试."""

    def test_subscription_creation(self):
        """创建订阅记录."""
        sub = ChannelSubscription(
            channel="test.channel",
            subscriber_id="agent.test",
        )
        assert sub.channel == "test.channel"
        assert sub.subscriber_id == "agent.test"
        assert sub.active is True

    def test_subscription_unsubscribe(self):
        """取消订阅."""
        sub = ChannelSubscription(
            channel="test.channel",
            subscriber_id="agent.test",
        )
        sub.unsubscribe()
        assert sub.active is False

    def test_subscription_with_callback(self):
        """订阅可携带回调函数."""
        received: list[Message] = []
        sub = ChannelSubscription(
            channel="test.channel",
            subscriber_id="agent.test",
            callback=lambda msg: received.append(msg),
        )
        msg = Message("test.channel", "agent.pub", {"data": 1})
        assert sub.callback is not None
        sub.callback(msg)
        assert len(received) == 1


# ============================================================
# 4. StatePropagator 测试
# ============================================================


class TestStatePropagator:
    """状态传播器测试 (共享状态 + Reducer 聚合)."""

    def test_propagator_creation(self, state_propagator):
        """创建状态传播器."""
        assert state_propagator is not None
        assert len(state_propagator.state) == 0

    def test_set_state(self, state_propagator):
        """设置状态 (LastValue Reducer, 覆盖式)."""
        state_propagator.set("learner_id", "stu-001", source_agent="agent.diagnosis")
        assert state_propagator.get("learner_id") == "stu-001"

    def test_get_nonexistent_state(self, state_propagator):
        """获取不存在的状态返回 None."""
        assert state_propagator.get("nonexistent") is None

    def test_state_with_scope(self, state_propagator):
        """分层作用域 (ADK 前缀模式: session/user/app/temp)."""
        state_propagator.set("session_var", 1, scope=StateScope.SESSION)
        state_propagator.set("user_pref", "dark", scope=StateScope.USER)
        state_propagator.set("app_config", {"v": 1}, scope=StateScope.APP)
        state_propagator.set("temp_cache", "x", scope=StateScope.TEMP)

        assert state_propagator.get("session_var") == 1
        assert state_propagator.get("user_pref") == "dark"
        assert state_propagator.get("app_config")["v"] == 1
        assert state_propagator.get("temp_cache") == "x"

    def test_state_update_event(self, state_propagator):
        """状态更新应生成 StateUpdate 事件."""
        state_propagator.set("step", 5, source_agent="agent.test")
        updates = state_propagator.get_updates()
        assert len(updates) == 1
        assert updates[0].key == "step"
        assert updates[0].value == 5
        assert updates[0].source_agent == "agent.test"
        assert updates[0].reducer_type == ReducerType.LAST_VALUE

    def test_reducer_last_value(self, state_propagator):
        """LastValue Reducer: 新值覆盖旧值 (LangGraph 默认)."""
        state_propagator.set("x", 1, reducer=ReducerType.LAST_VALUE)
        state_propagator.set("x", 2, reducer=ReducerType.LAST_VALUE)
        assert state_propagator.get("x") == 2

    def test_reducer_accumulate_list(self, state_propagator):
        """AccumulateList Reducer: 列表追加 (LangGraph add_messages 模式)."""
        state_propagator.set("messages", ["msg1"], reducer=ReducerType.ACCUMULATE_LIST)
        state_propagator.set("messages", ["msg2"], reducer=ReducerType.ACCUMULATE_LIST)
        assert state_propagator.get("messages") == ["msg1", "msg2"]

    def test_reducer_sum(self, state_propagator):
        """Sum Reducer: 数值累加 (BinaryOperatorAggregate 模式)."""
        state_propagator.set("token_count", 10, reducer=ReducerType.SUM)
        state_propagator.set("token_count", 5, reducer=ReducerType.SUM)
        assert state_propagator.get("token_count") == 15

    def test_reducer_max(self, state_propagator):
        """Max Reducer: 取最大值."""
        state_propagator.set("confidence", 0.8, reducer=ReducerType.MAX)
        state_propagator.set("confidence", 0.6, reducer=ReducerType.MAX)
        assert state_propagator.get("confidence") == 0.8

    def test_reducer_merge_dict(self, state_propagator):
        """MergeDict Reducer: 字典合并."""
        state_propagator.set("mastery", {"KP-01": 0.9}, reducer=ReducerType.MERGE_DICT)
        state_propagator.set("mastery", {"KP-02": 0.5}, reducer=ReducerType.MERGE_DICT)
        assert state_propagator.get("mastery") == {"KP-01": 0.9, "KP-02": 0.5}

    def test_state_checkpoint(self, state_propagator):
        """状态检查点 (LangGraph checkpoint 模式)."""
        state_propagator.set("step", 5)
        state_propagator.set("path", ["a", "b"])

        cp_id = state_propagator.checkpoint()
        assert cp_id.startswith("state-cp-")

        state_propagator.set("step", 10)
        assert state_propagator.get("step") == 10

        assert state_propagator.restore(cp_id) is True
        assert state_propagator.get("step") == 5
        assert state_propagator.get("path") == ["a", "b"]

    def test_state_restore_nonexistent(self, state_propagator):
        """恢复不存在的检查点返回 False."""
        assert state_propagator.restore("state-cp-nonexistent") is False

    def test_state_clear_temp_scope(self, state_propagator):
        """清除临时作用域状态 (ADK temp: 前缀模式)."""
        state_propagator.set("temp_data", "x", scope=StateScope.TEMP)
        state_propagator.set("session_data", "y", scope=StateScope.SESSION)

        state_propagator.clear_scope(StateScope.TEMP)

        assert state_propagator.get("temp_data") is None
        assert state_propagator.get("session_data") == "y"

    def test_state_snapshot(self, state_propagator):
        """状态快照 (完整状态导出)."""
        state_propagator.set("a", 1)
        state_propagator.set("b", "hello")
        snapshot = state_propagator.snapshot()
        assert snapshot["a"] == 1
        assert snapshot["b"] == "hello"

    def test_state_with_provenance(self, state_propagator):
        """状态更新应记录溯源 (Temporal 事件历史模式)."""
        state_propagator.set("x", 1, source_agent="agent.diagnosis")
        state_propagator.set("x", 2, source_agent="agent.generation")

        updates = state_propagator.get_updates()
        assert len(updates) == 2
        assert updates[0].source_agent == "agent.diagnosis"
        assert updates[1].source_agent == "agent.generation"
        assert updates[0].timestamp <= updates[1].timestamp


# ============================================================
# 5. StateUpdate 测试
# ============================================================


class TestStateUpdate:
    """状态更新事件测试."""

    def test_update_creation(self):
        """创建状态更新."""
        update = StateUpdate(
            key="learner_id",
            value="stu-001",
            source_agent="agent.diagnosis",
        )
        assert update.key == "learner_id"
        assert update.value == "stu-001"
        assert update.source_agent == "agent.diagnosis"
        assert update.reducer_type == ReducerType.LAST_VALUE
        assert update.timestamp > 0

    def test_update_with_reducer(self):
        """状态更新指定 Reducer 类型."""
        update = StateUpdate(
            key="messages",
            value=["msg1"],
            source_agent="agent.test",
            reducer_type=ReducerType.ACCUMULATE_LIST,
        )
        assert update.reducer_type == ReducerType.ACCUMULATE_LIST

    def test_update_with_scope(self):
        """状态更新指定作用域."""
        update = StateUpdate(
            key="user_pref",
            value="dark_mode",
            source_agent="agent.test",
            scope=StateScope.USER,
        )
        assert update.scope == StateScope.USER


# ============================================================
# 6. HandoffManager 测试
# ============================================================


class TestHandoffManager:
    """Agent 控制权移交测试 (OpenAI Agents SDK Handoff 模式)."""

    def test_handoff_manager_creation(self, handoff_manager):
        """创建 Handoff 管理器."""
        assert handoff_manager is not None

    @pytest.mark.asyncio
    async def test_handoff_execution(self, handoff_manager):
        """执行 Handoff: Agent A → Agent B."""
        async def target_handler(ctx: HandoffContext) -> dict[str, Any]:
            return {"result": "handled by B", "history_len": len(ctx.conversation_history)}

        handoff_manager.register_agent("agent.b", target_handler)

        ctx = HandoffContext(
            from_agent="agent.a",
            to_agent="agent.b",
            conversation_history=[{"role": "user", "content": "hello"}],
            state_snapshot={"step": 5},
        )

        result = await handoff_manager.execute_handoff(ctx)
        assert result["result"] == "handled by B"
        assert result["history_len"] == 1

    @pytest.mark.asyncio
    async def test_handoff_context_transfer(self, handoff_manager):
        """Handoff 应传递完整上下文 (对话历史 + 状态快照)."""
        received_ctx: list[HandoffContext] = []

        async def handler(ctx: HandoffContext) -> dict[str, Any]:
            received_ctx.append(ctx)
            return {"status": "ok"}

        handoff_manager.register_agent("agent.target", handler)

        original_history = [
            {"role": "user", "content": "什么是发光材料?"},
            {"role": "assistant", "content": "发光材料是..."},
        ]
        original_state = {"learner_id": "stu-001", "step": 3}

        ctx = HandoffContext(
            from_agent="agent.a",
            to_agent="agent.target",
            conversation_history=original_history,
            state_snapshot=original_state,
        )

        await handoff_manager.execute_handoff(ctx)

        assert len(received_ctx) == 1
        assert received_ctx[0].conversation_history == original_history
        assert received_ctx[0].state_snapshot["learner_id"] == "stu-001"
        assert received_ctx[0].state_snapshot["step"] == 3

    @pytest.mark.asyncio
    async def test_handoff_unknown_target_raises(self, handoff_manager):
        """移交到未注册的 Agent 应抛异常."""
        ctx = HandoffContext(
            from_agent="agent.a",
            to_agent="agent.unknown",
            conversation_history=[],
            state_snapshot={},
        )
        with pytest.raises(CommunicationError, match="not registered"):
            await handoff_manager.execute_handoff(ctx)

    def test_handoff_context_creation(self):
        """创建 Handoff 上下文."""
        ctx = HandoffContext(
            from_agent="agent.triage",
            to_agent="agent.specialist",
            conversation_history=[{"role": "user", "content": "hi"}],
            state_snapshot={"key": "value"},
        )
        assert ctx.from_agent == "agent.triage"
        assert ctx.to_agent == "agent.specialist"
        assert len(ctx.conversation_history) == 1
        assert ctx.state_snapshot["key"] == "value"
        assert ctx.state == HandoffState.PENDING

    @pytest.mark.asyncio
    async def test_handoff_chain(self, handoff_manager):
        """Handoff 链: A → B → C (OpenAI 多级移交)."""
        results: list[str] = []

        async def handler_b(ctx: HandoffContext) -> dict[str, Any]:
            results.append("B")
            # B 移交给 C
            ctx_c = HandoffContext(
                from_agent="agent.b",
                to_agent="agent.c",
                conversation_history=ctx.conversation_history + [{"B": "done"}],
                state_snapshot=ctx.state_snapshot,
            )
            return await handoff_manager.execute_handoff(ctx_c)

        async def handler_c(ctx: HandoffContext) -> dict[str, Any]:
            results.append("C")
            return {"chain": results}

        handoff_manager.register_agent("agent.b", handler_b)
        handoff_manager.register_agent("agent.c", handler_c)

        ctx = HandoffContext(
            from_agent="agent.a",
            to_agent="agent.b",
            conversation_history=[{"role": "user", "content": "start"}],
            state_snapshot={},
        )

        result = await handoff_manager.execute_handoff(ctx)
        assert results == ["B", "C"]
        assert result["chain"] == ["B", "C"]

    @pytest.mark.asyncio
    async def test_handoff_state_transitions(self, handoff_manager):
        """Handoff 状态转换: PENDING → IN_PROGRESS → COMPLETED."""
        states: list[HandoffState] = []

        async def handler(ctx: HandoffContext) -> dict[str, Any]:
            states.append(ctx.state)
            return {"status": "ok"}

        handoff_manager.register_agent("agent.b", handler)

        ctx = HandoffContext(
            from_agent="agent.a",
            to_agent="agent.b",
            conversation_history=[],
            state_snapshot={},
        )
        assert ctx.state == HandoffState.PENDING

        await handoff_manager.execute_handoff(ctx)
        assert ctx.state == HandoffState.COMPLETED
        assert states[0] == HandoffState.IN_PROGRESS


# ============================================================
# 7. MessageRouter 测试
# ============================================================


class TestMessageRouter:
    """消息路由器测试 (Fork 前缀隔离 + 频道路由)."""

    def test_router_creation(self, message_router):
        """创建消息路由器."""
        assert message_router is not None

    @pytest.mark.asyncio
    async def test_route_to_subscribers(self, message_router):
        """路由消息到频道订阅者."""
        received: list[Message] = []
        message_router.register_route(
            channel="learning.diagnosis.report",
            subscriber_id="agent.generation",
            callback=lambda msg: received.append(msg),
        )

        msg = Message(
            channel="learning.diagnosis.report",
            publisher="agent.diagnosis",
            payload={"report_id": "rpt-001"},
        )
        message_router.route(msg)

        assert len(received) == 1
        assert received[0].payload["report_id"] == "rpt-001"

    @pytest.mark.asyncio
    async def test_fork_prefix_isolation(self, message_router):
        """Fork 前缀隔离: fork.123.* 消息不路由到主会话订阅者."""
        main_received: list[Message] = []
        fork_received: list[Message] = []

        # 主会话订阅
        message_router.register_route(
            channel="learning.diagnosis.report",
            subscriber_id="agent.main",
            callback=lambda msg: main_received.append(msg),
        )
        # Fork 订阅 (带前缀)
        message_router.register_route(
            channel="fork.f123.learning.diagnosis.report",
            subscriber_id="agent.fork",
            callback=lambda msg: fork_received.append(msg),
        )

        # 发布到主会话
        message_router.route(Message("learning.diagnosis.report", "agent.pub", {"v": 1}))
        # 发布到 Fork
        message_router.route(Message("fork.f123.learning.diagnosis.report", "agent.pub", {"v": 2}))

        assert len(main_received) == 1
        assert main_received[0].payload["v"] == 1
        assert len(fork_received) == 1
        assert fork_received[0].payload["v"] == 2

    @pytest.mark.asyncio
    async def test_route_no_subscribers(self, message_router):
        """无订阅者的消息不报错 (静默丢弃)."""
        msg = Message("orphan.channel", "agent.test", {"data": 1})
        # 不应抛异常
        message_router.route(msg)

    @pytest.mark.asyncio
    async def test_route_priority_ordering(self, message_router):
        """高优先级消息应先路由 (Kafka 分区优先级模式, route_batch 批量排序)."""
        received: list[Message] = []
        message_router.register_route(
            channel="test.channel",
            subscriber_id="agent.test",
            callback=lambda msg: received.append(msg),
        )

        # 批量路由: 先低后高, route_batch 按优先级排序后分发
        message_router.route_batch([
            Message("test.channel", "agent.pub", {"order": 1},
                    priority=MessagePriority.LOW),
            Message("test.channel", "agent.pub", {"order": 2},
                    priority=MessagePriority.HIGH),
        ])

        # 高优先级应先被处理
        assert received[0].payload["order"] == 2
        assert received[1].payload["order"] == 1

    @pytest.mark.asyncio
    async def test_unregister_route(self, message_router):
        """注销路由后不再收到消息."""
        received: list[Message] = []
        message_router.register_route(
            "test.channel", "agent.test",
            callback=lambda msg: received.append(msg),
        )

        message_router.unregister_route("test.channel", "agent.test")

        message_router.route(Message("test.channel", "agent.pub", {"v": 1}))
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_route_with_fork_prefix_mapping(self, message_router):
        """Fork 前缀映射: Fork 内消息可映射回主频道."""
        fork_received: list[Message] = []
        message_router.register_route(
            channel="fork.f456.learning.diagnosis.report",
            subscriber_id="agent.fork",
            callback=lambda msg: fork_received.append(msg),
        )

        # 通过 Fork 前缀发布
        msg = Message(
            "fork.f456.learning.diagnosis.report",
            "agent.diagnosis",
            {"report_id": "rpt-fork"},
        )
        message_router.route(msg)

        assert len(fork_received) == 1
        # 消息应保留原始 Fork 前缀
        assert fork_received[0].channel == "fork.f456.learning.diagnosis.report"

    def test_list_channels(self, message_router):
        """列出所有已注册的频道."""
        message_router.register_route("channel.a", "agent.1")
        message_router.register_route("channel.b", "agent.2")
        message_router.register_route("channel.b", "agent.3")

        channels = message_router.list_channels()
        assert "channel.a" in channels
        assert "channel.b" in channels


# ============================================================
# 8. 集成测试
# ============================================================


class TestCommunicationIntegration:
    """通信与状态传递集成测试."""

    @pytest.mark.asyncio
    async def test_bus_with_state_propagator(self):
        """消息总线 + 状态传播器联动 (状态变更触发广播)."""
        bus = MessageBus()
        propagator = StatePropagator()

        bus.create_channel("state.updates")
        received_updates: list[StateUpdate] = []

        bus.subscribe(
            "state.updates", "agent.listener",
            callback=lambda msg: received_updates.append(
                StateUpdate(
                    key=msg.payload["key"],
                    value=msg.payload["value"],
                    source_agent=msg.publisher,
                    reducer_type=ReducerType(msg.payload.get("reducer", "last_value")),
                )
            ),
        )

        # 状态变更触发广播
        propagator.set("step", 5, source_agent="agent.diagnosis")

        # 手动发布状态变更
        bus.publish(Message(
            "state.updates", "agent.diagnosis",
            {"key": "step", "value": 5, "reducer": "last_value"},
        ))

        assert len(received_updates) == 1
        assert received_updates[0].key == "step"
        assert received_updates[0].value == 5

    @pytest.mark.asyncio
    async def test_full_communication_lifecycle(self):
        """完整通信生命周期: 创建 → 订阅 → 发布 → 接收 → 关闭."""
        bus = MessageBus()
        received: list[Message] = []

        # 1. 创建频道
        bus.create_channel("lifecycle.test")

        # 2. 订阅
        sub = bus.subscribe("lifecycle.test", "agent.test",
                            callback=lambda msg: received.append(msg))

        # 3. 发布
        bus.publish(Message("lifecycle.test", "agent.pub", {"phase": "publish"}))
        assert len(received) == 1

        # 4. 取消订阅
        sub.unsubscribe()
        bus.publish(Message("lifecycle.test", "agent.pub", {"phase": "after_unsub"}))
        assert len(received) == 1  # 未增加

        # 5. 关闭频道
        bus.close_channel("lifecycle.test")
        assert "lifecycle.test" in bus.closed_channels

    @pytest.mark.asyncio
    async def test_multi_agent_broadcast_topology(self):
        """多 Agent 广播拓扑 (L5 设计文档 7 频道拓扑)."""
        bus = MessageBus()
        propagator = StatePropagator()

        # 创建 7 个频道 (L5 设计文档)
        channels = [
            "learning.interaction.event",
            "learning.diagnosis.report",
            "learning.knowledge.gap",
            "knowledge.generation.output",
            "knowledge.review.result",
            "guidance.decision.command",
            "guidance.adaptation.trigger",
        ]
        for ch in channels:
            bus.create_channel(ch)

        # 订阅关系 (简化版)
        bus.subscribe("learning.interaction.event", "agent.diagnosis")
        bus.subscribe("learning.interaction.event", "agent.guidance")
        bus.subscribe("learning.diagnosis.report", "agent.generation")
        bus.subscribe("learning.diagnosis.report", "agent.review")
        bus.subscribe("learning.diagnosis.report", "agent.guidance")
        bus.subscribe("guidance.decision.command", "agent.diagnosis")
        bus.subscribe("guidance.decision.command", "agent.generation")
        bus.subscribe("guidance.decision.command", "agent.review")

        # 发布诊断报告
        diagnosis_msg = Message(
            "learning.diagnosis.report",
            "agent.diagnosis",
            {"report_id": "rpt-001", "kp_gaps": ["KP-12", "KP-18"]},
        )
        bus.publish(diagnosis_msg)

        # 验证消息历史
        history = bus.get_history("learning.diagnosis.report")
        assert len(history) == 1

        # 验证订阅者
        subs = bus.get_subscribers("learning.diagnosis.report")
        assert len(subs) == 3

        # 验证全局频道拓扑
        assert len(bus.channels) == 7

    @pytest.mark.asyncio
    async def test_handoff_with_state_propagator(self):
        """Handoff + 状态传播器联动."""
        propagator = StatePropagator()
        handoff_mgr = HandoffManager()

        # Agent A 设置状态
        propagator.set("learner_id", "stu-001", source_agent="agent.a")
        propagator.set("step", 5, source_agent="agent.a")

        async def handler_b(ctx: HandoffContext) -> dict[str, Any]:
            # Agent B 接收状态快照
            return {
                "learner_id": ctx.state_snapshot.get("learner_id"),
                "step": ctx.state_snapshot.get("step"),
            }

        handoff_mgr.register_agent("agent.b", handler_b)

        # 执行 Handoff, 传递状态快照
        ctx = HandoffContext(
            from_agent="agent.a",
            to_agent="agent.b",
            conversation_history=[],
            state_snapshot=propagator.snapshot(),
        )

        result = await handoff_mgr.execute_handoff(ctx)
        assert result["learner_id"] == "stu-001"
        assert result["step"] == 5

    @pytest.mark.asyncio
    async def test_router_with_bus_integration(self):
        """消息路由器 + 消息总线集成."""
        bus = MessageBus()
        router = MessageRouter()

        bus.create_channel("integration.test")
        received: list[Message] = []

        # 总线订阅触发路由器注册
        def combined_callback(msg: Message) -> None:
            received.append(msg)

        bus.subscribe("integration.test", "agent.test", callback=combined_callback)
        router.register_route("integration.test", "agent.test", callback=combined_callback)

        # 通过路由器路由
        msg = Message("integration.test", "agent.pub", {"via": "router"})
        router.route(msg)
        assert len(received) == 1

        # 通过总线发布
        bus.publish(Message("integration.test", "agent.pub", {"via": "bus"}))
        assert len(received) == 2


# ============================================================
# 9. 错误处理测试
# ============================================================


class TestCommunicationErrorHandling:
    """通信错误处理与恢复测试."""

    @pytest.mark.asyncio
    async def test_callback_exception_isolation(self, message_bus):
        """单个订阅者回调异常不影响其他订阅者 (隔离性)."""
        message_bus.create_channel("error.test")
        received: list[Message] = []

        def bad_callback(msg: Message) -> None:
            raise RuntimeError("Callback error")

        message_bus.subscribe("error.test", "agent.bad", callback=bad_callback)
        message_bus.subscribe("error.test", "agent.good",
                              callback=lambda msg: received.append(msg))

        message_bus.publish(Message("error.test", "agent.pub", {"v": 1}))

        # 好的订阅者仍应收到消息
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_publish_after_close_raises(self, message_bus):
        """关闭频道后发布应抛异常."""
        message_bus.create_channel("test.channel")
        message_bus.close_channel("test.channel")

        with pytest.raises(CommunicationError):
            message_bus.publish(Message("test.channel", "agent.pub", {}))

    def test_communication_error_creation(self):
        """创建通信错误."""
        err = CommunicationError("Test error")
        assert str(err) == "Test error"

    @pytest.mark.asyncio
    async def test_subscribe_after_close_raises(self, message_bus):
        """关闭频道后订阅应抛异常."""
        message_bus.create_channel("test.channel")
        message_bus.close_channel("test.channel")

        with pytest.raises(CommunicationError):
            message_bus.subscribe("test.channel", "agent.test")

    @pytest.mark.asyncio
    async def test_handoff_handler_exception(self, handoff_manager):
        """Handoff 处理器异常应标记为 FAILED."""
        async def bad_handler(ctx: HandoffContext) -> dict[str, Any]:
            raise RuntimeError("Handler crashed")

        handoff_manager.register_agent("agent.bad", bad_handler)

        ctx = HandoffContext(
            from_agent="agent.a",
            to_agent="agent.bad",
            conversation_history=[],
            state_snapshot={},
        )

        with pytest.raises(RuntimeError):
            await handoff_manager.execute_handoff(ctx)
        assert ctx.state == HandoffState.FAILED

    @pytest.mark.asyncio
    async def test_state_reducer_type_mismatch(self, state_propagator):
        """Reducer 类型不匹配应抛异常."""
        state_propagator.set("x", 1, reducer=ReducerType.SUM)
        with pytest.raises(CommunicationError, match="reducer.*mismatch"):
            state_propagator.set("x", "not_a_number", reducer=ReducerType.SUM)
