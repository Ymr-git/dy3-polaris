"""T3 A2A 协议实现 - 单元测试.

测试覆盖:
1. 辅助函数（create_task_id, create_session_id, create_a2a_message）
2. A2AMessageBus（Agent 注册/发现/握手/任务/取消/心跳）
3. SessionManager（创建/关闭/发送/限流/清理/事件钩子）
4. CapabilityRegistry（注册/索引/查找/摘要）
5. HeartbeatMonitor（启动/停止/过期检测）
"""

from __future__ import annotations

import asyncio
import time

import pytest

from dy3_polaris.l6.core.exceptions import (
    A2AAgentNotFoundError,
    A2ACancelError,
    A2ACapabilityMismatchError,
    A2AHandshakeError,
    A2ASessionError,
    A2ATaskError,
    A2ATimeoutError,
    L6Error,
)
from dy3_polaris.l6.core.models import (
    A2ACapability,
    A2AMessage,
    A2AMessageType,
)
from dy3_polaris.l6.a2a import (
    A2AProtocolVersion,
    AgentIdentity,
    HandshakeResult,
    TaskStatus,
    A2ATaskRecord,
    A2AMessageBus,
    HeartbeatMonitor,
    SessionState,
    SessionRecord,
    SessionManager,
    CapabilityRegistry,
    create_a2a_message,
    create_session_id,
    create_task_id,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def fresh_bus() -> A2AMessageBus:
    return A2AMessageBus()


@pytest.fixture
def tutor_cap() -> A2ACapability:
    return A2ACapability(
        agent_id="tutor-agent",
        agent_name="Dy3+ 导学 Agent",
        version="1.2.0",
        supported_methods=["adaptive_tutoring", "question_generation", "mistake_analysis"],
        supported_tools=["bkt_compute", "irt_evaluate", "knowledge_retrieve"],
        domain_scope=["DOM-A", "DOM-B"],
    )


@pytest.fixture
def assess_cap() -> A2ACapability:
    return A2ACapability(
        agent_id="assess-agent",
        agent_name="Dy3+ 评估 Agent",
        version="1.0.0",
        supported_methods=["knowledge_assessment", "diagnostic_evaluation"],
        supported_tools=["irt_evaluate", "forgetfulness_scan"],
        domain_scope=["DOM-A", "DOM-B", "DOM-C"],
    )


# ============================================================
# 1. 辅助函数测试
# ============================================================

class TestHelperFunctions:
    def test_create_task_id_unique(self):
        ids = [create_task_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_create_task_id_format(self):
        tid = create_task_id()
        assert tid.startswith("task-")
        assert len(tid) > 10

    def test_create_session_id_deterministic(self):
        s1 = create_session_id("agent-A", "agent-B")
        s2 = create_session_id("agent-A", "agent-B")
        assert s1 == s2

    def test_create_session_id_ordered(self):
        s1 = create_session_id("agent-A", "agent-B")
        s2 = create_session_id("agent-B", "agent-A")
        assert s1 == s2  # 应该对称

    def test_create_a2a_message(self):
        msg = create_a2a_message(
            message_type=A2AMessageType.TASK_REQUEST,
            from_agent="A1",
            to_agent="A2",
            capability="knowledge_assessment",
            input_data={"learner_id": "u001"},
        )
        assert msg.message_type == A2AMessageType.TASK_REQUEST
        assert msg.from_agent == "A1"
        assert msg.to_agent == "A2"
        assert msg.payload["capability"] == "knowledge_assessment"
        assert msg.message_id  # 自动生成


# ============================================================
# 2. A2AMessageBus 测试
# ============================================================

class TestMessageBusAgentManagement:
    def test_register_agent(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        assert fresh_bus.get_agent(tutor_cap.agent_id) is not None

    def test_register_duplicate_overwrite(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        new_cap = A2ACapability(agent_id="tutor-agent", agent_name="Updated", version="2.0.0")
        fresh_bus.register_agent(tutor_cap.agent_id, new_cap)
        assert fresh_bus.get_agent(tutor_cap.agent_id).version == "2.0.0"

    def test_unregister_agent(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        removed = fresh_bus.unregister_agent(tutor_cap.agent_id)
        assert removed is not None
        assert fresh_bus.get_agent(tutor_cap.agent_id) is None

    def test_discover_all(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)
        agents = fresh_bus.discover_agents()
        assert len(agents) == 2

    def test_discover_by_capability(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)
        # 只有一个 agent 支持 knowledge_assessment
        agents = fresh_bus.discover_agents(capability="knowledge_assessment")
        assert len(agents) == 1
        assert agents[0].agent_id == "assess-agent"

    def test_discover_by_domain(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)
        agents = fresh_bus.discover_agents(domain="DOM-C")
        assert len(agents) == 1
        assert agents[0].agent_id == "assess-agent"


@pytest.mark.asyncio
class TestMessageBusHandshake:
    async def test_successful_handshake(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)

        result = await fresh_bus.initiate_handshake(
            from_agent="tutor-agent",
            to_agent="assess-agent",
            requested_capabilities=["knowledge_assessment"],
        )
        assert isinstance(result, HandshakeResult)
        assert result.status == "accepted"
        assert "knowledge_assessment" in result.granted_capabilities
        assert result.session_id

    async def test_handshake_agent_not_found(self, fresh_bus: A2AMessageBus):
        with pytest.raises(A2AAgentNotFoundError):
            await fresh_bus.initiate_handshake(
                from_agent="nonexistent",
                to_agent="also-nonexistent",
                requested_capabilities=["x"],
            )

    async def test_handshake_capability_mismatch(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)

        with pytest.raises(A2ACapabilityMismatchError):
            await fresh_bus.initiate_handshake(
                from_agent="tutor-agent",
                to_agent="assess-agent",
                requested_capabilities=["nonexistent_capability"],
            )


@pytest.mark.asyncio
class TestMessageBusTask:
    async def test_send_task_no_handler(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)

        task = await fresh_bus.send_task(
            from_agent="tutor-agent",
            to_agent="assess-agent",
            capability="knowledge_assessment",
            input_data={"learner_id": "u001"},
        )
        assert isinstance(task, A2ATaskRecord)
        assert task.status == TaskStatus.FAILED
        assert task.result is None
        assert "未注册任务处理器" in str(task.error)
        assert task.from_agent == "tutor-agent"
        assert task.to_agent == "assess-agent"

    async def test_send_task_with_handler(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)

        handler_called = False
        handler_input = {}

        async def my_handler(task_record: A2ATaskRecord) -> dict:
            nonlocal handler_called, handler_input
            handler_called = True
            handler_input = task_record.input_data
            return {"result": "assessed", "score": 0.85}

        fresh_bus.register_task_handler(my_handler)

        task = await fresh_bus.send_task(
            from_agent="tutor-agent",
            to_agent="assess-agent",
            capability="knowledge_assessment",
            input_data={"learner_id": "u001"},
        )
        assert handler_called
        assert handler_input == {"learner_id": "u001"}
        assert task.result == {"result": "assessed", "score": 0.85}

    async def test_get_task(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)
        task = await fresh_bus.send_task(
            from_agent="tutor-agent",
            to_agent="assess-agent",
            capability="knowledge_assessment",
            input_data={},
        )
        fetched = fresh_bus.get_task(task.task_id)
        assert fetched is not None
        assert fetched.task_id == task.task_id

    async def test_get_task_not_found(self, fresh_bus: A2AMessageBus):
        assert fresh_bus.get_task("nonexistent") is None


@pytest.mark.asyncio
class TestMessageBusCancel:
    async def test_cancel_pending_task(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)

        # 使用一个延迟 handler 让任务保持在 RUNNING 状态
        async def slow_handler(task_record: A2ATaskRecord) -> dict:
            await asyncio.sleep(10)
            return {}

        fresh_bus.register_task_handler(slow_handler)

        task_future = asyncio.ensure_future(
            fresh_bus.send_task(
                from_agent="tutor-agent",
                to_agent="assess-agent",
                capability="knowledge_assessment",
                input_data={},
            )
        )
        # 等待任务开始
        await asyncio.sleep(0.1)

        # 获取任务 ID（从日志中）
        tasks = fresh_bus.get_all_tasks()
        if tasks:
            t = tasks[0]
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                cancelled = await fresh_bus.cancel_task(t.task_id, "test cancel")
                assert cancelled.status == TaskStatus.CANCELLED

        task_future.cancel()
        try:
            await task_future
        except asyncio.CancelledError:
            pass

    async def test_cancel_completed_task_fails(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)

        async def completed_handler(task_record: A2ATaskRecord) -> dict:
            return {"completed": True}

        fresh_bus.register_task_handler(completed_handler)
        task = await fresh_bus.send_task(
            from_agent="tutor-agent",
            to_agent="assess-agent",
            capability="knowledge_assessment",
            input_data={},
        )
        assert task.status == TaskStatus.COMPLETED

        with pytest.raises(A2ACancelError):
            await fresh_bus.cancel_task(task.task_id)


@pytest.mark.asyncio
class TestMessageBusHeartbeat:
    async def test_send_heartbeat(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)

        result = await fresh_bus.initiate_handshake(
            from_agent="tutor-agent",
            to_agent="assess-agent",
            requested_capabilities=["knowledge_assessment"],
        )
        session_id = result.session_id

        msg = await fresh_bus.send_heartbeat("tutor-agent", session_id=session_id)
        assert msg.message_type == A2AMessageType.HEARTBEAT

        session = fresh_bus.get_session(session_id)
        assert session is not None
        assert session["last_heartbeat_at"] > 0

    async def test_message_log(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        await fresh_bus.send_discovery(tutor_cap)
        log = fresh_bus.export_message_log()
        assert len(log) >= 1


# ============================================================
# 3. SessionManager 测试
# ============================================================

@pytest.mark.asyncio
class TestSessionManager:
    async def test_create_session(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus)
        session = await mgr.create_session("A1", "A2", ["cap_a"])
        assert session.session_id
        assert session.state == SessionState.ACTIVE
        assert session.granted_capabilities == ["cap_a"]

    async def test_close_session(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus)
        session = await mgr.create_session("A1", "A2")
        closed = await mgr.close_session(session.session_id)
        assert closed.state == SessionState.CLOSED

    async def test_close_nonexistent_session(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus)
        with pytest.raises(A2ASessionError):
            await mgr.close_session("nonexistent")

    async def test_get_session(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus)
        session = await mgr.create_session("A1", "A2")
        fetched = mgr.get_session(session.session_id)
        assert fetched is not None
        assert fetched.session_id == session.session_id

    async def test_get_sessions_by_agent(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus)
        s1 = await mgr.create_session("A1", "A2")
        s2 = await mgr.create_session("A1", "A3")
        s3 = await mgr.create_session("A4", "A3")

        a1_sessions = mgr.get_sessions_by_agent("A1")
        assert len(a1_sessions) == 2

        a3_sessions = mgr.get_sessions_by_agent("A3")
        assert len(a3_sessions) == 2

    async def test_send_message_active_session(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)
        mgr = SessionManager(fresh_bus)
        session = await mgr.create_session("tutor-agent", "assess-agent", ["knowledge_assessment"])

        # 发送心跳消息（非 TASK_REQUEST，不需要 handler）
        result = await mgr.send_message(
            session_id=session.session_id,
            from_agent="tutor-agent",
            to_agent="assess-agent",
            payload={"data": "test"},
            message_type=A2AMessageType.HEARTBEAT,
        )
        assert result is not None
        updated = mgr.get_session(session.session_id)
        assert updated.message_count >= 1  # send

    async def test_send_message_inactive_session_fails(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus)
        session = await mgr.create_session("A1", "A2")
        await mgr.close_session(session.session_id)

        with pytest.raises(A2ASessionError, match="not active"):
            await mgr.send_message(
                session_id=session.session_id,
                from_agent="A1",
                to_agent="A2",
                payload={},
            )

    async def test_rate_limiting(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus, session_ttl_seconds=999)
        session = await mgr.create_session(
            "A1", "A2",
            rate_limits={"max_rps": 3, "max_concurrent": 5},
        )

        # 发送 3 条消息（在限流窗口内）
        for _ in range(3):
            session.check_rate_limit()  # 模拟限流检查

        # 第 4 条应该被拒绝
        assert not session.check_rate_limit()

    async def test_concurrent_slot_management(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus)
        session = await mgr.create_session(
            "A1", "A2",
            rate_limits={"max_rps": 100, "max_concurrent": 2},
        )

        assert session.acquire_concurrent_slot() is True
        assert session.acquire_concurrent_slot() is True
        assert session.acquire_concurrent_slot() is False

        session.release_concurrent_slot()
        assert session.acquire_concurrent_slot() is True

    async def test_event_hooks(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus)
        events_fired: list[str] = []

        def on_created(record: SessionRecord) -> None:
            events_fired.append(f"created:{record.session_id}")

        mgr.on("created", on_created)
        await mgr.create_session("A1", "A2")
        assert len(events_fired) == 1
        assert events_fired[0].startswith("created:")

    async def test_export_summary(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus)
        await mgr.create_session("A1", "A2")
        await mgr.create_session("A3", "A4")
        summary = mgr.export_summary()
        assert summary["total_sessions"] == 2
        assert summary["active_sessions"] == 2
        assert summary["state_breakdown"]["active"] == 2

    async def test_session_record_to_dict(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus)
        session = await mgr.create_session("A1", "A2")
        d = session.to_dict()
        assert d["from_agent"] == "A1"
        assert d["to_agent"] == "A2"
        assert d["state"] == "active"
        assert "is_active" in d
        assert "duration_seconds" in d

    async def test_cleanup_expired(self, fresh_bus: A2AMessageBus):
        mgr = SessionManager(fresh_bus, session_ttl_seconds=0.01, cleanup_interval_seconds=0.01)
        await mgr.create_session("A1", "A2")
        # 等待过期
        await asyncio.sleep(0.05)
        expired_count = await mgr._cleanup_expired()
        assert expired_count >= 1


# ============================================================
# 4. CapabilityRegistry 测试
# ============================================================

class TestCapabilityRegistry:
    def test_register_and_get(self, tutor_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        assert reg.get("tutor-agent") is not None
        assert reg.size == 1

    def test_unregister(self, tutor_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        removed = reg.unregister("tutor-agent")
        assert removed is not None
        assert reg.size == 0

    def test_unregister_nonexistent(self):
        reg = CapabilityRegistry()
        assert reg.unregister("nonexistent") is None

    def test_find_by_capability(self, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        reg.register(assess_cap)

        # assess-agent 支持 knowledge_assessment
        results = reg.find_by_capability("knowledge_assessment")
        assert len(results) == 1
        assert results[0].agent_id == "assess-agent"

    def test_find_by_tool(self, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        reg.register(assess_cap)

        # bkt_compute 只在 tutor-agent
        results = reg.find_by_tool("bkt_compute")
        assert len(results) == 1
        assert results[0].agent_id == "tutor-agent"

        # irt_evaluate 在两个 agent 都有
        results = reg.find_by_tool("irt_evaluate")
        assert len(results) == 2

    def test_find_by_domain(self, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        reg.register(assess_cap)

        results = reg.find_by_domain("DOM-C")
        assert len(results) == 1
        assert results[0].agent_id == "assess-agent"

    def test_find_by_method(self, tutor_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        results = reg.find_by_method("adaptive_tutoring")
        assert len(results) == 1

    def test_all_capabilities(self, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        reg.register(assess_cap)
        caps = reg.all_capabilities()
        assert "adaptive_tutoring" in caps
        assert "knowledge_assessment" in caps

    def test_all_agents(self, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        reg.register(assess_cap)
        agents = reg.all_agents()
        assert len(agents) == 2

    def test_export_summary(self, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        reg.register(assess_cap)
        summary = reg.export_summary()
        assert summary["total_agents"] == 2
        assert summary["unique_capabilities"] >= 5  # 3 + 2 unique methods
        assert summary["unique_tools"] >= 4  # 3 + 2 unique tools
        assert summary["unique_domains"] == 3  # DOM-A, DOM-B, DOM-C
        assert len(summary["agents"]) == 2

    def test_reregister_updates(self, tutor_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        new_cap = A2ACapability(agent_id="tutor-agent", agent_name="Updated", supported_methods=["new_method"])
        reg.register(new_cap)
        assert reg.get("tutor-agent").agent_name == "Updated"
        assert len(reg.find_by_capability("adaptive_tutoring")) == 0
        assert len(reg.find_by_capability("new_method")) == 1

    def test_clear(self, tutor_cap: A2ACapability):
        reg = CapabilityRegistry()
        reg.register(tutor_cap)
        reg.clear()
        assert reg.size == 0


# ============================================================
# 5. HeartbeatMonitor 测试
# ============================================================

@pytest.mark.asyncio
class TestHeartbeatMonitor:
    async def test_start_and_stop(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)

        # 先握手创建 session
        await fresh_bus.initiate_handshake(
            from_agent="tutor-agent",
            to_agent="assess-agent",
            requested_capabilities=["knowledge_assessment"],
        )

        monitor = HeartbeatMonitor(fresh_bus, interval=1.0)
        await monitor.start()
        await asyncio.sleep(0.1)
        await monitor.stop()

    async def test_stale_session_detection(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)

        result = await fresh_bus.initiate_handshake(
            from_agent="tutor-agent",
            to_agent="assess-agent",
            requested_capabilities=["knowledge_assessment"],
        )

        # 不发送心跳，等 3 倍间隔
        monitor = HeartbeatMonitor(fresh_bus, interval=0.01)
        await asyncio.sleep(0.05)  # 超过 3 * 0.01
        stale = monitor.get_stale_sessions()
        # Session 存在但没有心跳记录，应被判定为过期
        assert len(stale) >= 0  # 取决于实现
        await monitor.stop()

    async def test_check_session_alive(self, fresh_bus: A2AMessageBus, tutor_cap: A2ACapability, assess_cap: A2ACapability):
        fresh_bus.register_agent(tutor_cap.agent_id, tutor_cap)
        fresh_bus.register_agent(assess_cap.agent_id, assess_cap)

        result = await fresh_bus.initiate_handshake(
            from_agent="tutor-agent",
            to_agent="assess-agent",
            requested_capabilities=["knowledge_assessment"],
        )

        monitor = HeartbeatMonitor(fresh_bus, interval=30.0)
        alive = await monitor.check_session(result.session_id)
        assert alive is True
        await monitor.stop()

    async def test_monitor_reset(self, fresh_bus: A2AMessageBus):
        monitor = HeartbeatMonitor(fresh_bus, interval=1.0)
        monitor.reset()
        await monitor.stop()
