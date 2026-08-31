"""会话管理与 Fork 模块测试 — TDD 测试用例.

测试覆盖:
1. SessionManager — 会话生命周期管理 (创建/激活/暂停/恢复/关闭)
2. EventLog — 不可变事件溯源 (append-only + replay)
3. SessionContext — 三层上下文分离 (state + events + memory)
4. ForkCheckpoint — 四类状态快照
5. ForkEvaluator — Fork 效果评估
6. SessionCompactor — 上下文压缩与 continue-as-new
7. EnhancedSessionForkManager — 增强会话分叉管理

融合世界先进方案:
- LangGraph: Thread/Checkpoint 模式
- OpenAI Agents SDK: 极简 Session 接口
- Google ADK: Session/State/Memory 三层模型
- Claude Code: JSONL 事件树 + Fork 分支
- Temporal: 事件历史重放
"""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from dy3_polaris.l5.session_manager import (
    EventLog,
    ForkCheckpoint,
    ForkEvaluator,
    ForkMergeScope,
    SessionCompactor,
    SessionContext,
    SessionEvent,
    SessionManager,
    SessionRecord,
    SessionState,
    SessionTier,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def session_manager():
    """创建 SessionManager 实例."""
    return SessionManager()


@pytest.fixture
def event_log():
    """创建空 EventLog."""
    return EventLog(session_id="sess-test-001")


@pytest.fixture
def session_context():
    """创建 SessionContext."""
    return SessionContext(session_id="sess-test-001", agent_id="agent.test.demo")


# ============================================================
# 1. SessionState 枚举测试
# ============================================================


class TestSessionState:
    """会话状态枚举测试."""

    def test_session_states_defined(self):
        """会话状态应包含完整生命周期."""
        assert SessionState.CREATED.value == "created"
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.PAUSED.value == "paused"
        assert SessionState.CLOSED.value == "closed"
        assert SessionState.ERROR.value == "error"

    def test_session_tier_defined(self):
        """会话层级应支持三层分离."""
        assert SessionTier.STATE.value == "state"
        assert SessionTier.EVENTS.value == "events"
        assert SessionTier.MEMORY.value == "memory"


# ============================================================
# 2. SessionEvent 测试
# ============================================================


class TestSessionEvent:
    """会话事件测试."""

    def test_event_creation(self):
        """创建事件应自动生成 ID 和时间戳."""
        event = SessionEvent(
            session_id="sess-001",
            event_type="message",
            data={"text": "hello"},
        )
        assert event.event_id.startswith("evt-")
        assert event.session_id == "sess-001"
        assert event.event_type == "message"
        assert event.data == {"text": "hello"}
        assert event.timestamp > 0
        assert event.parent_event_id is None

    def test_event_with_parent(self):
        """事件可以指定父事件 (用于 Fork 树)."""
        parent = SessionEvent(
            session_id="sess-001",
            event_type="message",
            data={"text": "parent"},
        )
        child = SessionEvent(
            session_id="sess-001",
            event_type="message",
            data={"text": "child"},
            parent_event_id=parent.event_id,
        )
        assert child.parent_event_id == parent.event_id

    def test_event_immutability(self):
        """事件应是不可变的 (frozen)."""
        event = SessionEvent(
            session_id="sess-001",
            event_type="message",
            data={"text": "hello"},
        )
        with pytest.raises((AttributeError, TypeError, ValidationError)):
            event.event_type = "changed"


# ============================================================
# 3. EventLog 测试
# ============================================================


class TestEventLog:
    """不可变事件日志测试 (Temporal/Claude Code 模式)."""

    def test_append_event(self, event_log):
        """追加事件到日志."""
        event = SessionEvent(
            session_id="sess-test-001",
            event_type="message",
            data={"text": "hello"},
        )
        event_log.append(event)
        assert len(event_log) == 1

    def test_append_multiple_events(self, event_log):
        """追加多个事件."""
        for i in range(5):
            event_log.append(SessionEvent(
                session_id="sess-test-001",
                event_type="message",
                data={"index": i},
            ))
        assert len(event_log) == 5

    def test_get_events_in_order(self, event_log):
        """事件应按追加顺序返回."""
        for i in range(3):
            event_log.append(SessionEvent(
                session_id="sess-test-001",
                event_type="message",
                data={"index": i},
            ))
        events = event_log.get_events()
        assert events[0].data["index"] == 0
        assert events[1].data["index"] == 1
        assert events[2].data["index"] == 2

    def test_get_events_with_limit(self, event_log):
        """应支持限制返回数量."""
        for i in range(10):
            event_log.append(SessionEvent(
                session_id="sess-test-001",
                event_type="message",
                data={"index": i},
            ))
        recent = event_log.get_events(limit=3)
        assert len(recent) == 3
        assert recent[0].data["index"] == 7
        assert recent[2].data["index"] == 9

    def test_replay_state(self, event_log):
        """重放事件重建状态."""
        event_log.append(SessionEvent(
            session_id="sess-test-001",
            event_type="state_set",
            data={"key": "learner_id", "value": "stu-001"},
        ))
        event_log.append(SessionEvent(
            session_id="sess-test-001",
            event_type="state_set",
            data={"key": "step", "value": 5},
        ))
        event_log.append(SessionEvent(
            session_id="sess-test-001",
            event_type="state_set",
            data={"key": "path", "value": ["a", "b", "c"]},
        ))

        state = event_log.replay()
        assert state["learner_id"] == "stu-001"
        assert state["step"] == 5
        assert state["path"] == ["a", "b", "c"]

    def test_event_log_immutable_history(self, event_log):
        """事件日志历史不可变 (append-only)."""
        event = SessionEvent(
            session_id="sess-test-001",
            event_type="message",
            data={"text": "hello"},
        )
        event_log.append(event)
        # 不能删除或修改已追加的事件
        events = event_log.get_events()
        assert len(events) == 1
        # 返回的列表应是副本，修改不影响内部状态
        events.clear()
        assert len(event_log) == 1

    def test_get_event_by_id(self, event_log):
        """按 ID 获取事件."""
        event = SessionEvent(
            session_id="sess-test-001",
            event_type="message",
            data={"text": "find me"},
        )
        event_log.append(event)
        found = event_log.get_event(event.event_id)
        assert found is not None
        assert found.data["text"] == "find me"

    def test_get_nonexistent_event(self, event_log):
        """获取不存在的事件返回 None."""
        assert event_log.get_event("evt-nonexistent") is None


# ============================================================
# 4. SessionContext 测试
# ============================================================


class TestSessionContext:
    """三层上下文测试 (ADK Session/State/Memory 模式)."""

    def test_context_initial_state(self, session_context):
        """新创建的上下文应有空状态."""
        assert session_context.session_id == "sess-test-001"
        assert session_context.agent_id == "agent.test.demo"
        assert len(session_context.state) == 0
        assert len(session_context.events) == 0
        assert len(session_context.memory) == 0

    def test_set_and_get_state(self, session_context):
        """设置和获取状态."""
        session_context.set_state("learner_id", "stu-001")
        assert session_context.get_state("learner_id") == "stu-001"

    def test_get_nonexistent_state(self, session_context):
        """获取不存在的状态返回 None."""
        assert session_context.get_state("nonexistent") is None

    def test_set_state_creates_event(self, session_context):
        """设置状态应自动创建事件."""
        session_context.set_state("step", 5)
        assert len(session_context.events) == 1
        event = session_context.events.get_events()[0]
        assert event.event_type == "state_set"
        assert event.data["key"] == "step"
        assert event.data["value"] == 5

    def test_memory_persistence(self, session_context):
        """记忆层独立于状态层."""
        session_context.set_memory("last_topic", "algebra")
        assert session_context.get_memory("last_topic") == "algebra"
        # 记忆不出现在状态中
        assert session_context.get_state("last_topic") is None

    def test_memory_does_not_create_event(self, session_context):
        """记忆操作不创建事件 (事件仅追踪状态变更)."""
        session_context.set_memory("fact", "value")
        assert len(session_context.events) == 0

    def test_context_checkpoint(self, session_context):
        """上下文检查点应保存完整状态快照."""
        session_context.set_state("step", 5)
        session_context.set_state("path", ["a", "b"])

        cp_id = session_context.checkpoint()
        assert cp_id.startswith("ctx-cp-")

        # 修改状态
        session_context.set_state("step", 10)
        assert session_context.get_state("step") == 10

        # 恢复
        assert session_context.restore(cp_id) is True
        assert session_context.get_state("step") == 5
        assert session_context.get_state("path") == ["a", "b"]

    def test_context_restore_nonexistent(self, session_context):
        """恢复不存在的检查点返回 False."""
        assert session_context.restore("ctx-cp-nonexistent") is False


# ============================================================
# 5. SessionManager 测试
# ============================================================


class TestSessionManager:
    """会话生命周期管理器测试."""

    def test_create_session(self, session_manager):
        """创建会话."""
        session = session_manager.create_session(
            agent_id="agent.test.demo",
            learner_id="stu-001",
        )
        # 统一命名空间: ag- (L5 Agent 执行会话)
        assert session.session_id.startswith("ag-")
        assert session.agent_id == "agent.test.demo"
        assert session.learner_id == "stu-001"
        assert session.state == SessionState.CREATED

    def test_activate_session(self, session_manager):
        """激活会话: CREATED → ACTIVE."""
        session = session_manager.create_session(
            agent_id="agent.test.demo",
            learner_id="stu-001",
        )
        session_manager.activate(session.session_id)
        updated = session_manager.get_session(session.session_id)
        assert updated.state == SessionState.ACTIVE

    def test_pause_session(self, session_manager):
        """暂停会话: ACTIVE → PAUSED."""
        session = session_manager.create_session(
            agent_id="agent.test.demo",
            learner_id="stu-001",
        )
        session_manager.activate(session.session_id)
        session_manager.pause(session.session_id)
        assert session_manager.get_session(session.session_id).state == SessionState.PAUSED

    def test_resume_session(self, session_manager):
        """恢复会话: PAUSED → ACTIVE."""
        session = session_manager.create_session(
            agent_id="agent.test.demo",
            learner_id="stu-001",
        )
        session_manager.activate(session.session_id)
        session_manager.pause(session.session_id)
        session_manager.resume(session.session_id)
        assert session_manager.get_session(session.session_id).state == SessionState.ACTIVE

    def test_close_session(self, session_manager):
        """关闭会话: ACTIVE → CLOSED."""
        session = session_manager.create_session(
            agent_id="agent.test.demo",
            learner_id="stu-001",
        )
        session_manager.activate(session.session_id)
        session_manager.close(session.session_id)
        assert session_manager.get_session(session.session_id).state == SessionState.CLOSED

    def test_invalid_transition_raises(self, session_manager):
        """非法状态转换应抛异常."""
        session = session_manager.create_session(
            agent_id="agent.test.demo",
            learner_id="stu-001",
        )
        # CREATED → PAUSED 是非法的
        with pytest.raises(Exception):
            session_manager.pause(session.session_id)

    def test_close_already_closed(self, session_manager):
        """关闭已关闭的会话应是幂等的."""
        session = session_manager.create_session(
            agent_id="agent.test.demo",
            learner_id="stu-001",
        )
        session_manager.activate(session.session_id)
        session_manager.close(session.session_id)
        session_manager.close(session.session_id)  # 不抛异常
        assert session_manager.get_session(session.session_id).state == SessionState.CLOSED

    def test_get_nonexistent_session(self, session_manager):
        """获取不存在的会话返回 None."""
        assert session_manager.get_session("sess-nonexistent") is None

    def test_list_sessions_by_learner(self, session_manager):
        """按学习者列出会话."""
        s1 = session_manager.create_session("agent.test.demo", "stu-001")
        s2 = session_manager.create_session("agent.test.demo", "stu-001")
        s3 = session_manager.create_session("agent.test.demo", "stu-002")
        sessions = session_manager.list_sessions(learner_id="stu-001")
        assert len(sessions) == 2

    def test_list_sessions_by_agent(self, session_manager):
        """按 Agent 列出会话."""
        session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.create_session("agent.diagnosis.learner", "stu-001")
        sessions = session_manager.list_sessions(agent_id="agent.test.demo")
        assert len(sessions) == 1

    def test_list_active_sessions(self, session_manager):
        """列出活跃会话."""
        s1 = session_manager.create_session("agent.test.demo", "stu-001")
        s2 = session_manager.create_session("agent.test.demo", "stu-002")
        session_manager.activate(s1.session_id)
        session_manager.activate(s2.session_id)
        session_manager.pause(s2.session_id)
        active = session_manager.list_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == s1.session_id

    def test_session_context_access(self, session_manager):
        """会话管理器应提供上下文访问."""
        session = session_manager.create_session("agent.test.demo", "stu-001")
        ctx = session_manager.get_context(session.session_id)
        assert ctx is not None
        ctx.set_state("step", 1)
        assert ctx.get_state("step") == 1

    def test_session_timeout_cleanup(self, session_manager):
        """超时会话应被自动清理."""
        session_manager._idle_timeout_s = 0.05
        session = session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.activate(session.session_id)
        time.sleep(0.08)
        session_manager.cleanup_expired()
        updated = session_manager.get_session(session.session_id)
        assert updated.state in (SessionState.PAUSED, SessionState.CLOSED)

    def test_session_provenance(self, session_manager):
        """会话操作应记录溯源."""
        session = session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.activate(session.session_id)
        session_manager.pause(session.session_id)

        record = session_manager.get_session(session.session_id)
        assert len(record.provenance) >= 3  # create + activate + pause
        actions = [p["action"] for p in record.provenance]
        assert "session.create" in actions
        assert "session.activate" in actions
        assert "session.pause" in actions


# ============================================================
# 6. ForkCheckpoint 测试
# ============================================================


class TestForkCheckpoint:
    """Fork 检查点测试 (四类状态快照)."""

    def test_create_fork_checkpoint(self):
        """创建包含四类状态的 Fork 检查点."""
        cp = ForkCheckpoint(
            session_id="sess-001",
            kernel_state={"var1": 42, "model": {"v": 1}},
            working_session={"step": 5, "path": ["a", "b"]},
            agent_outputs={"artifact_ids": ["art-001", "art-002"]},
            broadcast_queue_state={"offset": 100, "channel": "main"},
        )
        assert cp.session_id == "sess-001"
        assert cp.kernel_state["var1"] == 42
        assert cp.working_session["step"] == 5
        assert cp.agent_outputs["artifact_ids"] == ["art-001", "art-002"]
        assert cp.broadcast_queue_state["offset"] == 100
        assert cp.checkpoint_id.startswith("fcp-")

    def test_fork_checkpoint_with_missing_state(self):
        """缺失的状态类型应默认空字典."""
        cp = ForkCheckpoint(
            session_id="sess-001",
            kernel_state={"var1": 42},
        )
        assert cp.working_session == {}
        assert cp.agent_outputs == {}
        assert cp.broadcast_queue_state == {}

    def test_fork_checkpoint_to_dict(self):
        """检查点应可序列化为字典."""
        cp = ForkCheckpoint(
            session_id="sess-001",
            kernel_state={"var1": 42},
            working_session={"step": 5},
        )
        d = cp.to_dict()
        assert d["session_id"] == "sess-001"
        assert d["kernel_state"]["var1"] == 42
        assert d["working_session"]["step"] == 5
        assert "checkpoint_id" in d
        assert "timestamp" in d


# ============================================================
# 7. ForkEvaluator 测试
# ============================================================


class TestForkEvaluator:
    """Fork 效果评估器测试."""

    def test_evaluate_single_fork(self):
        """评估单个 Fork 的效果指标."""
        evaluator = ForkEvaluator()
        result = evaluator.evaluate(
            fork_id="fork-001",
            learning_gain=0.34,
            completion_time_s=245,
            resource_tokens=12500,
        )
        assert result.fork_id == "fork-001"
        assert result.learning_gain == 0.34
        assert result.completion_time_s == 245
        assert result.resource_tokens == 12500
        assert result.score > 0

    def test_select_best_fork(self):
        """从多个 Fork 中选择最优."""
        evaluator = ForkEvaluator()
        evaluator.evaluate("fork-001", learning_gain=0.20, completion_time_s=300, resource_tokens=15000)
        evaluator.evaluate("fork-002", learning_gain=0.34, completion_time_s=245, resource_tokens=12500)
        evaluator.evaluate("fork-003", learning_gain=0.28, completion_time_s=200, resource_tokens=10000)

        best = evaluator.select_best()
        assert best is not None
        # fork-002 有最高学习增益
        assert best.fork_id == "fork-002"

    def test_evaluate_empty_returns_none(self):
        """无评估数据时返回 None."""
        evaluator = ForkEvaluator()
        assert evaluator.select_best() is None

    def test_fork_evaluation_metrics(self):
        """评估指标应包含计算的综合分数."""
        evaluator = ForkEvaluator()
        result = evaluator.evaluate(
            fork_id="fork-001",
            learning_gain=0.5,
            completion_time_s=100,
            resource_tokens=5000,
        )
        # 学习增益越高、时间越短、资源越少 → 分数越高
        assert result.score > 0
        # 综合分数应在合理范围
        assert 0 <= result.score <= 1.0


# ============================================================
# 8. SessionCompactor 测试
# ============================================================


class TestSessionCompactor:
    """会话压缩器测试 (Claude Code compaction + Temporal continue-as-new)."""

    def test_compact_events(self):
        """压缩事件历史."""
        ctx = SessionContext("sess-001", "agent.test.demo")
        for i in range(20):
            ctx.set_state(f"key_{i}", i)

        compactor = SessionCompactor(max_events=10)
        compacted = compactor.compact(ctx)

        assert compacted is True
        assert len(ctx.events) <= 10

    def test_compact_preserves_recent_events(self):
        """压缩应保留最近的事件."""
        ctx = SessionContext("sess-001", "agent.test.demo")
        for i in range(20):
            ctx.set_state(f"key_{i}", i)

        compactor = SessionCompactor(max_events=5)
        compactor.compact(ctx)

        events = ctx.events.get_events()
        # 最近 5 个事件应保留
        assert len(events) == 5
        assert events[-1].data["key"] == "key_19"

    def test_compact_merges_state(self):
        """压缩应将旧事件的状态合并到当前状态."""
        ctx = SessionContext("sess-001", "agent.test.demo")
        ctx.set_state("a", 1)
        ctx.set_state("b", 2)
        ctx.set_state("c", 3)

        compactor = SessionCompactor(max_events=1)
        compactor.compact(ctx)

        # 状态应保留 (即使事件被压缩)
        assert ctx.get_state("a") == 1
        assert ctx.get_state("b") == 2
        assert ctx.get_state("c") == 3

    def test_compact_not_needed(self):
        """事件数不足时不需要压缩."""
        ctx = SessionContext("sess-001", "agent.test.demo")
        ctx.set_state("a", 1)

        compactor = SessionCompactor(max_events=10)
        compacted = compactor.compact(ctx)
        assert compacted is False

    def test_continue_as_new(self):
        """continue-as-new 应创建新的干净上下文，保留核心状态."""
        ctx = SessionContext("sess-001", "agent.test.demo")
        ctx.set_state("step", 10)
        ctx.set_state("learner_id", "stu-001")
        ctx.set_state("temp_data", "large_temporary_data")
        ctx.set_memory("last_topic", "algebra")

        compactor = SessionCompactor()
        new_ctx = compactor.continue_as_new(
            ctx,
            preserve_keys=["learner_id", "step"],
        )

        assert new_ctx.session_id != ctx.session_id
        assert new_ctx.get_state("learner_id") == "stu-001"
        assert new_ctx.get_state("step") == 10
        assert new_ctx.get_state("temp_data") is None
        assert len(new_ctx.events) == 0  # 新会话事件清空
        # 记忆应保留
        assert new_ctx.get_memory("last_topic") == "algebra"


# ============================================================
# 9. SessionManager + Fork 集成测试
# ============================================================


class TestSessionForkIntegration:
    """会话管理与 Fork 集成测试."""

    def test_fork_from_session(self, session_manager):
        """从活跃会话创建 Fork."""
        session = session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.activate(session.session_id)
        ctx = session_manager.get_context(session.session_id)
        ctx.set_state("model", "v1")

        fork_cp = session_manager.create_fork(
            session_id=session.session_id,
            trigger_type="ab_test",
            initiator="agent.guidance.decision",
            reason="A/B 测试两种教学策略",
        )
        assert fork_cp is not None
        assert fork_cp.session_id == session.session_id

    def test_fork_channel_isolation(self, session_manager):
        """Fork 应使用独立的频道前缀."""
        session = session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.activate(session.session_id)

        fork_record = session_manager.create_fork(
            session_id=session.session_id,
            trigger_type="debate_branch",
            initiator="agent.guidance.decision",
            reason="辩论模式",
        )
        assert fork_record.channel_prefix.startswith("fork.")

    def test_fork_merge_back(self, session_manager):
        """Fork 合并回主会话."""
        session = session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.activate(session.session_id)
        ctx = session_manager.get_context(session.session_id)
        ctx.set_state("model", "v1")

        # 创建 Fork
        fork_record = session_manager.create_fork(
            session_id=session.session_id,
            trigger_type="ab_test",
            initiator="agent.guidance.decision",
            reason="策略对比",
        )

        # 模拟 Fork 中产生更好的结果
        session_manager.record_fork_evaluation(
            fork_id=fork_record.fork_id,
            learning_gain=0.35,
            completion_time_s=200,
            resource_tokens=8000,
        )

        # 合并
        merged = session_manager.merge_fork(
            fork_id=fork_record.fork_id,
            target_session_id=session.session_id,
            merge_scope=[ForkMergeScope.KERNEL_STATE, ForkMergeScope.AGENT_OUTPUTS],
        )
        assert merged is True

    def test_fork_provenance_chain(self, session_manager):
        """Fork 操作应记录完整溯源链."""
        session = session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.activate(session.session_id)

        fork_record = session_manager.create_fork(
            session_id=session.session_id,
            trigger_type="debate_branch",
            initiator="agent.guidance.decision",
            reason="辩论分支探索",
        )

        session_manager.record_fork_evaluation(
            fork_id=fork_record.fork_id,
            learning_gain=0.3,
            completion_time_s=250,
            resource_tokens=10000,
        )

        session_manager.merge_fork(
            fork_id=fork_record.fork_id,
            target_session_id=session.session_id,
            merge_scope=[ForkMergeScope.KERNEL_STATE],
        )

        record = session_manager.get_session(session.session_id)
        provenance_actions = [p["action"] for p in record.provenance]
        assert "fork.create" in provenance_actions
        assert "fork.evaluate" in provenance_actions
        assert "fork.merge" in provenance_actions

    def test_fork_tree_depth_limit(self, session_manager):
        """Fork 树深度不应超过 3 层."""
        session = session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.activate(session.session_id)

        # 第 1 层 Fork
        fork1 = session_manager.create_fork(
            session_id=session.session_id,
            trigger_type="ab_test",
            initiator="agent.test",
            reason="level 1",
        )
        assert fork1.depth == 1

        # 第 2 层 Fork (从 fork1 的会话再 fork)
        fork2 = session_manager.create_fork(
            session_id=fork1.fork_id,
            trigger_type="ab_test",
            initiator="agent.test",
            reason="level 2",
        )
        assert fork2.depth == 2

        # 第 3 层 Fork
        fork3 = session_manager.create_fork(
            session_id=fork2.fork_id,
            trigger_type="ab_test",
            initiator="agent.test",
            reason="level 3",
        )
        assert fork3.depth == 3

        # 第 4 层应失败
        with pytest.raises(Exception):
            session_manager.create_fork(
                session_id=fork3.fork_id,
                trigger_type="ab_test",
                initiator="agent.test",
                reason="level 4",
            )

    def test_fork_concurrency_limit(self, session_manager):
        """每个父会话最多 5 个并发 Fork."""
        session = session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.activate(session.session_id)

        for i in range(5):
            session_manager.create_fork(
                session_id=session.session_id,
                trigger_type="ab_test",
                initiator="agent.test",
                reason=f"fork {i}",
            )

        # 第 6 个应失败
        with pytest.raises(Exception):
            session_manager.create_fork(
                session_id=session.session_id,
                trigger_type="ab_test",
                initiator="agent.test",
                reason="fork 6",
            )

    def test_list_forks_by_session(self, session_manager):
        """按会话列出 Fork."""
        session = session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.activate(session.session_id)

        for i in range(3):
            session_manager.create_fork(
                session_id=session.session_id,
                trigger_type="ab_test",
                initiator="agent.test",
                reason=f"fork {i}",
            )

        forks = session_manager.list_forks(session.session_id)
        assert len(forks) == 3

    def test_fork_timeout_cleanup(self, session_manager):
        """超时 Fork 应被自动清理."""
        session_manager._fork_timeout_s = 0.05
        session = session_manager.create_session("agent.test.demo", "stu-001")
        session_manager.activate(session.session_id)

        fork = session_manager.create_fork(
            session_id=session.session_id,
            trigger_type="ab_test",
            initiator="agent.test",
            reason="timeout test",
            timeout_seconds=0.05,
        )

        time.sleep(0.08)
        cleaned = session_manager.cleanup_expired_forks()
        assert fork.fork_id in cleaned
