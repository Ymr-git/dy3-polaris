"""T5 学习会话管理 — 测试套件.

遵循 TDD Red-Green-Refactor:
1. 先写测试 (RED): 每个测试描述期望行为
2. 验证测试失败 (feature missing)
3. 最小实现 (GREEN)
4. 重构 (保持绿色)

测试覆盖:
- 异常体系: JSON-RPC -32500 范围
- LearningSessionManager: 会话生命周期 (创建/获取/暂停/恢复/完成)
- CheckpointManager: 检查点创建/加载/列出
- ForkManager: Fork 创建/合并/丢弃/列表
- ContextTransfer: 会话间上下文继承
- 线程安全: 并发访问
- 边界情况: 空值/非法输入/不存在会话
- 集成测试: 全生命周期 + Fork 流程
"""

from __future__ import annotations

import threading
import time

import pytest

from dy3_polaris.l1.models import (
    ContextEnvelope,
    Interaction,
    LearningSession,
    SessionArtifact,
    SessionFork,
    SessionStatus,
    SessionType,
)
from dy3_polaris.l1.session_manager import (
    # 异常
    L1SessionError,
    SessionNotFoundError,
    SessionStateError,
    ForkError,
    CheckpointError,
    # 会话管理
    LearningSessionManager,
    # Fork 管理
    ForkManager,
    # 检查点
    CheckpointManager,
    # 上下文传递
    ContextTransfer,
    # 常量
    MAX_FORK_DEPTH,
    MAX_ACTIVE_SESSIONS_PER_USER,
    SESSION_IDLE_TIMEOUT_MS,
)


# ============================================================
# 辅助函数
# ============================================================


def make_session(
    user_id: str = "user-001",
    session_type: SessionType = SessionType.LEARNING,
) -> LearningSession:
    """创建 LearningSession 测试辅助."""
    return LearningSession(
        user_id=user_id,
        session_type=session_type,
    )


def make_envelope(user_id: str = "user-001", session_id: str = "sess-001") -> ContextEnvelope:
    """创建 ContextEnvelope 测试辅助."""
    return ContextEnvelope(user_id=user_id, session_id=session_id)


# ============================================================
# 1. 异常体系测试
# ============================================================


class TestExceptionHierarchy:
    """异常继承与 JSON-RPC 错误码测试."""

    def test_base_error_inherits_l6(self):
        """L1SessionError 继承 L6Error."""
        from dy3_polaris.l6.core.exceptions import L6Error
        assert issubclass(L1SessionError, L6Error)

    def test_base_error_jsonrpc_code(self):
        """L1SessionError JSON-RPC 码为 -32500."""
        err = L1SessionError("test")
        assert err._jsonrpc_code() == -32500

    def test_session_not_found_inherits_base(self):
        """SessionNotFoundError 继承 L1SessionError."""
        assert issubclass(SessionNotFoundError, L1SessionError)

    def test_session_not_found_jsonrpc_code(self):
        """SessionNotFoundError JSON-RPC 码为 -32501."""
        err = SessionNotFoundError("sess-xxx")
        assert err._jsonrpc_code() == -32501

    def test_session_state_error_inherits_base(self):
        """SessionStateError 继承 L1SessionError."""
        assert issubclass(SessionStateError, L1SessionError)

    def test_session_state_error_jsonrpc_code(self):
        """SessionStateError JSON-RPC 码为 -32502."""
        err = SessionStateError("sess-xxx", "invalid_transition")
        assert err._jsonrpc_code() == -32502

    def test_fork_error_inherits_base(self):
        """ForkError 继承 L1SessionError."""
        assert issubclass(ForkError, L1SessionError)

    def test_fork_error_jsonrpc_code(self):
        """ForkError JSON-RPC 码为 -32503."""
        err = ForkError("fork-xxx")
        assert err._jsonrpc_code() == -32503

    def test_checkpoint_error_inherits_base(self):
        """CheckpointError 继承 L1SessionError."""
        assert issubclass(CheckpointError, L1SessionError)

    def test_checkpoint_error_jsonrpc_code(self):
        """CheckpointError JSON-RPC 码为 -32504."""
        err = CheckpointError("sess-xxx", 0)
        assert err._jsonrpc_code() == -32504

    def test_session_not_found_contains_session_id(self):
        """SessionNotFoundError 包含 session_id 上下文."""
        err = SessionNotFoundError("sess-abc123")
        assert err.context.get("session_id") == "sess-abc123"


# ============================================================
# 2. LearningSessionManager 核心测试
# ============================================================


class TestLearningSessionManagerCreate:
    """会话创建测试."""

    def test_create_session(self):
        """创建学习会话."""
        manager = LearningSessionManager()
        session = manager.create_session(
            user_id="user-001",
            session_type=SessionType.LEARNING,
        )
        assert session.session_id.startswith("sess-")
        assert session.user_id == "user-001"
        assert session.session_type == SessionType.LEARNING
        assert session.status == SessionStatus.ACTIVE

    def test_create_diagnosis_session(self):
        """创建诊断会话."""
        manager = LearningSessionManager()
        session = manager.create_session(
            user_id="user-001",
            session_type=SessionType.DIAGNOSIS,
        )
        assert session.session_type == SessionType.DIAGNOSIS

    def test_create_lab_guide_session(self):
        """创建实验指导会话."""
        manager = LearningSessionManager()
        session = manager.create_session(
            user_id="user-001",
            session_type=SessionType.LAB_GUIDE,
        )
        assert session.session_type == SessionType.LAB_GUIDE

    def test_create_assessment_session(self):
        """创建测评会话."""
        manager = LearningSessionManager()
        session = manager.create_session(
            user_id="user-001",
            session_type=SessionType.ASSESSMENT,
        )
        assert session.session_type == SessionType.ASSESSMENT

    def test_create_session_with_context(self):
        """创建会话时加载上下文."""
        manager = LearningSessionManager()
        session = manager.create_session(
            user_id="user-001",
            session_type=SessionType.LEARNING,
        )
        assert isinstance(session.context, ContextEnvelope)
        assert session.context.session_id == session.session_id

    def test_create_session_max_limit_per_user(self):
        """单用户最大活跃会话数限制."""
        manager = LearningSessionManager()
        for i in range(MAX_ACTIVE_SESSIONS_PER_USER + 1):
            session = manager.create_session(
                user_id="user-001",
                session_type=SessionType.LEARNING,
            )
        # 超出限制时, 最早创建的应被自动清理
        active = manager.get_active_sessions("user-001")
        assert len(active) <= MAX_ACTIVE_SESSIONS_PER_USER


class TestLearningSessionManagerGet:
    """会话查询测试."""

    def test_get_session(self):
        """获取已创建会话."""
        manager = LearningSessionManager()
        created = manager.create_session(
            user_id="user-001",
            session_type=SessionType.LEARNING,
        )
        found = manager.get_session(created.session_id)
        assert found is not None
        assert found.session_id == created.session_id

    def test_get_session_not_found(self):
        """获取不存在的会话返回 None."""
        manager = LearningSessionManager()
        assert manager.get_session("sess-xxx") is None

    def test_get_active_sessions(self):
        """获取用户活跃会话列表."""
        manager = LearningSessionManager()
        s1 = manager.create_session("user-001", SessionType.LEARNING)
        s2 = manager.create_session("user-001", SessionType.DIAGNOSIS)
        manager.create_session("user-002", SessionType.LEARNING)

        active = manager.get_active_sessions("user-001")
        assert len(active) == 2
        ids = {s.session_id for s in active}
        assert s1.session_id in ids
        assert s2.session_id in ids

    def test_get_active_sessions_excludes_completed(self):
        """活跃会话列表排除已完成会话."""
        manager = LearningSessionManager()
        s1 = manager.create_session("user-001", SessionType.LEARNING)
        manager.complete_session(s1.session_id)

        active = manager.get_active_sessions("user-001")
        assert len(active) == 0


class TestLearningSessionManagerLifecycle:
    """会话生命周期测试."""

    def test_pause_session(self):
        """暂停会话."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.pause_session(session.session_id)
        assert session.status == SessionStatus.PAUSED

    def test_resume_session(self):
        """恢复暂停的会话."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.pause_session(session.session_id)
        manager.resume_session(session.session_id)
        assert session.status == SessionStatus.ACTIVE

    def test_complete_session(self):
        """完成会话."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.complete_session(session.session_id)
        assert session.status == SessionStatus.COMPLETED

    def test_pause_nonexistent_session(self):
        """暂停不存在的会话抛异常."""
        manager = LearningSessionManager()
        with pytest.raises(SessionNotFoundError):
            manager.pause_session("sess-xxx")

    def test_resume_nonexistent_session(self):
        """恢复不存在的会话抛异常."""
        manager = LearningSessionManager()
        with pytest.raises(SessionNotFoundError):
            manager.resume_session("sess-xxx")

    def test_complete_nonexistent_session(self):
        """完成不存在的会话抛异常."""
        manager = LearningSessionManager()
        with pytest.raises(SessionNotFoundError):
            manager.complete_session("sess-xxx")

    def test_pause_already_paused(self):
        """重复暂停不抛异常 (幂等)."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.pause_session(session.session_id)
        manager.pause_session(session.session_id)  # 不应抛异常
        assert session.status == SessionStatus.PAUSED

    def test_complete_already_completed(self):
        """重复完成不抛异常 (幂等)."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.complete_session(session.session_id)
        manager.complete_session(session.session_id)  # 不应抛异常
        assert session.status == SessionStatus.COMPLETED

    def test_resume_active_session(self):
        """恢复已是活跃状态的会话不抛异常 (幂等)."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.resume_session(session.session_id)
        assert session.status == SessionStatus.ACTIVE


class TestLearningSessionManagerInteraction:
    """会话交互与产出物测试."""

    def test_add_interaction(self):
        """记录交互."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        interaction = Interaction(
            interaction_type="qa",
            content="Dy3+ 的发光机制是什么?",
            response="Dy3+ 通过 4f-4f 跃迁产生特征发光...",
        )
        manager.add_interaction(session.session_id, interaction)
        assert len(session.interaction_log) == 1

    def test_add_interaction_nonexistent_session(self):
        """向不存在的会话添加交互抛异常."""
        manager = LearningSessionManager()
        interaction = Interaction(interaction_type="qa", content="test")
        with pytest.raises(SessionNotFoundError):
            manager.add_interaction("sess-xxx", interaction)

    def test_add_artifact(self):
        """关联产出物."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        artifact = SessionArtifact(
            artifact_type="knowledge_card",
            title="Dy3+ 能级跃迁",
            content="能级图与跃迁规则",
            confidence=0.92,
        )
        manager.add_artifact(session.session_id, artifact)
        assert len(session.artifacts) == 1

    def test_add_artifact_nonexistent_session(self):
        """向不存在的会话添加产出物抛异常."""
        manager = LearningSessionManager()
        artifact = SessionArtifact(artifact_type="card", title="test")
        with pytest.raises(SessionNotFoundError):
            manager.add_artifact("sess-xxx", artifact)

    def test_session_touch_updates_timestamp(self):
        """操作会话更新 updated_at."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        old_ts = session.updated_at
        time.sleep(0.01)
        manager.touch_session(session.session_id)
        assert session.updated_at > old_ts


# ============================================================
# 3. 检查点管理测试
# ============================================================


class TestCheckpointManager:
    """检查点管理测试 (设计文档 5.4)."""

    def test_create_checkpoint(self):
        """创建检查点."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        seq = manager.create_checkpoint(session.session_id)
        assert seq >= 0
        assert seq in session.checkpoint_indices

    def test_create_multiple_checkpoints(self):
        """创建多个检查点, 序列号递增."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        seq1 = manager.create_checkpoint(session.session_id)
        seq2 = manager.create_checkpoint(session.session_id)
        assert seq2 > seq1
        assert len(session.checkpoint_indices) == 2

    def test_list_checkpoints(self):
        """列出检查点."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.create_checkpoint(session.session_id)
        manager.create_checkpoint(session.session_id)
        checkpoints = manager.list_checkpoints(session.session_id)
        assert len(checkpoints) == 2

    def test_create_checkpoint_nonexistent_session(self):
        """为不存在的会话创建检查点抛异常."""
        manager = LearningSessionManager()
        with pytest.raises(SessionNotFoundError):
            manager.create_checkpoint("sess-xxx")

    def test_checkpoint_manager_direct(self):
        """CheckpointManager 独立测试."""
        cp_mgr = CheckpointManager()
        session = make_session()
        seq = cp_mgr.create_checkpoint(session)
        assert seq >= 0
        loaded = cp_mgr.load_checkpoint(session, seq)
        assert loaded.session_id == session.session_id


# ============================================================
# 4. Fork 管理测试
# ============================================================


class TestForkManagerCreate:
    """Fork 创建测试."""

    def test_fork_session(self):
        """从活跃会话创建 Fork."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.create_checkpoint(session.session_id)  # 创建检查点

        fork_session = manager.fork_session(
            session_id=session.session_id,
            fork_reason="学生手动",
            branch_label="路径A-先理论",
        )
        assert fork_session.session_id.startswith("sess-")
        assert fork_session.parent_session_id == session.session_id
        assert fork_session.status == SessionStatus.ACTIVE
        assert session.status == SessionStatus.FORKED

    def test_fork_inherits_context(self):
        """Fork 继承源会话上下文."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.create_checkpoint(session.session_id)

        fork_session = manager.fork_session(
            session_id=session.session_id,
            fork_reason="A/B测试",
            branch_label="路径B",
        )
        assert isinstance(fork_session.context, ContextEnvelope)
        assert fork_session.context.user_id == "user-001"

    def test_fork_nonexistent_session(self):
        """Fork 不存在的会话抛异常."""
        manager = LearningSessionManager()
        with pytest.raises(SessionNotFoundError):
            manager.fork_session("sess-xxx", "test", "A")

    def test_fork_completed_session(self):
        """Fork 已完成的会话抛异常."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.complete_session(session.session_id)
        with pytest.raises(SessionStateError):
            manager.fork_session(session.session_id, "test", "A")

    def test_max_fork_depth(self):
        """Fork 深度限制."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        current = session
        for i in range(MAX_FORK_DEPTH):
            manager.create_checkpoint(current.session_id)
            current = manager.fork_session(
                current.session_id,
                f"fork-{i}",
                f"branch-{i}",
            )
        # 再 Fork 一次应超出深度限制
        manager.create_checkpoint(current.session_id)
        with pytest.raises(ForkError):
            manager.fork_session(current.session_id, "too-deep", "X")


class TestForkManagerMerge:
    """Fork 合并测试."""

    def test_merge_fork(self):
        """合并 Fork 回主会话."""
        manager = LearningSessionManager()
        main = manager.create_session("user-001", SessionType.LEARNING)
        manager.create_checkpoint(main.session_id)

        fork = manager.fork_session(
            main.session_id,
            "对比测试",
            "路径B-先实验",
        )
        # Fork 分支产生一些交互
        manager.add_interaction(
            fork.session_id,
            Interaction(interaction_type="quiz", content="quiz1"),
        )

        # 合并
        manager.merge_fork(fork.session_id, main.session_id)
        assert fork.status == SessionStatus.COMPLETED

    def test_merge_nonexistent_fork(self):
        """合并不存在的 Fork 抛异常."""
        manager = LearningSessionManager()
        main = manager.create_session("user-001", SessionType.LEARNING)
        with pytest.raises(SessionNotFoundError):
            manager.merge_fork("sess-xxx", main.session_id)

    def test_merge_nonexistent_target(self):
        """合并到不存在的目标会话抛异常."""
        manager = LearningSessionManager()
        main = manager.create_session("user-001", SessionType.LEARNING)
        manager.create_checkpoint(main.session_id)
        fork = manager.fork_session(main.session_id, "test", "A")
        with pytest.raises(SessionNotFoundError):
            manager.merge_fork(fork.session_id, "sess-xxx")

    def test_discard_fork(self):
        """丢弃 Fork 分支."""
        manager = LearningSessionManager()
        main = manager.create_session("user-001", SessionType.LEARNING)
        manager.create_checkpoint(main.session_id)
        fork = manager.fork_session(main.session_id, "test", "A")

        manager.discard_fork(fork.session_id)
        assert fork.status == SessionStatus.COMPLETED

    def test_list_forks(self):
        """列出会话的所有 Fork."""
        manager = LearningSessionManager()
        main = manager.create_session("user-001", SessionType.LEARNING)
        manager.create_checkpoint(main.session_id)
        fork1 = manager.fork_session(main.session_id, "r1", "A")
        manager.create_checkpoint(main.session_id)
        fork2 = manager.fork_session(main.session_id, "r2", "B")

        forks = manager.list_forks(main.session_id)
        assert len(forks) == 2


# ============================================================
# 5. 上下文传递测试
# ============================================================


class TestContextTransfer:
    """会话间上下文传递测试 (设计文档 5.5)."""

    def test_inherit_context(self):
        """Fork 继承源会话上下文."""
        transfer = ContextTransfer()
        source = make_session()
        target = make_session()
        transfer.inherit_context(source, target)
        assert target.context.user_id == source.context.user_id

    def test_inherit_mastery_snapshot(self):
        """上下文继承包含掌握度快照."""
        from dy3_polaris.l1.models import MasterySnapshot
        transfer = ContextTransfer()
        source = make_session()
        source.context.mastery_snapshot = [
            MasterySnapshot(
                kc_id="kc-001", p_know=0.85, last_practiced_at=1_700_000_000_000
            )
        ]
        target = make_session()
        transfer.inherit_context(source, target)
        assert target.context.mastery_snapshot is not None
        assert len(target.context.mastery_snapshot) == 1
        assert target.context.mastery_snapshot[0].kc_id == "kc-001"

    def test_inherit_goals(self):
        """上下文继承包含学习目标."""
        from dy3_polaris.l1.models import LearningGoal
        transfer = ContextTransfer()
        source = make_session()
        source.context.goals = [LearningGoal(description="掌握 Dy3+ 发光机制")]
        target = make_session()
        transfer.inherit_context(source, target)
        assert len(target.context.goals) == 1


# ============================================================
# 6. 线程安全测试
# ============================================================


class TestThreadSafety:
    """线程安全测试."""

    def test_concurrent_create_session(self):
        """并发创建会话."""
        manager = LearningSessionManager()
        session_ids: list[str] = ["" for _ in range(20)]
        errors: list[Exception | None] = [None] * 20

        def create(idx: int) -> None:
            try:
                s = manager.create_session(
                    user_id=f"user-{idx}",
                    session_type=SessionType.LEARNING,
                )
                session_ids[idx] = s.session_id
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=create, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(e is None for e in errors)
        assert len(set(session_ids)) == 20

    def test_concurrent_add_interaction(self):
        """并发添加交互."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        errors: list[Exception | None] = [None] * 50

        def add(idx: int) -> None:
            try:
                manager.add_interaction(
                    session.session_id,
                    Interaction(interaction_type="qa", content=f"Q{idx}"),
                )
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=add, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(e is None for e in errors)
        assert len(session.interaction_log) == 50

    def test_concurrent_fork(self):
        """并发 Fork 同一会话."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.create_checkpoint(session.session_id)
        forks: list[LearningSession | None] = [None] * 5
        errors: list[Exception | None] = [None] * 5

        def do_fork(idx: int) -> None:
            try:
                forks[idx] = manager.fork_session(
                    session.session_id,
                    f"reason-{idx}",
                    f"branch-{idx}",
                )
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=do_fork, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(e is None for e in errors)
        # 部分成功, 部分因状态变化失败
        successful = [f for f in forks if f is not None]
        assert len(successful) >= 1


# ============================================================
# 7. 边界情况测试
# ============================================================


class TestEdgeCases:
    """边界情况测试."""

    def test_get_session_after_completion(self):
        """完成后仍可获取会话."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        manager.complete_session(session.session_id)
        found = manager.get_session(session.session_id)
        assert found is not None
        assert found.status == SessionStatus.COMPLETED

    def test_empty_interaction_log(self):
        """空交互日志."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        assert len(session.interaction_log) == 0

    def test_empty_artifact_list(self):
        """空产出物列表."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        assert len(session.artifacts) == 0

    def test_empty_checkpoint_list(self):
        """空检查点列表."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        assert len(session.checkpoint_indices) == 0
        assert manager.list_checkpoints(session.session_id) == []

    def test_list_forks_empty(self):
        """无 Fork 时返回空列表."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        assert manager.list_forks(session.session_id) == []

    def test_session_idle_timeout(self):
        """空闲超时检测."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        # 模拟长时间未操作
        session.updated_at = int(time.time() * 1000) - SESSION_IDLE_TIMEOUT_MS - 1000
        is_idle = manager.is_session_idle(session.session_id)
        assert is_idle is True

    def test_session_not_idle(self):
        """活跃会话不超时."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)
        is_idle = manager.is_session_idle(session.session_id)
        assert is_idle is False

    def test_active_sessions_empty_user(self):
        """无会话用户返回空列表."""
        manager = LearningSessionManager()
        assert manager.get_active_sessions("unknown-user") == []


# ============================================================
# 8. 集成测试: 全生命周期
# ============================================================


class TestIntegrationLifecycle:
    """全生命周期集成测试."""

    def test_full_session_lifecycle(self):
        """完整会话生命周期: 创建→交互→检查点→暂停→恢复→完成."""
        manager = LearningSessionManager()

        # 1. 创建会话
        session = manager.create_session("user-001", SessionType.LEARNING)
        assert session.status == SessionStatus.ACTIVE

        # 2. 添加交互
        manager.add_interaction(
            session.session_id,
            Interaction(interaction_type="qa", content="什么是晶体场?"),
        )

        # 3. 创建检查点
        seq = manager.create_checkpoint(session.session_id)
        assert seq >= 0

        # 4. 暂停
        manager.pause_session(session.session_id)
        assert session.status == SessionStatus.PAUSED

        # 5. 恢复
        manager.resume_session(session.session_id)
        assert session.status == SessionStatus.ACTIVE

        # 6. 添加产出物
        manager.add_artifact(
            session.session_id,
            SessionArtifact(artifact_type="card", title="晶体场理论"),
        )

        # 7. 完成
        manager.complete_session(session.session_id)
        assert session.status == SessionStatus.COMPLETED

    def test_fork_and_merge_lifecycle(self):
        """Fork 完整生命周期: 创建→Fork→分支交互→合并."""
        manager = LearningSessionManager()

        # 1. 创建主会话
        main = manager.create_session("user-001", SessionType.LEARNING)
        manager.add_interaction(
            main.session_id,
            Interaction(interaction_type="qa", content="主路径问题"),
        )

        # 2. 创建检查点
        manager.create_checkpoint(main.session_id)

        # 3. Fork
        fork = manager.fork_session(
            main.session_id,
            "学生想尝试另一种方法",
            "路径B-先实验",
        )
        assert fork.parent_session_id == main.session_id
        assert main.status == SessionStatus.FORKED

        # 4. Fork 分支产生交互
        manager.add_interaction(
            fork.session_id,
            Interaction(interaction_type="quiz", content="分支测验"),
        )

        # 5. 合并回主会话
        manager.merge_fork(fork.session_id, main.session_id)
        assert fork.status == SessionStatus.COMPLETED

        # 6. 验证 Fork 历史
        forks = manager.list_forks(main.session_id)
        assert len(forks) >= 1

    def test_fork_discard_flow(self):
        """Fork 丢弃流程: 创建→Fork→丢弃."""
        manager = LearningSessionManager()

        main = manager.create_session("user-001", SessionType.LEARNING)
        manager.create_checkpoint(main.session_id)
        fork = manager.fork_session(main.session_id, "尝试", "A")

        # 丢弃 Fork
        manager.discard_fork(fork.session_id)
        assert fork.status == SessionStatus.COMPLETED

    def test_session_with_hitl_integration(self):
        """会话与 HiTL 集成: 会话内触发紧急干预."""
        manager = LearningSessionManager()
        session = manager.create_session("user-001", SessionType.LEARNING)

        # 模拟添加多次错误交互
        for i in range(12):
            manager.add_interaction(
                session.session_id,
                Interaction(interaction_type="quiz", content=f"Q{i}", response="错误"),
            )

        # 会话应保持活跃 (紧急干预由 HiTLManager 处理)
        assert session.status == SessionStatus.ACTIVE

    def test_multiple_users_isolation(self):
        """多用户会话隔离."""
        manager = LearningSessionManager()

        s1 = manager.create_session("user-001", SessionType.LEARNING)
        s2 = manager.create_session("user-002", SessionType.LEARNING)
        s3 = manager.create_session("user-001", SessionType.DIAGNOSIS)

        u1_sessions = manager.get_active_sessions("user-001")
        u1_ids = {s.session_id for s in u1_sessions}
        assert s1.session_id in u1_ids
        assert s3.session_id in u1_ids
        assert s2.session_id not in u1_ids

        u2_sessions = manager.get_active_sessions("user-002")
        assert u2_sessions == [s2]

        # user-002 不应看到 user-001 的会话
        assert s1.session_id not in {s.session_id for s in u2_sessions}
