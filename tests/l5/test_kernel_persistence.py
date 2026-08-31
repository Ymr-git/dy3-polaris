"""持久化内核与状态管理测试.

TDD 测试用例 — 覆盖:
1. PersistentKernel — 状态机、变量保留、执行、checkpoint
2. KernelManager — 生命周期调度、空闲超时、错误恢复
3. CheckpointStore — 存储/加载/清理
4. SessionForkManager — Fork创建、合并、清理、树约束
5. StatePersistence — 协调持久化、自动checkpoint

融合世界先进方案:
- Claude Science: Persistent Kernels + Session Fork
- LangGraph: Checkpoint 机制 + 状态恢复
- Jupyter: Kernel 进程模型 + 变量空间
- Temporal: Activity 状态机 + 重试恢复
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from dy3_polaris.l5.kernel_persistence import (
    CheckpointStore,
    ForkConfig,
    ForkRecord,
    ForkStatus,
    KernelError,
    KernelManager,
    KernelState,
    KernelStateError,
    MaxForkDepthError,
    MaxForkConcurrencyError,
    MemoryCheckpointStore,
    PersistentKernel,
    RecoveryExceededError,
    SessionForkManager,
    StatePersistence,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def checkpoint_store():
    return MemoryCheckpointStore()


@pytest.fixture
def python_kernel(checkpoint_store):
    return PersistentKernel(
        kernel_type="python",
        instance_id="inst-test-001",
        checkpoint_store=checkpoint_store,
    )


@pytest.fixture
def kernel_manager(checkpoint_store):
    return KernelManager(
        checkpoint_store=checkpoint_store,
        idle_timeout_s=0.1,
        sleep_timeout_s=0.2,
        auto_checkpoint_interval_s=0.05,
    )


@pytest.fixture
def fork_manager(checkpoint_store):
    return SessionForkManager(checkpoint_store=checkpoint_store)


# ============================================================
# 1. PersistentKernel 测试
# ============================================================

class TestPersistentKernel:
    """持久化内核单元测试."""

    def test_kernel_initial_state(self, python_kernel):
        """新创建的内核应为 INITIALIZING 状态."""
        assert python_kernel.state == KernelState.INITIALIZING
        assert python_kernel.kernel_type == "python"
        assert python_kernel.kernel_id.startswith("kernel-")

    def test_kernel_warmup_transition(self, python_kernel):
        """预热后状态变为 WARMING_UP."""
        python_kernel.warmup()
        assert python_kernel.state == KernelState.WARMING_UP

    def test_kernel_activate_transition(self, python_kernel):
        """激活后状态变为 ACTIVE."""
        python_kernel.warmup()
        python_kernel.activate()
        assert python_kernel.state == KernelState.ACTIVE

    def test_kernel_idle_transition(self, python_kernel):
        """空闲后状态变为 IDLE."""
        python_kernel.warmup()
        python_kernel.activate()
        python_kernel.to_idle()
        assert python_kernel.state == KernelState.IDLE

    def test_kernel_sleep_transition(self, python_kernel):
        """长时间空闲后状态变为 SLEEPING."""
        python_kernel.warmup()
        python_kernel.activate()
        python_kernel.to_idle()
        python_kernel.to_sleeping()
        assert python_kernel.state == KernelState.SLEEPING

    def test_kernel_destroy_transition(self, python_kernel):
        """销毁后状态变为 DESTROYED."""
        python_kernel.warmup()
        python_kernel.activate()
        python_kernel.destroy()
        assert python_kernel.state == KernelState.DESTROYED

    def test_kernel_error_transition(self, python_kernel):
        """执行错误后状态变为 ERROR."""
        python_kernel.warmup()
        python_kernel.activate()
        python_kernel.to_error("test error")
        assert python_kernel.state == KernelState.ERROR
        assert python_kernel.error_message == "test error"

    def test_invalid_state_transition_raises(self, python_kernel):
        """非法状态转换应报错."""
        with pytest.raises(KernelStateError):
            python_kernel.activate()  # INITIALIZING 不能直接到 ACTIVE

    def test_kernel_variable_persistence(self, python_kernel):
        """变量应跨执行保留."""
        python_kernel.warmup()
        python_kernel.activate()
        python_kernel.set_variable("bkt_params", {"p_init": 0.3, "p_learn": 0.1})
        assert python_kernel.get_variable("bkt_params") == {"p_init": 0.3, "p_learn": 0.1}

    def test_kernel_execute_code(self, python_kernel):
        """执行代码应返回结果并保留变量."""
        python_kernel.warmup()
        python_kernel.activate()
        result = python_kernel.execute("x = 42; y = x * 2")
        assert result["success"] is True
        assert python_kernel.get_variable("x") == 42
        assert python_kernel.get_variable("y") == 84

    def test_kernel_execute_error(self, python_kernel):
        """执行错误代码应返回错误信息."""
        python_kernel.warmup()
        python_kernel.activate()
        result = python_kernel.execute("1/0")
        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]

    def test_kernel_checkpoint_and_restore(self, python_kernel):
        """checkpoint 后应能恢复状态."""
        python_kernel.warmup()
        python_kernel.activate()
        python_kernel.set_variable("model", {"version": "v1"})

        cp_id = python_kernel.checkpoint()
        assert cp_id is not None

        python_kernel.set_variable("model", {"version": "v2"})
        assert python_kernel.get_variable("model") == {"version": "v2"}

        restored = python_kernel.restore_from_checkpoint(cp_id)
        assert restored is True
        assert python_kernel.get_variable("model") == {"version": "v1"}

    def test_kernel_checkpoint_store_integration(self, python_kernel, checkpoint_store):
        """checkpoint 应保存到存储后端."""
        python_kernel.warmup()
        python_kernel.activate()
        python_kernel.set_variable("data", [1, 2, 3])

        cp_id = python_kernel.checkpoint()
        stored = checkpoint_store.load(python_kernel.kernel_id, cp_id)
        assert stored is not None
        assert stored["variables"]["data"] == [1, 2, 3]

    def test_kernel_memory_usage_tracking(self, python_kernel):
        """内存使用应被追踪."""
        python_kernel.warmup()
        python_kernel.activate()
        python_kernel.set_variable("large_array", list(range(10000)))
        usage = python_kernel.memory_usage_mb()
        assert usage > 0

    def test_kernel_destroy_clears_variables(self, python_kernel):
        """销毁后变量空间应被清空."""
        python_kernel.warmup()
        python_kernel.activate()
        python_kernel.set_variable("key", "value")
        python_kernel.destroy()
        assert python_kernel.get_variable("key") is None


# ============================================================
# 2. KernelManager 测试
# ============================================================

class TestKernelManager:
    """内核管理器测试."""

    def test_create_kernel(self, kernel_manager):
        """创建内核后应处于 WARMING_UP 状态."""
        kernel = kernel_manager.create_kernel("python", "inst-001")
        assert kernel.state == KernelState.WARMING_UP
        assert kernel_manager.get_kernel(kernel.kernel_id) is kernel

    def test_get_nonexistent_kernel(self, kernel_manager):
        """获取不存在的内核应返回 None."""
        assert kernel_manager.get_kernel("nonexistent") is None

    def test_destroy_kernel(self, kernel_manager):
        """销毁内核后应从管理器移除."""
        kernel = kernel_manager.create_kernel("python", "inst-001")
        kid = kernel.kernel_id
        kernel_manager.destroy_kernel(kid)
        assert kernel_manager.get_kernel(kid) is None
        assert kernel.state == KernelState.DESTROYED

    def test_idle_timeout_transitions_to_idle(self, kernel_manager):
        """空闲超时应自动切换到 IDLE."""
        kernel = kernel_manager.create_kernel("python", "inst-001")
        kernel.activate()
        time.sleep(0.15)
        kernel_manager.check_timeouts()
        assert kernel.state == KernelState.IDLE

    def test_sleep_timeout_transitions_to_sleeping(self, kernel_manager):
        """长时间空闲应自动切换到 SLEEPING."""
        kernel = kernel_manager.create_kernel("python", "inst-001")
        kernel.activate()
        time.sleep(0.25)
        kernel_manager.check_timeouts()
        assert kernel.state == KernelState.SLEEPING

    def test_wake_sleeping_kernel(self, kernel_manager):
        """唤醒 SLEEPING 内核应恢复到 ACTIVE."""
        kernel = kernel_manager.create_kernel("python", "inst-001")
        kernel.activate()
        kernel.to_idle()
        kernel.to_sleeping()

        kernel_manager.wake_kernel(kernel.kernel_id)
        assert kernel.state == KernelState.ACTIVE

    def test_wake_restores_checkpoint(self, kernel_manager):
        """唤醒时应从最近 checkpoint 恢复."""
        kernel = kernel_manager.create_kernel("python", "inst-001")
        kernel.activate()
        kernel.set_variable("test_var", "preserved")
        kernel.checkpoint()
        kernel.to_idle()
        kernel.to_sleeping()

        kernel_manager.wake_kernel(kernel.kernel_id)
        assert kernel.get_variable("test_var") == "preserved"

    def test_error_recovery(self, kernel_manager):
        """错误后应能从 checkpoint 恢复."""
        kernel = kernel_manager.create_kernel("python", "inst-001")
        kernel.activate()
        kernel.set_variable("important", "data")
        kernel.checkpoint()
        kernel.to_error("simulated error")

        recovered = kernel_manager.recover_kernel(kernel.kernel_id)
        assert recovered is True
        assert kernel.state == KernelState.ACTIVE
        assert kernel.get_variable("important") == "data"

    def test_recovery_failure_after_max_attempts(self, kernel_manager):
        """连续恢复失败超过 3 次应销毁内核."""
        kernel = kernel_manager.create_kernel("python", "inst-001")
        kernel.activate()
        kernel.to_error("error 1")

        # 没有 checkpoint，恢复会失败
        with pytest.raises(RecoveryExceededError):
            for _ in range(4):
                kernel_manager.recover_kernel(kernel.kernel_id)

        assert kernel.state == KernelState.DESTROYED

    def test_list_kernels_by_instance(self, kernel_manager):
        """按实例 ID 列出内核."""
        k1 = kernel_manager.create_kernel("python", "inst-a")
        k2 = kernel_manager.create_kernel("r", "inst-a")
        k3 = kernel_manager.create_kernel("python", "inst-b")

        inst_a_kernels = kernel_manager.list_kernels_by_instance("inst-a")
        assert len(inst_a_kernels) == 2
        assert k1 in inst_a_kernels
        assert k2 in inst_a_kernels
        assert k3 not in inst_a_kernels

    def test_auto_checkpoint(self, kernel_manager):
        """自动 checkpoint 应定期执行."""
        kernel = kernel_manager.create_kernel("python", "inst-001")
        kernel.activate()
        kernel.set_variable("counter", 1)

        time.sleep(0.08)
        kernel_manager.run_auto_checkpoints()

        # 应该至少有一个自动 checkpoint
        assert len(kernel_manager._checkpoint_store.list_checkpoints(kernel.kernel_id)) >= 1

    def test_kernel_resource_limits(self, kernel_manager):
        """内核应受资源限制."""
        kernel = kernel_manager.create_kernel("python", "inst-001")
        kernel.activate()
        # 模拟大量变量
        for i in range(1000):
            kernel.set_variable(f"var_{i}", i)
        assert kernel.memory_usage_mb() > 0


# ============================================================
# 3. CheckpointStore 测试
# ============================================================

class TestCheckpointStore:
    """检查点存储测试."""

    def test_save_and_load(self, checkpoint_store):
        """保存和加载 checkpoint."""
        data = {
            "kernel_id": "k-001",
            "cp_id": "cp-001",
            "variables": {"x": 1, "y": 2},
            "memory_usage_mb": 10.5,
        }
        checkpoint_store.save("k-001", "cp-001", data)
        loaded = checkpoint_store.load("k-001", "cp-001")
        assert loaded == data

    def test_load_nonexistent(self, checkpoint_store):
        """加载不存在的 checkpoint 返回 None."""
        assert checkpoint_store.load("k-001", "cp-nonexistent") is None

    def test_list_checkpoints(self, checkpoint_store):
        """列出内核的所有 checkpoint."""
        checkpoint_store.save("k-001", "cp-1", {"variables": {}})
        checkpoint_store.save("k-001", "cp-2", {"variables": {}})
        checkpoint_store.save("k-002", "cp-1", {"variables": {}})

        cps = checkpoint_store.list_checkpoints("k-001")
        assert len(cps) == 2
        assert "cp-1" in cps
        assert "cp-2" in cps

    def test_get_latest_checkpoint(self, checkpoint_store):
        """获取最新的 checkpoint."""
        checkpoint_store.save("k-001", "cp-1", {"variables": {"v": 1}})
        time.sleep(0.01)
        checkpoint_store.save("k-001", "cp-2", {"variables": {"v": 2}})

        latest = checkpoint_store.get_latest("k-001")
        assert latest is not None
        assert latest["variables"]["v"] == 2

    def test_delete_checkpoint(self, checkpoint_store):
        """删除 checkpoint."""
        checkpoint_store.save("k-001", "cp-1", {"variables": {}})
        checkpoint_store.delete("k-001", "cp-1")
        assert checkpoint_store.load("k-001", "cp-1") is None

    def test_delete_old_checkpoints(self, checkpoint_store):
        """清理旧 checkpoint（保留最近 N 个）."""
        for i in range(5):
            checkpoint_store.save("k-001", f"cp-{i}", {"variables": {"i": i}})
            time.sleep(0.01)

        checkpoint_store.delete_old("k-001", keep=2)
        cps = checkpoint_store.list_checkpoints("k-001")
        assert len(cps) == 2


# ============================================================
# 4. SessionForkManager 测试
# ============================================================

class TestSessionForkManager:
    """会话分叉管理器测试."""

    def test_create_fork(self, fork_manager, checkpoint_store):
        """从 checkpoint 创建 Fork."""
        checkpoint_store.save("sess-main", "cp-1", {
            "kernel_state": {"vars": {"x": 1}},
            "session_state": {"step": 5},
        })

        fork = fork_manager.create_fork(
            parent_session_id="sess-main",
            checkpoint_id="cp-1",
            trigger_type="ab_test",
            initiator="agent.guidance.decision",
            reason="测试两种策略",
        )

        assert fork.fork_id.startswith("fork-")
        assert fork.parent_session_id == "sess-main"
        assert fork.checkpoint_id == "cp-1"
        assert fork.status == ForkStatus.ACTIVE
        assert fork.depth == 1

    def test_fork_tree_depth_limit(self, fork_manager, checkpoint_store):
        """Fork 深度超过 3 层应报错."""
        checkpoint_store.save("sess-0", "cp-0", {"kernel_state": {}})

        f1 = fork_manager.create_fork("sess-0", "cp-0", "test", "agent", "")
        checkpoint_store.save(f1.fork_id, "cp-1", {"kernel_state": {}})

        f2 = fork_manager.create_fork(f1.fork_id, "cp-1", "test", "agent", "")
        checkpoint_store.save(f2.fork_id, "cp-2", {"kernel_state": {}})

        f3 = fork_manager.create_fork(f2.fork_id, "cp-2", "test", "agent", "")
        checkpoint_store.save(f3.fork_id, "cp-3", {"kernel_state": {}})

        with pytest.raises(MaxForkDepthError):
            fork_manager.create_fork(f3.fork_id, "cp-3", "test", "agent", "")

    def test_fork_concurrency_limit(self, fork_manager, checkpoint_store):
        """同一父 Session 最多 5 个活跃 Fork."""
        checkpoint_store.save("sess-main", "cp-base", {"kernel_state": {}})

        for i in range(5):
            fork_manager.create_fork("sess-main", "cp-base", "test", "agent", f"fork {i}")

        with pytest.raises(MaxForkConcurrencyError):
            fork_manager.create_fork("sess-main", "cp-base", "test", "agent", "too many")

    def test_merge_fork(self, fork_manager, checkpoint_store):
        """合并 Fork 回主 Session."""
        checkpoint_store.save("sess-main", "cp-1", {
            "kernel_state": {"vars": {"x": 1}},
        })

        fork = fork_manager.create_fork(
            "sess-main", "cp-1", "debate", "agent", "辩论分支"
        )

        result = fork_manager.merge_fork(fork.fork_id, "sess-main")
        assert result is True
        assert fork.status == ForkStatus.MERGED

    def test_archive_fork(self, fork_manager, checkpoint_store):
        """归档 Fork."""
        checkpoint_store.save("sess-main", "cp-1", {"kernel_state": {}})
        fork = fork_manager.create_fork("sess-main", "cp-1", "test", "agent", "")

        fork_manager.archive_fork(fork.fork_id)
        assert fork.status == ForkStatus.ARCHIVED

    def test_timeout_cleanup(self, fork_manager, checkpoint_store):
        """超时 Fork 自动回收."""
        checkpoint_store.save("sess-main", "cp-1", {"kernel_state": {}})
        fork = fork_manager.create_fork(
            "sess-main", "cp-1", "test", "agent", "",
            timeout_seconds=0.01,
        )

        time.sleep(0.02)
        fork_manager.cleanup_expired_forks()
        assert fork.status == ForkStatus.TIMED_OUT

    def test_list_forks_by_parent(self, fork_manager, checkpoint_store):
        """按父 Session 列出 Fork."""
        checkpoint_store.save("sess-main", "cp-1", {"kernel_state": {}})
        checkpoint_store.save("sess-other", "cp-1", {"kernel_state": {}})

        f1 = fork_manager.create_fork("sess-main", "cp-1", "test", "agent", "")
        f2 = fork_manager.create_fork("sess-main", "cp-1", "test", "agent", "")
        f3 = fork_manager.create_fork("sess-other", "cp-1", "test", "agent", "")

        main_forks = fork_manager.list_forks_by_parent("sess-main")
        assert len(main_forks) == 2
        assert f1.fork_id in [f.fork_id for f in main_forks]
        assert f2.fork_id in [f.fork_id for f in main_forks]
        assert f3.fork_id not in [f.fork_id for f in main_forks]

    def test_fork_channel_prefix(self, fork_manager, checkpoint_store):
        """Fork 应有独立的 channel 前缀."""
        checkpoint_store.save("sess-main", "cp-1", {"kernel_state": {}})
        fork = fork_manager.create_fork(
            "sess-main", "cp-1", "test", "agent", "",
            channel_prefix="fork.deb_001",
        )
        assert fork.channel_prefix == "fork.deb_001"

    def test_fork_provenance(self, fork_manager, checkpoint_store):
        """Fork 操作应记录 Provenance."""
        checkpoint_store.save("sess-main", "cp-1", {"kernel_state": {}})
        fork = fork_manager.create_fork(
            "sess-main", "cp-1", "ab_test", "agent.guidance.decision", "A/B 测试"
        )

        assert fork.provenance["action"] == "fork.create"
        assert fork.provenance["parent_session_id"] == "sess-main"
        assert fork.provenance["checkpoint_id"] == "cp-1"
        assert fork.provenance["trigger_type"] == "ab_test"
        assert "timestamp" in fork.provenance


# ============================================================
# 5. StatePersistence 测试
# ============================================================

class TestStatePersistence:
    """状态持久化协调器测试."""

    def test_coordinate_checkpoint(self):
        """协调多个内核的 checkpoint."""
        store = MemoryCheckpointStore()
        sp = StatePersistence(checkpoint_store=store)

        kernel1 = PersistentKernel("python", "inst-001", store)
        kernel1.warmup()
        kernel1.activate()
        kernel1.set_variable("a", 1)

        kernel2 = PersistentKernel("r", "inst-001", store)
        kernel2.warmup()
        kernel2.activate()
        kernel2.set_variable("b", 2)

        sp.checkpoint_all([kernel1, kernel2], session_id="sess-001")

        assert len(store.list_checkpoints(kernel1.kernel_id)) >= 1
        assert len(store.list_checkpoints(kernel2.kernel_id)) >= 1

    def test_coordinate_restore_session(self):
        """从 checkpoint 恢复整个会话."""
        store = MemoryCheckpointStore()
        sp = StatePersistence(checkpoint_store=store)

        kernel = PersistentKernel("python", "inst-001", store)
        kernel.warmup()
        kernel.activate()
        kernel.set_variable("model", "v1")
        cp_id = kernel.checkpoint()

        # 模拟状态变化
        kernel.set_variable("model", "v2")

        restored = sp.restore_session(
            kernel, session_id="sess-001", checkpoint_id=cp_id
        )
        assert restored is True
        assert kernel.get_variable("model") == "v1"

    def test_persist_session_state(self):
        """持久化会话状态."""
        store = MemoryCheckpointStore()
        sp = StatePersistence(checkpoint_store=store)

        session_state = {"step": 5, "learner_id": "stu-001", "path": ["a", "b"]}
        sp.persist_session_state("sess-001", session_state)

        loaded = store.load_session_state("sess-001")
        assert loaded == session_state

    def test_auto_checkpoint_scheduler(self):
        """自动 checkpoint 调度器."""
        store = MemoryCheckpointStore()
        sp = StatePersistence(
            checkpoint_store=store,
            auto_checkpoint_interval_s=0.05,
        )

        kernel = PersistentKernel("python", "inst-001", store)
        kernel.warmup()
        kernel.activate()
        kernel.set_variable("counter", 1)

        sp.register_kernel(kernel)
        time.sleep(0.08)
        sp.run_auto_checkpoints()

        cps = store.list_checkpoints(kernel.kernel_id)
        assert len(cps) >= 1

    def test_cleanup_old_checkpoints(self):
        """清理旧 checkpoint."""
        store = MemoryCheckpointStore()
        sp = StatePersistence(checkpoint_store=store)

        kernel = PersistentKernel("python", "inst-001", store)
        kernel.warmup()
        kernel.activate()

        for i in range(5):
            kernel.set_variable("idx", i)
            kernel.checkpoint()
            time.sleep(0.01)

        sp.cleanup_old_checkpoints(kernel.kernel_id, keep=2)
        assert len(store.list_checkpoints(kernel.kernel_id)) == 2
