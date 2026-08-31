"""持久化内核与状态管理模块 — L5 Agent Runtime 核心组件.

融合世界先进方案:
- Claude Science: Persistent Kernels + Session Fork 理念
- LangGraph: Checkpoint 机制 + 状态恢复
- Jupyter: Kernel 进程模型 + 变量空间隔离
- Temporal: Activity 状态机 + 重试恢复

本模块实现:
1. PersistentKernel — 持久化内核（状态机 + 变量保留 + 执行）
2. KernelManager — 内核生命周期管理（超时 + 恢复）
3. CheckpointStore — 检查点存储抽象（内存实现）
4. SessionForkManager — 会话分叉管理（Fork/合并/清理）
5. StatePersistence — 状态持久化协调器（自动 checkpoint）
"""

from __future__ import annotations

import ast
import logging
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class KernelState(str, Enum):
    """内核生命周期状态 (借鉴 Temporal Activity 状态机 + Jupyter Kernel).

    INITIALIZING → WARMING_UP → ACTIVE ⇄ IDLE → SLEEPING → DESTROYED
                                     ↘ ERROR → (恢复) → ACTIVE
    """

    INITIALIZING = "initializing"
    WARMING_UP = "warming_up"
    ACTIVE = "active"
    IDLE = "idle"
    SLEEPING = "sleeping"
    DESTROYED = "destroyed"
    ERROR = "error"


class ForkStatus(str, Enum):
    """Fork 会话状态."""

    ACTIVE = "active"
    MERGED = "merged"
    ARCHIVED = "archived"
    TIMED_OUT = "timed_out"


# ============================================================
# 异常定义
# ============================================================


class KernelStateError(Exception):
    """内核状态转换错误."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class KernelError(Exception):
    """内核执行错误."""

    def __init__(self, kernel_id: str, message: str) -> None:
        self.kernel_id = kernel_id
        self.message = message
        super().__init__(f"[{kernel_id}] {message}")


class RecoveryExceededError(Exception):
    """恢复尝试次数超限."""

    def __init__(self, kernel_id: str, attempts: int) -> None:
        self.kernel_id = kernel_id
        self.attempts = attempts
        super().__init__(
            f"Kernel {kernel_id} recovery failed after {attempts} attempts"
        )


class MaxForkDepthError(Exception):
    """Fork 深度超过限制."""

    def __init__(self, depth: int, max_depth: int) -> None:
        super().__init__(f"Fork depth {depth} exceeds maximum {max_depth}")


class MaxForkConcurrencyError(Exception):
    """Fork 并发数超过限制."""

    def __init__(self, count: int, max_count: int) -> None:
        super().__init__(f"Fork count {count} exceeds maximum {max_count}")


# ============================================================
# CheckpointStore — 检查点存储抽象
# ============================================================


class CheckpointStore(ABC):
    """检查点存储抽象基类.

    提供 save/load/list/delete 接口，支持内存和 PostgreSQL 实现。
    """

    @abstractmethod
    def save(self, kernel_id: str, cp_id: str, data: dict[str, Any]) -> None:
        """保存 checkpoint."""
        ...

    @abstractmethod
    def load(self, kernel_id: str, cp_id: str) -> dict[str, Any] | None:
        """加载 checkpoint."""
        ...

    @abstractmethod
    def list_checkpoints(self, kernel_id: str) -> list[str]:
        """列出内核的所有 checkpoint ID."""
        ...

    @abstractmethod
    def get_latest(self, kernel_id: str) -> dict[str, Any] | None:
        """获取最新的 checkpoint."""
        ...

    @abstractmethod
    def delete(self, kernel_id: str, cp_id: str) -> None:
        """删除 checkpoint."""
        ...

    @abstractmethod
    def delete_old(self, kernel_id: str, keep: int) -> None:
        """保留最近 N 个 checkpoint，删除其余."""
        ...

    @abstractmethod
    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        """保存会话状态."""
        ...

    @abstractmethod
    def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        """加载会话状态."""
        ...


class MemoryCheckpointStore(CheckpointStore):
    """内存检查点存储实现 (用于开发和测试)."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._timestamps: dict[str, dict[str, float]] = defaultdict(dict)
        self._session_states: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def save(self, kernel_id: str, cp_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._store[kernel_id][cp_id] = data
            self._timestamps[kernel_id][cp_id] = time.time()

    def load(self, kernel_id: str, cp_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._store.get(kernel_id, {}).get(cp_id)

    def list_checkpoints(self, kernel_id: str) -> list[str]:
        with self._lock:
            cps = list(self._store.get(kernel_id, {}).keys())
            # 按时间戳排序
            timestamps = self._timestamps.get(kernel_id, {})
            cps.sort(key=lambda cp: timestamps.get(cp, 0))
            return cps

    def get_latest(self, kernel_id: str) -> dict[str, Any] | None:
        with self._lock:
            cps = self.list_checkpoints(kernel_id)
            if not cps:
                return None
            return self._store[kernel_id].get(cps[-1])

    def delete(self, kernel_id: str, cp_id: str) -> None:
        with self._lock:
            if kernel_id in self._store and cp_id in self._store[kernel_id]:
                del self._store[kernel_id][cp_id]
                del self._timestamps[kernel_id][cp_id]

    def delete_old(self, kernel_id: str, keep: int) -> None:
        with self._lock:
            cps = self.list_checkpoints(kernel_id)
            to_delete = cps[:-keep] if len(cps) > keep else []
            for cp_id in to_delete:
                self.delete(kernel_id, cp_id)

    def save_session_state(self, session_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._session_states[session_id] = state

    def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._session_states.get(session_id)


# ============================================================
# PersistentKernel — 持久化内核
# ============================================================


class _SafeExecDict(dict):
    """安全的执行命名空间 — 限制危险操作."""

    def __init__(self) -> None:
        super().__init__()
        # 允许的基本函数
        self.update({
            "__builtins__": {
                "abs": abs, "all": all, "any": any, "bin": bin,
                "bool": bool, "chr": chr, "dict": dict, "divmod": divmod,
                "enumerate": enumerate, "filter": filter, "float": float,
                "format": format, "frozenset": frozenset, "hex": hex,
                "int": int, "isinstance": isinstance, "issubclass": issubclass,
                "len": len, "list": list, "map": map, "max": max, "min": min,
                "next": next, "oct": oct, "ord": ord, "pow": pow,
                "range": range, "reversed": reversed, "round": round,
                "set": set, "slice": slice, "sorted": sorted, "str": str,
                "sum": sum, "tuple": tuple, "type": type, "zip": zip,
                "print": lambda *args, **kwargs: None,  # 静默 print
            },
        })


class PersistentKernel:
    """持久化内核 (L5 设计文档 第三章).

    融合世界先进方案:
    - Claude Science: 持久变量空间
    - Jupyter: Kernel 进程模型
    - LangGraph: Checkpoint 状态恢复
    - Temporal: 状态机 + 重试

    核心能力:
    1. 六态生命周期管理
    2. 跨 sub-task 变量保留
    3. 代码执行（受限 Python）
    4. Checkpoint 保存/恢复
    5. 内存使用追踪
    """

    # 合法状态转换: 当前状态 → 允许的目标状态集合
    _VALID_TRANSITIONS: dict[KernelState, set[KernelState]] = {
        KernelState.INITIALIZING: {KernelState.WARMING_UP, KernelState.DESTROYED},
        KernelState.WARMING_UP: {KernelState.ACTIVE, KernelState.DESTROYED, KernelState.ERROR},
        KernelState.ACTIVE: {KernelState.IDLE, KernelState.DESTROYED, KernelState.ERROR},
        KernelState.IDLE: {KernelState.ACTIVE, KernelState.SLEEPING, KernelState.DESTROYED, KernelState.ERROR},
        KernelState.SLEEPING: {KernelState.ACTIVE, KernelState.DESTROYED},
        KernelState.ERROR: {KernelState.ACTIVE, KernelState.DESTROYED},
        KernelState.DESTROYED: set(),  # 终态
    }

    def __init__(
        self,
        kernel_type: str,
        instance_id: str,
        checkpoint_store: CheckpointStore,
        max_ram_mb: float = 512.0,
    ) -> None:
        self.kernel_type = kernel_type
        self.instance_id = instance_id
        self.kernel_id = f"kernel-{uuid.uuid4().hex[:12]}"
        self.checkpoint_store = checkpoint_store
        self.max_ram_mb = max_ram_mb

        self._state = KernelState.INITIALIZING
        self._variables: dict[str, Any] = {}
        self._namespace = _SafeExecDict()
        self._namespace.update(self._variables)

        self.created_at = time.time()
        self.last_active_at = self.created_at
        self.activated_at: float | None = None
        self.error_message: str = ""
        self._recovery_attempts = 0
        self._lock = threading.RLock()

        logger.info(
            f"[Kernel] Created {self.kernel_id} ({kernel_type}) for {instance_id}"
        )

    @property
    def state(self) -> KernelState:
        with self._lock:
            return self._state

    def _transition(self, target: KernelState) -> None:
        """执行状态转换（含合法性校验）."""
        with self._lock:
            current = self._state
            allowed = self._VALID_TRANSITIONS.get(current, set())
            if target not in allowed:
                raise KernelStateError(
                    f"Invalid transition: {current.value} → {target.value}"
                )
            self._state = target
            self.last_active_at = time.time()
            logger.debug(
                f"[Kernel] {self.kernel_id}: {current.value} → {target.value}"
            )

    def warmup(self) -> None:
        """预热: INITIALIZING → WARMING_UP."""
        self._transition(KernelState.WARMING_UP)
        # 模拟依赖库加载
        self._namespace["__kernel_ready__"] = True

    def activate(self) -> None:
        """激活: WARMING_UP/IDLE/SLEEPING/ERROR → ACTIVE."""
        with self._lock:
            if self._state == KernelState.WARMING_UP:
                self._transition(KernelState.ACTIVE)
                self.activated_at = time.time()
            elif self._state in (KernelState.IDLE, KernelState.SLEEPING, KernelState.ERROR):
                self._transition(KernelState.ACTIVE)
            else:
                raise KernelStateError(
                    f"Cannot activate from {self._state.value}"
                )

    def to_idle(self) -> None:
        """进入空闲: ACTIVE → IDLE."""
        self._transition(KernelState.IDLE)

    def to_sleeping(self) -> None:
        """进入休眠: IDLE → SLEEPING."""
        self._transition(KernelState.SLEEPING)

    def to_error(self, message: str) -> None:
        """进入错误状态."""
        with self._lock:
            self.error_message = message
            self._recovery_attempts += 1
            self._transition(KernelState.ERROR)
            logger.error(f"[Kernel] {self.kernel_id} ERROR: {message}")

    def destroy(self) -> None:
        """销毁内核，释放所有资源."""
        with self._lock:
            if self._state == KernelState.DESTROYED:
                return
            self._variables.clear()
            self._namespace.clear()
            self._state = KernelState.DESTROYED
            logger.info(f"[Kernel] Destroyed {self.kernel_id}")

    def set_variable(self, name: str, value: Any) -> None:
        """设置变量（跨 sub-task 保留）."""
        with self._lock:
            self._variables[name] = value
            self._namespace[name] = value
            self.last_active_at = time.time()

    def get_variable(self, name: str) -> Any | None:
        """获取变量."""
        with self._lock:
            return self._variables.get(name)

    def execute(self, code: str) -> dict[str, Any]:
        """执行代码（受限 Python 子集）.

        使用 ast 解析限制危险操作，在隔离命名空间中执行。
        返回结果字典: {"success": bool, "result": Any, "error": str}
        """
        with self._lock:
            if self._state not in (KernelState.ACTIVE, KernelState.WARMING_UP):
                return {
                    "success": False,
                    "result": None,
                    "error": f"Kernel not active (state={self._state.value})",
                }

            try:
                # 安全检查：禁止 import 和 __import__
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        return {
                            "success": False,
                            "result": None,
                            "error": "Import statements are not allowed",
                        }
                    if isinstance(node, ast.Call):
                        # 检查是否调用 __import__
                        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                            return {
                                "success": False,
                                "result": None,
                                "error": "__import__ is not allowed",
                            }

                # 执行代码
                exec(compile(tree, "<kernel>", "exec"), self._namespace)
                self.last_active_at = time.time()

                # 同步变量回 _variables
                for key in list(self._namespace.keys()):
                    if not key.startswith("_"):
                        self._variables[key] = self._namespace[key]

                return {"success": True, "result": None, "error": ""}
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                return {"success": False, "result": None, "error": error_msg}

    def checkpoint(self) -> str:
        """创建 checkpoint 并保存到存储后端.

        Returns:
            checkpoint ID
        """
        with self._lock:
            cp_id = f"cp-{uuid.uuid4().hex[:8]}"
            data = {
                "kernel_id": self.kernel_id,
                "cp_id": cp_id,
                "variables": dict(self._variables),
                "memory_usage_mb": self.memory_usage_mb(),
                "state": self._state.value,
                "timestamp": time.time(),
            }
            self.checkpoint_store.save(self.kernel_id, cp_id, data)
            logger.debug(f"[Kernel] Checkpoint {cp_id} saved for {self.kernel_id}")
            return cp_id

    def restore_from_checkpoint(self, cp_id: str) -> bool:
        """从 checkpoint 恢复状态.

        Returns:
            是否成功恢复
        """
        with self._lock:
            data = self.checkpoint_store.load(self.kernel_id, cp_id)
            if data is None:
                logger.warning(
                    f"[Kernel] Checkpoint {cp_id} not found for {self.kernel_id}"
                )
                return False

            self._variables = dict(data.get("variables", {}))
            self._namespace.clear()
            self._namespace.update(_SafeExecDict())
            self._namespace.update(self._variables)
            self.last_active_at = time.time()
            logger.info(
                f"[Kernel] Restored {self.kernel_id} from checkpoint {cp_id}"
            )
            return True

    def memory_usage_mb(self) -> float:
        """估算内存使用（MB）."""
        with self._lock:
            total = 0.0
            for v in self._variables.values():
                try:
                    total += sys.getsizeof(v) / (1024 * 1024)
                except Exception:
                    pass
            return round(total, 2)

    @property
    def is_alive(self) -> bool:
        return self.state not in (KernelState.DESTROYED, KernelState.ERROR)


# ============================================================
# KernelManager — 内核生命周期管理器
# ============================================================


class KernelManager:
    """内核生命周期管理器.

    功能:
    - 创建/销毁内核
    - 空闲/睡眠超时自动转换
    - 错误恢复（从 checkpoint 恢复，最多3次）
    - 自动 checkpoint 调度

    融合世界先进方案:
    - Jupyter: Kernel 进程管理
    - Temporal: Activity 心跳 + 超时
    - Kubernetes: 探针模型
    """

    def __init__(
        self,
        checkpoint_store: CheckpointStore,
        idle_timeout_s: float = 60.0,
        sleep_timeout_s: float = 300.0,
        auto_checkpoint_interval_s: float = 30.0,
        max_recovery_attempts: int = 3,
    ) -> None:
        self._checkpoint_store = checkpoint_store
        self._idle_timeout = idle_timeout_s
        self._sleep_timeout = sleep_timeout_s
        self._auto_checkpoint_interval = auto_checkpoint_interval_s
        self._max_recovery_attempts = max_recovery_attempts

        self._kernels: dict[str, PersistentKernel] = {}
        self._instance_index: dict[str, set[str]] = defaultdict(set)
        self._last_checkpoint_time: dict[str, float] = {}
        self._lock = threading.RLock()

    def create_kernel(
        self, kernel_type: str, instance_id: str
    ) -> PersistentKernel:
        """创建新内核."""
        kernel = PersistentKernel(
            kernel_type=kernel_type,
            instance_id=instance_id,
            checkpoint_store=self._checkpoint_store,
        )
        kernel.warmup()

        with self._lock:
            self._kernels[kernel.kernel_id] = kernel
            self._instance_index[instance_id].add(kernel.kernel_id)

        logger.info(
            f"[KernelManager] Created {kernel.kernel_id} ({kernel_type})"
        )
        return kernel

    def get_kernel(self, kernel_id: str) -> PersistentKernel | None:
        """获取内核."""
        with self._lock:
            return self._kernels.get(kernel_id)

    def destroy_kernel(self, kernel_id: str) -> None:
        """销毁内核."""
        with self._lock:
            kernel = self._kernels.get(kernel_id)
            if kernel is None:
                return
            kernel.destroy()
            self._kernels.pop(kernel_id, None)
            self._instance_index[kernel.instance_id].discard(kernel_id)
            logger.info(f"[KernelManager] Destroyed {kernel_id}")

    def list_kernels_by_instance(
        self, instance_id: str
    ) -> list[PersistentKernel]:
        """按实例 ID 列出内核."""
        with self._lock:
            ids = self._instance_index.get(instance_id, set())
            return [
                self._kernels[kid]
                for kid in ids
                if kid in self._kernels
            ]

    def check_timeouts(self) -> None:
        """检查所有内核的超时状态并自动转换."""
        now = time.time()
        with self._lock:
            for kernel in list(self._kernels.values()):
                if kernel.state == KernelState.ACTIVE:
                    idle_time = now - kernel.last_active_at
                    if idle_time > self._sleep_timeout:
                        # 超过睡眠超时 → SLEEPING
                        try:
                            kernel.to_idle()
                            kernel.to_sleeping()
                        except KernelStateError:
                            pass
                    elif idle_time > self._idle_timeout:
                        # 超过空闲超时 → IDLE
                        try:
                            kernel.to_idle()
                        except KernelStateError:
                            pass

    def wake_kernel(self, kernel_id: str) -> None:
        """唤醒 SLEEPING 内核."""
        kernel = self.get_kernel(kernel_id)
        if kernel is None:
            return

        if kernel.state == KernelState.SLEEPING:
            # 从最近 checkpoint 恢复
            latest = self._checkpoint_store.get_latest(kernel_id)
            if latest:
                kernel.restore_from_checkpoint(latest["cp_id"])
            kernel.activate()
            logger.info(f"[KernelManager] Woke {kernel_id} from sleeping")

    def recover_kernel(self, kernel_id: str) -> bool:
        """从错误中恢复内核.

        Returns:
            是否成功恢复

        Raises:
            RecoveryExceededError: 恢复次数超过上限
        """
        kernel = self.get_kernel(kernel_id)
        if kernel is None:
            return False

        with self._lock:
            if kernel._recovery_attempts >= self._max_recovery_attempts:
                kernel.destroy()
                raise RecoveryExceededError(
                    kernel_id, kernel._recovery_attempts
                )

            # 从最近 checkpoint 恢复
            latest = self._checkpoint_store.get_latest(kernel_id)
            if latest:
                kernel.restore_from_checkpoint(latest["cp_id"])
                kernel.activate()
                logger.info(
                    f"[KernelManager] Recovered {kernel_id} from checkpoint"
                )
                return True
            else:
                # 没有 checkpoint，恢复失败
                kernel._recovery_attempts += 1
                logger.warning(
                    f"[KernelManager] No checkpoint for {kernel_id}, "
                    f"recovery attempt {kernel._recovery_attempts}"
                )
                return False

    def run_auto_checkpoints(self) -> None:
        """为所有活跃内核执行自动 checkpoint."""
        now = time.time()
        with self._lock:
            for kernel in list(self._kernels.values()):
                if kernel.state not in (
                    KernelState.ACTIVE,
                    KernelState.IDLE,
                    KernelState.SLEEPING,
                ):
                    continue

                last_cp = self._last_checkpoint_time.get(kernel.kernel_id, 0)
                if now - last_cp >= self._auto_checkpoint_interval:
                    kernel.checkpoint()
                    self._last_checkpoint_time[kernel.kernel_id] = now


# ============================================================
# SessionForkManager — 会话分叉管理
# ============================================================


class ForkConfig:
    """Fork 配置."""

    def __init__(
        self,
        channel_prefix: str = "",
        timeout_seconds: float = 1800.0,
        agent_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.channel_prefix = channel_prefix
        self.timeout_seconds = timeout_seconds
        self.agent_overrides = agent_overrides or {}


class ForkRecord:
    """Fork 记录."""

    def __init__(
        self,
        fork_id: str,
        parent_session_id: str,
        checkpoint_id: str,
        trigger_type: str,
        initiator: str,
        reason: str,
        depth: int,
        channel_prefix: str,
        timeout_seconds: float,
    ) -> None:
        self.fork_id = fork_id
        self.parent_session_id = parent_session_id
        self.checkpoint_id = checkpoint_id
        self.trigger_type = trigger_type
        self.initiator = initiator
        self.reason = reason
        self.depth = depth
        self.status = ForkStatus.ACTIVE
        self.channel_prefix = channel_prefix
        self.timeout_seconds = timeout_seconds
        self.created_at = time.time()
        self.merged_at: float | None = None
        self.provenance = {
            "action": "fork.create",
            "parent_session_id": parent_session_id,
            "checkpoint_id": checkpoint_id,
            "trigger_type": trigger_type,
            "initiator": initiator,
            "reason": reason,
            "timestamp": self.created_at,
        }


class SessionForkManager:
    """会话分叉管理器 (L5 设计文档 第四章).

    融合世界先进方案:
    - Claude Science: Session Fork 理念
    - Git: 分支/合并语义
    - LangGraph: 并行路径探索

    约束:
    - 最大 Fork 深度: 3 层
    - 最大并发 Fork: 每个父 Session 5 个
    - Fork 超时: 30 分钟
    """

    MAX_FORK_DEPTH = 3
    MAX_FORK_CONCURRENCY = 5

    def __init__(self, checkpoint_store: CheckpointStore) -> None:
        self._checkpoint_store = checkpoint_store
        self._forks: dict[str, ForkRecord] = {}
        self._parent_index: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def create_fork(
        self,
        parent_session_id: str,
        checkpoint_id: str,
        trigger_type: str,
        initiator: str,
        reason: str,
        channel_prefix: str = "",
        timeout_seconds: float = 1800.0,
    ) -> ForkRecord:
        """创建 Fork.

        Raises:
            MaxForkDepthError: 超过最大 Fork 深度
            MaxForkConcurrencyError: 超过最大并发数
        """
        with self._lock:
            # 计算 Fork 深度
            depth = self._compute_depth(parent_session_id)
            if depth >= self.MAX_FORK_DEPTH:
                raise MaxForkDepthError(depth + 1, self.MAX_FORK_DEPTH)

            # 检查并发数
            active_count = self._count_active_forks(parent_session_id)
            if active_count >= self.MAX_FORK_CONCURRENCY:
                raise MaxForkConcurrencyError(
                    active_count + 1, self.MAX_FORK_CONCURRENCY
                )

            fork_id = f"fork-{uuid.uuid4().hex[:12]}"
            fork = ForkRecord(
                fork_id=fork_id,
                parent_session_id=parent_session_id,
                checkpoint_id=checkpoint_id,
                trigger_type=trigger_type,
                initiator=initiator,
                reason=reason,
                depth=depth + 1,
                channel_prefix=channel_prefix or f"fork.{fork_id[:8]}",
                timeout_seconds=timeout_seconds,
            )

            self._forks[fork_id] = fork
            self._parent_index[parent_session_id].add(fork_id)

            logger.info(
                f"[ForkManager] Created {fork_id} from {parent_session_id} "
                f"(depth={fork.depth}, trigger={trigger_type})"
            )
            return fork

    def _compute_depth(self, parent_id: str) -> int:
        """计算父 Session 的 Fork 深度."""
        fork = self._forks.get(parent_id)
        if fork is None:
            return 0
        return fork.depth

    def _count_active_forks(self, parent_id: str) -> int:
        """统计父 Session 的活跃 Fork 数."""
        count = 0
        for fork_id in self._parent_index.get(parent_id, set()):
            fork = self._forks.get(fork_id)
            if fork and fork.status == ForkStatus.ACTIVE:
                count += 1
        return count

    def get_fork(self, fork_id: str) -> ForkRecord | None:
        """获取 Fork 记录."""
        with self._lock:
            return self._forks.get(fork_id)

    def merge_fork(self, fork_id: str, target_session_id: str) -> bool:
        """合并 Fork 回目标 Session.

        Returns:
            是否成功合并
        """
        with self._lock:
            fork = self._forks.get(fork_id)
            if fork is None or fork.status != ForkStatus.ACTIVE:
                return False

            fork.status = ForkStatus.MERGED
            fork.merged_at = time.time()
            fork.provenance["merge_target"] = target_session_id
            fork.provenance["merged_at"] = fork.merged_at

            logger.info(
                f"[ForkManager] Merged {fork_id} into {target_session_id}"
            )
            return True

    def archive_fork(self, fork_id: str) -> bool:
        """归档 Fork."""
        with self._lock:
            fork = self._forks.get(fork_id)
            if fork is None:
                return False
            fork.status = ForkStatus.ARCHIVED
            logger.info(f"[ForkManager] Archived {fork_id}")
            return True

    def cleanup_expired_forks(self) -> list[str]:
        """清理超时的 Fork.

        Returns:
            被清理的 fork_id 列表
        """
        now = time.time()
        cleaned: list[str] = []

        with self._lock:
            for fork in list(self._forks.values()):
                if fork.status != ForkStatus.ACTIVE:
                    continue
                if now - fork.created_at > fork.timeout_seconds:
                    fork.status = ForkStatus.TIMED_OUT
                    cleaned.append(fork.fork_id)
                    logger.info(f"[ForkManager] Timed out {fork.fork_id}")

        return cleaned

    def list_forks_by_parent(self, parent_session_id: str) -> list[ForkRecord]:
        """按父 Session 列出 Fork."""
        with self._lock:
            fork_ids = self._parent_index.get(parent_session_id, set())
            return [
                self._forks[fid]
                for fid in fork_ids
                if fid in self._forks
            ]


# ============================================================
# StatePersistence — 状态持久化协调器
# ============================================================


class StatePersistence:
    """状态持久化协调器.

    协调多个内核和会话的持久化操作:
    - 批量 checkpoint
    - 会话状态保存/加载
    - 自动 checkpoint 调度
    - 旧 checkpoint 清理
    """

    def __init__(
        self,
        checkpoint_store: CheckpointStore,
        auto_checkpoint_interval_s: float = 30.0,
    ) -> None:
        self._checkpoint_store = checkpoint_store
        self._auto_checkpoint_interval = auto_checkpoint_interval_s
        self._registered_kernels: dict[str, PersistentKernel] = {}
        self._last_checkpoint_time: dict[str, float] = {}
        self._lock = threading.RLock()

    def register_kernel(self, kernel: PersistentKernel) -> None:
        """注册内核进行自动 checkpoint."""
        with self._lock:
            self._registered_kernels[kernel.kernel_id] = kernel
            self._last_checkpoint_time[kernel.kernel_id] = time.time()

    def unregister_kernel(self, kernel_id: str) -> None:
        """注销内核."""
        with self._lock:
            self._registered_kernels.pop(kernel_id, None)
            self._last_checkpoint_time.pop(kernel_id, None)

    def checkpoint_all(
        self, kernels: list[PersistentKernel], session_id: str
    ) -> list[str]:
        """为多个内核协调创建 checkpoint.

        Args:
            kernels: 需要 checkpoint 的内核列表
            session_id: 关联的会话 ID

        Returns:
            checkpoint ID 列表
        """
        cp_ids: list[str] = []
        with self._lock:
            for kernel in kernels:
                cp_id = kernel.checkpoint()
                cp_ids.append(cp_id)
                self._last_checkpoint_time[kernel.kernel_id] = time.time()
            logger.info(
                f"[StatePersistence] Batch checkpoint for session {session_id}: "
                f"{len(cp_ids)} kernels"
            )
        return cp_ids

    def restore_session(
        self,
        kernel: PersistentKernel,
        session_id: str,
        checkpoint_id: str,
    ) -> bool:
        """从 checkpoint 恢复会话状态.

        Args:
            kernel: 目标内核
            session_id: 会话 ID
            checkpoint_id: checkpoint ID

        Returns:
            是否成功恢复
        """
        with self._lock:
            result = kernel.restore_from_checkpoint(checkpoint_id)
            if result:
                logger.info(
                    f"[StatePersistence] Restored session {session_id} "
                    f"for kernel {kernel.kernel_id}"
                )
            return result

    def persist_session_state(
        self, session_id: str, session_state: dict[str, Any]
    ) -> None:
        """持久化会话状态.

        Args:
            session_id: 会话 ID
            session_state: 会话状态字典
        """
        with self._lock:
            self._checkpoint_store.save_session_state(
                session_id, session_state
            )
            logger.debug(
                f"[StatePersistence] Persisted session state for {session_id}"
            )

    def run_auto_checkpoints(self) -> None:
        """为所有已注册的内核执行自动 checkpoint."""
        now = time.time()
        with self._lock:
            for kernel_id, kernel in list(self._registered_kernels.items()):
                if kernel.state in (
                    KernelState.DESTROYED,
                ):
                    continue
                last_cp = self._last_checkpoint_time.get(kernel_id, 0)
                if now - last_cp >= self._auto_checkpoint_interval:
                    kernel.checkpoint()
                    self._last_checkpoint_time[kernel_id] = now

    def cleanup_old_checkpoints(
        self, kernel_id: str, keep: int = 5
    ) -> None:
        """清理旧 checkpoint.

        Args:
            kernel_id: 内核 ID
            keep: 保留的 checkpoint 数量
        """
        with self._lock:
            self._checkpoint_store.delete_old(kernel_id, keep)
            logger.debug(
                f"[StatePersistence] Cleaned old checkpoints for {kernel_id}, "
                f"keep={keep}"
            )