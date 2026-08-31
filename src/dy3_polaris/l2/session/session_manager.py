"""L2 个性化会话管理器.

融合世界先进方案:
- Jupyter: Kernel 会话生命周期 (创建/激活/暂停/关闭)
- Claude Science: Session Fork + Checkpoint 机制
- Khan Academy: 学习会话上下文继承
- Temporal: Activity 状态机 + 检查点恢复

会话状态机:
  created -> active -> paused -> active (可循环)
                    -> closed (终态)

会话检查点:
  - 每 N 步自动快照 (默认 N=5)
  - 支持手动添加检查点
  - 检查点包含 SHA-256 哈希用于完整性验证

设计依据:
- 参考 L1 session_manager.py 的线程安全会话管理模式 (threading.RLock + 内部索引)
- 参考 L2 store.py 的依赖注入 + 抽象存储模式
- SessionManager 通过依赖注入接收 L2Store, store 为 None 时内部创建 InMemoryL2Store
- 会话持久化委托给 store, manager 维护 learner_id -> [session_id] 索引用于活跃会话查询

线程安全: threading.RLock 保护内部索引; store 自身线程安全 (InMemoryL2Store 使用 RLock).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from typing import Any

from dy3_polaris.l2.exceptions import StoreError
from dy3_polaris.l2.models import SessionRecord
from dy3_polaris.l2.store import InMemoryL2Store, L2Store


# ============================================================
# 1. 常量定义
# ============================================================

# session_id 前缀 (统一命名空间: l2s-, 12 位 hex; 见 shared/ids.py)
SESSION_ID_PREFIX: str = "l2s"

# 自动检查点间隔 (每 N 步自动快照, 默认 N=5)
DEFAULT_CHECKPOINT_INTERVAL: int = 5

# 会话超时时间 (秒), 默认 30 分钟; 超时后 active 会话可被 cleanup_expired_sessions 关闭
DEFAULT_SESSION_TIMEOUT: float = 1800.0

# 会话状态常量
STATUS_CREATED: str = "created"
STATUS_ACTIVE: str = "active"
STATUS_PAUSED: str = "paused"
STATUS_CLOSED: str = "closed"
STATUS_MERGED: str = "merged"

# 合法状态转移表 (Activity 状态机):
#   created -> active
#   active  -> paused / closed
#   paused  -> active / closed
# (closed / merged 为终态, 无后续转移)
_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    STATUS_CREATED: frozenset({STATUS_ACTIVE}),
    STATUS_ACTIVE: frozenset({STATUS_PAUSED, STATUS_CLOSED}),
    STATUS_PAUSED: frozenset({STATUS_ACTIVE, STATUS_CLOSED}),
}

# fork 元数据中记录"继承检查点基线长度"的键 (用于 merge_fork 区分新增检查点)
_FORK_BASE_LEN_KEY: str = "fork_base_checkpoint_count"


# ============================================================
# 2. SessionManager 个性化会话管理器
# ============================================================


class SessionManager:
    """L2 个性化会话管理器.

    借鉴世界先进方案:
    - Jupyter: Kernel 会话生命周期 (创建/激活/暂停/关闭)
    - Claude Science: Session Fork + Checkpoint 机制 (检查点含 SHA-256 完整性哈希)
    - Khan Academy: 学习会话上下文继承 (goal 写入 context_envelope)
    - Temporal: Activity 状态机 + 检查点恢复

    核心功能:
    1. 会话生命周期: start -> active -> paused -> active (可循环) -> closed (终态)
    2. 检查点管理: 手动添加检查点, 每个检查点含 seq / ts / SHA-256 哈希
    3. 活跃会话查询: 按 learner_id 过滤 status="active" 的会话
    4. 依赖注入: 通过 L2Store 解耦持久化, store=None 时内部创建 InMemoryL2Store

    Args:
        store: L2 存储实例 (依赖注入). 为 None 时内部创建 InMemoryL2Store.

    Attributes:
        store: 实际使用的存储实例.

    线程安全: threading.RLock 保护内部 learner 索引; store 自身线程安全.
    """

    def __init__(self, store: L2Store | None = None) -> None:
        # 依赖注入: store 为 None 时内部创建 InMemoryL2Store
        self._store: L2Store = store if store is not None else InMemoryL2Store()
        self._lock = threading.RLock()
        # learner_id -> [session_id] 索引 (用于活跃会话查询, 参考 L1 _user_index 模式)
        self._learner_sessions: dict[str, list[str]] = {}
        # session_id -> 累计步骤计数 (record_step 递增, 每 N 步自动检查点)
        self._step_counter: dict[str, int] = {}
        # session_id -> 最后活动时间戳 (cleanup_expired_sessions 据此判定超时)
        self._last_activity: dict[str, float] = {}

    # --- 属性 ---

    @property
    def store(self) -> L2Store:
        """实际使用的存储实例."""
        return self._store

    # --- 会话创建与查询 ---

    def start_session(self, learner_id: str, goal: str = "") -> str:
        """创建新的个性化学习会话.

        创建 SessionRecord (started_at=time.time(), status="active"), 自动保存到 store,
        并注册到 learner 索引. goal 非空时写入 context_envelope["goal"].

        Args:
            learner_id: 学习者 ID
            goal: 学习目标 (可选, 非空时写入 context_envelope)

        Returns:
            新创建的 session_id (格式: "sess-" + uuid hex)

        Raises:
            StoreError: store 写入失败
        """
        session_id = f"{SESSION_ID_PREFIX}-{uuid.uuid4().hex[:12]}"
        started_at = time.time()
        context_envelope: dict[str, Any] | None = None
        if goal:
            context_envelope = {"goal": goal}

        session = SessionRecord(
            session_id=session_id,
            learner_id=learner_id,
            started_at=started_at,
            status=STATUS_ACTIVE,
            context_envelope=context_envelope,
            checkpoints=[],
        )
        self._store.save_session(session_id, session)

        with self._lock:
            self._learner_sessions.setdefault(learner_id, []).append(session_id)
            # 初始化步骤计数与最后活动时间 (供 record_step / cleanup 使用)
            self._step_counter.setdefault(session_id, 0)
            self._last_activity[session_id] = started_at

        return session_id

    def get_session(self, session_id: str) -> SessionRecord | None:
        """获取会话记录.

        Args:
            session_id: 会话 ID

        Returns:
            SessionRecord, 不存在返回 None
        """
        return self._store.get_session(session_id)

    def get_active_sessions(self, learner_id: str) -> list[SessionRecord]:
        """获取学习者的活跃会话 (status="active").

        遍历该学习者的所有会话 (经 learner 索引), 从 store 取回后过滤 status="active".

        Args:
            learner_id: 学习者 ID

        Returns:
            活跃会话列表 (status="active"), 无活跃会话时返回空列表
        """
        with self._lock:
            session_ids = list(self._learner_sessions.get(learner_id, []))

        active: list[SessionRecord] = []
        for sid in session_ids:
            session = self._store.get_session(sid)
            if session is not None and session.status == STATUS_ACTIVE:
                active.append(session)
        return active

    # --- 生命周期管理 ---

    def end_session(self, session_id: str) -> dict[str, Any]:
        """结束会话 (终态).

        设置 status="closed", 计算持续时间, 返回会话摘要字典.
        仅允许从 active / paused 转移到 closed (Activity 状态机校验).

        Args:
            session_id: 会话 ID

        Returns:
            会话摘要: {session_id, learner_id, duration, status: "closed"}

        Raises:
            StoreError: 会话不存在
            ValueError: 当前状态不允许转移到 closed (非法状态转移)
        """
        with self._lock:
            session = self._get_session_or_raise(session_id)
            self._validate_transition(session.status, STATUS_CLOSED)
            now = time.time()
            duration = now - session.started_at
            session.status = STATUS_CLOSED
            self._store.save_session(session_id, session)

        return {
            "session_id": session_id,
            "learner_id": session.learner_id,
            "duration": duration,
            "status": STATUS_CLOSED,
        }

    def pause_session(self, session_id: str) -> SessionRecord:
        """暂停会话 (status -> "paused").

        仅允许从 active 转移到 paused (Activity 状态机校验).

        Args:
            session_id: 会话 ID

        Returns:
            更新后的 SessionRecord

        Raises:
            StoreError: 会话不存在
            ValueError: 当前状态不允许转移到 paused (非法状态转移)
        """
        with self._lock:
            session = self._get_session_or_raise(session_id)
            self._validate_transition(session.status, STATUS_PAUSED)
            session.status = STATUS_PAUSED
            self._store.save_session(session_id, session)
            return session

    def resume_session(self, session_id: str) -> SessionRecord:
        """恢复会话 (status -> "active").

        仅允许从 paused 转移到 active (Activity 状态机校验).

        Args:
            session_id: 会话 ID

        Returns:
            更新后的 SessionRecord

        Raises:
            StoreError: 会话不存在
            ValueError: 当前状态不允许转移到 active (非法状态转移)
        """
        with self._lock:
            session = self._get_session_or_raise(session_id)
            self._validate_transition(session.status, STATUS_ACTIVE)
            session.status = STATUS_ACTIVE
            self._store.save_session(session_id, session)
            # 恢复后刷新最后活动时间 (避免刚恢复即被 cleanup 关闭)
            self._last_activity[session_id] = time.time()
            return session

    # --- 检查点管理 ---

    def add_checkpoint(
        self, session_id: str, checkpoint: dict[str, Any]
    ) -> SessionRecord:
        """为会话添加检查点.

        将检查点追加到 SessionRecord.checkpoints 列表. 检查点会被增强:
        - seq: 检查点序号 (从 0 递增)
        - ts: 检查点时间戳
        - sha256: 检查点内容的 SHA-256 哈希 (完整性验证, Claude Science Checkpoint 机制)

        原始 checkpoint 字段全部保留; 调用方传入的 dict 不会被修改 (浅拷贝).

        Args:
            session_id: 会话 ID
            checkpoint: 检查点内容字典

        Returns:
            更新后的 SessionRecord

        Raises:
            StoreError: 会话不存在
        """
        with self._lock:
            session = self._get_session_or_raise(session_id)
            # 浅拷贝, 避免修改调用方传入的 dict
            enriched = dict(checkpoint)
            enriched["seq"] = len(session.checkpoints)
            enriched["ts"] = time.time()
            enriched["sha256"] = self._compute_sha256(checkpoint)
            session.checkpoints.append(enriched)
            self._store.save_session(session_id, session)
            return session

    def record_step(
        self,
        session_id: str,
        context: dict[str, Any] | None = None,
    ) -> SessionRecord | None:
        """记录会话步骤, 每 ``DEFAULT_CHECKPOINT_INTERVAL`` 步自动创建检查点.

        每次调用递增该会话的步骤计数器, 并刷新最后活动时间戳;
        当步骤计数为 ``DEFAULT_CHECKPOINT_INTERVAL`` 的整数倍时,
        自动调用 ``add_checkpoint`` 创建检查点 (传入 ``context``).

        不存在的会话将抛出 ``StoreError``.

        Args:
            session_id: 会话 ID.
            context: 检查点上下文 (仅在创建检查点时使用), 默认 None -> 空字典.

        Returns:
            创建检查点时返回更新后的 ``SessionRecord``; 否则返回 None.

        Raises:
            StoreError: 会话不存在.
        """
        with self._lock:
            # 校验会话存在 (不存在抛 StoreError)
            self._get_session_or_raise(session_id)
            # 递增步骤计数
            self._step_counter.setdefault(session_id, 0)
            self._step_counter[session_id] += 1
            # 刷新最后活动时间
            self._last_activity[session_id] = time.time()
            # 每 N 步自动创建检查点 (add_checkpoint 内部亦持锁, RLock 可重入)
            if self._step_counter[session_id] % DEFAULT_CHECKPOINT_INTERVAL == 0:
                return self.add_checkpoint(session_id, context or {})
        return None

    # --- 会话分叉与检查点恢复 (Claude Science: Fork + Checkpoint) ---

    def fork_session(
        self,
        session_id: str,
        fork_reason: str = "branch",
        branch_label: str | None = None,
    ) -> SessionRecord:
        """分叉会话 — 创建继承源会话状态的新会话 (Claude Science: Session Fork).

        新会话继承源会话的 learner_id / context_envelope / checkpoints,
        并以 status="active" 启动. 分叉元数据写入新会话的 context_envelope:
        - ``fork_reason``: 分叉原因 (默认 "branch")
        - ``forked_from``: 源会话 ID
        - ``forked_at``: 分叉时间戳
        - ``branch_label``: 分支标签 (可选, 默认 None)
        - ``fork_base_checkpoint_count``: 继承的检查点基线长度 (供 merge_fork 使用)

        不修改源会话 (函数式风格, checkpoints 浅拷贝).

        Args:
            session_id: 源会话 ID
            fork_reason: 分叉原因, 默认 "branch"
            branch_label: 分支标签 (可选)

        Returns:
            新创建的 SessionRecord (status="active")

        Raises:
            StoreError: 源会话不存在
        """
        with self._lock:
            source = self._get_session_or_raise(session_id)
            new_session_id = f"{SESSION_ID_PREFIX}-{uuid.uuid4().hex[:12]}"
            forked_at = time.time()

            # 继承 context_envelope (浅拷贝源信封后追加分叉元数据)
            if source.context_envelope is not None:
                env: dict[str, Any] = dict(source.context_envelope)
            else:
                env = {}
            env["fork_reason"] = fork_reason
            env["forked_from"] = session_id
            env["forked_at"] = forked_at
            env["branch_label"] = branch_label
            env[_FORK_BASE_LEN_KEY] = len(source.checkpoints)

            new_session = SessionRecord(
                session_id=new_session_id,
                learner_id=source.learner_id,
                started_at=forked_at,
                status=STATUS_ACTIVE,
                context_envelope=env,
                checkpoints=list(source.checkpoints),
            )
            self._store.save_session(new_session_id, new_session)
            self._learner_sessions.setdefault(source.learner_id, []).append(
                new_session_id
            )
            # 初始化步骤计数与最后活动时间
            self._step_counter.setdefault(new_session_id, 0)
            self._last_activity[new_session_id] = forked_at
            return new_session

    def restore_checkpoint(
        self, session_id: str, checkpoint_seq: int
    ) -> SessionRecord:
        """恢复会话到指定检查点 (Temporal: 检查点恢复).

        将会话状态回滚到 ``checkpoint_seq`` 对应的检查点: 保留该检查点及其之前的
        所有检查点, 截断其后的检查点.

        Args:
            session_id: 会话 ID
            checkpoint_seq: 要恢复到的检查点序号

        Returns:
            恢复后的 SessionRecord

        Raises:
            StoreError: 会话不存在
            ValueError: 指定 seq 的检查点不存在
        """
        with self._lock:
            session = self._get_session_or_raise(session_id)
            target_index: int | None = None
            for i, cp in enumerate(session.checkpoints):
                if cp.get("seq") == checkpoint_seq:
                    target_index = i
                    break
            if target_index is None:
                raise ValueError(
                    f"检查点不存在: seq={checkpoint_seq} (会话 {session_id})"
                )
            # 截断: 保留目标检查点及其之前的部分
            session.checkpoints = list(session.checkpoints[: target_index + 1])
            self._store.save_session(session_id, session)
            return session

    def merge_fork(
        self, fork_session_id: str, target_session_id: str
    ) -> SessionRecord:
        """合并分叉会话回目标会话 (Claude Science: Fork 合并).

        将分叉会话在分叉后新增的检查点合并回目标会话 (seq 从目标现有数量续编),
        并将分叉会话标记为 ``status="merged"`` (终态).

        仅合并分叉后新增的检查点 (基于 fork 时记录的基线长度), 避免重复合并
        继承的检查点.

        Args:
            fork_session_id: 分叉会话 ID
            target_session_id: 目标会话 ID

        Returns:
            合并后的目标 SessionRecord

        Raises:
            StoreError: 分叉会话或目标会话不存在
        """
        with self._lock:
            fork_session = self._get_session_or_raise(fork_session_id)
            target_session = self._get_session_or_raise(target_session_id)

            # 分叉时继承的检查点基线长度 (缺省 0, 兼容旧分叉)
            base_len = 0
            if fork_session.context_envelope is not None:
                base_len = int(
                    fork_session.context_envelope.get(_FORK_BASE_LEN_KEY, 0)
                )
            # 仅合并分叉后新增的检查点
            new_checkpoints = list(fork_session.checkpoints[base_len:])
            next_seq = len(target_session.checkpoints)
            for cp in new_checkpoints:
                merged_cp = dict(cp)
                merged_cp["seq"] = next_seq
                next_seq += 1
                target_session.checkpoints.append(merged_cp)

            # 标记分叉会话为 merged (终态, 绕过标准状态机校验)
            fork_session.status = STATUS_MERGED
            self._store.save_session(fork_session_id, fork_session)
            self._store.save_session(target_session_id, target_session)
            return target_session

    # --- 会话超时清理与统计 ---

    def cleanup_expired_sessions(self, timeout: float | None = None) -> list[str]:
        """清理超时会话, 将超时的 active 会话转为 closed.

        遍历所有已注册会话, 对 ``status="active"`` 且最后活动时间距当前
        超过 ``timeout`` 秒的会话执行关闭操作 (status -> "closed").
        已关闭 / 已暂停 / 已合并的会话不受影响.

        Args:
            timeout: 超时阈值 (秒); 为 None 时使用 ``DEFAULT_SESSION_TIMEOUT`` (1800s).

        Returns:
            被清理 (关闭) 的 session_id 列表.
        """
        if timeout is None:
            timeout = DEFAULT_SESSION_TIMEOUT
        now = time.time()
        expired: list[str] = []

        with self._lock:
            # 收集所有已注册的 session_id (跨所有 learner)
            all_session_ids: list[str] = []
            for sids in self._learner_sessions.values():
                all_session_ids.extend(sids)

            for sid in all_session_ids:
                session = self._store.get_session(sid)
                if session is None:
                    continue
                # 仅清理 active 会话 (closed / paused / merged 不受影响)
                if session.status != STATUS_ACTIVE:
                    continue
                last_activity = self._last_activity.get(sid, session.started_at)
                if (now - last_activity) > timeout:
                    # active -> closed (合法转移)
                    session.status = STATUS_CLOSED
                    self._store.save_session(sid, session)
                    expired.append(sid)

        return expired

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """获取会话统计信息.

        返回包含会话元数据与运行时统计的字典:
        - ``session_id``: 会话 ID
        - ``learner_id``: 学习者 ID
        - ``status``: 当前状态
        - ``step_count``: 累计步骤数 (record_step 调用次数)
        - ``checkpoint_count``: 检查点数量
        - ``duration``: 会话持续时间 (秒, now - started_at)
        - ``started_at``: 会话创建时间戳

        Args:
            session_id: 会话 ID.

        Returns:
            统计信息字典.

        Raises:
            StoreError: 会话不存在.
        """
        with self._lock:
            session = self._get_session_or_raise(session_id)
            now = time.time()
            return {
                "session_id": session_id,
                "learner_id": session.learner_id,
                "status": session.status,
                "step_count": self._step_counter.get(session_id, 0),
                "checkpoint_count": len(session.checkpoints),
                "duration": now - session.started_at,
                "started_at": session.started_at,
            }

    # --- 内部方法 ---

    @staticmethod
    def is_valid_transition(from_status: str, to_status: str) -> bool:
        """判断状态转移是否合法 (Activity 状态机).

        合法转移:
        - created -> active
        - active  -> paused / closed
        - paused  -> active / closed

        closed / merged 为终态, 不允许任何后续转移.

        Args:
            from_status: 起始状态
            to_status: 目标状态

        Returns:
            True 表示转移合法, False 表示非法
        """
        return to_status in _VALID_TRANSITIONS.get(from_status, frozenset())

    def _validate_transition(self, from_status: str, to_status: str) -> None:
        """校验状态转移, 非法时抛出 ValueError.

        Args:
            from_status: 起始状态
            to_status: 目标状态

        Raises:
            ValueError: 状态转移非法 (不在 ``_VALID_TRANSITIONS`` 中)
        """
        if not self.is_valid_transition(from_status, to_status):
            raise ValueError(
                f"非法状态转移: {from_status} -> {to_status} "
                f"(合法转移: created->active, active->paused/closed, "
                f"paused->active/closed)"
            )

    def _get_session_or_raise(self, session_id: str) -> SessionRecord:
        """获取会话, 不存在时抛出 StoreError.

        Args:
            session_id: 会话 ID

        Returns:
            SessionRecord

        Raises:
            StoreError: 会话不存在
        """
        session = self._store.get_session(session_id)
        if session is None:
            raise StoreError(
                detail=f"会话不存在: {session_id}",
                context={
                    "session_id": session_id,
                    "operation": "get_session",
                },
            )
        return session

    @staticmethod
    def _compute_sha256(checkpoint: dict[str, Any]) -> str:
        """计算检查点内容的 SHA-256 哈希.

        使用 JSON 序列化 (sort_keys 确保确定性) 后计算哈希, 用于完整性验证.

        Args:
            checkpoint: 检查点内容字典

        Returns:
            SHA-256 十六进制摘要 (64 字符)
        """
        payload = json.dumps(checkpoint, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================
# __all__
# ============================================================

__all__ = [
    "SessionManager",
    "SESSION_ID_PREFIX",
    "DEFAULT_CHECKPOINT_INTERVAL",
    "DEFAULT_SESSION_TIMEOUT",
    "STATUS_CREATED",
    "STATUS_ACTIVE",
    "STATUS_PAUSED",
    "STATUS_CLOSED",
    "STATUS_MERGED",
]
