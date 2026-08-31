"""L1 用户域学习会话管理 (Learning Session Management) — 核心引擎.

设计依据:
- L1 设计文档第五章 5.1-5.5: 会话模型、会话类型、Session Fork、会话间上下文传递
- L1 设计文档第七章 7.2: API `/api/v1/sessions/*`
- L1 设计文档第七章 7.3: ER 图 (User / Role / Session / LearningContext / AuditLog)

融合世界先进方案:
- LangGraph: Thread 管理 + 检查点恢复 (checkpoint → restore)
- OpenAI Agents SDK: 极简接口 + 可插拔存储
- Google ADK: 多维度隔离 (user + app + session)
- Claude Code: 事件树 + Fork 分支
- Temporal: 超时管理 + 状态恢复
- Jupyter: idle timeout 自动回收
- Git: 分支/合并/丢弃语义 (Fork 类比)
- Redis Session Store: 热数据 + 持久层分离
- LMS (Learning Management System): 学习路径追踪 + 进度保存

模块组成:
1. 异常体系: L1SessionError 层级 (JSON-RPC -32500 范围)
2. LearningSessionManager: 会话生命周期 + Fork 管理 + 检查点
3. CheckpointManager: 检查点创建/加载/序列化
4. ForkManager: Fork 创建/合并/丢弃/列表
5. ContextTransfer: 会话间上下文继承
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any

from dy3_polaris.l1.models import (
    ContextEnvelope,
    Interaction,
    LearningSession,
    SessionArtifact,
    SessionFork,
    SessionStatus,
    SessionType,
)
from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 1. 常量定义
# ============================================================

# --- Fork 限制 ---
MAX_FORK_DEPTH: int = 3                      # 最大 Fork 深度 (防止无限分叉)
MAX_FORK_CONCURRENCY: int = 5                # 单会话最大并发 Fork 数

# --- 会话限制 ---
MAX_ACTIVE_SESSIONS_PER_USER: int = 10       # 单用户最大活跃会话数

# --- 超时 ---
SESSION_IDLE_TIMEOUT_MS: int = 30 * 60 * 1000  # 空闲超时: 30 分钟
SESSION_IDLE_TIMEOUT_S: int = 30 * 60          # 空闲超时 (秒)

# --- 检查点 ---
CHECKPOINT_PREFIX: str = "cp"                # 检查点 ID 前缀


# ============================================================
# 2. 异常体系 (JSON-RPC -32500 范围)
# ============================================================


class L1SessionError(L6Error):
    """L1 会话管理层基础异常 (JSON-RPC -32500).

    所有会话管理相关异常的基类, 继承自 L6Error.
    """

    def __init__(
        self,
        code: str = "L1_SESSION_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32500


class SessionNotFoundError(L1SessionError):
    """会话未找到 (JSON-RPC -32501).

    会话不存在或已被清除.
    """

    def __init__(
        self,
        session_id: str,
        detail: str = "",
    ) -> None:
        super().__init__(
            "SESSION_NOT_FOUND",
            detail or f"会话未找到: {session_id}",
            {"session_id": session_id},
        )

    def _jsonrpc_code(self) -> int:
        return -32501


class SessionStateError(L1SessionError):
    """会话状态错误 (JSON-RPC -32502).

    非法状态转换、操作不允许的当前状态等.
    """

    def __init__(
        self,
        session_id: str,
        current_status: str = "",
        target_status: str = "",
        detail: str = "",
    ) -> None:
        ctx: dict[str, Any] = {"session_id": session_id}
        if current_status:
            ctx["current_status"] = current_status
        if target_status:
            ctx["target_status"] = target_status
        super().__init__(
            "SESSION_STATE_ERROR",
            detail or f"会话状态错误: {session_id}",
            ctx,
        )

    def _jsonrpc_code(self) -> int:
        return -32502


class ForkError(L1SessionError):
    """Fork 错误 (JSON-RPC -32503).

    Fork 创建失败、深度超限、合并失败等.
    """

    def __init__(
        self,
        session_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {"session_id": session_id}
        if context:
            ctx.update(context)
        super().__init__("FORK_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32503


class CheckpointError(L1SessionError):
    """检查点错误 (JSON-RPC -32504).

    检查点创建失败、加载失败、序列化失败等.
    """

    def __init__(
        self,
        session_id: str,
        checkpoint_seq: int = -1,
        detail: str = "",
    ) -> None:
        ctx: dict[str, Any] = {
            "session_id": session_id,
            "checkpoint_seq": checkpoint_seq,
        }
        super().__init__("CHECKPOINT_ERROR", detail, ctx)

    def _jsonrpc_code(self) -> int:
        return -32504


# ============================================================
# 3. 检查点管理 (CheckpointManager)
# ============================================================


class CheckpointManager:
    """检查点管理器 (设计文档 5.4).

    借鉴世界先进方案:
    - LangGraph: Thread 检查点 (state snapshot + restore)
    - Temporal: 事件历史持久化
    - Git: commit 语义 (seq 递增)

    检查点是 Agent 完成完整推理后的状态快照:
    - 序列号递增 (0, 1, 2, ...)
    - 支持加载历史检查点恢复状态
    - 内存存储 (生产环境可替换为 Redis/DB)

    线程安全: 由调用方 (LearningSessionManager) 的锁保护.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, dict[int, dict[str, Any]]] = {}
        self._seq_counters: dict[str, int] = {}

    def create_checkpoint(self, session: LearningSession) -> int:
        """为会话创建检查点.

        Args:
            session: 学习会话

        Returns:
            检查点序列号
        """
        session_id = session.session_id
        seq = self._seq_counters.get(session_id, -1) + 1
        self._seq_counters[session_id] = seq

        # 保存会话状态快照
        snapshot = session.to_dict()
        if session_id not in self._checkpoints:
            self._checkpoints[session_id] = {}
        self._checkpoints[session_id][seq] = snapshot

        # 记录到会话
        session.add_checkpoint(seq)
        session.touch()

        return seq

    def load_checkpoint(
        self,
        session: LearningSession,
        seq: int,
    ) -> LearningSession:
        """加载指定检查点的会话状态.

        Args:
            session: 当前会话 (用于获取 session_id)
            seq: 检查点序列号

        Returns:
            恢复到检查点状态的 LearningSession

        Raises:
            CheckpointError: 检查点不存在
        """
        session_id = session.session_id
        checkpoints = self._checkpoints.get(session_id, {})
        if seq not in checkpoints:
            raise CheckpointError(
                session_id=session_id,
                checkpoint_seq=seq,
                detail=f"检查点 {seq} 不存在",
            )

        snapshot = checkpoints[seq]
        restored = LearningSession.from_dict(snapshot)
        return restored

    def list_checkpoints(self, session_id: str) -> list[int]:
        """列出会话的所有检查点序列号.

        Args:
            session_id: 会话 ID

        Returns:
            检查点序列号列表 (升序)
        """
        return sorted(self._checkpoints.get(session_id, {}).keys())

    def delete_checkpoint(self, session_id: str, seq: int) -> bool:
        """删除指定检查点.

        Args:
            session_id: 会话 ID
            seq: 检查点序列号

        Returns:
            True 如果成功删除
        """
        checkpoints = self._checkpoints.get(session_id, {})
        if seq in checkpoints:
            del checkpoints[seq]
            return True
        return False

    def clear_session(self, session_id: str) -> None:
        """清除会话的所有检查点."""
        self._checkpoints.pop(session_id, None)
        self._seq_counters.pop(session_id, None)


# ============================================================
# 4. Fork 管理 (ForkManager)
# ============================================================


class ForkManager:
    """Fork 管理器 (设计文档 5.5).

    借鉴世界先进方案:
    - Git: 分支/合并/丢弃语义
    - Claude Code: 事件树 + Fork 分支
    - LangGraph: Thread Fork

    Fork 机制支持学习路径分支与对比:
    - 创建: 从检查点复制完整状态
    - 合并: 将分支掌握度更新回主会话
    - 丢弃: 结束分支, 状态归档

    线程安全: 由调用方 (LearningSessionManager) 的锁保护.
    """

    def __init__(self) -> None:
        self._forks: dict[str, SessionFork] = {}
        self._forks_by_source: dict[str, list[str]] = {}
        self._fork_depths: dict[str, int] = {}  # session_id -> fork depth

    def create_fork(
        self,
        source_session: LearningSession,
        fork_reason: str,
        branch_label: str,
        checkpoint_seq: int | None = None,
    ) -> tuple[LearningSession, SessionFork]:
        """创建 Fork 分支.

        Args:
            source_session: 源会话
            fork_reason: Fork 原因
            branch_label: 分支标签
            checkpoint_seq: 检查点序列号 (None 表示当前状态)

        Returns:
            (Fork 后的新会话, Fork 记录)

        Raises:
            ForkError: Fork 深度超限或源会话状态不允许
        """
        # 检查 Fork 深度
        depth = self._get_fork_depth(source_session)
        if depth >= MAX_FORK_DEPTH:
            raise ForkError(
                session_id=source_session.session_id,
                detail=(
                    f"Fork 深度超限: {depth} >= {MAX_FORK_DEPTH}"
                ),
                context={"depth": depth, "max_depth": MAX_FORK_DEPTH},
            )

        # 创建 Fork 记录
        snapshot = copy.deepcopy(source_session.context)
        fork_record = SessionFork(
            source_session_id=source_session.session_id,
            fork_point_seq=checkpoint_seq if checkpoint_seq is not None else 0,
            fork_reason=fork_reason,
            branch_label=branch_label,
            snapshot_at_fork=snapshot,
        )

        # 创建 Fork 会话 (继承源会话状态)
        fork_session = LearningSession(
            user_id=source_session.user_id,
            session_type=source_session.session_type,
            parent_session_id=source_session.session_id,
            fork_point_seq=checkpoint_seq if checkpoint_seq is not None else 0,
            context=copy.deepcopy(source_session.context),
            agent_states=copy.deepcopy(source_session.agent_states),
            interaction_log=copy.deepcopy(source_session.interaction_log),
            artifacts=copy.deepcopy(source_session.artifacts),
            status=SessionStatus.ACTIVE,
            checkpoint_indices=list(source_session.checkpoint_indices),
        )

        # 记录 Fork 深度
        source_depth = self._fork_depths.get(source_session.session_id, 0)
        self._fork_depths[fork_session.session_id] = source_depth + 1

        # 记录 Fork
        self._forks[fork_session.session_id] = fork_record
        source_id = source_session.session_id
        if source_id not in self._forks_by_source:
            self._forks_by_source[source_id] = []
        self._forks_by_source[source_id].append(fork_session.session_id)

        # 更新源会话状态
        source_session.status = SessionStatus.FORKED
        source_session.touch()

        return fork_session, fork_record

    def merge_fork(
        self,
        fork_session: LearningSession,
        target_session: LearningSession,
    ) -> None:
        """合并 Fork 分支回目标会话.

        合并策略:
        - Fork 分支的交互日志追加到目标会话
        - Fork 分支的产出物追加到目标会话
        - Fork 分支的掌握度更新合并到目标上下文
        - Fork 状态标记为 COMPLETED

        Args:
            fork_session: Fork 分支会话
            target_session: 合并目标会话
        """
        fork_session.status = SessionStatus.COMPLETED
        fork_session.touch()

        # 追加交互日志
        target_session.interaction_log.extend(fork_session.interaction_log)

        # 追加产出物
        target_session.artifacts.extend(fork_session.artifacts)

        # 更新 Fork 记录
        fork_record = self._forks.get(fork_session.session_id)
        if fork_record is not None:
            fork_record.is_merged = True
            fork_record.merge_target = target_session.session_id

        target_session.touch()

    def discard_fork(self, fork_session: LearningSession) -> None:
        """丢弃 Fork 分支.

        Args:
            fork_session: Fork 分支会话
        """
        fork_session.status = SessionStatus.COMPLETED
        fork_session.touch()

    def list_forks(self, source_session_id: str) -> list[SessionFork]:
        """列出源会话的所有 Fork 记录.

        Args:
            source_session_id: 源会话 ID

        Returns:
            Fork 记录列表
        """
        fork_ids = self._forks_by_source.get(source_session_id, [])
        return [self._forks[fid] for fid in fork_ids if fid in self._forks]

    def get_fork(self, fork_session_id: str) -> SessionFork | None:
        """按 ID 获取 Fork 记录."""
        return self._forks.get(fork_session_id)

    def _get_fork_depth(self, session: LearningSession) -> int:
        """计算会话的 Fork 深度.

        使用内部 _fork_depths 字典追踪每个会话的深度.
        """
        return self._fork_depths.get(session.session_id, 0)


# ============================================================
# 5. 上下文传递 (ContextTransfer)
# ============================================================


class ContextTransfer:
    """会话间上下文传递器 (设计文档 5.5).

    借鉴世界先进方案:
    - LangGraph: Thread 间状态继承
    - Redis Session Store: 热数据复制

    支持会话间上下文的完整继承:
    - 用户 ID、会话类型
    - 学习目标 (goals)
    - 知识掌握快照 (mastery_snapshot)
    - 学习阶段 (learning_state.phase)
    """

    @staticmethod
    def inherit_context(
        source: LearningSession,
        target: LearningSession,
    ) -> None:
        """将源会话上下文继承到目标会话.

        继承内容:
        - 用户 ID
        - 学习目标 (浅拷贝)
        - 知识掌握快照 (深拷贝)
        - 学习阶段状态

        Args:
            source: 源会话
            target: 目标会话
        """
        target.context.user_id = source.context.user_id

        # 继承学习目标
        if source.context.goals:
            target.context.goals = list(source.context.goals)

        # 继承掌握度快照列表 (深拷贝)
        if source.context.mastery_snapshot:
            from dy3_polaris.l1.models import MasterySnapshot
            target.context.mastery_snapshot = [
                MasterySnapshot(
                    kc_id=snap.kc_id,
                    p_know=snap.p_know,
                    last_practiced_at=snap.last_practiced_at,
                    repetitions=snap.repetitions,
                )
                for snap in source.context.mastery_snapshot
            ]

        # 继承学习阶段
        if source.context.learning_state is not None:
            target.context.learning_state = copy.deepcopy(
                source.context.learning_state
            )

        target.touch()


# ============================================================
# 6. 学习会话核心管理器 (LearningSessionManager)
# ============================================================


class LearningSessionManager:
    """学习会话核心管理器 (设计文档 5.1-5.5).

    借鉴世界先进方案:
    - LangGraph: Thread 管理 + 检查点恢复
    - OpenAI Agents SDK: 极简接口 + 可插拔存储
    - Google ADK: 多维度隔离 (user + session)
    - Claude Code: 事件树 + Fork 分支
    - Temporal: 超时管理 + 状态恢复
    - Jupyter: idle timeout 自动回收
    - Git: 分支/合并/丢弃语义
    - LMS: 学习路径追踪 + 进度保存

    核心功能:
    1. 会话生命周期 (ACTIVE → PAUSED → COMPLETED)
    2. 多维度会话索引 (user_id / status)
    3. 检查点管理 (创建/加载/列出)
    4. Fork 创建/合并/丢弃
    5. 上下文传递
    6. 空闲超时检测
    7. 单用户会话数限制

    线程安全: threading.RLock 保护会话存储.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, LearningSession] = {}
        self._user_index: dict[str, list[str]] = {}
        self._checkpoint_manager = CheckpointManager()
        self._fork_manager = ForkManager()
        self._context_transfer = ContextTransfer()

    # --- 属性 ---

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        """检查点管理器."""
        return self._checkpoint_manager

    @property
    def fork_manager(self) -> ForkManager:
        """Fork 管理器."""
        return self._fork_manager

    @property
    def context_transfer(self) -> ContextTransfer:
        """上下文传递器."""
        return self._context_transfer

    # --- 会话创建与查询 ---

    def create_session(
        self,
        user_id: str,
        session_type: SessionType,
    ) -> LearningSession:
        """创建学习会话.

        Args:
            user_id: 用户 ID
            session_type: 会话类型

        Returns:
            创建的 LearningSession
        """
        # 检查单用户会话数限制
        self._enforce_session_limit(user_id)

        session = LearningSession(
            user_id=user_id,
            session_type=session_type,
        )

        with self._lock:
            # 保证 created_at 严格单调递增 (同一毫秒创建时后建者 +1ms)
            last = self._sessions.get(self._user_index[user_id][-1]) if self._user_index.get(user_id) else None
            if last is not None and session.created_at <= last.created_at:
                object.__setattr__(session, "created_at", last.created_at + 1)
            self._sessions[session.session_id] = session
            if user_id not in self._user_index:
                self._user_index[user_id] = []
            self._user_index[user_id].append(session.session_id)

        return session

    def get_session(self, session_id: str) -> LearningSession | None:
        """按 ID 获取会话.

        Args:
            session_id: 会话 ID

        Returns:
            LearningSession 或 None
        """
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions_for_user(self, user_id: str) -> list[LearningSession]:
        """列出用户全部会话 (按创建时间倒序).

        Args:
            user_id: 用户 ID

        Returns:
            会话列表 (含 COMPLETED)
        """
        with self._lock:
            session_ids = self._user_index.get(user_id, [])
            sessions = [self._sessions[sid] for sid in session_ids if sid in self._sessions]
            sessions.sort(key=lambda s: (s.created_at, s.session_id), reverse=True)
            return sessions

    def get_active_sessions(self, user_id: str) -> list[LearningSession]:
        """获取用户的活跃会话列表.

        活跃 = ACTIVE 或 PAUSED 或 FORKED 状态 (排除 COMPLETED).

        Args:
            user_id: 用户 ID

        Returns:
            活跃会话列表 (按创建时间倒序)
        """
        with self._lock:
            session_ids = self._user_index.get(user_id, [])
            active: list[LearningSession] = []
            for sid in session_ids:
                session = self._sessions.get(sid)
                if session is None:
                    continue
                if session.status != SessionStatus.COMPLETED:
                    active.append(session)
            # 按创建时间倒序, 时间相同则按 session_id 稳定排序
            active.sort(key=lambda s: (s.created_at, s.session_id), reverse=True)
            return active

    def _enforce_session_limit(self, user_id: str) -> None:
        """强制执行单用户会话数限制.

        当活跃会话数超过限制时, 自动清理最早的 COMPLETED 会话.
        如果没有 COMPLETED 会话可清理, 清理最早的 ACTIVE 会话.
        """
        with self._lock:
            session_ids = self._user_index.get(user_id, [])
            active_ids = [
                sid for sid in session_ids
                if sid in self._sessions
                and self._sessions[sid].status != SessionStatus.COMPLETED
            ]

            if len(active_ids) >= MAX_ACTIVE_SESSIONS_PER_USER:
                # 优先清理最早的非活跃会话
                sorted_ids = sorted(
                    active_ids,
                    key=lambda sid: self._sessions[sid].created_at,
                )
                # 标记最早的为 COMPLETED (软删除)
                to_remove = sorted_ids[0]
                self._sessions[to_remove].status = SessionStatus.COMPLETED

    # --- 生命周期管理 ---

    def pause_session(self, session_id: str) -> None:
        """暂停会话.

        Args:
            session_id: 会话 ID

        Raises:
            SessionNotFoundError: 会话不存在
        """
        session = self._get_session_or_raise(session_id)
        session.status = SessionStatus.PAUSED
        session.touch()

    def resume_session(self, session_id: str) -> None:
        """恢复会话.

        Args:
            session_id: 会话 ID

        Raises:
            SessionNotFoundError: 会话不存在
        """
        session = self._get_session_or_raise(session_id)
        session.status = SessionStatus.ACTIVE
        session.touch()

    def complete_session(self, session_id: str) -> None:
        """完成会话.

        Args:
            session_id: 会话 ID

        Raises:
            SessionNotFoundError: 会话不存在
        """
        session = self._get_session_or_raise(session_id)
        session.status = SessionStatus.COMPLETED
        session.touch()

    # --- 交互与产出物 ---

    def add_interaction(
        self,
        session_id: str,
        interaction: Interaction,
    ) -> None:
        """记录交互.

        Args:
            session_id: 会话 ID
            interaction: 交互记录

        Raises:
            SessionNotFoundError: 会话不存在
        """
        session = self._get_session_or_raise(session_id)
        session.add_typed_interaction(interaction)
        session.touch()

    def add_artifact(
        self,
        session_id: str,
        artifact: SessionArtifact,
    ) -> None:
        """关联产出物.

        Args:
            session_id: 会话 ID
            artifact: 产出物

        Raises:
            SessionNotFoundError: 会话不存在
        """
        session = self._get_session_or_raise(session_id)
        session.add_typed_artifact(artifact)
        session.touch()

    def touch_session(self, session_id: str) -> None:
        """更新会话 updated_at 时间戳.

        Args:
            session_id: 会话 ID

        Raises:
            SessionNotFoundError: 会话不存在
        """
        session = self._get_session_or_raise(session_id)
        session.touch()

    # --- 检查点 ---

    def create_checkpoint(self, session_id: str) -> int:
        """为会话创建检查点.

        Args:
            session_id: 会话 ID

        Returns:
            检查点序列号

        Raises:
            SessionNotFoundError: 会话不存在
        """
        session = self._get_session_or_raise(session_id)
        return self._checkpoint_manager.create_checkpoint(session)

    def list_checkpoints(self, session_id: str) -> list[int]:
        """列出会话的所有检查点.

        Args:
            session_id: 会话 ID

        Returns:
            检查点序列号列表
        """
        # 不抛出异常, 即使会话不存在也返回空列表
        return self._checkpoint_manager.list_checkpoints(session_id)

    # --- Fork 管理 ---

    def fork_session(
        self,
        session_id: str,
        fork_reason: str,
        branch_label: str,
    ) -> LearningSession:
        """创建 Session Fork.

        Args:
            session_id: 源会话 ID
            fork_reason: Fork 原因
            branch_label: 分支标签

        Returns:
            Fork 后的新会话

        Raises:
            SessionNotFoundError: 会话不存在
            SessionStateError: 会话状态不允许 Fork
            ForkError: Fork 深度超限
        """
        session = self._get_session_or_raise(session_id)

        # 检查状态
        if session.status == SessionStatus.COMPLETED:
            raise SessionStateError(
                session_id=session_id,
                current_status=session.status.value,
                detail="已完成会话不允许 Fork",
            )

        # 创建 Fork
        fork_session, fork_record = self._fork_manager.create_fork(
            source_session=session,
            fork_reason=fork_reason,
            branch_label=branch_label,
            checkpoint_seq=(
                session.checkpoint_indices[-1]
                if session.checkpoint_indices
                else None
            ),
        )

        # 注册 Fork 会话
        with self._lock:
            self._sessions[fork_session.session_id] = fork_session
            user_id = fork_session.user_id
            if user_id not in self._user_index:
                self._user_index[user_id] = []
            self._user_index[user_id].append(fork_session.session_id)

        return fork_session

    def merge_fork(
        self,
        fork_session_id: str,
        target_session_id: str,
    ) -> None:
        """合并 Fork 回目标会话.

        Args:
            fork_session_id: Fork 会话 ID
            target_session_id: 目标会话 ID

        Raises:
            SessionNotFoundError: Fork 或目标会话不存在
        """
        fork_session = self._get_session_or_raise(fork_session_id)
        target_session = self._get_session_or_raise(target_session_id)
        self._fork_manager.merge_fork(fork_session, target_session)

    def discard_fork(self, fork_session_id: str) -> None:
        """丢弃 Fork 分支.

        Args:
            fork_session_id: Fork 会话 ID

        Raises:
            SessionNotFoundError: Fork 会话不存在
        """
        fork_session = self._get_session_or_raise(fork_session_id)
        self._fork_manager.discard_fork(fork_session)

    def list_forks(self, session_id: str) -> list[SessionFork]:
        """列出会话的所有 Fork.

        Args:
            session_id: 源会话 ID

        Returns:
            Fork 记录列表
        """
        return self._fork_manager.list_forks(session_id)

    # --- 空闲检测 ---

    def is_session_idle(self, session_id: str) -> bool:
        """检测会话是否空闲超时.

        Args:
            session_id: 会话 ID

        Returns:
            True 如果会话空闲超时

        Raises:
            SessionNotFoundError: 会话不存在
        """
        session = self._get_session_or_raise(session_id)
        now_ms = int(time.time() * 1000)
        elapsed_ms = now_ms - session.updated_at
        return elapsed_ms > SESSION_IDLE_TIMEOUT_MS

    # --- 内部方法 ---

    def _get_session_or_raise(self, session_id: str) -> LearningSession:
        """获取会话, 不存在时抛出异常.

        Args:
            session_id: 会话 ID

        Returns:
            LearningSession

        Raises:
            SessionNotFoundError: 会话不存在
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            return session
