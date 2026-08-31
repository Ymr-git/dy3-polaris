"""会话管理与 Fork 模块 — L5 Agent Runtime 核心组件.

融合世界先进方案:
- LangGraph: Thread/Checkpoint 模式 + 超步级快照
- OpenAI Agents SDK: 极简 Session 接口 + 可插拔存储
- Google ADK: Session/State/Memory 三层上下文分离
- Claude Code: JSONL 事件树 + Fork 分支 + 父子链
- Temporal: 事件历史重放 + continue-as-new + 幂等性
- Jupyter: Kernel 进程级会话隔离

本模块实现:
1. SessionEvent — 不可变事件 (事件溯源基础)
2. EventLog — append-only 事件日志 (Temporal/Claude Code 模式)
3. SessionContext — 三层上下文 (ADK: State + Events + Memory)
4. SessionManager — 会话生命周期管理 (创建/激活/暂停/恢复/关闭)
5. ForkCheckpoint — 四类状态快照 (kernel_state/working_session/agent_outputs/broadcast_queue_state)
6. ForkEvaluator — Fork 效果评估 (learning_gain/completion_time/resource_tokens)
7. SessionCompactor — 上下文压缩 + continue-as-new (Claude Code/Temporal 模式)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# 统一 ID 命名空间 (单点: shared/ids.py)
from dy3_polaris.shared.ids import new_session_id

from .kernel_persistence import (
    CheckpointStore,
    ForkRecord,
    ForkStatus,
    MaxForkConcurrencyError,
    MaxForkDepthError,
    MemoryCheckpointStore,
    SessionForkManager,
)

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class SessionState(str, Enum):
    """会话生命周期状态.

    CREATED → ACTIVE ⇄ PAUSED → CLOSED
                    ↘ ERROR → ACTIVE (恢复)
    """

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    ERROR = "error"


class SessionTier(str, Enum):
    """会话上下文三层分离 (ADK 模式).

    STATE:  可变状态字典 (当前会话数据)
    EVENTS: 不可变事件日志 (append-only 溯源)
    MEMORY: 跨会话记忆 (长期知识)
    """

    STATE = "state"
    EVENTS = "events"
    MEMORY = "memory"


class ForkMergeScope(str, Enum):
    """Fork 合并范围 (L5 设计文档 4.3.2 节)."""

    KERNEL_STATE = "kernel_state"
    WORKING_SESSION = "working_session"
    AGENT_OUTPUTS = "agent_outputs"
    BROADCAST_MESSAGES = "broadcast_messages"


# ============================================================
# 异常定义
# ============================================================


class SessionStateError(Exception):
    """会话状态转换错误."""

    def __init__(self, current: str, target: str) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Invalid session transition: {current} → {target}")


class SessionNotFoundError(Exception):
    """会话不存在."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


# ============================================================
# SessionEvent — 不可变事件 (Claude Code/Temporal 模式)
# ============================================================


class SessionEvent(BaseModel):
    """不可变会话事件 (事件溯源基础).

    融合 Claude Code 的 uuid/parentUuid 树结构 + Temporal 的 append-only 事件历史.

    每个事件记录:
    - event_id: 唯一标识 (用于事件树构建)
    - session_id: 所属会话
    - event_type: 事件类型 (message/state_set/tool_call/...)
    - data: 事件数据
    - parent_event_id: 父事件 ID (用于 Fork 树, Claude Code 模式)
    - timestamp: 事件时间戳
    """

    event_id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    session_id: str = Field(..., description="所属会话 ID")
    event_type: str = Field(..., description="事件类型")
    data: dict[str, Any] = Field(default_factory=dict, description="事件数据")
    parent_event_id: str | None = Field(default=None, description="父事件 ID (Fork 树)")
    timestamp: float = Field(default_factory=time.time, description="事件时间戳")

    model_config = {"frozen": True}


# ============================================================
# EventLog — 不可变事件日志 (Temporal/Claude Code 模式)
# ============================================================


class EventLog:
    """append-only 不可变事件日志.

    融合 Temporal 事件历史重放 + Claude Code JSONL 追加日志.

    核心特性:
    1. append-only: 只能追加，不能删除或修改
    2. replay: 重放事件重建状态
    3. 父子链: 支持事件树 (Fork 分支)
    4. 线程安全: 读写锁保护
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._events: list[SessionEvent] = []
        self._index: dict[str, SessionEvent] = {}
        self._lock = threading.RLock()

    def append(self, event: SessionEvent) -> None:
        """追加事件到日志 (不可删除)."""
        with self._lock:
            self._events.append(event)
            self._index[event.event_id] = event

    def get_events(self, limit: int = 0) -> list[SessionEvent]:
        """获取事件列表 (按时间顺序).

        Args:
            limit: 返回最近 N 条 (0=全部)

        Returns:
            事件列表的副本
        """
        with self._lock:
            if limit > 0:
                return list(self._events[-limit:])
            return list(self._events)

    def get_event(self, event_id: str) -> SessionEvent | None:
        """按 ID 获取事件."""
        with self._lock:
            return self._index.get(event_id)

    def replay(self) -> dict[str, Any]:
        """重放事件重建状态 (Temporal 模式).

        仅处理 state_set 类型的事件，将 key-value 累积到状态字典.
        """
        with self._lock:
            state: dict[str, Any] = {}
            for event in self._events:
                if event.event_type == "state_set":
                    key = event.data.get("key")
                    value = event.data.get("value")
                    if key is not None:
                        state[key] = value
            return state

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


# ============================================================
# SessionContext — 三层上下文 (ADK Session/State/Memory 模式)
# ============================================================


class SessionContext:
    """三层会话上下文 (Google ADK 模式).

    分离三个职责:
    1. state: 可变状态字典 (当前会话数据, 如学习进度/路径)
    2. events: 不可变事件日志 (状态变更溯源, Temporal 模式)
    3. memory: 跨会话记忆 (长期知识, 如学习者偏好)

    设计原理:
    - 状态变更自动记录到事件日志 (审计 + 重放)
    - 记忆独立于状态 (不产生事件, 跨会话保留)
    - 检查点保存完整状态快照 (LangGraph 模式)
    """

    def __init__(
        self,
        session_id: str,
        agent_id: str,
    ) -> None:
        self.session_id = session_id
        self.agent_id = agent_id
        self._state: dict[str, Any] = {}
        self._memory: dict[str, Any] = {}
        self.events: EventLog = EventLog(session_id=session_id)
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @property
    def state(self) -> dict[str, Any]:
        """只读状态字典."""
        with self._lock:
            return dict(self._state)

    @property
    def memory(self) -> dict[str, Any]:
        """只读记忆字典."""
        with self._lock:
            return dict(self._memory)

    def set_state(self, key: str, value: Any) -> None:
        """设置状态 (自动创建事件)."""
        with self._lock:
            self._state[key] = value
            self.events.append(SessionEvent(
                session_id=self.session_id,
                event_type="state_set",
                data={"key": key, "value": value},
            ))

    def get_state(self, key: str) -> Any | None:
        """获取状态."""
        with self._lock:
            return self._state.get(key)

    def set_memory(self, key: str, value: Any) -> None:
        """设置记忆 (不创建事件, 跨会话保留)."""
        with self._lock:
            self._memory[key] = value

    def get_memory(self, key: str) -> Any | None:
        """获取记忆."""
        with self._lock:
            return self._memory.get(key)

    def checkpoint(self) -> str:
        """创建检查点 (LangGraph 模式).

        保存完整状态快照，可用于恢复.
        """
        with self._lock:
            cp_id = f"ctx-cp-{uuid.uuid4().hex[:8]}"
            self._checkpoints[cp_id] = {
                "state": dict(self._state),
                "timestamp": time.time(),
            }
            return cp_id

    def restore(self, cp_id: str) -> bool:
        """从检查点恢复状态."""
        with self._lock:
            cp = self._checkpoints.get(cp_id)
            if cp is None:
                return False
            self._state = dict(cp["state"])
            return True


# ============================================================
# SessionRecord — 会话记录
# ============================================================


class SessionRecord:
    """会话元数据记录."""

    def __init__(
        self,
        session_id: str,
        agent_id: str,
        learner_id: str,
        source_session_id: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.agent_id = agent_id
        self.learner_id = learner_id
        #: 来源会话 ID (统一会话闭环: 关联的 L1 用户会话)
        self.source_session_id = source_session_id
        self.state = SessionState.CREATED
        self.created_at = time.time()
        self.last_active_at = self.created_at
        self.closed_at: float | None = None
        self.provenance: list[dict[str, Any]] = []
        self._add_provenance("session.create", {
            "session_id": session_id,
            **({"source_session_id": source_session_id} if source_session_id else {}),
        })

    def _add_provenance(self, action: str, context: dict[str, Any] | None = None) -> None:
        """添加溯源记录."""
        self.provenance.append({
            "action": action,
            "context": context or {},
            "timestamp": time.time(),
        })
        self.last_active_at = time.time()


# ============================================================
# ForkCheckpoint — 四类状态快照 (L5 设计文档 4.2.1 节)
# ============================================================


class ForkCheckpoint:
    """Fork 检查点 — 四类状态快照 (L5 设计文档 4.2.1 节).

    一个完整的 Fork Checkpoint 包含:
    1. kernel_state: 所有绑定 Kernel 的变量空间快照
    2. working_session: Working Session 完整状态 (L2 checkpoint/fork 数据)
    3. agent_outputs: 所有 Agent 的输出记录和中间产物
    4. broadcast_queue_state: 学情广播总线消费位点 (Redis stream offset)
    """

    def __init__(
        self,
        session_id: str,
        kernel_state: dict[str, Any] | None = None,
        working_session: dict[str, Any] | None = None,
        agent_outputs: dict[str, Any] | None = None,
        broadcast_queue_state: dict[str, Any] | None = None,
    ) -> None:
        self.checkpoint_id = f"fcp-{uuid.uuid4().hex[:8]}"
        self.session_id = session_id
        self.kernel_state = kernel_state or {}
        self.working_session = working_session or {}
        self.agent_outputs = agent_outputs or {}
        self.broadcast_queue_state = broadcast_queue_state or {}
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "kernel_state": dict(self.kernel_state),
            "working_session": dict(self.working_session),
            "agent_outputs": dict(self.agent_outputs),
            "broadcast_queue_state": dict(self.broadcast_queue_state),
            "timestamp": self.timestamp,
        }


# ============================================================
# ForkEvaluationResult & ForkEvaluator
# ============================================================


class ForkEvaluationResult:
    """Fork 评估结果 (含质量维度, 集成 reflection_quality 模块)."""

    def __init__(
        self,
        fork_id: str,
        learning_gain: float,
        completion_time_s: float,
        resource_tokens: int,
        quality_score: float = 0.5,
    ) -> None:
        self.fork_id = fork_id
        self.learning_gain = learning_gain
        self.completion_time_s = completion_time_s
        self.resource_tokens = resource_tokens
        self.quality_score = max(0.0, min(1.0, quality_score))
        self.score = self._compute_score()

    def _compute_score(self) -> float:
        """计算综合分数.

        权重 (四维度, 集成 reflection_quality 质量评估):
        - learning_gain: 45% (越高越好, 教学效果优先)
        - quality_score: 20% (越高越好, CC1 审核质量)
        - completion_time: 17.5% (越短越好)
        - resource_tokens: 17.5% (越少越好)
        """
        # 归一化
        gain_score = min(self.learning_gain, 1.0)
        quality = self.quality_score
        time_score = max(0, 1.0 - self.completion_time_s / 600.0)
        resource_score = max(0, 1.0 - self.resource_tokens / 50000.0)

        return round(
            gain_score * 0.45 + quality * 0.20 + time_score * 0.175 + resource_score * 0.175,
            4,
        )


class ForkEvaluator:
    """Fork 效果评估器 (L5 设计文档 4.3.2 节).

    评估各 Fork 的效果指标，选定最优路径合并回主 Session.
    评估维度:
    - learning_gain: 学习增益 (BKT 掌握度变化)
    - completion_time_s: 完成时间 (秒)
    - resource_tokens: 资源消耗 (token 数)
    """

    def __init__(self) -> None:
        self._results: dict[str, ForkEvaluationResult] = {}

    def evaluate(
        self,
        fork_id: str,
        learning_gain: float,
        completion_time_s: float,
        resource_tokens: int,
        quality_score: float = 0.5,
    ) -> ForkEvaluationResult:
        """评估单个 Fork (含质量维度, 集成 reflection_quality 模块).

        Args:
            fork_id: Fork ID
            learning_gain: 学习增益 (BKT 掌握度变化)
            completion_time_s: 完成时间 (秒)
            resource_tokens: 资源消耗 (token 数)
            quality_score: CC1 审核质量分 (0.0-1.0, 默认 0.5 中性值)
        """
        result = ForkEvaluationResult(
            fork_id=fork_id,
            learning_gain=learning_gain,
            completion_time_s=completion_time_s,
            resource_tokens=resource_tokens,
            quality_score=quality_score,
        )
        self._results[fork_id] = result
        logger.info(
            f"[ForkEvaluator] Evaluated {fork_id}: "
            f"gain={learning_gain}, time={completion_time_s}s, "
            f"tokens={resource_tokens}, quality={quality_score}, "
            f"score={result.score}"
        )
        return result

    def select_best(self) -> ForkEvaluationResult | None:
        """选择最优 Fork (综合分数最高)."""
        if not self._results:
            return None
        return max(self._results.values(), key=lambda r: r.score)

    def get_result(self, fork_id: str) -> ForkEvaluationResult | None:
        """获取单个 Fork 的评估结果."""
        return self._results.get(fork_id)


# ============================================================
# SessionCompactor — 上下文压缩 (Claude Code/Temporal 模式)
# ============================================================


class SessionCompactor:
    """会话压缩器.

    融合 Claude Code 的自动 compaction + Temporal 的 continue-as-new.

    1. compact: 压缩事件历史 (保留最近 N 条, 旧事件状态合并到当前状态)
    2. continue_as_new: 创建新的干净上下文, 保留核心状态和记忆
    """

    def __init__(self, max_events: int = 100) -> None:
        self.max_events = max_events

    def compact(self, ctx: SessionContext) -> bool:
        """压缩事件历史.

        当事件数超过 max_events 时:
        1. 重放旧事件获取合并状态
        2. 保留最近 max_events 条事件
        3. 状态保持不变 (重放已包含在当前状态中)

        Returns:
            是否执行了压缩
        """
        with ctx._lock:
            events = ctx.events.get_events()
            if len(events) <= self.max_events:
                return False

            # 旧事件已经被 replay 到状态中，不需要额外处理
            # 只需截断事件日志 (保留最近的)
            keep_count = self.max_events
            with ctx.events._lock:
                ctx.events._events = ctx.events._events[-keep_count:]
                # 重建索引
                ctx.events._index = {e.event_id: e for e in ctx.events._events}

            logger.info(
                f"[Compactor] Compacted session {ctx.session_id}: "
                f"kept {keep_count} events"
            )
            return True

    def continue_as_new(
        self,
        ctx: SessionContext,
        preserve_keys: list[str] | None = None,
    ) -> SessionContext:
        """创建新的干净上下文 (Temporal continue-as-new 模式).

        保留指定的核心状态键和全部记忆，清空事件历史.
        新会话获得新的 session_id.

        Args:
            ctx: 原始上下文
            preserve_keys: 需要保留的状态键列表

        Returns:
            新的 SessionContext
        """
        preserve_keys = preserve_keys or []
        # 统一命名空间: ag- (L5 Agent 执行会话)
        new_sid = new_session_id("l5")

        new_ctx = SessionContext(
            session_id=new_sid,
            agent_id=ctx.agent_id,
        )

        # 保留核心状态 (直接设置，不创建事件)
        with ctx._lock:
            for key in preserve_keys:
                if key in ctx._state:
                    new_ctx._state[key] = ctx._state[key]

        # 保留全部记忆
        with ctx._lock:
            new_ctx._memory = dict(ctx._memory)

        logger.info(
            f"[Compactor] continue_as_new: {ctx.session_id} → {new_session_id}, "
            f"preserved {len(preserve_keys)} keys"
        )
        return new_ctx


# ============================================================
# SessionManager — 会话生命周期管理器
# ============================================================


# 合法状态转换
_SESSION_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.CREATED: {SessionState.ACTIVE, SessionState.CLOSED, SessionState.ERROR},
    SessionState.ACTIVE: {SessionState.PAUSED, SessionState.CLOSED, SessionState.ERROR},
    SessionState.PAUSED: {SessionState.ACTIVE, SessionState.CLOSED, SessionState.ERROR},
    SessionState.ERROR: {SessionState.ACTIVE, SessionState.CLOSED},
    SessionState.CLOSED: set(),  # 终态
}

# 状态 → 溯源动作名映射 (使用动词而非形容词)
_STATE_ACTION_MAP: dict[SessionState, str] = {
    SessionState.ACTIVE: "activate",
    SessionState.PAUSED: "pause",
    SessionState.CLOSED: "close",
    SessionState.ERROR: "error",
}


class SessionManager:
    """会话生命周期管理器.

    融合世界先进方案:
    - LangGraph: Thread 管理 + 检查点恢复
    - OpenAI Agents SDK: 极简接口 + 可插拔存储
    - Google ADK: 多维度隔离 (user + app + session)
    - Claude Code: 事件树 + Fork 分支
    - Temporal: 超时管理 + 状态恢复
    - Jupyter: idle timeout 自动回收

    核心功能:
    1. 会话生命周期 (CREATED → ACTIVE ⇄ PAUSED → CLOSED)
    2. 多维度会话索引 (agent_id / learner_id / state)
    3. 上下文管理 (三层: state + events + memory)
    4. Fork 创建/评估/合并 (集成 SessionForkManager)
    5. 超时自动清理
    6. 完整溯源记录
    """

    MAX_FORK_DEPTH = 3
    MAX_FORK_CONCURRENCY = 5

    def __init__(
        self,
        checkpoint_store: CheckpointStore | None = None,
        idle_timeout_s: float = 300.0,
        fork_timeout_s: float = 1800.0,
    ) -> None:
        self._checkpoint_store = checkpoint_store or MemoryCheckpointStore()
        self._idle_timeout_s = idle_timeout_s
        self._fork_timeout_s = fork_timeout_s

        self._sessions: dict[str, SessionRecord] = {}
        self._contexts: dict[str, SessionContext] = {}
        self._learner_index: dict[str, set[str]] = defaultdict(set)
        self._agent_index: dict[str, set[str]] = defaultdict(set)
        #: 来源会话索引 (统一会话闭环: source_session_id → L5 会话集合)
        self._source_index: dict[str, set[str]] = defaultdict(set)

        # Fork 管理
        self._fork_manager = SessionForkManager(self._checkpoint_store)
        self._fork_evaluators: dict[str, ForkEvaluator] = {}
        self._fork_checkpoints: dict[str, ForkCheckpoint] = {}
        self._fork_session_map: dict[str, str] = {}  # fork_id → parent_session_id

        self._lock = threading.RLock()
        self._reflection_engine: Any = None

    def set_reflection_engine(self, reflection_engine: Any) -> None:
        """设置反思引擎 (集成 reflection_quality 模块).

        配置后, Fork 合并将自动触发协作复盘.

        Args:
            reflection_engine: ReflectionEngine 实例
        """
        self._reflection_engine = reflection_engine
        logger.info("[SessionManager] ReflectionEngine configured for fork merge review")

    async def trigger_fork_merge_review(
        self,
        session_id: str,
        participants: list[str],
        metrics: dict[str, Any],
    ) -> None:
        """触发 Fork 合并协作复盘 (L5 设计文档 7.2.1).

        在 Fork 合并完成后, 调用 ReflectionEngine 进行联合复盘.
        未配置 reflection_engine 时静默跳过 (向后兼容).

        Args:
            session_id: 会话 ID
            participants: 参与 Agent 列表
            metrics: 协作指标 (learning_gain/total_duration_s/total_token_cost 等)
        """
        if self._reflection_engine is None:
            logger.debug(
                "[SessionManager] No reflection_engine configured, "
                "skipping fork merge review"
            )
            return

        from .reflection_quality import CollaborationTrigger

        await self._reflection_engine.collaboration_review(
            session_id=session_id,
            trigger=CollaborationTrigger.FORK_MERGE,
            participants=participants,
            metrics=metrics,
        )

    def create_session(
        self,
        agent_id: str,
        learner_id: str,
        source_session_id: str | None = None,
    ) -> SessionRecord:
        """创建新会话.

        Args:
            agent_id: Agent ID.
            learner_id: 学习者 ID.
            source_session_id: 来源会话 ID (L1 用户会话), 统一会话闭环关联.
        """
        # 统一命名空间: ag- (L5 Agent 执行会话, 经 source_session_id 关联 L1)
        session_id = new_session_id("l5")
        record = SessionRecord(
            session_id=session_id,
            agent_id=agent_id,
            learner_id=learner_id,
            source_session_id=source_session_id,
        )
        ctx = SessionContext(session_id=session_id, agent_id=agent_id)

        with self._lock:
            self._sessions[session_id] = record
            self._contexts[session_id] = ctx
            self._learner_index[learner_id].add(session_id)
            self._agent_index[agent_id].add(session_id)
            if source_session_id:
                self._source_index.setdefault(source_session_id, set()).add(session_id)

        logger.info(
            f"[SessionManager] Created session {session_id} "
            f"(agent={agent_id}, learner={learner_id})"
        )
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        """获取会话记录."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_context(self, session_id: str) -> SessionContext | None:
        """获取会话上下文."""
        with self._lock:
            return self._contexts.get(session_id)

    def _transition(self, session_id: str, target: SessionState) -> None:
        """执行状态转换 (含合法性校验)."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise SessionNotFoundError(session_id)

            if record.state == target:
                return  # 幂等

            if record.state == SessionState.CLOSED:
                return  # 终态，幂等

            allowed = _SESSION_TRANSITIONS.get(record.state, set())
            if target not in allowed:
                raise SessionStateError(record.state.value, target.value)

            old_state = record.state
            record.state = target
            action = _STATE_ACTION_MAP.get(target, target.value)
            record._add_provenance(
                f"session.{action}",
                {"from": old_state.value, "to": target.value},
            )

            if target == SessionState.CLOSED:
                record.closed_at = time.time()

    def activate(self, session_id: str) -> None:
        """激活会话: CREATED/PAUSED/ERROR → ACTIVE."""
        self._transition(session_id, SessionState.ACTIVE)

    def pause(self, session_id: str) -> None:
        """暂停会话: ACTIVE → PAUSED."""
        self._transition(session_id, SessionState.PAUSED)

    def resume(self, session_id: str) -> None:
        """恢复会话: PAUSED → ACTIVE."""
        self._transition(session_id, SessionState.ACTIVE)

    def close(self, session_id: str) -> None:
        """关闭会话: 任意 → CLOSED (幂等)."""
        self._transition(session_id, SessionState.CLOSED)

    def list_sessions(
        self,
        agent_id: str | None = None,
        learner_id: str | None = None,
    ) -> list[SessionRecord]:
        """按条件列出会话."""
        with self._lock:
            if learner_id:
                ids = self._learner_index.get(learner_id, set())
            elif agent_id:
                ids = self._agent_index.get(agent_id, set())
            else:
                ids = set(self._sessions.keys())
            return [self._sessions[sid] for sid in ids if sid in self._sessions]

    def list_active_sessions(self) -> list[SessionRecord]:
        """列出所有活跃会话."""
        with self._lock:
            return [
                r for r in self._sessions.values()
                if r.state == SessionState.ACTIVE
            ]

    def get_sessions_by_source(self, source_session_id: str) -> list[SessionRecord]:
        """按来源会话 (L1 用户会话) 查询关联的 L5 执行会话 (统一会话闭环)."""
        with self._lock:
            ids = self._source_index.get(source_session_id, set())
            return [self._sessions[sid] for sid in ids if sid in self._sessions]

    def cleanup_expired(self) -> list[str]:
        """清理超时会话.

        超过 idle_timeout 未活跃的 ACTIVE 会话自动转为 PAUSED.
        """
        now = time.time()
        expired: list[str] = []
        with self._lock:
            for record in self._sessions.values():
                if record.state == SessionState.ACTIVE:
                    idle_time = now - record.last_active_at
                    if idle_time > self._idle_timeout_s:
                        try:
                            self._transition(record.session_id, SessionState.PAUSED)
                            expired.append(record.session_id)
                        except SessionStateError:
                            pass
        return expired

    # ============================================================
    # Fork 管理 (集成 SessionForkManager)
    # ============================================================

    def create_fork(
        self,
        session_id: str,
        trigger_type: str,
        initiator: str,
        reason: str,
        channel_prefix: str = "",
        timeout_seconds: float | None = None,
    ) -> ForkRecord:
        """从会话创建 Fork.

        支持从 Fork 再创建 Fork (fork_id 作为 session_id 传入).
        创建 Fork 检查点并注册到 ForkManager.

        Returns:
            ForkRecord 记录 (附带 session_id 属性)
        """
        with self._lock:
            # 解析 fork_id → 原始 session_id (支持 Fork 树嵌套)
            resolved_session_id = session_id
            if session_id in self._fork_session_map:
                resolved_session_id = self._fork_session_map[session_id]

            record = self._sessions.get(resolved_session_id)
            if record is None:
                raise SessionNotFoundError(session_id)

            # 创建 Fork 检查点 (从解析后的会话上下文)
            ctx = self._contexts.get(resolved_session_id)
            fork_cp = ForkCheckpoint(
                session_id=resolved_session_id,
                kernel_state=ctx.state if ctx else {},
                working_session=ctx.state if ctx else {},
            )
            self._fork_checkpoints[fork_cp.checkpoint_id] = fork_cp

            # 通过 SessionForkManager 创建 Fork
            # 传入原始 session_id (可能是 fork_id) 以正确计算深度
            timeout = timeout_seconds or self._fork_timeout_s
            fork_record = self._fork_manager.create_fork(
                parent_session_id=session_id,
                checkpoint_id=fork_cp.checkpoint_id,
                trigger_type=trigger_type,
                initiator=initiator,
                reason=reason,
                channel_prefix=channel_prefix,
                timeout_seconds=timeout,
            )

            # 记录映射: fork_id → 原始 session_id
            self._fork_session_map[fork_record.fork_id] = resolved_session_id

            # 附加 session_id 属性 (便于调用方获取源会话)
            fork_record.session_id = resolved_session_id

            # 添加溯源
            record._add_provenance("fork.create", {
                "fork_id": fork_record.fork_id,
                "trigger_type": trigger_type,
                "initiator": initiator,
                "reason": reason,
                "checkpoint_id": fork_cp.checkpoint_id,
                "depth": fork_record.depth,
            })

            logger.info(
                f"[SessionManager] Created fork {fork_record.fork_id} "
                f"from session {resolved_session_id} (depth={fork_record.depth})"
            )
            return fork_record

    def record_fork_evaluation(
        self,
        fork_id: str,
        learning_gain: float,
        completion_time_s: float,
        resource_tokens: int,
        quality_score: float = 0.5,
    ) -> ForkEvaluationResult:
        """记录 Fork 评估结果 (含质量维度)."""
        with self._lock:
            session_id = self._fork_session_map.get(fork_id)
            if session_id is None:
                # 尝试从 fork_manager 获取
                fork = self._fork_manager.get_fork(fork_id)
                if fork:
                    session_id = fork.parent_session_id
                else:
                    raise SessionNotFoundError(f"Fork {fork_id} not found")

            # 获取或创建评估器
            if fork_id not in self._fork_evaluators:
                self._fork_evaluators[fork_id] = ForkEvaluator()

            result = self._fork_evaluators[fork_id].evaluate(
                fork_id=fork_id,
                learning_gain=learning_gain,
                completion_time_s=completion_time_s,
                resource_tokens=resource_tokens,
                quality_score=quality_score,
            )

            # 添加溯源
            record = self._sessions.get(session_id)
            if record:
                record._add_provenance("fork.evaluate", {
                    "fork_id": fork_id,
                    "learning_gain": learning_gain,
                    "completion_time_s": completion_time_s,
                    "resource_tokens": resource_tokens,
                    "quality_score": quality_score,
                    "score": result.score,
                })

            return result

    def merge_fork(
        self,
        fork_id: str,
        target_session_id: str,
        merge_scope: list[ForkMergeScope],
    ) -> bool:
        """合并 Fork 回目标会话.

        Args:
            fork_id: 要合并的 Fork ID
            target_session_id: 目标会话 ID
            merge_scope: 合并范围列表

        Returns:
            是否成功合并
        """
        with self._lock:
            # 通过 SessionForkManager 合并
            success = self._fork_manager.merge_fork(fork_id, target_session_id)

            if success:
                record = self._sessions.get(target_session_id)
                if record:
                    record._add_provenance("fork.merge", {
                        "fork_id": fork_id,
                        "target_session_id": target_session_id,
                        "merge_scope": [s.value for s in merge_scope],
                    })

                logger.info(
                    f"[SessionManager] Merged fork {fork_id} → {target_session_id}"
                )

            return success

    def list_forks(self, session_id: str) -> list[ForkRecord]:
        """列出会话的所有 Fork."""
        with self._lock:
            return self._fork_manager.list_forks_by_parent(session_id)

    def cleanup_expired_forks(self) -> list[str]:
        """清理超时 Fork."""
        with self._lock:
            return self._fork_manager.cleanup_expired_forks()
