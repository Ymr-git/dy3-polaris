"""Agent 交互记录器 — 记录 Agent 运行轨迹与跨 Agent 交互详情.

为 L5 Agent Runtime 提供可观测性:
1. 记录每个 Agent 的执行输入/输出/耗时
2. 记录跨 Agent 消息传递 (广播频道/交互内容)
3. 维护完整的交互链 (diagnosis→generation→review→decision)
4. 提供查询接口供前端可视化

遵循 KPA 溯源链设计原则，每条记录包含:
- 时间戳与唯一标识
- 输入快照与输出快照
- 执行者与接收者
- 交互阶段与协作模式
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InteractionPhase(str, Enum):
    """交互阶段枚举."""

    DIAGNOSIS = "diagnosis"           # 学情诊断阶段
    GENERATION = "generation"         # 知识生成阶段
    REVIEW = "review"                 # 审核校验阶段
    DECISION = "decision"             # 导学决策阶段
    FEEDBACK = "feedback"             # 反馈回流阶段
    ORCHESTRATION = "orchestration"   # 编排执行阶段
    SYSTEM = "system"                 # 系统事件


class InteractionType(str, Enum):
    """交互类型枚举."""

    AGENT_EXECUTION = "agent_execution"       # Agent 单次执行
    BROADCAST_SEND = "broadcast_send"         # 广播发送
    BROADCAST_RECEIVE = "broadcast_receive"   # 广播接收
    PIPELINE_STEP = "pipeline_step"           # 流水线步骤
    DEBATE_ROUND = "debate_round"             # 辩论轮次
    VOTING = "voting"                         # 投票
    FEEDBACK_LOOP = "feedback_loop"           # 反馈回路
    ERROR = "error"                           # 错误事件


@dataclass
class InteractionRecord:
    """单条 Agent 交互记录.

    包含执行上下文、输入输出、耗时和关联信息，
    支持前端按时间线、按 Agent、按阶段三种方式检索。
    """

    record_id: str = field(
        default_factory=lambda: f"int_{uuid.uuid4().hex[:12]}"
    )
    """记录唯一标识."""

    timestamp: float = field(default_factory=time.time)
    """事件发生时间 (Unix 时间戳)."""

    phase: InteractionPhase = InteractionPhase.SYSTEM
    """交互所属阶段."""

    interaction_type: InteractionType = InteractionType.AGENT_EXECUTION
    """交互类型."""

    agent_id: str = ""
    """执行 Agent 的 ID."""

    agent_name: str = ""
    """执行 Agent 的名称 (用于展示)."""

    action: str = ""
    """执行的动作描述 (如 "执行测试", "广播诊断结果")."""

    input_summary: dict[str, Any] = field(default_factory=dict)
    """输入摘要 (用于前端展示)."""

    output_summary: dict[str, Any] = field(default_factory=dict)
    """输出摘要 (用于前端展示)."""

    duration_ms: float = 0.0
    """执行耗时 (毫秒)."""

    status: str = "completed"
    """执行状态: completed / failed / timeout / pending."""

    related_agents: list[str] = field(default_factory=list)
    """关联的其他 Agent ID 列表."""

    channel: str = ""
    """使用的广播频道 (如 learning.knowledge.gap)."""

    phase_order: int = 0
    """在交互链中的序号 (用于前端排序)."""

    detail: dict[str, Any] = field(default_factory=dict)
    """详细数据 (完整输入/输出, 用于点击展开)."""

    parent_id: str = ""
    """父交互记录 ID (用于关联同一会话的交互链)."""


@dataclass
class InteractionChain:
    """一次完整的 4-Agent 交互链.

    对应一次完整的「提问→诊断→生成→审核→决策」流程。
    包含链中所有交互记录，以及总体统计信息。
    """

    chain_id: str = field(
        default_factory=lambda: f"chain_{uuid.uuid4().hex[:12]}"
    )
    """交互链唯一标识."""

    session_id: str = ""
    """关联的会话 ID."""

    learner_id: str = ""
    """关联的学习者 ID."""

    start_time: float = field(default_factory=time.time)
    """交互链开始时间."""

    end_time: float = 0.0
    """交互链结束时间."""

    records: list[InteractionRecord] = field(default_factory=list)
    """链中包含的所有交互记录."""

    query: str = ""
    """用户的原始提问."""

    final_answer: str = ""
    """最终回答/决策."""

    status: str = "running"
    """交互链状态: running / completed / failed."""

    @property
    def total_duration_ms(self) -> float:
        """总耗时 (毫秒)."""
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0.0

    @property
    def agent_count(self) -> int:
        """参与的 Agent 数量."""
        agents = set()
        for r in self.records:
            if r.agent_id:
                agents.add(r.agent_id)
        return len(agents)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典 (用于 API 响应)."""
        return {
            "chain_id": self.chain_id,
            "session_id": self.session_id,
            "learner_id": self.learner_id,
            "start_time": self.start_time,
            "end_time": self.end_time or time.time(),
            "total_duration_ms": self.total_duration_ms,
            "agent_count": self.agent_count,
            "record_count": len(self.records),
            "query": self.query,
            "final_answer": self.final_answer,
            "status": self.status,
            "records": [r.__dict__ for r in self.records],
        }


class InteractionRecorder:
    """Agent 交互记录器 — 记录与查询 Agent 运行/交互轨迹.

    特性:
    - 内存存储 (可选持久化扩展)
    - 按 Agent/阶段/时间范围检索
    - 自动维护交互链完整性
    - 线程安全 (含锁保护)
    """

    def __init__(self, max_chains: int = 100, max_records: int = 5000) -> None:
        self._chains: dict[str, InteractionChain] = {}
        """交互链存储: chain_id -> InteractionChain."""

        self._records: dict[str, InteractionRecord] = {}
        """交互记录存储: record_id -> InteractionRecord."""

        self._current_chain_id: str = ""
        """当前活跃的交互链 ID."""

        self._max_chains = max_chains
        """最大交互链数量."""

        self._max_records = max_records
        """最大记录数量."""

        self._phase_counter: int = 0
        """全局阶段计数器."""

        self._lock = None
        """线程锁 (懒初始化)."""

    def _ensure_lock(self) -> None:
        """确保线程锁已初始化."""
        if self._lock is None:
            import threading
            self._lock = threading.Lock()

    # ---- 交互链管理 ----

    def start_chain(
        self,
        session_id: str = "",
        learner_id: str = "",
        query: str = "",
    ) -> str:
        """开始一个新的交互链.

        Args:
            session_id: 关联的会话 ID.
            learner_id: 关联的学习者 ID.
            query: 用户的原始提问.

        Returns:
            交互链 ID.
        """
        self._ensure_lock()
        chain = InteractionChain(
            session_id=session_id,
            learner_id=learner_id,
            query=query,
        )
        with self._lock:
            self._chains[chain.chain_id] = chain
            self._current_chain_id = chain.chain_id
            # 清理旧链
            self._trim_chains()
        return chain.chain_id

    def end_chain(
        self,
        chain_id: str = "",
        final_answer: str = "",
        status: str = "completed",
    ) -> None:
        """结束一个交互链.

        Args:
            chain_id: 交互链 ID (默认使用当前链).
            final_answer: 最终回答.
            status: 完成状态.
        """
        self._ensure_lock()
        with self._lock:
            chain = self._chains.get(chain_id or self._current_chain_id)
            if chain:
                chain.end_time = time.time()
                chain.final_answer = final_answer
                chain.status = status

    def get_current_chain(self) -> InteractionChain | None:
        """获取当前活跃的交互链."""
        return self._chains.get(self._current_chain_id)

    # ---- 记录管理 ----

    def record_interaction(
        self,
        phase: InteractionPhase | str,
        interaction_type: InteractionType | str,
        agent_id: str,
        agent_name: str = "",
        action: str = "",
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
        status: str = "completed",
        related_agents: list[str] | None = None,
        channel: str = "",
        detail: dict[str, Any] | None = None,
        chain_id: str = "",
    ) -> str:
        """记录一次 Agent 交互.

        Args:
            phase: 交互阶段.
            interaction_type: 交互类型.
            agent_id: 执行 Agent ID.
            agent_name: Agent 名称.
            action: 执行动作描述.
            input_summary: 输入摘要.
            output_summary: 输出摘要.
            duration_ms: 执行耗时 (毫秒).
            status: 执行状态.
            related_agents: 关联 Agent ID 列表.
            channel: 广播频道.
            detail: 详细数据.
            chain_id: 所属交互链 ID (默认当前链).

        Returns:
            记录 ID.
        """
        self._ensure_lock()
        # 处理枚举类型
        if isinstance(phase, str):
            try:
                phase = InteractionPhase(phase)
            except ValueError:
                phase = InteractionPhase.SYSTEM
        if isinstance(interaction_type, str):
            try:
                interaction_type = InteractionType(interaction_type)
            except ValueError:
                interaction_type = InteractionType.AGENT_EXECUTION

        self._phase_counter += 1
        record = InteractionRecord(
            phase=phase,
            interaction_type=interaction_type,
            agent_id=agent_id,
            agent_name=agent_name or agent_id.split(".")[-1] if "." in agent_id else agent_id,
            action=action,
            input_summary=input_summary or {},
            output_summary=output_summary or {},
            duration_ms=duration_ms,
            status=status,
            related_agents=related_agents or [],
            channel=channel,
            phase_order=self._phase_counter,
            detail=detail or {},
        )

        with self._lock:
            self._records[record.record_id] = record
            # 关联到交互链
            cid = chain_id or self._current_chain_id
            if cid and cid in self._chains:
                record.parent_id = cid
                self._chains[cid].records.append(record)
            # 清理旧记录
            self._trim_records()

        return record.record_id

    def record_agent_execution(
        self,
        agent_id: str,
        agent_name: str = "",
        action: str = "",
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
        status: str = "completed",
        phase: InteractionPhase = InteractionPhase.SYSTEM,
        chain_id: str = "",
    ) -> str:
        """记录一次 Agent 执行.

        Args:
            agent_id: Agent ID.
            agent_name: Agent 名称.
            action: 执行动作描述.
            input_data: 输入数据.
            output_data: 输出数据.
            duration_ms: 执行耗时 (毫秒).
            status: 执行状态.
            phase: 交互阶段.
            chain_id: 所属交互链 ID.

        Returns:
            记录 ID.
        """
        # 提取摘要 (避免前端传输大量数据)
        input_summary = self._summarize(input_data or {})
        output_summary = self._summarize(output_data or {})

        return self.record_interaction(
            phase=phase,
            interaction_type=InteractionType.AGENT_EXECUTION,
            agent_id=agent_id,
            agent_name=agent_name,
            action=action,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_ms=duration_ms,
            status=status,
            detail={
                "full_input": self._truncate_dict(input_data or {}),
                "full_output": self._truncate_dict(output_data or {}),
            },
            chain_id=chain_id,
        )

    def record_broadcast(
        self,
        from_agent: str,
        from_name: str = "",
        to_agents: list[str] | None = None,
        channel: str = "",
        payload_summary: dict[str, Any] | None = None,
        phase: InteractionPhase = InteractionPhase.SYSTEM,
        chain_id: str = "",
    ) -> str:
        """记录一次广播交互.

        Args:
            from_agent: 发送方 Agent ID.
            from_name: 发送方名称.
            to_agents: 接收方 Agent ID 列表.
            channel: 广播频道.
            payload_summary: 消息载荷摘要.
            phase: 交互阶段.
            chain_id: 所属交互链 ID.

        Returns:
            记录 ID.
        """
        return self.record_interaction(
            phase=phase,
            interaction_type=InteractionType.BROADCAST_SEND,
            agent_id=from_agent,
            agent_name=from_name,
            action=f"广播消息到 {channel}",
            input_summary=payload_summary or {},
            output_summary={"channel": channel, "receivers": to_agents or []},
            related_agents=to_agents or [],
            channel=channel,
            chain_id=chain_id,
        )

    # ---- 查询接口 ----

    def get_all_chains(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """获取所有交互链摘要.

        Args:
            limit: 返回数量上限.
            offset: 偏移量.

        Returns:
            交互链摘要列表.
        """
        chains = sorted(
            self._chains.values(),
            key=lambda c: c.start_time,
            reverse=True,
        )
        return [self._chain_summary(c) for c in chains[offset:offset + limit]]

    def get_chain_detail(self, chain_id: str) -> dict[str, Any] | None:
        """获取单个交互链详情.

        Args:
            chain_id: 交互链 ID.

        Returns:
            交互链详情字典, 或 None (不存在).
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            return None
        return chain.to_dict()

    def get_all_records(
        self,
        limit: int = 50,
        offset: int = 0,
        agent_id: str = "",
        phase: str = "",
    ) -> list[dict[str, Any]]:
        """获取交互记录列表.

        Args:
            limit: 返回数量上限.
            offset: 偏移量.
            agent_id: 按 Agent ID 筛选.
            phase: 按阶段筛选.

        Returns:
            记录字典列表.
        """
        records = list(self._records.values())

        # 筛选
        if agent_id:
            records = [r for r in records if r.agent_id == agent_id]
        if phase:
            records = [r for r in records if r.phase.value == phase]

        records.sort(key=lambda r: r.timestamp, reverse=True)
        return [r.__dict__ for r in records[offset:offset + limit]]

    def get_records_by_chain(self, chain_id: str) -> list[dict[str, Any]]:
        """获取交互链中的所有记录.

        Args:
            chain_id: 交互链 ID.

        Returns:
            按阶段排序的记录列表.
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            return []
        sorted_records = sorted(chain.records, key=lambda r: r.phase_order)
        return [r.__dict__ for r in sorted_records]

    def get_latest_records(self, limit: int = 30) -> list[dict[str, Any]]:
        """获取最新的交互记录.

        Args:
            limit: 返回数量上限.

        Returns:
            记录字典列表.
        """
        records = sorted(
            self._records.values(),
            key=lambda r: r.timestamp,
            reverse=True,
        )
        return [r.__dict__ for r in records[:limit]]

    def get_stats(self) -> dict[str, Any]:
        """获取交互记录统计信息.

        Returns:
            统计信息字典.
        """
        total_records = len(self._records)
        total_chains = len(self._chains)
        completed_chains = sum(
            1 for c in self._chains.values() if c.status == "completed"
        )

        # 按 Agent 统计
        agent_stats: dict[str, int] = {}
        for r in self._records.values():
            agent_stats[r.agent_id] = agent_stats.get(r.agent_id, 0) + 1

        # 按阶段统计
        phase_stats: dict[str, int] = {}
        for r in self._records.values():
            p = r.phase.value
            phase_stats[p] = phase_stats.get(p, 0) + 1

        return {
            "total_records": total_records,
            "total_chains": total_chains,
            "completed_chains": completed_chains,
            "agent_stats": agent_stats,
            "phase_stats": phase_stats,
        }

    # ---- 内部工具 ----

    def _chain_summary(self, chain: InteractionChain) -> dict[str, Any]:
        """生成交互链摘要."""
        agent_set = set()
        phase_set = set()
        for r in chain.records:
            if r.agent_id:
                agent_set.add(r.agent_id)
            phase_set.add(r.phase.value)

        return {
            "chain_id": chain.chain_id,
            "session_id": chain.session_id,
            "learner_id": chain.learner_id,
            "start_time": chain.start_time,
            "end_time": chain.end_time or time.time(),
            "total_duration_ms": chain.total_duration_ms,
            "agent_count": len(agent_set),
            "record_count": len(chain.records),
            "agents": list(agent_set),
            "phases": list(phase_set),
            "query": chain.query,
            "status": chain.status,
        }

    def _summarize(self, data: dict[str, Any]) -> dict[str, Any]:
        """提取数据摘要 (用于前端展示, 避免大量数据传输)."""
        summary: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str):
                summary[k] = v[:100] + ("..." if len(v) > 100 else "")
            elif isinstance(v, (dict, list)):
                summary[k] = self._truncate_value(v)
            else:
                summary[k] = v
        return summary

    def _truncate_value(self, value: Any, max_len: int = 200) -> Any:
        """截断值."""
        s = str(value)
        if len(s) > max_len:
            return s[:max_len] + "..."
        return value

    def _truncate_dict(self, d: dict[str, Any], max_len: int = 500) -> dict[str, Any]:
        """截断字典中的值."""
        return {k: self._truncate_value(v, max_len) for k, v in d.items()}

    def _trim_chains(self) -> None:
        """清理超出上限的交互链."""
        if len(self._chains) <= self._max_chains:
            return
        sorted_chains = sorted(
            self._chains.values(),
            key=lambda c: c.start_time,
        )
        excess = len(sorted_chains) - self._max_chains
        for c in sorted_chains[:excess]:
            del self._chains[c.chain_id]

    def _trim_records(self) -> None:
        """清理超出上限的记录."""
        if len(self._records) <= self._max_records:
            return
        sorted_records = sorted(
            self._records.values(),
            key=lambda r: r.timestamp,
        )
        excess = len(sorted_records) - self._max_records
        for r in sorted_records[:excess]:
            del self._records[r.record_id]


# 全局单例 (由 L5 Runtime 初始化)
_default_recorder: InteractionRecorder | None = None


def get_recorder() -> InteractionRecorder:
    """获取全局交互记录器实例."""
    global _default_recorder
    if _default_recorder is None:
        _default_recorder = InteractionRecorder()
    return _default_recorder


def set_recorder(recorder: InteractionRecorder) -> None:
    """设置全局交互记录器实例."""
    global _default_recorder
    _default_recorder = recorder


__all__ = [
    "InteractionPhase",
    "InteractionType",
    "InteractionRecord",
    "InteractionChain",
    "InteractionRecorder",
    "get_recorder",
    "set_recorder",
]