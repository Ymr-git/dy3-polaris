"""CC2 人机协作层 — 数据模型.

融合八大世界级方案精华：
- REACT Framework 五维评分映射四级自主模式
- LangGraph interrupt/Command 中断恢复机制
- AutoGen max_consecutive_auto_reply 渐进自主
- Swarm escalate_to_human 紧急移交
- GAIA 三阶段协商协议
- Chaos Engineering 混沌感知升级
- CrewAI Task 级 human_input 标记
- Human-AI Negotiation Protocol 置信度协商

所有模型基于 pydantic v2，枚举采用 (str, Enum) 风格与 L6/L0 保持一致。
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ============================================================
# 枚举定义
# ============================================================


class CollaborationMode(str, enum.Enum):
    """协作模式 — REACT 五维评分映射四级自主 (REACT Framework 启发).

    四级人机协作模式，根据 REACT 矩阵
    (Risk/Explainability/Accuracy/Consequence/Time) 均分动态分配：
    - L1 SUPERVISED: 均分 0-1.5 → 所有操作需人类审批
    - L2 CONDITIONAL: 均分 1.5-2.5 → 边界内自主，异常升级
    - L3 MONITORED: 均分 2.5-3.5 → 自由运行，人类监控指标
    - L4 AUTONOMOUS: 均分 3.5-5.0 → 完全自主，事后审核
    """

    SUPERVISED = "supervised"
    CONDITIONAL = "conditional"
    MONITORED = "monitored"
    AUTONOMOUS = "autonomous"


class InterventionType(str, enum.Enum):
    """干预类型 (LangGraph interrupt + Swarm handoff 启发).

    定义人类干预的触发类型：
    - checkpoint: 关键决策点审批（LangGraph interrupt）
    - escalation: 异常升级（Swarm escalate_to_human）
    - negotiation: 协商回合（GAIA 协议）
    - override: 人类主动覆盖
    - review: 事后审核
    """

    CHECKPOINT = "checkpoint"
    ESCALATION = "escalation"
    NEGOTIATION = "negotiation"
    OVERRIDE = "override"
    REVIEW = "review"


class InterventionStatus(str, enum.Enum):
    """干预状态.

    完整生命周期：pending → active → resolved / expired / cancelled
    """

    PENDING = "pending"
    ACTIVE = "active"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class HumanDecision(str, enum.Enum):
    """人类决策 (LangGraph Command resume 启发).

    人类对干预的决策类型：
    - approve: 批准 Agent 提案
    - reject: 拒绝并阻止操作
    - modify: 修改后继续
    - counteroffer: 反提案（GAIA 协商）
    - skip: 跳过本次干预
    - delegate: 委托给其他 Agent
    - terminate: 终止当前流程
    """

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    COUNTEROFFER = "counteroffer"
    SKIP = "skip"
    DELEGATE = "delegate"
    TERMINATE = "terminate"


class NegotiationPhase(str, enum.Enum):
    """协商阶段 (GAIA 三阶段协议启发).

    Screening → Negotiation → Execution/Abandonment
    """

    SCREENING = "screening"
    NEGOTIATION = "negotiation"
    EXECUTION = "execution"
    ABANDONMENT = "abandonment"


class SwitchTrigger(str, enum.Enum):
    """模式切换触发条件.

    升级 (upgrade to more human control):
    - low_confidence: Agent 置信度低于阈值
    - anomaly_detected: 异常检测触发
    - policy_match: 合规策略匹配
    - time_window: 时间窗口变化
    - manual_request: 人类主动请求
    - chaos_detected: 混沌工程注入

    降级 (downgrade to more autonomy):
    - sustained_quality: 持续质量达标
    - no_override: 连续 N 次无覆盖
    - permission_granted: 人类确认权限提升
    - scheduled: 定时降级
    """

    LOW_CONFIDENCE = "low_confidence"
    ANOMALY_DETECTED = "anomaly_detected"
    POLICY_MATCH = "policy_match"
    TIME_WINDOW = "time_window"
    MANUAL_REQUEST = "manual_request"
    CHAOS_DETECTED = "chaos_detected"
    SUSTAINED_QUALITY = "sustained_quality"
    NO_OVERRIDE = "no_override"
    PERMISSION_GRANTED = "permission_granted"
    SCHEDULED = "scheduled"


class ReviewOutcome(str, enum.Enum):
    """审核结论."""

    ACCEPTED = "accepted"
    REVISED = "revised"
    REJECTED = "rejected"
    ESCALATED = "escalated"


# ============================================================
# 核心模型
# ============================================================


class REACTScore(BaseModel):
    """REACT 五维评分 (REACT Framework 启发).

    五个维度各 0-5 分，均分映射到协作模式等级：
    - Risk: 操作风险（0=无风险, 5=极高风险）
    - Explainability: 可解释性（0=完全透明, 5=完全黑盒）
    - Accuracy: 精度要求（0=容忍近似, 5=必须精确）
    - Consequence: 后果严重性（0=可逆, 5=不可逆）
    - Time: 时间敏感度（0=不敏感, 5=极度紧急）

    注意：分数越高表示越需要人类介入。
    """

    risk: float = Field(default=3.0, ge=0.0, le=5.0, description="操作风险 0-5")
    explainability: float = Field(default=3.0, ge=0.0, le=5.0, description="可解释性 0-5")
    accuracy: float = Field(default=3.0, ge=0.0, le=5.0, description="精度要求 0-5")
    consequence: float = Field(default=3.0, ge=0.0, le=5.0, description="后果严重性 0-5")
    time_sensitivity: float = Field(default=3.0, ge=0.0, le=5.0, description="时间敏感度 0-5")

    def average(self) -> float:
        """计算 REACT 均分."""
        return round(
            (self.risk + self.explainability + self.accuracy
             + self.consequence + self.time_sensitivity) / 5.0,
            3,
        )

    def to_mode(self) -> CollaborationMode:
        """REACT 均分 → 协作模式映射."""
        avg = self.average()
        if avg < 1.5:
            return CollaborationMode.AUTONOMOUS
        elif avg < 2.5:
            return CollaborationMode.MONITORED
        elif avg < 3.5:
            return CollaborationMode.CONDITIONAL
        else:
            return CollaborationMode.SUPERVISED


class AgentCollaborationProfile(BaseModel):
    """Agent 协作配置 (动态自主等级 + AutoGen max_consecutive_auto_reply 启发).

    每个 Agent 独立的协作配置，支持运行时动态调整。

    Attributes:
        agent_id: Agent 标识
        mode: 当前协作模式
        default_mode: 默认协作模式
        max_auto_steps: 最大连续自主步数（AutoGen 启发），0=无限
        auto_step_count: 当前连续自主步数计数器
        override_count: 历史人类覆盖次数
        confidence_threshold: 置信度低于此值时触发升级
        timeout_seconds: 人类干预超时（秒），0=无限等待
        escalation_targets: 异常升级目标列表
        enabled: 是否启用人机协作
        tags: 配置标签
    """

    agent_id: str = Field(description="Agent 标识")
    mode: CollaborationMode = Field(
        default=CollaborationMode.CONDITIONAL,
        description="当前协作模式",
    )
    default_mode: CollaborationMode = Field(
        default=CollaborationMode.CONDITIONAL,
        description="默认协作模式",
    )
    max_auto_steps: int = Field(
        default=10, ge=0, description="最大连续自主步数，0=无限",
    )
    auto_step_count: int = Field(
        default=0, ge=0, description="当前连续自主步数",
    )
    override_count: int = Field(
        default=0, ge=0, description="历史人类覆盖次数",
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="置信度升级阈值",
    )
    timeout_seconds: float = Field(
        default=300.0, ge=0.0, description="人类干预超时秒数",
    )
    escalation_targets: list[str] = Field(
        default_factory=list, description="异常升级目标",
    )
    enabled: bool = Field(default=True, description="是否启用")
    tags: list[str] = Field(default_factory=list)

    def reset_auto_steps(self) -> None:
        """重置自主步数计数器（人类干预后调用）."""
        self.auto_step_count = 0

    def increment_auto_step(self) -> bool:
        """递增自主步数，返回是否达到上限.

        Returns:
            True 如果已达到或超过 max_auto_steps
        """
        self.auto_step_count += 1
        if self.max_auto_steps > 0 and self.auto_step_count >= self.max_auto_steps:
            return True
        return False

    def record_override(self) -> None:
        """记录一次人类覆盖."""
        self.override_count += 1


class InterventionRequest(BaseModel):
    """干预请求 (LangGraph interrupt + Swarm escalate_to_human 启发).

    当 Agent 需要人类介入时生成的结构化请求。
    支持多种干预类型和载荷格式。

    Attributes:
        request_id: 干预请求唯一 ID
        agent_id: 请求干预的 Agent ID
        intervention_type: 干预类型
        reason: 干预原因
        payload: 干预载荷（供人类审核的数据）
        proposed_action: Agent 提议的动作
        confidence: Agent 对提议的置信度
        context: 额外上下文
        priority: 优先级 0-100
        timeout_seconds: 超时秒数
        created_at: 创建时间
    """

    request_id: str = Field(
        default_factory=lambda: f"intv-{uuid.uuid4().hex[:10]}",
    )
    agent_id: str = Field(description="请求 Agent ID")
    intervention_type: InterventionType = Field(
        default=InterventionType.CHECKPOINT,
    )
    reason: str = Field(default="", description="干预原因")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="干预载荷",
    )
    proposed_action: str = Field(default="", description="提议动作")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Agent 置信度",
    )
    context: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=50, ge=0, le=100, description="优先级")
    timeout_seconds: float = Field(default=300.0, ge=0.0)
    created_at: float = Field(default_factory=time.time)


class HumanResponse(BaseModel):
    """人类响应 (LangGraph Command resume 启发).

    人类对干预请求的响应。

    Attributes:
        response_id: 响应 ID
        request_id: 关联的干预请求 ID
        human_id: 人类操作者 ID
        decision: 决策类型
        feedback: 人类反馈文本
        modified_action: 修改后的动作（decision=modify 时）
        counteroffer: 反提案数据（decision=counteroffer 时）
        delegate_target: 委托目标（decision=delegate 时）
        responded_at: 响应时间
    """

    response_id: str = Field(
        default_factory=lambda: f"hresp-{uuid.uuid4().hex[:10]}",
    )
    request_id: str = Field(description="关联干预请求 ID")
    human_id: str = Field(default="", description="人类操作者 ID")
    decision: HumanDecision = Field(description="决策类型")
    feedback: str = Field(default="", description="反馈文本")
    modified_action: str = Field(default="", description="修改后动作")
    counteroffer: dict[str, Any] = Field(
        default_factory=dict, description="反提案数据",
    )
    delegate_target: str = Field(default="", description="委托目标")
    responded_at: float = Field(default_factory=time.time)


class InterventionRecord(BaseModel):
    """干预记录.

    完整的干预生命周期记录，从请求到解决。

    Attributes:
        record_id: 记录 ID
        request: 原始干预请求
        response: 人类响应（已解决时）
        status: 干预状态
        mode_at_time: 干预发生时的协作模式
        duration_seconds: 干预持续秒数
        resolution_summary: 解决摘要
        created_at: 创建时间
        resolved_at: 解决时间
    """

    record_id: str = Field(
        default_factory=lambda: f"irec-{uuid.uuid4().hex[:12]}",
    )
    request: InterventionRequest = Field(description="干预请求")
    response: HumanResponse | None = Field(default=None)
    status: InterventionStatus = Field(
        default=InterventionStatus.PENDING,
    )
    mode_at_time: CollaborationMode = Field(
        default=CollaborationMode.CONDITIONAL,
    )
    duration_seconds: float = Field(default=0.0, ge=0.0)
    resolution_summary: str = Field(default="")
    created_at: float = Field(default_factory=time.time)
    resolved_at: float | None = Field(default=None)

    def resolve(
        self,
        response: HumanResponse,
        summary: str = "",
    ) -> None:
        """解决干预记录."""
        self.response = response
        self.status = InterventionStatus.RESOLVED
        self.resolved_at = time.time()
        self.duration_seconds = round(
            self.resolved_at - self.created_at, 3,
        )
        self.resolution_summary = summary or f"{response.decision.value}: {response.feedback}"

    def expire(self) -> None:
        """标记为超时过期."""
        self.status = InterventionStatus.EXPIRED
        self.resolved_at = time.time()
        self.duration_seconds = round(
            self.resolved_at - self.created_at, 3,
        )

    def cancel(self) -> None:
        """取消干预."""
        self.status = InterventionStatus.CANCELLED
        self.resolved_at = time.time()


class NegotiationRound(BaseModel):
    """协商回合 (GAIA 协商协议启发).

    单轮协商的数据结构，包含双方提议和元信息。

    Attributes:
        round_number: 回合编号（从 1 开始）
        proposer: 提议方（"agent" 或 "human"）
        proposal: 提案内容
        confidence: 提案方置信度 0-1
        reasoning: 推理依据
        timestamp: 时间戳
    """

    round_number: int = Field(ge=1, description="回合编号")
    proposer: Literal["agent", "human"] = Field(description="提议方")
    proposal: dict[str, Any] = Field(default_factory=dict, description="提案内容")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度")
    reasoning: str = Field(default="", description="推理依据")
    timestamp: float = Field(default_factory=time.time)


class NegotiationSession(BaseModel):
    """协商会话 (GAIA 三阶段协议启发).

    完整的人机协商会话，管理多轮协商生命周期。

    Attributes:
        session_id: 会话 ID
        agent_id: 参与协商的 Agent ID
        human_id: 人类操作者 ID
        phase: 当前协商阶段
        rounds: 协商回合列表
        max_rounds: 最大协商轮次
        status: 会话状态
        topic: 协商主题
        final_decision: 最终决策
        created_at: 创建时间
    """

    session_id: str = Field(
        default_factory=lambda: f"nego-{uuid.uuid4().hex[:12]}",
    )
    agent_id: str = Field(description="Agent ID")
    human_id: str = Field(default="", description="人类 ID")
    phase: NegotiationPhase = Field(
        default=NegotiationPhase.SCREENING,
    )
    rounds: list[NegotiationRound] = Field(
        default_factory=list, description="协商回合",
    )
    max_rounds: int = Field(default=5, ge=1, le=20, description="最大轮次")
    status: InterventionStatus = Field(
        default=InterventionStatus.ACTIVE,
    )
    topic: str = Field(default="", description="协商主题")
    final_decision: HumanDecision | None = Field(default=None)
    created_at: float = Field(default_factory=time.time)

    @property
    def current_round(self) -> int:
        """当前回合数."""
        return len(self.rounds)

    @property
    def is_exhausted(self) -> bool:
        """是否已耗尽最大轮次."""
        return self.current_round >= self.max_rounds

    def add_round(
        self,
        proposer: Literal["agent", "human"],
        proposal: dict[str, Any],
        confidence: float = 0.5,
        reasoning: str = "",
    ) -> NegotiationRound:
        """添加一个协商回合."""
        round_data = NegotiationRound(
            round_number=self.current_round + 1,
            proposer=proposer,
            proposal=proposal,
            confidence=confidence,
            reasoning=reasoning,
        )
        self.rounds.append(round_data)

        # 自动推进阶段
        if self.phase == NegotiationPhase.SCREENING and self.current_round >= 1:
            self.phase = NegotiationPhase.NEGOTIATION

        return round_data

    def finalize(self, decision: HumanDecision) -> None:
        """结束协商，记录最终决策."""
        self.final_decision = decision
        if decision in (HumanDecision.APPROVE, HumanDecision.SKIP):
            self.phase = NegotiationPhase.EXECUTION
            self.status = InterventionStatus.RESOLVED
        else:
            self.phase = NegotiationPhase.ABANDONMENT
            self.status = InterventionStatus.RESOLVED


class ModeSwitchEvent(BaseModel):
    """模式切换事件.

    记录协作模式的动态切换，用于审计和度量。

    Attributes:
        event_id: 事件 ID
        agent_id: Agent ID
        from_mode: 切换前模式
        to_mode: 切换后模式
        trigger: 触发条件
        reason: 切换原因
        react_score: 触发时的 REACT 评分
        confidence_at_time: 触发时的 Agent 置信度
        timestamp: 时间戳
    """

    event_id: str = Field(
        default_factory=lambda: f"sw-{uuid.uuid4().hex[:10]}",
    )
    agent_id: str = Field(description="Agent ID")
    from_mode: CollaborationMode = Field(description="切换前模式")
    to_mode: CollaborationMode = Field(description="切换后模式")
    trigger: SwitchTrigger = Field(description="触发条件")
    reason: str = Field(default="", description="切换原因")
    react_score: REACTScore | None = Field(default=None)
    confidence_at_time: float | None = Field(default=None)
    timestamp: float = Field(default_factory=time.time)

    @property
    def is_upgrade(self) -> bool:
        """是否为升级（更多人类控制）."""
        order = {
            CollaborationMode.AUTONOMOUS: 0,
            CollaborationMode.MONITORED: 1,
            CollaborationMode.CONDITIONAL: 2,
            CollaborationMode.SUPERVISED: 3,
        }
        return order.get(self.to_mode, 0) > order.get(self.from_mode, 0)


class CollaborationConfig(BaseModel):
    """人机协作全局配置.

    CC2 层的顶层配置，定义全局默认行为和约束。

    Attributes:
        default_mode: 全局默认协作模式
        max_pending_interventions: 每个 Agent 最大待处理干预数
        intervention_timeout_seconds: 全局干预超时
        max_negotiation_rounds: 默认最大协商轮次
        enable_negotiation: 是否启用协商模式
        enable_auto_escalation: 是否启用自动升级
        consecutive_override_threshold: 连续无覆盖次数达到此值触发降级
        sustained_quality_window: 持续质量评估窗口（秒）
        sustained_quality_threshold: 持续质量阈值（准确率）
    """

    default_mode: CollaborationMode = Field(
        default=CollaborationMode.CONDITIONAL,
    )
    max_pending_interventions: int = Field(
        default=10, ge=1, le=100,
    )
    intervention_timeout_seconds: float = Field(default=300.0, ge=0.0)
    max_negotiation_rounds: int = Field(default=5, ge=1, le=20)
    enable_negotiation: bool = Field(default=True)
    enable_auto_escalation: bool = Field(default=True)
    consecutive_override_threshold: int = Field(
        default=5, ge=1, description="连续无覆盖次数达此值降级",
    )
    sustained_quality_window: float = Field(
        default=3600.0, ge=60.0, description="持续质量窗口秒数",
    )
    sustained_quality_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="持续质量准确率阈值",
    )
