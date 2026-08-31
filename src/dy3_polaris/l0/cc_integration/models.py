"""CC4 三横切集成 — 共享数据模型.

定义跨 CC1/CC2/CC3 桥接所需的共享数据结构,
包括桥接事件、治理上下文、反馈信号、健康检查等.

融合方案:
- CloudEvents (CNCF): 标准化事件格式 (source/type/subject/data)
- OpenTelemetry: trace_id/span_id 全链路传递
- Kubernetes: 声明式状态模型 (observed/desired/reconciling)
- Prometheus: 指标标签维度模型
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# 枚举定义
# ============================================================


class BridgeDirection(str, enum.Enum):
    """桥接方向 — 六向数据流."""

    CC1_TO_CC2 = "cc1_to_cc2"
    CC1_TO_CC3 = "cc1_to_cc3"
    CC2_TO_CC3 = "cc2_to_cc3"
    CC3_TO_CC1 = "cc3_to_cc1"
    CC3_TO_CC2 = "cc3_to_cc2"
    CC2_TO_CC1 = "cc2_to_cc1"


class GovernancePhase(str, enum.Enum):
    """治理闭环四阶段.

    Reconcile (调谐) → Evaluate (评估) → Act (执行) → Verify (验证)
    基于 Kubernetes Controller Pattern.
    """

    RECONCILE = "reconcile"
    EVALUATE = "evaluate"
    ACT = "act"
    VERIFY = "verify"


class FeedbackSignalType(str, enum.Enum):
    """反馈信号类型."""

    PROVENANCE_COMPLETENESS = "provenance_completeness"
    REVIEW_QUALITY = "review_quality"
    APPROVAL_EFFICIENCY = "approval_efficiency"
    HALLUCINATION_RATE = "hallucination_rate"
    CHAIN_INTEGRITY = "chain_integrity"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    ESCALATION_TRIGGER = "escalation_trigger"


class AlertSeverity(str, enum.Enum):
    """告警严重级别."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(str, enum.Enum):
    """健康状态 (Kubernetes liveness/readiness 启发)."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(str, enum.Enum):
    """断路器状态 (Hystrix 三态模型)."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ============================================================
# 桥接事件模型
# ============================================================


class BridgeEvent(BaseModel):
    """桥接事件 (CloudEvents 启发).

    标准化的事件格式, 用于跨 CC 模块通信.

    Attributes:
        event_id: 事件 ID
        source: 事件源 (cc1/cc2/cc3)
        target: 事件目标 (cc1/cc2/cc3)
        direction: 桥接方向
        event_type: 事件类型
        trace_id: OpenTelemetry trace ID
        span_id: OpenTelemetry span ID
        session_id: 会话 ID
        payload: 事件负载
        timestamp: 时间戳
        metadata: 附加元数据
    """

    event_id: str = Field(
        default_factory=lambda: f"be-{uuid.uuid4().hex[:12]}"
    )
    source: str = Field(description="事件源")
    target: str = Field(description="事件目标")
    direction: BridgeDirection = Field(description="桥接方向")
    event_type: str = Field(default="", description="事件类型")
    trace_id: str = Field(default="", description="OpenTelemetry trace ID")
    span_id: str = Field(default="", description="OpenTelemetry span ID")
    session_id: str = Field(default="", description="会话 ID")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="事件负载"
    )
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# 治理上下文与决策
# ============================================================


class GovernanceContext(BaseModel):
    """治理上下文 — 封装一次治理决策的完整输入.

    Attributes:
        context_id: 上下文 ID
        trace_id: 全链路 trace ID
        session_id: 会话 ID
        user_id: 用户 ID
        operation_type: 操作类型
        target_id: 操作目标 ID
        cc1_verdict: CC1 评审结论
        cc1_score: CC1 综合评分
        cc1_layer_scores: CC1 各层评分
        cc2_layer: CC2 协同层级
        cc2_approval_id: CC2 审批 ID
        cc3_annotation_id: CC3 标注 ID
        cc3_completeness: CC3 标注完整度
        phase: 当前治理阶段
        metadata: 附加元数据
    """

    context_id: str = Field(
        default_factory=lambda: f"gc-{uuid.uuid4().hex[:12]}"
    )
    trace_id: str = Field(default="")
    session_id: str = Field(default="")
    user_id: str = Field(default="")
    operation_type: str = Field(default="")
    target_id: str = Field(default="")
    # CC1 状态
    cc1_verdict: str = Field(default="", description="pass/flag/block")
    cc1_score: float = Field(default=0.0, description="CC1 综合评分 0-100")
    cc1_layer_scores: dict[str, float] = Field(
        default_factory=dict, description="四层评分"
    )
    cc1_issues: list[dict[str, Any]] = Field(default_factory=list)
    # CC2 状态
    cc2_layer: str = Field(default="", description="协同层级")
    cc2_approval_id: str = Field(default="")
    cc2_approval_status: str = Field(default="")
    # CC3 状态
    cc3_annotation_id: str = Field(default="")
    cc3_completeness: float = Field(default=0.0)
    cc3_chain_verified: bool = Field(default=True)
    # 治理阶段
    phase: GovernancePhase = Field(default=GovernancePhase.RECONCILE)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GovernanceDecision(BaseModel):
    """治理决策 — 治理闭环的输出.

    Attributes:
        decision_id: 决策 ID
        context_id: 关联的治理上下文 ID
        phase: 决策阶段
        action: 执行动作
        parameters: 动作参数
        rationale: 决策理由
        affected_modules: 受影响的模块列表
        feedback_signals: 关联的反馈信号
        created_at: 创建时间
    """

    decision_id: str = Field(
        default_factory=lambda: f"gd-{uuid.uuid4().hex[:12]}"
    )
    context_id: str = Field(default="")
    phase: GovernancePhase = Field(default=GovernancePhase.ACT)
    action: str = Field(default="", description="执行动作")
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="")
    affected_modules: list[str] = Field(default_factory=list)
    feedback_signals: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


# ============================================================
# 反馈飞轮模型
# ============================================================


class FeedbackSignal(BaseModel):
    """反馈信号 — CC3 向 CC1/CC2 的反馈.

    Attributes:
        signal_id: 信号 ID
        signal_type: 信号类型
        source: 信号源 (cc3)
        target: 信号目标 (cc1/cc2)
        trace_id: 关联的 trace ID
        annotation_id: KPA 标注 ID
        value: 信号值
        threshold: 阈值
        triggered: 是否触发
        message: 描述信息
        created_at: 创建时间
    """

    signal_id: str = Field(
        default_factory=lambda: f"fs-{uuid.uuid4().hex[:12]}"
    )
    signal_type: FeedbackSignalType = Field(
        default=FeedbackSignalType.PROVENANCE_COMPLETENESS
    )
    source: str = Field(default="cc3")
    target: str = Field(default="cc1")
    trace_id: str = Field(default="")
    annotation_id: str = Field(default="")
    value: float = Field(default=0.0)
    threshold: float = Field(default=0.0)
    triggered: bool = Field(default=False)
    message: str = Field(default="")
    created_at: float = Field(default_factory=time.time)


class FeedbackAction(BaseModel):
    """反馈动作 — 基于反馈信号触发的具体动作.

    Attributes:
        action_id: 动作 ID
        signal_id: 关联的反馈信号 ID
        target_module: 目标模块
        action_type: 动作类型
        parameters: 动作参数
        executed: 是否已执行
        result: 执行结果
        executed_at: 执行时间
    """

    action_id: str = Field(
        default_factory=lambda: f"fa-{uuid.uuid4().hex[:12]}"
    )
    signal_id: str = Field(default="")
    target_module: str = Field(default="")
    action_type: str = Field(default="")
    parameters: dict[str, Any] = Field(default_factory=dict)
    executed: bool = Field(default=False)
    result: dict[str, Any] = Field(default_factory=dict)
    executed_at: float = Field(default=0.0)


# ============================================================
# 健康检查模型
# ============================================================


class HealthCheck(BaseModel):
    """单模块健康检查结果.

    Attributes:
        module: 模块名 (cc1/cc2/cc3)
        status: 健康状态
        latency_ms: 检查延迟 (毫秒)
        details: 详细信息
        checked_at: 检查时间
    """

    module: str = Field(description="模块名")
    status: HealthStatus = Field(default=HealthStatus.UNKNOWN)
    latency_ms: float = Field(default=0.0)
    details: dict[str, Any] = Field(default_factory=dict)
    checked_at: float = Field(default_factory=time.time)


class SystemHealthReport(BaseModel):
    """系统级健康报告.

    Attributes:
        overall_status: 总体健康状态
        modules: 各模块健康检查结果
        active_alerts: 活跃告警
        circuit_states: 断路器状态
        checked_at: 检查时间
    """

    overall_status: HealthStatus = Field(default=HealthStatus.UNKNOWN)
    modules: dict[str, HealthCheck] = Field(default_factory=dict)
    active_alerts: list[dict[str, Any]] = Field(default_factory=list)
    circuit_states: dict[str, str] = Field(default_factory=dict)
    checked_at: float = Field(default_factory=time.time)


# ============================================================
# 断路器配置
# ============================================================


class CircuitBreakerConfig(BaseModel):
    """断路器配置 (Hystrix/Resilience4j 启发).

    Attributes:
        module: 保护的模块名
        failure_threshold: 失败阈值 (连续失败次数)
        recovery_timeout: 恢复超时 (秒, OPEN → HALF_OPEN)
        half_open_max_calls: 半开状态最大调用数
        success_threshold: 成功阈值 (HALF_OPEN → CLOSED)
        timeout_ms: 调用超时 (毫秒)
    """

    module: str = Field(description="保护的模块名")
    failure_threshold: int = Field(default=5, description="连续失败阈值")
    recovery_timeout: float = Field(
        default=30.0, description="恢复超时 (秒)"
    )
    half_open_max_calls: int = Field(
        default=3, description="半开状态最大调用数"
    )
    success_threshold: int = Field(
        default=3, description="成功阈值 (半开→关闭)"
    )
    timeout_ms: float = Field(
        default=5000.0, description="调用超时 (毫秒)"
    )


# ============================================================
# 治理指标模型
# ============================================================


class GovernanceMetrics(BaseModel):
    """治理指标汇总 — Prometheus 风格多维度指标.

    Attributes:
        total_bridges: 总桥接事件数
        bridge_success_rate: 桥接成功率
        avg_governance_latency_ms: 平均治理延迟
        feedback_loops_active: 活跃反馈循环数
        circuit_breaker_trips: 断路器跳闸次数
        cc1_pass_rate: CC1 通过率
        cc2_auto_approval_rate: CC2 自动批准率
        cc3_avg_completeness: CC3 平均完整度
        escalation_count: 升级次数
        collected_at: 采集时间
    """

    total_bridges: int = Field(default=0)
    bridge_success_rate: float = Field(default=0.0)
    avg_governance_latency_ms: float = Field(default=0.0)
    feedback_loops_active: int = Field(default=0)
    circuit_breaker_trips: int = Field(default=0)
    cc1_pass_rate: float = Field(default=0.0)
    cc2_auto_approval_rate: float = Field(default=0.0)
    cc3_avg_completeness: float = Field(default=0.0)
    escalation_count: int = Field(default=0)
    collected_at: float = Field(default_factory=time.time)
