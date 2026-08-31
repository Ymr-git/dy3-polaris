"""L0 治理层 — CC2 人机协作子包.

提供教育多 Agent 系统的人机协作核心能力，融合八大世界级方案：
- REACT Framework 五维评分映射四级自主模式
- LangGraph interrupt/Command 中断恢复机制
- AutoGen max_consecutive_auto_reply 渐进自主
- Swarm escalate_to_human 紧急移交
- GAIA 三阶段协商协议（Screening → Negotiation → Execution）
- Chaos Engineering 混沌感知紧急升级
- CrewAI Task 级 human_input 标记
- Human-AI Negotiation Protocol 置信度协商

增强组件 (CC-2 计划审批门):
- 决策路由引擎: RoutingEngine — 六维决策矩阵+加权评分+规则覆盖
- 审批工作流: ApprovalWorkflowManager — 四种审批模式+超时策略+信任窗口
- 抗疲劳机制: AntiFatigueManager — 频率控制+批量审批+智能预批+渐进信任
- 干预管理器: InterventionManager — 紧急暂停+手动接管+纠错反馈+创造请求
- KPI 指标引擎: KPIMetricsEngine — 9项KPI追踪+动态阈值调整
- REST API: CC2APIRouter — 48个API端点全覆盖

核心组件：
- 数据模型: REACTScore, AgentCollaborationProfile, InterventionRequest/Response/Record,
  NegotiationSession, NegotiationRound, ModeSwitchEvent, CollaborationConfig
- 引擎: CollaborationEngine（干预管理、模式切换、协商、升级、路由、审批、KPI 集成）
- 异常: CC2Error 体系（JSON-RPC -32300 ~ -32306）
"""

# 原有模型和异常导出
from .models import (
    AgentCollaborationProfile,
    CollaborationConfig,
    CollaborationMode,
    HumanDecision,
    HumanResponse,
    InterventionRecord,
    InterventionRequest,
    InterventionStatus,
    InterventionType,
    ModeSwitchEvent,
    NegotiationPhase,
    NegotiationRound,
    NegotiationSession,
    REACTScore,
    ReviewOutcome,
    SwitchTrigger,
)
from .exceptions import (
    CC2Error,
    EscalationTargetError,
    InterventionConflictError,
    InterventionTimeoutError,
    ModeSwitchError,
    NegotiationExhaustedError,
    ProfileNotFoundError,
)
from .engine import CollaborationEngine

# 增强组件导出 — 决策路由引擎
from .routing_engine import (
    RoutingEngine,
    RoutingContext,
    RoutingResult,
    RoutingRule,
    CollaborationLayer,
    RiskLevel,
    Reversibility,
    UserRole,
    ApprovalMode,
    TimeoutAction,
    InterventionTypeL4,
    Priority,
    RecoveryMode,
    DEFAULT_ROUTING_RULES,
)

# 增强组件导出 — 审批工作流
from .approval_workflow import (
    ApprovalWorkflowManager,
    ApprovalRequest,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalStatus,
    TrustModeWindow,
)

# 增强组件导出 — 抗疲劳机制
from .anti_fatigue import (
    AntiFatigueManager,
    FatigueConfig,
    FatigueLevel,
    FatigueState,
    ApprovalPattern,
    BatchApprovalGroup,
    BatchStatus,
    ProgressiveTrustRecord,
)

# 增强组件导出 — 干预管理器
from .intervention_manager import (
    InterventionManager,
    EmergencyPauseRequest,
    ManualOverrideRequest,
    CorrectionFeedback,
    CreativeRequest,
    InterventionEvent,
    InterventionAction,
    PauseScope,
    OverrideLevel,
    CorrectionType,
    CorrectionSeverity,
    CreativeRequestType,
    InterventionEventStatus,
)

# 增强组件导出 — KPI 指标引擎
from .kpi_metrics import (
    KPIMetricsEngine,
    KPISample,
    KPIStatus,
    KPICategory,
    TrendDirection,
    KPISummary,
    KPIThreshold,
    KPITrend,
)

# 增强组件导出 — REST API
from .api import CC2APIRouter

__all__ = [
    # ==================== 原有枚举 ====================
    "CollaborationMode",
    "InterventionType",
    "InterventionStatus",
    "HumanDecision",
    "NegotiationPhase",
    "SwitchTrigger",
    "ReviewOutcome",
    # ==================== 原有模型 ====================
    "REACTScore",
    "AgentCollaborationProfile",
    "InterventionRequest",
    "HumanResponse",
    "InterventionRecord",
    "NegotiationRound",
    "NegotiationSession",
    "ModeSwitchEvent",
    "CollaborationConfig",
    # ==================== 原有异常 ====================
    "CC2Error",
    "InterventionTimeoutError",
    "NegotiationExhaustedError",
    "ProfileNotFoundError",
    "ModeSwitchError",
    "InterventionConflictError",
    "EscalationTargetError",
    # ==================== 原有引擎 ====================
    "CollaborationEngine",
    # ==================== 路由引擎 ====================
    "RoutingEngine",
    "RoutingContext",
    "RoutingResult",
    "RoutingRule",
    "CollaborationLayer",
    "RiskLevel",
    "Reversibility",
    "UserRole",
    "ApprovalMode",
    "TimeoutAction",
    "InterventionTypeL4",
    "Priority",
    "RecoveryMode",
    "DEFAULT_ROUTING_RULES",
    # ==================== 审批工作流 ====================
    "ApprovalWorkflowManager",
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalRecord",
    "ApprovalStatus",
    "TrustModeWindow",
    # ==================== 抗疲劳 ====================
    "AntiFatigueManager",
    "FatigueConfig",
    "FatigueLevel",
    "FatigueState",
    "ApprovalPattern",
    "BatchApprovalGroup",
    "BatchStatus",
    "ProgressiveTrustRecord",
    # ==================== 干预管理器 ====================
    "InterventionManager",
    "EmergencyPauseRequest",
    "ManualOverrideRequest",
    "CorrectionFeedback",
    "CreativeRequest",
    "InterventionEvent",
    "InterventionAction",
    "PauseScope",
    "OverrideLevel",
    "CorrectionType",
    "CorrectionSeverity",
    "CreativeRequestType",
    "InterventionEventStatus",
    # ==================== KPI 指标 ====================
    "KPIMetricsEngine",
    "KPISample",
    "KPIStatus",
    "KPICategory",
    "TrendDirection",
    "KPISummary",
    "KPIThreshold",
    "KPITrend",
    # ==================== REST API ====================
    "CC2APIRouter",
]
