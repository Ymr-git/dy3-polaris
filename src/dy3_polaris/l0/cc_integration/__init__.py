"""L0 治理层 — CC4 三横切集成与治理闭环.

实现 CC1 (反幻觉) / CC2 (人机协作) / CC3 (溯源捕获) 三大横切机制的
双向集成与治理闭环, 形成完整的审计-决策-溯源飞轮.

核心能力:
- CC1→CC2 桥接: 评审结果注入路由决策 (verdict→confidence, BLOCK→强制审批)
- CC1→CC3 桥接: 评审结果自动标注到 KPA 校验维度
- CC2→CC3 桥接: 审批记录自动捕获到 KPA 决策维度
- CC3→CC1/CC2 反馈飞轮: 溯源完整性反馈→评审阈值调整+协同升级建议
- 统一网关: 单一入口路由 CC1/CC2/CC3 API
- 健康聚合: 跨模块健康状态汇总与告警

融合世界先进方案:
- Service Mesh (Istio/Linkerd): 横切关注点统一编排
- Event-Driven Architecture (Kafka/Pulsar): 松耦合事件驱动集成
- Control Plane (Kubernetes): 声明式配置+调谐循环 (reconcile loop)
- Observability (OpenTelemetry): 全链路 trace_id 传递与分布式追踪
- Circuit Breaker (Hystrix/Resilience4j): 级联故障隔离与降级
- Policy-as-Code (OPA/Gatekeeper): 声明式策略引擎
- GitOps (ArgoCD/Flux): 不可变审计日志 + 回滚能力
"""

from .models import (
    # 枚举
    BridgeDirection,
    GovernancePhase,
    FeedbackSignalType,
    AlertSeverity,
    HealthStatus,
    # 数据模型
    BridgeEvent,
    GovernanceContext,
    GovernanceDecision,
    FeedbackSignal,
    FeedbackAction,
    HealthCheck,
    SystemHealthReport,
    CircuitState,
    CircuitBreakerConfig,
    GovernanceMetrics,
)
from .exceptions import (
    CC4Error,
    BridgeConnectionError,
    FeedbackLoopError,
    GatewayRoutingError,
    HealthCheckError,
    CircuitBreakerOpenError,
    GovernancePolicyError,
)
from .cc1_cc2_bridge import CC1CC2Bridge
from .cc1_cc3_bridge import CC1CC3Bridge
from .cc2_cc3_bridge import CC2CC3Bridge
from .feedback_loop import FeedbackLoop
from .unified_gateway import UnifiedGateway
from .health_aggregator import HealthAggregator
from .circuit_breaker import CircuitBreaker

# REST API (可选导入)
try:
    from .api import CC4APIRouter
except ImportError:  # pragma: no cover
    CC4APIRouter = None  # type: ignore[assignment,misc]

__all__ = [
    # ==================== 枚举 ====================
    "BridgeDirection",
    "GovernancePhase",
    "FeedbackSignalType",
    "AlertSeverity",
    "HealthStatus",
    # ==================== 数据模型 ====================
    "BridgeEvent",
    "GovernanceContext",
    "GovernanceDecision",
    "FeedbackSignal",
    "FeedbackAction",
    "HealthCheck",
    "SystemHealthReport",
    "CircuitState",
    "CircuitBreakerConfig",
    "GovernanceMetrics",
    # ==================== 异常 ====================
    "CC4Error",
    "BridgeConnectionError",
    "FeedbackLoopError",
    "GatewayRoutingError",
    "HealthCheckError",
    "CircuitBreakerOpenError",
    "GovernancePolicyError",
    # ==================== 桥接器 ====================
    "CC1CC2Bridge",
    "CC1CC3Bridge",
    "CC2CC3Bridge",
    # ==================== 反馈飞轮 ====================
    "FeedbackLoop",
    # ==================== 网关与健康 ====================
    "UnifiedGateway",
    "HealthAggregator",
    "CircuitBreaker",
    # ==================== REST API ====================
    "CC4APIRouter",
]
