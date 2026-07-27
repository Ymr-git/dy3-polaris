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

核心组件：
- 数据模型: REACTScore, AgentCollaborationProfile, InterventionRequest/Response/Record,
  NegotiationSession, NegotiationRound, ModeSwitchEvent, CollaborationConfig
- 引擎: CollaborationEngine（干预管理、模式切换、协商、升级）
- 异常: CC2Error 体系（JSON-RPC -32300 ~ -32306）
"""

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

__all__ = [
    # 枚举
    "CollaborationMode",
    "InterventionType",
    "InterventionStatus",
    "HumanDecision",
    "NegotiationPhase",
    "SwitchTrigger",
    "ReviewOutcome",
    # 模型
    "REACTScore",
    "AgentCollaborationProfile",
    "InterventionRequest",
    "HumanResponse",
    "InterventionRecord",
    "NegotiationRound",
    "NegotiationSession",
    "ModeSwitchEvent",
    "CollaborationConfig",
    # 异常
    "CC2Error",
    "InterventionTimeoutError",
    "NegotiationExhaustedError",
    "ProfileNotFoundError",
    "ModeSwitchError",
    "InterventionConflictError",
    "EscalationTargetError",
    # 引擎
    "CollaborationEngine",
]
