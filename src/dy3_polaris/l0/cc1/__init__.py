"""L0 治理层 — CC1 防幻觉子包.

提供 LLM 输出防幻觉验证的完整能力，融合五大世界级方案：
- RAGAS 式声明分解与忠实度评分
- SelfCheckGPT 式多采样一致性检测
- CoVe 式四阶段验证管道
- Guardrails AI 式可插拔验证器注册与修正动作
- NeMo Guardrails 式分层 Rail 架构

核心组件：
- 数据模型: Claim, Evidence, VerificationReport, VerificationRequest 等
- 验证器: CitationVerifier, GroundednessVerifier, ConsistencyVerifier, FactCheckVerifier
- 管道: AntiHallucinationPipeline (四阶段编排)
- 存储: VerificationStore (线程安全报告与记录存储)
- 异常: CC1Error 体系 (JSON-RPC -32200 ~ -32206)
"""

from .models import (
    Claim,
    ClaimType,
    ClaimVerificationResult,
    Evidence,
    EvidenceType,
    HallucinationRecord,
    HallucinationSeverity,
    PipelineConfig,
    VerificationReport,
    VerificationRequest,
    VerificationStage,
    VerificationStatus,
    VerdictAction,
    VerifierConfig,
    VerifierType,
)
from .exceptions import (
    CC1Error,
    ClaimExtractionError,
    EvidenceInsufficientError,
    HallucinationDetectedError,
    PipelineConfigError,
    VerificationError,
    VerifierNotFoundError,
)
from .verifiers import (
    BaseVerifier,
    CitationVerifier,
    ConsistencyVerifier,
    FactCheckVerifier,
    GroundednessVerifier,
    VerifierRegistry,
)
from .pipeline import (
    AntiHallucinationPipeline,
    ClaimExtractor,
    EvidenceCollector,
)
from .store import VerificationStore

# 四层反幻觉评审引擎 (增强)
from .layers import (
    BaseReviewLayer,
    FactLayer,
    LAYER_WEIGHTS,
    LayerResult,
    LayerRuleResult,
    LogicLayer,
    NumericalLayer,
    ProvenanceLayer,
    ReviewLayerType,
    ReviewRule,
    RuleSeverity,
)
from .state_machine import (
    ReviewState,
    ReviewStateMachine,
    ReviewVerdict,
    SelfCorrectionLoop,
    SelfCorrectionRecord,
)
from .scoring import (
    CompositeScoringEngine,
    ScoringWeights,
)
from .review_pipeline import (
    LearnerLevel,
    LEARNER_LEVEL_THRESHOLDS,
    ReviewPipeline,
    ReviewPipelineConfig,
    ReviewResult,
)

# L2 推理链提取与 DAG 分析
from .reasoning_chain import (
    Contradiction,
    ContradictionDetector,
    ReasoningChainError,
    ReasoningChainExtractor,
    ReasoningDAG,
    ReasoningStep,
)

# L4 溯源链构建与权威性评级
from .provenance_chain import (
    AuthorityRater,
    ProvenanceChain,
    ProvenanceNode,
    SourceTier,
    TIER_1_JOURNALS,
    TIER_BASE_SCORES,
    SOURCE_TYPE_DEFAULT_TIER,
    VersionManager,
)

# L3 数值层计算引擎 (单位换算 / Judd-Ofelt / CIE 色度 / 误差分析)
from .computation import (
    CIECalculator,
    CIESpectrumPoint,
    CIEWhitePoint,
    ComputationError,
    CONST,
    ErrorAnalyzer,
    JuddOfeltCalculator,
    JuddOfeltRanges,
    PhysicalConstants,
    UnitConverter,
)

# 三阶段评审 (预评审 / 同步评审 / 后评审编排)
from .pre_review import (
    IntermediateSample,
    PreReviewContext,
    PreReviewEngine,
    PreReviewResult,
    ReviewStageType,
    SynchronousReviewHook,
    SynchronousReviewResult,
    ThreeStageReviewOrchestrator,
)

__all__ = [
    # 枚举
    "VerificationStage",
    "VerificationStatus",
    "ClaimType",
    "EvidenceType",
    "VerifierType",
    "VerdictAction",
    "HallucinationSeverity",
    # 模型
    "Claim",
    "Evidence",
    "ClaimVerificationResult",
    "VerificationReport",
    "VerificationRequest",
    "VerifierConfig",
    "PipelineConfig",
    "HallucinationRecord",
    # 异常
    "CC1Error",
    "VerificationError",
    "ClaimExtractionError",
    "VerifierNotFoundError",
    "HallucinationDetectedError",
    "PipelineConfigError",
    "EvidenceInsufficientError",
    # 验证器
    "BaseVerifier",
    "VerifierRegistry",
    "CitationVerifier",
    "GroundednessVerifier",
    "ConsistencyVerifier",
    "FactCheckVerifier",
    # 管道
    "ClaimExtractor",
    "EvidenceCollector",
    "AntiHallucinationPipeline",
    # 存储
    "VerificationStore",
    # 四层反幻觉评审引擎 (增强)
    "ReviewLayerType",
    "RuleSeverity",
    "ReviewRule",
    "LayerRuleResult",
    "LayerResult",
    "LAYER_WEIGHTS",
    "BaseReviewLayer",
    "FactLayer",
    "LogicLayer",
    "NumericalLayer",
    "ProvenanceLayer",
    "ReviewState",
    "ReviewVerdict",
    "ReviewStateMachine",
    "SelfCorrectionRecord",
    "SelfCorrectionLoop",
    "ScoringWeights",
    "CompositeScoringEngine",
    "ReviewPipelineConfig",
    "ReviewResult",
    "ReviewPipeline",
    "LearnerLevel",
    "LEARNER_LEVEL_THRESHOLDS",
    # L2 推理链与 DAG 分析
    "ReasoningStep",
    "ReasoningChainExtractor",
    "ReasoningDAG",
    "Contradiction",
    "ContradictionDetector",
    "ReasoningChainError",
    # L4 溯源链与权威性评级
    "SourceTier",
    "ProvenanceNode",
    "ProvenanceChain",
    "AuthorityRater",
    "VersionManager",
    "TIER_BASE_SCORES",
    "TIER_1_JOURNALS",
    "SOURCE_TYPE_DEFAULT_TIER",
    # L3 数值层计算引擎
    "PhysicalConstants",
    "ComputationError",
    "CONST",
    "UnitConverter",
    "JuddOfeltRanges",
    "JuddOfeltCalculator",
    "CIESpectrumPoint",
    "CIEWhitePoint",
    "CIECalculator",
    "ErrorAnalyzer",
    # 三阶段评审
    "ReviewStageType",
    "PreReviewContext",
    "PreReviewResult",
    "PreReviewEngine",
    "IntermediateSample",
    "SynchronousReviewResult",
    "SynchronousReviewHook",
    "ThreeStageReviewOrchestrator",
]
