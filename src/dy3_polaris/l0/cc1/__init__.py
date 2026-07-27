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
]
