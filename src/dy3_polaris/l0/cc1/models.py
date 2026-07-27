"""CC1 防幻觉层 — 数据模型.

定义防幻觉验证的核心数据结构，融合五大世界级方案精华：
- RAGAS 式声明分解与忠实度评分
- SelfCheckGPT 式一致性检测
- CoVe 式多阶段验证管道
- Guardrails AI 式验证器结果与修正动作
- NeMo Guardrails 式分层 Rail 架构

所有模型基于 pydantic v2，枚举采用 (str, Enum) 风格与 L6/L0 保持一致。
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# 枚举定义
# ============================================================


class VerificationStage(str, Enum):
    """验证阶段（NeMo Guardrails Rail 分层启发）.

    对应防幻觉管道的四个阶段：
    - input: 输入检查（prompt 注入、安全过滤）
    - claim_extraction: 声明提取（RAGAS 原子化分解）
    - verification: 声明验证（多验证器并行）
    - output: 输出决策（Guardrails 修正动作）
    """

    INPUT = "input"
    CLAIM_EXTRACTION = "claim_extraction"
    VERIFICATION = "verification"
    OUTPUT = "output"


class VerificationStatus(str, Enum):
    """验证状态."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    DEGRADED = "degraded"
    REFUSED = "refused"
    SKIPPED = "skipped"


class ClaimType(str, Enum):
    """声明类型（RAGAS 声明分类启发）.

    不同类型的声明采用不同验证策略。
    """

    FACTUAL = "factual"
    NUMERICAL = "numerical"
    CITATION = "citation"
    INFERENCE = "inference"
    OPINION = "opinion"
    DEFINITION = "definition"


class EvidenceType(str, Enum):
    """证据类型（LlamaIndex Citation + L6 KPA 溯源启发）.

    支撑声明正确性的证据来源类型。
    """

    RETRIEVED_CONTEXT = "retrieved_context"
    KNOWLEDGE_BASE = "knowledge_base"
    EXTERNAL_SOURCE = "external_source"
    COMPUTED = "computed"
    CONSENSUS = "consensus"
    NONE = "none"


class VerifierType(str, Enum):
    """验证器类型（Guardrails AI Validator 注册启发）.

    内置四种验证器，每种对应不同防幻觉策略。
    """

    CITATION = "citation"
    GROUNDEDNESS = "groundedness"
    CONSISTENCY = "consistency"
    FACT_CHECK = "fact_check"
    CUSTOM = "custom"


class VerdictAction(str, Enum):
    """判决动作（Guardrails AI on_fail 修正策略启发）.

    验证失败时执行的动作。
    """

    PASS = "pass"
    REFUSE = "refuse"
    DEGRADE = "degrade"
    FIX = "fix"
    REASK = "reask"
    LOG_ONLY = "log_only"


class HallucinationSeverity(str, Enum):
    """幻觉严重级别."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# 声明与证据
# ============================================================


class Claim(BaseModel):
    """原子声明（RAGAS 声明分解启发）.

    将 LLM 输出分解为最小可验证的事实单元。
    每个声明独立验证，支持逐条溯源。

    Attributes:
        claim_id: 声明唯一 ID
        text: 声明文本
        claim_type: 声明类型
        source_span: 在原始输出中的位置（字符偏移）
        evidence_ids: 关联的证据 ID 列表
        metadata: 额外元数据
    """

    claim_id: str = Field(default_factory=lambda: f"claim-{uuid.uuid4().hex[:10]}")
    text: str = Field(description="声明文本")
    claim_type: ClaimType = Field(default=ClaimType.FACTUAL)
    source_span: tuple[int, int] | None = Field(default=None, description="原文位置 [start, end)")
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """验证证据（LlamaIndex Citation + L6 KPA 溯源启发）.

    支撑声明正确性的来源信息。

    Attributes:
        evidence_id: 证据唯一 ID
        evidence_type: 证据类型
        content: 证据内容文本
        source_uri: 来源 URI（如知识库节点 ID、外部 URL）
        confidence: 证据置信度 (0-1)
        metadata: 额外元数据（如 retrieval_score）
    """

    evidence_id: str = Field(default_factory=lambda: f"evd-{uuid.uuid4().hex[:10]}")
    evidence_type: EvidenceType = Field(default=EvidenceType.NONE)
    content: str = Field(default="", description="证据内容")
    source_uri: str = Field(default="", description="来源 URI")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="证据置信度")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# 验证结果
# ============================================================


class ClaimVerificationResult(BaseModel):
    """单条声明验证结果（Guardrails ValidationResult 启发）.

    Attributes:
        claim_id: 被验证的声明 ID
        verifier_type: 验证器类型
        passed: 是否通过验证
        score: 验证分数 (0-1)，1 = 完全通过
        confidence: 验证置信度 (0-1)
        evidence_ids: 支持该结论的证据 ID
        reason: 验证理由
        metadata: 额外元数据
    """

    claim_id: str = Field(description="声明 ID")
    verifier_type: VerifierType = Field(description="验证器类型")
    passed: bool = Field(description="是否通过")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="验证分数 0-1")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="验证置信度 0-1")
    evidence_ids: list[str] = Field(default_factory=list, description="支持证据 ID")
    reason: str = Field(default="", description="验证理由")
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationReport(BaseModel):
    """完整验证报告（RAGAS 综合评估 + Guardrails Guard 结果启发）.

    聚合所有声明的验证结果，生成综合评估。

    Attributes:
        report_id: 报告 ID
        request_id: 原始请求 ID
        agent_id: 生成输出的 Agent ID
        original_output: 原始输出文本
        claims: 提取的声明列表
        evidence: 收集的证据列表
        claim_results: 各声明验证结果
        overall_score: 综合验证分数 (0-1)
        overall_confidence: 综合置信度 (0-1)
        faithfulness: 忠实度分数 (0-1) — 有证据支持的声明比例
        consistency: 一致性分数 (0-1) — 多采样一致性
        citation_coverage: 引用覆盖率 (0-1) — 有引用的声明比例
        hallucination_detected: 是否检测到幻觉
        hallucination_severity: 幻觉严重级别
        action: 最终判决动作
        corrected_output: 修正后的输出（action=fix 时填充）
        stage: 当前验证阶段
        status: 验证状态
        created_at: 创建时间
        completed_at: 完成时间
        metadata: 额外元数据
    """

    report_id: str = Field(default_factory=lambda: f"vrpt-{uuid.uuid4().hex[:12]}")
    request_id: str = Field(default="")
    agent_id: str = Field(default="")
    original_output: str = Field(default="")
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    claim_results: list[ClaimVerificationResult] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_detected: bool = Field(default=False)
    hallucination_severity: HallucinationSeverity = Field(default=HallucinationSeverity.NONE)
    action: VerdictAction = Field(default=VerdictAction.PASS)
    corrected_output: str | None = Field(default=None)
    stage: VerificationStage = Field(default=VerificationStage.INPUT)
    status: VerificationStatus = Field(default=VerificationStatus.PENDING)
    created_at: float = Field(default_factory=time.time)
    completed_at: float | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def claim_count(self) -> int:
        """声明总数."""
        return len(self.claims)

    @property
    def verified_count(self) -> int:
        """已验证声明数."""
        return len(self.claim_results)

    @property
    def passed_count(self) -> int:
        """通过验证的声明数."""
        return sum(1 for r in self.claim_results if r.passed)

    @property
    def failed_count(self) -> int:
        """未通过验证的声明数."""
        return sum(1 for r in self.claim_results if not r.passed)

    def compute_scores(self) -> None:
        """计算综合分数（RAGAS 调和平均启发）.

        - faithfulness: 通过 groundedness 验证的声明比例
        - consistency: 通过 consistency 验证的声明平均分
        - citation_coverage: 有证据关联的声明比例
        - overall_score: 三项的加权平均
        """
        if not self.claims:
            self.overall_score = 1.0
            self.overall_confidence = 1.0
            self.faithfulness = 1.0
            self.consistency = 1.0
            self.citation_coverage = 1.0
            return

        # 按验证器分组结果
        by_verifier: dict[str, list[ClaimVerificationResult]] = {}
        for r in self.claim_results:
            by_verifier.setdefault(r.verifier_type.value, []).append(r)

        # 忠实度：groundedness 验证通过率
        grounded_results = by_verifier.get(VerifierType.GROUNDEDNESS.value, [])
        if grounded_results:
            self.faithfulness = sum(r.score for r in grounded_results) / len(grounded_results)
        else:
            self.faithfulness = 1.0  # 无 groundedness 验证器视为完全忠实

        # 一致性：consistency 验证平均分
        consistency_results = by_verifier.get(VerifierType.CONSISTENCY.value, [])
        if consistency_results:
            self.consistency = sum(r.score for r in consistency_results) / len(consistency_results)
        else:
            self.consistency = 1.0

        # 引用覆盖率：有证据关联的声明比例
        claims_with_evidence = sum(1 for c in self.claims if c.evidence_ids)
        self.citation_coverage = claims_with_evidence / len(self.claims) if self.claims else 1.0

        # 综合分数：加权平均（忠实度 50%、一致性 30%、引用覆盖 20%）
        self.overall_score = round(
            self.faithfulness * 0.5 + self.consistency * 0.3 + self.citation_coverage * 0.2,
            4,
        )

        # 综合置信度：所有验证结果的平均置信度
        if self.claim_results:
            self.overall_confidence = round(
                sum(r.confidence for r in self.claim_results) / len(self.claim_results),
                4,
            )
        else:
            self.overall_confidence = 1.0

    def determine_hallucination(self) -> None:
        """根据分数判定幻觉严重级别."""
        if self.overall_score >= 0.9:
            self.hallucination_detected = False
            self.hallucination_severity = HallucinationSeverity.NONE
        elif self.overall_score >= 0.7:
            self.hallucination_detected = False
            self.hallucination_severity = HallucinationSeverity.LOW
        elif self.overall_score >= 0.5:
            self.hallucination_detected = True
            self.hallucination_severity = HallucinationSeverity.MEDIUM
        elif self.overall_score >= 0.3:
            self.hallucination_detected = True
            self.hallucination_severity = HallucinationSeverity.HIGH
        else:
            self.hallucination_detected = True
            self.hallucination_severity = HallucinationSeverity.CRITICAL


# ============================================================
# 验证请求
# ============================================================


class VerificationRequest(BaseModel):
    """验证请求.

    CC1 防幻觉管道的输入，描述待验证的 LLM 输出及其上下文。

    Attributes:
        request_id: 请求 ID
        agent_id: 生成输出的 Agent ID
        output_text: LLM 输出文本
        context_chunks: 检索到的上下文片段列表（RAG 场景）
        citations: 提供的引用列表
        reference_answer: 参考答案（如有）
        sample_outputs: 采样输出列表（SelfCheckGPT 场景）
        domain: 领域标识
        metadata: 额外元数据
    """

    request_id: str = Field(default_factory=lambda: f"vreq-{uuid.uuid4().hex[:12]}")
    agent_id: str = Field(default="")
    output_text: str = Field(description="待验证的 LLM 输出")
    context_chunks: list[str] = Field(default_factory=list, description="检索上下文片段")
    citations: list[str] = Field(default_factory=list, description="提供的引用 URI 列表")
    reference_answer: str | None = Field(default=None, description="参考答案")
    sample_outputs: list[str] = Field(default_factory=list, description="采样输出列表")
    domain: str = Field(default="", description="领域标识")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# 验证器配置
# ============================================================


class VerifierConfig(BaseModel):
    """验证器配置.

    定义单个验证器的运行参数。

    Attributes:
        verifier_type: 验证器类型
        enabled: 是否启用
        weight: 权重（用于综合评分）
        threshold: 通过阈值（分数 >= threshold 视为通过）
        params: 验证器特定参数
    """

    verifier_type: VerifierType = Field(description="验证器类型")
    enabled: bool = Field(default=True)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """防幻觉管道配置.

    Attributes:
        verifiers: 验证器配置列表
        pass_threshold: 综合通过阈值
        degrade_threshold: 降级阈值（低于此值降级输出）
        refuse_threshold: 拒绝阈值（低于此值拒绝输出）
        max_claims: 最大声明提取数
        enable_claim_correction: 是否启用声明修正
        enable_corrected_output: 是否生成修正输出
    """

    verifiers: list[VerifierConfig] = Field(default_factory=list)
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    degrade_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    refuse_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_claims: int = Field(default=50, ge=1, le=500)
    enable_claim_correction: bool = Field(default=True)
    enable_corrected_output: bool = Field(default=True)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> PipelineConfig:
        """确保阈值单调递减."""
        if self.refuse_threshold > self.degrade_threshold:
            self.refuse_threshold = self.degrade_threshold
        if self.degrade_threshold > self.pass_threshold:
            self.degrade_threshold = self.pass_threshold
        return self

    def get_verifier_config(self, vtype: VerifierType) -> VerifierConfig | None:
        """获取指定类型的验证器配置."""
        for vc in self.verifiers:
            if vc.verifier_type == vtype and vc.enabled:
                return vc
        return None


# ============================================================
# 幻觉检测记录
# ============================================================


class HallucinationRecord(BaseModel):
    """幻觉检测记录.

    当检测到幻觉时生成的结构化记录，用于审计和度量。

    Attributes:
        record_id: 记录 ID
        report_id: 关联验证报告 ID
        agent_id: 生成输出的 Agent ID
        severity: 严重级别
        failed_claims: 未通过验证的声明文本列表
        action_taken: 采取的动作
        original_score: 原始验证分数
        created_at: 创建时间
    """

    record_id: str = Field(default_factory=lambda: f"hall-{uuid.uuid4().hex[:12]}")
    report_id: str = Field(default="")
    agent_id: str = Field(default="")
    severity: HallucinationSeverity = Field(default=HallucinationSeverity.LOW)
    failed_claims: list[str] = Field(default_factory=list)
    action_taken: VerdictAction = Field(default=VerdictAction.LOG_ONLY)
    original_score: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: float = Field(default_factory=time.time)
