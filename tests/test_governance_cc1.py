"""G3 CC1 防幻觉层测试.

覆盖 CC1 防幻觉层的全部能力：
- 数据模型: 枚举、Claim、Evidence、VerificationReport 评分与幻觉判定
- 异常体系: CC1Error 层级、JSON-RPC 错误码
- 验证器: Citation/Groundedness/Consistency/FactCheck 四种内置验证器
- 验证器注册表: 注册、查找、自定义验证器
- 声明提取器: 分句、类型推断、位置追踪
- 证据收集器: 上下文/引用证据化、相似度关联
- 防幻觉管道: 四阶段流程、动作判决、修正输出、幻觉记录
- 验证存储: 报告/记录管理、多维查询、FIFO 淘汰、统计导出
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from dy3_polaris.l0.cc1 import (
    # 枚举
    ClaimType,
    EvidenceType,
    HallucinationSeverity,
    PipelineConfig,
    VerificationStage,
    VerificationStatus,
    VerdictAction,
    VerifierType,
    # 模型
    Claim,
    ClaimVerificationResult,
    Evidence,
    HallucinationRecord,
    VerificationReport,
    VerificationRequest,
    VerifierConfig,
    # 异常
    CC1Error,
    ClaimExtractionError,
    EvidenceInsufficientError,
    HallucinationDetectedError,
    PipelineConfigError,
    VerificationError,
    VerifierNotFoundError,
    # 验证器
    BaseVerifier,
    CitationVerifier,
    ConsistencyVerifier,
    FactCheckVerifier,
    GroundednessVerifier,
    VerifierRegistry,
    # 管道
    AntiHallucinationPipeline,
    ClaimExtractor,
    EvidenceCollector,
    # 存储
    VerificationStore,
)


# ============================================================
# 辅助工具
# ============================================================


def _make_claim(
    text: str = "测试声明",
    claim_type: ClaimType = ClaimType.FACTUAL,
    evidence_ids: list[str] | None = None,
) -> Claim:
    """快捷构建声明."""
    return Claim(
        text=text,
        claim_type=claim_type,
        evidence_ids=evidence_ids or [],
    )


def _make_evidence(
    content: str = "测试证据",
    evidence_type: EvidenceType = EvidenceType.RETRIEVED_CONTEXT,
    confidence: float = 0.8,
) -> Evidence:
    """快捷构建证据."""
    return Evidence(
        content=content,
        evidence_type=evidence_type,
        confidence=confidence,
    )


def _make_request(
    output_text: str = "水的沸点是100度。",
    context_chunks: list[str] | None = None,
    citations: list[str] | None = None,
    sample_outputs: list[str] | None = None,
    reference_answer: str | None = None,
    agent_id: str = "agent-test",
) -> VerificationRequest:
    """快捷构建验证请求."""
    return VerificationRequest(
        agent_id=agent_id,
        output_text=output_text,
        context_chunks=context_chunks or [],
        citations=citations or [],
        sample_outputs=sample_outputs or [],
        reference_answer=reference_answer,
    )


# ============================================================
# 枚举测试
# ============================================================


class Test枚举值:
    """验证所有枚举的字符串值."""

    def test_验证阶段枚举(self):
        assert VerificationStage.INPUT.value == "input"
        assert VerificationStage.CLAIM_EXTRACTION.value == "claim_extraction"
        assert VerificationStage.VERIFICATION.value == "verification"
        assert VerificationStage.OUTPUT.value == "output"

    def test_验证状态枚举(self):
        assert VerificationStatus.PENDING.value == "pending"
        assert VerificationStatus.IN_PROGRESS.value == "in_progress"
        assert VerificationStatus.PASSED.value == "passed"
        assert VerificationStatus.FAILED.value == "failed"
        assert VerificationStatus.DEGRADED.value == "degraded"
        assert VerificationStatus.REFUSED.value == "refused"
        assert VerificationStatus.SKIPPED.value == "skipped"

    def test_声明类型枚举(self):
        assert ClaimType.FACTUAL.value == "factual"
        assert ClaimType.NUMERICAL.value == "numerical"
        assert ClaimType.CITATION.value == "citation"
        assert ClaimType.INFERENCE.value == "inference"
        assert ClaimType.OPINION.value == "opinion"
        assert ClaimType.DEFINITION.value == "definition"

    def test_证据类型枚举(self):
        assert EvidenceType.RETRIEVED_CONTEXT.value == "retrieved_context"
        assert EvidenceType.KNOWLEDGE_BASE.value == "knowledge_base"
        assert EvidenceType.EXTERNAL_SOURCE.value == "external_source"
        assert EvidenceType.COMPUTED.value == "computed"
        assert EvidenceType.CONSENSUS.value == "consensus"
        assert EvidenceType.NONE.value == "none"

    def test_验证器类型枚举(self):
        assert VerifierType.CITATION.value == "citation"
        assert VerifierType.GROUNDEDNESS.value == "groundedness"
        assert VerifierType.CONSISTENCY.value == "consistency"
        assert VerifierType.FACT_CHECK.value == "fact_check"
        assert VerifierType.CUSTOM.value == "custom"

    def test_判决动作枚举(self):
        assert VerdictAction.PASS.value == "pass"
        assert VerdictAction.REFUSE.value == "refuse"
        assert VerdictAction.DEGRADE.value == "degrade"
        assert VerdictAction.FIX.value == "fix"
        assert VerdictAction.REASK.value == "reask"
        assert VerdictAction.LOG_ONLY.value == "log_only"

    def test_幻觉严重级别枚举(self):
        assert HallucinationSeverity.NONE.value == "none"
        assert HallucinationSeverity.LOW.value == "low"
        assert HallucinationSeverity.MEDIUM.value == "medium"
        assert HallucinationSeverity.HIGH.value == "high"
        assert HallucinationSeverity.CRITICAL.value == "critical"

    def test_枚举为字符串子类(self):
        """确保所有枚举继承 (str, Enum) 以支持 JSON 序列化."""
        assert isinstance(VerificationStage.INPUT, str)
        assert isinstance(ClaimType.FACTUAL, str)
        assert isinstance(VerifierType.CITATION, str)
        assert isinstance(VerdictAction.PASS, str)


# ============================================================
# 数据模型测试
# ============================================================


class TestClaim模型:
    """Claim 声明模型测试."""

    def test_默认创建(self):
        claim = Claim(text="测试声明")
        assert claim.text == "测试声明"
        assert claim.claim_type == ClaimType.FACTUAL
        assert claim.claim_id.startswith("claim-")
        assert claim.evidence_ids == []
        assert claim.source_span is None

    def test_带类型和位置创建(self):
        claim = Claim(
            text="水的沸点是100度",
            claim_type=ClaimType.NUMERICAL,
            source_span=(0, 8),
        )
        assert claim.claim_type == ClaimType.NUMERICAL
        assert claim.source_span == (0, 8)

    def test_自动生成唯一ID(self):
        c1 = Claim(text="声明1")
        c2 = Claim(text="声明2")
        assert c1.claim_id != c2.claim_id

    def test_元数据字段(self):
        claim = Claim(text="测试", metadata={"source": "wiki"})
        assert claim.metadata["source"] == "wiki"


class TestEvidence模型:
    """Evidence 证据模型测试."""

    def test_默认创建(self):
        evd = Evidence()
        assert evd.evidence_type == EvidenceType.NONE
        assert evd.content == ""
        assert evd.confidence == 0.0
        assert evd.evidence_id.startswith("evd-")

    def test_置信度范围约束(self):
        with pytest.raises(Exception):
            Evidence(confidence=1.5)
        with pytest.raises(Exception):
            Evidence(confidence=-0.1)

    def test_完整创建(self):
        evd = Evidence(
            evidence_type=EvidenceType.RETRIEVED_CONTEXT,
            content="水在标准大气压下沸点为100摄氏度",
            source_uri="context://0",
            confidence=0.85,
        )
        assert evd.evidence_type == EvidenceType.RETRIEVED_CONTEXT
        assert evd.confidence == 0.85


class TestClaimVerificationResult:
    """单条声明验证结果测试."""

    def test_默认创建(self):
        result = ClaimVerificationResult(
            claim_id="claim-001",
            verifier_type=VerifierType.CITATION,
            passed=True,
        )
        assert result.claim_id == "claim-001"
        assert result.verifier_type == VerifierType.CITATION
        assert result.passed is True
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_分数范围约束(self):
        with pytest.raises(Exception):
            ClaimVerificationResult(
                claim_id="c1",
                verifier_type=VerifierType.CITATION,
                passed=True,
                score=1.5,
            )


class TestVerificationReport评分:
    """VerificationReport 综合评分与幻觉判定测试."""

    def test_无声明时满分(self):
        report = VerificationReport()
        report.compute_scores()
        assert report.overall_score == 1.0
        assert report.faithfulness == 1.0
        assert report.consistency == 1.0
        assert report.citation_coverage == 1.0

    def test_无声明无幻觉(self):
        report = VerificationReport()
        report.compute_scores()
        report.determine_hallucination()
        assert report.hallucination_detected is False
        assert report.hallucination_severity == HallucinationSeverity.NONE

    def test_全部通过高分无幻觉(self):
        report = VerificationReport(
            claims=[_make_claim("声明1"), _make_claim("声明2")],
            claim_results=[
                ClaimVerificationResult(
                    claim_id="c1",
                    verifier_type=VerifierType.GROUNDEDNESS,
                    passed=True,
                    score=0.95,
                    confidence=0.9,
                ),
                ClaimVerificationResult(
                    claim_id="c2",
                    verifier_type=VerifierType.GROUNDEDNESS,
                    passed=True,
                    score=0.90,
                    confidence=0.85,
                ),
                ClaimVerificationResult(
                    claim_id="c1",
                    verifier_type=VerifierType.CONSISTENCY,
                    passed=True,
                    score=0.92,
                    confidence=0.8,
                ),
                ClaimVerificationResult(
                    claim_id="c2",
                    verifier_type=VerifierType.CONSISTENCY,
                    passed=True,
                    score=0.88,
                    confidence=0.8,
                ),
            ],
        )
        report.claims[0].evidence_ids = ["evd-1"]
        report.claims[1].evidence_ids = ["evd-2"]
        report.compute_scores()
        assert report.faithfulness > 0.9
        assert report.consistency > 0.85
        assert report.citation_coverage == 1.0
        assert report.overall_score > 0.9
        report.determine_hallucination()
        assert report.hallucination_severity == HallucinationSeverity.NONE

    def test_低分检测到幻觉(self):
        report = VerificationReport(
            claims=[_make_claim("声明1")],
            claim_results=[
                ClaimVerificationResult(
                    claim_id="c1",
                    verifier_type=VerifierType.GROUNDEDNESS,
                    passed=False,
                    score=0.1,
                    confidence=0.9,
                ),
                ClaimVerificationResult(
                    claim_id="c1",
                    verifier_type=VerifierType.CONSISTENCY,
                    passed=False,
                    score=0.1,
                    confidence=0.9,
                ),
            ],
        )
        report.compute_scores()
        # faithfulness=0.1, consistency=0.1, citation_coverage=0
        # overall = 0.1*0.5 + 0.1*0.3 + 0*0.2 = 0.08
        assert report.faithfulness < 0.2
        assert report.overall_score < 0.3
        report.determine_hallucination()
        assert report.hallucination_detected is True
        assert report.hallucination_severity == HallucinationSeverity.CRITICAL

    def test_严重级别阈值(self):
        """测试各严重级别阈值边界."""
        # >= 0.9 → NONE
        r = VerificationReport(overall_score=0.9)
        r.determine_hallucination()
        assert r.hallucination_severity == HallucinationSeverity.NONE

        # >= 0.7 → LOW
        r = VerificationReport(overall_score=0.7)
        r.determine_hallucination()
        assert r.hallucination_severity == HallucinationSeverity.LOW

        # >= 0.5 → MEDIUM
        r = VerificationReport(overall_score=0.5)
        r.determine_hallucination()
        assert r.hallucination_severity == HallucinationSeverity.MEDIUM
        assert r.hallucination_detected is True

        # >= 0.3 → HIGH
        r = VerificationReport(overall_score=0.3)
        r.determine_hallucination()
        assert r.hallucination_severity == HallucinationSeverity.HIGH

        # < 0.3 → CRITICAL
        r = VerificationReport(overall_score=0.29)
        r.determine_hallucination()
        assert r.hallucination_severity == HallucinationSeverity.CRITICAL

    def test_属性计数器(self):
        report = VerificationReport(
            claims=[_make_claim("c1"), _make_claim("c2"), _make_claim("c3")],
            claim_results=[
                ClaimVerificationResult(claim_id="c1", verifier_type=VerifierType.CITATION, passed=True),
                ClaimVerificationResult(claim_id="c2", verifier_type=VerifierType.CITATION, passed=False),
                ClaimVerificationResult(claim_id="c3", verifier_type=VerifierType.CITATION, passed=True),
            ],
        )
        assert report.claim_count == 3
        assert report.verified_count == 3
        assert report.passed_count == 2
        assert report.failed_count == 1

    def test_无groundedness验证器时忠实度默认满分(self):
        report = VerificationReport(
            claims=[_make_claim("c1")],
            claim_results=[
                ClaimVerificationResult(
                    claim_id="c1",
                    verifier_type=VerifierType.CITATION,
                    passed=True,
                    score=1.0,
                    confidence=0.9,
                ),
            ],
        )
        report.compute_scores()
        assert report.faithfulness == 1.0


class TestPipelineConfig:
    """管道配置模型测试."""

    def test_默认配置(self):
        config = PipelineConfig()
        assert config.pass_threshold == 0.7
        assert config.degrade_threshold == 0.5
        assert config.refuse_threshold == 0.3
        assert config.max_claims == 50
        assert config.enable_claim_correction is True

    def test_阈值单调递减校验(self):
        """refuse_threshold 不能大于 degrade_threshold."""
        config = PipelineConfig(
            pass_threshold=0.8,
            degrade_threshold=0.6,
            refuse_threshold=0.9,  # 应被修正为 0.6
        )
        assert config.refuse_threshold <= config.degrade_threshold

    def test_阈值递减校验_降级大于通过(self):
        """degrade_threshold 不能大于 pass_threshold."""
        config = PipelineConfig(
            pass_threshold=0.5,
            degrade_threshold=0.8,  # 应被修正为 0.5
            refuse_threshold=0.3,
        )
        assert config.degrade_threshold <= config.pass_threshold

    def test_获取验证器配置(self):
        config = PipelineConfig(
            verifiers=[
                VerifierConfig(verifier_type=VerifierType.CITATION, enabled=True),
                VerifierConfig(verifier_type=VerifierType.GROUNDEDNESS, enabled=False),
            ],
        )
        vc = config.get_verifier_config(VerifierType.CITATION)
        assert vc is not None
        assert vc.verifier_type == VerifierType.CITATION

        # 禁用的验证器不应返回
        assert config.get_verifier_config(VerifierType.GROUNDEDNESS) is None
        # 未配置的验证器返回 None
        assert config.get_verifier_config(VerifierType.FACT_CHECK) is None


class TestHallucinationRecord模型:
    """幻觉检测记录测试."""

    def test_默认创建(self):
        record = HallucinationRecord()
        assert record.severity == HallucinationSeverity.LOW
        assert record.action_taken == VerdictAction.LOG_ONLY
        assert record.record_id.startswith("hall-")

    def test_完整创建(self):
        record = HallucinationRecord(
            report_id="vrpt-001",
            agent_id="agent-001",
            severity=HallucinationSeverity.HIGH,
            failed_claims=["声明A", "声明B"],
            action_taken=VerdictAction.DEGRADE,
            original_score=0.35,
        )
        assert record.severity == HallucinationSeverity.HIGH
        assert len(record.failed_claims) == 2
        assert record.original_score == 0.35


# ============================================================
# 异常体系测试
# ============================================================


class Test异常体系:
    """CC1 异常层级与 JSON-RPC 错误码测试."""

    def test_CC1Error继承L6Error(self):
        from dy3_polaris.l6.core.exceptions import L6Error
        assert issubclass(CC1Error, L6Error)

    def test_基础异常JSONRPC码(self):
        err = CC1Error("测试错误")
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32200

    def test_VerificationError(self):
        err = VerificationError("验证失败")
        assert err.to_json_rpc_error()["code"] == -32201
        assert isinstance(err, CC1Error)

    def test_ClaimExtractionError(self):
        err = ClaimExtractionError("提取失败")
        assert err.to_json_rpc_error()["code"] == -32202

    def test_VerifierNotFoundError(self):
        err = VerifierNotFoundError("custom_verifier")
        assert err.to_json_rpc_error()["code"] == -32203
        assert "custom_verifier" in err.context.get("verifier_type", "")

    def test_HallucinationDetectedError(self):
        err = HallucinationDetectedError(score=0.2, threshold=0.3)
        assert err.to_json_rpc_error()["code"] == -32204
        assert err.context["score"] == 0.2
        assert err.context["threshold"] == 0.3

    def test_PipelineConfigError(self):
        err = PipelineConfigError("配置错误")
        assert err.to_json_rpc_error()["code"] == -32205

    def test_EvidenceInsufficientError(self):
        err = EvidenceInsufficientError(claim_id="claim-001", verifier_type="groundedness")
        assert err.to_json_rpc_error()["code"] == -32206
        assert err.context["claim_id"] == "claim-001"

    def test_异常层级继承(self):
        """所有 CC1 异常都继承 CC1Error."""
        assert issubclass(VerificationError, CC1Error)
        assert issubclass(ClaimExtractionError, CC1Error)
        assert issubclass(VerifierNotFoundError, CC1Error)
        assert issubclass(HallucinationDetectedError, CC1Error)
        assert issubclass(PipelineConfigError, CC1Error)
        assert issubclass(EvidenceInsufficientError, CC1Error)


# ============================================================
# 验证器测试
# ============================================================


class TestCitationVerifier:
    """引用覆盖率验证器测试."""

    def test_有证据需要引用类型(self):
        claim = _make_claim("地球是太阳系第三颗行星", ClaimType.FACTUAL, evidence_ids=["evd-1"])
        verifier = CitationVerifier()
        result = verifier.verify(claim, evidence=[_make_evidence("地球是太阳系第三颗行星")])
        assert result.passed is True
        assert result.score == 1.0
        assert result.verifier_type == VerifierType.CITATION

    def test_无证据需要引用类型(self):
        claim = _make_claim("地球是太阳系第三颗行星", ClaimType.FACTUAL)
        verifier = CitationVerifier()
        result = verifier.verify(claim)
        assert result.passed is False
        assert result.score == 0.0

    def test_无证据不强制引用类型(self):
        claim = _make_claim("我认为这很重要", ClaimType.OPINION)
        verifier = CitationVerifier()
        # score=0.5, 默认 threshold=0.7 → 不通过
        result = verifier.verify(claim)
        assert result.score == 0.5
        assert result.passed is False
        # 降低阈值后可通过
        result_low = verifier.verify(claim, threshold=0.4)
        assert result_low.passed is True

    def test_数值类型需要引用(self):
        claim = _make_claim("光速为299792458米每秒", ClaimType.NUMERICAL)
        verifier = CitationVerifier()
        result = verifier.verify(claim)
        assert result.passed is False
        assert result.score == 0.0

    def test_定义类型需要引用(self):
        claim = _make_claim("光合作用是指植物利用光能合成有机物的过程", ClaimType.DEFINITION)
        verifier = CitationVerifier()
        result = verifier.verify(claim)
        assert result.passed is False
        assert result.score == 0.0


class TestGroundednessVerifier:
    """忠实度验证器测试."""

    def test_无上下文时失败(self):
        claim = _make_claim("水的沸点是100度")
        verifier = GroundednessVerifier()
        result = verifier.verify(claim)
        assert result.passed is False
        assert result.score == 0.0
        assert result.confidence == 1.0

    def test_有匹配上下文时通过(self):
        claim = _make_claim("The boiling point of water is 100 degrees")
        verifier = GroundednessVerifier()
        result = verifier.verify(
            claim,
            context_chunks=["The boiling point of water is 100 degrees Celsius at standard pressure"],
            threshold=0.3,
        )
        assert result.score > 0.3
        assert result.passed is True

    def test_不匹配上下文时低分(self):
        claim = _make_claim("Python is a programming language")
        verifier = GroundednessVerifier()
        result = verifier.verify(
            claim,
            context_chunks=["The weather is nice today"],
            threshold=0.5,
        )
        assert result.score < 0.5

    def test_关键词提取过滤停用词(self):
        keywords = GroundednessVerifier._extract_keywords("The quick brown fox jumps over the lazy dog")
        assert "the" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords


class TestConsistencyVerifier:
    """一致性验证器测试."""

    def test_无采样输出时跳过(self):
        claim = _make_claim("测试声明")
        verifier = ConsistencyVerifier()
        result = verifier.verify(claim)
        assert result.passed is True
        assert result.score == 1.0
        assert "跳过" in result.reason

    def test_高一致性采样通过(self):
        claim = _make_claim("The capital of France is Paris")
        verifier = ConsistencyVerifier()
        result = verifier.verify(
            claim,
            sample_outputs=[
                "The capital of France is Paris",
                "Paris is the capital of France",
            ],
            threshold=0.5,
        )
        assert result.score > 0.5
        assert result.passed is True

    def test_低一致性采样失败(self):
        claim = _make_claim("The capital of France is Paris")
        verifier = ConsistencyVerifier()
        result = verifier.verify(
            claim,
            sample_outputs=[
                "The weather in Tokyo is rainy",
                "Python programming tutorial for beginners",
            ],
            threshold=0.7,
        )
        assert result.passed is False


class TestFactCheckVerifier:
    """事实核查验证器测试."""

    def test_无参考答案时跳过(self):
        claim = _make_claim("测试声明")
        verifier = FactCheckVerifier()
        result = verifier.verify(claim)
        assert result.passed is True
        assert result.score == 1.0

    def test_匹配参考答案通过(self):
        claim = _make_claim("The Earth orbits around the Sun")
        verifier = FactCheckVerifier()
        result = verifier.verify(
            claim,
            reference_answer="The Earth orbits around the Sun in our solar system",
            threshold=0.5,
        )
        assert result.passed is True
        assert result.score > 0.5

    def test_不匹配参考答案失败(self):
        claim = _make_claim("The Earth is flat and stationary")
        verifier = FactCheckVerifier()
        result = verifier.verify(
            claim,
            reference_answer="The Earth is a sphere that orbits the Sun",
            threshold=0.6,
        )
        assert result.passed is False


class TestVerifierRegistry:
    """验证器注册表测试."""

    def test_默认注册四种验证器(self):
        registry = VerifierRegistry()
        assert registry.count == 4
        assert VerifierType.CITATION in registry.list_types()
        assert VerifierType.GROUNDEDNESS in registry.list_types()
        assert VerifierType.CONSISTENCY in registry.list_types()
        assert VerifierType.FACT_CHECK in registry.list_types()

    def test_获取验证器(self):
        registry = VerifierRegistry()
        verifier = registry.get(VerifierType.CITATION)
        assert verifier is not None
        assert verifier.verifier_type == VerifierType.CITATION

    def test_获取不存在的验证器返回None(self):
        registry = VerifierRegistry()
        assert registry.get(VerifierType.CUSTOM) is None

    def test_注册自定义验证器(self):
        class MyCustomVerifier:
            verifier_type = VerifierType.CUSTOM

            def verify(self, claim, **kwargs):
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    verifier_type=VerifierType.CUSTOM,
                    passed=True,
                    score=1.0,
                    confidence=1.0,
                    reason="自定义验证通过",
                )

        registry = VerifierRegistry()
        registry.register(MyCustomVerifier())
        assert registry.count == 5
        verifier = registry.get(VerifierType.CUSTOM)
        assert verifier is not None

        claim = _make_claim("测试")
        result = verifier.verify(claim)
        assert result.passed is True
        assert result.reason == "自定义验证通过"

    def test_注册缺少verifier_type属性抛异常(self):
        class BadVerifier:
            def verify(self, claim, **kwargs):
                pass

        registry = VerifierRegistry()
        with pytest.raises(ValueError, match="verifier_type"):
            registry.register(BadVerifier())

    def test_BaseVerifier协议运行时检查(self):
        verifier = CitationVerifier()
        assert isinstance(verifier, BaseVerifier)


# ============================================================
# 声明提取器测试
# ============================================================


class TestClaimExtractor:
    """声明提取器测试."""

    def test_空文本返回空列表(self):
        extractor = ClaimExtractor()
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []

    def test_中文分句(self):
        extractor = ClaimExtractor()
        text = "水的沸点是100度。地球是圆的。光速很快。"
        claims = extractor.extract(text)
        assert len(claims) == 3
        assert "沸点" in claims[0].text
        assert "地球" in claims[1].text
        assert "光速" in claims[2].text

    def test_英文分句(self):
        extractor = ClaimExtractor()
        text = "Python is a language. Java is also popular. Rust is rising."
        claims = extractor.extract(text)
        assert len(claims) == 3

    def test_换行分句(self):
        extractor = ClaimExtractor()
        text = "第一行内容\n第二行内容\n第三行内容"
        claims = extractor.extract(text)
        assert len(claims) == 3

    def test_类型推断_数字(self):
        extractor = ClaimExtractor()
        claims = extractor.extract("The speed is 299792458 meters per second.")
        assert claims[0].claim_type == ClaimType.NUMERICAL

    def test_类型推断_引用(self):
        extractor = ClaimExtractor()
        claims = extractor.extract("According to [1] the result is confirmed.")
        assert claims[0].claim_type == ClaimType.CITATION

    def test_类型推断_定义(self):
        extractor = ClaimExtractor()
        claims = extractor.extract("Photosynthesis is defined as the process of converting light to energy.")
        assert claims[0].claim_type == ClaimType.DEFINITION

    def test_类型推断_推断(self):
        extractor = ClaimExtractor()
        claims = extractor.extract("Therefore the conclusion is valid.")
        assert claims[0].claim_type == ClaimType.INFERENCE

    def test_类型推断_观点(self):
        extractor = ClaimExtractor()
        claims = extractor.extract("I recommend using Python for this project.")
        assert claims[0].claim_type == ClaimType.OPINION

    def test_类型推断_默认事实(self):
        extractor = ClaimExtractor()
        claims = extractor.extract("The sky is blue today.")
        assert claims[0].claim_type == ClaimType.FACTUAL

    def test_source_span位置追踪(self):
        extractor = ClaimExtractor()
        text = "第一句。第二句。"
        claims = extractor.extract(text)
        assert len(claims) == 2
        for claim in claims:
            assert claim.source_span is not None
            start, end = claim.source_span
            assert text[start:end] == claim.text

    def test_max_claims限制(self):
        extractor = ClaimExtractor()
        text = "。".join([f"声明{i}" for i in range(20)])
        claims = extractor.extract(text, max_claims=5)
        assert len(claims) <= 5

    def test_过滤短句(self):
        extractor = ClaimExtractor()
        text = "ok. 这是完整的句子内容。"
        claims = extractor.extract(text)
        # "ok" 长度 < 3 应被过滤
        assert all(len(c.text) >= 3 for c in claims)


# ============================================================
# 证据收集器测试
# ============================================================


class TestEvidenceCollector:
    """证据收集器测试."""

    def test_从上下文创建证据(self):
        collector = EvidenceCollector()
        request = _make_request(
            context_chunks=["水在标准大气压下沸点为100摄氏度"],
        )
        claims = [_make_claim("水的沸点是100度")]
        evidence = collector.collect(request, claims)
        assert len(evidence) == 1
        assert evidence[0].evidence_type == EvidenceType.RETRIEVED_CONTEXT
        assert evidence[0].confidence == 0.8

    def test_从引用创建证据(self):
        collector = EvidenceCollector()
        request = _make_request(
            citations=["https://example.com/source1"],
        )
        claims = [_make_claim("测试声明")]
        evidence = collector.collect(request, claims)
        assert len(evidence) == 1
        assert evidence[0].evidence_type == EvidenceType.EXTERNAL_SOURCE
        assert evidence[0].source_uri == "https://example.com/source1"

    def test_证据关联到声明(self):
        collector = EvidenceCollector()
        request = _make_request(
            context_chunks=["The boiling point of water is 100 degrees Celsius"],
        )
        claims = [_make_claim("The boiling point of water is 100 degrees")]
        evidence = collector.collect(request, claims)
        assert len(claims[0].evidence_ids) > 0

    def test_无匹配时不关联(self):
        collector = EvidenceCollector()
        # 使用字符完全不重叠的中英文混合文本确保相似度低于阈值
        request = _make_request(
            context_chunks=["zzz qqq xxx"],
        )
        claims = [_make_claim("量子力学描述亚原子粒子的运动规律")]
        collector.collect(request, claims)
        assert len(claims[0].evidence_ids) == 0

    def test_最多关联3个证据(self):
        collector = EvidenceCollector()
        request = _make_request(
            context_chunks=[
                "water boiling point 100 degrees",
                "water boiling point is 100",
                "boiling point of water 100",
                "water boils at 100 degrees",
            ],
        )
        claims = [_make_claim("water boiling point 100 degrees")]
        collector.collect(request, claims)
        assert len(claims[0].evidence_ids) <= 3


# ============================================================
# 防幻觉管道测试
# ============================================================


class TestAntiHallucinationPipeline:
    """防幻觉验证管道测试."""

    def test_默认配置初始化(self):
        pipeline = AntiHallucinationPipeline()
        assert pipeline.config.pass_threshold == 0.7
        assert pipeline.registry.count == 4

    def test_空输出跳过验证(self):
        pipeline = AntiHallucinationPipeline()
        report = pipeline.verify_text("")
        assert report.status == VerificationStatus.SKIPPED
        assert report.action == VerdictAction.PASS

    def test_有上下文支持的输出通过(self):
        pipeline = AntiHallucinationPipeline()
        report = pipeline.verify_text(
            "The boiling point of water is 100 degrees Celsius.",
            context_chunks=["The boiling point of water is 100 degrees Celsius at standard atmospheric pressure"],
        )
        assert report.status == VerificationStatus.PASSED
        assert report.action == VerdictAction.PASS
        assert report.claim_count > 0

    def test_无上下文无引用的输出降级或拒绝(self):
        pipeline = AntiHallucinationPipeline()
        report = pipeline.verify_text(
            "The capital of Mars is New Berlin and the population is 5 million.",
        )
        # 无上下文时 groundedness 分数为 0，应降级或拒绝
        assert report.status in (VerificationStatus.DEGRADED, VerificationStatus.REFUSED)
        assert report.faithfulness < 0.1

    def test_多采样一致性验证(self):
        pipeline = AntiHallucinationPipeline()
        report = pipeline.verify_text(
            "The capital of France is Paris.",
            sample_outputs=[
                "The capital of France is Paris.",
                "Paris is the capital of France.",
            ],
        )
        assert report.consistency > 0.5

    def test_参考答案事实核查(self):
        pipeline = AntiHallucinationPipeline()
        report = pipeline.verify_text(
            "The Earth orbits the Sun.",
            reference_answer="The Earth orbits the Sun in the solar system.",
        )
        # 事实核查验证器应运行
        fact_check_results = [
            r for r in report.claim_results
            if r.verifier_type == VerifierType.FACT_CHECK
        ]
        assert len(fact_check_results) > 0

    def test_四阶段执行顺序(self):
        pipeline = AntiHallucinationPipeline()
        report = pipeline.verify_text(
            "Test statement.",
            context_chunks=["Test statement context"],
        )
        assert report.stage == VerificationStage.OUTPUT
        assert report.completed_at is not None

    def test_修正输出生成(self):
        """当输出部分通过时生成修正输出."""
        pipeline = AntiHallucinationPipeline()
        # 混合输出：一句有上下文支持，一句没有
        report = pipeline.verify_text(
            "The boiling point of water is 100 degrees. "
            "The population of Atlantis is 10 million.",
            context_chunks=["The boiling point of water is 100 degrees Celsius"],
        )
        if report.action in (VerdictAction.FIX, VerdictAction.DEGRADE):
            assert report.corrected_output is not None
            assert "验证修正" in report.corrected_output or len(report.corrected_output) > 0

    def test_create_hallucination_record_无幻觉时返回None(self):
        pipeline = AntiHallucinationPipeline()
        report = pipeline.verify_text(
            "The boiling point of water is 100 degrees Celsius.",
            context_chunks=["The boiling point of water is 100 degrees Celsius"],
        )
        record = pipeline.create_hallucination_record(report)
        assert record is None

    def test_create_hallucination_record_有幻觉时创建记录(self):
        pipeline = AntiHallucinationPipeline()
        report = pipeline.verify_text(
            "The capital of Mars is New Berlin.",
        )
        if report.hallucination_detected:
            record = pipeline.create_hallucination_record(report)
            assert record is not None
            assert record.report_id == report.report_id
            assert record.severity == report.hallucination_severity
            assert record.original_score == report.overall_score

    def test_自定义配置管道(self):
        config = PipelineConfig(
            verifiers=[
                VerifierConfig(verifier_type=VerifierType.CITATION, enabled=True, threshold=0.5),
            ],
            pass_threshold=0.5,
            degrade_threshold=0.3,
            refuse_threshold=0.1,
        )
        pipeline = AntiHallucinationPipeline(config=config)
        assert pipeline.config.pass_threshold == 0.5

    def test_注册自定义验证器(self):
        class AlwaysPassVerifier:
            verifier_type = VerifierType.CUSTOM

            def verify(self, claim, **kwargs):
                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    verifier_type=VerifierType.CUSTOM,
                    passed=True,
                    score=1.0,
                    confidence=1.0,
                    reason="总是通过",
                )

        pipeline = AntiHallucinationPipeline()
        pipeline.register_verifier(AlwaysPassVerifier())
        assert VerifierType.CUSTOM in pipeline.registry.list_types()

    def test_update_config方法(self):
        pipeline = AntiHallucinationPipeline()
        new_config = PipelineConfig(pass_threshold=0.8)
        pipeline.update_config(new_config)
        assert pipeline.config.pass_threshold == 0.8

    def test_验证器异常不影响管道(self):
        """验证器抛异常时应捕获并生成失败结果."""
        class CrashVerifier:
            verifier_type = VerifierType.CUSTOM

            def verify(self, claim, **kwargs):
                raise RuntimeError("验证器崩溃")

        pipeline = AntiHallucinationPipeline()
        pipeline.register_verifier(CrashVerifier())
        config = PipelineConfig(
            verifiers=[
                VerifierConfig(verifier_type=VerifierType.CUSTOM, enabled=True),
            ],
        )
        pipeline.update_config(config)
        report = pipeline.verify_text("测试声明")
        # 管道不应崩溃
        assert report.status in (
            VerificationStatus.PASSED,
            VerificationStatus.DEGRADED,
            VerificationStatus.REFUSED,
            VerificationStatus.FAILED,
        )


# ============================================================
# 验证存储测试
# ============================================================


class TestVerificationStore:
    """验证存储测试."""

    def test_初始化默认容量(self):
        store = VerificationStore()
        assert store.report_count == 0
        assert store.record_count == 0
        assert store.total_verifications == 0

    def test_添加和获取报告(self):
        store = VerificationStore()
        report = VerificationReport(report_id="rpt-001", agent_id="agent-1")
        store.add_report(report)
        assert store.report_count == 1
        retrieved = store.get_report("rpt-001")
        assert retrieved is not None
        assert retrieved.report_id == "rpt-001"

    def test_获取不存在的报告返回None(self):
        store = VerificationStore()
        assert store.get_report("nonexistent") is None

    def test_查询报告_按agent_id(self):
        store = VerificationStore()
        store.add_report(VerificationReport(report_id="r1", agent_id="agent-A"))
        store.add_report(VerificationReport(report_id="r2", agent_id="agent-B"))
        store.add_report(VerificationReport(report_id="r3", agent_id="agent-A"))
        results = store.query_reports(agent_id="agent-A")
        assert len(results) == 2

    def test_查询报告_按状态(self):
        store = VerificationStore()
        store.add_report(VerificationReport(
            report_id="r1",
            status=VerificationStatus.PASSED,
        ))
        store.add_report(VerificationReport(
            report_id="r2",
            status=VerificationStatus.REFUSED,
        ))
        results = store.query_reports(status=VerificationStatus.PASSED)
        assert len(results) == 1
        assert results[0].status == VerificationStatus.PASSED

    def test_查询报告_按分数范围(self):
        store = VerificationStore()
        store.add_report(VerificationReport(report_id="r1", overall_score=0.9))
        store.add_report(VerificationReport(report_id="r2", overall_score=0.4))
        store.add_report(VerificationReport(report_id="r3", overall_score=0.2))
        results = store.query_reports(min_score=0.5)
        assert len(results) == 1
        results = store.query_reports(max_score=0.3)
        assert len(results) == 1

    def test_查询报告_limit限制(self):
        store = VerificationStore()
        for i in range(10):
            store.add_report(VerificationReport(report_id=f"r{i}"))
        results = store.query_reports(limit=3)
        assert len(results) == 3

    def test_添加和获取幻觉记录(self):
        store = VerificationStore()
        record = HallucinationRecord(
            record_id="rec-001",
            report_id="rpt-001",
            agent_id="agent-1",
            severity=HallucinationSeverity.HIGH,
        )
        store.add_record(record)
        assert store.record_count == 1
        retrieved = store.get_record("rec-001")
        assert retrieved is not None
        assert retrieved.severity == HallucinationSeverity.HIGH

    def test_查询幻觉记录_按严重级别(self):
        store = VerificationStore()
        store.add_record(HallucinationRecord(
            record_id="r1", severity=HallucinationSeverity.HIGH,
        ))
        store.add_record(HallucinationRecord(
            record_id="r2", severity=HallucinationSeverity.LOW,
        ))
        results = store.query_records(severity=HallucinationSeverity.HIGH)
        assert len(results) == 1

    def test_统计信息(self):
        store = VerificationStore()
        store.add_report(VerificationReport(
            report_id="r1",
            status=VerificationStatus.PASSED,
            action=VerdictAction.PASS,
        ))
        store.add_report(VerificationReport(
            report_id="r2",
            status=VerificationStatus.REFUSED,
            action=VerdictAction.REFUSE,
        ))
        store.add_record(HallucinationRecord(
            record_id="rec1",
            severity=HallucinationSeverity.MEDIUM,
        ))
        stats = store.get_stats()
        assert stats["total_verifications"] == 2
        assert stats["passed"] == 1
        assert stats["refused"] == 1
        assert stats["hallucination_total"] == 1
        assert "by_severity" in stats
        assert "by_action" in stats

    def test_FIFO淘汰(self):
        store = VerificationStore(max_reports=3)
        for i in range(5):
            store.add_report(VerificationReport(report_id=f"r{i}"))
        assert store.report_count == 3
        # 最早的应该被淘汰
        assert store.get_report("r0") is None
        assert store.get_report("r1") is None
        assert store.get_report("r2") is not None
        assert store.get_report("r4") is not None
        # 总计数不受淘汰影响
        assert store.total_verifications == 5

    def test_导出全部数据(self):
        store = VerificationStore()
        store.add_report(VerificationReport(report_id="r1"))
        store.add_record(HallucinationRecord(record_id="rec1"))
        data = store.export_all()
        assert "reports" in data
        assert "records" in data
        assert "stats" in data
        assert len(data["reports"]) == 1
        assert len(data["records"]) == 1

    def test_导出摘要(self):
        store = VerificationStore()
        store.add_report(VerificationReport(report_id="r1"))
        summary = store.export_summary()
        assert "report_ids" in summary
        assert "record_ids" in summary
        assert "stats" in summary
        assert "r1" in summary["report_ids"]

    def test_清空存储(self):
        store = VerificationStore()
        store.add_report(VerificationReport(report_id="r1"))
        store.add_record(HallucinationRecord(record_id="rec1"))
        store.clear()
        assert store.report_count == 0
        assert store.record_count == 0
        assert store.total_verifications == 0

    def test_线程安全并发写入(self):
        store = VerificationStore(max_reports=200)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(20):
                    store.add_report(VerificationReport(
                        report_id=f"r{start}_{i}",
                        agent_id=f"agent-{start}",
                    ))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.total_verifications == 100
        assert store.report_count <= 200


# ============================================================
# 端到端集成测试
# ============================================================


class Test端到端集成:
    """CC1 防幻觉层端到端集成测试."""

    def test_完整验证流程_高质输出通过(self):
        """模拟完整 RAG 场景：有上下文支持的高质量输出."""
        pipeline = AntiHallucinationPipeline()
        store = VerificationStore()

        report = pipeline.verify_text(
            "The boiling point of water is 100 degrees Celsius at standard atmospheric pressure. "
            "This was first measured by Daniel Fahrenheit.",
            context_chunks=[
                "The boiling point of water is 100 degrees Celsius at standard atmospheric pressure",
                "Daniel Fahrenheit invented the mercury thermometer",
            ],
            citations=["https://en.wikipedia.org/wiki/Boiling_point"],
            agent_id="agent-tutor",
            domain="chemistry",
        )

        store.add_report(report)
        assert store.report_count == 1
        assert report.status == VerificationStatus.PASSED
        assert report.claim_count >= 2
        assert report.overall_score > 0.5

    def test_完整验证流程_幻觉输出被拒绝(self):
        """模拟无支撑的幻觉输出被拒绝."""
        pipeline = AntiHallucinationPipeline()
        store = VerificationStore()

        report = pipeline.verify_text(
            "The capital of Mars is New Berlin and has a population of 5 million people.",
            agent_id="agent-hallucinator",
        )

        store.add_report(report)

        # 无上下文 → groundedness 全部失败 → 低分
        assert report.faithfulness < 0.1
        assert report.overall_score < 0.5

        # 创建幻觉记录
        record = pipeline.create_hallucination_record(report)
        if record:
            store.add_record(record)
            assert store.record_count == 1

    def test_完整验证流程_部分正确输出被修正(self):
        """模拟部分正确的输出被修正（移除错误声明）."""
        pipeline = AntiHallucinationPipeline()

        report = pipeline.verify_text(
            "The boiling point of water is 100 degrees Celsius. "
            "The moon is made of cheese.",
            context_chunks=[
                "The boiling point of water is 100 degrees Celsius at standard pressure",
            ],
        )

        # 至少应检测到部分问题
        assert report.claim_count >= 2
        # 如果有未通过的声明且动作是 FIX/DEGRADE，应有修正输出
        if report.action in (VerdictAction.FIX, VerdictAction.DEGRADE):
            assert report.corrected_output is not None

    def test_存储统计端到端(self):
        """完整验证后存储统计正确."""
        pipeline = AntiHallucinationPipeline()
        store = VerificationStore()

        # 通过的验证
        r1 = pipeline.verify_text(
            "Water boils at 100 degrees Celsius.",
            context_chunks=["Water boils at 100 degrees Celsius at standard pressure"],
        )
        store.add_report(r1)

        # 失败的验证
        r2 = pipeline.verify_text(
            "Mars has oceans of liquid water.",
        )
        store.add_report(r2)

        # 幻觉记录
        if r2.hallucination_detected:
            rec = pipeline.create_hallucination_record(r2)
            if rec:
                store.add_record(rec)

        stats = store.get_stats()
        assert stats["total_verifications"] == 2
        assert stats["passed"] >= 1
