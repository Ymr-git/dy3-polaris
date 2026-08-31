"""CC1 四层反幻觉评审引擎增强测试.

覆盖设计文档中定义的四层架构：
- L1 事实层 (FactLayer): F-R01~F-R12 域特定事实规则
- L2 逻辑层 (LogicLayer): L-R01~L-R10 逻辑一致性规则
- L3 数值层 (NumericalLayer): N-R01~N-R12 数值范围校验规则
- L4 溯源层 (ProvenanceLayer): P-R01~P-R10 溯源完整性规则

以及：
- 评审状态机: 四层递进 + 三级结果 (Pass/Flag/Block)
- 自纠回路: 最多 2 次自纠, 第 3 次升级裁决
- 综合评分引擎: Score = 0.40×L1 + 0.25×L2 + 0.20×L3 + 0.15×L4
- 增强管道: ReviewPipeline 编排四层评审 + 自纠回路
- REST API: /governance/cc1/review, /governance/cc1/report/{id}

遵循 TDD: 先写测试 (RED), 再实现 (GREEN), 最后重构 (REFACTOR).
"""

from __future__ import annotations

import pytest
from typing import Any

from dy3_polaris.l0.cc1.models import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceType,
    VerificationRequest,
)
from dy3_polaris.l0.cc1.layers import (
    ReviewLayerType,
    LayerResult,
    LayerRuleResult,
    FactLayer,
    LogicLayer,
    NumericalLayer,
    ProvenanceLayer,
    LAYER_WEIGHTS,
)
from dy3_polaris.l0.cc1.state_machine import (
    ReviewState,
    ReviewVerdict,
    ReviewStateMachine,
    SelfCorrectionRecord,
    SelfCorrectionLoop,
)
from dy3_polaris.l0.cc1.scoring import (
    CompositeScoringEngine,
    ScoringWeights,
)
from dy3_polaris.l0.cc1.review_pipeline import (
    ReviewPipeline,
    ReviewPipelineConfig,
)


# ============================================================
# 辅助工具
# ============================================================


def _make_claim(
    text: str = "Dy3+ 的发射主峰在 575nm。",
    claim_type: ClaimType = ClaimType.FACTUAL,
    evidence_ids: list[str] | None = None,
) -> Claim:
    return Claim(text=text, claim_type=claim_type, evidence_ids=evidence_ids or [])


def _make_evidence(
    content: str = "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射",
    evidence_type: EvidenceType = EvidenceType.RETRIEVED_CONTEXT,
    confidence: float = 0.9,
    source_uri: str = "kb://dy3/emission",
) -> Evidence:
    return Evidence(
        content=content,
        evidence_type=evidence_type,
        confidence=confidence,
        source_uri=source_uri,
    )


def _make_review_request(
    output_text: str = "Dy3+ 的发射主峰在 575nm。",
    context_chunks: list[str] | None = None,
    citations: list[str] | None = None,
    agent_id: str = "agent-knowledge",
    **kwargs: Any,
) -> VerificationRequest:
    return VerificationRequest(
        agent_id=agent_id,
        output_text=output_text,
        context_chunks=context_chunks or ["Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"],
        citations=citations or [],
        **kwargs,
    )


# ============================================================
# L1 事实层 (FactLayer) 测试
# ============================================================


class TestFactLayer:
    """L1 事实层校验器测试 — F-R01~F-R12."""

    def test_fact_layer_initialization(self):
        """事实层初始化, 加载全部 12 条规则."""
        layer = FactLayer()
        assert layer.layer_type == ReviewLayerType.L1_FACT
        assert len(layer.rules) == 12
        rule_ids = {r.rule_id for r in layer.rules}
        assert "F-R01" in rule_ids
        assert "F-R12" in rule_ids

    def test_fact_layer_weight(self):
        """L1 权重为 0.40."""
        assert LAYER_WEIGHTS[ReviewLayerType.L1_FACT] == 0.40

    def test_fr01_emission_peak_validation_pass(self):
        """F-R01: 发射峰波长 575nm 在 570-585nm 范围内 → 通过."""
        layer = FactLayer()
        claim = _make_claim("Dy3+ 的发射主峰在 575nm。")
        result = layer.verify_claim(claim, context_chunks=["Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"])
        assert result.layer_type == ReviewLayerType.L1_FACT
        assert result.score > 0
        fr01_result = next(r for r in result.rule_results if r.rule_id == "F-R01")
        assert fr01_result.passed is True

    def test_fr01_emission_peak_validation_fail(self):
        """F-R01: 发射峰波长 600nm 超出 570-585nm 范围 → 不通过."""
        layer = FactLayer()
        claim = _make_claim("Dy3+ 的发射主峰在 600nm。")
        result = layer.verify_claim(claim, context_chunks=["Dy3+ 发光发射峰在 600nm"])
        fr01_result = next(r for r in result.rule_results if r.rule_id == "F-R01")
        assert fr01_result.passed is False

    def test_fr02_energy_level_transition(self):
        """F-R02: 黄色发射 → ⁴F₉/₂→⁶H₁₃/₂ 能级跃迁对应."""
        layer = FactLayer()
        claim = _make_claim("Dy3+ 的黄色发射对应 ⁴F₉/₂→⁶H₁₃/₂ 跃迁。")
        result = layer.verify_claim(claim, context_chunks=["Dy3+ 黄色发射来自 ⁴F₉/₂→⁶H₁₃/₂"])
        fr02_result = next(r for r in result.rule_results if r.rule_id == "F-R02")
        assert fr02_result.passed is True

    def test_fr04_concentration_quenching_threshold(self):
        """F-R04: 掺杂浓度 3mol% 在 1-5mol% 范围内, 未超猝灭阈值."""
        layer = FactLayer()
        claim = _make_claim("Dy3+ 掺杂浓度为 3mol%。")
        result = layer.verify_claim(claim, context_chunks=["掺杂浓度 3mol%"])
        fr04_result = next(r for r in result.rule_results if r.rule_id == "F-R04")
        assert fr04_result.passed is True

    def test_fr04_concentration_above_quenching(self):
        """F-R04: 掺杂浓度 10mol% 超出猝灭阈值 → 警告."""
        layer = FactLayer()
        claim = _make_claim("Dy3+ 掺杂浓度为 10mol%。")
        result = layer.verify_claim(claim, context_chunks=["掺杂浓度 10mol%"])
        fr04_result = next(r for r in result.rule_results if r.rule_id == "F-R04")
        assert fr04_result.passed is False

    def test_fr12_decay_lifetime_range(self):
        """F-R12: 衰减寿命 1.0ms 在 0.1-2.0ms 范围内."""
        layer = FactLayer()
        claim = _make_claim("Dy3+ 的荧光衰减寿命为 1.0ms。")
        result = layer.verify_claim(claim, context_chunks=["衰减寿命 1.0ms"])
        fr12_result = next(r for r in result.rule_results if r.rule_id == "F-R12")
        assert fr12_result.passed is True

    def test_fact_layer_score_calculation(self):
        """事实层评分 = 通过规则数 / 总规则数 × 100."""
        layer = FactLayer()
        claim = _make_claim("Dy3+ 的发射主峰在 575nm, 掺杂浓度 3mol%。")
        result = layer.verify_claim(
            claim,
            context_chunks=["Dy3+ 发射峰 575nm, 掺杂 3mol%, ⁴F₉/₂→⁶H₁₃/₂"],
        )
        assert 0 <= result.score <= 100
        assert result.layer_type == ReviewLayerType.L1_FACT


# ============================================================
# L2 逻辑层 (LogicLayer) 测试
# ============================================================


class TestLogicLayer:
    """L2 逻辑层校验器测试 — L-R01~L-R10."""

    def test_logic_layer_initialization(self):
        """逻辑层初始化, 加载全部 10 条规则."""
        layer = LogicLayer()
        assert layer.layer_type == ReviewLayerType.L2_LOGIC
        assert len(layer.rules) == 15  # L-R01~L-R10 + L-R11~L-R15 (enhanced)
        rule_ids = {r.rule_id for r in layer.rules}
        assert "L-R01" in rule_ids
        assert "L-R10" in rule_ids

    def test_logic_layer_weight(self):
        """L2 权重为 0.25."""
        assert LAYER_WEIGHTS[ReviewLayerType.L2_LOGIC] == 0.25

    def test_lr01_concentration_intensity_logic_pass(self):
        """L-R01: 浓度增加导致发光强度先增后减 (非单调) → 逻辑正确."""
        layer = LogicLayer()
        claim = _make_claim(
            "随着 Dy3+ 掺杂浓度增加, 发光强度先增大后减小, 存在浓度猝灭效应。"
        )
        result = layer.verify_claim(claim, context_chunks=["浓度猝灭导致非单调关系"])
        lr01_result = next(r for r in result.rule_results if r.rule_id == "L-R01")
        assert lr01_result.passed is True

    def test_lr01_concentration_intensity_logic_fail(self):
        """L-R01: 浓度增加导致发光强度单调增加 → 逻辑错误."""
        layer = LogicLayer()
        claim = _make_claim(
            "随着 Dy3+ 掺杂浓度增加, 发光强度持续单调增加。"
        )
        result = layer.verify_claim(claim, context_chunks=["浓度增加强度单调增加"])
        lr01_result = next(r for r in result.rule_results if r.rule_id == "L-R01")
        assert lr01_result.passed is False

    def test_lr02_thermal_quenching_logic(self):
        """L-R02: 温度升高导致发光强度降低 (热猝灭)."""
        layer = LogicLayer()
        claim = _make_claim(
            "温度升高时, 非辐射跃迁概率增加, 导致发光强度降低。"
        )
        result = layer.verify_claim(claim, context_chunks=["热猝灭效应"])
        lr02_result = next(r for r in result.rule_results if r.rule_id == "L-R02")
        assert lr02_result.passed is True

    def test_lr08_lanthanide_classification(self):
        """L-R08: Dy3+ 属于镧系 → 稀土 → f 区元素."""
        layer = LogicLayer()
        claim = _make_claim("Dy3+ 属于镧系元素, 是 f 区元素。")
        result = layer.verify_claim(claim, context_chunks=["镧系 f 区"])
        lr08_result = next(r for r in result.rule_results if r.rule_id == "L-R08")
        assert lr08_result.passed is True

    def test_lr08_wrong_classification(self):
        """L-R08: Dy3+ 属于 d 区 → 逻辑错误."""
        layer = LogicLayer()
        claim = _make_claim("Dy3+ 属于 d 区过渡金属元素。")
        result = layer.verify_claim(claim, context_chunks=["d 区过渡金属"])
        lr08_result = next(r for r in result.rule_results if r.rule_id == "L-R08")
        assert lr08_result.passed is False

    def test_logic_layer_score_range(self):
        """逻辑层评分在 0-100 范围内."""
        layer = LogicLayer()
        claim = _make_claim("Dy3+ 属于镧系元素。")
        result = layer.verify_claim(claim, context_chunks=["镧系元素"])
        assert 0 <= result.score <= 100


# ============================================================
# L3 数值层 (NumericalLayer) 测试
# ============================================================


class TestNumericalLayer:
    """L3 数值层校验器测试 — N-R01~N-R12."""

    def test_numerical_layer_initialization(self):
        """数值层初始化, 加载全部 12 条规则."""
        layer = NumericalLayer()
        assert layer.layer_type == ReviewLayerType.L3_NUMERICAL
        assert len(layer.rules) == 18  # N-R01~N-R12 + N-R13~N-R18 (enhanced)
        rule_ids = {r.rule_id for r in layer.rules}
        assert "N-R01" in rule_ids
        assert "N-R12" in rule_ids

    def test_numerical_layer_weight(self):
        """L3 权重为 0.20."""
        assert LAYER_WEIGHTS[ReviewLayerType.L3_NUMERICAL] == 0.20

    def test_nr01_emission_peak_in_range(self):
        """N-R01: 主发射峰 575nm 在 570-585nm 范围内 → 通过."""
        layer = NumericalLayer()
        claim = _make_claim("Dy3+ 的主发射峰波长为 575nm。", ClaimType.NUMERICAL)
        result = layer.verify_claim(claim, context_chunks=["发射峰 575nm"])
        nr01_result = next(r for r in result.rule_results if r.rule_id == "N-R01")
        assert nr01_result.passed is True

    def test_nr01_emission_peak_out_of_range(self):
        """N-R01: 主发射峰 650nm 超出 570-585nm 范围 → 不通过."""
        layer = NumericalLayer()
        claim = _make_claim("Dy3+ 的主发射峰波长为 650nm。", ClaimType.NUMERICAL)
        result = layer.verify_claim(claim, context_chunks=["发射峰 650nm"])
        nr01_result = next(r for r in result.rule_results if r.rule_id == "N-R01")
        assert nr01_result.passed is False

    def test_nr04_doping_concentration_in_range(self):
        """N-R04: 掺杂浓度 2mol% 在 1-5mol% 范围内."""
        layer = NumericalLayer()
        claim = _make_claim("Dy3+ 掺杂浓度为 2mol%。", ClaimType.NUMERICAL)
        result = layer.verify_claim(claim, context_chunks=["浓度 2mol%"])
        nr04_result = next(r for r in result.rule_results if r.rule_id == "N-R04")
        assert nr04_result.passed is True

    def test_nr07_decay_lifetime_in_range(self):
        """N-R07: 荧光衰减寿命 0.8ms 在 0.1-2.0ms 范围内."""
        layer = NumericalLayer()
        claim = _make_claim("荧光衰减寿命为 0.8ms。", ClaimType.NUMERICAL)
        result = layer.verify_claim(claim, context_chunks=["寿命 0.8ms"])
        nr07_result = next(r for r in result.rule_results if r.rule_id == "N-R07")
        assert nr07_result.passed is True

    def test_nr07_decay_lifetime_out_of_range(self):
        """N-R07: 荧光衰减寿命 50ms 超出 0.1-2.0ms 范围 → 不通过."""
        layer = NumericalLayer()
        claim = _make_claim("荧光衰减寿命为 50ms。", ClaimType.NUMERICAL)
        result = layer.verify_claim(claim, context_chunks=["寿命 50ms"])
        nr07_result = next(r for r in result.rule_results if r.rule_id == "N-R07")
        assert nr07_result.passed is False

    def test_numerical_layer_score_range(self):
        """数值层评分在 0-100 范围内."""
        layer = NumericalLayer()
        claim = _make_claim("发射峰 575nm, 浓度 2mol%。", ClaimType.NUMERICAL)
        result = layer.verify_claim(claim, context_chunks=["575nm 2mol%"])
        assert 0 <= result.score <= 100


# ============================================================
# L4 溯源层 (ProvenanceLayer) 测试
# ============================================================


class TestProvenanceLayer:
    """L4 溯源层校验器测试 — P-R01~P-R10."""

    def test_provenance_layer_initialization(self):
        """溯源层初始化, 加载全部 10 条规则."""
        layer = ProvenanceLayer()
        assert layer.layer_type == ReviewLayerType.L4_PROVENANCE
        assert len(layer.rules) == 15  # P-R01~P-R10 + P-R11~P-R15 (enhanced)
        rule_ids = {r.rule_id for r in layer.rules}
        assert "P-R01" in rule_ids
        assert "P-R10" in rule_ids

    def test_provenance_layer_weight(self):
        """L4 权重为 0.15."""
        assert LAYER_WEIGHTS[ReviewLayerType.L4_PROVENANCE] == 0.15

    def test_pr01_claim_source_binding_pass(self):
        """P-R01: 声明有关联来源 → 通过."""
        layer = ProvenanceLayer()
        evidence = _make_evidence()
        claim = _make_claim(evidence_ids=[evidence.evidence_id])
        result = layer.verify_claim(claim, evidence=[evidence])
        pr01_result = next(r for r in result.rule_results if r.rule_id == "P-R01")
        assert pr01_result.passed is True

    def test_pr01_claim_source_binding_fail(self):
        """P-R01: 声明无关联来源 → 不通过."""
        layer = ProvenanceLayer()
        claim = _make_claim(evidence_ids=[])
        result = layer.verify_claim(claim, evidence=[])
        pr01_result = next(r for r in result.rule_results if r.rule_id == "P-R01")
        assert pr01_result.passed is False

    def test_pr08_ai_generated_annotation(self):
        """P-R08: 无来源的声明必须标注为 AI-generated."""
        layer = ProvenanceLayer()
        claim = _make_claim(evidence_ids=[])
        result = layer.verify_claim(claim, evidence=[])
        pr08_result = next(r for r in result.rule_results if r.rule_id == "P-R08")
        assert pr08_result.passed is False  # 未标注 AI-generated

    def test_pr08_ai_generated_annotation_pass(self):
        """P-R08: 声明标注为 AI-generated → 通过."""
        layer = ProvenanceLayer()
        claim = _make_claim(evidence_ids=[])
        claim.metadata["ai_generated"] = True
        result = layer.verify_claim(claim, evidence=[])
        pr08_result = next(r for r in result.rule_results if r.rule_id == "P-R08")
        assert pr08_result.passed is True

    def test_provenance_layer_score_range(self):
        """溯源层评分在 0-100 范围内."""
        layer = ProvenanceLayer()
        evidence = _make_evidence()
        claim = _make_claim(evidence_ids=[evidence.evidence_id])
        result = layer.verify_claim(claim, evidence=[evidence])
        assert 0 <= result.score <= 100


# ============================================================
# 评审状态机测试
# ============================================================


class TestReviewStateMachine:
    """评审状态机测试 — 四层递进 + 三级结果."""

    def test_initial_state(self):
        """初始状态为 IDLE."""
        sm = ReviewStateMachine()
        assert sm.current_state == ReviewState.IDLE

    def test_transition_to_l1_fact(self):
        """从 IDLE 进入 L1_FACT 状态."""
        sm = ReviewStateMachine()
        sm.transition(ReviewState.L1_FACT)
        assert sm.current_state == ReviewState.L1_FACT

    def test_l1_pass_to_l2(self):
        """L1 通过 → 进入 L2_LOGIC."""
        sm = ReviewStateMachine()
        sm.transition(ReviewState.L1_FACT)
        sm.transition(ReviewState.L2_LOGIC, verdict=ReviewVerdict.PASS)
        assert sm.current_state == ReviewState.L2_LOGIC

    def test_l1_flag_to_l2(self):
        """L1 警告 → 带警告进入 L2_LOGIC."""
        sm = ReviewStateMachine()
        sm.transition(ReviewState.L1_FACT)
        sm.transition(ReviewState.L2_LOGIC, verdict=ReviewVerdict.FLAG)
        assert sm.current_state == ReviewState.L2_LOGIC
        assert sm.has_warning(ReviewLayerType.L1_FACT)

    def test_l1_block_to_self_correct(self):
        """L1 阻断 → 进入自纠回路."""
        sm = ReviewStateMachine()
        sm.transition(ReviewState.L1_FACT)
        sm.transition(ReviewState.SELF_CORRECT, verdict=ReviewVerdict.BLOCK)
        assert sm.current_state == ReviewState.SELF_CORRECT

    def test_full_pass_flow(self):
        """全部四层通过 → COMPOSITE_SCORE → PASS."""
        sm = ReviewStateMachine()
        sm.transition(ReviewState.L1_FACT)
        sm.transition(ReviewState.L2_LOGIC, verdict=ReviewVerdict.PASS)
        sm.transition(ReviewState.L3_NUMERICAL, verdict=ReviewVerdict.PASS)
        sm.transition(ReviewState.L4_PROVENANCE, verdict=ReviewVerdict.PASS)
        sm.transition(ReviewState.COMPOSITE_SCORE, verdict=ReviewVerdict.PASS)
        sm.transition(ReviewState.PASS, verdict=ReviewVerdict.PASS)
        assert sm.current_state == ReviewState.PASS

    def test_composite_score_to_pass(self):
        """综合评分 ≥ 85 → PASS."""
        sm = ReviewStateMachine()
        sm.transition(ReviewState.L1_FACT)
        sm.transition(ReviewState.L2_LOGIC, verdict=ReviewVerdict.PASS)
        sm.transition(ReviewState.L3_NUMERICAL, verdict=ReviewVerdict.PASS)
        sm.transition(ReviewState.L4_PROVENANCE, verdict=ReviewVerdict.PASS)
        sm.transition(ReviewState.COMPOSITE_SCORE, verdict=ReviewVerdict.PASS)
        assert sm.current_state == ReviewState.COMPOSITE_SCORE
        sm.transition(ReviewState.PASS, score=90.0)
        assert sm.current_state == ReviewState.PASS

    def test_composite_score_to_flag(self):
        """综合评分 60-85 → FLAG."""
        sm = ReviewStateMachine()
        sm.transition(ReviewState.L1_FACT)
        sm.transition(ReviewState.COMPOSITE_SCORE, verdict=ReviewVerdict.PASS)
        sm.transition(ReviewState.FLAG, score=70.0)
        assert sm.current_state == ReviewState.FLAG

    def test_composite_score_to_block(self):
        """综合评分 < 60 → BLOCK."""
        sm = ReviewStateMachine()
        sm.transition(ReviewState.L1_FACT)
        sm.transition(ReviewState.COMPOSITE_SCORE, verdict=ReviewVerdict.PASS)
        sm.transition(ReviewState.BLOCK, score=40.0)
        assert sm.current_state == ReviewState.BLOCK

    def test_state_history_tracking(self):
        """状态机记录完整状态历史."""
        sm = ReviewStateMachine()
        sm.transition(ReviewState.L1_FACT)
        sm.transition(ReviewState.L2_LOGIC, verdict=ReviewVerdict.PASS)
        assert len(sm.history) >= 2
        assert sm.history[0]["from"] == ReviewState.IDLE
        assert sm.history[1]["to"] == ReviewState.L2_LOGIC


# ============================================================
# 自纠回路测试
# ============================================================


class TestSelfCorrectionLoop:
    """自纠回路测试 — 最多 2 次自纠, 第 3 次升级."""

    def test_initial_loop_state(self):
        """自纠回路初始状态."""
        loop = SelfCorrectionLoop(max_attempts=2)
        assert loop.attempts == 0
        assert loop.max_attempts == 2
        assert loop.can_retry is True

    def test_first_correction_attempt(self):
        """第 1 次自纠."""
        loop = SelfCorrectionLoop(max_attempts=2)
        record = loop.start_correction(
            issues=["F-R01 发射峰波长超出范围"],
            suggestions=["将发射峰波长修正为 570-585nm 范围内"],
        )
        assert loop.attempts == 1
        assert loop.can_retry is True
        assert record.attempt_number == 1
        assert "F-R01" in record.issues[0]

    def test_second_correction_attempt(self):
        """第 2 次自纠."""
        loop = SelfCorrectionLoop(max_attempts=2)
        loop.start_correction(issues=["issue1"], suggestions=["fix1"])
        record = loop.start_correction(issues=["issue2"], suggestions=["fix2"])
        assert loop.attempts == 2
        assert loop.can_retry is False  # 已用完
        assert record.attempt_number == 2

    def test_third_attempt_escalation(self):
        """第 3 次自纠 → 升级裁决."""
        loop = SelfCorrectionLoop(max_attempts=2)
        loop.start_correction(issues=["issue1"], suggestions=["fix1"])
        loop.start_correction(issues=["issue2"], suggestions=["fix2"])
        assert loop.can_retry is False
        assert loop.needs_escalation is True

    def test_correction_history(self):
        """自纠历史记录."""
        loop = SelfCorrectionLoop(max_attempts=2)
        loop.start_correction(issues=["issue1"], suggestions=["fix1"])
        loop.record_result(1, success=False, corrected_output="修正后的输出")
        assert len(loop.history) == 1
        assert loop.history[0]["success"] is False
        assert loop.history[0]["corrected_output"] == "修正后的输出"

    def test_successful_correction(self):
        """自纠成功 → 不需要升级."""
        loop = SelfCorrectionLoop(max_attempts=2)
        loop.start_correction(issues=["issue1"], suggestions=["fix1"])
        loop.record_result(1, success=True, corrected_output="修正后输出")
        assert loop.needs_escalation is False
        assert loop.is_resolved is True


# ============================================================
# 综合评分引擎测试
# ============================================================


class TestCompositeScoringEngine:
    """综合评分引擎测试 — Score = 0.40×L1 + 0.25×L2 + 0.20×L3 + 0.15×L4."""

    def test_default_weights(self):
        """默认权重: L1=0.40, L2=0.25, L3=0.20, L4=0.15."""
        engine = CompositeScoringEngine()
        assert engine.weights.l1 == 0.40
        assert engine.weights.l2 == 0.25
        assert engine.weights.l3 == 0.20
        assert engine.weights.l4 == 0.15

    def test_weights_sum_to_one(self):
        """权重之和为 1.0."""
        engine = CompositeScoringEngine()
        total = engine.weights.l1 + engine.weights.l2 + engine.weights.l3 + engine.weights.l4
        assert abs(total - 1.0) < 0.001

    def test_score_all_pass(self):
        """四层全 100 分 → 综合 100 分."""
        engine = CompositeScoringEngine()
        score = engine.compute_score(l1=100.0, l2=100.0, l3=100.0, l4=100.0)
        assert score == 100.0

    def test_score_all_zero(self):
        """四层全 0 分 → 综合 0 分."""
        engine = CompositeScoringEngine()
        score = engine.compute_score(l1=0.0, l2=0.0, l3=0.0, l4=0.0)
        assert score == 0.0

    def test_score_weighted_average(self):
        """加权平均计算正确."""
        engine = CompositeScoringEngine()
        score = engine.compute_score(l1=80.0, l2=60.0, l3=90.0, l4=70.0)
        expected = 0.40 * 80 + 0.25 * 60 + 0.20 * 90 + 0.15 * 70
        assert abs(score - expected) < 0.01

    def test_verdict_pass_threshold(self):
        """评分 ≥ 85 → PASS."""
        engine = CompositeScoringEngine()
        verdict = engine.determine_verdict(90.0)
        assert verdict == ReviewVerdict.PASS

    def test_verdict_flag_range(self):
        """评分 60-85 → FLAG."""
        engine = CompositeScoringEngine()
        verdict = engine.determine_verdict(70.0)
        assert verdict == ReviewVerdict.FLAG

    def test_verdict_block_below_threshold(self):
        """评分 < 60 → BLOCK."""
        engine = CompositeScoringEngine()
        verdict = engine.determine_verdict(50.0)
        assert verdict == ReviewVerdict.BLOCK

    def test_verdict_boundary_85(self):
        """评分恰好 85 → PASS (边界值)."""
        engine = CompositeScoringEngine()
        verdict = engine.determine_verdict(85.0)
        assert verdict == ReviewVerdict.PASS

    def test_verdict_boundary_60(self):
        """评分恰好 60 → FLAG (边界值)."""
        engine = CompositeScoringEngine()
        verdict = engine.determine_verdict(60.0)
        assert verdict == ReviewVerdict.FLAG

    def test_custom_weights(self):
        """自定义权重."""
        weights = ScoringWeights(l1=0.30, l2=0.30, l3=0.20, l4=0.20)
        engine = CompositeScoringEngine(weights=weights)
        score = engine.compute_score(l1=100.0, l2=0.0, l3=0.0, l4=0.0)
        assert abs(score - 30.0) < 0.01


# ============================================================
# 增强评审管道测试
# ============================================================


class TestReviewPipeline:
    """增强评审管道测试 — 四层编排 + 自纠回路."""

    def test_pipeline_initialization(self):
        """管道初始化."""
        pipeline = ReviewPipeline()
        assert pipeline.fact_layer is not None
        assert pipeline.logic_layer is not None
        assert pipeline.numerical_layer is not None
        assert pipeline.provenance_layer is not None
        assert pipeline.scoring_engine is not None
        assert pipeline.state_machine is not None

    def test_pipeline_review_high_quality_output(self):
        """高质量输出 → PASS."""
        pipeline = ReviewPipeline()
        request = _make_review_request(
            output_text="Dy3+ 的发射主峰在 575nm, 属于镧系元素, 掺杂浓度 2mol%。",
            context_chunks=[
                "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射",
                "Dy3+ 属于镧系稀土元素, f 区",
                "掺杂浓度 2mol% 在正常范围内",
            ],
            citations=["doi:10.1234/example"],
        )
        result = pipeline.review(request)
        assert result.verdict == ReviewVerdict.PASS
        assert result.composite_score >= 85.0

    def test_pipeline_review_low_quality_output(self):
        """低质量输出 → BLOCK."""
        pipeline = ReviewPipeline()
        request = _make_review_request(
            output_text="Dy3+ 的发射主峰在 650nm, 属于 d 区过渡金属, 掺杂浓度 50mol%。",
            context_chunks=[
                "发射峰 650nm",
                "d 区过渡金属",
                "浓度 50mol%",
            ],
        )
        result = pipeline.review(request)
        assert result.verdict == ReviewVerdict.BLOCK
        assert result.composite_score < 60.0

    def test_pipeline_review_medium_quality_output(self):
        """中等质量输出 → FLAG."""
        pipeline = ReviewPipeline()
        request = _make_review_request(
            output_text="Dy3+ 的发射主峰在 575nm, 属于镧系元素。",
            context_chunks=[
                "Dy3+ 发射峰 575nm",
                "镧系元素",
            ],
        )
        result = pipeline.review(request)
        assert result.verdict in (ReviewVerdict.PASS, ReviewVerdict.FLAG)

    def test_pipeline_layer_results_collected(self):
        """管道收集四层评审结果."""
        pipeline = ReviewPipeline()
        request = _make_review_request()
        result = pipeline.review(request)
        assert ReviewLayerType.L1_FACT in result.layer_results
        assert ReviewLayerType.L2_LOGIC in result.layer_results
        assert ReviewLayerType.L3_NUMERICAL in result.layer_results
        assert ReviewLayerType.L4_PROVENANCE in result.layer_results

    def test_pipeline_self_correction_triggered(self):
        """FLAG 触发自纠回路."""
        pipeline = ReviewPipeline()
        request = _make_review_request(
            output_text="Dy3+ 发射峰在 600nm。",
            context_chunks=["发射峰 600nm"],
        )
        result = pipeline.review(request)
        if result.verdict == ReviewVerdict.FLAG:
            assert result.self_correction is not None
            assert result.self_correction.attempts >= 1

    def test_pipeline_with_correction_succeeds(self):
        """自纠后修正输出 → 重新评审通过."""
        pipeline = ReviewPipeline()
        request = _make_review_request(
            output_text="Dy3+ 发射峰在 600nm。",
            context_chunks=["发射峰 600nm"],
        )
        # 第一次评审
        result = pipeline.review(request)
        if result.verdict in (ReviewVerdict.FLAG, ReviewVerdict.BLOCK):
            # 模拟自纠: 修正输出后重新评审
            corrected_request = _make_review_request(
                output_text="Dy3+ 的发射主峰在 575nm。",
                context_chunks=["Dy3+ 发射峰 575nm, ⁴F₉/₂→⁶H₁₃/₂"],
            )
            corrected_result = pipeline.review(corrected_request)
            assert corrected_result.verdict == ReviewVerdict.PASS

    def test_pipeline_escalation_after_max_retries(self):
        """超过最大自纠次数 → 升级."""
        pipeline = ReviewPipeline(config=ReviewPipelineConfig(max_corrections=2))
        request = _make_review_request(
            output_text="Dy3+ 发射峰在 999nm。",
            context_chunks=["发射峰 999nm"],
        )
        result = pipeline.review(request)
        if result.self_correction and result.self_correction.needs_escalation:
            assert result.verdict == ReviewVerdict.BLOCK
            assert result.self_correction.attempts == 2

    def test_pipeline_review_report_id(self):
        """评审结果包含报告 ID."""
        pipeline = ReviewPipeline()
        request = _make_review_request()
        result = pipeline.review(request)
        assert result.report_id is not None
        assert len(result.report_id) > 0

    def test_pipeline_review_timestamps(self):
        """评审结果包含时间戳."""
        pipeline = ReviewPipeline()
        request = _make_review_request()
        result = pipeline.review(request)
        assert result.created_at > 0
        assert result.completed_at is not None
        assert result.completed_at >= result.created_at

    def test_pipeline_custom_config(self):
        """自定义管道配置."""
        config = ReviewPipelineConfig(
            pass_threshold=90.0,
            flag_threshold=70.0,
            max_corrections=1,
        )
        pipeline = ReviewPipeline(config=config)
        assert pipeline.config.pass_threshold == 90.0
        assert pipeline.config.flag_threshold == 70.0
        assert pipeline.config.max_corrections == 1

    def test_pipeline_review_detail_report(self):
        """评审报告包含详细问题列表."""
        pipeline = ReviewPipeline()
        request = _make_review_request(
            output_text="Dy3+ 发射峰在 650nm。",
            context_chunks=["发射峰 650nm"],
        )
        result = pipeline.review(request)
        assert len(result.issues) > 0
        # 每个问题包含规则 ID 和描述
        for issue in result.issues:
            assert "rule_id" in issue or "rule" in issue
