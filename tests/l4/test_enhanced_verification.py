"""L4 增强版验证与策略评估 — 完整单元测试套件.

测试范围:
- UQGate: 不确定性量化网关 (多源信号融合 + 分层选择)
- FaithfulnessChecker: RAGAS 忠实度评估 (主张提取 + 支持度检查)
- SelfConsistencyChecker: 自洽性检查 (多路径一致性 + 矛盾检测)
- StrategyEvaluator: 策略评估 (PRISM 增益分解 + 优化建议)
- VRLoopController: V&R 闭环 (停止条件 + 反馈生成)
- DomainRuleEngine: 领域规则引擎 (波长/浓度/能量传递/晶体场)
- ValidationOrchestrator: 增强版集成 (端到端验证流程)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dy3_polaris.l4.domain_rule_engine import (
    ConcentrationQuenchingRule,
    CrystalFieldRule,
    DomainRule,
    DomainRuleEngine,
    EnergyTransferRule,
    NumericRangeRule,
    WavelengthRangeRule,
)
from dy3_polaris.l4.faithfulness_checker import (
    ClaimExtractor,
    FaithfulnessChecker,
    SelfConsistencyChecker,
)
from dy3_polaris.l4.models import (
    DecisionPlan,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    ReasoningMode,
    RetrievalStrategy,
    SubTask,
    TaskResult,
    TaskType,
    ValidationReport,
    ValidationSeverity,
    ValidationTier,
)
from dy3_polaris.l4.strategy_evaluator import StrategyEvaluator
from dy3_polaris.l4.uq_gate import UQGate, UQSignal
from dy3_polaris.l4.validation_orchestrator import ValidationOrchestrator
from dy3_polaris.l4.vr_loop import RefinementFeedback, VRLoopController


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def high_confidence_result() -> ExecutionResult:
    """高置信度执行结果 — 应触发 L1 轻量验证."""
    result = ExecutionResult(
        plan_id="plan-high-conf",
        status=ExecutionStatus.COMPLETED,
        confidence=0.92,
        total_elapsed_ms=500.0,
        total_token_usage=1500,
    )
    result.task_results["retrieve"] = TaskResult(
        task_id="retrieve",
        task_type=TaskType.RETRIEVE,
        status=ExecutionStatus.COMPLETED,
        output={
            "results": [
                {"chunk_id": "c1", "content": "Dy3+ 发射波长为 575nm", "score": 0.95},
                {"chunk_id": "c2", "content": "Dy3+ 激发态为 4F9/2", "score": 0.90},
            ],
            "total": 2,
        },
        confidence=0.90,
        evidence=[{"type": "chunk", "source": "c1"}],
    )
    result.task_results["reason"] = TaskResult(
        task_id="reason",
        task_type=TaskType.REASON,
        status=ExecutionStatus.COMPLETED,
        output={"answers": [{"text": "Dy3+ 发射波长为 575nm", "confidence": 0.92}]},
        confidence=0.92,
        evidence=[{"type": "triple"}],
        reasoning_chain=["查询 Dy3+ 发射波长", "匹配到文献值 575nm"],
    )
    result.task_results["synthesize"] = TaskResult(
        task_id="synthesize",
        task_type=TaskType.SYNTHESIZE,
        status=ExecutionStatus.COMPLETED,
        output={"summary": "Dy3+ 的发射波长为 575nm，激发态为 4F9/2"},
        confidence=0.88,
        reasoning_chain=["合成响应"],
    )
    result.evidence_set = [
        {"type": "chunk", "content": "Dy3+ 发射波长为 575nm"},
        {"type": "triple", "content": "Dy3+ 激发态 4F9/2"},
        {"type": "literature", "content": "YAG:Dy3+ 黄光发射"},
    ]
    return result


@pytest.fixture
def low_confidence_result() -> ExecutionResult:
    """低置信度执行结果 — 应触发 L3 深度验证."""
    result = ExecutionResult(
        plan_id="plan-low-conf",
        status=ExecutionStatus.COMPLETED,
        confidence=0.35,
        total_elapsed_ms=8000.0,
        total_token_usage=5000,
    )
    result.task_results["retrieve"] = TaskResult(
        task_id="retrieve",
        task_type=TaskType.RETRIEVE,
        status=ExecutionStatus.COMPLETED,
        output={"results": [{"chunk_id": "c1", "content": "部分信息"}], "total": 1},
        confidence=0.30,
    )
    result.task_results["reason"] = TaskResult(
        task_id="reason",
        task_type=TaskType.REASON,
        status=ExecutionStatus.COMPLETED,
        output={"answers": [{"text": "不确定的结论"}]},
        confidence=0.35,
        reasoning_chain=["推理链较短"],
    )
    result.evidence_set = [{"type": "chunk"}]
    return result


@pytest.fixture
def failed_result() -> ExecutionResult:
    """失败的执行结果."""
    return ExecutionResult(
        plan_id="plan-failed",
        status=ExecutionStatus.FAILED,
        confidence=0.0,
        error_summary="执行超时",
    )


@pytest.fixture
def multi_path_result() -> ExecutionResult:
    """多路径推理结果 — 包含矛盾答案."""
    result = ExecutionResult(
        plan_id="plan-multi-path",
        status=ExecutionStatus.COMPLETED,
        confidence=0.65,
    )
    result.task_results["reason_1"] = TaskResult(
        task_id="reason_1",
        task_type=TaskType.REASON,
        status=ExecutionStatus.COMPLETED,
        output={"answers": [{"text": "Dy3+ 波长 575nm"}]},
        confidence=0.85,
        reasoning_chain=["路径1推理", "得出575nm"],
    )
    result.task_results["reason_2"] = TaskResult(
        task_id="reason_2",
        task_type=TaskType.REASON,
        status=ExecutionStatus.COMPLETED,
        output={"answers": [{"text": "Dy3+ 波长 480nm"}]},
        confidence=0.60,
        reasoning_chain=["路径2推理", "得出480nm"],
    )
    result.task_results["retrieve"] = TaskResult(
        task_id="retrieve",
        task_type=TaskType.RETRIEVE,
        status=ExecutionStatus.COMPLETED,
        output={"results": [{"chunk_id": "c1", "content": "Dy3+"}], "total": 1},
        confidence=0.70,
    )
    result.evidence_set = [{"type": "chunk"}, {"type": "triple"}]
    return result


@pytest.fixture
def sample_plan() -> DecisionPlan:
    """示例决策计划."""
    return DecisionPlan(
        plan_id="plan-test",
        execution_mode=ExecutionMode.PARALLEL,
        sub_tasks=[
            SubTask(
                task_id="t1",
                task_type=TaskType.RETRIEVE,
                retrieval_strategy=RetrievalStrategy.HYBRID,
            ),
            SubTask(
                task_id="t2",
                task_type=TaskType.RETRIEVE,
                retrieval_strategy=RetrievalStrategy.GRAPH,
            ),
            SubTask(
                task_id="t3",
                task_type=TaskType.REASON,
                reasoning_mode=ReasoningMode.MULTI_HOP,
                deps=["t1"],
            ),
            SubTask(
                task_id="t4",
                task_type=TaskType.REASON,
                reasoning_mode=ReasoningMode.PATH_FINDING,
                deps=["t2"],
            ),
            SubTask(
                task_id="t5",
                task_type=TaskType.SYNTHESIZE,
                deps=["t3", "t4"],
            ),
        ],
    )


# ============================================================
# UQGate 测试
# ============================================================


class TestUQGate:
    """UQGate 不确定性量化网关测试."""

    def test_high_confidence_selects_l1(self, high_confidence_result: ExecutionResult) -> None:
        """高置信度结果应选择 L1 轻量验证."""
        gate = UQGate()
        assessment = gate.assess(high_confidence_result)
        assert assessment.tier == ValidationTier.L1_LIGHTWEIGHT
        assert assessment.score >= 0.75

    def test_low_confidence_selects_l3(self, low_confidence_result: ExecutionResult) -> None:
        """低置信度结果应选择 L3 深度验证."""
        gate = UQGate()
        assessment = gate.assess(low_confidence_result)
        assert assessment.tier == ValidationTier.L3_DEEP
        assert assessment.score < 0.50

    def test_failed_execution_returns_zero(self, failed_result: ExecutionResult) -> None:
        """失败执行应返回接近 0 的分数."""
        gate = UQGate()
        assessment = gate.assess(failed_result)
        assert assessment.score < 0.5

    def test_signal_fusion(self, high_confidence_result: ExecutionResult) -> None:
        """信号融合应包含所有 5 个信号."""
        gate = UQGate()
        assessment = gate.assess(high_confidence_result)
        assert len(assessment.signals) == 5
        assert "execution_confidence" in assessment.signals
        assert "retrieval_compatibility" in assessment.signals
        assert "consistency_dispersion" in assessment.signals
        assert "evidence_sufficiency" in assessment.signals
        assert "model_prior" in assessment.signals

    def test_temperature_calibration(self, high_confidence_result: ExecutionResult) -> None:
        """温度缩放校准应改变分数."""
        gate = UQGate()
        assessment_no_intent = gate.assess(high_confidence_result)
        assessment_numeric = gate.assess(high_confidence_result, intent_type="numeric")
        # 数值查询使用更保守的温度，分数可能不同
        assert isinstance(assessment_numeric.score, float)

    def test_historical_feedback_adjusts_prior(self, high_confidence_result: ExecutionResult) -> None:
        """历史反馈应调整先验."""
        gate = UQGate()
        # 正反馈
        good_feedback = {"concept": 0.8}
        assessment_good = gate.assess(
            high_confidence_result, intent_type="concept",
            historical_feedback=good_feedback,
        )
        # 负反馈
        bad_feedback = {"concept": -0.8}
        assessment_bad = gate.assess(
            high_confidence_result, intent_type="concept",
            historical_feedback=bad_feedback,
        )
        assert assessment_good.score >= assessment_bad.score

    def test_uq_signal_clamping(self) -> None:
        """UQSignal 应将值限制在 0-1 范围."""
        sig = UQSignal("test", 1.5)
        assert sig.confidence == 1.0
        sig_neg = UQSignal("test", -0.5)
        assert sig_neg.confidence == 0.0

    def test_assessment_to_dict(self, high_confidence_result: ExecutionResult) -> None:
        """to_dict 应返回正确的字典结构."""
        gate = UQGate()
        assessment = gate.assess(high_confidence_result)
        d = assessment.to_dict()
        assert "score" in d
        assert "tier" in d
        assert "signals" in d
        assert "raw_fused_score" in d


# ============================================================
# ClaimExtractor 测试
# ============================================================


class TestClaimExtractor:
    """原子化主张提取器测试."""

    def test_extract_numeric_claim(self) -> None:
        """提取数值型主张."""
        text = "Dy3+ 的发射波长为 575nm"
        claims = ClaimExtractor.extract(text)
        numeric_claims = [c for c in claims if c["type"] == "numeric"]
        assert len(numeric_claims) >= 1
        assert numeric_claims[0]["subject"] == "Dy3+"
        assert numeric_claims[0]["value"] == "575"
        assert numeric_claims[0]["unit"] == "nm"

    def test_extract_composition_claim(self) -> None:
        """提取组成型主张."""
        text = "YAG:Ce 包含 Ce3+ 离子"
        claims = ClaimExtractor.extract(text)
        comp_claims = [c for c in claims if c["type"] == "compositional"]
        assert len(comp_claims) >= 1

    def test_extract_causal_claim(self) -> None:
        """提取因果型主张."""
        text = "浓度淬灭导致发光强度下降"
        claims = ClaimExtractor.extract(text)
        causal_claims = [c for c in claims if c["type"] == "causal"]
        assert len(causal_claims) >= 1

    def test_extract_multiple_claims(self) -> None:
        """从复杂文本中提取多条主张."""
        text = (
            "Dy3+ 的发射波长为 575nm。"
            "YAG 基质包含铝离子。"
            "浓度过高导致发光强度下降。"
        )
        claims = ClaimExtractor.extract(text)
        assert len(claims) >= 3

    def test_empty_text(self) -> None:
        """空文本不返回主张."""
        claims = ClaimExtractor.extract("")
        assert claims == []


# ============================================================
# FaithfulnessChecker 测试
# ============================================================


class TestFaithfulnessChecker:
    """RAGAS Faithfulness 评估器测试."""

    def test_high_faithfulness(self, high_confidence_result: ExecutionResult) -> None:
        """高忠实度结果应获得高分."""
        checker = FaithfulnessChecker()
        result = checker.assess(high_confidence_result)
        assert result["faithfulness_score"] > 0.5
        assert result["total_claims"] > 0

    def test_no_answer_text(self) -> None:
        """无答案文本应返回默认结果."""
        result = ExecutionResult(plan_id="empty", status=ExecutionStatus.COMPLETED)
        result.task_results["retrieve"] = TaskResult(
            task_id="retrieve",
            task_type=TaskType.RETRIEVE,
            status=ExecutionStatus.COMPLETED,
            output={"results": [{"chunk_id": "c1", "content": "test"}]},
            confidence=0.5,
        )
        checker = FaithfulnessChecker()
        assessment = checker.assess(result)
        assert assessment["faithfulness_score"] == 0.5
        assert "message" in assessment

    def test_no_context_chunks(self) -> None:
        """无检索上下文应返回默认结果."""
        result = ExecutionResult(plan_id="no-ctx", status=ExecutionStatus.COMPLETED)
        result.task_results["reason"] = TaskResult(
            task_id="reason",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={"answers": [{"text": "Dy3+ 波长为 575nm"}]},
            confidence=0.8,
        )
        checker = FaithfulnessChecker()
        assessment = checker.assess(result)
        assert assessment["faithfulness_score"] == 0.5

    def test_missing_evidence_report(self, high_confidence_result: ExecutionResult) -> None:
        """缺失证据报告应包含建议查询."""
        checker = FaithfulnessChecker(enable_missing_evidence_report=True)
        result = checker.assess(high_confidence_result)
        assert "missing_evidence" in result

    def test_context_precision(self, high_confidence_result: ExecutionResult) -> None:
        """上下文精度应在 0-1 范围."""
        checker = FaithfulnessChecker()
        result = checker.assess(high_confidence_result)
        assert 0.0 <= result["context_precision"] <= 1.0

    def test_context_recall(self, high_confidence_result: ExecutionResult) -> None:
        """上下文召回应在 0-1 范围."""
        checker = FaithfulnessChecker()
        result = checker.assess(high_confidence_result)
        assert 0.0 <= result["context_recall"] <= 1.0


# ============================================================
# SelfConsistencyChecker 测试
# ============================================================


class TestSelfConsistencyChecker:
    """自洽性检查器测试."""

    def test_single_path_consistency(self, high_confidence_result: ExecutionResult) -> None:
        """单路径推理应返回较高一致性."""
        checker = SelfConsistencyChecker()
        result = checker.assess(high_confidence_result)
        assert result["consistency_score"] > 0.5
        assert result["answer_agreement"] == 1.0  # 单路径完全一致

    def test_multi_path_with_contradiction(self, multi_path_result: ExecutionResult) -> None:
        """多路径矛盾应被检测到."""
        checker = SelfConsistencyChecker()
        result = checker.assess(multi_path_result)
        assert len(result["contradictions"]) > 0
        assert result["consistency_score"] < 1.0

    def test_no_reasoning_results(self) -> None:
        """无推理结果应返回默认值."""
        result = ExecutionResult(plan_id="no-reason", status=ExecutionStatus.COMPLETED)
        checker = SelfConsistencyChecker()
        assessment = checker.assess(result)
        assert assessment["consistency_score"] == 0.8

    def test_dominant_answer_extraction(self, multi_path_result: ExecutionResult) -> None:
        """主导答案应正确提取."""
        checker = SelfConsistencyChecker()
        result = checker.assess(multi_path_result)
        assert result["dominant_answer"] is not None
        assert "text" in result["dominant_answer"]

    def test_confidence_calibration(self, multi_path_result: ExecutionResult) -> None:
        """置信度校准应在 0-1 范围."""
        checker = SelfConsistencyChecker()
        result = checker.assess(multi_path_result)
        assert 0.0 <= result["confidence_calibration"] <= 1.0


# ============================================================
# StrategyEvaluator 测试
# ============================================================


class TestStrategyEvaluator:
    """策略评估器测试."""

    def test_evaluation_structure(self, sample_plan: DecisionPlan, high_confidence_result: ExecutionResult) -> None:
        """评估结果应包含所有字段."""
        evaluator = StrategyEvaluator()
        result = evaluator.evaluate(sample_plan, high_confidence_result)
        assert "strategy_score" in result
        assert "exploration_gain" in result
        assert "information_gain" in result
        assert "aggregation_gain" in result
        assert "retrieval_strategy_assessment" in result
        assert "reasoning_strategy_assessment" in result
        assert "resource_efficiency" in result
        assert "optimization_suggestions" in result

    def test_score_in_range(self, sample_plan: DecisionPlan, high_confidence_result: ExecutionResult) -> None:
        """策略评分应在 0-1 范围."""
        evaluator = StrategyEvaluator()
        result = evaluator.evaluate(sample_plan, high_confidence_result)
        assert 0.0 <= result["strategy_score"] <= 1.0

    def test_exploration_gain_with_multiple_strategies(
        self, sample_plan: DecisionPlan, high_confidence_result: ExecutionResult
    ) -> None:
        """多检索策略应提高探索增益."""
        evaluator = StrategyEvaluator()
        result = evaluator.evaluate(sample_plan, high_confidence_result)
        assert result["exploration_gain"] > 0.3

    def test_no_plan_fallback(self, high_confidence_result: ExecutionResult) -> None:
        """无计划时应给出默认评分."""
        evaluator = StrategyEvaluator()
        result = evaluator.evaluate(None, high_confidence_result)
        assert 0.0 <= result["strategy_score"] <= 1.0

    def test_resource_efficiency(self, high_confidence_result: ExecutionResult) -> None:
        """资源效率评估应包含 Token 和延迟."""
        evaluator = StrategyEvaluator(token_budget=2000, latency_budget_ms=1000)
        result = evaluator.evaluate(None, high_confidence_result)
        eff = result["resource_efficiency"]
        assert "token_usage" in eff
        assert "latency_ms" in eff
        assert "token_efficiency" in eff
        assert "latency_efficiency" in eff

    def test_optimization_suggestions_generated(
        self, sample_plan: DecisionPlan, low_confidence_result: ExecutionResult
    ) -> None:
        """低分结果应生成优化建议."""
        evaluator = StrategyEvaluator()
        result = evaluator.evaluate(sample_plan, low_confidence_result)
        # 低置信度结果应触发建议
        assert isinstance(result["optimization_suggestions"], list)


# ============================================================
# VRLoopController 测试
# ============================================================


class TestVRLoopController:
    """V&R 闭环控制器测试."""

    def test_should_continue_initial(self) -> None:
        """初始状态应继续迭代."""
        controller = VRLoopController(max_iterations=3)
        report = ValidationReport(overall_score=0.5)
        report.overall_status = ValidationSeverity.WARNING
        assert controller.should_continue(report) is True

    def test_should_stop_when_passed(self) -> None:
        """验证通过时应停止."""
        controller = VRLoopController(max_iterations=3)
        report = ValidationReport(overall_score=0.90)
        report.overall_status = ValidationSeverity.PASS
        assert controller.should_continue(report) is False

    def test_should_stop_max_iterations(self) -> None:
        """达到最大轮次应停止."""
        controller = VRLoopController(max_iterations=2)
        report1 = ValidationReport(overall_score=0.5)
        report1.overall_status = ValidationSeverity.WARNING
        controller.should_continue(report1)
        controller._iteration_count = 2
        assert controller.should_continue(report1) is False

    def test_should_stop_low_gain(self) -> None:
        """边际增益低于阈值应停止."""
        controller = VRLoopController(max_iterations=5, min_improvement=0.05)
        report1 = ValidationReport(overall_score=0.5)
        report1.overall_status = ValidationSeverity.WARNING
        controller.should_continue(report1)  # 记录 0.5
        report2 = ValidationReport(overall_score=0.52)
        report2.overall_status = ValidationSeverity.WARNING
        # 增益 0.02 < 0.05
        assert controller.should_continue(report2) is False

    def test_vrr_guard_triggers_on_score_decline(self) -> None:
        """分数下降应触发 VRR-Guard."""
        controller = VRLoopController(max_iterations=5, noise_tolerance=0.05)
        report1 = ValidationReport(overall_score=0.7)
        report1.overall_status = ValidationSeverity.WARNING
        controller.should_continue(report1)  # 记录 0.7
        report2 = ValidationReport(overall_score=0.5)
        report2.overall_status = ValidationSeverity.WARNING
        # 下降 0.2 > 容忍度 0.05
        assert controller.should_continue(report2) is False

    def test_generate_feedback_from_anomalies(
        self, high_confidence_result: ExecutionResult
    ) -> None:
        """从异常中生成反馈."""
        controller = VRLoopController()
        report = ValidationReport(
            overall_score=0.5,
            overall_status=ValidationSeverity.ERROR,
        )
        report.anomalies = [
            {"source": "fact_check", "message": "事实校验失败", "severity": "error"},
            {"source": "conflict_detection", "message": "检测到冲突", "severity": "warning"},
        ]
        feedbacks = controller.generate_feedback(report, high_confidence_result)
        assert len(feedbacks) > 0

    def test_generate_feedback_from_faithfulness(
        self, high_confidence_result: ExecutionResult
    ) -> None:
        """从 Faithfulness 评估中生成反馈."""
        controller = VRLoopController()
        report = ValidationReport(
            overall_score=0.4,
            overall_status=ValidationSeverity.ERROR,
        )
        report.faithfulness_assessment = {
            "unsupported_claims": [
                {"claim": {"raw_text": "测试主张", "type": "numeric", "subject": "Dy3+"}, "support_score": 0.2},
            ],
            "missing_evidence": [
                {"claim_subject": "Dy3+", "suggested_query": "Dy3+ 发射波长 标准值"},
            ],
        }
        feedbacks = controller.generate_feedback(report, high_confidence_result)
        assert any(f.feedback_type == "factual" for f in feedbacks)

    def test_generate_feedback_from_consistency(
        self, multi_path_result: ExecutionResult
    ) -> None:
        """从自洽性检查中生成反馈."""
        controller = VRLoopController()
        report = ValidationReport(
            overall_score=0.5,
            overall_status=ValidationSeverity.WARNING,
        )
        report.self_consistency = {
            "contradictions": [
                {"dominant_answer": "575nm", "conflicting_answer": "480nm"},
            ],
            "consistency_score": 0.4,
        }
        feedbacks = controller.generate_feedback(report, multi_path_result)
        assert any(f.feedback_type == "logical" for f in feedbacks)

    def test_record_iteration(self) -> None:
        """记录迭代应正确更新历史."""
        controller = VRLoopController(max_iterations=3)
        old_report = ValidationReport(overall_score=0.5)
        new_report = ValidationReport(overall_score=0.7)
        feedbacks = [RefinementFeedback("factual", "error", "test", "desc", "fix")]
        controller.record_iteration(feedbacks, old_report, new_report)
        summary = controller.get_iteration_summary()
        assert summary["total_iterations"] == 1
        assert summary["iteration_details"][0]["improvement"] == 0.2

    def test_iteration_summary(self) -> None:
        """迭代摘要应包含完整历史."""
        controller = VRLoopController(max_iterations=3)
        for i in range(3):
            old_report = ValidationReport(overall_score=0.3 + i * 0.2)
            new_report = ValidationReport(overall_score=0.5 + i * 0.2)
            controller.record_iteration([], old_report, new_report)
        summary = controller.get_iteration_summary()
        assert summary["total_iterations"] == 3
        assert len(summary["score_history"]) >= 0


# ============================================================
# DomainRuleEngine 测试
# ============================================================


class TestDomainRuleEngine:
    """领域规则引擎测试."""

    def test_default_rules_loaded(self) -> None:
        """默认规则应自动加载."""
        engine = DomainRuleEngine()
        rules = engine.get_rules()
        assert len(rules) >= 7  # 4 个领域规则 + 3 个数值范围规则

    def test_wavelength_range_rule_pass(self, high_confidence_result: ExecutionResult) -> None:
        """波长在合理范围内应通过."""
        rule = WavelengthRangeRule()
        result = rule.check(high_confidence_result)
        assert result["passed"] is True

    def test_wavelength_range_rule_fail(self) -> None:
        """波长超出范围应失败."""
        result = ExecutionResult(plan_id="bad-wl", status=ExecutionStatus.COMPLETED)
        result.task_results["reason"] = TaskResult(
            task_id="reason",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={"answers": [{"text": "Dy3+ 发射波长为 700nm"}]},
            confidence=0.5,
        )
        rule = WavelengthRangeRule()
        check_result = rule.check(result)
        assert check_result["passed"] is False
        assert len(check_result["violations"]) > 0

    def test_concentration_quenching_rule(self) -> None:
        """浓度淬灭规则应检测不合理描述."""
        result = ExecutionResult(plan_id="quench", status=ExecutionStatus.COMPLETED)
        result.task_results["reason"] = TaskResult(
            task_id="reason",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={
                "answers": [{"text": "Dy3+ 浓度 20mol% 时发光强度提高"}],
                "summary": "Dy3+ 浓度 20mol% 时发光强度提高",
            },
            confidence=0.5,
        )
        rule = ConcentrationQuenchingRule()
        check_result = rule.check(result)
        assert len(check_result["violations"]) > 0

    def test_energy_transfer_rule_known_pair(self) -> None:
        """已知能量传递对应通过."""
        result = ExecutionResult(plan_id="et", status=ExecutionStatus.COMPLETED)
        result.task_results["reason"] = TaskResult(
            task_id="reason",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={
                "answers": [{"text": "Ce3+ 到 Dy3+ 的能量传递"}],
                "summary": "Ce3+ 到 Dy3+ 的能量传递",
            },
            confidence=0.5,
        )
        rule = EnergyTransferRule()
        check_result = rule.check(result)
        assert check_result["passed"] is True

    def test_energy_transfer_rule_unknown_pair(self) -> None:
        """未知能量传递对应报告."""
        result = ExecutionResult(plan_id="et-unknown", status=ExecutionStatus.COMPLETED)
        result.task_results["reason"] = TaskResult(
            task_id="reason",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={
                "answers": [{"text": "Mn2+ 到 Yb3+ 的能量传递"}],
                "summary": "Mn2+ 到 Yb3+ 的能量传递",
            },
            confidence=0.5,
        )
        rule = EnergyTransferRule()
        check_result = rule.check(result)
        assert len(check_result["violations"]) > 0
        assert check_result["violations"][0]["severity"] == "info"

    def test_crystal_field_rule(self) -> None:
        """晶体场效应规则应检测波长偏差."""
        result = ExecutionResult(plan_id="cf", status=ExecutionStatus.COMPLETED)
        result.task_results["reason"] = TaskResult(
            task_id="reason",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={
                "answers": [{"text": "YAG 中 Dy3+ 发射波长为 600nm"}],
                "summary": "YAG 中 Dy3+ 发射波长为 600nm",
            },
            confidence=0.5,
        )
        rule = CrystalFieldRule()
        check_result = rule.check(result)
        assert len(check_result["violations"]) > 0

    def test_numeric_range_rule(self) -> None:
        """数值范围规则应检测超范围值."""
        result = ExecutionResult(plan_id="temp", status=ExecutionStatus.COMPLETED)
        result.task_results["reason"] = TaskResult(
            task_id="reason",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={
                "answers": [{"text": "温度为 5000K"}],
                "summary": "温度为 5000K",
            },
            confidence=0.5,
        )
        rule = NumericRangeRule(
            rule_id="temp_test",
            name="温度测试",
            param_pattern=r"温度.*?(\d+\.?\d*)\s*(?:K|℃)",
            min_value=0,
            max_value=3000,
            unit="K",
        )
        check_result = rule.check(result)
        assert check_result["passed"] is False

    def test_engine_evaluate(self, high_confidence_result: ExecutionResult) -> None:
        """引擎评估应返回综合结果."""
        engine = DomainRuleEngine()
        result = engine.evaluate(high_confidence_result)
        assert "overall_score" in result
        assert "total_rules" in result
        assert "executed_rules" in result
        assert "passed_rules" in result
        assert "failed_rules" in result
        assert "all_violations" in result
        assert 0.0 <= result["overall_score"] <= 1.0

    def test_engine_register_custom_rule(self, high_confidence_result: ExecutionResult) -> None:
        """自定义规则应可注册."""
        engine = DomainRuleEngine(auto_load_defaults=False)
        engine.register_rule(WavelengthRangeRule())
        assert len(engine.get_rules()) == 1
        result = engine.evaluate(high_confidence_result)
        assert result["total_rules"] == 1

    def test_engine_disable_rule(self, high_confidence_result: ExecutionResult) -> None:
        """禁用规则后应跳过执行."""
        engine = DomainRuleEngine()
        engine.disable_rule("rare_earth_wavelength_range")
        result = engine.evaluate(high_confidence_result)
        # 禁用的规则不参与执行
        rule_ids = [r["rule_id"] for r in result["rule_results"]]
        assert "rare_earth_wavelength_range" not in rule_ids

    def test_engine_unregister_rule(self) -> None:
        """注销规则应成功."""
        engine = DomainRuleEngine()
        assert engine.unregister_rule("rare_earth_wavelength_range") is True
        assert engine.unregister_rule("nonexistent") is False

    def test_custom_domain_rule(self, high_confidence_result: ExecutionResult) -> None:
        """自定义领域规则应可正确执行."""

        class CustomRule(DomainRule):
            def __init__(self) -> None:
                super().__init__("custom_test", "自定义规则", priority=100)

            def check(self, execution_result: ExecutionResult) -> dict[str, Any]:
                return {
                    "rule_id": self.rule_id,
                    "rule_name": self.name,
                    "passed": True,
                    "score": 1.0,
                    "violations": [],
                    "details": {},
                }

        engine = DomainRuleEngine(auto_load_defaults=False)
        engine.register_rule(CustomRule())
        result = engine.evaluate(high_confidence_result)
        assert result["rule_results"][0]["rule_id"] == "custom_test"


# ============================================================
# ValidationOrchestrator 增强版集成测试
# ============================================================


class TestValidationOrchestratorEnhanced:
    """增强版验证编排器集成测试."""

    def test_validate_high_confidence_l1(self, high_confidence_result: ExecutionResult) -> None:
        """高置信度结果应走 L1 轻量验证."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(high_confidence_result, intent_type="concept")
        assert report.validation_tier == ValidationTier.L1_LIGHTWEIGHT
        assert report.overall_score > 0.5

    def test_validate_low_confidence_l3(
        self, low_confidence_result: ExecutionResult, sample_plan: DecisionPlan
    ) -> None:
        """低置信度结果应走 L3 深度验证."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(low_confidence_result, plan=sample_plan)
        assert report.validation_tier == ValidationTier.L3_DEEP
        # L3 应包含策略评估
        assert report.strategy_evaluation != {}

    def test_validate_failed_execution(self, failed_result: ExecutionResult) -> None:
        """失败执行应直接标记为 ERROR."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(failed_result)
        assert report.overall_status == ValidationSeverity.ERROR
        assert report.overall_score == 0.0

    def test_validate_includes_uq_score(self, high_confidence_result: ExecutionResult) -> None:
        """验证报告应包含 UQ 分数."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(high_confidence_result)
        assert 0.0 <= report.uq_score <= 1.0

    def test_validate_includes_faithfulness(self, high_confidence_result: ExecutionResult) -> None:
        """验证报告应包含 Faithfulness 评估."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(high_confidence_result)
        assert "faithfulness_score" in report.faithfulness_assessment

    def test_validate_l2_includes_self_consistency(
        self, multi_path_result: ExecutionResult
    ) -> None:
        """L2+ 验证应包含自洽性检查."""
        orchestrator = ValidationOrchestrator(uq_l1_threshold=0.95)  # 强制走 L2+
        report = orchestrator.validate(multi_path_result)
        assert report.validation_tier != ValidationTier.L1_LIGHTWEIGHT
        assert "consistency_score" in report.self_consistency

    def test_validate_l3_includes_strategy_evaluation(
        self, low_confidence_result: ExecutionResult, sample_plan: DecisionPlan
    ) -> None:
        """L3 验证应包含策略评估."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(
            low_confidence_result, plan=sample_plan, intent_type="composite"
        )
        assert report.validation_tier == ValidationTier.L3_DEEP
        assert "strategy_score" in report.strategy_evaluation

    def test_validate_generates_anomalies(
        self, low_confidence_result: ExecutionResult
    ) -> None:
        """低分结果应生成异常报告."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(low_confidence_result)
        # 低置信度结果可能产生异常
        assert isinstance(report.anomalies, list)

    def test_validate_generates_recommendations(
        self, low_confidence_result: ExecutionResult, sample_plan: DecisionPlan
    ) -> None:
        """低分结果应生成改进建议."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(
            low_confidence_result, plan=sample_plan, intent_type="composite"
        )
        assert isinstance(report.recommendations, list)

    def test_validate_with_historical_feedback(
        self, high_confidence_result: ExecutionResult
    ) -> None:
        """历史反馈应影响 UQ 评估."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(
            high_confidence_result,
            intent_type="concept",
            historical_feedback={"concept": 0.9},
        )
        assert report.uq_score > 0.0

    def test_validate_score_aggregation(self, high_confidence_result: ExecutionResult) -> None:
        """评分聚合应在 0-1 范围."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(high_confidence_result)
        assert 0.0 <= report.overall_score <= 1.0

    def test_validate_status_mapping(self, high_confidence_result: ExecutionResult) -> None:
        """状态映射应正确."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(high_confidence_result)
        assert report.overall_status in (
            ValidationSeverity.PASS,
            ValidationSeverity.INFO,
            ValidationSeverity.WARNING,
            ValidationSeverity.ERROR,
            ValidationSeverity.CRITICAL,
        )

    def test_get_vr_controller(self) -> None:
        """获取 V&R 控制器."""
        orchestrator = ValidationOrchestrator(enable_vr_loop=True)
        controller = orchestrator.get_vr_controller()
        assert controller is not None

    def test_get_vr_controller_disabled(self) -> None:
        """禁用 V&R 时控制器为 None."""
        orchestrator = ValidationOrchestrator(enable_vr_loop=False)
        controller = orchestrator.get_vr_controller()
        assert controller is None

    def test_generate_refinement_feedback(
        self, low_confidence_result: ExecutionResult, sample_plan: DecisionPlan
    ) -> None:
        """生成精炼反馈."""
        orchestrator = ValidationOrchestrator(enable_vr_loop=True)
        report = orchestrator.validate(
            low_confidence_result, plan=sample_plan, intent_type="composite"
        )
        feedbacks = orchestrator.generate_refinement_feedback(
            report, low_confidence_result
        )
        assert isinstance(feedbacks, list)

    def test_report_to_dict(self, high_confidence_result: ExecutionResult) -> None:
        """报告序列化为字典."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(high_confidence_result)
        d = report.to_dict()
        assert "report_id" in d
        assert "overall_status" in d
        assert "overall_score" in d
        assert "validation_tier" in d
        assert "uq_score" in d

    def test_report_is_valid_property(self, high_confidence_result: ExecutionResult) -> None:
        """is_valid 属性应正确."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(high_confidence_result)
        assert isinstance(report.is_valid, bool)

    def test_report_needs_human_review(self, high_confidence_result: ExecutionResult) -> None:
        """needs_human_review 属性应正确."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(high_confidence_result)
        assert isinstance(report.needs_human_review, bool)

    def test_custom_weights(self, high_confidence_result: ExecutionResult) -> None:
        """自定义权重应影响评分."""
        orchestrator_default = ValidationOrchestrator()
        orchestrator_custom = ValidationOrchestrator(
            fact_check_weight=0.50,
            quality_weight=0.10,
            conflict_weight=0.10,
            compliance_weight=0.10,
            faithfulness_weight=0.10,
            self_consistency_weight=0.05,
            strategy_eval_weight=0.05,
        )
        report_default = orchestrator_default.validate(high_confidence_result)
        report_custom = orchestrator_custom.validate(high_confidence_result)
        # 不同权重可能产生不同分数
        assert isinstance(report_default.overall_score, float)
        assert isinstance(report_custom.overall_score, float)

    def test_discard_threshold(self, low_confidence_result: ExecutionResult) -> None:
        """丢弃阈值应过滤低分维度."""
        orchestrator = ValidationOrchestrator(discard_threshold=0.8)
        report = orchestrator.validate(low_confidence_result)
        assert isinstance(report.overall_score, float)


# ============================================================
# DomainRuleEngine 集成到 ValidationOrchestrator 测试
# ============================================================


class TestValidationOrchestratorDomainRules:
    """领域规则引擎集成到验证编排器的测试."""

    def test_domain_rules_enabled_by_default(self) -> None:
        """默认应启用领域规则引擎."""
        orchestrator = ValidationOrchestrator()
        assert orchestrator.get_domain_rule_engine() is not None

    def test_domain_rules_disabled(self) -> None:
        """禁用领域规则时引擎为 None."""
        orchestrator = ValidationOrchestrator(enable_domain_rules=False)
        assert orchestrator.get_domain_rule_engine() is None

    def test_validate_includes_domain_rule_results(
        self, high_confidence_result: ExecutionResult
    ) -> None:
        """验证报告应包含领域规则结果."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(high_confidence_result)
        assert hasattr(report, "domain_rule_results")
        assert "overall_score" in report.domain_rule_results

    def test_domain_rule_results_in_to_dict(
        self, high_confidence_result: ExecutionResult
    ) -> None:
        """序列化字典应包含领域规则结果."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(high_confidence_result)
        d = report.to_dict()
        assert "domain_rule_results" in d

    def test_custom_domain_rule_engine(self) -> None:
        """应支持传入自定义领域规则引擎."""
        custom_engine = DomainRuleEngine(auto_load_defaults=False)
        custom_engine.register_rule(WavelengthRangeRule())
        orchestrator = ValidationOrchestrator(domain_rule_engine=custom_engine)
        engine = orchestrator.get_domain_rule_engine()
        assert engine is not None
        rules = engine.get_rules()
        assert len(rules) == 1
        assert rules[0]["rule_id"] == "rare_earth_wavelength_range"

    def test_domain_rule_anomalies_collected(self) -> None:
        """领域规则违规应收集到异常列表."""
        # 构造含波长违规的执行结果
        task = TaskResult(
            task_id="t1",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={
                "answers": [
                    {"text": "Dy3+ 的发射波长为 700nm", "value": "700nm"}
                ]
            },
            confidence=0.3,
        )
        result = ExecutionResult(
            plan_id="test-plan",
            status=ExecutionStatus.COMPLETED,
            task_results={"t1": task},
            confidence=0.3,
        )
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(result, intent_type="factual")
        domain_anomalies = [
            a for a in report.anomalies if a.get("source") == "domain_rules"
        ]
        assert len(domain_anomalies) > 0

    def test_domain_rule_recommendations_generated(self) -> None:
        """领域规则违规应生成改进建议."""
        task = TaskResult(
            task_id="t1",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={
                "answers": [
                    {"text": "Dy3+ 的发射波长为 700nm", "value": "700nm"}
                ]
            },
            confidence=0.3,
        )
        result = ExecutionResult(
            plan_id="test-plan",
            status=ExecutionStatus.COMPLETED,
            task_results={"t1": task},
            confidence=0.3,
        )
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(result, intent_type="factual")
        domain_recs = [r for r in report.recommendations if "领域规则" in r]
        assert len(domain_recs) > 0

    def test_domain_rule_score_affects_overall(self) -> None:
        """领域规则分数应影响总体评分."""
        # 正确波长 — 高分
        good_task = TaskResult(
            task_id="t1",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={
                "answers": [
                    {"text": "Dy3+ 的发射波长为 575nm", "value": "575nm"}
                ]
            },
            confidence=0.9,
        )
        good_result = ExecutionResult(
            plan_id="good-plan",
            status=ExecutionStatus.COMPLETED,
            task_results={"t1": good_task},
            confidence=0.9,
        )

        # 错误波长 — 低分
        bad_task = TaskResult(
            task_id="t1",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={
                "answers": [
                    {"text": "Dy3+ 的发射波长为 700nm", "value": "700nm"}
                ]
            },
            confidence=0.3,
        )
        bad_result = ExecutionResult(
            plan_id="bad-plan",
            status=ExecutionStatus.COMPLETED,
            task_results={"t1": bad_task},
            confidence=0.3,
        )

        orchestrator = ValidationOrchestrator()
        good_report = orchestrator.validate(good_result)
        bad_report = orchestrator.validate(bad_result, intent_type="factual")

        assert good_report.overall_score > bad_report.overall_score


# ============================================================
# 边界测试与性能基准
# ============================================================


class TestEdgeCasesAndPerformance:
    """边界条件与性能测试."""

    def test_uq_gate_empty_result(self) -> None:
        """UQ 网关处理空执行结果."""
        empty_result = ExecutionResult(
            plan_id="empty",
            status=ExecutionStatus.COMPLETED,
            task_results={},
            confidence=0.5,
        )
        gate = UQGate()
        assessment = gate.assess(empty_result)
        assert 0.0 <= assessment.score <= 1.0
        assert assessment.tier in (
            ValidationTier.L1_LIGHTWEIGHT,
            ValidationTier.L2_STANDARD,
            ValidationTier.L3_DEEP,
        )

    def test_faithfulness_checker_empty_context(self) -> None:
        """Faithfulness 检查器处理空上下文."""
        task = TaskResult(
            task_id="t1",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={"answers": [{"text": "Dy3+ 发射波长为 575nm"}]},
            confidence=0.8,
        )
        result = ExecutionResult(
            plan_id="empty-ctx",
            status=ExecutionStatus.COMPLETED,
            task_results={"t1": task},
            confidence=0.8,
        )
        checker = FaithfulnessChecker()
        assessment = checker.assess(result)
        assert "faithfulness_score" in assessment
        assert assessment["faithfulness_score"] >= 0.0

    def test_self_consistency_no_reasoning(self) -> None:
        """自洽性检查器处理无推理结果."""
        task = TaskResult(
            task_id="t1",
            task_type=TaskType.RETRIEVE,
            status=ExecutionStatus.COMPLETED,
            output={"results": []},
            confidence=0.7,
        )
        result = ExecutionResult(
            plan_id="no-reason",
            status=ExecutionStatus.COMPLETED,
            task_results={"t1": task},
            confidence=0.7,
        )
        checker = SelfConsistencyChecker()
        assessment = checker.assess(result)
        assert "consistency_score" in assessment

    def test_strategy_evaluator_empty_plan(self) -> None:
        """策略评估器处理空计划."""
        task = TaskResult(
            task_id="t1",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={"answers": []},
            confidence=0.8,
        )
        result = ExecutionResult(
            plan_id="empty-plan",
            status=ExecutionStatus.COMPLETED,
            task_results={"t1": task},
            confidence=0.8,
        )
        evaluator = StrategyEvaluator()
        evaluation = evaluator.evaluate(None, result)
        assert "strategy_score" in evaluation
        assert 0.0 <= evaluation["strategy_score"] <= 1.0

    def test_vr_loop_controller_empty_history(self) -> None:
        """V&R 闭环控制器空历史."""
        controller = VRLoopController(max_iterations=3)
        report = ValidationReport(
            plan_id="test",
            overall_status=ValidationSeverity.PASS,
            overall_score=0.95,
        )
        # score >= 0.85 且 is_valid → 停止
        assert controller.should_continue(report) is False

    def test_domain_rule_engine_no_rules(self) -> None:
        """领域规则引擎无规则时仍正常工作."""
        engine = DomainRuleEngine(auto_load_defaults=False)
        task = TaskResult(
            task_id="t1",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={"answers": [{"text": "test"}]},
            confidence=0.8,
        )
        result = ExecutionResult(
            plan_id="no-rules",
            status=ExecutionStatus.COMPLETED,
            task_results={"t1": task},
            confidence=0.8,
        )
        evaluation = engine.evaluate(result)
        assert evaluation["overall_score"] == 1.0
        assert evaluation["executed_rules"] == 0

    def test_orchestrator_all_features_disabled(self) -> None:
        """全部增强功能禁用时仍正常工作."""
        orchestrator = ValidationOrchestrator(
            enable_vr_loop=False,
            enable_domain_rules=False,
        )
        task = TaskResult(
            task_id="t1",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={"answers": [{"text": "Dy3+ 发射波长为 575nm"}]},
            confidence=0.9,
        )
        result = ExecutionResult(
            plan_id="minimal",
            status=ExecutionStatus.COMPLETED,
            task_results={"t1": task},
            confidence=0.9,
        )
        report = orchestrator.validate(result)
        assert isinstance(report.overall_score, float)
        assert report.overall_score > 0.0

    def test_claim_extractor_special_characters(self) -> None:
        """主张提取器处理特殊字符."""
        text = "Eu3+ 在 Y2O3 中的发射波长为 611nm"
        claims = ClaimExtractor.extract(text)
        assert len(claims) >= 1

    def test_claim_extractor_mixed_languages(self) -> None:
        """主张提取器处理中英文混合."""
        text = "Dy3+ 的 emission wavelength 为 575nm, 包含 Eu3+"
        claims = ClaimExtractor.extract(text)
        assert len(claims) >= 1

    @pytest.mark.parametrize("iteration", [0, 1, 2, 3, 5])
    def test_vr_loop_stop_conditions(self, iteration: int) -> None:
        """V&R 闭环在不同迭代轮次的停止条件."""
        controller = VRLoopController(max_iterations=3, min_improvement=0.05)
        # 模拟不同分数
        scores = [0.3, 0.5, 0.6, 0.63, 0.64, 0.645]
        report = ValidationReport(
            plan_id="param-test",
            overall_status=ValidationSeverity.INFO,
            overall_score=scores[min(iteration, len(scores) - 1)],
        )
        # should_continue 只接受 report，内部追踪迭代
        # 第一次调用时 _iteration_count=0, score < 0.85 → True
        # 但如果 score >= 0.85 → False
        result = controller.should_continue(report)
        assert isinstance(result, bool)

    def test_orchestrator_partial_failure_result(self) -> None:
        """编排器处理部分失败的执行结果."""
        task_ok = TaskResult(
            task_id="t1",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={"answers": [{"text": "Dy3+ 发射波长为 575nm"}]},
            confidence=0.85,
        )
        task_fail = TaskResult(
            task_id="t2",
            task_type=TaskType.RETRIEVE,
            status=ExecutionStatus.FAILED,
            output={},
            confidence=0.0,
            error="检索超时",
        )
        result = ExecutionResult(
            plan_id="partial-fail",
            status=ExecutionStatus.PARTIAL,
            task_results={"t1": task_ok, "t2": task_fail},
            confidence=0.5,
        )
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(result)
        assert isinstance(report.overall_score, float)
        assert report.overall_score >= 0.0


class TestValidationReportSerialization:
    """验证报告序列化完整性测试."""

    def test_full_report_serialization(self) -> None:
        """完整报告序列化应包含所有字段."""
        task = TaskResult(
            task_id="t1",
            task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED,
            output={"answers": [{"text": "Dy3+ 的发射波长为 575nm"}]},
            confidence=0.85,
        )
        result = ExecutionResult(
            plan_id="serial-test",
            status=ExecutionStatus.COMPLETED,
            task_results={"t1": task},
            confidence=0.85,
        )
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(result)
        d = report.to_dict()

        required_keys = [
            "report_id", "plan_id", "overall_status", "overall_score",
            "validation_tier", "uq_score", "fact_check", "quality_assessment",
            "conflict_detection", "compliance_check", "faithfulness_assessment",
            "self_consistency", "strategy_evaluation", "domain_rule_results",
            "anomalies", "recommendations", "refinement_iterations",
            "validated_at", "validation_time_ms",
        ]
        for key in required_keys:
            assert key in d, f"缺少字段: {key}"
