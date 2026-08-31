"""OutputSynthesizer 单元测试 (TDD-RED).

测试范围:
- Platt Scaling 置信度校准
- 安全约束层 (SafetyConstraintLayer)
- 多格式输出合成
- 证据组织与排序
- OutputRecord 完整构建
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dy3_polaris.l4.models import (
    ActionRecord,
    ActionType,
    ConfidenceCalibrator,
    EvidenceItem,
    ExecutionResult,
    ExecutionStatus,
    OutputFormat,
    OutputRecord,
    SafetyConstraint,
    SafetyLevel,
    TaskResult,
    TaskType,
    ValidationReport,
    ValidationSeverity,
)


@pytest.fixture
def sample_action_record() -> ActionRecord:
    """构建示例 ActionRecord."""
    record = ActionRecord(
        plan_id="plan-test",
        action_type=ActionType.DIRECT_ANSWER,
        confidence=0.88,
        validation_score=0.92,
        execution_confidence=0.88,
        selection_reason="验证通过，直接输出",
        response_payload={
            "answers": [{"text": "Dy3+ 的激发态为 4F9/2", "confidence": 0.9}],
            "evidence": [
                {"type": "chunk", "content": "Dy3+ 能级跃迁数据", "source": "doc1"},
                {"type": "triple", "content": "Dy3+ -> 4F9/2", "source": "kg"},
            ],
            "reasoning_chain": ["检索", "推理", "合成"],
        },
    )
    return record


@pytest.fixture
def sample_execution_result() -> ExecutionResult:
    """构建示例 ExecutionResult."""
    result = ExecutionResult(plan_id="plan-test", status=ExecutionStatus.COMPLETED)
    result.task_results["reason"] = TaskResult(
        task_id="reason",
        task_type=TaskType.REASON,
        status=ExecutionStatus.COMPLETED,
        output={"answers": [{"text": "Dy3+ 的激发态为 4F9/2"}]},
        confidence=0.9,
        evidence=[{"type": "triple", "content": "Dy3+ -> 4F9/2"}],
        reasoning_chain=["多跳推理完成"],
    )
    result.evidence_set = [
        {"type": "chunk", "content": "能级数据", "source": "doc1", "confidence": 0.85},
        {"type": "triple", "content": "Dy3+ -> 4F9/2", "source": "kg", "confidence": 0.9},
    ]
    result.reasoning_chain = ["检索", "推理", "合成"]
    result.confidence = 0.88
    return result


@pytest.fixture
def sample_validation_report() -> ValidationReport:
    """构建示例 ValidationReport."""
    return ValidationReport(
        plan_id="plan-test",
        overall_status=ValidationSeverity.PASS,
        overall_score=0.92,
    )


# ============================================================
# Platt Scaling 置信度校准测试
# ============================================================


class TestConfidenceCalibrator:
    """Platt Scaling 置信度校准器测试."""

    def test_default_calibration_identity(self) -> None:
        """测试默认参数接近恒等映射."""
        calibrator = ConfidenceCalibrator(scale=1.0, bias=0.0)
        # sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731
        result = calibrator.calibrate(0.0)
        assert abs(result - 0.5) < 0.01

        result = calibrator.calibrate(1.0)
        assert abs(result - 0.731) < 0.01

    def test_calibration_bounds(self) -> None:
        """测试校准结果在边界内."""
        calibrator = ConfidenceCalibrator(
            scale=2.0, bias=-1.0,
            min_confidence=0.05, max_confidence=0.95,
        )
        # 极端值
        low = calibrator.calibrate(0.0)
        high = calibrator.calibrate(1.0)
        assert low >= 0.05
        assert high <= 0.95

    def test_calibration_monotonic(self) -> None:
        """测试校准后保持单调性."""
        calibrator = ConfidenceCalibrator(scale=3.0, bias=-1.5)
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = [calibrator.calibrate(v) for v in values]
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_calibration_extreme_scale(self) -> None:
        """测试大缩放参数使输出趋向极值."""
        # scale=10, bias=-5 使中点在 0.5 处
        calibrator = ConfidenceCalibrator(scale=10.0, bias=-5.0, max_confidence=0.9999)
        # 高置信度 (sigmoid(10*0.8-5) = sigmoid(3) ≈ 0.952)
        high = calibrator.calibrate(0.8)
        assert high > 0.95
        # 低置信度 (sigmoid(10*0.2-5) = sigmoid(-3) ≈ 0.047)
        low = calibrator.calibrate(0.2)
        assert low < 0.05


# ============================================================
# SafetyConstraintLayer 测试
# ============================================================


class TestSafetyConstraintLayer:
    """安全约束层测试."""

    def test_safe_content_passes(self) -> None:
        """测试安全内容通过."""
        from dy3_polaris.l4.output_synthesizer import SafetyConstraintLayer

        layer = SafetyConstraintLayer()
        level, warnings = layer.check(
            content="Dy3+ 的激发态波长是 575 nm",
            confidence=0.9,
            action_type=ActionType.DIRECT_ANSWER,
        )
        assert level == SafetyLevel.SAFE
        assert len(warnings) == 0

    def test_low_confidence_triggers_caution(self) -> None:
        """测试低置信度触发谨慎."""
        from dy3_polaris.l4.output_synthesizer import SafetyConstraintLayer

        layer = SafetyConstraintLayer(confidence_threshold=0.5)
        level, warnings = layer.check(
            content="可能的结果",
            confidence=0.3,
            action_type=ActionType.DIRECT_ANSWER,
        )
        assert level == SafetyLevel.CAUTION
        assert len(warnings) > 0

    def test_custom_constraint_triggers(self) -> None:
        """测试自定义约束触发."""
        from dy3_polaris.l4.output_synthesizer import SafetyConstraintLayer

        constraint = SafetyConstraint(
            name="no_absolutes",
            description="避免绝对化表述",
            pattern=r"(一定|必然|绝对|百分之百)",
            threshold=0.0,
            action=SafetyLevel.RESTRICTED,
            message="检测到绝对化表述，需添加免责声明",
        )
        layer = SafetyConstraintLayer(constraints=[constraint])
        level, warnings = layer.check(
            content="这个结果一定是正确的",
            confidence=0.9,
            action_type=ActionType.DIRECT_ANSWER,
        )
        assert level == SafetyLevel.RESTRICTED
        assert any("绝对" in w for w in warnings)

    def test_human_confirm_always_safe(self) -> None:
        """测试人工确认场景总是安全（不直接输出）."""
        from dy3_polaris.l4.output_synthesizer import SafetyConstraintLayer

        layer = SafetyConstraintLayer()
        level, _ = layer.check(
            content="需要人工确认",
            confidence=0.1,
            action_type=ActionType.HUMAN_CONFIRM,
        )
        # HUMAN_CONFIRM 不直接输出给用户，安全等级为 SAFE
        assert level == SafetyLevel.SAFE

    def test_blocked_level(self) -> None:
        """测试阻断级别."""
        from dy3_polaris.l4.output_synthesizer import SafetyConstraintLayer

        constraint = SafetyConstraint(
            name="block_pattern",
            description="阻断特定内容",
            pattern=r"(禁止|违法)",
            threshold=0.0,
            action=SafetyLevel.BLOCKED,
            message="内容包含敏感词",
        )
        layer = SafetyConstraintLayer(constraints=[constraint])
        level, warnings = layer.check(
            content="这是禁止的内容",
            confidence=0.9,
            action_type=ActionType.DIRECT_ANSWER,
        )
        assert level == SafetyLevel.BLOCKED
        assert len(warnings) > 0


# ============================================================
# OutputSynthesizer 测试
# ============================================================


class TestOutputSynthesizer:
    """输出合成器测试."""

    def test_synthesize_basic(
        self,
        sample_action_record: ActionRecord,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试基础输出合成."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        synthesizer = OutputSynthesizer()
        output = synthesizer.synthesize(
            action_record=sample_action_record,
            execution_result=sample_execution_result,
            validation_report=sample_validation_report,
            intent_type="numeric",
        )

        assert isinstance(output, OutputRecord)
        assert output.plan_id == "plan-test"
        assert output.action_type == ActionType.DIRECT_ANSWER.value
        assert output.content != ""
        assert output.summary != ""
        assert output.raw_confidence > 0
        assert output.calibrated_confidence > 0

    def test_synthesize_format_selection_concise(
        self,
        sample_action_record: ActionRecord,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试简洁格式选择."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        synthesizer = OutputSynthesizer()
        output = synthesizer.synthesize(
            action_record=sample_action_record,
            execution_result=sample_execution_result,
            validation_report=sample_validation_report,
            intent_type="numeric",
        )
        # 数值型高置信度 -> concise
        assert output.output_format == OutputFormat.CONCISE

    def test_synthesize_format_selection_explanatory(
        self,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试解释性格式选择."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        record = ActionRecord(
            plan_id="plan-test",
            action_type=ActionType.DIRECT_ANSWER,
            confidence=0.85,
            response_payload={"answers": [{"text": "能量传递是指..."}]},
        )

        synthesizer = OutputSynthesizer()
        output = synthesizer.synthesize(
            action_record=record,
            execution_result=sample_execution_result,
            validation_report=sample_validation_report,
            intent_type="concept",
        )
        # 概念型 -> explanatory
        assert output.output_format == OutputFormat.EXPLANATORY

    def test_synthesize_format_selection_comparative(
        self,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试比较型格式选择."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        record = ActionRecord(
            plan_id="plan-test",
            action_type=ActionType.DIRECT_ANSWER,
            confidence=0.85,
            response_payload={"answers": [{"text": "Dy3+ vs Eu3+ 对比"}]},
        )

        synthesizer = OutputSynthesizer()
        output = synthesizer.synthesize(
            action_record=record,
            execution_result=sample_execution_result,
            validation_report=sample_validation_report,
            intent_type="relational",
        )
        assert output.output_format == OutputFormat.COMPARATIVE

    def test_synthesize_calibrated_confidence(
        self,
        sample_action_record: ActionRecord,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试置信度校准."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        calibrator = ConfidenceCalibrator(scale=2.0, bias=-0.5)
        synthesizer = OutputSynthesizer(calibrator=calibrator)
        output = synthesizer.synthesize(
            action_record=sample_action_record,
            execution_result=sample_execution_result,
            validation_report=sample_validation_report,
            intent_type="numeric",
        )

        assert output.raw_confidence == sample_action_record.confidence
        # Platt Scaling 后应该不同于原始值
        assert output.calibrated_confidence != output.raw_confidence
        assert 0.0 <= output.calibrated_confidence <= 1.0

    def test_synthesize_evidence_organization(
        self,
        sample_action_record: ActionRecord,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试证据组织."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        synthesizer = OutputSynthesizer()
        output = synthesizer.synthesize(
            action_record=sample_action_record,
            execution_result=sample_execution_result,
            validation_report=sample_validation_report,
            intent_type="numeric",
        )

        assert len(output.evidence_items) > 0
        for item in output.evidence_items:
            assert isinstance(item, EvidenceItem)
            assert item.content != ""
        # 证据按置信度降序排列
        for i in range(len(output.evidence_items) - 1):
            assert output.evidence_items[i].confidence >= output.evidence_items[i + 1].confidence

    def test_synthesize_safety_assessment(
        self,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试安全评估."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        record = ActionRecord(
            plan_id="plan-test",
            action_type=ActionType.NEGOTIATE,
            confidence=0.4,
            validation_score=0.4,
            response_payload={
                "answers": [{"text": "可能的结果"}],
                "warnings": ["证据不足"],
            },
        )

        synthesizer = OutputSynthesizer()
        output = synthesizer.synthesize(
            action_record=record,
            execution_result=sample_execution_result,
            validation_report=sample_validation_report,
            intent_type="numeric",
        )

        # 低置信度 -> CAUTION
        assert output.safety_level == SafetyLevel.CAUTION
        assert len(output.safety_warnings) > 0
        assert output.needs_disclaimer

    def test_synthesize_reasoning_summary(
        self,
        sample_action_record: ActionRecord,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试推理链摘要."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        synthesizer = OutputSynthesizer()
        output = synthesizer.synthesize(
            action_record=sample_action_record,
            execution_result=sample_execution_result,
            validation_report=sample_validation_report,
            intent_type="numeric",
        )

        assert output.reasoning_summary != ""

    def test_synthesize_to_dict(
        self,
        sample_action_record: ActionRecord,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试序列化."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        synthesizer = OutputSynthesizer()
        output = synthesizer.synthesize(
            action_record=sample_action_record,
            execution_result=sample_execution_result,
            validation_report=sample_validation_report,
            intent_type="numeric",
        )

        d = output.to_dict()
        assert "output_id" in d
        assert "calibrated_confidence" in d
        assert "safety_level" in d
        assert "evidence_count" in d
        assert d["evidence_count"] > 0

    def test_synthesize_with_disclaimer(
        self,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试附带免责声明的输出."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        record = ActionRecord(
            plan_id="plan-test",
            action_type=ActionType.DIRECT_ANSWER,
            confidence=0.3,
            response_payload={"answers": [{"text": "不确定的答案"}]},
        )

        synthesizer = OutputSynthesizer()
        output = synthesizer.synthesize(
            action_record=record,
            execution_result=sample_execution_result,
            validation_report=sample_validation_report,
            intent_type="numeric",
        )

        assert output.safety_disclaimer != ""
        assert output.needs_disclaimer

    def test_update_calibrator_from_feedback(
        self,
        sample_action_record: ActionRecord,
        sample_execution_result: ExecutionResult,
        sample_validation_report: ValidationReport,
    ) -> None:
        """测试从反馈更新校准器."""
        from dy3_polaris.l4.output_synthesizer import OutputSynthesizer

        synthesizer = OutputSynthesizer()
        # 模拟反馈数据: (raw_confidence, actual_correct)
        feedback_data = [
            (0.9, True), (0.8, True), (0.7, False),
            (0.6, False), (0.5, False), (0.3, False),
        ]
        synthesizer.update_calibrator(feedback_data)

        assert synthesizer._calibrator.sample_count == 6

        # 校准后低置信度应该更低
        calibrated_low = synthesizer._calibrator.calibrate(0.3)
        calibrated_high = synthesizer._calibrator.calibrate(0.9)
        assert calibrated_low < calibrated_high
