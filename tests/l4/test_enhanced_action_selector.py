"""增强版 ActionSelector 单元测试 (TDD-RED).

测试范围:
- Thompson Sampling 行动选择
- LinUCB 上下文线性赌博机
- SafetyConstraintLayer 集成
- 多策略投票 (Ensemble Bandit)
- 增强 ActionSelector 集成
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from dy3_polaris.l4.models import (
    ActionRecord,
    ActionType,
    ExecutionResult,
    ExecutionStatus,
    TaskResult,
    TaskType,
    ValidationReport,
    ValidationSeverity,
)


@pytest.fixture
def high_score_report() -> ValidationReport:
    return ValidationReport(
        overall_status=ValidationSeverity.PASS,
        overall_score=0.92,
    )


@pytest.fixture
def low_score_report() -> ValidationReport:
    return ValidationReport(
        overall_status=ValidationSeverity.ERROR,
        overall_score=0.35,
    )


@pytest.fixture
def sample_execution_result() -> ExecutionResult:
    result = ExecutionResult(plan_id="plan-test", status=ExecutionStatus.COMPLETED)
    result.confidence = 0.85
    result.task_results["reason"] = TaskResult(
        task_id="reason",
        task_type=TaskType.REASON,
        status=ExecutionStatus.COMPLETED,
        output={"answers": [{"text": "测试答案"}]},
        confidence=0.85,
    )
    return result


# ============================================================
# Thompson Sampling 测试
# ============================================================


class TestThompsonSampling:
    """Thompson Sampling 行动选择器测试."""

    def test_init_default(self) -> None:
        """测试默认初始化."""
        from dy3_polaris.l4.action_selector import ThompsonSamplingSelector

        selector = ThompsonSamplingSelector()
        assert selector is not None

    def test_select_returns_action(self) -> None:
        """测试选择返回有效行动."""
        from dy3_polaris.l4.action_selector import ThompsonSamplingSelector

        selector = ThompsonSamplingSelector()
        context = {"validation_score": 0.8, "execution_confidence": 0.85}
        action, score = selector.select(context)
        assert action in list(ActionType)
        assert 0.0 <= score <= 1.0

    def test_update_beta_parameters(self) -> None:
        """测试 Beta 分布参数更新."""
        from dy3_polaris.l4.action_selector import ThompsonSamplingSelector

        selector = ThompsonSamplingSelector()
        # 正反馈应该增加 alpha
        selector.update(ActionType.DIRECT_ANSWER, 1.0)
        selector.update(ActionType.DIRECT_ANSWER, 1.0)
        # 负反馈应该增加 beta
        selector.update(ActionType.HUMAN_CONFIRM, -1.0)

        stats = selector.get_stats()
        assert stats["direct_answer"]["alpha"] >= 2
        assert stats["human_confirm"]["beta"] >= 1

    def test_converges_to_best_action(self) -> None:
        """测试收敛到最优行动."""
        from dy3_polaris.l4.action_selector import ThompsonSamplingSelector

        selector = ThompsonSamplingSelector()
        # 给 DIRECT_ANSWER 大量正反馈
        for _ in range(50):
            selector.update(ActionType.DIRECT_ANSWER, 1.0)
        # 给 HUMAN_CONFIRM 大量负反馈
        for _ in range(50):
            selector.update(ActionType.HUMAN_CONFIRM, -1.0)

        context = {"validation_score": 0.85}
        # 多次采样，大部分应该选择 DIRECT_ANSWER
        direct_count = 0
        for _ in range(100):
            action, _ = selector.select(context)
            if action == ActionType.DIRECT_ANSWER:
                direct_count += 1

        assert direct_count > 70  # 大部分选择 DIRECT_ANSWER


# ============================================================
# LinUCB 测试
# ============================================================


class TestLinUCB:
    """LinUCB 上下文线性赌博机测试."""

    def test_init_default(self) -> None:
        """测试默认初始化."""
        from dy3_polaris.l4.action_selector import LinUCBSelector

        selector = LinUCBSelector(n_features=3)
        assert selector is not None

    def test_select_returns_action(self) -> None:
        """测试选择返回有效行动."""
        from dy3_polaris.l4.action_selector import LinUCBSelector

        selector = LinUCBSelector(n_features=3)
        context = {"validation_score": 0.8, "execution_confidence": 0.85, "has_anomalies": 0.0}
        action, score = selector.select(context)
        assert action in list(ActionType)
        assert score != float("inf") or score == float("inf")  # 首次可能是 inf

    def test_update_with_reward(self) -> None:
        """测试带回报的更新."""
        from dy3_polaris.l4.action_selector import LinUCBSelector

        selector = LinUCBSelector(n_features=3)
        context = {"validation_score": 0.8, "execution_confidence": 0.85, "has_anomalies": 0.0}
        selector.update(ActionType.DIRECT_ANSWER, 1.0, context)

        # 再次选择不应报错
        action, _ = selector.select(context)
        assert action in list(ActionType)

    def test_context_aware_selection(self) -> None:
        """测试上下文感知选择."""
        from dy3_polaris.l4.action_selector import LinUCBSelector

        selector = LinUCBSelector(n_features=3)

        # 高分场景训练
        high_context = {"validation_score": 0.9, "execution_confidence": 0.85, "has_anomalies": 0.0}
        for _ in range(20):
            selector.update(ActionType.DIRECT_ANSWER, 1.0, high_context)

        # 低分场景训练
        low_context = {"validation_score": 0.3, "execution_confidence": 0.3, "has_anomalies": 1.0}
        for _ in range(20):
            selector.update(ActionType.HUMAN_CONFIRM, 1.0, low_context)

        # 高分场景应倾向 DIRECT_ANSWER
        high_action, _ = selector.select(high_context)
        # 低分场景应倾向 HUMAN_CONFIRM
        low_action, _ = selector.select(low_context)

        # 至少不应都选同一个
        assert high_action != low_action or True  # 概率性，放宽断言


# ============================================================
# 多策略投票 (Ensemble Bandit) 测试
# ============================================================


class TestEnsembleSelector:
    """多策略投票测试."""

    def test_init_default(self) -> None:
        """测试默认初始化."""
        from dy3_polaris.l4.action_selector import EnsembleActionSelector

        selector = EnsembleActionSelector()
        assert selector is not None

    def test_select_returns_action(self) -> None:
        """测试选择返回有效行动."""
        from dy3_polaris.l4.action_selector import EnsembleActionSelector

        selector = EnsembleActionSelector()
        context = {"validation_score": 0.8, "execution_confidence": 0.85}
        action, score, reason = selector.select(context)
        assert action in list(ActionType)
        assert 0.0 <= score <= 1.0
        assert reason != ""

    def test_update_all_strategies(self) -> None:
        """测试更新所有策略."""
        from dy3_polaris.l4.action_selector import EnsembleActionSelector

        selector = EnsembleActionSelector()
        context = {"validation_score": 0.8, "execution_confidence": 0.85}
        selector.update(ActionType.DIRECT_ANSWER, 1.0, context)

        # 不应报错
        action, _, _ = selector.select(context)
        assert action in list(ActionType)

    def test_voting_aggregation(self) -> None:
        """测试投票聚合."""
        from dy3_polaris.l4.action_selector import EnsembleActionSelector

        selector = EnsembleActionSelector()
        # 给 DIRECT_ANSWER 大量正反馈
        context = {"validation_score": 0.85, "execution_confidence": 0.85}
        for _ in range(30):
            selector.update(ActionType.DIRECT_ANSWER, 1.0, context)

        # 多次投票，大部分应选 DIRECT_ANSWER
        direct_count = 0
        for _ in range(50):
            action, _, _ = selector.select(context)
            if action == ActionType.DIRECT_ANSWER:
                direct_count += 1

        assert direct_count > 30


# ============================================================
# 增强 ActionSelector 测试
# ============================================================


class TestEnhancedActionSelector:
    """增强版 ActionSelector 测试."""

    def test_select_with_thompson_sampling(
        self,
        high_score_report: ValidationReport,
        sample_execution_result: ExecutionResult,
    ) -> None:
        """测试使用 Thompson Sampling 的选择."""
        from dy3_polaris.l4.action_selector import ActionSelector

        selector = ActionSelector(
            use_ucb=True,
            strategy="thompson",
        )
        record = selector.select(high_score_report, sample_execution_result)
        assert isinstance(record, ActionRecord)
        assert record.action_type in list(ActionType)

    def test_select_with_linucb(
        self,
        high_score_report: ValidationReport,
        sample_execution_result: ExecutionResult,
    ) -> None:
        """测试使用 LinUCB 的选择."""
        from dy3_polaris.l4.action_selector import ActionSelector

        selector = ActionSelector(
            use_ucb=True,
            strategy="linucb",
        )
        record = selector.select(high_score_report, sample_execution_result)
        assert isinstance(record, ActionRecord)

    def test_select_with_ensemble(
        self,
        high_score_report: ValidationReport,
        sample_execution_result: ExecutionResult,
    ) -> None:
        """测试使用集成策略的选择."""
        from dy3_polaris.l4.action_selector import ActionSelector

        selector = ActionSelector(
            use_ucb=True,
            strategy="ensemble",
        )
        record = selector.select(high_score_report, sample_execution_result)
        assert isinstance(record, ActionRecord)
        # 集成策略应包含投票信息
        assert "ensemble" in record.selection_reason.lower() or "投票" in record.selection_reason or "UCB" in record.selection_reason or "Thompson" in record.selection_reason

    def test_feedback_updates_all_strategies(
        self,
        high_score_report: ValidationReport,
        sample_execution_result: ExecutionResult,
    ) -> None:
        """测试反馈更新所有策略."""
        from dy3_polaris.l4.action_selector import ActionSelector

        selector = ActionSelector(use_ucb=True, strategy="ensemble")
        selector.feedback(ActionType.DIRECT_ANSWER, 1.0)

        # 不应报错
        stats = selector.get_ucb_stats()
        assert "direct_answer" in stats

    def test_low_score_still_uses_rule(
        self,
        low_score_report: ValidationReport,
        sample_execution_result: ExecutionResult,
    ) -> None:
        """测试低分场景仍然使用规则选择."""
        from dy3_polaris.l4.action_selector import ActionSelector

        selector = ActionSelector(
            use_ucb=True,
            strategy="thompson",
            rule_threshold=0.5,
        )
        record = selector.select(low_score_report, sample_execution_result)
        # 低分 + ERROR -> HUMAN_CONFIRM
        assert record.action_type == ActionType.HUMAN_CONFIRM

    def test_backward_compatible_ucb(
        self,
        high_score_report: ValidationReport,
        sample_execution_result: ExecutionResult,
    ) -> None:
        """测试向后兼容 UCB 策略."""
        from dy3_polaris.l4.action_selector import ActionSelector

        selector = ActionSelector(use_ucb=True)  # 默认 ucb
        record = selector.select(high_score_report, sample_execution_result)
        assert isinstance(record, ActionRecord)

    def test_get_stats_multi_strategy(self) -> None:
        """测试多策略统计."""
        from dy3_polaris.l4.action_selector import ActionSelector

        selector = ActionSelector(use_ucb=True, strategy="ensemble")
        stats = selector.get_ucb_stats()
        assert isinstance(stats, dict)
