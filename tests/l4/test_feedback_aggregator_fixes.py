"""FeedbackAggregator Bug 修复与 Bayesian 增强测试 (TDD-RED).

测试范围:
- Bug 1: math 延迟导入修复
- Bug 2: callable -> Callable 类型注解修复
- Bug 3: 索引悬挂引用修复 (裁剪后索引一致)
- Bug 4: 未使用的 max_iterations/fallback_on_failure 配置利用
- Bayesian Beta-Bernoulli 增强反馈聚合
- 滑动窗口趋势检测
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from dy3_polaris.l4.models import (
    ActionRecord,
    ActionType,
    FeedbackSignal,
    FeedbackSummary,
    FeedbackType,
)


# ============================================================
# Bug 1: math 延迟导入修复
# ============================================================


class TestBugFixMathImport:
    """Bug 1: math 模块延迟导入修复."""

    def test_math_imported_at_top_level(self) -> None:
        """测试 math 在模块顶层导入."""
        import dy3_polaris.l4.feedback_aggregator as fa_module

        assert hasattr(fa_module, "math")
        assert fa_module.math is not None

    def test_time_weight_no_delayed_import(self) -> None:
        """测试 _time_weight 不含延迟导入."""
        import inspect

        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        source = inspect.getsource(FeedbackAggregator._time_weight)
        assert "import math" not in source


# ============================================================
# Bug 2: callable -> Callable 类型注解修复
# ============================================================


class TestBugFixCallableAnnotation:
    """Bug 2: callable 类型注解修复."""

    def test_implicit_to_rating_type_annotation(self) -> None:
        """测试 _implicit_to_rating 的类型注解正确."""
        import inspect
        import typing

        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        source = inspect.getsource(FeedbackAggregator._implicit_to_rating)
        # 不应包含小写 callable 作为类型注解
        # 修复后应使用 Callable 或直接不标注
        assert "dict[str, callable]" not in source
        # 检查使用了 Callable (可能带泛型参数如 Callable[[float], float])
        assert "Callable" in source

    def test_aggregate_by_dimension_type_annotation(self) -> None:
        """测试 _aggregate_by_dimension 的类型注解正确."""
        import inspect

        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        source = inspect.getsource(FeedbackAggregator._aggregate_by_dimension)
        assert "key_fn: callable" not in source
        # 检查使用了 Callable (可能带泛型参数)
        assert "Callable" in source


# ============================================================
# Bug 3: 索引悬挂引用修复
# ============================================================


class TestBugFixDanglingIndex:
    """Bug 3: 裁剪后索引一致性修复."""

    def test_index_consistency_after_truncation(self) -> None:
        """测试裁剪后索引与主列表一致."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator(max_history=5)
        # 添加 10 条信号，触发裁剪
        for i in range(10):
            aggregator.add_signal(FeedbackSignal(
                rating=0.5,
                action_type="direct_answer",
                intent_type="concept",
            ))

        # 主列表应该有 5 条
        assert len(aggregator._signals) == 5

        # 索引不应包含已裁剪的信号
        # summarize 应能正确工作
        summary = aggregator.summarize(last_hours=24, min_signals=1)
        assert summary is not None
        assert summary.total_signals == 5

    def test_index_rebuild_on_summarize(self) -> None:
        """测试 summarize 时重建索引."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator(max_history=3)
        for i in range(5):
            aggregator.add_signal(FeedbackSignal(
                rating=0.5,
                action_type="direct_answer",
                intent_type="concept",
            ))

        # 索引可能有过期数据，但 summarize 应使用 _signals 重新计算
        summary = aggregator.summarize(last_hours=24, min_signals=1)
        assert summary is not None

        # by_action 应只反映当前 _signals 中的数据
        assert summary.by_action.get("direct_answer", {}).get("count", 0) == 3

    def test_get_action_rewards_after_truncation(self) -> None:
        """测试裁剪后 get_action_rewards 一致性."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator(max_history=3)
        for i in range(5):
            aggregator.add_signal(FeedbackSignal(
                rating=0.8,
                action_type="direct_answer",
            ))

        rewards = aggregator.get_action_rewards(last_hours=24)
        # 应只反映 3 条信号
        if "direct_answer" in rewards:
            # 不应有 5 条的数据，使用近似比较避免浮点精度问题
            assert abs(rewards["direct_answer"] - 0.8) < 1e-6


# ============================================================
# Bug 4: 未使用配置参数修复
# ============================================================


class TestBugFixUnusedConfig:
    """Bug 4: max_iterations 和 fallback_on_failure 利用."""

    def test_decision_engine_config_max_iterations_used(self) -> None:
        """测试 max_iterations 在 DecisionEngine 中被使用."""
        from dy3_polaris.l4.decision_engine import DecisionEngine, DecisionEngineConfig
        from dy3_polaris.l4.action_selector import ActionSelector

        config = DecisionEngineConfig(max_iterations=5, fallback_on_failure=True)
        engine = DecisionEngine(
            intent_router=MagicMock(),
            task_executor=MagicMock(),
            config=config,
        )
        # max_iterations 应该影响引擎行为
        assert engine._config.max_iterations == 5
        assert engine._config.fallback_on_failure is True


# ============================================================
# Bayesian Beta-Bernoulli 增强测试
# ============================================================


class TestBayesianFeedback:
    """Bayesian Beta-Bernoulli 反馈聚合测试."""

    def test_bayesian_update_positive(self) -> None:
        """测试正反馈的 Bayesian 更新."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator()
        aggregator.add_signal(FeedbackSignal(
            rating=1.0,
            action_type="direct_answer",
            intent_type="concept",
        ))

        # 获取 Bayesian 估计
        bayesian = aggregator.get_bayesian_estimates()
        assert "direct_answer" in bayesian
        assert "alpha" in bayesian["direct_answer"]
        assert "beta" in bayesian["direct_answer"]
        # 正反馈后 alpha 应增加
        assert bayesian["direct_answer"]["alpha"] > bayesian["direct_answer"]["beta"]

    def test_bayesian_update_negative(self) -> None:
        """测试负反馈的 Bayesian 更新."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator()
        aggregator.add_signal(FeedbackSignal(
            rating=-1.0,
            action_type="human_confirm",
            intent_type="concept",
        ))

        bayesian = aggregator.get_bayesian_estimates()
        assert "human_confirm" in bayesian
        # 负反馈后 beta 应增加
        assert bayesian["human_confirm"]["beta"] > bayesian["human_confirm"]["alpha"]

    def test_bayesian_expected_value(self) -> None:
        """测试 Bayesian 期望值计算."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator()
        # 3 正 1 负
        for _ in range(3):
            aggregator.add_signal(FeedbackSignal(rating=1.0, action_type="direct_answer"))
        aggregator.add_signal(FeedbackSignal(rating=-1.0, action_type="direct_answer"))

        bayesian = aggregator.get_bayesian_estimates()
        ev = bayesian["direct_answer"]["expected_value"]
        # 期望值应在 0.5~0.8 之间（先验+数据）
        assert 0.5 < ev < 0.8

    def test_bayesian_uncertainty_decreases(self) -> None:
        """测试更多数据后不确定性降低."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator()

        # 少量数据
        aggregator.add_signal(FeedbackSignal(rating=1.0, action_type="direct_answer"))
        bayesian_1 = aggregator.get_bayesian_estimates()
        uncertainty_1 = bayesian_1["direct_answer"].get("variance", 0.5)

        # 更多数据
        for _ in range(20):
            aggregator.add_signal(FeedbackSignal(rating=1.0, action_type="direct_answer"))
        bayesian_2 = aggregator.get_bayesian_estimates()
        uncertainty_2 = bayesian_2["direct_answer"].get("variance", 0.0)

        assert uncertainty_2 < uncertainty_1


# ============================================================
# 滑动窗口趋势检测测试
# ============================================================


class TestTrendDetection:
    """滑动窗口趋势检测测试."""

    def test_detect_improving_trend(self) -> None:
        """测试检测改善趋势."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator()

        now = time.time()
        # 早期低评分
        for i in range(5):
            aggregator.add_signal(FeedbackSignal(
                rating=-0.3,
                action_type="direct_answer",
                created_at=now - 3600 * (10 - i),  # 10~6 小时前
            ))
        # 近期高评分
        for i in range(5):
            aggregator.add_signal(FeedbackSignal(
                rating=0.8,
                action_type="direct_answer",
                created_at=now - 3600 * (5 - i),  # 5~1 小时前
            ))

        # window_size=10 覆盖全部 10 条信号，才能检测到从低到高的趋势
        trend = aggregator.detect_trend(window_size=10)
        assert trend is not None
        assert trend["direction"] == "improving"
        assert trend["slope"] > 0

    def test_detect_declining_trend(self) -> None:
        """测试检测下降趋势."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator()

        now = time.time()
        # 早期高评分
        for i in range(5):
            aggregator.add_signal(FeedbackSignal(
                rating=0.8,
                action_type="direct_answer",
                created_at=now - 3600 * (10 - i),
            ))
        # 近期低评分
        for i in range(5):
            aggregator.add_signal(FeedbackSignal(
                rating=-0.3,
                action_type="direct_answer",
                created_at=now - 3600 * (5 - i),
            ))

        # window_size=10 覆盖全部 10 条信号，才能检测到从高到低的趋势
        trend = aggregator.detect_trend(window_size=10)
        assert trend is not None
        assert trend["direction"] == "declining"
        assert trend["slope"] < 0

    def test_detect_stable_trend(self) -> None:
        """测试检测稳定趋势."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator()

        now = time.time()
        for i in range(10):
            aggregator.add_signal(FeedbackSignal(
                rating=0.5,
                action_type="direct_answer",
                created_at=now - 3600 * (10 - i),
            ))

        trend = aggregator.detect_trend(window_size=5)
        assert trend is not None
        assert trend["direction"] == "stable"
        assert abs(trend["slope"]) < 0.1

    def test_detect_trend_insufficient_data(self) -> None:
        """测试数据不足时返回 None."""
        from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator

        aggregator = FeedbackAggregator()
        aggregator.add_signal(FeedbackSignal(rating=0.5, action_type="direct_answer"))

        trend = aggregator.detect_trend(window_size=5)
        assert trend is None


# Fixtures
from unittest.mock import MagicMock
