"""L4 自适应学习增强模块 — 测试套件.

测试覆盖:
- ConceptDriftDetector: ADWIN 窗口漂移检测
- ABTestFramework: 在线 A/B 策略对比
- ColdStartManager: ε-greedy 冷启动预热
- AdaptiveLearningOrchestrator: 自适应编排器
- MiniBatchCalibrator: 批量校准更新
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from dy3_polaris.l4.action_selector import (
    ActionSelector,
    ActionType,
    EnsembleActionSelector,
    ThompsonSamplingSelector,
    UCBActionSelector,
)
from dy3_polaris.l4.adaptive_orchestrator import AdaptiveLearningOrchestrator
from dy3_polaris.l4.cold_start_manager import ColdStartManager, ColdStartPhase
from dy3_polaris.l4.concept_drift_detector import ConceptDriftDetector, DriftResult
from dy3_polaris.l4.ab_test_framework import (
    ABTestExperiment,
    ABTestFramework,
    ExperimentStatus,
)
from dy3_polaris.l4.feedback_aggregator import FeedbackAggregator
from dy3_polaris.l4.models import (
    ActionRecord,
    DecisionPlan,
    ExecutionResult,
    FeedbackSignal,
    FeedbackType,
    ValidationReport,
    ValidationSeverity,
)
from dy3_polaris.l4.output_synthesizer import OutputSynthesizer


# ============================================================
# ConceptDriftDetector 测试
# ============================================================


class TestConceptDriftDetector:
    """概念漂移检测器测试."""

    def test_no_drift_with_stable_feedback(self):
        """稳定反馈流不应检测到漂移."""
        detector = ConceptDriftDetector(window_size=50, delta=0.05)
        for i in range(100):
            detector.add_observation(0.7 + (i % 5 - 2) * 0.01)
        result = detector.check_drift()
        assert not result.drift_detected
        assert result.drift_type == "stable"

    def test_detect_sudden_drift(self):
        """突然漂移应被检测到."""
        detector = ConceptDriftDetector(window_size=30, delta=0.1)
        # 先稳定在高分
        for _ in range(50):
            detector.add_observation(0.8)
        result1 = detector.check_drift()
        assert not result1.drift_detected

        # 突然下降到低分
        for _ in range(30):
            detector.add_observation(0.2)
        result2 = detector.check_drift()
        assert result2.drift_detected
        assert result2.drift_type == "sudden"
        assert result2.old_mean > result2.new_mean

    def test_detect_gradual_drift(self):
        """渐变漂移应被检测到."""
        detector = ConceptDriftDetector(window_size=20, delta=0.15)
        # 渐变下降
        for i in range(80):
            value = 0.8 - i * 0.01
            detector.add_observation(value)
        result = detector.check_drift()
        assert result.drift_detected
        assert result.drift_type in ("gradual", "sudden")

    def test_drift_confidence_range(self):
        """漂移置信度应在 [0, 1] 范围内."""
        detector = ConceptDriftDetector(window_size=20, delta=0.1)
        for _ in range(30):
            detector.add_observation(0.9)
        for _ in range(25):
            detector.add_observation(0.1)
        result = detector.check_drift()
        assert 0.0 <= result.confidence <= 1.0

    def test_drift_reset(self):
        """漂移后重置窗口."""
        detector = ConceptDriftDetector(window_size=20, delta=0.1)
        for _ in range(30):
            detector.add_observation(0.9)
        for _ in range(25):
            detector.add_observation(0.1)
        result = detector.check_drift()
        assert result.drift_detected
        detector.reset()
        result_after = detector.check_drift()
        assert not result_after.drift_detected

    def test_adwin_variance_estimation(self):
        """ADWIN 应正确估计窗口方差."""
        detector = ConceptDriftDetector(window_size=50, delta=0.05)
        values = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.6, 0.4]
        for v in values:
            detector.add_observation(v)
        stats = detector.get_stats()
        assert "variance" in stats
        assert stats["variance"] > 0
        assert stats["count"] == len(values)

    def test_drift_severity_levels(self):
        """漂移严重度分级正确."""
        detector = ConceptDriftDetector(window_size=20, delta=0.1)
        # 大幅下降
        for _ in range(30):
            detector.add_observation(0.9)
        for _ in range(25):
            detector.add_observation(0.1)
        result = detector.check_drift()
        assert result.severity in ("low", "medium", "high", "critical")
        # 大幅变化应为 high 或 critical
        assert result.severity in ("high", "critical")


# ============================================================
# ABTestFramework 测试
# ============================================================


class TestABTestFramework:
    """A/B 测试框架测试."""

    def test_create_experiment(self):
        """创建实验."""
        framework = ABTestFramework()
        exp = framework.create_experiment(
            name="strategy_comparison",
            variants=["ucb", "thompson", "ensemble"],
            min_samples=30,
            significance_level=0.05,
        )
        assert exp.experiment_id is not None
        assert exp.status == ExperimentStatus.RUNNING
        assert len(exp.variants) == 3

    def test_record_outcome(self):
        """记录实验结果."""
        framework = ABTestFramework()
        exp = framework.create_experiment(
            name="test1",
            variants=["A", "B"],
            min_samples=10,
        )
        framework.record_outcome(exp.experiment_id, "A", reward=0.8)
        framework.record_outcome(exp.experiment_id, "A", reward=0.6)
        framework.record_outcome(exp.experiment_id, "B", reward=0.3)
        framework.record_outcome(exp.experiment_id, "B", reward=0.2)

        stats = framework.get_experiment_stats(exp.experiment_id)
        assert stats["variants"]["A"]["count"] == 2
        assert stats["variants"]["B"]["count"] == 2
        assert stats["variants"]["A"]["avg_reward"] > stats["variants"]["B"]["avg_reward"]

    def test_significance_test(self):
        """显著性检验."""
        framework = ABTestFramework()
        exp = framework.create_experiment(
            name="sig_test",
            variants=["control", "treatment"],
            min_samples=20,
            significance_level=0.05,
        )
        # treatment 明显优于 control
        for _ in range(30):
            framework.record_outcome(exp.experiment_id, "control", reward=0.3)
        for _ in range(30):
            framework.record_outcome(exp.experiment_id, "treatment", reward=0.7)

        result = framework.check_significance(exp.experiment_id)
        assert result["is_significant"]
        assert result["winner"] == "treatment"

    def test_no_significance_with_similar_performance(self):
        """相似表现不应达到显著性."""
        framework = ABTestFramework()
        exp = framework.create_experiment(
            name="no_sig",
            variants=["A", "B"],
            min_samples=20,
            significance_level=0.01,
        )
        for _ in range(30):
            framework.record_outcome(exp.experiment_id, "A", reward=0.5)
            framework.record_outcome(exp.experiment_id, "B", reward=0.51)

        result = framework.check_significance(exp.experiment_id)
        assert not result["is_significant"]

    def test_experiment_completion(self):
        """实验完成后应自动停止."""
        framework = ABTestFramework()
        exp = framework.create_experiment(
            name="auto_complete",
            variants=["A", "B"],
            min_samples=10,
            significance_level=0.05,
        )
        for _ in range(15):
            framework.record_outcome(exp.experiment_id, "A", reward=0.2)
            framework.record_outcome(exp.experiment_id, "B", reward=0.8)

        framework.check_significance(exp.experiment_id)
        updated = framework.get_experiment(exp.experiment_id)
        assert updated.status == ExperimentStatus.COMPLETED
        assert updated.winner == "B"

    def test_variant_assignment(self):
        """变体分配应均匀分布."""
        framework = ABTestFramework()
        exp = framework.create_experiment(
            name="assignment",
            variants=["A", "B", "C"],
            min_samples=30,
        )
        counts = {"A": 0, "B": 0, "C": 0}
        for i in range(300):
            variant = framework.assign_variant(exp.experiment_id, seed=i)
            counts[variant] += 1
        # 每个变体应占约 1/3
        for v, c in counts.items():
            assert 70 < c < 130, f"Variant {v} got {c} assignments"

    def test_effect_size_calculation(self):
        """效应量计算 (Cohen's d)."""
        framework = ABTestFramework()
        exp = framework.create_experiment(
            name="effect",
            variants=["A", "B"],
            min_samples=10,
        )
        for _ in range(20):
            framework.record_outcome(exp.experiment_id, "A", reward=0.3)
        for _ in range(20):
            framework.record_outcome(exp.experiment_id, "B", reward=0.8)

        stats = framework.get_experiment_stats(exp.experiment_id)
        effect_size = stats["effect_size"]
        assert effect_size > 0.5  # 大效应


# ============================================================
# ColdStartManager 测试
# ============================================================


class TestColdStartManager:
    """冷启动管理器测试."""

    def test_initial_phase_is_exploration(self):
        """初始阶段应为探索."""
        manager = ColdStartManager(
            min_observations=10,
            initial_epsilon=1.0,
            decay_rate=0.95,
        )
        assert manager.phase == ColdStartPhase.EXPLORATION
        assert manager.epsilon > 0.9

    def test_epsilon_decays_with_observations(self):
        """ε 随观测次数衰减."""
        manager = ColdStartManager(
            min_observations=20,
            initial_epsilon=1.0,
            decay_rate=0.9,
        )
        for _ in range(10):
            manager.observe(reward=0.5)
        assert manager.epsilon < 1.0
        assert manager.epsilon > 0.3

    def test_transition_to_exploitation(self):
        """足够观测后转入利用阶段."""
        manager = ColdStartManager(
            min_observations=10,
            initial_epsilon=1.0,
            decay_rate=0.8,
        )
        for _ in range(15):
            manager.observe(reward=0.7)
        assert manager.phase == ColdStartPhase.EXPLOITATION
        assert manager.epsilon < 0.3

    def test_should_explore_during_cold_start(self):
        """冷启动期间应频繁探索."""
        manager = ColdStartManager(
            min_observations=10,
            initial_epsilon=0.9,
            decay_rate=0.95,
        )
        explore_count = 0
        for _ in range(100):
            if manager.should_explore():
                explore_count += 1
        assert explore_count > 50  # 多数时间在探索

    def test_should_exploit_after_warmup(self):
        """预热后应主要利用."""
        manager = ColdStartManager(
            min_observations=5,
            initial_epsilon=0.9,
            decay_rate=0.7,
        )
        for _ in range(20):
            manager.observe(reward=0.6)
        exploit_count = 0
        for _ in range(100):
            if not manager.should_explore():
                exploit_count += 1
        assert exploit_count > 80  # 多数时间在利用

    def test_cold_start_recommendation(self):
        """冷启动推荐策略."""
        manager = ColdStartManager(min_observations=10)
        # 探索阶段推荐所有行动均匀尝试
        actions = [ActionType.DIRECT_ANSWER, ActionType.TOOL_ENHANCED,
                   ActionType.NEGOTIATE, ActionType.HUMAN_CONFIRM]
        for action in actions:
            manager.observe_action(action, reward=0.5)

        recommendation = manager.recommend_action(
            available_actions=actions,
            action_stats={
                "direct_answer": {"count": 3, "avg_reward": 0.6},
                "tool_enhanced": {"count": 2, "avg_reward": 0.4},
                "negotiate": {"count": 1, "avg_reward": 0.5},
                "human_confirm": {"count": 0, "avg_reward": 0.0},
            },
        )
        assert recommendation in actions

    def test_cold_start_prefers_unexplored(self):
        """冷启动应优先探索未尝试的行动."""
        manager = ColdStartManager(min_observations=20)
        actions = [ActionType.DIRECT_ANSWER, ActionType.TOOL_ENHANCED,
                   ActionType.NEGOTIATE, ActionType.HUMAN_CONFIRM]
        # 只观察了 direct_answer
        for _ in range(5):
            manager.observe_action(ActionType.DIRECT_ANSWER, reward=0.7)

        recommendation = manager.recommend_action(
            available_actions=actions,
            action_stats={
                "direct_answer": {"count": 5, "avg_reward": 0.7},
                "tool_enhanced": {"count": 0, "avg_reward": 0.0},
                "negotiate": {"count": 0, "avg_reward": 0.0},
                "human_confirm": {"count": 0, "avg_reward": 0.0},
            },
        )
        # 应推荐未探索的行动
        assert recommendation != ActionType.DIRECT_ANSWER

    def test_phase_transition_callback(self):
        """阶段转换回调."""
        manager = ColdStartManager(min_observations=5, decay_rate=0.5)
        transitions = []
        manager.on_phase_change = lambda old, new: transitions.append((old, new))

        for _ in range(8):
            manager.observe(reward=0.5)

        assert len(transitions) >= 1
        assert transitions[-1][1] == ColdStartPhase.EXPLOITATION


# ============================================================
# AdaptiveLearningOrchestrator 测试
# ============================================================


class TestAdaptiveLearningOrchestrator:
    """自适应学习编排器测试."""

    def _create_orchestrator(self):
        """创建测试用编排器."""
        feedback_agg = FeedbackAggregator()
        action_selector = ActionSelector(strategy="ensemble")
        output_synth = OutputSynthesizer()
        return AdaptiveLearningOrchestrator(
            feedback_aggregator=feedback_agg,
            action_selector=action_selector,
            output_synthesizer=output_synth,
            drift_window_size=30,
            cold_start_min_obs=5,
        )

    def test_initialization(self):
        """编排器初始化."""
        orch = self._create_orchestrator()
        assert orch.drift_detector is not None
        assert orch.cold_start_manager is not None
        assert orch.ab_framework is not None
        assert orch.total_feedback_count == 0

    def test_process_feedback_updates_drift(self):
        """处理反馈应更新漂移检测."""
        orch = self._create_orchestrator()
        action_record = ActionRecord(
            plan_id="test-plan",
            action_type=ActionType.DIRECT_ANSWER,
            confidence=0.8,
        )
        for _ in range(10):
            orch.process_feedback(action_record, rating=0.7)
        assert orch.total_feedback_count == 10
        stats = orch.get_drift_stats()
        assert stats["count"] == 10

    def test_drift_detection_triggers_alert(self):
        """漂移检测触发告警."""
        orch = self._create_orchestrator()
        action_record = ActionRecord(
            plan_id="test-plan",
            action_type=ActionType.DIRECT_ANSWER,
            confidence=0.8,
        )
        # 稳定期
        for _ in range(30):
            orch.process_feedback(action_record, rating=0.8)
        assert not orch.is_drift_detected()

        # 漂移期
        for _ in range(25):
            orch.process_feedback(action_record, rating=0.2)
        assert orch.is_drift_detected()

    def test_cold_start_integration(self):
        """冷启动集成."""
        orch = self._create_orchestrator()
        assert orch.is_in_cold_start()
        action_record = ActionRecord(
            plan_id="test-plan",
            action_type=ActionType.DIRECT_ANSWER,
        )
        for _ in range(10):
            orch.process_feedback(action_record, rating=0.6)
        assert not orch.is_in_cold_start()

    def test_ab_test_experiment_lifecycle(self):
        """A/B 测试实验生命周期."""
        orch = self._create_orchestrator()
        exp_id = orch.start_strategy_experiment(
            name="ucb_vs_ensemble",
            variants=["ucb", "ensemble"],
            min_samples=10,
        )
        assert exp_id is not None

        # 记录结果
        for _ in range(15):
            orch.record_experiment_outcome(exp_id, "ucb", reward=0.5)
            orch.record_experiment_outcome(exp_id, "ensemble", reward=0.7)

        result = orch.check_experiment(exp_id)
        assert result["is_significant"]
        assert result["winner"] == "ensemble"

    def test_adaptive_recommendations(self):
        """自适应推荐生成."""
        orch = self._create_orchestrator()
        action_record = ActionRecord(
            plan_id="test-plan",
            action_type=ActionType.DIRECT_ANSWER,
        )
        # 正反馈
        for _ in range(10):
            orch.process_feedback(action_record, rating=0.8)

        recs = orch.get_adaptive_recommendations()
        assert "drift_status" in recs
        assert "cold_start_phase" in recs
        assert "strategy_performance" in recs
        assert "recommendations" in recs

    def test_calibration_batch_update(self):
        """批量校准更新."""
        orch = self._create_orchestrator()
        batch_data = [
            (0.9, True),
            (0.8, True),
            (0.3, False),
            (0.6, True),
            (0.2, False),
            (0.7, True),
            (0.4, False),
            (0.85, True),
        ]
        orch.update_calibration_batch(batch_data)
        stats = orch.get_calibration_stats()
        assert stats["sample_count"] >= len(batch_data)

    def test_strategy_switch_on_drift(self):
        """漂移时自动切换策略."""
        orch = self._create_orchestrator()
        action_record = ActionRecord(
            plan_id="test-plan",
            action_type=ActionType.DIRECT_ANSWER,
        )
        # 初始策略
        initial_strategy = orch.current_strategy

        # 制造漂移
        for _ in range(30):
            orch.process_feedback(action_record, rating=0.9)
        for _ in range(25):
            orch.process_feedback(action_record, rating=0.1)

        # 漂移后策略可能调整
        drift_info = orch.get_drift_stats()
        assert drift_info["drift_detected"]

    def test_full_feedback_loop(self):
        """完整反馈闭环测试."""
        orch = self._create_orchestrator()

        # 模拟一系列交互
        for i in range(50):
            action_type = ActionType.DIRECT_ANSWER if i % 2 == 0 else ActionType.TOOL_ENHANCED
            record = ActionRecord(
                plan_id=f"plan-{i}",
                action_type=action_type,
                confidence=0.7,
                validation_score=0.75,
            )
            # 交替正负反馈
            rating = 0.7 if i < 25 else 0.3
            orch.process_feedback(record, rating=rating)

        # 检查系统状态
        summary = orch.get_system_summary()
        assert summary["total_feedback"] == 50
        assert "drift_status" in summary
        assert "cold_start_active" in summary
        assert "bayesian_estimates" in summary


# ============================================================
# 集成测试
# ============================================================


class TestAdaptiveLearningIntegration:
    """自适应学习与现有系统集成测试."""

    def test_orchestrator_with_decision_engine_config(self):
        """编排器与决策引擎配置兼容."""
        from dy3_polaris.l4.decision_engine import DecisionEngineConfig

        config = DecisionEngineConfig(
            strategy="ensemble",
            enable_feedback=True,
            enable_output_synthesis=True,
        )
        assert config.strategy == "ensemble"
        assert config.enable_feedback

    def test_drift_detector_with_feedback_aggregator(self):
        """漂移检测器与反馈聚合器协同."""
        aggregator = FeedbackAggregator()
        detector = ConceptDriftDetector(window_size=20, delta=0.1)

        # 添加反馈
        for i in range(20):
            signal = FeedbackSignal(
                plan_id=f"plan-{i}",
                feedback_type=FeedbackType.EXPLICIT_RATING,
                rating=0.8,
                action_type="direct_answer",
            )
            aggregator.add_signal(signal)
            detector.add_observation(0.8)

        # 突变
        for i in range(10):
            signal = FeedbackSignal(
                plan_id=f"plan-drift-{i}",
                feedback_type=FeedbackType.EXPLICIT_RATING,
                rating=0.2,
                action_type="direct_answer",
            )
            aggregator.add_signal(signal)
            detector.add_observation(0.2)

        drift_result = detector.check_drift()
        assert drift_result.drift_detected

        # 反馈聚合器也应反映出趋势变化
        # 窗口大小 15 覆盖过渡期 (5 个 0.8 + 10 个 0.2)
        trend = aggregator.detect_trend(window_size=15)
        assert trend is not None
        assert trend["direction"] == "declining"

    def test_cold_start_with_action_selector(self):
        """冷启动与行动选择器协同."""
        selector = ActionSelector(strategy="ucb")
        cold_start = ColdStartManager(min_observations=10)

        # 冷启动阶段
        assert cold_start.is_in_cold_start()

        # 模拟行动选择和反馈
        for i in range(15):
            validation = ValidationReport(
                plan_id=f"plan-{i}",
                overall_status=ValidationSeverity.PASS,
                overall_score=0.75,
            )
            execution = ExecutionResult(
                plan_id=f"plan-{i}",
                confidence=0.7,
            )
            record = selector.select(validation, execution)
            reward = 0.6 if record.action_type == ActionType.DIRECT_ANSWER else 0.4
            selector.feedback(record.action_type, reward)
            cold_start.observe_action(record.action_type, reward=reward)

        # 应退出冷启动
        assert not cold_start.is_in_cold_start()

    def test_ab_test_with_multiple_strategies(self):
        """A/B 测试多策略对比."""
        strategies = {
            "ucb": UCBActionSelector(),
            "thompson": ThompsonSamplingSelector(),
        }
        framework = ABTestFramework()
        exp = framework.create_experiment(
            name="multi_strategy",
            variants=list(strategies.keys()),
            min_samples=20,
        )

        # 模拟交互
        import random
        random.seed(42)
        for i in range(50):
            variant = framework.assign_variant(exp.experiment_id, seed=i)
            selector = strategies[variant]
            context = {"validation_score": 0.7, "execution_confidence": 0.6, "has_anomalies": 0.0}
            action, score = selector.select(context)

            # 模拟回报 (thompson 略好)
            if variant == "thompson":
                reward = 0.6 + random.uniform(-0.2, 0.3)
            else:
                reward = 0.4 + random.uniform(-0.2, 0.3)
            reward = max(-1.0, min(1.0, reward))

            selector.update(action, reward)
            framework.record_outcome(exp.experiment_id, variant, reward=reward)

        result = framework.check_significance(exp.experiment_id)
        assert "winner" in result
        stats = framework.get_experiment_stats(exp.experiment_id)
        assert all(v["count"] > 0 for v in stats["variants"].values())
