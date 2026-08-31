"""L4 决策引擎层 — 自适应学习编排器 (AdaptiveLearningOrchestrator).

融合世界先进方案的自适应编排:
- OLIVIA (2026): 步骤级反馈学习 + 策略在线调整
- ADWIN: 概念漂移检测 + 策略重置
- A/B Testing: 在线策略对比 + 最优策略选择
- ε-greedy Decay: 冷启动管理 + 平滑过渡
- Platt Scaling: 置信度在线校准
- Bayesian Estimation: Beta-Bernoulli 后验更新

核心职责:
    编排所有自适应学习组件，协调:
    1. 反馈聚合 (FeedbackAggregator)
    2. 漂移检测 (ConceptDriftDetector)
    3. 冷启动管理 (ColdStartManager)
    4. A/B 测试 (ABTestFramework)
    5. 行动选择 (ActionSelector)
    6. 输出合成 (OutputSynthesizer)

    当检测到概念漂移时:
    - 自动重置漂移检测器
    - 触发策略重新评估
    - 可选启动 A/B 测试对比新策略

Usage::

    orchestrator = AdaptiveLearningOrchestrator(
        feedback_aggregator=FeedbackAggregator(),
        action_selector=ActionSelector(strategy="ensemble"),
        output_synthesizer=OutputSynthesizer(),
    )

    # 处理用户反馈
    orchestrator.process_feedback(action_record, rating=0.8)

    # 检查系统状态
    if orchestrator.is_drift_detected():
        logger.warning("检测到概念漂移，考虑重置策略")

    # 启动策略对比实验
    exp_id = orchestrator.start_strategy_experiment(
        name="drift_recovery_test",
        variants=["ucb", "thompson", "ensemble"],
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .ab_test_framework import ABTestFramework
from .cold_start_manager import ColdStartManager, ColdStartPhase
from .concept_drift_detector import ConceptDriftDetector, DriftResult, DriftSeverity
from .feedback_aggregator import FeedbackAggregator
from .models import ActionRecord, ActionType, FeedbackSignal, FeedbackType
from .output_synthesizer import OutputSynthesizer
from .action_selector import ActionSelector

logger = logging.getLogger(__name__)


class AdaptiveLearningOrchestrator:
    """自适应学习编排器 — 协调所有自适应组件.

    整合漂移检测、冷启动管理、A/B 测试和反馈聚合，
    提供统一的自适应学习入口。

    Args:
        feedback_aggregator: 反馈聚合器
        action_selector: 行动选择器
        output_synthesizer: 输出合成器
        drift_window_size: 漂移检测窗口大小
        drift_delta: 漂移检测显著性水平
        cold_start_min_obs: 冷启动最小观测数
        cold_start_decay: 冷启动 ε 衰减率
        auto_reset_on_drift: 漂移时自动重置检测器
        auto_strategy_switch: 漂移时自动切换策略
    """

    def __init__(
        self,
        *,
        feedback_aggregator: FeedbackAggregator,
        action_selector: ActionSelector,
        output_synthesizer: OutputSynthesizer | None = None,
        drift_window_size: int = 100,
        drift_delta: float = 0.05,
        cold_start_min_obs: int = 20,
        cold_start_decay: float = 0.95,
        auto_reset_on_drift: bool = True,
        auto_strategy_switch: bool = False,
    ) -> None:
        """初始化自适应学习编排器.

        Args:
            feedback_aggregator: 反馈聚合器实例
            action_selector: 行动选择器实例
            output_synthesizer: 输出合成器实例 (可选)
            drift_window_size: 漂移检测窗口大小
            drift_delta: 漂移检测显著性水平 δ
            cold_start_min_obs: 冷启动最小观测数
            cold_start_decay: 冷启动 ε 衰减率
            auto_reset_on_drift: 检测到漂移后自动重置
            auto_strategy_switch: 检测到漂移后自动切换策略
        """
        self._feedback_aggregator = feedback_aggregator
        self._action_selector = action_selector
        self._output_synthesizer = output_synthesizer

        # 自适应组件
        self.drift_detector = ConceptDriftDetector(
            window_size=drift_window_size,
            delta=drift_delta,
        )
        self.cold_start_manager = ColdStartManager(
            min_observations=cold_start_min_obs,
            decay_rate=cold_start_decay,
        )
        self.ab_framework = ABTestFramework()

        # 配置
        self._auto_reset_on_drift = auto_reset_on_drift
        self._auto_strategy_switch = auto_strategy_switch

        # 状态追踪
        self._total_feedback_count: int = 0
        self._calibration_sample_count: int = 0
        self._current_strategy: str = "ensemble"
        self._last_drift_result: DriftResult | None = None
        self._strategy_history: list[dict[str, Any]] = []

        logger.info(
            "AdaptiveLearningOrchestrator 初始化 "
            "(漂移窗口=%d, 冷启动最小观测=%d, 自动重置=%s)",
            drift_window_size, cold_start_min_obs, auto_reset_on_drift,
        )

    # --------------------------------------------------------
    # 反馈处理
    # --------------------------------------------------------

    def process_feedback(
        self,
        action_record: ActionRecord,
        rating: float,
        *,
        comment: str = "",
        feedback_type: FeedbackType = FeedbackType.EXPLICIT_RATING,
        intent_type: str = "",
    ) -> FeedbackSignal | None:
        """处理用户反馈 — 统一入口.

        将反馈信号分发到所有自适应组件:
        1. FeedbackAggregator: 存储 + Bayesian 更新
        2. ConceptDriftDetector: 漂移检测
        3. ColdStartManager: 冷启动观测
        4. ActionSelector: 策略更新
        5. OutputSynthesizer: 校准更新

        Args:
            action_record: 行动记录
            rating: 评分 (-1 ~ 1)
            comment: 评论
            feedback_type: 反馈类型
            intent_type: 意图类型

        Returns:
            生成的 FeedbackSignal
        """
        self._total_feedback_count += 1

        # 1. 创建并存储反馈信号
        signal = FeedbackSignal(
            plan_id=action_record.plan_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            intent_type=intent_type,
            action_type=action_record.action_type.value,
            validation_score=action_record.validation_score,
            execution_confidence=action_record.execution_confidence,
        )
        self._feedback_aggregator.add_signal(signal)

        # 2. 漂移检测
        self.drift_detector.add_observation(rating)
        drift_result = self.drift_detector.check_drift()

        if drift_result.drift_detected:
            self._last_drift_result = drift_result
            logger.warning(
                "检测到概念漂移: 类型=%s, 严重度=%s, 旧均值=%.4f, 新均值=%.4f",
                drift_result.drift_type.value,
                drift_result.severity.value,
                drift_result.old_mean,
                drift_result.new_mean,
            )

            if self._auto_reset_on_drift:
                self.drift_detector.reset()

            if self._auto_strategy_switch:
                self._handle_strategy_switch(drift_result)

        # 3. 冷启动观测
        self.cold_start_manager.observe_action(
            action_record.action_type,
            reward=rating,
        )

        # 4. 行动选择器反馈
        self._action_selector.feedback(action_record.action_type, rating)

        # 5. 输出合成器校准更新 (如果有)
        if self._output_synthesizer is not None:
            # 将评分转换为 (confidence, is_correct) 对
            # 正评分 → 正确, 负评分 → 不正确
            is_correct = rating > 0
            self._output_synthesizer.update_calibrator(
                [(action_record.confidence, is_correct)]
            )
            self._calibration_sample_count += 1

        return signal

    def _handle_strategy_switch(self, drift_result: DriftResult) -> None:
        """处理策略切换.

        Args:
            drift_result: 漂移检测结果
        """
        old_strategy = self._current_strategy

        # 根据漂移严重度选择新策略
        if drift_result.severity == DriftSeverity.CRITICAL:
            new_strategy = "ucb"  # 保守策略
        elif drift_result.severity == DriftSeverity.HIGH:
            new_strategy = "thompson"  # 贝叶斯策略
        else:
            new_strategy = "ensemble"  # 集成策略

        if new_strategy != old_strategy:
            self._current_strategy = new_strategy
            self._strategy_history.append({
                "timestamp": time.time(),
                "old_strategy": old_strategy,
                "new_strategy": new_strategy,
                "drift_type": drift_result.drift_type.value,
                "severity": drift_result.severity.value,
                "delta_mean": drift_result.delta_mean,
            })
            logger.info(
                "策略切换: %s → %s (漂移严重度=%s)",
                old_strategy, new_strategy, drift_result.severity.value,
            )

    # --------------------------------------------------------
    # A/B 测试
    # --------------------------------------------------------

    def start_strategy_experiment(
        self,
        *,
        name: str,
        variants: list[str],
        min_samples: int = 30,
        significance_level: float = 0.05,
    ) -> str:
        """启动策略对比实验.

        Args:
            name: 实验名称
            variants: 变体名称列表 (如 ["ucb", "thompson", "ensemble"])
            min_samples: 每变体最小样本数
            significance_level: 显著性水平

        Returns:
            实验 ID
        """
        exp = self.ab_framework.create_experiment(
            name=name,
            variants=variants,
            min_samples=min_samples,
            significance_level=significance_level,
        )
        logger.info("启动策略实验: %s (变体=%s)", name, variants)
        return exp.experiment_id

    def record_experiment_outcome(
        self,
        experiment_id: str,
        variant: str,
        reward: float,
    ) -> None:
        """记录实验结果.

        Args:
            experiment_id: 实验 ID
            variant: 变体名称
            reward: 奖励值
        """
        self.ab_framework.record_outcome(experiment_id, variant, reward)

    def check_experiment(self, experiment_id: str) -> dict[str, Any]:
        """检查实验显著性.

        Args:
            experiment_id: 实验 ID

        Returns:
            检验结果
        """
        return self.ab_framework.check_significance(experiment_id)

    # --------------------------------------------------------
    # 批量校准
    # --------------------------------------------------------

    def update_calibration_batch(
        self,
        feedback_data: list[tuple[float, bool]],
    ) -> None:
        """批量更新校准器.

        Args:
            feedback_data: [(raw_confidence, actual_correct), ...]
        """
        if self._output_synthesizer is not None:
            self._output_synthesizer.update_calibrator(feedback_data)
            self._calibration_sample_count += len(feedback_data)
            logger.info("批量校准更新: %d 条数据", len(feedback_data))

    # --------------------------------------------------------
    # 状态查询
    # --------------------------------------------------------

    def is_drift_detected(self) -> bool:
        """是否检测到漂移."""
        return self._last_drift_result is not None and self._last_drift_result.drift_detected

    def is_in_cold_start(self) -> bool:
        """是否在冷启动期."""
        return self.cold_start_manager.is_in_cold_start()

    def get_drift_stats(self) -> dict[str, Any]:
        """获取漂移检测统计."""
        stats = self.drift_detector.get_stats()
        stats["drift_detected"] = self.is_drift_detected()
        if self._last_drift_result:
            stats["last_drift_type"] = self._last_drift_result.drift_type.value
            stats["last_drift_severity"] = self._last_drift_result.severity.value
            stats["last_drift_confidence"] = self._last_drift_result.confidence
        return stats

    def get_calibration_stats(self) -> dict[str, Any]:
        """获取校准统计."""
        if self._output_synthesizer is None:
            return {"enabled": False}
        # OutputSynthesizer 内部维护校准器
        # 返回基本统计
        return {
            "enabled": True,
            "sample_count": self._calibration_sample_count,
        }

    @property
    def current_strategy(self) -> str:
        """当前策略."""
        return self._current_strategy

    @property
    def total_feedback_count(self) -> int:
        """总反馈数."""
        return self._total_feedback_count

    # --------------------------------------------------------
    # 自适应推荐
    # --------------------------------------------------------

    def get_adaptive_recommendations(self) -> dict[str, Any]:
        """生成自适应推荐.

        基于当前系统状态生成策略调整建议。

        Returns:
            推荐字典
        """
        drift_stats = self.get_drift_stats()
        cold_start_stats = self.cold_start_manager.get_stats()
        bayesian = self._feedback_aggregator.get_bayesian_estimates()
        trend = self._feedback_aggregator.detect_trend()

        recommendations: list[str] = []

        # 漂移相关推荐
        if self.is_drift_detected():
            severity = self._last_drift_result.severity.value if self._last_drift_result else "unknown"
            recommendations.append(
                f"检测到概念漂移 (严重度={severity})，建议重置策略或启动 A/B 测试"
            )

        # 冷启动推荐
        if self.is_in_cold_start():
            recommendations.append(
                f"系统处于冷启动阶段 (ε={cold_start_stats['epsilon']:.4f})，"
                f"建议继续收集反馈数据"
            )

        # 趋势推荐
        if trend:
            if trend["direction"] == "declining":
                recommendations.append(
                    f"反馈趋势下降 (斜率={trend['slope']:.4f})，"
                    f"建议检查策略有效性"
                )
            elif trend["direction"] == "improving":
                recommendations.append(
                    f"反馈趋势改善 (斜率={trend['slope']:.4f})，"
                    f"当前策略表现良好"
                )

        # Bayesian 推荐
        if bayesian:
            best_action = max(
                bayesian.items(),
                key=lambda x: x[1].get("expected_value", 0.5),
            )
            worst_action = min(
                bayesian.items(),
                key=lambda x: x[1].get("expected_value", 0.5),
            )
            recommendations.append(
                f"Bayesian 最优行动: {best_action[0]} "
                f"(EV={best_action[1].get('expected_value', 0.5):.4f}), "
                f"最差: {worst_action[0]} "
                f"(EV={worst_action[1].get('expected_value', 0.5):.4f})"
            )

        return {
            "drift_status": drift_stats,
            "cold_start_phase": cold_start_stats["phase"],
            "cold_start_epsilon": cold_start_stats["epsilon"],
            "strategy_performance": self._action_selector.get_ucb_stats(),
            "bayesian_estimates": bayesian,
            "trend": trend,
            "recommendations": recommendations,
            "current_strategy": self._current_strategy,
            "total_feedback": self._total_feedback_count,
        }

    def get_system_summary(self) -> dict[str, Any]:
        """获取系统摘要.

        Returns:
            系统状态摘要
        """
        drift_stats = self.drift_detector.get_stats()
        cold_start_stats = self.cold_start_manager.get_stats()
        bayesian = self._feedback_aggregator.get_bayesian_estimates()
        summary = self._feedback_aggregator.summarize(last_hours=168.0)

        return {
            "total_feedback": self._total_feedback_count,
            "current_strategy": self._current_strategy,
            "drift_detected": self.is_drift_detected(),
            "drift_status": self.get_drift_stats(),
            "drift_count": drift_stats.get("drift_count", 0),
            "cold_start_active": self.is_in_cold_start(),
            "cold_start_phase": cold_start_stats["phase"],
            "cold_start_epsilon": cold_start_stats["epsilon"],
            "bayesian_estimates": bayesian,
            "feedback_summary": {
                "total_signals": summary.total_signals if summary else 0,
                "avg_rating": summary.avg_rating if summary else 0.0,
            } if summary else None,
            "strategy_history": self._strategy_history[-5:],  # 最近 5 次策略变更
        }


__all__ = [
    "AdaptiveLearningOrchestrator",
]
