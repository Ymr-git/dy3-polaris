"""L4 决策引擎层 — 反馈聚合器 (FeedbackAggregator).

融合世界先进方案的反馈学习设计:
- OLIVIA (2026): 步骤级反馈学习
  - 每步执行后收集反馈信号
  - 更新行动选择策略 (UCB 回报更新)
  - 长期趋势追踪与策略调整
- Contextual Bandit: 上下文感知反馈
  - 根据意图类型、行动类型、验证分数聚合反馈
  - 识别哪些上下文下哪些行动表现更好
- Bayesian Beta-Bernoulli: 贝叶斯反馈聚合
  - 为每个行动维护 Beta 后验分布
  - 提供期望值和不确定性度量
  - 支持先验知识和增量更新
- Sliding Window Trend: 滑动窗口趋势检测
  - 检测反馈评分的改善/下降趋势
  - 基于线性回归斜率判断方向
- Ebbinghaus 遗忘曲线: 旧反馈权重衰减
  - 近期反馈权重更高
  - 避免历史数据主导当前决策
- Self-RAG: [Retrieve] Token 自主判断
  - 基于反馈调整检索策略

核心职责:
    聚合 T5(ActionRecord) 产生的用户反馈，生成 FeedbackSummary，
    驱动 ActionSelector 的自适应学习和 DecisionPlanner 的策略优化。

反馈来源:
    1. 显式评分 — 用户 thumbs up/down
    2. 隐式信号 — 停留时间、点击、复制、分享
    3. 结果反馈 — 后续对话中用户是否纠正
    4. 跳过信号 — 用户未采纳直接忽略
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Any, Callable

from .models import (
    ActionRecord,
    ActionType,
    ExecutionResult,
    FeedbackSignal,
    FeedbackSummary,
    FeedbackType,
    ValidationReport,
)

logger = logging.getLogger(__name__)


# ============================================================
# 反馈聚合器
# ============================================================


class FeedbackAggregator:
    """反馈聚合器 — T6 核心模块.

    借鉴 OLIVIA 步骤级反馈学习和 Contextual Bandit:
    - 接收单条 FeedbackSignal，存储并更新统计
    - 定期聚合生成 FeedbackSummary
    - 提供策略调整建议给 ActionSelector 和 DecisionPlanner
    - Bayesian Beta-Bernoulli 后验估计
    - 滑动窗口趋势检测

    Usage::

        aggregator = FeedbackAggregator()
        aggregator.add_signal(signal)
        summary = aggregator.summarize(last_hours=24)
        # summary 用于调整行动选择策略

        # Bayesian 估计
        bayesian = aggregator.get_bayesian_estimates()
        # bayesian["direct_answer"]["expected_value"] -> 0.75

        # 趋势检测
        trend = aggregator.detect_trend(window_size=10)
        # trend["direction"] -> "improving"
    """

    def __init__(
        self,
        *,
        max_history: int = 10000,
        decay_half_life_hours: float = 168.0,  # 7 天半衰期
        bayesian_prior_alpha: float = 1.0,
        bayesian_prior_beta: float = 1.0,
    ) -> None:
        """初始化反馈聚合器.

        Args:
            max_history: 最大保留历史信号数
            decay_half_life_hours: 反馈权重半衰期 (小时)
            bayesian_prior_alpha: Beta 先验 alpha
            bayesian_prior_beta: Beta 先验 beta
        """
        self._signals: list[FeedbackSignal] = []
        self._max_history = max_history
        self._decay_half_life = decay_half_life_hours

        # Bayesian 先验参数
        self._prior_alpha = bayesian_prior_alpha
        self._prior_beta = bayesian_prior_beta

        # 累积统计 (加速查询)
        self._by_intent: dict[str, list[FeedbackSignal]] = defaultdict(list)
        self._by_action: dict[str, list[FeedbackSignal]] = defaultdict(list)

        # 索引脏标志 — 标记是否需要重建索引
        self._index_dirty: bool = False

        logger.info(
            "FeedbackAggregator 初始化完成 (最大历史=%d, 半衰期=%.1f小时)",
            max_history, decay_half_life_hours,
        )

    def add_signal(self, signal: FeedbackSignal) -> None:
        """添加反馈信号.

        Args:
            signal: 反馈信号
        """
        self._signals.append(signal)
        self._by_intent[signal.intent_type].append(signal)
        self._by_action[signal.action_type].append(signal)

        # 超限裁剪 (保留最新的)
        if len(self._signals) > self._max_history:
            removed = self._signals.pop(0)
            # 标记索引为脏，下次 summarize 时重建
            self._index_dirty = True
            logger.debug("反馈历史超限，移除最旧信号 %s", removed.signal_id)

        logger.debug(
            "添加反馈信号: type=%s, rating=%.2f, intent=%s, action=%s",
            signal.feedback_type.value, signal.rating,
            signal.intent_type, signal.action_type,
        )

    def add_explicit_feedback(
        self,
        action_record: ActionRecord,
        rating: float,
        comment: str = "",
    ) -> FeedbackSignal:
        """添加显式评分反馈.

        Args:
            action_record: 行动记录
            rating: 用户评分 (-1 ~ 1)
            comment: 用户评论

        Returns:
            生成的 FeedbackSignal
        """
        signal = FeedbackSignal(
            plan_id=action_record.plan_id,
            feedback_type=FeedbackType.EXPLICIT_RATING,
            rating=rating,
            comment=comment,
            intent_type="",  # 可从 action_record 提取
            action_type=action_record.action_type.value,
            validation_score=action_record.validation_score,
            execution_confidence=action_record.execution_confidence,
        )
        self.add_signal(signal)
        return signal

    def add_implicit_signal(
        self,
        action_record: ActionRecord,
        signal_type: str,
        value: float,
    ) -> FeedbackSignal:
        """添加隐式行为信号.

        Args:
            action_record: 行动记录
            signal_type: 信号类型 (dwell_time, click, copy, share, ...)
            value: 信号值

        Returns:
            生成的 FeedbackSignal
        """
        # 隐式信号转评分
        rating = self._implicit_to_rating(signal_type, value)

        signal = FeedbackSignal(
            plan_id=action_record.plan_id,
            feedback_type=FeedbackType.IMPLICIT_SIGNAL,
            rating=rating,
            comment=f"implicit:{signal_type}={value}",
            action_type=action_record.action_type.value,
            validation_score=action_record.validation_score,
        )
        self.add_signal(signal)
        return signal

    def summarize(
        self,
        *,
        last_hours: float = 24.0,
        min_signals: int = 5,
    ) -> FeedbackSummary | None:
        """聚合反馈生成摘要.

        Args:
            last_hours: 聚合最近 N 小时的反馈
            min_signals: 最少信号数（低于此值返回 None）

        Returns:
            FeedbackSummary 或 None
        """
        now = time.time()
        cutoff = now - last_hours * 3600

        # 如果索引脏了，重建索引以保证一致性
        if self._index_dirty:
            self._rebuild_indexes()
            self._index_dirty = False

        # 筛选时间窗口内的信号
        recent_signals = [s for s in self._signals if s.created_at >= cutoff]

        if len(recent_signals) < min_signals:
            logger.info(
                "反馈信号不足: 最近 %.0f 小时仅 %d 条 (最少需 %d)",
                last_hours, len(recent_signals), min_signals,
            )
            return None

        # 计算时间衰减权重
        weights = [self._time_weight(s.created_at, now) for s in recent_signals]
        total_weight = sum(weights)

        # 总体统计
        weighted_rating = sum(s.rating * w for s, w in zip(recent_signals, weights)) / total_weight

        summary = FeedbackSummary(
            period_start=cutoff,
            period_end=now,
            total_signals=len(recent_signals),
            avg_rating=round(weighted_rating, 4),
        )

        # 按意图统计 — 使用 recent_signals 而非索引
        summary.by_intent = self._aggregate_by_dimension(
            recent_signals, weights, lambda s: s.intent_type or "unknown"
        )

        # 按行动统计 — 使用 recent_signals 而非索引
        summary.by_action = self._aggregate_by_dimension(
            recent_signals, weights, lambda s: s.action_type or "unknown"
        )

        # 生成策略调整建议
        summary.adjustments = self._generate_adjustments(summary)

        logger.info(
            "反馈汇总生成: 信号=%d, 平均评分=%.4f, 建议=%d",
            summary.total_signals, summary.avg_rating, len(summary.adjustments),
        )

        return summary

    def get_action_rewards(self, last_hours: float = 168.0) -> dict[str, float]:
        """获取各行动类型的平均回报（用于 UCB 初始化）.

        Args:
            last_hours: 时间窗口

        Returns:
            action_type -> avg_reward
        """
        now = time.time()
        cutoff = now - last_hours * 3600

        # 直接遍历 _signals，不依赖索引
        action_ratings: dict[str, list[tuple[float, float]]] = defaultdict(list)

        for s in self._signals:
            if s.created_at < cutoff:
                continue
            weight = self._time_weight(s.created_at, now)
            action_ratings[s.action_type or "unknown"].append((s.rating, weight))

        rewards: dict[str, float] = {}
        for action, values in action_ratings.items():
            total_weight = sum(w for _, w in values)
            if total_weight > 0:
                rewards[action] = sum(r * w for r, w in values) / total_weight

        return rewards

    # --------------------------------------------------------
    # Bayesian Beta-Bernoulli 增强
    # --------------------------------------------------------

    def get_bayesian_estimates(
        self,
        *,
        last_hours: float = 168.0,
    ) -> dict[str, dict[str, float]]:
        """获取各行动的 Bayesian Beta-Bernoulli 后验估计.

        为每个行动类型维护 Beta(alpha, beta) 后验:
        - 正反馈 (rating > 0): alpha += |rating|
        - 负反馈 (rating < 0): beta += |rating|

        Args:
            last_hours: 时间窗口

        Returns:
            {action_type: {alpha, beta, expected_value, variance, ...}}
        """
        now = time.time()
        cutoff = now - last_hours * 3600

        # 按行动聚合正/负反馈
        action_alpha: dict[str, float] = defaultdict(lambda: self._prior_alpha)
        action_beta: dict[str, float] = defaultdict(lambda: self._prior_beta)
        action_count: dict[str, int] = defaultdict(int)

        for s in self._signals:
            if s.created_at < cutoff:
                continue
            action = s.action_type or "unknown"
            action_count[action] += 1
            if s.rating > 0:
                action_alpha[action] += s.rating
            elif s.rating < 0:
                action_beta[action] += abs(s.rating)

        # 计算后验统计
        estimates: dict[str, dict[str, float]] = {}
        for action in action_count:
            alpha = action_alpha[action]
            beta = action_beta[action]
            total = alpha + beta

            # Beta 分布的期望值
            ev = alpha / total if total > 0 else 0.5

            # Beta 分布的方差
            # Var = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
            variance = (alpha * beta) / ((total ** 2) * (total + 1)) if total > 0 else 0.25

            # 95% 置信区间 (正态近似)
            std = math.sqrt(variance)
            ci_lower = max(0.0, ev - 1.96 * std)
            ci_upper = min(1.0, ev + 1.96 * std)

            estimates[action] = {
                "alpha": round(alpha, 4),
                "beta": round(beta, 4),
                "expected_value": round(ev, 4),
                "variance": round(variance, 6),
                "std": round(std, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "sample_count": action_count[action],
            }

        return estimates

    # --------------------------------------------------------
    # 滑动窗口趋势检测
    # --------------------------------------------------------

    def detect_trend(
        self,
        *,
        window_size: int = 10,
        action_type: str | None = None,
    ) -> dict[str, Any] | None:
        """检测反馈评分的趋势变化.

        使用滑动窗口 + 线性回归斜率检测趋势:
        - slope > 0.05: "improving"
        - slope < -0.05: "declining"
        - |slope| <= 0.05: "stable"

        Args:
            window_size: 滑动窗口大小 (信号数)
            action_type: 仅分析特定行动类型 (None = 全部)

        Returns:
            {direction, slope, r_squared, window_size, avg_recent, avg_old} 或 None
        """
        # 筛选信号
        signals = self._signals
        if action_type is not None:
            signals = [s for s in signals if s.action_type == action_type]

        if len(signals) < window_size:
            logger.debug("趋势检测: 信号不足 (%d < %d)", len(signals), window_size)
            return None

        # 取最近 window_size 个信号
        recent = signals[-window_size:]

        # 按时间排序
        recent.sort(key=lambda s: s.created_at)

        # 线性回归: y = rating, x = time_index
        n = len(recent)
        x_vals = list(range(n))
        y_vals = [s.rating for s in recent]

        # 最小二乘法
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        denominator_x = sum((x - x_mean) ** 2 for x in x_vals)
        denominator_y = sum((y - y_mean) ** 2 for y in y_vals)

        slope = numerator / denominator_x if denominator_x > 0 else 0.0

        # R²
        if denominator_y > 0:
            r_squared = (numerator ** 2) / (denominator_x * denominator_y)
        else:
            r_squared = 0.0

        # 判断趋势方向
        if slope > 0.05:
            direction = "improving"
        elif slope < -0.05:
            direction = "declining"
        else:
            direction = "stable"

        # 前半段 vs 后半段平均
        mid = n // 2
        avg_old = sum(y_vals[:mid]) / mid if mid > 0 else 0.0
        avg_recent = sum(y_vals[mid:]) / (n - mid) if (n - mid) > 0 else 0.0

        return {
            "direction": direction,
            "slope": round(slope, 6),
            "r_squared": round(r_squared, 4),
            "window_size": window_size,
            "avg_old": round(avg_old, 4),
            "avg_recent": round(avg_recent, 4),
            "delta": round(avg_recent - avg_old, 4),
        }

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    def _time_weight(self, signal_time: float, now: float) -> float:
        """计算时间衰减权重 (指数衰减).

        w(t) = 2^(-(now - t) / half_life_seconds)
        """
        age_seconds = now - signal_time
        half_life_seconds = self._decay_half_life * 3600
        return math.pow(2.0, -age_seconds / half_life_seconds)

    @staticmethod
    def _implicit_to_rating(signal_type: str, value: float) -> float:
        """将隐式信号转换为评分 (-1 ~ 1)."""
        mappings: dict[str, Callable[[float], float]] = {
            "dwell_time": lambda v: min(1.0, v / 30.0) if v > 5 else -0.5,  # 停留 30s+ 为正
            "click": lambda v: 0.5 if v > 0 else 0.0,
            "copy": lambda v: 0.8 if v > 0 else 0.0,
            "share": lambda v: 1.0 if v > 0 else 0.0,
            "skip": lambda v: -0.5 if v > 0 else 0.0,
            "correct": lambda v: -1.0 if v > 0 else 0.0,  # 用户纠正 = 强负反馈
        }
        fn = mappings.get(signal_type, lambda v: 0.0)
        return fn(value)

    @staticmethod
    def _aggregate_by_dimension(
        signals: list[FeedbackSignal],
        weights: list[float],
        key_fn: Callable[[FeedbackSignal], str],
    ) -> dict[str, dict[str, float]]:
        """按维度聚合统计."""
        dim_data: dict[str, list[tuple[float, float]]] = defaultdict(list)

        for s, w in zip(signals, weights):
            dim_data[key_fn(s)].append((s.rating, w))

        result: dict[str, dict[str, float]] = {}
        for dim, values in dim_data.items():
            total_weight = sum(w for _, w in values)
            avg_rating = sum(r * w for r, w in values) / total_weight if total_weight > 0 else 0.0
            positive = sum(1 for r, _ in values if r > 0)
            negative = sum(1 for r, _ in values if r < 0)

            result[dim] = {
                "avg_rating": round(avg_rating, 4),
                "count": len(values),
                "positive_ratio": round(positive / len(values), 4) if values else 0.0,
                "negative_ratio": round(negative / len(values), 4) if values else 0.0,
            }

        return result

    @staticmethod
    def _generate_adjustments(summary: FeedbackSummary) -> list[dict[str, Any]]:
        """生成策略调整建议."""
        adjustments: list[dict[str, Any]] = []

        # 按行动分析
        for action, stats in summary.by_action.items():
            avg_rating = stats.get("avg_rating", 0.0)
            count = stats.get("count", 0)

            if count < 3:
                continue

            if avg_rating < -0.3:
                adjustments.append({
                    "target": "action_selector",
                    "action": action,
                    "adjustment": "reduce_frequency",
                    "reason": f"{action} 平均评分 {avg_rating:.2f} 较低，建议降低使用频率",
                    "strength": abs(avg_rating),
                })
            elif avg_rating > 0.5:
                adjustments.append({
                    "target": "action_selector",
                    "action": action,
                    "adjustment": "increase_frequency",
                    "reason": f"{action} 平均评分 {avg_rating:.2f} 较高，建议优先使用",
                    "strength": avg_rating,
                })

        # 全局评分低时建议保守策略
        if summary.avg_rating < -0.2:
            adjustments.append({
                "target": "action_selector",
                "adjustment": "conservative_mode",
                "reason": f"全局平均评分 {summary.avg_rating:.2f} 较低，启用保守模式",
                "strength": abs(summary.avg_rating),
            })

        return adjustments

    def _rebuild_indexes(self) -> None:
        """重建索引，确保与 _signals 一致."""
        self._by_intent = defaultdict(list)
        self._by_action = defaultdict(list)
        for s in self._signals:
            self._by_intent[s.intent_type].append(s)
            self._by_action[s.action_type].append(s)
        logger.debug("索引重建完成: %d 条信号", len(self._signals))


__all__ = [
    "FeedbackAggregator",
]
