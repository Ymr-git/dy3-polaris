"""DKT 启发的序列特征工程 (纯 Python 统计实现).

设计依据:
- DKT (Deep Knowledge Tracing, Piech et al., NeurIPS 2015): 用序列建模追踪
  知识状态演化; 本模块不引入深度学习框架, 而是用纯 Python 统计特征近似
  DKT 所需的序列信号 (滑动窗口正确率 / 趋势 / 速度 / 连续性 / 时序模式).
- DKVMN (Zhang et al., WWW 2017): 外部记忆矩阵刻画知识点状态; 本模块以
  ``response_time_stats`` / ``streak_info`` 等可解释统计量替代隐式记忆.
- SAKT (Pandey & Karypis, EDMM 2019): 自注意力加权历史交互; ``recent_trend``
  与 ``mastery_velocity`` 提供对近期交互的加权近似.

模块构成:
1. ``SequenceFeatures``: 序列特征数据类 (滑窗正确率 / 趋势 / 响应时间 /
   连续对错 / 掌握度速度 / 时序模式).
2. ``SequenceFeatureExtractor``: 序列特征提取器, 提供完整 ``extract`` 与
   各原子特征计算方法.
3. ``TemporalPatternClassifier``: 时序模式分类器 (steady_bloom / late_bloom /
   early_decay / oscillating / stable).

不依赖 numpy, 仅使用 ``math`` 与标准库.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from dy3_polaris.l2.models import AnswerRecord


# ============================================================
# 1. 常量定义
# ============================================================

# 趋势检测斜率阈值 (每步): |slope| 超过该值判定为 improving/declining
TREND_SLOPE_THRESHOLD: float = 0.01

# 响应时间趋势检测: 相对斜率阈值 (slope / mean), 适应不同量纲的响应时间
RT_RELATIVE_SLOPE_THRESHOLD: float = 0.01

# 时序模式分类阈值
STEADY_BLOOM_SLOPE: float = 0.02   # 稳步提升: 斜率 > 0.02
OSCILLATING_STD: float = 0.3       # 震荡: 标准差 > 0.3
STABLE_STD: float = 0.1            # 稳定: 标准差 < 0.1
LATE_BLOOM_FIRST_MAX: float = 0.5  # 后期爆发: 前半段 < 0.5
LATE_BLOOM_SECOND_MIN: float = 0.7  # 后期爆发: 后半段 > 0.7
EARLY_DECAY_FIRST_MIN: float = 0.7  # 早期衰退: 前半段 > 0.7
EARLY_DECAY_SECOND_MAX: float = 0.5  # 早期衰退: 后半段 < 0.5


# ============================================================
# 2. 辅助函数
# ============================================================


def _linear_slope(values: list[float]) -> float:
    """最小二乘线性回归斜率 (values 对等间距索引 0..n-1).

    Args:
        values: 数值序列.

    Returns:
        斜率; 序列长度 < 2 或分母为 0 时返回 0.0.
    """
    n = len(values)
    if n < 2:
        return 0.0
    xbar = (n - 1) / 2.0
    ybar = sum(values) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(values):
        dx = i - xbar
        num += dx * (y - ybar)
        den += dx * dx
    if den == 0.0:
        return 0.0
    return num / den


def _population_std(values: list[float]) -> float:
    """总体标准差."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(var)


def _softmax(scores: list[float]) -> list[float]:
    """数值稳定的 softmax."""
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    if z <= 0.0:
        n = len(scores)
        return [1.0 / n] * n
    return [e / z for e in exps]


# ============================================================
# 3. SequenceFeatures 数据类
# ============================================================


@dataclass
class SequenceFeatures:
    """序列特征数据类 (DKT 启发的统计特征聚合).

    Attributes:
        sliding_window_accuracy: 滑动窗口正确率序列 (每个窗口一个 [0,1] 值).
        recent_trend: 近期趋势, "improving" / "declining" / "stable".
        response_time_stats: 响应时间统计 ``{"mean","std","trend"}``.
        streak_info: 连续对错信息
            ``{"current_streak","max_correct_streak","max_wrong_streak","current_type"}``.
        mastery_velocity: 掌握度变化速度 (后半段正确率 - 前半段正确率, [-1,1]).
        temporal_pattern: 时序模式分类
            "steady_bloom" / "late_bloom" / "early_decay" / "oscillating" / "stable".
    """

    sliding_window_accuracy: list[float] = field(default_factory=list)
    recent_trend: str = "stable"
    response_time_stats: dict[str, Any] = field(default_factory=dict)
    streak_info: dict[str, Any] = field(default_factory=dict)
    mastery_velocity: float = 0.0
    temporal_pattern: str = "stable"


# ============================================================
# 4. TemporalPatternClassifier 时序模式分类器
# ============================================================


class TemporalPatternClassifier:
    """时序模式分类器.

    基于答题正确性序列的统计量 (前/后半段正确率、标准差、线性斜率) 将学习轨迹
    归类为五种时序模式之一:

    - ``early_decay``:  早期衰退 (前半段 > 0.7, 后半段 < 0.5);
    - ``late_bloom``:   后期爆发 (前半段 < 0.5, 后半段 > 0.7);
    - ``steady_bloom``: 稳步提升 (斜率 > 0.02);
    - ``oscillating``:  震荡 (标准差 > 0.3);
    - ``stable``:       稳定 (标准差 < 0.1).

    判定优先级: 先判定具明确半段条件的 early_decay / late_bloom, 再判定
    斜率驱动的 steady_bloom, 之后是方差驱动的 oscillating / stable,
    兜底为 ``stable``.
    """

    def classify(self, records: list[AnswerRecord]) -> str:
        """对答题记录序列进行时序模式分类.

        Args:
            records: 答题记录列表 (按时间顺序; 内部不再排序, 由调用方保证).

        Returns:
            时序模式标签字符串.
        """
        if not records:
            return "stable"
        corrects = [1 if r.correct else 0 for r in records]
        n = len(corrects)
        if n < 2:
            return "stable"

        mid = n // 2
        first = corrects[:mid]
        second = corrects[mid:]
        if not first or not second:
            return "stable"

        first_acc = sum(first) / len(first)
        second_acc = sum(second) / len(second)
        std = _population_std([float(c) for c in corrects])
        slope = _linear_slope([float(c) for c in corrects])

        # 1) 早期衰退 / 后期爆发 (具明确半段条件, 优先判定)
        if first_acc > EARLY_DECAY_FIRST_MIN and second_acc < EARLY_DECAY_SECOND_MAX:
            return "early_decay"
        if first_acc < LATE_BLOOM_FIRST_MAX and second_acc > LATE_BLOOM_SECOND_MIN:
            return "late_bloom"
        # 2) 稳步提升 (正向斜率)
        if slope > STEADY_BLOOM_SLOPE:
            return "steady_bloom"
        # 3) 震荡 (高方差)
        if std > OSCILLATING_STD:
            return "oscillating"
        # 4) 稳定 (低方差)
        if std < STABLE_STD:
            return "stable"
        # 兜底
        return "stable"


# ============================================================
# 5. SequenceFeatureExtractor 序列特征提取器
# ============================================================


class SequenceFeatureExtractor:
    """DKT 启发的序列特征提取器 (纯 Python 统计实现).

    无状态引擎类, 适合并发复用. 所有原子特征计算方法均以 ``list[AnswerRecord]``
    为输入, ``extract`` 汇总为 :class:`SequenceFeatures`.
    """

    def extract(
        self, records: list[AnswerRecord], window_size: int = 5
    ) -> SequenceFeatures:
        """提取完整序列特征.

        内部先按 ``timestamp`` 升序排序 (保证时序正确), 再调用各原子特征方法.

        Args:
            records: 答题记录列表 (可乱序, 内部按时间戳排序).
            window_size: 滑动窗口大小, 默认 5.

        Returns:
            :class:`SequenceFeatures` 实例.
        """
        sorted_records = sorted(records, key=lambda r: r.timestamp)
        sw = self._sliding_window_accuracy(sorted_records, window_size)
        return SequenceFeatures(
            sliding_window_accuracy=sw,
            recent_trend=self._detect_trend(sw),
            response_time_stats=self._response_time_analysis(sorted_records),
            streak_info=self._streak_analysis(sorted_records),
            mastery_velocity=self._mastery_velocity(sorted_records),
            temporal_pattern=self._classify_temporal_pattern(sorted_records),
        )

    # --- 滑动窗口正确率 ---

    def _sliding_window_accuracy(
        self, records: list[AnswerRecord], window_size: int
    ) -> list[float]:
        """计算滑动窗口正确率序列.

        以步长 1 滑动大小为 ``window_size`` 的窗口, 每个窗口输出答对率.
        - 空记录 -> ``[]``;
        - 记录数 < ``window_size`` -> 返回单元素列表 (整体正确率);
        - 否则 -> ``len - window_size + 1`` 个窗口正确率.

        Args:
            records: 答题记录列表 (应已按时间排序).
            window_size: 窗口大小.

        Returns:
            每个窗口的正确率列表.
        """
        n = len(records)
        if n == 0:
            return []
        if n < window_size:
            acc = sum(1 for r in records if r.correct) / n
            return [acc]
        corrects = [1 if r.correct else 0 for r in records]
        return [
            sum(corrects[i : i + window_size]) / window_size
            for i in range(n - window_size + 1)
        ]

    # --- 趋势检测 (线性回归) ---

    def _detect_trend(self, values: list[float]) -> str:
        """基于最小二乘线性回归斜率检测趋势.

        - 斜率 > 阈值 -> ``"improving"``;
        - 斜率 < -阈值 -> ``"declining"``;
        - 否则 -> ``"stable"``.

        Args:
            values: 数值序列 (如滑动窗口正确率).

        Returns:
            趋势标签.
        """
        if len(values) < 2:
            return "stable"
        slope = _linear_slope(values)
        if slope > TREND_SLOPE_THRESHOLD:
            return "improving"
        if slope < -TREND_SLOPE_THRESHOLD:
            return "declining"
        return "stable"

    # --- 响应时间分析 ---

    def _response_time_analysis(self, records: list[AnswerRecord]) -> dict[str, Any]:
        """响应时间统计分析.

        从各记录提取 ``response_time`` (None 视为未采集并跳过), 计算:

        - ``mean``: 平均响应时间;
        - ``std``: 总体标准差;
        - ``trend``: 基于相对斜率 (slope/mean) 的趋势,
          ``"increasing"`` (变慢) / ``"decreasing"`` (变快) / ``"stable"``.

        无数据时返回 ``{"mean": 0.0, "std": 0.0, "trend": "stable"}``.

        Args:
            records: 答题记录列表.

        Returns:
            响应时间统计字典.
        """
        rts = [
            float(r.response_time)
            for r in records
            if getattr(r, "response_time", None) is not None
        ]
        if not rts:
            return {"mean": 0.0, "std": 0.0, "trend": "stable"}
        n = len(rts)
        mean = sum(rts) / n
        std = _population_std(rts)
        if n < 2:
            trend = "stable"
        else:
            slope = _linear_slope(rts)
            rel = slope / mean if mean > 0 else 0.0
            if rel > RT_RELATIVE_SLOPE_THRESHOLD:
                trend = "increasing"
            elif rel < -RT_RELATIVE_SLOPE_THRESHOLD:
                trend = "decreasing"
            else:
                trend = "stable"
        return {"mean": mean, "std": std, "trend": trend}

    # --- 连续对错分析 ---

    def _streak_analysis(self, records: list[AnswerRecord]) -> dict[str, Any]:
        """连续对错分析.

        - ``current_streak``: 末尾连续相同结果的长度;
        - ``current_type``: 末尾结果类型 ``"correct"`` / ``"wrong"`` / ``"none"``;
        - ``max_correct_streak``: 最长连续答对数;
        - ``max_wrong_streak``: 最长连续答错数.

        Args:
            records: 答题记录列表.

        Returns:
            连续对错信息字典.
        """
        if not records:
            return {
                "current_streak": 0,
                "max_correct_streak": 0,
                "max_wrong_streak": 0,
                "current_type": "none",
            }
        corrects = [1 if r.correct else 0 for r in records]
        max_correct = 0
        max_wrong = 0
        cur_correct = 0
        cur_wrong = 0
        for c in corrects:
            if c:
                cur_correct += 1
                cur_wrong = 0
                if cur_correct > max_correct:
                    max_correct = cur_correct
            else:
                cur_wrong += 1
                cur_correct = 0
                if cur_wrong > max_wrong:
                    max_wrong = cur_wrong
        last = corrects[-1]
        current_streak = 0
        for c in reversed(corrects):
            if c == last:
                current_streak += 1
            else:
                break
        return {
            "current_streak": current_streak,
            "max_correct_streak": max_correct,
            "max_wrong_streak": max_wrong,
            "current_type": "correct" if last else "wrong",
        }

    # --- 掌握度变化速度 ---

    def _mastery_velocity(self, records: list[AnswerRecord]) -> float:
        """掌握度变化速度 (后半段正确率 - 前半段正确率).

        正值表示掌握度上升, 负值表示下降, 取值 [-1, 1]. 记录数 < 2 时返回 0.0.

        Args:
            records: 答题记录列表.

        Returns:
            掌握度变化速度.
        """
        n = len(records)
        if n < 2:
            return 0.0
        mid = n // 2
        first = records[:mid]
        second = records[mid:]
        if not first or not second:
            return 0.0
        first_acc = sum(1 for r in first if r.correct) / len(first)
        second_acc = sum(1 for r in second if r.correct) / len(second)
        return second_acc - first_acc

    # --- 时序模式分类 ---

    def _classify_temporal_pattern(self, records: list[AnswerRecord]) -> str:
        """时序模式分类 (委托 :class:`TemporalPatternClassifier`).

        Args:
            records: 答题记录列表.

        Returns:
            时序模式标签.
        """
        return TemporalPatternClassifier().classify(records)


# ============================================================
# __all__
# ============================================================

__all__ = [
    "SequenceFeatures",
    "SequenceFeatureExtractor",
    "TemporalPatternClassifier",
    "TREND_SLOPE_THRESHOLD",
    "STEADY_BLOOM_SLOPE",
    "OSCILLATING_STD",
    "STABLE_STD",
]
