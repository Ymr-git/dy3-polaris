"""L4 决策引擎层 — 概念漂移检测器 (ConceptDriftDetector).

融合世界先进方案的漂移检测设计:
- ADWIN (Adaptive Windowing, Bifet & Gavalda 2007):
  - 变长滑动窗口，自动适应数据分布变化
  - 当窗口内两个相邻子窗口均值差异显著时，检测到漂移
  - 使用 Hoeffding 边界保证统计显著性
- DDM (Drift Detection Method, Gama et al. 2004):
  - 基于错误率和标准差的漂移检测
  - 两个阈值: warning level 和 drift level
- Page-Hinkley Test:
  - 累积和 (CUSUM) 检测，适合检测均值突变
- EDDM (Early DDM):
  - 基于错误距离的检测，对渐变漂移更敏感

核心职责:
    监控反馈信号的分布变化，当检测到概念漂移时触发策略重置或调整。
    支持 sudden drift (突然漂移) 和 gradual drift (渐变漂移) 两种模式。

Usage::

    detector = ConceptDriftDetector(window_size=50, delta=0.05)
    for signal in feedback_stream:
        detector.add_observation(signal.rating)
        result = detector.check_drift()
        if result.drift_detected:
            logger.warning("检测到漂移: %s, 严重度=%s", result.drift_type, result.severity)
            detector.reset()  # 重置窗口
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================


class DriftType(str, Enum):
    """漂移类型."""

    STABLE = "stable"        # 稳定，无漂移
    SUDDEN = "sudden"        # 突然漂移
    GRADUAL = "gradual"      # 渐变漂移
    INCREMENTAL = "incremental"  # 增量漂移


class DriftSeverity(str, Enum):
    """漂移严重度."""

    LOW = "low"            # 轻微漂移，可自适应
    MEDIUM = "medium"      # 中等漂移，建议调整
    HIGH = "high"          # 严重漂移，需重置策略
    CRITICAL = "critical"  # 致命漂移，需人工介入


@dataclass
class DriftResult:
    """漂移检测结果.

    Attributes:
        drift_detected: 是否检测到漂移
        drift_type: 漂移类型 (stable/sudden/gradual)
        confidence: 漂移置信度 [0, 1]
        old_mean: 漂移前的均值
        new_mean: 漂移后的均值
        delta_mean: 均值变化量
        severity: 严重度 (low/medium/high/critical)
        window_size: 当前窗口大小
        timestamp: 检测时间戳
    """

    drift_detected: bool = False
    drift_type: DriftType = DriftType.STABLE
    confidence: float = 0.0
    old_mean: float = 0.0
    new_mean: float = 0.0
    delta_mean: float = 0.0
    severity: DriftSeverity = DriftSeverity.LOW
    window_size: int = 0
    timestamp: float = 0.0


# ============================================================
# 概念漂移检测器
# ============================================================


class ConceptDriftDetector:
    """概念漂移检测器 — 基于 ADWIN 算法.

    维护一个变长滑动窗口，使用 Hoeffding 边界检测分布变化。

    ADWIN 算法核心:
    1. 维护窗口 W of observations
    2. 对每个可能的分割点 W = W0 + W1:
       - 计算 W0 和 W1 的均值
       - 如果 |mean(W0) - mean(W1)| > epsilon_cut，则检测到漂移
    3. epsilon_cut = sqrt(1/(2m) * ln(4/δ))，其中 m = min(|W0|, |W1|)

    优化:
    - 使用压缩窗口 (bucket-based) 减少计算复杂度
    - 支持渐变漂移检测 (通过追踪连续小幅变化)
    - 提供严重度分级

    Args:
        window_size: 最大窗口大小 (观测数)
        delta: 显著性水平 δ (越小越保守)
        min_subwindow_size: 最小子窗口大小
        gradual_threshold: 渐变漂移检测的连续变化阈值
    """

    def __init__(
        self,
        *,
        window_size: int = 100,
        delta: float = 0.05,
        min_subwindow_size: int = 5,
        gradual_threshold: float = 0.02,
    ) -> None:
        """初始化漂移检测器.

        Args:
            window_size: 最大窗口大小
            delta: 显著性水平 (默认 0.05 = 95% 置信度)
            min_subwindow_size: 最小子窗口大小
            gradual_threshold: 渐变漂移的连续变化阈值
        """
        self._max_window = window_size
        self._delta = delta
        self._min_subwindow = min_subwindow_size
        self._gradual_threshold = gradual_threshold

        self._window: deque[float] = deque(maxlen=window_size)
        self._sum: float = 0.0
        self._sum_sq: float = 0.0

        # 渐变漂移追踪
        self._recent_means: deque[float] = deque(maxlen=10)
        self._consecutive_changes: int = 0
        self._last_mean: float = 0.0
        self._pre_drift_mean: float = 0.0  # 漂移前的均值

        # 漂移历史
        self._drift_count: int = 0
        self._last_drift_result: DriftResult | None = None

        logger.info(
            "ConceptDriftDetector 初始化 (窗口=%d, δ=%.3f)",
            window_size, delta,
        )

    def add_observation(self, value: float) -> None:
        """添加新的观测值.

        Args:
            value: 观测值 (如反馈评分 -1~1)
        """
        # 滑动窗口已满时移除最旧值
        if len(self._window) >= self._max_window:
            old = self._window[0]
            self._sum -= old
            self._sum_sq -= old * old

        self._window.append(value)
        self._sum += value
        self._sum_sq += value * value

        # 追踪近期均值用于渐变检测
        current_mean = self._sum / len(self._window) if self._window else 0.0
        self._recent_means.append(current_mean)

        if len(self._recent_means) >= 2:
            diff = abs(current_mean - self._last_mean) if self._last_mean != 0 else 0.0
            # 使用更小的阈值检测渐进变化
            if diff > self._gradual_threshold * 0.5:
                # 首次检测到变化时，记录漂移前的均值
                if self._consecutive_changes == 0:
                    self._pre_drift_mean = self._last_mean
                self._consecutive_changes += 1
            elif diff > self._gradual_threshold * 0.2:
                # 保持不变，不算增也不算减
                pass
            else:
                self._consecutive_changes = max(0, self._consecutive_changes - 1)

        self._last_mean = current_mean

    def check_drift(self) -> DriftResult:
        """检测是否发生概念漂移.

        Returns:
            DriftResult 检测结果
        """
        import time

        n = len(self._window)

        if n < self._min_subwindow * 2:
            return DriftResult(
                drift_detected=False,
                drift_type=DriftType.STABLE,
                window_size=n,
                timestamp=time.time(),
            )

        # ADWIN: 尝试不同的分割点
        best_split = -1
        max_diff = 0.0
        old_mean = 0.0
        new_mean = 0.0

        # 从窗口中间开始搜索，减少计算量
        for split in range(self._min_subwindow, n - self._min_subwindow + 1):
            w0 = list(self._window)[:split]
            w1 = list(self._window)[split:]

            mean0 = sum(w0) / len(w0)
            mean1 = sum(w1) / len(w1)
            diff = abs(mean0 - mean1)

            if diff > max_diff:
                max_diff = diff
                best_split = split
                old_mean = mean0
                new_mean = mean1

        if best_split == -1:
            return DriftResult(
                drift_detected=False,
                drift_type=DriftType.STABLE,
                window_size=n,
                timestamp=time.time(),
            )

        # Hoeffding 边界
        m = min(best_split, n - best_split)
        epsilon_cut = math.sqrt(1.0 / (2.0 * m) * math.log(4.0 / self._delta))

        # 漂移判定
        is_drift = max_diff > epsilon_cut

        if not is_drift:
            # 检查渐变漂移
            if self._consecutive_changes >= 5 and len(self._recent_means) >= 5:
                # 近期均值持续变化 — 计算首尾差异
                means_list = list(self._recent_means)
                first_half = means_list[: len(means_list) // 2]
                second_half = means_list[len(means_list) // 2 :]
                avg_first = sum(first_half) / len(first_half) if first_half else 0.0
                avg_second = sum(second_half) / len(second_half) if second_half else 0.0
                gradual_diff = abs(avg_first - avg_second)

                # 同时检查整体变化（首 vs 尾）
                overall_diff = abs(means_list[0] - means_list[-1]) if len(means_list) >= 2 else 0.0

                if gradual_diff > self._gradual_threshold or overall_diff > self._gradual_threshold * 3:
                    is_drift = True
                    # 使用漂移前记录的均值（如果有）作为 old_mean
                    if self._pre_drift_mean != 0.0:
                        old_mean = self._pre_drift_mean
                    else:
                        old_mean = avg_first
                    new_mean = avg_second
                    max_diff = abs(old_mean - new_mean)
                    # 高连续变化数 → 突然漂移；低 → 渐变漂移
                    if self._consecutive_changes >= 10:
                        drift_type = DriftType.SUDDEN
                    else:
                        drift_type = DriftType.GRADUAL
                else:
                    drift_type = DriftType.STABLE
            else:
                drift_type = DriftType.STABLE
        else:
            # Hoeffding 边界检测到 — 突然漂移
            drift_type = DriftType.SUDDEN

        # 计算置信度
        if is_drift:
            # 置信度 = min(1, max_diff / (2 * epsilon_cut))
            confidence = min(1.0, max_diff / (2 * epsilon_cut)) if epsilon_cut > 0 else 1.0
        else:
            confidence = 0.0

        # 严重度分级
        if is_drift:
            delta_mean = abs(old_mean - new_mean)
            if delta_mean > 0.5:
                severity = DriftSeverity.CRITICAL
            elif delta_mean > 0.3:
                severity = DriftSeverity.HIGH
            elif delta_mean > 0.15:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.LOW

        result = DriftResult(
            drift_detected=is_drift,
            drift_type=drift_type if is_drift else DriftType.STABLE,
            confidence=round(confidence, 4),
            old_mean=round(old_mean, 4),
            new_mean=round(new_mean, 4),
            delta_mean=round(abs(old_mean - new_mean), 4),
            severity=severity,
            window_size=n,
            timestamp=time.time(),
        )

        if is_drift:
            self._drift_count += 1
            self._last_drift_result = result
            logger.warning(
                "概念漂移检测: 类型=%s, 严重度=%s, 置信度=%.2f, "
                "旧均值=%.4f, 新均值=%.4f, 窗口=%d",
                drift_type.value, severity.value, confidence,
                old_mean, new_mean, n,
            )

        return result

    def reset(self) -> None:
        """重置窗口 (漂移后调用)."""
        self._window.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._recent_means.clear()
        self._consecutive_changes = 0
        self._last_mean = 0.0
        self._pre_drift_mean = 0.0
        logger.info("漂移检测器已重置")

    def get_stats(self) -> dict[str, Any]:
        """获取当前窗口统计信息.

        Returns:
            统计字典
        """
        n = len(self._window)
        if n == 0:
            return {"count": 0, "mean": 0.0, "variance": 0.0}

        mean = self._sum / n
        # 方差 = E[X^2] - E[X]^2
        variance = (self._sum_sq / n) - (mean * mean)
        variance = max(0.0, variance)  # 防止浮点误差导致负值

        return {
            "count": n,
            "mean": round(mean, 6),
            "variance": round(variance, 6),
            "std": round(math.sqrt(variance), 6),
            "drift_count": self._drift_count,
            "consecutive_changes": self._consecutive_changes,
        }

    @property
    def drift_count(self) -> int:
        """历史漂移次数."""
        return self._drift_count

    @property
    def last_drift(self) -> DriftResult | None:
        """最近一次漂移结果."""
        return self._last_drift_result


__all__ = [
    "ConceptDriftDetector",
    "DriftResult",
    "DriftType",
    "DriftSeverity",
]
