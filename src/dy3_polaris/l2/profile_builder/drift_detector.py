"""概念漂移检测器 — 检测学习者行为模式的变化.

融合世界先进方案:
- ADWIN (Adaptive Windowing): 自适应滑动窗口, 检测分布变化
- DDM (Drift Detection Method): 基于错误率的均值和标准差
- 2025 研究: BKT 是最稳定的 KT 模型, 但仍需漂移检测

当检测到漂移时, 建议触发模型重训练.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable


class LearnerDriftDetector:
    """概念漂移检测器 (面向学情画像) — ADWIN + DDM 双方法."""

    def __init__(
        self,
        adwin_delta: float = 0.002,
        ddm_warning_level: float = 2.0,
        ddm_drift_level: float = 3.0,
    ):
        self.adwin_delta = adwin_delta
        self.ddm_warning_level = ddm_warning_level
        self.ddm_drift_level = ddm_drift_level
        self._window: deque[float] = deque(maxlen=1000)
        self._ddm_min_p: float = float('inf')
        self._ddm_min_ps: float = float('inf')
        self._ddm_count: int = 0
        # 漂移触发自动重训练相关状态
        self._retraining_callback: Callable[[dict], Any] | None = None
        self._drift_history: list[dict[str, Any]] = []

    def add_observation(self, value: float) -> dict[str, Any]:
        """添加观测值并检测漂移.

        Args:
            value: 观测值 (如正确率 0/1, 或连续值)

        Returns:
            检测结果字典: {
                'drift_detected': bool,
                'method': 'adwin' | 'ddm' | None,
                'warning': bool,
                'value': float,
                'window_mean': float,
                'retraining_result': Any  # 仅当检测到漂移且设置了回调时存在
            }

        当检测到漂移时:
        1. 将漂移事件追加到历史记录 (``get_drift_history`` 可查询);
        2. 若设置了重训练回调 (``set_retraining_callback``), 自动调用
           ``trigger_retraining`` 并将结果挂在返回字典的 ``retraining_result`` 键下.
        """
        self._window.append(value)
        self._ddm_count += 1

        # ADWIN check
        adwin_drift = self._adwin_check()

        # DDM check
        ddm_result = self._ddm_check(value)

        window_mean = (
            sum(self._window) / len(self._window) if self._window else 0.0
        )

        if adwin_drift:
            result: dict[str, Any] = {
                'drift_detected': True,
                'method': 'adwin',
                'warning': False,
                'value': value,
                'window_mean': window_mean,
            }
        elif ddm_result['drift']:
            result = {
                'drift_detected': True,
                'method': 'ddm',
                'warning': ddm_result['warning'],
                'value': value,
                'window_mean': window_mean,
            }
        else:
            result = {
                'drift_detected': False,
                'method': None,
                'warning': ddm_result['warning'],
                'value': value,
                'window_mean': window_mean,
            }

        # 检测到漂移: 记录历史并自动触发重训练 (若设置了回调)
        if result['drift_detected']:
            drift_info: dict[str, Any] = {
                'method': result['method'],
                'value': value,
                'window_mean': window_mean,
                'warning': result['warning'],
                'observation_count': self._ddm_count,
            }
            self._drift_history.append(drift_info)
            if self._retraining_callback is not None:
                result['retraining_result'] = self.trigger_retraining(drift_info)

        return result

    def _adwin_check(self) -> bool:
        """ADWIN 检测: 比较窗口两半的均值差异."""
        if len(self._window) < 10:
            return False
        window_list = list(self._window)
        n = len(window_list)
        best_split = 0
        max_diff = 0.0

        for split in range(5, n - 5):
            w0 = window_list[:split]
            w1 = window_list[split:]
            mean_diff = abs(sum(w0) / len(w0) - sum(w1) / len(w1))
            if mean_diff > max_diff:
                max_diff = mean_diff
                best_split = split

        # Hoeffding bound
        epsilon = math.sqrt(2 * math.log(1 / self.adwin_delta) / n)

        if max_diff > epsilon:
            # Drift detected - shrink window
            for _ in range(best_split):
                if self._window:
                    self._window.popleft()
            return True
        return False

    def _ddm_check(self, value: float) -> dict[str, bool]:
        """DDM 检测: 基于错误率 (value=1=correct, value=0=error)."""
        # Track error rate (1 - value for binary)
        error = 1.0 - value if value in (0.0, 1.0) else max(0.0, 1.0 - value)

        n = self._ddm_count
        if n < 10:
            return {'drift': False, 'warning': False}

        # Running mean and std of error
        p = sum(1.0 - v for v in self._window) / n
        s = math.sqrt(p * (1 - p) / n) if 0 < p < 1 else 0.0

        if p + s < self._ddm_min_p + self._ddm_min_ps:
            self._ddm_min_p = p
            self._ddm_min_ps = s

        warning = (p + s) > (self._ddm_min_p + self.ddm_warning_level * self._ddm_min_ps)
        drift = (p + s) > (self._ddm_min_p + self.ddm_drift_level * self._ddm_min_ps)

        return {'drift': drift, 'warning': warning}

    def reset(self) -> None:
        """重置检测器状态."""
        self._window.clear()
        self._ddm_min_p = float('inf')
        self._ddm_min_ps = float('inf')
        self._ddm_count = 0
        self._drift_history.clear()

    # --- 漂移触发自动重训练 ---

    def set_retraining_callback(self, callback: Callable[[dict], Any]) -> None:
        """设置漂移触发时的重训练回调函数.

        设置后, 每次 ``add_observation`` 检测到漂移时会自动调用该回调,
        回调签名: ``callback(drift_info: dict) -> Any`` (返回重训练结果).

        传 None 可清除已设置的回调.

        Args:
            callback: 漂移触发回调; None 表示清除回调.
        """
        self._retraining_callback = callback

    def trigger_retraining(self, drift_info: dict[str, Any]) -> dict[str, Any] | None:
        """触发重训练 (若已设置回调).

        Args:
            drift_info: 漂移事件信息字典 (含 method / value / window_mean 等).

        Returns:
            回调返回的重训练结果; 未设置回调时返回 None.
        """
        if self._retraining_callback is None:
            return None
        return self._retraining_callback(drift_info)

    def get_drift_history(self) -> list[dict[str, Any]]:
        """返回历史漂移记录列表 (浅拷贝).

        每条记录含: method / value / window_mean / warning / observation_count.

        Returns:
            漂移事件字典列表 (按发生顺序); 无漂移时为空列表.
        """
        return list(self._drift_history)

    def get_stats(self) -> dict[str, Any]:
        """获取检测器统计信息."""
        return {
            'window_size': len(self._window),
            'window_mean': sum(self._window) / len(self._window) if self._window else 0.0,
            'ddm_min_p': self._ddm_min_p if self._ddm_min_p != float('inf') else 0.0,
            'observation_count': self._ddm_count,
        }


# 向后兼容别名 (已弃用，请使用 LearnerDriftDetector)
ConceptDriftDetector = LearnerDriftDetector

__all__ = [
    "LearnerDriftDetector",
    "ConceptDriftDetector",  # 向后兼容别名 (已弃用)
]
