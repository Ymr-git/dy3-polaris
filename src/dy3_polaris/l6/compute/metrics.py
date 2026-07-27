"""算力资源度量与可观测性.

跟踪算力调度的运行时指标：
- 资源利用率（按类型）
- 任务吞吐量与延迟
- 降级事件计数
- 队列等待时间

所有指标线程安全，适合异步环境。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Counter:
    """线程安全计数器."""
    _value: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    @property
    def value(self) -> int:
        return self._value


@dataclass
class _LatencyTracker:
    """延迟样本收集."""
    max_samples: int = 100
    _samples: list[float] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def record(self, ms: float) -> None:
        with self._lock:
            self._samples.append(ms)
            if len(self._samples) > self.max_samples:
                self._samples = self._samples[-self.max_samples:]

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def avg(self) -> float:
        with self._lock:
            return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def max(self) -> float:
        with self._lock:
            return max(self._samples) if self._samples else 0.0


class ComputeMetrics:
    """算力资源度量收集器.

    记录调度器运行时指标，支持一键导出。
    """

    def __init__(self) -> None:
        self._started_at = time.time()
        # 任务计数
        self._task_created = _Counter()
        self._task_completed = _Counter()
        self._task_failed = _Counter()
        self._task_cancelled = _Counter()
        # 降级计数
        self._degradations = _Counter()
        # 按类型计数
        self._allocations_by_type: dict[str, _Counter] = defaultdict(_Counter)
        # 任务执行延迟
        self._task_latency = _LatencyTracker(max_samples=200)
        # 队列等待时间
        self._queue_wait = _LatencyTracker(max_samples=200)

    # --------------------------------------------------------
    # 事件记录
    # --------------------------------------------------------

    def on_task_created(self, resource_type: str) -> None:
        self._task_created.inc()
        self._allocations_by_type[resource_type].inc()

    def on_task_completed(self, latency_ms: float = 0.0, wait_ms: float = 0.0) -> None:
        self._task_completed.inc()
        if latency_ms > 0:
            self._task_latency.record(latency_ms)
        if wait_ms > 0:
            self._queue_wait.record(wait_ms)

    def on_task_failed(self) -> None:
        self._task_failed.inc()

    def on_task_cancelled(self) -> None:
        self._task_cancelled.inc()

    def on_degradation(self) -> None:
        self._degradations.inc()

    # --------------------------------------------------------
    # 导出
    # --------------------------------------------------------

    def export(self) -> dict[str, Any]:
        total_finished = self._task_completed.value + self._task_failed.value + self._task_cancelled.value
        success_rate = self._task_completed.value / total_finished if total_finished > 0 else 0.0

        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "tasks": {
                "created": self._task_created.value,
                "completed": self._task_completed.value,
                "failed": self._task_failed.value,
                "cancelled": self._task_cancelled.value,
                "success_rate": round(success_rate, 4),
            },
            "degradations": self._degradations.value,
            "allocations_by_type": {t: c.value for t, c in self._allocations_by_type.items()},
            "latency_ms": {
                "avg": round(self._task_latency.avg, 2),
                "max": round(self._task_latency.max, 2),
                "samples": self._task_latency.count,
            },
            "queue_wait_ms": {
                "avg": round(self._queue_wait.avg, 2),
                "max": round(self._queue_wait.max, 2),
                "samples": self._queue_wait.count,
            },
        }

    def reset(self) -> None:
        self._task_created = _Counter()
        self._task_completed = _Counter()
        self._task_failed = _Counter()
        self._task_cancelled = _Counter()
        self._degradations = _Counter()
        self._allocations_by_type.clear()
        self._task_latency = _LatencyTracker(max_samples=200)
        self._queue_wait = _LatencyTracker(max_samples=200)
        self._started_at = time.time()


__all__ = ["ComputeMetrics"]
