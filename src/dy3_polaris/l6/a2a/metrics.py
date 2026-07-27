"""A2A 协议可观测性指标.

提供协议级指标收集与导出能力：
- 消息计数（按类型/方向）
- 任务统计（成功率/延迟/取消率）
- 会话统计（活跃数/平均时长）
- 能力利用率
- 吞吐量追踪

所有指标均为线程安全，适合异步环境。
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
    """延迟追踪器，记录最近 N 个样本."""
    max_samples: int = 100
    _samples: list[float] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._samples.append(latency_ms)
            if len(self._samples) > self.max_samples:
                self._samples = self._samples[-self.max_samples:]

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def avg_ms(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            return sum(self._samples) / len(self._samples)

    @property
    def max_ms(self) -> float:
        with self._lock:
            return max(self._samples) if self._samples else 0.0

    @property
    def p99_ms(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            s = sorted(self._samples)
            idx = min(int(len(s) * 0.99), len(s) - 1)
            return s[idx]


class A2AMetrics:
    """A2A 协议可观测性指标收集器.

    无侵入式指标收集，可挂载到 A2AMessageBus 上。
    所有计数器线程安全，可在异步环境中直接调用。

    使用示例:
        metrics = A2AMetrics()
        metrics.on_message_sent("TASK_REQUEST", "agent-A", "agent-B")
        metrics.on_task_completed(task_latency_ms=120)
        summary = metrics.export()
    """

    def __init__(self, max_latency_samples: int = 100) -> None:
        self._max_latency = max_latency_samples
        self._started_at: float = time.time()

        # 消息计数: (message_type, direction) -> Counter
        self._msg_counters: dict[tuple[str, str], _Counter] = defaultdict(_Counter)

        # 任务状态计数
        self._task_status_counts: dict[str, _Counter] = defaultdict(_Counter)

        # 任务延迟
        self._task_latency = _LatencyTracker(max_samples=max_latency_samples)

        # Agent 维度计数
        self._agent_task_counts: dict[str, _Counter] = defaultdict(_Counter)

        # 错误计数
        self._error_counts: dict[str, _Counter] = defaultdict(_Counter)

        # 会话事件
        self._session_created = _Counter()
        self._session_closed = _Counter()
        self._session_expired = _Counter()

        # 能力利用率: capability_name -> 被请求次数
        self._capability_usage: dict[str, _Counter] = defaultdict(_Counter)

    # --------------------------------------------------------
    # 消息事件
    # --------------------------------------------------------

    def on_message_sent(self, message_type: str, from_agent: str, to_agent: str) -> None:
        """记录消息发送."""
        self._msg_counters[(message_type, "sent")].inc()
        self._msg_counters[(message_type, f"sent:{from_agent}")].inc()

    def on_message_received(self, message_type: str, to_agent: str) -> None:
        """记录消息接收."""
        self._msg_counters[(message_type, "received")].inc()

    # --------------------------------------------------------
    # 任务事件
    # --------------------------------------------------------

    def on_task_created(self, task_id: str, from_agent: str, to_agent: str) -> None:
        """记录任务创建."""
        self._task_status_counts["created"].inc()
        self._agent_task_counts[to_agent].inc()

    def on_task_completed(self, latency_ms: float = 0.0) -> None:
        """记录任务完成."""
        self._task_status_counts["completed"].inc()
        if latency_ms > 0:
            self._task_latency.record(latency_ms)

    def on_task_failed(self) -> None:
        """记录任务失败."""
        self._task_status_counts["failed"].inc()

    def on_task_cancelled(self) -> None:
        """记录任务取消."""
        self._task_status_counts["cancelled"].inc()

    def on_task_timeout(self) -> None:
        """记录任务超时."""
        self._task_status_counts["timeout"].inc()

    # --------------------------------------------------------
    # 会话事件
    # --------------------------------------------------------

    def on_session_created(self) -> None:
        self._session_created.inc()

    def on_session_closed(self) -> None:
        self._session_closed.inc()

    def on_session_expired(self) -> None:
        self._session_expired.inc()

    # --------------------------------------------------------
    # 错误事件
    # --------------------------------------------------------

    def on_error(self, error_code: str) -> None:
        """记录错误."""
        self._error_counts[error_code].inc()

    # --------------------------------------------------------
    # 能力使用
    # --------------------------------------------------------

    def on_capability_requested(self, capability: str) -> None:
        """记录能力被请求."""
        self._capability_usage[capability].inc()

    # --------------------------------------------------------
    # 导出
    # --------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """导出所有指标为字典.

        Returns:
            包含所有维度指标的字典
        """
        uptime = time.time() - self._started_at

        # 任务成功率
        completed = self._task_status_counts["completed"].value
        failed = self._task_status_counts["failed"].value
        cancelled = self._task_status_counts["cancelled"].value
        timeout = self._task_status_counts["timeout"].value
        total_finished = completed + failed + cancelled + timeout
        success_rate = completed / total_finished if total_finished > 0 else 0.0

        # 消息类型统计
        msg_by_type: dict[str, int] = {}
        for (msg_type, direction), counter in self._msg_counters.items():
            if direction == "sent":
                msg_by_type[msg_type] = counter.value

        return {
            "uptime_seconds": round(uptime, 1),
            "messages": {
                "by_type": msg_by_type,
                "total_sent": sum(
                    c.value for (mt, d), c in self._msg_counters.items() if d == "sent"
                ),
                "total_received": sum(
                    c.value for (mt, d), c in self._msg_counters.items() if d == "received"
                ),
            },
            "tasks": {
                "created": self._task_status_counts["created"].value,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "timeout": timeout,
                "success_rate": round(success_rate, 4),
                "latency_ms": {
                    "avg": round(self._task_latency.avg_ms, 2),
                    "max": round(self._task_latency.max_ms, 2),
                    "p99": round(self._task_latency.p99_ms, 2),
                    "samples": self._task_latency.count,
                },
            },
            "sessions": {
                "created": self._session_created.value,
                "closed": self._session_closed.value,
                "expired": self._session_expired.value,
                "active": self._session_created.value - self._session_closed.value - self._session_expired.value,
            },
            "agents": {
                aid: c.value
                for aid, c in self._agent_task_counts.items()
            },
            "errors": {
                code: c.value
                for code, c in self._error_counts.items()
            },
            "capabilities": {
                name: c.value
                for name, c in self._capability_usage.items()
            },
        }

    def reset(self) -> None:
        """重置所有指标."""
        self._msg_counters.clear()
        self._task_status_counts.clear()
        self._task_latency = _LatencyTracker(max_samples=self._max_latency)
        self._agent_task_counts.clear()
        self._error_counts.clear()
        self._session_created = _Counter()
        self._session_closed = _Counter()
        self._session_expired = _Counter()
        self._capability_usage.clear()
        self._started_at = time.time()


__all__ = ["A2AMetrics"]
