"""L7 性能监控 — 性能预算 (monitoring/performance.py).

任务拆分 T6 · 设计文档 Ch.8.4。

性能预算监控:

| 指标 | 预算 |
|---|---|
| FCP | ≤1.5s |
| LCP | ≤2.5s |
| FID | ≤100ms |
| CLS | ≤0.1 |
| Artifact 渲染 (文本) | ≤500ms |
| Artifact 渲染 (图表) | ≤2s |
| WebSocket → UI | ≤200ms |

融合世界先进方案:
- Web Vitals (Google): 四大核心指标 + 预算告警
- Performance API mark/measure: 渲染链路埋点
"""

from __future__ import annotations

import time
from typing import Any

#: Web Vitals 预算 (Ch.8.4)
WEB_VITALS_BUDGET: dict[str, float] = {
    "FCP": 1.5,
    "LCP": 2.5,
    "FID": 0.1,
    "CLS": 0.1,
}

#: 渲染延迟预算 (秒)
RENDER_BUDGET: dict[str, float] = {
    "text": 0.5,
    "chart": 2.0,
    "molecule": 2.0,
    "formula": 0.5,
    "table": 0.5,
}

#: WebSocket 消息 → UI 预算 (秒)
WS_UI_BUDGET: float = 0.2


def check_vitals(vitals: dict[str, float]) -> dict[str, Any]:
    """检查 Web Vitals 是否超预算.

    Args:
        vitals: {FCP: s, LCP: s, FID: ms, CLS: 比值}。

    Returns:
        {overall, metrics: [{name, value, budget, ok}]}
    """
    metrics = []
    for name, budget in WEB_VITALS_BUDGET.items():
        value = float(vitals.get(name, 0.0))
        # FID 单位为 ms, 预算 100ms; 其余单位为 s
        ok = value <= (budget if name != "FID" else budget * 1000)
        metrics.append({
            "name": name,
            "value": value,
            "budget": budget,
            "ok": ok,
        })
    return {
        "overall": all(m["ok"] for m in metrics),
        "metrics": metrics,
    }


def check_render_latency(renderer_type: str, elapsed_ms: float) -> dict[str, Any]:
    """检查渲染延迟是否超预算.

    Args:
        renderer_type: text/chart/molecule/formula/table。
        elapsed_ms: 渲染耗时 (ms)。

    Returns:
        {budget_ms, elapsed_ms, ok, overshoot_ms}
    """
    budget_ms = RENDER_BUDGET.get(renderer_type, 1.0) * 1000
    elapsed_ms = float(elapsed_ms)
    return {
        "budget_ms": budget_ms,
        "elapsed_ms": round(elapsed_ms, 2),
        "ok": elapsed_ms <= budget_ms,
        "overshoot_ms": round(max(0.0, elapsed_ms - budget_ms), 2),
    }


def check_ws_latency(elapsed_ms: float) -> bool:
    """检查 WebSocket 消息到 UI 的延迟."""
    return float(elapsed_ms) <= WS_UI_BUDGET * 1000


class PerformanceTracker:
    """渲染性能追踪器 (Performance API mark/measure 语义).

    使用示例::

        tracker = PerformanceTracker()
        tracker.mark("render:start")
        ...
        tracker.mark("render:end")
        tracker.measure("render", "render:start", "render:end")
    """

    def __init__(self) -> None:
        self._marks: dict[str, float] = {}
        self._measures: list[dict[str, Any]] = []
        self._alerts: list[dict[str, Any]] = []

    def mark(self, name: str) -> None:
        """记录时间标记."""
        self._marks[name] = time.monotonic()

    def measure(self, name: str, start_mark: str, end_mark: str) -> dict[str, Any]:
        """测量两个标记间的耗时 (毫秒).

        Returns:
            {name, duration_ms, budget_ms, ok, overshoot_ms}
        """
        start = self._marks.get(start_mark)
        end = self._marks.get(end_mark)
        if start is None or end is None:
            return {"name": name, "duration_ms": 0.0, "budget_ms": None, "ok": True, "overshoot_ms": 0.0}
        duration_ms = (end - start) * 1000
        budget_ms = RENDER_BUDGET.get(name, 1.0) * 1000
        result = {
            "name": name,
            "duration_ms": round(duration_ms, 2),
            "budget_ms": budget_ms,
            "ok": duration_ms <= budget_ms,
            "overshoot_ms": round(max(0.0, duration_ms - budget_ms), 2),
        }
        self._measures.append(result)
        if not result["ok"]:
            self._alerts.append({"name": name, "type": "budget_overshoot", **result})
        return result

    def report(self) -> dict[str, Any]:
        """生成性能报告.

        Returns:
            {measures, alerts, alerts_count}
        """
        return {
            "measures": self._measures,
            "alerts": self._alerts,
            "alerts_count": len(self._alerts),
        }
