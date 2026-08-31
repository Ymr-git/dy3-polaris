"""L7 性能监控 — 包入口 (任务拆分 T6)."""

from __future__ import annotations

from .performance import (
    RENDER_BUDGET,
    WEB_VITALS_BUDGET,
    WS_UI_BUDGET,
    PerformanceTracker,
    check_render_latency,
    check_vitals,
    check_ws_latency,
)

__all__ = [
    "RENDER_BUDGET",
    "WEB_VITALS_BUDGET",
    "WS_UI_BUDGET",
    "PerformanceTracker",
    "check_render_latency",
    "check_vitals",
    "check_ws_latency",
]
