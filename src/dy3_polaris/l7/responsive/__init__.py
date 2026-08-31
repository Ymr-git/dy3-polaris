"""L7 响应式设计 — 包入口 (任务拆分 T6)."""

from __future__ import annotations

from .layout_manager import (
    BREAKPOINTS,
    DESKTOP_COLUMNS,
    MIN_TOUCH_SIZE,
    breakpoint_for,
    layout_plan,
    render_layout,
    responsive_css,
    responsive_layout_wrap,
)

__all__ = [
    "BREAKPOINTS",
    "DESKTOP_COLUMNS",
    "MIN_TOUCH_SIZE",
    "breakpoint_for",
    "layout_plan",
    "render_layout",
    "responsive_css",
    "responsive_layout_wrap",
]
