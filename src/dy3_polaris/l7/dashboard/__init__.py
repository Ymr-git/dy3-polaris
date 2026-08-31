"""L7 学情面板 — 包入口.

任务拆分 T4 交付物:

    bkt_dashboard.py      — BKT 学情面板 (热力图+单KP详情+瓶颈+知识拓扑)
    progress_panel.py     — 学习进度面板 (总体掌握度+域级+薄弱点+学习路径)
    debate_viz.py         — 辩论可视化 (时间线+收敛图+裁决雷达图)
    interaction_modes.py  — 面板交互模式 (下钻/上卷+时间旅行+对比)

设计文档: 02-设计/L7-体验呈现设计/layer7-experience-presentation.html Ch.5
"""

from __future__ import annotations

from .bkt_dashboard import render_bkt_dashboard, render_kp_detail
from .debate_viz import render_debate
from .interaction_modes import (
    render_comparison,
    render_drill_down,
    render_time_travel,
)
from .progress_panel import render_progress_panel

__all__ = [
    "render_bkt_dashboard",
    "render_kp_detail",
    "render_progress_panel",
    "render_debate",
    "render_drill_down",
    "render_time_travel",
    "render_comparison",
]
