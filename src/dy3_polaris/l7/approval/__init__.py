"""L7 CC2 审批 — 包入口.

任务拆分 T5 交付物:

    plan_preview.py   — 教学计划预览面板 (策略摘要/KP列表/Agent分配/时长/前置/预期)
    approval_flow.py  — 审批操作流程 (批准/拒绝/修改 + 历史记录)
    quick_mode.py     — 快速审批模式 (信任模式 + 规则预设 + 安全拦截)
    plan_rendering.py — 教学计划渲染 (策略文本 + 知识图谱高亮 + 预期对比)

设计文档: 02-设计/L7-体验呈现设计/layer7-experience-presentation.html Ch.7
"""

from __future__ import annotations

from .approval_flow import render_approval_flow
from .plan_preview import render_plan_preview
from .plan_rendering import render_plan_rendering
from .quick_mode import render_quick_mode

__all__ = [
    "render_plan_preview",
    "render_approval_flow",
    "render_quick_mode",
    "render_plan_rendering",
]
