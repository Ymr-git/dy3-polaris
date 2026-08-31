"""L7 溯源可视化 — 包入口.

任务拆分 T5 交付物:

    timeline.py            — 溯源时间线 (5 类事件 + 类型筛选 + 哈希链验证)
    decision_trace.py      — 决策溯源 (6 维雷达 + 5 步链路 + 三级深度)
    agent_contribution.py  — Agent 交互链时间线 (逐步: 谁 → 做了什么 → 传给谁)
    branch_merge.py        — 分支合并可视化 (Git 风格 + 原因标注)

设计文档: 02-设计/L7-体验呈现设计/layer7-experience-presentation.html Ch.6
"""

from __future__ import annotations

from .agent_contribution import render_agent_contribution
from .branch_merge import render_branch_merge
from .decision_trace import render_decision_trace
from .timeline import render_timeline

__all__ = [
    "render_timeline",
    "render_decision_trace",
    "render_agent_contribution",
    "render_branch_merge",
]
