"""L7 多模态输出 — 交互组件包入口.

任务拆分 T4 交付物:

    param_controller.py  — 参数调节器 (5 参数滑块 + 实时联动)
    virtual_lab.py       — 虚拟实验台 (选择材料→设定条件→模拟结果→输出Artifact组)
    graph_explorer.py     — 知识图谱探索器 (点击展开/双击详情/右键路径)

设计文档: 02-设计/L7-体验呈现设计/layer7-experience-presentation.html Ch.4 §4.3
"""

from __future__ import annotations

from .graph_explorer import render_graph_explorer
from .param_controller import render_param_controller
from .virtual_lab import render_virtual_lab

__all__ = [
    "render_param_controller",
    "render_virtual_lab",
    "render_graph_explorer",
]
