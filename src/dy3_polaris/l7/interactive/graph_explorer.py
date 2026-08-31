"""L7 多模态输出 — 知识图谱探索器 (graph_explorer.py).

任务拆分 T4 · 交互模块。

封装 GraphRenderer 的交互模式: 点击展开子图 / 双击进 KP 详情 /
右键查最短路径 / 拖拽缩放 / 节点过滤。

对 GraphRenderer 进行交互模式增强, 输出增强配置。
由于 GraphRenderer 已实现 vis.js 配置生成, 本模块复用其输出并
附加交互事件 schema 定义。
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import build_descriptor
from ..dashboard._common import dashboard_wrap
from ..renderers.graph_renderer import GraphRenderer


def render_graph_explorer(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    nodes: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
    bkt_state: dict[str, Any] | None = None,
    layout: str = "force",
    theme: str = "light",
) -> dict[str, Any]:
    """渲染知识图谱探索器 (增强版 GraphRenderer).

    在 GraphRenderer 输出基础上附加:
    - 点击展开子图: 点击节点 → 展开关联 KP
    - 双击详情: 双击节点 → 跳转 KP 详情面板
    - 右键查路径: 右键节点 → 查询到目标 KP 的最短路径
    - 节点过滤: 按域/掌握度/标签筛选

    Args:
        nodes: 自定义节点 (默认 42 KP 全集)。
        edges: 自定义边。
        bkt_state: BKT 状态 (用于着色)。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    if nodes is None or edges is None:
        # 委托 GraphRenderer 生成默认图谱
        g_artifact = Artifact(
            artifact_id="graph-explorer",
            payload={"nodes": nodes or [], "edges": edges or [], "layout": layout},
            learner_context={"bkt_state": bkt_state or {}},
        )
        g_desc = GraphRenderer().render(g_artifact, context or RenderContext())
        base_config = g_desc.config
    else:
        base_config = {}

    # 附加交互 schema
    interactions = {
        "click_expand": {
            "trigger": "click",
            "action": "expand_subgraph",
            "depth": 1,
            "max_nodes": 30,
        },
        "double_click_detail": {
            "trigger": "doubleClick",
            "action": "open_kp_detail",
            "target": "kp_detail_panel",
        },
        "context_menu": {
            "trigger": "rightClick",
            "actions": ["shortest_path", "toggle_highlight", "copy_kp_id"],
        },
        "drag": True,
        "zoom": {"enabled": True, "min_scale": 0.3, "max_scale": 3.0},
        "filter": {
            "by_domain": ["A", "B", "C", "D"],
            "by_mastery": ["mastered", "learning", "weak", "untracked"],
            "by_label": True,
        },
    }

    config = {
        **base_config,
        "type": "graph_explorer",
        "interactions": interactions,
    }
    html = dashboard_wrap(
        f'<div class="l7-graph-explorer" data-graph-id="explorer-{int(time.time())}" style="width:100%;height:480px"></div>',
        "l7-dashboard l7-explorer",
        theme,
    )
    descriptor = build_descriptor(
        artifact or Artifact(artifact_id="graph-explorer", payload={}),
        html=html,
        config=config,
        assets=(
            list(base_config.get("assets", []))
            if "assets" in base_config
            else ["https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"]
        ),
        metadata={"renderer": "GraphExplorer", "node_count": len(nodes or [])},
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor
