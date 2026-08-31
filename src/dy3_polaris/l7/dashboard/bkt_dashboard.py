"""L7 学情面板 — BKT Dashboard (bkt_dashboard.py).

任务拆分 T4 · 设计文档 §5.1。

输出 BKT 掌握度可视化的 ECharts 配置与 HTML 容器:

1. 42 KP 热力图 (rows=KP, cols=P(L)/P(K|L)/P(G)/P(S), 四域分组)
2. 单 KP 详情面板 (四参数条形图 + 学习轨迹时间线)
3. 瓶颈 KP 高亮 (P(L)>0.7 且 P(K|L)<0.3 红色脉冲)
4. 知识拓扑图 (节点大小=被依赖度, 边粗细=依赖强度)

输出契约:
    RenderDescriptor.html   — 挂载壳 (l7-dashboard 容器)
    RenderDescriptor.config — {heatmap, detail, bottlenecks, topology}
    RenderDescriptor.assets — [echarts.min.js]
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import (
    ALL_KP_IDS,
    KP_DOMAIN_IDS,
    KP_NAMES,
    KP_TO_DOMAIN,
    average_p_l,
    build_descriptor,
    esc,
    get_bkt_state,
)
from ._common import (
    bottlenecks,
    build_heatmap_option,
    build_kp_detail_option,
    colorblind_from,
    dashboard_wrap,
    extract_bkt_matrix,
)


def render_bkt_dashboard(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    history: dict[str, list[dict[str, Any]]] | None = None,
    dependencies: dict[str, int] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染 BKT 学情面板 (完整).

    Args:
        artifact: 关联 Artifact (携带 learner_context)。
        context: 渲染上下文 (携带 bkt_state)。
        history: KP 学习历史 {kp_id: [{step, p_l, p_k_l, p_g, p_s}]}。
        dependencies: KP→依赖数 (用于拓扑图节点大小)。
        theme: light / dark。

    Returns:
        RenderDescriptor (html + config)。
    """
    started = time.monotonic()
    matrix = extract_bkt_matrix(artifact, context)
    bn_list = bottlenecks(matrix)
    avg = average_p_l(get_bkt_state(artifact, context))
    colorblind = colorblind_from(artifact, context)

    # 1. 热力图
    heatmap_opt = build_heatmap_option(matrix, theme=theme, colorblind=colorblind)

    # 2. 瓶颈 KP 列表
    bottleneck_kps = [
        {"id": b["kp_id"], "name": b["name"], "p_l": b["p_l"], "p_k_l": b["p_k_l"]}
        for b in bn_list[:8]
    ]

    # 3. 知识拓扑图 (vis.js 简化版: nodes+edges)
    topology_nodes, topology_edges = _build_topology(dependencies, matrix, theme)

    # 4. HTML 容器
    html_content = _build_html(matrix, avg, bottleneck_kps, theme)

    config = {
        "type": "bkt_dashboard",
        "heatmap": heatmap_opt,
        "bottlenecks": bottleneck_kps,
        "bottleneck_count": len(bn_list),
        "topology": {"nodes": topology_nodes, "edges": topology_edges},
        "summary": {
            "total_kps": len(ALL_KP_IDS),
            "avg_p_l": round(avg, 4),
            "tracked": sum(1 for s in matrix.values() if s.get("p_l", 0.0) > 0),
        },
    }
    html = dashboard_wrap(html_content, "l7-dashboard l7-bkt", theme)
    descriptor = build_descriptor(
        artifact or Artifact(artifact_id="bkt-dashboard", payload={}),
        html=html,
        config=config,
        assets=["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        metadata={"renderer": "BKTDashboard", "tracked": config["summary"]["tracked"], "bottlenecks": len(bn_list)},
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor


def render_kp_detail(
    kp_id: str,
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    history: list[dict[str, Any]] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染单 KP 详情面板 (设计文档 §5.1.2).

    Args:
        kp_id: 目标 KP ID。
        artifact/context: BKT 状态来源。
        history: 学习轨迹历史 [{step, p_l, p_k_l, p_g, p_s}]。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    matrix = extract_bkt_matrix(artifact, context, kp_ids=[kp_id])
    state = matrix.get(kp_id, {})
    charts = build_kp_detail_option(kp_id, state, history, theme)
    name = KP_NAMES.get(kp_id, kp_id)
    domain = KP_TO_DOMAIN.get(kp_id, "")

    html_content = (
        f'<div class="kp-detail" data-kp="{esc(kp_id)}">'
        f'<h3>{esc(kp_id)} — {esc(name)}</h3>'
        f'<div class="kp-charts">{_chart_div("kp-params-chart")}{_chart_div("kp-trajectory-chart")}</div>'
        f"</div>"
    )

    config = {
        "type": "kp_detail",
        "kp_id": kp_id,
        "name": name,
        "domain": domain,
        "bkt_state": {k: round(v, 4) for k, v in state.items()},
        "charts": charts,
        "is_bottleneck": state.get("p_l", 0.0) > 0.7 and state.get("p_k_l", 1.0) > 0 and state.get("p_k_l", 0.0) < 0.3,
    }
    html = dashboard_wrap(html_content, "l7-dashboard l7-kp-detail", theme)
    descriptor = build_descriptor(
        artifact or Artifact(artifact_id=f"kp-detail-{kp_id}", payload={}),
        html=html,
        config=config,
        assets=["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        metadata={"renderer": "KPDetailPanel", "kp_id": kp_id},
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor


def _build_topology(
    dependencies: dict[str, int] | None,
    matrix: dict[str, dict[str, float]],
    theme: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构建知识拓扑图 (vis.js 格式)."""
    deps = dependencies or {}
    nodes = []
    for kp_id in ALL_KP_IDS:
        state = matrix.get(kp_id, {})
        p_l = state.get("p_l", 0.0)
        size = 10 + min(deps.get(kp_id, 0), 24)
        color = _node_color(p_l)
        nodes.append({
            "id": kp_id,
            "label": kp_id,
            "title": f"{kp_id}: {KP_NAMES.get(kp_id, '')} · P(L)={p_l:.2f}",
            "color": color,
            "size": size,
            "borderWidth": 3 if p_l < 0.5 else 1.5,
        })
    edges = []
    for child, parents in (_default_prerequisites()).items():
        for parent in parents:
            edges.append({"from": parent, "to": child, "arrows": "to", "label": "前置"})
    return nodes, edges


def _default_prerequisites() -> dict[str, list[str]]:
    """默认前置依赖关系 (A 域线性, B/C/D 域无前置)."""
    prereqs: dict[str, list[str]] = {}
    for domain in ("A", "B", "C", "D"):
        ids = KP_DOMAIN_IDS[domain]
        prereqs.setdefault("A-01", [])  # 根节点
        for i in range(1, len(ids)):
            prereqs.setdefault(ids[i], [ids[i - 1]])
    return prereqs


def _node_color(p_l: float) -> dict[str, str]:
    colors = {
        "mastered": {"background": "#16a34a", "border": "#16a34a"},
        "learning": {"background": "#4b3fe3", "border": "#4b3fe3"},
        "weak": {"background": "#d97706", "border": "#d97706"},
    }
    if p_l > 0.8:
        return colors["mastered"]
    if p_l >= 0.5:
        return colors["learning"]
    return colors["weak"]


def _chart_div(chart_id: str, height: int = 240) -> str:
    return f'<div class="l7-chart" data-chart-id="{chart_id}" style="width:100%;height:{height}px"></div>'


def _build_html(
    matrix: dict[str, dict[str, float]],
    avg: float,
    bottleneck_list: list[dict[str, Any]],
    theme: str,
) -> str:
    """构建面板 HTML 容器."""
    parts = []
    pct = f"{(avg * 100):.0f}%"
    parts.append(
        f'<div class="bkt-header"><h2>BKT 学情总览</h2>'
        f'<span class="bkt-avg-mastery">总体掌握度: <strong>{pct}</strong></span></div>'
    )
    parts.append(_chart_div("bkt-heatmap", 520))
    if bottleneck_list:
        items = "".join(
            f'<li class="bottleneck-item"><span class="kp-badge bottleneck">{esc(bn["id"])}</span> '
            f'{esc(bn["name"])} <small>P(L)={bn["p_l"]:.2f} P(K|L)={bn["p_k_l"]:.2f}</small></li>'
            for bn in bottleneck_list[:6]
        )
        parts.append(
            f'<div class="bkt-bottlenecks"><h4>⚠ 瓶颈 KP (虚假掌握)</h4><ul>{items}</ul></div>'
        )
    parts.append(_chart_div("bkt-topology", 400))
    return "\n".join(parts)
