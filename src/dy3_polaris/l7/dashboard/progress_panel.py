"""L7 学情面板 — 学习进度面板 (progress_panel.py).

任务拆分 T4 · 设计文档 §5.2。

输出学习进度可视化的 ECharts 配置与 HTML 容器:

1. 总体掌握度 (加权平均 + 环形进度条 + 7 天趋势)
2. 域级进度卡片 (A/B/C/D 四域平均掌握度 + 数量分布)
3. 薄弱点列表 (综合排序: P(L)+被依赖度+距离上次学习+瓶颈系数)
4. 学习路径推荐 (前置条件过滤 + BKT 加权 + DAG 展示)
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import (
    DOMAIN_LABELS,
    KP_DOMAIN_IDS,
    KP_NAMES,
    KP_TO_DOMAIN,
    build_descriptor,
    esc,
    get_bkt_state,
)
from ._common import (
    build_progress_ring_option,
    dashboard_wrap,
    domain_aggregates,
    extract_bkt_matrix,
    learning_path,
    weak_points,
)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def render_progress_panel(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    dependencies: dict[str, int] | None = None,
    prerequisites: dict[str, list[str]] | None = None,
    last_times: dict[str, float] | None = None,
    trend_7d: list[float] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染学习进度面板 (完整).

    Args:
        artifact/context: BKT 状态来源。
        dependencies: KP→被依赖数 (节点权重)。
        prerequisites: KP→前置 KP 列表 (学习路径过滤)。
        last_times: KP→最后学习时间戳 (薄弱点排序)。
        trend_7d: 最近 7 天平均 P(L) 走势 [day1, ..., day7]。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    matrix = extract_bkt_matrix(artifact, context)
    bkt_state = get_bkt_state(artifact, context)
    domains = domain_aggregates(matrix)
    weak_list = weak_points(matrix, dependencies, last_times, top_n=12)
    path = learning_path(matrix, prerequisites, dependencies)

    # 总体掌握度: 加权平均 (被依赖度越高的 KP 权重越大)
    deps = dependencies or {}
    weighted_sum = 0.0
    weight_total = 0.0
    for kp_id, state in matrix.items():
        p_l = state.get("p_l", 0.0)
        w = 1.0 + min(deps.get(kp_id, 0), 10) * 0.3
        weighted_sum += p_l * w
        weight_total += w
    avg = round(weighted_sum / weight_total, 4) if weight_total > 0 else 0.0

    ring = build_progress_ring_option(avg, domains, theme)

    # 趋势
    if trend_7d:
        tc = "#e5e5e5" if theme == "dark" else "#171717"
        trend_opt = {
            "xAxis": {"type": "category", "data": [f"d{i}" for i in range(len(trend_7d))], "axisLabel": {"color": tc}},
            "yAxis": {"type": "value", "min": 0, "max": 1, "axisLabel": {"color": tc}},
            "series": [{
                "type": "line", "smooth": True, "symbolSize": 4,
                "data": trend_7d,
                "areaStyle": {"opacity": 0.06, "color": "#4b3fe3"},
            }],
        }
    else:
        trend_opt = None

    html_content = _build_html(avg, domains, weak_list, path, theme)
    config = {
        "type": "progress_panel",
        "average_mastery": avg,
        "trend_7d": trend_7d or [],
        "trend_chart": trend_opt,
        "ring_chart": ring["ring_chart"],
        "domain_cards": {
            d: {
                "label": v["label"], "kp_count": v["kp_count"], "avg_p_l": v["avg_p_l"],
                "mastered": v["mastered"], "learning": v["learning"], "weak": v["weak"],
                "bottlenecks": v["bottlenecks"],
            }
            for d, v in domains.items()
        },
        "weak_points": weak_list,
        "learning_path": path,
    }
    html = dashboard_wrap(html_content, "l7-dashboard l7-progress", theme)
    descriptor = build_descriptor(
        artifact or Artifact(artifact_id="progress-panel", payload={}),
        html=html,
        config=config,
        assets=["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        metadata={"renderer": "ProgressPanel", "avg_mastery": round(avg, 4), "weak_count": len(weak_list)},
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor


def _build_html(
    avg: float,
    domains: dict[str, dict[str, Any]],
    weak_list: list[dict[str, Any]],
    path: list[dict[str, Any]],
    theme: str,
) -> str:
    parts = []
    pct = f"{(avg * 100):.1f}%"
    domain_cards_html = "".join(
        f'<div class="domain-card domain-{d}">'
        f'<span class="domain-label">{v["label"]}</span>'
        f'<span class="domain-avg">{(v["avg_p_l"] * 100):.0f}%</span>'
        f'<span class="domain-counts">已掌握 {v["mastered"]} · 学习中 {v["learning"]} · 待加强 {v["weak"]}</span>'
        f"</div>"
        for d, v in domains.items()
    )
    parts.append(
        f'<div class="progress-header"><h2>学习进度</h2>'
        f'<span class="mastery-ring-container">{pct}</span></div>'
    )
    parts.append(f'<div class="domain-cards">{domain_cards_html}</div>')
    if weak_list:
        items = "".join(
            f'<li class="weak-item" data-kp="{esc(w["kp_id"])}">'
            f'<span class="weak-rank">{i + 1}</span>'
            f'<span class="small-badge">{esc(w["kp_id"])}</span> '
            f'{esc(w["name"])} '
            f'<small>P(L)={w["p_l"]:.2f} · 被依赖 {w["dependents"]}次 · 距上次 {w["days_since_last"]:.0f}天 · 紧急度 {w["score"]:.2f}</small>'
            f"</li>"
            for i, w in enumerate(weak_list[:10])
        )
        parts.append(f'<div class="weak-points"><h4>📍 薄弱点 ({len(weak_list)} 个需关注)</h4><ol>{items}</ol></div>')
    if path:
        items = "".join(
            f'<li class="path-item{" path-next" if p.get("next_to_learn") else ""}">'
            f'{esc(p["kp_id"])} {esc(p.get("name", ""))} · P(L)={p["p_l"]:.2f} · 依赖 {p["dependents"]}</li>'
            for p in path[:8]
        )
        parts.append(f'<div class="learning-path"><h4>🎯 推荐学习路径</h4><ol>{items}</ol></div>')
    return "\n".join(parts)
