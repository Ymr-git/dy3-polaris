"""L7 学情面板 — 面板交互模式 (interaction_modes.py).

任务拆分 T4 · 设计文档 §5.4。

提供学情面板的三类交互模式:

1. 下钻/上卷 (Drill-Down / Roll-Up):
   总体 → 域级 → 单 KP, 面包屑导航 + 300ms ease-out 过渡

2. 时间旅行 (Time Travel):
   时间轴滑块回溯历史学情快照, 全面板组件同步更新

3. 对比模式 (Comparison Mode):
   教师端多学习者并排, 总体掌握度/域级进度/薄弱点重叠度

输出:
   每种模式生成带交互状态标记的 HTML 容器 + 前端驱动配置。
   前端 JS 负责实际 DOM 切换, 本模块构建状态机配置骨架。
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import (
    DOMAIN_LABELS,
    KP_DOMAIN_IDS,
    KP_NAMES,
    build_descriptor,
    esc,
)
from ._common import dashboard_wrap


def render_drill_down(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    current_level: str = "overall",
    breadcrumbs: list[dict[str, str]] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染下钻/上卷交互骨架.

    Args:
        current_level: overall / domain-{code} / kp-{kp_id}。
        breadcrumbs: [{label, target_level}] 面包屑导航。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    breadcrumbs = breadcrumbs or [{"label": "学情总览", "target_level": "overall"}]

    crumbs_html = (
        '<nav class="drilldown-breadcrumbs" aria-label="面包屑导航">'
        + "".join(
            f'<a class="crumb{" active" if i == len(breadcrumbs) - 1 else ""}" '
            f'href="#" data-target="{esc(bc["target_level"])}">{esc(bc["label"])}</a>'
            + ('<span class="crumb-sep">›</span>' if i < len(breadcrumbs) - 1 else "")
            for i, bc in enumerate(breadcrumbs)
        )
        + "</nav>"
    )
    html_content = (
        f'<div class="interaction-mode drilldown" data-level="{esc(current_level)}">'
        + crumbs_html
        + '</div>'
    )
    config = {
        "type": "drill_down",
        "current_level": current_level,
        "breadcrumbs": breadcrumbs,
        "levels": {
            "overall": {"label": "学情总览", "parent": None},
            **{f"domain-{d}": {"label": DOMAIN_LABELS[d], "parent": "overall"} for d in "ABCD"},
        },
        "transition_ms": 300,
    }
    html = dashboard_wrap(html_content, "l7-dashboard l7-drilldown", theme)
    descriptor = build_descriptor(
        artifact or Artifact(artifact_id="drill-down", payload={}),
        html=html,
        config=config,
        assets=[],
        metadata={"renderer": "DrillDown", "level": current_level},
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor


def render_time_travel(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    snapshots: list[dict[str, Any]] | None = None,
    current_index: int = -1,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染时间旅行交互骨架.

    Args:
        snapshots: [{timestamp, label, bkt_matrix}] 历史快照列表。
        current_index: 当前快照索引 (-1 为最新)。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    snapshots = snapshots or []
    if current_index < 0 and snapshots:
        current_index = len(snapshots) - 1

    timeline_html = ""
    if snapshots:
        ticks = "".join(
            f'<button class="time-tick{" active" if i == current_index else ""}" '
            f'data-index="{i}" title="{esc(s.get("label", ""))}">{esc(s.get("label", "快照 "+str(i+1)))}</button>'
            for i, s in enumerate(snapshots)
        )
        timeline_html = f'<div class="time-travel-timeline">{ticks}</div>'

    current_label = snapshots[current_index].get("label", "最新") if snapshots and 0 <= current_index < len(snapshots) else "最新"
    html_content = (
        f'<div class="interaction-mode time-travel" data-index="{current_index}" data-total="{len(snapshots)}">'
        f'<div class="time-travel-head"><h4>⏱ 时间旅行</h4><span>当前: {esc(current_label)}</span></div>'
        + timeline_html
        + "</div>"
    )
    config = {
        "type": "time_travel",
        "snapshot_count": len(snapshots),
        "current_index": current_index,
        "snapshots": snapshots,
    }
    html = dashboard_wrap(html_content, "l7-dashboard l7-timetravel", theme)
    descriptor = build_descriptor(
        artifact or Artifact(artifact_id="time-travel", payload={}),
        html=html,
        config=config,
        assets=[],
        metadata={"renderer": "TimeTravel", "snapshots": len(snapshots)},
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor


def render_comparison(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    learners: list[dict[str, Any]] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染对比模式骨架 (教师端多学习者并排).

    Args:
        learners: [{id, label, avg_p_l, domain_scores}] 学习者列表。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    learners = learners or []

    columns_html = "".join(
        f'<div class="compare-column" data-learner="{esc(ln["id"])}">'
        f'<h4>{esc(ln.get("label", ln["id"]))}</h4>'
        f'<div class="compare-mastery">{(ln.get("avg_p_l", 0.0) * 100):.0f}%</div>'
        f'<div class="compare-domains">'
        + "".join(
            f'<div class="compare-domain domain-{d}">'
            f'<span>{esc(DOMAIN_LABELS.get(d, d))}</span>'
            f'<span>{((ln.get("domain_scores", {}).get(d, 0.0) or 0.0) * 100):.0f}%</span>'
            f"</div>"
            for d in "ABCD"
        )
        + "</div></div>"
        for ln in learners
    )

    html_content = (
        f'<div class="interaction-mode comparison">'
        f'<h4>👥 对比模式 ({len(learners)} 人)</h4>'
        f'<div class="compare-grid">{columns_html}</div>'
        f"</div>"
    )
    config = {
        "type": "comparison",
        "learner_count": len(learners),
        "learners": learners,
    }
    html = dashboard_wrap(html_content, "l7-dashboard l7-compare", theme)
    descriptor = build_descriptor(
        artifact or Artifact(artifact_id="comparison", payload={}),
        html=html,
        config=config,
        assets=[],
        metadata={"renderer": "Comparison", "learners": len(learners)},
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor
