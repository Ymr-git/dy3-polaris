"""L7 溯源可视化 — 决策溯源 (decision_trace.py).

任务拆分 T5 · 设计文档 Ch.6.2。

横向流程图展示完整决策链路 (5 步):
1. 复杂度评估 — 六维雷达图 (知识深度/跨域广度/计算量/交互需求/创造需求/争议程度)
2. 范式选择 — 评分对比
3. Agent 调度 — 职责分配
4. 执行过程 — Artifact 序列
5. 裁决结果 — 三维评分

三级溯源深度: summary (仅关键节点) / standard (全量) / full (含内部推理)。
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import esc
from ._common import build_panel_descriptor, build_radar_option

#: 决策链路 5 步图标
_STEP_ICONS = ["🔍", "🧭", "🤖", "⚙️", "⚖️"]

#: summary 深度保留的关键步骤关键词
_SUMMARY_KEYWORDS = ("decision", "adjudication", "verdict", "裁决", "决策")


def _is_key_step(title: str) -> bool:
    """判断步骤是否为 summary 深度的关键节点 (包含匹配)."""
    lower = title.lower()
    return any(kw in lower for kw in _SUMMARY_KEYWORDS)


def render_decision_trace(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    steps: list[dict[str, Any]] | None = None,
    complexity: list[dict[str, Any]] | None = None,
    depth: str = "standard",
    full_access: bool = False,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染决策溯源 (Ch.6.2).

    Args:
        steps: 决策链路步骤 [{step/title, detail, score, agent, artifacts}].
        complexity: 六维复杂度 [{name, value, max}].
        depth: summary / standard / full.
        full_access: full 深度需授权才展示内部推理。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    steps = steps or []
    depth = depth if depth in ("summary", "standard", "full") else "standard"

    # 深度过滤 (summary 仅保留关键节点: 决策/裁决相关)
    visible_steps = []
    for s in steps:
        title = str(s.get("title") or s.get("step") or "")
        if depth == "summary" and not _is_key_step(title):
            continue
        visible_steps.append(s)

    # 六维复杂度雷达图
    radar_opt = None
    if complexity:
        radar_opt = build_radar_option(complexity, "六维复杂度评估", theme)

    # 步骤流程
    flow_html = ""
    for i, s in enumerate(visible_steps):
        icon = _STEP_ICONS[i % len(_STEP_ICONS)]
        title = esc(str(s.get("title") or s.get("step") or f"步骤 {i+1}"))
        detail = esc(str(s.get("detail") or s.get("description") or ""))
        agent = esc(str(s.get("agent") or ""))
        score = s.get("score")
        score_html = f'<span class="prov-score">{score}</span>' if score is not None else ""
        artifacts = s.get("artifacts") or []
        arts_html = (
            "<ul class='prov-artifacts'>" + "".join(f"<li>{esc(str(a))}</li>" for a in artifacts) + "</ul>"
            if artifacts
            else ""
        )
        # full 深度且未授权 → 隐藏内部推理
        internal = s.get("internal")
        internal_html = ""
        if internal:
            internal_html = (
                f'<div class="prov-internal">{esc(str(internal))}</div>'
                if full_access
                else '<div class="prov-masked">（内部推理，需完整模式授权）</div>'
            )
        flow_html += (
            f'<div class="prov-step" data-depth-index="{i}">'
            f'<div class="prov-step-head"><span class="prov-step-icon">{icon}</span>'
            f'<span class="prov-step-title">{title}</span>{score_html}</div>'
            f'<p class="prov-step-detail">{detail}</p>'
            f'{f"<div class=prov-step-agent>Agent: {agent}</div>" if agent else ""}'
            f"{arts_html}{internal_html}"
            f"</div>"
        )

    depth_label = {"summary": "摘要", "standard": "标准", "full": "完整"}[depth]
    html_content = (
        f'<div class="prov-decision-panel">'
        f'<div class="prov-header"><h3>决策溯源</h3>'
        f'<span class="prov-depth-badge">深度: {depth_label}</span></div>'
        f'{f"<div class=prov-chart id=complexity-radar></div>" if radar_opt else ""}'
        f'<div class="prov-flow">{flow_html or "<p class=prov-empty>暂无决策步骤</p>"}</div>'
        f"</div>"
    )

    config = {
        "type": "decision_trace",
        "depth": depth,
        "depth_label": depth_label,
        "step_count": len(visible_steps),
        "complexity_radar": radar_opt,
        "steps": visible_steps,
        "privacy": {"masked": not full_access, "full_access": full_access},
    }
    descriptor = build_panel_descriptor(
        artifact, html_content, config,
        ["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        {"renderer": "DecisionTrace", "steps": len(visible_steps), "depth": depth},
        "DecisionTrace",
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor
