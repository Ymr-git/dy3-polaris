"""L7 学情面板 — 辩论可视化 (debate_viz.py).

任务拆分 T4 · 设计文档 §5.3。

输出辩论过程的实时可视化 ECharts 配置与 HTML 容器:

1. 辩论时间线 (Agent 发言序列 + 立场标记 + 自动滚动)
2. 收敛过程图 (共识度折线, 波动触发 L4 防内耗)
3. 裁决结果展示 (三维评分雷达图 + 裁决摘要)
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import build_descriptor, esc
from ._common import (
    build_convergence_option,
    build_verdict_radar_option,
    dashboard_wrap,
)

#: 立场 → 颜色映射
_STANCE_COLORS: dict[str, str] = {"support": "#16a34a", "oppose": "#ef4444", "neutral": "#94a3b8"}


def render_debate(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    speeches: list[dict[str, Any]] | None = None,
    convergence: dict[str, Any] | None = None,
    verdict: dict[str, Any] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染辩论可视化 (完整).

    Args:
        speeches: [{agent, stance, summary, timestamp, full_text}] 发言序列。
        convergence: {rounds: [int], consensus: [float]} 共识度数据。
        verdict: {summary, dimensions: [{name, value, max}], selected_agent}。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    speeches = speeches or []
    convergence = convergence or {}
    verdict = verdict or {}

    # 时间线 HTML
    timeline_html = _build_timeline(speeches, theme)
    conv_opt = None
    if convergence.get("rounds"):
        conv_opt = build_convergence_option(convergence["rounds"], convergence["consensus"], theme)
    verdict_opt = None
    if verdict.get("dimensions"):
        verdict_opt = build_verdict_radar_option(verdict["dimensions"], theme)

    html_content = (
        '<div class="debate-viz">'
        + timeline_html
        + (f'<div class="l7-chart" data-chart-id="debate-convergence" style="width:100%;height:240px"></div>' if conv_opt else "")
        + (f'<div class="l7-chart" data-chart-id="debate-verdict" style="width:100%;height:280px"></div>' if verdict_opt else "")
        + (
            f'<div class="verdict-summary"><h4>裁决结果</h4>'
            f'<p>{esc(verdict.get("summary", ""))}</p>'
            f'<p>采纳 Agent: <strong>{esc(verdict.get("selected_agent", ""))}</strong></p></div>'
            if verdict else ""
        )
        + "</div>"
    )

    config = {
        "type": "debate_viz",
        "speech_count": len(speeches),
        "timeline": speeches,
        "convergence": conv_opt,
        "verdict": verdict_opt,
        "verdict_summary": verdict,
    }
    html = dashboard_wrap(html_content, "l7-dashboard l7-debate", theme)
    descriptor = build_descriptor(
        artifact or Artifact(artifact_id="debate-viz", payload={}),
        html=html,
        config=config,
        assets=["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        metadata={"renderer": "DebateViz", "speeches": len(speeches)},
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor


def _build_timeline(speeches: list[dict[str, Any]], theme: str) -> str:
    parts = ['<div class="debate-timeline" role="log" aria-live="polite">']
    for s in speeches:
        agent = str(s.get("agent", "?"))
        stance = str(s.get("stance", "neutral")).lower()
        summary = str(s.get("summary", ""))
        color = _STANCE_COLORS.get(stance, "#94a3b8")
        stance_label = {"support": "✅ 支持", "oppose": "❌ 反对", "neutral": "⚪ 中立"}.get(stance, stance)
        parts.append(
            f'<div class="debate-speech" data-stance="{stance}">'
            f'<span class="debate-agent" style="color:{color}">{esc(agent)}</span>'
            f'<span class="debate-stance" style="color:{color}">{stance_label}</span>'
            f'<p class="debate-summary">{esc(summary)}</p>'
            f"</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)
