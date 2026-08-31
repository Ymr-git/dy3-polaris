"""L7 CC2 审批 — 教学计划渲染 (plan_rendering.py).

任务拆分 T5 · 设计文档 Ch.7.2。

教学计划渲染三组件:
1. 策略文本渲染 (TextRenderer 语义, Socratic/讲解/练习/实验视觉标识)
2. 涉及知识图谱高亮 (涉及 KP 琥珀色边框+放大, 未涉及低透明度)
3. 预期学习效果预估 (当前 vs 预期对比柱状图)

融合世界先进方案:
- 渐进披露: 策略大纲 + 可展开
- 数据对比: 当前 vs 预期柱状图 (视觉差异引导判断)
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
    build_descriptor,
    esc,
)
from ._common import build_approval_descriptor


def render_plan_rendering(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    plan: dict[str, Any] | None = None,
    current_mastery: dict[str, float] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染教学计划 (Ch.7.2).

    Args:
        plan: 教学计划 {plan_id, strategy_type, content, kp_ids,
              expected_effect: {kp: {from, to}}}。
        current_mastery: 当前掌握度 {kp: p_l} (用于高亮着色)。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    plan = plan or {}
    current_mastery = current_mastery or {}
    kp_ids = list(plan.get("kp_ids") or [])
    expected = plan.get("expected_effect") or {}
    strategy_type = str(plan.get("strategy_type") or "knowledge").lower()

    # 1. 策略文本渲染 (TextRenderer 语义摘要)
    strategy_labels = {
        "socratic": "💬 苏格拉底对话 — 引导问题序列",
        "knowledge": "📚 知识讲解 — 知识结构大纲",
        "practice": "✏️ 练习测试 — 题目类型与难度分布",
        "experiment": "🧪 虚拟实验 — 实验流程与预期现象",
    }
    strategy_label = strategy_labels.get(strategy_type, f"📋 {strategy_type}")
    content = str(plan.get("content") or plan.get("summary") or "暂无策略文本")

    # 2. 知识图谱高亮 (42 KP 网格, 涉及 KP 琥珀高亮)
    kp_cells = []
    for domain in ("A", "B", "C", "D"):
        for kp in KP_DOMAIN_IDS[domain]:
            involved = kp in kp_ids
            p_l = current_mastery.get(kp)
            style = ""
            if involved:
                style = ' style="border:2px solid #d97706;opacity:1;transform:scale(1.08)"'
            elif p_l is None:
                style = ' style="opacity:0.25"'
            else:
                style = f' style="opacity:{0.5 + p_l * 0.4:.2f}"'
            badge = f'<span class="appr-kp-cell{"" if involved else " dim"}"{style} title="{esc(KP_NAMES.get(kp, kp))}">'
            badge += f'{esc(kp)}{"🔶" if involved else ""}</span>'
            kp_cells.append(badge)
    kp_map_html = '<div class="appr-kp-map">' + "".join(kp_cells) + "</div>"

    # 3. 当前 vs 预期对比柱状图
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    compare_chart = None
    if expected:
        labels = [str(k) for k in expected]
        current_vals = [float(expected[k].get("from", current_mastery.get(k, 0.0))) for k in expected]
        expected_vals = [float(expected[k].get("to", 0.0)) for k in expected]
        compare_chart = {
            "title": {"text": "预期学习效果 (当前 vs 预期)", "textStyle": {"color": tc, "fontSize": 13}},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["当前 P(L)", "预期 P(L)"], "textStyle": {"color": tc}},
            "xAxis": {"type": "category", "data": labels, "axisLabel": {"color": tc}},
            "yAxis": {"type": "value", "min": 0, "max": 1, "axisLabel": {"color": tc}},
            "series": [
                {"name": "当前 P(L)", "type": "bar", "data": current_vals, "itemStyle": {"color": "#94a3b8"}},
                {"name": "预期 P(L)", "type": "bar", "data": expected_vals, "itemStyle": {"color": "#16a34a"}},
            ],
        }
        # 预期不理想警告
        weak_improvements = [
            k for k in expected
            if float(expected[k].get("to", 0.0)) - float(expected[k].get("from", 0.0)) < 0.1
        ]
        warning_html = (
            f'<div class="appr-warning">⚠️ 以下 KP 预期提升幅度较小: {esc(", ".join(weak_improvements))}</div>'
            if weak_improvements
            else ""
        )
    else:
        warning_html = ""

    html_content = (
        f'<div class="appr-render">'
        f'<div class="appr-render-head"><h3>教学计划详情</h3>'
        f'<span class="appr-strategy-label">{esc(strategy_label)}</span></div>'
        f'<div class="appr-content"><p>{esc(content)}</p></div>'
        f'<h4>🗺 知识覆盖 ({len(kp_ids)}/42)</h4>'
        + kp_map_html
        + (
            f'<div class="l7-chart" data-chart-id="plan-compare" style="width:100%;height:220px"></div>'
            if compare_chart
            else ""
        )
        + warning_html
        + "</div>"
    )

    config = {
        "type": "plan_rendering",
        "strategy_type": strategy_type,
        "strategy_label": strategy_label,
        "kp_ids": kp_ids,
        "kp_count": len(kp_ids),
        "compare_chart": compare_chart,
        "expected_effect": expected,
    }
    descriptor = build_approval_descriptor(
        artifact, html_content, config,
        ["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        {"renderer": "PlanRendering", "plan_id": plan.get("plan_id", ""), "kp_count": len(kp_ids)},
        "PlanRendering",
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor
