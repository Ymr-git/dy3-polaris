"""L7 CC2 审批 — 教学计划预览面板 (plan_preview.py).

任务拆分 T5 · 设计文档 Ch.7.1.1。

结构化卡片展示 6 项内容:
1. 策略摘要 (策略类型 + 核心目标)
2. 涉及 KP 列表 (编号+标题, 可点击)
3. Agent 分配 (各 Agent 职责)
4. 预计时长
5. 前置条件
6. 预期学习效果 (BKT 预测)

融合世界先进方案:
- 渐进披露 (progressive disclosure): 概览卡片 + 可展开详情
- 结构化表单: 审批前的完整信息呈现
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import KP_NAMES, build_descriptor, esc
from ._common import RISK_LEVELS, build_approval_descriptor, normalize_plan

#: 策略类型 → 视觉标识 (设计文档 Ch.7.2.1)
_STRATEGY_META: dict[str, dict[str, str]] = {
    "socratic": {"icon": "💬", "label": "苏格拉底对话", "color": "#4b3fe3"},
    "knowledge": {"icon": "📚", "label": "知识讲解", "color": "#22a5f7"},
    "practice": {"icon": "✏️", "label": "练习测试", "color": "#10b981"},
    "experiment": {"icon": "🧪", "label": "虚拟实验", "color": "#f59e0b"},
}


def render_plan_preview(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    plan: dict[str, Any] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染教学计划预览面板 (Ch.7.1.1).

    Args:
        plan: L0 ApprovalRequest 或自定义教学计划字典。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    p = normalize_plan(plan or {})

    strategy_type = p["strategy_type"].lower()
    smeta = _STRATEGY_META.get(strategy_type, {"icon": "📋", "label": strategy_type, "color": "#94a3b8"})
    risk_label = RISK_LEVELS.get(p["risk_level"], p["risk_level"])
    status_color = "#16a34a" if p["status"] == "approved" else "#f59e0b"

    # KP 列表
    kps_html = "".join(
        f'<span class="appr-kp" data-kp="{esc(kp)}">{esc(kp)}</span>'
        f'<span class="appr-kp-name">{esc(KP_NAMES.get(kp, ""))}</span>'
        for kp in p["kp_ids"]
    )
    if not p["kp_ids"]:
        kps_html = '<span class="appr-muted">未指定</span>'

    # Agent 分配
    agents = p["agents"] or {}
    agents_html = "".join(
        f'<li><strong>{esc(aid)}</strong>: {esc(str(role))}</li>' for aid, role in agents.items()
    ) or '<li class="appr-muted">未指定</li>'

    # 预期效果
    expected = p["expected_effect"] or {}
    expected_html = "".join(
        f'<li><strong>{esc(kp)}</strong>: P(L) {expected[kp].get("from", "?")} → '
        f'{expected[kp].get("to", "?")}</li>'
        for kp in expected
    ) or '<li class="appr-muted">无预测</li>'

    html_content = (
        f'<div class="appr-preview">'
        f'<div class="appr-preview-head">'
        f'<span class="appr-strategy-icon" style="color:{smeta["color"]}">{smeta["icon"]}</span>'
        f'<div><h3>{esc(p["title"])}</h3>'
        f'<span class="appr-strategy-label" style="background:{smeta["color"]}22;color:{smeta["color"]}">{smeta["label"]}</span> '
        f'<span class="appr-risk" style="background:{"#ef4444" if p["risk_level"] in ("high","critical") else "#f59e0b"}22;'
        f'color:{"#ef4444" if p["risk_level"] in ("high","critical") else "#f59e0b"}">{risk_label}</span> '
        f'<span class="appr-status" style="color:{status_color}">{p["status"]}</span></div>'
        f"</div>"
        f'<div class="appr-summary"><p>{esc(p["summary"]) or "暂无策略摘要"}</p></div>'
        f'<div class="appr-grid">'
        f'<div class="appr-section"><h4>🎯 核心目标</h4><p>{esc(p["core_goal"]) or "—"}</p></div>'
        f'<div class="appr-section"><h4>📌 涉及知识点 ({len(p["kp_ids"])})</h4><div class="appr-kps">{kps_html}</div></div>'
        f'<div class="appr-section"><h4>🤖 Agent 分配</h4><ul>{agents_html}</ul></div>'
        f'<div class="appr-section"><h4>⏱ 预计时长</h4><p>{p["duration_minutes"]} 分钟</p></div>'
        f'<div class="appr-section"><h4>🔗 前置条件</h4><p>{esc(", ".join(p["prerequisites"]) or "无")}</p></div>'
        f'<div class="appr-section"><h4>📈 预期学习效果</h4><ul>{expected_html}</ul></div>'
        f"</div></div>"
    )

    config = {
        "type": "plan_preview",
        "plan": p,
        "strategy_meta": smeta,
    }
    descriptor = build_approval_descriptor(
        artifact, html_content, config,
        [],
        {"renderer": "PlanPreview", "plan_id": p["plan_id"], "kp_count": len(p["kp_ids"])},
        "PlanPreview",
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor
