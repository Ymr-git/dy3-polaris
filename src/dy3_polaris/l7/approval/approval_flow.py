"""L7 CC2 审批 — 审批操作流程 (approval_flow.py).

任务拆分 T5 · 设计文档 Ch.7.1.2 + 时序图 7.3。

审批三操作闭环:
- 批准 (approve): 确认无误立即执行, 审批记录写入 L0 Provenance Ledger
- 拒绝 (reject): 否决当前计划, 回退 L4 重新决策, 拒绝原因作上下文反馈
- 修改 (modify): 自然语言修改建议, 经 L6 MCP 回传 L4, Meta-Decider 调整后重新提交

历史审批记录时间线 (Ch.7.1.3)。

融合世界先进方案:
- 审批工作流 (AI 建议 + 人工审批 + 审计记录): 每个操作生成不可变记录
- 表单交互: 拒绝原因/修改建议输入框
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import build_descriptor, esc
from ._common import APPROVAL_STATUS, build_approval_descriptor


def render_approval_flow(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    plan: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染审批操作流程 (Ch.7.1.2).

    Args:
        plan: 待审批计划 {plan_id, title, ...}.
        history: 历史审批记录 [{timestamp, summary, result, comment}].

    Returns:
        RenderDescriptor (含三操作 schema 供前端触发)。
    """
    started = time.monotonic()
    plan = plan or {}
    history = history or []

    plan_id = str(plan.get("plan_id") or plan.get("request_id") or "")
    plan_title = str(plan.get("title") or plan.get("target") or "教学计划")

    html_content = (
        f'<div class="appr-flow">'
        f'<div class="appr-flow-head"><h3>审批操作</h3>'
        f'<span class="appr-plan-ref">{esc(plan_id)} · {esc(plan_title)}</span></div>'
        f'<div class="appr-actions" data-plan-id="{esc(plan_id)}">'
        f'<button class="appr-btn appr-approve" data-action="approve">✅ 批准</button>'
        f'<button class="appr-btn appr-reject" data-action="reject">❌ 拒绝</button>'
        f'<button class="appr-btn appr-modify" data-action="modify">✏️ 修改</button>'
        f"</div>"
        f'<div class="appr-reason-box" hidden>'
        f'<label for="appr-reason-input">审批意见 / 修改建议:</label>'
        f'<textarea id="appr-reason-input" rows="2" placeholder="如: 跳过 A-01, 直接从 A-03 开始..."></textarea>'
        f'<button class="appr-btn appr-submit" data-action="submit">提交</button>'
        f"</div>"
        f'<div class="appr-history"><h4>📜 历史审批记录</h4>'
        f'<div class="appr-history-list">'
        + "".join(
            _history_item(h) for h in history
        )
        + ("</div></div></div>"
           if history
           else '<p class="appr-muted">暂无历史记录</p></div></div>')
    )

    config = {
        "type": "approval_flow",
        "plan_id": plan_id,
        "actions": [
            {"action": "approve", "label": "批准"},
            {"action": "reject", "label": "拒绝", "requires_reason": True},
            {"action": "modify", "label": "修改", "requires_suggestions": True},
        ],
        "history": history,
    }
    descriptor = build_approval_descriptor(
        artifact, html_content, config,
        [],
        {"renderer": "ApprovalFlow", "plan_id": plan_id, "history_count": len(history)},
        "ApprovalFlow",
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor


def _history_item(h: dict[str, Any]) -> str:
    result = str(h.get("result") or h.get("decision") or "pending").lower()
    meta = APPROVAL_STATUS.get(result, {"color": "#94a3b8", "label": result})
    ts = h.get("timestamp") or 0.0
    time_str = time.strftime("%m-%d %H:%M", time.localtime(ts)) if ts else "--"
    return (
        f'<div class="appr-history-item">'
        f'<span class="appr-history-time">{time_str}</span>'
        f'<span class="appr-history-result" style="color:{meta["color"]}">{meta["label"]}</span>'
        f'<span class="appr-history-summary">{esc(str(h.get("summary", "")))}</span>'
        f'<span class="appr-history-comment">{esc(str(h.get("comment", h.get("reason", ""))))}</span>'
        f"</div>"
    )
