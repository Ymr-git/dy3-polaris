"""L7 CC2 审批 — 快速审批模式 (quick_mode.py).

任务拆分 T5 · 设计文档 Ch.7.1.4。

快速审批三要素:
1. 信任模式: 30 分钟自动通过窗口 (TrustModeWindow)
2. 规则预设: 域/KP 类型自动批准规则
3. 紧急拦截: 安全操作始终手动审批 (SAFETY_OPERATIONS)

融合 L0 CC2 ApprovalWorkflowManager 语义:
- TrustModeWindow (duration_seconds=1800)
- _check_trust_mode: 仅 LOW 风险 + REVERSIBLE + 非安全操作自动批准
- _check_rule_preset: 规则预设自动批准
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import build_descriptor, esc
from ._common import build_approval_descriptor

#: 安全操作集合 (始终手动审批, 设计文档 Ch.7.1.4)
SAFETY_OPERATIONS: set[str] = {
    "data_delete", "data_overwrite", "prompt_template_modify",
    "policy_change", "user_data_export",
    "高温实验", "化学试剂使用", "安全操作",
}


def render_quick_mode(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    trust_mode: dict[str, Any] | None = None,
    rule_presets: list[dict[str, Any]] | None = None,
    pending_count: int = 0,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染快速审批模式面板 (Ch.7.1.4).

    Args:
        trust_mode: {active, remaining_seconds, duration_seconds} 信任模式窗口。
        rule_presets: [{id, operation, risk_level, action}] 规则预设。
        pending_count: 待审批计划数。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    trust_mode = trust_mode or {}
    rule_presets = rule_presets or []
    active = bool(trust_mode.get("active"))
    remaining = float(trust_mode.get("remaining_seconds", 0.0))

    # 信任模式状态卡
    if active:
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        trust_html = (
            f'<div class="appr-trust active">'
            f'<span class="appr-trust-icon">🤝</span>'
            f'<div><strong>信任模式已启用</strong>'
            f'<p>剩余 {mins}:{secs:02d}，期间低风险操作自动批准</p></div>'
            f'<button class="appr-btn appr-deactivate" data-action="deactivate">停用</button>'
            f"</div>"
        )
    else:
        trust_html = (
            f'<div class="appr-trust">'
            f'<span class="appr-trust-icon">⏱</span>'
            f'<div><strong>信任模式</strong>'
            f'<p>信任系统决策 30 分钟，低风险操作自动通过</p></div>'
            f'<button class="appr-btn appr-activate" data-action="activate">启用</button>'
            f"</div>"
        )

    # 规则预设列表
    presets_html = "".join(
        f'<li class="appr-rule">'
        f'<span class="appr-rule-op">{esc(str(r.get("operation", "")))}</span>'
        f'<span class="appr-rule-risk">{esc(str(r.get("risk_level", "")))}</span>'
        f'<span class="appr-rule-action" style="color:#16a34a">{esc(str(r.get("action", "auto_approve")))}</span>'
        f'<button class="appr-btn appr-rule-remove" data-rule-id="{esc(str(r.get("id", "")))}">✕</button>'
        f"</li>"
        for r in rule_presets
    ) or '<li class="appr-muted">暂无规则预设</li>'

    # 安全拦截说明
    safety_html = (
        f'<div class="appr-safety">'
        f'<span class="appr-safety-icon">🛡</span>'
        f'<div><strong>安全拦截</strong>'
        f'<p>以下操作始终手动审批: {"、".join(sorted(SAFETY_OPERATIONS))}</p></div>'
        f"</div>"
    )

    html_content = (
        f'<div class="appr-quick">'
        f'<div class="appr-flow-head"><h3>快速审批</h3>'
        f'<span class="appr-pending">待审批: <strong>{pending_count}</strong></span></div>'
        + trust_html
        + f'<div class="appr-rules"><h4>📋 规则预设 ({len(rule_presets)})</h4><ul>{presets_html}</ul></div>'
        + safety_html
        + "</div>"
    )

    config = {
        "type": "quick_mode",
        "trust_mode": trust_mode,
        "rule_presets": rule_presets,
        "safety_operations": sorted(SAFETY_OPERATIONS),
        "pending_count": pending_count,
    }
    descriptor = build_approval_descriptor(
        artifact, html_content, config,
        [],
        {"renderer": "QuickMode", "trust_active": active, "presets": len(rule_presets)},
        "QuickMode",
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor
