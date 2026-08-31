"""L7 CC2 审批 — 公共数据层 (approval/_common.py).

任务拆分 T5 · 设计文档 Ch.7。

提供审批面板共享能力:

1. ApprovalRequest 归一化 — L0 ApprovalRequest 字段 → 计划预览
2. 审批状态映射 — PENDING/APPROVED/REJECTED/MODIFIED/TIMEOUT → 颜色/标签
3. 风险等级映射 — LOW/MEDIUM/HIGH/CRITICAL → 徽章
4. 统一样式与包装
"""

from __future__ import annotations

import time
from typing import Any

from ..renderers._common import build_descriptor as _build_desc
from ..renderers._common import esc as _esc
from ..models import Artifact, RenderDescriptor

#: 审批状态 → 颜色/标签
APPROVAL_STATUS: dict[str, dict[str, str]] = {
    "pending": {"color": "#f59e0b", "label": "待审批"},
    "approved": {"color": "#16a34a", "label": "已批准"},
    "rejected": {"color": "#ef4444", "label": "已拒绝"},
    "modified": {"color": "#22a5f7", "label": "已修改"},
    "timeout": {"color": "#94a3b8", "label": "已超时"},
    "auto_approved": {"color": "#16a34a", "label": "自动批准"},
    "executed": {"color": "#4b3fe3", "label": "已执行"},
}

#: 风险等级 → 徽章
RISK_LEVELS: dict[str, str] = {
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
    "critical": "⚠ 关键",
}


def normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """归一化 L0 ApprovalRequest → 教学计划预览.

    输入兼容: L0 ApprovalRequest (operation/target/risk_level/context/requester)
    与自定义计划 (plan_id/title/strategy/kps/agents/duration/prerequisites/expected).

    Returns:
        统一计划预览字典。
    """
    context = plan.get("context") or {}
    strategy = plan.get("strategy") or context.get("strategy") or {}
    return {
        "plan_id": plan.get("plan_id") or plan.get("request_id") or "",
        "title": plan.get("title") or plan.get("target") or "教学计划",
        "summary": str(plan.get("summary") or strategy.get("summary") or ""),
        "strategy_type": str(plan.get("strategy_type") or strategy.get("type") or "knowledge"),
        "core_goal": str(plan.get("core_goal") or strategy.get("goal") or ""),
        "kp_ids": list(plan.get("kp_ids") or context.get("kp_ids") or []),
        "agents": dict(plan.get("agents") or context.get("agent_assignments") or {}),
        "duration_minutes": plan.get("duration_minutes") or plan.get("estimated_duration_minutes") or 0,
        "prerequisites": list(plan.get("prerequisites") or []),
        "expected_effect": plan.get("expected_effect") or {},
        "risk_level": str(plan.get("risk_level") or "medium").lower(),
        "requester": plan.get("requester") or plan.get("agent_id") or "",
        "confidence": plan.get("confidence"),
        "status": str(plan.get("status") or "pending").lower(),
        "created_at": float(plan.get("created_at") or plan.get("timestamp") or time.time()),
    }


def approval_wrap(content: str, css_class: str = "l7-approval-panel", theme: str = "light") -> str:
    """审批面板统一包装."""
    from ..dashboard._common import dashboard_wrap

    return dashboard_wrap(content, css_class, theme)


def build_approval_descriptor(
    artifact: Artifact | None,
    html_content: str,
    config: dict[str, Any],
    assets: list[str] | None,
    metadata: dict[str, Any] | None,
    renderer_name: str,
) -> RenderDescriptor:
    """构建审批面板 RenderDescriptor (统一)."""
    theme = "light"
    if artifact is None:
        artifact = Artifact(artifact_id=f"appr-{int(time.time()*1000)%10**12:012d}", payload={})
    html = approval_wrap(html_content, "l7-approval-panel", theme)
    return _build_desc(
        artifact,
        html=html,
        config=config,
        assets=assets or ["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        metadata=metadata or {},
        renderer_name=renderer_name,
    )
