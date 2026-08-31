"""L7 CC2 审批 T5 — approval 子包单元测试.

测试覆盖:
1. 计划预览: 6 项内容、策略类型视觉标识、风险等级
2. 审批流程: 三操作、历史记录、原因输入框
3. 快速审批: 信任模式窗口、规则预设、安全拦截
4. 计划渲染: 知识图谱高亮、当前 vs 预期对比、警告
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.approval import (
    render_approval_flow,
    render_plan_preview,
    render_plan_rendering,
    render_quick_mode,
)
from dy3_polaris.l7.approval._common import APPROVAL_STATUS, normalize_plan
from dy3_polaris.l7.models import RenderDescriptor


class TestPlanPreview:
    """教学计划预览."""

    def _plan(self) -> dict:
        return {
            "plan_id": "plan-001",
            "title": "Dy3+ 能级跃迁教学",
            "strategy_type": "socratic",
            "summary": "通过提问引导理解 4f-4f 跃迁",
            "kp_ids": ["A-01", "A-05"],
            "agents": {"A1": "引导提问", "A2": "文献支撑"},
            "duration_minutes": 20,
            "prerequisites": ["A-01"],
            "expected_effect": {"A-05": {"from": 0.4, "to": 0.7}},
            "risk_level": "medium",
        }

    def test_render_preview(self):
        d = render_plan_preview(plan=self._plan())
        assert isinstance(d, RenderDescriptor)
        assert d.config["type"] == "plan_preview"
        assert "Dy3+ 能级跃迁教学" in d.html

    def test_kp_list_shown(self):
        d = render_plan_preview(plan=self._plan())
        assert "A-01" in d.html
        assert "A-05" in d.html

    def test_agent_assignments(self):
        d = render_plan_preview(plan=self._plan())
        assert "引导提问" in d.html

    def test_risk_level_badge(self):
        d = render_plan_preview(plan=self._plan())
        assert "中风险" in d.html
        high = {**self._plan(), "risk_level": "critical"}
        dh = render_plan_preview(plan=high)
        assert "关键" in dh.html

    def test_strategy_meta(self):
        d = render_plan_preview(plan=self._plan())
        assert d.config["strategy_meta"]["label"] == "苏格拉底对话"

    def test_normalize_plan_l0_compat(self):
        """兼容 L0 ApprovalRequest 字段."""
        plan = {
            "request_id": "apr-1", "target": "教学计划", "operation": "teach",
            "risk_level": "LOW", "requester": "A1",
            "context": {"kp_ids": ["A-01"], "agent_assignments": {"A1": "引导"}},
        }
        p = normalize_plan(plan)
        assert p["plan_id"] == "apr-1"
        assert p["kp_ids"] == ["A-01"]
        assert p["risk_level"] == "low"

    def test_empty_plan(self):
        d = render_plan_preview(plan={})
        assert "教学计划" in d.html


class TestApprovalFlow:
    """审批操作流程."""

    def test_three_actions(self):
        d = render_approval_flow(plan={"plan_id": "plan-001", "title": "计划"})
        assert d.config["type"] == "approval_flow"
        actions = d.config["actions"]
        assert [a["action"] for a in actions] == ["approve", "reject", "modify"]

    def test_approve_reject_modify_buttons(self):
        d = render_approval_flow(plan={"plan_id": "plan-001", "title": "计划"})
        assert "✅ 批准" in d.html
        assert "❌ 拒绝" in d.html
        assert "✏️ 修改" in d.html

    def test_reason_input(self):
        d = render_approval_flow(plan={"plan_id": "plan-001"})
        assert "appr-reason-input" in d.html

    def test_history_rendered(self):
        d = render_approval_flow(plan={"plan_id": "plan-001"}, history=[
            {"timestamp": 1000.0, "result": "approved", "summary": "批准", "comment": "同意"},
            {"timestamp": 2000.0, "result": "modified", "summary": "修改", "comment": "增加实验"},
        ])
        assert d.config["type"] == "approval_flow"
        assert len(d.config["history"]) == 2
        assert "已批准" in d.html
        assert "已修改" in d.html

    def test_empty_history(self):
        d = render_approval_flow(plan={"plan_id": "p"})
        assert "暂无历史记录" in d.html


class TestQuickMode:
    """快速审批模式."""

    def test_trust_mode_active(self):
        d = render_quick_mode(trust_mode={"active": True, "remaining_seconds": 900})
        assert d.config["trust_mode"]["active"] is True
        assert "信任模式已启用" in d.html
        assert "15:00" in d.html

    def test_trust_mode_inactive(self):
        d = render_quick_mode(trust_mode={"active": False})
        assert d.config["trust_mode"]["active"] is False
        assert "启用" in d.html

    def test_rule_presets(self):
        d = render_quick_mode(rule_presets=[{"id": "r1", "operation": "A域教学", "risk_level": "low"}])
        assert len(d.config["rule_presets"]) == 1
        assert "A域教学" in d.html

    def test_safety_operations_always_manual(self):
        from dy3_polaris.l7.approval.quick_mode import SAFETY_OPERATIONS

        assert "高温实验" in SAFETY_OPERATIONS
        assert "化学试剂使用" in SAFETY_OPERATIONS
        d = render_quick_mode()
        assert "高温实验" in d.html

    def test_pending_count(self):
        d = render_quick_mode(pending_count=5)
        assert d.config["pending_count"] == 5
        assert "<strong>5</strong>" in d.html


class TestPlanRendering:
    """教学计划渲染."""

    def _plan(self) -> dict:
        return {
            "plan_id": "plan-001",
            "strategy_type": "knowledge",
            "content": "知识讲解内容摘要",
            "kp_ids": ["A-01", "A-02"],
            "expected_effect": {"A-01": {"from": 0.3, "to": 0.6}, "A-02": {"from": 0.4, "to": 0.7}},
        }

    def test_render(self):
        d = render_plan_rendering(plan=self._plan())
        assert d.config["type"] == "plan_rendering"
        assert d.config["kp_count"] == 2
        assert "知识讲解内容摘要" in d.html

    def test_kp_map_highlight(self):
        d = render_plan_rendering(plan=self._plan())
        # 涉及 KP 带琥珀边框, 未涉及 KP 低透明度
        assert 'border:2px solid #d97706' in d.html
        assert "opacity:0.25" in d.html

    def test_compare_chart(self):
        d = render_plan_rendering(plan=self._plan())
        chart = d.config["compare_chart"]
        assert chart is not None
        assert len(chart["series"]) == 2  # 当前 vs 预期

    def test_weak_improvement_warning(self):
        plan = {**self._plan(), "expected_effect": {"A-01": {"from": 0.3, "to": 0.35}}}
        d = render_plan_rendering(plan=plan)
        assert "提升幅度较小" in d.html

    def test_strategy_label(self):
        d = render_plan_rendering(plan=self._plan())
        assert d.config["strategy_label"] == "📚 知识讲解 — 知识结构大纲"


class TestStatusMapping:
    """审批状态映射."""

    def test_all_statuses(self):
        for status in ("pending", "approved", "rejected", "modified", "timeout", "auto_approved", "executed"):
            assert status in APPROVAL_STATUS
            assert APPROVAL_STATUS[status]["label"]
