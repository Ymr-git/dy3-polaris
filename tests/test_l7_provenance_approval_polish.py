"""L7 溯源审批 T5 — 深化完善轮专项测试.

覆盖:
1. L0 LedgerEvent 真实结构对接 (event_type 枚举/trace_id/layer/payload)
2. HTML 转义安全 (XSS 防护: 恶意输入不注入)
3. L0 ApprovalRequest 归一化兼容
4. 边界: 空数据/缺失字段/非法输入不崩溃
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.approval import (
    render_approval_flow,
    render_plan_preview,
    render_quick_mode,
)
from dy3_polaris.l7.approval._common import normalize_plan
from dy3_polaris.l7.provenance import (
    render_agent_contribution,
    render_branch_merge,
    render_decision_trace,
    render_timeline,
)
from dy3_polaris.l7.provenance._common import normalize_events, verify_hash_chain


class TestL0LedgerEventCompat:
    """L0 LedgerEvent 真实结构对接."""

    def _l0_events(self) -> list[dict]:
        """模拟 L0 LedgerEvent 完整字段."""
        return [
            {
                "event_id": "evt-1", "event_type": "knowledge",
                "trace_id": "trace-001", "session_id": "sess-1",
                "agent_id": "A1", "layer": "L3",
                "timestamp": 1000.0,
                "payload": {"kp_id": "A-01", "summary": "知识讲解", "content": "4f 壳层"},
                "prev_hash": "", "event_hash": "abc123",
            },
            {
                "event_id": "evt-2", "event_type": "learner_profile",
                "trace_id": "trace-001", "session_id": "sess-1",
                "agent_id": "A1", "layer": "L2",
                "timestamp": 2000.0,
                "payload": {"kp_id": "A-01", "summary": "BKT 更新", "old": 0.3, "new": 0.6},
                "prev_hash": "abc123", "event_hash": "def456",
            },
        ]

    def test_l0_events_normalize(self):
        normalized = normalize_events(self._l0_events())
        assert len(normalized) == 2
        assert normalized[0]["kp_id"] == "A-01"
        assert normalized[0]["trace_id"] == "trace-001"
        assert normalized[0]["layer"] == "L3"
        assert normalized[0]["summary"] == "知识讲解"

    def test_l0_hash_chain_verified(self):
        d = render_timeline(events=self._l0_events())
        assert d.config["chain_verification"]["valid"] is True
        assert "哈希链完整" in d.html

    def test_l0_payload_summary_fallback(self):
        """payload.summary 回退为事件 summary."""
        ev = self._l0_events()[0]
        normalized = normalize_events([{**ev, "summary": None}])
        assert normalized[0]["summary"] == "知识讲解"  # 从 payload 提取


class TestEscaping:
    """HTML 转义安全 (XSS 防护)."""

    def test_timeline_escapes_script(self):
        events = [{
            "event_id": "evt-1", "event_type": "knowledge", "timestamp": 1.0,
            "summary": '<script>alert("xss")</script>',
            "prev_hash": "", "event_hash": "h",
        }]
        d = render_timeline(events=events)
        assert "<script>" not in d.html
        assert "&lt;script&gt;" in d.html

    def test_approval_escapes_script(self):
        plan = {"plan_id": "p1", "title": '<img src=x onerror=alert(1)>'}
        d = render_plan_preview(plan=plan)
        assert "<img" not in d.html
        assert "&lt;img" in d.html

    def test_contribution_escapes_agent_name(self):
        interactions = [{"agent_id": "agent.learning.diagnosis", "agent_name": '<svg onload=alert(1)>',
                         "action": '<img src=x onerror=alert(2)>', "phase_order": 1}]
        d = render_agent_contribution(interactions=interactions)
        assert "<svg" not in d.html
        assert "<img" not in d.html

    def test_decision_escapes_detail(self):
        steps = [{"title": "决策", "detail": '<b onmouseover="x()">危险</b>'}]
        d = render_decision_trace(steps=steps, depth="summary")
        # `<b` 已被转义 → 浏览器不执行, 不可注入可执行标签
        assert "&lt;b" in d.html
        assert '><b ' not in d.html


class TestL0ApprovalCompat:
    """L0 ApprovalRequest 归一化."""

    def test_l0_request_normalize(self):
        req = {
            "request_id": "apr-001",
            "operation": "teach_plan",
            "target": "Dy3+ 教学",
            "risk_level": "HIGH",
            "reversibility": "REVERSIBLE",
            "approval_mode": "DETAILED_REVIEW",
            "requester": "A1",
            "approver_roles": ["STUDENT"],
            "timeout_seconds": 300.0,
            "context": {"kp_ids": ["A-01", "A-02"], "agent_assignments": {"A1": "引导"}},
        }
        p = normalize_plan(req)
        assert p["plan_id"] == "apr-001"
        assert p["title"] == "Dy3+ 教学"
        assert p["risk_level"] == "high"
        assert p["kp_ids"] == ["A-01", "A-02"]

    def test_l0_approval_flow_renders(self):
        d = render_approval_flow(plan={"request_id": "apr-001", "target": "计划"})
        assert "apr-001" in d.html

    def test_high_risk_badge(self):
        d = render_plan_preview(plan={"plan_id": "p", "risk_level": "CRITICAL"})
        assert "关键" in d.html


class TestEdgeCases:
    """边界输入."""

    def test_timeline_malformed_events(self):
        """畸形事件不崩溃."""
        d = render_timeline(events=[None, {}, {"event_type": 123}])
        assert d.config["event_count"] >= 0

    def test_decision_no_steps(self):
        d = render_decision_trace(steps=None, depth="full")
        assert d.config["step_count"] == 0
        assert "暂无决策步骤" in d.html

    def test_contribution_malformed_interactions(self):
        """畸形交互记录不崩溃 (枚举/缺失字段/非字典)."""
        interactions = [
            None, {},
            {"agent_id": "A1", "phase_order": "not-an-int", "duration_ms": "bad"},
            {"agent_id": "agent.quality.review", "phase_order": 1,
             "interaction_type": "broadcast_send", "related_agents": ["x"]},
        ]
        d = render_agent_contribution(interactions=interactions)
        assert d.config["step_count"] >= 0
        assert d.config["type"] == "agent_contribution"

    def test_quick_mode_no_rules(self):
        d = render_quick_mode()
        assert "暂无规则预设" in d.html

    def test_verify_chain_utility_edge(self):
        assert verify_hash_chain([{"prev_hash": "", "event_hash": "a"},
                                  {"prev_hash": "b", "event_hash": "c"}])["valid"] is False

    def test_branch_merge_missing_nodes(self):
        d = render_branch_merge(
            mainline=[{"id": "m1"}],
            branches=[{"title": "b", "reason": "用户追问"}],
            merges=[],
        )
        assert d.config["branch_count"] == 1
        assert "用户追问" in d.html
