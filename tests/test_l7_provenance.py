"""L7 溯源可视化 T5 — provenance 子包单元测试.

测试覆盖:
1. 溯源时间线: 事件归一化、哈希链验证 (完整/中断)、隐私脱敏
2. 决策溯源: 三级深度过滤、六维雷达、内部推理授权
3. Agent 贡献图谱: 分组条形图、网络图
4. 分支合并: 主线/分支/合并、分支原因标注
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.models import RenderDescriptor
from dy3_polaris.l7.provenance import (
    render_agent_contribution,
    render_branch_merge,
    render_decision_trace,
    render_timeline,
)
from dy3_polaris.l7.provenance._common import normalize_events, verify_hash_chain


def _chain_events() -> list[dict]:
    """构造哈希链完整的事件序列."""
    return [
        {"event_id": "evt-1", "event_type": "knowledge", "timestamp": 1000.0,
         "agent_id": "A1", "summary": "讲解", "prev_hash": "", "event_hash": "h1"},
        {"event_id": "evt-2", "event_type": "learner_profile", "timestamp": 2000.0,
         "agent_id": "A1", "summary": "BKT 更新", "prev_hash": "h1", "event_hash": "h2"},
        {"event_id": "evt-3", "event_type": "decision", "timestamp": 3000.0,
         "agent_id": "MD", "summary": "决策", "prev_hash": "h2", "event_hash": "h3"},
    ]


class TestTimeline:
    """溯源时间线."""

    def test_renders_descriptor(self):
        d = render_timeline(events=_chain_events())
        assert isinstance(d, RenderDescriptor)
        assert d.config["type"] == "provenance_timeline"
        assert d.config["event_count"] == 3

    def test_hash_chain_valid(self):
        d = render_timeline(events=_chain_events())
        assert d.config["chain_verification"]["valid"] is True
        assert d.config["chain_verification"]["verified"] == 3
        assert "哈希链完整" in d.html

    def test_hash_chain_broken(self):
        events = _chain_events()
        events[1] = {**events[1], "prev_hash": "tampered"}
        d = render_timeline(events=events)
        assert d.config["chain_verification"]["valid"] is False
        assert d.config["chain_verification"]["first_break_index"] == 1
        assert "哈希链中断" in d.html

    def test_privacy_masked_by_default(self):
        events = _chain_events() + [{
            "event_id": "evt-4", "event_type": "interaction", "timestamp": 4000.0,
            "agent_id": "L1", "summary": "答题", "raw": "用户原始输入",
            "prev_hash": "h3", "event_hash": "h4",
        }]
        d = render_timeline(events=events)
        assert "已脱敏" in d.html
        assert "用户原始输入" not in d.html

    def test_privacy_full_access(self):
        events = _chain_events() + [{
            "event_id": "evt-4", "event_type": "interaction", "timestamp": 4000.0,
            "agent_id": "L1", "summary": "答题", "raw": "用户原始输入",
            "prev_hash": "h3", "event_hash": "h4",
        }]
        d = render_timeline(events=events, full_access=True)
        assert "用户原始输入" in d.html

    def test_l0_type_mapping(self):
        """L0 五类事件 → L7 可视化类型映射."""
        events = [
            {"event_type": "learner_profile", "summary": "a"},
            {"event_type": "knowledge", "summary": "b"},
            {"event_type": "interaction", "summary": "c"},
            {"event_type": "human_override", "summary": "d"},
        ]
        normalized = normalize_events(events)
        types = [n["type"] for n in normalized]
        assert types == ["state_change", "teaching", "test", "edit"]

    def test_empty_events(self):
        d = render_timeline(events=[])
        assert d.config["event_count"] == 0
        assert "暂无溯源记录" in d.html

    def test_verify_hash_chain_utility(self):
        assert verify_hash_chain([])["valid"] is True
        assert verify_hash_chain([{"prev_hash": "", "event_hash": "a"}])["valid"] is True


class TestDecisionTrace:
    """决策溯源."""

    def _steps(self) -> list[dict]:
        return [
            {"title": "复杂度评估", "detail": "六维评分", "score": 0.7},
            {"title": "范式选择", "detail": "选择 Pipeline", "score": 0.8},
            {"title": "Agent 调度", "detail": "调度 A1", "agent": "MD"},
            {"title": "执行过程", "detail": "产出 Artifact", "internal": "内部推理"},
            {"title": "裁决结果", "detail": "采纳 A1", "score": 0.9},
        ]

    def test_summary_depth_filters(self):
        d = render_decision_trace(steps=self._steps(), depth="summary")
        assert d.config["type"] == "decision_trace"
        assert d.config["depth"] == "summary"
        # summary 仅保留关键节点
        assert d.config["step_count"] < len(self._steps())
        assert d.config["step_count"] >= 1

    def test_standard_depth_all(self):
        d = render_decision_trace(steps=self._steps(), depth="standard")
        assert d.config["step_count"] == len(self._steps())

    def test_invalid_depth_fallback(self):
        d = render_decision_trace(steps=self._steps(), depth="bogus")
        assert d.config["depth"] == "standard"

    def test_complexity_radar(self):
        complexity = [{"name": "知识深度", "value": 80, "max": 100}]
        d = render_decision_trace(steps=self._steps(), complexity=complexity)
        assert d.config["complexity_radar"] is not None
        assert d.config["complexity_radar"]["series"][0]["type"] == "radar"

    def test_internal_hidden_without_access(self):
        d = render_decision_trace(steps=self._steps(), depth="full", full_access=False)
        assert "内部推理，需完整模式授权" in d.html  # 掩码提示
        assert 'prov-internal">内部推理<' not in d.html.replace("（内部推理，需完整模式授权）", "")

    def test_internal_shown_with_access(self):
        d = render_decision_trace(steps=self._steps(), depth="full", full_access=True)
        assert "内部推理" in d.html


class TestAgentContribution:
    """Agent 交互链时间线 (逐步: 谁 → 做了什么 → 传给谁)."""

    def _interactions(self) -> list[dict]:
        """构造一条完整 4-Agent 协同交互链 (含广播)."""
        return [
            {"agent_id": "agent.learning.diagnosis", "agent_name": "学情诊断",
             "action": "分析学习者画像", "phase": "diagnosis", "phase_order": 1,
             "duration_ms": 800, "status": "completed",
             "interaction_type": "agent_execution", "related_agents": [], "timestamp": 1000.0},
            {"agent_id": "agent.knowledge.generation", "agent_name": "知识生成",
             "action": "生成知识解释", "phase": "generation", "phase_order": 2,
             "duration_ms": 1200, "status": "completed",
             "interaction_type": "agent_execution", "related_agents": [], "timestamp": 2000.0},
            {"agent_id": "agent.quality.review", "agent_name": "审核校验",
             "action": "广播审核结果", "phase": "review", "phase_order": 3,
             "duration_ms": 600, "status": "completed",
             "interaction_type": "broadcast_send",
             "related_agents": ["agent.guidance.decision"],
             "channel": "knowledge.review.result", "timestamp": 3000.0},
            {"agent_id": "agent.guidance.decision", "agent_name": "导学决策",
             "action": "整合决策", "phase": "decision", "phase_order": 4,
             "duration_ms": 900, "status": "completed",
             "interaction_type": "agent_execution", "related_agents": [], "timestamp": 4000.0},
        ]

    def test_step_timeline(self):
        d = render_agent_contribution(interactions=self._interactions())
        assert d.config["type"] == "agent_contribution"
        assert d.config["step_count"] == 4
        assert d.config["agent_count"] == 4

    def test_steps_show_agent_action(self):
        d = render_agent_contribution(interactions=self._interactions())
        assert "第1步" in d.html
        assert "第4步" in d.html
        assert "学情诊断" in d.html  # 名称而非 ID
        assert "知识生成" in d.html
        assert "分析学习者画像" in d.html  # 做了什么

    def test_steps_show_pass_to(self):
        d = render_agent_contribution(interactions=self._interactions())
        # 广播步骤: 传给接收 Agent
        assert "传给" in d.html
        assert "导学决策" in d.html
        # 顺序执行步骤: 交接给下一环节
        assert "交接给" in d.html

    def test_empty_interactions(self):
        d = render_agent_contribution(interactions=[])
        assert d.config["step_count"] == 0
        assert "暂无交互记录" in d.html

    def test_backward_agents_summary(self):
        d = render_agent_contribution(agents=[{"id": "A1", "name": "引导"}])
        assert d.config["type"] == "agent_contribution"
        assert d.config["agent_count"] >= 1
        assert "暂无交互记录" in d.html  # 无逐步数据时给出空态


class TestBranchMerge:
    """分支合并可视化."""

    def _data(self):
        return {
            "mainline": [{"id": "m1", "title": "主线 v1"}, {"id": "m2", "title": "主线 v2"}],
            "branches": [{
                "title": "分支 A", "reason": "用户追问",
                "nodes": [{"title": "编辑 1"}, {"title": "编辑 2"}],
            }],
            "merges": [{"title": "合并", "result": "采纳分支 A"}],
        }

    def test_render(self):
        d = render_branch_merge(**self._data())
        assert d.config["type"] == "branch_merge"
        assert d.config["mainline_count"] == 2
        assert d.config["branch_count"] == 1
        assert d.config["merge_count"] == 1

    def test_reason_labeled(self):
        d = render_branch_merge(**self._data())
        assert "用户追问" in d.html
        assert "采纳分支 A" in d.html

    def test_unknown_reason_fallback(self):
        data = self._data()
        data["branches"][0]["reason"] = "未知原因"
        d = render_branch_merge(**data)
        assert "未知原因" in d.html

    def test_empty(self):
        d = render_branch_merge(mainline=[], branches=[], merges=[])
        assert d.config["branch_count"] == 0
