"""L7 学情面板 T4 — BKT Dashboard 与进度面板单元测试.

测试覆盖:
1. BKT Dashboard: 热力图配置、瓶颈列表、知识拓扑图
2. KP 详情: 四参数条形图、学习轨迹、瓶颈标记
3. 进度面板: 总体掌握度、域级聚合、薄弱点排序、学习路径推荐
4. 辩论可视化: 时间线、收敛图、裁决雷达图
5. 交互模式: 下钻/时间旅行/对比
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.dashboard import (
    render_bkt_dashboard,
    render_comparison,
    render_debate,
    render_drill_down,
    render_kp_detail,
    render_progress_panel,
    render_time_travel,
)
from dy3_polaris.l7.models import RenderContext, RenderDescriptor


_BKT_FIXTURE = {
    "A-01": {"p_l": 0.9, "p_k_l": 0.85, "p_g": 0.1, "p_s": 0.05},
    "A-02": {"p_l": 0.6, "p_k_l": 0.5, "p_g": 0.2, "p_s": 0.1},
    "A-03": {"p_l": 0.3, "p_k_l": 0.4, "p_g": 0.3, "p_s": 0.1},
    "A-04": {"p_l": 0.75, "p_k_l": 0.2, "p_g": 0.2, "p_s": 0.1},
    "B-01": {"p_l": 0.95, "p_k_l": 0.9, "p_g": 0.1, "p_s": 0.05},
    "C-01": {"p_l": 0.1, "p_k_l": 0.3, "p_g": 0.4, "p_s": 0.1},
    "D-01": {"p_l": 0.55, "p_k_l": 0.5, "p_g": 0.2, "p_s": 0.15},
    "D-02": {"p_l": 0.0, "p_k_l": 0.0, "p_g": 0.0, "p_s": 0.0},
}


def _ctx(**bkt):
    return RenderContext(bkt_state=bkt or _BKT_FIXTURE)


class TestBKTDashboard:
    """BKT 学情面板."""

    def test_renders_descriptor(self):
        d = render_bkt_dashboard(context=_ctx())
        assert isinstance(d, RenderDescriptor)
        assert d.config["type"] == "bkt_dashboard"
        assert d.config["summary"]["total_kps"] == 42

    def test_heatmap_has_data(self):
        d = render_bkt_dashboard(context=_ctx())
        heatmap = d.config["heatmap"]
        assert "series" in heatmap
        assert heatmap["series"][0]["type"] == "heatmap"

    def test_bottlenecks_detected(self):
        d = render_bkt_dashboard(context=_ctx())
        # A-04: P(L)=0.75>0.7 且 P(K|L)=0.2<0.3 → 瓶颈
        bns = d.config["bottlenecks"]
        assert any(bn["id"] == "A-04" for bn in bns)
        assert d.config["bottleneck_count"] >= 1

    def test_topology_nodes(self):
        d = render_bkt_dashboard(context=_ctx())
        nodes = d.config["topology"]["nodes"]
        assert len(nodes) == 42

    def test_summary_tracked(self):
        d = render_bkt_dashboard(context=_ctx())
        # D-02 p_l=0 不计入 tracked
        assert d.config["summary"]["tracked"] >= 5

    def test_empty_bkt_does_not_crash(self):
        d = render_bkt_dashboard(context=RenderContext())
        assert d.config["summary"]["avg_p_l"] == 0.0
        assert d.config["bottleneck_count"] == 0


class TestKPDetail:
    """单 KP 详情."""

    def test_kp_detail_basic(self):
        d = render_kp_detail("A-01", context=_ctx())
        assert d.config["type"] == "kp_detail"
        assert d.config["kp_id"] == "A-01"
        assert "charts" in d.config

    def test_kp_detail_bottleneck_flag(self):
        d = render_kp_detail("A-04", context=_ctx())
        assert d.config["is_bottleneck"] is True

    def test_missing_kp_zeros(self):
        d = render_kp_detail("ZZ-99", context=_ctx())
        assert d.config["bkt_state"]["p_l"] == 0.0


class TestProgressPanel:
    """学习进度面板."""

    def test_renders_descriptor(self):
        d = render_progress_panel(context=_ctx())
        assert d.config["type"] == "progress_panel"
        assert d.config["average_mastery"] > 0

    def test_domain_cards_four(self):
        d = render_progress_panel(context=_ctx())
        assert len(d.config["domain_cards"]) == 4
        assert "A" in d.config["domain_cards"]

    def test_weak_points_sorted(self):
        d = render_progress_panel(context=_ctx())
        wps = d.config["weak_points"]
        assert all(0 <= wp["score"] <= 1 for wp in wps)
        if len(wps) >= 2:
            assert wps[0]["score"] >= wps[1]["score"]

    def test_learning_path_prereqs(self):
        prerequisites = {"A-03": ["A-01", "A-02"]}
        d = render_progress_panel(context=_ctx(), prerequisites=prerequisites)
        path = d.config["learning_path"]
        # A-03 前置 A-01(p_l=0.9>0.6 满足) 和 A-02(p_l=0.6>=0.6 满足) → 可推荐
        assert any(p["kp_id"] == "A-03" for p in path)

    def test_ring_chart_present(self):
        d = render_progress_panel(context=_ctx())
        assert "series" in d.config["ring_chart"]


class TestDebateViz:
    """辩论可视化."""

    def test_timeline(self):
        d = render_debate(speeches=[
            {"agent": "A1", "stance": "support", "summary": "测试发言"},
        ])
        assert d.config["speech_count"] == 1
        assert "debate-timeline" in d.html

    def test_convergence_chart(self):
        d = render_debate(convergence={"rounds": [1, 2], "consensus": [0.2, 0.8]})
        assert d.config["convergence"] is not None

    def test_verdict_radar(self):
        d = render_debate(verdict={
            "summary": "测试裁决",
            "selected_agent": "A1",
            "dimensions": [
                {"name": "准确性", "value": 80, "max": 100},
                {"name": "完整性", "value": 70, "max": 100},
                {"name": "教学适切性", "value": 90, "max": 100},
            ],
        })
        assert d.config["verdict"] is not None
        assert "择" in d.html or "verdict" in d.html.lower()

    def test_empty_speeches_ok(self):
        d = render_debate()
        assert d.config["speech_count"] == 0


class TestInteractionModes:
    """交互模式."""

    def test_drill_down(self):
        d = render_drill_down(current_level="domain-A")
        assert d.config["type"] == "drill_down"
        assert d.config["current_level"] == "domain-A"

    def test_time_travel(self):
        d = render_time_travel(snapshots=[
            {"label": "D1", "timestamp": 1000},
            {"label": "D2", "timestamp": 2000},
        ])
        assert d.config["snapshot_count"] == 2

    def test_comparison(self):
        d = render_comparison(learners=[
            {"id": "s1", "label": "A", "avg_p_l": 0.8, "domain_scores": {}},
            {"id": "s2", "label": "B", "avg_p_l": 0.4, "domain_scores": {}},
        ])
        assert d.config["learner_count"] == 2


class TestInteractive:
    """参数调节器 + 虚拟实验台 + 图谱探索器."""

    def test_param_controller(self):
        from dy3_polaris.l7.interactive import render_param_controller, render_virtual_lab, render_graph_explorer

        d = render_param_controller()
        assert len(d.config["params"]) == 5
        for p in d.config["params"]:
            assert "chart" in d.config["charts"][p["key"]]

    def test_virtual_lab_output_artifacts(self):
        from dy3_polaris.l7.interactive import render_virtual_lab

        d = render_virtual_lab(host="NaGdF4", doping=2.0)
        assert len(d.config["output_artifacts"]) == 3
        assert d.config["predictions"]["peak"] == 575

    def test_graph_explorer_interactions(self):
        from dy3_polaris.l7.interactive import render_graph_explorer

        d = render_graph_explorer()
        assert "interactions" in d.config
        assert "click_expand" in d.config["interactions"]
