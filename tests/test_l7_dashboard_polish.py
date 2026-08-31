"""L7 学情面板 T4 — 深化完善轮单元测试.

覆盖完善轮修复与增强:
1. 热力图 tooltip formatter 无未定义变量依赖 (JSON 安全, 无 JS 函数字符串)
2. 色盲友好配色 (蓝橙渐变替换红绿)
3. 参数调节器图表 xAxis 无 data (value 轴) + 数据为 [x,y] 对
4. 虚拟实验台 spectrum_chart 配置 (lab-spectrum 容器有对应 option)
5. 面板统一样式嵌入 (dashboard_wrap)
6. _common 公共层边界 (空矩阵/无前置/瓶颈系数)
"""

from __future__ import annotations

import json

import pytest

from dy3_polaris.l7.dashboard._common import (
    DASHBOARD_CSS,
    bottlenecks,
    build_heatmap_option,
    dashboard_wrap,
    extract_bkt_matrix,
    learning_path,
    weak_points,
)
from dy3_polaris.l7.dashboard import render_bkt_dashboard, render_kp_detail
from dy3_polaris.l7.interactive import (
    render_param_controller,
    render_virtual_lab,
)
from dy3_polaris.l7.models import RenderContext, RenderDescriptor


class TestHeatmapJSONSafety:
    """热力图配置 JSON 安全性与 tooltip 修复."""

    def test_no_js_function_strings(self):
        """formatter 不得包含 JS 函数字符串 (前端 eval 风险)."""
        matrix = {"A-01": {"p_l": 0.9, "p_k_l": 0.8, "p_g": 0.1, "p_s": 0.1}}
        opt = build_heatmap_option(matrix)
        # 序列化必须成功 (无函数对象)
        dumped = json.dumps(opt, ensure_ascii=False)
        assert "function" not in dumped
        assert "yAxisNames" not in dumped  # 修复: 不再引用未定义变量

    def test_tooltip_uses_data_template(self):
        matrix = {"A-01": {"p_l": 0.9, "p_k_l": 0.8, "p_g": 0.1, "p_s": 0.1}}
        opt = build_heatmap_option(matrix)
        assert "{@[4]}" in opt["tooltip"]["formatter"]
        # data 元素含 [x, y, value, param_label, kp_id]
        assert len(opt["series"][0]["data"][0]) == 5

    def test_colorblind_gradient(self):
        matrix = {"A-01": {"p_l": 0.9, "p_k_l": 0.8, "p_g": 0.1, "p_s": 0.1}}
        normal = build_heatmap_option(matrix)
        cb = build_heatmap_option(matrix, colorblind=True)
        assert normal["visualMap"]["inRange"]["color"] != cb["visualMap"]["inRange"]["color"]
        # 色盲版使用蓝橙系 (含蓝色)
        assert "#2563eb" in cb["visualMap"]["inRange"]["color"]


class TestColorblindIntegration:
    """色盲开关贯通 BKT 面板."""

    def test_colorblind_from_context(self):
        ctx = RenderContext(bkt_state={"colorblind": True, "A-01": {"p_l": 0.5}})
        d = render_bkt_dashboard(context=ctx)
        colors = d.config["heatmap"]["visualMap"]["inRange"]["color"]
        assert "#2563eb" in colors

    def test_default_red_green(self):
        ctx = RenderContext(bkt_state={"A-01": {"p_l": 0.5}})
        d = render_bkt_dashboard(context=ctx)
        colors = d.config["heatmap"]["visualMap"]["inRange"]["color"]
        assert "#16a34a" in colors


class TestParamControllerCharts:
    """参数调节器图表结构 (xAxis 修复)."""

    def test_value_axis_no_data(self):
        d = render_param_controller()
        for p in d.config["params"]:
            chart = d.config["charts"][p["key"]]["chart"]
            assert "data" not in chart["xAxis"]  # value 轴不带 data
            assert chart["xAxis"]["type"] == "value"
            # series data 为 [x, y] 数值对
            first = chart["series"][0]["data"][0]
            assert isinstance(first, list) and len(first) == 2
            assert isinstance(first[0], (int, float))

    def test_five_params_all_have_charts(self):
        d = render_param_controller()
        assert len(d.config["charts"]) == 5
        for key in ("doping", "calc_temp", "exc_wavelength", "env_temp", "charge_ratio"):
            assert key in d.config["charts"]


class TestVirtualLabSpectrum:
    """虚拟实验台光谱图配置."""

    def test_spectrum_chart_present(self):
        d = render_virtual_lab(host="NaGdF4")
        assert "spectrum_chart" in d.config
        chart = d.config["spectrum_chart"]
        assert chart["series"][0]["type"] == "line"
        assert chart["series"][0]["markLine"] is not None

    def test_spectrum_peak_mark(self):
        d = render_virtual_lab(host="YPO4")
        chart = d.config["spectrum_chart"]
        assert chart["series"][0]["markLine"]["data"][0]["xAxis"] == 575

    def test_output_artifacts_unchanged(self):
        d = render_virtual_lab(host="BaMgAl10O17", doping=3.0)
        assert len(d.config["output_artifacts"]) == 3


class TestDashboardStyles:
    """面板统一样式."""

    def test_wrap_embeds_css(self):
        html = dashboard_wrap("<div>x</div>", "l7-dashboard l7-bkt", "light")
        assert "l7-dashboard" in html
        assert "<style>" in html
        assert ".domain-card" in html
        # light 模式根容器不带 dark 类 (CSS 中的 .dark 定义除外)
        assert 'l7-dashboard dark"' not in html

    def test_wrap_dark_theme(self):
        html = dashboard_wrap("<div>x</div>", "l7-dashboard", "dark")
        assert 'class="l7-render l7-dashboard dark"' in html

    def test_dashboards_use_dashboard_wrap(self):
        ctx = RenderContext(bkt_state={"A-01": {"p_l": 0.6}})
        d = render_bkt_dashboard(context=ctx)
        assert ".bkt-header" in d.html  # 面板样式已嵌入


class TestCommonEdgeCases:
    """公共层边界."""

    def test_bottlenecks_empty(self):
        assert bottlenecks({}) == []
        assert bottlenecks({"A-01": {"p_l": 0.5, "p_k_l": 0.8}}) == []

    def test_bottleneck_threshold(self):
        matrix = {
            "A-01": {"p_l": 0.8, "p_k_l": 0.9},   # 非瓶颈 (K|L 高)
            "A-02": {"p_l": 0.8, "p_k_l": 0.2},   # 瓶颈
            "A-03": {"p_l": 0.6, "p_k_l": 0.2},   # 非瓶颈 (P(L) 不高)
        }
        bns = bottlenecks(matrix)
        ids = [b["kp_id"] for b in bns]
        assert "A-02" in ids
        assert "A-01" not in ids
        assert "A-03" not in ids

    def test_learning_path_without_prereqs(self):
        matrix = {"A-01": {"p_l": 0.4}, "A-02": {"p_l": 0.9}}
        path = learning_path(matrix)  # 无前置定义 → 全部可推荐
        assert len(path) >= 1

    def test_learning_path_prereq_blocks(self):
        matrix = {
            "A-01": {"p_l": 0.2},  # 前置未达标
            "A-02": {"p_l": 0.3},
        }
        prereqs = {"A-02": ["A-01"]}  # A-02 依赖 A-01
        path = learning_path(matrix, prerequisites=prereqs)
        # A-01 前置 P(L)=0.2 < 0.6 → A-02 被阻塞
        assert not any(p["kp_id"] == "A-02" for p in path)

    def test_extract_bkt_matrix_zeros(self):
        matrix = extract_bkt_matrix(None, None)
        assert len(matrix) == 42
        assert matrix["A-01"]["p_l"] == 0.0

    def test_weak_points_scoring(self):
        matrix = {
            "A-01": {"p_l": 0.1, "p_k_l": 0.3},
            "A-02": {"p_l": 0.8, "p_k_l": 0.7},
        }
        wps = weak_points(matrix)
        # A-01 低掌握度 → 排前面
        assert wps[0]["kp_id"] == "A-01"

    def test_kp_detail_still_works(self):
        ctx = RenderContext(bkt_state={"A-01": {"p_l": 0.9, "p_k_l": 0.8, "p_g": 0.1, "p_s": 0.1}})
        d = render_kp_detail("A-01", context=ctx)
        assert isinstance(d, RenderDescriptor)
        assert "charts" in d.config
