"""L7 渲染器 T2 — ChartRenderer 与 GraphRenderer 单元测试.

测试覆盖:
1. ChartRenderer 7 条自动图表类型推断规则 (bar/line/pie/scatter/heatmap/radar/显式)
2. 领域自定义图: Jablonski 能级图 / 合成工艺流程图 / 光谱叠加
3. ECharts option 结构与交互配置
4. GraphRenderer: 节点/边归一化、BKT 学情着色、瓶颈检测、布局模式
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.exceptions import ArtifactValidationError
from dy3_polaris.l7.models import Artifact, ArtifactType, RenderContext, RenderDescriptor
from dy3_polaris.l7.renderers.chart_renderer import ChartRenderer
from dy3_polaris.l7.renderers.graph_renderer import GraphRenderer


def _chart_artifact(payload: dict, **kwargs) -> Artifact:
    return Artifact(
        type=ArtifactType.CHART,
        mime="application/vnd.dy3.chart+json",
        payload=payload,
        **kwargs,
    )


class TestChartRendererCore:
    """ChartRenderer 基础契约."""

    def test_mime_types(self):
        renderer = ChartRenderer()
        assert "application/vnd.dy3.chart+json" in renderer.supported_mime_types()
        assert "application/vnd.dy3.interactive+json" in renderer.supported_mime_types()

    def test_render_returns_descriptor(self):
        d = ChartRenderer().render(
            _chart_artifact({"title": "t", "data": [{"a": 1, "b": 2}]}),
            RenderContext(),
        )
        assert isinstance(d, RenderDescriptor)
        assert "echarts" in d.assets[0]
        assert d.config["option"]
        assert d.config["interactions"]["tooltip"] is True

    def test_missing_data_raises(self):
        with pytest.raises(ArtifactValidationError):
            ChartRenderer().render(_chart_artifact({"title": "t"}), RenderContext())


class TestChartInference:
    """7 条自动推断规则 (设计文档 §2.3.1)."""

    def test_rule1_discrete_plus_measure_bar(self):
        d = ChartRenderer().render(
            _chart_artifact(
                {"title": "浓度-强度", "data": [{"浓度": "1%", "强度": 100}, {"浓度": "2%", "强度": 180}]}
            ),
            RenderContext(),
        )
        assert d.config["chart_type"] == "bar"
        assert d.config["option"]["series"][0]["type"] == "bar"

    def test_rule2_temporal_line(self):
        d = ChartRenderer().render(
            _chart_artifact(
                {"title": "衰减曲线", "data": [{"时间": "0ms", "强度": 1.0}, {"时间": "10ms", "强度": 0.5}]}
            ),
            RenderContext(),
        )
        assert d.config["chart_type"] == "line"
        assert d.config["option"]["series"][0]["type"] == "line"
        assert "dataZoom" in d.config["option"]

    def test_rule3_category_ratio_pie(self):
        d = ChartRenderer().render(
            _chart_artifact(
                {"title": "能量分配", "data": [{"途径": "A", "占比": 0.4}, {"途径": "B", "占比": 0.6}]}
            ),
            RenderContext(),
        )
        assert d.config["chart_type"] == "pie"
        assert d.config["option"]["series"][0]["type"] == "pie"

    def test_rule4_two_quantitative_scatter(self):
        d = ChartRenderer().render(
            _chart_artifact(
                {"title": "CIE", "data": [{"x": 0.3, "y": 0.6, "v": 1.0}, {"x": 0.4, "y": 0.5, "v": 0.8}]}
            ),
            RenderContext(),
        )
        assert d.config["chart_type"] == "scatter"
        assert d.config["option"]["series"][0]["type"] == "scatter"

    def test_rule5_two_category_heatmap(self):
        d = ChartRenderer().render(
            _chart_artifact(
                {
                    "title": "热力",
                    "dimensions": ["行", "列"],
                    "data": [
                        {"行": "A", "列": "X", "值": 0.8},
                        {"行": "A", "列": "Y", "值": 0.4},
                        {"行": "B", "列": "X", "值": 0.6},
                        {"行": "B", "列": "Y", "值": 0.2},
                    ],
                }
            ),
            RenderContext(),
        )
        assert d.config["chart_type"] == "heatmap"
        assert d.config["option"]["series"][0]["type"] == "heatmap"
        assert "visualMap" in d.config["option"]

    def test_rule6_multi_measure_radar(self):
        d = ChartRenderer().render(
            _chart_artifact(
                {
                    "title": "性能",
                    "chart_type": "radar",
                    "dimensions": ["材料"],
                    "measures": [
                        {"field": "QE", "name": "量子效率", "max": 100},
                        {"field": "寿命", "name": "荧光寿命", "max": 100},
                    ],
                    "data": [{"材料": "NaGdF4", "QE": 85, "寿命": 70}],
                }
            ),
            RenderContext(),
        )
        assert d.config["chart_type"] == "radar"
        assert d.config["option"]["series"][0]["type"] == "radar"

    def test_rule7_explicit_type(self):
        d = ChartRenderer().render(
            _chart_artifact(
                {
                    "title": "显式",
                    "chart_type": "gauge",
                    "data": [{"a": 1, "b": 2}],
                }
            ),
            RenderContext(),
        )
        assert d.config["chart_type"] == "gauge"


class TestChartDomainKinds:
    """领域自定义图 (设计文档 §4.2)."""

    def test_jablonski(self):
        d = ChartRenderer().render(
            _chart_artifact(
                {
                    "graph_kind": "jablonski",
                    "levels": [
                        {"name": "6H15/2", "energy": 0, "type": "ground"},
                        {"name": "4F9/2", "energy": 2.1, "type": "excited4f"},
                        {"name": "4f5 5d", "energy": 5.0, "type": "excited5d"},
                    ],
                    "transitions": [
                        {"from": "6H15/2", "to": "4f5 5d", "kind": "absorption"},
                        {"from": "4f5 5d", "to": "4F9/2", "kind": "relaxation"},
                        {"from": "4F9/2", "to": "6H15/2", "kind": "emission", "wavelength": 575},
                    ],
                }
            ),
            RenderContext(),
        )
        assert d.config["chart_type"] == "domain-jablonski"
        # 能级散点 + 跃迁 markLine 两层 series
        assert len(d.config["option"]["series"]) == 2

    def test_process_flow(self):
        d = ChartRenderer().render(
            _chart_artifact(
                {
                    "graph_kind": "process",
                    "steps": [
                        {"name": "称量", "params": {"温度": "25°C"}},
                        {"name": "焙烧", "params": {"温度": "1200°C"}},
                    ],
                }
            ),
            RenderContext(),
        )
        assert d.config["chart_type"] == "domain-process"
        assert d.config["option"]["series"][0]["type"] == "graph"

    def test_spectrum(self):
        d = ChartRenderer().render(
            _chart_artifact(
                {
                    "graph_kind": "spectrum",
                    "x_label": "波长 (nm)",
                    "series": [
                        {"name": "PL", "data": [{"x": 400, "y": 0.2}, {"x": 500, "y": 0.9}, {"x": 575, "y": 1.0}]},
                    ],
                    "peaks": [{"x": 575, "label": "4F9/2→6H15/2"}],
                }
            ),
            RenderContext(),
        )
        assert d.config["chart_type"] == "domain-spectrum"
        assert d.config["option"]["series"][0]["type"] == "line"
        assert "markLine" in d.config["option"]["series"][-1]


# ============================================================
# GraphRenderer
# ============================================================

class TestGraphRendererCore:
    """GraphRenderer 基础契约."""

    def _graph_artifact(self, payload: dict, bkt: dict | None = None) -> Artifact:
        return Artifact(
            type=ArtifactType.GRAPH,
            mime="application/vnd.dy3.graph+json",
            payload=payload,
            learner_context={"bkt_state": bkt} if bkt else {},
        )

    def test_mime_types(self):
        assert "application/vnd.dy3.graph+json" in GraphRenderer().supported_mime_types()

    def test_render_descriptor(self):
        d = GraphRenderer().render(
            self._graph_artifact(
                {
                    "nodes": [{"id": "A-01", "name": "电子构型", "domain": "A"}],
                    "edges": [],
                }
            ),
            RenderContext(),
        )
        assert isinstance(d, RenderDescriptor)
        assert len(d.config["nodes"]) == 1
        assert "vis-network" in d.assets[0]

    def test_missing_nodes_raises(self):
        with pytest.raises(ArtifactValidationError):
            GraphRenderer().render(
                self._graph_artifact({"edges": []}), RenderContext()
            )

    def test_force_layout_default(self):
        d = GraphRenderer().render(
            self._graph_artifact({"nodes": [], "edges": []}), RenderContext()
        )
        assert d.config["layout"] == "force"
        assert d.config["options"]["physics"]["enabled"] is True

    def test_hierarchical_layout(self):
        d = GraphRenderer().render(
            self._graph_artifact({"nodes": [], "edges": [], "layout": "hierarchical"}),
            RenderContext(),
        )
        assert d.config["layout"] == "hierarchical"
        assert "hierarchical" in d.config["options"]["layout"]

    def test_invalid_layout_fallback(self):
        d = GraphRenderer().render(
            self._graph_artifact({"nodes": [], "edges": [], "layout": "bogus"}),
            RenderContext(),
        )
        assert d.config["layout"] == "force"


class TestGraphBKTColoring(TestGraphRendererCore):
    """BKT 学情着色 (设计文档 §2.4.2)."""

    def _graph_with_bkt(self):
        return self._graph_artifact(
            {
                "nodes": [
                    {"id": "A-01", "name": "已掌握", "domain": "A"},
                    {"id": "A-02", "name": "学习中", "domain": "A"},
                    {"id": "A-03", "name": "薄弱", "domain": "A"},
                    {"id": "A-04", "name": "瓶颈", "domain": "A"},
                ],
                "edges": [{"source": "A-01", "target": "A-02", "label": "基础"}],
            },
            bkt={
                "A-01": {"p_l": 0.9, "p_k_l": 0.85, "p_g": 0.1, "p_s": 0.05},
                "A-02": {"p_l": 0.6, "p_k_l": 0.5, "p_g": 0.2, "p_s": 0.1},
                "A-03": {"p_l": 0.3, "p_k_l": 0.4, "p_g": 0.3, "p_s": 0.1},
                "A-04": {"p_l": 0.75, "p_k_l": 0.2, "p_g": 0.3, "p_s": 0.1},
            },
        )

    def test_mastery_coloring(self):
        d = GraphRenderer().render(self._graph_with_bkt(), RenderContext())
        nodes = {n["id"]: n for n in d.config["nodes"]}
        # 已掌握节点降透明度
        assert nodes["A-01"].get("opacity") == 0.85
        # 薄弱节点加粗
        assert nodes["A-03"].get("borderWidth") == 2.5

    def test_bottleneck_detected(self):
        d = GraphRenderer().render(self._graph_with_bkt(), RenderContext())
        assert d.config["mastery"]["bottlenecks"] == 1
        nodes = {n["id"]: n for n in d.config["nodes"]}
        assert nodes["A-04"]["data"]["bottleneck"] is True
        assert "borderWidth" in nodes["A-04"]

    def test_mastery_summary_counts(self):
        d = GraphRenderer().render(self._graph_with_bkt(), RenderContext())
        summary = d.config["mastery"]
        assert summary["mastered"] == 1      # A-01 (0.9)
        assert summary["learning"] == 2      # A-02 (0.6) + A-04 (0.75)
        assert summary["weak"] == 1          # A-03 (0.3)
        assert summary["tracked"] == 4
        assert summary["bottlenecks"] == 1   # A-04 (P(L)=0.75>0.7 且 P(K|L)=0.2<0.3)

    def test_edge_relation_styles(self):
        d = GraphRenderer().render(
            self._graph_artifact(
                {
                    "nodes": [{"id": "A-01"}, {"id": "A-02"}, {"id": "A-03"}],
                    "edges": [
                        {"source": "A-01", "target": "A-02", "label": "基础"},
                        {"source": "A-01", "target": "A-03", "label": "关联"},
                    ],
                }
            ),
            RenderContext(),
        )
        edges = d.config["edges"]
        assert edges[0]["width"] == 1.5  # 基础
        assert edges[1]["dashes"] is True  # 关联

    def test_learning_path_highlight(self):
        d = GraphRenderer().render(
            self._graph_artifact(
                {
                    "nodes": [{"id": "A-01"}, {"id": "A-02"}, {"id": "A-03"}],
                    "edges": [
                        {"source": "A-01", "target": "A-02", "label": "基础"},
                        {"source": "A-02", "target": "A-03", "label": "基础"},
                    ],
                    "learning_path": {"recommended": ["A-01", "A-02"], "current": ["A-01"]},
                }
            ),
            RenderContext(),
        )
        edges = d.config["edges"]
        # 推荐路径边加粗
        assert any(e["width"] == 3.5 for e in edges)
