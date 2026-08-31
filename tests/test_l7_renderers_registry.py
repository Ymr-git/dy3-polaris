"""L7 渲染器 T2 — 注册集成与公共模块单元测试.

测试覆盖:
1. register_native_renderers: 一键注册 7 个原生渲染器, 11 个 MIME 映射
2. MIME → 渲染器路由正确性 (含 Pipeline 全链路)
3. 公共模块 _common: 42 KP 领域常量、BKT 状态提取、掌握度着色、瓶颈检测
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.registry import RendererRegistry
from dy3_polaris.l7.renderers import (
    register_native_renderers,
    native_renderer_classes,
)
from dy3_polaris.l7.renderers._common import (
    ALL_KP_IDS,
    KP_DOMAIN_IDS,
    KP_TO_DOMAIN,
    KP_LEVELS,
    KP_NAMES,
)
from dy3_polaris.l7.renderers import _common
from dy3_polaris.l7.renderers.text_renderer import TextRenderer
from dy3_polaris.l7.renderers.chart_renderer import ChartRenderer
from dy3_polaris.l7.renderers.graph_renderer import GraphRenderer
from dy3_polaris.l7.renderers.molecule_renderer import MoleculeRenderer
from dy3_polaris.l7.renderers.table_renderer import TableRenderer
from dy3_polaris.l7.renderers.formula_renderer import FormulaRenderer
from dy3_polaris.l7.renderers.provenance_renderer import ProvenanceRenderer


class TestRegisterNativeRenderers:
    """一键注册集成."""

    def test_registers_all_renderers(self):
        registry = RendererRegistry()
        mimes = register_native_renderers(registry)
        assert len(native_renderer_classes) == 7
        assert len(mimes) >= 10

    def test_all_native_mimes_registered(self):
        registry = RendererRegistry()
        register_native_renderers(registry)
        for mime in (
            "text/vnd.dy3+markdown",
            "application/vnd.dy3.chart+json",
            "application/vnd.dy3.graph+json",
            "application/vnd.dy3.molecule+json",
            "application/vnd.dy3.table+json",
            "application/vnd.dy3.formula+json",
            "application/vnd.dy3.provenance+json",
        ):
            assert registry.is_supported(mime), f"{mime} 未注册"

    def test_mime_routes_to_correct_renderer(self):
        registry = RendererRegistry()
        register_native_renderers(registry)
        assert isinstance(
            registry.get_renderer("text/vnd.dy3+markdown"), TextRenderer
        )
        assert isinstance(
            registry.get_renderer("application/vnd.dy3.chart+json"), ChartRenderer
        )
        assert isinstance(
            registry.get_renderer("application/vnd.dy3.graph+json"), GraphRenderer
        )
        assert isinstance(
            registry.get_renderer("application/vnd.dy3.molecule+json"), MoleculeRenderer
        )
        assert isinstance(
            registry.get_renderer("application/vnd.dy3.table+json"), TableRenderer
        )
        assert isinstance(
            registry.get_renderer("application/vnd.dy3.formula+json"), FormulaRenderer
        )
        assert isinstance(
            registry.get_renderer("application/vnd.dy3.provenance+json"),
            ProvenanceRenderer,
        )

    def test_interactive_mime_routes_to_chart(self):
        registry = RendererRegistry()
        register_native_renderers(registry)
        assert isinstance(
            registry.get_renderer("application/vnd.dy3.interactive+json"), ChartRenderer
        )

    def test_l7_package_exports(self):
        import dy3_polaris.l7 as l7

        for name in (
            "TextRenderer",
            "ChartRenderer",
            "GraphRenderer",
            "MoleculeRenderer",
            "TableRenderer",
            "FormulaRenderer",
            "ProvenanceRenderer",
            "register_native_renderers",
        ):
            assert hasattr(l7, name), f"l7 包缺少导出 {name}"


class TestCommonConstants:
    """_common 领域常量."""

    def test_42_kps_total(self):
        assert len(_common.ALL_KP_IDS) == 42

    def test_domain_distribution(self):
        assert len(KP_DOMAIN_IDS["A"]) == 13
        assert len(KP_DOMAIN_IDS["B"]) == 11
        assert len(KP_DOMAIN_IDS["C"]) == 10
        assert len(KP_DOMAIN_IDS["D"]) == 8

    def test_kp_to_domain(self):
        assert KP_TO_DOMAIN["A-01"] == "A"
        assert KP_TO_DOMAIN["D-08"] == "D"

    def test_kp_levels(self):
        assert KP_LEVELS["A-01"] == "L1"
        assert KP_LEVELS["A-06"] == "L2"
        assert KP_LEVELS["A-11"] == "L3"

    def test_kp_names_cover_all(self):
        for kp_id in ALL_KP_IDS:
            assert kp_id in KP_NAMES, f"{kp_id} 缺少名称"
            assert KP_NAMES[kp_id], f"{kp_id} 名称为空"


class TestCommonBKT:
    """_common BKT 工具."""

    def test_get_bkt_state_merge(self):
        from dy3_polaris.l7.models import Artifact, ArtifactType, RenderContext

        art = Artifact(
            type=ArtifactType.TEXT,
            mime="text/vnd.dy3+markdown",
            payload={"content": "x"},
            learner_context={
                "bkt_state": {"A-01": {"p_l": 0.5, "p_k_l": 0.4, "p_g": 0.2, "p_s": 0.1}}
            },
        )
        ctx = RenderContext(bkt_state={"A-02": {"p_l": 0.8, "p_k_l": 0.8, "p_g": 0.1, "p_s": 0.1}})
        merged = _common.get_bkt_state(art, ctx)
        assert "A-01" in merged and "A-02" in merged

    def test_get_kp_state_missing(self):
        assert _common.get_kp_state({}, "A-01") is None
        assert _common.get_kp_state(None, "A-01") is None

    def test_average_p_l(self):
        assert _common.average_p_l({"A-01": {"p_l": 0.2}, "A-02": {"p_l": 0.4}}) == pytest.approx(0.3)
        assert _common.average_p_l({}) == 0.0

    def test_is_bottleneck(self):
        assert _common.is_bottleneck({"p_l": 0.8, "p_k_l": 0.2}) is True
        assert _common.is_bottleneck({"p_l": 0.8, "p_k_l": 0.9}) is False
        assert _common.is_bottleneck({"p_l": 0.5, "p_k_l": 0.2}) is False

    def test_mastery_level(self):
        assert _common.mastery_level(0.9) == "mastered"
        assert _common.mastery_level(0.6) == "learning"
        assert _common.mastery_level(0.3) == "weak"
        assert _common.mastery_level(None) == "weak"

    def test_mastery_color_theme(self):
        assert _common.mastery_color(0.9, "light") != _common.mastery_color(0.9, "dark")
        assert _common.mastery_color(0.9, "light") != _common.mastery_color(0.9, "light", colorblind=True)
