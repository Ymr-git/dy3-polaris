"""L7 响应式/无障碍 T6 — responsive + accessibility + i18n + monitoring 单元测试."""

from __future__ import annotations

import pytest

from dy3_polaris.l7.accessibility import (
    COLORBLIND_BLUE_ORANGE,
    CONTRAST_LARGE,
    CONTRAST_NORMAL,
    KEYBOARD_SHORTCUTS,
    a11y_attributes,
    a11y_audit_report,
    audit_contrast,
    colorblind_css,
    contrast_ratio,
    high_contrast_css,
    ishihara_test,
    passes_aa,
    relative_luminance,
)
from dy3_polaris.l7.i18n import (
    format_date,
    format_number,
    glossary,
    i18n_init_config,
    is_supported_locale,
    normalize_locale,
    rtl_css,
    translate,
)
from dy3_polaris.l7.monitoring import (
    PerformanceTracker,
    check_render_latency,
    check_vitals,
    check_ws_latency,
)
from dy3_polaris.l7.responsive import (
    BREAKPOINTS,
    DESKTOP_COLUMNS,
    MIN_TOUCH_SIZE,
    breakpoint_for,
    layout_plan,
    render_layout,
    responsive_css,
)


class TestBreakpoints:
    """三断点判定."""

    def test_desktop(self):
        assert breakpoint_for(1280) == "desktop"
        assert breakpoint_for(1200) == "desktop"

    def test_tablet(self):
        assert breakpoint_for(1000) == "tablet"
        assert breakpoint_for(768) == "tablet"

    def test_mobile(self):
        assert breakpoint_for(500) == "mobile"
        assert breakpoint_for(767) == "mobile"

    def test_desktop_columns(self):
        plan = layout_plan(1400)
        assert plan["breakpoint"] == "desktop"
        assert plan["columns"] == [240, 1, 320]
        assert plan["nav_mode"] == "top_tabs_sidebar"
        assert plan["charts_mode"] == "full_interactive"

    def test_tablet_plan(self):
        plan = layout_plan(1000)
        assert plan["nav_mode"] == "drawer_hamburger"
        assert plan["charts_mode"] == "touch_optimized"
        assert plan["touch_size"] == 44

    def test_mobile_plan(self):
        plan = layout_plan(400)
        assert plan["nav_mode"] == "bottom_tabs"
        assert plan["charts_mode"] == "static_with_drill"
        assert plan["mobile_tabs"] is True

    def test_render_layout_html(self):
        rl = render_layout(1000)
        assert rl["config"]["breakpoint"] == "tablet"
        assert "l7-tabbar" in rl["html"]
        assert "l7-hamburger" in rl["html"]

    def test_render_layout_desktop(self):
        rl = render_layout(1400)
        assert "l7-nav" in rl["html"]
        assert "l7-side" in rl["html"]

    def test_responsive_css_media_queries(self):
        css = responsive_css()
        assert "@media" in css
        assert "prefers-reduced-motion" in css
        assert str(MIN_TOUCH_SIZE) in css


class TestContrast:
    """WCAG 对比度."""

    def test_luminance_black_white(self):
        assert relative_luminance("#000000") == 0.0
        assert relative_luminance("#ffffff") == 1.0

    def test_contrast_ratio(self):
        assert contrast_ratio("#ffffff", "#000000") >= 21.0
        assert contrast_ratio("#000000", "#ffffff") >= 21.0

    def test_passes_aa(self):
        assert passes_aa("#ffffff", "#000000") is True
        assert passes_aa("#000000", "#ffffff") is True
        assert passes_aa("#ffffff", "#ffffff") is False

    def test_large_text_threshold(self):
        # 相同颜色: 大号文本阈值 3:1 也不通过 (0 对比)
        assert passes_aa("#ffffff", "#ffffff", large=True) is False

    def test_audit_contrast(self):
        results = audit_contrast([
            ("good", "#ffffff", "#000000"),
            ("bad", "#ff0000", "#ffffff"),
        ])
        assert results[0]["passes"] is True
        assert results[1]["passes"] is False

    def test_invalid_color(self):
        assert relative_luminance("nope") == 0.0


class TestA11yModes:
    """高对比度 + 色盲友好 + Ishihara."""

    def test_high_contrast_css(self):
        css = high_contrast_css()
        assert "#ffffff" in css
        assert "#000000" in css
        assert "background-image:none" in css  # 禁用渐变

    def test_colorblind_blue_orange(self):
        css = colorblind_css("blue_orange")
        assert "l7-cb-mode" in css
        assert COLORBLIND_BLUE_ORANGE[0] in css

    def test_colorblind_texture(self):
        css = colorblind_css("texture")
        assert "repeating-linear-gradient" in css

    def test_ishihara(self):
        test = ishihara_test()
        assert test["type"] == "ishihara_simplified"
        assert test["recommended_scheme"] == "blue_orange"

    def test_shortcuts(self):
        assert "arrow_left" in KEYBOARD_SHORTCUTS
        assert "escape" in KEYBOARD_SHORTCUTS


class TestAria:
    """ARIA 属性生成."""

    def test_attributes(self):
        attrs = a11y_attributes(role="log", label="对话", live="polite")
        assert 'role="log"' in attrs
        assert 'aria-label="对话"' in attrs
        assert 'aria-live="polite"' in attrs

    def test_empty(self):
        assert a11y_attributes() == ""

    def test_keyshortcuts(self):
        attrs = a11y_attributes(role="application", keyshortcuts="ArrowLeft ArrowRight")
        assert 'aria-keyshortcuts="ArrowLeft ArrowRight"' in attrs

    def test_audit_report(self):
        report = a11y_audit_report([
            {"name": "a", "passed": True},
            {"name": "b", "passed": False},
        ])
        assert report["total"] == 2
        assert report["passed"] == 1
        assert report["passed_ratio"] == 0.5


class TestI18n:
    """国际化."""

    def test_translate_ui(self):
        assert translate("dashboard", "zh-CN") == "学情面板"
        assert translate("dashboard", "en-US") == "Dashboard"

    def test_translate_term(self):
        assert translate("quantum_efficiency", "en-US", term=True) == "Quantum Efficiency"
        assert translate("quantum_efficiency", "zh-CN", term=True) == "量子效率"

    def test_translate_missing(self):
        assert translate("nope_key") == "nope_key"

    def test_glossary(self):
        g = glossary("en-US")
        assert g["upconversion"] == "Upconversion"

    def test_locale_normalize(self):
        assert normalize_locale("en-US") == "en-US"
        assert normalize_locale("fr-FR") == "zh-CN"
        assert normalize_locale(None) == "zh-CN"

    def test_is_supported(self):
        assert is_supported_locale("zh-CN") is True
        assert is_supported_locale("de") is False

    def test_format_number(self):
        assert format_number(1234.5, "en-US") == "1,234.50"
        assert format_number(1234.5, "zh-CN") == "1234.50"

    def test_format_date(self):
        ts = 1600000000.0
        assert "年" in format_date(ts, "zh-CN")
        assert "-" in format_date(ts, "en-US")

    def test_rtl(self):
        assert rtl_css("zh-CN") == ""
        assert "direction:rtl" in rtl_css("ar")

    def test_init_config(self):
        cfg = i18n_init_config("zh-CN")
        assert cfg["lng"] == "zh-CN"
        assert cfg["fallbackLng"] == "zh-CN"
        assert "en-US" in cfg["supportedLngs"]
        assert "translation" in cfg["resources"]["zh-CN"]


class TestMonitoring:
    """性能监控."""

    def test_vitals_ok(self):
        result = check_vitals({"FCP": 1.0, "LCP": 2.0, "FID": 50, "CLS": 0.05})
        assert result["overall"] is True

    def test_vitals_over_budget(self):
        result = check_vitals({"FCP": 3.0, "LCP": 5.0, "FID": 300, "CLS": 0.5})
        assert result["overall"] is False
        fcp = next(m for m in result["metrics"] if m["name"] == "FCP")
        assert fcp["ok"] is False

    def test_render_latency(self):
        ok = check_render_latency("text", 100)
        assert ok["ok"] is True
        bad = check_render_latency("chart", 3000)
        assert bad["ok"] is False
        assert bad["overshoot_ms"] > 0

    def test_ws_latency(self):
        assert check_ws_latency(100) is True
        assert check_ws_latency(500) is False

    def test_tracker(self):
        tracker = PerformanceTracker()
        tracker.mark("start")
        tracker.mark("end")
        result = tracker.measure("text", "start", "end")
        assert result["duration_ms"] >= 0
        report = tracker.report()
        assert len(report["measures"]) == 1

    def test_tracker_missing_marks(self):
        tracker = PerformanceTracker()
        result = tracker.measure("chart", "nope1", "nope2")
        assert result["ok"] is True  # 无数据不告警
