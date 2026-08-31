"""L7 无障碍 — 包入口 (任务拆分 T6)."""

from __future__ import annotations

from .a11y_manager import (
    CONTRAST_LARGE,
    CONTRAST_NORMAL,
    COLORBLIND_BLUE_ORANGE,
    HIGH_CONTRAST_COLORS,
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

__all__ = [
    "CONTRAST_LARGE",
    "CONTRAST_NORMAL",
    "COLORBLIND_BLUE_ORANGE",
    "HIGH_CONTRAST_COLORS",
    "KEYBOARD_SHORTCUTS",
    "a11y_attributes",
    "a11y_audit_report",
    "audit_contrast",
    "colorblind_css",
    "contrast_ratio",
    "high_contrast_css",
    "ishihara_test",
    "passes_aa",
    "relative_luminance",
]
