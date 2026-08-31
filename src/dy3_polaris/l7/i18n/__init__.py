"""L7 国际化 — 包入口 (任务拆分 T6)."""

from __future__ import annotations

from .i18n_setup import (
    DEFAULT_LOCALE,
    SUPPORTED_LOCALES,
    TERMS,
    UI_STRINGS,
    format_date,
    format_number,
    glossary,
    i18n_init_config,
    is_supported_locale,
    normalize_locale,
    rtl_css,
    translate,
)

__all__ = [
    "DEFAULT_LOCALE",
    "SUPPORTED_LOCALES",
    "TERMS",
    "UI_STRINGS",
    "format_date",
    "format_number",
    "glossary",
    "i18n_init_config",
    "is_supported_locale",
    "normalize_locale",
    "rtl_css",
    "translate",
]
