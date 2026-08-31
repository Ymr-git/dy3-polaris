"""L7 国际化 — 国际化框架 (i18n_setup.py).

任务拆分 T6 · 设计文档 Ch.8.3。

基于 i18next 语义的中英文切换框架:
- 中文默认, 英文本地化
- 学科术语表 (Dy3+ 发光材料术语中英对照)
- 日期/数字本地化
- RTL 布局预留
"""

from __future__ import annotations

from typing import Any

#: 默认语言
DEFAULT_LOCALE: str = "zh-CN"

#: 支持的语言
SUPPORTED_LOCALES: list[str] = ["zh-CN", "en-US"]

#: 学科术语表 (Ch.8.3: Dy3+ 发光材料术语中英对照)
TERMS: dict[str, dict[str, str]] = {
    "zh-CN": {
        "quantum_efficiency": "量子效率",
        "concentration_quenching": "浓度猝灭",
        "thermal_quenching": "热猝灭",
        "host_material": "宿主材料",
        "energy_transfer": "能量传递",
        "upconversion": "上转换",
        "downconversion": "下转换",
        "lifetime": "荧光寿命",
        "emission_spectrum": "发射光谱",
        "excitation_spectrum": "激发光谱",
        "crystal_field": "晶体场",
        "jablonski": "雅布隆斯基图",
    },
    "en-US": {
        "quantum_efficiency": "Quantum Efficiency",
        "concentration_quenching": "Concentration Quenching",
        "thermal_quenching": "Thermal Quenching",
        "host_material": "Host Material",
        "energy_transfer": "Energy Transfer",
        "upconversion": "Upconversion",
        "downconversion": "Downconversion",
        "lifetime": "Luminescence Lifetime",
        "emission_spectrum": "Emission Spectrum",
        "excitation_spectrum": "Excitation Spectrum",
        "crystal_field": "Crystal Field",
        "jablonski": "Jablonski Diagram",
    },
}

#: UI 文本 key → 各语言翻译 (默认中文, 英文次之)
UI_STRINGS: dict[str, dict[str, str]] = {
    "dashboard": {"zh-CN": "学情面板", "en-US": "Dashboard"},
    "learn": {"zh-CN": "学习", "en-US": "Learn"},
    "explore": {"zh-CN": "探索", "en-US": "Explore"},
    "profile": {"zh-CN": "我的", "en-US": "Profile"},
    "approve": {"zh-CN": "批准", "en-US": "Approve"},
    "reject": {"zh-CN": "拒绝", "en-US": "Reject"},
    "modify": {"zh-CN": "修改", "en-US": "Modify"},
    "mastery": {"zh-CN": "掌握度", "en-US": "Mastery"},
    "bottleneck": {"zh-CN": "瓶颈", "en-US": "Bottleneck"},
    "loading": {"zh-CN": "加载中...", "en-US": "Loading..."},
    "retry": {"zh-CN": "重试", "en-US": "Retry"},
}


def is_supported_locale(locale: str) -> bool:
    """判断语言是否受支持."""
    return locale in SUPPORTED_LOCALES


def normalize_locale(locale: str | None) -> str:
    """归一化语言代码 (未支持回退中文默认)."""
    if locale and locale in SUPPORTED_LOCALES:
        return locale
    return DEFAULT_LOCALE


def translate(key: str, locale: str = DEFAULT_LOCALE, term: bool = False) -> str:
    """翻译文本.

    Args:
        key: i18n key 或术语 key。
        locale: 目标语言。
        term: True 时查学科术语表 (TERMS 结构为 {locale: {term: text}}),
            False 查 UI 文本表 (UI_STRINGS 结构为 {key: {locale: text}})。

    Returns:
        翻译结果, 缺失回退 key 本身。
    """
    if term:
        table = TERMS.get(normalize_locale(locale), TERMS[DEFAULT_LOCALE])
        return table.get(key, key)
    entry = UI_STRINGS.get(key, {})
    if not entry:
        return key
    return entry.get(locale, entry.get(DEFAULT_LOCALE, key))


def glossary(locale: str = DEFAULT_LOCALE) -> dict[str, str]:
    """返回学科术语表 (指定语言)."""
    return dict(TERMS.get(normalize_locale(locale), TERMS[DEFAULT_LOCALE]))


def format_number(value: float, locale: str = DEFAULT_LOCALE, digits: int = 2) -> str:
    """数字本地化 (千分位/小数点)."""
    text = f"{value:.{digits}f}"
    if locale == "en-US":
        int_part, _, frac = text.partition(".")
        int_part = "".join(
            ("," if i and (len(int_part) - i) % 3 == 0 else "") + ch
            for i, ch in enumerate(int_part)
        )
        return f"{int_part}.{frac}" if frac else int_part
    return text


def format_date(timestamp: float, locale: str = DEFAULT_LOCALE) -> str:
    """日期本地化."""
    import time

    if locale == "en-US":
        return time.strftime("%Y-%m-%d", time.localtime(timestamp))
    return time.strftime("%Y年%m月%d日", time.localtime(timestamp))


def rtl_css(locale: str = DEFAULT_LOCALE) -> str:
    """RTL 布局预留 (Ch.8.3).

    Returns:
        RTL CSS (仅阿拉伯语等 RTL 语言启用)。
    """
    if locale in ("ar", "he", "fa", "ur"):
        return (
            "<style class=\"l7-rtl\">"
            ".l7-rtl-mode{direction:rtl;text-align:right}"
            "</style>"
        )
    return ""


def i18n_init_config(locale: str = DEFAULT_LOCALE) -> dict[str, Any]:
    """生成 i18next 初始化配置 (供前端注入).

    Args:
        locale: 初始语言。

    Returns:
        {lng, fallbackLng, supportedLngs, resources} i18next 配置。
    """
    locale = normalize_locale(locale)
    resources: dict[str, dict[str, dict[str, str]]] = {}
    for loc in SUPPORTED_LOCALES:
        resources[loc] = {
            "translation": {
                key: UI_STRINGS.get(key, {}).get(loc, key)
                for key in UI_STRINGS
            }
        }
    return {
        "lng": locale,
        "fallbackLng": DEFAULT_LOCALE,
        "supportedLngs": SUPPORTED_LOCALES,
        "resources": resources,
    }
