"""L7 无障碍 — 无障碍管理器 (a11y_manager.py).

任务拆分 T6 · 设计文档 Ch.8.2。

WCAG 2.1 AA 合规管理器:

1. 对比度计算 — WCAG 相对亮度公式, AA 阈值 4.5:1 (正文) / 3:1 (大号)
2. 高对比度模式 — 纯白/纯黑/纯蓝, 禁用渐变 (Ch.8.2.3)
3. 色盲友好配色 — 蓝-橙方案 + 纹理方案 (Ch.8.2.4)
4. ARIA 生成 — role/aria-label/aria-live/aria-keyshortcuts
5. 键盘导航配置 — 快捷键 schema (方向键/+/-/Enter/Esc)

融合世界先进方案:
- WCAG 2.1 AA 对比度数学公式 (W3C)
- prefers-reduced-motion / focus-visible 现代 CSS
- ARIA 最佳实践: 原生元素优先, ARIA 补缺口
"""

from __future__ import annotations

from typing import Any

#: 对比度阈值 (WCAG 2.1 AA)
CONTRAST_NORMAL: float = 4.5
CONTRAST_LARGE: float = 3.0


def relative_luminance(hex_color: str) -> float:
    """WCAG 相对亮度计算.

    Args:
        hex_color: "#RRGGBB" 十六进制颜色。

    Returns:
        0.0 (黑) ~ 1.0 (白)。
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return 0.0
    try:
        rgb = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except ValueError:
        return 0.0
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in rgb
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG 对比度 (fg 与 bg 的亮度比)."""
    l1 = relative_luminance(fg)
    l2 = relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def passes_aa(fg: str, bg: str, large: bool = False) -> bool:
    """判断是否通过 WCAG 2.1 AA 对比度.

    Args:
        fg: 前景色。
        bg: 背景色。
        large: 是否大号文本 (≥18pt 或 14pt 加粗)。

    Returns:
        True 表示对比度达标。
    """
    threshold = CONTRAST_LARGE if large else CONTRAST_NORMAL
    return contrast_ratio(fg, bg) >= threshold


def audit_contrast(pairs: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """批量审计对比度.

    Args:
        pairs: [(label, fg, bg)] 列表。

    Returns:
        [{label, fg, bg, ratio, passes}] 审计结果。
    """
    results = []
    for label, fg, bg in pairs:
        ratio = round(contrast_ratio(fg, bg), 2)
        results.append({
            "label": label,
            "fg": fg,
            "bg": bg,
            "ratio": ratio,
            "passes": ratio >= CONTRAST_NORMAL,
        })
    return results


#: 高对比度模式配色 (Ch.8.2.3: 纯白/纯黑/纯蓝, 禁用渐变)
HIGH_CONTRAST_COLORS: dict[str, str] = {
    "bg": "#ffffff",
    "text": "#000000",
    "link": "#0000ff",
}


def high_contrast_css() -> str:
    """高对比度模式 CSS (禁用渐变/半透明)."""
    c = HIGH_CONTRAST_COLORS
    return f"""
<style class="l7-hc">
.l7-hc-mode{{background:{c['bg']} !important;color:{c['text']} !important}}
.l7-hc-mode *{{background-image:none !important;box-shadow:none !important;text-shadow:none !important;opacity:1 !important}}
.l7-hc-mode a,.l7-hc-mode .l7-hc-link{{color:{c['link']} !important;text-decoration:underline}}
.l7-hc-mode .l7-hc-border{{border:1px solid {c['text']} !important}}
</style>
"""


#: 色盲友好配色 (Ch.8.2.4)
COLORBLIND_BLUE_ORANGE: list[str] = ["#2563eb", "#0ea5e9", "#f97316", "#fbbf24"]
#: 纹理方案 CSS (无法分辨颜色也能区分数值)
_TEXTURE_CSS: str = """
.l7-texture-s{background-image:repeating-linear-gradient(45deg,currentColor 0 2px,transparent 2px 6px)}
.l7-texture-d{background-image:radial-gradient(currentColor 1.5px,transparent 1.5px)}
.l7-texture-g{background-image:repeating-linear-gradient(0deg,currentColor 0 2px,transparent 2px 4px),repeating-linear-gradient(90deg,currentColor 0 2px,transparent 2px 4px)}
"""


def colorblind_css(scheme: str = "blue_orange") -> str:
    """生成色盲友好配色 CSS.

    Args:
        scheme: blue_orange / texture。

    Returns:
        CSS 片段。
    """
    if scheme == "texture":
        return f"<style class=\"l7-cb\">{_TEXTURE_CSS}</style>"
    palette = COLORBLIND_BLUE_ORANGE
    return (
        "<style class=\"l7-cb\">"
        ".l7-cb-mode{--l7-cb-1:" + palette[0] + ";--l7-cb-2:" + palette[1]
        + ";--l7-cb-3:" + palette[2] + ";--l7-cb-4:" + palette[3] + "}"
        "</style>"
    )


def ishihara_test() -> dict[str, Any]:
    """Ishihara 色觉检测板简化版 (Ch.8.2.4).

    Returns:
        {colors, question, recommended_scheme} 检测配置。
    """
    return {
        "type": "ishihara_simplified",
        "question": "请选出你看到的数字",
        "options": [
            {"digits": "12", "colors": ["#2563eb", "#f97316"]},
            {"digits": "74", "colors": ["#16a34a", "#ef4444"]},
        ],
        "recommended_scheme": "blue_orange",  # 色盲友好默认推荐
    }


#: 键盘快捷键 schema (Ch.8.2.2)
KEYBOARD_SHORTCUTS: dict[str, str] = {
    "arrow_left": "移动焦点/上一步",
    "arrow_right": "移动焦点/下一步",
    "arrow_up": "上移",
    "arrow_down": "下移",
    "plus": "放大图表",
    "minus": "缩小图表",
    "enter": "展开详情",
    "escape": "关闭面板",
}


def a11y_attributes(role: str = "", label: str = "", live: str = "", keyshortcuts: str = "") -> str:
    """生成 ARIA 属性串.

    Args:
        role: ARIA role。
        label: aria-label (自动 HTML 转义防 XSS)。
        live: aria-live (polite/assertive)。
        keyshortcuts: aria-keyshortcuts。

    Returns:
        ARIA 属性 HTML 片段。
    """
    from ..renderers._common import esc

    parts = []
    if role:
        parts.append(f'role="{role}"')
    if label:
        parts.append(f'aria-label="{esc(label)}"')
    if live:
        parts.append(f'aria-live="{live}"')
    if keyshortcuts:
        parts.append(f'aria-keyshortcuts="{esc(keyshortcuts)}"')
    return " ".join(parts)


def a11y_audit_report(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总无障碍审计结果.

    Args:
        checks: [{name, passed, detail}] 检查项。

    Returns:
        {total, passed, failed, passed_ratio, checks} 报告。
    """
    passed = sum(1 for c in checks if c.get("passed"))
    total = len(checks)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "passed_ratio": round(passed / total, 4) if total else 1.0,
        "checks": checks,
    }
