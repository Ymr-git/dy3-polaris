"""L7 响应式设计 — 布局管理器 (layout_manager.py).

任务拆分 T6 · 设计文档 Ch.8.1。

三断点响应式布局:

| 断点 | 范围 | 布局 | 导航 |
|---|---|---|---|
| 桌面 | ≥1200px | 三栏 240:flex:320 | 顶部 Tab + 侧边 KP 树 |
| 平板 | 768-1199px | 两栏 (抽屉) | 汉堡菜单 + 底部快捷栏 |
| 移动 | <768px | 单栏 | 底部 Tab (4 入口) |

融合世界先进方案:
- CSS Grid + 媒体查询: 声明式断点布局
- 渐进增强: 桌面全功能 → 移动降级
- 最小触控尺寸 44px (WCAG 2.5.5 / Apple HIG)
"""

from __future__ import annotations

from typing import Any

#: 断点定义 (px)
BREAKPOINTS: dict[str, int] = {
    "mobile": 768,
    "tablet": 1200,
}
#: 桌面三栏宽度 (导航 : 主内容 : 侧边栏)
DESKTOP_COLUMNS: tuple[int, int, int] = (240, 1, 320)
#: 最小触控尺寸 (px, Ch.8.1.2)
MIN_TOUCH_SIZE: int = 44


def breakpoint_for(width: int) -> str:
    """按视口宽度判定断点.

    Args:
        width: 视口宽度 px。

    Returns:
        "desktop" / "tablet" / "mobile"。
    """
    if width >= BREAKPOINTS["tablet"]:
        return "desktop"
    if width >= BREAKPOINTS["mobile"]:
        return "tablet"
    return "mobile"


def layout_plan(width: int) -> dict[str, Any]:
    """生成断点对应的布局计划 (供前端挂载).

    Args:
        width: 视口宽度 px。

    Returns:
        {breakpoint, columns, nav_mode, charts_mode, touch_size,
         grid_template, transitions}
    """
    if width < 0:
        width = 0
    bp = breakpoint_for(width)
    if bp == "desktop":
        grid = f"{DESKTOP_COLUMNS[0]}px minmax(0,{DESKTOP_COLUMNS[1]}fr) {DESKTOP_COLUMNS[2]}px"
        return {
            "breakpoint": bp,
            "columns": list(DESKTOP_COLUMNS),
            "nav_mode": "top_tabs_sidebar",
            "charts_mode": "full_interactive",
            "touch_size": MIN_TOUCH_SIZE,
            "grid_template": grid,
            "mobile_tabs": False,
        }
    if bp == "tablet":
        return {
            "breakpoint": bp,
            "columns": [1, 1],
            "nav_mode": "drawer_hamburger",
            "charts_mode": "touch_optimized",
            "touch_size": MIN_TOUCH_SIZE,
            "grid_template": "minmax(0,1fr) minmax(0,1fr)",
            "mobile_tabs": True,
        }
    return {
        "breakpoint": bp,
        "columns": [1],
        "nav_mode": "bottom_tabs",
        "charts_mode": "static_with_drill",
        "touch_size": MIN_TOUCH_SIZE,
        "grid_template": "minmax(0,1fr)",
        "mobile_tabs": True,
    }


def responsive_css() -> str:
    """生成响应式 CSS (三断点 + 44px 触控 + 平滑过渡).

    桌面: 三栏 240px / flex / 320px。
    平板: 两栏, 侧边栏折叠为抽屉。
    移动: 单栏, 底部 Tab 固定。
    """
    return f"""
<style class="l7-responsive">
.l7-app{{display:grid;gap:0;transition:grid-template-columns .3s ease-out;min-height:100vh}}
.l7-app.desktop{{grid-template-columns:{DESKTOP_COLUMNS[0]}px minmax(0,1fr) {DESKTOP_COLUMNS[2]}px}}
.l7-app.tablet{{grid-template-columns:minmax(0,1fr)}}
.l7-app.mobile{{grid-template-columns:minmax(0,1fr)}}
.l7-app .l7-nav{{min-width:0;overflow-x:hidden}}
.l7-app .l7-main{{min-width:0}}
.l7-app .l7-side{{min-width:0}}
@media (max-width:{BREAKPOINTS['mobile'] - 1}px){{
  .l7-app.desktop{{grid-template-columns:minmax(0,1fr)}}
  .l7-app .l7-nav{{display:none}}
  .l7-app .l7-side{{display:none}}
  .l7-tabbar{{display:flex}}
  .l7-chart,.l7-molecule{{touch-action:manipulation}}
}}
@media (min-width:{BREAKPOINTS['mobile']}px) and (max-width:{BREAKPOINTS['tablet'] - 1}px){{
  .l7-app .l7-side{{display:none}}
  .l7-tabbar{{display:flex}}
}}
@media (min-width:{BREAKPOINTS['tablet']}px){{
  .l7-app .l7-nav{{display:block}}
  .l7-app .l7-side{{display:block}}
  .l7-tabbar{{display:none}}
}}
.l7-tabbar{{position:fixed;bottom:0;left:0;right:0;display:none;z-index:100;
  border-top:1px solid var(--l7-border, rgba(23,23,23,.12));background:var(--l7-surface, #fff)}}
.l7-tabbar button{{flex:1;min-height:{MIN_TOUCH_SIZE}px;border:none;background:transparent;
  font-size:12px;cursor:pointer;display:flex;flex-direction:column;align-items:center;gap:2px}}
.l7-touch-target{{min-width:{MIN_TOUCH_SIZE}px;min-height:{MIN_TOUCH_SIZE}px}}
@media (prefers-reduced-motion: reduce){{
  .l7-app{{transition:none}}
  *{{animation-duration:.01ms !important;animation-iteration-count:1 !important}}
}}
</style>
"""


def render_layout(width: int, theme: str = "light") -> dict[str, Any]:
    """渲染响应式布局骨架 (含断点样式与结构).

    Args:
        width: 视口宽度 px。
        theme: light / dark。

    Returns:
        {html, config, css} 三部分。
    """
    plan = layout_plan(width)
    bp = plan["breakpoint"]
    nav_entries = {
        "desktop": '<nav class="l7-nav" aria-label="主导航"><div class="l7-kp-tree" role="tree">'
                   '<div role="treeitem" aria-expanded="true">A 域</div>'
                   '<div role="treeitem" aria-expanded="false">B 域</div></div></nav>',
        "tablet": '<button class="l7-hamburger l7-touch-target" aria-label="打开导航菜单" aria-expanded="false">☰</button>',
        "mobile": "",
    }
    side_entries = {
        "desktop": '<aside class="l7-side" aria-label="学情概览"><div class="l7-mini-mastery">总体掌握度</div></aside>',
        "tablet": "",
        "mobile": "",
    }
    tabbar = (
        '<nav class="l7-tabbar" aria-label="底部导航">'
        '<button aria-label="学习">📖<span>学习</span></button>'
        '<button aria-label="学情">📊<span>学情</span></button>'
        '<button aria-label="探索">🔍<span>探索</span></button>'
        '<button aria-label="我的">👤<span>我的</span></button>'
        "</nav>"
    )
    html = (
        f'<div class="l7-app {bp}" data-breakpoint="{bp}">'
        + nav_entries[bp]
        + '<main class="l7-main" id="l7-main" role="main">'
        + f'<p class="l7-breakpoint-hint">断点: {bp} · 宽度 {width}px</p>'
        + "</main>"
        + side_entries[bp]
        + "</div>"
        + tabbar
    )
    return {
        "html": html,
        "config": plan,
        "css": responsive_css(),
    }


def responsive_layout_wrap(content: str, css_class: str = "l7-responsive-panel", theme: str = "light") -> str:
    """统一包装: 基础主题 + 响应式样式 + 内容."""
    from ..renderers._common import theme_css

    return (
        f'<div class="l7-render {css_class}" data-theme="{theme}">'
        f"{theme_css(theme)}{responsive_css()}{content}</div>"
    )
