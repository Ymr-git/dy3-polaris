"""L7 渲染器公共工具 — 领域常量、BKT 学情、HTML 与描述符辅助.

七大 Native Renderer (Text/Chart/Graph/Molecule/Table/Formula/Provenance)
共享的契约层，提供:

1. **42 KP 领域常量** — 依据 L2 个性化设计文档 (§2.1.1) 的四域划分:
   A 能级跃迁理论 (13) / B 材料体系设计 (11) / C 合成制备工艺 (10) / D 表征测试技术 (8)。
2. **BKT 学情状态提取** — 从 Artifact.learner_context 与 RenderContext.bkt_state
   中提取掌握度四参数 (P(L)/P(K|L)/P(G)/P(S))。
3. **学情着色** — P(L)>0.8 绿 / 0.5-0.8 靛蓝 / <0.5 琥珀 / 虚假掌握红色脉冲，
   与设计文档 §2.4.2 GraphRenderer 着色规则一致。
4. **HTML 工具** — 转义、主题 CSS 变量注入、响应式容器包裹。
5. **描述符构建** — 统一构造 RenderDescriptor (html/config/assets/metadata)，
   记录渲染耗时，供各渲染器 do_render 复用。

融合世界先进方案:
    - Vega-Lite / Altair: 编码通道 (encoding) 与语义类型推断的字段约定
    - Jupyter MIME Bundle: 多模态输出契约
    - WCAG 2.1: 对比度友好的着色体系 (light/dark 双主题)
"""

from __future__ import annotations

import html as _html
import time
from typing import Any

from ..models import Artifact, RenderContext, RenderDescriptor

# ============================================================
# 42 KP 领域常量 — 单点收敛: 由 L2 kp_catalog 派生 (SSOT)
# 消除"展示层持有领域目录"的架构倒挂; 本层仅 re-export 保持旧 API 兼容
# ============================================================

from dy3_polaris.l2.kp_catalog import (  # noqa: E402  (单点目录)
    ALL_KP_IDS,
    DOMAIN_LABELS,
    KP_DOMAIN_IDS,
    KP_LEVELS,
    KP_NAMES,
    KP_TO_DOMAIN,
    NODE_TO_KP,
)

#: BKT 四参数键名
BKT_PARAM_KEYS: tuple[str, ...] = ("p_l", "p_k_l", "p_g", "p_s")


def kp_name(kp_id: str) -> str:
    """返回 KP 名称，未知 KP 回退到 ID 本身."""
    return KP_NAMES.get(kp_id, kp_id)


# ============================================================
# BKT 学情状态提取
# ============================================================

def get_bkt_state(artifact: Artifact, context: RenderContext) -> dict[str, Any]:
    """合并提取 BKT 掌握度状态.

    优先级: ``context.bkt_state`` (渲染时快照) > ``artifact.learner_context['bkt_state']``。
    返回 ``{kp_id: {p_l, p_k_l, p_g, p_s}}`` 结构。

    Args:
        artifact: 待渲染的 Artifact (携带 learner_context)。
        context: 渲染上下文 (携带 bkt_state)。

    Returns:
        BKT 状态字典，无状态时返回空字典。
    """
    merged: dict[str, Any] = {}
    for source in (
        (context.bkt_state or {}) if context else {},
        ((artifact.learner_context or {}).get("bkt_state") or {}) if artifact else {},
    ):
        if isinstance(source, dict):
            for kp_id, state in source.items():
                # 忽略非 dict 值 (如 colorblind 等元数据键)
                if isinstance(state, dict):
                    merged.setdefault(kp_id, {}).update(state)
    return merged


def get_kp_state(
    bkt_state: dict[str, Any] | None, kp_id: str
) -> dict[str, float] | None:
    """提取单个 KP 的 BKT 参数.

    Args:
        bkt_state: BKT 状态字典 (get_bkt_state 产出)。
        kp_id: 知识点 ID。

    Returns:
        含 p_l/p_k_l/p_g/p_s 的字典 (缺失参数以 0.0 补全)，
        KP 不存在或状态为空时返回 None。
    """
    if not bkt_state:
        return None
    raw = bkt_state.get(kp_id)
    if not isinstance(raw, dict) or not raw:
        return None
    state = {k: float(raw.get(k, 0.0)) for k in BKT_PARAM_KEYS}
    return state


def average_p_l(bkt_state: dict[str, Any]) -> float:
    """计算全部已追踪 KP 的平均掌握概率 P(L).

    Args:
        bkt_state: BKT 状态字典。

    Returns:
        平均 P(L)，无状态时返回 0.0。
    """
    values = []
    for state in (bkt_state or {}).values():
        if isinstance(state, dict) and state.get("p_l") is not None:
            try:
                values.append(float(state["p_l"]))
            except (TypeError, ValueError):
                continue
    if not values:
        return 0.0
    return sum(values) / len(values)


def is_bottleneck(state: dict[str, float] | None) -> bool:
    """虚假掌握 (瓶颈) 检测 — P(L)>0.7 但 P(K|L)<0.3.

    依据设计文档 §5.1: "P(L)>0.7 且 P(K|L)<0.3 的虚假掌握"。

    Args:
        state: 单个 KP 的 BKT 参数。

    Returns:
        True 表示该 KP 为学习瓶颈。
    """
    if not state:
        return False
    return bool(state.get("p_l", 0.0) > 0.7 and state.get("p_k_l", 1.0) < 0.3)


# ============================================================
# 学情着色 (与设计文档 §2.4.2 一致)
# ============================================================

#: 掌握度档位 → (light 主题颜色, dark 主题颜色)
_MASTERY_PALETTE: dict[str, tuple[str, str]] = {
    "mastered": ("#16a34a", "#4ade80"),  # 绿 — P(L)>0.8
    "learning": ("#4b3fe3", "#818cf8"),  # 靛蓝 — 0.5<=P(L)<=0.8
    "weak": ("#d97706", "#fbbf24"),  # 琥珀 — P(L)<0.5
    "bottleneck": ("#dc2626", "#f87171"),  # 红 — 虚假掌握 (脉冲)
}


def mastery_level(p_l: float | None) -> str:
    """将 P(L) 映射为掌握度档位.

    Args:
        p_l: 掌握概率 (0~1)。

    Returns:
        mastered / learning / weak。
    """
    if p_l is None:
        return "weak"
    if p_l > 0.8:
        return "mastered"
    if p_l >= 0.5:
        return "learning"
    return "weak"


def mastery_color(
    p_l: float | None, theme: str = "light", colorblind: bool = False
) -> str:
    """返回掌握度对应的 CSS 颜色.

    Args:
        p_l: 掌握概率 (0~1)。
        theme: light / dark。
        colorblind: 色盲友好模式 (蓝橙方案)。

    Returns:
        CSS 颜色值。
    """
    level = mastery_level(p_l)
    if colorblind:
        # 蓝-橙友好方案: 以形状/明度差异为主
        return {
            "mastered": ("#2563eb", "#60a5fa"),
            "learning": ("#0d9488", "#2dd4bf"),
            "weak": ("#ea580c", "#fb923c"),
            "bottleneck": ("#7c3aed", "#a78bfa"),
        }[level][0 if theme == "light" else 1]
    return _MASTERY_PALETTE[level][0 if theme == "light" else 1]


def apply_mastery_style(
    p_l: float | None,
    theme: str = "light",
    extra: str = "",
    is_bn: bool = False,
) -> str:
    """生成节点/单元格的内联样式 (含瓶颈脉冲动画类).

    Args:
        p_l: 掌握概率。
        theme: light / dark。
        extra: 额外样式片段。
        is_bn: 是否为瓶颈 KP (附加红色脉冲)。

    Returns:
        ``style="..."`` 内联样式片段。
    """
    color = mastery_color(p_l, theme)
    style = f"border-color:{color};color:{color}"
    if extra:
        style = f"{style};{extra}"
    cls = ' class="bkt-bottleneck-pulse"' if is_bn else ""
    return f'style="{style}"{cls}'


# ============================================================
# HTML 工具
# ============================================================

def esc(text: Any) -> str:
    """HTML 转义 (None → 空串)."""
    if text is None:
        return ""
    return _html.escape(str(text), quote=True)


def theme_css(theme: str = "light") -> str:
    """注入主题 CSS 变量 (L7 渲染容器通用).

    Args:
        theme: light / dark。

    Returns:
        ``<style>`` 片段，定义渲染容器的 CSS 变量与基础样式。
    """
    if theme == "dark":
        vars_ = (
            "--l7-bg:#171717;--l7-text:#e5e5e5;--l7-muted:#a1a1aa;"
            "--l7-border:rgba(229,229,229,0.14);--l7-surface:#262626"
        )
    else:
        vars_ = (
            "--l7-bg:#ffffff;--l7-text:#171717;--l7-muted:#52525b;"
            "--l7-border:rgba(23,23,23,0.12);--l7-surface:#f7f7f8"
        )
    return (
        "<style>"
        f".l7-render{{display:block;{vars_};color:var(--l7-text);"
        "font-family:'SF Pro Text','PingFang SC',system-ui,sans-serif;"
        "font-size:14px;line-height:1.6}"
        ".l7-render *{{box-sizing:border-box}}"
        "</style>"
    )


def wrap(content: str, css_class: str = "", theme: str = "light") -> str:
    """将内容包裹进统一渲染容器 (含主题 CSS)."""
    classes = ("l7-render" + (f" {css_class}" if css_class else "")).strip()
    return (
        f'<div class="{classes}">{theme_css(theme)}{content}</div>'
    )


def kp_badge(kp_id: str, state: dict[str, float] | None, theme: str = "light") -> str:
    """生成带掌握度着色的 KP 徽章.

    Args:
        kp_id: KP ID。
        state: 该 KP 的 BKT 参数 (可为 None)。
        theme: 主题。

    Returns:
        ``<span class="kp-badge">`` HTML 片段。
    """
    p_l = state.get("p_l") if state else None
    is_bn = is_bottleneck(state)
    style = apply_mastery_style(p_l, theme, is_bn=is_bn)
    pct = f"{p_l * 100:.0f}%" if p_l is not None else "未追踪"
    return (
        f'<span class="kp-badge" title="{esc(kp_name(kp_id))} · P(L)={pct}" {style}>'
        f"{esc(kp_id)}</span>"
    )


# ============================================================
# 描述符构建
# ============================================================

def build_descriptor(
    artifact: Artifact,
    html: str | None = None,
    config: dict[str, Any] | None = None,
    assets: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    renderer_name: str = "",
) -> RenderDescriptor:
    """统一构建 RenderDescriptor (含渲染耗时记录).

    Args:
        artifact: 被渲染的 Artifact。
        html: 渲染产出的 HTML 片段。
        config: 前端渲染配置 (ECharts option / vis.js options 等)。
        assets: 前端依赖的静态资源 URL。
        metadata: 渲染元数据 (会合并 renderer 名与时间戳)。
        renderer_name: 渲染器类名。

    Returns:
        组装完成的 RenderDescriptor。
    """
    meta: dict[str, Any] = dict(metadata or {})
    if renderer_name:
        meta["renderer"] = renderer_name
    meta.setdefault("rendered_at", time.time())
    return RenderDescriptor(
        artifact_id=artifact.artifact_id,
        mime=artifact.mime,
        html=html,
        config=config or {},
        assets=assets or [],
        metadata=meta,
        rendered_at=time.time(),
        render_time_ms=0.0,
    )


def finalize(
    descriptor: RenderDescriptor, started_at: float
) -> RenderDescriptor:
    """填充渲染耗时并返回描述符.

    Args:
        descriptor: 待填充的描述符。
        started_at: 渲染开始时间戳 (time.monotonic)。

    Returns:
        已填充 render_time_ms 的描述符。
    """
    descriptor.render_time_ms = round((time.monotonic() - started_at) * 1000, 2)
    return descriptor
