"""L7 学情面板 — 公共数据提取层 (dashboard/_common.py).

任务拆分 T4 · 设计文档 Ch.5 (学情可视化面板)。

提供 7 个面板/交互组件共享的数据提取与聚合能力:

1. **BKT 状态提取** — 从 Artifact/RenderContext 提取完整 42 KP 掌握度矩阵
2. **域级聚合** — A/B/C/D 四域汇总（平均掌握度、数量分布）
3. **瓶颈检测** — P(L)>0.7 且 P(K|L)<0.3 的虚假掌握 KP
4. **薄弱点排序** — 综合 P(L) + 被依赖度 + 距上次学习时间 + 瓶颈系数
5. **学习路径推荐排序** — 前置条件过滤 + BKT 加权 + 被依赖度
6. **ECharts 配置构建** — 热力图/条形图/雷达图/折线图/环形进度条的
   声明式 option 构建器，供面板模块复用
"""

from __future__ import annotations

import time
from typing import Any

from ..renderers._common import (
    ALL_KP_IDS,
    BKT_PARAM_KEYS,
    DOMAIN_LABELS,
    KP_DOMAIN_IDS,
    KP_LEVELS,
    KP_NAMES,
    KP_TO_DOMAIN,
    average_p_l,
    get_bkt_state,
    get_kp_state,
    is_bottleneck,
    mastery_color,
    mastery_level,
)
from ..renderers._common import build_descriptor as _build_desc
from ..renderers._common import wrap as _wrap
from ..models import Artifact, RenderContext, RenderDescriptor

# ============================================================
# 数据提取
# ============================================================


def extract_bkt_matrix(
    artifact: Artifact | None,
    context: RenderContext | None,
    kp_ids: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """提取 42 KP BKT 四参数矩阵 (rows=KP, cols=params).

    Args:
        artifact: 关联 Artifact。
        context: 渲染上下文。
        kp_ids: 指定 KP 子集 (默认全部 42 KP)。

    Returns:
        {kp_id: {p_l, p_k_l, p_g, p_s}}, 缺失的 KP 以零值补全。
    """
    bkt = get_bkt_state(artifact, context) if artifact or context else {}
    kp_ids = kp_ids or ALL_KP_IDS
    zero = {k: 0.0 for k in BKT_PARAM_KEYS}
    matrix: dict[str, dict[str, float]] = {}
    for kp_id in kp_ids:
        state = get_kp_state(bkt, kp_id)
        if state is None:
            state = dict(zero)
        matrix[kp_id] = state
    return matrix


def domain_aggregates(
    matrix: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """域级聚合 — A/B/C/D 四域汇总.

    Returns:
        {domain: {label, kp_count, avg_p_l, mastered, learning, weak, bottlenecks}}
    """
    result: dict[str, dict[str, Any]] = {}
    for domain, ids in KP_DOMAIN_IDS.items():
        values = []
        mastered = learning = weak = bottlenecks = 0
        for kp_id in ids:
            state = matrix.get(kp_id, {})
            p_l = state.get("p_l", 0.0)
            if p_l > 0.8:
                mastered += 1
            elif p_l >= 0.5:
                learning += 1
            elif p_l > 0:
                weak += 1
            if is_bottleneck(state):
                bottlenecks += 1
            if p_l > 0:
                values.append(p_l)
        result[domain] = {
            "label": DOMAIN_LABELS.get(domain, domain),
            "kp_count": len(ids),
            "avg_p_l": round(sum(values) / len(values), 4) if values else 0.0,
            "mastered": mastered,
            "learning": learning,
            "weak": weak,
            "bottlenecks": bottlenecks,
        }
    return result


def bottlenecks(
    matrix: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """提取瓶颈 KP 列表 (P(L)>0.7 且 P(K|L)<0.3).

    Returns:
        [{kp_id, name, p_l, p_k_l, severity: 0~1}, ...] 按严重度降序。
    """
    bns: list[dict[str, Any]] = []
    for kp_id, state in matrix.items():
        if is_bottleneck(state):
            severity = (0.7 - state.get("p_k_l", 0.0)) * 3.33
            bns.append({
                "kp_id": kp_id,
                "name": KP_NAMES.get(kp_id, kp_id),
                "p_l": round(state["p_l"], 4),
                "p_k_l": round(state["p_k_l"], 4),
                "severity": round(min(max(severity, 0.0), 1.0), 4),
            })
    bns.sort(key=lambda x: x["severity"], reverse=True)
    return bns


def weak_points(
    matrix: dict[str, dict[str, float]],
    dependencies: dict[str, int] | None = None,
    last_times: dict[str, float] | None = None,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """薄弱点列表 (设计文档 §5.2.3) — 综合排序.

    排序公式 = w1*P(L) (越低越急) + w2*被依赖度 (越高越急)
                + w3*距上次学习时间 + w4*瓶颈系数。

    Args:
        matrix: BKT 矩阵。
        dependencies: KP→依赖数 (被其他 KP 引用的次数)。
        last_times: KP→上次学习时间戳 (秒)。
        top_n: 返回前 N 条。

    Returns:
        按紧急度降序排列的薄弱点列表。
    """
    deps = dependencies or {}
    ltimes = last_times or {}
    now = time.time()
    scored: list[dict[str, Any]] = []
    for kp_id, state in matrix.items():
        p_l = state.get("p_l", 1.0)
        if p_l >= 0.85:
            continue
        dep = deps.get(kp_id, 0)
        lt = ltimes.get(kp_id, 0.0)
        days_since = (now - lt) / 86400.0 if lt > 0 else 30.0
        bn_factor = max(0.0, (0.7 - state.get("p_k_l", 1.0)) * 3.0) if p_l > 0.7 else 0.0
        score = (
            (1.0 - p_l) * 0.4
            + min(dep / 10.0, 1.0) * 0.25
            + min(days_since / 30.0, 1.0) * 0.2
            + bn_factor * 0.15
        )
        scored.append({
            "kp_id": kp_id,
            "name": KP_NAMES.get(kp_id, kp_id),
            "p_l": round(p_l, 4),
            "dependents": dep,
            "days_since_last": round(days_since, 1),
            "score": round(score, 4),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def learning_path(
    matrix: dict[str, dict[str, float]],
    prerequisites: dict[str, list[str]] | None = None,
    dependents: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """学习路径推荐 (设计文档 §5.2.4).

    优先推荐: 前置条件全部满足 (P(L)>0.6) 且自身 P(L) 最低的 KP。
    多个候选时按被依赖度排序 (优先学习被更多 KP 依赖的核心 KP)。

    Returns:
        [{kp_id, name, p_l, is_ready, next_to_learn, dependents}, ...] 按推荐顺序。
    """
    prereqs = prerequisites or {}
    deps = dependents or {}
    # 过滤: 自身 P(L)<0.75 且前置全部满足
    candidates: list[tuple[str, float, int]] = []
    for kp_id, state in matrix.items():
        p_l = state.get("p_l", 1.0)
        if p_l >= 0.85:
            continue  # 已掌握
        pre_list = prereqs.get(kp_id, [])
        all_ready = all(matrix.get(p, {}).get("p_l", 0.0) >= 0.6 for p in pre_list)
        if not all_ready:
            continue
        candidates.append((kp_id, p_l, deps.get(kp_id, 0)))
    # 排序: P(L) 低的优先, 被依赖度高的优先
    candidates.sort(key=lambda x: (x[1], -x[2]))
    return [
        {
            "kp_id": kp_id,
            "name": KP_NAMES.get(kp_id, kp_id),
            "p_l": round(p_l, 4),
            "is_ready": True,
            "next_to_learn": i == 0,
            "dependents": dep,
            "domain": KP_TO_DOMAIN.get(kp_id, ""),
        }
        for i, (kp_id, p_l, dep) in enumerate(candidates)
    ]


# ============================================================
# ECharts 配置构建器 (声明式)
# ============================================================


def _base_options(title: str, theme: str) -> dict[str, Any]:
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    return {
        "title": {"text": title, "textStyle": {"color": tc, "fontSize": 14}},
        "backgroundColor": "transparent",
        "textStyle": {"color": tc},
    }


def build_heatmap_option(
    matrix: dict[str, dict[str, float]],
    title: str = "42 KP 学情热力图",
    theme: str = "light",
    colorblind: bool = False,
) -> dict[str, Any]:
    """BKT 热力图 ECharts option (§5.1.1).

    rows=42 KP, cols=P(L)/P(K|L)/P(G)/P(S), 四域分组分隔线。
    色盲友好: 蓝-橙渐变代替红-绿渐变 (设计文档 §8.2)。
    """
    opt = _base_options(title, theme)
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    x_axis = ["P(L)", "P(K|L)", "P(G)", "P(S)"]
    y_axis: list[str] = []
    data: list[list] = []
    for domain in ("A", "B", "C", "D"):
        for kp_id in KP_DOMAIN_IDS[domain]:
            y_axis.append(kp_id)
            state = matrix.get(kp_id, {})
            for col, key in enumerate(BKT_PARAM_KEYS):
                # [x, y, value, param_label, kp_id] — tooltip 直接引用数据元素
                data.append([
                    col, len(y_axis) - 1, round(state.get(key, 0.0), 4),
                    {"p_l": "P(L)", "p_k_l": "P(K|L)", "p_g": "P(G)", "p_s": "P(S)"}[key],
                    kp_id,
                ])
    # tooltip 使用 ECharts 字符串模板引用数据元素 (无外部变量依赖, JSON 安全)
    opt["tooltip"] = {
        "position": "top",
        "formatter": "{@[4]} · {@[3]}: {@[2]}",
    }
    opt["grid"] = {"left": 72, "right": 24, "bottom": 50, "top": 50}
    opt["xAxis"] = {
        "type": "category", "data": x_axis, "splitArea": {"show": True},
        "axisLabel": {"color": tc},
    }
    opt["yAxis"] = {
        "type": "category", "data": y_axis, "splitArea": {"show": True},
        "axisLabel": {"color": tc, "fontSize": 10},
    }
    opt["visualMap"] = {
        "min": 0, "max": 1, "calculable": True, "orient": "horizontal",
        "left": "center", "bottom": 0,
        "inRange": {"color": _heatmap_gradient(colorblind)},
        "text": ["高", "低"], "textStyle": {"color": tc},
    }
    opt["series"] = [{"type": "heatmap", "data": data, "label": {"show": False}}]
    return opt


def _heatmap_gradient(colorblind: bool) -> list[str]:
    """热力图配色 — 默认红绿, 色盲友好蓝橙 (WCAG)."""
    if colorblind:
        return ["#f97316", "#fbbf24", "#7dd3fc", "#2563eb"]
    return ["#f87171", "#fbbf24", "#4ade80", "#16a34a"]


def build_kp_detail_option(
    kp_id: str,
    state: dict[str, float],
    history: list[dict[str, Any]] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """单 KP 详情 ECharts option — 四参数条形图 + 学习轨迹.

    Args:
        kp_id: 知识点 ID。
        state: 当前 BKT 四参数。
        history: [{step, p_l, p_k_l, p_g, p_s}] 历史快照列表。
        theme: 主题。

    Returns:
        子图表配置 (ECharts grid 布局)。
    """
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    param_names = ["掌握概率 P(L)", "学习转移 P(K|L)", "猜测 P(G)", "失误 P(S)"]
    colors = ["#4b3fe3", "#22a5f7", "#f59e0b", "#10b981"]
    values = [state.get(k, 0.0) for k in BKT_PARAM_KEYS]
    bar = {
        "title": {"text": f"{kp_id} {KP_NAMES.get(kp_id, '')} 四参数", "textStyle": {"color": tc, "fontSize": 13}},
        "xAxis": {"type": "value", "max": 1.0, "axisLabel": {"color": tc}},
        "yAxis": {"type": "category", "data": param_names, "axisLabel": {"color": tc}},
        "series": [{
            "type": "bar", "data": [
                {"value": v, "itemStyle": {"color": c}} for v, c in zip(values, colors)
            ],
            "label": {"show": True, "position": "right", "formatter": "{c:.2f}", "color": tc},
        }],
    }
    full = {"kp_bar": bar}
    if history:
        steps = [h.get("step", i) for i, h in enumerate(history)]
        full["trajectory"] = {
            "title": {"text": "学习轨迹", "textStyle": {"color": tc, "fontSize": 13}},
            "xAxis": {"type": "category", "data": steps, "axisLabel": {"color": tc}},
            "yAxis": {"type": "value", "min": 0, "max": 1, "name": "P(L)", "axisLabel": {"color": tc}},
            "series": [{
                "type": "line", "smooth": True, "symbolSize": 5,
                "data": [h.get("p_l", 0.0) for h in history],
                "markLine": {"silent": True, "data": [
                    {"yAxis": 0.6, "lineStyle": {"color": "#f59e0b", "type": "dashed"}, "label": {"formatter": "阈值 0.6"}},
                ]},
            }],
            "tooltip": {"trigger": "axis"},
        }
    return full


def build_progress_ring_option(
    avg_p_l: float,
    domain_data: dict[str, dict[str, Any]],
    theme: str = "light",
) -> dict[str, Any]:
    """总体掌握度环形进度条 + 域级卡片数据.

    Returns:
        {average, ring_chart: ECharts option, domain_cards: list}
    """
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    ring_color = "#4b3fe3"
    if avg_p_l > 0.8:
        ring_color = "#16a34a"
    elif avg_p_l > 0.5:
        ring_color = "#f59e0b"
    ring = {
        "title": {"text": f"{(avg_p_l * 100):.1f}%", "subtext": "总体掌握度", "left": "center", "top": "center",
                  "textStyle": {"fontSize": 28, "color": ring_color}, "subtextStyle": {"color": tc, "fontSize": 12}},
        "series": [{
            "type": "pie", "radius": ["62%", "82%"], "center": ["50%", "50%"],
            "avoidLabelOverlap": False,
            "label": {"show": False},
            "data": [
                {"value": round(avg_p_l * 100, 1), "name": "已掌握", "itemStyle": {"color": ring_color}},
                {"value": round((1 - avg_p_l) * 100, 1), "name": "待提升", "itemStyle": {"color": "#e2e8f0"}},
            ],
        }],
    }
    return {"average": round(avg_p_l, 4), "ring_chart": ring, "domain_cards": domain_data}


def build_convergence_option(
    rounds: list[int],
    consensus: list[float],
    theme: str = "light",
) -> dict[str, Any]:
    """争辩收敛过程折线图 ECharts option (§5.3.2)."""
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    return {
        "title": {"text": "共识度收敛过程", "textStyle": {"color": tc, "fontSize": 13}},
        "xAxis": {"type": "category", "data": [f"轮次 {r}" for r in rounds], "axisLabel": {"color": tc}},
        "yAxis": {"type": "value", "min": 0, "max": 1, "name": "共识度", "axisLabel": {"color": tc}},
        "series": [{
            "type": "line", "smooth": True, "symbolSize": 8,
            "data": consensus,
            "areaStyle": {"opacity": 0.08, "color": "#4b3fe3"},
            "lineStyle": {"color": "#4b3fe3", "width": 2.5},
        }],
        "tooltip": {"trigger": "axis", "formatter": "{b}: {c:.3f}"},
    }


def build_verdict_radar_option(
    dimensions: list[dict[str, Any]],
    theme: str = "light",
) -> dict[str, Any]:
    """裁决结果三维雷达图 ECharts option (§5.3.3)."""
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    return {
        "title": {"text": "裁决结果", "textStyle": {"color": tc, "fontSize": 13}},
        "tooltip": {"trigger": "item"},
        "radar": {
            "indicator": [{"name": d["name"], "max": d.get("max", 100)} for d in dimensions],
            "radius": "58%",
            "splitNumber": 4,
        },
        "series": [{
            "type": "radar",
            "data": [{"value": [d.get("value", 0) for d in dimensions], "name": "裁决"}],
            "areaStyle": {"opacity": 0.15, "color": "#4b3fe3"},
            "lineStyle": {"color": "#4b3fe3", "width": 2},
        }],
    }


def wrap_and_descriptor(
    artifact: Artifact | None,
    html_content: str,
    config: dict[str, Any],
    assets: list[str] | None,
    metadata: dict[str, Any] | None,
    renderer_name: str,
) -> RenderDescriptor:
    """统一构建渲染容器 + RenderDescriptor (面板组件复用)."""
    theme = "light"
    if artifact is None:
        artifact = Artifact(
            artifact_id="panel-" + str(int(time.time() * 1000))[-12:],
            payload={},
        )
    html = dashboard_wrap(html_content, "l7-dashboard", theme)
    return _build_desc(
        artifact,
        html=html,
        config=config,
        assets=assets or ["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        metadata=metadata or {},
        renderer_name=renderer_name,
    )


# ============================================================
# 面板统一样式与工具
# ============================================================

#: 面板专属 CSS (双主题变量), 由 dashboard_wrap 嵌入容器
DASHBOARD_CSS: str = """
.l7-dashboard{--dash-border:rgba(23,23,23,.1);--dash-muted:#52525b;--dash-card:#ffffff;--dash-soft:#f4f4f5}
.l7-dashboard.dark{--dash-border:rgba(255,255,255,.14);--dash-muted:#a1a1aa;--dash-card:#27272a;--dash-soft:#1f1f23}
.l7-dashboard h2,.l7-dashboard h3,.l7-dashboard h4{font-weight:600;margin:10px 0 6px}
.l7-dashboard .bkt-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.l7-dashboard .bkt-avg-mastery{font-size:13px;color:var(--dash-muted)}
.l7-dashboard .bkt-bottlenecks{border:1px solid var(--dash-border);border-radius:8px;padding:10px 14px;margin:10px 0;background:var(--dash-soft)}
.l7-dashboard .bottleneck-item{list-style:none;padding:3px 0;font-size:13px}
.l7-dashboard .kp-badge{border-radius:999px;padding:1px 8px;font-size:11.5px;color:#fff}
.l7-dashboard .kp-badge.bottleneck{background:#ef4444;animation:bkt-pulse 1.8s ease-in-out infinite}
@keyframes bkt-pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.45)}50%{box-shadow:0 0 0 6px rgba(239,68,68,0)}}
.l7-dashboard .domain-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:10px 0}
.l7-dashboard .domain-card{border:1px solid var(--dash-border);border-radius:10px;padding:12px 14px;background:var(--dash-card)}
.l7-dashboard .domain-card .domain-label{display:block;font-size:12px;color:var(--dash-muted)}
.l7-dashboard .domain-card .domain-avg{display:block;font-size:22px;font-weight:600;margin:2px 0}
.l7-dashboard .domain-card .domain-counts{font-size:11.5px;color:var(--dash-muted)}
.l7-dashboard .weak-points,.l7-dashboard .learning-path{border:1px solid var(--dash-border);border-radius:8px;padding:10px 14px;margin:10px 0;background:var(--dash-card)}
.l7-dashboard .weak-points ol,.l7-dashboard .learning-path ol{margin:6px 0 0;padding-left:20px}
.l7-dashboard .weak-item{font-size:13px;padding:3px 0}
.l7-dashboard .weak-item .weak-rank{display:inline-block;min-width:18px;color:var(--dash-muted)}
.l7-dashboard .small-badge{border-radius:999px;padding:0 6px;font-size:11px;color:#fff;background:#4b3fe3}
.l7-dashboard .path-item{padding:3px 0;font-size:13px}
.l7-dashboard .path-next{font-weight:600;color:#d97706}
.l7-dashboard .debate-timeline{border-left:2px solid var(--dash-border);padding-left:16px;margin:8px 0;max-height:320px;overflow-y:auto}
.l7-dashboard .debate-speech{margin:8px 0;padding:8px 12px;border-radius:8px;background:var(--dash-soft)}
.l7-dashboard .debate-agent{font-weight:600;margin-right:8px}
.l7-dashboard .debate-stance{font-size:12px}
.l7-dashboard .debate-summary{margin:4px 0 0;font-size:13px}
.l7-dashboard .verdict-summary{border:1px solid var(--dash-border);border-radius:8px;padding:10px 14px;margin:8px 0;background:var(--dash-card)}
.l7-dashboard .drilldown-breadcrumbs{margin:8px 0;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.l7-dashboard .crumb{color:#4b3fe3;font-size:13px;text-decoration:none}
.l7-dashboard .crumb.active{color:var(--dash-muted);pointer-events:none}
.l7-dashboard .crumb-sep{color:var(--dash-muted)}
.l7-dashboard .time-travel-timeline{display:flex;gap:6px;overflow-x:auto;margin:8px 0}
.l7-dashboard .time-tick{border:1px solid var(--dash-border);border-radius:999px;padding:3px 10px;font-size:11.5px;background:var(--dash-card);cursor:pointer}
.l7-dashboard .time-tick.active{background:#4b3fe3;color:#fff;border-color:#4b3fe3}
.l7-dashboard .compare-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:8px 0}
.l7-dashboard .compare-column{border:1px solid var(--dash-border);border-radius:10px;padding:12px;background:var(--dash-card)}
.l7-dashboard .compare-mastery{font-size:24px;font-weight:600;margin:4px 0}
.l7-dashboard .compare-domain{display:flex;justify-content:space-between;font-size:12px;color:var(--dash-muted);padding:1px 0}
.l7-dashboard .param-sliders{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin:8px 0}
.l7-dashboard .param-slider{border:1px solid var(--dash-border);border-radius:8px;padding:8px 12px;background:var(--dash-card)}
.l7-dashboard .param-slider label{display:block;font-size:12.5px;color:var(--dash-muted)}
.l7-dashboard .param-value{font-weight:600;font-size:13px}
.l7-dashboard .param-slider input[type=range]{width:100%;margin-top:4px}
.l7-dashboard .lab-config{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:8px 0}
.l7-dashboard .lab-param{border:1px solid var(--dash-border);border-radius:8px;padding:8px 10px;background:var(--dash-card);font-size:12.5px}
.l7-dashboard .lab-param label{display:block;color:var(--dash-muted);font-size:11.5px}
.l7-dashboard .lab-results ul{margin:6px 0;padding-left:18px}
.l7-dashboard .lab-results li{font-size:13px;padding:2px 0}
"""


def dashboard_css(theme: str) -> str:
    """生成面板统一样式 (含主题切换类)."""
    return (
        f"<style>.l7-dashboard{{--dash-border:rgba(23,23,23,.1);"
        f"--dash-muted:#52525b;--dash-card:#ffffff;--dash-soft:#f4f4f5}}"
        f".l7-dashboard.dark{{--dash-border:rgba(255,255,255,.14);"
        f"--dash-muted:#a1a1aa;--dash-card:#27272a;--dash-soft:#1f1f23}}"
        + DASHBOARD_CSS
        + "</style>"
    )


def dashboard_wrap(content: str, css_class: str = "l7-dashboard", theme: str = "light") -> str:
    """将面板内容包裹进统一容器 (基础主题 + 面板专属样式)."""
    classes = ("l7-render " + css_class).strip()
    dark_cls = " dark" if theme == "dark" else ""
    return f'<div class="{classes}{dark_cls}">{dashboard_css(theme)}{content}</div>'


def colorblind_from(artifact: Artifact | None, context: RenderContext | None) -> bool:
    """从 Artifact/RenderContext 提取色盲友好开关 (设计文档 §8.2)."""
    if context is not None and context.bkt_state:
        return bool(context.bkt_state.get("colorblind"))
    if artifact is not None and artifact.learner_context:
        return bool((artifact.learner_context or {}).get("colorblind"))
    return False
