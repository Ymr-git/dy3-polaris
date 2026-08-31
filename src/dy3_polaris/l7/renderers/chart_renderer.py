"""L7 渲染器 — ChartRenderer (application/vnd.dy3.chart+json).

将结构化图表数据 Artifact 渲染为 ECharts 可消费的 option 配置。
服务端完成数据类型推断与 ECharts option 构建，前端直接注入 ECharts 实例。

实现能力 (对应 L7 设计文档 §2.3 + §4.2):

1. **7 条自动图表类型推断规则** (§2.3.1):
   - 1 离散维度 + 1 连续度量 → 柱状图 bar
   - 1 连续/时间维度 + 1 度量 → 折线图 line
   - 1 分类维度 + 比例度量 → 饼图 pie
   - 2 连续维度 + 1 度量 → 散点图 scatter
   - 2 分类维度 + 1 度量 → 热力图 heatmap
   - 多维度综合对比 → 雷达图 radar
   - 显式声明 chart_type → 按声明渲染 (最高优先)
2. **领域自定义图** (§4.2): Jablonski 能级图 / 材料性能雷达图 /
   合成工艺流程图 / 光谱叠加对比图 (graph_kind 路由)。
3. **交互配置** (§2.3.2): Tooltip / DataZoom / Legend Toggle / 导出 PNG/SVG。
4. **BKT 学情数据注入**: payload 携带 bkt 矩阵时自动生成热力图数据。

融合世界先进方案:
    - Vega-Lite 编码通道语义: 以 dimensions/measures 显式描述字段角色
    - ECharts 声明式 option: 前端零逻辑直接渲染
    - WCAG: 色盲友好 palette 切换 (blue-orange)

输出契约:
    RenderDescriptor.html   — 轻量挂载壳 (前端按 config 注入 ECharts)
    RenderDescriptor.config — {chart_type, option, interactions, colorblind}
    RenderDescriptor.assets — [echarts.min.js]
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..models import Artifact, RenderContext
from ._common import build_descriptor, wrap

#: 支持的 MIME 类型 (设计文档 §2.9)
_MIME_TYPES: list[str] = [
    "application/vnd.dy3.chart+json",
    "application/vnd.dy3.interactive+json",
]

#: 显式声明的图表类型 → ECharts series type
_EXPLICIT_TYPES: set[str] = {
    "bar", "line", "pie", "scatter", "heatmap", "radar", "boxplot",
    "sankey", "gauge", "funnel", "effectScatter", "graph",
}

#: 领域自定义图类型
_DOMAIN_KINDS: set[str] = {"jablonski", "radar", "process", "spectrum"}

#: 色盲友好 palette (蓝-橙方案, WCAG 对比度友好)
_PALETTE_NORMAL = ["#4b3fe3", "#22a5f7", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]
_PALETTE_CB = ["#2563eb", "#0ea5e9", "#f97316", "#16a34a", "#dc2626", "#7c3aed"]


def _palette(colorblind: bool) -> list[str]:
    return list(_PALETTE_CB if colorblind else _PALETTE_NORMAL)


def _is_numeric(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v)
            return True
        except ValueError:
            return False
    return False


def _classify(values: list[Any]) -> str:
    """推断字段语义类型 (离散/连续/时间/分类)."""
    if not values:
        return "nominal"
    if any(_is_numeric(v) for v in values):
        # 数值 → 连续 (若全为可排序数值)
        if all(_is_numeric(v) for v in values):
            return "quantitative"
    # 时间样式的字符串 (YYYY-MM-DD 等)
    if all(isinstance(v, str) and _looks_temporal(v) for v in values):
        return "temporal"
    return "nominal"


def _looks_temporal(v: str) -> bool:
    v = v.strip()
    # 简单时间样式识别: 含 - / : 分隔符 (日期/时间戳)
    if any(c in v for c in ("-", ":", "/")):
        return any(ch.isdigit() for ch in v)
    # 数字+时间单位后缀模式 (0ms / 10s / 5min / 2h)
    m = re.match(
        r"^[-+]?\d+(?:\.\d+)?\s*(ms|s|ns|us|min|h|day|week|month|year)$",
        v,
    )
    return bool(m) and any(ch.isdigit() for ch in v)


# ============================================================
# option 构建
# ============================================================

def _base_option(title: str, theme: str, colorblind: bool) -> dict[str, Any]:
    """ECharts 基础 option (标题/主题/调色板)."""
    text_color = "#e5e5e5" if theme == "dark" else "#171717"
    return {
        "color": _palette(colorblind),
        "title": {"text": title, "textStyle": {"color": text_color, "fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "legend": {"textStyle": {"color": text_color}, "type": "scroll"},
        "backgroundColor": "transparent",
        "textStyle": {"color": text_color},
    }


def _build_bar(dim: str, measures: list[dict[str, Any]], data: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    """柱状图: 1 离散维度 + 1 度量 (设计文档 §2.3.1 规则1)."""
    option = dict(base)
    option["xAxis"] = {
        "type": "category",
        "data": [r[dim] for r in data],
        "axisLabel": {"rotate": 0 if len(data) <= 8 else 30},
    }
    option["yAxis"] = {"type": "value", "name": measures[0].get("name", measures[0]["field"])}
    option["series"] = [
        {
            "name": m.get("name", m["field"]),
            "type": "bar",
            "data": [r.get(m["field"]) for r in data],
            "itemStyle": {"borderRadius": [4, 4, 0, 0]},
        }
        for m in measures
    ]
    return option


def _build_line(dim: str, measures: list[dict[str, Any]], data: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    """折线图: 1 连续/时间维度 + 1 度量 (规则2)."""
    option = dict(base)
    option["xAxis"] = {
        "type": "category",
        "data": [r[dim] for r in data],
        "boundaryGap": False,
    }
    option["yAxis"] = {"type": "value", "name": measures[0].get("name", measures[0]["field"])}
    option["dataZoom"] = [
        {"type": "inside", "start": 0, "end": 100},
        {"type": "slider", "start": 0, "end": 100, "height": 18},
    ]
    option["series"] = [
        {
            "name": m.get("name", m["field"]),
            "type": "line",
            "smooth": True,
            "symbolSize": 5,
            "data": [r.get(m["field"]) for r in data],
        }
        for m in measures
    ]
    return option


def _build_pie(dim: str, measure: dict[str, Any], data: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    """饼图: 1 分类维度 + 比例度量 (规则3)."""
    option = dict(base)
    total = sum(float(r.get(measure["field"], 0) or 0) for r in data) or 1.0
    option["tooltip"] = {
        "trigger": "item",
        "formatter": "{b}: {c} ({d}%)",
    }
    option["series"] = [
        {
            "name": measure.get("name", measure["field"]),
            "type": "pie",
            "radius": ["38%", "68%"],
            "center": ["50%", "55%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderRadius": 6, "borderColor": "transparent"},
            "label": {"formatter": "{b}\n{d}%"},
            "data": [
                {
                    "name": str(r[dim]),
                    "value": round(float(r.get(measure["field"], 0) or 0), 4),
                }
                for r in data
            ],
        }
    ]
    return option


def _build_scatter(dim_x: str, dim_y: str, measure: dict[str, Any], data: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    """散点图: 2 连续维度 + 1 度量 (规则4, 如 CIE 色坐标分布)."""
    option = dict(base)
    option["tooltip"] = {"trigger": "item"}
    option["xAxis"] = {"type": "value", "name": dim_x, "scale": True}
    option["yAxis"] = {"type": "value", "name": dim_y, "scale": True}
    option["series"] = [
        {
            "name": measure.get("name", measure["field"]),
            "type": "scatter",
            "symbolSize": lambda val: max(6, min(24, float(val[2] or 0) * 18)),
            "data": [
                [float(r[dim_x]), float(r[dim_y]), r.get(measure["field"], 0)] for r in data
            ],
            "itemStyle": {"opacity": 0.75},
        }
    ]
    return option


def _build_heatmap(
    dim_x: str, dim_y: str, measure: dict[str, Any], data: list[dict[str, Any]], base: dict[str, Any]
) -> dict[str, Any]:
    """热力图: 2 分类维度 + 1 度量 (规则5, 如 BKT 42 KP 掌握概率矩阵)."""
    option = dict(base)
    x_vals: list[str] = []
    y_vals: list[str] = []
    for r in data:
        if r.get(dim_x) not in x_vals:
            x_vals.append(str(r[dim_x]))
        if r.get(dim_y) not in y_vals:
            y_vals.append(str(r[dim_y]))
    value = [
        [x_vals.index(str(r[dim_x])), y_vals.index(str(r[dim_y])), float(r.get(measure["field"], 0) or 0)]
        for r in data
    ]
    option["tooltip"] = {"position": "top"}
    option["grid"] = {"left": 90, "right": 24, "bottom": 60, "top": 48}
    option["xAxis"] = {"type": "category", "data": x_vals, "splitArea": {"show": True}}
    option["yAxis"] = {"type": "category", "data": y_vals, "splitArea": {"show": True}}
    option["visualMap"] = {
        "min": 0, "max": 1, "calculable": True, "orient": "horizontal",
        "left": "center", "bottom": 0,
        "inRange": {"color": ["#f87171", "#fbbf24", "#4ade80", "#16a34a"]},
    }
    option["series"] = [
        {
            "name": measure.get("name", measure["field"]),
            "type": "heatmap",
            "data": value,
            "label": {"show": False},
            "emphasis": {"itemStyle": {"shadowBlur": 8, "shadowColor": "rgba(0,0,0,0.4)"}},
        }
    ]
    return option


def _build_radar(dim: str, measures: list[dict[str, Any]], data: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    """雷达图: 多维度综合对比 (规则6, 如材料性能多维对比)."""
    option = dict(base)
    indicators = [
        {"name": str(m.get("name", m["field"])), "max": m.get("max", 100)}
        for m in measures
    ]
    option["tooltip"] = {"trigger": "item"}
    option["radar"] = {
        "indicator": indicators,
        "radius": "62%",
        "splitNumber": 4,
    }
    option["series"] = [
        {
            "name": "性能对比",
            "type": "radar",
            "data": [
                {
                    "value": [float(r.get(f["field"], 0) or 0) for f in measures],
                    "name": str(r[dim]),
                }
                for r in data
            ],
        }
    ]
    return option


# ============================================================
# 领域自定义图
# ============================================================

def _build_jablonski(payload: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Jablonski 能级图 (§4.2.1) — Dy3+ 激发-弛豫-发光过程.

    payload["levels"]: [{name, energy, type: ground|excited4f|excited5d}]
    payload["transitions"]: [{from, to, kind: absorption|relaxation|emission, wavelength?}]
    """
    option = dict(base)
    option["title"] = dict(base.get("title") or {})
    option["title"]["text"] = option["title"].get("text") or "Dy3+ Jablonski 能级图"
    levels = payload.get("levels") or []
    transitions = payload.get("transitions") or []

    max_e = max((float(l.get("energy", 0)) for l in levels), default=1.0) or 1.0
    colors = {"ground": "#10b981", "excited4f": "#22a5f7", "excited5d": "#8b5cf6"}
    series: list[dict[str, Any]] = []

    # 能级横线 (effectScatter + 文本)
    level_data = [
        {
            "name": str(l.get("name", "")),
            "value": [float(l.get("energy", 0)), 0],
            "symbol": "roundRect",
            "symbolSize": [150, 6],
            "itemStyle": {"color": colors.get(str(l.get("type", "excited4f")), "#94a3b8")},
            "label": {"show": True, "position": "right", "formatter": str(l.get("name", "")), "color": base["textStyle"]["color"]},
        }
        for l in levels
    ]
    series.append({"type": "scatter", "data": level_data, "silent": True})

    # 跃迁箭头 (markLine)
    mark_lines: list[dict[str, Any]] = []
    for t in transitions:
        e_from = next((float(l.get("energy", 0)) for l in levels if str(l.get("name")) == str(t.get("from"))), 0)
        e_to = next((float(l.get("energy", 0)) for l in levels if str(l.get("name")) == str(t.get("to"))), 0)
        kind = str(t.get("kind", "absorption"))
        line_style: dict[str, Any] = {"color": "#ef4444", "width": 2}
        symbol = ["none", "arrow"]
        if kind == "relaxation":
            line_style = {"color": "#f59e0b", "width": 1.5, "type": "dashed", "curveness": 0.25}
            symbol = ["none", "none"]
        elif kind == "emission":
            line_style = {"color": "#facc15", "width": 3}
        mark_lines.append(
            {
                "name": f"{t.get('from')}→{t.get('to')}",
                "xAxis": e_to if e_to >= e_from else e_from,
                "yAxis": max(e_from, e_to) * 0.55,
                "symbol": symbol,
                "lineStyle": line_style,
                "label": {
                    "show": bool(t.get("wavelength")),
                    "formatter": f"{t.get('wavelength')} nm",
                    "color": "#facc15" if kind == "emission" else base["textStyle"]["color"],
                },
            }
        )
    # 用 markLine 挂在最上层散点上
    series.append(
        {
            "type": "scatter",
            "data": [{"value": [max_e * 0.5, max_e * 0.9], "symbol": "none"}],
            "markLine": {"silent": True, "data": mark_lines},
        }
    )
    option["xAxis"] = {"type": "value", "min": -max_e * 0.25, "max": max_e * 1.25, "show": False}
    option["yAxis"] = {"type": "value", "min": 0, "max": max_e * 1.15, "name": "能量 (eV)", "show": False}
    option["series"] = series
    option["animationDuration"] = 600
    return option


def _build_process_flow(payload: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """合成工艺流程图 (§4.2.5) — 步骤 + 参数 + 注意事项 (graph 类型)."""
    option = dict(base)
    steps = payload.get("steps") or []
    nodes = []
    edges = []
    for i, step in enumerate(steps):
        nodes.append(
            {
                "id": f"step-{i}",
                "name": str(step.get("name", f"步骤{i+1}")),
                "symbolSize": 64,
                "category": i,
                "x": (i % 3) * 180 - 180,
                "y": (i // 3) * 130,
                "itemStyle": {"color": _palette(False)[i % 6]},
                "label": {"show": True, "formatter": str(step.get("name", "")), "position": "bottom"},
                "tooltip": {"formatter": str(step.get("params", {}))},
            }
        )
        if i > 0:
            edges.append({"source": f"step-{i-1}", "target": f"step-{i}", "value": 1})
    option["series"] = [
        {
            "type": "graph",
            "layout": "none",
            "data": nodes,
            "links": edges,
            "roam": True,
            "lineStyle": {"color": "#94a3b8", "width": 2, "curveness": 0.15},
            "label": {"show": True, "color": base["textStyle"]["color"]},
            "emphasis": {"focus": "adjacency"},
        }
    ]
    return option


def _build_spectrum(payload: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """光谱叠加对比图 (§4.2.6) — 多系列折线 + 峰位标注."""
    option = dict(base)
    series_cfg = payload.get("series") or []
    x_label = str(payload.get("x_label", "波长 (nm)"))
    y_label = str(payload.get("y_label", "归一化强度"))
    x_axis: list[Any] = []
    series: list[dict[str, Any]] = []
    peaks = payload.get("peaks") or []

    for idx, sc in enumerate(series_cfg):
        data = sc.get("data") or []
        if not x_axis:
            x_axis = [d.get("x") for d in data]
        series.append(
            {
                "name": str(sc.get("name", f"系列{idx+1}")),
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "lineStyle": {"width": 2},
                "areaStyle": {"opacity": 0.06} if sc.get("fill") else None,
                "data": [d.get("y") for d in data],
            }
        )

    if peaks:
        mark_line_data = [
            {
                "name": str(p.get("label", "")),
                "xAxis": p.get("x"),
                "label": {"formatter": str(p.get("label", "")), "color": "#f59e0b", "position": "top"},
                "lineStyle": {"color": "#f59e0b", "type": "dashed", "width": 1},
            }
            for p in peaks
        ]
        if series:
            series[-1]["markLine"] = {"silent": True, "data": mark_line_data}

    option["xAxis"] = {"type": "category", "data": x_axis, "name": x_label, "nameLocation": "middle", "nameGap": 30}
    option["yAxis"] = {"type": "value", "name": y_label}
    option["dataZoom"] = [{"type": "inside", "start": 0, "end": 100}, {"type": "slider", "height": 16}]
    option["tooltip"] = {"trigger": "axis"}
    option["legend"] = {"type": "scroll", "top": 28}
    option["series"] = series
    return option


# ============================================================
# 推断与分发
# ============================================================

#: 已知分类/维度语义字段名 (Vega-Lite 语义类型推断辅助)
_KNOWN_DIM_FIELDS: tuple[str, ...] = (
    "name", "label", "kp_id", "材料", "工艺", "域", "domain",
    "浓度", "温度", "波长", "时间", "途径", "维度",
)


def _field_kind(values: list[Any]) -> str:
    """返回字段语义类型: temporal / quantitative / nominal."""
    if not values:
        return "nominal"
    if all(isinstance(v, str) and _looks_temporal(v) for v in values):
        return "temporal"
    if all(_is_numeric(v) for v in values):
        return "quantitative"
    return "nominal"


def _looks_like_ratio(vals: list[Any]) -> bool:
    """判断度量是否为比例 (饼图条件: 全在 [0,1] 且总和≈1)."""
    numeric = [float(v) for v in vals if v is not None and _is_numeric(v)]
    if not numeric or len(numeric) != len([v for v in vals if v is not None]):
        return False
    return all(0.0 <= v <= 1.0 for v in numeric) and abs(sum(numeric) - 1.0) < 0.15


def _infer_chart_type(payload: dict[str, Any], data: list[dict[str, Any]]) -> tuple[str, list[str], list[dict[str, Any]]]:
    """按 7 条规则推断图表类型 (Vega-Lite 语义推断).

    Args:
        payload: Artifact payload。
        data: 数据行列表。

    Returns:
        (chart_type, dims, measures) — dims 为维度字段列表 (scatter/heatmap 用 2 个)。
    """
    # 规则7: 显式声明优先 (measures 缺失时兜底推断)
    declared = str(payload.get("chart_type", "")).lower()
    if declared in _EXPLICIT_TYPES:
        dims, measures = _resolve_fields(payload, data)
        if not dims:
            dims = [next(iter(data[0]), "name")] if data else ["name"]
        if not measures:
            measures = [{"field": next(iter(data[0]), "value"), "name": "value"}]
        return declared, dims, measures

    dims, measures = _resolve_fields(payload, data)
    if not dims:
        dims = ["name"]
    if not measures:
        measures = [{"field": "value", "name": "value"}]

    # 多度量 → 多系列柱状图
    if len(measures) >= 2:
        return "bar", dims, measures

    # 双维度
    if len(dims) >= 2:
        kinds = [_field_kind([r.get(d) for r in data]) for d in dims[:2]]
        if all(k == "quantitative" for k in kinds):
            # 规则4: 2 连续维度 + 1 度量 → 散点 (如 CIE 色坐标)
            return "scatter", dims, measures
        if all(k == "nominal" for k in kinds):
            # 规则5: 2 分类维度 + 1 度量 → 热力图 (如 BKT 矩阵)
            return "heatmap", dims, measures
        return "bar", dims, measures

    dim = dims[0]
    dim_type = _field_kind([r.get(dim) for r in data] if data else [])
    measure = measures[0]
    vals = [r.get(measure["field"]) for r in data] if data else []
    m_type = _field_kind(vals)

    if dim_type == "temporal":
        # 规则2: 1 时间维度 + 1 度量 → 折线 (荧光衰减/热猝灭)
        return "line", dims, measures
    if dim_type == "quantitative" and m_type == "quantitative":
        # 规则2 变体: 1 连续维度 + 1 连续度量 → 折线
        return "line", dims, measures
    if dim_type == "nominal" and m_type == "quantitative":
        if _looks_like_ratio(vals):
            # 规则3: 1 分类维度 + 比例度量 → 饼图 (量子效率分配)
            return "pie", dims, measures
        # 规则1: 1 离散维度 + 1 连续度量 → 柱状 (掺杂浓度-强度)
        return "bar", dims, measures
    return "bar", dims, measures


def _resolve_fields(
    payload: dict[str, Any], data: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    """解析维度与度量字段 (显式优先, 否则从数据推断).

    Args:
        payload: Artifact payload (可含 dimensions/measures)。
        data: 数据行列表。

    Returns:
        (dims, measures)。
    """
    dims: list[str] = [str(d) for d in (payload.get("dimensions") or [])]
    measures: list[dict[str, Any]] = list(payload.get("measures") or [])
    if dims and measures:
        return dims, measures

    fields = list(data[0].keys()) if data else []
    if not fields:
        return dims, measures

    # 未显式提供时, 按语义类型推断 (Vega-Lite 式)
    known_dims: list[str] = []
    numeric_fields: list[str] = []
    for f in fields:
        vals = [r.get(f) for r in data]
        if f in _KNOWN_DIM_FIELDS or _field_kind(vals) == "nominal":
            known_dims.append(f)
        elif _field_kind(vals) in ("temporal", "quantitative"):
            numeric_fields.append(f)

    if not dims:
        if known_dims:
            dims = known_dims
        elif len(numeric_fields) >= 3:
            # 全数值: 前两个作维度 (散点), 其余作度量
            dims = numeric_fields[:2]
        elif numeric_fields:
            dims = [numeric_fields[0]]

    if not measures:
        measure_fields = [f for f in numeric_fields if f not in dims]
        measures = [{"field": f, "name": f} for f in measure_fields]
        if not measures:
            # 兜底: 第一个非维度字段作为度量
            for f in fields:
                if f not in dims:
                    measures = [{"field": f, "name": f}]
                    break
    return dims, measures


# ============================================================
# 渲染器
# ============================================================

class ChartRenderer:
    """图表渲染器 — 结构化数据 → ECharts option (服务端构建).

    使用示例::

        renderer = ChartRenderer()
        descriptor = renderer.render(artifact, context)
        # descriptor.config["option"] 直接作为 ECharts setOption 入参
    """

    _MIME_TYPES: list[str] = list(_MIME_TYPES)

    def render(self, artifact: Artifact, context: RenderContext):
        started = time.monotonic()
        if artifact is None or not artifact.payload:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload", detail="Chart artifact requires non-empty payload"
            )
        payload = artifact.payload
        graph_kind = str(payload.get("graph_kind", "")).lower()
        if (
            "chart_type" not in payload
            and "data" not in payload
            and graph_kind not in _DOMAIN_KINDS
        ):
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload",
                missing_fields=["chart_type", "data"],
                detail="Chart artifact requires 'chart_type', 'data' or 'graph_kind' in payload",
            )

        theme = (context.theme if context else "light") or "light"
        colorblind = bool(payload.get("colorblind")) or bool(
            (context.bkt_state or {}).get("colorblind")
        )
        title = str(payload.get("title", ""))
        data: list[dict[str, Any]] = payload.get("data") or []
        base = _base_option(title, theme, colorblind)

        if graph_kind in _DOMAIN_KINDS:
            if graph_kind == "jablonski":
                option = _build_jablonski(payload, base)
            elif graph_kind == "process":
                option = _build_process_flow(payload, base)
            elif graph_kind == "spectrum":
                option = _build_spectrum(payload, base)
            else:
                dim = (payload.get("dimensions") or ["name"])[0]
                option = _build_radar(dim, payload.get("measures") or [], data, base)
            chart_type = f"domain-{graph_kind}"
        else:
            chart_type, dims, measures = _infer_chart_type(payload, data)
            dim = dims[0] if dims else "name"
            if chart_type == "bar":
                option = _build_bar(dim, measures, data, base)
            elif chart_type == "line":
                option = _build_line(dim, measures, data, base)
            elif chart_type == "pie":
                option = _build_pie(dim, measures[0], data, base)
            elif chart_type == "scatter":
                dim_x = dim
                dim_y = dims[1] if (len(dims) > 1 and dims[1]) else self._find_second_numeric(data, dim)
                option = _build_scatter(dim_x, dim_y, measures[0], data, base)
            elif chart_type == "heatmap":
                dim_x = dim
                dim_y = dims[1] if (len(dims) > 1 and dims[1]) else self._find_second_numeric(data, dim)
                option = _build_heatmap(dim_x, dim_y, measures[0], data, base)
            elif chart_type == "radar":
                option = _build_radar(dim, measures, data, base)
            else:
                option = _build_bar(dim, measures, data, base)

        interactions = {
            "tooltip": True,
            "dataZoom": chart_type in ("line", "spectrum"),
            "legend_toggle": True,
            "export": {"png": True, "svg": True},
        }
        html = wrap(
            f'<div class="l7-chart" data-chart-id="{artifact.artifact_id}" '
            f'style="width:100%;height:{payload.get("height", 360)}px"></div>',
            "l7-chart-wrap",
            theme,
        )
        config = {
            "chart_type": chart_type,
            "option": option,
            "interactions": interactions,
            "colorblind": colorblind,
            "theme": theme,
            "renderer": "echarts",
        }
        descriptor = build_descriptor(
            artifact,
            html=html,
            config=config,
            assets=["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
            metadata={
                "renderer": "ChartRenderer",
                "inferred": chart_type if not graph_kind else f"domain-{graph_kind}",
                "row_count": len(data),
            },
        )
        descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
        return descriptor

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)

    @staticmethod
    def _find_second_numeric(data: list[dict[str, Any]], exclude: str) -> str:
        """在数据行中找到第二个数值字段 (用于散点/热力图的 Y 维度).

        Args:
            data: 数据行列表。
            exclude: 需要排除的字段名。

        Returns:
            第二个数值字段名；找不到时回退为排除字段本身。
        """
        if not data:
            return exclude
        numeric_fields = [
            k for k, v in data[0].items()
            if k != exclude and _is_numeric(v)
        ]
        return numeric_fields[0] if numeric_fields else exclude
