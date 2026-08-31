"""L7 多模态输出 — 参数调节器 (param_controller.py).

任务拆分 T4 · 设计文档 §4.3.1。

5 参数实时联动滑块 (Dy3+ 掺杂浓度/焙烧温度/激发波长/环境温度/电荷补偿剂),
每次拖动更新关联计算模型并输出 ChartRenderer 可消费的 ECharts option。

参数联动模型:

| 参数 | 范围 | 关联计算 | 输出图表 |
|---|---|---|---|
| 掺杂浓度 | 0.1-30 mol% | 浓度猝灭 (PL强度曲线) | 折线+散点 |
| 焙烧温度 | 200-1400°C | 结晶度 (XRD峰锐化) | 折线 |
| 激发波长 | 200-500nm | 吸收截面 (PLE高亮) | 折线+标注 |
| 环境温度 | 77-500K | 热猝灭 (温度依赖) | 折线(含T50%) |
| 电荷补偿剂 | 0-2.0 | 缺陷化学 (QE变化) | 折线+散点 |
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import build_descriptor, esc
from ..dashboard._common import dashboard_wrap

#: 5 参数定义
_PARAMS = [
    {"key": "doping", "label": "Dy3+ 掺杂浓度", "unit": "mol%", "min": 0.1, "max": 30.0, "default": 2.0, "step": 0.1},
    {"key": "calc_temp", "label": "焙烧温度", "unit": "°C", "min": 200, "max": 1400, "default": 900, "step": 10},
    {"key": "exc_wavelength", "label": "激发波长", "unit": "nm", "min": 200, "max": 500, "default": 350, "step": 5},
    {"key": "env_temp", "label": "环境温度", "unit": "K", "min": 77, "max": 500, "default": 298, "step": 1},
    {"key": "charge_ratio", "label": "电荷补偿剂比例", "unit": "", "min": 0.0, "max": 2.0, "default": 1.0, "step": 0.05},
]


def _concentration_quenching(concentration: float) -> dict[str, Any]:
    """浓度猝灭模型: PL强度 ~ A * C * exp(-k*C)."""
    x = [round(i * 0.5, 1) for i in range(1, 61)]  # 0.5~30 mol%
    y = [round(c * 120 * (2.71828 ** (-c * 0.08)), 2) for c in x]
    opt_x = max(x, key=lambda c: c * (2.71828 ** (-c * 0.08)))
    return {
        "x_axis": x,
        "y_axis": y,
        "current_value": round(concentration * 120 * (2.71828 ** (-concentration * 0.08)), 2),
        "optimal": opt_x,
        "mark_line": opt_x,
    }


def _crystallinity(temp: float) -> dict[str, Any]:
    """结晶度 vs 焙烧温度 (S 曲线)."""
    x = list(range(200, 1420, 20))
    import math

    y = [round(1.0 / (1.0 + math.exp(-(t - 800) / 120)), 4) for t in x]
    return {"x_axis": x, "y_axis": y, "current_value": round(1.0 / (1.0 + math.exp(-(temp - 800) / 120)), 4)}


def _absorption(exc_nm: float) -> dict[str, Any]:
    """激发波长 → 吸收截面 (PLE)."""
    x = list(range(200, 510, 5))
    import math

    y = [round(math.exp(-((w - 350) ** 2) / 8000) * 0.9 + 0.1, 4) for w in x]
    return {"x_axis": x, "y_axis": y, "current_value": round(math.exp(-((exc_nm - 350) ** 2) / 8000) * 0.9 + 0.1, 4)}


def _thermal_quenching(env_k: float) -> dict[str, Any]:
    """热猝灭: PL强度 ~ 1/(1 + exp((T-T50)/sigma))."""
    x = list(range(77, 510, 5))
    import math

    t50 = 380
    y = [round(1.0 / (1.0 + math.exp((t - t50) / 40)), 4) for t in x]
    return {
        "x_axis": x,
        "y_axis": y,
        "current_value": round(1.0 / (1.0 + math.exp((env_k - t50) / 40)), 4),
        "t50": t50,
    }


def _defect_chemistry(ratio: float) -> dict[str, Any]:
    """电荷补偿剂 → 量子效率."""
    x = [round(i * 0.1, 2) for i in range(1, 21)]
    import math

    y = [round(0.3 + 0.6 * math.exp(-((r - 1.0) ** 2) / 0.5), 4) for r in x]
    return {"x_axis": x, "y_axis": y, "current_value": round(0.3 + 0.6 * math.exp(-((ratio - 1.0) ** 2) / 0.5), 4), "optimal": 1.0}


_MODELS = {
    "doping": _concentration_quenching,
    "calc_temp": _crystallinity,
    "exc_wavelength": _absorption,
    "env_temp": _thermal_quenching,
    "charge_ratio": _defect_chemistry,
}


def render_param_controller(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    params: dict[str, float] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染参数调节器 (全部 5 参数 + 联动图表).

    Args:
        params: 初始参数值 {doping, calc_temp, exc_wavelength, env_temp, charge_ratio}。

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    params = params or {p["key"]: p["default"] for p in _PARAMS}
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    charts: dict[str, Any] = {}
    for param_def in _PARAMS:
        key = param_def["key"]
        val = params.get(key, param_def["default"])
        result = _MODELS[key](val)
        charts[key] = {
            "label": param_def["label"],
            "unit": param_def["unit"],
            "current_value": result["current_value"],
            "optimal": result.get("optimal"),
            "t50": result.get("t50"),
            "chart": {
                "xAxis": {"type": "value", "name": param_def["unit"], "axisLabel": {"color": tc}},
                "yAxis": {"type": "value", "axisLabel": {"color": tc}},
                "series": [{
                    "type": "line", "smooth": True, "symbolSize": 0,
                    "data": [[x, y] for x, y in zip(result["x_axis"], result["y_axis"])],
                    "areaStyle": {"opacity": 0.06},
                }],
            },
        }

    # 滑块 HTML
    sliders_html = "".join(
        f'<div class="param-slider" data-key="{p["key"]}">'
        f'<label>{esc(p["label"])} ({esc(p["unit"])})</label>'
        f'<span class="param-value">{params[p["key"]]}</span>'
        f'<input type="range" min="{p["min"]}" max="{p["max"]}" step="{p["step"]}" value="{params[p["key"]]}">'
        f"</div>"
        for p in _PARAMS
    )
    html_content = (
        f'<div class="param-controller"><h3>交互式参数调节器</h3>'
        f'<div class="param-sliders">{sliders_html}</div>'
        + "".join(
            f'<div class="l7-chart" data-chart-id="param-{p["key"]}" style="width:100%;height:180px"></div>'
            for p in _PARAMS
        )
        + "</div>"
    )

    config = {
        "type": "param_controller",
        "params": [
            {
                "key": p["key"], "label": p["label"], "unit": p["unit"],
                "min": p["min"], "max": p["max"], "step": p["step"],
                "value": params[p["key"]],
            }
            for p in _PARAMS
        ],
        "charts": charts,
    }
    html = dashboard_wrap(html_content, "l7-dashboard l7-params", theme)
    descriptor = build_descriptor(
        artifact or Artifact(artifact_id="param-controller", payload={}),
        html=html,
        config=config,
        assets=["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        metadata={"renderer": "ParamController"},
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor
