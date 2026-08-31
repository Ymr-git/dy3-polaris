"""L7 多模态输出 — 虚拟实验台 (virtual_lab.py).

任务拆分 T4 · 设计文档 §4.3.2。

虚拟实验台工作流: 选择宿主材料 → 设定条件 → 查看模拟结果 → 输出 Artifact 组。

支持材料: NaGdF4 / YPO4 / BaMgAl10O17 (含晶格信息)。

输出: 一组 Artifact (chart + molecule + text)，全部可编辑可溯源。
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, ArtifactType, RenderContext
from ..renderers._common import build_descriptor, esc
from ..dashboard._common import dashboard_wrap

#: 宿主材料定义 (设计文档 §4.3.2)
_HOSTS = {
    "NaGdF4": {
        "formula": "NaGdF4",
        "system": "六方晶系",
        "space_group": "P-6 2m",
        "description": "上转换/下转换发光基质，低声子能 (~350 cm⁻¹)，适合 Dy3+ 4f-4f 跃迁",
        "default_doping": 2.0,
        "default_temp": 900,
    },
    "YPO4": {
        "formula": "YPO4",
        "system": "四方晶系",
        "space_group": "I41/amd",
        "description": "高化学稳定性，适合高温合成，Dy3+ 占据 Y3+ 格位 (D2d 对称性)",
        "default_doping": 5.0,
        "default_temp": 1100,
    },
    "BaMgAl10O17": {
        "formula": "BaMgAl10O17",
        "system": "六方晶系",
        "space_group": "P63/mmc",
        "description": "β-氧化铝结构，Dy3+ 进入尖晶石层间隙，宽带 4f-5d 跃迁优势",
        "default_doping": 3.0,
        "default_temp": 1300,
    },
}

#: 合成方法 (设计文档 C 域覆盖)
_METHODS = {
    "solid_state": {"label": "固相烧结法", "temp_range": "800-1400°C"},
    "co_precipitation": {"label": "共沉淀法", "temp_range": "600-1000°C"},
    "sol_gel": {"label": "溶胶-凝胶法", "temp_range": "500-900°C"},
    "hydrothermal": {"label": "水热/溶剂热法", "temp_range": "150-250°C"},
}


def render_virtual_lab(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    host: str = "NaGdF4",
    doping: float = 2.0,
    method: str = "solid_state",
    calc_temp: float = 900,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染虚拟实验台。

    Args:
        host: 宿主材料名 (NaGdF4/YPO4/BaMgAl10O17)。
        doping: 掺杂浓度 (mol%)。
        method: 合成方法。
        calc_temp: 焙烧温度 (°C)。

    Returns:
        RenderDescriptor (含实验参数 + 预测性能 + 输出 Artifact 组)。
    """
    started = time.monotonic()
    host_info = _HOSTS.get(host, _HOSTS["NaGdF4"])
    method_info = _METHODS.get(method, _METHODS["solid_state"])
    predictions = _predict_performance(host, doping, calc_temp)
    spectrum_chart = _spectrum_option(predictions, theme)

    # 实验参数卡片 HTML
    html_content = (
        f'<div class="virtual-lab">'
        f'<h3>🧪 虚拟实验台</h3>'
        f'<div class="lab-config">'
        f'<div class="lab-param"><label>宿主材料</label><span>{esc(host)}</span></div>'
        f'<div class="lab-param"><label>晶系/空间群</label><span>{esc(host_info["system"])} / {esc(host_info["space_group"])}</span></div>'
        f'<div class="lab-param"><label>掺杂浓度</label><span>{doping:.1f} mol% Dy3+</span></div>'
        f'<div class="lab-param"><label>合成方法</label><span>{esc(method_info["label"])} ({esc(method_info["temp_range"])})</span></div>'
        f'<div class="lab-param"><label>焙烧温度</label><span>{calc_temp:.0f}°C</span></div>'
        f"</div>"
        f'<div class="lab-results"><h4>预测性能</h4>'
        f'<ul>'
        f'<li>预估量子效率: <strong>{predictions["qe"]:.1f}%</strong></li>'
        f'<li>预估荧光寿命: <strong>{predictions["lifetime"]:.2f} ms</strong></li>'
        f'<li>主发射峰: <strong>{predictions["peak"]} nm</strong> (黄光)</li>'
        f'<li>CIE 色坐标: <strong>({predictions["cie_x"]:.3f}, {predictions["cie_y"]:.3f})</strong></li>'
        f'<li>热稳定性 T50%: <strong>{predictions["t50"]} K</strong></li>'
        f"</ul></div>"
        f'<div class="l7-chart" data-chart-id="lab-spectrum" style="width:100%;height:200px"></div>'
        f"</div>"
    )

    # 输出 Artifact 组 (chart + molecule + text)
    output_artifacts = [
        {
            "type": "chart",
            "mime": "application/vnd.dy3.chart+json",
            "title": f"{host}:Dy3+ PL 光谱预测",
            "payload": {
                "chart_type": "line",
                "title": f"{host}:{doping}%Dy3+ 发射光谱 (模拟)",
                "data": [
                    {"波长": w, "强度": round(predictions["spectrum"].get(str(w), 0.0), 4)}
                    for w in range(400, 700, 10)
                ],
            },
        },
        {
            "type": "molecule",
            "mime": "application/vnd.dy3.molecule+json",
            "title": f"{host} 晶格结构",
            "payload": {"host": host},
        },
        {
            "type": "text",
            "mime": "text/vnd.dy3+markdown",
            "title": f"{host}:Dy3+ 实验总结",
            "payload": {
                "content": (
                    f"## {host}:{doping}%Dy3+ 虚拟实验结果\n\n"
                    f"- **宿主**: {host_info['formula']} ({host_info['system']}, {host_info['space_group']})\n"
                    f"- **掺杂**: {doping} mol% Dy3+\n"
                    f"- **合成**: {method_info['label']} @ {calc_temp:.0f}°C\n"
                    f"- **量子效率**: {predictions['qe']:.1f}%\n"
                    f"- **荧光寿命**: {predictions['lifetime']:.2f} ms\n"
                    f"- **主发射峰**: {predictions['peak']} nm\n"
                    f"- **CIE**: ({predictions['cie_x']:.3f}, {predictions['cie_y']:.3f})\n"
                ),
            },
        },
    ]

    config = {
        "type": "virtual_lab",
        "host": host,
        "host_info": host_info,
        "doping": doping,
        "method": method,
        "calc_temp": calc_temp,
        "predictions": predictions,
        "spectrum_chart": spectrum_chart,
        "output_artifacts": output_artifacts,
    }
    html = dashboard_wrap(html_content, "l7-dashboard l7-lab", theme)
    descriptor = build_descriptor(
        artifact or Artifact(artifact_id="virtual-lab", payload={}),
        html=html,
        config=config,
        assets=["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        metadata={"renderer": "VirtualLab", "host": host, "output_count": len(output_artifacts)},
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor


def _predict_performance(host: str, doping: float, temp: float) -> dict[str, Any]:
    """基于 L3 经验模型预测性能 (简化)."""
    import math

    base = {"NaGdF4": 85, "YPO4": 72, "BaMgAl10O17": 68}.get(host, 75)
    quenching = math.exp(-doping * 0.08)
    qe = round(base * quenching * (0.5 + 0.5 * (1.0 / (1.0 + math.exp(-(temp - 800) / 120)))), 1)
    lifetime = round((0.5 + doping * 0.02) * 1.5, 2)
    t50 = 350 + int(host in ("YPO4", "BaMgAl10O17")) * 60
    cie = {"NaGdF4": (0.45, 0.48), "YPO4": (0.42, 0.44), "BaMgAl10O17": (0.35, 0.38)}.get(host, (0.44, 0.46))
    peak = 575
    spectrum = {}
    for w in range(400, 700, 10):
        dist = abs(w - peak)
        spectrum[str(w)] = round(1.0 / (1.0 + (dist / 15) ** 2) * qe / 100, 4)
    return {
        "qe": qe,
        "lifetime": lifetime,
        "peak": peak,
        "cie_x": cie[0],
        "cie_y": cie[1],
        "t50": t50,
        "spectrum": spectrum,
    }


def _spectrum_option(predictions: dict[str, Any], theme: str) -> dict[str, Any]:
    """构建模拟 PL 光谱 ECharts option (供 lab-spectrum 容器)."""
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    wl = sorted(int(w) for w in predictions["spectrum"])
    return {
        "title": {"text": "模拟发射光谱", "textStyle": {"color": tc, "fontSize": 13}},
        "xAxis": {"type": "value", "name": "波长 (nm)", "min": 400, "max": 700, "axisLabel": {"color": tc}},
        "yAxis": {"type": "value", "name": "相对强度", "axisLabel": {"color": tc}},
        "tooltip": {"trigger": "axis", "formatter": "波长 {c0}: 强度 {c1:.3f}"},
        "series": [{
            "type": "line", "smooth": True, "symbolSize": 0,
            "data": [[w, predictions["spectrum"].get(str(w), 0.0)] for w in wl],
            "lineStyle": {"color": "#d97706", "width": 2},
            "areaStyle": {"opacity": 0.12, "color": "#d97706"},
            "markLine": {
                "silent": True,
                "data": [{"xAxis": predictions["peak"], "lineStyle": {"color": "#4b3fe3", "type": "dashed"}}],
                "label": {"formatter": f"{predictions['peak']} nm"},
            },
        }],
    }
