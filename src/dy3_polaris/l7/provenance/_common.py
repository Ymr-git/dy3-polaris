"""L7 溯源可视化 — 公共数据层 (provenance/_common.py).

任务拆分 T5 · 设计文档 Ch.6。

提供溯源可视化共享能力:

1. LedgerEvent 归一化 — L0 LedgerEvent 字段 → 可视化事件
2. 哈希链验证状态 — prev_hash/event_hash 完整性校验展示
3. 事件类型映射 — 5 类事件 → 图标/颜色/标签
4. ECharts 构建器 — 条形图/雷达图/折线图
5. 统一样式与包装
"""

from __future__ import annotations

import time
from typing import Any

from ..renderers._common import build_descriptor as _build_desc
from ..renderers._common import esc as _esc
from ..models import Artifact, RenderDescriptor

#: 事件类型 → 图标/颜色/标签 (设计文档 Ch.6.1)
EVENT_TYPES: dict[str, dict[str, str]] = {
    "teaching": {"icon": "📖", "color": "#4b3fe3", "label": "教学"},
    "test": {"icon": "✏️", "color": "#22a5f7", "label": "测试"},
    "state_change": {"icon": "🔄", "color": "#f59e0b", "label": "状态变更"},
    "decision": {"icon": "🧭", "color": "#8b5cf6", "label": "决策"},
    "edit": {"icon": "🛠", "color": "#10b981", "label": "编辑"},
}

#: 分支原因 → 颜色 (设计文档 Ch.6.4)
BRANCH_REASONS: dict[str, str] = {
    "用户追问": "#22a5f7",
    "方案探索": "#8b5cf6",
    "Agent冲突": "#f59e0b",
}


def normalize_events(events: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """归一化 L0 LedgerEvent → 可视化事件.

    支持的输入字段 (兼容 L0 LedgerEvent 与自定义):
        event_id/event_type/timestamp/trace_id/agent_id/layer/
        prev_hash/event_hash/payload

    Returns:
        [{id, type, icon, color, label, timestamp, agent, summary,
          chain_id, trace_id, raw, masked}]
    """
    result: list[dict[str, Any]] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("event_type") or ev.get("type") or "interaction").lower()
        # L0 事件类型 → L7 可视化类型映射
        vis_type = _map_l0_type(etype)
        meta = EVENT_TYPES.get(vis_type, {"icon": "📄", "color": "#94a3b8", "label": vis_type})
        payload = ev.get("payload") or {}
        result.append({
            "id": ev.get("event_id") or ev.get("id") or f"evt-{int(time.time()*1000)%10**6}",
            "type": vis_type,
            "icon": meta["icon"],
            "color": meta["color"],
            "label": meta["label"],
            "timestamp": float(ev.get("timestamp") or ev.get("time") or 0.0),
            "agent": ev.get("agent_id") or ev.get("agent") or "",
            "layer": ev.get("layer", ""),
            "summary": str(ev.get("summary") or payload.get("summary") or payload.get("content") or ev.get("content") or ""),
            "chain_id": ev.get("provenance_chain_id") or payload.get("chain_id") or "",
            "trace_id": ev.get("trace_id", ""),
            "kp_id": ev.get("kp_id") or payload.get("kp_id") or "",
            "raw": ev.get("raw") or payload.get("raw", ""),
            "prev_hash": ev.get("prev_hash", ""),
            "event_hash": ev.get("event_hash", ""),
        })
    return result


def _map_l0_type(etype: str) -> str:
    """L0 五类事件 → L7 可视化五类."""
    mapping = {
        "learner_profile": "state_change",
        "knowledge": "teaching",
        "interaction": "test",
        "human_override": "edit",
    }
    if etype in EVENT_TYPES:
        return etype
    return mapping.get(etype, "test")


def verify_hash_chain(events: list[dict[str, Any]]) -> dict[str, Any]:
    """验证事件链哈希完整性 (防篡改, WORM 日志思想).

    校验规则: 每条事件的 prev_hash == 前一条的 event_hash。

    Args:
        events: 已归一化事件列表 (需含 prev_hash/event_hash)。

    Returns:
        {total, verified, tampered, valid, first_break_index}
    """
    total = len(events)
    verified = 0
    tampered: list[int] = []
    prev_hash = ""
    for i, ev in enumerate(events):
        if ev.get("prev_hash", "") == prev_hash:
            verified += 1
        else:
            tampered.append(i)
        prev_hash = ev.get("event_hash", "")
    return {
        "total": total,
        "verified": verified,
        "tampered": tampered,
        "valid": not tampered,
        "first_break_index": tampered[0] if tampered else None,
    }


def build_bar_option(
    items: list[dict[str, Any]],
    title: str,
    value_key: str,
    label_key: str = "name",
    theme: str = "light",
    horizontal: bool = True,
) -> dict[str, Any]:
    """水平/垂直条形图 ECharts option."""
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    labels = [str(i.get(label_key, "")) for i in items]
    values = [float(i.get(value_key, 0.0)) for i in items]
    opt: dict[str, Any] = {
        "title": {"text": title, "textStyle": {"color": tc, "fontSize": 13}},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 90, "right": 30, "top": 40, "bottom": 30},
    }
    if horizontal:
        opt["xAxis"] = {"type": "value", "axisLabel": {"color": tc}}
        opt["yAxis"] = {"type": "category", "data": labels, "axisLabel": {"color": tc}}
    else:
        opt["xAxis"] = {"type": "category", "data": labels, "axisLabel": {"color": tc}}
        opt["yAxis"] = {"type": "value", "axisLabel": {"color": tc}}
    opt["series"] = [{
        "type": "bar",
        "data": [
            {"value": v, "itemStyle": {"color": i.get("color", "#4b3fe3")}}
            for v, i in zip(values, items)
        ],
        "label": {"show": True, "position": "right", "color": tc},
    }]
    return opt


def build_radar_option(
    dimensions: list[dict[str, Any]],
    title: str,
    theme: str = "light",
) -> dict[str, Any]:
    """雷达图 ECharts option."""
    tc = "#e5e5e5" if theme == "dark" else "#171717"
    return {
        "title": {"text": title, "textStyle": {"color": tc, "fontSize": 13}},
        "tooltip": {"trigger": "item"},
        "radar": {
            "indicator": [{"name": d["name"], "max": d.get("max", 100)} for d in dimensions],
            "radius": "58%",
        },
        "series": [{
            "type": "radar",
            "data": [{
                "value": [d.get("value", 0) for d in dimensions],
                "name": title,
                "areaStyle": {"opacity": 0.15},
            }],
        }],
    }


def provenance_wrap(content: str, css_class: str = "l7-provenance-panel", theme: str = "light") -> str:
    """溯源面板统一包装 (复用 dashboard 样式 + 溯源专属)."""
    from ..dashboard._common import dashboard_wrap

    return dashboard_wrap(content, css_class, theme)


def build_panel_descriptor(
    artifact: Artifact | None,
    html_content: str,
    config: dict[str, Any],
    assets: list[str] | None,
    metadata: dict[str, Any] | None,
    renderer_name: str,
) -> RenderDescriptor:
    """构建溯源面板 RenderDescriptor (统一)."""
    theme = "light"
    if artifact is None:
        artifact = Artifact(artifact_id=f"prov-{int(time.time()*1000)%10**12:012d}", payload={})
    html = provenance_wrap(html_content, "l7-provenance-panel", theme)
    return _build_desc(
        artifact,
        html=html,
        config=config,
        assets=assets or ["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        metadata=metadata or {},
        renderer_name=renderer_name,
    )
