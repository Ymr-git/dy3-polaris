"""L7 溯源可视化 — 溯源时间线 (timeline.py).

任务拆分 T5 · 设计文档 Ch.6.1。

纵向时间线展示 KP 交互历史:
- 5 种事件类型 (教学/测试/状态变更/决策/编辑), 图标+颜色区分
- 按类型筛选、按时间范围缩放
- 点击节点查看详情 (涉及 Agent/知识源/Provenance Chain ID)
- 哈希链验证状态展示 (prev_hash/event_hash 完整性)
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import build_descriptor, esc, wrap
from ._common import (
    EVENT_TYPES,
    build_panel_descriptor,
    normalize_events,
    verify_hash_chain,
)


def render_timeline(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    events: list[dict[str, Any]] | None = None,
    theme: str = "light",
    full_access: bool = False,
) -> dict[str, Any]:
    """渲染溯源时间线 (Ch.6.1).

    Args:
        events: L0 LedgerEvent 列表或自定义事件列表。
        full_access: 是否授权展示原始输入 (默认脱敏)。

    Returns:
        RenderDescriptor (html + config)。
    """
    started = time.monotonic()
    normalized = normalize_events(events)
    chain_status = verify_hash_chain(normalized)
    total = len(normalized)

    # 类型筛选 chips
    chips = "".join(
        f'<button class="prov-filter" data-type="{t}" style="--pc:{meta["color"]}">{meta["icon"]} {meta["label"]}</button>'
        for t, meta in EVENT_TYPES.items()
    )

    # 哈希链验证状态
    chain_badge = (
        f'<span class="prov-chain-ok">🔒 哈希链完整 ({chain_status["verified"]}/{total})</span>'
        if chain_status["valid"]
        else f'<span class="prov-chain-broken">⚠️ 哈希链中断 ({chain_status["first_break_index"] + 1} 处)</span>'
    )

    # 事件节点
    nodes = []
    for i, ev in enumerate(normalized):
        ts = ev["timestamp"]
        time_str = time.strftime("%m-%d %H:%M", time.localtime(ts)) if ts else "--"
        raw_html = ""
        if ev.get("raw"):
            raw_html = (
                f'<div class="prov-raw">{esc(ev["raw"])}</div>'
                if full_access
                else '<div class="prov-masked">（原文已脱敏，需完整模式授权）</div>'
            )
        meta_parts = []
        if ev["agent"]:
            meta_parts.append(f"Agent: {esc(ev['agent'])}")
        if ev["layer"]:
            meta_parts.append(f"层: {esc(ev['layer'])}")
        if ev["chain_id"]:
            meta_parts.append(f"Chain: {esc(ev['chain_id'])}")
        if ev["trace_id"]:
            meta_parts.append(f"Trace: {esc(ev['trace_id'])}")
        meta_html = " · ".join(meta_parts)
        nodes.append(
            f'<div class="prov-node" data-type="{ev["type"]}" style="--pc:{ev["color"]}">'
            f'<div class="prov-node-time">{time_str}</div>'
            f'<div class="prov-node-body">'
            f'<span class="prov-node-icon">{ev["icon"]}</span>'
            f'<span class="prov-node-type">{esc(ev["label"])}</span>'
            f'<p class="prov-node-summary">{esc(ev["summary"])}</p>'
            f'{raw_html}'
            f'<div class="prov-node-meta">{meta_html}</div></div></div>'
        )
    body = "\n".join(nodes) if nodes else '<p class="prov-empty">暂无溯源记录</p>'

    html_content = (
        f'<div class="prov-timeline-panel">'
        f'<div class="prov-header"><h3>溯源时间线</h3>{chain_badge}</div>'
        f'<div class="prov-filters">{chips}</div>'
        f'<div class="prov-list" role="log" aria-live="polite">{body}</div>'
        f"</div>"
    )

    config = {
        "type": "provenance_timeline",
        "event_count": total,
        "chain_verification": chain_status,
        "privacy": {"masked": not full_access, "full_access": full_access},
        "event_types": EVENT_TYPES,
    }
    descriptor = build_panel_descriptor(
        artifact, html_content, config,
        ["https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"],
        {"renderer": "ProvenanceTimeline", "events": total, "chain_valid": chain_status["valid"]},
        "ProvenanceTimeline",
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor
