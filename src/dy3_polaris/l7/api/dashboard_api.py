"""L7 API — Dashboard API (dashboard_api.py).

任务拆分 T6 · 设计文档 Ch.9.3。

Dashboard 端点处理器:
- GET /api/v1/dashboard/bkt — 42 KP 完整 BKT 状态 (kp_states/domain_summary/bottleneck_kps)
- GET /api/v1/dashboard/provenance/{kp_id} — 溯源时间线 (?depth=full)
- GET /api/v1/dashboard/contribution/{session_id} — Agent 贡献统计
"""

from __future__ import annotations

from typing import Any

from ..renderers._common import (
    DOMAIN_LABELS,
    KP_DOMAIN_IDS,
    KP_NAMES,
    KP_TO_DOMAIN,
    get_bkt_state,
)
from ..artifact_manager import ArtifactManager
from .error_codes import error_payload


def bkt_dashboard(manager: ArtifactManager, context_bkt: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET /api/v1/dashboard/bkt — 42 KP 完整 BKT 状态.

    Args:
        manager: ArtifactManager (可携带 learner_context)。
        context_bkt: 直接传入的 BKT 状态 (优先)。

    Returns:
        {kp_states, domain_summary, bottleneck_kps}。
    """
    bkt = context_bkt if context_bkt is not None else _extract_bkt_from_manager(manager)

    kp_states: list[dict[str, Any]] = []
    bottlenecks: list[dict[str, Any]] = []
    for kp_id, state in bkt.items():
        entry = {
            "kp_id": kp_id,
            "domain": KP_TO_DOMAIN.get(kp_id, ""),
            "name": KP_NAMES.get(kp_id, kp_id),
            "p_l": round(state.get("p_l", 0.0), 4),
            "p_k_l": round(state.get("p_k_l", 0.0), 4),
            "p_g": round(state.get("p_g", 0.0), 4),
            "p_s": round(state.get("p_s", 0.0), 4),
            "last_updated": state.get("last_updated", None),
        }
        kp_states.append(entry)
        # 瓶颈: P(L)>0.7 且 P(K|L)<0.3
        if state.get("p_l", 0.0) > 0.7 and state.get("p_k_l", 1.0) > 0 and state.get("p_k_l", 0.0) < 0.3:
            bottlenecks.append(entry)

    domain_summary = {}
    for domain, ids in KP_DOMAIN_IDS.items():
        values = [bkt.get(k, {}).get("p_l", 0.0) for k in ids if bkt.get(k, {}).get("p_l", 0.0) > 0]
        domain_summary[domain] = {
            "label": DOMAIN_LABELS[domain],
            "kp_count": len(ids),
            "avg_p_l": round(sum(values) / len(values), 4) if values else 0.0,
            "bottlenecks": sum(1 for k in ids if _is_bottleneck(bkt.get(k, {}))),
        }

    return {
        "kp_states": kp_states,
        "domain_summary": domain_summary,
        "bottleneck_kps": bottlenecks,
    }


def provenance_for_kp(
    kp_id: str,
    events: list[dict[str, Any]] | None = None,
    depth: str = "standard",
) -> dict[str, Any]:
    """GET /api/v1/dashboard/provenance/{kp_id} — 溯源时间线.

    Args:
        kp_id: 目标知识点。
        events: 该 KP 的事件列表 (默认空)。
        depth: summary/standard/full。

    Returns:
        按时间排序的事件列表, 每事件含 timestamp/event_type/agent_id/summary/detail_url。
    """
    if not events:
        return error_payload("DASHBOARD_NO_DATA", details={"kp_id": kp_id})
    normalized = sorted(events, key=lambda e: float(e.get("timestamp", 0.0)))
    result = []
    for ev in normalized:
        result.append({
            "timestamp": float(ev.get("timestamp", 0.0)),
            "event_type": str(ev.get("event_type", "interaction")),
            "agent_id": str(ev.get("agent_id", "")),
            "summary": str(ev.get("summary", "")),
            "detail_url": f"/api/v1/artifacts/{ev.get('artifact_id', '')}"
            if ev.get("artifact_id") else "",
            "depth": depth,
        })
    return {"events": result, "kp_id": kp_id, "depth": depth, "count": len(result)}


def contribution_for_session(
    session_id: str,
    agents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """GET /api/v1/dashboard/contribution/{session_id} — Agent 贡献统计.

    Args:
        session_id: 目标会话。
        agents: [{id, speech_count, citation_count, adopted_count, reputation_delta}]。

    Returns:
        {session_id, agents: {agent_id: {speech_count, citation_count, adopted_count, reputation_delta}}}。
    """
    if not agents:
        return error_payload("DASHBOARD_NO_DATA", details={"session_id": session_id})
    mapping = {}
    for a in agents:
        aid = str(a.get("id") or a.get("agent_id") or "?")
        mapping[aid] = {
            "speech_count": int(a.get("speech_count", 0)),
            "citation_count": int(a.get("citation_count", 0)),
            "adopted_count": int(a.get("adopted_count", 0)),
            "reputation_delta": round(float(a.get("reputation_delta", 0.0)), 4),
        }
    return {"session_id": session_id, "agents": mapping}


def _extract_bkt_from_manager(manager: ArtifactManager) -> dict[str, dict[str, Any]]:
    """从 ArtifactManager 的 artifacts 聚合 BKT 状态."""
    merged: dict[str, dict[str, Any]] = {}
    for artifact in manager.list_artifacts():
        learner = artifact.learner_context or {}
        bkt = learner.get("bkt_state") or {}
        if isinstance(bkt, dict):
            for kp, state in bkt.items():
                if isinstance(state, dict):
                    merged.setdefault(kp, {}).update(state)
    return merged


def _is_bottleneck(state: dict[str, Any]) -> bool:
    return state.get("p_l", 0.0) > 0.7 and state.get("p_k_l", 1.0) > 0 and state.get("p_k_l", 0.0) < 0.3
