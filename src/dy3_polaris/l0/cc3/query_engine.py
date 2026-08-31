"""CC3 溯源捕获层 — 查询引擎.

提供统一的溯源查询接口, 支持跨 KPA 标注、DL 辩论日志、
溯源链和 L0 Ledger 的联合查询。

核心能力:
- 按 trace_id 全链路回溯
- 按知识点 ID 查询完整溯源档案
- 按时间范围查询事件流
- 按 Agent ID 查询操作历史
- 跨数据源联合查询 (KPA + DL + Chain + Ledger)
- 溯源链路可视化数据生成

融合方案:
- OpenTelemetry: trace_id 端到端回溯
- Langfuse: trace 树结构查询
- OpenLineage: 血缘图查询
- Neo4j Cypher: 图查询语言启发
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .kpa_engine import KPAEngine
from .debate_logger import DebateLogger
from .provenance_chain_builder import ProvenanceChainBuilder
from .ledger_integration import LedgerIntegration
from .models import KPAAnnotation, DebateLog, LedgerEvent, EventType

logger = logging.getLogger(__name__)


class QueryEngine:
    """溯源查询引擎 — 跨数据源联合查询.

    统一查询接口, 支持从 KPA、DL、Chain、Ledger 四个数据源
    进行联合查询和溯源回溯。

    使用示例::

        kpa = KPAEngine()
        dl = DebateLogger()
        chain = ProvenanceChainBuilder()
        ledger = LedgerIntegration()

        engine = QueryEngine(kpa, dl, chain, ledger)
        result = engine.trace_by_trace_id("trace-001")
        result = engine.get_knowledge_provenance("kp-dy3-yag-4f")
    """

    def __init__(
        self,
        kpa_engine: KPAEngine | None = None,
        debate_logger: DebateLogger | None = None,
        chain_builder: ProvenanceChainBuilder | None = None,
        ledger: LedgerIntegration | None = None,
    ) -> None:
        """初始化查询引擎.

        Args:
            kpa_engine: KPA 标注引擎
            debate_logger: 辩论日志引擎
            chain_builder: 溯源链构建器
            ledger: L0 Ledger 集成器
        """
        self._kpa = kpa_engine or KPAEngine()
        self._dl = debate_logger or DebateLogger()
        self._chain = chain_builder or ProvenanceChainBuilder()
        self._ledger = ledger or LedgerIntegration()

    # ==========================================================
    # 按 trace_id 全链路回溯
    # ==========================================================

    def trace_by_trace_id(self, trace_id: str) -> dict[str, Any]:
        """按 trace_id 进行全链路回溯.

        从 L0 Ledger 查询该 trace 的所有事件,
        然后关联查询 KPA 标注和 DL 辩论日志。

        Args:
            trace_id: 全链路 trace ID

        Returns:
            回溯结果::

                {
                    "trace_id": str,
                    "total_events": int,
                    "events": [...],
                    "kpa_annotations": [...],
                    "debate_logs": [...],
                    "timeline": [...],
                }
        """
        events = self._ledger.query(trace_id=trace_id, limit=10000)

        kpa_ids: set[str] = set()
        dl_ids: set[str] = set()

        for event in events:
            if "kpa_annotation" in event.payload:
                kpa_ids.add(event.payload["kpa_annotation"].get("annotation_id", ""))
            if "debate_log" in event.payload:
                dl_ids.add(event.payload["debate_log"].get("debate_log_id", ""))

        kpa_annotations: list[dict[str, Any]] = []
        for aid in kpa_ids:
            if aid:
                try:
                    ann = self._kpa.get_annotation(aid)
                    kpa_annotations.append(ann.model_dump())
                except Exception:
                    pass

        debate_logs: list[dict[str, Any]] = []
        for did in dl_ids:
            if did:
                try:
                    dl = self._dl.get_log(did)
                    debate_logs.append(dl.model_dump())
                except Exception:
                    pass

        # 构建时间线
        timeline = [
            {
                "timestamp": e.timestamp,
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "layer": e.layer,
                "agent_id": e.agent_id,
            }
            for e in sorted(events, key=lambda x: x.timestamp)
        ]

        return {
            "trace_id": trace_id,
            "total_events": len(events),
            "events": [e.model_dump() for e in events],
            "kpa_annotations": kpa_annotations,
            "debate_logs": debate_logs,
            "timeline": timeline,
        }

    # ==========================================================
    # 按知识点 ID 查询完整溯源档案
    # ==========================================================

    def get_knowledge_provenance(
        self,
        knowledge_point_id: str,
    ) -> dict[str, Any]:
        """获取知识点的完整溯源档案.

        查询该知识点的:
        - KPA 七维标注
        - 关联的辩论日志
        - 溯源链节点
        - L0 Ledger 事件

        Args:
            knowledge_point_id: 知识点 ID

        Returns:
            溯源档案
        """
        # 查询 KPA 标注
        annotations = self._kpa.get_by_target(knowledge_point_id)

        # 查询关联的辩论日志
        related_debates: list[dict[str, Any]] = []
        for ann in annotations:
            if ann.decision.debate_id:
                dl = self._dl.get_by_debate(ann.decision.debate_id)
                if dl:
                    related_debates.append(dl.model_dump())

        # 查询 Ledger 事件
        ledger_events = self._ledger.query(limit=10000)
        related_events = [
            e.model_dump() for e in ledger_events
            if knowledge_point_id in str(e.payload)
        ]

        # 汇总
        completeness_scores = [a.completeness_score() for a in annotations]
        avg_completeness = (
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores
            else 0.0
        )

        return {
            "knowledge_point_id": knowledge_point_id,
            "annotations": [a.model_dump() for a in annotations],
            "annotation_count": len(annotations),
            "avg_completeness": round(avg_completeness, 4),
            "debate_logs": related_debates,
            "ledger_events": related_events,
            "has_provenance": len(annotations) > 0,
        }

    # ==========================================================
    # 按 Agent ID 查询操作历史
    # ==========================================================

    def get_agent_history(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> dict[str, Any]:
        """查询 Agent 的操作历史.

        Args:
            agent_id: Agent ID
            limit: 最多返回数

        Returns:
            操作历史
        """
        # KPA 标注
        kpa_annotations = self._kpa.list_annotations(limit=limit)
        agent_annotations = [
            a.model_dump() for a in kpa_annotations
            if a.annotator_agent == agent_id or a.generation.agent_id == agent_id
        ]

        # Ledger 事件
        all_events = self._ledger.query(limit=limit * 10)
        agent_events = [
            e.model_dump() for e in all_events
            if e.agent_id == agent_id
        ][:limit]

        return {
            "agent_id": agent_id,
            "kpa_annotations": agent_annotations,
            "ledger_events": agent_events,
            "total_operations": len(agent_annotations) + len(agent_events),
        }

    # ==========================================================
    # 按时间范围查询事件流
    # ==========================================================

    def get_timeline(
        self,
        start_time: float,
        end_time: float,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """按时间范围查询事件流.

        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            event_type: 事件类型筛选 (None=全部)
            limit: 最多返回数

        Returns:
            时间线事件列表
        """
        events = self._ledger.query_by_time_range(
            start_time, end_time, event_type, limit
        )

        return [
            {
                "timestamp": e.timestamp,
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "trace_id": e.trace_id,
                "session_id": e.session_id,
                "agent_id": e.agent_id,
                "layer": e.layer,
            }
            for e in sorted(events, key=lambda x: x.timestamp)
        ]

    # ==========================================================
    # 溯源链路可视化数据
    # ==========================================================

    def get_provenance_graph(
        self,
        target_id: str = "",
        max_depth: int = 5,
    ) -> dict[str, Any]:
        """生成溯源链路可视化图数据.

        生成 Cytoscape.js 兼容的图数据格式。

        Args:
            target_id: 目标对象 ID (空=全部)
            max_depth: 最大深度

        Returns:
            图数据::

                {
                    "nodes": [{id, label, type, ...}],
                    "edges": [{source, target, label, ...}],
                }
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        node_ids: set[str] = set()

        # 从 KPA 标注构建节点
        annotations = self._kpa.list_annotations(limit=1000)
        for ann in annotations:
            if target_id and ann.target_id != target_id:
                continue

            # 标注节点
            if ann.annotation_id not in node_ids:
                nodes.append({
                    "id": ann.annotation_id,
                    "label": f"KPA: {ann.target_id[:20]}",
                    "type": "annotation",
                    "completeness": ann.completeness_score(),
                })
                node_ids.add(ann.annotation_id)

            # 来源节点
            if ann.source.primary_source and ann.source.primary_source not in node_ids:
                nodes.append({
                    "id": f"src-{ann.source.primary_source[:20]}",
                    "label": ann.source.primary_source[:30],
                    "type": "source",
                    "tier": ann.source.trust_tier.value,
                })
                node_ids.add(f"src-{ann.source.primary_source[:20]}")
                edges.append({
                    "source": f"src-{ann.source.primary_source[:20]}",
                    "target": ann.annotation_id,
                    "label": "wasDerivedFrom",
                })

            # 生成 Agent 节点
            if ann.generation.agent_id and ann.generation.agent_id not in node_ids:
                nodes.append({
                    "id": ann.generation.agent_id,
                    "label": f"Agent: {ann.generation.agent_id[:20]}",
                    "type": "agent",
                })
                node_ids.add(ann.generation.agent_id)
                edges.append({
                    "source": ann.generation.agent_id,
                    "target": ann.annotation_id,
                    "label": "wasGeneratedBy",
                })

            # 关联知识点节点
            if ann.target_id and ann.target_id not in node_ids:
                nodes.append({
                    "id": ann.target_id,
                    "label": f"KP: {ann.target_id[:20]}",
                    "type": "knowledge",
                })
                node_ids.add(ann.target_id)
                edges.append({
                    "source": ann.annotation_id,
                    "target": ann.target_id,
                    "label": "annotates",
                })

        return {
            "nodes": nodes[:max_depth * 20],
            "edges": edges[:max_depth * 20],
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    # ==========================================================
    # 综合统计
    # ==========================================================

    def overview(self) -> dict[str, Any]:
        """获取 CC3 整体概览."""
        kpa_stats = self._kpa.statistics()
        dl_stats = self._dl.statistics()
        ledger_stats = self._ledger.statistics()
        chains = self._chain.list_chains()

        return {
            "kpa": kpa_stats,
            "debate_logs": dl_stats,
            "ledger": ledger_stats,
            "chains": {
                "total": len(chains),
                "chains": chains[:10],
            },
            "timestamp": time.time(),
        }


__all__ = ["QueryEngine"]
