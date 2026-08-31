"""CC3 溯源捕获层 — L0 Ledger 集成.

将 KPA 标注和 DL 辩论日志写入 L0 Provenance Ledger,
作为五类事件的扩展 payload 持久化存储。

核心能力:
- 将 KPA 标注转为 Ledger 事件 (KNOWLEDGE / DECISION)
- 将 DL 辩论日志转为 Ledger 事件 (DECISION)
- 事件哈希链维护 (append-only, 不可修改)
- 事件查询与回溯
- 跨层传递事件记录

融合方案:
- AWS CloudTrail: 不可变审计日志
- PostgreSQL INSERT ONLY: 强制不可变
- TimescaleDB: 时序数据自动分区
- OpenLineage: Dataset/Job/Run 血缘模型
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from typing import Any

from .models import (
    CrossLayerDirection,
    EventType,
    KPAAnnotation,
    DebateLog,
    LedgerEvent,
    TargetType,
)
from .exceptions import (
    CC3Error,
    StorageUnavailableError,
)

logger = logging.getLogger(__name__)


class LedgerIntegration:
    """L0 Provenance Ledger 集成器.

    将 CC3 的 KPA 标注和 DL 辩论日志统一写入 L0 Ledger,
    实现五类事件的 append-only 不可变存储。

    五类事件:
    1. LEARNER_PROFILE: 学习者画像变更
    2. KNOWLEDGE: 知识点创建/更新/采纳
    3. DECISION: 系统决策 (范式选择/辩论/审批)
    4. INTERACTION: 交互记录
    5. HUMAN_OVERRIDE: 人工干预

    使用示例::

        ledger = LedgerIntegration()
        event = ledger.write_kpa(annotation, trace_id="trace-001")
        event = ledger.write_dl(debate_log, trace_id="trace-001")
        events = ledger.query(trace_id="trace-001")
    """

    def __init__(self) -> None:
        """初始化 Ledger 集成器."""
        self._events: list[LedgerEvent] = []
        self._event_index: dict[str, LedgerEvent] = {}
        self._trace_index: dict[str, list[str]] = {}
        self._session_index: dict[str, list[str]] = {}
        self._lock = threading.RLock()

    # ==========================================================
    # 写入 KPA 标注
    # ==========================================================

    def write_kpa(
        self,
        annotation: KPAAnnotation,
        trace_id: str = "",
        session_id: str = "",
    ) -> LedgerEvent:
        """将 KPA 标注写入 L0 Ledger.

        根据 target_type 映射到事件类型:
        - KNOWLEDGE_POINT → KNOWLEDGE
        - DECISION → DECISION
        - 其他 → INTERACTION

        Args:
            annotation: KPA 标注
            trace_id: 全链路 trace ID
            session_id: 会话 ID

        Returns:
            创建的 LedgerEvent
        """
        with self._lock:
            # 映射事件类型
            type_map = {
                TargetType.KNOWLEDGE_POINT: EventType.KNOWLEDGE,
                TargetType.DECISION: EventType.DECISION,
                TargetType.CONTENT: EventType.INTERACTION,
                TargetType.ARTIFACT: EventType.INTERACTION,
                TargetType.DEBATE_OUTCOME: EventType.DECISION,
                TargetType.REVIEW_REPORT: EventType.INTERACTION,
            }
            event_type = type_map.get(
                annotation.target_type, EventType.INTERACTION
            )

            # 构建 payload
            payload = {
                "kpa_annotation": annotation.model_dump(),
                "completeness_score": annotation.completeness_score(),
                "filled_dimensions": annotation.filled_dimensions(),
            }

            event = self._create_event(
                event_type=event_type,
                trace_id=trace_id,
                session_id=session_id,
                agent_id=annotation.annotator_agent,
                layer="CC3",
                payload=payload,
            )

            logger.info(
                "写入 KPA 到 Ledger: event=%s, annotation=%s, type=%s",
                event.event_id,
                annotation.annotation_id,
                event_type.value,
            )
            return event

    # ==========================================================
    # 写入 DL 辩论日志
    # ==========================================================

    def write_dl(
        self,
        debate_log: DebateLog,
        trace_id: str = "",
        session_id: str = "",
    ) -> LedgerEvent:
        """将 DL 辩论日志写入 L0 Ledger.

        辩论日志映射为 DECISION 事件。

        Args:
            debate_log: 辩论日志
            trace_id: 全链路 trace ID
            session_id: 会话 ID

        Returns:
            创建的 LedgerEvent
        """
        with self._lock:
            payload = {
                "debate_log": debate_log.model_dump(),
                "convergence_reached": debate_log.convergence_reached,
                "final_divergence": debate_log.final_divergence,
                "total_rounds": len(debate_log.rounds),
            }

            event = self._create_event(
                event_type=EventType.DECISION,
                trace_id=trace_id or debate_log.session_id,
                session_id=session_id or debate_log.session_id,
                agent_id="debate-system",
                layer="CC3",
                payload=payload,
            )

            logger.info(
                "写入 DL 到 Ledger: event=%s, debate_log=%s, converged=%s",
                event.event_id,
                debate_log.debate_log_id,
                debate_log.convergence_reached,
            )
            return event

    # ==========================================================
    # 写入跨层传递事件
    # ==========================================================

    def write_cross_layer(
        self,
        direction: CrossLayerDirection,
        trace_id: str = "",
        session_id: str = "",
        agent_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        """写入跨层传递事件.

        记录数据在架构层之间的传递, 支持 8 种方向。

        Args:
            direction: 跨层方向
            trace_id: 全链路 trace ID
            session_id: 会话 ID
            agent_id: 传递 Agent ID
            payload: 传递的数据

        Returns:
            创建的 LedgerEvent
        """
        with self._lock:
            event = self._create_event(
                event_type=EventType.INTERACTION,
                trace_id=trace_id,
                session_id=session_id,
                agent_id=agent_id,
                layer=direction.value,
                payload={
                    "cross_layer_direction": direction.value,
                    "data": payload or {},
                },
            )

            logger.info(
                "写入跨层事件: event=%s, direction=%s",
                event.event_id,
                direction.value,
            )
            return event

    # ==========================================================
    # 写入人工干预事件
    # ==========================================================

    def write_human_override(
        self,
        trace_id: str = "",
        session_id: str = "",
        agent_id: str = "",
        override_type: str = "",
        override_detail: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        """写入人工干预事件.

        Args:
            trace_id: 全链路 trace ID
            session_id: 会话 ID
            agent_id: 干预者 ID
            override_type: 干预类型
            override_detail: 干预详情

        Returns:
            创建的 LedgerEvent
        """
        with self._lock:
            event = self._create_event(
                event_type=EventType.HUMAN_OVERRIDE,
                trace_id=trace_id,
                session_id=session_id,
                agent_id=agent_id,
                layer="CC3",
                payload={
                    "override_type": override_type,
                    "detail": override_detail or {},
                },
            )

            logger.info(
                "写入人工干预: event=%s, type=%s",
                event.event_id,
                override_type,
            )
            return event

    # ==========================================================
    # 内部事件创建
    # ==========================================================

    def _create_event(
        self,
        event_type: EventType,
        trace_id: str = "",
        session_id: str = "",
        agent_id: str = "",
        layer: str = "",
        payload: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        """创建 Ledger 事件 (append-only).

        自动维护 prev_hash 链式结构。
        """
        prev_hash = self._events[-1].event_hash if self._events else ""

        event = LedgerEvent(
            event_type=event_type,
            trace_id=trace_id,
            session_id=session_id,
            agent_id=agent_id,
            layer=layer,
            payload=payload or {},
            prev_hash=prev_hash,
        )
        event.event_hash = event.compute_event_hash()

        self._events.append(event)
        self._event_index[event.event_id] = event
        if trace_id:
            self._trace_index.setdefault(trace_id, []).append(event.event_id)
        if session_id:
            self._session_index.setdefault(session_id, []).append(event.event_id)

        return event

    # ==========================================================
    # 查询
    # ==========================================================

    def get_event(self, event_id: str) -> LedgerEvent | None:
        """按 ID 获取事件."""
        with self._lock:
            return self._event_index.get(event_id)

    def query(
        self,
        trace_id: str = "",
        session_id: str = "",
        event_type: EventType | None = None,
        layer: str = "",
        limit: int = 100,
    ) -> list[LedgerEvent]:
        """查询事件.

        Args:
            trace_id: 按 trace ID 筛选
            session_id: 按会话 ID 筛选
            event_type: 按事件类型筛选
            layer: 按层筛选
            limit: 最多返回数

        Returns:
            事件列表
        """
        with self._lock:
            results: list[LedgerEvent] = []

            # 优先使用索引
            if trace_id:
                ids = self._trace_index.get(trace_id, [])
                candidates = [self._event_index[eid] for eid in ids if eid in self._event_index]
            elif session_id:
                ids = self._session_index.get(session_id, [])
                candidates = [self._event_index[eid] for eid in ids if eid in self._event_index]
            else:
                candidates = list(self._events)

            for event in candidates:
                if event_type is not None and event.event_type != event_type:
                    continue
                if layer and event.layer != layer:
                    continue
                results.append(event)
                if len(results) >= limit:
                    break

            return results

    def query_by_time_range(
        self,
        start_time: float,
        end_time: float,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[LedgerEvent]:
        """按时间范围查询事件."""
        with self._lock:
            results = []
            for event in self._events:
                if event.timestamp < start_time or event.timestamp > end_time:
                    continue
                if event_type is not None and event.event_type != event_type:
                    continue
                results.append(event)
                if len(results) >= limit:
                    break
            return results

    # ==========================================================
    # 完整性验证
    # ==========================================================

    def verify_ledger(self) -> dict[str, Any]:
        """验证整个 Ledger 的完整性.

        检查所有事件的哈希链是否完整。
        """
        with self._lock:
            total = len(self._events)
            passed = 0
            failed = 0
            failures: list[dict[str, Any]] = []

            for i, event in enumerate(self._events):
                issues: list[str] = []

                # 检查 prev_hash
                expected_prev = self._events[i - 1].event_hash if i > 0 else ""
                if event.prev_hash != expected_prev:
                    issues.append("prev_hash 不匹配")

                # 检查 event_hash
                computed = event.compute_event_hash()
                if event.event_hash != computed:
                    issues.append("event_hash 不匹配")

                if issues:
                    failed += 1
                    failures.append({
                        "event_id": event.event_id,
                        "index": i,
                        "issues": issues,
                    })
                else:
                    passed += 1

            return {
                "total_events": total,
                "passed": passed,
                "failed": failed,
                "all_passed": failed == 0,
                "failures": failures,
            }

    # ==========================================================
    # 统计
    # ==========================================================

    def statistics(self) -> dict[str, Any]:
        """获取 Ledger 统计信息."""
        with self._lock:
            total = len(self._events)
            if total == 0:
                return {"total": 0}

            by_type: dict[str, int] = {}
            by_layer: dict[str, int] = {}
            for event in self._events:
                t = event.event_type.value
                by_type[t] = by_type.get(t, 0) + 1
                if event.layer:
                    by_layer[event.layer] = by_layer.get(event.layer, 0) + 1

            return {
                "total": total,
                "by_type": by_type,
                "by_layer": by_layer,
                "unique_traces": len(self._trace_index),
                "unique_sessions": len(self._session_index),
            }

    # ==========================================================
    # 清空 (测试用)
    # ==========================================================

    def clear(self) -> None:
        """清空所有事件."""
        with self._lock:
            self._events.clear()
            self._event_index.clear()
            self._trace_index.clear()
            self._session_index.clear()


__all__ = ["LedgerIntegration"]
