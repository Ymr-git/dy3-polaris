"""CC4 三横切集成 — CC2 → CC3 桥接器.

将 CC2 (人机协作审批) 的审批记录自动桥接到 CC3 (溯源捕获层),
写入 KPA 七维标注的「决策维度」, 形成完整的审计-决策-溯源闭环.

核心能力:
- 审批记录 (ApprovalRecord) → KPA 决策维度自动捕获
- 协同层级 (CollaborationLayer) → CC3 approval_level 字符串映射
- 路由结果 (RoutingResult) + 审批记录 → 决策路径 (decision_path) 构建
- 断路器保护 CC3 调用, 防止级联故障
- 桥接事件 (BridgeEvent) 全程记录, 支持统计与回溯

数据流::

    CC2 ApprovalWorkflowManager
        │  ApprovalRecord (record_id, request, decision, status, ...)
        ▼
    CC2CC3Bridge.bridge()
        │  1. 确保 KPA 标注存在 (_ensure_annotation)
        │  2. 提取审批信息 (_extract_approval_info)
        │  3. 映射协同层级 → approval_level (_map_collaboration_layer)
        │  4. 构建决策路径 (_build_decision_path)
        │  5. 调用 CCIntegration.on_cc2_approval_completed() [断路器保护]
        │  6. 记录桥接事件 (BridgeEvent)
        ▼
    CC3 KPA 决策维度 (DecisionDimension)
        - cc2_approval_id
        - cc2_approval_level
        - meta_decider_result
        - decision_path

融合方案:
- Event-Driven Architecture: 审批完成事件驱动溯源写入 (松耦合)
- OpenTelemetry: trace_id / session_id 全链路传递
- Hystrix / Resilience4j: 断路器隔离 CC3 故障, 快速失败
- W3C PROV: 审批决策作为 Activity 记入溯源链
- CloudEvents: 标准化桥接事件格式 (source / type / subject / data)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..cc2.approval_workflow import ApprovalRecord, ApprovalStatus
from ..cc2.routing_engine import CollaborationLayer, RoutingResult
from ..cc3.cc_integration import CCIntegration
from ..cc3.kpa_engine import KPAEngine
from ..cc3.models import TargetType
from .circuit_breaker import CircuitBreaker
from .exceptions import BridgeConnectionError, CircuitBreakerOpenError
from .models import BridgeDirection, BridgeEvent

logger = logging.getLogger(__name__)


# 审批记录对应的 KPA 标注对象类型.
# 优先使用 TargetType.APPROVAL_RECORD (若枚举已定义);
# 否则回退到 REVIEW_REPORT, 保证对当前代码库的兼容性.
_APPROVAL_TARGET_TYPE: TargetType = getattr(
    TargetType, "APPROVAL_RECORD", TargetType.REVIEW_REPORT
)


class CC2CC3Bridge:
    """CC2 → CC3 桥接器 — 审批记录到溯源决策维度的自动捕获.

    在 CC2 审批工作流完成后, 自动将审批记录桥接到 CC3 溯源层,
    写入 KPA 标注的决策维度 (DecisionDimension), 并在溯源链中追加节点.

    职责:
    - 确保 KPA 标注存在 (复用已有或新建)
    - 提取审批记录关键字段 (approval_id / status / mode / decision_by / response_time)
    - 将 CC2 协同层级映射为 CC3 approval_level 字符串
    - 由路由结果与审批记录构建决策路径
    - 通过断路器保护地调用 ``CCIntegration.on_cc2_approval_completed()``
    - 记录桥接事件, 维护统计指标

    使用示例::

        from dy3_polaris.l0.cc2.approval_workflow import (
            ApprovalWorkflowManager,
            ApprovalStatus,
        )
        from dy3_polaris.l0.cc_integration.cc2_cc3_bridge import CC2CC3Bridge

        bridge = CC2CC3Bridge()
        manager = ApprovalWorkflowManager()
        req = manager.create_request(
            operation="data_overwrite", target="kp-001"
        )
        record = manager.make_decision(
            req.request_id,
            decision=ApprovalStatus.APPROVED,
            decided_by="teacher_001",
        )

        result = bridge.bridge(
            approval_record=record,
            target_id="kp-001",
            trace_id="trace-abc",
            session_id="sess-001",
        )
        assert result["success"] is True
    """

    #: CollaborationLayer → CC3 approval_level 字符串映射
    _LAYER_MAP: dict[CollaborationLayer, str] = {
        CollaborationLayer.L1_IMPLICIT: "implicit",
        CollaborationLayer.L2_PROMPT: "prompt",
        CollaborationLayer.L3_APPROVAL: "approval",
        CollaborationLayer.L4_INTERVENTION: "intervention",
    }

    def __init__(
        self,
        cc_integration: CCIntegration | None = None,
        kpa_engine: KPAEngine | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """初始化 CC2 → CC3 桥接器.

        Args:
            cc_integration: CC3 跨切面集成器. 为 None 时内部创建.
            kpa_engine: KPA 标注引擎. 为 None 时从 cc_integration 内部获取或新建.
                注意: 为保证 ``_ensure_annotation`` 创建的标注对
                ``on_cc2_approval_completed`` 可见, 当 cc_integration 提供时,
                将优先复用其内部的 KPA 实例.
            circuit_breaker: 断路器. 为 None 时创建保护 ``cc2_to_cc3`` 的默认实例.
        """
        # 确保 cc_integration 与 kpa_engine 共享同一 KPA 存储,
        # 否则 _ensure_annotation 创建的标注对 on_cc2_approval_completed 不可见.
        if cc_integration is None:
            shared_kpa = kpa_engine if kpa_engine is not None else KPAEngine()
            cc_integration = CCIntegration(kpa_engine=shared_kpa)
            kpa_engine = shared_kpa
        else:
            internal_kpa = getattr(cc_integration, "_kpa", None)
            if internal_kpa is not None:
                # 优先复用 cc_integration 内部 KPA, 保证标注可见性
                kpa_engine = internal_kpa
            elif kpa_engine is None:
                kpa_engine = KPAEngine()

        self._cc_integration: CCIntegration = cc_integration
        self._kpa_engine: KPAEngine = kpa_engine
        self._circuit_breaker: CircuitBreaker = (
            circuit_breaker or CircuitBreaker("cc2_to_cc3")
        )

        self._events: list[BridgeEvent] = []
        self._stats: dict[str, Any] = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "by_approval_level": {},
            "response_times": [],
        }

    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------

    @property
    def cc_integration(self) -> CCIntegration:
        """底层 CC3 跨切面集成器."""
        return self._cc_integration

    @property
    def kpa_engine(self) -> KPAEngine:
        """底层 KPA 标注引擎."""
        return self._kpa_engine

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """保护 CC3 调用的断路器."""
        return self._circuit_breaker

    # --------------------------------------------------------
    # 主桥接方法
    # --------------------------------------------------------

    def bridge(
        self,
        approval_record: ApprovalRecord,
        routing_result: RoutingResult | None = None,
        annotation_id: str | None = None,
        target_id: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """将 CC2 审批记录桥接到 CC3 溯源决策维度.

        执行流程:
            1. 确保 KPA 标注存在 (复用 ``annotation_id`` 或按 ``target_id`` 新建)
            2. 提取审批信息 (approval_id / status / mode / decision_by / response_time)
            3. 将协同层级映射为 CC3 approval_level 字符串
            4. 由路由结果与审批记录构建决策路径
            5. 通过断路器调用 ``CCIntegration.on_cc2_approval_completed()``
            6. 记录桥接事件 (BridgeEvent)
            7. 返回桥接结果

        Args:
            approval_record: CC2 审批记录 (ApprovalRecord)
            routing_result: CC2 路由决策结果 (RoutingResult), 可选
            annotation_id: 已有 KPA 标注 ID, 提供时优先复用
            target_id: 操作目标 ID, 用于新建标注; 为空时从审批记录推导
            trace_id: 全链路 trace ID
            session_id: 会话 ID

        Returns:
            桥接结果::

                {
                    "success": bool,
                    "annotation_id": str,
                    "approval_id": str,
                    "approval_level": str,
                    "completeness": float,
                    "bridge_event_id": str,
                }

        Raises:
            BridgeConnectionError: 桥接过程中发生错误 (CC3 调用失败等)
            CircuitBreakerOpenError: CC3 断路器开启, 快速失败
        """
        if approval_record is None:
            raise BridgeConnectionError(
                "cc2", "cc3", "approval_record 不能为 None"
            )

        # 断路器开启时快速失败, 避免无效工作与孤立标注
        if self._circuit_breaker.is_open:
            status = self._circuit_breaker.get_status()
            opened_at = float(status.get("opened_at", 0.0) or 0.0)
            retry_after = max(
                0.0,
                self._circuit_breaker.config.recovery_timeout
                - (time.time() - opened_at),
            )
            raise CircuitBreakerOpenError(
                self._circuit_breaker.module, retry_after
            )

        # 推导 target_id (优先使用调用方传入, 其次取审批目标, 最后回退到记录 ID)
        if not target_id:
            target_id = (
                approval_record.request.target
                or approval_record.record_id
            )

        start_ts = time.time()

        # 1. 确保 KPA 标注存在
        try:
            annotation_id = self._ensure_annotation(target_id, annotation_id)
        except CircuitBreakerOpenError:
            raise
        except Exception as exc:
            logger.error("确保 KPA 标注失败: %s", exc)
            raise BridgeConnectionError(
                "cc2", "cc3", f"确保 KPA 标注失败: {exc}"
            )

        # 2. 提取审批信息
        approval_info = self._extract_approval_info(approval_record)
        approval_id = approval_info["approval_id"]

        # 审批仍为 PENDING 时给出告警 (不阻断桥接)
        if approval_record.status == ApprovalStatus.PENDING:
            logger.warning(
                "审批记录仍为 PENDING, 桥接可能不完整: %s", approval_id
            )

        # 3. 映射协同层级 → approval_level
        layer = (
            routing_result.recommended_layer
            if routing_result is not None
            else None
        )
        approval_level = self._map_collaboration_layer(layer)

        # 4. 构建决策路径
        decision_path = self._build_decision_path(
            routing_result, approval_record
        )

        # 5. 通过断路器调用 CC3 on_cc2_approval_completed
        meta_decider_result = (
            approval_info["decision"] or approval_info["status"]
        )
        paradigm_selected = str(
            approval_record.metadata.get("paradigm_selected", "")
        )
        debate_id = str(approval_record.metadata.get("debate_id", ""))

        try:
            result = self._circuit_breaker.call(
                self._cc_integration.on_cc2_approval_completed,
                annotation_id=annotation_id,
                approval_id=approval_id,
                approval_level=approval_level,
                meta_decider_result=meta_decider_result,
                paradigm_selected=paradigm_selected,
                debate_id=debate_id,
                decision_path=decision_path,
                trace_id=trace_id,
                session_id=session_id,
            )
        except CircuitBreakerOpenError:
            # 断路器开启, 直接向上传播 (不记为桥接失败)
            logger.warning(
                "CC2→CC3 桥接被断路器拦截: approval=%s", approval_id
            )
            raise
        except Exception as exc:
            logger.error(
                "CC2→CC3 桥接失败: approval=%s, error=%s",
                approval_id,
                exc,
            )
            self._record_event(
                trace_id=trace_id,
                session_id=session_id,
                payload={
                    "annotation_id": annotation_id,
                    "approval_id": approval_id,
                    "approval_level": approval_level,
                    "error": str(exc),
                },
                success=False,
            )
            self._update_stats(
                approval_level, success=False, response_time=0.0
            )
            raise BridgeConnectionError(
                "cc2",
                "cc3",
                f"on_cc2_approval_completed 调用失败: {exc}",
            )

        success = bool(result.get("success", False))
        completeness = float(result.get("completeness", 0.0))
        response_time = approval_info["response_time"]

        # 6. 记录桥接事件
        event = self._record_event(
            trace_id=trace_id,
            session_id=session_id,
            payload={
                "annotation_id": annotation_id,
                "approval_id": approval_id,
                "approval_level": approval_level,
                "completeness": completeness,
                "decision_path": decision_path,
                "approval_info": approval_info,
                "bridge_latency_ms": round(
                    (time.time() - start_ts) * 1000.0, 2
                ),
            },
            success=success,
        )

        # 7. 更新统计
        self._update_stats(
            approval_level,
            success=success,
            response_time=response_time,
        )

        logger.info(
            "CC2→CC3 桥接完成: approval=%s, annotation=%s, level=%s, "
            "completeness=%.2f, success=%s",
            approval_id,
            annotation_id,
            approval_level,
            completeness,
            success,
        )

        return {
            "success": success,
            "annotation_id": annotation_id,
            "approval_id": approval_id,
            "approval_level": approval_level,
            "completeness": completeness,
            "bridge_event_id": event.event_id,
        }

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    def _ensure_annotation(
        self,
        target_id: str,
        annotation_id: str | None,
    ) -> str:
        """创建或复用 KPA 标注.

        - 若 ``annotation_id`` 提供且标注存在, 直接复用;
        - 否则按 ``target_id`` 新建审批记录类型的标注.

        Args:
            target_id: 操作目标 ID
            annotation_id: 已有标注 ID (可选)

        Returns:
            标注 ID (复用或新建)
        """
        if annotation_id:
            try:
                self._kpa_engine.get_annotation(annotation_id)
                logger.debug("复用 KPA 标注: %s", annotation_id)
                return annotation_id
            except Exception:
                logger.warning(
                    "指定的 KPA 标注不存在, 将新建: %s", annotation_id
                )

        annotation = self._kpa_engine.create_annotation(
            target_type=_APPROVAL_TARGET_TYPE,
            target_id=target_id,
            target_metadata={
                "source_module": "cc2",
                "bridge": "cc2_cc3",
            },
            annotator_agent="cc2-cc3-bridge",
        )
        logger.info(
            "新建 KPA 标注: id=%s, target=%s",
            annotation.annotation_id,
            target_id,
        )
        return annotation.annotation_id

    def _extract_approval_info(
        self,
        approval_record: ApprovalRecord,
    ) -> dict[str, Any]:
        """从审批记录提取关键信息.

        Args:
            approval_record: CC2 审批记录

        Returns:
            审批信息字典::

                {
                    "approval_id": str,          # 审批记录 ID (record_id)
                    "request_id": str,
                    "operation": str,
                    "target": str,
                    "risk_level": str,
                    "approval_mode": str,        # 审批模式
                    "status": str,               # 审批状态
                    "decision_by": str,          # 决策人
                    "decision": str,             # 决策结果
                    "comment": str,
                    "modified_parameters": dict,
                    "response_time": float,      # 响应时间 (秒)
                    "metadata": dict,
                }
        """
        request = approval_record.request
        decision = approval_record.decision

        risk_level = (
            request.risk_level.value
            if request.risk_level is not None
            else ""
        )
        approval_mode = (
            request.approval_mode.value
            if request.approval_mode is not None
            else ""
        )
        status = (
            approval_record.status.value
            if approval_record.status is not None
            else ""
        )

        decision_by = ""
        decision_result = ""
        comment = ""
        modified_parameters: dict[str, Any] = {}
        if decision is not None:
            decision_by = decision.decided_by or ""
            decision_result = (
                decision.decision.value
                if decision.decision is not None
                else ""
            )
            comment = decision.comment or ""
            modified_parameters = dict(decision.modified_parameters or {})

        return {
            "approval_id": approval_record.record_id,
            "request_id": request.request_id,
            "operation": request.operation,
            "target": request.target,
            "risk_level": risk_level,
            "approval_mode": approval_mode,
            "status": status,
            "decision_by": decision_by,
            "decision": decision_result,
            "comment": comment,
            "modified_parameters": modified_parameters,
            "response_time": approval_record.response_time_seconds,
            "metadata": dict(approval_record.metadata or {}),
        }

    def _map_collaboration_layer(
        self,
        layer: CollaborationLayer | None,
    ) -> str:
        """将 CC2 协同层级映射为 CC3 approval_level 字符串.

        映射关系:
            - L1_IMPLICIT → "implicit"
            - L2_PROMPT → "prompt"
            - L3_APPROVAL → "approval"
            - L4_INTERVENTION → "intervention"
            - None (无路由结果) → "approval" (默认)

        Args:
            layer: CC2 协同层级, 可为 None

        Returns:
            CC3 approval_level 字符串
        """
        if layer is None:
            return "approval"
        if isinstance(layer, CollaborationLayer):
            return self._LAYER_MAP.get(layer, "approval")
        # 容错: 字符串形式的层级
        try:
            normalized = CollaborationLayer(layer)
        except (ValueError, KeyError):
            return "approval"
        return self._LAYER_MAP.get(normalized, "approval")

    def _build_decision_path(
        self,
        routing_result: RoutingResult | None,
        approval_record: ApprovalRecord,
    ) -> list[str]:
        """由路由结果与审批记录构建决策路径.

        路径格式 (审计轨迹风格)::

            [
                "routing:<rule_id>",     # 匹配的路由规则 ID
                "routing:<layer>",       # 推荐协同层级 (映射后)
                "approval:<mode>",       # 审批模式
                "approval:<status>",     # 审批状态
            ]

        当 ``routing_result`` 为 None 时, 路由段省略, 仅保留审批段.

        Args:
            routing_result: CC2 路由决策结果 (可为 None)
            approval_record: CC2 审批记录

        Returns:
            决策路径字符串列表
        """
        path: list[str] = []

        # 路由段
        if routing_result is not None:
            rule_id = routing_result.rule_id or "unknown"
            path.append(f"routing:{rule_id}")
            layer_str = self._map_collaboration_layer(
                routing_result.recommended_layer
            )
            path.append(f"routing:{layer_str}")

        # 审批模式: 优先取路由结果, 回退到审批记录
        approval_mode = ""
        if (
            routing_result is not None
            and routing_result.approval_mode is not None
        ):
            approval_mode = routing_result.approval_mode.value
        elif approval_record.request.approval_mode is not None:
            approval_mode = approval_record.request.approval_mode.value
        path.append(f"approval:{approval_mode or 'unknown'}")

        # 审批状态
        status = (
            approval_record.status.value
            if approval_record.status is not None
            else "unknown"
        )
        path.append(f"approval:{status}")

        return path

    def _record_event(
        self,
        *,
        trace_id: str,
        session_id: str,
        payload: dict[str, Any],
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> BridgeEvent:
        """记录一次桥接事件 (CloudEvents 风格).

        Args:
            trace_id: 全链路 trace ID
            session_id: 会话 ID
            payload: 事件负载
            success: 桥接是否成功
            metadata: 附加元数据

        Returns:
            已记录的 BridgeEvent
        """
        event = BridgeEvent(
            source="cc2",
            target="cc3",
            direction=BridgeDirection.CC2_TO_CC3,
            event_type="approval_completed",
            trace_id=trace_id,
            session_id=session_id,
            payload=payload,
            metadata={
                "success": success,
                **(metadata or {}),
            },
        )
        self._events.append(event)
        return event

    def _update_stats(
        self,
        approval_level: str,
        *,
        success: bool,
        response_time: float,
    ) -> None:
        """更新桥接统计指标.

        Args:
            approval_level: 本次桥接的 approval_level
            success: 桥接是否成功
            response_time: 审批响应时间 (秒)
        """
        self._stats["total"] += 1
        if success:
            self._stats["successful"] += 1
        else:
            self._stats["failed"] += 1

        by_level: dict[str, int] = self._stats["by_approval_level"]
        by_level[approval_level] = by_level.get(approval_level, 0) + 1

        if response_time and response_time > 0:
            self._stats["response_times"].append(response_time)

    # --------------------------------------------------------
    # 统计与事件查询
    # --------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """获取桥接统计指标.

        Returns:
            统计字典::

                {
                    "total": int,
                    "successful": int,
                    "failed": int,
                    "success_rate": float,
                    "by_approval_level": dict,
                    "avg_response_time": float,
                    "circuit_breaker": dict,
                }
        """
        total: int = self._stats["total"]
        successful: int = self._stats["successful"]
        response_times: list[float] = self._stats["response_times"]

        success_rate = (successful / total) if total > 0 else 0.0
        avg_response = (
            sum(response_times) / len(response_times)
            if response_times
            else 0.0
        )

        return {
            "total": total,
            "successful": successful,
            "failed": self._stats["failed"],
            "success_rate": round(success_rate, 4),
            "by_approval_level": dict(self._stats["by_approval_level"]),
            "avg_response_time": round(avg_response, 2),
            "circuit_breaker": self._circuit_breaker.get_status(),
        }

    def get_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """获取最近的桥接事件.

        Args:
            limit: 最多返回事件数 (按时间倒序, 最新在前)

        Returns:
            桥接事件字典列表
        """
        if limit <= 0:
            return []
        recent = self._events[-limit:]
        return [e.model_dump(mode="json") for e in reversed(recent)]

    def reset(self) -> None:
        """重置桥接器, 清空统计与事件."""
        self._stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "by_approval_level": {},
            "response_times": [],
        }
        self._events.clear()
        self._circuit_breaker.reset()
        logger.info("CC2→CC3 桥接器已重置")


__all__ = ["CC2CC3Bridge"]
