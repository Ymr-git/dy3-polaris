"""CC4 三横切集成 — CC1→CC2 桥接器.

将 CC1 (四层反幻觉评审) 的评审结果桥接到 CC2 (六维决策路由 + 人机协同审批),
形成 "评审 → 路由 → 审批" 的完整治理链路.

桥接流程::

    ReviewResult (CC1)
        │  verdict → confidence 转换 (衰减策略)
        ▼
    RoutingContext (CC2)
        │  RoutingEngine.route()
        ▼
    RoutingResult (CC2)
        │  recommended_layer == L3_APPROVAL ?
        ▼
    ApprovalRequest (CC2)  ←─ 自动创建 (ApprovalWorkflowManager)
        │
        ▼
    BridgeEvent (审计事件, CloudEvents 格式)

置信度转换策略 (verdict → confidence):
    ┌─────────┬───────────────────────────────────┬──────────┐
    │ verdict │ 公式                              │ 说明     │
    ├─────────┼───────────────────────────────────┼──────────┤
    │ PASS    │ composite_score / 100 × 1.0       │ 无衰减   │
    │ FLAG    │ composite_score / 100 × 0.7       │ 衰减 30% │
    │ BLOCK   │ composite_score / 100 × 0.3       │ 衰减 70% │
    └─────────┴───────────────────────────────────┴──────────┘

断路器保护:
- 所有 CC2 调用 (route / create_request) 均经过 CircuitBreaker 保护
- CC2 连续失败达阈值时断路器跳闸, 桥接降级返回错误事件
- 避免级联故障影响 CC1 评审主流程

融合世界先进方案:
- Service Mesh (Istio): 横切关注点统一编排 + 断路器隔离
- Event-Driven Architecture: CloudEvents 标准化桥接事件
- OpenTelemetry: trace_id 全链路传递
- Control Plane (Kubernetes): 声明式治理上下文 (GovernanceContext)
- Hystrix / Resilience4j: 断路器三态保护
- GAIA: 不可逆操作承诺检测 → 强制升级审批
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .models import (
    BridgeDirection,
    BridgeEvent,
    GovernanceContext,
    GovernanceDecision,
)
from .exceptions import BridgeConnectionError, CircuitBreakerOpenError
from .circuit_breaker import CircuitBreaker
from ..cc1.review_pipeline import ReviewResult, ReviewVerdict
from ..cc2.routing_engine import (
    ApprovalMode,
    CollaborationLayer,
    Reversibility,
    RiskLevel,
    RoutingContext,
    RoutingEngine,
    RoutingResult,
    UserRole,
)
from ..cc2.approval_workflow import ApprovalRequest, ApprovalWorkflowManager

logger = logging.getLogger(__name__)


#: 评审判决 → 置信度衰减系数.
#:
#: PASS 无衰减 (1.0), FLAG 衰减 30% (0.7), BLOCK 衰减 70% (0.3).
#: 低置信度会驱动 CC2 路由引擎向更高协同层级升级.
VERDICT_CONFIDENCE_MULTIPLIER: dict[ReviewVerdict, float] = {
    ReviewVerdict.PASS: 1.0,
    ReviewVerdict.FLAG: 0.7,
    ReviewVerdict.BLOCK: 0.3,
}


class CC1CC2Bridge:
    """CC1→CC2 桥接器 — 评审结果注入路由决策.

    将 CC1 四层反幻觉评审的 :class:`ReviewResult` 转换为
    CC2 六维路由上下文 :class:`RoutingContext`, 调用
    :class:`RoutingEngine` 进行决策路由, 并在路由到 L3 审批层时
    自动创建 :class:`ApprovalRequest`.

    核心职责:
        1. verdict → confidence 转换 (PASS / FLAG / BLOCK 衰减策略)
        2. CC1 元数据注入 (cc1_verdict, cc1_score, cc1_report_id, ...)
        3. 断路器保护的 CC2 调用
        4. L3 审批请求自动创建
        5. 桥接事件审计 (CloudEvents 格式)
        6. 治理上下文构建 (GovernanceContext / GovernanceDecision)

    使用示例::

        from dy3_polaris.l0.cc1.review_pipeline import ReviewPipeline
        from dy3_polaris.l0.cc2.routing_engine import RiskLevel
        from dy3_polaris.l0.cc_integration import CC1CC2Bridge

        pipeline = ReviewPipeline()
        result = pipeline.review(request)

        bridge = CC1CC2Bridge()
        outcome = bridge.bridge(
            review_result=result,
            operation_type="content_generation",
            risk_level=RiskLevel.MEDIUM,
            user_id="student-001",
        )

        if outcome["success"]:
            layer = outcome["routing_result"]["recommended_layer"]
            if outcome["approval_request"] is not None:
                print("需人工审批:", outcome["approval_request"]["request_id"])

    Note:
        本桥接器在 ``bridge`` 调用期间会将当前路由上下文暂存于实例属性,
        因此 **非线程安全**。并发场景下请为每个线程 / 任务创建独立实例,
        或通过外部锁串行化 ``bridge`` 调用。
    """

    #: 事件日志上限 (超出后保留最近一半)
    _MAX_EVENTS: int = 1000

    def __init__(
        self,
        routing_engine: RoutingEngine | None = None,
        approval_manager: ApprovalWorkflowManager | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """初始化 CC1→CC2 桥接器.

        所有依赖项为 None 时自动创建默认实例, 开箱即用.

        Args:
            routing_engine: CC2 路由引擎, 为 None 时创建默认实例.
            approval_manager: CC2 审批工作流管理器,
                为 None 时创建默认实例.
            circuit_breaker: 断路器, 为 None 时创建保护 CC2 的默认实例.
        """
        self._routing_engine: RoutingEngine = (
            routing_engine or RoutingEngine()
        )
        self._approval_manager: ApprovalWorkflowManager = (
            approval_manager or ApprovalWorkflowManager()
        )
        self._circuit_breaker: CircuitBreaker = (
            circuit_breaker or CircuitBreaker(module="cc2")
        )

        # 当前桥接上下文 (bridge 调用期间暂存, 供 _create_approval_if_needed 使用)
        self._current_context: RoutingContext | None = None

        # 桥接事件审计日志
        self._events: list[BridgeEvent] = []

        # 桥接统计
        self._stats: dict[str, Any] = {
            "total_bridges": 0,
            "successful_bridges": 0,
            "failed_bridges": 0,
            "circuit_breaker_trips": 0,
            "approvals_created": 0,
            "by_verdict": {"pass": 0, "flag": 0, "block": 0},
            "by_layer": {
                "l1_implicit": 0,
                "l2_prompt": 0,
                "l3_approval": 0,
                "l4_intervention": 0,
            },
            "total_confidence_sum": 0.0,
            "total_latency_ms_sum": 0.0,
        }

    # ========================================================
    # 属性
    # ========================================================

    @property
    def routing_engine(self) -> RoutingEngine:
        """CC2 路由引擎."""
        return self._routing_engine

    @property
    def approval_manager(self) -> ApprovalWorkflowManager:
        """CC2 审批工作流管理器."""
        return self._approval_manager

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """断路器."""
        return self._circuit_breaker

    # ========================================================
    # 核心桥接方法
    # ========================================================

    def bridge(
        self,
        review_result: ReviewResult,
        operation_type: str = "",
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        reversibility: Reversibility = Reversibility.PARTIALLY_REVERSIBLE,
        user_role: UserRole = UserRole.STUDENT,
        cognitive_load: float = 0.45,
        trust_score: float = 0.90,
        user_id: str = "",
        session_id: str = "",
        trace_id: str = "",
        **extra_metadata: Any,
    ) -> dict[str, Any]:
        """将 CC1 评审结果桥接到 CC2 路由决策.

        执行流程:
            1. 将 ``ReviewResult.verdict`` 转换为 ``confidence``
               (PASS / FLAG / BLOCK 衰减策略).
            2. 构建 :class:`RoutingContext`, 注入 CC1 元数据
               (cc1_verdict, cc1_score, cc1_report_id, ...).
            3. 经断路器调用 :meth:`RoutingEngine.route`.
            4. 若路由到 ``L3_APPROVAL``, 自动创建
               :class:`ApprovalRequest`.
            5. 记录 :class:`BridgeEvent` 审计事件.
            6. 返回桥接结果字典.

        Args:
            review_result: CC1 评审结果.
            operation_type: 操作类型标识 (如 ``"learning_path_reset"``).
            risk_level: 操作风险等级.
            reversibility: 操作可逆性.
            user_role: 用户角色.
            cognitive_load: 当前认知负荷 (0-1).
            trust_score: 用户信任度 (0-1).
            user_id: 用户 ID.
            session_id: 会话 ID.
            trace_id: OpenTelemetry 全链路 trace ID.
            **extra_metadata: 附加元数据, 注入到路由上下文 ``metadata``
                (如 ``consecutive_errors``, ``safety_related`` 等,
                可触 CC2 场景化路由规则).

        Returns:
            桥接结果字典::

                {
                    "success": bool,              # 桥接是否成功
                    "routing_result": dict|None,  # 路由决策结果
                    "approval_request": dict|None,# 审批请求 (仅 L3)
                    "bridge_event_id": str,       # 桥接事件 ID
                    "error": str,                 # 错误信息 (失败时)
                }
        """
        start_time = time.time()
        self._stats["total_bridges"] += 1

        # --------------------------------------------------------
        # 输入校验
        # --------------------------------------------------------
        if review_result is None:
            event = self._record_event(
                review_result=None,
                routing_result=None,
                approval_request=None,
                success=False,
                error="review_result 不能为 None",
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
                latency_ms=0.0,
            )
            self._stats["failed_bridges"] += 1
            return {
                "success": False,
                "routing_result": None,
                "approval_request": None,
                "bridge_event_id": event.event_id,
                "error": "review_result 不能为 None",
            }

        verdict = review_result.verdict
        verdict_key = verdict.value if hasattr(verdict, "value") else str(verdict)

        # 统计: 按 verdict
        if verdict_key in self._stats["by_verdict"]:
            self._stats["by_verdict"][verdict_key] += 1

        # --------------------------------------------------------
        # Step 1 + 2: 构建路由上下文
        # --------------------------------------------------------
        try:
            routing_context = self._build_routing_context(
                review_result=review_result,
                operation_type=operation_type,
                risk_level=risk_level,
                reversibility=reversibility,
                user_role=user_role,
                cognitive_load=cognitive_load,
                trust_score=trust_score,
                user_id=user_id,
                session_id=session_id,
                extra_metadata=extra_metadata,
            )
        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000.0
            self._stats["total_latency_ms_sum"] += latency_ms
            self._stats["failed_bridges"] += 1
            error_msg = f"路由上下文构建失败: {exc}"
            logger.exception(
                "CC1→CC2 路由上下文构建异常: report_id=%s",
                review_result.report_id,
            )
            event = self._record_event(
                review_result=review_result,
                routing_result=None,
                approval_request=None,
                success=False,
                error=error_msg,
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
                latency_ms=latency_ms,
            )
            return {
                "success": False,
                "routing_result": None,
                "approval_request": None,
                "bridge_event_id": event.event_id,
                "error": error_msg,
            }

        # 暂存上下文供 _create_approval_if_needed 使用
        self._current_context = routing_context

        # 统计置信度
        self._stats["total_confidence_sum"] += routing_context.confidence

        # --------------------------------------------------------
        # Step 3: 经断路器调用路由引擎
        # --------------------------------------------------------
        routing_result: RoutingResult | None = None
        error_msg = ""

        try:
            routing_result = self._circuit_breaker.call(
                self._routing_engine.route, routing_context
            )
        except CircuitBreakerOpenError as exc:
            error_msg = str(exc)
            self._stats["circuit_breaker_trips"] += 1
            logger.warning(
                "CC1→CC2 桥接断路器跳闸: report_id=%s, %s",
                review_result.report_id,
                exc,
            )
        except Exception as exc:
            bridge_error = BridgeConnectionError(
                source="cc1",
                target="cc2",
                reason=str(exc),
            )
            error_msg = str(bridge_error)
            logger.exception(
                "CC1→CC2 路由调用异常: report_id=%s",
                review_result.report_id,
            )

        if routing_result is None:
            # 路由失败 — 记录降级事件并返回
            latency_ms = (time.time() - start_time) * 1000.0
            self._stats["total_latency_ms_sum"] += latency_ms
            self._stats["failed_bridges"] += 1
            event = self._record_event(
                review_result=review_result,
                routing_result=None,
                approval_request=None,
                success=False,
                error=error_msg,
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
                latency_ms=latency_ms,
            )
            self._current_context = None
            return {
                "success": False,
                "routing_result": None,
                "approval_request": None,
                "bridge_event_id": event.event_id,
                "error": error_msg,
            }

        # --------------------------------------------------------
        # Step 4: L3 审批请求自动创建
        # --------------------------------------------------------
        approval_request = self._create_approval_if_needed(
            routing_result=routing_result,
            review_result=review_result,
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
        )

        # --------------------------------------------------------
        # Step 5: 统计 + 事件记录
        # --------------------------------------------------------
        layer_key = routing_result.recommended_layer.value
        if layer_key in self._stats["by_layer"]:
            self._stats["by_layer"][layer_key] += 1

        if approval_request is not None:
            self._stats["approvals_created"] += 1

        latency_ms = (time.time() - start_time) * 1000.0
        self._stats["total_latency_ms_sum"] += latency_ms
        self._stats["successful_bridges"] += 1

        event = self._record_event(
            review_result=review_result,
            routing_result=routing_result,
            approval_request=approval_request,
            success=True,
            error="",
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            latency_ms=latency_ms,
        )

        # 清理暂存上下文
        self._current_context = None

        # --------------------------------------------------------
        # Step 6: 返回结果
        # --------------------------------------------------------
        return {
            "success": True,
            "routing_result": routing_result.model_dump(),
            "approval_request": (
                approval_request.model_dump()
                if approval_request is not None
                else None
            ),
            "bridge_event_id": event.event_id,
            "error": "",
        }

    # ========================================================
    # 内部方法
    # ========================================================

    def _build_routing_context(
        self,
        review_result: ReviewResult,
        operation_type: str,
        risk_level: RiskLevel,
        reversibility: Reversibility,
        user_role: UserRole,
        cognitive_load: float,
        trust_score: float,
        user_id: str,
        session_id: str,
        extra_metadata: dict[str, Any],
    ) -> RoutingContext:
        """将 CC1 评审结果转换为 CC2 路由上下文.

        转换逻辑:
            - verdict → confidence (衰减策略, 见
              :data:`VERDICT_CONFIDENCE_MULTIPLIER`)
            - 注入 CC1 元数据到 ``metadata`` (cc1_verdict, cc1_score,
              cc1_report_id, cc1_issues_count, cc1_self_correction_count)

        Args:
            review_result: CC1 评审结果.
            operation_type: 操作类型.
            risk_level: 风险等级.
            reversibility: 可逆性.
            user_role: 用户角色.
            cognitive_load: 认知负荷 (0-1).
            trust_score: 信任度 (0-1).
            user_id: 用户 ID.
            session_id: 会话 ID.
            extra_metadata: 调用方附加元数据.

        Returns:
            CC2 六维路由上下文.
        """
        confidence = self._verdict_to_confidence(
            review_result.verdict, review_result.composite_score
        )

        # 自纠次数
        self_correction = review_result.self_correction
        self_correction_count = (
            self_correction.attempts if self_correction is not None else 0
        )

        # 构建 CC1 元数据
        metadata: dict[str, Any] = {
            "cc1_verdict": review_result.verdict.value,
            "cc1_score": review_result.composite_score,
            "cc1_report_id": review_result.report_id,
            "cc1_issues_count": len(review_result.issues),
            "cc1_self_correction_count": self_correction_count,
        }

        # 合并调用方附加元数据 (可触 CC2 场景化路由规则,
        # 如 consecutive_errors / safety_related / trust_mode_active)
        if extra_metadata:
            metadata.update(extra_metadata)

        # 钳制数值范围, 避免 RoutingContext 校验异常
        clamped_trust = max(0.0, min(1.0, trust_score))
        clamped_load = max(0.0, min(1.0, cognitive_load))

        return RoutingContext(
            operation_type=operation_type,
            risk_level=risk_level,
            confidence=confidence,
            trust_score=clamped_trust,
            reversibility=reversibility,
            user_role=user_role,
            cognitive_load=clamped_load,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )

    @staticmethod
    def _verdict_to_confidence(
        verdict: ReviewVerdict, composite_score: float
    ) -> float:
        """将评审判决转换为路由置信度.

        衰减策略:
            - PASS  → composite_score / 100 × 1.0
            - FLAG  → composite_score / 100 × 0.7
            - BLOCK → composite_score / 100 × 0.3

        结果钳制在 [0.0, 1.0] 区间.

        Args:
            verdict: 评审判决 (PASS / FLAG / BLOCK).
            composite_score: CC1 综合评分 (0-100).

        Returns:
            路由置信度 (0-1), 越低越倾向于升级协同层级.
        """
        multiplier = VERDICT_CONFIDENCE_MULTIPLIER.get(verdict, 1.0)
        confidence = (composite_score / 100.0) * multiplier
        return round(max(0.0, min(1.0, confidence)), 4)

    def _create_approval_if_needed(
        self,
        routing_result: RoutingResult,
        review_result: ReviewResult,
        user_id: str,
        session_id: str,
        trace_id: str,
    ) -> ApprovalRequest | None:
        """若路由到 L3 审批层, 自动创建审批请求.

        当 :attr:`RoutingResult.recommended_layer` 为
        ``L3_APPROVAL`` 时, 通过
        :meth:`ApprovalWorkflowManager.create_request` 创建标准化
        审批请求, 携带 CC1 评审上下文.

        Args:
            routing_result: 路由决策结果.
            review_result: CC1 评审结果.
            user_id: 用户 ID (用于信任模式检查).
            session_id: 会话 ID.
            trace_id: trace ID.

        Returns:
            审批请求; 若未路由到 L3 或创建失败则返回 None.
        """
        if routing_result.recommended_layer != CollaborationLayer.L3_APPROVAL:
            return None

        # 从暂存的路由上下文获取操作类型等参数
        ctx = self._current_context
        if ctx is None:
            logger.warning(
                "创建审批请求时路由上下文缺失: report_id=%s",
                review_result.report_id,
            )
            return None

        # 确定审批模式 (优先使用路由结果中的模式)
        approval_mode = (
            routing_result.approval_mode
            if routing_result.approval_mode is not None
            else ApprovalMode.DETAILED_REVIEW
        )

        # 确定审批人角色
        approver_roles = self._determine_approver_roles(ctx.user_role)

        # 构建审批上下文 (携带 CC1 评审信息, 供审批人参考)
        approval_context: dict[str, Any] = {
            "cc1_report_id": review_result.report_id,
            "cc1_verdict": review_result.verdict.value,
            "cc1_score": review_result.composite_score,
            "cc1_issues": review_result.issues,
            "cc1_self_correction_count": (
                review_result.self_correction.attempts
                if review_result.self_correction is not None
                else 0
            ),
            "session_id": session_id,
            "trace_id": trace_id,
            "routing_rule_id": routing_result.rule_id,
            "routing_score": routing_result.score,
        }

        try:
            approval_request = self._circuit_breaker.call(
                self._approval_manager.create_request,
                operation=ctx.operation_type,
                target="",
                risk_level=ctx.risk_level,
                reversibility=ctx.reversibility,
                approval_mode=approval_mode,
                requester=review_result.agent_id or user_id,
                approver_roles=approver_roles,
                timeout_seconds=routing_result.timeout_seconds,
                timeout_action=routing_result.timeout_action,
                alternatives=routing_result.alternatives,
                context=approval_context,
                policy_reference=(
                    routing_result.policy_reference
                    or routing_result.rule_id
                ),
                user_id=user_id,
            )
            logger.info(
                "CC1→CC2 自动创建审批请求: request_id=%s, operation=%s, "
                "mode=%s, report_id=%s",
                approval_request.request_id,
                ctx.operation_type,
                approval_mode.value,
                review_result.report_id,
            )
            return approval_request
        except CircuitBreakerOpenError as exc:
            logger.warning(
                "创建审批请求时断路器跳闸: report_id=%s, %s",
                review_result.report_id,
                exc,
            )
            self._stats["circuit_breaker_trips"] += 1
            return None
        except Exception:
            logger.exception(
                "创建审批请求失败: report_id=%s",
                review_result.report_id,
            )
            return None

    @staticmethod
    def _determine_approver_roles(
        user_role: UserRole,
    ) -> list[UserRole]:
        """根据操作发起者角色确定审批人角色.

        路由策略:
            - 学生操作 → 教师审批
            - 教师操作 → 管理员审批
            - 管理员 / 系统操作 → 管理员审批

        Args:
            user_role: 操作发起者角色.

        Returns:
            审批人角色列表.
        """
        if user_role == UserRole.STUDENT:
            return [UserRole.TEACHER]
        elif user_role == UserRole.TEACHER:
            return [UserRole.ADMIN]
        else:
            return [UserRole.ADMIN]

    def _record_event(
        self,
        review_result: ReviewResult | None,
        routing_result: RoutingResult | None,
        approval_request: ApprovalRequest | None,
        success: bool,
        error: str,
        trace_id: str,
        session_id: str,
        user_id: str,
        latency_ms: float = 0.0,
    ) -> BridgeEvent:
        """记录桥接审计事件 (CloudEvents 格式).

        构建 :class:`BridgeEvent`, 包含 CC1 评审摘要、CC2 路由决策、
        审批请求信息, 以及 :class:`GovernanceContext` /
        :class:`GovernanceDecision` 治理快照.

        Args:
            review_result: CC1 评审结果.
            routing_result: 路由决策结果.
            approval_request: 审批请求.
            success: 桥接是否成功.
            error: 错误信息.
            trace_id: trace ID.
            session_id: 会话 ID.
            user_id: 用户 ID.
            latency_ms: 桥接延迟 (毫秒).

        Returns:
            已记录的桥接事件.
        """
        # --- 构建事件负载 ---
        payload: dict[str, Any] = {
            "success": success,
            "error": error,
            "latency_ms": round(latency_ms, 2),
        }

        if review_result is not None:
            payload["cc1"] = {
                "report_id": review_result.report_id,
                "verdict": review_result.verdict.value,
                "composite_score": review_result.composite_score,
                "issues_count": len(review_result.issues),
                "self_correction_triggered": (
                    review_result.self_correction is not None
                ),
            }

        if routing_result is not None:
            payload["cc2_routing"] = {
                "result_id": routing_result.result_id,
                "recommended_layer": routing_result.recommended_layer.value,
                "approval_mode": (
                    routing_result.approval_mode.value
                    if routing_result.approval_mode is not None
                    else None
                ),
                "score": routing_result.score,
                "rule_id": routing_result.rule_id,
                "reasoning": routing_result.reasoning,
            }

        if approval_request is not None:
            payload["cc2_approval"] = {
                "request_id": approval_request.request_id,
                "operation": approval_request.operation,
                "approval_mode": approval_request.approval_mode.value,
                "timeout_seconds": approval_request.timeout_seconds,
            }

        # --- 构建治理快照 (声明式状态模型, Kubernetes 启发) ---
        governance_ctx = self._build_governance_context(
            review_result=review_result,
            routing_result=routing_result,
            approval_request=approval_request,
            session_id=session_id,
            user_id=user_id,
        )
        governance_decision = self._build_governance_decision(
            governance_ctx=governance_ctx,
            routing_result=routing_result,
            approval_request=approval_request,
            success=success,
        )

        payload["governance_context_id"] = governance_ctx.context_id
        payload["governance_decision_id"] = governance_decision.decision_id

        event = BridgeEvent(
            source="cc1",
            target="cc2",
            direction=BridgeDirection.CC1_TO_CC2,
            event_type=(
                "cc1.cc2.bridge.success"
                if success
                else "cc1.cc2.bridge.failure"
            ),
            trace_id=trace_id,
            session_id=session_id,
            payload=payload,
            metadata={
                "user_id": user_id,
                "governance_context": governance_ctx.model_dump(),
                "governance_decision": governance_decision.model_dump(),
            },
        )

        self._events.append(event)

        # 限制事件日志大小 (滑动窗口)
        if len(self._events) > self._MAX_EVENTS:
            self._events = self._events[-(self._MAX_EVENTS // 2):]

        logger.debug(
            "CC1→CC2 桥接事件: event_id=%s, success=%s, verdict=%s, layer=%s",
            event.event_id,
            success,
            payload.get("cc1", {}).get("verdict", "n/a"),
            payload.get("cc2_routing", {}).get("recommended_layer", "n/a"),
        )

        return event

    @staticmethod
    def _build_governance_context(
        review_result: ReviewResult | None,
        routing_result: RoutingResult | None,
        approval_request: ApprovalRequest | None,
        session_id: str,
        user_id: str,
    ) -> GovernanceContext:
        """构建治理上下文 (声明式治理状态快照).

        基于 Kubernetes Controller Pattern 的 observed state 模型,
        捕获 CC1 评审状态与 CC2 路由 / 审批状态, 用于治理闭环.

        Args:
            review_result: CC1 评审结果.
            routing_result: 路由决策结果.
            approval_request: 审批请求.
            session_id: 会话 ID.
            user_id: 用户 ID.

        Returns:
            治理上下文.
        """
        # CC1 状态
        cc1_verdict = ""
        cc1_score = 0.0
        cc1_layer_scores: dict[str, float] = {}
        cc1_issues: list[dict[str, Any]] = []

        if review_result is not None:
            cc1_verdict = review_result.verdict.value
            cc1_score = review_result.composite_score
            # 将枚举键 (ReviewLayerType) 转为字符串键
            cc1_layer_scores = {
                k.value if hasattr(k, "value") else str(k): v
                for k, v in review_result.layer_scores.items()
            }
            cc1_issues = list(review_result.issues)

        # CC2 状态
        cc2_layer = ""
        cc2_approval_id = ""
        cc2_approval_status = ""

        if routing_result is not None:
            cc2_layer = routing_result.recommended_layer.value

        if approval_request is not None:
            cc2_approval_id = approval_request.request_id

        return GovernanceContext(
            session_id=session_id,
            user_id=user_id,
            cc1_verdict=cc1_verdict,
            cc1_score=cc1_score,
            cc1_layer_scores=cc1_layer_scores,
            cc1_issues=cc1_issues,
            cc2_layer=cc2_layer,
            cc2_approval_id=cc2_approval_id,
            cc2_approval_status=cc2_approval_status,
        )

    @staticmethod
    def _build_governance_decision(
        governance_ctx: GovernanceContext,
        routing_result: RoutingResult | None,
        approval_request: ApprovalRequest | None,
        success: bool,
    ) -> GovernanceDecision:
        """构建治理决策 (治理闭环 ACT 阶段输出).

        Args:
            governance_ctx: 关联的治理上下文.
            routing_result: 路由决策结果.
            approval_request: 审批请求.
            success: 桥接是否成功.

        Returns:
            治理决策.
        """
        if not success or routing_result is None:
            return GovernanceDecision(
                context_id=governance_ctx.context_id,
                action="bridge_failed",
                rationale="CC1→CC2 桥接失败, 路由决策未生成",
                affected_modules=["cc1", "cc2"],
            )

        layer = routing_result.recommended_layer.value
        action = f"route_to_{layer}"
        if approval_request is not None:
            action += "_with_approval"

        parameters: dict[str, Any] = {
            "recommended_layer": layer,
            "routing_score": routing_result.score,
            "approval_mode": (
                routing_result.approval_mode.value
                if routing_result.approval_mode is not None
                else None
            ),
            "timeout_seconds": routing_result.timeout_seconds,
            "rule_id": routing_result.rule_id,
        }
        if approval_request is not None:
            parameters["approval_request_id"] = approval_request.request_id

        return GovernanceDecision(
            context_id=governance_ctx.context_id,
            action=action,
            parameters=parameters,
            rationale=routing_result.reasoning,
            affected_modules=["cc1", "cc2"],
        )

    # ========================================================
    # 统计与查询
    # ========================================================

    def get_statistics(self) -> dict[str, Any]:
        """返回桥接统计信息.

        Returns:
            统计字典, 包含::

                {
                    "total_bridges": int,
                    "successful_bridges": int,
                    "failed_bridges": int,
                    "success_rate": float,        # 百分比 (0-100)
                    "circuit_breaker_trips": int,
                    "approvals_created": int,
                    "by_verdict": dict,           # {pass, flag, block}
                    "by_layer": dict,             # {l1..l4}
                    "avg_confidence": float,
                    "avg_latency_ms": float,
                    "circuit_breaker_status": dict,
                }
        """
        total = self._stats["total_bridges"]
        successful = self._stats["successful_bridges"]

        avg_confidence = (
            self._stats["total_confidence_sum"] / total if total > 0 else 0.0
        )
        avg_latency = (
            self._stats["total_latency_ms_sum"] / total if total > 0 else 0.0
        )
        success_rate = (successful / total * 100.0) if total > 0 else 0.0

        return {
            "total_bridges": total,
            "successful_bridges": successful,
            "failed_bridges": self._stats["failed_bridges"],
            "success_rate": round(success_rate, 2),
            "circuit_breaker_trips": self._stats["circuit_breaker_trips"],
            "approvals_created": self._stats["approvals_created"],
            "by_verdict": dict(self._stats["by_verdict"]),
            "by_layer": dict(self._stats["by_layer"]),
            "avg_confidence": round(avg_confidence, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "circuit_breaker_status": self._circuit_breaker.get_status(),
        }

    def get_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """返回最近的桥接事件.

        Args:
            limit: 返回事件数量上限 (默认 50).

        Returns:
            事件字典列表, 按时间倒序排列 (最新在前).
        """
        if limit <= 0:
            return []
        events = self._events[-limit:]
        events = list(reversed(events))  # 最新在前
        return [e.model_dump() for e in events]

    def reset(self) -> None:
        """重置桥接器状态.

        清空事件日志与统计计数器, 并重置断路器到 CLOSED 状态.
        """
        self._events.clear()
        self._stats = {
            "total_bridges": 0,
            "successful_bridges": 0,
            "failed_bridges": 0,
            "circuit_breaker_trips": 0,
            "approvals_created": 0,
            "by_verdict": {"pass": 0, "flag": 0, "block": 0},
            "by_layer": {
                "l1_implicit": 0,
                "l2_prompt": 0,
                "l3_approval": 0,
                "l4_intervention": 0,
            },
            "total_confidence_sum": 0.0,
            "total_latency_ms_sum": 0.0,
        }
        self._circuit_breaker.reset()
        self._current_context = None
        logger.info("CC1→CC2 桥接器已重置")


__all__ = ["CC1CC2Bridge"]
