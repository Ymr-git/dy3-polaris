"""CC4 三横切集成 — 统一 API 网关.

实现单一入口的治理网关, 将请求路由到 CC1 / CC2 / CC3 三大横切模块,
编排完整的治理闭环: CC1→CC2→CC3→反馈飞轮.

核心能力:
- 单一治理入口: ``govern()`` 编排四阶段治理闭环
- CC1→CC2 桥接: 评审结果注入路由决策 (verdict→confidence)
- CC1→CC3 桥接: 评审结果自动标注到 KPA 校验维度
- CC2→CC3 桥接: 审批记录自动捕获到 KPA 决策维度
- 反馈飞轮: CC3 溯源完整性反馈 → CC1/CC2 调整
- 断路器保护: 所有跨模块调用经 CircuitBreaker 隔离
- 治理指标: GovernanceMetrics 多维度指标采集

治理闭环数据流::

    ReviewResult (CC1)
        │
        ├──→ CC1CC2Bridge ──→ RoutingResult + ApprovalRequest (CC2)
        │
        ├──→ CC1CC3Bridge ──→ KPAAnnotation (CC3, 校验维度)
        │
        │    (若 L3 审批)
        │    ApprovalRecord (CC2, make_decision)
        │        │
        │        └──→ CC2CC3Bridge ──→ KPAAnnotation (CC3, 决策维度)
        │
        └──→ FeedbackLoop ──→ 反馈信号 + 动作 (CC3→CC1/CC2)

融合世界先进方案:
- Service Mesh (Istio): 横切关注点统一编排 + 断路器隔离
- API Gateway (Kong/APISIX): 单一入口路由 + 协议转换
- Control Plane (Kubernetes): 声明式治理上下文 (GovernanceContext)
- Event-Driven Architecture: 桥接事件驱动松耦合集成
- Hystrix / Resilience4j: 断路器三态保护
- OpenTelemetry: trace_id 全链路传递
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .models import (
    BridgeDirection,
    BridgeEvent,
    GovernanceContext,
    GovernancePhase,
    GovernanceDecision,
    GovernanceMetrics,
)
from .exceptions import GatewayRoutingError, CircuitBreakerOpenError
from .circuit_breaker import CircuitBreaker
from .cc1_cc2_bridge import CC1CC2Bridge
from .cc1_cc3_bridge import CC1CC3Bridge
from .cc2_cc3_bridge import CC2CC3Bridge
from .feedback_loop import FeedbackLoop

logger = logging.getLogger(__name__)


class UnifiedGateway:
    """统一 API 网关 — CC1/CC2/CC3 治理闭环单一入口.

    将 CC1 (反幻觉评审) / CC2 (人机协作审批) / CC3 (溯源捕获) 三大
    横切模块编排为完整的治理闭环, 通过 :meth:`govern` 方法一键执行
    CC1→CC2→CC3→反馈飞轮的全链路治理.

    核心职责:
        1. 创建并持有三个桥接器 (CC1CC2Bridge / CC1CC3Bridge /
           CC2CC3Bridge) 与反馈飞轮 (FeedbackLoop)
        2. 确保桥接器间共享同一 CC3 集成器 / KPA 引擎实例
        3. 为所有跨模块调用配置断路器保护
        4. 编排治理闭环, 构建 :class:`GovernanceContext` 治理上下文
        5. 采集治理指标 (:class:`GovernanceMetrics`)

    使用示例::

        from dy3_polaris.l0.cc1.review_pipeline import (
            ReviewPipeline, VerificationRequest,
        )
        from dy3_polaris.l0.cc_integration import UnifiedGateway

        pipeline = ReviewPipeline()
        gateway = UnifiedGateway(cc1_pipeline=pipeline)

        review_result = pipeline.review(request)
        result = gateway.govern(
            review_result=review_result,
            operation_type="content_generation",
            user_id="student-001",
            session_id="sess-001",
            trace_id="trace-001",
        )

        if result["success"]:
            print("路由层级:", result["cc1_to_cc2"]["routing_result"]
                  ["recommended_layer"])

    Note:
        - 网关在 ``__init__`` 中创建桥接器, 所有依赖为 None 时自动
          创建默认实例, 开箱即用.
        - 治理过程非线程安全, 并发场景请为每个线程创建独立实例.
        - ``govern`` 对 L3 审批默认自动决策 (可通过 ``auto_complete_approval``
          参数关闭), 实际生产环境应接入人工审批回调.
    """

    #: 治理历史记录上限 (超出后保留最近一半).
    _MAX_HISTORY: int = 1000

    def __init__(
        self,
        cc1_pipeline: Any | None = None,
        cc2_routing_engine: Any | None = None,
        cc2_approval_manager: Any | None = None,
        cc3_kpa_engine: Any | None = None,
        cc3_cc_integration: Any | None = None,
        circuit_breakers: dict[str, CircuitBreaker] | None = None,
    ) -> None:
        """初始化统一网关.

        所有依赖项为 None 时自动创建默认实例, 开箱即用.
        确保 CC1CC3Bridge 与 CC2CC3Bridge 共享同一 CC3 集成器 /
        KPA 引擎实例, 避免 KPA 标注孤立.

        Args:
            cc1_pipeline: CC1 评审管线 (提供 ``review`` / ``get_statistics``).
            cc2_routing_engine: CC2 路由引擎.
            cc2_approval_manager: CC2 审批工作流管理器.
            cc3_kpa_engine: CC3 KPA 标注引擎.
            cc3_cc_integration: CC3 跨切面集成器.
            circuit_breakers: 断路器字典 {名称: CircuitBreaker},
                为 None 时按需自动创建.
        """
        self._cc1_pipeline = cc1_pipeline
        self._circuit_breakers: dict[str, CircuitBreaker] = (
            circuit_breakers or {}
        )

        # --------------------------------------------------------
        # 确保 CC3 桥接器共享同一 CCIntegration / KPAEngine
        # --------------------------------------------------------
        shared_cc_integration = cc3_cc_integration
        shared_kpa_engine = cc3_kpa_engine

        if shared_cc_integration is None and shared_kpa_engine is None:
            # 均未提供 → 创建共享实例
            try:
                from ..cc3.kpa_engine import KPAEngine
                from ..cc3.cc_integration import CCIntegration

                shared_kpa_engine = KPAEngine()
                shared_cc_integration = CCIntegration(
                    kpa_engine=shared_kpa_engine
                )
            except Exception as exc:
                logger.warning("CC3 共享实例创建失败: %s", exc)
        elif shared_cc_integration is None:
            # 仅提供 KPA → 创建共享 CCIntegration
            try:
                from ..cc3.cc_integration import CCIntegration

                shared_cc_integration = CCIntegration(
                    kpa_engine=shared_kpa_engine
                )
            except Exception as exc:
                logger.warning("CCIntegration 创建失败: %s", exc)
        elif shared_kpa_engine is None:
            # 仅提供 CCIntegration → 复用其内部 KPA
            shared_kpa_engine = getattr(
                shared_cc_integration, "_kpa", None
            )

        # --------------------------------------------------------
        # 创建桥接器 (断路器保护)
        # --------------------------------------------------------
        self._cc1_cc2_bridge = CC1CC2Bridge(
            routing_engine=cc2_routing_engine,
            approval_manager=cc2_approval_manager,
            circuit_breaker=self._get_breaker("cc2"),
        )
        self._cc1_cc3_bridge = CC1CC3Bridge(
            cc_integration=shared_cc_integration,
            kpa_engine=shared_kpa_engine,
            circuit_breaker=self._get_breaker("cc3"),
        )
        self._cc2_cc3_bridge = CC2CC3Bridge(
            cc_integration=shared_cc_integration,
            kpa_engine=shared_kpa_engine,
            circuit_breaker=self._get_breaker("cc2_to_cc3"),
        )

        # 从桥接器回引实际实例 (可能为桥接器内部创建的默认实例)
        self._cc2_routing_engine = self._cc1_cc2_bridge.routing_engine
        self._cc2_approval_manager = self._cc1_cc2_bridge.approval_manager
        self._cc3_kpa_engine = self._cc1_cc3_bridge.kpa_engine
        self._cc3_cc_integration = self._cc1_cc3_bridge.cc_integration

        # --------------------------------------------------------
        # 反馈飞轮 (容错初始化)
        # --------------------------------------------------------
        self._feedback_loop: FeedbackLoop | None = None
        self._init_feedback_loop(shared_cc_integration)

        # --------------------------------------------------------
        # 统计与历史
        # --------------------------------------------------------
        self._stats: dict[str, Any] = {
            "total_governances": 0,
            "successful_governances": 0,
            "failed_governances": 0,
            "cc1_to_cc2_runs": 0,
            "cc1_to_cc3_runs": 0,
            "cc2_to_cc3_runs": 0,
            "feedback_loop_runs": 0,
            "circuit_breaker_trips": 0,
            "total_latency_ms_sum": 0.0,
        }
        self._governance_history: list[dict[str, Any]] = []
        self._events: list[BridgeEvent] = []

    # ========================================================
    # 属性
    # ========================================================

    @property
    def cc1_cc2_bridge(self) -> CC1CC2Bridge:
        """CC1→CC2 桥接器."""
        return self._cc1_cc2_bridge

    @property
    def cc1_cc3_bridge(self) -> CC1CC3Bridge:
        """CC1→CC3 桥接器."""
        return self._cc1_cc3_bridge

    @property
    def cc2_cc3_bridge(self) -> CC2CC3Bridge:
        """CC2→CC3 桥接器."""
        return self._cc2_cc3_bridge

    @property
    def feedback_loop(self) -> FeedbackLoop | None:
        """反馈飞轮 (初始化失败时为 None)."""
        return self._feedback_loop

    @property
    def circuit_breakers(self) -> dict[str, CircuitBreaker]:
        """断路器字典."""
        return self._circuit_breakers

    # ========================================================
    # 核心治理方法
    # ========================================================

    def govern(
        self,
        review_result: Any | None = None,
        operation_type: str = "",
        risk_level: Any | None = None,
        user_id: str = "",
        session_id: str = "",
        trace_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """执行完整治理闭环 — CC1→CC2→CC3→反馈飞轮.

        编排流程:
            1. 若 ``review_result`` 提供:
               a. CC1→CC2 桥接: 评审结果注入路由决策, 获取
                  :class:`RoutingResult` 与 (可能) :class:`ApprovalRequest`.
               b. CC1→CC3 桥接: 评审结果标注到 KPA 校验维度, 获取
                  ``annotation_id``.
               c. 若路由到 ``L3_APPROVAL`` 且审批已创建: 自动完成审批
                  (``make_decision``) 并运行 CC2→CC3 桥接, 将审批记录
                  写入 KPA 决策维度.
               d. 反馈飞轮: 基于 ``annotation_id`` 执行 CC3→CC1/CC2 反馈.
            2. 返回包含所有桥接结果与反馈的综合字典.

        鲁棒性策略:
            - 任一桥接/反馈失败不中断后续步骤 (best-effort)
            - 断路器跳闸时记录并降级, 不抛出异常
            - 模块为 None 时跳过对应步骤
            - ``success`` 判定以 CC1→CC2 桥接成功为准

        Args:
            review_result: CC1 评审结果 (:class:`ReviewResult`).
                为 None 时返回失败结果.
            operation_type: 操作类型标识.
            risk_level: 操作风险等级 (CC2 ``RiskLevel``), 为 None 时
                由桥接器使用默认值 (MEDIUM).
            user_id: 用户 ID.
            session_id: 会话 ID.
            trace_id: OpenTelemetry 全链路 trace ID.
            **kwargs: 附加参数, 可包含:
                - ``reversibility`` / ``user_role`` / ``cognitive_load`` /
                  ``trust_score``: CC1→CC2 桥接器路由上下文参数.
                - ``target_id`` / ``target_type`` / ``annotation_id``:
                  CC1→CC3 桥接器标注参数.
                - ``approval_record``: 已决审批记录 (跳过自动决策).
                - ``approval_decision`` / ``decided_by`` /
                  ``approval_comment``: 自动审批决策参数.
                - ``auto_complete_approval``: 是否自动完成 L3 审批
                  (默认 True).

        Returns:
            综合治理结果字典::

                {
                    "success": bool,
                    "trace_id": str,
                    "session_id": str,
                    "user_id": str,
                    "operation_type": str,
                    "cc1_to_cc2": dict | None,   # CC1→CC2 桥接结果
                    "cc1_to_cc3": dict | None,   # CC1→CC3 桥接结果
                    "cc2_to_cc3": dict | None,   # CC2→CC3 桥接结果
                    "feedback": dict | None,     # 反馈飞轮结果
                    "governance_context_id": str,
                    "annotation_id": str,
                    "recommended_layer": str,
                    "latency_ms": float,
                    "error": str,
                }
        """
        start_time = time.time()
        self._stats["total_governances"] += 1

        # 初始化治理上下文
        governance_ctx = GovernanceContext(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            operation_type=operation_type,
            phase=GovernancePhase.RECONCILE,
        )

        result: dict[str, Any] = {
            "success": False,
            "trace_id": trace_id,
            "session_id": session_id,
            "user_id": user_id,
            "operation_type": operation_type,
            "cc1_to_cc2": None,
            "cc1_to_cc3": None,
            "cc2_to_cc3": None,
            "feedback": None,
            "governance_context_id": governance_ctx.context_id,
            "annotation_id": "",
            "recommended_layer": "",
            "latency_ms": 0.0,
            "error": "",
        }

        # 无评审结果 → 直接返回
        if review_result is None:
            result["error"] = "review_result 未提供, 无法执行治理"
            self._finalize(result, start_time, governance_ctx)
            return result

        # 注入 CC1 状态到治理上下文
        self._enrich_context_from_review(governance_ctx, review_result)

        try:
            # ------------------------------------------------
            # Step 1a: CC1→CC2 桥接 (路由决策)
            # ------------------------------------------------
            governance_ctx.phase = GovernancePhase.EVALUATE
            cc1_to_cc2 = self._run_cc1_to_cc2(
                review_result,
                operation_type=operation_type,
                risk_level=risk_level,
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
                **kwargs,
            )
            result["cc1_to_cc2"] = cc1_to_cc2

            routing_result_dict = cc1_to_cc2.get("routing_result")
            approval_request_dict = cc1_to_cc2.get("approval_request")
            recommended_layer = ""
            if isinstance(routing_result_dict, dict):
                recommended_layer = routing_result_dict.get(
                    "recommended_layer", ""
                )
            result["recommended_layer"] = recommended_layer
            governance_ctx.cc2_layer = recommended_layer

            if isinstance(approval_request_dict, dict):
                governance_ctx.cc2_approval_id = (
                    approval_request_dict.get("request_id", "")
                )

            # ------------------------------------------------
            # Step 1b: CC1→CC3 桥接 (标注评审结果)
            # ------------------------------------------------
            cc1_to_cc3 = self._run_cc1_to_cc3(
                review_result,
                session_id=session_id,
                trace_id=trace_id,
                **kwargs,
            )
            result["cc1_to_cc3"] = cc1_to_cc3

            annotation_id = ""
            if cc1_to_cc3.get("success"):
                annotation_id = cc1_to_cc3.get("annotation_id", "")
            result["annotation_id"] = annotation_id
            governance_ctx.cc3_annotation_id = annotation_id

            # ------------------------------------------------
            # Step 1c: 若 L3 审批, 完成审批并运行 CC2→CC3
            # ------------------------------------------------
            governance_ctx.phase = GovernancePhase.ACT
            auto_complete = kwargs.get("auto_complete_approval", True)

            if (
                cc1_to_cc2.get("success")
                and recommended_layer == "l3_approval"
                and isinstance(approval_request_dict, dict)
                and approval_request_dict.get("request_id")
            ):
                approval_record = None
                if kwargs.get("approval_record") is not None:
                    # 调用方传入已决审批记录
                    approval_record = kwargs["approval_record"]
                elif auto_complete:
                    # 自动完成审批 (默认批准)
                    approval_record = self._complete_approval(
                        approval_request_dict, **kwargs
                    )

                if approval_record is not None:
                    routing_result_obj = self._reconstruct_routing_result(
                        routing_result_dict
                    )
                    cc2_to_cc3 = self._run_cc2_to_cc3(
                        approval_record,
                        routing_result_obj,
                        annotation_id=annotation_id,
                        session_id=session_id,
                        trace_id=trace_id,
                        **kwargs,
                    )
                    result["cc2_to_cc3"] = cc2_to_cc3
                    governance_ctx.cc2_approval_status = (
                        getattr(
                            approval_record, "status",
                            None,
                        )
                    )
                    if governance_ctx.cc2_approval_status is not None:
                        governance_ctx.cc2_approval_status = (
                            governance_ctx.cc2_approval_status.value
                            if hasattr(
                                governance_ctx.cc2_approval_status, "value"
                            )
                            else str(governance_ctx.cc2_approval_status)
                        )

            # ------------------------------------------------
            # Step 1d: 反馈飞轮
            # ------------------------------------------------
            governance_ctx.phase = GovernancePhase.VERIFY
            if annotation_id:
                feedback = self._run_feedback_loop(
                    annotation_id,
                    session_id=session_id,
                    trace_id=trace_id,
                    **kwargs,
                )
                result["feedback"] = feedback

            # 判定整体成功 (以 CC1→CC2 桥接成功为准)
            result["success"] = bool(cc1_to_cc2.get("success", False))

        except CircuitBreakerOpenError as exc:
            result["error"] = str(exc)
            self._stats["circuit_breaker_trips"] += 1
            logger.warning(
                "治理过程中断路器跳闸: trace_id=%s, %s",
                trace_id,
                exc,
            )
        except Exception as exc:
            result["error"] = f"治理过程异常: {exc}"
            logger.exception(
                "治理过程异常: trace_id=%s", trace_id
            )

        self._finalize(result, start_time, governance_ctx)
        return result

    # ========================================================
    # 桥接执行方法
    # ========================================================

    def _run_cc1_to_cc2(
        self, review_result: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """运行 CC1→CC2 桥接 — 评审结果注入路由决策.

        经 :class:`CC1CC2Bridge` 将 CC1 评审结果转换为 CC2 路由上下文,
        执行路由决策并在路由到 L3 时自动创建审批请求.

        Args:
            review_result: CC1 评审结果.
            **kwargs: 桥接参数 (operation_type / risk_level / user_id /
                session_id / trace_id / reversibility / user_role /
                cognitive_load / trust_score + 附加元数据).

        Returns:
            桥接结果字典 (含 routing_result / approval_request).
        """
        self._stats["cc1_to_cc2_runs"] += 1

        if review_result is None:
            return {
                "success": False,
                "error": "review_result 未提供",
                "routing_result": None,
                "approval_request": None,
            }

        # 提取桥接器已知参数
        operation_type = kwargs.pop("operation_type", "")
        risk_level = kwargs.pop("risk_level", None)
        user_id = kwargs.pop("user_id", "")
        session_id = kwargs.pop("session_id", "")
        trace_id = kwargs.pop("trace_id", "")

        # 移除非本桥接器参数 (供其他步骤使用)
        for key in (
            "annotation_id",
            "target_id",
            "target_type",
            "approval_decision",
            "decided_by",
            "approval_comment",
            "approval_record",
            "auto_complete_approval",
        ):
            kwargs.pop(key, None)

        # 构建桥接器调用参数
        bridge_kwargs: dict[str, Any] = {
            "operation_type": operation_type,
            "user_id": user_id,
            "session_id": session_id,
            "trace_id": trace_id,
        }
        if risk_level is not None:
            bridge_kwargs["risk_level"] = risk_level

        # 已知桥接器可选参数
        for key in (
            "reversibility",
            "user_role",
            "cognitive_load",
            "trust_score",
        ):
            if key in kwargs:
                bridge_kwargs[key] = kwargs.pop(key)

        # 剩余 kwargs 作为 extra_metadata 传递
        try:
            result = self._cc1_cc2_bridge.bridge(
                review_result=review_result,
                **bridge_kwargs,
                **kwargs,
            )
            return result
        except CircuitBreakerOpenError as exc:
            self._stats["circuit_breaker_trips"] += 1
            logger.warning("CC1→CC2 桥接断路器跳闸: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "routing_result": None,
                "approval_request": None,
            }
        except Exception as exc:
            logger.exception("CC1→CC2 桥接失败")
            return {
                "success": False,
                "error": str(exc),
                "routing_result": None,
                "approval_request": None,
            }

    def _run_cc1_to_cc3(
        self, review_result: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """运行 CC1→CC3 桥接 — 评审结果标注到 KPA 校验维度.

        经 :class:`CC1CC3Bridge` 将 CC1 评审结果转换为 CC3 KPA 标注的
        校验维度数据, 写入 KPA / Ledger / 溯源链.

        Args:
            review_result: CC1 评审结果.
            **kwargs: 桥接参数 (session_id / trace_id / target_id /
                target_type / annotation_id).

        Returns:
            桥接结果字典 (含 annotation_id / completeness).
        """
        self._stats["cc1_to_cc3_runs"] += 1

        if review_result is None:
            return {
                "success": False,
                "error": "review_result 未提供",
                "annotation_id": "",
            }

        session_id = kwargs.get("session_id", "")
        trace_id = kwargs.get("trace_id", "")
        target_id = kwargs.get("target_id", "")
        target_type = kwargs.get("target_type")
        annotation_id = kwargs.get("annotation_id")

        bridge_kwargs: dict[str, Any] = {
            "trace_id": trace_id,
            "session_id": session_id,
        }
        if target_id:
            bridge_kwargs["target_id"] = target_id
        if target_type is not None:
            bridge_kwargs["target_type"] = target_type
        if annotation_id is not None:
            bridge_kwargs["annotation_id"] = annotation_id

        try:
            result = self._cc1_cc3_bridge.bridge(
                review_result=review_result, **bridge_kwargs
            )
            return result
        except CircuitBreakerOpenError as exc:
            self._stats["circuit_breaker_trips"] += 1
            logger.warning("CC1→CC3 桥接断路器跳闸: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "annotation_id": "",
            }
        except Exception as exc:
            logger.exception("CC1→CC3 桥接失败")
            return {
                "success": False,
                "error": str(exc),
                "annotation_id": "",
            }

    def _run_cc2_to_cc3(
        self,
        approval_record: Any,
        routing_result: Any | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """运行 CC2→CC3 桥接 — 审批记录写入 KPA 决策维度.

        经 :class:`CC2CC3Bridge` 将 CC2 审批记录桥接到 CC3 溯源层,
        写入 KPA 标注的决策维度并追加溯源链节点.

        Args:
            approval_record: CC2 审批记录 (:class:`ApprovalRecord`).
            routing_result: CC2 路由决策结果 (:class:`RoutingResult`),
                可为 None 或字典 (将尝试重建).
            **kwargs: 桥接参数 (annotation_id / session_id / trace_id /
                target_id).

        Returns:
            桥接结果字典 (含 approval_level / completeness).
        """
        self._stats["cc2_to_cc3_runs"] += 1

        if approval_record is None:
            return {"success": False, "error": "approval_record 未提供"}

        annotation_id = kwargs.get("annotation_id")
        session_id = kwargs.get("session_id", "")
        trace_id = kwargs.get("trace_id", "")
        target_id = kwargs.get("target_id", "")

        # 若 routing_result 为字典, 尝试重建为 RoutingResult 对象
        if isinstance(routing_result, dict):
            routing_result = self._reconstruct_routing_result(
                routing_result
            )

        try:
            result = self._cc2_cc3_bridge.bridge(
                approval_record=approval_record,
                routing_result=routing_result,
                annotation_id=annotation_id,
                target_id=target_id,
                trace_id=trace_id,
                session_id=session_id,
            )
            return result
        except CircuitBreakerOpenError as exc:
            self._stats["circuit_breaker_trips"] += 1
            logger.warning("CC2→CC3 桥接断路器跳闸: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "annotation_id": annotation_id or "",
            }
        except Exception as exc:
            logger.exception("CC2→CC3 桥接失败")
            return {
                "success": False,
                "error": str(exc),
                "annotation_id": annotation_id or "",
            }

    def _run_feedback_loop(
        self, annotation_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        """运行反馈飞轮 — CC3 溯源完整性反馈到 CC1/CC2.

        基于 KPA 标注 ID 评估溯源完整性, 生成反馈信号与动作
        (评审阈值调整 / 协同升级建议).

        Args:
            annotation_id: KPA 标注 ID.
            **kwargs: 参数 (session_id / trace_id).

        Returns:
            反馈飞轮结果字典 (含 signals / actions).
        """
        self._stats["feedback_loop_runs"] += 1

        if not annotation_id:
            return {
                "success": False,
                "error": "annotation_id 未提供",
            }

        if self._feedback_loop is None:
            return {
                "success": False,
                "error": "反馈飞轮未配置",
                "annotation_id": annotation_id,
            }

        session_id = kwargs.get("session_id", "")
        trace_id = kwargs.get("trace_id", "")

        try:
            result = self._feedback_loop.evaluate(
                annotation_id=annotation_id,
                trace_id=trace_id,
                session_id=session_id,
            )
            return result
        except AttributeError as exc:
            # FeedbackLoop 接口不匹配
            logger.warning(
                "反馈飞轮接口调用失败 (annotation=%s): %s",
                annotation_id,
                exc,
            )
            return {
                "success": False,
                "error": f"反馈飞轮接口不兼容: {exc}",
                "annotation_id": annotation_id,
            }
        except CircuitBreakerOpenError as exc:
            self._stats["circuit_breaker_trips"] += 1
            return {
                "success": False,
                "error": str(exc),
                "annotation_id": annotation_id,
            }
        except Exception as exc:
            logger.exception(
                "反馈飞轮执行失败: annotation=%s", annotation_id
            )
            return {
                "success": False,
                "error": str(exc),
                "annotation_id": annotation_id,
            }

    # ========================================================
    # 指标与统计
    # ========================================================

    def get_governance_metrics(self) -> GovernanceMetrics:
        """采集治理指标 — 汇总所有桥接器与模块的统计.

        从三个桥接器、CC1/CC2/CC3 模块采集指标, 构建
        :class:`GovernanceMetrics` 多维度指标快照.

        Returns:
            治理指标汇总 (Prometheus 风格多维度指标).
        """
        total_bridges = 0
        successful_bridges = 0
        cb_trips = 0

        # 汇总桥接器统计
        for bridge in (
            self._cc1_cc2_bridge,
            self._cc1_cc3_bridge,
            self._cc2_cc3_bridge,
        ):
            try:
                stats = bridge.get_statistics()
                total_bridges += stats.get(
                    "total_bridges", stats.get("total", 0)
                )
                successful_bridges += stats.get(
                    "successful_bridges", stats.get("successful", 0)
                )
                cb_trips += stats.get("circuit_breaker_trips", 0)
            except Exception:
                logger.debug("桥接器统计采集失败", exc_info=True)

        success_rate = (
            successful_bridges / total_bridges
            if total_bridges > 0
            else 0.0
        )

        # 平均治理延迟
        total_gov = self._stats["total_governances"]
        avg_latency = (
            self._stats["total_latency_ms_sum"] / total_gov
            if total_gov > 0
            else 0.0
        )

        # CC1 通过率
        cc1_pass_rate = self._collect_cc1_pass_rate()

        # CC2 自动批准率
        cc2_auto_rate = self._collect_cc2_auto_approval_rate()

        # CC3 平均完整度
        cc3_completeness = self._collect_cc3_completeness()

        # 升级次数 (L3 + L4 路由)
        escalation_count = self._collect_escalation_count()

        return GovernanceMetrics(
            total_bridges=total_bridges,
            bridge_success_rate=round(success_rate, 4),
            avg_governance_latency_ms=round(avg_latency, 2),
            feedback_loops_active=self._stats["feedback_loop_runs"],
            circuit_breaker_trips=(
                cb_trips + self._stats["circuit_breaker_trips"]
            ),
            cc1_pass_rate=round(cc1_pass_rate, 2),
            cc2_auto_approval_rate=round(cc2_auto_rate, 2),
            cc3_avg_completeness=round(cc3_completeness, 4),
            escalation_count=escalation_count,
        )

    def get_statistics(self) -> dict[str, Any]:
        """获取网关统计信息.

        Returns:
            统计字典, 包含治理计数、桥接器统计与断路器状态.
        """
        total = self._stats["total_governances"]
        successful = self._stats["successful_governances"]
        success_rate = (
            successful / total * 100.0 if total > 0 else 0.0
        )
        avg_latency = (
            self._stats["total_latency_ms_sum"] / total
            if total > 0
            else 0.0
        )

        # 桥接器统计 (容错)
        bridge_stats: dict[str, Any] = {}
        for name, bridge in (
            ("cc1_cc2", self._cc1_cc2_bridge),
            ("cc1_cc3", self._cc1_cc3_bridge),
            ("cc2_cc3", self._cc2_cc3_bridge),
        ):
            try:
                bridge_stats[name] = bridge.get_statistics()
            except Exception as exc:
                bridge_stats[name] = {"error": str(exc)}

        # 断路器状态
        cb_status: dict[str, Any] = {}
        for name, breaker in self._circuit_breakers.items():
            try:
                cb_status[name] = breaker.get_status()
            except Exception as exc:
                cb_status[name] = {"error": str(exc)}

        return {
            "total_governances": total,
            "successful_governances": successful,
            "failed_governances": self._stats["failed_governances"],
            "success_rate": round(success_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "cc1_to_cc2_runs": self._stats["cc1_to_cc2_runs"],
            "cc1_to_cc3_runs": self._stats["cc1_to_cc3_runs"],
            "cc2_to_cc3_runs": self._stats["cc2_to_cc3_runs"],
            "feedback_loop_runs": self._stats["feedback_loop_runs"],
            "circuit_breaker_trips": self._stats[
                "circuit_breaker_trips"
            ],
            "feedback_loop_configured": self._feedback_loop is not None,
            "bridge_statistics": bridge_stats,
            "circuit_breakers": cb_status,
        }

    def get_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """返回最近的治理事件.

        Args:
            limit: 返回事件数量上限 (默认 50).

        Returns:
            事件字典列表, 按时间倒序排列 (最新在前).
        """
        if limit <= 0:
            return []
        events = self._events[-limit:]
        return [e.model_dump() for e in reversed(events)]

    def reset(self) -> None:
        """重置网关状态.

        清空治理历史与事件日志, 重置统计计数器与断路器.
        不重置桥接器内部状态 (各桥接器有独立 ``reset`` 方法).
        """
        self._stats = {
            "total_governances": 0,
            "successful_governances": 0,
            "failed_governances": 0,
            "cc1_to_cc2_runs": 0,
            "cc1_to_cc3_runs": 0,
            "cc2_to_cc3_runs": 0,
            "feedback_loop_runs": 0,
            "circuit_breaker_trips": 0,
            "total_latency_ms_sum": 0.0,
        }
        self._governance_history.clear()
        self._events.clear()
        for breaker in self._circuit_breakers.values():
            breaker.reset()
        logger.info("统一网关已重置")

    # ========================================================
    # 内部辅助方法
    # ========================================================

    def _get_breaker(self, name: str) -> CircuitBreaker:
        """获取或创建指定名称的断路器.

        若 ``circuit_breakers`` 字典中不存在该名称, 则创建默认
        :class:`CircuitBreaker` 并缓存, 确保同一名称始终返回同一实例.

        Args:
            name: 断路器名称 (如 ``"cc2"`` / ``"cc3"``).

        Returns:
            断路器实例.
        """
        breaker = self._circuit_breakers.get(name)
        if breaker is None:
            breaker = CircuitBreaker(module=name)
            self._circuit_breakers[name] = breaker
        return breaker

    def _init_feedback_loop(
        self, cc_integration: Any | None
    ) -> None:
        """初始化反馈飞轮 (容错).

        优先传入共享 ``cc_integration`` 构造, 确保飞轮的
        ``check_provenance_for_cc1`` 能访问桥接器创建的标注.
        若 ``cc_integration`` 为 None 或构造失败, 回退到无参构造
        (飞轮内部自建 CCIntegration). 均失败时置为 None,
        网关在反馈步骤降级跳过.

        Args:
            cc_integration: CC3 跨切面集成器 (共享实例).
        """
        if cc_integration is not None:
            try:
                self._feedback_loop = FeedbackLoop(
                    cc_integration=cc_integration,
                    cc1_statistics_provider=self._cc1_pipeline,
                )
                return
            except Exception as exc:
                logger.warning(
                    "反馈飞轮 (cc_integration) 初始化失败, 回退到默认: %s",
                    exc,
                )
        try:
            self._feedback_loop = FeedbackLoop(
                cc1_statistics_provider=self._cc1_pipeline,
            )
        except Exception as exc:
            logger.warning(
                "反馈飞轮初始化失败, 治理过程将跳过反馈步骤: %s",
                exc,
            )
            self._feedback_loop = None

    def _complete_approval(
        self, approval_request_dict: dict[str, Any], **kwargs: Any
    ) -> Any | None:
        """自动完成 L3 审批请求, 返回审批记录.

        通过 :meth:`ApprovalWorkflowManager.make_decision` 对审批请求
        做出决策 (默认批准), 经断路器保护.

        Args:
            approval_request_dict: 审批请求字典 (含 ``request_id``).
            **kwargs: 可包含 ``approval_decision`` / ``decided_by`` /
                ``approval_comment``.

        Returns:
            审批记录; 失败时返回 None.
        """
        if self._cc2_approval_manager is None:
            logger.warning("审批管理器未配置, 无法自动完成审批")
            return None

        request_id = approval_request_dict.get("request_id", "")
        if not request_id:
            logger.warning("审批请求缺少 request_id, 无法自动完成")
            return None

        decided_by = kwargs.get("decided_by", "unified_gateway")
        comment = kwargs.get(
            "approval_comment", "网关自动决策"
        )

        md_kwargs: dict[str, Any] = {
            "request_id": request_id,
            "decided_by": decided_by,
            "comment": comment,
        }
        decision = kwargs.get("approval_decision")
        if decision is not None:
            md_kwargs["decision"] = decision

        try:
            breaker = self._get_breaker("cc2_approval")
            record = breaker.call(
                self._cc2_approval_manager.make_decision,
                **md_kwargs,
            )
            logger.info(
                "网关自动完成审批: request_id=%s, decided_by=%s",
                request_id,
                decided_by,
            )
            return record
        except CircuitBreakerOpenError as exc:
            self._stats["circuit_breaker_trips"] += 1
            logger.warning(
                "自动审批断路器跳闸: request_id=%s, %s",
                request_id,
                exc,
            )
            return None
        except Exception as exc:
            logger.warning(
                "自动完成审批失败: request_id=%s, %s",
                request_id,
                exc,
            )
            return None

    def _reconstruct_routing_result(
        self, routing_dict: dict[str, Any] | None
    ) -> Any | None:
        """从字典重建 RoutingResult 对象.

        CC1CC2Bridge 返回的路由结果为 ``model_dump()`` 字典,
        CC2CC3Bridge 需要 RoutingResult 对象. 本方法通过 pydantic
        ``model_validate`` 重建.

        Args:
            routing_dict: 路由结果字典.

        Returns:
            RoutingResult 对象; 重建失败时返回 None.
        """
        if not routing_dict:
            return None
        try:
            from ..cc2.routing_engine import RoutingResult

            return RoutingResult.model_validate(routing_dict)
        except Exception as exc:
            logger.debug("无法重建 RoutingResult: %s", exc)
            return None

    def _enrich_context_from_review(
        self, ctx: GovernanceContext, review_result: Any
    ) -> None:
        """从评审结果中提取 CC1 状态, 注入治理上下文.

        Args:
            ctx: 治理上下文.
            review_result: CC1 评审结果.
        """
        try:
            verdict = getattr(review_result, "verdict", None)
            if verdict is not None:
                ctx.cc1_verdict = (
                    verdict.value if hasattr(verdict, "value") else str(verdict)
                )
            ctx.cc1_score = float(
                getattr(review_result, "composite_score", 0.0)
            )
            layer_scores = getattr(review_result, "layer_scores", None)
            if layer_scores:
                ctx.cc1_layer_scores = {
                    (k.value if hasattr(k, "value") else str(k)): float(v)
                    for k, v in layer_scores.items()
                }
        except Exception:
            logger.debug("注入 CC1 状态到治理上下文失败", exc_info=True)

    def _finalize(
        self,
        result: dict[str, Any],
        start_time: float,
        governance_ctx: GovernanceContext,
    ) -> None:
        """收尾治理流程 — 更新统计、记录历史与事件.

        Args:
            result: 治理结果字典 (就地更新 latency_ms).
            start_time: 治理开始时间.
            governance_ctx: 治理上下文.
        """
        latency_ms = (time.time() - start_time) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)
        self._stats["total_latency_ms_sum"] += latency_ms

        if result["success"]:
            self._stats["successful_governances"] += 1
        else:
            self._stats["failed_governances"] += 1

        # 记录治理历史
        self._governance_history.append(
            {
                "trace_id": result.get("trace_id", ""),
                "session_id": result.get("session_id", ""),
                "success": result.get("success", False),
                "latency_ms": round(latency_ms, 2),
                "recommended_layer": result.get("recommended_layer", ""),
                "annotation_id": result.get("annotation_id", ""),
                "timestamp": time.time(),
            }
        )
        if len(self._governance_history) > self._MAX_HISTORY:
            keep = self._MAX_HISTORY // 2
            self._governance_history = self._governance_history[-keep:]

        # 记录治理事件 (CloudEvents 格式)
        self._record_governance_event(result, governance_ctx)

    def _record_governance_event(
        self,
        result: dict[str, Any],
        governance_ctx: GovernanceContext,
    ) -> None:
        """记录治理审计事件 (CloudEvents 格式).

        Args:
            result: 治理结果字典.
            governance_ctx: 治理上下文.
        """
        success = result.get("success", False)
        decision = GovernanceDecision(
            context_id=governance_ctx.context_id,
            phase=governance_ctx.phase,
            action=(
                "governance_completed"
                if success
                else "governance_failed"
            ),
            rationale=result.get("error", "") or "治理闭环执行完成",
            affected_modules=["cc1", "cc2", "cc3"],
        )

        event = BridgeEvent(
            source="gateway",
            target="cc1_cc2_cc3",
            direction=BridgeDirection.CC1_TO_CC2,
            event_type=(
                "gateway.governance.success"
                if success
                else "gateway.governance.failure"
            ),
            trace_id=result.get("trace_id", ""),
            session_id=result.get("session_id", ""),
            payload={
                "success": success,
                "operation_type": result.get("operation_type", ""),
                "recommended_layer": result.get("recommended_layer", ""),
                "annotation_id": result.get("annotation_id", ""),
                "latency_ms": result.get("latency_ms", 0.0),
                "error": result.get("error", ""),
                "cc1_to_cc2_success": (
                    result.get("cc1_to_cc2", {}).get("success", False)
                    if result.get("cc1_to_cc2")
                    else False
                ),
                "cc1_to_cc3_success": (
                    result.get("cc1_to_cc3", {}).get("success", False)
                    if result.get("cc1_to_cc3")
                    else False
                ),
                "cc2_to_cc3_success": (
                    result.get("cc2_to_cc3", {}).get("success", False)
                    if result.get("cc2_to_cc3")
                    else False
                ),
                "feedback_success": (
                    result.get("feedback", {}).get("success", False)
                    if result.get("feedback")
                    else False
                ),
            },
            metadata={
                "governance_context": governance_ctx.model_dump(),
                "governance_decision": decision.model_dump(),
                "user_id": result.get("user_id", ""),
            },
        )
        self._events.append(event)
        if len(self._events) > self._MAX_HISTORY:
            keep = self._MAX_HISTORY // 2
            self._events = self._events[-keep:]

    def _collect_cc1_pass_rate(self) -> float:
        """采集 CC1 评审通过率 (百分比)."""
        if self._cc1_pipeline is None:
            return 0.0
        try:
            stats = self._cc1_pipeline.get_statistics()
            return float(stats.get("pass_rate", 0.0))
        except Exception:
            return 0.0

    def _collect_cc2_auto_approval_rate(self) -> float:
        """采集 CC2 自动批准率 (百分比)."""
        if self._cc2_approval_manager is None:
            return 0.0
        try:
            stats = self._cc2_approval_manager.get_statistics()
            total = stats.get("total", 0)
            auto = stats.get("auto_approved_count", 0)
            return (auto / total * 100.0) if total > 0 else 0.0
        except Exception:
            return 0.0

    def _collect_cc3_completeness(self) -> float:
        """采集 CC3 KPA 平均完整度 (0-1)."""
        if self._cc3_kpa_engine is None:
            return 0.0
        try:
            stats = self._cc3_kpa_engine.statistics()
            return float(stats.get("avg_completeness", 0.0))
        except Exception:
            return 0.0

    def _collect_escalation_count(self) -> int:
        """采集升级次数 (L3 审批 + L4 干预路由数)."""
        if self._cc2_routing_engine is None:
            return 0
        try:
            stats = self._cc2_routing_engine.get_statistics()
            by_layer = stats.get("by_layer", {})
            return int(
                by_layer.get("l3_approval", 0)
                + by_layer.get("l4_intervention", 0)
            )
        except Exception:
            return 0


__all__ = ["UnifiedGateway"]
