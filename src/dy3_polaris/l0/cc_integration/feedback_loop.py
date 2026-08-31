"""CC4 三横切集成 — CC3→CC1/CC2 反馈飞轮.

实现溯源完整性反馈驱动的评审阈值调整与协同升级建议飞轮,
打通 ``CC3 (溯源捕获) → CC1 (反幻觉) / CC2 (人机协作)`` 的反向数据流,
形成 "溯源完整性 → 评审阈值收紧 / 协同层级升级" 的自适应闭环.

飞轮流程::

    CC3 check_provenance_for_cc1 (溯源完整性报告)
        │
        ├─ completeness_score < 0.5 ──→ PROVENANCE_COMPLETENESS 信号
        ├─ not chain_verified       ──→ CHAIN_INTEGRITY          信号
        ├─ needs_escalation         ──→ ESCALATION_TRIGGER       信号
        ├─ cc1_pass_rate < 0.8      ──→ REVIEW_QUALITY          信号
        └─ (always)                 ──→ THRESHOLD_ADJUSTMENT     信号
        ▼
    FeedbackLoop.generate_signals()
        ▼
    FeedbackLoop.create_actions()  (信号 → 动作)
        │  adjust_threshold → CC1 (收紧 / 放宽 pass/flag 阈值)
        │  suggest_escalation → CC2 (升级协同层级)
        ▼
    FeedbackLoop.execute_action()  (断路器保护, 事件审计)
        ▼
    BridgeEvent (CloudEvents 审计事件)

阈值调整策略 (completeness → 阈值增量)::

    ┌────────────────────┬──────────────────────┬──────────────────────┬──────────┐
    │ completeness       │ pass_threshold_delta │ flag_threshold_delta │ 方向     │
    ├────────────────────┼──────────────────────┼──────────────────────┼──────────┤
    │ < 0.3              │ +5.0                 │ +3.0                 │ 收紧     │
    │ < 0.5              │ +3.0                 │ +2.0                 │ 收紧     │
    │ >= 0.8             │ -2.0                 │  0.0                 │ 放宽     │
    │ 其他 (0.5 ~ 0.8)   │  0.0                 │  0.0                 │ 维持     │
    └────────────────────┴──────────────────────┴──────────────────────┴──────────┘

    delta > 0 表示提高阈值 (更严格, 完整度低时收紧评审);
    delta < 0 表示降低阈值 (更宽松, 完整度高时放宽评审).

融合世界先进方案:
- Control Plane (Kubernetes): Reconcile-Evaluate-Act-Verify 调谐循环
- Control Theory: 反馈飞轮 (feedback flywheel) 驱动系统向期望态收敛
- Event-Driven Architecture: CloudEvents 标准化反馈事件
- Hystrix / Resilience4j: 断路器保护 CC3 调用, 防止级联故障
- OpenTelemetry: trace_id / session_id 全链路传递
- Guardrails AI: 可配置阈值 + 动作映射
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from .models import (
    AlertSeverity,
    BridgeDirection,
    BridgeEvent,
    FeedbackAction,
    FeedbackSignal,
    FeedbackSignalType,
    GovernancePhase,
)
from .exceptions import CircuitBreakerOpenError, FeedbackLoopError
from .circuit_breaker import CircuitBreaker
from ..cc3.cc_integration import CCIntegration

logger = logging.getLogger(__name__)


#: 信号类型 → (动作类型, 目标模块) 映射.
#:
#: 每个触发的信号映射到一个反馈动作, 驱动 CC1 / CC2 的动态调整.
_SIGNAL_ACTION_MAP: dict[FeedbackSignalType, tuple[str, str]] = {
    FeedbackSignalType.PROVENANCE_COMPLETENESS: ("adjust_threshold", "cc1"),
    FeedbackSignalType.CHAIN_INTEGRITY: ("adjust_threshold", "cc1"),
    FeedbackSignalType.THRESHOLD_ADJUSTMENT: ("adjust_threshold", "cc1"),
    FeedbackSignalType.REVIEW_QUALITY: ("adjust_threshold", "cc1"),
    FeedbackSignalType.ESCALATION_TRIGGER: ("suggest_escalation", "cc2"),
}

#: 完整度 → 阈值增量策略表.
#:
#: ``(pass_delta, flag_delta, direction, reason)`` —
#: 完整度越低, 评审阈值越严; 完整度越高, 评审阈值越宽.
_THRESHOLD_STRATEGY: list[tuple[float, float, float, str, str]] = [
    # (completeness_upper_bound, pass_delta, flag_delta, direction, reason)
    (0.3, 5.0, 3.0, "stricter", "完整度严重不足, 大幅收紧评审阈值"),
    (0.5, 3.0, 2.0, "stricter", "完整度较低, 适度收紧评审阈值"),
    (0.8, 0.0, 0.0, "none", "完整度中等, 维持当前评审阈值"),
]


class FeedbackLoop:
    """CC3→CC1/CC2 反馈飞轮 — 溯源完整性驱动的评审阈值调整与协同升级.

    基于 CC3 溯源完整性报告, 生成反馈信号并转化为对 CC1 (评审阈值)
    和 CC2 (协同层级) 的动态调整建议, 形成自适应治理闭环.

    核心职责:
        1. 调用 ``CCIntegration.check_provenance_for_cc1`` 获取溯源完整性
        2. 调用 ``CCIntegration.check_escalation_for_cc2`` 获取升级建议
        3. 生成反馈信号 (信号类型 / 阈值 / 触发标记)
        4. 信号 → 动作映射 (阈值调整 / 升级建议)
        5. 执行动作并记录审计事件 (CloudEvents 格式)
        6. 维护统计与事件日志, 支持查询与回溯

    治理阶段映射 (Kubernetes Controller Pattern)::

        Reconcile: 拉取溯源报告 (check_provenance / check_escalation)
        Evaluate : 生成信号 (generate_signals)
        Act      : 创建并执行动作 (create_actions / execute_action)
        Verify   : 统计与事件查询 (get_statistics / get_events)

    使用示例::

        from dy3_polaris.l0.cc_integration import FeedbackLoop

        loop = FeedbackLoop()
        outcome = loop.evaluate(
            annotation_id="kpa-001",
            trace_id="trace-001",
            session_id="sess-001",
        )

        if outcome["signals"]:
            for signal in outcome["signals"]:
                print(signal["signal_type"], signal["triggered"])

        # 查看阈值调整建议
        adj = loop.adjust_cc1_thresholds(0.25)
        # → {"pass_threshold_delta": 5.0, "flag_threshold_delta": 3.0, ...}

    Note:
        - 反馈飞轮在 ``evaluate`` 调用期间非线程安全, 并发场景下请为
          每个线程 / 任务创建独立实例.
        - ``create_actions`` 依赖 ``evaluate`` 中缓存的最近一份升级报告
          (``_last_escalation_report``) 以填充升级动作参数; 单独调用
          ``create_actions`` 时将使用默认升级参数.
    """

    #: 事件日志上限 (超出后保留最近一半)
    _MAX_EVENTS: int = 1000

    #: CC1 通过率 mock 偏移量 — pass_rate ≈ completeness + offset.
    #:
    #: 生产环境应替换为从 CC1 ReviewPipeline.get_statistics() 获取的真实通过率.
    #: CC1 通过率告警阈值
    _CC1_PASS_RATE_THRESHOLD: float = 0.8

    def __init__(
        self,
        cc_integration: CCIntegration | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        cc1_statistics_provider: Callable[[], dict[str, Any]] | Any | None = None,
    ) -> None:
        """初始化反馈飞轮.

        所有依赖项为 None 时自动创建默认实例, 开箱即用.

        Args:
            cc_integration: CC3 跨切面集成器, 为 None 时自动创建.
                反馈飞轮通过它获取溯源完整性报告与升级建议.
            circuit_breaker: 断路器, 为 None 时创建保护 CC3 的默认实例.
                用于在 CC3 连续失败时熔断, 避免反馈飞轮级联故障.
        """
        self._cc_integration: CCIntegration = (
            cc_integration or CCIntegration()
        )
        self._circuit_breaker: CircuitBreaker = (
            circuit_breaker or CircuitBreaker(module="cc3")
        )
        self._cc1_statistics_provider = cc1_statistics_provider

        # 反馈信号与动作存储
        self._signals: list[FeedbackSignal] = []
        self._actions: list[FeedbackAction] = []
        self._events: list[BridgeEvent] = []

        # 已执行信号 ID 集合 — 用于判断信号是否 "活跃" (未执行)
        self._executed_signal_ids: set[str] = set()

        # 最近一次 evaluate 缓存的报告 (供 create_actions 使用)
        self._last_prov_report: dict[str, Any] | None = None
        self._last_escalation_report: dict[str, Any] | None = None

        # 统计计数器
        self._stats: dict[str, Any] = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "total_signals": 0,
            "triggered_signals": 0,
            "total_actions": 0,
            "executed_actions": 0,
            "threshold_adjustments": 0,
            "escalations_suggested": 0,
            "circuit_breaker_trips": 0,
            "completeness_score_sum": 0.0,
            "total_latency_ms_sum": 0.0,
        }

    # ========================================================
    # 属性
    # ========================================================

    @property
    def cc_integration(self) -> CCIntegration:
        """CC3 跨切面集成器."""
        return self._cc_integration

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """断路器."""
        return self._circuit_breaker

    # ========================================================
    # 核心方法 — evaluate (治理闭环主入口)
    # ========================================================

    def evaluate(
        self,
        annotation_id: str,
        trace_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """评估溯源完整性并生成反馈信号与动作 (治理闭环主入口).

        执行流程 (Reconcile → Evaluate → Act → Verify):
            1. Reconcile: 经断路器调用 ``check_provenance_for_cc1``
               获取溯源完整性报告.
            2. Reconcile: 经断路器调用 ``check_escalation_for_cc2``
               获取 CC2 升级建议.
            3. Evaluate: 基于报告生成反馈信号 (:meth:`generate_signals`).
            4. Act: 为触发的信号创建反馈动作 (:meth:`create_actions`).
            5. Act: 逐个执行动作 (:meth:`execute_action`).
            6. Verify: 汇总建议并记录审计事件.

        Args:
            annotation_id: KPA 标注 ID.
            trace_id: OpenTelemetry 全链路 trace ID.
            session_id: 会话 ID.

        Returns:
            反馈评估结果字典::

                {
                    "success": bool,            # 评估是否成功
                    "annotation_id": str,
                    "trace_id": str,
                    "session_id": str,
                    "signals": list[dict],       # 反馈信号 (序列化)
                    "actions": list[dict],      # 反馈动作 (序列化)
                    "prov_report": dict,        # 溯源完整性报告
                    "escalation_report": dict,   # CC2 升级建议报告
                    "recommendations": list[str],# 建议
                    "error": str,               # 错误信息 (失败时)
                }
        """
        start_time = time.time()
        self._stats["total_evaluations"] += 1
        error_msg = ""

        # ----------------------------------------------------
        # Step 1 & 2: Reconcile — 获取溯源报告 (断路器保护)
        # ----------------------------------------------------
        try:
            prov_report = self._circuit_breaker.call(
                self._cc_integration.check_provenance_for_cc1,
                annotation_id,
            )
        except CircuitBreakerOpenError as exc:
            self._stats["circuit_breaker_trips"] += 1
            error_msg = str(exc)
            logger.warning(
                "反馈飞轮获取溯源报告断路器跳闸: annotation=%s, %s",
                annotation_id,
                exc,
            )
            prov_report = {
                "annotation_id": annotation_id,
                "source_complete": False,
                "source_tier": "unknown",
                "has_doi": False,
                "chain_verified": False,
                "completeness_score": 0.0,
                "recommendation": f"断路器开启, 降级为不完整: {exc}",
            }
        except Exception as exc:
            error_msg = str(exc)
            logger.exception(
                "反馈飞轮获取溯源报告异常: annotation=%s",
                annotation_id,
            )
            prov_report = {
                "annotation_id": annotation_id,
                "source_complete": False,
                "source_tier": "unknown",
                "has_doi": False,
                "chain_verified": False,
                "completeness_score": 0.0,
                "recommendation": f"溯源检查异常: {exc}",
            }

        try:
            escalation_report = self._circuit_breaker.call(
                self._cc_integration.check_escalation_for_cc2,
                annotation_id,
            )
        except CircuitBreakerOpenError as exc:
            self._stats["circuit_breaker_trips"] += 1
            if error_msg:
                error_msg = f"{error_msg}; {exc}"
            else:
                error_msg = str(exc)
            logger.warning(
                "反馈飞轮获取升级建议断路器跳闸: annotation=%s, %s",
                annotation_id,
                exc,
            )
            escalation_report = {
                "annotation_id": annotation_id,
                "needs_escalation": True,
                "reason": f"断路器开启, 降级为需升级: {exc}",
                "suggested_level": "approval",
                "risk_factors": ["断路器开启, 无法确认溯源状态"],
                "completeness_score": prov_report.get(
                    "completeness_score", 0.0
                ),
            }
        except Exception as exc:
            if error_msg:
                error_msg = f"{error_msg}; {exc}"
            else:
                error_msg = str(exc)
            logger.exception(
                "反馈飞轮获取升级建议异常: annotation=%s",
                annotation_id,
            )
            escalation_report = {
                "annotation_id": annotation_id,
                "needs_escalation": True,
                "reason": f"升级检查异常: {exc}",
                "suggested_level": "approval",
                "risk_factors": ["升级检查异常"],
                "completeness_score": prov_report.get(
                    "completeness_score", 0.0
                ),
            }

        # 缓存报告供 create_actions 使用
        self._last_prov_report = prov_report
        self._last_escalation_report = escalation_report

        # ----------------------------------------------------
        # Step 3: Evaluate — 生成反馈信号
        # ----------------------------------------------------
        action_results: list[dict[str, Any]] = []
        execution_errors: list[str] = []
        if error_msg:
            # 检查工具故障只能表示“未知”，不能伪装成低质量
            # 信号并驱动阈值收紧或升级动作。
            signals: list[FeedbackSignal] = []
            actions: list[FeedbackAction] = []
            recommendations = [
                "治理依赖不可用，本次未生成阈值调整或升级动作"
            ]
        else:
            signals = self.generate_signals(
                annotation_id,
                prov_report,
                escalation_report,
                trace_id=trace_id,
            )
            self._signals.extend(signals)

            # ----------------------------------------------------
            # Step 4: Act — 创建反馈动作
            # ----------------------------------------------------
            actions = self.create_actions(signals)
            self._actions.extend(actions)

            # ----------------------------------------------------
            # Step 5: Act — 执行动作
            # ----------------------------------------------------
            for action in actions:
                try:
                    result = self.execute_action(action)
                    action_results.append(result)
                except FeedbackLoopError as exc:
                    execution_errors.append(str(exc))
                    action_results.append(
                        {"success": False, "error": str(exc)}
                    )

            # ----------------------------------------------------
            # Step 6: Verify — 汇总建议 + 记录事件
            # ----------------------------------------------------
            recommendations = self._build_recommendations(
                prov_report, escalation_report, signals
            )

        latency_ms = (time.time() - start_time) * 1000.0
        self._stats["total_latency_ms_sum"] += latency_ms
        completeness = prov_report.get("completeness_score", 0.0)
        self._stats["completeness_score_sum"] += completeness

        success = not error_msg and not execution_errors
        if success:
            self._stats["successful_evaluations"] += 1
        else:
            self._stats["failed_evaluations"] += 1
            if execution_errors:
                error_msg = (error_msg + "; " if error_msg else "") + (
                    "; ".join(execution_errors)
                )

        self._record_event(
            annotation_id=annotation_id,
            trace_id=trace_id,
            session_id=session_id,
            prov_report=prov_report,
            escalation_report=escalation_report,
            signals=signals,
            actions=actions,
            action_results=action_results,
            recommendations=recommendations,
            success=success,
            error=error_msg,
            latency_ms=latency_ms,
        )

        logger.info(
            "反馈飞轮评估完成: annotation=%s, signals=%d, actions=%d, "
            "completeness=%.2f, escalated=%s",
            annotation_id,
            len(signals),
            len(actions),
            completeness,
            escalation_report.get("needs_escalation", False),
        )

        return {
            "success": success,
            "annotation_id": annotation_id,
            "trace_id": trace_id,
            "session_id": session_id,
            "signals": [s.model_dump() for s in signals],
            "actions": [a.model_dump() for a in actions],
            "action_results": action_results,
            "prov_report": prov_report,
            "escalation_report": escalation_report,
            "recommendations": recommendations,
            "error": error_msg,
        }

    # ========================================================
    # 信号生成
    # ========================================================

    def generate_signals(
        self,
        annotation_id: str,
        prov_report: dict[str, Any],
        escalation_report: dict[str, Any],
        trace_id: str = "",
    ) -> list[FeedbackSignal]:
        """基于溯源报告与升级建议生成反馈信号.

        信号生成规则:
            - ``completeness_score < 0.5`` → ``PROVENANCE_COMPLETENESS``
              (value=completeness, threshold=0.5, triggered=True)
            - ``not chain_verified`` → ``CHAIN_INTEGRITY``
              (value=0, threshold=1, triggered=True)
            - ``needs_escalation`` → ``ESCALATION_TRIGGER``
              (value=1, threshold=0, triggered=True)
            - 存在真实样本且 ``cc1_pass_rate < 0.8`` → ``REVIEW_QUALITY``
              (value=pass_rate, threshold=0.8, triggered=True)
            - 始终生成 ``THRESHOLD_ADJUSTMENT``
              (value=completeness, triggered=True)

        Args:
            annotation_id: KPA 标注 ID.
            prov_report: ``check_provenance_for_cc1`` 返回的溯源报告.
            escalation_report: ``check_escalation_for_cc2`` 返回的升级报告.

        Returns:
            反馈信号列表.
        """
        signals: list[FeedbackSignal] = []

        completeness = float(prov_report.get("completeness_score", 1.0))
        chain_verified = bool(prov_report.get("chain_verified", True))
        needs_escalation = bool(
            escalation_report.get("needs_escalation", False)
        )

        # 1. 溯源完整度信号
        if completeness < 0.5:
            signals.append(
                FeedbackSignal(
                    signal_type=FeedbackSignalType.PROVENANCE_COMPLETENESS,
                    source="cc3",
                    target="cc1",
                    trace_id=trace_id,
                    annotation_id=annotation_id,
                    value=completeness,
                    threshold=0.5,
                    triggered=True,
                    message=(
                        f"溯源完整度 {completeness:.2f} 低于阈值 0.50, "
                        f"建议收紧 CC1 评审阈值"
                    ),
                )
            )

        # 2. 溯源链完整性信号
        if not chain_verified:
            signals.append(
                FeedbackSignal(
                    signal_type=FeedbackSignalType.CHAIN_INTEGRITY,
                    source="cc3",
                    target="cc1",
                    trace_id=trace_id,
                    annotation_id=annotation_id,
                    value=0.0,
                    threshold=1.0,
                    triggered=True,
                    message="溯源链验证失败, 存在断裂节点, 需修复",
                )
            )

        # 3. 升级触发信号
        if needs_escalation:
            signals.append(
                FeedbackSignal(
                    signal_type=FeedbackSignalType.ESCALATION_TRIGGER,
                    source="cc3",
                    target="cc2",
                    trace_id=trace_id,
                    annotation_id=annotation_id,
                    value=1.0,
                    threshold=0.0,
                    triggered=True,
                    message=escalation_report.get(
                        "reason", "溯源缺失, 需升级 CC2 协同层级"
                    ),
                )
            )

        # 4. 评审质量信号：只消费 CC1 的真实评审样本。
        cc1_pass_rate = self._get_cc1_pass_rate()
        if (
            cc1_pass_rate is not None
            and cc1_pass_rate < self._CC1_PASS_RATE_THRESHOLD
        ):
            signals.append(
                FeedbackSignal(
                    signal_type=FeedbackSignalType.REVIEW_QUALITY,
                    source="cc3",
                    target="cc1",
                    trace_id=trace_id,
                    annotation_id=annotation_id,
                    value=cc1_pass_rate,
                    threshold=self._CC1_PASS_RATE_THRESHOLD,
                    triggered=True,
                    message=(
                        f"CC1 通过率 {cc1_pass_rate:.2f} 低于阈值 "
                        f"{self._CC1_PASS_RATE_THRESHOLD:.2f}"
                    ),
                )
            )

        # 5. 阈值调整信号 (始终生成)
        signals.append(
            FeedbackSignal(
                signal_type=FeedbackSignalType.THRESHOLD_ADJUSTMENT,
                source="cc3",
                target="cc1",
                trace_id=trace_id,
                annotation_id=annotation_id,
                value=completeness,
                threshold=0.5,
                triggered=True,
                message=(
                    f"基于完整度 {completeness:.2f} 的阈值调整反馈"
                ),
            )
        )

        return signals

    # ========================================================
    # 动作创建
    # ========================================================

    def create_actions(
        self,
        signals: list[FeedbackSignal],
    ) -> list[FeedbackAction]:
        """基于触发的反馈信号创建反馈动作.

        信号 → 动作映射:
            - ``PROVENANCE_COMPLETENESS`` / ``CHAIN_INTEGRITY`` /
              ``REVIEW_QUALITY`` / ``THRESHOLD_ADJUSTMENT``
              → ``adjust_threshold`` (target=cc1)
            - ``ESCALATION_TRIGGER`` → ``suggest_escalation`` (target=cc2)

        仅为 ``triggered=True`` 的信号创建动作.

        Args:
            signals: 反馈信号列表.

        Returns:
            反馈动作列表.
        """
        actions: list[FeedbackAction] = []
        completeness = self._extract_completeness(signals)

        for signal in signals:
            if not signal.triggered:
                continue

            mapping = _SIGNAL_ACTION_MAP.get(signal.signal_type)
            if mapping is None:
                logger.debug(
                    "信号类型 %s 无动作映射, 跳过",
                    signal.signal_type,
                )
                continue

            action_type, target_module = mapping
            parameters = self._build_action_parameters(
                signal, action_type, completeness
            )

            actions.append(
                FeedbackAction(
                    signal_id=signal.signal_id,
                    target_module=target_module,
                    action_type=action_type,
                    parameters=parameters,
                )
            )

        return actions

    # ========================================================
    # 动作执行
    # ========================================================

    def execute_action(
        self,
        action: FeedbackAction,
    ) -> dict[str, Any]:
        """执行单个反馈动作.

        动作类型分发:
            - ``adjust_threshold``: 调用 :meth:`adjust_cc1_thresholds`
              计算 CC1 阈值增量.
            - ``suggest_escalation``: 调用 :meth:`suggest_cc2_escalation`
              生成 CC2 升级建议.

        执行后标记动作为已执行, 记录结果与时间戳, 并将关联信号 ID
        加入已执行集合 (使其不再出现在 ``get_active_signals`` 中).

        Args:
            action: 待执行的反馈动作.

        Returns:
            动作执行结果字典.

        Raises:
            FeedbackLoopError: 动作执行失败 (未知动作类型等).
        """
        if action.executed:
            logger.debug(
                "动作 %s 已执行, 返回缓存结果",
                action.action_id,
            )
            return action.result

        try:
            result = self._dispatch_action(action)
        except FeedbackLoopError:
            raise
        except Exception as exc:
            raise FeedbackLoopError(
                signal_type=action.action_type,
                reason=f"动作执行失败: {exc}",
            ) from exc

        # 标记已执行
        action.executed = True
        action.executed_at = time.time()
        action.result = result

        # 记录已执行信号
        if action.signal_id:
            self._executed_signal_ids.add(action.signal_id)

        # 统计
        self._stats["executed_actions"] += 1
        if action.action_type == "adjust_threshold":
            self._stats["threshold_adjustments"] += 1
        elif action.action_type == "suggest_escalation":
            self._stats["escalations_suggested"] += 1

        logger.debug(
            "反馈动作执行完成: id=%s, type=%s, target=%s, success=%s",
            action.action_id,
            action.action_type,
            action.target_module,
            result.get("success", True),
        )

        return result

    # ========================================================
    # CC1 阈值调整
    # ========================================================

    def adjust_cc1_thresholds(
        self,
        completeness_score: float,
    ) -> dict[str, Any]:
        """根据溯源完整度计算 CC1 评审阈值调整量.

        调整策略 (完整度越低, 评审越严):
            - ``completeness < 0.3``: pass +5, flag +3 (大幅收紧)
            - ``completeness < 0.5``: pass +3, flag +2 (适度收紧)
            - ``completeness >= 0.8``: pass -2, flag 0 (放宽)
            - 其他: 无调整

        增量约定: ``delta > 0`` 表示提高阈值 (更严格),
        ``delta < 0`` 表示降低阈值 (更宽松).

        Args:
            completeness_score: 溯源完整度 (0.0-1.0).

        Returns:
            阈值调整结果字典::

                {
                    "success": bool,
                    "completeness_score": float,
                    "direction": "stricter" | "lenient" | "none",
                    "pass_threshold_delta": float,
                    "flag_threshold_delta": float,
                    "reason": str,
                }
        """
        completeness = float(completeness_score)

        # 收紧区间 (按上界升序匹配, 命中即停)
        for upper, pass_delta, flag_delta, direction, reason in (
            _THRESHOLD_STRATEGY
        ):
            if completeness < upper:
                return {
                    "success": True,
                    "completeness_score": completeness,
                    "direction": direction,
                    "pass_threshold_delta": pass_delta,
                    "flag_threshold_delta": flag_delta,
                    "reason": reason,
                }

        # 放宽区间 (完整度高时降低通过阈值, 更宽松)
        return {
            "success": True,
            "completeness_score": completeness,
            "direction": "lenient",
            "pass_threshold_delta": -2.0,
            "flag_threshold_delta": 0.0,
            "reason": "完整度较高, 适度放宽评审阈值",
        }

    # ========================================================
    # CC2 升级建议
    # ========================================================

    def suggest_cc2_escalation(
        self,
        escalation_report: dict[str, Any],
    ) -> dict[str, Any]:
        """基于升级报告生成 CC2 协同层级升级建议.

        将 CC3 的溯源升级建议转化为可直接注入 CC2
        :class:`RoutingContext` 的路由上下文更新.

        Args:
            escalation_report: ``check_escalation_for_cc2`` 返回的升级报告,
                包含 ``needs_escalation`` / ``suggested_level`` /
                ``reason`` / ``risk_factors`` / ``completeness_score``.

        Returns:
            CC2 升级建议字典::

                {
                    "success": bool,
                    "needs_escalation": bool,
                    "suggested_level": str,
                    "reason": str,
                    "risk_factors": list[str],
                    "completeness_score": float,
                    "routing_context_metadata": dict,
                    "severity": str,
                }
        """
        needs_escalation = bool(
            escalation_report.get("needs_escalation", False)
        )
        suggested_level = escalation_report.get(
            "suggested_level", "implicit"
        )
        reason = escalation_report.get("reason", "")
        risk_factors = list(escalation_report.get("risk_factors", []))
        completeness = float(
            escalation_report.get("completeness_score", 0.0)
        )

        # 严重级别判定
        if not needs_escalation:
            severity = AlertSeverity.INFO.value
        elif suggested_level == "intervention":
            severity = AlertSeverity.CRITICAL.value
        elif suggested_level == "approval":
            severity = AlertSeverity.ERROR.value
        elif suggested_level == "prompt":
            severity = AlertSeverity.WARNING.value
        else:
            severity = AlertSeverity.INFO.value

        # 构建路由上下文 metadata (注入 RoutingContext.metadata)
        routing_context_metadata: dict[str, Any] = {
            "cc3_provenance_complete": completeness >= 0.5,
            "cc3_completeness_score": completeness,
            "cc3_suggested_level": suggested_level,
            "cc3_risk_factors": risk_factors,
        }
        if needs_escalation:
            routing_context_metadata["cc3_escalation_required"] = True
            routing_context_metadata["cc3_escalation_reason"] = reason

        return {
            "success": True,
            "needs_escalation": needs_escalation,
            "suggested_level": suggested_level,
            "reason": reason,
            "risk_factors": risk_factors,
            "completeness_score": completeness,
            "routing_context_metadata": routing_context_metadata,
            "severity": severity,
        }

    # ========================================================
    # 查询方法
    # ========================================================

    def get_active_signals(self) -> list[FeedbackSignal]:
        """获取所有活跃 (已触发但尚未执行) 的反馈信号.

        活跃信号 = ``triggered=True`` 且其关联动作尚未执行
        (信号 ID 不在已执行集合中).

        Returns:
            活跃反馈信号列表.
        """
        return [
            s
            for s in self._signals
            if s.triggered and s.signal_id not in self._executed_signal_ids
        ]

    def get_statistics(self) -> dict[str, Any]:
        """获取反馈飞轮统计信息.

        Returns:
            统计字典, 包含::

                {
                    "total_evaluations": int,
                    "successful_evaluations": int,
                    "failed_evaluations": int,
                    "success_rate": float,
                    "total_signals": int,
                    "triggered_signals": int,
                    "active_signals": int,
                    "total_actions": int,
                    "executed_actions": int,
                    "threshold_adjustments": int,
                    "escalations_suggested": int,
                    "circuit_breaker_trips": int,
                    "avg_completeness": float,
                    "avg_latency_ms": float,
                    "circuit_breaker_status": dict,
                }
        """
        total_eval = self._stats["total_evaluations"]
        successful = self._stats["successful_evaluations"]

        avg_completeness = (
            self._stats["completeness_score_sum"] / total_eval
            if total_eval > 0
            else 0.0
        )
        avg_latency = (
            self._stats["total_latency_ms_sum"] / total_eval
            if total_eval > 0
            else 0.0
        )
        success_rate = (
            successful / total_eval * 100.0 if total_eval > 0 else 0.0
        )

        triggered = sum(1 for s in self._signals if s.triggered)

        return {
            "total_evaluations": total_eval,
            "successful_evaluations": successful,
            "failed_evaluations": self._stats["failed_evaluations"],
            "success_rate": round(success_rate, 2),
            "total_signals": len(self._signals),
            "triggered_signals": triggered,
            "active_signals": len(self.get_active_signals()),
            "total_actions": len(self._actions),
            "executed_actions": self._stats["executed_actions"],
            "threshold_adjustments": self._stats[
                "threshold_adjustments"
            ],
            "escalations_suggested": self._stats[
                "escalations_suggested"
            ],
            "circuit_breaker_trips": self._stats[
                "circuit_breaker_trips"
            ],
            "avg_completeness": round(avg_completeness, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "circuit_breaker_status": self._circuit_breaker.get_status(),
        }

    def get_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """获取最近的反馈事件.

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

    # ========================================================
    # 内部方法
    # ========================================================

    def _dispatch_action(
        self,
        action: FeedbackAction,
    ) -> dict[str, Any]:
        """根据动作类型分发执行.

        Args:
            action: 待执行的动作.

        Returns:
            动作执行结果.

        Raises:
            FeedbackLoopError: 未知动作类型.
        """
        if action.action_type == "adjust_threshold":
            completeness = float(
                action.parameters.get("completeness_score", 1.0)
            )
            adjustment = self.adjust_cc1_thresholds(completeness)
            return {
                "success": True,
                "action_id": action.action_id,
                "action_type": action.action_type,
                "target_module": action.target_module,
                "adjustment": adjustment,
            }

        if action.action_type == "suggest_escalation":
            escalation_report = self._build_escalation_report_for_action(
                action
            )
            suggestion = self.suggest_cc2_escalation(escalation_report)
            return {
                "success": True,
                "action_id": action.action_id,
                "action_type": action.action_type,
                "target_module": action.target_module,
                "suggestion": suggestion,
            }

        raise FeedbackLoopError(
            signal_type=action.action_type or "unknown",
            reason=f"未知动作类型: {action.action_type!r}",
        )

    def _build_action_parameters(
        self,
        signal: FeedbackSignal,
        action_type: str,
        completeness: float,
    ) -> dict[str, Any]:
        """为动作构建参数字典.

        Args:
            signal: 关联的反馈信号.
            action_type: 动作类型.
            completeness: 溯源完整度 (从信号中提取).

        Returns:
            动作参数字典.
        """
        parameters: dict[str, Any] = {
            "signal_type": signal.signal_type.value,
            "signal_value": signal.value,
            "signal_threshold": signal.threshold,
            "completeness_score": completeness,
        }

        if action_type == "adjust_threshold":
            adjustment = self.adjust_cc1_thresholds(completeness)
            parameters["pass_threshold_delta"] = adjustment[
                "pass_threshold_delta"
            ]
            parameters["flag_threshold_delta"] = adjustment[
                "flag_threshold_delta"
            ]
            parameters["direction"] = adjustment["direction"]
            parameters["reason"] = adjustment["reason"]

        elif action_type == "suggest_escalation":
            escalation_report = (
                self._last_escalation_report
                if self._last_escalation_report is not None
                else {}
            )
            parameters["suggested_level"] = escalation_report.get(
                "suggested_level", "approval"
            )
            parameters["reason"] = signal.message or escalation_report.get(
                "reason", ""
            )
            parameters["risk_factors"] = escalation_report.get(
                "risk_factors", []
            )

        return parameters

    def _build_escalation_report_for_action(
        self,
        action: FeedbackAction,
    ) -> dict[str, Any]:
        """为升级动作构建升级报告 (优先使用缓存的报告).

        Args:
            action: 升级动作.

        Returns:
            升级报告字典.
        """
        if self._last_escalation_report is not None:
            return self._last_escalation_report

        params = action.parameters
        return {
            "annotation_id": "",
            "needs_escalation": True,
            "reason": params.get("reason", ""),
            "suggested_level": params.get("suggested_level", "approval"),
            "risk_factors": params.get("risk_factors", []),
            "completeness_score": params.get("completeness_score", 0.0),
        }

    @staticmethod
    def _extract_completeness(
        signals: list[FeedbackSignal],
    ) -> float:
        """从信号列表中提取溯源完整度.

        优先取 ``THRESHOLD_ADJUSTMENT`` / ``PROVENANCE_COMPLETENESS``
        信号的 value (即完整度), 找不到时默认 1.0 (假设完整).

        Args:
            signals: 反馈信号列表.

        Returns:
            溯源完整度 (0.0-1.0).
        """
        preferred = (
            FeedbackSignalType.THRESHOLD_ADJUSTMENT,
            FeedbackSignalType.PROVENANCE_COMPLETENESS,
        )
        for signal in signals:
            if signal.signal_type in preferred:
                return float(signal.value)
        for signal in signals:
            if signal.value > 0:
                return float(signal.value)
        return 1.0

    def _get_cc1_pass_rate(self) -> float | None:
        """读取真实 CC1 评审通过率.

        ``ReviewPipeline.get_statistics`` 返回百分比口径。无提供者、
        读取失败或样本数为 0 时返回 ``None``，表示 unknown。
        """
        provider = self._cc1_statistics_provider
        if provider is None:
            return None
        try:
            if callable(provider):
                statistics = provider()
            else:
                getter = getattr(provider, "get_statistics", None)
                if not callable(getter):
                    return None
                statistics = getter()
            if not isinstance(statistics, dict):
                return None
            if int(statistics.get("total") or 0) <= 0:
                return None
            rate = float(statistics.get("pass_rate"))
            if rate > 1.0:
                rate /= 100.0
            return round(max(0.0, min(1.0, rate)), 4)
        except (TypeError, ValueError, AttributeError):
            logger.warning("CC1 真实统计读取失败，本次评审质量为 unknown")
            return None

    def _build_recommendations(
        self,
        prov_report: dict[str, Any],
        escalation_report: dict[str, Any],
        signals: list[FeedbackSignal],
    ) -> list[str]:
        """基于报告与信号汇总建议.

        Args:
            prov_report: 溯源完整性报告.
            escalation_report: 升级建议报告.
            signals: 反馈信号列表.

        Returns:
            建议字符串列表.
        """
        recommendations: list[str] = []
        completeness = prov_report.get("completeness_score", 1.0)

        if completeness < 0.5:
            recommendations.append(
                "溯源完整度不足, 建议提高 CC1 通过阈值并补充来源维度"
            )
        if not prov_report.get("chain_verified", True):
            recommendations.append(
                "溯源链断裂, 建议 CC1 加强溯源层校验并触发链路修复"
            )
        if escalation_report.get("needs_escalation", False):
            recommendations.append(
                f"溯源风险触发升级, 建议 CC2 路由到 "
                f"{escalation_report.get('suggested_level')} 协同层级"
            )
        if not prov_report.get("has_doi", False):
            recommendations.append(
                "来源缺少 DOI, 建议补充数字对象标识符"
            )
        if not prov_report.get("source_complete", True):
            recommendations.append(
                "来源维度不完整, 建议补充实验条件或原始来源"
            )

        if not recommendations:
            recommendations.append("溯源完整, 反馈飞轮维持当前评审阈值")

        return recommendations

    def _record_event(
        self,
        annotation_id: str,
        trace_id: str,
        session_id: str,
        prov_report: dict[str, Any],
        escalation_report: dict[str, Any],
        signals: list[FeedbackSignal],
        actions: list[FeedbackAction],
        action_results: list[dict[str, Any]],
        recommendations: list[str],
        success: bool,
        error: str,
        latency_ms: float,
    ) -> BridgeEvent:
        """记录反馈飞轮审计事件 (CloudEvents 格式).

        Args:
            annotation_id: KPA 标注 ID.
            trace_id: trace ID.
            session_id: 会话 ID.
            prov_report: 溯源完整性报告.
            escalation_report: 升级建议报告.
            signals: 反馈信号列表.
            actions: 反馈动作列表.
            action_results: 动作执行结果列表.
            recommendations: 建议列表.
            success: 评估是否成功.
            error: 错误信息.
            latency_ms: 评估延迟 (毫秒).

        Returns:
            已记录的桥接事件.
        """
        triggered_count = sum(1 for s in signals if s.triggered)
        escalation_count = sum(
            1 for a in actions if a.action_type == "suggest_escalation"
        )
        threshold_count = sum(
            1 for a in actions if a.action_type == "adjust_threshold"
        )

        # 主方向: 有升级时 CC3→CC2, 否则 CC3→CC1
        direction = (
            BridgeDirection.CC3_TO_CC2
            if escalation_count > 0
            else BridgeDirection.CC3_TO_CC1
        )

        payload: dict[str, Any] = {
            "success": success,
            "error": error,
            "latency_ms": round(latency_ms, 2),
            "annotation_id": annotation_id,
            "phase": GovernancePhase.ACT.value,
            "completeness_score": prov_report.get(
                "completeness_score", 0.0
            ),
            "signals": {
                "total": len(signals),
                "triggered": triggered_count,
                "types": [s.signal_type.value for s in signals],
            },
            "actions": {
                "total": len(actions),
                "threshold_adjustments": threshold_count,
                "escalations": escalation_count,
                "results": action_results,
            },
            "prov_report": prov_report,
            "escalation_report": {
                "needs_escalation": escalation_report.get(
                    "needs_escalation", False
                ),
                "suggested_level": escalation_report.get(
                    "suggested_level", ""
                ),
                "reason": escalation_report.get("reason", ""),
                "risk_factors": escalation_report.get("risk_factors", []),
            },
            "recommendations": recommendations,
        }

        event = BridgeEvent(
            source="cc3",
            target="cc1_cc2",
            direction=direction,
            event_type="feedback_evaluated",
            trace_id=trace_id,
            session_id=session_id,
            payload=payload,
        )
        self._events.append(event)

        # 事件日志超限裁剪 (保留最近一半)
        if len(self._events) > self._MAX_EVENTS:
            keep = self._MAX_EVENTS // 2
            self._events = self._events[-keep:]

        return event

    # ========================================================
    # 重置
    # ========================================================

    def reset(self) -> None:
        """重置反馈飞轮状态.

        清空信号、动作、事件日志与统计计数器, 并重置断路器到
        CLOSED 状态.
        """
        self._signals.clear()
        self._actions.clear()
        self._events.clear()
        self._executed_signal_ids.clear()
        self._last_prov_report = None
        self._last_escalation_report = None
        self._stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "total_signals": 0,
            "triggered_signals": 0,
            "total_actions": 0,
            "executed_actions": 0,
            "threshold_adjustments": 0,
            "escalations_suggested": 0,
            "circuit_breaker_trips": 0,
            "completeness_score_sum": 0.0,
            "total_latency_ms_sum": 0.0,
        }
        self._circuit_breaker.reset()
        logger.info("反馈飞轮已重置")


__all__ = ["FeedbackLoop"]
