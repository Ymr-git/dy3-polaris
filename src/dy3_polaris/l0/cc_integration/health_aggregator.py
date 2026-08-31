"""CC4 三横切集成 — 跨模块健康聚合器.

实现 CC1 / CC2 / CC3 三大横切模块的健康状态聚合, 提供统一的
存活探针、断路器状态汇总、告警生成与总体健康判定能力.

核心能力:
- 各模块存活探针 (调用 get_statistics / statistics, 测量响应延迟)
- 延迟降级判定 (>= 1000ms → DEGRADED)
- 断路器三态汇总 (CLOSED / OPEN / HALF_OPEN)
- 告警生成 (UNHEALTHY → CRITICAL, DEGRADED → WARNING)
- 总体健康状态判定 (Kubernetes liveness/readiness 启发)
- 聚合指标导出 (Prometheus 风格多维度)

健康判定策略::

    ┌────────────────────────────────┬──────────────┐
    │ 条件                           │ 状态         │
    ├────────────────────────────────┼──────────────┤
    │ 模块未配置 (None)              │ UNKNOWN      │
    │ 探针成功, 延迟 < 1000ms        │ HEALTHY      │
    │ 探针成功, 延迟 >= 1000ms       │ DEGRADED     │
    │ 探针异常                        │ UNHEALTHY    │
    └────────────────────────────────┴──────────────┘

总体状态 (overall_status) 取最差值:
    UNHEALTHY > DEGRADED > (HEALTHY | UNKNOWN 混合) > HEALTHY

融合世界先进方案:
- Kubernetes: 存活/就绪探针 + 声明式健康状态
- Prometheus: 多维度指标采集与告警
- Istio: Service Mesh 级健康聚合 (Outlier Detection)
- Hystrix: 断路器状态仪表盘
- Envoy: 主动健康检查 + 被动断路
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from .models import (
    HealthCheck,
    HealthStatus,
    SystemHealthReport,
    AlertSeverity,
    CircuitState,
)
from .exceptions import HealthCheckError
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class HealthAggregator:
    """跨模块健康聚合器 — CC1/CC2/CC3 健康状态汇总.

    通过对每个横切模块执行轻量级探针 (调用 ``get_statistics()`` /
    ``statistics()``), 测量响应延迟并判定健康状态, 聚合为
    :class:`SystemHealthReport`.

    使用示例::

        from dy3_polaris.l0.cc1.review_pipeline import ReviewPipeline
        from dy3_polaris.l0.cc2.routing_engine import RoutingEngine
        from dy3_polaris.l0.cc2.approval_workflow import ApprovalWorkflowManager
        from dy3_polaris.l0.cc3.kpa_engine import KPAEngine
        from dy3_polaris.l0.cc_integration import HealthAggregator, CircuitBreaker

        aggregator = HealthAggregator(
            cc1_pipeline=ReviewPipeline(),
            cc2_routing_engine=RoutingEngine(),
            cc2_approval_manager=ApprovalWorkflowManager(),
            cc3_kpa_engine=KPAEngine(),
            circuit_breakers={"cc2": CircuitBreaker("cc2")},
        )

        report = aggregator.check_health()
        print(report.overall_status)  # HealthStatus.HEALTHY

        if report.active_alerts:
            for alert in report.active_alerts:
                print(alert["severity"], alert["message"])

    Note:
        本聚合器为 **只读监控组件**, 不修改任何模块状态.
        断路器状态仅读取, 不触发状态转换.
    """

    #: 延迟降级阈值 (毫秒) — 探针延迟超过此值判定为 DEGRADED.
    _LATENCY_DEGRADED_MS: float = 1000.0

    def __init__(
        self,
        cc1_pipeline: Any | None = None,
        cc2_routing_engine: Any | None = None,
        cc2_approval_manager: Any | None = None,
        cc3_kpa_engine: Any | None = None,
        circuit_breakers: dict[str, CircuitBreaker] | None = None,
    ) -> None:
        """初始化健康聚合器.

        所有依赖项均可为 None (对应模块未配置, 探针返回 UNKNOWN).
        断路器字典为 None 时使用空字典 (无断路器监控).

        Args:
            cc1_pipeline: CC1 评审管线 (需提供 ``get_statistics()``).
            cc2_routing_engine: CC2 路由引擎 (需提供 ``get_statistics()``
                与 ``routing_history`` 属性).
            cc2_approval_manager: CC2 审批工作流管理器 (需提供
                ``get_statistics()`` 与 ``records`` 属性).
            cc3_kpa_engine: CC3 KPA 标注引擎 (需提供 ``statistics()``).
            circuit_breakers: 断路器字典 {模块名: CircuitBreaker},
                用于汇总断路器三态.
        """
        self._cc1_pipeline = cc1_pipeline
        self._cc2_routing_engine = cc2_routing_engine
        self._cc2_approval_manager = cc2_approval_manager
        self._cc3_kpa_engine = cc3_kpa_engine
        self._circuit_breakers: dict[str, CircuitBreaker] = (
            circuit_breakers or {}
        )

        # 最近一次健康报告缓存 (供 get_metrics 引用)
        self._last_report: SystemHealthReport | None = None

    # ========================================================
    # 属性
    # ========================================================

    @property
    def circuit_breakers(self) -> dict[str, CircuitBreaker]:
        """断路器字典."""
        return self._circuit_breakers

    @property
    def last_report(self) -> SystemHealthReport | None:
        """最近一次健康报告 (未执行过检查时为 None)."""
        return self._last_report

    # ========================================================
    # 核心方法
    # ========================================================

    def check_health(self) -> SystemHealthReport:
        """执行全量健康检查并聚合为系统健康报告.

        依次对 CC1 / CC2 / CC3 执行存活探针, 汇总断路器状态,
        生成告警, 判定总体健康状态.

        Returns:
            系统健康报告, 包含各模块检查结果、断路器状态、活跃告警
            与总体状态.
        """
        try:
            checks: dict[str, HealthCheck] = {
                "cc1": self._check_cc1(),
                "cc2": self._check_cc2(),
                "cc3": self._check_cc3(),
            }
            circuit_states = self._check_circuits()
            alerts = self._generate_alerts(checks)
            overall = self._determine_overall(checks)

            report = SystemHealthReport(
                overall_status=overall,
                modules=checks,
                active_alerts=alerts,
                circuit_states=circuit_states,
                checked_at=time.time(),
            )
            self._last_report = report

            logger.info(
                "健康检查完成: overall=%s, cc1=%s, cc2=%s, cc3=%s, "
                "alerts=%d, circuits=%s",
                overall.value,
                checks["cc1"].status.value,
                checks["cc2"].status.value,
                checks["cc3"].status.value,
                len(alerts),
                circuit_states,
            )
            return report
        except Exception as exc:
            logger.exception("健康检查过程中发生未预期异常")
            raise HealthCheckError("system", str(exc)) from exc

    # ========================================================
    # 单模块探针
    # ========================================================

    def _check_cc1(self) -> HealthCheck:
        """检查 CC1 (四层反幻觉评审) 健康状态.

        通过调用 ``cc1_pipeline.get_statistics()`` 探测 CC1 存活性,
        测量响应延迟并判定状态.

        Returns:
            CC1 健康检查结果.
        """
        if self._cc1_pipeline is None:
            return HealthCheck(
                module="cc1",
                status=HealthStatus.UNKNOWN,
                details={"reason": "cc1_pipeline 未配置"},
            )
        return self._probe("cc1", self._cc1_pipeline.get_statistics)

    def _check_cc2(self) -> HealthCheck:
        """检查 CC2 (人机协作审批) 健康状态.

        CC2 由路由引擎与审批工作流管理器两部分组成, 分别探测:
        - ``routing_engine.get_statistics()`` + ``routing_history``
        - ``approval_manager.get_statistics()`` + ``records``

        任一已配置子组件探针异常 → UNHEALTHY (仍记录成功部分的指标).
        两子组件均未配置 → UNKNOWN.

        Returns:
            CC2 健康检查结果.
        """
        if (
            self._cc2_routing_engine is None
            and self._cc2_approval_manager is None
        ):
            return HealthCheck(
                module="cc2",
                status=HealthStatus.UNKNOWN,
                details={"reason": "cc2 模块未配置"},
            )

        start = time.time()
        details: dict[str, Any] = {}
        errors: list[str] = []

        # 路由引擎探测
        if self._cc2_routing_engine is not None:
            try:
                details["routing_statistics"] = (
                    self._cc2_routing_engine.get_statistics()
                )
                details["routing_history_count"] = len(
                    self._cc2_routing_engine.routing_history
                )
            except Exception as exc:
                logger.exception("CC2 routing_engine 健康检查失败")
                errors.append(f"routing_engine: {exc}")

        # 审批管理器探测
        if self._cc2_approval_manager is not None:
            try:
                details["approval_statistics"] = (
                    self._cc2_approval_manager.get_statistics()
                )
                details["approval_records_count"] = len(
                    self._cc2_approval_manager.records
                )
            except Exception as exc:
                logger.exception("CC2 approval_manager 健康检查失败")
                errors.append(f"approval_manager: {exc}")

        latency_ms = (time.time() - start) * 1000.0

        if errors:
            details["errors"] = errors
            return HealthCheck(
                module="cc2",
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency_ms, 2),
                details=details,
            )

        status = (
            HealthStatus.HEALTHY
            if latency_ms < self._LATENCY_DEGRADED_MS
            else HealthStatus.DEGRADED
        )
        return HealthCheck(
            module="cc2",
            status=status,
            latency_ms=round(latency_ms, 2),
            details=details,
        )

    def _check_cc3(self) -> HealthCheck:
        """检查 CC3 (溯源捕获层) 健康状态.

        通过调用 ``cc3_kpa_engine.statistics()`` 探测 CC3 存活性,
        测量响应延迟并判定状态.

        Returns:
            CC3 健康检查结果.
        """
        if self._cc3_kpa_engine is None:
            return HealthCheck(
                module="cc3",
                status=HealthStatus.UNKNOWN,
                details={"reason": "cc3_kpa_engine 未配置"},
            )
        return self._probe("cc3", self._cc3_kpa_engine.statistics)

    # ========================================================
    # 断路器与告警
    # ========================================================

    def _check_circuits(self) -> dict[str, str]:
        """汇总所有断路器的当前状态.

        遍历 ``circuit_breakers`` 字典, 调用每个断路器的
        ``get_status()`` 提取状态值.

        Returns:
            {断路器名: 状态字符串} 映射, 状态为
            ``"closed"`` / ``"open"`` / ``"half_open"`` / ``"unknown"``.
        """
        result: dict[str, str] = {}
        for name, breaker in self._circuit_breakers.items():
            try:
                status = breaker.get_status()
                result[name] = status.get(
                    "state", CircuitState.CLOSED.value
                )
            except Exception as exc:
                logger.warning(
                    "断路器状态查询失败: %s, %s", name, exc
                )
                result[name] = "unknown"
        return result

    def _generate_alerts(
        self, checks: dict[str, HealthCheck]
    ) -> list[dict[str, Any]]:
        """根据模块检查结果生成告警.

        - UNHEALTHY → CRITICAL 告警
        - DEGRADED → WARNING 告警
        - HEALTHY / UNKNOWN → 不产生告警

        Args:
            checks: 各模块健康检查结果.

        Returns:
            告警字典列表, 每条包含 module / severity / message /
            latency_ms / details / checked_at.
        """
        alerts: list[dict[str, Any]] = []
        for name, check in checks.items():
            if check.status == HealthStatus.UNHEALTHY:
                alerts.append(
                    {
                        "module": name,
                        "severity": AlertSeverity.CRITICAL.value,
                        "message": f"模块 {name} 不可用",
                        "latency_ms": check.latency_ms,
                        "details": check.details,
                        "checked_at": check.checked_at,
                    }
                )
            elif check.status == HealthStatus.DEGRADED:
                alerts.append(
                    {
                        "module": name,
                        "severity": AlertSeverity.WARNING.value,
                        "message": (
                            f"模块 {name} 性能降级 "
                            f"(延迟 {check.latency_ms:.1f}ms)"
                        ),
                        "latency_ms": check.latency_ms,
                        "details": check.details,
                        "checked_at": check.checked_at,
                    }
                )
        return alerts

    def _determine_overall(
        self, checks: dict[str, HealthCheck]
    ) -> HealthStatus:
        """根据各模块状态判定系统总体健康状态.

        判定优先级 (取最差状态):
            1. 任一 UNHEALTHY → UNHEALTHY
            2. 任一 DEGRADED (无 UNHEALTHY) → DEGRADED
            3. 全部 HEALTHY → HEALTHY
            4. 全部 UNKNOWN → UNKNOWN
            5. HEALTHY 与 UNKNOWN 混合 (无故障/降级) → DEGRADED
               (可观测性不完整视为降级)

        Args:
            checks: 各模块健康检查结果.

        Returns:
            系统总体健康状态.
        """
        if not checks:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in checks.values()]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        if all(s == HealthStatus.UNKNOWN for s in statuses):
            return HealthStatus.UNKNOWN

        # HEALTHY + UNKNOWN 混合: 可观测性不完整, 判定为 DEGRADED
        return HealthStatus.DEGRADED

    # ========================================================
    # 聚合指标
    # ========================================================

    def get_metrics(self) -> dict[str, Any]:
        """获取聚合指标 — 从所有模块采集统计信息.

        与 :meth:`check_health` 不同, 本方法直接调用各模块的统计
        接口采集详细指标, 不做延迟/健康判定, 适合对接 Prometheus
        等监控系统.

        每个模块的采集独立 try/except, 单模块失败不影响其他模块.

        Returns:
            聚合指标字典::

                {
                    "modules": {
                        "cc1": {...},   # CC1 统计或 {"error": ...}
                        "cc2": {...},   # CC2 路由+审批统计
                        "cc3": {...},   # CC3 KPA 统计
                    },
                    "circuit_breakers": {name: status_dict, ...},
                    "overall_status": str,   # 最近一次总体状态
                    "active_alerts": int,    # 最近一次告警数
                    "last_checked_at": float,
                    "collected_at": float,
                }
        """
        metrics: dict[str, Any] = {
            "modules": {},
            "circuit_breakers": {},
            "collected_at": time.time(),
        }

        # --- CC1 ---
        if self._cc1_pipeline is not None:
            try:
                metrics["modules"]["cc1"] = (
                    self._cc1_pipeline.get_statistics()
                )
            except Exception as exc:
                metrics["modules"]["cc1"] = {"error": str(exc)}
        else:
            metrics["modules"]["cc1"] = {"configured": False}

        # --- CC2 ---
        cc2_metrics: dict[str, Any] = {}
        if self._cc2_routing_engine is not None:
            try:
                cc2_metrics["routing"] = (
                    self._cc2_routing_engine.get_statistics()
                )
            except Exception as exc:
                cc2_metrics["routing_error"] = str(exc)
        if self._cc2_approval_manager is not None:
            try:
                cc2_metrics["approval"] = (
                    self._cc2_approval_manager.get_statistics()
                )
            except Exception as exc:
                cc2_metrics["approval_error"] = str(exc)
        metrics["modules"]["cc2"] = cc2_metrics

        # --- CC3 ---
        if self._cc3_kpa_engine is not None:
            try:
                metrics["modules"]["cc3"] = self._cc3_kpa_engine.statistics()
            except Exception as exc:
                metrics["modules"]["cc3"] = {"error": str(exc)}
        else:
            metrics["modules"]["cc3"] = {"configured": False}

        # --- 断路器 ---
        for name, breaker in self._circuit_breakers.items():
            try:
                metrics["circuit_breakers"][name] = breaker.get_status()
            except Exception as exc:
                metrics["circuit_breakers"][name] = {"error": str(exc)}

        # --- 最近报告摘要 ---
        if self._last_report is not None:
            metrics["overall_status"] = self._last_report.overall_status.value
            metrics["active_alerts"] = len(self._last_report.active_alerts)
            metrics["last_checked_at"] = self._last_report.checked_at

        return metrics

    # ========================================================
    # 内部辅助
    # ========================================================

    def _probe(
        self, module: str, func: Callable[[], Any]
    ) -> HealthCheck:
        """执行通用健康探针 — 调用无参函数并测量延迟.

        探针逻辑:
            - 调用成功且延迟 < 阈值 → HEALTHY
            - 调用成功但延迟 >= 阈值 → DEGRADED
            - 调用异常 → UNHEALTHY

        Args:
            module: 模块名 (cc1/cc2/cc3).
            func: 无参可调用对象 (如绑定的 ``get_statistics`` 方法).

        Returns:
            健康检查结果.
        """
        start = time.time()
        try:
            result = func()
            latency_ms = (time.time() - start) * 1000.0
            status = (
                HealthStatus.HEALTHY
                if latency_ms < self._LATENCY_DEGRADED_MS
                else HealthStatus.DEGRADED
            )
            details = (
                result if isinstance(result, dict) else {"value": result}
            )
            return HealthCheck(
                module=module,
                status=status,
                latency_ms=round(latency_ms, 2),
                details=details,
            )
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000.0
            logger.exception("%s 健康检查探针异常", module)
            return HealthCheck(
                module=module,
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency_ms, 2),
                details={"error": str(exc)},
            )


__all__ = ["HealthAggregator"]
