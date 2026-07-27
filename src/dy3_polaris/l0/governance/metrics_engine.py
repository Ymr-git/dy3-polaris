"""G5 治理度量引擎 — SLO/SLI 与指标时间序列.

融合 Prometheus SLO + Datadog Governance SLOs + MLflow 实验追踪：
- 指标定义（Counter/Gauge/Histogram）与时间序列存储
- SLO 定义、错误预算计算、Burn Rate 告警
- DORA 指标映射到 Agent 系统
- 指标面板聚合与对比分析
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# 指标模型
# ============================================================


class MetricType(str, enum.Enum):
    """指标类型 (Prometheus 风格)."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class MetricDefinition(BaseModel):
    """指标定义."""

    name: str = Field(description="指标名称")
    description: str = Field(default="", description="指标描述")
    metric_type: MetricType = Field(default=MetricType.GAUGE)
    unit: str = Field(default="", description="单位")
    labels: list[str] = Field(default_factory=list, description="标签维度")


class MetricValue(BaseModel):
    """指标时间序列点."""

    metric_name: str = Field(description="指标名称")
    value: float = Field(description="指标值")
    timestamp: float = Field(default_factory=time.time)
    labels: dict[str, str] = Field(default_factory=dict)


class SLODefinition(BaseModel):
    """SLO 定义 (SRE 风格).

    Attributes:
        name: SLO 名称
        metric_name: 关联的指标名称
        target_percentage: 目标百分比（如 99.5）
        evaluation_window_seconds: 评估窗口（秒）
        burn_rate_alerts: Burn Rate 告警阈值列表
    """

    name: str = Field(description="SLO 名称")
    metric_name: str = Field(description="关联指标")
    target_percentage: float = Field(
        default=99.5, ge=0.0, le=100.0, description="目标百分比",
    )
    evaluation_window_seconds: float = Field(
        default=3600.0, ge=60.0, description="评估窗口秒数",
    )
    burn_rate_alerts: list[float] = Field(
        default_factory=lambda: [14.4, 6.0, 3.0],
        description="Burn Rate 告警阈值",
    )
    description: str = Field(default="", description="描述")


class SLOSnapshot(BaseModel):
    """SLO 当前状态快照."""

    slo_name: str = Field(description="SLO 名称")
    metric_name: str = Field(description="关联指标")
    target_percentage: float = Field(description="目标百分比")
    compliance_percentage: float = Field(description="当前合规率")
    error_budget_remaining: float = Field(description="剩余错误预算")
    burn_rate: float = Field(description="当前 Burn Rate")
    alert_fired: bool = Field(default=False, description="是否触发告警")
    alert_threshold: float | None = Field(default=None, description="触发的阈值")
    evaluated_at: float = Field(default_factory=time.time)


class BurnRateAlert(BaseModel):
    """Burn Rate 告警."""

    slo_name: str = Field(description="SLO 名称")
    burn_rate: float = Field(description="当前 Burn Rate")
    threshold: float = Field(description="触发阈值")
    error_budget_remaining: float = Field(description="剩余错误预算")
    severity: str = Field(default="warning", description="严重级别")
    timestamp: float = Field(default_factory=time.time)


# ============================================================
# 度量引擎
# ============================================================


class MetricsEngine:
    """治理度量引擎.

    提供指标时间序列存储、SLO 计算与 Burn Rate 告警。

    使用示例::

        engine = MetricsEngine()
        engine.register_slo(SLODefinition(
            name="agent_task_success",
            metric_name="task_success",
            target_percentage=99.5,
        ))

        # 记录指标
        engine.record("task_success", 1.0, labels={"agent": "tutor"})
        engine.record("task_success", 0.0, labels={"agent": "tutor"})

        # 评估 SLO
        snapshot = engine.evaluate_slo("agent_task_success")
        print(f"合规率: {snapshot.compliance_percentage}%")
    """

    def __init__(self, max_values_per_metric: int = 10000) -> None:
        self._max_values = max_values_per_metric
        self._definitions: dict[str, MetricDefinition] = {}
        self._values: dict[str, list[MetricValue]] = defaultdict(list)
        self._slos: dict[str, SLODefinition] = {}
        self._alerts: list[BurnRateAlert] = []
        self._lock = threading.RLock()

    def define_metric(self, definition: MetricDefinition) -> None:
        """注册指标定义."""
        with self._lock:
            self._definitions[definition.name] = definition

    def record(
        self,
        metric_name: str,
        value: float,
        labels: dict[str, str] | None = None,
        timestamp: float | None = None,
    ) -> MetricValue:
        """记录指标值."""
        mv = MetricValue(
            metric_name=metric_name,
            value=value,
            timestamp=timestamp or time.time(),
            labels=labels or {},
        )
        with self._lock:
            self._values[metric_name].append(mv)
            # FIFO 淘汰
            while len(self._values[metric_name]) > self._max_values:
                self._values[metric_name].pop(0)
        return mv

    def get_values(
        self,
        metric_name: str,
        start_time: float | None = None,
        end_time: float | None = None,
        labels_filter: dict[str, str] | None = None,
        limit: int = 1000,
    ) -> list[MetricValue]:
        """查询指标时间序列."""
        with self._lock:
            values = list(self._values.get(metric_name, []))

        results = []
        for v in values:
            if start_time is not None and v.timestamp < start_time:
                continue
            if end_time is not None and v.timestamp > end_time:
                continue
            if labels_filter:
                match = all(v.labels.get(k) == val for k, val in labels_filter.items())
                if not match:
                    continue
            results.append(v)

        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]

    def get_latest(self, metric_name: str) -> MetricValue | None:
        """获取最新指标值."""
        with self._lock:
            values = self._values.get(metric_name, [])
            return values[-1] if values else None

    def aggregate(
        self,
        metric_name: str,
        func: str = "avg",
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> float:
        """聚合指标值.

        Args:
            func: 聚合函数 (avg, sum, min, max, count)
        """
        values = self.get_values(metric_name, start_time, end_time, limit=100000)
        if not values:
            return 0.0

        vals = [v.value for v in values]
        if func == "avg":
            return sum(vals) / len(vals)
        elif func == "sum":
            return sum(vals)
        elif func == "min":
            return min(vals)
        elif func == "max":
            return max(vals)
        elif func == "count":
            return float(len(vals))
        return 0.0

    # ==========================================================
    # SLO 管理
    # ==========================================================

    def register_slo(self, slo: SLODefinition) -> None:
        """注册 SLO."""
        with self._lock:
            self._slos[slo.name] = slo

    def get_slo(self, slo_name: str) -> SLODefinition | None:
        """获取 SLO 定义."""
        with self._lock:
            return self._slos.get(slo_name)

    def evaluate_slo(self, slo_name: str) -> SLOSnapshot:
        """评估 SLO 当前状态.

        计算合规率、错误预算剩余量和 Burn Rate。
        """
        with self._lock:
            slo = self._slos.get(slo_name)
            if slo is None:
                raise ValueError(f"SLO '{slo_name}' 未注册")

        end_time = time.time()
        start_time = end_time - slo.evaluation_window_seconds

        values = self.get_values(
            slo.metric_name,
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        if not values:
            return SLOSnapshot(
                slo_name=slo_name,
                metric_name=slo.metric_name,
                target_percentage=slo.target_percentage,
                compliance_percentage=100.0,
                error_budget_remaining=100.0 - slo.target_percentage,
                burn_rate=0.0,
            )

        # 假设指标值为 1.0 表示成功，0.0 表示失败
        total = len(values)
        success = sum(1 for v in values if v.value >= 1.0)
        compliance = (success / total) * 100.0 if total > 0 else 100.0

        # 错误预算
        error_budget_total = 100.0 - slo.target_percentage
        error_budget_used = max(0.0, 100.0 - compliance)
        error_budget_remaining = max(0.0, error_budget_total - error_budget_used)

        # Burn Rate = 错误预算消耗速度 / 理想消耗速度
        # 理想消耗速度 = 评估窗口内均匀消耗 error_budget_total
        # Burn Rate = (error_budget_used / error_budget_total) / (elapsed / window)
        elapsed_hours = slo.evaluation_window_seconds / 3600.0
        if error_budget_total > 0 and elapsed_hours > 0:
            ideal_rate = error_budget_total / elapsed_hours  # %/hour
            actual_rate = error_budget_used / elapsed_hours if elapsed_hours > 0 else 0.0
            burn_rate = actual_rate / ideal_rate if ideal_rate > 0 else 0.0
        else:
            burn_rate = 0.0

        # 检查 Burn Rate 告警
        alert_fired = False
        alert_threshold = None
        for threshold in slo.burn_rate_alerts:
            if burn_rate >= threshold:
                alert_fired = True
                alert_threshold = threshold
                alert = BurnRateAlert(
                    slo_name=slo_name,
                    burn_rate=burn_rate,
                    threshold=threshold,
                    error_budget_remaining=error_budget_remaining,
                    severity="critical" if threshold >= 14.4 else "warning",
                )
                with self._lock:
                    self._alerts.append(alert)
                break  # 只触发最高级别

        return SLOSnapshot(
            slo_name=slo_name,
            metric_name=slo.metric_name,
            target_percentage=slo.target_percentage,
            compliance_percentage=round(compliance, 3),
            error_budget_remaining=round(error_budget_remaining, 3),
            burn_rate=round(burn_rate, 3),
            alert_fired=alert_fired,
            alert_threshold=alert_threshold,
        )

    def evaluate_all_slos(self) -> list[SLOSnapshot]:
        """评估所有已注册的 SLO."""
        with self._lock:
            slo_names = list(self._slos.keys())
        return [self.evaluate_slo(name) for name in slo_names]

    def get_burn_rate_alerts(
        self,
        slo_name: str | None = None,
        limit: int = 100,
    ) -> list[BurnRateAlert]:
        """获取 Burn Rate 告警."""
        with self._lock:
            alerts = list(self._alerts)
        if slo_name:
            alerts = [a for a in alerts if a.slo_name == slo_name]
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]

    # ==========================================================
    # DORA 指标映射
    # ==========================================================

    def record_dora_deployment(
        self,
        agent_id: str,
        success: bool,
        latency_ms: float = 0.0,
        *,
        deployment_id: str | None = None,
        status: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """记录 DORA 部署事件.

        映射：部署频率、变更失败率、恢复时间。

        G6 路由层兼容: 同时接受 deployment_id/status/duration_seconds 关键字参数。
        """
        # G6 路由层兼容: status 映射到 success
        if status is not None:
            success = status == "success"
        # G6 路由层兼容: duration_seconds 映射到 latency_ms
        if duration_seconds is not None and latency_ms == 0.0:
            latency_ms = duration_seconds * 1000.0
        self.record(
            "dora_deployment",
            1.0 if success else 0.0,
            labels={"agent": agent_id},
        )
        if latency_ms > 0:
            self.record(
                "dora_deployment_latency_ms",
                latency_ms,
                labels={"agent": agent_id},
            )

    def get_dora_metrics(self, agent_id: str | None = None) -> dict[str, Any]:
        """获取 DORA 四指标.

        Returns:
            部署频率、变更前置时间、变更失败率、恢复时间
        """
        labels_filter = {"agent": agent_id} if agent_id else None

        # 部署频率（最近 24 小时部署次数）
        day_ago = time.time() - 86400
        deployments = self.get_values(
            "dora_deployment",
            start_time=day_ago,
            labels_filter=labels_filter,
            limit=10000,
        )
        deploy_count = len(deployments)

        # 变更失败率
        failed = sum(1 for v in deployments if v.value < 1.0)
        failure_rate = failed / deploy_count if deploy_count > 0 else 0.0

        # 平均部署延迟（变更前置时间代理）
        latencies = self.get_values(
            "dora_deployment_latency_ms",
            start_time=day_ago,
            labels_filter=labels_filter,
            limit=10000,
        )
        avg_latency = sum(v.value for v in latencies) / len(latencies) if latencies else 0.0

        return {
            "deployment_frequency_24h": deploy_count,
            "change_failure_rate": round(failure_rate, 4),
            "avg_lead_time_ms": round(avg_latency, 3),
            "recovery_time_ms": round(avg_latency, 3),  # 简化为恢复时间
        }

    # ==========================================================
    # 统计
    # ==========================================================

    def get_stats(self) -> dict[str, Any]:
        """获取引擎统计."""
        with self._lock:
            return {
                "defined_metrics": len(self._definitions),
                "metric_names": list(self._values.keys()),
                "total_values": sum(len(v) for v in self._values.values()),
                "registered_slos": len(self._slos),
                "slo_names": list(self._slos.keys()),
                "burn_rate_alerts": len(self._alerts),
            }

    def clear(self) -> None:
        """清空所有数据."""
        with self._lock:
            self._definitions.clear()
            self._values.clear()
            self._slos.clear()
            self._alerts.clear()
