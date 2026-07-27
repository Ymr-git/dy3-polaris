"""L3 领域知识层 — 指标收集与监控.

融合世界先进方案的指标监控体系:
- Prometheus: 指标类型 (Counter/Gauge/Histogram/Summary) + exposition format
- LangSmith: 链路追踪和可观测性 (trace/span 级别指标)
- Langfuse: LLM 应用监控 (generation/usage/cost 指标)
- OpenTelemetry: 分布式追踪标准 + 指标语义约定

指标类型:
1. Counter   — 单调递增计数器 (借鉴 Prometheus Counter)
2. Gauge     — 可增可减仪表 (借鉴 Prometheus Gauge)
3. Histogram — 直方图分布 (借鉴 Prometheus Histogram + buckets)
4. Timer     — 计时器 (上下文管理器, 基于 Histogram 实现)

MetricsCollector 集中管理所有 L3 层指标:
- retrieval_latency: 检索延迟
- cache_hit_rate: 缓存命中率
- index_size: 索引大小
- query_throughput: 查询吞吐量
- fact_check_pass_rate: 事实校验通过率
- ingestion_success_rate: 摄入成功率
- connector_availability: 连接器可用率

所有指标均线程安全 (threading.RLock)。
支持 Prometheus 格式导出和快照序列化。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 默认配置
# ============================================================

# 延迟直方图默认桶边界 (秒), 借鉴 Prometheus 默认桶 + LLM 应用监控优化
DEFAULT_LATENCY_BUCKETS: list[float] = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
]

# 百分位列表
_PERCENTILES: list[tuple[str, float]] = [
    ("p50", 0.50),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
]


# ============================================================
# 指标类型与采样
# ============================================================


class MetricType(str, Enum):
    """指标类型 (借鉴 Prometheus 四种基本指标类型).

    Attributes:
        COUNTER: 单调递增计数器 (如请求总数、错误总数)
        GAUGE: 可增可减仪表 (如队列长度、内存使用量)
        HISTOGRAM: 直方图分布 (如延迟分布、响应大小分布)
        TIMER: 计时器 (基于 Histogram 实现, 记录操作耗时)
    """

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricSample:
    """指标采样 (借鉴 OpenTelemetry MetricData + Prometheus Sample).

    表示某一时刻的指标采样值, 携带标签和时间戳。

    Attributes:
        name: 指标名称
        value: 采样值
        metric_type: 指标类型
        labels: 标签键值对 (如 {"method": "vector", "status": "success"})
        timestamp: 采样时间戳 (Unix epoch 秒)
        description: 指标描述
    """

    name: str
    value: float
    metric_type: MetricType
    labels: dict[str, str]
    timestamp: float
    description: str = ""


# ============================================================
# 计数器 — 单调递增指标
# ============================================================


class Counter:
    """计数器指标 (借鉴 Prometheus Counter).

    计数器是单调递增的指标, 只能增加不能减少。
    适用于累计型统计, 如:
    - 查询总数 (query_count)
    - 检索次数 (retrieval_count)
    - 摄入成功次数 (ingestion_success_count)

    支持标签维度: 不同标签组合对应不同的计数值。
    例如: query_count{method="vector"} 和 query_count{method="keyword"}。

    Attributes:
        _name: 指标名称
        _description: 指标描述
        _lock: 线程安全锁
        _labeled_values: 标签维度计数值 {label_key: value}
        _total: 全部标签维度的总计值
    """

    def __init__(self, name: str, *, description: str = "") -> None:
        """初始化计数器.

        Args:
            name: 指标名称 (如 "retrieval_count")
            description: 指标描述
        """
        self._name = name
        self._description = description
        self._lock = threading.RLock()
        self._labeled_values: dict[str, float] = {}
        self._total: float = 0.0

    @staticmethod
    def _labels_to_key(labels: dict[str, str] | None) -> str:
        """将标签字典转为可哈希的字符串键 (借鉴 Prometheus label 排序)."""
        if not labels:
            return ""
        # 按 key 排序保证一致性
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    @staticmethod
    def _key_to_labels(key: str) -> dict[str, str]:
        """将标签键还原为标签字典."""
        if not key:
            return {}
        result: dict[str, str] = {}
        for pair in key.split("|"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                result[k] = v
        return result

    def inc(
        self,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """增加计数值 (借鉴 Prometheus Counter.inc).

        Args:
            value: 增量值 (必须 >= 0)
            labels: 标签键值对 (可选)

        Raises:
            ValueError: 增量值为负数
        """
        if value < 0:
            raise ValueError(
                f"计数器 '{self._name}' 只能递增, 收到负值: {value}"
            )

        label_key = self._labels_to_key(labels)
        with self._lock:
            self._labeled_values[label_key] = (
                self._labeled_values.get(label_key, 0.0) + value
            )
            self._total += value

    @property
    def name(self) -> str:
        """指标名称."""
        return self._name

    @property
    def description(self) -> str:
        """指标描述."""
        return self._description

    @property
    def value(self) -> float:
        """总计值 (所有标签维度之和)."""
        with self._lock:
            return self._total

    def get_labeled_values(self) -> dict[str, float]:
        """获取按标签维度分组的计数值 (副本)."""
        with self._lock:
            return dict(self._labeled_values)

    def snapshot(self) -> dict[str, Any]:
        """生成快照 (借鉴 OpenTelemetry MetricData 序列化).

        Returns:
            包含指标完整状态的字典
        """
        with self._lock:
            labeled: list[dict[str, Any]] = []
            for key, val in self._labeled_values.items():
                labeled.append({
                    "labels": self._key_to_labels(key),
                    "value": val,
                })
            return {
                "name": self._name,
                "type": MetricType.COUNTER.value,
                "description": self._description,
                "value": self._total,
                "labeled_values": labeled,
            }


# ============================================================
# 仪表 — 可增可减指标
# ============================================================


@dataclass
class _GaugeEntry:
    """仪表内部条目 (不对外暴露)."""

    value: float
    description: str
    timestamp: float = field(default_factory=time.time)


# ============================================================
# 直方图 — 分布统计指标
# ============================================================


class Histogram:
    """直方图指标 (借鉴 Prometheus Histogram).

    直方图统计值的分布情况, 适用于:
    - 检索延迟分布 (retrieval_latency)
    - 响应大小分布 (response_size)
    - 批处理耗时分布 (batch_duration)

    实现要点:
    - 存储所有观测值以计算精确百分位
    - 维护桶计数用于 Prometheus 导出
    - 桶边界可自定义 (默认为延迟优化桶)

    Attributes:
        _name: 指标名称
        _description: 指标描述
        _buckets: 桶上界列表 (升序)
        _lock: 线程安全锁
        _observations: 所有观测值列表
        _sum: 观测值总和
        _count: 观测次数
        _bucket_counts: 每个桶的累积计数
    """

    def __init__(
        self,
        name: str,
        *,
        buckets: list[float] | None = None,
        description: str = "",
    ) -> None:
        """初始化直方图.

        Args:
            name: 指标名称 (如 "retrieval_latency")
            buckets: 桶上界列表 (升序), None 则使用默认延迟桶
            description: 指标描述
        """
        self._name = name
        self._description = description
        self._buckets = sorted(buckets) if buckets else list(DEFAULT_LATENCY_BUCKETS)
        self._lock = threading.RLock()
        self._observations: list[float] = []
        self._sum: float = 0.0
        self._count: int = 0
        self._bucket_counts: list[int] = [0] * len(self._buckets)

    @property
    def name(self) -> str:
        """指标名称."""
        return self._name

    @property
    def description(self) -> str:
        """指标描述."""
        return self._description

    def observe(
        self,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """记录一次观测 (借鉴 Prometheus Histogram.observe).

        Args:
            value: 观测值
            labels: 标签键值对 (可选, 当前实现记录在观测元数据中)
        """
        with self._lock:
            self._observations.append(value)
            self._sum += value
            self._count += 1
            # 更新桶计数 (累积计数: 值 <= 桶上界则该桶 +1)
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._bucket_counts[i] += 1

    @property
    def count(self) -> int:
        """观测次数."""
        with self._lock:
            return self._count

    @property
    def sum(self) -> float:
        """观测值总和."""
        with self._lock:
            return self._sum

    @property
    def avg(self) -> float:
        """观测值平均值."""
        with self._lock:
            if self._count == 0:
                return 0.0
            return self._sum / self._count

    @property
    def percentile(self) -> dict[str, float]:
        """百分位数 (p50/p90/p95/p99, 借鉴 Prometheus Histogram_quantile).

        使用线性插值法计算精确百分位。

        Returns:
            百分位数字典, 如 {"p50": 0.05, "p90": 0.2, "p95": 0.35, "p99": 0.8}
        """
        with self._lock:
            if not self._observations:
                return {name: 0.0 for name, _ in _PERCENTILES}

            sorted_obs = sorted(self._observations)
            return {
                name: self._compute_percentile(sorted_obs, p)
                for name, p in _PERCENTILES
            }

    @staticmethod
    def _compute_percentile(sorted_data: list[float], p: float) -> float:
        """计算百分位数 (线性插值法, 借鉴 NumPy percentile).

        Args:
            sorted_data: 已排序的数据列表
            p: 百分位 (0.0 ~ 1.0)

        Returns:
            百分位值
        """
        if not sorted_data:
            return 0.0
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]

        # 线性插值
        k = (n - 1) * p
        f = int(k)
        c = k - f

        if f + 1 < n:
            return sorted_data[f] + (sorted_data[f + 1] - sorted_data[f]) * c
        return sorted_data[f]

    def snapshot(self) -> dict[str, Any]:
        """生成快照.

        Returns:
            包含指标完整状态的字典, 包括桶分布和百分位
        """
        with self._lock:
            buckets_repr = {
                f"{bound}": count
                for bound, count in zip(self._buckets, self._bucket_counts)
            }
            # 添加 +Inf 桶 (总计数)
            buckets_repr["+Inf"] = self._count

            pct = self.percentile
            return {
                "name": self._name,
                "type": MetricType.HISTOGRAM.value,
                "description": self._description,
                "count": self._count,
                "sum": round(self._sum, 6),
                "avg": round(self.avg, 6),
                "percentile": pct,
                "buckets": buckets_repr,
            }


# ============================================================
# 计时器 — 上下文管理器
# ============================================================


class Timer:
    """计时器 (上下文管理器, 借鉴 OpenTelemetry Span + LangSmith trace).

    用于测量代码块的执行时间, 退出上下文时自动记录到 MetricsCollector
    关联的 Histogram 中。

    Usage::

        collector = MetricsCollector()
        with collector.timer("retrieval_latency"):
            results = store.search_text("催化剂")
        # 退出 with 块后, 耗时自动记录到 retrieval_latency 直方图

    Attributes:
        _name: 指标名称
        _collector: 关联的指标收集器
        _start: 开始时间戳
        _elapsed: 已记录的耗时 (秒)
    """

    def __init__(self, name: str, collector: MetricsCollector) -> None:
        """初始化计时器.

        Args:
            name: 指标名称 (将作为 Histogram 名称)
            collector: 关联的指标收集器
        """
        self._name = name
        self._collector = collector
        self._start: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self) -> Timer:
        """进入上下文, 记录开始时间."""
        self._start = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """退出上下文, 计算耗时并记录到收集器."""
        self._elapsed = time.time() - self._start
        self._collector._record_timer(self._name, self._elapsed)

    @property
    def elapsed(self) -> float:
        """已记录的耗时 (秒). 仅在退出上下文后有效."""
        return self._elapsed


# ============================================================
# 指标收集器 — 集中管理
# ============================================================


class MetricsCollector:
    """指标收集器 (借鉴 LangSmith + Langfuse 可观测性平台).

    集中管理所有 L3 层指标, 提供统一的指标创建、查询和导出接口。

    预定义的 L3 层监控指标:
    - retrieval_latency: 检索延迟 (Histogram, 秒)
    - cache_hit_rate: 缓存命中率 (Gauge, 0.0~1.0)
    - index_size: 索引大小 (Gauge, 条目数)
    - query_throughput: 查询吞吐量 (Counter, 次数/秒)
    - fact_check_pass_rate: 事实校验通过率 (Gauge, 0.0~1.0)
    - ingestion_success_rate: 摄入成功率 (Gauge, 0.0~1.0)
    - connector_availability: 连接器可用率 (Gauge, 0.0~1.0)

    Usage::

        collector = MetricsCollector()

        # 计数器
        counter = collector.counter("query_count", description="查询总数")
        counter.inc(labels={"method": "vector"})

        # 直方图
        hist = collector.histogram("retrieval_latency", description="检索延迟")
        hist.observe(0.05)

        # 计时器
        with collector.timer("index_build_time"):
            build_index()

        # 仪表
        collector.gauge("cache_hit_rate", 0.85, description="缓存命中率")

        # 快照
        snapshot = collector.snapshot()

        # Prometheus 格式导出
        prom_text = collector.export_prometheus()

    Attributes:
        _lock: 线程安全锁
        _counters: 计数器字典 {name: Counter}
        _histograms: 直方图字典 {name: Histogram}
        _gauges: 仪表字典 {name: _GaugeEntry}
    """

    def __init__(self) -> None:
        """初始化指标收集器."""
        self._lock = threading.RLock()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._gauges: dict[str, _GaugeEntry] = {}

    # --------------------------------------------------------
    # 指标创建与获取
    # --------------------------------------------------------

    def counter(self, name: str, *, description: str = "") -> Counter:
        """获取或创建计数器 (幂等).

        如果同名计数器已存在, 返回已有实例 (忽略新的 description);
        否则创建新计数器。

        Args:
            name: 指标名称
            description: 指标描述 (仅创建时生效)

        Returns:
            计数器实例
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description=description)
            return self._counters[name]

    def histogram(
        self,
        name: str,
        *,
        buckets: list[float] | None = None,
        description: str = "",
    ) -> Histogram:
        """获取或创建直方图 (幂等).

        Args:
            name: 指标名称
            buckets: 桶上界列表 (仅创建时生效)
            description: 指标描述 (仅创建时生效)

        Returns:
            直方图实例
        """
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name, buckets=buckets, description=description
                )
            return self._histograms[name]

    def timer(self, name: str) -> Timer:
        """创建计时器.

        每次调用返回新的 Timer 实例, 退出上下文时自动记录到
        同名 Histogram 中。

        Args:
            name: 指标名称 (将作为 Histogram 名称)

        Returns:
            Timer 实例
        """
        return Timer(name, self)

    def gauge(
        self,
        name: str,
        value: float,
        *,
        description: str = "",
    ) -> None:
        """设置仪表值 (借鉴 Prometheus Gauge.set).

        仪表记录当前时刻的值, 每次设置覆盖旧值。

        Args:
            name: 指标名称
            value: 仪表值
            description: 指标描述
        """
        with self._lock:
            existing = self._gauges.get(name)
            old_desc = existing.description if existing else description
            self._gauges[name] = _GaugeEntry(
                value=value,
                description=description or old_desc,
                timestamp=time.time(),
            )

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    def _record_timer(self, name: str, elapsed: float) -> None:
        """记录计时器观测值 (由 Timer.__exit__ 调用)."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name, description="Timer metric"
                )
            self._histograms[name].observe(elapsed)

    # --------------------------------------------------------
    # 查询与导出
    # --------------------------------------------------------

    def get_counter(self, name: str) -> Counter | None:
        """获取计数器 (不存在返回 None)."""
        with self._lock:
            return self._counters.get(name)

    def get_histogram(self, name: str) -> Histogram | None:
        """获取直方图 (不存在返回 None)."""
        with self._lock:
            return self._histograms.get(name)

    def get_gauge(self, name: str) -> float | None:
        """获取仪表值 (不存在返回 None)."""
        with self._lock:
            entry = self._gauges.get(name)
            return entry.value if entry else None

    def snapshot(self) -> dict[str, Any]:
        """生成所有指标的快照 (借鉴 Langfuse snapshot).

        Returns:
            包含所有指标状态的字典
        """
        with self._lock:
            counters = {
                name: counter.snapshot()
                for name, counter in self._counters.items()
            }
            histograms = {
                name: hist.snapshot()
                for name, hist in self._histograms.items()
            }
            gauges = {
                name: {
                    "name": name,
                    "type": MetricType.GAUGE.value,
                    "description": entry.description,
                    "value": entry.value,
                    "timestamp": entry.timestamp,
                }
                for name, entry in self._gauges.items()
            }
            return {
                "counters": counters,
                "histograms": histograms,
                "gauges": gauges,
                "summary": {
                    "counter_count": len(counters),
                    "histogram_count": len(histograms),
                    "gauge_count": len(gauges),
                },
            }

    def reset(self) -> None:
        """重置所有指标 (清空全部数据)."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
        logger.info("所有指标已重置")

    def export_prometheus(self) -> str:
        """导出为 Prometheus exposition 格式 (借鉴 Prometheus /metrics endpoint).

        格式遵循 Prometheus 文本格式规范:
        - # HELP 行: 指标描述
        - # TYPE 行: 指标类型
        - 数据行: metric_name{labels} value

        Returns:
            Prometheus 格式的指标文本
        """
        lines: list[str] = []

        with self._lock:
            # 导出计数器
            for name, counter in sorted(self._counters.items()):
                desc = counter.description or name
                lines.append(f"# HELP {name} {desc}")
                lines.append(f"# TYPE {name} counter")

                labeled = counter.get_labeled_values()
                if not labeled:
                    lines.append(f"{name} {counter.value}")
                else:
                    for key, val in sorted(labeled.items()):
                        labels = Counter._key_to_labels(key)
                        if labels:
                            label_str = ",".join(
                                f'{k}="{v}"' for k, v in sorted(labels.items())
                            )
                            lines.append(f'{name}{{{label_str}}} {val}')
                        else:
                            lines.append(f"{name} {val}")

                lines.append("")

            # 导出直方图
            for name, hist in sorted(self._histograms.items()):
                desc = hist.description or name
                lines.append(f"# HELP {name} {desc}")
                lines.append(f"# TYPE {name} histogram")

                snap = hist.snapshot()
                buckets = snap["buckets"]
                for bound_str, count in buckets.items():
                    if bound_str == "+Inf":
                        lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                    else:
                        lines.append(f'{name}_bucket{{le="{bound_str}"}} {count}')

                lines.append(f"{name}_count {snap['count']}")
                lines.append(f"{name}_sum {snap['sum']}")
                lines.append("")

            # 导出仪表
            for name, entry in sorted(self._gauges.items()):
                desc = entry.description or name
                lines.append(f"# HELP {name} {desc}")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {entry.value}")
                lines.append("")

        return "\n".join(lines)


__all__ = [
    "MetricType",
    "MetricSample",
    "Counter",
    "Histogram",
    "Timer",
    "MetricsCollector",
]
