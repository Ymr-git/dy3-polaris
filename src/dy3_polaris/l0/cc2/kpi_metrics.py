"""KPI 指标引擎模块
==================

本模块实现了 CC2 计划-审批门控系统（Plan-Approval Gate）的 KPI 指标引擎，
用于追踪人机协作系统的 9 项关键绩效指标（KPI），并支持动态阈值调整、
趋势分析、告警生成与仪表盘数据输出。

9 项 KPI：
    1. 审批响应时间（Approval Response Time）—— 从请求到决策的平均时间
    2. 自动批准率（Auto-Approval Rate）—— 自动批准操作占比
    3. 审批拒绝率（Approval Rejection Rate）—— 被拒绝操作占比
    4. 干预触发率（Intervention Trigger Rate）—— 每会话 L4 干预次数
    5. CC1 联动率（CC1 Integration Rate）—— CC1 审查触发 L3/L4 占比
    6. 疲劳指数（Fatigue Index）—— 平均用户疲劳评分
    7. 信任度演进（Trust Evolution）—— 信任分数趋势
    8. 批量审批效率（Batch Efficiency）—— 批量与单独审批比率
    9. 纠错反馈率（Correction Rate）—— 每 100 次操作的纠错次数

融合方案：
    - **NIST AI RMF Measure**：持续 KPI 度量与报告，建立可审计的指标基线
    - **Google SRE SLI/SLO**：服务级别指标与错误预算，量化系统可靠性
    - **Datadog / Grafana**：仪表盘式指标可视化，支持多维数据下钻
    - **Prometheus**：时间序列指标与告警，滑动窗口趋势分析

线程安全：
    所有公共方法均通过 ``threading.RLock`` 保护，支持多线程并发调用。

动态阈值调整：
    当历史数据持续改善时，引擎可根据百分位数自动收紧阈值，
    实现"水涨船高"的渐进式质量目标管理。
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "KPICategory",
    "KPIStatus",
    "TrendDirection",
    "KPISample",
    "KPIThreshold",
    "KPITrend",
    "KPISummary",
    "KPIMetricsEngine",
    "KPI_NAMES",
]

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════════════════════════════════════


class KPICategory(str, Enum):
    """KPI 分类枚举。

    按照 NIST AI RMF 的度量维度，将 9 项 KPI 划分为四大类别：
    效率、安全、质量、信任。
    """

    EFFICIENCY = "efficiency"
    """效率类 —— 衡量审批流程的速度与吞吐量。"""

    SAFETY = "safety"
    """安全类 —— 衡量风险控制与人工干预的有效性。"""

    QUALITY = "quality"
    """质量类 —— 衡量联动机制与纠错反馈的健康度。"""

    TRUST = "trust"
    """信任类 —— 衡量用户疲劳度与信任度演进。"""


class KPIStatus(str, Enum):
    """KPI 状态枚举，采用交通灯模型。

    借鉴 Google SRE 的错误预算理念：
    - 绿色：指标在 SLO 目标内，错误预算充足
    - 黄色：指标接近预算耗尽，需要关注
    - 红色：指标超出预算，需要立即干预
    """

    GREEN = "green"
    """绿色 —— 健康，指标在目标范围内。"""

    YELLOW = "yellow"
    """黄色 —— 警告，指标接近或略微超出阈值。"""

    RED = "red"
    """红色 —— 危险，指标严重超出阈值，需要立即处理。"""


class TrendDirection(str, Enum):
    """趋势方向枚举。

    用于表示 KPI 指标在滑动窗口内的变化方向。
    """

    UP = "up"
    """上升 —— 指标值呈上升趋势。"""

    DOWN = "down"
    """下降 —— 指标值呈下降趋势。"""

    STABLE = "stable"
    """稳定 —— 指标值变化幅度在阈值范围内。"""


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic 数据模型
# ═══════════════════════════════════════════════════════════════════════════


class KPISample(BaseModel):
    """单次 KPI 采样数据。

    每条采样记录代表某一时刻某项 KPI 的观测值，
    携带上下文信息与元数据以支持多维分析。

    Attributes:
        kpi_name: KPI 标识名称
        value: 采样值
        timestamp: 采样时间戳（UTC）
        context: 上下文信息（会话 ID、用户 ID 等）
        metadata: 元数据（数据来源、置信度等）
    """

    model_config = ConfigDict(extra="forbid")

    kpi_name: str = Field(..., description="KPI 标识名称")
    value: float = Field(..., description="采样值")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="采样时间戳（UTC）",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="上下文信息（会话 ID、用户 ID 等）",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="元数据（数据来源、置信度等）",
    )


class KPIThreshold(BaseModel):
    """KPI 阈值配置。

    阈值判定规则根据 ``is_inverse`` 标志分为两种模式：

    **逆指标（``is_inverse=True``，值越低越好）**：
        - 绿色: ``value <= green_max``
        - 黄色: ``green_max < value <= yellow_max``
        - 红色: ``value > yellow_max``（即 ``value >= red_min``）

    **正向指标（``is_inverse=False``，值越高越好）**：
        - 绿色: ``value >= green_max``
        - 黄色: ``yellow_max <= value < green_max``
        - 红色: ``value < yellow_max``（即 ``value < red_min``）

    Attributes:
        kpi_name: KPI 标识名称
        green_max: 绿色阈值边界
        yellow_max: 黄色阈值边界
        red_min: 红色起始阈值
        is_inverse: True 表示值越低越好
    """

    model_config = ConfigDict(extra="forbid")

    kpi_name: str = Field(..., description="KPI 标识名称")
    green_max: float = Field(..., description="绿色阈值边界")
    yellow_max: float = Field(..., description="黄色阈值边界")
    red_min: float = Field(..., description="红色起始阈值")
    is_inverse: bool = Field(
        default=False, description="True 表示值越低越好"
    )


class KPITrend(BaseModel):
    """KPI 趋势分析结果。

    基于滑动窗口对 KPI 采样数据进行趋势分析，
    输出方向、变化幅度与置信度。

    Attributes:
        kpi_name: KPI 标识名称
        direction: 趋势方向（上升/下降/稳定）
        change_percent: 变化百分比
        window_samples: 参与分析的样本数
        confidence: 置信度（0.0 - 1.0）
    """

    model_config = ConfigDict(extra="forbid")

    kpi_name: str = Field(..., description="KPI 标识名称")
    direction: TrendDirection = Field(..., description="趋势方向")
    change_percent: float = Field(..., description="变化百分比")
    window_samples: int = Field(..., description="参与分析的样本数")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="置信度（0.0 - 1.0）"
    )


class KPISummary(BaseModel):
    """KPI 汇总信息。

    汇集单项 KPI 的当前值、目标值、阈值、状态、趋势等
    全部关键信息，用于仪表盘展示与决策支持。

    Attributes:
        kpi_name: KPI 标识名称
        category: KPI 分类
        current_value: 当前最新值
        target_value: 目标值
        threshold_yellow: 黄色阈值
        threshold_red: 红色阈值
        status: 当前状态
        trend: 趋势信息
        samples_count: 采样总数
        last_updated: 最后更新时间
        description: KPI 描述
    """

    model_config = ConfigDict(extra="forbid")

    kpi_name: str = Field(..., description="KPI 标识名称")
    category: KPICategory = Field(..., description="KPI 分类")
    current_value: Optional[float] = Field(
        default=None, description="当前最新值"
    )
    target_value: float = Field(..., description="目标值")
    threshold_yellow: float = Field(..., description="黄色阈值")
    threshold_red: float = Field(..., description="红色阈值")
    status: KPIStatus = Field(
        default=KPIStatus.GREEN, description="当前状态"
    )
    trend: Optional[KPITrend] = Field(default=None, description="趋势信息")
    samples_count: int = Field(default=0, description="采样总数")
    last_updated: Optional[datetime] = Field(
        default=None, description="最后更新时间"
    )
    description: str = Field(default="", description="KPI 描述")


# ═══════════════════════════════════════════════════════════════════════════
# 常量与默认配置
# ═══════════════════════════════════════════════════════════════════════════

#: 趋势变化判定阈值（百分比），变化幅度小于此值视为稳定
_TREND_CHANGE_THRESHOLD: float = 5.0

#: 默认滑动窗口大小
_DEFAULT_WINDOW_SIZE: int = 100

#: 告警历史最大保留条数
_MAX_ALERT_HISTORY: int = 1000

#: KPI 状态到健康分的映射（借鉴 Google SRE SLI 评分）
_STATUS_SCORES: dict[KPIStatus, float] = {
    KPIStatus.GREEN: 100.0,
    KPIStatus.YELLOW: 50.0,
    KPIStatus.RED: 0.0,
}

#: 告警严重程度映射
_ALERT_SEVERITY: dict[KPIStatus, str] = {
    KPIStatus.RED: "critical",
    KPIStatus.YELLOW: "warning",
    KPIStatus.GREEN: "info",
}

#: KPI 健康分权重（总和为 1.0）
#: 安全类权重最高，信任类次之，效率与质量类均衡分配
_KPI_WEIGHTS: dict[str, float] = {
    "approval_response_time": 0.12,
    "auto_approval_rate": 0.12,
    "approval_rejection_rate": 0.15,
    "intervention_trigger_rate": 0.15,
    "cc1_integration_rate": 0.10,
    "fatigue_index": 0.12,
    "trust_evolution": 0.12,
    "batch_efficiency": 0.07,
    "correction_rate": 0.05,
}

#: KPI 默认定义注册表
#: 每项包含：分类、描述、目标值、默认阈值
_KPI_DEFINITIONS: dict[str, dict[str, Any]] = {
    "approval_response_time": {
        "category": KPICategory.EFFICIENCY,
        "description": "审批响应时间 —— 从请求到决策的平均时间（秒）",
        "target_value": 30.0,
        "threshold": KPIThreshold(
            kpi_name="approval_response_time",
            green_max=30.0,
            yellow_max=120.0,
            red_min=120.0,
            is_inverse=True,
        ),
    },
    "auto_approval_rate": {
        "category": KPICategory.EFFICIENCY,
        "description": "自动批准率 —— 自动批准操作占比（%）",
        "target_value": 60.0,
        "threshold": KPIThreshold(
            kpi_name="auto_approval_rate",
            green_max=60.0,
            yellow_max=30.0,
            red_min=30.0,
            is_inverse=False,
        ),
    },
    "approval_rejection_rate": {
        "category": KPICategory.SAFETY,
        "description": "审批拒绝率 —— 被拒绝操作占比（%）",
        "target_value": 5.0,
        "threshold": KPIThreshold(
            kpi_name="approval_rejection_rate",
            green_max=5.0,
            yellow_max=15.0,
            red_min=15.0,
            is_inverse=True,
        ),
    },
    "intervention_trigger_rate": {
        "category": KPICategory.SAFETY,
        "description": "干预触发率 —— 每会话 L4 干预次数",
        "target_value": 2.0,
        "threshold": KPIThreshold(
            kpi_name="intervention_trigger_rate",
            green_max=2.0,
            yellow_max=5.0,
            red_min=5.0,
            is_inverse=True,
        ),
    },
    "cc1_integration_rate": {
        "category": KPICategory.QUALITY,
        "description": "CC1 联动率 —— CC1 审查触发 L3/L4 占比（%）",
        "target_value": 80.0,
        "threshold": KPIThreshold(
            kpi_name="cc1_integration_rate",
            green_max=80.0,
            yellow_max=50.0,
            red_min=50.0,
            is_inverse=False,
        ),
    },
    "fatigue_index": {
        "category": KPICategory.TRUST,
        "description": "疲劳指数 —— 平均用户疲劳评分",
        "target_value": 25.0,
        "threshold": KPIThreshold(
            kpi_name="fatigue_index",
            green_max=25.0,
            yellow_max=50.0,
            red_min=50.0,
            is_inverse=True,
        ),
    },
    "trust_evolution": {
        "category": KPICategory.TRUST,
        "description": "信任度演进 —— 信任分数趋势",
        "target_value": 0.7,
        "threshold": KPIThreshold(
            kpi_name="trust_evolution",
            green_max=0.7,
            yellow_max=0.5,
            red_min=0.5,
            is_inverse=False,
        ),
    },
    "batch_efficiency": {
        "category": KPICategory.EFFICIENCY,
        "description": "批量审批效率 —— 批量审批与单独审批比率（%）",
        "target_value": 40.0,
        "threshold": KPIThreshold(
            kpi_name="batch_efficiency",
            green_max=40.0,
            yellow_max=20.0,
            red_min=20.0,
            is_inverse=False,
        ),
    },
    "correction_rate": {
        "category": KPICategory.QUALITY,
        "description": "纠错反馈率 —— 每 100 次操作的纠错次数",
        "target_value": 1.0,
        "threshold": KPIThreshold(
            kpi_name="correction_rate",
            green_max=1.0,
            yellow_max=3.0,
            red_min=3.0,
            is_inverse=True,
        ),
    },
}

#: 所有已注册 KPI 名称元组（按定义顺序）
KPI_NAMES: tuple[str, ...] = tuple(_KPI_DEFINITIONS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# KPI 指标引擎
# ═══════════════════════════════════════════════════════════════════════════


class KPIMetricsEngine:
    """KPI 指标引擎。

    线程安全的 KPI 指标追踪引擎，融合 NIST AI RMF Measure、
    Google SRE SLI/SLO、Datadog/Grafana 仪表盘与 Prometheus
    时间序列告警的设计理念。

    核心能力：
        - **采样记录**：接收来自 AntiFatigueManager、ApprovalWorkflowManager、
          RoutingEngine 等外部系统的 KPI 数据
        - **阈值检查**：基于交通灯模型实时判定 KPI 状态
        - **趋势分析**：滑动窗口趋势计算，输出方向、变化率与置信度
        - **动态阈值调整**：根据历史数据百分位数自动收紧阈值
        - **仪表盘数据**：生成包含整体健康分与告警的仪表盘数据包
        - **告警系统**：状态变更时自动生成告警，支持严重程度分级

    线程安全：
        所有公共方法通过 ``threading.RLock`` 保护，
        RLock 的可重入特性确保嵌套调用安全。

    用法示例::

        engine = KPIMetricsEngine()

        # 记录采样
        engine.record_sample("approval_response_time", 25.5,
                             context={"session_id": "s001"})

        # 获取汇总
        summary = engine.get_kpi_summary("approval_response_time")

        # 获取仪表盘数据
        dashboard = engine.get_dashboard_data()
    """

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        custom_thresholds: Optional[dict[str, KPIThreshold]] = None,
    ) -> None:
        """初始化 KPI 指标引擎。

        Args:
            window_size: 滑动窗口大小（默认 100 个样本）。
                趋势分析默认使用此窗口大小的采样数据。
            custom_thresholds: 自定义阈值配置，覆盖默认阈值。
                键为 KPI 名称，值为 ``KPIThreshold`` 对象。

        Raises:
            TypeError: 当 ``window_size`` 不是正整数时。
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise TypeError(
                f"window_size 必须为正整数，收到: {window_size!r}"
            )

        self._lock: threading.RLock = threading.RLock()
        self._window_size: int = window_size

        # 每项 KPI 的采样历史（deque 自动限长）
        self._samples: dict[str, deque[KPISample]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # 阈值配置（深拷贝默认定义，避免共享可变状态）
        self._thresholds: dict[str, KPIThreshold] = {}
        for kpi_name, definition in _KPI_DEFINITIONS.items():
            self._thresholds[kpi_name] = definition["threshold"].model_copy()

        # 应用自定义阈值
        if custom_thresholds:
            for kpi_name, threshold in custom_thresholds.items():
                self._thresholds[kpi_name] = threshold.model_copy()

        # 告警历史（状态变更记录）
        self._alert_history: deque[dict[str, Any]] = deque(
            maxlen=_MAX_ALERT_HISTORY
        )

        # 上次状态记录（用于检测状态变更）
        self._last_status: dict[str, KPIStatus] = {}

        logger.info(
            "KPI 指标引擎初始化完成 —— 追踪 %d 项 KPI，滑动窗口 %d",
            len(self._thresholds),
            window_size,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 采样记录
    # ─────────────────────────────────────────────────────────────────────

    def record_sample(
        self,
        kpi_name: str,
        value: float,
        context: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> KPISample:
        """记录一次 KPI 采样。

        将采样值追加到对应 KPI 的时间序列中，并立即执行阈值检查。
        如果状态发生变更（进入黄色或红色），自动生成告警记录。

        Args:
            kpi_name: KPI 标识名称，必须在已注册列表中。
            value: 采样值。
            context: 上下文信息（如会话 ID、用户 ID 等），可选。
            metadata: 元数据（如数据来源、置信度等），可选。

        Returns:
            创建的 :class:`KPISample` 对象。

        Raises:
            KeyError: 当 ``kpi_name`` 不在已注册的 KPI 列表中时。
        """
        with self._lock:
            if kpi_name not in self._thresholds:
                raise KeyError(
                    f"未知 KPI 名称: '{kpi_name}'，"
                    f"已注册的 KPI: {list(self._thresholds.keys())}"
                )

            sample = KPISample(
                kpi_name=kpi_name,
                value=value,
                context=context or {},
                metadata=metadata or {},
            )
            self._samples[kpi_name].append(sample)

            # 阈值检查并生成告警
            status = self._check_threshold_internal(kpi_name, value)
            self._check_and_generate_alert(kpi_name, value, status)

            logger.debug(
                "记录 KPI 采样: %s = %.4f (状态: %s)",
                kpi_name,
                value,
                status.value,
            )
            return sample

    # ─────────────────────────────────────────────────────────────────────
    # 阈值检查
    # ─────────────────────────────────────────────────────────────────────

    def check_threshold(self, kpi_name: str, value: float) -> KPIStatus:
        """检查给定值对应的 KPI 状态。

        根据当前阈值配置，判定 ``value`` 处于绿色、黄色还是红色区间。

        Args:
            kpi_name: KPI 标识名称。
            value: 待检查的值。

        Returns:
            :class:`KPIStatus` 枚举值。

        Raises:
            KeyError: 当 ``kpi_name`` 不存在时。
        """
        with self._lock:
            return self._check_threshold_internal(kpi_name, value)

    def _check_threshold_internal(
        self, kpi_name: str, value: float
    ) -> KPIStatus:
        """内部阈值检查（不加锁，供已持锁的方法调用）。

        判定规则：
            - 逆指标（``is_inverse=True``，值越低越好）：
              ``value <= green_max`` → 绿色
              ``green_max < value <= yellow_max`` → 黄色
              ``value > yellow_max`` → 红色
            - 正向指标（``is_inverse=False``，值越高越好）：
              ``value >= green_max`` → 绿色
              ``yellow_max <= value < green_max`` → 黄色
              ``value < yellow_max`` → 红色
        """
        threshold = self._thresholds.get(kpi_name)
        if threshold is None:
            raise KeyError(f"未知 KPI 名称: '{kpi_name}'")

        if threshold.is_inverse:
            if value <= threshold.green_max:
                return KPIStatus.GREEN
            elif value <= threshold.yellow_max:
                return KPIStatus.YELLOW
            else:
                return KPIStatus.RED
        else:
            if value >= threshold.green_max:
                return KPIStatus.GREEN
            elif value >= threshold.yellow_max:
                return KPIStatus.YELLOW
            else:
                return KPIStatus.RED

    def get_threshold(self, kpi_name: str) -> KPIThreshold:
        """获取指定 KPI 的当前阈值配置。

        Args:
            kpi_name: KPI 标识名称。

        Returns:
            ``KPIThreshold`` 对象的副本。

        Raises:
            KeyError: 当 ``kpi_name`` 不存在时。
        """
        with self._lock:
            if kpi_name not in self._thresholds:
                raise KeyError(f"未知 KPI 名称: '{kpi_name}'")
            return self._thresholds[kpi_name].model_copy()

    # ─────────────────────────────────────────────────────────────────────
    # 动态阈值调整
    # ─────────────────────────────────────────────────────────────────────

    def adjust_threshold(
        self,
        kpi_name: str,
        green_max: Optional[float] = None,
        yellow_max: Optional[float] = None,
    ) -> KPIThreshold:
        """动态调整 KPI 阈值。

        基于历史数据进行动态阈值调整。如果未提供具体阈值参数，
        则根据历史采样数据的百分位数自动计算建议阈值。

        **动态调整策略**（借鉴 SRE 错误预算渐进收紧理念）：

        - 当历史数据持续改善时，自动收紧阈值以提高标准
        - 逆指标（值越低越好）：使用 P75 / P95 百分位
        - 正向指标（值越高越好）：使用 P25 / P5 百分位
        - 单次调整幅度不超过当前阈值的 50%，防止剧烈波动

        Args:
            kpi_name: KPI 标识名称。
            green_max: 新的绿色阈值边界。``None`` 则自动计算。
            yellow_max: 新的黄色阈值边界。``None`` 则自动计算。

        Returns:
            更新后的 :class:`KPIThreshold` 对象。

        Raises:
            KeyError: 当 ``kpi_name`` 不存在时。
        """
        with self._lock:
            threshold = self._thresholds.get(kpi_name)
            if threshold is None:
                raise KeyError(f"未知 KPI 名称: '{kpi_name}'")

            old_green = threshold.green_max
            old_yellow = threshold.yellow_max

            # 未提供阈值时，根据历史数据自动计算
            if green_max is None or yellow_max is None:
                suggested_green, suggested_yellow = (
                    self._compute_suggested_threshold(kpi_name, threshold)
                )
                if green_max is None:
                    green_max = suggested_green
                if yellow_max is None:
                    yellow_max = suggested_yellow

            # 确保阈值关系正确
            if threshold.is_inverse:
                # 逆指标: green_max < yellow_max
                if yellow_max <= green_max:
                    yellow_max = green_max * 1.5
            else:
                # 正向指标: green_max > yellow_max
                if green_max <= yellow_max:
                    green_max = yellow_max * 1.5

            red_min = yellow_max

            new_threshold = KPIThreshold(
                kpi_name=kpi_name,
                green_max=green_max,
                yellow_max=yellow_max,
                red_min=red_min,
                is_inverse=threshold.is_inverse,
            )
            self._thresholds[kpi_name] = new_threshold

            logger.info(
                "KPI '%s' 阈值已调整: green %.2f→%.2f, "
                "yellow %.2f→%.2f",
                kpi_name,
                old_green,
                green_max,
                old_yellow,
                yellow_max,
            )
            return new_threshold

    def _compute_suggested_threshold(
        self, kpi_name: str, threshold: KPIThreshold
    ) -> tuple[float, float]:
        """根据历史数据计算建议阈值。

        使用百分位数法，基于历史采样分布计算建议阈值：

        - 逆指标（值越低越好）：
            - ``green_max`` = P75（75% 的数据在此以下，表现良好）
            - ``yellow_max`` = P95
        - 正向指标（值越高越好）：
            - ``green_max`` = P25（75% 的数据在此以上，表现良好）
            - ``yellow_max`` = P5

        安全约束：
            - 样本数不足（< 10）时返回当前阈值
            - 单次调整幅度限制在 ±50% 以内
            - 计算结果非正时回退到当前阈值

        Args:
            kpi_name: KPI 标识名称。
            threshold: 当前阈值配置。

        Returns:
            ``(suggested_green_max, suggested_yellow_max)`` 元组。
        """
        samples = list(self._samples.get(kpi_name, deque()))

        # 样本不足时，保持当前阈值
        if len(samples) < 10:
            return threshold.green_max, threshold.yellow_max

        values = sorted(s.value for s in samples)

        def _percentile(data: list[float], p: float) -> float:
            """计算百分位数（线性插值法）。"""
            k = (len(data) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            return data[f] + (data[c] - data[f]) * (k - f)

        if threshold.is_inverse:
            # 逆指标：值越低越好
            green_max = _percentile(values, 0.75)
            yellow_max = _percentile(values, 0.95)
        else:
            # 正向指标：值越高越好
            green_max = _percentile(values, 0.25)
            yellow_max = _percentile(values, 0.05)

        # 限制调整幅度（±50%）
        max_change = 0.5
        green_max = max(
            threshold.green_max * (1 - max_change),
            min(threshold.green_max * (1 + max_change), green_max),
        )
        yellow_max = max(
            threshold.yellow_max * (1 - max_change),
            min(threshold.yellow_max * (1 + max_change), yellow_max),
        )

        # 防止非正值
        if green_max <= 0:
            green_max = threshold.green_max
        if yellow_max <= 0:
            yellow_max = threshold.yellow_max

        return green_max, yellow_max

    def reset_thresholds(self) -> None:
        """重置所有 KPI 阈值为默认值。"""
        with self._lock:
            for kpi_name, definition in _KPI_DEFINITIONS.items():
                self._thresholds[kpi_name] = definition[
                    "threshold"
                ].model_copy()
            logger.info("所有 KPI 阈值已重置为默认值")

    # ─────────────────────────────────────────────────────────────────────
    # 趋势分析
    # ─────────────────────────────────────────────────────────────────────

    def compute_trend(
        self, kpi_name: str, window_size: int = _DEFAULT_WINDOW_SIZE
    ) -> KPITrend:
        """计算 KPI 趋势。

        使用滑动窗口分析趋势，将窗口内样本分为前半段和后半段，
        计算均值变化百分比以确定趋势方向。

        **算法步骤**：
            1. 取最近 ``window_size`` 个样本
            2. 将样本等分为前半段和后半段
            3. 分别计算两段均值
            4. 计算变化百分比 ``((后段均值 - 前段均值) / |前段均值|) * 100``
            5. 根据变化百分比与阈值判定方向
            6. 基于样本量和数据变异系数计算置信度

        Args:
            kpi_name: KPI 标识名称。
            window_size: 窗口大小（样本数），默认 100。

        Returns:
            :class:`KPITrend` 对象。样本不足时返回稳定趋势（置信度 0）。

        Raises:
            KeyError: 当 ``kpi_name`` 不存在时。
        """
        with self._lock:
            if kpi_name not in self._thresholds:
                raise KeyError(f"未知 KPI 名称: '{kpi_name}'")

            samples = list(self._samples.get(kpi_name, deque()))
            window_samples = min(window_size, len(samples))

            if window_samples < 2:
                return KPITrend(
                    kpi_name=kpi_name,
                    direction=TrendDirection.STABLE,
                    change_percent=0.0,
                    window_samples=window_samples,
                    confidence=0.0,
                )

            # 取最近 window_size 个样本
            window = samples[-window_samples:]
            mid = max(1, len(window) // 2)

            first_half = [s.value for s in window[:mid]]
            second_half = [s.value for s in window[mid:]]

            if not first_half or not second_half:
                return KPITrend(
                    kpi_name=kpi_name,
                    direction=TrendDirection.STABLE,
                    change_percent=0.0,
                    window_samples=window_samples,
                    confidence=0.0,
                )

            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            # 计算变化百分比
            if abs(first_avg) < 1e-10:
                change_percent = (
                    0.0 if abs(second_avg) < 1e-10 else 100.0
                )
            else:
                change_percent = (
                    (second_avg - first_avg) / abs(first_avg) * 100.0
                )

            # 判定趋势方向
            if abs(change_percent) < _TREND_CHANGE_THRESHOLD:
                direction = TrendDirection.STABLE
            elif change_percent > 0:
                direction = TrendDirection.UP
            else:
                direction = TrendDirection.DOWN

            # 计算置信度
            confidence = self._compute_trend_confidence(
                first_half, second_half, window_samples
            )

            return KPITrend(
                kpi_name=kpi_name,
                direction=direction,
                change_percent=round(change_percent, 2),
                window_samples=window_samples,
                confidence=round(confidence, 4),
            )

    def _compute_trend_confidence(
        self,
        first_half: list[float],
        second_half: list[float],
        total_samples: int,
    ) -> float:
        """计算趋势置信度。

        置信度由两个因子加权合成：
            1. **样本量因子**：样本越多越可靠（对数缩放，100 个样本时为 1.0）
            2. **方差因子**：变异系数越小越可靠

        最终置信度 = 0.5 * 样本量因子 + 0.5 * 方差因子

        Args:
            first_half: 前半段数据。
            second_half: 后半段数据。
            total_samples: 总样本数。

        Returns:
            置信度，范围 [0.0, 1.0]。
        """
        # 样本量因子（对数缩放）
        sample_factor = min(
            1.0, math.log10(max(total_samples, 1)) / math.log10(100)
        )

        # 方差因子（变异系数越小，置信度越高）
        all_values = first_half + second_half
        if len(all_values) < 2:
            return 0.0

        mean_val = statistics.mean(all_values)
        if abs(mean_val) < 1e-10:
            variance_factor = 0.5
        else:
            stdev = statistics.stdev(all_values)
            cv = stdev / abs(mean_val)  # 变异系数
            variance_factor = max(0.0, 1.0 - cv)

        confidence = 0.5 * sample_factor + 0.5 * variance_factor
        return max(0.0, min(1.0, confidence))

    # ─────────────────────────────────────────────────────────────────────
    # 汇总与查询
    # ─────────────────────────────────────────────────────────────────────

    def get_kpi_summary(self, kpi_name: str) -> KPISummary:
        """获取单个 KPI 的汇总信息。

        汇集当前值、目标值、阈值、状态、趋势等全部关键信息。

        Args:
            kpi_name: KPI 标识名称。

        Returns:
            :class:`KPISummary` 对象。无采样数据时 ``current_value`` 为
            ``None``，``status`` 为 ``GREEN``，``trend`` 为 ``None``。

        Raises:
            KeyError: 当 ``kpi_name`` 不存在时。
        """
        with self._lock:
            if kpi_name not in _KPI_DEFINITIONS:
                raise KeyError(f"未知 KPI 名称: '{kpi_name}'")

            definition = _KPI_DEFINITIONS[kpi_name]
            threshold = self._thresholds[kpi_name]
            samples = self._samples.get(kpi_name, deque())

            current_value = samples[-1].value if samples else None
            samples_count = len(samples)
            last_updated = samples[-1].timestamp if samples else None

            status = KPIStatus.GREEN
            if current_value is not None:
                status = self._check_threshold_internal(
                    kpi_name, current_value
                )

            trend = (
                self.compute_trend(kpi_name)
                if samples_count >= 2
                else None
            )

            return KPISummary(
                kpi_name=kpi_name,
                category=definition["category"],
                current_value=current_value,
                target_value=definition["target_value"],
                threshold_yellow=threshold.yellow_max,
                threshold_red=threshold.red_min,
                status=status,
                trend=trend,
                samples_count=samples_count,
                last_updated=last_updated,
                description=definition["description"],
            )

    def get_all_summaries(self) -> dict[str, KPISummary]:
        """获取所有 KPI 的汇总信息。

        Returns:
            KPI 名称到 :class:`KPISummary` 的字典。
        """
        with self._lock:
            return {
                kpi_name: self.get_kpi_summary(kpi_name)
                for kpi_name in _KPI_DEFINITIONS
            }

    def get_kpi_history(
        self, kpi_name: str, limit: int = 100
    ) -> list[KPISample]:
        """获取 KPI 历史采样记录。

        Args:
            kpi_name: KPI 标识名称。
            limit: 返回的最大记录数（从最近的开始）。
                设为 0 或负数则返回全部记录。

        Returns:
            :class:`KPISample` 列表，按时间正序排列（旧→新）。

        Raises:
            KeyError: 当 ``kpi_name`` 不存在时。
        """
        with self._lock:
            if kpi_name not in _KPI_DEFINITIONS:
                raise KeyError(f"未知 KPI 名称: '{kpi_name}'")

            samples = list(self._samples.get(kpi_name, deque()))
            if limit > 0:
                samples = samples[-limit:]
            return samples

    # ─────────────────────────────────────────────────────────────────────
    # 仪表盘
    # ─────────────────────────────────────────────────────────────────────

    def get_dashboard_data(self) -> dict[str, Any]:
        """获取仪表盘数据。

        生成包含所有 KPI 状态、整体健康分、分类得分、告警与统计信息的
        仪表盘数据包，支持 Datadog / Grafana 风格的可视化集成。

        整体健康分计算：
            按各 KPI 权重加权平均状态得分（绿色=100, 黄色=50, 红色=0），
            范围 [0, 100]。

        Returns:
            仪表盘数据字典，包含以下字段：

            - ``kpis``: 所有 KPI 的汇总信息（JSON 序列化格式）
            - ``overall_health_score``: 整体健康分（0-100）
            - ``category_scores``: 各分类平均得分
            - ``alerts``: 当前告警列表
            - ``statistics``: 统计信息
            - ``generated_at``: 生成时间（ISO 8601）
        """
        with self._lock:
            summaries = self.get_all_summaries()

            # 计算整体健康分与分类得分
            overall_score = 0.0
            category_scores: dict[str, list[float]] = defaultdict(list)

            for kpi_name, summary in summaries.items():
                weight = _KPI_WEIGHTS.get(kpi_name, 0.0)
                score = _STATUS_SCORES.get(summary.status, 0.0)
                overall_score += weight * score
                category_scores[summary.category.value].append(score)

            category_avg: dict[str, float] = {
                cat: round(statistics.mean(scores), 2) if scores else 0.0
                for cat, scores in category_scores.items()
            }

            # 从汇总生成当前告警
            alerts = self._build_alerts_from_summaries(summaries)

            # 统计信息
            stats = self._build_statistics_from_summaries(summaries)

            return {
                "kpis": {
                    name: summary.model_dump(mode="json")
                    for name, summary in summaries.items()
                },
                "overall_health_score": round(overall_score, 2),
                "category_scores": category_avg,
                "alerts": alerts,
                "statistics": stats,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    # ─────────────────────────────────────────────────────────────────────
    # 告警系统
    # ─────────────────────────────────────────────────────────────────────

    def get_alerts(self) -> list[dict[str, Any]]:
        """获取当前告警列表。

        返回所有当前处于黄色或红色状态的 KPI 告警，
        按严重程度排序（critical 在前）。

        Returns:
            告警字典列表，每个字典包含：

            - ``kpi_name``: KPI 名称
            - ``status``: 当前状态
            - ``value``: 当前值
            - ``threshold``: 阈值信息
            - ``severity``: 严重程度（``critical`` / ``warning``）
            - ``message``: 告警消息
            - ``timestamp``: 时间戳（ISO 8601）
        """
        with self._lock:
            summaries = self.get_all_summaries()
            return self._build_alerts_from_summaries(summaries)

    def get_alert_history(
        self, limit: int = 100
    ) -> list[dict[str, Any]]:
        """获取告警历史记录。

        返回状态变更时生成的告警历史记录，按时间倒序排列。

        Args:
            limit: 返回的最大记录数。

        Returns:
            告警字典列表（最新的在前）。
        """
        with self._lock:
            history = list(self._alert_history)
            if limit > 0:
                history = history[-limit:]
            history.reverse()
            return history

    def _build_alerts_from_summaries(
        self, summaries: dict[str, KPISummary]
    ) -> list[dict[str, Any]]:
        """从汇总信息构建当前告警列表（内部方法，不加锁）。"""
        alerts: list[dict[str, Any]] = []

        for kpi_name, summary in summaries.items():
            if summary.status == KPIStatus.GREEN:
                continue

            threshold = self._thresholds[kpi_name]
            severity = _ALERT_SEVERITY.get(summary.status, "info")
            message = self._generate_alert_message(
                kpi_name, summary, threshold
            )

            alerts.append(
                {
                    "kpi_name": kpi_name,
                    "status": summary.status.value,
                    "value": summary.current_value,
                    "threshold": threshold.model_dump(mode="json"),
                    "severity": severity,
                    "message": message,
                    "timestamp": summary.last_updated.isoformat()
                    if summary.last_updated
                    else datetime.now(timezone.utc).isoformat(),
                }
            )

        # 按严重程度排序：critical 在前
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda a: severity_order.get(a["severity"], 3))
        return alerts

    def _check_and_generate_alert(
        self,
        kpi_name: str,
        value: float,
        status: KPIStatus,
    ) -> None:
        """检查状态变更并生成告警记录（内部方法，不加锁）。

        当 KPI 状态从绿色变为黄色/红色，或从黄色变为红色时，
        生成一条告警记录并存入历史。
        """
        last_status = self._last_status.get(kpi_name)

        if status != KPIStatus.GREEN and status != last_status:
            threshold = self._thresholds[kpi_name]
            severity = _ALERT_SEVERITY.get(status, "info")

            alert = {
                "kpi_name": kpi_name,
                "status": status.value,
                "value": value,
                "threshold": threshold.model_dump(mode="json"),
                "severity": severity,
                "message": self._generate_alert_message_simple(
                    kpi_name, value, status, threshold
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "status_change",
                "previous_status": last_status.value
                if last_status
                else None,
            }
            self._alert_history.append(alert)

            logger.warning(
                "KPI 告警: %s = %.4f (状态: %s → %s, 严重度: %s)",
                kpi_name,
                value,
                last_status.value if last_status else "无",
                status.value,
                severity,
            )

        self._last_status[kpi_name] = status

    def _generate_alert_message(
        self,
        kpi_name: str,
        summary: KPISummary,
        threshold: KPIThreshold,
    ) -> str:
        """生成告警消息（详细版）。

        根据 KPI 状态和指标方向（逆指标/正向指标）生成精准的告警消息：
        - 黄色状态：值已越过绿色阈值边界
        - 红色状态：值已越过黄色阈值边界
        """
        # 提取 KPI 简称（描述中 "——" 之前的部分）
        desc = summary.description
        short_name = desc.split("——")[0].strip() if "——" in desc else desc
        value = summary.current_value

        if summary.status == KPIStatus.RED:
            # 红色：值已越过黄色阈值边界
            if threshold.is_inverse:
                boundary_desc = f"已超过黄色阈值 {threshold.yellow_max}"
            else:
                boundary_desc = f"已低于黄色阈值 {threshold.yellow_max}"
            return (
                f"[严重] {short_name} 当前值为 {value}，"
                f"{boundary_desc}，需要立即处理"
            )
        elif summary.status == KPIStatus.YELLOW:
            # 黄色：值已越过绿色阈值边界
            if threshold.is_inverse:
                boundary_desc = f"已超过绿色阈值 {threshold.green_max}"
            else:
                boundary_desc = f"已低于绿色阈值 {threshold.green_max}"
            return (
                f"[警告] {short_name} 当前值为 {value}，"
                f"{boundary_desc}，请关注"
            )
        else:
            return f"[正常] {short_name} 当前值为 {value}"

    def _generate_alert_message_simple(
        self,
        kpi_name: str,
        value: float,
        status: KPIStatus,
        threshold: KPIThreshold,
    ) -> str:
        """生成告警消息（简要版）。

        根据 KPI 状态和指标方向生成精准的告警描述。
        """
        if status == KPIStatus.RED:
            if threshold.is_inverse:
                return (
                    f"KPI '{kpi_name}' 值 {value:.4f} "
                    f"超过黄色阈值 {threshold.yellow_max}"
                )
            else:
                return (
                    f"KPI '{kpi_name}' 值 {value:.4f} "
                    f"低于黄色阈值 {threshold.yellow_max}"
                )
        elif status == KPIStatus.YELLOW:
            if threshold.is_inverse:
                return (
                    f"KPI '{kpi_name}' 值 {value:.4f} "
                    f"超过绿色阈值 {threshold.green_max}"
                )
            else:
                return (
                    f"KPI '{kpi_name}' 值 {value:.4f} "
                    f"低于绿色阈值 {threshold.green_max}"
                )
        else:
            return f"KPI '{kpi_name}' 恢复正常"

    # ─────────────────────────────────────────────────────────────────────
    # 统计信息
    # ─────────────────────────────────────────────────────────────────────

    def get_statistics(self) -> dict[str, Any]:
        """获取引擎统计信息。

        Returns:
            统计信息字典，包含：

            - ``total_samples``: 总采样数
            - ``kpi_sample_counts``: 各 KPI 采样数
            - ``status_distribution``: 状态分布（green/yellow/red 计数）
            - ``overall_health_score``: 整体健康分（0-100）
            - ``alert_count``: 告警历史总数
            - ``tracked_kpis``: 追踪的 KPI 数量
            - ``window_size``: 滑动窗口大小
        """
        with self._lock:
            summaries = self.get_all_summaries()
            return self._build_statistics_from_summaries(summaries)

    def _build_statistics_from_summaries(
        self, summaries: dict[str, KPISummary]
    ) -> dict[str, Any]:
        """从汇总信息构建统计信息（内部方法，不加锁）。"""
        total_samples = sum(s.samples_count for s in summaries.values())

        status_distribution: dict[str, int] = {
            KPIStatus.GREEN.value: 0,
            KPIStatus.YELLOW.value: 0,
            KPIStatus.RED.value: 0,
        }
        for summary in summaries.values():
            status_distribution[summary.status.value] += 1

        overall_score = 0.0
        for kpi_name, summary in summaries.items():
            weight = _KPI_WEIGHTS.get(kpi_name, 0.0)
            score = _STATUS_SCORES.get(summary.status, 0.0)
            overall_score += weight * score

        kpi_sample_counts = {
            name: summary.samples_count
            for name, summary in summaries.items()
        }

        return {
            "total_samples": total_samples,
            "kpi_sample_counts": kpi_sample_counts,
            "status_distribution": status_distribution,
            "overall_health_score": round(overall_score, 2),
            "alert_count": len(self._alert_history),
            "tracked_kpis": len(_KPI_DEFINITIONS),
            "window_size": self._window_size,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 外部系统集成接口
    # ─────────────────────────────────────────────────────────────────────

    def ingest_from_anti_fatigue(
        self,
        fatigue_score: float,
        context: Optional[dict[str, Any]] = None,
    ) -> KPISample:
        """从 AntiFatigueManager 接收疲劳数据。

        将用户疲劳评分记录为 ``fatigue_index`` KPI 采样。

        Args:
            fatigue_score: 疲劳评分（0-100，越高越疲劳）。
            context: 上下文信息。

        Returns:
            创建的 :class:`KPISample` 对象。
        """
        return self.record_sample(
            kpi_name="fatigue_index",
            value=fatigue_score,
            context=context or {},
            metadata={"source": "AntiFatigueManager"},
        )

    def ingest_from_approval_workflow(
        self,
        response_time: Optional[float] = None,
        auto_approved: Optional[bool] = None,
        rejected: Optional[bool] = None,
        is_batch: Optional[bool] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> list[KPISample]:
        """从 ApprovalWorkflowManager 接收审批工作流数据。

        根据提供的参数自动记录对应的 KPI 采样：

        - ``response_time`` → ``approval_response_time``
        - ``auto_approved`` → ``auto_approval_rate``（True=100, False=0）
        - ``rejected`` → ``approval_rejection_rate``（True=100, False=0）
        - ``is_batch`` → ``batch_efficiency``（True=100, False=0）

        Args:
            response_time: 审批响应时间（秒），可选。
            auto_approved: 是否自动批准，可选。
            rejected: 是否被拒绝，可选。
            is_batch: 是否批量审批，可选。
            context: 上下文信息。

        Returns:
            创建的 :class:`KPISample` 列表。
        """
        samples: list[KPISample] = []
        ctx = context or {}
        metadata = {"source": "ApprovalWorkflowManager"}

        if response_time is not None:
            samples.append(
                self.record_sample(
                    kpi_name="approval_response_time",
                    value=response_time,
                    context=ctx,
                    metadata=metadata,
                )
            )

        if auto_approved is not None:
            samples.append(
                self.record_sample(
                    kpi_name="auto_approval_rate",
                    value=100.0 if auto_approved else 0.0,
                    context=ctx,
                    metadata=metadata,
                )
            )

        if rejected is not None:
            samples.append(
                self.record_sample(
                    kpi_name="approval_rejection_rate",
                    value=100.0 if rejected else 0.0,
                    context=ctx,
                    metadata=metadata,
                )
            )

        if is_batch is not None:
            samples.append(
                self.record_sample(
                    kpi_name="batch_efficiency",
                    value=100.0 if is_batch else 0.0,
                    context=ctx,
                    metadata=metadata,
                )
            )

        return samples

    def ingest_from_routing_engine(
        self,
        intervention_count: Optional[int] = None,
        cc1_triggered: Optional[bool] = None,
        correction_count: Optional[int] = None,
        trust_score: Optional[float] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> list[KPISample]:
        """从 RoutingEngine 接收路由引擎数据。

        根据提供的参数自动记录对应的 KPI 采样：

        - ``intervention_count`` → ``intervention_trigger_rate``
        - ``cc1_triggered`` → ``cc1_integration_rate``（True=100, False=0）
        - ``correction_count`` → ``correction_rate``
        - ``trust_score`` → ``trust_evolution``

        Args:
            intervention_count: 每会话 L4 干预次数，可选。
            cc1_triggered: CC1 是否触发 L3/L4，可选。
            correction_count: 每 100 次操作的纠错次数，可选。
            trust_score: 信任分数（0.0-1.0），可选。
            context: 上下文信息。

        Returns:
            创建的 :class:`KPISample` 列表。
        """
        samples: list[KPISample] = []
        ctx = context or {}
        metadata = {"source": "RoutingEngine"}

        if intervention_count is not None:
            samples.append(
                self.record_sample(
                    kpi_name="intervention_trigger_rate",
                    value=float(intervention_count),
                    context=ctx,
                    metadata=metadata,
                )
            )

        if cc1_triggered is not None:
            samples.append(
                self.record_sample(
                    kpi_name="cc1_integration_rate",
                    value=100.0 if cc1_triggered else 0.0,
                    context=ctx,
                    metadata=metadata,
                )
            )

        if correction_count is not None:
            samples.append(
                self.record_sample(
                    kpi_name="correction_rate",
                    value=float(correction_count),
                    context=ctx,
                    metadata=metadata,
                )
            )

        if trust_score is not None:
            samples.append(
                self.record_sample(
                    kpi_name="trust_evolution",
                    value=trust_score,
                    context=ctx,
                    metadata=metadata,
                )
            )

        return samples

    # ─────────────────────────────────────────────────────────────────────
    # 清理与维护
    # ─────────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """清空所有采样数据、告警历史与状态记录。

        阈值配置不受影响。
        """
        with self._lock:
            self._samples.clear()
            self._alert_history.clear()
            self._last_status.clear()
            logger.info("KPI 指标引擎数据已清空（阈值配置保留）")

    def clear_kpi(self, kpi_name: str) -> None:
        """清空指定 KPI 的采样数据与状态记录。

        Args:
            kpi_name: KPI 标识名称。

        Raises:
            KeyError: 当 ``kpi_name`` 不存在时。
        """
        with self._lock:
            if kpi_name not in _KPI_DEFINITIONS:
                raise KeyError(f"未知 KPI 名称: '{kpi_name}'")
            if kpi_name in self._samples:
                self._samples[kpi_name].clear()
            self._last_status.pop(kpi_name, None)
            logger.info("KPI '%s' 采样数据已清空", kpi_name)

    # ─────────────────────────────────────────────────────────────────────
    # 属性与魔术方法
    # ─────────────────────────────────────────────────────────────────────

    @property
    def tracked_kpis(self) -> list[str]:
        """返回所有追踪的 KPI 名称列表。"""
        with self._lock:
            return list(_KPI_DEFINITIONS.keys())

    @property
    def window_size(self) -> int:
        """返回滑动窗口大小。"""
        return self._window_size

    def __repr__(self) -> str:
        """返回引擎的字符串表示。"""
        with self._lock:
            total = sum(len(d) for d in self._samples.values())
            return (
                f"KPIMetricsEngine("
                f"kpis={len(_KPI_DEFINITIONS)}, "
                f"window={self._window_size}, "
                f"samples={total})"
            )
