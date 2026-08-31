"""CC3 溯源捕获层 — KPI 指标引擎 (KPA Metrics Engine).

追踪 CC3 溯源捕获层的四类 KPI 指标, 提供实时监控与仪表盘导出:

1. 覆盖率 (Coverage):
   - annotation_coverage: 知识点 KPA 标注覆盖率 (目标: >95%)
   - dimension_fill_rate: 七维平均完整度 (目标: >80%)
   - source_diversity: 来源类型多样性比 (目标: >0.6)

2. 完整性 (Integrity):
   - hash_verification_rate: 标注哈希验证通过率 (目标: 100%)
   - chain_integrity_rate: 溯源链完整性率 (目标: 100%)
   - merkle_verification_rate: Merkle 证明验证率 (目标: 100%)

3. 性能 (Performance):
   - annotation_latency_ms: 标注创建平均延迟 (目标: <100ms)
   - chain_build_latency_ms: Merkle 树构建平均延迟 (目标: <500ms)
   - query_latency_ms: 查询平均延迟 (目标: <50ms)

4. 合规性 (Compliance):
   - doi_coverage: 期刊来源 DOI 覆盖率 (目标: >90%)
   - cc1_linkage_rate: CC1 评审关联率 (目标: >85%)
   - debate_coverage: 复杂决策 (复杂度31-65) 辩论日志覆盖率 (目标: >70%)

核心能力:
- 四类指标自动采集与状态评估 (pass/warning/fail)
- 指标汇总与通过率计算
- 仪表盘格式导出 (含分类详情与改进建议)
- 性能延迟追踪 (滑动窗口采样)

融合方案:
- Google SRE SLI/SLO: 服务水平指标与目标
- Prometheus: 指标采集与告警模型
- Grafana: 仪表盘可视化格式
- RFC 6962 Certificate Transparency: Merkle 证明完整性验证
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .models import KPACategory
from .kpa_engine import KPAEngine
from .debate_logger import DebateLogger
from .provenance_chain_builder import ProvenanceChainBuilder
from .ledger_integration import LedgerIntegration

logger = logging.getLogger(__name__)


# ============================================================
# 指标目标值定义
# ============================================================

# Coverage 覆盖率目标 (值越高越好)
TARGET_ANNOTATION_COVERAGE = 0.95   # 知识点标注覆盖率 >95%
TARGET_DIMENSION_FILL_RATE = 0.80   # 七维完整度 >80%
TARGET_SOURCE_DIVERSITY = 0.60      # 来源多样性 >0.6

# Integrity 完整性目标 (值越高越好)
TARGET_HASH_VERIFICATION = 1.0      # 哈希验证率 100%
TARGET_CHAIN_INTEGRITY = 1.0        # 链完整性率 100%
TARGET_MERKLE_VERIFICATION = 1.0    # Merkle 验证率 100%

# Performance 性能目标 (值越低越好)
TARGET_ANNOTATION_LATENCY_MS = 100.0   # 标注创建延迟 <100ms
TARGET_CHAIN_BUILD_LATENCY_MS = 500.0  # Merkle 树构建延迟 <500ms
TARGET_QUERY_LATENCY_MS = 50.0         # 查询延迟 <50ms

# Compliance 合规性目标 (值越高越好)
TARGET_DOI_COVERAGE = 0.90         # DOI 覆盖率 >90%
TARGET_CC1_LINKAGE = 0.85          # CC1 关联率 >85%
TARGET_DEBATE_COVERAGE = 0.70      # 辩论覆盖率 >70%

# 状态评估阈值
_WARNING_RATIO_HIGH = 0.90   # higher_is_better: 警告阈值 = 目标 × 0.90
_WARNING_RATIO_LOW = 1.20    # lower_is_better: 警告阈值 = 目标 × 1.20

# Dy3+ 领域已知来源类型数 (journal/textbook/preprint/experiment/database/internal)
_KNOWN_SOURCE_TYPES = 6

# 辩论触发复杂度范围
_DEBATE_COMPLEXITY_MIN = 31
_DEBATE_COMPLEXITY_MAX = 65

# 延迟采样滑动窗口大小
_LATENCY_WINDOW_SIZE = 1000


# ============================================================
# 数据模型
# ============================================================


@dataclass
class MetricSample:
    """单个指标采样数据.

    记录某一时刻某项指标的测量值、目标值及评估状态。

    Attributes:
        metric_name: 指标名称
        category: 指标分类 (KPACategory)
        value: 当前值
        target: 目标值
        unit: 单位 (ratio / ms)
        timestamp: 采样时间戳 (Unix epoch)
        status: 状态 ("pass" / "warning" / "fail")
    """

    metric_name: str
    category: KPACategory
    value: float
    target: float
    unit: str
    timestamp: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化的字典格式."""
        return {
            "metric_name": self.metric_name,
            "category": self.category.value,
            "value": round(self.value, 4),
            "target": self.target,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "status": self.status,
        }


@dataclass
class MetricsSummary:
    """指标分类汇总.

    对某一类别下所有指标采样的聚合统计。

    Attributes:
        category: 指标分类 (KPACategory)
        total_metrics: 总指标数
        passed: 通过 (pass) 指标数
        warnings: 警告 (warning) 指标数
        failed: 失败 (fail) 指标数
        pass_rate: 通过率 (0.0-1.0)
        samples: 该类别下所有指标采样列表
    """

    category: KPACategory
    total_metrics: int
    passed: int
    warnings: int
    failed: int
    pass_rate: float
    samples: list[MetricSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化的字典格式."""
        return {
            "category": self.category.value,
            "total_metrics": self.total_metrics,
            "passed": self.passed,
            "warnings": self.warnings,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "samples": [s.to_dict() for s in self.samples],
        }


# ============================================================
# KPI 指标引擎
# ============================================================


class KPAMetricsEngine:
    """KPI 指标引擎 — CC3 溯源捕获层指标追踪与监控.

    追踪四类 KPI 指标, 提供指标采集、状态评估、汇总统计与仪表盘导出:

    - 覆盖率 (Coverage): 标注覆盖、维度完整度、来源多样性
    - 完整性 (Integrity): 哈希验证、链完整性、Merkle 证明
    - 性能 (Performance): 标注延迟、链构建延迟、查询延迟
    - 合规性 (Compliance): DOI 覆盖、CC1 关联、辩论覆盖

    每个指标根据目标值自动评估状态:
    - pass: 达到目标
    - warning: 接近目标 (高优指标达 90%, 低优指标达 120%)
    - fail: 未达到目标

    性能指标依赖外部通过 record_annotation_latency /
    record_chain_build_latency / record_query_latency 方法记录的延迟数据。

    使用示例::

        kpa = KPAEngine()
        dl = DebateLogger()
        chain = ProvenanceChainBuilder()
        ledger = LedgerIntegration()

        metrics = KPAMetricsEngine(kpa, dl, chain, ledger)
        metrics.set_total_knowledge_points(500)

        # 记录性能延迟 (实际应用中在各操作后调用)
        metrics.record_annotation_latency(45.2)
        metrics.record_query_latency(12.8)

        # 采集所有指标
        summary = metrics.collect_all()
        print(f"通过率: {summary['overall_pass_rate']:.1%}")

        # 导出仪表盘数据
        dashboard = metrics.export_dashboard()
        for rec in dashboard["recommendations"]:
            print(rec)
    """

    def __init__(
        self,
        kpa_engine: KPAEngine,
        debate_logger: DebateLogger,
        chain_builder: ProvenanceChainBuilder,
        ledger: LedgerIntegration,
    ) -> None:
        """初始化 KPI 指标引擎.

        Args:
            kpa_engine: KPA 标注引擎
            debate_logger: 辩论日志引擎
            chain_builder: 溯源链构建器
            ledger: L0 Ledger 集成器
        """
        self._kpa = kpa_engine
        self._dl = debate_logger
        self._chain = chain_builder
        self._ledger = ledger

        # 性能延迟追踪 (滑动窗口)
        self._annotation_latencies: list[float] = []
        self._chain_build_latencies: list[float] = []
        self._query_latencies: list[float] = []

        # 知识点总数 (用于覆盖率计算, 需外部设置)
        self._total_knowledge_points: int = 0

    # ==========================================================
    # 状态评估
    # ==========================================================

    def get_metric_status(
        self,
        value: float,
        target: float,
        higher_is_better: bool = True,
    ) -> str:
        """评估指标状态 (pass / warning / fail).

        对于 higher_is_better=True (值越高越好, 如覆盖率、完整率):
        - pass: value >= target
        - warning: target × 0.9 <= value < target
        - fail: value < target × 0.9

        对于 higher_is_better=False (值越低越好, 如延迟):
        - pass: value <= target
        - warning: target < value <= target × 1.2
        - fail: value > target × 1.2

        Args:
            value: 当前值
            target: 目标值
            higher_is_better: 值越高是否越好 (默认 True)

        Returns:
            状态字符串: "pass" / "warning" / "fail"
        """
        if higher_is_better:
            if value >= target:
                return "pass"
            elif value >= target * _WARNING_RATIO_HIGH:
                return "warning"
            else:
                return "fail"
        else:
            if value <= target:
                return "pass"
            elif value <= target * _WARNING_RATIO_LOW:
                return "warning"
            else:
                return "fail"

    # ==========================================================
    # 覆盖率指标采集
    # ==========================================================

    def collect_coverage(self) -> list[MetricSample]:
        """收集覆盖率 (Coverage) 类指标.

        包含:
        - annotation_coverage: 知识点 KPA 标注覆盖率 (目标: >95%)
          覆盖率 = 有标注的知识点数 / 知识点总数
        - dimension_fill_rate: 七维平均完整度 (目标: >80%)
          取 KPA 引擎中所有标注的平均完整度
        - source_diversity: 来源类型多样性比 (目标: >0.6)
          已使用来源类型数 / 已知来源类型数 (6 种)

        Returns:
            覆盖率指标采样列表 (3 条)
        """
        now = time.time()
        samples: list[MetricSample] = []

        stats = self._kpa.statistics()
        total_targets = stats.get("total_targets", 0)

        # --- annotation_coverage ---
        if self._total_knowledge_points > 0:
            coverage = min(
                total_targets / self._total_knowledge_points, 1.0
            )
        elif total_targets > 0:
            # 未设置知识点总数, 假定已有标注目标均达标
            coverage = 1.0
        else:
            coverage = 0.0
        samples.append(MetricSample(
            metric_name="annotation_coverage",
            category=KPACategory.COVERAGE,
            value=coverage,
            target=TARGET_ANNOTATION_COVERAGE,
            unit="ratio",
            timestamp=now,
            status=self.get_metric_status(
                coverage, TARGET_ANNOTATION_COVERAGE,
            ),
        ))

        # --- dimension_fill_rate ---
        dim_fill_rate = stats.get("avg_completeness", 0.0)
        samples.append(MetricSample(
            metric_name="dimension_fill_rate",
            category=KPACategory.COVERAGE,
            value=dim_fill_rate,
            target=TARGET_DIMENSION_FILL_RATE,
            unit="ratio",
            timestamp=now,
            status=self.get_metric_status(
                dim_fill_rate, TARGET_DIMENSION_FILL_RATE,
            ),
        ))

        # --- source_diversity ---
        annotations = self._kpa.list_annotations(limit=10**9)
        source_types: set[str] = set()
        for ann in annotations:
            if ann.source.source_type:
                source_types.add(ann.source.source_type)
        source_diversity = (
            len(source_types) / _KNOWN_SOURCE_TYPES
            if _KNOWN_SOURCE_TYPES > 0
            else 0.0
        )
        source_diversity = min(source_diversity, 1.0)
        samples.append(MetricSample(
            metric_name="source_diversity",
            category=KPACategory.COVERAGE,
            value=source_diversity,
            target=TARGET_SOURCE_DIVERSITY,
            unit="ratio",
            timestamp=now,
            status=self.get_metric_status(
                source_diversity, TARGET_SOURCE_DIVERSITY,
            ),
        ))

        logger.debug(
            "覆盖率指标采集完成: coverage=%.4f, fill_rate=%.4f, "
            "diversity=%.4f",
            coverage, dim_fill_rate, source_diversity,
        )
        return samples

    # ==========================================================
    # 完整性指标采集
    # ==========================================================

    def collect_integrity(self) -> list[MetricSample]:
        """收集完整性 (Integrity) 类指标.

        包含:
        - hash_verification_rate: 标注哈希验证通过率 (目标: 100%)
          遍历所有标注, 检查 SHA-256 不可变哈希是否匹配
        - chain_integrity_rate: 溯源链完整性率 (目标: 100%)
          遍历所有溯源链, 检查哈希链接与时间戳单调性
        - merkle_verification_rate: Merkle 证明验证率 (目标: 100%)
          遍历所有链的所有节点, 验证 Merkle 包含证明

        无数据时 (无标注/无链) 默认返回 1.0 (pass),
        表示不存在完整性违规。

        Returns:
            完整性指标采样列表 (3 条)
        """
        now = time.time()
        samples: list[MetricSample] = []

        # --- hash_verification_rate ---
        annotations = self._kpa.list_annotations(limit=10**9)
        total_ann = len(annotations)
        hash_passed = 0
        for ann in annotations:
            try:
                if ann.verify_hash():
                    hash_passed += 1
            except Exception:
                pass
        hash_rate = hash_passed / total_ann if total_ann > 0 else 1.0
        samples.append(MetricSample(
            metric_name="hash_verification_rate",
            category=KPACategory.INTEGRITY,
            value=hash_rate,
            target=TARGET_HASH_VERIFICATION,
            unit="ratio",
            timestamp=now,
            status=self.get_metric_status(
                hash_rate, TARGET_HASH_VERIFICATION,
            ),
        ))

        # --- chain_integrity_rate ---
        chains = self._chain.list_chains()
        total_chains = len(chains)
        chains_passed = 0
        for meta in chains:
            cid = meta.get("chain_id", "")
            try:
                report = self._chain.verify_chain(cid)
                if report.get("all_passed", False):
                    chains_passed += 1
            except Exception:
                pass
        chain_rate = (
            chains_passed / total_chains if total_chains > 0 else 1.0
        )
        samples.append(MetricSample(
            metric_name="chain_integrity_rate",
            category=KPACategory.INTEGRITY,
            value=chain_rate,
            target=TARGET_CHAIN_INTEGRITY,
            unit="ratio",
            timestamp=now,
            status=self.get_metric_status(
                chain_rate, TARGET_CHAIN_INTEGRITY,
            ),
        ))

        # --- merkle_verification_rate ---
        total_proofs = 0
        proofs_passed = 0
        for meta in chains:
            cid = meta.get("chain_id", "")
            try:
                length = self._chain.get_chain_length(cid)
                for idx in range(length):
                    total_proofs += 1
                    if self._chain.verify_merkle_proof(cid, idx):
                        proofs_passed += 1
            except Exception:
                pass
        merkle_rate = (
            proofs_passed / total_proofs if total_proofs > 0 else 1.0
        )
        samples.append(MetricSample(
            metric_name="merkle_verification_rate",
            category=KPACategory.INTEGRITY,
            value=merkle_rate,
            target=TARGET_MERKLE_VERIFICATION,
            unit="ratio",
            timestamp=now,
            status=self.get_metric_status(
                merkle_rate, TARGET_MERKLE_VERIFICATION,
            ),
        ))

        logger.debug(
            "完整性指标采集完成: hash=%.4f, chain=%.4f, merkle=%.4f",
            hash_rate, chain_rate, merkle_rate,
        )
        return samples

    # ==========================================================
    # 性能指标采集
    # ==========================================================

    def collect_performance(self) -> list[MetricSample]:
        """收集性能 (Performance) 类指标.

        包含:
        - annotation_latency_ms: 标注创建平均延迟 (目标: <100ms)
        - chain_build_latency_ms: Merkle 树构建平均延迟 (目标: <500ms)
        - query_latency_ms: 查询平均延迟 (目标: <50ms)

        性能指标依赖 record_annotation_latency /
        record_chain_build_latency / record_query_latency 方法
        记录的延迟数据。若无记录则返回 0.0ms。

        Returns:
            性能指标采样列表 (3 条)
        """
        now = time.time()
        samples: list[MetricSample] = []

        # --- annotation_latency_ms ---
        avg_ann = self._avg_latency(self._annotation_latencies)
        samples.append(MetricSample(
            metric_name="annotation_latency_ms",
            category=KPACategory.PERFORMANCE,
            value=avg_ann,
            target=TARGET_ANNOTATION_LATENCY_MS,
            unit="ms",
            timestamp=now,
            status=self.get_metric_status(
                avg_ann, TARGET_ANNOTATION_LATENCY_MS,
                higher_is_better=False,
            ),
        ))

        # --- chain_build_latency_ms ---
        avg_chain = self._avg_latency(self._chain_build_latencies)
        samples.append(MetricSample(
            metric_name="chain_build_latency_ms",
            category=KPACategory.PERFORMANCE,
            value=avg_chain,
            target=TARGET_CHAIN_BUILD_LATENCY_MS,
            unit="ms",
            timestamp=now,
            status=self.get_metric_status(
                avg_chain, TARGET_CHAIN_BUILD_LATENCY_MS,
                higher_is_better=False,
            ),
        ))

        # --- query_latency_ms ---
        avg_query = self._avg_latency(self._query_latencies)
        samples.append(MetricSample(
            metric_name="query_latency_ms",
            category=KPACategory.PERFORMANCE,
            value=avg_query,
            target=TARGET_QUERY_LATENCY_MS,
            unit="ms",
            timestamp=now,
            status=self.get_metric_status(
                avg_query, TARGET_QUERY_LATENCY_MS,
                higher_is_better=False,
            ),
        ))

        logger.debug(
            "性能指标采集完成: ann=%.2fms, chain=%.2fms, "
            "query=%.2fms",
            avg_ann, avg_chain, avg_query,
        )
        return samples

    # ==========================================================
    # 合规性指标采集
    # ==========================================================

    def collect_compliance(self) -> list[MetricSample]:
        """收集合规性 (Compliance) 类指标.

        包含:
        - doi_coverage: 期刊来源 DOI 覆盖率 (目标: >90%)
          期刊来源中有 DOI 的比例
        - cc1_linkage_rate: CC1 评审关联率 (目标: >85%)
          标注中关联了 CC1 评审报告的比例
        - debate_coverage: 复杂决策辩论日志覆盖率 (目标: >70%)
          复杂度 31-65 的辩论中, 具有完整轮次与裁决的比例

        无数据时默认返回 1.0 (pass), 表示不存在合规违规。

        Returns:
            合规性指标采样列表 (3 条)
        """
        now = time.time()
        samples: list[MetricSample] = []

        annotations = self._kpa.list_annotations(limit=10**9)
        total_ann = len(annotations)

        # --- doi_coverage ---
        journal_total = 0
        journal_with_doi = 0
        for ann in annotations:
            if ann.source.source_type == "journal":
                journal_total += 1
                if ann.source.source_metadata.get("doi", ""):
                    journal_with_doi += 1
        doi_coverage = (
            journal_with_doi / journal_total
            if journal_total > 0
            else 1.0
        )
        samples.append(MetricSample(
            metric_name="doi_coverage",
            category=KPACategory.COMPLIANCE,
            value=doi_coverage,
            target=TARGET_DOI_COVERAGE,
            unit="ratio",
            timestamp=now,
            status=self.get_metric_status(
                doi_coverage, TARGET_DOI_COVERAGE,
            ),
        ))

        # --- cc1_linkage_rate ---
        cc1_linked = sum(
            1 for ann in annotations if ann.validation.cc1_review_id
        )
        cc1_rate = cc1_linked / total_ann if total_ann > 0 else 1.0
        samples.append(MetricSample(
            metric_name="cc1_linkage_rate",
            category=KPACategory.COMPLIANCE,
            value=cc1_rate,
            target=TARGET_CC1_LINKAGE,
            unit="ratio",
            timestamp=now,
            status=self.get_metric_status(
                cc1_rate, TARGET_CC1_LINKAGE,
            ),
        ))

        # --- debate_coverage ---
        # 复杂决策: 复杂度评分在 31-65 区间应触发辩论
        debate_logs = self._dl.list_logs(limit=10**9)
        complex_total = 0
        complex_covered = 0
        for log in debate_logs:
            complexity = log.pre_debate.complexity_score
            if _DEBATE_COMPLEXITY_MIN <= complexity <= _DEBATE_COMPLEXITY_MAX:
                complex_total += 1
                # 完整辩论日志: 有轮次记录且有裁决
                if (
                    len(log.rounds) > 0
                    and log.adjudicator_verdict is not None
                ):
                    complex_covered += 1
        debate_coverage = (
            complex_covered / complex_total
            if complex_total > 0
            else 1.0
        )
        samples.append(MetricSample(
            metric_name="debate_coverage",
            category=KPACategory.COMPLIANCE,
            value=debate_coverage,
            target=TARGET_DEBATE_COVERAGE,
            unit="ratio",
            timestamp=now,
            status=self.get_metric_status(
                debate_coverage, TARGET_DEBATE_COVERAGE,
            ),
        ))

        logger.debug(
            "合规性指标采集完成: doi=%.4f, cc1=%.4f, "
            "debate=%.4f",
            doi_coverage, cc1_rate, debate_coverage,
        )
        return samples

    # ==========================================================
    # 全量采集与汇总
    # ==========================================================

    def collect_all(self) -> dict[str, Any]:
        """采集全部四类指标并返回汇总.

        依次调用 collect_coverage / collect_integrity /
        collect_performance / collect_compliance, 聚合为汇总报告。

        Returns:
            汇总字典::

                {
                    "timestamp": float,
                    "categories": {
                        "coverage": MetricsSummary,
                        "integrity": MetricsSummary,
                        "performance": MetricsSummary,
                        "compliance": MetricsSummary,
                    },
                    "total_metrics": int,
                    "total_passed": int,
                    "total_warnings": int,
                    "total_failed": int,
                    "overall_pass_rate": float,
                }
        """
        now = time.time()

        coverage_samples = self.collect_coverage()
        integrity_samples = self.collect_integrity()
        performance_samples = self.collect_performance()
        compliance_samples = self.collect_compliance()

        category_map = [
            ("coverage", KPACategory.COVERAGE, coverage_samples),
            ("integrity", KPACategory.INTEGRITY, integrity_samples),
            ("performance", KPACategory.PERFORMANCE, performance_samples),
            ("compliance", KPACategory.COMPLIANCE, compliance_samples),
        ]

        summaries: dict[str, MetricsSummary] = {}
        for name, category, samples in category_map:
            summaries[name] = self._build_summary(category, samples)

        total_metrics = sum(s.total_metrics for s in summaries.values())
        total_passed = sum(s.passed for s in summaries.values())
        total_warnings = sum(s.warnings for s in summaries.values())
        total_failed = sum(s.failed for s in summaries.values())
        overall_pass_rate = (
            total_passed / total_metrics if total_metrics > 0 else 0.0
        )

        logger.info(
            "全量指标采集完成: total=%d, pass=%d, warn=%d, "
            "fail=%d, rate=%.2f%%",
            total_metrics, total_passed, total_warnings,
            total_failed, overall_pass_rate * 100,
        )

        return {
            "timestamp": now,
            "categories": summaries,
            "total_metrics": total_metrics,
            "total_passed": total_passed,
            "total_warnings": total_warnings,
            "total_failed": total_failed,
            "overall_pass_rate": round(overall_pass_rate, 4),
        }

    # ==========================================================
    # 仪表盘导出
    # ==========================================================

    def export_dashboard(self) -> dict[str, Any]:
        """导出仪表盘格式的指标数据.

        生成包含分类详情、汇总统计和改进建议的仪表盘数据,
        适合直接用于可视化展示 (如 Grafana 风格面板)。

        对每个 fail 指标生成 "未达标" 建议, 对每个 warning 指标
        生成 "接近临界" 建议, 便于运维人员快速定位问题。

        Returns:
            仪表盘数据字典::

                {
                    "generated_at": float,
                    "categories": [
                        {
                            "category": str,
                            "category_label": str,
                            "total_metrics": int,
                            "passed": int,
                            "warnings": int,
                            "failed": int,
                            "pass_rate": float,
                            "metrics": [
                                {name, value, target, unit, status, ...}
                            ],
                        },
                        ...
                    ],
                    "summary": {
                        "total_metrics": int,
                        "total_passed": int,
                        "total_warnings": int,
                        "total_failed": int,
                        "overall_pass_rate": float,
                    },
                    "recommendations": [str, ...],
                }
        """
        all_data = self.collect_all()
        now = time.time()

        categories_detail: list[dict[str, Any]] = []
        recommendations: list[str] = []

        for cat_name, summary in all_data["categories"].items():
            cat_detail = {
                "category": cat_name,
                "category_label": cat_name.upper(),
                "total_metrics": summary.total_metrics,
                "passed": summary.passed,
                "warnings": summary.warnings,
                "failed": summary.failed,
                "pass_rate": summary.pass_rate,
                "metrics": [s.to_dict() for s in summary.samples],
            }
            categories_detail.append(cat_detail)

            # 为未达标和接近临界的指标生成改进建议
            for sample in summary.samples:
                if sample.status == "fail":
                    recommendations.append(
                        f"[{cat_name.upper()}] "
                        f"{sample.metric_name} 未达标: "
                        f"当前={sample.value:.4f}{sample.unit}, "
                        f"目标={sample.target}{sample.unit}"
                    )
                elif sample.status == "warning":
                    recommendations.append(
                        f"[{cat_name.upper()}] "
                        f"{sample.metric_name} 接近临界: "
                        f"当前={sample.value:.4f}{sample.unit}, "
                        f"目标={sample.target}{sample.unit}"
                    )

        logger.info(
            "仪表盘导出完成: %d 个分类, %d 条建议",
            len(categories_detail), len(recommendations),
        )

        return {
            "generated_at": now,
            "categories": categories_detail,
            "summary": {
                "total_metrics": all_data["total_metrics"],
                "total_passed": all_data["total_passed"],
                "total_warnings": all_data["total_warnings"],
                "total_failed": all_data["total_failed"],
                "overall_pass_rate": all_data["overall_pass_rate"],
            },
            "recommendations": recommendations,
        }

    # ==========================================================
    # 性能延迟记录
    # ==========================================================

    def record_annotation_latency(self, latency_ms: float) -> None:
        """记录标注创建延迟 (毫秒).

        应在每次创建 KPA 标注后调用, 记录实际耗时。
        保留最近 _LATENCY_WINDOW_SIZE 条记录 (滑动窗口)。

        Args:
            latency_ms: 标注创建耗时 (毫秒)
        """
        self._annotation_latencies.append(latency_ms)
        if len(self._annotation_latencies) > _LATENCY_WINDOW_SIZE:
            self._annotation_latencies = (
                self._annotation_latencies[-_LATENCY_WINDOW_SIZE:]
            )

    def record_chain_build_latency(self, latency_ms: float) -> None:
        """记录 Merkle 树构建延迟 (毫秒).

        应在每次构建 Merkle 树后调用。

        Args:
            latency_ms: Merkle 树构建耗时 (毫秒)
        """
        self._chain_build_latencies.append(latency_ms)
        if len(self._chain_build_latencies) > _LATENCY_WINDOW_SIZE:
            self._chain_build_latencies = (
                self._chain_build_latencies[-_LATENCY_WINDOW_SIZE:]
            )

    def record_query_latency(self, latency_ms: float) -> None:
        """记录查询延迟 (毫秒).

        应在每次执行溯源查询后调用。

        Args:
            latency_ms: 查询耗时 (毫秒)
        """
        self._query_latencies.append(latency_ms)
        if len(self._query_latencies) > _LATENCY_WINDOW_SIZE:
            self._query_latencies = (
                self._query_latencies[-_LATENCY_WINDOW_SIZE:]
            )

    # ==========================================================
    # 配置
    # ==========================================================

    def set_total_knowledge_points(self, total: int) -> None:
        """设置知识点总数 (用于 annotation_coverage 计算).

        覆盖率 = 有 KPA 标注的知识点数 / 知识点总数。
        若不设置, 则假定所有已有标注目标均已覆盖 (覆盖率=100%)。

        Args:
            total: 知识点总数

        Raises:
            ValueError: 总数为负数
        """
        if total < 0:
            raise ValueError(f"知识点总数不能为负数: {total}")
        self._total_knowledge_points = total

    # ==========================================================
    # 内部辅助方法
    # ==========================================================

    @staticmethod
    def _avg_latency(latencies: list[float]) -> float:
        """计算平均延迟 (毫秒).

        Args:
            latencies: 延迟采样列表

        Returns:
            平均延迟; 列表为空时返回 0.0
        """
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    @staticmethod
    def _build_summary(
        category: KPACategory,
        samples: list[MetricSample],
    ) -> MetricsSummary:
        """从指标采样列表构建分类汇总.

        Args:
            category: 指标分类
            samples: 该分类下的指标采样列表

        Returns:
            MetricsSummary 汇总对象
        """
        total = len(samples)
        passed = sum(1 for s in samples if s.status == "pass")
        warnings = sum(1 for s in samples if s.status == "warning")
        failed = sum(1 for s in samples if s.status == "fail")
        pass_rate = passed / total if total > 0 else 0.0

        return MetricsSummary(
            category=category,
            total_metrics=total,
            passed=passed,
            warnings=warnings,
            failed=failed,
            pass_rate=round(pass_rate, 4),
            samples=samples,
        )

    # ==========================================================
    # 清空 (测试用)
    # ==========================================================

    def clear_latency_records(self) -> None:
        """清空所有延迟记录 (测试用)."""
        self._annotation_latencies.clear()
        self._chain_build_latencies.clear()
        self._query_latencies.clear()


__all__ = [
    "MetricSample",
    "MetricsSummary",
    "KPAMetricsEngine",
    # 指标目标值
    "TARGET_ANNOTATION_COVERAGE",
    "TARGET_DIMENSION_FILL_RATE",
    "TARGET_SOURCE_DIVERSITY",
    "TARGET_HASH_VERIFICATION",
    "TARGET_CHAIN_INTEGRITY",
    "TARGET_MERKLE_VERIFICATION",
    "TARGET_ANNOTATION_LATENCY_MS",
    "TARGET_CHAIN_BUILD_LATENCY_MS",
    "TARGET_QUERY_LATENCY_MS",
    "TARGET_DOI_COVERAGE",
    "TARGET_CC1_LINKAGE",
    "TARGET_DEBATE_COVERAGE",
]
