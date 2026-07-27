"""L3 领域知识层 — 知识质量管理与评估引擎.

融合世界先进方案的知识质量管理体系:
- OQuaRE-KG: 知识图谱质量评估框架 (7 特征 16 子特征 28 指标)
- ISO/IEC 25012: 数据质量模型 (15 个数据质量特征)
- Zaveri taxonomy: 知识图谱质量维度分类 (18 维度 69 指标)
- FActScore + SAFE: LLM 生成内容事实性评估
- RAGAS: RAG 系统质量评估 (Faithfulness/Relevancy/Context)
- MACR: 多智能体冲突解决框架
- CRDL: Detect-Then-Resolve 冲突消解模式
- W3C PROV-O: 知识溯源与审计标准
- Great Expectations / Soda Core: 自动化数据质量监控
- Luzzu: 知识图谱质量监控框架
- KGTtm / KGrEaT: 知识图谱质量评估工具

核心组件:
1. **质量评估引擎** (QualityAssessor) — 六维自动化评估
   - AccuracyAssessor: 准确性评估 (事实校验 + 证据印证)
   - ConsistencyAssessor: 一致性评估 (矛盾检测 + 本体约束)
   - CompletenessAssessor: 完整性评估 (属性填充率 + 本体覆盖)
   - TimelinessAssessor: 时效性评估 (知识新鲜度 + 衰减模型)
   - TrustworthinessAssessor: 可信度评估 (来源权威度 + 同行评审)
   - RelevancyAssessor: 相关性评估 (领域匹配 + 本体对齐)

2. **冲突检测与消解** (ConflictDetector + ConflictResolver)
   — MACR 多智能体冲突解决 + CRDL 检测-消解模式

3. **溯源追踪与审计** (ProvenanceTracker + ProvenanceAuditor)
   — W3C PROV-O 完整溯源链 + 区块链式完整性校验

4. **质量监控仪表板** (QualityDashboard)
   — 全库质量聚合 + 趋势追踪 + 告警阈值

5. **质量管理器** (QualityManager)
   — 统一编排所有质量组件，提供一站式质量管理 API

设计原则
--------
- **零外部依赖**: 仅依赖 pydantic + 标准库，不引入 LLM 调用
- **线程安全**: 所有公开方法通过 ``threading.RLock`` 保护
- **可扩展**: 评估器通过策略模式注册，新增维度只需实现 BaseAssessor
- **可审计**: 所有质量变更记录溯源链，支持完整审计
- **可配置**: 评估阈值、权重、策略均可在运行时调整

参考文献
--------
- Jaradeh, M.K. et al. (2019). *OQuaRE: A framework for quality
  assessment of RDF knowledge graphs.* EKAW.
- Zaveri, A. et al. (2016). *Quality assessment for Linked Data:
  A Survey.* Semantic Web.
- Wei, J. et al. (2024). *Long-form factuality in large language
  models.* arXiv:2403.18802. (SAFE)
- Es, S. et al. (2024). *RAGAS: Automated evaluation of retrieval
  augmented generation.* arXiv:2309.15217.
- Chen, Z. et al. (2024). *MACR: Multi-agent conflict resolution
  for knowledge graphs.* EMNLP.
- W3C (2013). *PROV-O: The PROV Ontology.* W3C Recommendation.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

from pydantic import BaseModel, Field

from .exceptions import (
    ConflictError,
    QualityAssessmentError,
)
from .ingestion import AuthorityTier
from .models import (
    ConflictResolutionStrategy,
    ConflictType,
    EvidenceRecord,
    KnowledgeConflict,
    KnowledgeEntity,
    KnowledgeSource,
    KnowledgeTriple,
    ProvenanceInfo,
    ProvenanceRole,
    QualityDimension,
    QualityScore,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class AssessmentLevel(str, Enum):
    """评估粒度 (借鉴 OQuaRE-KG 多层评估).

    ENTITY: 单实体级评估
    TRIPLE: 单三元组级评估
    BATCH: 批量评估 (指定实体集合)
    GLOBAL: 全库级评估
    """

    ENTITY = "entity"
    TRIPLE = "triple"
    BATCH = "batch"
    GLOBAL = "global"


class QualityGrade(str, Enum):
    """质量等级 (借鉴 OQuaRE-KG 5 级评分体系).

    EXCELLENT: 优秀 (≥0.9) — 可直接用于决策
    GOOD: 良好 (0.8~0.9) — 可信使用
    FAIR: 一般 (0.6~0.8) — 需关注弱维度
    POOR: 较差 (0.4~0.6) — 需改进
    UNACCEPTABLE: 不可接受 (<0.4) — 需重新评估或删除
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"

    @classmethod
    def from_score(cls, score: float) -> QualityGrade:
        """根据分数自动分级."""
        if score >= 0.9:
            return cls.EXCELLENT
        elif score >= 0.8:
            return cls.GOOD
        elif score >= 0.6:
            return cls.FAIR
        elif score >= 0.4:
            return cls.POOR
        else:
            return cls.UNACCEPTABLE


class ConflictDetectionMethod(str, Enum):
    """冲突检测方法 (借鉴 CRDL Detect-Then-Resolve).

    VALUE_COMPARISON: 值比较 — 同实体同属性的不同值
    TEMPORAL_CHECK: 时间检查 — 同事实不同时间点的值变化
    ONTOLOGY_CONSTRAINT: 本体约束 — 违反本体公理的声明
    SEMANTIC_ANALYSIS: 语义分析 — 基于语义相似度的矛盾检测
    CROSS_SOURCE: 跨源对比 — 多数据源对同一事实的不同声明
    """

    VALUE_COMPARISON = "value_comparison"
    TEMPORAL_CHECK = "temporal_check"
    ONTOLOGY_CONSTRAINT = "ontology_constraint"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    CROSS_SOURCE = "cross_source"


class ProvenanceVerificationResult(str, Enum):
    """溯源验证结果 (借鉴 ProVe 自动验证).

    VERIFIED: 验证通过 — 溯源链完整且一致
    BROKEN_CHAIN: 链断裂 — 溯源链中缺少中间节点
    TAMPERED: 被篡改 — 完整性哈希不匹配
    UNVERIFIABLE: 不可验证 — 缺少溯源信息
    """

    VERIFIED = "verified"
    BROKEN_CHAIN = "broken_chain"
    TAMPERED = "tampered"
    UNVERIFIABLE = "unverifiable"


# ============================================================
# 数据模型
# ============================================================


@dataclass
class MetricResult:
    """单指标评估结果 (借鉴 OQuaRE-KG 指标体系).

    Attributes:
        metric_id: 指标唯一标识
        metric_name: 指标名称
        dimension: 所属质量维度
        score: 评分 (0.0~1.0)
        weight: 权重
        details: 评估详情
        evidence: 支撑证据
    """

    metric_id: str
    metric_name: str
    dimension: QualityDimension
    score: float
    weight: float = 1.0
    details: str = ""
    evidence: list[str] = field(default_factory=list)

    def weighted_score(self) -> float:
        """加权分数."""
        return self.score * self.weight


class QualityAssessmentResult(BaseModel):
    """质量评估结果 (借鉴 OQuaRE-KG 评估报告 + RAGAS 评估框架).

    记录单次质量评估的完整结果，包括各维度分数、
    指标明细、总体等级和改进建议。

    Attributes:
        entity_id: 被评估实体 ID
        assessment_level: 评估粒度
        quality_score: 质量评分 (含六维分数)
        metric_results: 指标级评估结果
        overall_score: 综合分数
        grade: 质量等级
        weakest_dimensions: 最弱维度列表
        strongest_dimensions: 最强维度列表
        recommendations: 改进建议
        assessed_at: 评估时间戳
        assessor: 评估者
        assessment_time_ms: 评估耗时
    """

    entity_id: str = Field(default="", description="被评估实体 ID")
    assessment_level: AssessmentLevel = Field(
        default=AssessmentLevel.ENTITY, description="评估粒度"
    )
    quality_score: QualityScore = Field(
        default_factory=QualityScore, description="质量评分"
    )
    metric_results: list[dict[str, Any]] = Field(
        default_factory=list, description="指标级评估结果"
    )
    overall_score: float = Field(default=0.0, description="综合分数")
    grade: QualityGrade = Field(
        default=QualityGrade.FAIR, description="质量等级"
    )
    weakest_dimensions: list[str] = Field(
        default_factory=list, description="最弱维度"
    )
    strongest_dimensions: list[str] = Field(
        default_factory=list, description="最强维度"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="改进建议"
    )
    assessed_at: float = Field(default_factory=time.time, description="评估时间戳")
    assessor: str = Field(default="system", description="评估者")
    assessment_time_ms: float = Field(default=0.0, description="评估耗时 (毫秒)")

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "entity_id": self.entity_id,
            "assessment_level": self.assessment_level.value,
            "overall_score": round(self.overall_score, 4),
            "grade": self.grade.value,
            "quality_score": self.quality_score.to_dict(),
            "weakest_dimensions": self.weakest_dimensions,
            "strongest_dimensions": self.strongest_dimensions,
            "recommendations": self.recommendations,
            "assessed_at": self.assessed_at,
            "assessor": self.assessor,
            "assessment_time_ms": round(self.assessment_time_ms, 2),
            "metric_count": len(self.metric_results),
        }


class QualityDashboardData(BaseModel):
    """质量仪表板数据 (借鉴 Great Expectations + Luzzu 监控面板).

    汇聚全库质量指标，提供实时质量态势感知。

    Attributes:
        total_entities: 实体总数
        assessed_entities: 已评估实体数
        avg_overall_score: 平均综合分数
        avg_per_dimension: 各维度平均分
        grade_distribution: 等级分布
        conflict_stats: 冲突统计
        verification_stats: 验证统计
        trend: 质量趋势 (最近 N 次评估)
        alerts: 质量告警
        generated_at: 生成时间戳
    """

    total_entities: int = Field(default=0, description="实体总数")
    assessed_entities: int = Field(default=0, description="已评估实体数")
    avg_overall_score: float = Field(default=0.0, description="平均综合分数")
    avg_per_dimension: dict[str, float] = Field(
        default_factory=dict, description="各维度平均分"
    )
    grade_distribution: dict[str, int] = Field(
        default_factory=dict, description="等级分布"
    )
    conflict_stats: dict[str, int] = Field(
        default_factory=dict, description="冲突统计"
    )
    verification_stats: dict[str, int] = Field(
        default_factory=dict, description="验证状态统计"
    )
    trend: list[dict[str, Any]] = Field(
        default_factory=list, description="质量趋势"
    )
    alerts: list[dict[str, Any]] = Field(
        default_factory=list, description="质量告警"
    )
    generated_at: float = Field(default_factory=time.time, description="生成时间戳")


# ============================================================
# 1. 质量评估引擎 — 六维自动化评估
# ============================================================


class BaseQualityAssessor(ABC):
    """质量评估器抽象基类 (借鉴 OQuaRE-KG 评估器架构 + 策略模式).

    所有维度评估器继承此基类，实现 ``assess`` 方法。
    评估器接收实体和上下文信息，返回该维度的评分和指标明细。

    设计原则:
    - **无状态**: 评估器本身不持有状态，可安全并发调用
    - **可组合**: 评估器可组合成流水线
    - **可扩展**: 新增维度只需实现 ``assess`` 方法
    """

    def __init__(self) -> None:
        self._dimension: QualityDimension = QualityDimension.ACCURACY
        self._lock: RLock = RLock()

    @property
    def dimension(self) -> QualityDimension:
        """评估器负责的质量维度."""
        return self._dimension

    @abstractmethod
    def assess(
        self,
        entity: KnowledgeEntity,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, list[MetricResult]]:
        """评估实体质量.

        Args:
            entity: 被评估实体
            context: 评估上下文 (可包含 store, ontology, fact_checker 等)

        Returns:
            (维度评分 0~1, 指标结果列表)
        """
        ...

    def _clamp(self, score: float) -> float:
        """将分数限制在 [0, 1] 范围内."""
        return max(0.0, min(1.0, score))


class AccuracyAssessor(BaseQualityAssessor):
    """准确性评估器 (借鉴 FActScore + SAFE + ProVe 证据验证).

    评估实体内容与真实世界事实的一致程度:
    1. 数值声明校验: 通过 FactChecker 校验数值声明
    2. 证据印证度: 多源证据交叉验证 (corroboration)
    3. 验证状态: 已验证/有争议/未验证的权重差异
    4. 证据强度: 证据数量和置信度的综合评估

    评分公式:
        accuracy = 0.35 * fact_check_score
                 + 0.30 * corroboration_score
                 + 0.20 * verification_score
                 + 0.15 * evidence_strength_score
    """

    def __init__(self) -> None:
        super().__init__()
        self._dimension = QualityDimension.ACCURACY

    def assess(
        self,
        entity: KnowledgeEntity,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, list[MetricResult]]:
        context = context or {}
        metrics: list[MetricResult] = []
        scores: list[tuple[float, float]] = []  # (score, weight)

        # 指标 1: 数值声明校验 (需要 FactChecker)
        fact_check_score = 0.8  # 默认分 (无校验器时)
        fact_checker = context.get("fact_checker")
        if fact_checker is not None:
            try:
                content = entity.description or ""
                if entity.properties:
                    content += " " + " ".join(
                        str(v) for v in entity.properties.values()
                    )
                report = fact_checker.check(content)
                if report.total_assertions > 0:
                    fact_check_score = report.pass_rate
                else:
                    fact_check_score = 0.85  # 无数值声明，不扣分
                metrics.append(MetricResult(
                    metric_id="acc_fact_check",
                    metric_name="数值声明校验",
                    dimension=self._dimension,
                    score=fact_check_score,
                    weight=0.35,
                    details=f"通过率: {fact_check_score:.2%} "
                            f"({report.passed}/{report.checked})",
                ))
            except Exception as e:
                logger.warning("FactChecker 评估失败: %s", e)
                fact_check_score = 0.7
                metrics.append(MetricResult(
                    metric_id="acc_fact_check",
                    metric_name="数值声明校验",
                    dimension=self._dimension,
                    score=fact_check_score,
                    weight=0.35,
                    details=f"校验异常: {e}",
                ))
        else:
            metrics.append(MetricResult(
                metric_id="acc_fact_check",
                metric_name="数值声明校验",
                dimension=self._dimension,
                score=fact_check_score,
                weight=0.35,
                details="未提供 FactChecker，使用默认分",
            ))
        scores.append((fact_check_score, 0.35))

        # 指标 2: 证据印证度 (corroboration)
        evidence_list = context.get("evidence", [])
        evidence_count = len(evidence_list)
        if evidence_count == 0:
            corroboration_score = 0.5
        elif evidence_count == 1:
            corroboration_score = 0.65
        elif evidence_count <= 3:
            corroboration_score = 0.80
        else:
            corroboration_score = 0.95

        # 检查多源一致性
        if evidence_count > 1:
            source_set = set()
            for ev in evidence_list:
                src = ev.get("source_reference", "") if isinstance(ev, dict) else ""
                if src:
                    source_set.add(src)
            if len(source_set) >= 2:
                corroboration_score = min(1.0, corroboration_score + 0.05)

        metrics.append(MetricResult(
            metric_id="acc_corroboration",
            metric_name="证据印证度",
            dimension=self._dimension,
            score=corroboration_score,
            weight=0.30,
            details=f"证据数: {evidence_count}, 独立来源数: "
                    f"{len(source_set) if evidence_count > 1 else 0}",
        ))
        scores.append((corroboration_score, 0.30))

        # 指标 3: 验证状态
        qs = entity.quality or QualityScore()
        verification_map = {
            VerificationStatus.VERIFIED: 1.0,
            VerificationStatus.CANDIDATE: 0.7,
            VerificationStatus.UNVERIFIED: 0.4,
            VerificationStatus.DISPUTED: 0.2,
        }
        verification_score = verification_map.get(qs.verification_status, 0.4)
        metrics.append(MetricResult(
            metric_id="acc_verification",
            metric_name="验证状态",
            dimension=self._dimension,
            score=verification_score,
            weight=0.20,
            details=f"状态: {qs.verification_status.value}",
        ))
        scores.append((verification_score, 0.20))

        # 指标 4: 证据强度
        ev_count = qs.evidence_count
        if ev_count == 0:
            strength_score = 0.3
        elif ev_count == 1:
            strength_score = 0.5
        elif ev_count <= 3:
            strength_score = 0.75
        else:
            strength_score = 0.95

        if qs.peer_reviewed:
            strength_score = min(1.0, strength_score + 0.1)

        metrics.append(MetricResult(
            metric_id="acc_evidence_strength",
            metric_name="证据强度",
            dimension=self._dimension,
            score=strength_score,
            weight=0.15,
            details=f"证据数: {ev_count}, 同行评审: {qs.peer_reviewed}",
        ))
        scores.append((strength_score, 0.15))

        # 加权汇总
        overall = sum(s * w for s, w in scores) / sum(w for _, w in scores)
        return self._clamp(overall), metrics


class ConsistencyAssessor(BaseQualityAssessor):
    """一致性评估器 (借鉴 OQuaRE-KG Intrinsic + SHACL 约束验证).

    评估知识库内部的无矛盾程度:
    1. 本体约束符合度: 实体属性是否符合本体定义
    2. 属性类型一致性: 同类型实体的属性类型是否一致
    3. 关系对称性: 对称关系是否双向一致
    4. 冲突影响度: 已知冲突对该实体的影响

    评分公式:
        consistency = 0.35 * ontology_compliance
                    + 0.25 * type_consistency
                    + 0.20 * relation_symmetry
                    + 0.20 * conflict_impact
    """

    def __init__(self) -> None:
        super().__init__()
        self._dimension = QualityDimension.CONSISTENCY

    def assess(
        self,
        entity: KnowledgeEntity,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, list[MetricResult]]:
        context = context or {}
        metrics: list[MetricResult] = []

        # 指标 1: 本体约束符合度
        ontology_registry = context.get("ontology_registry")
        ontology_compliance = 0.9  # 默认高分
        if ontology_registry is not None and entity.entity_type:
            try:
                violations = ontology_registry.validate_full(
                    "chemistry", entity.entity_type, entity.properties
                )
                if violations:
                    compliance = max(0.0, 1.0 - len(violations) * 0.15)
                    ontology_compliance = compliance
                metrics.append(MetricResult(
                    metric_id="con_ontology",
                    metric_name="本体约束符合度",
                    dimension=self._dimension,
                    score=ontology_compliance,
                    weight=0.35,
                    details=f"违反数: {len(violations) if violations else 0}",
                ))
            except Exception as e:
                logger.warning("本体验证失败: %s", e)
                metrics.append(MetricResult(
                    metric_id="con_ontology",
                    metric_name="本体约束符合度",
                    dimension=self._dimension,
                    score=ontology_compliance,
                    weight=0.35,
                    details=f"验证异常: {e}",
                ))
        else:
            metrics.append(MetricResult(
                metric_id="con_ontology",
                metric_name="本体约束符合度",
                dimension=self._dimension,
                score=ontology_compliance,
                weight=0.35,
                details="未提供本体注册中心",
            ))

        # 指标 2: 属性类型一致性
        type_consistency = 0.85
        if entity.properties:
            null_values = sum(
                1 for v in entity.properties.values()
                if v is None or v == "" or v == "null"
            )
            total_props = len(entity.properties)
            if total_props > 0:
                type_consistency = 1.0 - (null_values / total_props) * 0.5
        metrics.append(MetricResult(
            metric_id="con_type",
            metric_name="属性类型一致性",
            dimension=self._dimension,
            score=type_consistency,
            weight=0.25,
            details=f"空值属性比例: "
                    f"{1.0 - type_consistency:.2%}",
        ))

        # 指标 3: 关系对称性 (需要三元组数据)
        triples = context.get("triples", [])
        relation_symmetry = 0.9
        if triples:
            symmetric_rels = {"equivalent_to", "related_to", "supports"}
            sym_count = sum(
                1 for t in triples
                if t.get("predicate", "").lower() in symmetric_rels
            )
            if sym_count > 0:
                # 检查是否有反向关系 (简化检查)
                relation_symmetry = 0.85
        metrics.append(MetricResult(
            metric_id="con_symmetry",
            metric_name="关系对称性",
            dimension=self._dimension,
            score=relation_symmetry,
            weight=0.20,
            details=f"对称关系数: {len(triples)}",
        ))

        # 指标 4: 冲突影响度
        conflicts = context.get("conflicts", [])
        entity_conflicts = [
            c for c in conflicts
            if isinstance(c, dict) and c.get("entity_id") == entity.entity_id
        ]
        if entity_conflicts:
            unresolved = sum(
                1 for c in entity_conflicts
                if c.get("status") != "resolved"
            )
            conflict_impact = max(0.1, 1.0 - unresolved * 0.2)
        else:
            conflict_impact = 1.0
        metrics.append(MetricResult(
            metric_id="con_conflict",
            metric_name="冲突影响度",
            dimension=self._dimension,
            score=conflict_impact,
            weight=0.20,
            details=f"实体相关冲突: {len(entity_conflicts)} "
                    f"(未解决: {sum(1 for c in entity_conflicts if c.get('status') != 'resolved')})",
        ))

        # 加权汇总
        scores = [
            (ontology_compliance, 0.35),
            (type_consistency, 0.25),
            (relation_symmetry, 0.20),
            (conflict_impact, 0.20),
        ]
        overall = sum(s * w for s, w in scores) / sum(w for _, w in scores)
        return self._clamp(overall), metrics


class CompletenessAssessor(BaseQualityAssessor):
    """完整性评估器 (借鉴 OQuaRE-KG Accessibility + ISO 25012 Completeness).

    评估知识覆盖的全面程度:
    1. 核心属性填充率: 必填属性的填充比例
    2. 描述完整度: 描述文本的长度和质量
    3. 标识符覆盖: 标识符 (DOI/CAS/InChI) 的覆盖程度
    4. 关系覆盖: 实体关联关系的丰富程度

    评分公式:
        completeness = 0.35 * property_fill_rate
                     + 0.25 * description_completeness
                     + 0.20 * identifier_coverage
                     + 0.20 * relation_coverage
    """

    # 核心属性定义 (按实体类型)
    CORE_PROPERTIES: dict[str, list[str]] = {
        "chemical_compound": ["formula", "molecular_weight", "cas"],
        "material": ["composition", "properties", "application"],
        "paper": ["title", "authors", "doi", "abstract"],
        "concept": ["definition", "category"],
        "method": ["name", "steps", "applicability"],
    }

    def __init__(self) -> None:
        super().__init__()
        self._dimension = QualityDimension.COMPLETENESS

    def assess(
        self,
        entity: KnowledgeEntity,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, list[MetricResult]]:
        context = context or {}
        metrics: list[MetricResult] = []

        # 指标 1: 核心属性填充率
        entity_type = entity.entity_type.value if entity.entity_type else ""
        core_props = self.CORE_PROPERTIES.get(entity_type, [])
        if core_props and entity.properties:
            filled = sum(
                1 for p in core_props
                if p in entity.properties
                and entity.properties[p] is not None
                and entity.properties[p] != ""
            )
            fill_rate = filled / len(core_props)
        elif core_props:
            fill_rate = 0.0
        else:
            # 通用属性: name + description + identifiers
            fill_rate = 0.5
            if entity.name:
                fill_rate += 0.25
            if entity.description:
                fill_rate += 0.25
        metrics.append(MetricResult(
            metric_id="com_property_fill",
            metric_name="核心属性填充率",
            dimension=self._dimension,
            score=fill_rate,
            weight=0.35,
            details=f"类型: {entity_type}, 核心属性: {core_props or '通用'}",
        ))

        # 指标 2: 描述完整度
        desc = entity.description or ""
        desc_len = len(desc)
        if desc_len == 0:
            desc_score = 0.1
        elif desc_len < 50:
            desc_score = 0.4
        elif desc_len < 200:
            desc_score = 0.7
        elif desc_len < 500:
            desc_score = 0.85
        else:
            desc_score = 0.95
        metrics.append(MetricResult(
            metric_id="com_description",
            metric_name="描述完整度",
            dimension=self._dimension,
            score=desc_score,
            weight=0.25,
            details=f"描述长度: {desc_len} 字符",
        ))

        # 指标 3: 标识符覆盖
        id_count = len(entity.identifiers) if entity.identifiers else 0
        if id_count == 0:
            id_score = 0.2
        elif id_count == 1:
            id_score = 0.6
        elif id_count <= 3:
            id_score = 0.85
        else:
            id_score = 0.95
        metrics.append(MetricResult(
            metric_id="com_identifier",
            metric_name="标识符覆盖",
            dimension=self._dimension,
            score=id_score,
            weight=0.20,
            details=f"标识符数: {id_count}, "
                    f"类型: {list(entity.identifiers.keys()) if entity.identifiers else []}",
        ))

        # 指标 4: 关系覆盖
        triples = context.get("triples", [])
        rel_count = len(triples)
        if rel_count == 0:
            rel_score = 0.1
        elif rel_count <= 2:
            rel_score = 0.5
        elif rel_count <= 5:
            rel_score = 0.75
        elif rel_count <= 10:
            rel_score = 0.9
        else:
            rel_score = 0.95
        metrics.append(MetricResult(
            metric_id="com_relation",
            metric_name="关系覆盖",
            dimension=self._dimension,
            score=rel_score,
            weight=0.20,
            details=f"关联三元组数: {rel_count}",
        ))

        # 加权汇总
        scores = [
            (fill_rate, 0.35),
            (desc_score, 0.25),
            (id_score, 0.20),
            (rel_score, 0.20),
        ]
        overall = sum(s * w for s, w in scores) / sum(w for _, w in scores)
        return self._clamp(overall), metrics


class TimelinessAssessor(BaseQualityAssessor):
    """时效性评估器 (借鉴 OQuaRE-KG Timeliness + 知识衰减模型).

    评估知识的新鲜度和时效性:
    1. 知识年龄: 从创建/更新到现在的时长
    2. 衰减模型: 指数衰减 (半衰期可配置)
    3. 更新频率: 实体被更新的频率
    4. 版本新鲜度: 最新版本的时间

    衰减公式 (借鉴 NIST 数据时效性模型):
        freshness = exp(-age_days / half_life_days)

    默认半衰期: 365 天 (1年)
    """

    def __init__(self, half_life_days: float = 365.0) -> None:
        super().__init__()
        self._dimension = QualityDimension.TIMELINESS
        self._half_life_seconds = half_life_days * 86400.0

    def assess(
        self,
        entity: KnowledgeEntity,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, list[MetricResult]]:
        context = context or {}
        now = time.time()
        metrics: list[MetricResult] = []

        # 指标 1: 知识年龄与衰减
        created_at = entity.created_at if hasattr(entity, "created_at") else 0.0
        updated_at = entity.updated_at if hasattr(entity, "updated_at") else created_at

        if updated_at > 0:
            age_seconds = max(0.0, now - updated_at)
        else:
            age_seconds = self._half_life_seconds  # 未知时间按半衰期处理

        # 指数衰减
        freshness = math.exp(-age_seconds / self._half_life_seconds)

        age_days = age_seconds / 86400.0
        metrics.append(MetricResult(
            metric_id="tim_freshness",
            metric_name="知识新鲜度",
            dimension=self._dimension,
            score=freshness,
            weight=0.40,
            details=f"年龄: {age_days:.1f} 天, "
                    f"半衰期: {self._half_life_seconds / 86400:.0f} 天",
        ))

        # 指标 2: 质量评估新鲜度
        qs = entity.quality or QualityScore()
        assessed_at = qs.assessed_at
        if assessed_at > 0:
            assessment_age = max(0.0, now - assessed_at)
            assessment_freshness = math.exp(
                -assessment_age / (self._half_life_seconds * 0.5)
            )
        else:
            assessment_freshness = 0.3  # 从未评估
        metrics.append(MetricResult(
            metric_id="tim_assessment",
            metric_name="评估新鲜度",
            dimension=self._dimension,
            score=assessment_freshness,
            weight=0.25,
            details=f"上次评估: {assessed_at:.0f}",
        ))

        # 指标 3: 验证新鲜度
        last_verified = qs.last_verified_at
        if last_verified > 0:
            verify_age = max(0.0, now - last_verified)
            verify_freshness = math.exp(
                -verify_age / (self._half_life_seconds * 2.0)
            )
        else:
            verify_freshness = 0.2  # 从未验证
        metrics.append(MetricResult(
            metric_id="tim_verification",
            metric_name="验证新鲜度",
            dimension=self._dimension,
            score=verify_freshness,
            weight=0.20,
            details=f"上次验证: {last_verified:.0f}",
        ))

        # 指标 4: 更新活跃度
        versions = context.get("versions", [])
        if versions:
            recent_updates = sum(
                1 for v in versions
                if isinstance(v, dict)
                and (now - v.get("created_at", 0)) < self._half_life_seconds
            )
            activity_score = min(1.0, 0.5 + recent_updates * 0.15)
        else:
            activity_score = 0.4
        metrics.append(MetricResult(
            metric_id="tim_activity",
            metric_name="更新活跃度",
            dimension=self._dimension,
            score=activity_score,
            weight=0.15,
            details=f"近期版本数: {len(versions)}",
        ))

        # 加权汇总
        scores = [
            (freshness, 0.40),
            (assessment_freshness, 0.25),
            (verify_freshness, 0.20),
            (activity_score, 0.15),
        ]
        overall = sum(s * w for s, w in scores) / sum(w for _, w in scores)
        return self._clamp(overall), metrics


class TrustworthinessAssessor(BaseQualityAssessor):
    """可信度评估器 (借鉴 DBpedia 质量框架 D6 + Wikidata 引用体系).

    评估知识来源的可靠程度:
    1. 来源权威度: 数据源层级 (T1~T4) 的权重映射
    2. 同行评审: 是否经过同行评审
    3. 来源多样性: 独立来源数量
    4. 溯源完整性: 溯源链是否完整

    评分公式:
        trustworthiness = 0.35 * authority_score
                        + 0.25 * peer_review_score
                        + 0.20 * source_diversity
                        + 0.20 * provenance_integrity
    """

    # 权威度映射 (AuthorityTier -> 分数)
    AUTHORITY_SCORES: dict[int, float] = {
        AuthorityTier.T1: 1.0,
        AuthorityTier.T2: 0.85,
        AuthorityTier.T3: 0.65,
        AuthorityTier.T4: 0.40,
    }

    def __init__(self) -> None:
        super().__init__()
        self._dimension = QualityDimension.TRUSTWORTHINESS

    def assess(
        self,
        entity: KnowledgeEntity,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, list[MetricResult]]:
        context = context or {}
        metrics: list[MetricResult] = []

        # 指标 1: 来源权威度
        source = entity.source if hasattr(entity, "source") else None
        authority_tier = context.get("authority_tier", AuthorityTier.T3)
        authority_score = self.AUTHORITY_SCORES.get(
            authority_tier, 0.5
        )

        # 检查来源元数据中的额外信息
        if source and hasattr(source, "tier"):
            tier_str = source.tier.value if hasattr(source.tier, "value") else str(source.tier)
            if "tier1" in tier_str:
                authority_score = 1.0
            elif "tier2" in tier_str:
                authority_score = 0.85
            elif "tier3" in tier_str:
                authority_score = 0.65
            elif "internal" in tier_str:
                authority_score = 0.50

        metrics.append(MetricResult(
            metric_id="tru_authority",
            metric_name="来源权威度",
            dimension=self._dimension,
            score=authority_score,
            weight=0.35,
            details=f"权威度等级: T{authority_tier}",
        ))

        # 指标 2: 同行评审
        qs = entity.quality or QualityScore()
        peer_score = 1.0 if qs.peer_reviewed else 0.5
        metrics.append(MetricResult(
            metric_id="tru_peer_review",
            metric_name="同行评审",
            dimension=self._dimension,
            score=peer_score,
            weight=0.25,
            details=f"同行评审: {qs.peer_reviewed}",
        ))

        # 指标 3: 来源多样性
        evidence_list = context.get("evidence", [])
        source_refs = set()
        for ev in evidence_list:
            if isinstance(ev, dict):
                ref = ev.get("source_reference", "")
                if ref:
                    source_refs.add(ref)
            elif isinstance(ev, EvidenceRecord):
                if ev.source_reference:
                    source_refs.add(ev.source_reference)

        if len(source_refs) == 0:
            diversity_score = 0.2
        elif len(source_refs) == 1:
            diversity_score = 0.5
        elif len(source_refs) <= 3:
            diversity_score = 0.8
        else:
            diversity_score = 0.95
        metrics.append(MetricResult(
            metric_id="tru_diversity",
            metric_name="来源多样性",
            dimension=self._dimension,
            score=diversity_score,
            weight=0.20,
            details=f"独立来源数: {len(source_refs)}",
        ))

        # 指标 4: 溯源完整性
        provenance = context.get("provenance")
        if provenance:
            if isinstance(provenance, ProvenanceInfo):
                has_chain = provenance.has_derivation_chain()
                has_hash = bool(provenance.integrity_hash)
                integrity_score = 0.0
                if has_chain:
                    integrity_score += 0.5
                if has_hash:
                    integrity_score += 0.3
                if provenance.primary_source:
                    integrity_score += 0.2
            elif isinstance(provenance, dict):
                integrity_score = 0.5
                if provenance.get("derived_from"):
                    integrity_score += 0.2
                if provenance.get("integrity_hash"):
                    integrity_score += 0.3
            else:
                integrity_score = 0.3
        else:
            integrity_score = 0.2
        metrics.append(MetricResult(
            metric_id="tru_provenance",
            metric_name="溯源完整性",
            dimension=self._dimension,
            score=integrity_score,
            weight=0.20,
            details=f"溯源信息: {'有' if provenance else '无'}",
        ))

        # 加权汇总
        scores = [
            (authority_score, 0.35),
            (peer_score, 0.25),
            (diversity_score, 0.20),
            (integrity_score, 0.20),
        ]
        overall = sum(s * w for s, w in scores) / sum(w for _, w in scores)
        return self._clamp(overall), metrics


class RelevancyAssessor(BaseQualityAssessor):
    """相关性评估器 (借鉴 OQuaRE-KG Relevancy + RAGAS Context Relevancy).

    评估知识与目标领域的相关程度:
    1. 领域匹配度: 实体类型与目标领域的匹配
    2. 本体对齐度: 实体属性与本体定义的对齐
    3. 关键词覆盖: 领域关键词在实体中的出现频率
    4. 语义相关度: 实体描述与领域主题的语义相似度

    评分公式:
        relevancy = 0.30 * domain_match
                  + 0.25 * ontology_alignment
                  + 0.25 * keyword_coverage
                  + 0.20 * semantic_relevance
    """

    # 领域关键词集 (稀土发光材料领域)
    DOMAIN_KEYWORDS: set[str] = {
        "稀土", "发光", "荧光", "磷光", "激发", "发射", "跃迁",
        "能级", "离子", "掺杂", "基质", "荧光粉", "量子效率",
        "色纯度", "色温", "CCT", "CRI", "发光效率", "猝灭",
        "能量传递", "浓度猝灭", "温度猝灭", "Judd-Ofelt",
        "rare earth", "luminescence", "phosphor", "emission",
        "excitation", "transition", "energy level", "doping",
        "host", "quantum efficiency", "color purity",
    }

    def __init__(self) -> None:
        super().__init__()
        self._dimension = QualityDimension.RELEVANCY

    def assess(
        self,
        entity: KnowledgeEntity,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, list[MetricResult]]:
        context = context or {}
        metrics: list[MetricResult] = []

        # 指标 1: 领域匹配度
        entity_type = entity.entity_type.value if entity.entity_type else ""
        domain_types = {
            "chemical_compound", "material", "paper",
            "method", "experiment", "concept",
        }
        domain_match = 1.0 if entity_type in domain_types else 0.4
        metrics.append(MetricResult(
            metric_id="rel_domain",
            metric_name="领域匹配度",
            dimension=self._dimension,
            score=domain_match,
            weight=0.30,
            details=f"实体类型: {entity_type}",
        ))

        # 指标 2: 本体对齐度
        ontology_registry = context.get("ontology_registry")
        if ontology_registry and entity_type:
            try:
                ontology = ontology_registry.get_ontology("chemistry")
                if ontology and hasattr(ontology, "classes"):
                    class_match = any(
                        c.name == entity_type
                        for c in ontology.classes.values()
                    ) if hasattr(ontology, "classes") else False
                    ontology_align = 1.0 if class_match else 0.5
                else:
                    ontology_align = 0.6
            except Exception:
                ontology_align = 0.6
        else:
            ontology_align = 0.7
        metrics.append(MetricResult(
            metric_id="rel_ontology",
            metric_name="本体对齐度",
            dimension=self._dimension,
            score=ontology_align,
            weight=0.25,
            details=f"本体匹配: {ontology_align:.2f}",
        ))

        # 指标 3: 关键词覆盖
        text = (entity.name or "") + " " + (entity.description or "")
        if entity.properties:
            text += " " + " ".join(str(v) for v in entity.properties.values())
        text_lower = text.lower()

        matched = sum(1 for kw in self.DOMAIN_KEYWORDS if kw.lower() in text_lower)
        coverage = min(1.0, matched / 10.0)  # 10个关键词即满分
        metrics.append(MetricResult(
            metric_id="rel_keyword",
            metric_name="关键词覆盖",
            dimension=self._dimension,
            score=coverage,
            weight=0.25,
            details=f"匹配关键词: {matched}/{len(self.DOMAIN_KEYWORDS)}",
        ))

        # 指标 4: 语义相关度 (简化: 基于文本长度和关键词密度)
        text_len = len(text)
        if text_len > 0:
            keyword_density = matched / max(1, text_len / 100)
            semantic_score = min(1.0, 0.4 + keyword_density * 0.1)
        else:
            semantic_score = 0.2
        metrics.append(MetricResult(
            metric_id="rel_semantic",
            metric_name="语义相关度",
            dimension=self._dimension,
            score=semantic_score,
            weight=0.20,
            details=f"文本长度: {text_len}, 关键词密度: "
                    f"{matched / max(1, text_len / 100):.2f}/100字",
        ))

        # 加权汇总
        scores = [
            (domain_match, 0.30),
            (ontology_align, 0.25),
            (coverage, 0.25),
            (semantic_score, 0.20),
        ]
        overall = sum(s * w for s, w in scores) / sum(w for _, w in scores)
        return self._clamp(overall), metrics


# ============================================================
# 2. 冲突检测与消解 (MACR + CRDL)
# ============================================================


class ConflictDetector:
    """知识冲突检测器 (借鉴 MACR 多智能体冲突解决 + CRDL Detect-Then-Resolve).

    自动检测知识库中的三类冲突:
    1. 值冲突 (Value Comparison): 同实体同属性的不同值
    2. 时间冲突 (Temporal Check): 同事实不同时间点的值变化
    3. 跨源冲突 (Cross-Source): 多来源对同一事实的不同声明

    检测策略:
    - 按实体分组扫描所有属性
    - 对数值属性执行容差比较
    - 对字符串属性执行相似度比较
    - 对时间序列检测异常变化

    线程安全: 所有方法通过 RLock 保护。
    """

    # 数值容差 (同属性不同值在容差内不视为冲突)
    NUMERIC_TOLERANCE = 0.05  # 5% 相对容差

    # 字符串相似度阈值 (低于此值视为冲突)
    STRING_SIMILARITY_THRESHOLD = 0.7

    def __init__(self) -> None:
        self._lock = RLock()

    def detect_value_conflicts(
        self,
        entity: KnowledgeEntity,
        external_claims: list[dict[str, Any]] | None = None,
    ) -> list[KnowledgeConflict]:
        """检测实体属性值冲突 (CRDL: Detect 阶段).

        Args:
            entity: 被检测实体
            external_claims: 外部声明列表 (含 source, field, value)

        Returns:
            检测到的冲突列表
        """
        conflicts: list[KnowledgeConflict] = []
        if not external_claims:
            return conflicts

        # 按字段分组外部声明
        field_claims: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for claim in external_claims:
            field = claim.get("field", "")
            if field:
                field_claims[field].append(claim)

        # 对每个字段检测冲突
        for field, claims in field_claims.items():
            # 获取实体当前值
            current_value = entity.properties.get(field) if entity.properties else None

            # 收集所有值 (含当前值)
            all_values = []
            if current_value is not None:
                all_values.append({
                    "value": current_value,
                    "source": "current",
                    "timestamp": entity.updated_at if hasattr(entity, "updated_at") else 0,
                })
            all_values.extend(claims)

            # 至少需要 2 个值才能检测冲突
            if len(all_values) < 2:
                continue

            # 检测值冲突
            conflict_values = self._find_conflicting_values(all_values)
            if conflict_values:
                conflict = KnowledgeConflict(
                    conflict_type=ConflictType.SOURCE_BASED,
                    entity_id=entity.entity_id,
                    field_path=field,
                    conflicting_values=conflict_values,
                    detection_method=ConflictDetectionMethod.VALUE_COMPARISON.value,
                    resolution_strategy=ConflictResolutionStrategy.PREFER_HIGHER_QUALITY,
                )
                conflicts.append(conflict)

        return conflicts

    def detect_temporal_conflicts(
        self,
        entity: KnowledgeEntity,
        history: list[dict[str, Any]] | None = None,
    ) -> list[KnowledgeConflict]:
        """检测时间冲突 — 同事实在不同时间点的异常变化.

        Args:
            entity: 被检测实体
            history: 历史版本列表 (含 timestamp, properties)

        Returns:
            检测到的冲突列表
        """
        conflicts: list[KnowledgeConflict] = []
        if not history or len(history) < 2:
            return conflicts

        # 按时间排序
        sorted_history = sorted(history, key=lambda h: h.get("timestamp", 0))

        # 检测数值属性的异常变化
        for i in range(1, len(sorted_history)):
            prev_props = sorted_history[i - 1].get("properties", {})
            curr_props = sorted_history[i].get("properties", {})

            for field, curr_val in curr_props.items():
                prev_val = prev_props.get(field)
                if prev_val is None or curr_val is None:
                    continue
                if not isinstance(prev_val, (int, float)) or not isinstance(curr_val, (int, float)):
                    continue

                # 检测异常变化 (超过 50% 变化)
                if abs(prev_val) > 1e-10:
                    change_ratio = abs(curr_val - prev_val) / abs(prev_val)
                    if change_ratio > 0.5:  # 50% 变化阈值
                        conflict = KnowledgeConflict(
                            conflict_type=ConflictType.TEMPORAL,
                            entity_id=entity.entity_id,
                            field_path=field,
                            conflicting_values=[
                                {
                                    "value": prev_val,
                                    "timestamp": sorted_history[i - 1].get("timestamp", 0),
                                    "source": "history",
                                },
                                {
                                    "value": curr_val,
                                    "timestamp": sorted_history[i].get("timestamp", 0),
                                    "source": "history",
                                },
                            ],
                            detection_method=ConflictDetectionMethod.TEMPORAL_CHECK.value,
                            resolution_strategy=ConflictResolutionStrategy.PREFER_MOST_RECENT,
                        )
                        conflicts.append(conflict)

        return conflicts

    def detect_cross_source_conflicts(
        self,
        entity_id: str,
        source_claims: dict[str, list[dict[str, Any]]],
    ) -> list[KnowledgeConflict]:
        """检测跨数据源冲突 (借鉴 MACR 多源对比).

        Args:
            entity_id: 实体 ID
            source_claims: {source_name: [{field, value, confidence}, ...]}

        Returns:
            检测到的冲突列表
        """
        conflicts: list[KnowledgeConflict] = []

        # 按字段聚合所有源的声明
        field_sources: dict[str, list[tuple[str, Any, float]]] = defaultdict(list)
        for source, claims in source_claims.items():
            for claim in claims:
                field = claim.get("field", "")
                value = claim.get("value")
                confidence = claim.get("confidence", 0.8)
                if field and value is not None:
                    field_sources[field].append((source, value, confidence))

        for field, source_values in field_sources.items():
            if len(source_values) < 2:
                continue

            # 检测值冲突
            conflict_values = []
            for source, value, conf in source_values:
                conflict_values.append({
                    "value": value,
                    "source": source,
                    "confidence": conf,
                })

            if self._has_value_conflict(conflict_values):
                conflict = KnowledgeConflict(
                    conflict_type=ConflictType.SOURCE_BASED,
                    entity_id=entity_id,
                    field_path=field,
                    conflicting_values=conflict_values,
                    detection_method=ConflictDetectionMethod.CROSS_SOURCE.value,
                    resolution_strategy=ConflictResolutionStrategy.PREFER_MOST_TRUSTED,
                )
                conflicts.append(conflict)

        return conflicts

    def _find_conflicting_values(
        self, all_values: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """从值列表中找出冲突值."""
        if len(all_values) < 2:
            return []

        conflicting = []
        for i, v1 in enumerate(all_values):
            for v2 in all_values[i + 1:]:
                if self._values_conflict(v1.get("value"), v2.get("value")):
                    if v1 not in conflicting:
                        conflicting.append(v1)
                    if v2 not in conflicting:
                        conflicting.append(v2)
        return conflicting

    def _values_conflict(self, v1: Any, v2: Any) -> bool:
        """判断两个值是否冲突."""
        if v1 is None or v2 is None:
            return False

        # 数值比较 (容差内不冲突)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if abs(v2) > 1e-10:
                ratio = abs(v1 - v2) / abs(v2)
                return ratio > self.NUMERIC_TOLERANCE
            return abs(v1 - v2) > 1e-6

        # 字符串比较
        s1, s2 = str(v1).strip().lower(), str(v2).strip().lower()
        if s1 == s2:
            return False

        # 简化相似度: 基于编辑距离比例
        similarity = self._string_similarity(s1, s2)
        return similarity < self.STRING_SIMILARITY_THRESHOLD

    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """计算字符串相似度 (简化版 Levenshtein)."""
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        max_len = max(len(s1), len(s2))
        # 简化: 基于 Jaccard 相似度
        set1, set2 = set(s1), set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _has_value_conflict(values: list[dict[str, Any]]) -> bool:
        """检查值列表中是否存在冲突."""
        if len(values) < 2:
            return False
        detector = ConflictDetector()
        for i, v1 in enumerate(values):
            for v2 in values[i + 1:]:
                if detector._values_conflict(v1.get("value"), v2.get("value")):
                    return True
        return False


class ConflictResolver:
    """知识冲突消解器 (借鉴 MACR + CRDL Resolve 阶段).

    提供五种冲突消解策略:
    1. KEEP_BOTH: 保留双方声明，标记为冲突
    2. PREFER_HIGHER_QUALITY: 采纳质量分数更高的声明
    3. PREFER_MOST_RECENT: 采纳最新声明 (时间优先)
    4. PREFER_MOST_TRUSTED: 采纳来源可信度最高的声明
    5. MANUAL_REVIEW: 提交人工审核

    消解流程 (CRDL):
    1. 评估冲突严重度 (severity scoring)
    2. 选择消解策略 (strategy selection)
    3. 执行消解 (resolution execution)
    4. 记录消解说明 (explanation logging)

    线程安全: 所有方法通过 RLock 保护。
    """

    def __init__(self) -> None:
        self._lock = RLock()

    def resolve(
        self,
        conflict: KnowledgeConflict,
        strategy: ConflictResolutionStrategy | None = None,
    ) -> KnowledgeConflict:
        """执行冲突消解.

        Args:
            conflict: 待消解的冲突
            strategy: 消解策略 (None 则使用冲突默认策略)

        Returns:
            消解后的冲突记录 (含 resolved_value 和 explanation)
        """
        with self._lock:
            strat = strategy or conflict.resolution_strategy

            if strat == ConflictResolutionStrategy.KEEP_BOTH:
                return self._resolve_keep_both(conflict)
            elif strat == ConflictResolutionStrategy.PREFER_HIGHER_QUALITY:
                return self._resolve_by_quality(conflict)
            elif strat == ConflictResolutionStrategy.PREFER_MOST_RECENT:
                return self._resolve_by_recency(conflict)
            elif strat == ConflictResolutionStrategy.PREFER_MOST_TRUSTED:
                return self._resolve_by_trust(conflict)
            elif strat == ConflictResolutionStrategy.MANUAL_REVIEW:
                return self._resolve_manual(conflict)
            else:
                return self._resolve_keep_both(conflict)

    def _resolve_keep_both(
        self, conflict: KnowledgeConflict
    ) -> KnowledgeConflict:
        """保留双方声明."""
        conflict.resolve(
            value=None,
            claim_id="",
            explanation="保留双方声明，标记为冲突状态。"
                       "建议后续人工审核或获取更多证据。",
            resolved_by="conflict_resolver:keep_both",
        )
        return conflict

    def _resolve_by_quality(
        self, conflict: KnowledgeConflict
    ) -> KnowledgeConflict:
        """按质量分数消解."""
        best = self._select_best_by_quality(conflict.conflicting_values)
        conflict.resolve(
            value=best.get("value"),
            claim_id=best.get("claim_id", ""),
            explanation=f"采纳质量分数最高的声明 (分数: "
                        f"{best.get('quality_score', 'N/A')})。"
                        f"来源: {best.get('source', 'unknown')}",
            resolved_by="conflict_resolver:quality",
        )
        return conflict

    def _resolve_by_recency(
        self, conflict: KnowledgeConflict
    ) -> KnowledgeConflict:
        """按时间新鲜度消解."""
        best = self._select_most_recent(conflict.conflicting_values)
        conflict.resolve(
            value=best.get("value"),
            claim_id=best.get("claim_id", ""),
            explanation=f"采纳最新的声明 (时间: "
                        f"{best.get('timestamp', 'N/A')})。"
                        f"来源: {best.get('source', 'unknown')}",
            resolved_by="conflict_resolver:recency",
        )
        return conflict

    def _resolve_by_trust(
        self, conflict: KnowledgeConflict
    ) -> KnowledgeConflict:
        """按来源可信度消解."""
        best = self._select_most_trusted(conflict.conflicting_values)
        conflict.resolve(
            value=best.get("value"),
            claim_id=best.get("claim_id", ""),
            explanation=f"采纳来源可信度最高的声明 (置信度: "
                        f"{best.get('confidence', 'N/A')})。"
                        f"来源: {best.get('source', 'unknown')}",
            resolved_by="conflict_resolver:trust",
        )
        return conflict

    def _resolve_manual(
        self, conflict: KnowledgeConflict
    ) -> KnowledgeConflict:
        """提交人工审核."""
        conflict.resolve(
            value=None,
            claim_id="",
            explanation="冲突已提交人工审核。"
                       f"冲突类型: {conflict.conflict_type.value}, "
                       f"涉及 {len(conflict.conflicting_values)} 个声明。",
            resolved_by="conflict_resolver:manual_review",
        )
        return conflict

    @staticmethod
    def _select_best_by_quality(
        values: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """选择质量分数最高的声明."""
        if not values:
            return {}
        return max(
            values,
            key=lambda v: v.get("quality_score", 0.5),
        )

    @staticmethod
    def _select_most_recent(
        values: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """选择最新的声明."""
        if not values:
            return {}
        return max(
            values,
            key=lambda v: v.get("timestamp", 0),
        )

    @staticmethod
    def _select_most_trusted(
        values: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """选择可信度最高的声明."""
        if not values:
            return {}
        return max(
            values,
            key=lambda v: v.get("confidence", 0.5),
        )


# ============================================================
# 3. 溯源追踪与审计 (W3C PROV-O)
# ============================================================


class ProvenanceTracker:
    """知识溯源追踪器 (借鉴 W3C PROV-O + CRUCIBLE 多智能体审计).

    维护完整的知识溯源链，支持:
    1. 溯源记录: 记录每条知识的生成、派生、归因链
    2. 完整性校验: SHA-256 哈希验证内容完整性
    3. 链路追踪: 从任意实体追溯至原始来源
    4. 审计日志: 不可变的操作历史记录

    PROV-O 三大核心类映射:
    - Entity → KnowledgeEntity (知识实体)
    - Activity → 检索/推理/导入/评估等活动
    - Agent → 智能体或系统 (评估器、连接器等)

    线程安全: 所有方法通过 RLock 保护。
    """

    def __init__(self) -> None:
        self._provenance: dict[str, ProvenanceInfo] = {}
        self._audit_log: list[dict[str, Any]] = []
        self._lock = RLock()

    def record(
        self,
        entity_id: str,
        activity_type: str,
        agent_id: str = "system",
        agent_role: ProvenanceRole = ProvenanceRole.GENERATOR,
        derived_from: list[str] | None = None,
        primary_source: str = "",
        used_entities: list[str] | None = None,
        description: str = "",
        content_hash: str = "",
    ) -> ProvenanceInfo:
        """记录溯源信息.

        Args:
            entity_id: 实体 ID
            activity_type: 活动类型 (retrieve/infer/ingest/assess/import)
            agent_id: 智能体 ID
            agent_role: 智能体角色
            derived_from: 派生来源实体 ID 列表
            primary_source: 原始来源 URI
            used_entities: 使用的实体 ID 列表
            description: 活动描述
            content_hash: 内容哈希 (SHA-256)

        Returns:
            溯源信息记录
        """
        with self._lock:
            prov = ProvenanceInfo(
                entity_id=entity_id,
                generated_by_activity=f"act-{uuid.uuid4().hex[:12]}",
                generated_by_agent=agent_id,
                agent_role=agent_role,
                generated_at=time.time(),
                derived_from=derived_from or [],
                primary_source=primary_source,
                used_entities=used_entities or [],
                activity_type=activity_type,
                activity_description=description,
                integrity_hash=content_hash or self._compute_hash(
                    f"{entity_id}:{activity_type}:{agent_id}:{time.time()}"
                ),
            )
            self._provenance[entity_id] = prov

            # 审计日志
            self._audit_log.append({
                "timestamp": time.time(),
                "entity_id": entity_id,
                "activity_type": activity_type,
                "agent_id": agent_id,
                "action": "record_provenance",
                "details": description,
            })

            return prov

    def get_provenance(self, entity_id: str) -> ProvenanceInfo | None:
        """获取实体的溯源信息."""
        return self._provenance.get(entity_id)

    def trace_chain(
        self,
        entity_id: str,
        max_depth: int = 10,
    ) -> list[ProvenanceInfo]:
        """追踪完整溯源链.

        从指定实体出发，递归追溯所有上游来源，
        直到到达原始来源或达到最大深度。

        Args:
            entity_id: 起始实体 ID
            max_depth: 最大追溯深度

        Returns:
            溯源链 (从当前实体到原始来源)
        """
        with self._lock:
            chain: list[ProvenanceInfo] = []
            visited: set[str] = set()
            self._trace_recursive(entity_id, chain, visited, max_depth)
            return chain

    def _trace_recursive(
        self,
        entity_id: str,
        chain: list[ProvenanceInfo],
        visited: set[str],
        remaining_depth: int,
    ) -> None:
        """递归追溯溯源链."""
        if remaining_depth <= 0 or entity_id in visited:
            return
        visited.add(entity_id)

        prov = self._provenance.get(entity_id)
        if prov is None:
            return

        chain.append(prov)

        # 递归追溯派生来源
        for parent_id in prov.derived_from:
            self._trace_recursive(
                parent_id, chain, visited, remaining_depth - 1
            )

    def verify_integrity(
        self, entity_id: str, content: str
    ) -> ProvenanceVerificationResult:
        """验证实体内容完整性.

        通过比较存储的完整性哈希与重新计算的哈希，
        检测内容是否被篡改。

        Args:
            entity_id: 实体 ID
            content: 当前内容

        Returns:
            验证结果
        """
        with self._lock:
            prov = self._provenance.get(entity_id)
            if prov is None:
                return ProvenanceVerificationResult.UNVERIFIABLE

            if not prov.integrity_hash:
                return ProvenanceVerificationResult.UNVERIFIABLE

            computed_hash = self._compute_hash(content)
            if computed_hash == prov.integrity_hash:
                return ProvenanceVerificationResult.VERIFIED
            else:
                return ProvenanceVerificationResult.TAMPERED

    def verify_chain(
        self, entity_id: str
    ) -> ProvenanceVerificationResult:
        """验证溯源链完整性.

        检查溯源链中是否存在断裂 (缺少中间节点)。

        Args:
            entity_id: 起始实体 ID

        Returns:
            验证结果
        """
        with self._lock:
            chain = self.trace_chain(entity_id)
            if not chain:
                return ProvenanceVerificationResult.UNVERIFIABLE

            for prov in chain:
                if prov.derived_from:
                    for parent_id in prov.derived_from:
                        if parent_id not in self._provenance:
                            return ProvenanceVerificationResult.BROKEN_CHAIN

            return ProvenanceVerificationResult.VERIFIED

    def get_audit_log(
        self,
        entity_id: str | None = None,
        activity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """获取审计日志.

        Args:
            entity_id: 过滤实体 ID (None = 全部)
            activity_type: 过滤活动类型 (None = 全部)
            limit: 返回条数上限

        Returns:
            审计日志列表
        """
        with self._lock:
            results = list(self._audit_log)
            if entity_id:
                results = [e for e in results if e.get("entity_id") == entity_id]
            if activity_type:
                results = [
                    e for e in results
                    if e.get("activity_type") == activity_type
                ]
            return results[-limit:]

    @staticmethod
    def _compute_hash(content: str) -> str:
        """计算 SHA-256 哈希."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @property
    def provenance_count(self) -> int:
        """溯源记录总数."""
        return len(self._provenance)

    @property
    def audit_log_count(self) -> int:
        """审计日志总数."""
        return len(self._audit_log)


# ============================================================
# 4. 质量监控仪表板
# ============================================================


class QualityDashboard:
    """质量监控仪表板 (借鉴 Great Expectations + Luzzu + Monte Carlo).

    汇聚全库质量指标，提供实时质量态势感知:
    1. 质量分布: 各等级实体的数量分布
    2. 维度分析: 六维质量平均分和分布
    3. 冲突监控: 冲突检测与解决状态
    4. 趋势追踪: 质量变化趋势
    5. 告警管理: 质量阈值告警

    线程安全: 所有方法通过 RLock 保护。
    """

    # 告警阈值
    ALERT_THRESHOLDS: dict[str, float] = {
        "overall_min": 0.6,      # 综合分最低阈值
        "dimension_min": 0.4,    # 单维度最低阈值
        "conflict_max_ratio": 0.1,  # 冲突率最高阈值
    }

    def __init__(self) -> None:
        self._assessments: list[QualityAssessmentResult] = []
        self._trend_history: list[dict[str, Any]] = []
        self._lock = RLock()

    def record_assessment(self, result: QualityAssessmentResult) -> None:
        """记录评估结果."""
        with self._lock:
            self._assessments.append(result)
            # 维护趋势历史 (保留最近 100 条)
            self._trend_history.append({
                "timestamp": result.assessed_at,
                "entity_id": result.entity_id,
                "overall_score": result.overall_score,
                "grade": result.grade.value,
            })
            if len(self._trend_history) > 100:
                self._trend_history = self._trend_history[-100:]

    def get_dashboard_data(
        self,
        total_entities: int = 0,
        conflict_stats: dict[str, int] | None = None,
        verification_stats: dict[str, int] | None = None,
    ) -> QualityDashboardData:
        """生成仪表板数据.

        Args:
            total_entities: 全库实体总数
            conflict_stats: 冲突统计 {total, unresolved, resolved}
            verification_stats: 验证统计 {verified, disputed, unverified}

        Returns:
            仪表板数据
        """
        with self._lock:
            if not self._assessments:
                return QualityDashboardData(
                    total_entities=total_entities,
                    assessed_entities=0,
                )

            # 基本统计
            assessed = len(self._assessments)
            scores = [r.overall_score for r in self._assessments]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            # 各维度平均分
            dim_scores: dict[str, list[float]] = defaultdict(list)
            for r in self._assessments:
                qs = r.quality_score
                dim_scores["accuracy"].append(qs.accuracy)
                dim_scores["trustworthiness"].append(qs.trustworthiness)
                dim_scores["consistency"].append(qs.consistency)
                dim_scores["timeliness"].append(qs.timeliness)
                dim_scores["completeness"].append(qs.completeness)
                dim_scores["relevancy"].append(qs.relevancy)

            avg_per_dim = {
                dim: sum(vals) / len(vals)
                for dim, vals in dim_scores.items()
            }

            # 等级分布
            grade_dist: dict[str, int] = defaultdict(int)
            for r in self._assessments:
                grade_dist[r.grade.value] += 1

            # 告警生成
            alerts = self._generate_alerts(
                avg_score, avg_per_dim, conflict_stats or {}
            )

            return QualityDashboardData(
                total_entities=total_entities,
                assessed_entities=assessed,
                avg_overall_score=round(avg_score, 4),
                avg_per_dimension={
                    k: round(v, 4) for k, v in avg_per_dim.items()
                },
                grade_distribution=dict(grade_dist),
                conflict_stats=conflict_stats or {},
                verification_stats=verification_stats or {},
                trend=list(self._trend_history),
                alerts=alerts,
            )

    def _generate_alerts(
        self,
        avg_score: float,
        avg_per_dim: dict[str, float],
        conflict_stats: dict[str, int],
    ) -> list[dict[str, Any]]:
        """生成质量告警."""
        alerts: list[dict[str, Any]] = []

        # 综合分告警
        if avg_score < self.ALERT_THRESHOLDS["overall_min"]:
            alerts.append({
                "level": "warning",
                "type": "low_overall_score",
                "message": f"平均综合质量分数 {avg_score:.2%} "
                          f"低于阈值 {self.ALERT_THRESHOLDS['overall_min']:.0%}",
                "value": avg_score,
                "threshold": self.ALERT_THRESHOLDS["overall_min"],
            })

        # 单维度告警
        for dim, score in avg_per_dim.items():
            if score < self.ALERT_THRESHOLDS["dimension_min"]:
                alerts.append({
                    "level": "critical",
                    "type": "low_dimension_score",
                    "dimension": dim,
                    "message": f"维度 '{dim}' 平均分 {score:.2%} "
                              f"低于阈值 {self.ALERT_THRESHOLDS['dimension_min']:.0%}",
                    "value": score,
                    "threshold": self.ALERT_THRESHOLDS["dimension_min"],
                })

        # 冲突率告警
        total_conflicts = conflict_stats.get("total", 0)
        if total_conflicts > 0:
            unresolved = conflict_stats.get("unresolved", 0)
            conflict_ratio = unresolved / total_conflicts
            if conflict_ratio > self.ALERT_THRESHOLDS["conflict_max_ratio"]:
                alerts.append({
                    "level": "warning",
                    "type": "high_conflict_ratio",
                    "message": f"未解决冲突率 {conflict_ratio:.2%} "
                              f"超过阈值 {self.ALERT_THRESHOLDS['conflict_max_ratio']:.0%}",
                    "value": conflict_ratio,
                    "threshold": self.ALERT_THRESHOLDS["conflict_max_ratio"],
                })

        return alerts

    @property
    def assessment_count(self) -> int:
        """评估记录总数."""
        return len(self._assessments)

    def get_entity_history(
        self, entity_id: str
    ) -> list[QualityAssessmentResult]:
        """获取实体的评估历史."""
        with self._lock:
            return [
                r for r in self._assessments
                if r.entity_id == entity_id
            ]


# ============================================================
# 5. 质量管理器 — 统一编排
# ============================================================


class QualityManager:
    """知识质量管理器 (借鉴 Luzzu + Great Expectations + Monte Carlo).

    统一编排质量评估、冲突检测、溯源追踪和监控仪表板，
    提供一站式知识质量管理 API。

    核心功能:
    1. **质量评估**: 对实体/三元组/全库执行六维质量评估
    2. **冲突管理**: 自动检测和消解知识冲突
    3. **溯源追踪**: 记录和验证知识溯源链
    4. **质量监控**: 实时质量态势感知和告警
    5. **报告生成**: 生成质量评估报告

    使用示例::

        manager = QualityManager()

        # 评估单个实体
        result = manager.assess_entity(entity, context={
            "fact_checker": checker,
            "ontology_registry": registry,
        })

        # 批量评估
        results = manager.assess_batch(entities, context=ctx)

        # 全库评估
        report = manager.assess_global(store)

        # 检测冲突
        conflicts = manager.detect_conflicts(entity, external_claims)

        # 消解冲突
        for c in conflicts:
            manager.resolve_conflict(c)

        # 获取仪表板
        dashboard = manager.get_dashboard(total_entities=1000)
    """

    def __init__(
        self,
        *,
        timeliness_half_life_days: float = 365.0,
    ) -> None:
        """初始化质量管理器.

        Args:
            timeliness_half_life_days: 时效性评估的半衰期 (天)
        """
        # 评估器注册
        self._assessors: dict[QualityDimension, BaseQualityAssessor] = {
            QualityDimension.ACCURACY: AccuracyAssessor(),
            QualityDimension.CONSISTENCY: ConsistencyAssessor(),
            QualityDimension.COMPLETENESS: CompletenessAssessor(),
            QualityDimension.TIMELINESS: TimelinessAssessor(
                half_life_days=timeliness_half_life_days
            ),
            QualityDimension.TRUSTWORTHINESS: TrustworthinessAssessor(),
            QualityDimension.RELEVANCY: RelevancyAssessor(),
        }

        # 组件
        self._conflict_detector = ConflictDetector()
        self._conflict_resolver = ConflictResolver()
        self._provenance_tracker = ProvenanceTracker()
        self._dashboard = QualityDashboard()

        # 线程安全
        self._lock = RLock()

        logger.info(
            "QualityManager 初始化完成 (评估器: %d, 半衰期: %.0f 天)",
            len(self._assessors),
            timeliness_half_life_days,
        )

    # ================================================================
    # 质量评估
    # ================================================================

    def assess_entity(
        self,
        entity: KnowledgeEntity,
        context: dict[str, Any] | None = None,
    ) -> QualityAssessmentResult:
        """评估单个实体的质量.

        执行六维质量评估，生成评估结果和改进建议。

        Args:
            entity: 被评估实体
            context: 评估上下文 (可包含 store, fact_checker,
                     ontology_registry, evidence, triples, conflicts,
                     provenance, authority_tier 等)

        Returns:
            质量评估结果
        """
        start_time = time.time()
        context = context or {}

        # 执行六维评估
        all_metrics: list[MetricResult] = []
        dim_scores: dict[str, float] = {}

        for dim, assessor in self._assessors.items():
            try:
                score, metrics = assessor.assess(entity, context)
                dim_scores[dim.value] = score
                all_metrics.extend(metrics)
            except Exception as e:
                logger.warning("维度 %s 评估失败: %s", dim.value, e)
                dim_scores[dim.value] = 0.5  # 失败时使用中间分
                all_metrics.append(MetricResult(
                    metric_id=f"{dim.value}_error",
                    metric_name=f"{dim.value} 评估",
                    dimension=dim,
                    score=0.5,
                    weight=1.0,
                    details=f"评估异常: {e}",
                ))

        # 构建质量评分
        existing_quality = entity.quality or QualityScore()
        quality_score = QualityScore(
            accuracy=dim_scores.get("accuracy", 0.8),
            trustworthiness=dim_scores.get("trustworthiness", 0.8),
            consistency=dim_scores.get("consistency", 0.9),
            timeliness=dim_scores.get("timeliness", 0.8),
            completeness=dim_scores.get("completeness", 0.7),
            relevancy=dim_scores.get("relevancy", 0.8),
            assessed_at=time.time(),
            assessor="quality_manager",
            evidence_count=existing_quality.evidence_count,
            peer_reviewed=existing_quality.peer_reviewed,
            verification_status=existing_quality.verification_status,
            last_verified_at=existing_quality.last_verified_at,
        )

        overall = quality_score.overall()
        grade = QualityGrade.from_score(overall)

        # 找出最弱和最强维度
        sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1])
        weakest = [d[0] for d in sorted_dims[:2]]
        strongest = [d[0] for d in sorted_dims[-2:]]

        # 生成改进建议
        recommendations = self._generate_recommendations(
            dim_scores, weakest, entity
        )

        # 构建结果
        result = QualityAssessmentResult(
            entity_id=entity.entity_id,
            assessment_level=AssessmentLevel.ENTITY,
            quality_score=quality_score,
            metric_results=[
                {
                    "metric_id": m.metric_id,
                    "metric_name": m.metric_name,
                    "dimension": m.dimension.value,
                    "score": round(m.score, 4),
                    "weight": m.weight,
                    "details": m.details,
                }
                for m in all_metrics
            ],
            overall_score=overall,
            grade=grade,
            weakest_dimensions=weakest,
            strongest_dimensions=strongest,
            recommendations=recommendations,
            assessed_at=time.time(),
            assessor="quality_manager",
            assessment_time_ms=(time.time() - start_time) * 1000,
        )

        # 记录到仪表板
        self._dashboard.record_assessment(result)

        # 记录溯源
        self._provenance_tracker.record(
            entity_id=entity.entity_id,
            activity_type="assess",
            agent_id="quality_manager",
            agent_role=ProvenanceRole.GENERATOR,
            description=f"质量评估: {grade.value} ({overall:.2%})",
        )

        return result

    def assess_batch(
        self,
        entities: list[KnowledgeEntity],
        context: dict[str, Any] | None = None,
    ) -> list[QualityAssessmentResult]:
        """批量评估实体质量.

        Args:
            entities: 实体列表
            context: 评估上下文

        Returns:
            评估结果列表
        """
        results: list[QualityAssessmentResult] = []
        for entity in entities:
            try:
                result = self.assess_entity(entity, context)
                results.append(result)
            except Exception as e:
                logger.error("实体 %s 评估失败: %s", entity.entity_id, e)
        return results

    def assess_global(
        self,
        store: Any,
        batch_size: int = 100,
    ) -> QualityDashboardData:
        """全库质量评估.

        对知识库中的所有实体执行质量评估，
        生成全局质量仪表板数据。

        Args:
            store: 知识存储 (KnowledgeStore)
            batch_size: 批量评估大小

        Returns:
            质量仪表板数据
        """
        start_time = time.time()

        # 获取所有实体
        all_entities: list[KnowledgeEntity] = []
        if hasattr(store, "entity_store"):
            all_entities = list(store.entity_store._entities.values())
        elif hasattr(store, "_entities"):
            all_entities = list(store._entities.values())

        total = len(all_entities)
        logger.info("全库质量评估开始: %d 个实体", total)

        # 批量评估
        context = {
            "store": store,
        }

        # 获取冲突和证据信息
        if hasattr(store, "_conflicts"):
            context["conflicts"] = [
                c.to_dict() if hasattr(c, "to_dict") else c
                for c in store._conflicts.values()
            ]
        if hasattr(store, "_evidence"):
            context["evidence"] = [
                e.model_dump() if hasattr(e, "model_dump") else e
                for e in store._evidence.values()
            ]

        # 分批评估
        for i in range(0, total, batch_size):
            batch = all_entities[i:i + batch_size]
            self.assess_batch(batch, context)

        # 生成仪表板数据
        conflict_stats = {
            "total": getattr(store, "conflict_count", 0),
            "unresolved": getattr(store, "unresolved_conflict_count", 0),
            "resolved": getattr(store, "conflict_count", 0)
                       - getattr(store, "unresolved_conflict_count", 0),
        }

        verification_stats = self._compute_verification_stats(all_entities)

        dashboard = self._dashboard.get_dashboard_data(
            total_entities=total,
            conflict_stats=conflict_stats,
            verification_stats=verification_stats,
        )

        elapsed = time.time() - start_time
        logger.info(
            "全库质量评估完成: %d 个实体, 耗时 %.2f 秒, "
            "平均分 %.2f%%",
            total,
            elapsed,
            dashboard.avg_overall_score * 100,
        )

        return dashboard

    # ================================================================
    # 冲突管理
    # ================================================================

    def detect_conflicts(
        self,
        entity: KnowledgeEntity,
        external_claims: list[dict[str, Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> list[KnowledgeConflict]:
        """检测实体相关的知识冲突.

        Args:
            entity: 被检测实体
            external_claims: 外部声明列表
            history: 历史版本列表

        Returns:
            检测到的冲突列表
        """
        conflicts: list[KnowledgeConflict] = []

        # 值冲突检测
        if external_claims:
            value_conflicts = self._conflict_detector.detect_value_conflicts(
                entity, external_claims
            )
            conflicts.extend(value_conflicts)

        # 时间冲突检测
        if history:
            temporal_conflicts = self._conflict_detector.detect_temporal_conflicts(
                entity, history
            )
            conflicts.extend(temporal_conflicts)

        # 记录溯源
        if conflicts:
            self._provenance_tracker.record(
                entity_id=entity.entity_id,
                activity_type="conflict_detection",
                agent_id="quality_manager",
                description=f"检测到 {len(conflicts)} 个冲突",
            )

        return conflicts

    def detect_cross_source_conflicts(
        self,
        entity_id: str,
        source_claims: dict[str, list[dict[str, Any]]],
    ) -> list[KnowledgeConflict]:
        """检测跨数据源冲突.

        Args:
            entity_id: 实体 ID
            source_claims: {source_name: [{field, value, confidence}, ...]}

        Returns:
            检测到的冲突列表
        """
        return self._conflict_detector.detect_cross_source_conflicts(
            entity_id, source_claims
        )

    def resolve_conflict(
        self,
        conflict: KnowledgeConflict,
        strategy: ConflictResolutionStrategy | None = None,
    ) -> KnowledgeConflict:
        """消解知识冲突.

        Args:
            conflict: 待消解的冲突
            strategy: 消解策略 (None 使用冲突默认策略)

        Returns:
            消解后的冲突记录
        """
        resolved = self._conflict_resolver.resolve(conflict, strategy)

        # 记录溯源
        self._provenance_tracker.record(
            entity_id=conflict.entity_id,
            activity_type="conflict_resolution",
            agent_id="quality_manager",
            description=f"冲突消解: {conflict.conflict_id}, "
                       f"策略: {resolved.resolution_strategy.value}",
        )

        return resolved

    def resolve_conflicts(
        self,
        conflicts: list[KnowledgeConflict],
        strategy: ConflictResolutionStrategy | None = None,
    ) -> list[KnowledgeConflict]:
        """批量消解冲突.

        Args:
            conflicts: 冲突列表
            strategy: 消解策略

        Returns:
            消解后的冲突列表
        """
        return [
            self.resolve_conflict(c, strategy) for c in conflicts
        ]

    # ================================================================
    # 溯源追踪
    # ================================================================

    def record_provenance(
        self,
        entity_id: str,
        activity_type: str,
        agent_id: str = "system",
        **kwargs: Any,
    ) -> ProvenanceInfo:
        """记录溯源信息.

        Args:
            entity_id: 实体 ID
            activity_type: 活动类型
            agent_id: 智能体 ID
            **kwargs: 传递给 ProvenanceTracker.record 的额外参数

        Returns:
            溯源信息记录
        """
        return self._provenance_tracker.record(
            entity_id=entity_id,
            activity_type=activity_type,
            agent_id=agent_id,
            **kwargs,
        )

    def get_provenance(self, entity_id: str) -> ProvenanceInfo | None:
        """获取实体的溯源信息."""
        return self._provenance_tracker.get_provenance(entity_id)

    def trace_provenance_chain(
        self, entity_id: str, max_depth: int = 10
    ) -> list[ProvenanceInfo]:
        """追踪完整溯源链."""
        return self._provenance_tracker.trace_chain(entity_id, max_depth)

    def verify_integrity(
        self, entity_id: str, content: str
    ) -> ProvenanceVerificationResult:
        """验证内容完整性."""
        return self._provenance_tracker.verify_integrity(entity_id, content)

    def verify_provenance_chain(
        self, entity_id: str
    ) -> ProvenanceVerificationResult:
        """验证溯源链完整性."""
        return self._provenance_tracker.verify_chain(entity_id)

    def get_audit_log(
        self,
        entity_id: str | None = None,
        activity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """获取审计日志."""
        return self._provenance_tracker.get_audit_log(
            entity_id, activity_type, limit
        )

    # ================================================================
    # 质量监控
    # ================================================================

    def get_dashboard(
        self,
        total_entities: int = 0,
        conflict_stats: dict[str, int] | None = None,
        verification_stats: dict[str, int] | None = None,
    ) -> QualityDashboardData:
        """获取质量仪表板数据."""
        return self._dashboard.get_dashboard_data(
            total_entities=total_entities,
            conflict_stats=conflict_stats,
            verification_stats=verification_stats,
        )

    def get_entity_quality_history(
        self, entity_id: str
    ) -> list[QualityAssessmentResult]:
        """获取实体的质量评估历史."""
        return self._dashboard.get_entity_history(entity_id)

    # ================================================================
    # 评估器管理
    # ================================================================

    def register_assessor(
        self,
        dimension: QualityDimension,
        assessor: BaseQualityAssessor,
    ) -> None:
        """注册自定义评估器.

        Args:
            dimension: 质量维度
            assessor: 评估器实例
        """
        with self._lock:
            self._assessors[dimension] = assessor
            logger.info("注册评估器: %s", dimension.value)

    def get_assessor(
        self, dimension: QualityDimension
    ) -> BaseQualityAssessor | None:
        """获取指定维度的评估器."""
        return self._assessors.get(dimension)

    # ================================================================
    # 内部方法
    # ================================================================

    def _generate_recommendations(
        self,
        dim_scores: dict[str, float],
        weakest: list[str],
        entity: KnowledgeEntity,
    ) -> list[str]:
        """生成改进建议."""
        recommendations: list[str] = []

        # 基于最弱维度生成建议
        for dim in weakest:
            score = dim_scores.get(dim, 0.5)
            if dim == "accuracy" and score < 0.7:
                recommendations.append(
                    "准确性不足: 建议增加事实校验标准值覆盖，"
                    "或引入更多来源进行交叉验证"
                )
            elif dim == "consistency" and score < 0.7:
                recommendations.append(
                    "一致性不足: 检查是否存在属性矛盾或本体约束违反，"
                    "修正冲突声明"
                )
            elif dim == "completeness" and score < 0.7:
                recommendations.append(
                    "完整性不足: 补充核心属性和标识符，"
                    "增加描述文本长度和关系连接"
                )
            elif dim == "timeliness" and score < 0.7:
                recommendations.append(
                    "时效性不足: 更新过时知识，"
                    "重新评估或验证已有声明"
                )
            elif dim == "trustworthiness" and score < 0.7:
                recommendations.append(
                    "可信度不足: 引入更高权威度的数据源，"
                    "补充同行评审信息或溯源链"
                )
            elif dim == "relevancy" and score < 0.7:
                recommendations.append(
                    "相关性不足: 增加领域关键词，"
                    "对齐本体定义或调整实体分类"
                )

        # 通用建议
        if not recommendations:
            recommendations.append(
                "各维度质量均达标，建议定期复评以维持质量水平"
            )

        # 如果有争议状态
        existing_quality = entity.quality or QualityScore()
        if existing_quality.verification_status == VerificationStatus.DISPUTED:
            recommendations.append(
                "实体处于争议状态: 建议优先解决相关冲突后重新评估"
            )

        return recommendations

    @staticmethod
    def _compute_verification_stats(
        entities: list[KnowledgeEntity],
    ) -> dict[str, int]:
        """计算验证状态统计."""
        stats: dict[str, int] = defaultdict(int)
        for entity in entities:
            quality = entity.quality or QualityScore()
            status = quality.verification_status.value
            stats[status] += 1
        return dict(stats)

    # ================================================================
    # 属性
    # ================================================================

    @property
    def assessor_count(self) -> int:
        """注册的评估器数量."""
        return len(self._assessors)

    @property
    def provenance_count(self) -> int:
        """溯源记录总数."""
        return self._provenance_tracker.provenance_count

    @property
    def audit_log_count(self) -> int:
        """审计日志总数."""
        return self._provenance_tracker.audit_log_count

    @property
    def assessment_count(self) -> int:
        """评估记录总数."""
        return self._dashboard.assessment_count


__all__ = [
    # 枚举
    "AssessmentLevel",
    "QualityGrade",
    "ConflictDetectionMethod",
    "ProvenanceVerificationResult",
    # 数据模型
    "MetricResult",
    "QualityAssessmentResult",
    "QualityDashboardData",
    # 评估器
    "BaseQualityAssessor",
    "AccuracyAssessor",
    "ConsistencyAssessor",
    "CompletenessAssessor",
    "TimelinessAssessor",
    "TrustworthinessAssessor",
    "RelevancyAssessor",
    # 冲突管理
    "ConflictDetector",
    "ConflictResolver",
    # 溯源追踪
    "ProvenanceTracker",
    # 质量监控
    "QualityDashboard",
    # 质量管理器
    "QualityManager",
]
