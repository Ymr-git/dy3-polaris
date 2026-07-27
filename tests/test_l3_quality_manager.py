"""L3 知识质量管理与评估模块测试套件.

覆盖五个核心组件:
- quality_manager: 六维质量评估引擎 (准确性/一致性/完整性/时效性/可信度/相关性)
- ConflictDetector + ConflictResolver: 冲突检测与消解 (MACR + CRDL)
- ProvenanceTracker: 溯源追踪与审计 (W3C PROV-O)
- QualityDashboard: 质量监控仪表板
- QualityManager: 统一编排管理器
"""

from __future__ import annotations

import logging
import time

import pytest

from dy3_polaris.l3 import (
    AccuracyAssessor,
    AssessmentLevel,
    CompletenessAssessor,
    ConflictDetector,
    ConflictDetectionMethod,
    ConflictResolutionStrategy,
    ConflictResolver,
    ConflictType,
    ConsistencyAssessor,
    EntityType,
    KnowledgeEntity,
    KnowledgeStore,
    KnowledgeConflict,
    ProvenanceInfo,
    ProvenanceRole,
    ProvenanceTracker,
    ProvenanceVerificationResult,
    QualityAssessmentResult,
    QualityDashboard,
    QualityDashboardData,
    QualityDimension,
    QualityGrade,
    QualityManager,
    QualityScore,
    RelevancyAssessor,
    TimelinessAssessor,
    TrustworthinessAssessor,
    VerificationStatus,
)
from dy3_polaris.l3.ingestion import AuthorityTier

logging.disable(logging.CRITICAL)


# ============================================================
# 测试数据工厂
# ============================================================


def make_entity(
    name: str = "Dy3+离子",
    entity_type: EntityType = EntityType.CHEMICAL_COMPOUND,
    description: str = "Dy3+是镝离子的三价态，在稀土发光材料中用作激活离子，"
                       "其4F9/2→6H13/2跃迁产生580nm黄色发射。",
    properties: dict | None = None,
    quality: QualityScore | None = None,
    **kwargs,
) -> KnowledgeEntity:
    """创建测试实体."""
    if properties is None:
        properties = {
            "formula": "Dy3+",
            "cas": "7429-91-6",
            "emission_wavelength": 580,
            "ionic_radius": 1.07,
        }
    if quality is None:
        quality = QualityScore()
    return KnowledgeEntity(
        name=name,
        entity_type=entity_type,
        domain="chemistry",
        description=description,
        properties=properties,
        quality=quality,
        **kwargs,
    )


def make_low_quality_entity() -> KnowledgeEntity:
    """创建低质量测试实体."""
    return KnowledgeEntity(
        name="低质量实体",
        entity_type=EntityType.CONCEPT,
        domain="test",
        description="x",
        properties={},
        quality=QualityScore(
            accuracy=0.3,
            trustworthiness=0.2,
            consistency=0.4,
            timeliness=0.1,
            completeness=0.2,
            relevancy=0.3,
            verification_status=VerificationStatus.DISPUTED,
        ),
    )


# ============================================================
# 1. 质量评估引擎测试
# ============================================================


class TestQualityAssessors:
    """六维质量评估器测试."""

    def test_accuracy_assessor_basic(self):
        """测试准确性评估器基本功能."""
        assessor = AccuracyAssessor()
        entity = make_entity()
        score, metrics = assessor.assess(entity)

        assert 0.0 <= score <= 1.0
        assert len(metrics) == 4  # 四个指标
        assert all(m.score >= 0.0 and m.score <= 1.0 for m in metrics)

    def test_accuracy_assessor_with_evidence(self):
        """测试有证据时的准确性评估."""
        assessor = AccuracyAssessor()
        entity = make_entity(
            quality=QualityScore(
                evidence_count=5,
                peer_reviewed=True,
                verification_status=VerificationStatus.VERIFIED,
            )
        )
        context = {
            "evidence": [
                {"source_reference": "DOI:10.1038/xxx", "confidence": 0.9},
                {"source_reference": "NIST-SRD", "confidence": 0.95},
                {"source_reference": "CRC-Handbook", "confidence": 0.85},
            ]
        }
        score, metrics = assessor.assess(entity, context)

        # 有多源证据 + 已验证 + 同行评审 → 高分
        assert score >= 0.7
        metric_dict = {m.metric_id: m for m in metrics}
        assert metric_dict["acc_corroboration"].score >= 0.8
        assert metric_dict["acc_verification"].score == 1.0

    def test_accuracy_assessor_disputed(self):
        """测试争议状态的准确性评估."""
        assessor = AccuracyAssessor()
        entity = make_entity(
            quality=QualityScore(
                verification_status=VerificationStatus.DISPUTED,
                evidence_count=0,
            )
        )
        score, metrics = assessor.assess(entity)

        # 争议状态 + 无证据 → 低分
        assert score < 0.6
        metric_dict = {m.metric_id: m for m in metrics}
        assert metric_dict["acc_verification"].score == 0.2

    def test_consistency_assessor_basic(self):
        """测试一致性评估器基本功能."""
        assessor = ConsistencyAssessor()
        entity = make_entity()
        score, metrics = assessor.assess(entity)

        assert 0.0 <= score <= 1.0
        assert len(metrics) == 4

    def test_consistency_assessor_with_conflicts(self):
        """测试有冲突时的一致性评估."""
        assessor = ConsistencyAssessor()
        entity = make_entity()
        context = {
            "conflicts": [
                {"entity_id": entity.entity_id, "status": "detected"},
                {"entity_id": entity.entity_id, "status": "detected"},
                {"entity_id": entity.entity_id, "status": "detected"},
                {"entity_id": entity.entity_id, "status": "detected"},
                {"entity_id": entity.entity_id, "status": "detected"},
            ]
        }
        score, metrics = assessor.assess(entity, context)

        # 有多个未解决冲突 → 一致性降低
        assert score < 0.85
        metric_dict = {m.metric_id: m for m in metrics}
        assert metric_dict["con_conflict"].score < 0.5

    def test_completeness_assessor_basic(self):
        """测试完整性评估器基本功能."""
        assessor = CompletenessAssessor()
        entity = make_entity()
        score, metrics = assessor.assess(entity)

        assert 0.0 <= score <= 1.0
        assert len(metrics) == 4

    def test_completeness_assessor_empty_entity(self):
        """测试空实体的完整性评估."""
        assessor = CompletenessAssessor()
        entity = make_low_quality_entity()
        score, metrics = assessor.assess(entity)

        # 空实体 → 低完整性
        assert score < 0.4

    def test_completeness_assessor_rich_entity(self):
        """测试丰富实体的完整性评估."""
        assessor = CompletenessAssessor()
        entity = make_entity(
            description="Dy3+是镝离子的三价态，在稀土发光材料中用作激活离子。"
                       "其4F9/2→6H13/2跃迁产生580nm黄色发射，"
                       "广泛应用于白光LED和荧光粉领域。"
                       "Dy3+的离子半径为1.07Å，配位数通常为6-8。"
                       "在YAG基质中，Dy3+的发射峰位于480nm(蓝)和580nm(黄)。",
            properties={
                "formula": "Dy3+",
                "cas": "7429-91-6",
                "molecular_weight": 162.5,
                "emission_wavelength": 580,
                "ionic_radius": 1.07,
                "coordination_number": 8,
            },
            identifiers={
                "cas": "7429-91-6",
                "inchi": "InChI=1S/Dy/q+3",
                "smiles": "[Dy+3]",
            },
        )
        context = {
            "triples": [
                {"predicate": "EMITS_AT"},
                {"predicate": "DOPED_IN"},
                {"predicate": "HAS_PROPERTY"},
            ]
        }
        score, metrics = assessor.assess(entity, context)

        # 丰富实体 → 高完整性
        assert score >= 0.7

    def test_timeliness_assessor_basic(self):
        """测试时效性评估器基本功能."""
        assessor = TimelinessAssessor(half_life_days=365)
        entity = make_entity()
        score, metrics = assessor.assess(entity)

        assert 0.0 <= score <= 1.0
        assert len(metrics) == 4

    def test_timeliness_assessor_recent(self):
        """测试新创建实体的时效性."""
        assessor = TimelinessAssessor(half_life_days=365)
        entity = make_entity()
        # 确保创建时间是当前
        entity.updated_at = time.time()
        entity.quality.assessed_at = time.time()
        entity.quality.last_verified_at = time.time()

        score, metrics = assessor.assess(entity)

        # 刚创建 → 高时效性
        assert score >= 0.7

    def test_timeliness_assessor_stale(self):
        """测试过时实体的时效性."""
        assessor = TimelinessAssessor(half_life_days=30)  # 30天半衰期
        entity = make_entity()
        # 设置为 100 天前
        old_time = time.time() - 100 * 86400
        entity.updated_at = old_time
        entity.quality.assessed_at = old_time
        entity.quality.last_verified_at = 0  # 从未验证

        score, metrics = assessor.assess(entity)

        # 100天前 (远超30天半衰期) → 低时效性
        assert score < 0.5

    def test_trustworthiness_assessor_basic(self):
        """测试可信度评估器基本功能."""
        assessor = TrustworthinessAssessor()
        entity = make_entity()
        context = {"authority_tier": AuthorityTier.T2}
        score, metrics = assessor.assess(entity, context)

        assert 0.0 <= score <= 1.0
        assert len(metrics) == 4

    def test_trustworthiness_t1_source(self):
        """测试 T1 权威来源的可信度."""
        assessor = TrustworthinessAssessor()
        entity = make_entity(
            quality=QualityScore(peer_reviewed=True)
        )
        prov = ProvenanceInfo(
            entity_id=entity.entity_id,
            derived_from=["source-1"],
            primary_source="Nature",
            integrity_hash="abc123def456",
        )
        context = {
            "authority_tier": AuthorityTier.T1,
            "evidence": [
                {"source_reference": "Nature", "confidence": 0.99},
                {"source_reference": "NIST", "confidence": 0.95},
                {"source_reference": "CRC", "confidence": 0.90},
            ],
            "provenance": prov,
        }
        score, metrics = assessor.assess(entity, context)

        # T1 + 同行评审 + 多源 + 完整溯源 → 高可信度
        assert score >= 0.80

    def test_trustworthiness_t4_source(self):
        """测试 T4 基础来源的可信度."""
        assessor = TrustworthinessAssessor()
        entity = make_entity(
            quality=QualityScore(peer_reviewed=False)
        )
        context = {"authority_tier": AuthorityTier.T4}
        score, metrics = assessor.assess(entity, context)

        # T4 + 无同行评审 → 低可信度
        assert score < 0.6

    def test_relevancy_assessor_basic(self):
        """测试相关性评估器基本功能."""
        assessor = RelevancyAssessor()
        entity = make_entity()
        score, metrics = assessor.assess(entity)

        assert 0.0 <= score <= 1.0
        assert len(metrics) == 4

    def test_relevancy_domain_entity(self):
        """测试领域实体的相关性."""
        assessor = RelevancyAssessor()
        entity = make_entity(
            entity_type=EntityType.CHEMICAL_COMPOUND,
            description="Dy3+离子在稀土发光材料中的应用，"
                       "研究其激发、发射、跃迁和能量传递机理。",
        )
        score, metrics = assessor.assess(entity)

        # 化学实体 + 领域关键词 → 高相关性
        assert score >= 0.7

    def test_relevancy_non_domain_entity(self):
        """测试非领域实体的相关性."""
        assessor = RelevancyAssessor()
        entity = make_entity(
            entity_type=EntityType.PERSON,
            name="张三",
            description="一个普通人。",
            properties={},
        )
        score, metrics = assessor.assess(entity)

        # 非领域实体 → 低相关性
        assert score < 0.6


# ============================================================
# 2. 冲突检测与消解测试
# ============================================================


class TestConflictDetector:
    """冲突检测器测试."""

    def test_detect_value_conflicts(self):
        """测试值冲突检测."""
        detector = ConflictDetector()
        entity = make_entity(
            properties={"emission_wavelength": 580, "ionic_radius": 1.07}
        )
        external_claims = [
            {"field": "emission_wavelength", "value": 575, "source": "source_A"},
            {"field": "emission_wavelength", "value": 610, "source": "source_B"},
            {"field": "ionic_radius", "value": 1.07, "source": "source_A"},  # 一致
        ]
        conflicts = detector.detect_value_conflicts(entity, external_claims)

        # 580 vs 610 应该冲突 (差异 > 5%)
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == ConflictType.SOURCE_BASED
        assert conflicts[0].detection_method == "value_comparison"

    def test_detect_value_conflicts_no_conflict(self):
        """测试无冲突的情况."""
        detector = ConflictDetector()
        entity = make_entity(
            properties={"emission_wavelength": 580}
        )
        external_claims = [
            {"field": "emission_wavelength", "value": 579, "source": "A"},  # 在容差内
        ]
        conflicts = detector.detect_value_conflicts(entity, external_claims)

        # 580 vs 579 在 5% 容差内 → 无冲突
        assert len(conflicts) == 0

    def test_detect_temporal_conflicts(self):
        """测试时间冲突检测."""
        detector = ConflictDetector()
        entity = make_entity(
            properties={"emission_wavelength": 580}
        )
        history = [
            {"timestamp": time.time() - 86400, "properties": {"emission_wavelength": 300}},
            {"timestamp": time.time(), "properties": {"emission_wavelength": 580}},
        ]
        conflicts = detector.detect_temporal_conflicts(entity, history)

        # 300 → 580 变化超过 50% → 时间冲突
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == ConflictType.TEMPORAL

    def test_detect_cross_source_conflicts(self):
        """测试跨源冲突检测."""
        detector = ConflictDetector()
        entity_id = "test-entity-001"
        source_claims = {
            "NIST": [{"field": "boiling_point", "value": 100.0, "confidence": 0.99}],
            "Wikipedia": [{"field": "boiling_point", "value": 120.0, "confidence": 0.8}],
        }
        conflicts = detector.detect_cross_source_conflicts(entity_id, source_claims)

        # 100 vs 120 冲突
        assert len(conflicts) >= 1
        assert conflicts[0].detection_method == "cross_source"

    def test_string_conflict_detection(self):
        """测试字符串冲突检测."""
        detector = ConflictDetector()
        entity = make_entity(
            properties={"name": "镝离子"}
        )
        external_claims = [
            {"field": "name", "value": "铽离子", "source": "wrong_source"},
        ]
        conflicts = detector.detect_value_conflicts(entity, external_claims)

        # "镝离子" vs "铽离子" → 冲突
        assert len(conflicts) >= 1


class TestConflictResolver:
    """冲突消解器测试."""

    def test_resolve_by_quality(self):
        """测试按质量消解."""
        resolver = ConflictResolver()
        conflict = KnowledgeConflict(
            conflict_type=ConflictType.SOURCE_BASED,
            entity_id="test-001",
            field_path="emission_wavelength",
            conflicting_values=[
                {"value": 580, "source": "A", "quality_score": 0.9},
                {"value": 610, "source": "B", "quality_score": 0.6},
            ],
            detection_method="value_comparison",
            resolution_strategy=ConflictResolutionStrategy.PREFER_HIGHER_QUALITY,
        )
        resolved = resolver.resolve(conflict)

        assert resolved.is_resolved()
        assert resolved.resolved_value == 580  # 质量更高的值
        assert "quality" in resolved.resolution_explanation.lower() or "质量" in resolved.resolution_explanation

    def test_resolve_by_recency(self):
        """测试按时间消解."""
        resolver = ConflictResolver()
        now = time.time()
        conflict = KnowledgeConflict(
            conflict_type=ConflictType.TEMPORAL,
            entity_id="test-002",
            field_path="boiling_point",
            conflicting_values=[
                {"value": 100, "source": "old", "timestamp": now - 86400},
                {"value": 105, "source": "new", "timestamp": now},
            ],
            detection_method="temporal_check",
            resolution_strategy=ConflictResolutionStrategy.PREFER_MOST_RECENT,
        )
        resolved = resolver.resolve(conflict)

        assert resolved.is_resolved()
        assert resolved.resolved_value == 105  # 更新的值

    def test_resolve_by_trust(self):
        """测试按可信度消解."""
        resolver = ConflictResolver()
        conflict = KnowledgeConflict(
            conflict_type=ConflictType.SOURCE_BASED,
            entity_id="test-003",
            field_path="density",
            conflicting_values=[
                {"value": 7.9, "source": "NIST", "confidence": 0.99},
                {"value": 8.5, "source": "blog", "confidence": 0.3},
            ],
            detection_method="cross_source",
            resolution_strategy=ConflictResolutionStrategy.PREFER_MOST_TRUSTED,
        )
        resolved = resolver.resolve(conflict)

        assert resolved.is_resolved()
        assert resolved.resolved_value == 7.9  # 更可信的值

    def test_resolve_keep_both(self):
        """测试保留双方策略."""
        resolver = ConflictResolver()
        conflict = KnowledgeConflict(
            conflict_type=ConflictType.SEMANTIC,
            entity_id="test-004",
            field_path="category",
            conflicting_values=[
                {"value": "metal", "source": "A"},
                {"value": "nonmetal", "source": "B"},
            ],
            detection_method="semantic_analysis",
            resolution_strategy=ConflictResolutionStrategy.KEEP_BOTH,
        )
        resolved = resolver.resolve(conflict)

        assert resolved.is_resolved()
        assert resolved.resolved_value is None  # 保留双方不选值

    def test_resolve_manual_review(self):
        """测试人工审核策略."""
        resolver = ConflictResolver()
        conflict = KnowledgeConflict(
            conflict_type=ConflictType.SEMANTIC,
            entity_id="test-005",
            conflicting_values=[
                {"value": "A", "source": "X"},
                {"value": "B", "source": "Y"},
            ],
            detection_method="semantic_analysis",
            resolution_strategy=ConflictResolutionStrategy.MANUAL_REVIEW,
        )
        resolved = resolver.resolve(conflict)

        assert resolved.is_resolved()
        assert "人工" in resolved.resolution_explanation or "manual" in resolved.resolution_explanation.lower()


# ============================================================
# 3. 溯源追踪测试
# ============================================================


class TestProvenanceTracker:
    """溯源追踪器测试."""

    def test_record_provenance(self):
        """测试溯源记录."""
        tracker = ProvenanceTracker()
        prov = tracker.record(
            entity_id="entity-001",
            activity_type="ingest",
            agent_id="ingestion_pipeline",
            description="从 PDF 文档导入知识",
        )

        assert prov.entity_id == "entity-001"
        assert prov.activity_type == "ingest"
        assert prov.generated_by_agent == "ingestion_pipeline"
        assert prov.integrity_hash != ""  # 自动生成哈希
        assert tracker.provenance_count == 1

    def test_get_provenance(self):
        """测试获取溯源."""
        tracker = ProvenanceTracker()
        tracker.record(
            entity_id="entity-002",
            activity_type="assess",
            agent_id="quality_manager",
        )
        prov = tracker.get_provenance("entity-002")

        assert prov is not None
        assert prov.activity_type == "assess"

    def test_trace_chain(self):
        """测试溯源链追踪."""
        tracker = ProvenanceTracker()

        # 构建溯源链: C → B → A (原始)
        tracker.record(
            entity_id="A",
            activity_type="ingest",
            description="原始来源",
        )
        tracker.record(
            entity_id="B",
            activity_type="derive",
            derived_from=["A"],
            description="从 A 派生",
        )
        tracker.record(
            entity_id="C",
            activity_type="derive",
            derived_from=["B"],
            description="从 B 派生",
        )

        chain = tracker.trace_chain("C")

        # 应该追踪到完整链条
        assert len(chain) >= 2
        assert chain[0].entity_id == "C"
        # 应包含 B 或 A
        entity_ids = [p.entity_id for p in chain]
        assert "B" in entity_ids

    def test_verify_integrity(self):
        """测试完整性验证."""
        tracker = ProvenanceTracker()
        content = "Dy3+ emission wavelength is 580nm"
        content_hash = tracker._compute_hash(content)

        tracker.record(
            entity_id="entity-003",
            activity_type="ingest",
            content_hash=content_hash,
        )

        # 验证正确内容
        result = tracker.verify_integrity("entity-003", content)
        assert result == ProvenanceVerificationResult.VERIFIED

        # 验证篡改内容
        result = tracker.verify_integrity("entity-003", "tampered content")
        assert result == ProvenanceVerificationResult.TAMPERED

    def test_verify_chain_broken(self):
        """测试断裂链验证."""
        tracker = ProvenanceTracker()
        tracker.record(
            entity_id="X",
            activity_type="derive",
            derived_from=["missing_parent"],  # 不存在的父节点
        )

        result = tracker.verify_chain("X")
        assert result == ProvenanceVerificationResult.BROKEN_CHAIN

    def test_verify_unverifiable(self):
        """测试不可验证 (无溯源)."""
        tracker = ProvenanceTracker()
        result = tracker.verify_chain("nonexistent")
        assert result == ProvenanceVerificationResult.UNVERIFIABLE

    def test_audit_log(self):
        """测试审计日志."""
        tracker = ProvenanceTracker()
        tracker.record(entity_id="e1", activity_type="ingest")
        tracker.record(entity_id="e2", activity_type="assess")
        tracker.record(entity_id="e1", activity_type="update")

        # 全部日志
        all_logs = tracker.get_audit_log()
        assert len(all_logs) == 3

        # 按实体过滤
        e1_logs = tracker.get_audit_log(entity_id="e1")
        assert len(e1_logs) == 2

        # 按活动类型过滤
        assess_logs = tracker.get_audit_log(activity_type="assess")
        assert len(assess_logs) == 1


# ============================================================
# 4. 质量仪表板测试
# ============================================================


class TestQualityDashboard:
    """质量监控仪表板测试."""

    def test_empty_dashboard(self):
        """测试空仪表板."""
        dashboard = QualityDashboard()
        data = dashboard.get_dashboard_data(total_entities=100)

        assert data.total_entities == 100
        assert data.assessed_entities == 0
        assert data.avg_overall_score == 0.0

    def test_record_and_retrieve(self):
        """测试记录和检索评估结果."""
        dashboard = QualityDashboard()
        result = QualityAssessmentResult(
            entity_id="test-001",
            overall_score=0.85,
            grade=QualityGrade.GOOD,
            quality_score=QualityScore(
                accuracy=0.9, trustworthiness=0.85, consistency=0.88,
                timeliness=0.82, completeness=0.80, relevancy=0.85,
            ),
        )
        dashboard.record_assessment(result)

        data = dashboard.get_dashboard_data()
        assert data.assessed_entities == 1
        assert data.avg_overall_score == pytest.approx(0.85, rel=0.01)
        assert data.grade_distribution.get("good") == 1

    def test_grade_distribution(self):
        """测试等级分布."""
        dashboard = QualityDashboard()
        grades = [
            (0.95, QualityGrade.EXCELLENT),
            (0.85, QualityGrade.GOOD),
            (0.85, QualityGrade.GOOD),
            (0.65, QualityGrade.FAIR),
            (0.35, QualityGrade.UNACCEPTABLE),
        ]
        for score, grade in grades:
            dashboard.record_assessment(QualityAssessmentResult(
                entity_id=f"test-{score}",
                overall_score=score,
                grade=grade,
                quality_score=QualityScore(accuracy=score),
            ))

        data = dashboard.get_dashboard_data()
        assert data.grade_distribution.get("excellent") == 1
        assert data.grade_distribution.get("good") == 2
        assert data.grade_distribution.get("fair") == 1
        assert data.grade_distribution.get("unacceptable") == 1

    def test_alerts_generation(self):
        """测试告警生成."""
        dashboard = QualityDashboard()
        # 记录低分评估
        dashboard.record_assessment(QualityAssessmentResult(
            entity_id="low-quality",
            overall_score=0.3,
            grade=QualityGrade.UNACCEPTABLE,
            quality_score=QualityScore(
                accuracy=0.2, trustworthiness=0.3, consistency=0.4,
                timeliness=0.1, completeness=0.2, relevancy=0.3,
            ),
        ))

        data = dashboard.get_dashboard_data(
            conflict_stats={"total": 10, "unresolved": 8}
        )

        # 应生成告警
        assert len(data.alerts) > 0
        # 应有低分告警
        low_score_alerts = [a for a in data.alerts if a["type"] == "low_overall_score"]
        assert len(low_score_alerts) >= 1

    def test_entity_history(self):
        """测试实体评估历史."""
        dashboard = QualityDashboard()
        for i in range(3):
            dashboard.record_assessment(QualityAssessmentResult(
                entity_id="history-test",
                overall_score=0.7 + i * 0.05,
                grade=QualityGrade.GOOD,
                quality_score=QualityScore(),
            ))

        history = dashboard.get_entity_history("history-test")
        assert len(history) == 3
        # 应按时间顺序
        scores = [h.overall_score for h in history]
        assert scores[0] < scores[-1]


# ============================================================
# 5. QualityManager 统一编排测试
# ============================================================


class TestQualityManager:
    """质量管理器统一编排测试."""

    def test_init(self):
        """测试初始化."""
        manager = QualityManager()
        assert manager.assessor_count == 6  # 六维评估器
        assert manager.provenance_count == 0
        assert manager.audit_log_count == 0
        assert manager.assessment_count == 0

    def test_assess_entity(self):
        """测试单实体评估."""
        manager = QualityManager()
        entity = make_entity()

        result = manager.assess_entity(entity)

        assert result.entity_id == entity.entity_id
        assert result.assessment_level == AssessmentLevel.ENTITY
        assert 0.0 <= result.overall_score <= 1.0
        assert isinstance(result.grade, QualityGrade)
        assert len(result.metric_results) > 0
        assert len(result.recommendations) > 0
        assert result.assessment_time_ms > 0

        # 应记录到仪表板
        assert manager.assessment_count == 1

        # 应记录溯源
        assert manager.provenance_count >= 1

    def test_assess_entity_with_context(self):
        """测试带上下文的实体评估."""
        manager = QualityManager()
        entity = make_entity()
        context = {
            "authority_tier": AuthorityTier.T1,
            "evidence": [
                {"source_reference": "Nature", "confidence": 0.95},
                {"source_reference": "NIST", "confidence": 0.99},
            ],
            "triples": [
                {"predicate": "EMITS_AT"},
                {"predicate": "DOPED_IN"},
            ],
        }
        result = manager.assess_entity(entity, context)

        # 有 T1 来源 + 多源证据 → 高分
        assert result.overall_score >= 0.6

    def test_assess_low_quality_entity(self):
        """测试低质量实体评估."""
        manager = QualityManager()
        entity = make_low_quality_entity()

        result = manager.assess_entity(entity)

        # 低质量实体 → 低分
        assert result.overall_score < 0.6
        assert result.grade in [QualityGrade.POOR, QualityGrade.UNACCEPTABLE, QualityGrade.FAIR]
        assert len(result.recommendations) > 0

    def test_assess_entity_with_none_quality(self):
        """测试 quality=None 的实体评估 (健壮性)."""
        manager = QualityManager()
        # 创建不设置 quality 的实体 (模拟实际使用场景)
        entity = KnowledgeEntity(
            name="无质量评分实体",
            entity_type=EntityType.CONCEPT,
            domain="test",
            description="测试无 quality 属性的实体",
        )
        assert entity.quality is None  # 确认 quality 默认为 None

        # 不应崩溃
        result = manager.assess_entity(entity)
        assert result.overall_score > 0.0
        assert isinstance(result.grade, QualityGrade)
        assert len(result.metric_results) > 0

    def test_assess_batch(self):
        """测试批量评估."""
        manager = QualityManager()
        entities = [
            make_entity(name=f"entity-{i}")
            for i in range(5)
        ]
        results = manager.assess_batch(entities)

        assert len(results) == 5
        assert manager.assessment_count == 5

    def test_assess_global(self):
        """测试全库评估."""
        manager = QualityManager()
        store = KnowledgeStore()

        # 添加测试实体
        for i in range(10):
            entity = make_entity(name=f"global-entity-{i}")
            store.add_entity(entity)

        dashboard = manager.assess_global(store)

        assert dashboard.total_entities == 10
        assert dashboard.assessed_entities == 10
        assert dashboard.avg_overall_score > 0.0
        assert len(dashboard.grade_distribution) > 0

    def test_detect_and_resolve_conflicts(self):
        """测试冲突检测与消解流程."""
        manager = QualityManager()
        entity = make_entity(
            properties={"emission_wavelength": 580}
        )

        # 检测冲突 (610 vs 580 差异仅 ~4.9%, 在 5% 容差内不触发;
        # 使用 700 确保超过 5% 容差阈值)
        external_claims = [
            {"field": "emission_wavelength", "value": 700, "source": "conflicting_source"},
        ]
        conflicts = manager.detect_conflicts(entity, external_claims=external_claims)

        assert len(conflicts) >= 1

        # 消解冲突
        resolved = manager.resolve_conflict(conflicts[0])

        assert resolved.is_resolved()
        assert resolved.resolved_value is not None

    def test_provenance_tracking(self):
        """测试溯源追踪."""
        manager = QualityManager()
        entity = make_entity()

        # 评估实体 (自动记录溯源)
        manager.assess_entity(entity)

        # 手动记录溯源
        prov = manager.record_provenance(
            entity_id=entity.entity_id,
            activity_type="update",
            agent_id="test_agent",
            description="测试溯源记录",
        )

        assert prov is not None
        assert prov.activity_type == "update"

        # 获取溯源
        retrieved = manager.get_provenance(entity.entity_id)
        assert retrieved is not None

        # 追踪链
        chain = manager.trace_provenance_chain(entity.entity_id)
        assert len(chain) >= 1

    def test_verify_integrity(self):
        """测试完整性验证."""
        manager = QualityManager()
        content = "test content for integrity"
        content_hash = ProvenanceTracker._compute_hash(content)

        manager.record_provenance(
            entity_id="verify-test",
            activity_type="ingest",
            content_hash=content_hash,
        )

        # 验证正确内容
        result = manager.verify_integrity("verify-test", content)
        assert result == ProvenanceVerificationResult.VERIFIED

        # 验证篡改内容
        result = manager.verify_integrity("verify-test", "tampered")
        assert result == ProvenanceVerificationResult.TAMPERED

    def test_get_dashboard(self):
        """测试获取仪表板."""
        manager = QualityManager()
        entity = make_entity()
        manager.assess_entity(entity)

        dashboard = manager.get_dashboard(
            total_entities=100,
            conflict_stats={"total": 5, "unresolved": 2},
        )

        assert dashboard.total_entities == 100
        assert dashboard.assessed_entities == 1
        assert isinstance(dashboard, QualityDashboardData)

    def test_register_custom_assessor(self):
        """测试注册自定义评估器."""
        manager = QualityManager()

        # 创建自定义评估器
        class CustomAssessor(AccuracyAssessor):
            def assess(self, entity, context=None):
                return 0.99, [type("M", (), {
                    "metric_id": "custom",
                    "metric_name": "自定义指标",
                    "dimension": self._dimension,
                    "score": 0.99,
                    "weight": 1.0,
                    "details": "自定义评估",
                    "evidence": [],
                })()]

        manager.register_assessor(
            QualityDimension.ACCURACY,
            CustomAssessor(),
        )

        entity = make_entity()
        result = manager.assess_entity(entity)

        # 自定义评估器应该生效
        assert result.quality_score.accuracy == 0.99

    def test_entity_quality_history(self):
        """测试实体质量历史."""
        manager = QualityManager()
        entity = make_entity()

        # 多次评估
        for _ in range(3):
            manager.assess_entity(entity)

        history = manager.get_entity_quality_history(entity.entity_id)
        assert len(history) == 3

    def test_audit_log(self):
        """测试审计日志."""
        manager = QualityManager()
        entity = make_entity()

        manager.assess_entity(entity)
        manager.record_provenance(
            entity_id=entity.entity_id,
            activity_type="manual_update",
        )

        logs = manager.get_audit_log()
        assert len(logs) >= 2

        # 按实体过滤
        entity_logs = manager.get_audit_log(entity_id=entity.entity_id)
        assert len(entity_logs) >= 2

    def test_recommendations_generation(self):
        """测试改进建议生成."""
        manager = QualityManager()

        # 低质量实体应有更多建议
        low_entity = make_low_quality_entity()
        low_result = manager.assess_entity(low_entity)
        assert len(low_result.recommendations) >= 2

        # 高质量实体建议较少
        high_entity = make_entity(
            quality=QualityScore(
                accuracy=0.95, trustworthiness=0.95, consistency=0.95,
                timeliness=0.95, completeness=0.90, relevancy=0.90,
                verification_status=VerificationStatus.VERIFIED,
                evidence_count=5,
                peer_reviewed=True,
            )
        )
        high_result = manager.assess_entity(high_entity)
        # 高质量实体也应该有建议 (至少一条)
        assert len(high_result.recommendations) >= 1


# ============================================================
# 6. QualityGrade 测试
# ============================================================


class TestQualityGrade:
    """质量等级测试."""

    def test_grade_from_score(self):
        """测试分数到等级的映射."""
        assert QualityGrade.from_score(0.95) == QualityGrade.EXCELLENT
        assert QualityGrade.from_score(0.85) == QualityGrade.GOOD
        assert QualityGrade.from_score(0.70) == QualityGrade.FAIR
        assert QualityGrade.from_score(0.50) == QualityGrade.POOR
        assert QualityGrade.from_score(0.30) == QualityGrade.UNACCEPTABLE

    def test_grade_boundaries(self):
        """测试等级边界值."""
        assert QualityGrade.from_score(0.9) == QualityGrade.EXCELLENT
        assert QualityGrade.from_score(0.89) == QualityGrade.GOOD
        assert QualityGrade.from_score(0.8) == QualityGrade.GOOD
        assert QualityGrade.from_score(0.79) == QualityGrade.FAIR
        assert QualityGrade.from_score(0.6) == QualityGrade.FAIR
        assert QualityGrade.from_score(0.59) == QualityGrade.POOR
        assert QualityGrade.from_score(0.4) == QualityGrade.POOR
        assert QualityGrade.from_score(0.39) == QualityGrade.UNACCEPTABLE


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
