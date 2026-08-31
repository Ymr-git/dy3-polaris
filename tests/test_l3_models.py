"""L3 领域知识层全面测试套件.

覆盖 3 个源文件:
- exceptions.py: 10 个异常类 (L3Error 基类 + 9 个子类)，错误码 -32400~-32409
- models.py: 10 个枚举 + 13 个 pydantic 模型
- ontology.py: 本体定义 + 注册中心

测试类别:
1.  异常体系 (TestL3Exceptions)
2.  枚举定义 (TestEnums)
3.  KnowledgeSource 数据源 (TestKnowledgeSource)
4.  KnowledgeQualifier 限定符 (TestKnowledgeQualifier)
5.  KnowledgeTriple 三元组 (TestKnowledgeTriple)
6.  KnowledgeEntity 知识实体 (TestKnowledgeEntity)
7.  DocumentChunk 文档切片 (TestDocumentChunk)
8.  QualityScore 质量评分 (TestQualityScore)
9.  ProvenanceInfo 溯源信息 (TestProvenanceInfo)
10. EmbeddingVector 向量 (TestEmbeddingVector)
11. RetrievalResult 检索结果 (TestRetrievalResult)
12. KnowledgeBaseStats 知识库统计 (TestKnowledgeBaseStats)
13. IngestResult 导入结果 (TestIngestResult)
14. RetrievalFilter 检索过滤器 (TestRetrievalFilter)
15. OntologyProperty 本体属性 (TestOntologyProperty)
16. OntologyRelation 本体关系 (TestOntologyRelation)
17. OntologyClass 本体类 (TestOntologyClass)
18. DomainOntology 领域本体 (TestDomainOntology)
19. OntologyRegistry 本体注册中心 (TestOntologyRegistry)
20. 预构建本体内容验证 (TestPrebuiltOntologies)
21. 集成场景测试 (TestIntegrationScenarios)
"""

from __future__ import annotations

import logging
import time

logging.disable(logging.CRITICAL)

import pytest
from pydantic import ValidationError

from dy3_polaris.l3.exceptions import (
    ChunkingError,
    DuplicateEntityError,
    EmbeddingError,
    EntityNotFoundError,
    IngestError,
    L3Error,
    OntologyValidationError,
    ProvenanceError,
    QualityAssessmentError,
    RetrievalError,
)
from dy3_polaris.l3.models import (
    AccessLevel,
    ChunkRelationship,
    ChunkRelationshipType,
    ChunkingStrategy,
    ContentModality,
    DocumentChunk,
    EmbeddingVector,
    EntityType,
    IngestResult,
    KnowledgeBaseStats,
    KnowledgeEntity,
    KnowledgeQualifier,
    KnowledgeSource,
    KnowledgeTriple,
    ProvenanceInfo,
    ProvenanceRole,
    QualityDimension,
    QualityScore,
    RelationType,
    RetrievalFilter,
    RetrievalResult,
    SourceTier,
    StatementRank,
)
from dy3_polaris.l3.ontology import (
    DomainOntology,
    DomainType,
    OntologyClass,
    OntologyProperty,
    OntologyRegistry,
    OntologyRelation,
)
from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 1. 异常测试
# ============================================================


class TestL3Exceptions:
    """L3 异常体系测试 — 验证 10 个异常类的实例化、属性、错误码和继承关系."""

    # --- L3Error 基类 ---

    def test_l3_error_继承_l6_error(self) -> None:
        """L3Error 应继承 L6Error."""
        assert issubclass(L3Error, L6Error)
        assert issubclass(L3Error, Exception)

    def test_l3_error_默认构造(self) -> None:
        """L3Error 默认构造应使用默认 code 和空 detail/context."""
        err = L3Error()
        assert err.code == "L3_ERROR"
        assert err.detail == ""
        assert err.context == {}

    def test_l3_error_带详情和上下文(self) -> None:
        """L3Error 应正确存储 detail 和 context."""
        err = L3Error(detail="测试错误", context={"key": "value"})
        assert err.detail == "测试错误"
        assert err.context == {"key": "value"}

    def test_l3_error_jsonrpc_code(self) -> None:
        """L3Error 的 jsonrpc_code 应为 -32400."""
        err = L3Error()
        assert err._jsonrpc_code() == -32400

    def test_l3_error_to_json_rpc_error_有数据(self) -> None:
        """L3Error 转 JSON-RPC 错误对象 (有 detail 和 context)."""
        err = L3Error(detail="测试", context={"k": "v"})
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32400
        assert rpc["message"] == "L3_ERROR"
        assert rpc["data"]["detail"] == "测试"
        assert rpc["data"]["k"] == "v"

    def test_l3_error_to_json_rpc_error_无数据(self) -> None:
        """L3Error 无 detail 和 context 时 data 应为 None."""
        err = L3Error()
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32400
        assert rpc["message"] == "L3_ERROR"
        assert rpc["data"] is None

    def test_l3_error_可被抛出和捕获(self) -> None:
        """L3Error 应能被 try/except 捕获."""
        with pytest.raises(L3Error) as exc_info:
            raise L3Error(detail="抛出测试")
        assert "抛出测试" in str(exc_info.value)

    def test_l3_error_可被_l6_error捕获(self) -> None:
        """L3Error 应能被 L6Error 捕获 (继承关系)."""
        with pytest.raises(L6Error):
            raise L3Error(detail="继承捕获")

    # --- EntityNotFoundError ---

    def test_entity_not_found_error_基本属性(self) -> None:
        """EntityNotFoundError 应正确设置 entity_id 和 code."""
        err = EntityNotFoundError("e-123")
        assert err.entity_id == "e-123"
        assert err.code == "L3_ENTITY_NOT_FOUND"

    def test_entity_not_found_error_jsonrpc_code(self) -> None:
        """EntityNotFoundError 的 jsonrpc_code 应为 -32401."""
        err = EntityNotFoundError("e-123")
        assert err._jsonrpc_code() == -32401

    def test_entity_not_found_error_继承关系(self) -> None:
        """EntityNotFoundError 应继承 L3Error 和 L6Error."""
        assert issubclass(EntityNotFoundError, L3Error)
        assert issubclass(EntityNotFoundError, L6Error)

    def test_entity_not_found_error_默认detail(self) -> None:
        """EntityNotFoundError 默认 detail 应包含 entity_id."""
        err = EntityNotFoundError("e-456")
        assert "e-456" in err.detail

    def test_entity_not_found_error_自定义detail(self) -> None:
        """EntityNotFoundError 应支持自定义 detail."""
        err = EntityNotFoundError("e-789", detail="自定义详情")
        assert err.detail == "自定义详情"

    def test_entity_not_found_error_上下文传递(self) -> None:
        """EntityNotFoundError 应将 entity_id 和额外 context 合并."""
        err = EntityNotFoundError("e-001", context={"extra": "info"})
        assert err.context["entity_id"] == "e-001"
        assert err.context["extra"] == "info"

    def test_entity_not_found_error_json_rpc_error(self) -> None:
        """EntityNotFoundError 的 JSON-RPC 错误对象."""
        err = EntityNotFoundError("e-002")
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32401
        assert rpc["message"] == "L3_ENTITY_NOT_FOUND"
        assert rpc["data"]["entity_id"] == "e-002"

    # --- DuplicateEntityError ---

    def test_duplicate_entity_error_基本属性(self) -> None:
        """DuplicateEntityError 应正确设置 entity_id 和 code."""
        err = DuplicateEntityError("e-dup")
        assert err.entity_id == "e-dup"
        assert err.code == "L3_DUPLICATE_ENTITY"

    def test_duplicate_entity_error_jsonrpc_code(self) -> None:
        """DuplicateEntityError 的 jsonrpc_code 应为 -32402."""
        err = DuplicateEntityError("e-dup")
        assert err._jsonrpc_code() == -32402

    def test_duplicate_entity_error_继承关系(self) -> None:
        """DuplicateEntityError 应继承 L3Error."""
        assert issubclass(DuplicateEntityError, L3Error)

    def test_duplicate_entity_error_上下文传递(self) -> None:
        """DuplicateEntityError 应将 entity_id 放入 context."""
        err = DuplicateEntityError("e-dup", context={"field": "name"})
        assert err.context["entity_id"] == "e-dup"
        assert err.context["field"] == "name"

    # --- OntologyValidationError ---

    def test_ontology_validation_error_基本属性(self) -> None:
        """OntologyValidationError 应正确设置 entity_type 和 violation."""
        err = OntologyValidationError(entity_type="compound", violation="missing_field")
        assert err.entity_type == "compound"
        assert err.violation == "missing_field"
        assert err.code == "L3_ONTOLOGY_VALIDATION"

    def test_ontology_validation_error_jsonrpc_code(self) -> None:
        """OntologyValidationError 的 jsonrpc_code 应为 -32403."""
        err = OntologyValidationError("compound", "missing")
        assert err._jsonrpc_code() == -32403

    def test_ontology_validation_error_上下文传递(self) -> None:
        """OntologyValidationError 应将 entity_type 和 violation 放入 context."""
        err = OntologyValidationError("compound", "missing", context={"field": "formula"})
        assert err.context["entity_type"] == "compound"
        assert err.context["violation"] == "missing"
        assert err.context["field"] == "formula"

    # --- QualityAssessmentError ---

    def test_quality_assessment_error_基本属性(self) -> None:
        """QualityAssessmentError 应正确设置 dimension."""
        err = QualityAssessmentError(dimension="accuracy")
        assert err.dimension == "accuracy"
        assert err.code == "L3_QUALITY_ASSESSMENT"

    def test_quality_assessment_error_jsonrpc_code(self) -> None:
        """QualityAssessmentError 的 jsonrpc_code 应为 -32404."""
        err = QualityAssessmentError("accuracy")
        assert err._jsonrpc_code() == -32404

    def test_quality_assessment_error_上下文传递(self) -> None:
        """QualityAssessmentError 应将 dimension 放入 context."""
        err = QualityAssessmentError("accuracy", context={"score": 0.3})
        assert err.context["dimension"] == "accuracy"
        assert err.context["score"] == 0.3

    # --- ProvenanceError ---

    def test_provenance_error_基本属性(self) -> None:
        """ProvenanceError 应正确设置 entity_id 和 chain_break."""
        err = ProvenanceError(entity_id="e-1", chain_break="hash_mismatch")
        assert err.entity_id == "e-1"
        assert err.chain_break == "hash_mismatch"
        assert err.code == "L3_PROVENANCE"

    def test_provenance_error_jsonrpc_code(self) -> None:
        """ProvenanceError 的 jsonrpc_code 应为 -32405."""
        err = ProvenanceError("e-1", "break")
        assert err._jsonrpc_code() == -32405

    def test_provenance_error_上下文传递(self) -> None:
        """ProvenanceError 应将 entity_id 和 chain_break 放入 context."""
        err = ProvenanceError("e-1", "break", context={"depth": 3})
        assert err.context["entity_id"] == "e-1"
        assert err.context["chain_break"] == "break"
        assert err.context["depth"] == 3

    # --- ChunkingError ---

    def test_chunking_error_基本属性(self) -> None:
        """ChunkingError 应正确设置 document_id 和 strategy."""
        err = ChunkingError(document_id="doc-1", strategy="semantic")
        assert err.document_id == "doc-1"
        assert err.strategy == "semantic"
        assert err.code == "L3_CHUNKING"

    def test_chunking_error_jsonrpc_code(self) -> None:
        """ChunkingError 的 jsonrpc_code 应为 -32406."""
        err = ChunkingError("doc-1", "semantic")
        assert err._jsonrpc_code() == -32406

    # --- EmbeddingError ---

    def test_embedding_error_基本属性(self) -> None:
        """EmbeddingError 应正确设置 content_id 和 model."""
        err = EmbeddingError(content_id="c-1", model="text-embedding-3")
        assert err.content_id == "c-1"
        assert err.model == "text-embedding-3"
        assert err.code == "L3_EMBEDDING"

    def test_embedding_error_jsonrpc_code(self) -> None:
        """EmbeddingError 的 jsonrpc_code 应为 -32407."""
        err = EmbeddingError("c-1", "model-x")
        assert err._jsonrpc_code() == -32407

    # --- RetrievalError ---

    def test_retrieval_error_基本属性(self) -> None:
        """RetrievalError 应正确设置 query 和 reason."""
        err = RetrievalError(query="水", reason="timeout")
        assert err.query == "水"
        assert err.reason == "timeout"
        assert err.code == "L3_RETRIEVAL"

    def test_retrieval_error_jsonrpc_code(self) -> None:
        """RetrievalError 的 jsonrpc_code 应为 -32408."""
        err = RetrievalError("query", "reason")
        assert err._jsonrpc_code() == -32408

    # --- IngestError ---

    def test_ingest_error_基本属性(self) -> None:
        """IngestError 应正确设置 source 和 count."""
        err = IngestError(source="pdf", count=5)
        assert err.source == "pdf"
        assert err.count == 5
        assert err.code == "L3_INGEST"

    def test_ingest_error_jsonrpc_code(self) -> None:
        """IngestError 的 jsonrpc_code 应为 -32409."""
        err = IngestError("pdf", 5)
        assert err._jsonrpc_code() == -32409

    # --- 综合验证 ---

    @pytest.mark.parametrize(
        "exc_cls,expected_code,expected_jsonrpc",
        [
            (L3Error, "L3_ERROR", -32400),
            (EntityNotFoundError, "L3_ENTITY_NOT_FOUND", -32401),
            (DuplicateEntityError, "L3_DUPLICATE_ENTITY", -32402),
            (OntologyValidationError, "L3_ONTOLOGY_VALIDATION", -32403),
            (QualityAssessmentError, "L3_QUALITY_ASSESSMENT", -32404),
            (ProvenanceError, "L3_PROVENANCE", -32405),
            (ChunkingError, "L3_CHUNKING", -32406),
            (EmbeddingError, "L3_EMBEDDING", -32407),
            (RetrievalError, "L3_RETRIEVAL", -32408),
            (IngestError, "L3_INGEST", -32409),
        ],
    )
    def test_所有异常_继承_l3_error(self, exc_cls, expected_code, expected_jsonrpc) -> None:
        """所有 L3 异常都应继承 L3Error."""
        assert issubclass(exc_cls, L3Error)

    def test_所有jsonrpc_code_唯一且在范围内(self) -> None:
        """所有 10 个异常的 jsonrpc_code 应唯一且在 -32400~-32409 范围内."""
        codes = [
            L3Error()._jsonrpc_code(),
            EntityNotFoundError("e")._jsonrpc_code(),
            DuplicateEntityError("e")._jsonrpc_code(),
            OntologyValidationError()._jsonrpc_code(),
            QualityAssessmentError()._jsonrpc_code(),
            ProvenanceError()._jsonrpc_code(),
            ChunkingError()._jsonrpc_code(),
            EmbeddingError()._jsonrpc_code(),
            RetrievalError()._jsonrpc_code(),
            IngestError()._jsonrpc_code(),
        ]
        assert len(set(codes)) == 10  # 唯一
        assert all(-32409 <= c <= -32400 for c in codes)  # 范围


# ============================================================
# 2. 枚举测试
# ============================================================


class TestEnums:
    """验证所有 10 个枚举的定义、值和 str+Enum 混合继承."""

    # --- ContentModality ---

    @pytest.mark.parametrize(
        "member,expected",
        [
            (ContentModality.TEXT, "text"),
            (ContentModality.IMAGE, "image"),
            (ContentModality.TABLE, "table"),
            (ContentModality.EQUATION, "equation"),
            (ContentModality.CODE, "code"),
            (ContentModality.MIXED, "mixed"),
        ],
    )
    def test_content_modality_枚举值(self, member, expected) -> None:
        assert member.value == expected

    def test_content_modality_成员数量(self) -> None:
        assert len(ContentModality) == 6

    def test_content_modality_str继承(self) -> None:
        assert issubclass(ContentModality, str)
        assert ContentModality.TEXT == "text"

    # --- EntityType ---

    @pytest.mark.parametrize(
        "member,expected",
        [
            (EntityType.CONCEPT, "concept"),
            (EntityType.CHEMICAL_COMPOUND, "chemical_compound"),
            (EntityType.MATERIAL, "material"),
            (EntityType.PAPER, "paper"),
            (EntityType.TEXTBOOK, "textbook"),
            (EntityType.DATASET, "dataset"),
            (EntityType.METHOD, "method"),
            (EntityType.PERSON, "person"),
            (EntityType.ORGANIZATION, "organization"),
            (EntityType.DOCUMENT_CHUNK, "document_chunk"),
            (EntityType.COURSE, "course"),
            (EntityType.EXPERIMENT, "experiment"),
            (EntityType.TOPIC, "topic"),
            (EntityType.KNOWLEDGE_POINT, "knowledge_point"),
            (EntityType.FACT, "fact"),
            (EntityType.ROLE, "role"),
            (EntityType.QUESTION, "question"),
            (EntityType.ION, "ion"),
            (EntityType.ENERGY_LEVEL, "energy_level"),
            (EntityType.PARAMETER, "parameter"),
        ],
    )
    def test_entity_type_枚举值(self, member, expected) -> None:
        assert member.value == expected

    def test_entity_type_成员数量(self) -> None:
        assert len(EntityType) == 20

    def test_entity_type_str继承(self) -> None:
        assert issubclass(EntityType, str)
        assert EntityType.CHEMICAL_COMPOUND == "chemical_compound"

    # --- RelationType ---

    @pytest.mark.parametrize(
        "member,expected",
        [
            (RelationType.CITES, "cites"),
            (RelationType.DERIVED_FROM, "derived_from"),
            (RelationType.PART_OF, "part_of"),
            (RelationType.RELATED_TO, "related_to"),
            (RelationType.AUTHORED_BY, "authored_by"),
            (RelationType.PUBLISHED_IN, "published_in"),
            (RelationType.HAS_PROPERTY, "has_property"),
            (RelationType.CONTRADICTS, "contradicts"),
            (RelationType.SUPPORTS, "supports"),
            (RelationType.EQUIVALENT_TO, "equivalent_to"),
            (RelationType.DEPENDS_ON, "depends_on"),
            (RelationType.INSTANTIATES, "instantiates"),
            (RelationType.SUPERSEDES, "supersedes"),
            (RelationType.REFERENCES, "references"),
            (RelationType.PREREQUISITE_OF, "prerequisite_of"),
            (RelationType.DEEPENS, "deepens"),
            (RelationType.ANALOGOUS_TO, "analogous_to"),
            (RelationType.AFFECTS, "affects"),
            (RelationType.CHARACTERIZED_BY, "characterized_by"),
            (RelationType.SUBCONCEPT_OF, "subconcept_of"),
            (RelationType.APPLIES_TO, "applies_to"),
            (RelationType.MENTIONS, "mentions"),
            (RelationType.MEASURED_BY, "measured_by"),
            (RelationType.DOPED_WITH, "doped_with"),
        ],
    )
    def test_relation_type_枚举值(self, member, expected) -> None:
        assert member.value == expected

    def test_relation_type_成员数量(self) -> None:
        assert len(RelationType) == 24

    def test_relation_type_str继承(self) -> None:
        assert issubclass(RelationType, str)

    # --- SourceTier ---

    @pytest.mark.parametrize(
        "member,expected",
        [
            (SourceTier.TIER1_PUBLIC, "tier1_public"),
            (SourceTier.TIER2_INDUSTRY, "tier2_industry"),
            (SourceTier.TIER3_CAMPUS, "tier3_campus"),
            (SourceTier.INTERNAL_DOCUMENT, "internal_document"),
        ],
    )
    def test_source_tier_枚举值(self, member, expected) -> None:
        assert member.value == expected

    def test_source_tier_成员数量(self) -> None:
        assert len(SourceTier) == 4

    def test_source_tier_str继承(self) -> None:
        assert issubclass(SourceTier, str)

    # --- QualityDimension ---

    @pytest.mark.parametrize(
        "member,expected",
        [
            (QualityDimension.ACCURACY, "accuracy"),
            (QualityDimension.TRUSTWORTHINESS, "trustworthiness"),
            (QualityDimension.CONSISTENCY, "consistency"),
            (QualityDimension.TIMELINESS, "timeliness"),
            (QualityDimension.COMPLETENESS, "completeness"),
            (QualityDimension.RELEVANCY, "relevancy"),
        ],
    )
    def test_quality_dimension_枚举值(self, member, expected) -> None:
        assert member.value == expected

    def test_quality_dimension_成员数量(self) -> None:
        assert len(QualityDimension) == 6

    def test_quality_dimension_str继承(self) -> None:
        assert issubclass(QualityDimension, str)

    # --- AccessLevel ---

    @pytest.mark.parametrize(
        "member,expected",
        [
            (AccessLevel.PUBLIC, "public"),
            (AccessLevel.INTERNAL, "internal"),
            (AccessLevel.RESTRICTED, "restricted"),
            (AccessLevel.CONFIDENTIAL, "confidential"),
        ],
    )
    def test_access_level_枚举值(self, member, expected) -> None:
        assert member.value == expected

    def test_access_level_成员数量(self) -> None:
        assert len(AccessLevel) == 4

    def test_access_level_str继承(self) -> None:
        assert issubclass(AccessLevel, str)

    # --- ChunkingStrategy ---

    @pytest.mark.parametrize(
        "member,expected",
        [
            (ChunkingStrategy.FIXED_LENGTH, "fixed_length"),
            (ChunkingStrategy.SEMANTIC_PARAGRAPH, "semantic_paragraph"),
            (ChunkingStrategy.RECURSIVE_CHAR, "recursive_char"),
            (ChunkingStrategy.STRUCTURED_HEADING, "structured_heading"),
        ],
    )
    def test_chunking_strategy_枚举值(self, member, expected) -> None:
        assert member.value == expected

    def test_chunking_strategy_成员数量(self) -> None:
        assert len(ChunkingStrategy) == 4

    def test_chunking_strategy_str继承(self) -> None:
        assert issubclass(ChunkingStrategy, str)

    # --- StatementRank ---

    @pytest.mark.parametrize(
        "member,expected",
        [
            (StatementRank.PREFERRED, "preferred"),
            (StatementRank.NORMAL, "normal"),
            (StatementRank.DEPRECATED, "deprecated"),
        ],
    )
    def test_statement_rank_枚举值(self, member, expected) -> None:
        assert member.value == expected

    def test_statement_rank_成员数量(self) -> None:
        assert len(StatementRank) == 3

    def test_statement_rank_str继承(self) -> None:
        assert issubclass(StatementRank, str)

    # --- ChunkRelationshipType ---

    @pytest.mark.parametrize(
        "member,expected",
        [
            (ChunkRelationshipType.PARENT, "parent"),
            (ChunkRelationshipType.CHILD, "child"),
            (ChunkRelationshipType.PREVIOUS, "previous"),
            (ChunkRelationshipType.NEXT, "next"),
            (ChunkRelationshipType.SOURCE, "source"),
        ],
    )
    def test_chunk_relationship_type_枚举值(self, member, expected) -> None:
        assert member.value == expected

    def test_chunk_relationship_type_成员数量(self) -> None:
        assert len(ChunkRelationshipType) == 5

    def test_chunk_relationship_type_str继承(self) -> None:
        assert issubclass(ChunkRelationshipType, str)

    # --- ProvenanceRole ---

    @pytest.mark.parametrize(
        "member,expected",
        [
            (ProvenanceRole.GENERATOR, "generator"),
            (ProvenanceRole.CONTRIBUTOR, "contributor"),
            (ProvenanceRole.VALIDATOR, "validator"),
            (ProvenanceRole.CURATOR, "curator"),
        ],
    )
    def test_provenance_role_枚举值(self, member, expected) -> None:
        assert member.value == expected

    def test_provenance_role_成员数量(self) -> None:
        assert len(ProvenanceRole) == 4

    def test_provenance_role_str继承(self) -> None:
        assert issubclass(ProvenanceRole, str)


# ============================================================
# 3. KnowledgeSource 测试
# ============================================================


class TestKnowledgeSource:
    """KnowledgeSource 数据源元数据测试."""

    def test_创建_必填字段(self) -> None:
        """创建时 source_id 和 name 为必填."""
        src = KnowledgeSource(source_id="nist", name="NIST WebBook")
        assert src.source_id == "nist"
        assert src.name == "NIST WebBook"

    def test_默认值(self) -> None:
        """验证默认值: tier, endpoint, auth_required, access_level, reliability, last_synced."""
        src = KnowledgeSource(source_id="test", name="Test")
        assert src.tier == SourceTier.INTERNAL_DOCUMENT
        assert src.endpoint == ""
        assert src.auth_required is False
        assert src.access_level == AccessLevel.INTERNAL
        assert src.reliability == 0.8
        assert src.last_synced == 0.0
        assert src.metadata == {}

    def test_创建_全部字段(self) -> None:
        """创建时指定所有字段."""
        src = KnowledgeSource(
            source_id="pubchem",
            name="PubChem",
            tier=SourceTier.TIER1_PUBLIC,
            endpoint="https://pubchem.ncbi.nlm.nih.gov",
            auth_required=False,
            access_level=AccessLevel.PUBLIC,
            reliability=0.95,
            last_synced=time.time(),
            metadata={"version": "1.0"},
        )
        assert src.tier == SourceTier.TIER1_PUBLIC
        assert src.access_level == AccessLevel.PUBLIC
        assert src.reliability == 0.95
        assert src.metadata["version"] == "1.0"

    def test_reliability_范围下限(self) -> None:
        """reliability=0.0 应合法."""
        src = KnowledgeSource(source_id="t", name="t", reliability=0.0)
        assert src.reliability == 0.0

    def test_reliability_范围上限(self) -> None:
        """reliability=1.0 应合法."""
        src = KnowledgeSource(source_id="t", name="t", reliability=1.0)
        assert src.reliability == 1.0

    def test_reliability_超出上限(self) -> None:
        """reliability>1.0 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeSource(source_id="t", name="t", reliability=1.1)

    def test_reliability_超出下限(self) -> None:
        """reliability<0.0 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeSource(source_id="t", name="t", reliability=-0.1)

    def test_is_stale_默认过期(self) -> None:
        """last_synced=0.0 时应判定为过期."""
        src = KnowledgeSource(source_id="t", name="t")
        assert src.is_stale() is True

    def test_is_stale_刚同步(self) -> None:
        """last_synced 为当前时间时应未过期."""
        src = KnowledgeSource(source_id="t", name="t", last_synced=time.time())
        assert src.is_stale() is False

    def test_is_stale_过期同步(self) -> None:
        """last_synced 超过 24 小时应判定为过期."""
        src = KnowledgeSource(source_id="t", name="t", last_synced=time.time() - 100000)
        assert src.is_stale() is True

    def test_is_stale_自定义阈值未过期(self) -> None:
        """自定义 max_age_seconds, 在阈值内未过期."""
        src = KnowledgeSource(source_id="t", name="t", last_synced=time.time() - 10)
        assert src.is_stale(max_age_seconds=60) is False

    def test_is_stale_自定义阈值已过期(self) -> None:
        """自定义 max_age_seconds, 超过阈值过期."""
        src = KnowledgeSource(source_id="t", name="t", last_synced=time.time() - 10)
        assert src.is_stale(max_age_seconds=5) is True

    def test_metadata_独立实例(self) -> None:
        """两个实例的 metadata 应相互独立."""
        s1 = KnowledgeSource(source_id="t1", name="t1", metadata={"k": "v1"})
        s2 = KnowledgeSource(source_id="t2", name="t2", metadata={"k": "v2"})
        assert s1.metadata["k"] == "v1"
        assert s2.metadata["k"] == "v2"


# ============================================================
# 4. KnowledgeQualifier 测试
# ============================================================


class TestKnowledgeQualifier:
    """KnowledgeQualifier 声明限定符测试."""

    def test_创建_基本(self) -> None:
        """创建限定符."""
        q = KnowledgeQualifier(name="condition", value="1 atm")
        assert q.name == "condition"
        assert q.value == "1 atm"

    def test_qualifier_id_自动生成(self) -> None:
        """qualifier_id 应自动生成且以 'q-' 开头."""
        q = KnowledgeQualifier(name="x", value="y")
        assert q.qualifier_id.startswith("q-")
        assert len(q.qualifier_id) == 10  # "q-" + 8 hex chars

    def test_qualifier_id_唯一性(self) -> None:
        """两个实例的 qualifier_id 应不同."""
        q1 = KnowledgeQualifier(name="a", value=1)
        q2 = KnowledgeQualifier(name="b", value=2)
        assert q1.qualifier_id != q2.qualifier_id

    def test_qualifier_id_手动指定(self) -> None:
        """支持手动指定 qualifier_id."""
        q = KnowledgeQualifier(qualifier_id="q-custom", name="x", value="y")
        assert q.qualifier_id == "q-custom"

    def test_value_type_默认值(self) -> None:
        """value_type 默认为 'string'."""
        q = KnowledgeQualifier(name="x", value="y")
        assert q.value_type == "string"

    def test_value_type_自定义(self) -> None:
        """支持自定义 value_type."""
        q = KnowledgeQualifier(name="temp", value=100, value_type="number")
        assert q.value_type == "number"

    def test_value_支持任意类型(self) -> None:
        """value 应支持字符串、数字、布尔值等."""
        q_str = KnowledgeQualifier(name="s", value="text")
        q_num = KnowledgeQualifier(name="n", value=42)
        q_bool = KnowledgeQualifier(name="b", value=True)
        assert q_str.value == "text"
        assert q_num.value == 42
        assert q_bool.value is True

    def test_缺少name_校验失败(self) -> None:
        """缺少 name 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeQualifier(value="y")


# ============================================================
# 5. KnowledgeTriple 测试
# ============================================================


class TestKnowledgeTriple:
    """KnowledgeTriple SPO 三元组测试."""

    def test_创建_实体引用三元组(self) -> None:
        """创建宾语为实体引用的三元组."""
        t = KnowledgeTriple(
            subject_id="e-1",
            predicate="cites",
            object_id="e-2",
        )
        assert t.subject_id == "e-1"
        assert t.predicate == "cites"
        assert t.object_id == "e-2"
        assert t.object_is_literal is False

    def test_创建_字面值三元组(self) -> None:
        """创建宾语为字面值的三元组."""
        t = KnowledgeTriple(
            subject_id="e-1",
            predicate="has_property",
            object_value="H2O",
            object_is_literal=True,
        )
        assert t.object_value == "H2O"
        assert t.object_is_literal is True
        assert t.object_id == ""

    def test_triple_id_自动生成(self) -> None:
        """triple_id 应自动生成且以 't-' 开头."""
        t = KnowledgeTriple(subject_id="e-1", predicate="p", object_id="e-2")
        assert t.triple_id.startswith("t-")

    def test_默认rank为normal(self) -> None:
        """默认 rank 应为 NORMAL."""
        t = KnowledgeTriple(subject_id="e-1", predicate="p", object_id="e-2")
        assert t.rank == StatementRank.NORMAL

    def test_默认confidence为1(self) -> None:
        """默认 confidence 应为 1.0."""
        t = KnowledgeTriple(subject_id="e-1", predicate="p", object_id="e-2")
        assert t.confidence == 1.0

    def test_实体引用_object_id为空_校验失败(self) -> None:
        """object_is_literal=False 且 object_id 为空应抛出 ValueError."""
        with pytest.raises(ValidationError):
            KnowledgeTriple(subject_id="e-1", predicate="p")

    def test_字面值_object_value为None_校验失败(self) -> None:
        """object_is_literal=True 且 object_value 为 None 应抛出 ValueError."""
        with pytest.raises(ValidationError):
            KnowledgeTriple(
                subject_id="e-1",
                predicate="p",
                object_is_literal=True,
            )

    def test_字面值_object_value为空字符串_合法(self) -> None:
        """object_is_literal=True 且 object_value='' 应合法 (空字符串非 None)."""
        t = KnowledgeTriple(
            subject_id="e-1",
            predicate="p",
            object_value="",
            object_is_literal=True,
        )
        assert t.object_value == ""

    def test_字面值_object_value为零_合法(self) -> None:
        """object_is_literal=True 且 object_value=0 应合法 (0 非 None)."""
        t = KnowledgeTriple(
            subject_id="e-1",
            predicate="p",
            object_value=0,
            object_is_literal=True,
        )
        assert t.object_value == 0

    def test_is_preferred_首选声明(self) -> None:
        """rank=PREFERRED 时 is_preferred() 应为 True."""
        t = KnowledgeTriple(
            subject_id="e-1", predicate="p", object_id="e-2",
            rank=StatementRank.PREFERRED,
        )
        assert t.is_preferred() is True
        assert t.is_deprecated() is False

    def test_is_deprecated_弃用声明(self) -> None:
        """rank=DEPRECATED 时 is_deprecated() 应为 True."""
        t = KnowledgeTriple(
            subject_id="e-1", predicate="p", object_id="e-2",
            rank=StatementRank.DEPRECATED,
        )
        assert t.is_deprecated() is True
        assert t.is_preferred() is False

    def test_normal_rank_既非preferred也非deprecated(self) -> None:
        """rank=NORMAL 时 is_preferred() 和 is_deprecated() 都为 False."""
        t = KnowledgeTriple(subject_id="e-1", predicate="p", object_id="e-2")
        assert t.is_preferred() is False
        assert t.is_deprecated() is False

    def test_qualifiers_列表(self) -> None:
        """三元组应支持携带 qualifiers 列表."""
        q1 = KnowledgeQualifier(name="condition", value="1 atm")
        q2 = KnowledgeQualifier(name="source", value="NIST")
        t = KnowledgeTriple(
            subject_id="e-1", predicate="p", object_id="e-2",
            qualifiers=[q1, q2],
        )
        assert len(t.qualifiers) == 2
        assert t.qualifiers[0].name == "condition"
        assert t.qualifiers[1].name == "source"

    def test_qualifiers_默认空列表(self) -> None:
        """qualifiers 默认为空列表."""
        t = KnowledgeTriple(subject_id="e-1", predicate="p", object_id="e-2")
        assert t.qualifiers == []

    def test_confidence_范围验证(self) -> None:
        """confidence 超出 0.0~1.0 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeTriple(subject_id="e-1", predicate="p", object_id="e-2", confidence=1.5)
        with pytest.raises(ValidationError):
            KnowledgeTriple(subject_id="e-1", predicate="p", object_id="e-2", confidence=-0.1)

    def test_created_at_自动填充(self) -> None:
        """created_at 应自动填充为当前时间戳."""
        before = time.time()
        t = KnowledgeTriple(subject_id="e-1", predicate="p", object_id="e-2")
        after = time.time()
        assert before <= t.created_at <= after


# ============================================================
# 6. KnowledgeEntity 测试
# ============================================================


class TestKnowledgeEntity:
    """KnowledgeEntity 知识实体测试."""

    def test_创建_必填字段(self) -> None:
        """entity_type 和 name 为必填."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="概念A")
        assert e.entity_type == EntityType.CONCEPT
        assert e.name == "概念A"

    def test_entity_id_自动生成(self) -> None:
        """entity_id 应自动生成且以 'e-' 开头."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        assert e.entity_id.startswith("e-")

    def test_entity_id_唯一性(self) -> None:
        """两个实例的 entity_id 应不同."""
        e1 = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="a")
        e2 = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="b")
        assert e1.entity_id != e2.entity_id

    def test_默认值(self) -> None:
        """验证默认值: domain, access_level, version, triples 等."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        assert e.domain == "general"
        assert e.access_level == AccessLevel.INTERNAL
        assert e.version == 1
        assert e.triples == []
        assert e.identifiers == {}
        assert e.properties == {}
        assert e.parent_entity_id == ""
        assert e.source is None
        assert e.quality is None
        assert e.provenance is None

    def test_name_不能为空(self) -> None:
        """name 为空字符串应抛出 ValidationError (min_length=1)."""
        with pytest.raises(ValidationError):
            KnowledgeEntity(entity_type=EntityType.CONCEPT, name="")

    def test_add_triple_添加三元组(self) -> None:
        """add_triple 应将三元组添加到列表."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        t = KnowledgeTriple(subject_id=e.entity_id, predicate="p", object_id="e-2")
        e.add_triple(t)
        assert len(e.triples) == 1
        assert e.triples[0] is t

    def test_add_triple_修正subject_id(self) -> None:
        """add_triple 应将 triple 的 subject_id 修正为实体的 entity_id."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        t = KnowledgeTriple(subject_id="wrong-id", predicate="p", object_id="e-2")
        e.add_triple(t)
        assert t.subject_id == e.entity_id

    def test_add_triple_更新时间戳(self) -> None:
        """add_triple 应更新 updated_at 时间戳."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        old_updated = e.updated_at
        time.sleep(0.01)
        t = KnowledgeTriple(subject_id=e.entity_id, predicate="p", object_id="e-2")
        e.add_triple(t)
        assert e.updated_at > old_updated

    def test_get_triples_by_predicate(self) -> None:
        """按谓词筛选三元组."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        t1 = KnowledgeTriple(subject_id=e.entity_id, predicate="cites", object_id="e-2")
        t2 = KnowledgeTriple(subject_id=e.entity_id, predicate="has_property", object_id="e-3")
        t3 = KnowledgeTriple(subject_id=e.entity_id, predicate="cites", object_id="e-4")
        e.add_triple(t1)
        e.add_triple(t2)
        e.add_triple(t3)
        cites = e.get_triples_by_predicate("cites")
        assert len(cites) == 2
        assert all(t.predicate == "cites" for t in cites)

    def test_get_preferred_triples(self) -> None:
        """获取所有首选声明."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        t_pref = KnowledgeTriple(
            subject_id=e.entity_id, predicate="p", object_id="e-2",
            rank=StatementRank.PREFERRED,
        )
        t_norm = KnowledgeTriple(
            subject_id=e.entity_id, predicate="p", object_id="e-3",
            rank=StatementRank.NORMAL,
        )
        e.add_triple(t_pref)
        e.add_triple(t_norm)
        preferred = e.get_preferred_triples()
        assert len(preferred) == 1
        assert preferred[0].rank == StatementRank.PREFERRED

    def test_get_active_triples(self) -> None:
        """获取所有非弃用声明."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        t_norm = KnowledgeTriple(
            subject_id=e.entity_id, predicate="p", object_id="e-2",
            rank=StatementRank.NORMAL,
        )
        t_dep = KnowledgeTriple(
            subject_id=e.entity_id, predicate="p", object_id="e-3",
            rank=StatementRank.DEPRECATED,
        )
        t_pref = KnowledgeTriple(
            subject_id=e.entity_id, predicate="p", object_id="e-4",
            rank=StatementRank.PREFERRED,
        )
        e.add_triple(t_norm)
        e.add_triple(t_dep)
        e.add_triple(t_pref)
        active = e.get_active_triples()
        assert len(active) == 2  # NORMAL + PREFERRED, 排除 DEPRECATED

    def test_has_identifier_存在(self) -> None:
        """has_identifier 应返回 True 当标识符存在."""
        e = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="水",
            identifiers={"cas": "7732-18-5", "doi": "10.xxx"},
        )
        assert e.has_identifier("cas") is True
        assert e.has_identifier("doi") is True

    def test_has_identifier_不存在(self) -> None:
        """has_identifier 应返回 False 当标识符不存在."""
        e = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="水",
            identifiers={"cas": "7732-18-5"},
        )
        assert e.has_identifier("doi") is False

    def test_has_identifier_空映射(self) -> None:
        """has_identifier 应返回 False 当 identifiers 为空."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        assert e.has_identifier("cas") is False

    def test_is_newer_than_版本更高(self) -> None:
        """is_newer_than 应返回 True 当版本更高."""
        e1 = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", version=2)
        e2 = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", version=1)
        assert e1.is_newer_than(e2) is True

    def test_is_newer_than_版本更低(self) -> None:
        """is_newer_than 应返回 False 当版本更低."""
        e1 = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", version=1)
        e2 = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", version=2)
        assert e1.is_newer_than(e2) is False

    def test_is_newer_than_版本相同(self) -> None:
        """is_newer_than 应返回 False 当版本相同."""
        e1 = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", version=1)
        e2 = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", version=1)
        assert e1.is_newer_than(e2) is False

    def test_touch_更新时间戳(self) -> None:
        """touch 应更新 updated_at."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        old = e.updated_at
        time.sleep(0.01)
        e.touch()
        assert e.updated_at > old

    def test_创建_带source和quality(self) -> None:
        """创建时携带 source 和 quality."""
        src = KnowledgeSource(source_id="nist", name="NIST")
        qual = QualityScore(accuracy=0.9)
        e = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="水",
            source=src, quality=qual,
        )
        assert e.source is not None
        assert e.source.source_id == "nist"
        assert e.quality is not None
        assert e.quality.accuracy == 0.9


# ============================================================
# 7. DocumentChunk 测试 (含 ChunkRelationship)
# ============================================================


class TestChunkRelationship:
    """ChunkRelationship 切片间关系测试."""

    def test_创建_基本(self) -> None:
        rel = ChunkRelationship(
            relation_type=ChunkRelationshipType.PARENT,
            target_chunk_id="c-parent",
        )
        assert rel.relation_type == ChunkRelationshipType.PARENT
        assert rel.target_chunk_id == "c-parent"

    def test_target_metadata_默认空(self) -> None:
        rel = ChunkRelationship(
            relation_type=ChunkRelationshipType.NEXT,
            target_chunk_id="c-1",
        )
        assert rel.target_metadata == {}

    def test_target_metadata_自定义(self) -> None:
        rel = ChunkRelationship(
            relation_type=ChunkRelationshipType.CHILD,
            target_chunk_id="c-1",
            target_metadata={"section": "1.2"},
        )
        assert rel.target_metadata["section"] == "1.2"

    def test_缺少relation_type_校验失败(self) -> None:
        with pytest.raises(ValidationError):
            ChunkRelationship(target_chunk_id="c-1")

    def test_缺少target_chunk_id_校验失败(self) -> None:
        with pytest.raises(ValidationError):
            ChunkRelationship(relation_type=ChunkRelationshipType.PARENT)


class TestDocumentChunk:
    """DocumentChunk 文档切片测试."""

    def test_创建_必填字段(self) -> None:
        """document_id 和 content 为必填."""
        chunk = DocumentChunk(document_id="doc-1", content="测试内容")
        assert chunk.document_id == "doc-1"
        assert chunk.content == "测试内容"

    def test_chunk_id_自动生成(self) -> None:
        """chunk_id 应自动生成且以 'c-' 开头."""
        chunk = DocumentChunk(document_id="d", content="x")
        assert chunk.chunk_id.startswith("c-")

    def test_char_count_自动填充(self) -> None:
        """char_count 应自动填充为 content 长度."""
        chunk = DocumentChunk(document_id="d", content="hello world")
        assert chunk.char_count == 11

    def test_token_count_自动填充(self) -> None:
        """token_count 应自动填充为 max(1, len(content)//4)."""
        chunk = DocumentChunk(document_id="d", content="hello world")  # 11 chars
        assert chunk.token_count == max(1, 11 // 4)  # max(1, 2) = 2

    def test_token_count_短文本至少为1(self) -> None:
        """短文本的 token_count 至少为 1."""
        chunk = DocumentChunk(document_id="d", content="ab")  # 2 chars, 2//4=0
        assert chunk.token_count == 1

    def test_char_count_手动指定_保留(self) -> None:
        """手动指定 char_count 时应保留原值."""
        chunk = DocumentChunk(document_id="d", content="hello", char_count=999)
        assert chunk.char_count == 999

    def test_token_count_手动指定_保留(self) -> None:
        """手动指定 token_count 时应保留原值."""
        chunk = DocumentChunk(document_id="d", content="hello", token_count=42)
        assert chunk.token_count == 42

    def test_content_不能为空(self) -> None:
        """content 为空字符串应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            DocumentChunk(document_id="d", content="")

    def test_默认值(self) -> None:
        """验证默认值: content_type, chunk_index, page, strategy 等."""
        chunk = DocumentChunk(document_id="d", content="x")
        assert chunk.content_type == ContentModality.TEXT
        assert chunk.chunk_index == 0
        assert chunk.page == 0
        assert chunk.strategy == ChunkingStrategy.FIXED_LENGTH
        assert chunk.overlap_prev == 0
        assert chunk.relationships == []
        assert chunk.embedding is None
        assert chunk.quality is None

    def test_has_embedding_无向量(self) -> None:
        """无 embedding 时 has_embedding() 应为 False."""
        chunk = DocumentChunk(document_id="d", content="x")
        assert chunk.has_embedding() is False

    def test_has_embedding_有向量(self) -> None:
        """有非空 embedding 时 has_embedding() 应为 True."""
        chunk = DocumentChunk(document_id="d", content="x")
        chunk.embedding = EmbeddingVector(content_id=chunk.chunk_id, vector=[0.1, 0.2])
        assert chunk.has_embedding() is True

    def test_has_embedding_空向量(self) -> None:
        """embedding 向量为空列表时 has_embedding() 应为 False."""
        chunk = DocumentChunk(document_id="d", content="x")
        chunk.embedding = EmbeddingVector(content_id=chunk.chunk_id, vector=[])
        assert chunk.has_embedding() is False

    def test_get_parent_chunk_id_有父切片(self) -> None:
        """有 PARENT 关系时应返回 target_chunk_id."""
        chunk = DocumentChunk(document_id="d", content="x")
        chunk.relationships = [
            ChunkRelationship(relation_type=ChunkRelationshipType.PARENT, target_chunk_id="c-parent"),
            ChunkRelationship(relation_type=ChunkRelationshipType.NEXT, target_chunk_id="c-next"),
        ]
        assert chunk.get_parent_chunk_id() == "c-parent"

    def test_get_parent_chunk_id_无父切片(self) -> None:
        """无 PARENT 关系时应返回 None."""
        chunk = DocumentChunk(document_id="d", content="x")
        chunk.relationships = [
            ChunkRelationship(relation_type=ChunkRelationshipType.NEXT, target_chunk_id="c-next"),
        ]
        assert chunk.get_parent_chunk_id() is None

    def test_get_parent_chunk_id_空关系列表(self) -> None:
        """关系列表为空时应返回 None."""
        chunk = DocumentChunk(document_id="d", content="x")
        assert chunk.get_parent_chunk_id() is None

    def test_get_source_document_有source元数据(self) -> None:
        """metadata 中有 'source' 时应返回该值."""
        chunk = DocumentChunk(document_id="d", content="x", metadata={"source": "PDF教材"})
        assert chunk.get_source_document() == "PDF教材"

    def test_get_source_document_无source元数据(self) -> None:
        """metadata 中无 'source' 时应返回 document_id."""
        chunk = DocumentChunk(document_id="doc-123", content="x")
        assert chunk.get_source_document() == "doc-123"

    def test_relationships_列表操作(self) -> None:
        """relationships 列表应支持添加多个关系."""
        chunk = DocumentChunk(document_id="d", content="x")
        chunk.relationships.append(
            ChunkRelationship(relation_type=ChunkRelationshipType.PARENT, target_chunk_id="c-1")
        )
        chunk.relationships.append(
            ChunkRelationship(relation_type=ChunkRelationshipType.NEXT, target_chunk_id="c-2")
        )
        assert len(chunk.relationships) == 2


# ============================================================
# 8. QualityScore 测试
# ============================================================


class TestQualityScore:
    """QualityScore 知识质量多维评分测试."""

    def test_创建_默认值(self) -> None:
        """验证默认值."""
        q = QualityScore()
        assert q.accuracy == 0.8
        assert q.trustworthiness == 0.8
        assert q.consistency == 0.9
        assert q.timeliness == 0.8
        assert q.completeness == 0.7
        assert q.relevancy == 0.8
        assert q.assessor == "system"

    def test_overall_加权平均(self) -> None:
        """默认权重下 overall() 应为加权平均值."""
        q = QualityScore()
        expected = (
            0.8 * 0.25 + 0.8 * 0.20 + 0.9 * 0.15 +
            0.8 * 0.15 + 0.7 * 0.10 + 0.8 * 0.15
        )
        assert abs(q.overall() - expected) < 0.0001

    def test_overall_全1分(self) -> None:
        """所有维度为 1.0 时 overall() 应为 1.0."""
        q = QualityScore(
            accuracy=1.0, trustworthiness=1.0, consistency=1.0,
            timeliness=1.0, completeness=1.0, relevancy=1.0,
        )
        assert abs(q.overall() - 1.0) < 0.0001

    def test_overall_全0分(self) -> None:
        """所有维度为 0.0 时 overall() 应为 0.0."""
        q = QualityScore(
            accuracy=0.0, trustworthiness=0.0, consistency=0.0,
            timeliness=0.0, completeness=0.0, relevancy=0.0,
        )
        assert abs(q.overall() - 0.0) < 0.0001

    def test_is_acceptable_达标(self) -> None:
        """默认分数 (0.805) 超过阈值 0.6 应可接受."""
        q = QualityScore()
        assert q.is_acceptable(0.6) is True

    def test_is_acceptable_未达标(self) -> None:
        """默认分数 (0.805) 低于阈值 0.9 应不可接受."""
        q = QualityScore()
        assert q.is_acceptable(0.9) is False

    def test_is_acceptable_默认阈值(self) -> None:
        """默认阈值 0.6, 默认分数应可接受."""
        q = QualityScore()
        assert q.is_acceptable() is True

    def test_weakest_dimension(self) -> None:
        """默认值下最弱维度应为 completeness (0.7)."""
        q = QualityScore()
        assert q.weakest_dimension() == QualityDimension.COMPLETENESS.value

    def test_strongest_dimension(self) -> None:
        """默认值下最强维度应为 consistency (0.9)."""
        q = QualityScore()
        assert q.strongest_dimension() == QualityDimension.CONSISTENCY.value

    def test_weakest_dimension_自定义(self) -> None:
        """自定义分数下最弱维度应正确识别."""
        q = QualityScore(accuracy=0.3, trustworthiness=0.9, consistency=0.9,
                         timeliness=0.9, completeness=0.9, relevancy=0.9)
        assert q.weakest_dimension() == QualityDimension.ACCURACY.value

    def test_strongest_dimension_自定义(self) -> None:
        """自定义分数下最强维度应正确识别."""
        q = QualityScore(accuracy=0.5, trustworthiness=0.5, consistency=0.5,
                         timeliness=0.5, completeness=0.5, relevancy=1.0)
        assert q.strongest_dimension() == QualityDimension.RELEVANCY.value

    def test_set_weights_合法权重(self) -> None:
        """权重和为 1.0 时应成功设置."""
        q = QualityScore()
        q.set_weights({"accuracy": 0.5, "trustworthiness": 0.5,
                       "consistency": 0.0, "timeliness": 0.0,
                       "completeness": 0.0, "relevancy": 0.0})
        assert q.weights["accuracy"] == 0.5

    def test_set_weights_权重和不足(self) -> None:
        """权重和明显不足 1.0 时应抛出 ValueError."""
        q = QualityScore()
        with pytest.raises(ValueError):
            q.set_weights({"accuracy": 0.3, "trustworthiness": 0.3,
                           "consistency": 0.0, "timeliness": 0.0,
                           "completeness": 0.0, "relevancy": 0.0})

    def test_set_weights_权重和超出(self) -> None:
        """权重和超出 1.0 时应抛出 ValueError."""
        q = QualityScore()
        with pytest.raises(ValueError):
            q.set_weights({"accuracy": 0.5, "trustworthiness": 0.6,
                           "consistency": 0.0, "timeliness": 0.0,
                           "completeness": 0.0, "relevancy": 0.0})

    def test_set_weights_容差范围内(self) -> None:
        """权重和在 0.01 容差内应成功设置."""
        q = QualityScore()
        q.set_weights({"accuracy": 0.26, "trustworthiness": 0.20,
                       "consistency": 0.15, "timeliness": 0.15,
                       "completeness": 0.10, "relevancy": 0.14})  # sum=1.00
        assert q.weights["accuracy"] == 0.26

    def test_set_weights_影响overall(self) -> None:
        """自定义权重后 overall() 应使用新权重计算."""
        q = QualityScore(accuracy=1.0, trustworthiness=0.0, consistency=0.0,
                         timeliness=0.0, completeness=0.0, relevancy=0.0)
        q.set_weights({"accuracy": 1.0, "trustworthiness": 0.0,
                       "consistency": 0.0, "timeliness": 0.0,
                       "completeness": 0.0, "relevancy": 0.0})
        assert abs(q.overall() - 1.0) < 0.0001

    def test_set_weights_不影响其他实例(self) -> None:
        """set_weights 不应影响其他实例的权重."""
        q1 = QualityScore()
        q2 = QualityScore()
        q1.set_weights({"accuracy": 1.0, "trustworthiness": 0.0,
                        "consistency": 0.0, "timeliness": 0.0,
                        "completeness": 0.0, "relevancy": 0.0})
        # q2 的权重应保持默认
        assert q2.weights["accuracy"] == 0.25

    def test_to_dict_序列化(self) -> None:
        """to_dict 应返回包含所有维度和 overall 的字典."""
        q = QualityScore(accuracy=0.9, assessor="agent-1")
        d = q.to_dict()
        assert d["accuracy"] == 0.9
        assert "overall" in d
        assert "trustworthiness" in d
        assert "consistency" in d
        assert "timeliness" in d
        assert "completeness" in d
        assert "relevancy" in d
        assert "assessed_at" in d
        assert d["assessor"] == "agent-1"

    def test_to_dict_overall_四舍五入(self) -> None:
        """to_dict 中 overall 应四舍五入到 4 位小数."""
        q = QualityScore()
        d = q.to_dict()
        assert d["overall"] == round(q.overall(), 4)

    def test_维度范围验证_超出上限(self) -> None:
        """维度值超出 1.0 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            QualityScore(accuracy=1.1)

    def test_维度范围验证_超出下限(self) -> None:
        """维度值低于 0.0 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            QualityScore(trustworthiness=-0.1)


# ============================================================
# 9. ProvenanceInfo 测试
# ============================================================


class TestProvenanceInfo:
    """ProvenanceInfo 知识溯源信息测试."""

    def test_创建_必填字段(self) -> None:
        """entity_id 为必填."""
        p = ProvenanceInfo(entity_id="e-1")
        assert p.entity_id == "e-1"

    def test_默认值(self) -> None:
        """验证默认值."""
        p = ProvenanceInfo(entity_id="e-1")
        assert p.generated_by_activity == ""
        assert p.generated_by_agent == ""
        assert p.agent_role == ProvenanceRole.GENERATOR
        assert p.derived_from == []
        assert p.primary_source == ""
        assert p.revision_of == ""
        assert p.quoted_from == ""
        assert p.used_entities == []

    def test_has_derivation_chain_无派生(self) -> None:
        """无 derived_from 和 primary_source 时应返回 False."""
        p = ProvenanceInfo(entity_id="e-1")
        assert p.has_derivation_chain() is False

    def test_has_derivation_chain_有derived_from(self) -> None:
        """有 derived_from 时应返回 True."""
        p = ProvenanceInfo(entity_id="e-1", derived_from=["e-0"])
        assert p.has_derivation_chain() is True

    def test_has_derivation_chain_有primary_source(self) -> None:
        """有 primary_source 时应返回 True."""
        p = ProvenanceInfo(entity_id="e-1", primary_source="https://example.com")
        assert p.has_derivation_chain() is True

    def test_has_derivation_chain_两者都有(self) -> None:
        """derived_from 和 primary_source 都有时应返回 True."""
        p = ProvenanceInfo(entity_id="e-1", derived_from=["e-0"], primary_source="https://x")
        assert p.has_derivation_chain() is True

    def test_is_original_原始来源(self) -> None:
        """无派生链时应为原始来源."""
        p = ProvenanceInfo(entity_id="e-1")
        assert p.is_original() is True

    def test_is_original_非原始(self) -> None:
        """有派生链时应非原始来源."""
        p = ProvenanceInfo(entity_id="e-1", derived_from=["e-0"])
        assert p.is_original() is False

    def test_trace_depth_无派生链(self) -> None:
        """无派生链时深度为 0."""
        p = ProvenanceInfo(entity_id="e-1")
        assert p.trace_depth() == 0

    def test_trace_depth_有派生链无map(self) -> None:
        """有派生链但无 map 时深度为 1."""
        p = ProvenanceInfo(entity_id="e-1", derived_from=["e-0"])
        assert p.trace_depth() == 1

    def test_trace_depth_单层派生(self) -> None:
        """单层派生 (A→B, B 为原始) 深度为 1."""
        p_b = ProvenanceInfo(entity_id="e-B")
        p_a = ProvenanceInfo(entity_id="e-A", derived_from=["e-B"])
        provenance_map = {"e-B": p_b}
        assert p_a.trace_depth(provenance_map) == 1

    def test_trace_depth_多层派生(self) -> None:
        """多层派生 (A→B→C, C 为原始) 深度为 2."""
        p_c = ProvenanceInfo(entity_id="e-C")
        p_b = ProvenanceInfo(entity_id="e-B", derived_from=["e-C"])
        p_a = ProvenanceInfo(entity_id="e-A", derived_from=["e-B"])
        provenance_map = {"e-B": p_b, "e-C": p_c}
        assert p_a.trace_depth(provenance_map) == 2

    def test_trace_depth_父节点不在map中(self) -> None:
        """派生来源不在 map 中时深度为 1."""
        p_a = ProvenanceInfo(entity_id="e-A", derived_from=["e-unknown"])
        provenance_map: dict[str, ProvenanceInfo] = {}
        assert p_a.trace_depth(provenance_map) == 1

    def test_trace_depth_多父节点取最大深度(self) -> None:
        """多个派生来源时取最大深度."""
        p_c = ProvenanceInfo(entity_id="e-C")
        p_b = ProvenanceInfo(entity_id="e-B", derived_from=["e-C"])
        p_a = ProvenanceInfo(entity_id="e-A", derived_from=["e-B", "e-C"])
        provenance_map = {"e-B": p_b, "e-C": p_c}
        # A→B→C (depth 2) vs A→C (depth 1), 取 max = 2
        assert p_a.trace_depth(provenance_map) == 2


# ============================================================
# 10. EmbeddingVector 测试
# ============================================================


class TestEmbeddingVector:
    """EmbeddingVector 向量化数据测试."""

    def test_创建_基本(self) -> None:
        """创建向量."""
        v = EmbeddingVector(content_id="c-1", vector=[0.1, 0.2, 0.3])
        assert v.content_id == "c-1"
        assert v.vector == [0.1, 0.2, 0.3]

    def test_dim_自动填充(self) -> None:
        """dim 应自动填充为向量长度."""
        v = EmbeddingVector(content_id="c-1", vector=[0.1, 0.2, 0.3])
        assert v.dim == 3

    def test_dim_手动指定_保留(self) -> None:
        """手动指定 dim 时应保留原值."""
        v = EmbeddingVector(content_id="c-1", vector=[0.1, 0.2], dim=512)
        assert v.dim == 512

    def test_dim_空向量时为零(self) -> None:
        """向量为空时 dim 应保持为 0."""
        v = EmbeddingVector(content_id="c-1", vector=[])
        assert v.dim == 0

    def test_is_empty_空向量(self) -> None:
        """空向量时 is_empty() 应为 True."""
        v = EmbeddingVector(content_id="c-1", vector=[])
        assert v.is_empty() is True

    def test_is_empty_非空向量(self) -> None:
        """非空向量时 is_empty() 应为 False."""
        v = EmbeddingVector(content_id="c-1", vector=[0.1])
        assert v.is_empty() is False

    def test_cosine_similarity_相同向量(self) -> None:
        """相同向量的余弦相似度应为 1.0."""
        v1 = EmbeddingVector(content_id="c-1", vector=[1.0, 0.0, 0.0])
        v2 = EmbeddingVector(content_id="c-2", vector=[1.0, 0.0, 0.0])
        assert abs(v1.cosine_similarity(v2) - 1.0) < 0.0001

    def test_cosine_similarity_正交向量(self) -> None:
        """正交向量的余弦相似度应为 0.0."""
        v1 = EmbeddingVector(content_id="c-1", vector=[1.0, 0.0])
        v2 = EmbeddingVector(content_id="c-2", vector=[0.0, 1.0])
        assert abs(v1.cosine_similarity(v2) - 0.0) < 0.0001

    def test_cosine_similarity_相反向量(self) -> None:
        """相反向量的余弦相似度应为 -1.0."""
        v1 = EmbeddingVector(content_id="c-1", vector=[1.0, 0.0])
        v2 = EmbeddingVector(content_id="c-2", vector=[-1.0, 0.0])
        assert abs(v1.cosine_similarity(v2) - (-1.0)) < 0.0001

    def test_cosine_similarity_维度不匹配返回零(self) -> None:
        """维度不匹配时应返回 0.0."""
        v1 = EmbeddingVector(content_id="c-1", vector=[1.0, 0.0])
        v2 = EmbeddingVector(content_id="c-2", vector=[1.0, 0.0, 0.0])
        assert v1.cosine_similarity(v2) == 0.0

    def test_cosine_similarity_空向量返回零(self) -> None:
        """空向量时应返回 0.0."""
        v1 = EmbeddingVector(content_id="c-1", vector=[])
        v2 = EmbeddingVector(content_id="c-2", vector=[1.0])
        assert v1.cosine_similarity(v2) == 0.0

    def test_cosine_similarity_零向量返回零(self) -> None:
        """零向量 (全0) 时应返回 0.0 (避免除以零)."""
        v1 = EmbeddingVector(content_id="c-1", vector=[0.0, 0.0])
        v2 = EmbeddingVector(content_id="c-2", vector=[1.0, 0.0])
        assert v1.cosine_similarity(v2) == 0.0

    def test_cosine_similarity_对称性(self) -> None:
        """余弦相似度应满足对称性."""
        v1 = EmbeddingVector(content_id="c-1", vector=[0.3, 0.7, 0.1])
        v2 = EmbeddingVector(content_id="c-2", vector=[0.5, 0.2, 0.9])
        assert abs(v1.cosine_similarity(v2) - v2.cosine_similarity(v1)) < 0.0001

    def test_sparse_vector_默认空(self) -> None:
        """sparse_vector 默认为空字典."""
        v = EmbeddingVector(content_id="c-1", vector=[0.1])
        assert v.sparse_vector == {}

    def test_sparse_vector_自定义(self) -> None:
        """sparse_vector 应支持稀疏向量."""
        v = EmbeddingVector(
            content_id="c-1", vector=[0.1],
            sparse_vector={0: 0.5, 3: 0.8},
        )
        assert v.sparse_vector[0] == 0.5
        assert v.sparse_vector[3] == 0.8


# ============================================================
# 11. RetrievalResult 测试
# ============================================================


class TestRetrievalResult:
    """RetrievalResult 知识检索结果测试."""

    def test_创建_必填字段(self) -> None:
        """query 为必填."""
        r = RetrievalResult(query="水")
        assert r.query == "水"

    def test_默认值(self) -> None:
        """验证默认值."""
        r = RetrievalResult(query="q")
        assert r.results == []
        assert r.scores == []
        assert r.total == 0
        assert r.retrieval_time_ms == 0.0
        assert r.source_type == "vector"
        assert r.filters == {}

    def test_is_empty_空结果(self) -> None:
        """结果列表为空时 is_empty() 应为 True."""
        r = RetrievalResult(query="q")
        assert r.is_empty() is True

    def test_is_empty_非空结果(self) -> None:
        """结果列表非空时 is_empty() 应为 False."""
        r = RetrievalResult(query="q", results=[{"id": "1"}])
        assert r.is_empty() is False

    def test_top_k_降序排列(self) -> None:
        """top_k 应按分数降序返回."""
        r = RetrievalResult(
            query="q",
            results=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
            scores=[0.5, 0.9, 0.3],
        )
        top = r.top_k(2)
        assert len(top) == 2
        assert top[0][0]["id"] == "b"  # 0.9
        assert top[1][0]["id"] == "a"  # 0.5

    def test_top_k_k大于结果数(self) -> None:
        """k 大于结果数时应返回所有结果."""
        r = RetrievalResult(
            query="q", results=[{"id": "a"}], scores=[0.5],
        )
        top = r.top_k(10)
        assert len(top) == 1

    def test_top_k_k为零(self) -> None:
        """k=0 时应返回空列表."""
        r = RetrievalResult(
            query="q", results=[{"id": "a"}], scores=[0.5],
        )
        top = r.top_k(0)
        assert len(top) == 0

    def test_best_score_有分数(self) -> None:
        """有分数时应返回最高分."""
        r = RetrievalResult(query="q", scores=[0.3, 0.9, 0.5])
        assert r.best_score() == 0.9

    def test_best_score_无分数(self) -> None:
        """无分数时应返回 0.0."""
        r = RetrievalResult(query="q")
        assert r.best_score() == 0.0

    def test_total_不能为负(self) -> None:
        """total 为负数应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            RetrievalResult(query="q", total=-1)


# ============================================================
# 12. KnowledgeBaseStats 测试
# ============================================================


class TestKnowledgeBaseStats:
    """KnowledgeBaseStats 知识库统计信息测试."""

    def test_创建_默认值(self) -> None:
        """验证默认值全为零或空."""
        s = KnowledgeBaseStats()
        assert s.total_entities == 0
        assert s.total_chunks == 0
        assert s.total_triples == 0
        assert s.total_sources == 0
        assert s.entities_by_type == {}
        assert s.chunks_by_modality == {}
        assert s.avg_quality == 0.0
        assert s.indexed_vectors == 0

    def test_is_empty_全空(self) -> None:
        """实体和切片都为 0 时 is_empty() 应为 True."""
        s = KnowledgeBaseStats()
        assert s.is_empty() is True

    def test_is_empty_有实体(self) -> None:
        """有实体时 is_empty() 应为 False."""
        s = KnowledgeBaseStats(total_entities=10)
        assert s.is_empty() is False

    def test_is_empty_有切片(self) -> None:
        """有切片时 is_empty() 应为 False."""
        s = KnowledgeBaseStats(total_chunks=5)
        assert s.is_empty() is False

    def test_is_empty_实体和切片都有(self) -> None:
        """实体和切片都有时 is_empty() 应为 False."""
        s = KnowledgeBaseStats(total_entities=3, total_chunks=7)
        assert s.is_empty() is False

    def test_avg_quality_范围验证(self) -> None:
        """avg_quality 超出 0.0~1.0 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeBaseStats(avg_quality=1.5)
        with pytest.raises(ValidationError):
            KnowledgeBaseStats(avg_quality=-0.1)


# ============================================================
# 13. IngestResult 测试
# ============================================================


class TestIngestResult:
    """IngestResult 知识批量导入结果测试."""

    def test_创建_默认值(self) -> None:
        """验证默认值全为零或空."""
        r = IngestResult()
        assert r.source == ""
        assert r.total == 0
        assert r.success == 0
        assert r.failed == 0
        assert r.skipped == 0
        assert r.errors == []
        assert r.ingested_ids == []

    def test_is_full_success_全部成功(self) -> None:
        """failed=0 且 skipped=0 时应为完全成功."""
        r = IngestResult(total=10, success=10, failed=0, skipped=0)
        assert r.is_full_success() is True

    def test_is_full_success_有失败(self) -> None:
        """failed>0 时不应为完全成功."""
        r = IngestResult(total=10, success=8, failed=2, skipped=0)
        assert r.is_full_success() is False

    def test_is_full_success_有跳过(self) -> None:
        """skipped>0 时不应为完全成功."""
        r = IngestResult(total=10, success=8, failed=0, skipped=2)
        assert r.is_full_success() is False

    def test_is_full_success_全零(self) -> None:
        """total=0, success=0 时应为完全成功 (0==0 and 0==0)."""
        r = IngestResult()
        assert r.is_full_success() is True

    def test_success_rate_正常计算(self) -> None:
        """success_rate 应正确计算."""
        r = IngestResult(total=10, success=8)
        assert r.success_rate() == 0.8

    def test_success_rate_全部成功(self) -> None:
        """全部成功时 success_rate 应为 1.0."""
        r = IngestResult(total=10, success=10)
        assert r.success_rate() == 1.0

    def test_success_rate_total为零(self) -> None:
        """total=0 时 success_rate 应为 0.0."""
        r = IngestResult(total=0, success=0)
        assert r.success_rate() == 0.0

    def test_字段不能为负(self) -> None:
        """total/success/failed/skipped 不能为负数."""
        with pytest.raises(ValidationError):
            IngestResult(total=-1)
        with pytest.raises(ValidationError):
            IngestResult(success=-1)


# ============================================================
# 14. RetrievalFilter 测试
# ============================================================


class TestRetrievalFilter:
    """RetrievalFilter 知识检索过滤器测试."""

    def test_默认值(self) -> None:
        """验证默认值."""
        f = RetrievalFilter()
        assert f.domain is None
        assert f.entity_types == []
        assert f.content_types == []
        assert f.source_tiers == []
        assert f.access_level == AccessLevel.INTERNAL
        assert f.min_quality == 0.0
        assert f.min_confidence == 0.0
        assert f.exclude_deprecated is True

    # --- matches_entity ---

    def test_matches_entity_无过滤条件(self) -> None:
        """无过滤条件时所有实体应匹配."""
        f = RetrievalFilter()
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        assert f.matches_entity(e) is True

    def test_matches_entity_领域匹配(self) -> None:
        """领域匹配时应返回 True."""
        f = RetrievalFilter(domain="chemistry")
        e = KnowledgeEntity(entity_type=EntityType.CHEMICAL_COMPOUND, name="水", domain="chemistry")
        assert f.matches_entity(e) is True

    def test_matches_entity_领域不匹配(self) -> None:
        """领域不匹配时应返回 False."""
        f = RetrievalFilter(domain="chemistry")
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", domain="general")
        assert f.matches_entity(e) is False

    def test_matches_entity_类型匹配(self) -> None:
        """实体类型匹配时应返回 True."""
        f = RetrievalFilter(entity_types=[EntityType.CHEMICAL_COMPOUND, EntityType.MATERIAL])
        e = KnowledgeEntity(entity_type=EntityType.CHEMICAL_COMPOUND, name="水")
        assert f.matches_entity(e) is True

    def test_matches_entity_类型不匹配(self) -> None:
        """实体类型不匹配时应返回 False."""
        f = RetrievalFilter(entity_types=[EntityType.CHEMICAL_COMPOUND])
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        assert f.matches_entity(e) is False

    def test_matches_entity_来源层级匹配(self) -> None:
        """来源层级匹配时应返回 True."""
        src = KnowledgeSource(source_id="nist", name="NIST", tier=SourceTier.TIER1_PUBLIC)
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", source=src)
        f = RetrievalFilter(source_tiers=[SourceTier.TIER1_PUBLIC])
        assert f.matches_entity(e) is True

    def test_matches_entity_来源层级不匹配(self) -> None:
        """来源层级不匹配时应返回 False."""
        src = KnowledgeSource(source_id="nist", name="NIST", tier=SourceTier.TIER1_PUBLIC)
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", source=src)
        f = RetrievalFilter(source_tiers=[SourceTier.INTERNAL_DOCUMENT])
        assert f.matches_entity(e) is False

    def test_matches_entity_无来源时来源过滤跳过(self) -> None:
        """实体无 source 时来源层级过滤应跳过 (返回 True)."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", source=None)
        f = RetrievalFilter(source_tiers=[SourceTier.TIER1_PUBLIC])
        assert f.matches_entity(e) is True

    def test_matches_entity_质量达标(self) -> None:
        """质量分数达标时应返回 True."""
        e = KnowledgeEntity(
            entity_type=EntityType.CONCEPT, name="x",
            quality=QualityScore(accuracy=1.0, trustworthiness=1.0, consistency=1.0,
                                 timeliness=1.0, completeness=1.0, relevancy=1.0),
        )
        f = RetrievalFilter(min_quality=0.5)
        assert f.matches_entity(e) is True

    def test_matches_entity_质量不达标(self) -> None:
        """质量分数不达标时应返回 False."""
        e = KnowledgeEntity(
            entity_type=EntityType.CONCEPT, name="x",
            quality=QualityScore(accuracy=0.2, trustworthiness=0.2, consistency=0.2,
                                 timeliness=0.2, completeness=0.2, relevancy=0.2),
        )
        f = RetrievalFilter(min_quality=0.5)
        assert f.matches_entity(e) is False

    def test_matches_entity_无质量评分(self) -> None:
        """质量评分为 None 且 min_quality>0 时应返回 False."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x")
        f = RetrievalFilter(min_quality=0.5)
        assert f.matches_entity(e) is False

    def test_matches_entity_日期范围匹配(self) -> None:
        """创建时间在日期范围内时应返回 True."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", created_at=1000.0)
        f = RetrievalFilter(date_from=500.0, date_to=1500.0)
        assert f.matches_entity(e) is True

    def test_matches_entity_早于日期范围(self) -> None:
        """创建时间早于 date_from 时应返回 False."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", created_at=100.0)
        f = RetrievalFilter(date_from=500.0)
        assert f.matches_entity(e) is False

    def test_matches_entity_晚于日期范围(self) -> None:
        """创建时间晚于 date_to 时应返回 False."""
        e = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="x", created_at=2000.0)
        f = RetrievalFilter(date_to=1500.0)
        assert f.matches_entity(e) is False

    def test_matches_entity_多条件组合_全匹配(self) -> None:
        """多个条件全部匹配时应返回 True."""
        src = KnowledgeSource(source_id="nist", name="NIST", tier=SourceTier.TIER1_PUBLIC)
        e = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="水",
            domain="chemistry", source=src,
            quality=QualityScore(accuracy=0.9),
        )
        f = RetrievalFilter(
            domain="chemistry",
            entity_types=[EntityType.CHEMICAL_COMPOUND],
            source_tiers=[SourceTier.TIER1_PUBLIC],
            min_quality=0.5,
        )
        assert f.matches_entity(e) is True

    def test_matches_entity_多条件组合_一项不匹配(self) -> None:
        """多个条件中一项不匹配时应返回 False."""
        src = KnowledgeSource(source_id="nist", name="NIST", tier=SourceTier.TIER1_PUBLIC)
        e = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="水",
            domain="chemistry", source=src,
        )
        f = RetrievalFilter(
            domain="chemistry",
            entity_types=[EntityType.MATERIAL],  # 类型不匹配
        )
        assert f.matches_entity(e) is False

    # --- matches_chunk ---

    def test_matches_chunk_无过滤条件(self) -> None:
        """无过滤条件时所有切片应匹配."""
        f = RetrievalFilter()
        c = DocumentChunk(document_id="d", content="x")
        assert f.matches_chunk(c) is True

    def test_matches_chunk_内容类型匹配(self) -> None:
        """内容类型匹配时应返回 True."""
        f = RetrievalFilter(content_types=[ContentModality.TEXT])
        c = DocumentChunk(document_id="d", content="x", content_type=ContentModality.TEXT)
        assert f.matches_chunk(c) is True

    def test_matches_chunk_内容类型不匹配(self) -> None:
        """内容类型不匹配时应返回 False."""
        f = RetrievalFilter(content_types=[ContentModality.IMAGE])
        c = DocumentChunk(document_id="d", content="x", content_type=ContentModality.TEXT)
        assert f.matches_chunk(c) is False

    def test_matches_chunk_质量达标(self) -> None:
        """切片质量达标时应返回 True."""
        c = DocumentChunk(
            document_id="d", content="x",
            quality=QualityScore(accuracy=1.0, trustworthiness=1.0, consistency=1.0,
                                 timeliness=1.0, completeness=1.0, relevancy=1.0),
        )
        f = RetrievalFilter(min_quality=0.5)
        assert f.matches_chunk(c) is True

    def test_matches_chunk_质量不达标(self) -> None:
        """切片质量不达标时应返回 False."""
        c = DocumentChunk(
            document_id="d", content="x",
            quality=QualityScore(accuracy=0.1, trustworthiness=0.1, consistency=0.1,
                                 timeliness=0.1, completeness=0.1, relevancy=0.1),
        )
        f = RetrievalFilter(min_quality=0.5)
        assert f.matches_chunk(c) is False

    def test_matches_chunk_无质量评分(self) -> None:
        """切片无质量评分且 min_quality>0 时应返回 False."""
        c = DocumentChunk(document_id="d", content="x")
        f = RetrievalFilter(min_quality=0.5)
        assert f.matches_chunk(c) is False

    def test_matches_chunk_日期范围匹配(self) -> None:
        """切片创建时间在范围内时应返回 True."""
        c = DocumentChunk(document_id="d", content="x", created_at=1000.0)
        f = RetrievalFilter(date_from=500.0, date_to=1500.0)
        assert f.matches_chunk(c) is True

    def test_matches_chunk_早于日期范围(self) -> None:
        """切片创建时间早于 date_from 时应返回 False."""
        c = DocumentChunk(document_id="d", content="x", created_at=100.0)
        f = RetrievalFilter(date_from=500.0)
        assert f.matches_chunk(c) is False


# ============================================================
# 15. OntologyProperty 测试
# ============================================================


class TestOntologyProperty:
    """OntologyProperty 本体属性定义测试."""

    def test_创建_必填字段(self) -> None:
        """name 为必填."""
        p = OntologyProperty(name="formula")
        assert p.name == "formula"

    def test_默认值(self) -> None:
        """验证默认值."""
        p = OntologyProperty(name="x")
        assert p.display_name == ""
        assert p.description == ""
        assert p.property_type == "datatype"
        assert p.domain == []
        assert p.range == "string"
        assert p.required is False
        assert p.cardinality == 0
        assert p.default_value is None
        assert p.enum_values == []

    def test_创建_全部字段(self) -> None:
        """创建时指定所有字段."""
        p = OntologyProperty(
            name="state",
            display_name="状态",
            description="物质状态",
            property_type="datatype",
            domain=[EntityType.CHEMICAL_COMPOUND],
            range="string",
            required=True,
            cardinality=1,
            default_value="solid",
            enum_values=["solid", "liquid", "gas"],
        )
        assert p.display_name == "状态"
        assert p.property_type == "datatype"
        assert p.required is True
        assert p.cardinality == 1
        assert p.default_value == "solid"
        assert p.enum_values == ["solid", "liquid", "gas"]

    def test_object类型属性(self) -> None:
        """property_type='object' 的对象属性."""
        p = OntologyProperty(name="author", property_type="object", range="Person")
        assert p.property_type == "object"
        assert p.range == "Person"

    def test_cardinality_不能为负(self) -> None:
        """cardinality 为负数应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            OntologyProperty(name="x", cardinality=-1)

    def test_domain_实体类型列表(self) -> None:
        """domain 应支持实体类型列表."""
        p = OntologyProperty(name="x", domain=[EntityType.PAPER, EntityType.TEXTBOOK])
        assert EntityType.PAPER in p.domain
        assert EntityType.TEXTBOOK in p.domain

    def test_enum_values_枚举约束(self) -> None:
        """enum_values 应正确存储枚举可选值."""
        p = OntologyProperty(name="level", enum_values=["low", "medium", "high"])
        assert len(p.enum_values) == 3


# ============================================================
# 16. OntologyRelation 测试
# ============================================================


class TestOntologyRelation:
    """OntologyRelation 本体关系定义测试."""

    def test_创建_必填字段(self) -> None:
        """name 为必填."""
        r = OntologyRelation(name="cites")
        assert r.name == "cites"

    def test_默认值(self) -> None:
        """验证默认值."""
        r = OntologyRelation(name="x")
        assert r.display_name == ""
        assert r.description == ""
        assert r.domain == []
        assert r.range == []
        assert r.inverse_of == ""
        assert r.transitive is False
        assert r.symmetric is False
        assert r.functional is False

    def test_创建_全部字段(self) -> None:
        """创建时指定所有字段."""
        r = OntologyRelation(
            name="cites",
            display_name="引用",
            description="论文引用关系",
            domain=[EntityType.PAPER],
            range=[EntityType.PAPER],
            inverse_of="cited_by",
            transitive=False,
            symmetric=False,
            functional=False,
        )
        assert r.display_name == "引用"
        assert r.domain == [EntityType.PAPER]
        assert r.range == [EntityType.PAPER]
        assert r.inverse_of == "cited_by"

    def test_对称关系(self) -> None:
        """symmetric=True 的对称关系."""
        r = OntologyRelation(name="equivalent_to", symmetric=True)
        assert r.symmetric is True

    def test_传递关系(self) -> None:
        """transitive=True 的传递关系."""
        r = OntologyRelation(name="part_of", transitive=True)
        assert r.transitive is True

    def test_函数性关系(self) -> None:
        """functional=True 的函数性关系."""
        r = OntologyRelation(name="authored_by", functional=True)
        assert r.functional is True

    def test_domain_range_实体类型列表(self) -> None:
        """domain 和 range 应支持实体类型列表."""
        r = OntologyRelation(
            name="authored_by",
            domain=[EntityType.PAPER, EntityType.TEXTBOOK],
            range=[EntityType.PERSON],
        )
        assert len(r.domain) == 2
        assert EntityType.PERSON in r.range


# ============================================================
# 17. OntologyClass 测试
# ============================================================


class TestOntologyClass:
    """OntologyClass 本体类定义测试."""

    def _make_class(self) -> OntologyClass:
        """创建测试用 OntologyClass."""
        return OntologyClass(
            class_id="cls-test",
            entity_type=EntityType.CHEMICAL_COMPOUND,
            display_name="化学化合物",
            properties=[
                OntologyProperty(name="formula", display_name="分子式", required=True),
                OntologyProperty(name="cas_number", display_name="CAS号"),
                OntologyProperty(name="state", display_name="状态",
                                 enum_values=["solid", "liquid", "gas"]),
            ],
            allowed_relations=[RelationType.HAS_PROPERTY, RelationType.RELATED_TO],
            parent_type=EntityType.CONCEPT,
        )

    def test_创建_必填字段(self) -> None:
        """class_id 和 entity_type 为必填."""
        c = OntologyClass(class_id="c1", entity_type=EntityType.CONCEPT)
        assert c.class_id == "c1"
        assert c.entity_type == EntityType.CONCEPT

    def test_has_property_存在(self) -> None:
        """has_property 应返回 True 当属性存在."""
        c = self._make_class()
        assert c.has_property("formula") is True

    def test_has_property_不存在(self) -> None:
        """has_property 应返回 False 当属性不存在."""
        c = self._make_class()
        assert c.has_property("nonexistent") is False

    def test_get_property_存在(self) -> None:
        """get_property 应返回属性定义."""
        c = self._make_class()
        p = c.get_property("formula")
        assert p is not None
        assert p.name == "formula"
        assert p.required is True

    def test_get_property_不存在(self) -> None:
        """get_property 应返回 None 当属性不存在."""
        c = self._make_class()
        assert c.get_property("nonexistent") is None

    def test_required_properties(self) -> None:
        """required_properties 应返回所有必需属性."""
        c = self._make_class()
        req = c.required_properties()
        assert len(req) == 1
        assert req[0].name == "formula"

    def test_required_properties_无必需(self) -> None:
        """无必需属性时 required_properties 应返回空列表."""
        c = OntologyClass(
            class_id="c1", entity_type=EntityType.CONCEPT,
            properties=[OntologyProperty(name="x")],
        )
        assert c.required_properties() == []

    def test_is_subclass_of_是子类(self) -> None:
        """is_subclass_of 应返回 True 当 parent_type 匹配."""
        c = self._make_class()
        assert c.is_subclass_of(EntityType.CONCEPT) is True

    def test_is_subclass_of_不是子类(self) -> None:
        """is_subclass_of 应返回 False 当 parent_type 不匹配."""
        c = self._make_class()
        assert c.is_subclass_of(EntityType.MATERIAL) is False

    def test_is_subclass_of_无父类(self) -> None:
        """无父类时 is_subclass_of 应返回 False."""
        c = OntologyClass(class_id="c1", entity_type=EntityType.CONCEPT)
        assert c.is_subclass_of(EntityType.CONCEPT) is False

    def test_allowed_relations_列表(self) -> None:
        """allowed_relations 应正确存储关系类型列表."""
        c = self._make_class()
        assert RelationType.HAS_PROPERTY in c.allowed_relations
        assert RelationType.RELATED_TO in c.allowed_relations


# ============================================================
# 18. DomainOntology 测试
# ============================================================


class TestDomainOntology:
    """DomainOntology 领域本体测试."""

    def _make_ontology(self) -> DomainOntology:
        """创建测试用 DomainOntology."""
        return DomainOntology(
            ontology_id="onto-test",
            domain="test",
            display_name="测试本体",
            classes=[
                OntologyClass(
                    class_id="cls-a",
                    entity_type=EntityType.CHEMICAL_COMPOUND,
                    properties=[
                        OntologyProperty(name="formula", required=True),
                        OntologyProperty(name="state", enum_values=["solid", "liquid", "gas"]),
                    ],
                ),
                OntologyClass(
                    class_id="cls-b",
                    entity_type=EntityType.MATERIAL,
                    properties=[
                        OntologyProperty(name="composition", required=True),
                    ],
                ),
            ],
            relations=[
                OntologyRelation(
                    name=RelationType.HAS_PROPERTY.value,
                    domain=[EntityType.CHEMICAL_COMPOUND],
                ),
                OntologyRelation(
                    name=RelationType.EQUIVALENT_TO.value,
                    domain=[EntityType.CHEMICAL_COMPOUND],
                    range=[EntityType.CHEMICAL_COMPOUND],
                ),
            ],
            global_properties=[
                OntologyProperty(name="name", required=True),
            ],
        )

    def test_创建_必填字段(self) -> None:
        """ontology_id 和 domain 为必填."""
        o = DomainOntology(ontology_id="o1", domain="d1")
        assert o.ontology_id == "o1"
        assert o.domain == "d1"

    def test_get_class_存在(self) -> None:
        """get_class 应返回类定义."""
        o = self._make_ontology()
        c = o.get_class(EntityType.CHEMICAL_COMPOUND)
        assert c is not None
        assert c.class_id == "cls-a"

    def test_get_class_不存在(self) -> None:
        """get_class 应返回 None 当类型不存在."""
        o = self._make_ontology()
        assert o.get_class(EntityType.PERSON) is None

    def test_get_relation_存在(self) -> None:
        """get_relation 应返回关系定义."""
        o = self._make_ontology()
        r = o.get_relation("has_property")
        assert r is not None
        assert r.name == "has_property"

    def test_get_relation_不存在(self) -> None:
        """get_relation 应返回 None 当关系不存在."""
        o = self._make_ontology()
        assert o.get_relation("nonexistent") is None

    def test_validate_entity_type_存在(self) -> None:
        """validate_entity_type 应返回 True 当类型已定义."""
        o = self._make_ontology()
        assert o.validate_entity_type(EntityType.CHEMICAL_COMPOUND) is True

    def test_validate_entity_type_不存在(self) -> None:
        """validate_entity_type 应返回 False 当类型未定义."""
        o = self._make_ontology()
        assert o.validate_entity_type(EntityType.PERSON) is False

    def test_validate_relation_合法(self) -> None:
        """validate_relation 应返回 True 当主语宾语都在定义域/值域内."""
        o = self._make_ontology()
        assert o.validate_relation(
            "equivalent_to", EntityType.CHEMICAL_COMPOUND, EntityType.CHEMICAL_COMPOUND
        ) is True

    def test_validate_relation_主语不在定义域(self) -> None:
        """validate_relation 应返回 False 当主语不在定义域."""
        o = self._make_ontology()
        assert o.validate_relation(
            "equivalent_to", EntityType.MATERIAL, EntityType.CHEMICAL_COMPOUND
        ) is False

    def test_validate_relation_宾语不在值域(self) -> None:
        """validate_relation 应返回 False 当宾语不在值域."""
        o = self._make_ontology()
        assert o.validate_relation(
            "equivalent_to", EntityType.CHEMICAL_COMPOUND, EntityType.MATERIAL
        ) is False

    def test_validate_relation_关系不存在(self) -> None:
        """validate_relation 应返回 False 当关系不存在."""
        o = self._make_ontology()
        assert o.validate_relation(
            "nonexistent", EntityType.CHEMICAL_COMPOUND, EntityType.CHEMICAL_COMPOUND
        ) is False

    def test_validate_relation_空定义域时任意主语合法(self) -> None:
        """关系定义域为空时任意主语类型应合法."""
        o = DomainOntology(
            ontology_id="o", domain="d",
            relations=[
                OntologyRelation(name="related_to"),  # domain 和 range 都为空
            ],
        )
        assert o.validate_relation(
            "related_to", EntityType.CONCEPT, EntityType.PERSON
        ) is True

    def test_validate_properties_通过(self) -> None:
        """validate_properties 应返回空列表当属性合法."""
        o = self._make_ontology()
        violations = o.validate_properties(
            EntityType.CHEMICAL_COMPOUND, {"formula": "H2O", "state": "liquid"}
        )
        assert violations == []

    def test_validate_properties_缺少必需属性(self) -> None:
        """validate_properties 应报告缺少的必需属性."""
        o = self._make_ontology()
        violations = o.validate_properties(
            EntityType.CHEMICAL_COMPOUND, {"state": "liquid"}
        )
        assert any("formula" in v for v in violations)

    def test_validate_properties_枚举值非法(self) -> None:
        """validate_properties 应报告不在枚举中的属性值."""
        o = self._make_ontology()
        violations = o.validate_properties(
            EntityType.CHEMICAL_COMPOUND, {"formula": "H2O", "state": "plasma"}
        )
        assert any("state" in v for v in violations)

    def test_validate_properties_类型未定义(self) -> None:
        """validate_properties 应报告未定义的实体类型."""
        o = self._make_ontology()
        violations = o.validate_properties(EntityType.PERSON, {})
        assert any("未在本体中定义" in v for v in violations)

    def test_validate_properties_全局属性枚举验证(self) -> None:
        """validate_properties 应验证全局属性的枚举值."""
        o = DomainOntology(
            ontology_id="o", domain="d",
            classes=[
                OntologyClass(
                    class_id="c", entity_type=EntityType.CONCEPT,
                    properties=[OntologyProperty(name="definition", required=True)],
                ),
            ],
            global_properties=[
                OntologyProperty(name="safety_level",
                                 enum_values=["low", "medium", "high"]),
            ],
        )
        violations = o.validate_properties(
            EntityType.CONCEPT, {"definition": "x", "safety_level": "extreme"}
        )
        assert any("safety_level" in v for v in violations)

    def test_class_count(self) -> None:
        """class_count 应返回类数量."""
        o = self._make_ontology()
        assert o.class_count() == 2

    def test_relation_count(self) -> None:
        """relation_count 应返回关系数量."""
        o = self._make_ontology()
        assert o.relation_count() == 2


# ============================================================
# 19. OntologyRegistry 测试
# ============================================================


class TestOntologyRegistry:
    """OntologyRegistry 本体注册中心测试."""

    def test_初始化_预构建本体(self) -> None:
        """初始化时应注册 4 个预构建本体."""
        reg = OntologyRegistry()
        domains = reg.list_domains()
        assert len(domains) == 4
        assert "general" in domains
        assert "chemistry" in domains
        assert "materials" in domains
        assert "education" in domains

    def test_get_ontology_存在(self) -> None:
        """get_ontology 应返回指定领域的本体."""
        reg = OntologyRegistry()
        onto = reg.get_ontology("chemistry")
        assert onto is not None
        assert onto.domain == "chemistry"

    def test_get_ontology_不存在(self) -> None:
        """get_ontology 应返回 None 当领域不存在."""
        reg = OntologyRegistry()
        assert reg.get_ontology("nonexistent") is None

    def test_list_domains(self) -> None:
        """list_domains 应返回所有已注册领域."""
        reg = OntologyRegistry()
        domains = reg.list_domains()
        assert len(domains) == 4

    def test_validate_entity_type_合法(self) -> None:
        """validate_entity_type 应返回 True 当类型在领域本体中定义."""
        reg = OntologyRegistry()
        assert reg.validate_entity_type("chemistry", EntityType.CHEMICAL_COMPOUND) is True

    def test_validate_entity_type_不合法(self) -> None:
        """validate_entity_type 应返回 False 当类型未在领域本体中定义."""
        reg = OntologyRegistry()
        assert reg.validate_entity_type("chemistry", EntityType.TEXTBOOK) is False

    def test_validate_entity_type_领域不存在(self) -> None:
        """validate_entity_type 应返回 False 当领域不存在."""
        reg = OntologyRegistry()
        assert reg.validate_entity_type("nonexistent", EntityType.CONCEPT) is False

    def test_validate_properties_通过(self) -> None:
        """validate_properties 应返回空列表当属性合法."""
        reg = OntologyRegistry()
        violations = reg.validate_properties(
            "chemistry", EntityType.CHEMICAL_COMPOUND, {"formula": "H2O"}
        )
        assert violations == []

    def test_validate_properties_领域不存在(self) -> None:
        """validate_properties 应返回错误信息当领域不存在."""
        reg = OntologyRegistry()
        violations = reg.validate_properties(
            "nonexistent", EntityType.CONCEPT, {}
        )
        assert any("领域本体未找到" in v for v in violations)

    def test_validate_relation_合法(self) -> None:
        """validate_relation 应返回 True 当关系合法."""
        reg = OntologyRegistry()
        assert reg.validate_relation(
            "chemistry", "equivalent_to",
            EntityType.CHEMICAL_COMPOUND, EntityType.CHEMICAL_COMPOUND,
        ) is True

    def test_validate_relation_不合法(self) -> None:
        """validate_relation 应返回 False 当关系不合法."""
        reg = OntologyRegistry()
        assert reg.validate_relation(
            "chemistry", "cites",
            EntityType.CHEMICAL_COMPOUND, EntityType.CHEMICAL_COMPOUND,
        ) is False

    def test_validate_relation_领域不存在(self) -> None:
        """validate_relation 应返回 False 当领域不存在."""
        reg = OntologyRegistry()
        assert reg.validate_relation(
            "nonexistent", "cites",
            EntityType.PAPER, EntityType.PAPER,
        ) is False

    def test_get_class_存在(self) -> None:
        """get_class 应返回类定义."""
        reg = OntologyRegistry()
        c = reg.get_class("chemistry", EntityType.CHEMICAL_COMPOUND)
        assert c is not None
        assert c.entity_type == EntityType.CHEMICAL_COMPOUND

    def test_get_class_领域不存在(self) -> None:
        """get_class 应返回 None 当领域不存在."""
        reg = OntologyRegistry()
        assert reg.get_class("nonexistent", EntityType.CONCEPT) is None

    def test_total_classes(self) -> None:
        """total_classes 应返回所有领域的类总数."""
        reg = OntologyRegistry()
        # general: 4, chemistry: 3, materials: 3, education: 3 = 13
        assert reg.total_classes() == 13

    def test_total_relations(self) -> None:
        """total_relations 应返回所有领域的关系总数."""
        reg = OntologyRegistry()
        # general: 7, chemistry: 4, materials: 4, education: 4 = 19
        assert reg.total_relations() == 19

    def test_register_自定义本体(self) -> None:
        """register 应能注册自定义本体."""
        reg = OntologyRegistry()
        custom = DomainOntology(
            ontology_id="onto-custom",
            domain="custom",
            display_name="自定义本体",
            classes=[
                OntologyClass(class_id="c1", entity_type=EntityType.CONCEPT),
            ],
        )
        reg.register(custom)
        assert "custom" in reg.list_domains()
        assert reg.get_ontology("custom") is not None
        assert reg.validate_entity_type("custom", EntityType.CONCEPT) is True


# ============================================================
# 20. 预构建本体内容验证
# ============================================================


class TestPrebuiltOntologies:
    """预构建本体内容详细验证."""

    def test_domain_type_常量(self) -> None:
        """DomainType 应定义 4 个领域常量."""
        assert DomainType.CHEMISTRY == "chemistry"
        assert DomainType.MATERIALS == "materials"
        assert DomainType.EDUCATION == "education"
        assert DomainType.GENERAL == "general"

    # --- 化学本体 ---

    def test_化学本体_基本信息(self) -> None:
        reg = OntologyRegistry()
        onto = reg.get_ontology("chemistry")
        assert onto is not None
        assert onto.display_name == "化学本体"
        assert onto.class_count() == 3

    def test_化学本体_化合物有formula属性(self) -> None:
        """CHEMICAL_COMPOUND 类应有 formula 属性."""
        reg = OntologyRegistry()
        c = reg.get_class("chemistry", EntityType.CHEMICAL_COMPOUND)
        assert c is not None
        assert c.has_property("formula") is True
        formula_prop = c.get_property("formula")
        assert formula_prop is not None
        assert formula_prop.required is True

    def test_化学本体_化合物有HAS_PROPERTY关系(self) -> None:
        """化学本体应定义 HAS_PROPERTY 关系."""
        reg = OntologyRegistry()
        onto = reg.get_ontology("chemistry")
        rel = onto.get_relation("has_property")
        assert rel is not None
        assert EntityType.CHEMICAL_COMPOUND in rel.domain

    def test_化学本体_化合物有state枚举属性(self) -> None:
        """CHEMICAL_COMPOUND 应有 state 枚举属性."""
        reg = OntologyRegistry()
        c = reg.get_class("chemistry", EntityType.CHEMICAL_COMPOUND)
        state_prop = c.get_property("state")
        assert state_prop is not None
        assert "solid" in state_prop.enum_values
        assert "liquid" in state_prop.enum_values
        assert "gas" in state_prop.enum_values

    def test_化学本体_化合物是CONCEPT子类(self) -> None:
        """CHEMICAL_COMPOUND 应是 CONCEPT 的子类."""
        reg = OntologyRegistry()
        c = reg.get_class("chemistry", EntityType.CHEMICAL_COMPOUND)
        assert c.is_subclass_of(EntityType.CONCEPT) is True

    def test_化学本体_安全等级全局属性(self) -> None:
        """化学本体全局属性应包含 safety_level 枚举."""
        reg = OntologyRegistry()
        onto = reg.get_ontology("chemistry")
        safety = next(
            (p for p in onto.global_properties if p.name == "safety_level"), None
        )
        assert safety is not None
        assert "low" in safety.enum_values
        assert "extreme" in safety.enum_values

    # --- 材料本体 ---

    def test_材料本体_基本信息(self) -> None:
        reg = OntologyRegistry()
        onto = reg.get_ontology("materials")
        assert onto is not None
        assert onto.display_name == "材料科学本体"
        assert onto.class_count() == 3

    def test_材料本体_材料有material_class枚举属性(self) -> None:
        """MATERIAL 类应有 material_class 枚举属性."""
        reg = OntologyRegistry()
        c = reg.get_class("materials", EntityType.MATERIAL)
        assert c is not None
        mc_prop = c.get_property("material_class")
        assert mc_prop is not None
        assert mc_prop.required is True
        assert "metal" in mc_prop.enum_values
        assert "ceramic" in mc_prop.enum_values
        assert "polymer" in mc_prop.enum_values
        assert "composite" in mc_prop.enum_values

    def test_材料本体_材料有composition必需属性(self) -> None:
        """MATERIAL 类应有 composition 必需属性."""
        reg = OntologyRegistry()
        c = reg.get_class("materials", EntityType.MATERIAL)
        comp = c.get_property("composition")
        assert comp is not None
        assert comp.required is True

    # --- 教育本体 ---

    def test_教育本体_基本信息(self) -> None:
        reg = OntologyRegistry()
        onto = reg.get_ontology("education")
        assert onto is not None
        assert onto.display_name == "教育本体"
        assert onto.class_count() == 3

    def test_教育本体_论文有CITES关系(self) -> None:
        """教育本体应定义 CITES 关系."""
        reg = OntologyRegistry()
        onto = reg.get_ontology("education")
        rel = onto.get_relation("cites")
        assert rel is not None
        assert EntityType.PAPER in rel.domain
        assert EntityType.PAPER in rel.range

    def test_教育本体_教材有isbn属性(self) -> None:
        """TEXTBOOK 类应有 isbn 属性."""
        reg = OntologyRegistry()
        c = reg.get_class("education", EntityType.TEXTBOOK)
        assert c is not None
        assert c.has_property("isbn") is True
        isbn_prop = c.get_property("isbn")
        assert isbn_prop is not None

    def test_教育本体_教材有author必需属性(self) -> None:
        """TEXTBOOK 类应有 author 必需属性."""
        reg = OntologyRegistry()
        c = reg.get_class("education", EntityType.TEXTBOOK)
        author_prop = c.get_property("author")
        assert author_prop is not None
        assert author_prop.required is True

    def test_教育本体_论文有doi属性(self) -> None:
        """PAPER 类应有 doi 属性."""
        reg = OntologyRegistry()
        c = reg.get_class("education", EntityType.PAPER)
        assert c is not None
        assert c.has_property("doi") is True

    def test_教育本体_论文有title必需属性(self) -> None:
        """PAPER 类应有 title 必需属性."""
        reg = OntologyRegistry()
        c = reg.get_class("education", EntityType.PAPER)
        title_prop = c.get_property("title")
        assert title_prop is not None
        assert title_prop.required is True

    # --- 通用本体 ---

    def test_通用本体_基本信息(self) -> None:
        reg = OntologyRegistry()
        onto = reg.get_ontology("general")
        assert onto is not None
        assert onto.display_name == "通用本体"
        assert onto.class_count() == 4

    def test_通用本体_概念有definition属性(self) -> None:
        """CONCEPT 类应有 definition 属性."""
        reg = OntologyRegistry()
        c = reg.get_class("general", EntityType.CONCEPT)
        assert c is not None
        assert c.has_property("definition") is True
        def_prop = c.get_property("definition")
        assert def_prop is not None
        assert def_prop.required is True

    def test_通用本体_概念有category属性(self) -> None:
        """CONCEPT 类应有 category 属性."""
        reg = OntologyRegistry()
        c = reg.get_class("general", EntityType.CONCEPT)
        assert c.has_property("category") is True

    def test_通用本体_有RELATED_TO对称关系(self) -> None:
        """通用本体应定义 RELATED_TO 对称关系."""
        reg = OntologyRegistry()
        onto = reg.get_ontology("general")
        rel = onto.get_relation("related_to")
        assert rel is not None
        assert rel.symmetric is True

    def test_通用本体_有EQUIVALENT_TO传递对称关系(self) -> None:
        """通用本体应定义 EQUIVALENT_TO 传递且对称关系."""
        reg = OntologyRegistry()
        onto = reg.get_ontology("general")
        rel = onto.get_relation("equivalent_to")
        assert rel is not None
        assert rel.transitive is True
        assert rel.symmetric is True

    def test_通用本体_全局属性有name必需(self) -> None:
        """通用本体全局属性应包含 name (必需)."""
        reg = OntologyRegistry()
        onto = reg.get_ontology("general")
        name_prop = next(
            (p for p in onto.global_properties if p.name == "name"), None
        )
        assert name_prop is not None
        assert name_prop.required is True


# ============================================================
# 21. 集成场景测试
# ============================================================


class TestIntegrationScenarios:
    """集成场景测试 — 端到端验证 L3 模型的实际使用."""

    def test_化学知识完整建模(self) -> None:
        """化学知识完整建模: 化合物实体 → 属性三元组 → 质量评分 → 溯源 → 本体验证."""
        # 1. 创建数据源
        source = KnowledgeSource(
            source_id="nist",
            name="NIST WebBook",
            tier=SourceTier.TIER1_PUBLIC,
            reliability=0.95,
            last_synced=time.time(),
        )
        assert source.is_stale() is False

        # 2. 创建化合物实体
        entity = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND,
            name="水",
            domain="chemistry",
            identifiers={"cas": "7732-18-5"},
            properties={"formula": "H2O", "molecular_weight": 18.015},
            source=source,
        )
        assert entity.entity_id.startswith("e-")
        assert entity.has_identifier("cas")

        # 3. 添加属性三元组
        triple_formula = KnowledgeTriple(
            subject_id=entity.entity_id,
            predicate=RelationType.HAS_PROPERTY.value,
            object_value="H2O",
            object_is_literal=True,
            rank=StatementRank.PREFERRED,
        )
        triple_bp = KnowledgeTriple(
            subject_id=entity.entity_id,
            predicate=RelationType.HAS_PROPERTY.value,
            object_value=100.0,
            object_is_literal=True,
            qualifiers=[KnowledgeQualifier(name="condition", value="1 atm")],
        )
        entity.add_triple(triple_formula)
        entity.add_triple(triple_bp)

        # 验证三元组方法
        assert len(entity.get_triples_by_predicate(RelationType.HAS_PROPERTY.value)) == 2
        assert len(entity.get_preferred_triples()) == 1
        assert len(entity.get_active_triples()) == 2

        # 4. 质量评分
        entity.quality = QualityScore(accuracy=0.95, trustworthiness=0.95)
        assert entity.quality.is_acceptable() is True

        # 5. 溯源信息
        entity.provenance = ProvenanceInfo(
            entity_id=entity.entity_id,
            primary_source="https://webbook.nist.gov",
            activity_type="retrieve",
        )
        assert entity.provenance.has_derivation_chain() is True
        assert entity.provenance.is_original() is False

        # 6. 本体验证
        registry = OntologyRegistry()
        violations = registry.validate_properties(
            "chemistry", EntityType.CHEMICAL_COMPOUND, entity.properties
        )
        assert violations == []

    def test_化学知识_本体验证失败(self) -> None:
        """缺少必需属性 formula 时本体验证应失败."""
        registry = OntologyRegistry()
        violations = registry.validate_properties(
            "chemistry", EntityType.CHEMICAL_COMPOUND,
            {"molecular_weight": 18.0},  # 缺少 formula
        )
        assert any("formula" in v for v in violations)

    def test_化学知识_关系验证(self) -> None:
        """验证化学本体的 EQUIVALENT_TO 关系."""
        registry = OntologyRegistry()
        # 合法: CHEMICAL_COMPOUND -> CHEMICAL_COMPOUND
        assert registry.validate_relation(
            "chemistry", "equivalent_to",
            EntityType.CHEMICAL_COMPOUND, EntityType.CHEMICAL_COMPOUND,
        ) is True
        # 非法: MATERIAL -> CHEMICAL_COMPOUND (MATERIAL 不在定义域)
        assert registry.validate_relation(
            "chemistry", "equivalent_to",
            EntityType.MATERIAL, EntityType.CHEMICAL_COMPOUND,
        ) is False

    def test_文档切片场景(self) -> None:
        """文档切片场景: 创建切片 → 添加关系 → 添加向量 → 质量评分."""
        # 1. 创建切片
        chunk = DocumentChunk(
            document_id="doc-001",
            content="水是一种无机物，化学式为H2O，是生命必需的物质。",
            content_type=ContentModality.TEXT,
            chunk_index=0,
            section="1.1",
            page=1,
        )
        assert chunk.char_count > 0
        assert chunk.token_count > 0
        assert chunk.has_embedding() is False

        # 2. 添加关系
        parent_rel = ChunkRelationship(
            relation_type=ChunkRelationshipType.PARENT,
            target_chunk_id="c-parent-001",
        )
        next_rel = ChunkRelationship(
            relation_type=ChunkRelationshipType.NEXT,
            target_chunk_id="c-next-001",
        )
        chunk.relationships = [parent_rel, next_rel]
        assert chunk.get_parent_chunk_id() == "c-parent-001"

        # 3. 添加向量
        chunk.embedding = EmbeddingVector(
            content_id=chunk.chunk_id,
            vector=[0.1, 0.2, 0.3, 0.4],
        )
        assert chunk.has_embedding() is True
        assert chunk.embedding.dim == 4

        # 4. 质量评分
        chunk.quality = QualityScore(accuracy=0.9, completeness=0.8)
        assert chunk.quality.is_acceptable() is True

        # 5. 溯源信息
        chunk.provenance = ProvenanceInfo(
            entity_id=chunk.chunk_id,
            generated_by_activity="act-chunk-001",
            activity_type="chunk",
        )
        assert chunk.provenance.is_original() is True

    def test_文档切片_来源文档(self) -> None:
        """切片的 get_source_document 应正确返回来源."""
        chunk = DocumentChunk(
            document_id="doc-001",
            content="测试内容",
            metadata={"source": "高等数学教材.pdf"},
        )
        assert chunk.get_source_document() == "高等数学教材.pdf"

    def test_多源知识融合(self) -> None:
        """多源知识融合: 两个不同来源的同一化合物 → EQUIVALENT_TO 关系."""
        # 1. 两个不同来源
        source_nist = KnowledgeSource(
            source_id="nist", name="NIST",
            tier=SourceTier.TIER1_PUBLIC, reliability=0.95,
        )
        source_pubchem = KnowledgeSource(
            source_id="pubchem", name="PubChem",
            tier=SourceTier.TIER1_PUBLIC, reliability=0.90,
        )

        # 2. 两个实体代表同一化合物
        entity_nist = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND,
            name="水",
            domain="chemistry",
            identifiers={"cas": "7732-18-5"},
            properties={"formula": "H2O"},
            source=source_nist,
            quality=QualityScore(accuracy=0.95, trustworthiness=0.95),
        )
        entity_pubchem = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND,
            name="Water",
            domain="chemistry",
            identifiers={"cid": "962"},
            properties={"formula": "H2O"},
            source=source_pubchem,
            quality=QualityScore(accuracy=0.90, trustworthiness=0.90),
        )

        # 3. EQUIVALENT_TO 关系
        equiv_triple = KnowledgeTriple(
            subject_id=entity_nist.entity_id,
            predicate=RelationType.EQUIVALENT_TO.value,
            object_id=entity_pubchem.entity_id,
        )
        entity_nist.add_triple(equiv_triple)

        # 4. 验证
        equiv_triples = entity_nist.get_triples_by_predicate(
            RelationType.EQUIVALENT_TO.value
        )
        assert len(equiv_triples) == 1
        assert equiv_triples[0].object_id == entity_pubchem.entity_id

        # 5. 本体验证关系合法
        registry = OntologyRegistry()
        assert registry.validate_relation(
            "chemistry", RelationType.EQUIVALENT_TO.value,
            EntityType.CHEMICAL_COMPOUND, EntityType.CHEMICAL_COMPOUND,
        ) is True

        # 6. 两个来源的实体都通过本体验证
        violations_nist = registry.validate_properties(
            "chemistry", EntityType.CHEMICAL_COMPOUND, entity_nist.properties
        )
        assert violations_nist == []
        violations_pubchem = registry.validate_properties(
            "chemistry", EntityType.CHEMICAL_COMPOUND, entity_pubchem.properties
        )
        assert violations_pubchem == []

    def test_多源知识_质量对比(self) -> None:
        """多源知识融合后应能对比质量分数."""
        e1 = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="A",
            quality=QualityScore(accuracy=0.95, trustworthiness=0.95,
                                 consistency=0.9, timeliness=0.9,
                                 completeness=0.9, relevancy=0.9),
        )
        e2 = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="B",
            quality=QualityScore(accuracy=0.80, trustworthiness=0.80,
                                 consistency=0.8, timeliness=0.8,
                                 completeness=0.7, relevancy=0.8),
        )
        assert e1.quality is not None
        assert e2.quality is not None
        assert e1.quality.overall() > e2.quality.overall()
        assert e1.quality.is_acceptable(0.8) is True

    def test_知识库统计_聚合场景(self) -> None:
        """知识库统计应正确聚合多个实体和切片."""
        # 创建多个实体
        entities = [
            KnowledgeEntity(entity_type=EntityType.CHEMICAL_COMPOUND, name=f"化合物{i}")
            for i in range(3)
        ]
        entities.append(KnowledgeEntity(entity_type=EntityType.MATERIAL, name="材料A"))

        # 创建切片
        chunks = [
            DocumentChunk(document_id="doc-1", content=f"内容{i}")
            for i in range(5)
        ]

        stats = KnowledgeBaseStats(
            total_entities=len(entities),
            total_chunks=len(chunks),
            total_triples=10,
            total_sources=2,
            entities_by_type={
                EntityType.CHEMICAL_COMPOUND.value: 3,
                EntityType.MATERIAL.value: 1,
            },
            chunks_by_modality={ContentModality.TEXT.value: 5},
            avg_quality=0.85,
            indexed_vectors=5,
        )
        assert stats.is_empty() is False
        assert stats.total_entities == 4
        assert stats.total_chunks == 5
        assert stats.entities_by_type[EntityType.CHEMICAL_COMPOUND.value] == 3

    def test_检索过滤_端到端(self) -> None:
        """检索过滤器端到端: 创建多个实体 → 过滤 → 验证结果."""
        source_good = KnowledgeSource(
            source_id="nist", name="NIST",
            tier=SourceTier.TIER1_PUBLIC, reliability=0.95,
        )
        source_internal = KnowledgeSource(
            source_id="internal", name="内部文档",
            tier=SourceTier.INTERNAL_DOCUMENT, reliability=0.6,
        )

        e_chem_good = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="水",
            domain="chemistry", source=source_good,
            quality=QualityScore(accuracy=0.95),
        )
        e_chem_low = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="过氧化氢",
            domain="chemistry", source=source_internal,
            quality=QualityScore(accuracy=0.3),
        )
        e_material = KnowledgeEntity(
            entity_type=EntityType.MATERIAL, name="钢",
            domain="materials", source=source_good,
            quality=QualityScore(accuracy=0.9),
        )

        # 过滤: 化学领域 + TIER1来源 + 质量达标
        f = RetrievalFilter(
            domain="chemistry",
            source_tiers=[SourceTier.TIER1_PUBLIC],
            min_quality=0.5,
        )
        assert f.matches_entity(e_chem_good) is True
        assert f.matches_entity(e_chem_low) is False  # 来源不匹配
        assert f.matches_entity(e_material) is False  # 领域不匹配

    def test_批量导入结果_场景(self) -> None:
        """批量导入结果场景: 模拟从 PDF 导入多个切片."""
        result = IngestResult(
            source="pdf://教材.pdf",
            total=20,
            success=15,
            failed=3,
            skipped=2,
            errors=[
                {"chunk_id": "c-1", "reason": "解析失败"},
                {"chunk_id": "c-2", "reason": "编码错误"},
                {"chunk_id": "c-3", "reason": "内容为空"},
            ],
            ingested_ids=[f"c-{i}" for i in range(4, 19)],
            duration_ms=1250.5,
        )
        assert result.is_full_success() is False
        assert result.success_rate() == 0.75
        assert len(result.errors) == 3
        assert len(result.ingested_ids) == 15
