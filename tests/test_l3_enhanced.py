"""L3 领域知识层增强功能测试套件.

覆盖 L3 层新增功能:
- 5 个新异常类 (ConflictError ~ EntityMergeError, 错误码 -32410 ~ -32414)
- 7 个新枚举 (KnowledgeStatus, ConflictType, ConflictResolutionStrategy, QueryOperator,
  PropertyDataType, InferenceRuleType, VerificationStatus)
- 知识图谱容器 (KnowledgeGraph)
- 知识冲突管理 (KnowledgeConflict)
- 版本管理 (KnowledgeVersion, ChangeRecord)
- 证据记录 (EvidenceRecord)
- 结构化查询 (KnowledgeQuery, QueryCondition)
- 子图提取配置 (SubgraphConfig)
- SHACL 风格属性验证
- 类层级管理 (继承属性/关系)
- 本体推理引擎 (传递闭包/逆关系/对称闭包/属性链/子类继承)
- 本体公理验证 (不相交/函数性)
- 跨本体映射
- 增强注册中心
- 复杂集成场景
"""

from __future__ import annotations

import logging
import time

logging.disable(logging.CRITICAL)

import pytest
from pydantic import ValidationError

from dy3_polaris.l3 import *  # noqa: F401,F403
from dy3_polaris.l3.exceptions import (
    ConflictError,
    EntityMergeError,
    InferenceError,
    L3Error,
    QueryError,
    VersionConflictError,
)
from dy3_polaris.l3.models import (
    ChangeRecord,
    ConflictResolutionStrategy,
    ConflictType,
    DocumentChunk,
    EmbeddingVector,
    EntityType,
    EvidenceRecord,
    InferenceRuleType,
    KnowledgeConflict,
    KnowledgeEntity,
    KnowledgeGraph,
    KnowledgeQuery,
    KnowledgeStatus,
    KnowledgeTriple,
    KnowledgeVersion,
    PropertyDataType,
    ProvenanceInfo,
    QualityScore,
    QueryCondition,
    QueryOperator,
    RelationType,
    SubgraphConfig,
    VerificationStatus,
)
from dy3_polaris.l3.ontology import (
    DomainOntology,
    DomainType,
    OntologyAxiom,
    OntologyClass,
    OntologyMapping,
    OntologyProperty,
    OntologyRegistry,
    OntologyRelation,
    OntologyRule,
)
from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 测试辅助: 构建自定义本体
# ============================================================


def _build_hierarchy_ontology() -> DomainOntology:
    """构建含三层继承层级的测试本体 (CONCEPT → CHEMICAL_COMPOUND → MATERIAL)."""
    return DomainOntology(
        ontology_id="onto-hier",
        domain="hierarchy_test",
        display_name="层级测试本体",
        classes=[
            OntologyClass(
                class_id="cls-root",
                entity_type=EntityType.CONCEPT,
                display_name="根类",
                properties=[
                    OntologyProperty(name="base_prop", display_name="基础属性", required=True),
                    OntologyProperty(name="shared_prop", display_name="共享属性"),
                ],
                allowed_relations=[RelationType.RELATED_TO, RelationType.PART_OF],
            ),
            OntologyClass(
                class_id="cls-child",
                entity_type=EntityType.CHEMICAL_COMPOUND,
                display_name="子类",
                parent_type=EntityType.CONCEPT,
                properties=[
                    OntologyProperty(name="child_prop", display_name="子类属性"),
                    OntologyProperty(name="shared_prop", display_name="覆盖属性"),
                ],
                allowed_relations=[RelationType.CITES],
            ),
            OntologyClass(
                class_id="cls-grandchild",
                entity_type=EntityType.MATERIAL,
                display_name="孙类",
                parent_type=EntityType.CHEMICAL_COMPOUND,
                properties=[
                    OntologyProperty(name="grandchild_prop", display_name="孙类属性"),
                ],
                allowed_relations=[RelationType.DERIVED_FROM],
            ),
        ],
        relations=[
            OntologyRelation(name=RelationType.RELATED_TO.value, symmetric=True),
            OntologyRelation(name=RelationType.PART_OF.value, transitive=True),
            OntologyRelation(name=RelationType.CITES.value),
            OntologyRelation(name=RelationType.DERIVED_FROM.value, transitive=True),
        ],
    )


def _build_inverse_ontology() -> DomainOntology:
    """构建含逆关系定义的测试本体."""
    return DomainOntology(
        ontology_id="onto-inv",
        domain="inverse_test",
        display_name="逆关系测试本体",
        classes=[
            OntologyClass(
                class_id="cls-inv",
                entity_type=EntityType.CONCEPT,
                display_name="测试类",
            ),
        ],
        relations=[
            OntologyRelation(
                name="parent_of",
                display_name="父级",
                inverse_of="child_of",
            ),
            OntologyRelation(
                name="child_of",
                display_name="子级",
                inverse_of="parent_of",
            ),
            OntologyRelation(name="non_inverse", display_name="无逆关系"),
        ],
    )


def _build_rules_ontology() -> DomainOntology:
    """构建含推理规则的测试本体."""
    return DomainOntology(
        ontology_id="onto-rules",
        domain="rules_test",
        display_name="规则测试本体",
        classes=[
            OntologyClass(
                class_id="cls-rules",
                entity_type=EntityType.CONCEPT,
                display_name="规则测试类",
            ),
        ],
        relations=[
            OntologyRelation(name="part_of", transitive=True),
            OntologyRelation(name="related_to", symmetric=True),
            OntologyRelation(name="plain_rel"),
        ],
        inference_rules=[
            OntologyRule(
                rule_type=InferenceRuleType.TRANSITIVE_CLOSURE,
                applies_to_relation="part_of",
            ),
            OntologyRule(
                rule_type=InferenceRuleType.SYMMETRIC_CLOSURE,
                applies_to_relation="related_to",
            ),
        ],
    )


def _build_shacl_ontology() -> DomainOntology:
    """构建含 SHACL 约束的测试本体."""
    return DomainOntology(
        ontology_id="onto-shacl",
        domain="shacl_test",
        display_name="SHACL测试本体",
        classes=[
            OntologyClass(
                class_id="cls-shacl",
                entity_type=EntityType.CONCEPT,
                display_name="SHACL测试类",
                properties=[
                    OntologyProperty(
                        name="color",
                        display_name="颜色",
                        enum_values=["red", "green", "blue"],
                    ),
                    OntologyProperty(
                        name="age",
                        display_name="年龄",
                        data_type=PropertyDataType.INTEGER,
                        min_value=0,
                        max_value=150,
                    ),
                    OntologyProperty(
                        name="label",
                        display_name="标签",
                        data_type=PropertyDataType.STRING,
                        min_length=2,
                        max_length=20,
                    ),
                    OntologyProperty(
                        name="email",
                        display_name="邮箱",
                        pattern=r"^[^@]+@[^@]+\.[^@]+$",
                    ),
                    OntologyProperty(
                        name="score",
                        display_name="分数",
                        data_type=PropertyDataType.FLOAT,
                        min_value=0.0,
                        max_value=100.0,
                    ),
                    OntologyProperty(
                        name="active",
                        display_name="是否活跃",
                        data_type=PropertyDataType.BOOLEAN,
                    ),
                    OntologyProperty(
                        name="timestamp",
                        display_name="时间戳",
                        data_type=PropertyDataType.DATETIME,
                    ),
                    OntologyProperty(
                        name="tags_list",
                        display_name="标签列表",
                        min_count=1,
                        max_count=5,
                    ),
                    OntologyProperty(
                        name="single_val",
                        display_name="单值属性",
                        max_count=1,
                    ),
                ],
            ),
        ],
        axioms=[
            OntologyAxiom(
                axiom_type="disjoint",
                subject="concept",
                object="material",
                description="概念和材料不相交",
            ),
            OntologyAxiom(
                axiom_type="functional",
                subject="single_val",
                description="single_val 是函数性属性",
            ),
        ],
        mappings=[
            OntologyMapping(
                source_domain="shacl_test",
                target_domain="chemistry",
                mapping_type="equivalent",
                source_entity_type=EntityType.CONCEPT,
                target_entity_type=EntityType.CHEMICAL_COMPOUND,
            ),
            OntologyMapping(
                source_domain="shacl_test",
                target_domain="materials",
                mapping_type="related",
                source_entity_type=EntityType.CONCEPT,
                target_entity_type=EntityType.MATERIAL,
            ),
        ],
    )


def _make_entity(**kwargs) -> KnowledgeEntity:
    """创建测试实体的便捷函数."""
    defaults = {
        "entity_type": EntityType.CHEMICAL_COMPOUND,
        "name": "测试实体",
    }
    defaults.update(kwargs)
    return KnowledgeEntity(**defaults)


# ============================================================
# 1. 新异常测试
# ============================================================


class TestNewExceptions:
    """测试 5 个新异常类: ConflictError, VersionConflictError, QueryError, InferenceError, EntityMergeError."""

    # --- ConflictError (-32410) ---

    def test_conflict_error_实例化(self) -> None:
        """ConflictError 应正确实例化."""
        err = ConflictError(conflict_type="temporal", entity_id="e-001")
        assert err.conflict_type == "temporal"
        assert err.entity_id == "e-001"

    def test_conflict_error_错误码(self) -> None:
        """ConflictError 的 JSON-RPC 错误码应为 -32410."""
        assert ConflictError()._jsonrpc_code() == -32410

    def test_conflict_error_继承_l3_error(self) -> None:
        """ConflictError 应继承 L3Error 和 L6Error."""
        assert issubclass(ConflictError, L3Error)
        assert issubclass(ConflictError, L6Error)

    def test_conflict_error_默认detail(self) -> None:
        """ConflictError 默认 detail 应包含 conflict_type 和 entity_id."""
        err = ConflictError(conflict_type="semantic", entity_id="e-002")
        assert "semantic" in err.detail
        assert "e-002" in err.detail

    def test_conflict_error_自定义detail(self) -> None:
        """ConflictError 应支持自定义 detail."""
        err = ConflictError(conflict_type="temporal", entity_id="e-003", detail="自定义冲突")
        assert err.detail == "自定义冲突"

    def test_conflict_error_json_rpc_error(self) -> None:
        """ConflictError 的 JSON-RPC 错误对象."""
        err = ConflictError(conflict_type="source_based", entity_id="e-004")
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32410
        assert rpc["data"]["conflict_type"] == "source_based"
        assert rpc["data"]["entity_id"] == "e-004"

    def test_conflict_error_可被抛出和捕获(self) -> None:
        """ConflictError 应能被 try/except 捕获."""
        with pytest.raises(ConflictError) as exc_info:
            raise ConflictError(conflict_type="temporal", entity_id="e-005")
        assert exc_info.value.conflict_type == "temporal"

    def test_conflict_error_可被_l3_error捕获(self) -> None:
        """ConflictError 应能被 L3Error 捕获."""
        with pytest.raises(L3Error):
            raise ConflictError()

    # --- VersionConflictError (-32411) ---

    def test_version_conflict_error_实例化(self) -> None:
        """VersionConflictError 应正确设置属性."""
        err = VersionConflictError(entity_id="e-100", expected_version=3, actual_version=2)
        assert err.entity_id == "e-100"
        assert err.expected_version == 3
        assert err.actual_version == 2

    def test_version_conflict_error_错误码(self) -> None:
        """VersionConflictError 的 JSON-RPC 错误码应为 -32411."""
        assert VersionConflictError()._jsonrpc_code() == -32411

    def test_version_conflict_error_继承关系(self) -> None:
        """VersionConflictError 应继承 L3Error."""
        assert issubclass(VersionConflictError, L3Error)

    def test_version_conflict_error_默认detail(self) -> None:
        """VersionConflictError 默认 detail 应包含版本信息."""
        err = VersionConflictError(entity_id="e-101", expected_version=5, actual_version=4)
        assert "e-101" in err.detail
        assert "v5" in err.detail
        assert "v4" in err.detail

    def test_version_conflict_error_json_rpc_error(self) -> None:
        """VersionConflictError 的 JSON-RPC 错误对象."""
        err = VersionConflictError(entity_id="e-102", expected_version=2, actual_version=1)
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32411
        assert rpc["data"]["entity_id"] == "e-102"
        assert rpc["data"]["expected_version"] == 2
        assert rpc["data"]["actual_version"] == 1

    def test_version_conflict_error_上下文传递(self) -> None:
        """VersionConflictError 应支持额外 context."""
        err = VersionConflictError(
            entity_id="e-103", expected_version=1, actual_version=2, context={"extra": "info"}
        )
        assert err.context["extra"] == "info"

    # --- QueryError (-32412) ---

    def test_query_error_实例化(self) -> None:
        """QueryError 应正确设置 query 和 reason."""
        err = QueryError(query="SELECT * FROM kg", reason="语法错误")
        assert err.query == "SELECT * FROM kg"
        assert err.reason == "语法错误"

    def test_query_error_错误码(self) -> None:
        """QueryError 的 JSON-RPC 错误码应为 -32412."""
        assert QueryError()._jsonrpc_code() == -32412

    def test_query_error_继承关系(self) -> None:
        """QueryError 应继承 L3Error."""
        assert issubclass(QueryError, L3Error)

    def test_query_error_默认detail(self) -> None:
        """QueryError 默认 detail 应包含 query 和 reason."""
        err = QueryError(query="q1", reason="超时")
        assert "q1" in err.detail
        assert "超时" in err.detail

    def test_query_error_json_rpc_error(self) -> None:
        """QueryError 的 JSON-RPC 错误对象."""
        err = QueryError(query="q2", reason="无效条件")
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32412
        assert rpc["data"]["query"] == "q2"
        assert rpc["data"]["reason"] == "无效条件"

    # --- InferenceError (-32413) ---

    def test_inference_error_实例化(self) -> None:
        """InferenceError 应正确设置 rule_type."""
        err = InferenceError(rule_type="transitive_closure")
        assert err.rule_type == "transitive_closure"

    def test_inference_error_错误码(self) -> None:
        """InferenceError 的 JSON-RPC 错误码应为 -32413."""
        assert InferenceError()._jsonrpc_code() == -32413

    def test_inference_error_继承关系(self) -> None:
        """InferenceError 应继承 L3Error."""
        assert issubclass(InferenceError, L3Error)

    def test_inference_error_默认detail(self) -> None:
        """InferenceError 默认 detail 应包含 rule_type."""
        err = InferenceError(rule_type="inverse_relation")
        assert "inverse_relation" in err.detail

    def test_inference_error_json_rpc_error(self) -> None:
        """InferenceError 的 JSON-RPC 错误对象."""
        err = InferenceError(rule_type="symmetric_closure")
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32413
        assert rpc["data"]["rule_type"] == "symmetric_closure"

    # --- EntityMergeError (-32414) ---

    def test_entity_merge_error_实例化(self) -> None:
        """EntityMergeError 应正确设置属性."""
        err = EntityMergeError(
            source_entity_id="e-src", target_entity_id="e-tgt", reason="不可调和冲突"
        )
        assert err.source_entity_id == "e-src"
        assert err.target_entity_id == "e-tgt"
        assert err.reason == "不可调和冲突"

    def test_entity_merge_error_错误码(self) -> None:
        """EntityMergeError 的 JSON-RPC 错误码应为 -32414."""
        assert EntityMergeError()._jsonrpc_code() == -32414

    def test_entity_merge_error_继承关系(self) -> None:
        """EntityMergeError 应继承 L3Error."""
        assert issubclass(EntityMergeError, L3Error)

    def test_entity_merge_error_默认detail(self) -> None:
        """EntityMergeError 默认 detail 应包含源/目标 ID 和原因."""
        err = EntityMergeError(
            source_entity_id="e-s", target_entity_id="e-t", reason="类型不一致"
        )
        assert "e-s" in err.detail
        assert "e-t" in err.detail
        assert "类型不一致" in err.detail

    def test_entity_merge_error_json_rpc_error(self) -> None:
        """EntityMergeError 的 JSON-RPC 错误对象."""
        err = EntityMergeError(
            source_entity_id="e-s2", target_entity_id="e-t2", reason="冲突"
        )
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32414
        assert rpc["data"]["source_entity_id"] == "e-s2"
        assert rpc["data"]["target_entity_id"] == "e-t2"
        assert rpc["data"]["reason"] == "冲突"

    # --- 错误码唯一性 ---

    def test_五个新异常错误码各不相同(self) -> None:
        """5 个新异常的 JSON-RPC 错误码应各不相同."""
        codes = [
            ConflictError()._jsonrpc_code(),
            VersionConflictError()._jsonrpc_code(),
            QueryError()._jsonrpc_code(),
            InferenceError()._jsonrpc_code(),
            EntityMergeError()._jsonrpc_code(),
        ]
        assert len(set(codes)) == 5
        assert codes == [-32410, -32411, -32412, -32413, -32414]


# ============================================================
# 2. 新枚举测试
# ============================================================


class TestNewEnums:
    """测试 7 个新枚举的值、str 行为和唯一性."""

    # --- KnowledgeStatus ---

    def test_knowledge_status_所有值存在(self) -> None:
        """KnowledgeStatus 应包含 DRAFT, ACTIVE, ARCHIVED, DEPRECATED."""
        assert KnowledgeStatus.DRAFT == "draft"
        assert KnowledgeStatus.ACTIVE == "active"
        assert KnowledgeStatus.ARCHIVED == "archived"
        assert KnowledgeStatus.DEPRECATED == "deprecated"

    def test_knowledge_status_是str枚举(self) -> None:
        """KnowledgeStatus 应是 (str, Enum) 类型."""
        assert isinstance(KnowledgeStatus.ACTIVE, str)
        assert KnowledgeStatus.ACTIVE == "active"

    def test_knowledge_status_值唯一(self) -> None:
        """KnowledgeStatus 所有值应唯一."""
        values = [m.value for m in KnowledgeStatus]
        assert len(values) == len(set(values))

    # --- ConflictType ---

    def test_conflict_type_所有值存在(self) -> None:
        """ConflictType 应包含 TEMPORAL, SOURCE_BASED, SEMANTIC."""
        assert ConflictType.TEMPORAL == "temporal"
        assert ConflictType.SOURCE_BASED == "source_based"
        assert ConflictType.SEMANTIC == "semantic"

    def test_conflict_type_是str枚举(self) -> None:
        """ConflictType 应是 (str, Enum) 类型."""
        assert isinstance(ConflictType.TEMPORAL, str)

    def test_conflict_type_值唯一(self) -> None:
        """ConflictType 所有值应唯一."""
        values = [m.value for m in ConflictType]
        assert len(values) == len(set(values))

    # --- ConflictResolutionStrategy ---

    def test_conflict_resolution_strategy_所有值存在(self) -> None:
        """ConflictResolutionStrategy 应包含全部 5 个策略."""
        assert ConflictResolutionStrategy.KEEP_BOTH == "keep_both"
        assert ConflictResolutionStrategy.PREFER_HIGHER_QUALITY == "prefer_higher_quality"
        assert ConflictResolutionStrategy.PREFER_MOST_RECENT == "prefer_most_recent"
        assert ConflictResolutionStrategy.PREFER_MOST_TRUSTED == "prefer_most_trusted"
        assert ConflictResolutionStrategy.MANUAL_REVIEW == "manual_review"

    def test_conflict_resolution_strategy_是str枚举(self) -> None:
        """ConflictResolutionStrategy 应是 (str, Enum) 类型."""
        assert isinstance(ConflictResolutionStrategy.KEEP_BOTH, str)

    def test_conflict_resolution_strategy_值唯一(self) -> None:
        """ConflictResolutionStrategy 所有值应唯一."""
        values = [m.value for m in ConflictResolutionStrategy]
        assert len(values) == len(set(values))

    # --- QueryOperator ---

    def test_query_operator_所有值存在(self) -> None:
        """QueryOperator 应包含全部 11 个操作符."""
        assert QueryOperator.EQ == "eq"
        assert QueryOperator.NE == "ne"
        assert QueryOperator.GT == "gt"
        assert QueryOperator.GTE == "gte"
        assert QueryOperator.LT == "lt"
        assert QueryOperator.LTE == "lte"
        assert QueryOperator.CONTAINS == "contains"
        assert QueryOperator.STARTS_WITH == "starts_with"
        assert QueryOperator.ENDS_WITH == "ends_with"
        assert QueryOperator.IN == "in"
        assert QueryOperator.REGEX == "regex"

    def test_query_operator_是str枚举(self) -> None:
        """QueryOperator 应是 (str, Enum) 类型."""
        assert isinstance(QueryOperator.EQ, str)

    def test_query_operator_值唯一(self) -> None:
        """QueryOperator 所有值应唯一."""
        values = [m.value for m in QueryOperator]
        assert len(values) == len(set(values))
        assert len(values) == 11

    # --- PropertyDataType ---

    def test_property_data_type_所有值存在(self) -> None:
        """PropertyDataType 应包含全部 7 个数据类型."""
        assert PropertyDataType.STRING == "string"
        assert PropertyDataType.INTEGER == "integer"
        assert PropertyDataType.FLOAT == "float"
        assert PropertyDataType.BOOLEAN == "boolean"
        assert PropertyDataType.DATETIME == "datetime"
        assert PropertyDataType.ENTITY_REF == "entity_ref"
        assert PropertyDataType.LIST == "list"

    def test_property_data_type_是str枚举(self) -> None:
        """PropertyDataType 应是 (str, Enum) 类型."""
        assert isinstance(PropertyDataType.STRING, str)

    def test_property_data_type_值唯一(self) -> None:
        """PropertyDataType 所有值应唯一."""
        values = [m.value for m in PropertyDataType]
        assert len(values) == len(set(values))

    # --- InferenceRuleType ---

    def test_inference_rule_type_所有值存在(self) -> None:
        """InferenceRuleType 应包含全部 5 个规则类型."""
        assert InferenceRuleType.TRANSITIVE_CLOSURE == "transitive_closure"
        assert InferenceRuleType.INVERSE_RELATION == "inverse_relation"
        assert InferenceRuleType.SUBCLASS_INHERITANCE == "subclass_inheritance"
        assert InferenceRuleType.SYMMETRIC_CLOSURE == "symmetric_closure"
        assert InferenceRuleType.PROPERTY_CHAIN == "property_chain"

    def test_inference_rule_type_是str枚举(self) -> None:
        """InferenceRuleType 应是 (str, Enum) 类型."""
        assert isinstance(InferenceRuleType.TRANSITIVE_CLOSURE, str)

    def test_inference_rule_type_值唯一(self) -> None:
        """InferenceRuleType 所有值应唯一."""
        values = [m.value for m in InferenceRuleType]
        assert len(values) == len(set(values))

    # --- VerificationStatus ---

    def test_verification_status_所有值存在(self) -> None:
        """VerificationStatus 应包含 UNVERIFIED, CANDIDATE, VERIFIED, DISPUTED."""
        assert VerificationStatus.UNVERIFIED == "unverified"
        assert VerificationStatus.CANDIDATE == "candidate"
        assert VerificationStatus.VERIFIED == "verified"
        assert VerificationStatus.DISPUTED == "disputed"

    def test_verification_status_是str枚举(self) -> None:
        """VerificationStatus 应是 (str, Enum) 类型."""
        assert isinstance(VerificationStatus.VERIFIED, str)

    def test_verification_status_值唯一(self) -> None:
        """VerificationStatus 所有值应唯一."""
        values = [m.value for m in VerificationStatus]
        assert len(values) == len(set(values))


# ============================================================
# 3. 知识图谱容器测试
# ============================================================


class TestKnowledgeGraph:
    """测试 KnowledgeGraph 模型的图管理能力."""

    def test_graph_id_默认格式(self) -> None:
        """默认 graph_id 应以 'kg-' 开头."""
        kg = KnowledgeGraph()
        assert kg.graph_id.startswith("kg-")

    def test_graph_id_唯一性(self) -> None:
        """两次创建的 graph_id 应不同."""
        kg1 = KnowledgeGraph()
        kg2 = KnowledgeGraph()
        assert kg1.graph_id != kg2.graph_id

    def test_add_entity_和_get_entity(self) -> None:
        """add_entity 和 get_entity 应正确添加和获取实体."""
        kg = KnowledgeGraph()
        entity = _make_entity()
        kg.add_entity(entity)
        assert kg.get_entity(entity.entity_id) is entity

    def test_get_entity_不存在返回None(self) -> None:
        """获取不存在的实体应返回 None."""
        kg = KnowledgeGraph()
        assert kg.get_entity("nonexistent") is None

    def test_remove_entity_存在(self) -> None:
        """remove_entity 应移除存在的实体并返回 True."""
        kg = KnowledgeGraph()
        entity = _make_entity()
        kg.add_entity(entity)
        assert kg.remove_entity(entity.entity_id) is True
        assert kg.get_entity(entity.entity_id) is None

    def test_remove_entity_不存在(self) -> None:
        """remove_entity 移除不存在的实体应返回 False."""
        kg = KnowledgeGraph()
        assert kg.remove_entity("nonexistent") is False

    def test_add_triple_和_triple_count(self) -> None:
        """add_triple 应添加三元组, triple_count 应统计总数."""
        kg = KnowledgeGraph()
        entity = _make_entity()
        kg.add_entity(entity)
        triple = KnowledgeTriple(
            subject_id=entity.entity_id,
            predicate="related_to",
            object_id="e-other",
        )
        kg.add_triple(triple)
        assert kg.triple_count() == 1

    def test_triple_count_含实体内部三元组(self) -> None:
        """triple_count 应包含实体内部三元组和图谱级三元组."""
        kg = KnowledgeGraph()
        entity = _make_entity()
        entity.add_triple(
            KnowledgeTriple(
                subject_id=entity.entity_id,
                predicate="has_property",
                object_value="H2O",
                object_is_literal=True,
            )
        )
        kg.add_entity(entity)
        kg.add_triple(
            KnowledgeTriple(
                subject_id=entity.entity_id,
                predicate="related_to",
                object_id="e-other",
            )
        )
        # 1 internal + 1 graph-level
        assert kg.triple_count() == 2

    def test_entity_count(self) -> None:
        """entity_count 应返回实体数量."""
        kg = KnowledgeGraph()
        assert kg.entity_count() == 0
        kg.add_entity(_make_entity(name="e1"))
        kg.add_entity(_make_entity(name="e2"))
        assert kg.entity_count() == 2

    def test_active_entity_count(self) -> None:
        """active_entity_count 应只统计 ACTIVE 状态的实体."""
        kg = KnowledgeGraph()
        e1 = _make_entity(name="e1", status=KnowledgeStatus.ACTIVE)
        e2 = _make_entity(name="e2", status=KnowledgeStatus.ARCHIVED)
        e3 = _make_entity(name="e3", status=KnowledgeStatus.ACTIVE)
        kg.add_entity(e1)
        kg.add_entity(e2)
        kg.add_entity(e3)
        assert kg.active_entity_count() == 2

    def test_get_entities_by_type(self) -> None:
        """get_entities_by_type 应按类型过滤实体."""
        kg = KnowledgeGraph()
        e1 = _make_entity(name="e1", entity_type=EntityType.CHEMICAL_COMPOUND)
        e2 = _make_entity(name="e2", entity_type=EntityType.MATERIAL)
        kg.add_entity(e1)
        kg.add_entity(e2)
        chems = kg.get_entities_by_type(EntityType.CHEMICAL_COMPOUND)
        assert len(chems) == 1
        assert chems[0] is e1

    def test_get_entities_by_domain(self) -> None:
        """get_entities_by_domain 应按领域过滤实体."""
        kg = KnowledgeGraph()
        e1 = _make_entity(name="e1", domain="chemistry")
        e2 = _make_entity(name="e2", domain="materials")
        kg.add_entity(e1)
        kg.add_entity(e2)
        chems = kg.get_entities_by_domain("chemistry")
        assert len(chems) == 1
        assert chems[0] is e1

    def test_find_entity_by_name_名称匹配(self) -> None:
        """find_entity_by_name 应按名称查找实体 (忽略大小写)."""
        kg = KnowledgeGraph()
        entity = _make_entity(name="Water")
        kg.add_entity(entity)
        found = kg.find_entity_by_name("water")
        assert found is entity

    def test_find_entity_by_name_别名匹配(self) -> None:
        """find_entity_by_name 应支持别名查找 (忽略大小写)."""
        kg = KnowledgeGraph()
        entity = _make_entity(name="水", aliases=["H2O", "Water"])
        kg.add_entity(entity)
        found = kg.find_entity_by_name("h2o")
        assert found is entity

    def test_find_entity_by_name_不存在(self) -> None:
        """find_entity_by_name 查找不存在的实体应返回 None."""
        kg = KnowledgeGraph()
        assert kg.find_entity_by_name("不存在") is None

    def test_find_entities_by_tag(self) -> None:
        """find_entities_by_tag 应按标签查找实体."""
        kg = KnowledgeGraph()
        e1 = _make_entity(name="e1", tags=["inorganic", "liquid"])
        e2 = _make_entity(name="e2", tags=["organic"])
        kg.add_entity(e1)
        kg.add_entity(e2)
        results = kg.find_entities_by_tag("inorganic")
        assert len(results) == 1
        assert results[0] is e1

    def test_get_stats_返回统计(self) -> None:
        """get_stats 应返回 KnowledgeBaseStats 对象."""
        kg = KnowledgeGraph()
        kg.add_entity(_make_entity(name="e1"))
        stats = kg.get_stats()
        assert stats.total_entities == 1
        assert isinstance(stats, KnowledgeBaseStats)

    def test_get_stats_按类型统计(self) -> None:
        """get_stats 的 entities_by_type 应正确分组."""
        kg = KnowledgeGraph()
        kg.add_entity(_make_entity(name="e1", entity_type=EntityType.CHEMICAL_COMPOUND))
        kg.add_entity(_make_entity(name="e2", entity_type=EntityType.CHEMICAL_COMPOUND))
        kg.add_entity(_make_entity(name="e3", entity_type=EntityType.MATERIAL))
        stats = kg.get_stats()
        assert stats.entities_by_type.get("chemical_compound") == 2
        assert stats.entities_by_type.get("material") == 1

    def test_unresolved_conflict_count(self) -> None:
        """unresolved_conflict_count 应统计未解决冲突."""
        kg = KnowledgeGraph()
        c1 = KnowledgeConflict(
            conflict_type=ConflictType.TEMPORAL, entity_id="e1", status="detected"
        )
        c2 = KnowledgeConflict(
            conflict_type=ConflictType.SEMANTIC, entity_id="e2", status="resolved"
        )
        kg.conflicts = [c1, c2]
        assert kg.unresolved_conflict_count() == 1

    def test_is_empty_空图谱(self) -> None:
        """空图谱 is_empty 应为 True."""
        kg = KnowledgeGraph()
        assert kg.is_empty() is True

    def test_is_empty_非空图谱(self) -> None:
        """有实体的图谱 is_empty 应为 False."""
        kg = KnowledgeGraph()
        kg.add_entity(_make_entity())
        assert kg.is_empty() is False

    def test_touch_更新时间戳(self) -> None:
        """touch() 应更新 updated_at 时间戳."""
        kg = KnowledgeGraph()
        old_time = kg.updated_at
        time.sleep(0.01)
        kg.touch()
        assert kg.updated_at > old_time


# ============================================================
# 4. 知识冲突测试
# ============================================================


class TestKnowledgeConflict:
    """测试 KnowledgeConflict 模型的冲突管理能力."""

    def test_conflict_id_默认格式(self) -> None:
        """默认 conflict_id 应以 'cf-' 开头."""
        c = KnowledgeConflict(conflict_type=ConflictType.TEMPORAL, entity_id="e1")
        assert c.conflict_id.startswith("cf-")

    def test_基本属性(self) -> None:
        """KnowledgeConflict 应正确设置 conflict_type, entity_id, field_path."""
        c = KnowledgeConflict(
            conflict_type=ConflictType.SOURCE_BASED,
            entity_id="e-001",
            field_path="properties.boiling_point",
        )
        assert c.conflict_type == ConflictType.SOURCE_BASED
        assert c.entity_id == "e-001"
        assert c.field_path == "properties.boiling_point"

    def test_conflicting_values(self) -> None:
        """conflicting_values 应存储冲突值列表."""
        values = [
            {"value": 100, "source": "nist"},
            {"value": 99, "source": "cas"},
        ]
        c = KnowledgeConflict(
            conflict_type=ConflictType.TEMPORAL,
            entity_id="e1",
            conflicting_values=values,
        )
        assert len(c.conflicting_values) == 2

    def test_detection_method_默认值(self) -> None:
        """detection_method 默认应为 'manual'."""
        c = KnowledgeConflict(conflict_type=ConflictType.SEMANTIC, entity_id="e1")
        assert c.detection_method == "manual"

    def test_resolution_strategy_默认值(self) -> None:
        """resolution_strategy 默认应为 KEEP_BOTH."""
        c = KnowledgeConflict(conflict_type=ConflictType.SEMANTIC, entity_id="e1")
        assert c.resolution_strategy == ConflictResolutionStrategy.KEEP_BOTH

    def test_status_默认值(self) -> None:
        """status 默认应为 'detected'."""
        c = KnowledgeConflict(conflict_type=ConflictType.SEMANTIC, entity_id="e1")
        assert c.status == "detected"

    def test_status_无效值抛出异常(self) -> None:
        """status 不在 allowed 集合中应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeConflict(
                conflict_type=ConflictType.SEMANTIC, entity_id="e1", status="invalid"
            )

    def test_is_resolved_未解决(self) -> None:
        """status='detected' 时 is_resolved 应为 False."""
        c = KnowledgeConflict(conflict_type=ConflictType.TEMPORAL, entity_id="e1")
        assert c.is_resolved() is False

    def test_is_resolved_已解决(self) -> None:
        """status='resolved' 时 is_resolved 应为 True."""
        c = KnowledgeConflict(
            conflict_type=ConflictType.TEMPORAL, entity_id="e1", status="resolved"
        )
        assert c.is_resolved() is True

    def test_is_ignored(self) -> None:
        """status='ignored' 时 is_ignored 应为 True."""
        c = KnowledgeConflict(
            conflict_type=ConflictType.TEMPORAL, entity_id="e1", status="ignored"
        )
        assert c.is_ignored() is True

    def test_needs_manual_review_需要(self) -> None:
        """策略为 MANUAL_REVIEW 且未解决时 needs_manual_review 应为 True."""
        c = KnowledgeConflict(
            conflict_type=ConflictType.TEMPORAL,
            entity_id="e1",
            resolution_strategy=ConflictResolutionStrategy.MANUAL_REVIEW,
        )
        assert c.needs_manual_review() is True

    def test_needs_manual_review_不需要(self) -> None:
        """策略非 MANUAL_REVIEW 时 needs_manual_review 应为 False."""
        c = KnowledgeConflict(
            conflict_type=ConflictType.TEMPORAL,
            entity_id="e1",
            resolution_strategy=ConflictResolutionStrategy.KEEP_BOTH,
        )
        assert c.needs_manual_review() is False

    def test_needs_manual_review_已解决(self) -> None:
        """策略为 MANUAL_REVIEW 但已解决时 needs_manual_review 应为 False."""
        c = KnowledgeConflict(
            conflict_type=ConflictType.TEMPORAL,
            entity_id="e1",
            resolution_strategy=ConflictResolutionStrategy.MANUAL_REVIEW,
            status="resolved",
        )
        assert c.needs_manual_review() is False

    def test_resolve_方法(self) -> None:
        """resolve() 应设置值、声明ID、说明、状态和时间."""
        c = KnowledgeConflict(conflict_type=ConflictType.TEMPORAL, entity_id="e1")
        c.resolve(value=100, claim_id="cl-001", explanation="采纳NIST数据", resolved_by="agent-1")
        assert c.resolved_value == 100
        assert c.resolved_claim_id == "cl-001"
        assert c.resolution_explanation == "采纳NIST数据"
        assert c.status == "resolved"
        assert c.resolved_at > 0.0
        assert c.resolved_by == "agent-1"
        assert c.is_resolved() is True

    def test_ignore_方法(self) -> None:
        """ignore() 应设置状态为 ignored."""
        c = KnowledgeConflict(conflict_type=ConflictType.TEMPORAL, entity_id="e1")
        c.ignore(reason="低优先级", by="admin")
        assert c.status == "ignored"
        assert c.is_ignored() is True
        assert c.resolution_explanation == "低优先级"
        assert c.resolved_by == "admin"

    def test_conflicting_value_count(self) -> None:
        """conflicting_value_count 应返回冲突值数量."""
        c = KnowledgeConflict(
            conflict_type=ConflictType.SOURCE_BASED,
            entity_id="e1",
            conflicting_values=[{"v": 1}, {"v": 2}, {"v": 3}],
        )
        assert c.conflicting_value_count() == 3


# ============================================================
# 5. 知识版本测试
# ============================================================


class TestKnowledgeVersion:
    """测试 KnowledgeVersion 模型的版本管理能力."""

    def test_version_id_默认格式(self) -> None:
        """默认 version_id 应以 'v-' 开头."""
        v = KnowledgeVersion(entity_id="e1")
        assert v.version_id.startswith("v-")

    def test_基本属性(self) -> None:
        """KnowledgeVersion 应正确设置 entity_id, revision_number, parent_version_id."""
        v = KnowledgeVersion(
            entity_id="e-001",
            revision_number=3,
            parent_version_id="v-parent-001",
        )
        assert v.entity_id == "e-001"
        assert v.revision_number == 3
        assert v.parent_version_id == "v-parent-001"

    def test_changeset_默认为空(self) -> None:
        """changeset 默认应为空列表."""
        v = KnowledgeVersion(entity_id="e1")
        assert v.changeset == []

    def test_changeset_含变更记录(self) -> None:
        """changeset 应能存储 ChangeRecord 列表."""
        changes = [
            ChangeRecord(change_type="modify", entity_id="e1", field_path="name"),
            ChangeRecord(change_type="add", entity_id="e1", field_path="tags"),
        ]
        v = KnowledgeVersion(entity_id="e1", changeset=changes)
        assert len(v.changeset) == 2

    def test_valid_from_和_valid_until_默认值(self) -> None:
        """valid_from 默认为当前时间, valid_until 默认为 0.0."""
        v = KnowledgeVersion(entity_id="e1")
        assert v.valid_from > 0.0
        assert v.valid_until == 0.0

    def test_created_at_和_created_by_默认值(self) -> None:
        """created_at 默认为当前时间, created_by 默认为 'system'."""
        v = KnowledgeVersion(entity_id="e1")
        assert v.created_at > 0.0
        assert v.created_by == "system"

    def test_is_current_当前版本(self) -> None:
        """valid_until=0.0 时 is_current 应为 True."""
        v = KnowledgeVersion(entity_id="e1", valid_until=0.0)
        assert v.is_current() is True

    def test_is_current_历史版本(self) -> None:
        """valid_until>0 时 is_current 应为 False."""
        v = KnowledgeVersion(entity_id="e1", valid_until=1000.0)
        assert v.is_current() is False

    def test_is_valid_at_在范围内(self) -> None:
        """时间戳在有效范围内时 is_valid_at 应为 True."""
        v = KnowledgeVersion(entity_id="e1", valid_from=100.0, valid_until=200.0)
        assert v.is_valid_at(150.0) is True

    def test_is_valid_at_早于生效时间(self) -> None:
        """时间戳早于 valid_from 时 is_valid_at 应为 False."""
        v = KnowledgeVersion(entity_id="e1", valid_from=100.0, valid_until=200.0)
        assert v.is_valid_at(50.0) is False

    def test_is_valid_at_晚于失效时间(self) -> None:
        """时间戳 >= valid_until 时 is_valid_at 应为 False."""
        v = KnowledgeVersion(entity_id="e1", valid_from=100.0, valid_until=200.0)
        assert v.is_valid_at(200.0) is False

    def test_is_valid_at_当前版本无上界(self) -> None:
        """valid_until=0.0 (当前版本) 时任意大于 valid_from 的时间都有效."""
        v = KnowledgeVersion(entity_id="e1", valid_from=100.0, valid_until=0.0)
        assert v.is_valid_at(999999.0) is True

    def test_change_count(self) -> None:
        """change_count 应返回 changeset 长度."""
        v = KnowledgeVersion(
            entity_id="e1",
            changeset=[
                ChangeRecord(change_type="add", entity_id="e1", field_path="f1"),
                ChangeRecord(change_type="delete", entity_id="e1", field_path="f2"),
            ],
        )
        assert v.change_count() == 2

    def test_change_count_空changeset(self) -> None:
        """空 changeset 的 change_count 应为 0."""
        v = KnowledgeVersion(entity_id="e1")
        assert v.change_count() == 0

    def test_has_parent_有父版本(self) -> None:
        """有 parent_version_id 时 has_parent 应为 True."""
        v = KnowledgeVersion(entity_id="e1", parent_version_id="v-parent")
        assert v.has_parent() is True

    def test_has_parent_无父版本(self) -> None:
        """无 parent_version_id 时 has_parent 应为 False."""
        v = KnowledgeVersion(entity_id="e1")
        assert v.has_parent() is False


# ============================================================
# 6. 变更记录测试
# ============================================================


class TestChangeRecord:
    """测试 ChangeRecord 模型."""

    def test_基本属性(self) -> None:
        """ChangeRecord 应正确设置所有属性."""
        cr = ChangeRecord(
            change_type="modify",
            entity_id="e-001",
            field_path="properties.formula",
            old_value="H2O",
            new_value="H₂O",
            changed_by="agent-1",
            reason="格式修正",
        )
        assert cr.change_type == "modify"
        assert cr.entity_id == "e-001"
        assert cr.field_path == "properties.formula"
        assert cr.old_value == "H2O"
        assert cr.new_value == "H₂O"
        assert cr.changed_by == "agent-1"
        assert cr.reason == "格式修正"

    def test_changed_at_默认值(self) -> None:
        """changed_at 默认应为当前时间."""
        cr = ChangeRecord(change_type="add", entity_id="e1", field_path="f")
        assert cr.changed_at > 0.0

    def test_changed_by_默认值(self) -> None:
        """changed_by 默认应为 'system'."""
        cr = ChangeRecord(change_type="add", entity_id="e1", field_path="f")
        assert cr.changed_by == "system"

    def test_change_type_validator_合法值(self) -> None:
        """change_type 为 add/modify/delete 应通过验证."""
        for ct in ["add", "modify", "delete"]:
            cr = ChangeRecord(change_type=ct, entity_id="e1", field_path="f")
            assert cr.change_type == ct

    def test_change_type_validator_非法值(self) -> None:
        """change_type 为非法值应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            ChangeRecord(change_type="invalid", entity_id="e1", field_path="f")

    def test_is_add(self) -> None:
        """change_type='add' 时 is_add 应为 True."""
        cr = ChangeRecord(change_type="add", entity_id="e1", field_path="f")
        assert cr.is_add() is True
        assert cr.is_modify() is False
        assert cr.is_delete() is False

    def test_is_modify(self) -> None:
        """change_type='modify' 时 is_modify 应为 True."""
        cr = ChangeRecord(change_type="modify", entity_id="e1", field_path="f")
        assert cr.is_modify() is True
        assert cr.is_add() is False

    def test_is_delete(self) -> None:
        """change_type='delete' 时 is_delete 应为 True."""
        cr = ChangeRecord(change_type="delete", entity_id="e1", field_path="f")
        assert cr.is_delete() is True
        assert cr.is_add() is False


# ============================================================
# 7. 证据记录测试
# ============================================================


class TestEvidenceRecord:
    """测试 EvidenceRecord 模型."""

    def test_evidence_id_默认格式(self) -> None:
        """默认 evidence_id 应以 'ev-' 开头."""
        ev = EvidenceRecord(entity_id="e1")
        assert ev.evidence_id.startswith("ev-")

    def test_基本属性(self) -> None:
        """EvidenceRecord 应正确设置所有属性."""
        ev = EvidenceRecord(
            entity_id="e-001",
            triple_id="t-001",
            source_type="paper",
            source_reference="doi:10.1234/test",
            source_content="水的沸点为100°C",
            confidence=0.95,
            verified_by="manual",
            verifier="expert-1",
        )
        assert ev.entity_id == "e-001"
        assert ev.triple_id == "t-001"
        assert ev.source_type == "paper"
        assert ev.source_reference == "doi:10.1234/test"
        assert ev.confidence == 0.95
        assert ev.verified_by == "manual"
        assert ev.verifier == "expert-1"

    def test_confidence_默认值(self) -> None:
        """confidence 默认应为 0.8."""
        ev = EvidenceRecord(entity_id="e1")
        assert ev.confidence == 0.8

    def test_confidence_范围约束(self) -> None:
        """confidence 超出 [0.0, 1.0] 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            EvidenceRecord(entity_id="e1", confidence=1.5)
        with pytest.raises(ValidationError):
            EvidenceRecord(entity_id="e1", confidence=-0.1)

    def test_verified_at_默认值(self) -> None:
        """verified_at 默认应为 0.0 (未验证)."""
        ev = EvidenceRecord(entity_id="e1")
        assert ev.verified_at == 0.0

    def test_is_verified_未验证(self) -> None:
        """verified_at=0.0 时 is_verified 应为 False."""
        ev = EvidenceRecord(entity_id="e1")
        assert ev.is_verified() is False

    def test_is_verified_已验证(self) -> None:
        """verified_at>0 时 is_verified 应为 True."""
        ev = EvidenceRecord(entity_id="e1", verified_at=time.time())
        assert ev.is_verified() is True

    def test_is_strong_强证据(self) -> None:
        """confidence>=0.8 时 is_strong 应为 True."""
        ev = EvidenceRecord(entity_id="e1", confidence=0.8)
        assert ev.is_strong() is True

    def test_is_strong_弱证据(self) -> None:
        """confidence<0.8 时 is_strong 应为 False."""
        ev = EvidenceRecord(entity_id="e1", confidence=0.79)
        assert ev.is_strong() is False

    def test_is_strong_边界值(self) -> None:
        """confidence=0.8 恰好为强证据."""
        ev = EvidenceRecord(entity_id="e1", confidence=0.8)
        assert ev.is_strong() is True

    def test_source_type_默认值(self) -> None:
        """source_type 默认应为 'document'."""
        ev = EvidenceRecord(entity_id="e1")
        assert ev.source_type == "document"


# ============================================================
# 8. 结构化查询测试
# ============================================================


class TestKnowledgeQuery:
    """测试 KnowledgeQuery 模型."""

    def test_query_id_默认格式(self) -> None:
        """默认 query_id 应以 'q-' 开头."""
        q = KnowledgeQuery()
        assert q.query_id.startswith("q-")

    def test_默认值(self) -> None:
        """KnowledgeQuery 默认值应正确."""
        q = KnowledgeQuery()
        assert q.domain == "general"
        assert q.conditions == []
        assert q.max_hops == 0
        assert q.limit == 100
        assert q.offset == 0
        assert q.sort_by == ""
        assert q.sort_desc is False
        assert q.include_graph is False
        assert q.timestamp_filter == 0.0

    def test_自定义值(self) -> None:
        """KnowledgeQuery 应支持自定义所有字段."""
        q = KnowledgeQuery(
            domain="chemistry",
            conditions=[QueryCondition(field="name", value="水")],
            max_hops=3,
            limit=50,
            offset=10,
            sort_by="created_at",
            sort_desc=True,
            include_graph=True,
            timestamp_filter=time.time(),
        )
        assert q.domain == "chemistry"
        assert len(q.conditions) == 1
        assert q.max_hops == 3
        assert q.limit == 50
        assert q.offset == 10
        assert q.sort_by == "created_at"
        assert q.sort_desc is True
        assert q.include_graph is True
        assert q.timestamp_filter > 0.0

    def test_has_conditions_有条件(self) -> None:
        """有条件时 has_conditions 应为 True."""
        q = KnowledgeQuery(
            conditions=[QueryCondition(field="name", value="test")]
        )
        assert q.has_conditions() is True

    def test_has_conditions_无条件(self) -> None:
        """无条件时 has_conditions 应为 False."""
        q = KnowledgeQuery()
        assert q.has_conditions() is False

    def test_has_traversal_有遍历(self) -> None:
        """max_hops>0 时 has_traversal 应为 True."""
        q = KnowledgeQuery(max_hops=2)
        assert q.has_traversal() is True

    def test_has_traversal_无遍历(self) -> None:
        """max_hops=0 时 has_traversal 应为 False."""
        q = KnowledgeQuery(max_hops=0)
        assert q.has_traversal() is False

    def test_has_temporal_filter_有时间过滤(self) -> None:
        """timestamp_filter>0 时 has_temporal_filter 应为 True."""
        q = KnowledgeQuery(timestamp_filter=1000.0)
        assert q.has_temporal_filter() is True

    def test_has_temporal_filter_无时间过滤(self) -> None:
        """timestamp_filter=0 时 has_temporal_filter 应为 False."""
        q = KnowledgeQuery(timestamp_filter=0.0)
        assert q.has_temporal_filter() is False


# ============================================================
# 9. 查询条件测试
# ============================================================


class TestQueryCondition:
    """测试 QueryCondition 的 matches() 方法覆盖所有操作符."""

    # --- EQ ---

    def test_eq_匹配(self) -> None:
        """EQ 操作符: target == value 时匹配."""
        cond = QueryCondition(field="name", operator=QueryOperator.EQ, value="水")
        assert cond.matches("水") is True

    def test_eq_不匹配(self) -> None:
        """EQ 操作符: target != value 时不匹配."""
        cond = QueryCondition(field="name", operator=QueryOperator.EQ, value="水")
        assert cond.matches("乙醇") is False

    # --- NE ---

    def test_ne_匹配(self) -> None:
        """NE 操作符: target != value 时匹配."""
        cond = QueryCondition(field="name", operator=QueryOperator.NE, value="水")
        assert cond.matches("乙醇") is True

    def test_ne_不匹配(self) -> None:
        """NE 操作符: target == value 时不匹配."""
        cond = QueryCondition(field="name", operator=QueryOperator.NE, value="水")
        assert cond.matches("水") is False

    # --- GT ---

    def test_gt_匹配(self) -> None:
        """GT 操作符: target > value 时匹配."""
        cond = QueryCondition(field="weight", operator=QueryOperator.GT, value=50)
        assert cond.matches(100) is True

    def test_gt_不匹配(self) -> None:
        """GT 操作符: target <= value 时不匹配."""
        cond = QueryCondition(field="weight", operator=QueryOperator.GT, value=50)
        assert cond.matches(50) is False

    def test_gt_类型错误(self) -> None:
        """GT 操作符: 非数值类型应返回 False."""
        cond = QueryCondition(field="weight", operator=QueryOperator.GT, value=50)
        assert cond.matches("not_a_number") is False

    # --- GTE ---

    def test_gte_匹配(self) -> None:
        """GTE 操作符: target >= value 时匹配."""
        cond = QueryCondition(field="weight", operator=QueryOperator.GTE, value=50)
        assert cond.matches(50) is True

    def test_gte_不匹配(self) -> None:
        """GTE 操作符: target < value 时不匹配."""
        cond = QueryCondition(field="weight", operator=QueryOperator.GTE, value=50)
        assert cond.matches(49) is False

    # --- LT ---

    def test_lt_匹配(self) -> None:
        """LT 操作符: target < value 时匹配."""
        cond = QueryCondition(field="weight", operator=QueryOperator.LT, value=50)
        assert cond.matches(49) is True

    def test_lt_不匹配(self) -> None:
        """LT 操作符: target >= value 时不匹配."""
        cond = QueryCondition(field="weight", operator=QueryOperator.LT, value=50)
        assert cond.matches(50) is False

    # --- LTE ---

    def test_lte_匹配(self) -> None:
        """LTE 操作符: target <= value 时匹配."""
        cond = QueryCondition(field="weight", operator=QueryOperator.LTE, value=50)
        assert cond.matches(50) is True

    def test_lte_不匹配(self) -> None:
        """LTE 操作符: target > value 时不匹配."""
        cond = QueryCondition(field="weight", operator=QueryOperator.LTE, value=50)
        assert cond.matches(51) is False

    # --- CONTAINS ---

    def test_contains_匹配(self) -> None:
        """CONTAINS 操作符: value 是 target 的子串时匹配."""
        cond = QueryCondition(field="desc", operator=QueryOperator.CONTAINS, value="化学")
        assert cond.matches("这是一本化学教材") is True

    def test_contains_不匹配(self) -> None:
        """CONTAINS 操作符: value 不是 target 的子串时不匹配."""
        cond = QueryCondition(field="desc", operator=QueryOperator.CONTAINS, value="物理")
        assert cond.matches("这是一本化学教材") is False

    # --- STARTS_WITH ---

    def test_starts_with_匹配(self) -> None:
        """STARTS_WITH 操作符: target 以 value 开头时匹配."""
        cond = QueryCondition(field="name", operator=QueryOperator.STARTS_WITH, value="化学")
        assert cond.matches("化学化合物") is True

    def test_starts_with_不匹配(self) -> None:
        """STARTS_WITH 操作符: target 不以 value 开头时不匹配."""
        cond = QueryCondition(field="name", operator=QueryOperator.STARTS_WITH, value="物理")
        assert cond.matches("化学化合物") is False

    # --- ENDS_WITH ---

    def test_ends_with_匹配(self) -> None:
        """ENDS_WITH 操作符: target 以 value 结尾时匹配."""
        cond = QueryCondition(field="name", operator=QueryOperator.ENDS_WITH, value="化合物")
        assert cond.matches("化学化合物") is True

    def test_ends_with_不匹配(self) -> None:
        """ENDS_WITH 操作符: target 不以 value 结尾时不匹配."""
        cond = QueryCondition(field="name", operator=QueryOperator.ENDS_WITH, value="元素")
        assert cond.matches("化学化合物") is False

    # --- IN ---

    def test_in_列表匹配(self) -> None:
        """IN 操作符: target 在列表中时匹配."""
        cond = QueryCondition(
            field="type", operator=QueryOperator.IN, value=["solid", "liquid", "gas"]
        )
        assert cond.matches("liquid") is True

    def test_in_列表不匹配(self) -> None:
        """IN 操作符: target 不在列表中时不匹配."""
        cond = QueryCondition(
            field="type", operator=QueryOperator.IN, value=["solid", "liquid", "gas"]
        )
        assert cond.matches("plasma") is False

    def test_in_单值(self) -> None:
        """IN 操作符: value 为单值时应将其作为单元素列表处理."""
        cond = QueryCondition(field="type", operator=QueryOperator.IN, value="solid")
        assert cond.matches("solid") is True
        assert cond.matches("liquid") is False

    # --- REGEX ---

    def test_regex_匹配(self) -> None:
        """REGEX 操作符: target 匹配正则模式时匹配."""
        cond = QueryCondition(
            field="email", operator=QueryOperator.REGEX, value=r"^[^@]+@[^@]+$"
        )
        assert cond.matches("user@example.com") is True

    def test_regex_不匹配(self) -> None:
        """REGEX 操作符: target 不匹配正则模式时不匹配."""
        cond = QueryCondition(
            field="email", operator=QueryOperator.REGEX, value=r"^[^@]+@[^@]+$"
        )
        assert cond.matches("invalid_email") is False

    def test_regex_无效正则(self) -> None:
        """REGEX 操作符: 无效正则应返回 False (不抛异常)."""
        cond = QueryCondition(
            field="text", operator=QueryOperator.REGEX, value=r"[invalid"
        )
        assert cond.matches("test") is False

    # --- negate ---

    def test_negate_eq_取反匹配(self) -> None:
        """negate=True 时 EQ 结果应取反."""
        cond = QueryCondition(
            field="name", operator=QueryOperator.EQ, value="水", negate=True
        )
        assert cond.matches("水") is False

    def test_negate_eq_取反不匹配(self) -> None:
        """negate=True 时 EQ 不匹配的结果取反后为 True."""
        cond = QueryCondition(
            field="name", operator=QueryOperator.EQ, value="水", negate=True
        )
        assert cond.matches("乙醇") is True

    def test_negate_gt_取反(self) -> None:
        """negate=True 时 GT 结果应取反."""
        cond = QueryCondition(
            field="weight", operator=QueryOperator.GT, value=50, negate=True
        )
        assert cond.matches(100) is False
        assert cond.matches(30) is True

    def test_negate_contains_取反(self) -> None:
        """negate=True 时 CONTAINS 结果应取反."""
        cond = QueryCondition(
            field="desc", operator=QueryOperator.CONTAINS, value="化学", negate=True
        )
        assert cond.matches("物理教材") is True
        assert cond.matches("化学教材") is False

    def test_negate_in_取反(self) -> None:
        """negate=True 时 IN 结果应取反."""
        cond = QueryCondition(
            field="type",
            operator=QueryOperator.IN,
            value=["solid", "liquid"],
            negate=True,
        )
        assert cond.matches("gas") is True
        assert cond.matches("solid") is False


# ============================================================
# 10. 子图提取配置测试
# ============================================================


class TestSubgraphConfig:
    """测试 SubgraphConfig 模型."""

    def test_entity_focus_必填(self) -> None:
        """entity_focus 为必填字段."""
        with pytest.raises(ValidationError):
            SubgraphConfig()

    def test_默认值(self) -> None:
        """SubgraphConfig 默认值应正确."""
        cfg = SubgraphConfig(entity_focus="e-001")
        assert cfg.max_depth == 2
        assert cfg.max_entities == 50
        assert cfg.min_confidence == 0.5
        assert cfg.min_quality == 0.0
        assert cfg.include_deprecated is False
        assert cfg.traverse_strategy == "bfs"

    def test_max_depth_范围约束(self) -> None:
        """max_depth 超出 [1, 10] 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            SubgraphConfig(entity_focus="e1", max_depth=0)
        with pytest.raises(ValidationError):
            SubgraphConfig(entity_focus="e1", max_depth=11)

    def test_max_entities_范围约束(self) -> None:
        """max_entities 超出 [1, 500] 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            SubgraphConfig(entity_focus="e1", max_entities=0)
        with pytest.raises(ValidationError):
            SubgraphConfig(entity_focus="e1", max_entities=501)

    def test_min_confidence_范围约束(self) -> None:
        """min_confidence 超出 [0.0, 1.0] 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            SubgraphConfig(entity_focus="e1", min_confidence=-0.1)
        with pytest.raises(ValidationError):
            SubgraphConfig(entity_focus="e1", min_confidence=1.1)

    def test_min_quality_范围约束(self) -> None:
        """min_quality 超出 [0.0, 1.0] 应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            SubgraphConfig(entity_focus="e1", min_quality=-0.1)
        with pytest.raises(ValidationError):
            SubgraphConfig(entity_focus="e1", min_quality=1.1)

    def test_traverse_strategy_合法值(self) -> None:
        """traverse_strategy 为合法值应通过验证."""
        for strategy in ["bfs", "shortest_path", "confidence_weighted"]:
            cfg = SubgraphConfig(entity_focus="e1", traverse_strategy=strategy)
            assert cfg.traverse_strategy == strategy

    def test_traverse_strategy_非法值(self) -> None:
        """traverse_strategy 为非法值应抛出 ValidationError."""
        with pytest.raises(ValidationError):
            SubgraphConfig(entity_focus="e1", traverse_strategy="invalid")

    def test_include_deprecated_设为True(self) -> None:
        """include_deprecated 应可设为 True."""
        cfg = SubgraphConfig(entity_focus="e1", include_deprecated=True)
        assert cfg.include_deprecated is True




class TestSHACLValidation:
    """测试 SHACL 风格属性验证."""

    # --- OntologyProperty.validate_value() ---

    def test_validate_value_枚举值合法(self) -> None:
        """枚举值合法时 validate_value 应返回空列表."""
        prop = OntologyProperty(name="color", enum_values=["red", "green", "blue"])
        assert prop.validate_value("red") == []

    def test_validate_value_枚举值非法(self) -> None:
        """枚举值非法时 validate_value 应返回违规信息."""
        prop = OntologyProperty(name="color", enum_values=["red", "green", "blue"])
        violations = prop.validate_value("purple")
        assert len(violations) == 1
        assert "purple" in violations[0]

    def test_validate_value_数据类型STRING合法(self) -> None:
        """STRING 类型检查通过."""
        prop = OntologyProperty(name="name", data_type=PropertyDataType.STRING)
        assert prop.validate_value("hello") == []

    def test_validate_value_数据类型STRING非法(self) -> None:
        """STRING 类型检查: 传入整数应违规."""
        prop = OntologyProperty(name="name", data_type=PropertyDataType.STRING)
        violations = prop.validate_value(123)
        assert len(violations) == 1

    def test_validate_value_数据类型INTEGER合法(self) -> None:
        """INTEGER 类型检查通过."""
        prop = OntologyProperty(name="age", data_type=PropertyDataType.INTEGER)
        assert prop.validate_value(42) == []

    def test_validate_value_数据类型INTEGER_布尔应违规(self) -> None:
        """INTEGER 类型检查: 布尔值应违规."""
        prop = OntologyProperty(name="age", data_type=PropertyDataType.INTEGER)
        violations = prop.validate_value(True)
        assert len(violations) == 1

    def test_validate_value_数据类型INTEGER_浮点应违规(self) -> None:
        """INTEGER 类型检查: 浮点数应违规."""
        prop = OntologyProperty(name="age", data_type=PropertyDataType.INTEGER)
        violations = prop.validate_value(3.14)
        assert len(violations) == 1

    def test_validate_value_数据类型FLOAT合法(self) -> None:
        """FLOAT 类型检查: 整数和浮点都通过."""
        prop = OntologyProperty(name="score", data_type=PropertyDataType.FLOAT)
        assert prop.validate_value(3.14) == []
        assert prop.validate_value(42) == []

    def test_validate_value_数据类型BOOLEAN合法(self) -> None:
        """BOOLEAN 类型检查通过."""
        prop = OntologyProperty(name="active", data_type=PropertyDataType.BOOLEAN)
        assert prop.validate_value(True) == []
        assert prop.validate_value(False) == []

    def test_validate_value_数据类型BOOLEAN非法(self) -> None:
        """BOOLEAN 类型检查: 字符串应违规."""
        prop = OntologyProperty(name="active", data_type=PropertyDataType.BOOLEAN)
        violations = prop.validate_value("yes")
        assert len(violations) == 1

    def test_validate_value_数据类型DATETIME合法(self) -> None:
        """DATETIME 类型检查: 时间戳和字符串都通过."""
        prop = OntologyProperty(name="ts", data_type=PropertyDataType.DATETIME)
        assert prop.validate_value(1234567890.0) == []
        assert prop.validate_value("2024-01-01") == []

    def test_validate_value_min_value违规(self) -> None:
        """min_value 检查: 值小于最小值应违规."""
        prop = OntologyProperty(name="age", min_value=0)
        violations = prop.validate_value(-1)
        assert len(violations) == 1

    def test_validate_value_max_value违规(self) -> None:
        """max_value 检查: 值大于最大值应违规."""
        prop = OntologyProperty(name="age", max_value=150)
        violations = prop.validate_value(200)
        assert len(violations) == 1

    def test_validate_value_min_length违规(self) -> None:
        """min_length 检查: 字符串太短应违规."""
        prop = OntologyProperty(name="name", min_length=3)
        violations = prop.validate_value("ab")
        assert len(violations) == 1

    def test_validate_value_max_length违规(self) -> None:
        """max_length 检查: 字符串太长应违规."""
        prop = OntologyProperty(name="name", max_length=5)
        violations = prop.validate_value("abcdefgh")
        assert len(violations) == 1

    def test_validate_value_pattern合法(self) -> None:
        """pattern 检查: 匹配模式应通过."""
        prop = OntologyProperty(name="email", pattern=r"^[^@]+@[^@]+$")
        assert prop.validate_value("user@example.com") == []

    def test_validate_value_pattern违规(self) -> None:
        """pattern 检查: 不匹配模式应违规."""
        prop = OntologyProperty(name="email", pattern=r"^[^@]+@[^@]+$")
        violations = prop.validate_value("invalid")
        assert len(violations) == 1

    # --- DomainOntology 验证方法 ---

    def test_validate_data_types_通过(self) -> None:
        """validate_data_types: 全部合法时应返回空列表."""
        onto = _build_shacl_ontology()
        props = {"color": "red", "age": 25, "label": "test_label"}
        assert onto.validate_data_types(EntityType.CONCEPT, props) == []

    def test_validate_data_types_枚举违规(self) -> None:
        """validate_data_types: 枚举值非法时应返回违规."""
        onto = _build_shacl_ontology()
        violations = onto.validate_data_types(EntityType.CONCEPT, {"color": "purple"})
        assert len(violations) > 0

    def test_validate_data_types_类型违规(self) -> None:
        """validate_data_types: 数据类型不匹配时应返回违规."""
        onto = _build_shacl_ontology()
        violations = onto.validate_data_types(EntityType.CONCEPT, {"age": "not_a_number"})
        assert len(violations) > 0

    def test_validate_cardinality_min_count违规(self) -> None:
        """validate_cardinality: 值数量少于 min_count 应违规."""
        onto = _build_shacl_ontology()
        # tags_list has min_count=1, not providing it should violate
        violations = onto.validate_cardinality(EntityType.CONCEPT, {})
        assert any("tags_list" in v for v in violations)

    def test_validate_cardinality_max_count违规(self) -> None:
        """validate_cardinality: 值数量超过 max_count 应违规."""
        onto = _build_shacl_ontology()
        violations = onto.validate_cardinality(
            EntityType.CONCEPT, {"tags_list": ["a", "b", "c", "d", "e", "f"]}
        )
        assert any("tags_list" in v for v in violations)

    def test_validate_cardinality_通过(self) -> None:
        """validate_cardinality: 值数量在范围内应通过."""
        onto = _build_shacl_ontology()
        violations = onto.validate_cardinality(EntityType.CONCEPT, {"tags_list": ["a", "b"]})
        assert violations == []

    def test_validate_full_通过(self) -> None:
        """validate_full: 全部合法时应返回空列表."""
        onto = _build_shacl_ontology()
        props = {"color": "red", "age": 25, "label": "valid", "tags_list": ["t1"]}
        assert onto.validate_full(EntityType.CONCEPT, props) == []

    def test_validate_full_含违规(self) -> None:
        """validate_full: 有违规时应返回违规列表."""
        onto = _build_shacl_ontology()
        props = {"color": "purple", "age": 200}
        violations = onto.validate_full(EntityType.CONCEPT, props)
        assert len(violations) > 0

    def test_validate_all_含公理验证(self) -> None:
        """validate_all: 应包含公理验证."""
        onto = _build_shacl_ontology()
        # single_val is functional, providing list > 1 should violate
        props = {"single_val": ["a", "b"]}
        violations = onto.validate_all(EntityType.CONCEPT, props)
        assert any("single_val" in v for v in violations)

    def test_validate_all_通过(self) -> None:
        """validate_all: 全部合法时应返回空列表."""
        onto = _build_shacl_ontology()
        props = {"color": "red", "age": 25, "tags_list": ["t1"]}
        assert onto.validate_all(EntityType.CONCEPT, props) == []


# ============================================================
# 18. 类层级管理测试
# ============================================================


class TestClassHierarchy:
    """测试类继承层级管理."""

    def test_get_all_properties_无父类(self) -> None:
        """无父类时 get_all_properties 应返回自身属性."""
        onto = _build_hierarchy_ontology()
        cls = onto.get_class(EntityType.CONCEPT)
        assert cls is not None
        props = cls.get_all_properties(onto)
        assert len(props) == 2  # base_prop, shared_prop

    def test_get_all_properties_含继承(self) -> None:
        """有父类时 get_all_properties 应包含父类属性."""
        onto = _build_hierarchy_ontology()
        cls = onto.get_class(EntityType.CHEMICAL_COMPOUND)
        assert cls is not None
        props = cls.get_all_properties(onto)
        prop_names = {p.name for p in props}
        # 应包含父类 base_prop, shared_prop + 自身 child_prop
        assert "base_prop" in prop_names
        assert "shared_prop" in prop_names
        assert "child_prop" in prop_names

    def test_get_all_properties_子类覆盖父类(self) -> None:
        """子类应覆盖父类同名属性."""
        onto = _build_hierarchy_ontology()
        cls = onto.get_class(EntityType.CHEMICAL_COMPOUND)
        assert cls is not None
        props = cls.get_all_properties(onto)
        shared = next(p for p in props if p.name == "shared_prop")
        # 子类的 shared_prop display_name 为 "覆盖属性"
        assert shared.display_name == "覆盖属性"

    def test_get_all_allowed_relations_含继承(self) -> None:
        """get_all_allowed_relations 应包含父类关系."""
        onto = _build_hierarchy_ontology()
        cls = onto.get_class(EntityType.CHEMICAL_COMPOUND)
        assert cls is not None
        rels = cls.get_all_allowed_relations(onto)
        # 父类: RELATED_TO, PART_OF + 子类: CITES
        assert RelationType.RELATED_TO in rels
        assert RelationType.PART_OF in rels
        assert RelationType.CITES in rels

    def test_get_required_properties_含继承(self) -> None:
        """get_required_properties 应包含父类的必需属性."""
        onto = _build_hierarchy_ontology()
        cls = onto.get_class(EntityType.CHEMICAL_COMPOUND)
        assert cls is not None
        required = cls.get_required_properties(onto)
        req_names = {p.name for p in required}
        # base_prop is required in parent
        assert "base_prop" in req_names

    def test_get_class_hierarchy_两层(self) -> None:
        """get_class_hierarchy 应返回从根到当前的层级."""
        onto = _build_hierarchy_ontology()
        hierarchy = onto.get_class_hierarchy(EntityType.CHEMICAL_COMPOUND)
        assert hierarchy == [EntityType.CONCEPT, EntityType.CHEMICAL_COMPOUND]

    def test_get_class_hierarchy_三层(self) -> None:
        """get_class_hierarchy 应返回三层完整层级."""
        onto = _build_hierarchy_ontology()
        hierarchy = onto.get_class_hierarchy(EntityType.MATERIAL)
        assert hierarchy == [EntityType.CONCEPT, EntityType.CHEMICAL_COMPOUND, EntityType.MATERIAL]

    def test_get_class_hierarchy_根类(self) -> None:
        """根类的 hierarchy 应只包含自身."""
        onto = _build_hierarchy_ontology()
        hierarchy = onto.get_class_hierarchy(EntityType.CONCEPT)
        assert hierarchy == [EntityType.CONCEPT]

    def test_get_class_hierarchy_不存在(self) -> None:
        """不存在的类型应返回空列表."""
        onto = _build_hierarchy_ontology()
        hierarchy = onto.get_class_hierarchy(EntityType.PERSON)
        assert hierarchy == []

    def test_get_subclasses_直接子类(self) -> None:
        """get_subclasses 应返回直接子类."""
        onto = _build_hierarchy_ontology()
        subs = onto.get_subclasses(EntityType.CONCEPT)
        assert EntityType.CHEMICAL_COMPOUND in subs
        assert EntityType.MATERIAL not in subs  # MATERIAL 是孙类

    def test_get_all_subclasses_递归(self) -> None:
        """get_all_subclasses 应返回所有子类 (递归)."""
        onto = _build_hierarchy_ontology()
        all_subs = onto.get_all_subclasses(EntityType.CONCEPT)
        assert EntityType.CHEMICAL_COMPOUND in all_subs
        assert EntityType.MATERIAL in all_subs

    def test_is_subclass_of_是子类(self) -> None:
        """is_subclass_of: CHEMICAL_COMPOUND 是 CONCEPT 的子类."""
        onto = _build_hierarchy_ontology()
        assert onto.is_subclass_of(EntityType.CHEMICAL_COMPOUND, EntityType.CONCEPT) is True

    def test_is_subclass_of_间接子类(self) -> None:
        """is_subclass_of: MATERIAL 是 CONCEPT 的间接子类."""
        onto = _build_hierarchy_ontology()
        assert onto.is_subclass_of(EntityType.MATERIAL, EntityType.CONCEPT) is True

    def test_is_subclass_of_不是子类(self) -> None:
        """is_subclass_of: CONCEPT 不是 CHEMICAL_COMPOUND 的子类."""
        onto = _build_hierarchy_ontology()
        assert onto.is_subclass_of(EntityType.CONCEPT, EntityType.CHEMICAL_COMPOUND) is False

    def test_is_subclass_of_自身(self) -> None:
        """is_subclass_of: 类型是自身的子类 (在层级中)."""
        onto = _build_hierarchy_ontology()
        assert onto.is_subclass_of(EntityType.CHEMICAL_COMPOUND, EntityType.CHEMICAL_COMPOUND) is True


# ============================================================
# 19. 本体推理引擎测试
# ============================================================


class TestInferenceEngine:
    """测试本体推理引擎的各类推理能力."""

    # --- 传递闭包 ---

    def test_transitive_closure_基本推理(self) -> None:
        """传递闭包: A→B, B→C ⟹ A→C."""
        onto = _build_rules_ontology()
        triples = [("A", "part_of", "B"), ("B", "part_of", "C")]
        inferred = onto.infer_transitive_closure(triples, "part_of")
        assert ("A", "part_of", "C") in inferred

    def test_transitive_closure_多级(self) -> None:
        """传递闭包: A→B→C→D ⟹ A→C, A→D, B→D."""
        onto = _build_rules_ontology()
        triples = [
            ("A", "part_of", "B"),
            ("B", "part_of", "C"),
            ("C", "part_of", "D"),
        ]
        inferred = onto.infer_transitive_closure(triples, "part_of")
        inferred_set = set(inferred)
        assert ("A", "part_of", "C") in inferred_set
        assert ("A", "part_of", "D") in inferred_set
        assert ("B", "part_of", "D") in inferred_set

    def test_transitive_closure_非传递关系返回空(self) -> None:
        """非传递关系应返回空列表."""
        onto = _build_rules_ontology()
        triples = [("A", "plain_rel", "B"), ("B", "plain_rel", "C")]
        inferred = onto.infer_transitive_closure(triples, "plain_rel")
        assert inferred == []

    # --- 逆关系 ---

    def test_inverse_relations_基本推理(self) -> None:
        """逆关系: A→B(parent_of), inverse=child_of ⟹ B→A(child_of)."""
        onto = _build_inverse_ontology()
        triples = [("A", "parent_of", "B")]
        inferred = onto.infer_inverse_relations(triples, "parent_of")
        assert ("B", "child_of", "A") in inferred

    def test_inverse_relations_多条(self) -> None:
        """逆关系: 多条三元组应全部推理."""
        onto = _build_inverse_ontology()
        triples = [("A", "parent_of", "B"), ("C", "parent_of", "D")]
        inferred = onto.infer_inverse_relations(triples, "parent_of")
        assert ("B", "child_of", "A") in inferred
        assert ("D", "child_of", "C") in inferred

    def test_inverse_relations_无逆关系返回空(self) -> None:
        """无逆关系定义的关系应返回空列表."""
        onto = _build_inverse_ontology()
        triples = [("A", "non_inverse", "B")]
        inferred = onto.infer_inverse_relations(triples, "non_inverse")
        assert inferred == []

    # --- 对称闭包 ---

    def test_symmetric_closure_基本推理(self) -> None:
        """对称闭包: A→B ⟹ B→A."""
        onto = _build_rules_ontology()
        triples = [("A", "related_to", "B")]
        inferred = onto.infer_symmetric_closure(triples, "related_to")
        assert ("B", "related_to", "A") in inferred

    def test_symmetric_closure_不重复已有(self) -> None:
        """对称闭包: 已存在的反向关系不应重复推理."""
        onto = _build_rules_ontology()
        triples = [("A", "related_to", "B"), ("B", "related_to", "A")]
        inferred = onto.infer_symmetric_closure(triples, "related_to")
        # A→B 的反向 B→A 已存在, B→A 的反向 A→B 已存在
        assert ("B", "related_to", "A") not in inferred
        assert ("A", "related_to", "B") not in inferred

    def test_symmetric_closure_非对称关系返回空(self) -> None:
        """非对称关系应返回空列表."""
        onto = _build_rules_ontology()
        triples = [("A", "part_of", "B")]
        inferred = onto.infer_symmetric_closure(triples, "part_of")
        assert inferred == []

    # --- 属性链 ---

    def test_property_chain_基本推理(self) -> None:
        """属性链: A→B(R1), B→C(R2) ⟹ A→C(R1_R2)."""
        onto = _build_rules_ontology()
        triples = [("Alice", "part_of", "NYC"), ("NYC", "related_to", "USA")]
        chain = ["part_of", "related_to"]
        inferred = onto.infer_property_chain(triples, chain)
        assert ("Alice", "part_of_related_to", "USA") in inferred

    def test_property_chain_单元素链返回空(self) -> None:
        """属性链长度 < 2 应返回空列表."""
        onto = _build_rules_ontology()
        triples = [("A", "part_of", "B")]
        inferred = onto.infer_property_chain(triples, ["part_of"])
        assert inferred == []

    def test_property_chain_不重复已有(self) -> None:
        """属性链: 已存在的合成关系不应重复推理."""
        onto = _build_rules_ontology()
        triples = [
            ("A", "r1", "B"),
            ("B", "r2", "C"),
            ("A", "r1_r2", "C"),
        ]
        inferred = onto.infer_property_chain(triples, ["r1", "r2"])
        assert ("A", "r1_r2", "C") not in inferred

    # --- 子类继承 ---

    def test_subclass_inheritance_基本推理(self) -> None:
        """子类继承: x 是 MATERIAL 的实例, MATERIAL→CHEMICAL_COMPOUND→CONCEPT ⟹ x 也是 CONCEPT 和 CHEMICAL_COMPOUND 的实例."""
        onto = _build_hierarchy_ontology()
        instances = [("e1", EntityType.MATERIAL)]
        inferred = onto.infer_subclass_inheritance(instances)
        inferred_set = set(inferred)
        assert ("e1", EntityType.CONCEPT) in inferred_set
        assert ("e1", EntityType.CHEMICAL_COMPOUND) in inferred_set

    def test_subclass_inheritance_根类无推理(self) -> None:
        """子类继承: 根类实例无父类, 不推理新实例关系."""
        onto = _build_hierarchy_ontology()
        instances = [("e1", EntityType.CONCEPT)]
        inferred = onto.infer_subclass_inheritance(instances)
        assert inferred == []

    # --- apply_inference_rules ---

    def test_apply_inference_rules_传递和对称(self) -> None:
        """apply_inference_rules 应同时应用传递闭包和对称闭包规则."""
        onto = _build_rules_ontology()
        triples = [
            ("A", "part_of", "B"),
            ("B", "part_of", "C"),
            ("X", "related_to", "Y"),
        ]
        inferred = onto.apply_inference_rules(triples)
        inferred_set = set(inferred)
        # 传递闭包
        assert ("A", "part_of", "C") in inferred_set
        # 对称闭包
        assert ("Y", "related_to", "X") in inferred_set

    def test_apply_inference_rules_不重复(self) -> None:
        """apply_inference_rules 不应返回已存在的三元组."""
        onto = _build_rules_ontology()
        triples = [("A", "part_of", "B"), ("B", "part_of", "C")]
        inferred = onto.apply_inference_rules(triples)
        for t in inferred:
            assert t not in set(triples)

    def test_apply_inference_rules_空规则(self) -> None:
        """无推理规则的本体应返回空列表."""
        onto = DomainOntology(
            ontology_id="onto-empty",
            domain="empty_test",
            classes=[OntologyClass(class_id="c1", entity_type=EntityType.CONCEPT)],
        )
        triples = [("A", "related_to", "B")]
        assert onto.apply_inference_rules(triples) == []

    def test_apply_inference_rules_禁用规则不应用(self) -> None:
        """禁用的推理规则不应被应用."""
        onto = DomainOntology(
            ontology_id="onto-disabled",
            domain="disabled_test",
            classes=[OntologyClass(class_id="c1", entity_type=EntityType.CONCEPT)],
            relations=[OntologyRelation(name="related_to", symmetric=True)],
            inference_rules=[
                OntologyRule(
                    rule_type=InferenceRuleType.SYMMETRIC_CLOSURE,
                    applies_to_relation="related_to",
                    enabled=False,
                ),
            ],
        )
        triples = [("A", "related_to", "B")]
        assert onto.apply_inference_rules(triples) == []


# ============================================================
# 20. 本体公理测试
# ============================================================


class TestOntologyAxioms:
    """测试本体公理验证."""

    def test_ontology_axiom_实例化(self) -> None:
        """OntologyAxiom 应正确实例化."""
        ax = OntologyAxiom(
            axiom_type="disjoint",
            subject="concept",
            object="material",
            description="概念和材料不相交",
        )
        assert ax.axiom_type == "disjoint"
        assert ax.subject == "concept"
        assert ax.object == "material"
        assert ax.description == "概念和材料不相交"

    def test_ontology_axiom_默认id(self) -> None:
        """OntologyAxiom 应有默认 axiom_id."""
        ax = OntologyAxiom(axiom_type="functional", subject="prop1")
        assert ax.axiom_id.startswith("ax-")

    def test_validate_disjoint_有违规(self) -> None:
        """validate_disjoint: 两个不相交类型同时出现应违规."""
        onto = _build_shacl_ontology()
        violations = onto.validate_disjoint([EntityType.CONCEPT, EntityType.MATERIAL])
        assert len(violations) > 0

    def test_validate_disjoint_无违规(self) -> None:
        """validate_disjoint: 无不相交类型对时应通过."""
        onto = _build_shacl_ontology()
        violations = onto.validate_disjoint([EntityType.CONCEPT, EntityType.PERSON])
        assert violations == []

    def test_validate_axioms_函数性属性违规(self) -> None:
        """validate_axioms: 函数性属性提供多个值应违规."""
        onto = _build_shacl_ontology()
        violations = onto.validate_axioms(
            EntityType.CONCEPT, {"single_val": ["a", "b", "c"]}
        )
        assert any("single_val" in v for v in violations)

    def test_validate_axioms_函数性属性通过(self) -> None:
        """validate_axioms: 函数性属性提供单值应通过."""
        onto = _build_shacl_ontology()
        violations = onto.validate_axioms(EntityType.CONCEPT, {"single_val": "a"})
        assert violations == []

    def test_validate_all_含公理(self) -> None:
        """validate_all: 应同时包含 validate_full 和 validate_axioms 的结果."""
        onto = _build_shacl_ontology()
        props = {"color": "purple", "single_val": ["a", "b"]}
        violations = onto.validate_all(EntityType.CONCEPT, props)
        # 应包含枚举违规和函数性违规
        assert any("purple" in v for v in violations)
        assert any("single_val" in v for v in violations)

    def test_validate_all_全部通过(self) -> None:
        """validate_all: 全部合法时应返回空列表."""
        onto = _build_shacl_ontology()
        props = {"color": "red", "age": 25, "tags_list": ["t1"]}
        assert onto.validate_all(EntityType.CONCEPT, props) == []


# ============================================================
# 21. 跨本体映射测试
# ============================================================


class TestOntologyMapping:
    """测试跨本体映射."""

    def test_ontology_mapping_实例化(self) -> None:
        """OntologyMapping 应正确实例化."""
        m = OntologyMapping(
            source_domain="chemistry",
            target_domain="materials",
            mapping_type="equivalent",
            source_entity_type=EntityType.CHEMICAL_COMPOUND,
            target_entity_type=EntityType.MATERIAL,
            confidence=0.9,
        )
        assert m.source_domain == "chemistry"
        assert m.target_domain == "materials"
        assert m.mapping_type == "equivalent"
        assert m.source_entity_type == EntityType.CHEMICAL_COMPOUND
        assert m.target_entity_type == EntityType.MATERIAL
        assert m.confidence == 0.9

    def test_ontology_mapping_默认id(self) -> None:
        """OntologyMapping 应有默认 mapping_id."""
        m = OntologyMapping(
            source_domain="a",
            target_domain="b",
            source_entity_type=EntityType.CONCEPT,
            target_entity_type=EntityType.MATERIAL,
        )
        assert m.mapping_id.startswith("mp-")

    def test_find_mappings_无过滤(self) -> None:
        """find_mappings: 无过滤条件应返回所有映射."""
        onto = _build_shacl_ontology()
        mappings = onto.find_mappings()
        assert len(mappings) == 2

    def test_find_mappings_按实体类型过滤(self) -> None:
        """find_mappings: 按 source_entity_type 过滤."""
        onto = _build_shacl_ontology()
        mappings = onto.find_mappings(entity_type=EntityType.CONCEPT)
        assert len(mappings) == 2  # 两个映射的 source 都是 CONCEPT

    def test_find_mappings_按目标领域过滤(self) -> None:
        """find_mappings: 按 target_domain 过滤."""
        onto = _build_shacl_ontology()
        mappings = onto.find_mappings(target_domain="chemistry")
        assert len(mappings) == 1
        assert mappings[0].target_domain == "chemistry"

    def test_find_mappings_组合过滤(self) -> None:
        """find_mappings: 同时按实体类型和目标领域过滤."""
        onto = _build_shacl_ontology()
        mappings = onto.find_mappings(
            entity_type=EntityType.CONCEPT, target_domain="materials"
        )
        assert len(mappings) == 1
        assert mappings[0].target_domain == "materials"

    def test_get_equivalent_type_存在(self) -> None:
        """get_equivalent_type: 存在等价映射时应返回目标类型."""
        onto = _build_shacl_ontology()
        eq = onto.get_equivalent_type(EntityType.CONCEPT, "chemistry")
        assert eq == EntityType.CHEMICAL_COMPOUND

    def test_get_equivalent_type_不存在(self) -> None:
        """get_equivalent_type: 无等价映射时应返回 None."""
        onto = _build_shacl_ontology()
        # materials 映射是 "related" 而非 "equivalent"
        eq = onto.get_equivalent_type(EntityType.CONCEPT, "materials")
        assert eq is None

    def test_get_equivalent_type_领域不存在(self) -> None:
        """get_equivalent_type: 目标领域不存在映射时应返回 None."""
        onto = _build_shacl_ontology()
        eq = onto.get_equivalent_type(EntityType.CONCEPT, "nonexistent")
        assert eq is None


# ============================================================
# 22. 增强注册中心测试
# ============================================================


class TestOntologyRegistryEnhanced:
    """测试 OntologyRegistry 的增强方法."""

    def setup_method(self) -> None:
        """每个测试方法前创建注册中心并注册测试本体."""
        self.registry = OntologyRegistry()
        self.registry.register(_build_hierarchy_ontology())
        self.registry.register(_build_inverse_ontology())
        self.registry.register(_build_rules_ontology())
        self.registry.register(_build_shacl_ontology())

    # --- SHACL 验证封装 ---

    def test_validate_data_types_通过(self) -> None:
        """registry.validate_data_types: 合法属性应通过."""
        violations = self.registry.validate_data_types(
            "shacl_test", EntityType.CONCEPT, {"color": "red", "age": 25}
        )
        assert violations == []

    def test_validate_data_types_违规(self) -> None:
        """registry.validate_data_types: 非法属性应返回违规."""
        violations = self.registry.validate_data_types(
            "shacl_test", EntityType.CONCEPT, {"age": "not_a_number"}
        )
        assert len(violations) > 0

    def test_validate_data_types_领域不存在(self) -> None:
        """registry.validate_data_types: 领域不存在应返回错误信息."""
        violations = self.registry.validate_data_types(
            "nonexistent", EntityType.CONCEPT, {}
        )
        assert any("未找到" in v for v in violations)

    def test_validate_cardinality_通过(self) -> None:
        """registry.validate_cardinality: 合法基数应通过."""
        violations = self.registry.validate_cardinality(
            "shacl_test", EntityType.CONCEPT, {"tags_list": ["a", "b"]}
        )
        assert violations == []

    def test_validate_cardinality_违规(self) -> None:
        """registry.validate_cardinality: 非法基数应返回违规."""
        violations = self.registry.validate_cardinality(
            "shacl_test", EntityType.CONCEPT, {"tags_list": ["a", "b", "c", "d", "e", "f"]}
        )
        assert len(violations) > 0

    def test_validate_full_通过(self) -> None:
        """registry.validate_full: 全部合法应通过."""
        violations = self.registry.validate_full(
            "shacl_test", EntityType.CONCEPT,
            {"color": "red", "age": 25, "tags_list": ["t1"]},
        )
        assert violations == []

    def test_validate_all_含公理(self) -> None:
        """registry.validate_all: 应包含公理验证."""
        violations = self.registry.validate_all(
            "shacl_test", EntityType.CONCEPT, {"single_val": ["a", "b"]}
        )
        assert any("single_val" in v for v in violations)

    # --- 类层级封装 ---

    def test_get_class_hierarchy(self) -> None:
        """registry.get_class_hierarchy 应返回类层级."""
        hierarchy = self.registry.get_class_hierarchy("hierarchy_test", EntityType.MATERIAL)
        assert hierarchy == [EntityType.CONCEPT, EntityType.CHEMICAL_COMPOUND, EntityType.MATERIAL]

    def test_get_class_hierarchy_领域不存在(self) -> None:
        """registry.get_class_hierarchy: 领域不存在应返回空列表."""
        assert self.registry.get_class_hierarchy("nonexistent", EntityType.CONCEPT) == []

    def test_get_subclasses(self) -> None:
        """registry.get_subclasses 应返回直接子类."""
        subs = self.registry.get_subclasses("hierarchy_test", EntityType.CONCEPT)
        assert EntityType.CHEMICAL_COMPOUND in subs

    def test_get_subclasses_领域不存在(self) -> None:
        """registry.get_subclasses: 领域不存在应返回空列表."""
        assert self.registry.get_subclasses("nonexistent", EntityType.CONCEPT) == []

    def test_get_all_subclasses(self) -> None:
        """registry.get_all_subclasses 应返回所有子类."""
        all_subs = self.registry.get_all_subclasses("hierarchy_test", EntityType.CONCEPT)
        assert EntityType.CHEMICAL_COMPOUND in all_subs
        assert EntityType.MATERIAL in all_subs

    def test_is_subclass_of_是子类(self) -> None:
        """registry.is_subclass_of: 正确判断子类关系."""
        assert self.registry.is_subclass_of(
            "hierarchy_test", EntityType.MATERIAL, EntityType.CONCEPT
        ) is True

    def test_is_subclass_of_不是子类(self) -> None:
        """registry.is_subclass_of: 正确判断非子类关系."""
        assert self.registry.is_subclass_of(
            "hierarchy_test", EntityType.CONCEPT, EntityType.MATERIAL
        ) is False

    def test_is_subclass_of_领域不存在(self) -> None:
        """registry.is_subclass_of: 领域不存在应返回 False."""
        assert self.registry.is_subclass_of(
            "nonexistent", EntityType.CONCEPT, EntityType.MATERIAL
        ) is False

    # --- 推理封装 ---

    def test_apply_inference_rules(self) -> None:
        """registry.apply_inference_rules 应应用推理规则."""
        triples = [("A", "part_of", "B"), ("B", "part_of", "C"), ("X", "related_to", "Y")]
        inferred = self.registry.apply_inference_rules("rules_test", triples)
        inferred_set = set(inferred)
        assert ("A", "part_of", "C") in inferred_set
        assert ("Y", "related_to", "X") in inferred_set

    def test_apply_inference_rules_领域不存在(self) -> None:
        """registry.apply_inference_rules: 领域不存在应返回空列表."""
        assert self.registry.apply_inference_rules("nonexistent", []) == []

    def test_infer_transitive_closure(self) -> None:
        """registry.infer_transitive_closure 应推理传递闭包."""
        triples = [("A", "part_of", "B"), ("B", "part_of", "C")]
        inferred = self.registry.infer_transitive_closure("rules_test", triples, "part_of")
        assert ("A", "part_of", "C") in inferred

    def test_infer_inverse_relations(self) -> None:
        """registry.infer_inverse_relations 应推理逆关系."""
        triples = [("A", "parent_of", "B")]
        inferred = self.registry.infer_inverse_relations("inverse_test", triples, "parent_of")
        assert ("B", "child_of", "A") in inferred

    def test_infer_symmetric_closure(self) -> None:
        """registry.infer_symmetric_closure 应推理对称闭包."""
        triples = [("A", "related_to", "B")]
        inferred = self.registry.infer_symmetric_closure("rules_test", triples, "related_to")
        assert ("B", "related_to", "A") in inferred

    def test_infer_property_chain(self) -> None:
        """registry.infer_property_chain 应推理属性链."""
        triples = [("A", "part_of", "B"), ("B", "related_to", "C")]
        inferred = self.registry.infer_property_chain("rules_test", triples, ["part_of", "related_to"])
        assert ("A", "part_of_related_to", "C") in inferred

    def test_infer_subclass_inheritance(self) -> None:
        """registry.infer_subclass_inheritance 应推理子类继承."""
        instances = [("e1", EntityType.MATERIAL)]
        inferred = self.registry.infer_subclass_inheritance("hierarchy_test", instances)
        inferred_set = set(inferred)
        assert ("e1", EntityType.CONCEPT) in inferred_set

    # --- 跨本体映射封装 ---

    def test_find_mappings(self) -> None:
        """registry.find_mappings 应查找映射."""
        mappings = self.registry.find_mappings("shacl_test")
        assert len(mappings) == 2

    def test_find_mappings_按目标领域(self) -> None:
        """registry.find_mappings: 按目标领域过滤."""
        mappings = self.registry.find_mappings("shacl_test", target_domain="chemistry")
        assert len(mappings) == 1

    def test_find_mappings_领域不存在(self) -> None:
        """registry.find_mappings: 领域不存在应返回空列表."""
        assert self.registry.find_mappings("nonexistent") == []

    def test_get_equivalent_type(self) -> None:
        """registry.get_equivalent_type 应返回等价类型."""
        eq = self.registry.get_equivalent_type("shacl_test", EntityType.CONCEPT, "chemistry")
        assert eq == EntityType.CHEMICAL_COMPOUND

    def test_get_equivalent_type_领域不存在(self) -> None:
        """registry.get_equivalent_type: 领域不存在应返回 None."""
        assert self.registry.get_equivalent_type("nonexistent", EntityType.CONCEPT, "x") is None

    # --- 统计 ---

    def test_total_axioms(self) -> None:
        """total_axioms 应返回所有领域的公理总数."""
        # shacl_test 本体有 2 个公理
        total = self.registry.total_axioms()
        assert total >= 2

    def test_total_mappings(self) -> None:
        """total_mappings 应返回所有领域的映射总数."""
        # shacl_test 本体有 2 个映射
        total = self.registry.total_mappings()
        assert total >= 2


# ============================================================
# 23. 增强集成场景测试
# ============================================================


class TestEnhancedIntegrationScenarios:
    """复杂集成场景测试 — 端到端验证 L3 增强功能的实际使用."""

    def test_知识图谱完整工作流(self) -> None:
        """知识图谱工作流: 创建实体 → 添加三元组 → 检测冲突 → 解决冲突."""
        # 1. 创建知识图谱
        kg = KnowledgeGraph(domain="chemistry", name="化学知识图谱")
        assert kg.is_empty() is True

        # 2. 创建并添加实体
        water = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND,
            name="水",
            domain="chemistry",
            aliases=["H2O", "Water"],
            tags=["inorganic", "solvent"],
            properties={"formula": "H2O", "boiling_point": 100},
            status=KnowledgeStatus.ACTIVE,
        )
        kg.add_entity(water)
        assert kg.entity_count() == 1
        assert kg.is_empty() is False

        # 3. 添加跨实体三元组
        triple = KnowledgeTriple(
            subject_id=water.entity_id,
            predicate=RelationType.RELATED_TO.value,
            object_id="e-other",
        )
        kg.add_triple(triple)
        assert kg.triple_count() >= 1

        # 4. 检测冲突
        conflict = KnowledgeConflict(
            conflict_type=ConflictType.SOURCE_BASED,
            entity_id=water.entity_id,
            field_path="properties.boiling_point",
            conflicting_values=[
                {"value": 100, "source": "nist"},
                {"value": 99.97, "source": "cas"},
            ],
            resolution_strategy=ConflictResolutionStrategy.PREFER_MOST_TRUSTED,
        )
        kg.conflicts.append(conflict)
        assert kg.unresolved_conflict_count() == 1

        # 5. 解决冲突
        conflict.resolve(value=100, claim_id="cl-001", explanation="采纳NIST数据")
        assert conflict.is_resolved() is True
        assert kg.unresolved_conflict_count() == 0

        # 6. 统计
        stats = kg.get_stats()
        assert stats.total_entities == 1

    def test_版本管理工作流(self) -> None:
        """版本管理工作流: 创建实体 → 版本化 → 查询历史版本."""
        # 1. 创建实体
        entity = _make_entity(name="乙醇", properties={"formula": "C2H5OH"})

        # 2. 创建第一版
        v1 = KnowledgeVersion(
            entity_id=entity.entity_id,
            revision_number=1,
            valid_from=100.0,
            valid_until=200.0,
            changeset=[
                ChangeRecord(
                    change_type="add",
                    entity_id=entity.entity_id,
                    field_path="name",
                    new_value="乙醇",
                ),
            ],
        )

        # 3. 创建第二版 (当前版本)
        v2 = KnowledgeVersion(
            entity_id=entity.entity_id,
            revision_number=2,
            parent_version_id=v1.version_id,
            valid_from=200.0,
            valid_until=0.0,  # 当前版本
            changeset=[
                ChangeRecord(
                    change_type="modify",
                    entity_id=entity.entity_id,
                    field_path="properties.formula",
                    old_value="C2H5OH",
                    new_value="C₂H₅OH",
                ),
            ],
        )

        # 4. 验证版本属性
        assert v1.has_parent() is False
        assert v2.has_parent() is True
        assert v1.is_current() is False
        assert v2.is_current() is True
        assert v1.change_count() == 1
        assert v2.change_count() == 1

        # 5. 时间旅行查询
        assert v1.is_valid_at(150.0) is True
        assert v1.is_valid_at(250.0) is False
        assert v2.is_valid_at(150.0) is False
        assert v2.is_valid_at(250.0) is True

    def test_推理工作流(self) -> None:
        """推理工作流: 创建本体 → 添加规则 → 添加三元组 → 应用推理."""
        # 1. 创建含推理规则的本体
        onto = _build_rules_ontology()

        # 2. 添加已知三元组
        triples = [
            ("A", "part_of", "B"),
            ("B", "part_of", "C"),
            ("C", "part_of", "D"),
            ("X", "related_to", "Y"),
        ]

        # 3. 应用推理
        inferred = onto.apply_inference_rules(triples)
        inferred_set = set(inferred)

        # 4. 验证推理结果
        # 传递闭包: A→C, A→D, B→D
        assert ("A", "part_of", "C") in inferred_set
        assert ("A", "part_of", "D") in inferred_set
        assert ("B", "part_of", "D") in inferred_set
        # 对称闭包: Y→X
        assert ("Y", "related_to", "X") in inferred_set

        # 5. 推理结果不应包含原始三元组
        for t in inferred:
            assert t not in set(triples)

    def test_跨域映射工作流(self) -> None:
        """跨域映射工作流: 创建化学-材料映射 → 查找等价类型."""
        # 1. 创建含映射的本体
        onto = _build_shacl_ontology()

        # 2. 查找所有映射
        all_mappings = onto.find_mappings()
        assert len(all_mappings) == 2

        # 3. 查找特定目标领域的映射
        chem_mappings = onto.find_mappings(target_domain="chemistry")
        assert len(chem_mappings) == 1
        assert chem_mappings[0].mapping_type == "equivalent"

        # 4. 获取等价类型
        eq_type = onto.get_equivalent_type(EntityType.CONCEPT, "chemistry")
        assert eq_type == EntityType.CHEMICAL_COMPOUND

        # 5. 验证非等价映射返回 None
        assert onto.get_equivalent_type(EntityType.CONCEPT, "materials") is None

    def test_shacl验证工作流(self) -> None:
        """SHACL 验证工作流: 创建实体属性 → 验证数据类型/基数/公理."""
        onto = _build_shacl_ontology()

        # 1. 合法属性应通过所有验证
        valid_props = {
            "color": "red",
            "age": 25,
            "label": "valid_label",
            "email": "user@example.com",
            "score": 85.5,
            "active": True,
            "tags_list": ["tag1", "tag2"],
        }
        assert onto.validate_full(EntityType.CONCEPT, valid_props) == []
        assert onto.validate_all(EntityType.CONCEPT, valid_props) == []

        # 2. 非法属性应被检出
        invalid_props = {
            "color": "purple",  # 枚举违规
            "age": 200,  # 超出 max_value
            "tags_list": [],  # 低于 min_count
            "single_val": ["a", "b"],  # 函数性违规
        }
        violations = onto.validate_all(EntityType.CONCEPT, invalid_props)
        assert len(violations) >= 4  # 至少 4 个违规

    def test_查询工作流(self) -> None:
        """查询工作流: 创建知识图谱 → 构建查询 → 应用条件过滤."""
        # 1. 创建知识图谱并添加实体
        kg = KnowledgeGraph(domain="chemistry")
        water = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND,
            name="水",
            domain="chemistry",
            aliases=["H2O"],
            tags=["inorganic"],
            properties={"formula": "H2O"},
        )
        ethanol = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND,
            name="乙醇",
            domain="chemistry",
            tags=["organic"],
            properties={"formula": "C2H5OH"},
        )
        steel = KnowledgeEntity(
            entity_type=EntityType.MATERIAL,
            name="钢",
            domain="materials",
            tags=["metal"],
        )
        kg.add_entity(water)
        kg.add_entity(ethanol)
        kg.add_entity(steel)

        # 2. 构建查询: 查找化学领域的化合物
        query = KnowledgeQuery(
            domain="chemistry",
            conditions=[
                QueryCondition(
                    field="entity_type",
                    operator=QueryOperator.EQ,
                    value=EntityType.CHEMICAL_COMPOUND,
                ),
            ],
        )
        assert query.has_conditions() is True

        # 3. 手动应用查询条件
        results = []
        for entity in kg.entities.values():
            if all(
                cond.matches(getattr(entity, cond.field, None))
                for cond in query.conditions
            ):
                results.append(entity)

        assert len(results) == 2
        result_names = {e.name for e in results}
        assert "水" in result_names
        assert "乙醇" in result_names
        assert "钢" not in result_names

        # 4. 构建带名称包含条件的查询
        query2 = KnowledgeQuery(
            conditions=[
                QueryCondition(
                    field="entity_type",
                    operator=QueryOperator.EQ,
                    value=EntityType.CHEMICAL_COMPOUND,
                ),
                QueryCondition(
                    field="name",
                    operator=QueryOperator.CONTAINS,
                    value="乙",
                ),
            ],
        )
        results2 = []
        for entity in kg.entities.values():
            if all(
                cond.matches(getattr(entity, cond.field, None))
                for cond in query2.conditions
            ):
                results2.append(entity)
        assert len(results2) == 1
        assert results2[0].name == "乙醇"

        # 5. 使用 negate 条件
        query3 = KnowledgeQuery(
            conditions=[
                QueryCondition(
                    field="tags",
                    operator=QueryOperator.IN,
                    value=["organic"],
                    negate=True,
                ),
            ],
        )
        # 这个查询逻辑上不太适用于 list 字段, 但验证 negate 机制可用
        # 改为简单字段
        query3 = KnowledgeQuery(
            conditions=[
                QueryCondition(
                    field="domain",
                    operator=QueryOperator.NE,
                    value="materials",
                ),
            ],
        )
        results3 = []
        for entity in kg.entities.values():
            if all(
                cond.matches(getattr(entity, cond.field, None))
                for cond in query3.conditions
            ):
                results3.append(entity)
        assert len(results3) == 2  # water 和 ethanol, 排除 steel
