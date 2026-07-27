"""L3 高级模块综合测试.

测试 6 个 L3 知识存储引擎新模块:
1. response_synthesizer.py — 响应合成器
2. access_control.py      — 访问控制
3. audit_trail.py          — 审计轨迹
4. graph_reasoner.py       — 图推理器
5. schema_evolution.py     — 模式演进
6. kb_manager.py           — 知识库管理器
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import time

import pytest

from dy3_polaris.l3 import (
    # 核心模型
    AccessLevel,
    EntityType,
    KnowledgeEntity,
    KnowledgeStore,
    KnowledgeTriple,
    PropertyDataType,
    RelationType,
    RetrievalResult,
    # 本体
    DomainOntology,
    InferenceRuleType,
    OntologyAxiom,
    OntologyClass,
    OntologyProperty,
    OntologyRelation,
    OntologyRegistry,
    OntologyRule,
    # 响应合成器
    Citation,
    EvidencePiece,
    EvidenceType,
    QueryType,
    ResponseSynthesizer,
    SynthesisConfig,
    SynthesisMode,
    SynthesizedResponse,
    # 访问控制
    AccessControlledStore,
    AccessControlManager,
    AccessDecision,
    AccessDeniedError,
    AccessPolicy,
    AccessRequest,
    AccessResult,
    Permission,
    ResourceType as AccessResourceType,
    Role,
    User,
    # 审计轨迹
    AuditEntry,
    AuditError,
    AuditQuery,
    AuditStats,
    AuditTrail,
    ChangeDiff,
    OperationType,
    ResourceType as AuditResourceType,
    # 图推理器
    GraphReasoner,
    InferenceRule,
    PathResult,
    ReasoningError,
    ReasoningMode,
    ReasoningResult,
    # 模式演进
    ChangeType,
    CompatibilityLevel,
    MigrationPlan,
    MigrationStep,
    SchemaChange,
    SchemaDiff,
    SchemaEvolutionError,
    SchemaEvolutionManager,
    SchemaVersion,
    # 知识库管理器
    KBConfig,
    KBManagerError,
    KBLifecycleState,
    KBSnapshot,
    HealthStatus,
    KnowledgeBaseManager,
    RetentionPolicy,
)


# ============================================================
# TestResponseSynthesizer
# ============================================================


class TestResponseSynthesizer:
    """响应合成器测试。"""

    @pytest.fixture
    def synthesizer(self):
        return ResponseSynthesizer()

    @pytest.fixture
    def sample_result(self):
        return RetrievalResult(
            query="什么是Dy3+离子",
            results=[
                {
                    "type": "entity",
                    "entity_id": "e1",
                    "name": "Dy3+",
                    "description": "镝离子",
                    "content": "Dy3+是镝的三价离子",
                },
                {
                    "type": "chunk",
                    "chunk_id": "c1",
                    "content": "Dy3+的发射波长为580nm",
                    "document_id": "d1",
                },
            ],
            scores=[0.95, 0.85],
            total=2,
        )

    def test_synthesize_compact(self, synthesizer, sample_result):
        """测试紧凑模式合成。"""
        synth = ResponseSynthesizer(SynthesisConfig(mode=SynthesisMode.COMPACT))
        resp = synth.synthesize(sample_result)
        assert isinstance(resp, SynthesizedResponse)
        assert resp.answer
        assert resp.source_count == 2
        assert resp.synthesis_mode == SynthesisMode.COMPACT
        assert resp.confidence > 0.0

    def test_synthesize_refine(self, synthesizer, sample_result):
        """测试精炼模式合成。"""
        synth = ResponseSynthesizer(SynthesisConfig(mode=SynthesisMode.REFINE))
        resp = synth.synthesize(sample_result)
        assert resp.answer
        assert resp.source_count == 2
        assert resp.synthesis_mode == SynthesisMode.REFINE
        assert len(resp.evidence_pieces) == 2

    def test_synthesize_tree(self, synthesizer, sample_result):
        """测试树摘要模式合成。"""
        synth = ResponseSynthesizer(SynthesisConfig(mode=SynthesisMode.TREE_SUMMARIZE))
        resp = synth.synthesize(sample_result)
        assert resp.answer
        assert resp.source_count == 2
        assert resp.synthesis_mode == SynthesisMode.TREE_SUMMARIZE

    def test_synthesize_template(self, synthesizer, sample_result):
        """测试模板模式合成。"""
        synth = ResponseSynthesizer(SynthesisConfig(mode=SynthesisMode.TEMPLATE))
        resp = synth.synthesize(sample_result)
        assert resp.answer
        assert resp.source_count == 2
        assert resp.synthesis_mode == SynthesisMode.TEMPLATE

    def test_synthesize_bullet(self, synthesizer, sample_result):
        """测试要点模式合成。"""
        synth = ResponseSynthesizer(SynthesisConfig(mode=SynthesisMode.BULLET))
        resp = synth.synthesize(sample_result)
        assert resp.answer
        assert resp.source_count == 2
        assert resp.synthesis_mode == SynthesisMode.BULLET

    def test_synthesize_empty_results(self, synthesizer):
        """测试空检索结果合成。"""
        empty_result = RetrievalResult(query="不存在的知识", results=[], total=0)
        resp = synthesizer.synthesize(empty_result)
        # 空结果置信度为 0，来源数为 0
        assert resp.confidence == 0.0
        assert resp.source_count == 0
        assert len(resp.evidence_pieces) == 0
        assert resp.metadata.get("empty") is True

    def test_detect_query_type_definition(self, synthesizer):
        """测试定义类查询类型检测。"""
        qt = synthesizer._detect_query_type("什么是Dy3+离子")
        assert qt == QueryType.DEFINITION.value

    def test_detect_query_type_comparison(self, synthesizer):
        """测试比较类查询类型检测。"""
        qt = synthesizer._detect_query_type("Dy3+和Eu3+的区别")
        assert qt == QueryType.COMPARISON.value

    def test_detect_query_type_numeric(self, synthesizer):
        """测试数值类查询类型检测。"""
        qt = synthesizer._detect_query_type("Dy3+的发射波长是多少nm")
        assert qt == QueryType.NUMERIC.value

    def test_detect_query_type_relational(self, synthesizer):
        """测试关系类查询类型检测。"""
        qt = synthesizer._detect_query_type("Dy3+与YAG的关系")
        assert qt == QueryType.RELATIONAL.value

    def test_detect_query_type_procedural(self, synthesizer):
        """测试流程类查询类型检测。"""
        qt = synthesizer._detect_query_type("如何制备Dy3+掺杂的YAG材料")
        assert qt == QueryType.PROCEDURAL.value

    def test_confidence_calculation(self, synthesizer, sample_result):
        """测试置信度计算。"""
        resp = synthesizer.synthesize(sample_result)
        assert 0.0 < resp.confidence <= 1.0
        # 高分数应产生较高置信度
        assert resp.confidence >= 0.3

    def test_citations_generated(self, synthesizer, sample_result):
        """测试引用生成。"""
        synth = ResponseSynthesizer(SynthesisConfig(include_citations=True, max_citations=10))
        resp = synth.synthesize(sample_result)
        assert len(resp.citations) > 0
        for cit in resp.citations:
            assert isinstance(cit, Citation)
            assert cit.source_type in ("entity", "chunk", "triple")
            assert 0.0 <= cit.relevance_score <= 1.0

    def test_evidence_pieces(self, synthesizer, sample_result):
        """测试证据片段提取。"""
        resp = synthesizer.synthesize(sample_result)
        assert len(resp.evidence_pieces) == 2
        for ev in resp.evidence_pieces:
            assert isinstance(ev, EvidencePiece)
            assert ev.content
            assert ev.evidence_type in (
                EvidenceType.DIRECT.value,
                EvidenceType.INFERRED.value,
                EvidenceType.CONTEXTUAL.value,
            )

    def test_stats(self, synthesizer, sample_result):
        """测试统计信息。"""
        synthesizer.synthesize(sample_result)
        stats = synthesizer.get_stats()
        assert stats["synthesis_count"] == 1
        assert stats["mode"] == SynthesisMode.COMPACT.value
        assert "max_tokens" in stats
        assert "max_citations" in stats


# ============================================================
# TestAccessControl
# ============================================================


class TestAccessControl:
    """访问控制测试。"""

    @pytest.fixture
    def acm(self):
        return AccessControlManager()

    def test_register_user(self, acm):
        """测试用户注册。"""
        user = User(user_id="u1", username="alice", roles=[Role.READER])
        acm.register_user(user)
        retrieved = acm.get_user("u1")
        assert retrieved is not None
        assert retrieved.username == "alice"
        assert Role.READER in retrieved.roles

    def test_get_user(self, acm):
        """测试获取用户（不存在时返回 None）。"""
        assert acm.get_user("nonexistent") is None

    def test_admin_can_access_all(self, acm):
        """测试管理员可以访问所有资源和级别。"""
        admin = User(
            user_id="admin1", username="admin",
            roles=[Role.ADMIN], access_level=AccessLevel.CONFIDENTIAL,
        )
        acm.register_user(admin)
        # 管理员应能删除机密实体
        result = acm.enforce(
            "admin1", Permission.DELETE, AccessResourceType.ENTITY,
            resource_access_level=AccessLevel.CONFIDENTIAL,
        )
        assert result is True

    def test_reader_can_read_public(self, acm):
        """测试读者可以读取公开资源。"""
        reader = User(
            user_id="r1", username="reader",
            roles=[Role.READER], access_level=AccessLevel.PUBLIC,
        )
        acm.register_user(reader)
        result = acm.enforce(
            "r1", Permission.READ, AccessResourceType.ENTITY,
            resource_access_level=AccessLevel.PUBLIC,
        )
        assert result is True

    def test_reader_cannot_delete(self, acm):
        """测试读者不能删除资源。"""
        reader = User(
            user_id="r2", username="reader2",
            roles=[Role.READER], access_level=AccessLevel.INTERNAL,
        )
        acm.register_user(reader)
        with pytest.raises(AccessDeniedError):
            acm.enforce(
                "r2", Permission.DELETE, AccessResourceType.ENTITY,
                resource_access_level=AccessLevel.PUBLIC,
            )

    def test_guest_can_only_read_public(self, acm):
        """测试访客只能读取公开资源。"""
        guest = User(
            user_id="g1", username="guest",
            roles=[Role.GUEST], access_level=AccessLevel.PUBLIC,
        )
        acm.register_user(guest)
        # 访客可以读公开
        assert acm.enforce(
            "g1", Permission.READ, AccessResourceType.ENTITY,
            resource_access_level=AccessLevel.PUBLIC,
        )
        # 访客不能读内部
        with pytest.raises(AccessDeniedError):
            acm.enforce(
                "g1", Permission.READ, AccessResourceType.ENTITY,
                resource_access_level=AccessLevel.INTERNAL,
            )

    def test_editor_can_write_internal(self, acm):
        """测试编辑者可以写入内部级别资源。"""
        editor = User(
            user_id="e1", username="editor",
            roles=[Role.EDITOR], access_level=AccessLevel.INTERNAL,
        )
        acm.register_user(editor)
        result = acm.enforce(
            "e1", Permission.CREATE, AccessResourceType.ENTITY,
            resource_access_level=AccessLevel.INTERNAL,
        )
        assert result is True

    def test_contributor_can_only_write_public(self, acm):
        """测试贡献者可以创建但不能更新/删除。"""
        contributor = User(
            user_id="c1", username="contributor",
            roles=[Role.CONTRIBUTOR], access_level=AccessLevel.INTERNAL,
        )
        acm.register_user(contributor)
        # 贡献者可以创建公开级别实体
        assert acm.enforce(
            "c1", Permission.CREATE, AccessResourceType.ENTITY,
            resource_access_level=AccessLevel.PUBLIC,
        )
        # 贡献者也可以创建内部级别实体
        assert acm.enforce(
            "c1", Permission.CREATE, AccessResourceType.ENTITY,
            resource_access_level=AccessLevel.INTERNAL,
        )
        # 贡献者不能更新实体
        with pytest.raises(AccessDeniedError):
            acm.enforce(
                "c1", Permission.UPDATE, AccessResourceType.ENTITY,
                resource_access_level=AccessLevel.PUBLIC,
            )
        # 贡献者不能删除实体
        with pytest.raises(AccessDeniedError):
            acm.enforce(
                "c1", Permission.DELETE, AccessResourceType.ENTITY,
                resource_access_level=AccessLevel.PUBLIC,
            )

    def test_access_denied_raises_error(self, acm):
        """测试访问被拒绝时抛出异常。"""
        reader = User(
            user_id="r3", username="reader3",
            roles=[Role.READER], access_level=AccessLevel.PUBLIC,
        )
        acm.register_user(reader)
        with pytest.raises(AccessDeniedError) as exc_info:
            acm.enforce(
                "r3", Permission.EXPORT, AccessResourceType.ENTITY,
                resource_access_level=AccessLevel.PUBLIC,
            )
        assert exc_info.value.user_id == "r3"
        assert exc_info.value.permission == Permission.EXPORT

    def test_level_allows(self, acm):
        """测试访问级别检查（静态方法）。"""
        # PUBLIC 可以访问 PUBLIC
        assert acm._level_allows(AccessLevel.PUBLIC, AccessLevel.PUBLIC)
        # INTERNAL 可以访问 PUBLIC 和 INTERNAL
        assert acm._level_allows(AccessLevel.INTERNAL, AccessLevel.PUBLIC)
        assert acm._level_allows(AccessLevel.INTERNAL, AccessLevel.INTERNAL)
        # INTERNAL 不能访问 RESTRICTED
        assert not acm._level_allows(AccessLevel.INTERNAL, AccessLevel.RESTRICTED)
        # CONFIDENTIAL 可以访问所有
        assert acm._level_allows(AccessLevel.CONFIDENTIAL, AccessLevel.PUBLIC)

    def test_add_custom_policy(self, acm):
        """测试添加自定义策略。"""
        user = User(
            user_id="custom1", username="custom_user",
            roles=[Role.READER], access_level=AccessLevel.INTERNAL,
        )
        acm.register_user(user)
        custom_policy = AccessPolicy(
            policy_id="custom-export",
            name="自定义导出策略",
            roles=[Role.READER],
            permissions=[Permission.EXPORT],
            resource_types=[AccessResourceType.ENTITY],
            access_levels=[AccessLevel.INTERNAL],
            priority=50,
            decision=AccessDecision.ALLOW,
        )
        acm.add_policy(custom_policy)
        # 现在 reader 应该可以导出内部实体
        result = acm.enforce(
            "custom1", Permission.EXPORT, AccessResourceType.ENTITY,
            resource_access_level=AccessLevel.INTERNAL,
        )
        assert result is True

    def test_remove_policy(self, acm):
        """测试移除策略。"""
        custom_policy = AccessPolicy(
            policy_id="temp-policy",
            name="临时策略",
            roles=[Role.GUEST],
            permissions=[Permission.DELETE],
            resource_types=[AccessResourceType.ENTITY],
            access_levels=[AccessLevel.PUBLIC],
            priority=50,
            decision=AccessDecision.ALLOW,
        )
        acm.add_policy(custom_policy)
        assert acm.get_policy("temp-policy") is not None
        acm.remove_policy("temp-policy")
        assert acm.get_policy("temp-policy") is None

    def test_access_log(self, acm):
        """测试访问日志记录。"""
        admin = User(
            user_id="log-admin", username="logadmin",
            roles=[Role.ADMIN], access_level=AccessLevel.CONFIDENTIAL,
        )
        acm.register_user(admin)
        acm.enforce(
            "log-admin", Permission.READ, AccessResourceType.ENTITY,
            resource_access_level=AccessLevel.PUBLIC,
        )
        logs = acm.get_access_log(user_id="log-admin")
        assert len(logs) >= 1
        assert all(isinstance(r, AccessResult) for r in logs)
        assert all(r.user_id == "log-admin" for r in logs)

    def test_stats(self, acm):
        """测试访问控制统计信息。"""
        user = User(
            user_id="stats-user", username="stats",
            roles=[Role.READER], access_level=AccessLevel.INTERNAL,
        )
        acm.register_user(user)
        acm.enforce(
            "stats-user", Permission.READ, AccessResourceType.ENTITY,
            resource_access_level=AccessLevel.PUBLIC,
        )
        stats = acm.get_stats()
        assert stats["users"] >= 1
        assert stats["policies"] >= 5  # 5 default policies
        assert stats["total_access_requests"] >= 1
        assert "allow_rate" in stats

    def test_access_controlled_store(self, acm):
        """测试访问控制存储装饰器。"""
        editor = User(
            user_id="ac-editor", username="ac_editor",
            roles=[Role.EDITOR], access_level=AccessLevel.INTERNAL,
        )
        acm.register_user(editor)

        store = KnowledgeStore()
        ac_store = AccessControlledStore(store, acm)

        # 编辑者可以创建公开实体
        entity = KnowledgeEntity(
            entity_type=EntityType.CONCEPT,
            name="测试实体",
            access_level=AccessLevel.PUBLIC,
        )
        added = ac_store.add_entity(entity, user_id="ac-editor")
        assert added.entity_id == entity.entity_id

        # 编辑者可以读取实体
        retrieved = ac_store.get_entity(entity.entity_id, user_id="ac-editor")
        assert retrieved is not None
        assert retrieved.name == "测试实体"

        # 编辑者可以更新实体
        updated = ac_store.update_entity(
            entity.entity_id, {"description": "更新描述"}, user_id="ac-editor",
        )
        assert updated.description == "更新描述"


# ============================================================
# TestAuditTrail
# ============================================================


class TestAuditTrail:
    """审计轨迹测试。"""

    @pytest.fixture
    def trail(self):
        return AuditTrail()

    def test_log_create(self, trail):
        """测试记录创建操作。"""
        entry = trail.log(
            OperationType.CREATE,
            AuditResourceType.ENTITY,
            "entity-001",
            "user-001",
            after={"name": "Dy3+", "type": "element"},
        )
        assert isinstance(entry, AuditEntry)
        assert entry.operation == OperationType.CREATE
        assert entry.resource_id == "entity-001"
        assert entry.user_id == "user-001"
        assert entry.after_state["name"] == "Dy3+"
        assert len(entry.diffs) > 0

    def test_log_update(self, trail):
        """测试记录更新操作。"""
        before = {"name": "Dy3+", "wavelength": "580nm"}
        after = {"name": "Dy3+", "wavelength": "575nm", "intensity": "high"}
        entry = trail.log(
            OperationType.UPDATE,
            AuditResourceType.ENTITY,
            "entity-002",
            "user-002",
            before=before,
            after=after,
        )
        assert entry.operation == OperationType.UPDATE
        assert entry.before_state["wavelength"] == "580nm"
        assert entry.after_state["wavelength"] == "575nm"
        # 应检测到 wavelength 变更和 intensity 新增
        diff_fields = {d.field for d in entry.diffs}
        assert "wavelength" in diff_fields
        assert "intensity" in diff_fields

    def test_log_delete(self, trail):
        """测试记录删除操作。"""
        entry = trail.log(
            OperationType.DELETE,
            AuditResourceType.TRIPLE,
            "triple-001",
            "user-003",
            before={"subject": "Dy", "predicate": "doped_in", "object": "YAG"},
        )
        assert entry.operation == OperationType.DELETE
        assert entry.before_state["subject"] == "Dy"
        assert entry.after_state == {}

    def test_compute_diff(self, trail):
        """测试差异计算。"""
        before = {"a": 1, "b": 2, "c": 3}
        after = {"a": 1, "b": 20, "d": 4}
        diffs = trail.compute_diff(before, after)
        diff_map = {d.field: d for d in diffs}
        # a 没有变化所以不在 diffs 中
        assert "a" not in diff_map
        assert diff_map["b"].change_type == "modified"
        assert diff_map["b"].old_value == 2
        assert diff_map["b"].new_value == 20
        assert diff_map["c"].change_type == "removed"
        assert diff_map["d"].change_type == "added"

    def test_query_by_user(self, trail):
        """测试按用户查询审计条目。"""
        trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "e1", "alice")
        trail.log(OperationType.UPDATE, AuditResourceType.ENTITY, "e2", "bob")
        trail.log(OperationType.DELETE, AuditResourceType.ENTITY, "e3", "alice")
        query = AuditQuery(user_id="alice")
        results = trail.query(query)
        assert len(results) == 2
        assert all(r.user_id == "alice" for r in results)

    def test_query_by_operation(self, trail):
        """测试按操作类型查询。"""
        trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "e1", "u1")
        trail.log(OperationType.UPDATE, AuditResourceType.ENTITY, "e2", "u1")
        trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "e3", "u1")
        query = AuditQuery(operation=OperationType.CREATE)
        results = trail.query(query)
        assert len(results) == 2
        assert all(r.operation == OperationType.CREATE for r in results)

    def test_query_by_resource(self, trail):
        """测试按资源 ID 查询。"""
        trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "res-001", "u1")
        trail.log(OperationType.UPDATE, AuditResourceType.ENTITY, "res-001", "u2")
        trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "res-002", "u1")
        query = AuditQuery(resource_id="res-001")
        results = trail.query(query)
        assert len(results) == 2
        assert all(r.resource_id == "res-001" for r in results)

    def test_get_resource_history(self, trail):
        """测试获取资源完整历史。"""
        for i in range(5):
            trail.log(
                OperationType.UPDATE, AuditResourceType.ENTITY,
                "hist-001", "u1",
                before={"v": i}, after={"v": i + 1},
            )
        history = trail.get_resource_history("hist-001")
        assert len(history) == 5
        # 应按时间戳降序排列
        timestamps = [e.timestamp for e in history]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_get_user_activity(self, trail):
        """测试获取用户活动。"""
        for i in range(3):
            trail.log(
                OperationType.CREATE, AuditResourceType.ENTITY,
                f"e-{i}", "activity-user",
            )
        activity = trail.get_user_activity("activity-user")
        assert len(activity) == 3
        assert all(e.user_id == "activity-user" for e in activity)

    def test_export_json(self, trail):
        """测试 JSON 导出。"""
        trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "e1", "u1")
        trail.log(OperationType.UPDATE, AuditResourceType.ENTITY, "e2", "u2")
        json_str = trail.export_json()
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 2
        assert "entry_id" in data[0]
        assert "operation" in data[0]

    def test_export_csv(self, trail):
        """测试 CSV 导出。"""
        trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "e1", "u1")
        trail.log(OperationType.DELETE, AuditResourceType.TRIPLE, "t1", "u2")
        csv_str = trail.export_csv()
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        # 表头 + 2 行数据
        assert len(rows) == 3
        assert "entry_id" in rows[0]
        assert "operation" in rows[0]

    def test_seal(self, trail):
        """测试密封审计轨迹。"""
        trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "e1", "u1")
        assert not trail.sealed
        trail.seal()
        assert trail.sealed
        # 密封后追加应抛出异常
        with pytest.raises(AuditError):
            trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "e2", "u1")

    def test_verify_integrity(self, trail):
        """测试完整性校验。"""
        trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "e1", "u1")
        trail.log(OperationType.UPDATE, AuditResourceType.ENTITY, "e1", "u1")
        trail.log(OperationType.DELETE, AuditResourceType.ENTITY, "e2", "u2")
        assert trail.verify_integrity() is True

    def test_stats(self, trail):
        """测试审计统计信息。"""
        trail.log(OperationType.CREATE, AuditResourceType.ENTITY, "e1", "alice")
        trail.log(OperationType.UPDATE, AuditResourceType.TRIPLE, "t1", "bob")
        trail.log(OperationType.DELETE, AuditResourceType.ENTITY, "e2", "alice")
        stats = trail.get_stats()
        assert isinstance(stats, AuditStats)
        assert stats.total_entries == 3
        assert stats.entries_by_operation.get("create") == 1
        assert stats.entries_by_operation.get("update") == 1
        assert stats.entries_by_operation.get("delete") == 1
        assert stats.entries_by_user.get("alice") == 2
        assert stats.entries_by_user.get("bob") == 1
        assert stats.time_range is not None


# ============================================================
# TestGraphReasoner
# ============================================================


class TestGraphReasoner:
    """图推理器测试。"""

    @pytest.fixture
    def store_with_data(self):
        store = KnowledgeStore()
        e1 = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="Dy", entity_id="e1",
        )
        e2 = KnowledgeEntity(
            entity_type=EntityType.MATERIAL, name="YAG", entity_id="e2",
        )
        e3 = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="Y", entity_id="e3",
        )
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_entity(e3)
        store.add_triple(KnowledgeTriple(
            subject_id="e1", predicate="DOPED_IN", object_id="e2",
            subject_name="Dy", object_name="YAG",
        ))
        store.add_triple(KnowledgeTriple(
            subject_id="e3", predicate="HAS_ELEMENT", object_id="e2",
            subject_name="Y", object_name="YAG",
        ))
        return store

    @pytest.fixture
    def reasoner(self, store_with_data):
        return GraphReasoner(store_with_data)

    def test_find_shortest_path(self, reasoner):
        """测试查找最短路径。"""
        result = reasoner.find_shortest_path("e1", "e2")
        assert result is not None
        assert isinstance(result, PathResult)
        assert result.path[0] == "e1"
        assert result.path[-1] == "e2"
        assert result.hop_count >= 1

    def test_find_shortest_path_no_path(self, reasoner):
        """测试无路径时返回 None。"""
        # e2 -> e1 无出边（有向图），无法到达
        result = reasoner.find_shortest_path("e2", "e1", directed=True)
        assert result is None

    def test_find_k_shortest_paths(self, reasoner):
        """测试 K 最短路径。"""
        results = reasoner.find_k_shortest_paths("e1", "e2", k=3)
        assert isinstance(results, list)
        assert len(results) >= 1
        # 第一条应是最短路径
        assert results[0].path[0] == "e1"
        assert results[0].path[-1] == "e2"

    def test_multi_hop_reasoning(self, reasoner):
        """测试多跳推理。"""
        # 从 e1 沿 DOPED_IN 出发
        results = reasoner.multi_hop_reasoning("e1", ["DOPED_IN"])
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0]["entity_id"] == "e2"
        assert results[0]["hop"] == 1

    def test_forward_chaining(self, reasoner):
        """测试前向链式推理。"""
        # 添加传递性三元组用于推理
        store = reasoner._store
        store.add_triple(KnowledgeTriple(
            subject_id="e2", predicate="is_a", object_id="e3",
            subject_name="YAG", object_name="Y",
        ))
        store.add_triple(KnowledgeTriple(
            subject_id="e3", predicate="is_a", object_id="e1",
            subject_name="Y", object_name="Dy",
        ))
        inferred = reasoner.forward_chaining()
        assert isinstance(inferred, list)
        # is_a 传递性应推理出新三元组
        assert len(inferred) > 0

    def test_predict_links(self, reasoner):
        """测试链接预测。"""
        # 添加更多连接以创建共同邻居
        store = reasoner._store
        e4 = KnowledgeEntity(
            entity_type=EntityType.CONCEPT, name="Host", entity_id="e4",
        )
        store.add_entity(e4)
        store.add_triple(KnowledgeTriple(
            subject_id="e1", predicate="related_to", object_id="e4",
        ))
        store.add_triple(KnowledgeTriple(
            subject_id="e3", predicate="related_to", object_id="e4",
        ))
        predictions = reasoner.predict_links("e1", top_k=5)
        assert isinstance(predictions, list)
        # e3 和 e1 有共同邻居 e4，应出现在预测中
        pred_ids = [p["entity_id"] for p in predictions]
        assert "e3" in pred_ids

    def test_pattern_match(self, reasoner):
        """测试子图模式匹配。"""
        pattern = {
            "nodes": [
                {"var": "x", "type": EntityType.CHEMICAL_COMPOUND},
                {"var": "y", "type": EntityType.MATERIAL},
            ],
            "edges": [
                {"from": "x", "to": "y", "predicate": "DOPED_IN"},
            ],
        }
        results = reasoner.pattern_match(pattern)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert "bindings" in results[0]
        assert "names" in results[0]

    def test_analogical_reasoning(self, reasoner):
        """测试类比推理。"""
        # source_pair = (e1, e2)，关系 DOPED_IN
        # 查找 e3 以相同谓词关联的实体
        store = reasoner._store
        e4 = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="Eu", entity_id="e4",
        )
        e5 = KnowledgeEntity(
            entity_type=EntityType.MATERIAL, name="Glass", entity_id="e5",
        )
        store.add_entity(e4)
        store.add_entity(e5)
        store.add_triple(KnowledgeTriple(
            subject_id="e4", predicate="DOPED_IN", object_id="e5",
        ))
        results = reasoner.analogical_reasoning(("e1", "e2"), "e4")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0]["source_relation"] == "DOPED_IN"
        assert results[0]["predicted_entity"] == "e5"

    def test_explain_path(self, reasoner):
        """测试路径解释生成。"""
        explanation = reasoner.explain_path(
            ["e1", "e2"],
            [{"predicate": "DOPED_IN", "weight": 1.0}],
        )
        assert isinstance(explanation, str)
        assert "Dy" in explanation
        assert "DOPED_IN" in explanation
        assert "YAG" in explanation

    def test_add_rule(self, reasoner):
        """测试添加推理规则。"""
        initial_count = len(reasoner.get_rules())
        new_rule = InferenceRule(
            rule_id="custom-rule-001",
            name="自定义规则",
            condition_pattern={"type": "match_infer", "predicate": "custom_pred"},
            inference_pattern={"predicate": "inferred_pred"},
            confidence=0.8,
            description="测试用自定义推理规则",
        )
        reasoner.add_rule(new_rule)
        rules = reasoner.get_rules()
        assert len(rules) == initial_count + 1
        assert any(r.rule_id == "custom-rule-001" for r in rules)

    def test_get_rules(self, reasoner):
        """测试获取推理规则列表。"""
        rules = reasoner.get_rules()
        assert isinstance(rules, list)
        # 默认有 3 条规则
        assert len(rules) >= 3
        rule_ids = {r.rule_id for r in rules}
        assert "rule_transitive_is_a" in rule_ids
        assert "rule_inverse_has_part" in rule_ids

    def test_stats(self, reasoner):
        """测试推理器统计信息。"""
        stats = reasoner.get_stats()
        assert isinstance(stats, dict)
        assert stats["rules_count"] >= 3
        assert stats["entities_count"] == 3
        assert stats["triples_count"] == 2
        assert "reasoning_modes" in stats


# ============================================================
# TestSchemaEvolution
# ============================================================


class TestSchemaEvolution:
    """模式演进测试。"""

    @pytest.fixture
    def manager(self):
        return SchemaEvolutionManager()

    def _make_ontology_v1(self) -> DomainOntology:
        """创建 v1 本体。"""
        cls = OntologyClass(
            class_id="cls-element",
            entity_type=EntityType.CHEMICAL_COMPOUND,
            display_name="化学元素",
            description="化学元素类",
            properties=[
                OntologyProperty(
                    name="atomic_number",
                    display_name="原子序数",
                    property_type="datatype",
                    range="integer",
                    data_type=PropertyDataType.INTEGER,
                    required=True,
                ),
                OntologyProperty(
                    name="symbol",
                    display_name="元素符号",
                    property_type="datatype",
                    range="string",
                    data_type=PropertyDataType.STRING,
                ),
            ],
        )
        rel = OntologyRelation(
            name="doped_in",
            display_name="掺杂于",
            domain=[EntityType.CHEMICAL_COMPOUND],
            range=[EntityType.MATERIAL],
        )
        return DomainOntology(
            ontology_id="ont-chem-v1",
            domain="chemistry",
            display_name="化学本体",
            version="1.0.0",
            classes=[cls],
            relations=[rel],
        )

    def _make_ontology_v2(self) -> DomainOntology:
        """创建 v2 本体（新增属性和关系）。"""
        cls = OntologyClass(
            class_id="cls-element",
            entity_type=EntityType.CHEMICAL_COMPOUND,
            display_name="化学元素",
            description="化学元素类",
            properties=[
                OntologyProperty(
                    name="atomic_number",
                    display_name="原子序数",
                    property_type="datatype",
                    range="integer",
                    data_type=PropertyDataType.INTEGER,
                    required=True,
                ),
                OntologyProperty(
                    name="symbol",
                    display_name="元素符号",
                    property_type="datatype",
                    range="string",
                    data_type=PropertyDataType.STRING,
                ),
                OntologyProperty(
                    name="atomic_mass",
                    display_name="原子量",
                    property_type="datatype",
                    range="float",
                    data_type=PropertyDataType.FLOAT,
                ),
            ],
        )
        new_cls = OntologyClass(
            class_id="cls-material",
            entity_type=EntityType.MATERIAL,
            display_name="材料",
            description="材料类",
        )
        rel = OntologyRelation(
            name="doped_in",
            display_name="掺杂于",
            domain=[EntityType.CHEMICAL_COMPOUND],
            range=[EntityType.MATERIAL],
        )
        new_rel = OntologyRelation(
            name="has_property",
            display_name="具有属性",
        )
        return DomainOntology(
            ontology_id="ont-chem-v2",
            domain="chemistry",
            display_name="化学本体",
            version="2.0.0",
            classes=[cls, new_cls],
            relations=[rel, new_rel],
        )

    def _make_ontology_v3_breaking(self) -> DomainOntology:
        """创建 v3 本体（破坏性变更：删除属性和类）。"""
        cls = OntologyClass(
            class_id="cls-element",
            entity_type=EntityType.CHEMICAL_COMPOUND,
            display_name="化学元素",
            description="化学元素类",
            properties=[
                OntologyProperty(
                    name="atomic_number",
                    display_name="原子序数",
                    property_type="datatype",
                    range="integer",
                    data_type=PropertyDataType.INTEGER,
                    required=True,
                ),
                # symbol 属性被删除
            ],
        )
        return DomainOntology(
            ontology_id="ont-chem-v3",
            domain="chemistry",
            display_name="化学本体",
            version="3.0.0",
            classes=[cls],
            relations=[],
        )

    def test_record_version(self, manager):
        """测试记录版本。"""
        ont = self._make_ontology_v1()
        version = manager.record_version("chemistry", ont, description="初始版本")
        assert isinstance(version, SchemaVersion)
        assert version.version == "1.0.0"
        assert version.domain == "chemistry"
        assert version.description == "初始版本"
        # 首个版本无变更
        assert len(version.changes) == 0

    def test_compute_diff(self, manager):
        """测试计算版本差异。"""
        manager.record_version("chemistry", self._make_ontology_v1())
        manager.record_version("chemistry", self._make_ontology_v2())
        diff = manager.compute_diff("chemistry", "1.0.0", "2.0.0")
        assert isinstance(diff, SchemaDiff)
        assert diff.from_version == "1.0.0"
        assert diff.to_version == "2.0.0"
        assert len(diff.changes) > 0
        # v2 新增了 atomic_mass 属性和 has_property 关系
        change_types = {c.change_type for c in diff.changes}
        assert ChangeType.PROPERTY_ADDED in change_types or ChangeType.RELATION_ADDED in change_types

    def test_check_compatibility_full(self, manager):
        """测试完全兼容性检查。"""
        ont_v1 = self._make_ontology_v1()
        manager.record_version("chemistry", ont_v1)
        # v2 只有新增 -> FULL 兼容
        manager.record_version("chemistry", self._make_ontology_v2())
        compat = manager.check_compatibility("chemistry", "1.0.0", "2.0.0")
        assert compat in (CompatibilityLevel.FULL, CompatibilityLevel.PARTIAL)

    def test_check_compatibility_breaking(self, manager):
        """测试破坏性兼容性检查。"""
        manager.record_version("chemistry", self._make_ontology_v1())
        manager.record_version("chemistry", self._make_ontology_v2())
        manager.record_version("chemistry", self._make_ontology_v3_breaking())
        compat = manager.check_compatibility("chemistry", "2.0.0", "3.0.0")
        assert compat in (CompatibilityLevel.BREAKING, CompatibilityLevel.INCOMPATIBLE)

    def test_generate_migration_plan(self, manager):
        """测试生成迁移计划。"""
        manager.record_version("chemistry", self._make_ontology_v1())
        manager.record_version("chemistry", self._make_ontology_v2())
        plan = manager.generate_migration_plan("chemistry", "1.0.0", "2.0.0")
        assert isinstance(plan, MigrationPlan)
        assert plan.from_version == "1.0.0"
        assert plan.to_version == "2.0.0"
        assert len(plan.steps) > 0
        assert plan.compatibility in (
            CompatibilityLevel.FULL, CompatibilityLevel.PARTIAL,
        )

    def test_execute_migration(self, manager):
        """测试执行迁移计划。"""
        manager.record_version("chemistry", self._make_ontology_v1())
        manager.record_version("chemistry", self._make_ontology_v2())
        plan = manager.generate_migration_plan("chemistry", "1.0.0", "2.0.0")
        result = manager.execute_migration(plan)
        assert result["status"] == "completed"
        assert result["completed_steps"] == len(plan.steps)
        assert result["mode"] == "实际"

    def test_execute_migration_dry_run(self, manager):
        """测试模拟执行迁移计划。"""
        manager.record_version("chemistry", self._make_ontology_v1())
        manager.record_version("chemistry", self._make_ontology_v2())
        plan = manager.generate_migration_plan("chemistry", "1.0.0", "2.0.0")
        result = manager.execute_migration(plan, dry_run=True)
        assert result["status"] == "completed"
        assert result["mode"] == "模拟"
        for sr in result["step_results"]:
            assert sr["status"] == "simulated"

    def test_rollback_migration(self, manager):
        """测试回滚迁移计划。"""
        manager.record_version("chemistry", self._make_ontology_v1())
        manager.record_version("chemistry", self._make_ontology_v2())
        plan = manager.generate_migration_plan("chemistry", "1.0.0", "2.0.0")
        manager.execute_migration(plan)
        result = manager.rollback_migration(plan)
        assert result["status"] == "completed"
        assert result["rolled_back_steps"] > 0

    def test_get_version_history(self, manager):
        """测试获取版本历史。"""
        manager.record_version("chemistry", self._make_ontology_v1())
        manager.record_version("chemistry", self._make_ontology_v2())
        history = manager.get_version_history("chemistry")
        assert len(history) == 2
        assert history[0].version == "1.0.0"
        assert history[1].version == "2.0.0"

    def test_get_latest_version(self, manager):
        """测试获取最新版本。"""
        manager.record_version("chemistry", self._make_ontology_v1())
        manager.record_version("chemistry", self._make_ontology_v2())
        latest = manager.get_latest_version("chemistry")
        assert latest is not None
        assert latest.version == "2.0.0"
        # 无历史版本时返回 None
        assert manager.get_latest_version("nonexistent") is None


# ============================================================
# TestKBManager
# ============================================================


class TestKBManager:
    """知识库管理器测试。"""

    @pytest.fixture
    def manager(self):
        mgr = KnowledgeBaseManager()
        # 使用临时目录作为基础目录
        tmp_dir = tempfile.mkdtemp(prefix="kb_test_")
        mgr._base_dir = type(mgr._base_dir)(tmp_dir)
        yield mgr
        # 清理临时目录
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_create_kb(self, manager):
        """测试创建知识库。"""
        config = KBConfig(kb_id="kb-001", name="测试知识库", domain="chemistry")
        kb_id = manager.create_kb(config)
        assert kb_id == "kb-001"
        state = manager.get_state("kb-001")
        assert state == KBLifecycleState.ACTIVE

    def test_open_kb(self, manager):
        """测试打开知识库。"""
        config = KBConfig(kb_id="kb-002", name="测试KB", domain="general")
        manager.create_kb(config)
        store = manager.open_kb("kb-002")
        assert store is not None
        # 可以写入数据
        entity = KnowledgeEntity(
            entity_type=EntityType.CONCEPT, name="测试实体",
        )
        store.add_entity(entity)
        assert store.entity_count() == 1

    def test_close_kb(self, manager):
        """测试关闭知识库。"""
        config = KBConfig(kb_id="kb-003", name="关闭测试")
        manager.create_kb(config)
        manager.close_kb("kb-003")
        # 关闭后仍可重新打开
        store = manager.open_kb("kb-003")
        assert store is not None

    def test_delete_kb(self, manager):
        """测试删除知识库。"""
        config = KBConfig(kb_id="kb-004", name="删除测试")
        manager.create_kb(config)
        # 必须先转为 ARCHIVED 才能删除
        manager.transition_state("kb-004", KBLifecycleState.DEPRECATED)
        manager.transition_state("kb-004", KBLifecycleState.ARCHIVED)
        manager.delete_kb("kb-004")
        state = manager.get_state("kb-004")
        assert state == KBLifecycleState.DELETED

    def test_transition_state(self, manager):
        """测试状态流转。"""
        config = KBConfig(kb_id="kb-005", name="状态测试")
        manager.create_kb(config)
        # ACTIVE -> READONLY
        new_state = manager.transition_state("kb-005", KBLifecycleState.READONLY)
        assert new_state == KBLifecycleState.READONLY
        # READONLY -> DEPRECATED
        new_state = manager.transition_state("kb-005", KBLifecycleState.DEPRECATED)
        assert new_state == KBLifecycleState.DEPRECATED
        # DEPRECATED -> ARCHIVED
        new_state = manager.transition_state("kb-005", KBLifecycleState.ARCHIVED)
        assert new_state == KBLifecycleState.ARCHIVED

    def test_invalid_transition(self, manager):
        """测试非法状态流转。"""
        config = KBConfig(kb_id="kb-006", name="非法流转测试")
        manager.create_kb(config)
        # ACTIVE -> ARCHIVED 是非法的
        with pytest.raises(KBManagerError):
            manager.transition_state("kb-006", KBLifecycleState.ARCHIVED)

    def test_backup_restore(self, manager):
        """测试备份与恢复。"""
        config = KBConfig(
            kb_id="kb-007", name="备份测试",
            auto_backup=False,
        )
        manager.create_kb(config)
        store = manager.open_kb("kb-007")
        entity = KnowledgeEntity(
            entity_type=EntityType.CHEMICAL_COMPOUND, name="Dy",
        )
        store.add_entity(entity)
        # 创建备份
        snapshot = manager.backup("kb-007", description="测试备份")
        assert isinstance(snapshot, KBSnapshot)
        assert snapshot.snapshot_id
        assert snapshot.total_entities >= 1
        # 恢复
        result = manager.restore("kb-007", snapshot.snapshot_id)
        assert result["restored"] is True
        assert "stats" in result

    def test_list_snapshots(self, manager):
        """测试列出快照。"""
        config = KBConfig(kb_id="kb-008", name="快照列表测试", auto_backup=False)
        manager.create_kb(config)
        manager.backup("kb-008", description="快照1")
        manager.backup("kb-008", description="快照2")
        snapshots = manager.list_snapshots("kb-008")
        assert len(snapshots) == 2
        # 应按时间戳降序
        assert snapshots[0].timestamp >= snapshots[1].timestamp

    def test_run_gc(self, manager):
        """测试垃圾回收。"""
        config = KBConfig(kb_id="kb-009", name="GC测试")
        manager.create_kb(config)
        store = manager.open_kb("kb-009")
        # 添加实体但不添加三元组引用 -> 孤儿实体
        e1 = KnowledgeEntity(
            entity_type=EntityType.CONCEPT, name="孤儿实体",
        )
        store.add_entity(e1)
        # 添加被引用的实体
        e2 = KnowledgeEntity(
            entity_type=EntityType.CONCEPT, name="被引用实体",
        )
        store.add_entity(e2)
        store.add_triple(KnowledgeTriple(
            subject_id=e2.entity_id, predicate="related_to",
            object_id=e2.entity_id,
        ))
        result = manager.run_gc("kb-009")
        assert result["kb_id"] == "kb-009"
        assert result["orphaned_found"] >= 1
        assert result["removed"] >= 1

    def test_enforce_retention(self, manager):
        """测试执行保留策略。"""
        config = KBConfig(
            kb_id="kb-010", name="保留策略测试",
            retention=RetentionPolicy(
                max_age_days=365,
                max_entries=1000000,
                auto_archive=True,
                quality_threshold=0.3,
            ),
        )
        manager.create_kb(config)
        store = manager.open_kb("kb-010")
        entity = KnowledgeEntity(
            entity_type=EntityType.CONCEPT, name="保留测试实体",
        )
        store.add_entity(entity)
        result = manager.enforce_retention("kb-010")
        assert result["kb_id"] == "kb-010"
        assert "archived" in result
        assert "deleted" in result
        assert "remaining_entities" in result

    def test_check_health(self, manager):
        """测试健康检查。"""
        config = KBConfig(kb_id="kb-011", name="健康检查测试")
        manager.create_kb(config)
        store = manager.open_kb("kb-011")
        entity = KnowledgeEntity(
            entity_type=EntityType.CONCEPT, name="健康检查实体",
        )
        store.add_entity(entity)
        health = manager.check_health("kb-011")
        assert isinstance(health, HealthStatus)
        assert health.total_entities == 1
        assert health.state in ("healthy", "warning", "critical")
        assert isinstance(health.issues, list)

    def test_list_kbs(self, manager):
        """测试列出所有知识库。"""
        manager.create_kb(KBConfig(kb_id="kb-list-1", name="KB1"))
        manager.create_kb(KBConfig(kb_id="kb-list-2", name="KB2"))
        kbs = manager.list_kbs()
        assert len(kbs) == 2
        kb_ids = {kb["kb_id"] for kb in kbs}
        assert "kb-list-1" in kb_ids
        assert "kb-list-2" in kb_ids

    def test_get_config(self, manager):
        """测试获取知识库配置。"""
        config = KBConfig(
            kb_id="kb-config", name="配置测试",
            domain="physics", description="测试配置获取",
        )
        manager.create_kb(config)
        retrieved = manager.get_config("kb-config")
        assert retrieved is not None
        assert retrieved.name == "配置测试"
        assert retrieved.domain == "physics"
        # 不存在的知识库返回 None
        assert manager.get_config("nonexistent") is None

    def test_update_config(self, manager):
        """测试更新知识库配置。"""
        config = KBConfig(kb_id="kb-update", name="原始名称", domain="general")
        manager.create_kb(config)
        updated = manager.update_config("kb-update", {"name": "更新后名称", "domain": "chemistry"})
        assert updated.name == "更新后名称"
        assert updated.domain == "chemistry"
        # kb_id 不可更新
        updated2 = manager.update_config("kb-update", {"kb_id": "new-id"})
        assert updated2.kb_id == "kb-update"
