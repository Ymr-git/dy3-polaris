"""L3 知识存储引擎和检索引擎测试套件.

覆盖范围:
- EntityStore: CRUD、多维索引、搜索、合并、批量操作、统计
- TripleStore: CRUD、图查询、BFS遍历、路径查找、时间有效性
- ChunkStore: CRUD、全文检索、向量检索、文档管理
- KnowledgeStore: 统一操作、版本管理、冲突追踪、证据、子图提取、结构化查询、批量导入
- 检索引擎: 向量检索、关键词检索、图检索、混合检索、融合策略
"""

from __future__ import annotations

import threading
import time

import pytest

from dy3_polaris.l3 import (
    ChunkStore,
    ConflictResolutionStrategy,
    ConflictType,
    DocumentChunk,
    EmbeddingVector,
    EntityStore,
    EntityType,
    EvidenceRecord,
    HybridRetriever,
    KnowledgeConflict,
    KnowledgeEntity,
    KnowledgeQuery,
    KnowledgeSource,
    KnowledgeStore,
    KnowledgeTriple,
    QueryCondition,
    QueryOperator,
    RelationType,
    RetrievalEngine,
    RetrievalFilter,
    RetrievalResult,
    SourceTier,
    StatementRank,
    SubgraphConfig,
    TripleStore,
    VectorRetriever,
    KeywordRetriever,
    GraphRetriever,
)
from dy3_polaris.l3.exceptions import (
    DuplicateEntityError,
    EntityNotFoundError,
    VersionConflictError,
)


# ============================================================
# 测试数据工厂
# ============================================================


def make_entity(
    name: str = "测试实体",
    entity_type: EntityType = EntityType.CONCEPT,
    domain: str = "test",
    identifiers: dict | None = None,
    tags: list[str] | None = None,
    aliases: list[str] | None = None,
    **kwargs,
) -> KnowledgeEntity:
    """创建测试实体."""
    return KnowledgeEntity(
        name=name,
        entity_type=entity_type,
        domain=domain,
        identifiers=identifiers or {},
        tags=tags or [],
        aliases=aliases or [],
        **kwargs,
    )


def make_triple(
    subject_id: str,
    predicate: str = RelationType.RELATED_TO.value,
    object_id: str = "",
    confidence: float = 1.0,
    rank: StatementRank = StatementRank.NORMAL,
    **kwargs,
) -> KnowledgeTriple:
    """创建测试三元组."""
    return KnowledgeTriple(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        confidence=confidence,
        rank=rank,
        **kwargs,
    )


def make_chunk(
    content: str = "测试内容",
    document_id: str = "doc-001",
    chunk_index: int = 0,
    **kwargs,
) -> DocumentChunk:
    """创建测试切片."""
    return DocumentChunk(
        content=content,
        document_id=document_id,
        chunk_index=chunk_index,
        **kwargs,
    )


# ============================================================
# EntityStore 测试
# ============================================================


class TestEntityStore:
    """实体存储测试."""

    def test_add_and_get_entity(self):
        """添加和获取实体."""
        store = EntityStore()
        entity = make_entity(name="水", identifiers={"cas": "7732-18-5"})
        store.add_entity(entity)

        assert store.count() == 1
        assert store.exists(entity.entity_id)
        assert store.get_entity(entity.entity_id) is entity

    def test_add_duplicate_id_raises(self):
        """重复 entity_id 抛异常."""
        store = EntityStore()
        entity = make_entity(name="水")
        store.add_entity(entity)

        with pytest.raises(DuplicateEntityError):
            store.add_entity(entity)

    def test_add_duplicate_identifier_raises(self):
        """重复标识符抛异常."""
        store = EntityStore()
        e1 = make_entity(name="水", identifiers={"cas": "7732-18-5"})
        store.add_entity(e1)

        e2 = make_entity(name="Water", identifiers={"cas": "7732-18-5"})
        with pytest.raises(DuplicateEntityError):
            store.add_entity(e2)

    def test_add_duplicate_identifier_skip(self):
        """跳过重复标识符."""
        store = EntityStore()
        e1 = make_entity(name="水", identifiers={"cas": "7732-18-5"})
        store.add_entity(e1)

        e2 = make_entity(name="Water", identifiers={"cas": "7732-18-5"})
        store.add_entity(e2, check_duplicate=False)
        assert store.count() == 2

    def test_get_entity_or_raise(self):
        """获取不存在的实体抛异常."""
        store = EntityStore()
        with pytest.raises(EntityNotFoundError):
            store.get_entity_or_raise("non-existent")

    def test_update_entity(self):
        """更新实体字段."""
        store = EntityStore()
        entity = make_entity(name="水", description="一种液体")
        store.add_entity(entity)

        updated = store.update_entity(entity.entity_id, description="无色无味的液体")
        assert updated.description == "无色无味的液体"
        assert updated.version == 2

    def test_update_entity_optimistic_lock(self):
        """乐观版本控制."""
        store = EntityStore()
        entity = make_entity(name="水")
        store.add_entity(entity)

        # 正确版本号
        store.update_entity(entity.entity_id, expected_version=1, description="更新1")

        # 错误版本号
        with pytest.raises(VersionConflictError):
            store.update_entity(entity.entity_id, expected_version=1, description="更新2")

    def test_remove_entity(self):
        """移除实体."""
        store = EntityStore()
        entity = make_entity(name="水", tags=["chemistry"])
        store.add_entity(entity)

        removed = store.remove_entity(entity.entity_id)
        assert removed is not None
        assert store.count() == 0
        assert not store.exists(entity.entity_id)

        # 移除后索引也清除
        assert len(store.find_by_tag("chemistry")) == 0

    def test_find_by_type(self):
        """按类型查找."""
        store = EntityStore()
        e1 = make_entity(name="水", entity_type=EntityType.CHEMICAL_COMPOUND)
        e2 = make_entity(name="铁", entity_type=EntityType.MATERIAL)
        e3 = make_entity(name="二氧化碳", entity_type=EntityType.CHEMICAL_COMPOUND)
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_entity(e3)

        compounds = store.find_by_type(EntityType.CHEMICAL_COMPOUND)
        assert len(compounds) == 2

    def test_find_by_domain(self):
        """按领域查找."""
        store = EntityStore()
        e1 = make_entity(name="水", domain="chemistry")
        e2 = make_entity(name="铁", domain="materials")
        store.add_entity(e1)
        store.add_entity(e2)

        assert len(store.find_by_domain("chemistry")) == 1
        assert len(store.find_by_domain("materials")) == 1

    def test_find_by_name(self):
        """按名称查找."""
        store = EntityStore()
        e1 = make_entity(name="水", aliases=["H2O", "water"])
        store.add_entity(e1)

        assert len(store.find_by_name("水")) == 1
        assert len(store.find_by_name("H2O")) == 1
        assert len(store.find_by_name("water")) == 1
        assert len(store.find_by_name("unknown")) == 0

    def test_find_by_tag(self):
        """按标签查找."""
        store = EntityStore()
        e1 = make_entity(name="水", tags=["chemistry", "liquid"])
        e2 = make_entity(name="铁", tags=["materials", "solid"])
        store.add_entity(e1)
        store.add_entity(e2)

        assert len(store.find_by_tag("chemistry")) == 1
        assert len(store.find_by_tag("liquid")) == 1

    def test_find_by_identifier(self):
        """按标识符查找."""
        store = EntityStore()
        e1 = make_entity(name="水", identifiers={"cas": "7732-18-5"})
        store.add_entity(e1)

        found = store.find_by_identifier("cas", "7732-18-5")
        assert found is not None
        assert found.name == "水"

        assert store.find_by_identifier("cas", "9999-99-9") is None

    def test_search_multi_condition(self):
        """多条件组合搜索."""
        store = EntityStore()
        e1 = make_entity(name="水", entity_type=EntityType.CHEMICAL_COMPOUND, domain="chemistry", tags=["liquid"])
        e2 = make_entity(name="铁", entity_type=EntityType.MATERIAL, domain="materials", tags=["solid"])
        e3 = make_entity(name="乙醇", entity_type=EntityType.CHEMICAL_COMPOUND, domain="chemistry", tags=["liquid"])
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_entity(e3)

        # 按类型 + 领域
        results = store.search(entity_type=EntityType.CHEMICAL_COMPOUND, domain="chemistry")
        assert len(results) == 2

        # 按标签
        results = store.search(tag="liquid")
        assert len(results) == 2

    def test_merge_entities(self):
        """实体合并."""
        store = EntityStore()
        source = make_entity(
            name="H2O",
            aliases=["water"],
            tags=["chemistry"],
            identifiers={"cas": "7732-18-5"},
            properties={"formula": "H2O"},
        )
        target = make_entity(
            name="水",
            aliases=["蒸馏水"],
            tags=["liquid"],
            identifiers={"inchi": "H2O/c1H2"},
            properties={"boiling_point": 100},
        )
        store.add_entity(source)
        store.add_entity(target)

        merged = store.merge_entities(source.entity_id, target.entity_id)

        assert store.count() == 1
        assert merged.name == "水"
        assert "H2O" in merged.aliases  # source.name 被加入别名
        assert "water" in merged.aliases
        assert "chemistry" in merged.tags
        assert "liquid" in merged.tags
        assert "cas" in merged.identifiers
        assert "inchi" in merged.identifiers
        assert "formula" in merged.properties
        assert "boiling_point" in merged.properties

    def test_bulk_add(self):
        """批量添加."""
        store = EntityStore()
        entities = [make_entity(name=f"实体{i}") for i in range(10)]
        success, skipped, ids = store.bulk_add(entities)

        assert success == 10
        assert skipped == 0
        assert len(ids) == 10

    def test_bulk_add_with_duplicates(self):
        """批量添加含重复."""
        store = EntityStore()
        e1 = make_entity(name="水", identifiers={"cas": "7732-18-5"})
        store.add_entity(e1)

        e2 = make_entity(name="Water", identifiers={"cas": "7732-18-5"})
        e3 = make_entity(name="铁")
        success, skipped, _ = store.bulk_add([e2, e3], skip_duplicates=True)

        assert success == 1
        assert skipped == 1

    def test_get_stats(self):
        """获取统计."""
        store = EntityStore()
        store.add_entity(make_entity(name="水", entity_type=EntityType.CHEMICAL_COMPOUND, domain="chemistry"))
        store.add_entity(make_entity(name="铁", entity_type=EntityType.MATERIAL, domain="materials"))

        stats = store.get_stats()
        assert stats["total_entities"] == 2
        assert "chemical_compound" in stats["by_type"]
        assert "chemistry" in stats["by_domain"]

    def test_clear(self):
        """清空存储."""
        store = EntityStore()
        store.add_entity(make_entity(name="水"))
        store.clear()
        assert store.count() == 0


# ============================================================
# TripleStore 测试
# ============================================================


class TestTripleStore:
    """三元组存储测试."""

    def test_add_and_get_triple(self):
        """添加和获取三元组."""
        store = TripleStore()
        triple = make_triple("e-001", "related_to", "e-002")
        store.add_triple(triple)

        assert store.count() == 1
        assert store.get_triple(triple.triple_id) is triple

    def test_remove_triple(self):
        """移除三元组."""
        store = TripleStore()
        triple = make_triple("e-001", "related_to", "e-002")
        store.add_triple(triple)

        removed = store.remove_triple(triple.triple_id)
        assert removed is not None
        assert store.count() == 0
        assert len(store.get_by_subject("e-001")) == 0

    def test_get_by_subject(self):
        """按主语查询."""
        store = TripleStore()
        store.add_triple(make_triple("e-001", "cites", "e-002"))
        store.add_triple(make_triple("e-001", "related_to", "e-003"))
        store.add_triple(make_triple("e-002", "cites", "e-003"))

        results = store.get_by_subject("e-001")
        assert len(results) == 2

    def test_get_by_object(self):
        """按宾语查询."""
        store = TripleStore()
        store.add_triple(make_triple("e-001", "cites", "e-003"))
        store.add_triple(make_triple("e-002", "cites", "e-003"))

        results = store.get_by_object("e-003")
        assert len(results) == 2

    def test_get_by_predicate(self):
        """按谓词查询."""
        store = TripleStore()
        store.add_triple(make_triple("e-001", "cites", "e-002"))
        store.add_triple(make_triple("e-003", "cites", "e-004"))
        store.add_triple(make_triple("e-001", "related_to", "e-003"))

        results = store.get_by_predicate("cites")
        assert len(results) == 2

    def test_get_by_subject_predicate(self):
        """复合索引查询."""
        store = TripleStore()
        store.add_triple(make_triple("e-001", "cites", "e-002"))
        store.add_triple(make_triple("e-001", "related_to", "e-003"))

        results = store.get_by_subject_predicate("e-001", "cites")
        assert len(results) == 1
        assert results[0].object_id == "e-002"

    def test_get_outgoing_with_filter(self):
        """出边过滤 (置信度/排名)."""
        store = TripleStore()
        store.add_triple(make_triple("e-001", "cites", "e-002", confidence=0.9))
        store.add_triple(make_triple("e-001", "cites", "e-003", confidence=0.3))
        store.add_triple(make_triple("e-001", "related_to", "e-004", confidence=1.0, rank=StatementRank.DEPRECATED))

        # 置信度过滤
        results = store.get_outgoing("e-001", min_confidence=0.5)
        assert len(results) == 1

        # 排除已弃用
        results = store.get_outgoing("e-001", exclude_deprecated=True)
        assert len(results) == 2

        # 仅首选
        store.add_triple(make_triple("e-001", "supports", "e-005", rank=StatementRank.PREFERRED))
        results = store.get_outgoing("e-001", only_preferred=True)
        assert len(results) == 1

    def test_get_neighbors(self):
        """邻居查询."""
        store = TripleStore()
        store.add_triple(make_triple("e-001", "cites", "e-002"))
        store.add_triple(make_triple("e-001", "related_to", "e-003"))
        store.add_triple(make_triple("e-004", "cites", "e-001"))

        neighbors = store.get_neighbors("e-001", direction="both")
        assert set(neighbors) == {"e-002", "e-003", "e-004"}

        out_neighbors = store.get_neighbors("e-001", direction="out")
        assert set(out_neighbors) == {"e-002", "e-003"}

        in_neighbors = store.get_neighbors("e-001", direction="in")
        assert set(in_neighbors) == {"e-004"}

    def test_get_path(self):
        """路径查找."""
        store = TripleStore()
        store.add_triple(make_triple("A", "related_to", "B"))
        store.add_triple(make_triple("B", "related_to", "C"))
        store.add_triple(make_triple("C", "related_to", "D"))

        path = store.get_path("A", "D", max_depth=5)
        assert path is not None
        assert path == ["A", "B", "C", "D"]

        # 无路径
        path = store.get_path("A", "Z", max_depth=3)
        assert path is None

    def test_traverse_bfs(self):
        """BFS 遍历."""
        store = TripleStore()
        store.add_triple(make_triple("A", "related_to", "B"))
        store.add_triple(make_triple("B", "related_to", "C"))
        store.add_triple(make_triple("C", "related_to", "D"))
        store.add_triple(make_triple("A", "related_to", "E"))

        entities, triples = store.traverse_bfs("A", max_depth=2)
        assert "A" in entities
        assert "B" in entities
        assert "C" in entities
        assert "E" in entities
        assert len(triples) >= 3

    def test_temporal_validity(self):
        """时间有效性过滤."""
        store = TripleStore()
        now = time.time()

        # 有效三元组
        t1 = make_triple("e-001", "has_property", "e-002", valid_from=now - 100, valid_until=0)
        # 过期三元组
        t2 = make_triple("e-001", "has_property", "e-003", valid_from=now - 200, valid_until=now - 50)
        # 未生效三元组
        t3 = make_triple("e-001", "has_property", "e-004", valid_from=now + 100, valid_until=0)

        store.add_triple(t1)
        store.add_triple(t2)
        store.add_triple(t3)

        # 当前有效
        results = store.get_outgoing("e-001", valid_at=now)
        assert len(results) == 1
        assert results[0].object_id == "e-002"

    def test_get_stats(self):
        """获取统计."""
        store = TripleStore()
        store.add_triple(make_triple("e-001", "cites", "e-002", confidence=0.9))
        store.add_triple(make_triple("e-003", "cites", "e-004", confidence=0.7, rank=StatementRank.PREFERRED))

        stats = store.get_stats()
        assert stats["total_triples"] == 2
        assert stats["by_predicate"]["cites"] == 2
        assert stats["preferred_count"] == 1


# ============================================================
# ChunkStore 测试
# ============================================================


class TestChunkStore:
    """切片存储测试."""

    def test_add_and_get_chunk(self):
        """添加和获取切片."""
        store = ChunkStore()
        chunk = make_chunk(content="化学反应是物质转化的过程")
        store.add_chunk(chunk)

        assert store.count() == 1
        assert store.get_chunk(chunk.chunk_id) is chunk

    def test_add_duplicate_raises(self):
        """重复切片抛异常."""
        store = ChunkStore()
        chunk = make_chunk(content="测试")
        store.add_chunk(chunk)

        with pytest.raises(DuplicateEntityError):
            store.add_chunk(chunk)

    def test_remove_chunk(self):
        """移除切片."""
        store = ChunkStore()
        chunk = make_chunk(content="测试")
        store.add_chunk(chunk)

        removed = store.remove_chunk(chunk.chunk_id)
        assert removed is not None
        assert store.count() == 0

    def test_get_by_document(self):
        """按文档获取切片."""
        store = ChunkStore()
        store.add_chunk(make_chunk(content="段落1", document_id="doc-001", chunk_index=0))
        store.add_chunk(make_chunk(content="段落2", document_id="doc-001", chunk_index=1))
        store.add_chunk(make_chunk(content="段落A", document_id="doc-002", chunk_index=0))

        chunks = store.get_by_document("doc-001")
        assert len(chunks) == 2
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

    def test_search_text(self):
        """全文检索."""
        store = ChunkStore()
        store.add_chunk(make_chunk(content="水的化学式是H2O", document_id="doc-001"))
        store.add_chunk(make_chunk(content="铁是一种金属元素", document_id="doc-002"))
        store.add_chunk(make_chunk(content="化学反应涉及电子转移", document_id="doc-003"))

        results = store.search_text("化学", top_k=5)
        assert len(results) > 0
        # 化学相关的切片应该排在前面
        chunk_ids = [r[0].chunk_id for r in results]
        assert any("化学" in r[0].content for r in results)

    def test_search_text_scoped(self):
        """限定文档范围的全文检索."""
        store = ChunkStore()
        store.add_chunk(make_chunk(content="化学反应重要", document_id="doc-001"))
        store.add_chunk(make_chunk(content="化学反应机理", document_id="doc-002"))

        results = store.search_text("化学", top_k=10, document_id="doc-001")
        assert all(r[0].document_id == "doc-001" for r in results)

    def test_add_embedding(self):
        """添加向量."""
        store = ChunkStore()
        chunk = make_chunk(content="测试向量")
        store.add_chunk(chunk)

        vector = [0.1, 0.2, 0.3, 0.4]
        store.add_embedding(chunk.chunk_id, vector, model="test-model")

        assert store.has_embedding(chunk.chunk_id)
        assert store.indexed_vector_count() == 1

    def test_add_embedding_nonexistent(self):
        """为不存在的切片添加向量抛异常."""
        store = ChunkStore()
        with pytest.raises(EntityNotFoundError):
            store.add_embedding("nonexistent", [0.1, 0.2])

    def test_search_vector(self):
        """向量检索."""
        store = ChunkStore()
        store.add_chunk(make_chunk(content="内容A", document_id="doc-001"))
        store.add_chunk(make_chunk(content="内容B", document_id="doc-002"))

        # 添加向量
        chunks = list(store._chunks.values())
        store.add_embedding(chunks[0].chunk_id, [1.0, 0.0, 0.0])
        store.add_embedding(chunks[1].chunk_id, [0.0, 1.0, 0.0])

        # 搜索
        results = store.search_vector([1.0, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        # 第一个结果应该与查询向量最相似
        assert results[0][1] > results[1][1]

    def test_remove_document(self):
        """移除文档的所有切片."""
        store = ChunkStore()
        store.add_chunk(make_chunk(content="段落1", document_id="doc-001"))
        store.add_chunk(make_chunk(content="段落2", document_id="doc-001"))
        store.add_chunk(make_chunk(content="段落A", document_id="doc-002"))

        count = store.remove_document("doc-001")
        assert count == 2
        assert store.count() == 1

    def test_get_stats(self):
        """获取统计."""
        store = ChunkStore()
        store.add_chunk(make_chunk(content="测试内容A", document_id="doc-001"))
        store.add_chunk(make_chunk(content="测试内容B", document_id="doc-001"))

        stats = store.get_stats()
        assert stats["total_chunks"] == 2
        assert stats["total_documents"] == 1

    def test_clear(self):
        """清空存储."""
        store = ChunkStore()
        store.add_chunk(make_chunk(content="测试"))
        store.clear()
        assert store.count() == 0


# ============================================================
# KnowledgeStore 测试
# ============================================================


class TestKnowledgeStore:
    """统一知识存储测试."""

    def test_add_entity_with_version(self):
        """添加实体自动创建版本."""
        store = KnowledgeStore()
        entity = make_entity(name="水")
        store.add_entity(entity)

        history = store.get_version_history(entity.entity_id)
        assert len(history) == 1
        assert history[0].revision_number == 1

    def test_update_entity_tracks_version(self):
        """更新实体记录版本."""
        store = KnowledgeStore()
        entity = make_entity(name="水", description="液体")
        store.add_entity(entity)

        store.update_entity(entity.entity_id, description="无色液体", reason="补充描述")

        history = store.get_version_history(entity.entity_id)
        assert len(history) == 2
        assert history[1].revision_number == 2
        assert len(history[1].changeset) > 0

    def test_get_current_version(self):
        """获取当前版本."""
        store = KnowledgeStore()
        entity = make_entity(name="水")
        store.add_entity(entity)
        store.update_entity(entity.entity_id, description="更新")

        version = store.get_current_version(entity.entity_id)
        assert version is not None
        assert version.is_current()

    def test_restore_version(self):
        """版本恢复."""
        store = KnowledgeStore()
        entity = make_entity(name="水", description="原始描述")
        store.add_entity(entity)
        original_desc = entity.description

        store.update_entity(entity.entity_id, description="新描述")

        # 恢复到第一个版本
        history = store.get_version_history(entity.entity_id)
        store.restore_version(entity.entity_id, history[0].version_id)

        restored = store.get_entity(entity.entity_id)
        assert restored.description == original_desc

    def test_remove_entity_cleans_triples(self):
        """移除实体同时清理关联三元组."""
        store = KnowledgeStore()
        e1 = make_entity(name="水")
        e2 = make_entity(name="铁")
        store.add_entity(e1)
        store.add_entity(e2)

        t1 = make_triple(e1.entity_id, "related_to", e2.entity_id)
        store.add_triple(t1)

        store.remove_entity(e1.entity_id)

        assert store.entity_count() == 1
        assert store.triple_count() == 0

    def test_conflict_management(self):
        """冲突管理."""
        store = KnowledgeStore()
        entity = make_entity(name="水")
        store.add_entity(entity)

        conflict = KnowledgeConflict(
            conflict_type=ConflictType.SOURCE_BASED,
            entity_id=entity.entity_id,
            field_path="properties.boiling_point",
            conflicting_values=[
                {"value": 100, "source": "NIST"},
                {"value": 99.8, "source": "other"},
            ],
        )
        store.add_conflict(conflict)

        assert store.conflict_count == 1
        assert store.unresolved_conflict_count == 1

        # 解决冲突
        store.resolve_conflict(
            conflict.conflict_id,
            value=100,
            claim_id="claim-001",
            explanation="NIST 更可信",
        )

        assert store.unresolved_conflict_count == 0
        resolved = store.get_conflict(conflict.conflict_id)
        assert resolved.is_resolved()

    def test_evidence_management(self):
        """证据管理."""
        store = KnowledgeStore()
        entity = make_entity(name="水")
        store.add_entity(entity)

        evidence = EvidenceRecord(
            entity_id=entity.entity_id,
            source_reference="NIST Chemistry WebBook",
            source_content="水的沸点在1atm下为100°C",
            confidence=0.95,
        )
        store.add_evidence(evidence)

        assert store.evidence_count == 1
        evidence_list = store.get_evidence_for_entity(entity.entity_id)
        assert len(evidence_list) == 1

    def test_extract_subgraph(self):
        """子图提取."""
        store = KnowledgeStore()
        e1 = make_entity(name="水")
        e2 = make_entity(name="氢气")
        e3 = make_entity(name="氧气")
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_entity(e3)

        store.add_triple(make_triple(e1.entity_id, "derived_from", e2.entity_id))
        store.add_triple(make_triple(e1.entity_id, "derived_from", e3.entity_id))

        config = SubgraphConfig(
            entity_focus=e1.entity_id,
            max_depth=2,
            max_entities=10,
        )
        subgraph = store.extract_subgraph(config)

        assert subgraph.entity_count() >= 1
        assert subgraph.triple_count() >= 2

    def test_structured_query(self):
        """结构化查询."""
        store = KnowledgeStore()
        store.add_entity(make_entity(name="水", entity_type=EntityType.CHEMICAL_COMPOUND, domain="chemistry"))
        store.add_entity(make_entity(name="铁", entity_type=EntityType.MATERIAL, domain="materials"))

        query = KnowledgeQuery(
            domain="chemistry",
            conditions=[
                QueryCondition(field="entity_type", operator=QueryOperator.EQ, value="chemical_compound"),
            ],
        )
        results = store.query(query)
        assert len(results) == 1
        assert results[0].name == "水"

    def test_ingest(self):
        """批量导入."""
        store = KnowledgeStore()
        entities = [make_entity(name=f"实体{i}") for i in range(5)]
        triples = [make_triple(f"e-{i:03d}", "related_to", f"e-{i+1:03d}") for i in range(3)]

        result = store.ingest(entities=entities, triples=triples, source="test")

        assert result.total == 8
        assert result.success == 8
        assert result.failed == 0
        assert result.is_full_success()

    def test_ingest_with_duplicates(self):
        """批量导入含重复."""
        store = KnowledgeStore()
        e1 = make_entity(name="水", identifiers={"cas": "7732-18-5"})
        store.add_entity(e1)

        e2 = make_entity(name="Water", identifiers={"cas": "7732-18-5"})
        result = store.ingest(entities=[e2], source="test")

        assert result.total == 1
        assert result.skipped == 1

    def test_get_stats(self):
        """获取统计."""
        store = KnowledgeStore()
        store.add_entity(make_entity(name="水"))
        store.add_chunk(make_chunk(content="测试切片"))

        stats = store.get_stats()
        assert stats.total_entities == 1
        assert stats.total_chunks == 1

    def test_get_detailed_stats(self):
        """获取详细统计."""
        store = KnowledgeStore()
        entity = make_entity(name="水")
        store.add_entity(entity)
        store.update_entity(entity.entity_id, description="更新")

        stats = store.get_detailed_stats()
        assert "entities" in stats
        assert "triples" in stats
        assert "chunks" in stats
        assert "versions" in stats
        assert "conflicts" in stats

    def test_clear(self):
        """清空所有存储."""
        store = KnowledgeStore()
        store.add_entity(make_entity(name="水"))
        store.add_triple(make_triple("e-001", "related_to", "e-002"))
        store.add_chunk(make_chunk(content="测试"))

        store.clear()
        assert store.entity_count() == 0
        assert store.triple_count() == 0
        assert store.chunk_count() == 0

    def test_filter_entities(self):
        """使用 RetrievalFilter 过滤实体."""
        store = KnowledgeStore()
        store.add_entity(make_entity(name="水", domain="chemistry"))
        store.add_entity(make_entity(name="铁", domain="materials"))

        filt = RetrievalFilter(domain="chemistry")
        results = store.filter_entities(filt)
        assert len(results) == 1
        assert results[0].name == "水"


# ============================================================
# 检索引擎测试
# ============================================================


class TestVectorRetriever:
    """向量检索器测试."""

    def test_retrieve_without_vector_raises(self):
        """未提供向量抛异常."""
        store = KnowledgeStore()
        retriever = VectorRetriever(store)

        from dy3_polaris.l3.exceptions import RetrievalError
        with pytest.raises(RetrievalError):
            retriever.retrieve("test", query_vector=None)

    def test_retrieve_with_vector(self):
        """向量检索."""
        store = KnowledgeStore()
        chunk1 = make_chunk(content="向量测试A", document_id="doc-001")
        chunk2 = make_chunk(content="向量测试B", document_id="doc-002")
        store.add_chunk(chunk1)
        store.add_chunk(chunk2)

        store.chunk_store.add_embedding(chunk1.chunk_id, [1.0, 0.0, 0.0])
        store.chunk_store.add_embedding(chunk2.chunk_id, [0.0, 1.0, 0.0])

        retriever = VectorRetriever(store)
        result = retriever.retrieve("测试", query_vector=[1.0, 0.1, 0.0], top_k=2)

        assert isinstance(result, RetrievalResult)
        assert result.source_type == "vector"
        assert len(result.results) == 2
        assert result.retrieval_time_ms >= 0

    def test_retrieve_with_filter(self):
        """带过滤的向量检索."""
        store = KnowledgeStore()
        from dy3_polaris.l3 import ContentModality

        chunk1 = make_chunk(content="文本A", document_id="doc-001", content_type=ContentModality.TEXT)
        chunk2 = make_chunk(content="表格B", document_id="doc-002", content_type=ContentModality.TABLE)
        store.add_chunk(chunk1)
        store.add_chunk(chunk2)

        store.chunk_store.add_embedding(chunk1.chunk_id, [1.0, 0.0])
        store.chunk_store.add_embedding(chunk2.chunk_id, [0.9, 0.1])

        retriever = VectorRetriever(store)
        filt = RetrievalFilter(content_types=[ContentModality.TEXT])
        result = retriever.retrieve("测试", query_vector=[1.0, 0.0], top_k=5, filter=filt)

        assert all(r.get("content_type") == "text" for r in result.results)


class TestKeywordRetriever:
    """关键词检索器测试."""

    def test_retrieve_basic(self):
        """基础关键词检索."""
        store = KnowledgeStore()
        store.add_chunk(make_chunk(content="水的化学式是H2O，由氢和氧组成", document_id="doc-001"))
        store.add_chunk(make_chunk(content="铁是一种常见的金属元素", document_id="doc-002"))
        store.add_chunk(make_chunk(content="化学反应涉及电子的转移过程", document_id="doc-003"))

        retriever = KeywordRetriever(store)
        result = retriever.retrieve("化学", top_k=5)

        assert result.source_type == "keyword"
        assert len(result.results) > 0
        assert result.best_score() > 0

    def test_retrieve_empty_query(self):
        """空查询返回空结果."""
        store = KnowledgeStore()
        store.add_chunk(make_chunk(content="测试内容"))

        retriever = KeywordRetriever(store)
        result = retriever.retrieve("", top_k=5)
        assert result.is_empty()

    def test_retrieve_entities(self):
        """实体关键词检索."""
        store = KnowledgeStore()
        store.add_entity(make_entity(name="水", description="常见的化学物质", aliases=["H2O"]))
        store.add_entity(make_entity(name="铁", description="金属元素"))

        retriever = KeywordRetriever(store)
        results = retriever.retrieve_entities("水", top_k=5)

        assert len(results) > 0
        assert results[0][0].name == "水"


class TestGraphRetriever:
    """图检索器测试."""

    def test_retrieve_without_entity_raises(self):
        """未提供 entity_id 抛异常."""
        store = KnowledgeStore()
        retriever = GraphRetriever(store)

        from dy3_polaris.l3.exceptions import RetrievalError
        with pytest.raises(RetrievalError):
            retriever.retrieve("test", entity_id=None)

    def test_retrieve_with_entity(self):
        """图检索."""
        store = KnowledgeStore()
        e1 = make_entity(name="水")
        e2 = make_entity(name="氢气")
        e3 = make_entity(name="氧气")
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_entity(e3)

        store.add_triple(make_triple(e1.entity_id, "derived_from", e2.entity_id, confidence=0.9))
        store.add_triple(make_triple(e1.entity_id, "derived_from", e3.entity_id, confidence=0.8))

        retriever = GraphRetriever(store)
        result = retriever.retrieve(
            "水的组成",
            entity_id=e1.entity_id,
            max_depth=2,
            min_confidence=0.5,
        )

        assert result.source_type == "graph"
        assert len(result.results) > 0
        # 起始实体应该在结果中
        assert any(r.get("is_focus") for r in result.results)

    def test_find_path(self):
        """路径查找."""
        store = KnowledgeStore()
        e1 = make_entity(name="A")
        e2 = make_entity(name="B")
        e3 = make_entity(name="C")
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_entity(e3)

        store.add_triple(make_triple(e1.entity_id, "related_to", e2.entity_id))
        store.add_triple(make_triple(e2.entity_id, "related_to", e3.entity_id))

        retriever = GraphRetriever(store)
        result = retriever.find_path(e1.entity_id, e3.entity_id, max_depth=5)

        assert not result.is_empty()
        assert len(result.results) >= 3  # A, B, C 三个实体

    def test_get_neighbors(self):
        """邻居查询."""
        store = KnowledgeStore()
        e1 = make_entity(name="水")
        e2 = make_entity(name="氢气")
        e3 = make_entity(name="氧气")
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_entity(e3)

        store.add_triple(make_triple(e1.entity_id, "derived_from", e2.entity_id))
        store.add_triple(make_triple(e1.entity_id, "derived_from", e3.entity_id))

        retriever = GraphRetriever(store)
        result = retriever.get_neighbors(e1.entity_id, top_k=10)

        assert not result.is_empty()
        assert len(result.results) == 2


class TestHybridRetriever:
    """混合检索器测试."""

    def test_retrieve_rrf(self):
        """RRF 融合检索."""
        store = KnowledgeStore()
        store.add_chunk(make_chunk(content="化学反应原理", document_id="doc-001"))
        store.add_chunk(make_chunk(content="物理定律", document_id="doc-002"))

        # 添加向量
        chunks = list(store.chunk_store._chunks.values())
        store.chunk_store.add_embedding(chunks[0].chunk_id, [1.0, 0.0])
        store.chunk_store.add_embedding(chunks[1].chunk_id, [0.0, 1.0])

        retriever = HybridRetriever(store, fusion_strategy="rrf")
        result = retriever.retrieve(
            "化学",
            query_vector=[1.0, 0.1],
            top_k=5,
        )

        assert result.source_type == "hybrid"
        assert len(result.results) > 0

    def test_retrieve_weighted(self):
        """加权融合检索."""
        store = KnowledgeStore()
        store.add_chunk(make_chunk(content="化学反应原理", document_id="doc-001"))

        chunks = list(store.chunk_store._chunks.values())
        store.chunk_store.add_embedding(chunks[0].chunk_id, [1.0, 0.0])

        retriever = HybridRetriever(
            store,
            fusion_strategy="weighted",
            weights={"vector": 0.5, "keyword": 0.5},
        )
        result = retriever.retrieve("化学", query_vector=[1.0, 0.0], top_k=5)

        assert result.source_type == "hybrid"

    def test_retrieve_interleave(self):
        """交替合并检索."""
        store = KnowledgeStore()
        store.add_chunk(make_chunk(content="化学反应原理A", document_id="doc-001"))
        store.add_chunk(make_chunk(content="化学反应原理B", document_id="doc-002"))

        chunks = list(store.chunk_store._chunks.values())
        store.chunk_store.add_embedding(chunks[0].chunk_id, [1.0, 0.0])
        store.chunk_store.add_embedding(chunks[1].chunk_id, [0.9, 0.1])

        retriever = HybridRetriever(store, fusion_strategy="interleave")
        result = retriever.retrieve("化学", query_vector=[1.0, 0.0], top_k=5)

        assert result.source_type == "hybrid"

    def test_retrieve_keyword_only(self):
        """仅关键词检索."""
        store = KnowledgeStore()
        store.add_chunk(make_chunk(content="化学反应原理", document_id="doc-001"))

        retriever = HybridRetriever(store)
        result = retriever.retrieve("化学", top_k=5, retrievers=["keyword"])

        assert len(result.results) > 0

    def test_retrieve_no_results(self):
        """无结果的混合检索."""
        store = KnowledgeStore()
        retriever = HybridRetriever(store)
        result = retriever.retrieve("不存在的查询", top_k=5)

        assert result.is_empty()


class TestRetrievalEngine:
    """检索引擎门面测试."""

    def test_keyword_search(self):
        """关键词搜索."""
        store = KnowledgeStore()
        store.add_chunk(make_chunk(content="化学反应是物质转化的过程", document_id="doc-001"))

        engine = RetrievalEngine(store)
        result = engine.keyword_search("化学", top_k=5)

        assert isinstance(result, RetrievalResult)
        assert result.source_type == "keyword"

    def test_vector_search(self):
        """向量搜索."""
        store = KnowledgeStore()
        chunk = make_chunk(content="向量测试", document_id="doc-001")
        store.add_chunk(chunk)
        store.chunk_store.add_embedding(chunk.chunk_id, [1.0, 0.0])

        engine = RetrievalEngine(store)
        result = engine.vector_search([1.0, 0.0], query="测试", top_k=5)

        assert result.source_type == "vector"

    def test_graph_search(self):
        """图搜索."""
        store = KnowledgeStore()
        e1 = make_entity(name="水")
        e2 = make_entity(name="氢气")
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_triple(make_triple(e1.entity_id, "derived_from", e2.entity_id))

        engine = RetrievalEngine(store)
        result = engine.graph_search(e1.entity_id, max_depth=2, min_confidence=0.0)

        assert result.source_type == "graph"

    def test_hybrid_search(self):
        """混合搜索."""
        store = KnowledgeStore()
        chunk = make_chunk(content="化学反应原理", document_id="doc-001")
        store.add_chunk(chunk)
        store.chunk_store.add_embedding(chunk.chunk_id, [1.0, 0.0])

        e1 = make_entity(name="化学")
        store.add_entity(e1)

        engine = RetrievalEngine(store)
        result = engine.hybrid_search(
            "化学",
            query_vector=[1.0, 0.0],
            entity_id=e1.entity_id,
            top_k=5,
        )

        assert result.source_type == "hybrid"

    def test_get_neighbors(self):
        """邻居查询."""
        store = KnowledgeStore()
        e1 = make_entity(name="水")
        e2 = make_entity(name="氢气")
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_triple(make_triple(e1.entity_id, "related_to", e2.entity_id))

        engine = RetrievalEngine(store)
        result = engine.get_neighbors(e1.entity_id, min_confidence=0.0)

        assert not result.is_empty()


# ============================================================
# 线程安全测试
# ============================================================


class TestThreadSafety:
    """线程安全测试."""

    def test_concurrent_entity_add(self):
        """并发添加实体."""
        store = EntityStore()
        errors: list[Exception] = []

        def add_entities(start: int, count: int) -> None:
            try:
                for i in range(start, start + count):
                    store.add_entity(make_entity(name=f"实体{i}"))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_entities, args=(0, 50)),
            threading.Thread(target=add_entities, args=(50, 50)),
            threading.Thread(target=add_entities, args=(100, 50)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count() == 150

    def test_concurrent_triple_add(self):
        """并发添加三元组."""
        store = TripleStore()
        errors: list[Exception] = []

        def add_triples(start: int, count: int) -> None:
            try:
                for i in range(start, start + count):
                    store.add_triple(make_triple(f"e-{i:04d}", "related_to", f"e-{i+1:04d}"))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_triples, args=(0, 50)),
            threading.Thread(target=add_triples, args=(50, 50)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count() == 100


# ============================================================
# 集成测试
# ============================================================


class TestIntegration:
    """端到端集成测试."""

    def test_full_knowledge_pipeline(self):
        """完整知识处理流程."""
        store = KnowledgeStore()

        # 1. 创建知识实体
        water = make_entity(
            name="水",
            entity_type=EntityType.CHEMICAL_COMPOUND,
            domain="chemistry",
            identifiers={"cas": "7732-18-5"},
            aliases=["H2O", "water"],
            tags=["chemistry", "liquid", "essential"],
            properties={"formula": "H2O", "boiling_point": 100, "molecular_weight": 18.015},
        )
        hydrogen = make_entity(
            name="氢气",
            entity_type=EntityType.CHEMICAL_COMPOUND,
            domain="chemistry",
            identifiers={"cas": "1333-74-0"},
            aliases=["H2", "hydrogen"],
            properties={"formula": "H2", "molecular_weight": 2.016},
        )
        oxygen = make_entity(
            name="氧气",
            entity_type=EntityType.CHEMICAL_COMPOUND,
            domain="chemistry",
            identifiers={"cas": "7782-44-7"},
            aliases=["O2", "oxygen"],
            properties={"formula": "O2", "molecular_weight": 31.998},
        )

        store.add_entity(water)
        store.add_entity(hydrogen)
        store.add_entity(oxygen)

        # 2. 添加关系三元组
        store.add_triple(make_triple(
            water.entity_id, RelationType.DERIVED_FROM.value, hydrogen.entity_id,
            confidence=0.99,
        ))
        store.add_triple(make_triple(
            water.entity_id, RelationType.DERIVED_FROM.value, oxygen.entity_id,
            confidence=0.99,
        ))

        # 3. 添加文档切片
        chunk = make_chunk(
            content="水是由氢和氧组成的化合物，化学式为H2O。在标准大气压下，水的沸点为100摄氏度。",
            document_id=water.entity_id,
        )
        store.add_chunk(chunk)

        # 4. 添加向量
        store.chunk_store.add_embedding(chunk.chunk_id, [0.8, 0.2, 0.1], model="test")

        # 5. 添加证据
        evidence = EvidenceRecord(
            entity_id=water.entity_id,
            source_reference="NIST Chemistry WebBook",
            source_content="Water: H2O, MW=18.015, BP=100°C",
            confidence=0.98,
        )
        store.add_evidence(evidence)

        # 6. 验证存储
        assert store.entity_count() == 3
        assert store.triple_count() == 2
        assert store.chunk_count() == 1
        assert store.evidence_count == 1

        # 7. 检索测试
        engine = RetrievalEngine(store)

        # 关键词检索
        kw_result = engine.keyword_search("水 化合物", top_k=5)
        assert not kw_result.is_empty()

        # 向量检索
        vec_result = engine.vector_search([0.8, 0.2, 0.1], query="水", top_k=5)
        assert not vec_result.is_empty()

        # 图检索
        graph_result = engine.graph_search(
            water.entity_id, max_depth=2, min_confidence=0.0, top_k=10
        )
        assert not graph_result.is_empty()
        # 应该能找到氢气和氧气
        entity_names = [r.get("name") for r in graph_result.results if r.get("type") == "entity"]
        assert "氢气" in entity_names or "氧气" in entity_names

        # 8. 版本管理
        store.update_entity(water.entity_id, description="水是最常见的化合物")
        history = store.get_version_history(water.entity_id)
        assert len(history) == 2

        # 9. 统计
        stats = store.get_detailed_stats()
        assert stats["entities"]["total_entities"] == 3
        assert stats["triples"]["total_triples"] == 2

    def test_large_scale_ingest(self):
        """大规模导入测试."""
        store = KnowledgeStore()

        entities = [
            make_entity(name=f"化合物_{i}", entity_type=EntityType.CHEMICAL_COMPOUND)
            for i in range(100)
        ]
        chunks = [
            make_chunk(content=f"这是第{i}个文档的内容，描述了化学性质", document_id=f"doc-{i:03d}")
            for i in range(50)
        ]

        result = store.ingest(entities=entities, chunks=chunks, source="bulk_test")

        assert result.success == 150
        assert result.failed == 0
        assert store.entity_count() == 100
        assert store.chunk_count() == 50
