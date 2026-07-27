"""L3 持久化层、事务管理器和重排器全面测试套件.

覆盖范围:
- PersistenceManager: 快照保存/加载、增量日志、JSON/JSON-LD 导出导入、
  完整性校验、压缩合并
- Transaction: 提交、回滚、上下文管理器、保存点、undo log 回放
- TransactionManager: 事务生命周期管理、活跃事务追踪
- MMRReranker: 最大边际相关性、向量/Jaccard 多样性、lambda 参数
- MetadataBoostReranker: 类型/验证/置信度/标签加权
- QualityBoostReranker: 多维质量分数加权
- RecencyBoostReranker: 指数/高斯/线性时间衰减
- GraphCentralityReranker: 度中心性、图邻近性、社区归属
- CompositeReranker: 多阶段管道、阶段截断
- RetrievalEngine 重排集成: 关键词检索+重排、动态设置重排器
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time

import pytest

from dy3_polaris.l3.persistence import (
    PersistenceManager,
    Transaction,
    TransactionManager,
    TransactionState,
)
from dy3_polaris.l3.reranker import (
    RerankStrategy,
    BaseReranker,
    MMRReranker,
    MetadataBoostReranker,
    QualityBoostReranker,
    RecencyBoostReranker,
    GraphCentralityReranker,
    CompositeReranker,
)
from dy3_polaris.l3.models import (
    KnowledgeEntity,
    EntityType,
    KnowledgeTriple,
    DocumentChunk,
    RetrievalResult,
    QualityScore,
    QualityDimension,
    KnowledgeStatus,
    AccessLevel,
    RelationType,
)
from dy3_polaris.l3.store import KnowledgeStore
from dy3_polaris.l3.retrieval import RetrievalEngine
from dy3_polaris.l3.exceptions import IngestError, RetrievalError, L3Error


# ============================================================
# Helper 函数
# ============================================================


def make_entity(name="测试实体", entity_type=EntityType.CONCEPT, **kwargs):
    """创建测试实体."""
    return KnowledgeEntity(entity_type=entity_type, name=name, **kwargs)


def make_triple(subject_id="e-001", predicate="related_to", object_id="e-002", **kwargs):
    """创建测试三元组."""
    return KnowledgeTriple(
        subject_id=subject_id, predicate=predicate, object_id=object_id, **kwargs
    )


def make_chunk(content="测试内容", **kwargs):
    """创建测试切片."""
    kwargs.setdefault("document_id", "doc-001")
    return DocumentChunk(content=content, **kwargs)


def make_result(query="test", results=None, scores=None):
    """创建测试检索结果."""
    if results is None:
        results = []
    if scores is None:
        scores = [1.0 / (i + 1) for i in range(len(results))]
    return RetrievalResult(
        query=query, results=results, scores=scores, total=len(results)
    )


def make_entity_doc(name="测试实体", entity_type=EntityType.CONCEPT, **kwargs):
    """创建测试实体字典 (用于重排器测试)."""
    entity = make_entity(name=name, entity_type=entity_type, **kwargs)
    return entity.model_dump(mode="json")


# ============================================================
# 持久化管理器测试
# ============================================================


class TestPersistenceManager:
    """PersistenceManager 持久化管理器测试."""

    def test_save_and_load_snapshot(self):
        """快照保存和加载: 保存后加载应恢复全部数据."""
        store = KnowledgeStore()
        entity1 = make_entity(name="实体A", entity_type=EntityType.CONCEPT)
        entity2 = make_entity(name="实体B", entity_type=EntityType.MATERIAL)
        store.add_entity(entity1)
        store.add_entity(entity2)
        triple = make_triple(
            subject_id=entity1.entity_id,
            predicate="related_to",
            object_id=entity2.entity_id,
        )
        store.add_triple(triple)
        chunk = make_chunk(content="持久化测试内容", document_id=entity1.entity_id)
        store.add_chunk(chunk)

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(store, tmpdir)
            snapshot_path = pm.save_snapshot()
            assert snapshot_path.exists()
            assert (snapshot_path / "manifest.json").exists()

            # 用新存储加载
            store2 = KnowledgeStore()
            pm2 = PersistenceManager(store2, tmpdir)
            pm2.load_snapshot(snapshot_path)

            assert store2.get_entity(entity1.entity_id) is not None
            assert store2.get_entity(entity2.entity_id) is not None
            assert store2.get_entity(entity1.entity_id).name == "实体A"
            assert store2.get_triple(triple.triple_id) is not None
            assert store2.get_chunk(chunk.chunk_id) is not None

    def test_snapshot_info(self):
        """快照信息: get_snapshot_info 应返回正确的元数据."""
        store = KnowledgeStore()
        entity = make_entity(name="信息测试实体")
        store.add_entity(entity)

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(store, tmpdir)
            snapshot_path = pm.save_snapshot()
            info = pm.get_snapshot_info(snapshot_path)

            assert info["format_version"] == "1.0"
            assert "created_at" in info
            assert "counts" in info
            assert info["counts"]["entities"] == 1
            assert "checksums" in info
            assert "manifest_checksum" in info
            assert "file_sizes" in info
            assert "total_size" in info
            assert info["path"] == str(snapshot_path)

    def test_verify_integrity(self):
        """校验和验证: 未损坏的快照应通过验证."""
        store = KnowledgeStore()
        entity = make_entity(name="完整性实体")
        store.add_entity(entity)

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(store, tmpdir)
            snapshot_path = pm.save_snapshot()
            assert pm.verify_integrity(snapshot_path) is True

    def test_verify_integrity_corrupted(self):
        """损坏文件验证失败: 篡改文件后验证应返回 False."""
        store = KnowledgeStore()
        entity = make_entity(name="损坏测试实体")
        store.add_entity(entity)

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(store, tmpdir)
            snapshot_path = pm.save_snapshot()

            # 篡改 entities.jsonl 文件
            entities_file = snapshot_path / "entities.jsonl"
            original = entities_file.read_text(encoding="utf-8")
            entities_file.write_text(original + "corrupted_data\n", encoding="utf-8")

            assert pm.verify_integrity(snapshot_path) is False

    def test_export_and_import_json(self):
        """JSON 导出导入: 导出后清空存储再导入应恢复数据."""
        store = KnowledgeStore()
        entity1 = make_entity(name="JSON实体1", entity_type=EntityType.CONCEPT)
        entity2 = make_entity(name="JSON实体2", entity_type=EntityType.PERSON)
        store.add_entity(entity1)
        store.add_entity(entity2)
        chunk = make_chunk(content="JSON导出测试")
        store.add_chunk(chunk)

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(store, tmpdir)
            json_path = os.path.join(tmpdir, "export.json")
            pm.export_json(json_path)
            assert os.path.exists(json_path)

            # 清空存储后导入
            store.clear()
            assert store.get_entity(entity1.entity_id) is None

            result = pm.import_json(json_path)
            assert result.success >= 2  # 至少导入 2 个实体
            assert store.get_entity(entity1.entity_id) is not None
            assert store.get_entity(entity1.entity_id).name == "JSON实体1"

    def test_export_and_import_jsonld(self):
        """JSON-LD 导出导入: 导出后清空存储再导入应恢复实体."""
        store = KnowledgeStore()
        entity = make_entity(
            name="JSON-LD实体",
            entity_type=EntityType.CHEMICAL_COMPOUND,
            description="测试化学物质",
            domain="chemistry",
            tags=["化学", "测试"],
            identifiers={"cas": "7732-18-5"},
        )
        store.add_entity(entity)

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(store, tmpdir)
            jsonld_path = os.path.join(tmpdir, "export.jsonld")
            pm.export_jsonld(jsonld_path)
            assert os.path.exists(jsonld_path)

            # 验证 JSON-LD 结构
            with open(jsonld_path, encoding="utf-8") as f:
                data = json.load(f)
            assert "@context" in data
            assert "@graph" in data
            assert len(data["@graph"]) >= 1

            # 清空存储后导入
            store.clear()
            assert store.get_entity(entity.entity_id) is None

            result = pm.import_jsonld(jsonld_path)
            assert result.success >= 1
            restored = store.get_entity(entity.entity_id)
            assert restored is not None
            assert restored.name == "JSON-LD实体"

    def test_save_and_load_incremental(self):
        """增量日志保存和加载: 保存变更后加载应返回相同变更记录."""
        store = KnowledgeStore()

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(store, tmpdir)
            changes = [
                {"op": "add_entity", "entity_id": "e-001", "name": "增量实体1"},
                {"op": "add_entity", "entity_id": "e-002", "name": "增量实体2"},
                {"op": "update_entity", "entity_id": "e-001", "field": "name", "value": "更新名称"},
            ]
            wal_path = pm.save_incremental(changes)
            assert wal_path.exists()

            loaded = pm.load_incremental(wal_path)
            assert len(loaded) == 3
            assert loaded[0]["op"] == "add_entity"
            assert loaded[0]["entity_id"] == "e-001"
            assert loaded[2]["op"] == "update_entity"

    def test_compact(self):
        """压缩快照: compact 应创建 zip 文件并清理增量日志."""
        store = KnowledgeStore()
        entity = make_entity(name="压缩测试实体")
        store.add_entity(entity)

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(store, tmpdir)

            # 先保存增量日志
            pm.save_incremental([{"op": "add_entity", "entity_id": "e-001"}])
            pm.save_incremental([{"op": "update_entity", "entity_id": "e-001"}])

            wal_files_before = list(os.path.dirname(str(tmpdir)).split())
            # 确认 WAL 文件存在
            wal_files = []
            for f in os.listdir(tmpdir):
                if f.startswith("wal_"):
                    wal_files.append(f)
            assert len(wal_files) >= 2

            # 执行压缩
            zip_path = pm.compact()
            assert zip_path.exists()
            assert str(zip_path).endswith(".zip")

            # WAL 文件应被清理
            wal_files_after = [
                f for f in os.listdir(tmpdir) if f.startswith("wal_")
            ]
            assert len(wal_files_after) == 0

    def test_load_nonexistent(self):
        """加载不存在的快照: 应抛出 IngestError."""
        store = KnowledgeStore()

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(store, tmpdir)
            nonexistent_path = os.path.join(tmpdir, "nonexistent_snapshot")
            with pytest.raises(IngestError):
                pm.load_snapshot(nonexistent_path)

    def test_empty_store_snapshot(self):
        """空存储快照: 空存储的快照应能正确保存和加载."""
        store = KnowledgeStore()

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(store, tmpdir)
            snapshot_path = pm.save_snapshot()
            assert snapshot_path.exists()

            # 加载到新存储
            store2 = KnowledgeStore()
            pm2 = PersistenceManager(store2, tmpdir)
            pm2.load_snapshot(snapshot_path)

            stats = store2.get_stats()
            assert stats.total_entities == 0
            assert stats.total_chunks == 0
            assert stats.total_triples == 0


# ============================================================
# 事务测试
# ============================================================


class TestTransaction:
    """Transaction 事务测试."""

    def test_commit(self):
        """提交事务: commit 后数据应持久化."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        entity = make_entity(name="提交测试实体")

        tx = txm.begin()
        tx.add_entity(entity)
        tx.commit()

        assert store.get_entity(entity.entity_id) is not None
        assert tx.state == TransactionState.COMMITTED

    def test_rollback(self):
        """回滚事务: rollback 后数据应被撤销."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        entity = make_entity(name="回滚测试实体")

        tx = txm.begin()
        tx.add_entity(entity)
        tx.rollback()

        assert store.get_entity(entity.entity_id) is None
        assert tx.state == TransactionState.ROLLED_BACK

    def test_context_manager_commit(self):
        """上下文管理器正常提交: with 块正常退出应自动提交."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        entity = make_entity(name="上下文提交实体")

        with txm.begin() as tx:
            tx.add_entity(entity)

        assert store.get_entity(entity.entity_id) is not None
        assert tx.state == TransactionState.COMMITTED

    def test_context_manager_rollback(self):
        """上下文管理器异常回滚: with 块异常退出应自动回滚."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        entity = make_entity(name="上下文回滚实体")

        with pytest.raises(ValueError):
            with txm.begin() as tx:
                tx.add_entity(entity)
                raise ValueError("测试异常")

        assert store.get_entity(entity.entity_id) is None
        assert tx.state == TransactionState.ROLLED_BACK

    def test_savepoint(self):
        """保存点: 创建保存点后可继续操作."""
        store = KnowledgeStore()
        txm = TransactionManager(store)

        tx = txm.begin()
        tx.savepoint("sp1")
        entity = make_entity(name="保存点测试实体")
        tx.add_entity(entity)
        # 保存点存在, 操作正常
        assert store.get_entity(entity.entity_id) is not None

    def test_rollback_to_savepoint(self):
        """回滚到保存点: 只撤销保存点之后的操作."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        entity1 = make_entity(name="保存点前实体")
        entity2 = make_entity(name="保存点后实体")

        tx = txm.begin()
        tx.add_entity(entity1)
        tx.savepoint("sp1")
        tx.add_entity(entity2)

        # 两个实体都存在
        assert store.get_entity(entity1.entity_id) is not None
        assert store.get_entity(entity2.entity_id) is not None

        tx.rollback_to_savepoint("sp1")

        # entity2 被撤销, entity1 保留
        assert store.get_entity(entity1.entity_id) is not None
        assert store.get_entity(entity2.entity_id) is None

        tx.commit()
        assert store.get_entity(entity1.entity_id) is not None

    def test_release_savepoint(self):
        """释放保存点: 释放后不能回滚到该保存点."""
        store = KnowledgeStore()
        txm = TransactionManager(store)

        tx = txm.begin()
        tx.savepoint("sp1")
        tx.release_savepoint("sp1")

        # 释放后回滚到该保存点应抛异常
        with pytest.raises(L3Error):
            tx.rollback_to_savepoint("sp1")

        tx.rollback()

    def test_add_entity_rollback(self):
        """添加实体后回滚: 回滚后实体应被移除."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        entity = make_entity(name="添加回滚实体")

        tx = txm.begin()
        tx.add_entity(entity)
        assert store.get_entity(entity.entity_id) is not None
        tx.rollback()
        assert store.get_entity(entity.entity_id) is None

    def test_update_entity_rollback(self):
        """更新实体后回滚: 回滚后实体应恢复原始值."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        entity = make_entity(name="原始名称", description="原始描述")
        store.add_entity(entity)
        original_name = entity.name

        tx = txm.begin()
        tx.update_entity(entity.entity_id, name="更新后名称")
        updated = store.get_entity(entity.entity_id)
        assert updated.name == "更新后名称"

        tx.rollback()
        restored = store.get_entity(entity.entity_id)
        assert restored.name == original_name

    def test_remove_entity_rollback(self):
        """删除实体后回滚: 回滚后实体应被恢复."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        entity = make_entity(name="删除回滚实体")
        store.add_entity(entity)

        tx = txm.begin()
        tx.remove_entity(entity.entity_id)
        assert store.get_entity(entity.entity_id) is None

        tx.rollback()
        assert store.get_entity(entity.entity_id) is not None
        assert store.get_entity(entity.entity_id).name == "删除回滚实体"

    def test_add_triple_rollback(self):
        """添加三元组后回滚: 回滚后三元组应被移除."""
        store = KnowledgeStore()
        entity1 = make_entity(name="主语实体")
        entity2 = make_entity(name="宾语实体")
        store.add_entity(entity1)
        store.add_entity(entity2)

        txm = TransactionManager(store)
        triple = make_triple(
            subject_id=entity1.entity_id,
            predicate="related_to",
            object_id=entity2.entity_id,
        )

        tx = txm.begin()
        tx.add_triple(triple)
        assert store.get_triple(triple.triple_id) is not None
        tx.rollback()
        assert store.get_triple(triple.triple_id) is None

    def test_add_chunk_rollback(self):
        """添加切片后回滚: 回滚后切片应被移除."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        chunk = make_chunk(content="切片回滚测试")

        tx = txm.begin()
        tx.add_chunk(chunk)
        assert store.get_chunk(chunk.chunk_id) is not None
        tx.rollback()
        assert store.get_chunk(chunk.chunk_id) is None

    def test_nested_savepoints(self):
        """嵌套保存点: 支持多层保存点的创建和回滚."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        entity1 = make_entity(name="第一层实体")
        entity2 = make_entity(name="第二层实体")
        entity3 = make_entity(name="第三层实体")

        tx = txm.begin()
        tx.add_entity(entity1)
        tx.savepoint("sp1")
        tx.add_entity(entity2)
        tx.savepoint("sp2")
        tx.add_entity(entity3)

        # 三个实体都存在
        assert store.get_entity(entity1.entity_id) is not None
        assert store.get_entity(entity2.entity_id) is not None
        assert store.get_entity(entity3.entity_id) is not None

        # 回滚到 sp2: 撤销 entity3
        tx.rollback_to_savepoint("sp2")
        assert store.get_entity(entity1.entity_id) is not None
        assert store.get_entity(entity2.entity_id) is not None
        assert store.get_entity(entity3.entity_id) is None

        # 回滚到 sp1: 撤销 entity2
        tx.rollback_to_savepoint("sp1")
        assert store.get_entity(entity1.entity_id) is not None
        assert store.get_entity(entity2.entity_id) is None

        tx.commit()
        assert store.get_entity(entity1.entity_id) is not None

    def test_transaction_state(self):
        """事务状态检查: 状态应正确转换."""
        store = KnowledgeStore()
        txm = TransactionManager(store)

        # ACTIVE -> COMMITTED
        tx1 = txm.begin()
        assert tx1.state == TransactionState.ACTIVE
        tx1.commit()
        assert tx1.state == TransactionState.COMMITTED

        # ACTIVE -> ROLLED_BACK
        tx2 = txm.begin()
        assert tx2.state == TransactionState.ACTIVE
        tx2.rollback()
        assert tx2.state == TransactionState.ROLLED_BACK

        # 已结束的事务不能再操作
        with pytest.raises(L3Error):
            tx1.commit()
        with pytest.raises(L3Error):
            tx2.rollback()


# ============================================================
# 事务管理器测试
# ============================================================


class TestTransactionManager:
    """TransactionManager 事务管理器测试."""

    def test_begin(self):
        """开始事务: begin 应返回活跃状态的事务."""
        store = KnowledgeStore()
        txm = TransactionManager(store)

        tx = txm.begin()
        assert isinstance(tx, Transaction)
        assert tx.state == TransactionState.ACTIVE
        assert len(tx.tx_id) > 0

    def test_active_count(self):
        """活跃事务数: 活跃事务计数应正确反映当前事务数."""
        store = KnowledgeStore()
        txm = TransactionManager(store)

        assert txm.active_count == 0
        tx1 = txm.begin()
        assert txm.active_count == 1
        tx2 = txm.begin()
        assert txm.active_count == 2

        tx1.commit()
        assert txm.active_count == 1
        tx2.rollback()
        assert txm.active_count == 0

    def test_get_active_transactions(self):
        """获取活跃事务列表: 应返回所有活跃事务的 ID."""
        store = KnowledgeStore()
        txm = TransactionManager(store)

        tx1 = txm.begin()
        tx2 = txm.begin()
        active = txm.get_active_transactions()
        assert len(active) == 2
        assert tx1.tx_id in active
        assert tx2.tx_id in active

        tx1.commit()
        active = txm.get_active_transactions()
        assert len(active) == 1
        assert tx1.tx_id not in active

    def test_multiple_transactions(self):
        """多事务并行: 多个事务可同时操作不同数据."""
        store = KnowledgeStore()
        txm = TransactionManager(store)
        entity1 = make_entity(name="事务1实体")
        entity2 = make_entity(name="事务2实体")

        tx1 = txm.begin()
        tx2 = txm.begin()

        tx1.add_entity(entity1)
        tx2.add_entity(entity2)

        # 两个实体都在存储中 (事务内可见)
        assert store.get_entity(entity1.entity_id) is not None
        assert store.get_entity(entity2.entity_id) is not None

        tx1.commit()
        tx2.commit()

        # 提交后两个实体都持久化
        assert store.get_entity(entity1.entity_id) is not None
        assert store.get_entity(entity2.entity_id) is not None
        assert txm.active_count == 0


# ============================================================
# MMR 重排器测试
# ============================================================


class TestMMRReranker:
    """MMRReranker 最大边际相关性重排器测试."""

    def test_rerank_basic(self):
        """基本重排: 多条结果重排后应返回 top_k 条."""
        reranker = MMRReranker(lambda_=0.7)
        results = [
            ({"name": "苹果", "description": "水果"}, 1.0),
            ({"name": "香蕉", "description": "水果"}, 0.8),
            ({"name": "汽车", "description": "交通工具"}, 0.6),
            ({"name": "电脑", "description": "电子设备"}, 0.4),
        ]
        reranked = reranker.rerank("水果", results, top_k=4)
        assert len(reranked) == 4
        # 第一条应是最相关的
        assert reranked[0][1] == 1.0

    def test_rerank_empty_results(self):
        """空结果重排: 输入为空时应返回空列表."""
        reranker = MMRReranker()
        reranked = reranker.rerank("test", [], top_k=10)
        assert reranked == []

    def test_rerank_single_result(self):
        """单条结果重排: 单条结果应原样返回."""
        reranker = MMRReranker()
        results = [({"name": "唯一结果"}, 1.0)]
        reranked = reranker.rerank("test", results, top_k=10)
        assert len(reranked) == 1
        assert reranked[0][1] == 1.0

    def test_diversity_with_vectors(self):
        """向量多样性: 相似向量应被分散, 不同向量优先选择."""
        reranker = MMRReranker(lambda_=0.5, use_attribute_overlap=False)
        # result0 和 result1 向量非常相似, result2 完全不同
        # B 的分数接近 C, 使归一化后多样性惩罚占主导
        results = [
            ({"name": "A", "embedding": {"vector": [1.0, 0.0]}}, 1.0),
            ({"name": "B", "embedding": {"vector": [0.99, 0.01]}}, 0.85),
            ({"name": "C", "embedding": {"vector": [0.0, 1.0]}}, 0.8),
        ]
        reranked = reranker.rerank("test", results, top_k=3)

        # 第一条应是最相关的 (score=1.0)
        assert reranked[0][0]["name"] == "A"
        # 第二条应是 C (与 A 不相似), 而非 B (与 A 非常相似)
        assert reranked[1][0]["name"] == "C"
        # 第三条是 B
        assert reranked[2][0]["name"] == "B"

    def test_diversity_without_vectors(self):
        """无向量时 Jaccard 相似度: 文本相似的结果应被分散."""
        reranker = MMRReranker(lambda_=0.3, use_attribute_overlap=False)
        # result0 和 result1 文本高度相似, result2 完全不同
        results = [
            ({"name": "化学", "description": "化学化学反应"}, 1.0),
            ({"name": "化学", "description": "化学化学反应"}, 0.9),
            ({"name": "物理", "description": "物理物理力学"}, 0.8),
        ]
        reranked = reranker.rerank("test", results, top_k=3)

        # 第一条是最相关的
        assert reranked[0][1] == 1.0
        # 第二条应与第一条不同 (物理而非重复的化学)
        assert reranked[1][0]["name"] == "物理"

    def test_lambda_parameter(self):
        """lambda 参数影响: lambda=1 不过滤多样性, lambda=0 最大多样性."""
        # 构造相似向量对
        results = [
            ({"name": "A", "embedding": {"vector": [1.0, 0.0]}}, 1.0),
            ({"name": "B", "embedding": {"vector": [0.99, 0.01]}}, 0.9),
            ({"name": "C", "embedding": {"vector": [0.0, 1.0]}}, 0.8),
        ]

        # lambda=1.0: 纯相关性, 按 score 降序
        reranker_rel = MMRReranker(lambda_=1.0, use_attribute_overlap=False)
        reranked_rel = reranker_rel.rerank("test", results, top_k=3)
        scores_rel = [s for _, s in reranked_rel]
        assert scores_rel == sorted(scores_rel, reverse=True)
        # A(1.0), B(0.9), C(0.8) 按分数排列
        assert reranked_rel[0][0]["name"] == "A"
        assert reranked_rel[1][0]["name"] == "B"
        assert reranked_rel[2][0]["name"] == "C"

        # lambda=0.0: 最大多样性, 相似项被分散
        reranker_div = MMRReranker(lambda_=0.0, use_attribute_overlap=False)
        reranked_div = reranker_div.rerank("test", results, top_k=3)
        # 第一条仍是 A (所有 max_sim=0, 取第一个)
        assert reranked_div[0][0]["name"] == "A"
        # 第二条应是 C (与 A 最不相似)
        assert reranked_div[1][0]["name"] == "C"

    def test_rerank_result(self):
        """rerank_result 方法返回 RetrievalResult."""
        reranker = MMRReranker(lambda_=0.7)
        result = make_result(
            query="test",
            results=[
                {"name": "结果1"},
                {"name": "结果2"},
                {"name": "结果3"},
            ],
            scores=[1.0, 0.8, 0.6],
        )
        reranked = reranker.rerank_result("test", result, top_k=3)
        assert isinstance(reranked, RetrievalResult)
        assert len(reranked.results) == 3
        assert len(reranked.scores) == 3
        assert "rerank:mmr" in reranked.source_type


# ============================================================
# 元数据加权重排器测试
# ============================================================


class TestMetadataBoostReranker:
    """MetadataBoostReranker 元数据加权重排器测试."""

    def test_type_boost(self):
        """实体类型加权: 高权重类型的实体应获得更高分数."""
        reranker = MetadataBoostReranker()
        results = [
            ({"name": "概念", "entity_type": "concept"}, 1.0),
            ({"name": "化合物", "entity_type": "chemical_compound"}, 1.0),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        # chemical_compound 权重 (0.30) 高于 concept (0.05)
        assert reranked[0][0]["entity_type"] == "chemical_compound"
        assert reranked[0][1] > reranked[1][1]

    def test_verified_boost(self):
        """已验证实体加权: is_verified=True 的实体应获得更高分数."""
        reranker = MetadataBoostReranker(verified_boost=0.5)
        results = [
            ({"name": "未验证", "entity_type": "concept", "is_verified": False}, 1.0),
            ({"name": "已验证", "entity_type": "concept", "is_verified": True}, 1.0),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        assert reranked[0][0]["name"] == "已验证"
        assert reranked[0][1] > reranked[1][1]

    def test_confidence_boost(self):
        """置信度加权: 高置信度的实体应获得更高分数."""
        reranker = MetadataBoostReranker(confidence_weight=0.5)
        results = [
            ({"name": "低置信度", "entity_type": "concept", "confidence_score": 0.2}, 1.0),
            ({"name": "高置信度", "entity_type": "concept", "confidence_score": 0.9}, 1.0),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        assert reranked[0][0]["name"] == "高置信度"
        assert reranked[0][1] > reranked[1][1]

    def test_tag_boost(self):
        """标签匹配加权: 标签与查询匹配的实体应获得更高分数."""
        reranker = MetadataBoostReranker(tag_boost=0.5)
        results = [
            ({"name": "无标签", "entity_type": "concept", "tags": []}, 1.0),
            ({"name": "有标签", "entity_type": "concept", "tags": ["化学", "test"]}, 1.0),
        ]
        reranked = reranker.rerank("化学 test", results, top_k=2)

        assert reranked[0][0]["name"] == "有标签"
        assert reranked[0][1] > reranked[1][1]

    def test_no_boost(self):
        """无加权因子时分数不变: 所有 boost 因子为 0 时分数应保持不变."""
        reranker = MetadataBoostReranker(
            type_weight={},
            domain_weight={},
            tag_boost=0.0,
            verified_boost=0.0,
            confidence_weight=0.0,
        )
        results = [
            ({"name": "A", "entity_type": "concept"}, 0.8),
            ({"name": "B", "entity_type": "material"}, 1.0),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        # 分数不变, 按原始分数降序
        assert reranked[0][1] == 1.0
        assert reranked[1][1] == 0.8


# ============================================================
# 质量加权重排器测试
# ============================================================


class TestQualityBoostReranker:
    """QualityBoostReranker 质量分数加权重排器测试."""

    def test_quality_boost(self):
        """质量分数加权: 高质量实体应获得更高分数."""
        reranker = QualityBoostReranker(quality_weight=0.5)
        high_quality = QualityScore(
            accuracy=0.95, trustworthiness=0.9, consistency=0.9,
            timeliness=0.9, completeness=0.85, relevancy=0.9,
        )
        low_quality = QualityScore(
            accuracy=0.3, trustworthiness=0.3, consistency=0.3,
            timeliness=0.3, completeness=0.3, relevancy=0.3,
        )
        results = [
            ({"name": "低质量", "quality": low_quality.to_dict()}, 1.0),
            ({"name": "高质量", "quality": high_quality.to_dict()}, 1.0),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        assert reranked[0][0]["name"] == "高质量"
        assert reranked[0][1] > reranked[1][1]

    def test_no_quality(self):
        """无质量分数时不变: 没有 quality 字段的实体分数不应改变."""
        reranker = QualityBoostReranker(quality_weight=0.3)
        results = [
            ({"name": "A"}, 0.8),
            ({"name": "B"}, 1.0),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        # 无 quality 字段, 分数不变
        assert reranked[0][1] == 1.0
        assert reranked[1][1] == 0.8

    def test_different_dimensions(self):
        """不同质量维度加权: 不同维度配置应产生不同结果."""
        # 只关注 accuracy
        reranker_acc = QualityBoostReranker(
            quality_weight=0.5,
            dimension_weights={"accuracy": 1.0},
        )
        # 只关注 trustworthiness
        reranker_trust = QualityBoostReranker(
            quality_weight=0.5,
            dimension_weights={"trustworthiness": 1.0},
        )

        # 不含 overall 字段, 使 _compute_quality_score 使用逐维度加权计算
        high_acc_low_trust = {
            "accuracy": 0.9, "trustworthiness": 0.1,
        }
        low_acc_high_trust = {
            "accuracy": 0.1, "trustworthiness": 0.9,
        }
        results = [
            ({"name": "高准确低可信", "quality": high_acc_low_trust}, 1.0),
            ({"name": "低准确高可信", "quality": low_acc_high_trust}, 1.0),
        ]

        # accuracy 优先: 高准确排前面
        reranked_acc = reranker_acc.rerank("test", results, top_k=2)
        assert reranked_acc[0][0]["name"] == "高准确低可信"

        # trustworthiness 优先: 高可信排前面
        reranked_trust = reranker_trust.rerank("test", results, top_k=2)
        assert reranked_trust[0][0]["name"] == "低准确高可信"


# ============================================================
# 时间新颖性重排器测试
# ============================================================


class TestRecencyBoostReranker:
    """RecencyBoostReranker 时间新颖性重排器测试."""

    def test_exponential_decay(self):
        """指数衰减: 指数衰减函数应使最新内容得分更高."""
        now = time.time()
        reranker = RecencyBoostReranker(
            decay_function="exponential",
            half_life_days=30.0,
            now=now,
        )
        recent_time = now - 86400  # 1天前
        old_time = now - 86400 * 60  # 60天前
        results = [
            ({"name": "旧内容", "updated_at": old_time}, 1.0),
            ({"name": "新内容", "updated_at": recent_time}, 1.0),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        assert reranked[0][0]["name"] == "新内容"
        assert reranked[0][1] > reranked[1][1]

    def test_gaussian_decay(self):
        """高斯衰减: 高斯衰减函数应使最新内容得分更高."""
        now = time.time()
        reranker = RecencyBoostReranker(
            decay_function="gaussian",
            half_life_days=30.0,
            sigma_days=30.0,
            now=now,
        )
        recent_time = now - 86400  # 1天前
        old_time = now - 86400 * 90  # 90天前
        results = [
            ({"name": "旧内容", "updated_at": old_time}, 1.0),
            ({"name": "新内容", "updated_at": recent_time}, 1.0),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        assert reranked[0][0]["name"] == "新内容"
        assert reranked[0][1] > reranked[1][1]

    def test_linear_decay(self):
        """线性衰减: 线性衰减函数应使最新内容得分更高."""
        now = time.time()
        reranker = RecencyBoostReranker(
            decay_function="linear",
            half_life_days=30.0,
            max_age_days=120.0,
            now=now,
        )
        recent_time = now - 86400  # 1天前
        old_time = now - 86400 * 100  # 100天前
        results = [
            ({"name": "旧内容", "updated_at": old_time}, 1.0),
            ({"name": "新内容", "updated_at": recent_time}, 1.0),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        assert reranked[0][0]["name"] == "新内容"
        assert reranked[0][1] > reranked[1][1]

    def test_recent_gets_higher_score(self):
        """最新内容得分更高: 越新的内容应获得越高的加权分数."""
        now = time.time()
        reranker = RecencyBoostReranker(
            decay_function="exponential",
            half_life_days=10.0,
            now=now,
        )
        # 三个不同时间的内容
        t1 = now - 86400 * 1   # 1天前
        t2 = now - 86400 * 5   # 5天前
        t3 = now - 86400 * 20  # 20天前
        results = [
            ({"name": "最旧", "updated_at": t3}, 1.0),
            ({"name": "中等", "updated_at": t2}, 1.0),
            ({"name": "最新", "updated_at": t1}, 1.0),
        ]
        reranked = reranker.rerank("test", results, top_k=3)

        # 按新颖性降序排列
        assert reranked[0][0]["name"] == "最新"
        assert reranked[1][0]["name"] == "中等"
        assert reranked[2][0]["name"] == "最旧"
        # 分数应递减
        assert reranked[0][1] > reranked[1][1] > reranked[2][1]


# ============================================================
# 图中心性重排器测试
# ============================================================


class TestGraphCentralityReranker:
    """GraphCentralityReranker 图中心性重排器测试."""

    def test_centrality_boost(self):
        """中心性加权: 图中连接更多的实体应获得更高分数."""
        from dy3_polaris.l3.models import KnowledgeGraph

        # 构建图: e1 连接 e2 和 e3 (高中心性), e2 只连接 e1
        e1 = make_entity(name="中心实体", entity_id="e-center")
        e2 = make_entity(name="边缘实体1", entity_id="e-edge1")
        e3 = make_entity(name="边缘实体2", entity_id="e-edge2")

        e1.add_triple(make_triple(
            subject_id="e-center", predicate="related_to", object_id="e-edge1"
        ))
        e1.add_triple(make_triple(
            subject_id="e-center", predicate="related_to", object_id="e-edge2"
        ))

        graph = KnowledgeGraph(
            entities={"e-center": e1, "e-edge1": e2, "e-edge2": e3},
        )

        reranker = GraphCentralityReranker(graph=graph, alpha=0.5, beta=0.0, community_weight=0.0)
        results = [
            ({"name": "边缘实体1", "entity_id": "e-edge1"}, 1.0),
            ({"name": "中心实体", "entity_id": "e-center"}, 1.0),
        ]
        reranked = reranker.rerank("中心实体", results, top_k=2)

        # 中心实体度中心性更高, 应排前面
        assert reranked[0][0]["name"] == "中心实体"
        assert reranked[0][1] > reranked[1][1]

    def test_without_graph(self):
        """无图时的处理: 不提供 KnowledgeGraph 时从结果构建图结构."""
        reranker = GraphCentralityReranker(graph=None, alpha=0.3, beta=0.3)
        # 结果中包含 triples 字段, 可构建局部图
        results = [
            (
                {
                    "name": "实体A",
                    "entity_id": "e-a",
                    "triples": [
                        {"object_id": "e-b", "object_is_literal": False},
                    ],
                },
                1.0,
            ),
            (
                {
                    "name": "实体B",
                    "entity_id": "e-b",
                    "triples": [],
                },
                0.8,
            ),
        ]
        # 不抛异常即可
        reranked = reranker.rerank("test", results, top_k=2)
        assert len(reranked) == 2

    def test_proximity(self):
        """图邻近性: 与查询实体图距离更近的实体应获得更高分数."""
        from dy3_polaris.l3.models import KnowledgeGraph

        # 构建线性图: e1 -> e2 -> e3
        e1 = make_entity(name="锚点", entity_id="e-anchor", domain="test")
        e2 = make_entity(name="近邻", entity_id="e-near", domain="test")
        e3 = make_entity(name="远邻", entity_id="e-far", domain="test")

        e1.add_triple(make_triple(
            subject_id="e-anchor", predicate="related_to", object_id="e-near"
        ))
        e2.add_triple(make_triple(
            subject_id="e-near", predicate="related_to", object_id="e-far"
        ))

        graph = KnowledgeGraph(
            entities={"e-anchor": e1, "e-near": e2, "e-far": e3},
        )

        reranker = GraphCentralityReranker(
            graph=graph, alpha=0.0, beta=0.5, community_weight=0.0,
            max_graph_distance=5,
        )
        results = [
            ({"name": "远邻", "entity_id": "e-far"}, 1.0),
            ({"name": "近邻", "entity_id": "e-near"}, 1.0),
        ]
        reranked = reranker.rerank("锚点", results, top_k=2)

        # 近邻 (距离1) 应比远邻 (距离2) 获得更高分数
        assert reranked[0][0]["name"] == "近邻"
        assert reranked[0][1] > reranked[1][1]


# ============================================================
# 组合重排器测试
# ============================================================


class TestCompositeReranker:
    """CompositeReranker 组合重排器测试."""

    def test_pipeline(self):
        """管道顺序执行: 多个重排器应按顺序依次应用."""
        pipeline = CompositeReranker([
            MetadataBoostReranker(),
            MMRReranker(lambda_=0.7),
        ])
        results = [
            ({"name": "A", "entity_type": "concept"}, 1.0),
            ({"name": "B", "entity_type": "chemical_compound"}, 0.8),
            ({"name": "C", "entity_type": "material"}, 0.6),
        ]
        reranked = pipeline.rerank("test", results, top_k=3)

        assert len(reranked) == 3
        # 所有结果应被重排 (不是原始顺序)
        names = [r[0]["name"] for r in reranked]
        assert set(names) == {"A", "B", "C"}

    def test_empty_pipeline(self):
        """空管道: 无重排器时应返回原始结果 (截断到 top_k)."""
        pipeline = CompositeReranker([])
        results = [
            ({"name": "A"}, 1.0),
            ({"name": "B"}, 0.8),
            ({"name": "C"}, 0.6),
        ]
        reranked = pipeline.rerank("test", results, top_k=2)

        assert len(reranked) == 2
        # 应按原始顺序截断
        assert reranked[0][0]["name"] == "A"
        assert reranked[1][0]["name"] == "B"

    def test_top_k_per_stage(self):
        """阶段截断: top_k_per_stage 应在中间阶段截断结果数."""
        pipeline = CompositeReranker(
            [
                MetadataBoostReranker(),
                MMRReranker(lambda_=0.7),
            ],
            top_k_per_stage=2,
        )
        results = [
            ({"name": "A", "entity_type": "concept"}, 1.0),
            ({"name": "B", "entity_type": "chemical_compound"}, 0.9),
            ({"name": "C", "entity_type": "material"}, 0.8),
            ({"name": "D", "entity_type": "method"}, 0.7),
            ({"name": "E", "entity_type": "dataset"}, 0.6),
        ]
        reranked = pipeline.rerank("test", results, top_k=3)

        # 中间阶段截断到 2, 最终阶段截断到 3 (但只有 2 条)
        assert len(reranked) <= 3


# ============================================================
# 检索引擎重排集成测试
# ============================================================


class TestRetrievalEngineRerank:
    """RetrievalEngine 重排集成测试."""

    def test_keyword_search_with_rerank(self):
        """关键词检索+重排: 启用重排后应返回重排后的结果."""
        store = KnowledgeStore()
        # 添加可搜索的切片
        chunk1 = make_chunk(content="化学反应催化剂研究", document_id="doc1")
        chunk2 = make_chunk(content="化学反应动力学分析", document_id="doc2")
        chunk3 = make_chunk(content="物理力学基础理论", document_id="doc3")
        store.add_chunk(chunk1)
        store.add_chunk(chunk2)
        store.add_chunk(chunk3)

        reranker = MMRReranker(lambda_=0.7)
        engine = RetrievalEngine(store, reranker=reranker)

        # 不带重排的检索
        result_no_rerank = engine.keyword_search("化学反应", top_k=3)
        assert not result_no_rerank.is_empty()

        # 带重排的检索
        result_rerank = engine.keyword_search("化学反应", top_k=3, rerank=True)
        assert not result_rerank.is_empty()
        assert len(result_rerank.results) == len(result_rerank.scores)
        assert "rerank" in result_rerank.source_type

    def test_rerank_method(self):
        """rerank 方法: 对 RetrievalResult 重排应返回新的 RetrievalResult."""
        store = KnowledgeStore()
        reranker = MMRReranker(lambda_=0.7)
        engine = RetrievalEngine(store, reranker=reranker)

        result = make_result(
            query="test",
            results=[
                {"name": "结果1", "entity_type": "concept"},
                {"name": "结果2", "entity_type": "chemical_compound"},
                {"name": "结果3", "entity_type": "material"},
            ],
            scores=[1.0, 0.8, 0.6],
        )
        reranked = engine.rerank(result, query="test", top_k=3)

        assert isinstance(reranked, RetrievalResult)
        assert len(reranked.results) == 3
        assert "rerank" in reranked.source_type

    def test_rerank_without_reranker(self):
        """无重排器时调用 rerank 抛异常: 未设置重排器时应抛出 RetrievalError."""
        store = KnowledgeStore()
        engine = RetrievalEngine(store)

        result = make_result(query="test", results=[{"name": "A"}], scores=[1.0])
        with pytest.raises(RetrievalError):
            engine.rerank(result, query="test")

    def test_set_reranker(self):
        """动态设置重排器: 通过属性设置重排器后应可正常重排."""
        store = KnowledgeStore()
        engine = RetrievalEngine(store)

        # 初始无重排器
        assert engine.reranker is None

        result = make_result(
            query="test",
            results=[{"name": "A"}, {"name": "B"}],
            scores=[1.0, 0.8],
        )
        with pytest.raises(RetrievalError):
            engine.rerank(result, query="test")

        # 动态设置重排器
        engine.reranker = MMRReranker(lambda_=0.7)
        assert engine.reranker is not None

        # 现在可以重排
        reranked = engine.rerank(result, query="test", top_k=2)
        assert isinstance(reranked, RetrievalResult)
        assert len(reranked.results) == 2
