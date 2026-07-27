"""L3 领域知识层 — HNSW 向量索引与缓存层测试套件.

覆盖范围:
- HNSWIndex: 增删改查、度量、过滤、召回率、线程安全、大数量集
- LRUCache: 存取、LRU 淘汰、TTL 过期、统计、线程安全
- QueryCache: 实体/检索/遍历缓存、级联失效、查询哈希
- CachedKnowledgeStore: 读缓存、写失效、属性访问
"""

from __future__ import annotations

import math
import random
import threading
import time

import pytest

from dy3_polaris.l3.cache import (
    CacheStats,
    CachedKnowledgeStore,
    LRUCache,
    QueryCache,
)
from dy3_polaris.l3.hnsw_index import HNSWIndex
from dy3_polaris.l3.models import (
    DocumentChunk,
    EntityType,
    KnowledgeEntity,
    KnowledgeTriple,
    RetrievalResult,
)
from dy3_polaris.l3.store import KnowledgeStore


# ============================================================
# 测试数据工厂
# ============================================================


def make_entity(
    name: str = "测试实体",
    entity_type: EntityType = EntityType.CONCEPT,
    domain: str = "test",
    **kwargs,
) -> KnowledgeEntity:
    """创建测试实体."""
    return KnowledgeEntity(
        name=name,
        entity_type=entity_type,
        domain=domain,
        **kwargs,
    )


def make_triple(
    subject_id: str,
    predicate: str = "related_to",
    object_id: str = "e-target",
    confidence: float = 1.0,
    **kwargs,
) -> KnowledgeTriple:
    """创建测试三元组 (宾语为实体, object_id 非空)."""
    return KnowledgeTriple(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        confidence=confidence,
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
# 向量辅助函数
# ============================================================


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度 (越高越相似)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """计算欧氏距离 (越小越近)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def brute_force_topk(
    vectors: dict[str, list[float]],
    query: list[float],
    top_k: int,
    metric: str = "cosine",
) -> list[str]:
    """暴力搜索 top-k, 返回向量 ID 列表 (按相似度降序).

    用于与 HNSW 近似搜索结果对比, 计算召回率。
    """
    scored: list[tuple[str, float]] = []
    for vid, vec in vectors.items():
        if metric == "cosine":
            score = _cosine_similarity(query, vec)
        else:
            score = -_euclidean_distance(query, vec)
        scored.append((vid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [vid for vid, _ in scored[:top_k]]


def compute_recall(hnsw_ids: list[str], brute_ids: list[str]) -> float:
    """计算召回率 = |交集| / |暴力结果数|."""
    if not brute_ids:
        return 1.0
    return len(set(hnsw_ids) & set(brute_ids)) / len(brute_ids)


# ============================================================
# HNSW 向量索引测试
# ============================================================


class TestHNSWIndex:
    """HNSW 向量索引测试."""

    def test_add_and_search_basic(self):
        """基本添加和搜索: 添加少量向量后应返回按相似度排序的结果."""
        index = HNSWIndex(dim=3, ef_construction=50, ef_search=50)
        index.add("a", [1.0, 0.0, 0.0])
        index.add("b", [0.0, 1.0, 0.0])
        index.add("c", [0.0, 0.0, 1.0])
        results = index.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # 最相似的是 a
        assert results[0][0] == "a"
        # 分数降序
        assert results[0][1] >= results[1][1]

    def test_search_empty_index(self):
        """空索引搜索: 空索引应返回空列表."""
        index = HNSWIndex(dim=3)
        assert index.search([1.0, 0.0, 0.0], top_k=5) == []

    def test_search_single_vector(self):
        """单向量搜索: 索引中仅有一个向量时搜索应返回该向量且相似度接近 1."""
        index = HNSWIndex(dim=3, ef_construction=50, ef_search=50)
        index.add("only", [1.0, 2.0, 3.0])
        results = index.search([1.0, 2.0, 3.0], top_k=5)
        assert len(results) == 1
        assert results[0][0] == "only"
        # 余弦相似度应为 1.0
        assert results[0][1] > 0.99

    def test_cosine_metric(self):
        """余弦度量: 应按余弦相似度排序, 角度越小越靠前."""
        index = HNSWIndex(dim=3, metric="cosine", ef_construction=50, ef_search=50)
        index.add("a", [1.0, 0.0, 0.0])
        index.add("b", [1.0, 1.0, 0.0])  # 与 a 夹角 45 度
        index.add("c", [0.0, 1.0, 0.0])  # 与 a 夹角 90 度
        results = index.search([1.0, 0.0, 0.0], top_k=3)
        ids = [r[0] for r in results]
        assert ids[0] == "a"
        # b (cos=0.707) 应排在 c (cos=0) 之前
        assert ids.index("b") < ids.index("c")

    def test_euclidean_metric(self):
        """欧氏度量: 应按欧氏距离排序, 距离越小越靠前."""
        index = HNSWIndex(
            dim=3, metric="euclidean", ef_construction=50, ef_search=50
        )
        index.add("a", [1.0, 0.0, 0.0])  # 距原点 1
        index.add("b", [2.0, 0.0, 0.0])  # 距原点 2
        index.add("c", [5.0, 0.0, 0.0])  # 距原点 5
        results = index.search([0.0, 0.0, 0.0], top_k=3)
        ids = [r[0] for r in results]
        assert ids[0] == "a"
        assert ids.index("b") < ids.index("c")

    def test_remove_vector(self):
        """删除向量: 删除后向量不可检索, 其余向量正常."""
        index = HNSWIndex(dim=3, ef_construction=50, ef_search=50)
        index.add("a", [1.0, 0.0, 0.0])
        index.add("b", [0.0, 1.0, 0.0])
        assert index.remove("a") is True
        assert index.size() == 1
        assert index.get("a") is None
        # b 仍可搜索
        results = index.search([0.0, 1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "b"

    def test_remove_nonexistent(self):
        """删除不存在的向量: 应返回 False."""
        index = HNSWIndex(dim=3)
        assert index.remove("nope") is False

    def test_search_with_filter(self):
        """带过滤的搜索: 仅返回满足过滤条件的向量."""
        index = HNSWIndex(dim=3, ef_construction=50, ef_search=50)
        for i in range(10):
            index.add(f"a{i}", [float(i), 0.0, 0.0], metadata={"cat": "a"})
        for i in range(10):
            index.add(f"b{i}", [float(i), 1.0, 0.0], metadata={"cat": "b"})
        results = index.search(
            [5.0, 0.0, 0.0],
            top_k=5,
            filter_fn=lambda m: m.get("cat") == "a",
        )
        assert 1 <= len(results) <= 5
        for vid, _ in results:
            assert vid.startswith("a")

    def test_recall_rate(self):
        """召回率测试: 构建 200 个向量, 验证 top-10 召回率 >= 0.8."""
        random.seed(42)
        dim = 64
        n = 200
        vectors: dict[str, list[float]] = {}
        for i in range(n):
            vec = [random.gauss(0, 1) for _ in range(dim)]
            vectors[f"v{i}"] = vec

        index = HNSWIndex(
            dim=dim, M=16, ef_construction=200, ef_search=100
        )
        for vid, vec in vectors.items():
            index.add(vid, vec)
        assert index.size() == n

        total_recall = 0.0
        num_queries = 10
        for _ in range(num_queries):
            query = [random.gauss(0, 1) for _ in range(dim)]
            hnsw_ids = [r[0] for r in index.search(query, top_k=10)]
            brute_ids = brute_force_topk(vectors, query, 10, "cosine")
            total_recall += compute_recall(hnsw_ids, brute_ids)
        avg_recall = total_recall / num_queries
        assert avg_recall >= 0.8, f"召回率过低: {avg_recall:.4f}"

    def test_set_ef_search(self):
        """动态调整搜索精度: 设置合法值生效, 非法值抛异常."""
        index = HNSWIndex(dim=3, ef_search=50)
        index.set_ef_search(100)
        # 设置 < 1 应抛出 ValueError
        with pytest.raises(ValueError):
            index.set_ef_search(0)
        # 调整后搜索仍正常
        index.add("a", [1.0, 0.0, 0.0])
        results = index.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1

    def test_get_vector(self):
        """获取向量: 应返回向量及其元数据."""
        index = HNSWIndex(dim=3)
        vec = [1.0, 2.0, 3.0]
        meta = {"src": "doc1"}
        index.add("v1", vec, metadata=meta)
        result = index.get("v1")
        assert result is not None
        got_vec, got_meta = result
        assert got_vec == vec
        assert got_meta == meta

    def test_get_nonexistent(self):
        """获取不存在的向量: 应返回 None."""
        index = HNSWIndex(dim=3)
        assert index.get("nope") is None

    def test_size(self):
        """索引大小: 应正确反映当前向量数量."""
        index = HNSWIndex(dim=3)
        assert index.size() == 0
        index.add("a", [1.0, 0.0, 0.0])
        assert index.size() == 1
        index.add("b", [0.0, 1.0, 0.0])
        assert index.size() == 2
        index.remove("a")
        assert index.size() == 1

    def test_dim_property(self):
        """维度属性: 固定维度与自动推断维度均正确."""
        # 固定维度
        index = HNSWIndex(dim=4)
        assert index.dim == 4
        index.add("a", [1.0, 0.0, 0.0, 0.0])
        assert index.dim == 4
        # 自动推断维度
        index2 = HNSWIndex()  # dim=0
        assert index2.dim == 0
        index2.add("a", [1.0, 0.0, 0.0])
        assert index2.dim == 3

    def test_clear(self):
        """清空索引: 清空后大小为 0, 搜索返回空."""
        index = HNSWIndex(dim=3)
        index.add("a", [1.0, 0.0, 0.0])
        index.add("b", [0.0, 1.0, 0.0])
        index.clear()
        assert index.size() == 0
        assert index.search([1.0, 0.0, 0.0]) == []
        # 清空后可重新添加
        index.add("c", [1.0, 1.0, 1.0])
        assert index.size() == 1

    def test_get_stats(self):
        """统计信息: 应返回节点数、层级、连接数等字段."""
        index = HNSWIndex(dim=3, ef_construction=50, ef_search=50)
        for i in range(10):
            index.add(f"v{i}", [float(i), 0.0, 0.0])
        stats = index.get_stats()
        assert stats["node_count"] == 10
        assert "max_level" in stats
        assert "avg_connections" in stats
        assert "memory_estimate" in stats
        assert "layer_distribution" in stats
        assert stats["memory_estimate"] > 0

    def test_thread_safety(self):
        """线程安全: 多线程并发添加和搜索不应抛异常, 最终数量正确."""
        index = HNSWIndex(dim=8, ef_construction=50, ef_search=50)
        errors: list[Exception] = []

        def adder(tid: int) -> None:
            try:
                for i in range(25):
                    index.add(
                        f"t{tid}-{i}",
                        [float(i), float(tid), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def searcher() -> None:
            try:
                for _ in range(20):
                    index.search(
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=5
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads: list[threading.Thread] = [
            threading.Thread(target=adder, args=(t,)) for t in range(4)
        ]
        threads += [threading.Thread(target=searcher) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"线程中出现异常: {errors}"
        assert index.size() == 100  # 4 * 25

    def test_dimension_mismatch(self):
        """维度不匹配处理: 维度不一致的向量应被跳过, 不计入索引."""
        index = HNSWIndex(dim=4)
        index.add("v1", [1.0, 0.0, 0.0, 0.0])
        # 维度为 3, 与索引维度 4 不匹配, 应被跳过
        index.add("v2", [1.0, 0.0, 0.0])
        assert index.size() == 1
        assert index.get("v2") is None
        assert index.get("v1") is not None

    def test_reinsert_vector(self):
        """重新插入向量: 相同 ID 重新插入应覆盖旧向量."""
        index = HNSWIndex(dim=4, ef_construction=50, ef_search=50)
        index.add("v1", [1.0, 0.0, 0.0, 0.0])
        index.add("v1", [0.0, 1.0, 0.0, 0.0])  # 重新插入
        result = index.get("v1")
        assert result is not None
        got_vec, _ = result
        assert got_vec == [0.0, 1.0, 0.0, 0.0]
        assert index.size() == 1

    def test_large_dataset(self):
        """大数据集测试: 500 个向量应正确构建并保持较高召回率."""
        random.seed(123)
        dim = 32
        index = HNSWIndex(
            dim=dim, M=16, ef_construction=100, ef_search=100
        )
        vectors: dict[str, list[float]] = {}
        for i in range(500):
            vec = [random.gauss(0, 1) for _ in range(dim)]
            index.add(f"v{i}", vec)
            vectors[f"v{i}"] = vec
        assert index.size() == 500

        query = [random.gauss(0, 1) for _ in range(dim)]
        results = index.search(query, top_k=10)
        assert len(results) == 10

        # 验证召回率
        hnsw_ids = [r[0] for r in results]
        brute_ids = brute_force_topk(vectors, query, 10, "cosine")
        recall = compute_recall(hnsw_ids, brute_ids)
        assert recall >= 0.8, f"大数量集召回率过低: {recall:.4f}"


# ============================================================
# LRU 缓存测试
# ============================================================


class TestLRUCache:
    """LRU 缓存测试."""

    def test_put_and_get(self):
        """基本存取: 写入后应能读取到对应值."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_get_nonexistent(self):
        """获取不存在的键: 应返回 None."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        assert cache.get("nope") is None

    def test_lru_eviction(self):
        """LRU 淘汰策略: 超出容量时应淘汰最久未使用的条目."""
        cache: LRUCache[str, int] = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # 应淘汰 "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.stats().evictions == 1

    def test_remove(self):
        """删除条目: 删除后不可读取, 重复删除返回 False."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.put("a", 1)
        assert cache.remove("a") is True
        assert cache.get("a") is None
        assert cache.remove("a") is False

    def test_clear(self):
        """清空缓存: 清空后大小为 0."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size() == 0

    def test_size(self):
        """缓存大小: 应正确反映当前条目数."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        assert cache.size() == 0
        cache.put("a", 1)
        assert cache.size() == 1
        cache.put("b", 2)
        assert cache.size() == 2

    def test_contains(self):
        """包含检查: 存在返回 True, 不存在返回 False."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.put("a", 1)
        assert cache.contains("a") is True
        assert cache.contains("b") is False

    def test_keys(self):
        """获取所有键: 应返回当前所有缓存键."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.put("a", 1)
        cache.put("b", 2)
        keys = cache.keys()
        assert set(keys) == {"a", "b"}

    def test_stats(self):
        """统计信息: 应正确记录命中、未命中和命中率."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.put("a", 1)
        cache.get("a")  # 命中
        cache.get("b")  # 未命中
        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    def test_ttl_expiration(self):
        """TTL 过期: 过期条目应返回 None 并被清除."""
        cache: LRUCache[str, int] = LRUCache(max_size=10, ttl=10.0)
        cache.put("k1", 1)
        assert cache.get("k1") == 1
        # 直接将过期时间设为过去, 模拟 TTL 到期
        cache._cache["k1"] = (1, time.time() - 1.0)
        assert cache.get("k1") is None
        assert cache.size() == 0

    def test_ttl_custom(self):
        """自定义 TTL: put 时指定 ttl 应覆盖缓存默认 ttl."""
        cache: LRUCache[str, int] = LRUCache(max_size=10, ttl=100.0)
        now = time.time()
        cache.put("default", 1)
        cache.put("custom", 2, ttl=1.0)
        _, default_expiry = cache._cache["default"]
        _, custom_expiry = cache._cache["custom"]
        # 默认 TTL 条目过期时间较远
        assert default_expiry > now + 50
        # 自定义短 TTL 条目过期时间较近但尚未过期
        assert now < custom_expiry < now + 5

    def test_stats_reset(self):
        """重置统计: 重置后所有计数器归零."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.put("a", 1)
        cache.get("a")  # 命中
        cache.get("b")  # 未命中
        stats = cache.stats()
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_requests == 0
        assert stats.hit_rate == 0.0

    def test_zero_size_cache(self):
        """零大小缓存: 禁用模式下 put 为空操作, get 返回 None."""
        cache: LRUCache[str, int] = LRUCache(max_size=0)
        cache.put("a", 1)  # 空操作
        assert cache.get("a") is None
        assert cache.size() == 0
        # 仍记录未命中
        assert cache.stats().misses == 1

    def test_overwrite_value(self):
        """覆盖已有值: 同一键再次写入应覆盖旧值且不增加条目数."""
        cache: LRUCache[str, int] = LRUCache(max_size=10)
        cache.put("a", 1)
        cache.put("a", 2)
        assert cache.get("a") == 2
        assert cache.size() == 1

    def test_thread_safety(self):
        """线程安全: 多线程并发读写不应抛异常."""
        cache: LRUCache[str, int] = LRUCache(max_size=1000)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for i in range(100):
                    cache.put(f"k{i}", i)
                    cache.get(f"k{i}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"线程中出现异常: {errors}"


# ============================================================
# 查询缓存测试
# ============================================================


class TestQueryCache:
    """查询缓存测试."""

    def test_entity_cache(self):
        """实体缓存: 写入实体后应能读取到."""
        qc = QueryCache(max_size=10, ttl=0)
        entity = make_entity(name="E1")
        qc.put_entity("e1", entity)
        assert qc.get_entity("e1") is entity
        assert qc.get_entity("e2") is None

    def test_retrieval_cache(self):
        """检索结果缓存: 写入检索结果后应能读取到."""
        qc = QueryCache(max_size=10, ttl=0)
        result = RetrievalResult(query="q", results=[{"a": 1}])
        qh = QueryCache.make_query_hash("q")
        qc.put_retrieval(qh, result)
        assert qc.get_retrieval(qh) is result

    def test_traversal_cache(self):
        """遍历缓存: 写入遍历结果后应能按源 ID 和深度读取到."""
        qc = QueryCache(max_size=10, ttl=0)
        qc.put_traversal("e1", 3, [{"path": "a->b"}])
        assert qc.get_traversal("e1", 3) == [{"path": "a->b"}]
        # 不同深度应未命中
        assert qc.get_traversal("e1", 2) is None

    def test_invalidate_entity(self):
        """实体失效: 应级联失效实体、遍历和依赖该实体的检索缓存."""
        qc = QueryCache(max_size=10, ttl=0)
        qc.put_entity("e1", make_entity(name="E1"))
        qc.put_traversal("e1", 2, [{"node": "x"}])
        # 构造依赖 e1 的检索结果
        result = RetrievalResult(
            query="q",
            results=[{"entity_id": "e1", "score": 0.9}],
            scores=[0.9],
        )
        qh = QueryCache.make_query_hash("q")
        qc.put_retrieval(qh, result)

        count = qc.invalidate_entity("e1")
        assert count == 3  # 实体 + 遍历 + 检索
        assert qc.get_entity("e1") is None
        assert qc.get_traversal("e1", 2) is None
        assert qc.get_retrieval(qh) is None

    def test_invalidate_all(self):
        """全部失效: 应清除所有命名空间的缓存."""
        qc = QueryCache(max_size=10, ttl=0)
        qc.put_entity("e1", make_entity(name="E1"))
        qc.put_retrieval("h1", RetrievalResult(query="q"))
        qc.put_traversal("e1", 1, [])
        count = qc.invalidate_all()
        assert count == 3
        assert qc.get_entity("e1") is None
        assert qc.get_retrieval("h1") is None
        assert qc.get_traversal("e1", 1) is None

    def test_make_query_hash(self):
        """查询哈希生成: 相同查询参数生成相同哈希, 不同则不同."""
        h1 = QueryCache.make_query_hash("query", top_k=10)
        h2 = QueryCache.make_query_hash("query", top_k=10)
        h3 = QueryCache.make_query_hash("query", top_k=20)
        h4 = QueryCache.make_query_hash("other", top_k=10)
        assert h1 == h2
        assert h1 != h3
        assert h1 != h4
        assert len(h1) == 64  # SHA-256 十六进制长度

    def test_cache_stats(self):
        """缓存统计: 应返回 entity/retrieval/traversal 三个命名空间的统计."""
        qc = QueryCache(max_size=10, ttl=0)
        stats = qc.stats()
        assert "entity" in stats
        assert "retrieval" in stats
        assert "traversal" in stats
        assert isinstance(stats["entity"], CacheStats)

    def test_clear(self):
        """清空缓存: 应清除所有缓存条目."""
        qc = QueryCache(max_size=10, ttl=0)
        qc.put_entity("e1", make_entity(name="E1"))
        qc.put_retrieval("h1", RetrievalResult(query="q"))
        qc.put_traversal("e1", 1, [])
        qc.clear()
        assert qc.get_entity("e1") is None
        assert qc.get_retrieval("h1") is None
        assert qc.get_traversal("e1", 1) is None

    def test_different_queries_different_cache(self):
        """不同查询不同缓存: 不同查询哈希应缓存不同结果."""
        qc = QueryCache(max_size=10, ttl=0)
        h1 = QueryCache.make_query_hash("query1")
        h2 = QueryCache.make_query_hash("query2")
        r1 = RetrievalResult(query="query1", results=[{"id": 1}])
        r2 = RetrievalResult(query="query2", results=[{"id": 2}])
        qc.put_retrieval(h1, r1)
        qc.put_retrieval(h2, r2)
        assert qc.get_retrieval(h1).results == [{"id": 1}]
        assert qc.get_retrieval(h2).results == [{"id": 2}]


# ============================================================
# 缓存知识存储测试
# ============================================================


class TestCachedKnowledgeStore:
    """缓存知识存储测试."""

    def test_get_entity_cached(self):
        """实体缓存读取: 首次读取回源并回填, 再次读取命中缓存."""
        store = KnowledgeStore()
        cached_store = CachedKnowledgeStore(store)
        entity = make_entity(name="E1")
        store.add_entity(entity)
        eid = entity.entity_id

        # 首次读取: 未命中 -> 回源 -> 回填
        r1 = cached_store.get_entity(eid)
        assert r1 is not None
        assert r1.name == "E1"
        # 再次读取: 命中缓存
        r2 = cached_store.get_entity(eid)
        assert r2 is not None

        stats = cached_store.cache_stats()
        assert stats["entity"].hits >= 1
        assert stats["entity"].misses >= 1

    def test_add_entity_invalidates(self):
        """添加实体失效缓存: 新增实体应失效检索缓存."""
        store = KnowledgeStore()
        cached_store = CachedKnowledgeStore(store)
        # 预置一条检索缓存
        qh = QueryCache.make_query_hash("q")
        cached_store.cache.put_retrieval(qh, RetrievalResult(query="q"))
        assert cached_store.cache.get_retrieval(qh) is not None
        # 添加实体应失效检索缓存
        cached_store.add_entity(make_entity(name="E1"))
        assert cached_store.cache.get_retrieval(qh) is None

    def test_update_entity_invalidates(self):
        """更新实体失效缓存: 更新后缓存应持有最新值."""
        store = KnowledgeStore()
        cached_store = CachedKnowledgeStore(store)
        entity = make_entity(name="E1")
        store.add_entity(entity)
        eid = entity.entity_id

        # 缓存实体
        cached_store.get_entity(eid)
        assert cached_store.cache.get_entity(eid) is not None

        # 更新实体
        updated = cached_store.update_entity(eid, name="E1-updated")
        assert updated.name == "E1-updated"
        # 缓存应持有最新值
        cached_entity = cached_store.cache.get_entity(eid)
        assert cached_entity is not None
        assert cached_entity.name == "E1-updated"

    def test_remove_entity_invalidates(self):
        """删除实体失效缓存: 删除后缓存与读取均返回 None."""
        store = KnowledgeStore()
        cached_store = CachedKnowledgeStore(store)
        entity = make_entity(name="E1")
        store.add_entity(entity)
        eid = entity.entity_id

        # 缓存实体
        cached_store.get_entity(eid)
        assert cached_store.cache.get_entity(eid) is not None

        # 删除实体
        removed = cached_store.remove_entity(eid)
        assert removed is not None
        assert cached_store.cache.get_entity(eid) is None
        assert cached_store.get_entity(eid) is None

    def test_get_triple_cached(self):
        """三元组缓存: 首次读取回源并回填, 再次读取命中缓存."""
        store = KnowledgeStore()
        cached_store = CachedKnowledgeStore(store)
        triple = make_triple(subject_id="e1", object_id="e2")
        store.add_triple(triple)
        tid = triple.triple_id

        r1 = cached_store.get_triple(tid)
        assert r1 is not None
        r2 = cached_store.get_triple(tid)
        assert r2 is not None

        stats = cached_store.cache_stats()
        assert stats["triple"].hits >= 1
        assert stats["triple"].misses >= 1

    def test_get_chunk_cached(self):
        """切片缓存: 首次读取回源并回填, 再次读取命中缓存."""
        store = KnowledgeStore()
        cached_store = CachedKnowledgeStore(store)
        chunk = make_chunk(content="hello", document_id="doc1")
        store.add_chunk(chunk)
        cid = chunk.chunk_id

        r1 = cached_store.get_chunk(cid)
        assert r1 is not None
        r2 = cached_store.get_chunk(cid)
        assert r2 is not None

        stats = cached_store.cache_stats()
        assert stats["chunk"].hits >= 1
        assert stats["chunk"].misses >= 1

    def test_inner_store_property(self):
        """内部存储属性: 应返回被包装的原始 KnowledgeStore."""
        store = KnowledgeStore()
        cached_store = CachedKnowledgeStore(store)
        assert cached_store.inner_store is store

    def test_cache_property(self):
        """缓存属性: 应返回 QueryCache 实例."""
        store = KnowledgeStore()
        cached_store = CachedKnowledgeStore(store)
        assert isinstance(cached_store.cache, QueryCache)

    def test_cache_stats(self):
        """缓存统计: 应包含 entity/retrieval/traversal/triple/chunk 命名空间."""
        store = KnowledgeStore()
        cached_store = CachedKnowledgeStore(store)
        stats = cached_store.cache_stats()
        for key in ("entity", "retrieval", "traversal", "triple", "chunk"):
            assert key in stats
            assert isinstance(stats[key], CacheStats)
