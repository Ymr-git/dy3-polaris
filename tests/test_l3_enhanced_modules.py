"""L3 领域知识层增强模块综合测试套件.

覆盖四个增强模块:
- query_rewriter: 查询重写引擎 (5 种策略 + 领域词典)
- embedding: 嵌入管理器 (LRU+TTL 缓存 + 伪嵌入 + 归一化 + 统计)
- metrics: 指标收集与监控 (Counter/Histogram/Timer/Gauge + Prometheus 导出)
- community: 知识图谱社区检测 (标签传播/连通分量/Louvain + 模块度)
"""

from __future__ import annotations

import logging
import math
import time

import pytest

from dy3_polaris.l3 import (
    Community,
    CommunityAlgorithm,
    CommunityDetectionResult,
    CommunityDetector,
    Counter,
    EmbeddingBackend,
    EmbeddingCache,
    EmbeddingManager,
    EmbeddingResult,
    EntityType,
    Histogram,
    KnowledgeEntity,
    KnowledgeStore,
    KnowledgeTriple,
    MetricSample,
    MetricType,
    MetricsCollector,
    RelationType,
    RewrittenQuery,
    QueryRewriter,
    RewriteStrategy,
    Timer,
)

logging.disable(logging.CRITICAL)


# ============================================================
# 测试数据工厂
# ============================================================


def make_entity(
    name: str = "测试实体",
    entity_type: EntityType = EntityType.CONCEPT,
    domain: str = "test",
    description: str = "测试描述",
    **kwargs,
) -> KnowledgeEntity:
    """创建测试实体."""
    return KnowledgeEntity(
        name=name,
        entity_type=entity_type,
        domain=domain,
        description=description,
        **kwargs,
    )


def make_triple(
    subject_id: str,
    predicate: str = RelationType.RELATED_TO.value,
    object_id: str = "",
    confidence: float = 1.0,
    **kwargs,
) -> KnowledgeTriple:
    """创建测试三元组."""
    return KnowledgeTriple(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        confidence=confidence,
        **kwargs,
    )


def build_two_triangle_store() -> KnowledgeStore:
    """构建含两个独立三角形 (e-a,e-b,e-c) 与 (e-d,e-e,e-f) 的知识存储.

    邻接关系:
        a - b - c - a   (三角形 1)
        d - e - f - d   (三角形 2)
    """
    store = KnowledgeStore()
    entities = {
        "e-a": make_entity("实体A", EntityType.MATERIAL, description="三角形1节点A"),
        "e-b": make_entity("实体B", EntityType.MATERIAL, description="三角形1节点B"),
        "e-c": make_entity("实体C", EntityType.MATERIAL, description="三角形1节点C"),
        "e-d": make_entity("实体D", EntityType.MATERIAL, description="三角形2节点D"),
        "e-e": make_entity("实体E", EntityType.MATERIAL, description="三角形2节点E"),
        "e-f": make_entity("实体F", EntityType.MATERIAL, description="三角形2节点F"),
    }
    # 强制指定 entity_id
    for eid, ent in entities.items():
        ent.entity_id = eid
        store.add_entity(ent, check_duplicate=False)

    edges = [
        ("e-a", "e-b"), ("e-b", "e-c"), ("e-c", "e-a"),
        ("e-d", "e-e"), ("e-e", "e-f"), ("e-f", "e-d"),
    ]
    for subj, obj in edges:
        store.add_triple(make_triple(subj, RelationType.RELATED_TO.value, obj))

    return store


# ============================================================
# 查询重写引擎: RewriteStrategy 枚举
# ============================================================


class TestRewriteStrategy:
    """RewriteStrategy 枚举测试."""

    def test_枚举包含五种策略(self):
        assert RewriteStrategy.SYNONYM
        assert RewriteStrategy.DECOMPOSE
        assert RewriteStrategy.HYDE
        assert RewriteStrategy.EXPAND
        assert RewriteStrategy.CONTEXTUAL

    def test_枚举值与字符串值一致(self):
        assert RewriteStrategy.SYNONYM.value == "synonym"
        assert RewriteStrategy.DECOMPOSE.value == "decompose"
        assert RewriteStrategy.HYDE.value == "hyde"
        assert RewriteStrategy.EXPAND.value == "expand"
        assert RewriteStrategy.CONTEXTUAL.value == "contextual"

    def test_枚举成员数量为五(self):
        assert len(list(RewriteStrategy)) == 5

    def test_枚举为字符串子类(self):
        assert isinstance(RewriteStrategy.SYNONYM, str)
        assert RewriteStrategy.SYNONYM == "synonym"


# ============================================================
# 查询重写引擎: RewrittenQuery 数据类
# ============================================================


class TestRewrittenQuery:
    """RewrittenQuery 数据类测试."""

    def test_默认字段值(self):
        rq = RewrittenQuery(
            original="原始",
            rewritten="重写",
            strategy=RewriteStrategy.EXPAND,
        )
        assert rq.sub_queries == []
        assert rq.confidence == 0.0
        assert rq.metadata == {}

    def test_完整字段构造(self):
        rq = RewrittenQuery(
            original="波长",
            rewritten="波长扩展",
            strategy=RewriteStrategy.SYNONYM,
            sub_queries=["子查询1"],
            confidence=0.85,
            metadata={"matched_terms": ["波长"]},
        )
        assert rq.original == "波长"
        assert rq.rewritten == "波长扩展"
        assert rq.strategy == RewriteStrategy.SYNONYM
        assert rq.sub_queries == ["子查询1"]
        assert rq.confidence == 0.85
        assert rq.metadata["matched_terms"] == ["波长"]

    def test_repr_包含策略与置信度(self):
        rq = RewrittenQuery(
            original="原始查询文本",
            rewritten="重写后的查询文本",
            strategy=RewriteStrategy.HYDE,
            confidence=0.7321,
        )
        repr_str = repr(rq)
        assert "hyde" in repr_str
        assert "0.732" in repr_str


# ============================================================
# 查询重写引擎: QueryRewriter
# ============================================================


class TestQueryRewriter:
    """QueryRewriter 查询重写引擎测试."""

    @pytest.fixture
    def rewriter(self) -> QueryRewriter:
        return QueryRewriter()

    # ---- rewrite() 五种策略 ----

    def test_rewrite_expand_默认策略(self, rewriter):
        rq = rewriter.rewrite("波长和效率")
        assert rq.strategy == RewriteStrategy.EXPAND
        assert rq.original == "波长和效率"
        assert rq.rewritten  # 非空
        assert "matched_terms" in rq.metadata

    def test_rewrite_synonym_同义词扩展(self, rewriter):
        rq = rewriter.rewrite("波长", strategy=RewriteStrategy.SYNONYM)
        assert rq.strategy == RewriteStrategy.SYNONYM
        assert "波长" in rq.rewritten
        # 应包含同义词扩展标注
        assert "OR" in rq.rewritten or rq.rewritten == "波长"

    def test_rewrite_decompose_子问题分解(self, rewriter):
        rq = rewriter.rewrite("波长和效率", strategy=RewriteStrategy.DECOMPOSE)
        assert rq.strategy == RewriteStrategy.DECOMPOSE
        assert len(rq.sub_queries) >= 2
        assert "||" in rq.rewritten or len(rq.sub_queries) >= 2

    def test_rewrite_hyde_假设文档(self, rewriter):
        rq = rewriter.rewrite("Dy3+的发射波长", strategy=RewriteStrategy.HYDE)
        assert rq.strategy == RewriteStrategy.HYDE
        assert len(rq.rewritten) > len("Dy3+的发射波长")
        assert "keywords" in rq.metadata

    def test_rewrite_contextual_上下文压缩(self, rewriter):
        rq = rewriter.rewrite("关于波长的效率是什么", strategy=RewriteStrategy.CONTEXTUAL)
        assert rq.strategy == RewriteStrategy.CONTEXTUAL
        assert "compressed" in rq.metadata

    def test_rewrite_置信度在合法范围(self, rewriter):
        for strategy in RewriteStrategy:
            rq = rewriter.rewrite("波长效率和浓度猝灭", strategy=strategy)
            assert 0.0 <= rq.confidence <= 1.0

    def test_rewrite_空查询处理(self, rewriter):
        rq = rewriter.rewrite("")
        assert rq.rewritten == ""
        assert rq.confidence == 0.0
        assert rq.metadata.get("empty") is True

    def test_rewrite_空白查询处理(self, rewriter):
        rq = rewriter.rewrite("   ")
        assert rq.metadata.get("empty") is True
        assert rq.confidence == 0.0

    def test_rewrite_None查询处理(self, rewriter):
        rq = rewriter.rewrite(None)  # type: ignore[arg-type]
        assert rq.metadata.get("empty") is True

    # ---- rewrite_multi() ----

    def test_rewrite_multi_默认全部策略(self, rewriter):
        variants = rewriter.rewrite_multi("荧光猝灭与浓度关系")
        assert len(variants) == 5
        strategies_seen = {v.strategy for v in variants}
        assert strategies_seen == set(RewriteStrategy)

    def test_rewrite_multi_自定义策略子集(self, rewriter):
        variants = rewriter.rewrite_multi(
            "波长", strategies=[RewriteStrategy.HYDE, RewriteStrategy.SYNONYM]
        )
        assert len(variants) == 2
        assert variants[0].strategy == RewriteStrategy.HYDE
        assert variants[1].strategy == RewriteStrategy.SYNONYM

    def test_rewrite_multi_空查询(self, rewriter):
        variants = rewriter.rewrite_multi("")
        assert len(variants) == 5
        for v in variants:
            assert v.confidence == 0.0

    # ---- expand_synonyms() ----

    def test_expand_synonyms_基本扩展(self, rewriter):
        result = rewriter.expand_synonyms("波长")
        assert "波长" in result
        # 应追加未出现在原查询中的同义词
        assert "OR" in result

    def test_expand_synonyms_无匹配术语返回原文(self, rewriter):
        result = rewriter.expand_synonyms("zzz无术语查询yyy")
        assert result == "zzz无术语查询yyy"

    def test_expand_synonyms_空查询(self, rewriter):
        assert rewriter.expand_synonyms("") == ""
        assert rewriter.expand_synonyms("   ") == "   "

    def test_expand_synonyms_多术语同时扩展(self, rewriter):
        result = rewriter.expand_synonyms("波长和效率")
        assert "波长" in result
        assert "效率" in result

    # ---- decompose() ----

    def test_decompose_连词拆分(self, rewriter):
        subs = rewriter.decompose("波长和效率")
        assert len(subs) >= 2

    def test_decompose_句子分隔符拆分(self, rewriter):
        subs = rewriter.decompose("波长？效率")
        assert len(subs) >= 2

    def test_decompose_不可分解返回原查询(self, rewriter):
        subs = rewriter.decompose("发光材料")
        assert subs == ["发光材料"]

    def test_decompose_空查询返回空列表(self, rewriter):
        assert rewriter.decompose("") == []
        assert rewriter.decompose("   ") == []

    def test_decompose_多连词复合查询(self, rewriter):
        subs = rewriter.decompose("波长和效率以及浓度")
        assert len(subs) >= 2

    # ---- generate_hyde() ----

    def test_generate_hyde_生成文档(self, rewriter):
        doc = rewriter.generate_hyde("Dy3+的发射波长")
        assert len(doc) > 20
        assert "波长" in doc

    def test_generate_hyde_空查询(self, rewriter):
        assert rewriter.generate_hyde("") == ""

    def test_generate_hyde_确定性(self, rewriter):
        doc1 = rewriter.generate_hyde("浓度猝灭")
        doc2 = rewriter.generate_hyde("浓度猝灭")
        assert doc1 == doc2

    # ---- extract_keywords() ----

    def test_extract_keywords_提取领域术语(self, rewriter):
        kws = rewriter.extract_keywords("稀土发光材料的浓度猝灭机理")
        assert len(kws) >= 1
        # 至少包含一个领域词汇 (术语或其同义词, 最长匹配优先)
        # "浓度猝灭"是"猝灭"的同义词, "发光"是"荧光"的同义词
        assert any(k in kws for k in ["浓度", "猝灭", "荧光", "浓度猝灭", "发光"])

    def test_extract_keywords_空查询(self, rewriter):
        assert rewriter.extract_keywords("") == []
        assert rewriter.extract_keywords("   ") == []

    def test_extract_keywords_过滤停用词(self, rewriter):
        kws = rewriter.extract_keywords("的 波长 和 效率")
        assert "的" not in kws
        assert "和" not in kws

    def test_extract_keywords_去重(self, rewriter):
        kws = rewriter.extract_keywords("波长 波长 效率")
        # 同一关键词不应重复出现
        assert len(kws) == len(set(kws))

    # ---- 内置领域词典 ----

    def test_内置领域词典包含核心术语(self, rewriter):
        dd = rewriter.domain_dict
        for term in ["波长", "能级", "跃迁", "效率", "浓度", "基质", "荧光", "猝灭"]:
            assert term in dd

    def test_内置领域词典有八个核心术语(self, rewriter):
        assert len(rewriter.domain_dict) == 8

    def test_domain_dict_返回副本不影响内部状态(self, rewriter):
        dd = rewriter.domain_dict
        dd["新术语"] = ["新同义词"]
        # 内部不应被修改
        assert "新术语" not in rewriter.domain_dict

    # ---- add_domain_term() ----

    def test_add_domain_term_添加新术语(self, rewriter):
        rewriter.add_domain_term("量子产率", ["quantum yield", "QY"])
        dd = rewriter.domain_dict
        assert "量子产率" in dd
        assert dd["量子产率"] == ["quantum yield", "QY"]
        # 新术语应能被匹配
        rq = rewriter.rewrite("量子产率", strategy=RewriteStrategy.SYNONYM)
        assert "量子产率" in rq.metadata["matched_terms"]

    def test_add_domain_term_更新已有术语(self, rewriter):
        original_syns = rewriter.domain_dict["波长"]
        rewriter.add_domain_term("波长", ["新同义词"])
        dd = rewriter.domain_dict
        assert dd["波长"] == ["新同义词"]
        assert dd["波长"] != original_syns

    # ---- 自定义领域词典 ----

    def test_自定义领域词典初始化(self):
        custom = QueryRewriter(domain_dict={"催化剂": ["catalyst", "cat"]})
        dd = custom.domain_dict
        assert "催化剂" in dd
        # 不应包含内置术语
        assert "波长" not in dd

    def test_自定义词典深拷贝隔离(self):
        source = {"术语": ["同义词"]}
        custom = QueryRewriter(domain_dict=source)
        source["术语"].append("外部修改")
        # 内部不应被外部修改影响
        assert custom.domain_dict["术语"] == ["同义词"]

    # ---- repr ----

    def test_repr_显示术语数量(self, rewriter):
        assert "QueryRewriter" in repr(rewriter)
        assert "8" in repr(rewriter)


# ============================================================
# 嵌入管理器: EmbeddingBackend 枚举
# ============================================================


class TestEmbeddingBackend:
    """EmbeddingBackend 枚举测试."""

    def test_枚举包含四种后端(self):
        assert EmbeddingBackend.OPENAI
        assert EmbeddingBackend.SENTENCE_TRANSFORMERS
        assert EmbeddingBackend.COHERE
        assert EmbeddingBackend.CUSTOM

    def test_枚举值与字符串一致(self):
        assert EmbeddingBackend.OPENAI.value == "openai"
        assert EmbeddingBackend.SENTENCE_TRANSFORMERS.value == "sentence_transformers"
        assert EmbeddingBackend.COHERE.value == "cohere"
        assert EmbeddingBackend.CUSTOM.value == "custom"

    def test_枚举成员数量为四(self):
        assert len(list(EmbeddingBackend)) == 4


# ============================================================
# 嵌入管理器: EmbeddingResult 数据类
# ============================================================


class TestEmbeddingResult:
    """EmbeddingResult 数据类测试."""

    def test_字段构造(self):
        r = EmbeddingResult(
            text="hello",
            vector=[0.1, 0.2],
            model="m",
            dim=2,
            cached=False,
            latency_ms=1.5,
        )
        assert r.text == "hello"
        assert r.vector == [0.1, 0.2]
        assert r.model == "m"
        assert r.dim == 2
        assert r.cached is False
        assert r.latency_ms == 1.5

    def test_repr_包含关键字段(self):
        r = EmbeddingResult(
            text="hello",
            vector=[0.1],
            model="m",
            dim=1,
            cached=True,
            latency_ms=0.123,
        )
        s = repr(r)
        assert "m" in s
        assert "cached=True" in s


# ============================================================
# 嵌入管理器: EmbeddingCache
# ============================================================


class TestEmbeddingCache:
    """EmbeddingCache 缓存测试 (LRU + TTL)."""

    def test_get_set_基本操作(self):
        cache = EmbeddingCache(max_size=10, ttl_seconds=3600)
        assert cache.get("text", "model") is None
        cache.set("text", "model", [0.1, 0.2])
        assert cache.get("text", "model") == [0.1, 0.2]

    def test_get_未命中(self):
        cache = EmbeddingCache(max_size=10)
        assert cache.get("missing", "model") is None

    def test_set_覆盖已存在键(self):
        cache = EmbeddingCache(max_size=10)
        cache.set("t", "m", [0.1])
        cache.set("t", "m", [0.9])
        assert cache.get("t", "m") == [0.9]

    def test_LRU_容量淘汰最久未使用(self):
        cache = EmbeddingCache(max_size=2, ttl_seconds=0)
        cache.set("a", "m", [1.0])
        cache.set("b", "m", [2.0])
        # 访问 a 使其成为 MRU
        cache.get("a", "m")
        # 插入 c, 应淘汰 b (LRU)
        cache.set("c", "m", [3.0])
        assert cache.get("b", "m") is None
        assert cache.get("a", "m") == [1.0]
        assert cache.get("c", "m") == [3.0]
        stats = cache.stats
        assert stats["evictions"] >= 1

    def test_TTL_过期惰性删除(self):
        cache = EmbeddingCache(max_size=10, ttl_seconds=3600)
        cache.set("hello", "model-x", [0.1, 0.2])
        assert cache.get("hello", "model-x") is not None
        # 白盒: 将过期时间戳置为过去, 触发惰性过期
        with cache._lock:
            for key in list(cache._cache):
                vec, _ = cache._cache[key]
                cache._cache[key] = (vec, time.time() - 1.0)
        assert cache.get("hello", "model-x") is None
        assert len(cache) == 0

    def test_TTL_永不过期(self):
        cache = EmbeddingCache(max_size=10, ttl_seconds=0)
        cache.set("t", "m", [0.5])
        # ttl=0 表示永不过期
        assert cache.get("t", "m") == [0.5]

    def test_clear_清空缓存(self):
        cache = EmbeddingCache(max_size=10)
        cache.set("a", "m", [1.0])
        cache.set("b", "m", [2.0])
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a", "m") is None

    def test_clear_不影响统计计数(self):
        cache = EmbeddingCache(max_size=10)
        cache.set("a", "m", [1.0])
        cache.get("a", "m")
        cache.get("missing", "m")
        hits_before = cache.stats["hits"]
        misses_before = cache.stats["misses"]
        cache.clear()
        assert cache.stats["hits"] == hits_before
        assert cache.stats["misses"] == misses_before

    def test_stats_统计信息字段(self):
        cache = EmbeddingCache(max_size=10, ttl_seconds=100)
        cache.set("a", "m", [1.0])
        cache.get("a", "m")
        cache.get("miss", "m")
        stats = cache.stats
        assert "size" in stats
        assert "max_size" in stats
        assert "ttl_seconds" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "evictions" in stats
        assert "hit_rate" in stats
        assert stats["max_size"] == 10
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1

    def test_stats_命中率计算(self):
        cache = EmbeddingCache(max_size=10)
        cache.set("a", "m", [1.0])
        cache.get("a", "m")  # hit
        cache.get("b", "m")  # miss
        cache.get("a", "m")  # hit
        stats = cache.stats
        total = stats["hits"] + stats["misses"]
        assert stats["hit_rate"] == pytest.approx(stats["hits"] / total)

    def test_stats_空缓存命中率为零(self):
        cache = EmbeddingCache(max_size=10)
        assert cache.stats["hit_rate"] == 0.0
        assert cache.stats["total_requests"] == 0

    def test_禁用缓存_max_size为零(self):
        cache = EmbeddingCache(max_size=0)
        cache.set("a", "m", [1.0])
        assert cache.get("a", "m") is None
        assert cache.stats["misses"] == 1

    def test_len_返回条目数(self):
        cache = EmbeddingCache(max_size=10)
        assert len(cache) == 0
        cache.set("a", "m", [1.0])
        assert len(cache) == 1
        cache.set("b", "m", [2.0])
        assert len(cache) == 2

    def test_不同模型独立缓存(self):
        cache = EmbeddingCache(max_size=10)
        cache.set("text", "model-a", [1.0])
        cache.set("text", "model-b", [2.0])
        assert cache.get("text", "model-a") == [1.0]
        assert cache.get("text", "model-b") == [2.0]


# ============================================================
# 嵌入管理器: EmbeddingManager
# ============================================================


class _BadDimManager(EmbeddingManager):
    """返回错误维度向量的测试用管理器."""

    def _compute_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2]  # 故意返回长度 2


class TestEmbeddingManager:
    """EmbeddingManager 嵌入管理器测试."""

    @pytest.fixture
    def manager(self) -> EmbeddingManager:
        return EmbeddingManager(
            backend=EmbeddingBackend.CUSTOM,
            model_name="pseudo-128",
            dim=128,
            normalize=True,
        )

    # ---- embed() ----

    def test_embed_单文本嵌入(self, manager):
        result = manager.embed("Dy3+离子的发射波长")
        assert isinstance(result, EmbeddingResult)
        assert result.text == "Dy3+离子的发射波长"
        assert result.model == "pseudo-128"
        assert result.dim == 128
        assert len(result.vector) == 128
        assert result.cached is False
        assert result.latency_ms >= 0.0

    def test_embed_维度验证(self, manager):
        result = manager.embed("测试")
        assert result.dim == 128
        assert len(result.vector) == 128

    def test_embed_L2归一化范数为1(self, manager):
        result = manager.embed("归一化测试文本")
        norm = math.sqrt(sum(x * x for x in result.vector))
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_embed_关闭归一化时不归一化(self):
        mgr = EmbeddingManager(
            backend=EmbeddingBackend.CUSTOM,
            dim=32,
            normalize=False,
        )
        result = mgr.embed("不归一化文本")
        norm = math.sqrt(sum(x * x for x in result.vector))
        # 伪嵌入值域 [-1,1], 未归一化时范数通常不为 1
        assert norm != pytest.approx(1.0, abs=1e-9)

    def test_embed_确定性伪嵌入(self, manager):
        r1 = manager.embed("确定性测试")
        r2 = manager.embed("确定性测试")
        assert r1.vector == r2.vector

    def test_embed_不同文本向量不同(self, manager):
        r1 = manager.embed("文本甲")
        r2 = manager.embed("文本乙")
        assert r1.vector != r2.vector

    # ---- 缓存命中/未命中 ----

    def test_embed_缓存命中(self, manager):
        first = manager.embed("缓存测试")
        assert first.cached is False
        second = manager.embed("缓存测试")
        assert second.cached is True
        assert second.vector == first.vector

    def test_embed_缓存未命中统计(self, manager):
        manager.reset_stats()
        manager.embed("文本一")
        manager.embed("文本二")
        stats = manager.stats
        assert stats["cache_misses"] == 2
        assert stats["cache_hits"] == 0
        assert stats["compute_count"] == 2

    def test_embed_缓存命中后不重复计算(self, manager):
        manager.reset_stats()
        manager.embed("重复文本")
        manager.embed("重复文本")
        manager.embed("重复文本")
        stats = manager.stats
        # 仅计算一次
        assert stats["compute_count"] == 1
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1

    # ---- embed_batch() ----

    def test_embed_batch_批量嵌入(self, manager):
        texts = ["波长", "能级", "跃迁"]
        results = manager.embed_batch(texts)
        assert len(results) == 3
        for r, t in zip(results, texts):
            assert r.text == t
            assert r.dim == 128

    def test_embed_batch_顺序一致(self, manager):
        texts = ["甲", "乙", "丙"]
        results = manager.embed_batch(texts)
        assert [r.text for r in results] == texts

    def test_embed_batch_空列表(self, manager):
        assert manager.embed_batch([]) == []

    def test_embed_batch_缓存复用(self, manager):
        manager.reset_stats()
        manager.embed_batch(["共享", "文本"])
        manager.reset_stats()
        # 第二次批量应全部命中缓存
        results = manager.embed_batch(["共享", "文本"])
        assert all(r.cached for r in results)
        assert manager.stats["cache_hits"] == 2
        assert manager.stats["compute_count"] == 0

    # ---- 维度验证异常 ----

    def test_embed_维度不匹配抛出异常(self):
        mgr = _BadDimManager(dim=4)
        with pytest.raises(ValueError, match="维度不匹配"):
            mgr.embed("任意文本")

    # ---- 外部后端 NotImplementedError ----

    @pytest.mark.parametrize("backend", [
        EmbeddingBackend.OPENAI,
        EmbeddingBackend.SENTENCE_TRANSFORMERS,
        EmbeddingBackend.COHERE,
    ])
    def test_外部后端抛出NotImplementedError(self, backend):
        mgr = EmbeddingManager(
            backend=backend,
            dim=8,
            cache=EmbeddingCache(max_size=0),  # 禁用缓存确保触发计算
        )
        with pytest.raises(NotImplementedError, match="外部 AI 库"):
            mgr.embed("任意文本")

    # ---- 维度非法 ----

    def test_维度为零抛出ValueError(self):
        with pytest.raises(ValueError, match="正整数"):
            EmbeddingManager(dim=0)

    def test_维度为负抛出ValueError(self):
        with pytest.raises(ValueError, match="正整数"):
            EmbeddingManager(dim=-10)

    # ---- 统计 ----

    def test_stats_统计字段(self, manager):
        manager.reset_stats()
        manager.embed("统计测试")
        stats = manager.stats
        assert stats["backend"] == "custom"
        assert stats["model"] == "pseudo-128"
        assert stats["dim"] == 128
        assert stats["normalize"] is True
        assert stats["total_embeds"] == 1
        assert stats["compute_count"] == 1
        assert "avg_latency_ms" in stats
        assert "cache_stats" in stats

    def test_reset_stats_重置计数器(self, manager):
        manager.embed("重置前")
        assert manager.stats["total_embeds"] >= 1
        manager.reset_stats()
        stats = manager.stats
        assert stats["total_embeds"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["compute_count"] == 0
        assert stats["avg_latency_ms"] == 0.0

    def test_reset_stats_不影响缓存(self, manager):
        manager.embed("缓存保留")
        manager.reset_stats()
        # 缓存应仍命中
        result = manager.embed("缓存保留")
        assert result.cached is True

    # ---- 属性 ----

    def test_透明属性(self, manager):
        assert manager.backend == EmbeddingBackend.CUSTOM
        assert manager.model_name == "pseudo-128"
        assert manager.dim == 128
        assert manager.normalize is True
        assert manager.cache is not None

    def test_repr_包含关键信息(self, manager):
        s = repr(manager)
        assert "custom" in s
        assert "pseudo-128" in s
        assert "128" in s

    def test_自定义缓存注入(self):
        custom_cache = EmbeddingCache(max_size=5, ttl_seconds=60)
        mgr = EmbeddingManager(dim=16, cache=custom_cache)
        assert mgr.cache is custom_cache


# ============================================================
# 指标监控: MetricType 枚举
# ============================================================


class TestMetricType:
    """MetricType 枚举测试."""

    def test_枚举包含四种类型(self):
        assert MetricType.COUNTER
        assert MetricType.GAUGE
        assert MetricType.HISTOGRAM
        assert MetricType.TIMER

    def test_枚举值与字符串一致(self):
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMER.value == "timer"

    def test_枚举成员数量为四(self):
        assert len(list(MetricType)) == 4


# ============================================================
# 指标监控: MetricSample 数据类
# ============================================================


class TestMetricSample:
    """MetricSample 数据类测试."""

    def test_字段构造(self):
        sample = MetricSample(
            name="req_count",
            value=42.0,
            metric_type=MetricType.COUNTER,
            labels={"method": "vector"},
            timestamp=1000.0,
        )
        assert sample.name == "req_count"
        assert sample.value == 42.0
        assert sample.metric_type == MetricType.COUNTER
        assert sample.labels == {"method": "vector"}
        assert sample.timestamp == 1000.0
        assert sample.description == ""

    def test_带描述构造(self):
        sample = MetricSample(
            name="latency",
            value=0.5,
            metric_type=MetricType.HISTOGRAM,
            labels={},
            timestamp=0.0,
            description="延迟分布",
        )
        assert sample.description == "延迟分布"


# ============================================================
# 指标监控: Counter
# ============================================================


class TestCounter:
    """Counter 计数器测试."""

    @pytest.fixture
    def counter(self) -> Counter:
        return Counter("req_count", description="请求计数")

    def test_inc_默认增量为一(self, counter):
        counter.inc()
        assert counter.value == 1.0
        counter.inc()
        assert counter.value == 2.0

    def test_inc_自定义增量(self, counter):
        counter.inc(5.0)
        assert counter.value == 5.0
        counter.inc(2.5)
        assert counter.value == 7.5

    def test_inc_负值抛出ValueError(self, counter):
        with pytest.raises(ValueError, match="只能递增"):
            counter.inc(-1.0)

    def test_inc_零值不递增(self, counter):
        counter.inc(0.0)
        assert counter.value == 0.0

    def test_inc_标签维度(self, counter):
        counter.inc(labels={"method": "vector"})
        counter.inc(labels={"method": "vector"})
        counter.inc(labels={"method": "keyword"})
        labeled = counter.get_labeled_values()
        assert labeled["method=vector"] == 2.0
        assert labeled["method=keyword"] == 1.0

    def test_inc_标签维度总计(self, counter):
        counter.inc(3.0, labels={"a": "1"})
        counter.inc(2.0, labels={"a": "2"})
        assert counter.value == 5.0

    def test_value_属性总计(self, counter):
        counter.inc(10.0)
        counter.inc(labels={"x": "y"})
        assert counter.value == 11.0

    def test_name_和_description_属性(self, counter):
        assert counter.name == "req_count"
        assert counter.description == "请求计数"

    def test_snapshot_无标签(self, counter):
        counter.inc(5.0)
        snap = counter.snapshot()
        assert snap["name"] == "req_count"
        assert snap["type"] == "counter"
        assert snap["value"] == 5.0
        # 无标签递增: labeled_values 含一条空标签条目
        assert len(snap["labeled_values"]) == 1
        assert snap["labeled_values"][0]["labels"] == {}
        assert snap["labeled_values"][0]["value"] == 5.0

    def test_snapshot_含标签(self, counter):
        counter.inc(2.0, labels={"method": "vector"})
        snap = counter.snapshot()
        assert snap["value"] == 2.0
        assert len(snap["labeled_values"]) == 1
        assert snap["labeled_values"][0]["labels"] == {"method": "vector"}
        assert snap["labeled_values"][0]["value"] == 2.0

    def test_标签键排序一致性(self, counter):
        counter.inc(labels={"b": "2", "a": "1"})
        counter.inc(labels={"a": "1", "b": "2"})
        labeled = counter.get_labeled_values()
        # 相同标签 (不同顺序) 应合并到同一键
        assert len(labeled) == 1
        assert list(labeled.values())[0] == 2.0


# ============================================================
# 指标监控: Histogram
# ============================================================


class TestHistogram:
    """Histogram 直方图测试."""

    @pytest.fixture
    def hist(self) -> Histogram:
        return Histogram("latency", description="延迟分布")

    def test_observe_记录观测(self, hist):
        hist.observe(0.1)
        hist.observe(0.2)
        assert hist.count == 2

    def test_count_属性(self, hist):
        assert hist.count == 0
        hist.observe(1.0)
        hist.observe(2.0)
        assert hist.count == 2

    def test_sum_属性(self, hist):
        hist.observe(1.0)
        hist.observe(2.0)
        hist.observe(3.0)
        assert hist.sum == 6.0

    def test_avg_属性(self, hist):
        hist.observe(1.0)
        hist.observe(2.0)
        hist.observe(3.0)
        assert hist.avg == pytest.approx(2.0)

    def test_avg_空直方图返回零(self, hist):
        assert hist.avg == 0.0

    def test_sum_空直方图返回零(self, hist):
        assert hist.sum == 0.0

    def test_percentile_百分位计算(self, hist):
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            hist.observe(v)
        pct = hist.percentile
        assert "p50" in pct
        assert "p90" in pct
        assert "p95" in pct
        assert "p99" in pct
        # p50 应在中位数附近
        assert pct["p50"] == pytest.approx(5.5, abs=1.0)

    def test_percentile_空直方图返回零(self, hist):
        pct = hist.percentile
        assert pct["p50"] == 0.0
        assert pct["p99"] == 0.0

    def test_percentile_单一观测(self, hist):
        hist.observe(42.0)
        pct = hist.percentile
        assert pct["p50"] == 42.0
        assert pct["p99"] == 42.0

    def test_snapshot_包含完整信息(self, hist):
        hist.observe(0.05)
        hist.observe(0.15)
        snap = hist.snapshot()
        assert snap["name"] == "latency"
        assert snap["type"] == "histogram"
        assert snap["count"] == 2
        assert "+Inf" in snap["buckets"]
        assert snap["buckets"]["+Inf"] == 2
        assert "percentile" in snap

    def test_自定义桶边界(self):
        hist = Histogram("custom", buckets=[0.1, 0.5, 1.0])
        hist.observe(0.05)
        hist.observe(0.3)
        hist.observe(2.0)
        snap = hist.snapshot()
        # 桶计数为累积
        assert snap["buckets"]["0.1"] == 1
        assert snap["buckets"]["0.5"] == 2
        assert snap["buckets"]["1.0"] == 2
        assert snap["buckets"]["+Inf"] == 3

    def test_name_和_description_属性(self, hist):
        assert hist.name == "latency"
        assert hist.description == "延迟分布"


# ============================================================
# 指标监控: Timer
# ============================================================


class TestTimer:
    """Timer 计时器测试."""

    def test_上下文管理器协议(self):
        collector = MetricsCollector()
        timer = collector.timer("op_time")
        with timer as t:
            assert t is timer
        assert timer.elapsed > 0.0

    def test_计时功能(self):
        collector = MetricsCollector()
        with collector.timer("timed_op"):
            time.sleep(0.01)
        hist = collector.get_histogram("timed_op")
        assert hist is not None
        assert hist.count == 1
        assert hist.sum >= 0.01

    def test_exit_后记录到收集器(self):
        collector = MetricsCollector()
        with collector.timer("recorded"):
            pass
        snap = collector.snapshot()
        assert "recorded" in snap["histograms"]

    def test_elapsed_退出上下文后有效(self):
        collector = MetricsCollector()
        timer = collector.timer("elapsed_test")
        assert timer.elapsed == 0.0
        with timer:
            pass
        assert timer.elapsed > 0.0

    def test_异常退出仍记录(self):
        collector = MetricsCollector()
        with pytest.raises(RuntimeError):
            with collector.timer("error_op"):
                raise RuntimeError("boom")
        # 即使抛异常也应记录耗时
        assert collector.get_histogram("error_op") is not None


# ============================================================
# 指标监控: MetricsCollector
# ============================================================


class TestMetricsCollector:
    """MetricsCollector 指标收集器测试."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        return MetricsCollector()

    def test_counter_创建计数器(self, collector):
        c = collector.counter("qps", description="查询数")
        assert isinstance(c, Counter)
        assert c.name == "qps"
        # 应可通过 get_counter 获取
        assert collector.get_counter("qps") is c

    def test_counter_幂等获取(self, collector):
        c1 = collector.counter("dup")
        c2 = collector.counter("dup")
        assert c1 is c2

    def test_histogram_创建直方图(self, collector):
        h = collector.histogram("lat", description="延迟")
        assert isinstance(h, Histogram)
        assert collector.get_histogram("lat") is h

    def test_histogram_幂等获取(self, collector):
        h1 = collector.histogram("dup_hist")
        h2 = collector.histogram("dup_hist")
        assert h1 is h2

    def test_timer_创建计时器(self, collector):
        t = collector.timer("timed")
        assert isinstance(t, Timer)
        # 每次返回新实例
        t2 = collector.timer("timed")
        assert t is not t2

    def test_gauge_设置仪表值(self, collector):
        collector.gauge("cache_hit_rate", 0.85, description="缓存命中率")
        assert collector.get_gauge("cache_hit_rate") == 0.85

    def test_gauge_覆盖旧值(self, collector):
        collector.gauge("temp", 1.0)
        collector.gauge("temp", 2.0)
        assert collector.get_gauge("temp") == 2.0

    def test_gauge_不存在返回None(self, collector):
        assert collector.get_gauge("missing") is None

    def test_get_counter_不存在返回None(self, collector):
        assert collector.get_counter("missing") is None

    def test_get_histogram_不存在返回None(self, collector):
        assert collector.get_histogram("missing") is None

    def test_snapshot_快照结构(self, collector):
        collector.counter("c1").inc()
        collector.histogram("h1").observe(1.0)
        collector.gauge("g1", 5.0)
        snap = collector.snapshot()
        assert "counters" in snap
        assert "histograms" in snap
        assert "gauges" in snap
        assert "summary" in snap
        assert snap["summary"]["counter_count"] == 1
        assert snap["summary"]["histogram_count"] == 1
        assert snap["summary"]["gauge_count"] == 1

    def test_snapshot_计数器内容(self, collector):
        collector.counter("cnt").inc(7.0)
        snap = collector.snapshot()
        assert snap["counters"]["cnt"]["value"] == 7.0

    def test_snapshot_仪表内容(self, collector):
        collector.gauge("gv", 3.14, description="仪表")
        snap = collector.snapshot()
        assert snap["gauges"]["gv"]["value"] == 3.14
        assert snap["gauges"]["gv"]["type"] == "gauge"

    def test_reset_清空所有指标(self, collector):
        collector.counter("c").inc()
        collector.histogram("h").observe(1.0)
        collector.gauge("g", 1.0)
        collector.reset()
        snap = collector.snapshot()
        assert snap["summary"]["counter_count"] == 0
        assert snap["summary"]["histogram_count"] == 0
        assert snap["summary"]["gauge_count"] == 0

    def test_export_prometheus_计数器格式(self, collector):
        collector.counter("req_count", description="请求数").inc(5.0)
        text = collector.export_prometheus()
        assert "# HELP req_count 请求数" in text
        assert "# TYPE req_count counter" in text
        assert "req_count 5.0" in text

    def test_export_prometheus_带标签计数器(self, collector):
        c = collector.counter("req")
        c.inc(2.0, labels={"method": "vector"})
        text = collector.export_prometheus()
        assert 'req{method="vector"} 2.0' in text

    def test_export_prometheus_直方图格式(self, collector):
        collector.histogram("latency", description="延迟").observe(0.05)
        text = collector.export_prometheus()
        assert "# HELP latency 延迟" in text
        assert "# TYPE latency histogram" in text
        assert "latency_count 1" in text
        assert 'latency_bucket{le="+Inf"}' in text

    def test_export_prometheus_仪表格式(self, collector):
        collector.gauge("mem_usage", 128.5, description="内存")
        text = collector.export_prometheus()
        assert "# HELP mem_usage 内存" in text
        assert "# TYPE mem_usage gauge" in text
        assert "mem_usage 128.5" in text

    def test_export_prometheus_空收集器(self, collector):
        text = collector.export_prometheus()
        assert text == ""

    def test_export_prometheus_综合导出(self, collector):
        collector.counter("total").inc(10.0)
        collector.histogram("lat").observe(0.1)
        collector.gauge("rate", 0.9)
        text = collector.export_prometheus()
        # 三类指标都应出现
        assert "total" in text
        assert "lat" in text
        assert "rate" in text


# ============================================================
# 社区检测: CommunityAlgorithm 枚举
# ============================================================


class TestCommunityAlgorithm:
    """CommunityAlgorithm 枚举测试."""

    def test_枚举包含四种算法(self):
        assert CommunityAlgorithm.LOUVAIN
        assert CommunityAlgorithm.LABEL_PROP
        assert CommunityAlgorithm.CONNECTED
        assert CommunityAlgorithm.LEIDEN

    def test_枚举值与字符串一致(self):
        assert CommunityAlgorithm.LOUVAIN.value == "louvain"
        assert CommunityAlgorithm.LABEL_PROP.value == "label_prop"
        assert CommunityAlgorithm.CONNECTED.value == "connected"
        assert CommunityAlgorithm.LEIDEN.value == "leiden"

    def test_枚举成员数量为四(self):
        assert len(list(CommunityAlgorithm)) == 4


# ============================================================
# 社区检测: Community 数据类
# ============================================================


class TestCommunity:
    """Community 社区数据类测试."""

    def test_默认字段值(self):
        c = Community(community_id=0, entity_ids=["e-1"], triple_ids=["t-1"])
        assert c.summary == ""
        assert c.level == 0
        assert c.parent_id is None
        assert c.metadata == {}

    def test_size_属性(self):
        c = Community(
            community_id=0,
            entity_ids=["e-1", "e-2"],
            triple_ids=["t-1", "t-2", "t-3"],
        )
        assert c.size == 5  # 2 实体 + 3 三元组

    def test_entity_count_属性(self):
        c = Community(
            community_id=0,
            entity_ids=["e-1", "e-2", "e-3"],
            triple_ids=["t-1"],
        )
        assert c.entity_count == 3

    def test_size_空社区为零(self):
        c = Community(community_id=0, entity_ids=[], triple_ids=[])
        assert c.size == 0
        assert c.entity_count == 0

    def test_完整字段构造(self):
        c = Community(
            community_id=2,
            entity_ids=["e-1"],
            triple_ids=["t-1"],
            summary="摘要",
            level=1,
            parent_id=0,
            metadata={"key": "val"},
        )
        assert c.community_id == 2
        assert c.summary == "摘要"
        assert c.level == 1
        assert c.parent_id == 0
        assert c.metadata == {"key": "val"}


# ============================================================
# 社区检测: CommunityDetectionResult 数据类
# ============================================================


class TestCommunityDetectionResult:
    """CommunityDetectionResult 社区检测结果测试."""

    def test_默认字段值(self):
        result = CommunityDetectionResult(
            communities=[],
            algorithm=CommunityAlgorithm.LABEL_PROP,
            total_entities=0,
            total_communities=0,
        )
        assert result.modularity == 0.0
        assert result.detection_time_ms == 0.0
        assert result.levels == 1

    def test_完整字段构造(self):
        comm = Community(community_id=0, entity_ids=["e-1"], triple_ids=[])
        result = CommunityDetectionResult(
            communities=[comm],
            algorithm=CommunityAlgorithm.CONNECTED,
            total_entities=1,
            total_communities=1,
            modularity=0.5,
            detection_time_ms=1.23,
            levels=2,
        )
        assert result.algorithm == CommunityAlgorithm.CONNECTED
        assert result.total_entities == 1
        assert result.total_communities == 1
        assert result.modularity == 0.5
        assert result.detection_time_ms == 1.23
        assert result.levels == 2
        assert len(result.communities) == 1


# ============================================================
# 社区检测: CommunityDetector
# ============================================================


class TestCommunityDetector:
    """CommunityDetector 社区检测器测试."""

    @pytest.fixture
    def two_triangles(self):
        """两个独立三角形的实体与邻接表."""
        entity_ids = ["e-a", "e-b", "e-c", "e-d", "e-e", "e-f"]
        adjacency = {
            "e-a": ["e-b", "e-c"],
            "e-b": ["e-a", "e-c"],
            "e-c": ["e-a", "e-b"],
            "e-d": ["e-e", "e-f"],
            "e-e": ["e-d", "e-f"],
            "e-f": ["e-d", "e-e"],
        }
        return entity_ids, adjacency

    @pytest.fixture
    def store_with_two_triangles(self) -> KnowledgeStore:
        return build_two_triangle_store()

    # ---- detect() 三种算法 ----

    def test_detect_标签传播算法(self, two_triangles):
        entity_ids, adjacency = two_triangles
        detector = CommunityDetector(algorithm=CommunityAlgorithm.LABEL_PROP)
        result = detector.detect(entity_ids, adjacency)
        assert result.algorithm == CommunityAlgorithm.LABEL_PROP
        assert result.total_entities == 6
        # 两个独立三角形应检测出 2 个社区
        assert result.total_communities == 2
        # 所有实体都应被分配
        all_entities = set()
        for c in result.communities:
            all_entities.update(c.entity_ids)
        assert all_entities == set(entity_ids)

    def test_detect_连通分量算法(self, two_triangles):
        entity_ids, adjacency = two_triangles
        detector = CommunityDetector(algorithm=CommunityAlgorithm.CONNECTED)
        result = detector.detect(entity_ids, adjacency)
        assert result.algorithm == CommunityAlgorithm.CONNECTED
        assert result.total_communities == 2
        # 验证三角形内部连通
        comm_entities = [set(c.entity_ids) for c in result.communities]
        assert {"e-a", "e-b", "e-c"} in comm_entities
        assert {"e-d", "e-e", "e-f"} in comm_entities

    def test_detect_Louvain算法(self, two_triangles):
        entity_ids, adjacency = two_triangles
        detector = CommunityDetector(algorithm=CommunityAlgorithm.LOUVAIN)
        result = detector.detect(entity_ids, adjacency)
        assert result.algorithm == CommunityAlgorithm.LOUVAIN
        # 应能检测出社区结构 (至少 1 个, 最多 2 个)
        assert 1 <= result.total_communities <= 2
        # 所有实体被覆盖
        all_entities = set()
        for c in result.communities:
            all_entities.update(c.entity_ids)
        assert all_entities == set(entity_ids)

    def test_detect_默认算法为标签传播(self):
        detector = CommunityDetector()
        assert detector._algorithm == CommunityAlgorithm.LABEL_PROP

    # ---- 空图与单节点 ----

    def test_detect_空图处理(self):
        detector = CommunityDetector()
        result = detector.detect([], {})
        assert result.communities == []
        assert result.total_entities == 0
        assert result.total_communities == 0
        assert result.modularity == 0.0

    def test_detect_单节点图(self):
        detector = CommunityDetector(algorithm=CommunityAlgorithm.CONNECTED)
        result = detector.detect(["e-1"], {"e-1": []})
        assert result.total_communities == 1
        assert result.communities[0].entity_ids == ["e-1"]
        assert result.communities[0].entity_count == 1

    def test_detect_单节点标签传播(self):
        detector = CommunityDetector(algorithm=CommunityAlgorithm.LABEL_PROP)
        result = detector.detect(["e-1"], {"e-1": []})
        assert result.total_communities == 1

    def test_detect_单节点Louvain(self):
        detector = CommunityDetector(algorithm=CommunityAlgorithm.LOUVAIN)
        result = detector.detect(["e-1"], {"e-1": []})
        assert result.total_communities == 1

    # ---- 多社区检测 ----

    def test_detect_多社区检测_连通分量(self):
        # 三个独立节点对
        entity_ids = ["a", "b", "c", "d", "e", "f"]
        adjacency = {
            "a": ["b"], "b": ["a"],
            "c": ["d"], "d": ["c"],
            "e": ["f"], "f": ["e"],
        }
        detector = CommunityDetector(algorithm=CommunityAlgorithm.CONNECTED)
        result = detector.detect(entity_ids, adjacency)
        assert result.total_communities == 3
        # 每个社区应有 2 个实体
        for c in result.communities:
            assert c.entity_count == 2

    def test_detect_全连通图单社区(self):
        # 完全图 K4 (所有节点互连)
        entity_ids = ["a", "b", "c", "d"]
        adjacency = {
            "a": ["b", "c", "d"],
            "b": ["a", "c", "d"],
            "c": ["a", "b", "d"],
            "d": ["a", "b", "c"],
        }
        detector = CommunityDetector(algorithm=CommunityAlgorithm.CONNECTED)
        result = detector.detect(entity_ids, adjacency)
        assert result.total_communities == 1
        assert result.communities[0].entity_count == 4

    # ---- 模块度计算 ----

    def test_modularity_分离社区为正值(self, two_triangles):
        entity_ids, adjacency = two_triangles
        detector = CommunityDetector(algorithm=CommunityAlgorithm.CONNECTED)
        result = detector.detect(entity_ids, adjacency)
        # 两个完全分离的三角形, 模块度应为正
        assert result.modularity > 0.0

    def test_modularity_单社区为零(self):
        # 单一连通社区 (无跨社区结构)
        entity_ids = ["a", "b", "c"]
        adjacency = {"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}
        detector = CommunityDetector(algorithm=CommunityAlgorithm.CONNECTED)
        result = detector.detect(entity_ids, adjacency)
        # 单社区时模块度为 0
        assert result.modularity == pytest.approx(0.0, abs=1e-6)

    def test_modularity_空图为零(self):
        detector = CommunityDetector()
        result = detector.detect([], {})
        assert result.modularity == 0.0

    def test_modularity_范围合法(self, two_triangles):
        entity_ids, adjacency = two_triangles
        for algo in CommunityAlgorithm:
            detector = CommunityDetector(algorithm=algo)
            result = detector.detect(entity_ids, adjacency)
            assert -0.5 <= result.modularity <= 1.0

    # ---- 内部算法方法 ----

    def test_label_propagation_直接调用(self):
        detector = CommunityDetector(algorithm=CommunityAlgorithm.LABEL_PROP)
        entity_ids = ["a", "b", "c", "d"]
        adjacency = {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}
        communities = detector._label_propagation(entity_ids, adjacency)
        assert len(communities) == 2

    def test_connected_components_直接调用(self):
        detector = CommunityDetector()
        entity_ids = ["a", "b", "c"]
        adjacency = {"a": ["b"], "b": ["a"], "c": []}
        communities = detector._connected_components(entity_ids, adjacency)
        assert len(communities) == 2
        # 孤立节点单独成社区
        sizes = sorted(c.entity_count for c in communities)
        assert sizes == [1, 2]

    def test_louvain_无边图退化为连通分量(self):
        detector = CommunityDetector(algorithm=CommunityAlgorithm.LOUVAIN)
        entity_ids = ["a", "b", "c"]
        adjacency = {"a": [], "b": [], "c": []}
        communities = detector._louvain(entity_ids, adjacency)
        # 无边时应调用连通分量
        assert len(communities) == 3

    def test_louvain_有边图返回社区(self):
        detector = CommunityDetector(algorithm=CommunityAlgorithm.LOUVAIN)
        entity_ids = ["a", "b", "c", "d"]
        adjacency = {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}
        communities = detector._louvain(entity_ids, adjacency)
        assert len(communities) >= 1
        # 所有实体被覆盖
        all_e = set()
        for c in communities:
            all_e.update(c.entity_ids)
        assert all_e == set(entity_ids)

    # ---- _normalize_adjacency ----

    def test_normalize_adjacency_去除自环(self):
        entity_ids = ["a", "b"]
        adjacency = {"a": ["a", "b"], "b": ["a"]}
        normalized = CommunityDetector._normalize_adjacency(entity_ids, adjacency)
        assert "a" not in normalized["a"]
        assert "b" in normalized["a"]

    def test_normalize_adjacency_确保双向(self):
        # 单向边应被补全为双向
        entity_ids = ["a", "b"]
        adjacency = {"a": ["b"], "b": []}
        normalized = CommunityDetector._normalize_adjacency(entity_ids, adjacency)
        assert "b" in normalized["a"]
        assert "a" in normalized["b"]

    def test_normalize_adjacency_去除重复邻居(self):
        entity_ids = ["a", "b"]
        adjacency = {"a": ["b", "b", "b"], "b": ["a"]}
        normalized = CommunityDetector._normalize_adjacency(entity_ids, adjacency)
        assert normalized["a"] == ["b"]

    def test_normalize_adjacency_过滤无效实体(self):
        entity_ids = ["a", "b"]
        # "c" 不在实体列表中应被过滤
        adjacency = {"a": ["b", "c"], "b": ["a"]}
        normalized = CommunityDetector._normalize_adjacency(entity_ids, adjacency)
        assert "c" not in normalized["a"]
        assert normalized["a"] == ["b"]

    # ---- detect_from_store() ----

    def test_detect_from_store_基本检测(self, store_with_two_triangles):
        detector = CommunityDetector(algorithm=CommunityAlgorithm.CONNECTED)
        result = detector.detect_from_store(store_with_two_triangles)
        assert result.total_entities == 6
        assert result.total_communities == 2
        # 应填充三元组 ID
        for c in result.communities:
            # 三角形有 3 条边 (三元组)
            assert len(c.triple_ids) == 3

    def test_detect_from_store_空存储(self):
        store = KnowledgeStore()
        detector = CommunityDetector()
        result = detector.detect_from_store(store)
        assert result.total_entities == 0
        assert result.total_communities == 0

    # ---- generate_summary() ----

    def test_generate_summary_含实体与关系(self, store_with_two_triangles):
        detector = CommunityDetector()
        result = detector.detect_from_store(store_with_two_triangles)
        community = result.communities[0]
        summary = detector.generate_summary(community, store_with_two_triangles)
        assert isinstance(summary, str)
        assert "实体" in summary
        assert "统计" in summary

    def test_generate_summary_空社区(self, store_with_two_triangles):
        detector = CommunityDetector()
        empty_community = Community(community_id=99, entity_ids=[], triple_ids=[])
        summary = detector.generate_summary(empty_community, store_with_two_triangles)
        assert "空社区" in summary

    def test_generate_summary_包含三元组关系(self, store_with_two_triangles):
        detector = CommunityDetector()
        result = detector.detect_from_store(store_with_two_triangles)
        community = result.communities[0]
        summary = detector.generate_summary(community, store_with_two_triangles)
        # 应包含关系描述
        assert "关系" in summary

    # ---- get_community_entities() ----

    def test_get_community_entities_返回实体列表(self, store_with_two_triangles):
        detector = CommunityDetector()
        result = detector.detect_from_store(store_with_two_triangles)
        community = result.communities[0]
        entities = detector.get_community_entities(community, store_with_two_triangles)
        assert len(entities) == community.entity_count
        for e in entities:
            assert hasattr(e, "name")

    def test_get_community_entities_空社区(self, store_with_two_triangles):
        detector = CommunityDetector()
        empty = Community(community_id=0, entity_ids=[], triple_ids=[])
        entities = detector.get_community_entities(empty, store_with_two_triangles)
        assert entities == []

    # ---- 检测结果元信息 ----

    def test_detect_检测时间大于零(self, two_triangles):
        entity_ids, adjacency = two_triangles
        detector = CommunityDetector()
        result = detector.detect(entity_ids, adjacency)
        assert result.detection_time_ms >= 0.0

    def test_detect_社区ID连续编号(self, two_triangles):
        entity_ids, adjacency = two_triangles
        detector = CommunityDetector(algorithm=CommunityAlgorithm.CONNECTED)
        result = detector.detect(entity_ids, adjacency)
        ids = [c.community_id for c in result.communities]
        assert ids == list(range(len(ids)))

    def test_detect_层级默认为1(self, two_triangles):
        entity_ids, adjacency = two_triangles
        detector = CommunityDetector()
        result = detector.detect(entity_ids, adjacency)
        assert result.levels == 1


# ============================================================
# 跨模块集成测试
# ============================================================


class TestCrossModuleIntegration:
    """跨模块集成场景测试."""

    def test_查询重写与嵌入管理协同(self):
        """查询重写结果可被嵌入管理器处理."""
        rewriter = QueryRewriter()
        manager = EmbeddingManager(dim=64, normalize=True)

        variants = rewriter.rewrite_multi("波长和效率")
        vectors = []
        for v in variants:
            result = manager.embed(v.rewritten)
            vectors.append(result.vector)
        # 5 个变体均成功生成向量
        assert len(vectors) == 5
        assert all(len(vec) == 64 for vec in vectors)

    def test_指标收集器监控嵌入缓存(self):
        """用 MetricsCollector 监控 EmbeddingManager 缓存命中率."""
        manager = EmbeddingManager(dim=32)
        collector = MetricsCollector()
        hit_counter = collector.counter("embed_hits")
        miss_counter = collector.counter("embed_misses")

        manager.embed("缓存文本")
        manager.embed("缓存文本")  # 命中
        manager.embed("新文本")

        stats = manager.stats
        hit_counter.inc(stats["cache_hits"])
        miss_counter.inc(stats["cache_misses"])

        assert hit_counter.value == stats["cache_hits"]
        assert miss_counter.value == stats["cache_misses"]

    def test_社区检测与指标监控协同(self):
        """社区检测结果用指标监控."""
        detector = CommunityDetector(algorithm=CommunityAlgorithm.CONNECTED)
        collector = MetricsCollector()

        entity_ids = ["a", "b", "c", "d"]
        adjacency = {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}
        result = detector.detect(entity_ids, adjacency)

        collector.gauge("community_count", result.total_communities)
        collector.gauge("modularity", result.modularity)

        assert collector.get_gauge("community_count") == 2
        assert collector.get_gauge("modularity") > 0.0

    def test_查询重写与社区知识库协同(self):
        """从知识库检测社区后, 查询重写器可用于构建检索查询."""
        store = build_two_triangle_store()
        detector = CommunityDetector(algorithm=CommunityAlgorithm.CONNECTED)
        result = detector.detect_from_store(store)

        rewriter = QueryRewriter()
        # 为每个社区生成假设文档查询
        for community in result.communities:
            entities = detector.get_community_entities(community, store)
            query = " ".join(e.name for e in entities)
            rq = rewriter.rewrite(query, strategy=RewriteStrategy.HYDE)
            assert rq.rewritten  # 非空
