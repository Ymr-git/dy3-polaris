"""L3 领域知识层 — 知识检索引擎.

融合世界先进方案的检索引擎设计:
- Milvus: 向量相似性搜索 + 标量预过滤
- Weaviate: BM25 + 向量混合检索 + 模块化检索器
- GraphRAG: 图遍历检索 + 社区感知检索
- LlamaIndex: 检索器抽象 + 组合模式
- Cohere Rerank: 多阶段检索 (召回 → 重排)
- RRF (Reciprocal Rank Fusion): 多路检索结果融合

四类检索器:
1. VectorRetriever  — 向量相似性检索 (借鉴 Milvus ANN + Weaviate 预过滤)
2. KeywordRetriever — BM25 全文检索 (借鉴 Weaviate BM25 + Elasticsearch)
3. GraphRetriever   — 图遍历检索 (借鉴 GraphRAG 子图提取 + Neo4j Cypher)
4. HybridRetriever  — 混合检索 (借鉴 RRF 融合 + Cohere 多阶段重排)

所有检索器均基于 KnowledgeStore，支持 RetrievalFilter 过滤。
检索结果统一为 RetrievalResult 格式，包含相关性分数和元数据。
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from .exceptions import RetrievalError
from .models import (
    DocumentChunk,
    EntityType,
    KnowledgeEntity,
    KnowledgeGraph,
    KnowledgeQuery,
    KnowledgeTriple,
    RetrievalFilter,
    RetrievalResult,
    SubgraphConfig,
)
from .store import KnowledgeStore

logger = logging.getLogger(__name__)


# ============================================================
# 基础检索器抽象
# ============================================================


class BaseRetriever:
    """检索器基类 (借鉴 LlamaIndex BaseRetriever).

    定义检索器的统一接口，所有检索器继承此类。

    Attributes:
        store: 关联的知识存储
        source_type: 检索来源类型标识
    """

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store
        self.source_type: str = "base"

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: RetrievalFilter | None = None,
    ) -> RetrievalResult:
        """执行检索 (子类实现)."""
        raise NotImplementedError("子类必须实现 retrieve 方法")

    def _make_result(
        self,
        query: str,
        results: list[tuple[dict[str, Any], float]],
        source_type: str,
        retrieval_time_ms: float,
        filter: RetrievalFilter | None = None,
    ) -> RetrievalResult:
        """构造统一检索结果."""
        return RetrievalResult(
            query=query,
            results=[r[0] for r in results],
            scores=[r[1] for r in results],
            total=len(results),
            retrieval_time_ms=round(retrieval_time_ms, 2),
            source_type=source_type,
            filters=filter.model_dump(mode="json") if filter else {},
        )


# ============================================================
# 向量检索器 — 语义相似性搜索
# ============================================================


class VectorRetriever(BaseRetriever):
    """向量相似性检索器 (借鉴 Milvus ANN + Weaviate 预过滤).

    基于向量索引进行语义相似性搜索，支持:
    - 余弦相似度 / 欧氏距离
    - 元数据预过滤 (借鉴 Weaviate 预过滤模式)
    - 多模态内容过滤
    - 质量分数过滤

    检索流程:
    1. 将查询文本转换为向量 (外部编码，此处接收向量)
    2. 按过滤条件预过滤候选集
    3. 在候选集中执行向量相似性搜索
    4. 返回 top_k 结果

    Attributes:
        store: 关联的知识存储
        source_type: "vector"
    """

    def __init__(self, store: KnowledgeStore) -> None:
        super().__init__(store)
        self.source_type = "vector"

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: RetrievalFilter | None = None,
        query_vector: list[float] | None = None,
    ) -> RetrievalResult:
        """向量相似性检索.

        Args:
            query: 查询文本 (用于结果记录，实际搜索使用 query_vector)
            top_k: 返回前 k 个结果
            filter: 过滤条件
            query_vector: 查询向量 (必须提供)

        Returns:
            检索结果

        Raises:
            RetrievalError: 未提供查询向量或向量维度不匹配
        """
        start_time = time.time()

        if query_vector is None:
            raise RetrievalError(
                query=query,
                reason="向量检索需要提供 query_vector",
            )

        try:
            # 构建预过滤函数 (借鉴 Weaviate 预过滤)
            filter_fn = self._build_filter_fn(filter) if filter else None

            # 执行向量搜索
            raw_results = self.store.search_vector(
                query_vector,
                top_k=top_k,
                filter_fn=filter_fn,
            )

            # 转换为统一结果格式
            results: list[tuple[dict[str, Any], float]] = []
            for chunk, score in raw_results:
                # 二次过滤 (质量、日期等)
                if filter is not None and not filter.matches_chunk(chunk):
                    continue

                results.append((
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": chunk.content[:500],  # 截断长文本
                        "content_type": chunk.content_type.value,
                        "section": chunk.section,
                        "page": chunk.page,
                        "char_count": chunk.char_count,
                        "language": chunk.language,
                        "metadata": chunk.metadata,
                    },
                    float(score),
                ))

                if len(results) >= top_k:
                    break

            elapsed = (time.time() - start_time) * 1000
            return self._make_result(query, results, self.source_type, elapsed, filter)

        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(query=query, reason=str(exc)) from exc

    def _build_filter_fn(self, filter: RetrievalFilter) -> Any:
        """构建向量索引预过滤函数 (借鉴 Weaviate 预过滤)."""
        def filter_fn(metadata: dict[str, Any]) -> bool:
            # 内容类型过滤
            if filter.content_types:
                ct = metadata.get("content_type", "")
                ct_values = [c.value for c in filter.content_types]
                if ct not in ct_values:
                    return False

            # 语言过滤 (通过元数据)
            # 日期过滤通过 chunk 二次过滤处理
            return True

        return filter_fn

    def retrieve_entities(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filter: RetrievalFilter | None = None,
    ) -> list[tuple[KnowledgeEntity, float]]:
        """对实体进行向量检索 (基于实体的关联切片向量).

        通过检索与实体关联的切片向量，间接实现实体级向量检索。
        """
        raw_results = self.store.search_vector(query_vector, top_k=top_k * 3)

        entity_scores: dict[str, list[float]] = {}
        for chunk, score in raw_results:
            # 通过切片的 document_id 关联到实体
            doc_id = chunk.document_id
            entity = self.store.get_entity(doc_id)
            if entity is not None:
                if filter is not None and not filter.matches_entity(entity):
                    continue
                if doc_id not in entity_scores:
                    entity_scores[doc_id] = []
                entity_scores[doc_id].append(score)

        # 聚合分数 (取最大值)
        results: list[tuple[KnowledgeEntity, float]] = []
        for entity_id, scores in entity_scores.items():
            entity = self.store.get_entity(entity_id)
            if entity is not None:
                results.append((entity, max(scores)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ============================================================
# 关键词检索器 — BM25 全文检索
# ============================================================


class KeywordRetriever(BaseRetriever):
    """BM25 全文检索器 (借鉴 Weaviate BM25 + Elasticsearch 倒排索引).

    基于倒排索引和 BM25 算法进行全文检索，支持:
    - 中英文混合分词
    - BM25 相关性评分
    - 文档范围限定
    - 质量和日期过滤

    BM25 优势:
    - 精确匹配: 对关键词查询的精确度高
    - 可解释性: 评分基于词频和文档频率，可解释
    - 效率: 倒排索引支持高效检索

    Attributes:
        store: 关联的知识存储
        source_type: "keyword"
    """

    def __init__(self, store: KnowledgeStore) -> None:
        super().__init__(store)
        self.source_type = "keyword"

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: RetrievalFilter | None = None,
    ) -> RetrievalResult:
        """BM25 全文检索.

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            filter: 过滤条件 (支持 content_types, min_quality, date 范围)

        Returns:
            检索结果
        """
        start_time = time.time()

        try:
            # 执行 BM25 搜索
            raw_results = self.store.search_text(query, top_k=top_k * 3)

            # 过滤和格式化
            results: list[tuple[dict[str, Any], float]] = []
            for chunk, score in raw_results:
                # 应用过滤条件
                if filter is not None and not filter.matches_chunk(chunk):
                    continue

                # 高亮匹配关键词 (借鉴搜索引擎摘要)
                snippet = self._make_snippet(chunk.content, query, max_length=300)

                results.append((
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": snippet,
                        "full_content_length": chunk.char_count,
                        "content_type": chunk.content_type.value,
                        "section": chunk.section,
                        "page": chunk.page,
                        "language": chunk.language,
                        "bm25_score": round(float(score), 4),
                    },
                    float(score),
                ))

                if len(results) >= top_k:
                    break

            elapsed = (time.time() - start_time) * 1000
            return self._make_result(query, results, self.source_type, elapsed, filter)

        except Exception as exc:
            raise RetrievalError(query=query, reason=str(exc)) from exc

    @staticmethod
    def _make_snippet(content: str, query: str, max_length: int = 300) -> str:
        """生成搜索结果摘要 (借鉴搜索引擎摘要).

        在内容中找到第一个匹配查询词的位置，截取周围文本作为摘要。
        """
        if len(content) <= max_length:
            return content

        # 简单关键词定位
        query_lower = query.lower()
        content_lower = content.lower()

        pos = content_lower.find(query_lower.split()[0]) if query_lower.split() else -1

        if pos == -1:
            # 尝试找单个词
            for word in query_lower.split():
                pos = content_lower.find(word)
                if pos != -1:
                    break

        if pos == -1:
            return content[:max_length] + "..."

        # 截取周围文本
        start = max(0, pos - max_length // 3)
        end = min(len(content), start + max_length)

        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""

        return prefix + content[start:end] + suffix

    def retrieve_entities(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: RetrievalFilter | None = None,
    ) -> list[tuple[KnowledgeEntity, float]]:
        """对实体名称和描述进行关键词检索."""
        start_time = time.time()

        # 搜索实体名称
        name_results = []
        # 简单实现: 遍历实体匹配名称和描述
        query_lower = query.lower()
        query_terms = query_lower.split()

        for entity in self.store.entity_store.list_entities(limit=10000):
            if filter is not None and not filter.matches_entity(entity):
                continue

            # 在名称和描述中搜索
            searchable_text = f"{entity.name} {entity.description}".lower()
            score = 0.0
            for term in query_terms:
                if term in searchable_text:
                    score += 1.0
                if term in entity.name.lower():
                    score += 2.0  # 名称匹配权重更高

            # 别名匹配
            for alias in entity.aliases:
                if term in alias.lower():
                    score += 1.5

            if score > 0:
                name_results.append((entity, score))

        name_results.sort(key=lambda x: x[1], reverse=True)
        return name_results[:top_k]


# ============================================================
# 图检索器 — 图遍历检索
# ============================================================


class GraphRetriever(BaseRetriever):
    """图遍历检索器 (借鉴 GraphRAG 子图提取 + Neo4j Cypher 遍历).

    基于知识图谱的图结构进行检索，支持:
    - BFS 子图遍历 (借鉴 GraphRAG 实体中心子图提取)
    - 最短路径查找 (借鉴 Neo4j shortestPath)
    - 邻居查询 (出边/入边/双向)
    - 多跳关系推理 (借鉴 OWL property chain)
    - 置信度和质量过滤

    图检索优势:
    - 关系感知: 能发现通过关系连接的知识
    - 多跳推理: 支持跨实体的间接关联发现
    - 子图上下文: 提供知识的上下文环境

    Attributes:
        store: 关联的知识存储
        source_type: "graph"
    """

    def __init__(self, store: KnowledgeStore) -> None:
        super().__init__(store)
        self.source_type = "graph"

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: RetrievalFilter | None = None,
        entity_id: str | None = None,
        max_depth: int = 2,
        min_confidence: float = 0.5,
    ) -> RetrievalResult:
        """图遍历检索.

        从指定实体出发，BFS 遍历获取关联知识。

        Args:
            query: 查询文本 (用于结果记录)
            top_k: 返回前 k 个结果
            filter: 过滤条件
            entity_id: 起始实体 ID (必须提供)
            max_depth: 遍历深度
            min_confidence: 最低关系置信度

        Returns:
            检索结果 (包含关联实体和三元组)

        Raises:
            RetrievalError: 未提供 entity_id 或实体不存在
        """
        start_time = time.time()

        if entity_id is None:
            raise RetrievalError(
                query=query,
                reason="图检索需要提供 entity_id",
            )

        try:
            # 检查实体是否存在
            entity = self.store.get_entity(entity_id)
            if entity is None:
                raise RetrievalError(
                    query=query,
                    reason=f"实体不存在: {entity_id}",
                )

            # BFS 遍历获取子图
            config = SubgraphConfig(
                entity_focus=entity_id,
                max_depth=max_depth,
                max_entities=top_k * 5,
                min_confidence=min_confidence,
                min_quality=filter.min_quality if filter else 0.0,
                include_deprecated=not (filter.exclude_deprecated if filter else True),
            )

            subgraph = self.store.extract_subgraph(config)

            # 格式化结果
            results: list[tuple[dict[str, Any], float]] = []

            # 添加起始实体 (最高分数)
            results.append((
                {
                    "type": "entity",
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "entity_type": entity.entity_type.value,
                    "description": entity.description[:200],
                    "domain": entity.domain,
                    "is_focus": True,
                },
                1.0,
            ))

            # 添加邻居实体 (按距离衰减分数)
            for neighbor in subgraph.entities.values():
                if neighbor.entity_id == entity_id:
                    continue

                if filter is not None and not filter.matches_entity(neighbor):
                    continue

                # 计算分数 (距离越远分数越低)
                # 通过图遍历深度模拟距离
                path = self.store.triple_store.get_path(entity_id, neighbor.entity_id, max_depth=max_depth)
                distance = len(path) - 1 if path else max_depth
                score = 1.0 / (1.0 + distance)

                results.append((
                    {
                        "type": "entity",
                        "entity_id": neighbor.entity_id,
                        "name": neighbor.name,
                        "entity_type": neighbor.entity_type.value,
                        "description": neighbor.description[:200],
                        "domain": neighbor.domain,
                        "distance": distance,
                        "path": path,
                    },
                    score,
                ))

            # 添加关键三元组
            for triple in subgraph.triples[:top_k * 2]:
                results.append((
                    {
                        "type": "triple",
                        "triple_id": triple.triple_id,
                        "subject_id": triple.subject_id,
                        "predicate": triple.predicate,
                        "object_id": triple.object_id,
                        "confidence": triple.confidence,
                        "rank": triple.rank.value,
                    },
                    triple.confidence * 0.8,  # 三元组分数略低于实体
                ))

            # 按分数排序并截断
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]

            elapsed = (time.time() - start_time) * 1000
            return self._make_result(query, results, self.source_type, elapsed, filter)

        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(query=query, reason=str(exc)) from exc

    def find_path(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int = 5,
        min_confidence: float = 0.0,
    ) -> RetrievalResult:
        """查找两个实体之间的最短路径.

        Args:
            source_id: 起点实体 ID
            target_id: 终点实体 ID
            max_depth: 最大搜索深度
            min_confidence: 最低关系置信度

        Returns:
            检索结果 (包含路径上的实体和关系)
        """
        start_time = time.time()

        try:
            path = self.store.triple_store.get_path(
                source_id, target_id,
                max_depth=max_depth,
                min_confidence=min_confidence,
            )

            results: list[tuple[dict[str, Any], float]] = []

            if path:
                # 添加路径上的实体
                for i, entity_id in enumerate(path):
                    entity = self.store.get_entity(entity_id)
                    if entity:
                        score = 1.0 / (1.0 + i)  # 越靠近起点分数越高
                        results.append((
                            {
                                "type": "entity",
                                "entity_id": entity.entity_id,
                                "name": entity.name,
                                "entity_type": entity.entity_type.value,
                                "path_position": i,
                            },
                            score,
                        ))

                # 添加路径上的关系
                for i in range(len(path) - 1):
                    triples = self.store.triple_store.get_by_subject_predicate(
                        path[i], ""
                    )
                    # 找到连接 path[i] 和 path[i+1] 的三元组
                    for triple in self.store.triple_store.get_outgoing(path[i]):
                        if triple.object_id == path[i + 1]:
                            results.append((
                                {
                                    "type": "triple",
                                    "triple_id": triple.triple_id,
                                    "subject_id": triple.subject_id,
                                    "predicate": triple.predicate,
                                    "object_id": triple.object_id,
                                    "confidence": triple.confidence,
                                },
                                0.8 / (1.0 + i),
                            ))
                            break

            query_str = f"path:{source_id}->{target_id}"
            elapsed = (time.time() - start_time) * 1000
            return self._make_result(query_str, results, self.source_type, elapsed)

        except Exception as exc:
            raise RetrievalError(
                query=f"path:{source_id}->{target_id}",
                reason=str(exc),
            ) from exc

    def get_neighbors(
        self,
        entity_id: str,
        *,
        direction: str = "both",
        min_confidence: float = 0.0,
        top_k: int = 20,
    ) -> RetrievalResult:
        """获取实体的邻居节点.

        Args:
            entity_id: 实体 ID
            direction: 方向 ("out"/"in"/"both")
            min_confidence: 最低关系置信度
            top_k: 返回上限

        Returns:
            检索结果 (包含邻居实体和关系)
        """
        start_time = time.time()

        try:
            neighbor_ids = self.store.triple_store.get_neighbors(
                entity_id,
                direction=direction,
                min_confidence=min_confidence,
            )

            results: list[tuple[dict[str, Any], float]] = []

            # 添加邻居实体
            for nid in neighbor_ids[:top_k]:
                neighbor = self.store.get_entity(nid)
                if neighbor:
                    # 获取关系详情
                    outgoing = self.store.triple_store.get_outgoing(
                        entity_id, min_confidence=min_confidence
                    )
                    incoming = self.store.triple_store.get_incoming(
                        entity_id, min_confidence=min_confidence
                    )

                    best_score = 0.0
                    best_relation = ""

                    for t in outgoing:
                        if t.object_id == nid and t.confidence > best_score:
                            best_score = t.confidence
                            best_relation = t.predicate

                    for t in incoming:
                        if t.subject_id == nid and t.confidence > best_score:
                            best_score = t.confidence
                            best_relation = f"←{t.predicate}"

                    results.append((
                        {
                            "type": "entity",
                            "entity_id": neighbor.entity_id,
                            "name": neighbor.name,
                            "entity_type": neighbor.entity_type.value,
                            "domain": neighbor.domain,
                            "relation": best_relation,
                            "confidence": best_score,
                        },
                        best_score,
                    ))

            results.sort(key=lambda x: x[1], reverse=True)
            query_str = f"neighbors:{entity_id}"
            elapsed = (time.time() - start_time) * 1000
            return self._make_result(query_str, results, self.source_type, elapsed)

        except Exception as exc:
            raise RetrievalError(
                query=f"neighbors:{entity_id}",
                reason=str(exc),
            ) from exc


# ============================================================
# 混合检索器 — 多路融合检索
# ============================================================


class HybridRetriever(BaseRetriever):
    """混合检索器 (借鉴 RRF 融合 + Cohere 多阶段重排).

    融合向量检索、关键词检索和图检索的结果，通过加权融合
    或 Reciprocal Rank Fusion (RRF) 合并排序。

    支持的融合策略:
    - "weighted": 加权分数融合 (各检索器分数归一化后加权)
    - "rrf": Reciprocal Rank Fusion (基于排名的融合，无需分数归一化)
    - "interleave": 交替合并 (各检索器轮流取结果)

    RRF 公式 (借鉴 Cormack et al., 2009):
        RRF_score(d) = Σ 1 / (k + rank_i(d))
    其中 k 为平滑常数 (默认 60)，rank_i(d) 为文档 d 在第 i 个检索器中的排名。

    混合检索优势:
    - 互补性: 向量检索擅长语义匹配，关键词检索擅长精确匹配，图检索擅长关系发现
    - 鲁棒性: 单一检索器失效时其他检索器仍可提供结果
    - 全面性: 覆盖不同类型的知识需求

    Attributes:
        store: 关联的知识存储
        vector_retriever: 向量检索器
        keyword_retriever: 关键词检索器
        graph_retriever: 图检索器
        fusion_strategy: 融合策略
        weights: 各检索器权重 (仅 weighted 策略使用)
        rrf_k: RRF 平滑常数
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        fusion_strategy: str = "rrf",
        weights: dict[str, float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        super().__init__(store)
        self.source_type = "hybrid"
        self.vector_retriever = VectorRetriever(store)
        self.keyword_retriever = KeywordRetriever(store)
        self.graph_retriever = GraphRetriever(store)
        self.fusion_strategy = fusion_strategy
        self.weights = weights or {
            "vector": 0.4,
            "keyword": 0.4,
            "graph": 0.2,
        }
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: RetrievalFilter | None = None,
        query_vector: list[float] | None = None,
        entity_id: str | None = None,
        retrievers: list[str] | None = None,
    ) -> RetrievalResult:
        """混合检索.

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            filter: 过滤条件
            query_vector: 查询向量 (向量检索需要)
            entity_id: 起始实体 ID (图检索需要)
            retrievers: 使用的检索器列表 (None=全部)

        Returns:
            融合后的检索结果
        """
        start_time = time.time()

        active_retrievers = retrievers or ["vector", "keyword", "graph"]
        multi_results: dict[str, list[tuple[dict[str, Any], float]]] = {}

        # 并行执行各检索器 (此处顺序执行，生产环境可并行化)
        if "vector" in active_retrievers and query_vector is not None:
            try:
                vr = self.vector_retriever.retrieve(
                    query, top_k=top_k * 2, filter=filter, query_vector=query_vector
                )
                multi_results["vector"] = list(zip(vr.results, vr.scores))
            except RetrievalError as e:
                logger.warning("向量检索失败: %s", e)

        if "keyword" in active_retrievers:
            try:
                kr = self.keyword_retriever.retrieve(
                    query, top_k=top_k * 2, filter=filter
                )
                multi_results["keyword"] = list(zip(kr.results, kr.scores))
            except RetrievalError as e:
                logger.warning("关键词检索失败: %s", e)

        if "graph" in active_retrievers and entity_id is not None:
            try:
                gr = self.graph_retriever.retrieve(
                    query, top_k=top_k * 2, filter=filter,
                    entity_id=entity_id,
                )
                multi_results["graph"] = list(zip(gr.results, gr.scores))
            except RetrievalError as e:
                logger.warning("图检索失败: %s", e)

        # 融合结果
        if self.fusion_strategy == "rrf":
            fused = self._fuse_rrf(multi_results, top_k)
        elif self.fusion_strategy == "weighted":
            fused = self._fuse_weighted(multi_results, top_k)
        elif self.fusion_strategy == "interleave":
            fused = self._fuse_interleave(multi_results, top_k)
        else:
            fused = self._fuse_rrf(multi_results, top_k)

        elapsed = (time.time() - start_time) * 1000
        return self._make_result(query, fused, self.source_type, elapsed, filter)

    def _fuse_rrf(
        self,
        multi_results: dict[str, list[tuple[dict[str, Any], float]]],
        top_k: int,
    ) -> list[tuple[dict[str, Any], float]]:
        """Reciprocal Rank Fusion (借鉴 Cormack et al., 2009).

        RRF_score(d) = Σ 1 / (k + rank_i(d))

        优势: 无需分数归一化，对不同检索器的分数尺度不敏感。
        """
        # 收集所有唯一结果 (以 chunk_id 或 entity_id 为键)
        all_items: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = defaultdict(float)

        for retriever_name, results in multi_results.items():
            for rank, (item, _score) in enumerate(results):
                # 生成唯一键
                item_key = self._get_item_key(item)
                if item_key not in all_items:
                    all_items[item_key] = item

                # RRF 分数
                rrf_scores[item_key] += 1.0 / (self.rrf_k + rank + 1)

        # 按融合分数排序
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        return [
            (all_items[key], rrf_scores[key])
            for key in sorted_keys[:top_k]
        ]

    def _fuse_weighted(
        self,
        multi_results: dict[str, list[tuple[dict[str, Any], float]]],
        top_k: int,
    ) -> list[tuple[dict[str, Any], float]]:
        """加权分数融合.

        各检索器分数归一化到 [0, 1] 后加权求和。
        """
        # 收集所有唯一结果和各检索器的归一化分数
        all_items: dict[str, dict[str, Any]] = {}
        normalized_scores: dict[str, dict[str, float]] = defaultdict(dict)

        for retriever_name, results in multi_results.items():
            if not results:
                continue

            # 归一化分数 (min-max 归一化)
            scores = [s for _, s in results]
            if not scores:
                continue

            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            for item, score in results:
                item_key = self._get_item_key(item)
                if item_key not in all_items:
                    all_items[item_key] = item

                if score_range > 0:
                    norm_score = (score - min_score) / score_range
                else:
                    norm_score = 1.0

                normalized_scores[item_key][retriever_name] = norm_score

        # 加权求和
        fused_scores: dict[str, float] = {}
        for item_key, score_map in normalized_scores.items():
            total = 0.0
            for retriever_name, norm_score in score_map.items():
                weight = self.weights.get(retriever_name, 0.0)
                total += norm_score * weight
            fused_scores[item_key] = total

        # 按融合分数排序
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

        return [
            (all_items[key], fused_scores[key])
            for key in sorted_keys[:top_k]
        ]

    def _fuse_interleave(
        self,
        multi_results: dict[str, list[tuple[dict[str, Any], float]]],
        top_k: int,
    ) -> list[tuple[dict[str, Any], float]]:
        """交替合并 (各检索器轮流取结果).

        简单但公平的融合策略，确保每个检索器都有结果。
        """
        # 按检索器名称排序以保证确定性
        retriever_names = sorted(multi_results.keys())
        if not retriever_names:
            return []

        # 初始化各检索器的指针
        pointers: dict[str, int] = {name: 0 for name in retriever_names}
        seen_keys: set[str] = set()
        fused: list[tuple[dict[str, Any], float]] = []

        # 轮流取结果
        while len(fused) < top_k:
            made_progress = False
            for name in retriever_names:
                results = multi_results[name]
                ptr = pointers[name]

                # 跳过已见过的结果
                while ptr < len(results):
                    item, score = results[ptr]
                    item_key = self._get_item_key(item)
                    if item_key not in seen_keys:
                        seen_keys.add(item_key)
                        fused.append((item, score))
                        pointers[name] = ptr + 1
                        made_progress = True
                        break
                    ptr += 1
                pointers[name] = ptr

                if len(fused) >= top_k:
                    break

            if not made_progress:
                break

        return fused[:top_k]

    @staticmethod
    def _get_item_key(item: dict[str, Any]) -> str:
        """获取结果项的唯一键 (用于去重和融合)."""
        # 优先使用 chunk_id, 其次 entity_id, 最后用 content 哈希
        if "chunk_id" in item:
            return f"chunk:{item['chunk_id']}"
        if "entity_id" in item:
            return f"entity:{item['entity_id']}"
        if "triple_id" in item:
            return f"triple:{item['triple_id']}"
        return f"item:{hash(item.get('content', ''))}"


# ============================================================
# 检索引擎门面 — 统一入口
# ============================================================


class RetrievalEngine:
    """检索引擎门面 (借鉴 LlamaIndex RetrieverManager + Cohere Rerank).

    统一管理所有检索器，提供单一入口。
    支持可选的结果重排 (reranking)，在检索后对结果进行二次排序。

    Usage::

        engine = RetrievalEngine(store)
        # 向量检索
        result = engine.vector_search(query_vector=[0.1, 0.2, ...])
        # 关键词检索
        result = engine.keyword_search("化学反应机理")
        # 图检索
        result = engine.graph_search(entity_id="e-xxx", max_depth=2)
        # 混合检索
        result = engine.hybrid_search(
            "催化剂", query_vector=[...], entity_id="e-xxx"
        )
        # 带重排的检索
        from dy3_polaris.l3 import MMRReranker
        engine = RetrievalEngine(store, reranker=MMRReranker(lambda_=0.7))
        result = engine.keyword_search("催化剂", rerank=True)
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        fusion_strategy: str = "rrf",
        weights: dict[str, float] | None = None,
        reranker: Any = None,
    ) -> None:
        """初始化检索引擎.

        Args:
            store: 知识存储
            fusion_strategy: 混合检索融合策略 ("rrf" / "weighted" / "interleave")
            weights: 各检索器权重 (仅 weighted 策略)
            reranker: 可选的重排器实例 (BaseReranker 子类)
        """
        self.store = store
        self.vector_retriever = VectorRetriever(store)
        self.keyword_retriever = KeywordRetriever(store)
        self.graph_retriever = GraphRetriever(store)
        self.hybrid_retriever = HybridRetriever(
            store, fusion_strategy=fusion_strategy, weights=weights
        )
        self._reranker = reranker

    @property
    def reranker(self) -> Any:
        """当前重排器."""
        return self._reranker

    @reranker.setter
    def reranker(self, value: Any) -> None:
        """设置重排器."""
        self._reranker = value

    def rerank(
        self, result: RetrievalResult, query: str = "", top_k: int = 10
    ) -> RetrievalResult:
        """对检索结果进行重排.

        Args:
            result: 原始检索结果
            query: 查询文本
            top_k: 重排后返回的结果数

        Returns:
            重排后的检索结果

        Raises:
            RetrievalError: 未设置重排器
        """
        if self._reranker is None:
            raise RetrievalError(
                query=query, reason="未设置重排器 (reranker=None)"
            )
        return self._reranker.rerank_result(query, result, top_k)

    def vector_search(
        self,
        query_vector: list[float],
        *,
        query: str = "",
        top_k: int = 10,
        filter: RetrievalFilter | None = None,
        rerank: bool = False,
    ) -> RetrievalResult:
        """向量检索.

        Args:
            query_vector: 查询向量
            query: 查询文本 (用于重排)
            top_k: 返回结果数
            filter: 过滤条件
            rerank: 是否启用重排 (需预先设置 reranker)
        """
        result = self.vector_retriever.retrieve(
            query=query or "vector_search",
            top_k=top_k,
            filter=filter,
            query_vector=query_vector,
        )
        if rerank and self._reranker is not None:
            result = self._reranker.rerank_result(query, result, top_k)
        return result

    def keyword_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: RetrievalFilter | None = None,
        rerank: bool = False,
    ) -> RetrievalResult:
        """关键词检索.

        Args:
            query: 查询文本
            top_k: 返回结果数
            filter: 过滤条件
            rerank: 是否启用重排 (需预先设置 reranker)
        """
        result = self.keyword_retriever.retrieve(
            query=query, top_k=top_k, filter=filter
        )
        if rerank and self._reranker is not None:
            result = self._reranker.rerank_result(query, result, top_k)
        return result

    def graph_search(
        self,
        entity_id: str,
        *,
        query: str = "",
        top_k: int = 10,
        max_depth: int = 2,
        min_confidence: float = 0.5,
        filter: RetrievalFilter | None = None,
        rerank: bool = False,
    ) -> RetrievalResult:
        """图检索.

        Args:
            entity_id: 起始实体 ID
            query: 查询文本 (用于重排)
            top_k: 返回结果数
            max_depth: 图遍历最大深度
            min_confidence: 最低置信度阈值
            filter: 过滤条件
            rerank: 是否启用重排 (需预先设置 reranker)
        """
        result = self.graph_retriever.retrieve(
            query=query or f"graph:{entity_id}",
            top_k=top_k,
            filter=filter,
            entity_id=entity_id,
            max_depth=max_depth,
            min_confidence=min_confidence,
        )
        if rerank and self._reranker is not None:
            result = self._reranker.rerank_result(query, result, top_k)
        return result

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter: RetrievalFilter | None = None,
        query_vector: list[float] | None = None,
        entity_id: str | None = None,
        retrievers: list[str] | None = None,
        rerank: bool = False,
    ) -> RetrievalResult:
        """混合检索.

        Args:
            query: 查询文本
            top_k: 返回结果数
            filter: 过滤条件
            query_vector: 查询向量 (向量检索用)
            entity_id: 起始实体 ID (图检索用)
            retrievers: 参与的检索器列表
            rerank: 是否启用重排 (需预先设置 reranker)
        """
        result = self.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            filter=filter,
            query_vector=query_vector,
            entity_id=entity_id,
            retrievers=retrievers,
        )
        if rerank and self._reranker is not None:
            result = self._reranker.rerank_result(query, result, top_k)
        return result

    def find_path(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int = 5,
        min_confidence: float = 0.0,
    ) -> RetrievalResult:
        """路径查找."""
        return self.graph_retriever.find_path(
            source_id, target_id,
            max_depth=max_depth,
            min_confidence=min_confidence,
        )

    def get_neighbors(
        self,
        entity_id: str,
        *,
        direction: str = "both",
        min_confidence: float = 0.0,
        top_k: int = 20,
    ) -> RetrievalResult:
        """邻居查询."""
        return self.graph_retriever.get_neighbors(
            entity_id,
            direction=direction,
            min_confidence=min_confidence,
            top_k=top_k,
        )


__all__ = [
    "BaseRetriever",
    "VectorRetriever",
    "KeywordRetriever",
    "GraphRetriever",
    "HybridRetriever",
    "RetrievalEngine",
]
