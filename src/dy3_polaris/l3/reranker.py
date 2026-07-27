"""L3 领域知识层 — 知识检索结果重排器.

融合世界先进方案的重排器设计:
- Carbonell & Goldstein (1998): MMR 最大边际相关性 (兼顾相关性与多样性)
- Elasticsearch: function_score + boosting query (多因子元数据加权)
- DBpedia/Wikidata: 质量评估框架 (多维质量分数加权)
- Google: freshness boost + decay functions (时间新颖性加权)
- GraphRAG: 社区检测 + PageRank 中心性 (图结构加权)
- Cohere: 多阶段重排 pipeline (组合重排)

重排策略:
1. MMRReranker               — 最大边际相关性 (兼顾相关性与多样性)
2. MetadataBoostReranker     — 元数据加权 (类型/领域/标签/验证/置信度)
3. QualityBoostReranker      — 质量分数加权 (多维质量评估)
4. RecencyBoostReranker      — 时间新颖性加权 (指数/高斯/线性衰减)
5. GraphCentralityReranker   — 图中心性加权 (度中心性 + 图距离 + 社区)
6. CompositeReranker          — 组合重排 (多阶段 pipeline)

所有重排器继承 BaseReranker，统一接口:
- rerank(query, results, top_k): 对 (dict, score) 列表重排
- rerank_result(query, result, top_k): 对 RetrievalResult 重排，返回新 RetrievalResult

设计理念:
- 借鉴 Cohere Rerank API: 召回阶段追求高召回率 (宽召回)，
  重排阶段追求高精确率 (精排序)，两阶段分离提升整体效果。
- 借鉴 LlamaIndex NodePostprocessor: 重排器作为检索 pipeline 的后处理节点，
  可组合、可插拔。
- 借鉴 Elasticsearch function_score: 多因子加权叠加，灵活控制各因子影响。
- 借鉴 Cohere 多阶段重排: 粗排 (元数据/质量加权) → 精排 (MMR 多样性)，
  逐步精炼结果。

Usage::

    from dy3_polaris.l3.reranker import (
        MMRReranker, MetadataBoostReranker,
        QualityBoostReranker, CompositeReranker,
    )

    # 单一重排器
    reranker = MMRReranker(lambda_=0.7)
    new_result = reranker.rerank_result(query, result, top_k=10)

    # 组合重排 pipeline
    pipeline = CompositeReranker([
        MetadataBoostReranker(type_weight={"chemical_compound": 0.3}),
        QualityBoostReranker(quality_weight=0.25),
        MMRReranker(lambda_=0.7),
    ])
    final_result = pipeline.rerank_result(query, result, top_k=10)
"""

from __future__ import annotations

import enum
import logging
import math
import re
import time
from typing import Any

from .models import KnowledgeGraph, RetrievalResult

logger = logging.getLogger(__name__)


# ============================================================
# 重排策略枚举
# ============================================================


class RerankStrategy(enum.Enum):
    """重排策略枚举.

    标识不同的重排策略，用于配置和日志追踪。

    Attributes:
        MMR: 最大边际相关性 (兼顾相关性与多样性)
        METADATA_BOOST: 元数据加权 (类型/领域/标签/验证/置信度)
        QUALITY_BOOST: 质量分数加权 (多维质量评估)
        RECENCY_BOOST: 时间新颖性加权 (衰减函数)
        GRAPH_BOOST: 图中心性加权 (度中心性 + 图距离 + 社区)
        COMPOSITE: 组合策略 (多阶段 pipeline)
    """

    MMR = "mmr"
    METADATA_BOOST = "metadata"
    QUALITY_BOOST = "quality"
    RECENCY_BOOST = "recency"
    GRAPH_BOOST = "graph"
    COMPOSITE = "composite"


# ============================================================
# 重排器抽象基类
# ============================================================


class BaseReranker:
    """重排器抽象基类 (借鉴 Cohere Rerank API 设计).

    定义重排器的统一接口。所有重排器继承此类并实现 rerank 方法。
    rerank_result 方法提供对 RetrievalResult 的便捷重排入口。

    设计理念 (借鉴 Cohere 两阶段检索):
        召回阶段 (Retriever): 宽召回，追求高召回率
        重排阶段 (Reranker): 精排序，追求高精确率

    子类需设置 ``strategy_name`` 类属性标识重排策略名称，并实现
    ``rerank`` 方法。``rerank_result`` 方法为通用实现，无需覆盖。

    Attributes:
        strategy_name: 重排策略名称 (子类设置)
    """

    strategy_name: str = "base"

    def rerank(
        self,
        query: str,
        results: list[tuple[dict[str, Any], float]],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """对检索结果进行重排 (子类实现).

        Args:
            query: 查询文本
            results: 检索结果列表 [(result_dict, score), ...]
            top_k: 返回前 k 个结果

        Returns:
            重排后的结果列表 [(result_dict, new_score), ...]
        """
        raise NotImplementedError("子类必须实现 rerank 方法")

    def rerank_result(
        self,
        query: str,
        result: RetrievalResult,
        top_k: int = 10,
    ) -> RetrievalResult:
        """对 RetrievalResult 进行重排，返回新的 RetrievalResult.

        从 RetrievalResult 中提取 (dict, score) 对，调用 rerank 方法
        进行重排，然后构造新的 RetrievalResult (更新 results 和 scores)。

        Args:
            query: 查询文本
            result: 原始检索结果
            top_k: 返回前 k 个结果

        Returns:
            重排后的新 RetrievalResult
        """
        start_time = time.time()

        # 提取 (dict, score) 对
        pairs = list(zip(result.results, result.scores))

        # 执行重排
        reranked = self.rerank(query, pairs, top_k=top_k)

        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug(
            "重排完成 [%s]: 输入 %d 条, 输出 %d 条, 耗时 %.2fms",
            self.strategy_name, len(pairs), len(reranked), elapsed_ms,
        )

        return RetrievalResult(
            query=result.query,
            results=[item for item, _ in reranked],
            scores=[score for _, score in reranked],
            total=len(reranked),
            retrieval_time_ms=round(elapsed_ms, 2),
            source_type=f"{result.source_type}+rerank:{self.strategy_name}",
            filters=result.filters,
            trace_id=result.trace_id,
        )

    # ---- 通用辅助方法 (供子类使用) ----

    @staticmethod
    def _get_text(doc: dict[str, Any]) -> str:
        """提取文档文本 (切片用 content, 实体用 name+description).

        Args:
            doc: 文档字典 (KnowledgeEntity 或 DocumentChunk 的序列化结果)

        Returns:
            文档文本字符串
        """
        if doc.get("content"):
            return str(doc["content"])
        parts: list[str] = []
        if doc.get("name"):
            parts.append(str(doc["name"]))
        if doc.get("description"):
            parts.append(str(doc["description"]))
        return " ".join(parts)

    @staticmethod
    def _get_vector(doc: dict[str, Any]) -> list[float] | None:
        """提取文档向量 (优先 embedding.vector, 其次顶层 vector).

        Args:
            doc: 文档字典

        Returns:
            向量列表，无向量时返回 None
        """
        emb = doc.get("embedding")
        if isinstance(emb, dict):
            vec = emb.get("vector")
            if isinstance(vec, list) and vec:
                return [float(v) for v in vec]
        vec = doc.get("vector")
        if isinstance(vec, list) and vec:
            return [float(v) for v in vec]
        return None

    @staticmethod
    def _get_quality(doc: dict[str, Any]) -> dict[str, Any] | None:
        """提取质量评分字典.

        Args:
            doc: 文档字典

        Returns:
            质量评分字典，无质量信息时返回 None
        """
        quality = doc.get("quality")
        if isinstance(quality, dict):
            return quality
        return None

    @staticmethod
    def _get_timestamp(doc: dict[str, Any]) -> float:
        """提取时间戳 (优先 updated_at, 其次 created_at).

        Args:
            doc: 文档字典

        Returns:
            Unix 时间戳，无时间信息时返回 0.0
        """
        ts = doc.get("updated_at", 0.0)
        if not ts:
            ts = doc.get("created_at", 0.0)
        try:
            return float(ts)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _get_entity_type(doc: dict[str, Any]) -> str:
        """提取实体类型 (返回枚举字符串值).

        Args:
            doc: 文档字典

        Returns:
            实体类型字符串 (如 "chemical_compound")，无类型时返回空字符串
        """
        return str(doc.get("entity_type", ""))

    @staticmethod
    def _get_domain(doc: dict[str, Any]) -> str:
        """提取领域.

        优先取顶层 domain 字段，其次从 metadata 中查找。

        Args:
            doc: 文档字典

        Returns:
            领域字符串，无领域信息时返回空字符串
        """
        domain = doc.get("domain", "")
        if not domain:
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                domain = metadata.get("domain", "")
        return str(domain)

    @staticmethod
    def _get_tags(doc: dict[str, Any]) -> list[str]:
        """提取标签列表.

        优先取顶层 tags 字段，其次从 metadata 中查找。

        Args:
            doc: 文档字典

        Returns:
            标签字符串列表
        """
        tags = doc.get("tags", [])
        if not tags:
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                tags = metadata.get("tags", [])
        if isinstance(tags, list):
            return [str(t) for t in tags]
        return []

    @staticmethod
    def _get_entity_id(doc: dict[str, Any]) -> str:
        """提取实体/文档标识 (优先 entity_id, 其次 chunk_id, 最后 document_id).

        Args:
            doc: 文档字典

        Returns:
            标识字符串，无标识时返回空字符串
        """
        for key in ("entity_id", "chunk_id", "document_id"):
            val = doc.get(key)
            if val:
                return str(val)
        return ""

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """简单分词 (小写化 + 词元提取, 支持中英文).

        使用正则表达式 \\w+ 匹配 Unicode 单词字符 (含中文)，
        小写化后返回词元集合。

        Args:
            text: 输入文本

        Returns:
            词元集合 (小写)
        """
        if not text:
            return set()
        return set(re.findall(r"\w+", text.lower()))

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度.

        Args:
            vec1: 向量 1
            vec2: 向量 2

        Returns:
            余弦相似度 [-1, 1]，维度不匹配或零向量时返回 0.0
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def _jaccard_similarity(set1: set[str], set2: set[str]) -> float:
        """计算 Jaccard 相似度 (基于词集合).

        Jaccard = |A ∩ B| / |A ∪ B|

        Args:
            set1: 词集合 1
            set2: 词集合 2

        Returns:
            Jaccard 相似度 [0, 1]，两个空集时返回 0.0
        """
        if not set1 and not set2:
            return 0.0
        union = set1 | set2
        if not union:
            return 0.0
        return len(set1 & set2) / len(union)

    @staticmethod
    def _normalize_scores(
        results: list[tuple[dict[str, Any], float]],
    ) -> list[tuple[dict[str, Any], float]]:
        """将分数 min-max 归一化到 [0, 1].

        Args:
            results: 检索结果列表

        Returns:
            归一化后的结果列表 (保持原始顺序)
        """
        if not results:
            return []
        scores = [s for _, s in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        if score_range == 0.0:
            return [(doc, 1.0) for doc, _ in results]
        return [
            (doc, (score - min_score) / score_range)
            for doc, score in results
        ]


# ============================================================
# MMR 重排器 — 最大边际相关性
# ============================================================


class MMRReranker(BaseReranker):
    """最大边际相关性重排器 (借鉴 Carbonell & Goldstein, 1998).

    MMR 在相关性和多样性之间取得平衡:
    - 高 λ: 更注重相关性 (结果可能与查询高度相关但彼此冗余)
    - 低 λ: 更注重多样性 (结果覆盖更广泛的信息，但单条相关性可能较低)

    MMR 公式::

        MMR(d) = argmax_{d ∈ R\\D} [λ * Sim(d, q) - (1-λ) * max_{d' ∈ D} Sim(d, d')]

    其中:
        - R: 所有候选文档集合
        - D: 已选文档集合
        - Sim(d, q): 文档与查询的相关性 (使用归一化后的基础分数)
        - Sim(d, d'): 文档间相似度
        - λ: 相关性-多样性权衡参数 (0-1)

    文档相似度计算方式:
        - 有向量时: 余弦相似度 (映射到 [0, 1])
        - 无向量时: Jaccard 相似度 (基于词集合)
        - 支持属性重叠度计算 (实体类型/领域/标签等结构化属性)

    Attributes:
        lambda_: 相关性-多样性权衡参数 (0=最大多样性, 1=最大相关性)
        use_attribute_overlap: 是否在相似度计算中考虑属性重叠度
    """

    strategy_name = "mmr"

    def __init__(
        self,
        lambda_: float = 0.7,
        use_attribute_overlap: bool = True,
    ) -> None:
        self.lambda_ = lambda_
        self.use_attribute_overlap = use_attribute_overlap

    def rerank(
        self,
        query: str,
        results: list[tuple[dict[str, Any], float]],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """MMR 重排.

        使用贪心算法迭代选择文档: 每轮选择使 MMR 分数最大的文档，
        直到选满 top_k 个或候选耗尽。

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回前 k 个结果

        Returns:
            MMR 重排后的结果列表 (保留原始分数)
        """
        if not results:
            logger.debug("MMR 重排: 输入为空, 返回空列表")
            return []

        if len(results) == 1:
            return list(results)

        logger.debug(
            "MMR 重排开始: %d 条候选, λ=%.2f, top_k=%d",
            len(results), self.lambda_, top_k,
        )

        # 归一化基础分数到 [0, 1] (作为 Sim(d, q))
        normalized = self._normalize_scores(results)

        # 预计算每个文档的向量、词集合和属性键集合 (避免重复计算)
        vectors: list[list[float] | None] = []
        token_sets: list[set[str]] = []
        attr_sets: list[set[str]] = []

        for doc, _ in normalized:
            vectors.append(self._get_vector(doc))
            token_sets.append(self._tokenize(self._get_text(doc)))
            if self.use_attribute_overlap:
                attr_sets.append(self._extract_attribute_keys(doc))
            else:
                attr_sets.append(set())

        # 贪心选择
        n = len(normalized)
        selected_indices: list[int] = []
        remaining = set(range(n))

        while remaining and len(selected_indices) < top_k:
            best_idx = -1
            best_mmr = -math.inf

            for idx in remaining:
                # 相关性项: λ * Sim(d, q)
                relevance = self.lambda_ * normalized[idx][1]

                # 多样性惩罚项: (1-λ) * max_{d' ∈ D} Sim(d, d')
                max_sim = 0.0
                for sel_idx in selected_indices:
                    sim = self._document_similarity(
                        idx, sel_idx, vectors, token_sets, attr_sets,
                    )
                    if sim > max_sim:
                        max_sim = sim

                mmr_score = relevance - (1.0 - self.lambda_) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.discard(best_idx)
            else:
                break

        # 返回选中的结果 (保留原始分数)
        reranked = [results[i] for i in selected_indices]

        logger.debug("MMR 重排完成: 输出 %d 条", len(reranked))
        return reranked

    def _document_similarity(
        self,
        idx1: int,
        idx2: int,
        vectors: list[list[float] | None],
        token_sets: list[set[str]],
        attr_sets: list[set[str]],
    ) -> float:
        """计算两个文档之间的相似度.

        优先使用向量余弦相似度 (映射到 [0, 1])，无向量时使用 Jaccard
        相似度 (基于词集合)。若启用属性重叠度，则加权融合文本相似度
        和属性相似度。

        Args:
            idx1: 文档 1 索引
            idx2: 文档 2 索引
            vectors: 预计算的向量列表
            token_sets: 预计算的词集合列表
            attr_sets: 预计算的属性键集合列表

        Returns:
            相似度分数 [0, 1]
        """
        vec1 = vectors[idx1]
        vec2 = vectors[idx2]

        # 向量余弦相似度
        if vec1 is not None and vec2 is not None:
            cos_sim = self._cosine_similarity(vec1, vec2)
            # 余弦相似度范围 [-1, 1]，映射到 [0, 1]
            text_sim = (cos_sim + 1.0) / 2.0
        else:
            # 无向量时使用 Jaccard 相似度
            text_sim = self._jaccard_similarity(
                token_sets[idx1], token_sets[idx2],
            )

        if not self.use_attribute_overlap:
            return text_sim

        # 属性重叠度
        attr_sim = self._jaccard_similarity(
            attr_sets[idx1], attr_sets[idx2],
        )

        # 加权融合: 文本相似度 0.7 + 属性相似度 0.3
        return 0.7 * text_sim + 0.3 * attr_sim

    @staticmethod
    def _extract_attribute_keys(doc: dict[str, Any]) -> set[str]:
        """提取文档的属性键集合 (用于属性重叠度计算).

        提取实体类型、领域、标签、内容类型、语言、标识符类型等
        结构化属性，构建属性键集合用于 Jaccard 相似度计算。

        Args:
            doc: 文档字典

        Returns:
            属性键集合 (如 {"type:chemical_compound", "domain:chemistry", ...})
        """
        attrs: set[str] = set()

        # 实体类型
        entity_type = doc.get("entity_type", "")
        if entity_type:
            attrs.add(f"type:{entity_type}")

        # 领域
        domain = doc.get("domain", "")
        if domain:
            attrs.add(f"domain:{domain}")

        # 标签
        tags = doc.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                attrs.add(f"tag:{tag}")

        # 内容类型 (切片)
        content_type = doc.get("content_type", "")
        if content_type:
            attrs.add(f"ctype:{content_type}")

        # 语言
        language = doc.get("language", "")
        if language:
            attrs.add(f"lang:{language}")

        # 标识符类型 (如 cas, doi 等)
        identifiers = doc.get("identifiers", {})
        if isinstance(identifiers, dict):
            for id_type in identifiers:
                attrs.add(f"id:{id_type}")

        return attrs


# ============================================================
# 元数据加权重排器 — Elasticsearch function_score
# ============================================================


class MetadataBoostReranker(BaseReranker):
    """元数据加权重排器 (借鉴 Elasticsearch function_score + boosting query).

    根据文档的元数据属性进行加权，提升特定类型、领域、标签等
    匹配条件的文档分数。

    支持以下加权因子:
        - type_weight: 实体类型权重 (如 CHEMICAL_COMPOUND 比 CONCEPT 权重高)
        - domain_weight: 领域权重
        - tag_boost: 标签匹配加权 (查询词与标签匹配时加权)
        - verified_boost: 已验证实体加权
        - confidence_weight: 置信度加权

    公式::

        final_score = base_score * (1 + Σ weight_i * factor_i)

    其中每个因子的计算方式:
        - type 因子: type_weight 字典中对应类型的 boost 值 (无则 0)
        - domain 因子: domain_weight 字典中对应领域的 boost 值 (无则 0)
        - tag 因子: 查询词与标签的匹配比例 * tag_boost
        - verified 因子: is_verified 为 True 时 * verified_boost
        - confidence 因子: confidence_score (0-1) * confidence_weight

    Attributes:
        type_weight: 实体类型权重映射 {type_string: boost_value}
        domain_weight: 领域权重映射 {domain_string: boost_value}
        tag_boost: 标签匹配加权系数
        verified_boost: 已验证加权系数
        confidence_weight: 置信度加权系数
    """

    strategy_name = "metadata_boost"

    # 默认实体类型权重 (借鉴知识图谱实体重要性分层)
    DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
        "chemical_compound": 0.30,
        "material": 0.25,
        "method": 0.20,
        "experiment": 0.20,
        "paper": 0.15,
        "dataset": 0.15,
        "textbook": 0.10,
        "course": 0.10,
        "person": 0.10,
        "organization": 0.10,
        "concept": 0.05,
        "document_chunk": 0.05,
    }

    def __init__(
        self,
        *,
        type_weight: dict[str, float] | None = None,
        domain_weight: dict[str, float] | None = None,
        tag_boost: float = 0.20,
        verified_boost: float = 0.30,
        confidence_weight: float = 0.15,
    ) -> None:
        self.type_weight = (
            type_weight if type_weight is not None
            else dict(self.DEFAULT_TYPE_WEIGHTS)
        )
        self.domain_weight = domain_weight or {}
        self.tag_boost = tag_boost
        self.verified_boost = verified_boost
        self.confidence_weight = confidence_weight

    def rerank(
        self,
        query: str,
        results: list[tuple[dict[str, Any], float]],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """元数据加权重排.

        对每个文档计算元数据加权总和，按公式调整分数后降序排序。

        Args:
            query: 查询文本 (用于标签匹配)
            results: 检索结果列表
            top_k: 返回前 k 个结果

        Returns:
            加权后的结果列表 (按新分数降序)
        """
        if not results:
            return []

        logger.debug(
            "元数据加权重排开始: %d 条候选, top_k=%d", len(results), top_k,
        )

        query_tokens = self._tokenize(query)

        boosted: list[tuple[dict[str, Any], float]] = []
        for doc, base_score in results:
            boost_sum = self._compute_boost(doc, query_tokens)
            new_score = base_score * (1.0 + boost_sum)
            boosted.append((doc, new_score))

        # 按新分数降序排序
        boosted.sort(key=lambda x: x[1], reverse=True)
        reranked = boosted[:top_k]

        logger.debug("元数据加权重排完成: 输出 %d 条", len(reranked))
        return reranked

    def _compute_boost(
        self,
        doc: dict[str, Any],
        query_tokens: set[str],
    ) -> float:
        """计算文档的元数据加权总和 (Σ weight_i * factor_i).

        Args:
            doc: 文档字典
            query_tokens: 查询词集合 (用于标签匹配)

        Returns:
            加权总和
        """
        boost_sum = 0.0

        # 1. 实体类型加权
        entity_type = self._get_entity_type(doc)
        if entity_type:
            boost_sum += self.type_weight.get(entity_type, 0.0)

        # 2. 领域加权
        domain = self._get_domain(doc)
        if domain:
            boost_sum += self.domain_weight.get(domain, 0.0)

        # 3. 标签匹配加权
        if self.tag_boost > 0.0 and query_tokens:
            tags = self._get_tags(doc)
            if tags:
                tag_tokens: set[str] = set()
                for tag in tags:
                    tag_tokens.update(self._tokenize(tag))
                if tag_tokens:
                    match_ratio = len(query_tokens & tag_tokens) / max(
                        len(tag_tokens), 1,
                    )
                    boost_sum += self.tag_boost * match_ratio

        # 4. 已验证加权
        if self.verified_boost > 0.0:
            is_verified = doc.get("is_verified", False)
            if is_verified:
                boost_sum += self.verified_boost

        # 5. 置信度加权
        if self.confidence_weight > 0.0:
            confidence = doc.get("confidence_score", 0.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            boost_sum += self.confidence_weight * confidence

        return boost_sum


# ============================================================
# 质量分数加权重排器 — DBpedia 质量评估框架
# ============================================================


class QualityBoostReranker(BaseReranker):
    """质量分数加权重排器 (借鉴 DBpedia 质量评估框架).

    使用实体/切片中的 quality 字段 (QualityScore 序列化字典) 进行加权。
    支持对 accuracy, completeness, consistency, timeliness, relevancy,
    trustworthiness 各维度进行独立加权，计算综合质量分数后对基础分数
    进行提升。

    综合质量分数计算::

        quality_score = Σ(w_i * dim_i) / Σ(w_i)

    最终分数::

        final_score = base_score * (1 + quality_weight * quality_score)

    当文档无质量信息时，quality_score 为 0.0 (不影响基础分数)。

    Attributes:
        quality_weight: 质量加权系数 (控制质量对最终分数的影响)
        dimension_weights: 各维度权重 {dimension_name: weight}
    """

    strategy_name = "quality_boost"

    # 默认维度权重 (借鉴 QualityScore._weights)
    DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
        "accuracy": 0.25,
        "trustworthiness": 0.20,
        "consistency": 0.15,
        "timeliness": 0.15,
        "completeness": 0.10,
        "relevancy": 0.15,
    }

    def __init__(
        self,
        *,
        quality_weight: float = 0.30,
        dimension_weights: dict[str, float] | None = None,
    ) -> None:
        self.quality_weight = quality_weight
        self.dimension_weights = (
            dimension_weights if dimension_weights is not None
            else dict(self.DEFAULT_DIMENSION_WEIGHTS)
        )

    def rerank(
        self,
        query: str,
        results: list[tuple[dict[str, Any], float]],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """质量加权重排.

        对每个文档计算综合质量分数，按公式调整基础分数后降序排序。

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回前 k 个结果

        Returns:
            质量加权后的结果列表 (按新分数降序)
        """
        if not results:
            return []

        logger.debug(
            "质量加权重排开始: %d 条候选, quality_weight=%.2f, top_k=%d",
            len(results), self.quality_weight, top_k,
        )

        boosted: list[tuple[dict[str, Any], float]] = []
        for doc, base_score in results:
            quality_score = self._compute_quality_score(doc)
            new_score = base_score * (1.0 + self.quality_weight * quality_score)
            boosted.append((doc, new_score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        reranked = boosted[:top_k]

        logger.debug("质量加权重排完成: 输出 %d 条", len(reranked))
        return reranked

    def _compute_quality_score(self, doc: dict[str, Any]) -> float:
        """计算文档的综合质量分数.

        优先使用序列化的 overall 字段 (来自 QualityScore.to_dict())，
        否则按维度权重逐维度加权计算。

        Args:
            doc: 文档字典

        Returns:
            综合质量分数 [0, 1]，无质量信息时返回 0.0
        """
        quality = self._get_quality(doc)
        if not quality:
            return 0.0

        # 若有序列化的 overall 字段，直接使用
        overall = quality.get("overall")
        if overall is not None:
            try:
                return max(0.0, min(1.0, float(overall)))
            except (TypeError, ValueError):
                logger.debug("overall 字段无法转换为浮点数: %r, 回退到逐维度计算", overall)

        # 逐维度加权计算
        weighted_sum = 0.0
        weight_total = 0.0
        for dim_name, weight in self.dimension_weights.items():
            dim_value = quality.get(dim_name)
            if dim_value is not None:
                try:
                    dim_value = float(dim_value)
                    dim_value = max(0.0, min(1.0, dim_value))
                    weighted_sum += weight * dim_value
                    weight_total += weight
                except (TypeError, ValueError):
                    continue

        if weight_total == 0.0:
            return 0.0
        return weighted_sum / weight_total


# ============================================================
# 时间新颖性加权重排器 — Google freshness + decay functions
# ============================================================


class RecencyBoostReranker(BaseReranker):
    """时间新颖性加权重排器 (借鉴 Google freshness boost + Elasticsearch decay functions).

    使用实体/切片的时间戳进行新颖性加权，越新的内容获得越高的分数提升。

    支持三种衰减函数:
        - exponential (指数衰减): boost = exp(-λ * age_days)
          其中 λ = ln(2) / half_life_days
        - gaussian (高斯衰减): boost = exp(-(age_days / σ)^2)
        - linear (线性衰减): boost = max(0, 1 - age_days / max_age)

    半衰期参数控制衰减速度: 内容新颖性减半所需的天数。
    对于指数衰减，半衰期与衰减常数的关系为 λ = ln(2) / half_life_days。

    Attributes:
        decay_function: 衰减函数类型 ("exponential"/"gaussian"/"linear")
        half_life_days: 半衰期 (天)，控制指数衰减速度
        sigma_days: 高斯衰减标准差 (天)，默认等于半衰期
        max_age_days: 线性衰减最大年龄 (天)，默认 4 倍半衰期
        now: 当前时间戳 (用于测试，默认为 time.time())
    """

    strategy_name = "recency_boost"

    def __init__(
        self,
        *,
        decay_function: str = "exponential",
        half_life_days: float = 30.0,
        sigma_days: float | None = None,
        max_age_days: float | None = None,
        now: float | None = None,
    ) -> None:
        self.decay_function = decay_function
        self.half_life_days = half_life_days
        # 高斯衰减 σ 默认等于半衰期
        self.sigma_days = sigma_days if sigma_days is not None else half_life_days
        # 线性衰减最大年龄默认等于 4 倍半衰期
        self.max_age_days = (
            max_age_days if max_age_days is not None
            else half_life_days * 4.0
        )
        self.now = now if now is not None else time.time()

    def rerank(
        self,
        query: str,
        results: list[tuple[dict[str, Any], float]],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """时间新颖性加权重排.

        对每个文档计算时间衰减因子，乘以基础分数后降序排序。

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回前 k 个结果

        Returns:
            时间加权后的结果列表 (按新分数降序)
        """
        if not results:
            return []

        logger.debug(
            "时间新颖性重排开始: %d 条候选, 衰减=%s, 半衰期=%.1f天, top_k=%d",
            len(results), self.decay_function, self.half_life_days, top_k,
        )

        boosted: list[tuple[dict[str, Any], float]] = []
        for doc, base_score in results:
            boost = self._compute_recency_boost(doc)
            new_score = base_score * boost
            boosted.append((doc, new_score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        reranked = boosted[:top_k]

        logger.debug("时间新颖性重排完成: 输出 %d 条", len(reranked))
        return reranked

    def _compute_recency_boost(self, doc: dict[str, Any]) -> float:
        """计算文档的时间新颖性加权因子.

        根据文档时间戳计算年龄 (天数)，再按选定的衰减函数
        计算加权因子。无时间戳信息时返回 1.0 (不加权)。

        Args:
            doc: 文档字典

        Returns:
            新颖性加权因子 (≥0，越新越接近 1.0)
        """
        timestamp = self._get_timestamp(doc)
        if timestamp <= 0.0:
            # 无时间戳信息，不加权
            return 1.0

        age_seconds = self.now - timestamp
        if age_seconds < 0.0:
            # 未来时间戳 (时钟偏移)，视为最新
            age_seconds = 0.0

        age_days = age_seconds / 86400.0  # 86400 秒 = 1 天

        if self.decay_function == "exponential":
            # 指数衰减: boost = exp(-λ * age_days)
            if self.half_life_days <= 0.0:
                return 1.0
            lambda_ = math.log(2.0) / self.half_life_days
            return math.exp(-lambda_ * age_days)

        if self.decay_function == "gaussian":
            # 高斯衰减: boost = exp(-(age_days / σ)^2)
            if self.sigma_days <= 0.0:
                return 1.0
            return math.exp(-((age_days / self.sigma_days) ** 2))

        if self.decay_function == "linear":
            # 线性衰减: boost = max(0, 1 - age_days / max_age)
            if self.max_age_days <= 0.0:
                return 1.0
            return max(0.0, 1.0 - age_days / self.max_age_days)

        # 未知衰减函数，回退到指数衰减
        logger.warning("未知衰减函数 '%s', 回退到指数衰减", self.decay_function)
        if self.half_life_days <= 0.0:
            return 1.0
        lambda_ = math.log(2.0) / self.half_life_days
        return math.exp(-lambda_ * age_days)


# ============================================================
# 图中心性加权重排器 — GraphRAG 社区检测 + PageRank
# ============================================================


class GraphCentralityReranker(BaseReranker):
    """图中心性加权重排器 (借鉴 GraphRAG 社区检测 + PageRank 中心性).

    基于知识图谱结构进行重排，综合考虑:
        - 度中心性 (degree centrality): 实体在图中连接越多，重要性越高
        - 图距离 (graph proximity): 候选实体与查询实体的图距离越近，分数越高
        - 社区归属 (community membership): 同一社区 (领域) 内的实体相互加权

    公式::

        final_score = base_score * (1 + α * centrality + β * proximity + γ * community)

    其中:
        - α: 中心性权重系数
        - β: 邻近性权重系数
        - γ: 社区归属加权系数
        - centrality: 归一化度中心性 [0, 1]
        - proximity: 与查询实体的图邻近性 [0, 1] (1/(1+distance))
        - community: 社区归属因子 [0, 1]

    若未提供 KnowledgeGraph，则从检索结果中实体序列化的 triples 字段
    构建局部图结构。

    Attributes:
        graph: 知识图谱 (可选，若提供则使用其完整结构)
        alpha: 中心性权重系数
        beta: 邻近性权重系数
        community_weight: 社区归属加权系数
        max_graph_distance: BFS 最大搜索深度
    """

    strategy_name = "graph_boost"

    def __init__(
        self,
        *,
        graph: KnowledgeGraph | None = None,
        alpha: float = 0.20,
        beta: float = 0.30,
        community_weight: float = 0.15,
        max_graph_distance: int = 5,
    ) -> None:
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.community_weight = community_weight
        self.max_graph_distance = max_graph_distance
        # 预计算邻接表和度中心性 (仅当提供 KnowledgeGraph 时)
        self._adjacency: dict[str, set[str]] = {}
        self._degree_centrality: dict[str, float] = {}
        if graph is not None:
            self._build_graph_structure()

    def _build_graph_structure(self) -> None:
        """从 KnowledgeGraph 构建邻接表和度中心性.

        遍历实体内部三元组和跨实体三元组，构建无向图的邻接表，
        然后计算每个节点的归一化度中心性。
        """
        if self.graph is None:
            return

        # 构建邻接表 (无向图)
        adj: dict[str, set[str]] = {
            eid: set() for eid in self.graph.entities
        }

        # 实体内部三元组
        for entity in self.graph.entities.values():
            for triple in entity.triples:
                if triple.object_id and not triple.object_is_literal:
                    if triple.object_id in adj:
                        adj[entity.entity_id].add(triple.object_id)
                        adj[triple.object_id].add(entity.entity_id)

        # 跨实体三元组
        for triple in self.graph.triples:
            if triple.object_id and not triple.object_is_literal:
                sid = triple.subject_id
                oid = triple.object_id
                if sid not in adj:
                    adj[sid] = set()
                if oid not in adj:
                    adj[oid] = set()
                adj[sid].add(oid)
                adj[oid].add(sid)

        self._adjacency = adj

        # 计算度中心性 (归一化: degree / (n-1))
        n = len(adj)
        for eid, neighbors in adj.items():
            degree = len(neighbors)
            if n > 1:
                self._degree_centrality[eid] = degree / (n - 1)
            else:
                self._degree_centrality[eid] = 0.0

        max_degree = max(
            (len(neighbors) for neighbors in adj.values()),
            default=0,
        )
        logger.debug(
            "图结构构建完成: %d 节点, 最大度=%d", n, max_degree,
        )

    def _build_adjacency_from_results(
        self,
        results: list[tuple[dict[str, Any], float]],
    ) -> tuple[dict[str, set[str]], dict[str, float]]:
        """从检索结果中构建邻接表和度中心性 (无外部 KnowledgeGraph 时使用).

        利用实体结果中序列化的 triples 字段构建邻接关系。

        Args:
            results: 检索结果列表

        Returns:
            (邻接表, 度中心性字典) 元组
        """
        adj: dict[str, set[str]] = {}

        for doc, _ in results:
            entity_id = self._get_entity_id(doc)
            if not entity_id:
                continue
            if entity_id not in adj:
                adj[entity_id] = set()

            # 从序列化的 triples 构建边
            triples = doc.get("triples", [])
            if isinstance(triples, list):
                for triple in triples:
                    if not isinstance(triple, dict):
                        continue
                    object_id = triple.get("object_id", "")
                    object_is_literal = triple.get("object_is_literal", False)
                    if object_id and not object_is_literal:
                        adj[entity_id].add(object_id)
                        if object_id not in adj:
                            adj[object_id] = set()
                        adj[object_id].add(entity_id)

        # 计算度中心性
        degree_centrality: dict[str, float] = {}
        n = len(adj)
        for eid, neighbors in adj.items():
            degree = len(neighbors)
            if n > 1:
                degree_centrality[eid] = degree / (n - 1)
            else:
                degree_centrality[eid] = 0.0

        logger.debug("从结果构建图结构: %d 节点", n)
        return adj, degree_centrality

    def rerank(
        self,
        query: str,
        results: list[tuple[dict[str, Any], float]],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """图中心性加权重排.

        对每个文档计算图结构加权 (度中心性 + 图邻近性 + 社区归属)，
        按公式调整基础分数后降序排序。

        Args:
            query: 查询文本 (用于定位查询实体)
            results: 检索结果列表
            top_k: 返回前 k 个结果

        Returns:
            图加权后的结果列表 (按新分数降序)
        """
        if not results:
            return []

        logger.debug(
            "图中心性重排开始: %d 条候选, α=%.2f, β=%.2f, top_k=%d",
            len(results), self.alpha, self.beta, top_k,
        )

        # 构建图结构
        if self.graph is not None:
            adjacency = self._adjacency
            degree_centrality = self._degree_centrality
        else:
            adjacency, degree_centrality = self._build_adjacency_from_results(
                results,
            )

        # 无法构建图结构时，直接返回原始结果
        if not adjacency:
            logger.debug("图中心性重排: 无法构建图结构, 返回原始结果")
            return results[:top_k]

        # 定位查询实体 (在图中匹配查询文本)
        query_entity_ids = self._find_query_entities(query, results, adjacency)

        # BFS 计算查询实体的图距离 → 邻近性
        proximity_map = self._compute_proximity(query_entity_ids, adjacency)

        # 收集查询实体领域 (用于社区加权)
        query_domains = self._collect_query_domains(
            query_entity_ids, results,
        )

        # 图加权重排
        boosted: list[tuple[dict[str, Any], float]] = []
        for doc, base_score in results:
            entity_id = self._get_entity_id(doc)
            centrality = degree_centrality.get(entity_id, 0.0)
            proximity = proximity_map.get(entity_id, 0.0)
            community = self._compute_community_boost(doc, query_domains)

            boost = (
                self.alpha * centrality
                + self.beta * proximity
                + self.community_weight * community
            )
            new_score = base_score * (1.0 + boost)
            boosted.append((doc, new_score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        reranked = boosted[:top_k]

        logger.debug("图中心性重排完成: 输出 %d 条", len(reranked))
        return reranked

    def _find_query_entities(
        self,
        query: str,
        results: list[tuple[dict[str, Any], float]],
        adjacency: dict[str, set[str]],
    ) -> list[str]:
        """在图中定位与查询相关的实体.

        策略:
        1. 若有 KnowledgeGraph，按名称/别名/描述匹配查询文本
        2. 若无匹配，使用检索结果中得分最高且存在于图中的实体作为锚点

        Args:
            query: 查询文本
            results: 检索结果列表
            adjacency: 邻接表

        Returns:
            查询实体 ID 列表
        """
        query_lower = query.lower().strip()
        query_entity_ids: list[str] = []

        # 策略 1: 在 KnowledgeGraph 中匹配
        if self.graph is not None:
            for entity in self.graph.entities.values():
                if entity.match_name_or_alias(query):
                    query_entity_ids.append(entity.entity_id)
                elif query_lower and (
                    query_lower in entity.name.lower()
                    or query_lower in entity.description.lower()
                ):
                    query_entity_ids.append(entity.entity_id)

        # 策略 2: 使用检索结果中前几个在图中存在的实体作为锚点
        if not query_entity_ids:
            for doc, _score in results[:5]:
                entity_id = self._get_entity_id(doc)
                if entity_id and entity_id in adjacency:
                    query_entity_ids.append(entity_id)

        return query_entity_ids

    def _compute_proximity(
        self,
        query_entity_ids: list[str],
        adjacency: dict[str, set[str]],
    ) -> dict[str, float]:
        """多源 BFS 计算所有节点到查询实体的图距离，转换为邻近性分数.

        proximity = 1 / (1 + distance)
        距离为 0 (查询实体自身) 时 proximity = 1.0

        Args:
            query_entity_ids: 查询实体 ID 列表 (BFS 源点)
            adjacency: 邻接表

        Returns:
            {entity_id: proximity_score} 映射
        """
        if not query_entity_ids or not adjacency:
            return {}

        # 多源 BFS (使用索引指针避免 O(n) 的 pop(0))
        distances: dict[str, int] = {}
        queue: list[str] = list(query_entity_ids)
        for qid in query_entity_ids:
            distances[qid] = 0

        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            current_dist = distances[current]
            if current_dist >= self.max_graph_distance:
                continue
            for neighbor in adjacency.get(current, set()):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        # 距离转邻近性
        return {
            eid: 1.0 / (1.0 + dist) for eid, dist in distances.items()
        }

    def _collect_query_domains(
        self,
        query_entity_ids: list[str],
        results: list[tuple[dict[str, Any], float]],
    ) -> set[str]:
        """收集查询实体所属的领域集合 (用于社区归属判断).

        从检索结果和 KnowledgeGraph 中收集查询实体的领域信息。

        Args:
            query_entity_ids: 查询实体 ID 列表
            results: 检索结果列表

        Returns:
            领域字符串集合
        """
        domains: set[str] = set()
        qid_set = set(query_entity_ids)

        # 从检索结果中收集
        for doc, _ in results:
            entity_id = self._get_entity_id(doc)
            if entity_id in qid_set:
                domain = self._get_domain(doc)
                if domain:
                    domains.add(domain)

        # 从 KnowledgeGraph 中补充
        if self.graph is not None:
            for qid in query_entity_ids:
                entity = self.graph.get_entity(qid)
                if entity:
                    domains.add(entity.domain)

        return domains

    def _compute_community_boost(
        self,
        doc: dict[str, Any],
        query_domains: set[str],
    ) -> float:
        """计算社区归属加权因子.

        若文档所属领域与查询实体领域一致，则返回 1.0 (同一社区)，
        否则返回 0.0。领域作为社区的近似划分。

        Args:
            doc: 文档字典
            query_domains: 查询实体领域集合

        Returns:
            社区归属因子 [0, 1]
        """
        if not query_domains:
            return 0.0
        doc_domain = self._get_domain(doc)
        if doc_domain and doc_domain in query_domains:
            return 1.0
        return 0.0


# ============================================================
# 组合重排器 — Cohere 多阶段重排 pipeline
# ============================================================


class CompositeReranker(BaseReranker):
    """组合重排器 (借鉴 Cohere 多阶段重排 pipeline).

    将多个重排器按管道顺序串联应用，每个重排器对前一个的输出进行重排。
    最终输出 top_k 结果。

    典型用法::

        reranker = CompositeReranker([
            MetadataBoostReranker(...),
            QualityBoostReranker(...),
            MMRReranker(lambda_=0.7),
        ])
        result = reranker.rerank_result(query, retrieval_result, top_k=10)

    管道设计理念 (借鉴 Cohere 多阶段重排):
        1. 粗排阶段: MetadataBoost / QualityBoost 等加权排序 (宽保留)
        2. 精排阶段: MMR 多样性重排 (窄截断)
        每个阶段输出作为下一阶段输入，逐步精炼结果。

    Attributes:
        rerankers: 重排器列表 (按应用顺序)
        top_k_per_stage: 每阶段保留的结果数 (None=中间阶段不截断)
    """

    strategy_name = "composite"

    def __init__(
        self,
        rerankers: list[BaseReranker],
        *,
        top_k_per_stage: int | None = None,
    ) -> None:
        self.rerankers = rerankers
        self.top_k_per_stage = top_k_per_stage

    def rerank(
        self,
        query: str,
        results: list[tuple[dict[str, Any], float]],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """组合重排 (管道顺序应用各重排器).

        按列表顺序依次调用每个重排器的 rerank 方法，前一阶段的输出
        作为后一阶段的输入。最终阶段使用指定的 top_k 截断结果。

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 最终返回前 k 个结果

        Returns:
            组合重排后的结果列表
        """
        if not results:
            return []

        if not self.rerankers:
            return results[:top_k]

        logger.debug(
            "组合重排开始: %d 条候选, %d 个重排器, top_k=%d",
            len(results), len(self.rerankers), top_k,
        )

        current_results = list(results)

        for i, reranker in enumerate(self.rerankers):
            is_last = i == len(self.rerankers) - 1

            # 确定本阶段的 top_k
            if is_last:
                # 最终阶段使用请求的 top_k
                stage_top_k = top_k
            elif self.top_k_per_stage is not None:
                # 中间阶段按配置截断
                stage_top_k = min(self.top_k_per_stage, len(current_results))
            else:
                # 中间阶段不截断 (保留所有结果供下一阶段处理)
                stage_top_k = len(current_results)

            logger.debug(
                "组合重排阶段 %d/%d [%s]: 输入 %d 条, top_k=%d",
                i + 1, len(self.rerankers), reranker.strategy_name,
                len(current_results), stage_top_k,
            )

            current_results = reranker.rerank(
                query, current_results, top_k=stage_top_k,
            )

        logger.debug("组合重排完成: 输出 %d 条", len(current_results))
        return current_results


__all__ = [
    "RerankStrategy",
    "BaseReranker",
    "MMRReranker",
    "MetadataBoostReranker",
    "QualityBoostReranker",
    "RecencyBoostReranker",
    "GraphCentralityReranker",
    "CompositeReranker",
]
