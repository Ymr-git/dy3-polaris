"""L3 领域知识层 — 跨库对齐融合引擎.

融合世界先进方案的多源融合设计:
- GraphRAG: 社区感知 + 跨源实体对齐
- RRF (Reciprocal Rank Fusion): 多路检索结果融合
- DBpedia 质量框架: 多维度质量评估
- Wikidata 跨源对齐: 统一标识符映射
- Salesforce Cross-Encoder Rerank: 跨源重排

跨库对齐策略:
1. 实体对齐: 通过统一 kp_id 对齐三库结果 (向量/图谱/结构化)
2. RRF 融合: Score = Σ w_i / (k + rank_i(d))
3. 去重合并: 相同 kp_id 的结果合并，保留最高分
4. 来源标注: 记录每个结果的来源库 (vector/graph/exact)

三库对齐机制 (借鉴规划文档):
- Neo4j: KnowledgePoint.kp_id 节点属性
- Milvus: knowledge_point_ids ARRAY 字段 (一个 chunk 关联多 KP)
- PostgreSQL: fact_check_standards.kp_id 外键
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .models import RetrievalResult

logger = logging.getLogger(__name__)


# ============================================================
# 枚举与数据模型
# ============================================================


class SourceType(str, Enum):
    """知识来源类型 (借鉴 GraphRAG 多源检索).

    VECTOR: 向量库检索结果 (语义相似)
    GRAPH: 图谱库检索结果 (关系推理)
    EXACT: 结构化库检索结果 (精确匹配)
    KEYWORD: 关键词检索结果 (BM25)
    FUSED: 融合后结果 (多源合并)
    """

    VECTOR = "vector"
    GRAPH = "graph"
    EXACT = "exact"
    KEYWORD = "keyword"
    FUSED = "fused"


@dataclass
class AlignedItem:
    """对齐后的知识项 (借鉴 Wikidata 跨源对齐).

    Attributes:
        kp_id: 知识点 ID (对齐键)
        content: 内容文本
        score: 融合后分数
        sources: 来源列表 (多源时记录所有来源)
        source_scores: 各来源分数
        metadata: 元数据
    """

    kp_id: str
    content: str
    score: float
    sources: list[SourceType] = field(default_factory=list)
    source_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_count(self) -> int:
        """来源数量 (多源命中加分)."""
        return len(self.sources)

    @property
    def is_multi_source(self) -> bool:
        """是否多源命中."""
        return self.source_count > 1


@dataclass
class FusionConfig:
    """融合配置 (借鉴 RRF + GraphRAG 权重配置).

    Attributes:
        k: RRF 参数 (默认 60)
        weights: 各来源权重
        multi_source_boost: 多源命中加分
        min_score: 最低分数阈值
        max_results: 最大返回结果数
    """

    k: int = 60
    weights: dict[str, float] = field(default_factory=lambda: {
        "vector": 0.4,
        "graph": 0.3,
        "exact": 0.3,
        "keyword": 0.2,
    })
    multi_source_boost: float = 0.15
    min_score: float = 0.0
    max_results: int = 10


@dataclass
class AlignmentResult:
    """跨库对齐结果.

    Attributes:
        query: 原始查询
        items: 对齐后的知识项列表
        total_sources: 参与融合的来源数
        total_raw: 原始结果总数
        total_aligned: 对齐后结果数
        multi_source_count: 多源命中数
        fusion_time_ms: 融合耗时
        config: 融合配置
    """

    query: str
    items: list[AlignedItem] = field(default_factory=list)
    total_sources: int = 0
    total_raw: int = 0
    total_aligned: int = 0
    multi_source_count: int = 0
    fusion_time_ms: float = 0.0
    config: FusionConfig | None = None

    @property
    def results(self) -> list[dict[str, Any]]:
        """转换为检索结果格式."""
        return [
            {
                "kp_id": item.kp_id,
                "content": item.content,
                "score": item.score,
                "sources": [s.value for s in item.sources],
                "source_scores": item.source_scores,
                "is_multi_source": item.is_multi_source,
                "metadata": item.metadata,
            }
            for item in self.items
        ]

    @property
    def scores(self) -> list[float]:
        """分数列表."""
        return [item.score for item in self.items]

    @property
    def total(self) -> int:
        """结果总数."""
        return len(self.items)


# ============================================================
# 跨库对齐器
# ============================================================


class CrossDBAligner:
    """跨库对齐融合引擎 (借鉴 GraphRAG + RRF + Wikidata 对齐).

    将来自不同知识库 (向量/图谱/结构化) 的检索结果进行对齐和融合:
    1. 提取所有结果中的 kp_id 作为对齐键
    2. 按 kp_id 聚合各源结果
    3. RRF 融合计算最终分数
    4. 多源命中加分
    5. 去重排序返回

    Usage::

        from dy3_polaris.l3 import CrossDBAligner

        aligner = CrossDBAligner()

        # 添加各源检索结果
        aligner.add_source("vector", vector_results, scores)
        aligner.add_source("graph", graph_results, scores)
        aligner.add_source("exact", exact_results, scores)

        # 融合
        result = aligner.fuse(query="Dy3+跃迁波长")
        print(result.total)
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        """初始化跨库对齐器.

        Args:
            config: 融合配置 (默认 RRF k=60)
        """
        self.config = config or FusionConfig()
        self._sources: dict[str, list[tuple[dict[str, Any], float]]] = {}

    def add_source(
        self,
        source_type: str | SourceType,
        results: list[dict[str, Any]],
        scores: list[float],
    ) -> None:
        """添加一个来源的检索结果.

        Args:
            source_type: 来源类型
            results: 检索结果列表
            scores: 对应分数列表
        """
        if isinstance(source_type, SourceType):
            source_type = source_type.value

        self._sources[source_type] = list(zip(results, scores))

    def add_retrieval_result(
        self,
        source_type: str | SourceType,
        result: RetrievalResult,
    ) -> None:
        """添加 RetrievalResult 格式的检索结果.

        Args:
            source_type: 来源类型
            result: 检索结果
        """
        if isinstance(source_type, SourceType):
            source_type = source_type.value

        self._sources[source_type] = list(zip(result.results, result.scores))

    def fuse(self, query: str = "") -> AlignmentResult:
        """执行跨库融合.

        Args:
            query: 原始查询 (用于记录)

        Returns:
            对齐融合结果
        """
        start_time = time.time()

        # 如果没有来源，返回空结果
        if not self._sources:
            return AlignmentResult(
                query=query,
                config=self.config,
                fusion_time_ms=0.0,
            )

        # 1. 提取所有 kp_id 并建立对齐映射
        aligned: dict[str, AlignedItem] = {}
        total_raw = 0

        for source_type, items in self._sources.items():
            weight = self.config.weights.get(source_type, 0.2)

            for rank, (item, score) in enumerate(items):
                total_raw += 1

                # 提取 kp_id (多种可能的字段名)
                kp_id = (
                    item.get("kp_id")
                    or item.get("knowledge_point_id")
                    or item.get("kp_anchors")
                    or item.get("entity_id")
                    or item.get("chunk_id")
                    or f"unknown_{rank}_{source_type}"
                )

                # 如果 kp_id 是列表 (如 kp_anchors)，取第一个
                if isinstance(kp_id, list) and kp_id:
                    kp_id = kp_id[0]

                # RRF 分数: w / (k + rank + 1)
                rrf_score = weight / (self.config.k + rank + 1)

                if kp_id not in aligned:
                    aligned[kp_id] = AlignedItem(
                        kp_id=kp_id,
                        content=str(item.get("content", "")),
                        score=0.0,
                        sources=[],
                        source_scores={},
                        metadata=item.get("metadata", {}),
                    )

                # 累加 RRF 分数
                aligned[kp_id].score += rrf_score
                st_enum = SourceType(source_type) if source_type in [s.value for s in SourceType] else SourceType.FUSED
                if st_enum not in aligned[kp_id].sources:
                    aligned[kp_id].sources.append(st_enum)
                aligned[kp_id].source_scores[source_type] = rrf_score

                # 保留最长内容
                new_content = str(item.get("content", ""))
                if len(new_content) > len(aligned[kp_id].content):
                    aligned[kp_id].content = new_content

        # 2. 多源命中加分
        for item in aligned.values():
            if item.is_multi_source:
                item.score += self.config.multi_source_boost * item.source_count

        # 3. 过滤和排序
        items = [
            item for item in aligned.values()
            if item.score >= self.config.min_score
        ]
        items.sort(key=lambda x: x.score, reverse=True)
        items = items[: self.config.max_results]

        elapsed = (time.time() - start_time) * 1000
        multi_count = sum(1 for item in items if item.is_multi_source)

        return AlignmentResult(
            query=query,
            items=items,
            total_sources=len(self._sources),
            total_raw=total_raw,
            total_aligned=len(items),
            multi_source_count=multi_count,
            fusion_time_ms=round(elapsed, 2),
            config=self.config,
        )

    def clear(self) -> None:
        """清空已添加的来源数据."""
        self._sources.clear()

    @staticmethod
    def fuse_results(
        results: dict[str, RetrievalResult],
        *,
        config: FusionConfig | None = None,
        query: str = "",
    ) -> AlignmentResult:
        """静态方法: 一次性融合多个 RetrievalResult.

        Args:
            results: 来源类型 → RetrievalResult 映射
            config: 融合配置
            query: 原始查询

        Returns:
            对齐融合结果
        """
        aligner = CrossDBAligner(config)
        for source_type, result in results.items():
            aligner.add_retrieval_result(source_type, result)
        return aligner.fuse(query)


# ============================================================
# 质量加权融合器
# ============================================================


class QualityWeightedFuser:
    """质量加权融合器 (借鉴 DBpedia 质量框架 + GraphRAG 社区评分).

    在 RRF 融合基础上，根据知识来源质量等级加权:
    - T1 权威 (1.0): 国标/SCI 期刊
    - T2 推荐 (0.8): 行业报告/教材
    - T3 参考 (0.6): 网络资源/预印本
    - T4 补充 (0.4): 个人笔记/博客

    Usage::

        from dy3_polaris.l3 import QualityWeightedFuser

        fuser = QualityWeightedFuser()
        fuser.add_source("vector", results, scores, quality_tier=1)
        fuser.add_source("graph", results, scores, quality_tier=2)
        result = fuser.fuse("Dy3+发光机理")
    """

    QUALITY_WEIGHTS: dict[int, float] = {
        1: 1.0,  # T1 权威
        2: 0.8,  # T2 推荐
        3: 0.6,  # T3 参考
        4: 0.4,  # T4 补充
    }

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._sources: dict[str, list[tuple[dict[str, Any], float, int]]] = {}

    def add_source(
        self,
        source_type: str | SourceType,
        results: list[dict[str, Any]],
        scores: list[float],
        *,
        quality_tier: int = 2,
    ) -> None:
        """添加带质量等级的来源结果.

        Args:
            source_type: 来源类型
            results: 结果列表
            scores: 分数列表
            quality_tier: 质量等级 (1-4)
        """
        if isinstance(source_type, SourceType):
            source_type = source_type.value

        quality_weight = self.QUALITY_WEIGHTS.get(quality_tier, 0.6)
        self._sources[source_type] = [
            (r, s * quality_weight, quality_tier)
            for r, s in zip(results, scores)
        ]

    def fuse(self, query: str = "") -> AlignmentResult:
        """执行质量加权融合."""
        start_time = time.time()

        if not self._sources:
            return AlignmentResult(query=query, config=self.config)

        aligned: dict[str, AlignedItem] = {}
        total_raw = 0

        for source_type, items in self._sources.items():
            weight = self.config.weights.get(source_type, 0.2)

            for rank, (item, score, tier) in enumerate(items):
                total_raw += 1

                kp_id = (
                    item.get("kp_id")
                    or item.get("entity_id")
                    or item.get("chunk_id")
                    or f"unknown_{rank}_{source_type}"
                )

                if isinstance(kp_id, list) and kp_id:
                    kp_id = kp_id[0]

                # 质量加权 RRF 分数
                quality_weight = self.QUALITY_WEIGHTS.get(tier, 0.6)
                rrf_score = (weight * quality_weight) / (self.config.k + rank + 1)

                if kp_id not in aligned:
                    aligned[kp_id] = AlignedItem(
                        kp_id=kp_id,
                        content=str(item.get("content", "")),
                        score=0.0,
                        sources=[],
                        source_scores={},
                        metadata={**item.get("metadata", {}), "quality_tier": tier},
                    )

                aligned[kp_id].score += rrf_score
                st_enum = SourceType(source_type) if source_type in [s.value for s in SourceType] else SourceType.FUSED
                if st_enum not in aligned[kp_id].sources:
                    aligned[kp_id].sources.append(st_enum)
                aligned[kp_id].source_scores[source_type] = rrf_score

                new_content = str(item.get("content", ""))
                if len(new_content) > len(aligned[kp_id].content):
                    aligned[kp_id].content = new_content

        # 多源加分
        for item in aligned.values():
            if item.is_multi_source:
                item.score += self.config.multi_source_boost * item.source_count

        items = sorted(aligned.values(), key=lambda x: x.score, reverse=True)
        items = items[: self.config.max_results]

        elapsed = (time.time() - start_time) * 1000
        multi_count = sum(1 for item in items if item.is_multi_source)

        return AlignmentResult(
            query=query,
            items=items,
            total_sources=len(self._sources),
            total_raw=total_raw,
            total_aligned=len(items),
            multi_source_count=multi_count,
            fusion_time_ms=round(elapsed, 2),
            config=self.config,
        )

    def clear(self) -> None:
        """清空已添加的来源数据."""
        self._sources.clear()


__all__ = [
    "SourceType",
    "AlignedItem",
    "FusionConfig",
    "AlignmentResult",
    "CrossDBAligner",
    "QualityWeightedFuser",
]
