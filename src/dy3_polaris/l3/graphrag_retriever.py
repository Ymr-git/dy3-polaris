"""L3 领域知识层 — GraphRAG 双通道检索融合引擎.

融合世界先进方案的 GraphRAG 检索引擎:
- Microsoft GraphRAG (Edge et al., 2024): 层次化社区 + 社区报告 + 全局检索
- OMD-GraphRAG (Open Multi-Document GraphRAG): 多文档子图融合 + 自适应检索
- PageRank (Brin & Page, 1998): 实体中心性排序
- RRF (Reciprocal Rank Fusion): 多通道检索结果融合
- GraphRAG Local Search: 实体中心子图提取 + 关系遍历

双通道检索架构:
1. Local Search (局部搜索):
   - 从查询实体出发，按置信度加权 BFS 提取子图
   - 基于 PageRank 中心性对子图实体排序
   - 记录遍历路径，支持可解释推理
   - 三种子图提取策略: HOP_BASED / CONFIDENCE_WEIGHTED / COMMUNITY_AWARE

2. Global Search (全局搜索):
   - 基于社区摘要 (Community Reports) 的全文/关键词匹配
   - 跨社区连接发现 (桥接实体识别)
   - 支持层次化社区聚合检索

融合策略:
- LOCAL_ONLY: 仅局部搜索 (查询包含具体实体名)
- GLOBAL_ONLY: 仅全局搜索 (宏观/主题类查询)
- ADAPTIVE: 自适应融合 (根据查询特征自动选择)
- ENSEMBLE: 双通道结果 RRF 融合 (最大化召回)

设计借鉴:
- Microsoft GraphRAG 的 map-reduce 全局摘要范式
- OMD-GraphRAG 的自适应检索路由
- Neo4j GDS 的 PageRank + 社区检测组合
- LlamaIndex 的检索器组合模式

所有算法仅依赖 Python 标准库实现 (numpy 可选加速)。
线程安全 (threading.RLock)。
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any

from pydantic import BaseModel, Field

from .community import (
    Community,
    CommunityAlgorithm,
    CommunityDetector,
    CommunityHierarchy,
)
from .models import (
    EntityType,
    KnowledgeEntity,
    KnowledgeTriple,
    SubgraphConfig,
)
from .store import KnowledgeStore

logger = logging.getLogger(__name__)

# numpy 可选加速 (零外部依赖原则下，缺失则回退纯 Python)
try:
    import numpy as np  # type: ignore[import-not-found]

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - 可选依赖
    _HAS_NUMPY = False


# ============================================================
# 子图提取策略与融合策略枚举
# ============================================================


class SubgraphStrategy(str):
    """子图提取策略标识 (字符串常量).

    借鉴 Microsoft GraphRAG local search 与 OMD-GraphRAG 模式:
    - HOP_BASED: 固定跳数扩展 (简单高效，适用于稠密图)
    - CONFIDENCE_WEIGHTED: 置信度加权扩展 (优先高置信关系)
    - COMMUNITY_AWARE: 社区感知扩展 (优先同社区实体)
    """

    HOP_BASED = "hop_based"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    COMMUNITY_AWARE = "community_aware"


class FusionStrategy(str):
    """双通道融合策略标识 (字符串常量).

    - LOCAL_ONLY: 仅局部搜索
    - GLOBAL_ONLY: 仅全局搜索
    - ADAPTIVE: 自适应融合 (查询包含具体实体名 -> 局部优先，否则 -> 全局优先)
    - ENSEMBLE: 双通道结果 RRF 融合
    """

    LOCAL_ONLY = "local_only"
    GLOBAL_ONLY = "global_only"
    ADAPTIVE = "adaptive"
    ENSEMBLE = "ensemble"


# ============================================================
# 数据类: 子图提取结果 / 社区摘要 / 检索结果
# ============================================================


class ExtractedSubgraph(BaseModel):
    """子图提取结果 (借鉴 Microsoft GraphRAG Extracted Subgraph).

    从查询实体出发，按策略提取的局部子图。
    包含实体集合、三元组列表、根实体、社区 ID 与统计信息。

    Attributes:
        entities: 子图包含的实体 {entity_id: KnowledgeEntity}
        triples: 子图包含的三元组列表
        root_entities: 提取起点的根实体 ID 列表
        community_ids: 子图触及的社区 ID 集合
        extraction_stats: 提取统计信息 (跳数、剪枝数等)
    """

    entities: dict[str, KnowledgeEntity] = Field(
        default_factory=dict, description="子图实体 {entity_id: KnowledgeEntity}"
    )
    triples: list[KnowledgeTriple] = Field(
        default_factory=list, description="子图三元组列表"
    )
    root_entities: list[str] = Field(
        default_factory=list, description="提取起点根实体 ID 列表"
    )
    community_ids: set[int] = Field(
        default_factory=set, description="子图触及的社区 ID 集合"
    )
    extraction_stats: dict[str, Any] = Field(
        default_factory=dict, description="提取统计信息"
    )

    model_config = {"arbitrary_types_allowed": True}


class CommunitySummary(BaseModel):
    """社区摘要 (借鉴 GraphRAG Community Reports).

    为单个社区生成的结构化摘要，包含关键实体、核心关系、
    属性统计、主题标签与摘要文本。

    Attributes:
        community_id: 社区唯一标识
        entity_count: 社区实体数
        triple_count: 社区三元组数
        key_entities: 关键实体名列表 (最多 10 个)
        core_relations: 核心关系描述列表 (最多 10 个)
        property_stats: 属性统计 {属性名: 统计值}
        topic_tags: 主题标签列表
        summary_text: 摘要文本
        level: 层级 (0=最细)
    """

    community_id: int = Field(..., description="社区唯一标识")
    entity_count: int = Field(default=0, ge=0, description="社区实体数")
    triple_count: int = Field(default=0, ge=0, description="社区三元组数")
    key_entities: list[str] = Field(
        default_factory=list, description="关键实体名列表 (最多 10 个)"
    )
    core_relations: list[str] = Field(
        default_factory=list, description="核心关系描述列表 (最多 10 个)"
    )
    property_stats: dict[str, Any] = Field(
        default_factory=dict, description="属性统计"
    )
    topic_tags: list[str] = Field(
        default_factory=list, description="主题标签列表"
    )
    summary_text: str = Field(default="", description="摘要文本")
    level: int = Field(default=0, ge=0, description="层级 (0=最细)")


class LocalSearchResult(BaseModel):
    """局部搜索结果.

    Attributes:
        root_entities: 查询起始根实体 ID 列表
        subgraph: 提取的子图
        ranked_entities: 排序后的实体列表 (entity_id, relevance_score)
        entity_details: 排序后的实体详情字典列表
        traversal_paths: 遍历路径记录列表
    """

    root_entities: list[str] = Field(
        default_factory=list, description="查询起始根实体 ID 列表"
    )
    subgraph: ExtractedSubgraph = Field(
        default_factory=ExtractedSubgraph, description="提取的子图"
    )
    ranked_entities: list[tuple[str, float]] = Field(
        default_factory=list, description="排序后的实体 (entity_id, score)"
    )
    entity_details: list[dict[str, Any]] = Field(
        default_factory=list, description="排序后的实体详情列表"
    )
    traversal_paths: list[dict[str, Any]] = Field(
        default_factory=list, description="遍历路径记录"
    )

    model_config = {"arbitrary_types_allowed": True}


class GlobalSearchResult(BaseModel):
    """全局搜索结果.

    Attributes:
        relevant_communities: 相关社区 (community_id, relevance_score)
        community_summaries: 相关社区的摘要列表
        cross_community_connections: 跨社区连接记录
    """

    relevant_communities: list[tuple[int, float]] = Field(
        default_factory=list, description="相关社区 (community_id, score)"
    )
    community_summaries: list[CommunitySummary] = Field(
        default_factory=list, description="相关社区摘要列表"
    )
    cross_community_connections: list[dict[str, Any]] = Field(
        default_factory=list, description="跨社区连接记录"
    )


class GraphRAGResult(BaseModel):
    """GraphRAG 双通道检索最终结果.

    Attributes:
        query: 原始查询文本
        strategy: 使用的融合策略
        local_results: 局部搜索结果 (None 表示未执行)
        global_results: 全局搜索结果 (None 表示未执行)
        fused_results: RRF 融合后的最终结果列表
        community_summaries: 相关社区摘要列表
        reasoning_context: 为 LLM 生成的推理上下文文本
        search_time_ms: 总检索耗时 (毫秒)
    """

    query: str = Field(..., description="原始查询文本")
    strategy: str = Field(default=FusionStrategy.ADAPTIVE, description="融合策略")
    local_results: LocalSearchResult | None = Field(
        default=None, description="局部搜索结果"
    )
    global_results: GlobalSearchResult | None = Field(
        default=None, description="全局搜索结果"
    )
    fused_results: list[dict[str, Any]] = Field(
        default_factory=list, description="RRF 融合后的最终结果"
    )
    community_summaries: list[CommunitySummary] = Field(
        default_factory=list, description="相关社区摘要列表"
    )
    reasoning_context: str = Field(
        default="", description="为 LLM 生成的推理上下文文本"
    )
    search_time_ms: float = Field(default=0.0, ge=0.0, description="总检索耗时 (ms)")

    model_config = {"arbitrary_types_allowed": True}


# ============================================================
# 简化 PageRank 实现 (借鉴 Brin & Page, 1998)
# ============================================================


def _simplified_pagerank(
    adjacency: dict[str, list[str]],
    weights: dict[str, dict[str, float]] | None = None,
    damping: float = 0.85,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """简化的 PageRank 实现 (借鉴 Brin & Page, 1998).

    用于局部搜索中实体的中心性排序。零外部依赖 (numpy 可选加速)。

    算法说明:
        PageRank 迭代公式:
            PR(v) = (1 - d) / N + d * Σ_{u∈In(v)} PR(u) / OutDegree(u)
        其中:
        - d: 阻尼系数 (damping), 通常 0.85, 表示用户以 1-d 概率随机跳转
        - N: 图中节点总数
        - In(v): 指向 v 的节点集合
        - OutDegree(u): 节点 u 的出度 (考虑权重时为出边权重之和)

    设计要点:
    - 悬挂节点 (dangling node, 无出边) 的 PR 均匀分发给所有节点，
      避免排名泄漏 (借鉴 PageRank 悬挂节点处理)。
    - 支持加权边 (weights[u][v] = w)，未提供权重时默认 1.0。
    - 收敛判定: L1 范数变化 < tolerance 或达到最大迭代数。
    - numpy 可用时使用矩阵运算加速 (向量化)。

    Args:
        adjacency: 邻接表 {entity_id: [neighbor_id, ...]} (有向图)
        weights: 可选边权重 {u: {v: w}}; None 表示等权
        damping: 阻尼系数, 默认 0.85
        max_iterations: 最大迭代次数, 默认 50
        tolerance: 收敛阈值 (L1 范数), 默认 1e-6

    Returns:
        {entity_id: pagerank_score} 字典, 分数和约为 1.0。
        空图返回空字典。
    """
    nodes = list(adjacency.keys())
    n = len(nodes)
    if n == 0:
        return {}

    node_index = {nid: i for i, nid in enumerate(nodes)}

    # 构建出度与转移权重
    if _HAS_NUMPY:
        # numpy 向量化实现
        transition = np.zeros((n, n), dtype=np.float64)
        for u, neighbors in adjacency.items():
            ui = node_index[u]
            if not neighbors:
                continue
            if weights and u in weights:
                w_map = weights[u]
                total_w = sum(w_map.get(v, 1.0) for v in neighbors)
                if total_w <= 0.0:
                    continue
                for v in neighbors:
                    vi = node_index[v]
                    transition[vi, ui] = w_map.get(v, 1.0) / total_w
            else:
                deg = len(neighbors)
                for v in neighbors:
                    vi = node_index[v]
                    transition[vi, ui] = 1.0 / deg

        # 悬挂节点检测
        out_sum = transition.sum(axis=0)
        dangling = np.where(out_sum == 0.0)[0]

        ranks = np.full(n, 1.0 / n, dtype=np.float64)
        teleport = (1.0 - damping) / n

        for _ in range(max_iterations):
            # 悬挂节点 PR 均匀分发
            dangling_sum = ranks[dangling].sum() if len(dangling) > 0 else 0.0
            new_ranks = teleport + damping * (
                transition @ ranks + dangling_sum / n
            )
            diff = np.abs(new_ranks - ranks).sum()
            ranks = new_ranks
            if diff < tolerance:
                break

        total = ranks.sum()
        if total > 0:
            ranks = ranks / total
        return {nodes[i]: float(ranks[i]) for i in range(n)}

    # 纯 Python 实现 (无 numpy)
    out_weight: dict[str, float] = {}
    norm_transition: dict[str, dict[str, float]] = defaultdict(dict)
    for u, neighbors in adjacency.items():
        if not neighbors:
            out_weight[u] = 0.0
            continue
        if weights and u in weights:
            w_map = weights[u]
            total_w = sum(w_map.get(v, 1.0) for v in neighbors)
            if total_w <= 0.0:
                out_weight[u] = 0.0
                continue
            for v in neighbors:
                norm_transition[v][u] = w_map.get(v, 1.0) / total_w
            out_weight[u] = total_w
        else:
            deg = len(neighbors)
            for v in neighbors:
                norm_transition[v][u] = 1.0 / deg
            out_weight[u] = float(deg)

    dangling_nodes = [u for u in nodes if out_weight[u] == 0.0]
    ranks = {nid: 1.0 / n for nid in nodes}
    teleport = (1.0 - damping) / n

    for _ in range(max_iterations):
        dangling_sum = sum(ranks[u] for u in dangling_nodes)
        new_ranks: dict[str, float] = {}
        diff = 0.0
        for v in nodes:
            incoming = norm_transition.get(v, {})
            s = sum(ranks[u] * w for u, w in incoming.items())
            new_ranks[v] = teleport + damping * (s + dangling_sum / n)
            diff += abs(new_ranks[v] - ranks[v])
        ranks = new_ranks
        if diff < tolerance:
            break

    total = sum(ranks.values())
    if total > 0:
        ranks = {k: v / total for k, v in ranks.items()}
    return ranks


# ============================================================
# SubgraphExtractor — 子图提取器
# ============================================================


class SubgraphExtractor:
    """子图提取器 (GraphRAG local search + OMD-GraphRAG 模式).

    从查询实体出发，按置信度加权的 BFS 提取相关子图。
    支持三种策略:
    - HOP_BASED: 固定跳数扩展 (简单高效)
    - CONFIDENCE_WEIGHTED: 置信度加权扩展 (借用 graph_reasoner_v2 的理念)
    - COMMUNITY_AWARE: 社区感知扩展 (优先扩展同社区实体)

    设计借鉴:
    - Microsoft GraphRAG local search 的实体中心子图提取
    - OMD-GraphRAG 的多文档子图聚合
    - Neo4j GDS 的子图遍历 API

    线程安全: 所有可变状态受 RLock 保护。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def extract(
        self,
        query_entities: list[str],
        store: KnowledgeStore,
        max_hops: int = 3,
        strategy: str = "community_aware",
        min_confidence: float = 0.3,
        max_entities: int = 100,
        community_map: dict[str, int] | None = None,
    ) -> ExtractedSubgraph:
        """从查询实体出发提取子图.

        Args:
            query_entities: 查询实体 ID 列表 (提取起点)
            store: 知识存储
            max_hops: 最大跳数 (BFS 深度), 默认 3
            strategy: 提取策略
                ("hop_based"/"confidence_weighted"/"community_aware")
            min_confidence: 关系置信度下限, 默认 0.3
            max_entities: 子图实体数上限, 默认 100
            community_map: 实体 -> 社区 ID 映射 (COMMUNITY_AWARE 策略需要)

        Returns:
            ExtractedSubgraph 提取的子图

        Note:
            - strategy 不区分大小写
            - COMMUNITY_AWARE 策略未提供 community_map 时退化为
              CONFIDENCE_WEIGHTED
        """
        with self._lock:
            return self._extract_impl(
                query_entities,
                store,
                max_hops,
                strategy,
                min_confidence,
                max_entities,
                community_map,
            )

    def _extract_impl(
        self,
        query_entities: list[str],
        store: KnowledgeStore,
        max_hops: int,
        strategy: str,
        min_confidence: float,
        max_entities: int,
        community_map: dict[str, int] | None,
    ) -> ExtractedSubgraph:
        """子图提取实现 (内部，调用方已加锁)."""
        strategy_norm = (strategy or "").lower()
        if strategy_norm not in (
            SubgraphStrategy.HOP_BASED,
            SubgraphStrategy.CONFIDENCE_WEIGHTED,
            SubgraphStrategy.COMMUNITY_AWARE,
        ):
            logger.warning(
                "未知子图提取策略 %r，回退为 hop_based", strategy
            )
            strategy_norm = SubgraphStrategy.HOP_BASED

        # COMMUNITY_AWARE 无映射则退化为置信度加权
        if (
            strategy_norm == SubgraphStrategy.COMMUNITY_AWARE
            and not community_map
        ):
            logger.debug(
                "COMMUNITY_AWARE 未提供 community_map，"
                "退化为 CONFIDENCE_WEIGHTED"
            )
            strategy_norm = SubgraphStrategy.CONFIDENCE_WEIGHTED

        entities: dict[str, KnowledgeEntity] = {}
        triples: list[KnowledgeTriple] = []
        seen_triple_ids: set[str] = set()
        community_ids: set[int] = set()
        pruned_count = 0

        # 有效根实体 (store 中存在)
        root_entities: list[str] = []
        for qid in query_entities:
            ent = store.get_entity(qid)
            if ent is None:
                logger.debug("查询实体 %r 不存在，跳过", qid)
                continue
            entities[qid] = ent
            root_entities.append(qid)
            if community_map and qid in community_map:
                community_ids.add(community_map[qid])

        if not root_entities:
            return ExtractedSubgraph(
                entities={},
                triples=[],
                root_entities=[],
                community_ids=set(),
                extraction_stats={
                    "strategy": strategy_norm,
                    "max_hops": max_hops,
                    "pruned": 0,
                    "visited": 0,
                },
            )

        # BFS 扩展
        # frontier: (entity_id, hop, accumulated_confidence)
        frontier: deque[tuple[str, int, float]] = deque()
        for rid in root_entities:
            frontier.append((rid, 0, 1.0))

        visited: set[str] = set(root_entities)
        max_hop_reached = 0

        while frontier and len(entities) < max_entities:
            current_id, hop, acc_conf = frontier.popleft()
            if hop >= max_hops:
                continue
            max_hop_reached = max(max_hop_reached, hop)

            # 获取当前实体的关联三元组
            related = store.get_entity_triples(
                current_id,
                direction="both",
                min_confidence=min_confidence,
            )

            for triple in related:
                # 跳过已收录三元组
                if triple.triple_id in seen_triple_ids:
                    continue
                seen_triple_ids.add(triple.triple_id)
                triples.append(triple)

                # 确定邻居实体 ID (跳过字面值)
                neighbor_id = ""
                if triple.subject_id == current_id and not triple.object_is_literal:
                    neighbor_id = triple.object_id
                elif triple.object_id == current_id:
                    neighbor_id = triple.subject_id
                else:
                    # current_id 可能是隐式参与，跳过
                    continue

                if not neighbor_id or neighbor_id in visited:
                    continue

                # 策略判定: 是否扩展此邻居
                edge_conf = triple.confidence
                new_acc = acc_conf * edge_conf

                if strategy_norm == SubgraphStrategy.HOP_BASED:
                    # 固定跳数: 只要置信度达标即扩展
                    should_expand = edge_conf >= min_confidence
                elif strategy_norm == SubgraphStrategy.CONFIDENCE_WEIGHTED:
                    # 置信度加权: 累积置信度衰减判定
                    should_expand = new_acc >= min_confidence
                elif strategy_norm == SubgraphStrategy.COMMUNITY_AWARE:
                    # 社区感知: 优先同社区，跨社区需更高置信度
                    cur_comm = community_map.get(current_id) if community_map else None
                    nb_comm = community_map.get(neighbor_id) if community_map else None
                    if cur_comm is not None and nb_comm is not None:
                        if cur_comm == nb_comm:
                            should_expand = edge_conf >= min_confidence
                        else:
                            # 跨社区: 提高阈值
                            should_expand = edge_conf >= max(
                                min_confidence, 0.6
                            ) and new_acc >= min_confidence
                    else:
                        should_expand = new_acc >= min_confidence
                else:  # pragma: no cover - 已校验
                    should_expand = edge_conf >= min_confidence

                if not should_expand:
                    pruned_count += 1
                    continue

                if len(entities) >= max_entities:
                    pruned_count += 1
                    continue

                neighbor_ent = store.get_entity(neighbor_id)
                if neighbor_ent is None:
                    pruned_count += 1
                    continue

                visited.add(neighbor_id)
                entities[neighbor_id] = neighbor_ent
                if community_map and neighbor_id in community_map:
                    community_ids.add(community_map[neighbor_id])
                frontier.append((neighbor_id, hop + 1, new_acc))

        return ExtractedSubgraph(
            entities=entities,
            triples=triples,
            root_entities=root_entities,
            community_ids=community_ids,
            extraction_stats={
                "strategy": strategy_norm,
                "max_hops": max_hops,
                "max_hop_reached": max_hop_reached,
                "pruned": pruned_count,
                "visited": len(entities),
                "triple_count": len(triples),
                "community_count": len(community_ids),
            },
        )


# ============================================================
# CommunitySummarizer — 社区摘要器
# ============================================================


class CommunitySummarizer:
    """社区摘要器 (GraphRAG 社区报告生成).

    为每个社区生成结构化摘要，支持:
    - RULE_BASED: 规则生成 (从实体属性+三元组模式生成)
    - LLM_READY: 生成结构化 prompt 供 LLM 生成摘要

    社区摘要包含: 关键实体列表、核心关系、属性统计、主题标签。

    设计借鉴:
    - Microsoft GraphRAG 的社区报告 (Community Reports) 生成
    - GraphRAG map-reduce 全局摘要范式
    - LlamaIndex 的社区摘要 prompt 模板

    线程安全: 所有可变状态受 RLock 保护。
    """

    # 关键实体与核心关系的上限
    MAX_KEY_ENTITIES = 10
    MAX_CORE_RELATIONS = 10

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def summarize_community(
        self,
        community: Community,
        store: KnowledgeStore,
        strategy: str = "rule_based",
    ) -> CommunitySummary:
        """为单个社区生成结构化摘要.

        Args:
            community: 待摘要的社区
            store: 知识存储 (用于获取实体与三元组详情)
            strategy: 摘要策略 ("rule_based"/"llm_ready")

        Returns:
            CommunitySummary 社区摘要
        """
        with self._lock:
            return self._summarize_impl(community, store, strategy)

    def _summarize_impl(
        self,
        community: Community,
        store: KnowledgeStore,
        strategy: str,
    ) -> CommunitySummary:
        """社区摘要实现 (内部，调用方已加锁)."""
        strategy_norm = (strategy or "").lower()
        if strategy_norm not in ("rule_based", "llm_ready"):
            logger.warning("未知摘要策略 %r，回退为 rule_based", strategy)
            strategy_norm = "rule_based"

        # 收集实体
        entities: list[KnowledgeEntity] = []
        for eid in community.entity_ids:
            ent = store.get_entity(eid)
            if ent is not None:
                entities.append(ent)

        # 收集三元组
        triples: list[KnowledgeTriple] = []
        for tid in community.triple_ids:
            tr = store.get_triple(tid)
            if tr is not None:
                triples.append(tr)

        # 关键实体: 按置信度+三元组数排序
        entity_scores: list[tuple[KnowledgeEntity, float]] = []
        for ent in entities:
            ent_triple_count = len(ent.triples)
            score = ent.confidence_score + 0.1 * ent_triple_count
            entity_scores.append((ent, score))
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        key_entities = [
            e.name for e, _ in entity_scores[: self.MAX_KEY_ENTITIES]
        ]

        # 核心关系: 按置信度排序的三元组描述
        triple_sorted = sorted(
            triples, key=lambda t: t.confidence, reverse=True
        )
        core_relations: list[str] = []
        seen_rel: set[str] = set()
        for tr in triple_sorted:
            subj = store.get_entity(tr.subject_id)
            obj_ent = (
                store.get_entity(tr.object_id)
                if tr.object_id and not tr.object_is_literal
                else None
            )
            subj_name = subj.name if subj else tr.subject_id
            obj_name = (
                obj_ent.name
                if obj_ent
                else (
                    str(tr.object_value)
                    if tr.object_is_literal
                    else tr.object_id
                )
            )
            desc = f"{subj_name} --[{tr.predicate}]--> {obj_name}"
            if desc not in seen_rel:
                seen_rel.add(desc)
                core_relations.append(desc)
            if len(core_relations) >= self.MAX_CORE_RELATIONS:
                break

        # 属性统计
        property_stats = self._compute_property_stats(entities, triples)

        # 主题标签: 从实体 tags + entity_type + domain 聚合
        topic_tags = self._extract_topic_tags(entities)

        # 摘要文本
        if strategy_norm == "llm_ready":
            summary_text = self._build_llm_ready_prompt(
                community, entities, triples, key_entities, core_relations
            )
        else:
            summary_text = self._build_rule_based_summary(
                community, entities, triples, key_entities, core_relations
            )

        return CommunitySummary(
            community_id=community.community_id,
            entity_count=len(entities),
            triple_count=len(triples),
            key_entities=key_entities,
            core_relations=core_relations,
            property_stats=property_stats,
            topic_tags=topic_tags,
            summary_text=summary_text,
            level=community.level,
        )

    def _compute_property_stats(
        self,
        entities: list[KnowledgeEntity],
        triples: list[KnowledgeTriple],
    ) -> dict[str, Any]:
        """计算社区属性统计.

        统计内容:
        - entity_type_distribution: 实体类型分布
        - domain_distribution: 领域分布
        - avg_confidence: 平均三元组置信度
        - predicate_distribution: 谓词分布
        - verified_ratio: 已验证实体比例
        """
        type_dist: dict[str, int] = defaultdict(int)
        domain_dist: dict[str, int] = defaultdict(int)
        pred_dist: dict[str, int] = defaultdict(int)
        verified_count = 0

        for ent in entities:
            type_dist[ent.entity_type.value] += 1
            domain_dist[ent.domain] += 1
            if ent.is_verified:
                verified_count += 1

        for tr in triples:
            pred_dist[tr.predicate] += 1

        avg_conf = (
            sum(t.confidence for t in triples) / len(triples)
            if triples
            else 0.0
        )
        verified_ratio = (
            verified_count / len(entities) if entities else 0.0
        )

        return {
            "entity_type_distribution": dict(type_dist),
            "domain_distribution": dict(domain_dist),
            "predicate_distribution": dict(pred_dist),
            "avg_confidence": round(avg_conf, 4),
            "verified_ratio": round(verified_ratio, 4),
        }

    def _extract_topic_tags(self, entities: list[KnowledgeEntity]) -> list[str]:
        """从实体聚合主题标签 (tags + domain 去重)."""
        tag_counter: dict[str, int] = defaultdict(int)
        for ent in entities:
            for tag in ent.tags:
                tag_counter[tag] += 1
            if ent.domain and ent.domain != "general":
                tag_counter[ent.domain] += 1
        # 按频次降序
        sorted_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_tags[:8]]

    def _build_rule_based_summary(
        self,
        community: Community,
        entities: list[KnowledgeEntity],
        triples: list[KnowledgeTriple],
        key_entities: list[str],
        core_relations: list[str],
    ) -> str:
        """规则生成摘要文本 (RULE_BASED 策略)."""
        lines: list[str] = []
        lines.append(
            f"# 社区 #{community.community_id} 摘要 (层级 L{community.level})"
        )
        lines.append(
            f"包含 {len(entities)} 个实体, {len(triples)} 条三元组。"
        )
        if key_entities:
            lines.append("关键实体: " + ", ".join(key_entities))
        if core_relations:
            lines.append("核心关系:")
            for rel in core_relations:
                lines.append(f"  - {rel}")
        if entities:
            types = defaultdict(int)
            for e in entities:
                types[e.entity_type.value] += 1
            type_str = ", ".join(f"{k}({v})" for k, v in types.items())
            lines.append(f"实体类型分布: {type_str}")
        return "\n".join(lines)

    def _build_llm_ready_prompt(
        self,
        community: Community,
        entities: list[KnowledgeEntity],
        triples: list[KnowledgeTriple],
        key_entities: list[str],
        core_relations: list[str],
    ) -> str:
        """生成结构化 prompt 供 LLM 生成摘要 (LLM_READY 策略)."""
        lines: list[str] = []
        lines.append("你是一名知识图谱分析专家。请基于以下社区信息生成结构化摘要。")
        lines.append("")
        lines.append(f"## 社区元信息")
        lines.append(f"- 社区 ID: {community.community_id}")
        lines.append(f"- 层级: L{community.level}")
        lines.append(f"- 实体数: {len(entities)}")
        lines.append(f"- 三元组数: {len(triples)}")
        lines.append("")
        lines.append("## 关键实体")
        for name in key_entities:
            lines.append(f"- {name}")
        lines.append("")
        lines.append("## 核心关系")
        for rel in core_relations:
            lines.append(f"- {rel}")
        lines.append("")
        lines.append("## 请输出")
        lines.append("1. 社区主题 (一句话)")
        lines.append("2. 关键发现 (3-5 条)")
        lines.append("3. 潜在应用场景")
        return "\n".join(lines)

    def generate_global_summary(
        self,
        communities: list[CommunitySummary],
        store: KnowledgeStore,
    ) -> str:
        """生成全局摘要 (借鉴 GraphRAG map-reduce 全局摘要).

        将多个社区摘要聚合为全局知识概览文本。
        采用 reduce 阶段: 按层级与主题聚合，输出结构化全局摘要。

        Args:
            communities: 社区摘要列表
            store: 知识存储

        Returns:
            全局摘要文本
        """
        with self._lock:
            if not communities:
                return "（无社区可生成全局摘要）"

            # 按层级分组
            by_level: dict[int, list[CommunitySummary]] = defaultdict(list)
            for cs in communities:
                by_level[cs.level].append(cs)

            lines: list[str] = []
            lines.append("# GraphRAG 全局知识概览")
            lines.append(
                f"共聚合 {len(communities)} 个社区，分布于 "
                f"{len(by_level)} 个层级。"
            )
            lines.append("")

            # 全局主题标签聚合
            global_tags: dict[str, int] = defaultdict(int)
            for cs in communities:
                for tag in cs.topic_tags:
                    global_tags[tag] += 1
            if global_tags:
                top_tags = sorted(
                    global_tags.items(), key=lambda x: x[1], reverse=True
                )[:10]
                lines.append(
                    "全局主题: "
                    + ", ".join(t for t, _ in top_tags)
                )
                lines.append("")

            # 按层级输出
            for level in sorted(by_level.keys()):
                level_summaries = by_level[level]
                total_entities = sum(cs.entity_count for cs in level_summaries)
                total_triples = sum(cs.triple_count for cs in level_summaries)
                lines.append(f"## 层级 L{level} ({len(level_summaries)} 个社区)")
                lines.append(
                    f"实体总数: {total_entities}, 三元组总数: {total_triples}"
                )
                for cs in level_summaries[:5]:  # 每层最多展示 5 个
                    lines.append(
                        f"- 社区 #{cs.community_id}: "
                        f"{cs.summary_text.split(chr(10))[0] if cs.summary_text else ''}"
                    )
                if len(level_summaries) > 5:
                    lines.append(f"  ... 其余 {len(level_summaries) - 5} 个社区略")
                lines.append("")

            return "\n".join(lines)


# ============================================================
# GraphRAGRetriever — GraphRAG 双通道检索器 (核心类)
# ============================================================


class GraphRAGRetriever:
    """GraphRAG 双通道检索融合器 (Microsoft GraphRAG + OMD-GraphRAG).

    双通道设计:
    - Local Search (局部搜索): 子图提取 + 实体排序 + 关系遍历
    - Global Search (全局搜索): 社区摘要检索 + 跨社区推理

    融合策略:
    - LOCAL_ONLY: 仅局部搜索
    - GLOBAL_ONLY: 仅全局搜索
    - ADAPTIVE: 自适应融合 (查询包含具体实体名 -> 局部优先，否则 -> 全局优先)
    - ENSEMBLE: 双通道结果 RRF 融合

    设计借鉴:
    - Microsoft GraphRAG 的 local + global 双通道检索
    - OMD-GraphRAG 的自适应检索路由
    - RRF (Reciprocal Rank Fusion) 多路结果融合
    - Neo4j GDS 的 PageRank 中心性排序

    线程安全: 所有可变状态受 RLock 保护。
    """

    # RRF 融合参数
    RRF_K: int = 60
    # 自适应策略: 局部优先时的实体匹配阈值
    ADAPTIVE_ENTITY_MATCH_THRESHOLD: int = 1
    # 默认局部搜索跳数
    DEFAULT_MAX_HOPS: int = 3
    # 默认子图实体上限
    DEFAULT_MAX_ENTITIES: int = 100
    # 默认最低置信度
    DEFAULT_MIN_CONFIDENCE: float = 0.3

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        *,
        community_map: dict[str, int] | None = None,
        communities: list[Community] | None = None,
    ) -> None:
        """初始化 GraphRAG 检索器.

        Args:
            store: 默认知识存储 (search 时可覆盖)
            community_map: 实体 -> 社区 ID 映射 (可选，用于社区感知子图提取)
            communities: 社区列表 (可选，用于全局搜索)
        """
        self._store = store
        self._community_map = dict(community_map) if community_map else {}
        self._communities: list[Community] = (
            list(communities) if communities else []
        )
        self._extractor = SubgraphExtractor()
        self._summarizer = CommunitySummarizer()
        # 社区摘要缓存 {community_id: CommunitySummary}
        self._summary_cache: dict[int, CommunitySummary] = {}
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 配置与状态管理
    # --------------------------------------------------------

    def set_store(self, store: KnowledgeStore) -> None:
        """设置默认知识存储."""
        with self._lock:
            self._store = store

    def set_community_map(self, community_map: dict[str, int]) -> None:
        """设置实体 -> 社区 ID 映射."""
        with self._lock:
            self._community_map = dict(community_map)

    def set_communities(self, communities: list[Community]) -> None:
        """设置社区列表 (用于全局搜索)."""
        with self._lock:
            self._communities = list(communities)
            self._summary_cache.clear()  # 社区变更，清空摘要缓存

    def clear_cache(self) -> None:
        """清空社区摘要缓存."""
        with self._lock:
            self._summary_cache.clear()

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        store: KnowledgeStore | None = None,
        *,
        strategy: str = FusionStrategy.ADAPTIVE,
        entity_ids: list[str] | None = None,
        communities: list[Community] | None = None,
        max_hops: int = DEFAULT_MAX_HOPS,
        top_k: int = 10,
        **kwargs: Any,
    ) -> GraphRAGResult:
        """主入口: 执行 GraphRAG 双通道检索.

        自动根据策略选择局部/全局/融合检索。

        Args:
            query: 查询文本
            store: 知识存储 (None 使用初始化时设置的)
            strategy: 融合策略
                ("local_only"/"global_only"/"adaptive"/"ensemble")
            entity_ids: 查询实体 ID 列表 (局部搜索起点);
                None 时自动从查询解析
            communities: 社区列表 (全局搜索); None 使用内置社区
            max_hops: 局部搜索最大跳数
            top_k: 返回结果数上限
            **kwargs: 额外参数 (min_confidence, max_entities 等)

        Returns:
            GraphRAGResult 检索结果
        """
        start_time = time.time()
        target_store = store or self._store
        if target_store is None:
            return GraphRAGResult(
                query=query,
                strategy=strategy,
                reasoning_context="错误: 未提供知识存储。",
                search_time_ms=0.0,
            )

        target_communities = (
            communities if communities is not None else self._communities
        )

        strategy_norm = (strategy or "").lower()
        if strategy_norm not in (
            FusionStrategy.LOCAL_ONLY,
            FusionStrategy.GLOBAL_ONLY,
            FusionStrategy.ADAPTIVE,
            FusionStrategy.ENSEMBLE,
        ):
            logger.warning("未知融合策略 %r，回退为 adaptive", strategy)
            strategy_norm = FusionStrategy.ADAPTIVE

        # 自适应: 分析查询选择策略
        if strategy_norm == FusionStrategy.ADAPTIVE:
            strategy_norm = self._select_adaptive_strategy(
                query, entity_ids, target_store
            )

        local_result: LocalSearchResult | None = None
        global_result: GlobalSearchResult | None = None
        fused: list[dict[str, Any]] = []
        community_summaries: list[CommunitySummary] = []

        # 解析查询实体 (若未提供)
        resolved_entity_ids = entity_ids
        if (
            strategy_norm
            in (FusionStrategy.LOCAL_ONLY, FusionStrategy.ENSEMBLE)
            and not resolved_entity_ids
        ):
            resolved_entity_ids = self._resolve_query_entities(
                query, target_store
            )

        # 执行局部搜索
        if strategy_norm in (FusionStrategy.LOCAL_ONLY, FusionStrategy.ENSEMBLE):
            if resolved_entity_ids:
                local_result = self.local_search(
                    query,
                    resolved_entity_ids,
                    target_store,
                    max_hops=max_hops,
                    top_k=top_k,
                    **kwargs,
                )

        # 执行全局搜索
        if strategy_norm in (FusionStrategy.GLOBAL_ONLY, FusionStrategy.ENSEMBLE):
            if target_communities:
                global_result = self.global_search(
                    query, target_communities, target_store, top_k=top_k
                )
                community_summaries = global_result.community_summaries

        # RRF 融合 (ENSEMBLE)
        if strategy_norm == FusionStrategy.ENSEMBLE:
            fused = self._rrf_fuse(local_result, global_result, top_k=top_k)
        elif strategy_norm == FusionStrategy.LOCAL_ONLY and local_result:
            fused = [
                {"entity_id": eid, "score": sc, "source": "local"}
                for eid, sc in local_result.ranked_entities[:top_k]
            ]
        elif strategy_norm == FusionStrategy.GLOBAL_ONLY and global_result:
            fused = [
                {
                    "community_id": cid,
                    "score": sc,
                    "source": "global",
                }
                for cid, sc in global_result.relevant_communities[:top_k]
            ]

        # 生成推理上下文
        reasoning_context = self._build_reasoning_context(
            query, strategy_norm, local_result, global_result, fused
        )

        elapsed = (time.time() - start_time) * 1000
        return GraphRAGResult(
            query=query,
            strategy=strategy_norm,
            local_results=local_result,
            global_results=global_result,
            fused_results=fused,
            community_summaries=community_summaries,
            reasoning_context=reasoning_context,
            search_time_ms=round(elapsed, 2),
        )

    # --------------------------------------------------------
    # 局部搜索
    # --------------------------------------------------------

    def local_search(
        self,
        query: str,
        entity_ids: list[str],
        store: KnowledgeStore,
        max_hops: int = DEFAULT_MAX_HOPS,
        *,
        top_k: int = 10,
        strategy: str = "community_aware",
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_entities: int = DEFAULT_MAX_ENTITIES,
    ) -> LocalSearchResult:
        """局部搜索: 从查询实体提取子图，基于 PageRank 中心性排序.

        流程 (借鉴 Microsoft GraphRAG local search):
        1. SubgraphExtractor 提取相关子图
        2. 构建子图邻接表与边权重
        3. _simplified_pagerank 计算实体中心性
        4. 按中心性排序输出实体详情
        5. 记录遍历路径 (可解释性)

        Args:
            query: 查询文本 (用于结果记录)
            entity_ids: 查询实体 ID 列表 (提取起点)
            store: 知识存储
            max_hops: 最大跳数
            top_k: 返回实体数上限
            strategy: 子图提取策略
            min_confidence: 最低置信度
            max_entities: 子图实体数上限

        Returns:
            LocalSearchResult 局部搜索结果
        """
        with self._lock:
            # 1. 提取子图
            subgraph = self._extractor.extract(
                query_entities=entity_ids,
                store=store,
                max_hops=max_hops,
                strategy=strategy,
                min_confidence=min_confidence,
                max_entities=max_entities,
                community_map=self._community_map,
            )

            # 2. 构建邻接表与边权重
            adjacency, weights = self._build_adjacency_from_subgraph(subgraph)

            # 3. PageRank 中心性排序
            if adjacency:
                pagerank_scores = _simplified_pagerank(
                    adjacency,
                    weights=weights,
                    damping=0.85,
                    max_iterations=50,
                )
            else:
                pagerank_scores = {eid: 0.0 for eid in subgraph.entities}

            # 根实体加权 (查询起点给予初始提升)
            root_set = set(subgraph.root_entities)
            ranked: list[tuple[str, float]] = []
            for eid, score in pagerank_scores.items():
                boost = 1.5 if eid in root_set else 1.0
                ranked.append((eid, score * boost))
            ranked.sort(key=lambda x: x[1], reverse=True)
            ranked = ranked[:top_k]

            # 4. 实体详情
            entity_details: list[dict[str, Any]] = []
            for eid, score in ranked:
                ent = subgraph.entities.get(eid)
                if ent is None:
                    continue
                entity_details.append(
                    {
                        "entity_id": eid,
                        "name": ent.name,
                        "entity_type": ent.entity_type.value,
                        "description": ent.description[:200]
                        if ent.description
                        else "",
                        "domain": ent.domain,
                        "confidence_score": ent.confidence_score,
                        "tags": list(ent.tags),
                        "relevance_score": round(score, 6),
                        "is_root": eid in root_set,
                    }
                )

            # 5. 遍历路径 (从根实体到各排序实体的最短跳数)
            traversal_paths = self._compute_traversal_paths(
                subgraph.root_entities, adjacency, ranked
            )

            return LocalSearchResult(
                root_entities=subgraph.root_entities,
                subgraph=subgraph,
                ranked_entities=ranked,
                entity_details=entity_details,
                traversal_paths=traversal_paths,
            )

    # --------------------------------------------------------
    # 全局搜索
    # --------------------------------------------------------

    def global_search(
        self,
        query: str,
        communities: list[Community],
        store: KnowledgeStore,
        *,
        top_k: int = 10,
    ) -> GlobalSearchResult:
        """全局搜索: 基于社区摘要的全文/关键词匹配，返回相关社区.

        流程 (借鉴 Microsoft GraphRAG global search):
        1. 为每个社区生成摘要 (缓存)
        2. 提取查询关键词
        3. 关键词匹配社区摘要 (BM25 风格打分)
        4. 返回 top_k 相关社区
        5. 识别跨社区连接 (桥接实体)

        Args:
            query: 查询文本
            communities: 候选社区列表
            store: 知识存储
            top_k: 返回社区数上限

        Returns:
            GlobalSearchResult 全局搜索结果
        """
        with self._lock:
            # 1. 生成/获取社区摘要
            summaries: list[CommunitySummary] = []
            for comm in communities:
                cs = self._get_or_create_summary(comm, store)
                summaries.append(cs)

            if not summaries:
                return GlobalSearchResult(
                    relevant_communities=[],
                    community_summaries=[],
                    cross_community_connections=[],
                )

            # 2. 查询关键词提取
            query_terms = self._extract_query_terms(query)
            query_term_set = set(query_terms)

            # 3. 关键词匹配打分 (BM25 风格)
            # 统计文档频率
            doc_freq: dict[str, int] = defaultdict(int)
            for cs in summaries:
                cs_terms = self._extract_summary_terms(cs)
                for term in set(cs_terms):
                    doc_freq[term] += 1
            n_docs = len(summaries)

            scored: list[tuple[int, float]] = []
            for cs in summaries:
                cs_terms = self._extract_summary_terms(cs)
                if not cs_terms:
                    scored.append((cs.community_id, 0.0))
                    continue
                term_freq: dict[str, int] = defaultdict(int)
                for t in cs_terms:
                    term_freq[t] += 1
                cs_len = len(cs_terms)
                avg_len = (
                    sum(
                        len(self._extract_summary_terms(c))
                        for c in summaries
                    )
                    / n_docs
                )
                score = 0.0
                for qt in query_term_set:
                    if qt not in term_freq:
                        continue
                    tf = term_freq[qt]
                    df = doc_freq.get(qt, 0)
                    if df == 0:
                        continue
                    # BM25
                    k1 = 1.5
                    b = 0.75
                    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                    tf_norm = (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * cs_len / max(avg_len, 1))
                    )
                    score += idf * tf_norm
                scored.append((cs.community_id, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            relevant = scored[:top_k]

            relevant_ids = {cid for cid, _ in relevant}
            relevant_summaries = [
                cs for cs in summaries if cs.community_id in relevant_ids
            ]

            # 4. 跨社区连接 (桥接实体)
            cross_connections = self._find_cross_community_connections(
                communities, relevant_ids, store
            )

            return GlobalSearchResult(
                relevant_communities=relevant,
                community_summaries=relevant_summaries,
                cross_community_connections=cross_connections,
            )

    # --------------------------------------------------------
    # 自适应搜索
    # --------------------------------------------------------

    def adaptive_search(
        self,
        query: str,
        store: KnowledgeStore | None = None,
        *,
        entity_ids: list[str] | None = None,
        communities: list[Community] | None = None,
        max_hops: int = DEFAULT_MAX_HOPS,
        top_k: int = 10,
        **kwargs: Any,
    ) -> GraphRAGResult:
        """自适应搜索: 分析查询特征选择最优策略.

        决策逻辑 (借鉴 OMD-GraphRAG 自适应路由):
        - 若查询能匹配到具体实体 (entity_ids 非空或名称匹配命中)
          -> LOCAL_ONLY (局部优先)
        - 否则若存在社区
          -> GLOBAL_ONLY (全局优先)
        - 若两者皆可用
          -> ENSEMBLE (双通道融合)

        Args:
            query: 查询文本
            store: 知识存储
            entity_ids: 预设查询实体 ID (None 时自动解析)
            communities: 社区列表
            max_hops: 局部搜索跳数
            top_k: 返回结果数上限
            **kwargs: 额外参数

        Returns:
            GraphRAGResult 检索结果
        """
        return self.search(
            query,
            store,
            strategy=FusionStrategy.ADAPTIVE,
            entity_ids=entity_ids,
            communities=communities,
            max_hops=max_hops,
            top_k=top_k,
            **kwargs,
        )

    # ============================================================
    # 内部辅助方法
    # ============================================================

    def _select_adaptive_strategy(
        self,
        query: str,
        entity_ids: list[str] | None,
        store: KnowledgeStore,
    ) -> str:
        """自适应策略选择 (内部).

        分析查询特征决定使用 LOCAL_ONLY / GLOBAL_ONLY / ENSEMBLE。
        """
        # 解析查询实体
        resolved = entity_ids
        if not resolved:
            resolved = self._resolve_query_entities(query, store)

        has_local = bool(resolved)
        has_global = bool(self._communities)

        if has_local and has_global:
            # 两者皆可用: 若实体匹配数 >= 阈值，局部优先；否则融合
            if len(resolved) >= self.ADAPTIVE_ENTITY_MATCH_THRESHOLD:
                return FusionStrategy.LOCAL_ONLY
            return FusionStrategy.ENSEMBLE
        if has_local:
            return FusionStrategy.LOCAL_ONLY
        if has_global:
            return FusionStrategy.GLOBAL_ONLY
        # 都不可用: 默认全局 (若有社区) 否则局部
        return FusionStrategy.GLOBAL_ONLY

    def _resolve_query_entities(
        self,
        query: str,
        store: KnowledgeStore,
    ) -> list[str]:
        """从查询文本解析实体 ID (内部).

        采用名称/别名匹配 (借鉴 LlamaIndex entity resolution)。
        简化实现: 遍历实体，若名称或别名出现在查询中则命中。
        """
        if not query:
            return []
        query_lower = query.lower()
        matched_ids: list[str] = []
        # 遍历所有实体 (大规模场景应使用倒排索引)
        entities = store.entity_store.list_entities(limit=100000)
        for ent in entities:
            # 名称精确包含
            if ent.name and ent.name.lower() in query_lower:
                matched_ids.append(ent.entity_id)
                continue
            # 别名匹配
            hit = False
            for alias in ent.aliases:
                if alias and alias.lower() in query_lower:
                    hit = True
                    break
            if hit:
                matched_ids.append(ent.entity_id)
        return matched_ids

    def _build_adjacency_from_subgraph(
        self,
        subgraph: ExtractedSubgraph,
    ) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
        """从子图构建邻接表与边权重 (内部).

        返回:
            (adjacency, weights)
            - adjacency: {entity_id: [neighbor_id, ...]}
            - weights: {u: {v: weight}} (基于三元组置信度)
        """
        adjacency: dict[str, list[str]] = {
            eid: [] for eid in subgraph.entities
        }
        weights: dict[str, dict[str, float]] = defaultdict(dict)

        entity_id_set = set(subgraph.entities.keys())
        for triple in subgraph.triples:
            if triple.object_is_literal:
                continue
            if not triple.object_id:
                continue
            u = triple.subject_id
            v = triple.object_id
            if u not in entity_id_set or v not in entity_id_set:
                continue
            # 无向图: 双向加边
            if v not in adjacency[u]:
                adjacency[u].append(v)
            if u not in adjacency[v]:
                adjacency[v].append(u)
            w = triple.confidence
            # 取最大置信度 (多条同向边)
            if v not in weights[u] or w > weights[u][v]:
                weights[u][v] = w
            if u not in weights[v] or w > weights[v][u]:
                weights[v][u] = w

        return adjacency, dict(weights)

    def _compute_traversal_paths(
        self,
        root_entities: list[str],
        adjacency: dict[str, list[str]],
        ranked: list[tuple[str, float]],
    ) -> list[dict[str, Any]]:
        """计算从根实体到各排序实体的 BFS 最短路径 (内部)."""
        if not root_entities:
            return []

        # 多源 BFS
        dist: dict[str, int] = {}
        parent: dict[str, str | None] = {}
        queue: deque[str] = deque()
        for r in root_entities:
            dist[r] = 0
            parent[r] = None
            queue.append(r)

        while queue:
            u = queue.popleft()
            for v in adjacency.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)

        paths: list[dict[str, Any]] = []
        ranked_ids = {eid for eid, _ in ranked}
        for eid in ranked_ids:
            if eid not in dist:
                paths.append(
                    {"entity_id": eid, "hops": -1, "path": []}
                )
                continue
            # 回溯路径
            path: list[str] = []
            cur: str | None = eid
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur)
            path.reverse()
            paths.append(
                {
                    "entity_id": eid,
                    "hops": dist[eid],
                    "path": path,
                }
            )
        return paths

    def _get_or_create_summary(
        self,
        community: Community,
        store: KnowledgeStore,
    ) -> CommunitySummary:
        """获取或生成社区摘要 (带缓存，内部)."""
        # 缓存键包含 level 与 community_id
        cache_key = community.community_id
        if cache_key in self._summary_cache:
            cached = self._summary_cache[cache_key]
            # 若社区摘要已存在文本且层级匹配，直接返回
            if cached.level == community.level:
                return cached
        cs = self._summarizer.summarize_community(
            community, store, strategy="rule_based"
        )
        self._summary_cache[cache_key] = cs
        return cs

    def _extract_query_terms(self, query: str) -> list[str]:
        """提取查询关键词 (内部).

        简化实现: 中文按字符 bigram + 英文按空格分词。
        """
        if not query:
            return []
        terms: list[str] = []
        # 英文/数字 token
        en_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", query)
        terms.extend(t.lower() for t in en_tokens)
        # 中文 bigram (适用于无分词场景)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", query)
        for i in range(len(chinese_chars) - 1):
            terms.append(chinese_chars[i] + chinese_chars[i + 1])
        # 单字也加入 (短查询兜底)
        if len(chinese_chars) <= 4:
            terms.extend(chinese_chars)
        return terms

    def _extract_summary_terms(self, cs: CommunitySummary) -> list[str]:
        """从社区摘要提取索引词 (内部)."""
        terms: list[str] = []
        # 关键实体名
        for name in cs.key_entities:
            if name:
                terms.append(name.lower())
                # 英文部分
                en_tokens = re.findall(
                    r"[A-Za-z][A-Za-z0-9_-]{1,}", name
                )
                terms.extend(t.lower() for t in en_tokens)
                # 中文 bigram
                chinese_chars = re.findall(r"[\u4e00-\u9fff]", name)
                for i in range(len(chinese_chars) - 1):
                    terms.append(chinese_chars[i] + chinese_chars[i + 1])
        # 主题标签
        for tag in cs.topic_tags:
            terms.append(tag.lower())
        # 谓词 (从 core_relations 提取 [predicate])
        for rel in cs.core_relations:
            m = re.search(r"\[([^\]]+)\]", rel)
            if m:
                terms.append(m.group(1).lower())
        # 摘要文本 token
        if cs.summary_text:
            en_tokens = re.findall(
                r"[A-Za-z][A-Za-z0-9_-]{1,}", cs.summary_text
            )
            terms.extend(t.lower() for t in en_tokens)
            chinese_chars = re.findall(r"[\u4e00-\u9fff]", cs.summary_text)
            for i in range(len(chinese_chars) - 1):
                terms.append(chinese_chars[i] + chinese_chars[i + 1])
        return terms

    def _find_cross_community_connections(
        self,
        communities: list[Community],
        relevant_ids: set[int],
        store: KnowledgeStore,
    ) -> list[dict[str, Any]]:
        """识别跨社区连接 (桥接实体，内部).

        借鉴 GraphRAG 跨社区推理: 寻找连接不同社区的实体 (桥接节点)。
        """
        if not relevant_ids or len(communities) < 2:
            return []

        # 构建实体 -> 社区集合 映射
        entity_to_communities: dict[str, set[int]] = defaultdict(set)
        comm_by_id: dict[int, Community] = {}
        for comm in communities:
            comm_by_id[comm.community_id] = comm
            for eid in comm.entity_ids:
                entity_to_communities[eid].add(comm.community_id)

        connections: list[dict[str, Any]] = []
        seen_bridges: set[tuple[int, int, str]] = set()
        for eid, comm_set in entity_to_communities.items():
            # 只关注相关社区之间的桥接
            relevant_comm_set = comm_set & relevant_ids
            if len(relevant_comm_set) < 2:
                continue
            ent = store.get_entity(eid)
            ent_name = ent.name if ent else eid
            comm_list = sorted(relevant_comm_set)
            for i in range(len(comm_list)):
                for j in range(i + 1, len(comm_list)):
                    key = (comm_list[i], comm_list[j], eid)
                    if key in seen_bridges:
                        continue
                    seen_bridges.add(key)
                    connections.append(
                        {
                            "bridge_entity_id": eid,
                            "bridge_entity_name": ent_name,
                            "community_a": comm_list[i],
                            "community_b": comm_list[j],
                        }
                    )
        return connections

    def _rrf_fuse(
        self,
        local_result: LocalSearchResult | None,
        global_result: GlobalSearchResult | None,
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """RRF (Reciprocal Rank Fusion) 双通道结果融合 (内部).

        借鉴 RRF (Cormack et al., 2009):
            RRF_score(d) = Σ 1 / (k + rank_i(d))
        其中 k 为平滑常数 (默认 60)，rank_i 为文档在第 i 路结果中的排名。

        本实现将局部实体与全局社区统一为 "结果项" 并分别 RRF，
        输出融合后的统一列表 (source 标识来源通道)。
        """
        k = self.RRF_K
        scores: dict[str, float] = defaultdict(float)
        meta: dict[str, dict[str, Any]] = {}

        if local_result:
            for rank, (eid, sc) in enumerate(local_result.ranked_entities):
                key = f"local:{eid}"
                scores[key] += 1.0 / (k + rank + 1)
                meta[key] = {
                    "source": "local",
                    "entity_id": eid,
                    "original_score": sc,
                    "rank": rank + 1,
                }

        if global_result:
            for rank, (cid, sc) in enumerate(
                global_result.relevant_communities
            ):
                key = f"global:{cid}"
                scores[key] += 1.0 / (k + rank + 1)
                meta[key] = {
                    "source": "global",
                    "community_id": cid,
                    "original_score": sc,
                    "rank": rank + 1,
                }

        fused = []
        for key, rrf_score in scores.items():
            item = dict(meta[key])
            item["fused_score"] = round(rrf_score, 6)
            fused.append(item)
        fused.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused[:top_k]

    def _build_reasoning_context(
        self,
        query: str,
        strategy: str,
        local_result: LocalSearchResult | None,
        global_result: GlobalSearchResult | None,
        fused: list[dict[str, Any]],
    ) -> str:
        """为 LLM 生成推理上下文文本 (内部).

        整合局部实体详情、全局社区摘要与融合结果，
        生成结构化的推理上下文，供下游 LLM 推理使用。
        """
        lines: list[str] = []
        lines.append("# GraphRAG 推理上下文")
        lines.append(f"查询: {query}")
        lines.append(f"策略: {strategy}")
        lines.append("")

        if local_result and local_result.entity_details:
            lines.append("## 局部搜索结果 (实体)")
            for d in local_result.entity_details[:8]:
                root_mark = " [根实体]" if d.get("is_root") else ""
                lines.append(
                    f"- {d['name']} ({d['entity_type']})"
                    f"{root_mark} | 相关度: {d['relevance_score']}"
                )
                if d.get("description"):
                    lines.append(f"  描述: {d['description']}")
            if local_result.traversal_paths:
                lines.append("")
                lines.append("## 遍历路径")
                for tp in local_result.traversal_paths[:5]:
                    path_str = " -> ".join(tp.get("path", []))
                    lines.append(
                        f"- {tp['entity_id']} (跳数 {tp['hops']}): {path_str}"
                    )
            lines.append("")

        if global_result and global_result.community_summaries:
            lines.append("## 全局搜索结果 (社区)")
            for cs in global_result.community_summaries[:5]:
                lines.append(
                    f"- 社区 #{cs.community_id} (L{cs.level}): "
                    f"{cs.entity_count} 实体, {cs.triple_count} 三元组"
                )
                if cs.key_entities:
                    lines.append(
                        "  关键实体: " + ", ".join(cs.key_entities[:5])
                    )
                if cs.topic_tags:
                    lines.append(
                        "  主题: " + ", ".join(cs.topic_tags[:5])
                    )
            if global_result.cross_community_connections:
                lines.append("")
                lines.append("## 跨社区连接")
                for cc in global_result.cross_community_connections[:5]:
                    lines.append(
                        f"- {cc['bridge_entity_name']} "
                        f"连接社区 #{cc['community_a']} <-> #{cc['community_b']}"
                    )
            lines.append("")

        if fused:
            lines.append("## 融合结果")
            for item in fused[:8]:
                src = item.get("source", "?")
                score = item.get("fused_score", 0.0)
                if src == "local":
                    lines.append(
                        f"- [局部] {item.get('entity_id')} "
                        f"(RRF={score})"
                    )
                elif src == "global":
                    lines.append(
                        f"- [全局] 社区 #{item.get('community_id')} "
                        f"(RRF={score})"
                    )
            lines.append("")

        lines.append("## 推理建议")
        if strategy == FusionStrategy.LOCAL_ONLY:
            lines.append(
                "基于局部子图推理: 重点关注根实体周边关系与中心实体。"
            )
        elif strategy == FusionStrategy.GLOBAL_ONLY:
            lines.append(
                "基于全局社区推理: 重点关注社区主题与跨社区连接。"
            )
        elif strategy == FusionStrategy.ENSEMBLE:
            lines.append(
                "基于双通道融合推理: 结合局部实体细节与全局社区主题。"
            )
        else:
            lines.append("根据上述证据综合推理。")

        return "\n".join(lines)


# ============================================================
# 公开 API
# ============================================================


__all__ = [
    # 枚举/常量
    "SubgraphStrategy",
    "FusionStrategy",
    # 数据类
    "ExtractedSubgraph",
    "CommunitySummary",
    "LocalSearchResult",
    "GlobalSearchResult",
    "GraphRAGResult",
    # 核心类
    "SubgraphExtractor",
    "CommunitySummarizer",
    "GraphRAGRetriever",
    # 工具函数
    "_simplified_pagerank",
]
