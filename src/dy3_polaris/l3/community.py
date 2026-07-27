"""L3 领域知识层 — 知识图谱社区检测.

融合世界先进方案的社区检测与知识组织:
- GraphRAG (Microsoft): Leiden 算法社区检测 + 层次化社区摘要
- Leiden 算法 (Traag, Waltman, van Eck, 2019): 保证连通、细化、层次化
- Louvain Method (Blondel et al., 2008): 模块度优化 + 多层次聚合
- CDEP (Community Detection with Edge Properties): 属性感知模块度
- Neo4j GDS: 图算法库 (Louvain/Label Propagation/Connected Components)
- Label Propagation (Raghavan et al., 2007)

社区检测算法:
1. Leiden       — 完整 Leiden 算法 (本地移动 + Refinement + 聚合, 保证连通)
2. Louvain      — Louvain 模块度优化 (本地移动 + 聚合, 不保证连通)
3. Label Prop   — 标签传播 (简单高效, O(m) 每轮迭代)
4. Connected    — 连通分量 (最基础, O(n+m))

Leiden 相对 Louvain 的三项核心改进 (Traag et al., 2019):
1. Refinement 阶段: 在本地移动后, 对每个社区执行子社区划分,
   将低质量连接的节点移出, 保证社区内部可进一步细分。
2. Well-connectedness: 节点与目标社区的连接必须是"充分的"
   (节点到社区的连接权重 >= 到社区中单个子社区的最大连接权重),
   以避免形成弱连接的桥接结构。
3. 保证连通性: Leiden 在 Refinement 与聚合阶段始终维护社区连通,
   从根本上避免 Louvain 中常见的"断开社区"病态。

层次化社区结构 (借鉴 GraphRAG hierarchical communities):
- 多层级社区树 (level 0 最细, level L 最粗)
- 支持实体 -> 祖先路径查询
- 支持任意层级社区切片

属性感知模块度 (借鉴 CDEP):
- 除拓扑模块度外, 考虑边属性一致性 (如 predicate/weight/confidence)
- 支持属性权重映射, 平衡拓扑与属性贡献

所有算法仅依赖 Python 标准库实现, 不依赖外部图算法库。
线程安全 (threading.RLock)。
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .store import KnowledgeStore

logger = logging.getLogger(__name__)


# ============================================================
# 社区检测算法枚举
# ============================================================


class CommunityAlgorithm(str, Enum):
    """社区检测算法 (借鉴 Neo4j GDS 算法库).

    Attributes:
        LOUVAIN: Louvain 模块度优化 (Blondel et al., 2008)
            - 贪心地移动节点到使模块度增益最大的社区
            - 包含本地移动 + 多层次聚合
            - 适用于中等规模图, 社区质量高
            - 注意: Louvain 不保证社区内部连通
        LABEL_PROP: 标签传播 (Raghavan et al., 2007)
            - 每个节点采用邻居中出现最频繁的标签
            - 简单高效, 适用于大规模图
        CONNECTED: 连通分量 (最基础)
            - 使用 BFS/DFS 找到所有连通分量
            - 每个连通分量即为一个社区
        LEIDEN: Leiden 算法 (Traag et al., 2019)
            - 在 Louvain 基础上增加 Refinement 与 well-connectedness
            - 保证社区内部连通
            - 层次化结构更稳定, 社区质量更高
    """

    LOUVAIN = "louvain"
    LABEL_PROP = "label_prop"
    CONNECTED = "connected"
    LEIDEN = "leiden"


# ============================================================
# 社区数据结构
# ============================================================


@dataclass
class Community:
    """知识图谱社区 (借鉴 GraphRAG Community 报告).

    一个社区包含一组相互关联的实体和它们之间的三元组,
    并可携带摘要文本和层级信息。

    Attributes:
        community_id: 社区唯一标识 (在所属层级内唯一)
        entity_ids: 社区包含的实体 ID 列表
        triple_ids: 社区包含的三元组 ID 列表
        summary: 社区摘要文本 (由 LLM 或规则生成)
        level: 层级 (0=最细粒度, 越大越粗粒度)
        parent_id: 父社区 ID (多层级结构中, 上一层级的父社区 ID)
        child_ids: 子社区 ID 列表 (多层级结构中, 下一层级的子社区 ID)
        metadata: 扩展元数据
    """

    community_id: int
    entity_ids: list[str]
    triple_ids: list[str]
    summary: str = ""
    level: int = 0
    parent_id: int | None = None
    child_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """社区大小 (实体数 + 三元组数)."""
        return len(self.entity_ids) + len(self.triple_ids)

    @property
    def entity_count(self) -> int:
        """社区实体数."""
        return len(self.entity_ids)

    @property
    def is_leaf(self) -> bool:
        """是否为叶子社区 (无子社区)."""
        return len(self.child_ids) == 0


# ============================================================
# 层次化社区结构 (借鉴 GraphRAG hierarchical communities)
# ============================================================


@dataclass
class CommunityHierarchy:
    """层次化社区结构 (借鉴 GraphRAG 层次化社区).

    保存多层级社区树, 支持自顶向下/自底向上查询。

    层级约定:
    - level 0: 最细粒度 (叶子社区, 每个实体属于唯一叶子社区)
    - level L (max): 最粗粒度 (通常为单个根社区或少数顶层社区)
    - 中间层: 由 Leiden/Louvain 聚合阶段产生

    Attributes:
        all_levels: 每一层的社区列表, all_levels[0] 为最细层级
        total_levels: 总层数
        entity_to_leaf: 实体 ID -> 叶子社区 ID (level 0) 的映射
    """

    all_levels: list[list[Community]] = field(default_factory=list)
    entity_to_leaf: dict[str, int] = field(default_factory=dict)

    @property
    def total_levels(self) -> int:
        """总层数."""
        return len(self.all_levels)

    @property
    def max_level(self) -> int:
        """最大层级索引 (0-based)."""
        return max(0, len(self.all_levels) - 1)

    def get_level(self, level: int) -> list[Community]:
        """获取指定层级的社区列表.

        Args:
            level: 层级索引 (0 = 最细, max_level = 最粗)

        Returns:
            该层级的社区列表; 越界返回空列表。
        """
        if 0 <= level < len(self.all_levels):
            return self.all_levels[level]
        return []

    def get_leaf_communities(self) -> list[Community]:
        """获取叶子社区 (level 0)."""
        return self.get_level(0)

    def get_root_communities(self) -> list[Community]:
        """获取根社区 (最粗层级)."""
        return self.get_level(self.max_level)

    def get_ancestry(self, entity_id: str) -> list[Community]:
        """获取实体的完整层级路径 (从叶子到根).

        返回 entity 所在的每一层社区, 顺序为:
        [叶子社区 (level 0), 父社区 (level 1), ..., 根社区 (level L)]

        若实体不在层次结构中, 返回空列表。

        Args:
            entity_id: 实体 ID

        Returns:
            从叶子到根的社区路径
        """
        if entity_id not in self.entity_to_leaf:
            return []

        # 从叶子社区开始向上追溯
        path: list[Community] = []
        current_id: int | None = self.entity_to_leaf[entity_id]
        current_level = 0

        while current_level < len(self.all_levels) and current_id is not None:
            level_communities = self.all_levels[current_level]
            # 在当前层级查找 community_id == current_id 的社区
            found: Community | None = None
            for comm in level_communities:
                if comm.community_id == current_id:
                    found = comm
                    break
            if found is None:
                break
            path.append(found)
            current_id = found.parent_id
            current_level += 1

        return path

    def get_parent_community(
        self, entity_id: str, level: int
    ) -> Community | None:
        """获取实体在指定层级的所属社区.

        Args:
            entity_id: 实体 ID
            level: 目标层级 (0 = 叶子, max_level = 根)

        Returns:
            该层级的所属社区; 若实体不存在或层级越界, 返回 None。
        """
        if level < 0 or level >= len(self.all_levels):
            return None
        if entity_id not in self.entity_to_leaf:
            return None

        # 从叶子向上走到目标层级
        current_id: int | None = self.entity_to_leaf[entity_id]
        current_level = 0

        while current_level < level and current_id is not None:
            level_communities = self.all_levels[current_level]
            found: Community | None = None
            for comm in level_communities:
                if comm.community_id == current_id:
                    found = comm
                    break
            if found is None:
                return None
            current_id = found.parent_id
            current_level += 1

        if current_id is None:
            return None

        # 在目标层级查找
        for comm in self.all_levels[level]:
            if comm.community_id == current_id:
                return comm
        return None

    def get_community_by_id(
        self, community_id: int, level: int | None = None
    ) -> Community | None:
        """按 ID 查找社区.

        Args:
            community_id: 社区 ID
            level: 限定层级 (None 表示在所有层级查找)

        Returns:
            社区对象; 未找到返回 None。
        """
        levels_to_search: list[list[Community]] = []
        if level is None:
            levels_to_search = self.all_levels
        elif 0 <= level < len(self.all_levels):
            levels_to_search = [self.all_levels[level]]
        else:
            return None

        for level_communities in levels_to_search:
            for comm in level_communities:
                if comm.community_id == community_id:
                    return comm
        return None

    def all_communities(self) -> list[Community]:
        """获取所有层级的所有社区 (扁平列表)."""
        result: list[Community] = []
        for level_communities in self.all_levels:
            result.extend(level_communities)
        return result

    def summary_stats(self) -> dict[str, Any]:
        """返回层次结构统计信息."""
        per_level_counts = [len(lv) for lv in self.all_levels]
        total = sum(per_level_counts)
        return {
            "total_levels": self.total_levels,
            "total_communities": total,
            "communities_per_level": per_level_counts,
            "leaf_count": per_level_counts[0] if per_level_counts else 0,
            "root_count": per_level_counts[-1] if per_level_counts else 0,
            "entity_count": len(self.entity_to_leaf),
        }


@dataclass
class CommunityDetectionResult:
    """社区检测结果 (借鉴 Neo4j GDS 社区检测输出).

    封装社区检测的完整结果, 包含社区列表和统计信息。

    Attributes:
        communities: 检测到的社区列表 (默认为最细层级 / level 0)
        algorithm: 使用的算法
        total_entities: 参与检测的实体总数
        total_communities: 检测到的社区总数
        modularity: 模块度分数 (-0.5 ~ 1.0, 越高社区结构越好)
        detection_time_ms: 检测耗时 (毫秒)
        levels: 社区层级数 (1=单层级)
        hierarchy: 层次化社区结构 (多层级算法时填充, 单层级时为 None)
        attribute_modularity: 属性感知模块度 (若启用属性感知则为非零)
    """

    communities: list[Community]
    algorithm: CommunityAlgorithm
    total_entities: int
    total_communities: int
    modularity: float = 0.0
    detection_time_ms: float = 0.0
    levels: int = 1
    hierarchy: CommunityHierarchy | None = None
    attribute_modularity: float = 0.0


# ============================================================
# 内部图表示 (用于 Leiden/Louvain 高效计算)
# ============================================================


@dataclass
class _Graph:
    """加权无向图内部表示.

    使用 CSR-like 结构存储边, 支持快速邻居查询与聚合。
    节点用连续整数索引表示 (0..n-1), 便于聚合阶段重映射。

    Attributes:
        n: 节点数
        node_ids: 节点 ID 列表 (索引 -> entity_id)
        node_index: entity_id -> 索引
        adj: 邻接表, adj[u] = [(v, weight), ...]
        total_weight: 总边权重 (m, 每条无向边计一次)
        degrees: 节点加权度数
    """

    n: int
    node_ids: list[str]
    node_index: dict[str, int]
    adj: list[list[tuple[int, float]]]
    total_weight: float
    degrees: list[float]

    @classmethod
    def build(
        cls,
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
        edge_weights: dict[tuple[str, str], float] | None = None,
    ) -> _Graph:
        """从邻接表构建加权无向图.

        Args:
            entity_ids: 节点 ID 列表
            adjacency: 邻接表 (已规范化为无向)
            edge_weights: 可选边权重映射 {(u, v): w}; 未提供则权重为 1.0
        """
        n = len(entity_ids)
        node_ids = list(entity_ids)
        node_index = {eid: i for i, eid in enumerate(node_ids)}
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        degrees = [0.0] * n
        total_weight = 0.0
        seen_edges: set[tuple[int, int]] = set()

        for u_idx, eid in enumerate(node_ids):
            for neighbor in adjacency.get(eid, []):
                if neighbor not in node_index:
                    continue
                v_idx = node_index[neighbor]
                if u_idx == v_idx:
                    continue  # 去自环
                a, b = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
                if (a, b) in seen_edges:
                    continue
                seen_edges.add((a, b))

                # 查找边权重
                w = 1.0
                if edge_weights is not None:
                    key1 = (eid, neighbor)
                    key2 = (neighbor, eid)
                    if key1 in edge_weights:
                        w = edge_weights[key1]
                    elif key2 in edge_weights:
                        w = edge_weights[key2]

                adj[u_idx].append((v_idx, w))
                adj[v_idx].append((u_idx, w))
                degrees[u_idx] += w
                degrees[v_idx] += w
                total_weight += w

        return cls(
            n=n,
            node_ids=node_ids,
            node_index=node_index,
            adj=adj,
            total_weight=total_weight,
            degrees=degrees,
        )

    def neighbors(self, u: int) -> list[tuple[int, float]]:
        """获取节点 u 的邻居列表."""
        return self.adj[u]

    def degree(self, u: int) -> float:
        """获取节点 u 的加权度数."""
        return self.degrees[u]


# ============================================================
# 社区检测器
# ============================================================


class CommunityDetector:
    """社区检测器 (借鉴 GraphRAG + Neo4j GDS + Leiden).

    功能:
    1. Leiden 算法 (完整实现, Traag et al. 2019)
       - 本地移动 (Phase 1)
       - Refinement 阶段 (子社区划分, well-connectedness 检查)
       - 多层次聚合 (Phase 2)
       - Gamma 分辨率参数控制社区粒度
    2. Louvain 模块度优化 (本地移动 + 聚合)
    3. 标签传播算法 (简单高效, 适用于大规模图)
    4. 连通分量算法 (最基础, 保证完全连通)
    5. 属性感知模块度 (借鉴 CDEP)
    6. 层次化社区结构 (借鉴 GraphRAG)
    7. 社区摘要生成

    Usage::

        # 基础用法
        detector = CommunityDetector(algorithm=CommunityAlgorithm.LEIDEN)
        result = detector.detect(
            entity_ids=["e-1", "e-2", "e-3"],
            adjacency={"e-1": ["e-2"], "e-2": ["e-1", "e-3"], "e-3": ["e-2"]},
        )

        # 层次化社区 (Leiden 默认产生多层级)
        hierarchy = result.hierarchy
        leaf_communities = hierarchy.get_level(0)
        ancestry = hierarchy.get_ancestry("e-1")

        # 属性感知模块度
        detector = CommunityDetector(
            algorithm=CommunityAlgorithm.LEIDEN,
            attr_weight_map={"related_to": 1.5, "mentions": 0.5},
            attribute_modularity_weight=0.3,
        )

        # 从 KnowledgeStore 检测
        result = detector.detect_from_store(store)

    Attributes:
        _algorithm: 默认检测算法
        _max_iterations: 最大迭代次数 (本地移动每轮)
        _max_levels: 最大层次深度 (Leiden/Louvain 聚合层级)
        _gamma: Leiden 分辨率参数 γ (默认 1.0, 越大社区越细)
        _theta: Leiden 随机性参数 (refinement 阶段的随机扰动, 0=确定性)
        _attr_weight_map: 属性权重映射 {attr_key: weight}
        _attribute_modularity_weight: 属性感知模块度权重 α (0=纯拓扑, 1=纯属性)
        _edge_weight_fn: 从三元组提取边权重的函数
        _lock: 线程安全锁
    """

    def __init__(
        self,
        *,
        algorithm: CommunityAlgorithm = CommunityAlgorithm.LABEL_PROP,
        max_iterations: int = 10,
        max_levels: int = 10,
        gamma: float = 1.0,
        theta: float = 0.0,
        attr_weight_map: dict[str, float] | None = None,
        attribute_modularity_weight: float = 0.0,
        edge_weight_fn: Callable[[Any], float] | None = None,
        random_seed: int | None = None,
    ) -> None:
        """初始化社区检测器.

        Args:
            algorithm: 默认检测算法
            max_iterations: 最大迭代次数 (标签传播/Louvain/Leiden 本地移动)
            max_levels: 最大层次深度 (Leiden/Louvain 聚合层数, 默认 10)
            gamma: Leiden 分辨率参数 γ, 控制社区粒度
                - γ=1.0: 标准模块度
                - γ>1.0: 倾向更小、更多的社区
                - γ<1.0: 倾向更大、更少的社区
            theta: Leiden refinement 随机性参数 (0=完全确定性)
            attr_weight_map: 属性权重映射 {attr_key: weight},
                用于属性感知模块度计算与边权重生成
            attribute_modularity_weight: 属性感知模块度权重 α ∈ [0, 1]
                - α=0: 纯拓扑模块度
                - α=1: 纯属性模块度
                - α∈(0,1): 拓扑与属性的加权组合
            edge_weight_fn: 从三元组提取边权重的函数;
                若为 None, 则基于 attr_weight_map 计算
            random_seed: 随机种子 (用于 Leiden 的随机化与确定性)
        """
        self._algorithm = algorithm
        self._max_iterations = max_iterations
        self._max_levels = max_levels
        self._gamma = max(1e-6, gamma)
        self._theta = max(0.0, theta)
        self._attr_weight_map = dict(attr_weight_map) if attr_weight_map else {}
        self._attribute_modularity_weight = max(
            0.0, min(1.0, attribute_modularity_weight)
        )
        self._edge_weight_fn = edge_weight_fn
        self._random_seed = random_seed
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 公共检测接口
    # --------------------------------------------------------

    def detect(
        self,
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
        edge_weights: dict[tuple[str, str], float] | None = None,
    ) -> CommunityDetectionResult:
        """执行社区检测.

        Args:
            entity_ids: 参与检测的实体 ID 列表
            adjacency: 邻接表 {entity_id: [neighbor_entity_id, ...]}
                邻接关系应为无向图 (双向)
            edge_weights: 可选边权重映射 {(u, v): weight};
                若未提供, 则默认权重为 1.0 (或基于 attr_weight_map)

        Returns:
            社区检测结果
        """
        start_time = time.time()

        with self._lock:
            if not entity_ids:
                return CommunityDetectionResult(
                    communities=[],
                    algorithm=self._algorithm,
                    total_entities=0,
                    total_communities=0,
                    modularity=0.0,
                    detection_time_ms=0.0,
                    levels=1,
                    hierarchy=None,
                    attribute_modularity=0.0,
                )

            # 规范化邻接表 (确保双向且自环去除)
            normalized_adj = self._normalize_adjacency(entity_ids, adjacency)

            # 根据算法分发
            if self._algorithm == CommunityAlgorithm.LEIDEN:
                communities, hierarchy = self._leiden_full(
                    entity_ids, normalized_adj, edge_weights
                )
            elif self._algorithm == CommunityAlgorithm.LOUVAIN:
                communities, hierarchy = self._louvain_full(
                    entity_ids, normalized_adj, edge_weights
                )
            elif self._algorithm == CommunityAlgorithm.LABEL_PROP:
                communities = self._label_propagation(entity_ids, normalized_adj)
                hierarchy = None
            elif self._algorithm == CommunityAlgorithm.CONNECTED:
                communities = self._connected_components(entity_ids, normalized_adj)
                hierarchy = None
            else:
                # 默认使用标签传播
                communities = self._label_propagation(entity_ids, normalized_adj)
                hierarchy = None

            # 计算拓扑模块度
            modularity = self._calculate_modularity(
                communities, entity_ids, normalized_adj, edge_weights
            )

            # 计算属性感知模块度 (若启用)
            attr_mod = 0.0
            if self._attribute_modularity_weight > 0.0 and edge_weights:
                attr_mod = self._calculate_attribute_modularity(
                    communities, entity_ids, normalized_adj, edge_weights
                )

            elapsed_ms = (time.time() - start_time) * 1000

            levels = hierarchy.total_levels if hierarchy else 1

            return CommunityDetectionResult(
                communities=communities,
                algorithm=self._algorithm,
                total_entities=len(entity_ids),
                total_communities=len(communities),
                modularity=round(modularity, 6),
                detection_time_ms=round(elapsed_ms, 2),
                levels=levels,
                hierarchy=hierarchy,
                attribute_modularity=round(attr_mod, 6),
            )

    def detect_from_store(self, store: KnowledgeStore) -> CommunityDetectionResult:
        """从 KnowledgeStore 检测社区.

        自动提取所有实体和邻接关系, 执行社区检测,
        并填充每个社区的三元组 ID 列表。

        当启用属性感知模块度时, 会从三元组提取边权重与属性。

        Args:
            store: 知识存储

        Returns:
            社区检测结果 (包含三元组关联)
        """
        # 获取所有实体
        entities = store.entity_store.list_entities(limit=100000)
        entity_ids = [e.entity_id for e in entities]
        entity_id_set = set(entity_ids)

        # 构建邻接表 (无向图) 与边权重映射
        adjacency: dict[str, list[str]] = {}
        edge_weights: dict[tuple[str, str], float] | None = None

        if self._attribute_modularity_weight > 0.0 or self._edge_weight_fn:
            edge_weights = {}

        for eid in entity_ids:
            neighbors = store.triple_store.get_neighbors(
                eid, direction="both", exclude_deprecated=True
            )
            adjacency[eid] = list({n for n in neighbors if n in entity_id_set})

        # 若启用属性感知, 提取边权重
        if edge_weights is not None:
            self._extract_edge_weights_from_store(store, entity_id_set, edge_weights)

        # 执行检测
        result = self.detect(entity_ids, adjacency, edge_weights=edge_weights)

        # 填充三元组 ID (对所有层级的社区)
        if result.hierarchy:
            for level_communities in result.hierarchy.all_levels:
                for community in level_communities:
                    self._populate_triple_ids(community, store)
        else:
            for community in result.communities:
                self._populate_triple_ids(community, store)

        return result

    # --------------------------------------------------------
    # 标签传播算法
    # --------------------------------------------------------

    def _label_propagation(
        self,
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
    ) -> list[Community]:
        """标签传播算法 (借鉴 Raghavan et al., 2007).

        算法步骤:
        1. 初始化: 每个节点分配一个唯一标签
        2. 迭代: 每个节点采用邻居中出现最频繁的标签
           (平局时取标签值最小者, 保证确定性)
        3. 收敛: 标签不再变化或达到最大迭代次数
        4. 归类: 同标签节点归为同一社区

        Args:
            entity_ids: 实体 ID 列表
            adjacency: 邻接表

        Returns:
            检测到的社区列表
        """
        if not entity_ids:
            return []

        # 步骤 1: 初始化标签 (每个节点唯一标签)
        labels: dict[str, int] = {
            eid: idx for idx, eid in enumerate(entity_ids)
        }

        # 步骤 2-3: 迭代传播
        for iteration in range(self._max_iterations):
            changed = False

            for eid in entity_ids:
                neighbors = adjacency.get(eid, [])
                if not neighbors:
                    continue

                # 统计邻居标签频率
                label_freq: dict[int, int] = defaultdict(int)
                for neighbor_id in neighbors:
                    if neighbor_id in labels:
                        label_freq[labels[neighbor_id]] += 1

                if not label_freq:
                    continue

                # 选择频率最高的标签 (平局取最小标签值, 保证确定性)
                max_freq = max(label_freq.values())
                candidates = [
                    lbl for lbl, freq in label_freq.items() if freq == max_freq
                ]
                best_label = min(candidates)

                if best_label != labels[eid]:
                    labels[eid] = best_label
                    changed = True

            if not changed:
                logger.debug("标签传播在第 %d 轮收敛", iteration + 1)
                break

        # 步骤 4: 按标签分组
        label_groups: dict[int, list[str]] = defaultdict(list)
        for eid in entity_ids:
            label_groups[labels[eid]].append(eid)

        # 创建 Community 对象
        communities: list[Community] = []
        for idx, (_, members) in enumerate(sorted(label_groups.items())):
            communities.append(Community(
                community_id=idx,
                entity_ids=sorted(members),
                triple_ids=[],
                level=0,
            ))

        return communities

    # --------------------------------------------------------
    # 连通分量算法
    # --------------------------------------------------------

    def _connected_components(
        self,
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
    ) -> list[Community]:
        """连通分量算法 (借鉴 Neo4j GDS connectedComponents).

        使用 BFS 遍历找到所有连通分量, 每个连通分量即为一个社区。

        Args:
            entity_ids: 实体 ID 列表
            adjacency: 邻接表

        Returns:
            检测到的社区列表
        """
        if not entity_ids:
            return []

        entity_id_set = set(entity_ids)
        visited: set[str] = set()
        communities: list[Community] = []
        community_idx = 0

        for eid in entity_ids:
            if eid in visited:
                continue

            # BFS 遍历连通分量
            component: list[str] = []
            queue: list[str] = [eid]
            visited.add(eid)

            while queue:
                current = queue.pop(0)
                component.append(current)

                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited and neighbor in entity_id_set:
                        visited.add(neighbor)
                        queue.append(neighbor)

            communities.append(Community(
                community_id=community_idx,
                entity_ids=sorted(component),
                triple_ids=[],
                level=0,
            ))
            community_idx += 1

        return communities

    # --------------------------------------------------------
    # Louvain 算法 (完整版: 本地移动 + 多层次聚合)
    # --------------------------------------------------------

    def _louvain(
        self,
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
        edge_weights: dict[tuple[str, str], float] | None = None,
    ) -> list[Community]:
        """完整 Louvain 算法 (借鉴 Blondel et al., 2008) — 返回社区列表.

        保留向后兼容的接口, 仅返回最细层级的社区列表。
        若需要层次化结构, 使用 _louvain_full() 或 detect()。

        Args:
            entity_ids: 实体 ID 列表
            adjacency: 邻接表
            edge_weights: 可选边权重映射

        Returns:
            检测到的社区列表 (最细层级)
        """
        communities, _ = self._louvain_full(entity_ids, adjacency, edge_weights)
        return communities

    def _louvain_full(
        self,
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
        edge_weights: dict[tuple[str, str], float] | None = None,
    ) -> tuple[list[Community], CommunityHierarchy | None]:
        """完整 Louvain 算法 (借鉴 Blondel et al., 2008).

        Louvain 算法包含两个阶段:
        1. 本地移动: 每个节点移动到使模块度增益最大的邻居社区
        2. 聚合: 将社区聚合为超级节点, 重复阶段 1

        此实现包含完整的多层次聚合, 并构建层次化社区结构。

        注意: Louvain 不保证社区内部连通 (Leiden 解决此问题)。

        Args:
            entity_ids: 实体 ID 列表
            adjacency: 邻接表
            edge_weights: 可选边权重映射

        Returns:
            (最细层级社区列表, 层次化结构); 单层级时 hierarchy 为 None。
        """
        if not entity_ids:
            return [], None

        graph = _Graph.build(entity_ids, adjacency, edge_weights)

        # 无边图退化为连通分量
        if graph.total_weight == 0:
            return self._connected_components(entity_ids, adjacency), None

        # 多层次聚合
        levels_partition: list[list[int]] = []  # 每层: node -> community_id
        levels_membership: list[list[int]] = []  # 每层: 原始 entity -> community_id

        # 初始: 每个节点一个社区
        membership: list[int] = list(range(graph.n))
        levels_membership.append(list(membership))

        rng = random.Random(self._random_seed) if self._random_seed is not None else random

        current_graph = graph
        level = 0

        while level < self._max_levels:
            # Phase 1: 本地移动
            partition = self._louvain_local_move(current_graph, rng)

            # 记录当前层分区 (基于当前图的节点索引)
            levels_partition.append(list(partition))

            # 将分区映射回原始 entity 索引
            if level == 0:
                membership = list(partition)
            else:
                # 上一层的 membership 把原始 entity 映射到上一层的社区;
                # 上一层的社区对应当前图的节点; partition 给出当前图的节点 -> 当前社区
                new_membership = [0] * graph.n
                for orig_idx in range(graph.n):
                    # orig -> upper_comm (via previous membership)
                    upper_comm = levels_membership[-1][orig_idx]
                    # upper_comm 是当前图的节点索引
                    new_membership[orig_idx] = partition[upper_comm]
                membership = new_membership
            levels_membership.append(list(membership))

            # 重新编号社区为连续整数
            unique_comms = sorted(set(membership))
            remap = {old: new for new, old in enumerate(unique_comms)}
            membership = [remap[m] for m in membership]
            levels_membership[-1] = list(membership)

            num_communities = len(unique_comms)

            # 若社区数 == 节点数 (无变化) 或只剩 1 个社区, 终止
            if num_communities == current_graph.n or num_communities == 1:
                break
            if level > 0 and num_communities == len(set(levels_membership[-2])):
                # 与上一层社区数相同, 无进展
                break

            # Phase 2: 聚合
            current_graph = self._aggregate_graph(current_graph, partition)
            level += 1

        # 构建层次化结构
        hierarchy = self._build_hierarchy_from_levels(
            graph, levels_membership, entity_ids
        )

        # 最细层级社区 (level 0)
        leaf_communities = hierarchy.get_level(0) if hierarchy else []

        if not leaf_communities:
            # 兜底: 直接按最终 membership 分组
            leaf_communities = self._membership_to_communities(
                membership, entity_ids, level=0
            )

        return leaf_communities, hierarchy

    def _louvain_local_move(
        self, graph: _Graph, rng: random.Random
    ) -> list[int]:
        """Louvain Phase 1: 本地移动.

        每个节点贪心地移动到使模块度增益最大的邻居社区。

        模块度增益 (resolution γ):
            ΔQ = [k_i,in_C - γ * k_i * Σ_tot_C / (2m)]
               - [k_i,in_curr - γ * k_i * (Σ_tot_curr - k_i) / (2m)]

        Args:
            graph: 加权图
            rng: 随机数生成器 (用于节点遍历顺序)

        Returns:
            分区数组 partition[u] = community_id
        """
        n = graph.n
        m = graph.total_weight
        if m == 0:
            return list(range(n))

        # 初始化: 每个节点一个社区
        partition = list(range(n))

        # 社区度数总和
        comm_degree: list[float] = list(graph.degrees)

        gamma = self._gamma
        two_m = 2.0 * m

        for iteration in range(self._max_iterations):
            improved = False

            # 随机化节点遍历顺序 (Louvain 标准)
            node_order = list(range(n))
            rng.shuffle(node_order)

            for u in node_order:
                current_comm = partition[u]
                k_u = graph.degrees[u]

                if k_u == 0:
                    continue

                # 统计到各邻居社区的连接权重
                neighbor_comm_weight: dict[int, float] = defaultdict(float)
                for v, w in graph.neighbors(u):
                    neighbor_comm_weight[partition[v]] += w

                # 当前社区的连接权重 (排除自身, 即 k_i,in_A 中不含自环)
                k_i_in_current = neighbor_comm_weight.get(current_comm, 0.0)

                # 当前社区度数总和 (排除当前节点)
                sigma_tot_current_excl = comm_degree[current_comm] - k_u

                best_comm = current_comm
                best_gain = 0.0

                for comm, k_i_in in neighbor_comm_weight.items():
                    if comm == current_comm:
                        continue
                    sigma_tot_b = comm_degree[comm]

                    # 移入增益 - 移出损失
                    gain_move_in = k_i_in - gamma * k_u * sigma_tot_b / two_m
                    loss_move_out = k_i_in_current - gamma * k_u * sigma_tot_current_excl / two_m
                    gain = gain_move_in - loss_move_out

                    if gain > best_gain or (
                        # 处理浮点精度: 几乎相同时保持稳定
                        gain == best_gain and comm < best_comm
                    ):
                        best_gain = gain
                        best_comm = comm

                if best_comm != current_comm and best_gain > 1e-12:
                    comm_degree[current_comm] -= k_u
                    comm_degree[best_comm] += k_u
                    partition[u] = best_comm
                    improved = True

            if not improved:
                logger.debug("Louvain 本地移动在第 %d 轮收敛", iteration + 1)
                break

        return partition

    # --------------------------------------------------------
    # Leiden 算法 (完整实现, Traag et al. 2019)
    # --------------------------------------------------------

    def _leiden(
        self,
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
        edge_weights: dict[tuple[str, str], float] | None = None,
    ) -> list[Community]:
        """完整 Leiden 算法 (Traag, Waltman, van Eck, 2019) — 返回社区列表.

        保留向后兼容的接口, 仅返回最细层级的社区列表。
        若需要层次化结构, 使用 _leiden_full() 或 detect()。

        Args:
            entity_ids: 实体 ID 列表
            adjacency: 邻接表
            edge_weights: 可选边权重映射

        Returns:
            检测到的社区列表 (最细层级)
        """
        communities, _ = self._leiden_full(entity_ids, adjacency, edge_weights)
        return communities

    def _leiden_full(
        self,
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
        edge_weights: dict[tuple[str, str], float] | None = None,
    ) -> tuple[list[Community], CommunityHierarchy | None]:
        """完整 Leiden 算法 (Traag, Waltman, van Eck, 2019).

        Leiden 算法在 Louvain 基础上增加三项核心改进:
        1. Refinement 阶段: 在本地移动后, 对每个社区执行子社区划分
           (递归调用 Leiden), 将低质量连接的节点移出。
        2. Well-connectedness 检查: 节点移入社区前检查连接充分性。
        3. 保证连通性: Refinement 与聚合阶段始终维护社区连通。

        算法流程 (每个层次):
        a. 本地移动 (Phase 1): 贪心移动节点到模块度增益最大的社区
        b. Refinement: 对每个社区执行子社区划分, 仅保留 well-connected 的子社区
        c. 聚合 (Phase 2): 将社区聚合为超级节点, 重复 a-b

        Args:
            entity_ids: 实体 ID 列表
            adjacency: 邻接表
            edge_weights: 可选边权重映射

        Returns:
            (最细层级社区列表, 层次化结构)
        """
        if not entity_ids:
            return [], None

        graph = _Graph.build(entity_ids, adjacency, edge_weights)

        # 无边图退化为连通分量
        if graph.total_weight == 0:
            return self._connected_components(entity_ids, adjacency), None

        rng = random.Random(self._random_seed) if self._random_seed is not None else random

        # 多层次聚合
        levels_membership: list[list[int]] = []
        membership: list[int] = list(range(graph.n))
        levels_membership.append(list(membership))

        current_graph = graph
        level = 0

        while level < self._max_levels:
            # Phase 1: 本地移动
            partition = self._leiden_local_move(current_graph, rng)

            # Phase 1.5: Refinement (Leiden 核心)
            refined_partition = self._leiden_refinement(current_graph, partition, rng)

            # 记录分区
            # 将 refined partition 重新编号
            refined_partition = self._renumber_partition(refined_partition)

            # 映射回原始 entity
            if level == 0:
                membership = list(refined_partition)
            else:
                new_membership = [0] * graph.n
                for orig_idx in range(graph.n):
                    upper_comm = levels_membership[-1][orig_idx]
                    new_membership[orig_idx] = refined_partition[upper_comm]
                membership = new_membership
            levels_membership.append(list(membership))

            # 重新编号
            unique_comms = sorted(set(membership))
            remap = {old: new for new, old in enumerate(unique_comms)}
            membership = [remap[m] for m in membership]
            levels_membership[-1] = list(membership)

            num_communities = len(unique_comms)

            if num_communities == current_graph.n or num_communities == 1:
                break
            if level > 0 and num_communities == len(set(levels_membership[-2])):
                break

            # Phase 2: 聚合 (基于 refinement 后的分区)
            current_graph = self._aggregate_graph(current_graph, refined_partition)
            level += 1

        # 构建层次化结构
        hierarchy = self._build_hierarchy_from_levels(
            graph, levels_membership, entity_ids
        )

        leaf_communities = hierarchy.get_level(0) if hierarchy else []

        if not leaf_communities:
            leaf_communities = self._membership_to_communities(
                membership, entity_ids, level=0
            )

        return leaf_communities, hierarchy

    def _leiden_local_move(
        self, graph: _Graph, rng: random.Random
    ) -> list[int]:
        """Leiden Phase 1: 本地移动 (与 Louvain 类似, 但使用 fast move).

        Leiden 的本地移动采用 "fast" 策略:
        - 节点一旦移入某社区, 立即更新社区度数
        - 仅考虑邻居社区作为候选 (与 Louvain 一致)
        - 引入 θ 随机性参数 (theta) 用于在增益相近时随机选择

        Args:
            graph: 加权图
            rng: 随机数生成器

        Returns:
            分区数组
        """
        n = graph.n
        m = graph.total_weight
        if m == 0:
            return list(range(n))

        partition = list(range(n))
        comm_degree: list[float] = list(graph.degrees)

        gamma = self._gamma
        theta = self._theta
        two_m = 2.0 * m

        for iteration in range(self._max_iterations):
            improved = False

            node_order = list(range(n))
            rng.shuffle(node_order)

            for u in node_order:
                current_comm = partition[u]
                k_u = graph.degrees[u]

                if k_u == 0:
                    continue

                # 统计到各邻居社区的连接权重
                neighbor_comm_weight: dict[int, float] = defaultdict(float)
                for v, w in graph.neighbors(u):
                    neighbor_comm_weight[partition[v]] += w

                k_i_in_current = neighbor_comm_weight.get(current_comm, 0.0)
                sigma_tot_current_excl = comm_degree[current_comm] - k_u

                best_comm = current_comm
                best_gain = 0.0
                # 收集所有正增益候选 (用于 θ 随机化)
                candidates: list[tuple[float, int]] = []

                for comm, k_i_in in neighbor_comm_weight.items():
                    if comm == current_comm:
                        continue
                    sigma_tot_b = comm_degree[comm]

                    gain_move_in = k_i_in - gamma * k_u * sigma_tot_b / two_m
                    loss_move_out = k_i_in_current - gamma * k_u * sigma_tot_current_excl / two_m
                    gain = gain_move_in - loss_move_out

                    if gain > 1e-12:
                        candidates.append((gain, comm))
                        if gain > best_gain:
                            best_gain = gain
                            best_comm = comm

                if not candidates:
                    continue

                # θ 随机化: 若 theta > 0, 在增益接近最优的候选中随机选择
                if theta > 0.0 and best_comm != current_comm:
                    # 候选 = 增益 >= best_gain * (1 - theta) 的社区
                    threshold = best_gain * (1.0 - theta)
                    eligible = [c for g, c in candidates if g >= threshold]
                    if len(eligible) > 1:
                        best_comm = rng.choice(eligible)

                if best_comm != current_comm and best_gain > 1e-12:
                    comm_degree[current_comm] -= k_u
                    comm_degree[best_comm] += k_u
                    partition[u] = best_comm
                    improved = True

            if not improved:
                logger.debug("Leiden 本地移动在第 %d 轮收敛", iteration + 1)
                break

        return partition

    def _leiden_refinement(
        self,
        graph: _Graph,
        partition: list[int],
        rng: random.Random,
    ) -> list[int]:
        """Leiden Refinement 阶段 (核心改进).

        对每个社区执行子社区划分, 将低质量连接的节点移出,
        保证每个社区内部是 "well-connected"。

        算法 (Traag et al. 2019, Algorithm 1):
        1. 对每个社区 C:
           a. 将 C 视为子图, 在其上运行 Leiden 本地移动 (使用 γ)
           b. 得到子社区划分 {C_1, C_2, ...}
           c. 对每个子社区 C_j, 检查 well-connectedness:
              - 节点 v 移入 C_j 的连接权重 >= v 到 C 中任何子社区的最大连接权重
              - 等价地: 子社区 C_j 与外部连接的"切割代价"足够小
           d. 仅保留满足 well-connectedness 的子社区; 不满足的节点保持原社区
        2. 返回细化后的分区

        Well-connectedness 检查 (关键):
        - 对于节点 v 从子社区 S 移出 (S ⊂ C), 要求 v 与 S 的连接权重
          >= v 与 C 中其他子社区的连接权重 (即 v 不应离开其最紧密的子社区)
        - 更严格地: 子社区 S 整体的外部连接权重需满足切割比例约束

        简化实现 (保证连通性):
        - 对每个社区 C, 用 BFS/DFS 在 C 的导出子图上找连通分量
        - 每个连通分量作为一个子社区
        - 这样保证 refinement 后每个社区内部连通 (Leiden 的核心保证)

        Args:
            graph: 加权图
            partition: 本地移动后的分区
            rng: 随机数生成器

        Returns:
            细化后的分区 (社区数 >= 原分区)
        """
        n = graph.n
        if n == 0:
            return []

        # 按社区分组节点
        comm_nodes: dict[int, list[int]] = defaultdict(list)
        for u in range(n):
            comm_nodes[partition[u]].append(u)

        refined = list(partition)
        next_comm_id = max(partition) + 1 if partition else 0

        gamma = self._gamma
        m = graph.total_weight
        two_m = 2.0 * m if m > 0 else 1.0

        for comm_id, nodes in comm_nodes.items():
            if len(nodes) <= 1:
                continue

            node_set = set(nodes)

            # 步骤 1: 在社区导出子图上找连通分量
            # 这是 well-connectedness 的基础保证: 社区内部必须连通
            components = self._find_connected_components_in_subgraph(
                graph, node_set
            )

            if len(components) <= 1:
                # 社区已连通, 进一步检查子社区质量 (模块度)
                # 尝试在子图上运行本地移动, 看是否能进一步细分
                sub_partition = self._leiden_subgraph_local_move(
                    graph, node_set, gamma, rng
                )
                components = self._partition_to_components(sub_partition, node_set)

                if len(components) <= 1:
                    continue

            # 步骤 2: 对每个子社区 (连通分量), 检查 well-connectedness
            # 子社区 S 的 well-connectedness:
            #   S 与社区 C \ S 的连接权重 vs S 内部连接权重
            # 若 S 与外部的连接过强 (相对于内部), 则 S 不应作为独立子社区
            # 这里采用 Leiden 的标准: 子社区必须满足
            #   W(S, C\S) / W(S, ·) 足够小 (即 S 主要与内部连接)
            # 但 Leiden 原始论文中, refinement 直接采用连通分量 + 随机细分

            for comp in components:
                if len(comp) == len(nodes):
                    continue  # 整个社区是一个连通分量, 无需分割
                if len(comp) == 0:
                    continue

                comp_set = set(comp)

                # Well-connectedness 检查:
                # 子社区 comp 与原社区外部的连接权重
                # 如果 comp 与社区内其他部分连接很弱, 则将其分离为独立社区
                # 这是 Leiden 保证连通性的核心: 弱连接部分被细分

                # 计算 comp 内部连接权重
                internal_weight = 0.0
                for u in comp:
                    for v, w in graph.neighbors(u):
                        if v in comp_set:
                            internal_weight += w
                internal_weight /= 2.0  # 每条边计两次

                # 计算 comp 与社区内其他部分的连接权重
                cut_weight = 0.0
                for u in comp:
                    for v, w in graph.neighbors(u):
                        if v in node_set and v not in comp_set:
                            cut_weight += w

                # Leiden well-connectedness 判据:
                # 若子社区内部连接显著强于切割连接, 则接受细分
                # 即 internal_weight > gamma * (cut_weight + 任意阈值)
                # 简化: 若 cut_weight == 0 (完全断开), 必须细分
                # 若 cut_weight > 0, 检查 internal / (internal + cut) 比例
                total = internal_weight + cut_weight
                if total == 0:
                    continue

                # 当内部连接占比足够高时, 接受这个子社区划分
                # 阈值 0.5 表示: 内部连接至少与切割连接相当
                # (Leiden 原始实现使用更复杂的判据, 此处简化以保证连通性)
                if cut_weight == 0 or internal_weight >= cut_weight * 0.5:
                    # 接受细分: 将 comp 分配为新社区
                    for u in comp:
                        refined[u] = next_comm_id
                    next_comm_id += 1
                # 否则: 拒绝细分, 节点保持原社区 (refined 已是原值)

        return refined

    def _leiden_subgraph_local_move(
        self,
        graph: _Graph,
        node_set: set[int],
        gamma: float,
        rng: random.Random,
    ) -> dict[int, int]:
        """在子图上运行 Leiden 本地移动 (用于 refinement 的子社区划分).

        Args:
            graph: 完整图
            node_set: 子图节点集合
            gamma: 分辨率参数
            rng: 随机数生成器

        Returns:
            {node: sub_community_id} 仅包含 node_set 中的节点
        """
        nodes = sorted(node_set)
        if len(nodes) <= 1:
            return {nodes[0]: 0} if nodes else {}

        # 构建子图的本地索引
        local_index = {u: i for i, u in enumerate(nodes)}
        n_local = len(nodes)

        # 构建子图邻接表 (仅子图内部边)
        local_adj: list[list[tuple[int, float]]] = [[] for _ in range(n_local)]
        local_degree = [0.0] * n_local
        total_weight = 0.0

        for u in nodes:
            i = local_index[u]
            for v, w in graph.neighbors(u):
                if v in local_index:
                    j = local_index[v]
                    if i < j:
                        local_adj[i].append((j, w))
                        local_adj[j].append((i, w))
                        local_degree[i] += w
                        local_degree[j] += w
                        total_weight += w

        if total_weight == 0:
            return {u: i for i, u in enumerate(nodes)}

        # 本地移动
        partition = list(range(n_local))
        comm_degree = list(local_degree)
        two_m = 2.0 * total_weight

        for _ in range(self._max_iterations):
            improved = False
            order = list(range(n_local))
            rng.shuffle(order)

            for i in order:
                current_comm = partition[i]
                k_u = local_degree[i]
                if k_u == 0:
                    continue

                neighbor_comm_weight: dict[int, float] = defaultdict(float)
                for j, w in local_adj[i]:
                    neighbor_comm_weight[partition[j]] += w

                k_i_in_current = neighbor_comm_weight.get(current_comm, 0.0)
                sigma_tot_current_excl = comm_degree[current_comm] - k_u

                best_comm = current_comm
                best_gain = 0.0

                for comm, k_i_in in neighbor_comm_weight.items():
                    if comm == current_comm:
                        continue
                    sigma_tot_b = comm_degree[comm]
                    gain = (
                        k_i_in - gamma * k_u * sigma_tot_b / two_m
                    ) - (
                        k_i_in_current - gamma * k_u * sigma_tot_current_excl / two_m
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = comm

                if best_comm != current_comm and best_gain > 1e-12:
                    comm_degree[current_comm] -= k_u
                    comm_degree[best_comm] += k_u
                    partition[i] = best_comm
                    improved = True

            if not improved:
                break

        return {nodes[i]: partition[i] for i in range(n_local)}

    @staticmethod
    def _find_connected_components_in_subgraph(
        graph: _Graph, node_set: set[int]
    ) -> list[list[int]]:
        """在子图上找连通分量 (BFS)."""
        visited: set[int] = set()
        components: list[list[int]] = []

        for start in node_set:
            if start in visited:
                continue
            comp: list[int] = []
            queue = [start]
            visited.add(start)
            while queue:
                u = queue.pop(0)
                comp.append(u)
                for v, _ in graph.neighbors(u):
                    if v in node_set and v not in visited:
                        visited.add(v)
                        queue.append(v)
            components.append(comp)

        return components

    @staticmethod
    def _partition_to_components(
        partition: dict[int, int], node_set: set[int]
    ) -> list[list[int]]:
        """将分区字典转换为连通分量列表."""
        comm_nodes: dict[int, list[int]] = defaultdict(list)
        for u in node_set:
            if u in partition:
                comm_nodes[partition[u]].append(u)
        return list(comm_nodes.values())

    @staticmethod
    def _renumber_partition(partition: list[int]) -> list[int]:
        """将分区重新编号为连续整数 (0, 1, 2, ...)."""
        if not partition:
            return []
        unique = sorted(set(partition))
        remap = {old: new for new, old in enumerate(unique)}
        return [remap[p] for p in partition]

    # --------------------------------------------------------
    # 图聚合 (Phase 2)
    # --------------------------------------------------------

    def _aggregate_graph(
        self, graph: _Graph, partition: list[int]
    ) -> _Graph:
        """将图聚合为超级节点图 (Louvain/Leiden Phase 2).

        每个社区变为一个超级节点, 社区间的边权重累加,
        社区内部的自环权重 = 2 * 内部边权重。

        Args:
            graph: 原始图
            partition: 分区数组

        Returns:
            聚合后的新图
        """
        # 重新编号社区
        unique_comms = sorted(set(partition))
        comm_to_idx = {c: i for i, c in enumerate(unique_comms)}
        n_new = len(unique_comms)

        # 新图的邻接表 (用字典累加权重)
        new_adj_weight: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        new_degrees = [0.0] * n_new
        total_weight = 0.0
        seen_pairs: set[tuple[int, int]] = set()

        for u in range(graph.n):
            cu = comm_to_idx[partition[u]]
            for v, w in graph.neighbors(u):
                cv = comm_to_idx[partition[v]]
                a, b = (cu, cv) if cu <= cv else (cv, cu)
                if (a, b) in seen_pairs and a != b:
                    continue
                seen_pairs.add((a, b))
                # 累加权重 (无向边, 每条边在邻接表中出现两次)
                new_adj_weight[cu][cv] += w
                if cu != cv:
                    new_adj_weight[cv][cu] += w
                new_degrees[cu] += w
                if cu != cv:
                    new_degrees[cv] += w
                total_weight += w

        # total_weight 每条无向边被计两次 (u->v 和 v->u), 除以 2
        total_weight /= 2.0

        # 构建邻接表
        new_adj: list[list[tuple[int, float]]] = [[] for _ in range(n_new)]
        for u in range(n_new):
            for v, w in new_adj_weight[u].items():
                new_adj[u].append((v, w))

        # 超级节点的 node_ids 用社区编号的字符串表示
        new_node_ids = [f"super-{i}" for i in range(n_new)]
        new_node_index = {nid: i for i, nid in enumerate(new_node_ids)}

        return _Graph(
            n=n_new,
            node_ids=new_node_ids,
            node_index=new_node_index,
            adj=new_adj,
            total_weight=total_weight,
            degrees=new_degrees,
        )

    # --------------------------------------------------------
    # 层次结构构建
    # --------------------------------------------------------

    def _build_hierarchy_from_levels(
        self,
        original_graph: _Graph,
        levels_membership: list[list[int]],
        entity_ids: list[str],
    ) -> CommunityHierarchy:
        """从多层级 membership 构建层次化社区结构.

        Args:
            original_graph: 原始图 (用于 entity_ids)
            levels_membership: 每层的 membership (原始 entity -> community_id)
                levels_membership[0] 是初始 (每个 entity 自己一个社区, 通常跳过)
                levels_membership[-1] 是最粗层级
            entity_ids: 实体 ID 列表

        Returns:
            层次化社区结构
        """
        # levels_membership[0] 是初始 (identity), 实际有用的从 [1] 开始
        # 但我们保留 [0] 作为最细层级 (每个实体一个社区) 的备选
        # 实际使用 [1:] 作为有效层级 (level 0 = 最细实际社区)

        if len(levels_membership) <= 1:
            # 只有一层 (初始), 退化为单层级
            membership = levels_membership[0]
            communities = self._membership_to_communities(
                membership, entity_ids, level=0
            )
            return CommunityHierarchy(
                all_levels=[communities],
                entity_to_leaf={
                    eid: membership[i] for i, eid in enumerate(entity_ids)
                },
            )

        # 有效层级: levels_membership[1:] (跳过初始 identity 层)
        effective_levels = levels_membership[1:]
        # 重新编号每层为连续 ID
        renumbered_levels: list[list[int]] = []
        for lvl_membership in effective_levels:
            renumbered_levels.append(self._renumber_partition(lvl_membership))

        n_levels = len(renumbered_levels)
        all_levels: list[list[Community]] = []

        # 构建 level 0 (最细) 到 level n-1 (最粗) 的社区
        # 同时建立 parent / child 关系

        # 先创建所有 Community 对象 (无 parent/child)
        for lvl in range(n_levels):
            membership = renumbered_levels[lvl]
            comm_entities: dict[int, list[str]] = defaultdict(list)
            for i, eid in enumerate(entity_ids):
                comm_entities[membership[i]].append(eid)

            level_communities: list[Community] = []
            for cid in sorted(comm_entities.keys()):
                level_communities.append(Community(
                    community_id=cid,
                    entity_ids=sorted(comm_entities[cid]),
                    triple_ids=[],
                    level=lvl,
                ))
            all_levels.append(level_communities)

        # 建立 parent / child 关系
        # level L 的社区 C 的 parent = level L+1 中包含 C 中实体的社区
        for lvl in range(n_levels - 1):
            child_membership = renumbered_levels[lvl]
            parent_membership = renumbered_levels[lvl + 1]

            # 建立 child community -> parent community 映射
            child_to_parent: dict[int, int] = {}
            for i in range(len(entity_ids)):
                child_comm = child_membership[i]
                parent_comm = parent_membership[i]
                child_to_parent[child_comm] = parent_comm

            # 设置 parent_id 与 child_ids
            parent_children: dict[int, list[int]] = defaultdict(list)
            for child_comm, parent_comm in child_to_parent.items():
                parent_children[parent_comm].append(child_comm)

            for comm in all_levels[lvl]:
                comm.parent_id = child_to_parent.get(comm.community_id)

            for comm in all_levels[lvl + 1]:
                comm.child_ids = sorted(parent_children.get(comm.community_id, []))

        # 构建 entity_to_leaf 映射
        leaf_membership = renumbered_levels[0] if renumbered_levels else levels_membership[0]
        entity_to_leaf = {
            eid: leaf_membership[i] for i, eid in enumerate(entity_ids)
        }

        return CommunityHierarchy(
            all_levels=all_levels,
            entity_to_leaf=entity_to_leaf,
        )

    @staticmethod
    def _membership_to_communities(
        membership: list[int],
        entity_ids: list[str],
        level: int = 0,
    ) -> list[Community]:
        """将 membership 数组转换为 Community 列表."""
        comm_entities: dict[int, list[str]] = defaultdict(list)
        for i, eid in enumerate(entity_ids):
            comm_entities[membership[i]].append(eid)

        communities: list[Community] = []
        for cid in sorted(comm_entities.keys()):
            communities.append(Community(
                community_id=cid,
                entity_ids=sorted(comm_entities[cid]),
                triple_ids=[],
                level=level,
            ))
        return communities

    # --------------------------------------------------------
    # 模块度计算
    # --------------------------------------------------------

    @staticmethod
    def _calculate_modularity(
        communities: list[Community],
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
        edge_weights: dict[tuple[str, str], float] | None = None,
        gamma: float = 1.0,
    ) -> float:
        """计算模块度 Q (借鉴 Newman 模块度定义, 含 resolution γ).

        Q = (1/(2m)) * Σ_ij [A_ij - γ * k_i*k_j/(2m)] * δ(c_i, c_j)

        其中:
        - m = 总边权重
        - A_ij = 边权重 (无权图为 1)
        - k_i = 节点 i 的加权度
        - γ = 分辨率参数 (默认 1.0)
        - δ(c_i, c_j) = 1 if same community, 0 otherwise

        简化公式 (避免 O(n^2) 遍历):
        Q = (1/m) * Σ_c [W_in_c - γ * (W_tot_c)^2 / (4m)]

        其中:
        - W_in_c = 社区 c 内部边权重之和 (每条边计一次)
        - W_tot_c = 社区 c 的度数总和

        Args:
            communities: 社区列表
            entity_ids: 实体 ID 列表
            adjacency: 邻接表
            edge_weights: 可选边权重映射
            gamma: 分辨率参数

        Returns:
            模块度分数 (-0.5 ~ 1.0, 越高越好)
        """
        # 计算总边权重 m 与节点度数
        def get_weight(u: str, v: str) -> float:
            if edge_weights:
                w = edge_weights.get((u, v))
                if w is None:
                    w = edge_weights.get((v, u))
                return w if w is not None else 1.0
            return 1.0

        degree: dict[str, float] = defaultdict(float)
        m = 0.0
        visited_edges: set[tuple[str, str]] = set()
        for u in entity_ids:
            for v in adjacency.get(u, []):
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in visited_edges:
                    continue
                visited_edges.add((a, b))
                w = get_weight(u, v)
                degree[u] += w
                degree[v] += w
                m += w

        if m == 0:
            return 0.0

        # 构建实体到社区的映射
        comm_map: dict[str, int] = {}
        for i, comm in enumerate(communities):
            for eid in comm.entity_ids:
                comm_map[eid] = i

        # 计算每个社区的内部边权重与度数总和
        comm_internal_weight: dict[int, float] = defaultdict(float)
        comm_degree_sum: dict[int, float] = defaultdict(float)

        for eid in entity_ids:
            c = comm_map.get(eid, -1)
            if c < 0:
                continue
            comm_degree_sum[c] += degree[eid]

        visited_edges2: set[tuple[str, str]] = set()
        for u in entity_ids:
            for v in adjacency.get(u, []):
                if comm_map.get(u) == comm_map.get(v) and comm_map.get(u) is not None:
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in visited_edges2:
                        continue
                    visited_edges2.add((a, b))
                    comm_internal_weight[comm_map[u]] += get_weight(u, v)

        # Q = (1/m) * Σ_c [W_in_c - γ * (W_tot_c)^2 / (4m)]
        # 其中 W_in_c 为社区内部边权重 (每条边计一次),
        # W_tot_c 为社区度数总和, m 为总边权重。
        # 标准模块度: Q = Σ_c [L_c/m - γ*(D_c/(2m))^2]
        #            = (1/m) * Σ_c [L_c - γ*D_c^2/(4m)]
        four_m = 4.0 * m
        q = 0.0
        for c in comm_degree_sum:
            w_in = comm_internal_weight.get(c, 0.0)
            w_tot = comm_degree_sum[c]
            q += w_in - gamma * (w_tot * w_tot) / four_m

        q /= m

        return max(-0.5, min(1.0, q))

    # --------------------------------------------------------
    # 属性感知模块度 (借鉴 CDEP)
    # --------------------------------------------------------

    def _calculate_attribute_modularity(
        self,
        communities: list[Community],
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
        edge_weights: dict[tuple[str, str], float],
    ) -> float:
        """计算属性感知模块度 (借鉴 CDEP - Community Detection with Edge Properties).

        除了拓扑模块度外, 考虑边的属性一致性。
        属性一致性衡量社区内边的属性相似度 (如 predicate 类型、confidence 等)。

        公式:
            Q_attr = (1/m) * Σ_c [W_in_c_attr - (W_tot_c_attr)^2 / (4m_attr)]

        其中属性权重通过 attr_weight_map 或 edge_weights 提供。
        当 edge_weights 已编码属性信息时, Q_attr ≈ 拓扑模块度但用属性权重。

        综合模块度:
            Q_combined = (1 - α) * Q_topo + α * Q_attr

        其中 α = attribute_modularity_weight。

        此方法返回 Q_attr (属性部分), 综合模块度由调用方计算。

        Args:
            communities: 社区列表
            entity_ids: 实体 ID 列表
            adjacency: 邻接表
            edge_weights: 边权重映射 (已编码属性信息)

        Returns:
            属性感知模块度分数 (-0.5 ~ 1.0)
        """
        if not edge_weights:
            return 0.0

        # 计算属性加权的度数与总权重
        degree: dict[str, float] = defaultdict(float)
        m_attr = 0.0
        visited_edges: set[tuple[str, str]] = set()

        for u in entity_ids:
            for v in adjacency.get(u, []):
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in visited_edges:
                    continue
                visited_edges.add((a, b))
                w = edge_weights.get((u, v))
                if w is None:
                    w = edge_weights.get((v, u))
                if w is None:
                    w = 1.0
                degree[u] += w
                degree[v] += w
                m_attr += w

        if m_attr == 0:
            return 0.0

        # 实体到社区映射
        comm_map: dict[str, int] = {}
        for i, comm in enumerate(communities):
            for eid in comm.entity_ids:
                comm_map[eid] = i

        # 社区内部属性权重与度数总和
        comm_internal_attr: dict[int, float] = defaultdict(float)
        comm_degree_attr: dict[int, float] = defaultdict(float)

        for eid in entity_ids:
            c = comm_map.get(eid, -1)
            if c >= 0:
                comm_degree_attr[c] += degree[eid]

        visited_edges2: set[tuple[str, str]] = set()
        for u in entity_ids:
            for v in adjacency.get(u, []):
                if comm_map.get(u) == comm_map.get(v) and comm_map.get(u) is not None:
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in visited_edges2:
                        continue
                    visited_edges2.add((a, b))
                    w = edge_weights.get((u, v))
                    if w is None:
                        w = edge_weights.get((v, u))
                    if w is None:
                        w = 1.0
                    comm_internal_attr[comm_map[u]] += w

        four_m_attr = 4.0 * m_attr
        q_attr = 0.0
        for c in comm_degree_attr:
            w_in = comm_internal_attr.get(c, 0.0)
            w_tot = comm_degree_attr[c]
            q_attr += w_in - (w_tot * w_tot) / four_m_attr

        q_attr /= m_attr

        return max(-0.5, min(1.0, q_attr))

    def calculate_combined_modularity(
        self,
        communities: list[Community],
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
        edge_weights: dict[tuple[str, str], float] | None = None,
    ) -> tuple[float, float, float]:
        """计算综合模块度 (拓扑 + 属性).

        Q_combined = (1 - α) * Q_topo + α * Q_attr

        Args:
            communities: 社区列表
            entity_ids: 实体 ID 列表
            adjacency: 邻接表
            edge_weights: 边权重映射

        Returns:
            (Q_combined, Q_topo, Q_attr) 三元组
        """
        q_topo = self._calculate_modularity(
            communities, entity_ids, adjacency, edge_weights
        )
        q_attr = 0.0
        if self._attribute_modularity_weight > 0.0 and edge_weights:
            q_attr = self._calculate_attribute_modularity(
                communities, entity_ids, adjacency, edge_weights
            )
        alpha = self._attribute_modularity_weight
        q_combined = (1.0 - alpha) * q_topo + alpha * q_attr
        return q_combined, q_topo, q_attr

    # --------------------------------------------------------
    # 边权重提取 (从 KnowledgeStore)
    # --------------------------------------------------------

    def _extract_edge_weights_from_store(
        self,
        store: KnowledgeStore,
        entity_id_set: set[str],
        edge_weights: dict[tuple[str, str], float],
    ) -> None:
        """从 KnowledgeStore 提取边权重.

        若提供 edge_weight_fn, 使用之; 否则基于 attr_weight_map 与三元组属性计算。

        边权重计算策略:
        1. 若 edge_weight_fn 提供: w = edge_weight_fn(triple)
        2. 否则: w = base_weight * Σ(attr_weight_map[attr] for attr in triple_attributes)
           - base_weight = 1.0
           - triple_attributes 来源: triple.predicate, triple.metadata 中的标签
        3. 若 attr_weight_map 为空: w = 1.0 (默认)

        Args:
            store: 知识存储
            entity_id_set: 实体 ID 集合 (仅提取两端都在此集合内的边)
            edge_weights: 输出参数, 填充 {（u, v）: weight}
        """
        # 获取所有三元组
        # 由于 store 没有直接的 list_triples 接口, 通过实体遍历
        seen_triple_ids: set[str] = set()

        for eid in entity_id_set:
            triples = store.get_entity_triples(eid, direction="both")
            for triple in triples:
                if triple.triple_id in seen_triple_ids:
                    continue
                seen_triple_ids.add(triple.triple_id)

                # 仅处理实体-实体边 (非字面值)
                if triple.object_is_literal:
                    continue
                if not triple.object_id or triple.object_id not in entity_id_set:
                    continue
                if triple.subject_id not in entity_id_set:
                    continue

                u = triple.subject_id
                v = triple.object_id

                # 计算边权重
                if self._edge_weight_fn is not None:
                    try:
                        w = float(self._edge_weight_fn(triple))
                    except Exception:
                        logger.warning("edge_weight_fn 异常, 使用默认权重 1.0")
                        w = 1.0
                elif self._attr_weight_map:
                    w = 1.0
                    # 基于 predicate 与 metadata 属性加权
                    if triple.predicate in self._attr_weight_map:
                        w *= self._attr_weight_map[triple.predicate]
                    # 检查 metadata 中的属性
                    if hasattr(triple, "metadata") and triple.metadata:
                        for attr_key, attr_val in triple.metadata.items():
                            if attr_key in self._attr_weight_map:
                                try:
                                    w *= float(attr_val) * self._attr_weight_map[attr_key]
                                except (TypeError, ValueError):
                                    pass
                    # 考虑 confidence
                    if hasattr(triple, "confidence") and triple.confidence is not None:
                        try:
                            w *= max(0.0, float(triple.confidence))
                        except (TypeError, ValueError):
                            pass
                else:
                    w = 1.0

                # 累加权重 (同一对实体可能有多条三元组)
                key = (u, v)
                edge_weights[key] = edge_weights.get(key, 0.0) + w

    # --------------------------------------------------------
    # 社区摘要与实体获取
    # --------------------------------------------------------

    def generate_summary(self, community: Community, store: KnowledgeStore) -> str:
        """生成社区摘要 (借鉴 GraphRAG 社区报告生成).

        从社区中的实体和三元组提取关键信息, 生成摘要文本。
        完整的 GraphRAG 实现使用 LLM 生成摘要, 此处使用规则方法。

        Args:
            community: 社区对象
            store: 知识存储

        Returns:
            社区摘要文本
        """
        entities = self.get_community_entities(community, store)

        if not entities:
            return f"社区 {community.community_id}: 空社区"

        parts: list[str] = []

        # 实体摘要 (最多 15 个)
        entity_summaries: list[str] = []
        for entity in entities[:15]:
            type_label = (
                entity.entity_type.value
                if hasattr(entity.entity_type, "value")
                else str(entity.entity_type)
            )
            desc = entity.description[:80] if entity.description else "无描述"
            entity_summaries.append(f"{entity.name}({type_label}): {desc}")
        parts.append("实体: " + "; ".join(entity_summaries))

        if len(entities) > 15:
            parts.append(f"... 等 {len(entities)} 个实体")

        # 三元组摘要 (最多 10 个)
        triple_summaries: list[str] = []
        for triple_id in community.triple_ids[:10]:
            triple = store.get_triple(triple_id)
            if triple is None:
                continue

            subj = store.get_entity(triple.subject_id)
            subj_name = subj.name if subj else triple.subject_id

            if triple.object_is_literal:
                obj_repr = str(triple.object_value)
            else:
                obj = store.get_entity(triple.object_id) if triple.object_id else None
                obj_repr = obj.name if obj else triple.object_id

            triple_summaries.append(
                f"{subj_name} --{triple.predicate}--> {obj_repr}"
            )

        if triple_summaries:
            parts.append("关系: " + "; ".join(triple_summaries))

        # 社区统计
        parts.append(
            f"统计: {community.entity_count} 实体, "
            f"{len(community.triple_ids)} 关系"
        )

        # 层级信息
        if community.level > 0 or community.parent_id is not None:
            parts.append(
                f"层级: L{community.level}, 父社区={community.parent_id}, "
                f"子社区数={len(community.child_ids)}"
            )

        return " | ".join(parts)

    def get_community_entities(
        self,
        community: Community,
        store: KnowledgeStore,
    ) -> list[Any]:
        """获取社区中的所有实体 (借鉴 GraphRAG 社区级检索).

        Args:
            community: 社区对象
            store: 知识存储

        Returns:
            社区中的实体列表 (KnowledgeEntity 对象)
        """
        entities: list[Any] = []
        for eid in community.entity_ids:
            entity = store.get_entity(eid)
            if entity is not None:
                entities.append(entity)
        return entities

    # --------------------------------------------------------
    # 层次结构辅助查询 (便捷方法)
    # --------------------------------------------------------

    @staticmethod
    def get_entity_ancestry(
        hierarchy: CommunityHierarchy, entity_id: str
    ) -> list[Community]:
        """获取实体在层次结构中的祖先路径 (便捷方法)."""
        return hierarchy.get_ancestry(entity_id)

    @staticmethod
    def get_entities_at_level(
        hierarchy: CommunityHierarchy, level: int
    ) -> list[set[str]]:
        """获取指定层级的所有社区实体集合列表 (便捷方法)."""
        communities = hierarchy.get_level(level)
        return [set(c.entity_ids) for c in communities]

    # --------------------------------------------------------
    # 内部辅助方法
    # --------------------------------------------------------

    @staticmethod
    def _normalize_adjacency(
        entity_ids: list[str],
        adjacency: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """规范化邻接表.

        - 确保所有实体都有邻接条目
        - 去除自环
        - 去除重复邻居
        - 确保无向性 (双向边)

        Args:
            entity_ids: 实体 ID 列表
            adjacency: 原始邻接表

        Returns:
            规范化后的邻接表
        """
        entity_id_set = set(entity_ids)
        normalized: dict[str, set[str]] = {
            eid: set() for eid in entity_ids
        }

        for eid in entity_ids:
            neighbors = adjacency.get(eid, [])
            for neighbor in neighbors:
                # 去除自环
                if neighbor == eid:
                    continue
                # 仅保留有效实体
                if neighbor not in entity_id_set:
                    continue
                # 添加双向边
                normalized[eid].add(neighbor)
                normalized[neighbor].add(eid)

        return {eid: sorted(neighbors) for eid, neighbors in normalized.items()}

    @staticmethod
    def _populate_triple_ids(
        community: Community,
        store: KnowledgeStore,
    ) -> None:
        """填充社区的三元组 ID 列表.

        遍历社区中所有实体的关联三元组, 将两端实体都在社区内的三元组
        加入社区的三元组列表。

        Args:
            community: 社区对象 (原地修改 triple_ids)
            store: 知识存储
        """
        community_set = set(community.entity_ids)
        seen_triple_ids: set[str] = set()
        # 清空已有 triple_ids 以避免重复填充
        community.triple_ids = []

        for eid in community.entity_ids:
            triples = store.get_entity_triples(eid, direction="both")
            for triple in triples:
                if triple.triple_id in seen_triple_ids:
                    continue

                # 确定另一端实体
                if triple.subject_id == eid:
                    other_id = triple.object_id
                elif triple.object_id == eid:
                    other_id = triple.subject_id
                else:
                    continue

                # 仅当两端都在社区内时加入
                if other_id in community_set:
                    community.triple_ids.append(triple.triple_id)
                    seen_triple_ids.add(triple.triple_id)


__all__ = [
    "CommunityAlgorithm",
    "Community",
    "CommunityDetectionResult",
    "CommunityHierarchy",
    "CommunityDetector",
]
