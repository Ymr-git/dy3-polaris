"""L3 领域知识层 — HNSW 向量索引.

借鉴 Malkov & Yashunin 2016 论文 "Efficient and robust approximate nearest
neighbor search using HNSW"、Milvus HNSW 实现、Faiss HNSW 实现。

HNSW (Hierarchical Navigable Small World) 是一种基于多层图的
近似最近邻 (ANN) 搜索算法。相比暴力搜索 (VectorIndex), HNSW 在大规模
向量集合上提供了显著更快的搜索速度, 同时保持较高的召回率。

核心思想:
1. 多层图结构: 上层稀疏 (少数节点), 下层密集 (全部节点)
2. 层级分配: 每个节点按指数衰减概率分配到最高层
3. 搜索: 从最上层贪婪搜索, 逐层下降, 底层进行 ef_search 搜索
4. 连接: 新节点与最近邻建立双向连接, 保持每层最大连接数限制

算法参考:
- Malkov, Y.A. & Yashunin, D.A. (2016). "Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World graphs."
  arXiv:1603.09320.
- Milvus HNSW 索引实现
- Faiss HNSW 索引实现

复杂度:
- 构建: O(N * log(N) * ef_construction)
- 搜索: O(log(N) * ef_search)
- 内存: O(N * M)

与 VectorIndex 的关系:
- VectorIndex: 暴力搜索 (FLAT), 精确但慢, 适合小规模数据
- HNSWIndex: 近似搜索, 快速但非精确, 适合大规模数据
- 两者接口兼容, 可按需替换

Usage::

    from dy3_polaris.l3.hnsw_index import HNSWIndex

    index = HNSWIndex(dim=128, M=16, ef_construction=200, metric="cosine")
    index.add("vec-1", [0.1, 0.2, ...], metadata={"source": "doc1"})
    index.add("vec-2", [0.3, 0.4, ...], metadata={"source": "doc2"})

    # 搜索
    results = index.search([0.15, 0.25, ...], top_k=10)

    # 带预过滤的搜索 (借鉴 Weaviate 预过滤模式)
    results = index.search(
        [0.15, 0.25, ...], top_k=10,
        filter_fn=lambda meta: meta.get("source") == "doc1",
    )

    # 获取统计信息
    stats = index.get_stats()

    # 动态调整搜索精度
    index.set_ef_search(100)  # 提高召回率
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import random
import threading
from typing import Any

logger = logging.getLogger(__name__)


class HNSWIndex:
    """HNSW 向量索引 (借鉴 Malkov & Yashunin 2016 + Milvus HNSW + Faiss HNSW).

    HNSW (Hierarchical Navigable Small World) 是一种基于多层图的
    近似最近邻 (ANN) 搜索算法。

    核心思想:
    1. 多层图结构: 上层稀疏 (少数节点), 下层密集 (全部节点)
    2. 层级分配: 每个节点按指数衰减概率分配到最高层
    3. 搜索: 从最上层贪婪搜索, 逐层下降, 底层进行 ef_search 搜索
    4. 连接: 新节点与最近邻建立双向连接, 保持每层最大连接数限制

    复杂度:
    - 构建: O(N * log(N) * ef_construction)
    - 搜索: O(log(N) * ef_search)
    - 内存: O(N * M)

    参数说明:
    - M: 每个节点的最大连接数 (层 0 以上), 推荐 16-64
    - M_max0: 层 0 的最大连接数 (通常 2*M)
    - ef_construction: 构建时候选列表大小, 推荐 200-500
    - ef_search: 搜索时候选列表大小, 推荐 50-200
    - m_L: 层级生成参数, 通常 1/ln(M)

    Attributes:
        _M: 最大连接数
        _M_max0: 层 0 最大连接数
        _ef_construction: 构建时候选列表大小
        _ef_search: 搜索时候选列表大小
        _m_L: 层级生成参数
        _metric: 距离度量 ("cosine" 或 "euclidean")
        _dim: 向量维度
        _vectors: 向量存储 {id: vector}
        _metadata: 元数据存储 {id: metadata}
        _graph: 多层图 {layer: {node_id: [neighbor_ids]}}
        _entry_point: 入口节点 ID
        _max_level: 当前最大层级
        _element_levels: {node_id: max_level}
        _lock: 线程安全锁
    """

    def __init__(
        self,
        dim: int = 0,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: str = "cosine",
    ) -> None:
        """初始化 HNSW 索引.

        Args:
            dim: 向量维度 (0 表示自动推断)
            M: 每个节点的最大连接数 (层 0 以上), 推荐 16-64
            ef_construction: 构建时候选列表大小, 推荐 200-500
            ef_search: 搜索时候选列表大小, 推荐 50-200
            metric: 距离度量 ("cosine" 或 "euclidean")

        Raises:
            ValueError: 参数不合法
        """
        if M < 2:
            raise ValueError("M 必须 >= 2")
        if ef_construction < 1:
            raise ValueError("ef_construction 必须 >= 1")
        if ef_search < 1:
            raise ValueError("ef_search 必须 >= 1")
        if metric not in ("cosine", "euclidean"):
            raise ValueError(
                f"不支持的度量: {metric}, 支持 'cosine' 或 'euclidean'"
            )

        self._M: int = M
        self._M_max0: int = M * 2
        self._ef_construction: int = ef_construction
        self._ef_search: int = ef_search
        self._m_L: float = 1.0 / math.log(M) if M > 1 else 1.0
        self._metric: str = metric
        self._dim: int = dim

        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._graph: dict[int, dict[str, list[str]]] = {}
        self._entry_point: str | None = None
        self._max_level: int = -1
        self._element_levels: dict[str, int] = {}
        self._lock = threading.RLock()

    # ============================================================
    # 公共 API
    # ============================================================

    def add(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加向量到索引.

        如果 vector_id 已存在, 先移除旧向量再重新插入。

        Args:
            vector_id: 向量唯一标识
            vector: 密集向量
            metadata: 元数据 (用于预过滤)
        """
        with self._lock:
            if not vector:
                logger.warning("空向量, 跳过: %s", vector_id)
                return

            # 自动推断维度
            if self._dim == 0:
                self._dim = len(vector)

            # 维度校验
            if len(vector) != self._dim:
                logger.warning(
                    "向量维度 %d 与索引维度 %d 不匹配, 跳过: %s",
                    len(vector), self._dim, vector_id,
                )
                return

            # 如果已存在, 先移除 (重新插入)
            if vector_id in self._vectors:
                logger.debug("向量已存在, 重新插入: %s", vector_id)
                self._remove_locked(vector_id)

            # 存储向量和元数据
            vec_copy = list(vector)
            self._vectors[vector_id] = vec_copy
            self._metadata[vector_id] = metadata or {}

            # 插入图结构
            self._insert(vector_id, vec_copy)

            logger.debug(
                "添加向量 %s, 当前节点数 %d", vector_id, len(self._vectors)
            )

    def remove(self, vector_id: str) -> bool:
        """从索引中移除向量.

        移除节点时:
        1. 从所有层的邻居列表中移除引用
        2. 如果是入口节点, 重新选择入口节点

        Args:
            vector_id: 向量唯一标识

        Returns:
            是否成功移除
        """
        with self._lock:
            return self._remove_locked(vector_id)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_fn: Any = None,
    ) -> list[tuple[str, float]]:
        """向量相似性搜索.

        使用 HNSW 算法进行近似最近邻搜索。
        支持 filter_fn 预过滤 (借鉴 Weaviate 预过滤模式):
        - 搜索时遍历完整图结构进行导航
        - 仅将通过过滤的节点加入结果集

        Args:
            query_vector: 查询向量
            top_k: 返回前 k 个结果
            filter_fn: 预过滤函数 (metadata -> bool)

        Returns:
            [(vector_id, score)] 按分数降序排列。
            cosine 度量: score = cosine_similarity (越高越相似)。
            euclidean 度量: score = -distance (越高越近)。
        """
        with self._lock:
            if not self._vectors or self._entry_point is None:
                return []

            # 维度校验
            if self._dim > 0 and len(query_vector) != self._dim:
                logger.warning(
                    "查询向量维度 %d 与索引维度 %d 不匹配",
                    len(query_vector), self._dim,
                )
                return []

            # HNSW 搜索
            raw_results = self._search_knn(query_vector, top_k, filter_fn)

            # 将距离转换为分数 (分数越高越相似)
            scored_results: list[tuple[str, float]] = []
            for vid, dist in raw_results:
                if self._metric == "cosine":
                    score = 1.0 - dist  # 距离转相似度
                else:
                    score = -dist  # 负距离 (越高越近)
                scored_results.append((vid, score))

            return scored_results

    def get(self, vector_id: str) -> tuple[list[float], dict[str, Any]] | None:
        """获取向量和元数据.

        Args:
            vector_id: 向量唯一标识

        Returns:
            (vector, metadata) 或 None (不存在)
        """
        with self._lock:
            if vector_id not in self._vectors:
                return None
            return (
                list(self._vectors[vector_id]),
                dict(self._metadata.get(vector_id, {})),
            )

    def size(self) -> int:
        """返回索引中的向量数量."""
        with self._lock:
            return len(self._vectors)

    @property
    def dim(self) -> int:
        """向量维度."""
        return self._dim

    def set_ef_search(self, ef: int) -> None:
        """设置搜索时候选列表大小.

        较大的 ef_search 提高召回率但降低搜索速度。

        Args:
            ef: 搜索时候选列表大小 (>= 1)

        Raises:
            ValueError: ef < 1
        """
        if ef < 1:
            raise ValueError("ef_search 必须 >= 1")
        self._ef_search = ef
        logger.debug("设置 ef_search = %d", ef)

    def clear(self) -> None:
        """清空索引.

        保留维度和参数配置, 仅清空数据和图结构。
        """
        with self._lock:
            self._vectors.clear()
            self._metadata.clear()
            self._graph.clear()
            self._element_levels.clear()
            self._entry_point = None
            self._max_level = -1
            logger.debug("索引已清空")

    def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息.

        Returns:
            包含以下字段的字典:
            - node_count: 节点总数
            - max_level: 当前最大层级
            - avg_connections: 层 0 平均连接数
            - memory_estimate: 估算内存使用量 (字节)
            - layer_distribution: 各层节点数 {layer_str: count}
        """
        with self._lock:
            node_count = len(self._vectors)

            # 层分布和连接统计
            layer_distribution: dict[str, int] = {}
            total_connections_l0 = 0
            for level, layer_graph in sorted(self._graph.items()):
                layer_node_count = len(layer_graph)
                layer_distribution[str(level)] = layer_node_count
                if level == 0:
                    for neighbors in layer_graph.values():
                        total_connections_l0 += len(neighbors)

            avg_connections = (
                total_connections_l0 / node_count if node_count > 0 else 0.0
            )

            # 内存估算
            # 向量: node_count * dim * 8 bytes (float64)
            vector_mem = node_count * self._dim * 8
            # 元数据: 粗略估算每个节点 200 字节
            metadata_mem = node_count * 200
            # 图结构: 每条连接约 50 字节 (字符串 ID + 列表开销)
            graph_mem = sum(
                len(neighbors) * 50
                for layer_graph in self._graph.values()
                for neighbors in layer_graph.values()
            )
            memory_estimate = vector_mem + metadata_mem + graph_mem

            stats: dict[str, Any] = {
                "node_count": node_count,
                "max_level": self._max_level if self._max_level >= 0 else 0,
                "avg_connections": round(avg_connections, 2),
                "memory_estimate": memory_estimate,
                "layer_distribution": layer_distribution,
            }

            logger.debug("索引统计: %s", json.dumps(stats, ensure_ascii=False))
            return stats

    # ============================================================
    # 内部算法实现
    # ============================================================

    def _random_level(self) -> int:
        """按指数衰减概率分配层级 (Malkov 论文 Eq. 2).

        层级 l 的概率: P(l) = exp(-l / m_L) * (1 - exp(-1 / m_L))
        等价于: l = floor(-ln(uniform()) * m_L)

        上层节点稀疏, 底层节点密集, 形成跳表式的层次结构。

        Returns:
            分配的层级 (>= 0)
        """
        r = random.random()
        # 避免 log(0) (random.random() 理论上可能返回 0.0)
        if r < 1e-10:
            r = 1e-10
        return int(-math.log(r) * self._m_L)

    def _distance(self, a: list[float], b: list[float]) -> float:
        """计算两个向量之间的距离.

        cosine 度量: distance = 1 - cosine_similarity (越小越近)
        euclidean 度量: 欧氏距离 (越小越近)

        Args:
            a: 向量 A
            b: 向量 B

        Returns:
            距离值 (越小表示越近)
        """
        if self._metric == "cosine":
            return self._cosine_distance(a, b)
        return self._euclidean_distance(a, b)

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        """余弦距离 (1 - cosine_similarity).

        距离越小表示越相似。范围 [0, 2]。
        """
        if not a or not b or len(a) != len(b):
            return 1.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - dot / (norm_a * norm_b)

    @staticmethod
    def _euclidean_distance(a: list[float], b: list[float]) -> float:
        """欧氏距离."""
        if not a or not b or len(a) != len(b):
            return float("inf")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _search_layer(
        self,
        query: list[float],
        entry_points: list[str],
        ef: int,
        layer: int,
        filter_fn: Any = None,
    ) -> list[tuple[str, float]]:
        """在指定层搜索最近邻 (贪心搜索 + 动态候选列表).

        使用优先队列维护候选列表 (Malkov 论文 Algorithm 2):
        - candidate_queue: 待探索节点 (最小堆, 按距离升序)
        - result_heap: 已找到的最近邻 (最大堆, 堆顶为最远结果)
        - visited: 已访问节点集合

        搜索策略:
        1. 从 entry_points 初始化候选队列和结果堆
        2. 从候选队列取出最近节点
        3. 如果该节点距离 > 结果堆中最差节点距离且结果已满, 停止
        4. 探索该节点的邻居, 将更近的加入候选队列和结果堆
        5. 结果堆保持 ef 个最近邻

        当 filter_fn 提供时 (借鉴 Weaviate 预过滤):
        - 所有节点都参与图导航 (保证连通性)
        - 仅通过过滤的节点加入结果堆

        Args:
            query: 查询向量
            entry_points: 入口节点 ID 列表
            ef: 候选列表大小
            layer: 搜索的层级
            filter_fn: 可选的过滤函数 (metadata -> bool)

        Returns:
            [(node_id, distance)] 按距离升序排列
        """
        visited: set[str] = set()
        # 最小堆: (distance, node_id) — 待探索节点
        candidate_queue: list[tuple[float, str]] = []
        # 最大堆: (-distance, node_id) — 已找到的最近邻
        result_heap: list[tuple[float, str]] = []

        # 初始化: 从入口点开始
        for ep in entry_points:
            if ep in visited or ep not in self._vectors:
                continue
            dist = self._distance(query, self._vectors[ep])
            heapq.heappush(candidate_queue, (dist, ep))
            visited.add(ep)

            # 加入结果堆 (如果通过过滤)
            if filter_fn is None or filter_fn(self._metadata.get(ep, {})):
                heapq.heappush(result_heap, (-dist, ep))
                if len(result_heap) > ef:
                    heapq.heappop(result_heap)

        # 贪心搜索主循环
        while candidate_queue:
            dist, node = heapq.heappop(candidate_queue)

            # 停止条件: 候选队列中最优节点比结果堆中最差节点更远
            # (且结果堆已满), 此时无法找到更好的结果
            if result_heap and len(result_heap) >= ef:
                worst_result_dist = -result_heap[0][0]
                if dist > worst_result_dist:
                    break

            # 探索邻居
            neighbors = self._graph.get(layer, {}).get(node, [])
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                if neighbor not in self._vectors:
                    continue

                neighbor_dist = self._distance(query, self._vectors[neighbor])

                # 判断是否值得探索
                should_explore = False
                if len(result_heap) < ef:
                    should_explore = True
                elif result_heap:
                    worst_result_dist = -result_heap[0][0]
                    if neighbor_dist < worst_result_dist:
                        should_explore = True
                else:
                    should_explore = True

                if should_explore:
                    heapq.heappush(
                        candidate_queue, (neighbor_dist, neighbor)
                    )

                    # 加入结果堆 (如果通过过滤)
                    if filter_fn is None or filter_fn(
                        self._metadata.get(neighbor, {})
                    ):
                        heapq.heappush(
                            result_heap, (-neighbor_dist, neighbor)
                        )
                        if len(result_heap) > ef:
                            heapq.heappop(result_heap)

        # 转换为有序列表 (距离升序)
        results = [
            (node_id, -neg_dist) for neg_dist, node_id in result_heap
        ]
        results.sort(key=lambda x: x[1])
        return results

    def _select_neighbors(
        self,
        candidates: list[tuple[str, float]],
        M: int,
    ) -> list[str]:
        """从候选列表中选择 M 个最近邻 (简单最近邻选择).

        使用简单最近邻选择策略 (Malkov 论文 Algorithm 3 的简单版本):
        按距离排序, 取前 M 个。

        更高级的选择策略 (如启发式选择, 考虑多样性) 可在未来实现。

        Args:
            candidates: 候选列表 [(node_id, distance)]
            M: 选择的邻居数量

        Returns:
            选中的邻居 ID 列表
        """
        sorted_candidates = sorted(candidates, key=lambda x: x[1])
        return [node_id for node_id, _ in sorted_candidates[:M]]

    def _insert(self, vector_id: str, vector: list[float]) -> None:
        """插入新节点 (Malkov 论文 Algorithm 1).

        插入流程:
        1. 随机分配层级 level
        2. 从最高层到 level+1: 贪心搜索 (ef=1) 找到每层最近节点
        3. 从 level 到 0: 搜索 ef_construction 候选, 选择 M 个连接
        4. 建立双向连接, 必要时修剪多余连接
        5. 如果 level > max_level, 更新入口节点

        Args:
            vector_id: 节点 ID
            vector: 节点向量
        """
        level = self._random_level()

        # 首个节点: 直接设为入口
        if self._entry_point is None:
            self._entry_point = vector_id
            self._max_level = level
            self._element_levels[vector_id] = level
            for l in range(level + 1):
                if l not in self._graph:
                    self._graph[l] = {}
                self._graph[l][vector_id] = []
            logger.debug("插入首节点 %s, 层级 %d", vector_id, level)
            return

        self._element_levels[vector_id] = level

        # 初始化新节点在所有 <= level 层的图结构
        for l in range(level + 1):
            if l not in self._graph:
                self._graph[l] = {}
            if vector_id not in self._graph[l]:
                self._graph[l][vector_id] = []

        # 阶段 1: 从最高层到 level+1, 贪心搜索 (ef=1)
        # 在这些层中, 新节点不存在, 仅找到每层最近节点作为下降入口
        ep: list[str] = [self._entry_point]
        for l in range(self._max_level, level, -1):
            results = self._search_layer(vector, ep, 1, l)
            if results:
                ep = [results[0][0]]  # 取最近节点

        # 阶段 2: 从 min(level, max_level) 到 0, 搜索并建立连接
        for l in range(min(level, self._max_level), -1, -1):
            # 搜索 ef_construction 个候选邻居
            candidates = self._search_layer(
                vector, ep, self._ef_construction, l
            )

            # 选择 M 个最近邻居
            selected = self._select_neighbors(candidates, self._M)

            # 建立双向连接: 新节点 -> 邻居
            self._graph[l][vector_id] = list(selected)

            # 建立双向连接: 邻居 -> 新节点, 并修剪超额连接
            max_conn = self._M_max0 if l == 0 else self._M
            for neighbor_id in selected:
                neighbor_list = self._graph[l].get(neighbor_id, [])
                if vector_id not in neighbor_list:
                    neighbor_list.append(vector_id)
                    self._graph[l][neighbor_id] = neighbor_list

                    # 修剪超额连接 (Malkov 论文 Algorithm 4)
                    if len(neighbor_list) > max_conn:
                        self._prune_connections(neighbor_id, l, max_conn)

            # 更新下一层的入口点 (使用当前层所有搜索结果)
            if candidates:
                ep = [node_id for node_id, _ in candidates]

        # 如果新节点层级更高, 更新入口节点
        if level > self._max_level:
            self._max_level = level
            self._entry_point = vector_id
            logger.debug(
                "更新入口节点为 %s, 最大层级 %d", vector_id, level
            )

    def _search_knn(
        self,
        query: list[float],
        top_k: int,
        filter_fn: Any = None,
    ) -> list[tuple[str, float]]:
        """K-NN 搜索 (Malkov 论文 Algorithm 2).

        搜索流程:
        1. 从最高层到层 1: 贪心搜索 (ef=1), 逐层下降找到最近入口
        2. 层 0: ef_search 搜索, 收集候选结果
        3. 返回 top_k 最近邻

        Args:
            query: 查询向量
            top_k: 返回前 k 个结果
            filter_fn: 可选的过滤函数 (metadata -> bool)

        Returns:
            [(node_id, distance)] 按距离升序排列
        """
        if self._entry_point is None or not self._vectors:
            return []

        ep: list[str] = [self._entry_point]

        # 阶段 1: 从最高层到层 1, 贪心搜索 (ef=1)
        for l in range(self._max_level, 0, -1):
            results = self._search_layer(query, ep, 1, l)
            if results:
                ep = [results[0][0]]

        # 阶段 2: 层 0, ef_search 搜索
        ef = max(self._ef_search, top_k)
        results = self._search_layer(query, ep, ef, 0, filter_fn)

        return results[:top_k]

    def _prune_connections(
        self,
        node_id: str,
        layer: int,
        M_max: int,
    ) -> None:
        """修剪节点的连接, 保留 M_max 个最近邻.

        当节点的连接数超过最大限制时, 按距离排序保留最近的 M_max 个。
        (Malkov 论文 Algorithm 4 的简单版本)

        修剪仅影响被修剪节点的连接列表, 不删除反向引用
        (与其他 HNSW 实现如 Faiss/Milvus 一致)。

        Args:
            node_id: 节点 ID
            layer: 层级
            M_max: 最大连接数
        """
        neighbors = self._graph.get(layer, {}).get(node_id, [])
        if len(neighbors) <= M_max:
            return

        node_vector = self._vectors[node_id]
        candidates: list[tuple[float, str]] = []
        for neighbor_id in neighbors:
            if neighbor_id not in self._vectors:
                continue
            dist = self._distance(node_vector, self._vectors[neighbor_id])
            candidates.append((dist, neighbor_id))

        # 按距离排序, 保留最近的 M_max 个
        candidates.sort(key=lambda x: x[0])
        self._graph[layer][node_id] = [
            nid for _, nid in candidates[:M_max]
        ]

    def _remove_locked(self, vector_id: str) -> bool:
        """移除节点 (已持有锁).

        移除流程:
        1. 从所有层的邻居列表中移除引用
        2. 从向量、元数据、层级记录中移除
        3. 如果是入口节点, 重新选择层级最高的节点作为新入口

        注意: 由于修剪操作可能产生单向连接, 移除时仅清理直接邻居
        的引用。残留的悬空引用在搜索时通过向量存在性检查自动跳过。

        Args:
            vector_id: 节点 ID

        Returns:
            是否成功移除
        """
        if vector_id not in self._vectors:
            return False

        node_level = self._element_levels.get(vector_id, 0)

        # 从所有层移除节点的连接和引用
        for l in range(node_level + 1):
            layer_graph = self._graph.get(l, {})
            neighbors = layer_graph.get(vector_id, [])

            # 从邻居的列表中移除该节点 (清理反向引用)
            for neighbor_id in neighbors:
                neighbor_list = layer_graph.get(neighbor_id, [])
                if vector_id in neighbor_list:
                    neighbor_list.remove(vector_id)
                    layer_graph[neighbor_id] = neighbor_list

            # 移除节点本身
            if vector_id in layer_graph:
                del layer_graph[vector_id]

        # 从存储中移除
        del self._vectors[vector_id]
        if vector_id in self._metadata:
            del self._metadata[vector_id]
        if vector_id in self._element_levels:
            del self._element_levels[vector_id]

        # 处理入口节点移除
        if vector_id == self._entry_point:
            if not self._vectors:
                # 索引为空, 重置状态
                self._entry_point = None
                self._max_level = -1
                self._graph.clear()
            else:
                # 选择层级最高的节点作为新入口
                new_ep = max(
                    self._element_levels,
                    key=lambda k: self._element_levels[k],
                )
                self._entry_point = new_ep
                self._max_level = self._element_levels[new_ep]

                # 清理新入口层级以上的空层
                for l in list(self._graph.keys()):
                    if l > self._max_level:
                        del self._graph[l]

        logger.debug("移除节点 %s", vector_id)
        return True


__all__ = ["HNSWIndex"]
