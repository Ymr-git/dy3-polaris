"""记忆图谱 - 节点/边/查询/衰减/扩散激活.

支持:
- 6 种节点类型 (学习者/知识/技能/评估/资源/会话)
- 5 种边类型 (前置/关联/派生/学习/引用)
- 艾宾浩斯遗忘曲线衰减模型 (strength 指数衰减)
- 扩散激活 (spreading activation, 认知科学启发)
- DFS 环检测 (防止前置依赖环, 保证 DAG)
- BFS 最短路径查找
- 多条件节点搜索 (类型/强度/元数据/内容)
- 子图提取
- 导出/导入
- 线程安全操作

与广播协议的关系:
- 广播总线负责实时事件分发, 记忆图谱负责持久化学情记忆
- 广播事件的 payload 可触发记忆图谱的 reinforce/spreading_activation
- 衰减机制确保长期未访问的学情记忆自动弱化
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict, deque
from enum import Enum
from typing import Any

from dy3_polaris.l6.core.exceptions import (
    EdgeNotFoundError,
    GraphCycleError,
    MemoryGraphError,
    NodeNotFoundError,
)


# ============================================================
# 枚举
# ============================================================

class NodeType(str, Enum):
    """记忆图谱节点类型."""
    LEARNER = "learner"           # 学习者画像
    KNOWLEDGE = "knowledge"       # 知识点
    SKILL = "skill"               # 技能
    ASSESSMENT = "assessment"     # 评估结果
    RESOURCE = "resource"         # 学习资源
    SESSION = "session"           # 学习会话


class EdgeType(str, Enum):
    """记忆图谱边类型."""
    PREREQUISITE = "prerequisite"  # 前置依赖 (A 是 B 的前置)
    RELATED = "related"            # 关联
    DERIVED = "derived"            # 派生
    LEARNED = "learned"            # 学习关系 (学习者 → 知识点)
    REFERENCES = "references"      # 引用


# ============================================================
# 数据模型
# ============================================================

class MemoryNode:
    """记忆图谱节点.

    使用 __slots__ 优化内存占用。
    strength 表示记忆强度 [0, 1], 随时间衰减, 访问时增强。
    """

    __slots__ = (
        "node_id",
        "node_type",
        "content",
        "metadata",
        "created_at",
        "last_accessed_at",
        "access_count",
        "strength",
    )

    def __init__(
        self,
        node_id: str | None = None,
        node_type: NodeType = NodeType.KNOWLEDGE,
        content: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        strength: float = 1.0,
    ) -> None:
        self.node_id = node_id or uuid.uuid4().hex[:12]
        self.node_type = node_type
        self.content = content or {}
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.last_accessed_at = self.created_at
        self.access_count = 0
        self.strength = max(0.0, min(1.0, strength))

    def touch(self) -> None:
        """更新访问时间和计数."""
        self.last_accessed_at = time.time()
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "strength": self.strength,
        }


class MemoryEdge:
    """记忆图谱边."""

    __slots__ = (
        "source_id",
        "target_id",
        "edge_type",
        "weight",
        "created_at",
        "metadata",
    )

    def __init__(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.RELATED,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = max(0.0, min(1.0, weight))
        self.created_at = time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


# ============================================================
# 度量收集
# ============================================================

class MemoryGraphMetrics:
    """线程安全记忆图谱度量收集器."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._access_count = 0
        self._decay_count = 0
        self._pruned_count = 0
        self._spreading_count = 0
        self._cycle_rejected = 0

    def record_access(self) -> None:
        with self._lock:
            self._access_count += 1

    def record_decay(self, pruned: int = 0) -> None:
        with self._lock:
            self._decay_count += 1
            self._pruned_count += pruned

    def record_spreading(self) -> None:
        with self._lock:
            self._spreading_count += 1

    def record_cycle_rejected(self) -> None:
        with self._lock:
            self._cycle_rejected += 1

    def export(self) -> dict[str, Any]:
        with self._lock:
            return {
                "access_count": self._access_count,
                "decay_count": self._decay_count,
                "pruned_count": self._pruned_count,
                "spreading_count": self._spreading_count,
                "cycle_rejected": self._cycle_rejected,
            }

    def reset(self) -> None:
        with self._lock:
            self._access_count = 0
            self._decay_count = 0
            self._pruned_count = 0
            self._spreading_count = 0
            self._cycle_rejected = 0


# ============================================================
# 记忆图谱引擎
# ============================================================

class MemoryGraph:
    """记忆图谱引擎.

    核心功能:
    - 节点管理: add_node / remove_node / get_node / touch_node
    - 边管理: add_edge / remove_edge / get_edge (含 DFS 环检测)
    - 查询: neighbors / find_path / search / subgraph
    - 衰减: decay (艾宾浩斯模型, strength *= factor) / prune (清除低强度节点)
    - 强化: reinforce / spreading_activation (认知科学扩散激活)
    - 导出/导入

    线程安全: 所有公共方法均受 ``threading.RLock`` 保护。

    衰减模型:
    - 每次 decay() 调用, 所有节点 strength *= decay_factor
    - strength < min_strength 的节点被自动清除 (连同关联边)
    - touch_node / reinforce / spreading_activation 会增强 strength

    扩散激活:
    - 从起始节点沿出边扩散, 每层衰减
    - 激活量 = 上层激活 * 边权重 * 衰减率
    - 激活量 > 0.01 的节点会被部分强化

    Example::

        graph = MemoryGraph(decay_factor=0.9, min_strength=0.05)
        graph.add_node("kp-1", NodeType.KNOWLEDGE, {"title": "化学键"})
        graph.add_node("kp-2", NodeType.KNOWLEDGE, {"title": "分子轨道"})
        graph.add_edge("kp-1", "kp-2", EdgeType.PREREQUISITE)  # 化学键是分子轨道的前置
        graph.spreading_activation("kp-1")  # 访问化学键, 扩散激活分子轨道
        graph.decay()  # 衰减所有节点
    """

    def __init__(
        self,
        decay_factor: float = 0.95,
        min_strength: float = 0.01,
        spreading_depth: int = 2,
        spreading_decay: float = 0.5,
    ) -> None:
        self._decay_factor = decay_factor
        self._min_strength = min_strength
        self._spreading_depth = spreading_depth
        self._spreading_decay = spreading_decay

        self._lock = threading.RLock()
        self._nodes: dict[str, MemoryNode] = {}
        # 邻接表: source_id -> {target_id: MemoryEdge}
        self._edges: dict[str, dict[str, MemoryEdge]] = defaultdict(dict)
        # 反向邻接表: target_id -> [source_id, ...]
        self._reverse_edges: dict[str, list[str]] = defaultdict(list)
        self._metrics = MemoryGraphMetrics()

    # ============================================================
    # 节点管理
    # ============================================================

    def add_node(
        self,
        node_id: str | None = None,
        node_type: NodeType = NodeType.KNOWLEDGE,
        content: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        strength: float = 1.0,
    ) -> MemoryNode:
        """添加节点. 如果 node_id 已存在则更新.

        Args:
            node_id: 节点 ID, None 自动生成 12 位十六进制
            node_type: 节点类型
            content: 内容字典
            metadata: 元数据字典
            strength: 初始强度 [0, 1]

        Returns:
            创建或更新的 MemoryNode
        """
        with self._lock:
            nid = node_id or uuid.uuid4().hex[:12]
            if nid in self._nodes:
                # 更新已有节点
                node = self._nodes[nid]
                node.node_type = node_type
                if content is not None:
                    node.content = content
                if metadata is not None:
                    node.metadata = metadata
                node.strength = max(0.0, min(1.0, strength))
                return node

            node = MemoryNode(
                node_id=nid,
                node_type=node_type,
                content=content,
                metadata=metadata,
                strength=strength,
            )
            self._nodes[nid] = node
            return node

    def remove_node(self, node_id: str) -> bool:
        """删除节点及其所有关联边 (出边和入边).

        Raises:
            NodeNotFoundError: 节点不存在
        """
        with self._lock:
            if node_id not in self._nodes:
                raise NodeNotFoundError(node_id)

            # 删除出边
            if node_id in self._edges:
                del self._edges[node_id]

            # 删除入边
            sources = self._reverse_edges.pop(node_id, [])
            for src in sources:
                if node_id in self._edges.get(src, {}):
                    del self._edges[src][node_id]

            del self._nodes[node_id]
            return True

    def get_node(self, node_id: str) -> MemoryNode:
        """获取节点.

        Raises:
            NodeNotFoundError: 节点不存在
        """
        with self._lock:
            if node_id not in self._nodes:
                raise NodeNotFoundError(node_id)
            return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        """检查节点是否存在."""
        with self._lock:
            return node_id in self._nodes

    def touch_node(self, node_id: str) -> MemoryNode:
        """访问节点 — 更新访问时间/计数, 轻微强化强度.

        Raises:
            NodeNotFoundError: 节点不存在
        """
        with self._lock:
            if node_id not in self._nodes:
                raise NodeNotFoundError(node_id)
            node = self._nodes[node_id]
            node.touch()
            # 轻微强化: +0.05, 上限 1.0
            node.strength = min(1.0, node.strength + 0.05)
            self._metrics.record_access()
            return node

    # ============================================================
    # 边管理
    # ============================================================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.RELATED,
        weight: float = 1.0,
        check_cycle: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """添加边.

        对于 PREREQUISITE 类型边, 默认进行环检测以确保 DAG 性质。

        Args:
            source_id: 源节点 ID
            target_id: 目标节点 ID
            edge_type: 边类型
            weight: 权重 [0, 1]
            check_cycle: 是否检测环 (PREREQUISITE 类型建议 True)
            metadata: 附加元数据

        Returns:
            创建的 MemoryEdge

        Raises:
            NodeNotFoundError: 源或目标节点不存在
            MemoryGraphError: 自环
            GraphCycleError: 检测到环 (PREREQUISITE 类型)
        """
        with self._lock:
            if source_id not in self._nodes:
                raise NodeNotFoundError(source_id)
            if target_id not in self._nodes:
                raise NodeNotFoundError(target_id)
            if source_id == target_id:
                raise MemoryGraphError(
                    "MEMORY_SELF_LOOP",
                    f"Self-loop not allowed: {source_id}",
                )

            # PREREQUISITE 边进行环检测
            if check_cycle and edge_type == EdgeType.PREREQUISITE:
                cycle = self._detect_cycle(target_id, source_id)
                if cycle:
                    self._metrics.record_cycle_rejected()
                    raise GraphCycleError(cycle)

            edge = MemoryEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                metadata=metadata,
            )
            self._edges[source_id][target_id] = edge
            if source_id not in self._reverse_edges[target_id]:
                self._reverse_edges[target_id].append(source_id)
            return edge

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        """删除边.

        Raises:
            EdgeNotFoundError: 边不存在
        """
        with self._lock:
            if source_id not in self._edges or target_id not in self._edges[source_id]:
                raise EdgeNotFoundError(source_id, target_id)
            del self._edges[source_id][target_id]
            if source_id in self._reverse_edges.get(target_id, []):
                self._reverse_edges[target_id].remove(source_id)
            return True

    def get_edge(self, source_id: str, target_id: str) -> MemoryEdge:
        """获取边.

        Raises:
            EdgeNotFoundError: 边不存在
        """
        with self._lock:
            if source_id not in self._edges or target_id not in self._edges[source_id]:
                raise EdgeNotFoundError(source_id, target_id)
            return self._edges[source_id][target_id]

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """检查边是否存在."""
        with self._lock:
            return source_id in self._edges and target_id in self._edges[source_id]

    # ============================================================
    # 查询
    # ============================================================

    def neighbors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
        direction: str = "out",
    ) -> list[str]:
        """获取邻居节点.

        Args:
            node_id: 节点 ID
            edge_type: 边类型过滤, None 表示所有类型
            direction: "out" (出边), "in" (入边), "both" (双向)

        Returns:
            邻居节点 ID 列表

        Raises:
            NodeNotFoundError: 节点不存在
        """
        with self._lock:
            if node_id not in self._nodes:
                raise NodeNotFoundError(node_id)

            result: list[str] = []
            seen: set[str] = set()

            if direction in ("out", "both"):
                for tid, edge in self._edges.get(node_id, {}).items():
                    if edge_type is None or edge.edge_type == edge_type:
                        if tid not in seen:
                            result.append(tid)
                            seen.add(tid)

            if direction in ("in", "both"):
                for src in self._reverse_edges.get(node_id, []):
                    edge = self._edges[src].get(node_id)
                    if edge and (edge_type is None or edge.edge_type == edge_type):
                        if src not in seen:
                            result.append(src)
                            seen.add(src)

            return result

    def find_path(self, source_id: str, target_id: str) -> list[str] | None:
        """BFS 最短路径查找.

        Returns:
            路径节点 ID 列表, 或 None (不存在路径)

        Raises:
            NodeNotFoundError: 源或目标节点不存在
        """
        with self._lock:
            if source_id not in self._nodes:
                raise NodeNotFoundError(source_id)
            if target_id not in self._nodes:
                raise NodeNotFoundError(target_id)

            if source_id == target_id:
                return [source_id]

            visited: set[str] = {source_id}
            queue: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])

            while queue:
                current, path = queue.popleft()
                for neighbor_id in self._edges.get(current, {}):
                    if neighbor_id == target_id:
                        return path + [neighbor_id]
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, path + [neighbor_id]))

            return None

    def search(
        self,
        node_type: NodeType | None = None,
        min_strength: float | None = None,
        max_strength: float | None = None,
        metadata_key: str | None = None,
        metadata_value: Any = None,
        content_key: str | None = None,
        content_value: Any = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        """多条件搜索节点.

        所有非 None 条件为 AND 关系。

        Args:
            node_type: 节点类型过滤
            min_strength: 最小强度
            max_strength: 最大强度
            metadata_key: 元数据键必须存在
            metadata_value: 元数据键值必须匹配 (需同时提供 metadata_key)
            content_key: 内容键必须存在
            content_value: 内容键值必须匹配 (需同时提供 content_key)
            limit: 最大返回数

        Returns:
            匹配的 MemoryNode 列表
        """
        with self._lock:
            results: list[MemoryNode] = []
            for node in self._nodes.values():
                if node_type is not None and node.node_type != node_type:
                    continue
                if min_strength is not None and node.strength < min_strength:
                    continue
                if max_strength is not None and node.strength > max_strength:
                    continue
                if metadata_key is not None:
                    if metadata_key not in node.metadata:
                        continue
                    if metadata_value is not None and node.metadata[metadata_key] != metadata_value:
                        continue
                if content_key is not None:
                    if content_key not in node.content:
                        continue
                    if content_value is not None and node.content[content_key] != content_value:
                        continue
                results.append(node)
                if len(results) >= limit:
                    break
            return results

    def subgraph(self, node_ids: list[str]) -> dict[str, Any]:
        """提取子图 (指定节点集及其内部边).

        Returns:
            {"nodes": [node_dict, ...], "edges": [edge_dict, ...]}
        """
        with self._lock:
            nodes_data: list[dict[str, Any]] = []
            edges_data: list[dict[str, Any]] = []
            id_set = set(node_ids)
            for nid in node_ids:
                if nid in self._nodes:
                    nodes_data.append(self._nodes[nid].to_dict())
            for src in node_ids:
                if src in self._edges:
                    for tid, edge in self._edges[src].items():
                        if tid in id_set:
                            edges_data.append(edge.to_dict())
            return {"nodes": nodes_data, "edges": edges_data}

    # ============================================================
    # 衰减与强化
    # ============================================================

    def decay(self, factor: float | None = None) -> int:
        """对所有节点执行衰减.

        艾宾浩斯遗忘曲线模型: ``strength *= factor``。
        强度低于 ``min_strength`` 的节点被自动清除。

        Args:
            factor: 自定义衰减因子, None 使用构造时设置的默认值

        Returns:
            被清除的节点数
        """
        with self._lock:
            f = factor if factor is not None else self._decay_factor
            pruned_ids: list[str] = []

            for nid, node in self._nodes.items():
                node.strength *= f
                if node.strength < self._min_strength:
                    pruned_ids.append(nid)

            # 清除低强度节点 (需要逐个调用 remove_node 以清理边)
            for nid in pruned_ids:
                # 直接内部删除, 避免重复获取锁
                if nid in self._edges:
                    del self._edges[nid]
                sources = self._reverse_edges.pop(nid, [])
                for src in sources:
                    if nid in self._edges.get(src, {}):
                        del self._edges[src][nid]
                del self._nodes[nid]

            self._metrics.record_decay(len(pruned_ids))
            return len(pruned_ids)

    def reinforce(self, node_id: str, amount: float = 0.1) -> MemoryNode:
        """强化节点 (增加强度).

        Args:
            node_id: 节点 ID
            amount: 增量 (会被 clamp 到 [0, 1] 总范围)

        Returns:
            更新后的 MemoryNode

        Raises:
            NodeNotFoundError: 节点不存在
        """
        with self._lock:
            if node_id not in self._nodes:
                raise NodeNotFoundError(node_id)
            node = self._nodes[node_id]
            node.strength = min(1.0, node.strength + amount)
            node.touch()
            self._metrics.record_access()
            return node

    def spreading_activation(
        self,
        node_id: str,
        depth: int | None = None,
        decay: float | None = None,
    ) -> dict[str, float]:
        """扩散激活.

        认知科学启发的记忆检索模型:
        - 访问一个知识节点会部分激活其关联节点
        - 激活量随距离衰减

        算法:
        1. 起始节点激活量 = 1.0, 强化 +0.1
        2. 每层扩散: neighbor_activation = parent_activation * edge_weight * decay_rate
        3. 激活量 > 0.01 的节点被部分强化 (strength += activation * 0.1)
        4. 已访问节点不重复激活

        Args:
            node_id: 起始节点
            depth: 扩散深度, None 使用默认值
            decay: 衰减率, None 使用默认值

        Returns:
            {node_id: activation_level} 激活映射 (仅包含激活量 > 0 的节点)

        Raises:
            NodeNotFoundError: 起始节点不存在
        """
        with self._lock:
            if node_id not in self._nodes:
                raise NodeNotFoundError(node_id)

            d = depth if depth is not None else self._spreading_depth
            dec = decay if decay is not None else self._spreading_decay

            # 强化起始节点
            source = self._nodes[node_id]
            source.strength = min(1.0, source.strength + 0.1)
            source.touch()

            activations: dict[str, float] = {node_id: 1.0}
            visited: set[str] = {node_id}
            current_wave: list[tuple[str, float]] = [(node_id, 1.0)]

            for _ in range(d):
                next_wave: list[tuple[str, float]] = []
                for nid, activation in current_wave:
                    neighbors = self._edges.get(nid, {})
                    for tid, edge in neighbors.items():
                        if tid in visited:
                            continue
                        visited.add(tid)
                        spread = activation * edge.weight * dec
                        if spread > 0.01:
                            next_wave.append((tid, spread))
                            # 部分强化邻居
                            neighbor = self._nodes[tid]
                            neighbor.strength = min(1.0, neighbor.strength + spread * 0.1)
                            if tid not in activations:
                                activations[tid] = spread
                            else:
                                activations[tid] = max(activations[tid], spread)

                current_wave = next_wave
                if not current_wave:
                    break

            self._metrics.record_spreading()
            self._metrics.record_access()
            return activations

    # ============================================================
    # 环检测
    # ============================================================

    def _detect_cycle(self, source: str, target: str) -> list[str] | None:
        """DFS 检测从 source 到 target 是否存在路径.

        调用方传入 _detect_cycle(target_id, source_id), 即检查
        target_id → ... → source_id 的路径是否存在。
        若存在, 则添加 source_id → target_id 边后形成环。

        Args:
            source: DFS 起点 (即边的 target_id, 环检测的起点)
            target: DFS 查找目标 (即边的 source_id, 环检测的终点)

        Returns:
            环路径列表 (source → ... → target), 或 None
        """
        visited: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> list[str] | None:
            if node == target:
                return path + [node]
            if node in visited:
                return None
            visited.add(node)
            path.append(node)
            for neighbor_id in self._edges.get(node, {}):
                result = dfs(neighbor_id)
                if result is not None:
                    return result
            path.pop()
            return None

        return dfs(source)

    def has_cycle(self) -> bool:
        """检测整个图谱是否存在环 (仅考虑 PREREQUISITE 边).

        使用 DFS 三色标记法。
        """
        with self._lock:
            WHITE, GRAY, BLACK = 0, 1, 2
            color: dict[str, int] = defaultdict(lambda: WHITE)

            def dfs(node: str) -> bool:
                color[node] = GRAY
                for tid, edge in self._edges.get(node, {}).items():
                    if edge.edge_type != EdgeType.PREREQUISITE:
                        continue
                    if color[tid] == GRAY:
                        return True
                    if color[tid] == WHITE and dfs(tid):
                        return True
                color[node] = BLACK
                return False

            for nid in self._nodes:
                if color[nid] == WHITE:
                    if dfs(nid):
                        return True
            return False

    # ============================================================
    # 导出/导入
    # ============================================================

    def export(self) -> dict[str, Any]:
        """导出整个图谱.

        Returns:
            {"nodes": [...], "edges": [...], "metrics": {...}}
        """
        with self._lock:
            nodes = [n.to_dict() for n in self._nodes.values()]
            edges: list[dict[str, Any]] = []
            for src_edges in self._edges.values():
                for edge in src_edges.values():
                    edges.append(edge.to_dict())
            return {
                "nodes": nodes,
                "edges": edges,
                "metrics": self._metrics.export(),
            }

    def import_data(self, data: dict[str, Any]) -> None:
        """导入图谱数据 (追加, 不清除已有数据).

        Args:
            data: {"nodes": [...], "edges": [...]}
        """
        with self._lock:
            for node_data in data.get("nodes", []):
                node = MemoryNode(
                    node_id=node_data["node_id"],
                    node_type=NodeType(node_data["node_type"]),
                    content=node_data.get("content", {}),
                    metadata=node_data.get("metadata", {}),
                    strength=node_data.get("strength", 1.0),
                )
                node.created_at = node_data.get("created_at", time.time())
                node.last_accessed_at = node_data.get("last_accessed_at", node.created_at)
                node.access_count = node_data.get("access_count", 0)
                self._nodes[node.node_id] = node

            for edge_data in data.get("edges", []):
                edge = MemoryEdge(
                    source_id=edge_data["source_id"],
                    target_id=edge_data["target_id"],
                    edge_type=EdgeType(edge_data["edge_type"]),
                    weight=edge_data.get("weight", 1.0),
                    metadata=edge_data.get("metadata", {}),
                )
                edge.created_at = edge_data.get("created_at", time.time())
                self._edges[edge.source_id][edge.target_id] = edge
                if edge.source_id not in self._reverse_edges[edge.target_id]:
                    self._reverse_edges[edge.target_id].append(edge.source_id)

    # ============================================================
    # 度量与统计
    # ============================================================

    def get_metrics(self) -> dict[str, Any]:
        """获取度量统计."""
        with self._lock:
            metrics = self._metrics.export()
            metrics["node_count"] = len(self._nodes)
            metrics["edge_count"] = self._count_edges()
            # 节点类型分布
            type_dist: dict[str, int] = defaultdict(int)
            for node in self._nodes.values():
                type_dist[node.node_type.value] += 1
            metrics["type_distribution"] = dict(type_dist)
            return metrics

    def reset(self) -> None:
        """重置图谱 (清除所有节点、边和度量)."""
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._reverse_edges.clear()
            self._metrics.reset()

    def node_count(self) -> int:
        """当前节点数."""
        with self._lock:
            return len(self._nodes)

    def edge_count(self) -> int:
        """当前边数."""
        with self._lock:
            return self._count_edges()

    def _count_edges(self) -> int:
        """内部: 统计边数 (不加锁)."""
        return sum(len(edges) for edges in self._edges.values())
