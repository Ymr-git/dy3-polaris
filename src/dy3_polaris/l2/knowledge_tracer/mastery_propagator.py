"""KG 驱动的掌握度传播引擎.

设计依据:
- 知识图谱 (Knowledge Graph) 中的前置关系 (prerequisite) 表征知识点间的
  依赖结构: 掌握前置知识点会提升后继知识点的先验掌握概率.
- 贝叶斯网络结构先验: 用前置掌握度的加权聚合作为后继先验的加性提升.

传播公式 (加性提升):
    P(L0_B) = base_P(L0) + alpha * sum(weight_i * P(L_A_i))

其中:
- ``base_P(L0)``: 后继知识点当前掌握度 (base)
- ``alpha``: 传播系数 (默认 0.3), 控制前置掌握对后继先验的影响强度
- ``weight_i``: 第 i 个前置知识点的权重 (本实现默认等权 1.0)
- ``P(L_A_i)``: 第 i 个前置知识点的掌握度

结果 clamp 到 [0.0, 1.0] 以保证概率有效.

说明:
- MasteryPropagator 为无状态引擎类, 不持有图谱状态.
- 前置关系 (含权重) 由调用方通过 ``prerequisites`` 参数显式提供,
  便于与外部 KG (如 L3 graph_reasoner) 解耦.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any


# ============================================================
# 1. 常量定义
# ============================================================

# 传播系数 alpha: 前置掌握度对后继先验的加性提升强度
DEFAULT_ALPHA: float = 0.3

# 前置知识点默认等权 (每个前置贡献相同权重)
DEFAULT_PREREQ_WEIGHT: float = 1.0

# 多跳传播每跳衰减基数 (防过载): 第 d 层贡献乘以 0.5 ** d
HOP_DECAY_BASE: float = 0.5

# 异构图传播: 不同边类型的传播系数 (alpha)
HETEROGENEOUS_EDGE_ALPHAS: dict[str, float] = {
    "prerequisite": 0.3,  # 正向加强 (前置 -> 后继)
    "similarity": 0.5,    # 相似知识点强传播
    "complement": 0.2,    # 互补知识点弱传播
}

# 趋势检测斜率阈值 (单位: 每步)
TREND_SLOPE_THRESHOLD: float = 0.01


# ============================================================
# 2. MasteryPropagator 无状态引擎类
# ============================================================


class MasteryPropagator:
    """KG 驱动的掌握度传播器.

    根据前置知识点的掌握度, 对后继知识点的掌握度进行加性提升:

        P(L0_B) = base + alpha * sum(weight_i * prereq_mastery_i)

    结果 clamp 到 [0.0, 1.0].

    两种使用模式:
    1. 无状态模式 (原有): 前置关系由调用方通过 ``propagate`` 方法参数显式传入.
    2. 有状态模式 (新增): 通过 ``set_kg_graph`` 设置图谱后, 调用 ``propagate_mastery``
       自动从 store 读取前置掌握度并传播, 修复 pipeline 静默失效问题.

    Attributes:
        alpha: 传播系数 (默认 0.3).
        default_weight: 单个前置知识点的默认权重 (默认 1.0, 等权).
        _kg_graph: 内部知识图谱邻接表 (有状态模式), 默认空字典.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        default_weight: float = DEFAULT_PREREQ_WEIGHT,
    ) -> None:
        """初始化掌握度传播器.

        Args:
            alpha: 传播系数, 默认 0.3.
            default_weight: 单个前置知识点默认权重, 默认 1.0 (等权).
        """
        self.alpha = alpha
        self.default_weight = default_weight
        # 内部 KG 图谱 (有状态模式): {kp_id: [(prereq_id, weight), ...]}
        self._kg_graph: dict[str, list[tuple[str, float]]] = {}

    # --- KG 图谱设置 (有状态模式) ---

    def set_kg_graph(
        self,
        graph: dict[str, list[tuple[str, float]]],
    ) -> None:
        """设置知识图谱邻接表 (有状态传播模式).

        图谱格式: ``{kp_id: [(prereq_id, weight), ...]}``
        表示 kp_id 的前置知识点为 prereq_id, 边权重为 weight.

        设置后, ``propagate_mastery`` 将自动从该图谱读取前置关系,
        并从 store 读取前置知识点的掌握度进行传播.

        Args:
            graph: 知识图谱邻接表.
        """
        self._kg_graph = dict(graph)

    # --- Store 感知传播 (修复 pipeline 静默失效) ---

    def propagate_mastery(
        self,
        learner_id: str,
        kp_id: str,
        mastery: float,
        store: Any,
    ) -> None:
        """从 store 读取前置掌握度, 传播提升目标知识点掌握度, 并写回 store.

        流程:
        1. 从 ``_kg_graph`` 读取 kp_id 的前置知识点列表;
        2. 对每个前置知识点, 从 store 读取其当前掌握度;
        3. 调用 ``propagate`` 计算传播提升后的掌握度;
        4. 将提升后的掌握度写回 store (保留 attempts / correct_count / bkt_params).

        无前置知识点时: 不修改 store 中的状态 (掌握度不变).
        store 为 None 时: 静默无操作 (优雅降级).

        Args:
            learner_id: 学习者 ID.
            kp_id: 目标知识点 ID.
            mastery: 目标知识点当前掌握度 (BKT 更新后的值).
            store: L2 存储层 (需实现 get_tracing_state / save_tracing_state).
        """
        if store is None:
            return

        prerequisites_raw = self._kg_graph.get(kp_id, [])
        if not prerequisites_raw:
            return  # 无前置, 不修改

        # 构建带掌握度的前置列表: [(prereq_id, prereq_mastery, weight), ...]
        prereqs: list[tuple[str, float, float]] = []
        for item in prerequisites_raw:
            if len(item) == 2:
                prereq_id, weight = item  # type: ignore[misc]
            else:
                prereq_id = item[0]
                weight = item[2] if len(item) >= 3 else self.default_weight

            prereq_state = store.get_tracing_state(learner_id, prereq_id)
            prereq_mastery = float(prereq_state.mastery_prob) if prereq_state else 0.0
            prereqs.append((prereq_id, prereq_mastery, weight))

        # 计算传播提升后的掌握度
        boosted = self.propagate(kp_id, mastery, prereqs)

        # 写回 store (保留原有计数和参数)
        state = store.get_tracing_state(learner_id, kp_id)
        if state is not None:
            from dy3_polaris.l2.models import TracingState
            new_state = TracingState(
                kp_id=state.kp_id,
                mastery_prob=boosted,
                attempts=state.attempts,
                correct_count=state.correct_count,
                last_attempt_time=state.last_attempt_time,
                bkt_params=dict(state.bkt_params),
            )
            store.save_tracing_state(learner_id, kp_id, new_state)
        else:
            from dy3_polaris.l2.models import TracingState
            store.save_tracing_state(learner_id, kp_id, TracingState(
                kp_id=kp_id, mastery_prob=boosted,
            ))

    # --- 元组归一化 (2-tuple / 3-tuple 兼容) ---

    @staticmethod
    def _unpack(
        item: tuple[str, float] | tuple[str, float, float],
        default_weight: float,
    ) -> tuple[float, float]:
        """归一化前置/后继元组为 (mastery, weight).

        - 3-tuple ``(kp_id, mastery, weight)``: 使用显式权重;
        - 2-tuple ``(kp_id, mastery)``: 使用 ``default_weight`` (向后兼容).

        Args:
            item: 前置/后继元组.
            default_weight: 2-tuple 时使用的默认权重.

        Returns:
            ``(mastery, weight)`` 二元组.
        """
        if len(item) == 3:
            _kp_id, mastery, weight = item  # type: ignore[misc]
            return float(mastery), float(weight)
        _kp_id, mastery = item  # type: ignore[misc]
        return float(mastery), default_weight

    # --- 传播 (单跳, 加权) ---

    def propagate(
        self,
        kp_id: str,
        current_mastery: float,
        prerequisites: list[tuple[str, float] | tuple[str, float, float]],
    ) -> float:
        """根据前置知识点掌握度传播提升后继知识点的掌握度.

        公式:
            boosted = current_mastery + alpha * sum(weight * prereq_mastery)
            result  = clamp(boosted, 0.0, 1.0)

        每个前置知识点可显式指定权重:
        - 3-tuple ``(prereq_kp_id, prereq_mastery, weight)``: 使用该权重;
        - 2-tuple ``(prereq_kp_id, prereq_mastery)``: 使用 ``default_weight``
          (默认 1.0, 等权, 向后兼容).

        Args:
            kp_id: 后继知识点 ID (仅作标识, 不参与计算).
            current_mastery: 后继知识点当前掌握度 (base).
            prerequisites: 前置知识点列表, 每项为
                ``(prereq_kp_id, prereq_mastery)`` 或
                ``(prereq_kp_id, prereq_mastery, weight)``.

        Returns:
            传播提升后并 clamp 到 [0.0, 1.0] 的掌握度.
        """
        boost = 0.0
        for item in prerequisites:
            prereq_mastery, weight = self._unpack(item, self.default_weight)
            boost += weight * prereq_mastery
        boosted = current_mastery + self.alpha * boost
        # clamp 到 [0, 1] 保证概率有效
        return max(0.0, min(1.0, boosted))

    # --- 多跳传播 (KG BFS) ---

    def propagate_multi_hop(
        self,
        kp_id: str,
        current_mastery: float,
        kg_graph: dict[str, list[tuple[str, float]]],
        mastery_map: dict[str, float],
        max_depth: int = 3,
    ) -> float:
        """KG 驱动的多跳掌握度传播 (BFS / 拓扑遍历).

        从 ``kp_id`` 出发, 沿前置关系 (``kg_graph``) 做广度优先遍历,
        至多传播 ``max_depth`` 跳. 每个前置节点的贡献为:

            contribution = alpha * (路径边权乘积) * mastery_map[node]
                           * (HOP_DECAY_BASE ** depth)

        其中 ``HOP_DECAY_BASE = 0.5``, 第 d 层贡献乘以 ``0.5 ** d``
        以防止深层链路过载. 路径边权为从 ``kp_id`` 到该节点沿途各边权重
        的乘积 (每条前置边都衰减影响). ``mastery_map`` 中缺失的节点掌握度
        视为 0 (不贡献).

        使用 ``visited`` 集合对节点去重, 天然处理环 (不重复贡献, 不无限递归).

        Args:
            kp_id: 目标 (后继) 知识点 ID.
            current_mastery: 目标知识点当前掌握度 (base).
            kg_graph: 知识图谱邻接表, ``{kp_id: [(prereq_id, weight), ...]}``.
            mastery_map: 各知识点已知掌握度 ``{kp_id: mastery}``.
            max_depth: 最大传播跳数, 默认 3.

        Returns:
            传播提升后并 clamp 到 [0.0, 1.0] 的掌握度.
        """
        boost = 0.0
        visited: set[str] = {kp_id}
        # BFS 队列: (node, accumulated_path_weight, depth)
        queue: deque[tuple[str, float, int]] = deque()
        for prereq_id, weight in kg_graph.get(kp_id, []):
            queue.append((prereq_id, float(weight), 1))

        while queue:
            node, acc_weight, depth = queue.popleft()
            # 环检测 / 去重: 已访问节点跳过
            if node in visited:
                continue
            visited.add(node)
            if depth > max_depth:
                continue

            mastery = float(mastery_map.get(node, 0.0))
            decay = HOP_DECAY_BASE ** depth
            boost += self.alpha * acc_weight * mastery * decay

            # 继续向更深层前置传播
            if depth < max_depth:
                for next_id, next_weight in kg_graph.get(node, []):
                    if next_id not in visited:
                        queue.append(
                            (next_id, acc_weight * float(next_weight), depth + 1)
                        )

        boosted = current_mastery + boost
        return max(0.0, min(1.0, boosted))

    # --- 反向传播 (后继 -> 当前) ---

    def propagate_reverse(
        self,
        kp_id: str,
        current_mastery: float,
        successors: list[tuple[str, float] | tuple[str, float, float]],
    ) -> float:
        """反向掌握度传播: 由后继知识点掌握度提升当前知识点掌握度.

        语义: 若学习者已掌握较难的后继知识点, 则其更易掌握的前置知识点
        (当前节点) 的掌握度也应被上调 (证据反向推断).

        公式 (与正向传播同构, 但作用对象为后继集合):

            boosted = current_mastery + alpha * sum(weight * successor_mastery)
            result  = clamp(boosted, 0.0, 1.0)

        Args:
            kp_id: 当前 (前置) 知识点 ID (仅作标识, 不参与计算).
            current_mastery: 当前知识点掌握度 (base).
            successors: 后继知识点列表, 每项为
                ``(successor_id, mastery)`` 或
                ``(successor_id, mastery, weight)``.

        Returns:
            反向传播提升后并 clamp 到 [0.0, 1.0] 的掌握度.
        """
        boost = 0.0
        for item in successors:
            succ_mastery, weight = self._unpack(item, self.default_weight)
            boost += weight * succ_mastery
        boosted = current_mastery + self.alpha * boost
        return max(0.0, min(1.0, boosted))

    # --- GNN 式传播 (注意力加权多层) ---

    def propagate_gnn(
        self,
        kp_id: str,
        current_mastery: float,
        kg_graph: dict[str, list[tuple[str, float]]],
        mastery_map: dict[str, float],
        max_depth: int = 3,
        attention_weights: dict[tuple[str, str], float] | None = None,
    ) -> float:
        """GNN 式注意力加权多层掌握度传播.

        对标 GKT (NeurIPS 2020) / AKT (KDD 2020) 的图卷积传播思想,
        每一层使用注意力权重聚合邻居信息, 层间经 ReLU 激活:

            h_v^{(l+1)} = ReLU( Σ_{u∈N(v)} α_{vu} * W_{vu} * h_u^{(l)} )

        其中:
        - ``α_{vu}``: 节点 v 对邻居 u 的注意力权重 (softmax 归一化, 和为 1);
        - ``W_{vu}``: 边权重 (取自 ``kg_graph`` 中 (u, weight) 的 weight);
        - ``h_u^{(l)}``: 邻居 u 在第 l 层的隐表示; 第 0 层为输入掌握度
          (目标节点用 ``current_mastery``, 其余节点用 ``mastery_map``, 缺失视为 0).

        注意力得分来源:
        - 若提供 ``attention_weights``: 以 ``(v, u) -> score`` 作为原始得分,
          经 softmax 归一化得到 α;
        - 否则使用基于掌握度差异的自注意力:
          ``score_u = mastery_u / sum(mastery_neighbors)``, 再 softmax.
          (邻居掌握度全为 0 时退化为等权注意力.)

        环处理: 递归聚合时维护当前路径集合, 路径中已出现的节点不再深入展开
        (改用其输入掌握度), 避免无限递归.

        最终传播结果:

            boosted = current_mastery + alpha * gnn_aggregation
            result  = clamp(boosted, 0.0, 1.0)

        Args:
            kp_id: 目标知识点 ID.
            current_mastery: 目标知识点当前掌握度 (base).
            kg_graph: 知识图谱邻接表 ``{kp_id: [(neighbor_id, weight), ...]}``.
            mastery_map: 各知识点已知掌握度 ``{kp_id: mastery}``.
            max_depth: GNN 传播层数 (感受野半径), 默认 3.
            attention_weights: 可选, ``{(v, u): score}`` 注意力原始得分.

        Returns:
            GNN 聚合传播后并 clamp 到 [0.0, 1.0] 的掌握度.
        """
        cm = float(current_mastery)

        def input_mastery(node: str) -> float:
            """节点第 0 层输入掌握度 (目标节点用 current_mastery)."""
            if node == kp_id:
                return cm
            return float(mastery_map.get(node, 0.0))

        def attention(
            v: str, neighbors: list[tuple[str, float]]
        ) -> list[tuple[str, float, float]]:
            """计算节点 v 对各邻居的 (neighbor_id, edge_weight, alpha)."""
            if not neighbors:
                return []
            scores: list[float] = []
            for nid, _w in neighbors:
                if attention_weights is not None:
                    scores.append(float(attention_weights.get((v, nid), input_mastery(nid))))
                else:
                    total = sum(input_mastery(n) for n, _ in neighbors)
                    if total <= 0.0:
                        scores.append(1.0)  # 退化为等权
                    else:
                        scores.append(input_mastery(nid) / total)
            # 数值稳定的 softmax
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            z = sum(exps)
            if z <= 0.0:
                n = len(neighbors)
                return [(nid, w, 1.0 / n) for nid, w in neighbors]
            return [
                (nid, float(w), e / z)
                for (nid, w), e in zip(neighbors, exps)
            ]

        def aggregate(node: str, depth: int, path: frozenset[str]) -> float:
            """递归聚合节点 node 在第 depth 层的隐表示 (带 ReLU 激活).

            - 第 0 层: 节点输入掌握度;
            - 第 l+1 层 (有邻居): ReLU(Σ α * W * h_neighbor^{(l)});
            - 叶子节点 (无邻居) 在任意层保留其输入掌握度 (残差式, 不被置零).
            """
            if depth <= 0:
                return input_mastery(node)
            neighbors = kg_graph.get(node, [])
            if not neighbors:
                # 叶子节点: 保留输入掌握度 (其掌握度即为传给父节点的信号)
                return input_mastery(node)
            att = attention(node, neighbors)
            agg = 0.0
            for nid, w, alpha in att:
                if nid in path:
                    # 环: 不再深入, 使用输入掌握度
                    h = input_mastery(nid)
                else:
                    h = aggregate(nid, depth - 1, path | {nid})
                agg += alpha * w * h
            # 每层输出经 ReLU 激活后传到下一层
            return max(0.0, agg)

        # GNN 聚合只反映邻居信号 (目标节点自身掌握度作为 base 单独相加, 不计入聚合).
        # 目标节点无邻居时, 聚合为 0 (无邻居信号可聚合).
        if not kg_graph.get(kp_id):
            gnn_aggregation = 0.0
        else:
            gnn_aggregation = aggregate(kp_id, max_depth, frozenset({kp_id}))
        boosted = cm + self.alpha * gnn_aggregation
        return max(0.0, min(1.0, boosted))

    # --- 注意力加权单跳传播 ---

    def propagate_attention(
        self,
        kp_id: str,
        current_mastery: float,
        prerequisites: list[str],
        mastery_map: dict[str, float],
        attention_fn: str = "dot_product",
    ) -> float:
        """注意力加权单跳掌握度传播.

        对标 AKT (KDD 2020) / SAKT (EDMM 2019) 的注意力机制, 对单个前置知识点
        集合用注意力权重聚合, 再加性提升当前掌握度:

            α_i      = softmax(score_i)            (对所有前置归一化, 和为 1)
            boosted  = current_mastery + alpha * Σ α_i * mastery_i
            result   = clamp(boosted, 0.0, 1.0)

        注意力得分函数 (``attention_fn``):
        - ``"dot_product"`` (默认): ``score_i = mastery_i * current_mastery``
          (掌握度与当前状态越接近 / 越高, 注意力越大);
        - ``"additive"``: 加性注意力标量简化版
          ``score_i = tanh(W1 * mastery_i + W2 * current_mastery)``,
          其中 W1 = W2 = 1.0, v = 1.0 (标量退化, 无需学习参数).

        ``prerequisites`` 为前置知识点 ID 列表, 其掌握度从 ``mastery_map`` 取
        (缺失视为 0, 不贡献). 单前置时 softmax 归一化得 α = 1.0.

        Args:
            kp_id: 目标知识点 ID (仅作标识).
            current_mastery: 目标知识点当前掌握度 (base).
            prerequisites: 前置知识点 ID 列表.
            mastery_map: 前置知识点掌握度 ``{kp_id: mastery}``.
            attention_fn: 注意力函数, "dot_product" 或 "additive", 默认前者.

        Returns:
            注意力加权传播后并 clamp 到 [0.0, 1.0] 的掌握度.
        """
        cm = float(current_mastery)
        if not prerequisites:
            return max(0.0, min(1.0, cm))

        masteries = [float(mastery_map.get(pid, 0.0)) for pid in prerequisites]
        if attention_fn == "additive":
            scores = [math.tanh(m + cm) for m in masteries]
        else:
            # 默认 dot_product
            scores = [m * cm for m in masteries]

        # softmax 归一化
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps)
        if z <= 0.0:
            alphas = [1.0 / len(prerequisites)] * len(prerequisites)
        else:
            alphas = [e / z for e in exps]

        weighted = sum(a * mval for a, mval in zip(alphas, masteries))
        boosted = cm + self.alpha * weighted
        return max(0.0, min(1.0, boosted))

    # --- 异构图传播 (不同边类型不同系数) ---

    def propagate_heterogeneous(
        self,
        kp_id: str,
        current_mastery: float,
        kg_graph: dict[str, list[tuple[str, float]]],
        mastery_map: dict[str, float],
        edge_types: dict[str, str],
        max_depth: int = 3,
    ) -> float:
        """异构图掌握度传播: 不同边类型使用不同传播系数.

        对标 GIKT (AAAI 2020) 的图交互式思想, 为异构知识图谱中不同语义的边
        分配不同传播系数 (alpha):

        - ``"prerequisite"`` (默认): alpha = 0.3 (正向加强);
        - ``"similarity"``:    alpha = 0.5 (相似知识点强传播);
        - ``"complement"``:    alpha = 0.2 (互补知识点弱传播).

        传播方式与 ``propagate_multi_hop`` 同构 (BFS 多跳, 每跳 0.5^depth 衰减,
        ``visited`` 去重处理环), 但每条边的传播系数取自其边类型:

            contribution(node u) = alpha_edge * acc_path_weight
                                   * mastery_map[u] * (0.5 ** depth)

        ``edge_types`` 以 ``"v->u"`` 字符串为键映射到边类型; 未指定的边默认
        ``"prerequisite"``.

        Args:
            kp_id: 目标知识点 ID.
            current_mastery: 目标知识点当前掌握度 (base).
            kg_graph: 知识图谱邻接表 ``{kp_id: [(neighbor_id, weight), ...]}``.
            mastery_map: 各知识点已知掌握度 ``{kp_id: mastery}``.
            edge_types: 边类型映射 ``{"v->u": "prerequisite"|"similarity"|"complement"}``.
            max_depth: 最大传播跳数, 默认 3.

        Returns:
            异构传播后并 clamp 到 [0.0, 1.0] 的掌握度.
        """
        boost = 0.0
        visited: set[str] = {kp_id}
        queue: deque[tuple[str, float, float, int]] = deque()
        for nid, w in kg_graph.get(kp_id, []):
            etype = edge_types.get(f"{kp_id}->{nid}", "prerequisite")
            a = HETEROGENEOUS_EDGE_ALPHAS.get(etype, self.alpha)
            queue.append((nid, float(w), a, 1))

        while queue:
            node, acc_weight, edge_alpha, depth = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if depth > max_depth:
                continue

            mastery = float(mastery_map.get(node, 0.0))
            decay = HOP_DECAY_BASE ** depth
            boost += edge_alpha * acc_weight * mastery * decay

            if depth < max_depth:
                for next_id, next_weight in kg_graph.get(node, []):
                    if next_id not in visited:
                        etype = edge_types.get(f"{node}->{next_id}", "prerequisite")
                        na = HETEROGENEOUS_EDGE_ALPHAS.get(etype, self.alpha)
                        queue.append(
                            (next_id, acc_weight * float(next_weight), na, depth + 1)
                        )

        boosted = current_mastery + boost
        return max(0.0, min(1.0, boosted))


# ============================================================
# __all__
# ============================================================

__all__ = [
    "MasteryPropagator",
    "DEFAULT_ALPHA",
    "HOP_DECAY_BASE",
    "HETEROGENEOUS_EDGE_ALPHAS",
    "TREND_SLOPE_THRESHOLD",
]
