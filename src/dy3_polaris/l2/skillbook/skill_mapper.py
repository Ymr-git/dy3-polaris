"""技能树映射器 — BKT + IRT → 技能树可视化.

融合世界先进方案:
- Squirrel AI: 纳米级知识分解 + 知识图谱
- ALEKS: 知识空间理论 + 技能树
- Duolingo: 技能依赖图 + 学习路径

技能状态映射:
- mastery >= 0.7 -> "mastered" (已掌握)
- 0.4 <= mastery < 0.7 -> "learning" (学习中)
- 0 < mastery < 0.4 -> "weak" (薄弱)
- mastery == 0 -> "not_started" (未开始)

技能级别映射:
- mastery < 0.4 -> "L0" (入门)
- 0.4 <= mastery < 0.7 -> "L1" (进阶)
- mastery >= 0.7 -> "L2" (精通)

全局能力:
- theta ∈ [-3, 3] 映射到 [0, 1]: (theta + 3) / 6
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dy3_polaris.l2.models import IRTState, TracingState


# ============================================================
# 1. 常量定义
# ============================================================

# 技能状态 / 级别阈值
_MASTERY_MASTERED: float = 0.7   # >= 0.7 已掌握 / L2 精通
_MASTERY_LEARNING: float = 0.4   # >= 0.4 学习中 / L1 进阶

# IRT theta 映射端点 (theta ∈ [-3, 3] -> [0, 1])
_THETA_MIN: float = -3.0
_THETA_SPAN: float = 6.0


# ============================================================
# 2. 内部辅助函数 (状态 / 级别计算)
# ============================================================


def _status_from_mastery(mastery: float) -> str:
    """根据掌握度计算技能状态字符串.

    映射规则:
    - mastery == 0        -> "not_started" (未开始)
    - 0 < mastery < 0.4   -> "weak"        (薄弱)
    - 0.4 <= mastery < 0.7 -> "learning"   (学习中)
    - mastery >= 0.7      -> "mastered"    (已掌握)

    Args:
        mastery: 掌握度 [0.0, 1.0].

    Returns:
        状态字符串 (not_started / weak / learning / mastered).
    """
    if mastery <= 0.0:
        return "not_started"
    if mastery < _MASTERY_LEARNING:
        return "weak"
    if mastery < _MASTERY_MASTERED:
        return "learning"
    return "mastered"


def _level_from_mastery(mastery: float) -> str:
    """根据掌握度计算技能级别字符串.

    映射规则:
    - mastery < 0.4        -> "L0" (入门)
    - 0.4 <= mastery < 0.7 -> "L1" (进阶)
    - mastery >= 0.7       -> "L2" (精通)

    Args:
        mastery: 掌握度 [0.0, 1.0].

    Returns:
        级别字符串 (L0 / L1 / L2).
    """
    if mastery < _MASTERY_LEARNING:
        return "L0"
    if mastery < _MASTERY_MASTERED:
        return "L1"
    return "L2"


def _theta_to_global_ability(theta: float) -> float:
    """将 IRT 能力参数 theta 映射到 [0, 1] 全局能力值.

    公式: global_ability = (theta + 3) / 6
    - theta = -3 -> 0.0 (能力下限)
    - theta =  0 -> 0.5 (中等能力)
    - theta =  3 -> 1.0 (能力上限)

    Args:
        theta: IRT 能力参数 (标准分尺度, 通常 ∈ [-3, 3]).

    Returns:
        全局能力值 [0.0, 1.0].
    """
    return (theta - _THETA_MIN) / _THETA_SPAN


# ============================================================
# 3. SkillNode 技能节点数据类
# ============================================================


@dataclass
class SkillNode:
    """技能树节点 — 单个知识点的技能状态可视化.

    由 ``TracingState`` 映射而来, 字段:
    - ``kp_id``: 知识点 ID (来自 TracingState.kp_id)
    - ``name``: 知识点显示名称 (优先取 KG 名称, 缺省为 kp_id)
    - ``mastery``: 掌握度 [0.0, 1.0] (来自 TracingState.mastery_prob)
    - ``status``: 技能状态 (not_started / weak / learning / mastered),
      由 ``mastery`` 自动推导, 也可由 ``from_dict`` 显式传入覆盖
    - ``level``: 技能级别 (L0 / L1 / L2), 由 ``mastery`` 自动推导,
      也可由 ``from_dict`` 显式传入覆盖

    ``status`` / ``level`` 默认 ``None``, 在 ``__post_init__`` 中按
    ``mastery`` 自动计算; 若调用方显式提供则保留原值 (用于反序列化场景).

    Attributes:
        kp_id: 知识点 ID.
        name: 知识点显示名称, 缺省为 kp_id.
        mastery: 掌握度 [0.0, 1.0], 默认 0.0.
        status: 技能状态字符串, 缺省由 mastery 自动推导.
        level: 技能级别字符串, 缺省由 mastery 自动推导.
    """

    kp_id: str
    name: str | None = None
    mastery: float = 0.0
    status: str | None = None
    level: str | None = None

    def __post_init__(self) -> None:
        """初始化后处理: 补全 name / status / level 的缺省值.

        - ``name`` 缺省 -> 取 ``kp_id``
        - ``status`` 缺省 -> 由 ``mastery`` 推导
        - ``level`` 缺省 -> 由 ``mastery`` 推导
        """
        if self.name is None:
            self.name = self.kp_id
        if self.status is None:
            self.status = _status_from_mastery(self.mastery)
        if self.level is None:
            self.level = _level_from_mastery(self.mastery)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典.

        Returns:
            含 kp_id / name / mastery / status / level 五字段的字典.
        """
        return {
            "kp_id": self.kp_id,
            "name": self.name,
            "mastery": self.mastery,
            "status": self.status,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillNode:
        """从字典反序列化.

        若字典中含 ``status`` / ``level`` 键, 则显式传入 (覆盖自动推导),
        否则由 ``mastery`` 自动推导. ``name`` 缺省时回退到 ``kp_id``.

        Args:
            d: 字典, 至少含 ``kp_id`` 键.

        Returns:
            还原后的 SkillNode 实例.
        """
        return cls(
            kp_id=d["kp_id"],
            name=d.get("name"),
            mastery=d.get("mastery", 0.0),
            status=d.get("status"),
            level=d.get("level"),
        )


# ============================================================
# 4. SkillEdge 技能边数据类
# ============================================================


@dataclass
class SkillEdge:
    """技能树边 — 知识点间依赖关系.

    字段:
    - ``from_kp``: 起始知识点 ID (前置)
    - ``to_kp``: 目标知识点 ID (后继)
    - ``edge_type``: 边类型 ("prerequisite" 前置依赖 / "related" 关联)

    Attributes:
        from_kp: 起始知识点 ID.
        to_kp: 目标知识点 ID.
        edge_type: 边类型, 默认 "prerequisite".
    """

    from_kp: str
    to_kp: str
    edge_type: str = "prerequisite"

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典.

        Returns:
            含 from_kp / to_kp / edge_type 三字段的字典.
        """
        return {
            "from_kp": self.from_kp,
            "to_kp": self.to_kp,
            "edge_type": self.edge_type,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillEdge:
        """从字典反序列化.

        Args:
            d: 字典, 至少含 ``from_kp`` / ``to_kp`` 键.

        Returns:
            还原后的 SkillEdge 实例.
        """
        return cls(
            from_kp=d["from_kp"],
            to_kp=d["to_kp"],
            edge_type=d.get("edge_type", "prerequisite"),
        )


# ============================================================
# 5. SkillMapper 技能树映射器 (无状态引擎)
# ============================================================


class SkillMapper:
    """技能树映射器 (无状态引擎).

    将 BKT 知识追踪结果 (``TracingState``) 与 IRT 能力评估结果
    (``IRTState``) 映射为可视化的技能树:

        {
            "global_ability": float,      # theta -> [0, 1]
            "nodes": list[SkillNode],     # 每个 TracingState -> SkillNode
            "edges": list[SkillEdge],     # 从 kg_structure 提取 (可选)
        }

    设计参考:
    - Squirrel AI: 纳米级知识分解, 知识点掌握度驱动的技能状态
    - ALEKS: 知识空间理论, 前置依赖 (prerequisite) 关系
    - Duolingo: 技能依赖图, learning/mastered 状态可视化

    该类不持有任何学习者状态 (无状态引擎), 所有数据通过方法入参显式传递,
    相同输入产生相同输出, 可安全多实例并发复用.

    融合 L2 已实现模块:
    - ``knowledge_tracer.BKTTracer`` 产出 ``TracingState`` (mastery_prob)
    - ``ability_assessor.IRTEstimator`` 产出 ``IRTState`` (theta)
    - 本映射器将二者融合为统一的技能树视图
    """

    # --- 静态映射方法 ---

    @staticmethod
    def get_skill_status(mastery: float) -> str:
        """根据掌握度返回技能状态字符串.

        映射规则:
        - mastery == 0        -> "not_started"
        - 0 < mastery < 0.4   -> "weak"
        - 0.4 <= mastery < 0.7 -> "learning"
        - mastery >= 0.7      -> "mastered"

        Args:
            mastery: 掌握度 [0.0, 1.0].

        Returns:
            状态字符串 (not_started / weak / learning / mastered).
        """
        return _status_from_mastery(mastery)

    @staticmethod
    def get_skill_level(mastery: float) -> str:
        """根据掌握度返回技能级别字符串.

        映射规则:
        - mastery < 0.4        -> "L0" (入门)
        - 0.4 <= mastery < 0.7 -> "L1" (进阶)
        - mastery >= 0.7       -> "L2" (精通)

        Args:
            mastery: 掌握度 [0.0, 1.0].

        Returns:
            级别字符串 (L0 / L1 / L2).
        """
        return _level_from_mastery(mastery)

    # --- 内部辅助: 从 kg_structure 构建名称查找表 ---

    @staticmethod
    def _build_name_lookup(
        kg_structure: dict[str, Any] | None,
    ) -> dict[str, str]:
        """从 kg_structure 的 nodes 列表构建 {kp_id: name} 查找表.

        kg_structure.nodes 格式: [{"kp_id": "kp1", "name": "..."}, ...]
        缺失 name 字段的节点不入表 (由调用方回退到 kp_id).

        Args:
            kg_structure: 知识图谱结构字典 (可为 None).

        Returns:
            ``{kp_id: name}`` 映射; 无 kg_structure 或无 nodes 键时返回空字典.
        """
        if not kg_structure:
            return {}
        kg_nodes = kg_structure.get("nodes")
        if not kg_nodes:
            return {}
        lookup: dict[str, str] = {}
        for node in kg_nodes:
            kp_id = node.get("kp_id")
            name = node.get("name")
            if kp_id is not None and name is not None:
                lookup[kp_id] = name
        return lookup

    # --- 内部辅助: 从 kg_structure 构建边列表 ---

    @staticmethod
    def _build_edges(
        kg_structure: dict[str, Any] | None,
    ) -> list[SkillEdge]:
        """从 kg_structure 的 edges 列表构建 SkillEdge 列表.

        kg_structure.edges 格式:
            [{"from": "kp1", "to": "kp2", "type": "prerequisite"}, ...]

        Args:
            kg_structure: 知识图谱结构字典 (可为 None).

        Returns:
            SkillEdge 列表; 无 kg_structure 或无 edges 键时返回空列表.
        """
        if not kg_structure:
            return []
        kg_edges = kg_structure.get("edges")
        if not kg_edges:
            return []
        edges: list[SkillEdge] = []
        for e in kg_edges:
            from_kp = e.get("from")
            to_kp = e.get("to")
            if from_kp is None or to_kp is None:
                continue
            edge_type = e.get("type", "prerequisite")
            edges.append(
                SkillEdge(
                    from_kp=from_kp,
                    to_kp=to_kp,
                    edge_type=edge_type,
                )
            )
        return edges

    # --- 核心: 构建技能树 ---

    def to_skill_tree(
        self,
        tracing_states: dict[str, TracingState],
        irt_state: IRTState,
        kg_structure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """将 BKT 追踪状态 + IRT 能力状态映射为技能树.

        映射逻辑:
        1. ``global_ability`` = (irt_state.theta + 3) / 6, 映射到 [0, 1]
        2. ``nodes``: 遍历 ``tracing_states``, 每个 ``TracingState`` 生成
           一个 ``SkillNode``:
           - ``kp_id`` <- TracingState.kp_id
           - ``mastery`` <- TracingState.mastery_prob
           - ``name`` <- kg_structure 名称查找表, 缺省为 kp_id
           - ``status`` / ``level`` <- 由 mastery 自动推导
        3. ``edges``: 从 ``kg_structure.edges`` 提取 (无 kg_structure 时为空)

        不修改入参 ``tracing_states`` 与 ``kg_structure`` (函数式风格).

        Args:
            tracing_states: 知识点追踪状态映射 ``{kp_id: TracingState}``.
            irt_state: 学习者 IRT 能力状态 (取 theta 字段).
            kg_structure: 知识图谱结构 (可选), 格式::

                {
                    "nodes": [{"kp_id": "kp1", "name": "..."}, ...],
                    "edges": [{"from": "kp1", "to": "kp2",
                               "type": "prerequisite"}, ...],
                }

        Returns:
            技能树字典::

                {
                    "global_ability": float,
                    "nodes": list[SkillNode],
                    "edges": list[SkillEdge],
                }
        """
        # 1. 全局能力: theta -> [0, 1]
        global_ability = _theta_to_global_ability(irt_state.theta)

        # 2. 名称查找表 (kg_structure 提供知识点显示名称)
        name_lookup = self._build_name_lookup(kg_structure)

        # 3. 构建节点列表 (不修改入参 tracing_states)
        nodes: list[SkillNode] = []
        for state in tracing_states.values():
            kp_id = state.kp_id
            name = name_lookup.get(kp_id, kp_id)
            nodes.append(
                SkillNode(
                    kp_id=kp_id,
                    name=name,
                    mastery=state.mastery_prob,
                )
            )

        # 4. 构建边列表 (从 kg_structure 提取)
        edges = self._build_edges(kg_structure)

        return {
            "global_ability": global_ability,
            "nodes": nodes,
            "edges": edges,
        }

    # --- 摘要统计 ---

    def get_summary(self, skill_tree: dict[str, Any]) -> dict[str, Any]:
        """统计技能树摘要 — 各状态节点数与全局能力.

        统计字段:
        - ``total_kps``: 知识点总数 (nodes 长度)
        - ``mastered``: 已掌握节点数 (status == "mastered")
        - ``learning``: 学习中节点数 (status == "learning")
        - ``weak``: 薄弱节点数 (status == "weak")
        - ``not_started``: 未开始节点数 (status == "not_started")
        - ``global_ability``: 全局能力值 (取自 skill_tree["global_ability"])

        Args:
            skill_tree: ``to_skill_tree`` 产出的技能树字典.

        Returns:
            摘要字典, 含 total_kps / mastered / learning / weak /
            not_started / global_ability 六字段.
        """
        nodes: list[SkillNode] = skill_tree.get("nodes", [])
        counts: dict[str, int] = {
            "mastered": 0,
            "learning": 0,
            "weak": 0,
            "not_started": 0,
        }
        for node in nodes:
            status = node.status
            if status in counts:
                counts[status] += 1

        return {
            "total_kps": len(nodes),
            "mastered": counts["mastered"],
            "learning": counts["learning"],
            "weak": counts["weak"],
            "not_started": counts["not_started"],
            "global_ability": skill_tree.get("global_ability", 0.0),
        }

    # --- 学习路径推荐 (ALEKS 知识空间理论 + Duolingo 依赖图) ---

    @staticmethod
    def _build_prereq_map(
        skill_tree: dict[str, Any],
    ) -> dict[str, list[str]]:
        """从技能树边构建前置依赖映射 {to_kp: [from_kp, ...]}.

        仅纳入 ``edge_type == "prerequisite"`` 的边 (忽略 "related" 等关联边).

        Args:
            skill_tree: 技能树字典.

        Returns:
            ``{to_kp: [from_kp, ...]}`` 映射; 无前置依赖的节点不在键中.
        """
        prereqs: dict[str, list[str]] = {}
        for edge in skill_tree.get("edges", []):
            if getattr(edge, "edge_type", None) != "prerequisite":
                continue
            from_kp = getattr(edge, "from_kp", None)
            to_kp = getattr(edge, "to_kp", None)
            if from_kp is None or to_kp is None:
                continue
            prereqs.setdefault(to_kp, []).append(from_kp)
        return prereqs

    @staticmethod
    def _build_status_lookup(
        skill_tree: dict[str, Any],
    ) -> dict[str, str]:
        """构建 {kp_id: status} 查找表.

        Args:
            skill_tree: 技能树字典.

        Returns:
            ``{kp_id: status}`` 映射.
        """
        return {
            node.kp_id: node.status
            for node in skill_tree.get("nodes", [])
        }

    def get_unlocked_skills(self, skill_tree: dict[str, Any]) -> list[SkillNode]:
        """获取已解锁的技能节点 — 所有前置依赖均为 "mastered".

        节点解锁条件: 所有指向该节点的 prerequisite 边的起始节点状态均为
        "mastered"。无前置依赖的节点视为已解锁 (空前置即满足).

        Args:
            skill_tree: ``to_skill_tree`` 产出的技能树字典.

        Returns:
            已解锁的 SkillNode 列表 (保持节点原顺序).
        """
        nodes: list[SkillNode] = skill_tree.get("nodes", [])
        status_by_kp = self._build_status_lookup(skill_tree)
        prereqs = self._build_prereq_map(skill_tree)

        unlocked: list[SkillNode] = []
        for node in nodes:
            preds = prereqs.get(node.kp_id, [])
            if all(status_by_kp.get(p) == "mastered" for p in preds):
                unlocked.append(node)
        return unlocked

    def get_locked_skills(self, skill_tree: dict[str, Any]) -> list[SkillNode]:
        """获取被锁定的技能节点 — 至少有一个 "not_started"/"weak" 前置.

        节点锁定条件: 存在至少一条指向该节点的 prerequisite 边, 其起始节点
        状态为 "not_started" 或 "weak"。

        Args:
            skill_tree: ``to_skill_tree`` 产出的技能树字典.

        Returns:
            被锁定的 SkillNode 列表 (保持节点原顺序).
        """
        nodes: list[SkillNode] = skill_tree.get("nodes", [])
        status_by_kp = self._build_status_lookup(skill_tree)
        prereqs = self._build_prereq_map(skill_tree)

        locked: list[SkillNode] = []
        for node in nodes:
            preds = prereqs.get(node.kp_id, [])
            if any(
                status_by_kp.get(p) in ("not_started", "weak")
                for p in preds
            ):
                locked.append(node)
        return locked

    def recommend_next_skill(
        self,
        skill_tree: dict[str, Any],
        max_recommendations: int = 3,
    ) -> list[str]:
        """推荐下一批学习技能 — 解锁的 learning 节点 (最薄弱优先).

        推荐逻辑:
        1. 在已解锁节点中筛选 status == "learning" 的节点;
        2. 按 mastery 升序排序 (最薄弱的优先, 集中精力补短板);
        3. 取前 ``max_recommendations`` 个, 返回其 kp_id 列表;
        4. 若无解锁的 learning 节点, 回退推荐 "weak" 节点供复习 (同样升序,
           受 ``max_recommendations`` 限制).

        Args:
            skill_tree: ``to_skill_tree`` 产出的技能树字典.
            max_recommendations: 最大推荐数量, 默认 3.

        Returns:
            推荐的 kp_id 列表 (按 mastery 升序). 无可推荐时返回空列表.
        """
        nodes: list[SkillNode] = skill_tree.get("nodes", [])
        unlocked = self.get_unlocked_skills(skill_tree)

        # 1. 解锁的 learning 节点
        learning_unlocked = [
            n for n in unlocked if n.status == "learning"
        ]
        if learning_unlocked:
            learning_unlocked.sort(key=lambda n: n.mastery)
            return [
                n.kp_id for n in learning_unlocked[:max_recommendations]
            ]

        # 2. 回退: weak 节点复习
        weak_nodes = [n for n in nodes if n.status == "weak"]
        weak_nodes.sort(key=lambda n: n.mastery)
        return [n.kp_id for n in weak_nodes[:max_recommendations]]

    def compute_progress(self, skill_tree: dict[str, Any]) -> dict[str, Any]:
        """计算技能树学习进度.

        统计字段:
        - ``total_nodes``: 节点总数
        - ``mastered_count``: 已掌握节点数 (status == "mastered")
        - ``learning_count``: 学习中节点数 (status == "learning")
        - ``weak_count``: 薄弱节点数 (status == "weak")
        - ``progress_percentage``: 掌握进度百分比
          (= mastered_count / total_nodes * 100, 空树为 0.0)

        Args:
            skill_tree: ``to_skill_tree`` 产出的技能树字典.

        Returns:
            进度统计字典.
        """
        nodes: list[SkillNode] = skill_tree.get("nodes", [])
        total_nodes = len(nodes)
        mastered_count = sum(1 for n in nodes if n.status == "mastered")
        learning_count = sum(1 for n in nodes if n.status == "learning")
        weak_count = sum(1 for n in nodes if n.status == "weak")

        progress_percentage = (
            mastered_count / total_nodes * 100.0
            if total_nodes > 0
            else 0.0
        )

        return {
            "total_nodes": total_nodes,
            "mastered_count": mastered_count,
            "learning_count": learning_count,
            "weak_count": weak_count,
            "progress_percentage": progress_percentage,
        }


# ============================================================
# __all__
# ============================================================

__all__ = [
    "SkillNode",
    "SkillEdge",
    "SkillMapper",
]
