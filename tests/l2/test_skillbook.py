"""L2 skillbook 子模块测试 — SkillNode / SkillEdge / SkillMapper.

测试覆盖 (TDD):
1. SkillNode (数据类):
   - 字段: kp_id, name, mastery, status, level
   - status 映射: mastery >= 0.7 -> mastered; 0.4 <= m < 0.7 -> learning;
     0 < m < 0.4 -> weak; mastery == 0 -> not_started
   - level 映射: m < 0.4 -> L0; 0.4 <= m < 0.7 -> L1; m >= 0.7 -> L2
   - to_dict() / from_dict() 往返序列化
2. SkillEdge (数据类):
   - 字段: from_kp, to_kp, edge_type ("prerequisite"/"related")
   - to_dict() / from_dict() 往返序列化
3. SkillMapper (无状态引擎):
   - to_skill_tree(tracing_states, irt_state, kg_structure=None) -> dict
       - global_ability = (theta + 3) / 6
       - nodes 从 tracing_states 构建
       - edges 从 kg_structure 提取 (无 kg_structure 时为空列表)
   - get_skill_status(mastery) / get_skill_level(mastery) 静态方法
   - get_summary(skill_tree) 统计各状态数量与 global_ability
4. 模块导出: skillbook 导出 SkillNode / SkillEdge / SkillMapper
"""

from __future__ import annotations

import pytest

from dy3_polaris.l2.models import IRTState, TracingState
from dy3_polaris.l2.skillbook import SkillEdge, SkillMapper, SkillNode


# ============================================================
# 1. SkillNode - 基本字段与构造
# ============================================================


class TestSkillNodeFields:
    """SkillNode 字段与默认值测试."""

    def test_skill_node_basic_fields(self):
        """SkillNode 含 kp_id / name / mastery / status / level 五字段."""
        node = SkillNode(
            kp_id="kp-001",
            name="加法",
            mastery=0.8,
        )
        assert node.kp_id == "kp-001"
        assert node.name == "加法"
        assert node.mastery == pytest.approx(0.8)
        assert node.status == "mastered"
        assert node.level == "L2"

    def test_skill_node_name_defaults_to_kp_id(self):
        """未提供 name 时, name 默认为 kp_id."""
        node = SkillNode(kp_id="kp-002", mastery=0.5)
        assert node.name == "kp-002"

    def test_skill_node_mastery_zero_is_not_started(self):
        """mastery == 0 -> status="not_started", level="L0"."""
        node = SkillNode(kp_id="kp", mastery=0.0)
        assert node.status == "not_started"
        assert node.level == "L0"

    def test_skill_node_mastery_default_zero(self):
        """未提供 mastery 时默认 0.0 (not_started)."""
        node = SkillNode(kp_id="kp")
        assert node.mastery == pytest.approx(0.0)
        assert node.status == "not_started"
        assert node.level == "L0"


# ============================================================
# 2. SkillNode - status 映射
# ============================================================


class TestSkillNodeStatus:
    """SkillNode status 字段根据 mastery 自动映射."""

    def test_status_mastered_at_threshold_0_7(self):
        """mastery == 0.7 -> status="mastered" (边界, 包含)."""
        node = SkillNode(kp_id="kp", mastery=0.7)
        assert node.status == "mastered"

    def test_status_mastered_above_threshold(self):
        """mastery > 0.7 -> status="mastered"."""
        node = SkillNode(kp_id="kp", mastery=0.95)
        assert node.status == "mastered"

    def test_status_learning_at_threshold_0_4(self):
        """mastery == 0.4 -> status="learning" (边界, 包含)."""
        node = SkillNode(kp_id="kp", mastery=0.4)
        assert node.status == "learning"

    def test_status_learning_between_0_4_and_0_7(self):
        """0.4 <= mastery < 0.7 -> status="learning"."""
        node = SkillNode(kp_id="kp", mastery=0.55)
        assert node.status == "learning"

    def test_status_learning_just_below_0_7(self):
        """mastery 略小于 0.7 -> status="learning"."""
        node = SkillNode(kp_id="kp", mastery=0.699)
        assert node.status == "learning"

    def test_status_weak_below_0_4(self):
        """0 < mastery < 0.4 -> status="weak"."""
        node = SkillNode(kp_id="kp", mastery=0.3)
        assert node.status == "weak"

    def test_status_weak_just_above_zero(self):
        """mastery 略大于 0 -> status="weak"."""
        node = SkillNode(kp_id="kp", mastery=0.001)
        assert node.status == "weak"

    def test_status_just_below_0_4_is_weak(self):
        """mastery 略小于 0.4 -> status="weak"."""
        node = SkillNode(kp_id="kp", mastery=0.399)
        assert node.status == "weak"

    def test_status_not_started_only_at_zero(self):
        """mastery == 0 才是 not_started (其余 < 0.4 都是 weak)."""
        assert SkillNode(kp_id="kp", mastery=0.0).status == "not_started"
        assert SkillNode(kp_id="kp", mastery=0.001).status == "weak"

    def test_status_full_mastery(self):
        """mastery == 1.0 -> status="mastered"."""
        node = SkillNode(kp_id="kp", mastery=1.0)
        assert node.status == "mastered"


# ============================================================
# 3. SkillNode - level 映射
# ============================================================


class TestSkillNodeLevel:
    """SkillNode level 字段根据 mastery 自动映射."""

    def test_level_l0_below_0_4(self):
        """mastery < 0.4 -> level="L0"."""
        for m in (0.0, 0.1, 0.3, 0.399):
            assert SkillNode(kp_id="kp", mastery=m).level == "L0"

    def test_level_l1_between_0_4_and_0_7(self):
        """0.4 <= mastery < 0.7 -> level="L1"."""
        for m in (0.4, 0.5, 0.6, 0.699):
            assert SkillNode(kp_id="kp", mastery=m).level == "L1"

    def test_level_l2_at_or_above_0_7(self):
        """mastery >= 0.7 -> level="L2"."""
        for m in (0.7, 0.8, 0.9, 1.0):
            assert SkillNode(kp_id="kp", mastery=m).level == "L2"

    def test_level_l0_at_zero(self):
        """mastery == 0 -> level="L0" (与 not_started 对应)."""
        assert SkillNode(kp_id="kp", mastery=0.0).level == "L0"

    def test_level_boundary_0_4_is_l1(self):
        """边界 mastery == 0.4 -> level="L1"."""
        assert SkillNode(kp_id="kp", mastery=0.4).level == "L1"

    def test_level_boundary_0_7_is_l2(self):
        """边界 mastery == 0.7 -> level="L2"."""
        assert SkillNode(kp_id="kp", mastery=0.7).level == "L2"


# ============================================================
# 4. SkillNode - 序列化
# ============================================================


class TestSkillNodeSerialization:
    """SkillNode to_dict / from_dict 往返序列化测试."""

    def test_to_dict_contains_all_fields(self):
        """to_dict 包含 kp_id / name / mastery / status / level."""
        node = SkillNode(kp_id="kp-001", name="加法", mastery=0.8)
        d = node.to_dict()
        assert d["kp_id"] == "kp-001"
        assert d["name"] == "加法"
        assert d["mastery"] == pytest.approx(0.8)
        assert d["status"] == "mastered"
        assert d["level"] == "L2"

    def test_from_dict_reconstructs_node(self):
        """from_dict 还原 SkillNode 实例."""
        original = SkillNode(kp_id="kp-001", name="减法", mastery=0.5)
        d = original.to_dict()
        restored = SkillNode.from_dict(d)
        assert restored.kp_id == original.kp_id
        assert restored.name == original.name
        assert restored.mastery == pytest.approx(original.mastery)
        assert restored.status == original.status
        assert restored.level == original.level

    def test_roundtrip_preserves_status_and_level(self):
        """往返序列化保持 status / level 一致."""
        for m in (0.0, 0.3, 0.5, 0.8, 1.0):
            node = SkillNode(kp_id="kp", mastery=m)
            restored = SkillNode.from_dict(node.to_dict())
            assert restored.status == node.status
            assert restored.level == node.level

    def test_to_dict_does_not_mutate_node(self):
        """to_dict 不修改原对象."""
        node = SkillNode(kp_id="kp", name="n", mastery=0.6)
        d = node.to_dict()
        d["mastery"] = 0.99
        d["status"] = "mastered"
        assert node.mastery == pytest.approx(0.6)
        assert node.status == "learning"

    def test_from_dict_with_explicit_status_and_level(self):
        """from_dict 可接受外部传入的 status / level (即使与 mastery 不符)."""
        d = {"kp_id": "kp", "name": "n", "mastery": 0.5,
             "status": "mastered", "level": "L2"}
        node = SkillNode.from_dict(d)
        assert node.status == "mastered"
        assert node.level == "L2"


# ============================================================
# 5. SkillEdge - 字段与序列化
# ============================================================


class TestSkillEdge:
    """SkillEdge 数据类测试."""

    def test_skill_edge_basic_fields(self):
        """SkillEdge 含 from_kp / to_kp / edge_type 三字段."""
        edge = SkillEdge(from_kp="kp-A", to_kp="kp-B", edge_type="prerequisite")
        assert edge.from_kp == "kp-A"
        assert edge.to_kp == "kp-B"
        assert edge.edge_type == "prerequisite"

    def test_skill_edge_related_type(self):
        """edge_type 支持 "related"."""
        edge = SkillEdge(from_kp="kp-A", to_kp="kp-C", edge_type="related")
        assert edge.edge_type == "related"

    def test_skill_edge_to_dict(self):
        """to_dict 包含 from_kp / to_kp / edge_type."""
        edge = SkillEdge(from_kp="kp-A", to_kp="kp-B", edge_type="prerequisite")
        d = edge.to_dict()
        assert d["from_kp"] == "kp-A"
        assert d["to_kp"] == "kp-B"
        assert d["edge_type"] == "prerequisite"

    def test_skill_edge_from_dict_reconstructs(self):
        """from_dict 还原 SkillEdge 实例."""
        original = SkillEdge(from_kp="kp-1", to_kp="kp-2", edge_type="related")
        restored = SkillEdge.from_dict(original.to_dict())
        assert restored.from_kp == original.from_kp
        assert restored.to_kp == original.to_kp
        assert restored.edge_type == original.edge_type

    def test_skill_edge_roundtrip(self):
        """往返序列化保持字段一致."""
        for et in ("prerequisite", "related"):
            edge = SkillEdge(from_kp="a", to_kp="b", edge_type=et)
            restored = SkillEdge.from_dict(edge.to_dict())
            assert restored.edge_type == et


# ============================================================
# 6. SkillMapper - get_skill_status / get_skill_level 静态方法
# ============================================================


class TestSkillMapperStaticMethods:
    """SkillMapper.get_skill_status / get_skill_level 静态方法测试."""

    def test_get_skill_status_mastered(self):
        """mastery >= 0.7 -> 'mastered'."""
        assert SkillMapper.get_skill_status(0.7) == "mastered"
        assert SkillMapper.get_skill_status(0.9) == "mastered"
        assert SkillMapper.get_skill_status(1.0) == "mastered"

    def test_get_skill_status_learning(self):
        """0.4 <= mastery < 0.7 -> 'learning'."""
        assert SkillMapper.get_skill_status(0.4) == "learning"
        assert SkillMapper.get_skill_status(0.5) == "learning"
        assert SkillMapper.get_skill_status(0.699) == "learning"

    def test_get_skill_status_weak(self):
        """0 < mastery < 0.4 -> 'weak'."""
        assert SkillMapper.get_skill_status(0.001) == "weak"
        assert SkillMapper.get_skill_status(0.3) == "weak"
        assert SkillMapper.get_skill_status(0.399) == "weak"

    def test_get_skill_status_not_started(self):
        """mastery == 0 -> 'not_started'."""
        assert SkillMapper.get_skill_status(0.0) == "not_started"

    def test_get_skill_level_l0(self):
        """mastery < 0.4 -> 'L0'."""
        assert SkillMapper.get_skill_level(0.0) == "L0"
        assert SkillMapper.get_skill_level(0.3) == "L0"
        assert SkillMapper.get_skill_level(0.399) == "L0"

    def test_get_skill_level_l1(self):
        """0.4 <= mastery < 0.7 -> 'L1'."""
        assert SkillMapper.get_skill_level(0.4) == "L1"
        assert SkillMapper.get_skill_level(0.5) == "L1"
        assert SkillMapper.get_skill_level(0.699) == "L1"

    def test_get_skill_level_l2(self):
        """mastery >= 0.7 -> 'L2'."""
        assert SkillMapper.get_skill_level(0.7) == "L2"
        assert SkillMapper.get_skill_level(0.9) == "L2"
        assert SkillMapper.get_skill_level(1.0) == "L2"

    def test_get_skill_status_consistent_with_skill_node(self):
        """get_skill_status 与 SkillNode.status 一致."""
        for m in (0.0, 0.1, 0.4, 0.5, 0.7, 0.9, 1.0):
            node = SkillNode(kp_id="kp", mastery=m)
            assert node.status == SkillMapper.get_skill_status(m)

    def test_get_skill_level_consistent_with_skill_node(self):
        """get_skill_level 与 SkillNode.level 一致."""
        for m in (0.0, 0.1, 0.4, 0.5, 0.7, 0.9, 1.0):
            node = SkillNode(kp_id="kp", mastery=m)
            assert node.level == SkillMapper.get_skill_level(m)

    def test_static_methods_callable_without_instance(self):
        """静态方法无需实例化即可调用."""
        assert SkillMapper.get_skill_status(0.8) == "mastered"
        assert SkillMapper.get_skill_level(0.8) == "L2"


# ============================================================
# 7. SkillMapper - to_skill_tree: global_ability 映射
# ============================================================


class TestSkillMapperGlobalAbility:
    """SkillMapper.to_skill_tree 中 global_ability (theta -> [0,1]) 映射测试."""

    def test_global_ability_formula(self):
        """global_ability = (theta + 3) / 6."""
        mapper = SkillMapper()
        irt = IRTState(theta=0.0)
        tree = mapper.to_skill_tree({}, irt)
        # (0 + 3) / 6 = 0.5
        assert tree["global_ability"] == pytest.approx(0.5)

    def test_global_ability_theta_positive_3(self):
        """theta = 3 -> global_ability = 1.0 (上限)."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=3.0))
        assert tree["global_ability"] == pytest.approx(1.0)

    def test_global_ability_theta_negative_3(self):
        """theta = -3 -> global_ability = 0.0 (下限)."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=-3.0))
        assert tree["global_ability"] == pytest.approx(0.0)

    def test_global_ability_theta_2(self):
        """theta = 2 -> global_ability = 5/6."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=2.0))
        assert tree["global_ability"] == pytest.approx(5.0 / 6.0)

    def test_global_ability_theta_negative_1(self):
        """theta = -1 -> global_ability = 2/6 = 1/3."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=-1.0))
        assert tree["global_ability"] == pytest.approx(1.0 / 3.0)

    def test_global_ability_is_float(self):
        """global_ability 为 float 类型."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=0.5))
        assert isinstance(tree["global_ability"], float)


# ============================================================
# 8. SkillMapper - to_skill_tree: nodes 构建
# ============================================================


class TestSkillMapperNodes:
    """SkillMapper.to_skill_tree 中 nodes 构建测试."""

    def _make_tracing_state(self, kp_id, mastery, **kwargs):
        return TracingState(kp_id=kp_id, mastery_prob=mastery, **kwargs)

    def test_nodes_empty_tracing_states(self):
        """空 tracing_states -> nodes 为空列表."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=0.0))
        assert tree["nodes"] == []

    def test_nodes_single_tracing_state(self):
        """单个 TracingState -> 单个 SkillNode."""
        mapper = SkillMapper()
        states = {"kp-1": self._make_tracing_state("kp-1", 0.8)}
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        nodes = tree["nodes"]
        assert len(nodes) == 1
        assert nodes[0].kp_id == "kp-1"
        assert nodes[0].mastery == pytest.approx(0.8)
        assert nodes[0].status == "mastered"
        assert nodes[0].level == "L2"

    def test_nodes_multiple_tracing_states(self):
        """多个 TracingState -> 多个 SkillNode."""
        mapper = SkillMapper()
        states = {
            "kp-1": self._make_tracing_state("kp-1", 0.0),
            "kp-2": self._make_tracing_state("kp-2", 0.5),
            "kp-3": self._make_tracing_state("kp-3", 0.9),
        }
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        nodes = tree["nodes"]
        assert len(nodes) == 3
        kp_ids = {n.kp_id for n in nodes}
        assert kp_ids == {"kp-1", "kp-2", "kp-3"}

    def test_nodes_mastery_from_mastery_prob(self):
        """SkillNode.mastery 取自 TracingState.mastery_prob."""
        mapper = SkillMapper()
        states = {"kp-1": self._make_tracing_state("kp-1", 0.65)}
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        assert tree["nodes"][0].mastery == pytest.approx(0.65)

    def test_nodes_status_derived_from_mastery(self):
        """SkillNode.status 由 mastery 自动推导."""
        mapper = SkillMapper()
        states = {
            "kp-1": self._make_tracing_state("kp-1", 0.3),
            "kp-2": self._make_tracing_state("kp-2", 0.5),
            "kp-3": self._make_tracing_state("kp-3", 0.8),
            "kp-4": self._make_tracing_state("kp-4", 0.0),
        }
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        status_by_kp = {n.kp_id: n.status for n in tree["nodes"]}
        assert status_by_kp["kp-1"] == "weak"
        assert status_by_kp["kp-2"] == "learning"
        assert status_by_kp["kp-3"] == "mastered"
        assert status_by_kp["kp-4"] == "not_started"

    def test_nodes_level_derived_from_mastery(self):
        """SkillNode.level 由 mastery 自动推导."""
        mapper = SkillMapper()
        states = {
            "kp-1": self._make_tracing_state("kp-1", 0.3),
            "kp-2": self._make_tracing_state("kp-2", 0.5),
            "kp-3": self._make_tracing_state("kp-3", 0.8),
        }
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        level_by_kp = {n.kp_id: n.level for n in tree["nodes"]}
        assert level_by_kp["kp-1"] == "L0"
        assert level_by_kp["kp-2"] == "L1"
        assert level_by_kp["kp-3"] == "L2"

    def test_nodes_name_defaults_to_kp_id_without_kg(self):
        """无 kg_structure 时, SkillNode.name 默认为 kp_id."""
        mapper = SkillMapper()
        states = {"kp-1": self._make_tracing_state("kp-1", 0.5)}
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        assert tree["nodes"][0].name == "kp-1"

    def test_nodes_are_skill_node_instances(self):
        """nodes 列表元素均为 SkillNode 实例."""
        mapper = SkillMapper()
        states = {"kp-1": self._make_tracing_state("kp-1", 0.5)}
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        assert all(isinstance(n, SkillNode) for n in tree["nodes"])

    def test_nodes_use_mastery_prob_not_other_fields(self):
        """SkillNode 只使用 mastery_prob, 不使用 attempts/correct_count."""
        mapper = SkillMapper()
        states = {
            "kp-1": self._make_tracing_state(
                "kp-1", 0.8, attempts=10, correct_count=9,
            ),
        }
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        node = tree["nodes"][0]
        assert node.mastery == pytest.approx(0.8)
        assert node.status == "mastered"


# ============================================================
# 9. SkillMapper - to_skill_tree: edges 构建
# ============================================================


class TestSkillMapperEdges:
    """SkillMapper.to_skill_tree 中 edges 构建测试."""

    def test_edges_empty_without_kg_structure(self):
        """无 kg_structure 参数 -> edges 为空列表."""
        mapper = SkillMapper()
        states = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        assert tree["edges"] == []

    def test_edges_empty_when_kg_structure_is_none(self):
        """kg_structure=None -> edges 为空列表."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree(
            {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)},
            IRTState(theta=0.0),
            kg_structure=None,
        )
        assert tree["edges"] == []

    def test_edges_empty_when_kg_structure_has_no_edges(self):
        """kg_structure 无 edges 键 -> edges 为空列表."""
        mapper = SkillMapper()
        kg = {"nodes": [{"kp_id": "kp-1", "name": "n1"}]}
        tree = mapper.to_skill_tree(
            {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)},
            IRTState(theta=0.0),
            kg_structure=kg,
        )
        assert tree["edges"] == []

    def test_edges_from_kg_structure_prerequisite(self):
        """从 kg_structure 提取 prerequisite 边."""
        mapper = SkillMapper()
        kg = {
            "nodes": [
                {"kp_id": "kp-A", "name": "加法"},
                {"kp_id": "kp-B", "name": "减法"},
            ],
            "edges": [
                {"from": "kp-A", "to": "kp-B", "type": "prerequisite"},
            ],
        }
        tree = mapper.to_skill_tree(
            {
                "kp-A": TracingState(kp_id="kp-A", mastery_prob=0.8),
                "kp-B": TracingState(kp_id="kp-B", mastery_prob=0.4),
            },
            IRTState(theta=0.0),
            kg_structure=kg,
        )
        edges = tree["edges"]
        assert len(edges) == 1
        assert isinstance(edges[0], SkillEdge)
        assert edges[0].from_kp == "kp-A"
        assert edges[0].to_kp == "kp-B"
        assert edges[0].edge_type == "prerequisite"

    def test_edges_multiple_from_kg_structure(self):
        """从 kg_structure 提取多条边 (含 prerequisite / related)."""
        mapper = SkillMapper()
        kg = {
            "nodes": [],
            "edges": [
                {"from": "kp-A", "to": "kp-B", "type": "prerequisite"},
                {"from": "kp-B", "to": "kp-C", "type": "prerequisite"},
                {"from": "kp-A", "to": "kp-C", "type": "related"},
            ],
        }
        tree = mapper.to_skill_tree({}, IRTState(theta=0.0), kg_structure=kg)
        edges = tree["edges"]
        assert len(edges) == 3
        types = [e.edge_type for e in edges]
        assert types.count("prerequisite") == 2
        assert types.count("related") == 1

    def test_edges_are_skill_edge_instances(self):
        """edges 列表元素均为 SkillEdge 实例."""
        mapper = SkillMapper()
        kg = {
            "nodes": [],
            "edges": [
                {"from": "a", "to": "b", "type": "prerequisite"},
            ],
        }
        tree = mapper.to_skill_tree({}, IRTState(theta=0.0), kg_structure=kg)
        assert all(isinstance(e, SkillEdge) for e in tree["edges"])


# ============================================================
# 10. SkillMapper - to_skill_tree: kg_structure 名称注入
# ============================================================


class TestSkillMapperKgNames:
    """SkillMapper.to_skill_tree 中 kg_structure 注入节点名称测试."""

    def test_node_name_from_kg_structure(self):
        """kg_structure 提供节点名称时, SkillNode.name 取 KG 名称."""
        mapper = SkillMapper()
        kg = {
            "nodes": [{"kp_id": "kp-1", "name": "一元一次方程"}],
            "edges": [],
        }
        states = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        tree = mapper.to_skill_tree(
            states, IRTState(theta=0.0), kg_structure=kg,
        )
        assert tree["nodes"][0].name == "一元一次方程"

    def test_node_name_fallback_to_kp_id_when_kg_missing_name(self):
        """kg_structure 节点缺 name 字段时, 回退到 kp_id."""
        mapper = SkillMapper()
        kg = {"nodes": [{"kp_id": "kp-1"}], "edges": []}
        states = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        tree = mapper.to_skill_tree(
            states, IRTState(theta=0.0), kg_structure=kg,
        )
        assert tree["nodes"][0].name == "kp-1"

    def test_node_name_fallback_when_kp_not_in_kg(self):
        """TracingState 的 kp_id 不在 kg_structure.nodes 中 -> name 用 kp_id."""
        mapper = SkillMapper()
        kg = {
            "nodes": [{"kp_id": "kp-other", "name": "其他"}],
            "edges": [],
        }
        states = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        tree = mapper.to_skill_tree(
            states, IRTState(theta=0.0), kg_structure=kg,
        )
        assert tree["nodes"][0].name == "kp-1"

    def test_node_name_fallback_when_kg_has_no_nodes_key(self):
        """kg_structure 无 nodes 键时, name 用 kp_id."""
        mapper = SkillMapper()
        kg = {"edges": []}
        states = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        tree = mapper.to_skill_tree(
            states, IRTState(theta=0.0), kg_structure=kg,
        )
        assert tree["nodes"][0].name == "kp-1"


# ============================================================
# 11. SkillMapper - to_skill_tree: 返回结构
# ============================================================


class TestSkillMapperTreeStructure:
    """SkillMapper.to_skill_tree 返回结构测试."""

    def test_tree_has_required_keys(self):
        """返回字典含 global_ability / nodes / edges 三键."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=0.0))
        assert set(tree.keys()) >= {"global_ability", "nodes", "edges"}

    def test_tree_nodes_is_list(self):
        """nodes 为 list 类型."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=0.0))
        assert isinstance(tree["nodes"], list)

    def test_tree_edges_is_list(self):
        """edges 为 list 类型."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=0.0))
        assert isinstance(tree["edges"], list)

    def test_tree_full_structure(self):
        """完整技能树结构: global_ability + nodes + edges."""
        mapper = SkillMapper()
        states = {
            "kp-A": TracingState(kp_id="kp-A", mastery_prob=0.8),
            "kp-B": TracingState(kp_id="kp-B", mastery_prob=0.3),
        }
        kg = {
            "nodes": [
                {"kp_id": "kp-A", "name": "加法"},
                {"kp_id": "kp-B", "name": "减法"},
            ],
            "edges": [
                {"from": "kp-A", "to": "kp-B", "type": "prerequisite"},
            ],
        }
        tree = mapper.to_skill_tree(
            states, IRTState(theta=1.0), kg_structure=kg,
        )
        assert tree["global_ability"] == pytest.approx((1.0 + 3) / 6.0)
        assert len(tree["nodes"]) == 2
        assert len(tree["edges"]) == 1


# ============================================================
# 12. SkillMapper - get_summary
# ============================================================


class TestSkillMapperGetSummary:
    """SkillMapper.get_summary 测试 — 技能树摘要统计."""

    def test_summary_has_required_keys(self):
        """返回字典含 total_kps / mastered / learning / weak / not_started / global_ability."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=0.0))
        summary = mapper.get_summary(tree)
        expected_keys = {
            "total_kps", "mastered", "learning",
            "weak", "not_started", "global_ability",
        }
        assert set(summary.keys()) >= expected_keys

    def test_summary_empty_tree(self):
        """空技能树: 全部计数为 0."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=0.0))
        summary = mapper.get_summary(tree)
        assert summary["total_kps"] == 0
        assert summary["mastered"] == 0
        assert summary["learning"] == 0
        assert summary["weak"] == 0
        assert summary["not_started"] == 0
        assert summary["global_ability"] == pytest.approx(0.5)

    def test_summary_counts_by_status(self):
        """按 status 统计各状态节点数."""
        mapper = SkillMapper()
        states = {
            "kp-1": TracingState(kp_id="kp-1", mastery_prob=0.0),    # not_started
            "kp-2": TracingState(kp_id="kp-2", mastery_prob=0.3),    # weak
            "kp-3": TracingState(kp_id="kp-3", mastery_prob=0.5),    # learning
            "kp-4": TracingState(kp_id="kp-4", mastery_prob=0.8),    # mastered
            "kp-5": TracingState(kp_id="kp-5", mastery_prob=0.9),    # mastered
        }
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        summary = mapper.get_summary(tree)
        assert summary["total_kps"] == 5
        assert summary["mastered"] == 2
        assert summary["learning"] == 1
        assert summary["weak"] == 1
        assert summary["not_started"] == 1

    def test_summary_total_equals_sum_of_statuses(self):
        """total_kps == mastered + learning + weak + not_started."""
        mapper = SkillMapper()
        states = {
            "kp-1": TracingState(kp_id="kp-1", mastery_prob=0.0),
            "kp-2": TracingState(kp_id="kp-2", mastery_prob=0.3),
            "kp-3": TracingState(kp_id="kp-3", mastery_prob=0.5),
            "kp-4": TracingState(kp_id="kp-4", mastery_prob=0.8),
        }
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        summary = mapper.get_summary(tree)
        total = (
            summary["mastered"]
            + summary["learning"]
            + summary["weak"]
            + summary["not_started"]
        )
        assert summary["total_kps"] == total

    def test_summary_global_ability_from_tree(self):
        """global_ability 取自技能树的 global_ability 字段."""
        mapper = SkillMapper()
        tree = mapper.to_skill_tree({}, IRTState(theta=2.0))
        summary = mapper.get_summary(tree)
        assert summary["global_ability"] == pytest.approx((2.0 + 3) / 6.0)

    def test_summary_all_mastered(self):
        """全部掌握: mastered == total_kps, 其余为 0."""
        mapper = SkillMapper()
        states = {
            "kp-1": TracingState(kp_id="kp-1", mastery_prob=0.9),
            "kp-2": TracingState(kp_id="kp-2", mastery_prob=0.8),
        }
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        summary = mapper.get_summary(tree)
        assert summary["mastered"] == 2
        assert summary["total_kps"] == 2
        assert summary["learning"] == 0
        assert summary["weak"] == 0
        assert summary["not_started"] == 0

    def test_summary_counts_are_int(self):
        """统计计数为 int 类型."""
        mapper = SkillMapper()
        states = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0))
        summary = mapper.get_summary(tree)
        assert isinstance(summary["total_kps"], int)
        assert isinstance(summary["mastered"], int)
        assert isinstance(summary["learning"], int)
        assert isinstance(summary["weak"], int)
        assert isinstance(summary["not_started"], int)


# ============================================================
# 13. SkillMapper - 无状态性
# ============================================================


class TestSkillMapperStateless:
    """SkillMapper 无状态引擎测试 — 多次实例化行为一致."""

    def test_multiple_instances_equivalent(self):
        """多次实例化 SkillMapper 行为一致."""
        m1 = SkillMapper()
        m2 = SkillMapper()
        states = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        irt = IRTState(theta=0.0)
        t1 = m1.to_skill_tree(states, irt)
        t2 = m2.to_skill_tree(states, irt)
        assert t1["global_ability"] == t2["global_ability"]
        assert len(t1["nodes"]) == len(t2["nodes"])
        assert t1["nodes"][0].status == t2["nodes"][0].status

    def test_does_not_mutate_tracing_states(self):
        """to_skill_tree 不修改入参 tracing_states."""
        mapper = SkillMapper()
        states = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        original_mastery = states["kp-1"].mastery_prob
        mapper.to_skill_tree(states, IRTState(theta=0.0))
        assert states["kp-1"].mastery_prob == original_mastery

    def test_does_not_mutate_kg_structure(self):
        """to_skill_tree 不修改入参 kg_structure."""
        mapper = SkillMapper()
        kg = {
            "nodes": [{"kp_id": "kp-1", "name": "n1"}],
            "edges": [{"from": "a", "to": "b", "type": "prerequisite"}],
        }
        original_nodes = list(kg["nodes"])
        original_edges = list(kg["edges"])
        states = {"kp-1": TracingState(kp_id="kp-1", mastery_prob=0.5)}
        mapper.to_skill_tree(states, IRTState(theta=0.0), kg_structure=kg)
        assert kg["nodes"] == original_nodes
        assert kg["edges"] == original_edges


# ============================================================
# 14. 模块导出
# ============================================================


class TestSkillbookModuleExport:
    """skillbook 模块导出测试."""

    def test_skillbook_exports_skill_node(self):
        """skillbook 导出 SkillNode."""
        from dy3_polaris.l2 import skillbook
        assert hasattr(skillbook, "SkillNode")
        assert "SkillNode" in skillbook.__all__

    def test_skillbook_exports_skill_edge(self):
        """skillbook 导出 SkillEdge."""
        from dy3_polaris.l2 import skillbook
        assert hasattr(skillbook, "SkillEdge")
        assert "SkillEdge" in skillbook.__all__

    def test_skillbook_exports_skill_mapper(self):
        """skillbook 导出 SkillMapper."""
        from dy3_polaris.l2 import skillbook
        assert hasattr(skillbook, "SkillMapper")
        assert "SkillMapper" in skillbook.__all__

    def test_can_import_from_skillbook_package(self):
        """可从 skillbook 包直接导入三个类."""
        from dy3_polaris.l2.skillbook import (
            SkillEdge,
            SkillMapper,
            SkillNode,
        )
        assert SkillNode is not None
        assert SkillEdge is not None
        assert SkillMapper is not None

    def test_can_import_from_skill_mapper_module(self):
        """可从 skill_mapper 模块导入三个类."""
        from dy3_polaris.l2.skillbook.skill_mapper import (
            SkillEdge,
            SkillMapper,
            SkillNode,
        )
        assert SkillNode is not None
        assert SkillEdge is not None
        assert SkillMapper is not None


# ============================================================
# 15. SkillMapper - 学习路径推荐辅助函数
# ============================================================


def _make_tree(
    nodes_spec: list[tuple[str, float]],
    prereq_edges: list[tuple[str, str]] | None = None,
    related_edges: list[tuple[str, str]] | None = None,
) -> dict:
    """构造技能树字典 (用于推荐 / 解锁 / 锁定 / 进度测试).

    Args:
        nodes_spec: [(kp_id, mastery), ...]
        prereq_edges: [(from_kp, to_kp), ...] 前置依赖边
        related_edges: [(from_kp, to_kp), ...] 关联边
    """
    nodes = [SkillNode(kp_id=kp, mastery=m) for kp, m in nodes_spec]
    edges: list[SkillEdge] = []
    for frm, to in prereq_edges or []:
        edges.append(SkillEdge(from_kp=frm, to_kp=to, edge_type="prerequisite"))
    for frm, to in related_edges or []:
        edges.append(SkillEdge(from_kp=frm, to_kp=to, edge_type="related"))
    return {"global_ability": 0.5, "nodes": nodes, "edges": edges}


# ============================================================
# 16. SkillMapper - get_unlocked_skills
# ============================================================


class TestSkillMapperGetUnlockedSkills:
    """SkillMapper.get_unlocked_skills 测试 — 前置全部 mastered 的节点."""

    def test_get_unlocked_empty_tree(self):
        """空技能树 -> 空列表."""
        mapper = SkillMapper()
        tree = _make_tree([])
        assert mapper.get_unlocked_skills(tree) == []

    def test_get_unlocked_no_prereqs_all_unlocked(self):
        """无前置依赖的节点全部解锁 (空前置视为已满足)."""
        mapper = SkillMapper()
        tree = _make_tree([("kp-A", 0.9), ("kp-B", 0.2)])
        unlocked = mapper.get_unlocked_skills(tree)
        assert {n.kp_id for n in unlocked} == {"kp-A", "kp-B"}

    def test_get_unlocked_all_prereqs_mastered(self):
        """前置全部 mastered -> 节点解锁."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.5)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        unlocked = mapper.get_unlocked_skills(tree)
        # kp-A 无前置 -> 解锁; kp-B 前置 kp-A 已 mastered -> 解锁
        assert {n.kp_id for n in unlocked} == {"kp-A", "kp-B"}

    def test_get_unlocked_prereq_not_mastered(self):
        """前置未 mastered -> 节点未解锁."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.5), ("kp-B", 0.6)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        unlocked = mapper.get_unlocked_skills(tree)
        # kp-A 无前置 -> 解锁; kp-B 前置 kp-A 为 learning -> 未解锁
        assert {n.kp_id for n in unlocked} == {"kp-A"}

    def test_get_unlocked_multiple_prereqs_all_mastered(self):
        """多个前置全部 mastered -> 节点解锁."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.8), ("kp-C", 0.5)],
            prereq_edges=[("kp-A", "kp-C"), ("kp-B", "kp-C")],
        )
        unlocked = mapper.get_unlocked_skills(tree)
        assert {n.kp_id for n in unlocked} == {"kp-A", "kp-B", "kp-C"}

    def test_get_unlocked_multiple_prereqs_one_not_mastered(self):
        """多个前置中有一个未 mastered -> 节点未解锁."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.2), ("kp-C", 0.5)],
            prereq_edges=[("kp-A", "kp-C"), ("kp-B", "kp-C")],
        )
        unlocked = mapper.get_unlocked_skills(tree)
        # kp-C 前置 kp-B 为 weak -> 未解锁
        assert "kp-C" not in {n.kp_id for n in unlocked}

    def test_get_unlocked_ignores_related_edges(self):
        """related 类型边不影响解锁判定."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.2), ("kp-B", 0.5)],
            related_edges=[("kp-A", "kp-B")],
        )
        unlocked = mapper.get_unlocked_skills(tree)
        # kp-B 只有 related 边 (非 prerequisite) -> 视为无前置 -> 解锁
        assert {n.kp_id for n in unlocked} == {"kp-A", "kp-B"}

    def test_get_unlocked_returns_skill_node_list(self):
        """返回值为 SkillNode 列表."""
        mapper = SkillMapper()
        tree = _make_tree([("kp-A", 0.9)])
        unlocked = mapper.get_unlocked_skills(tree)
        assert isinstance(unlocked, list)
        assert all(isinstance(n, SkillNode) for n in unlocked)

    def test_get_unlocked_works_with_to_skill_tree_output(self):
        """get_unlocked_skills 兼容 to_skill_tree 产出."""
        mapper = SkillMapper()
        states = {
            "kp-A": TracingState(kp_id="kp-A", mastery_prob=0.9),
            "kp-B": TracingState(kp_id="kp-B", mastery_prob=0.5),
        }
        kg = {
            "nodes": [],
            "edges": [{"from": "kp-A", "to": "kp-B", "type": "prerequisite"}],
        }
        tree = mapper.to_skill_tree(states, IRTState(theta=0.0), kg_structure=kg)
        unlocked = mapper.get_unlocked_skills(tree)
        assert {n.kp_id for n in unlocked} == {"kp-A", "kp-B"}


# ============================================================
# 17. SkillMapper - get_locked_skills
# ============================================================


class TestSkillMapperGetLockedSkills:
    """SkillMapper.get_locked_skills 测试 — 含 not_started/weak 前置的节点."""

    def test_get_locked_empty_tree(self):
        """空技能树 -> 空列表."""
        mapper = SkillMapper()
        assert mapper.get_locked_skills(_make_tree([])) == []

    def test_get_locked_no_prereqs_not_locked(self):
        """无前置依赖的节点不被锁定."""
        mapper = SkillMapper()
        tree = _make_tree([("kp-A", 0.9), ("kp-B", 0.2)])
        assert mapper.get_locked_skills(tree) == []

    def test_get_locked_has_not_started_prereq(self):
        """前置为 not_started -> 节点锁定."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.0), ("kp-B", 0.5)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        locked = mapper.get_locked_skills(tree)
        assert {n.kp_id for n in locked} == {"kp-B"}

    def test_get_locked_has_weak_prereq(self):
        """前置为 weak -> 节点锁定."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.2), ("kp-B", 0.5)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        locked = mapper.get_locked_skills(tree)
        assert {n.kp_id for n in locked} == {"kp-B"}

    def test_get_locked_has_mastered_prereq_only_not_locked(self):
        """前置为 mastered -> 节点不锁定."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.5)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        assert mapper.get_locked_skills(tree) == []

    def test_get_locked_has_learning_prereq_not_locked(self):
        """前置为 learning (非 not_started/weak) -> 节点不锁定."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.5), ("kp-B", 0.6)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        assert mapper.get_locked_skills(tree) == []

    def test_get_locked_ignores_related_edges(self):
        """related 类型边不触发锁定."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.0), ("kp-B", 0.5)],
            related_edges=[("kp-A", "kp-B")],
        )
        assert mapper.get_locked_skills(tree) == []

    def test_get_locked_multiple_prereqs_one_blocking(self):
        """多个前置中有一个 not_started/weak -> 节点锁定."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.1), ("kp-C", 0.5)],
            prereq_edges=[("kp-A", "kp-C"), ("kp-B", "kp-C")],
        )
        locked = mapper.get_locked_skills(tree)
        assert {n.kp_id for n in locked} == {"kp-C"}

    def test_get_locked_returns_skill_node_list(self):
        """返回值为 SkillNode 列表."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.0), ("kp-B", 0.5)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        locked = mapper.get_locked_skills(tree)
        assert isinstance(locked, list)
        assert all(isinstance(n, SkillNode) for n in locked)


# ============================================================
# 18. SkillMapper - recommend_next_skill
# ============================================================


class TestSkillMapperRecommendNextSkill:
    """SkillMapper.recommend_next_skill 测试 — 学习路径推荐."""

    def test_recommend_returns_learning_unlocked(self):
        """推荐解锁的 learning 节点."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.5), ("kp-C", 0.6)],
            prereq_edges=[("kp-A", "kp-B"), ("kp-A", "kp-C")],
        )
        recs = mapper.recommend_next_skill(tree)
        assert set(recs) == {"kp-B", "kp-C"}

    def test_recommend_sorted_ascending_by_mastery(self):
        """推荐按 mastery 升序 (最薄弱优先)."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.6), ("kp-C", 0.45)],
            prereq_edges=[("kp-A", "kp-B"), ("kp-A", "kp-C")],
        )
        recs = mapper.recommend_next_skill(tree)
        # kp-C (0.45) < kp-B (0.6)
        assert recs == ["kp-C", "kp-B"]

    def test_recommend_max_recommendations_limits_count(self):
        """max_recommendations 限制返回数量."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.6), ("kp-C", 0.45), ("kp-D", 0.55)],
            prereq_edges=[
                ("kp-A", "kp-B"), ("kp-A", "kp-C"), ("kp-A", "kp-D"),
            ],
        )
        recs = mapper.recommend_next_skill(tree, max_recommendations=2)
        assert len(recs) == 2
        # 升序前两个: 0.45 (kp-C), 0.55 (kp-D)
        assert recs == ["kp-C", "kp-D"]

    def test_recommend_default_max_3(self):
        """默认 max_recommendations=3."""
        mapper = SkillMapper()
        nodes = [("kp-A", 0.9)] + [
            (f"kp-{i}", m) for i, m in enumerate([0.45, 0.5, 0.55, 0.6], start=1)
        ]
        prereqs = [("kp-A", f"kp-{i}") for i in range(1, 5)]
        tree = _make_tree(nodes, prereq_edges=prereqs)
        recs = mapper.recommend_next_skill(tree)
        assert len(recs) == 3

    def test_recommend_excludes_mastered_nodes(self):
        """已 mastered 节点不参与推荐."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.95)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        # kp-B 已 mastered -> 不推荐; 无 learning 节点 -> 回退 weak (无) -> []
        recs = mapper.recommend_next_skill(tree)
        assert recs == []

    def test_recommend_excludes_locked_learning(self):
        """被锁定的 learning 节点不参与推荐."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.2), ("kp-B", 0.5)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        # kp-B 为 learning 但前置 kp-A 为 weak (锁定) -> 回退 weak -> ["kp-A"]
        recs = mapper.recommend_next_skill(tree)
        assert recs == ["kp-A"]

    def test_recommend_no_unlocked_learning_returns_weak(self):
        """无可推荐的解锁 learning 节点 -> 返回 weak 节点复习."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.45), ("kp-C", 0.2)],
            prereq_edges=[("kp-C", "kp-B")],  # kp-B 前置 kp-C 为 weak -> 锁定
        )
        recs = mapper.recommend_next_skill(tree)
        # 无解锁 learning (kp-B 锁定) -> 回退 weak -> ["kp-C"]
        assert recs == ["kp-C"]

    def test_recommend_weak_sorted_ascending(self):
        """weak 回退按 mastery 升序."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-E", 0.2), ("kp-F", 0.5), ("kp-G", 0.1)],
            prereq_edges=[("kp-E", "kp-F"), ("kp-G", "kp-F")],
            # kp-F 为 learning 但前置 kp-E/kp-G 为 weak -> 锁定; 无解锁 learning
        )
        recs = mapper.recommend_next_skill(tree)
        # weak: kp-E(0.2), kp-G(0.1) [kp-A mastered, kp-F(0.5) learning]
        # 升序: kp-G(0.1), kp-E(0.2)
        assert recs == ["kp-G", "kp-E"]

    def test_recommend_weak_respects_max(self):
        """weak 回退也受 max_recommendations 限制."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-E", 0.3), ("kp-F", 0.2), ("kp-G", 0.1)],
        )
        recs = mapper.recommend_next_skill(tree, max_recommendations=2)
        # weak 升序: kp-G(0.1), kp-F(0.2), kp-E(0.3) -> 取前 2
        assert recs == ["kp-G", "kp-F"]

    def test_recommend_empty_tree(self):
        """空技能树 -> 空推荐."""
        mapper = SkillMapper()
        assert mapper.recommend_next_skill(_make_tree([])) == []

    def test_recommend_all_mastered(self):
        """全部 mastered -> 无推荐."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.95)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        assert mapper.recommend_next_skill(tree) == []

    def test_recommend_returns_list_of_str(self):
        """推荐返回 kp_id 字符串列表."""
        mapper = SkillMapper()
        tree = _make_tree(
            [("kp-A", 0.9), ("kp-B", 0.5)],
            prereq_edges=[("kp-A", "kp-B")],
        )
        recs = mapper.recommend_next_skill(tree)
        assert isinstance(recs, list)
        assert all(isinstance(k, str) for k in recs)


# ============================================================
# 19. SkillMapper - compute_progress
# ============================================================


class TestSkillMapperComputeProgress:
    """SkillMapper.compute_progress 测试 — 技能树进度统计."""

    def test_compute_progress_keys(self):
        """返回字典含必要字段."""
        mapper = SkillMapper()
        tree = _make_tree([("kp-A", 0.5)])
        progress = mapper.compute_progress(tree)
        expected_keys = {
            "total_nodes", "mastered_count", "learning_count",
            "weak_count", "progress_percentage",
        }
        assert set(progress.keys()) >= expected_keys

    def test_compute_progress_empty_tree(self):
        """空技能树: 全部计数 0, 进度 0.0."""
        mapper = SkillMapper()
        progress = mapper.compute_progress(_make_tree([]))
        assert progress["total_nodes"] == 0
        assert progress["mastered_count"] == 0
        assert progress["learning_count"] == 0
        assert progress["weak_count"] == 0
        assert progress["progress_percentage"] == 0.0

    def test_compute_progress_counts(self):
        """正确统计各状态节点数."""
        mapper = SkillMapper()
        tree = _make_tree([
            ("kp-1", 0.0),    # not_started
            ("kp-2", 0.2),    # weak
            ("kp-3", 0.3),    # weak
            ("kp-4", 0.5),    # learning
            ("kp-5", 0.8),    # mastered
            ("kp-6", 0.9),    # mastered
        ])
        progress = mapper.compute_progress(tree)
        assert progress["total_nodes"] == 6
        assert progress["mastered_count"] == 2
        assert progress["learning_count"] == 1
        assert progress["weak_count"] == 2

    def test_compute_progress_percentage(self):
        """progress_percentage = mastered_count / total_nodes * 100."""
        mapper = SkillMapper()
        tree = _make_tree([
            ("kp-1", 0.8),  # mastered
            ("kp-2", 0.5),  # learning
            ("kp-3", 0.2),  # weak
            ("kp-4", 0.9),  # mastered
        ])
        progress = mapper.compute_progress(tree)
        # 2/4 * 100 = 50.0
        assert progress["progress_percentage"] == pytest.approx(50.0)

    def test_compute_progress_all_mastered(self):
        """全部 mastered -> 进度 100.0."""
        mapper = SkillMapper()
        tree = _make_tree([("kp-1", 0.9), ("kp-2", 0.8)])
        progress = mapper.compute_progress(tree)
        assert progress["mastered_count"] == 2
        assert progress["progress_percentage"] == pytest.approx(100.0)

    def test_compute_progress_none_mastered(self):
        """无 mastered -> 进度 0.0."""
        mapper = SkillMapper()
        tree = _make_tree([("kp-1", 0.5), ("kp-2", 0.2)])
        progress = mapper.compute_progress(tree)
        assert progress["mastered_count"] == 0
        assert progress["progress_percentage"] == 0.0

    def test_compute_progress_counts_are_int(self):
        """计数为 int 类型."""
        mapper = SkillMapper()
        tree = _make_tree([("kp-1", 0.8), ("kp-2", 0.5), ("kp-3", 0.2)])
        progress = mapper.compute_progress(tree)
        assert isinstance(progress["total_nodes"], int)
        assert isinstance(progress["mastered_count"], int)
        assert isinstance(progress["learning_count"], int)
        assert isinstance(progress["weak_count"], int)

    def test_compute_progress_percentage_is_float(self):
        """progress_percentage 为 float 类型."""
        mapper = SkillMapper()
        tree = _make_tree([("kp-1", 0.8), ("kp-2", 0.5), ("kp-3", 0.2)])
        progress = mapper.compute_progress(tree)
        assert isinstance(progress["progress_percentage"], float)

    def test_compute_progress_single_node_mastered(self):
        """单节点 mastered -> 进度 100.0."""
        mapper = SkillMapper()
        tree = _make_tree([("kp-1", 0.7)])
        progress = mapper.compute_progress(tree)
        assert progress["total_nodes"] == 1
        assert progress["mastered_count"] == 1
        assert progress["progress_percentage"] == pytest.approx(100.0)
