"""L2 知识追踪增强模块测试 — GNN 式知识图谱传播 / DKT 启发序列特征工程.

测试覆盖 (TDD):
1. MasteryPropagator.propagate_gnn: GNN 式注意力加权多层传播
   - 默认自注意力 (基于掌握度差异) / 显式 attention_weights / 多层 ReLU 聚合 / 环处理 / clamp
2. MasteryPropagator.propagate_attention: 注意力加权单跳传播
   - dot_product / additive 两种注意力函数 / softmax 归一化 / clamp
3. MasteryPropagator.propagate_heterogeneous: 异构图传播
   - prerequisite(0.3) / similarity(0.5) / complement(0.2) 不同边类型传播系数
4. SequenceFeatureExtractor: DKT 启发序列特征
   - 滑动窗口正确率 / 趋势检测 / 响应时间分析 / 连续对错 / 掌握度速度
5. TemporalPatternClassifier: 时序模式分类
   - steady_bloom / late_bloom / early_decay / oscillating / stable
6. 边界条件: 空输入 / 单条记录 / 全对 / 全错

对标方案: GKT (NeurIPS 2020) / AKT (KDD 2020) / DKT (NeurIPS 2015) / SAKT (EDMM 2019).
"""

from __future__ import annotations

import math

import pytest

from dy3_polaris.l2.knowledge_tracer import (
    MasteryPropagator,
    SequenceFeatureExtractor,
    SequenceFeatures,
    TemporalPatternClassifier,
)
from dy3_polaris.l2.models import AnswerRecord


# ============================================================
# 测试辅助函数
# ============================================================


def _softmax(scores: list[float]) -> list[float]:
    """数值稳定的 softmax (与实现一致, 用于测试期望值计算)."""
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def _make_records(
    corrects: list[bool],
    response_times: list[float] | None = None,
    kp: str = "kp-1",
) -> list[AnswerRecord]:
    """由正确性序列构造 AnswerRecord 列表 (时间戳升序).

    Args:
        corrects: 正确性序列.
        response_times: 可选响应时间序列 (秒); 不提供则不设置.
        kp: 知识点 ID.
    """
    recs: list[AnswerRecord] = []
    for i, c in enumerate(corrects):
        rt = response_times[i] if response_times is not None else None
        recs.append(
            AnswerRecord(
                learner_id="l1",
                kp_id=kp,
                correct=bool(c),
                timestamp=float(i),
                difficulty=0.5,
                response_time=rt,
            )
        )
    return recs


# ============================================================
# 1. MasteryPropagator - propagate_gnn (GNN 式传播)
# ============================================================


class TestMasteryPropagatorGNN:
    """MasteryPropagator.propagate_gnn 测试 — GNN 式注意力加权多层传播."""

    def test_gnn_no_neighbors_returns_base(self):
        """无邻居 -> 返回原掌握度 (无聚合)."""
        prop = MasteryPropagator()
        kg = {"C": []}
        result = prop.propagate_gnn("C", 0.5, kg, {}, max_depth=3)
        assert result == pytest.approx(0.5)

    def test_gnn_single_neighbor_default_attention(self):
        """单邻居默认注意力 (softmax 单值=1.0): boost = alpha * w * mastery (depth1)."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0)], "B": []}
        mastery_map = {"B": 0.7}
        # gnn_agg = ReLU(1.0 * 1.0 * 0.7) = 0.7; boosted = 0.5 + 0.3*0.7 = 0.71
        result = prop.propagate_gnn("C", 0.5, kg, mastery_map, max_depth=1)
        assert result == pytest.approx(0.71, abs=1e-9)

    def test_gnn_multi_layer_chain_uses_deeper_node(self):
        """链 C->B->A: depth1 用 B 的掌握度, depth2 经一层聚合用 A 的掌握度."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0)], "B": [("A", 1.0)], "A": []}
        mastery_map = {"B": 0.7, "A": 0.9}
        # depth1: gnn_agg = ReLU(1.0 * h_B^0) = 0.7 -> 0.5 + 0.3*0.7 = 0.71
        r1 = prop.propagate_gnn("C", 0.5, kg, mastery_map, max_depth=1)
        assert r1 == pytest.approx(0.71, abs=1e-9)
        # depth2: h_B^1 = ReLU(1.0 * h_A^0)=0.9; gnn_agg = ReLU(1.0*0.9)=0.9
        # boosted = 0.5 + 0.3*0.9 = 0.77
        r2 = prop.propagate_gnn("C", 0.5, kg, mastery_map, max_depth=2)
        assert r2 == pytest.approx(0.77, abs=1e-9)

    def test_gnn_attention_weights_softmax_aggregation(self):
        """显式 attention_weights: 对提供的得分做 softmax 加权聚合."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0), ("D", 1.0)], "B": [], "D": []}
        mastery_map = {"B": 0.8, "D": 0.4}
        attn = {("C", "B"): 2.0, ("C", "D"): 1.0}
        # scores=[2.0,1.0]; softmax -> α_B, α_D
        alpha_b, alpha_d = _softmax([2.0, 1.0])
        gnn_agg = alpha_b * 1.0 * 0.8 + alpha_d * 1.0 * 0.4  # ReLU 不改变正值
        expected = 0.5 + 0.3 * gnn_agg
        result = prop.propagate_gnn(
            "C", 0.5, kg, mastery_map, max_depth=1, attention_weights=attn
        )
        assert result == pytest.approx(expected, abs=1e-9)

    def test_gnn_default_attention_proportional_to_mastery(self):
        """默认注意力: 高掌握度邻居获得更高权重 (boost 高于等权)."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0), ("D", 1.0)], "B": [], "D": []}
        mastery_map = {"B": 0.9, "D": 0.1}
        # 默认注意力 softmax([0.9/1.0, 0.1/1.0]) -> 偏向 B(高掌握度)
        scores = [0.9 / 1.0, 0.1 / 1.0]
        alpha_b, alpha_d = _softmax(scores)
        gnn_agg = alpha_b * 0.9 + alpha_d * 0.1
        expected = 0.5 + 0.3 * gnn_agg
        result = prop.propagate_gnn("C", 0.5, kg, mastery_map, max_depth=1)
        assert result == pytest.approx(expected, abs=1e-9)
        # 高掌握度邻居权重大于低掌握度邻居
        assert alpha_b > alpha_d

    def test_gnn_zero_mastery_neighbors_no_boost(self):
        """邻居掌握度全为 0 -> 聚合为 0, 无提升."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0), ("D", 1.0)], "B": [], "D": []}
        mastery_map = {"B": 0.0, "D": 0.0}
        result = prop.propagate_gnn("C", 0.5, kg, mastery_map, max_depth=3)
        assert result == pytest.approx(0.5)

    def test_gnn_missing_mastery_treated_as_zero(self):
        """mastery_map 中缺失的邻居掌握度视为 0."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0)], "B": []}
        result = prop.propagate_gnn("C", 0.5, kg, {}, max_depth=3)
        assert result == pytest.approx(0.5)

    def test_gnn_clamp_upper_bound(self):
        """结果超过 1.0 时 clamp 到 1.0."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0), ("D", 1.0)], "B": [], "D": []}
        mastery_map = {"B": 1.0, "D": 1.0}
        result = prop.propagate_gnn("C", 0.95, kg, mastery_map, max_depth=3)
        assert result == pytest.approx(1.0)

    def test_gnn_result_in_unit_interval(self):
        """GNN 传播结果始终在 [0, 1]."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 0.9), ("D", 0.9)], "B": [("A", 0.9)], "A": [], "D": []}
        mastery_map = {"B": 0.9, "D": 0.9, "A": 0.9}
        for base in (0.0, 0.3, 0.7, 1.0):
            r = prop.propagate_gnn("C", base, kg, mastery_map, max_depth=3)
            assert 0.0 <= r <= 1.0

    def test_gnn_handles_cycle_without_infinite_loop(self):
        """图中存在环时不应无限循环 (路径去重)."""
        prop = MasteryPropagator()
        kg = {"A": [("B", 0.5)], "B": [("A", 0.5)]}
        mastery_map = {"A": 0.8, "B": 0.6}
        result = prop.propagate_gnn("A", 0.5, kg, mastery_map, max_depth=3)
        assert 0.0 <= result <= 1.0


# ============================================================
# 2. MasteryPropagator - propagate_attention (注意力单跳传播)
# ============================================================


class TestMasteryPropagatorAttention:
    """MasteryPropagator.propagate_attention 测试 — 注意力加权单跳传播."""

    def test_attention_no_prerequisites_returns_base(self):
        """无前置 -> 返回原掌握度."""
        prop = MasteryPropagator()
        result = prop.propagate_attention("C", 0.5, [], {})
        assert result == pytest.approx(0.5)

    def test_attention_dot_product_single(self):
        """dot_product 单前置: softmax 单值=1, boost = alpha * mastery."""
        prop = MasteryPropagator()
        # α=1.0; Σ α_i*m_i = 0.8; boosted = 0.5 + 0.3*0.8 = 0.74
        result = prop.propagate_attention(
            "C", 0.5, ["B"], {"B": 0.8}, attention_fn="dot_product"
        )
        assert result == pytest.approx(0.74, abs=1e-9)

    def test_attention_dot_product_multiple(self):
        """dot_product 多前置: α_i = softmax(mastery_i * current_mastery)."""
        prop = MasteryPropagator()
        cm = 0.5
        mastery_map = {"B": 0.8, "D": 0.4}
        scores = [0.8 * cm, 0.4 * cm]  # [0.4, 0.2]
        alpha_b, alpha_d = _softmax(scores)
        weighted = alpha_b * 0.8 + alpha_d * 0.4
        expected = cm + 0.3 * weighted
        result = prop.propagate_attention(
            "C", cm, ["B", "D"], mastery_map, attention_fn="dot_product"
        )
        assert result == pytest.approx(expected, abs=1e-9)

    def test_attention_dot_product_weights_sum_to_one(self):
        """等掌握度前置 -> 等权注意力 (各 α=1/n)."""
        prop = MasteryPropagator()
        mastery_map = {"B": 0.6, "D": 0.6}
        cm = 0.5
        # scores=[0.3,0.3] -> softmax 等权 [0.5,0.5]
        alpha_b, alpha_d = _softmax([0.6 * cm, 0.6 * cm])
        assert alpha_b == pytest.approx(0.5)
        assert alpha_d == pytest.approx(0.5)
        weighted = 0.5 * 0.6 + 0.5 * 0.6
        expected = cm + 0.3 * weighted
        result = prop.propagate_attention(
            "C", cm, ["B", "D"], mastery_map, attention_fn="dot_product"
        )
        assert result == pytest.approx(expected, abs=1e-9)

    def test_attention_additive_multiple(self):
        """additive: α_i = softmax(tanh(mastery_i + current_mastery)) 标量简化版."""
        prop = MasteryPropagator()
        cm = 0.5
        mastery_map = {"B": 0.8, "D": 0.4}
        scores = [math.tanh(0.8 + cm), math.tanh(0.4 + cm)]
        alpha_b, alpha_d = _softmax(scores)
        weighted = alpha_b * 0.8 + alpha_d * 0.4
        expected = cm + 0.3 * weighted
        result = prop.propagate_attention(
            "C", cm, ["B", "D"], mastery_map, attention_fn="additive"
        )
        assert result == pytest.approx(expected, abs=1e-9)

    def test_attention_additive_differs_from_dot_product(self):
        """additive 与 dot_product 在多前置时结果不同."""
        prop = MasteryPropagator()
        mastery_map = {"B": 0.8, "D": 0.4}
        r_dot = prop.propagate_attention(
            "C", 0.5, ["B", "D"], mastery_map, attention_fn="dot_product"
        )
        r_add = prop.propagate_attention(
            "C", 0.5, ["B", "D"], mastery_map, attention_fn="additive"
        )
        assert r_dot != pytest.approx(r_add, abs=1e-6)

    def test_attention_missing_mastery_treated_as_zero(self):
        """mastery_map 缺失的前置掌握度视为 0 (不贡献)."""
        prop = MasteryPropagator()
        result = prop.propagate_attention(
            "C", 0.5, ["B"], {}, attention_fn="dot_product"
        )
        assert result == pytest.approx(0.5)

    def test_attention_clamp_upper_bound(self):
        """结果超过 1.0 时 clamp 到 1.0."""
        prop = MasteryPropagator()
        result = prop.propagate_attention(
            "C", 0.95, ["B"], {"B": 1.0}, attention_fn="dot_product"
        )
        assert result == pytest.approx(1.0)

    def test_attention_default_fn_is_dot_product(self):
        """未指定 attention_fn 时默认使用 dot_product."""
        prop = MasteryPropagator()
        r_default = prop.propagate_attention("C", 0.5, ["B"], {"B": 0.8})
        r_dot = prop.propagate_attention(
            "C", 0.5, ["B"], {"B": 0.8}, attention_fn="dot_product"
        )
        assert r_default == pytest.approx(r_dot, abs=1e-12)


# ============================================================
# 3. MasteryPropagator - propagate_heterogeneous (异构图传播)
# ============================================================


class TestMasteryPropagatorHeterogeneous:
    """MasteryPropagator.propagate_heterogeneous 测试 — 异构边类型传播."""

    def test_heterogeneous_no_edges_returns_base(self):
        """无边 -> 返回原掌握度."""
        prop = MasteryPropagator()
        result = prop.propagate_heterogeneous("C", 0.5, {"C": []}, {}, {})
        assert result == pytest.approx(0.5)

    def test_heterogeneous_similarity_stronger_than_prerequisite(self):
        """similarity(0.5) 传播强于 prerequisite(0.3)."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0)], "B": []}
        mastery_map = {"B": 0.8}
        r_sim = prop.propagate_heterogeneous(
            "C", 0.5, kg, mastery_map, {"C->B": "similarity"}
        )
        r_pre = prop.propagate_heterogeneous(
            "C", 0.5, kg, mastery_map, {"C->B": "prerequisite"}
        )
        # sim: 0.5*1.0*0.8*0.5 = 0.2 -> 0.7
        assert r_sim == pytest.approx(0.7, abs=1e-9)
        # pre: 0.3*1.0*0.8*0.5 = 0.12 -> 0.62
        assert r_pre == pytest.approx(0.62, abs=1e-9)
        assert r_sim > r_pre

    def test_heterogeneous_complement_weakest(self):
        """complement(0.2) 传播最弱."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0)], "B": []}
        mastery_map = {"B": 0.8}
        r_comp = prop.propagate_heterogeneous(
            "C", 0.5, kg, mastery_map, {"C->B": "complement"}
        )
        # 0.2*1.0*0.8*0.5 = 0.08 -> 0.58
        assert r_comp == pytest.approx(0.58, abs=1e-9)

    def test_heterogeneous_default_type_is_prerequisite(self):
        """edge_types 未指定的边默认为 prerequisite(0.3)."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0)], "B": []}
        mastery_map = {"B": 0.8}
        r_default = prop.propagate_heterogeneous("C", 0.5, kg, mastery_map, {})
        r_pre = prop.propagate_heterogeneous(
            "C", 0.5, kg, mastery_map, {"C->B": "prerequisite"}
        )
        assert r_default == pytest.approx(r_pre, abs=1e-12)

    def test_heterogeneous_mixed_edge_types(self):
        """混合边类型: similarity + complement 同时存在."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0), ("D", 1.0)], "B": [], "D": []}
        mastery_map = {"B": 0.8, "D": 0.6}
        edge_types = {"C->B": "similarity", "C->D": "complement"}
        # B(sim): 0.5*1.0*0.8*0.5 = 0.2
        # D(comp): 0.2*1.0*0.6*0.5 = 0.06
        # boost = 0.26 -> 0.76
        result = prop.propagate_heterogeneous(
            "C", 0.5, kg, mastery_map, edge_types, max_depth=3
        )
        assert result == pytest.approx(0.76, abs=1e-9)

    def test_heterogeneous_multi_hop_uses_edge_type_alpha(self):
        """多跳传播: 每跳使用对应边类型的传播系数."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0)], "B": [("A", 1.0)], "A": []}
        mastery_map = {"B": 0.8, "A": 0.9}
        edge_types = {"C->B": "similarity", "B->A": "complement"}
        # depth1 B(sim): 0.5*1.0*0.8*0.5 = 0.2
        # depth2 A(comp): 0.2*(1.0*1.0)*0.9*0.25 = 0.045
        # boost = 0.245 -> 0.745
        result = prop.propagate_heterogeneous(
            "C", 0.5, kg, mastery_map, edge_types, max_depth=3
        )
        assert result == pytest.approx(0.745, abs=1e-9)

    def test_heterogeneous_result_in_unit_interval(self):
        """异构传播结果始终在 [0, 1]."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0), ("D", 1.0)], "B": [], "D": []}
        mastery_map = {"B": 1.0, "D": 1.0}
        edge_types = {"C->B": "similarity", "C->D": "similarity"}
        for base in (0.0, 0.3, 0.7, 1.0):
            r = prop.propagate_heterogeneous(
                "C", base, kg, mastery_map, edge_types, max_depth=3
            )
            assert 0.0 <= r <= 1.0


# ============================================================
# 4. SequenceFeatureExtractor - 滑动窗口 / 趋势 / 速度
# ============================================================


class TestSequenceSlidingWindowAndTrend:
    """SequenceFeatureExtractor 滑动窗口正确率与趋势检测测试."""

    def test_sliding_window_accuracy_basic(self):
        """滑动窗口正确率: window_size=3, 序列 [1,1,1,0,0,0]."""
        extractor = SequenceFeatureExtractor()
        records = _make_records([1, 1, 1, 0, 0, 0])
        sw = extractor._sliding_window_accuracy(records, window_size=3)
        # 窗口: [1,1,1]->1.0, [1,1,0]->2/3, [1,0,0]->1/3, [0,0,0]->0.0
        assert sw == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0], abs=1e-9)

    def test_sliding_window_size_affects_count(self):
        """window_size 越大, 窗口数越少."""
        extractor = SequenceFeatureExtractor()
        records = _make_records([1, 1, 0, 1, 0, 1])
        sw3 = extractor._sliding_window_accuracy(records, window_size=3)
        sw5 = extractor._sliding_window_accuracy(records, window_size=5)
        assert len(sw3) == 4  # 6 - 3 + 1
        assert len(sw5) == 2  # 6 - 5 + 1

    def test_sliding_window_fewer_than_window_size(self):
        """记录数少于 window_size -> 返回单元素整体正确率."""
        extractor = SequenceFeatureExtractor()
        records = _make_records([1, 0, 1])
        sw = extractor._sliding_window_accuracy(records, window_size=5)
        assert sw == pytest.approx([2 / 3], abs=1e-9)

    def test_sliding_window_empty(self):
        """空记录 -> 空列表."""
        extractor = SequenceFeatureExtractor()
        assert extractor._sliding_window_accuracy([], window_size=3) == []

    def test_detect_trend_improving(self):
        """上升序列 -> improving."""
        extractor = SequenceFeatureExtractor()
        assert extractor._detect_trend([0.0, 1 / 3, 2 / 3, 1.0]) == "improving"

    def test_detect_trend_declining(self):
        """下降序列 -> declining."""
        extractor = SequenceFeatureExtractor()
        assert extractor._detect_trend([1.0, 2 / 3, 1 / 3, 0.0]) == "declining"

    def test_detect_trend_stable(self):
        """平稳序列 -> stable."""
        extractor = SequenceFeatureExtractor()
        assert extractor._detect_trend([0.5, 0.5, 0.5, 0.5]) == "stable"

    def test_detect_trend_single_value_stable(self):
        """单值序列 -> stable (无法计算斜率)."""
        extractor = SequenceFeatureExtractor()
        assert extractor._detect_trend([0.7]) == "stable"

    def test_mastery_velocity_positive(self):
        """后半段正确率高于前半段 -> 正速度."""
        extractor = SequenceFeatureExtractor()
        records = _make_records([0, 0, 1, 1, 1, 1])
        vel = extractor._mastery_velocity(records)
        # first[0,0,1]=1/3, second[1,1,1]=1.0 -> 2/3
        assert vel == pytest.approx(2 / 3, abs=1e-9)

    def test_mastery_velocity_negative(self):
        """前半段正确率高于后半段 -> 负速度."""
        extractor = SequenceFeatureExtractor()
        records = _make_records([1, 1, 1, 0, 0, 1])
        vel = extractor._mastery_velocity(records)
        # first[1,1,1]=1.0, second[0,0,1]=1/3 -> -2/3
        assert vel == pytest.approx(-2 / 3, abs=1e-9)

    def test_mastery_velocity_single_record_zero(self):
        """单条记录 -> 速度 0."""
        extractor = SequenceFeatureExtractor()
        assert extractor._mastery_velocity(_make_records([1])) == pytest.approx(0.0)


# ============================================================
# 5. SequenceFeatureExtractor - 响应时间 / 连续对错
# ============================================================


class TestSequenceResponseTimeAndStreak:
    """SequenceFeatureExtractor 响应时间分析与连续对错分析测试."""

    def test_response_time_analysis_stats(self):
        """响应时间统计: mean / std / trend."""
        extractor = SequenceFeatureExtractor()
        records = _make_records(
            [1, 1, 1, 1], response_times=[10.0, 8.0, 6.0, 4.0]
        )
        stats = extractor._response_time_analysis(records)
        assert stats["mean"] == pytest.approx(7.0, abs=1e-9)
        # std = sqrt(((3)^2+(1)^2+(1)^2+(3)^2)/4) = sqrt(5)
        assert stats["std"] == pytest.approx(math.sqrt(5.0), abs=1e-9)
        assert stats["trend"] == "decreasing"

    def test_response_time_increasing_trend(self):
        """响应时间上升 -> trend=increasing (变慢)."""
        extractor = SequenceFeatureExtractor()
        records = _make_records(
            [1, 1, 1, 1], response_times=[4.0, 6.0, 8.0, 10.0]
        )
        stats = extractor._response_time_analysis(records)
        assert stats["trend"] == "increasing"

    def test_response_time_no_data(self):
        """无响应时间数据 -> mean=0, std=0, trend=stable."""
        extractor = SequenceFeatureExtractor()
        records = _make_records([1, 0, 1])  # 未设置 response_time
        stats = extractor._response_time_analysis(records)
        assert stats["mean"] == pytest.approx(0.0)
        assert stats["std"] == pytest.approx(0.0)
        assert stats["trend"] == "stable"

    def test_streak_all_correct(self):
        """全对: current_streak=n, max_correct=n, max_wrong=0."""
        extractor = SequenceFeatureExtractor()
        records = _make_records([1, 1, 1, 1])
        info = extractor._streak_analysis(records)
        assert info["current_streak"] == 4
        assert info["max_correct_streak"] == 4
        assert info["max_wrong_streak"] == 0
        assert info["current_type"] == "correct"

    def test_streak_all_wrong(self):
        """全错: current_streak=n, max_wrong=n, max_correct=0."""
        extractor = SequenceFeatureExtractor()
        records = _make_records([0, 0, 0])
        info = extractor._streak_analysis(records)
        assert info["current_streak"] == 3
        assert info["max_correct_streak"] == 0
        assert info["max_wrong_streak"] == 3
        assert info["current_type"] == "wrong"

    def test_streak_mixed(self):
        """混合序列: [1,1,0,0,1] -> current=1, max_correct=2, max_wrong=2."""
        extractor = SequenceFeatureExtractor()
        records = _make_records([1, 1, 0, 0, 1])
        info = extractor._streak_analysis(records)
        assert info["current_streak"] == 1
        assert info["max_correct_streak"] == 2
        assert info["max_wrong_streak"] == 2
        assert info["current_type"] == "correct"

    def test_streak_empty(self):
        """空记录 -> 全 0."""
        extractor = SequenceFeatureExtractor()
        info = extractor._streak_analysis([])
        assert info["current_streak"] == 0
        assert info["max_correct_streak"] == 0
        assert info["max_wrong_streak"] == 0


# ============================================================
# 6. SequenceFeatureExtractor - extract 完整提取
# ============================================================


class TestSequenceExtract:
    """SequenceFeatureExtractor.extract 完整序列特征提取测试."""

    def test_extract_returns_sequence_features_instance(self):
        """extract 返回 SequenceFeatures 实例."""
        extractor = SequenceFeatureExtractor()
        features = extractor.extract(_make_records([1, 0, 1, 1, 0, 1]))
        assert isinstance(features, SequenceFeatures)

    def test_extract_populates_all_fields(self):
        """extract 填充所有字段."""
        extractor = SequenceFeatureExtractor()
        features = extractor.extract(_make_records([1, 0, 1, 1, 0, 1]))
        assert isinstance(features.sliding_window_accuracy, list)
        assert features.recent_trend in {"improving", "declining", "stable"}
        assert isinstance(features.response_time_stats, dict)
        assert isinstance(features.streak_info, dict)
        assert isinstance(features.mastery_velocity, float)
        assert features.temporal_pattern in {
            "steady_bloom",
            "late_bloom",
            "early_decay",
            "oscillating",
            "stable",
        }

    def test_extract_empty_records(self):
        """空记录 -> 安全返回默认特征."""
        extractor = SequenceFeatureExtractor()
        features = extractor.extract([])
        assert features.sliding_window_accuracy == []
        assert features.recent_trend == "stable"
        assert features.mastery_velocity == pytest.approx(0.0)
        assert features.streak_info["current_streak"] == 0
        assert features.temporal_pattern == "stable"

    def test_extract_single_record(self):
        """单条记录 -> 安全返回 (不抛异常)."""
        extractor = SequenceFeatureExtractor()
        features = extractor.extract(_make_records([1]))
        assert features.streak_info["current_streak"] == 1
        assert features.streak_info["current_type"] == "correct"
        assert features.mastery_velocity == pytest.approx(0.0)
        assert features.temporal_pattern == "stable"

    def test_extract_all_correct(self):
        """全对序列 -> 高正确率, max_correct_streak=n."""
        extractor = SequenceFeatureExtractor()
        features = extractor.extract(_make_records([1] * 8))
        assert features.streak_info["max_correct_streak"] == 8
        assert all(a == pytest.approx(1.0) for a in features.sliding_window_accuracy)
        assert features.temporal_pattern == "stable"

    def test_extract_all_wrong(self):
        """全错序列 -> 零正确率, max_wrong_streak=n."""
        extractor = SequenceFeatureExtractor()
        features = extractor.extract(_make_records([0] * 8))
        assert features.streak_info["max_wrong_streak"] == 8
        assert all(a == pytest.approx(0.0) for a in features.sliding_window_accuracy)
        assert features.temporal_pattern == "stable"

    def test_extract_sorts_by_timestamp(self):
        """乱序时间戳输入按时间排序后处理 (近期趋势基于时序)."""
        extractor = SequenceFeatureExtractor()
        # 故意打乱: 第 0 条时间戳最大
        records = [
            AnswerRecord("l1", "k", True, timestamp=300.0, response_time=5.0),
            AnswerRecord("l1", "k", False, timestamp=100.0, response_time=10.0),
            AnswerRecord("l1", "k", True, timestamp=200.0, response_time=7.0),
        ]
        features = extractor.extract(records, window_size=2)
        # 排序后正确性序列: [0(100), 1(200), 1(300)] -> 滑窗(2): [0.5, 1.0]
        assert features.sliding_window_accuracy == pytest.approx([0.5, 1.0], abs=1e-9)
        assert features.recent_trend == "improving"


# ============================================================
# 7. TemporalPatternClassifier - 时序模式分类
# ============================================================


class TestTemporalPatternClassifier:
    """TemporalPatternClassifier 测试 — 时序模式分类."""

    def test_pattern_steady_bloom(self):
        """稳步提升 (斜率>0.02, 未跨 0.5/0.7 阈值) -> steady_bloom."""
        clf = TemporalPatternClassifier()
        # [0,1,1,1,1,1,1,1,1,1]: first_half=0.6, second=1.0, slope>0.02
        records = _make_records([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert clf.classify(records) == "steady_bloom"

    def test_pattern_late_bloom(self):
        """后期爆发 (前半<0.5, 后半>0.7) -> late_bloom."""
        clf = TemporalPatternClassifier()
        records = _make_records([0, 0, 0, 0, 1, 1, 1, 1])
        assert clf.classify(records) == "late_bloom"

    def test_pattern_early_decay(self):
        """早期衰退 (前半>0.7, 后半<0.5) -> early_decay."""
        clf = TemporalPatternClassifier()
        records = _make_records([1, 1, 1, 1, 0, 0, 0, 0])
        assert clf.classify(records) == "early_decay"

    def test_pattern_oscillating(self):
        """震荡 (标准差>0.3) -> oscillating."""
        clf = TemporalPatternClassifier()
        records = _make_records([1, 0, 1, 0, 1, 0, 1, 0])
        assert clf.classify(records) == "oscillating"

    def test_pattern_stable_all_correct(self):
        """全对 (标准差<0.1) -> stable."""
        clf = TemporalPatternClassifier()
        records = _make_records([1, 1, 1, 1, 1, 1])
        assert clf.classify(records) == "stable"

    def test_pattern_stable_all_wrong(self):
        """全错 (标准差<0.1) -> stable."""
        clf = TemporalPatternClassifier()
        records = _make_records([0, 0, 0, 0, 0, 0])
        assert clf.classify(records) == "stable"

    def test_pattern_empty_records(self):
        """空记录 -> stable."""
        clf = TemporalPatternClassifier()
        assert clf.classify([]) == "stable"

    def test_pattern_single_record(self):
        """单条记录 -> stable."""
        clf = TemporalPatternClassifier()
        assert clf.classify(_make_records([1])) == "stable"

    def test_extract_temporal_pattern_matches_classifier(self):
        """extract 的 temporal_pattern 与 TemporalPatternClassifier 一致."""
        extractor = SequenceFeatureExtractor()
        clf = TemporalPatternClassifier()
        records = _make_records([0, 0, 0, 0, 1, 1, 1, 1])
        features = extractor.extract(records)
        assert features.temporal_pattern == clf.classify(records)
