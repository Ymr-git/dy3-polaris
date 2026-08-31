"""L2 知识追踪子模块测试 — BKTTracer / MasteryPropagator / ForgettingModel.

测试覆盖 (TDD):
1. BKTTracer:
   - init_state(kp_id, difficulty): 难度 -> p_l0 线性映射 (0.0->0.7, 1.0->0.3)
   - update(state, correct, timestamp): BKT 前向算法 (后验更新 + 转移), 计数更新
   - batch_trace(records): 按时间戳排序后逐条更新, 多 kp 隔离
   - predict_correct_prob(state): P(C) = P(L)*(1-S) + (1-P(L))*G
2. MasteryPropagator:
   - propagate(kp_id, mastery, prerequisites): KG 驱动掌握度传播, alpha=0.3, clamp [0,1]
3. ForgettingModel:
   - decay(mastery, delta_t_hours, stability): 艾宾浩斯遗忘曲线, 仅 delta_t>168 触发衰减
   - should_review(state, current_time, threshold): 衰减后掌握度 < 阈值则需复习
"""

import math

import pytest

from dy3_polaris.l2.knowledge_tracer import (
    BKTTracer,
    ForgettingModel,
    MasteryPropagator,
)
from dy3_polaris.l2.models import (
    DEFAULT_BKT_PARAMS,
    AnswerRecord,
    TracingState,
)


# ============================================================
# 1. BKTTracer - init_state
# ============================================================


class TestBKTTracerInitState:
    """BKTTracer.init_state 测试 — 难度到先验掌握概率 p_l0 的线性映射."""

    def test_init_state_easy_difficulty(self):
        """难度 0.0 (最易) -> p_l0 = 0.7 (高先验掌握概率)."""
        tracer = BKTTracer()
        state = tracer.init_state("kp-001", 0.0)
        assert state.kp_id == "kp-001"
        assert state.mastery_prob == pytest.approx(0.7)
        assert state.attempts == 0
        assert state.correct_count == 0
        assert state.last_attempt_time == 0.0

    def test_init_state_hard_difficulty(self):
        """难度 1.0 (最难) -> p_l0 = 0.3 (低先验掌握概率)."""
        tracer = BKTTracer()
        state = tracer.init_state("kp-001", 1.0)
        assert state.mastery_prob == pytest.approx(0.3)

    def test_init_state_medium_difficulty(self):
        """难度 0.5 (中等) -> p_l0 = 0.5."""
        tracer = BKTTracer()
        state = tracer.init_state("kp-001", 0.5)
        assert state.mastery_prob == pytest.approx(0.5)

    def test_init_state_linear_mapping(self):
        """难度到 p_l0 为线性映射: p_l0 = 0.7 - 0.4 * difficulty."""
        tracer = BKTTracer()
        for difficulty in (0.0, 0.25, 0.5, 0.75, 1.0):
            expected = 0.7 - 0.4 * difficulty
            state = tracer.init_state("kp", difficulty)
            assert state.mastery_prob == pytest.approx(expected)

    def test_init_state_bkt_params_contains_p_l0(self):
        """init_state 后 bkt_params['p_l0'] 等于映射后的先验."""
        tracer = BKTTracer()
        state = tracer.init_state("kp-001", 0.25)
        assert state.bkt_params["p_l0"] == pytest.approx(0.6)

    def test_init_state_bkt_params_has_four_keys(self):
        """init_state 后 bkt_params 含 BKT 四参数 p_l0/p_t/p_g/p_s."""
        tracer = BKTTracer()
        state = tracer.init_state("kp-001", 0.5)
        assert set(state.bkt_params.keys()) >= {"p_l0", "p_t", "p_g", "p_s"}
        assert state.bkt_params["p_t"] == DEFAULT_BKT_PARAMS["p_t"]
        assert state.bkt_params["p_g"] == DEFAULT_BKT_PARAMS["p_g"]
        assert state.bkt_params["p_s"] == DEFAULT_BKT_PARAMS["p_s"]


# ============================================================
# 2. BKTTracer - update
# ============================================================


class TestBKTTracerUpdate:
    """BKTTracer.update 测试 — BKT 前向算法 (后验更新 + 转移)."""

    def _make_state(self, mastery=0.5) -> TracingState:
        """构造默认参数的 TracingState (p_l0/p_t/p_g/p_s = 0.5/0.1/0.2/0.1)."""
        return TracingState(
            kp_id="kp-001",
            mastery_prob=mastery,
            bkt_params=dict(DEFAULT_BKT_PARAMS),
        )

    def test_update_correct_increases_mastery(self):
        """答对后掌握概率应上升."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.5)
        new_state = tracer.update(state, correct=True, timestamp=100.0)
        assert new_state.mastery_prob > 0.5
        # 精确值: 9/11 + (2/11)*0.1 = 46/55 ≈ 0.83636
        assert new_state.mastery_prob == pytest.approx(46 / 55, abs=1e-9)

    def test_update_wrong_decreases_mastery(self):
        """答错后掌握概率应下降."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.5)
        new_state = tracer.update(state, correct=False, timestamp=100.0)
        assert new_state.mastery_prob < 0.5
        # 精确值: 1/9 + (8/9)*0.1 = 0.2
        assert new_state.mastery_prob == pytest.approx(0.2, abs=1e-9)

    def test_update_correct_posterior_formula(self):
        """答对后验: P(L|correct) = P(L)*(1-S) / (P(L)*(1-S) + (1-P(L))*G).

        然后转移: P(L)' = P(L|correct) + (1-P(L|correct))*T.
        """
        tracer = BKTTracer()
        # 自定义参数: p_s=0.2, p_g=0.3, p_t=0.15, mastery=0.6
        state = TracingState(
            kp_id="kp",
            mastery_prob=0.6,
            bkt_params={"p_l0": 0.5, "p_t": 0.15, "p_g": 0.3, "p_s": 0.2},
        )
        new_state = tracer.update(state, correct=True, timestamp=1.0)
        # posterior = 0.6*0.8 / (0.6*0.8 + 0.4*0.3) = 0.48 / 0.6 = 0.8
        # transit    = 0.8 + 0.2*0.15 = 0.83
        assert new_state.mastery_prob == pytest.approx(0.83, abs=1e-9)

    def test_update_wrong_posterior_formula(self):
        """答错后验: P(L|wrong) = P(L)*S / (P(L)*S + (1-P(L))*(1-G))."""
        tracer = BKTTracer()
        state = TracingState(
            kp_id="kp",
            mastery_prob=0.6,
            bkt_params={"p_l0": 0.5, "p_t": 0.15, "p_g": 0.3, "p_s": 0.2},
        )
        new_state = tracer.update(state, correct=False, timestamp=1.0)
        # posterior = 0.6*0.2 / (0.6*0.2 + 0.4*0.7) = 0.12 / 0.4 = 0.3
        # transit    = 0.3 + 0.7*0.15 = 0.405
        assert new_state.mastery_prob == pytest.approx(0.405, abs=1e-9)

    def test_update_increments_attempts(self):
        """每次 update 使 attempts +1."""
        tracer = BKTTracer()
        state = self._make_state()
        state.attempts = 3
        new_state = tracer.update(state, correct=True, timestamp=1.0)
        assert new_state.attempts == 4

    def test_update_correct_increments_correct_count(self):
        """答对时 correct_count +1."""
        tracer = BKTTracer()
        state = self._make_state()
        state.correct_count = 2
        new_state = tracer.update(state, correct=True, timestamp=1.0)
        assert new_state.correct_count == 3

    def test_update_wrong_does_not_increment_correct_count(self):
        """答错时 correct_count 不变."""
        tracer = BKTTracer()
        state = self._make_state()
        state.correct_count = 2
        new_state = tracer.update(state, correct=False, timestamp=1.0)
        assert new_state.correct_count == 2

    def test_update_sets_last_attempt_time(self):
        """update 后 last_attempt_time = timestamp."""
        tracer = BKTTracer()
        state = self._make_state()
        new_state = tracer.update(state, correct=True, timestamp=999.0)
        assert new_state.last_attempt_time == 999.0

    def test_update_preserves_kp_id(self):
        """update 后 kp_id 保持不变."""
        tracer = BKTTracer()
        state = self._make_state()
        new_state = tracer.update(state, correct=True, timestamp=1.0)
        assert new_state.kp_id == "kp-001"

    def test_update_preserves_bkt_params(self):
        """update 不修改 bkt_params (p_l0 保持先验, 不随观测更新)."""
        tracer = BKTTracer()
        state = self._make_state()
        original_params = dict(state.bkt_params)
        new_state = tracer.update(state, correct=True, timestamp=1.0)
        assert new_state.bkt_params == original_params

    def test_update_is_functional_does_not_mutate_input(self):
        """update 返回新对象, 不修改原 state (函数式风格)."""
        tracer = BKTTracer()
        state = self._make_state()
        original_mastery = state.mastery_prob
        original_attempts = state.attempts
        tracer.update(state, correct=True, timestamp=1.0)
        assert state.mastery_prob == original_mastery
        assert state.attempts == original_attempts

    def test_update_mastery_clamped_to_unit_interval(self):
        """update 后 mastery_prob 始终落在 [0, 1] (数值稳定)."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.99)
        new_state = tracer.update(state, correct=True, timestamp=1.0)
        assert 0.0 <= new_state.mastery_prob <= 1.0
        state2 = self._make_state(mastery=0.01)
        new_state2 = tracer.update(state2, correct=False, timestamp=1.0)
        assert 0.0 <= new_state2.mastery_prob <= 1.0

    def test_update_consecutive_correct_approaches_one(self):
        """连续答对使掌握概率趋近 1.0."""
        tracer = BKTTracer()
        state = tracer.init_state("kp", 0.5)
        for _ in range(20):
            state = tracer.update(state, correct=True, timestamp=1.0)
        assert state.mastery_prob > 0.99


# ============================================================
# 3. BKTTracer - batch_trace
# ============================================================


class TestBKTTracerBatchTrace:
    """BKTTracer.batch_trace 测试 — 历史重建 (按时间戳排序逐条更新)."""

    def test_batch_trace_empty_returns_empty_dict(self):
        """空记录列表返回空字典."""
        tracer = BKTTracer()
        result = tracer.batch_trace([])
        assert result == {}

    def test_batch_trace_single_record(self):
        """单条记录: 返回单 kp 状态."""
        tracer = BKTTracer()
        records = [
            AnswerRecord(
                learner_id="l1", kp_id="kp-1", correct=True,
                timestamp=100.0, difficulty=0.5,
            ),
        ]
        result = tracer.batch_trace(records)
        assert "kp-1" in result
        assert result["kp-1"].attempts == 1
        assert result["kp-1"].correct_count == 1
        assert result["kp-1"].mastery_prob == pytest.approx(46 / 55, abs=1e-9)

    def test_batch_trace_sorts_by_timestamp(self):
        """乱序输入按时间戳排序后逐条更新 (先早后晚)."""
        tracer = BKTTracer()
        # 故意打乱时间顺序
        records = [
            AnswerRecord("l1", "kp-1", correct=True, timestamp=200.0, difficulty=0.5),
            AnswerRecord("l1", "kp-1", correct=True, timestamp=100.0, difficulty=0.5),
        ]
        result = tracer.batch_trace(records)
        state = result["kp-1"]
        # 两次答对: 0.5 -> 46/55 -> 77/80 = 0.9625
        assert state.mastery_prob == pytest.approx(0.9625, abs=1e-9)
        assert state.attempts == 2
        assert state.correct_count == 2
        assert state.last_attempt_time == 200.0

    def test_batch_trace_multiple_kp_isolated(self):
        """多 kp 记录按各自 kp_id 独立追踪."""
        tracer = BKTTracer()
        records = [
            AnswerRecord("l1", "kp-1", correct=True, timestamp=200.0, difficulty=0.5),
            AnswerRecord("l1", "kp-2", correct=False, timestamp=150.0, difficulty=0.5),
            AnswerRecord("l1", "kp-1", correct=True, timestamp=100.0, difficulty=0.5),
        ]
        result = tracer.batch_trace(records)
        assert set(result.keys()) == {"kp-1", "kp-2"}
        # kp-1: 两次答对 -> 0.9625
        assert result["kp-1"].mastery_prob == pytest.approx(0.9625, abs=1e-9)
        assert result["kp-1"].attempts == 2
        # kp-2: 一次答错 (从 0.5) -> 0.2
        assert result["kp-2"].mastery_prob == pytest.approx(0.2, abs=1e-9)
        assert result["kp-2"].attempts == 1
        assert result["kp-2"].correct_count == 0

    def test_batch_trace_returns_tracing_state_instances(self):
        """batch_trace 返回值为 TracingState 实例."""
        tracer = BKTTracer()
        records = [AnswerRecord("l1", "kp-1", correct=True, timestamp=1.0)]
        result = tracer.batch_trace(records)
        assert isinstance(result["kp-1"], TracingState)

    def test_batch_trace_uses_difficulty_for_prior(self):
        """新 kp 使用首条记录的 difficulty 初始化先验 p_l0."""
        tracer = BKTTracer()
        # difficulty=0.0 -> p_l0=0.7, 一次答错
        records = [
            AnswerRecord("l1", "kp-1", correct=False, timestamp=1.0, difficulty=0.0),
        ]
        result = tracer.batch_trace(records)
        # 从 0.7 答错: posterior = 0.7*0.1/(0.7*0.1+0.3*0.8) = 0.07/0.31
        # transit = 0.07/0.31 + (1-0.07/0.31)*0.1
        posterior = 0.7 * 0.1 / (0.7 * 0.1 + 0.3 * 0.8)
        expected = posterior + (1 - posterior) * 0.1
        assert result["kp-1"].mastery_prob == pytest.approx(expected, abs=1e-9)
        assert result["kp-1"].bkt_params["p_l0"] == pytest.approx(0.7)


# ============================================================
# 4. BKTTracer - predict_correct_prob
# ============================================================


class TestBKTTracerPredict:
    """BKTTracer.predict_correct_prob 测试 — 预测下次正确率."""

    def test_predict_formula_default(self):
        """P(C) = P(L)*(1-S) + (1-P(L))*G (默认参数)."""
        tracer = BKTTracer()
        state = TracingState(
            kp_id="kp",
            mastery_prob=0.5,
            bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1},
        )
        # 0.5*0.9 + 0.5*0.2 = 0.55
        assert tracer.predict_correct_prob(state) == pytest.approx(0.55, abs=1e-9)

    def test_predict_high_mastery(self):
        """高掌握度 -> 高预测正确率 (接近 1-S)."""
        tracer = BKTTracer()
        state = TracingState(
            kp_id="kp",
            mastery_prob=0.9,
            bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1},
        )
        # 0.9*0.9 + 0.1*0.2 = 0.81 + 0.02 = 0.83
        assert tracer.predict_correct_prob(state) == pytest.approx(0.83, abs=1e-9)

    def test_predict_low_mastery(self):
        """低掌握度 -> 预测正确率接近猜测概率 G."""
        tracer = BKTTracer()
        state = TracingState(
            kp_id="kp",
            mastery_prob=0.0,
            bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.25, "p_s": 0.1},
        )
        # mastery=0 -> P(C) = G = 0.25
        assert tracer.predict_correct_prob(state) == pytest.approx(0.25, abs=1e-9)

    def test_predict_full_mastery(self):
        """完全掌握 (mastery=1) -> P(C) = 1 - S."""
        tracer = BKTTracer()
        state = TracingState(
            kp_id="kp",
            mastery_prob=1.0,
            bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.15},
        )
        # 1.0*0.85 + 0.0*0.2 = 0.85
        assert tracer.predict_correct_prob(state) == pytest.approx(0.85, abs=1e-9)

    def test_predict_in_unit_interval(self):
        """预测正确率始终在 [0, 1] 内."""
        tracer = BKTTracer()
        for mastery in (0.0, 0.25, 0.5, 0.75, 1.0):
            state = TracingState(
                kp_id="kp",
                mastery_prob=mastery,
                bkt_params=dict(DEFAULT_BKT_PARAMS),
            )
            p = tracer.predict_correct_prob(state)
            assert 0.0 <= p <= 1.0


# ============================================================
# 5. MasteryPropagator
# ============================================================


class TestMasteryPropagator:
    """MasteryPropagator 测试 — KG 驱动掌握度传播."""

    def test_propagate_no_prerequisites(self):
        """无前置知识点 -> 返回原掌握度 (clamp 后)."""
        prop = MasteryPropagator()
        result = prop.propagate("kp-B", 0.5, [])
        assert result == pytest.approx(0.5)

    def test_propagate_single_prerequisite(self):
        """单个前置: boosted = base + alpha * mastery (alpha=0.3)."""
        prop = MasteryPropagator()
        # 0.5 + 0.3 * 0.8 = 0.74
        result = prop.propagate("kp-B", 0.5, [("kp-A", 0.8)])
        assert result == pytest.approx(0.74, abs=1e-9)

    def test_propagate_multiple_prerequisites(self):
        """多个前置: boosted = base + alpha * sum(mastery)."""
        prop = MasteryPropagator()
        # 0.5 + 0.3 * (0.8 + 0.6) = 0.5 + 0.42 = 0.92
        result = prop.propagate("kp-B", 0.5, [("kp-A", 0.8), ("kp-C", 0.6)])
        assert result == pytest.approx(0.92, abs=1e-9)

    def test_propagate_alpha_is_0_3(self):
        """传播系数 alpha = 0.3."""
        prop = MasteryPropagator()
        assert prop.alpha == pytest.approx(0.3)

    def test_propagate_clamp_upper_bound(self):
        """结果超过 1.0 时 clamp 到 1.0."""
        prop = MasteryPropagator()
        # 0.9 + 0.3 * 0.9 = 1.17 -> clamp 1.0
        result = prop.propagate("kp-B", 0.9, [("kp-A", 0.9)])
        assert result == pytest.approx(1.0)

    def test_propagate_clamp_lower_bound(self):
        """结果不小于 0.0 (低掌握度无前置)."""
        prop = MasteryPropagator()
        result = prop.propagate("kp-B", 0.0, [])
        assert result == pytest.approx(0.0)

    def test_propagate_zero_mastery_prereq_no_boost(self):
        """前置掌握度为 0 时不增加掌握度."""
        prop = MasteryPropagator()
        result = prop.propagate("kp-B", 0.5, [("kp-A", 0.0)])
        assert result == pytest.approx(0.5)

    def test_propagate_formula_matches_spec(self):
        """公式: P(L0_B) = base + alpha * sum(prereq_mastery)."""
        prop = MasteryPropagator()
        base = 0.4
        prereqs = [("p1", 0.7), ("p2", 0.5), ("p3", 0.2)]
        expected = base + 0.3 * sum(m for _, m in prereqs)
        assert prop.propagate("kp-B", base, prereqs) == pytest.approx(
            expected, abs=1e-9
        )

    def test_propagate_ignores_kp_id_in_prereq_tuple(self):
        """propagate 使用前置元组的掌握度, kp_id 仅作标识."""
        prop = MasteryPropagator()
        r1 = prop.propagate("kp-B", 0.5, [("any-id", 0.8)])
        r2 = prop.propagate("kp-B", 0.5, [("other-id", 0.8)])
        assert r1 == pytest.approx(r2)


# ============================================================
# 6. ForgettingModel - decay
# ============================================================


class TestForgettingModelDecay:
    """ForgettingModel.decay 测试 — 艾宾浩斯遗忘曲线."""

    def test_decay_no_decay_within_week(self):
        """delta_t <= 168 小时 (7 天) 不衰减, 返回原值."""
        model = ForgettingModel()
        assert model.decay(0.8, 0.0) == pytest.approx(0.8)
        assert model.decay(0.8, 100.0) == pytest.approx(0.8)
        assert model.decay(0.8, 168.0) == pytest.approx(0.8)

    def test_decay_triggers_after_week(self):
        """delta_t > 168 小时触发衰减, 掌握度下降."""
        model = ForgettingModel()
        decayed = model.decay(0.8, 169.0)
        # lambda=0.007, m = 0.8 * exp(-0.007 * 1)
        expected = 0.8 * math.exp(-0.007 * 1)
        assert decayed == pytest.approx(expected, abs=1e-9)
        assert decayed < 0.8

    def test_decay_formula(self):
        """公式: m(t) = mastery * exp(-lambda * (delta_t - 168))."""
        model = ForgettingModel()
        mastery = 0.9
        delta_t = 336.0  # 14 天, 超出 168 小时
        expected = mastery * math.exp(-0.007 * (336.0 - 168.0))
        assert model.decay(mastery, delta_t) == pytest.approx(expected, abs=1e-9)

    def test_decay_lambda_base_rate(self):
        """基础遗忘率 lambda = 0.007 (stability=1.0)."""
        model = ForgettingModel()
        assert model.base_lambda == pytest.approx(0.007)

    def test_decay_stability_scales_lambda(self):
        """stability 越大遗忘越慢: lambda = 0.007 / stability."""
        model = ForgettingModel()
        mastery = 0.8
        delta_t = 336.0
        # stability=2.0 -> lambda=0.0035
        expected = mastery * math.exp(-(0.007 / 2.0) * (delta_t - 168.0))
        assert model.decay(mastery, delta_t, stability=2.0) == pytest.approx(
            expected, abs=1e-9
        )

    def test_decay_higher_stability_slower_decay(self):
        """stability 越大, 同时间衰减后掌握度越高."""
        model = ForgettingModel()
        delta_t = 500.0
        low = model.decay(0.8, delta_t, stability=1.0)
        high = model.decay(0.8, delta_t, stability=5.0)
        assert high > low

    def test_decay_boundary_exactly_168(self):
        """边界 delta_t == 168 不触发衰减 (仅 > 168 触发)."""
        model = ForgettingModel()
        assert model.decay(0.7, 168.0) == pytest.approx(0.7)

    def test_decay_boundary_just_over_168(self):
        """delta_t 略大于 168 触发轻微衰减."""
        model = ForgettingModel()
        decayed = model.decay(0.7, 168.0 + 1e-6)
        assert decayed < 0.7

    def test_decay_negative_delta_returns_original(self):
        """负 delta_t (时间倒流) 视为无衰减."""
        model = ForgettingModel()
        assert model.decay(0.8, -10.0) == pytest.approx(0.8)

    def test_decay_result_in_unit_interval(self):
        """衰减后掌握度始终在 [0, 1] (不超过原值)."""
        model = ForgettingModel()
        for delta_t in (0.0, 168.0, 500.0, 2000.0):
            decayed = model.decay(0.8, delta_t)
            assert 0.0 <= decayed <= 0.8


# ============================================================
# 7. ForgettingModel - should_review
# ============================================================


class TestForgettingModelShouldReview:
    """ForgettingModel.should_review 测试 — 是否需要复习."""

    def _make_state(self, mastery=0.8, last=0.0) -> TracingState:
        return TracingState(
            kp_id="kp-001",
            mastery_prob=mastery,
            last_attempt_time=last,
            bkt_params=dict(DEFAULT_BKT_PARAMS),
        )

    def test_should_review_false_when_recent_and_high_mastery(self):
        """近期作答且掌握度高 -> 不需要复习."""
        model = ForgettingModel()
        state = self._make_state(mastery=0.8, last=0.0)
        # current_time = 100s (远小于 168 小时), 无衰减 -> 0.8 >= 0.5
        assert model.should_review(state, current_time=100.0) is False

    def test_should_review_true_when_low_mastery(self):
        """掌握度低于阈值 (即使无衰减) -> 需要复习."""
        model = ForgettingModel()
        state = self._make_state(mastery=0.3, last=0.0)
        # 无衰减 -> 0.3 < 0.5
        assert model.should_review(state, current_time=100.0) is True

    def test_should_review_true_after_long_gap(self):
        """长时间未复习, 衰减后掌握度低于阈值 -> 需要复习."""
        model = ForgettingModel()
        state = self._make_state(mastery=0.3, last=0.0)
        # 500 小时后: 0.3 * exp(-0.007 * 332) ≈ 0.029 < 0.5
        current_time = 500.0 * 3600.0  # 秒
        assert model.should_review(state, current_time=current_time) is True

    def test_should_review_false_when_high_mastery_even_after_decay(self):
        """高掌握度短时间衰减后仍高于阈值 -> 不需要复习.

        should_review 使用 compute_retention (连续衰减, 无 168h 硬阈值):
        50 小时: 0.8 * exp(-0.007 * 50) ≈ 0.564 >= 0.5 -> 不复习.
        """
        model = ForgettingModel()
        state = self._make_state(mastery=0.8, last=0.0)
        # 50 小时 (连续衰减): 0.8 * exp(-0.007 * 50) ≈ 0.564 >= 0.5
        current_time = 50.0 * 3600.0
        assert model.should_review(state, current_time=current_time) is False

    def test_should_review_uses_threshold(self):
        """自定义 threshold 改变判定结果."""
        model = ForgettingModel()
        state = self._make_state(mastery=0.6, last=0.0)
        # 无衰减 -> 0.6; threshold=0.5 -> False, threshold=0.7 -> True
        assert model.should_review(state, current_time=10.0, threshold=0.5) is False
        assert model.should_review(state, current_time=10.0, threshold=0.7) is True

    def test_should_review_default_threshold(self):
        """默认 threshold = 0.5."""
        model = ForgettingModel()
        assert model.default_threshold == pytest.approx(0.5)

    def test_should_review_uses_time_delta(self):
        """should_review 基于时间差 (current_time - last_attempt_time) 判定."""
        model = ForgettingModel()
        # last=1000s, current=1000+500h*3600, 与 last=0,current=500h 等价
        state = self._make_state(mastery=0.3, last=1000.0)
        current = 1000.0 + 500.0 * 3600.0
        assert model.should_review(state, current_time=current) is True

    def test_should_review_uses_state_mastery_and_time(self):
        """should_review 综合使用 state.mastery_prob 与 state.last_attempt_time."""
        model = ForgettingModel()
        # 两个状态: 相同时间差, 不同初始掌握度
        high = self._make_state(mastery=0.9, last=0.0)
        low = self._make_state(mastery=0.3, last=0.0)
        current = 1000.0 * 3600.0  # 1000 小时
        assert model.should_review(high, current) is True
        assert model.should_review(low, current) is True


# ============================================================
# 8. BKTTracer - validate_params (参数约束校验)
# ============================================================


class TestBKTTracerValidateParams:
    """BKTTracer.validate_params 测试 — BKT 参数约束 p_g + p_s < 1."""

    def test_validate_params_valid_no_raise(self):
        """合法参数 (p_g+p_s<1, 各参数在 [0,1]) 不抛异常."""
        tracer = BKTTracer()
        # p_g=0.2, p_s=0.1 -> 0.3 < 1
        assert tracer.validate_params(
            {"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1}
        ) is True

    def test_validate_params_raises_when_pg_plus_ps_ge_one(self):
        """p_g + p_s >= 1 时抛 ValueError."""
        tracer = BKTTracer()
        # p_g=0.6, p_s=0.5 -> 1.1 >= 1
        with pytest.raises(ValueError):
            tracer.validate_params(
                {"p_l0": 0.5, "p_t": 0.1, "p_g": 0.6, "p_s": 0.5}
            )

    def test_validate_params_raises_when_pg_plus_ps_equals_one(self):
        """p_g + p_s == 1 (边界) 也视为非法 (要求严格 < 1)."""
        tracer = BKTTracer()
        with pytest.raises(ValueError):
            tracer.validate_params(
                {"p_l0": 0.5, "p_t": 0.1, "p_g": 0.5, "p_s": 0.5}
            )

    def test_validate_params_raises_when_param_out_of_range(self):
        """任一参数超出 [0, 1] 抛 ValueError."""
        tracer = BKTTracer()
        with pytest.raises(ValueError):
            tracer.validate_params(
                {"p_l0": 1.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1}
            )
        with pytest.raises(ValueError):
            tracer.validate_params(
                {"p_l0": 0.5, "p_t": -0.1, "p_g": 0.2, "p_s": 0.1}
            )

    def test_validate_params_default_bkt_params_valid(self):
        """DEFAULT_BKT_PARAMS 应通过校验."""
        tracer = BKTTracer()
        assert tracer.validate_params(dict(DEFAULT_BKT_PARAMS)) is True


# ============================================================
# 9. BKTTracer - update_individualized (BPT 个性化参数)
# ============================================================


def _logit(p: float) -> float:
    """logit 变换 (测试辅助)."""
    p = max(min(p, 1.0 - 1e-12), 1e-12)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """sigmoid 函数 (测试辅助)."""
    return 1.0 / (1.0 + math.exp(-x))


class TestBKTTracerUpdateIndividualized:
    """BKTTracer.update_individualized 测试 — BPT 个性化参数 (logit-sigmoid 融合)."""

    def _make_state(self, mastery=0.5) -> TracingState:
        return TracingState(
            kp_id="kp-001",
            mastery_prob=mastery,
            bkt_params=dict(DEFAULT_BKT_PARAMS),
        )

    def test_update_individualized_fallback_when_no_learner_params(self):
        """未提供 learner_params 时回退到标准 update (结果一致)."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.5)
        standard = tracer.update(state, correct=True, timestamp=10.0)
        individualized = tracer.update_individualized(
            state, correct=True, timestamp=10.0, learner_params=None
        )
        assert individualized.mastery_prob == pytest.approx(
            standard.mastery_prob, abs=1e-12
        )
        assert individualized.attempts == standard.attempts
        assert individualized.correct_count == standard.correct_count
        assert individualized.last_attempt_time == standard.last_attempt_time

    def test_update_individualized_fallback_when_empty_learner_params(self):
        """空 learner_params 字典也回退到标准 update."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.5)
        standard = tracer.update(state, correct=False, timestamp=10.0)
        individualized = tracer.update_individualized(
            state, correct=False, timestamp=10.0, learner_params={}
        )
        assert individualized.mastery_prob == pytest.approx(
            standard.mastery_prob, abs=1e-12
        )

    def test_update_individualized_logit_sigmoid_fusion(self):
        """融合公式: fused = sigmoid(logit(skill) + logit(learner))."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.5)
        learner_params = {
            "learner_p_t": 0.3,
            "learner_p_g": 0.15,
            "learner_p_s": 0.05,
        }
        # 手动计算融合后的参数
        skill = dict(DEFAULT_BKT_PARAMS)
        fused_t = _sigmoid(_logit(skill["p_t"]) + _logit(0.3))
        fused_g = _sigmoid(_logit(skill["p_g"]) + _logit(0.15))
        fused_s = _sigmoid(_logit(skill["p_s"]) + _logit(0.05))

        # 用融合参数手动跑一次标准前向更新 (答对)
        p_l = 0.5
        num = p_l * (1.0 - fused_s)
        den = num + (1.0 - p_l) * fused_g
        p_l_post = num / den
        expected = p_l_post + (1.0 - p_l_post) * fused_t

        result = tracer.update_individualized(
            state, correct=True, timestamp=1.0, learner_params=learner_params
        )
        assert result.mastery_prob == pytest.approx(expected, abs=1e-9)

    def test_update_individualized_differs_from_standard(self):
        """提供 learner_params 时结果应与标准 update 不同."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.5)
        learner_params = {
            "learner_p_t": 0.3,
            "learner_p_g": 0.15,
            "learner_p_s": 0.05,
        }
        standard = tracer.update(state, correct=True, timestamp=1.0)
        individualized = tracer.update_individualized(
            state, correct=True, timestamp=1.0, learner_params=learner_params
        )
        assert individualized.mastery_prob != pytest.approx(
            standard.mastery_prob, abs=1e-6
        )

    def test_update_individualized_preserves_skill_bkt_params(self):
        """融合为瞬时计算, 返回状态的 bkt_params 保持技能级原值."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.5)
        original_params = dict(state.bkt_params)
        result = tracer.update_individualized(
            state,
            correct=True,
            timestamp=1.0,
            learner_params={"learner_p_t": 0.3, "learner_p_g": 0.15,
                            "learner_p_s": 0.05},
        )
        assert result.bkt_params == original_params

    def test_update_individualized_partial_learner_params(self):
        """仅提供部分 learner 参数时, 其余使用技能级原值 (不融合)."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.5)
        # 仅提供 learner_p_t, p_g/p_s 保持技能级
        learner_params = {"learner_p_t": 0.3}
        skill = dict(DEFAULT_BKT_PARAMS)
        fused_t = _sigmoid(_logit(skill["p_t"]) + _logit(0.3))
        fused_g = skill["p_g"]
        fused_s = skill["p_s"]

        p_l = 0.5
        num = p_l * (1.0 - fused_s)
        den = num + (1.0 - p_l) * fused_g
        p_l_post = num / den
        expected = p_l_post + (1.0 - p_l_post) * fused_t

        result = tracer.update_individualized(
            state, correct=True, timestamp=1.0, learner_params=learner_params
        )
        assert result.mastery_prob == pytest.approx(expected, abs=1e-9)

    def test_update_individualized_increments_counters(self):
        """update_individualized 同样更新 attempts/correct_count/时间戳."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.5)
        state.attempts = 4
        state.correct_count = 2
        result = tracer.update_individualized(
            state,
            correct=True,
            timestamp=777.0,
            learner_params={"learner_p_t": 0.3},
        )
        assert result.attempts == 5
        assert result.correct_count == 3
        assert result.last_attempt_time == 777.0

    def test_update_individualized_does_not_mutate_input(self):
        """update_individualized 不修改入参 state (函数式风格)."""
        tracer = BKTTracer()
        state = self._make_state(mastery=0.5)
        original_mastery = state.mastery_prob
        tracer.update_individualized(
            state,
            correct=True,
            timestamp=1.0,
            learner_params={"learner_p_t": 0.3, "learner_p_g": 0.15,
                            "learner_p_s": 0.05},
        )
        assert state.mastery_prob == original_mastery


# ============================================================
# 10. BKTTracer - fit_params (梯度上升参数学习)
# ============================================================


def _expected_log_likelihood(corrects, params) -> float:
    """独立实现 BKT 对数似然 (测试校验用)."""
    p_l = float(params["p_l0"])
    p_t = float(params["p_t"])
    p_g = float(params["p_g"])
    p_s = float(params["p_s"])
    ll = 0.0
    for c in corrects:
        pc = p_l * (1.0 - p_s) + (1.0 - p_l) * p_g
        po = pc if c else (1.0 - pc)
        ll += math.log(max(po, 1e-12))
        if c:
            num = p_l * (1.0 - p_s)
            den = num + (1.0 - p_l) * p_g
        else:
            num = p_l * p_s
            den = num + (1.0 - p_l) * (1.0 - p_g)
        p_l_post = num / den if den > 1e-12 else p_l
        p_l = p_l_post + (1.0 - p_l_post) * p_t
        p_l = min(max(p_l, 1e-12), 1.0 - 1e-12)
    return ll


def _records_from_corrects(corrects, kp="kp-1", difficulty=0.5):
    """由正确性序列构造 AnswerRecord 列表 (时间戳升序, 单位秒)."""
    return [
        AnswerRecord("l1", kp, bool(c), float(t) * 3600.0, difficulty)
        for t, c in enumerate(corrects)
    ]


class TestBKTTracerFitParams:
    """BKTTracer.fit_params 测试 — 梯度上升学习 p_t/p_g/p_s."""

    def test_fit_params_returns_dict_with_four_keys(self):
        """返回字典含 BKT 四参数 p_l0/p_t/p_g/p_s."""
        tracer = BKTTracer()
        records = _records_from_corrects([True, False, True, True, False])
        fitted = tracer.fit_params(records, max_iter=50)
        assert set(fitted.keys()) >= {"p_l0", "p_t", "p_g", "p_s"}

    def test_fit_params_values_in_unit_interval(self):
        """拟合后各参数落在 (0, 1) 内."""
        tracer = BKTTracer()
        records = _records_from_corrects([True, False, True, True, False, True])
        fitted = tracer.fit_params(records, max_iter=50)
        for k in ("p_l0", "p_t", "p_g", "p_s"):
            assert 0.0 < fitted[k] < 1.0, f"{k}={fitted[k]} 不在 (0,1)"

    def test_fit_params_satisfies_constraint_pg_plus_ps_lt_one(self):
        """拟合结果满足约束 p_g + p_s < 1."""
        tracer = BKTTracer()
        records = _records_from_corrects([True, False, True, True, False])
        fitted = tracer.fit_params(records, max_iter=100)
        assert fitted["p_g"] + fitted["p_s"] < 1.0

    def test_fit_params_improves_log_likelihood(self):
        """梯度上升后对数似然不应下降 (>= 初始似然)."""
        tracer = BKTTracer()
        corrects = [True, False, True, True, False, True, False, True, True, True]
        records = _records_from_corrects(corrects)
        init_params = dict(DEFAULT_BKT_PARAMS)
        init_ll = tracer.log_likelihood(records, init_params)
        fitted = tracer.fit_params(records, max_iter=100)
        fitted_ll = tracer.log_likelihood(records, fitted)
        assert fitted_ll >= init_ll - 1e-9

    def test_log_likelihood_formula(self):
        """log_likelihood 与独立实现的前向对数似然一致."""
        tracer = BKTTracer()
        corrects = [True, False, True, True, False]
        records = _records_from_corrects(corrects)
        params = {"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1}
        expected = _expected_log_likelihood(corrects, params)
        assert tracer.log_likelihood(records, params) == pytest.approx(
            expected, abs=1e-9
        )

    def test_fit_params_all_wrong_lowers_guess(self):
        """全错序列 -> 拟合猜测概率 p_g 应下降 (低于初始)."""
        tracer = BKTTracer()
        records = _records_from_corrects([False] * 15)
        fitted = tracer.fit_params(records, max_iter=100)
        assert fitted["p_g"] < DEFAULT_BKT_PARAMS["p_g"]

    def test_fit_params_all_correct_lowers_slip(self):
        """全对序列 -> 拟合失误概率 p_s 应较低 (<= 0.3)."""
        tracer = BKTTracer()
        records = _records_from_corrects([True] * 15)
        fitted = tracer.fit_params(records, max_iter=100)
        assert fitted["p_s"] <= 0.3

    def test_fit_params_respects_max_iter(self):
        """max_iter=1 也能返回合法结果 (单步梯度上升)."""
        tracer = BKTTracer()
        records = _records_from_corrects([True, False, True, False])
        fitted = tracer.fit_params(records, max_iter=1)
        assert 0.0 < fitted["p_t"] < 1.0
        assert fitted["p_g"] + fitted["p_s"] < 1.0


# ============================================================
# 11. ForgettingModel - compute_stability (动态稳定性)
# ============================================================


class TestForgettingModelComputeStability:
    """ForgettingModel.compute_stability 测试 — 动态记忆稳定性."""

    def test_compute_stability_zero_attempts(self):
        """0 次作答 -> stability = MIN_STABILITY = 1.0."""
        model = ForgettingModel()
        assert model.compute_stability(attempts=0, correct_count=0) == pytest.approx(1.0)

    def test_compute_stability_formula(self):
        """公式: stability = MIN_STABILITY + attempts * STABILITY_GAIN (1.0 + 0.5*attempts)."""
        model = ForgettingModel()
        for attempts in (0, 1, 2, 5, 10, 20):
            expected = 1.0 + attempts * 0.5
            assert model.compute_stability(attempts, correct_count=0) == pytest.approx(
                expected
            )

    def test_compute_stability_constants(self):
        """MIN_STABILITY=1.0, STABILITY_GAIN=0.5."""
        from dy3_polaris.l2.knowledge_tracer.forgetting import (
            MIN_STABILITY,
            STABILITY_GAIN,
        )
        assert MIN_STABILITY == pytest.approx(1.0)
        assert STABILITY_GAIN == pytest.approx(0.5)

    def test_compute_stability_monotonic_in_attempts(self):
        """attempts 越多 stability 越高 (遗忘越慢)."""
        model = ForgettingModel()
        prev = model.compute_stability(0, 0)
        for a in range(1, 15):
            cur = model.compute_stability(a, a)
            assert cur > prev
            prev = cur

    def test_compute_stability_accepts_correct_count(self):
        """签名接受 correct_count 参数 (与 attempts 一并传入)."""
        model = ForgettingModel()
        # correct_count 不应导致异常; attempts 主导公式
        s = model.compute_stability(attempts=10, correct_count=5)
        assert s == pytest.approx(1.0 + 10 * 0.5)


# ============================================================
# 12. ForgettingModel - compute_retention (平滑衰减)
# ============================================================


class TestForgettingModelComputeRetention:
    """ForgettingModel.compute_retention 测试 — 连续指数衰减 (无硬阈值)."""

    def test_compute_retention_zero_delta_returns_mastery(self):
        """delta_t=0 -> 无衰减, 返回原掌握度."""
        model = ForgettingModel()
        assert model.compute_retention(0.8, 0.0, 1.0) == pytest.approx(0.8)

    def test_compute_retention_formula(self):
        """公式: retention = mastery * exp(-(lambda/stability) * delta_t)."""
        model = ForgettingModel()
        mastery, delta_t, stability = 0.8, 100.0, 1.0
        expected = mastery * math.exp(-(0.007 / stability) * delta_t)
        assert model.compute_retention(mastery, delta_t, stability) == pytest.approx(
            expected, abs=1e-9
        )

    def test_compute_retention_smooth_no_hard_threshold(self):
        """平滑衰减: delta_t < 168 仍有衰减 (区别于 decay 的硬阈值)."""
        model = ForgettingModel()
        # 100 小时 < 168, decay 返回原值, 但 compute_retention 应衰减
        assert model.decay(0.8, 100.0) == pytest.approx(0.8)
        assert model.compute_retention(0.8, 100.0, 1.0) < 0.8

    def test_compute_retention_decreases_with_time(self):
        """时间越长, 保留率越低."""
        model = ForgettingModel()
        r_short = model.compute_retention(0.8, 100.0, 1.0)
        r_long = model.compute_retention(0.8, 500.0, 1.0)
        assert r_long < r_short

    def test_compute_retention_higher_stability_higher_retention(self):
        """stability 越大, 同时间保留率越高."""
        model = ForgettingModel()
        r_low = model.compute_retention(0.8, 500.0, 1.0)
        r_high = model.compute_retention(0.8, 500.0, 5.0)
        assert r_high > r_low

    def test_compute_retention_in_unit_interval(self):
        """保留率始终在 [0, 1]."""
        model = ForgettingModel()
        for mastery in (0.0, 0.3, 0.8, 1.0):
            for delta_t in (0.0, 10.0, 168.0, 1000.0):
                r = model.compute_retention(mastery, delta_t, 1.0)
                assert 0.0 <= r <= 1.0

    def test_compute_retention_negative_delta_returns_mastery(self):
        """负 delta_t (时间倒流) 视为无衰减."""
        model = ForgettingModel()
        assert model.compute_retention(0.8, -50.0, 1.0) == pytest.approx(0.8)


# ============================================================
# 13. ForgettingModel - should_review (动态稳定性集成)
# ============================================================


class TestForgettingModelShouldReviewStability:
    """ForgettingModel.should_review 测试 — 集成 compute_stability."""

    def _make_state(self, mastery=0.7, attempts=0, correct=0, last=0.0) -> TracingState:
        return TracingState(
            kp_id="kp-001",
            mastery_prob=mastery,
            attempts=attempts,
            correct_count=correct,
            last_attempt_time=last,
            bkt_params=dict(DEFAULT_BKT_PARAMS),
        )

    def test_should_review_uses_stability_from_attempts(self):
        """attempts 多 -> stability 高 -> 衰减慢 -> 不需复习; attempts 少 -> 需复习.

        同掌握度 (0.7) 与时间间隔 (500h):
        - attempts=0  -> stability=1.0  -> 衰减后 < 0.5 -> 需复习
        - attempts=20 -> stability=11.0 -> 衰减后 >= 0.5 -> 不需复习
        """
        model = ForgettingModel()
        current = 500.0 * 3600.0  # 500 小时 (秒)
        low_stab = self._make_state(mastery=0.7, attempts=0, correct=0)
        high_stab = self._make_state(mastery=0.7, attempts=20, correct=15)
        assert model.should_review(low_stab, current) is True
        assert model.should_review(high_stab, current) is False

    def test_should_review_higher_attempts_higher_decayed_mastery(self):
        """attempts 多的状态, 衰减后有效掌握度更高 (间接验证 compute_stability 接入)."""
        model = ForgettingModel()
        current = 500.0 * 3600.0
        low = self._make_state(mastery=0.7, attempts=0)
        high = self._make_state(mastery=0.7, attempts=20, correct=15)
        # 通过 decay 验证 stability 已随 attempts 提升
        delta_h = (current - low.last_attempt_time) / 3600.0
        d_low = model.decay(low.mastery_prob, delta_h,
                            stability=model.compute_stability(0, 0))
        d_high = model.decay(high.mastery_prob, delta_h,
                             stability=model.compute_stability(20, 15))
        assert d_high > d_low


# ============================================================
# 14. MasteryPropagator - propagate 加权 (3-tuple) 与向后兼容
# ============================================================


class TestMasteryPropagatorWeights:
    """MasteryPropagator.propagate 测试 — 支持 (kp_id, mastery, weight) 三元组."""

    def test_propagate_three_tuple_uses_weight(self):
        """三元组 (kp_id, mastery, weight) 使用指定权重."""
        prop = MasteryPropagator()
        # 0.5 + 0.3 * (0.5 * 0.8) = 0.5 + 0.12 = 0.62
        result = prop.propagate("kp-B", 0.5, [("kp-A", 0.8, 0.5)])
        assert result == pytest.approx(0.62, abs=1e-9)

    def test_propagate_two_tuple_backward_compatible(self):
        """二元组 (kp_id, mastery) 向后兼容, 使用 default_weight=1.0."""
        prop = MasteryPropagator()
        # 0.5 + 0.3 * (1.0 * 0.8) = 0.74
        result = prop.propagate("kp-B", 0.5, [("kp-A", 0.8)])
        assert result == pytest.approx(0.74, abs=1e-9)

    def test_propagate_mixed_two_and_three_tuples(self):
        """混合二元/三元组: 各自按对应权重计算."""
        prop = MasteryPropagator()
        # 0.5 + 0.3 * (1.0*0.8 + 0.5*0.6) = 0.5 + 0.3*(0.8+0.3) = 0.5 + 0.33 = 0.83
        result = prop.propagate(
            "kp-B", 0.5, [("kp-A", 0.8), ("kp-C", 0.6, 0.5)]
        )
        assert result == pytest.approx(0.83, abs=1e-9)

    def test_propagate_zero_weight_no_boost(self):
        """权重为 0 的前置不贡献提升."""
        prop = MasteryPropagator()
        result = prop.propagate("kp-B", 0.5, [("kp-A", 0.9, 0.0)])
        assert result == pytest.approx(0.5)

    def test_propagate_weight_amplifies_boost(self):
        """权重越大, 提升越强."""
        prop = MasteryPropagator()
        r_low = prop.propagate("kp-B", 0.5, [("kp-A", 0.8, 0.2)])
        r_high = prop.propagate("kp-B", 0.5, [("kp-A", 0.8, 1.0)])
        assert r_high > r_low

    def test_propagate_three_tuple_clamp_upper(self):
        """加权后超过 1.0 时 clamp 到 1.0."""
        prop = MasteryPropagator()
        result = prop.propagate("kp-B", 0.9, [("kp-A", 0.9, 1.0)])
        assert result == pytest.approx(1.0)


# ============================================================
# 15. MasteryPropagator - propagate_multi_hop (多跳传播)
# ============================================================


class TestMasteryPropagatorMultiHop:
    """MasteryPropagator.propagate_multi_hop 测试 — KG 多跳 BFS 传播."""

    def test_multi_hop_no_prerequisites(self):
        """无前置 -> 返回原掌握度."""
        prop = MasteryPropagator()
        kg = {"C": []}
        mastery_map = {}
        result = prop.propagate_multi_hop("C", 0.5, kg, mastery_map, max_depth=3)
        assert result == pytest.approx(0.5)

    def test_multi_hop_single_depth(self):
        """max_depth=1 仅传播直接前置 (深度1, 衰减 0.5)."""
        prop = MasteryPropagator()
        # C -> B(0.8); B 的掌握度 0.7
        kg = {"C": [("B", 0.8)], "B": [("A", 0.6)]}
        mastery_map = {"B": 0.7, "A": 0.9}
        # depth1: 0.3 * 0.8 * 0.7 * 0.5^1 = 0.084
        result = prop.propagate_multi_hop("C", 0.5, kg, mastery_map, max_depth=1)
        assert result == pytest.approx(0.5 + 0.084, abs=1e-9)

    def test_multi_hop_chain_two_depths(self):
        """链 C->B->A, max_depth=3: 深度1 (B) + 深度2 (A) 均贡献."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 0.8)], "B": [("A", 0.6)], "A": []}
        mastery_map = {"B": 0.7, "A": 0.9}
        # depth1 B: 0.3 * (0.8) * 0.7 * 0.5 = 0.084
        # depth2 A: 0.3 * (0.8*0.6) * 0.9 * 0.25 = 0.0324
        # boost = 0.1164
        result = prop.propagate_multi_hop("C", 0.5, kg, mastery_map, max_depth=3)
        assert result == pytest.approx(0.5 + 0.1164, abs=1e-9)

    def test_multi_hop_tree_multiple_prereqs(self):
        """树: C 有两个直接前置 B(0.8,0.7) 与 D(0.5,0.6), max_depth=3."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 0.8), ("D", 0.5)], "B": [], "D": []}
        mastery_map = {"B": 0.7, "D": 0.6}
        # depth1 B: 0.3*0.8*0.7*0.5 = 0.084
        # depth1 D: 0.3*0.5*0.6*0.5 = 0.045
        # boost = 0.129
        result = prop.propagate_multi_hop("C", 0.5, kg, mastery_map, max_depth=3)
        assert result == pytest.approx(0.5 + 0.129, abs=1e-9)

    def test_multi_hop_depth_decay_prevents_overboost(self):
        """更深的前置贡献更小 (0.5^depth 衰减防过载)."""
        prop = MasteryPropagator()
        # 同一个 A, 一次作为深度1直接前置, 一次作为深度2间接前置
        kg_direct = {"C": [("A", 0.6)], "A": []}
        kg_indirect = {"C": [("B", 0.6)], "B": [("A", 0.6)], "A": []}
        mastery_map = {"A": 0.9, "B": 0.0}  # B 掌握度为 0, 仅看 A 的贡献
        r_direct = prop.propagate_multi_hop(
            "C", 0.5, kg_direct, mastery_map, max_depth=3
        )
        r_indirect = prop.propagate_multi_hop(
            "C", 0.5, kg_indirect, mastery_map, max_depth=3
        )
        # 直接 (深度1) 贡献: 0.3*0.6*0.9*0.5 = 0.081
        # 间接 (深度2) 贡献: 0.3*0.36*0.9*0.25 = 0.0243
        assert r_direct > r_indirect

    def test_multi_hop_missing_mastery_treated_as_zero(self):
        """mastery_map 中缺失的前置掌握度视为 0 (不贡献)."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 0.8)], "B": []}
        result = prop.propagate_multi_hop("C", 0.5, kg, {}, max_depth=3)
        assert result == pytest.approx(0.5)

    def test_multi_hop_clamp_upper_bound(self):
        """结果超过 1.0 时 clamp 到 1.0."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 1.0), ("D", 1.0)], "B": [], "D": []}
        mastery_map = {"B": 1.0, "D": 1.0}
        result = prop.propagate_multi_hop("C", 0.9, kg, mastery_map, max_depth=3)
        assert result == pytest.approx(1.0)

    def test_multi_hop_result_in_unit_interval(self):
        """多跳传播结果始终在 [0, 1]."""
        prop = MasteryPropagator()
        kg = {"C": [("B", 0.9)], "B": [("A", 0.9)], "A": []}
        mastery_map = {"B": 0.9, "A": 0.9}
        for base in (0.0, 0.3, 0.7, 1.0):
            r = prop.propagate_multi_hop("C", base, kg, mastery_map, max_depth=3)
            assert 0.0 <= r <= 1.0

    def test_multi_hop_handles_cycle_without_infinite_loop(self):
        """图中存在环时不应无限循环 (visited 去重)."""
        prop = MasteryPropagator()
        # A <-> B 互为前置 (环)
        kg = {"A": [("B", 0.5)], "B": [("A", 0.5)]}
        mastery_map = {"A": 0.8, "B": 0.6}
        result = prop.propagate_multi_hop("A", 0.5, kg, mastery_map, max_depth=3)
        assert 0.0 <= result <= 1.0


# ============================================================
# 16. MasteryPropagator - propagate_reverse (反向传播)
# ============================================================


class TestMasteryPropagatorReverse:
    """MasteryPropagator.propagate_reverse 测试 — 反向 (后继->当前) 传播."""

    def test_reverse_no_successors(self):
        """无后继 -> 返回原掌握度."""
        prop = MasteryPropagator()
        assert prop.propagate_reverse("A", 0.5, []) == pytest.approx(0.5)

    def test_reverse_two_tuple_uses_default_weight(self):
        """二元组 (successor_id, mastery) 使用 default_weight=1.0."""
        prop = MasteryPropagator()
        # 0.5 + 0.3 * (1.0 * 0.8) = 0.74
        result = prop.propagate_reverse("A", 0.5, [("B", 0.8)])
        assert result == pytest.approx(0.74, abs=1e-9)

    def test_reverse_three_tuple_uses_weight(self):
        """三元组 (successor_id, mastery, weight) 使用指定权重."""
        prop = MasteryPropagator()
        # 0.5 + 0.3 * (0.5 * 0.8) = 0.62
        result = prop.propagate_reverse("A", 0.5, [("B", 0.8, 0.5)])
        assert result == pytest.approx(0.62, abs=1e-9)

    def test_reverse_multiple_successors(self):
        """多个后继: boosted = base + alpha * sum(weight * mastery)."""
        prop = MasteryPropagator()
        # 0.4 + 0.3 * (1.0*0.8 + 0.5*0.6) = 0.4 + 0.3*1.1 = 0.73
        result = prop.propagate_reverse(
            "A", 0.4, [("B", 0.8), ("C", 0.6, 0.5)]
        )
        assert result == pytest.approx(0.73, abs=1e-9)

    def test_reverse_clamp_upper_bound(self):
        """结果超过 1.0 时 clamp 到 1.0."""
        prop = MasteryPropagator()
        result = prop.propagate_reverse("A", 0.9, [("B", 0.9, 1.0)])
        assert result == pytest.approx(1.0)

    def test_reverse_zero_mastery_no_boost(self):
        """后继掌握度为 0 时不提升."""
        prop = MasteryPropagator()
        result = prop.propagate_reverse("A", 0.5, [("B", 0.0, 1.0)])
        assert result == pytest.approx(0.5)

    def test_reverse_higher_successor_mastery_higher_boost(self):
        """后继掌握度越高, 反向提升越强."""
        prop = MasteryPropagator()
        r_low = prop.propagate_reverse("A", 0.5, [("B", 0.3, 1.0)])
        r_high = prop.propagate_reverse("A", 0.5, [("B", 0.9, 1.0)])
        assert r_high > r_low
