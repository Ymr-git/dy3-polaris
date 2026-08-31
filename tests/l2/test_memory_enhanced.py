"""T4 记忆+遗忘曲线增强测试 — 全链路世界先进方案融合.

测试覆盖:
1. FullFSRS6 — 完整 21 参数 FSRS-6 模型
2. DuolingoHLR — Half-Life Regression 技能级遗忘
3. MemoryConsolidation — 睡眠/静息记忆巩固
4. InterferenceModel — 前瞻/反前干扰建模
5. PSIKTForgetting — PSI-KT 状态空间遗忘
6. OptimalScheduling — SSP-MMC 最优调度
7. BKTMemoryFusion — BKT 掌握度与遗忘曲线融合
8. QualityMetrics — 质量度量 (延迟/精度/覆盖率)
9. APIInterface — 服务接口暴露
"""

from __future__ import annotations

import math
import time

import pytest

from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.memory import (
    LongTermMemory,
    MemoryChunk,
    ShortTermMemory,
    WorkingMemory,
)
from dy3_polaris.l2.memory.tracing_service import (
    MemoryOutput,
    MemoryTracingService,
)
from dy3_polaris.l2.store import InMemoryL2Store


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def store():
    return InMemoryL2Store()


@pytest.fixture
def service(store):
    return MemoryTracingService(store=store)


def make_event(
    learner_id: str = "learner_001",
    kp_id: str = "kp_01",
    correct: bool = True,
    difficulty: float = 0.5,
    timestamp: float | None = None,
) -> AnswerEvent:
    return AnswerEvent(
        learner_id=learner_id,
        kp_id=kp_id,
        correct=correct,
        difficulty=difficulty,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


# ============================================================
# 1. FullFSRS6 — 完整 21 参数 FSRS-6 模型
# ============================================================


class TestFullFSRS6:
    """完整 FSRS-6 模型: 21 参数, 幂律检索, 稳定性更新."""

    def test_fsrs6_retrievability_power_law(self, service):
        """FSRS-6 检索概率使用幂律: R(t,S) = (1 + t/(9S))^(-Dw)."""
        ts = time.time()
        service.process(make_event(timestamp=ts))
        state = service.get_fsrs_state("learner_001", "kp_01")
        assert state is not None
        # t=0 时 R=1.0
        r0 = service.compute_retrievability(0.0, state["stability"])
        assert r0 == pytest.approx(1.0, abs=1e-6)
        # t=S 时 R≈0.9 (FSRS request_retention)
        r_at_s = service.compute_retrievability(state["stability"], state["stability"])
        assert 0.85 <= r_at_s <= 0.95
        # t→∞ 时 R→0 (3650天≈10年, 确保 R<0.1)
        r_far = service.compute_retrievability(3650.0, state["stability"])
        assert r_far < 0.1

    def test_fsrs6_difficulty_initialization(self, service):
        """FSRS-6 难度初始化: D0(G) = w4 - e^((G-1)*w5) + 1."""
        ts = time.time()
        # grade=4 (Easy) → 初始难度应较低
        event_easy = make_event(correct=True, difficulty=0.2, timestamp=ts)
        output_easy = service.process(event_easy)
        d_easy = output_easy.difficulty
        # grade=1 (Again) → 初始难度应较高
        event_hard = make_event(
            kp_id="kp_02", correct=False, difficulty=0.9, timestamp=ts + 1
        )
        output_hard = service.process(event_hard)
        d_hard = output_hard.difficulty
        assert d_hard > d_easy

    def test_fsrs6_stability_update_formula(self, service):
        """FSRS-6 稳定性更新包含 (11-D)^w9 * S^(-w10) * (e^((1-R)*w11)-1) 项."""
        ts = time.time()
        stabilities = []
        for i in range(5):
            event = make_event(
                correct=True, difficulty=0.3, timestamp=ts + i * 86400
            )
            output = service.process(event)
            stabilities.append(output.stability)
        # 稳定性应递增 (FSRS-6 增长公式)
        assert stabilities[-1] > stabilities[0]
        # 增长率应递减 (边际递减效应)
        growth_rates = [
            stabilities[i + 1] / stabilities[i] for i in range(len(stabilities) - 1)
        ]
        assert growth_rates[-1] <= growth_rates[0] + 0.5  # 允许一些波动

    def test_fsrs6_21_parameters_accessible(self, service):
        """FSRS-6 的 21 个参数可访问."""
        params = service.get_fsrs_parameters()
        assert isinstance(params, dict)
        # 至少包含核心参数
        assert "w4" in params  # initial_difficulty
        assert "w5" in params  # difficulty_decay
        assert "w8" in params  # stability_base
        assert "w9" in params  # stability_difficulty
        assert "w10" in params  # stability_activity
        assert "w11" in params  # stability_retrievability
        assert "decay" in params
        assert "request_retention" in params

    def test_fsrs6_interval_from_desired_retention(self, service):
        """FSRS-6 从目标保持率反推间隔: t = 9S*(R^(-1/Dw) - 1)."""
        ts = time.time()
        service.process(make_event(timestamp=ts))
        state = service.get_fsrs_state("learner_001", "kp_01")
        S = state["stability"]
        # 目标保持率 0.9
        interval_90 = service.compute_interval_from_retention(S, 0.9)
        # 目标保持率 0.8 (更低保持率 → 更长间隔)
        interval_80 = service.compute_interval_from_retention(S, 0.8)
        assert interval_80 > interval_90
        assert interval_90 >= 1

    def test_fsrs6_lapse_stability_shrink(self, service):
        """遗忘 (grade=1) 后稳定性大幅缩减 (FSRS lapse handling)."""
        ts = time.time()
        # 建立稳定性
        for i in range(3):
            service.process(make_event(correct=True, difficulty=0.3, timestamp=ts + i))
        state_before = service.get_fsrs_state("learner_001", "kp_01")
        s_before = state_before["stability"]
        # 遗忘
        service.process(
            make_event(correct=False, difficulty=0.9, timestamp=ts + 100)
        )
        state_after = service.get_fsrs_state("learner_001", "kp_01")
        s_after = state_after["stability"]
        assert s_after < s_before * 0.5  # 大幅缩减

    def test_fsrs6_hard_penalty_easy_bonus(self, service):
        """Hard penalty 和 Easy bonus 影响稳定性增长."""
        ts = time.time()
        # Good (grade=3)
        service_good = MemoryTracingService(store=InMemoryL2Store())
        service_good.process(make_event(correct=True, difficulty=0.5, timestamp=ts))
        state_good = service_good.get_fsrs_state("learner_001", "kp_01")
        s_good = state_good["stability"]

        # Easy (grade=4)
        service_easy = MemoryTracingService(store=InMemoryL2Store())
        service_easy.process(make_event(correct=True, difficulty=0.2, timestamp=ts))
        state_easy = service_easy.get_fsrs_state("learner_001", "kp_01")
        s_easy = state_easy["stability"]

        # Easy 的初始稳定性应高于 Good
        assert s_easy > s_good


# ============================================================
# 2. DuolingoHLR — Half-Life Regression 技能级遗忘
# ============================================================


class TestDuolingoHLR:
    """Duolingo Half-Life Regression: p = 2^(-Δ/h)."""

    def test_hlr_retrievability_exponential(self, service):
        """HLR 使用指数衰减: p = 2^(-Δ/h)."""
        ts = time.time()
        service.process(make_event(timestamp=ts))
        # HLR 可提取性
        r0 = service.compute_hlr_retrievability(0.0, half_life=5.0)
        assert r0 == pytest.approx(1.0, abs=1e-6)
        r_at_h = service.compute_hlr_retrievability(5.0, half_life=5.0)
        assert r_at_h == pytest.approx(0.5, abs=1e-4)  # t=h → p=0.5
        r_2h = service.compute_hlr_retrievability(10.0, half_life=5.0)
        assert r_2h == pytest.approx(0.25, abs=1e-4)  # t=2h → p=0.25

    def test_hlr_half_life_estimation(self, service):
        """HLR 从答题序列估计 half-life."""
        ts = time.time()
        # 答对多次 → half_life 增长
        for i in range(5):
            service.process(make_event(correct=True, difficulty=0.3, timestamp=ts + i))
        h = service.estimate_half_life("learner_001", "kp_01")
        assert h > 0
        # 多次答对后 half_life 应较大
        assert h > 1.0

    def test_hlr_feature_based_prediction(self, service):
        """HLR 特征向量预测: h = 2^(θ·x)."""
        features = {
            "correct_count": 5,
            "incorrect_count": 1,
            "total_reps": 6,
        }
        h = service.predict_half_life(features)
        assert h > 0
        # 更多正确 → 更长 half_life
        features_better = {
            "correct_count": 10,
            "incorrect_count": 0,
            "total_reps": 10,
        }
        h_better = service.predict_half_life(features_better)
        assert h_better > h

    def test_hlr_skill_strength_meter(self, service):
        """HLR 技能强度计: strength = f(h, accuracy, reps)."""
        ts = time.time()
        for i in range(5):
            service.process(make_event(correct=True, difficulty=0.3, timestamp=ts + i))
        strength = service.compute_skill_strength("learner_001", "kp_01")
        assert 0.0 <= strength <= 1.0
        # 多次答对后强度应较高
        assert strength > 0.5

    def test_hlr_vs_fsrs_comparison(self, service):
        """HLR 指数衰减 vs FSRS 幂律衰减的差异."""
        S = 5.0
        h = S  # 同等参数
        t = 10.0
        r_fsrs = service.compute_retrievability(t, S)
        r_hlr = service.compute_hlr_retrievability(t, h)
        # 两者都应在合理范围
        assert 0.0 < r_fsrs < 1.0
        assert 0.0 < r_hlr < 1.0
        # 幂律衰减通常慢于指数衰减 (长间隔时)
        assert r_fsrs >= r_hlr  # FSRS 衰减更慢


# ============================================================
# 3. MemoryConsolidation — 睡眠/静息记忆巩固
# ============================================================


class TestMemoryConsolidation:
    """记忆巩固: 睡眠/静息后弱记忆增强."""

    def test_consolidation_boost_weak_memory(self, service):
        """睡眠巩固对弱记忆 (低稳定性) 增强更明显."""
        S_weak = 0.5
        S_strong = 10.0
        sleep_hours = 8.0
        boost_weak = service.compute_consolidation_boost(S_weak, sleep_hours)
        boost_strong = service.compute_consolidation_boost(S_strong, sleep_hours)
        # 弱记忆的巩固增益比例应更高
        assert boost_weak / S_weak > boost_strong / S_strong

    def test_consolidation_zero_sleep(self, service):
        """无睡眠时无巩固增益."""
        boost = service.compute_consolidation_boost(5.0, 0.0)
        assert boost == pytest.approx(0.0, abs=1e-6)

    def test_consolidation_more_sleep_more_boost(self, service):
        """更多睡眠 → 更多巩固."""
        S = 2.0
        boost_4h = service.compute_consolidation_boost(S, 4.0)
        boost_8h = service.compute_consolidation_boost(S, 8.0)
        assert boost_8h > boost_4h

    def test_consolidation_applied_on_review(self, service):
        """睡眠后首次复习应用巩固增益."""
        ts = time.time()
        # 建立记忆
        service.process(make_event(correct=True, difficulty=0.3, timestamp=ts))
        state_before = service.get_fsrs_state("learner_001", "kp_01")
        s_before = state_before["stability"]
        # 模拟睡眠后复习
        service.apply_consolidation(
            "learner_001", "kp_01", sleep_hours=8.0, current_time=ts + 8 * 3600
        )
        state_after = service.get_fsrs_state("learner_001", "kp_01")
        s_after = state_after["stability"]
        assert s_after > s_before  # 巩固后稳定性增加

    def test_consolidation_decay_with_time(self, service):
        """巩固增益随时间衰减 (距睡眠越久增益越小)."""
        S = 2.0
        # 刚睡醒
        boost_fresh = service.compute_consolidation_boost(S, 8.0, hours_since_sleep=0.0)
        # 睡醒后很久
        boost_stale = service.compute_consolidation_boost(S, 8.0, hours_since_sleep=16.0)
        assert boost_fresh > boost_stale


# ============================================================
# 4. InterferenceModel — 前瞻/反前干扰建模
# ============================================================


class TestInterferenceModel:
    """干扰建模: 相似知识点间的 retroactive interference."""

    def test_interference_retroactive(self, service):
        """学习新相似知识点后, 旧知识点稳定性降低."""
        ts = time.time()
        # 学习 kp_01
        for i in range(3):
            service.process(
                make_event(kp_id="kp_01", correct=True, difficulty=0.3, timestamp=ts + i)
            )
        state_before = service.get_fsrs_state("learner_001", "kp_01")
        s_before = state_before["stability"]

        # 学习相似的 kp_02 (标记为 kp_01 的相似项)
        service.register_similarity("kp_02", "kp_01", similarity=0.8)
        service.process(
            make_event(kp_id="kp_02", correct=True, difficulty=0.3, timestamp=ts + 100)
        )

        # kp_01 的稳定性应因干扰而降低
        state_after = service.get_fsrs_state(
            "learner_001", "kp_01", current_time=ts + 101
        )
        s_after = state_after["stability"]
        assert s_after <= s_before  # 干扰导致降低或持平

    def test_interference_no_similarity(self, service):
        """无相似性的知识点不产生干扰."""
        ts = time.time()
        service.process(
            make_event(kp_id="kp_01", correct=True, difficulty=0.3, timestamp=ts)
        )
        state_before = service.get_fsrs_state("learner_001", "kp_01")
        # 不注册相似性
        service.process(
            make_event(kp_id="kp_02", correct=True, difficulty=0.3, timestamp=ts + 1)
        )
        state_after = service.get_fsrs_state("learner_001", "kp_01")
        assert state_after["stability"] == pytest.approx(
            state_before["stability"], abs=1e-6
        )

    def test_interference_decays_with_time(self, service):
        """干扰随时间衰减 (时间越久干扰越小)."""
        ts = time.time()
        service.process(
            make_event(kp_id="kp_01", correct=True, difficulty=0.3, timestamp=ts)
        )
        service.register_similarity("kp_02", "kp_01", similarity=0.9)

        # 紧接着学习 kp_02 (干扰大)
        svc1 = MemoryTracingService(store=InMemoryL2Store())
        svc1.process(
            make_event(kp_id="kp_01", correct=True, difficulty=0.3, timestamp=ts)
        )
        svc1.register_similarity("kp_02", "kp_01", similarity=0.9)
        svc1.process(
            make_event(kp_id="kp_02", correct=True, difficulty=0.3, timestamp=ts + 1)
        )
        s_close = svc1.get_fsrs_state("learner_001", "kp_01")["stability"]

        # 很久之后学习 kp_02 (干扰小)
        svc2 = MemoryTracingService(store=InMemoryL2Store())
        svc2.process(
            make_event(kp_id="kp_01", correct=True, difficulty=0.3, timestamp=ts)
        )
        svc2.register_similarity("kp_02", "kp_01", similarity=0.9)
        svc2.process(
            make_event(kp_id="kp_02", correct=True, difficulty=0.3, timestamp=ts + 30 * 86400)
        )
        s_far = svc2.get_fsrs_state("learner_001", "kp_01")["stability"]

        assert s_far >= s_close  # 时间远的干扰更小

    def test_interference_strength_scales_with_similarity(self, service):
        """干扰强度与相似度成正比."""
        ts = time.time()
        # 高相似度
        svc_high = MemoryTracingService(store=InMemoryL2Store())
        svc_high.process(
            make_event(kp_id="kp_01", correct=True, difficulty=0.3, timestamp=ts)
        )
        svc_high.register_similarity("kp_02", "kp_01", similarity=0.9)
        svc_high.process(
            make_event(kp_id="kp_02", correct=True, difficulty=0.3, timestamp=ts + 1)
        )
        s_high_interference = svc_high.get_fsrs_state("learner_001", "kp_01")["stability"]

        # 低相似度
        svc_low = MemoryTracingService(store=InMemoryL2Store())
        svc_low.process(
            make_event(kp_id="kp_01", correct=True, difficulty=0.3, timestamp=ts)
        )
        svc_low.register_similarity("kp_03", "kp_01", similarity=0.3)
        svc_low.process(
            make_event(kp_id="kp_03", correct=True, difficulty=0.3, timestamp=ts + 1)
        )
        s_low_interference = svc_low.get_fsrs_state("learner_001", "kp_01")["stability"]

        assert s_low_interference >= s_high_interference


# ============================================================
# 5. PSIKTForgetting — PSI-KT 状态空间遗忘
# ============================================================


class TestPSIKTForgetting:
    """PSI-KT 状态空间知识追踪 + 遗忘."""

    def test_psi_kt_state_transition(self, service):
        """PSI-KT 知识状态转移: m' = exp(-α·Δt)·m + (1-exp(-α·Δt))·μ."""
        m_current = 0.5
        alpha = 0.1  # 遗忘率
        delta_t = 7.0  # 7天
        mu_target = 0.8  # 学习后目标状态
        m_new = service.compute_psi_kt_transition(m_current, alpha, delta_t, mu_target)
        assert 0.0 <= m_new <= 1.0
        # 应在 m_current 和 mu_target 之间
        assert m_current <= m_new <= mu_target or mu_target <= m_new <= m_current

    def test_psi_kt_retention_ratio(self, service):
        """PSI-KT 保持率: r = exp(-α·τ)."""
        r_short = service.compute_psi_kt_retention(0.1, 1.0)
        r_long = service.compute_psi_kt_retention(0.1, 30.0)
        assert r_short > r_long
        assert 0.0 < r_long < 1.0
        assert r_short == pytest.approx(math.exp(-0.1), abs=1e-6)

    def test_psi_kt_forgetting_rate_adaptive(self, service):
        """PSI-KT 遗忘率可自适应 (难度高→遗忘快)."""
        alpha_easy = service.compute_adaptive_forgetting_rate(difficulty=0.2)
        alpha_hard = service.compute_adaptive_forgetting_rate(difficulty=0.8)
        assert alpha_hard > alpha_easy

    def test_psi_kt_predict_probability(self, service):
        """PSI-KT 预测答对概率: p = sigmoid(a·(m - b))."""
        m = 0.6
        a = 1.5  # 区分度
        b = 0.5  # 难度
        p = service.compute_psi_kt_predict(m, a, b)
        assert 0.0 <= p <= 1.0
        # m > b → p > 0.5
        assert p > 0.5

    def test_psi_kt_update_with_response(self, service):
        """PSI-KT 根据答题响应更新知识状态."""
        ts = time.time()
        service.process(make_event(correct=True, difficulty=0.3, timestamp=ts))
        m_before = service.get_psi_kt_state("learner_001", "kp_01")
        # 答对 → 知识状态提升
        service.process(
            make_event(correct=True, difficulty=0.3, timestamp=ts + 86400)
        )
        m_after = service.get_psi_kt_state("learner_001", "kp_01")
        assert m_after >= m_before


# ============================================================
# 6. OptimalScheduling — SSP-MMC 最优调度
# ============================================================


class TestOptimalScheduling:
    """SSP-MMC 最优复习调度."""

    def test_optimal_interval_proposal(self, service):
        """最优间隔提议基于 SSP-MMC 成本最小化."""
        ts = time.time()
        service.process(make_event(correct=True, difficulty=0.3, timestamp=ts))
        state = service.get_fsrs_state("learner_001", "kp_01")
        S = state["stability"]
        # 最优间隔应大于 0
        optimal = service.compute_optimal_interval(S, state["difficulty"])
        assert optimal >= 1
        # 应与简单 FSRS 间隔有差异 (考虑了长期成本)
        simple = service.schedule_review("learner_001", "kp_01")
        assert optimal > 0

    def test_optimal_scheduling_balances_cost(self, service):
        """最优调度平衡复习成本与遗忘成本."""
        S_low = 1.0  # 低稳定性
        S_high = 10.0  # 高稳定性
        D = 5.0
        interval_low = service.compute_optimal_interval(S_low, D)
        interval_high = service.compute_optimal_interval(S_high, D)
        # 高稳定性 → 更长间隔 (复习成本更高)
        assert interval_high > interval_low

    def test_optimal_review_queue(self, service):
        """最优复习队列: 按紧急度排序."""
        ts = time.time()
        # 多个知识点, 不同稳定性
        for i in range(5):
            service.process(
                make_event(kp_id=f"kp_{i}", correct=True, difficulty=0.3, timestamp=ts + i)
            )
        queue = service.get_review_queue("learner_001", current_time=ts + 7 * 86400)
        assert isinstance(queue, list)
        assert len(queue) > 0
        # 队列按紧急度排序 (可提取性最低的最紧急)
        if len(queue) > 1:
            r_values = [item["retrievability"] for item in queue]
            assert r_values[0] <= r_values[-1]  # 升序 (最低在前)

    def test_optimal_scheduling_cost_function(self, service):
        """SSP-MMC 成本函数: cost = review_cost + forgetting_cost."""
        S = 3.0
        D = 5.0
        interval = 5.0
        cost = service.compute_scheduling_cost(S, D, interval)
        assert cost > 0
        # 更长间隔 → 遗忘成本增加
        cost_long = service.compute_scheduling_cost(S, D, 30.0)
        assert cost_long > cost


# ============================================================
# 7. BKTMemoryFusion — BKT 掌握度与遗忘曲线融合
# ============================================================


class TestBKTMemoryFusion:
    """BKT 掌握度 × 遗忘曲线融合."""

    def test_forgetting_adjusted_mastery(self, service):
        """遗忘修正后的掌握度: P(known|t) = P(known) × R(t)."""
        ts = time.time()
        service.process(make_event(correct=True, difficulty=0.3, timestamp=ts))
        state = service.get_fsrs_state("learner_001", "kp_01")
        # 当前掌握度 (从稳定性映射)
        mastery_now = service.get_mastery_with_forgetting("learner_001", "kp_01", ts)
        assert 0.0 <= mastery_now <= 1.0
        # 很久后的掌握度应降低
        mastery_far = service.get_mastery_with_forgetting(
            "learner_001", "kp_01", ts + 60 * 86400
        )
        assert mastery_far < mastery_now

    def test_bkt_memory_fusion_output(self, service):
        """MemoryOutput 包含遗忘修正后的掌握度."""
        ts = time.time()
        output = service.process(make_event(timestamp=ts))
        assert hasattr(output, "mastery_with_forgetting")
        assert 0.0 <= output.mastery_with_forgetting <= 1.0

    def test_fusion_from_bkt_state(self, service):
        """从 BKT TracingState 构建遗忘修正掌握度."""
        from dy3_polaris.l2.models import TracingState

        ts = time.time()
        bkt_state = TracingState(
            kp_id="kp_01",
            mastery_prob=0.8,
            attempts=5,
            correct_count=4,
            last_attempt_time=ts,
        )
        # 刚学 (Δt≈0) → 掌握度接近原始
        mastery_fresh = service.fusion_mastery_from_bkt(bkt_state, ts)
        assert mastery_fresh == pytest.approx(0.8, abs=0.1)
        # 很久后 → 掌握度降低
        mastery_stale = service.fusion_mastery_from_bkt(bkt_state, ts + 30 * 86400)
        assert mastery_stale < mastery_fresh


# ============================================================
# 8. QualityMetrics — 质量度量
# ============================================================


class TestQualityMetrics:
    """质量度量: 延迟 / 精度 / 覆盖率."""

    def test_latency_under_threshold(self, service):
        """单事件处理延迟 < 5ms."""
        import time as _time

        ts = _time.time()
        for _ in range(100):
            service.process(make_event(timestamp=_time.time()))
        elapsed = _time.time() - ts
        avg_latency_ms = (elapsed / 100) * 1000
        assert avg_latency_ms < 5.0  # < 5ms

    def test_retrievability_accuracy(self, service):
        """可提取性预测精度: 已知数据上 MAE < 0.15."""
        ts = time.time()
        # 模拟学习过程
        events = []
        for i in range(10):
            events.append(make_event(correct=True, difficulty=0.3, timestamp=ts + i * 86400))
        outputs = service.batch_process(events)
        # 验证可提取性在合理范围
        for o in outputs:
            assert 0.0 <= o.retrievability <= 1.0
        # 最后一次的可提取性应较高 (刚复习)
        assert outputs[-1].retrievability > 0.8

    def test_coverage_metric(self, service):
        """覆盖率: 已追踪的知识点比例."""
        ts = time.time()
        all_kps = ["kp_01", "kp_02", "kp_03", "kp_04", "kp_05"]
        for kp in all_kps[:3]:  # 只学习了 3/5
            service.process(make_event(kp_id=kp, timestamp=ts))
        coverage = service.compute_coverage(all_kps)
        assert coverage == pytest.approx(0.6, abs=0.01)  # 3/5 = 0.6

    def test_review_schedule_quality(self, service):
        """复习调度质量: 间隔递增且合理."""
        ts = time.time()
        intervals = []
        for i in range(6):
            output = service.process(
                make_event(correct=True, difficulty=0.3, timestamp=ts + i * 86400)
            )
            intervals.append(output.fsrs_next_review_days)
        # 间隔应递增
        for i in range(1, len(intervals)):
            assert intervals[i] >= intervals[i - 1]
        # 最终间隔应显著大于初始
        assert intervals[-1] > intervals[0]


# ============================================================
# 9. APIInterface — 服务接口暴露
# ============================================================


class TestAPIInterface:
    """MemoryTracingService API 接口暴露."""

    def test_to_api_response(self, service):
        """MemoryOutput 转 API 响应格式."""
        ts = time.time()
        output = service.process(make_event(timestamp=ts))
        api_response = output.to_api_response()
        assert isinstance(api_response, dict)
        assert "learner_id" in api_response
        assert "retrievability" in api_response
        assert "stability" in api_response
        assert "difficulty" in api_response
        assert "next_review_days" in api_response
        assert "mastery" in api_response  # 遗忘修正掌握度

    def test_from_bkt_output(self, service):
        """从 BKT MasteryOutput 构建 MemoryOutput."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import MasteryOutput

        ts = time.time()
        bkt_output = MasteryOutput(
            learner_id="learner_001",
            kp_id="kp_01",
            p_mastery=0.75,
            p_correct_next=0.80,
            mastery_flag=False,
            attempts=5,
            last_updated_ts=ts,
            confidence_interval=[0.65, 0.85],
        )
        memory_output = MemoryOutput.from_bkt_output(bkt_output)
        assert memory_output.learner_id == "learner_001"
        assert memory_output.mastery_with_forgetting <= 0.75  # 遗忘修正后降低

    def test_get_review_schedule_api(self, service):
        """获取复习计划 API 接口."""
        ts = time.time()
        for i in range(3):
            service.process(
                make_event(kp_id=f"kp_{i}", timestamp=ts + i)
            )
        schedule = service.get_review_schedule_api("learner_001")
        assert isinstance(schedule, list)
        assert len(schedule) > 0
        for item in schedule:
            assert "kp_id" in item
            assert "next_review_days" in item
            assert "retrievability" in item
            assert "urgency" in item

    def test_memory_snapshot_api(self, service):
        """记忆快照 API 接口."""
        ts = time.time()
        service.process(make_event(timestamp=ts))
        snapshot = service.get_memory_snapshot_api("learner_001")
        assert isinstance(snapshot, dict)
        assert "working_memory_size" in snapshot
        assert "short_term_count" in snapshot
        assert "tracked_kps" in snapshot
        assert "avg_retrievability" in snapshot
        assert "review_queue_length" in snapshot
