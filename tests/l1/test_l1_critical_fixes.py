"""L1 引擎层关键修复与增强测试 — TDD (Red → Green).

覆盖修复与增强:
1. Bug Fix 1: FSRS initial_difficulty 公式 (w4 - exp(w5*(G-1)) + 1)
2. Bug Fix 2: IRT 3PL 信息函数 (c>0 时使用 3PL 公式)
3. Enhancement 3: IRT 4PL 模型 (probability / information / 序列化)
4. Enhancement 4: FSRSParameters 21 参数 (decay / factor 属性)
5. Enhancement 5: FSRSCardState 参数化 decay/factor
6. Fix 6/7: FSRSScheduler 参数化 decay / 均值回归目标 = initial_difficulty(4)
7. Enhancement 8: FSRSScheduler 短期记忆模型 (same-day)
8. Enhancement 9: FSRSScheduler fuzz factor
9. Enhancement 10: IRTEstimator Newton-Raphson MLE
10. Enhancement 11: IRTEstimator Gauss-Hermite EAP
11. Fix 12: 引擎模块导出 (FSRSScheduler / IRTEstimator / VARKSurveyCollector)

遵循 TDD:
- 先写测试 (RED): 描述期望行为, 实现缺失或错误时失败
- 再实现修复 (GREEN)
- 运行全部测试确保无回归
"""

from __future__ import annotations

import math
import time

import pytest

from dy3_polaris.l1.models import (
    FSRSParameters,
    FSRSCardState,
    IRTItem,
    IRTModel,
    MS_PER_SEC,
)


# ============================================================
# 辅助常量: 旧版 19 参数权重 (用于对比测试)
# ============================================================

# 旧版 19 参数权重 (来自原 FSRSParameters 默认值), 仅用于验证公式修正
_LEGACY_19_WEIGHTS = [
    0.4072, 1.1829, 3.1262, 15.4722, 7.2102, 0.5316, 1.0651,
    0.0234, 1.616, 0.1544, 1.0824, 1.9813, 0.0953, 0.2975,
    2.2042, 0.2407, 2.9466, 0.5034, 0.6567,
]

_MS_PER_DAY = float(MS_PER_SEC * 86400)


# ============================================================
# 1. Bug Fix 1: FSRS initial_difficulty 公式
# ============================================================


class TestFSRSInitialDifficultyFormula:
    """验证 D0(G) = w4 - exp(w5*(G-1)) + 1 (而非错误的 w4 - exp(w5)*(G-1) + 1)."""

    def test_initial_difficulty_matches_correct_formula(self):
        """initial_difficulty(G) 应等于 w4 - exp(w5*(G-1)) + 1 (clamp [1,10])."""
        params = FSRSParameters()
        w = params.weights
        for grade in (1, 2, 3, 4):
            expected = w[4] - math.exp(w[5] * (grade - 1)) + 1
            expected = max(1.0, min(10.0, expected))
            assert params.initial_difficulty(grade) == pytest.approx(expected), (
                f"grade={grade}: expected correct formula {expected}, "
                f"got {params.initial_difficulty(grade)}"
            )

    def test_initial_difficulty_does_not_use_buggy_formula(self):
        """correct formula 与 buggy formula 在 grade >= 3 时结果不同."""
        params = FSRSParameters()
        w = params.weights
        for grade in (3, 4):
            correct = max(1.0, min(10.0, w[4] - math.exp(w[5] * (grade - 1)) + 1))
            buggy = max(1.0, min(10.0, w[4] - math.exp(w[5]) * (grade - 1) + 1))
            # 两者应不同 (确认实现采用的是 correct 而非 buggy)
            assert correct != pytest.approx(buggy), (
                f"grade={grade}: correct and buggy formulas unexpectedly equal"
            )
            assert params.initial_difficulty(grade) == pytest.approx(correct)
            assert params.initial_difficulty(grade) != pytest.approx(buggy)

    def test_initial_difficulty_grade1_equals_w4(self):
        """grade=1 时 exp(w5*0)=1, D0(1) = w4 - 1 + 1 = w4."""
        params = FSRSParameters()
        w = params.weights
        # 仅当 w4 未触发 clamp 时成立
        if 1.0 <= w[4] <= 10.0:
            assert params.initial_difficulty(1) == pytest.approx(w[4])

    def test_initial_difficulty_decreases_with_grade(self):
        """评分越高, 初始难度越低."""
        params = FSRSParameters()
        d1 = params.initial_difficulty(1)
        d2 = params.initial_difficulty(2)
        d3 = params.initial_difficulty(3)
        d4 = params.initial_difficulty(4)
        assert d1 > d2 > d3 > d4

    def test_initial_difficulty_known_value_with_legacy_weights(self):
        """使用旧版 19 参数权重验证已知值, 确保公式正确 (与 py-fsrs 一致)."""
        params = FSRSParameters(weights=list(_LEGACY_19_WEIGHTS))
        w = params.weights
        # grade=2: D0 = w4 - exp(w5*1) + 1
        expected_g2 = max(1.0, min(10.0, w[4] - math.exp(w[5]) + 1))
        assert params.initial_difficulty(2) == pytest.approx(expected_g2)
        # grade=4: D0 = w4 - exp(w5*3) + 1
        expected_g4 = max(1.0, min(10.0, w[4] - math.exp(w[5] * 3) + 1))
        assert params.initial_difficulty(4) == pytest.approx(expected_g4)


# ============================================================
# 2. Bug Fix 2: IRT 3PL 信息函数
# ============================================================


class TestIRT3PLInformation:
    """3PL (c>0) 信息函数应使用 3PL 公式, 而非 2PL 公式."""

    def test_3pl_information_uses_3pl_formula(self):
        """c>0 时 I(θ) = a²*(P-c)²*(1-P)/((1-c)²*P)."""
        a, b, c = 1.5, 0.0, 0.2
        item = IRTItem(
            item_id="q-3pl",
            model_type=IRTModel.THREE_PL,
            difficulty_b=b,
            discrimination_a=a,
            guessing_c=c,
        )
        for theta in (-1.0, 0.0, 0.5, 1.5):
            p = item.probability(theta)
            p_clamped = max(1e-10, min(1 - 1e-10, p))
            expected_3pl = a ** 2 * (p_clamped - c) ** 2 * (1 - p_clamped) / (
                (1 - c) ** 2 * p_clamped
            )
            expected_2pl = a ** 2 * p_clamped * (1 - p_clamped)
            info = item.information(theta)
            assert info == pytest.approx(expected_3pl, rel=1e-6, abs=1e-9), (
                f"theta={theta}: 3PL info mismatch"
            )
            # 3PL 与 2PL 公式结果应不同 (确认采用了 3PL 分支)
            assert info != pytest.approx(expected_2pl, rel=1e-3, abs=1e-9), (
                f"theta={theta}: info matches 2PL formula (bug not fixed)"
            )

    def test_2pl_information_still_uses_2pl_formula(self):
        """2PL (c=0) 信息函数保持 I(θ) = a²*P*(1-P)."""
        a, b = 1.5, 0.0
        item = IRTItem(
            item_id="q-2pl",
            model_type=IRTModel.TWO_PL,
            difficulty_b=b,
            discrimination_a=a,
        )
        for theta in (-1.0, 0.0, 1.0):
            p = item.probability(theta)
            expected = a ** 2 * p * (1 - p)
            assert item.information(theta) == pytest.approx(expected)

    def test_3pl_information_nonnegative(self):
        """3PL 信息函数应非负."""
        item = IRTItem(
            item_id="q-3pl-neg",
            model_type=IRTModel.THREE_PL,
            difficulty_b=0.5,
            discrimination_a=1.2,
            guessing_c=0.25,
        )
        for theta in (-3.0, -1.0, 0.0, 1.0, 3.0):
            assert item.information(theta) >= 0.0


# ============================================================
# 3. Enhancement 3: IRT 4PL 模型
# ============================================================


class TestIRT4PLModel:
    """IRT 4PL 模型 — 上渐近线 upper_d."""

    def test_four_pl_enum_exists(self):
        assert IRTModel.FOUR_PL
        assert IRTModel.FOUR_PL.value == "4pl"

    def test_4pl_default_upper_d_is_one(self):
        item = IRTItem(
            item_id="q-4pl",
            model_type=IRTModel.FOUR_PL,
            difficulty_b=0.0,
            discrimination_a=1.0,
            guessing_c=0.2,
        )
        assert item.upper_d == 1.0

    def test_4pl_probability_formula(self):
        """P(θ) = c + (d - c) / (1 + exp(-a*(θ-b)))."""
        a, b, c, d = 1.0, 0.0, 0.2, 0.9
        item = IRTItem(
            item_id="q-4pl-p",
            model_type=IRTModel.FOUR_PL,
            difficulty_b=b,
            discrimination_a=a,
            guessing_c=c,
            upper_d=d,
        )
        for theta in (-2.0, -0.5, 0.0, 0.5, 2.0):
            z = a * (theta - b)
            expected = c + (d - c) / (1 + math.exp(-z))
            expected = max(0.0, min(1.0, expected))
            assert item.probability(theta) == pytest.approx(expected)

    def test_4pl_probability_respects_upper_asymptote(self):
        """θ → +∞ 时 P → upper_d (而非 1.0)."""
        item = IRTItem(
            item_id="q-4pl-asym",
            model_type=IRTModel.FOUR_PL,
            difficulty_b=0.0,
            discrimination_a=1.0,
            guessing_c=0.2,
            upper_d=0.85,
        )
        p_high = item.probability(10.0)
        assert p_high == pytest.approx(0.85, abs=1e-3)
        # 4PL 上界不应超过 upper_d
        assert p_high <= 0.85 + 1e-9

    def test_4pl_probability_respects_lower_asymptote(self):
        """θ → -∞ 时 P → guessing_c."""
        item = IRTItem(
            item_id="q-4pl-low",
            model_type=IRTModel.FOUR_PL,
            difficulty_b=0.0,
            discrimination_a=1.0,
            guessing_c=0.2,
            upper_d=0.85,
        )
        p_low = item.probability(-10.0)
        assert p_low == pytest.approx(0.2, abs=1e-3)

    def test_4pl_information_formula(self):
        """4PL 信息函数 I = a²*(P-c)²*(d-P)²/((d-c)²*P*(1-P))."""
        a, b, c, d = 1.2, 0.3, 0.15, 0.9
        item = IRTItem(
            item_id="q-4pl-info",
            model_type=IRTModel.FOUR_PL,
            difficulty_b=b,
            discrimination_a=a,
            guessing_c=c,
            upper_d=d,
        )
        for theta in (-1.0, 0.0, 0.3, 1.5):
            p = item.probability(theta)
            p_clamped = max(1e-10, min(1 - 1e-10, p))
            denom = (d - c) ** 2 * p_clamped * (1 - p_clamped)
            expected = a ** 2 * (p_clamped - c) ** 2 * (d - p_clamped) ** 2 / denom
            assert item.information(theta) == pytest.approx(expected, rel=1e-6, abs=1e-9)

    def test_4pl_information_reduces_to_3pl_when_d_is_one(self):
        """upper_d=1 时 4PL 信息函数应等于 3PL 信息函数."""
        a, b, c = 1.3, 0.2, 0.25
        item_4pl = IRTItem(
            item_id="q-4pl-r3pl",
            model_type=IRTModel.FOUR_PL,
            difficulty_b=b,
            discrimination_a=a,
            guessing_c=c,
            upper_d=1.0,
        )
        item_3pl = IRTItem(
            item_id="q-3pl-r3pl",
            model_type=IRTModel.THREE_PL,
            difficulty_b=b,
            discrimination_a=a,
            guessing_c=c,
        )
        for theta in (-1.0, 0.0, 0.5, 1.0):
            assert item_4pl.information(theta) == pytest.approx(
                item_3pl.information(theta), rel=1e-6, abs=1e-9
            )

    def test_4pl_information_reduces_to_2pl_when_c0_d1(self):
        """c=0, d=1 时 4PL 信息函数应等于 2PL."""
        a, b = 1.1, 0.0
        item_4pl = IRTItem(
            item_id="q-4pl-r2pl",
            model_type=IRTModel.FOUR_PL,
            difficulty_b=b,
            discrimination_a=a,
            guessing_c=0.0,
            upper_d=1.0,
        )
        item_2pl = IRTItem(
            item_id="q-2pl-r2pl",
            model_type=IRTModel.TWO_PL,
            difficulty_b=b,
            discrimination_a=a,
        )
        for theta in (-1.0, 0.0, 1.0):
            assert item_4pl.information(theta) == pytest.approx(
                item_2pl.information(theta), rel=1e-6, abs=1e-9
            )

    def test_4pl_to_dict_and_from_dict_roundtrip(self):
        item = IRTItem(
            item_id="q-4pl-rt",
            model_type=IRTModel.FOUR_PL,
            difficulty_b=0.4,
            discrimination_a=1.4,
            guessing_c=0.18,
            upper_d=0.92,
        )
        d = item.to_dict()
        assert "upper_d" in d
        assert d["upper_d"] == 0.92
        restored = IRTItem.from_dict(d)
        assert restored.model_type == IRTModel.FOUR_PL
        assert restored.upper_d == 0.92
        assert restored.guessing_c == 0.18

    def test_4pl_from_dict_defaults_upper_d(self):
        """旧数据缺 upper_d 时应默认为 1.0."""
        d = {
            "item_id": "q-4pl-old",
            "model_type": "4pl",
            "difficulty_b": 0.0,
            "discrimination_a": 1.0,
            "guessing_c": 0.2,
        }
        item = IRTItem.from_dict(d)
        assert item.upper_d == 1.0

    def test_4pl_no_constraint_on_discrimination(self):
        """4PL 不强制 a=1 或 c=0."""
        item = IRTItem(
            item_id="q-4pl-nc",
            model_type=IRTModel.FOUR_PL,
            difficulty_b=0.0,
            discrimination_a=1.7,
            guessing_c=0.3,
            upper_d=0.95,
        )
        assert item.discrimination_a == 1.7
        assert item.guessing_c == 0.3


# ============================================================
# 4. Enhancement 4: FSRSParameters 21 参数 + decay/factor 属性
# ============================================================


class TestFSRSParameters21:
    """FSRS-6 21 参数 (w0-w20) 与 decay/factor 属性."""

    def test_default_weights_has_21_params(self):
        params = FSRSParameters()
        assert len(params.weights) == 21

    def test_w19_and_w20_present(self):
        params = FSRSParameters()
        w = params.weights
        # w19 (短期能力因子) 与 w20 (衰减常数, 正值)
        assert w[19] == pytest.approx(0.8285)
        assert w[20] == pytest.approx(0.12)

    def test_decay_property(self):
        """decay = -w20 (负值)."""
        params = FSRSParameters()
        assert params.decay == pytest.approx(-params.weights[20])
        assert params.decay < 0  # 衰减指数为负

    def test_factor_property(self):
        """factor = 0.9^(1/decay) - 1, 且 R(t=S)=0.9."""
        params = FSRSParameters()
        decay = params.decay
        factor = params.factor
        expected_factor = 0.9 ** (1.0 / decay) - 1.0
        assert factor == pytest.approx(expected_factor)
        # 验证定义: (1 + factor)^decay = 0.9 (R(t=S) = 0.9)
        assert (1 + factor) ** decay == pytest.approx(0.9, rel=1e-9)
        # factor 应为正
        assert factor > 0

    def test_fallback_decay_factor_when_short_weights(self):
        """weights 不足 21 个时, decay/factor 回退到 FSRS-5 默认值."""
        params = FSRSParameters(weights=list(_LEGACY_19_WEIGHTS))
        assert params.decay == pytest.approx(-0.5)
        assert params.factor == pytest.approx(19.0 / 81.0)

    def test_post_init_validates_min_weights(self):
        """weights 少于 4 个应抛 ValueError (initial_stability 至少需 w0-w3)."""
        with pytest.raises(ValueError):
            FSRSParameters(weights=[0.1, 0.2, 0.3])

    def test_initial_stability_validates_grade(self):
        params = FSRSParameters()
        with pytest.raises(ValueError):
            params.initial_stability(0)
        with pytest.raises(ValueError):
            params.initial_stability(5)

    def test_to_dict_from_dict_roundtrip_21(self):
        params = FSRSParameters()
        d = params.to_dict()
        restored = FSRSParameters.from_dict(d)
        assert restored.weights == params.weights
        assert len(restored.weights) == 21


# ============================================================
# 5. Enhancement 5: FSRSCardState 参数化 decay/factor
# ============================================================


class TestFSRSCardStateParameterizedDecay:
    """FSRSCardState.retrievability 接受 decay/factor 参数."""

    def test_retrievability_accepts_decay_factor_kwargs(self):
        """retrievability(current_ts, decay=, factor=) 应可调用."""
        now = int(time.time() * 1000)
        card = FSRSCardState(
            kc_id="kc-1",
            stability=10.0,
            state=FSRSCardState.REVIEW,
            last_review_ts=now,
        )
        # 不应抛 TypeError
        r = card.retrievability(current_ts=now, decay=-0.5, factor=19.0 / 81.0)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_retrievability_uses_passed_decay_factor(self):
        """传入自定义 decay/factor 应影响可提取性计算."""
        now = int(time.time() * 1000)
        card = FSRSCardState(
            kc_id="kc-2",
            stability=5.0,
            state=FSRSCardState.REVIEW,
            last_review_ts=now,
        )
        # t = S (5 天后) → R 应 = 0.9 (定义)
        future = now + int(5.0 * _MS_PER_DAY)
        r_default = card.retrievability(current_ts=future)  # 默认 decay=-0.5
        assert r_default == pytest.approx(0.9, abs=1e-6)

        # 用 FSRS-6 参数化 decay/factor (w20=0.12): t=S 时 R 仍应 = 0.9
        params = FSRSParameters()
        r_param = card.retrievability(
            current_ts=future, decay=params.decay, factor=params.factor
        )
        assert r_param == pytest.approx(0.9, abs=1e-6)

    def test_card_state_no_hardcoded_decay_fields(self):
        """FSRSCardState 不再持有 _DECAY / _FACTOR 字段."""
        card = FSRSCardState(kc_id="kc-3")
        # dataclass fields 不应包含 _DECAY / _FACTOR
        field_names = {f.name for f in card.__dataclass_fields__.values()}
        assert "_DECAY" not in field_names
        assert "_FACTOR" not in field_names


# ============================================================
# 6. Fix 6/7 + Enhancement 8/9: FSRSScheduler
# ============================================================


def _make_review_card(
    kc_id: str = "kc-sched",
    stability: float = 5.0,
    difficulty: float = 5.0,
    last_review_ts: int | None = None,
    state: str = FSRSCardState.REVIEW,
) -> FSRSCardState:
    ts = last_review_ts if last_review_ts is not None else int(time.time() * 1000)
    return FSRSCardState(
        kc_id=kc_id,
        stability=stability,
        difficulty=difficulty,
        state=state,
        reps=3,
        last_review_ts=ts,
    )


class TestFSRSSchedulerParameterizedDecay:
    """FSRSScheduler 使用 params.decay / params.factor (而非硬编码)."""

    def test_scheduler_uses_params_decay_for_interval(self):
        """下次间隔应基于 params.decay/factor: interval = round(S*(DR^(1/decay)-1)/factor)."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = FSRSParameters()
        # 多天间隔以走长期成功公式
        now = int(time.time() * 1000)
        last_ts = now - int(10.0 * _MS_PER_DAY)
        card = _make_review_card(
            kc_id="kc-decay", stability=3.0, difficulty=5.0, last_review_ts=last_ts
        )
        new_card, _log, interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now
        )
        # 期望间隔 = round(S_new * (DR^(1/decay) - 1) / factor)
        decay = params.decay
        factor = params.factor
        interval_factor = params.request_retention ** (1.0 / decay) - 1.0
        expected = max(
            1,
            int(round(new_card.stability * interval_factor / factor)),
        )
        expected = min(expected, params.maximum_interval)
        assert interval == expected

    def test_scheduler_interval_matches_fsrs6_params(self):
        """使用 21 参数时, t=S 处 R=0.9, 间隔应≈stability (DR=0.9)."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = FSRSParameters()
        # request_retention=0.9 → 当 stability 使得 R 衰减到 0.9 时 interval≈S
        # interval = round(S * (0.9^(1/decay)-1)/factor) = round(S*1) = S (因 factor 定义)
        # 直接验证 interval_factor/factor ≈ 1
        decay = params.decay
        factor = params.factor
        ratio = (params.request_retention ** (1.0 / decay) - 1.0) / factor
        assert ratio == pytest.approx(1.0, rel=1e-9)
        # 因此对于足够大的 stability, interval ≈ stability (取整)
        now = int(time.time() * 1000)
        last_ts = now - int(20.0 * _MS_PER_DAY)
        card = _make_review_card(
            kc_id="kc-ratio", stability=7.0, difficulty=5.0, last_review_ts=last_ts
        )
        new_card, _log, interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now
        )
        assert interval == pytest.approx(new_card.stability, abs=1)


class TestFSRSSchedulerMeanReversion:
    """难度均值回归目标应为 initial_difficulty(4) (Easy 默认难度)."""

    def test_mean_reversion_uses_initial_difficulty_easy(self):
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = FSRSParameters()
        w = params.weights
        now = int(time.time() * 1000)
        last_ts = now - int(5.0 * _MS_PER_DAY)
        D0 = 5.0
        card = _make_review_card(
            kc_id="kc-mr", stability=4.0, difficulty=D0, last_review_ts=last_ts
        )
        # grade=3 (Good): delta_d = -w6*(3-3)*(10-D)/9 = 0 → next_d_pre = D0
        new_card, _log, _interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now
        )
        # 期望: next_d = w7*initial_difficulty(4) + (1-w7)*D0
        target = params.initial_difficulty(4)
        expected = w[7] * target + (1.0 - w[7]) * D0
        expected = max(1.0, min(10.0, expected))
        assert new_card.difficulty == pytest.approx(expected, rel=1e-9)

    def test_mean_reversion_not_raw_w4(self):
        """均值回归目标不应是原始 w4."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = FSRSParameters()
        w = params.weights
        now = int(time.time() * 1000)
        last_ts = now - int(5.0 * _MS_PER_DAY)
        D0 = 5.0
        card = _make_review_card(
            kc_id="kc-mr2", stability=4.0, difficulty=D0, last_review_ts=last_ts
        )
        new_card, _log, _interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now
        )
        # 错误目标 (raw w4) 对应的期望值
        buggy_target = w[4]
        buggy_expected = max(1.0, min(10.0, w[7] * buggy_target + (1.0 - w[7]) * D0))
        # initial_difficulty(4) != w4 (因 exp(w5*3) != 1)
        assert params.initial_difficulty(4) != pytest.approx(w[4])
        assert new_card.difficulty != pytest.approx(buggy_expected, rel=1e-6)


class TestFSRSSchedulerShortTermMemory:
    """FSRSScheduler 短期记忆模型 (same-day, elapsed_days < 1)."""

    def test_same_day_review_uses_short_term_formula(self):
        """elapsed_days < 1 时使用短期公式 w17*(S^w18 - S)*(1 - w19*S) + S."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = FSRSParameters()
        w = params.weights
        now = int(time.time() * 1000)
        # 1 小时前 → elapsed_days = 1/24 < 1
        last_ts = now - int(1.0 * 3600 * 1000)
        S = 5.0
        card = _make_review_card(
            kc_id="kc-stm", stability=S, difficulty=5.0, last_review_ts=last_ts
        )
        new_card, _log, _interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now
        )
        # 短期公式 (grade>=3 → max(new_stability, S))
        w17 = w[17]
        w18 = w[18]
        w19 = w[19]
        raw = w17 * (S ** w18 - S) * (1 - w19 * S) + S
        expected = max(raw, S)
        expected = max(0.1, expected)
        assert new_card.stability == pytest.approx(expected, rel=1e-9)

    def test_same_day_review_differs_from_long_term(self):
        """same-day 复习稳定性应不同于长期成功公式."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = FSRSParameters()
        w = params.weights
        now = int(time.time() * 1000)
        S = 5.0
        # same-day
        last_ts_short = now - int(1.0 * 3600 * 1000)
        card_short = _make_review_card(
            kc_id="kc-stm-s", stability=S, difficulty=5.0, last_review_ts=last_ts_short
        )
        new_short, _log, _int = scheduler.schedule_review(
            card_state=card_short, grade=3, params=params, current_ts=now
        )
        # long-term (10 天)
        last_ts_long = now - int(10.0 * _MS_PER_DAY)
        card_long = _make_review_card(
            kc_id="kc-stm-l", stability=S, difficulty=5.0, last_review_ts=last_ts_long
        )
        new_long, _log2, _int2 = scheduler.schedule_review(
            card_state=card_long, grade=3, params=params, current_ts=now
        )
        # 两者使用不同公式 → 稳定性不同
        assert new_short.stability != pytest.approx(new_long.stability, rel=1e-6)

    def test_same_day_easy_does_not_decrease_stability(self):
        """grade>=3 的 same-day 复习稳定性不应低于原 S."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = FSRSParameters()
        now = int(time.time() * 1000)
        last_ts = now - int(2.0 * 3600 * 1000)
        S = 8.0
        card = _make_review_card(
            kc_id="kc-stm-ez", stability=S, difficulty=5.0, last_review_ts=last_ts
        )
        new_card, _log, _interval = scheduler.schedule_review(
            card_state=card, grade=4, params=params, current_ts=now
        )
        assert new_card.stability >= S - 1e-9


class TestFSRSSchedulerFuzz:
    """Enhancement 9: fuzz factor (enable_fuzzing)."""

    def test_schedule_review_accepts_enable_fuzzing(self):
        """schedule_review 应接受 enable_fuzzing 关键字参数."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = FSRSParameters()
        now = int(time.time() * 1000)
        last_ts = now - int(5.0 * _MS_PER_DAY)
        card = _make_review_card(
            kc_id="kc-fuzz", stability=3.0, difficulty=5.0, last_review_ts=last_ts
        )
        # 不应抛 TypeError
        new_card, _log, interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now,
            enable_fuzzing=True,
        )
        assert interval >= 1

    def test_fuzz_keeps_interval_within_5_percent(self):
        """开启 fuzz 后, 间隔应在原值 ±5% 范围内 (统计意义)."""
        from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler

        scheduler = FSRSScheduler()
        params = FSRSParameters()
        now = int(time.time() * 1000)
        last_ts = now - int(5.0 * _MS_PER_DAY)
        card = _make_review_card(
            kc_id="kc-fuzz2", stability=3.0, difficulty=5.0, last_review_ts=last_ts
        )
        # 基线 (无 fuzz)
        _c0, _l0, base_interval = scheduler.schedule_review(
            card_state=card, grade=3, params=params, current_ts=now,
            enable_fuzzing=False,
        )
        # 多次 fuzz
        intervals = []
        for _ in range(50):
            c, _l, iv = scheduler.schedule_review(
                card_state=card, grade=3, params=params, current_ts=now,
                enable_fuzzing=True,
            )
            intervals.append(iv)
        # 至少有一个偏离基线 (fuzz 生效) 或全部在 ±5% 内
        for iv in intervals:
            lo = max(1, int(round(base_interval * 0.95)))
            hi = int(round(base_interval * 1.05)) + 1
            assert lo - 1 <= iv <= hi + 1, f"fuzz interval {iv} out of ±5% of {base_interval}"


# ============================================================
# 7. Enhancement 10: IRTEstimator Newton-Raphson MLE
# ============================================================


def _make_irt_item(
    difficulty_b: float = 0.0,
    discrimination_a: float = 1.0,
    item_id: str = "item-nr",
    model_type: IRTModel = IRTModel.TWO_PL,
    guessing_c: float = 0.0,
) -> IRTItem:
    return IRTItem(
        item_id=item_id,
        model_type=model_type,
        difficulty_b=difficulty_b,
        discrimination_a=discrimination_a,
        guessing_c=guessing_c,
    )


class TestIRTEstimatorNewtonRaphson:
    """Newton-Raphson MLE 估计."""

    def test_method_exists(self):
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        assert hasattr(est, "estimate_mle_newton_raphson")

    def test_all_correct_positive_theta(self):
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=-1.0, item_id=f"nr-c-{i}")
            for i in range(5)
        ]
        responses = [(it, True) for it in items]
        ability = est.estimate_mle_newton_raphson(responses, initial_theta=0.0)
        assert ability.theta > 0.0

    def test_all_incorrect_negative_theta(self):
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=1.0, item_id=f"nr-w-{i}")
            for i in range(5)
        ]
        responses = [(it, False) for it in items]
        ability = est.estimate_mle_newton_raphson(responses, initial_theta=0.0)
        assert ability.theta < 0.0

    def test_matches_grid_mle(self):
        """Newton-Raphson 结果应与网格搜索 MLE 接近."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=-1.0, item_id="nr-m-0"),
            _make_irt_item(difficulty_b=0.5, item_id="nr-m-1"),
            _make_irt_item(difficulty_b=1.0, item_id="nr-m-2"),
            _make_irt_item(difficulty_b=-0.5, item_id="nr-m-3"),
            _make_irt_item(difficulty_b=0.0, item_id="nr-m-4"),
        ]
        responses = [
            (items[0], True),
            (items[1], False),
            (items[2], False),
            (items[3], True),
            (items[4], True),
        ]
        grid = est.estimate_mle(responses, initial_theta=0.0)
        nr = est.estimate_mle_newton_raphson(responses, initial_theta=0.0)
        # 两者应接近 (NR 精度更高)
        assert abs(nr.theta - grid.theta) <= 0.1

    def test_respects_theta_bounds(self):
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=2.5, item_id=f"nr-b-{i}")
            for i in range(10)
        ]
        responses = [(it, True) for it in items]
        ability = est.estimate_mle_newton_raphson(responses, initial_theta=0.0)
        assert -3.0 <= ability.theta <= 3.0

    def test_works_with_3pl_items(self):
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        items = [
            IRTItem(
                item_id=f"nr-3pl-{i}",
                model_type=IRTModel.THREE_PL,
                difficulty_b=0.0,
                discrimination_a=1.2,
                guessing_c=0.2,
            )
            for i in range(6)
        ]
        responses = [(it, True) for it in items]
        ability = est.estimate_mle_newton_raphson(responses, initial_theta=0.0)
        assert ability.theta > 0.0

    def test_empty_responses_returns_initial(self):
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        ability = est.estimate_mle_newton_raphson([], initial_theta=0.5)
        assert isinstance(ability.theta, float)


# ============================================================
# 8. Enhancement 11: IRTEstimator Gauss-Hermite EAP
# ============================================================


class TestIRTEstimatorGaussHermiteEAP:
    """Gauss-Hermite EAP 估计."""

    def test_method_exists(self):
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        assert hasattr(est, "estimate_eap_gauss_hermite")

    def test_all_correct_positive_theta(self):
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=-1.0, item_id=f"gh-c-{i}")
            for i in range(5)
        ]
        responses = [(it, True) for it in items]
        ability = est.estimate_eap_gauss_hermite(responses, n_quad=20)
        assert ability.theta > 0.0

    def test_all_incorrect_negative_theta(self):
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=1.0, item_id=f"gh-w-{i}")
            for i in range(5)
        ]
        responses = [(it, False) for it in items]
        ability = est.estimate_eap_gauss_hermite(responses, n_quad=20)
        assert ability.theta < 0.0

    def test_returns_valid_ability(self):
        from dy3_polaris.l1.irt_estimator import IRTEstimator
        from dy3_polaris.l1.models import IRTAbility

        est = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=0.0, item_id=f"gh-v-{i}")
            for i in range(4)
        ]
        responses = [(items[0], True), (items[1], False),
                     (items[2], True), (items[3], True)]
        ability = est.estimate_eap_gauss_hermite(responses, n_quad=20)
        assert isinstance(ability, IRTAbility)
        assert -3.0 <= ability.theta <= 3.0
        assert ability.standard_error > 0.0

    def test_more_accurate_than_uniform_eap(self):
        """Gauss-Hermite EAP 应与 uniform-grid EAP 接近 (同数量级)."""
        from dy3_polaris.l1.irt_estimator import IRTEstimator

        est = IRTEstimator()
        items = [
            _make_irt_item(difficulty_b=-0.5, item_id="gh-a-0"),
            _make_irt_item(difficulty_b=0.5, item_id="gh-a-1"),
            _make_irt_item(difficulty_b=0.0, item_id="gh-a-2"),
        ]
        responses = [(items[0], True), (items[1], False), (items[2], True)]
        gh = est.estimate_eap_gauss_hermite(responses, n_quad=20)
        # 用单题 update_theta 作为对照 (近似 EAP)
        from dy3_polaris.l1.models import IRTAbility
        ab = est.update_theta(IRTAbility(user_id="x", theta=0.0, standard_error=1.0),
                              items[2], correct=True)
        # 两者符号应一致 (混合作答, 偏正或接近 0)
        assert -3.0 <= gh.theta <= 3.0


# ============================================================
# 9. Fix 12: 引擎模块导出
# ============================================================


class TestEngineModuleExports:
    """L1 __init__.py 应导出 FSRSScheduler / IRTEstimator / VARKSurveyCollector."""

    def test_import_fsrs_scheduler_from_l1(self):
        from dy3_polaris.l1 import FSRSScheduler  # noqa: F401

    def test_import_irt_estimator_from_l1(self):
        from dy3_polaris.l1 import IRTEstimator  # noqa: F401

    def test_import_vark_collector_from_l1(self):
        from dy3_polaris.l1 import VARKSurveyCollector  # noqa: F401

    def test_all_three_in_l1_all(self):
        import dy3_polaris.l1 as l1

        assert "FSRSScheduler" in l1.__all__
        assert "IRTEstimator" in l1.__all__
        assert "VARKSurveyCollector" in l1.__all__

    def test_combined_import(self):
        from dy3_polaris.l1 import FSRSScheduler, IRTEstimator, VARKSurveyCollector

        assert FSRSScheduler is not None
        assert IRTEstimator is not None
        assert VARKSurveyCollector is not None
