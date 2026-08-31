"""L2 ability_assessor 子模块测试 — IRT 能力评估 + CAT 自适应选题.

测试覆盖 (TDD):
1. IRTEstimator (L2 版本, 面向 IRTState, item_params 用 dict):
   - predict_correct: 3PL 项目反应函数 P = c + (1-c)/(1+exp(-a*(theta-b)))
   - information: Fisher 信息量 (2PL / 3PL 分支公式)
   - update_theta: 贝叶斯 EAP 后验更新 (单题在线)
   - estimate_mle: 最大似然估计 (网格搜索, 批量离线)
2. CATSelector:
   - select_next: 最大 Fisher 信息准则选题 + 排除已答题目
   - should_stop: 终止条件 (SE < 阈值 或 题数 >= 上限)
   - estimate_ability: 从作答序列估计能力
3. 模块导出: ability_assessor 导出 IRTEstimator / CATSelector
"""

from __future__ import annotations

import math
import random

import pytest

from dy3_polaris.l2.models import IRTState
from dy3_polaris.l2.ability_assessor import CATSelector, IRTEstimator
from dy3_polaris.l2.ability_assessor.irt import _extract_params


# ============================================================
# 1. IRTEstimator.predict_correct 测试
# ============================================================


class TestPredictCorrect:
    """IRTEstimator.predict_correct — 3PL 项目反应函数."""

    def test_2pl_at_theta_equals_b_is_half(self):
        """2PL (c=0) 在 theta=b 时 P=0.5."""
        est = IRTEstimator()
        p = est.predict_correct(theta=0.0, a=1.0, b=0.0, c=0.0)
        assert p == pytest.approx(0.5)

    def test_2pl_high_theta_high_probability(self):
        """2PL 高能力 theta 对应高答对概率."""
        est = IRTEstimator()
        p_high = est.predict_correct(theta=2.0, a=1.0, b=0.0, c=0.0)
        p_low = est.predict_correct(theta=-2.0, a=1.0, b=0.0, c=0.0)
        assert p_high > 0.5
        assert p_low < 0.5
        assert p_high > p_low

    def test_2pl_known_value(self):
        """2PL 已知值: a=1, b=0, theta=1 -> P=1/(1+exp(-1))≈0.7311."""
        est = IRTEstimator()
        p = est.predict_correct(theta=1.0, a=1.0, b=0.0, c=0.0)
        assert p == pytest.approx(1.0 / (1.0 + math.exp(-1.0)), rel=1e-6)

    def test_3pl_at_theta_equals_b(self):
        """3PL 在 theta=b 时 P = c + (1-c)*0.5 = (1+c)/2."""
        est = IRTEstimator()
        c = 0.2
        p = est.predict_correct(theta=0.0, a=1.0, b=0.0, c=c)
        assert p == pytest.approx((1.0 + c) / 2.0)

    def test_3pl_floor_is_guessing(self):
        """3PL 极低能力时 P 趋近于猜测下限 c."""
        est = IRTEstimator()
        c = 0.25
        p = est.predict_correct(theta=-10.0, a=1.0, b=0.0, c=c)
        assert p == pytest.approx(c, abs=1e-4)

    def test_3pl_ceiling_below_one(self):
        """3PL 极高能力时 P 趋近于 1 (但仍受 c 影响公式)."""
        est = IRTEstimator()
        p = est.predict_correct(theta=10.0, a=1.0, b=0.0, c=0.2)
        assert p == pytest.approx(1.0, abs=1e-3)

    def test_predict_correct_in_unit_interval(self):
        """预测答对概率始终落在 [0, 1]."""
        est = IRTEstimator()
        for theta in (-3.0, -1.0, 0.0, 1.0, 3.0):
            for a in (0.8, 1.0, 2.5):
                for b in (-2.0, 0.0, 2.0):
                    for c in (0.0, 0.1, 0.3):
                        p = est.predict_correct(theta, a, b, c)
                        assert 0.0 <= p <= 1.0

    def test_default_c_is_zero(self):
        """未传 c 时默认 0.0 (2PL)."""
        est = IRTEstimator()
        p_default = est.predict_correct(theta=0.0, a=1.0, b=0.0)
        p_explicit = est.predict_correct(theta=0.0, a=1.0, b=0.0, c=0.0)
        assert p_default == pytest.approx(p_explicit)

    def test_increasing_a_steepens_curve(self):
        """区分度 a 越大, 曲线在 b 处越陡 (theta=b 仍为 0.5, 但偏离更快)."""
        est = IRTEstimator()
        # theta 略大于 b, a 越大 P 越高
        p_low_a = est.predict_correct(theta=0.5, a=0.5, b=0.0, c=0.0)
        p_high_a = est.predict_correct(theta=0.5, a=2.0, b=0.0, c=0.0)
        assert p_high_a > p_low_a


# ============================================================
# 2. IRTEstimator.information 测试
# ============================================================


class TestInformation:
    """IRTEstimator.information — Fisher 信息量 (2PL / 3PL)."""

    def test_2pl_at_theta_equals_b(self):
        """2PL 在 theta=b 时 I = a^2 * 0.5 * 0.5 = a^2/4."""
        est = IRTEstimator()
        a = 2.0
        info = est.information(theta=0.0, a=a, b=0.0, c=0.0)
        assert info == pytest.approx(a * a * 0.25)

    def test_2pl_known_value(self):
        """2PL a=1, b=0, theta=0 -> I = 0.25."""
        est = IRTEstimator()
        info = est.information(theta=0.0, a=1.0, b=0.0, c=0.0)
        assert info == pytest.approx(0.25)

    def test_3pl_known_value(self):
        """3PL a=1, b=0, c=0.2, theta=0 -> I = a^2*(P-c)^2*(1-P)/((1-c)^2*P)."""
        est = IRTEstimator()
        a, b, c = 1.0, 0.0, 0.2
        p = est.predict_correct(0.0, a, b, c)
        expected = (
            a * a * (p - c) ** 2 * (1 - p) / ((1 - c) ** 2 * p)
        )
        info = est.information(theta=0.0, a=a, b=b, c=c)
        assert info == pytest.approx(expected, rel=1e-6)

    def test_information_non_negative(self):
        """信息量始终非负."""
        est = IRTEstimator()
        for theta in (-3.0, -1.0, 0.0, 1.0, 3.0):
            for a in (0.8, 1.0, 2.5):
                for b in (-2.0, 0.0, 2.0):
                    for c in (0.0, 0.1, 0.3):
                        info = est.information(theta, a, b, c)
                        assert info >= 0.0

    def test_2pl_peak_near_theta_equals_b(self):
        """2PL 信息峰值在 theta≈b 处 (ZPD)."""
        est = IRTEstimator()
        a, b, c = 1.5, 0.5, 0.0
        info_at_b = est.information(b, a, b, c)
        # 偏离 b 的信息量应更小
        info_above = est.information(b + 2.0, a, b, c)
        info_below = est.information(b - 2.0, a, b, c)
        assert info_at_b > info_above
        assert info_at_b > info_below

    def test_information_scales_with_a_squared(self):
        """信息量与 a^2 成正比 (2PL, theta=b)."""
        est = IRTEstimator()
        info_a1 = est.information(0.0, a=1.0, b=0.0, c=0.0)
        info_a2 = est.information(0.0, a=2.0, b=0.0, c=0.0)
        assert info_a2 == pytest.approx(4.0 * info_a1)

    def test_default_c_is_zero(self):
        """未传 c 时默认 0.0 (2PL 信息公式)."""
        est = IRTEstimator()
        info_default = est.information(theta=0.0, a=1.0, b=0.0)
        info_explicit = est.information(theta=0.0, a=1.0, b=0.0, c=0.0)
        assert info_default == pytest.approx(info_explicit)

    def test_extreme_theta_near_zero_info(self):
        """极端 theta 处信息量趋近 0 (2PL)."""
        est = IRTEstimator()
        info_extreme = est.information(theta=10.0, a=1.0, b=0.0, c=0.0)
        assert info_extreme == pytest.approx(0.0, abs=1e-3)


# ============================================================
# 3. IRTEstimator.update_theta 测试
# ============================================================


class TestUpdateTheta:
    """IRTEstimator.update_theta — 贝叶斯 EAP 后验更新."""

    def test_returns_irt_state(self):
        """update_theta 返回 IRTState 实例."""
        est = IRTEstimator()
        state = IRTState(theta=0.0, se=1.0, response_count=0)
        new_state = est.update_theta(state, {"a": 1.0, "b": 0.0, "c": 0.0}, True)
        assert isinstance(new_state, IRTState)

    def test_increments_response_count(self):
        """每次更新 response_count 自增 1."""
        est = IRTEstimator()
        state = IRTState(theta=0.0, se=1.0, response_count=5)
        new_state = est.update_theta(state, {"a": 1.0, "b": 0.0, "c": 0.0}, True)
        assert new_state.response_count == 6

    def test_correct_easy_item_increases_theta(self):
        """答对简单题 (b<当前theta) 后 theta 上升."""
        est = IRTEstimator()
        state = IRTState(theta=0.0, se=1.0, response_count=0)
        new_state = est.update_theta(
            state, {"a": 1.0, "b": -1.0, "c": 0.0}, correct=True
        )
        assert new_state.theta > state.theta

    def test_wrong_easy_item_decreases_theta(self):
        """答错简单题 (b<当前theta) 后 theta 下降."""
        est = IRTEstimator()
        state = IRTState(theta=0.0, se=1.0, response_count=0)
        new_state = est.update_theta(
            state, {"a": 1.0, "b": -1.0, "c": 0.0}, correct=False
        )
        assert new_state.theta < state.theta

    def test_correct_hard_item_increases_theta(self):
        """答对难题 (b>当前theta) 后 theta 上升."""
        est = IRTEstimator()
        state = IRTState(theta=0.0, se=1.0, response_count=0)
        new_state = est.update_theta(
            state, {"a": 1.0, "b": 1.0, "c": 0.0}, correct=True
        )
        assert new_state.theta > state.theta

    def test_se_is_finite_and_positive(self):
        """更新后 se 为有限正数."""
        est = IRTEstimator()
        state = IRTState(theta=0.0, se=1.0, response_count=0)
        new_state = est.update_theta(state, {"a": 1.0, "b": 0.0, "c": 0.0}, True)
        assert math.isfinite(new_state.se)
        assert new_state.se > 0.0

    def test_se_decreases_with_more_observations(self):
        """多次观测后 se 相比初始先验下降."""
        est = IRTEstimator()
        state = IRTState(theta=0.0, se=1.0, response_count=0)
        initial_se = state.se
        for _ in range(8):
            state = est.update_theta(
                state, {"a": 1.2, "b": 0.0, "c": 0.0}, correct=True
            )
        assert state.se < initial_se

    def test_deterministic_same_input_same_output(self):
        """相同输入产生相同输出 (无状态引擎)."""
        est = IRTEstimator()
        state = IRTState(theta=0.3, se=0.5, response_count=4)
        item = {"a": 1.5, "b": 0.2, "c": 0.1}
        out1 = est.update_theta(state, item, True)
        out2 = est.update_theta(state, item, True)
        assert out1.theta == pytest.approx(out2.theta)
        assert out1.se == pytest.approx(out2.se)
        assert out1.response_count == out2.response_count

    def test_theta_clamped_to_range(self):
        """极端先验下 theta 仍钳制在 [-3, 3]."""
        est = IRTEstimator()
        # 高能力 + 连续答对极简单题, 不应超出范围
        state = IRTState(theta=2.9, se=0.2, response_count=20)
        for _ in range(5):
            state = est.update_theta(
                state, {"a": 2.0, "b": -2.0, "c": 0.0}, correct=True
            )
        assert -3.0 <= state.theta <= 3.0

    def test_works_with_3pl_params(self):
        """3PL 参数 (c>0) 不报错且返回有效状态."""
        est = IRTEstimator()
        state = IRTState(theta=0.0, se=1.0, response_count=0)
        new_state = est.update_theta(
            state, {"a": 1.2, "b": 0.5, "c": 0.2}, correct=True
        )
        assert isinstance(new_state, IRTState)
        assert -3.0 <= new_state.theta <= 3.0

    def test_preserves_last_update_time(self):
        """update_theta 保留传入 state 的 last_update_time (不在本方法修改)."""
        est = IRTEstimator()
        state = IRTState(
            theta=0.0, se=1.0, response_count=0, last_update_time=1234.5
        )
        new_state = est.update_theta(state, {"a": 1.0, "b": 0.0, "c": 0.0}, True)
        assert new_state.last_update_time == 1234.5


# ============================================================
# 4. IRTEstimator.estimate_mle 测试
# ============================================================


class TestEstimateMLE:
    """IRTEstimator.estimate_mle — 最大似然估计 (网格搜索)."""

    def test_empty_returns_default(self):
        """空响应回退: theta=0.0, se=1.0."""
        est = IRTEstimator()
        state = est.estimate_mle([])
        assert isinstance(state, IRTState)
        assert state.theta == pytest.approx(0.0)
        assert state.se == pytest.approx(1.0)
        assert state.response_count == 0

    def test_returns_irt_state(self):
        """estimate_mle 返回 IRTState 实例."""
        est = IRTEstimator()
        responses = [({"a": 1.0, "b": 0.0, "c": 0.0}, True)]
        state = est.estimate_mle(responses)
        assert isinstance(state, IRTState)

    def test_all_correct_easy_positive_theta(self):
        """全部答对简单题 -> 正 theta."""
        est = IRTEstimator()
        easy_item = {"a": 1.0, "b": -1.0, "c": 0.0}
        responses = [(easy_item, True)] * 6
        state = est.estimate_mle(responses)
        assert state.theta > 0.0

    def test_all_wrong_easy_negative_theta(self):
        """全部答错简单题 -> 负 theta."""
        est = IRTEstimator()
        easy_item = {"a": 1.0, "b": -1.0, "c": 0.0}
        responses = [(easy_item, False)] * 6
        state = est.estimate_mle(responses)
        assert state.theta < 0.0

    def test_response_count_matches(self):
        """response_count 等于响应数."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.0, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 0.5, "c": 0.0}, False),
            ({"a": 1.0, "b": -0.5, "c": 0.0}, True),
        ]
        state = est.estimate_mle(responses)
        assert state.response_count == 3

    def test_se_is_inverse_sqrt_info(self):
        """SE = 1 / sqrt(总信息量)."""
        est = IRTEstimator()
        responses = [({"a": 1.0, "b": 0.0, "c": 0.0}, True)] * 4
        state = est.estimate_mle(responses)
        total_info = sum(
            est.information(state.theta, r[0]["a"], r[0]["b"], r[0]["c"])
            for r in responses
        )
        if total_info > 0.0:
            assert state.se == pytest.approx(1.0 / math.sqrt(total_info), rel=1e-6)

    def test_clamped_to_range(self):
        """MLE theta 钳制在 [-3, 3]."""
        est = IRTEstimator()
        # 全对极难题, 倾向于推到上限
        responses = [({"a": 2.0, "b": 2.5, "c": 0.0}, True)] * 10
        state = est.estimate_mle(responses)
        assert -3.0 <= state.theta <= 3.0

    def test_deterministic(self):
        """相同输入产生相同输出."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.0, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 1.0, "c": 0.0}, False),
        ]
        s1 = est.estimate_mle(responses)
        s2 = est.estimate_mle(responses)
        assert s1.theta == pytest.approx(s2.theta)
        assert s1.se == pytest.approx(s2.se)

    def test_works_with_3pl(self):
        """3PL 参数批量估计不报错."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.2, "b": 0.0, "c": 0.2}, True),
            ({"a": 1.0, "b": 0.5, "c": 0.1}, False),
        ]
        state = est.estimate_mle(responses)
        assert isinstance(state, IRTState)
        assert -3.0 <= state.theta <= 3.0


# ============================================================
# 5. CATSelector.select_next 测试
# ============================================================


class TestSelectNext:
    """CATSelector.select_next — 最大 Fisher 信息准则选题."""

    def test_returns_max_info_item(self):
        """返回当前 theta 下信息量最大的题目."""
        selector = CATSelector()
        # theta=0, q1 (b=0) 信息量最大 (2PL 峰值在 b)
        items = [
            {"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": 2.0, "c": 0.0},
        ]
        chosen = selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert chosen is not None
        assert chosen["item_id"] == "q1"

    def test_excludes_administered(self):
        """排除已答题目, 返回次优."""
        selector = CATSelector()
        items = [
            {"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": 2.0, "c": 0.0},
        ]
        chosen = selector.select_next(
            theta=0.0, available_items=items, administered_ids={"q1"}
        )
        assert chosen is not None
        assert chosen["item_id"] == "q2"

    def test_empty_items_returns_none(self):
        """无可用题目返回 None."""
        selector = CATSelector()
        chosen = selector.select_next(theta=0.0, available_items=[], administered_ids=set())
        assert chosen is None

    def test_all_administered_returns_none(self):
        """所有题目均已答返回 None."""
        selector = CATSelector()
        items = [
            {"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": 1.0, "c": 0.0},
        ]
        chosen = selector.select_next(
            theta=0.0, available_items=items, administered_ids={"q1", "q2"}
        )
        assert chosen is None

    def test_returns_dict_with_item_id(self):
        """返回值是包含 item_id 的 dict."""
        selector = CATSelector()
        items = [{"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0}]
        chosen = selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert isinstance(chosen, dict)
        assert "item_id" in chosen

    def test_prefers_b_near_theta(self):
        """2PL 下优先选择 b≈theta 的题目 (信息峰值)."""
        selector = CATSelector()
        # theta=1.0, 应选 b=1.0 的题 (而非 b=-1.0)
        items = [
            {"item_id": "q_low", "a": 1.0, "b": -1.0, "c": 0.0},
            {"item_id": "q_match", "a": 1.0, "b": 1.0, "c": 0.0},
            {"item_id": "q_far", "a": 1.0, "b": 2.5, "c": 0.0},
        ]
        chosen = selector.select_next(theta=1.0, available_items=items, administered_ids=set())
        assert chosen is not None
        assert chosen["item_id"] == "q_match"

    def test_higher_discrimination_selected_when_b_equal(self):
        """难度相同时, 区分度高的题目信息量更大."""
        selector = CATSelector()
        items = [
            {"item_id": "q_low_a", "a": 0.8, "b": 0.0, "c": 0.0},
            {"item_id": "q_high_a", "a": 2.0, "b": 0.0, "c": 0.0},
        ]
        chosen = selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert chosen is not None
        assert chosen["item_id"] == "q_high_a"

    def test_administered_as_list_also_works(self):
        """administered_ids 可传 list (内部转集合)."""
        selector = CATSelector()
        items = [
            {"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": 2.0, "c": 0.0},
        ]
        chosen = selector.select_next(
            theta=0.0, available_items=items, administered_ids=["q1"]
        )
        assert chosen is not None
        assert chosen["item_id"] == "q2"


# ============================================================
# 6. CATSelector.should_stop 测试
# ============================================================


class TestShouldStop:
    """CATSelector.should_stop — 终止条件."""

    def test_stop_when_se_below_threshold(self):
        """SE < 阈值 -> 终止."""
        selector = CATSelector()
        assert selector.should_stop(current_se=0.2, count=5) is True

    def test_stop_when_count_reaches_max(self):
        """题数 >= 上限 -> 终止."""
        selector = CATSelector()
        assert selector.should_stop(current_se=0.9, count=20) is True

    def test_stop_when_count_exceeds_max(self):
        """题数超过上限 -> 终止."""
        selector = CATSelector()
        assert selector.should_stop(current_se=0.9, count=25) is True

    def test_continue_when_neither_met(self):
        """两个条件均不满足 -> 继续."""
        selector = CATSelector()
        assert selector.should_stop(current_se=0.5, count=10) is False

    def test_boundary_se_equal_threshold_continues(self):
        """SE == 阈值 (严格小于) 且题数不足 -> 继续."""
        selector = CATSelector()
        # se=0.3 不 < 0.3, count=10 < 20 -> 继续
        assert selector.should_stop(current_se=0.3, count=10) is False

    def test_custom_max_items(self):
        """自定义 max_items."""
        selector = CATSelector()
        assert selector.should_stop(current_se=0.9, count=10, max_items=10) is True
        assert selector.should_stop(current_se=0.9, count=9, max_items=10) is False

    def test_custom_se_threshold(self):
        """自定义 se_threshold."""
        selector = CATSelector()
        assert selector.should_stop(
            current_se=0.45, count=5, se_threshold=0.5
        ) is True
        assert selector.should_stop(
            current_se=0.55, count=5, se_threshold=0.5
        ) is False

    def test_default_params(self):
        """默认 max_items=20, se_threshold=0.3."""
        selector = CATSelector()
        # 仅传必填参数, 默认阈值下继续
        assert selector.should_stop(current_se=0.5, count=10) is False


# ============================================================
# 7. CATSelector.estimate_ability 测试
# ============================================================


class TestEstimateAbility:
    """CATSelector.estimate_ability — 从作答序列估计能力."""

    def test_empty_returns_default(self):
        """空作答序列回退: theta=0.0, se=1.0."""
        selector = CATSelector()
        state = selector.estimate_ability([])
        assert isinstance(state, IRTState)
        assert state.theta == pytest.approx(0.0)
        assert state.se == pytest.approx(1.0)

    def test_returns_irt_state(self):
        """返回 IRTState 实例."""
        selector = CATSelector()
        responses = [({"a": 1.0, "b": 0.0, "c": 0.0}, True)]
        state = selector.estimate_ability(responses)
        assert isinstance(state, IRTState)

    def test_correct_easy_positive_theta(self):
        """答对简单题序列 -> 正 theta."""
        selector = CATSelector()
        responses = [({"a": 1.0, "b": -1.0, "c": 0.0}, True)] * 6
        state = selector.estimate_ability(responses)
        assert state.theta > 0.0

    def test_wrong_easy_negative_theta(self):
        """答错简单题序列 -> 负 theta."""
        selector = CATSelector()
        responses = [({"a": 1.0, "b": -1.0, "c": 0.0}, False)] * 6
        state = selector.estimate_ability(responses)
        assert state.theta < 0.0

    def test_response_count_matches(self):
        """response_count 等于响应数."""
        selector = CATSelector()
        responses = [
            ({"a": 1.0, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 0.5, "c": 0.0}, False),
        ]
        state = selector.estimate_ability(responses)
        assert state.response_count == 2

    def test_consistent_with_mle(self):
        """estimate_ability 与 IRTEstimator.estimate_mle 结果一致."""
        selector = CATSelector()
        est = IRTEstimator()
        responses = [
            ({"a": 1.2, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 1.0, "c": 0.0}, False),
            ({"a": 1.5, "b": -0.5, "c": 0.0}, True),
        ]
        s_cat = selector.estimate_ability(responses)
        s_mle = est.estimate_mle(responses)
        assert s_cat.theta == pytest.approx(s_mle.theta)
        assert s_cat.se == pytest.approx(s_mle.se)


# ============================================================
# 8. 模块导出测试
# ============================================================


class TestAbilityAssessorExports:
    """ability_assessor 模块导出测试."""

    def test_export_irt_estimator(self):
        """ability_assessor 导出 IRTEstimator."""
        from dy3_polaris.l2.ability_assessor import IRTEstimator as E
        assert E is IRTEstimator

    def test_export_cat_selector(self):
        """ability_assessor 导出 CATSelector."""
        from dy3_polaris.l2.ability_assessor import CATSelector as C
        assert C is CATSelector

    def test_classes_instantiable(self):
        """IRTEstimator / CATSelector 可无参实例化."""
        assert IRTEstimator() is not None
        assert CATSelector() is not None

    def test_estimator_is_stateless(self):
        """IRTEstimator 多实例行为一致 (无状态)."""
        e1 = IRTEstimator()
        e2 = IRTEstimator()
        state = IRTState(theta=0.0, se=1.0, response_count=0)
        item = {"a": 1.0, "b": 0.0, "c": 0.0}
        out1 = e1.update_theta(state, item, True)
        out2 = e2.update_theta(state, item, True)
        assert out1.theta == pytest.approx(out2.theta)


# ============================================================
# 9. IRTEstimator.estimate_mle_newton_raphson 测试 (Task 1.1)
# ============================================================


class TestEstimateMLENewtonRaphson:
    """IRTEstimator.estimate_mle_newton_raphson — Newton-Raphson MLE."""

    def test_method_exists(self):
        """estimate_mle_newton_raphson 方法存在且可调用."""
        est = IRTEstimator()
        assert hasattr(est, "estimate_mle_newton_raphson")
        assert callable(est.estimate_mle_newton_raphson)

    def test_empty_returns_default(self):
        """空响应回退: theta=0.0, se=1.0."""
        est = IRTEstimator()
        state = est.estimate_mle_newton_raphson([])
        assert isinstance(state, IRTState)
        assert state.theta == pytest.approx(0.0)
        assert state.se == pytest.approx(1.0)
        assert state.response_count == 0

    def test_returns_irt_state(self):
        """返回 IRTState 实例."""
        est = IRTEstimator()
        responses = [({"a": 1.0, "b": 0.0, "c": 0.0}, True)]
        state = est.estimate_mle_newton_raphson(responses)
        assert isinstance(state, IRTState)

    def test_all_correct_easy_positive_theta(self):
        """全部答对简单题 -> 正 theta."""
        est = IRTEstimator()
        easy_item = {"a": 1.0, "b": -1.0, "c": 0.0}
        responses = [(easy_item, True)] * 6
        state = est.estimate_mle_newton_raphson(responses)
        assert state.theta > 0.0

    def test_all_wrong_easy_negative_theta(self):
        """全部答错简单题 -> 负 theta."""
        est = IRTEstimator()
        easy_item = {"a": 1.0, "b": -1.0, "c": 0.0}
        responses = [(easy_item, False)] * 6
        state = est.estimate_mle_newton_raphson(responses)
        assert state.theta < 0.0

    def test_response_count_matches(self):
        """response_count 等于响应数."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.0, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 0.5, "c": 0.0}, False),
            ({"a": 1.0, "b": -0.5, "c": 0.0}, True),
        ]
        state = est.estimate_mle_newton_raphson(responses)
        assert state.response_count == 3

    def test_matches_grid_search(self):
        """Newton-Raphson 与网格搜索结果一致 (2PL 对数似然为凹函数, 全局最优唯一)."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.2, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 0.5, "c": 0.0}, False),
            ({"a": 1.5, "b": -0.5, "c": 0.0}, True),
            ({"a": 1.0, "b": 1.0, "c": 0.0}, False),
            ({"a": 1.3, "b": 0.2, "c": 0.0}, True),
        ]
        grid = est.estimate_mle(responses)
        nr = est.estimate_mle_newton_raphson(responses)
        assert nr.theta == pytest.approx(grid.theta, abs=0.05)

    def test_converges_within_few_iterations(self):
        """少量迭代即收敛 (<=10), 结果与网格搜索一致."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.2, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 0.5, "c": 0.0}, False),
            ({"a": 1.5, "b": -0.5, "c": 0.0}, True),
            ({"a": 1.0, "b": 1.0, "c": 0.0}, False),
        ]
        grid = est.estimate_mle(responses)
        nr = est.estimate_mle_newton_raphson(responses, max_iter=10)
        assert nr.theta == pytest.approx(grid.theta, abs=0.05)

    def test_custom_initial_theta_converges_same(self):
        """不同初始 theta 收敛到同一最优 (2PL 凹似然)."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.2, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 0.5, "c": 0.0}, False),
            ({"a": 1.5, "b": -0.5, "c": 0.0}, True),
        ]
        nr0 = est.estimate_mle_newton_raphson(responses, initial_theta=0.0)
        nr_neg = est.estimate_mle_newton_raphson(responses, initial_theta=-2.0)
        nr_pos = est.estimate_mle_newton_raphson(responses, initial_theta=2.0)
        assert nr0.theta == pytest.approx(nr_neg.theta, abs=0.05)
        assert nr0.theta == pytest.approx(nr_pos.theta, abs=0.05)

    def test_clamped_to_range(self):
        """全对极难题 -> theta 钳制在 [-3, 3]."""
        est = IRTEstimator()
        responses = [({"a": 2.0, "b": 2.5, "c": 0.0}, True)] * 10
        state = est.estimate_mle_newton_raphson(responses)
        assert -3.0 <= state.theta <= 3.0

    def test_se_is_inverse_sqrt_info(self):
        """SE = 1 / sqrt(总信息量)."""
        est = IRTEstimator()
        responses = [({"a": 1.0, "b": 0.0, "c": 0.0}, True)] * 4
        state = est.estimate_mle_newton_raphson(responses)
        total_info = sum(
            est.information(state.theta, r[0]["a"], r[0]["b"], r[0]["c"])
            for r in responses
        )
        if total_info > 0.0:
            assert state.se == pytest.approx(1.0 / math.sqrt(total_info), rel=1e-6)

    def test_se_finite_and_positive(self):
        """SE 为有限正数."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.2, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 0.5, "c": 0.0}, False),
        ]
        state = est.estimate_mle_newton_raphson(responses)
        assert math.isfinite(state.se)
        assert state.se > 0.0

    def test_deterministic(self):
        """相同输入产生相同输出."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.0, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 1.0, "c": 0.0}, False),
        ]
        s1 = est.estimate_mle_newton_raphson(responses)
        s2 = est.estimate_mle_newton_raphson(responses)
        assert s1.theta == pytest.approx(s2.theta)
        assert s1.se == pytest.approx(s2.se)

    def test_works_with_3pl(self):
        """3PL 参数 Newton-Raphson 估计不报错且返回有效状态."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.2, "b": 0.0, "c": 0.2}, True),
            ({"a": 1.0, "b": 0.5, "c": 0.1}, False),
            ({"a": 1.5, "b": -0.5, "c": 0.15}, True),
        ]
        state = est.estimate_mle_newton_raphson(responses)
        assert isinstance(state, IRTState)
        assert -3.0 <= state.theta <= 3.0


# ============================================================
# 10. _extract_params 参数校验测试 (Task 1.2)
# ============================================================


class TestExtractParamsValidation:
    """_extract_params 参数合法性校验 (a>0, b∈[-3,3], c∈[0,0.5])."""

    def test_valid_params(self):
        """合法参数正常返回."""
        a, b, c = _extract_params({"a": 1.5, "b": 0.5, "c": 0.2})
        assert (a, b, c) == (1.5, 0.5, 0.2)

    def test_defaults(self):
        """缺失键回退默认值."""
        a, b, c = _extract_params({})
        assert (a, b, c) == (1.0, 0.0, 0.0)

    def test_invalid_a_zero_raises(self):
        """a=0 抛出 ValueError."""
        with pytest.raises(ValueError):
            _extract_params({"a": 0.0, "b": 0.0, "c": 0.0})

    def test_invalid_a_negative_raises(self):
        """a<0 抛出 ValueError."""
        with pytest.raises(ValueError):
            _extract_params({"a": -1.0, "b": 0.0, "c": 0.0})

    def test_invalid_b_too_high_raises(self):
        """b>3 抛出 ValueError."""
        with pytest.raises(ValueError):
            _extract_params({"a": 1.0, "b": 3.5, "c": 0.0})

    def test_invalid_b_too_low_raises(self):
        """b<-3 抛出 ValueError."""
        with pytest.raises(ValueError):
            _extract_params({"a": 1.0, "b": -3.5, "c": 0.0})

    def test_invalid_c_negative_raises(self):
        """c<0 抛出 ValueError."""
        with pytest.raises(ValueError):
            _extract_params({"a": 1.0, "b": 0.0, "c": -0.1})

    def test_invalid_c_too_high_raises(self):
        """c>0.5 抛出 ValueError."""
        with pytest.raises(ValueError):
            _extract_params({"a": 1.0, "b": 0.0, "c": 0.6})

    def test_boundary_values_ok(self):
        """边界值合法: a 略大于 0, b=±3, c=0 与 c=0.5."""
        a, b, c = _extract_params({"a": 0.01, "b": -3.0, "c": 0.0})
        assert (a, b, c) == (0.01, -3.0, 0.0)
        a, b, c = _extract_params({"a": 0.01, "b": 3.0, "c": 0.5})
        assert (a, b, c) == (0.01, 3.0, 0.5)

    def test_update_theta_propagates_validation(self):
        """update_theta 在非法参数下抛出 ValueError."""
        est = IRTEstimator()
        state = IRTState(theta=0.0, se=1.0, response_count=0)
        with pytest.raises(ValueError):
            est.update_theta(state, {"a": -1.0, "b": 0.0, "c": 0.0}, True)

    def test_estimate_mle_propagates_validation(self):
        """estimate_mle 在非法参数下抛出 ValueError."""
        est = IRTEstimator()
        responses = [({"a": 1.0, "b": 5.0, "c": 0.0}, True)]
        with pytest.raises(ValueError):
            est.estimate_mle(responses)


# ============================================================
# 11. IRTEstimator.predict_correct_4pl 测试 (Task 1.3)
# ============================================================


class TestPredictCorrect4PL:
    """IRTEstimator.predict_correct_4pl — 4PL 项目反应函数."""

    def test_method_exists(self):
        """predict_correct_4pl 方法存在."""
        est = IRTEstimator()
        assert hasattr(est, "predict_correct_4pl")
        assert callable(est.predict_correct_4pl)

    def test_default_d_equals_3pl(self):
        """d=1.0 (默认) 时 4PL 退化为 3PL."""
        est = IRTEstimator()
        for theta in (-2.0, 0.0, 2.0):
            p3 = est.predict_correct(theta, a=1.2, b=0.0, c=0.2)
            p4 = est.predict_correct_4pl(theta, a=1.2, b=0.0, c=0.2)
            assert p4 == pytest.approx(p3)

    def test_default_d_is_one(self):
        """未传 d 时默认 1.0."""
        est = IRTEstimator()
        p_default = est.predict_correct_4pl(theta=0.0, a=1.0, b=0.0, c=0.0)
        p_explicit = est.predict_correct_4pl(theta=0.0, a=1.0, b=0.0, c=0.0, d=1.0)
        assert p_default == pytest.approx(p_explicit)

    def test_upper_asymptote_d(self):
        """极高能力时 P 趋近于上渐近线 d (<1)."""
        est = IRTEstimator()
        p = est.predict_correct_4pl(theta=10.0, a=1.0, b=0.0, c=0.1, d=0.9)
        assert p == pytest.approx(0.9, abs=1e-3)

    def test_lower_asymptote_c(self):
        """极低能力时 P 趋近于下渐近线 c."""
        est = IRTEstimator()
        p = est.predict_correct_4pl(theta=-10.0, a=1.0, b=0.0, c=0.2, d=0.9)
        assert p == pytest.approx(0.2, abs=1e-3)

    def test_at_theta_equals_b(self):
        """theta=b 时 P = c + (d-c)*0.5 = (c+d)/2."""
        est = IRTEstimator()
        c, d = 0.2, 0.9
        p = est.predict_correct_4pl(theta=0.0, a=1.0, b=0.0, c=c, d=d)
        assert p == pytest.approx((c + d) / 2.0)

    def test_known_value(self):
        """4PL 已知值: P = c + (d-c)/(1+exp(-a(theta-b)))."""
        est = IRTEstimator()
        a, b, c, d = 1.5, 0.5, 0.1, 0.95
        theta = 1.0
        expected = c + (d - c) / (1.0 + math.exp(-a * (theta - b)))
        p = est.predict_correct_4pl(theta, a, b, c, d)
        assert p == pytest.approx(expected, rel=1e-6)

    def test_in_unit_interval(self):
        """预测答对概率始终落在 [0, 1]."""
        est = IRTEstimator()
        for theta in (-3.0, -1.0, 0.0, 1.0, 3.0):
            for a in (0.8, 1.5, 2.5):
                for b in (-2.0, 0.0, 2.0):
                    for c in (0.0, 0.1, 0.3):
                        for d in (0.7, 0.9, 1.0):
                            p = est.predict_correct_4pl(theta, a, b, c, d)
                            assert 0.0 <= p <= 1.0

    def test_p_between_c_and_d(self):
        """P 始终介于下渐近线 c 与上渐近线 d 之间."""
        est = IRTEstimator()
        c, d = 0.2, 0.8
        for theta in (-3.0, -1.0, 0.0, 1.0, 3.0):
            p = est.predict_correct_4pl(theta, a=1.5, b=0.0, c=c, d=d)
            assert c - 1e-9 <= p <= d + 1e-9

    def test_increasing_theta_increases_p(self):
        """P 随 theta 单调递增 (a>0)."""
        est = IRTEstimator()
        prev = -1.0
        for theta in (-3.0, -1.5, 0.0, 1.5, 3.0):
            p = est.predict_correct_4pl(theta, a=1.2, b=0.0, c=0.15, d=0.85)
            assert p >= prev - 1e-9
            prev = p


# ============================================================
# 12. CATSelector 选题策略测试 (Task 2.1 / 2.2)
# ============================================================


class TestSelectionStrategies:
    """CATSelector 多选题策略 (fisher_info / b_match / kl_info / randomesque)."""

    def test_default_strategy_is_fisher_info(self):
        """默认策略为 fisher_info."""
        selector = CATSelector()
        assert selector.selection_strategy == "fisher_info"

    def test_fisher_info_strategy_matches_default(self):
        """显式 fisher_info 与默认行为一致."""
        items = [
            {"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": 2.0, "c": 0.0},
        ]
        s_default = CATSelector()
        s_fisher = CATSelector(selection_strategy="fisher_info")
        c1 = s_default.select_next(theta=0.0, available_items=items, administered_ids=set())
        c2 = s_fisher.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert c1["item_id"] == c2["item_id"] == "q1"

    def test_b_match_selects_closest_b(self):
        """b_match 选择难度 b 最接近当前 theta 的题目."""
        selector = CATSelector(selection_strategy="b_match")
        items = [
            {"item_id": "q1", "a": 1.0, "b": -2.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": 0.5, "c": 0.0},
            {"item_id": "q3", "a": 1.0, "b": 2.0, "c": 0.0},
        ]
        chosen = selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert chosen is not None
        assert chosen["item_id"] == "q2"

    def test_b_match_ignores_discrimination(self):
        """b_match 仅看难度接近度, 不看区分度 a."""
        selector = CATSelector(selection_strategy="b_match")
        items = [
            {"item_id": "q_low_a", "a": 0.5, "b": 0.0, "c": 0.0},
            {"item_id": "q_high_a", "a": 5.0, "b": 0.5, "c": 0.0},
        ]
        chosen = selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        # b=0.0 比 b=0.5 更接近 theta=0
        assert chosen["item_id"] == "q_low_a"

    def test_b_match_negative_theta(self):
        """b_match 在负 theta 下选择最接近的负难度题."""
        selector = CATSelector(selection_strategy="b_match")
        items = [
            {"item_id": "q1", "a": 1.0, "b": 2.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": -1.5, "c": 0.0},
            {"item_id": "q3", "a": 1.0, "b": 1.0, "c": 0.0},
        ]
        chosen = selector.select_next(theta=-1.0, available_items=items, administered_ids=set())
        assert chosen["item_id"] == "q2"

    def test_kl_info_prefers_b_near_theta(self):
        """kl_info 在 b≈theta 处信息量最大."""
        selector = CATSelector(selection_strategy="kl_info")
        items = [
            {"item_id": "q_far", "a": 1.0, "b": -2.0, "c": 0.0},
            {"item_id": "q_match", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q_far2", "a": 1.0, "b": 2.0, "c": 0.0},
        ]
        chosen = selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert chosen is not None
        assert chosen["item_id"] == "q_match"

    def test_kl_info_scales_with_discrimination(self):
        """难度相同时, kl_info 偏好区分度高的题目."""
        selector = CATSelector(selection_strategy="kl_info")
        items = [
            {"item_id": "q_low_a", "a": 0.8, "b": 0.0, "c": 0.0},
            {"item_id": "q_high_a", "a": 2.0, "b": 0.0, "c": 0.0},
        ]
        chosen = selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert chosen["item_id"] == "q_high_a"

    def test_randomesque_picks_from_top_n(self):
        """randomesque 从 Fisher 信息 top-N 中随机选一个."""
        rng = random.Random(42)
        selector = CATSelector(
            selection_strategy="randomesque", randomesque_n=3, rng=rng
        )
        items = [
            {"item_id": "q0", "a": 2.0, "b": 0.0, "c": 0.0},
            {"item_id": "q1", "a": 1.9, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.8, "b": 0.0, "c": 0.0},
            {"item_id": "q3", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q4", "a": 0.5, "b": 0.0, "c": 0.0},
        ]
        chosen = selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert chosen is not None
        assert chosen["item_id"] in {"q0", "q1", "q2"}

    def test_randomesque_deterministic_with_seed(self):
        """相同随机种子 -> 相同选题 (可复现)."""
        items = [
            {"item_id": "q0", "a": 2.0, "b": 0.0, "c": 0.0},
            {"item_id": "q1", "a": 1.9, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.8, "b": 0.0, "c": 0.0},
            {"item_id": "q3", "a": 1.0, "b": 0.0, "c": 0.0},
        ]
        s1 = CATSelector(
            selection_strategy="randomesque", randomesque_n=3, rng=random.Random(123)
        )
        s2 = CATSelector(
            selection_strategy="randomesque", randomesque_n=3, rng=random.Random(123)
        )
        c1 = s1.select_next(theta=0.0, available_items=items, administered_ids=set())
        c2 = s2.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert c1["item_id"] == c2["item_id"]

    def test_randomesque_n_larger_than_pool(self):
        """randomesque_n 超过候选数时, 全部候选均可能被选."""
        rng = random.Random(7)
        selector = CATSelector(
            selection_strategy="randomesque", randomesque_n=10, rng=rng
        )
        items = [
            {"item_id": "q0", "a": 2.0, "b": 0.0, "c": 0.0},
            {"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0},
        ]
        chosen = selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert chosen["item_id"] in {"q0", "q1"}

    def test_unknown_strategy_raises(self):
        """未知策略抛出 ValueError."""
        selector = CATSelector(selection_strategy="unknown_strategy")
        items = [{"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0}]
        with pytest.raises(ValueError):
            selector.select_next(theta=0.0, available_items=items, administered_ids=set())

    def test_strategy_excludes_administered(self):
        """所有策略均排除已答题目."""
        selector = CATSelector(selection_strategy="b_match")
        items = [
            {"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": 0.5, "c": 0.0},
        ]
        chosen = selector.select_next(
            theta=0.0, available_items=items, administered_ids={"q1"}
        )
        assert chosen["item_id"] == "q2"

    def test_strategy_empty_items_returns_none(self):
        """无可用题目各策略均返回 None."""
        for strategy in ("fisher_info", "b_match", "kl_info"):
            selector = CATSelector(selection_strategy=strategy)
            chosen = selector.select_next(
                theta=0.0, available_items=[], administered_ids=set()
            )
            assert chosen is None


# ============================================================
# 13. CATSelector 内容平衡测试 (Task 2.3)
# ============================================================


class TestContentBalance:
    """CATSelector 内容平衡 (content_constraints)."""

    def test_accepts_content_constraints(self):
        """__init__ 接受 content_constraints 参数."""
        selector = CATSelector(content_constraints={"algebra": 0.5, "geometry": 0.5})
        assert selector.content_constraints == {"algebra": 0.5, "geometry": 0.5}

    def test_no_constraints_default(self):
        """默认 content_constraints 为 None."""
        selector = CATSelector()
        assert selector.content_constraints is None

    def test_prefers_under_target_area(self):
        """已超出目标比例的内容域被抑制, 优先选欠配额域."""
        constraints = {"algebra": 0.5, "geometry": 0.5}
        selector = CATSelector(content_constraints=constraints)
        items = [
            {"item_id": "a1", "a": 2.0, "b": 0.0, "c": 0.0, "content_area": "algebra"},
            {"item_id": "a2", "a": 1.9, "b": 0.0, "c": 0.0, "content_area": "algebra"},
            {"item_id": "g1", "a": 1.0, "b": 0.0, "c": 0.0, "content_area": "geometry"},
            {"item_id": "g2", "a": 0.9, "b": 0.0, "c": 0.0, "content_area": "geometry"},
        ]
        first = selector.select_next(
            theta=0.0, available_items=items, administered_ids=set()
        )
        assert first["item_id"] == "a1"
        # algebra 现占比 1.0 >= 0.5, 第二题应选 geometry
        second = selector.select_next(
            theta=0.0, available_items=items, administered_ids={"a1"}
        )
        assert second["item_id"] in {"g1", "g2"}

    def test_falls_back_when_all_saturated(self):
        """所有内容域均已达标时, 回退到全局最优选题."""
        constraints = {"algebra": 0.5, "geometry": 0.5}
        selector = CATSelector(content_constraints=constraints)
        items = [
            {"item_id": "a1", "a": 2.0, "b": 0.0, "c": 0.0, "content_area": "algebra"},
            {"item_id": "g1", "a": 1.0, "b": 0.0, "c": 0.0, "content_area": "geometry"},
            {"item_id": "a2", "a": 1.8, "b": 0.0, "c": 0.0, "content_area": "algebra"},
            {"item_id": "g2", "a": 0.8, "b": 0.0, "c": 0.0, "content_area": "geometry"},
        ]
        # 第一题 a1, 第二题 g1 -> 1:1 均达标
        selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        selector.select_next(
            theta=0.0, available_items=items, administered_ids={"a1"}
        )
        # 第三题: 两域均饱和, 回退全局最优 -> a2 (info 高于 g2)
        third = selector.select_next(
            theta=0.0, available_items=items, administered_ids={"a1", "g1"}
        )
        assert third is not None
        assert third["item_id"] == "a2"

    def test_no_constraints_ignores_content_area(self):
        """无约束时 content_area 不影响选题."""
        selector = CATSelector()
        items = [
            {"item_id": "a1", "a": 2.0, "b": 0.0, "c": 0.0, "content_area": "algebra"},
            {"item_id": "g1", "a": 1.0, "b": 0.0, "c": 0.0, "content_area": "geometry"},
        ]
        chosen = selector.select_next(
            theta=0.0, available_items=items, administered_ids=set()
        )
        assert chosen["item_id"] == "a1"

    def test_unconstrained_items_always_allowed(self):
        """无 content_area 的题目不受内容平衡限制."""
        constraints = {"algebra": 0.5, "geometry": 0.5}
        selector = CATSelector(content_constraints=constraints)
        items = [
            {"item_id": "a1", "a": 2.0, "b": 0.0, "c": 0.0, "content_area": "algebra"},
            {"item_id": "x1", "a": 2.5, "b": 0.0, "c": 0.0},  # 无 content_area
        ]
        # 即便 algebra 未达标, 无 content_area 的高信息题仍可选
        chosen = selector.select_next(
            theta=0.0, available_items=items, administered_ids=set()
        )
        assert chosen["item_id"] == "x1"

    def test_content_balance_with_b_match_strategy(self):
        """内容平衡与 b_match 策略组合工作."""
        constraints = {"algebra": 0.5, "geometry": 0.5}
        selector = CATSelector(
            selection_strategy="b_match", content_constraints=constraints
        )
        items = [
            {"item_id": "a1", "a": 1.0, "b": 0.0, "c": 0.0, "content_area": "algebra"},
            {"item_id": "g1", "a": 1.0, "b": 0.1, "c": 0.0, "content_area": "geometry"},
        ]
        first = selector.select_next(
            theta=0.0, available_items=items, administered_ids=set()
        )
        assert first["item_id"] == "a1"
        # algebra 饱和后, b_match 在 geometry 域选题
        second = selector.select_next(
            theta=0.0, available_items=items, administered_ids={"a1"}
        )
        assert second["item_id"] == "g1"


# ============================================================
# 14. CATSelector.should_stop min_items 测试 (Task 2.4)
# ============================================================


class TestShouldStopMinItems:
    """CATSelector.should_stop — min_items 参数."""

    def test_min_items_prevents_se_termination(self):
        """SE 已达标但题数不足 min_items -> 继续."""
        selector = CATSelector()
        assert selector.should_stop(current_se=0.1, count=3, min_items=5) is False

    def test_min_items_allows_se_termination_when_met(self):
        """SE 达标且题数 >= min_items -> 终止."""
        selector = CATSelector()
        assert selector.should_stop(current_se=0.1, count=5, min_items=5) is True

    def test_min_items_default_zero(self):
        """默认 min_items=0, SE 达标即终止."""
        selector = CATSelector()
        assert selector.should_stop(current_se=0.1, count=1) is True

    def test_min_items_does_not_affect_max_items(self):
        """max_items 终止不受 min_items 影响."""
        selector = CATSelector()
        # count >= max_items 即使 < min_items 也终止
        assert selector.should_stop(
            current_se=0.1, count=20, max_items=20, min_items=30
        ) is True

    def test_min_items_above_count_continues(self):
        """题数远低于 min_items 且 SE 未达标 -> 继续."""
        selector = CATSelector()
        assert selector.should_stop(current_se=0.5, count=2, min_items=10) is False

    def test_min_items_keyword_only(self):
        """min_items 可作为关键字参数传入 (向后兼容位置参数)."""
        selector = CATSelector()
        # 旧式位置参数调用仍有效
        assert selector.should_stop(0.5, 10) is False
        # 新增 min_items 关键字
        assert selector.should_stop(0.1, 3, min_items=5) is False


# ============================================================
# 15. CATSelector 曝光追踪测试 (Task 2.5)
# ============================================================


class TestExposureTracking:
    """CATSelector.get_exposure_stats — 曝光追踪."""

    def test_method_exists(self):
        """get_exposure_stats 方法存在."""
        selector = CATSelector()
        assert hasattr(selector, "get_exposure_stats")
        assert callable(selector.get_exposure_stats)

    def test_empty_initially(self):
        """初始曝光统计为空."""
        selector = CATSelector()
        assert selector.get_exposure_stats() == {}

    def test_tracks_single_selection(self):
        """单次选题后曝光统计记录该题."""
        selector = CATSelector()
        items = [
            {"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": 2.0, "c": 0.0},
        ]
        selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert selector.get_exposure_stats() == {"q1": 1}

    def test_tracks_multiple_selections(self):
        """多次选题累计记录各题曝光次数."""
        selector = CATSelector()
        items = [
            {"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": 2.0, "c": 0.0},
        ]
        selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        selector.select_next(
            theta=0.0, available_items=items, administered_ids={"q1"}
        )
        assert selector.get_exposure_stats() == {"q1": 1, "q2": 1}

    def test_increments_on_repeat(self):
        """同一题被多次选中, 计数递增."""
        selector = CATSelector()
        items = [{"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0}]
        selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        assert selector.get_exposure_stats() == {"q1": 3}

    def test_returns_copy(self):
        """返回的字典为副本, 修改不影响内部状态."""
        selector = CATSelector()
        items = [{"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0}]
        selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        stats = selector.get_exposure_stats()
        stats["q1"] = 999
        stats["injected"] = 1
        assert selector.get_exposure_stats() == {"q1": 1}

    def test_no_selection_when_none_returned(self):
        """返回 None (无可用题) 时不记录曝光."""
        selector = CATSelector()
        items = [{"item_id": "q1", "a": 1.0, "b": 0.0, "c": 0.0}]
        result = selector.select_next(
            theta=0.0, available_items=items, administered_ids={"q1"}
        )
        assert result is None
        assert selector.get_exposure_stats() == {}

    def test_exposure_tracked_across_strategies(self):
        """不同策略下曝光追踪均生效."""
        rng = random.Random(0)
        selector = CATSelector(selection_strategy="randomesque", rng=rng)
        items = [
            {"item_id": "q1", "a": 2.0, "b": 0.0, "c": 0.0},
            {"item_id": "q2", "a": 1.0, "b": 0.0, "c": 0.0},
        ]
        selector.select_next(theta=0.0, available_items=items, administered_ids=set())
        stats = selector.get_exposure_stats()
        assert sum(stats.values()) == 1
        assert set(stats.keys()).issubset({"q1", "q2"})
