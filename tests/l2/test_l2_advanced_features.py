"""L2 高级特性测试 — 冷启动策略 / 概念漂移检测 / ZPD 量化.

测试覆盖 (TDD):
1. LearnerColdStartManager: 冷启动判定 / 策略名 / theta 混合估计 / mastery 混合估计 /
   默认学习风格 / 初始内容推荐.
2. LearnerDriftDetector: 稳定无漂移 / 突然下滑检测 / DDM warning 先于 drift /
   reset 清空状态 / get_stats 统计信息.
3. ZPDCalculator: ZPD 边界计算 / 难度推荐落在 ZPD 内 / 题目区域分类.
"""

from __future__ import annotations

import pytest

from dy3_polaris.l2.ability_assessor import ZPDCalculator, ZPDResult
from dy3_polaris.l2.profile_builder import LearnerColdStartManager, LearnerDriftDetector


# ============================================================
# 1. LearnerColdStartManager
# ============================================================


class TestLearnerColdStartManagerDetection:
    """LearnerColdStartManager 冷启动判定与策略名."""

    def test_is_cold_start_true_for_less_than_10(self):
        """记录数 < 10 视为冷启动."""
        mgr = LearnerColdStartManager()
        assert mgr.is_cold_start(0) is True
        assert mgr.is_cold_start(1) is True
        assert mgr.is_cold_start(5) is True
        assert mgr.is_cold_start(9) is True

    def test_is_cold_start_false_for_10_or_more(self):
        """记录数 >= 10 不再冷启动."""
        mgr = LearnerColdStartManager()
        assert mgr.is_cold_start(10) is False
        assert mgr.is_cold_start(15) is False
        assert mgr.is_cold_start(100) is False

    def test_is_cold_start_custom_threshold(self):
        """自定义阈值生效."""
        mgr = LearnerColdStartManager(cold_start_threshold=5)
        assert mgr.is_cold_start(4) is True
        assert mgr.is_cold_start(5) is False

    def test_get_strategy_population_average(self):
        """0 条记录 -> population_average."""
        mgr = LearnerColdStartManager()
        assert mgr.get_strategy(0) == "population_average"

    def test_get_strategy_partial_personalization(self):
        """1-9 条记录 -> partial_personalization."""
        mgr = LearnerColdStartManager()
        assert mgr.get_strategy(1) == "partial_personalization"
        assert mgr.get_strategy(5) == "partial_personalization"
        assert mgr.get_strategy(9) == "partial_personalization"

    def test_get_strategy_full_personalization(self):
        """10+ 条记录 -> full_personalization."""
        mgr = LearnerColdStartManager()
        assert mgr.get_strategy(10) == "full_personalization"
        assert mgr.get_strategy(50) == "full_personalization"


class TestLearnerColdStartManagerTheta:
    """LearnerColdStartManager.estimate_initial_theta 群体先验与观测值混合."""

    def test_zero_records_returns_population_prior(self):
        """0 条记录 -> 群体先验 (theta=0.0, se=0.5)."""
        mgr = LearnerColdStartManager()
        theta, se = mgr.estimate_initial_theta(observed_theta=1.5, record_count=0)
        assert theta == pytest.approx(0.0)
        assert se == pytest.approx(0.5)

    def test_none_observed_returns_population_prior(self):
        """观测值为 None -> 群体先验."""
        mgr = LearnerColdStartManager()
        theta, se = mgr.estimate_initial_theta(observed_theta=None, record_count=5)
        assert theta == pytest.approx(0.0)
        assert se == pytest.approx(0.5)

    def test_partial_blends_population_and_observed(self):
        """5 条记录 (weight=0.5) -> 加权混合, 不是纯观测值也不是纯群体值."""
        mgr = LearnerColdStartManager()
        # weight = 5/10 = 0.5
        # theta = (1-0.5)*0.0 + 0.5*1.0 = 0.5
        # se    = 0.5*(1-0.5) + 0.3*0.5 = 0.25 + 0.15 = 0.4
        theta, se = mgr.estimate_initial_theta(observed_theta=1.0, record_count=5)
        assert theta == pytest.approx(0.5)
        assert se == pytest.approx(0.4)
        # 混合: 介于群体先验 (0.0) 与观测值 (1.0) 之间
        assert 0.0 < theta < 1.0

    def test_blending_weight_scales_with_records(self):
        """记录数越多, theta 越接近观测值."""
        mgr = LearnerColdStartManager()
        t_few, _ = mgr.estimate_initial_theta(observed_theta=2.0, record_count=2)
        t_more, _ = mgr.estimate_initial_theta(observed_theta=2.0, record_count=8)
        assert t_few < t_more < 2.0

    def test_full_personalization_returns_observed(self):
        """10+ 条记录 -> weight=1.0, theta=观测值, se=0.3."""
        mgr = LearnerColdStartManager()
        theta, se = mgr.estimate_initial_theta(observed_theta=1.5, record_count=10)
        assert theta == pytest.approx(1.5)
        assert se == pytest.approx(0.3)

    def test_se_decreases_with_more_records(self):
        """记录数越多, 标准误越小."""
        mgr = LearnerColdStartManager()
        _, se_few = mgr.estimate_initial_theta(observed_theta=0.5, record_count=2)
        _, se_more = mgr.estimate_initial_theta(observed_theta=0.5, record_count=9)
        assert se_few > se_more

    def test_returns_tuple_of_two_floats(self):
        """返回 (theta, se) 二元组且均为 float."""
        mgr = LearnerColdStartManager()
        result = mgr.estimate_initial_theta(observed_theta=0.3, record_count=4)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestLearnerColdStartManagerMastery:
    """LearnerColdStartManager.estimate_initial_mastery 群体先验与观测值混合."""

    def test_zero_records_returns_population_mastery(self):
        """0 条记录 -> 群体平均掌握度 0.5."""
        mgr = LearnerColdStartManager()
        assert mgr.estimate_initial_mastery(observed_mastery=0.9, record_count=0) == pytest.approx(0.5)

    def test_none_observed_returns_population_mastery(self):
        """观测值为 None -> 群体平均掌握度."""
        mgr = LearnerColdStartManager()
        assert mgr.estimate_initial_mastery(observed_mastery=None, record_count=5) == pytest.approx(0.5)

    def test_partial_blends_population_and_observed(self):
        """5 条记录 (weight=0.5) -> 0.5*0.5 + 0.5*0.9 = 0.7."""
        mgr = LearnerColdStartManager()
        m = mgr.estimate_initial_mastery(observed_mastery=0.9, record_count=5)
        assert m == pytest.approx(0.7)
        assert 0.5 < m < 0.9

    def test_full_personalization_returns_observed(self):
        """10+ 条记录 -> 观测掌握度."""
        mgr = LearnerColdStartManager()
        m = mgr.estimate_initial_mastery(observed_mastery=0.85, record_count=12)
        assert m == pytest.approx(0.85)


class TestLearnerColdStartManagerStyleAndContent:
    """LearnerColdStartManager 默认学习风格与初始内容推荐."""

    def test_default_learning_style_is_multimodal(self):
        """冷启动默认学习风格为 multimodal (多模态)."""
        mgr = LearnerColdStartManager()
        assert mgr.get_default_learning_style() == "multimodal"

    def test_recommend_initial_content_returns_first_five(self):
        """available_kps 超过 5 个时返回前 5 个."""
        mgr = LearnerColdStartManager()
        kps = ["kp-1", "kp-2", "kp-3", "kp-4", "kp-5", "kp-6", "kp-7"]
        result = mgr.recommend_initial_content(record_count=0, available_kps=kps)
        assert result == ["kp-1", "kp-2", "kp-3", "kp-4", "kp-5"]
        assert len(result) == 5

    def test_recommend_initial_content_fewer_than_five(self):
        """available_kps 少于 5 个时返回全部."""
        mgr = LearnerColdStartManager()
        kps = ["kp-A", "kp-B"]
        result = mgr.recommend_initial_content(record_count=0, available_kps=kps)
        assert result == ["kp-A", "kp-B"]

    def test_recommend_initial_content_none(self):
        """available_kps=None -> 空列表."""
        mgr = LearnerColdStartManager()
        assert mgr.recommend_initial_content(record_count=0, available_kps=None) == []

    def test_recommend_initial_content_empty(self):
        """available_kps=[] -> 空列表."""
        mgr = LearnerColdStartManager()
        assert mgr.recommend_initial_content(record_count=0, available_kps=[]) == []


# ============================================================
# 2. LearnerDriftDetector
# ============================================================


class TestLearnerDriftDetectorStable:
    """LearnerDriftDetector 稳定表现无漂移."""

    def test_no_drift_for_stable_all_correct(self):
        """稳定全对表现 -> 无漂移."""
        det = LearnerDriftDetector()
        for _ in range(30):
            result = det.add_observation(1.0)
            assert result["drift_detected"] is False

    def test_no_warning_for_stable_all_correct(self):
        """稳定全对表现 -> 无 warning."""
        det = LearnerDriftDetector()
        for _ in range(30):
            result = det.add_observation(1.0)
            assert result["warning"] is False

    def test_stable_result_has_correct_keys(self):
        """返回结果含必需键."""
        det = LearnerDriftDetector()
        result = det.add_observation(1.0)
        for key in ("drift_detected", "method", "warning", "value", "window_mean"):
            assert key in result


class TestLearnerDriftDetectorSuddenDrop:
    """LearnerDriftDetector 突然下滑检测漂移."""

    def test_detects_drift_when_performance_drops(self):
        """稳定全对后突然全错 -> 检测到漂移."""
        det = LearnerDriftDetector()
        # 先建立稳定全对历史
        for _ in range(15):
            det.add_observation(1.0)
        # 突然下滑
        drift_seen = False
        for _ in range(10):
            result = det.add_observation(0.0)
            if result["drift_detected"]:
                drift_seen = True
                break
        assert drift_seen is True

    def test_drift_result_method_is_set(self):
        """检测到漂移时 method 为 'adwin' 或 'ddm'."""
        det = LearnerDriftDetector()
        for _ in range(15):
            det.add_observation(1.0)
        method = None
        for _ in range(10):
            result = det.add_observation(0.0)
            if result["drift_detected"]:
                method = result["method"]
                break
        assert method in ("adwin", "ddm")


class TestLearnerDriftDetectorDDMWarningBeforeDrift:
    """LearnerDriftDetector DDM warning 先于 drift."""

    def test_warning_before_drift(self):
        """渐变漂移序列: DDM warning 出现在 drift 之前.

        序列设计 (经算法仿真验证):
        - 前 10 条: [0,0,0,1,1,1,1,1,1,1] (3 错 7 对), 在 n=10 建立
          min_p=0.3, min_ps≈0.145 (非零, 避免完美 streak 导致 min_ps=0).
        - 后 9 条: 全错, p+s 逐步上升, 先越过 warning 阈值
          (min_p + 2*min_ps ≈ 0.59) 再越过 drift 阈值
          (min_p + 3*min_ps ≈ 0.735), 且 ADWIN 不抢先触发.
        """
        det = LearnerDriftDetector()
        seq = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        results = [det.add_observation(v) for v in seq]

        # 至少存在一个 warning-only (warning=True 且 drift=False)
        warning_only_indices = [
            i for i, r in enumerate(results)
            if r["warning"] and not r["drift_detected"]
        ]
        drift_indices = [
            i for i, r in enumerate(results) if r["drift_detected"]
        ]
        assert len(warning_only_indices) > 0, "应至少出现一次 warning-only"
        assert len(drift_indices) > 0, "应最终检测到漂移"
        # warning-only 出现在 drift 之前
        assert min(warning_only_indices) < min(drift_indices)

    def test_warning_only_does_not_report_drift(self):
        """warning-only 阶段 drift_detected 必为 False."""
        det = LearnerDriftDetector()
        seq = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for v in seq:
            r = det.add_observation(v)
            # warning 可为 True, 但若未 drift 则 drift_detected 必为 False
            if r["warning"] and not r["drift_detected"]:
                assert r["drift_detected"] is False
                assert r["method"] is None


class TestLearnerDriftDetectorReset:
    """LearnerDriftDetector.reset 清空状态."""

    def test_reset_clears_window(self):
        """reset 后窗口清空."""
        det = LearnerDriftDetector()
        for _ in range(20):
            det.add_observation(1.0)
        assert len(det._window) > 0
        det.reset()
        assert len(det._window) == 0

    def test_reset_clears_observation_count(self):
        """reset 后观测计数归零."""
        det = LearnerDriftDetector()
        for _ in range(15):
            det.add_observation(1.0)
        assert det._ddm_count == 15
        det.reset()
        assert det._ddm_count == 0

    def test_reset_clears_ddm_min(self):
        """reset 后 DDM 最小值重置为 inf."""
        det = LearnerDriftDetector()
        for _ in range(15):
            det.add_observation(1.0)
        det.reset()
        assert det._ddm_min_p == float("inf")
        assert det._ddm_min_ps == float("inf")

    def test_reset_get_stats_clean(self):
        """reset 后 get_stats 反映空状态."""
        det = LearnerDriftDetector()
        for _ in range(15):
            det.add_observation(1.0)
        det.reset()
        stats = det.get_stats()
        assert stats["window_size"] == 0
        assert stats["observation_count"] == 0
        assert stats["window_mean"] == 0.0


class TestLearnerDriftDetectorStats:
    """LearnerDriftDetector.get_stats 统计信息."""

    def test_get_stats_keys(self):
        """get_stats 返回必需键."""
        det = LearnerDriftDetector()
        for _ in range(10):
            det.add_observation(1.0)
        stats = det.get_stats()
        for key in ("window_size", "window_mean", "ddm_min_p", "observation_count"):
            assert key in stats

    def test_get_stats_after_mixed_observations(self):
        """混合观测后 get_stats 数值正确.

        序列 [1,0,1,0,1,0,1,0,1,0]: n=10, p=0.5, s≈0.158,
        min_p=0.5, window_mean=0.5, window_size=10, observation_count=10.
        """
        det = LearnerDriftDetector()
        seq = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        for v in seq:
            det.add_observation(v)
        stats = det.get_stats()
        assert stats["window_size"] == 10
        assert stats["observation_count"] == 10
        assert stats["window_mean"] == pytest.approx(0.5)
        assert stats["ddm_min_p"] == pytest.approx(0.5)

    def test_get_stats_ddm_min_p_zero_when_inf(self):
        """未达到 DDM 最小记录数 (n<10) 时 ddm_min_p 报告为 0.0."""
        det = LearnerDriftDetector()
        det.add_observation(1.0)
        det.add_observation(0.0)
        stats = det.get_stats()
        assert stats["ddm_min_p"] == 0.0
        assert stats["observation_count"] == 2


# ============================================================
# 3. ZPDCalculator
# ============================================================


def _make_item_bank() -> list[dict]:
    """构造测试题库: b ∈ {-2,-1,0,1,2}, a=1.0, c=0.0."""
    return [
        {"item_id": "i1", "difficulty_b": -2.0, "discrimination_a": 1.0, "guessing_c": 0.0},
        {"item_id": "i2", "difficulty_b": -1.0, "discrimination_a": 1.0, "guessing_c": 0.0},
        {"item_id": "i3", "difficulty_b": 0.0, "discrimination_a": 1.0, "guessing_c": 0.0},
        {"item_id": "i4", "difficulty_b": 1.0, "discrimination_a": 1.0, "guessing_c": 0.0},
        {"item_id": "i5", "difficulty_b": 2.0, "discrimination_a": 1.0, "guessing_c": 0.0},
    ]


class TestZPDCalculatorCalculate:
    """ZPDCalculator.calculate_zpd 边界计算.

    theta=2.0, 题库 b∈{-2,-1,0,1,2}, a=1.0, c=0.0.
    P(theta=2,b) = 1/(1+exp(b-2)):
      b=-2 -> P≈0.982 (>0.9 independent)
      b=-1 -> P≈0.953 (>0.9 independent, 最后一个 >0.9)
      b= 0 -> P≈0.881 (zpd)
      b= 1 -> P≈0.731 (zpd)
      b= 2 -> P=0.500 (zpd, 最后一个 >0.3)
    故 independent_threshold=-1, zpd_upper=2, optimal=(-1+2)/2=0.5.
    """

    def test_calculate_zpd_returns_zpd_result(self):
        """calculate_zpd 返回 ZPDResult 实例."""
        calc = ZPDCalculator()
        result = calc.calculate_zpd(theta=2.0, item_bank=_make_item_bank())
        assert isinstance(result, ZPDResult)

    def test_independent_threshold(self):
        """独立区上界 = 最后一个 P>0.9 的题目难度 = -1.0."""
        calc = ZPDCalculator()
        result = calc.calculate_zpd(theta=2.0, item_bank=_make_item_bank())
        assert result.independent_threshold == pytest.approx(-1.0)

    def test_zpd_lower_equals_independent_threshold(self):
        """ZPD 下界 = 独立区上界."""
        calc = ZPDCalculator()
        result = calc.calculate_zpd(theta=2.0, item_bank=_make_item_bank())
        assert result.zpd_lower == pytest.approx(-1.0)

    def test_zpd_upper(self):
        """ZPD 上界 = 最后一个 P>0.3 的题目难度 = 2.0."""
        calc = ZPDCalculator()
        result = calc.calculate_zpd(theta=2.0, item_bank=_make_item_bank())
        assert result.zpd_upper == pytest.approx(2.0)

    def test_optimal_difficulty_is_center(self):
        """最优难度 = ZPD 中心 = (zpd_lower + zpd_upper)/2 = 0.5."""
        calc = ZPDCalculator()
        result = calc.calculate_zpd(theta=2.0, item_bank=_make_item_bank())
        assert result.optimal_difficulty == pytest.approx(0.5)

    def test_recommended_difficulty_equals_optimal(self):
        """推荐难度默认 = 最优难度."""
        calc = ZPDCalculator()
        result = calc.calculate_zpd(theta=2.0, item_bank=_make_item_bank())
        assert result.recommended_difficulty == pytest.approx(result.optimal_difficulty)

    def test_calculate_zpd_empty_item_bank(self):
        """空题库 -> 解析逆解给出明确定义的 ZPD 区间 (非零宽), 不再退化为 (0.5,0.5)."""
        calc = ZPDCalculator()
        result = calc.calculate_zpd(theta=0.0, item_bank=[])
        # 解析边界: b = theta + ln((1-P)/(P-c))/a (theta=0, a=1, c=0)
        # 下界 P=0.9 → ln(0.1/0.9)≈-2.197; 上界 P=0.3 → ln(0.7/0.3)≈0.847
        assert result.independent_threshold == pytest.approx(-2.19722, abs=1e-3)
        assert result.zpd_lower == result.independent_threshold
        assert result.frustration_threshold == pytest.approx(0.84730, abs=1e-3)
        assert result.zpd_upper == result.frustration_threshold
        assert result.zpd_lower < result.zpd_upper  # 非零宽区间
        assert result.optimal_difficulty == pytest.approx((result.zpd_lower + result.zpd_upper) / 2.0)
        assert result.recommended_difficulty == result.optimal_difficulty

    def test_frustration_threshold(self):
        """挫折区阈值 = 最后一个 P>0.3 的题目难度 = 2.0 (与 zpd_upper 一致)."""
        calc = ZPDCalculator()
        result = calc.calculate_zpd(theta=2.0, item_bank=_make_item_bank())
        assert result.frustration_threshold == pytest.approx(2.0)


class TestZPDCalculatorRecommendDifficulty:
    """ZPDCalculator.recommend_difficulty 推荐难度落在 ZPD 内."""

    def test_recommend_in_zpd_mid_scaffold(self):
        """scaffold=0.5 -> 推荐难度 = zpd_lower + 0.5*(zpd_upper-zpd_lower) = 0.5."""
        calc = ZPDCalculator()
        rec = calc.recommend_difficulty(
            theta=2.0, scaffold_level=0.5, item_bank=_make_item_bank()
        )
        assert rec == pytest.approx(0.5)

    def test_recommend_in_zpd_range(self):
        """推荐难度落在 [zpd_lower, zpd_upper] 区间内."""
        calc = ZPDCalculator()
        bank = _make_item_bank()
        zpd = calc.calculate_zpd(theta=2.0, item_bank=bank)
        for scaffold in (0.0, 0.25, 0.5, 0.75, 1.0):
            rec = calc.recommend_difficulty(theta=2.0, scaffold_level=scaffold, item_bank=bank)
            assert zpd.zpd_lower - 1e-9 <= rec <= zpd.zpd_upper + 1e-9

    def test_recommend_scaffold_zero_is_lower_bound(self):
        """scaffold=0.0 -> 推荐难度 = zpd_lower."""
        calc = ZPDCalculator()
        bank = _make_item_bank()
        zpd = calc.calculate_zpd(theta=2.0, item_bank=bank)
        rec = calc.recommend_difficulty(theta=2.0, scaffold_level=0.0, item_bank=bank)
        assert rec == pytest.approx(zpd.zpd_lower)

    def test_recommend_scaffold_one_is_upper_bound(self):
        """scaffold=1.0 -> 推荐难度 = zpd_upper."""
        calc = ZPDCalculator()
        bank = _make_item_bank()
        zpd = calc.calculate_zpd(theta=2.0, item_bank=bank)
        rec = calc.recommend_difficulty(theta=2.0, scaffold_level=1.0, item_bank=bank)
        assert rec == pytest.approx(zpd.zpd_upper)

    def test_recommend_no_item_bank_fallback(self):
        """无题库 -> 回退 theta + 0.5*scaffold_level."""
        calc = ZPDCalculator()
        rec = calc.recommend_difficulty(theta=0.0, scaffold_level=0.5, item_bank=None)
        assert rec == pytest.approx(0.25)


class TestZPDCalculatorClassifyItem:
    """ZPDCalculator.classify_item 题目区域分类.

    theta=0.0, a=1.0, c=0.0: P = 1/(1+exp(b)).
      b=-3 -> P≈0.953 (>0.9 -> independent)
      b= 0 -> P=0.500 (zpd)
      b= 2 -> P≈0.119 (<=0.3 -> frustration)
    """

    def test_classify_independent(self):
        """高正确率题目 -> independent."""
        calc = ZPDCalculator()
        assert calc.classify_item(theta=0.0, difficulty_b=-3.0) == "independent"

    def test_classify_zpd(self):
        """中等正确率题目 -> zpd."""
        calc = ZPDCalculator()
        assert calc.classify_item(theta=0.0, difficulty_b=0.0) == "zpd"

    def test_classify_frustration(self):
        """低正确率题目 -> frustration."""
        calc = ZPDCalculator()
        assert calc.classify_item(theta=0.0, difficulty_b=2.0) == "frustration"

    def test_classify_returns_valid_zone(self):
        """分类结果始终为三种区域之一."""
        calc = ZPDCalculator()
        for b in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
            zone = calc.classify_item(theta=0.0, difficulty_b=b)
            assert zone in ("independent", "zpd", "frustration")

    def test_classify_high_ability_makes_hard_items_zpd(self):
        """高能力者: 原本 frustration 的题目进入 zpd."""
        calc = ZPDCalculator()
        # theta=2.0, b=2.0 -> P=0.5 -> zpd
        assert calc.classify_item(theta=2.0, difficulty_b=2.0) == "zpd"
        # theta=3.0, b=2.0 -> P≈0.731 -> zpd
        assert calc.classify_item(theta=3.0, difficulty_b=2.0) == "zpd"
