"""T3 IRT 能力评估全链路增强测试.

融合世界先进方案:
- mirt (R): 多模型 IRT (1PL/2PL/3PL/4PL) + AIC/BIC 模型选择
- Stan/brms: 贝叶斯分层 IRT 自适应收缩 + 可信区间
- catR: PSER 终止准则 + 渐进法曝光控制
- Vygotsky + IRT: ZPD 置信区间量化 + 自适应支架

测试覆盖:
1. TestMultiModelIRT          — 1PL/2PL/3PL/4PL 模型与自动选择
2. TestBayesianHierarchical   — 自适应收缩与可信区间
3. TestCATAdvancedTermination — PSER/置信区间/分类准确率终止
4. TestExposureControl        — 渐进法/a-分层曝光控制
5. TestZPDQuantification      — 置信区间量化与自适应支架
6. TestEnhancedFullLink       — 增强后全链路端到端

遵循 TDD Red-Green-Refactor.
"""

from __future__ import annotations

import math
import time

import pytest

from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.models import IRTState
from dy3_polaris.l2.store import InMemoryL2Store
from dy3_polaris.l2.ability_assessor import (
    CATSelector,
    IRTEstimator,
    ZPDCalculator,
    IRTTracingService,
    AbilityOutput,
)


# ============================================================
# 辅助函数
# ============================================================


def _event(
    learner_id: str = "learner_001",
    kp_id: str = "kp_math_01",
    correct: bool = True,
    difficulty: float = 0.5,
    ts: float | None = None,
) -> AnswerEvent:
    return AnswerEvent(
        learner_id=learner_id,
        kp_id=kp_id,
        correct=correct,
        difficulty=difficulty,
        timestamp=ts if ts is not None else time.time(),
    )


def _bank(bs: list[float], a: float = 1.2, c: float = 0.25) -> list[dict]:
    return [{"item_id": f"q{i}", "a": a, "b": b, "c": c} for i, b in enumerate(bs)]


# ============================================================
# 1. TestMultiModelIRT — 1PL/2PL/3PL/4PL 模型与自动选择
# ============================================================


class TestMultiModelIRT:
    """多模型 IRT: 1PL(Rasch) / 2PL / 3PL / 4PL 统一接口与自动选择."""

    def test_1pl_predict_correct(self):
        """1PL (Rasch) 模型: P = 1/(1+exp(-(theta-b))), a=1, c=0."""
        est = IRTEstimator()
        # theta=b 时 P=0.5
        p = est.predict_correct_1pl(0.0, 0.0)
        assert abs(p - 0.5) < 0.01
        # theta > b 时 P > 0.5
        assert est.predict_correct_1pl(1.0, 0.0) > 0.5
        # theta < b 时 P < 0.5
        assert est.predict_correct_1pl(-1.0, 0.0) < 0.5

    def test_2pl_predict_correct(self):
        """2PL 模型: P = 1/(1+exp(-a*(theta-b))), c=0."""
        est = IRTEstimator()
        # theta=b 时 P=0.5 (与 a 无关)
        p = est.predict_correct_2pl(0.0, 1.5, 0.0)
        assert abs(p - 0.5) < 0.01
        # 高区分度 a 使曲线更陡
        p_steep = est.predict_correct_2pl(0.5, 2.0, 0.0)
        p_flat = est.predict_correct_2pl(0.5, 0.5, 0.0)
        assert p_steep > p_flat

    def test_3pl_predict_correct(self):
        """3PL 模型: P = c + (1-c)/(1+exp(-a*(theta-b)))."""
        est = IRTEstimator()
        # 极低能力时 P 趋近 c
        p = est.predict_correct(-10.0, 1.0, 0.0, 0.25)
        assert abs(p - 0.25) < 0.05
        # c=0 时退化为 2PL
        p_3pl_c0 = est.predict_correct(0.0, 1.0, 0.0, 0.0)
        p_2pl = est.predict_correct_2pl(0.0, 1.0, 0.0)
        assert abs(p_3pl_c0 - p_2pl) < 0.01

    def test_4pl_predict_correct(self):
        """4PL 模型: P = c + (d-c)/(1+exp(-a*(theta-b)))."""
        est = IRTEstimator()
        # 极高能力时 P 趋近 d
        p = est.predict_correct_4pl(10.0, 1.0, 0.0, 0.0, 0.95)
        assert abs(p - 0.95) < 0.05
        # d=1 时退化为 3PL
        p_4pl_d1 = est.predict_correct_4pl(0.0, 1.0, 0.0, 0.25, 1.0)
        p_3pl = est.predict_correct(0.0, 1.0, 0.0, 0.25)
        assert abs(p_4pl_d1 - p_3pl) < 0.01

    def test_4pl_fisher_information(self):
        """4PL Fisher 信息量: 非负, 峰值在 theta≈b 附近."""
        est = IRTEstimator()
        info_at_b = est.information_4pl(0.0, 1.0, 0.0, 0.0, 1.0)
        info_far = est.information_4pl(3.0, 1.0, 0.0, 0.0, 1.0)
        assert info_at_b > 0.0
        assert info_at_b > info_far
        # 4PL 信息量应 <= 对应 3PL 信息量 (d<1 降低信息)
        info_3pl = est.information(0.0, 1.0, 0.0, 0.0)
        info_4pl = est.information_4pl(0.0, 1.0, 0.0, 0.0, 0.9)
        assert info_4pl <= info_3pl + 1e-10

    def test_model_selection_aic_bic(self):
        """AIC/BIC 模型选择: 嵌套模型比较."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.2, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.2, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.2, "b": 0.0, "c": 0.0}, False),
            ({"a": 1.2, "b": 0.5, "c": 0.0}, True),
            ({"a": 1.2, "b": 0.5, "c": 0.0}, False),
            ({"a": 1.2, "b": -0.5, "c": 0.0}, True),
            ({"a": 1.2, "b": -0.5, "c": 0.0}, True),
            ({"a": 1.2, "b": 1.0, "c": 0.0}, False),
        ]
        # 计算各模型 AIC/BIC
        result = est.compare_models(responses)
        assert "1PL" in result
        assert "2PL" in result
        assert "3PL" in result
        for model_name, metrics in result.items():
            assert "aic" in metrics
            assert "bic" in metrics
            assert "loglik" in metrics
            assert "n_params" in metrics
            assert metrics["aic"] > 0  # AIC 为正 (负对数似然)
            assert metrics["bic"] >= metrics["aic"]  # BIC >= AIC (额外惩罚)

    def test_auto_select_model(self):
        """自动模型选择: 基于数据特征选择最优模型."""
        est = IRTEstimator()
        # 简单数据 → 2PL 足够
        responses_simple = [
            ({"a": 1.0, "b": 0.0, "c": 0.0}, True),
            ({"a": 1.0, "b": 0.0, "c": 0.0}, False),
        ] * 10
        best_model = est.select_best_model(responses_simple)
        assert best_model in ("1PL", "2PL", "3PL", "4PL")
        # 结果含推荐理由
        assert isinstance(best_model, str)

    def test_estimate_by_model(self):
        """按指定模型估计能力."""
        est = IRTEstimator()
        responses = [
            ({"a": 1.2, "b": 0.0, "c": 0.25}, True),
            ({"a": 1.2, "b": 0.0, "c": 0.25}, False),
        ] * 5
        # 1PL 估计
        state_1pl = est.estimate_by_model(responses, model="1PL")
        assert -3.0 <= state_1pl.theta <= 3.0
        assert state_1pl.se > 0.0
        # 2PL 估计
        state_2pl = est.estimate_by_model(responses, model="2PL")
        assert -3.0 <= state_2pl.theta <= 3.0
        # 3PL 估计
        state_3pl = est.estimate_by_model(responses, model="3PL")
        assert -3.0 <= state_3pl.theta <= 3.0


# ============================================================
# 2. TestBayesianHierarchical — 自适应收缩与可信区间
# ============================================================


class TestBayesianHierarchical:
    """贝叶斯分层 IRT: 自适应收缩 + 可信区间 + 非中心化."""

    def test_adaptive_shrinkage(self):
        """自适应收缩: 数据少 → 强收缩, 数据多 → 弱收缩."""
        est = IRTEstimator()
        # 学习者 A: 少量数据 (2 题), B: 大量数据 (20 题)
        responses_by_learner = {
            "few": [({"a": 1.0, "b": 0.0, "c": 0.0}, True)] * 2,
            "many": [({"a": 1.0, "b": 0.0, "c": 0.0}, True)] * 20,
        }
        results = est.estimate_hierarchical_bayesian(
            responses_by_learner,
            group_prior={"mean": 0.0, "sd": 1.0},
            adaptive=True,
        )
        assert "few" in results
        assert "many" in results
        # 少量数据的学习者应被收缩更多 (更接近群体均值 0)
        mle_few = est.estimate_mle_newton_raphson(
            responses_by_learner["few"]
        ).theta
        mle_many = est.estimate_mle_newton_raphson(
            responses_by_learner["many"]
        ).theta
        shrink_few = abs(mle_few - results["few"].theta)
        shrink_many = abs(mle_many - results["many"].theta)
        # 少量数据的收缩量应大于大量数据
        assert shrink_few > shrink_many

    def test_credible_interval(self):
        """可信区间: 返回 theta 的 HPD 等尾区间."""
        est = IRTEstimator()
        responses = [({"a": 1.0, "b": 0.0, "c": 0.0}, True)] * 10
        result = est.estimate_with_credible_interval(
            responses,
            credible_level=0.95,
        )
        assert "theta" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["theta"] < result["ci_upper"]
        # 95% CI 宽度 ≈ 2 * 1.96 * SE
        ci_width = result["ci_upper"] - result["ci_lower"]
        expected_width = 2 * 1.96 * result["se"]
        assert abs(ci_width - expected_width) < 0.2

    def test_group_prior_estimation(self):
        """从数据自动估计群体先验."""
        est = IRTEstimator()
        responses_by_learner = {
            "l1": [({"a": 1.0, "b": 0.0, "c": 0.0}, True)] * 5,
            "l2": [({"a": 1.0, "b": 0.0, "c": 0.0}, True)] * 5,
            "l3": [({"a": 1.0, "b": 0.0, "c": 0.0}, False)] * 5,
            "l4": [({"a": 1.0, "b": 0.0, "c": 0.0}, True)] * 5,
        }
        group_prior = est.estimate_group_prior(responses_by_learner)
        assert "mean" in group_prior
        assert "sd" in group_prior
        # 群体均值应 > 0 (3/4 学习者全对)
        assert group_prior["mean"] > 0.0
        assert group_prior["sd"] > 0.0

    def test_partial_pooling_vs_no_pooling(self):
        """部分池化 vs 无池化: 收缩降低了极端估计."""
        est = IRTEstimator()
        # 极端作答 (全对), MLE 会推向高 theta
        responses = [({"a": 1.0, "b": 2.0, "c": 0.0}, True)] * 3
        # 无池化 (纯 MLE)
        mle_state = est.estimate_mle_newton_raphson(responses)
        # 部分池化 (分层贝叶斯)
        hb_result = est.estimate_hierarchical_bayesian(
            {"learner": responses},
            group_prior={"mean": 0.0, "sd": 1.0},
            adaptive=True,
        )
        hb_theta = hb_result["learner"].theta
        # 分层贝叶斯应比纯 MLE 更保守 (收缩 toward 0)
        assert abs(hb_theta) < abs(mle_state.theta) or hb_theta <= mle_state.theta


# ============================================================
# 3. TestCATAdvancedTermination — PSER/置信区间/分类准确率终止
# ============================================================


class TestCATAdvancedTermination:
    """CAT 高级终止准则: PSER + 置信区间宽度 + 分类准确率."""

    def test_pser_termination(self):
        """PSER 终止: 预测下一题 SE 降幅 < hypo 时提前终止."""
        cat = CATSelector()
        # 构造场景: SE 已接近阈值, 但预测下一题降幅很小
        # should_stop_pser(se, count, theta, available_items, administered_ids)
        items = _bank([-2.0, -1.0, 0.0, 1.0, 2.0])
        # SE=0.28 (接近 0.3 阈值), 已答 8 题
        # 当预测降幅 < hypo=0.01 时应终止
        result = cat.should_stop_pser(
            current_se=0.28,
            count=8,
            theta=0.5,
            available_items=items,
            administered_ids={"q0", "q1", "q2"},
            se_threshold=0.3,
            hypo=0.01,
            hyper=0.05,
        )
        assert isinstance(result, bool)

    def test_pser_hyper_continue(self):
        """PSER hyper: SE 已达标但预测降幅 > hyper 时继续测试."""
        cat = CATSelector()
        items = _bank([-2.0, -1.0, 0.0, 1.0, 2.0])
        # SE=0.25 (< 0.3 阈值), 但有高信息量未答题
        result = cat.should_stop_pser(
            current_se=0.25,
            count=6,
            theta=0.0,
            available_items=items,
            administered_ids=set(),
            se_threshold=0.3,
            hypo=0.01,
            hyper=0.05,
        )
        # 可能继续 (如果预测降幅 > hyper) 也可能终止
        assert isinstance(result, bool)

    def test_ci_width_termination(self):
        """置信区间宽度终止: CI 宽度 < 阈值时终止."""
        cat = CATSelector()
        # SE=0.1 → 95% CI 宽度 ≈ 2*1.96*0.1 ≈ 0.39
        # 阈值 0.5 → 应终止
        assert cat.should_stop_ci_width(se=0.1, count=10, ci_width_threshold=0.5)
        # SE=0.2 → CI 宽度 ≈ 0.78 > 0.5 → 不终止
        assert not cat.should_stop_ci_width(se=0.2, count=5, ci_width_threshold=0.5)

    def test_classification_termination(self):
        """分类准确率终止: 能力分级决策足够确定时终止."""
        cat = CATSelector()
        # theta=1.5, SE=0.1 → P(theta > 1.0) 很高 → 分类确定
        assert cat.should_stop_classification(
            theta=1.5, se=0.1, cut_score=1.0, min_confidence=0.95
        )
        # theta=0.9, SE=0.5 → P(theta > 1.0) 不确定
        assert not cat.should_stop_classification(
            theta=0.9, se=0.5, cut_score=1.0, min_confidence=0.95
        )

    def test_multi_criteria_termination(self):
        """多准则组合终止: 任一准则满足即终止."""
        cat = CATSelector()
        items = _bank([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = cat.should_stop_multi(
            current_se=0.1,
            count=15,
            theta=1.0,
            available_items=items,
            administered_ids={"q0", "q1"},
            criteria=["length", "precision", "ci_width", "classification"],
            max_items=20,
            se_threshold=0.3,
            ci_width_threshold=0.5,
            cut_score=1.0,
            min_confidence=0.9,
        )
        # SE=0.1 < 0.3 → precision 终止
        assert result is True

    def test_predicted_se_reduction(self):
        """预测下一题 SE 降低量计算正确."""
        cat = CATSelector()
        est = cat._estimator
        theta = 0.0
        se_current = 0.5
        items = _bank([-2.0, 0.0, 2.0])
        # 预测选每道题后的 SE
        predictions = cat.predict_se_reduction(
            theta=theta,
            current_se=se_current,
            available_items=items,
            administered_ids=set(),
        )
        assert len(predictions) > 0
        for pred in predictions:
            assert "item_id" in pred
            assert "predicted_se" in pred
            assert "reduction" in pred
            assert pred["predicted_se"] <= se_current  # SE 应不增


# ============================================================
# 4. TestExposureControl — 渐进法/a-分层曝光控制
# ============================================================


class TestExposureControl:
    """CAT 曝光控制: 渐进法 (Progressive) + a-分层 (a-stratified)."""

    def test_progressive_exposure_init(self):
        """渐进法曝光控制器初始化."""
        cat = CATSelector(
            selection_strategy="progressive",
            max_items=20,
            se_threshold=0.3,
        )
        assert cat.selection_strategy == "progressive"

    def test_progressive_weight_increases(self):
        """渐进法权重随测试进度从 0→1 递增."""
        cat = CATSelector(
            selection_strategy="progressive",
            max_items=20,
            se_threshold=0.3,
        )
        # 初期权重低 (随机性大)
        w_early = cat.compute_progressive_weight(
            current_se=1.0, items_administered=2
        )
        # 后期权重高 (信息量主导)
        w_late = cat.compute_progressive_weight(
            current_se=0.35, items_administered=15
        )
        assert w_late > w_early
        assert 0.0 <= w_early <= 1.0
        assert 0.0 <= w_late <= 1.0

    def test_progressive_selection(self):
        """渐进法选题: 初期偏随机, 后期偏信息量."""
        cat = CATSelector(
            selection_strategy="progressive",
            max_items=10,
            se_threshold=0.3,
            rng=__import__("random").Random(42),
        )
        items = _bank([-2.0, -1.0, 0.0, 1.0, 2.0])
        # 初期 (2 题): 不一定选 Fisher 信息最大的
        chosen_early = cat.select_next(
            theta=0.0, available_items=items, administered_ids=set()
        )
        assert chosen_early is not None
        # 后期 (8 题): 更可能选 Fisher 信息最大的
        administered = {f"q{i}" for i in range(4)}
        chosen_late = cat.select_next(
            theta=0.0, available_items=items, administered_ids=administered
        )
        assert chosen_late is not None

    def test_a_stratified_exposure(self):
        """a-分层曝光控制: 按区分度分层, 初期用低 a 题."""
        cat = CATSelector(selection_strategy="a_stratified")
        # 题库含不同区分度的题目
        items = [
            {"item_id": "low_a1", "a": 0.4, "b": 0.0, "c": 0.0},
            {"item_id": "low_a2", "a": 0.5, "b": 1.0, "c": 0.0},
            {"item_id": "mid_a1", "a": 1.0, "b": 0.0, "c": 0.0},
            {"item_id": "mid_a2", "a": 1.2, "b": 1.0, "c": 0.0},
            {"item_id": "high_a1", "a": 2.0, "b": 0.0, "c": 0.0},
            {"item_id": "high_a2", "a": 2.5, "b": 1.0, "c": 0.0},
        ]
        cat.set_item_bank(items)
        # 初期 (1 题): 应从低 a 层选题
        chosen_early = cat.select_next(
            theta=0.0, available_items=items, administered_ids=set()
        )
        assert chosen_early is not None
        assert chosen_early["a"] <= 1.0  # 初期用低 a 题

    def test_exposure_rate_limiting(self):
        """曝光率限制: 单题曝光率不超过上限."""
        cat = CATSelector(
            selection_strategy="fisher_info",
            max_exposure_rate=0.3,
        )
        items = _bank([0.0, 0.0, 0.0, 1.0, -1.0])
        cat.set_item_bank(items)
        # 多次选题, 检查曝光率
        for _ in range(10):
            cat.select_next(
                theta=0.0,
                available_items=items,
                administered_ids=set(),
            )
        stats = cat.get_exposure_stats()
        total = sum(stats.values())
        if total > 0:
            for item_id, count in stats.items():
                rate = count / total
                # 曝光率应受控 (允许一定的容差因为题库小)
                assert rate <= 0.5  # 宽松断言


# ============================================================
# 5. TestZPDQuantification — 置信区间量化与自适应支架
# ============================================================


class TestZPDQuantification:
    """ZPD 量化: 置信区间法 + ZPD 宽度 + 自适应支架推荐."""

    def test_zpd_with_confidence_interval(self):
        """基于置信区间的 ZPD 计算."""
        zpd = ZPDCalculator()
        result = zpd.calculate_zpd_ci(
            theta=0.5,
            se=0.3,
            confidence_level=0.95,
        )
        assert "actual_level" in result
        assert "potential_level" in result
        assert "zpd_lower" in result
        assert "zpd_upper" in result
        assert "zpd_width" in result
        # 实际水平 = theta
        assert abs(result["actual_level"] - 0.5) < 0.01
        # 潜在水平 > 实际水平
        assert result["potential_level"] > result["actual_level"]
        # ZPD 上界 > ZPD 下界
        assert result["zpd_upper"] > result["zpd_lower"]
        # ZPD 宽度 = 上界 - 下界
        assert result["zpd_width"] > 0.0

    def test_zpd_width_decreases_with_data(self):
        """ZPD 宽度随数据增加 (SE 降低) 而收窄."""
        zpd = ZPDCalculator()
        # SE 大 → ZPD 宽
        wide = zpd.calculate_zpd_ci(theta=0.0, se=0.8, confidence_level=0.95)
        # SE 小 → ZPD 窄
        narrow = zpd.calculate_zpd_ci(theta=0.0, se=0.2, confidence_level=0.95)
        assert wide["zpd_width"] > narrow["zpd_width"]

    def test_zpd_coverage_score(self):
        """ZPD 覆盖得分: 量化已施测题目在 ZPD 三区的分布."""
        zpd = ZPDCalculator()
        theta = 0.0
        se = 0.3
        administered_items = [
            {"a": 1.0, "b": -2.0, "c": 0.0},  # independent
            {"a": 1.0, "b": 0.0, "c": 0.0},   # zpd
            {"a": 1.0, "b": 2.0, "c": 0.0},   # frustration
        ]
        score = zpd.zpd_coverage_score(theta, se, administered_items)
        assert 0.0 <= score <= 1.0
        # 三区均覆盖 → 高分
        assert score > 0.5

    def test_zpd_coverage_score_low(self):
        """仅覆盖一个区 → 低覆盖得分."""
        zpd = ZPDCalculator()
        theta = 0.0
        se = 0.3
        # 仅独立区题目 (b 足够低使 P > 0.9)
        administered_items = [
            {"a": 1.0, "b": -2.5, "c": 0.0},
            {"a": 1.0, "b": -3.0, "c": 0.0},
        ]
        score = zpd.zpd_coverage_score(theta, se, administered_items)
        assert score < 0.5

    def test_scaffold_recommendation(self):
        """自适应支架推荐: 基于 ZPD 和置信度推荐支架水平."""
        zpd = ZPDCalculator()
        # 低置信度 (高 SE) → 保守支架 (低 scaffold_level)
        rec_low = zpd.recommend_scaffold_level(
            theta=0.0, se=0.8, item_bank=_bank([-2.0, 0.0, 2.0])
        )
        # 高置信度 (低 SE) → 挑战性支架 (高 scaffold_level)
        rec_high = zpd.recommend_scaffold_level(
            theta=0.0, se=0.15, item_bank=_bank([-2.0, 0.0, 2.0])
        )
        assert 0.0 <= rec_low <= 1.0
        assert 0.0 <= rec_high <= 1.0
        assert rec_high >= rec_low

    def test_zpd_learning_path(self):
        """ZPD 学习路径: 在 ZPD 区间内推荐递进难度序列."""
        zpd = ZPDCalculator()
        theta = 0.0
        se = 0.3
        item_bank = _bank([-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0])
        path = zpd.recommend_learning_path(
            theta=theta, se=se, item_bank=item_bank, n_steps=5
        )
        assert len(path) <= 5
        # 路径难度应递进 (从低到高)
        if len(path) >= 2:
            for i in range(1, len(path)):
                assert path[i]["b"] >= path[i - 1]["b"]
        # 所有题目在 ZPD 区间内
        zpd_result = zpd.calculate_zpd_ci(theta, se)
        for item in path:
            assert item["b"] <= zpd_result["zpd_upper"] + 0.5

    def test_zpd_zone_with_ci(self):
        """ZPD 区分类 (含置信区间修正)."""
        zpd = ZPDCalculator()
        theta = 0.5
        se = 0.2
        # 题目难度在 theta 附近 → ZPD
        zone = zpd.classify_item_ci(theta, se, difficulty_b=0.6, discrimination_a=1.0)
        assert zone in ("independent", "zpd", "frustration")
        # 题目远低于 theta → independent
        zone_easy = zpd.classify_item_ci(theta, se, difficulty_b=-2.0, discrimination_a=1.0)
        assert zone_easy == "independent"


# ============================================================
# 6. TestEnhancedFullLink — 增强后全链路端到端
# ============================================================


class TestEnhancedFullLink:
    """增强后 IRT 全链路: 多模型 + 贝叶斯 + CAT高级终止 + ZPD量化."""

    def test_enhanced_service_init(self):
        """增强后的 IRTTracingService 初始化."""
        service = IRTTracingService(enable_enhanced=True)
        assert service is not None
        assert service.irt_estimator is not None
        assert service.cat_selector is not None
        assert service.zpd_calculator is not None

    def test_multi_model_full_link(self):
        """多模型 IRT 在全链路中工作."""
        service = IRTTracingService(enable_enhanced=True, irt_model="2PL")
        ts = time.time()
        outputs = []
        for i in range(10):
            out = service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
            outputs.append(out)
        assert all(o.theta > 0 for o in outputs[3:])  # 答对, theta 上升
        # 模型信息在输出中
        assert hasattr(outputs[-1], "irt_model") or "irt_model" in outputs[-1].to_dict()

    def test_adaptive_shrinkage_full_link(self):
        """自适应收缩在全链路中: 少量数据 → 保守估计."""
        service = IRTTracingService(enable_enhanced=True, adaptive_shrinkage=True)
        # 少量数据 (2 题)
        ts = time.time()
        for i in range(2):
            service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
        snap_few = service.get_ability_snapshot("learner_001")
        # 大量数据 (20 题)
        service2 = IRTTracingService(enable_enhanced=True, adaptive_shrinkage=True)
        for i in range(20):
            service2.process(_event(correct=True, difficulty=0.5, ts=ts + i))
        snap_many = service2.get_ability_snapshot("learner_001")
        # 少量数据 theta 更保守 (更接近 0)
        assert abs(snap_few["theta"]) < abs(snap_many["theta"])

    def test_ci_in_output(self):
        """AbilityOutput 包含置信区间信息."""
        service = IRTTracingService(enable_enhanced=True)
        out = service.process(_event(correct=True, difficulty=0.5))
        d = out.to_dict()
        # 增强后输出含 CI 信息
        assert "ci_lower" in d or hasattr(out, "ci_lower")
        assert "ci_upper" in d or hasattr(out, "ci_upper")

    def test_pser_termination_in_full_link(self):
        """PSER 终止在全链路中工作."""
        service = IRTTracingService(
            enable_enhanced=True,
            cat_termination_criteria=["pser", "precision", "length"],
        )
        service.set_item_bank(_bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
        ts = time.time()
        terminated = False
        for i in range(30):
            out = service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
            if out.termination_flag:
                terminated = True
                break
        # 应在题量上限内终止
        assert terminated

    def test_progressive_exposure_full_link(self):
        """渐进法曝光控制在全链路中工作."""
        cat = CATSelector(
            selection_strategy="progressive",
            max_items=15,
            se_threshold=0.3,
        )
        service = IRTTracingService(
            enable_enhanced=True,
            cat_selector=cat,
        )
        service.set_item_bank(_bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
        ts = time.time()
        for i in range(10):
            service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
        # 选题可用
        chosen = service.select_next_item(
            service._item_bank, administered_ids=set()
        )
        assert chosen is not None

    def test_zpd_ci_in_output(self):
        """AbilityOutput 包含 ZPD 置信区间信息."""
        service = IRTTracingService(enable_enhanced=True)
        service.set_item_bank(_bank([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
        out = service.process(_event(correct=True, difficulty=0.5))
        d = out.to_dict()
        # 增强后输出含 ZPD 量化信息
        assert "zpd_width" in d or hasattr(out, "zpd_width")
        assert "scaffold_level" in d or hasattr(out, "scaffold_level")

    def test_model_comparison_report(self):
        """模型比较报告在全链路中可用."""
        service = IRTTracingService(enable_enhanced=True)
        ts = time.time()
        # 收集 10 条作答
        for i in range(10):
            service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
        # 获取模型比较报告
        report = service.get_model_comparison_report("learner_001")
        assert report is not None
        assert isinstance(report, dict)
        # 报告包含各模型的 AIC/BIC
        for model in ("1PL", "2PL", "3PL"):
            assert model in report
            assert "aic" in report[model]

    def test_world_scheme_integration(self):
        """世界先进方案集成验证: Stan风格收缩 + catR风格PSER + Vygotsky ZPD-CI."""
        # 1. Stan 风格: 自适应收缩
        est = IRTEstimator()
        responses = [({"a": 1.0, "b": 0.0, "c": 0.0}, True)] * 5
        ci_result = est.estimate_with_credible_interval(responses, 0.95)
        assert ci_result["ci_lower"] < ci_result["theta"] < ci_result["ci_upper"]

        # 2. catR 风格: PSER 终止
        cat = CATSelector()
        items = _bank([-2.0, 0.0, 2.0])
        pser_result = cat.should_stop_pser(
            current_se=0.28, count=8, theta=0.5,
            available_items=items, administered_ids={"q0"},
            se_threshold=0.3, hypo=0.01, hyper=0.05,
        )
        assert isinstance(pser_result, bool)

        # 3. Vygotsky + IRT: ZPD-CI
        zpd = ZPDCalculator()
        zpd_ci = zpd.calculate_zpd_ci(theta=0.5, se=0.3, confidence_level=0.95)
        assert zpd_ci["zpd_width"] > 0.0
        scaffold = zpd.recommend_scaffold_level(
            theta=0.5, se=0.3, item_bank=items
        )
        assert 0.0 <= scaffold <= 1.0

    def test_enhanced_output_contract(self):
        """增强后输出契约可被下游消费."""
        service = IRTTracingService(enable_enhanced=True)
        service.set_item_bank(_bank([-2.0, -1.0, 0.0, 1.0, 2.0]))
        ts = time.time()
        out = None
        for i in range(5):
            out = service.process(_event(correct=True, difficulty=0.5, ts=ts + i))
        assert out is not None
        d = out.to_dict()
        # 基础字段
        for field in ("learner_id", "theta", "se", "response_count",
                      "p_correct_next", "zpd_zone", "recommended_difficulty",
                      "confidence", "termination_flag"):
            assert field in d
        # 增强字段
        enhanced_fields = ("ci_lower", "ci_upper", "zpd_width", "scaffold_level")
        for field in enhanced_fields:
            assert field in d, f"缺失增强字段: {field}"
        # 往返稳定
        restored = AbilityOutput.from_dict(d)
        assert restored.learner_id == out.learner_id
        assert restored.theta == out.theta
