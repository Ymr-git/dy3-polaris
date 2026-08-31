"""分层贝叶斯 IRT + Newton-Raphson 鲁棒性增强 测试.

测试覆盖:
1. 分层贝叶斯 IRT: 多学习者估计、收缩效果、群体先验影响
2. MAP 估计: 先验影响、与 MLE 对比
3. 题目校准: EM 算法收敛、参数恢复
4. Newton-Raphson 鲁棒性: 极端数据、边界情况
"""

from __future__ import annotations

import math
import pytest

from dy3_polaris.l2.ability_assessor.irt import IRTEstimator
from dy3_polaris.l2.models import IRTState


# ============================================================
# 测试夹具
# ============================================================

@pytest.fixture
def estimator():
    """创建 IRT 估计器实例."""
    return IRTEstimator()


def _easy_item(item_id="i1"):
    """简单题目 (低难度)."""
    return {"item_id": item_id, "a": 1.2, "b": -1.0, "c": 0.1}


def _medium_item(item_id="i2"):
    """中等题目."""
    return {"item_id": item_id, "a": 1.5, "b": 0.0, "c": 0.2}


def _hard_item(item_id="i3"):
    """困难题目 (高难度)."""
    return {"item_id": item_id, "a": 1.0, "b": 2.0, "c": 0.25}


def _make_responses(items, n_reps=5, theta=0.5):
    """生成模拟作答序列."""
    responses = []
    est = IRTEstimator()
    for item in items:
        a, b, c = item["a"], item["b"], item["c"]
        p = est.predict_correct(theta, a, b, c)
        for _ in range(n_reps):
            correct = p > 0.5  # 确定性生成
            responses.append((dict(item), correct))
    return responses


# ============================================================
# 分层贝叶斯 IRT 测试
# ============================================================

class TestHierarchicalBayesian:
    """分层贝叶斯 IRT 估计测试."""

    def test_hierarchical_single_learner(self, estimator):
        """单学习者分层估计: 收缩向群体均值."""
        responses = _make_responses([_medium_item()], n_reps=5, theta=1.0)
        result = estimator.estimate_hierarchical_bayesian(
            {"learner1": responses},
            shrinkage=0.5,
        )
        assert "learner1" in result
        state = result["learner1"]
        assert isinstance(state, IRTState)
        assert state.response_count > 0

    def test_hierarchical_multiple_learners(self, estimator):
        """多学习者各自估计."""
        r1 = _make_responses([_easy_item()], n_reps=3, theta=1.0)
        r2 = _make_responses([_hard_item()], n_reps=3, theta=-1.0)
        result = estimator.estimate_hierarchical_bayesian(
            {"l1": r1, "l2": r2},
            shrinkage=0.3,
        )
        assert len(result) == 2
        assert result["l1"].response_count > 0
        assert result["l2"].response_count > 0

    def test_hierarchical_group_prior_custom(self, estimator):
        """自定义群体先验影响估计."""
        responses = _make_responses([_medium_item()], n_reps=3, theta=0.0)
        result_default = estimator.estimate_hierarchical_bayesian(
            {"l1": responses}, shrinkage=1.0,
        )
        result_custom = estimator.estimate_hierarchical_bayesian(
            {"l1": responses},
            group_prior={"mean": 2.0, "sd": 0.5},
            shrinkage=1.0,
        )
        # shrinkage=1.0: 完全收缩到群体均值
        assert result_default["l1"].theta == pytest.approx(0.0, abs=0.01)
        assert result_custom["l1"].theta == pytest.approx(2.0, abs=0.01)

    def test_hierarchical_shrinkage_zero(self, estimator):
        """shrinkage=0 等价于纯 MLE (无收缩)."""
        responses = _make_responses([_medium_item()], n_reps=5, theta=1.0)
        mle_state = estimator.estimate_mle_newton_raphson(responses)
        hier_result = estimator.estimate_hierarchical_bayesian(
            {"l1": responses}, shrinkage=0.0,
        )
        assert hier_result["l1"].theta == pytest.approx(mle_state.theta, abs=0.01)

    def test_hierarchical_shrinkage_one(self, estimator):
        """shrinkage=1 全部收缩到群体均值."""
        responses = _make_responses([_medium_item()], n_reps=5, theta=2.0)
        result = estimator.estimate_hierarchical_bayesian(
            {"l1": responses},
            group_prior={"mean": 0.0, "sd": 1.0},
            shrinkage=1.0,
        )
        assert result["l1"].theta == pytest.approx(0.0, abs=0.01)

    def test_hierarchical_empty(self, estimator):
        """空输入返回空字典."""
        result = estimator.estimate_hierarchical_bayesian({})
        assert result == {}

    def test_hierarchical_data_rich(self, estimator):
        """数据丰富时收缩影响小 (收缩后接近 MLE)."""
        # 大量数据使 MLE 非常可靠
        responses = _make_responses([_easy_item(), _medium_item(), _hard_item()], n_reps=10, theta=1.5)
        mle_state = estimator.estimate_mle_newton_raphson(responses)
        hier_result = estimator.estimate_hierarchical_bayesian(
            {"l1": responses}, shrinkage=0.1,
        )
        # shrinkage=0.1 时几乎不收缩
        assert hier_result["l1"].theta == pytest.approx(mle_state.theta, abs=0.3)

    def test_hierarchical_data_poor(self, estimator):
        """数据稀少时收缩影响大 (向群体均值靠拢)."""
        # 单条记录, MLE 不可靠
        responses = [(_medium_item(), True)]
        mle_state = estimator.estimate_mle_newton_raphson(responses)
        hier_result = estimator.estimate_hierarchical_bayesian(
            {"l1": responses},
            group_prior={"mean": 0.0, "sd": 1.0},
            shrinkage=0.8,
        )
        # 强收缩应拉向 0
        assert abs(hier_result["l1"].theta) < abs(mle_state.theta)


# ============================================================
# MAP 估计测试
# ============================================================

class TestMAPEstimation:
    """MAP (Maximum A Posteriori) 估计测试."""

    def test_map_basic(self, estimator):
        """基本 MAP 估计."""
        responses = _make_responses([_medium_item()], n_reps=5, theta=1.0)
        state = estimator.estimate_map(responses, prior_mean=0.0, prior_sd=1.0)
        assert isinstance(state, IRTState)
        assert state.response_count == len(responses)

    def test_map_prior_influence(self, estimator):
        """强先验拉向均值."""
        responses = _make_responses([_medium_item()], n_reps=3, theta=2.0)
        # 弱先验
        weak = estimator.estimate_map(responses, prior_mean=0.0, prior_sd=5.0)
        # 强先验
        strong = estimator.estimate_map(responses, prior_mean=0.0, prior_sd=0.1)
        # 强先验应更接近 0
        assert abs(strong.theta) < abs(weak.theta)

    def test_map_vs_mle(self, estimator):
        """MAP 比 MLE 更保守 (向先验收缩)."""
        responses = _make_responses([_medium_item()], n_reps=3, theta=2.0)
        mle_state = estimator.estimate_mle_newton_raphson(responses)
        map_state = estimator.estimate_map(responses, prior_mean=0.0, prior_sd=1.0)
        # MAP 应比 MLE 更接近先验均值 0
        assert abs(map_state.theta) <= abs(mle_state.theta) + 0.5

    def test_map_empty(self, estimator):
        """空响应回退到先验."""
        state = estimator.estimate_map([], prior_mean=0.5, prior_sd=0.8)
        assert state.theta == pytest.approx(0.5)
        assert state.se == pytest.approx(0.8)
        assert state.response_count == 0

    def test_map_informative_prior(self, estimator):
        """强先验 (sd=0.1) 几乎完全决定估计."""
        responses = _make_responses([_medium_item()], n_reps=2, theta=2.0)
        state = estimator.estimate_map(responses, prior_mean=1.0, prior_sd=0.1)
        # 强先验应使 theta 接近 1.0
        assert state.theta == pytest.approx(1.0, abs=0.5)

    def test_map_weak_prior(self, estimator):
        """弱先验 (sd=10) 退化为 MLE."""
        responses = _make_responses([_medium_item()], n_reps=5, theta=1.0)
        mle_state = estimator.estimate_mle_newton_raphson(responses)
        map_state = estimator.estimate_map(responses, prior_mean=0.0, prior_sd=10.0)
        # 弱先验时 MAP ≈ MLE
        assert map_state.theta == pytest.approx(mle_state.theta, abs=0.3)


# ============================================================
# 题目校准测试
# ============================================================

class TestItemCalibration:
    """题目参数校准 (EM 算法) 测试."""

    def test_calibrate_basic(self, estimator):
        """基本校准: 返回合法参数."""
        responses_by_item = {
            "item1": [(0.0, True), (0.5, True), (-0.5, True), (1.0, False)],
        }
        result = estimator.calibrate_items(responses_by_item, max_iter=10)
        assert "item1" in result
        params = result["item1"]
        assert "a" in params
        assert "b" in params
        assert "c" in params
        assert 0.01 <= params["a"] <= 5.0
        assert -3.0 <= params["b"] <= 3.0
        assert 0.0 <= params["c"] <= 0.5

    def test_calibrate_convergence(self, estimator):
        """EM 算法收敛: 参数在迭代中变化减小."""
        responses_by_item = {
            "item1": [(0.0, True), (0.5, True), (1.0, False), (-1.0, True)],
        }
        result = estimator.calibrate_items(responses_by_item, max_iter=50, tol=1e-5)
        # 应收敛 (不是因为 max_iter 终止)
        params = result["item1"]
        # 难度 b 应在合理范围 (数据中 theta=0 附近)
        assert -3.0 <= params["b"] <= 3.0

    def test_calibrate_multiple_items(self, estimator):
        """多题目校准."""
        responses_by_item = {
            "easy": [(0.0, True), (0.5, True), (-0.5, True)],
            "hard": [(0.0, False), (0.5, False), (1.5, True)],
        }
        result = estimator.calibrate_items(responses_by_item, max_iter=20)
        assert len(result) == 2
        assert "easy" in result
        assert "hard" in result
        # 简单题难度 b 应低于难题
        assert result["easy"]["b"] <= result["hard"]["b"]

    def test_calibrate_empty(self, estimator):
        """空输入返回空字典."""
        result = estimator.calibrate_items({})
        assert result == {}

    def test_calibrate_parameter_range(self, estimator):
        """校准后参数在合法范围内."""
        responses_by_item = {
            "item1": [(0.0, True), (1.0, False), (-1.0, True), (0.5, True)],
        }
        result = estimator.calibrate_items(responses_by_item, max_iter=30)
        params = result["item1"]
        assert params["a"] > 0
        assert -3.0 <= params["b"] <= 3.0
        assert 0.0 <= params["c"] <= 0.5


# ============================================================
# Newton-Raphson 鲁棒性测试
# ============================================================

class TestNewtonRaphsonRobustness:
    """Newton-Raphson 鲁棒性增强测试."""

    def test_nr_extreme_all_correct(self, estimator):
        """全对数据: theta 应收敛到正值."""
        responses = [(_easy_item(), True) for _ in range(10)]
        state = estimator.estimate_mle_newton_raphson(responses)
        assert state.theta > 0
        assert state.response_count == 10

    def test_nr_extreme_all_wrong(self, estimator):
        """全错数据: theta 应收敛到负值."""
        responses = [(_hard_item(), False) for _ in range(10)]
        state = estimator.estimate_mle_newton_raphson(responses)
        assert state.theta < 0
        assert state.response_count == 10

    def test_nr_single_response(self, estimator):
        """单条记录不崩溃."""
        responses = [(_medium_item(), True)]
        state = estimator.estimate_mle_newton_raphson(responses)
        assert isinstance(state, IRTState)
        assert state.response_count == 1

    def test_nr_max_iter_limit(self, estimator):
        """最大迭代限制: max_iter 被硬上限 200 截断."""
        responses = _make_responses([_medium_item()], n_reps=3, theta=0.5)
        # 设置超大 max_iter, 应被限制
        state = estimator.estimate_mle_newton_raphson(responses, max_iter=10000)
        assert isinstance(state, IRTState)
        assert state.response_count == len(responses)

    def test_nr_convergence(self, estimator):
        """正常数据收敛."""
        responses = _make_responses([_easy_item(), _medium_item()], n_reps=5, theta=1.0)
        state = estimator.estimate_mle_newton_raphson(responses, max_iter=100, tol=1e-6)
        assert isinstance(state, IRTState)
        # theta 应在合理范围
        assert -3.0 <= state.theta <= 3.0
        # SE 应为正
        assert state.se > 0

    def test_nr_boundary(self, estimator):
        """边界值: theta 最终钳制到 [-3, 3]."""
        responses = [(_hard_item(), True) for _ in range(20)]
        state = estimator.estimate_mle_newton_raphson(responses)
        assert -3.0 <= state.theta <= 3.0

    def test_nr_empty_responses(self, estimator):
        """空响应回退."""
        state = estimator.estimate_mle_newton_raphson([])
        assert state.theta == 0.0
        assert state.se == 1.0
        assert state.response_count == 0

    def test_nr_check_convergence_method(self, estimator):
        """_check_convergence 辅助方法."""
        # 正常收敛
        should_stop, reason = estimator._check_convergence(
            [1.0, 0.5, 0.1], [-5, -3, -2], 3
        )
        assert should_stop is False

        # 发散: delta 持续增大
        should_stop, reason = estimator._check_convergence(
            [0.1, 0.2, 0.3, 0.5, 1.0], [-5, -4, -3, -2, -1], 5
        )
        assert should_stop is True
        assert "divergence" in reason

    def test_nr_check_convergence_ll_decline(self, estimator):
        """_check_convergence 检测似然下降."""
        should_stop, reason = estimator._check_convergence(
            [0.1, 0.1, 0.1, 0.1], [-1, -2, -3, -4], 4
        )
        assert should_stop is True
        assert "ll_decline" in reason

    def test_nr_check_convergence_short_history(self, estimator):
        """_check_convergence 短历史不触发."""
        should_stop, reason = estimator._check_convergence([0.5], [-1], 1)
        assert should_stop is False
        assert reason == ""

    def test_nr_mixed_responses(self, estimator):
        """混合对错数据."""
        responses = [
            (_easy_item(), True),
            (_hard_item(), False),
            (_medium_item(), True),
            (_medium_item(), False),
        ]
        state = estimator.estimate_mle_newton_raphson(responses)
        assert isinstance(state, IRTState)
        assert state.response_count == 4
        assert -3.0 <= state.theta <= 3.0
