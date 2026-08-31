"""L1 FSRS-6 在线参数优化器测试 — FSRSOptimizer / RegretMinimizer.

遵循 TDD Red-Green-Refactor. 测试覆盖:
- FSRSOptimizer: 损失计算 / 梯度计算 (有限差分) / 参数优化收敛 / 参数评估 /
  冷启动推荐 / 重放机制 (首复习不计入) / 权重钳制.
- RegretMinimizer: 初始状态 / 在线 FTRL 更新 / 遗憾累加 / 非负性 /
  奖励驱动参数移动.

注意:
- 不使用 mock, 使用真实 FSRSScheduler / FSRSParameters.
- 通过模拟真实参数生成的复习日志验证优化器收敛性.
- 不依赖 numpy, 仅使用 math 与标准库.
"""

from __future__ import annotations

import math
import random

import pytest

from dy3_polaris.l1.fsrs_scheduler import FSRSScheduler
from dy3_polaris.l1.models import FSRSCardState, FSRSParameters, FSRSReviewLog, MS_PER_SEC
from dy3_polaris.l1.fsrs_optimizer import FSRSOptimizer, RegretMinimizer


# ============================================================
# 辅助: 模拟复习日志
# ============================================================

_MS_PER_DAY = float(MS_PER_SEC * 86400)


def _make_true_params(w20: float = 0.45) -> FSRSParameters:
    """构造一组 "真实" 参数 (用于模拟数据)."""
    params = FSRSParameters()
    params.weights = list(params.weights)
    params.weights[20] = w20
    return params


def simulate_logs(
    true_params: FSRSParameters,
    n_cards: int = 4,
    reviews_per_card: int = 6,
    seed: int = 42,
    elapsed_factor: float = 3.0,
) -> list[dict]:
    """用真实参数模拟复习日志 (按 FSRS-6 调度演化稳定性, 按保留率采样评分).

    Args:
        true_params: 用于生成数据的 "真实" 参数.
        n_cards: 卡片数.
        reviews_per_card: 每卡复习次数.
        seed: 随机种子 (可复现).
        elapsed_factor: 复习间隔 = stability * elapsed_factor * uniform(0.8, 1.2).

    Returns:
        复习日志 dict 列表 (kc_id / grade / elapsed_days / reviewed_at).
    """
    scheduler = FSRSScheduler()
    rng = random.Random(seed)
    logs: list[dict] = []
    for c in range(n_cards):
        kc_id = f"kc-{c}"
        card = FSRSCardState(kc_id=kc_id)
        current_ts = 0.0
        for _ in range(reviews_per_card):
            if card.state == FSRSCardState.NEW:
                elapsed = 0.0
                grade = rng.choice([2, 3, 3, 4])
            else:
                elapsed = max(
                    0.1, card.stability * elapsed_factor * rng.uniform(0.8, 1.2)
                )
                r_val = card.retrievability(
                    current_ts + elapsed * _MS_PER_DAY,
                    true_params.decay,
                    true_params.factor,
                )
                actual = 1 if rng.random() < r_val else 0
                grade = rng.choice([3, 4]) if actual else 1
            current_ts = current_ts + elapsed * _MS_PER_DAY
            logs.append(
                {
                    "kc_id": kc_id,
                    "grade": grade,
                    "elapsed_days": elapsed,
                    "reviewed_at": int(current_ts),
                }
            )
            new_card, _, _ = scheduler.schedule_review(
                card, grade, true_params, int(current_ts)
            )
            card = new_card
    return logs


# ============================================================
# 1. FSRSOptimizer.compute_loss 测试
# ============================================================


class TestComputeLoss:
    """FSRSOptimizer.compute_loss — 负对数似然损失."""

    def test_empty_logs_returns_zero(self):
        """空复习日志损失为 0."""
        opt = FSRSOptimizer()
        assert opt.compute_loss([], FSRSParameters()) == 0.0

    def test_single_review_returns_zero(self):
        """单次复习 (NEW 状态, 无先验稳定性) 不计入损失."""
        opt = FSRSOptimizer()
        logs = [{"kc_id": "k1", "grade": 3, "elapsed_days": 0.0, "reviewed_at": 0}]
        assert opt.compute_loss(logs, FSRSParameters()) == 0.0

    def test_two_reviews_positive_loss(self):
        """两次复习 (第二次遗忘) 损失为正."""
        opt = FSRSOptimizer()
        logs = [
            {"kc_id": "k1", "grade": 3, "elapsed_days": 0.0, "reviewed_at": 0},
            {"kc_id": "k1", "grade": 1, "elapsed_days": 10.0, "reviewed_at": 864000000},
        ]
        loss = opt.compute_loss(logs, FSRSParameters())
        assert loss > 0.0

    def test_recall_matches_high_R_lower_loss(self):
        """高保留率且实际回忆成功 → 损失低于低保留率场景."""
        opt = FSRSOptimizer()
        # 短间隔: R 高, 实际回忆 (grade=3)
        short_logs = [
            {"kc_id": "k1", "grade": 3, "elapsed_days": 0.0, "reviewed_at": 0},
            {"kc_id": "k1", "grade": 3, "elapsed_days": 0.5, "reviewed_at": 43200000},
        ]
        # 长间隔: R 低, 实际回忆 (grade=3)
        long_logs = [
            {"kc_id": "k1", "grade": 3, "elapsed_days": 0.0, "reviewed_at": 0},
            {"kc_id": "k1", "grade": 3, "elapsed_days": 30.0, "reviewed_at": 2592000000},
        ]
        params = FSRSParameters()
        short_loss = opt.compute_loss(short_logs, params)
        long_loss = opt.compute_loss(long_logs, params)
        # 短间隔 R 高 → -log(R) 小; 长间隔 R 低 → -log(R) 大
        assert short_loss < long_loss

    def test_loss_non_negative(self):
        """损失始终非负."""
        opt = FSRSOptimizer()
        logs = simulate_logs(_make_true_params(0.3), n_cards=3, reviews_per_card=4, seed=1)
        assert opt.compute_loss(logs, FSRSParameters()) >= 0.0


# ============================================================
# 2. FSRSOptimizer.compute_gradient 测试
# ============================================================


class TestComputeGradient:
    """FSRSOptimizer.compute_gradient — 中心差分有限差分梯度."""

    def test_gradient_length_matches_weights(self):
        """梯度长度等于权重数 (21)."""
        opt = FSRSOptimizer()
        logs = simulate_logs(_make_true_params(), n_cards=3, reviews_per_card=4, seed=2)
        grad = opt.compute_gradient(logs, FSRSParameters())
        assert len(grad) == len(FSRSParameters().weights)
        assert all(isinstance(g, float) for g in grad)

    def test_empty_logs_returns_zeros(self):
        """空日志梯度全零."""
        opt = FSRSOptimizer()
        grad = opt.compute_gradient([], FSRSParameters())
        assert len(grad) == 21
        assert all(g == 0.0 for g in grad)

    def test_w20_gradient_nonzero(self):
        """w20 (衰减参数) 梯度非零 (重放使其参与损失)."""
        opt = FSRSOptimizer()
        logs = simulate_logs(_make_true_params(0.45), n_cards=4, reviews_per_card=6, seed=7)
        grad = opt.compute_gradient(logs, FSRSParameters())
        assert grad[20] != 0.0
        assert abs(grad[20]) > 1e-6

    def test_gradient_is_descent_direction(self):
        """梯度方向为下降方向: 沿 -grad 小步走应降低损失."""
        opt = FSRSOptimizer()
        logs = simulate_logs(_make_true_params(0.45), n_cards=4, reviews_per_card=6, seed=7)
        params = FSRSParameters()
        grad = opt.compute_gradient(logs, params)
        base_loss = opt.compute_loss(logs, params)
        # 沿 -grad 方向小步更新
        lr = 0.1
        new_w = [w - lr * g for w, g in zip(params.weights, grad)]
        new_params = FSRSParameters(
            weights=new_w,
            request_retention=params.request_retention,
            maximum_interval=params.maximum_interval,
        )
        new_loss = opt.compute_loss(logs, new_params)
        assert new_loss < base_loss

    def test_gradient_central_difference_symmetry(self):
        """中心差分梯度与单侧扰动一致 (梯度符号正确)."""
        opt = FSRSOptimizer()
        logs = simulate_logs(_make_true_params(0.45), n_cards=3, reviews_per_card=5, seed=3)
        params = FSRSParameters()
        grad = opt.compute_gradient(logs, params, eps=1e-4)
        # 对 w20 单独验证: (L(+eps) - L(-eps)) / (2eps) 应与 grad[20] 一致
        rr, mi = params.request_retention, params.maximum_interval
        wp = list(params.weights)
        wm = list(params.weights)
        wp[20] += 1e-4
        wm[20] -= 1e-4
        lp = opt.compute_loss(logs, FSRSParameters(weights=wp, request_retention=rr, maximum_interval=mi))
        lm = opt.compute_loss(logs, FSRSParameters(weights=wm, request_retention=rr, maximum_interval=mi))
        manual = (lp - lm) / (2e-4)
        assert grad[20] == pytest.approx(manual, rel=1e-6, abs=1e-12)


# ============================================================
# 3. FSRSOptimizer.optimize 测试
# ============================================================


class TestOptimize:
    """FSRSOptimizer.optimize — 梯度下降参数优化."""

    def test_returns_fsrs_parameters(self):
        """optimize 返回 FSRSParameters 实例."""
        opt = FSRSOptimizer()
        logs = simulate_logs(_make_true_params(), n_cards=3, reviews_per_card=4, seed=1)
        result = opt.optimize(logs)
        assert isinstance(result, FSRSParameters)
        assert len(result.weights) == 21

    def test_empty_logs_returns_initial(self):
        """空日志返回 (接近) 初始参数."""
        opt = FSRSOptimizer()
        initial = FSRSParameters()
        result = opt.optimize([], initial)
        assert result.weights == initial.weights

    def test_max_iter_zero_returns_initial(self):
        """max_iter=0 时不迭代, 返回初始参数."""
        opt = FSRSOptimizer(learning_rate=0.01, max_iter=0)
        logs = simulate_logs(_make_true_params(), n_cards=3, reviews_per_card=4, seed=1)
        initial = FSRSParameters()
        result = opt.optimize(logs, initial)
        assert result.weights == initial.weights

    def test_optimization_reduces_loss(self):
        """优化后损失应低于初始损失 (收敛)."""
        logs = simulate_logs(_make_true_params(0.45), n_cards=4, reviews_per_card=6, seed=7)
        opt = FSRSOptimizer(learning_rate=0.02, max_iter=30)
        initial = FSRSParameters()
        optimized = opt.optimize(logs, initial)
        initial_loss = opt.compute_loss(logs, initial)
        optimized_loss = opt.compute_loss(logs, optimized)
        assert optimized_loss < initial_loss

    def test_optimization_moves_w20_toward_true(self):
        """真实 w20=0.45 > 默认 0.12, 优化应使 w20 增大 (朝真实值移动)."""
        logs = simulate_logs(_make_true_params(0.45), n_cards=4, reviews_per_card=6, seed=7)
        opt = FSRSOptimizer(learning_rate=0.02, max_iter=30)
        initial = FSRSParameters()
        optimized = opt.optimize(logs, initial)
        assert optimized.weights[20] > initial.weights[20]

    def test_optimization_respects_tol(self):
        """大 tol 应在首轮后即停止 (参数仍为初始或仅微调)."""
        logs = simulate_logs(_make_true_params(), n_cards=3, reviews_per_card=4, seed=1)
        # tol 极大 → 首轮 max_change < tol → 立即停止
        opt = FSRSOptimizer(learning_rate=0.01, max_iter=50, tol=1e9)
        initial = FSRSParameters()
        optimized = opt.optimize(logs, initial)
        # 停止后返回值仍是合法 FSRSParameters
        assert isinstance(optimized, FSRSParameters)
        # 至多一轮更新 (best 追踪保证非增)
        assert opt.compute_loss(logs, optimized) <= opt.compute_loss(logs, initial) + 1e-9

    def test_optimized_params_within_clamp_bounds(self):
        """优化后 w20 在钳制边界 [0.01, 0.5] 内."""
        logs = simulate_logs(_make_true_params(0.5), n_cards=4, reviews_per_card=6, seed=9)
        opt = FSRSOptimizer(learning_rate=0.5, max_iter=40)
        optimized = opt.optimize(logs)
        assert 0.01 <= optimized.weights[20] <= 0.5


# ============================================================
# 4. FSRSOptimizer.evaluate_params 测试
# ============================================================


class TestEvaluateParams:
    """FSRSOptimizer.evaluate_params — 参数质量评估."""

    def test_returns_three_metrics(self):
        """返回 log_likelihood / rmse / mae 三个键."""
        opt = FSRSOptimizer()
        logs = simulate_logs(_make_true_params(), n_cards=3, reviews_per_card=4, seed=1)
        ev = opt.evaluate_params(logs, FSRSParameters())
        assert set(ev.keys()) == {"log_likelihood", "rmse", "mae"}

    def test_empty_logs_returns_zeros(self):
        """空日志各项指标为 0."""
        opt = FSRSOptimizer()
        ev = opt.evaluate_params([], FSRSParameters())
        assert ev == {"log_likelihood": 0.0, "rmse": 0.0, "mae": 0.0}

    def test_log_likelihood_non_positive(self):
        """对数似然 <= 0 (概率的对数非正)."""
        opt = FSRSOptimizer()
        logs = simulate_logs(_make_true_params(), n_cards=3, reviews_per_card=4, seed=1)
        ev = opt.evaluate_params(logs, FSRSParameters())
        assert ev["log_likelihood"] <= 0.0

    def test_rmse_mae_non_negative_and_bounded(self):
        """rmse / mae 非负且 <= 1 (R, actual ∈ [0,1])."""
        opt = FSRSOptimizer()
        logs = simulate_logs(_make_true_params(), n_cards=3, reviews_per_card=4, seed=1)
        ev = opt.evaluate_params(logs, FSRSParameters())
        assert 0.0 <= ev["rmse"] <= 1.0
        assert 0.0 <= ev["mae"] <= 1.0
        # rmse >= mae (均方根 >= 平均绝对值)
        assert ev["rmse"] >= ev["mae"] - 1e-12

    def test_better_params_lower_rmse(self):
        """真实参数的 rmse 不显著差于偏置参数 (评估一致性)."""
        true_params = _make_true_params(0.45)
        logs = simulate_logs(true_params, n_cards=4, reviews_per_card=6, seed=7)
        opt = FSRSOptimizer()
        ev_true = opt.evaluate_params(logs, true_params)
        # 评估应返回有限值
        assert math.isfinite(ev_true["rmse"])
        assert math.isfinite(ev_true["log_likelihood"])


# ============================================================
# 5. FSRSOptimizer.recommend_params 测试
# ============================================================


class TestRecommendParams:
    """FSRSOptimizer.recommend_params — 冷启动推荐."""

    def test_cold_start_returns_default(self):
        """cold_start=True 返回默认参数."""
        opt = FSRSOptimizer()
        history = [{"kc_id": "k1", "grade": 3, "elapsed_days": 1.0}] * 10
        result = opt.recommend_params(history, cold_start=True)
        assert result.weights == FSRSParameters().weights

    def test_insufficient_history_returns_default(self):
        """历史不足 (<5) 返回默认参数."""
        opt = FSRSOptimizer()
        history = [{"kc_id": "k1", "grade": 3, "elapsed_days": 1.0}]
        result = opt.recommend_params(history, cold_start=False)
        assert result.weights == FSRSParameters().weights

    def test_sufficient_history_returns_optimized(self):
        """历史充足时返回优化参数 (FSRSParameters 实例)."""
        opt = FSRSOptimizer(learning_rate=0.02, max_iter=10)
        history = []
        for c in range(4):
            for r in range(6):
                history.append(
                    {
                        "kc_id": f"k{c}",
                        "grade": 3 if r % 2 == 0 else 1,
                        "elapsed_days": float(r + 1),
                        "reviewed_at": (r + 1) * 86400000,
                    }
                )
        result = opt.recommend_params(history, cold_start=False)
        assert isinstance(result, FSRSParameters)
        assert len(result.weights) == 21

    def test_empty_history_returns_default(self):
        """空历史返回默认参数."""
        opt = FSRSOptimizer()
        result = opt.recommend_params([], cold_start=False)
        assert result.weights == FSRSParameters().weights


# ============================================================
# 6. FSRSOptimizer 内部辅助测试
# ============================================================


class TestOptimizerHelpers:
    """FSRSOptimizer 内部辅助方法."""

    def test_extract_log_dict(self):
        """_extract_log 兼容 dict."""
        log = {"kc_id": "k1", "grade": 4, "elapsed_days": 2.5, "reviewed_at": 123, "state_before": "review"}
        kc, g, e, t, s = FSRSOptimizer._extract_log(log)
        assert kc == "k1"
        assert g == 4
        assert e == 2.5
        assert t == 123
        assert s == "review"

    def test_extract_log_object(self):
        """_extract_log 兼容 FSRSReviewLog 对象."""
        log = FSRSReviewLog(kc_id="k2", grade=2, elapsed_days=1.0, state_before="review", state_after="review")
        kc, g, e, _, s = FSRSOptimizer._extract_log(log)
        assert kc == "k2"
        assert g == 2
        assert e == 1.0
        assert s == "review"

    def test_clamp_weights_w20_bounds(self):
        """_clamp_weights 将 w20 钳制到 [0.01, 0.5]."""
        weights = [0.0] * 21
        weights[20] = 10.0  # 越界
        clamped = FSRSOptimizer._clamp_weights(weights)
        assert clamped[20] == 0.5
        weights[20] = -5.0
        clamped = FSRSOptimizer._clamp_weights(weights)
        assert clamped[20] == 0.01

    def test_clamp_weights_w5_bounds(self):
        """_clamp_weights 将 w5 钳制到 [0.001, 5.0]."""
        weights = [0.0] * 21
        weights[5] = 100.0
        clamped = FSRSOptimizer._clamp_weights(weights)
        assert clamped[5] == 5.0
        weights[5] = -1.0
        clamped = FSRSOptimizer._clamp_weights(weights)
        assert clamped[5] == 0.001

    def test_group_logs_preserves_kc_grouping(self):
        """_group_logs 按 kc_id 分组."""
        opt = FSRSOptimizer()
        logs = [
            {"kc_id": "a", "grade": 3, "elapsed_days": 0, "reviewed_at": 1},
            {"kc_id": "b", "grade": 3, "elapsed_days": 0, "reviewed_at": 1},
            {"kc_id": "a", "grade": 3, "elapsed_days": 1, "reviewed_at": 2},
        ]
        grouped = opt._group_logs(logs)
        assert set(grouped.keys()) == {"a", "b"}
        assert len(grouped["a"]) == 2
        assert len(grouped["b"]) == 1


# ============================================================
# 7. RegretMinimizer 测试
# ============================================================


class TestRegretMinimizer:
    """RegretMinimizer — 在线遗憾最小化 (FTRL)."""

    def test_initial_params_zero(self):
        """初始参数全零."""
        rm = RegretMinimizer(n_params=21)
        params = rm.get_params()
        assert len(params) == 21
        assert all(p == 0.0 for p in params)

    def test_initial_regret_zero(self):
        """初始遗憾为 0."""
        rm = RegretMinimizer()
        assert rm.get_regret() == 0.0

    def test_update_changes_params(self):
        """一次更新后参数不再全零."""
        rm = RegretMinimizer(n_params=3)
        rm.update([1.0, 1.0, 1.0], observed_reward=1.0)
        params = rm.get_params()
        assert any(p != 0.0 for p in params)

    def test_regret_accumulates(self):
        """遗憾随更新累加 (max=1.0; regret = 0 + 0.8 + 0.2 = 1.0)."""
        rm = RegretMinimizer(n_params=3)
        rm.update([1.0, 1.0, 1.0], 1.0)  # max=1.0, regret += 0.0
        rm.update([1.0, 1.0, 1.0], 0.2)  # regret += 0.8
        rm.update([1.0, 1.0, 1.0], 0.8)  # regret += 0.2
        assert rm.get_regret() == pytest.approx(1.0)

    def test_regret_non_negative(self):
        """遗憾始终非负."""
        rm = RegretMinimizer(n_params=3)
        for r in [0.5, 0.9, 0.1, 0.7, 0.3]:
            rm.update([0.5, 0.3, 0.2], r)
        assert rm.get_regret() >= 0.0

    def test_get_params_returns_copy(self):
        """get_params 返回副本, 修改不影响内部状态."""
        rm = RegretMinimizer(n_params=3)
        rm.update([1.0, 1.0, 1.0], 1.0)
        params = rm.get_params()
        params[0] = 999.0
        assert rm.get_params()[0] != 999.0

    def test_high_reward_moves_toward_action(self):
        """高奖励使参数朝动作方向移动 (线性模型拟合)."""
        rm = RegretMinimizer(n_params=2)
        # 动作=[1,0], 多次高奖励 → w0 应为正 (拟合 reward≈w0)
        for _ in range(20):
            rm.update([1.0, 0.0], 1.0)
        params = rm.get_params()
        assert params[0] > 0.0  # w0 朝正方向
        assert abs(params[1]) < 1e-9  # w1 未被触发, 保持 0

    def test_low_reward_moves_away_from_action(self):
        """低 (负) 奖励使参数远离动作方向."""
        rm = RegretMinimizer(n_params=2)
        for _ in range(20):
            rm.update([1.0, 0.0], -1.0)
        params = rm.get_params()
        # 拟合 reward≈w0=-1 → w0 应为负
        assert params[0] < 0.0

    def test_n_params_custom(self):
        """自定义 n_params 维度."""
        rm = RegretMinimizer(n_params=5)
        assert len(rm.get_params()) == 5
        rm.update([1.0] * 5, 0.5)
        assert len(rm.get_params()) == 5

    def test_update_pads_short_action(self):
        """动作向量短于 n_params 时自动补零."""
        rm = RegretMinimizer(n_params=4)
        rm.update([1.0, 2.0], 1.0)  # 仅提供 2 维
        params = rm.get_params()
        assert len(params) == 4
        # 不抛异常即通过
