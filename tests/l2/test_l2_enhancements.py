"""L2 增强功能测试 (TDD: RED -> GREEN).

测试覆盖三大增强:
1. BKT fit_params:
   - 收敛检测 (tol 较大时提前停止)
   - p_l0 被学习 (偏离默认值)
   - 返回最佳参数 (最高对数似然, best-tracking)
2. SessionManager:
   - record_step 每 N 步自动创建检查点
   - cleanup_expired_sessions 关闭超时会话
   - get_session_stats 返回正确统计
3. ForgettingModel:
   - should_review 使用连续衰减 (168h 边界无跳变)
   - compute_stability 利用 correct_count (正确率越高稳定性越高)
   - compute_review_interval 返回合理复习间隔
"""

from __future__ import annotations

import math
import time

import pytest

from dy3_polaris.l2.knowledge_tracer import BKTTracer, ForgettingModel
from dy3_polaris.l2.models import DEFAULT_BKT_PARAMS, AnswerRecord, TracingState
from dy3_polaris.l2.session import SessionManager
from dy3_polaris.l2.session.session_manager import DEFAULT_CHECKPOINT_INTERVAL


# ============================================================
# 测试辅助
# ============================================================


def _records_from_corrects(corrects, kp="kp-1", difficulty=0.5):
    """由正确性序列构造 AnswerRecord 列表 (时间戳升序, 单位秒)."""
    return [
        AnswerRecord("l1", kp, bool(c), float(t) * 3600.0, difficulty)
        for t, c in enumerate(corrects)
    ]


# ============================================================
# 1. BKT fit_params — 收敛检测
# ============================================================


class TestBKTFitParamsConvergence:
    """BKTTracer.fit_params 收敛检测测试 — tol 较大时提前停止."""

    def test_fit_params_accepts_tol_parameter(self):
        """fit_params 接受 tol 关键字参数 (签名增强)."""
        tracer = BKTTracer()
        records = _records_from_corrects([True, False, True, True, False])
        # 不应抛 TypeError (旧实现无 tol 参数)
        fitted = tracer.fit_params(records, max_iter=10, tol=1e-6)
        assert set(fitted.keys()) >= {"p_l0", "p_t", "p_g", "p_s"}

    def test_fit_params_convergence_stops_early(self):
        """收敛检测: tol 较大时提前停止, log_likelihood 调用次数远少于满迭代."""
        tracer = BKTTracer()
        records = _records_from_corrects([True, False, True, True, False, True])
        original_ll = tracer.log_likelihood

        def make_counter():
            state = {"n": 0}

            def counting_ll(recs, params):
                state["n"] += 1
                return original_ll(recs, params)

            return state, counting_ll

        # 松弛 tol=1e9: 第 1 次迭代后 LL 变化必 < 1e9, 提前停止
        loose_state, loose_ll = make_counter()
        tracer.log_likelihood = loose_ll
        tracer.fit_params(records, max_iter=200, tol=1e9)
        loose_calls = loose_state["n"]

        # 严格 tol=0.0: abs(diff) < 0.0 永不成立, 跑满 200 次迭代
        strict_state, strict_ll = make_counter()
        tracer.log_likelihood = strict_ll
        tracer.fit_params(records, max_iter=200, tol=0.0)
        strict_calls = strict_state["n"]

        # 松弛应远少于严格 (提前停止)
        assert loose_calls < strict_calls
        # 松弛应在很少几次迭代内停止 (远小于满迭代 200 次 * 每轮 9 次调用)
        assert loose_calls < strict_calls * 0.5

    def test_fit_params_default_tol_is_small(self):
        """默认 tol=1e-6: 正常数据下仍能跑较多迭代 (不会立刻误收敛)."""
        tracer = BKTTracer()
        records = _records_from_corrects([True, False, True, True, False, True])
        original_ll = tracer.log_likelihood
        state = {"n": 0}

        def counting_ll(recs, params):
            state["n"] += 1
            return original_ll(recs, params)

        tracer.log_likelihood = counting_ll
        tracer.fit_params(records, max_iter=100)  # 使用默认 tol
        # 默认 tol 较小, 至少应执行若干次迭代 (而非第 1 轮即停)
        assert state["n"] >= 10


# ============================================================
# 2. BKT fit_params — p_l0 被学习
# ============================================================


class TestBKTFitParamsLearnsPL0:
    """BKTTracer.fit_params 学习 p_l0 测试 — p_l0 应偏离默认值."""

    def test_fit_params_learns_p_l0_all_correct(self):
        """全对序列 -> 先验掌握概率 p_l0 应上升 (高于默认 0.5)."""
        tracer = BKTTracer()
        records = _records_from_corrects([True] * 20)
        fitted = tracer.fit_params(records, max_iter=100)
        # 旧实现不学习 p_l0, 保持 0.5; 新实现应使其偏离 (> 0.5)
        assert fitted["p_l0"] > DEFAULT_BKT_PARAMS["p_l0"]

    def test_fit_params_learns_p_l0_all_wrong(self):
        """全错序列 -> 先验掌握概率 p_l0 应下降 (低于默认 0.5)."""
        tracer = BKTTracer()
        records = _records_from_corrects([False] * 20)
        fitted = tracer.fit_params(records, max_iter=100)
        assert fitted["p_l0"] < DEFAULT_BKT_PARAMS["p_l0"]

    def test_fit_params_p_l0_in_unit_interval(self):
        """学习后的 p_l0 落在 (0, 1) 内."""
        tracer = BKTTracer()
        records = _records_from_corrects([True, False, True, True, False, True])
        fitted = tracer.fit_params(records, max_iter=50)
        assert 0.0 < fitted["p_l0"] < 1.0


# ============================================================
# 3. BKT fit_params — 返回最佳参数 (best-tracking)
# ============================================================


class TestBKTFitParamsBestParams:
    """BKTTracer.fit_params best-tracking 测试 — 返回最高似然参数."""

    def test_fit_params_returns_best_params_highest_ll(self):
        """返回最佳参数: 即使梯度上升震荡 (最终似然低于初始), 仍返回 >= 初始似然的参数.

        使用 mostly_wrong30 数据集: 旧实现最终似然 (-163) 远低于初始 (-11.3);
        best-tracking 应返回峰值参数 (似然 >= 初始).
        """
        tracer = BKTTracer()
        # 25 错 + 5 对: 梯度上升会剧烈震荡, 最终似然远低于峰值
        records = _records_from_corrects([False] * 25 + [True] * 5)
        init_params = dict(DEFAULT_BKT_PARAMS)
        init_ll = tracer.log_likelihood(records, init_params)
        fitted = tracer.fit_params(records, max_iter=100)
        fitted_ll = tracer.log_likelihood(records, fitted)
        # best-tracking 保证返回参数的似然不低于初始似然
        assert fitted_ll >= init_ll - 1e-9

    def test_fit_params_best_ll_at_least_initial(self):
        """best-tracking: 任意数据下拟合后似然 >= 初始似然 (best 初始化为初始参数)."""
        tracer = BKTTracer()
        for corrects in (
            [True, False, True, True, False, True, True, False, True, True],
            [True] * 15,
            [False] * 15,
            [i % 2 == 0 for i in range(20)],
        ):
            records = _records_from_corrects(corrects)
            init_ll = tracer.log_likelihood(records, dict(DEFAULT_BKT_PARAMS))
            fitted = tracer.fit_params(records, max_iter=100)
            fitted_ll = tracer.log_likelihood(records, fitted)
            assert fitted_ll >= init_ll - 1e-9, (
                f"fitted_ll={fitted_ll} < init_ll={init_ll} for {corrects}"
            )


# ============================================================
# 4. SessionManager — record_step 自动检查点
# ============================================================


class TestSessionManagerRecordStep:
    """SessionManager.record_step 测试 — 每 N 步自动创建检查点."""

    def test_record_step_exists_and_returns_none_or_checkpoint(self):
        """record_step 方法存在, 返回 None 或检查点对象."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        result = mgr.record_step(sid, {"step": 1})
        assert result is None  # 第 1 步不创建检查点

    def test_record_step_creates_auto_checkpoint_every_5_steps(self):
        """每 DEFAULT_CHECKPOINT_INTERVAL (5) 步自动创建一个检查点."""
        assert DEFAULT_CHECKPOINT_INTERVAL == 5
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        returns = []
        for i in range(1, 13):  # 步骤 1..12
            returns.append(mgr.record_step(sid, {"step": i}))
        # 第 5、10 步创建检查点 (truthy), 其余 None
        assert returns[4] is not None  # step 5
        assert returns[9] is not None  # step 10
        for i in (0, 1, 2, 3, 5, 6, 7, 8, 10, 11):
            assert returns[i] is None
        sess = mgr.get_session(sid)
        assert len(sess.checkpoints) == 2
        # 检查点保留传入的 context
        assert sess.checkpoints[0]["step"] == 5
        assert sess.checkpoints[1]["step"] == 10

    def test_record_step_increments_step_counter(self):
        """record_step 递增步骤计数 (经 get_session_stats 可见)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        for i in range(7):
            mgr.record_step(sid, {"step": i})
        stats = mgr.get_session_stats(sid)
        assert stats["step_count"] == 7

    def test_record_step_nonexistent_raises(self):
        """record_step 不存在的会话抛出 StoreError."""
        from dy3_polaris.l2.exceptions import StoreError

        mgr = SessionManager()
        with pytest.raises(StoreError):
            mgr.record_step("sess-nonexistent", {"step": 1})

    def test_record_step_no_context_creates_checkpoint(self):
        """record_step 不传 context 时, 检查点仍能创建 (使用空字典)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        for _ in range(5):
            mgr.record_step(sid)  # 无 context
        sess = mgr.get_session(sid)
        assert len(sess.checkpoints) == 1


# ============================================================
# 5. SessionManager — cleanup_expired_sessions
# ============================================================


class TestSessionManagerCleanupExpired:
    """SessionManager.cleanup_expired_sessions 测试 — 关闭超时的 active 会话."""

    def test_cleanup_closes_timed_out_active_session(self):
        """超时的 active 会话被关闭, 并返回其 session_id."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        # 模拟 2 小时前最后活动
        mgr._last_activity[sid] = time.time() - 7200.0
        expired = mgr.cleanup_expired_sessions(timeout=1800.0)
        assert sid in expired
        assert mgr.get_session(sid).status == "closed"

    def test_cleanup_keeps_recent_active_session(self):
        """近期活动的 active 会话不被关闭."""
        mgr = SessionManager()
        sid_recent = mgr.start_session("learner-001")
        # 最近活动 (now), 不超时
        mgr._last_activity[sid_recent] = time.time()
        expired = mgr.cleanup_expired_sessions(timeout=1800.0)
        assert sid_recent not in expired
        assert mgr.get_session(sid_recent).status == "active"

    def test_cleanup_skips_already_closed_session(self):
        """已关闭的会话不被重复清理 (不在 expired 列表)."""
        mgr = SessionManager()
        sid_closed = mgr.start_session("learner-001")
        mgr.end_session(sid_closed)  # 已 closed
        mgr._last_activity[sid_closed] = time.time() - 7200.0  # 超时但已关闭
        expired = mgr.cleanup_expired_sessions(timeout=1800.0)
        assert sid_closed not in expired
        assert mgr.get_session(sid_closed).status == "closed"

    def test_cleanup_skips_paused_session(self):
        """paused 会话不被 cleanup 关闭 (仅清理 active)."""
        mgr = SessionManager()
        sid_paused = mgr.start_session("learner-001")
        mgr.pause_session(sid_paused)
        mgr._last_activity[sid_paused] = time.time() - 7200.0
        expired = mgr.cleanup_expired_sessions(timeout=1800.0)
        assert sid_paused not in expired
        assert mgr.get_session(sid_paused).status == "paused"

    def test_cleanup_returns_only_expired_list(self):
        """cleanup 返回值为被清理的 session_id 列表 (str 元素)."""
        mgr = SessionManager()
        sid_old = mgr.start_session("learner-001")
        sid_new = mgr.start_session("learner-001")
        mgr._last_activity[sid_old] = time.time() - 7200.0
        mgr._last_activity[sid_new] = time.time()
        expired = mgr.cleanup_expired_sessions(timeout=1800.0)
        assert isinstance(expired, list)
        assert all(isinstance(s, str) for s in expired)
        assert sid_old in expired
        assert sid_new not in expired

    def test_cleanup_default_timeout_uses_constant(self):
        """不传 timeout 时使用 DEFAULT_SESSION_TIMEOUT (1800s)."""
        from dy3_polaris.l2.session.session_manager import DEFAULT_SESSION_TIMEOUT

        assert DEFAULT_SESSION_TIMEOUT == 1800.0
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        # 设为 2000 秒前 (> 1800 默认超时)
        mgr._last_activity[sid] = time.time() - 2000.0
        expired = mgr.cleanup_expired_sessions()  # 使用默认 timeout
        assert sid in expired


# ============================================================
# 6. SessionManager — get_session_stats
# ============================================================


class TestSessionManagerGetStats:
    """SessionManager.get_session_stats 测试 — 返回步骤/检查点/时长等统计."""

    def test_get_session_stats_returns_dict(self):
        """get_session_stats 返回统计字典."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        stats = mgr.get_session_stats(sid)
        assert isinstance(stats, dict)

    def test_get_session_stats_contains_session_id(self):
        """统计字典包含 session_id."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        stats = mgr.get_session_stats(sid)
        assert stats["session_id"] == sid

    def test_get_session_stats_step_and_checkpoint_counts(self):
        """7 步 -> step_count=7, 每 5 步一检查点 -> checkpoint_count=1."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        for i in range(7):
            mgr.record_step(sid, {"step": i})
        stats = mgr.get_session_stats(sid)
        assert stats["step_count"] == 7
        assert stats["checkpoint_count"] == 1
        assert stats["status"] == "active"

    def test_get_session_stats_zero_for_new_session(self):
        """新建会话 (无步骤) -> step_count=0, checkpoint_count=0."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        stats = mgr.get_session_stats(sid)
        assert stats["step_count"] == 0
        assert stats["checkpoint_count"] == 0

    def test_get_session_stats_has_duration(self):
        """统计包含 duration (>= 0)."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-001")
        stats = mgr.get_session_stats(sid)
        assert "duration" in stats
        assert stats["duration"] >= 0.0

    def test_get_session_stats_has_learner_id(self):
        """统计包含 learner_id."""
        mgr = SessionManager()
        sid = mgr.start_session("learner-xyz")
        stats = mgr.get_session_stats(sid)
        assert stats["learner_id"] == "learner-xyz"

    def test_get_session_stats_nonexistent_raises(self):
        """get_session_stats 不存在的会话抛出 StoreError."""
        from dy3_polaris.l2.exceptions import StoreError

        mgr = SessionManager()
        with pytest.raises(StoreError):
            mgr.get_session_stats("sess-nonexistent")


# ============================================================
# 7. ForgettingModel — should_review 连续衰减
# ============================================================


class TestForgettingModelShouldReviewContinuous:
    """ForgettingModel.should_review 测试 — 使用 compute_retention 连续衰减."""

    def _make_state(self, mastery=0.8, attempts=0, correct=0, last=0.0):
        return TracingState(
            kp_id="kp-001",
            mastery_prob=mastery,
            attempts=attempts,
            correct_count=correct,
            last_attempt_time=last,
            bkt_params=dict(DEFAULT_BKT_PARAMS),
        )

    def test_should_review_decays_before_168h(self):
        """连续衰减: 168h 以内也衰减 — 100h 时 0.8 掌握度应已低于阈值需复习.

        旧实现用 decay (硬阈值): 100h < 168h 不衰减 -> 0.8 >= 0.5 -> 不复习 (False).
        新实现用 compute_retention: 0.8*exp(-0.007*100) ≈ 0.397 < 0.5 -> 复习 (True).
        """
        model = ForgettingModel()
        state = self._make_state(mastery=0.8, attempts=0, correct=0, last=0.0)
        current = 100.0 * 3600.0  # 100 小时
        assert model.should_review(state, current_time=current) is True

    def test_should_review_no_discontinuity_at_168h_boundary(self):
        """168h 边界两侧判定一致 (无硬阈值跳变).

        mastery == threshold == 0.5:
        - 旧 decay: 167.9h 不衰减 -> 0.5 < 0.5 为 False; 168.1h 衰减 -> True (跳变).
        - 新 compute_retention: 两侧均早已衰减到 0.5 以下 -> 均 True (平滑).
        """
        model = ForgettingModel()

        def review_at(hours):
            st = self._make_state(mastery=0.5, attempts=0, correct=0, last=0.0)
            return model.should_review(st, current_time=hours * 3600.0, threshold=0.5)

        just_below = review_at(167.9)
        just_above = review_at(168.1)
        assert just_below is True  # 旧实现此处为 False -> RED
        assert just_above is True

    def test_should_review_continuous_smooth_change(self):
        """连续衰减: 167.9h 与 168.1h 的底层保留率接近 (无跳变)."""
        model = ForgettingModel()
        stability = model.compute_stability(attempts=0, correct_count=0)
        v_below = model.compute_retention(0.5, 167.9, stability)
        v_above = model.compute_retention(0.5, 168.1, stability)
        # 平滑: 两侧差值极小 (无跳变)
        assert abs(v_below - v_above) < 0.01
        # 且均明显低于 0.5 (168h 之前就已衰减)
        assert v_below < 0.5
        assert v_above < 0.5


# ============================================================
# 8. ForgettingModel — compute_stability 利用 correct_count
# ============================================================


class TestForgettingModelStabilityCorrectCount:
    """ForgettingModel.compute_stability 测试 — 利用 correct_count (正确率)."""

    def test_higher_accuracy_higher_stability(self):
        """同 attempts 下, 正确率越高 stability 越高."""
        model = ForgettingModel()
        low_acc = model.compute_stability(attempts=10, correct_count=2)   # 正确率 0.2
        high_acc = model.compute_stability(attempts=10, correct_count=10)  # 正确率 1.0
        assert high_acc > low_acc

    def test_low_accuracy_no_bonus(self):
        """正确率 <= 0.5 时无 accuracy bonus (与仅 attempts 一致)."""
        model = ForgettingModel()
        base = model.compute_stability(attempts=10, correct_count=0)
        low_acc = model.compute_stability(attempts=10, correct_count=2)  # 正确率 0.2
        mid_acc = model.compute_stability(attempts=10, correct_count=5)  # 正确率 0.5
        # 正确率 <= 0.5 -> 无 bonus, 三者相等
        assert base == pytest.approx(low_acc)
        assert base == pytest.approx(mid_acc)

    def test_zero_attempts_returns_min_stability(self):
        """0 次作答 -> stability = MIN_STABILITY (1.0), 无 bonus."""
        model = ForgettingModel()
        assert model.compute_stability(attempts=0, correct_count=0) == pytest.approx(1.0)

    def test_stability_monotonic_in_correct_count(self):
        """固定 attempts, correct_count 越多 stability 越高 (正确率提升)."""
        model = ForgettingModel()
        prev = model.compute_stability(attempts=20, correct_count=10)  # 0.5
        for cc in (11, 13, 15, 18, 20):
            cur = model.compute_stability(attempts=20, correct_count=cc)
            assert cur >= prev
            prev = cur

    def test_stability_formula_high_accuracy(self):
        """高正确率公式: base + max(0,(rate-0.5)*2)*GAIN*attempts."""
        model = ForgettingModel()
        from dy3_polaris.l2.knowledge_tracer.forgetting import (
            MIN_STABILITY,
            STABILITY_GAIN,
        )
        attempts, correct = 10, 10
        rate = correct / attempts  # 1.0
        expected = (
            MIN_STABILITY
            + attempts * STABILITY_GAIN
            + max(0.0, (rate - 0.5) * 2.0) * STABILITY_GAIN * attempts
        )
        assert model.compute_stability(attempts, correct) == pytest.approx(expected)


# ============================================================
# 9. ForgettingModel — compute_review_interval
# ============================================================


class TestForgettingModelReviewInterval:
    """ForgettingModel.compute_review_interval 测试 — 推荐复习间隔."""

    def test_compute_review_interval_exists(self):
        """compute_review_interval 方法存在."""
        model = ForgettingModel()
        assert hasattr(model, "compute_review_interval")

    def test_compute_review_interval_positive(self):
        """mastery > target 时返回正的间隔 (小时)."""
        model = ForgettingModel()
        interval = model.compute_review_interval(0.9, stability=1.0, target_retention=0.8)
        assert isinstance(interval, float)
        assert interval > 0

    def test_compute_review_interval_formula(self):
        """公式: t = -ln(target/mastery) / (base_lambda/stability)."""
        model = ForgettingModel()
        mastery, stability, target = 0.9, 1.0, 0.8
        lam = model.base_lambda / stability
        expected = -math.log(target / mastery) / lam
        assert model.compute_review_interval(mastery, stability, target) == pytest.approx(
            expected, rel=1e-6
        )

    def test_higher_stability_longer_interval(self):
        """stability 越大 -> 衰减越慢 -> 复习间隔越长."""
        model = ForgettingModel()
        short = model.compute_review_interval(0.9, stability=1.0, target_retention=0.8)
        long = model.compute_review_interval(0.9, stability=5.0, target_retention=0.8)
        assert long > short

    def test_higher_target_retention_shorter_interval(self):
        """目标保留率越高 -> 越早复习 -> 间隔越短."""
        model = ForgettingModel()
        low_target = model.compute_review_interval(0.9, stability=1.0, target_retention=0.7)
        high_target = model.compute_review_interval(0.9, stability=1.0, target_retention=0.85)
        assert high_target < low_target

    def test_compute_review_interval_zero_mastery(self):
        """mastery <= 0 -> 间隔 0 (无需复习, 已无掌握度)."""
        model = ForgettingModel()
        assert model.compute_review_interval(0.0, stability=1.0, target_retention=0.9) == 0.0

    def test_compute_review_interval_target_ge_mastery(self):
        """target >= mastery -> 间隔 0 (已低于目标, 立即复习)."""
        model = ForgettingModel()
        assert model.compute_review_interval(0.9, stability=1.0, target_retention=0.95) == 0.0
        assert model.compute_review_interval(0.9, stability=1.0, target_retention=0.9) == 0.0

    def test_compute_review_interval_target_ge_one(self):
        """target >= 1.0 -> 间隔 0 (无法达到的保留率)."""
        model = ForgettingModel()
        assert model.compute_review_interval(0.9, stability=1.0, target_retention=1.0) == 0.0
