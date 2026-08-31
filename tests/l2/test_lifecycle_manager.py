"""L2 学习者生命周期管理器 / 漂移触发自动重训练 测试.

测试覆盖 (TDD):
1. LearnerLifecycleState: dataclass 字段 / to_dict / from_dict 往返.
2. LearnerLifecycleManager.get_phase: cold_start / warming / stable / drifting.
3. LearnerLifecycleManager.get_confidence: 置信度公式 / se 钳制 / 漂移折半.
4. LearnerLifecycleManager.recommend_action: 五阶段动作映射.
5. LearnerLifecycleManager.update: 状态更新 / 漂移检测 / zpd_zone / 阶段流转.
6. LearnerLifecycleManager.handle_drift: 重训练接受 / 回滚机制.
7. LearnerLifecycleManager.get_lifecycle_summary: 摘要 / 未知学习者.
8. DriftAwareRetrainer: retrain 接受 / 回滚 / should_retrain 判定.
9. LearnerDriftDetector 增强: 回调设置 / trigger_retraining / 漂移历史 / 自动触发.
10. BKTTracer 增量更新: incremental_fit 参数约束 / compute_gradient_single 梯度方向.
11. 线程安全: 并发 update 不抛异常.
"""

from __future__ import annotations

import threading

import pytest

from dy3_polaris.l2.ability_assessor.zpd import ZPDCalculator
from dy3_polaris.l2.knowledge_tracer.bkt import BKTTracer
from dy3_polaris.l2.models import DEFAULT_BKT_PARAMS, AnswerRecord, TracingState
from dy3_polaris.l2.profile_builder.drift_detector import LearnerDriftDetector
from dy3_polaris.l2.profile_builder.lifecycle_manager import (
    DriftAwareRetrainer,
    LearnerLifecycleManager,
    LearnerLifecycleState,
)


# ============================================================
# 辅助函数
# ============================================================


def _make_records(
    n: int,
    kp_id: str = "kp-1",
    correct: bool = True,
    start_ts: float = 1.0,
) -> list[AnswerRecord]:
    """构造 n 条同知识点答题记录."""
    return [
        AnswerRecord(
            learner_id="l1",
            kp_id=kp_id,
            correct=correct,
            timestamp=start_ts + float(i),
        )
        for i in range(n)
    ]


# ============================================================
# 1. LearnerLifecycleState
# ============================================================


class TestLearnerLifecycleState:
    """LearnerLifecycleState dataclass 测试."""

    def test_default_values(self):
        """默认字段值合理."""
        state = LearnerLifecycleState(learner_id="l1")
        assert state.learner_id == "l1"
        assert state.phase == "cold_start"
        assert state.record_count == 0
        assert state.last_drift_time is None
        assert state.drift_count == 0
        assert state.recalibration_count == 0
        assert state.current_theta == 0.0
        assert state.current_mastery == 0.5
        assert state.zpd_zone == "zpd"
        assert state.confidence == 0.0

    def test_to_dict_keys(self):
        """to_dict 含全部字段."""
        state = LearnerLifecycleState(learner_id="l1", phase="stable", record_count=10)
        d = state.to_dict()
        for key in (
            "learner_id",
            "phase",
            "record_count",
            "last_drift_time",
            "drift_count",
            "recalibration_count",
            "current_theta",
            "current_mastery",
            "zpd_zone",
            "confidence",
        ):
            assert key in d

    def test_from_dict(self):
        """from_dict 正确反序列化."""
        d = {
            "learner_id": "l2",
            "phase": "drifting",
            "record_count": 15,
            "last_drift_time": 123.4,
            "drift_count": 2,
            "recalibration_count": 1,
            "current_theta": 0.8,
            "current_mastery": 0.6,
            "zpd_zone": "frustration",
            "confidence": 0.3,
        }
        state = LearnerLifecycleState.from_dict(d)
        assert state.learner_id == "l2"
        assert state.phase == "drifting"
        assert state.record_count == 15
        assert state.last_drift_time == 123.4
        assert state.drift_count == 2
        assert state.recalibration_count == 1
        assert state.current_theta == 0.8
        assert state.current_mastery == 0.6
        assert state.zpd_zone == "frustration"
        assert state.confidence == 0.3

    def test_roundtrip(self):
        """to_dict / from_dict 往返保持一致."""
        state = LearnerLifecycleState(
            learner_id="l3",
            phase="warming",
            record_count=5,
            last_drift_time=None,
            drift_count=0,
            recalibration_count=0,
            current_theta=0.2,
            current_mastery=0.55,
            zpd_zone="zpd",
            confidence=0.42,
        )
        restored = LearnerLifecycleState.from_dict(state.to_dict())
        assert restored == state


# ============================================================
# 2. get_phase
# ============================================================


class TestGetPhase:
    """LearnerLifecycleManager.get_phase 阶段判定."""

    def test_cold_start_zero(self):
        """0 条记录 -> cold_start."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        assert mgr.get_phase(0, has_drift=False) == "cold_start"

    def test_warming(self):
        """1 ~ threshold-1 条 -> warming."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        assert mgr.get_phase(1, has_drift=False) == "warming"
        assert mgr.get_phase(9, has_drift=False) == "warming"

    def test_stable_at_threshold(self):
        """>= threshold 且无漂移 -> stable."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        assert mgr.get_phase(10, has_drift=False) == "stable"
        assert mgr.get_phase(100, has_drift=False) == "stable"

    def test_drifting_with_drift(self):
        """检测到漂移 -> drifting (漂移优先)."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        assert mgr.get_phase(10, has_drift=True) == "drifting"
        assert mgr.get_phase(50, has_drift=True) == "drifting"

    def test_drift_priority_over_warming(self):
        """漂移优先于 warming."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        assert mgr.get_phase(5, has_drift=True) == "drifting"

    def test_custom_threshold(self):
        """自定义阈值生效."""
        mgr = LearnerLifecycleManager(cold_start_threshold=3)
        assert mgr.get_phase(0, has_drift=False) == "cold_start"
        assert mgr.get_phase(2, has_drift=False) == "warming"
        assert mgr.get_phase(3, has_drift=False) == "stable"


# ============================================================
# 3. get_confidence
# ============================================================


class TestGetConfidence:
    """LearnerLifecycleManager.get_confidence 置信度计算.

    公式: confidence = (1 - min(1, se)) * min(1, record_count / threshold) * (0.5 if has_drift else 1.0)
    """

    def test_full_confidence(self):
        """充足记录 + se=0 + 无漂移 -> 1.0."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        assert mgr.get_confidence(10, 0.0, False) == pytest.approx(1.0)

    def test_zero_records(self):
        """0 条记录 -> 0.0 (数据因子为 0)."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        assert mgr.get_confidence(0, 0.0, False) == pytest.approx(0.0)

    def test_drift_halves_confidence(self):
        """漂移时置信度折半."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        no_drift = mgr.get_confidence(10, 0.0, False)
        with_drift = mgr.get_confidence(10, 0.0, True)
        assert with_drift == pytest.approx(no_drift * 0.5)
        assert with_drift == pytest.approx(0.5)

    def test_se_clipped_to_one(self):
        """se > 1 时钳制为 1 -> se 因子为 0."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        assert mgr.get_confidence(10, 2.0, False) == pytest.approx(0.0)

    def test_partial_data_factor(self):
        """记录数为阈值一半 -> 数据因子 0.5."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        # (1 - 0.2) * (5/10) * 1.0 = 0.8 * 0.5 = 0.4
        assert mgr.get_confidence(5, 0.2, False) == pytest.approx(0.4)

    def test_confidence_in_unit_range(self):
        """置信度始终落在 [0, 1]."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        for rc in (0, 1, 5, 10, 100):
            for se in (0.0, 0.1, 0.5, 1.0, 2.0):
                for d in (True, False):
                    c = mgr.get_confidence(rc, se, d)
                    assert 0.0 <= c <= 1.0


# ============================================================
# 4. recommend_action
# ============================================================


class TestRecommendAction:
    """LearnerLifecycleManager.recommend_action 五阶段动作映射."""

    @pytest.mark.parametrize(
        "phase,action",
        [
            ("cold_start", "use_population_average"),
            ("warming", "partial_personalization"),
            ("stable", "continue_monitoring"),
            ("drifting", "trigger_retraining"),
            ("recalibrating", "await_convergence"),
        ],
    )
    def test_action_mapping(self, phase, action):
        """各阶段对应推荐动作."""
        mgr = LearnerLifecycleManager()
        state = LearnerLifecycleState(learner_id="l1", phase=phase)
        assert mgr.recommend_action(state) == action

    def test_unknown_phase_fallback(self):
        """未知阶段回退为 continue_monitoring."""
        mgr = LearnerLifecycleManager()
        state = LearnerLifecycleState(learner_id="l1", phase="unknown_phase")
        assert mgr.recommend_action(state) == "continue_monitoring"


# ============================================================
# 5. update
# ============================================================


class TestUpdate:
    """LearnerLifecycleManager.update 状态更新测试."""

    def test_update_returns_lifecycle_state(self):
        """update 返回 LearnerLifecycleState 实例."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        state = mgr.update("l1", 0, 1.0)
        assert isinstance(state, LearnerLifecycleState)

    def test_update_cold_start(self):
        """0 条记录 -> phase=cold_start."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        state = mgr.update("l1", 0, 1.0)
        assert state.phase == "cold_start"
        assert state.record_count == 0

    def test_update_warming(self):
        """少量记录 -> phase=warming."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        state = mgr.update("l1", 5, 1.0)
        assert state.phase == "warming"

    def test_update_stable(self):
        """充足记录且无漂移 -> phase=stable."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        state = mgr.update("l1", 10, 1.0)
        assert state.phase == "stable"

    def test_update_records_theta_mastery(self):
        """update 记录 theta / mastery."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        state = mgr.update("l1", 10, 1.0, theta=0.7, mastery=0.8)
        assert state.current_theta == pytest.approx(0.7)
        assert state.current_mastery == pytest.approx(0.8)

    def test_update_zpd_zone_independent(self):
        """高能力 + 低难度 -> zpd_zone=independent."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        state = mgr.update("l1", 10, 1.0, theta=2.0, difficulty=-2.0)
        assert state.zpd_zone == "independent"

    def test_update_zpd_zone_frustration(self):
        """低能力 + 高难度 -> zpd_zone=frustration."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        state = mgr.update("l1", 10, 1.0, theta=-2.0, difficulty=2.0)
        assert state.zpd_zone == "frustration"

    def test_update_zpd_zone_zpd(self):
        """中能力 + 中难度 -> zpd_zone=zpd."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        state = mgr.update("l1", 10, 1.0, theta=0.0, difficulty=0.0)
        assert state.zpd_zone == "zpd"

    def test_update_confidence_in_range(self):
        """update 计算的 confidence 落在 [0, 1]."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        state = mgr.update("l1", 10, 1.0)
        assert 0.0 <= state.confidence <= 1.0

    def test_update_detects_drift(self):
        """稳定全对后突然全错 -> 检测到漂移 (drift_count > 0)."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        lid = "l1"
        # 建立稳定全对历史
        for i in range(15):
            mgr.update(lid, record_count=i + 1, observation=1.0)
        # 突然全错, 触发漂移
        drift_seen = False
        for j in range(15):
            state = mgr.update(lid, record_count=15 + j + 1, observation=0.0)
            if state.drift_count > 0 or state.phase == "drifting":
                drift_seen = True
                break
        assert drift_seen

    def test_update_drift_count_is_cumulative(self):
        """drift_count 累计不下降."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        lid = "l1"
        for i in range(15):
            mgr.update(lid, record_count=i + 1, observation=1.0)
        counts = []
        for j in range(15):
            state = mgr.update(lid, record_count=15 + j + 1, observation=0.0)
            counts.append(state.drift_count)
        # 累计计数单调不下降
        assert counts == sorted(counts)
        assert counts[-1] >= counts[0]

    def test_update_last_drift_time_set(self):
        """检测到漂移时 last_drift_time 被设置."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        lid = "l1"
        for i in range(15):
            mgr.update(lid, record_count=i + 1, observation=1.0)
        last_drift = None
        for j in range(15):
            state = mgr.update(lid, record_count=15 + j + 1, observation=0.0)
            if state.drift_count > 0:
                last_drift = state.last_drift_time
                break
        assert last_drift is not None


# ============================================================
# 6. handle_drift
# ============================================================


class TestHandleDrift:
    """LearnerLifecycleManager.handle_drift 漂移处理测试."""

    def test_handle_drift_return_keys(self):
        """返回字典含必需键."""
        bkt = BKTTracer()
        mgr = LearnerLifecycleManager()
        records = _make_records(20, correct=True)
        result = mgr.handle_drift("l1", records, bkt)
        for key in ("old_params", "new_params", "ll_improvement", "accepted"):
            assert key in result

    def test_handle_drift_accepts(self):
        """首次重训练 (从默认参数) 改善对数似然 -> accepted=True."""
        bkt = BKTTracer()
        mgr = LearnerLifecycleManager()
        records = _make_records(20, correct=True)
        result = mgr.handle_drift("l1", records, bkt)
        assert result["accepted"] is True
        assert result["ll_improvement"] > 0.0
        assert result["new_params"] != result["old_params"]

    def test_handle_drift_rollback_when_no_improvement(self):
        """当前参数已是最优 -> 重训练无改善 -> 回滚 accepted=False."""
        bkt = BKTTracer()
        mgr = LearnerLifecycleManager()
        records = _make_records(20, correct=True)
        # 预先拟合到最优参数, 使重训练无法进一步改善
        optimal = bkt.fit_params(records)
        mgr._bkt_params["l1"] = dict(optimal)
        result = mgr.handle_drift("l1", records, bkt)
        assert result["accepted"] is False
        assert result["new_params"] == result["old_params"]
        assert result["ll_improvement"] == pytest.approx(0.0)

    def test_handle_drift_old_params_is_default_for_new_learner(self):
        """新学习者的 old_params 为默认 BKT 参数."""
        bkt = BKTTracer()
        mgr = LearnerLifecycleManager()
        records = _make_records(5, correct=True)
        result = mgr.handle_drift("new-lid", records, bkt)
        assert result["old_params"]["p_l0"] == DEFAULT_BKT_PARAMS["p_l0"]
        assert result["old_params"]["p_t"] == DEFAULT_BKT_PARAMS["p_t"]


# ============================================================
# 7. get_lifecycle_summary
# ============================================================


class TestLifecycleSummary:
    """LearnerLifecycleManager.get_lifecycle_summary 测试."""

    def test_summary_unknown_learner(self):
        """未知学习者 -> exists=False."""
        mgr = LearnerLifecycleManager()
        summary = mgr.get_lifecycle_summary("unknown")
        assert summary["exists"] is False

    def test_summary_known_learner(self):
        """已知学习者 -> exists=True 且含推荐动作."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        mgr.update("l1", 10, 1.0, theta=0.5, mastery=0.7)
        summary = mgr.get_lifecycle_summary("l1")
        assert summary["exists"] is True
        assert summary["learner_id"] == "l1"
        assert summary["phase"] == "stable"
        assert summary["recommended_action"] == "continue_monitoring"

    def test_summary_includes_state_fields(self):
        """摘要包含生命周期状态字段."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        mgr.update("l1", 5, 1.0, theta=0.3, mastery=0.6)
        summary = mgr.get_lifecycle_summary("l1")
        for key in ("phase", "record_count", "drift_count", "confidence", "zpd_zone"):
            assert key in summary


# ============================================================
# 8. DriftAwareRetrainer
# ============================================================


class TestDriftAwareRetrainer:
    """DriftAwareRetrainer 重训练接受 / 回滚测试."""

    def test_retrain_return_keys(self):
        """retrain 返回字典含必需键."""
        bkt = BKTTracer()
        retrainer = DriftAwareRetrainer(bkt)
        records = _make_records(10, correct=True)
        result = retrainer.retrain(records, dict(DEFAULT_BKT_PARAMS))
        for key in (
            "new_params",
            "old_params",
            "ll_before",
            "ll_after",
            "improvement",
            "accepted",
        ):
            assert key in result

    def test_retrain_accepts_when_improvement_sufficient(self):
        """改善达到阈值 -> accepted=True, 参数更新."""
        bkt = BKTTracer()
        retrainer = DriftAwareRetrainer(bkt, min_improvement=0.0)
        records = _make_records(20, correct=True)
        old = dict(DEFAULT_BKT_PARAMS)
        result = retrainer.retrain(records, old)
        assert result["accepted"] is True
        assert result["ll_after"] >= result["ll_before"]
        assert result["improvement"] >= 0.0
        assert result["new_params"] != old

    def test_retrain_rolls_back_when_improvement_insufficient(self):
        """改善不足 -> 回滚, new_params == old_params."""
        bkt = BKTTracer()
        # 极高阈值, 任何改善都不满足 -> 必然回滚
        retrainer = DriftAwareRetrainer(bkt, min_improvement=1e9)
        records = _make_records(20, correct=True)
        old = dict(DEFAULT_BKT_PARAMS)
        result = retrainer.retrain(records, old)
        assert result["accepted"] is False
        assert result["new_params"] == old

    def test_retrain_ll_consistency(self):
        """ll_after - ll_before == improvement (一致性)."""
        bkt = BKTTracer()
        retrainer = DriftAwareRetrainer(bkt, min_improvement=0.0)
        records = _make_records(15, correct=True)
        result = retrainer.retrain(records, dict(DEFAULT_BKT_PARAMS))
        assert result["improvement"] == pytest.approx(
            result["ll_after"] - result["ll_before"]
        )

    def test_retrain_empty_records(self):
        """空记录 -> 不接受, 返回旧参数."""
        bkt = BKTTracer()
        retrainer = DriftAwareRetrainer(bkt)
        result = retrainer.retrain([], dict(DEFAULT_BKT_PARAMS))
        assert result["accepted"] is False
        assert result["new_params"] == result["old_params"]

    def test_should_retrain_on_drift(self):
        """检测到漂移 -> 应重训练 (无论间隔)."""
        retrainer = DriftAwareRetrainer(BKTTracer())
        assert retrainer.should_retrain(0, 5, drift_detected=True) is True
        assert retrainer.should_retrain(100, 105, drift_detected=True) is True

    def test_should_retrain_on_interval_exceeded(self):
        """距上次重训练超过 max_retraining_interval -> 应重训练."""
        retrainer = DriftAwareRetrainer(BKTTracer(), max_retraining_interval=100)
        assert retrainer.should_retrain(0, 100, drift_detected=False) is True
        assert retrainer.should_retrain(0, 150, drift_detected=False) is True

    def test_should_retrain_false(self):
        """无漂移且未超过间隔 -> 不应重训练."""
        retrainer = DriftAwareRetrainer(BKTTracer(), max_retraining_interval=100)
        assert retrainer.should_retrain(50, 60, drift_detected=False) is False
        assert retrainer.should_retrain(0, 99, drift_detected=False) is False


# ============================================================
# 9. LearnerDriftDetector 增强
# ============================================================


class TestDriftDetectorRetraining:
    """LearnerDriftDetector 重训练触发增强测试."""

    def test_set_retraining_callback(self):
        """set_retraining_callback 存储回调."""
        det = LearnerDriftDetector()

        def cb(info: dict) -> dict:
            return {"ok": True}

        det.set_retraining_callback(cb)
        assert det.trigger_retraining({"method": "adwin"}) == {"ok": True}

    def test_trigger_retraining_no_callback_returns_none(self):
        """未设置回调 -> trigger_retraining 返回 None."""
        det = LearnerDriftDetector()
        assert det.trigger_retraining({"method": "adwin"}) is None

    def test_trigger_retraining_with_callback(self):
        """设置回调 -> trigger_retraining 调用回调并返回结果."""
        det = LearnerDriftDetector()

        def cb(info: dict) -> dict:
            return {"method": info["method"], "retrained": True}

        det.set_retraining_callback(cb)
        result = det.trigger_retraining({"method": "ddm"})
        assert result == {"method": "ddm", "retrained": True}

    def test_get_drift_history_empty(self):
        """无漂移时历史为空."""
        det = LearnerDriftDetector()
        assert det.get_drift_history() == []

    def test_drift_history_records_events(self):
        """检测到漂移时记录到历史."""
        det = LearnerDriftDetector()
        for _ in range(15):
            det.add_observation(1.0)
        for _ in range(15):
            det.add_observation(0.0)
        history = det.get_drift_history()
        assert len(history) > 0
        assert "method" in history[0]
        assert history[0]["method"] in ("adwin", "ddm")

    def test_auto_trigger_retraining_on_drift(self):
        """检测到漂移时自动调用重训练回调."""
        det = LearnerDriftDetector()
        called: list[dict] = []

        def cb(info: dict) -> dict:
            called.append(info)
            return {"accepted": True}

        det.set_retraining_callback(cb)
        for _ in range(15):
            det.add_observation(1.0)
        triggered = False
        for _ in range(15):
            result = det.add_observation(0.0)
            if result["drift_detected"]:
                assert result.get("retraining_result") == {"accepted": True}
                triggered = True
                break
        assert triggered
        assert len(called) > 0

    def test_no_retraining_result_key_without_callback(self):
        """未设置回调时漂移结果不含 retraining_result 键."""
        det = LearnerDriftDetector()
        for _ in range(15):
            det.add_observation(1.0)
        for _ in range(15):
            result = det.add_observation(0.0)
            if result["drift_detected"]:
                assert "retraining_result" not in result
                break


# ============================================================
# 10. BKT 增量参数更新
# ============================================================


class TestBKTIncrementalFit:
    """BKTTracer.incremental_fit / compute_gradient_single 测试."""

    def test_incremental_fit_returns_params(self):
        """incremental_fit 返回含四参数的字典."""
        bkt = BKTTracer()
        state = TracingState(
            kp_id="kp-1", mastery_prob=0.9, bkt_params=dict(DEFAULT_BKT_PARAMS)
        )
        record = AnswerRecord("l1", "kp-1", correct=False, timestamp=1.0)
        params = bkt.incremental_fit(state, record, learning_rate=0.1)
        assert isinstance(params, dict)
        for key in ("p_l0", "p_t", "p_g", "p_s"):
            assert key in params

    def test_incremental_fit_maintains_constraint(self):
        """incremental_fit 保持 p_g + p_s < 1 约束."""
        bkt = BKTTracer()
        state = TracingState(
            kp_id="kp-1", mastery_prob=0.9, bkt_params=dict(DEFAULT_BKT_PARAMS)
        )
        record = AnswerRecord("l1", "kp-1", correct=False, timestamp=1.0)
        params = bkt.incremental_fit(state, record, learning_rate=0.5)
        assert params["p_g"] + params["p_s"] < 1.0
        # validate_params 不抛异常
        assert bkt.validate_params(params) is True

    def test_incremental_fit_valid_range(self):
        """incremental_fit 参数落在 (0, 1)."""
        bkt = BKTTracer()
        state = TracingState(
            kp_id="kp-1", mastery_prob=0.5, bkt_params=dict(DEFAULT_BKT_PARAMS)
        )
        for correct in (True, False):
            record = AnswerRecord("l1", "kp-1", correct=correct, timestamp=1.0)
            params = bkt.incremental_fit(state, record, learning_rate=1.0)
            for key in ("p_l0", "p_t", "p_g", "p_s"):
                assert 0.0 < params[key] < 1.0

    def test_incremental_fit_does_not_mutate_state(self):
        """incremental_fit 不修改入参 state 的 bkt_params."""
        bkt = BKTTracer()
        original = dict(DEFAULT_BKT_PARAMS)
        state = TracingState(kp_id="kp-1", mastery_prob=0.9, bkt_params=dict(original))
        record = AnswerRecord("l1", "kp-1", correct=False, timestamp=1.0)
        bkt.incremental_fit(state, record, learning_rate=0.5)
        assert state.bkt_params == original

    def test_compute_gradient_single_keys(self):
        """compute_gradient_single 返回四参数梯度."""
        bkt = BKTTracer()
        state = TracingState(
            kp_id="kp-1", mastery_prob=0.9, bkt_params=dict(DEFAULT_BKT_PARAMS)
        )
        record = AnswerRecord("l1", "kp-1", correct=False, timestamp=1.0)
        grads = bkt.compute_gradient_single(state, record, dict(DEFAULT_BKT_PARAMS))
        for key in ("p_l0", "p_t", "p_g", "p_s"):
            assert key in grads

    def test_gradient_slip_positive_on_unexpected_wrong(self):
        """高掌握度 + 答错 -> p_s 梯度为正 (失误解释意外答错)."""
        bkt = BKTTracer()
        state = TracingState(
            kp_id="kp-1", mastery_prob=0.9, bkt_params=dict(DEFAULT_BKT_PARAMS)
        )
        record = AnswerRecord("l1", "kp-1", correct=False, timestamp=1.0)
        grads = bkt.compute_gradient_single(state, record, dict(DEFAULT_BKT_PARAMS))
        assert grads["p_s"] > 0.0

    def test_gradient_guess_positive_on_unexpected_correct(self):
        """低掌握度 + 答对 -> p_g 梯度为正 (猜测解释意外答对)."""
        bkt = BKTTracer()
        state = TracingState(
            kp_id="kp-1", mastery_prob=0.1, bkt_params=dict(DEFAULT_BKT_PARAMS)
        )
        record = AnswerRecord("l1", "kp-1", correct=True, timestamp=1.0)
        grads = bkt.compute_gradient_single(state, record, dict(DEFAULT_BKT_PARAMS))
        assert grads["p_g"] > 0.0

    def test_incremental_fit_increases_slip_on_unexpected_wrong(self):
        """高掌握度 + 答错 -> incremental_fit 后 p_s 增大."""
        bkt = BKTTracer()
        state = TracingState(
            kp_id="kp-1", mastery_prob=0.9, bkt_params=dict(DEFAULT_BKT_PARAMS)
        )
        record = AnswerRecord("l1", "kp-1", correct=False, timestamp=1.0)
        params = bkt.incremental_fit(state, record, learning_rate=0.01)
        assert params["p_s"] > DEFAULT_BKT_PARAMS["p_s"]


# ============================================================
# 11. 线程安全
# ============================================================


class TestThreadSafety:
    """LearnerLifecycleManager 并发安全测试."""

    def test_concurrent_update_no_exception(self):
        """多线程并发 update 不抛异常."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)
        errors: list[Exception] = []

        def worker(lid: str) -> None:
            try:
                for i in range(50):
                    mgr.update(lid, record_count=i + 1, observation=1.0)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"l{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_update_all_learners_tracked(self):
        """并发后所有学习者均被追踪."""
        mgr = LearnerLifecycleManager(cold_start_threshold=10)

        def worker(lid: str) -> None:
            for i in range(20):
                mgr.update(lid, record_count=i + 1, observation=1.0)

        threads = [threading.Thread(target=worker, args=(f"l{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i in range(4):
            assert mgr.get_lifecycle_summary(f"l{i}")["exists"] is True
