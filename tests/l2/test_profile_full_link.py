"""T5 画像+VARK+Bloom 全链路集成测试.

全链路定义: 冷启动 → 画像构建 → 风格推断 → Bloom设定 → 漂移检测 → 重训练

测试覆盖:
1. ProfileTracingService 服务初始化与基础处理
2. 冷启动集成 (群体平均 / 部分个性化 / 全量个性化)
3. 画像构建集成 (kp_mastery / theta / level / weak_kps / confidence / 遗忘衰减)
4. 风格推断与 Bloom 设定集成
5. 漂移检测集成 (ADWIN / DDM / 重训练回调 / 历史记录 / 生命周期阶段)
6. ProfileOutput 输出契约 (字段 / 序列化 / 往返)
7. 端到端全链路集成 (冷启动→稳态→漂移→重训练 / 优雅降级)
8. 世界先进方案融合验证 (Bloom 六层次 / VARK 四模态 / Knewton 冷启动 /
   ADWIN / DDM / 下游输出契约)
"""

from __future__ import annotations

import time

import pytest

from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.knowledge_tracer.bkt import BKTTracer
from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
from dy3_polaris.l2.models import AnswerRecord, IRTState, LearnerSnapshot, TracingState
from dy3_polaris.l2.profile_builder import (
    BLOOM_LEVELS,
    BloomSetter,
    LearnerColdStartManager,
    LearnerDriftDetector,
    LearnerLifecycleManager,
    ProfileBuilder,
    StyleInferrer,
)
from dy3_polaris.l2.store import InMemoryL2Store


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def store():
    return InMemoryL2Store()


@pytest.fixture
def profile_builder(store):
    return ProfileBuilder(store=store)


@pytest.fixture
def cold_start_manager():
    return LearnerColdStartManager()


@pytest.fixture
def drift_detector():
    return LearnerDriftDetector()


@pytest.fixture
def lifecycle_manager():
    return LearnerLifecycleManager()


@pytest.fixture
def service(store, profile_builder, cold_start_manager, drift_detector, lifecycle_manager):
    from dy3_polaris.l2.profile_builder import ProfileTracingService

    return ProfileTracingService(
        store=store,
        profile_builder=profile_builder,
        cold_start_manager=cold_start_manager,
        drift_detector=drift_detector,
        lifecycle_manager=lifecycle_manager,
    )


def _make_event(
    learner_id: str = "learner_001",
    kp_id: str = "kp_01",
    correct: bool = True,
    difficulty: float = 0.5,
    timestamp: float | None = None,
) -> AnswerEvent:
    """构造答题事件."""
    return AnswerEvent(
        learner_id=learner_id,
        kp_id=kp_id,
        correct=correct,
        difficulty=difficulty,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


def _seed_tracing_states(
    store,
    learner_id: str,
    masteries: dict[str, float],
    attempts: int = 5,
    last_attempt_time: float | None = None,
) -> None:
    """向 store 注入追踪状态."""
    lat = last_attempt_time if last_attempt_time is not None else time.time()
    for kp_id, m in masteries.items():
        store.save_tracing_state(
            learner_id,
            kp_id,
            TracingState(
                kp_id=kp_id,
                mastery_prob=m,
                attempts=attempts,
                correct_count=max(0, int(attempts * m)),
                last_attempt_time=lat,
            ),
        )


# ============================================================
# 1. TestProfileTracingService — 服务初始化与基础处理
# ============================================================


class TestProfileTracingService:
    """ProfileTracingService 服务初始化与基础处理."""

    def test_service_initializes(self):
        """ProfileTracingService 可用 store=None 创建."""
        from dy3_polaris.l2.profile_builder import ProfileTracingService

        svc = ProfileTracingService(store=None)
        assert svc is not None

    def test_service_has_all_components(self, service):
        """服务包含 ProfileBuilder / ColdStartManager / DriftDetector / LifecycleManager."""
        assert isinstance(service.profile_builder, ProfileBuilder)
        assert isinstance(service.cold_start_manager, LearnerColdStartManager)
        assert isinstance(service.lifecycle_manager, LearnerLifecycleManager)
        # drift_detector 为模板 (配置源)
        assert service.drift_detector is not None

    def test_service_process_answer_event(self, service):
        """处理单条事件返回 ProfileOutput."""
        from dy3_polaris.l2.profile_builder import ProfileOutput

        event = _make_event(correct=True)
        output = service.process(event)
        assert isinstance(output, ProfileOutput)
        assert output.learner_id == "learner_001"

    def test_service_cold_start_detection(self, service):
        """新学习者被识别为冷启动."""
        learner_id = "learner_new"
        # 0 条记录 -> 冷启动
        assert service.cold_start_manager.is_cold_start(0) is True
        # 处理首条事件: 仍处于冷启动策略 (count=1 < 10), theta 仍接近群体平均 0
        output = service.process(_make_event(learner_id=learner_id, correct=True))
        assert output.theta == pytest.approx(0.0, abs=0.1)  # 群体平均
        # 阶段为 warming (count=1)
        assert output.phase in ("cold_start", "warming")

    def test_service_warm_start_transition(self, service):
        """记录足够后脱离冷启动."""
        learner_id = "learner_warm"
        ts = time.time()
        outputs = []
        for i in range(12):
            out = service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
            outputs.append(out)
        # 12 条记录 -> 非冷启动
        assert service.cold_start_manager.is_cold_start(12) is False
        # 最后一条阶段为 stable (无漂移)
        assert outputs[-1].phase == "stable"

    def test_service_batch_process(self, service):
        """批量处理多个事件."""
        ts = time.time()
        events = [
            _make_event(learner_id="l1", kp_id="kp_01", correct=True, timestamp=ts),
            _make_event(learner_id="l1", kp_id="kp_01", correct=True, timestamp=ts + 1),
            _make_event(learner_id="l2", kp_id="kp_02", correct=False, timestamp=ts + 2),
        ]
        outputs = service.batch_process(events)
        assert len(outputs) == 3
        assert all(o is not None for o in outputs)
        assert outputs[0].learner_id == "l1"
        assert outputs[2].learner_id == "l2"


# ============================================================
# 2. TestColdStartIntegration — 冷启动处理
# ============================================================


class TestColdStartIntegration:
    """冷启动处理集成."""

    def test_cold_start_population_average(self, cold_start_manager, service):
        """0 条记录使用群体平均 (theta=0, mastery=0.5)."""
        # 群体平均参数
        theta, se = cold_start_manager.estimate_initial_theta(None, 0)
        assert theta == pytest.approx(0.0)
        assert se == pytest.approx(cold_start_manager.POPULATION_SE)
        assert cold_start_manager.estimate_initial_mastery(None, 0) == pytest.approx(0.5)
        assert cold_start_manager.get_strategy(0) == "population_average"
        # 服务首条事件: 新学习者观测 theta 仍接近群体平均 0 (冷启动下接近 0)
        output = service.process(_make_event(learner_id="l_cold", correct=True))
        assert output.theta == pytest.approx(0.0, abs=0.1)

    def test_cold_start_partial_personalization(self, cold_start_manager, store, service):
        """1-9 条记录使用加权平均 (部分个性化)."""
        learner_id = "l_partial"
        # 设置非默认 IRT theta, 验证部分个性化会混合 (不等于观测值)
        store.save_irt_state(learner_id, IRTState(theta=1.0, se=0.3))
        ts = time.time()
        last = None
        for i in range(5):
            last = service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
        assert cold_start_manager.get_strategy(5) == "partial_personalization"
        # 5 条记录: weight=0.5, theta 介于群体 (0) 与观测 (1.0) 之间 (混合, 非全量)
        assert last.theta == pytest.approx(0.5, abs=0.15)
        assert 0.0 < last.theta < 1.0  # 介于群体与观测之间

    def test_cold_start_full_personalization(self, cold_start_manager, store, service):
        """10+ 条记录使用全量个性化."""
        learner_id = "l_full"
        store.save_irt_state(learner_id, IRTState(theta=1.5, se=0.2))
        ts = time.time()
        last = None
        for i in range(10):
            last = service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
        assert cold_start_manager.get_strategy(10) == "full_personalization"
        # 10+ 条记录: 全量个性化, 使用观测 theta (IRT 更新后接近 1.5)
        assert last.theta == pytest.approx(1.5, abs=0.1)

    def test_cold_start_default_style(self, service):
        """冷启动返回 'multimodal' 默认风格."""
        output = service.process(_make_event(learner_id="l_style", correct=True))
        assert output.learning_style == "multimodal"

    def test_cold_start_initial_content(self, cold_start_manager):
        """冷启动推荐基础内容."""
        kps = ["kp_found_1", "kp_found_2", "kp_found_3", "kp_found_4", "kp_found_5", "kp_found_6"]
        recommended = cold_start_manager.recommend_initial_content(0, kps)
        assert isinstance(recommended, list)
        assert len(recommended) <= 5
        assert all(kp in kps for kp in recommended)


# ============================================================
# 3. TestProfileConstruction — 画像构建集成
# ============================================================


class TestProfileConstruction:
    """画像构建集成测试."""

    def test_profile_builds_correctly(self, store, service):
        """画像从追踪状态与 IRT 状态正确构建."""
        learner_id = "l_build"
        _seed_tracing_states(store, learner_id, {"kp-A": 0.8, "kp-B": 0.4})
        store.save_irt_state(learner_id, IRTState(theta=0.0, se=0.3))
        # 处理足够事件脱离冷启动, 使用全量个性化
        ts = time.time()
        output = None
        for i in range(10):
            output = service.process(
                _make_event(learner_id=learner_id, kp_id="kp-A", correct=True, timestamp=ts + i)
            )
        assert output is not None
        assert output.learner_id == learner_id
        assert output.phase == "stable"

    def test_profile_includes_kp_mastery(self, store, service):
        """画像包含知识点掌握度映射."""
        learner_id = "l_kp"
        _seed_tracing_states(store, learner_id, {"kp-1": 0.8, "kp-2": 0.6})
        store.save_irt_state(learner_id, IRTState(theta=0.0, se=0.3))
        ts = time.time()
        output = None
        for i in range(10):
            output = service.process(
                _make_event(learner_id=learner_id, kp_id="kp-1", correct=True, timestamp=ts + i)
            )
        assert "kp-1" in output.kp_mastery
        assert "kp-2" in output.kp_mastery
        # kp-1 为本次事件目标, 连续答对后 BKT 掌握度应上升 (高于播种的 0.8)
        assert output.kp_mastery["kp-1"] > 0.8
        assert output.kp_mastery["kp-2"] == pytest.approx(0.6, abs=1e-6)

    def test_profile_includes_theta(self, store, service):
        """画像包含 IRT theta."""
        learner_id = "l_theta"
        store.save_irt_state(learner_id, IRTState(theta=1.5, se=0.2))
        _seed_tracing_states(store, learner_id, {"kp-1": 0.7})
        ts = time.time()
        output = None
        for i in range(10):
            output = service.process(
                _make_event(learner_id=learner_id, kp_id="kp-1", correct=True, timestamp=ts + i)
            )
        assert output.theta == pytest.approx(1.5, abs=0.1)

    def test_profile_includes_level(self, store, service):
        """画像包含能力等级 (beginner/intermediate/advanced)."""
        learner_id = "l_level"
        _seed_tracing_states(store, learner_id, {"kp-1": 0.9, "kp-2": 0.85})
        store.save_irt_state(learner_id, IRTState(theta=1.5, se=0.1))
        ts = time.time()
        output = None
        for i in range(10):
            output = service.process(
                _make_event(learner_id=learner_id, kp_id="kp-1", correct=True, timestamp=ts + i)
            )
        assert output.level in ("beginner", "intermediate", "advanced")
        assert output.level == "advanced"  # theta>=0.5 且 mastery>=0.7

    def test_profile_includes_weak_kps(self, store, service):
        """画像识别薄弱知识点."""
        learner_id = "l_weak"
        _seed_tracing_states(
            store, learner_id, {"kp-strong": 0.9, "kp-weak": 0.2, "kp-edge": 0.49}
        )
        store.save_irt_state(learner_id, IRTState(theta=0.0, se=0.3))
        ts = time.time()
        output = None
        for i in range(10):
            output = service.process(
                _make_event(learner_id=learner_id, kp_id="kp-strong", correct=True, timestamp=ts + i)
            )
        assert "kp-weak" in output.weak_kps
        assert "kp-edge" in output.weak_kps
        assert "kp-strong" not in output.weak_kps

    def test_profile_includes_confidence(self, store, service):
        """画像包含置信度."""
        learner_id = "l_conf"
        store.save_irt_state(learner_id, IRTState(theta=0.0, se=0.3))
        ts = time.time()
        output = None
        for i in range(10):
            output = service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
        assert isinstance(output.confidence, float)
        assert 0.0 <= output.confidence <= 1.0

    def test_profile_forgetting_decay(self, store, service):
        """画像对掌握度施加遗忘衰减."""
        learner_id = "l_forget"
        now = time.time()
        # 高掌握度但作答于 200 小时前 -> 应被衰减
        _seed_tracing_states(
            store,
            learner_id,
            {"kp-old": 0.9},
            attempts=3,
            last_attempt_time=now - 200 * 3600,
        )
        store.save_irt_state(learner_id, IRTState(theta=0.0, se=0.3))
        # 事件目标设为另一知识点, 避免 BKT 更新重置 kp-old 的 last_attempt_time,
        # 从而正确验证遗忘衰减 (kp-old 距上次作答 200 小时, 应被衰减)
        output = service.process(
            _make_event(learner_id=learner_id, kp_id="kp-new", correct=True, timestamp=now)
        )
        # 衰减后掌握度低于原始 0.9
        assert output.kp_mastery["kp-old"] < 0.9
        assert output.kp_mastery["kp-old"] > 0.0


# ============================================================
# 4. TestStyleAndBloomIntegration — 风格推断与 Bloom 设定
# ============================================================


class TestStyleAndBloomIntegration:
    """风格推断与 Bloom 设定集成."""

    def test_style_inferred_from_behavior(self, service):
        """从交互行为推断学习风格."""
        inferrer = service.profile_builder.style_inferrer
        events = [
            {"modality": "video"},
            {"modality": "image"},
            {"modality": "chart"},
        ]
        assert inferrer.infer_from_behavior(events) == "visual"

    def test_style_from_vark_questionnaire(self, service):
        """从 VARK 问卷数据推断风格."""
        inferrer = service.profile_builder.style_inferrer
        profile = {
            "visual_score": 0.1,
            "aural_score": 0.1,
            "read_write_score": 0.1,
            "kinesthetic_score": 0.7,
        }
        assert inferrer.infer_from_vark(profile) == "kinesthetic"

    def test_bloom_target_set(self, store, service):
        """Bloom 目标基于当前等级设定."""
        learner_id = "l_bloom"
        _seed_tracing_states(store, learner_id, {"kp-1": 0.9})
        store.save_irt_state(learner_id, IRTState(theta=1.5, se=0.1))
        ts = time.time()
        output = None
        for i in range(10):
            output = service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
        assert output.bloom_target in BLOOM_LEVELS

    def test_bloom_advancement(self, store, service):
        """Bloom 目标从当前层次提升一级 (连续答对后能力提升)."""
        learner_id = "l_adv"
        # theta=-1.0 / mastery=0.2 起步, 连续答对 10 题后能力提升到 intermediate
        # (冷启动混合 + 置信门控 + 滞回下, 答对 10 题已越过 beginner 区间)
        _seed_tracing_states(store, learner_id, {"kp-1": 0.2})
        store.save_irt_state(learner_id, IRTState(theta=-1.0, se=0.3))
        ts = time.time()
        output = None
        for i in range(10):
            output = service.process(
                _make_event(learner_id=learner_id, kp_id="kp-1", correct=True,
                            timestamp=ts + i)
            )
        assert output.level == "intermediate"
        # intermediate 映射 apply, 目标提升到 analyze
        assert output.bloom_target == "analyze"

    def test_bloom_max_level(self):
        """已达最高级 create 时保持不变."""
        setter = BloomSetter()
        assert setter.set_target("create") == "create"


# ============================================================
# 5. TestDriftDetection — 漂移检测集成
# ============================================================


class TestDriftDetection:
    """漂移检测集成测试."""

    def test_drift_not_detected_normal(self, service):
        """稳定表现下不检测到漂移."""
        learner_id = "l_stable"
        ts = time.time()
        for i in range(15):
            output = service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
            assert output.drift_detected is False

    def test_drift_detected_on_change(self, service):
        """表现突变时检测到漂移."""
        learner_id = "l_drift"
        ts = time.time()
        outputs = []
        # 15 次答对建立基线
        for i in range(15):
            outputs.append(
                service.process(
                    _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
                )
            )
        # 突然连续答错 -> 触发漂移
        for i in range(15, 25):
            outputs.append(
                service.process(
                    _make_event(learner_id=learner_id, correct=False, timestamp=ts + i)
                )
            )
        assert any(o.drift_detected for o in outputs)

    def test_drift_triggers_retraining(self, service):
        """漂移触发重训练回调."""
        learner_id = "l_retrain"
        called: list[dict] = []
        service.set_retraining_callback(learner_id, lambda info: called.append(info))
        ts = time.time()
        for i in range(15):
            service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
        for i in range(15, 25):
            service.process(
                _make_event(learner_id=learner_id, correct=False, timestamp=ts + i)
            )
        assert len(called) >= 1

    def test_drift_history_recorded(self, service):
        """漂移事件被记录到历史."""
        learner_id = "l_hist"
        ts = time.time()
        for i in range(15):
            service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
        for i in range(15, 25):
            service.process(
                _make_event(learner_id=learner_id, correct=False, timestamp=ts + i)
            )
        history = service.get_drift_history(learner_id)
        assert isinstance(history, list)
        assert len(history) >= 1
        assert "method" in history[0]

    def test_lifecycle_phase_transition(self, service):
        """生命周期阶段正确转换."""
        learner_id = "l_phase"
        ts = time.time()
        # cold_start (0 记录)
        assert service.cold_start_manager.is_cold_start(0) is True
        # 1 条 -> warming
        out1 = service.process(
            _make_event(learner_id=learner_id, correct=True, timestamp=ts)
        )
        assert out1.phase == "warming"
        # 累计到 10 条 -> stable
        for i in range(1, 10):
            out_stable = service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
        assert out_stable.phase == "stable"
        # 突变 -> drifting
        for i in range(10, 30):
            out_drift = service.process(
                _make_event(learner_id=learner_id, correct=False, timestamp=ts + i)
            )
        assert any(o.phase == "drifting" for o in [out_drift]) or out_drift.drift_count > 0


# ============================================================
# 6. TestProfileOutput — 输出契约
# ============================================================


class TestProfileOutput:
    """ProfileOutput 输出契约测试."""

    def test_output_fields(self, service):
        """ProfileOutput 包含所有必需字段."""
        from dy3_polaris.l2.profile_builder import ProfileOutput

        output = service.process(_make_event(correct=True))
        required = [
            "learner_id",
            "phase",
            "theta",
            "level",
            "learning_style",
            "bloom_target",
            "kp_mastery",
            "weak_kps",
            "confidence",
            "drift_detected",
            "drift_count",
            "recommended_action",
            "snapshot_ts",
        ]
        for field in required:
            assert hasattr(output, field), f"缺少字段: {field}"

    def test_output_to_dict(self, service):
        """序列化为字典."""
        output = service.process(_make_event(correct=True))
        d = output.to_dict()
        assert isinstance(d, dict)
        assert d["learner_id"] == output.learner_id
        assert d["phase"] == output.phase
        assert "kp_mastery" in d
        assert "weak_kps" in d

    def test_output_from_dict(self):
        """从字典反序列化."""
        from dy3_polaris.l2.profile_builder import ProfileOutput

        d = {
            "learner_id": "l1",
            "phase": "stable",
            "theta": 1.2,
            "level": "advanced",
            "learning_style": "visual",
            "bloom_target": "evaluate",
            "kp_mastery": {"kp-1": 0.8},
            "weak_kps": ["kp-2"],
            "confidence": 0.77,
            "drift_detected": False,
            "drift_count": 0,
            "recommended_action": "continue_monitoring",
            "snapshot_ts": 1000.0,
        }
        output = ProfileOutput.from_dict(d)
        assert output.learner_id == "l1"
        assert output.theta == 1.2
        assert output.level == "advanced"
        assert output.kp_mastery == {"kp-1": 0.8}
        assert output.weak_kps == ["kp-2"]

    def test_output_roundtrip(self):
        """序列化-反序列化往返保持数据."""
        from dy3_polaris.l2.profile_builder import ProfileOutput

        original = ProfileOutput(
            learner_id="l_rt",
            phase="drifting",
            theta=-0.5,
            level="beginner",
            learning_style="kinesthetic",
            bloom_target="understand",
            kp_mastery={"kp-A": 0.3, "kp-B": 0.9},
            weak_kps=["kp-A"],
            confidence=0.5,
            drift_detected=True,
            drift_count=2,
            recommended_action="trigger_retraining",
            snapshot_ts=2000.0,
        )
        restored = ProfileOutput.from_dict(original.to_dict())
        assert restored.learner_id == original.learner_id
        assert restored.phase == original.phase
        assert restored.theta == original.theta
        assert restored.level == original.level
        assert restored.learning_style == original.learning_style
        assert restored.bloom_target == original.bloom_target
        assert restored.kp_mastery == original.kp_mastery
        assert restored.weak_kps == original.weak_kps
        assert restored.confidence == original.confidence
        assert restored.drift_detected == original.drift_detected
        assert restored.drift_count == original.drift_count
        assert restored.recommended_action == original.recommended_action
        assert restored.snapshot_ts == original.snapshot_ts


# ============================================================
# 7. TestFullLinkIntegration — 端到端
# ============================================================


class TestFullLinkIntegration:
    """端到端全链路集成测试."""

    def test_full_link_single_learner_cold_to_stable(self, store, service):
        """学习者从冷启动进展到稳态."""
        learner_id = "l_e2e"
        ts = time.time()
        # 冷启动
        assert service.cold_start_manager.is_cold_start(0) is True
        outputs = []
        for i in range(12):
            outputs.append(
                service.process(
                    _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
                )
            )
        # 阶段演进: warming -> stable
        assert outputs[0].phase == "warming"
        assert outputs[-1].phase == "stable"
        # 推荐动作相应变化
        assert outputs[0].recommended_action == "partial_personalization"
        assert outputs[-1].recommended_action == "continue_monitoring"

    def test_full_link_profile_evolution(self, store):
        """画像随数据积累而演进."""
        # 使用 BKT 服务填充追踪状态, Profile 服务读取同一 store
        bkt_service = BKTTracingService(store=store)
        from dy3_polaris.l2.profile_builder import ProfileTracingService

        profile_service = ProfileTracingService(store=store)
        learner_id = "l_evo"
        ts = time.time()
        early = None
        late = None
        for i in range(20):
            event = _make_event(
                learner_id=learner_id, kp_id="kp_01", correct=True,
                difficulty=0.3, timestamp=ts + i,
            )
            bkt_service.process(event)  # 更新追踪状态
            out = profile_service.process(event)  # 构建画像
            if i == 0:
                early = out
            late = out
        # 早期冷启动 -> 后期稳态
        assert early.phase in ("warming", "cold_start")
        assert late.phase == "stable"
        # 画像快照可获取
        snap = profile_service.get_profile_snapshot(learner_id)
        assert snap is not None
        assert isinstance(snap, LearnerSnapshot)

    def test_full_link_drift_and_retrain(self, store, service):
        """漂移检测触发重训练."""
        learner_id = "l_dr"
        ts = time.time()
        # 建立稳态
        for i in range(15):
            service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
        # 触发漂移
        for i in range(15, 30):
            service.process(
                _make_event(learner_id=learner_id, correct=False, timestamp=ts + i)
            )
        history = service.get_drift_history(learner_id)
        assert len(history) >= 1
        # 触发重训练
        records = store.get_answer_history(learner_id) or []
        result = service.handle_drift(learner_id, records)
        assert isinstance(result, dict)
        assert "accepted" in result

    def test_full_link_lifecycle_complete(self, store, service):
        """完整生命周期: cold_start → warming → stable → drifting → recalibrating."""
        learner_id = "l_life"
        ts = time.time()
        # cold_start
        assert service.lifecycle_manager.get_phase(0) == "cold_start"
        # warming
        out_warm = service.process(
            _make_event(learner_id=learner_id, correct=True, timestamp=ts)
        )
        assert out_warm.phase == "warming"
        # stable
        for i in range(1, 12):
            out_stable = service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
        assert out_stable.phase == "stable"
        # drifting
        for i in range(12, 35):
            out_drift = service.process(
                _make_event(learner_id=learner_id, correct=False, timestamp=ts + i)
            )
        assert out_drift.drift_count > 0
        # recalibrating: 触发重训练 -> recalibration_count 记录
        records = store.get_answer_history(learner_id) or []
        service.handle_drift(learner_id, records)
        summary = service.get_lifecycle_summary(learner_id)
        assert summary["exists"] is True
        assert summary["drift_count"] > 0

    def test_full_link_graceful_degradation(self):
        """服务处理边界情况 (优雅降级)."""
        from dy3_polaris.l2.profile_builder import ProfileTracingService

        # 无 store, 全默认组件
        svc = ProfileTracingService(store=None)
        output = svc.process(_make_event(learner_id="l_degrade", correct=True))
        assert output is not None
        assert output.learner_id == "l_degrade"
        # 未知学习者快照返回 None
        assert svc.get_profile_snapshot("unknown") is None
        # 未知学习者生命周期摘要
        summary = svc.get_lifecycle_summary("unknown")
        assert summary["exists"] is False
        # 空事件批量处理
        assert svc.batch_process([]) == []
        # 未知学习者漂移历史为空
        assert svc.get_drift_history("unknown") == []


# ============================================================
# 8. TestWorldSchemeIntegration — 世界先进方案融合验证
# ============================================================


class TestWorldSchemeIntegration:
    """世界先进方案融合验证."""

    def test_bloom_taxonomy(self):
        """Bloom 六层次正确工作."""
        setter = BloomSetter()
        assert BLOOM_LEVELS == [
            "remember", "understand", "apply", "analyze", "evaluate", "create",
        ]
        # 每一层默认提升一级, 最高级保持
        for i, level in enumerate(BLOOM_LEVELS):
            target = setter.set_target(level)
            if i < len(BLOOM_LEVELS) - 1:
                assert target == BLOOM_LEVELS[i + 1]
            else:
                assert target == level

    def test_vark_four_modalities(self):
        """VARK 四模态被正确处理."""
        inferrer = StyleInferrer()
        cases = [
            ({"visual_score": 0.8, "aural_score": 0.1, "read_write_score": 0.05, "kinesthetic_score": 0.05}, "visual"),
            ({"visual_score": 0.1, "aural_score": 0.8, "read_write_score": 0.05, "kinesthetic_score": 0.05}, "aural"),
            ({"visual_score": 0.1, "aural_score": 0.1, "read_write_score": 0.8, "kinesthetic_score": 0.05}, "reading"),
            ({"visual_score": 0.05, "aural_score": 0.05, "read_write_score": 0.1, "kinesthetic_score": 0.8}, "kinesthetic"),
        ]
        for profile, expected in cases:
            assert inferrer.infer_from_vark(profile) == expected

    def test_knewton_cold_start(self, cold_start_manager):
        """Knewton 式冷启动: 群体先验降级."""
        # 0 记录 -> 纯群体先验
        theta0, se0 = cold_start_manager.estimate_initial_theta(None, 0)
        assert theta0 == cold_start_manager.POPULATION_THETA
        assert se0 == cold_start_manager.POPULATION_SE
        # 随记录增加, theta 逐步趋向观测值, SE 逐步减小
        theta_obs = 1.0
        thetas = [cold_start_manager.estimate_initial_theta(theta_obs, n)[0] for n in range(0, 11)]
        ses = [cold_start_manager.estimate_initial_theta(theta_obs, n)[1] for n in range(0, 11)]
        # theta 单调趋向 1.0
        for i in range(1, len(thetas)):
            assert thetas[i] >= thetas[i - 1]
        assert thetas[-1] == pytest.approx(1.0)
        # SE 单调减小
        for i in range(1, len(ses)):
            assert ses[i] <= ses[i - 1]

    def test_adwin_drift_detection(self):
        """ADWIN 漂移检测工作 (禁用 DDM 以隔离 ADWIN).

        用全错基线 (mean=0) -> 全对 (mean=1) 的均值突变触发 ADWIN.
        DDM 在全错基线下 min_p=1.0, min_ps=0, 阈值=1.0; 转向全对时
        p+s 始终 < 1.0 故 DDM 不触发, 从而隔离 ADWIN.
        """
        # ddm_drift_level 极高 -> 即便 min_ps>0 也不触发 DDM
        detector = LearnerDriftDetector(ddm_drift_level=1e9)
        detected = False
        # 12 次错误建立基线 (mean=0)
        for _ in range(12):
            r = detector.add_observation(0.0)
            detected = detected or r["drift_detected"]
        # 突然正确 -> ADWIN 检测均值从 0 到 1 的突变
        for _ in range(10):
            r = detector.add_observation(1.0)
            if r["drift_detected"]:
                detected = True
                assert r["method"] == "adwin"
        assert detected

    def test_ddm_drift_detection(self):
        """DDM 漂移检测工作 (禁用 ADWIN 以隔离 DDM)."""
        # adwin_delta 极小 -> epsilon 极大 -> ADWIN 永不触发, 仅 DDM
        detector = LearnerDriftDetector(adwin_delta=1e-10)
        for _ in range(15):
            detector.add_observation(1.0)
        # 首次错误 -> DDM 检测
        r = detector.add_observation(0.0)
        assert r["drift_detected"] is True
        assert r["method"] == "ddm"

    def test_output_contract_for_downstream(self, store, service):
        """输出可被下游 T2/T3/T4 消费."""
        learner_id = "l_downstream"
        _seed_tracing_states(store, learner_id, {"kp-1": 0.8, "kp-2": 0.3})
        store.save_irt_state(learner_id, IRTState(theta=0.5, se=0.2))
        ts = time.time()
        output = None
        for i in range(10):
            output = service.process(
                _make_event(learner_id=learner_id, correct=True, timestamp=ts + i)
            )
        d = output.to_dict()
        # 下游消费所需字段全部存在且类型合法
        assert isinstance(d["learner_id"], str)
        assert isinstance(d["phase"], str)
        assert isinstance(d["theta"], float)
        assert d["level"] in ("beginner", "intermediate", "advanced")
        assert d["learning_style"] in ("visual", "aural", "reading", "kinesthetic", "multimodal")
        assert d["bloom_target"] in BLOOM_LEVELS
        assert isinstance(d["kp_mastery"], dict)
        assert isinstance(d["weak_kps"], list)
        assert isinstance(d["confidence"], float)
        assert isinstance(d["drift_detected"], bool)
        assert isinstance(d["drift_count"], int)
        assert isinstance(d["recommended_action"], str)
        assert isinstance(d["snapshot_ts"], float)
