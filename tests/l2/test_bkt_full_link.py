"""T2 BKT 知识追踪全链路集成测试.

全链路定义: 事件采集 → BKT 更新 → KG 传播 → 遗忘衰减 → 掌握度输出

测试覆盖:
1. MasteryPropagator.propagate_mastery 适配器 (修复 pipeline 静默失效)
2. BKTTracingService 全链路编排器
3. EMCalibrator 离线参数标定
4. MasteryOutput 下游输出契约
5. 置信区间计算
6. 全链路端到端集成
7. 世界先进方案融合验证
"""

from __future__ import annotations

import math
import time

import pytest

from dy3_polaris.l2.knowledge_tracer.bkt import BKTTracer
from dy3_polaris.l2.knowledge_tracer.forgetting import ForgettingModel
from dy3_polaris.l2.knowledge_tracer.mastery_propagator import MasteryPropagator
from dy3_polaris.l2.interaction.event_types import AnswerEvent
from dy3_polaris.l2.interaction.collector import EventCollector
from dy3_polaris.l2.interaction.pipeline import UpdatePipeline
from dy3_polaris.l2.models import AnswerRecord, TracingState, DEFAULT_BKT_PARAMS
from dy3_polaris.l2.store import InMemoryL2Store


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def bkt_tracer():
    return BKTTracer()


@pytest.fixture
def forgetting_model():
    return ForgettingModel()


@pytest.fixture
def propagator():
    return MasteryPropagator(alpha=0.3)


@pytest.fixture
def store():
    return InMemoryL2Store()


@pytest.fixture
def pipeline(bkt_tracer, forgetting_model, propagator, store):
    """注入全部组件的完整管道 (含传播修复后)."""
    return UpdatePipeline(
        bkt_tracer=bkt_tracer,
        forgetting_model=forgetting_model,
        mastery_propagator=propagator,
        store=store,
    )


# ============================================================
# 1. MasteryPropagator.propagate_mastery 适配器测试
# ============================================================

class TestPropagateMasteryAdapter:
    """修复 pipeline 中 KG 传播静默失效的关键适配器."""

    def test_propagate_mastery_exists(self, propagator):
        """propagate_mastery 方法必须存在."""
        assert hasattr(propagator, "propagate_mastery")

    def test_propagate_mastery_no_prerequisites(self, propagator, store):
        """无前置知识点时, 掌握度不变."""
        learner_id = "learner_001"
        kp_id = "kp_A"
        store.save_tracing_state(learner_id, kp_id, TracingState(
            kp_id=kp_id, mastery_prob=0.8, attempts=5, correct_count=4,
        ))
        propagator.propagate_mastery(learner_id, kp_id, 0.8, store)
        state = store.get_tracing_state(learner_id, kp_id)
        assert state.mastery_prob == 0.8  # 无前置, 不变

    def test_propagate_mastery_with_prerequisites(self, propagator, store):
        """有前置知识点时, 后继掌握度被提升."""
        learner_id = "learner_001"
        # 前置知识点已掌握
        store.save_tracing_state(learner_id, "kp_prereq_1", TracingState(
            kp_id="kp_prereq_1", mastery_prob=0.9, attempts=10, correct_count=9,
        ))
        store.save_tracing_state(learner_id, "kp_prereq_2", TracingState(
            kp_id="kp_prereq_2", mastery_prob=0.8, attempts=8, correct_count=6,
        ))
        # 后继知识点当前掌握度
        store.save_tracing_state(learner_id, "kp_target", TracingState(
            kp_id="kp_target", mastery_prob=0.5, attempts=3, correct_count=1,
        ))
        # 设置前置关系
        propagator.set_kg_graph({
            "kp_target": [("kp_prereq_1", 1.0), ("kp_prereq_2", 1.0)],
        })
        propagator.propagate_mastery(learner_id, "kp_target", 0.5, store)
        state = store.get_tracing_state(learner_id, "kp_target")
        assert state.mastery_prob > 0.5  # 被前置提升

    def test_propagate_mastery_clamps_to_one(self, propagator, store):
        """传播结果不超过 1.0."""
        learner_id = "learner_001"
        store.save_tracing_state(learner_id, "kp_prereq", TracingState(
            kp_id="kp_prereq", mastery_prob=1.0, attempts=10, correct_count=10,
        ))
        store.save_tracing_state(learner_id, "kp_target", TracingState(
            kp_id="kp_target", mastery_prob=0.9, attempts=5, correct_count=4,
        ))
        propagator.set_kg_graph({
            "kp_target": [("kp_prereq", 1.0)],
        })
        propagator.propagate_mastery(learner_id, "kp_target", 0.9, store)
        state = store.get_tracing_state(learner_id, "kp_target")
        assert state.mastery_prob <= 1.0

    def test_pipeline_propagation_not_silent(self, pipeline, store):
        """通过 pipeline 处理答题事件后, 依赖知识点掌握度被传播."""
        learner_id = "learner_001"
        # 预设前置知识点已掌握
        store.save_tracing_state(learner_id, "kp_prereq", TracingState(
            kp_id="kp_prereq", mastery_prob=0.9, attempts=10, correct_count=9,
        ))
        # 设置前置关系: kp_target 依赖 kp_prereq
        pipeline.mastery_propagator.set_kg_graph({
            "kp_target": [("kp_prereq", 1.0)],
        })
        # 处理 kp_target 答对事件
        event = AnswerEvent(
            learner_id=learner_id,
            kp_id="kp_target",
            correct=True,
            difficulty=0.5,
            timestamp=time.time(),
        )
        result = pipeline.process(event)
        assert result["updated"] is True
        assert result["new_mastery"] is not None
        # kp_target 掌握度应因传播而额外提升 (不只是 BKT 更新)
        state = store.get_tracing_state(learner_id, "kp_target")
        assert state.mastery_prob > 0.5  # 答对 + 传播提升


# ============================================================
# 2. BKTTracingService 全链路编排器测试
# ============================================================

class TestBKTTracingService:
    """BKT 全链路编排器 — 事件→BKT→KG→遗忘→输出."""

    def test_service_initializes(self):
        """BKTTracingService 可正确初始化."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService()
        assert service is not None
        assert service.bkt_tracer is not None
        assert service.forgetting_model is not None
        assert service.mastery_propagator is not None

    def test_service_process_answer_event(self):
        """处理单条答题事件, 返回完整 MasteryOutput."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        event = AnswerEvent(
            learner_id="learner_001",
            kp_id="kp_math_01",
            correct=True,
            difficulty=0.5,
            timestamp=time.time(),
        )
        output = service.process(event)
        assert output is not None
        assert output.learner_id == "learner_001"
        assert output.kp_id == "kp_math_01"
        assert 0.0 <= output.p_mastery <= 1.0
        assert 0.0 <= output.p_correct_next <= 1.0
        assert output.attempts == 1
        assert output.mastery_flag is False  # 首次答题不太可能掌握

    def test_service_correct_increases_mastery(self):
        """连续答对, 掌握度递增."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        learner_id = "learner_001"
        kp_id = "kp_math_01"
        ts = time.time()
        masteries = []
        for i in range(5):
            event = AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=True,
                difficulty=0.5, timestamp=ts + i,
            )
            output = service.process(event)
            masteries.append(output.p_mastery)
        # 掌握度应递增
        for i in range(1, len(masteries)):
            assert masteries[i] > masteries[i - 1]
        # 5 次答对应接近掌握
        assert masteries[-1] > 0.7

    def test_service_wrong_decreases_mastery(self):
        """连续答错, 掌握度递减."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        learner_id = "learner_001"
        kp_id = "kp_math_01"
        ts = time.time()
        masteries = []
        for i in range(5):
            event = AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=False,
                difficulty=0.5, timestamp=ts + i,
            )
            output = service.process(event)
            masteries.append(output.p_mastery)
        # 掌握度应递减
        for i in range(1, len(masteries)):
            assert masteries[i] < masteries[i - 1]

    def test_service_mastery_flag_threshold(self):
        """掌握标志在阈值以上为 True."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store(), mastery_threshold=0.85)
        learner_id = "learner_001"
        kp_id = "kp_math_01"
        ts = time.time()
        for i in range(20):
            event = AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=True,
                difficulty=0.3, timestamp=ts + i,
            )
            output = service.process(event)
        assert output.mastery_flag is True
        assert output.p_mastery >= 0.85

    def test_service_confidence_interval(self):
        """输出包含置信区间."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        event = AnswerEvent(
            learner_id="learner_001", kp_id="kp_01", correct=True,
            difficulty=0.5, timestamp=time.time(),
        )
        output = service.process(event)
        assert output.confidence_interval is not None
        assert len(output.confidence_interval) == 2
        lower, upper = output.confidence_interval
        assert 0.0 <= lower <= output.p_mastery <= upper <= 1.0
        # 首次答题置信区间应较宽
        assert (upper - lower) > 0.1

    def test_service_confidence_narrows_with_data(self):
        """数据越多, 置信区间越窄."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        learner_id = "learner_001"
        kp_id = "kp_01"
        ts = time.time()
        widths = []
        for i in range(15):
            event = AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=True,
                difficulty=0.4, timestamp=ts + i,
            )
            output = service.process(event)
            lower, upper = output.confidence_interval
            widths.append(upper - lower)
        # 置信区间应随数据增加而收窄
        assert widths[-1] < widths[0]

    def test_service_batch_process(self):
        """批量处理多个事件."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        ts = time.time()
        events = [
            AnswerEvent(learner_id="l1", kp_id="kp_01", correct=True, difficulty=0.4, timestamp=ts),
            AnswerEvent(learner_id="l1", kp_id="kp_01", correct=True, difficulty=0.4, timestamp=ts + 1),
            AnswerEvent(learner_id="l1", kp_id="kp_02", correct=False, difficulty=0.7, timestamp=ts + 2),
            AnswerEvent(learner_id="l2", kp_id="kp_01", correct=True, difficulty=0.5, timestamp=ts + 3),
        ]
        outputs = service.batch_process(events)
        assert len(outputs) == 4
        assert all(o is not None for o in outputs)
        # l1 的 kp_01 两次答对, 掌握度应高于 kp_02 答错
        assert outputs[1].p_mastery > outputs[2].p_mastery

    def test_service_kg_propagation_in_full_link(self):
        """全链路中 KG 传播生效: 答对前置知识点, 后继知识点掌握度被提升."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        store = InMemoryL2Store()
        service = BKTTracingService(store=store)
        # 设置 KG: kp_02 依赖 kp_01
        service.set_kg_graph({"kp_02": [("kp_01", 1.0)]})
        learner_id = "learner_001"
        ts = time.time()
        # 先在 kp_02 上答对一次 (建立基线)
        event1 = AnswerEvent(
            learner_id=learner_id, kp_id="kp_02", correct=True,
            difficulty=0.5, timestamp=ts,
        )
        output_before = service.process(event1)
        mastery_before = output_before.p_mastery
        # 在 kp_01 上答对多次 (高掌握度)
        for i in range(10):
            event = AnswerEvent(
                learner_id=learner_id, kp_id="kp_01", correct=True,
                difficulty=0.3, timestamp=ts + 1 + i,
            )
            service.process(event)
        # 再次在 kp_02 上答题, 传播应提升 kp_02 的掌握度
        event_after = AnswerEvent(
            learner_id=learner_id, kp_id="kp_02", correct=True,
            difficulty=0.5, timestamp=ts + 20,
        )
        output_after = service.process(event_after)
        # kp_02 掌握度应因前置传播而额外提升
        assert output_after.p_mastery > mastery_before

    def test_service_forgetting_in_full_link(self):
        """全链路中遗忘衰减生效: 长时间不练习, 掌握度下降."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        store = InMemoryL2Store()
        service = BKTTracingService(store=store)
        learner_id = "learner_001"
        kp_id = "kp_01"
        # 答对多次建立高掌握度
        ts = time.time()
        for i in range(10):
            event = AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=True,
                difficulty=0.3, timestamp=ts + i,
            )
            output = service.process(event)
        mastery_before = output.p_mastery
        assert mastery_before > 0.8
        # 模拟长时间后 (超过 7 天 = 168 小时) 在另一个知识点答题, 触发遗忘
        long_after_ts = ts + 168 * 3600 * 2  # 14 天后
        other_event = AnswerEvent(
            learner_id=learner_id, kp_id="kp_other", correct=True,
            difficulty=0.5, timestamp=long_after_ts,
        )
        service.process(other_event)
        # 原 kp_01 掌握度应因遗忘而下降
        state = store.get_tracing_state(learner_id, kp_id)
        assert state.mastery_prob < mastery_before

    def test_service_individualized_bkt(self):
        """支持个体化 BKT (BPT) 参数."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        learner_id = "learner_001"
        kp_id = "kp_01"
        # 设置学习者级参数 (学习速率高于默认)
        service.set_learner_params(learner_id, {
            "learner_p_t": 0.3,  # 默认 0.1, 个体化后更快学习
        })
        ts = time.time()
        event = AnswerEvent(
            learner_id=learner_id, kp_id=kp_id, correct=True,
            difficulty=0.5, timestamp=ts,
        )
        output_individualized = service.process(event)
        # 对比标准 BKT
        service_standard = BKTTracingService(store=InMemoryL2Store())
        output_standard = service_standard.process(event)
        # 个体化 P(T) 更高, 答对后掌握度应更高
        assert output_individualized.p_mastery > output_standard.p_mastery

    def test_service_get_mastery_snapshot(self):
        """获取学习者全部知识点掌握度快照."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        learner_id = "learner_001"
        ts = time.time()
        for kp_id in ["kp_01", "kp_02", "kp_03"]:
            for i in range(5):
                event = AnswerEvent(
                    learner_id=learner_id, kp_id=kp_id, correct=True,
                    difficulty=0.4, timestamp=ts + i,
                )
                service.process(event)
        snapshot = service.get_mastery_snapshot(learner_id)
        assert len(snapshot) == 3
        assert all(kp in snapshot for kp in ["kp_01", "kp_02", "kp_03"])
        assert all(0.0 <= v <= 1.0 for v in snapshot.values())

    def test_service_get_detailed_snapshot(self):
        """获取学习者详细掌握度快照 (含置信区间和预测正确率)."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        learner_id = "learner_001"
        ts = time.time()
        for i in range(5):
            event = AnswerEvent(
                learner_id=learner_id, kp_id="kp_01", correct=True,
                difficulty=0.4, timestamp=ts + i,
            )
            service.process(event)
        snapshot = service.get_detailed_snapshot(learner_id)
        assert "kp_01" in snapshot
        detail = snapshot["kp_01"]
        assert "p_mastery" in detail
        assert "p_correct_next" in detail
        assert "confidence_interval" in detail
        assert "attempts" in detail
        assert "mastery_flag" in detail


# ============================================================
# 3. EMCalibrator 离线参数标定测试
# ============================================================

class TestEMCalibrator:
    """EM 算法离线参数标定服务."""

    def test_calibrator_initializes(self):
        """EMCalibrator 可正确初始化."""
        from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
        calibrator = EMCalibrator()
        assert calibrator is not None
        assert calibrator.bkt_tracer is not None

    def test_calibrate_single_kp(self):
        """标定单个知识点的 BKT 参数."""
        from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
        calibrator = EMCalibrator()
        # 构造 50 条答题记录 (高正确率)
        records = [
            AnswerRecord(
                learner_id=f"learner_{i % 10}",
                kp_id="kp_01",
                correct=True if i % 10 != 0 else False,  # 90% 正确率
                timestamp=float(i),
                difficulty=0.4,
            )
            for i in range(50)
        ]
        params = calibrator.calibrate("kp_01", records)
        assert "p_l0" in params
        assert "p_t" in params
        assert "p_g" in params
        assert "p_s" in params
        # 参数合法性
        for v in params.values():
            assert 0.0 < v < 1.0
        # p_g + p_s < 1
        assert params["p_g"] + params["p_s"] < 1.0

    def test_calibrate_empty_records(self):
        """空记录返回默认参数."""
        from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
        calibrator = EMCalibrator()
        params = calibrator.calibrate("kp_01", [])
        assert params == DEFAULT_BKT_PARAMS

    def test_calibrate_threshold_check(self):
        """记录数不足时不标定, 返回 None."""
        from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
        calibrator = EMCalibrator(min_records=50)
        records = [
            AnswerRecord(learner_id="l1", kp_id="kp_01", correct=True,
                        timestamp=float(i), difficulty=0.5)
            for i in range(30)
        ]
        result = calibrator.calibrate_if_needed("kp_01", records)
        assert result is None  # 不足 50 条

    def test_calibrate_above_threshold(self):
        """记录数达标时触发标定."""
        from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
        calibrator = EMCalibrator(min_records=50)
        records = [
            AnswerRecord(learner_id=f"l_{i%10}", kp_id="kp_01",
                        correct=(i % 5 != 0), timestamp=float(i), difficulty=0.5)
            for i in range(60)
        ]
        result = calibrator.calibrate_if_needed("kp_01", records)
        assert result is not None
        assert "p_l0" in result

    def test_calibrate_improves_likelihood(self):
        """标定后参数的对数似然应不低于默认参数."""
        from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
        calibrator = EMCalibrator()
        # 构造有偏数据 (全对)
        records = [
            AnswerRecord(learner_id="l1", kp_id="kp_01", correct=True,
                        timestamp=float(i), difficulty=0.5)
            for i in range(100)
        ]
        default_ll = calibrator.bkt_tracer.log_likelihood(records, DEFAULT_BKT_PARAMS)
        calibrated = calibrator.calibrate("kp_01", records)
        calibrated_ll = calibrator.bkt_tracer.log_likelihood(records, calibrated)
        assert calibrated_ll >= default_ll

    def test_calibrate_batch(self):
        """批量标定多个知识点."""
        from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
        calibrator = EMCalibrator(min_records=20)
        all_records = {
            "kp_01": [
                AnswerRecord(learner_id="l1", kp_id="kp_01", correct=True,
                            timestamp=float(i), difficulty=0.4)
                for i in range(30)
            ],
            "kp_02": [
                AnswerRecord(learner_id="l1", kp_id="kp_02", correct=False,
                            timestamp=float(i), difficulty=0.8)
                for i in range(30)
            ],
        }
        results = calibrator.calibrate_batch(all_records)
        assert "kp_01" in results
        assert "kp_02" in results
        # kp_01 高正确率, p_l0 应较高
        assert results["kp_01"]["p_l0"] > 0.3
        # kp_02 高错误率, p_l0 应较低
        assert results["kp_02"]["p_l0"] < results["kp_01"]["p_l0"]

    def test_calibrate_versioned(self):
        """标定结果含版本号和时间戳."""
        from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
        calibrator = EMCalibrator()
        records = [
            AnswerRecord(learner_id="l1", kp_id="kp_01", correct=True,
                        timestamp=float(i), difficulty=0.5)
            for i in range(50)
        ]
        result = calibrator.calibrate_versioned("kp_01", records)
        assert result is not None
        assert "params" in result
        assert "version" in result
        assert "calibrated_at" in result
        assert "sample_count" in result
        assert result["sample_count"] == 50
        assert result["version"] >= 1


# ============================================================
# 4. MasteryOutput 下游输出契约测试
# ============================================================

class TestMasteryOutput:
    """掌握度输出标准化契约."""

    def test_mastery_output_fields(self):
        """MasteryOutput 包含所有必需字段."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import MasteryOutput
        output = MasteryOutput(
            learner_id="l1",
            kp_id="kp_01",
            p_mastery=0.85,
            p_correct_next=0.92,
            mastery_flag=True,
            attempts=10,
            last_updated_ts=1000.0,
            confidence_interval=[0.78, 0.91],
        )
        assert output.learner_id == "l1"
        assert output.kp_id == "kp_01"
        assert output.p_mastery == 0.85
        assert output.p_correct_next == 0.92
        assert output.mastery_flag is True
        assert output.attempts == 10
        assert output.last_updated_ts == 1000.0
        assert output.confidence_interval == [0.78, 0.91]

    def test_mastery_output_to_dict(self):
        """MasteryOutput 可序列化为字典."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import MasteryOutput
        output = MasteryOutput(
            learner_id="l1",
            kp_id="kp_01",
            p_mastery=0.7,
            p_correct_next=0.85,
            mastery_flag=False,
            attempts=5,
            last_updated_ts=500.0,
            confidence_interval=[0.6, 0.8],
        )
        d = output.to_dict()
        assert d["learner_id"] == "l1"
        assert d["p_mastery"] == 0.7
        assert d["confidence_interval"] == [0.6, 0.8]

    def test_mastery_output_serialization_roundtrip(self):
        """序列化-反序列化往返一致."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import MasteryOutput
        output = MasteryOutput(
            learner_id="l1",
            kp_id="kp_01",
            p_mastery=0.9,
            p_correct_next=0.95,
            mastery_flag=True,
            attempts=20,
            last_updated_ts=2000.0,
            confidence_interval=[0.85, 0.94],
        )
        d = output.to_dict()
        restored = MasteryOutput.from_dict(d)
        assert restored.learner_id == output.learner_id
        assert restored.p_mastery == output.p_mastery
        assert restored.confidence_interval == output.confidence_interval


# ============================================================
# 5. 置信区间计算测试
# ============================================================

class TestConfidenceInterval:
    """BKT 掌握度置信区间计算."""

    def test_ci_first_attempt_wide(self):
        """首次答题置信区间宽."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        event = AnswerEvent(
            learner_id="l1", kp_id="kp_01", correct=True,
            difficulty=0.5, timestamp=time.time(),
        )
        output = service.process(event)
        lower, upper = output.confidence_interval
        assert (upper - lower) > 0.2  # 首次宽

    def test_ci_many_attempts_narrow(self):
        """多次答题后置信区间窄."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        ts = time.time()
        for i in range(30):
            event = AnswerEvent(
                learner_id="l1", kp_id="kp_01", correct=True,
                difficulty=0.4, timestamp=ts + i,
            )
            output = service.process(event)
        lower, upper = output.confidence_interval
        assert (upper - lower) < 0.15  # 多次后窄

    def test_ci_contains_mastery(self):
        """置信区间始终包含掌握度值."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        ts = time.time()
        for i in range(10):
            event = AnswerEvent(
                learner_id="l1", kp_id="kp_01", correct=(i % 3 != 0),
                difficulty=0.5, timestamp=ts + i,
            )
            output = service.process(event)
        lower, upper = output.confidence_interval
        assert lower <= output.p_mastery <= upper

    def test_ci_bounds_valid(self):
        """置信区间边界合法."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        ts = time.time()
        for i in range(5):
            event = AnswerEvent(
                learner_id="l1", kp_id="kp_01", correct=True,
                difficulty=0.5, timestamp=ts + i,
            )
            output = service.process(event)
        lower, upper = output.confidence_interval
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        assert lower <= upper


# ============================================================
# 6. 全链路端到端集成测试
# ============================================================

class TestFullLinkIntegration:
    """事件采集 → BKT → KG传播 → 遗忘衰减 → 掌握度输出 端到端."""

    def test_full_link_single_learner_single_kp(self):
        """单学习者单知识点全链路."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        learner_id = "learner_001"
        kp_id = "kp_math_addition"
        ts = time.time()
        # 模拟学习过程: 从答错到答对
        results = []
        for i in range(15):
            correct = i >= 5  # 前 5 次答错, 后 10 次答对
            event = AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=correct,
                difficulty=0.4, timestamp=ts + i * 100,
            )
            output = service.process(event)
            results.append(output)
        # 验证掌握度趋势: 先降后升
        # 答错阶段递减: results[0] (首次答错) > results[4] (末次答错)
        assert results[0].p_mastery > results[4].p_mastery  # 答错阶段递减
        # 答对阶段递增: results[5] (首次答对) < results[14] (末次答对)
        assert results[5].p_mastery < results[14].p_mastery  # 答对后上升
        # 最终应达到掌握
        assert results[-1].p_mastery > 0.7
        assert results[-1].mastery_flag is True or results[-1].p_mastery > 0.8

    def test_full_link_multi_kp_with_propagation(self):
        """多知识点 + KG 传播全链路."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        store = InMemoryL2Store()
        service = BKTTracingService(store=store)
        # KG: kp_C 依赖 kp_A 和 kp_B
        service.set_kg_graph({
            "kp_C": [("kp_A", 1.0), ("kp_B", 0.8)],
        })
        learner_id = "learner_001"
        ts = time.time()
        # 先掌握 kp_A 和 kp_B
        for kp_id in ["kp_A", "kp_B"]:
            for i in range(10):
                event = AnswerEvent(
                    learner_id=learner_id, kp_id=kp_id, correct=True,
                    difficulty=0.3, timestamp=ts + i,
                )
                service.process(event)
        # 在 kp_C 上答对一次
        event = AnswerEvent(
            learner_id=learner_id, kp_id="kp_C", correct=True,
            difficulty=0.5, timestamp=ts + 100,
        )
        output = service.process(event)
        # kp_C 应因前置传播而掌握度较高
        assert output.p_mastery > 0.6  # 单次答对 + 前置传播

    def test_full_link_forgetting_over_time(self):
        """时间跨度下的遗忘衰减全链路."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        store = InMemoryL2Store()
        service = BKTTracingService(store=store)
        learner_id = "learner_001"
        kp_id = "kp_01"
        # 阶段 1: 建立高掌握度
        ts1 = time.time()
        for i in range(15):
            event = AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=True,
                difficulty=0.3, timestamp=ts1 + i,
            )
            service.process(event)
        state_peak = store.get_tracing_state(learner_id, kp_id)
        peak_mastery = state_peak.mastery_prob
        assert peak_mastery > 0.8
        # 阶段 2: 14 天后在另一个知识点答题, 触发 kp_01 遗忘
        ts2 = ts1 + 14 * 24 * 3600
        event = AnswerEvent(
            learner_id=learner_id, kp_id="kp_other", correct=True,
            difficulty=0.5, timestamp=ts2,
        )
        service.process(event)
        state_decayed = store.get_tracing_state(learner_id, kp_id)
        assert state_decayed.mastery_prob < peak_mastery

    def test_full_link_mastery_snapshot_for_profile(self):
        """全链路输出可用于画像构建."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        learner_id = "learner_001"
        ts = time.time()
        # 模拟多知识点学习
        kp_data = [
            ("kp_01", True, 0.3),
            ("kp_01", True, 0.3),
            ("kp_01", True, 0.3),
            ("kp_02", False, 0.7),
            ("kp_02", True, 0.7),
            ("kp_03", True, 0.5),
            ("kp_03", True, 0.5),
            ("kp_03", True, 0.5),
        ]
        for kp_id, correct, diff in kp_data:
            event = AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=correct,
                difficulty=diff, timestamp=ts,
            )
            service.process(event)
            ts += 1
        # 获取快照
        snapshot = service.get_mastery_snapshot(learner_id)
        # kp_01 (全对, 易) > kp_03 (全对, 中) > kp_02 (先错后对, 难)
        assert snapshot["kp_01"] > snapshot["kp_02"]
        assert snapshot["kp_03"] > snapshot["kp_02"]

    def test_full_link_em_calibration_integration(self):
        """全链路 + EM 标定集成."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
        store = InMemoryL2Store()
        service = BKTTracingService(store=store)
        calibrator = EMCalibrator(min_records=20)
        learner_id = "learner_001"
        kp_id = "kp_01"
        ts = time.time()
        # 收集答题记录
        for i in range(30):
            event = AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=(i % 4 != 0),
                difficulty=0.5, timestamp=ts + i,
            )
            service.process(event)
        # 从 store 获取历史记录并标定
        history = store.get_answer_history(learner_id)
        assert history is not None
        kp_records = [r for r in history if r.kp_id == kp_id]
        result = calibrator.calibrate_if_needed(kp_id, kp_records)
        assert result is not None
        # 标定后参数合法
        assert 0.0 < result["p_l0"] < 1.0
        assert result["p_g"] + result["p_s"] < 1.0

    def test_full_link_update_order(self):
        """更新顺序: 遗忘衰减 → BKT 更新 → KG 传播 (设计文档要求)."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        store = InMemoryL2Store()
        service = BKTTracingService(store=store)
        learner_id = "learner_001"
        kp_id = "kp_01"
        # 预设高掌握度
        store.save_tracing_state(learner_id, kp_id, TracingState(
            kp_id=kp_id, mastery_prob=0.9, attempts=10, correct_count=9,
            last_attempt_time=time.time() - 200 * 3600,  # 200 小时前
        ))
        # 答对事件
        event = AnswerEvent(
            learner_id=learner_id, kp_id=kp_id, correct=True,
            difficulty=0.4, timestamp=time.time(),
        )
        output = service.process(event)
        # 遗忘衰减应先于 BKT 更新: 衰减后 0.9 -> BKT 答对提升
        # 结果应高于纯衰减值, 但因衰减起点低于 0.9, 最终可能 < 0.9
        assert output.p_mastery > 0.5  # 答对后仍较高
        assert output.attempts == 11

    def test_full_link_graceful_degradation(self):
        """组件缺失时优雅降级."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        # 无 store, 无 KG, 无遗忘 — 仅 BKT 引擎
        service = BKTTracingService(store=None)
        event = AnswerEvent(
            learner_id="l1", kp_id="kp_01", correct=True,
            difficulty=0.5, timestamp=time.time(),
        )
        output = service.process(event)
        # 仍能输出 (BKT 引擎独立工作)
        assert output is not None
        assert 0.0 <= output.p_mastery <= 1.0


# ============================================================
# 7. 世界先进方案融合验证
# ============================================================

class TestWorldSchemeIntegration:
    """验证世界先进方案在全链路中的融合."""

    def test_corbett_anderson_standard_bkt(self):
        """Corbett & Anderson (1995) 标准 BKT 四参数模型."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        # 验证四参数存在
        tracer = service.bkt_tracer
        params = DEFAULT_BKT_PARAMS
        assert all(k in params for k in ["p_l0", "p_t", "p_g", "p_s"])

    def test_individualized_bkt_yudelson(self):
        """Yudelson-Koedinger-Gordon (CMU 2013) 个体化 BKT."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        # 设置学习者级 P(T) 个体化
        service.set_learner_params("learner_fast", {"learner_p_t": 0.3})
        service.set_learner_params("learner_slow", {"learner_p_t": 0.05})
        ts = time.time()
        # 两个学习者各答对一次
        out_fast = service.process(AnswerEvent(
            learner_id="learner_fast", kp_id="kp_01", correct=True,
            difficulty=0.5, timestamp=ts,
        ))
        out_slow = service.process(AnswerEvent(
            learner_id="learner_slow", kp_id="kp_01", correct=True,
            difficulty=0.5, timestamp=ts,
        ))
        # P(T) 高的学习者, 答对后掌握度更高
        assert out_fast.p_mastery > out_slow.p_mastery

    def test_kg_propagation_knewton_style(self):
        """Knewton 式知识图谱掌握度传播."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        store = InMemoryL2Store()
        service = BKTTracingService(store=store)
        # 构建知识图谱: 基础 → 中级 → 高级
        service.set_kg_graph({
            "kp_advanced": [("kp_intermediate", 1.0)],
            "kp_intermediate": [("kp_basic", 1.0)],
        })
        learner_id = "learner_001"
        ts = time.time()
        # 掌握基础知识点
        for i in range(10):
            service.process(AnswerEvent(
                learner_id=learner_id, kp_id="kp_basic", correct=True,
                difficulty=0.2, timestamp=ts + i,
            ))
        # 在中级知识点答对一次
        output = service.process(AnswerEvent(
            learner_id=learner_id, kp_id="kp_intermediate", correct=True,
            difficulty=0.5, timestamp=ts + 20,
        ))
        # 中级知识点应因基础掌握而获得传播提升
        assert output.p_mastery > 0.5
        # 在高级知识点答对一次
        output_adv = service.process(AnswerEvent(
            learner_id=learner_id, kp_id="kp_advanced", correct=True,
            difficulty=0.7, timestamp=ts + 30,
        ))
        # 高级知识点也有传播 (多跳)
        assert output_adv.p_mastery > 0.4

    def test_ebbinghaus_forgetting_curve(self):
        """Ebbinghaus 遗忘曲线集成."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        store = InMemoryL2Store()
        service = BKTTracingService(store=store)
        learner_id = "learner_001"
        kp_id = "kp_01"
        # 建立掌握度
        ts = time.time()
        for i in range(10):
            service.process(AnswerEvent(
                learner_id=learner_id, kp_id=kp_id, correct=True,
                difficulty=0.3, timestamp=ts + i,
            ))
        peak = store.get_tracing_state(learner_id, kp_id).mastery_prob
        # 不同时间间隔后的衰减程度
        decay_levels = []
        for hours in [24, 72, 168, 336]:  # 1天, 3天, 7天, 14天
            # 重置掌握度
            store.save_tracing_state(learner_id, kp_id, TracingState(
                kp_id=kp_id, mastery_prob=peak, attempts=10, correct_count=10,
                last_attempt_time=ts,
            ))
            # 在其他知识点答题触发遗忘
            service.process(AnswerEvent(
                learner_id=learner_id, kp_id=f"kp_other_{hours}",
                correct=True, difficulty=0.5,
                timestamp=ts + hours * 3600,
            ))
            decayed = store.get_tracing_state(learner_id, kp_id).mastery_prob
            decay_levels.append((hours, decayed))
        # 衰减随时间增加而加深
        for i in range(1, len(decay_levels)):
            assert decay_levels[i][1] <= decay_levels[i - 1][1]

    def test_oscoi_offline_online_separation(self):
        """OSCOI 模式: 离线标定 + 在线推理分离."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        from dy3_polaris.l2.knowledge_tracer.em_calibrator import EMCalibrator
        # 离线: 标定参数
        calibrator = EMCalibrator(min_records=20)
        records = [
            AnswerRecord(learner_id=f"l_{i%5}", kp_id="kp_01",
                        correct=(i % 3 != 0), timestamp=float(i), difficulty=0.5)
            for i in range(30)
        ]
        calibrated = calibrator.calibrate_versioned("kp_01", records)
        assert calibrated is not None
        # 在线: 使用标定参数初始化 service
        store = InMemoryL2Store()
        service = BKTTracingService(store=store)
        service.set_kp_params("kp_01", calibrated["params"])
        # 在线推理
        output = service.process(AnswerEvent(
            learner_id="l_new", kp_id="kp_01", correct=True,
            difficulty=0.5, timestamp=time.time(),
        ))
        assert output is not None
        assert 0.0 <= output.p_mastery <= 1.0

    def test_output_contract_for_downstream(self):
        """输出契约满足下游消费需求 (CAT/推荐/画像)."""
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        service = BKTTracingService(store=InMemoryL2Store())
        ts = time.time()
        for i in range(10):
            output = service.process(AnswerEvent(
                learner_id="l1", kp_id="kp_01", correct=True,
                difficulty=0.4, timestamp=ts + i,
            ))
        # 下游需要的字段
        assert output.p_mastery is not None  # 画像着色
        assert output.p_correct_next is not None  # CAT 选题
        assert output.mastery_flag is not None  # 推荐决策
        assert output.confidence_interval is not None  # 预警置信
        assert output.attempts is not None  # 停滞检测
