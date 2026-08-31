"""L2 interaction 子模块测试 — 事件类型 / 采集器 / 更新管道.

测试覆盖 (TDD RED 阶段, 实现尚未存在, 预期因 ImportError 失败):
1. 事件类型 (event_types.py):
   - AnswerEvent: 字段 / 默认值 / to_answer_record() / to_dict()-from_dict() 往返
   - QueryEvent : 字段 / 默认值 / to_dict()-from_dict() 往返
   - BehaviorEvent: 字段 / 默认值 / 合法动作 / to_dict()-from_dict() 往返
2. EventCollector (collector.py):
   - collect_answer(): 生成 AnswerEvent, 默认 difficulty/question_id/timestamp
   - validate(): learner_id 非空 / difficulty [0,1] / timestamp 单调递增
   - collect_batch(): 按 question_id 去重 (保留最新)
3. UpdatePipeline (pipeline.py):
   - 依赖注入 (bkt_tracer / irt_estimator / store 均可缺省, 优雅降级)
   - process(): AnswerEvent 触发 BKT + IRT 更新, 返回更新摘要
   - process(): QueryEvent / BehaviorEvent 不触发 BKT/IRT
   - batch_process(): 按时间戳排序后逐个处理
"""

from __future__ import annotations

import time

import pytest

from dy3_polaris.l2.models import AnswerRecord, IRTState, TracingState
from dy3_polaris.l2.store import InMemoryL2Store
from dy3_polaris.l2.interaction import (
    AnswerEvent,
    BehaviorEvent,
    EventCollector,
    QueryEvent,
    UpdatePipeline,
)


# ============================================================
# 测试辅助: BKT 追踪器 / IRT 估计器假实现 (鸭子类型)
# ============================================================


class FakeBKTTracer:
    """BKT 追踪器假实现 — update(state, correct, timestamp) -> TracingState.

    简化语义: 答对 mastery +0.1, 答错 -0.1, 钳制到 [0,1].
    """

    def update(
        self, state: TracingState, correct: bool, timestamp: float
    ) -> TracingState:
        delta = 0.1 if correct else -0.1
        new_mastery = min(1.0, max(0.0, state.mastery_prob + delta))
        return TracingState(
            kp_id=state.kp_id,
            mastery_prob=new_mastery,
            attempts=state.attempts + 1,
            correct_count=state.correct_count + (1 if correct else 0),
            last_attempt_time=timestamp,
            bkt_params=dict(state.bkt_params),
        )


class FakeIRTEstimator:
    """IRT 估计器假实现 — update_theta(state, item_params, correct) -> IRTState.

    简化语义: 答对 theta +0.2, 答错 -0.2.
    """

    def update_theta(
        self, state: IRTState, item_params: dict, correct: bool
    ) -> IRTState:
        delta = 0.2 if correct else -0.2
        return IRTState(
            theta=state.theta + delta,
            se=state.se,
            response_count=state.response_count + 1,
            last_update_time=state.last_update_time,
        )


# ============================================================
# 1. AnswerEvent 测试
# ============================================================


class TestAnswerEvent:
    """AnswerEvent 答题事件测试."""

    def test_creation_all_fields(self):
        """AnswerEvent 全字段创建."""
        ev = AnswerEvent(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=True,
            difficulty=0.7,
            question_id="q-001",
            timestamp=1000.0,
        )
        assert ev.learner_id == "learner-001"
        assert ev.kp_id == "kp-001"
        assert ev.correct is True
        assert ev.difficulty == 0.7
        assert ev.question_id == "q-001"
        assert ev.timestamp == 1000.0

    def test_defaults(self):
        """AnswerEvent 默认值: difficulty=0.5, question_id=None, timestamp 自动."""
        before = time.time()
        ev = AnswerEvent(learner_id="l", kp_id="k", correct=True)
        after = time.time()
        assert ev.difficulty == 0.5
        assert ev.question_id is None
        assert isinstance(ev.timestamp, float)
        assert before <= ev.timestamp <= after

    def test_correct_is_bool(self):
        """AnswerEvent correct 为 bool 类型."""
        ev = AnswerEvent(learner_id="l", kp_id="k", correct=False)
        assert isinstance(ev.correct, bool)
        assert ev.correct is False

    def test_to_answer_record(self):
        """AnswerEvent.to_answer_record() 转为 L2 AnswerRecord, 字段一致."""
        ev = AnswerEvent(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=True,
            difficulty=0.8,
            question_id="q-009",
            timestamp=1234.5,
        )
        rec = ev.to_answer_record()
        assert isinstance(rec, AnswerRecord)
        assert rec.learner_id == "learner-001"
        assert rec.kp_id == "kp-001"
        assert rec.correct is True
        assert rec.difficulty == 0.8
        assert rec.question_id == "q-009"
        assert rec.timestamp == 1234.5

    def test_to_answer_record_defaults(self):
        """AnswerEvent 默认 difficulty/question_id 透传到 AnswerRecord."""
        ev = AnswerEvent(learner_id="l", kp_id="k", correct=True, timestamp=1.0)
        rec = ev.to_answer_record()
        assert rec.difficulty == 0.5
        assert rec.question_id is None

    def test_to_dict(self):
        """AnswerEvent.to_dict() 返回字典."""
        ev = AnswerEvent(
            learner_id="l1", kp_id="k1", correct=True, timestamp=1.0
        )
        d = ev.to_dict()
        assert isinstance(d, dict)
        assert d["learner_id"] == "l1"
        assert d["correct"] is True
        assert d["event_type"] == "answer"

    def test_roundtrip(self):
        """AnswerEvent to_dict()/from_dict() 往返一致."""
        ev = AnswerEvent(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=False,
            difficulty=0.8,
            question_id="q-009",
            timestamp=1234.5,
        )
        restored = AnswerEvent.from_dict(ev.to_dict())
        assert restored.learner_id == ev.learner_id
        assert restored.kp_id == ev.kp_id
        assert restored.correct == ev.correct
        assert restored.difficulty == ev.difficulty
        assert restored.question_id == ev.question_id
        assert restored.timestamp == ev.timestamp


# ============================================================
# 2. QueryEvent 测试
# ============================================================


class TestQueryEvent:
    """QueryEvent 查询事件测试."""

    def test_creation_all_fields(self):
        """QueryEvent 全字段创建."""
        ev = QueryEvent(
            learner_id="learner-001",
            query_text="如何求导数",
            kp_ids=["kp-1", "kp-2"],
            timestamp=1000.0,
        )
        assert ev.learner_id == "learner-001"
        assert ev.query_text == "如何求导数"
        assert ev.kp_ids == ["kp-1", "kp-2"]
        assert ev.timestamp == 1000.0

    def test_defaults(self):
        """QueryEvent 默认值: kp_ids=[], timestamp 自动."""
        before = time.time()
        ev = QueryEvent(learner_id="l", query_text="hello")
        after = time.time()
        assert ev.kp_ids == []
        assert isinstance(ev.timestamp, float)
        assert before <= ev.timestamp <= after

    def test_kp_ids_default_is_isolated(self):
        """QueryEvent 默认 kp_ids 不共享引用 (每个实例独立列表)."""
        e1 = QueryEvent(learner_id="l1", query_text="a")
        e2 = QueryEvent(learner_id="l2", query_text="b")
        e1.kp_ids.append("kp-x")
        assert e2.kp_ids == []

    def test_to_dict(self):
        """QueryEvent.to_dict() 返回字典."""
        ev = QueryEvent(
            learner_id="l1", query_text="hello", kp_ids=["kp-1"], timestamp=1.0
        )
        d = ev.to_dict()
        assert isinstance(d, dict)
        assert d["learner_id"] == "l1"
        assert d["query_text"] == "hello"
        assert d["kp_ids"] == ["kp-1"]
        assert d["event_type"] == "query"

    def test_roundtrip(self):
        """QueryEvent to_dict()/from_dict() 往返一致."""
        ev = QueryEvent(
            learner_id="learner-001",
            query_text="三角函数",
            kp_ids=["kp-a", "kp-b", "kp-c"],
            timestamp=999.0,
        )
        restored = QueryEvent.from_dict(ev.to_dict())
        assert restored.learner_id == ev.learner_id
        assert restored.query_text == ev.query_text
        assert restored.kp_ids == ev.kp_ids
        assert restored.timestamp == ev.timestamp


# ============================================================
# 3. BehaviorEvent 测试
# ============================================================


class TestBehaviorEvent:
    """BehaviorEvent 行为事件测试."""

    def test_creation_all_fields(self):
        """BehaviorEvent 全字段创建."""
        ev = BehaviorEvent(
            learner_id="learner-001",
            action="study",
            duration=120.0,
            kp_id="kp-001",
            timestamp=1000.0,
        )
        assert ev.learner_id == "learner-001"
        assert ev.action == "study"
        assert ev.duration == 120.0
        assert ev.kp_id == "kp-001"
        assert ev.timestamp == 1000.0

    def test_defaults(self):
        """BehaviorEvent 默认值: duration=0.0, kp_id=None, timestamp 自动."""
        before = time.time()
        ev = BehaviorEvent(learner_id="l", action="review")
        after = time.time()
        assert ev.duration == 0.0
        assert ev.kp_id is None
        assert isinstance(ev.timestamp, float)
        assert before <= ev.timestamp <= after

    def test_valid_actions(self):
        """BehaviorEvent 接受 study/review/skip 三种动作."""
        for action in ("study", "review", "skip"):
            ev = BehaviorEvent(learner_id="l", action=action)
            assert ev.action == action

    def test_to_dict(self):
        """BehaviorEvent.to_dict() 返回字典."""
        ev = BehaviorEvent(
            learner_id="l1",
            action="skip",
            duration=5.0,
            kp_id="kp-1",
            timestamp=1.0,
        )
        d = ev.to_dict()
        assert isinstance(d, dict)
        assert d["learner_id"] == "l1"
        assert d["action"] == "skip"
        assert d["duration"] == 5.0
        assert d["kp_id"] == "kp-1"
        assert d["event_type"] == "behavior"

    def test_roundtrip(self):
        """BehaviorEvent to_dict()/from_dict() 往返一致."""
        ev = BehaviorEvent(
            learner_id="learner-001",
            action="study",
            duration=300.0,
            kp_id="kp-001",
            timestamp=4321.0,
        )
        restored = BehaviorEvent.from_dict(ev.to_dict())
        assert restored.learner_id == ev.learner_id
        assert restored.action == ev.action
        assert restored.duration == ev.duration
        assert restored.kp_id == ev.kp_id
        assert restored.timestamp == ev.timestamp


# ============================================================
# 4. EventCollector 测试
# ============================================================


class TestEventCollector:
    """EventCollector 事件采集器测试 — 收集 / 验证 / 去重."""

    # --- collect_answer ---

    def test_collect_answer_returns_answer_event(self):
        """collect_answer 返回 AnswerEvent."""
        c = EventCollector()
        ev = c.collect_answer(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=True,
            difficulty=0.6,
            question_id="q-001",
            timestamp=1000.0,
        )
        assert isinstance(ev, AnswerEvent)
        assert ev.learner_id == "learner-001"
        assert ev.kp_id == "kp-001"
        assert ev.correct is True
        assert ev.difficulty == 0.6
        assert ev.question_id == "q-001"
        assert ev.timestamp == 1000.0

    def test_collect_answer_defaults(self):
        """collect_answer 默认 difficulty=0.5, question_id=None, timestamp 自动."""
        c = EventCollector()
        before = time.time()
        ev = c.collect_answer(learner_id="l", kp_id="k", correct=True)
        after = time.time()
        assert ev.difficulty == 0.5
        assert ev.question_id is None
        assert isinstance(ev.timestamp, float)
        assert before <= ev.timestamp <= after

    def test_collect_answer_records_timestamp(self):
        """collect_answer 后, 同 learner 的更早时间戳 validate 失败."""
        c = EventCollector()
        c.collect_answer(learner_id="l", kp_id="k", correct=True, timestamp=100.0)
        # 更早时间戳应 validate 失败
        early = AnswerEvent(
            learner_id="l", kp_id="k", correct=True, timestamp=50.0
        )
        assert c.validate(early) is False

    # --- validate ---

    def test_validate_valid_event(self):
        """validate 合法事件返回 True."""
        c = EventCollector()
        ev = AnswerEvent(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=True,
            difficulty=0.5,
            timestamp=100.0,
        )
        assert c.validate(ev) is True

    def test_validate_empty_learner_id(self):
        """validate learner_id 为空返回 False."""
        c = EventCollector()
        ev = AnswerEvent(
            learner_id="", kp_id="kp-001", correct=True, timestamp=100.0
        )
        assert c.validate(ev) is False

    def test_validate_none_learner_id(self):
        """validate learner_id 为 None 返回 False."""
        c = EventCollector()
        ev = AnswerEvent(
            learner_id=None,  # type: ignore[arg-type]
            kp_id="kp-001",
            correct=True,
            timestamp=100.0,
        )
        assert c.validate(ev) is False

    def test_validate_difficulty_below_zero(self):
        """validate difficulty < 0 返回 False."""
        c = EventCollector()
        ev = AnswerEvent(
            learner_id="l", kp_id="k", correct=True, difficulty=-0.1, timestamp=1.0
        )
        assert c.validate(ev) is False

    def test_validate_difficulty_above_one(self):
        """validate difficulty > 1 返回 False."""
        c = EventCollector()
        ev = AnswerEvent(
            learner_id="l", kp_id="k", correct=True, difficulty=1.1, timestamp=1.0
        )
        assert c.validate(ev) is False

    def test_validate_difficulty_boundary_zero(self):
        """validate difficulty=0.0 边界合法."""
        c = EventCollector()
        ev = AnswerEvent(
            learner_id="l", kp_id="k", correct=True, difficulty=0.0, timestamp=1.0
        )
        assert c.validate(ev) is True

    def test_validate_difficulty_boundary_one(self):
        """validate difficulty=1.0 边界合法."""
        c = EventCollector()
        ev = AnswerEvent(
            learner_id="l", kp_id="k", correct=True, difficulty=1.0, timestamp=1.0
        )
        assert c.validate(ev) is True

    def test_validate_monotonic_timestamp_increasing(self):
        """validate timestamp 单调递增: 递增序列全部通过."""
        c = EventCollector()
        assert c.validate(
            AnswerEvent(learner_id="l", kp_id="k", correct=True, timestamp=10.0)
        )
        assert c.validate(
            AnswerEvent(learner_id="l", kp_id="k", correct=True, timestamp=20.0)
        )
        assert c.validate(
            AnswerEvent(learner_id="l", kp_id="k", correct=True, timestamp=30.0)
        )

    def test_validate_monotonic_timestamp_equal(self):
        """validate timestamp 相等视为非递减 (通过)."""
        c = EventCollector()
        assert c.validate(
            AnswerEvent(learner_id="l", kp_id="k", correct=True, timestamp=10.0)
        )
        assert c.validate(
            AnswerEvent(learner_id="l", kp_id="k", correct=True, timestamp=10.0)
        )

    def test_validate_monotonic_timestamp_decreasing(self):
        """validate timestamp 回退返回 False."""
        c = EventCollector()
        c.validate(
            AnswerEvent(learner_id="l", kp_id="k", correct=True, timestamp=100.0)
        )
        assert c.validate(
            AnswerEvent(learner_id="l", kp_id="k", correct=True, timestamp=50.0)
        ) is False

    def test_validate_monotonic_per_learner_isolated(self):
        """validate 单调性按 learner_id 隔离."""
        c = EventCollector()
        # learner-a 在 t=100
        assert c.validate(
            AnswerEvent(learner_id="a", kp_id="k", correct=True, timestamp=100.0)
        )
        # learner-b 在 t=50 (< 100, 但不同 learner, 应通过)
        assert c.validate(
            AnswerEvent(learner_id="b", kp_id="k", correct=True, timestamp=50.0)
        )

    def test_validate_query_event(self):
        """validate QueryEvent (无 difficulty) 通过."""
        c = EventCollector()
        ev = QueryEvent(
            learner_id="l", query_text="hello", kp_ids=[], timestamp=1.0
        )
        assert c.validate(ev) is True

    def test_validate_behavior_event(self):
        """validate BehaviorEvent (无 difficulty) 通过."""
        c = EventCollector()
        ev = BehaviorEvent(
            learner_id="l", action="study", duration=10.0, timestamp=1.0
        )
        assert c.validate(ev) is True

    def test_validate_query_event_empty_learner_id(self):
        """validate QueryEvent 空 learner_id 失败."""
        c = EventCollector()
        ev = QueryEvent(learner_id="", query_text="x", timestamp=1.0)
        assert c.validate(ev) is False

    # --- collect_batch ---

    def test_collect_batch_dedup_by_question_id(self):
        """collect_batch 按 question_id 去重, 保留最新 (最大 timestamp)."""
        c = EventCollector()
        events = [
            AnswerEvent(
                learner_id="l", kp_id="k", correct=False, difficulty=0.5,
                question_id="q-1", timestamp=10.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-1", timestamp=30.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-1", timestamp=20.0,
            ),
        ]
        result = c.collect_batch(events)
        assert len(result) == 1
        # 保留 timestamp 最大的 (30.0, correct=True)
        assert result[0].timestamp == 30.0
        assert result[0].correct is True

    def test_collect_batch_keeps_distinct_question_ids(self):
        """collect_batch 不同 question_id 均保留."""
        c = EventCollector()
        events = [
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-1", timestamp=10.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-2", timestamp=20.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-3", timestamp=30.0,
            ),
        ]
        result = c.collect_batch(events)
        assert len(result) == 3

    def test_collect_batch_keeps_events_without_question_id(self):
        """collect_batch 保留 question_id=None 的事件 (不去重)."""
        c = EventCollector()
        events = [
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id=None, timestamp=10.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id=None, timestamp=20.0,
            ),
        ]
        result = c.collect_batch(events)
        assert len(result) == 2

    def test_collect_batch_mixed_dedup_and_none(self):
        """collect_batch 混合: 重复 question_id 去重 + None 全保留."""
        c = EventCollector()
        events = [
            AnswerEvent(
                learner_id="l", kp_id="k", correct=False, difficulty=0.5,
                question_id="q-1", timestamp=10.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-1", timestamp=40.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id=None, timestamp=20.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id=None, timestamp=30.0,
            ),
        ]
        result = c.collect_batch(events)
        # q-1 去重为 1 条 (ts=40.0) + 2 条 None = 3
        assert len(result) == 3
        q1 = [e for e in result if e.question_id == "q-1"]
        assert len(q1) == 1
        assert q1[0].timestamp == 40.0

    def test_collect_batch_empty(self):
        """collect_batch 空列表返回空列表."""
        c = EventCollector()
        assert c.collect_batch([]) == []

    def test_collect_batch_preserves_order_first_occurrence(self):
        """collect_batch 去重后保留首次出现位置, 值为最新."""
        c = EventCollector()
        events = [
            AnswerEvent(
                learner_id="l", kp_id="k", correct=False, difficulty=0.5,
                question_id="q-1", timestamp=10.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-2", timestamp=20.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-1", timestamp=50.0,
            ),
        ]
        result = c.collect_batch(events)
        assert len(result) == 2
        # q-1 出现在首位, 但值为最新 (ts=50.0)
        assert result[0].question_id == "q-1"
        assert result[0].timestamp == 50.0
        assert result[1].question_id == "q-2"


# ============================================================
# 5. UpdatePipeline 测试
# ============================================================


class TestUpdatePipeline:
    """UpdatePipeline 更新管道测试 — 依赖注入 / 优雅降级 / 实时更新."""

    # --- 构造与依赖注入 ---

    def test_init_defaults_none(self):
        """UpdatePipeline 默认依赖均为 None."""
        p = UpdatePipeline()
        assert p.bkt_tracer is None
        assert p.irt_estimator is None
        assert p.store is None

    def test_init_with_dependencies(self):
        """UpdatePipeline 接受依赖注入."""
        store = InMemoryL2Store()
        bkt = FakeBKTTracer()
        irt = FakeIRTEstimator()
        p = UpdatePipeline(
            bkt_tracer=bkt, irt_estimator=irt, store=store
        )
        assert p.bkt_tracer is bkt
        assert p.irt_estimator is irt
        assert p.store is store

    # --- process: 无依赖优雅降级 ---

    def test_process_answer_no_deps(self):
        """无依赖时 process AnswerEvent 返回 updated=False, 不抛异常."""
        p = UpdatePipeline()
        ev = AnswerEvent(
            learner_id="l", kp_id="k", correct=True, difficulty=0.5, timestamp=1.0
        )
        summary = p.process(ev)
        assert summary["learner_id"] == "l"
        assert summary["kp_id"] == "k"
        assert summary["event_type"] == "answer"
        assert summary["updated"] is False
        assert summary["new_mastery"] is None
        assert summary["new_theta"] is None

    def test_process_invalid_event(self):
        """process 非法事件 (空 learner_id) 返回 updated=False."""
        p = UpdatePipeline()
        ev = AnswerEvent(
            learner_id="", kp_id="k", correct=True, timestamp=1.0
        )
        summary = p.process(ev)
        assert summary["updated"] is False

    # --- process: store 持久化 ---

    def test_process_answer_persists_to_store(self):
        """process AnswerEvent 将 AnswerRecord 持久化到 store.answer_history."""
        store = InMemoryL2Store()
        p = UpdatePipeline(store=store)
        ev = AnswerEvent(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=True,
            difficulty=0.6,
            question_id="q-001",
            timestamp=10.0,
        )
        p.process(ev)
        history = store.get_answer_history("learner-001")
        assert history is not None
        assert len(history) == 1
        assert history[0].correct is True
        assert history[0].difficulty == 0.6
        assert history[0].question_id == "q-001"

    def test_process_answer_appends_history(self):
        """process 多次 AnswerEvent 追加到 answer_history."""
        store = InMemoryL2Store()
        p = UpdatePipeline(store=store)
        p.process(
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, timestamp=10.0
            )
        )
        p.process(
            AnswerEvent(
                learner_id="l", kp_id="k", correct=False, timestamp=20.0
            )
        )
        history = store.get_answer_history("l")
        assert history is not None
        assert len(history) == 2

    # --- process: BKT 更新 ---

    def test_process_answer_bkt_update(self):
        """process AnswerEvent 调用 bkt_tracer.update 更新 mastery."""
        store = InMemoryL2Store()
        p = UpdatePipeline(bkt_tracer=FakeBKTTracer(), store=store)
        ev = AnswerEvent(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=True,
            difficulty=0.5,
            timestamp=10.0,
        )
        summary = p.process(ev)
        assert summary["updated"] is True
        assert summary["new_mastery"] is not None
        # FakeBKTTracer: 0.5 + 0.1 = 0.6
        assert summary["new_mastery"] == pytest.approx(0.6)

    def test_process_answer_bkt_state_persisted(self):
        """process AnswerEvent 后 tracing_state 持久化到 store."""
        store = InMemoryL2Store()
        p = UpdatePipeline(bkt_tracer=FakeBKTTracer(), store=store)
        p.process(
            AnswerEvent(
                learner_id="learner-001",
                kp_id="kp-001",
                correct=True,
                difficulty=0.5,
                timestamp=10.0,
            )
        )
        state = store.get_tracing_state("learner-001", "kp-001")
        assert state is not None
        assert state.mastery_prob == pytest.approx(0.6)
        assert state.attempts == 1
        assert state.correct_count == 1

    def test_process_answer_bkt_wrong_decreases_mastery(self):
        """process AnswerEvent 答错降低 mastery."""
        store = InMemoryL2Store()
        p = UpdatePipeline(bkt_tracer=FakeBKTTracer(), store=store)
        # 先答对一次 (0.5 -> 0.6)
        p.process(
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, timestamp=10.0
            )
        )
        # 再答错一次 (0.6 -> 0.5)
        summary = p.process(
            AnswerEvent(
                learner_id="l", kp_id="k", correct=False, timestamp=20.0
            )
        )
        assert summary["new_mastery"] == pytest.approx(0.5)

    # --- process: IRT 更新 ---

    def test_process_answer_irt_update(self):
        """process AnswerEvent 调用 irt_estimator.update_theta 更新 theta."""
        store = InMemoryL2Store()
        p = UpdatePipeline(irt_estimator=FakeIRTEstimator(), store=store)
        ev = AnswerEvent(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=True,
            difficulty=0.5,
            timestamp=10.0,
        )
        summary = p.process(ev)
        assert summary["updated"] is True
        assert summary["new_theta"] is not None
        # FakeIRTEstimator: 0.0 + 0.2 = 0.2
        assert summary["new_theta"] == pytest.approx(0.2)

    def test_process_answer_irt_state_persisted(self):
        """process AnswerEvent 后 irt_state 持久化到 store."""
        store = InMemoryL2Store()
        p = UpdatePipeline(irt_estimator=FakeIRTEstimator(), store=store)
        p.process(
            AnswerEvent(
                learner_id="learner-001",
                kp_id="kp-001",
                correct=True,
                difficulty=0.5,
                timestamp=10.0,
            )
        )
        irt_state = store.get_irt_state("learner-001")
        assert irt_state is not None
        assert irt_state.theta == pytest.approx(0.2)
        assert irt_state.response_count == 1

    # --- process: 全依赖 ---

    def test_process_answer_all_dependencies(self):
        """process AnswerEvent 全依赖: 持久化 + BKT + IRT 同时更新."""
        store = InMemoryL2Store()
        p = UpdatePipeline(
            bkt_tracer=FakeBKTTracer(),
            irt_estimator=FakeIRTEstimator(),
            store=store,
        )
        ev = AnswerEvent(
            learner_id="learner-001",
            kp_id="kp-001",
            correct=True,
            difficulty=0.7,
            question_id="q-001",
            timestamp=10.0,
        )
        summary = p.process(ev)
        assert summary["updated"] is True
        assert summary["new_mastery"] == pytest.approx(0.6)
        assert summary["new_theta"] == pytest.approx(0.2)
        # 持久化校验
        assert len(store.get_answer_history("learner-001")) == 1
        assert store.get_tracing_state("learner-001", "kp-001") is not None
        assert store.get_irt_state("learner-001") is not None

    def test_process_answer_bkt_only(self):
        """仅注入 bkt_tracer: 更新 mastery, theta 为 None."""
        p = UpdatePipeline(bkt_tracer=FakeBKTTracer())
        ev = AnswerEvent(
            learner_id="l", kp_id="k", correct=True, timestamp=10.0
        )
        summary = p.process(ev)
        assert summary["updated"] is True
        assert summary["new_mastery"] is not None
        assert summary["new_theta"] is None

    def test_process_answer_irt_only(self):
        """仅注入 irt_estimator: 更新 theta, mastery 为 None."""
        p = UpdatePipeline(irt_estimator=FakeIRTEstimator())
        ev = AnswerEvent(
            learner_id="l", kp_id="k", correct=True, timestamp=10.0
        )
        summary = p.process(ev)
        assert summary["updated"] is True
        assert summary["new_theta"] is not None
        assert summary["new_mastery"] is None

    # --- process: QueryEvent / BehaviorEvent ---

    def test_process_query_event(self):
        """process QueryEvent 不触发 BKT/IRT, updated=False."""
        p = UpdatePipeline(
            bkt_tracer=FakeBKTTracer(), irt_estimator=FakeIRTEstimator()
        )
        ev = QueryEvent(
            learner_id="l", query_text="hello", kp_ids=["k"], timestamp=10.0
        )
        summary = p.process(ev)
        assert summary["event_type"] == "query"
        assert summary["learner_id"] == "l"
        assert summary["updated"] is False
        assert summary["new_mastery"] is None
        assert summary["new_theta"] is None

    def test_process_behavior_event(self):
        """process BehaviorEvent 不触发 BKT/IRT, updated=False."""
        p = UpdatePipeline(
            bkt_tracer=FakeBKTTracer(), irt_estimator=FakeIRTEstimator()
        )
        ev = BehaviorEvent(
            learner_id="l", action="study", duration=100.0, timestamp=10.0
        )
        summary = p.process(ev)
        assert summary["event_type"] == "behavior"
        assert summary["learner_id"] == "l"
        assert summary["updated"] is False
        assert summary["new_mastery"] is None
        assert summary["new_theta"] is None

    def test_process_query_invalid_learner(self):
        """process QueryEvent 空 learner_id 返回 updated=False."""
        p = UpdatePipeline()
        ev = QueryEvent(learner_id="", query_text="x", timestamp=10.0)
        summary = p.process(ev)
        assert summary["updated"] is False

    # --- batch_process ---

    def test_batch_process_returns_summaries(self):
        """batch_process 返回每个事件的更新摘要列表."""
        store = InMemoryL2Store()
        p = UpdatePipeline(
            bkt_tracer=FakeBKTTracer(),
            irt_estimator=FakeIRTEstimator(),
            store=store,
        )
        events = [
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-1", timestamp=10.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=False, difficulty=0.5,
                question_id="q-2", timestamp=20.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-3", timestamp=30.0,
            ),
        ]
        results = p.batch_process(events)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all(r["updated"] is True for r in results)

    def test_batch_process_sorts_by_timestamp(self):
        """batch_process 按时间戳排序后处理 (乱序输入也能正常更新)."""
        store = InMemoryL2Store()
        p = UpdatePipeline(
            bkt_tracer=FakeBKTTracer(), store=store
        )
        # 故意乱序
        events = [
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-3", timestamp=30.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-1", timestamp=10.0,
            ),
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                question_id="q-2", timestamp=20.0,
            ),
        ]
        results = p.batch_process(events)
        assert len(results) == 3
        # 全部成功更新 (排序后单调递增, validate 通过)
        assert all(r["updated"] is True for r in results)
        # 累计 3 次答对: 0.5 -> 0.6 -> 0.7 -> 0.8
        assert results[-1]["new_mastery"] == pytest.approx(0.8)

    def test_batch_process_empty(self):
        """batch_process 空列表返回空列表."""
        p = UpdatePipeline()
        assert p.batch_process([]) == []

    def test_batch_process_mixed_event_types(self):
        """batch_process 混合事件类型: AnswerEvent 更新, 其余不更新."""
        store = InMemoryL2Store()
        p = UpdatePipeline(
            bkt_tracer=FakeBKTTracer(), irt_estimator=FakeIRTEstimator(),
            store=store,
        )
        events = [
            AnswerEvent(
                learner_id="l", kp_id="k", correct=True, difficulty=0.5,
                timestamp=10.0,
            ),
            QueryEvent(
                learner_id="l", query_text="hi", kp_ids=[], timestamp=20.0
            ),
            BehaviorEvent(
                learner_id="l", action="review", duration=10.0, timestamp=30.0
            ),
        ]
        results = p.batch_process(events)
        assert len(results) == 3
        assert results[0]["event_type"] == "answer"
        assert results[0]["updated"] is True
        assert results[1]["event_type"] == "query"
        assert results[1]["updated"] is False
        assert results[2]["event_type"] == "behavior"
        assert results[2]["updated"] is False
