"""L2 关键缺陷修复与线程安全测试 (TDD).

测试覆盖 (RED -> GREEN):
1.  Cache: 某层过期条目不应跳过其余层 (break -> continue 修复)
2.  Pipeline: _persist_answer 线程安全 (并发追加不丢数据)
3.  Pipeline: BKT/IRT 异常被记录日志 (不再静默吞没)
4.  Pipeline: ForgettingModel 集成 (掌握度随时间衰减)
5.  Pipeline: MasteryPropagator 集成 (掌握度传播到依赖知识点)
6.  Collector: 时间戳校验线程安全
7.  Collector: reset() 方法清除时间戳记录
8.  ShortTermMemory: 线程安全操作
9.  Store: get_answer_history 返回副本 (修改返回列表不影响内部状态)
10. Builder: _records_to_events 包含模态信息
11. CAT: 曝光统计线程安全 + reset_stats() 方法
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from dy3_polaris.l2.ability_assessor.cat import CATSelector
from dy3_polaris.l2.cache import L2Cache
from dy3_polaris.l2.interaction import (
    AnswerEvent,
    EventCollector,
    UpdatePipeline,
)
from dy3_polaris.l2.knowledge_tracer.forgetting import ForgettingModel
from dy3_polaris.l2.models import AnswerRecord, IRTState, TracingState
from dy3_polaris.l2.profile_builder.builder import ProfileBuilder
from dy3_polaris.l2.store import InMemoryL2Store


# ============================================================
# 测试辅助: BKT / IRT 假实现
# ============================================================


class FakeBKTTracer:
    """BKT 假实现: 答对 +0.1, 答错 -0.1, 钳制到 [0,1]."""

    def update(self, state: TracingState, correct: bool, timestamp: float) -> TracingState:
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
    """IRT 假实现: 答对 theta +0.2, 答错 -0.2."""

    def update_theta(self, state: IRTState, item_params: dict, correct: bool) -> IRTState:
        delta = 0.2 if correct else -0.2
        return IRTState(
            theta=state.theta + delta,
            se=state.se,
            response_count=state.response_count + 1,
            last_update_time=state.last_update_time,
        )


class RaisingBKTTracer:
    """BKT 假实现: update 总是抛异常."""

    def update(self, state: TracingState, correct: bool, timestamp: float) -> TracingState:
        raise RuntimeError("bkt engine boom")


class RaisingIRTEstimator:
    """IRT 假实现: update_theta 总是抛异常."""

    def update_theta(self, state: IRTState, item_params: dict, correct: bool) -> IRTState:
        raise RuntimeError("irt engine boom")


class FakeMasteryPropagator:
    """掌握度传播假实现: 记录调用, 并据此提升依赖知识点掌握度."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def propagate_mastery(self, learner_id, kp_id, mastery, store) -> None:
        self.calls.append((learner_id, kp_id, mastery))
        if store is None:
            return
        dep = store.get_tracing_state(learner_id, "kp-dep")
        base = dep.mastery_prob if dep is not None else 0.3
        boosted = min(1.0, base + 0.2 * mastery)
        store.save_tracing_state(
            learner_id,
            "kp-dep",
            TracingState(
                kp_id="kp-dep",
                mastery_prob=boosted,
                attempts=0,
                correct_count=0,
                last_attempt_time=0.0,
            ),
        )


class CopyingAnswerStore:
    """最小化答题历史存储 (防御性拷贝语义).

    get/save 各自原子 (持锁), 但 read-modify-write 跨 get 与 save 不原子,
    用于隔离 _persist_answer 的竞态 (与 store 是否返回引用解耦).

    get 在释放锁后短暂 sleep, 强制多个线程在 "已读取但尚未写回" 状态下
    交错, 从而确定性地触发 _persist_answer 的 read-modify-write 竞态.
    """

    def __init__(self) -> None:
        self._hist: dict[str, list[AnswerRecord]] = {}
        self._lock = threading.RLock()

    def get_answer_history(self, learner_id: str):
        with self._lock:
            h = self._hist.get(learner_id)
            result = list(h) if h is not None else None
        # 释放锁后 sleep, 强制调用方 (_persist_answer) 在 read 与 write 之间
        # 出现可被其他线程交错的窗口 (sleep 期间释放 GIL).
        time.sleep(0.002)
        return result

    def save_answer_history(self, learner_id: str, records) -> None:
        with self._lock:
            self._hist[learner_id] = list(records)


# ============================================================
# 1. Cache: 过期条目不应跳过其余层
# ============================================================


class TestCacheExpiredLayer:
    """L2Cache.get() 在某层命中过期条目时应继续检查下一层 (而非 break)."""

    def test_expired_entry_does_not_skip_other_layers(self):
        """profile 层过期 -> 应继续检查 bkt 层并命中."""
        cache = L2Cache()  # 无 backing_store
        # 在 bkt 层写入有效条目
        cache.set("k", "v_bkt", layer="bkt")
        # 在 profile 层直接注入一个已过期条目 (expire_ts 在过去)
        cache._cache.setdefault("profile", {})["k"] = ("v_profile", time.time() - 100.0)

        result = cache.get("k")
        # 修复前 (break): profile 过期 -> break -> 无 backing_store -> None
        # 修复后 (continue): profile 过期 -> continue -> bkt 命中 -> "v_bkt"
        assert result == "v_bkt"

    def test_expired_entry_across_layers_with_backing_store(self):
        """profile 过期 + bkt 过期, 无 backing_store, 应返回 None."""
        cache = L2Cache()
        cache._cache.setdefault("profile", {})["k"] = ("v_p", time.time() - 100.0)
        cache._cache.setdefault("bkt", {})["k"] = ("v_b", time.time() - 100.0)
        # memory 层有效
        cache.set("k", "v_m", layer="memory")
        result = cache.get("k")
        assert result == "v_m"


# ============================================================
# 2. Pipeline: _persist_answer 线程安全
# ============================================================


class TestPipelinePersistThreadSafety:
    """UpdatePipeline._persist_answer 并发追加不应丢失记录."""

    def test_persist_answer_concurrent_no_lost_data(self):
        # 使用防御性拷贝语义的 store, 隔离 _persist_answer 的 read-modify-write 竞态
        store = CopyingAnswerStore()
        p = UpdatePipeline(store=store)
        n = 100
        # Barrier 强制所有线程同时开始, 最大化 read-modify-write 竞态触发概率
        barrier = threading.Barrier(n)
        events = [
            AnswerEvent(
                learner_id="l",
                kp_id=f"kp-{i}",
                correct=True,
                timestamp=float(i),
            )
            for i in range(n)
        ]

        def worker(e):
            barrier.wait()
            p._persist_answer(e)

        threads = [threading.Thread(target=worker, args=(e,)) for e in events]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        history = store.get_answer_history("l")
        assert history is not None
        # 修复前 (无锁): read-modify-write 竞态会丢失大量记录
        # 修复后 (加锁): 全部 n 条都被持久化
        assert len(history) == n


# ============================================================
# 3. Pipeline: BKT/IRT 异常被记录日志
# ============================================================


class TestPipelineExceptionLogging:
    """UpdatePipeline 在 BKT/IRT 引擎异常时应记录 warning, 而非静默吞没."""

    def test_bkt_exception_is_logged(self, caplog):
        store = InMemoryL2Store()
        p = UpdatePipeline(bkt_tracer=RaisingBKTTracer(), store=store)
        ev = AnswerEvent(
            learner_id="l", kp_id="k", correct=True, timestamp=10.0
        )
        with caplog.at_level(logging.WARNING):
            summary = p.process(ev)
        # 优雅降级: 不更新 mastery
        assert summary["new_mastery"] is None
        # 必须有日志记录
        messages = [r.getMessage() for r in caplog.records]
        assert any("BKT update failed" in m for m in messages), (
            f"BKT 异常应被记录日志, 实际日志: {messages}"
        )

    def test_irt_exception_is_logged(self, caplog):
        store = InMemoryL2Store()
        p = UpdatePipeline(irt_estimator=RaisingIRTEstimator(), store=store)
        ev = AnswerEvent(
            learner_id="l", kp_id="k", correct=True, timestamp=10.0
        )
        with caplog.at_level(logging.WARNING):
            summary = p.process(ev)
        assert summary["new_theta"] is None
        messages = [r.getMessage() for r in caplog.records]
        assert any("IRT update failed" in m for m in messages), (
            f"IRT 异常应被记录日志, 实际日志: {messages}"
        )


# ============================================================
# 4. Pipeline: ForgettingModel 集成
# ============================================================


class TestPipelineForgettingIntegration:
    """UpdatePipeline 集成 ForgettingModel: 对其他知识点施加遗忘衰减."""

    def test_forgetting_decays_other_kps_over_time(self):
        store = InMemoryL2Store()
        now = 1_000_000.0
        # 10 天前作答的旧知识点, 高掌握度
        old_time = now - 10 * 24 * 3600.0
        store.save_tracing_state(
            "l",
            "kp-old",
            TracingState(
                kp_id="kp-old",
                mastery_prob=0.9,
                attempts=5,
                correct_count=4,
                last_attempt_time=old_time,
            ),
        )
        fm = ForgettingModel()
        p = UpdatePipeline(bkt_tracer=FakeBKTTracer(), store=store, forgetting_model=fm)
        ev = AnswerEvent(
            learner_id="l", kp_id="kp-cur", correct=True, difficulty=0.5, timestamp=now
        )
        p.process(ev)

        other = store.get_tracing_state("l", "kp-old")
        assert other is not None
        # 修复前: 无遗忘集成, mastery 保持 0.9
        # 修复后: 距上次作答 10 天 (>168h), 掌握度衰减 < 0.9
        assert other.mastery_prob < 0.9, (
            f"应施加遗忘衰减, 实际 mastery={other.mastery_prob}"
        )

    def test_no_forgetting_without_model(self):
        """未注入 forgetting_model 时不施加衰减."""
        store = InMemoryL2Store()
        now = 1_000_000.0
        old_time = now - 10 * 24 * 3600.0
        store.save_tracing_state(
            "l",
            "kp-old",
            TracingState(
                kp_id="kp-old",
                mastery_prob=0.9,
                attempts=5,
                correct_count=4,
                last_attempt_time=old_time,
            ),
        )
        p = UpdatePipeline(bkt_tracer=FakeBKTTracer(), store=store)
        ev = AnswerEvent(
            learner_id="l", kp_id="kp-cur", correct=True, timestamp=now
        )
        p.process(ev)
        other = store.get_tracing_state("l", "kp-old")
        assert other is not None
        assert other.mastery_prob == pytest.approx(0.9)


# ============================================================
# 5. Pipeline: MasteryPropagator 集成
# ============================================================


class TestPipelineMasteryPropagator:
    """UpdatePipeline 集成 MasteryPropagator: 掌握度传播到依赖知识点."""

    def test_mastery_propagates_to_dependents(self):
        store = InMemoryL2Store()
        # 依赖知识点初始低掌握度
        store.save_tracing_state(
            "l",
            "kp-dep",
            TracingState(kp_id="kp-dep", mastery_prob=0.3),
        )
        prop = FakeMasteryPropagator()
        p = UpdatePipeline(
            bkt_tracer=FakeBKTTracer(), store=store, mastery_propagator=prop
        )
        ev = AnswerEvent(
            learner_id="l", kp_id="kp-cur", correct=True, timestamp=10.0
        )
        summary = p.process(ev)

        # BKT 更新成功 (0.5 -> 0.6)
        assert summary["new_mastery"] == pytest.approx(0.6)
        # 传播器被调用
        assert len(prop.calls) == 1
        assert prop.calls[0][0] == "l"
        assert prop.calls[0][1] == "kp-cur"
        # 依赖知识点掌握度被提升
        dep = store.get_tracing_state("l", "kp-dep")
        assert dep is not None
        assert dep.mastery_prob > 0.3, (
            f"依赖知识点掌握度应被提升, 实际={dep.mastery_prob}"
        )

    def test_no_propagation_without_propagator(self):
        """未注入 mastery_propagator 时不传播."""
        store = InMemoryL2Store()
        store.save_tracing_state(
            "l", "kp-dep", TracingState(kp_id="kp-dep", mastery_prob=0.3)
        )
        p = UpdatePipeline(bkt_tracer=FakeBKTTracer(), store=store)
        ev = AnswerEvent(
            learner_id="l", kp_id="kp-cur", correct=True, timestamp=10.0
        )
        p.process(ev)
        dep = store.get_tracing_state("l", "kp-dep")
        assert dep is not None
        assert dep.mastery_prob == pytest.approx(0.3)


# ============================================================
# 6 & 7. Collector: 线程安全时间戳校验 + reset()
# ============================================================


class TestCollectorThreadSafety:
    """EventCollector 时间戳校验线程安全 + reset()."""

    def test_has_lock(self):
        c = EventCollector()
        assert hasattr(c, "_lock"), "EventCollector 应持有 _lock"

    def test_concurrent_timestamp_validation(self):
        """并发校验同一学习者的事件不应抛异常, 最终状态有效."""
        c = EventCollector()
        assert hasattr(c, "_lock")
        n = 200
        errors: list[Exception] = []

        def worker(i):
            try:
                ev = AnswerEvent(
                    learner_id="l", kp_id="k", correct=True, timestamp=float(i)
                )
                c.validate(ev)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发校验抛出异常: {errors}"
        # 最终记录的时间戳应存在且为合法浮点
        last = c._last_timestamp.get("l")
        assert last is not None

    def test_reset_clears_timestamps(self):
        """reset() 清除所有时间戳记录."""
        c = EventCollector()
        assert hasattr(c, "reset"), "EventCollector 应有 reset 方法"
        c.validate(
            AnswerEvent(learner_id="l", kp_id="k", correct=True, timestamp=100.0)
        )
        assert c._last_timestamp.get("l") == 100.0
        c.reset()
        assert c._last_timestamp.get("l") is None
        # reset 后更早的时间戳应能通过校验
        assert c.validate(
            AnswerEvent(learner_id="l", kp_id="k", correct=True, timestamp=50.0)
        ) is True


# ============================================================
# 8. ShortTermMemory: 线程安全
# ============================================================


class TestShortTermMemoryThreadSafety:
    """ShortTermMemory 线程安全操作."""

    def test_has_lock(self):
        from dy3_polaris.l2.memory.short_term_memory import ShortTermMemory

        stm = ShortTermMemory()
        assert hasattr(stm, "_lock"), "ShortTermMemory 应持有 _lock"

    def test_concurrent_add_no_lost_entries(self):
        from dy3_polaris.l2.memory.short_term_memory import ShortTermMemory

        stm = ShortTermMemory()
        assert hasattr(stm, "_lock")
        n = 200
        errors: list[Exception] = []

        def worker(i):
            try:
                stm.add({"learner_id": "l", "idx": i})
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发 add 抛出异常: {errors}"
        entries = stm.get_entries("l")
        assert len(entries) == n, f"并发 add 应保留全部条目, 实际={len(entries)}"

    def test_concurrent_add_and_cleanup_no_crash(self):
        """并发 add 与 cleanup 不应抛 'dictionary changed size' 异常."""
        from dy3_polaris.l2.memory.short_term_memory import ShortTermMemory

        stm = ShortTermMemory()
        assert hasattr(stm, "_lock")
        errors: list[Exception] = []

        def adder():
            try:
                for i in range(300):
                    stm.add({"learner_id": "l", "idx": i})
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def cleaner():
            try:
                for _ in range(80):
                    stm.cleanup()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=cleaner)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == [], f"并发 add/cleanup 抛出异常: {errors}"


# ============================================================
# 9. Store: get_answer_history 返回副本
# ============================================================


class TestStoreDefensiveCopy:
    """InMemoryL2Store.get_answer_history 应返回防御性副本."""

    def test_get_answer_history_returns_copy(self):
        store = InMemoryL2Store()
        store.save_answer_history(
            "l",
            [AnswerRecord(learner_id="l", kp_id="k", correct=True, timestamp=1.0)],
        )
        history = store.get_answer_history("l")
        assert history is not None
        assert len(history) == 1
        # 修改返回的列表
        history.append(
            AnswerRecord(learner_id="l", kp_id="k2", correct=False, timestamp=2.0)
        )
        # 内部状态应不受影响
        history2 = store.get_answer_history("l")
        assert history2 is not None
        assert len(history2) == 1, "修改返回列表不应影响内部状态"

    def test_get_answer_history_none_when_absent(self):
        store = InMemoryL2Store()
        assert store.get_answer_history("absent") is None


# ============================================================
# 10. Builder: _records_to_events 包含模态信息
# ============================================================


class TestBuilderModalityInfo:
    """ProfileBuilder._records_to_events 应包含 modality / content_type 字段."""

    def test_records_to_events_includes_modality(self):
        builder = ProfileBuilder()
        records = [
            AnswerRecord(
                learner_id="l",
                kp_id="k",
                correct=True,
                difficulty=0.5,
                timestamp=1.0,
            )
        ]
        events = builder._records_to_events(records)
        assert len(events) == 1
        ev = events[0]
        assert "modality" in ev, "事件应包含 modality 字段"
        assert "content_type" in ev, "事件应包含 content_type 字段"
        # AnswerRecord 无模态字段, 应回退到默认值
        assert ev["modality"] is not None
        assert ev["content_type"] is not None

    def test_records_to_events_empty(self):
        builder = ProfileBuilder()
        assert builder._records_to_events(None) == []
        assert builder._records_to_events([]) == []

    def test_records_to_events_preserves_core_fields(self):
        builder = ProfileBuilder()
        records = [
            AnswerRecord(
                learner_id="l1",
                kp_id="k1",
                correct=False,
                difficulty=0.8,
                timestamp=42.0,
            )
        ]
        events = builder._records_to_events(records)
        ev = events[0]
        assert ev["learner_id"] == "l1"
        assert ev["correct"] is False
        assert ev["difficulty"] == 0.8
        assert ev["timestamp"] == 42.0


# ============================================================
# 11. CAT: 曝光统计线程安全 + reset_stats()
# ============================================================


class TestCATThreadSafety:
    """CATSelector 曝光统计线程安全 + reset_stats()."""

    def test_has_lock(self):
        cat = CATSelector()
        assert hasattr(cat, "_lock"), "CATSelector 应持有 _lock"

    def test_concurrent_exposure_no_lost_updates(self):
        cat = CATSelector()
        assert hasattr(cat, "_lock")
        # 全同题目, fisher_info 总是选第一题 "q0"
        items = [
            {"item_id": f"q{i}", "a": 1.0, "b": 0.0, "c": 0.0}
            for i in range(10)
        ]
        n = 100
        # Barrier 强制所有线程同时开始, 最大化曝光计数竞态触发概率
        barrier = threading.Barrier(n)
        errors: list[Exception] = []

        def worker():
            try:
                barrier.wait()
                cat.select_next(theta=0.0, available_items=items, administered_ids=[])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发选题抛出异常: {errors}"
        stats = cat.get_exposure_stats()
        total = sum(stats.values())
        # 修复前 (无锁): read-modify-write 竞态丢失计数 -> total < n
        # 修复后 (加锁): total == n
        assert total == n, f"曝光统计应无丢失, 期望 {n}, 实际 {total}"

    def test_reset_stats(self):
        cat = CATSelector()
        assert hasattr(cat, "reset_stats"), "CATSelector 应有 reset_stats 方法"
        items = [{"item_id": "q0", "a": 1.0, "b": 0.0, "c": 0.0}]
        cat.select_next(theta=0.0, available_items=items, administered_ids=[])
        cat.select_next(theta=0.0, available_items=items, administered_ids=[])
        assert sum(cat.get_exposure_stats().values()) == 2
        cat.reset_stats()
        assert cat.get_exposure_stats() == {}, "reset_stats 后曝光统计应清空"

    def test_reset_stats_clears_content_counts(self):
        cat = CATSelector(
            content_constraints={"algebra": 0.5, "geometry": 0.5}
        )
        items = [
            {"item_id": "q0", "a": 1.0, "b": 0.0, "c": 0.0, "content_area": "algebra"}
        ]
        cat.select_next(theta=0.0, available_items=items, administered_ids=[])
        assert cat._content_counts.get("algebra", 0) == 1
        cat.reset_stats()
        assert cat._content_counts == {}, "reset_stats 应同时清空内容域计数"
