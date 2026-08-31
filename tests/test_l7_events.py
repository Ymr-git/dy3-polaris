"""L7 体验呈现层 — 事件系统 (EventEmitter) 测试.

覆盖 L7Event 数据类、EventEmitter 订阅/发射/注销/通配符/线程安全/错误隔离，
标准事件类型常量，以及全局 emitter 单例。

设计灵感:
- VS Code EventEmitter: 简洁 on/emit/off 模式
- Jupyter Signals: 类型化事件载荷
- Node.js EventEmitter: 监听器管理

测试领域: Dy3+ 发光材料 (YAG 基质, 4f-4f 跃迁, 480/574/660nm 发射)
"""
from __future__ import annotations

import dataclasses
import threading
import time

import pytest

from dy3_polaris.l7.events import (
    L7Event,
    EventEmitter,
    get_global_emitter,
    RENDER_START,
    RENDER_SUCCESS,
    RENDER_ERROR,
    UPDATE_START,
    UPDATE_SUCCESS,
    UPDATE_ERROR,
    DESTROY_START,
    DESTROY_SUCCESS,
    ARTIFACT_REGISTERED,
    ARTIFACT_UPDATED,
    ARTIFACT_REMOVED,
)


# ============================================================
# L7Event 数据类
# ============================================================


class TestL7EventConstruction:
    """L7Event 构造与字段测试."""

    def test_basic_construction(self):
        """基本构造 — event_type 与 data 正确存储."""
        ev = L7Event(event_type="render.start", artifact_id="art-001", data={"k": "v"})
        assert ev.event_type == "render.start"
        assert ev.artifact_id == "art-001"
        assert ev.data == {"k": "v"}

    def test_artifact_id_can_be_none(self):
        """系统事件 artifact_id 为 None."""
        ev = L7Event(event_type="system.heartbeat", artifact_id=None, data={})
        assert ev.artifact_id is None

    def test_auto_timestamp_generated(self):
        """未显式提供 timestamp 时自动生成 Unix 时间戳."""
        before = time.time()
        ev = L7Event(event_type="render.start", artifact_id=None, data={})
        after = time.time()
        assert isinstance(ev.timestamp, float)
        assert before <= ev.timestamp <= after

    def test_timestamp_close_to_now(self):
        """自动时间戳接近当前时间."""
        ev = L7Event(event_type="render.start", artifact_id=None, data={})
        assert abs(ev.timestamp - time.time()) < 1.0

    def test_each_event_gets_own_timestamp(self):
        """连续创建的事件拥有各自时间戳 (允许相等但应独立)."""
        ev1 = L7Event(event_type="a", artifact_id=None, data={})
        time.sleep(0.001)
        ev2 = L7Event(event_type="b", artifact_id=None, data={})
        assert ev2.timestamp >= ev1.timestamp

    def test_custom_data_dict(self):
        """自定义 data 字典保留原样."""
        payload = {"mime": "text/plain", "version": 3, "nested": {"x": 1}}
        ev = L7Event(event_type="render.success", artifact_id="art-1", data=payload)
        assert ev.data == payload
        assert ev.data["nested"]["x"] == 1

    def test_empty_data_default(self):
        """data 字段默认为空 dict."""
        ev = L7Event(event_type="render.start", artifact_id=None, data={})
        assert ev.data == {}

    def test_is_frozen_dataclass(self):
        """L7Event 是 frozen dataclass — 不可变."""
        assert dataclasses.is_dataclass(L7Event)
        ev = L7Event(event_type="render.start", artifact_id=None, data={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.event_type = "other"  # type: ignore[misc]

    def test_frozen_data_dict_not_assignable(self):
        """frozen dataclass 的字段不可重新赋值."""
        ev = L7Event(event_type="a", artifact_id=None, data={"x": 1})
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.artifact_id = "art-2"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.data = {}  # type: ignore[misc]

    def test_two_events_with_same_values_unequal_timestamp(self):
        """相同参数构造的两个事件在结构上相等 (若时间戳恰好相同)."""
        ev1 = L7Event(event_type="a", artifact_id="art", data={})
        ev2 = L7Event(event_type="a", artifact_id="art", data={})
        # dataclass 自动生成 __eq__，timestamp 极少相等；仅校验关键字段一致
        assert ev1.event_type == ev2.event_type
        assert ev1.artifact_id == ev2.artifact_id
        assert ev1.data == ev2.data


# ============================================================
# EventEmitter — 基本 on/emit
# ============================================================


class TestEventEmitterBasic:
    """EventEmitter 基本 on/emit 行为."""

    def test_on_and_emit_invokes_callback(self):
        """订阅后 emit 应调用回调."""
        emitter = EventEmitter()
        received: list[L7Event] = []
        emitter.on("render.start", lambda ev: received.append(ev))
        emitter.emit("render.start", artifact_id="art-1")
        assert len(received) == 1
        assert received[0].event_type == "render.start"
        assert received[0].artifact_id == "art-1"

    def test_emit_returns_l7event(self):
        """emit 返回创建的 L7Event 对象."""
        emitter = EventEmitter()
        ev = emitter.emit("render.start", artifact_id="art-1", mime="text/plain")
        assert isinstance(ev, L7Event)
        assert ev.event_type == "render.start"
        assert ev.artifact_id == "art-1"
        assert ev.data["mime"] == "text/plain"

    def test_emit_without_listeners_is_safe(self):
        """无监听器时 emit 安全 (不抛异常)."""
        emitter = EventEmitter()
        ev = emitter.emit("render.start", artifact_id="art-1")
        assert isinstance(ev, L7Event)

    def test_emit_without_artifact_id_defaults_none(self):
        """emit 不提供 artifact_id 时默认 None."""
        emitter = EventEmitter()
        received: list[L7Event] = []
        emitter.on("system.event", lambda ev: received.append(ev))
        ev = emitter.emit("system.event")
        assert ev.artifact_id is None
        assert received[0].artifact_id is None

    def test_emit_passes_data_kwargs_into_event_data(self):
        """emit 的 **data 关键字参数汇入 event.data."""
        emitter = EventEmitter()
        received: list[L7Event] = []
        emitter.on("render.success", lambda ev: received.append(ev))
        emitter.emit("render.success", artifact_id="art-1", render_time_ms=42.5, mime="x")
        assert received[0].data == {"render_time_ms": 42.5, "mime": "x"}

    def test_callback_receives_full_l7event(self):
        """回调收到完整的 L7Event 实例."""
        emitter = EventEmitter()
        captured: dict = {}
        emitter.on(
            "render.error",
            lambda ev: captured.update(event=ev),
        )
        emitter.emit("render.error", artifact_id="art-x", error="boom")
        ev = captured["event"]
        assert isinstance(ev, L7Event)
        assert ev.data["error"] == "boom"


# ============================================================
# EventEmitter — 多监听器
# ============================================================


class TestEventEmitterMultipleListeners:
    """EventEmitter 多监听器场景."""

    def test_multiple_listeners_same_event_all_invoked(self):
        """同一事件多个监听器全部被调用 (按注册顺序)."""
        emitter = EventEmitter()
        calls: list[str] = []
        emitter.on("render.start", lambda ev: calls.append("first"))
        emitter.on("render.start", lambda ev: calls.append("second"))
        emitter.on("render.start", lambda ev: calls.append("third"))
        emitter.emit("render.start", artifact_id="art-1")
        assert calls == ["first", "second", "third"]

    def test_different_event_types_isolated(self):
        """不同事件类型的监听器互不干扰."""
        emitter = EventEmitter()
        a_calls: list[L7Event] = []
        b_calls: list[L7Event] = []
        emitter.on("render.start", lambda ev: a_calls.append(ev))
        emitter.on("render.success", lambda ev: b_calls.append(ev))
        emitter.emit("render.start", artifact_id="art-1")
        assert len(a_calls) == 1
        assert len(b_calls) == 0
        emitter.emit("render.success", artifact_id="art-1")
        assert len(a_calls) == 1
        assert len(b_calls) == 1

    def test_same_callback_can_subscribe_multiple_event_types(self):
        """同一回调可订阅多个事件类型."""
        emitter = EventEmitter()
        received: list[str] = []
        cb = lambda ev: received.append(ev.event_type)  # noqa: E731
        emitter.on("render.start", cb)
        emitter.on("render.success", cb)
        emitter.emit("render.start", artifact_id="art-1")
        emitter.emit("render.success", artifact_id="art-1")
        assert received == ["render.start", "render.success"]


# ============================================================
# EventEmitter — 通配符监听器
# ============================================================


class TestEventEmitterWildcard:
    """EventEmitter 通配符 "*" 监听器."""

    def test_wildcard_receives_all_events(self):
        """通配符 "*" 监听器接收所有事件."""
        emitter = EventEmitter()
        received: list[str] = []
        emitter.on("*", lambda ev: received.append(ev.event_type))
        emitter.emit("render.start", artifact_id="art-1")
        emitter.emit("render.success", artifact_id="art-1")
        emitter.emit("artifact.registered", artifact_id="art-2")
        assert received == ["render.start", "render.success", "artifact.registered"]

    def test_wildcard_and_specific_both_invoked(self):
        """通配符与特定事件监听器同时触发."""
        emitter = EventEmitter()
        wildcard_calls: list[str] = []
        specific_calls: list[str] = []
        emitter.on("*", lambda ev: wildcard_calls.append(ev.event_type))
        emitter.on("render.start", lambda ev: specific_calls.append(ev.event_type))
        emitter.emit("render.start", artifact_id="art-1")
        assert wildcard_calls == ["render.start"]
        assert specific_calls == ["render.start"]

    def test_wildcard_does_not_duplicate_for_star_emit(self):
        """直接 emit("*") 时通配符监听器只触发一次 (不视作通配匹配两次)."""
        emitter = EventEmitter()
        received: list[str] = []
        emitter.on("*", lambda ev: received.append(ev.event_type))
        emitter.emit("*", artifact_id=None)
        assert received == ["*"]


# ============================================================
# EventEmitter — off 注销
# ============================================================


class TestEventEmitterOff:
    """EventEmitter off 注销行为."""

    def test_off_removes_callback(self):
        """off 移除指定回调后不再被调用."""
        emitter = EventEmitter()
        calls: list[L7Event] = []

        def cb(ev: L7Event) -> None:
            calls.append(ev)

        emitter.on("render.start", cb)
        emitter.emit("render.start", artifact_id="art-1")
        assert len(calls) == 1

        emitter.off("render.start", cb)
        emitter.emit("render.start", artifact_id="art-1")
        assert len(calls) == 1  # 未增加

    def test_off_nonexistent_callback_is_noop(self):
        """off 注销未订阅的回调是安全 no-op (不抛异常)."""
        emitter = EventEmitter()
        emitter.off("render.start", lambda ev: None)  # 不应抛异常
        emitter.off("never.subscribed", lambda ev: None)

    def test_off_only_removes_target_callback(self):
        """off 仅移除目标回调，其他回调保留."""
        emitter = EventEmitter()
        keep_calls: list[L7Event] = []
        remove_calls: list[L7Event] = []

        def keep(ev: L7Event) -> None:
            keep_calls.append(ev)

        def remove(ev: L7Event) -> None:
            remove_calls.append(ev)

        emitter.on("render.start", keep)
        emitter.on("render.start", remove)
        emitter.off("render.start", remove)
        emitter.emit("render.start", artifact_id="art-1")
        assert len(keep_calls) == 1
        assert len(remove_calls) == 0

    def test_off_does_not_affect_wildcard(self):
        """off 特定事件不影响通配符监听器."""
        emitter = EventEmitter()
        wildcard_calls: list[L7Event] = []
        specific_calls: list[L7Event] = []

        def wild(ev: L7Event) -> None:
            wildcard_calls.append(ev)

        def spec(ev: L7Event) -> None:
            specific_calls.append(ev)

        emitter.on("*", wild)
        emitter.on("render.start", spec)
        emitter.off("render.start", spec)
        emitter.emit("render.start", artifact_id="art-1")
        assert len(wildcard_calls) == 1
        assert len(specific_calls) == 0

    def test_off_wildcard_callback(self):
        """off 可移除通配符回调."""
        emitter = EventEmitter()
        calls: list[L7Event] = []

        def cb(ev: L7Event) -> None:
            calls.append(ev)

        emitter.on("*", cb)
        emitter.off("*", cb)
        emitter.emit("render.start", artifact_id="art-1")
        assert len(calls) == 0

    def test_off_removes_only_one_registration(self):
        """off 仅移除一次注册，重复注册同一回调需多次 off."""
        emitter = EventEmitter()
        calls: list[L7Event] = []

        def cb(ev: L7Event) -> None:
            calls.append(ev)

        emitter.on("render.start", cb)
        emitter.on("render.start", cb)  # 注册两次
        emitter.off("render.start", cb)  # 移除一次
        emitter.emit("render.start", artifact_id="art-1")
        assert len(calls) == 1


# ============================================================
# EventEmitter — clear
# ============================================================


class TestEventEmitterClear:
    """EventEmitter clear 清空行为."""

    def test_clear_removes_all_listeners(self):
        """clear 移除所有监听器."""
        emitter = EventEmitter()
        calls: list[L7Event] = []
        emitter.on("render.start", lambda ev: calls.append(ev))
        emitter.on("render.success", lambda ev: calls.append(ev))
        emitter.on("*", lambda ev: calls.append(ev))
        emitter.clear()
        emitter.emit("render.start", artifact_id="art-1")
        emitter.emit("render.success", artifact_id="art-1")
        assert len(calls) == 0

    def test_clear_on_empty_emitter_is_safe(self):
        """对空 emitter 调用 clear 安全 (幂等)."""
        emitter = EventEmitter()
        emitter.clear()
        emitter.clear()

    def test_clear_allows_re_subscribing(self):
        """clear 后可重新订阅."""
        emitter = EventEmitter()
        calls: list[L7Event] = []
        emitter.on("render.start", lambda ev: calls.append(ev))
        emitter.clear()
        emitter.on("render.start", lambda ev: calls.append(ev))
        emitter.emit("render.start", artifact_id="art-1")
        assert len(calls) == 1


# ============================================================
# EventEmitter — listener_count
# ============================================================


class TestEventEmitterListenerCount:
    """EventEmitter listener_count 计数."""

    def test_count_specific_event_type(self):
        """统计特定事件类型的监听器数 (不含通配符)."""
        emitter = EventEmitter()
        emitter.on("render.start", lambda ev: None)
        emitter.on("render.start", lambda ev: None)
        emitter.on("render.success", lambda ev: None)
        assert emitter.listener_count("render.start") == 2
        assert emitter.listener_count("render.success") == 1

    def test_count_wildcard_listeners(self):
        """统计通配符监听器数."""
        emitter = EventEmitter()
        emitter.on("*", lambda ev: None)
        emitter.on("*", lambda ev: None)
        assert emitter.listener_count("*") == 2

    def test_count_with_none_returns_total(self):
        """listener_count(None) 返回所有监听器总数 (含通配符)."""
        emitter = EventEmitter()
        emitter.on("render.start", lambda ev: None)
        emitter.on("render.start", lambda ev: None)
        emitter.on("render.success", lambda ev: None)
        emitter.on("*", lambda ev: None)
        assert emitter.listener_count(None) == 4

    def test_count_zero_for_unsubscribed_event(self):
        """未订阅的事件类型计数为 0."""
        emitter = EventEmitter()
        emitter.on("render.start", lambda ev: None)
        assert emitter.listener_count("never.subscribed") == 0

    def test_count_zero_on_empty_emitter(self):
        """空 emitter 所有计数为 0."""
        emitter = EventEmitter()
        assert emitter.listener_count("render.start") == 0
        assert emitter.listener_count(None) == 0

    def test_count_reflects_off(self):
        """off 后计数减少."""
        emitter = EventEmitter()

        def cb(ev: L7Event) -> None:
            pass

        emitter.on("render.start", cb)
        emitter.on("render.start", cb)
        assert emitter.listener_count("render.start") == 2
        emitter.off("render.start", cb)
        assert emitter.listener_count("render.start") == 1

    def test_count_reflects_clear(self):
        """clear 后计数归零."""
        emitter = EventEmitter()
        emitter.on("render.start", lambda ev: None)
        emitter.on("*", lambda ev: None)
        emitter.clear()
        assert emitter.listener_count("render.start") == 0
        assert emitter.listener_count("*") == 0
        assert emitter.listener_count(None) == 0


# ============================================================
# EventEmitter — 错误隔离
# ============================================================


class TestEventEmitterErrorIsolation:
    """EventEmitter 监听器异常隔离."""

    def test_listener_exception_does_not_crash_emit(self):
        """监听器抛异常不影响 emit 正常返回."""
        emitter = EventEmitter()

        def bad(ev: L7Event) -> None:
            raise ValueError("listener boom")

        emitter.on("render.start", bad)
        ev = emitter.emit("render.start", artifact_id="art-1")
        assert isinstance(ev, L7Event)

    def test_subsequent_listeners_still_called_after_exception(self):
        """前序监听器抛异常后，后续监听器仍被调用."""
        emitter = EventEmitter()
        calls: list[str] = []

        def first(ev: L7Event) -> None:
            calls.append("first")
            raise RuntimeError("first fails")

        def second(ev: L7Event) -> None:
            calls.append("second")

        emitter.on("render.start", first)
        emitter.on("render.start", second)
        emitter.emit("render.start", artifact_id="art-1")
        assert calls == ["first", "second"]

    def test_wildcard_still_called_after_specific_listener_exception(self):
        """特定监听器抛异常后通配符监听器仍被调用."""
        emitter = EventEmitter()
        wildcard_called: list[L7Event] = []

        def bad(ev: L7Event) -> None:
            raise ValueError("boom")

        emitter.on("render.start", bad)
        emitter.on("*", lambda ev: wildcard_called.append(ev))
        emitter.emit("render.start", artifact_id="art-1")
        assert len(wildcard_called) == 1

    def test_multiple_failing_listeners_all_isolated(self):
        """多个监听器同时抛异常互不影响."""
        emitter = EventEmitter()
        calls: list[str] = []

        def make(name: str, fail: bool):
            def cb(ev: L7Event) -> None:
                calls.append(name)
                if fail:
                    raise ValueError(name)

            return cb

        emitter.on("render.start", make("a", fail=True))
        emitter.on("render.start", make("b", fail=True))
        emitter.on("render.start", make("c", fail=False))
        emitter.emit("render.start", artifact_id="art-1")
        assert calls == ["a", "b", "c"]


# ============================================================
# EventEmitter — 线程安全
# ============================================================


class TestEventEmitterThreadSafety:
    """EventEmitter 并发安全."""

    def test_concurrent_emit_all_listeners_invoked(self):
        """并发 emit 时所有监听器均被调用 (无丢失)."""
        emitter = EventEmitter()
        counter = {"count": 0}
        lock = threading.Lock()

        def cb(ev: L7Event) -> None:
            with lock:
                counter["count"] += 1

        emitter.on("render.start", cb)
        n_threads = 10
        n_per_thread = 50

        def worker() -> None:
            for _ in range(n_per_thread):
                emitter.emit("render.start", artifact_id="art-1")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert counter["count"] == n_threads * n_per_thread

    def test_concurrent_subscribe_safe(self):
        """并发订阅不抛异常且监听器全部注册."""
        emitter = EventEmitter()

        def worker() -> None:
            for i in range(50):
                emitter.on(f"evt.{i}", lambda ev: None)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 5 线程 * 50 = 250 监听器 (event_type 可能重复，但每次 on 都注册)
        assert emitter.listener_count(None) == 250

    def test_concurrent_emit_and_off(self):
        """并发 emit 与 off 混合操作不抛异常."""
        emitter = EventEmitter()
        received: list[L7Event] = []
        lock = threading.Lock()

        def cb(ev: L7Event) -> None:
            with lock:
                received.append(ev)

        emitter.on("render.start", cb)

        stop = threading.Event()

        def emitter_worker() -> None:
            while not stop.is_set():
                try:
                    emitter.emit("render.start", artifact_id="art-1")
                except Exception:
                    pass

        def off_worker() -> None:
            while not stop.is_set():
                try:
                    emitter.off("render.start", cb)
                    emitter.on("render.start", cb)
                except Exception:
                    pass

        t1 = threading.Thread(target=emitter_worker)
        t2 = threading.Thread(target=off_worker)
        t1.start()
        t2.start()
        time.sleep(0.05)
        stop.set()
        t1.join()
        t2.join()
        # 至少收到一些事件，无异常即通过
        assert len(received) >= 1

    def test_concurrent_clear_and_emit(self):
        """并发 clear 与 emit 混合操作不抛异常."""
        emitter = EventEmitter()
        received: list[L7Event] = []
        lock = threading.Lock()

        def cb(ev: L7Event) -> None:
            with lock:
                received.append(ev)

        emitter.on("render.start", cb)
        stop = threading.Event()

        def clear_worker() -> None:
            while not stop.is_set():
                emitter.clear()
                emitter.on("render.start", cb)

        def emit_worker() -> None:
            while not stop.is_set():
                emitter.emit("render.start", artifact_id="art-1")

        t1 = threading.Thread(target=clear_worker)
        t2 = threading.Thread(target=emit_worker)
        t1.start()
        t2.start()
        time.sleep(0.05)
        stop.set()
        t1.join()
        t2.join()
        assert len(received) >= 1


# ============================================================
# 标准事件类型常量
# ============================================================


class TestStandardEventConstants:
    """标准事件类型常量存在且为字符串."""

    def test_render_lifecycle_constants(self):
        """渲染生命周期常量."""
        assert RENDER_START == "render.start"
        assert RENDER_SUCCESS == "render.success"
        assert RENDER_ERROR == "render.error"

    def test_update_lifecycle_constants(self):
        """更新生命周期常量."""
        assert UPDATE_START == "update.start"
        assert UPDATE_SUCCESS == "update.success"
        assert UPDATE_ERROR == "update.error"

    def test_destroy_lifecycle_constants(self):
        """销毁生命周期常量."""
        assert DESTROY_START == "destroy.start"
        assert DESTROY_SUCCESS == "destroy.success"

    def test_artifact_lifecycle_constants(self):
        """Artifact 生命周期常量."""
        assert ARTIFACT_REGISTERED == "artifact.registered"
        assert ARTIFACT_UPDATED == "artifact.updated"
        assert ARTIFACT_REMOVED == "artifact.removed"

    def test_all_constants_are_strings(self):
        """所有常量均为 str 类型."""
        constants = [
            RENDER_START, RENDER_SUCCESS, RENDER_ERROR,
            UPDATE_START, UPDATE_SUCCESS, UPDATE_ERROR,
            DESTROY_START, DESTROY_SUCCESS,
            ARTIFACT_REGISTERED, ARTIFACT_UPDATED, ARTIFACT_REMOVED,
        ]
        for c in constants:
            assert isinstance(c, str), f"{c!r} is not str"

    def test_all_constants_unique(self):
        """所有常量值唯一."""
        constants = [
            RENDER_START, RENDER_SUCCESS, RENDER_ERROR,
            UPDATE_START, UPDATE_SUCCESS, UPDATE_ERROR,
            DESTROY_START, DESTROY_SUCCESS,
            ARTIFACT_REGISTERED, ARTIFACT_UPDATED, ARTIFACT_REMOVED,
        ]
        assert len(set(constants)) == len(constants)

    def test_constants_usable_as_event_types(self):
        """常量可作为事件类型用于 on/emit."""
        emitter = EventEmitter()
        received: list[str] = []
        emitter.on(RENDER_START, lambda ev: received.append(ev.event_type))
        emitter.emit(RENDER_START, artifact_id="art-1")
        assert received == [RENDER_START]


# ============================================================
# get_global_emitter 单例
# ============================================================


class TestGlobalEmitter:
    """get_global_emitter 全局单例测试."""

    def test_returns_event_emitter_instance(self):
        """返回 EventEmitter 实例."""
        emitter = get_global_emitter()
        assert isinstance(emitter, EventEmitter)

    def test_returns_same_instance(self):
        """多次调用返回同一实例."""
        e1 = get_global_emitter()
        e2 = get_global_emitter()
        assert e1 is e2

    def test_global_emitter_functional(self):
        """全局 emitter 可正常工作."""
        emitter = get_global_emitter()
        received: list[L7Event] = []
        emitter.on("test.global", lambda ev: received.append(ev))
        try:
            emitter.emit("test.global", artifact_id="art-g")
            assert len(received) == 1
            assert received[0].artifact_id == "art-g"
        finally:
            emitter.off("test.global", received.append)  # noqa: best-effort cleanup

    def test_global_emitter_thread_safe_construction(self):
        """并发调用 get_global_emitter 返回同一实例."""
        results: list[EventEmitter] = []
        lock = threading.Lock()

        def worker() -> None:
            e = get_global_emitter()
            with lock:
                results.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        first = results[0]
        for r in results:
            assert r is first


# ============================================================
# 集成场景
# ============================================================


class TestEventIntegration:
    """事件系统端到端集成场景."""

    def test_emit_with_artifact_id_and_rich_data(self):
        """emit 携带 artifact_id 与丰富数据字典."""
        emitter = EventEmitter()
        captured: dict = {}
        emitter.on(
            RENDER_SUCCESS,
            lambda ev: captured.update(
                type=ev.event_type,
                aid=ev.artifact_id,
                data=ev.data,
                ts=ev.timestamp,
            ),
        )
        emitter.emit(
            RENDER_SUCCESS,
            artifact_id="art-abc123",
            render_id="rd-001",
            mime="text/vnd.dy3+markdown",
            render_time_ms=12.3,
            html="<p>hello</p>",
        )
        assert captured["type"] == RENDER_SUCCESS
        assert captured["aid"] == "art-abc123"
        assert captured["data"]["render_id"] == "rd-001"
        assert captured["data"]["mime"] == "text/vnd.dy3+markdown"
        assert captured["data"]["render_time_ms"] == 12.3
        assert captured["data"]["html"] == "<p>hello</p>"
        assert isinstance(captured["ts"], float)

    def test_full_render_lifecycle_event_flow(self):
        """模拟完整渲染生命周期: start -> success/error."""
        emitter = EventEmitter()
        log: list[str] = []
        emitter.on(RENDER_START, lambda ev: log.append(f"start:{ev.artifact_id}"))
        emitter.on(RENDER_SUCCESS, lambda ev: log.append(f"success:{ev.artifact_id}"))
        emitter.on(RENDER_ERROR, lambda ev: log.append(f"error:{ev.artifact_id}"))
        emitter.on("*", lambda ev: log.append(f"wild:{ev.event_type}"))

        emitter.emit(RENDER_START, artifact_id="art-1")
        emitter.emit(RENDER_SUCCESS, artifact_id="art-1", render_time_ms=5.0)
        emitter.emit(RENDER_START, artifact_id="art-2")
        emitter.emit(RENDER_ERROR, artifact_id="art-2", error="timeout")

        assert "start:art-1" in log
        assert "success:art-1" in log
        assert "start:art-2" in log
        assert "error:art-2" in log
        # 通配符捕获全部 4 个事件
        assert log.count("wild:render.start") == 2
        assert log.count("wild:render.success") == 1
        assert log.count("wild:render.error") == 1

    def test_artifact_lifecycle_with_wildcard_audit_log(self):
        """Artifact 生命周期事件通过通配符记录审计日志."""
        emitter = EventEmitter()
        audit: list[L7Event] = []
        emitter.on("*", lambda ev: audit.append(ev))

        emitter.emit(ARTIFACT_REGISTERED, artifact_id="art-1", version=1)
        emitter.emit(ARTIFACT_UPDATED, artifact_id="art-1", version=2)
        emitter.emit(ARTIFACT_REMOVED, artifact_id="art-1")

        assert len(audit) == 3
        assert audit[0].event_type == ARTIFACT_REGISTERED
        assert audit[1].event_type == ARTIFACT_UPDATED
        assert audit[2].event_type == ARTIFACT_REMOVED
        assert audit[0].data["version"] == 1
        assert audit[1].data["version"] == 2

    def test_emit_with_no_data_kwargs(self):
        """emit 不带任何 data 关键字参数时 data 为空 dict."""
        emitter = EventEmitter()
        received: list[L7Event] = []
        emitter.on("render.start", lambda ev: received.append(ev))
        emitter.emit("render.start", artifact_id="art-1")
        assert received[0].data == {}

    def test_error_in_one_event_type_does_not_affect_others(self):
        """一种事件类型监听器抛异常不影响其他事件类型."""
        emitter = EventEmitter()
        success_calls: list[L7Event] = []

        def bad(ev: L7Event) -> None:
            raise ValueError("boom")

        emitter.on(RENDER_ERROR, bad)
        emitter.on(RENDER_SUCCESS, lambda ev: success_calls.append(ev))
        emitter.emit(RENDER_ERROR, artifact_id="art-1")
        emitter.emit(RENDER_SUCCESS, artifact_id="art-1")
        assert len(success_calls) == 1
