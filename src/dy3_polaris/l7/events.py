"""L7 体验呈现层 — 事件系统 (EventEmitter).

提供轻量级发布/订阅机制，允许 L7 层各组件订阅渲染、更新、销毁及
Artifact 生命周期事件。事件系统解耦了事件生产者与消费者，便于
实现审计日志、性能监控、调试追踪等横切关注点。

设计灵感:
- VS Code EventEmitter: 简洁的 on/emit/off 模式
- Jupyter Signals: 类型化事件载荷 (L7Event dataclass)
- Node.js EventEmitter: 监听器管理 (on/off/clear/listener_count)

核心特性:
    - 不可变事件对象 (frozen dataclass) — L7Event 创建后不可修改
    - 通配符订阅 — ``on("*", cb)`` 监听所有事件
    - 错误隔离 — 单个监听器抛异常不影响其他监听器与发射器
    - 线程安全 — 所有操作通过 ``threading.Lock`` 保护
    - 全局单例 — ``get_global_emitter()`` 提供模块级共享实例

架构:
    L7Event (frozen dataclass)
    ├── event_type: str           # e.g. "render.start"
    ├── artifact_id: str | None   # 关联 Artifact ID (系统事件为 None)
    ├── timestamp: float          # Unix 时间戳 (自动生成)
    └── data: dict[str, Any]      # 事件载荷

    EventEmitter
    ├── _listeners: dict[str, list[Callable]]  # event_type -> 回调列表
    └── _lock: threading.Lock                   # 并发控制
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ============================================================
# 标准事件类型常量
# ============================================================

#: 渲染生命周期 — 渲染开始
RENDER_START: str = "render.start"
#: 渲染生命周期 — 渲染成功
RENDER_SUCCESS: str = "render.success"
#: 渲染生命周期 — 渲染失败
RENDER_ERROR: str = "render.error"

#: 更新生命周期 — 增量更新开始
UPDATE_START: str = "update.start"
#: 更新生命周期 — 增量更新成功
UPDATE_SUCCESS: str = "update.success"
#: 更新生命周期 — 增量更新失败
UPDATE_ERROR: str = "update.error"

#: 销毁生命周期 — 销毁开始
DESTROY_START: str = "destroy.start"
#: 销毁生命周期 — 销毁完成
DESTROY_SUCCESS: str = "destroy.success"

#: Artifact 生命周期 — Artifact 注册
ARTIFACT_REGISTERED: str = "artifact.registered"
#: Artifact 生命周期 — Artifact 更新
ARTIFACT_UPDATED: str = "artifact.updated"
#: Artifact 生命周期 — Artifact 移除
ARTIFACT_REMOVED: str = "artifact.removed"
#: Artifact 生命周期 — Artifact 审核 (CC1)
ARTIFACT_REVIEWED: str = "artifact.reviewed"
#: Artifact 生命周期 — Artifact 归档
ARTIFACT_ARCHIVED: str = "artifact.archived"
#: Artifact 生命周期 — Artifact 恢复 (反归档)
ARTIFACT_RESTORED: str = "artifact.restored"
#: Artifact 生命周期 — Artifact 分支合并
ARTIFACT_MERGED: str = "artifact.merged"


# ============================================================
# L7Event 不可变事件对象
# ============================================================


@dataclass(frozen=True)
class L7Event:
    """L7 事件 — 不可变的事件描述对象 (frozen dataclass).

    由 ``EventEmitter.emit()`` 自动创建并分发给监听器。创建后不可修改，
    保证事件在分发链路中的一致性。

    Attributes:
        event_type: 事件类型字符串，如 ``"render.start"``、``"artifact.registered"``。
        artifact_id: 关联的 Artifact ID；系统级事件为 None。
        timestamp: Unix 时间戳，默认通过 ``time.time()`` 自动生成。
        data: 事件载荷字典，携带事件相关的结构化数据。
    """

    event_type: str
    artifact_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


# ============================================================
# EventEmitter 事件发射器
# ============================================================


# 监听器回调类型 — 接收 L7Event 参数，无返回值
EventListener = Callable[[L7Event], None]

# 模块级日志器 — 用于记录监听器异常 (不中断发射流程)
_logger = logging.getLogger("dy3_polaris.l7.events")

# 通配符事件类型 — 订阅所有事件
_WILDCARD: str = "*"


class EventEmitter:
    """事件发射器 — 发布/订阅模式的核心组件.

    允许组件通过 ``on()`` 订阅特定事件类型 (或通配符 ``"*"`` 订阅全部事件)，
    通过 ``emit()`` 发射事件并通知所有匹配的监听器。

    线程安全说明:
        所有公开方法均通过 ``threading.Lock`` 保护。``emit()`` 在锁内仅完成
        监听器快照拷贝，回调执行在锁外进行，避免监听器回调内部再次操作
        同一发射器时产生死锁。

    使用示例::

        emitter = EventEmitter()

        # 订阅特定事件
        def on_render_start(event: L7Event) -> None:
            print(f"Render started: {event.artifact_id}")

        emitter.on(RENDER_START, on_render_start)

        # 订阅所有事件 (审计日志)
        emitter.on("*", lambda ev: audit_log.append(ev))

        # 发射事件
        emitter.emit(RENDER_START, artifact_id="art-001", mime="text/plain")

        # 注销
        emitter.off(RENDER_START, on_render_start)
    """

    def __init__(self) -> None:
        """初始化事件发射器 — 创建空的监听器注册表与互斥锁."""
        # event_type -> 监听器回调列表 (允许重复注册同一回调)
        self._listeners: dict[str, list[EventListener]] = {}
        # 并发控制
        self._lock: threading.Lock = threading.Lock()

    # ============================================================
    # 订阅
    # ============================================================

    def on(self, event_type: str, callback: EventListener) -> None:
        """订阅指定事件类型.

        Args:
            event_type: 事件类型字符串。传入 ``"*"`` 可订阅所有事件
                (通配符监听器在任意事件发射时均会被通知)。
            callback: 事件回调函数，签名为 ``callback(event: L7Event) -> None``。

        Note:
            - 同一回调可对同一事件类型多次注册，每次注册都会在发射时各触发一次。
            - 同一回调可订阅多个不同的事件类型。
        """
        with self._lock:
            listeners = self._listeners.get(event_type)
            if listeners is None:
                listeners = []
                self._listeners[event_type] = listeners
            listeners.append(callback)

    # ============================================================
    # 注销
    # ============================================================

    def off(self, event_type: str, callback: EventListener) -> None:
        """注销指定事件类型的回调.

        Args:
            event_type: 事件类型字符串。
            callback: 要移除的回调函数。

        Note:
            - 仅移除首次匹配的一处注册 (若同一回调注册多次，需多次调用)。
            - 安全操作: 若回调未订阅或事件类型不存在，静默无操作 (no-op)，
              不抛出异常。
        """
        with self._lock:
            listeners = self._listeners.get(event_type)
            if not listeners:
                return
            try:
                listeners.remove(callback)
            except ValueError:
                # 回调不在列表中 — 安全 no-op
                pass

    # ============================================================
    # 发射
    # ============================================================

    def emit(
        self,
        event_type: str,
        artifact_id: str | None = None,
        **data: Any,
    ) -> L7Event:
        """创建并发射事件，通知所有匹配的监听器.

        创建 ``L7Event`` 实例，然后通知该事件类型的所有监听器以及通配符
        ``"*"`` 监听器。监听器按注册顺序依次调用。

        Args:
            event_type: 事件类型字符串。
            artifact_id: 关联的 Artifact ID，默认 None (系统事件)。
            **data: 事件载荷键值对，汇入 ``L7Event.data`` 字典。

        Returns:
            创建的 ``L7Event`` 实例。

        Note:
            - **错误隔离**: 若某监听器回调抛出异常，异常会被捕获并记录到日志，
              不影响后续监听器的调用，也不影响 ``emit`` 正常返回。
            - **避免重复**: 当 ``event_type`` 为 ``"*"`` 时，通配符监听器仅触发
              一次 (不会因既是特定类型又是通配符而重复)。
            - **无监听器安全**: 若没有任何匹配的监听器，事件仍会被创建并返回。
        """
        # 创建不可变事件对象 (拷贝 data 防止外部修改影响事件)
        event = L7Event(
            event_type=event_type,
            artifact_id=artifact_id,
            data=dict(data),
        )

        # 在锁内快照监听器列表，在锁外执行回调 (避免死锁)
        with self._lock:
            # 特定事件类型的监听器
            specific = list(self._listeners.get(event_type, []))
            # 通配符监听器
            wildcard = list(self._listeners.get(_WILDCARD, []))

        # 当 event_type 本身为 "*" 时，specific 即通配符监听器，不再追加 wildcard
        if event_type == _WILDCARD:
            listeners_to_call = specific
        else:
            listeners_to_call = specific + wildcard

        # 依次调用监听器 (锁外执行，允许回调内操作同一发射器)
        for callback in listeners_to_call:
            try:
                callback(event)
            except Exception:
                _logger.exception(
                    "EventEmitter listener for event_type=%r raised an exception",
                    event_type,
                )

        return event

    # ============================================================
    # 维护
    # ============================================================

    def clear(self) -> None:
        """移除所有监听器 (幂等).

        清空全部事件类型的监听器注册表，包括通配符监听器。
        对空发射器调用亦安全。
        """
        with self._lock:
            self._listeners.clear()

    # ============================================================
    # 查询
    # ============================================================

    def listener_count(self, event_type: str | None = None) -> int:
        """统计监听器数量.

        Args:
            event_type: 指定事件类型。传入 None 则统计所有事件类型的监听器
                总数 (含通配符)。传入 ``"*"`` 则统计通配符监听器数。

        Returns:
            监听器数量。未订阅的事件类型返回 0。
        """
        with self._lock:
            if event_type is None:
                return sum(len(lst) for lst in self._listeners.values())
            return len(self._listeners.get(event_type, []))


# ============================================================
# 全局事件发射器单例
# ============================================================

_global_emitter: EventEmitter | None = None
_global_lock = threading.Lock()


def get_global_emitter() -> EventEmitter:
    """获取全局 EventEmitter 单例.

    使用双重检查锁定 (double-checked locking) 保证线程安全的惰性初始化。
    首次调用时创建实例，后续调用返回同一实例。

    Returns:
        全局共享的 ``EventEmitter`` 实例。
    """
    global _global_emitter
    if _global_emitter is None:
        with _global_lock:
            if _global_emitter is None:
                _global_emitter = EventEmitter()
    return _global_emitter


def reset_global_emitter() -> None:
    """重置全局事件发射器单例 (仅用于测试).

    将全局单例置为 None，下次调用 ``get_global_emitter()`` 将创建新实例。
    """
    global _global_emitter
    with _global_lock:
        _global_emitter = None
