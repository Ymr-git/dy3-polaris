"""学情广播协议 - 发布/订阅/事件路由.

支持:
- 层级主题通配匹配 (learner.* → learner.profile.updated)
- 单层通配 * 与多层通配 **
- 同步/异步投递模式
- 事件过滤 (subscriber 可提供 filter_fn)
- 事件历史日志 (可选, 用于审计/回放)
- 订阅者激活/停用
- 投递度量统计
- 线程安全操作

与 A2A 协议的关系:
- A2A 的 to_agent="*" 是点对点广播, 广播总线是发布订阅模型
- SessionManager 的 on()/_fire_hook() 是轻量事件钩子, 广播总线是其增强版
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Callable

from dy3_polaris.l6.core.exceptions import (
    BroadcastDeliveryError,
    BroadcastError,
    SubscriberNotFoundError,
)


# ============================================================
# 枚举
# ============================================================

class DeliveryMode(str, Enum):
    """投递模式."""
    SYNC = "sync"     # 同步: publish 时立即回调
    ASYNC = "async"   # 异步: 入队, 由消费者线程处理 (当前实现为标记, 实际仍同步)


# ============================================================
# 数据模型
# ============================================================

class BroadcastEvent:
    """广播事件.

    使用 __slots__ 优化内存占用。
    """

    __slots__ = ("event_id", "topic", "payload", "source", "timestamp", "metadata")

    def __init__(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        self.event_id = event_id or uuid.uuid4().hex[:16]
        self.topic = topic
        self.payload = payload or {}
        self.source = source
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "payload": self.payload,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class Subscription:
    """订阅记录."""

    __slots__ = (
        "subscriber_id",
        "topic_pattern",
        "callback",
        "filter_fn",
        "created_at",
        "delivery_mode",
        "active",
        "delivered_count",
        "failed_count",
    )

    def __init__(
        self,
        subscriber_id: str,
        topic_pattern: str,
        callback: Callable[[BroadcastEvent], None],
        filter_fn: Callable[[BroadcastEvent], bool] | None = None,
        delivery_mode: DeliveryMode = DeliveryMode.SYNC,
    ) -> None:
        self.subscriber_id = subscriber_id
        self.topic_pattern = topic_pattern
        self.callback = callback
        self.filter_fn = filter_fn
        self.created_at = time.time()
        self.delivery_mode = delivery_mode
        self.active = True
        self.delivered_count = 0
        self.failed_count = 0


# ============================================================
# 主题匹配
# ============================================================

def match_topic(pattern: str, topic: str) -> bool:
    """层级通配匹配.

    支持两种通配符:
    - ``*``: 匹配单层 (learner.* → learner.profile, 但不匹配 learner.profile.updated)
    - ``**``: 匹配多层 (learner.** → learner.profile.updated, learner.assessment.score)

    精确匹配: pattern == topic 时直接返回 True。

    Args:
        pattern: 订阅模式 (可含通配符)
        topic: 事件主题 (精确, 不含通配符)

    Returns:
        是否匹配
    """
    if pattern == topic:
        return True
    if "*" not in pattern:
        return False

    pattern_parts = pattern.split(".")
    topic_parts = topic.split(".")

    pi = 0
    ti = 0
    while pi < len(pattern_parts) and ti < len(topic_parts):
        pp = pattern_parts[pi]
        if pp == "**":
            # 多层通配: 匹配剩余所有层
            return True
        if pp == "*":
            # 单层通配: 跳过一层
            pi += 1
            ti += 1
            continue
        if pp != topic_parts[ti]:
            return False
        pi += 1
        ti += 1

    # 两者同时耗尽才匹配
    return pi == len(pattern_parts) and ti == len(topic_parts)


# ============================================================
# 度量收集
# ============================================================

class BroadcastMetrics:
    """线程安全广播度量收集器."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._publish_count = 0
        self._delivery_count = 0
        self._failure_count = 0
        self._filter_rejected = 0

    def record_publish(self) -> None:
        with self._lock:
            self._publish_count += 1

    def record_delivery(self, success: bool) -> None:
        with self._lock:
            if success:
                self._delivery_count += 1
            else:
                self._failure_count += 1

    def record_filter_reject(self) -> None:
        with self._lock:
            self._filter_rejected += 1

    def export(self) -> dict[str, Any]:
        with self._lock:
            return {
                "publish_count": self._publish_count,
                "delivery_count": self._delivery_count,
                "failure_count": self._failure_count,
                "filter_rejected": self._filter_rejected,
            }

    def reset(self) -> None:
        with self._lock:
            self._publish_count = 0
            self._delivery_count = 0
            self._failure_count = 0
            self._filter_rejected = 0


# ============================================================
# 广播总线
# ============================================================

class BroadcastBus:
    """学情广播总线.

    核心功能:
    - subscribe: 订阅主题 (支持 ``*`` 和 ``**`` 通配符)
    - unsubscribe: 取消订阅 (按 subscriber_id, 可选 topic_pattern)
    - publish: 发布事件到所有匹配订阅者
    - 事件过滤: subscriber 可提供 filter_fn
    - 事件历史日志: 可选, 用于审计/回放
    - 订阅者激活/停用: deactivate/activate
    - 投递度量统计

    线程安全: 所有公共方法均受 ``threading.RLock`` 保护。
    投递在锁外执行以防止回调中的死锁。

    Example::

        bus = BroadcastBus(event_log_enabled=True)
        bus.subscribe("agent-1", "learner.*", lambda e: print(e.topic))
        bus.publish("learner.profile.updated", {"student": "张三"})
    """

    def __init__(
        self,
        max_subscribers_per_topic: int = 100,
        event_log_enabled: bool = False,
        event_log_max_size: int = 1000,
    ) -> None:
        self._max_subscribers = max_subscribers_per_topic
        self._event_log_enabled = event_log_enabled
        self._event_log_max_size = event_log_max_size

        self._lock = threading.RLock()
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._event_log: deque[BroadcastEvent] = deque(
            maxlen=event_log_max_size if event_log_enabled else 0
        )
        self._metrics = BroadcastMetrics()

    # ---- 订阅管理 ----

    def subscribe(
        self,
        subscriber_id: str,
        topic_pattern: str,
        callback: Callable[[BroadcastEvent], None],
        filter_fn: Callable[[BroadcastEvent], bool] | None = None,
        delivery_mode: DeliveryMode = DeliveryMode.SYNC,
    ) -> Subscription:
        """订阅主题.

        Args:
            subscriber_id: 订阅者唯一标识
            topic_pattern: 主题模式 (支持 ``*`` 和 ``**`` 通配符)
            callback: 事件回调函数
            filter_fn: 可选过滤函数, 返回 True 才投递
            delivery_mode: 投递模式 (SYNC/ASYNC)

        Returns:
            创建的 Subscription 对象

        Raises:
            BroadcastError: 主题订阅者已满或重复订阅
        """
        with self._lock:
            subs = self._subscriptions[topic_pattern]
            if len(subs) >= self._max_subscribers:
                raise BroadcastError(
                    "BROADCAST_TOPIC_FULL",
                    f"Topic '{topic_pattern}' reached max subscribers ({self._max_subscribers})",
                )

            # 检查重复订阅
            for s in subs:
                if s.subscriber_id == subscriber_id:
                    raise BroadcastError(
                        "BROADCAST_DUPLICATE_SUBSCRIPTION",
                        f"Subscriber '{subscriber_id}' already subscribed to '{topic_pattern}'",
                    )

            sub = Subscription(
                subscriber_id=subscriber_id,
                topic_pattern=topic_pattern,
                callback=callback,
                filter_fn=filter_fn,
                delivery_mode=delivery_mode,
            )
            subs.append(sub)
            return sub

    def unsubscribe(self, subscriber_id: str, topic_pattern: str | None = None) -> int:
        """取消订阅.

        Args:
            subscriber_id: 订阅者 ID
            topic_pattern: 主题模式, None 表示取消该订阅者的所有订阅

        Returns:
            取消的订阅数

        Raises:
            SubscriberNotFoundError: 订阅者不存在
        """
        with self._lock:
            removed = 0

            if topic_pattern is not None:
                subs = self._subscriptions.get(topic_pattern, [])
                before = len(subs)
                self._subscriptions[topic_pattern] = [
                    s for s in subs if s.subscriber_id != subscriber_id
                ]
                removed = before - len(self._subscriptions[topic_pattern])
                if not self._subscriptions[topic_pattern]:
                    del self._subscriptions[topic_pattern]
            else:
                for pattern in list(self._subscriptions.keys()):
                    subs = self._subscriptions[pattern]
                    before = len(subs)
                    self._subscriptions[pattern] = [
                        s for s in subs if s.subscriber_id != subscriber_id
                    ]
                    removed += before - len(self._subscriptions[pattern])
                    if not self._subscriptions[pattern]:
                        del self._subscriptions[pattern]

            if removed == 0:
                raise SubscriberNotFoundError(subscriber_id)

            return removed

    def deactivate_subscriber(self, subscriber_id: str) -> int:
        """停用订阅者 (不删除, 标记为 inactive, 不再接收事件).

        Returns:
            停用的订阅数
        """
        with self._lock:
            count = 0
            for subs in self._subscriptions.values():
                for sub in subs:
                    if sub.subscriber_id == subscriber_id and sub.active:
                        sub.active = False
                        count += 1
            return count

    def activate_subscriber(self, subscriber_id: str) -> int:
        """激活已停用的订阅者.

        Returns:
            激活的订阅数
        """
        with self._lock:
            count = 0
            for subs in self._subscriptions.values():
                for sub in subs:
                    if sub.subscriber_id == subscriber_id and not sub.active:
                        sub.active = True
                        count += 1
            return count

    # ---- 发布 ----

    def publish(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> BroadcastEvent:
        """发布事件.

        将事件投递到所有匹配的订阅者。
        匹配规则: 订阅模式与事件主题进行通配匹配。
        投递在锁外执行, 避免回调死锁。

        Args:
            topic: 事件主题 (精确, 不含通配符)
            payload: 事件载荷
            source: 事件来源
            metadata: 附加元数据

        Returns:
            创建的 BroadcastEvent

        Raises:
            BroadcastDeliveryError: 当有投递失败时 (仍会尝试投递给其他订阅者)
        """
        event = BroadcastEvent(
            topic=topic,
            payload=payload,
            source=source,
            metadata=metadata,
        )

        # 在锁内收集匹配的订阅者
        with self._lock:
            self._metrics.record_publish()

            if self._event_log_enabled:
                self._event_log.append(event)

            matched: list[Subscription] = []
            for pattern, subs in self._subscriptions.items():
                if match_topic(pattern, topic):
                    for sub in subs:
                        if sub.active:
                            matched.append(sub)

        # 在锁外投递, 防止回调中的死锁
        failed_count = 0
        for sub in matched:
            # 应用过滤器
            if sub.filter_fn is not None:
                try:
                    if not sub.filter_fn(event):
                        self._metrics.record_filter_reject()
                        continue
                except Exception:
                    sub.failed_count += 1
                    failed_count += 1
                    self._metrics.record_delivery(False)
                    continue

            # 投递
            try:
                sub.callback(event)
                sub.delivered_count += 1
                self._metrics.record_delivery(True)
            except Exception:
                sub.failed_count += 1
                failed_count += 1
                self._metrics.record_delivery(False)

        if failed_count > 0:
            raise BroadcastDeliveryError(topic, failed_count)

        return event

    # ---- 查询 ----

    def get_subscribers(self, topic: str) -> list[str]:
        """获取匹配某主题的所有活跃订阅者 ID."""
        with self._lock:
            result: list[str] = []
            seen: set[str] = set()
            for pattern, subs in self._subscriptions.items():
                if match_topic(pattern, topic):
                    for sub in subs:
                        if sub.active and sub.subscriber_id not in seen:
                            result.append(sub.subscriber_id)
                            seen.add(sub.subscriber_id)
            return result

    def list_topics(self) -> list[str]:
        """列出所有已注册的主题模式."""
        with self._lock:
            return list(self._subscriptions.keys())

    def get_subscriptions(self, subscriber_id: str) -> list[Subscription]:
        """获取某订阅者的所有订阅."""
        with self._lock:
            result: list[Subscription] = []
            for subs in self._subscriptions.values():
                for sub in subs:
                    if sub.subscriber_id == subscriber_id:
                        result.append(sub)
            return result

    # ---- 事件日志 ----

    def get_event_log(self) -> list[BroadcastEvent]:
        """获取事件历史日志副本 (需启用 event_log)."""
        with self._lock:
            return list(self._event_log)

    def clear_event_log(self) -> None:
        """清空事件日志."""
        with self._lock:
            self._event_log.clear()

    # ---- 度量 ----

    def get_metrics(self) -> dict[str, Any]:
        """获取度量统计."""
        with self._lock:
            metrics = self._metrics.export()
            metrics["topics"] = list(self._subscriptions.keys())
            metrics["active_subscriptions"] = sum(
                1
                for subs in self._subscriptions.values()
                for s in subs
                if s.active
            )
            metrics["total_subscriptions"] = self._total_subscribers()
            return metrics

    # ---- 重置 ----

    def reset(self) -> None:
        """重置总线 (清除所有订阅、日志和度量)."""
        with self._lock:
            self._subscriptions.clear()
            self._event_log.clear()
            self._metrics.reset()

    # ---- 内部 ----

    def _total_subscribers(self) -> int:
        """当前总订阅数 (含 inactive)."""
        return sum(len(subs) for subs in self._subscriptions.values())
