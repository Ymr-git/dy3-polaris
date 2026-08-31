"""Outbox 总线包装 — L5 MessageBus 的可靠投递适配层.

机制:
- publish(msg) 先将事件 append 到 Outbox (与写操作同步入箱)
- 随后 deliver() 由注册的投递回调 (真实 MessageBus.publish) 投递
- 投递失败保留在 Outbox 待重试 (retry_failed), 形成发件箱闭环
"""
from __future__ import annotations

from typing import Any

from dy3_polaris.shared.outbox import Outbox


class OutboxWiredBus:
    """MessageBus 的 Outbox 包装 (委托全部其余属性/方法)."""

    def __init__(self, bus: Any, outbox: Outbox | None = None) -> None:
        self._bus = bus
        self._outbox = outbox or Outbox()
        self._outbox.set_deliverer(self._deliver)
        self._last_delivered: int = 0

    @property
    def outbox(self) -> Outbox:
        return self._outbox

    def _deliver(self, channel: str, payload: dict[str, Any]) -> None:
        from dy3_polaris.l5.communication import Message

        msg = Message(**payload)
        self._bus.publish(msg)

    def publish(self, msg: Any) -> None:
        """发布消息: 先入 Outbox, 再统一投递."""
        data = msg.model_dump() if hasattr(msg, "model_dump") else dict(vars(msg))
        self._outbox.append(data.get("channel", ""), data)
        self._last_delivered = self._outbox.deliver()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bus, name)


__all__ = ["OutboxWiredBus"]
