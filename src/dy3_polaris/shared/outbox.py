"""Outbox — 跨层事件发件箱 (投递可靠性).

设计:
- 写侧: 业务操作将事件 append 到 Outbox (同进程, 与写操作同步落箱)
- 投递侧: deliver() 将待投递事件交给目标总线 (L5 MessageBus / L6 Broadcast),
  成功标记 delivered; 失败保留待重试
- 纯内存实现 (项目为单进程内存架构), 提供计数/清理/重试 API
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable


class OutboxRecord:
    """发件箱记录."""

    __slots__ = ("record_id", "channel", "payload", "created_at", "delivered_at", "error")

    def __init__(self, channel: str, payload: dict[str, Any]) -> None:
        self.record_id = f"ob-{uuid.uuid4().hex[:12]}"
        self.channel = channel
        self.payload = payload
        self.created_at = time.time()
        self.delivered_at: float | None = None
        self.error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "channel": self.channel,
            "payload": self.payload,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "error": self.error,
        }


class Outbox:
    """进程内发件箱: append → deliver (投递回调) → 标记完成."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: list[OutboxRecord] = []
        self._deliverer: Callable[[str, dict[str, Any]], None] | None = None

    def set_deliverer(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        """注册投递回调 (channel, payload) -> None (如 MessageBus.publish)."""
        self._deliverer = fn

    def append(self, channel: str, payload: dict[str, Any]) -> OutboxRecord:
        """写侧入箱 (不自动投递, 由 deliver 统一处理)."""
        record = OutboxRecord(channel, payload)
        with self._lock:
            self._records.append(record)
        return record

    def deliver(self, max_records: int = 100) -> int:
        """投递所有未完成记录 (同步); 返回成功投递数."""
        with self._lock:
            pending = [
                r for r in self._records
                if r.delivered_at is None and not r.error
            ][:max_records]
        delivered = 0
        for record in pending:
            if self._deliverer is None:
                record.error = "no deliverer"
                continue
            try:
                self._deliverer(record.channel, record.payload)
                record.delivered_at = time.time()
                delivered += 1
            except Exception as exc:  # noqa: BLE001
                record.error = str(exc)
        return delivered

    def pending_count(self) -> int:
        """待投递记录数."""
        with self._lock:
            return sum(
                1 for r in self._records
                if r.delivered_at is None and not r.error
            )

    def total_count(self) -> int:
        with self._lock:
            return len(self._records)

    def failed_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._records if r.error)

    def retry_failed(self) -> int:
        """清空失败标记并重投; 返回成功数."""
        with self._lock:
            for r in self._records:
                if r.error:
                    r.error = None
        return self.deliver()

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._records]

    def clear(self) -> int:
        """清空全部记录 (含已完成); 返回清理数."""
        with self._lock:
            n = len(self._records)
            self._records.clear()
        return n

    def prune(self, max_age_seconds: float = 24 * 3600.0) -> int:
        """清理超龄已完成记录."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            before = len(self._records)
            self._records = [
                r for r in self._records
                if not (r.delivered_at is not None and r.created_at < cutoff)
            ]
            return before - len(self._records)


__all__ = ["Outbox", "OutboxRecord"]
