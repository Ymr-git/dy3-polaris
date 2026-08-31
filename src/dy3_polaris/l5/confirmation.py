"""不确定结果确认 — 决策 Agent 向提问者发起确认的挂起存储."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


def extract_answer(payload: dict[str, Any] | None) -> str:
    """从决策响应载荷中提取可展示的主答案."""
    payload = payload or {}
    answer = payload.get("answer")
    if answer not in (None, ""):
        return str(answer)
    answers = payload.get("answers") or []
    if answers:
        first = answers[0]
        if isinstance(first, dict):
            return str(first.get("text") or first.get("content") or str(first))
        return str(first)
    summary = payload.get("summary")
    return str(summary) if summary else ""


@dataclass
class PendingConfirmation:
    """一次等待提问者确认的决策结果."""

    plan_id: str
    task_id: str
    task_state: str
    task_events: list[dict[str, Any]]
    query: str
    action_type: str
    confidence: float
    answer: str
    learner_id: str = ""
    pipeline: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)
    session: dict[str, Any] | None = None
    learner: dict[str, Any] | None = None
    safety_level: str = "safe"
    confirmation_questions: list[str] = field(default_factory=list)
    reason: str = ""
    created_at: float = field(default_factory=time.time)


class ConfirmationStore:
    """带 TTL 的挂起确认存储."""

    def __init__(self, ttl_s: float = 900.0) -> None:
        self._ttl_s = ttl_s
        self._items: dict[str, PendingConfirmation] = {}
        self._lock = threading.RLock()

    def put(self, confirmation: PendingConfirmation) -> None:
        """写入待确认记录."""
        with self._lock:
            self.prune()
            self._items[confirmation.plan_id] = confirmation

    def get(self, plan_id: str) -> PendingConfirmation | None:
        """读取待确认记录（过期自动移除）."""
        with self._lock:
            self.prune()
            item = self._items.get(plan_id)
            if item is not None and time.time() - item.created_at > self._ttl_s:
                self._items.pop(plan_id, None)
                return None
            return item

    def pop(self, plan_id: str) -> PendingConfirmation | None:
        """取走并移除待确认记录."""
        with self._lock:
            item = self.get(plan_id)
            if item is not None:
                self._items.pop(plan_id, None)
            return item

    def prune(self) -> None:
        """清理过期记录."""
        now = time.time()
        expired = [
            pid
            for pid, item in self._items.items()
            if now - item.created_at > self._ttl_s
        ]
        for pid in expired:
            self._items.pop(pid, None)

    def list_active(self) -> list[PendingConfirmation]:
        """列出当前未过期记录."""
        with self._lock:
            self.prune()
            return list(self._items.values())


__all__ = [
    "ConfirmationStore",
    "PendingConfirmation",
    "extract_answer",
]
