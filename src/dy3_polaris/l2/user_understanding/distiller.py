"""记忆蒸馏器 — 短期信号 → 任务/习惯分类.

规则:
- 同类信号出现 >= habit_threshold 次 → habit (长期习惯)
- GOAL 类型信号 (长目标) 恒为 habit
- 其余 → task (本次任务需求, 保留在短期记忆)
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from dy3_polaris.l2.user_understanding.models import (
    HabitRecord,
    SignalCategory,
    SignalType,
    UserSignal,
)

_ALWAYS_HABIT_TYPES = {SignalType.GOAL}


class MemoryDistiller:
    """短期信号蒸馏分类器 (无状态)."""

    def __init__(self, habit_threshold: int = 3) -> None:
        self._threshold = habit_threshold

    def distill(self, signals: list[UserSignal]) -> tuple[list[UserSignal], list[UserSignal]]:
        """分类信号.

        Returns:
            (task_signals, habit_signals)
        """
        task: list[UserSignal] = []
        habit: list[UserSignal] = []
        # 按 (signal_type, 关键 payload key) 统计频次
        key_counts: Counter[tuple[str, str]] = Counter()
        for s in signals:
            pk = self._payload_key(s)
            key_counts[(s.signal_type.value, pk)] += 1
        for s in signals:
            pk = self._payload_key(s)
            n = key_counts[(s.signal_type.value, pk)]
            if s.signal_type in _ALWAYS_HABIT_TYPES or n >= self._threshold:
                s.category = SignalCategory.HABIT
                habit.append(s)
            else:
                s.category = SignalCategory.TASK
                task.append(s)
        return task, habit

    def to_habit_records(self, habit_signals: list[UserSignal]) -> list[HabitRecord]:
        """将习惯信号转为 HabitRecord (同 key 同 value 合并计数)."""
        merged: dict[tuple[str, str], int] = {}
        for s in habit_signals:
            key = self._record_key(s)
            if key is None:
                continue
            value = self._record_value(s)
            if value is None:
                continue
            k = (key, value)
            merged[k] = merged.get(k, 0) + 1
        return [HabitRecord(key=k, value=v, count=c) for (k, v), c in merged.items()]

    # ---- 内部辅助 ----

    @staticmethod
    def _payload_key(s: UserSignal) -> str:
        if s.signal_type == SignalType.INTEREST:
            return str(s.payload.get("topic", ""))
        if s.signal_type == SignalType.VARK:
            return "vark"
        if s.signal_type == SignalType.FRUSTRATION:
            return "frustration"
        return str(s.payload.get("pace") or s.payload.get("preference") or "")

    @staticmethod
    def _record_key(s: UserSignal) -> str | None:
        return {
            SignalType.PACE: "pace",
            SignalType.EXPRESSION: "expression",
            SignalType.QUESTION_STYLE: "question_style",
            SignalType.FRUSTRATION: "frustration",
        }.get(s.signal_type)

    @staticmethod
    def _record_value(s: UserSignal) -> str | None:
        if s.signal_type == SignalType.FRUSTRATION:
            return "high" if s.payload.get("level", 0) >= 0.5 else "low"
        return str(s.payload.get("pace") or s.payload.get("preference") or "")
