"""L2 事件采集器 — 收集、验证、去重.

设计依据:
- xAPI 事件验证规范: Actor (learner_id) 必填, Object 字段域校验
- Khan Academy 事件去重策略: 按 question_id 幂等去重, 保留最新作答

EventCollector 职责:
1. collect_answer(): 构造 AnswerEvent (timestamp 默认 time.time())
2. validate():       校验 learner_id 非空 / difficulty ∈ [0,1] / timestamp 单调递增
3. collect_batch():  批量收集, 按 question_id 去重 (保留最新 timestamp)

单调递增语义:
- validate 在校验通过后会记录该 learner 的最新时间戳 (_last_timestamp),
  后续同 learner 的时间戳回退将被判为非法.
- 单调性按 learner_id 隔离 (不同 learner 互不影响).
"""

from __future__ import annotations

import threading
import time
from typing import Any

from dy3_polaris.l2.interaction.event_types import AnswerEvent


# ============================================================
# EventCollector
# ============================================================


class EventCollector:
    """L2 事件采集器 — 收集 / 验证 / 去重.

    Attributes:
        _last_timestamp: learner_id -> 上次已校验通过的时间戳, 用于单调递增校验.
        _lock: 保护 _last_timestamp 读写 (线程安全的时间戳单调性校验).
    """

    def __init__(self) -> None:
        # learner_id -> 上次通过 validate 的时间戳
        self._last_timestamp: dict[str, float] = {}
        # 保护 _last_timestamp 的读-改-写, 避免并发校验竞态
        self._lock = threading.RLock()

    # --- 收集单个答题事件 ---

    def collect_answer(
        self,
        learner_id: str,
        kp_id: str,
        correct: bool,
        difficulty: float = 0.5,
        question_id: str | None = None,
        timestamp: float | None = None,
    ) -> AnswerEvent:
        """收集一次答题事件, 构造 AnswerEvent 并校验记录时间戳.

        Args:
            learner_id: 学习者 ID
            kp_id: 知识点 ID
            correct: 是否答对
            difficulty: 题目难度 [0.0, 1.0], 默认 0.5
            question_id: 题目 ID (可选)
            timestamp: 事件时间戳, 默认 time.time()

        Returns:
            构造好的 AnswerEvent (已通过 validate 并记录时间戳).
        """
        ev = AnswerEvent(
            learner_id=learner_id,
            kp_id=kp_id,
            correct=correct,
            difficulty=difficulty,
            question_id=question_id,
            timestamp=timestamp if timestamp is not None else time.time(),
        )
        # 校验并记录时间戳 (合法则更新 _last_timestamp)
        self.validate(ev)
        return ev

    # --- 验证 ---

    def validate(self, event: Any) -> bool:
        """验证事件合法性.

        校验规则:
        1. learner_id 非空 (None / 空串均非法)
        2. 若事件含 difficulty 字段, 则 difficulty ∈ [0.0, 1.0]
        3. timestamp 单调非递减 (>= 该 learner 上次通过校验的时间戳)

        副作用: 校验通过时, 更新 _last_timestamp[learner_id] = event.timestamp,
        以此驱动后续同 learner 事件的单调递增校验.

        Args:
            event: AnswerEvent / QueryEvent / BehaviorEvent 或任意带相应属性的对象.

        Returns:
            合法返回 True, 否则 False.
        """
        # 1. learner_id 非空
        learner_id = getattr(event, "learner_id", None)
        if not learner_id:  # None 或空串
            return False

        # 2. difficulty ∈ [0, 1] (仅当事件含 difficulty 字段时校验)
        difficulty = getattr(event, "difficulty", None)
        if difficulty is not None:
            try:
                if not (0.0 <= float(difficulty) <= 1.0):
                    return False
            except (TypeError, ValueError):
                return False

        # 3. timestamp 单调非递减 (按 learner_id 隔离)
        ts = getattr(event, "timestamp", None)
        if ts is not None:
            with self._lock:
                last = self._last_timestamp.get(learner_id)
                if last is not None and ts < last:
                    return False
                # 校验通过, 记录最新时间戳
                self._last_timestamp[learner_id] = ts

        return True

    # --- 重置 ---

    def reset(self) -> None:
        """清除所有时间戳记录.

        重置后, 单调递增校验回到初始状态: 任意时间戳的事件均可通过校验
        (无历史基准). 适用于跨会话/跨批次重新开始收集的场景.
        """
        with self._lock:
            self._last_timestamp.clear()

    # --- 批量收集 + 去重 ---

    def collect_batch(self, events: list[Any]) -> list[Any]:
        """批量收集事件, 按 question_id 去重 (保留最新 timestamp).

        去重策略 (Khan Academy 风格):
        - 同一 question_id 的多个事件, 仅保留 timestamp 最大的 (最新作答).
        - question_id 为 None 的事件不参与去重, 全部保留.
        - 结果保留各 question_id 首次出现的相对位置, 但值为最新事件.

        去重后, 按时间戳升序逐个调用 validate 以维护单调递增状态.

        Args:
            events: 事件列表 (AnswerEvent / QueryEvent / BehaviorEvent).

        Returns:
            去重后的事件列表 (保持原序, 同 question_id 替换为最新).
        """
        if not events:
            return []

        # 1. 找出每个 question_id 对应的最新事件 (最大 timestamp)
        latest_by_qid: dict[str, Any] = {}
        for ev in events:
            qid = getattr(ev, "question_id", None)
            if qid is None:
                continue
            prev = latest_by_qid.get(qid)
            if prev is None or ev.timestamp > prev.timestamp:
                latest_by_qid[qid] = ev

        # 2. 按原序构建结果: 同 question_id 仅保留首个位置, 值替换为最新
        seen: set[str] = set()
        result: list[Any] = []
        for ev in events:
            qid = getattr(ev, "question_id", None)
            if qid is None:
                # 无 question_id 的事件全部保留
                result.append(ev)
                continue
            if qid in seen:
                continue
            seen.add(qid)
            result.append(latest_by_qid[qid])

        # 3. 按时间戳升序 validate, 维护单调递增状态
        for ev in sorted(result, key=lambda e: e.timestamp):
            self.validate(ev)

        return result


# ============================================================
# __all__
# ============================================================

__all__ = [
    "EventCollector",
]
