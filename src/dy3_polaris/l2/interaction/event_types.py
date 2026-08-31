"""L2 交互事件类型定义.

融合世界先进方案:
- xAPI (Experience API): Actor-Verb-Object 标准化学习事件
- Caliper Analytics: IMS Global 学习事件互操作标准
- Khan Academy: 事件驱动 BKT 实时更新

事件类型:
- AnswerEvent : 答题事件 (触发 BKT + IRT 更新)
- QueryEvent  : 查询事件 (推断学习兴趣)
- BehaviorEvent: 行为事件 (学习/复习/跳过)

实现风格对齐 L2 models.py: ``from __future__ import annotations`` + ``@dataclass``,
每个事件提供 ``to_dict()`` / ``from_dict()`` 往返序列化方法; ``timestamp`` 默认 ``time.time()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from dy3_polaris.l2.models import AnswerRecord


# ============================================================
# 1. 常量定义
# ============================================================

# BehaviorEvent 合法动作集合 (study=学习 / review=复习 / skip=跳过)
VALID_ACTIONS: frozenset[str] = frozenset({"study", "review", "skip"})


# ============================================================
# 2. 答题事件 (AnswerEvent)
# ============================================================


@dataclass
class AnswerEvent:
    """答题事件 — 触发 BKT 知识追踪 + IRT 能力估计更新.

    对应 xAPI 的 "answered" 动词, Actor=learner, Object=question.

    Attributes:
        learner_id: 学习者 ID
        kp_id: 知识点 ID
        correct: 是否答对 (严格 bool)
        difficulty: 题目难度 [0.0, 1.0], 默认 0.5
        question_id: 题目 ID (可选, 用于去重)
        timestamp: 事件时间戳 (秒, float), 默认 time.time()
    """

    learner_id: str
    kp_id: str
    correct: bool
    difficulty: float = 0.5
    question_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_answer_record(self) -> AnswerRecord:
        """转换为 L2 AnswerRecord (BKT/IRT 更新管道的输入信号).

        Returns:
            与本事件字段一致的 AnswerRecord.
        """
        return AnswerRecord(
            learner_id=self.learner_id,
            kp_id=self.kp_id,
            correct=self.correct,
            timestamp=self.timestamp,
            difficulty=self.difficulty,
            question_id=self.question_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (含 event_type 标识)."""
        return {
            "event_type": "answer",
            "learner_id": self.learner_id,
            "kp_id": self.kp_id,
            "correct": self.correct,
            "difficulty": self.difficulty,
            "question_id": self.question_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnswerEvent:
        """从字典反序列化 (event_type 字段可选, 不参与构造)."""
        return cls(
            learner_id=d["learner_id"],
            kp_id=d["kp_id"],
            correct=d["correct"],
            difficulty=d.get("difficulty", 0.5),
            question_id=d.get("question_id"),
            timestamp=d.get("timestamp", time.time()),
        )


# ============================================================
# 3. 查询事件 (QueryEvent)
# ============================================================


@dataclass
class QueryEvent:
    """查询事件 — 学习者主动查询, 用于推断学习兴趣.

    对应 xAPI 的 "asked" / "searched" 动词.

    Attributes:
        learner_id: 学习者 ID
        query_text: 查询文本
        kp_ids: 关联知识点 ID 列表 (可空), 默认空列表
        timestamp: 事件时间戳 (秒, float), 默认 time.time()
    """

    learner_id: str
    query_text: str
    kp_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (kp_ids 浅拷贝避免共享引用)."""
        return {
            "event_type": "query",
            "learner_id": self.learner_id,
            "query_text": self.query_text,
            "kp_ids": list(self.kp_ids),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QueryEvent:
        """从字典反序列化."""
        return cls(
            learner_id=d["learner_id"],
            query_text=d["query_text"],
            kp_ids=list(d.get("kp_ids", [])),
            timestamp=d.get("timestamp", time.time()),
        )


# ============================================================
# 4. 行为事件 (BehaviorEvent)
# ============================================================


@dataclass
class BehaviorEvent:
    """行为事件 — 学习/复习/跳过等学习行为.

    对应 xAPI 的 "experienced" / "reviewed" / "skipped" 动词.

    Attributes:
        learner_id: 学习者 ID
        action: 行为类型 (study/review/skip)
        duration: 行为持续时长 (秒), 默认 0.0
        kp_id: 关联知识点 ID (可选)
        timestamp: 事件时间戳 (秒, float), 默认 time.time()
    """

    learner_id: str
    action: str
    duration: float = 0.0
    kp_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "event_type": "behavior",
            "learner_id": self.learner_id,
            "action": self.action,
            "duration": self.duration,
            "kp_id": self.kp_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BehaviorEvent:
        """从字典反序列化."""
        return cls(
            learner_id=d["learner_id"],
            action=d["action"],
            duration=d.get("duration", 0.0),
            kp_id=d.get("kp_id"),
            timestamp=d.get("timestamp", time.time()),
        )


# ============================================================
# __all__
# ============================================================

__all__ = [
    "VALID_ACTIONS",
    "AnswerEvent",
    "QueryEvent",
    "BehaviorEvent",
]
