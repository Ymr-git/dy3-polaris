"""用户理解数据模型 — 信号 + 画像.

信号 (UserSignal): 语料提取器产出的结构化事实片段, 供蒸馏器分类.
画像 (UnderstandingProfile): 长期用户理解画像, 挂载于 LearnerSnapshot.extras.user_profile.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SignalType(str, Enum):
    """信号类型."""
    INTEREST = "interest"            # 兴趣主题
    GOAL = "goal"                    # 学习目标
    PACE = "pace"                    # 学习节奏
    EXPRESSION = "expression"        # 表达偏好
    FRUSTRATION = "frustration"      # 挫败信号
    QUESTION_STYLE = "question_style"  # 提问风格
    VARK = "vark"                    # 行为 VARK
    PREFERENCE = "preference"        # 内容偏好


class SignalCategory(str, Enum):
    """蒸馏分类: 本次任务需求 vs 长期习惯."""
    TASK = "task"
    HABIT = "habit"


@dataclass
class UserSignal:
    """结构化信号片段."""
    learner_id: str
    signal_type: SignalType
    payload: dict[str, Any]
    source: str = "corpus"           # corpus / question / behavior / practice
    timestamp: float = field(default_factory=time.time)
    category: SignalCategory | None = None  # 由蒸馏器填写

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "signal_type": self.signal_type.value,
            "payload": dict(self.payload),
            "source": self.source,
            "timestamp": self.timestamp,
            "category": self.category.value if self.category else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UserSignal":
        return cls(
            learner_id=d["learner_id"],
            signal_type=SignalType(d["signal_type"]),
            payload=dict(d.get("payload", {})),
            source=d.get("source", "corpus"),
            timestamp=float(d.get("timestamp", time.time())),
            category=SignalCategory(d["category"]) if d.get("category") else None,
        )


@dataclass
class HabitRecord:
    """习惯信号记录 (用于频次统计)."""
    key: str
    value: str
    count: int = 1


@dataclass
class UnderstandingProfile:
    """用户理解画像 (长期).

    存储于 LearnerSnapshot.extras["user_profile"].
    """
    learner_id: str
    interests: list[dict[str, Any]] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)
    pace: str = "unknown"            # concentrated / fragmented / unknown
    expression: str = "unknown"      # concise / detailed / unknown
    question_style: str = "unknown"  # frequent / rare / unknown
    vark_behavior: dict[str, float] = field(default_factory=dict)
    frustration_level: float = 0.0   # [0,1]
    preferences: dict[str, Any] = field(default_factory=dict)
    # Optional self-declared cold-start facts.  These are priors for Diagnosis,
    # not mastery claims and never replace observed BKT/IRT evidence.
    declared_background: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0          # [0,1], 信号量归一化
    proactive_asked: int = 0         # 已主动提问次数 (频率控制)
    _habits: dict[str, list[str]] = field(default_factory=dict)  # key -> [value...]
    last_updated: float = field(default_factory=time.time)

    def add_habit(self, rec: HabitRecord) -> None:
        """记录一次习惯信号 (同值累计计数用列表长度表示)."""
        lst = self._habits.setdefault(rec.key, [])
        lst.append(rec.value)
        lst[:] = lst[-20:]  # 只保留最近 20 次

    def merge_from_habits(self) -> None:
        """按频次阈值 (>=3 次) 将 _habits 中的习惯写入画像字段."""
        now = time.time()
        for key, values in self._habits.items():
            if len(values) < 3:
                continue
            # 取最近值 (同值连续) — 简化: 众数
            most = max(set(values), key=values.count)
            if key == "pace":
                self.pace = most
            elif key == "expression":
                self.expression = most
            elif key == "question_style":
                self.question_style = most
        total_signals = sum(len(v) for v in self._habits.values()) + len(self.interests) + len(self.goals)
        self.confidence = min(1.0, total_signals / 8.0)
        self.last_updated = now

    def bump_proactive_asked(self) -> int:
        """自增主动提问计数, 返回新值 (频率控制用)."""
        self.proactive_asked += 1
        self.last_updated = time.time()
        return self.proactive_asked

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "interests": list(self.interests),
            "goals": list(self.goals),
            "pace": self.pace,
            "expression": self.expression,
            "question_style": self.question_style,
            "vark_behavior": dict(self.vark_behavior),
            "frustration_level": self.frustration_level,
            "preferences": dict(self.preferences),
            "declared_background": dict(self.declared_background),
            "confidence": self.confidence,
            "proactive_asked": self.proactive_asked,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UnderstandingProfile":
        return cls(
            learner_id=d["learner_id"],
            interests=list(d.get("interests", [])),
            goals=list(d.get("goals", [])),
            pace=d.get("pace", "unknown"),
            expression=d.get("expression", "unknown"),
            question_style=d.get("question_style", "unknown"),
            vark_behavior=dict(d.get("vark_behavior", {})),
            frustration_level=float(d.get("frustration_level", 0.0)),
            preferences=dict(d.get("preferences", {})),
            declared_background=dict(d.get("declared_background", {})),
            confidence=float(d.get("confidence", 0.0)),
            proactive_asked=int(d.get("proactive_asked", 0)),
            last_updated=float(d.get("last_updated", time.time())),
        )
