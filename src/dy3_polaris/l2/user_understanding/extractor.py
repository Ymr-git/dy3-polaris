"""语料提取器 — 从对话语料提取结构化用户信号.

提取维度:
- 兴趣主题: 知识库主题词命中 + 频次
- 挫败信号: 否定/情绪词频次归一化
- 目标信号: 目标类关键词 ("考研/竞赛/就业/想学好")
- 节奏信号: 会话内时间间隔分布 (中位数 < 阈值 → fragmented)
- 表达偏好: 消息长度分布 (均值 < 阈值 → concise)
- 提问风格: 消息计数密度
- 行为 VARK: 内容类型关键词 (video/image/audio/text/simulation)
"""
from __future__ import annotations

import time
from typing import Any

from dy3_polaris.l2.user_understanding.models import SignalType, UserSignal
from dy3_polaris.l2.user_understanding.privacy import PrivacyGate

_TOPIC_SCORE = 1.0
_HABIT_THRESHOLD_S = 3600.0  # 会话间隔 1h 内 → fragmented
_CONCISE_LEN = 30            # 消息均长 < 30 字 → concise

_FRUSTRATION_WORDS = ["太难", "看不懂", "不会", "做错", "沮丧", "放弃", "崩溃", "不懂", "卡住", "没思路"]
_GOAL_WORDS = [
    ("考研", "long_term"), ("保研", "long_term"), ("读研", "long_term"),
    ("竞赛", "short_term"), ("考试", "short_term"), ("期末", "short_term"),
    ("拿高分", "short_term"), ("提升成绩", "short_term"), ("工作", "long_term"),
    ("就业", "long_term"), ("兴趣", "long_term"), ("学好", "long_term"),
]
_NEGATION_WORDS = ["不用", "不需要", "别", "不要", "无需"]

_VARK_KEYWORDS: dict[str, str] = {
    "视频": "V", "动画": "V", "图": "V", "图表": "V", "看": "V",
    "听": "A", "音频": "A", "讲座": "A", "播客": "A",
    "读": "R", "文本": "R", "笔记": "R", "文章": "R", "书": "R",
    "练习": "K", "做题": "K", "模拟": "K", "实验": "K", "动手": "K", "交互": "K",
}


class CorpusExtractor:
    """从对话语料提取用户信号 (无状态)."""

    def __init__(self, known_topics: list[str] | None = None, gate: PrivacyGate | None = None) -> None:
        self._topics = known_topics or []
        self._gate = gate or PrivacyGate()

    # ---- 对外接口 ----

    def extract(self, learner_id: str, turns: list[dict[str, Any]]) -> list[UserSignal]:
        """从对话轮次提取信号列表.

        Args:
            learner_id: 学习者 ID
            turns: [{"role": "user"/"assist", "text": str, "ts": float?}, ...]

        Returns:
            提取出的 UserSignal 列表 (已过隐私门).
        """
        user_texts = [t.get("text", "") for t in turns if t.get("role") == "user" and t.get("text")]
        if not user_texts:
            return []
        # 隐私门: 任一轮含敏感信息则整批丢弃 (保守策略)
        for t in user_texts:
            ok, _ = self._gate.check(t)
            if not ok:
                return []
        sigs: list[UserSignal] = []
        sigs.extend(self._extract_interests(learner_id, user_texts))
        sigs.extend(self._extract_frustration(learner_id, user_texts))
        sigs.extend(self._extract_goals(learner_id, user_texts))
        sigs.extend(self._extract_pace(learner_id, turns))
        sigs.extend(self._extract_expression(learner_id, user_texts))
        sigs.extend(self._extract_vark(learner_id, user_texts))
        return sigs

    # ---- 各维度提取 ----

    def _extract_interests(self, learner_id: str, texts: list[str]) -> list[UserSignal]:
        counts: dict[str, int] = {}
        for t in texts:
            for topic in self._topics:
                if topic in t:
                    counts[topic] = counts.get(topic, 0) + 1
        out: list[UserSignal] = []
        for topic, n in counts.items():
            out.append(UserSignal(
                learner_id=learner_id,
                signal_type=SignalType.INTEREST,
                payload={"topic": topic, "weight": min(1.0, n * 0.4), "count": n},
                source="corpus",
            ))
        return out

    def _extract_frustration(self, learner_id: str, texts: list[str]) -> list[UserSignal]:
        hits = 0
        for t in texts:
            for w in _FRUSTRATION_WORDS:
                if w in t:
                    hits += 1
                    break
        if not hits:
            return []
        level = min(1.0, hits / 3.0)
        return [UserSignal(
            learner_id=learner_id,
            signal_type=SignalType.FRUSTRATION,
            payload={"level": level, "count": hits},
            source="corpus",
        )]

    def _extract_goals(self, learner_id: str, texts: list[str]) -> list[UserSignal]:
        out: list[UserSignal] = []
        for t in texts:
            for word, gtype in _GOAL_WORDS:
                if word in t and gtype == "long_term":
                    out.append(UserSignal(
                        learner_id=learner_id,
                        signal_type=SignalType.GOAL,
                        payload={"text": t[:40], "type": gtype, "keyword": word},
                        source="corpus",
                    ))
                    return out  # 长目标取一条即可
        return out

    def _extract_pace(self, learner_id: str, turns: list[dict[str, Any]]) -> list[UserSignal]:
        tss = [float(t.get("ts", 0)) for t in turns if t.get("ts")]
        if len(tss) < 2:
            return []
        gaps = [b - a for a, b in zip(tss, tss[1:])]
        gaps = [g for g in gaps if 0 < g < 86400]
        if not gaps:
            return []
        gaps.sort()
        median = gaps[len(gaps) // 2]
        pace = "fragmented" if median < _HABIT_THRESHOLD_S else "concentrated"
        return [UserSignal(
            learner_id=learner_id,
            signal_type=SignalType.PACE,
            payload={"pace": pace, "median_gap_s": median},
            source="corpus",
        )]

    def _extract_expression(self, learner_id: str, texts: list[str]) -> list[UserSignal]:
        if not texts:
            return []
        avg_len = sum(len(t) for t in texts) / len(texts)
        pref = "concise" if avg_len < _CONCISE_LEN else "detailed"
        return [UserSignal(
            learner_id=learner_id,
            signal_type=SignalType.EXPRESSION,
            payload={"preference": pref, "avg_len": round(avg_len, 1)},
            source="corpus",
        )]

    def _extract_vark(self, learner_id: str, texts: list[str]) -> list[UserSignal]:
        counts: dict[str, int] = {"V": 0, "A": 0, "R": 0, "K": 0}
        for t in texts:
            for kw, mod in _VARK_KEYWORDS.items():
                if kw in t:
                    counts[mod] = counts.get(mod, 0) + 1
        if not any(counts.values()):
            return []
        total = sum(counts.values())
        return [UserSignal(
            learner_id=learner_id,
            signal_type=SignalType.VARK,
            payload={k: round(v / total, 3) for k, v in counts.items()},
            source="corpus",
        )]
