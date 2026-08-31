"""画像推理引擎 — 将信号聚合到 UnderstandingProfile (规则引擎, 无 LLM 依赖).

规则:
- 兴趣: 同主题合并, 权重取 max + 时间衰减
- 目标: 追加 (长目标保留, 最多 5 条)
- 挫败: 取最近信号 level (指数平滑)
- 节奏/表达: 取最近值
- VARK: 归一化合并 (加权平均)
"""
from __future__ import annotations

from typing import Any

from dy3_polaris.l2.user_understanding.models import (
    SignalType,
    UnderstandingProfile,
    UserSignal,
)

_MAX_GOALS = 5
_MAX_INTERESTS = 10
_DECAY = 0.85  # 兴趣权重时间衰减系数


class ProfileInference:
    """将 UserSignal 列表聚合到画像 (无状态)."""

    def apply(self, profile: UnderstandingProfile, signals: list[UserSignal]) -> None:
        """将信号应用到画像 (就地更新)."""
        for sig in signals:
            st = sig.signal_type
            if st == SignalType.INTEREST:
                self._apply_interest(profile, sig)
            elif st == SignalType.GOAL:
                self._apply_goal(profile, sig)
            elif st == SignalType.FRUSTRATION:
                profile.frustration_level = min(1.0, sig.payload.get("level", 0.5))
            elif st == SignalType.PACE:
                profile.pace = sig.payload.get("pace", "unknown")
            elif st == SignalType.EXPRESSION:
                profile.expression = sig.payload.get("preference", "unknown")
            elif st == SignalType.VARK:
                self._apply_vark(profile, sig)
            elif st == SignalType.PREFERENCE:
                profile.preferences.update(sig.payload)
        # 补充推理: 习惯信号 (来自蒸馏器)
        profile.merge_from_habits()

    def _apply_interest(self, profile: UnderstandingProfile, sig: UserSignal) -> None:
        topic = str(sig.payload.get("topic", ""))
        if not topic:
            return
        weight = float(sig.payload.get("weight", 0.3))
        count = int(sig.payload.get("count", 1))
        for item in profile.interests:
            if item["topic"] == topic:
                item["weight"] = min(1.0, item.get("weight", 0.0) * _DECAY + weight)
                item["count"] = item.get("count", 0) + count
                return
        profile.interests.append({"topic": topic, "weight": weight, "count": count, "source": sig.source})
        profile.interests[:] = sorted(profile.interests, key=lambda x: x["weight"], reverse=True)[:_MAX_INTERESTS]

    def _apply_goal(self, profile: UnderstandingProfile, sig: UserSignal) -> None:
        g = {"text": sig.payload.get("text", ""), "type": sig.payload.get("type", "long_term"),
             "confidence": 0.6, "source": sig.source}
        if not g["text"]:
            return
        profile.goals.append(g)
        profile.goals[:] = profile.goals[-_MAX_GOALS:]

    def _apply_vark(self, profile: UnderstandingProfile, sig: UserSignal) -> None:
        cur = dict(profile.vark_behavior)
        incoming = {k: float(v) for k, v in sig.payload.items() if k in ("V", "A", "R", "K")}
        if not incoming:
            return
        if not cur:
            cur = incoming
        else:
            alpha = 0.5
            for k, v in incoming.items():
                cur[k] = round(cur.get(k, 0.0) * (1 - alpha) + v * alpha, 3)
        total = sum(cur.values()) or 1.0
        profile.vark_behavior = {k: round(v / total, 3) for k, v in cur.items()}
