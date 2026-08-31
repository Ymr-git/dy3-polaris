"""VARK 学习风格采集引擎 — 问卷采集与行为推断.

融合世界先进方案:
- VARK 问卷模型 (Fleming & Mills, 1992):
    16 题标准问卷, 每题对应 V/A/R/K 四种模态之一.
- 行为推断 (Behavioral Inference):
    从学习行为事件 (event_type + modality) 推断模态偏好,
    基于事件频率归一化为四维分数.
- xAPI Actor-Verb-Object: 标准化学习事件采集与审计.

VARK 四模态:
- V (Visual): 视觉 — 视频/动画/图像/图表
- A (Aural): 听觉 — 音频/讲座/播客
- R (Read/Write): 读写 — 文本/阅读/笔记/文章
- K (Kinesthetic): 动觉 — 模拟/实验/练习/交互
"""

from __future__ import annotations

from typing import Any

from dy3_polaris.l1.models import VARKProfile, VARKStyle


class VARKSurveyCollector:
    """VARK 学习风格采集器.

    提供两种采集方式:
    1. collect_survey: 问卷采集 (16 题, 每题 1-4 对应 V/A/R/K)
    2. infer_from_behavior: 行为推断 (从学习事件推断模态偏好)
    """

    # 问卷答案 → 模态映射 (1=V, 2=A, 3=R, 4=K)
    _ANSWER_MAP: dict[int, str] = {
        1: "V",
        2: "A",
        3: "R",
        4: "K",
    }

    # 模态关键词 → VARK 字母映射
    # 支持直接 VARK 风格名 (visual/aural/read_write/kinesthetic)
    # 和内容类型关键词 (video/audio/text/simulation 等)
    _MODALITY_MAP: dict[str, str] = {
        # 直接 VARK 风格名
        "visual": "V",
        "aural": "A",
        "auditory": "A",
        "read_write": "R",
        "readwrite": "R",
        "reading": "R",
        "kinesthetic": "K",
        # 内容类型关键词 → Visual
        "video": "V",
        "animation": "V",
        "image": "V",
        "diagram": "V",
        "chart": "V",
        # 内容类型关键词 → Aural
        "audio": "A",
        "lecture": "A",
        "podcast": "A",
        "sound": "A",
        # 内容类型关键词 → Read/Write
        "text": "R",
        "note": "R",
        "article": "R",
        "document": "R",
        # 内容类型关键词 → Kinesthetic
        "simulation": "K",
        "experiment": "K",
        "practice": "K",
        "exercise": "K",
        "interactive": "K",
        "quiz": "K",
        "hands_on": "K",
    }

    def collect_survey(
        self,
        user_id: str,
        answers: list[int],
    ) -> VARKProfile:
        """问卷采集 — 16 题, 每题 1-4 对应 V(1)/A(2)/R(3)/K(4).

        统计各模态计数, 除以总题数得到四维分数 [0.0, 1.0].
        primary_style 由 VARKProfile 自动推导.

        Args:
            user_id: 用户 ID.
            answers: 16 个答案, 每个为 1-4.

        Returns:
            VARKProfile 画像.
        """
        counts: dict[str, int] = {"V": 0, "A": 0, "R": 0, "K": 0}
        for ans in answers:
            modality = self._ANSWER_MAP.get(ans, "V")
            counts[modality] += 1

        total = len(answers) if answers else 1
        return VARKProfile(
            user_id=user_id,
            visual_score=counts["V"] / total,
            aural_score=counts["A"] / total,
            read_write_score=counts["R"] / total,
            kinesthetic_score=counts["K"] / total,
        )

    def infer_from_behavior(
        self,
        user_id: str,
        events: list[dict[str, Any]],
    ) -> VARKProfile:
        """行为推断 — 从学习行为事件推断模态偏好.

        事件含 event_type 和 modality 字段, 按模态归类后归一化为分数.
        modality 映射优先级: modality 字段 > event_type > content_type.

        Args:
            user_id: 用户 ID.
            events: 学习事件列表, 每项含 event_type/modality/content_type.

        Returns:
            VARKProfile 画像.
        """
        counts: dict[str, int] = {"V": 0, "A": 0, "R": 0, "K": 0}
        for event in events:
            modality = self._detect_modality(event)
            counts[modality] += 1

        total = len(events) if events else 1
        return VARKProfile(
            user_id=user_id,
            visual_score=counts["V"] / total,
            aural_score=counts["A"] / total,
            read_write_score=counts["R"] / total,
            kinesthetic_score=counts["K"] / total,
        )

    def _detect_modality(self, event: dict[str, Any]) -> str:
        """从事件中检测 VARK 模态.

        检查优先级: modality > event_type > content_type.
        """
        # 1. 检查 modality 字段 (直接匹配)
        modality_raw = str(event.get("modality", "")).lower().strip()
        if modality_raw and modality_raw in self._MODALITY_MAP:
            return self._MODALITY_MAP[modality_raw]

        # 2. 检查 event_type 字段 (关键词匹配)
        event_type = str(event.get("event_type", "")).lower().strip()
        for keyword, mode in self._MODALITY_MAP.items():
            if keyword in event_type:
                return mode

        # 3. 检查 content_type 字段 (关键词匹配)
        content_type = str(event.get("content_type", "")).lower().strip()
        for keyword, mode in self._MODALITY_MAP.items():
            if keyword in content_type:
                return mode

        # 默认归为 Visual
        return "V"
