"""学习风格推断器 — VARK 问卷 + 行为推断.

融合世界先进方案:
- Fleming & Mills (1992): VARK 四模态模型
- Khan Academy: 行为驱动的学习风格推断
- xAPI: 标准化学习事件采集

推断策略:
1. 问卷优先: 有 VARK 问卷数据时直接使用
2. 行为推断: 无问卷时从学习行为事件推断
3. 默认兜底: 无任何数据时默认 "reading"

VARK 四模态 -> 风格标签映射:
- V (Visual)     -> "visual"
- A (Aural)      -> "aural"
- R (Read/Write) -> "reading"
- K (Kinesthetic) -> "kinesthetic"
- 多维接近 (差 < 0.05) -> "multimodal"

设计说明:
- StyleInferrer 为无状态引擎类, 不持有学习者状态.
- 行为推断复用模态关键词映射, 默认模态为 "R" (reading),
  与无数据兜底策略一致.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# 1. 常量定义
# ============================================================

# 默认学习风格 (无数据时兜底)
DEFAULT_STYLE: str = "reading"

# 多模态判定阈值: 最高分与次高分差值 < 此值 -> multimodal
_MULTIMODAL_THRESHOLD: float = 0.05

# VARK 模态字母 -> 风格标签
_MODA_TO_STYLE: dict[str, str] = {
    "V": "visual",
    "A": "aural",
    "R": "reading",
    "K": "kinesthetic",
}

# VARK 四维分数键名 (支持长键与短键)
_VARK_KEYS: dict[str, str] = {
    "visual_score": "V",
    "visual": "V",
    "aural_score": "A",
    "aural": "A",
    "read_write_score": "R",
    "read_write": "R",
    "kinesthetic_score": "K",
    "kinesthetic": "K",
}

# 模态关键词 -> VARK 字母映射 (用于行为事件检测)
# 支持 VARK 风格名 + 内容类型关键词
_MODALITY_KEYWORDS: dict[str, str] = {
    # Visual
    "visual": "V",
    "video": "V",
    "animation": "V",
    "image": "V",
    "diagram": "V",
    "chart": "V",
    # Aural
    "aural": "A",
    "auditory": "A",
    "audio": "A",
    "lecture": "A",
    "podcast": "A",
    "sound": "A",
    # Read/Write
    "read_write": "R",
    "readwrite": "R",
    "reading": "R",
    "text": "R",
    "note": "R",
    "article": "R",
    "document": "R",
    # Kinesthetic
    "kinesthetic": "K",
    "simulation": "K",
    "experiment": "K",
    "practice": "K",
    "exercise": "K",
    "interactive": "K",
    "quiz": "K",
    "hands_on": "K",
}

# 无法识别模态时的默认字母 (R = reading, 与无数据兜底一致)
_DEFAULT_MODALITY: str = "R"


# ============================================================
# 2. StyleInferrer 无状态引擎类
# ============================================================


class StyleInferrer:
    """学习风格推断器 (无状态引擎).

    提供两种推断方式:
    1. ``infer_from_vark``: 从 VARK 四维分数推断主要风格 (问卷优先).
    2. ``infer_from_behavior``: 从学习行为事件推断风格 (行为推断).

    推断规则:
    - 单维度主导 (最高分与其他维度差 >= 0.05): 返回该维度对应风格.
    - 多维接近 (最高分与次高分差 < 0.05): 返回 "multimodal".
    - 无有效数据 (空字典 / 全零 / 空事件): 返回 "reading" (默认兜底).

    无状态: 相同输入产生相同输出, 可安全多实例并发使用.
    """

    def infer_from_vark(self, vark_profile: dict[str, Any]) -> str:
        """从 VARK 四维分数推断主要学习风格.

        从字典中提取 VARK 四维分数 (支持长键 visual_score / 短键 visual),
        找到最高分维度. 若多个维度与最高分差值 < 0.05, 返回 "multimodal".

        Args:
            vark_profile: VARK 画像字典, 含 visual_score/aural_score/
                read_write_score/kinesthetic_score (或短键名).

        Returns:
            学习风格标签: "visual"/"aural"/"reading"/"kinesthetic"/"multimodal".
            无有效数据 (空字典或全零) 时返回 "reading".
        """
        # 提取四维分数
        scores: dict[str, float] = {"V": 0.0, "A": 0.0, "R": 0.0, "K": 0.0}
        for key, modality in _VARK_KEYS.items():
            if key in vark_profile:
                scores[modality] = max(scores[modality], float(vark_profile[key]))

        return self._infer_from_scores(scores)

    def infer_from_behavior(self, events: list[dict[str, Any]]) -> str:
        """从学习行为事件推断学习风格.

        逐条检测事件的模态 (modality > event_type > content_type 关键词匹配),
        按模态计数后归一化为四维分数, 再推断主要风格.

        Args:
            events: 学习事件列表, 每项含 modality/event_type/content_type.

        Returns:
            学习风格标签: "visual"/"aural"/"reading"/"kinesthetic"/"multimodal".
            空事件列表时返回 "reading" (默认兜底).
        """
        if not events:
            return DEFAULT_STYLE

        # 按模态计数
        counts: dict[str, int] = {"V": 0, "A": 0, "R": 0, "K": 0}
        for event in events:
            modality = self._detect_modality(event)
            counts[modality] += 1

        # 归一化为分数
        total = len(events)
        scores: dict[str, float] = {m: c / total for m, c in counts.items()}

        return self._infer_from_scores(scores)

    # --- 内部方法 ---

    @staticmethod
    def _infer_from_scores(scores: dict[str, float]) -> str:
        """从四维分数推断风格 (共享逻辑).

        - 全零 -> "reading" (无有效数据兜底).
        - 多维接近最高分 (差 < 0.05) -> "multimodal".
        - 否则 -> 最高分维度对应风格.

        Args:
            scores: {"V": float, "A": float, "R": float, "K": float}.

        Returns:
            风格标签.
        """
        max_score = max(scores.values())

        # 全零 -> 默认 "reading"
        if max_score <= 0.0:
            return DEFAULT_STYLE

        # 检查是否有多个维度接近最高分 (差 < 0.05)
        near_max = [s for s in scores.values() if max_score - s < _MULTIMODAL_THRESHOLD]
        if len(near_max) >= 2:
            return "multimodal"

        # 返回最高分维度对应风格
        best_modality = max(scores, key=scores.get)
        return _MODA_TO_STYLE[best_modality]

    @staticmethod
    def _detect_modality(event: dict[str, Any]) -> str:
        """从事件中检测 VARK 模态.

        检查优先级: modality 字段 > event_type 关键词 > content_type 关键词.
        无法识别时默认 "R" (reading).

        Args:
            event: 学习事件字典.

        Returns:
            VARK 模态字母: "V"/"A"/"R"/"K".
        """
        # 1. 检查 modality 字段 (直接匹配)
        modality_raw = str(event.get("modality", "")).lower().strip()
        if modality_raw and modality_raw in _MODALITY_KEYWORDS:
            return _MODALITY_KEYWORDS[modality_raw]

        # 2. 检查 event_type 字段 (关键词包含匹配)
        event_type = str(event.get("event_type", "")).lower().strip()
        for keyword, mode in _MODALITY_KEYWORDS.items():
            if keyword in event_type:
                return mode

        # 3. 检查 content_type 字段 (关键词包含匹配)
        content_type = str(event.get("content_type", "")).lower().strip()
        for keyword, mode in _MODALITY_KEYWORDS.items():
            if keyword in content_type:
                return mode

        # 默认 "R" (reading)
        return _DEFAULT_MODALITY


# ============================================================
# __all__
# ============================================================

__all__ = [
    "StyleInferrer",
]
