"""CC1 四层反幻觉评审引擎 — 综合评分引擎.

实现设计文档中定义的加权评分公式:

    Score = 0.40 × L1 + 0.25 × L2 + 0.20 × L3 + 0.15 × L4

判决阈值:
- Score >= 85  → PASS  (高质量, 可信输出)
- 60 <= Score < 85 → FLAG (有风险, 需修正或标注)
- Score < 60   → BLOCK (严重幻觉, 拒绝输出)

融合世界先进方案:
- RAGAS: 加权多维评分 (faithfulness, answer_relevancy, context_recall)
- FActScore: 原子事实粒度评分
- DeepEval: 多维度综合评估
- Guardrails AI: 可配置阈值与动作映射
"""

from __future__ import annotations

from dataclasses import dataclass

from .layers import LAYER_WEIGHTS, ReviewLayerType
from .state_machine import ReviewVerdict


# ============================================================
# 评分权重配置
# ============================================================


@dataclass(frozen=True)
class ScoringWeights:
    """四层评分权重.

    权重之和应为 1.0.

    Attributes:
        l1: 事实层权重 (默认 0.40)
        l2: 逻辑层权重 (默认 0.25)
        l3: 数值层权重 (默认 0.20)
        l4: 溯源层权重 (默认 0.15)
    """

    l1: float = 0.40
    l2: float = 0.25
    l3: float = 0.20
    l4: float = 0.15

    def __post_init__(self) -> None:
        """验证权重之和为 1.0."""
        total = self.l1 + self.l2 + self.l3 + self.l4
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"权重之和必须为 1.0, 当前为 {total:.4f}"
            )

    @classmethod
    def from_layer_weights(cls) -> "ScoringWeights":
        """从 LAYER_WEIGHTS 常量创建."""
        return cls(
            l1=LAYER_WEIGHTS[ReviewLayerType.L1_FACT],
            l2=LAYER_WEIGHTS[ReviewLayerType.L2_LOGIC],
            l3=LAYER_WEIGHTS[ReviewLayerType.L3_NUMERICAL],
            l4=LAYER_WEIGHTS[ReviewLayerType.L4_PROVENANCE],
        )


# ============================================================
# 综合评分引擎
# ============================================================


class CompositeScoringEngine:
    """综合评分引擎.

    根据四层评审分数计算加权综合分数, 并根据阈值判定判决.

    评分公式::

        Score = w1 × L1 + w2 × L2 + w3 × L3 + w4 × L4

    判决阈值 (可配置)::

        Score >= pass_threshold  → PASS
        flag_threshold <= Score < pass_threshold → FLAG
        Score < flag_threshold  → BLOCK

    使用示例::

        engine = CompositeScoringEngine()
        score = engine.compute_score(l1=85.0, l2=70.0, l3=90.0, l4=80.0)
        verdict = engine.determine_verdict(score)
        assert verdict == ReviewVerdict.PASS
    """

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        pass_threshold: float = 85.0,
        flag_threshold: float = 60.0,
    ) -> None:
        self._weights = weights or ScoringWeights.from_layer_weights()
        self._pass_threshold = pass_threshold
        self._flag_threshold = flag_threshold

    @property
    def weights(self) -> ScoringWeights:
        """评分权重."""
        return self._weights

    @property
    def pass_threshold(self) -> float:
        """通过阈值."""
        return self._pass_threshold

    @property
    def flag_threshold(self) -> float:
        """警告阈值."""
        return self._flag_threshold

    def compute_score(
        self,
        l1: float,
        l2: float,
        l3: float,
        l4: float,
    ) -> float:
        """计算加权综合分数.

        Args:
            l1: 事实层评分 (0-100)
            l2: 逻辑层评分 (0-100)
            l3: 数值层评分 (0-100)
            l4: 溯源层评分 (0-100)

        Returns:
            综合评分 (0-100)
        """
        score = (
            self._weights.l1 * l1
            + self._weights.l2 * l2
            + self._weights.l3 * l3
            + self._weights.l4 * l4
        )
        return round(score, 2)

    def determine_verdict(self, score: float) -> ReviewVerdict:
        """根据评分判定判决.

        Args:
            score: 综合评分 (0-100)

        Returns:
            评审判决 (PASS / FLAG / BLOCK)
        """
        if score >= self._pass_threshold:
            return ReviewVerdict.PASS
        elif score >= self._flag_threshold:
            return ReviewVerdict.FLAG
        else:
            return ReviewVerdict.BLOCK

    def compute_and_determine(
        self,
        l1: float,
        l2: float,
        l3: float,
        l4: float,
    ) -> tuple[float, ReviewVerdict]:
        """计算评分并判定判决 (便捷方法)."""
        score = self.compute_score(l1, l2, l3, l4)
        verdict = self.determine_verdict(score)
        return score, verdict

    def compute_layer_contribution(
        self,
        l1: float,
        l2: float,
        l3: float,
        l4: float,
    ) -> dict[ReviewLayerType, float]:
        """计算各层对综合分数的贡献.

        Returns:
            层类型 → 贡献分数
        """
        return {
            ReviewLayerType.L1_FACT: round(self._weights.l1 * l1, 2),
            ReviewLayerType.L2_LOGIC: round(self._weights.l2 * l2, 2),
            ReviewLayerType.L3_NUMERICAL: round(self._weights.l3 * l3, 2),
            ReviewLayerType.L4_PROVENANCE: round(self._weights.l4 * l4, 2),
        }
