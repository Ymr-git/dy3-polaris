"""L4 决策引擎层 — 不确定性量化网关 (UQGate).

借鉴世界先进方案:
- UniCR (2025): 统一置信度与拒绝框架，融合异构不确定性证据
- CoT-UQ (2025): 利用 CoT 推理过程改进响应级不确定性量化
- MUSE (2025): 基于 Jensen-Shannon 散度的多模型不确定性聚合

核心职责:
    融合多源不确定性信号，输出校准后的置信度分数，
    并根据置信度自动选择验证层级 (L1/L2/L3)。

不确定性信号:
    1. 执行置信度 (execution_confidence): TaskExecutor 产出的综合置信度
    2. 检索兼容性 (retrieval_compatibility): 检索结果与查询的匹配度
    3. 自洽性离散度 (consistency_dispersion): 多路径推理答案的一致性
    4. 证据充分性 (evidence_sufficiency): 证据数量与质量
    5. 模型先验 (model_prior): 基于意图类型和历史反馈的先验置信度
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .models import (
    ExecutionResult,
    ExecutionStatus,
    TaskType,
    ValidationTier,
)

logger = logging.getLogger(__name__)


class UQSignal:
    """单个不确定性信号."""

    def __init__(
        self,
        name: str,
        raw_value: float,
        *,
        weight: float = 1.0,
        calibrated: bool = False,
    ) -> None:
        self.name = name
        self.raw_value = raw_value
        self.weight = weight
        self.calibrated = calibrated

    @property
    def confidence(self) -> float:
        """将原始值转为置信度 (0~1)."""
        return max(0.0, min(1.0, self.raw_value))


class UQGate:
    """不确定性量化网关 — 借鉴 UniCR 多证据融合框架.

    融合多源不确定性信号，输出校准后的置信度分数，
    并根据置信度阈值选择验证层级。

    Usage::

        gate = UQGate(
            l1_threshold=0.75,
            l2_threshold=0.5,
        )
        uq_result = gate.assess(execution_result, intent_type="numeric")
        # uq_result.score -> 校准后置信度
        # uq_result.tier  -> 建议验证层级
    """

    def __init__(
        self,
        *,
        l1_threshold: float = 0.75,
        l2_threshold: float = 0.50,
        l3_threshold: float = 0.30,
        # 信号权重
        weight_execution: float = 0.30,
        weight_retrieval: float = 0.25,
        weight_consistency: float = 0.20,
        weight_evidence: float = 0.15,
        weight_prior: float = 0.10,
    ) -> None:
        """初始化 UQ 网关.

        Args:
            l1_threshold: L1 轻量验证的置信度下限 (>= 此值走 L1)
            l2_threshold: L2 标准验证的置信度下限 (>= 此值走 L2)
            l3_threshold: L3 深度验证的置信度下限 (>= 此值走 L3)
            weight_execution: 执行置信度权重
            weight_retrieval: 检索兼容性权重
            weight_consistency: 自洽性离散度权重
            weight_evidence: 证据充分性权重
            weight_prior: 模型先验权重
        """
        self._thresholds = {
            ValidationTier.L1_LIGHTWEIGHT: l1_threshold,
            ValidationTier.L2_STANDARD: l2_threshold,
            ValidationTier.L3_DEEP: l3_threshold,
        }
        self._weights = {
            "execution": weight_execution,
            "retrieval": weight_retrieval,
            "consistency": weight_consistency,
            "evidence": weight_evidence,
            "prior": weight_prior,
        }

        logger.info(
            "UQGate 初始化 (阈值: L1>=%.2f, L2>=%.2f, L3>=%.2f)",
            l1_threshold, l2_threshold, l3_threshold,
        )

    def assess(
        self,
        execution_result: ExecutionResult,
        *,
        intent_type: str = "",
        historical_feedback: dict[str, float] | None = None,
    ) -> UQAssessment:
        """评估执行结果的不确定性.

        Args:
            execution_result: T3 产出的执行结果
            intent_type: 意图类型 (用于先验调整)
            historical_feedback: 历史反馈统计 (意图类型 -> 平均评分)

        Returns:
            UQAssessment 不确定性量化结果
        """
        signals: list[UQSignal] = []

        # 信号 1: 执行置信度
        exec_conf = execution_result.confidence
        if execution_result.status == ExecutionStatus.FAILED:
            exec_conf = 0.0
        signals.append(UQSignal(
            "execution_confidence", exec_conf,
            weight=self._weights["execution"],
        ))

        # 信号 2: 检索兼容性
        retrieval_conf = self._assess_retrieval_compatibility(execution_result)
        signals.append(UQSignal(
            "retrieval_compatibility", retrieval_conf,
            weight=self._weights["retrieval"],
        ))

        # 信号 3: 自洽性离散度
        consistency = self._assess_consistency(execution_result)
        signals.append(UQSignal(
            "consistency_dispersion", consistency,
            weight=self._weights["consistency"],
        ))

        # 信号 4: 证据充分性
        evidence_score = self._assess_evidence_sufficiency(execution_result)
        signals.append(UQSignal(
            "evidence_sufficiency", evidence_score,
            weight=self._weights["evidence"],
        ))

        # 信号 5: 模型先验
        prior = self._assess_prior(intent_type, historical_feedback)
        signals.append(UQSignal(
            "model_prior", prior,
            weight=self._weights["prior"],
        ))

        # 加权融合
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            fused_score = 0.0
        else:
            fused_score = sum(s.confidence * s.weight for s in signals) / total_weight

        # 温度缩放校准 (借鉴 UniCR)
        calibrated_score = self._temperature_scale(fused_score, intent_type)

        # 选择验证层级
        tier = self._select_tier(calibrated_score)

        # 收集信号详情
        signal_details = {
            s.name: {
                "raw_value": round(s.confidence, 4),
                "weight": round(s.weight, 4),
            }
            for s in signals
        }

        logger.info(
            "UQ 评估完成: score=%.4f, tier=%s, signals=%s",
            calibrated_score, tier.value, signal_details,
        )

        return UQAssessment(
            score=calibrated_score,
            tier=tier,
            signals=signal_details,
            raw_fused_score=round(fused_score, 4),
        )

    # --------------------------------------------------------
    # 信号评估方法
    # --------------------------------------------------------

    @staticmethod
    def _assess_retrieval_compatibility(result: ExecutionResult) -> float:
        """评估检索兼容性 — 检索结果与查询的匹配度."""
        retrieve_results = result.get_results_by_type(TaskType.RETRIEVE)
        if not retrieve_results:
            return 0.3  # 无检索结果，低兼容性

        scores = []
        for tr in retrieve_results:
            # 基于检索结果数量和置信度
            output = tr.output
            total = output.get("total", 0)
            if total == 0:
                scores.append(0.2)
                continue

            # 结果越多越好（但有上限）
            count_score = min(1.0, total / 5.0)
            # 置信度
            conf_score = tr.confidence
            # 综合
            scores.append(0.5 * count_score + 0.5 * conf_score)

        return sum(scores) / len(scores) if scores else 0.3

    @staticmethod
    def _assess_consistency(result: ExecutionResult) -> float:
        """评估自洽性 — 多路径推理答案的一致性."""
        reason_results = result.get_results_by_type(TaskType.REASON)
        if not reason_results:
            return 0.8  # 无推理结果，默认较高一致性

        if len(reason_results) == 1:
            # 单路径推理，基于置信度
            return reason_results[0].confidence

        # 多路径推理，检查答案一致性
        answer_sets = []
        for tr in reason_results:
            answers = tr.output.get("answers", [])
            if answers:
                answer_sets.append(set(
                    str(a.get("text", "")) if isinstance(a, dict) else str(a)
                    for a in answers
                ))

        if len(answer_sets) < 2:
            return sum(r.confidence for r in reason_results) / len(reason_results)

        # 计算两两 Jaccard 相似度
        similarities = []
        for i in range(len(answer_sets)):
            for j in range(i + 1, len(answer_sets)):
                union = answer_sets[i] | answer_sets[j]
                if union:
                    inter = answer_sets[i] & answer_sets[j]
                    similarities.append(len(inter) / len(union))

        if not similarities:
            return 0.5

        avg_sim = sum(similarities) / len(similarities)
        # 结合推理置信度
        avg_conf = sum(r.confidence for r in reason_results) / len(reason_results)
        return 0.6 * avg_sim + 0.4 * avg_conf

    @staticmethod
    def _assess_evidence_sufficiency(result: ExecutionResult) -> float:
        """评估证据充分性 — 证据数量与质量."""
        evidence_count = len(result.evidence_set)
        if evidence_count == 0:
            return 0.2

        # 证据数量得分 (5条以上为满分)
        count_score = min(1.0, evidence_count / 5.0)

        # 证据类型多样性
        evidence_types = set()
        for ev in result.evidence_set:
            if isinstance(ev, dict):
                evidence_types.add(ev.get("type", "unknown"))
            else:
                evidence_types.add("unknown")
        diversity_score = min(1.0, len(evidence_types) / 3.0)

        return 0.6 * count_score + 0.4 * diversity_score

    @staticmethod
    def _assess_prior(
        intent_type: str,
        historical_feedback: dict[str, float] | None,
    ) -> float:
        """评估模型先验 — 基于意图类型和历史反馈."""
        # 基础先验
        base_prior = 0.7

        # 意图类型调整
        intent_adjustments = {
            "concept": 0.1,     # 概念查询通常较可靠
            "numeric": -0.1,    # 数值查询需要精确，先验略低
            "relational": 0.0,  # 关系查询中等
            "composite": -0.15, # 复合查询最不确定
        }
        base_prior += intent_adjustments.get(intent_type, 0.0)

        # 历史反馈调整
        if historical_feedback and intent_type in historical_feedback:
            avg_rating = historical_feedback[intent_type]
            # rating -1~1 映射到 0~1
            feedback_score = (avg_rating + 1.0) / 2.0
            base_prior = 0.5 * base_prior + 0.5 * feedback_score

        return max(0.1, min(1.0, base_prior))

    # --------------------------------------------------------
    # 校准与层级选择
    # --------------------------------------------------------

    @staticmethod
    def _temperature_scale(score: float, intent_type: str = "") -> float:
        """温度缩放校准 (借鉴 UniCR).

        对不同意图类型使用不同的温度参数，
        使校准后的分数更好地反映真实正确概率。
        """
        # 不同意图类型的温度参数
        temperatures = {
            "concept": 1.0,     # 概念查询: 标准温度
            "numeric": 0.8,     # 数值查询: 更保守
            "relational": 1.1,  # 关系查询: 略宽松
            "composite": 0.7,   # 复合查询: 最保守
        }
        temp = temperatures.get(intent_type, 1.0)

        if temp == 1.0:
            return max(0.0, min(1.0, score))

        # 温度缩放: score^(1/T)
        calibrated = math.pow(max(0.001, score), 1.0 / temp)
        return max(0.0, min(1.0, calibrated))

    def _select_tier(self, score: float) -> ValidationTier:
        """根据校准后分数选择验证层级."""
        if score >= self._thresholds[ValidationTier.L1_LIGHTWEIGHT]:
            return ValidationTier.L1_LIGHTWEIGHT
        if score >= self._thresholds[ValidationTier.L2_STANDARD]:
            return ValidationTier.L2_STANDARD
        return ValidationTier.L3_DEEP


class UQAssessment:
    """不确定性量化评估结果."""

    def __init__(
        self,
        score: float,
        tier: ValidationTier,
        signals: dict[str, dict[str, float]],
        raw_fused_score: float,
    ) -> None:
        self.score = score
        self.tier = tier
        self.signals = signals
        self.raw_fused_score = raw_fused_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "tier": self.tier.value,
            "signals": self.signals,
            "raw_fused_score": round(self.raw_fused_score, 4),
        }


__all__ = [
    "UQGate",
    "UQSignal",
    "UQAssessment",
]
