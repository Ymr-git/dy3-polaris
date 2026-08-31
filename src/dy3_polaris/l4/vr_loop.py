"""L4 决策引擎层 — 验证-精炼闭环 (VRLoop).

借鉴世界先进方案:
- VRR-Stop (2026): 验证-修复循环的鲁棒停止框架
  - 四参数噪声模型: 验证器假阳/假阴率 + 修复器修复率/损坏率
  - 基于真实边际增益的停止决策
  - VRR-Guard 保守回退策略
- VeReaFine (2025): 迭代验证-推理-精炼 RAG
  - 检索、验证、生成三阶段交错
  - 识别缺失证据并触发补充检索
- PAG (2025): 选择性修正机制
  - 只有验证器高置信地检测到错误时才触发修正

核心职责:
    1. 当验证不通过时，生成结构化反馈 (定位问题类型)
    2. 根据反馈类型触发定向修正
    3. 基于边际增益判断是否继续迭代
    4. 最多 max_iterations 轮，防止过度修正
"""

from __future__ import annotations

import logging
from typing import Any

from .models import (
    ExecutionResult,
    ExecutionStatus,
    ValidationReport,
    ValidationSeverity,
    ValidationTier,
)

logger = logging.getLogger(__name__)


class RefinementFeedback:
    """结构化精炼反馈."""

    def __init__(
        self,
        feedback_type: str,
        severity: str,
        location: str,
        description: str,
        suggested_action: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.feedback_type = feedback_type      # factual/logical/incomplete/stylistic
        self.severity = severity                # error/warning/info
        self.location = location                # 问题定位 (task_id / claim / dimension)
        self.description = description          # 问题描述
        self.suggested_action = suggested_action  # 建议操作
        self.details = details or {}


class VRLoopController:
    """验证-精炼闭环控制器.

    借鉴 VRR-Stop 理论:
    - 追踪每轮验证的分数变化
    - 当边际增益递减时停止迭代
    - 当验证器噪声可能导致"越修越差"时保守回退

    借鉴 VeReaFine:
    - 根据反馈类型触发不同修正策略
    - 缺失证据 -> 补充检索
    - 逻辑错误 -> 推理链修正
    - 事实错误 -> 数据校验

    Usage::

        controller = VRLoopController(max_iterations=3)
        while controller.should_continue(report):
            feedback = controller.generate_feedback(report, execution_result)
            # 执行修正...
            new_report = validate(corrected_result)
            controller.record_iteration(feedback, report, new_report)
            report = new_report
    """

    def __init__(
        self,
        *,
        max_iterations: int = 3,
        min_improvement: float = 0.05,
        noise_tolerance: float = 0.1,
        enable_guard: bool = True,
    ) -> None:
        """初始化 V&R 闭环控制器.

        Args:
            max_iterations: 最大迭代轮次
            min_improvement: 最小改善幅度，低于此值停止
            noise_tolerance: 验证器噪声容忍度
            enable_guard: 是否启用 VRR-Guard 保守回退
        """
        self._max_iterations = max_iterations
        self._min_improvement = min_improvement
        self._noise_tolerance = noise_tolerance
        self._enable_guard = enable_guard

        self._iteration_count = 0
        self._score_history: list[float] = []
        self._feedback_history: list[dict[str, Any]] = []

        logger.info(
            "VRLoopController 初始化 (最大轮次: %d, 最小改善: %.2f)",
            max_iterations, min_improvement,
        )

    def should_continue(self, report: ValidationReport) -> bool:
        """判断是否应继续迭代.

        借鉴 VRR-Stop 停止条件:
        1. 已达到最大轮次
        2. 验证已通过
        3. 边际增益低于阈值
        4. VRR-Guard: 分数下降趋势

        Args:
            report: 当前验证报告

        Returns:
            是否应继续迭代
        """
        # 条件 1: 已达到最大轮次
        if self._iteration_count >= self._max_iterations:
            logger.debug("停止: 已达到最大轮次 %d", self._max_iterations)
            return False

        # 条件 2: 验证已通过
        if report.is_valid and report.overall_score >= 0.85:
            logger.debug("停止: 验证已通过 (score=%.4f)", report.overall_score)
            return False

        # 记录分数
        self._score_history.append(report.overall_score)

        # 条件 3: 边际增益低于阈值 (至少 2 轮才能计算)
        if len(self._score_history) >= 2:
            gain = self._score_history[-1] - self._score_history[-2]
            if gain < self._min_improvement:
                logger.info(
                    "停止: 边际增益 %.4f 低于阈值 %.2f",
                    gain, self._min_improvement,
                )
                return False

        # 条件 4: VRR-Guard — 分数下降
        if self._enable_guard and len(self._score_history) >= 2:
            current = self._score_history[-1]
            previous = self._score_history[-2]
            if current < previous - self._noise_tolerance:
                logger.warning(
                    "VRR-Guard 触发: 分数下降 (%.4f -> %.4f)，停止迭代防止过度修正",
                    previous, current,
                )
                return False

        return True

    def generate_feedback(
        self,
        report: ValidationReport,
        execution_result: ExecutionResult,
    ) -> list[RefinementFeedback]:
        """根据验证报告生成结构化反馈.

        Args:
            report: 验证报告
            execution_result: 执行结果

        Returns:
            反馈列表
        """
        feedbacks: list[RefinementFeedback] = []

        # 从异常中生成反馈
        for anomaly in report.anomalies:
            source = anomaly.get("source", "")
            message = anomaly.get("message", "")
            severity = anomaly.get("severity", "warning")

            feedback = self._anomaly_to_feedback(source, message, severity, report)
            if feedback:
                feedbacks.append(feedback)

        # 从 Faithfulness 评估中生成反馈
        faithfulness = report.faithfulness_assessment
        if faithfulness and faithfulness.get("unsupported_claims"):
            for unsupported in faithfulness["unsupported_claims"]:
                claim = unsupported.get("claim", {})
                feedbacks.append(RefinementFeedback(
                    feedback_type="factual",
                    severity="error",
                    location=claim.get("raw_text", ""),
                    description=f"主张未被检索上下文支持 (支持度: {unsupported.get('support_score', 0):.2f})",
                    suggested_action="supplement_retrieval",
                    details={
                        "claim_type": claim.get("type", ""),
                        "suggested_query": next(
                            (me["suggested_query"]
                             for me in faithfulness.get("missing_evidence", [])
                             if me.get("claim_subject") == claim.get("subject", "")),
                            "",
                        ),
                    },
                ))

        # 从自洽性检查中生成反馈
        consistency = report.self_consistency
        if consistency and consistency.get("contradictions"):
            for contradiction in consistency["contradictions"]:
                feedbacks.append(RefinementFeedback(
                    feedback_type="logical",
                    severity="error",
                    location="reasoning_paths",
                    description=(
                        f"推理路径矛盾: 主导答案 '{contradiction.get('dominant_answer', '')}' "
                        f"vs 冲突答案 '{contradiction.get('conflicting_answer', '')}'"
                    ),
                    suggested_action="reasoning_correction",
                    details=contradiction,
                ))

        # 从质量评估中生成反馈
        quality = report.quality_assessment
        if quality and quality.get("dimensions"):
            weak_dims = {
                k: v for k, v in quality["dimensions"].items()
                if v < 0.6
            }
            for dim, score in weak_dims.items():
                feedbacks.append(RefinementFeedback(
                    feedback_type="incomplete",
                    severity="warning" if score >= 0.4 else "error",
                    location=f"quality.{dim}",
                    description=f"质量维度 '{dim}' 得分 {score:.2f} 低于阈值",
                    suggested_action="quality_enhancement",
                    details={"dimension": dim, "score": score},
                ))

        # 从合规检查中生成反馈
        compliance = report.compliance_check
        if compliance and compliance.get("checks"):
            for check in compliance["checks"]:
                if not check.get("passed", True):
                    feedbacks.append(RefinementFeedback(
                        feedback_type="incomplete",
                        severity=check.get("severity", "warning"),
                        location=f"compliance.{check.get('check', '')}",
                        description=check.get("message", ""),
                        suggested_action="resource_optimization",
                        details=check,
                    ))

        logger.info(
            "生成 %d 条精炼反馈 (类型分布: %s)",
            len(feedbacks),
            {f.feedback_type: sum(1 for x in feedbacks if x.feedback_type == f.feedback_type)
             for f in feedbacks},
        )

        return feedbacks

    def record_iteration(
        self,
        feedbacks: list[RefinementFeedback],
        old_report: ValidationReport,
        new_report: ValidationReport,
    ) -> None:
        """记录一轮迭代的结果.

        Args:
            feedbacks: 本轮的反馈
            old_report: 修正前的验证报告
            new_report: 修正后的验证报告
        """
        self._iteration_count += 1

        iteration_record = {
            "iteration": self._iteration_count,
            "old_score": round(old_report.overall_score, 4),
            "new_score": round(new_report.overall_score, 4),
            "improvement": round(new_report.overall_score - old_report.overall_score, 4),
            "feedback_count": len(feedbacks),
            "feedback_types": [f.feedback_type for f in feedbacks],
            "actions_taken": [f.suggested_action for f in feedbacks],
        }

        self._feedback_history.append(iteration_record)

        logger.info(
            "V&R 迭代 %d 完成: 分数 %.4f -> %.4f (改善 %.4f)",
            self._iteration_count,
            old_report.overall_score,
            new_report.overall_score,
            iteration_record["improvement"],
        )

    def get_iteration_summary(self) -> dict[str, Any]:
        """获取迭代历史摘要."""
        return {
            "total_iterations": self._iteration_count,
            "max_iterations": self._max_iterations,
            "score_history": [round(s, 4) for s in self._score_history],
            "iteration_details": self._feedback_history,
            "final_improvement": (
                round(self._score_history[-1] - self._score_history[0], 4)
                if len(self._score_history) >= 2
                else 0.0
            ),
        }

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    @staticmethod
    def _anomaly_to_feedback(
        source: str,
        message: str,
        severity: str,
        report: ValidationReport,
    ) -> RefinementFeedback | None:
        """将异常转换为结构化反馈."""
        if source == "fact_check":
            return RefinementFeedback(
                feedback_type="factual",
                severity=severity,
                location="fact_check",
                description=message,
                suggested_action="fact_verification",
                details={"fact_check_score": report.fact_check.get("score", 0)},
            )
        if source == "conflict_detection":
            return RefinementFeedback(
                feedback_type="logical",
                severity=severity,
                location="conflict_detection",
                description=message,
                suggested_action="conflict_resolution",
                details={"conflict_score": report.conflict_detection.get("score", 0)},
            )
        if source == "compliance":
            return RefinementFeedback(
                feedback_type="incomplete",
                severity=severity,
                location="compliance",
                description=message,
                suggested_action="resource_optimization",
            )
        # 未知来源
        if message:
            return RefinementFeedback(
                feedback_type="stylistic",
                severity=severity,
                location=source or "unknown",
                description=message,
                suggested_action="general_improvement",
            )
        return None


__all__ = [
    "RefinementFeedback",
    "VRLoopController",
]
