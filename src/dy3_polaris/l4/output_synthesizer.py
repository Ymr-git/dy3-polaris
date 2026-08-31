"""L4 决策引擎层 — 输出合成器 (OutputSynthesizer).

融合世界先进方案的输出合成设计:
- Platt Scaling: 置信度校准
  - 将原始置信度通过 sigmoid 变换映射到校准置信度
  - 支持从历史反馈数据在线学习校准参数
- SafetyConstraintLayer: 安全感知输出
  - 基于置信度、行动类型、内容模式的多维安全检查
  - 四级安全等级: SAFE / CAUTION / RESTRICTED / BLOCKED
- OLIVIA: 上下文感知格式选择
  - 根据意图类型、置信度、验证状态选择最优输出格式
  - 六种格式: CONCISE / DETAILED / STRUCTURED / EXPLANATORY / COMPARATIVE / SUMMARIZED
- TDP: 输出层的上下文隔离
  - 输出合成与行动选择解耦
  - 支持独立的安全检查和置信度校准

核心职责:
    将 ActionRecord 转化为最终可交付给用户的 OutputRecord，
    包含格式化内容、校准置信度、安全评估和证据组织。

流程:
    ActionRecord → [格式选择] → [内容合成] → [置信度校准]
    → [安全检查] → [证据组织] → OutputRecord
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .models import (
    ActionRecord,
    ActionType,
    ConfidenceCalibrator,
    EvidenceItem,
    ExecutionResult,
    OutputFormat,
    OutputRecord,
    SafetyConstraint,
    SafetyLevel,
    TaskType,
    ValidationReport,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


# ============================================================
# 安全约束层
# ============================================================


class SafetyConstraintLayer:
    """安全约束层 — 多维度安全检查.

    借鉴 SafetyConstraintLayer:
    - 置信度门控: 低置信度内容自动降级
    - 模式匹配: 正则表达式检测敏感内容
    - 行动类型感知: HUMAN_CONFIRM 不直接输出

    Usage::

        layer = SafetyConstraintLayer()
        level, warnings = layer.check(
            content="Dy3+ 的波长是 575 nm",
            confidence=0.9,
            action_type=ActionType.DIRECT_ANSWER,
        )
        if level == SafetyLevel.BLOCKED:
            # 不输出
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.5,
        constraints: list[SafetyConstraint] | None = None,
    ) -> None:
        """初始化安全约束层.

        Args:
            confidence_threshold: 置信度阈值，低于此值触发 CAUTION
            constraints: 自定义安全约束列表
        """
        self._confidence_threshold = confidence_threshold
        self._constraints: list[SafetyConstraint] = constraints or self._default_constraints()

        logger.info(
            "SafetyConstraintLayer 初始化 (置信度阈值=%.2f, 约束数=%d)",
            confidence_threshold, len(self._constraints),
        )

    def check(
        self,
        content: str,
        confidence: float,
        action_type: ActionType,
    ) -> tuple[SafetyLevel, list[str]]:
        """执行安全检查.

        Args:
            content: 待检查内容
            confidence: 置信度
            action_type: 行动类型

        Returns:
            (安全等级, 警告列表)
        """
        warnings: list[str] = []
        max_level = SafetyLevel.SAFE

        # HUMAN_CONFIRM 不直接输出给用户，安全等级为 SAFE
        if action_type == ActionType.HUMAN_CONFIRM:
            return SafetyLevel.SAFE, []

        # 置信度门控
        if confidence < self._confidence_threshold:
            max_level = SafetyLevel.CAUTION
            warnings.append(
                f"置信度较低 ({confidence:.2f} < {self._confidence_threshold:.2f})，"
                "结果可能不准确"
            )

        # 自定义约束检查
        # 注意: SafetyLevel 是 str Enum, 不能用字符串值比较 (alphabetical order)
        # 必须用枚举定义顺序比较
        level_order = [
            SafetyLevel.SAFE,
            SafetyLevel.CAUTION,
            SafetyLevel.RESTRICTED,
            SafetyLevel.BLOCKED,
        ]
        for constraint in self._constraints:
            if self._check_constraint(constraint, content):
                if level_order.index(constraint.action) > level_order.index(max_level):
                    max_level = constraint.action
                warnings.append(constraint.message)

        return max_level, warnings

    @staticmethod
    def _check_constraint(constraint: SafetyConstraint, content: str) -> bool:
        """检查单个约束."""
        if not constraint.pattern:
            return False
        try:
            return bool(re.search(constraint.pattern, content))
        except re.error:
            logger.warning("无效正则模式: %s", constraint.pattern)
            return False

    @staticmethod
    def _default_constraints() -> list[SafetyConstraint]:
        """默认安全约束."""
        return [
            SafetyConstraint(
                name="no_absolutes",
                description="避免绝对化表述",
                pattern=r"(一定|必然|绝对|百分之百|毫无疑问)",
                threshold=0.0,
                action=SafetyLevel.RESTRICTED,
                message="检测到绝对化表述，建议添加不确定性说明",
            ),
            SafetyConstraint(
                name="no_sensitive_data",
                description="敏感数据保护",
                pattern=r"(密码|密钥|token|password|secret)",
                threshold=0.0,
                action=SafetyLevel.BLOCKED,
                message="内容可能包含敏感信息",
            ),
        ]


# ============================================================
# 输出合成器
# ============================================================


class OutputSynthesizer:
    """输出合成器 — T5+ 核心模块.

    将 ActionRecord 转化为最终可交付的 OutputRecord:
    1. 上下文感知格式选择 (OLIVIA)
    2. 内容合成 (多格式)
    3. Platt Scaling 置信度校准
    4. SafetyConstraintLayer 安全检查
    5. 证据组织与排序

    Usage::

        synthesizer = OutputSynthesizer()
        output = synthesizer.synthesize(
            action_record=record,
            execution_result=result,
            validation_report=report,
            intent_type="numeric",
        )
        if output.is_safe_to_output:
            print(output.content)
    """

    def __init__(
        self,
        *,
        calibrator: ConfidenceCalibrator | None = None,
        safety_layer: SafetyConstraintLayer | None = None,
    ) -> None:
        """初始化输出合成器.

        Args:
            calibrator: 置信度校准器 (Platt Scaling)
            safety_layer: 安全约束层
        """
        self._calibrator = calibrator or ConfidenceCalibrator()
        self._safety_layer = safety_layer or SafetyConstraintLayer()

        logger.info(
            "OutputSynthesizer 初始化完成 (校准器: scale=%.2f, bias=%.2f)",
            self._calibrator.scale, self._calibrator.bias,
        )

    def synthesize(
        self,
        action_record: ActionRecord,
        execution_result: ExecutionResult,
        validation_report: ValidationReport,
        *,
        intent_type: str = "",
    ) -> OutputRecord:
        """合成最终输出.

        Args:
            action_record: 行动记录 (T5 产出)
            execution_result: 执行结果 (T3 产出)
            validation_report: 验证报告 (T4 产出)
            intent_type: 意图类型

        Returns:
            OutputRecord 最终输出记录
        """
        # 1. 格式选择
        output_format = self._select_format(
            intent_type, action_record.confidence, validation_report
        )

        # 2. 内容合成
        content = self._compose_content(action_record, execution_result, output_format)
        summary = self._compose_summary(action_record, execution_result)
        structured_data = self._build_structured_data(action_record, execution_result)

        # 3. 置信度校准 (Platt Scaling)
        raw_confidence = action_record.confidence
        calibrated_confidence = self._calibrator.calibrate(raw_confidence)

        # 4. 安全检查
        # 使用 raw 和 calibrated 中的较小值进行安全检查
        # 确保低原始置信度即使被校准抬高仍触发安全约束
        safety_confidence = min(raw_confidence, calibrated_confidence)
        safety_level, safety_warnings = self._safety_layer.check(
            content=content,
            confidence=safety_confidence,
            action_type=action_record.action_type,
        )
        safety_disclaimer = self._build_disclaimer(safety_level, safety_warnings)

        # 5. 证据组织
        evidence_items = self._organize_evidence(action_record, execution_result)

        # 6. 推理链摘要
        reasoning_summary = self._summarize_reasoning(execution_result)

        output = OutputRecord(
            plan_id=action_record.plan_id,
            action_record_id=action_record.record_id,
            output_format=output_format,
            content=content,
            summary=summary,
            structured_data=structured_data,
            raw_confidence=raw_confidence,
            calibrated_confidence=round(calibrated_confidence, 4),
            calibration_params={
                "scale": self._calibrator.scale,
                "bias": self._calibrator.bias,
                "sample_count": self._calibrator.sample_count,
            },
            safety_level=safety_level,
            safety_warnings=safety_warnings,
            safety_disclaimer=safety_disclaimer,
            evidence_items=evidence_items,
            reasoning_summary=reasoning_summary,
            action_type=action_record.action_type.value,
            intent_type=intent_type,
        )

        logger.info(
            "输出合成完成: format=%s, 校准置信度=%.4f, 安全=%s, 证据=%d",
            output_format.value, calibrated_confidence,
            safety_level.value, len(evidence_items),
        )

        return output

    def update_calibrator(
        self,
        feedback_data: list[tuple[float, bool]],
    ) -> None:
        """从反馈数据更新 Platt Scaling 校准器.

        使用梯度下降拟合 logistic 回归参数:
            p = sigmoid(a * x + b)
        其中 x 是原始置信度，目标是实际正确性。

        Args:
            feedback_data: [(raw_confidence, actual_correct), ...]
        """
        if not feedback_data:
            return

        # 简化版梯度下降
        scale = self._calibrator.scale
        bias = self._calibrator.bias
        lr = 0.1  # 学习率

        for raw_conf, correct in feedback_data:
            target = 1.0 if correct else 0.0
            z = scale * raw_conf + bias
            pred = 1.0 / (1.0 + (2.718281828459045 ** (-z)))
            error = pred - target

            # 梯度
            grad_scale = error * raw_conf
            grad_bias = error

            scale -= lr * grad_scale
            bias -= lr * grad_bias

            # 约束在模型允许范围内
            scale = max(0.1, min(10.0, scale))
            bias = max(-5.0, min(5.0, bias))

        self._calibrator.scale = scale
        self._calibrator.bias = bias
        self._calibrator.sample_count += len(feedback_data)

        logger.info(
            "校准器更新: scale=%.4f, bias=%.4f, 样本=%d",
            scale, bias, self._calibrator.sample_count,
        )

    # --------------------------------------------------------
    # 格式选择
    # --------------------------------------------------------

    @staticmethod
    def _select_format(
        intent_type: str,
        confidence: float,
        validation_report: ValidationReport,
    ) -> OutputFormat:
        """根据上下文选择输出格式.

        策略:
        - concept → EXPLANATORY (教学式)
        - numeric → CONCISE (高置信度) / DETAILED (低置信度)
        - relational → COMPARATIVE
        - composite → STRUCTURED
        - 低验证分数 → DETAILED (附带完整证据)
        """
        # 低验证分数优先 DETAILED
        if validation_report.overall_score < 0.5:
            return OutputFormat.DETAILED

        format_map: dict[str, OutputFormat] = {
            "concept": OutputFormat.EXPLANATORY,
            "numeric": OutputFormat.CONCISE,
            "relational": OutputFormat.COMPARATIVE,
            "composite": OutputFormat.STRUCTURED,
        }

        return format_map.get(intent_type, OutputFormat.SUMMARIZED)

    # --------------------------------------------------------
    # 内容合成
    # --------------------------------------------------------

    @staticmethod
    def _compose_content(
        action_record: ActionRecord,
        execution_result: ExecutionResult,
        output_format: OutputFormat,
    ) -> str:
        """合成主输出内容."""
        answers = action_record.response_payload.get("answers", [])
        if not answers:
            # 从执行结果提取
            for tr in execution_result.get_results_by_type(TaskType.REASON):
                answers.extend(tr.output.get("answers", []))

        if not answers:
            return "无法生成有效回答，请重新表述您的问题。"

        primary_answer = answers[0] if isinstance(answers[0], str) else answers[0].get("text", str(answers[0]))

        if output_format == OutputFormat.CONCISE:
            return primary_answer

        if output_format == OutputFormat.DETAILED:
            parts = [primary_answer]
            reasoning = execution_result.reasoning_chain
            if reasoning:
                parts.append("\n推理过程: " + " → ".join(reasoning))
            evidence = action_record.response_payload.get("evidence", [])
            if evidence:
                parts.append("\n支持证据:")
                for i, ev in enumerate(evidence[:5], 1):
                    content = ev.get("content", str(ev)) if isinstance(ev, dict) else str(ev)
                    parts.append(f"  {i}. {content}")
            return "\n".join(parts)

        if output_format == OutputFormat.STRUCTURED:
            parts = [primary_answer, "\n详细信息:"]
            for tr in execution_result.task_results.values():
                if tr.is_success:
                    parts.append(f"  - {tr.task_type.value}: {tr.output}")
            return "\n".join(parts)

        if output_format == OutputFormat.EXPLANATORY:
            parts = [primary_answer]
            reasoning = execution_result.reasoning_chain
            if reasoning:
                parts.append("\n解释: " + " → ".join(reasoning))
            return "\n".join(parts)

        if output_format == OutputFormat.COMPARATIVE:
            parts = [primary_answer]
            answers_rest = answers[1:] if len(answers) > 1 else []
            if answers_rest:
                parts.append("\n对比分析:")
                for a in answers_rest:
                    text = a if isinstance(a, str) else a.get("text", str(a))
                    parts.append(f"  - {text}")
            return "\n".join(parts)

        # SUMMARIZED
        return primary_answer

    @staticmethod
    def _compose_summary(
        action_record: ActionRecord,
        execution_result: ExecutionResult,
    ) -> str:
        """合成一句话摘要."""
        answers = action_record.response_payload.get("answers", [])
        if not answers:
            for tr in execution_result.get_results_by_type(TaskType.REASON):
                answers.extend(tr.output.get("answers", []))

        if answers:
            first = answers[0]
            text = first if isinstance(first, str) else first.get("text", str(first))
            # 截断到 100 字符
            return text[:100] + ("..." if len(text) > 100 else "")

        return "结果摘要不可用"

    @staticmethod
    def _build_structured_data(
        action_record: ActionRecord,
        execution_result: ExecutionResult,
    ) -> dict[str, Any]:
        """构建结构化数据."""
        data: dict[str, Any] = {
            "plan_id": execution_result.plan_id,
            "action_type": action_record.action_type.value,
            "confidence": action_record.confidence,
        }

        # 提取答案
        answers = action_record.response_payload.get("answers", [])
        if answers:
            data["answers"] = [
                a if isinstance(a, str) else a.get("text", str(a))
                for a in answers
            ]

        # 提取证据摘要
        data["evidence_count"] = len(execution_result.evidence_set)
        data["reasoning_steps"] = len(execution_result.reasoning_chain)

        return data

    # --------------------------------------------------------
    # 安全免责声明
    # --------------------------------------------------------

    @staticmethod
    def _build_disclaimer(
        safety_level: SafetyLevel,
        warnings: list[str],
    ) -> str:
        """构建安全免责声明."""
        if safety_level == SafetyLevel.SAFE:
            return ""

        parts: list[str] = []
        if safety_level == SafetyLevel.CAUTION:
            parts.append("请注意: 以上结果存在一定不确定性。")
        elif safety_level == SafetyLevel.RESTRICTED:
            parts.append("免责声明: 以上结果仅供参考，请结合专业判断使用。")
        elif safety_level == SafetyLevel.BLOCKED:
            parts.append("警告: 输出内容被安全系统阻断，请联系管理员审核。")

        if warnings:
            parts.append("具体提示: " + "; ".join(warnings[:3]))

        return " ".join(parts)

    # --------------------------------------------------------
    # 证据组织
    # --------------------------------------------------------

    @staticmethod
    def _organize_evidence(
        action_record: ActionRecord,
        execution_result: ExecutionResult,
    ) -> list[EvidenceItem]:
        """组织证据，按置信度降序排列."""
        items: list[EvidenceItem] = []

        # 从执行结果提取
        for i, ev in enumerate(execution_result.evidence_set):
            if isinstance(ev, dict):
                items.append(EvidenceItem(
                    evidence_id=f"ev-{i}",
                    content=ev.get("content", str(ev)),
                    source=ev.get("source", ""),
                    source_type=ev.get("type", ""),
                    confidence=ev.get("confidence", 0.5),
                    relevance=ev.get("relevance", 0.5),
                ))
            else:
                items.append(EvidenceItem(
                    evidence_id=f"ev-{i}",
                    content=str(ev),
                    confidence=0.5,
                ))

        # 从子任务结果提取
        for tr in execution_result.task_results.values():
            for ev in tr.evidence:
                if isinstance(ev, dict) and ev not in execution_result.evidence_set:
                    items.append(EvidenceItem(
                        evidence_id=f"tr-{tr.task_id}",
                        content=ev.get("content", str(ev)),
                        source=ev.get("source", tr.task_id),
                        source_type=ev.get("type", ""),
                        confidence=tr.confidence,
                    ))

        # 按置信度降序排列
        items.sort(key=lambda x: x.confidence, reverse=True)

        return items

    # --------------------------------------------------------
    # 推理链摘要
    # --------------------------------------------------------

    @staticmethod
    def _summarize_reasoning(execution_result: ExecutionResult) -> str:
        """生成推理链摘要."""
        chain = execution_result.reasoning_chain
        if not chain:
            return "推理链不可用"

        if len(chain) <= 3:
            return " → ".join(chain)

        return f"{chain[0]} → ... → {chain[-1]} (共 {len(chain)} 步)"


__all__ = [
    "OutputSynthesizer",
    "SafetyConstraintLayer",
]
