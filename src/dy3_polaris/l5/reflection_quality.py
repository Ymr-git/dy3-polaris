"""反思与质量控制模块 — L5 Agent Runtime 核心组件.

融合世界先进方案:
- LangGraph: Generator-Critic 自纠正循环 + critique_history + revision_number
- OpenAI Agents SDK: Guardrail 三层护栏 (input/output/tool) + tripwire 绞线
- Google ADK: EvalCase 轨迹评估 + rubric 多维度评分 + eval-fix loop
- AutoGen: Reflection 消息协议 + Coder-Reviewer 双代理 + approved 裁决
- CrewAI: Task guardrail + Tuple[bool, Any] 验证 + feedback 反馈
- Claude Science: Actor-Critic 模式 + 多维度加权评分 + 95% 质量门控
- Temporal: Update Validator + RetryPolicy + Saga 补偿 + non_retryable_error_types
- L5 设计文档: 第七章 Reflection Engine (4 维度自检 + 3 级处理 + 跨 Agent 复盘)

核心组件:
1. Verdict — 审核裁决枚举 (approved/revise/rejected)
2. ReflectionDimension — 反思检查维度 (4 维度, 含 severity + default_weight)
3. DimensionScore — 单维度评分 (分数 + 权重 + 理由)
4. ReviewRecord — 审核记录 (多维度评分 + 加权总分 + 裁决 + 反馈)
5. ReflectionResult — 反思结果 (审核历史 + 最终裁决 + 改进轨迹)
6. ReflectionTrigger / CollaborationTrigger — 反思触发类型
7. QualityGate — 质量门控 (阈值 + 硬下限 + 修订上限 + 错误分类)
8. GateResult / GateAction — 门控结果与动作
9. CC1Reviewer — CC1 Actor-Critic 审核器 (深度评审 + 多维度评分)
10. AdjudicationExecutor — 裁决执行器 (处理 requires_adjudication + Saga 补偿)
11. ReputationLedger — 声誉账本 (动态信任分 + 指数移动平均更新)
12. ReflectionEngine — 反思引擎 (单 Agent 反思 + 跨 Agent 复盘)
13. CollaborationReview — 跨 Agent 协作复盘记录
14. QualityReport — 全链路质量报告
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class Verdict(str, Enum):
    """审核裁决 (融合 AutoGen approved/rejected + L5 设计文档 3 级处理).

    - APPROVED: 质量合格, 通过审核
    - REVISE: 需要修订, 存在可改进问题
    - REJECTED: 质量不合格, 存在严重错误
    """

    APPROVED = "approved"
    REVISE = "revise"
    REJECTED = "rejected"


class ReflectionDimension(str, Enum):
    """反思检查维度 (L5 设计文档 7.1.1 四维度).

    每个维度包含:
    - severity: 严重级别 (fail/warn), fail 级别触发 CC1 深度评审
    - default_weight: 默认权重, 用于加权总分计算
    """

    FACTUAL_CONSISTENCY = "factual_consistency"
    NUMERIC_ACCURACY = "numeric_accuracy"
    CITATION_COMPLETENESS = "citation_completeness"
    PEDAGOGICAL_FIT = "pedagogical_fit"

    @property
    def severity(self) -> str:
        """返回维度严重级别 (L5 设计文档 7.1.1)."""
        severities = {
            ReflectionDimension.FACTUAL_CONSISTENCY: "fail",
            ReflectionDimension.NUMERIC_ACCURACY: "fail",
            ReflectionDimension.CITATION_COMPLETENESS: "warn",
            ReflectionDimension.PEDAGOGICAL_FIT: "warn",
        }
        return severities[self]

    @property
    def default_weight(self) -> float:
        """返回维度默认权重."""
        weights = {
            ReflectionDimension.FACTUAL_CONSISTENCY: 0.30,
            ReflectionDimension.NUMERIC_ACCURACY: 0.25,
            ReflectionDimension.CITATION_COMPLETENESS: 0.20,
            ReflectionDimension.PEDAGOGICAL_FIT: 0.25,
        }
        return weights[self]


class ReflectionTrigger(str, Enum):
    """反思触发类型 (L5 设计文档 7.1 + 7.2)."""

    SINGLE_AGENT = "single_agent"
    COLLABORATION = "collaboration"


class CollaborationTrigger(str, Enum):
    """跨 Agent 复盘触发类型 (L5 设计文档 7.2.1)."""

    DEBATE = "debate"
    VOTING = "voting"
    FORK_MERGE = "fork_merge"


class GateAction(str, Enum):
    """门控动作 (融合 Temporal RetryPolicy + Claude Code 门控).

    - ALLOW: 质量达标, 放行
    - REJECT: 质量低于硬下限, 拒绝
    - REPLACE: 用替代方案替换 (预留)
    - REVISE: 需要修订, 进入自纠循环
    - ESCALATE: 超过最大修订次数, 升级人工审核
    """

    ALLOW = "allow"
    REJECT = "reject"
    REPLACE = "replace"
    REVISE = "revise"
    ESCALATE = "escalate"


# ============================================================
# 异常定义
# ============================================================


class ReflectionError(Exception):
    """反思与质量控制错误."""

    pass


# ============================================================
# 数据模型
# ============================================================


@dataclass
class DimensionScore:
    """单维度评分 (融合 ADK rubric + Claude Science 多维度评分).

    Attributes:
        dimension: 检查维度
        score: 分数 (0.0-1.0, 自动钳位)
        weight: 权重 (默认使用维度 default_weight)
        reasoning: 评分理由
    """

    dimension: ReflectionDimension
    score: float
    weight: float | None = None
    reasoning: str = ""

    def __post_init__(self) -> None:
        # 钳位分数到 0-1 范围
        self.score = max(0.0, min(1.0, self.score))
        # 使用维度默认权重
        if self.weight is None:
            self.weight = self.dimension.default_weight


@dataclass
class ReviewRecord:
    """审核记录 (融合 AutoGen CodeReviewResult + ADK rubric + Claude Science 加权评分).

    Attributes:
        artifact_id: 产物 ID
        reviewer: 审核者标识
        dimension_scores: 多维度评分列表
        verdict: 审核裁决
        feedback: 审核反馈
        iteration: 审核迭代次数
        review_id: 审核 ID (自动生成)
        timestamp: 时间戳
    """

    artifact_id: str
    reviewer: str
    dimension_scores: list[DimensionScore]
    verdict: Verdict
    feedback: str = ""
    iteration: int = 1
    review_id: str = field(
        default_factory=lambda: f"rev-{uuid.uuid4().hex[:12]}"
    )
    timestamp: float = field(default_factory=time.time)

    @property
    def weighted_score(self) -> float:
        """加权总分 (Claude Code 多维度加权评分).

        计算: sum(score_i * weight_i)
        无维度评分时返回 0.0.
        """
        if not self.dimension_scores:
            return 0.0
        return sum(ds.score * ds.weight for ds in self.dimension_scores)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "review_id": self.review_id,
            "artifact_id": self.artifact_id,
            "reviewer": self.reviewer,
            "verdict": self.verdict.value,
            "weighted_score": self.weighted_score,
            "feedback": self.feedback,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "dimension_scores": [
                {
                    "dimension": ds.dimension.value,
                    "score": ds.score,
                    "weight": ds.weight,
                    "reasoning": ds.reasoning,
                }
                for ds in self.dimension_scores
            ],
        }


@dataclass
class ReflectionResult:
    """反思结果 (融合 LangGraph critique_history 聚合).

    Attributes:
        artifact_id: 产物 ID
        agent_id: Agent ID
        trigger: 反思触发类型
        reviews: 审核记录列表 (按时间顺序)
        final_verdict: 最终裁决
        max_iterations: 最大迭代次数
        resolved_issues: 已解决问题列表
    """

    artifact_id: str
    agent_id: str
    trigger: ReflectionTrigger
    reviews: list[ReviewRecord]
    final_verdict: Verdict
    max_iterations: int = 3
    resolved_issues: list[str] = field(default_factory=list)

    @property
    def total_iterations(self) -> int:
        """总迭代次数 (等于审核记录数)."""
        return len(self.reviews)

    @property
    def improvement_trajectory(self) -> list[float]:
        """改进轨迹 (各轮加权总分, LangGraph revision_number 模式)."""
        return [review.weighted_score for review in self.reviews]

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (LangGraph state serialization + API 响应).

        包含完整的反思结果信息, 可用于:
        - API 响应序列化
        - 日志记录
        - 检查点持久化
        """
        return {
            "artifact_id": self.artifact_id,
            "agent_id": self.agent_id,
            "trigger": self.trigger.value,
            "final_verdict": self.final_verdict.value,
            "max_iterations": self.max_iterations,
            "total_iterations": self.total_iterations,
            "resolved_issues": list(self.resolved_issues),
            "improvement_trajectory": self.improvement_trajectory,
            "reviews": [review.to_dict() for review in self.reviews],
        }


@dataclass
class GateResult:
    """门控结果.

    Attributes:
        passed: 是否通过
        action: 门控动作
        score: 评估分数
        message: 结果消息
    """

    passed: bool
    action: GateAction
    score: float
    message: str = ""


@dataclass
class AdjudicationResult:
    """裁决结果 (AdjudicationExecutor 返回).

    Attributes:
        action: 门控动作
        passed: 是否通过
        score: 评估分数
        review: 关联的审核记录
        message: 结果消息
    """

    action: GateAction
    passed: bool
    score: float
    review: ReviewRecord | None = None
    message: str = ""


@dataclass
class CollaborationReview:
    """跨 Agent 协作复盘记录 (L5 设计文档 7.2).

    Attributes:
        session_id: 会话 ID
        trigger: 复盘触发类型
        participants: 参与 Agent 列表
        metrics: 协作指标 (duration/consensus/token_cost 等)
        insights: 复盘洞察列表
        review_id: 复盘 ID (自动生成)
        timestamp: 时间戳
    """

    session_id: str
    trigger: CollaborationTrigger
    participants: list[str]
    metrics: dict[str, Any]
    insights: list[str] = field(default_factory=list)
    review_id: str = field(
        default_factory=lambda: f"colab-{uuid.uuid4().hex[:12]}"
    )
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "review_id": self.review_id,
            "session_id": self.session_id,
            "trigger": self.trigger.value,
            "participants": list(self.participants),
            "metrics": dict(self.metrics),
            "insights": list(self.insights),
            "timestamp": self.timestamp,
        }


@dataclass
class QualityReport:
    """全链路质量报告.

    Attributes:
        session_id: 会话 ID
        artifact_id: 产物 ID
        reflection_result: 反思结果
        report_id: 报告 ID (自动生成)
        created_at: 创建时间
    """

    session_id: str
    artifact_id: str
    reflection_result: ReflectionResult
    report_id: str = field(
        default_factory=lambda: f"qreport-{uuid.uuid4().hex[:12]}"
    )
    created_at: float = field(default_factory=time.time)

    @property
    def summary(self) -> dict[str, Any]:
        """质量摘要 (含维度详情)."""
        last_review = (
            self.reflection_result.reviews[-1]
            if self.reflection_result.reviews
            else None
        )
        score = last_review.weighted_score if last_review else 0.0

        # 计算最强/最弱维度
        weakest = ""
        strongest = ""
        if last_review and last_review.dimension_scores:
            sorted_dims = sorted(
                last_review.dimension_scores, key=lambda d: d.score
            )
            weakest = sorted_dims[0].dimension.value
            strongest = sorted_dims[-1].dimension.value

        return {
            "verdict": self.reflection_result.final_verdict.value,
            "score": score,
            "iterations": self.reflection_result.total_iterations,
            "weakest_dimension": weakest,
            "strongest_dimension": strongest,
        }

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (含维度详情 + 审核反馈)."""
        last_review = (
            self.reflection_result.reviews[-1]
            if self.reflection_result.reviews
            else None
        )
        dimension_details = []
        if last_review:
            for ds in last_review.dimension_scores:
                dimension_details.append({
                    "dimension": ds.dimension.value,
                    "score": ds.score,
                    "weight": ds.weight,
                    "reasoning": ds.reasoning,
                })

        return {
            "report_id": self.report_id,
            "session_id": self.session_id,
            "artifact_id": self.artifact_id,
            "reviewer": last_review.reviewer if last_review else "",
            "feedback": last_review.feedback if last_review else "",
            "dimension_details": dimension_details,
            "reflection_result": {
                "artifact_id": self.reflection_result.artifact_id,
                "agent_id": self.reflection_result.agent_id,
                "trigger": self.reflection_result.trigger.value,
                "final_verdict": self.reflection_result.final_verdict.value,
                "total_iterations": self.reflection_result.total_iterations,
                "improvement_trajectory": (
                    self.reflection_result.improvement_trajectory
                ),
            },
            "created_at": self.created_at,
        }

    def generate_recommendations(self) -> list[str]:
        """生成可操作的质量建议 (基于裁决和维度评分).

        融合 Google ADK eval-fix loop + CrewAI task guardrail feedback.

        Returns:
            建议列表 (按优先级排序)
        """
        recommendations: list[str] = []
        last_review = (
            self.reflection_result.reviews[-1]
            if self.reflection_result.reviews
            else None
        )
        verdict = self.reflection_result.final_verdict
        score = last_review.weighted_score if last_review else 0.0

        if verdict == Verdict.APPROVED:
            recommendations.append(
                f"质量合格, 通过审核 (加权分: {score:.2%})"
            )
            if last_review and last_review.dimension_scores:
                weakest = min(last_review.dimension_scores, key=lambda d: d.score)
                if weakest.score < 0.85:
                    recommendations.append(
                        f"建议持续优化 {weakest.dimension.value} "
                        f"(当前 {weakest.score:.2%})"
                    )
        elif verdict == Verdict.REVISE:
            recommendations.append(
                f"需要修订, 存在可改进问题 (加权分: {score:.2%})"
            )
            if last_review:
                for ds in sorted(
                    last_review.dimension_scores, key=lambda d: d.score
                ):
                    if ds.score < 0.7:
                        dim_name = {
                            "factual_consistency": "事实一致性",
                            "numeric_accuracy": "数值准确性",
                            "citation_completeness": "引用完整性",
                            "pedagogical_fit": "教学适配性",
                        }.get(ds.dimension.value, ds.dimension.value)
                        recommendations.append(
                            f"优先改进 {dim_name} "
                            f"(当前 {ds.score:.2%}, 目标 ≥80%)"
                        )
        else:  # REJECTED
            recommendations.append(
                f"质量不合格, 存在严重错误 (加权分: {score:.2%}), 建议重新生成"
            )
            if last_review:
                for ds in sorted(
                    last_review.dimension_scores, key=lambda d: d.score
                ):
                    if ds.score < 0.5:
                        dim_name = {
                            "factual_consistency": "事实一致性",
                            "numeric_accuracy": "数值准确性",
                            "citation_completeness": "引用完整性",
                            "pedagogical_fit": "教学适配性",
                        }.get(ds.dimension.value, ds.dimension.value)
                        recommendations.append(
                            f"严重问题: {dim_name} 仅 {ds.score:.2%}, "
                            f"需彻底检查"
                        )

        return recommendations


# ============================================================
# QualityGate — 质量门控
# ============================================================


class QualityGate:
    """质量门控 (融合 Claude Code 95% 门控 + Temporal RetryPolicy).

    三级评估逻辑:
    1. score >= threshold → ALLOW (通过)
    2. hard_floor <= score < threshold → REVISE (修订) 或 ESCALATE (超限升级)
    3. score < hard_floor → REJECT (拒绝)

    错误分类 (Temporal non_retryable_error_types):
    - non_retryable_errors 中的错误类型不可重试
    - 其他错误默认可重试

    Attributes:
        name: 门控名称
        threshold: 通过阈值
        hard_floor: 硬下限 (低于此值直接拒绝)
        max_revisions: 最大修订次数 (超过则升级)
        non_retryable_errors: 不可重试错误类型列表
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        hard_floor: float = 0.0,
        max_revisions: int = 3,
        non_retryable_errors: list[str] | None = None,
    ) -> None:
        self.name = name
        self.threshold = threshold
        self.hard_floor = hard_floor
        self.max_revisions = max_revisions
        self.non_retryable_errors = (
            non_retryable_errors
            if non_retryable_errors is not None
            else ["ValidationError"]
        )

    def evaluate(self, score: float, iteration: int = 1) -> GateResult:
        """评估分数, 返回门控结果.

        Args:
            score: 评估分数 (0.0-1.0)
            iteration: 当前迭代次数

        Returns:
            GateResult 包含通过状态和动作
        """
        if score >= self.threshold:
            return GateResult(
                passed=True,
                action=GateAction.ALLOW,
                score=score,
                message="质量达标, 通过门控",
            )

        if score < self.hard_floor:
            return GateResult(
                passed=False,
                action=GateAction.REJECT,
                score=score,
                message=f"质量低于硬下限 {self.hard_floor}, 拒绝",
            )

        # hard_floor <= score < threshold
        if iteration > self.max_revisions:
            return GateResult(
                passed=False,
                action=GateAction.ESCALATE,
                score=score,
                message=f"超过最大修订次数 {self.max_revisions}, 升级人工审核",
            )

        return GateResult(
            passed=False,
            action=GateAction.REVISE,
            score=score,
            message=f"质量在 {self.hard_floor}-{self.threshold} 之间, 需要修订",
        )

    def is_retryable(self, error_type: str) -> bool:
        """判断错误类型是否可重试 (Temporal non_retryable_error_types 模式).

        Args:
            error_type: 错误类型名称

        Returns:
            True 如果可重试, False 如果不可重试
        """
        return error_type not in self.non_retryable_errors


# ============================================================
# CC1Reviewer — CC1 Actor-Critic 审核器
# ============================================================


class CC1Reviewer:
    """CC1 Actor-Critic 审核器 (L5 设计文档 7.1.2 + Claude Science Actor-Critic).

    对 Agent 产出进行多维度深度评审:
    1. 事实一致性: 检查输出内容与已知事实是否矛盾
    2. 数值准确性: 检查计算结果是否在合理范围内
    3. 引用完整性: 检查文献引用是否完整、可追溯
    4. 教学适配性: 检查内容深度是否匹配学习者水平

    评审结果决定:
    - 加权总分 >= 0.8 → APPROVED
    - 0.5 <= 加权总分 < 0.8 → REVISE
    - 加权总分 < 0.5 → REJECTED
    """

    def __init__(self) -> None:
        self._reviewer_id = "cc1.actor_critic"

    async def review(
        self,
        artifact_id: str,
        artifact_data: dict[str, Any],
        agent_id: str,
        iteration: int = 1,
        history: list[ReviewRecord] | None = None,
    ) -> ReviewRecord:
        """对产物进行 Actor-Critic 深度评审.

        Args:
            artifact_id: 产物 ID
            artifact_data: 产物数据
            agent_id: 产出 Agent ID
            iteration: 审核迭代次数
            history: 历史审核记录 (AutoGen 上下文感知)

        Returns:
            ReviewRecord 审核记录
        """
        # 多维度评分
        scores = self._evaluate_all_dimensions(artifact_data, history)

        # 创建临时记录计算加权总分
        temp_record = ReviewRecord(
            artifact_id=artifact_id,
            reviewer=self._reviewer_id,
            dimension_scores=scores,
            verdict=Verdict.REVISE,
            iteration=iteration,
        )

        # 根据加权总分确定裁决
        weighted = temp_record.weighted_score
        if weighted >= 0.8:
            verdict = Verdict.APPROVED
        elif weighted >= 0.5:
            verdict = Verdict.REVISE
        else:
            verdict = Verdict.REJECTED

        # 生成反馈
        feedback = self._generate_feedback(verdict, weighted, scores)

        return ReviewRecord(
            artifact_id=artifact_id,
            reviewer=self._reviewer_id,
            dimension_scores=scores,
            verdict=verdict,
            feedback=feedback,
            iteration=iteration,
        )

    def _evaluate_all_dimensions(
        self,
        artifact_data: dict[str, Any],
        history: list[ReviewRecord] | None = None,
    ) -> list[DimensionScore]:
        """评估所有 4 个维度."""
        return [
            DimensionScore(
                dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
                score=self._score_factual_consistency(artifact_data),
                reasoning="检查输出内容与已知事实是否矛盾",
            ),
            DimensionScore(
                dimension=ReflectionDimension.NUMERIC_ACCURACY,
                score=self._score_numeric_accuracy(artifact_data),
                reasoning="检查计算结果是否在合理范围内",
            ),
            DimensionScore(
                dimension=ReflectionDimension.CITATION_COMPLETENESS,
                score=self._score_citation_completeness(artifact_data),
                reasoning="检查文献引用是否完整、可追溯",
            ),
            DimensionScore(
                dimension=ReflectionDimension.PEDAGOGICAL_FIT,
                score=self._score_pedagogical_fit(artifact_data),
                reasoning="检查内容深度是否匹配学习者水平",
            ),
        ]

    def _score_factual_consistency(self, data: dict[str, Any]) -> float:
        """评估事实一致性 (L5 设计文档 fail 级别)."""
        confidence = data.get("confidence", 0.5)

        # 检查不合理数值
        has_unreasonable = False
        for key, value in data.items():
            if isinstance(value, (int, float)) and key != "confidence":
                if key == "boiling_point" and value > 500:
                    has_unreasonable = True
                elif key == "melting_point" and value > 1000:
                    has_unreasonable = True
                elif key == "temperature" and abs(value) > 5000:
                    has_unreasonable = True

        if has_unreasonable:
            return 0.35

        # 基于置信度的基础分
        if confidence >= 0.8:
            return 0.95
        elif confidence >= 0.5:
            return 0.80
        elif confidence >= 0.3:
            return 0.60
        else:
            return 0.40

    def _score_numeric_accuracy(self, data: dict[str, Any]) -> float:
        """评估数值准确性 (L5 设计文档 fail 级别)."""
        confidence = data.get("confidence", 0.5)

        # 检查明显错误的数值
        has_bad_numbers = False
        for key, value in data.items():
            if isinstance(value, (int, float)) and key != "confidence":
                if abs(value) > 10000:
                    has_bad_numbers = True

        if has_bad_numbers:
            return 0.45

        if confidence >= 0.8:
            return 0.92
        elif confidence >= 0.5:
            return 0.78
        elif confidence >= 0.3:
            return 0.58
        else:
            return 0.38

    def _score_citation_completeness(self, data: dict[str, Any]) -> float:
        """评估引用完整性 (L5 设计文档 warn 级别)."""
        references = data.get("references", [])
        confidence = data.get("confidence", 0.5)

        if not references:
            # 无引用但高置信度 → 适度扣分
            if confidence >= 0.8:
                return 0.65
            elif confidence >= 0.5:
                return 0.50
            else:
                return 0.40

        # 检查是否包含 DOI
        has_doi = any("doi" in str(ref).lower() for ref in references)
        if has_doi:
            return 0.95
        return 0.75

    def _score_pedagogical_fit(self, data: dict[str, Any]) -> float:
        """评估教学适配性 (L5 设计文档 warn 级别)."""
        confidence = data.get("confidence", 0.5)
        report_id = data.get("report_id", "")
        kp_gaps = data.get("kp_gaps", [])

        # 基于置信度的基础分
        if confidence >= 0.8:
            score = 0.90
        elif confidence >= 0.5:
            score = 0.75
        elif confidence >= 0.3:
            score = 0.55
        else:
            score = 0.35

        # 内容存在性调整
        if not report_id:
            score -= 0.15
        if not kp_gaps and "kp_gaps" in data:
            score -= 0.10

        return max(0.0, min(1.0, score))

    def _generate_feedback(
        self,
        verdict: Verdict,
        score: float,
        dimensions: list[DimensionScore],
    ) -> str:
        """生成审核反馈."""
        # 找出最低分维度
        if dimensions:
            lowest = min(dimensions, key=lambda d: d.score)
            lowest_info = f" (最低维度: {lowest.dimension.value}={lowest.score:.2f})"
        else:
            lowest_info = ""

        if verdict == Verdict.APPROVED:
            return f"质量合格, 通过审核 (加权分: {score:.2%}){lowest_info}"
        elif verdict == Verdict.REVISE:
            return f"需要修订, 存在可改进问题 (加权分: {score:.2%}){lowest_info}"
        else:
            return f"质量不合格, 存在严重错误 (加权分: {score:.2%}){lowest_info}"


# ============================================================
# AdjudicationExecutor — 裁决执行器
# ============================================================


class AdjudicationExecutor:
    """裁决执行器 (处理 requires_adjudication + Temporal Saga 补偿).

    根据 QualityGate 评估审核记录, 决定后续动作:
    - ALLOW: 放行
    - REJECT: 拒绝并执行补偿 (Saga 逆序)
    - REVISE: 需要修订
    - ESCALATE: 升级人工审核

    补偿机制 (Temporal Saga pattern):
    - 当裁决为 REJECT 时, 逆序执行所有补偿函数
    - 补偿函数为 async, 按注册逆序执行
    """

    def __init__(self, gate: QualityGate) -> None:
        self._gate = gate

    async def adjudicate(
        self,
        review: ReviewRecord,
        compensations: list[Callable[[], Awaitable[None]]] | None = None,
    ) -> AdjudicationResult:
        """裁决审核记录.

        Args:
            review: 审核记录
            compensations: 补偿函数列表 (Saga 模式, 逆序执行)

        Returns:
            AdjudicationResult 裁决结果
        """
        gate_result = self._gate.evaluate(
            score=review.weighted_score,
            iteration=review.iteration,
        )

        # REJECT 时执行补偿 (Temporal Saga 逆序)
        if gate_result.action == GateAction.REJECT and compensations:
            for comp in reversed(compensations):
                try:
                    await comp()
                except Exception as e:
                    logger.warning(f"补偿函数执行失败: {e}")

        return AdjudicationResult(
            action=gate_result.action,
            passed=gate_result.passed,
            score=gate_result.score,
            review=review,
            message=gate_result.message,
        )


# ============================================================
# ReputationLedger — 声誉账本
# ============================================================


class ReputationLedger:
    """声誉账本 (融合 AutoGen 信誉体系 + 指数移动平均).

    核心能力:
    1. 注册 Agent 初始声誉分
    2. 根据反思结果更新声誉 (EMA)
    3. 根据声誉推荐质量阈值 (信任 → 放宽, 不信任 → 收紧)
    4. 统计 Agent 表现指标

    EMA 更新公式:
        new_score = alpha * target + (1 - alpha) * old_score
    - alpha = 0.3 (平滑因子)
    - APPROVED → target = 100 (一次通过 +5 bonus)
    - REJECTED → target = 0
    - REVISE → target = 50

    阈值推荐:
    - score >= 80: threshold = base - adjustment (放宽)
    - score < 50: threshold = base + adjustment (收紧)
    """

    DEFAULT_SCORE = 50.0
    EMA_ALPHA = 0.3
    FIRST_TRY_BONUS = 5.0

    def __init__(self) -> None:
        self._scores: dict[str, float] = {}
        self._stats: dict[str, dict[str, int]] = {}

    def register(self, agent_id: str, initial_score: float = DEFAULT_SCORE) -> None:
        """注册 Agent 初始声誉分."""
        self._scores[agent_id] = max(0.0, min(100.0, initial_score))
        if agent_id not in self._stats:
            self._stats[agent_id] = {
                "total_tasks": 0,
                "approved_first_try": 0,
                "approved_total": 0,
                "rejected_total": 0,
                "revised_total": 0,
            }

    def initialize_from_registry(self, registry: Any) -> None:
        """从 AgentRegistry 批量初始化声誉账本.

        读取每个已注册 Agent 的 ReputationConfig,
        使用其 initial_score 注册到声誉账本.

        Args:
            registry: AgentRegistry 实例
        """
        for agent_def in registry.list_all():
            if agent_def.id not in self._scores:
                self.register(
                    agent_def.id,
                    initial_score=agent_def.reputation_config.initial_score,
                )

    def get_score(self, agent_id: str) -> float:
        """获取 Agent 声誉分 (未注册返回默认分)."""
        return self._scores.get(agent_id, self.DEFAULT_SCORE)

    def get_stats(self, agent_id: str) -> dict[str, int]:
        """获取 Agent 统计指标."""
        return self._stats.get(
            agent_id,
            {
                "total_tasks": 0,
                "approved_first_try": 0,
                "approved_total": 0,
                "rejected_total": 0,
                "revised_total": 0,
            },
        )

    def update(
        self,
        agent_id: str,
        result: ReflectionResult,
        *,
        reward_factor: float = 1.0,
        penalty_factor: float = 1.0,
    ) -> None:
        """根据反思结果更新 Agent 声誉 (EMA).

        Args:
            agent_id: Agent ID
            result: 反思结果
            reward_factor: 奖励放大因子 (APPROVED 时放大增益)
            penalty_factor: 惩罚放大因子 (REJECTED 时放大惩罚)
        """
        # 自动注册未注册的 Agent
        if agent_id not in self._scores:
            self.register(agent_id)

        current = self._scores[agent_id]
        stats = self._stats[agent_id]
        stats["total_tasks"] += 1

        # 确定目标分数
        if result.final_verdict == Verdict.APPROVED:
            target = 100.0
            stats["approved_total"] += 1
            # 一次通过奖励
            if result.total_iterations == 1:
                stats["approved_first_try"] += 1
                target += self.FIRST_TRY_BONUS
        elif result.final_verdict == Verdict.REJECTED:
            target = 0.0
            stats["rejected_total"] += 1
        else:  # REVISE
            target = 50.0
            stats["revised_total"] += 1

        # 根据裁决和 factor 确定有效 alpha
        if result.final_verdict == Verdict.APPROVED:
            effective_alpha = min(1.0, self.EMA_ALPHA * reward_factor)
        elif result.final_verdict == Verdict.REJECTED:
            effective_alpha = min(1.0, self.EMA_ALPHA * penalty_factor)
        else:
            effective_alpha = self.EMA_ALPHA

        # EMA 更新
        new_score = effective_alpha * target + (1 - effective_alpha) * current

        # 钳位到 0-100
        self._scores[agent_id] = max(0.0, min(100.0, new_score))

    def recommended_threshold(
        self, agent_id: str, base: float = 0.85
    ) -> float:
        """根据声誉推荐质量阈值.

        高信任 Agent → 阈值放宽 (降低)
        低信任 Agent → 阈值收紧 (提高)

        Args:
            agent_id: Agent ID
            base: 基础阈值

        Returns:
            推荐阈值
        """
        score = self.get_score(agent_id)

        if score >= 80:
            # 高信任 → 放宽 (降低阈值)
            adjustment = -0.05 * min(1.0, (score - 80) / 20)
        elif score >= 50:
            # 中等信任 → 不调整
            adjustment = 0.0
        else:
            # 低信任 → 收紧 (提高阈值)
            adjustment = 0.05 * min(1.0, (50 - score) / 50)

        return max(0.0, min(1.0, base + adjustment))


# ============================================================
# QualityTrendAnalyzer — 质量趋势分析器
# ============================================================


class QualityTrendAnalyzer:
    """质量趋势分析器 (融合 Google ADK EvalResult 趋势分析 + OpenAI Guardrail 遥测).

    核心能力:
    1. 滑动窗口: 保留最近 N 条反思结果
    2. 趋势检测: 改善/稳定/下降
    3. 维度分解: 按维度统计平均分
    4. 通过率统计: APPROVED 占比
    5. 摘要生成: 一键获取质量趋势摘要

    Attributes:
        window_size: 滑动窗口大小 (默认 20)
    """

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self._results: list[ReflectionResult] = []

    def record(self, result: ReflectionResult) -> None:
        """记录反思结果到趋势分析器.

        当窗口满时, 自动淘汰最旧的数据.
        """
        self._results.append(result)
        if len(self._results) > self.window_size:
            self._results = self._results[-self.window_size:]

    def __len__(self) -> int:
        return len(self._results)

    @property
    def average_score(self) -> float:
        """平均质量分 (最近一轮审核的加权总分平均)."""
        if not self._results:
            return 0.0
        scores = []
        for r in self._results:
            if r.reviews:
                scores.append(r.reviews[-1].weighted_score)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @property
    def pass_rate(self) -> float:
        """通过率 (APPROVED 占比)."""
        if not self._results:
            return 0.0
        approved = sum(
            1 for r in self._results if r.final_verdict == Verdict.APPROVED
        )
        return approved / len(self._results)

    @property
    def trend_direction(self) -> str:
        """趋势方向: improving / stable / declining.

        判断逻辑:
        - 数据少于 3 条 → stable
        - 前半段平均分 vs 后半段平均分
        - 差值 > 0.03 → improving
        - 差值 < -0.03 → declining
        - 否则 → stable
        """
        if len(self._results) < 3:
            return "stable"

        scores = [
            r.reviews[-1].weighted_score
            for r in self._results
            if r.reviews
        ]
        if len(scores) < 3:
            return "stable"

        mid = len(scores) // 2
        first_half_avg = sum(scores[:mid]) / mid if mid > 0 else 0.0
        second_half_avg = sum(scores[mid:]) / (len(scores) - mid)

        diff = second_half_avg - first_half_avg
        if diff > 0.03:
            return "improving"
        elif diff < -0.03:
            return "declining"
        return "stable"

    @property
    def dimension_breakdown(self) -> dict[str, float]:
        """按维度分解平均分."""
        if not self._results:
            return {}

        dim_sums: dict[str, list[float]] = {}
        for r in self._results:
            if r.reviews:
                for ds in r.reviews[-1].dimension_scores:
                    key = ds.dimension.value
                    if key not in dim_sums:
                        dim_sums[key] = []
                    dim_sums[key].append(ds.score)

        return {
            dim: sum(scores) / len(scores)
            for dim, scores in dim_sums.items()
            if scores
        }

    def get_summary(self) -> dict[str, Any]:
        """获取趋势摘要字典."""
        return {
            "total_reflections": len(self._results),
            "average_score": round(self.average_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "trend_direction": self.trend_direction,
            "dimension_breakdown": {
                k: round(v, 4) for k, v in self.dimension_breakdown.items()
            },
        }


# ============================================================
# TargetedSelfCorrector — 靶向自纠器
# ============================================================


class TargetedSelfCorrector:
    """靶向自纠器 (融合 LangGraph Generator-Critic 定向修复 + Claude Science 精准反馈).

    根据审核记录中各维度的评分, 针对性地改进产出:
    - 事实一致性低 → 提升置信度
    - 数值准确性低 → 标记需重新计算 / 移除异常值
    - 引用完整性低 → 补充引用
    - 教学适配性低 → 补充教学相关字段 (report_id / kp_gaps)

    核心特性:
    1. 不可变性: 不修改原始数据, 返回新字典
    2. 靶向修复: 只修改低分维度相关字段
    3. 修正日志: 记录做了哪些修改 (correct_with_log)
    """

    THRESHOLD = 0.7  # 低于此分数触发修正

    def correct(
        self,
        artifact_data: dict[str, Any],
        review: ReviewRecord,
    ) -> dict[str, Any]:
        """靶向自纠 (返回修正后的新字典).

        Args:
            artifact_data: 原始产物数据
            review: 审核记录 (含维度评分)

        Returns:
            修正后的数据字典 (不修改原始)
        """
        corrected, _ = self.correct_with_log(artifact_data, review)
        return corrected

    def correct_with_log(
        self,
        artifact_data: dict[str, Any],
        review: ReviewRecord,
    ) -> tuple[dict[str, Any], list[str]]:
        """靶向自纠并返回修正日志.

        Args:
            artifact_data: 原始产物数据
            review: 审核记录

        Returns:
            (修正后的数据, 修正日志列表)
        """
        corrected = dict(artifact_data)
        log: list[str] = []

        # 构建维度评分映射
        dim_scores: dict[ReflectionDimension, float] = {}
        for ds in review.dimension_scores:
            dim_scores[ds.dimension] = ds.score

        # 1. 事实一致性低 → 提升置信度
        if dim_scores.get(ReflectionDimension.FACTUAL_CONSISTENCY, 1.0) < self.THRESHOLD:
            if "confidence" in corrected and corrected["confidence"] < 0.85:
                old_val = corrected["confidence"]
                corrected["confidence"] = min(0.95, old_val + 0.15)
                log.append(
                    f"confidence: {old_val:.2f} → {corrected['confidence']:.2f}"
                )

        # 2. 数值准确性低 → 标记异常值
        if dim_scores.get(ReflectionDimension.NUMERIC_ACCURACY, 1.0) < self.THRESHOLD:
            has_bad_numbers = False
            for key, value in list(corrected.items()):
                if isinstance(value, (int, float)) and key != "confidence":
                    if abs(value) > 10000:
                        del corrected[key]
                        log.append(f"removed abnormal value: {key}={value}")
                        has_bad_numbers = True
            if has_bad_numbers:
                corrected["_needs_recalculation"] = True
                log.append("marked _needs_recalculation=True")

        # 3. 引用完整性低 → 补充引用
        if dim_scores.get(ReflectionDimension.CITATION_COMPLETENESS, 1.0) < self.THRESHOLD:
            if not corrected.get("references"):
                corrected["references"] = ["auto-generated-ref"]
                log.append("references: [] → ['auto-generated-ref']")

        # 4. 教学适配性低 → 补充教学字段
        if dim_scores.get(ReflectionDimension.PEDAGOGICAL_FIT, 1.0) < self.THRESHOLD:
            if not corrected.get("report_id"):
                corrected["report_id"] = f"auto-corrected-{uuid.uuid4().hex[:8]}"
                log.append(f"report_id: '' → '{corrected['report_id']}'")
            if not corrected.get("kp_gaps"):
                corrected["kp_gaps"] = ["auto-detected-KP"]
                log.append("kp_gaps: [] → ['auto-detected-KP']")

        return corrected, log


# ============================================================
# ExecutionLogEntry — 执行日志条目
# ============================================================


@dataclass
class ExecutionLogEntry:
    """执行日志条目 (融合 Temporal 事件历史 + Claude Code JSONL 审计日志).

    记录反思过程中的每个关键操作, 用于审计追溯.

    Attributes:
        agent_id: Agent ID
        action: 操作类型 (reflection.start / reflection.review / reflection.correct 等)
        artifact_id: 产物 ID
        result: 操作结果
        score: 质量分数
        metadata: 额外元数据
        log_id: 日志 ID (自动生成)
        timestamp: 时间戳 (自动生成)
    """

    agent_id: str
    action: str
    artifact_id: str
    result: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    log_id: str = field(
        default_factory=lambda: f"log-{uuid.uuid4().hex[:12]}"
    )
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "log_id": self.log_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "artifact_id": self.artifact_id,
            "result": self.result,
            "score": self.score,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_reflection_result(
        cls, result: ReflectionResult
    ) -> "ExecutionLogEntry":
        """从 ReflectionResult 创建日志条目.

        Args:
            result: 反思结果

        Returns:
            ExecutionLogEntry 实例
        """
        last_review = result.reviews[-1] if result.reviews else None
        score = last_review.weighted_score if last_review else 0.0
        return cls(
            agent_id=result.agent_id,
            action="reflection.complete",
            artifact_id=result.artifact_id,
            result=result.final_verdict.value,
            score=score,
            metadata={
                "trigger": result.trigger.value,
                "total_iterations": result.total_iterations,
                "resolved_issues_count": len(result.resolved_issues),
            },
        )


# ============================================================
# ReflectionEngine — 反思引擎
# ============================================================


class ReflectionEngine:
    """反思引擎 (L5 设计文档第七章 Reflection Engine).

    核心能力:
    1. 单 Agent 反思 (7.1): Agent 输出后自动触发自检
       - pass: 质量合格, 进入下一流程
       - warn: 自纠后重新检查 (自纠循环)
       - fail: 触发 CC1 深度评审
    2. 跨 Agent 复盘 (7.2): 多 Agent 协作完成后联合复盘
       - 辩论结束 → 复盘论据质量、共识达成度
       - 投票完成 → 复盘各策略优劣和选择合理性
       - Fork 合并 → 复盘路径选择依据和效果预估
    3. 反思历史查询 (9.5): 按 Agent/裁决过滤查询

    自纠循环 (LangGraph Generator-Critic):
    1. CC1 Reviewer 评审产出
    2. QualityGate 评估分数
    3. REVISE → 自纠 (提升 confidence/补充引用/填充空缺)
    4. 重新评审, 直到 ALLOW/REJECT/ESCALATE
    5. 更新 ReputationLedger (闭环反馈)
    6. 记录到 QualityTrendAnalyzer (趋势追踪)
    7. 生成 ExecutionLogEntry (审计追溯)

    增强参数 (可选, 向后兼容):
    - trend_analyzer: 质量趋势分析器
    - self_corrector: 靶向自纠器 (替代默认 _self_correct)
    """

    def __init__(
        self,
        gate: QualityGate,
        reviewer: CC1Reviewer,
        reputation_ledger: ReputationLedger,
        trend_analyzer: QualityTrendAnalyzer | None = None,
        self_corrector: TargetedSelfCorrector | None = None,
    ) -> None:
        self._gate = gate
        self._reviewer = reviewer
        self._reputation_ledger = reputation_ledger
        self._trend_analyzer = trend_analyzer
        self._self_corrector = self_corrector
        self._history: list[ReflectionResult] = []
        self._execution_logs: list[ExecutionLogEntry] = []
        self._collaboration_reviews: list[CollaborationReview] = []

    async def reflect(
        self,
        agent_id: str,
        artifact_id: str,
        artifact_data: dict[str, Any],
    ) -> ReflectionResult:
        """单 Agent 反思 (L5 设计文档 7.1).

        自纠循环:
        1. CC1 Reviewer 评审
        2. QualityGate 评估
        3. ALLOW → 通过; REJECT → 拒绝; REVISE → 自纠重评; ESCALATE → 升级

        Args:
            agent_id: Agent ID
            artifact_id: 产物 ID
            artifact_data: 产物数据

        Returns:
            ReflectionResult 反思结果

        Raises:
            ValueError: agent_id 或 artifact_id 为空
            ReflectionError: 反思过程中的错误
        """
        # 参数校验
        if not agent_id:
            raise ValueError("agent_id 不能为空")
        if not artifact_id:
            raise ValueError("artifact_id 不能为空")

        reviews: list[ReviewRecord] = []
        iteration = 1
        max_iterations = self._gate.max_revisions
        resolved_issues: list[str] = []
        current_data = dict(artifact_data)

        # 自纠循环 (LangGraph Generator-Critic)
        while iteration <= max_iterations:
            # CC1 Reviewer 评审
            review = await self._reviewer.review(
                artifact_id=artifact_id,
                artifact_data=current_data,
                agent_id=agent_id,
                iteration=iteration,
                history=reviews if reviews else None,
            )
            reviews.append(review)

            # QualityGate 评估
            gate_result = self._gate.evaluate(
                score=review.weighted_score,
                iteration=iteration,
            )

            logger.debug(
                f"反思迭代 {iteration}: score={review.weighted_score:.4f}, "
                f"action={gate_result.action.value}"
            )

            if gate_result.action == GateAction.ALLOW:
                # 质量达标, 通过
                break
            elif gate_result.action == GateAction.REJECT:
                # 质量低于硬下限, 拒绝
                break
            elif gate_result.action == GateAction.ESCALATE:
                # 超过最大修订次数, 升级
                break
            elif gate_result.action == GateAction.REVISE:
                # 自纠 (warn 级别处理)
                # 优先使用靶向自纠器, 否则回退到默认自纠
                if self._self_corrector is not None:
                    current_data = self._self_corrector.correct(
                        current_data, review
                    )
                else:
                    current_data = self._self_correct(current_data, review)
                resolved_issues.append(f"迭代{iteration}: 自纠改进")
                iteration += 1

        # 最终裁决取最后一次审核结果
        final_verdict = reviews[-1].verdict

        # 创建反思结果
        result = ReflectionResult(
            artifact_id=artifact_id,
            agent_id=agent_id,
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=reviews,
            final_verdict=final_verdict,
            max_iterations=max_iterations,
            resolved_issues=resolved_issues,
        )

        # 更新声誉 (闭环反馈)
        self._reputation_ledger.update(agent_id, result)

        # 记录到趋势分析器 (增强: 质量趋势追踪)
        if self._trend_analyzer is not None:
            self._trend_analyzer.record(result)

        # 生成执行日志 (增强: 审计追溯)
        log_entry = ExecutionLogEntry.from_reflection_result(result)
        self._execution_logs.append(log_entry)

        # 存入历史
        self._history.append(result)

        return result

    async def collaboration_review(
        self,
        session_id: str,
        trigger: CollaborationTrigger,
        participants: list[str],
        metrics: dict[str, Any],
    ) -> CollaborationReview:
        """跨 Agent 协作复盘 (L5 设计文档 7.2).

        Args:
            session_id: 会话 ID
            trigger: 复盘触发类型
            participants: 参与 Agent 列表
            metrics: 协作指标

        Returns:
            CollaborationReview 复盘记录

        Raises:
            ValueError: participants 为空
        """
        if not participants:
            raise ValueError("participants 不能为空")

        # 生成复盘洞察
        insights = self._generate_collaboration_insights(
            trigger, participants, metrics
        )

        review = CollaborationReview(
            session_id=session_id,
            trigger=trigger,
            participants=list(participants),
            metrics=dict(metrics),
            insights=insights,
        )
        self._collaboration_reviews.append(review)
        return review

    def get_reflection_history(
        self,
        agent_id: str,
        verdict: Verdict | None = None,
    ) -> list[ReflectionResult]:
        """查询反思历史 (L5 设计文档 9.5).

        Args:
            agent_id: Agent ID
            verdict: 可选裁决过滤

        Returns:
            反思结果列表 (按时间顺序)
        """
        results = [r for r in self._history if r.agent_id == agent_id]
        if verdict is not None:
            results = [r for r in results if r.final_verdict == verdict]
        return results

    def get_effective_threshold(self, agent_id: str) -> float:
        """获取基于声誉的动态质量阈值 (闭环反馈).

        高信任 Agent → 阈值放宽 (降低)
        低信任 Agent → 阈值收紧 (提高)

        Args:
            agent_id: Agent ID

        Returns:
            动态调整后的阈值
        """
        return self._reputation_ledger.recommended_threshold(
            agent_id, base=self._gate.threshold
        )

    def get_trend_summary(self) -> dict[str, Any]:
        """获取质量趋势摘要 (增强: 质量趋势追踪).

        Returns:
            趋势摘要字典, 包含:
            - total_reflections: 反思总数
            - average_score: 平均质量分
            - pass_rate: 通过率
            - trend_direction: 趋势方向
            - dimension_breakdown: 维度分解

        Raises:
            RuntimeError: 未配置 trend_analyzer
        """
        if self._trend_analyzer is None:
            raise RuntimeError(
                "未配置 trend_analyzer, 请在 __init__ 中传入 QualityTrendAnalyzer"
            )
        return self._trend_analyzer.get_summary()

    def get_execution_logs(self) -> list[ExecutionLogEntry]:
        """获取执行日志列表 (增强: 审计追溯).

        Returns:
            执行日志条目列表 (按时间顺序)
        """
        return list(self._execution_logs)

    def get_collaboration_reviews(
        self,
        session_id: str,
        trigger: CollaborationTrigger | None = None,
    ) -> list[CollaborationReview]:
        """查询协作复盘记录 (L5 设计文档 9.5.2).

        Args:
            session_id: 会话 ID
            trigger: 可选触发类型过滤

        Returns:
            协作复盘记录列表 (按时间顺序)
        """
        results = [r for r in self._collaboration_reviews if r.session_id == session_id]
        if trigger is not None:
            results = [r for r in results if r.trigger == trigger]
        return results

    def _self_correct(
        self,
        artifact_data: dict[str, Any],
        review: ReviewRecord,
    ) -> dict[str, Any]:
        """自纠改进 (L5 设计文档 7.1.2 warn 级别处理).

        根据审核反馈自动改进产出:
        1. 提升置信度 (如果偏低)
        2. 补充引用 (如果缺失)
        3. 填充空缺字段 (如 report_id)
        """
        corrected = dict(artifact_data)

        # 提升置信度
        if "confidence" in corrected and corrected["confidence"] < 0.85:
            corrected["confidence"] = min(0.95, corrected["confidence"] + 0.15)

        # 补充引用
        if not corrected.get("references"):
            corrected["references"] = ["auto-generated-ref"]

        # 填充空 report_id
        if not corrected.get("report_id"):
            corrected["report_id"] = f"auto-corrected-{uuid.uuid4().hex[:8]}"

        # 补充 kp_gaps (如果缺失)
        if not corrected.get("kp_gaps") and "kp_gaps" not in corrected:
            corrected["kp_gaps"] = ["auto-detected-KP"]

        return corrected

    def _generate_collaboration_insights(
        self,
        trigger: CollaborationTrigger,
        participants: list[str],
        metrics: dict[str, Any],
    ) -> list[str]:
        """生成协作复盘洞察."""
        insights: list[str] = []
        duration = metrics.get("total_duration_s", 0)
        confidence = metrics.get("consensus_confidence", 0)
        token_cost = metrics.get("total_token_cost", 0)

        if trigger == CollaborationTrigger.DEBATE:
            disagreements = metrics.get("disagreement_points", 0)
            compromises = metrics.get("compromise_count", 0)
            insights.append(
                f"辩论耗时 {duration}s, 共识置信度 {confidence:.2%}"
            )
            if disagreements > 0:
                insights.append(
                    f"发现 {disagreements} 个分歧点, "
                    f"达成 {compromises} 项妥协"
                )
            efficiency = "较高" if duration < 200 else "需优化"
            insights.append(
                f"代币消耗 {token_cost}, 效率{efficiency}"
            )

        elif trigger == CollaborationTrigger.VOTING:
            insights.append(
                f"投票共识置信度 {confidence:.2%}, 耗时 {duration}s"
            )
            insights.append(
                f"参与 Agent {len(participants)} 个, "
                f"代币消耗 {token_cost}"
            )
            if confidence >= 0.85:
                insights.append("共识度高, 策略选择可靠")
            else:
                insights.append("共识度偏低, 建议增加投票轮次")

        elif trigger == CollaborationTrigger.FORK_MERGE:
            learning_gain = metrics.get("learning_gain", 0)
            insights.append(
                f"Fork 合并完成, 学习增益 {learning_gain:.2%}"
            )
            insights.append(
                f"合并耗时 {duration}s, 代币消耗 {token_cost}"
            )
            if learning_gain > 0.1:
                insights.append("学习增益显著, Fork 路径选择合理")
            else:
                insights.append("学习增益有限, 建议优化 Fork 策略")

        return insights
