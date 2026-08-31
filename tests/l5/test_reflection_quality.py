"""反思与质量控制模块测试 — TDD 测试用例.

测试覆盖:
1. Verdict — 审核裁决枚举 (approved/revise/rejected)
2. ReflectionDimension — 反思检查维度 (4 维度: 事实一致性/数值准确性/引用完整性/教学适配性)
3. DimensionScore — 单维度评分 (分数 + 权重 + 理由)
4. ReviewRecord — 审核记录 (多维度评分 + 加权总分 + 裁决 + 反馈)
5. ReflectionResult — 反思结果 (审核历史 + 最终裁决 + 改进轨迹)
6. ReflectionTrigger — 反思触发类型 (single_agent/debate/voting/fork_merge)
7. QualityGate — 质量门控 (阈值 + 硬下限 + 修订上限 + 错误分类)
8. GateResult — 门控结果 (通过/动作/分数/消息)
9. GateAction — 门控动作 (allow/reject/replace/revise/escalate)
10. CC1Reviewer — CC1 Actor-Critic 审核器 (深度评审 + 多维度评分)
11. AdjudicationExecutor — 裁决执行器 (处理 requires_adjudication)
12. ReputationLedger — 声誉账本 (动态信任分 + 指数移动平均更新)
13. ReflectionEngine — 反思引擎 (单 Agent 反思 + 跨 Agent 复盘)
14. CollaborationReview — 跨 Agent 协作复盘记录
15. QualityReport — 全链路质量报告
16. 集成测试 — 与 OrchestrationEngine/ArtifactManager/SessionManager 联动
17. 错误处理 — 质量控制异常与恢复

融合世界先进方案:
- LangGraph: Generator-Critic 自纠正循环 + critique_history + revision_number
- OpenAI Agents SDK: Guardrail 三层护栏 (input/output/tool) + tripwire 绊线
- Google ADK: EvalCase 轨迹评估 + rubric 多维度评分 + eval-fix loop
- AutoGen: Reflection 消息协议 + Coder-Reviewer 双代理 + approved 裁决
- CrewAI: Task guardrail + Tuple[bool, Any] 验证 + feedback 反馈
- Claude Science: Actor-Critic 模式 + 多维度加权评分 + 95% 质量门控
- Temporal: Update Validator + RetryPolicy + Saga 补偿 + non_retryable_error_types
- L5 设计文档: 第七章 Reflection Engine (4 维度自检 + 3 级处理 + 跨 Agent 复盘)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from dy3_polaris.l5.reflection_quality import (
    AdjudicationExecutor,
    AdjudicationResult,
    CC1Reviewer,
    CollaborationReview,
    CollaborationTrigger,
    DimensionScore,
    GateAction,
    GateResult,
    QualityGate,
    QualityReport,
    ReflectionDimension,
    ReflectionEngine,
    ReflectionError,
    ReflectionResult,
    ReflectionTrigger,
    ReputationLedger,
    ReviewRecord,
    Verdict,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def quality_gate():
    """创建质量门控实例."""
    return QualityGate(
        name="default_gate",
        threshold=0.85,
        hard_floor=0.5,
        max_revisions=3,
    )


@pytest.fixture
def cc1_reviewer():
    """创建 CC1 Actor-Critic 审核器实例."""
    return CC1Reviewer()


@pytest.fixture
def reputation_ledger():
    """创建声誉账本实例."""
    return ReputationLedger()


@pytest.fixture
def reflection_engine(quality_gate, cc1_reviewer, reputation_ledger):
    """创建反思引擎实例."""
    return ReflectionEngine(
        gate=quality_gate,
        reviewer=cc1_reviewer,
        reputation_ledger=reputation_ledger,
    )


@pytest.fixture
def sample_dimension_scores():
    """样本维度评分."""
    return [
        DimensionScore(
            dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
            score=0.95,
            weight=0.3,
            reasoning="所有引用数据均与已知事实一致",
        ),
        DimensionScore(
            dimension=ReflectionDimension.NUMERIC_ACCURACY,
            score=0.90,
            weight=0.25,
            reasoning="计算结果在合理范围内, 单位正确",
        ),
        DimensionScore(
            dimension=ReflectionDimension.CITATION_COMPLETENESS,
            score=0.85,
            weight=0.20,
            reasoning="大部分引用完整, 1处缺少DOI",
        ),
        DimensionScore(
            dimension=ReflectionDimension.PEDAGOGICAL_FIT,
            score=0.88,
            weight=0.25,
            reasoning="内容深度匹配学习者掌握水平",
        ),
    ]


@pytest.fixture
def sample_review_record(sample_dimension_scores):
    """样本审核记录."""
    return ReviewRecord(
        artifact_id="art-001",
        reviewer="cc1.actor_critic",
        dimension_scores=sample_dimension_scores,
        verdict=Verdict.APPROVED,
        feedback="质量合格, 通过审核",
        iteration=1,
    )


# ============================================================
# 1. Verdict 测试
# ============================================================


class TestVerdict:
    """审核裁决枚举测试."""

    def test_all_verdicts_defined(self):
        """3 种裁决应全部定义."""
        assert Verdict.APPROVED == "approved"
        assert Verdict.REVISE == "revise"
        assert Verdict.REJECTED == "rejected"


# ============================================================
# 2. ReflectionDimension 测试
# ============================================================


class TestReflectionDimension:
    """反思检查维度测试 (L5 设计文档 7.1.1 四维度)."""

    def test_all_dimensions_defined(self):
        """4 种检查维度应全部定义."""
        assert ReflectionDimension.FACTUAL_CONSISTENCY == "factual_consistency"
        assert ReflectionDimension.NUMERIC_ACCURACY == "numeric_accuracy"
        assert ReflectionDimension.CITATION_COMPLETENESS == "citation_completeness"
        assert ReflectionDimension.PEDAGOGICAL_FIT == "pedagogical_fit"

    def test_dimension_severity(self):
        """维度应有严重级别 (L5 设计文档)."""
        assert ReflectionDimension.FACTUAL_CONSISTENCY.severity == "fail"
        assert ReflectionDimension.NUMERIC_ACCURACY.severity == "fail"
        assert ReflectionDimension.CITATION_COMPLETENESS.severity == "warn"
        assert ReflectionDimension.PEDAGOGICAL_FIT.severity == "warn"

    def test_dimension_default_weight(self):
        """维度应有默认权重."""
        assert ReflectionDimension.FACTUAL_CONSISTENCY.default_weight == 0.30
        assert ReflectionDimension.NUMERIC_ACCURACY.default_weight == 0.25
        assert ReflectionDimension.CITATION_COMPLETENESS.default_weight == 0.20
        assert ReflectionDimension.PEDAGOGICAL_FIT.default_weight == 0.25


# ============================================================
# 3. DimensionScore 测试
# ============================================================


class TestDimensionScore:
    """单维度评分测试."""

    def test_score_creation(self):
        """创建维度评分."""
        ds = DimensionScore(
            dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
            score=0.95,
            weight=0.30,
            reasoning="事实一致",
        )
        assert ds.dimension == ReflectionDimension.FACTUAL_CONSISTENCY
        assert ds.score == 0.95
        assert ds.weight == 0.30
        assert ds.reasoning == "事实一致"

    def test_score_with_default_weight(self):
        """维度评分可使用维度默认权重."""
        ds = DimensionScore(
            dimension=ReflectionDimension.NUMERIC_ACCURACY,
            score=0.88,
        )
        assert ds.weight == 0.25  # 使用维度默认权重

    def test_score_clamped_to_0_1(self):
        """分数应限制在 0-1 范围."""
        ds = DimensionScore(
            dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
            score=1.5,
        )
        assert ds.score == 1.0

        ds2 = DimensionScore(
            dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
            score=-0.3,
        )
        assert ds2.score == 0.0


# ============================================================
# 4. ReviewRecord 测试
# ============================================================


class TestReviewRecord:
    """审核记录测试 (融合 AutoGen CodeReviewResult + ADK rubric)."""

    def test_record_creation(self, sample_dimension_scores):
        """创建审核记录."""
        record = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1.actor_critic",
            dimension_scores=sample_dimension_scores,
            verdict=Verdict.APPROVED,
            feedback="质量合格",
            iteration=1,
        )
        assert record.artifact_id == "art-001"
        assert record.reviewer == "cc1.actor_critic"
        assert len(record.dimension_scores) == 4
        assert record.verdict == Verdict.APPROVED
        assert record.feedback == "质量合格"
        assert record.iteration == 1
        assert record.review_id.startswith("rev-")
        assert record.timestamp > 0

    def test_weighted_score_calculation(self, sample_dimension_scores):
        """加权总分应正确计算 (Claude Code 多维度加权评分)."""
        record = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1.actor_critic",
            dimension_scores=sample_dimension_scores,
            verdict=Verdict.APPROVED,
        )
        # 0.95*0.30 + 0.90*0.25 + 0.85*0.20 + 0.88*0.25 = 0.9025
        expected = 0.95 * 0.30 + 0.90 * 0.25 + 0.85 * 0.20 + 0.88 * 0.25
        assert abs(record.weighted_score - expected) < 0.001

    def test_record_with_empty_scores(self):
        """无维度评分时加权总分为 0."""
        record = ReviewRecord(
            artifact_id="art-001",
            reviewer="agent.self",
            dimension_scores=[],
            verdict=Verdict.REVISE,
        )
        assert record.weighted_score == 0.0

    def test_record_to_dict(self, sample_dimension_scores):
        """审核记录应可序列化为字典."""
        record = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1.actor_critic",
            dimension_scores=sample_dimension_scores,
            verdict=Verdict.APPROVED,
            feedback="pass",
        )
        d = record.to_dict()
        assert d["artifact_id"] == "art-001"
        assert d["verdict"] == "approved"
        assert d["weighted_score"] > 0
        assert len(d["dimension_scores"]) == 4
        assert "review_id" in d
        assert "timestamp" in d


# ============================================================
# 5. ReflectionResult 测试
# ============================================================


class TestReflectionResult:
    """反思结果测试 (融合 LangGraph critique_history 聚合)."""

    def test_result_creation(self, sample_review_record):
        """创建反思结果."""
        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.generation",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[sample_review_record],
            final_verdict=Verdict.APPROVED,
            max_iterations=3,
        )
        assert result.artifact_id == "art-001"
        assert result.agent_id == "agent.generation"
        assert result.trigger == ReflectionTrigger.SINGLE_AGENT
        assert len(result.reviews) == 1
        assert result.final_verdict == Verdict.APPROVED
        assert result.total_iterations == 1
        assert result.max_iterations == 3

    def test_improvement_trajectory(self, sample_dimension_scores):
        """改进轨迹应记录各轮评分 (LangGraph revision_number 模式)."""
        reviews = []
        for i in range(3):
            scores = [
                DimensionScore(
                    dimension=ds.dimension,
                    score=min(1.0, ds.score + i * 0.05),
                    weight=ds.weight,
                )
                for ds in sample_dimension_scores
            ]
            reviews.append(ReviewRecord(
                artifact_id="art-001",
                reviewer="cc1.actor_critic",
                dimension_scores=scores,
                verdict=Verdict.REVISE if i < 2 else Verdict.APPROVED,
                iteration=i + 1,
            ))

        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.generation",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=reviews,
            final_verdict=Verdict.APPROVED,
            max_iterations=3,
        )
        trajectory = result.improvement_trajectory
        assert len(trajectory) == 3
        assert trajectory[0] < trajectory[1] < trajectory[2]  # 评分递增

    def test_resolved_issues(self, sample_review_record):
        """反思结果应记录已解决问题."""
        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.generation",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[sample_review_record],
            final_verdict=Verdict.APPROVED,
            max_iterations=3,
            resolved_issues=["数值偏差已修正", "引用DOI已补充"],
        )
        assert len(result.resolved_issues) == 2
        assert "数值偏差已修正" in result.resolved_issues


# ============================================================
# 6. ReflectionTrigger / CollaborationTrigger 测试
# ============================================================


class TestReflectionTrigger:
    """反思触发类型测试 (L5 设计文档 7.2.1)."""

    def test_all_triggers_defined(self):
        """所有触发类型应定义."""
        assert ReflectionTrigger.SINGLE_AGENT == "single_agent"
        assert ReflectionTrigger.COLLABORATION == "collaboration"

    def test_collaboration_triggers(self):
        """跨 Agent 复盘触发类型 (L5 设计文档 7.2.1)."""
        assert CollaborationTrigger.DEBATE == "debate"
        assert CollaborationTrigger.VOTING == "voting"
        assert CollaborationTrigger.FORK_MERGE == "fork_merge"


# ============================================================
# 7. QualityGate 测试
# ============================================================


class TestQualityGate:
    """质量门控测试 (融合 Claude Code 95% 门控 + Temporal RetryPolicy)."""

    def test_gate_creation(self, quality_gate):
        """创建质量门控."""
        assert quality_gate.name == "default_gate"
        assert quality_gate.threshold == 0.85
        assert quality_gate.hard_floor == 0.5
        assert quality_gate.max_revisions == 3

    def test_gate_evaluate_pass(self, quality_gate):
        """分数高于阈值 → 通过."""
        result = quality_gate.evaluate(score=0.90, iteration=1)
        assert result.passed is True
        assert result.action == GateAction.ALLOW

    def test_gate_evaluate_revise(self, quality_gate):
        """分数在硬下限和阈值之间 → 修订."""
        result = quality_gate.evaluate(score=0.70, iteration=1)
        assert result.passed is False
        assert result.action == GateAction.REVISE

    def test_gate_evaluate_reject(self, quality_gate):
        """分数低于硬下限 → 拒绝."""
        result = quality_gate.evaluate(score=0.30, iteration=1)
        assert result.passed is False
        assert result.action == GateAction.REJECT

    def test_gate_evaluate_max_revisions_escalate(self, quality_gate):
        """超过最大修订次数 → 升级人工审核."""
        result = quality_gate.evaluate(score=0.70, iteration=4)
        assert result.passed is False
        assert result.action == GateAction.ESCALATE

    def test_gate_with_non_retryable_errors(self):
        """不可重试错误配置 (Temporal non_retryable_error_types)."""
        gate = QualityGate(
            name="strict_gate",
            threshold=0.95,
            hard_floor=0.6,
            non_retryable_errors=["ValidationError", "SafetyError"],
        )
        assert "ValidationError" in gate.non_retryable_errors
        assert "SafetyError" in gate.non_retryable_errors

    def test_gate_is_retryable(self, quality_gate):
        """判断错误是否可重试 (Temporal 模式)."""
        assert quality_gate.is_retryable("TransientError") is True
        assert quality_gate.is_retryable("ValidationError") is False


# ============================================================
# 8. GateResult / GateAction 测试
# ============================================================


class TestGateResult:
    """门控结果测试."""

    def test_all_actions_defined(self):
        """5 种门控动作应全部定义."""
        assert GateAction.ALLOW == "allow"
        assert GateAction.REJECT == "reject"
        assert GateAction.REPLACE == "replace"
        assert GateAction.REVISE == "revise"
        assert GateAction.ESCALATE == "escalate"

    def test_result_creation(self):
        """创建门控结果."""
        result = GateResult(
            passed=True,
            action=GateAction.ALLOW,
            score=0.95,
            message="质量合格",
        )
        assert result.passed is True
        assert result.action == GateAction.ALLOW
        assert result.score == 0.95
        assert result.message == "质量合格"


# ============================================================
# 9. CC1Reviewer 测试
# ============================================================


class TestCC1Reviewer:
    """CC1 Actor-Critic 审核器测试 (L5 设计文档 7.1.2 + Claude Science Actor-Critic)."""

    def test_reviewer_creation(self, cc1_reviewer):
        """创建 CC1 审核器."""
        assert cc1_reviewer is not None

    @pytest.mark.asyncio
    async def test_review_approve(self, cc1_reviewer):
        """高质量产出 → APPROVED."""
        artifact_data = {
            "report_id": "rpt-001",
            "kp_gaps": ["KP-12"],
            "confidence": 0.92,
            "references": ["doi:10.1xxx", "NIST-WebBook"],
        }
        record = await cc1_reviewer.review(
            artifact_id="art-001",
            artifact_data=artifact_data,
            agent_id="agent.generation",
            iteration=1,
        )
        assert record.verdict == Verdict.APPROVED
        assert record.weighted_score > 0.8
        assert len(record.dimension_scores) == 4

    @pytest.mark.asyncio
    async def test_review_reject_low_quality(self, cc1_reviewer):
        """低质量产出 → REVISE 或 REJECTED."""
        artifact_data = {
            "report_id": "",
            "kp_gaps": [],
            "confidence": 0.1,
            "references": [],
        }
        record = await cc1_reviewer.review(
            artifact_id="art-002",
            artifact_data=artifact_data,
            agent_id="agent.generation",
            iteration=1,
        )
        assert record.verdict in (Verdict.REVISE, Verdict.REJECTED)
        assert record.weighted_score < 0.7

    @pytest.mark.asyncio
    async def test_review_with_history(self, cc1_reviewer, sample_review_record):
        """带历史审核记录的审核 (AutoGen 上下文感知)."""
        artifact_data = {"report_id": "rpt-003", "confidence": 0.85}
        record = await cc1_reviewer.review(
            artifact_id="art-003",
            artifact_data=artifact_data,
            agent_id="agent.generation",
            iteration=2,
            history=[sample_review_record],
        )
        assert record.iteration == 2
        assert len(record.dimension_scores) == 4

    @pytest.mark.asyncio
    async def test_review_factual_inconsistency(self, cc1_reviewer):
        """事实不一致 → fail 级别 (L5 设计文档)."""
        artifact_data = {
            "report_id": "rpt-004",
            "boiling_point": 999,  # 不合理数值
            "confidence": 0.5,
        }
        record = await cc1_reviewer.review(
            artifact_id="art-004",
            artifact_data=artifact_data,
            agent_id="agent.generation",
            iteration=1,
        )
        # 事实一致性维度应低分
        factual_score = next(
            ds.score for ds in record.dimension_scores
            if ds.dimension == ReflectionDimension.FACTUAL_CONSISTENCY
        )
        assert factual_score < 0.7


# ============================================================
# 10. AdjudicationExecutor 测试
# ============================================================


class TestAdjudicationExecutor:
    """裁决执行器测试 (处理 requires_adjudication)."""

    @pytest.mark.asyncio
    async def test_adjudicate_approve(self, quality_gate):
        """高质量 → 批准."""
        executor = AdjudicationExecutor(gate=quality_gate)
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1.actor_critic",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.95, weight=1.0),
            ],
            verdict=Verdict.APPROVED,
        )
        result = await executor.adjudicate(review)
        assert result.action == GateAction.ALLOW
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_adjudicate_reject(self, quality_gate):
        """低质量 → 拒绝 + 补偿."""
        compensation_called = False

        async def compensation():
            nonlocal compensation_called
            compensation_called = True

        executor = AdjudicationExecutor(gate=quality_gate)
        review = ReviewRecord(
            artifact_id="art-002",
            reviewer="cc1.actor_critic",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.2, weight=1.0),
            ],
            verdict=Verdict.REJECTED,
        )
        result = await executor.adjudicate(review, compensations=[compensation])
        assert result.action == GateAction.REJECT
        assert result.passed is False
        assert compensation_called is True  # 补偿被逆序执行

    @pytest.mark.asyncio
    async def test_adjudicate_escalate(self, quality_gate):
        """超过最大修订 → 升级人工."""
        executor = AdjudicationExecutor(gate=quality_gate)
        review = ReviewRecord(
            artifact_id="art-003",
            reviewer="cc1.actor_critic",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.70, weight=1.0),
            ],
            verdict=Verdict.REVISE,
            iteration=5,  # 超过 max_revisions=3
        )
        result = await executor.adjudicate(review)
        assert result.action == GateAction.ESCALATE
        assert result.passed is False


# ============================================================
# 11. ReputationLedger 测试
# ============================================================


class TestReputationLedger:
    """声誉账本测试 (融合 AutoGen 信誉体系 + 指数移动平均)."""

    def test_ledger_creation(self, reputation_ledger):
        """创建声誉账本."""
        assert reputation_ledger is not None

    def test_register_agent(self, reputation_ledger):
        """注册 Agent 初始声誉."""
        reputation_ledger.register("agent.generation", initial_score=85.0)
        score = reputation_ledger.get_score("agent.generation")
        assert score == 85.0

    def test_get_unregistered_returns_default(self, reputation_ledger):
        """未注册 Agent 返回默认分."""
        score = reputation_ledger.get_score("agent.unknown")
        assert score == 50.0  # 默认中性分

    def test_update_on_success(self, reputation_ledger):
        """成功 → 声誉提升."""
        reputation_ledger.register("agent.generation", initial_score=80.0)
        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.generation",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[ReviewRecord(
                artifact_id="art-001",
                reviewer="cc1.actor_critic",
                dimension_scores=[
                    DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.95, weight=1.0),
                ],
                verdict=Verdict.APPROVED,
                iteration=1,
            )],
            final_verdict=Verdict.APPROVED,
            max_iterations=3,
        )
        reputation_ledger.update("agent.generation", result)
        new_score = reputation_ledger.get_score("agent.generation")
        assert new_score > 80.0  # 声誉提升

    def test_update_on_failure(self, reputation_ledger):
        """失败 → 声誉下降."""
        reputation_ledger.register("agent.generation", initial_score=80.0)
        result = ReflectionResult(
            artifact_id="art-002",
            agent_id="agent.generation",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[ReviewRecord(
                artifact_id="art-002",
                reviewer="cc1.actor_critic",
                dimension_scores=[
                    DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.3, weight=1.0),
                ],
                verdict=Verdict.REJECTED,
                iteration=3,
            )],
            final_verdict=Verdict.REJECTED,
            max_iterations=3,
        )
        reputation_ledger.update("agent.generation", result)
        new_score = reputation_ledger.get_score("agent.generation")
        assert new_score < 80.0  # 声誉下降

    def test_first_try_bonus(self, reputation_ledger):
        """一次通过 → 额外奖励."""
        reputation_ledger.register("agent.generation", initial_score=80.0)
        result = ReflectionResult(
            artifact_id="art-003",
            agent_id="agent.generation",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[ReviewRecord(
                artifact_id="art-003",
                reviewer="cc1.actor_critic",
                dimension_scores=[
                    DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.95, weight=1.0),
                ],
                verdict=Verdict.APPROVED,
                iteration=1,
            )],
            final_verdict=Verdict.APPROVED,
            max_iterations=3,
        )
        reputation_ledger.update("agent.generation", result)
        stats = reputation_ledger.get_stats("agent.generation")
        assert stats["approved_first_try"] == 1
        assert stats["total_tasks"] == 1

    def test_recommended_threshold(self, reputation_ledger):
        """高信任 Agent → 阈值放宽; 低信任 → 阈值收紧."""
        reputation_ledger.register("agent.trusted", initial_score=95.0)
        reputation_ledger.register("agent.untrusted", initial_score=20.0)

        trusted_threshold = reputation_ledger.recommended_threshold("agent.trusted", base=0.85)
        untrusted_threshold = reputation_ledger.recommended_threshold("agent.untrusted", base=0.85)

        assert trusted_threshold <= 0.85  # 信任 → 放宽
        assert untrusted_threshold >= 0.85  # 不信任 → 收紧

    def test_score_bounds(self, reputation_ledger):
        """声誉分应在 0-100 范围."""
        reputation_ledger.register("agent.test", initial_score=99.0)
        # 多次成功不应超过 100
        for _ in range(10):
            result = ReflectionResult(
                artifact_id="art-x",
                agent_id="agent.test",
                trigger=ReflectionTrigger.SINGLE_AGENT,
                reviews=[ReviewRecord(
                    artifact_id="art-x",
                    reviewer="cc1",
                    dimension_scores=[
                        DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.99, weight=1.0),
                    ],
                    verdict=Verdict.APPROVED,
                    iteration=1,
                )],
                final_verdict=Verdict.APPROVED,
                max_iterations=3,
            )
            reputation_ledger.update("agent.test", result)
        assert reputation_ledger.get_score("agent.test") <= 100.0


# ============================================================
# 12. ReflectionEngine 测试
# ============================================================


class TestReflectionEngine:
    """反思引擎测试 (L5 设计文档第七章 Reflection Engine)."""

    @pytest.mark.asyncio
    async def test_single_agent_reflection_pass(self, reflection_engine):
        """单 Agent 反思 → pass (L5 设计文档 7.1)."""
        result = await reflection_engine.reflect(
            agent_id="agent.generation",
            artifact_id="art-001",
            artifact_data={"report_id": "rpt-001", "confidence": 0.92},
        )
        assert result.trigger == ReflectionTrigger.SINGLE_AGENT
        assert result.final_verdict == Verdict.APPROVED
        assert result.total_iterations >= 1

    @pytest.mark.asyncio
    async def test_single_agent_reflection_with_correction(self, reflection_engine):
        """单 Agent 反思 → warn → 自纠 → pass (L5 设计文档 7.1.2)."""
        result = await reflection_engine.reflect(
            agent_id="agent.generation",
            artifact_id="art-002",
            artifact_data={"report_id": "rpt-002", "confidence": 0.75},
        )
        assert result.total_iterations >= 1
        assert result.final_verdict in (Verdict.APPROVED, Verdict.REVISE)

    @pytest.mark.asyncio
    async def test_single_agent_reflection_fail(self, reflection_engine):
        """单 Agent 反思 → fail → 触发 CC1 (L5 设计文档 7.1.2)."""
        result = await reflection_engine.reflect(
            agent_id="agent.generation",
            artifact_id="art-003",
            artifact_data={"report_id": "", "confidence": 0.1},
        )
        assert result.final_verdict in (Verdict.REVISE, Verdict.REJECTED)

    @pytest.mark.asyncio
    async def test_reflection_updates_reputation(self, reflection_engine, reputation_ledger):
        """反思结果应更新声誉 (闭环反馈)."""
        reputation_ledger.register("agent.generation", initial_score=80.0)
        await reflection_engine.reflect(
            agent_id="agent.generation",
            artifact_id="art-004",
            artifact_data={"report_id": "rpt-004", "confidence": 0.95},
        )
        score = reputation_ledger.get_score("agent.generation")
        assert score != 80.0  # 声誉已更新

    @pytest.mark.asyncio
    async def test_collaboration_review_debate(self, reflection_engine):
        """跨 Agent 复盘 → 辩论触发 (L5 设计文档 7.2.1)."""
        review = await reflection_engine.collaboration_review(
            session_id="sess-001",
            trigger=CollaborationTrigger.DEBATE,
            participants=["agent.generation", "agent.review"],
            metrics={
                "total_duration_s": 187,
                "consensus_confidence": 0.82,
                "disagreement_points": 3,
                "total_token_cost": 28500,
            },
        )
        assert review.session_id == "sess-001"
        assert review.trigger == CollaborationTrigger.DEBATE
        assert len(review.participants) == 2
        assert review.metrics["consensus_confidence"] == 0.82
        assert len(review.insights) > 0

    @pytest.mark.asyncio
    async def test_collaboration_review_voting(self, reflection_engine):
        """跨 Agent 复盘 → 投票触发 (L5 设计文档 7.2.1)."""
        review = await reflection_engine.collaboration_review(
            session_id="sess-002",
            trigger=CollaborationTrigger.VOTING,
            participants=["agent.a", "agent.b", "agent.c"],
            metrics={
                "total_duration_s": 120,
                "consensus_confidence": 0.90,
                "total_token_cost": 15000,
            },
        )
        assert review.trigger == CollaborationTrigger.VOTING
        assert len(review.participants) == 3

    @pytest.mark.asyncio
    async def test_collaboration_review_fork_merge(self, reflection_engine):
        """跨 Agent 复盘 → Fork 合并触发 (L5 设计文档 7.2.1)."""
        review = await reflection_engine.collaboration_review(
            session_id="sess-003",
            trigger=CollaborationTrigger.FORK_MERGE,
            participants=["agent.generation"],
            metrics={
                "total_duration_s": 300,
                "learning_gain": 0.15,
                "total_token_cost": 50000,
            },
        )
        assert review.trigger == CollaborationTrigger.FORK_MERGE

    @pytest.mark.asyncio
    async def test_get_reflection_history(self, reflection_engine):
        """查询反思历史 (L5 设计文档 9.5)."""
        # 产生几条反思记录
        for i in range(3):
            await reflection_engine.reflect(
                agent_id="agent.generation",
                artifact_id=f"art-{i:03d}",
                artifact_data={"report_id": f"rpt-{i}", "confidence": 0.90},
            )
        history = reflection_engine.get_reflection_history(agent_id="agent.generation")
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_reflection_history_by_verdict(self, reflection_engine):
        """按裁决过滤反思历史."""
        await reflection_engine.reflect(
            agent_id="agent.test",
            artifact_id="art-pass",
            artifact_data={"report_id": "rpt-pass", "confidence": 0.95},
        )
        await reflection_engine.reflect(
            agent_id="agent.test",
            artifact_id="art-fail",
            artifact_data={"report_id": "", "confidence": 0.1},
        )
        approved = reflection_engine.get_reflection_history(
            agent_id="agent.test", verdict=Verdict.APPROVED,
        )
        assert all(r.final_verdict == Verdict.APPROVED for r in approved)


# ============================================================
# 13. CollaborationReview 测试
# ============================================================


class TestCollaborationReview:
    """跨 Agent 协作复盘记录测试 (L5 设计文档 7.2)."""

    def test_review_creation(self):
        """创建协作复盘记录."""
        review = CollaborationReview(
            session_id="sess-001",
            trigger=CollaborationTrigger.DEBATE,
            participants=["agent.a", "agent.b"],
            metrics={
                "total_duration_s": 187,
                "consensus_confidence": 0.82,
                "disagreement_points": 3,
                "compromise_count": 1,
                "total_token_cost": 28500,
                "tool_calls": 14,
            },
            insights=[
                "辩论在第2轮达成共识, 效率高于平均值",
                "审核Agent发现1处数值偏差",
            ],
        )
        assert review.session_id == "sess-001"
        assert review.trigger == CollaborationTrigger.DEBATE
        assert len(review.participants) == 2
        assert review.metrics["consensus_confidence"] == 0.82
        assert len(review.insights) == 2
        assert review.review_id.startswith("colab-")

    def test_review_to_dict(self):
        """复盘记录应可序列化."""
        review = CollaborationReview(
            session_id="sess-002",
            trigger=CollaborationTrigger.VOTING,
            participants=["agent.x"],
            metrics={"total_duration_s": 100},
        )
        d = review.to_dict()
        assert d["session_id"] == "sess-002"
        assert d["trigger"] == "voting"
        assert "review_id" in d
        assert "timestamp" in d


# ============================================================
# 14. QualityReport 测试
# ============================================================


class TestQualityReport:
    """全链路质量报告测试."""

    def test_report_creation(self, sample_review_record):
        """创建质量报告."""
        report = QualityReport(
            session_id="sess-001",
            artifact_id="art-001",
            reflection_result=ReflectionResult(
                artifact_id="art-001",
                agent_id="agent.generation",
                trigger=ReflectionTrigger.SINGLE_AGENT,
                reviews=[sample_review_record],
                final_verdict=Verdict.APPROVED,
                max_iterations=3,
            ),
        )
        assert report.session_id == "sess-001"
        assert report.artifact_id == "art-001"
        assert report.reflection_result is not None
        assert report.report_id.startswith("qreport-")

    def test_report_summary(self, sample_review_record):
        """质量报告应生成摘要."""
        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.generation",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[sample_review_record],
            final_verdict=Verdict.APPROVED,
            max_iterations=3,
        )
        report = QualityReport(
            session_id="sess-001",
            artifact_id="art-001",
            reflection_result=result,
        )
        summary = report.summary
        assert "verdict" in summary
        assert summary["verdict"] == "approved"
        assert "score" in summary
        assert "iterations" in summary

    def test_report_to_dict(self, sample_review_record):
        """质量报告应可序列化."""
        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.generation",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[sample_review_record],
            final_verdict=Verdict.APPROVED,
            max_iterations=3,
        )
        report = QualityReport(
            session_id="sess-001",
            artifact_id="art-001",
            reflection_result=result,
        )
        d = report.to_dict()
        assert d["session_id"] == "sess-001"
        assert "report_id" in d
        assert "reflection_result" in d


# ============================================================
# 15. 集成测试
# ============================================================


class TestReflectionIntegration:
    """反思与质量控制集成测试 (与现有系统联动)."""

    @pytest.mark.asyncio
    async def test_reflection_with_orchestration(self, reflection_engine):
        """反思 + 编排引擎联动 (OrchestrationResult → 反思)."""
        # 模拟编排结果
        orchestration_output = {
            "task_id": "t1",
            "agent_id": "agent.diagnosis",
            "output": {"report_id": "rpt-001", "kp_gaps": ["KP-12"], "confidence": 0.90},
        }

        # 对编排结果进行反思
        result = await reflection_engine.reflect(
            agent_id=orchestration_output["agent_id"],
            artifact_id="art-orch-001",
            artifact_data=orchestration_output["output"],
        )
        assert result.final_verdict in (Verdict.APPROVED, Verdict.REVISE)

    @pytest.mark.asyncio
    async def test_reflection_with_artifact_lifecycle(self, reflection_engine):
        """反思 + 产物生命周期联动 (反思结果 → ArtifactState)."""
        result = await reflection_engine.reflect(
            agent_id="agent.generation",
            artifact_id="art-life-001",
            artifact_data={"report_id": "rpt-life", "confidence": 0.95},
        )
        # 通过反思 → 可触发 REVIEWED 状态
        if result.final_verdict == Verdict.APPROVED:
            assert result.reviews[-1].verdict == Verdict.APPROVED

    @pytest.mark.asyncio
    async def test_adjudication_with_compensations(self, quality_gate):
        """裁决 + 补偿回滚 (Temporal Saga 模式)."""
        executed = []

        async def comp1():
            executed.append("comp1")

        async def comp2():
            executed.append("comp2")

        executor = AdjudicationExecutor(gate=quality_gate)
        review = ReviewRecord(
            artifact_id="art-saga",
            reviewer="cc1.actor_critic",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.2, weight=1.0),
            ],
            verdict=Verdict.REJECTED,
        )
        result = await executor.adjudicate(review, compensations=[comp1, comp2])
        assert result.action == GateAction.REJECT
        # 补偿应逆序执行 (Temporal Saga 模式)
        assert executed == ["comp2", "comp1"]

    @pytest.mark.asyncio
    async def test_reputation_feedback_loop(self, reflection_engine, reputation_ledger):
        """声誉反馈闭环 (反思 → 声誉更新 → 阈值调整)."""
        reputation_ledger.register("agent.generation", initial_score=80.0)

        # 多次成功反思
        for i in range(3):
            await reflection_engine.reflect(
                agent_id="agent.generation",
                artifact_id=f"art-feedback-{i}",
                artifact_data={"report_id": f"rpt-{i}", "confidence": 0.95},
            )

        final_score = reputation_ledger.get_score("agent.generation")
        assert final_score > 80.0  # 声誉持续提升

        # 信任度提高后, 推荐阈值应放宽
        threshold = reputation_ledger.recommended_threshold("agent.generation", base=0.85)
        assert threshold <= 0.85

    @pytest.mark.asyncio
    async def test_multi_agent_quality_topology(self, reflection_engine, reputation_ledger):
        """多 Agent 质量拓扑 (4 个核心 Agent 反思)."""
        reputation_ledger.register("agent.diagnosis", initial_score=85.0)
        reputation_ledger.register("agent.generation", initial_score=80.0)
        reputation_ledger.register("agent.review", initial_score=90.0)
        reputation_ledger.register("agent.guidance", initial_score=75.0)

        agents_data = [
            ("agent.diagnosis", {"report": "diagnosis", "confidence": 0.90}),
            ("agent.generation", {"graph": "kg", "confidence": 0.85}),
            ("agent.review", {"review": "pass", "confidence": 0.95}),
            ("agent.guidance", {"plan": "study", "confidence": 0.80}),
        ]

        for agent_id, data in agents_data:
            await reflection_engine.reflect(
                agent_id=agent_id,
                artifact_id=f"art-topo-{agent_id}",
                artifact_data=data,
            )

        # 所有 Agent 都应有反思历史
        for agent_id, _ in agents_data:
            history = reflection_engine.get_reflection_history(agent_id=agent_id)
            assert len(history) >= 1


# ============================================================
# 16. 错误处理测试
# ============================================================


class TestReflectionErrorHandling:
    """反思与质量控制错误处理测试."""

    def test_reflection_error_creation(self):
        """创建反思错误."""
        err = ReflectionError("Test error")
        assert str(err) == "Test error"

    @pytest.mark.asyncio
    async def test_reflection_with_empty_agent_id_raises(self, reflection_engine):
        """空 agent_id 应抛异常."""
        with pytest.raises((ValueError, ReflectionError)):
            await reflection_engine.reflect(
                agent_id="",
                artifact_id="art-001",
                artifact_data={},
            )

    @pytest.mark.asyncio
    async def test_reflection_with_empty_artifact_id_raises(self, reflection_engine):
        """空 artifact_id 应抛异常."""
        with pytest.raises((ValueError, ReflectionError)):
            await reflection_engine.reflect(
                agent_id="agent.test",
                artifact_id="",
                artifact_data={},
            )

    def test_gate_zero_threshold(self):
        """阈值为 0 时所有分数都通过."""
        gate = QualityGate(name="zero_gate", threshold=0.0, hard_floor=0.0)
        result = gate.evaluate(score=0.01, iteration=1)
        assert result.passed is True

    def test_gate_perfect_threshold(self):
        """阈值为 1.0 时只有满分通过."""
        gate = QualityGate(name="perfect_gate", threshold=1.0, hard_floor=0.5)
        result_pass = gate.evaluate(score=1.0, iteration=1)
        assert result_pass.passed is True

        result_fail = gate.evaluate(score=0.99, iteration=1)
        assert result_fail.passed is False

    @pytest.mark.asyncio
    async def test_collaboration_review_empty_participants_raises(self, reflection_engine):
        """空参与者列表应抛异常."""
        with pytest.raises((ValueError, ReflectionError)):
            await reflection_engine.collaboration_review(
                session_id="sess-001",
                trigger=CollaborationTrigger.DEBATE,
                participants=[],
                metrics={},
            )
