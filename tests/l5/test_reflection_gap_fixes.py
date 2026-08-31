"""反思与质量控制模块缺口修复测试 — TDD 测试用例 (第三阶段).

针对深度审查发现的设计缺口, 按 L5 设计文档要求修复:

1. ReflectionEngine 存储协作复盘 + 按 session_id 查询 (设计 9.5.2)
2. ArtifactManager.review_artifact 实现自纠循环 (设计 7.1.2 warn 级别)
3. OrchestrationEngine 触发跨 Agent 复盘 (设计 7.2.1 辩论/投票)
4. SessionManager merge_fork 触发复盘 (设计 7.2.1 Fork 合并)
5. ReputationLedger 使用 penalty_factor/reward_factor (死代码激活)
6. _is_retryable_error 参数语义修正 (Bug 修复)
7. 声誉阈值推荐接入 QualityGate (闭环反馈)

融合世界先进方案:
- LangGraph: Generator-Critic 自纠循环 + state persistence
- AutoGen: Reputation system with configurable factors
- Temporal: RetryPolicy error type classification
- Claude Science: Adaptive quality thresholds based on trust
- Google ADK: Collaboration review persistence
"""

from __future__ import annotations

import pytest
from typing import Any

from dy3_polaris.l5.reflection_quality import (
    AdjudicationExecutor,
    CC1Reviewer,
    CollaborationReview,
    CollaborationTrigger,
    DimensionScore,
    GateAction,
    GateResult,
    QualityGate,
    QualityReport,
    QualityTrendAnalyzer,
    ReflectionDimension,
    ReflectionEngine,
    ReflectionResult,
    ReflectionTrigger,
    ReputationLedger,
    ReviewRecord,
    TargetedSelfCorrector,
    Verdict,
)


# ============================================================
# Fix 1: ReflectionEngine 存储协作复盘 + 按 session_id 查询
# (设计 9.5.2: 查询跨 Agent 复盘记录)
# ============================================================


class TestCollaborationReviewStorage:
    """协作复盘存储与查询测试 (设计 9.5.2)."""

    def _make_engine(self) -> ReflectionEngine:
        """创建测试用 ReflectionEngine."""
        gate = QualityGate(name="test", threshold=0.8, hard_floor=0.4)
        return ReflectionEngine(
            gate=gate,
            reviewer=CC1Reviewer(),
            reputation_ledger=ReputationLedger(),
        )

    @pytest.mark.asyncio
    async def test_collaboration_review_stored(self):
        """collaboration_review 完成后应存储复盘记录."""
        engine = self._make_engine()
        review = await engine.collaboration_review(
            session_id="sess-001",
            trigger=CollaborationTrigger.DEBATE,
            participants=["agent.a", "agent.b"],
            metrics={"total_duration_s": 120, "consensus_confidence": 0.85},
        )
        assert review is not None

        reviews = engine.get_collaboration_reviews("sess-001")
        assert len(reviews) == 1
        assert reviews[0].session_id == "sess-001"

    @pytest.mark.asyncio
    async def test_get_collaboration_reviews_by_session(self):
        """按 session_id 查询协作复盘记录."""
        engine = self._make_engine()
        await engine.collaboration_review(
            session_id="sess-001",
            trigger=CollaborationTrigger.DEBATE,
            participants=["agent.a", "agent.b"],
            metrics={},
        )
        await engine.collaboration_review(
            session_id="sess-002",
            trigger=CollaborationTrigger.VOTING,
            participants=["agent.c", "agent.d"],
            metrics={},
        )

        reviews_1 = engine.get_collaboration_reviews("sess-001")
        assert len(reviews_1) == 1
        assert reviews_1[0].trigger == CollaborationTrigger.DEBATE

        reviews_2 = engine.get_collaboration_reviews("sess-002")
        assert len(reviews_2) == 1
        assert reviews_2[0].trigger == CollaborationTrigger.VOTING

    @pytest.mark.asyncio
    async def test_get_collaboration_reviews_filtered_by_trigger(self):
        """按 trigger 类型过滤协作复盘记录."""
        engine = self._make_engine()
        await engine.collaboration_review(
            session_id="sess-001",
            trigger=CollaborationTrigger.DEBATE,
            participants=["agent.a"],
            metrics={},
        )
        await engine.collaboration_review(
            session_id="sess-001",
            trigger=CollaborationTrigger.VOTING,
            participants=["agent.b"],
            metrics={},
        )
        await engine.collaboration_review(
            session_id="sess-001",
            trigger=CollaborationTrigger.FORK_MERGE,
            participants=["agent.c"],
            metrics={},
        )

        debates = engine.get_collaboration_reviews(
            "sess-001", trigger=CollaborationTrigger.DEBATE
        )
        assert len(debates) == 1
        assert debates[0].trigger == CollaborationTrigger.DEBATE

    @pytest.mark.asyncio
    async def test_get_collaboration_reviews_empty_session(self):
        """查询不存在的 session 返回空列表."""
        engine = self._make_engine()
        reviews = engine.get_collaboration_reviews("nonexistent")
        assert reviews == []

    @pytest.mark.asyncio
    async def test_collaboration_review_count_accumulates(self):
        """同一 session 多次复盘记录累积."""
        engine = self._make_engine()
        for i in range(3):
            await engine.collaboration_review(
                session_id="sess-001",
                trigger=CollaborationTrigger.DEBATE,
                participants=["agent.a"],
                metrics={"round": i},
            )
        reviews = engine.get_collaboration_reviews("sess-001")
        assert len(reviews) == 3


# ============================================================
# Fix 2: ArtifactManager.review_artifact 实现自纠循环
# (设计 7.1.2: warn 级别 → 自纠 → 重新检查)
# ============================================================


class TestArtifactManagerSelfCorrection:
    """ArtifactManager 审核自纠循环测试 (设计 7.1.2)."""

    def _make_manager(self):
        """创建带 CC1 审核的 ArtifactManager."""
        from dy3_polaris.l5.artifact_manager import (
            ArtifactManager,
            InMemoryArtifactStore,
            ArtifactState,
            ArtifactType,
        )
        from dy3_polaris.l5.reflection_quality import CC1Reviewer, QualityGate

        store = InMemoryArtifactStore()
        manager = ArtifactManager(store=store)
        manager.set_reviewer(CC1Reviewer())
        manager.set_quality_gate(
            QualityGate(name="artifact", threshold=0.8, hard_floor=0.4, max_revisions=3)
        )
        return manager

    @pytest.mark.asyncio
    async def test_review_artifact_revise_triggers_self_correction(self):
        """REVISE 动作触发自纠循环."""
        manager = self._make_manager()
        from dy3_polaris.l5.artifact_manager import ArtifactType, ArtifactState

        art = manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={"confidence": 0.4, "references": []},
        )
        # 手动转换到 RENDERED 状态
        art.transition_to(ArtifactState.RENDERED)

        result = await manager.review_artifact(art.artifact_id)

        # 低置信度应触发 REVISE → 自纠 → 可能通过
        assert result is not None
        # 自纠后置信度应被提升, 迭代次数 >= 1
        assert result.total_iterations >= 1

    @pytest.mark.asyncio
    async def test_review_artifact_pass_on_high_quality(self):
        """高质量产物直接通过审核."""
        manager = self._make_manager()
        from dy3_polaris.l5.artifact_manager import ArtifactType, ArtifactState

        art = manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={
                "confidence": 0.9,
                "references": ["doi:10.1234/test"],
                "report_id": "rpt-001",
                "kp_gaps": ["KP-01"],
            },
        )
        art.transition_to(ArtifactState.RENDERED)

        result = await manager.review_artifact(art.artifact_id)

        assert result.final_verdict == Verdict.APPROVED
        assert result.total_iterations == 1

    @pytest.mark.asyncio
    async def test_review_artifact_respects_max_revisions(self):
        """自纠循环不超过最大修订次数."""
        manager = self._make_manager()
        from dy3_polaris.l5.artifact_manager import ArtifactType, ArtifactState

        art = manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={"confidence": 0.1},  # 极低置信度, 难以通过
        )
        art.transition_to(ArtifactState.RENDERED)

        result = await manager.review_artifact(art.artifact_id)

        # 最大修订次数为 3, 总迭代不超过 max_revisions + 1
        assert result.total_iterations <= 4


# ============================================================
# Fix 3: OrchestrationEngine 触发跨 Agent 复盘
# (设计 7.2.1: 辩论/投票完成后触发联合复盘)
# ============================================================


class TestOrchestrationCrossAgentReview:
    """编排引擎跨 Agent 复盘触发测试 (设计 7.2.1)."""

    def _make_engine_with_reflection(self):
        """创建带反思引擎的编排引擎."""
        from dy3_polaris.l5.orchestration_engine import OrchestrationEngine
        from dy3_polaris.l5.reflection_quality import (
            AdjudicationExecutor,
            CC1Reviewer,
            QualityGate,
            ReflectionEngine,
            ReputationLedger,
        )

        gate = QualityGate(name="orch", threshold=0.8, hard_floor=0.4)
        reflection_engine = ReflectionEngine(
            gate=gate,
            reviewer=CC1Reviewer(),
            reputation_ledger=ReputationLedger(),
        )
        engine = OrchestrationEngine(
            adjudication_executor=AdjudicationExecutor(gate),
            quality_gate=gate,
            reflection_engine=reflection_engine,
        )
        return engine, reflection_engine

    @pytest.mark.asyncio
    async def test_debate_triggers_collaboration_review(self):
        """辩论完成后触发协作复盘."""
        from dy3_polaris.l5.orchestration_engine import (
            OrchestrationEngine,
            OrchestrationResult,
            OrchestrationState,
            OrchestrationParadigm,
        )

        engine, reflection_engine = self._make_engine_with_reflection()

        result = OrchestrationResult(
            plan_id="plan-001",
            state=OrchestrationState.COMPLETED,
        )

        await engine.trigger_collaboration_review(
            session_id="sess-001",
            result=result,
            participants=["agent.pro", "agent.anti"],
            paradigm=OrchestrationParadigm.DEBATE,
        )

        reviews = reflection_engine.get_collaboration_reviews("sess-001")
        assert len(reviews) == 1
        assert reviews[0].trigger == CollaborationTrigger.DEBATE

    @pytest.mark.asyncio
    async def test_voting_triggers_collaboration_review(self):
        """投票完成后触发协作复盘."""
        from dy3_polaris.l5.orchestration_engine import (
            OrchestrationResult,
            OrchestrationState,
            OrchestrationParadigm,
        )

        engine, reflection_engine = self._make_engine_with_reflection()

        result = OrchestrationResult(
            plan_id="plan-002",
            state=OrchestrationState.COMPLETED,
        )

        await engine.trigger_collaboration_review(
            session_id="sess-002",
            result=result,
            participants=["agent.s1", "agent.s2", "agent.s3"],
            paradigm=OrchestrationParadigm.VOTING,
        )

        reviews = reflection_engine.get_collaboration_reviews("sess-002")
        assert len(reviews) == 1
        assert reviews[0].trigger == CollaborationTrigger.VOTING

    @pytest.mark.asyncio
    async def test_no_review_when_reflection_engine_absent(self):
        """未配置反思引擎时不触发复盘 (向后兼容)."""
        from dy3_polaris.l5.orchestration_engine import (
            OrchestrationEngine,
            OrchestrationResult,
            OrchestrationState,
            OrchestrationParadigm,
        )

        engine = OrchestrationEngine()  # 无 reflection_engine

        result = OrchestrationResult(
            plan_id="plan-003",
            state=OrchestrationState.COMPLETED,
        )

        # 不应抛出异常
        await engine.trigger_collaboration_review(
            session_id="sess-003",
            result=result,
            participants=["agent.a"],
            paradigm=OrchestrationParadigm.DEBATE,
        )


# ============================================================
# Fix 4: SessionManager merge_fork 触发复盘
# (设计 7.2.1: Fork 合并后复盘)
# ============================================================


class TestSessionManagerForkReview:
    """SessionManager Fork 合并复盘测试 (设计 7.2.1)."""

    @pytest.mark.asyncio
    async def test_merge_fork_triggers_review(self):
        """Fork 合并后触发协作复盘."""
        from dy3_polaris.l5.session_manager import SessionManager
        from dy3_polaris.l5.reflection_quality import (
            CC1Reviewer,
            QualityGate,
            ReflectionEngine,
            ReputationLedger,
        )

        gate = QualityGate(name="session", threshold=0.8)
        reflection_engine = ReflectionEngine(
            gate=gate,
            reviewer=CC1Reviewer(),
            reputation_ledger=ReputationLedger(),
        )
        sm = SessionManager()
        sm.set_reflection_engine(reflection_engine)

        # 创建主会话
        main_session = sm.create_session(
            agent_id="agent.main",
            learner_id="learner-001",
        )

        # 触发 Fork 合并复盘
        await sm.trigger_fork_merge_review(
            session_id=main_session.session_id,
            participants=["agent.knowledge", "agent.review"],
            metrics={
                "learning_gain": 0.15,
                "total_duration_s": 200,
                "total_token_cost": 15000,
            },
        )

        reviews = reflection_engine.get_collaboration_reviews(
            main_session.session_id
        )
        assert len(reviews) == 1
        assert reviews[0].trigger == CollaborationTrigger.FORK_MERGE

    @pytest.mark.asyncio
    async def test_merge_fork_without_reflection_engine(self):
        """未配置反思引擎时不崩溃 (向后兼容)."""
        from dy3_polaris.l5.session_manager import SessionManager

        sm = SessionManager()  # 无 reflection_engine

        # 不应抛出异常
        await sm.trigger_fork_merge_review(
            session_id="sess-fake",
            participants=["agent.a"],
            metrics={},
        )


# ============================================================
# Fix 5: ReputationLedger 使用 penalty_factor / reward_factor
# (死代码激活: ReputationConfig 中的配置生效)
# ============================================================


class TestReputationConfigFactors:
    """ReputationConfig penalty/reward factor 激活测试."""

    def _make_result(self, verdict: Verdict, iterations: int = 1) -> ReflectionResult:
        """创建测试用 ReflectionResult."""
        scores = [
            DimensionScore(dimension=d, score=0.8)
            for d in ReflectionDimension
        ]
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1",
            dimension_scores=scores,
            verdict=verdict,
            iteration=iterations,
        )
        return ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.test",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[review],
            final_verdict=verdict,
        )

    def test_reward_factor_amplifies_approved(self):
        """reward_factor > 1.0 放大 APPROVED 的声誉增益."""
        ledger = ReputationLedger()
        ledger.register("agent.test", initial_score=50.0)

        result = self._make_result(Verdict.APPROVED)

        # 默认 reward_factor = 1.0
        ledger.update("agent.test", result, reward_factor=1.0)
        default_score = ledger.get_score("agent.test")

        # 重置
        ledger.register("agent.test", initial_score=50.0)
        ledger.update("agent.test", result, reward_factor=2.0)
        amplified_score = ledger.get_score("agent.test")

        assert amplified_score > default_score

    def test_penalty_factor_amplifies_rejected(self):
        """penalty_factor > 1.0 放大 REJECTED 的声誉惩罚."""
        ledger = ReputationLedger()
        ledger.register("agent.test", initial_score=50.0)

        result = self._make_result(Verdict.REJECTED)

        # 默认 penalty_factor = 1.0
        ledger.update("agent.test", result, penalty_factor=1.0)
        default_score = ledger.get_score("agent.test")

        # 重置
        ledger.register("agent.test", initial_score=50.0)
        ledger.update("agent.test", result, penalty_factor=2.0)
        penalized_score = ledger.get_score("agent.test")

        assert penalized_score < default_score

    def test_default_factors_preserve_behavior(self):
        """默认 factor (1.0) 保持原有行为."""
        ledger = ReputationLedger()
        ledger.register("agent.test", initial_score=50.0)
        result = self._make_result(Verdict.APPROVED)

        ledger.update("agent.test", result)
        score = ledger.get_score("agent.test")

        # 原有行为: EMA_ALPHA=0.3, target=100+5 (first try)
        # new = 0.3 * 105 + 0.7 * 50 = 31.5 + 35 = 66.5
        assert abs(score - 66.5) < 0.5


# ============================================================
# Fix 6: _is_retryable_error 参数语义修正
# (Bug: 传入错误消息而非错误类型)
# ============================================================


class TestRetryableErrorSemantics:
    """_is_retryable_error 参数语义修正测试."""

    def test_quality_gate_is_retryable_by_type(self):
        """QualityGate.is_retryable 按错误类型判断."""
        gate = QualityGate(
            name="test",
            threshold=0.8,
            non_retryable_errors=["ValidationError", "PermissionError"],
        )

        assert gate.is_retryable("TimeoutError") is True
        assert gate.is_retryable("ValidationError") is False
        assert gate.is_retryable("PermissionError") is False

    def test_orchestration_is_retryable_extracts_type(self):
        """OrchestrationEngine._is_retryable_error 从异常提取类型名."""
        from dy3_polaris.l5.orchestration_engine import OrchestrationEngine
        from dy3_polaris.l5.reflection_quality import QualityGate

        gate = QualityGate(
            name="orch",
            threshold=0.8,
            non_retryable_errors=["ValidationError"],
        )
        engine = OrchestrationEngine(quality_gate=gate)

        # 传入实际异常对象
        try:
            raise ValueError("some error")
        except ValueError as e:
            assert engine._is_retryable_error(e) is True

        # 传入 non_retryable 类型的异常
        class ValidationError(Exception):
            pass

        try:
            raise ValidationError("validation failed")
        except ValidationError as e:
            assert engine._is_retryable_error(e) is False

    def test_orchestration_is_retryable_fallback_without_gate(self):
        """未配置 QualityGate 时回退到字符串匹配."""
        from dy3_polaris.l5.orchestration_engine import OrchestrationEngine

        engine = OrchestrationEngine()

        try:
            raise TimeoutError("timed out")
        except TimeoutError as e:
            # TimeoutError 应该可重试
            assert engine._is_retryable_error(e) is True

    def test_orchestration_is_retryable_string_fallback(self):
        """传入字符串时回退到字符串模式匹配."""
        from dy3_polaris.l5.orchestration_engine import OrchestrationEngine

        engine = OrchestrationEngine()

        assert engine._is_retryable_error("TimeoutError: timed out") is True
        assert engine._is_retryable_error("validation error: bad input") is False


# ============================================================
# Fix 7: 声誉阈值推荐接入 QualityGate
# (闭环反馈: 高信任→放宽, 低信任→收紧)
# ============================================================


class TestReputationThresholdFeedback:
    """声誉阈值推荐接入 QualityGate 测试 (闭环反馈)."""

    def test_recommended_threshold_high_trust(self):
        """高信任 Agent 阈值放宽."""
        ledger = ReputationLedger()
        ledger.register("agent.trusted", initial_score=95.0)

        threshold = ledger.recommended_threshold("agent.trusted", base=0.85)
        assert threshold < 0.85  # 放宽 (降低)

    def test_recommended_threshold_low_trust(self):
        """低信任 Agent 阈值收紧."""
        ledger = ReputationLedger()
        ledger.register("agent.untrusted", initial_score=20.0)

        threshold = ledger.recommended_threshold("agent.untrusted", base=0.85)
        assert threshold > 0.85  # 收紧 (提高)

    def test_recommended_threshold_medium_trust(self):
        """中等信任 Agent 阈值不变."""
        ledger = ReputationLedger()
        ledger.register("agent.medium", initial_score=65.0)

        threshold = ledger.recommended_threshold("agent.medium", base=0.85)
        assert abs(threshold - 0.85) < 0.001  # 不变

    def test_engine_uses_reputation_threshold(self):
        """ReflectionEngine 根据声誉动态调整门控阈值."""
        gate = QualityGate(name="test", threshold=0.85, hard_floor=0.4)
        ledger = ReputationLedger()
        ledger.register("agent.trusted", initial_score=95.0)
        ledger.register("agent.untrusted", initial_score=20.0)

        engine = ReflectionEngine(
            gate=gate,
            reviewer=CC1Reviewer(),
            reputation_ledger=ledger,
        )

        # 高信任 Agent 获得更低阈值 (更宽松)
        trusted_threshold = engine.get_effective_threshold("agent.trusted")
        untrusted_threshold = engine.get_effective_threshold("agent.untrusted")

        assert trusted_threshold < 0.85
        assert untrusted_threshold > 0.85
        assert trusted_threshold < untrusted_threshold
