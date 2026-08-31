"""反思与质量控制模块集成测试.

验证 reflection_quality 模块与 L5 其他核心模块的集成:
1. ArtifactManager ↔ CC1Reviewer: 产物审核工作流
2. OrchestrationEngine ↔ AdjudicationExecutor: 裁决处理 + QualityGate 重试
3. SessionManager ↔ ReflectionEngine: Fork 质量评估 + 跨 Agent 复盘
4. AgentDefinition ↔ ReputationLedger: 声誉配置联动
"""

from __future__ import annotations

import asyncio
import pytest

from dy3_polaris.l5.artifact_manager import (
    Artifact,
    ArtifactManager,
    ArtifactState,
    ArtifactType,
    InMemoryArtifactStore,
)
from dy3_polaris.l5.agent_definition import (
    AgentDefinition,
    AgentRegistry,
    DecisionAuthority,
    PromptReference,
    ReputationConfig,
    SelfEvolutionConfig,
)
from dy3_polaris.l5.orchestration_engine import (
    OrchestrationEngine,
    OrchestrationParadigm,
    OrchestrationPlan,
    OrchestrationResult,
    OrchestrationState,
    OrchestrationTask,
)
from dy3_polaris.l5.reflection_quality import (
    AdjudicationExecutor,
    CC1Reviewer,
    CollaborationTrigger,
    GateAction,
    QualityGate,
    ReflectionEngine,
    ReputationLedger,
    Verdict,
)
from dy3_polaris.l5.session_manager import (
    ForkEvaluator,
    ForkMergeScope,
    SessionManager,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def quality_gate():
    """标准质量门控."""
    return QualityGate(
        name="standard",
        threshold=0.80,
        hard_floor=0.40,
        max_revisions=3,
    )


@pytest.fixture
def cc1_reviewer():
    """CC1 审核器."""
    return CC1Reviewer()


@pytest.fixture
def reputation_ledger():
    """声誉账本."""
    return ReputationLedger()


@pytest.fixture
def reflection_engine(quality_gate, cc1_reviewer, reputation_ledger):
    """反思引擎."""
    return ReflectionEngine(
        gate=quality_gate,
        reviewer=cc1_reviewer,
        reputation_ledger=reputation_ledger,
    )


@pytest.fixture
def artifact_manager():
    """产物管理器."""
    store = InMemoryArtifactStore()
    return ArtifactManager(store=store)


@pytest.fixture
def adjudication_executor(quality_gate):
    """裁决执行器."""
    return AdjudicationExecutor(gate=quality_gate)


@pytest.fixture
def session_manager():
    """会话管理器."""
    return SessionManager()


@pytest.fixture
def agent_registry():
    """Agent 注册中心 (含测试 Agent)."""
    registry = AgentRegistry()
    agent_def = AgentDefinition(
        id="agent.generation.quiz",
        name="Quiz Generation Agent",
        role="Generate quiz questions for chemistry learners",
        system_prompt=PromptReference(template_id="quiz_gen", version="v1.0.0"),
        tools=["tool.quiz_gen"],
        reputation_config=ReputationConfig(
            initial_score=85.0,
            penalty_factor=1.2,
            reward_factor=1.1,
        ),
    )
    registry.register(agent_def)
    return registry


# ============================================================
# 1. ArtifactManager ↔ CC1Reviewer 集成测试
# ============================================================


class TestArtifactManagerCC1Integration:
    """产物管理器与 CC1 审核器集成测试."""

    @pytest.mark.asyncio
    async def test_review_artifact_updates_cc1_status_pass(
        self,
        artifact_manager,
        cc1_reviewer,
        quality_gate,
    ):
        """高质量产物 → CC1 审核通过 → cc1_status 更新为 pass."""
        # 创建高质量产物
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation.quiz",
            payload={
                "content": "Water boils at 100°C at sea level.",
                "confidence": 0.95,
                "references": ["doi:10.1000/physchem"],
                "report_id": "rpt-001",
                "kp_gaps": ["KP-boiling-point"],
            },
        )

        # 执行 CC1 审核
        result = await artifact_manager.review_artifact(
            artifact_id=art.artifact_id,
            reviewer=cc1_reviewer,
            gate=quality_gate,
        )

        # 验证: cc1_status 更新为 pass
        versions = artifact_manager.get_version_history(art.artifact_id)
        assert versions[-1].cc1_status == "pass"
        assert result.final_verdict == Verdict.APPROVED

    @pytest.mark.asyncio
    async def test_review_artifact_updates_cc1_status_fail(
        self,
        artifact_manager,
        cc1_reviewer,
        quality_gate,
    ):
        """低质量产物 → CC1 审核不通过 → cc1_status 更新为 fail."""
        # 创建低质量产物 (不合理数值)
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation.quiz",
            payload={
                "content": "Water boils at 9999°C.",
                "confidence": 0.1,
                "boiling_point": 9999,
            },
        )

        # 执行 CC1 审核
        result = await artifact_manager.review_artifact(
            artifact_id=art.artifact_id,
            reviewer=cc1_reviewer,
            gate=quality_gate,
        )

        # 验证: cc1_status 更新为 fail
        versions = artifact_manager.get_version_history(art.artifact_id)
        assert versions[-1].cc1_status == "fail"
        assert result.final_verdict == Verdict.REJECTED

    @pytest.mark.asyncio
    async def test_review_artifact_transitions_state(
        self,
        artifact_manager,
        cc1_reviewer,
        quality_gate,
    ):
        """审核通过后, 产物状态从 CREATED 不直接跳转到 REVIEWED (需先 RENDERED)."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation.quiz",
            payload={
                "content": "Test content",
                "confidence": 0.9,
                "references": ["doi:10.1000/test"],
                "report_id": "rpt-002",
            },
        )

        # CREATED → RENDERED (模拟渲染)
        art.transition_to(ArtifactState.RENDERED)

        # 审核
        await artifact_manager.review_artifact(
            artifact_id=art.artifact_id,
            reviewer=cc1_reviewer,
            gate=quality_gate,
        )

        # 验证: 状态转换为 REVIEWED
        assert art.state == ArtifactState.REVIEWED

    @pytest.mark.asyncio
    async def test_review_artifact_not_found_raises(
        self,
        artifact_manager,
        cc1_reviewer,
        quality_gate,
    ):
        """审核不存在的产物 → 抛出 ArtifactNotFoundError."""
        from dy3_polaris.l5.artifact_manager import ArtifactNotFoundError

        with pytest.raises(ArtifactNotFoundError):
            await artifact_manager.review_artifact(
                artifact_id="nonexistent",
                reviewer=cc1_reviewer,
                gate=quality_gate,
            )


# ============================================================
# 2. OrchestrationEngine ↔ AdjudicationExecutor 集成测试
# ============================================================


class TestOrchestrationAdjudicationIntegration:
    """编排引擎与裁决执行器集成测试."""

    @pytest.mark.asyncio
    async def test_adjudication_triggered_on_debate_no_convergence(
        self,
        quality_gate,
        adjudication_executor,
    ):
        """辩论未收敛 → requires_adjudication=True → AdjudicationExecutor 处理."""
        engine = OrchestrationEngine(
            adjudication_executor=adjudication_executor,
        )

        # 模拟辩论执行 (不收敛, 设置 requires_adjudication)
        result = OrchestrationResult(
            plan_id="debate-test",
            state=OrchestrationState.COMPLETED,
        )
        result.requires_adjudication = True

        # 执行裁决处理
        adj_result = await engine.handle_adjudication(result)

        # 验证: 裁决被处理
        assert adj_result is not None
        assert adj_result.action in (
            GateAction.ALLOW,
            GateAction.REJECT,
            GateAction.REVISE,
            GateAction.ESCALATE,
        )

    @pytest.mark.asyncio
    async def test_adjudication_not_triggered_when_no_flag(
        self,
        quality_gate,
        adjudication_executor,
    ):
        """无 requires_adjudication → 不执行裁决."""
        engine = OrchestrationEngine(
            adjudication_executor=adjudication_executor,
        )

        result = OrchestrationResult(
            plan_id="pipeline-test",
            state=OrchestrationState.COMPLETED,
        )

        adj_result = await engine.handle_adjudication(result)
        assert adj_result is None

    def test_quality_gate_replaces_hardcoded_retry(
        self,
        quality_gate,
    ):
        """QualityGate.is_retryable 替换硬编码重试判断."""
        engine = OrchestrationEngine(quality_gate=quality_gate)

        # 不可重试的错误 (来自 QualityGate.non_retryable_errors)
        assert engine._is_retryable_error("ValidationError") is False

        # 可重试的错误
        assert engine._is_retryable_error("ConnectionError") is True

    def test_quality_gate_fallback_when_not_set(self):
        """未设置 QualityGate 时, 回退到硬编码模式."""
        engine = OrchestrationEngine()

        # 硬编码模式仍然工作
        assert engine._is_retryable_error("validation error") is False
        assert engine._is_retryable_error("connection error") is True

    @pytest.mark.asyncio
    async def test_adjudication_with_compensation(
        self,
        quality_gate,
        adjudication_executor,
    ):
        """裁决 REJECT 时执行补偿 (Saga 模式)."""
        engine = OrchestrationEngine(
            adjudication_executor=adjudication_executor,
        )

        compensation_called = []

        async def compensation_1():
            compensation_called.append("comp1")

        async def compensation_2():
            compensation_called.append("comp2")

        # 创建一个低质量审核记录 (会被 REJECT)
        from dy3_polaris.l5.reflection_quality import (
            DimensionScore,
            ReflectionDimension,
            ReviewRecord,
        )

        review = ReviewRecord(
            artifact_id="art-test",
            reviewer="cc1.actor_critic",
            dimension_scores=[
                DimensionScore(
                    dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
                    score=0.1,
                ),
                DimensionScore(
                    dimension=ReflectionDimension.NUMERIC_ACCURACY,
                    score=0.1,
                ),
                DimensionScore(
                    dimension=ReflectionDimension.CITATION_COMPLETENESS,
                    score=0.1,
                ),
                DimensionScore(
                    dimension=ReflectionDimension.PEDAGOGICAL_FIT,
                    score=0.1,
                ),
            ],
            verdict=Verdict.REJECTED,
        )

        result = OrchestrationResult(
            plan_id="debate-compensation",
            state=OrchestrationState.COMPLETED,
        )
        result.requires_adjudication = True

        # 直接调用裁决执行器
        adj_result = await adjudication_executor.adjudicate(
            review=review,
            compensations=[compensation_1, compensation_2],
        )

        # 验证: 补偿被逆序执行
        assert adj_result.action == GateAction.REJECT
        assert compensation_called == ["comp2", "comp1"]


# ============================================================
# 3. SessionManager ↔ ForkEvaluator 质量维度集成测试
# ============================================================


class TestSessionManagerForkQualityIntegration:
    """会话管理器与 Fork 质量评估集成测试."""

    def test_fork_evaluation_with_quality_score(
        self,
        session_manager,
    ):
        """Fork 评估包含质量维度 → 综合分数受质量影响."""
        session_id = "sess-test-001"
        session_manager.create_session(
            agent_id="agent.generation.quiz",
            learner_id="learner-001",
        )

        # 创建 Fork
        fork = session_manager.create_fork(
            session_id=session_manager._sessions and list(session_manager._sessions.keys())[0],
            trigger_type="ab_test",
            initiator="agent.decision.tutor",
            reason="test alternative explanation",
        )

        # 评估 Fork (含质量分)
        result = session_manager.record_fork_evaluation(
            fork_id=fork.fork_id,
            learning_gain=0.8,
            completion_time_s=120.0,
            resource_tokens=5000,
            quality_score=0.92,
        )

        # 验证: 质量分影响综合分数
        assert hasattr(result, "quality_score")
        assert result.quality_score == 0.92

    def test_fork_evaluator_quality_affects_score(self):
        """质量分高的 Fork 综合分数更高."""
        evaluator = ForkEvaluator()

        # 低质量 Fork
        low_quality = evaluator.evaluate(
            fork_id="fork-low-q",
            learning_gain=0.8,
            completion_time_s=120.0,
            resource_tokens=5000,
            quality_score=0.3,
        )

        # 高质量 Fork
        high_quality = evaluator.evaluate(
            fork_id="fork-high-q",
            learning_gain=0.8,
            completion_time_s=120.0,
            resource_tokens=5000,
            quality_score=0.95,
        )

        # 验证: 高质量分数更高
        assert high_quality.score > low_quality.score

    def test_fork_evaluator_without_quality_backward_compat(self):
        """不传 quality_score 时向后兼容 (使用默认值)."""
        evaluator = ForkEvaluator()

        result = evaluator.evaluate(
            fork_id="fork-no-q",
            learning_gain=0.7,
            completion_time_s=150.0,
            resource_tokens=8000,
        )

        # 验证: 默认质量分不影响向后兼容
        assert hasattr(result, "quality_score")
        assert result.quality_score == 0.5  # 默认中性值


# ============================================================
# 4. AgentDefinition ↔ ReputationLedger 集成测试
# ============================================================


class TestAgentDefinitionReputationIntegration:
    """Agent 定义与声誉账本集成测试."""

    def test_registry_to_reputation_ledger(
        self,
        agent_registry,
    ):
        """从 AgentRegistry 批量初始化 ReputationLedger."""
        ledger = ReputationLedger()

        # 从注册中心初始化声誉账本
        ledger.initialize_from_registry(agent_registry)

        # 验证: Agent 初始声誉分来自 ReputationConfig
        score = ledger.get_score("agent.generation.quiz")
        assert score == 85.0  # 来自 ReputationConfig.initial_score

    def test_reputation_config_penalty_reward_applied(
        self,
        agent_registry,
        quality_gate,
        cc1_reviewer,
    ):
        """ReputationConfig 的 penalty_factor/reward_factor 影响声誉更新."""
        ledger = ReputationLedger()
        ledger.initialize_from_registry(agent_registry)

        engine = ReflectionEngine(
            gate=quality_gate,
            reviewer=cc1_reviewer,
            reputation_ledger=ledger,
        )

        # 模拟一次成功反思
        asyncio.run(engine.reflect(
            agent_id="agent.generation.quiz",
            artifact_id="art-001",
            artifact_data={
                "content": "Correct answer",
                "confidence": 0.95,
                "references": ["doi:10.1000/test"],
                "report_id": "rpt-001",
            },
        ))

        # 验证: 成功后声誉提升
        new_score = ledger.get_score("agent.generation.quiz")
        assert new_score > 85.0  # 声誉提升

    def test_unregistered_agent_uses_default(
        self,
        agent_registry,
    ):
        """未注册的 Agent 使用默认声誉分."""
        ledger = ReputationLedger()
        ledger.initialize_from_registry(agent_registry)

        score = ledger.get_score("agent.unknown.new")
        assert score == ReputationLedger.DEFAULT_SCORE


# ============================================================
# 5. 端到端集成测试
# ============================================================


class TestEndToEndIntegration:
    """端到端集成测试: 产物创建 → 反思 → 裁决 → 声誉更新."""

    @pytest.mark.asyncio
    async def test_full_quality_pipeline(
        self,
        artifact_manager,
        cc1_reviewer,
        quality_gate,
        reputation_ledger,
        agent_registry,
    ):
        """完整质量管线: 创建产物 → CC1 审核 → 声誉更新."""
        # 1. 初始化声誉账本
        reputation_ledger.initialize_from_registry(agent_registry)

        # 2. 创建反思引擎
        engine = ReflectionEngine(
            gate=quality_gate,
            reviewer=cc1_reviewer,
            reputation_ledger=reputation_ledger,
        )

        # 3. 创建产物
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation.quiz",
            payload={
                "content": "H₂O has a molar mass of 18.015 g/mol.",
                "confidence": 0.92,
                "references": ["doi:10.1000/chem"],
                "report_id": "rpt-chem-001",
                "kp_gaps": ["KP-molar-mass"],
            },
        )

        # 4. 执行反思
        result = await engine.reflect(
            agent_id="agent.generation.quiz",
            artifact_id=art.artifact_id,
            artifact_data=art.payload,
        )

        # 5. 验证: 质量通过
        assert result.final_verdict == Verdict.APPROVED

        # 6. 验证: 声誉提升
        new_score = reputation_ledger.get_score("agent.generation.quiz")
        assert new_score > 85.0

        # 7. 验证: 反思历史可查
        history = engine.get_reflection_history("agent.generation.quiz")
        assert len(history) == 1
        assert history[0].final_verdict == Verdict.APPROVED

    @pytest.mark.asyncio
    async def test_collaboration_review_after_fork_merge(
        self,
        session_manager,
        reflection_engine,
    ):
        """Fork 合并后触发跨 Agent 复盘."""
        # 创建会话
        record = session_manager.create_session(
            agent_id="agent.decision.tutor",
            learner_id="learner-001",
        )

        # 创建 Fork
        fork = session_manager.create_fork(
            session_id=record.session_id,
            trigger_type="ab_test",
            initiator="agent.decision.tutor",
            reason="test alternative path",
        )

        # 评估 Fork
        session_manager.record_fork_evaluation(
            fork_id=fork.fork_id,
            learning_gain=0.15,
            completion_time_s=200.0,
            resource_tokens=10000,
            quality_score=0.85,
        )

        # 合并 Fork
        session_manager.merge_fork(
            fork_id=fork.fork_id,
            target_session_id=record.session_id,
            merge_scope=[ForkMergeScope.KERNEL_STATE],
        )

        # 触发跨 Agent 复盘
        review = await reflection_engine.collaboration_review(
            session_id=record.session_id,
            trigger=CollaborationTrigger.FORK_MERGE,
            participants=["agent.generation.quiz", "agent.decision.tutor"],
            metrics={
                "total_duration_s": 200.0,
                "consensus_confidence": 0.85,
                "total_token_cost": 10000,
                "learning_gain": 0.15,
            },
        )

        # 验证: 复盘洞察已生成
        assert len(review.insights) > 0
        assert any("Fork" in insight or "学习增益" in insight for insight in review.insights)
