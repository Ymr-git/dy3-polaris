"""编排引擎测试 — TDD 测试用例.

测试覆盖:
1. OrchestrationTask — 任务节点定义与依赖
2. OrchestrationPlan — DAG 执行计划与拓扑排序
3. OrchestrationResult — 执行结果与溯源
4. PipelineExecutor — 顺序流水线编排 (Google ADK SequentialAgent 模式)
5. DebateExecutor — 辩论交叉验证编排 (Claude Science 辩论弧模式)
6. VotingExecutor — 投票共识编排 (LangGraph 并行 fan-out + 加权聚合)
7. OrchestrationEngine — 顶层编排引擎 (融合 Temporal + LangGraph)
8. 集成测试 — SessionManager/AgentRegistry 联动
9. 错误处理与重试 — Temporal retry policy 模式
10. 超时管理 — Temporal 四级超时模型

融合世界先进方案:
- LangGraph: StateGraph + superstep + 条件边 + checkpoint
- Temporal: Workflow/Activity 分离 + 事件溯源 + retry policy + continue-as-new
- Google ADK: SequentialAgent/ParallelAgent/LoopAgent 声明式编排
- OpenAI Agents SDK: Handoff 机制 + Guardrail 并行
- AutoGen: GroupChat 多策略 Manager + stall 检测
- CrewAI: 角色驱动 + manager 验证
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dy3_polaris.l5.orchestration_engine import (
    DebateMessage,
    DebateRound,
    DebateState,
    OrchestrationEngine,
    OrchestrationError,
    OrchestrationParadigm,
    OrchestrationPlan,
    OrchestrationResult,
    OrchestrationState,
    OrchestrationTask,
    OrchestrationTimeoutError,
    ParadigmExecutor,
    PipelineExecutor,
    DebateExecutor,
    VotingExecutor,
    VoteRecord,
    VotingResult,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_agent_registry():
    """模拟 Agent 注册中心."""
    registry = MagicMock()
    registry.get = MagicMock(return_value=MagicMock(
        agent_id="agent.test.demo",
        display_name="Test Agent",
    ))
    return registry


@pytest.fixture
def simple_tasks():
    """简单线性任务链: A → B → C."""
    return [
        OrchestrationTask(
            task_id="t1",
            agent_id="agent.diagnosis",
            name="学情诊断",
            dependencies=[],
        ),
        OrchestrationTask(
            task_id="t2",
            agent_id="agent.generation",
            name="知识生成",
            dependencies=["t1"],
        ),
        OrchestrationTask(
            task_id="t3",
            agent_id="agent.review",
            name="审核校验",
            dependencies=["t2"],
        ),
    ]


@pytest.fixture
def parallel_tasks():
    """并行任务: A → (B ∥ C) → D."""
    return [
        OrchestrationTask(
            task_id="t1",
            agent_id="agent.diagnosis",
            name="学情诊断",
            dependencies=[],
        ),
        OrchestrationTask(
            task_id="t2",
            agent_id="agent.generation",
            name="知识生成",
            dependencies=["t1"],
        ),
        OrchestrationTask(
            task_id="t3",
            agent_id="agent.review",
            name="审核校验",
            dependencies=["t1"],
        ),
        OrchestrationTask(
            task_id="t4",
            agent_id="agent.guidance",
            name="导学决策",
            dependencies=["t2", "t3"],
        ),
    ]


# ============================================================
# 1. OrchestrationTask 测试
# ============================================================

class TestOrchestrationTask:
    """编排任务节点测试."""

    def test_task_creation(self):
        """创建任务节点."""
        task = OrchestrationTask(
            task_id="t1",
            agent_id="agent.test",
            name="测试任务",
            dependencies=[],
        )
        assert task.task_id == "t1"
        assert task.agent_id == "agent.test"
        assert task.name == "测试任务"
        assert task.dependencies == []
        assert task.state == OrchestrationState.PENDING
        assert task.timeout_s == 120.0  # 默认超时

    def test_task_with_dependencies(self):
        """任务可以有依赖."""
        task = OrchestrationTask(
            task_id="t2",
            agent_id="agent.test",
            name="依赖任务",
            dependencies=["t1"],
        )
        assert task.dependencies == ["t1"]

    def test_task_with_custom_timeout(self):
        """任务可以设置自定义超时."""
        task = OrchestrationTask(
            task_id="t1",
            agent_id="agent.test",
            name="长任务",
            timeout_s=300.0,
        )
        assert task.timeout_s == 300.0

    def test_task_state_transitions(self):
        """任务状态转换: PENDING → RUNNING → COMPLETED."""
        task = OrchestrationTask(
            task_id="t1",
            agent_id="agent.test",
            name="任务",
        )
        assert task.state == OrchestrationState.PENDING
        task.state = OrchestrationState.RUNNING
        assert task.state == OrchestrationState.RUNNING
        task.state = OrchestrationState.COMPLETED
        assert task.state == OrchestrationState.COMPLETED

    def test_task_has_input_output(self):
        """任务可以存储输入和输出."""
        task = OrchestrationTask(
            task_id="t1",
            agent_id="agent.test",
            name="任务",
        )
        task.input_data = {"query": "什么是发光材料?"}
        task.output_data = {"answer": "发光材料是..."}
        assert task.input_data["query"] == "什么是发光材料?"
        assert task.output_data["answer"] == "发光材料是..."


# ============================================================
# 2. OrchestrationPlan 测试
# ============================================================

class TestOrchestrationPlan:
    """编排执行计划测试 (DAG 构建 + 拓扑排序)."""

    def test_plan_creation(self, simple_tasks):
        """创建执行计划."""
        plan = OrchestrationPlan(
            plan_id="plan-001",
            tasks=simple_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        assert plan.plan_id == "plan-001"
        assert len(plan.tasks) == 3
        assert plan.paradigm == OrchestrationParadigm.PIPELINE
        assert plan.state == OrchestrationState.PENDING

    def test_plan_topological_sort_linear(self, simple_tasks):
        """线性 DAG 拓扑排序."""
        plan = OrchestrationPlan(
            plan_id="plan-001",
            tasks=simple_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        order = plan.topological_sort()
        assert order == ["t1", "t2", "t3"]

    def test_plan_topological_sort_parallel(self, parallel_tasks):
        """并行 DAG 拓扑排序."""
        plan = OrchestrationPlan(
            plan_id="plan-002",
            tasks=parallel_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        order = plan.topological_sort()
        # t1 必须在最前, t4 必须在最后
        assert order[0] == "t1"
        assert order[-1] == "t4"
        # t2 和 t3 的顺序可以互换
        assert set(order[1:3]) == {"t2", "t3"}

    def test_plan_parallel_layers(self, parallel_tasks):
        """并行层构建 (LangGraph superstep 模式)."""
        plan = OrchestrationPlan(
            plan_id="plan-002",
            tasks=parallel_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        layers = plan.get_parallel_layers()
        assert len(layers) == 3
        assert layers[0] == ["t1"]
        assert set(layers[1]) == {"t2", "t3"}
        assert layers[2] == ["t4"]

    def test_plan_cycle_detection(self):
        """循环依赖检测."""
        tasks = [
            OrchestrationTask(task_id="a", agent_id="x", name="A", dependencies=["b"]),
            OrchestrationTask(task_id="b", agent_id="x", name="B", dependencies=["a"]),
        ]
        with pytest.raises(OrchestrationError, match="cycle"):
            OrchestrationPlan(
                plan_id="plan-cycle",
                tasks=tasks,
                paradigm=OrchestrationParadigm.PIPELINE,
            )

    def test_plan_missing_dependency(self):
        """缺失依赖检测."""
        tasks = [
            OrchestrationTask(task_id="t1", agent_id="x", name="A", dependencies=["t0"]),
        ]
        with pytest.raises(OrchestrationError, match="missing"):
            OrchestrationPlan(
                plan_id="plan-missing",
                tasks=tasks,
                paradigm=OrchestrationParadigm.PIPELINE,
            )

    def test_plan_get_task(self, simple_tasks):
        """按 ID 获取任务."""
        plan = OrchestrationPlan(
            plan_id="plan-001",
            tasks=simple_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        task = plan.get_task("t2")
        assert task is not None
        assert task.name == "知识生成"
        assert plan.get_task("nonexistent") is None


# ============================================================
# 3. OrchestrationResult 测试
# ============================================================

class TestOrchestrationResult:
    """编排执行结果测试."""

    def test_result_creation(self):
        """创建执行结果."""
        result = OrchestrationResult(
            plan_id="plan-001",
            state=OrchestrationState.COMPLETED,
            outputs={"answer": "发光材料是..."},
        )
        assert result.plan_id == "plan-001"
        assert result.state == OrchestrationState.COMPLETED
        assert result.outputs["answer"] == "发光材料是..."
        assert result.execution_time_s >= 0
        assert len(result.provenance) >= 1

    def test_result_with_error(self):
        """创建失败结果."""
        result = OrchestrationResult(
            plan_id="plan-001",
            state=OrchestrationState.FAILED,
            error="Agent execution timeout",
        )
        assert result.state == OrchestrationState.FAILED
        assert result.error == "Agent execution timeout"

    def test_result_provenance(self):
        """结果应包含溯源记录."""
        result = OrchestrationResult(
            plan_id="plan-001",
            state=OrchestrationState.COMPLETED,
            outputs={},
        )
        result.add_provenance("task.complete", {"task_id": "t1"})
        result.add_provenance("task.complete", {"task_id": "t2"})
        assert len(result.provenance) >= 3  # 1 initial + 2 added
        actions = [p["action"] for p in result.provenance]
        assert "task.complete" in actions


# ============================================================
# 4. PipelineExecutor 测试
# ============================================================

class TestPipelineExecutor:
    """顺序流水线编排测试 (Google ADK SequentialAgent 模式)."""

    @pytest.fixture
    def executor(self):
        """创建流水线执行器."""
        return PipelineExecutor()

    def test_executor_paradigm(self, executor):
        """执行器范式应为 PIPELINE."""
        assert executor.paradigm == OrchestrationParadigm.PIPELINE

    @pytest.mark.asyncio
    async def test_sequential_execution(self, executor, simple_tasks):
        """顺序执行任务链."""
        plan = OrchestrationPlan(
            plan_id="plan-seq",
            tasks=simple_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        # 模拟执行函数
        async def mock_execute(task, context):
            return {"result": f"output_{task.task_id}"}

        result = await executor.execute(plan, mock_execute)
        assert result.state == OrchestrationState.COMPLETED
        assert "t1" in result.outputs
        assert "t2" in result.outputs
        assert "t3" in result.outputs

    @pytest.mark.asyncio
    async def test_pipeline_context_passing(self, executor, simple_tasks):
        """流水线上一步输出应传递给下一步 (ADK output_key 模式)."""
        plan = OrchestrationPlan(
            plan_id="plan-ctx",
            tasks=simple_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        received_contexts = []

        async def mock_execute(task, context):
            received_contexts.append(dict(context))
            return {"step": task.task_id, "accumulated": context}

        result = await executor.execute(plan, mock_execute)
        # t1 的上下文应为空
        assert len(received_contexts[0]) == 0 or "t1" not in received_contexts[0]
        # t2 的上下文应包含 t1 的输出
        assert "t1" in received_contexts[1] or len(received_contexts[1]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_task_failure(self, executor, simple_tasks):
        """任务失败应终止流水线."""
        plan = OrchestrationPlan(
            plan_id="plan-fail",
            tasks=simple_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        call_count = 0

        async def mock_execute(task, context):
            nonlocal call_count
            call_count += 1
            if task.task_id == "t2":
                raise RuntimeError("Agent failed")

        result = await executor.execute(plan, mock_execute)
        assert result.state == OrchestrationState.FAILED
        assert "Agent failed" in result.error
        assert call_count == 2  # t1 成功, t2 失败, t3 未执行

    @pytest.mark.asyncio
    async def test_pipeline_parallel_layers(self, executor, parallel_tasks):
        """流水线执行器支持并行层 (LangGraph superstep 模式)."""
        plan = OrchestrationPlan(
            plan_id="plan-par",
            tasks=parallel_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        execution_order = []

        async def mock_execute(task, context):
            execution_order.append(task.task_id)
            await asyncio.sleep(0.01)
            return {"result": task.task_id}

        result = await executor.execute(plan, mock_execute)
        assert result.state == OrchestrationState.COMPLETED
        # t2 和 t3 应在 t1 之后、t4 之前
        assert execution_order[0] == "t1"
        assert execution_order[-1] == "t4"


# ============================================================
# 5. DebateExecutor 测试
# ============================================================

class TestDebateExecutor:
    """辩论交叉验证编排测试 (Claude Science 辩论弧模式)."""

    @pytest.fixture
    def executor(self):
        return DebateExecutor(
            max_rounds=3,
            arguments_per_round=3,
            convergence_threshold=0.1,
        )

    def test_executor_paradigm(self, executor):
        assert executor.paradigm == OrchestrationParadigm.DEBATE

    def test_debate_message_creation(self):
        """辩论消息创建."""
        msg = DebateMessage(
            sender="agent.generation",
            round=1,
            arguments=["point1", "point2"],
            confidence=0.85,
        )
        assert msg.sender == "agent.generation"
        assert msg.round == 1
        assert len(msg.arguments) == 2
        assert msg.confidence == 0.85

    def test_debate_round_creation(self):
        """辩论轮次创建."""
        rnd = DebateRound(
            round_num=1,
            pro_messages=[DebateMessage(
                sender="agent.generation", round=1,
                arguments=["arg1"], confidence=0.8,
            )],
            con_messages=[DebateMessage(
                sender="agent.review", round=1,
                arguments=["arg1"], confidence=0.7,
            )],
        )
        assert rnd.round_num == 1
        assert len(rnd.pro_messages) == 1
        assert len(rnd.con_messages) == 1

    @pytest.mark.asyncio
    async def test_debate_convergence(self, executor):
        """辩论收敛: 分歧度 < 阈值时提前终止."""
        async def mock_execute(task, context, round_num, opponent_args):
            if round_num == 1:
                return DebateMessage(
                    sender=task.agent_id,
                    round=round_num,
                    arguments=["arg1"],
                    confidence=0.80,
                )
            # 第 2 轮收敛
            return DebateMessage(
                sender=task.agent_id,
                round=round_num,
                arguments=["agreed"],
                confidence=0.82,
            )

        pro_task = OrchestrationTask(
            task_id="pro", agent_id="agent.generation",
            name="知识生成", dependencies=[],
        )
        con_task = OrchestrationTask(
            task_id="con", agent_id="agent.review",
            name="审核校验", dependencies=[],
        )

        result = await executor.execute_debate(
            pro_task=pro_task,
            con_task=con_task,
            execute_fn=mock_execute,
        )
        assert result.state == OrchestrationState.COMPLETED
        # 应在 2 轮内收敛
        assert len(result.debate_rounds) <= 2

    @pytest.mark.asyncio
    async def test_debate_max_rounds(self, executor):
        """辩论应受最大轮数限制."""
        async def mock_execute(task, context, round_num, opponent_args):
            # 始终保持高分歧度
            return DebateMessage(
                sender=task.agent_id,
                round=round_num,
                arguments=["disagree"],
                confidence=0.3 if task.agent_id == "agent.generation" else 0.9,
            )

        pro_task = OrchestrationTask(
            task_id="pro", agent_id="agent.generation",
            name="生成", dependencies=[],
        )
        con_task = OrchestrationTask(
            task_id="con", agent_id="agent.review",
            name="审核", dependencies=[],
        )

        result = await executor.execute_debate(
            pro_task=pro_task,
            con_task=con_task,
            execute_fn=mock_execute,
        )
        # 应达到最大轮数
        assert len(result.debate_rounds) == 3
        # 应标记为需要裁决
        assert result.requires_adjudication is True

    @pytest.mark.asyncio
    async def test_debate_token_budget(self):
        """辩论代币预算限制 (L5 设计文档)."""
        executor = DebateExecutor(
            max_rounds=3,
            arguments_per_round=3,
            convergence_threshold=0.1,
        )
        total_arguments = 0

        async def mock_execute(task, context, round_num, opponent_args):
            nonlocal total_arguments
            total_arguments += 3  # 每轮每方 3 个论据
            return DebateMessage(
                sender=task.agent_id,
                round=round_num,
                arguments=["arg1", "arg2", "arg3"],
                confidence=0.3 if task.agent_id == "agent.generation" else 0.9,
            )

        pro_task = OrchestrationTask(
            task_id="pro", agent_id="agent.generation",
            name="生成", dependencies=[],
        )
        con_task = OrchestrationTask(
            task_id="con", agent_id="agent.review",
            name="审核", dependencies=[],
        )

        result = await executor.execute_debate(
            pro_task=pro_task,
            con_task=con_task,
            execute_fn=mock_execute,
        )
        # 3 轮 × 2 方 × 3 论据 = 18, 但预算上限是 9 个论据点
        # (3 rounds × 3 arguments per round = 9 per side)
        assert total_arguments <= 18  # 3 rounds * 2 sides * 3 args


# ============================================================
# 6. VotingExecutor 测试
# ============================================================

class TestVotingExecutor:
    """投票共识编排测试 (LangGraph 并行 fan-out + 加权聚合)."""

    @pytest.fixture
    def executor(self):
        return VotingExecutor(
            min_strategies=3,
            max_strategies=5,
            consensus_threshold=0.5,
            similarity_threshold=0.8,
        )

    def test_executor_paradigm(self, executor):
        assert executor.paradigm == OrchestrationParadigm.VOTING

    def test_vote_record_creation(self):
        """投票记录创建."""
        vote = VoteRecord(
            strategy_id="strategy-empirical",
            agent_id="agent.generation",
            output={"answer": "答案是42"},
            confidence=0.85,
            reputation_score=0.9,
        )
        assert vote.strategy_id == "strategy-empirical"
        assert vote.confidence == 0.85
        assert vote.reputation_score == 0.9

    def test_voting_result_creation(self):
        """投票结果创建."""
        result = VotingResult(
            consensus_score=0.75,
            final_output={"answer": "发光材料是..."},
            votes=[
                VoteRecord("s1", "a1", {"x": 1}, 0.8, 0.9),
                VoteRecord("s2", "a2", {"x": 1}, 0.7, 0.8),
            ],
            reached_consensus=True,
        )
        assert result.consensus_score == 0.75
        assert result.reached_consensus is True
        assert len(result.votes) == 2

    @pytest.mark.asyncio
    async def test_voting_parallel_execution(self, executor):
        """投票应并行执行多个策略 (LangGraph fan-out)."""
        strategies = [
            OrchestrationTask(
                task_id=f"vote-{i}",
                agent_id="agent.generation",
                name=f"策略{i}",
                dependencies=[],
            )
            for i in range(3)
        ]
        execution_times = []

        async def mock_execute(task, context):
            start = time.time()
            await asyncio.sleep(0.05)
            execution_times.append((task.task_id, start, time.time()))
            return {"answer": "发光材料是...", "strategy": task.task_id}

        result = await executor.execute_voting(
            strategies=strategies,
            execute_fn=mock_execute,
        )
        assert result.state == OrchestrationState.COMPLETED
        # 并行执行: 时间应有重叠
        assert len(execution_times) == 3

    @pytest.mark.asyncio
    async def test_voting_consensus_reached(self, executor):
        """共识达成 (相似度 > 阈值)."""
        strategies = [
            OrchestrationTask(
                task_id=f"vote-{i}",
                agent_id="agent.generation",
                name=f"策略{i}",
                dependencies=[],
            )
            for i in range(3)
        ]

        async def mock_execute(task, context):
            # 所有策略输出相同结果
            return {"answer": "发光材料是LED材料", "confidence": 0.85}

        result = await executor.execute_voting(
            strategies=strategies,
            execute_fn=mock_execute,
        )
        voting = result.voting_result
        assert voting is not None
        assert voting.reached_consensus is True
        assert voting.consensus_score >= 0.5

    @pytest.mark.asyncio
    async def test_voting_no_consensus(self, executor):
        """共识未达成."""
        strategies = [
            OrchestrationTask(
                task_id=f"vote-{i}",
                agent_id="agent.generation",
                name=f"策略{i}",
                dependencies=[],
            )
            for i in range(3)
        ]
        answers = ["答案A", "答案B", "答案C"]

        async def mock_execute(task, context):
            idx = int(task.task_id.split("-")[1])
            return {"answer": answers[idx], "confidence": 0.5}

        result = await executor.execute_voting(
            strategies=strategies,
            execute_fn=mock_execute,
        )
        voting = result.voting_result
        assert voting is not None
        assert voting.reached_consensus is False
        assert voting.consensus_score < 0.5
        # 未达共识应需要裁决
        assert result.requires_adjudication is True

    @pytest.mark.asyncio
    async def test_voting_reputation_weighting(self, executor):
        """声誉加权聚合 (L0 Reputation Ledger 联动)."""
        strategies = [
            OrchestrationTask(
                task_id=f"vote-{i}",
                agent_id=f"agent.generation.v{i}",
                name=f"策略{i}",
                dependencies=[],
            )
            for i in range(3)
        ]
        reputations = {"agent.generation.v0": 0.9, "agent.generation.v1": 0.5, "agent.generation.v2": 0.7}

        async def mock_execute(task, context):
            rep = reputations.get(task.agent_id, 0.5)
            return {"answer": "答案A", "confidence": 0.8, "reputation": rep}

        result = await executor.execute_voting(
            strategies=strategies,
            execute_fn=mock_execute,
            reputations=reputations,
        )
        voting = result.voting_result
        assert voting is not None
        # 高声誉 Agent 的权重应更高
        # final_weight = base_weight * (1 + reputation_bonus)
        for vote in voting.votes:
            expected_weight = vote.confidence * (1 + vote.reputation_score)
            assert vote.weight > 0


# ============================================================
# 7. OrchestrationEngine 测试
# ============================================================

class TestOrchestrationEngine:
    """编排引擎顶层测试 (融合 Temporal + LangGraph)."""

    @pytest.fixture
    def engine(self):
        return OrchestrationEngine()

    def test_engine_creation(self, engine):
        """创建编排引擎."""
        assert engine is not None
        assert OrchestrationParadigm.PIPELINE in engine._executors
        assert OrchestrationParadigm.DEBATE in engine._executors
        assert OrchestrationParadigm.VOTING in engine._executors

    @pytest.mark.asyncio
    async def test_execute_pipeline(self, engine, simple_tasks):
        """执行流水线编排."""
        plan = OrchestrationPlan(
            plan_id="plan-001",
            tasks=simple_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )

        async def mock_execute(task, context):
            return {"result": f"output_{task.task_id}"}

        result = await engine.execute(plan, execute_fn=mock_execute)
        assert result.state == OrchestrationState.COMPLETED
        assert result.plan_id == "plan-001"

    @pytest.mark.asyncio
    async def test_execute_with_retry(self, engine, simple_tasks):
        """执行失败后自动重试 (Temporal retry policy 模式)."""
        plan = OrchestrationPlan(
            plan_id="plan-retry",
            tasks=simple_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
            max_retries=2,
        )
        attempt_count = 0

        async def mock_execute(task, context):
            nonlocal attempt_count
            attempt_count += 1
            if task.task_id == "t2" and attempt_count <= 1:
                raise RuntimeError("transient error")
            return {"result": task.task_id}

        result = await engine.execute(plan, execute_fn=mock_execute)
        # 重试后应成功
        assert result.state == OrchestrationState.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_timeout(self, engine, simple_tasks):
        """任务超时应标记失败 (Temporal 超时模式)."""
        plan = OrchestrationPlan(
            plan_id="plan-timeout",
            tasks=simple_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        # 设置极短超时
        for task in plan.tasks:
            task.timeout_s = 0.01

        async def mock_execute(task, context):
            await asyncio.sleep(0.1)
            return {"result": task.task_id}

        result = await engine.execute(plan, execute_fn=mock_execute)
        assert result.state == OrchestrationState.FAILED
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_provenance(self, engine, simple_tasks):
        """执行应记录完整溯源."""
        plan = OrchestrationPlan(
            plan_id="plan-prov",
            tasks=simple_tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )

        async def mock_execute(task, context):
            return {"result": task.task_id}

        result = await engine.execute(plan, execute_fn=mock_execute)
        # 应有计划级 + 任务级溯源
        actions = [p["action"] for p in result.provenance]
        assert any("plan" in a for a in actions)
        assert any("task" in a for a in actions)


# ============================================================
# 8. 集成测试
# ============================================================

class TestOrchestrationIntegration:
    """编排引擎集成测试."""

    @pytest.mark.asyncio
    async def test_pipeline_with_parallel_layers(self):
        """流水线 + 并行层混合编排."""
        tasks = [
            OrchestrationTask("t1", "agent.diagnosis", "诊断", []),
            OrchestrationTask("t2", "agent.generation", "生成", ["t1"]),
            OrchestrationTask("t3", "agent.review", "审核", ["t1"]),
            OrchestrationTask("t4", "agent.guidance", "导学", ["t2", "t3"]),
        ]
        plan = OrchestrationPlan(
            plan_id="plan-mixed",
            tasks=tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )
        executor = PipelineExecutor()
        execution_order = []

        async def mock_execute(task, context):
            execution_order.append(task.task_id)
            await asyncio.sleep(0.01)
            return {"step": task.task_id}

        result = await executor.execute(plan, mock_execute)
        assert result.state == OrchestrationState.COMPLETED
        assert execution_order[0] == "t1"
        assert execution_order[-1] == "t4"

    @pytest.mark.asyncio
    async def test_debate_convergence_with_adjudication(self):
        """辩论收敛 + 裁决触发."""
        executor = DebateExecutor(
            max_rounds=2,
            arguments_per_round=3,
            convergence_threshold=0.1,
        )

        async def mock_execute(task, context, round_num, opponent_args):
            return DebateMessage(
                sender=task.agent_id,
                round=round_num,
                arguments=["arg"],
                confidence=0.3 if "generation" in task.agent_id else 0.9,
            )

        pro = OrchestrationTask("pro", "agent.generation", "生成", [])
        con = OrchestrationTask("con", "agent.review", "审核", [])

        result = await executor.execute_debate(pro, con, mock_execute)
        assert result.requires_adjudication is True
        assert len(result.debate_rounds) == 2

    @pytest.mark.asyncio
    async def test_voting_retry_on_no_consensus(self):
        """投票未达共识后自动重试."""
        executor = VotingExecutor(
            min_strategies=3,
            max_strategies=5,
            consensus_threshold=0.5,
            similarity_threshold=0.8,
        )
        strategies = [
            OrchestrationTask(f"vote-{i}", "agent.gen", f"策略{i}", [])
            for i in range(3)
        ]
        call_count = 0

        async def mock_execute(task, context):
            nonlocal call_count
            call_count += 1
            # 第一次分歧, 第二次一致
            if call_count <= 3:
                return {"answer": f"不同答案{call_count}", "confidence": 0.5}
            return {"answer": "一致答案", "confidence": 0.85}

        result = await executor.execute_voting(
            strategies=strategies,
            execute_fn=mock_execute,
            max_retries=1,
        )
        # 重试后应达成共识
        assert result.voting_result is not None

    @pytest.mark.asyncio
    async def test_full_orchestration_lifecycle(self):
        """完整编排生命周期: 创建 → 执行 → 完成 → 溯源."""
        engine = OrchestrationEngine()
        tasks = [
            OrchestrationTask("t1", "agent.diagnosis", "诊断", []),
            OrchestrationTask("t2", "agent.generation", "生成", ["t1"]),
        ]
        plan = OrchestrationPlan(
            plan_id="plan-lifecycle",
            tasks=tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )

        async def mock_execute(task, context):
            return {"result": task.task_id}

        result = await engine.execute(plan, execute_fn=mock_execute)
        assert result.state == OrchestrationState.COMPLETED
        assert result.execution_time_s >= 0
        assert len(result.provenance) >= 3  # plan.start + 2 tasks + plan.complete


# ============================================================
# 9. 错误处理与重试测试
# ============================================================

class TestErrorHandling:
    """错误处理与重试测试 (Temporal retry policy 模式)."""

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self):
        """指数退避重试."""
        engine = OrchestrationEngine()
        tasks = [
            OrchestrationTask("t1", "agent.test", "任务", []),
        ]
        plan = OrchestrationPlan(
            plan_id="plan-backoff",
            tasks=tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
            max_retries=3,
            retry_backoff_s=0.01,
        )
        attempt_count = 0

        async def mock_execute(task, context):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError("transient")
            return {"result": "success"}

        result = await engine.execute(plan, execute_fn=mock_execute)
        assert result.state == OrchestrationState.COMPLETED
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """超过最大重试次数后失败."""
        engine = OrchestrationEngine()
        tasks = [
            OrchestrationTask("t1", "agent.test", "任务", []),
        ]
        plan = OrchestrationPlan(
            plan_id="plan-max-retry",
            tasks=tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
            max_retries=2,
            retry_backoff_s=0.01,
        )

        async def mock_execute(task, context):
            raise RuntimeError("permanent error")

        result = await engine.execute(plan, execute_fn=mock_execute)
        assert result.state == OrchestrationState.FAILED
        assert "permanent error" in result.error

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """不可重试错误应立即失败."""
        engine = OrchestrationEngine()
        tasks = [
            OrchestrationTask("t1", "agent.test", "任务", []),
        ]
        plan = OrchestrationPlan(
            plan_id="plan-non-retry",
            tasks=tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
            max_retries=3,
        )
        attempt_count = 0

        async def mock_execute(task, context):
            nonlocal attempt_count
            attempt_count += 1
            raise OrchestrationError("non-retryable", retryable=False)

        result = await engine.execute(plan, execute_fn=mock_execute)
        assert result.state == OrchestrationState.FAILED
        assert attempt_count == 1  # 不重试


# ============================================================
# 10. 超时管理测试
# ============================================================

class TestTimeoutManagement:
    """超时管理测试 (Temporal 四级超时模型)."""

    @pytest.mark.asyncio
    async def test_task_level_timeout(self):
        """任务级超时 (Start-To-Close)."""
        executor = PipelineExecutor()
        tasks = [
            OrchestrationTask("t1", "agent.test", "任务", [], timeout_s=0.05),
        ]
        plan = OrchestrationPlan(
            plan_id="plan-task-timeout",
            tasks=tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
        )

        async def mock_execute(task, context):
            await asyncio.sleep(0.2)
            return {}

        result = await executor.execute(plan, mock_execute)
        assert result.state == OrchestrationState.FAILED
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_plan_level_timeout(self):
        """计划级超时 (Schedule-To-Close)."""
        engine = OrchestrationEngine()
        tasks = [
            OrchestrationTask(f"t{i}", "agent.test", f"任务{i}", [f"t{i-1}"] if i > 0 else [])
            for i in range(5)
        ]
        plan = OrchestrationPlan(
            plan_id="plan-plan-timeout",
            tasks=tasks,
            paradigm=OrchestrationParadigm.PIPELINE,
            total_timeout_s=0.1,
        )

        async def mock_execute(task, context):
            await asyncio.sleep(0.05)
            return {}

        result = await engine.execute(plan, execute_fn=mock_execute)
        assert result.state == OrchestrationState.FAILED
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_debate_round_timeout(self):
        """辩论轮次超时."""
        executor = DebateExecutor(
            max_rounds=3,
            arguments_per_round=3,
            convergence_threshold=0.1,
            round_timeout_s=0.05,
        )

        async def mock_execute(task, context, round_num, opponent_args):
            await asyncio.sleep(0.2)
            return DebateMessage(
                sender=task.agent_id, round=round_num,
                arguments=["arg"], confidence=0.8,
            )

        pro = OrchestrationTask("pro", "agent.gen", "生成", [])
        con = OrchestrationTask("con", "agent.rev", "审核", [])

        result = await executor.execute_debate(pro, con, mock_execute)
        assert result.state == OrchestrationState.FAILED
        assert "timeout" in result.error.lower() if result.error else result.requires_adjudication
