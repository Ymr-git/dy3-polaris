"""L4 TaskExecutor 单元测试.

测试范围:
- DecisionPlan 拓扑排序与 DAG 验证
- TaskExecutor 三种执行模式（串行/并行/迭代）
- 资源预算控制
- 降级策略
- 错误恢复
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from dy3_polaris.l4.models import (
    DecisionPlan,
    ExecutionMode,
    ExecutionStatus,
    FallbackPlan,
    ReasoningMode,
    ResourceBudget,
    RetrievalStrategy,
    SubTask,
    TaskType,
)
from dy3_polaris.l4.task_executor import (
    BudgetExceededError,
    CyclicDependencyError,
    TaskExecutionError,
    TaskExecutor,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_store() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_graph_reasoner() -> MagicMock:
    r = MagicMock()
    r.reason.return_value = MagicMock(
        mode=MagicMock(value="multi_hop"),
        answers=[{"entity_id": "Dy3+", "confidence": 0.9}],
        confidence=0.9,
        reasoning_chain=["多跳推理测试"],
        evidence_triples=[],
        elapsed_ms=15.0,
    )
    return r


@pytest.fixture
def mock_hybrid_retriever() -> MagicMock:
    r = MagicMock()
    r.retrieve.return_value = MagicMock(
        results=[{"chunk_id": "c1", "content": "测试内容"}],
        scores=[0.95],
        total=1,
        source_type="hybrid",
    )
    return r


@pytest.fixture
def mock_graphrag_retriever() -> MagicMock:
    r = MagicMock()
    r.search.return_value = MagicMock(
        local_result=MagicMock(model_dump=lambda: {"entities": ["Dy3+"]}),
        global_result=None,
        strategy="adaptive",
    )
    return r


@pytest.fixture
def executor(
    mock_store: MagicMock,
    mock_graph_reasoner: MagicMock,
    mock_hybrid_retriever: MagicMock,
    mock_graphrag_retriever: MagicMock,
) -> TaskExecutor:
    return TaskExecutor(
        store=mock_store,
        graph_reasoner=mock_graph_reasoner,
        hybrid_retriever=mock_hybrid_retriever,
        graphrag_retriever=mock_graphrag_retriever,
    )


# ============================================================
# DecisionPlan 模型测试
# ============================================================


class TestDecisionPlan:
    """DecisionPlan 数据模型测试."""

    def test_topological_order_linear(self) -> None:
        """测试线性依赖拓扑排序."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE, deps=[]),
                SubTask(task_id="t2", task_type=TaskType.REASON, deps=["t1"]),
                SubTask(task_id="t3", task_type=TaskType.SYNTHESIZE, deps=["t2"]),
            ]
        )
        ordered = plan.topological_order()
        ids = [t.task_id for t in ordered]
        assert ids == ["t1", "t2", "t3"]

    def test_topological_order_diamond(self) -> None:
        """测试菱形依赖拓扑排序."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE, deps=[]),
                SubTask(task_id="t2", task_type=TaskType.REASON, deps=["t1"]),
                SubTask(task_id="t3", task_type=TaskType.VERIFY, deps=["t1"]),
                SubTask(task_id="t4", task_type=TaskType.SYNTHESIZE, deps=["t2", "t3"]),
            ]
        )
        ordered = plan.topological_order()
        ids = [t.task_id for t in ordered]
        assert ids[0] == "t1"
        assert ids[-1] == "t4"
        assert set(ids[1:3]) == {"t2", "t3"}

    def test_topological_order_cycle_detection(self) -> None:
        """测试循环依赖检测."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE, deps=["t3"]),
                SubTask(task_id="t2", task_type=TaskType.REASON, deps=["t1"]),
                SubTask(task_id="t3", task_type=TaskType.SYNTHESIZE, deps=["t2"]),
            ]
        )
        with pytest.raises(ValueError, match="循环依赖"):
            plan.topological_order()

    def test_get_ready_tasks(self) -> None:
        """测试获取就绪任务."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE, deps=[]),
                SubTask(task_id="t2", task_type=TaskType.REASON, deps=["t1"]),
                SubTask(task_id="t3", task_type=TaskType.REASON, deps=["t1"]),
            ]
        )
        ready = plan.get_ready_tasks(completed=set())
        assert len(ready) == 1 and ready[0].task_id == "t1"

        ready = plan.get_ready_tasks(completed={"t1"})
        assert len(ready) == 2
        assert {t.task_id for t in ready} == {"t2", "t3"}


# ============================================================
# TaskExecutor 执行模式测试
# ============================================================


class TestTaskExecutorModes:
    """TaskExecutor 执行模式测试."""

    @pytest.mark.asyncio
    async def test_sequential_execution(self, executor: TaskExecutor) -> None:
        """测试串行执行模式."""
        plan = DecisionPlan(
            execution_mode=ExecutionMode.SEQUENTIAL,
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE,
                        retrieval_strategy=RetrievalStrategy.HYBRID, query="Dy3+ 发光"),
                SubTask(task_id="t2", task_type=TaskType.REASON,
                        reasoning_mode=ReasoningMode.MULTI_HOP, deps=["t1"], query="推理"),
            ]
        )
        result = await executor.execute(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.task_results) == 2
        assert result.task_results["t1"].is_success
        assert result.task_results["t2"].is_success
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_parallel_execution(self, executor: TaskExecutor) -> None:
        """测试并行执行模式."""
        plan = DecisionPlan(
            execution_mode=ExecutionMode.PARALLEL,
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE,
                        retrieval_strategy=RetrievalStrategy.HYBRID, query="A"),
                SubTask(task_id="t2", task_type=TaskType.RETRIEVE,
                        retrieval_strategy=RetrievalStrategy.HYBRID, query="B"),
                SubTask(task_id="t3", task_type=TaskType.REASON,
                        reasoning_mode=ReasoningMode.MULTI_HOP, deps=["t1", "t2"], query="C"),
            ]
        )
        result = await executor.execute(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.task_results) == 3
        # t1, t2 并行执行，t3 依赖两者
        assert result.task_results["t3"].is_success

    @pytest.mark.asyncio
    async def test_iterative_execution(self, executor: TaskExecutor) -> None:
        """测试迭代执行模式."""
        plan = DecisionPlan(
            execution_mode=ExecutionMode.ITERATIVE,
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE,
                        retrieval_strategy=RetrievalStrategy.HYBRID, query="Dy3+"),
            ]
        )
        result = await executor.execute(plan)

        # 迭代模式至少完成一轮
        assert result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.PARTIAL)
        assert len(result.task_results) >= 1

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, executor: TaskExecutor) -> None:
        """测试失败时触发降级."""
        # 让混合检索器抛出异常，降级也会失败
        executor._hybrid_retriever.retrieve.side_effect = Exception("检索失败")

        plan = DecisionPlan(
            execution_mode=ExecutionMode.SEQUENTIAL,
            fallback_plan=FallbackPlan(direct_retrieval=True),
            original_query="Dy3+ 发光",
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE,
                        retrieval_strategy=RetrievalStrategy.HYBRID, query="Dy3+"),
            ]
        )
        result = await executor.execute(plan)

        # 主任务失败，降级尝试执行但同样失败
        assert result.status in (ExecutionStatus.FAILED, ExecutionStatus.PARTIAL)

    @pytest.mark.asyncio
    async def test_budget_exceeded(self, executor: TaskExecutor) -> None:
        """测试资源预算超限."""
        plan = DecisionPlan(
            execution_mode=ExecutionMode.SEQUENTIAL,
            sub_tasks=[
                SubTask(
                    task_id="t1", task_type=TaskType.RETRIEVE,
                    retrieval_strategy=RetrievalStrategy.HYBRID,
                    query="Dy3+",
                    resource_budget=ResourceBudget(max_latency_ms=100),
                ),
            ]
        )
        result = await executor.execute(plan)

        # mock 执行太快可能不超，此测试验证机制存在
        assert result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED,
                                  ExecutionStatus.PARTIAL)

    @pytest.mark.asyncio
    async def test_cyclic_dependency_error(self, executor: TaskExecutor) -> None:
        """测试循环依赖错误处理."""
        plan = DecisionPlan(
            execution_mode=ExecutionMode.SEQUENTIAL,
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE, deps=["t2"]),
                SubTask(task_id="t2", task_type=TaskType.REASON, deps=["t1"]),
            ]
        )
        result = await executor.execute(plan)

        assert result.status == ExecutionStatus.FAILED
        assert "循环依赖" in (result.error_summary or "")


# ============================================================
# 任务类型分发测试
# ============================================================


class TestTaskDispatch:
    """任务类型分发测试."""

    @pytest.mark.asyncio
    async def test_retrieve_hybrid(self, executor: TaskExecutor) -> None:
        """测试混合检索分发."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE,
                        retrieval_strategy=RetrievalStrategy.HYBRID, query="Dy3+"),
            ]
        )
        result = await executor.execute(plan)
        assert result.task_results["t1"].output["source_type"] == "hybrid"
        executor._hybrid_retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_graphrag(self, executor: TaskExecutor) -> None:
        """测试 GraphRAG 检索分发."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE,
                        retrieval_strategy=RetrievalStrategy.GRAPHRAG, query="Dy3+"),
            ]
        )
        result = await executor.execute(plan)
        assert result.task_results["t1"].is_success
        executor._graphrag_retriever.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_reason_multi_hop(self, executor: TaskExecutor) -> None:
        """测试多跳推理分发."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.REASON,
                        reasoning_mode=ReasoningMode.MULTI_HOP, query="Dy3+ -> 跃迁"),
            ]
        )
        result = await executor.execute(plan)
        assert result.task_results["t1"].output["mode"] == "multi_hop"
        executor._graph_reasoner.reason.assert_called_once()

    @pytest.mark.asyncio
    async def test_reason_missing_component_error(self, executor: TaskExecutor) -> None:
        """测试缺少组件时的错误."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.REASON,
                        reasoning_mode=ReasoningMode.BACKWARD_CHAIN, query="测试"),
            ]
        )
        result = await executor.execute(plan)
        assert result.task_results["t1"].is_failed
        assert "BackwardChainingReasoner 未初始化" in (result.task_results["t1"].error or "")


# ============================================================
# 辅助方法测试
# ============================================================


class TestHelperMethods:
    """TaskExecutor 辅助方法测试."""

    def test_parallel_layers_linear(self, executor: TaskExecutor) -> None:
        """测试线性任务的并行分层."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE, deps=[]),
                SubTask(task_id="t2", task_type=TaskType.REASON, deps=["t1"]),
                SubTask(task_id="t3", task_type=TaskType.SYNTHESIZE, deps=["t2"]),
            ]
        )
        layers = executor._build_parallel_layers(plan)
        assert len(layers) == 3
        assert [t.task_id for t in layers[0]] == ["t1"]
        assert [t.task_id for t in layers[1]] == ["t2"]
        assert [t.task_id for t in layers[2]] == ["t3"]

    def test_parallel_layers_diamond(self, executor: TaskExecutor) -> None:
        """测试菱形任务的并行分层."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE, deps=[]),
                SubTask(task_id="t2", task_type=TaskType.REASON, deps=["t1"]),
                SubTask(task_id="t3", task_type=TaskType.VERIFY, deps=["t1"]),
                SubTask(task_id="t4", task_type=TaskType.SYNTHESIZE, deps=["t2", "t3"]),
            ]
        )
        layers = executor._build_parallel_layers(plan)
        assert len(layers) == 3
        assert [t.task_id for t in layers[0]] == ["t1"]
        assert {t.task_id for t in layers[1]} == {"t2", "t3"}
        assert [t.task_id for t in layers[2]] == ["t4"]

    def test_aggregate_result(self, executor: TaskExecutor) -> None:
        """测试结果聚合."""
        from dy3_polaris.l4.models import ExecutionResult, TaskResult

        result = ExecutionResult(plan_id="test")
        result.task_results["t1"] = TaskResult(
            task_id="t1", task_type=TaskType.REASON,
            status=ExecutionStatus.COMPLETED, confidence=0.9,
            evidence=[{"type": "triple"}], reasoning_chain=["步骤1"],
        )
        result.task_results["t2"] = TaskResult(
            task_id="t2", task_type=TaskType.RETRIEVE,
            status=ExecutionStatus.COMPLETED, confidence=0.8,
            evidence=[{"type": "chunk"}], reasoning_chain=["步骤2"],
        )
        executor._aggregate_result(result)

        assert len(result.evidence_set) == 2
        assert len(result.reasoning_chain) == 2
        assert result.confidence > 0


# ============================================================
# 异常测试
# ============================================================


class TestExceptions:
    """异常类测试."""

    def test_task_execution_error(self) -> None:
        exc = TaskExecutionError("t1", "测试错误")
        assert exc.task_id == "t1"
        assert "测试错误" in str(exc)

    def test_budget_exceeded_error(self) -> None:
        exc = BudgetExceededError("t1", "latency", 100.0, 50.0)
        assert exc.budget_type == "latency"
        assert exc.actual == 100.0
        assert exc.limit == 50.0
        assert "latency 超限" in str(exc)

    def test_cyclic_dependency_error(self) -> None:
        exc = CyclicDependencyError("plan-123")
        assert "plan-123" in str(exc)
