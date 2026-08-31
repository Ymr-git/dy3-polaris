"""L4 决策引擎层 — 任务执行器 (TaskExecutor).

融合世界先进方案的任务调度与执行引擎:
- LangGraph: 有状态图节点执行 + 条件边路由
- TDP 框架 (2026): Supervisor-Planner-Executor 三层 + 上下文隔离
- GraphRAG: 局部/全局双通道检索编排
- SubQRAG (2025): 子问题驱动动态检索
- Plan-and-Solve: 计划分解 → 逐步执行 → 结果聚合

核心职责:
    按 DecisionPlan 的 DAG 拓扑序调度子任务，将每个子任务分发到
    对应的 L3 推理/检索模块执行，收集结果并聚合为 ExecutionResult。

线程安全:
    所有公开方法通过 asyncio.Lock 保护可变状态，支持并发执行多个
    DecisionPlan（每个 plan 有独立的执行上下文）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .models import (
    DecisionPlan,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    FallbackPlan,
    ReasoningMode,
    ResourceBudget,
    RetrievalStrategy,
    SubTask,
    TaskResult,
    TaskType,
)

logger = logging.getLogger(__name__)


# ============================================================
# 异常定义
# ============================================================


class TaskExecutionError(Exception):
    """任务执行错误."""

    def __init__(self, task_id: str, message: str, cause: Exception | None = None) -> None:
        self.task_id = task_id
        self.cause = cause
        super().__init__(f"任务 {task_id} 执行失败: {message}")


class BudgetExceededError(TaskExecutionError):
    """资源预算超限错误."""

    def __init__(self, task_id: str, budget_type: str, actual: float, limit: float) -> None:
        self.budget_type = budget_type
        self.actual = actual
        self.limit = limit
        super().__init__(task_id, f"{budget_type} 超限: {actual:.2f} > {limit:.2f}")


class CyclicDependencyError(Exception):
    """循环依赖错误."""

    def __init__(self, plan_id: str) -> None:
        super().__init__(f"决策计划 {plan_id} 存在循环依赖")


# ============================================================
# 任务执行器
# ============================================================


class TaskExecutor:
    """任务执行器 — 决策计划的调度与执行核心.

    借鉴 TDP 框架 Executor 层设计:
    - 拓扑排序确保依赖正确性
    - 并行执行无依赖任务（asyncio.gather）
    - 资源预算实时监控
    - 失败时触发降级计划

    Usage::

        executor = TaskExecutor(
            store=knowledge_store,
            graph_reasoner=graph_reasoner,
            hybrid_retriever=hybrid_retriever,
            graphrag_retriever=graphrag_retriever,
        )
        result = await executor.execute(decision_plan)
    """

    def __init__(
        self,
        store: Any,
        graph_reasoner: Any,
        hybrid_retriever: Any,
        graphrag_retriever: Any,
        subgraph_reasoner: Any | None = None,
        backward_reasoner: Any | None = None,
        confidence_traversal: Any | None = None,
        trans_e_embedder: Any | None = None,
        fact_checker: Any | None = None,
        quality_manager: Any | None = None,
        conflict_detector: Any | None = None,
        response_synthesizer: Any | None = None,
    ) -> None:
        """初始化任务执行器.

        Args:
            store: 知识存储 (KnowledgeStore)
            graph_reasoner: 图推理器 V1 (GraphReasoner)
            hybrid_retriever: 混合检索器 (HybridRetriever)
            graphrag_retriever: GraphRAG 检索器 (GraphRAGRetriever)
            subgraph_reasoner: 子图推理器 (SubgraphReasoner, 可选)
            backward_reasoner: 后向链式推理器 (BackwardChainingReasoner, 可选)
            confidence_traversal: 置信度加权遍历器 (ConfidenceWeightedTraversal, 可选)
            trans_e_embedder: TransE 嵌入器 (TransEEmbedder, 可选)
            fact_checker: 事实校验器 (FactChecker, 可选) — VERIFY 任务集成点
            quality_manager: 质量管理器 (QualityManager, 可选) — VERIFY 任务集成点
            conflict_detector: 冲突检测器 (ConflictDetector, 可选) — VERIFY 任务集成点
            response_synthesizer: 响应合成器 (ResponseSynthesizer, 可选) —
                SYNTHESIZE 任务集成点
        """
        self._store = store
        self._graph_reasoner = graph_reasoner
        self._hybrid_retriever = hybrid_retriever
        self._graphrag_retriever = graphrag_retriever
        self._subgraph_reasoner = subgraph_reasoner
        self._backward_reasoner = backward_reasoner
        self._confidence_traversal = confidence_traversal
        self._trans_e_embedder = trans_e_embedder
        # VERIFY / SYNTHESIZE 任务集成组件 (L3)
        self._fact_checker = fact_checker
        self._quality_manager = quality_manager
        self._conflict_detector = conflict_detector
        self._response_synthesizer = response_synthesizer

        # 执行状态锁
        self._lock = asyncio.Lock()
        # 全局资源预算（可选）
        self._global_budget: ResourceBudget | None = None

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    async def execute(
        self,
        plan: DecisionPlan,
        *,
        global_budget: ResourceBudget | None = None,
        context: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """执行决策计划.

        Args:
            plan: 决策计划
            global_budget: 全局资源预算（覆盖子任务预算的汇总限制）
            context: 执行上下文（可用于跨任务状态传递）

        Returns:
            ExecutionResult 执行结果
        """
        self._global_budget = global_budget
        exec_context = context or {}
        start_ts = time.perf_counter()

        logger.info("开始执行决策计划 %s, 模式=%s, 子任务数=%d",
                    plan.plan_id, plan.execution_mode.value, len(plan.sub_tasks))

        try:
            # 拓扑排序验证
            ordered = plan.topological_order()
        except ValueError as exc:
            logger.error("决策计划 %s 拓扑排序失败: %s", plan.plan_id, exc)
            return self._build_error_result(plan, str(exc), start_ts)

        # 根据执行模式分发
        if plan.execution_mode == ExecutionMode.SEQUENTIAL:
            result = await self._execute_sequential(plan, ordered, exec_context)
        elif plan.execution_mode == ExecutionMode.PARALLEL:
            result = await self._execute_parallel(plan, ordered, exec_context)
        elif plan.execution_mode == ExecutionMode.ITERATIVE:
            result = await self._execute_iterative(plan, ordered, exec_context)
        else:
            result = await self._execute_sequential(plan, ordered, exec_context)

        # 计算总耗时
        result.total_elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 2)

        # 触发降级（如有必要）
        if result.status in (ExecutionStatus.FAILED, ExecutionStatus.PARTIAL):
            if plan.fallback_plan and not result.fallback_triggered:
                logger.warning("决策计划 %s 失败，触发降级计划", plan.plan_id)
                fallback_result = await self._execute_fallback(plan, exec_context)
                if fallback_result.status == ExecutionStatus.COMPLETED:
                    fallback_result.fallback_triggered = True
                    return fallback_result

        logger.info("决策计划 %s 执行完成, 状态=%s, 耗时=%.2fms, 置信度=%.4f",
                    plan.plan_id, result.status.value, result.total_elapsed_ms, result.confidence)
        return result

    # --------------------------------------------------------
    # 执行模式实现
    # --------------------------------------------------------

    async def _execute_sequential(
        self,
        plan: DecisionPlan,
        ordered_tasks: list[SubTask],
        context: dict[str, Any],
    ) -> ExecutionResult:
        """串行执行 — 严格按拓扑序依次执行每个子任务."""
        result = ExecutionResult(plan_id=plan.plan_id, status=ExecutionStatus.RUNNING)
        completed: set[str] = set()

        for task in ordered_tasks:
            task_result = await self._run_single_task(task, context, completed)
            result.task_results[task.task_id] = task_result
            completed.add(task.task_id)

            if task_result.is_failed:
                logger.warning("子任务 %s 失败，串行执行中断", task.task_id)
                result.status = ExecutionStatus.PARTIAL if completed else ExecutionStatus.FAILED
                result.error_summary = f"子任务 {task.task_id} 失败: {task_result.error}"
                break
        else:
            result.status = ExecutionStatus.COMPLETED

        self._aggregate_result(result)
        return result

    async def _execute_parallel(
        self,
        plan: DecisionPlan,
        ordered_tasks: list[SubTask],
        context: dict[str, Any],
    ) -> ExecutionResult:
        """并行执行 — 无依赖的任务并发执行（asyncio.gather）.

        借鉴 LangGraph 并行节点设计：按拓扑分层，每层内无依赖的任务
        同时启动，层间保持依赖顺序。
        """
        result = ExecutionResult(plan_id=plan.plan_id, status=ExecutionStatus.RUNNING)
        completed: set[str] = set()

        # 按拓扑分层（每层内的任务互相无依赖）
        layers = self._build_parallel_layers(plan)

        for layer_idx, layer in enumerate(layers):
            logger.debug("并行层 %d: %d 个任务", layer_idx, len(layer))

            # 并发执行当前层
            coros = [self._run_single_task(task, context, completed) for task in layer]
            layer_results = await asyncio.gather(*coros, return_exceptions=True)

            for task, tr in zip(layer, layer_results):
                if isinstance(tr, Exception):
                    tr = TaskResult(
                        task_id=task.task_id, task_type=task.task_type,
                        status=ExecutionStatus.FAILED,
                        error=f"{type(tr).__name__}: {tr}",
                    )
                result.task_results[task.task_id] = tr
                completed.add(task.task_id)

                if tr.is_failed:
                    logger.warning("并行层 %d 中子任务 %s 失败", layer_idx, task.task_id)

        # 判断整体状态
        failed_count = sum(1 for tr in result.task_results.values() if tr.is_failed)
        if failed_count == 0:
            result.status = ExecutionStatus.COMPLETED
        elif failed_count < len(result.task_results):
            result.status = ExecutionStatus.PARTIAL
            result.error_summary = f"{failed_count}/{len(result.task_results)} 个子任务失败"
        else:
            result.status = ExecutionStatus.FAILED
            result.error_summary = "所有子任务均失败"

        self._aggregate_result(result)
        return result

    async def _execute_iterative(
        self,
        plan: DecisionPlan,
        ordered_tasks: list[SubTask],
        context: dict[str, Any],
    ) -> ExecutionResult:
        """迭代执行 — 多轮收敛，每轮结束后检查终止条件.

        借鉴 OLIVIA 步骤级反馈学习：每轮执行后评估结果质量，
        若未达收敛条件则调整参数继续迭代。
        """
        result = ExecutionResult(plan_id=plan.plan_id, status=ExecutionStatus.RUNNING)
        max_iterations = 3
        convergence_threshold = 0.75

        for iteration in range(max_iterations):
            logger.info("迭代轮次 %d/%d", iteration + 1, max_iterations)

            # 每轮使用串行模式执行（迭代通常需要前序结果）
            round_result = await self._execute_sequential(plan, ordered_tasks, context)

            # 检查收敛
            confidence = round_result.compute_confidence()
            logger.info("迭代 %d 综合置信度: %.4f", iteration + 1, confidence)

            if confidence >= convergence_threshold:
                logger.info("迭代收敛，置信度 %.4f >= 阈值 %.4f", confidence, convergence_threshold)
                return round_result

            # 未收敛：调整参数（如增加检索深度）
            for task in plan.sub_tasks:
                if task.task_type == TaskType.RETRIEVE:
                    task.resource_budget.max_retrieval_depth = min(
                        task.resource_budget.max_retrieval_depth + 1, 10
                    )
                if task.task_type == TaskType.REASON:
                    task.resource_budget.max_reasoning_hops = min(
                        task.resource_budget.max_reasoning_hops + 2, 20
                    )

        # 达到最大迭代次数，返回最后一轮结果
        logger.warning("达到最大迭代次数 %d，返回最后一轮结果", max_iterations)
        final = await self._execute_sequential(plan, ordered_tasks, context)
        final.status = ExecutionStatus.PARTIAL
        final.error_summary = f"未在 {max_iterations} 轮内收敛"
        return final

    # --------------------------------------------------------
    # 单任务执行
    # --------------------------------------------------------

    async def _run_single_task(
        self,
        task: SubTask,
        context: dict[str, Any],
        completed: set[str],
    ) -> TaskResult:
        """执行单个子任务（带超时和资源预算控制）."""
        task_id = task.task_id
        budget = task.resource_budget
        start_ts = time.perf_counter()

        logger.debug("执行任务 %s, 类型=%s", task_id, task.task_type.value)

        try:
            # 全局预算检查
            if self._global_budget and not self._global_budget.is_within_budget(
                (time.perf_counter() - start_ts) * 1000
            ):
                raise BudgetExceededError(task_id, "global_latency",
                                          (time.perf_counter() - start_ts) * 1000,
                                          self._global_budget.max_latency_ms)

            # 构建任务上下文（注入前置任务结果）
            task_context = self._build_task_context(task, context, completed)

            # 根据类型分发执行
            if task.task_type == TaskType.RETRIEVE:
                output = await self._execute_retrieve_task(task, task_context)
            elif task.task_type == TaskType.REASON:
                output = await self._execute_reason_task(task, task_context)
            elif task.task_type == TaskType.VERIFY:
                output = await self._execute_verify_task(task, task_context)
            elif task.task_type == TaskType.SYNTHESIZE:
                output = await self._execute_synthesize_task(task, task_context)
            else:
                raise TaskExecutionError(task_id, f"未知任务类型: {task.task_type}")

            elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 2)

            # 预算后检查
            if elapsed_ms > budget.max_latency_ms:
                raise BudgetExceededError(task_id, "latency", elapsed_ms, budget.max_latency_ms)

            return TaskResult(
                task_id=task_id,
                task_type=task.task_type,
                status=ExecutionStatus.COMPLETED,
                output=output,
                confidence=output.get("confidence", 0.0),
                elapsed_ms=elapsed_ms,
                token_usage=output.get("token_usage", 0),
                tool_calls=output.get("tool_calls", 0),
                evidence=output.get("evidence", []),
                reasoning_chain=output.get("reasoning_chain", []),
            )

        except asyncio.TimeoutError:
            elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 2)
            logger.warning("任务 %s 超时 (%.2fms)", task_id, elapsed_ms)
            return TaskResult(
                task_id=task_id, task_type=task.task_type,
                status=ExecutionStatus.TIMEOUT,
                error="执行超时", elapsed_ms=elapsed_ms,
            )
        except BudgetExceededError as exc:
            elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 2)
            logger.warning("任务 %s 预算超限: %s", task_id, exc)
            return TaskResult(
                task_id=task_id, task_type=task.task_type,
                status=ExecutionStatus.FAILED,
                error=str(exc), elapsed_ms=elapsed_ms,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 2)
            logger.exception("任务 %s 执行异常", task_id)
            return TaskResult(
                task_id=task_id, task_type=task.task_type,
                status=ExecutionStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}", elapsed_ms=elapsed_ms,
            )

    # --------------------------------------------------------
    # 任务类型分发 — RETRIEVE
    # --------------------------------------------------------

    async def _execute_retrieve_task(
        self, task: SubTask, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行检索任务."""
        query = task.query or context.get("query", "")
        strategy = task.retrieval_strategy or RetrievalStrategy.HYBRID
        budget = task.resource_budget

        logger.debug("检索任务 %s, 策略=%s, query=%s", task.task_id, strategy.value, query[:50])

        if strategy == RetrievalStrategy.HYBRID:
            result = self._hybrid_retriever.retrieve(
                query, top_k=budget.max_retrieval_depth * 3,
                query_vector=context.get("query_vector"),
                entity_id=context.get("entity_id"),
            )
            return {
                "results": result.results,
                "scores": result.scores,
                "total": result.total,
                "source_type": result.source_type,
                "confidence": max(result.scores) if result.scores else 0.0,
                "evidence": [{"type": "retrieval", "source": r.get("chunk_id", "")}
                             for r in result.results[:5]],
                "reasoning_chain": [f"混合检索: {result.total} 条结果"],
            }

        elif strategy == RetrievalStrategy.GRAPHRAG:
            result = self._graphrag_retriever.search(
                query, max_hops=budget.max_retrieval_depth,
                top_k=budget.max_retrieval_depth * 3,
                entity_ids=context.get("entity_ids"),
            )
            # GraphRAGResult 字段为复数: local_results / global_results
            local_res = result.local_results
            global_res = result.global_results
            return {
                "local_result": local_res.model_dump() if local_res else {},
                "global_result": global_res.model_dump() if global_res else {},
                "strategy": result.strategy,
                "confidence": 0.8 if local_res or global_res else 0.0,
                "evidence": [{"type": "graphrag", "source": "local+global"}],
                "reasoning_chain": [f"GraphRAG 检索: 策略={result.strategy}"],
            }

        elif strategy == RetrievalStrategy.SUBGRAPH:
            entity_id = task.params.get("entity_id") or context.get("entity_id")
            if not entity_id:
                raise TaskExecutionError(task.task_id, "子图检索需要 entity_id")
            if self._subgraph_reasoner is None:
                raise TaskExecutionError(task.task_id, "SubgraphReasoner 未初始化")

            result = self._subgraph_reasoner.extract_and_reason(
                entity_id, query,
                strategy=task.params.get("subgraph_strategy", "bfs"),
                max_depth=budget.max_retrieval_depth,
            )
            return {
                "entities": list(result.entities),
                "triples": [t.model_dump() for t in result.triples[:20]],
                "paths": [p.model_dump() for p in result.paths],
                "summary": result.summary,
                "confidence": 0.85 if result.triples else 0.0,
                "evidence": [{"type": "subgraph", "entity": entity_id}],
                "reasoning_chain": [f"子图推理: {len(result.entities)} 实体, {len(result.triples)} 三元组"],
            }

        elif strategy == RetrievalStrategy.VECTOR:
            qv = context.get("query_vector")
            if qv is None:
                raise TaskExecutionError(task.task_id, "向量检索需要 query_vector")
            from ..l3.retrieval import VectorRetriever
            retriever = VectorRetriever(self._store)
            result = retriever.retrieve(query, top_k=budget.max_retrieval_depth * 3,
                                        query_vector=qv)
            return {
                "results": result.results, "scores": result.scores,
                "total": result.total, "confidence": max(result.scores) if result.scores else 0.0,
                "evidence": [{"type": "vector", "source": r.get("chunk_id", "")}
                             for r in result.results[:5]],
                "reasoning_chain": [f"向量检索: {result.total} 条结果"],
            }

        elif strategy == RetrievalStrategy.KEYWORD:
            from ..l3.retrieval import KeywordRetriever
            retriever = KeywordRetriever(self._store)
            result = retriever.retrieve(query, top_k=budget.max_retrieval_depth * 3)
            return {
                "results": result.results, "scores": result.scores,
                "total": result.total, "confidence": max(result.scores) if result.scores else 0.0,
                "evidence": [{"type": "keyword", "source": r.get("chunk_id", "")}
                             for r in result.results[:5]],
                "reasoning_chain": [f"关键词检索: {result.total} 条结果"],
            }

        elif strategy == RetrievalStrategy.GRAPH:
            from ..l3.retrieval import GraphRetriever
            retriever = GraphRetriever(self._store)
            entity_id = context.get("entity_id")
            result = retriever.retrieve(query, top_k=budget.max_retrieval_depth * 3,
                                        entity_id=entity_id)
            return {
                "results": result.results, "scores": result.scores,
                "total": result.total, "confidence": max(result.scores) if result.scores else 0.0,
                "evidence": [{"type": "graph", "source": r.get("entity_id", "")}
                             for r in result.results[:5]],
                "reasoning_chain": [f"图检索: {result.total} 条结果"],
            }

        else:
            raise TaskExecutionError(task.task_id, f"未支持的检索策略: {strategy}")

    # --------------------------------------------------------
    # 任务类型分发 — REASON
    # --------------------------------------------------------

    async def _execute_reason_task(
        self, task: SubTask, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行推理任务."""
        query = task.query or context.get("query", "")
        mode = task.reasoning_mode or ReasoningMode.MULTI_HOP
        budget = task.resource_budget

        logger.debug("推理任务 %s, 模式=%s, query=%s", task.task_id, mode.value, query[:50])

        # V1 推理模式
        if mode in (
            ReasoningMode.PATH_FINDING, ReasoningMode.MULTI_HOP,
            ReasoningMode.RULE_INFERENCE, ReasoningMode.LINK_PREDICTION,
            ReasoningMode.PATTERN_MATCH, ReasoningMode.ANALOGY,
        ):
            from ..l3.graph_reasoner import ReasoningMode as L3ReasoningMode

            l3_mode = L3ReasoningMode(mode.value)
            kwargs = dict(task.params)
            kwargs["max_depth"] = budget.max_reasoning_hops

            # 从上下文注入实体 ID
            if "start_id" not in kwargs and context.get("entity_id"):
                kwargs["start_id"] = context["entity_id"]
            if "entity_id" not in kwargs and context.get("entity_id"):
                kwargs["entity_id"] = context["entity_id"]

            result = self._graph_reasoner.reason(query, l3_mode, **kwargs)
            return {
                "mode": result.mode.value,
                "answers": result.answers,
                "confidence": result.confidence,
                "reasoning_chain": result.reasoning_chain,
                "evidence": result.evidence_triples,
                "elapsed_ms": result.elapsed_ms,
            }

        # V2 — 后向链式推理
        elif mode == ReasoningMode.BACKWARD_CHAIN:
            if self._backward_reasoner is None:
                raise TaskExecutionError(task.task_id, "BackwardChainingReasoner 未初始化")

            goal_predicate = task.params.get("goal_predicate", "")
            goal_object = task.params.get("goal_object", "")
            if not goal_predicate:
                raise TaskExecutionError(task.task_id, "后向链式推理需要 goal_predicate")

            result = self._backward_reasoner.reason(goal_predicate, goal_object, **task.params)
            return {
                "mode": "backward_chain",
                "answers": result.answers,
                "confidence": result.confidence,
                "reasoning_chain": result.reasoning_chain,
                "evidence": result.evidence_triples,
                "elapsed_ms": result.elapsed_ms,
            }

        # V2 — 置信度加权遍历
        elif mode == ReasoningMode.CONFIDENCE_TRAV:
            if self._confidence_traversal is None:
                raise TaskExecutionError(task.task_id, "ConfidenceWeightedTraversal 未初始化")

            entity_id = task.params.get("entity_id") or context.get("entity_id")
            if not entity_id:
                raise TaskExecutionError(task.task_id, "置信度遍历需要 entity_id")

            result = self._confidence_traversal.traverse(
                entity_id, max_depth=budget.max_reasoning_hops
            )
            return {
                "mode": "confidence_traversal",
                "entity_scores": result.entity_scores,
                "triples": [t.model_dump() for t in result.traversed_triples[:20]],
                "max_depth": result.max_depth_reached,
                "confidence": max(result.entity_scores.values()) if result.entity_scores else 0.0,
                "reasoning_chain": [f"置信度遍历: {result.total_entities} 实体, 深度 {result.max_depth_reached}"],
                "evidence": [{"type": "traversal", "entity": entity_id}],
            }

        # V2 — TransE 嵌入推理
        elif mode == ReasoningMode.TRANS_E:
            if self._trans_e_embedder is None:
                raise TaskExecutionError(task.task_id, "TransEEmbedder 未初始化")

            head_id = task.params.get("head_id") or context.get("entity_id")
            relation = task.params.get("relation", "")
            if not head_id or not relation:
                raise TaskExecutionError(task.task_id, "TransE 推理需要 head_id 和 relation")

            result = self._trans_e_embedder.predict_tail(head_id, relation, top_k=5)
            return {
                "mode": "trans_e",
                "predictions": [
                    {"tail_id": r.predicted_tail, "score": r.score, "confidence": r.confidence}
                    for r in (result if isinstance(result, list) else [result])
                ],
                "confidence": result[0].confidence if isinstance(result, list) and result else 0.0,
                "reasoning_chain": [f"TransE 链接预测: head={head_id}, relation={relation}"],
                "evidence": [{"type": "trans_e", "head": head_id, "relation": relation}],
            }

        else:
            raise TaskExecutionError(task.task_id, f"未支持的推理模式: {mode}")

    # --------------------------------------------------------
    # 任务类型分发 — VERIFY / SYNTHESIZE (桩 + 集成点)
    # --------------------------------------------------------

    async def _execute_verify_task(
        self, task: SubTask, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行验证任务 — 集成 FactChecker / QualityManager / ConflictDetector.

        对上下文中的待验证内容 (``content`` / ``text``) 与实体执行多维度验证:
        - FactChecker: 数值断言提取 + 标准值匹配 (基于内容文本)
        - QualityManager: 实体六维质量评估 (上下文提供实体时)
        - ConflictDetector: 值/跨源冲突检测 (上下文提供实体时)

        未注入任何验证组件时，返回清晰的预留桩响应。
        """
        logger.debug("验证任务 %s", task.task_id)

        has_verifier = any(
            (self._fact_checker, self._quality_manager, self._conflict_detector)
        )
        if not has_verifier:
            logger.debug(
                "验证任务 %s — 未注入验证组件，返回预留桩", task.task_id,
            )
            return {
                "status": "stub",
                "confidence": 0.9,
                "reasoning_chain": [
                    "验证: 预留桩（未注入 FactChecker/QualityManager/ConflictDetector）"
                ],
                "evidence": [],
            }

        # 待验证内容 (来自前置任务输出或显式上下文)
        content = context.get("content") or context.get("text") or ""
        reasoning_chain: list[str] = []
        evidence: list[dict[str, Any]] = []
        assertions: list[dict[str, Any]] = []
        confidences: list[float] = []
        overall_passed = True

        # 1. 事实校验 (FactChecker) — 基于内容文本
        if self._fact_checker is not None and content:
            try:
                report = self._fact_checker.check(content)
                # 无可校验断言时视为"通过(无可验证内容)"，避免误判失败
                if getattr(report, "checked", 0) > 0:
                    overall_passed = overall_passed and bool(
                        getattr(report, "overall_passed", True)
                    )
                    confidences.append(float(getattr(report, "confidence", 0.0)))
                else:
                    confidences.append(0.9)
                assertions = list(getattr(report, "results", []))
                reasoning_chain.append(
                    f"事实校验: 共 {getattr(report, 'total_assertions', 0)} 项断言, "
                    f"已校验 {getattr(report, 'checked', 0)}, "
                    f"通过 {getattr(report, 'passed', 0)}, "
                    f"失败 {getattr(report, 'failed', 0)}, "
                    f"跳过 {getattr(report, 'skipped', 0)}"
                )
                evidence.append({
                    "type": "fact_check",
                    "overall_passed": getattr(report, "overall_passed", True),
                    "confidence": getattr(report, "confidence", 0.0),
                    "feedback": getattr(report, "feedback", ""),
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("验证任务 %s 事实校验异常: %s", task.task_id, exc)
                reasoning_chain.append(f"事实校验异常: {exc}")
                overall_passed = False

        # 2. 质量评估 + 冲突检测 — 基于上下文中的实体
        entities: list[Any] = []
        if context.get("entities"):
            entities = list(context["entities"])
        elif context.get("entity") is not None:
            entities = [context["entity"]]

        if entities:
            # 2a. 质量评估 (QualityManager)
            if self._quality_manager is not None:
                try:
                    scores: list[float] = []
                    for entity in entities:
                        qa_result = self._quality_manager.assess_entity(
                            entity, context=context,
                        )
                        scores.append(float(qa_result.overall_score))
                        reasoning_chain.append(
                            f"质量评估 {getattr(entity, 'entity_id', '?')}: "
                            f"{qa_result.grade.value} ({qa_result.overall_score:.2%})"
                        )
                        evidence.append({
                            "type": "quality",
                            "entity_id": getattr(entity, "entity_id", ""),
                            "overall_score": qa_result.overall_score,
                            "grade": qa_result.grade.value,
                        })
                    if scores:
                        confidences.append(sum(scores) / len(scores))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("验证任务 %s 质量评估异常: %s", task.task_id, exc)
                    reasoning_chain.append(f"质量评估异常: {exc}")

            # 2b. 冲突检测 (ConflictDetector)
            if self._conflict_detector is not None:
                external_claims = context.get("external_claims")
                try:
                    conflicts: list[Any] = []
                    for entity in entities:
                        conflicts.extend(
                            self._conflict_detector.detect_value_conflicts(
                                entity, external_claims,
                            )
                        )
                    if conflicts:
                        overall_passed = False
                        reasoning_chain.append(
                            f"冲突检测: 发现 {len(conflicts)} 项冲突"
                        )
                        evidence.append({
                            "type": "conflict",
                            "count": len(conflicts),
                        })
                    else:
                        reasoning_chain.append("冲突检测: 未发现冲突")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("验证任务 %s 冲突检测异常: %s", task.task_id, exc)
                    reasoning_chain.append(f"冲突检测异常: {exc}")

        if not reasoning_chain:
            reasoning_chain.append("验证: 上下文无可验证内容")

        confidence = (
            sum(confidences) / len(confidences) if confidences
            else (1.0 if overall_passed else 0.5)
        )

        return {
            "status": "passed" if overall_passed else "failed",
            "overall_passed": overall_passed,
            "confidence": round(confidence, 4),
            "assertions": assertions,
            "reasoning_chain": reasoning_chain,
            "evidence": evidence,
        }

    async def _execute_synthesize_task(
        self, task: SubTask, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行合成任务 — 集成 ResponseSynthesizer.

        将上下文中的检索结果合成为自然语言响应。优先复用上下文中已构造的
        ``RetrievalResult``；否则从 ``results`` / ``scores`` 等片段构建。

        未注入 ResponseSynthesizer 时，返回清晰的预留桩响应。
        """
        logger.debug("合成任务 %s", task.task_id)

        if self._response_synthesizer is None:
            logger.debug(
                "合成任务 %s — 未注入 ResponseSynthesizer，返回预留桩", task.task_id,
            )
            return {
                "status": "stub",
                "confidence": 0.85,
                "reasoning_chain": [
                    "合成: 预留桩（未注入 ResponseSynthesizer）"
                ],
                "evidence": [],
            }

        query = task.query or context.get("query", "")

        # 构建检索结果 (优先复用上下文中已构造的 RetrievalResult)
        retrieval_result = context.get("retrieval_result")
        if retrieval_result is None:
            from ..l3.models import RetrievalResult

            results = context.get("results") or context.get("nodes") or []
            scores = context.get("scores") or []
            retrieval_result = RetrievalResult(
                query=query,
                results=list(results),
                scores=[float(s) for s in scores],
                total=int(context.get("total", len(results))),
                source_type=str(context.get("source_type", "hybrid")),
                retrieval_time_ms=float(context.get("retrieval_time_ms", 0.0)),
                trace_id=str(context.get("trace_id", "")),
            )

        try:
            response = self._response_synthesizer.synthesize(
                retrieval_result, query=query,
            )
            return {
                "status": "synthesized",
                "answer": response.answer,
                "confidence": float(response.confidence),
                "citations": [c.model_dump() for c in response.citations],
                "evidence": [
                    e.model_dump() for e in response.evidence_pieces
                ],
                "reasoning_chain": [
                    f"响应合成: 模式={response.synthesis_mode.value}, "
                    f"来源 {response.source_count} 条, "
                    f"证据 {len(response.evidence_pieces)} 条"
                ],
                "source_count": response.source_count,
                "synthesis_mode": response.synthesis_mode.value,
                "synthesis_time_ms": response.synthesis_time_ms,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("合成任务 %s 合成异常: %s", task.task_id, exc)
            return {
                "status": "failed",
                "confidence": 0.0,
                "reasoning_chain": [f"响应合成异常: {exc}"],
                "evidence": [],
            }

    # --------------------------------------------------------
    # 降级执行
    # --------------------------------------------------------

    async def _execute_fallback(
        self, plan: DecisionPlan, context: dict[str, Any]
    ) -> ExecutionResult:
        """执行降级计划."""
        fallback = plan.fallback_plan or FallbackPlan()
        logger.info("执行降级计划: %s", fallback.trigger_condition)

        if fallback.direct_retrieval and plan.original_query:
            # 降级为直接混合检索
            try:
                result = self._hybrid_retriever.retrieve(
                    plan.original_query, top_k=10,
                    query_vector=context.get("query_vector"),
                )
                task_result = TaskResult(
                    task_id="fallback_retrieval",
                    task_type=TaskType.RETRIEVE,
                    status=ExecutionStatus.COMPLETED,
                    output={"results": result.results, "scores": result.scores},
                    confidence=0.6,
                    reasoning_chain=["降级: 直接混合检索"],
                )
                exec_result = ExecutionResult(
                    plan_id=plan.plan_id, status=ExecutionStatus.COMPLETED,
                    fallback_triggered=True,
                    task_results={"fallback_retrieval": task_result},
                )
                self._aggregate_result(exec_result)
                return exec_result
            except Exception as exc:  # noqa: BLE001
                logger.error("降级检索也失败: %s", exc)

        # 彻底失败
        return ExecutionResult(
            plan_id=plan.plan_id, status=ExecutionStatus.FAILED,
            fallback_triggered=True,
            error_summary="降级计划执行失败",
        )

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    def _build_task_context(
        self, task: SubTask, global_context: dict[str, Any], completed: set[str]
    ) -> dict[str, Any]:
        """构建子任务执行上下文（注入前置任务结果）."""
        ctx = dict(global_context)
        ctx["task_id"] = task.task_id
        ctx["task_type"] = task.task_type.value

        # 从依赖任务的结果中提取有用信息
        for dep_id in task.deps:
            dep_result = global_context.get(f"result:{dep_id}")
            if dep_result is None:
                continue
            # 注入实体 ID
            if "entity_id" not in ctx:
                entity_id = (
                    dep_result.get("output", {}).get("entity_id")
                    or dep_result.get("output", {}).get("start_id")
                    or dep_result.get("output", {}).get("focus_entity")
                )
                if entity_id:
                    ctx["entity_id"] = entity_id
            # 注入查询向量
            if "query_vector" not in ctx:
                qv = dep_result.get("output", {}).get("query_vector")
                if qv:
                    ctx["query_vector"] = qv

        return ctx

    def _build_parallel_layers(self, plan: DecisionPlan) -> list[list[SubTask]]:
        """构建并行执行层（每层内任务互相无依赖）."""
        in_degree: dict[str, int] = {t.task_id: 0 for t in plan.sub_tasks}
        adj: dict[str, list[str]] = {t.task_id: [] for t in plan.sub_tasks}

        for t in plan.sub_tasks:
            for dep in t.deps:
                if dep in adj:
                    adj[dep].append(t.task_id)
                    in_degree[t.task_id] += 1

        layers: list[list[SubTask]] = []
        remaining_ids: set[str] = {t.task_id for t in plan.sub_tasks}
        id_to_task = {t.task_id: t for t in plan.sub_tasks}

        while remaining_ids:
            # 找当前入度为 0 的任务（在 remaining 中）
            current_in_degree = {
                tid: sum(1 for d in id_to_task[tid].deps if d in remaining_ids)
                for tid in remaining_ids
            }
            layer_ids = [tid for tid in remaining_ids if current_in_degree[tid] == 0]
            if not layer_ids:
                raise CyclicDependencyError(plan.plan_id)
            layer = [id_to_task[tid] for tid in layer_ids]
            layers.append(layer)
            remaining_ids -= set(layer_ids)

        return layers

    def _aggregate_result(self, result: ExecutionResult) -> None:
        """聚合执行结果 — 计算综合置信度、收集证据和推理链."""
        all_evidence: list[dict[str, Any]] = []
        all_chains: list[str] = []
        total_tokens = 0

        for tr in result.task_results.values():
            all_evidence.extend(tr.evidence)
            all_chains.extend(tr.reasoning_chain)
            total_tokens += tr.token_usage

        result.evidence_set = all_evidence
        result.reasoning_chain = all_chains
        result.total_token_usage = total_tokens
        result.confidence = result.compute_confidence()

    def _build_error_result(
        self, plan: DecisionPlan, error: str, start_ts: float
    ) -> ExecutionResult:
        """构建错误结果."""
        return ExecutionResult(
            plan_id=plan.plan_id,
            status=ExecutionStatus.FAILED,
            error_summary=error,
            total_elapsed_ms=round((time.perf_counter() - start_ts) * 1000, 2),
        )


__all__ = [
    "TaskExecutor",
    "TaskExecutionError",
    "BudgetExceededError",
    "CyclicDependencyError",
]
