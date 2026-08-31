"""L4 决策引擎层 — 决策计划生成器 (DecisionPlanner).

融合世界先进方案的计划生成设计:
- TDP 框架 (2026): Supervisor-Planner-Executor 三层架构
  - Supervisor: 监督计划合法性、资源约束、安全策略
  - Planner: 将意图分解为子任务 DAG
  - Executor: 调度执行 (在 TaskExecutor 中实现)
- Plan-and-Solve: 计划分解 → 逐步求解
- ReACT: 推理与行动交替，每步评估是否需要工具
- LangGraph: 条件边 + 检查点，支持动态重规划

核心职责:
    将 T1 产出的 RoutedResult (意图 + 实体 + 查询) 转化为
    DecisionPlan (子任务 DAG + 执行策略 + 资源预算)。

设计原则:
    1. 零外部 LLM 依赖 — 规则 + 模板驱动计划生成
    2. 意图驱动 — 不同意图类型生成不同的子任务拓扑
    3. 资源感知 — 根据任务复杂度自动估算资源预算
    4. 可降级 — 所有计划内置 FallbackPlan
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .models import (
    DecisionPlan,
    ExecutionMode,
    FallbackPlan,
    ReasoningMode,
    ResourceBudget,
    RetrievalStrategy,
    SubTask,
    TaskType,
)

logger = logging.getLogger(__name__)


# ============================================================
# 计划模板库
# ============================================================


class PlanTemplate:
    """计划模板 — 预定义的子任务 DAG 模式.

    借鉴 TDP 框架的任务模板和 LangGraph 的预置图结构:
    每种意图类型对应一个最优的子任务拓扑。
    """

    @staticmethod
    def concept_query(query: str, entity_id: str | None = None) -> list[SubTask]:
        """概念查询模板: 检索 → 合成.

        流程: RETRIEVE(向量/关键词) → SYNTHESIZE(响应合成)
        """
        tasks: list[SubTask] = [
            SubTask(
                task_id="retrieve_concept",
                task_type=TaskType.RETRIEVE,
                retrieval_strategy=RetrievalStrategy.HYBRID,
                query=query,
                expected_output="概念定义与解释的相关文本片段",
                resource_budget=ResourceBudget(max_tokens=2048, max_retrieval_depth=3),
            ),
            SubTask(
                task_id="synthesize_concept",
                task_type=TaskType.SYNTHESIZE,
                deps=["retrieve_concept"],
                query=query,
                expected_output="结构化的概念解释响应",
                resource_budget=ResourceBudget(max_tokens=4096),
            ),
        ]
        if entity_id:
            tasks[0].params["entity_id"] = entity_id
        return tasks

    @staticmethod
    def numeric_query(query: str, entity_id: str | None = None) -> list[SubTask]:
        """数值查询模板: 检索 → 推理 → 验证 → 合成.

        流程: RETRIEVE(精确检索) → REASON(数值定位) → VERIFY(事实校验) → SYNTHESIZE
        """
        tasks: list[SubTask] = [
            SubTask(
                task_id="retrieve_numeric",
                task_type=TaskType.RETRIEVE,
                retrieval_strategy=RetrievalStrategy.HYBRID,
                query=query,
                expected_output="含数值参数的文本片段",
                resource_budget=ResourceBudget(max_tokens=2048, max_retrieval_depth=3),
            ),
            SubTask(
                task_id="reason_numeric",
                task_type=TaskType.REASON,
                deps=["retrieve_numeric"],
                reasoning_mode=ReasoningMode.MULTI_HOP,
                query=query,
                expected_output="数值推理结果与定位",
                resource_budget=ResourceBudget(max_tokens=2048, max_reasoning_hops=5),
            ),
            SubTask(
                task_id="verify_numeric",
                task_type=TaskType.VERIFY,
                deps=["reason_numeric"],
                query=query,
                expected_output="数值事实校验报告",
                resource_budget=ResourceBudget(max_tokens=1024),
            ),
            SubTask(
                task_id="synthesize_numeric",
                task_type=TaskType.SYNTHESIZE,
                deps=["verify_numeric"],
                query=query,
                expected_output="带置信度的数值回答",
                resource_budget=ResourceBudget(max_tokens=2048),
            ),
        ]
        if entity_id:
            tasks[0].params["entity_id"] = entity_id
            tasks[1].params["entity_id"] = entity_id
        return tasks

    @staticmethod
    def relational_query(query: str, entity_id: str | None = None) -> list[SubTask]:
        """关系查询模板: 子图提取 → 路径推理 → 合成.

        流程: RETRIEVE(子图) → REASON(路径查找) → SYNTHESIZE
        """
        tasks: list[SubTask] = [
            SubTask(
                task_id="retrieve_subgraph",
                task_type=TaskType.RETRIEVE,
                retrieval_strategy=RetrievalStrategy.SUBGRAPH,
                query=query,
                expected_output="查询实体相关的子图",
                resource_budget=ResourceBudget(max_tokens=2048, max_retrieval_depth=4),
            ),
            SubTask(
                task_id="reason_path",
                task_type=TaskType.REASON,
                deps=["retrieve_subgraph"],
                reasoning_mode=ReasoningMode.PATH_FINDING,
                query=query,
                expected_output="实体间路径与关系链",
                resource_budget=ResourceBudget(max_tokens=2048, max_reasoning_hops=8),
            ),
            SubTask(
                task_id="synthesize_relational",
                task_type=TaskType.SYNTHESIZE,
                deps=["reason_path"],
                query=query,
                expected_output="关系解释与影响分析",
                resource_budget=ResourceBudget(max_tokens=4096),
            ),
        ]
        if entity_id:
            tasks[0].params["entity_id"] = entity_id
            tasks[1].params["entity_id"] = entity_id
        return tasks

    @staticmethod
    def composite_query(query: str, entity_id: str | None = None) -> list[SubTask]:
        """复合查询模板: 多路并行检索 → 多模式推理 → 验证 → 合成.

        流程: RETRIEVE(混合) + RETRIEVE(GraphRAG) → REASON(多跳) + REASON(规则)
              → VERIFY → SYNTHESIZE
        """
        tasks: list[SubTask] = [
            SubTask(
                task_id="retrieve_hybrid",
                task_type=TaskType.RETRIEVE,
                retrieval_strategy=RetrievalStrategy.HYBRID,
                query=query,
                expected_output="混合检索结果",
                resource_budget=ResourceBudget(max_tokens=2048, max_retrieval_depth=3),
            ),
            SubTask(
                task_id="retrieve_graphrag",
                task_type=TaskType.RETRIEVE,
                retrieval_strategy=RetrievalStrategy.GRAPHRAG,
                query=query,
                expected_output="GraphRAG 全局+局部检索结果",
                resource_budget=ResourceBudget(max_tokens=2048, max_retrieval_depth=3),
            ),
            SubTask(
                task_id="reason_multi_hop",
                task_type=TaskType.REASON,
                deps=["retrieve_hybrid", "retrieve_graphrag"],
                reasoning_mode=ReasoningMode.MULTI_HOP,
                query=query,
                expected_output="多跳推理结果",
                resource_budget=ResourceBudget(max_tokens=2048, max_reasoning_hops=6),
            ),
            SubTask(
                task_id="reason_rule",
                task_type=TaskType.REASON,
                deps=["retrieve_hybrid", "retrieve_graphrag"],
                reasoning_mode=ReasoningMode.RULE_INFERENCE,
                query=query,
                expected_output="规则推理结果",
                resource_budget=ResourceBudget(max_tokens=2048, max_reasoning_hops=5),
            ),
            SubTask(
                task_id="verify_composite",
                task_type=TaskType.VERIFY,
                deps=["reason_multi_hop", "reason_rule"],
                query=query,
                expected_output="交叉验证报告",
                resource_budget=ResourceBudget(max_tokens=1024),
            ),
            SubTask(
                task_id="synthesize_composite",
                task_type=TaskType.SYNTHESIZE,
                deps=["verify_composite"],
                query=query,
                expected_output="综合回答",
                resource_budget=ResourceBudget(max_tokens=4096),
            ),
        ]
        if entity_id:
            tasks[0].params["entity_id"] = entity_id
            tasks[1].params["entity_id"] = entity_id
            tasks[2].params["entity_id"] = entity_id
            tasks[3].params["entity_id"] = entity_id
        return tasks


# ============================================================
# 决策计划生成器
# ============================================================


class DecisionPlanner:
    """决策计划生成器 — T2 核心模块.

    将 T1(RoutedResult) 转化为 T2(DecisionPlan)，包含:
    1. 意图分析 — 根据意图类型选择计划模板
    2. 子任务生成 — 从模板实例化具体 SubTask
    3. 策略选择 — 决定执行模式（串行/并行/迭代）
    4. 资源估算 — 为每个子任务分配资源预算
    5. 降级计划 — 生成 FallbackPlan

    Usage::

        planner = DecisionPlanner()
        plan = planner.plan(routed_result)
        # plan 被传入 TaskExecutor.execute()
    """

    def __init__(self) -> None:
        """初始化决策计划生成器."""
        self._templates = PlanTemplate()
        logger.info("DecisionPlanner 初始化完成")

    def plan(
        self,
        routed_result: Any,
        *,
        context_id: str = "",
        learner_profile: dict[str, Any] | None = None,
    ) -> DecisionPlan:
        """生成决策计划.

        Args:
            routed_result: T1 产出的路由结果 (RoutedResult)
            context_id: 上下文 ID
            learner_profile: 学习者画像 (用于调整策略)

        Returns:
            DecisionPlan 决策计划
        """
        start_ts = time.perf_counter()

        # 提取关键信息
        intent_type = self._extract_intent_type(routed_result)
        query = self._extract_query(routed_result)
        entity_id = self._extract_entity_id(routed_result)

        logger.info("开始生成决策计划: intent=%s, query=%s", intent_type, query[:50])

        # 选择模板并生成子任务
        sub_tasks = self._build_sub_tasks(intent_type, query, entity_id)

        # 选择执行模式
        execution_mode = self._select_execution_mode(sub_tasks, intent_type)

        # 资源估算
        estimated_tokens, estimated_latency = self._estimate_resources(sub_tasks)

        # 生成降级计划
        fallback_plan = self._build_fallback_plan(intent_type, query)

        # 组装决策计划
        plan = DecisionPlan(
            sub_tasks=sub_tasks,
            execution_mode=execution_mode,
            fallback_plan=fallback_plan,
            estimated_total_tokens=estimated_tokens,
            estimated_total_latency_ms=estimated_latency,
            context_id=context_id,
            original_query=query,
        )

        plan_time_ms = round((time.perf_counter() - start_ts) * 1000, 2)
        logger.info(
            "决策计划生成完成: plan_id=%s, 子任务=%d, 模式=%s, 预估延迟=%dms, 耗时=%.2fms",
            plan.plan_id, len(sub_tasks), execution_mode.value,
            estimated_latency, plan_time_ms,
        )

        return plan

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    def _extract_intent_type(self, routed_result: Any) -> str:
        """从 RoutedResult 中提取意图类型."""
        if hasattr(routed_result, "intent") and hasattr(routed_result.intent, "intent_type"):
            return str(routed_result.intent.intent_type.value)
        if isinstance(routed_result, dict):
            intent = routed_result.get("intent", {})
            if isinstance(intent, dict):
                return intent.get("intent_type", "composite")
            return str(intent)
        return "composite"

    def _extract_query(self, routed_result: Any) -> str:
        """从 RoutedResult 中提取查询文本."""
        if isinstance(routed_result, dict):
            return routed_result.get("query", "")
        if hasattr(routed_result, "retrieval_result"):
            qr = routed_result.retrieval_result
            if hasattr(qr, "query"):
                return qr.query
        if hasattr(routed_result, "query"):
            return routed_result.query
        return ""

    def _extract_entity_id(self, routed_result: Any) -> str | None:
        """从 RoutedResult 中提取实体 ID."""
        entities: list[Any] = []
        if hasattr(routed_result, "intent") and hasattr(routed_result.intent, "extracted_entities"):
            entities = routed_result.intent.extracted_entities
        elif isinstance(routed_result, dict):
            intent = routed_result.get("intent", {})
            if isinstance(intent, dict):
                entities = intent.get("extracted_entities", [])

        for ent in entities:
            if getattr(ent, "entity_type", "") in ("ion", "formula"):
                return getattr(ent, "text", None) or getattr(ent, "value", None)
            if isinstance(ent, dict) and ent.get("entity_type") in ("ion", "formula"):
                return ent.get("text") or ent.get("value")
        return None

    def _build_sub_tasks(
        self, intent_type: str, query: str, entity_id: str | None
    ) -> list[SubTask]:
        """根据意图类型构建子任务列表."""
        intent_map: dict[str, callable] = {
            "concept": self._templates.concept_query,
            "numeric": self._templates.numeric_query,
            "relational": self._templates.relational_query,
            "composite": self._templates.composite_query,
        }

        template_fn = intent_map.get(intent_type, self._templates.composite_query)
        return template_fn(query, entity_id)

    def _select_execution_mode(self, sub_tasks: list[SubTask], intent_type: str) -> ExecutionMode:
        """选择执行模式.

        策略:
        - 概念查询: 串行（简单两阶段）
        - 数值查询: 串行（有严格依赖链）
        - 关系查询: 串行（子图依赖路径）
        - 复合查询: 并行（多路检索可并发）
        """
        if intent_type == "composite":
            return ExecutionMode.PARALLEL
        if intent_type == "numeric" and len(sub_tasks) > 2:
            return ExecutionMode.SEQUENTIAL
        return ExecutionMode.SEQUENTIAL

    def _estimate_resources(self, sub_tasks: list[SubTask]) -> tuple[int, int]:
        """估算资源需求.

        Returns:
            (预估总 Token, 预估总延迟毫秒)
        """
        total_tokens = 0
        total_latency = 0

        for task in sub_tasks:
            budget = task.resource_budget
            total_tokens += budget.max_tokens
            # 简单估算: 检索 ~100ms/depth, 推理 ~200ms/hop, 验证 ~50ms, 合成 ~300ms
            latency = 0
            if task.task_type == TaskType.RETRIEVE:
                latency = 100 * budget.max_retrieval_depth + 50
            elif task.task_type == TaskType.REASON:
                latency = 200 * budget.max_reasoning_hops + 100
            elif task.task_type == TaskType.VERIFY:
                latency = 80
            elif task.task_type == TaskType.SYNTHESIZE:
                latency = 300
            total_latency += latency

        # 并行优化: 如果有并行层，取最大层延迟而非总和
        return total_tokens, int(total_latency * 1.2)  # 20% 缓冲

    def _build_fallback_plan(self, intent_type: str, query: str) -> FallbackPlan:
        """构建降级计划."""
        return FallbackPlan(
            trigger_condition="any_failure",
            fallback_mode=ExecutionMode.SEQUENTIAL,
            direct_retrieval=True,
            max_retries=1,
        )


__all__ = [
    "DecisionPlanner",
    "PlanTemplate",
]
