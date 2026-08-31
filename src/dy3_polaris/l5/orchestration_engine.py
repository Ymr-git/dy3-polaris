"""编排引擎模块 — L5 Agent Runtime 核心组件.

融合世界先进方案:
- LangGraph: StateGraph + superstep + 条件边 + checkpoint
- Temporal: Workflow/Activity 分离 + 事件溯源 + retry policy + continue-as-new
- Google ADK: SequentialAgent/ParallelAgent/LoopAgent 声明式编排
- OpenAI Agents SDK: Handoff 机制 + Guardrail 并行
- AutoGen: GroupChat 多策略 Manager + stall 检测
- CrewAI: 角色驱动 + manager 验证

本模块实现:
1. OrchestrationTask — 任务节点 (含依赖/超时/状态)
2. OrchestrationPlan — DAG 执行计划 (拓扑排序 + 并行层 + 循环检测)
3. OrchestrationResult — 执行结果 (含溯源)
4. PipelineExecutor — 顺序流水线编排 (ADK SequentialAgent + LangGraph superstep)
5. DebateExecutor — 辩论交叉验证编排 (辩论弧 + 收敛 + 代币预算)
6. VotingExecutor — 投票共识编排 (并行 fan-out + 声誉加权聚合)
7. OrchestrationEngine — 顶层编排引擎 (重试 + 超时 + 溯源)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class OrchestrationState(str, Enum):
    """编排状态 (融合 Temporal Activity 状态机 + LangGraph 节点状态).

    PENDING → RUNNING → COMPLETED
                     ↘ FAILED
                     ↘ TIMEOUT
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class OrchestrationParadigm(str, Enum):
    """编排范式 (L5 设计文档第八章 8.2.3).

    - PIPELINE: 顺序流水线 (复杂度 0-30)
    - DEBATE: 辩论交叉验证 (复杂度 31-65)
    - VOTING: 投票共识 (复杂度 66-100)
    """

    PIPELINE = "pipeline"
    DEBATE = "debate"
    VOTING = "voting"


class DebateState(str, Enum):
    """辩论状态 (Claude Science 辩论弧模式)."""

    INITIALIZED = "initialized"
    ROUND_IN_PROGRESS = "round_in_progress"
    CONVERGED = "converged"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    TIMED_OUT = "timed_out"


# ============================================================
# 异常定义
# ============================================================


class OrchestrationError(Exception):
    """编排错误 (支持可重试/不可重试标记, Temporal 模式)."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        self.message = message
        self.retryable = retryable
        super().__init__(message)


class OrchestrationTimeoutError(OrchestrationError):
    """编排超时错误 (Temporal 四级超时模型)."""

    def __init__(self, message: str, timeout_type: str = "task") -> None:
        self.timeout_type = timeout_type
        super().__init__(message, retryable=False)


# ============================================================
# OrchestrationTask — 任务节点
# ============================================================


class OrchestrationTask:
    """编排任务节点 (融合 LangGraph 节点 + Temporal Activity).

    每个任务包含:
    - task_id: 唯一标识
    - agent_id: 执行 Agent
    - name: 任务名称
    - dependencies: 依赖任务 ID 列表 (DAG 边)
    - timeout_s: 任务级超时 (Temporal Start-To-Close)
    - state: 任务执行状态
    - input_data / output_data: 输入输出
    """

    def __init__(
        self,
        task_id: str,
        agent_id: str,
        name: str = "",
        dependencies: list[str] | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.task_id = task_id
        self.agent_id = agent_id
        self.name = name or task_id
        self.dependencies = list(dependencies) if dependencies else []
        self.timeout_s = timeout_s
        self.state = OrchestrationState.PENDING
        self.input_data: dict[str, Any] = {}
        self.output_data: dict[str, Any] = {}
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.error: str | None = None
        self.retry_count = 0


# ============================================================
# OrchestrationPlan — DAG 执行计划
# ============================================================


class OrchestrationPlan:
    """DAG 执行计划 (融合 LangGraph StateGraph + Temporal Workflow).

    核心能力:
    1. DAG 构建: 从任务列表构建有向依赖图
    2. 拓扑排序: Kahn 算法 (确定执行顺序)
    3. 并行层: 按拓扑分层, 同层任务无依赖可并行 (LangGraph superstep)
    4. 循环检测: 构建时检测循环依赖
    5. 依赖完整性: 检测缺失依赖

    编排参数 (Temporal retry policy):
    - max_retries: 最大重试次数
    - retry_backoff_s: 重试退避时间
    - total_timeout_s: 计划级总超时 (Schedule-To-Close)
    """

    def __init__(
        self,
        plan_id: str,
        tasks: list[OrchestrationTask],
        paradigm: OrchestrationParadigm = OrchestrationParadigm.PIPELINE,
        max_retries: int = 1,
        retry_backoff_s: float = 1.0,
        total_timeout_s: float | None = None,
    ) -> None:
        self.plan_id = plan_id
        self.tasks = tasks
        self.paradigm = paradigm
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.total_timeout_s = total_timeout_s
        self.state = OrchestrationState.PENDING
        self.created_at = time.time()

        # 构建索引和 DAG
        self._task_map: dict[str, OrchestrationTask] = {t.task_id: t for t in tasks}
        self._validate_and_build_dag()

    def _validate_and_build_dag(self) -> None:
        """验证 DAG 完整性 (循环检测 + 缺失依赖检测)."""
        # 检查缺失依赖
        for task in self.tasks:
            for dep in task.dependencies:
                if dep not in self._task_map:
                    raise OrchestrationError(
                        f"missing dependency: task '{task.task_id}' depends on "
                        f"non-existent task '{dep}'",
                        retryable=False,
                    )

        # 循环检测 (Kahn 算法)
        in_degree: dict[str, int] = {t.task_id: 0 for t in self.tasks}
        adj: dict[str, list[str]] = defaultdict(list)

        for task in self.tasks:
            for dep in task.dependencies:
                adj[dep].append(task.task_id)
                in_degree[task.task_id] += 1

        queue = deque([tid for tid, d in in_degree.items() if d == 0])
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(self.tasks):
            raise OrchestrationError(
                "Dependency cycle detected in orchestration plan",
                retryable=False,
            )

    def topological_sort(self) -> list[str]:
        """Kahn 算法拓扑排序 (返回任务 ID 顺序)."""
        in_degree: dict[str, int] = {t.task_id: 0 for t in self.tasks}
        adj: dict[str, list[str]] = defaultdict(list)

        for task in self.tasks:
            for dep in task.dependencies:
                adj[dep].append(task.task_id)
                in_degree[task.task_id] += 1

        queue = deque([tid for tid, d in in_degree.items() if d == 0])
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def get_parallel_layers(self) -> list[list[str]]:
        """构建并行层 (LangGraph superstep 模式).

        每层内的任务无依赖关系, 可并行执行.
        层间有严格依赖, 必须按顺序执行.
        """
        in_degree: dict[str, int] = {t.task_id: 0 for t in self.tasks}
        adj: dict[str, list[str]] = defaultdict(list)

        for task in self.tasks:
            for dep in task.dependencies:
                adj[dep].append(task.task_id)
                in_degree[task.task_id] += 1

        layers: list[list[str]] = []
        current_layer = [tid for tid, d in in_degree.items() if d == 0]

        while current_layer:
            layers.append(current_layer)
            next_layer: list[str] = []
            for node in current_layer:
                for neighbor in adj[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_layer.append(neighbor)
            current_layer = next_layer

        return layers

    def get_task(self, task_id: str) -> OrchestrationTask | None:
        """按 ID 获取任务."""
        return self._task_map.get(task_id)

    @property
    def task_ids(self) -> list[str]:
        """所有任务 ID."""
        return list(self._task_map.keys())


# ============================================================
# OrchestrationResult — 执行结果
# ============================================================


class OrchestrationResult:
    """编排执行结果 (含溯源记录, Temporal 事件历史模式).

    记录:
    - plan_id: 计划 ID
    - state: 最终状态
    - outputs: 各任务输出 (task_id → output)
    - error: 错误信息 (失败时)
    - execution_time_s: 总执行时间
    - provenance: 溯源记录列表
    """

    def __init__(
        self,
        plan_id: str,
        state: OrchestrationState,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.plan_id = plan_id
        self.state = state
        self.outputs = outputs or {}
        self.error = error
        self.execution_time_s = 0.0
        self.provenance: list[dict[str, Any]] = []
        self.requires_adjudication = False
        self.voting_result: VotingResult | None = None
        self.debate_rounds: list[DebateRound] = []
        self._start_time = time.time()
        self.add_provenance("plan.start", {"plan_id": plan_id})

    def add_provenance(self, action: str, context: dict[str, Any] | None = None) -> None:
        """添加溯源记录."""
        self.provenance.append({
            "action": action,
            "context": context or {},
            "timestamp": time.time(),
        })

    def finalize(self) -> None:
        """完成结果 (计算执行时间, 记录完成溯源)."""
        self.execution_time_s = time.time() - self._start_time
        self.add_provenance("plan.complete", {
            "state": self.state.value,
            "execution_time_s": self.execution_time_s,
        })


# ============================================================
# Debate 数据模型
# ============================================================


class DebateMessage:
    """辩论消息 (Claude Science 辩论弧模式).

    每条消息包含:
    - sender: 发送方 Agent ID
    - round: 辩论轮次
    - arguments: 论据列表 (受代币预算限制)
    - confidence: 置信度 (用于收敛判断)
    """

    def __init__(
        self,
        sender: str,
        round: int,
        arguments: list[str] | None = None,
        confidence: float = 0.5,
    ) -> None:
        self.sender = sender
        self.round = round
        self.arguments = arguments or []
        self.confidence = confidence
        self.timestamp = time.time()


class DebateRound:
    """辩论轮次 (一轮包含 Pro 和 Con 双方消息)."""

    def __init__(
        self,
        round_num: int,
        pro_messages: list[DebateMessage] | None = None,
        con_messages: list[DebateMessage] | None = None,
    ) -> None:
        self.round_num = round_num
        self.pro_messages = pro_messages or []
        self.con_messages = con_messages or []
        self.divergence: float = 0.0
        self.timestamp = time.time()

    def compute_divergence(self) -> float:
        """计算分歧度: |conf_pro - conf_con| / (conf_pro + conf_con)."""
        if not self.pro_messages or not self.con_messages:
            return 1.0
        pro_conf = self.pro_messages[-1].confidence
        con_conf = self.con_messages[-1].confidence
        total = pro_conf + con_conf
        if total == 0:
            return 1.0
        self.divergence = abs(pro_conf - con_conf) / total
        return self.divergence


# ============================================================
# Voting 数据模型
# ============================================================


class VoteRecord:
    """投票记录 (含声誉加权, L0 Reputation Ledger 联动).

    final_weight = confidence * (1 + reputation_score)
    """

    def __init__(
        self,
        strategy_id: str,
        agent_id: str,
        output: dict[str, Any],
        confidence: float,
        reputation_score: float = 0.5,
    ) -> None:
        self.strategy_id = strategy_id
        self.agent_id = agent_id
        self.output = output
        self.confidence = confidence
        self.reputation_score = reputation_score
        self.weight = confidence * (1 + reputation_score)
        self.timestamp = time.time()


class VotingResult:
    """投票结果 (含共识度计算)."""

    def __init__(
        self,
        consensus_score: float,
        final_output: dict[str, Any] | None,
        votes: list[VoteRecord],
        reached_consensus: bool,
    ) -> None:
        self.consensus_score = consensus_score
        self.final_output = final_output or {}
        self.votes = votes
        self.reached_consensus = reached_consensus
        self.timestamp = time.time()


# ============================================================
# ParadigmExecutor — 范式执行器抽象基类
# ============================================================


class ParadigmExecutor(ABC):
    """编排范式执行器抽象基类.

    每种编排范式 (PIPELINE/DEBATE/VOTING) 实现自己的执行逻辑.
    """

    @property
    @abstractmethod
    def paradigm(self) -> OrchestrationParadigm:
        """返回范式类型."""
        ...

    @abstractmethod
    async def execute(
        self,
        plan: OrchestrationPlan,
        execute_fn: Callable[[OrchestrationTask, dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> OrchestrationResult:
        """执行编排计划."""
        ...


# ============================================================
# PipelineExecutor — 顺序流水线编排
# ============================================================


class PipelineExecutor(ParadigmExecutor):
    """顺序流水线执行器 (融合 Google ADK SequentialAgent + LangGraph superstep).

    核心特性:
    1. 按拓扑排序顺序执行任务
    2. 同层无依赖任务并行执行 (LangGraph superstep 模式)
    3. 上一步输出传递给下一步 (ADK output_key 模式)
    4. 任务级超时控制 (Temporal Start-To-Close)
    5. 任务失败终止流水线
    """

    def __init__(self) -> None:
        self._paradigm = OrchestrationParadigm.PIPELINE

    @property
    def paradigm(self) -> OrchestrationParadigm:
        return self._paradigm

    async def execute(
        self,
        plan: OrchestrationPlan,
        execute_fn: Callable[[OrchestrationTask, dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> OrchestrationResult:
        """执行流水线编排."""
        result = OrchestrationResult(
            plan_id=plan.plan_id,
            state=OrchestrationState.RUNNING,
        )
        plan.state = OrchestrationState.RUNNING

        layers = plan.get_parallel_layers()
        context: dict[str, Any] = {}  # 累积上下文 (ADK output_key)

        try:
            for layer_idx, layer in enumerate(layers):
                # LangGraph superstep: 同层任务并行执行
                if len(layer) == 1:
                    # 单任务: 直接执行
                    task = plan.get_task(layer[0])
                    if task is None:
                        raise OrchestrationError(
                            f"Task '{layer[0]}' not found in plan",
                            retryable=False,
                        )
                    task.input_data = dict(context)
                    output = await self._execute_task_with_timeout(task, context, execute_fn)
                    task.output_data = output
                    task.state = OrchestrationState.COMPLETED
                    context[task.task_id] = output
                    result.outputs[task.task_id] = output
                    result.add_provenance("task.complete", {
                        "task_id": task.task_id,
                        "layer": layer_idx,
                    })
                else:
                    # 多任务: 并行执行 (LangGraph fan-out)
                    tasks_in_layer: list[OrchestrationTask] = []
                    for tid in layer:
                        t = plan.get_task(tid)
                        if t is None:
                            raise OrchestrationError(
                                f"Task '{tid}' not found in plan",
                                retryable=False,
                            )
                        tasks_in_layer.append(t)
                    coroutines = [
                        self._execute_task_with_timeout(t, dict(context), execute_fn)
                        for t in tasks_in_layer
                    ]
                    outputs = await asyncio.gather(*coroutines, return_exceptions=True)

                    for task, output in zip(tasks_in_layer, outputs):
                        if isinstance(output, Exception):
                            raise output
                        task.output_data = output
                        task.state = OrchestrationState.COMPLETED
                        context[task.task_id] = output
                        result.outputs[task.task_id] = output
                        result.add_provenance("task.complete", {
                            "task_id": task.task_id,
                            "layer": layer_idx,
                        })

            result.state = OrchestrationState.COMPLETED
            plan.state = OrchestrationState.COMPLETED

        except OrchestrationTimeoutError as e:
            result.state = OrchestrationState.FAILED
            result.error = f"Task timeout: {e.message}"
            plan.state = OrchestrationState.FAILED
            result.add_provenance("task.timeout", {"error": e.message})

        except Exception as e:
            result.state = OrchestrationState.FAILED
            result.error = str(e)
            plan.state = OrchestrationState.FAILED
            result.add_provenance("task.failed", {"error": str(e)})

        result.finalize()
        return result

    async def _execute_task_with_timeout(
        self,
        task: OrchestrationTask,
        context: dict[str, Any],
        execute_fn: Callable[[OrchestrationTask, dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """带超时执行单个任务 (Temporal Start-To-Close).

        使用 asyncio.wait 替代 asyncio.wait_for 以避免嵌套 wait_for
        在 Python 3.10 中的 CancelledError 吞没问题.
        """
        task.state = OrchestrationState.RUNNING
        task.started_at = time.time()

        exec_task = asyncio.ensure_future(execute_fn(task, context))
        try:
            done, _pending = await asyncio.wait(
                {exec_task}, timeout=task.timeout_s,
            )

            if exec_task in done:
                task.completed_at = time.time()
                # 获取结果 (可能抛出异常)
                return exec_task.result()
            # 超时
            exec_task.cancel()
            task.state = OrchestrationState.TIMEOUT
            raise OrchestrationTimeoutError(
                f"Task '{task.task_id}' exceeded timeout of {task.timeout_s}s",
                timeout_type="start_to_close",
            )
        except OrchestrationTimeoutError:
            raise
        except asyncio.CancelledError:
            # 外部取消 (如计划级超时) — 清理并传播
            exec_task.cancel()
            raise
        except Exception:
            task.state = OrchestrationState.FAILED
            raise


# ============================================================
# DebateExecutor — 辩论交叉验证编排
# ============================================================


class DebateExecutor(ParadigmExecutor):
    """辩论交叉验证执行器 (Claude Science 辩论弧模式).

    核心特性 (L5 设计文档 4.3.1 节):
    1. Pro/Con 双方并行启动, 通过辩论弧交换论据
    2. 收敛阈值: divergence < 0.1 时提前终止
    3. 代币预算: 3 轮 × 每轮 3 论据 = 9 个论据点上限
    4. 最大轮数限制: 3 轮
    5. 未收敛时标记需要裁决 (requires_adjudication)
    6. 轮次超时控制
    """

    def __init__(
        self,
        max_rounds: int = 3,
        arguments_per_round: int = 3,
        convergence_threshold: float = 0.1,
        round_timeout_s: float | None = None,
    ) -> None:
        self._paradigm = OrchestrationParadigm.DEBATE
        self.max_rounds = max_rounds
        self.arguments_per_round = arguments_per_round
        self.convergence_threshold = convergence_threshold
        self.round_timeout_s = round_timeout_s

    @property
    def paradigm(self) -> OrchestrationParadigm:
        return self._paradigm

    async def execute(
        self,
        plan: OrchestrationPlan,
        execute_fn: Callable[[OrchestrationTask, dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> OrchestrationResult:
        """执行辩论编排 (委托给 execute_debate)."""
        # 对于通过 OrchestrationPlan 调用的情况, 取前两个任务作为 Pro/Con
        tasks = plan.tasks
        if len(tasks) < 2:
            raise OrchestrationError("Debate requires at least 2 tasks (pro and con)")

        # 包装 execute_fn 适配辩论接口
        async def debate_execute(task, context, round_num, opponent_args):
            merged_ctx = {**context, "round": round_num, "opponent_args": opponent_args}
            return await execute_fn(task, merged_ctx)

        return await self.execute_debate(
            pro_task=tasks[0],
            con_task=tasks[1],
            execute_fn=debate_execute,
        )

    async def execute_debate(
        self,
        pro_task: OrchestrationTask,
        con_task: OrchestrationTask,
        execute_fn: Callable[
            [OrchestrationTask, dict[str, Any], int, list[str]],
            Awaitable[DebateMessage],
        ],
    ) -> OrchestrationResult:
        """执行辩论交叉验证."""
        result = OrchestrationResult(
            plan_id=f"debate-{uuid.uuid4().hex[:8]}",
            state=OrchestrationState.RUNNING,
        )
        result.add_provenance("debate.start", {
            "pro_agent": pro_task.agent_id,
            "con_agent": con_task.agent_id,
            "max_rounds": self.max_rounds,
        })

        debate_rounds: list[DebateRound] = []
        token_budget = self.max_rounds * self.arguments_per_round
        tokens_used = 0

        try:
            for round_num in range(1, self.max_rounds + 1):
                # 检查代币预算
                if tokens_used >= token_budget:
                    result.add_provenance("debate.token_budget_exhausted", {
                        "tokens_used": tokens_used,
                        "budget": token_budget,
                    })
                    break

                # 收集上一轮对手论据
                pro_opponent_args: list[str] = []
                con_opponent_args: list[str] = []
                if debate_rounds:
                    last_round = debate_rounds[-1]
                    con_opponent_args = [
                        arg for msg in last_round.con_messages for arg in msg.arguments
                    ]
                    pro_opponent_args = [
                        arg for msg in last_round.pro_messages for arg in msg.arguments
                    ]

                # 并行执行 Pro 和 Con (Claude Science 双工通道)
                context: dict[str, Any] = {"round": round_num}

                coroutines = [
                    self._execute_debate_round(
                        pro_task, context, round_num, con_opponent_args, execute_fn,
                    ),
                    self._execute_debate_round(
                        con_task, context, round_num, pro_opponent_args, execute_fn,
                    ),
                ]

                try:
                    pro_msg, con_msg = await asyncio.gather(*coroutines)
                except OrchestrationTimeoutError as e:
                    result.state = OrchestrationState.FAILED
                    result.error = f"Debate round timeout: {e.message}"
                    result.add_provenance("debate.timeout", {"round": round_num})
                    result.finalize()
                    return result

                # 统计代币
                tokens_used += len(pro_msg.arguments) + len(con_msg.arguments)

                # 创建辩论轮次
                debate_round = DebateRound(
                    round_num=round_num,
                    pro_messages=[pro_msg],
                    con_messages=[con_msg],
                )
                divergence = debate_round.compute_divergence()
                debate_rounds.append(debate_round)

                result.add_provenance("debate.round_complete", {
                    "round": round_num,
                    "divergence": divergence,
                    "tokens_used": tokens_used,
                })

                # 检查收敛 (L5 设计文档: divergence < 0.1)
                if divergence < self.convergence_threshold:
                    result.state = OrchestrationState.COMPLETED
                    result.add_provenance("debate.converged", {
                        "round": round_num,
                        "divergence": divergence,
                    })
                    break

            else:
                # 达到最大轮数仍未收敛
                result.state = OrchestrationState.COMPLETED
                result.requires_adjudication = True
                result.add_provenance("debate.max_rounds_reached", {
                    "rounds": self.max_rounds,
                    "requires_adjudication": True,
                })

        except Exception as e:
            result.state = OrchestrationState.FAILED
            result.error = str(e)
            result.add_provenance("debate.error", {"error": str(e)})

        result.debate_rounds = debate_rounds
        result.finalize()
        return result

    async def _execute_debate_round(
        self,
        task: OrchestrationTask,
        context: dict[str, Any],
        round_num: int,
        opponent_args: list[str],
        execute_fn: Callable,
    ) -> DebateMessage:
        """执行单轮辩论 (带超时)."""
        if self.round_timeout_s:
            try:
                return await asyncio.wait_for(
                    execute_fn(task, context, round_num, opponent_args),
                    timeout=self.round_timeout_s,
                )
            except asyncio.TimeoutError:
                raise OrchestrationTimeoutError(
                    f"Debate round {round_num} timeout",
                    timeout_type="debate_round",
                )
        return await execute_fn(task, context, round_num, opponent_args)


# ============================================================
# VotingExecutor — 投票共识编排
# ============================================================


class VotingExecutor(ParadigmExecutor):
    """投票共识执行器 (LangGraph 并行 fan-out + 加权聚合).

    核心特性 (L5 设计文档 4.3.3 节):
    1. 3-5 种独立策略并行执行 (LangGraph fan-out)
    2. 共识度计算: agree_pairs / total_pairs (相似度 > 0.8 视为一致)
    3. 声誉加权: final_weight = confidence * (1 + reputation_score)
    4. 并发上限受 Fork 并发 5 约束
    5. 未达共识自动重试 (更换策略组合)
    6. 重试后仍不达共识标记需要裁决
    """

    def __init__(
        self,
        min_strategies: int = 3,
        max_strategies: int = 5,
        consensus_threshold: float = 0.5,
        similarity_threshold: float = 0.8,
    ) -> None:
        self._paradigm = OrchestrationParadigm.VOTING
        self.min_strategies = min_strategies
        self.max_strategies = max_strategies
        self.consensus_threshold = consensus_threshold
        self.similarity_threshold = similarity_threshold

    @property
    def paradigm(self) -> OrchestrationParadigm:
        return self._paradigm

    async def execute(
        self,
        plan: OrchestrationPlan,
        execute_fn: Callable[[OrchestrationTask, dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> OrchestrationResult:
        """执行投票编排 (委托给 execute_voting)."""
        return await self.execute_voting(
            strategies=plan.tasks,
            execute_fn=execute_fn,
        )

    async def execute_voting(
        self,
        strategies: list[OrchestrationTask],
        execute_fn: Callable[[OrchestrationTask, dict[str, Any]], Awaitable[dict[str, Any]]],
        reputations: dict[str, float] | None = None,
        max_retries: int = 0,
    ) -> OrchestrationResult:
        """执行投票共识编排."""
        reputations = reputations or {}
        result = OrchestrationResult(
            plan_id=f"voting-{uuid.uuid4().hex[:8]}",
            state=OrchestrationState.RUNNING,
        )
        result.add_provenance("voting.start", {
            "strategies": len(strategies),
            "consensus_threshold": self.consensus_threshold,
        })

        attempt = 0
        while True:
            attempt += 1

            # 并行执行所有策略 (LangGraph fan-out)
            context: dict[str, Any] = {"attempt": attempt}
            coroutines = [
                self._execute_strategy(s, context, execute_fn) for s in strategies
            ]
            raw_outputs = await asyncio.gather(*coroutines, return_exceptions=True)

            # 构建 VoteRecord 列表
            votes: list[VoteRecord] = []
            for strategy, output in zip(strategies, raw_outputs):
                if isinstance(output, Exception):
                    result.add_provenance("voting.strategy_failed", {
                        "strategy_id": strategy.task_id,
                        "error": str(output),
                    })
                    continue

                confidence = output.get("confidence", 0.5) if isinstance(output, dict) else 0.5
                rep_score = reputations.get(strategy.agent_id, output.get("reputation", 0.5) if isinstance(output, dict) else 0.5)

                vote = VoteRecord(
                    strategy_id=strategy.task_id,
                    agent_id=strategy.agent_id,
                    output=output if isinstance(output, dict) else {"result": str(output)},
                    confidence=confidence,
                    reputation_score=rep_score,
                )
                votes.append(vote)

            if not votes:
                result.state = OrchestrationState.FAILED
                result.error = "All strategies failed"
                result.finalize()
                return result

            # 计算共识度
            consensus_score, final_output = self._compute_consensus(votes)
            reached = consensus_score >= self.consensus_threshold

            voting_result = VotingResult(
                consensus_score=consensus_score,
                final_output=final_output,
                votes=votes,
                reached_consensus=reached,
            )

            result.add_provenance("voting.round_complete", {
                "attempt": attempt,
                "consensus_score": consensus_score,
                "reached_consensus": reached,
            })

            if reached or attempt > max_retries:
                result.voting_result = voting_result
                result.state = OrchestrationState.COMPLETED
                result.outputs = final_output
                if not reached:
                    result.requires_adjudication = True
                    result.add_provenance("voting.no_consensus", {
                        "requires_adjudication": True,
                    })
                else:
                    result.add_provenance("voting.consensus_reached", {
                        "score": consensus_score,
                    })
                break

            # 重试: 等待一小段时间后重新执行
            result.add_provenance("voting.retry", {"attempt": attempt})

        result.finalize()
        return result

    async def _execute_strategy(
        self,
        task: OrchestrationTask,
        context: dict[str, Any],
        execute_fn: Callable,
    ) -> dict[str, Any]:
        """执行单个投票策略."""
        task.state = OrchestrationState.RUNNING
        task.started_at = time.time()
        try:
            output = await execute_fn(task, context)
            task.state = OrchestrationState.COMPLETED
            task.completed_at = time.time()
            task.output_data = output
            return output
        except Exception:
            task.state = OrchestrationState.FAILED
            raise

    def _compute_consensus(self, votes: list[VoteRecord]) -> tuple[float, dict[str, Any]]:
        """计算共识度 (相似度 > threshold 视为一致).

        consensus_score = agree_pairs / total_pairs
        final_output = 加权最高票
        """
        if len(votes) < 2:
            return 1.0, votes[0].output if votes else {}

        agree_pairs = 0
        total_pairs = 0

        for i in range(len(votes)):
            for j in range(i + 1, len(votes)):
                total_pairs += 1
                if self._is_similar(votes[i].output, votes[j].output):
                    agree_pairs += 1

        consensus_score = agree_pairs / total_pairs if total_pairs > 0 else 0.0

        # 选择加权最高的输出
        best_vote = max(votes, key=lambda v: v.weight)
        return consensus_score, best_vote.output

    def _is_similar(self, a: dict[str, Any], b: dict[str, Any]) -> bool:
        """判断两个输出是否相似 (简化: 比较 answer 字段)."""
        a_answer = str(a.get("answer", a))
        b_answer = str(b.get("answer", b))

        # 简单相似度: 完全匹配或 Jaccard 相似度
        if a_answer == b_answer:
            return True

        # Jaccard 相似度 (基于字符集)
        a_set = set(a_answer)
        b_set = set(b_answer)
        if not a_set and not b_set:
            return True
        intersection = a_set & b_set
        union = a_set | b_set
        jaccard = len(intersection) / len(union) if union else 0.0

        return jaccard >= self.similarity_threshold


# ============================================================
# OrchestrationEngine — 顶层编排引擎
# ============================================================


class OrchestrationEngine:
    """顶层编排引擎 (融合 Temporal Workflow + LangGraph StateGraph).

    核心能力:
    1. 根据范式路由到对应执行器 (Strategy Pattern)
    2. 重试策略 (Temporal retry policy: max_retries + backoff)
    3. 计划级超时 (Temporal Schedule-To-Close)
    4. 不可重试错误立即失败
    5. 完整溯源记录 (Temporal 事件历史)

    融合方案:
    - Temporal: Workflow/Activity 分离, retry policy, 超时管理
    - LangGraph: 图模型, superstep 粒度恢复
    - Google ADK: 声明式范式选择
    - AutoGen: 多策略 Manager
    """

    def __init__(
        self,
        *,
        adjudication_executor: Any | None = None,
        quality_gate: Any | None = None,
        reflection_engine: Any | None = None,
    ) -> None:
        self._executors: dict[OrchestrationParadigm, ParadigmExecutor] = {
            OrchestrationParadigm.PIPELINE: PipelineExecutor(),
            OrchestrationParadigm.DEBATE: DebateExecutor(),
            OrchestrationParadigm.VOTING: VotingExecutor(),
        }
        self._lock = asyncio.Lock()
        self._adjudication_executor = adjudication_executor
        self._quality_gate = quality_gate
        self._reflection_engine = reflection_engine
        # 编排结果缓存: plan_id -> OrchestrationResult
        self._results: dict[str, OrchestrationResult] = {}

    def get_result(self, plan_id: str) -> OrchestrationResult | None:
        """按 plan_id 获取编排结果 (None 表示不存在)."""
        return self._results.get(plan_id)

    def register_executor(
        self,
        paradigm: OrchestrationParadigm,
        executor: ParadigmExecutor,
    ) -> None:
        """注册自定义范式执行器 (可插拔, OpenAI Agents SDK 模式)."""
        self._executors[paradigm] = executor

    async def execute(
        self,
        plan: OrchestrationPlan,
        execute_fn: Callable[[OrchestrationTask, dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> OrchestrationResult:
        """执行编排计划.

        流程 (Temporal Workflow 模式):
        1. 选择范式执行器
        2. 执行计划 (含重试)
        3. 计划级超时保护
        4. 记录溯源
        """
        executor = self._executors.get(plan.paradigm)
        if executor is None:
            raise OrchestrationError(
                f"No executor registered for paradigm: {plan.paradigm}",
                retryable=False,
            )

        # 计划级超时 (Temporal Schedule-To-Close)
        if plan.total_timeout_s:
            try:
                result = await asyncio.wait_for(
                    self._execute_with_retry(plan, executor, execute_fn),
                    timeout=plan.total_timeout_s,
                )
                self._results[plan.plan_id] = result
                return result
            except asyncio.TimeoutError:
                result = OrchestrationResult(
                    plan_id=plan.plan_id,
                    state=OrchestrationState.FAILED,
                    error=f"Plan timeout: exceeded {plan.total_timeout_s}s",
                )
                result.add_provenance("plan.timeout", {
                    "total_timeout_s": plan.total_timeout_s,
                })
                result.finalize()
                self._results[plan.plan_id] = result
                return result

        result = await self._execute_with_retry(plan, executor, execute_fn)
        self._results[plan.plan_id] = result
        return result

    async def trigger_collaboration_review(
        self,
        session_id: str,
        result: OrchestrationResult,
        participants: list[str],
        paradigm: OrchestrationParadigm,
    ) -> None:
        """触发跨 Agent 协作复盘 (L5 设计文档 7.2.1).

        在辩论/投票完成后, 调用 ReflectionEngine 进行联合复盘.
        未配置 reflection_engine 时静默跳过 (向后兼容).

        Args:
            session_id: 会话 ID
            result: 编排结果
            participants: 参与 Agent 列表
            paradigm: 编排范式 (DEBATE/VOTING)
        """
        if self._reflection_engine is None:
            logger.debug(
                "[OrchestrationEngine] No reflection_engine configured, "
                "skipping collaboration review"
            )
            return

        from .reflection_quality import CollaborationTrigger

        # 根据范式映射触发类型
        trigger_map = {
            OrchestrationParadigm.DEBATE: CollaborationTrigger.DEBATE,
            OrchestrationParadigm.VOTING: CollaborationTrigger.VOTING,
        }
        trigger = trigger_map.get(paradigm, CollaborationTrigger.DEBATE)

        # 构建协作指标
        metrics = {
            "total_duration_s": result.execution_time_s,
            "consensus_confidence": 0.0,
            "total_token_cost": 0,
        }

        # 从投票结果中提取共识置信度
        if result.voting_result is not None:
            metrics["consensus_confidence"] = result.voting_result.consensus_score

        # 从辩论轮次中提取分歧信息
        if result.debate_rounds:
            metrics["disagreement_points"] = len(result.debate_rounds)
            metrics["compromise_count"] = sum(
                1 for r in result.debate_rounds if r.divergence < 0.3
            )

        await self._reflection_engine.collaboration_review(
            session_id=session_id,
            trigger=trigger,
            participants=participants,
            metrics=metrics,
        )

        result.add_provenance("plan.collaboration_review", {
            "trigger": trigger.value,
            "participants": len(participants),
        })

    async def handle_adjudication(
        self,
        result: OrchestrationResult,
    ) -> Any | None:
        """处理需要裁决的结果 (集成 AdjudicationExecutor).

        当 OrchestrationResult.requires_adjudication 为 True 时,
        使用 AdjudicationExecutor 进行裁决处理.

        Args:
            result: 编排结果

        Returns:
            AdjudicationResult 或 None (无需裁决时)
        """
        if not result.requires_adjudication:
            return None

        if self._adjudication_executor is None:
            logger.warning(
                "[OrchestrationEngine] Result requires adjudication "
                "but no AdjudicationExecutor configured"
            )
            return None

        from .reflection_quality import (
            AdjudicationResult as _AdjResult,
            CC1Reviewer,
            DimensionScore,
            GateAction,
            QualityGate as _QG,
            ReflectionDimension,
            ReviewRecord,
            Verdict,
        )

        # 构建 ReviewRecord (从编排结果中提取)
        review = ReviewRecord(
            artifact_id=result.plan_id,
            reviewer="orchestration.adjudicator",
            dimension_scores=[
                DimensionScore(
                    dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
                    score=0.7,
                    reasoning="编排完成, 需要裁决评估",
                ),
            ],
            verdict=Verdict.REVISE,
            iteration=1,
        )

        adj_result = await self._adjudication_executor.adjudicate(
            review=review,
        )

        result.add_provenance("plan.adjudicated", {
            "action": adj_result.action.value,
            "passed": adj_result.passed,
            "score": adj_result.score,
        })

        return adj_result

    async def _execute_with_retry(
        self,
        plan: OrchestrationPlan,
        executor: ParadigmExecutor,
        execute_fn: Callable,
    ) -> OrchestrationResult:
        """带重试执行 (Temporal retry policy 模式)."""
        max_retries = plan.max_retries
        backoff = plan.retry_backoff_s

        for attempt in range(max_retries + 1):
            try:
                # 包装 execute_fn 加入重试逻辑
                wrapped_fn = self._wrap_execute_fn(execute_fn, attempt)
                result = await executor.execute(plan, wrapped_fn)

                if result.state == OrchestrationState.COMPLETED:
                    return result

                # 检查是否可重试
                if attempt < max_retries and self._is_retryable_error(result.error):
                    logger.warning(
                        f"[OrchestrationEngine] Attempt {attempt + 1} failed, "
                        f"retrying in {backoff}s... (error: {result.error})"
                    )
                    await asyncio.sleep(backoff)
                    # 重置任务状态
                    for task in plan.tasks:
                        task.state = OrchestrationState.PENDING
                        task.error = None
                    continue

                return result

            except OrchestrationError as e:
                if not e.retryable or attempt >= max_retries:
                    result = OrchestrationResult(
                        plan_id=plan.plan_id,
                        state=OrchestrationState.FAILED,
                        error=str(e),
                    )
                    result.add_provenance("plan.error", {
                        "attempt": attempt + 1,
                        "error": str(e),
                        "retryable": e.retryable,
                    })
                    result.finalize()
                    return result

                logger.warning(
                    f"[OrchestrationEngine] Attempt {attempt + 1} error, "
                    f"retrying in {backoff}s... (error: {e})"
                )
                await asyncio.sleep(backoff)

        # 所有重试失败
        result = OrchestrationResult(
            plan_id=plan.plan_id,
            state=OrchestrationState.FAILED,
            error=f"Max retries ({max_retries}) exceeded",
        )
        result.add_provenance("plan.max_retries_exceeded", {
            "max_retries": max_retries,
        })
        result.finalize()
        return result

    def _wrap_execute_fn(self, execute_fn: Callable, attempt: int) -> Callable:
        """包装执行函数, 注入重试信息."""
        async def wrapped(task: OrchestrationTask, context: dict[str, Any]) -> dict[str, Any]:
            task.retry_count = attempt
            context["_attempt"] = attempt
            return await execute_fn(task, context)

        return wrapped

    def _is_retryable_error(self, error: Any) -> bool:
        """判断错误是否可重试 (Temporal non_retryable_error_types 模式).

        支持两种输入:
        - 字符串: 错误消息或错误类型名
        - Exception 对象: 自动提取类型名

        当配置了 QualityGate 时, 使用其 is_retryable 方法.
        否则回退到硬编码模式判断.
        """
        if error is None:
            return False

        # 提取错误类型名
        if isinstance(error, BaseException):
            error_type_name = type(error).__name__
            error_str = str(error)
        else:
            error_type_name = str(error)
            error_str = str(error).lower()

        # 优先使用 QualityGate (集成 reflection_quality 模块)
        if self._quality_gate is not None:
            return self._quality_gate.is_retryable(error_type_name)

        # 回退: 硬编码模式 (字符串匹配)
        error_lower = error_str.lower() if error_str else error_type_name.lower()
        non_retryable_patterns = [
            "non-retryable",
            "validation error",
            "validationerror",
            "permission denied",
            "permissionerror",
            "not found",
        ]
        return not any(p in error_lower for p in non_retryable_patterns)
