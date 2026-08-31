"""L4 决策引擎层 — 核心数据模型.

融合世界先进方案的决策引擎数据模型:
- LangGraph StateGraph: 状态化节点 + 条件边
- TDP 框架: Supervisor-Planner-Executor 三层上下文隔离
- PRISM MHCV: 多维度验证结果聚合
- OLIVIA: 四级行动策略映射

数据契约流转:
    T1(RoutedResult) → T2(DecisionPlan) → T3(ExecutionResult)
    → T4(ValidationReport) → T5(ActionRecord) → T6(FeedbackSignal)
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# 枚举定义
# ============================================================


class TaskType(str, Enum):
    """子任务类型 (借鉴 LangGraph 节点分类 + TDP 任务分解).

    RETRIEVE:   知识检索 — 调用向量/关键词/图/混合检索器
    REASON:     图推理 — 调用路径查找/多跳/规则/链接预测/模式匹配/类比
    VERIFY:     事实验证 — 调用数值校验/质量评估/冲突检测
    SYNTHESIZE: 信息合成 — 调用响应合成/子图摘要
    """

    RETRIEVE = "retrieve"
    REASON = "reason"
    VERIFY = "verify"
    SYNTHESIZE = "synthesize"


class ExecutionMode(str, Enum):
    """执行模式 (借鉴 TDP 框架 + LangGraph 并行节点).

    SEQUENTIAL: 串行执行 — 严格按 DAG 拓扑序，有依赖的任务依次执行
    PARALLEL:   并行执行 — 无依赖的任务并发执行（asyncio.gather）
    ITERATIVE:  迭代执行 — 多轮收敛，每轮结束后检查终止条件
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ITERATIVE = "iterative"


class ExecutionStatus(str, Enum):
    """执行状态."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"      # 部分成功（有子任务失败但继续执行）
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class RetrievalStrategy(str, Enum):
    """检索策略标识.

    VECTOR:      向量相似性检索
    KEYWORD:     BM25 关键词检索
    GRAPH:       图遍历检索
    HYBRID:      混合检索（RRF 融合）
    GRAPHRAG:    GraphRAG 双通道检索
    SUBGRAPH:    子图提取+推理
    """

    VECTOR = "vector"
    KEYWORD = "keyword"
    GRAPH = "graph"
    HYBRID = "hybrid"
    GRAPHRAG = "graphrag"
    SUBGRAPH = "subgraph"


class ReasoningMode(str, Enum):
    """推理模式标识 (与 L3 GraphReasoner 对齐).

    PATH_FINDING:    路径查找（最短路径/K-最短路径）
    MULTI_HOP:       多跳推理
    RULE_INFERENCE:  规则推理（前向链式）
    LINK_PREDICTION: 链接预测
    PATTERN_MATCH:   模式匹配
    ANALOGY:         类比推理
    TRANS_E:         TransE 嵌入推理
    BACKWARD_CHAIN:  后向链式推理
    CONFIDENCE_TRAV: 置信度加权遍历
    """

    PATH_FINDING = "path_finding"
    MULTI_HOP = "multi_hop"
    RULE_INFERENCE = "rule_inference"
    LINK_PREDICTION = "link_prediction"
    PATTERN_MATCH = "pattern_match"
    ANALOGY = "analogy"
    TRANS_E = "trans_e"
    BACKWARD_CHAIN = "backward_chain"
    CONFIDENCE_TRAV = "confidence_traversal"


# ============================================================
# 数据模型
# ============================================================


class ResourceBudget(BaseModel):
    """资源预算 (借鉴 TDP 框架资源估算 + OLIVIA 成本约束).

    为每个子任务设定资源上限，防止单任务耗尽系统资源。
    """

    max_tokens: int = Field(default=4096, ge=128, description="最大 Token 数")
    max_tool_calls: int = Field(default=10, ge=0, description="最大工具调用次数")
    max_latency_ms: int = Field(default=30000, ge=100, description="最大延迟毫秒")
    max_retrieval_depth: int = Field(default=3, ge=1, le=10, description="最大检索深度")
    max_reasoning_hops: int = Field(default=5, ge=1, le=20, description="最大推理跳数")

    def is_within_budget(self, elapsed_ms: float, tool_calls: int = 0) -> bool:
        """检查是否在预算内."""
        return elapsed_ms <= self.max_latency_ms and tool_calls <= self.max_tool_calls


class SubTask(BaseModel):
    """子任务定义 (借鉴 LangGraph 节点 + TDP 子任务).

    每个子任务是 DecisionPlan 的原子执行单元，包含类型、依赖、
    推理/检索策略、资源预算和预期输出描述。
    """

    task_id: str = Field(description="子任务唯一标识")
    task_type: TaskType = Field(description="子任务类型")
    deps: list[str] = Field(default_factory=list, description="前置依赖 task_id 列表")
    reasoning_mode: ReasoningMode | None = Field(default=None, description="推理模式")
    retrieval_strategy: RetrievalStrategy | None = Field(default=None, description="检索策略")
    resource_budget: ResourceBudget = Field(default_factory=ResourceBudget, description="资源预算")
    expected_output: str = Field(default="", description="预期输出描述")
    query: str = Field(default="", description="子任务查询文本")
    params: dict[str, Any] = Field(default_factory=dict, description="额外参数")

    model_config = {"frozen": False}


class FallbackPlan(BaseModel):
    """降级计划 (借鉴 LangGraph 错误恢复 + TDP 局部重规划).

    当主计划执行失败时，触发降级策略。
    """

    trigger_condition: str = Field(default="any_failure", description="触发条件")
    fallback_mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL, description="降级执行模式")
    simplified_tasks: list[SubTask] = Field(default_factory=list, description="简化子任务列表")
    direct_retrieval: bool = Field(default=True, description="是否降级为直接检索")
    max_retries: int = Field(default=1, ge=0, le=3, description="最大重试次数")


class DecisionPlan(BaseModel):
    """决策计划 (TDP 框架核心产出).

    将 RoutedResult 转化为可执行的结构化计划，包含:
    - 子任务 DAG（拓扑有序列表）
    - 执行策略（串行/并行/迭代）
    - 资源预算
    - 降级计划
    """

    plan_id: str = Field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:12]}")
    sub_tasks: list[SubTask] = Field(default_factory=list, description="子任务列表（DAG 拓扑有序）")
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL, description="执行策略")
    fallback_plan: FallbackPlan | None = Field(default=None, description="降级计划")
    estimated_total_tokens: int = Field(default=0, ge=0, description="预估总 Token")
    estimated_total_latency_ms: int = Field(default=0, ge=0, description="预估总延迟毫秒")
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    context_id: str = Field(default="", description="关联上下文 ID")
    original_query: str = Field(default="", description="原始查询")

    model_config = {"frozen": False}

    def get_task(self, task_id: str) -> SubTask | None:
        """按 ID 获取子任务."""
        for t in self.sub_tasks:
            if t.task_id == task_id:
                return t
        return None

    def get_ready_tasks(self, completed: set[str]) -> list[SubTask]:
        """获取当前可执行的子任务（所有依赖已完成）."""
        ready: list[SubTask] = []
        for t in self.sub_tasks:
            if t.task_id in completed:
                continue
            if all(d in completed for d in t.deps):
                ready.append(t)
        return ready

    def topological_order(self) -> list[SubTask]:
        """返回拓扑排序后的子任务列表（Kahn 算法）."""
        from collections import deque

        in_degree: dict[str, int] = {t.task_id: 0 for t in self.sub_tasks}
        adj: dict[str, list[str]] = {t.task_id: [] for t in self.sub_tasks}

        for t in self.sub_tasks:
            for dep in t.deps:
                if dep in adj:
                    adj[dep].append(t.task_id)
                    in_degree[t.task_id] += 1

        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        order: list[str] = []

        while queue:
            tid = queue.popleft()
            order.append(tid)
            for next_id in adj[tid]:
                in_degree[next_id] -= 1
                if in_degree[next_id] == 0:
                    queue.append(next_id)

        if len(order) != len(self.sub_tasks):
            raise ValueError("DecisionPlan 中存在循环依赖")

        id_to_task = {t.task_id: t for t in self.sub_tasks}
        return [id_to_task[tid] for tid in order]


class TaskResult(BaseModel):
    """子任务执行结果."""

    task_id: str = Field(description="子任务 ID")
    task_type: TaskType = Field(description="子任务类型")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING, description="执行状态")
    output: dict[str, Any] = Field(default_factory=dict, description="输出数据")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="结果置信度")
    elapsed_ms: float = Field(default=0.0, ge=0.0, description="执行耗时毫秒")
    token_usage: int = Field(default=0, ge=0, description="Token 消耗")
    tool_calls: int = Field(default=0, ge=0, description="工具调用次数")
    error: str | None = Field(default=None, description="错误信息")
    evidence: list[dict[str, Any]] = Field(default_factory=list, description="证据列表")
    reasoning_chain: list[str] = Field(default_factory=list, description="推理链")

    model_config = {"frozen": False}

    @property
    def is_success(self) -> bool:
        """是否成功."""
        return self.status == ExecutionStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """是否失败."""
        return self.status in (ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT)


class ExecutionResult(BaseModel):
    """决策计划执行结果 (TaskExecutor 输出).

    汇总所有子任务的执行结果，形成统一的 ReasoningResult 风格输出。
    """

    plan_id: str = Field(description="决策计划 ID")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING, description="整体状态")
    task_results: dict[str, TaskResult] = Field(default_factory=dict, description="子任务结果映射")
    reasoning_chain: list[str] = Field(default_factory=list, description="完整推理链")
    evidence_set: list[dict[str, Any]] = Field(default_factory=list, description="证据集合")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="综合置信度")
    total_elapsed_ms: float = Field(default=0.0, ge=0.0, description="总耗时毫秒")
    total_token_usage: int = Field(default=0, ge=0, description="总 Token 消耗")
    source_metadata: dict[str, Any] = Field(default_factory=dict, description="来源元数据")
    fallback_triggered: bool = Field(default=False, description="是否触发降级")
    error_summary: str | None = Field(default=None, description="错误摘要")

    model_config = {"frozen": False}

    def get_task_result(self, task_id: str) -> TaskResult | None:
        """获取指定子任务结果."""
        return self.task_results.get(task_id)

    def get_results_by_type(self, task_type: TaskType) -> list[TaskResult]:
        """按类型获取子任务结果."""
        return [r for r in self.task_results.values() if r.task_type == task_type]

    def compute_confidence(self) -> float:
        """计算综合置信度（加权平均）."""
        results = list(self.task_results.values())
        if not results:
            return 0.0
        weights = {"reason": 0.4, "verify": 0.3, "retrieve": 0.2, "synthesize": 0.1}
        total_weight = 0.0
        weighted_sum = 0.0
        for r in results:
            w = weights.get(r.task_type.value, 0.1)
            total_weight += w
            weighted_sum += r.confidence * w
        return round(weighted_sum / total_weight, 6) if total_weight > 0 else 0.0


# ============================================================
# T4: 验证与策略评估 — 数据模型
# ============================================================


class ValidationSeverity(str, Enum):
    """验证严重级别."""

    PASS = "pass"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationTier(str, Enum):
    """验证层级 — UQ 驱动的分层验证."""

    L1_LIGHTWEIGHT = "l1_lightweight"   # 轻量验证: 自洽性 + Faithfulness 快速扫描
    L2_STANDARD = "l2_standard"         # 标准验证: 完整 RAGAS + 多文献交叉验证
    L3_DEEP = "l3_deep"                 # 深度验证: 推理链逻辑验证 + 多智能体辩论


class ValidationReport(BaseModel):
    """验证报告 (增强版 PRISM MHCV + UQ 驱动分层验证 + V&R 闭环).

    对 ExecutionResult 执行多维度验证:
    - 事实校验 (FactChecker): 数值声明与标准值比对
    - 质量评估 (QualityManager): 六维质量评分
    - 冲突检测 (ConflictDetector): 知识冲突识别
    - 合规检查 (ComplianceChecker): 策略与约束合规
    - Faithfulness 评估 (RAGAS): 生成答案与检索上下文的事实一致性
    - 自洽性检查 (Self-Consistency): 多路径推理一致性
    - 策略评估 (StrategyEvaluator): 推理策略优劣评估

    聚合方式: Discard-Weighted Voting (低质量结果丢弃后加权投票)
    """

    report_id: str = Field(default_factory=lambda: f"val-{uuid.uuid4().hex[:12]}")
    plan_id: str = Field(default="", description="关联决策计划 ID")
    overall_status: ValidationSeverity = Field(default=ValidationSeverity.PASS, description="总体状态")
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0, description="综合验证分数")

    # 验证层级
    validation_tier: ValidationTier = Field(
        default=ValidationTier.L1_LIGHTWEIGHT, description="执行的验证层级"
    )
    uq_score: float = Field(default=1.0, ge=0.0, le=1.0, description="不确定性量化分数")

    # 各维度验证结果
    fact_check: dict[str, Any] = Field(default_factory=dict, description="事实校验结果")
    quality_assessment: dict[str, Any] = Field(default_factory=dict, description="质量评估结果")
    conflict_detection: dict[str, Any] = Field(default_factory=dict, description="冲突检测结果")
    compliance_check: dict[str, Any] = Field(default_factory=dict, description="合规检查结果")

    # 增强维度
    faithfulness_assessment: dict[str, Any] = Field(
        default_factory=dict, description="RAGAS Faithfulness 评估结果"
    )
    self_consistency: dict[str, Any] = Field(
        default_factory=dict, description="自洽性检查结果"
    )
    strategy_evaluation: dict[str, Any] = Field(
        default_factory=dict, description="策略评估结果"
    )
    domain_rule_results: dict[str, Any] = Field(
        default_factory=dict, description="领域规则引擎评估结果"
    )

    # 异常与建议
    anomalies: list[dict[str, Any]] = Field(default_factory=list, description="检测到的异常")
    recommendations: list[str] = Field(default_factory=list, description="改进建议")

    # V&R 闭环信息
    refinement_iterations: int = Field(default=0, description="验证-精炼迭代轮次")
    refinement_history: list[dict[str, Any]] = Field(
        default_factory=list, description="精炼历史记录"
    )

    # 元数据
    validated_at: float = Field(default_factory=time.time, description="验证时间戳")
    validation_time_ms: float = Field(default=0.0, description="验证耗时毫秒")

    model_config = {"frozen": False}

    @property
    def is_valid(self) -> bool:
        """是否通过验证（无 ERROR/CRITICAL）."""
        return self.overall_status not in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)

    @property
    def needs_human_review(self) -> bool:
        """是否需要人工复核."""
        return self.overall_status in (ValidationSeverity.WARNING, ValidationSeverity.ERROR)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "report_id": self.report_id,
            "plan_id": self.plan_id,
            "overall_status": self.overall_status.value,
            "overall_score": round(self.overall_score, 4),
            "validation_tier": self.validation_tier.value,
            "uq_score": round(self.uq_score, 4),
            "fact_check": self.fact_check,
            "quality_assessment": self.quality_assessment,
            "conflict_detection": self.conflict_detection,
            "compliance_check": self.compliance_check,
            "faithfulness_assessment": self.faithfulness_assessment,
            "self_consistency": self.self_consistency,
            "strategy_evaluation": self.strategy_evaluation,
            "domain_rule_results": self.domain_rule_results,
            "anomalies": self.anomalies,
            "recommendations": self.recommendations,
            "refinement_iterations": self.refinement_iterations,
            "validated_at": self.validated_at,
            "validation_time_ms": self.validation_time_ms,
        }


# ============================================================
# T5: 行动选择 — 数据模型
# ============================================================


class ActionType(str, Enum):
    """行动类型 (OLIVIA 四级行动策略).

    DIRECT_ANSWER:   直接回答 — 验证通过，直接输出结果
    TOOL_ENHANCED:   工具增强 — 需要调用外部工具补充信息
    NEGOTIATE:       协商确认 — 验证有警告，需要与用户确认
    HUMAN_CONFIRM:   人工确认 — 验证失败，必须转人工
    """

    DIRECT_ANSWER = "direct_answer"
    TOOL_ENHANCED = "tool_enhanced"
    NEGOTIATE = "negotiate"
    HUMAN_CONFIRM = "human_confirm"


class ActionRecord(BaseModel):
    """行动记录 (ActionSelector 产出).

    根据 ValidationReport 选择最优行动策略，记录决策依据。
    """

    record_id: str = Field(default_factory=lambda: f"act-{uuid.uuid4().hex[:12]}")
    plan_id: str = Field(default="", description="关联决策计划 ID")
    action_type: ActionType = Field(default=ActionType.DIRECT_ANSWER, description="选择的行动类型")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="行动选择置信度")

    # 决策依据
    validation_score: float = Field(default=0.0, description="验证分数")
    execution_confidence: float = Field(default=0.0, description="执行结果置信度")
    selection_reason: str = Field(default="", description="选择理由")

    # 行动参数
    response_payload: dict[str, Any] = Field(default_factory=dict, description="响应载荷")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="待调用的工具")
    clarification_questions: list[str] = Field(default_factory=list, description="澄清问题")

    # 元数据
    created_at: float = Field(default_factory=time.time, description="创建时间戳")

    model_config = {"frozen": False}


# ============================================================
# T6: 反馈与自适应学习 — 数据模型
# ============================================================


class FeedbackType(str, Enum):
    """反馈类型 — 收敛自 L2 统一枚举 (全系统单点, 见 l2/models.FeedbackType).

    保留本枚举别名以兼容旧调用方; 新代码应使用 l2.models.FeedbackType.
    """

    EXPLICIT_RATING = "explicit_rating"       # 用户显式评分
    IMPLICIT_SIGNAL = "implicit_signal"       # 隐式信号（停留时间、点击等）
    OUTCOME_FEEDBACK = "outcome_feedback"     # 结果反馈（是否正确）
    CORRECTION = "correction"                 # 用户纠正
    SKIP = "skip"                             # 跳过/忽略

    def to_unified(self) -> str:
        """映射到统一反馈类型 (l2.models.FeedbackType)."""
        mapping = {
            self.EXPLICIT_RATING: "explicit_rating",
            self.IMPLICIT_SIGNAL: "implicit_result",
            self.OUTCOME_FEEDBACK: "agent_outcome",
            self.CORRECTION: "human_feedback",
            self.SKIP: "skip",
        }
        return mapping.get(self, "implicit_result")


class FeedbackSignal(BaseModel):
    """反馈信号 (FeedbackAggregator 产出).

    闭环反馈驱动自适应学习，支持:
    - 显式评分反馈
    - 隐式行为信号
    - 结果正确性反馈
    - 用户纠正
    """

    signal_id: str = Field(default_factory=lambda: f"fb-{uuid.uuid4().hex[:12]}")
    plan_id: str = Field(default="", description="关联决策计划 ID")
    feedback_type: FeedbackType = Field(default=FeedbackType.IMPLICIT_SIGNAL, description="反馈类型")
    rating: float = Field(default=0.0, ge=-1.0, le=1.0, description="评分 (-1~1)")
    comment: str = Field(default="", description="用户评论")

    # 信号详情
    intent_type: str = Field(default="", description="意图类型")
    action_type: str = Field(default="", description="行动类型")
    execution_confidence: float = Field(default=0.0, description="执行置信度")
    validation_score: float = Field(default=0.0, description="验证分数")

    # 纠正信息
    correction: dict[str, Any] = Field(default_factory=dict, description="纠正内容")

    # 元数据
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    source: str = Field(default="system", description="反馈来源")

    model_config = {"frozen": False}


class FeedbackSummary(BaseModel):
    """反馈汇总 (自适应学习输入).

    聚合一段时间内的反馈信号，生成策略调整建议。
    """

    summary_id: str = Field(default_factory=lambda: f"fs-{uuid.uuid4().hex[:12]}")
    period_start: float = Field(default=0.0, description="统计起始时间")
    period_end: float = Field(default=0.0, description="统计结束时间")
    total_signals: int = Field(default=0, description="信号总数")
    avg_rating: float = Field(default=0.0, description="平均评分")

    # 分维度统计
    by_intent: dict[str, dict[str, float]] = Field(default_factory=dict, description="按意图统计")
    by_action: dict[str, dict[str, float]] = Field(default_factory=dict, description="按行动统计")

    # 策略调整建议
    adjustments: list[dict[str, Any]] = Field(default_factory=list, description="策略调整建议")

    # 元数据
    generated_at: float = Field(default_factory=time.time, description="生成时间戳")

    model_config = {"frozen": False}


# ============================================================
# T5+: 输出合成 — 数据模型 (OutputSynthesizer)
# ============================================================


class OutputFormat(str, Enum):
    """输出格式枚举 (借鉴 OLIVIA 多模态输出 + TDP 上下文隔离).

    CONCISE:       简洁回答 — 直接给结论，附关键证据
    DETAILED:      详细回答 — 完整推理链 + 多角度证据
    STRUCTURED:    结构化回答 — 表格/列表形式组织信息
    EXPLANATORY:   解释性回答 — 教学式展开，适合概念类查询
    COMPARATIVE:   比较型回答 — 多实体对比分析
    SUMMARIZED:    摘要型回答 — 关键信息提炼
    """

    CONCISE = "concise"
    DETAILED = "detailed"
    STRUCTURED = "structured"
    EXPLANATORY = "explanatory"
    COMPARATIVE = "comparative"
    SUMMARIZED = "summarized"


class SafetyLevel(str, Enum):
    """安全等级枚举 (借鉴 SafetyConstraintLayer).

    SAFE:        安全 — 可直接输出
    CAUTION:     谨慎 — 附带不确定性提示
    RESTRICTED:  受限 — 需附加免责声明
    BLOCKED:     阻断 — 不宜输出，需人工审核
    """

    SAFE = "safe"
    CAUTION = "caution"
    RESTRICTED = "restricted"
    BLOCKED = "blocked"


class EvidenceItem(BaseModel):
    """结构化证据项.

    用于组织输出中的证据，支持来源追踪和置信度标注。
    """

    evidence_id: str = Field(default="", description="证据 ID")
    content: str = Field(default="", description="证据内容")
    source: str = Field(default="", description="证据来源")
    source_type: str = Field(default="", description="来源类型 (chunk/triple/web/...)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="证据置信度")
    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="与查询的相关性")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据")

    model_config = {"frozen": False}


class OutputRecord(BaseModel):
    """输出记录 (OutputSynthesizer 产出).

    将 ActionRecord 转化为最终可交付给用户的结构化输出，
    包含格式化内容、校准置信度、安全评估和证据组织。

    融合世界先进方案:
    - Platt Scaling: 置信度校准
    - SafetyConstraintLayer: 安全感知输出
    - OLIVIA: 上下文感知格式选择
    - TDP: 输出层的上下文隔离
    """

    output_id: str = Field(default_factory=lambda: f"out-{uuid.uuid4().hex[:12]}")
    plan_id: str = Field(default="", description="关联决策计划 ID")
    action_record_id: str = Field(default="", description="关联行动记录 ID")

    # 格式与内容
    output_format: OutputFormat = Field(
        default=OutputFormat.CONCISE, description="输出格式"
    )
    content: str = Field(default="", description="主输出内容")
    summary: str = Field(default="", description="一句话摘要")
    structured_data: dict[str, Any] = Field(
        default_factory=dict, description="结构化数据 (表格/列表等)"
    )

    # 置信度校准 (Platt Scaling)
    raw_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="原始置信度")
    calibrated_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="校准后置信度"
    )
    calibration_params: dict[str, Any] = Field(
        default_factory=dict, description="校准参数"
    )

    # 安全评估
    safety_level: SafetyLevel = Field(
        default=SafetyLevel.SAFE, description="安全等级"
    )
    safety_warnings: list[str] = Field(
        default_factory=list, description="安全警告"
    )
    safety_disclaimer: str = Field(default="", description="安全免责声明")

    # 证据组织
    evidence_items: list[EvidenceItem] = Field(
        default_factory=list, description="结构化证据列表"
    )
    reasoning_summary: str = Field(default="", description="推理链摘要")

    # 元数据
    action_type: str = Field(default="", description="行动类型")
    intent_type: str = Field(default="", description="意图类型")
    created_at: float = Field(default_factory=time.time, description="创建时间戳")

    model_config = {"frozen": False}

    @property
    def is_safe_to_output(self) -> bool:
        """是否可安全输出."""
        return self.safety_level in (SafetyLevel.SAFE, SafetyLevel.CAUTION)

    @property
    def needs_disclaimer(self) -> bool:
        """是否需要免责声明."""
        return self.safety_level in (
            SafetyLevel.CAUTION,
            SafetyLevel.RESTRICTED,
            SafetyLevel.BLOCKED,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "output_id": self.output_id,
            "plan_id": self.plan_id,
            "output_format": self.output_format.value,
            "content": self.content,
            "summary": self.summary,
            "structured_data": self.structured_data,
            "raw_confidence": round(self.raw_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "safety_level": self.safety_level.value,
            "safety_warnings": self.safety_warnings,
            "safety_disclaimer": self.safety_disclaimer,
            "evidence_count": len(self.evidence_items),
            "reasoning_summary": self.reasoning_summary,
            "action_type": self.action_type,
            "intent_type": self.intent_type,
            "is_safe_to_output": self.is_safe_to_output,
            "needs_disclaimer": self.needs_disclaimer,
        }


class SafetyConstraint(BaseModel):
    """安全约束定义.

    定义输出安全检查的约束规则，用于 SafetyConstraintLayer。
    """

    constraint_id: str = Field(default_factory=lambda: f"sc-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="", description="约束名称")
    description: str = Field(default="", description="约束描述")
    pattern: str = Field(default="", description="匹配模式 (正则表达式)")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="触发阈值")
    action: SafetyLevel = Field(default=SafetyLevel.CAUTION, description="触发后的安全等级")
    message: str = Field(default="", description="触发后的警告消息")

    model_config = {"frozen": False}


class ConfidenceCalibrator(BaseModel):
    """Platt Scaling 置信度校准器配置.

    使用逻辑回归将原始置信度映射到校准置信度:
        p_calibrated = sigmoid(a * p_raw + b)

    其中 a, b 为可学习参数，通过历史反馈数据拟合。
    """

    scale: float = Field(default=1.0, ge=0.1, le=10.0, description="缩放参数 a")
    bias: float = Field(default=0.0, ge=-5.0, le=5.0, description="偏置参数 b")
    min_confidence: float = Field(default=0.01, ge=0.0, le=0.5, description="最小置信度下限")
    max_confidence: float = Field(default=0.99, ge=0.5, le=1.0, description="最大置信度上限")
    sample_count: int = Field(default=0, ge=0, description="校准样本数")

    model_config = {"frozen": False}

    def calibrate(self, raw_confidence: float) -> float:
        """执行 Platt Scaling 校准.

        Args:
            raw_confidence: 原始置信度 (0~1)

        Returns:
            校准后置信度 (0~1)
        """
        import math

        z = self.scale * raw_confidence + self.bias
        calibrated = 1.0 / (1.0 + math.exp(-z))
        return max(self.min_confidence, min(self.max_confidence, calibrated))
