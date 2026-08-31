"""L4 决策引擎层 — 推理与决策执行编排.

融合世界先进方案的决策引擎设计:
- LangGraph: 有状态图 + 条件边 + 检查点
- TDP 框架 (2026): Supervisor-Planner-Executor 三层 + 上下文隔离
- PRISM MHCV: 多维度异构协同验证
- OLIVIA (2026): 上下文线性赌博机 + UCB 行动选择
- GraphRAG: 多向量表示 + 跨文档关系聚合
- SubQRAG (2025): 子问题驱动动态图 RAG

增强版验证与策略评估:
- UniCR (2025): 不确定性量化网关 — 多源信号融合 + 分层验证
- RAGAS (2024): Faithfulness/Relevancy/Precision/Recall 评估
- CISC (2025): 置信度感知自洽性 — 加权多数投票
- VeriCoT (2025): 神经符号 CoT 验证 — 逻辑一致性检查
- VeReaFine (2025): 迭代验证-推理-精炼 — 缺失证据识别
- VRR-Stop (2026): 验证-修复循环的鲁棒停止框架
- PRISM (2026): 增益分解理论 — 探索/信息/聚合三维策略评估
- HydraRAG (2025): 三因子评分 — 多源交叉验证

本层职责:
1. DecisionPlanner — 将 RoutedResult 转化为可执行的 DecisionPlan
2. TaskExecutor — 按 DecisionPlan 调度已有推理/检索模块
3. ValidationOrchestrator — 增强版多维度验证编排 (UQ 驱动分层验证)
4. ActionSelector — 根据验证结果选择行动策略
5. FeedbackAggregator — 闭环反馈驱动自适应学习
6. DecisionEngine — 顶层编排器，串联 T1~T6

增强模块:
7. UQGate — 不确定性量化网关
8. FaithfulnessChecker — RAGAS 忠实度评估器
9. SelfConsistencyChecker — 自洽性检查器
10. StrategyEvaluator — 策略评估器
11. VRLoopController — 验证-精炼闭环控制器
12. DomainRuleEngine — 领域适配验证规则引擎
"""

from .ab_test_framework import (
    ABTestExperiment,
    ABTestFramework,
    ExperimentStatus,
    VariantStats,
)
from .adaptive_orchestrator import AdaptiveLearningOrchestrator
from .action_selector import (
    ActionSelector,
    EnsembleActionSelector,
    LinUCBSelector,
    RuleBasedSelector,
    ThompsonSamplingSelector,
    UCBActionSelector,
)
from .decision_engine import (
    DecisionEngine,
    DecisionEngineConfig,
)
from .decision_planner import (
    DecisionPlanner,
    PlanTemplate,
)
from .domain_rule_engine import DomainRuleEngine
from .faithfulness_checker import (
    ClaimExtractor,
    FaithfulnessChecker,
    SelfConsistencyChecker,
)
from .cold_start_manager import (
    ColdStartManager,
    ColdStartPhase,
)
from .concept_drift_detector import (
    ConceptDriftDetector,
    DriftResult,
    DriftSeverity,
    DriftType,
)
from .feedback_aggregator import FeedbackAggregator
from .output_synthesizer import (
    OutputSynthesizer,
    SafetyConstraintLayer,
)
from .models import (
    ActionRecord,
    ActionType,
    ConfidenceCalibrator,
    DecisionPlan,
    EvidenceItem,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    FallbackPlan,
    FeedbackSignal,
    FeedbackSummary,
    FeedbackType,
    OutputFormat,
    OutputRecord,
    ReasoningMode,
    ResourceBudget,
    RetrievalStrategy,
    SafetyConstraint,
    SafetyLevel,
    SubTask,
    TaskResult,
    TaskType,
    ValidationReport,
    ValidationSeverity,
    ValidationTier,
)
from .strategy_evaluator import StrategyEvaluator
from .task_executor import (
    BudgetExceededError,
    CyclicDependencyError,
    TaskExecutionError,
    TaskExecutor,
)
from .uq_gate import (
    UQAssessment,
    UQGate,
    UQSignal,
)
from .validation_orchestrator import ValidationOrchestrator
from .vr_loop import (
    RefinementFeedback,
    VRLoopController,
)

__all__ = [
    # 顶层编排器
    "DecisionEngine",
    "DecisionEngineConfig",
    # 计划生成
    "DecisionPlanner",
    "PlanTemplate",
    # 任务执行
    "TaskExecutor",
    "TaskExecutionError",
    "BudgetExceededError",
    "CyclicDependencyError",
    # 验证 (增强版)
    "ValidationOrchestrator",
    # UQ 网关
    "UQGate",
    "UQSignal",
    "UQAssessment",
    # Faithfulness 评估
    "FaithfulnessChecker",
    "SelfConsistencyChecker",
    "ClaimExtractor",
    # 策略评估
    "StrategyEvaluator",
    # V&R 闭环
    "VRLoopController",
    "RefinementFeedback",
    # 领域规则引擎
    "DomainRuleEngine",
    # 行动选择
    "ActionSelector",
    "UCBActionSelector",
    "ThompsonSamplingSelector",
    "LinUCBSelector",
    "EnsembleActionSelector",
    "RuleBasedSelector",
    # 反馈
    "FeedbackAggregator",
    # 自适应学习 (新增)
    "AdaptiveLearningOrchestrator",
    "ConceptDriftDetector",
    "DriftResult",
    "DriftType",
    "DriftSeverity",
    "ABTestFramework",
    "ABTestExperiment",
    "ExperimentStatus",
    "VariantStats",
    "ColdStartManager",
    "ColdStartPhase",
    # 输出合成
    "OutputSynthesizer",
    "SafetyConstraintLayer",
    # 数据模型
    "DecisionPlan",
    "ExecutionMode",
    "ExecutionResult",
    "ExecutionStatus",
    "FallbackPlan",
    "ResourceBudget",
    "SubTask",
    "TaskResult",
    "TaskType",
    "ReasoningMode",
    "RetrievalStrategy",
    "ValidationReport",
    "ValidationSeverity",
    "ValidationTier",
    "ActionRecord",
    "ActionType",
    "FeedbackSignal",
    "FeedbackSummary",
    "FeedbackType",
    # 输出合成 (新增)
    "OutputFormat",
    "OutputRecord",
    "EvidenceItem",
    "SafetyLevel",
    "SafetyConstraint",
    "ConfidenceCalibrator",
]
