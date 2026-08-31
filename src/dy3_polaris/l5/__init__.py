"""L5 Agent Runtime — Agent 定义、注册、持久化内核、编排引擎、通信、产物管理与反思引擎.

融合世界先进方案的 Agent 运行时体系:
- LangGraph: 有状态节点 + 条件边 + 检查点
- OpenAI Agents SDK: Agent Card + Handoff 机制
- Google ADK: DAG 任务分解 + Agent 注册
- CrewAI: 角色化 Agent 定义
- AutoGen: 消息传递 + Agent 注册表
- Claude Science: Persistent Kernels + Session Fork
- Jupyter: Kernel 进程模型 + 变量空间隔离
- Temporal: Activity 状态机 + 重试恢复

核心组件:
1. AgentDefinition — Agent 注册表定义模型
2. AgentRegistry — Agent 注册中心
3. PromptVersionManager — Prompt 版本管理 (A/B 测试 + 回滚)
4. AgentFactory — 六步实例化流水线
5. AgentInstance — 运行时实例与生命周期管理
6. DependencyResolver — 依赖图解析与拓扑排序
7. AgentDiscoveryService — 多维度 Agent 发现与就绪检查
8. AgentCompatibilityChecker — Agent 兼容性验证
9. HealthMonitor — 实例健康监控与心跳检测
10. PersistentKernel — 持久化执行内核（六态生命周期 + 变量保留）
11. KernelManager — 内核生命周期管理（超时 + 恢复）
12. CheckpointStore — 检查点存储抽象
13. SessionForkManager — 会话分叉管理（Fork/合并/清理）
14. StatePersistence — 状态持久化协调器（自动 checkpoint）
15. SessionManager — 会话生命周期管理器 (创建/激活/暂停/关闭 + Fork)
16. SessionContext — 三层上下文 (State + Events + Memory)
17. EventLog — 不可变事件溯源日志 (append-only + replay)
18. ForkCheckpoint — 四类状态快照 (kernel/working_session/outputs/broadcast)
19. ForkEvaluator — Fork 效果评估与最优路径选择
20. SessionCompactor — 上下文压缩 + continue-as-new
21. OrchestrationTask — 编排任务节点 (含依赖/超时/状态)
22. OrchestrationPlan — DAG 执行计划 (拓扑排序 + 并行层 + 循环检测)
23. OrchestrationResult — 执行结果 (含溯源)
24. PipelineExecutor — 顺序流水线编排 (ADK SequentialAgent + LangGraph superstep)
25. DebateExecutor — 辩论交叉验证编排 (辩论弧 + 收敛 + 代币预算)
26. VotingExecutor — 投票共识编排 (并行 fan-out + 声誉加权聚合)
27. OrchestrationEngine — 顶层编排引擎 (重试 + 超时 + 溯源)
28. Message — 不可变消息载体 (统一消息格式, frozen dataclass)
29. MessageBus — 消息总线 (Pub/Sub 引擎 + 频道管理 + 消息历史)
30. ChannelSubscription — 频道订阅管理 (活跃/取消)
31. StatePropagator — 状态传播器 (共享状态 + 5 种 Reducer 聚合)
32. StateUpdate — 状态更新事件 (含 Reducer 类型 + 作用域 + 溯源)
33. HandoffContext — 移交上下文 (对话历史 + 状态快照)
34. HandoffManager — Agent 控制权移交 (OpenAI Handoff 单向移交)
35. MessageRouter — 消息路由 (Fork 前缀隔离 + 优先级批量路由)
36. Artifact — 不可变产物载体 (元数据 + payload + 溯源链 + 学习上下文)
37. ArtifactType — 产物类型枚举 (8 种: text/chart/graph/molecule/table/formula/provenance/interactive)
38. ArtifactState — 产物生命周期状态机 (5 阶段: created → rendered → reviewed → edited → archived)
39. ArtifactVersion — 版本记录 (版本号 + 内容哈希 + CC1 状态 + 编辑操作)
40. ArtifactStore — 抽象存储接口 (save/load/list/delete/versions)
41. InMemoryArtifactStore — 内存存储实现 (线程安全 + 版本管理)
42. ArtifactManager — 产物管理器 (创建/更新/版本/搜索/归档/溯源)
43. ArtifactEdit — 编辑操作记录 (L5 Artifact-Edit Channel + 编辑意图/状态/审核)
44. ArtifactProvenance — 产物溯源记录 (Claude Science 五维度溯源 + L5 Provenance Ledger)
45. Verdict — 审核裁决枚举 (approved/revise/rejected)
46. ReflectionDimension — 反思检查维度 (4 维度: 事实一致性/数值准确性/引用完整性/教学适配性)
47. DimensionScore — 单维度评分 (分数 + 权重 + 理由)
48. ReviewRecord — 审核记录 (多维度评分 + 加权总分 + 裁决 + 反馈)
49. ReflectionResult — 反思结果 (审核历史 + 最终裁决 + 改进轨迹)
50. QualityGate — 质量门控 (阈值 + 硬下限 + 修订上限 + 错误分类)
51. GateResult / GateAction — 门控结果与动作
52. CC1Reviewer — CC1 Actor-Critic 审核器 (深度评审 + 多维度评分)
53. AdjudicationExecutor — 裁决执行器 (Saga 补偿 + 逆序执行)
54. ReputationLedger — 声誉账本 (动态信任分 + EMA 更新 + 阈值推荐)
55. ReflectionEngine — 反思引擎 (单 Agent 反思 + 跨 Agent 复盘 + 自纠循环)
56. CollaborationReview — 跨 Agent 协作复盘记录
57. QualityReport — 全链路质量报告
"""

from .agent_definition import (
    AgentDefinition,
    AgentFactory,
    AgentInstance,
    AgentInstanceState,
    AgentAlreadyExistsError,
    AgentNotFoundError,
    AgentRegistry,
    AgentRegistryError,
    BroadcastChannel,
    BroadcastMode,
    DecisionAuthority,
    FactoryError,
    FactoryStep,
    KernelBinding,
    KernelHandle,
    MemoryConfig,
    PromptReference,
    PromptVersion,
    PromptVersionError,
    PromptVersionManager,
    ReputationConfig,
    SelfEvolutionConfig,
    WorkingSession,
)
from .agent_discovery import (
    AgentCompatibilityChecker,
    AgentDiscoveryService,
    AgentNotReadyError,
    DependencyCycleError,
    DependencyEdge,
    DependencyMissingError,
    DependencyResolver,
    HealthMonitor,
    HealthRecord,
    HealthStatus,
)
from .kernel_persistence import (
    CheckpointStore,
    ForkConfig,
    ForkRecord,
    ForkStatus,
    KernelError,
    KernelManager,
    KernelState,
    MaxForkConcurrencyError,
    MaxForkDepthError,
    MemoryCheckpointStore,
    PersistentKernel,
    RecoveryExceededError,
    SessionForkManager,
    StatePersistence,
)
from .session_manager import (
    EventLog,
    ForkCheckpoint,
    ForkEvaluationResult,
    ForkEvaluator,
    ForkMergeScope,
    SessionCompactor,
    SessionContext,
    SessionEvent,
    SessionManager,
    SessionNotFoundError,
    SessionRecord,
    SessionState,
    SessionStateError,
    SessionTier,
)
from .orchestration_engine import (
    DebateExecutor,
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
    VoteRecord,
    VotingExecutor,
    VotingResult,
)
from .communication import (
    ChannelSubscription,
    CommunicationError,
    HandoffContext,
    HandoffManager,
    HandoffState,
    Message,
    MessageBus,
    MessagePriority,
    MessageRouter,
    ReducerType,
    StatePropagator,
    StateScope,
    StateUpdate,
)
from .artifact_manager import (
    Artifact,
    ArtifactEdit,
    ArtifactEditState,
    ArtifactError,
    ArtifactManager,
    ArtifactNotFoundError,
    ArtifactProvenance,
    ArtifactState,
    ArtifactStore,
    ArtifactType,
    ArtifactVersion,
    InMemoryArtifactStore,
)
from .interaction_recorder import (
    InteractionChain,
    InteractionPhase,
    InteractionRecord,
    InteractionRecorder,
    InteractionType,
    get_recorder,
    set_recorder,
)
from .reflection_quality import (
    AdjudicationExecutor,
    AdjudicationResult,
    CC1Reviewer,
    CollaborationReview,
    CollaborationTrigger,
    DimensionScore,
    ExecutionLogEntry,
    GateAction,
    GateResult,
    QualityGate,
    QualityReport,
    QualityTrendAnalyzer,
    ReflectionDimension,
    ReflectionEngine,
    ReflectionError,
    ReflectionResult,
    ReflectionTrigger,
    ReputationLedger,
    ReviewRecord,
    TargetedSelfCorrector,
    Verdict,
)

__all__ = [
    # 数据模型
    "AgentDefinition",
    "BroadcastChannel",
    "BroadcastMode",
    "DecisionAuthority",
    "KernelBinding",
    "KernelHandle",
    "MemoryConfig",
    "PromptReference",
    "PromptVersion",
    "ReputationConfig",
    "SelfEvolutionConfig",
    "WorkingSession",
    # 注册中心
    "AgentRegistry",
    "AgentRegistryError",
    "AgentNotFoundError",
    "AgentAlreadyExistsError",
    # 版本管理
    "PromptVersionManager",
    "PromptVersionError",
    # 工厂与实例
    "AgentFactory",
    "AgentInstance",
    "AgentInstanceState",
    "FactoryStep",
    "FactoryError",
    # 发现与健康监控
    "DependencyResolver",
    "DependencyEdge",
    "DependencyCycleError",
    "DependencyMissingError",
    "AgentDiscoveryService",
    "AgentCompatibilityChecker",
    "AgentNotReadyError",
    "HealthMonitor",
    "HealthRecord",
    "HealthStatus",
    # 持久化内核
    "PersistentKernel",
    "KernelState",
    "KernelError",
    "RecoveryExceededError",
    "KernelManager",
    "CheckpointStore",
    "MemoryCheckpointStore",
    "SessionForkManager",
    "ForkRecord",
    "ForkConfig",
    "ForkStatus",
    "MaxForkDepthError",
    "MaxForkConcurrencyError",
    "StatePersistence",
    # 会话管理
    "SessionState",
    "SessionTier",
    "SessionEvent",
    "EventLog",
    "SessionContext",
    "SessionRecord",
    "SessionManager",
    "SessionStateError",
    "SessionNotFoundError",
    # Fork 管理 (增强)
    "ForkCheckpoint",
    "ForkMergeScope",
    "ForkEvaluator",
    "ForkEvaluationResult",
    # 上下文压缩
    "SessionCompactor",
    # 编排引擎
    "OrchestrationState",
    "OrchestrationParadigm",
    "OrchestrationTask",
    "OrchestrationPlan",
    "OrchestrationResult",
    "OrchestrationError",
    "OrchestrationTimeoutError",
    "ParadigmExecutor",
    "PipelineExecutor",
    "DebateExecutor",
    "DebateState",
    "DebateMessage",
    "DebateRound",
    "VotingExecutor",
    "VoteRecord",
    "VotingResult",
    "OrchestrationEngine",
    # 通信与状态传递
    "Message",
    "MessageBus",
    "MessagePriority",
    "MessageRouter",
    "ChannelSubscription",
    "CommunicationError",
    "StatePropagator",
    "StateUpdate",
    "StateScope",
    "ReducerType",
    "HandoffContext",
    "HandoffManager",
    "HandoffState",
    # 交互记录
    "InteractionPhase",
    "InteractionType",
    "InteractionRecord",
    "InteractionChain",
    "InteractionRecorder",
    "get_recorder",
    "set_recorder",
    # 产物管理
    "Artifact",
    "ArtifactType",
    "ArtifactState",
    "ArtifactEditState",
    "ArtifactVersion",
    "ArtifactStore",
    "InMemoryArtifactStore",
    "ArtifactManager",
    "ArtifactEdit",
    "ArtifactProvenance",
    "ArtifactError",
    "ArtifactNotFoundError",
    # 反思与质量控制
    "Verdict",
    "ReflectionDimension",
    "ReflectionTrigger",
    "CollaborationTrigger",
    "DimensionScore",
    "ReviewRecord",
    "ReflectionResult",
    "GateAction",
    "GateResult",
    "QualityGate",
    "CC1Reviewer",
    "AdjudicationExecutor",
    "AdjudicationResult",
    "ReputationLedger",
    "ReflectionEngine",
    "CollaborationReview",
    "QualityReport",
    "QualityTrendAnalyzer",
    "TargetedSelfCorrector",
    "ExecutionLogEntry",
    "ReflectionError",
]
