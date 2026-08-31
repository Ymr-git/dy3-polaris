"""L2 个性化层 — 学情画像、知识追踪、能力评估、记忆、会话、交互、技能树.

L2 层是 Dy3+ Polaris 多智能体系统的个性化引擎层，负责:
1. 学情画像构建 (Profile Builder)
2. 知识追踪 (BKT 贝叶斯知识追踪 + KG 传播 + 遗忘模型)
3. 能力评估 (IRT 项目反应理论 + CAT 自适应选题)
4. 学习者记忆 (工作/短期/长期三记忆体系)
5. 个性化会话管理 (生命周期 + 检查点)
6. 人机交互建模 (事件采集 + 更新管道)
7. 技能树映射 (BKT+IRT → 可视化技能树)

本包导出:
- 异常体系: L2Error 层级 (JSON-RPC -32300 范围)
- 数据模型: AnswerRecord / TracingState / IRTState / LearnerSnapshot / SessionRecord
- 存储层:   L2Store 抽象基类 + InMemoryL2Store 内存实现
- 缓存层:   L2Cache 分层 TTL + write-through backing store
- 知识追踪: BKTTracer / MasteryPropagator / ForgettingModel
- 能力评估: IRTEstimator / CATSelector
- 交互管道: AnswerEvent / QueryEvent / BehaviorEvent / EventCollector / UpdatePipeline
- 会话管理: SessionManager
- 记忆体系: MemoryChunk / WorkingMemory / ShortTermMemory / LongTermMemory
- 画像构建: ProfileBuilder / LevelEstimator / StyleInferrer / BloomSetter
- 技能树:   SkillNode / SkillEdge / SkillMapper

设计依据: 02-设计/L2-个性化引擎设计 (参考 L1 用户域基础设施分层模式).
融合方案: Corbett&Anderson BKT / Bayesian IRT EAP / FSRS-6 / VARK / Bloom / ZPD /
         Knewton 三引擎 / ALEKS 知识空间 / Khan Academy / Duolingo / Squirrel AI.
"""

from dy3_polaris.l2.exceptions import (
    IRTError,
    L2Error,
    MemoryError,
    ProfileNotFoundError,
    StoreError,
    TracingError,
)
from dy3_polaris.l2.models import (
    AnswerRecord,
    IRTState,
    LearnerSnapshot,
    SessionRecord,
    TracingState,
)
from dy3_polaris.l2.store import InMemoryL2Store, L2Store
from dy3_polaris.l2.cache import L2Cache

# 子模块导出
from dy3_polaris.l2.knowledge_tracer import (
    BKTTracer,
    ForgettingModel,
    MasteryPropagator,
)
from dy3_polaris.l2.ability_assessor import (
    CATSelector,
    IRTEstimator,
    ZPDCalculator,
    ZPDResult,
)
from dy3_polaris.l2.interaction import (
    AnswerEvent,
    BehaviorEvent,
    EventCollector,
    QueryEvent,
    UpdatePipeline,
)
from dy3_polaris.l2.session import SessionManager
from dy3_polaris.l2.memory import (
    LongTermMemory,
    MemoryChunk,
    ShortTermMemory,
    WorkingMemory,
)
from dy3_polaris.l2.profile_builder import (
    BloomSetter,
    ColdStartManager,
    ConceptDriftDetector,
    DriftAwareRetrainer,
    LearnerColdStartManager,
    LearnerDriftDetector,
    LearnerLifecycleManager,
    LearnerLifecycleState,
    LevelEstimator,
    ProfileBuilder,
    StyleInferrer,
)
from dy3_polaris.l2.skillbook import (
    SkillEdge,
    SkillMapper,
    SkillNode,
)

__all__ = [
    # 异常体系
    "L2Error",
    "ProfileNotFoundError",
    "TracingError",
    "IRTError",
    "MemoryError",
    "StoreError",
    # 数据模型
    "AnswerRecord",
    "TracingState",
    "IRTState",
    "LearnerSnapshot",
    "SessionRecord",
    # 存储层
    "L2Store",
    "InMemoryL2Store",
    # 缓存层
    "L2Cache",
    # 知识追踪
    "BKTTracer",
    "MasteryPropagator",
    "ForgettingModel",
    # 能力评估
    "IRTEstimator",
    "CATSelector",
    "ZPDCalculator",
    "ZPDResult",
    # 交互管道
    "AnswerEvent",
    "QueryEvent",
    "BehaviorEvent",
    "EventCollector",
    "UpdatePipeline",
    # 会话管理
    "SessionManager",
    # 记忆体系
    "MemoryChunk",
    "WorkingMemory",
    "ShortTermMemory",
    "LongTermMemory",
    # 画像构建
    "ProfileBuilder",
    "LevelEstimator",
    "StyleInferrer",
    "BloomSetter",
    "LearnerColdStartManager",
    "ColdStartManager",  # 向后兼容别名 (已弃用)
    "LearnerDriftDetector",
    "ConceptDriftDetector",  # 向后兼容别名 (已弃用)
    "LearnerLifecycleManager",
    "LearnerLifecycleState",
    "DriftAwareRetrainer",
    # 技能树
    "SkillNode",
    "SkillEdge",
    "SkillMapper",
]
