"""L0 治理层 — CC3 溯源捕获子包.

提供教育多 Agent 系统的全链路溯源捕获能力，融合八大世界级方案：
- W3C PROV: Entity-Activity-Agent 溯源三元组映射
- C2PA: 加密签名断言 (tamper-evident)
- OpenTelemetry GenAI: 标准化 trace_id/span_id 跨层传递
- RFC 6962 Certificate Transparency: Merkle 树 append-only 日志
- OpenLineage: Dataset/Job/Run 血缘模型
- MLflow2PROV: 实验级溯源图
- Langfuse: LLM trace 树可视化
- JSON Patch RFC 6902: 演化维度版本 diff

双核心架构:
- KPA (Knowledge Provenance Annotation): 七维知识溯源标注
  1. 来源 (Source) — 原始数据来源 (NIST/DOI/实验条件)
  2. 生成 (Generation) — 生成者及生成环境
  3. 校验 (Validation) — CC1 四层评审结果
  4. 决策 (Decision) — 系统决策路径
  5. 演化 (Evolution) — 版本历史与变更
  6. 传播 (Propagation) — 使用轨迹与引用
  7. 关联 (Relation) — 语义关联网络

- DL (Debate Log): 辩论日志系统
  - Pre-Debate / During-Debate / Post-Debate 三级记录
  - Summary (永久) / Full (90天) / Debug (30天) 三级存储
  - 分歧度计算与收敛检测

增强组件 (CC-3 溯源捕获):
- KPA 标注引擎: KPAEngine — 七维标注+15条Dy3+领域规则+C2PA签名+W3C PROV映射
- 辩论日志引擎: DebateLogger — 三级日志+分歧度计算+收敛检测+完整性校验
- 溯源链构建器: ProvenanceChainBuilder — SHA-256哈希链+Merkle树压缩+包含证明
- L0 Ledger 集成: LedgerIntegration — 五类事件append-only存储+时间范围查询
- 查询引擎: QueryEngine — 跨数据源联合查询+trace_id全链路回溯
- CC1/CC2 集成器: CCIntegration — 评审/审批自动映射+溯源缺失升级建议
- KPI 指标引擎: KPAMetricsEngine — 四类12项KPI+仪表盘导出
- 可视化适配器: ProvenanceVisualizer — Cytoscape/D3/Mermaid/ECharts四格式
- REST API: CC3APIRouter — 9大端点组全覆盖
"""

# ============================================================
# 枚举与数据模型
# ============================================================
from .models import (
    # 枚举
    TargetType,
    SourceTier,
    ChangeType,
    ValidationVerdict,
    DebateRole,
    LogVerbosity,
    CounterType,
    ConvergenceStatus,
    EventType,
    CrossLayerDirection,
    KPACategory,
    # KPA 七维标注模型
    SourceDimension,
    GenerationDimension,
    ValidationDimension,
    DecisionDimension,
    EvolutionDimension,
    PropagationDimension,
    RelationDimension,
    KPAAnnotation,
    # 辩论日志模型
    DebateArgument,
    DebateCounterargument,
    DebateRound,
    AdjudicatorVerdict,
    DebateResourceUsage,
    DebateOutcome,
    PreDebateRecord,
    DebateLog,
    # 溯源链与 Ledger 模型
    ProvenanceChainNode,
    LedgerEvent,
    AuditVerificationResult,
)

# ============================================================
# 异常定义
# ============================================================
from .exceptions import (
    CC3Error,
    HashMismatchError,
    AnnotationNotFoundError,
    DebateLogNotFoundError,
    SchemaValidationError,
    ChainBrokenError,
    StorageUnavailableError,
)

# ============================================================
# KPA 七维标注引擎
# ============================================================
from .kpa_engine import (
    KPAEngine,
    Dy3AnnotationRule,
    DY3_ANNOTATION_RULES,
    # 来源维度规则
    RS01_DOIFormatRule,
    RS02_SourceTierRule,
    RS03_Dy3WavelengthRule,
    RS04_RetrievalTimestampRule,
    RS05_SecondarySourceRule,
    # 生成维度规则
    RG01_TraceIDRule,
    RG02_CodeHashRule,
    RG03_EnvironmentHashRule,
    # 校验维度规则
    RV01_CC1LinkageRule,
    RV02_FourLayerScoreRule,
    # 演化维度规则
    RE01_VersionChainRule,
    RE02_JSONPatchRule,
    # 关联维度规则
    RR01_PrerequisiteRule,
    RR02_Dy3DomainRelationRule,
    # 传播维度规则
    RP01_PropagationInitRule,
)

# ============================================================
# 辩论日志引擎
# ============================================================
from .debate_logger import (
    DebateLogger,
    DivergenceCalculator,
    PromptSanitizer,
)

# ============================================================
# 溯源链构建器
# ============================================================
from .provenance_chain_builder import (
    MerkleTree,
    ProvenanceChainBuilder,
)

# ============================================================
# L0 Ledger 集成
# ============================================================
from .ledger_integration import LedgerIntegration

# ============================================================
# 查询引擎
# ============================================================
from .query_engine import QueryEngine

# ============================================================
# CC1/CC2 跨切面集成
# ============================================================
from .cc_integration import CCIntegration

# ============================================================
# KPI 指标引擎
# ============================================================
from .metrics import (
    KPAMetricsEngine,
    MetricSample,
    MetricsSummary,
)

# ============================================================
# 可视化适配器
# ============================================================
from .visualizer_adapter import ProvenanceVisualizer

# ============================================================
# REST API 路由
# ============================================================
try:
    from .api import CC3APIRouter
except ImportError:  # pragma: no cover - api 模块可能尚未实现
    CC3APIRouter = None  # type: ignore[assignment,misc]

__all__ = [
    # ==================== 枚举 ====================
    "TargetType",
    "SourceTier",
    "ChangeType",
    "ValidationVerdict",
    "DebateRole",
    "LogVerbosity",
    "CounterType",
    "ConvergenceStatus",
    "EventType",
    "CrossLayerDirection",
    "KPACategory",
    # ==================== KPA 七维模型 ====================
    "SourceDimension",
    "GenerationDimension",
    "ValidationDimension",
    "DecisionDimension",
    "EvolutionDimension",
    "PropagationDimension",
    "RelationDimension",
    "KPAAnnotation",
    # ==================== 辩论日志模型 ====================
    "DebateArgument",
    "DebateCounterargument",
    "DebateRound",
    "AdjudicatorVerdict",
    "DebateResourceUsage",
    "DebateOutcome",
    "PreDebateRecord",
    "DebateLog",
    # ==================== 溯源链与 Ledger 模型 ====================
    "ProvenanceChainNode",
    "LedgerEvent",
    "AuditVerificationResult",
    # ==================== 异常 ====================
    "CC3Error",
    "HashMismatchError",
    "AnnotationNotFoundError",
    "DebateLogNotFoundError",
    "SchemaValidationError",
    "ChainBrokenError",
    "StorageUnavailableError",
    # ==================== KPA 引擎 ====================
    "KPAEngine",
    "Dy3AnnotationRule",
    "DY3_ANNOTATION_RULES",
    "RS01_DOIFormatRule",
    "RS02_SourceTierRule",
    "RS03_Dy3WavelengthRule",
    "RS04_RetrievalTimestampRule",
    "RS05_SecondarySourceRule",
    "RG01_TraceIDRule",
    "RG02_CodeHashRule",
    "RG03_EnvironmentHashRule",
    "RV01_CC1LinkageRule",
    "RV02_FourLayerScoreRule",
    "RE01_VersionChainRule",
    "RE02_JSONPatchRule",
    "RR01_PrerequisiteRule",
    "RR02_Dy3DomainRelationRule",
    "RP01_PropagationInitRule",
    # ==================== 辩论日志引擎 ====================
    "DebateLogger",
    "DivergenceCalculator",
    "PromptSanitizer",
    # ==================== 溯源链 ====================
    "MerkleTree",
    "ProvenanceChainBuilder",
    # ==================== L0 Ledger ====================
    "LedgerIntegration",
    # ==================== 查询引擎 ====================
    "QueryEngine",
    # ==================== CC1/CC2 集成 ====================
    "CCIntegration",
    # ==================== KPI 指标 ====================
    "KPAMetricsEngine",
    "MetricSample",
    "MetricsSummary",
    # ==================== 可视化 ====================
    "ProvenanceVisualizer",
    # ==================== REST API ====================
    "CC3APIRouter",
]
