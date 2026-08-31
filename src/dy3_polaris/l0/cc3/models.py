"""CC3 溯源捕获层 — 数据模型.

定义 CC3 双核心架构的全部数据模型:
- KPA (Knowledge Provenance Annotation): 七维知识溯源标注
- DL (Debate Log): 辩论日志系统

七维标注模型 (KPA):
1. 来源 (Source) — 原始数据来源 (NIST/DOI/实验条件)
2. 生成 (Generation) — 生成者及生成环境
3. 校验 (Validation) — CC1 四层评审结果
4. 决策 (Decision) — 系统决策路径
5. 演化 (Evolution) — 版本历史与变更
6. 传播 (Propagation) — 使用轨迹与引用
7. 关联 (Relation) — 语义关联网络

辩论日志模型 (DL):
- 元数据层 / 轮次层 / 收敛层 / 裁决层 / 资源层 / 结果层
- 三级日志: Summary (永久) / Full (90天→冷存储) / Debug (30天清理)

融合方案:
- W3C PROV: Entity-Activity-Agent 溯源三元组
- C2PA: 加密签名断言 (tamper-evident)
- OpenTelemetry GenAI: 标准化 span/trace ID
- RFC 6962 Certificate Transparency: Merkle 树 append-only 日志
- OpenLineage: Dataset/Job/Run 血缘模型
- MLflow2PROV: 实验级溯源图
- Langfuse: LLM trace 树可视化
- JSON Patch RFC 6902: 演化维度版本 diff
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ============================================================
# 枚举定义
# ============================================================


class TargetType(str, enum.Enum):
    """标注对象类型."""

    KNOWLEDGE_POINT = "kp"
    CONTENT = "content"
    DECISION = "decision"
    ARTIFACT = "artifact"
    DEBATE_OUTCOME = "debate_outcome"
    REVIEW_REPORT = "review_report"


class SourceTier(str, enum.Enum):
    """来源权威等级 (W3C PROV + OpenAlex 启发).

    TIER_1: 顶级期刊/标准 (NIST, DOI 顶级期刊)
    TIER_2: 权威教材/综述 (Springer, Wiley 教材)
    TIER_3: 预印本/会议 (arXiv, 会议论文)
    TIER_4: 内部文档/Agent生成 (内部知识库)
    TIER_5: 未验证来源 (网页, 个人笔记)
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    TIER_5 = "tier_5"


class ChangeType(str, enum.Enum):
    """演化变更类型."""

    CREATED = "created"
    CORRECTED = "corrected"
    ENHANCED = "enhanced"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


class ValidationVerdict(str, enum.Enum):
    """CC1 评审结论."""

    PASS = "pass"
    PASS_WITH_NOTES = "pass_with_notes"
    FAIL = "fail"


class DebateRole(str, enum.Enum):
    """辩论角色 (GAIA 协商协议启发)."""

    GENERATOR = "generator"
    REVIEWER = "reviewer"
    ADJUDICATOR = "adjudicator"
    OBSERVER = "observer"


class LogVerbosity(str, enum.Enum):
    """日志级别 (三级存储策略).

    SUMMARY: 摘要级 — L0 常规表, 永久保留
    FULL: 完整级 — L0 Archive 表, 90天后转冷存储
    DEBUG: 调试级 — Session Fork/对象存储, 30天后清理
    """

    SUMMARY = "summary"
    FULL = "full"
    DEBUG = "debug"


class CounterType(str, enum.Enum):
    """反驳类型."""

    DIRECT_REFUTATION = "direct_refutation"
    EVIDENCE_BASED = "evidence_based"
    METHODOLOGICAL = "methodological"
    SCOPE_LIMITATION = "scope_limitation"
    META_COUNTER = "meta_counter"


class ConvergenceStatus(str, enum.Enum):
    """收敛状态."""

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    FORCE_RESOLVED = "force_resolved"
    ABORTED = "aborted"


class EventType(str, enum.Enum):
    """L0 Provenance Ledger 事件类型 (五类)."""

    LEARNER_PROFILE = "learner_profile"
    KNOWLEDGE = "knowledge"
    DECISION = "decision"
    INTERACTION = "interaction"
    HUMAN_OVERRIDE = "human_override"


class CrossLayerDirection(str, enum.Enum):
    """跨层传递方向 (8 种)."""

    L2_TO_L3 = "l2_to_l3"
    L3_TO_L4 = "l3_to_l4"
    L4_TO_L5 = "l4_to_l5"
    L5_TO_L6 = "l5_to_l6"
    L6_TO_L7 = "l6_to_l7"
    L7_TO_L0 = "l7_to_l0"
    CC1_TO_CC3 = "cc1_to_cc3"
    CC2_TO_CC3 = "cc2_to_cc3"


class KPACategory(str, enum.Enum):
    """KPI 指标分类."""

    COVERAGE = "coverage"
    INTEGRITY = "integrity"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"


# ============================================================
# KPA 七维标注数据模型
# ============================================================


class SourceDimension(BaseModel):
    """来源维度 — 原始数据来源.

    融合方案:
    - W3C PROV: Entity 的 wasDerivedFrom 关系
    - OpenAlex/Crossref: 期刊权威性分级
    - DataCite: 持久标识符 (DOI)
    """

    primary_source: str = Field(
        default="",
        description="主要来源 (DOI/URL/NIST标准号/实验条件描述)",
    )
    source_type: str = Field(
        default="",
        description="来源类型 (journal/textbook/preprint/experiment/database/internal)",
    )
    trust_tier: SourceTier = Field(
        default=SourceTier.TIER_3,
        description="来源权威等级",
    )
    secondary_sources: list[str] = Field(
        default_factory=list,
        description="次要来源列表",
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="来源元数据 (作者/年份/页码/DOI等)",
    )
    retrieval_method: str = Field(
        default="",
        description="检索方法 (manual/api/citation_chain/mcp_tool)",
    )
    retrieval_timestamp: float = Field(
        default=0.0,
        description="检索时间戳",
    )

    def is_filled(self) -> bool:
        """检查来源维度是否已填充."""
        return bool(self.primary_source)

    def completeness(self) -> float:
        """来源维度完整度 (0.0-1.0)."""
        fields = [
            bool(self.primary_source),
            bool(self.source_type),
            self.trust_tier != SourceTier.TIER_3,
            len(self.secondary_sources) > 0,
            len(self.source_metadata) > 0,
            bool(self.retrieval_method),
            self.retrieval_timestamp > 0,
        ]
        return sum(fields) / len(fields)


class GenerationDimension(BaseModel):
    """生成维度 — 生成者及生成环境.

    融合方案:
    - OpenTelemetry GenAI: model/version/prompt_hash 标准化属性
    - MLflow2PROV: 代码版本 + 超参数追踪
    - Langfuse: trace 树结构
    """

    agent_id: str = Field(default="", description="生成 Agent ID")
    agent_version: str = Field(default="", description="Agent 版本")
    agent_role: str = Field(default="", description="Agent 角色")
    code_hash: str = Field(default="", description="代码哈希 (git commit SHA)")
    prompt_version: str = Field(default="", description="Prompt 模板版本")
    model_cfg: dict[str, Any] = Field(
        default_factory=dict,
        description="模型配置 (model_name/temperature/max_tokens等)",
    )
    generation_timestamp: float = Field(
        default_factory=time.time,
        description="生成时间戳",
    )
    generation_duration_ms: float = Field(
        default=0.0,
        description="生成耗时 (毫秒)",
    )
    trace_id: str = Field(
        default="",
        description="OpenTelemetry trace ID (跨层传递)",
    )
    span_id: str = Field(
        default="",
        description="OpenTelemetry span ID",
    )
    environment_hash: str = Field(
        default="",
        description="运行环境哈希 (Python版本/依赖版本)",
    )

    def is_filled(self) -> bool:
        """检查生成维度是否已填充."""
        return bool(self.agent_id)

    def completeness(self) -> float:
        """生成维度完整度 (0.0-1.0)."""
        fields = [
            bool(self.agent_id),
            bool(self.agent_version),
            bool(self.agent_role),
            bool(self.code_hash),
            bool(self.prompt_version),
            len(self.model_cfg) > 0,
            self.generation_timestamp > 0,
            self.generation_duration_ms > 0,
            bool(self.trace_id),
            bool(self.environment_hash),
        ]
        return sum(fields) / len(fields)


class ValidationDimension(BaseModel):
    """校验维度 — CC1 四层评审结果.

    融合方案:
    - CC1 反幻觉评审: 四层评分 (事实/逻辑/数值/溯源)
    - RAGAS: faithfulness/answer_relevancy
    - FActScore: 原子事实级校验
    """

    cc1_review_id: str = Field(
        default="",
        description="CC1 评审报告 ID (一对一关联)",
    )
    four_layer_scores: dict[str, float] = Field(
        default_factory=dict,
        description="四层评分: {factual, logical, numerical, provenance}",
    )
    verdict: ValidationVerdict = Field(
        default=ValidationVerdict.PASS,
        description="评审结论",
    )
    validation_issues: list[dict[str, Any]] = Field(
        default_factory=list,
        description="问题列表",
    )
    standard_value_check: dict[str, Any] = Field(
        default_factory=dict,
        description="标准值校验结果",
    )
    mcp_tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="MCP 工具调用记录",
    )
    self_correction_count: int = Field(
        default=0,
        description="自纠回路迭代次数",
    )
    validated_at: float = Field(
        default=0.0,
        description="校验时间戳",
    )

    def is_filled(self) -> bool:
        """检查校验维度是否已填充."""
        return bool(self.cc1_review_id)

    def completeness(self) -> float:
        """校验维度完整度 (0.0-1.0)."""
        fields = [
            bool(self.cc1_review_id),
            len(self.four_layer_scores) > 0,
            self.verdict is not None,
            len(self.validation_issues) >= 0,  # 空列表也算
            len(self.standard_value_check) > 0,
            len(self.mcp_tool_calls) > 0,
            self.self_correction_count >= 0,
            self.validated_at > 0,
        ]
        return sum(fields) / len(fields)


class DecisionDimension(BaseModel):
    """决策维度 — 系统决策路径.

    融合方案:
    - CC2 Plan-Approval: 审批/协同记录
    - GAIA: 协商协议决策
    - REACT: 风险评估驱动决策
    """

    meta_decider_result: str = Field(
        default="",
        description="Meta-Decider 决策结果",
    )
    paradigm_selected: str = Field(
        default="",
        description="选择的讲解范式",
    )
    adjudicator_verdict: str = Field(
        default="",
        description="Adjudicator 裁决结果",
    )
    cc2_approval_id: str = Field(
        default="",
        description="CC2 审批记录 ID",
    )
    cc2_approval_level: str = Field(
        default="",
        description="CC2 协同层级 (implicit/prompt/approval/intervention)",
    )
    debate_id: str = Field(
        default="",
        description="辩论 ID (如触发辩论)",
    )
    decision_path: list[str] = Field(
        default_factory=list,
        description="决策路径节点列表",
    )
    decision_timestamp: float = Field(
        default=0.0,
        description="决策时间戳",
    )

    def is_filled(self) -> bool:
        """检查决策维度是否已填充."""
        return bool(self.meta_decider_result) or bool(self.paradigm_selected)

    def completeness(self) -> float:
        """决策维度完整度 (0.0-1.0)."""
        fields = [
            bool(self.meta_decider_result),
            bool(self.paradigm_selected),
            bool(self.adjudicator_verdict),
            bool(self.cc2_approval_id),
            bool(self.cc2_approval_level),
            bool(self.debate_id),
            len(self.decision_path) > 0,
            self.decision_timestamp > 0,
        ]
        return sum(fields) / len(fields)


class EvolutionDimension(BaseModel):
    """演化维度 — 版本历史与变更.

    融合方案:
    - JSON Patch RFC 6902: 结构化版本 diff
    - Git: 版本链式管理
    - DataCite: 元数据版本管理
    """

    version: str = Field(default="1.0.0", description="当前版本号")
    version_chain: list[dict[str, Any]] = Field(
        default_factory=list,
        description="版本链: [{version, timestamp, change_type, actor}]",
    )
    change_type: ChangeType = Field(
        default=ChangeType.CREATED,
        description="当前变更类型",
    )
    diff_snapshot: list[dict[str, Any]] = Field(
        default_factory=list,
        description="版本差异 (JSON Patch RFC 6902 格式)",
    )
    evolution_timeline: list[dict[str, Any]] = Field(
        default_factory=list,
        description="演化时间线",
    )
    parent_version: str = Field(
        default="",
        description="父版本号",
    )

    def is_filled(self) -> bool:
        """检查演化维度是否已填充."""
        return len(self.version_chain) > 0 or self.change_type != ChangeType.CREATED

    def completeness(self) -> float:
        """演化维度完整度 (0.0-1.0)."""
        fields = [
            bool(self.version),
            len(self.version_chain) > 0,
            self.change_type is not None,
            len(self.diff_snapshot) > 0,
            len(self.evolution_timeline) > 0,
            bool(self.parent_version) or self.change_type == ChangeType.CREATED,
        ]
        return sum(fields) / len(fields)


class PropagationDimension(BaseModel):
    """传播维度 — 使用轨迹与引用.

    融合方案:
    - OpenLineage: Dataset/Job/Run 模型
    - Google Scholar: 引用计数
    - Langfuse: 使用追踪
    """

    session_references: list[str] = Field(
        default_factory=list,
        description="引用此知识的会话列表",
    )
    agent_usages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Agent 使用记录: [{agent_id, timestamp, context}]",
    )
    learner_consumptions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="学习者消费记录: [{learner_id, timestamp, interaction_type}]",
    )
    citation_count: int = Field(default=0, description="被引用次数")
    last_accessed_at: float = Field(
        default=0.0,
        description="最后访问时间戳",
    )
    propagation_graph: dict[str, Any] = Field(
        default_factory=dict,
        description="传播图 (节点+边)",
    )

    def is_filled(self) -> bool:
        """检查传播维度是否已填充."""
        return (
            len(self.session_references) > 0
            or len(self.agent_usages) > 0
            or self.citation_count > 0
        )

    def completeness(self) -> float:
        """传播维度完整度 (0.0-1.0)."""
        fields = [
            len(self.session_references) > 0,
            len(self.agent_usages) > 0,
            len(self.learner_consumptions) > 0,
            self.citation_count > 0,
            self.last_accessed_at > 0,
            len(self.propagation_graph) > 0,
        ]
        return sum(fields) / len(fields)


class RelationDimension(BaseModel):
    """关联维度 — 语义关联网络.

    融合方案:
    - Knowledge Graph: 实体关系图
    - Cytoscape.js: 网络可视化格式
    - Neo4j: 图数据库查询
    """

    prerequisites: list[str] = Field(
        default_factory=list,
        description="前置知识 ID 列表",
    )
    successors: list[str] = Field(
        default_factory=list,
        description="后继知识 ID 列表",
    )
    same_domain_relations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="同域关系: [{target_id, relation_type, strength}]",
    )
    cross_domain_relations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="跨域关系: [{target_id, source_domain, target_domain, strength}]",
    )
    relation_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="平均关联强度 (0.0-1.0)",
    )
    network_centrality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="网络中心度 (0.0-1.0)",
    )

    def is_filled(self) -> bool:
        """检查关联维度是否已填充."""
        return (
            len(self.prerequisites) > 0
            or len(self.successors) > 0
            or len(self.same_domain_relations) > 0
        )

    def completeness(self) -> float:
        """关联维度完整度 (0.0-1.0)."""
        fields = [
            len(self.prerequisites) > 0,
            len(self.successors) > 0,
            len(self.same_domain_relations) > 0,
            len(self.cross_domain_relations) > 0,
            self.relation_strength > 0,
            self.network_centrality > 0,
        ]
        return sum(fields) / len(fields)


# ============================================================
# KPA 标注主模型
# ============================================================


class KPAAnnotation(BaseModel):
    """KPA 知识溯源标注 — 七维标注主模型.

    为系统中每个知识实体构建立体化溯源档案。
    每个标注包含七个维度, 支持部分填充和渐进完善。

    Attributes:
        annotation_id: 标注唯一 ID
        target_type: 标注对象类型
        target_id: 标注对象 ID
        source: 来源维度
        generation: 生成维度
        validation: 校验维度
        decision: 决策维度
        evolution: 演化维度
        propagation: 传播维度
        relation: 关联维度
        created_at: 创建时间
        updated_at: 更新时间
        annotator_agent: 标注 Agent ID
        immutable_hash: 不可变哈希 (SHA-256)
    """

    annotation_id: str = Field(
        default_factory=lambda: f"kpa-{uuid.uuid4().hex[:12]}",
    )
    target_type: TargetType = Field(
        default=TargetType.KNOWLEDGE_POINT,
    )
    target_id: str = Field(default="", description="标注对象 ID")
    target_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="标注对象元数据",
    )

    # 七维标注
    source: SourceDimension = Field(default_factory=SourceDimension)
    generation: GenerationDimension = Field(default_factory=GenerationDimension)
    validation: ValidationDimension = Field(default_factory=ValidationDimension)
    decision: DecisionDimension = Field(default_factory=DecisionDimension)
    evolution: EvolutionDimension = Field(default_factory=EvolutionDimension)
    propagation: PropagationDimension = Field(default_factory=PropagationDimension)
    relation: RelationDimension = Field(default_factory=RelationDimension)

    # 元数据
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    annotator_agent: str = Field(default="cc3-provenance-agent")
    immutable_hash: str = Field(default="", description="不可变哈希")

    # C2PA 签名 (可选)
    signature: str = Field(default="", description="C2PA 式加密签名")

    def model_post_init(self, __context: Any) -> None:
        """创建后计算不可变哈希."""
        if not self.immutable_hash:
            self.immutable_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """计算标注的 SHA-256 哈希.

        哈希覆盖七维标注核心内容 + 元数据,
        用于溯源链完整性校验。
        """
        payload = {
            "annotation_id": self.annotation_id,
            "target_type": self.target_type.value,
            "target_id": self.target_id,
            "source": self.source.model_dump(),
            "generation": self.generation.model_dump(),
            "validation": self.validation.model_dump(),
            "decision": self.decision.model_dump(),
            "evolution": self.evolution.model_dump(),
            "propagation": self.propagation.model_dump(),
            "relation": self.relation.model_dump(),
            "created_at": self.created_at,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """验证不可变哈希是否匹配."""
        return self.compute_hash() == self.immutable_hash

    def completeness_score(self) -> float:
        """计算七维标注整体完整度 (0.0-1.0)."""
        scores = [
            self.source.completeness(),
            self.generation.completeness(),
            self.validation.completeness(),
            self.decision.completeness(),
            self.evolution.completeness(),
            self.propagation.completeness(),
            self.relation.completeness(),
        ]
        return round(sum(scores) / len(scores), 4)

    def filled_dimensions(self) -> list[str]:
        """返回已填充的维度名称列表."""
        dims = {
            "source": self.source.is_filled(),
            "generation": self.generation.is_filled(),
            "validation": self.validation.is_filled(),
            "decision": self.decision.is_filled(),
            "evolution": self.evolution.is_filled(),
            "propagation": self.propagation.is_filled(),
            "relation": self.relation.is_filled(),
        }
        return [k for k, v in dims.items() if v]

    def missing_dimensions(self) -> list[str]:
        """返回未填充的维度名称列表."""
        all_dims = {
            "source", "generation", "validation", "decision",
            "evolution", "propagation", "relation",
        }
        return sorted(all_dims - set(self.filled_dimensions()))


# ============================================================
# 辩论日志数据模型
# ============================================================


class DebateArgument(BaseModel):
    """辩论论点.

    Generator 提出的论点或 Reviewer 的反驳论点。
    """

    point_id: str = Field(
        default_factory=lambda: f"arg-{uuid.uuid4().hex[:8]}",
    )
    point: str = Field(default="", description="论点内容")
    source: str = Field(default="", description="论据来源")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="置信度 (0.0-1.0)",
    )
    evidence_type: str = Field(
        default="",
        description="证据类型 (citation/experiment/calculation/logic/analogy)",
    )
    evidence_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="证据元数据",
    )


class DebateCounterargument(BaseModel):
    """辩论反驳.

    Reviewer 对 Generator 论点的反驳。
    """

    counter_id: str = Field(
        default_factory=lambda: f"ctr-{uuid.uuid4().hex[:8]}",
    )
    targets: list[str] = Field(
        default_factory=list,
        description="目标论点 ID 列表 (支持一对多)",
    )
    counter: str = Field(default="", description="反驳内容")
    source: str = Field(default="", description="反驳来源")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
    )
    counter_type: CounterType = Field(
        default=CounterType.DIRECT_REFUTATION,
    )


class DebateRound(BaseModel):
    """辩论轮次记录."""

    round_number: int = Field(default=1, ge=1)
    generator_arguments: list[DebateArgument] = Field(
        default_factory=list,
        description="Generator 论点列表",
    )
    reviewer_counterarguments: list[DebateCounterargument] = Field(
        default_factory=list,
        description="Reviewer 反驳列表",
    )
    round_divergence: float = Field(
        default=0.0, ge=0.0,
        description="本轮分歧度",
    )
    round_timestamp: float = Field(
        default_factory=time.time,
    )
    round_duration_ms: float = Field(
        default=0.0,
        description="本轮耗时 (毫秒)",
    )


class AdjudicatorVerdict(BaseModel):
    """裁决结果."""

    adjudicator_id: str = Field(default="", description="裁决 Agent ID")
    consensus_position: str = Field(
        default="",
        description="共识立场",
    )
    three_dimensional_score: dict[str, float] = Field(
        default_factory=dict,
        description="三维评分: {accuracy, completeness, pedagogical_fit}",
    )
    adopted_arguments: list[str] = Field(
        default_factory=list,
        description="采纳的论点 ID 列表",
    )
    rejected_arguments: list[str] = Field(
        default_factory=list,
        description="驳回的论点 ID 列表",
    )
    modification_notes: str = Field(
        default="",
        description="修改说明",
    )
    verdict_timestamp: float = Field(
        default_factory=time.time,
    )


class DebateResourceUsage(BaseModel):
    """辩论资源消耗."""

    total_tokens: int = Field(default=0)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    api_calls: int = Field(default=0)
    compute_time_ms: float = Field(default=0.0)
    external_tool_calls: int = Field(default=0)
    estimated_cost: float = Field(default=0.0, description="估算成本 (美元)")


class DebateOutcome(BaseModel):
    """辩论结果影响."""

    final_consensus: str = Field(default="", description="最终共识")
    affected_kp_ids: list[str] = Field(
        default_factory=list,
        description="受影响的知识点 ID 列表",
    )
    kg_relations_updated: list[dict[str, Any]] = Field(
        default_factory=list,
        description="知识图谱关系更新",
    )
    bkt_adjustments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="BKT 参数调整",
    )
    adopted_into_kb: bool = Field(
        default=False,
        description="是否采纳进知识库",
    )
    kb_version_after: str = Field(
        default="",
        description="知识库版本 (采纳后)",
    )


class PreDebateRecord(BaseModel):
    """辩论前记录 (Pre-Debate).

    记录辩论触发原因和边界预设。
    """

    complexity_score: float = Field(
        default=0.0,
        description="复杂度评分 (31-65 区间触发辩论)",
    )
    threshold_range: str = Field(
        default="31-65",
        description="触发阈值范围",
    )
    historical_context: str = Field(
        default="",
        description="历史上下文",
    )
    focus_area: str = Field(default="", description="辩论焦点")
    excluded_topics: list[str] = Field(
        default_factory=list,
        description="排除的话题",
    )
    source_tier_requirement: SourceTier = Field(
        default=SourceTier.TIER_2,
        description="来源等级要求",
    )
    acceptable_evidence_types: list[str] = Field(
        default_factory=list,
        description="可接受的证据类型",
    )
    time_range_constraint: float = Field(
        default=0.0,
        description="时间约束 (秒, 0=无限制)",
    )
    participant_configs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="参与 Agent 配置",
    )


class DebateLog(BaseModel):
    """辩论日志 — 完整辩论记录.

    包含辩论的完整生命周期:
    Pre-Debate → During-Debate (轮次) → Post-Debate (裁决+结果)

    三级日志级别:
    - Summary: 仅元数据 + 收敛结果 + 裁决
    - Full: + 全部轮次 + 资源消耗
    - Debug: + Prompt 输入输出 (脱敏) + 中间状态

    Attributes:
        debate_log_id: 日志 ID
        debate_id: 辩论 ID
        task_id: 关联任务 ID
        verbosity: 日志级别
        pre_debate: 辩论前记录
        rounds: 辩论轮次列表
        convergence_reached: 是否收敛
        final_divergence: 最终分歧度
        convergence_round: 收敛轮次
        divergence_curve: 分歧度曲线
        adjudicator_verdict: 裁决结果
        resource_usage: 资源消耗
        outcome: 辩论结果
        created_at: 创建时间
        immutable_hash: 不可变哈希
    """

    debate_log_id: str = Field(
        default_factory=lambda: f"dl-{uuid.uuid4().hex[:12]}",
    )
    debate_id: str = Field(default="", description="辩论 ID")
    task_id: str = Field(default="", description="关联任务 ID")
    session_id: str = Field(default="", description="会话 ID")
    trigger_reason: str = Field(default="", description="触发原因")
    verbosity: LogVerbosity = Field(
        default=LogVerbosity.SUMMARY,
        description="日志级别",
    )

    # 辩论前
    pre_debate: PreDebateRecord = Field(
        default_factory=PreDebateRecord,
    )

    # 辩论中 (轮次)
    rounds: list[DebateRound] = Field(
        default_factory=list,
        description="辩论轮次列表",
    )

    # 收敛信息
    convergence_status: ConvergenceStatus = Field(
        default=ConvergenceStatus.NOT_CONVERGED,
    )
    convergence_reached: bool = Field(default=False)
    final_divergence: float = Field(default=0.0)
    convergence_round: int = Field(default=0)
    divergence_curve: list[float] = Field(
        default_factory=list,
        description="分歧度曲线 (每轮一个值)",
    )
    convergence_threshold: float = Field(
        default=0.1,
        description="收敛阈值 (divergence < threshold)",
    )
    max_rounds: int = Field(default=3, description="最大轮次")

    # 裁决
    adjudicator_verdict: AdjudicatorVerdict | None = Field(
        default=None,
    )

    # 资源消耗
    resource_usage: DebateResourceUsage = Field(
        default_factory=DebateResourceUsage,
    )

    # 结果
    outcome: DebateOutcome = Field(
        default_factory=DebateOutcome,
    )

    # 元数据
    created_at: float = Field(default_factory=time.time)
    persisted_at: float = Field(
        default=0.0,
        description="持久化时间 (辩论结束后5秒内)",
    )
    immutable_hash: str = Field(default="")

    # 调试信息 (仅 DEBUG 级别)
    debug_prompts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="调试级 Prompt 记录 (脱敏后)",
    )

    def model_post_init(self, __context: Any) -> None:
        """创建后计算不可变哈希."""
        if not self.immutable_hash:
            self.immutable_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """计算辩论日志的 SHA-256 哈希."""
        payload = {
            "debate_log_id": self.debate_log_id,
            "debate_id": self.debate_id,
            "task_id": self.task_id,
            "verbosity": self.verbosity.value,
            "pre_debate": self.pre_debate.model_dump(),
            "rounds": [r.model_dump() for r in self.rounds],
            "convergence_status": self.convergence_status.value,
            "convergence_reached": self.convergence_reached,
            "final_divergence": self.final_divergence,
            "adjudicator_verdict": (
                self.adjudicator_verdict.model_dump()
                if self.adjudicator_verdict
                else None
            ),
            "outcome": self.outcome.model_dump(),
            "created_at": self.created_at,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """验证不可变哈希是否匹配."""
        return self.compute_hash() == self.immutable_hash

    def is_converged(self) -> bool:
        """检查辩论是否收敛."""
        return (
            self.convergence_reached
            or self.convergence_status == ConvergenceStatus.CONVERGED
        )


# ============================================================
# 溯源链节点模型
# ============================================================


class ProvenanceChainNode(BaseModel):
    """溯源链节点.

    每个节点代表溯源链中的一个处理步骤,
    通过 prev_hash 形成不可篡改的链式结构。

    融合方案:
    - RFC 6962 Certificate Transparency: Merkle 树 append-only
    - 区块链: prev_hash 链式校验
    - OpenTelemetry: span 父子关系
    """

    node_id: str = Field(
        default_factory=lambda: f"pcn-{uuid.uuid4().hex[:10]}",
    )
    chain_id: str = Field(default="", description="所属链 ID")
    node_index: int = Field(default=0, ge=0, description="节点序号")
    annotation_id: str = Field(
        default="",
        description="关联的 KPA 标注 ID",
    )
    target_id: str = Field(default="", description="处理对象 ID")
    agent_id: str = Field(default="", description="处理 Agent ID")
    agent_role: str = Field(
        default="annotator",
        description="Agent 角色 (annotator/generator/reviewer/adjudicator)",
    )
    layer: str = Field(
        default="",
        description="所属架构层 (L2-L7/CC1-CC3)",
    )
    direction: CrossLayerDirection | None = Field(
        default=None,
        description="跨层传递方向",
    )
    timestamp: float = Field(default_factory=time.time)
    node_hash: str = Field(default="", description="本节点哈希")
    prev_hash: str = Field(default="", description="前一节点哈希")
    merkle_proof: str = Field(
        default="",
        description="Merkle 证明 (稀疏证明)",
    )

    def compute_node_hash(self) -> str:
        """计算节点哈希."""
        payload = {
            "node_id": self.node_id,
            "chain_id": self.chain_id,
            "node_index": self.node_index,
            "annotation_id": self.annotation_id,
            "target_id": self.target_id,
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "layer": self.layer,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        """创建后计算节点哈希."""
        if not self.node_hash:
            self.node_hash = self.compute_node_hash()


# ============================================================
# L0 Ledger 事件模型
# ============================================================


class LedgerEvent(BaseModel):
    """L0 Provenance Ledger 事件.

    KPA 和 DL 作为 L0 五类事件的扩展 payload 写入。
    所有事件 append-only, 不可修改。

    融合方案:
    - AWS CloudTrail: 不可变审计日志
    - PostgreSQL INSERT ONLY: 强制不可变
    - TimescaleDB: 时序数据自动分区
    """

    event_id: str = Field(
        default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}",
    )
    event_type: EventType = Field(
        default=EventType.INTERACTION,
        description="事件类型 (五类)",
    )
    trace_id: str = Field(
        default="",
        description="全链路 trace_id (跨 Agent 端到端回溯)",
    )
    session_id: str = Field(default="", description="会话 ID")
    agent_id: str = Field(default="", description="Agent ID")
    layer: str = Field(default="", description="架构层")
    timestamp: float = Field(default_factory=time.time)
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="事件负载 (KPA/DL 数据)",
    )
    prev_hash: str = Field(default="", description="前一事件哈希")
    event_hash: str = Field(default="", description="本事件哈希")

    def compute_event_hash(self) -> str:
        """计算事件哈希."""
        payload = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "layer": self.layer,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        """创建后计算事件哈希."""
        if not self.event_hash:
            self.event_hash = self.compute_event_hash()


# ============================================================
# 审计验证结果模型
# ============================================================


class AuditVerificationResult(BaseModel):
    """溯源审计验证结果."""

    verification_id: str = Field(
        default_factory=lambda: f"avr-{uuid.uuid4().hex[:10]}",
    )
    scope: str = Field(
        default="",
        description="验证范围 (artifact/session/time_range)",
    )
    scope_id: str = Field(default="", description="范围 ID")
    total_records: int = Field(default=0, description="总记录数")
    passed_records: int = Field(default=0, description="通过记录数")
    failed_records: int = Field(default=0, description="失败记录数")
    hash_chain_verified: bool = Field(default=False)
    actor_consistency_verified: bool = Field(default=False)
    timestamp_monotonic: bool = Field(default=False)
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    failures: list[dict[str, Any]] = Field(
        default_factory=list,
        description="失败详情列表",
    )
    verified_at: float = Field(default_factory=time.time)
