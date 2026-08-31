"""L3 领域知识层 — 核心数据模型.

融合世界先进方案的数据模型体系：
- W3C OWL/RDF: SPO 三元组 + 类层级 + 属性约束
- Wikidata: 声明-限定符-引用 四层结构 + 排名机制
- Schema.org: 单根继承类型树 + 标准属性
- LlamaIndex: Document→Node 层级 + 关系保留
- Dublin Core: 15 个核心元数据属性
- PROV-O: Entity-Activity-Agent 溯源三元组
- DBpedia/Wikidata: 11 维质量框架 (精选 6 维)
- RAG-Anything: 多模态内容类型 (text/image/table/equation)
- Milvus/Pinecone: 向量 + 元数据 + 分区键

所有模型基于 pydantic v2，枚举采用 (str, Enum) 风格与 L6 保持一致。
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# 枚举定义
# ============================================================


class ContentModality(str, Enum):
    """内容模态类型 (借鉴 RAG-Anything 多模态框架).

    统一表示不同模态的知识片段，支持跨模态检索。
    """

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    CODE = "code"
    MIXED = "mixed"


class EntityType(str, Enum):
    """知识实体类型 (借鉴 Schema.org 类型树 + ChemOnt + Materials Ontology).

    采用单根继承语义: 所有类型共享基础属性 (name, description, identifier)。
    """

    CONCEPT = "concept"
    CHEMICAL_COMPOUND = "chemical_compound"
    MATERIAL = "material"
    PAPER = "paper"
    TEXTBOOK = "textbook"
    DATASET = "dataset"
    METHOD = "method"
    PERSON = "person"
    ORGANIZATION = "organization"
    DOCUMENT_CHUNK = "document_chunk"
    COURSE = "course"
    EXPERIMENT = "experiment"
    # ---- 教育知识图谱分层类型 (知识图谱重构 P0) ----
    TOPIC = "topic"  # 章/节 主题节点 (part_of 挂载知识点)
    KNOWLEDGE_POINT = "knowledge_point"  # 教学知识点
    FACT = "fact"  # 权威事实
    ROLE = "role"  # 职业角色
    QUESTION = "question"  # 习题
    # ---- 领域实体子类型 ----
    ION = "ion"  # 离子 (激活剂 Dy3+/敏化剂 Yb3+/基质阳离子)
    ENERGY_LEVEL = "energy_level"  # 能级/跃迁 (4F9/2, 6H15/2, 4f-4f, 4f-5d ...)
    PARAMETER = "parameter"  # 性能参数 (量子效率/色坐标/色温/荧光寿命/T50 ...)


class RelationType(str, Enum):
    """实体间关系类型 (借鉴 OWL ObjectProperty + RDF 谓词).

    覆盖引用、派生、组成、属性、矛盾等语义关系。
    """

    CITES = "cites"
    DERIVED_FROM = "derived_from"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    AUTHORED_BY = "authored_by"
    PUBLISHED_IN = "published_in"
    HAS_PROPERTY = "has_property"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    EQUIVALENT_TO = "equivalent_to"
    DEPENDS_ON = "depends_on"
    INSTANTIATES = "instantiates"
    SUPERSEDES = "supersedes"
    REFERENCES = "references"
    # ---- 教学语义关系 (知识点间横向/纵向拓展, 镝-绿色健康照明领域) ----
    # 纵向: 沿"前提 → 深化"钻深/溯源
    PREREQUISITE_OF = "prerequisite_of"  # 前提: 学 B 前先掌握 A (A 是 B 的 prerequisite)
    DEEPENS = "deepens"  # 深化: A 的延伸/应用 (prerequisite_of 的逆)
    # 横向: 跨知识点/跨域的关联 (旁通/跳转)
    ANALOGOUS_TO = "analogous_to"  # 同类机制/对比 (如浓度猝灭 ↔ 热猝灭)
    AFFECTS = "affects"  # 因果/影响 (跨域, 如掺杂配比 → 浓度猝灭)
    CHARACTERIZED_BY = "characterized_by"  # 表征关联 (机理/材料 → 用什么方法测)
    SUBCONCEPT_OF = "subconcept_of"  # 上下位: 广义概念 → 狭义概念 (猝灭 → 浓度猝灭)
    APPLIES_TO = "applies_to"  # 应用: 材料/机理 → 应用场景 (单基质白光 → 健康照明)
    # ---- 实体/事实/知识点 桥接关系 (知识图谱重构 P0) ----
    MENTIONS = "mentions"  # 知识点/事实 → 实体 (材料/离子/方法/参数)
    MEASURED_BY = "measured_by"  # 性能参数 → 表征方法
    DOPED_WITH = "doped_with"  # 材料 → 激活剂离子


class SourceTier(str, Enum):
    """数据源层级 (对应 L6 connector_tools.py 的三层分类).

    TIER1: 公共数据库 (NIST, PubChem, arXiv 等)
    TIER2: 行业数据库 (CAS, WoS, SciFinder 等)
    TIER3: 校园数据源 (图书馆, LIMS, 教务等)
    INTERNAL: 内部文档 (PDF, 教材, 论文切片等)
    """

    TIER1_PUBLIC = "tier1_public"
    TIER2_INDUSTRY = "tier2_industry"
    TIER3_CAMPUS = "tier3_campus"
    INTERNAL_DOCUMENT = "internal_document"


class QualityDimension(str, Enum):
    """知识质量评估维度 (精选自 DBpedia/Wikidata 11 维框架).

    选取对多智能体决策最关键的 6 个维度:
    - accuracy: 与真实世界的一致程度
    - trustworthiness: 来源可靠程度
    - consistency: 知识库内部无矛盾
    - timeliness: 知识新鲜度和时效性
    - completeness: 知识覆盖的全面程度
    - relevancy: 与目标领域的相关程度
    """

    ACCURACY = "accuracy"
    TRUSTWORTHINESS = "trustworthiness"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    COMPLETENESS = "completeness"
    RELEVANCY = "relevancy"


class AccessLevel(str, Enum):
    """访问控制级别 (借鉴企业知识管理 RBAC 模型)."""

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class ChunkingStrategy(str, Enum):
    """文档切片策略."""

    FIXED_LENGTH = "fixed_length"
    SEMANTIC_PARAGRAPH = "semantic_paragraph"
    RECURSIVE_CHAR = "recursive_char"
    STRUCTURED_HEADING = "structured_heading"


class StatementRank(str, Enum):
    """声明排名 (借鉴 Wikidata 排名机制).

    PREFERRED: 首选声明 (当前最佳值)
    NORMAL: 正常声明 (默认)
    DEPRECATED: 已弃用声明 (过时或不准确)
    """

    PREFERRED = "preferred"
    NORMAL = "normal"
    DEPRECATED = "deprecated"


class ChunkRelationshipType(str, Enum):
    """切片间关系类型 (借鉴 LlamaIndex NodeRelationship).

    保留切片的结构关系，支持层级检索和自动合并。
    """

    PARENT = "parent"
    CHILD = "child"
    PREVIOUS = "previous"
    NEXT = "next"
    SOURCE = "source"


class ProvenanceRole(str, Enum):
    """溯源角色 (借鉴 PROV-O Agent 角色)."""

    GENERATOR = "generator"
    CONTRIBUTOR = "contributor"
    VALIDATOR = "validator"
    CURATOR = "curator"


class KnowledgeStatus(str, Enum):
    """知识状态生命周期 (借鉴 Wikidata 修订状态 + 企业知识管理生命周期).

    DRAFT: 草稿状态，尚未审核
    ACTIVE: 活跃状态，可供检索使用
    ARCHIVED: 已归档，不再活跃但保留历史
    DEPRECATED: 已弃用，不推荐使用但仍可访问
    """

    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ConflictType(str, Enum):
    """知识冲突类型 (借鉴 MACR 多智能体冲突解决框架).

    TEMPORAL: 时间冲突 — 同一事实在不同时间点的值不同
    SOURCE_BASED: 来源冲突 — 不同来源对同一事实的声明不同
    SEMANTIC: 语义冲突 — 声明在语义层面相互矛盾
    """

    TEMPORAL = "temporal"
    SOURCE_BASED = "source_based"
    SEMANTIC = "semantic"


class ConflictResolutionStrategy(str, Enum):
    """冲突解决策略 (借鉴 MACR + Detect-Then-Resolve 模式).

    KEEP_BOTH: 保留双方声明，标记为冲突
    PREFER_HIGHER_QUALITY: 采纳质量分数更高的声明
    PREFER_MOST_RECENT: 采纳最新的声明 (时间优先)
    PREFER_MOST_TRUSTED: 采纳来源可信度最高的声明
    MANUAL_REVIEW: 提交人工审核
    """

    KEEP_BOTH = "keep_both"
    PREFER_HIGHER_QUALITY = "prefer_higher_quality"
    PREFER_MOST_RECENT = "prefer_most_recent"
    PREFER_MOST_TRUSTED = "prefer_most_trusted"
    MANUAL_REVIEW = "manual_review"


class QueryOperator(str, Enum):
    """查询操作符 (借鉴 SPARQL FILTER + SHACL 约束操作符).

    EQ: 等于
    NE: 不等于
    GT: 大于
    GTE: 大于等于
    LT: 小于
    LTE: 小于等于
    CONTAINS: 包含 (字符串)
    STARTS_WITH: 以...开头
    ENDS_WITH: 以...结尾
    IN: 在集合中
    REGEX: 正则匹配
    """

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    REGEX = "regex"


class PropertyDataType(str, Enum):
    """属性数据类型 (借鉴 SHACL sh:datatype + OWL 数据类型).

    STRING: 字符串
    INTEGER: 整数
    FLOAT: 浮点数
    BOOLEAN: 布尔值
    DATETIME: 日期时间 (ISO 8601 时间戳)
    ENTITY_REF: 实体引用 (object 属性)
    LIST: 列表类型
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ENTITY_REF = "entity_ref"
    LIST = "list"


class InferenceRuleType(str, Enum):
    """推理规则类型 (借鉴 OWL Reasoner 推理能力分类).

    TRANSITIVE_CLOSURE: 传递闭包 (A→B, B→C ⟹ A→C)
    INVERSE_RELATION: 逆关系推理 (A→B, inverse(B→A) ⟹ B→A)
    SUBCLASS_INHERITANCE: 子类继承 (B subClassOf A, x instanceOf B ⟹ x instanceOf A)
    SYMMETRIC_CLOSURE: 对称闭包 (A→B ⟹ B→A)
    PROPERTY_CHAIN: 属性链 (A→B→C ⟹ A→C)
    """

    TRANSITIVE_CLOSURE = "transitive_closure"
    INVERSE_RELATION = "inverse_relation"
    SUBCLASS_INHERITANCE = "subclass_inheritance"
    SYMMETRIC_CLOSURE = "symmetric_closure"
    PROPERTY_CHAIN = "property_chain"


class VerificationStatus(str, Enum):
    """知识验证状态 (借鉴 ProVe 自动溯源验证 + 证据分层).

    UNVERIFIED: 未验证 (新导入，尚未审核)
    CANDIDATE: 候选 (已通过初步检查，待最终验证)
    VERIFIED: 已验证 (通过自动验证或人工审核)
    DISPUTED: 有争议 (存在冲突或被质疑)
    """

    UNVERIFIED = "unverified"
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    DISPUTED = "disputed"


# ============================================================
# 数据源元数据 (Dublin Core + connector_tools.py)
# ============================================================


class KnowledgeSource(BaseModel):
    """知识数据源元数据.

    描述知识的来源信息，支持 25 个已注册连接器 + 内部文档源。
    借鉴 Dublin Core 的 source/identifier/publisher 字段。

    Attributes:
        source_id: 数据源唯一标识 (如 "nist", "pubchem", "internal_pdf")
        name: 数据源名称
        tier: 数据源层级
        endpoint: API 端点 URL (外部数据源)
        auth_required: 是否需要认证
        access_level: 访问控制级别
        reliability: 来源可信度 (0.0~1.0)，影响 QualityScore.trustworthiness
        last_synced: 最后同步时间戳
        metadata: 扩展元数据
    """

    source_id: str = Field(..., description="数据源唯一标识")
    name: str = Field(..., description="数据源名称")
    tier: SourceTier = Field(default=SourceTier.INTERNAL_DOCUMENT, description="数据源层级")
    endpoint: str = Field(default="", description="API 端点 URL")
    auth_required: bool = Field(default=False, description="是否需要认证")
    access_level: AccessLevel = Field(default=AccessLevel.INTERNAL, description="访问控制级别")
    reliability: float = Field(default=0.8, ge=0.0, le=1.0, description="来源可信度")
    last_synced: float = Field(default=0.0, description="最后同步时间戳")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    def is_stale(self, max_age_seconds: float = 86400.0) -> bool:
        """检查数据源是否过期 (默认 24 小时)."""
        if self.last_synced == 0.0:
            return True
        return (time.time() - self.last_synced) > max_age_seconds


# ============================================================
# Wikidata 声明-限定符模型
# ============================================================


class KnowledgeQualifier(BaseModel):
    """声明限定符 (借鉴 Wikidata Qualifier).

    为 SPO 三元组附加元信息，如时间范围、置信度、方法等。
    例如: "水 —沸点→ 100°C" + 限定符 {条件: "1 atm", 来源: "NIST"}

    Attributes:
        qualifier_id: 限定符唯一标识
        name: 限定符名称 (如 "condition", "method", "temperature_range")
        value: 限定符值
        value_type: 值类型 (string/number/boolean/datetime)
    """

    qualifier_id: str = Field(default_factory=lambda: f"q-{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="限定符名称")
    value: Any = Field(..., description="限定符值")
    value_type: str = Field(default="string", description="值类型")


class KnowledgeTriple(BaseModel):
    """SPO 三元组 (借鉴 W3C RDF + Wikidata 声明).

    知识的最小表示单元: Subject-Predicate-Object。
    每个三元组可携带限定符、引用和排名。

    Attributes:
        triple_id: 三元组唯一标识
        subject_id: 主语实体 ID
        predicate: 谓词 (RelationType 值)
        object_id: 宾语实体 ID (或字面值)
        object_value: 宾语字面值 (当宾语不是实体时)
        object_is_literal: 宾语是否为字面值
        qualifiers: 限定符列表 (Wikidata 风格)
        rank: 声明排名 (PREFERRED/NORMAL/DEPRECATED)
        confidence: 置信度 (0.0~1.0)
        source_id: 引用来源 ID
        created_at: 创建时间戳
    """

    triple_id: str = Field(default_factory=lambda: f"t-{uuid.uuid4().hex[:12]}")
    subject_id: str = Field(..., description="主语实体 ID")
    predicate: str = Field(..., description="谓词 (RelationType 值)")
    object_id: str = Field(default="", description="宾语实体 ID")
    object_value: Any = Field(default=None, description="宾语字面值")
    object_is_literal: bool = Field(default=False, description="宾语是否为字面值")
    qualifiers: list[KnowledgeQualifier] = Field(default_factory=list, description="限定符列表")
    rank: StatementRank = Field(default=StatementRank.NORMAL, description="声明排名")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")
    source_id: str = Field(default="", description="引用来源 ID")
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    # ---- 增强字段: 时间有效性 (借鉴 valid_at/expired_at 模式) ----
    valid_from: float = Field(default=0.0, description="事实生效时间戳 (0=始终有效)")
    valid_until: float = Field(default=0.0, description="事实失效时间戳 (0=仍然有效)")

    @model_validator(mode="after")
    def _validate_object(self) -> KnowledgeTriple:
        """确保 object_id 或 object_value 至少有一个有效."""
        if not self.object_is_literal and not self.object_id:
            raise ValueError("宾语为实体时 object_id 不能为空")
        if self.object_is_literal and self.object_value is None:
            raise ValueError("宾语为字面值时 object_value 不能为 None")
        return self

    def is_preferred(self) -> bool:
        """是否为首选声明."""
        return self.rank == StatementRank.PREFERRED

    def is_deprecated(self) -> bool:
        """是否为已弃用声明."""
        return self.rank == StatementRank.DEPRECATED

    def is_valid_at(self, timestamp: float | None = None) -> bool:
        """在指定时间点是否有效 (借鉴 valid_at/expired_at 时间边界查询).

        Args:
            timestamp: 查询时间戳，None 表示当前时间

        Returns:
            是否在该时间点有效
        """
        if timestamp is None:
            timestamp = time.time()
        if self.valid_from > 0.0 and timestamp < self.valid_from:
            return False
        if self.valid_until > 0.0 and timestamp >= self.valid_until:
            return False
        return True

    def is_currently_valid(self) -> bool:
        """当前是否有效."""
        return self.is_valid_at(time.time())


# ============================================================
# 知识实体 (Schema.org + OWL + Dublin Core)
# ============================================================


class KnowledgeEntity(BaseModel):
    """知识实体 (核心模型).

    融合 Schema.org 类型树、OWL 类层级、Dublin Core 元数据、
    Wikidata 声明-限定符、PROV-O 溯源。

    一个知识实体代表领域中的一个概念、物体、事件或文档片段，
    拥有类型、属性、关系、质量评分和溯源信息。

    Attributes:
        entity_id: 实体唯一标识
        entity_type: 实体类型 (EntityType)
        name: 实体名称 (Dublin Core: title)
        description: 实体描述 (Dublin Core: description)
        identifiers: 外部标识符映射 (如 {"cas": "7732-18-5", "doi": "10.xxx"})
        properties: 动态属性 (Dublin Core: subject/creator/publisher 等)
        triples: 以此实体为主语的三元组列表
        domain: 所属领域 (如 "chemistry", "materials", "education")
        access_level: 访问控制级别
        version: 版本号
        parent_entity_id: 父实体 ID (支持类层级)
        created_at: 创建时间戳
        updated_at: 最后更新时间戳
        source: 知识来源
        quality: 质量评分
        provenance: 溯源信息
        metadata: 扩展元数据
    """

    entity_id: str = Field(default_factory=lambda: f"e-{uuid.uuid4().hex[:12]}")
    entity_type: EntityType = Field(..., description="实体类型")
    name: str = Field(..., min_length=1, description="实体名称")
    description: str = Field(default="", description="实体描述")
    identifiers: dict[str, str] = Field(default_factory=dict, description="外部标识符映射")
    properties: dict[str, Any] = Field(default_factory=dict, description="动态属性")
    triples: list[KnowledgeTriple] = Field(default_factory=list, description="以此实体为主语的三元组")
    domain: str = Field(default="general", description="所属领域")
    access_level: AccessLevel = Field(default=AccessLevel.INTERNAL, description="访问控制级别")
    version: int = Field(default=1, description="版本号")
    parent_entity_id: str = Field(default="", description="父实体 ID (类层级)")
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    updated_at: float = Field(default_factory=time.time, description="最后更新时间戳")
    source: KnowledgeSource | None = Field(default=None, description="知识来源")
    quality: QualityScore | None = Field(default=None, description="质量评分")
    provenance: ProvenanceInfo | None = Field(default=None, description="溯源信息")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")
    # ---- 增强字段 (借鉴 Wikidata 别名/标签 + 企业知识管理) ----
    tags: list[str] = Field(default_factory=list, description="标签列表 (便于分类和检索)")
    aliases: list[str] = Field(default_factory=list, description="别名列表 (同义词/缩写)")
    language: str = Field(default="zh", description="主要语言 (ISO 639-1)")
    status: KnowledgeStatus = Field(default=KnowledgeStatus.ACTIVE, description="知识状态")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="整体置信度 [0,1]")
    is_verified: bool = Field(default=False, description="是否已验证")

    def touch(self) -> None:
        """更新 updated_at 时间戳."""
        self.updated_at = time.time()

    def add_triple(self, triple: KnowledgeTriple) -> None:
        """添加三元组并更新时间戳."""
        if triple.subject_id != self.entity_id:
            triple.subject_id = self.entity_id
        self.triples.append(triple)
        self.touch()

    def get_triples_by_predicate(self, predicate: str) -> list[KnowledgeTriple]:
        """按谓词筛选三元组."""
        return [t for t in self.triples if t.predicate == predicate]

    def get_preferred_triples(self) -> list[KnowledgeTriple]:
        """获取所有首选声明."""
        return [t for t in self.triples if t.is_preferred()]

    def get_active_triples(self) -> list[KnowledgeTriple]:
        """获取所有非弃用声明."""
        return [t for t in self.triples if not t.is_deprecated()]

    def has_identifier(self, id_type: str) -> bool:
        """是否拥有指定类型的外部标识符."""
        return id_type in self.identifiers

    def is_newer_than(self, other: KnowledgeEntity) -> bool:
        """版本是否比另一个实体新."""
        return self.version > other.version

    def is_active(self) -> bool:
        """是否为活跃状态."""
        return self.status == KnowledgeStatus.ACTIVE

    def has_alias(self, alias: str) -> bool:
        """是否拥有指定别名 (忽略大小写)."""
        return alias.lower() in (a.lower() for a in self.aliases)

    def has_tag(self, tag: str) -> bool:
        """是否拥有指定标签."""
        return tag in self.tags

    def match_name_or_alias(self, query: str) -> bool:
        """名称或别名是否匹配查询 (忽略大小写)."""
        q = query.lower()
        if self.name.lower() == q:
            return True
        return self.has_alias(query)


# ============================================================
# 文档切片模型 (LlamaIndex Node + Pinecone Record)
# ============================================================


class ChunkRelationship(BaseModel):
    """切片间关系 (借鉴 LlamaIndex RelatedNodeInfo).

    Attributes:
        relation_type: 关系类型
        target_chunk_id: 目标切片 ID
        target_metadata: 目标切片的元数据快照
    """

    relation_type: ChunkRelationshipType = Field(..., description="关系类型")
    target_chunk_id: str = Field(..., description="目标切片 ID")
    target_metadata: dict[str, Any] = Field(default_factory=dict, description="目标切片元数据快照")


class DocumentChunk(BaseModel):
    """文档切片 (借鉴 LlamaIndex TextNode + Pinecone Record).

    PDF/教材/论文解析后的原子知识单元，保留结构关系。
    每个切片包含文本内容、元数据、关系、向量和质量信息。

    Attributes:
        chunk_id: 切片唯一标识
        document_id: 所属文档 ID
        content: 文本内容
        content_type: 内容模态
        chunk_index: 在文档中的序号
        char_count: 字符数
        token_count: 估算 token 数
        section: 所属章节
        page: 页码
        strategy: 切片策略
        overlap_prev: 与前一切片的重叠字符数
        relationships: 与其他切片的关系
        metadata: 扩展元数据 (Dublin Core: source/creator/date 等)
        embedding: 向量 (延迟填充)
        quality: 质量评分
        provenance: 溯源信息
        created_at: 创建时间戳
    """

    chunk_id: str = Field(default_factory=lambda: f"c-{uuid.uuid4().hex[:12]}")
    document_id: str = Field(..., description="所属文档 ID")
    content: str = Field(..., min_length=1, description="文本内容")
    content_type: ContentModality = Field(default=ContentModality.TEXT, description="内容模态")
    chunk_index: int = Field(default=0, ge=0, description="在文档中的序号")
    char_count: int = Field(default=0, description="字符数")
    token_count: int = Field(default=0, description="估算 token 数")
    section: str = Field(default="", description="所属章节")
    page: int = Field(default=0, ge=0, description="页码")
    strategy: ChunkingStrategy = Field(default=ChunkingStrategy.FIXED_LENGTH, description="切片策略")
    overlap_prev: int = Field(default=0, ge=0, description="与前一切片的重叠字符数")
    relationships: list[ChunkRelationship] = Field(default_factory=list, description="切片间关系")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")
    embedding: EmbeddingVector | None = Field(default=None, description="向量")
    quality: QualityScore | None = Field(default=None, description="质量评分")
    provenance: ProvenanceInfo | None = Field(default=None, description="溯源信息")
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    # ---- 增强字段: 语言与布局 (借鉴 RAG-Anything 多模态 + PDF 解析) ----
    language: str = Field(default="zh", description="切片语言 (ISO 639-1)")
    language_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="语言检测置信度")
    bbox: list[float] = Field(default_factory=list, description="边界框 [x0, y0, x1, y1] (PDF 坐标)")
    heading_level: int = Field(default=0, ge=0, description="标题层级 (0=非标题, 1=H1, 2=H2, ...)")

    @model_validator(mode="after")
    def _auto_fill_counts(self) -> DocumentChunk:
        """自动填充字符数和 token 数."""
        if self.char_count == 0:
            self.char_count = len(self.content)
        if self.token_count == 0:
            self.token_count = max(1, len(self.content) // 4)
        return self

    def has_embedding(self) -> bool:
        """是否已向量化."""
        return self.embedding is not None and len(self.embedding.vector) > 0

    def get_parent_chunk_id(self) -> str | None:
        """获取父切片 ID."""
        for rel in self.relationships:
            if rel.relation_type == ChunkRelationshipType.PARENT:
                return rel.target_chunk_id
        return None

    def get_source_document(self) -> str:
        """获取来源文档标识."""
        return self.metadata.get("source", self.document_id)


# ============================================================
# 质量评分 (DBpedia/Wikidata 11 维框架精选)
# ============================================================


class QualityScore(BaseModel):
    """知识质量多维评分.

    精选 DBpedia/Wikidata 11 维质量框架中最关键的 6 个维度，
    为多智能体决策提供可计算的置信度分数。

    每个维度 0.0~1.0，综合分数为加权平均。

    Attributes:
        accuracy: 准确性 (与真实世界的一致程度)
        trustworthiness: 可信度 (来源可靠程度)
        consistency: 一致性 (知识库内部无矛盾)
        timeliness: 时效性 (知识新鲜度)
        completeness: 完整性 (知识覆盖全面程度)
        relevancy: 相关性 (与目标领域的相关程度)
        assessed_at: 评估时间戳
        assessor: 评估者 (agent_id 或 "system")
    """

    accuracy: float = Field(default=0.8, ge=0.0, le=1.0, description="准确性")
    trustworthiness: float = Field(default=0.8, ge=0.0, le=1.0, description="可信度")
    consistency: float = Field(default=0.9, ge=0.0, le=1.0, description="一致性")
    timeliness: float = Field(default=0.8, ge=0.0, le=1.0, description="时效性")
    completeness: float = Field(default=0.7, ge=0.0, le=1.0, description="完整性")
    relevancy: float = Field(default=0.8, ge=0.0, le=1.0, description="相关性")
    assessed_at: float = Field(default_factory=time.time, description="评估时间戳")
    assessor: str = Field(default="system", description="评估者")
    # ---- 增强字段: 证据追踪 (借鉴 ProVe + 证据分层) ----
    evidence_count: int = Field(default=0, ge=0, description="支持证据数量 (多源 corroboration)")
    peer_reviewed: bool = Field(default=False, description="是否经过同行评审")
    verification_status: VerificationStatus = Field(default=VerificationStatus.UNVERIFIED, description="验证状态")
    last_verified_at: float = Field(default=0.0, description="最后验证时间戳 (0=从未验证)")

    # 维度权重 (可自定义)
    _weights: dict[str, float] = {
        QualityDimension.ACCURACY.value: 0.25,
        QualityDimension.TRUSTWORTHINESS.value: 0.20,
        QualityDimension.CONSISTENCY.value: 0.15,
        QualityDimension.TIMELINESS.value: 0.15,
        QualityDimension.COMPLETENESS.value: 0.10,
        QualityDimension.RELEVANCY.value: 0.15,
    }

    @property
    def weights(self) -> dict[str, float]:
        """获取当前权重."""
        return self._weights

    def set_weights(self, weights: dict[str, float]) -> None:
        """自定义权重 (所有权重之和必须为 1.0)."""
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"权重之和必须为 1.0, 当前: {total}")
        self._weights = weights

    def overall(self) -> float:
        """计算综合质量分数 (加权平均)."""
        scores = {
            QualityDimension.ACCURACY.value: self.accuracy,
            QualityDimension.TRUSTWORTHINESS.value: self.trustworthiness,
            QualityDimension.CONSISTENCY.value: self.consistency,
            QualityDimension.TIMELINESS.value: self.timeliness,
            QualityDimension.COMPLETENESS.value: self.completeness,
            QualityDimension.RELEVANCY.value: self.relevancy,
        }
        return sum(scores[k] * self._weights.get(k, 0.0) for k in scores)

    def is_acceptable(self, threshold: float = 0.6) -> bool:
        """综合分数是否达到可接受阈值."""
        return self.overall() >= threshold

    def weakest_dimension(self) -> str:
        """获取得分最低的维度."""
        scores = {
            QualityDimension.ACCURACY.value: self.accuracy,
            QualityDimension.TRUSTWORTHINESS.value: self.trustworthiness,
            QualityDimension.CONSISTENCY.value: self.consistency,
            QualityDimension.TIMELINESS.value: self.timeliness,
            QualityDimension.COMPLETENESS.value: self.completeness,
            QualityDimension.RELEVANCY.value: self.relevancy,
        }
        return min(scores, key=scores.get)

    def strongest_dimension(self) -> str:
        """获取得分最高的维度."""
        scores = {
            QualityDimension.ACCURACY.value: self.accuracy,
            QualityDimension.TRUSTWORTHINESS.value: self.trustworthiness,
            QualityDimension.CONSISTENCY.value: self.consistency,
            QualityDimension.TIMELINESS.value: self.timeliness,
            QualityDimension.COMPLETENESS.value: self.completeness,
            QualityDimension.RELEVANCY.value: self.relevancy,
        }
        return max(scores, key=scores.get)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (含综合分数)."""
        return {
            "accuracy": self.accuracy,
            "trustworthiness": self.trustworthiness,
            "consistency": self.consistency,
            "timeliness": self.timeliness,
            "completeness": self.completeness,
            "relevancy": self.relevancy,
            "overall": round(self.overall(), 4),
            "assessed_at": self.assessed_at,
            "assessor": self.assessor,
            "evidence_count": self.evidence_count,
            "peer_reviewed": self.peer_reviewed,
            "verification_status": self.verification_status.value,
            "last_verified_at": self.last_verified_at,
        }

    def is_verified(self) -> bool:
        """是否已验证."""
        return self.verification_status == VerificationStatus.VERIFIED

    def is_disputed(self) -> bool:
        """是否有争议."""
        return self.verification_status == VerificationStatus.DISPUTED

    def corroboration_level(self) -> str:
        """获取证据强度等级 (借鉴证据分层).

        Returns:
            "none" (0 证据), "weak" (1), "moderate" (2-3), "strong" (4+)
        """
        if self.evidence_count == 0:
            return "none"
        elif self.evidence_count == 1:
            return "weak"
        elif self.evidence_count <= 3:
            return "moderate"
        else:
            return "strong"


# ============================================================
# 溯源信息 (PROV-O)
# ============================================================


class ProvenanceInfo(BaseModel):
    """知识溯源信息 (借鉴 W3C PROV-O).

    记录知识的生成、派生、归因链，支持完整审计。

    PROV-O 三大核心类映射:
    - Entity → KnowledgeEntity / DocumentChunk
    - Activity → 检索/推理/导入等活动
    - Agent → 智能体或系统

    Attributes:
        entity_id: 被溯源的实体 ID
        generated_by_activity: 生成此实体的活动 ID
        generated_by_agent: 生成此实体的智能体 ID
        agent_role: 智能体角色
        generated_at: 生成时间戳
        derived_from: 派生来源实体 ID 列表 (prov:wasDerivedFrom)
        primary_source: 原始来源 URI (prov:hadPrimarySource)
        revision_of: 修订前版本实体 ID (prov:wasRevisionOf)
        quoted_from: 引用来源实体 ID (prov:wasQuotedFrom)
        used_entities: 生成过程中使用的实体 ID 列表 (prov:used)
        activity_type: 活动类型 (如 "retrieve", "infer", "ingest", "chunk")
        activity_description: 活动描述
    """

    entity_id: str = Field(..., description="被溯源的实体 ID")
    generated_by_activity: str = Field(default="", description="生成活动 ID")
    generated_by_agent: str = Field(default="", description="生成智能体 ID")
    agent_role: ProvenanceRole = Field(default=ProvenanceRole.GENERATOR, description="智能体角色")
    generated_at: float = Field(default_factory=time.time, description="生成时间戳")
    derived_from: list[str] = Field(default_factory=list, description="派生来源实体 ID 列表")
    primary_source: str = Field(default="", description="原始来源 URI")
    revision_of: str = Field(default="", description="修订前版本实体 ID")
    quoted_from: str = Field(default="", description="引用来源实体 ID")
    used_entities: list[str] = Field(default_factory=list, description="使用的实体 ID 列表")
    activity_type: str = Field(default="", description="活动类型")
    activity_description: str = Field(default="", description="活动描述")
    # ---- 增强字段: 完整性校验 (借鉴区块链哈希 + 数字签名) ----
    integrity_hash: str = Field(default="", description="内容完整性哈希 (SHA-256)")
    signature: str = Field(default="", description="数字签名 (可选，用于防篡改验证)")

    def has_derivation_chain(self) -> bool:
        """是否有派生链."""
        return len(self.derived_from) > 0 or bool(self.primary_source)

    def is_original(self) -> bool:
        """是否为原始来源 (非派生)."""
        return not self.has_derivation_chain()

    def trace_depth(self, provenance_map: dict[str, ProvenanceInfo] | None = None) -> int:
        """计算溯源链深度 (需要提供 provenance_map 来查找上游)."""
        if not self.has_derivation_chain():
            return 0
        if provenance_map is None:
            return 1
        max_depth = 0
        for parent_id in self.derived_from:
            parent = provenance_map.get(parent_id)
            if parent is not None:
                max_depth = max(max_depth, parent.trace_depth(provenance_map) + 1)
            else:
                max_depth = max(max_depth, 1)
        return max_depth


# ============================================================
# 向量模型 (Milvus/Pinecone Record)
# ============================================================


class EmbeddingVector(BaseModel):
    """向量化数据 (借鉴 Milvus/Pinecone 向量记录).

    Attributes:
        content_id: 对应的内容 ID (chunk_id 或 entity_id)
        vector: 密集向量 (语义搜索)
        sparse_vector: 稀疏向量 (BM25 全文检索，可选)
        dim: 向量维度
        model: 编码模型名称 (如 "text-embedding-3-small")
        created_at: 编码时间戳
    """

    content_id: str = Field(..., description="对应的内容 ID")
    vector: list[float] = Field(default_factory=list, description="密集向量")
    sparse_vector: dict[int, float] = Field(default_factory=dict, description="稀疏向量 (BM25)")
    dim: int = Field(default=0, description="向量维度")
    model: str = Field(default="default", description="编码模型名称")
    created_at: float = Field(default_factory=time.time, description="编码时间戳")
    # ---- 增强字段: 向量元信息 ----
    normalized: bool = Field(default=False, description="是否已 L2 归一化")
    batch_id: str = Field(default="", description="批量编码 ID (便于追踪和回滚)")
    model_version: str = Field(default="", description="模型版本 (便于模型升级时的向量重建)")

    @model_validator(mode="after")
    def _auto_fill_dim(self) -> EmbeddingVector:
        """自动填充维度."""
        if self.dim == 0 and self.vector:
            self.dim = len(self.vector)
        return self

    def is_empty(self) -> bool:
        """向量是否为空."""
        return len(self.vector) == 0

    def cosine_similarity(self, other: EmbeddingVector) -> float:
        """计算余弦相似度."""
        if self.is_empty() or other.is_empty():
            return 0.0
        if self.dim != other.dim:
            return 0.0
        dot = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = sum(a * a for a in self.vector) ** 0.5
        norm_b = sum(b * b for b in other.vector) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ============================================================
# 检索结果
# ============================================================


class RetrievalResult(BaseModel):
    """知识检索结果.

    Attributes:
        query: 原始查询
        results: 检索到的切片/实体列表
        scores: 每个结果的相关性分数
        total: 总匹配数 (可能大于返回数)
        retrieval_time_ms: 检索耗时 (毫秒)
        source_type: 检索来源 ("vector", "keyword", "graph", "hybrid")
        filters: 使用的过滤条件
        trace_id: 全链路追踪 ID
    """

    query: str = Field(..., description="原始查询")
    results: list[dict[str, Any]] = Field(default_factory=list, description="检索结果列表")
    scores: list[float] = Field(default_factory=list, description="相关性分数列表")
    total: int = Field(default=0, ge=0, description="总匹配数")
    retrieval_time_ms: float = Field(default=0.0, ge=0.0, description="检索耗时 (毫秒)")
    source_type: str = Field(default="vector", description="检索来源类型")
    filters: dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    trace_id: str = Field(default="", description="全链路追踪 ID")

    def is_empty(self) -> bool:
        """结果是否为空."""
        return len(self.results) == 0

    def top_k(self, k: int) -> list[tuple[dict[str, Any], float]]:
        """获取前 k 个结果 (按分数降序)."""
        paired = list(zip(self.results, self.scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:k]

    def best_score(self) -> float:
        """最高分数."""
        return max(self.scores) if self.scores else 0.0


# ============================================================
# 知识库统计
# ============================================================


class KnowledgeBaseStats(BaseModel):
    """知识库统计信息.

    Attributes:
        total_entities: 实体总数
        total_chunks: 切片总数
        total_triples: 三元组总数
        total_sources: 数据源总数
        entities_by_type: 按类型分组的实体数
        chunks_by_modality: 按模态分组的切片数
        avg_quality: 平均质量分数
        indexed_vectors: 已索引向量数
        last_updated: 最后更新时间戳
    """

    total_entities: int = Field(default=0, ge=0, description="实体总数")
    total_chunks: int = Field(default=0, ge=0, description="切片总数")
    total_triples: int = Field(default=0, ge=0, description="三元组总数")
    total_sources: int = Field(default=0, ge=0, description="数据源总数")
    entities_by_type: dict[str, int] = Field(default_factory=dict, description="按类型分组的实体数")
    chunks_by_modality: dict[str, int] = Field(default_factory=dict, description="按模态分组的切片数")
    avg_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="平均质量分数")
    indexed_vectors: int = Field(default=0, ge=0, description="已索引向量数")
    last_updated: float = Field(default_factory=time.time, description="最后更新时间戳")

    def is_empty(self) -> bool:
        """知识库是否为空."""
        return self.total_entities == 0 and self.total_chunks == 0


# ============================================================
# 批量导入结果
# ============================================================


class IngestResult(BaseModel):
    """知识批量导入结果.

    Attributes:
        source: 导入来源标识
        total: 尝试导入总数
        success: 成功数
        failed: 失败数
        skipped: 跳过数 (重复)
        errors: 失败详情列表
        ingested_ids: 成功导入的 ID 列表
        duration_ms: 导入耗时 (毫秒)
        trace_id: 全链路追踪 ID
    """

    source: str = Field(default="", description="导入来源标识")
    total: int = Field(default=0, ge=0, description="尝试导入总数")
    success: int = Field(default=0, ge=0, description="成功数")
    failed: int = Field(default=0, ge=0, description="失败数")
    skipped: int = Field(default=0, ge=0, description="跳过数 (重复)")
    errors: list[dict[str, Any]] = Field(default_factory=list, description="失败详情列表")
    ingested_ids: list[str] = Field(default_factory=list, description="成功导入的 ID 列表")
    duration_ms: float = Field(default=0.0, ge=0.0, description="导入耗时 (毫秒)")
    trace_id: str = Field(default="", description="全链路追踪 ID")

    def is_full_success(self) -> bool:
        """是否全部成功."""
        return self.failed == 0 and self.skipped == 0

    def success_rate(self) -> float:
        """成功率."""
        if self.total == 0:
            return 0.0
        return self.success / self.total


# ============================================================
# 检索过滤器
# ============================================================


class RetrievalFilter(BaseModel):
    """知识检索过滤器 (借鉴 Milvus 标量过滤 + Pinecone metadata filter).

    Attributes:
        domain: 领域过滤
        entity_types: 实体类型过滤
        content_types: 内容模态过滤
        source_tiers: 数据源层级过滤
        access_level: 最大访问级别
        min_quality: 最低质量分数
        min_confidence: 最低置信度
        date_from: 起始日期时间戳
        date_to: 截止日期时间戳
        tags: 标签过滤
        exclude_deprecated: 是否排除已弃用声明
    """

    domain: str | None = Field(default=None, description="领域过滤")
    entity_types: list[EntityType] = Field(default_factory=list, description="实体类型过滤")
    content_types: list[ContentModality] = Field(default_factory=list, description="内容模态过滤")
    source_tiers: list[SourceTier] = Field(default_factory=list, description="数据源层级过滤")
    access_level: AccessLevel = Field(default=AccessLevel.INTERNAL, description="最大访问级别")
    min_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="最低质量分数")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="最低置信度")
    date_from: float = Field(default=0.0, description="起始日期时间戳")
    date_to: float = Field(default=0.0, description="截止日期时间戳")
    tags: list[str] = Field(default_factory=list, description="标签过滤")
    exclude_deprecated: bool = Field(default=True, description="是否排除已弃用声明")

    def matches_entity(self, entity: KnowledgeEntity) -> bool:
        """检查实体是否满足过滤条件."""
        if self.domain and entity.domain != self.domain:
            return False
        if self.entity_types and entity.entity_type not in self.entity_types:
            return False
        if self.source_tiers and entity.source and entity.source.tier not in self.source_tiers:
            return False
        if self.min_quality > 0.0:
            q = entity.quality
            if q is None or q.overall() < self.min_quality:
                return False
        if self.date_from > 0.0 and entity.created_at < self.date_from:
            return False
        if self.date_to > 0.0 and entity.created_at > self.date_to:
            return False
        return True

    def matches_chunk(self, chunk: DocumentChunk) -> bool:
        """检查切片是否满足过滤条件."""
        if self.content_types and chunk.content_type not in self.content_types:
            return False
        if self.min_quality > 0.0:
            q = chunk.quality
            if q is None or q.overall() < self.min_quality:
                return False
        if self.date_from > 0.0 and chunk.created_at < self.date_from:
            return False
        if self.date_to > 0.0 and chunk.created_at > self.date_to:
            return False
        return True


# ============================================================
# 子图提取配置 (借鉴 GraphRAG 子图提取模式)
# ============================================================


class SubgraphConfig(BaseModel):
    """子图提取配置 (借鉴 GraphRAG 实体中心子图提取).

    定义从知识图谱中提取子图的参数，支持跨文档子图聚合。

    Attributes:
        entity_focus: 中心实体锚点 ID
        max_depth: 遍历深度 (最大跳数，默认 2)
        max_entities: 子图大小上限
        min_confidence: 关系置信度过滤阈值
        min_quality: 最低质量分数
        include_deprecated: 是否包含已弃用声明
        traverse_strategy: 遍历策略 ("bfs"/"shortest_path"/"confidence_weighted")
    """

    entity_focus: str = Field(..., description="中心实体锚点 ID")
    max_depth: int = Field(default=2, ge=1, le=10, description="遍历深度 (最大跳数)")
    max_entities: int = Field(default=50, ge=1, le=500, description="子图大小上限")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="关系置信度过滤阈值")
    min_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="最低质量分数")
    include_deprecated: bool = Field(default=False, description="是否包含已弃用声明")
    traverse_strategy: str = Field(default="bfs", description="遍历策略")

    @field_validator("traverse_strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        allowed = {"bfs", "shortest_path", "confidence_weighted"}
        if v not in allowed:
            raise ValueError(f"traverse_strategy 必须是 {allowed} 之一")
        return v


# ============================================================
# 结构化查询 (借鉴 SPARQL + Milvus 标量过滤)
# ============================================================


class QueryCondition(BaseModel):
    """单个查询条件 (借鉴 SPARQL FILTER + SHACL 约束).

    Attributes:
        field: 查询字段名 (如 "entity_type", "domain", "name")
        operator: 查询操作符
        value: 查询值
        negate: 是否取反
    """

    field: str = Field(..., description="查询字段名")
    operator: QueryOperator = Field(default=QueryOperator.EQ, description="查询操作符")
    value: Any = Field(..., description="查询值")
    negate: bool = Field(default=False, description="是否取反")

    def matches(self, target: Any) -> bool:
        """检查目标值是否满足此条件."""
        result: bool
        if self.operator == QueryOperator.EQ:
            result = target == self.value
        elif self.operator == QueryOperator.NE:
            result = target != self.value
        elif self.operator == QueryOperator.GT:
            try:
                result = float(target) > float(self.value)
            except (TypeError, ValueError):
                result = False
        elif self.operator == QueryOperator.GTE:
            try:
                result = float(target) >= float(self.value)
            except (TypeError, ValueError):
                result = False
        elif self.operator == QueryOperator.LT:
            try:
                result = float(target) < float(self.value)
            except (TypeError, ValueError):
                result = False
        elif self.operator == QueryOperator.LTE:
            try:
                result = float(target) <= float(self.value)
            except (TypeError, ValueError):
                result = False
        elif self.operator == QueryOperator.CONTAINS:
            result = str(self.value) in str(target)
        elif self.operator == QueryOperator.STARTS_WITH:
            result = str(target).startswith(str(self.value))
        elif self.operator == QueryOperator.ENDS_WITH:
            result = str(target).endswith(str(self.value))
        elif self.operator == QueryOperator.IN:
            result = target in (self.value if isinstance(self.value, (list, set, tuple)) else [self.value])
        elif self.operator == QueryOperator.REGEX:
            import re
            try:
                result = bool(re.search(str(self.value), str(target)))
            except re.error:
                result = False
        else:
            result = False
        return (not result) if self.negate else result


class KnowledgeQuery(BaseModel):
    """结构化知识查询 (借鉴 SPARQL 图查询 + GraphQL 查询模型).

    支持多条件组合查询、图遍历和聚合。

    Attributes:
        query_id: 查询唯一标识
        domain: 目标领域
        conditions: 查询条件列表 (AND 组合)
        hop_conditions: 多跳查询条件 (图遍历)
        max_hops: 最大跳数
        limit: 返回结果上限
        offset: 分页偏移
        sort_by: 排序字段
        sort_desc: 是否降序排序
        include_graph: 是否包含图结构 (子图)
        timestamp_filter: 时间戳过滤 (只返回该时间点有效的知识)
    """

    query_id: str = Field(default_factory=lambda: f"q-{uuid.uuid4().hex[:12]}")
    domain: str = Field(default="general", description="目标领域")
    conditions: list[QueryCondition] = Field(default_factory=list, description="查询条件列表 (AND)")
    max_hops: int = Field(default=0, ge=0, description="最大跳数 (0=不遍历)")
    limit: int = Field(default=100, ge=1, le=10000, description="返回结果上限")
    offset: int = Field(default=0, ge=0, description="分页偏移")
    sort_by: str = Field(default="", description="排序字段")
    sort_desc: bool = Field(default=False, description="是否降序排序")
    include_graph: bool = Field(default=False, description="是否包含图结构")
    timestamp_filter: float = Field(default=0.0, description="时间戳过滤 (0=不过滤)")

    def has_conditions(self) -> bool:
        """是否有查询条件."""
        return len(self.conditions) > 0

    def has_traversal(self) -> bool:
        """是否有图遍历."""
        return self.max_hops > 0

    def has_temporal_filter(self) -> bool:
        """是否有时间过滤."""
        return self.timestamp_filter > 0.0


# ============================================================
# 版本管理 (借鉴 ConVer-G + DBpedia-TKG)
# ============================================================


class ChangeRecord(BaseModel):
    """单条变更记录 (借鉴 DBpedia-TKG 三元组级变更追踪).

    Attributes:
        change_type: 变更类型 ("add"/"modify"/"delete")
        entity_id: 变更的实体 ID
        field_path: 变更的字段路径 (如 "properties.formula")
        old_value: 旧值
        new_value: 新值
        changed_at: 变更时间戳
        changed_by: 变更者 (agent_id 或 user_id)
        reason: 变更原因
    """

    change_type: str = Field(..., description="变更类型 (add/modify/delete)")
    entity_id: str = Field(..., description="变更的实体 ID")
    field_path: str = Field(..., description="变更的字段路径")
    old_value: Any = Field(default=None, description="旧值")
    new_value: Any = Field(default=None, description="新值")
    changed_at: float = Field(default_factory=time.time, description="变更时间戳")
    changed_by: str = Field(default="system", description="变更者")
    reason: str = Field(default="", description="变更原因")

    @field_validator("change_type")
    @classmethod
    def _validate_change_type(cls, v: str) -> str:
        allowed = {"add", "modify", "delete"}
        if v not in allowed:
            raise ValueError(f"change_type 必须是 {allowed} 之一")
        return v

    def is_add(self) -> bool:
        return self.change_type == "add"

    def is_modify(self) -> bool:
        return self.change_type == "modify"

    def is_delete(self) -> bool:
        return self.change_type == "delete"


class KnowledgeVersion(BaseModel):
    """知识版本记录 (借鉴 ConVer-G 并发版本管理 + DBpedia 快照).

    记录知识实体的版本历史，支持变更集追踪和时间旅行查询。

    Attributes:
        version_id: 版本唯一标识
        entity_id: 关联的实体 ID
        revision_number: 修订号
        parent_version_id: 父版本 ID (变更链)
        changeset: 变更记录列表
        snapshot: 实体快照 (完整序列化)
        valid_from: 此版本生效时间
        valid_until: 此版本失效时间 (None=当前版本)
        created_at: 版本创建时间
        created_by: 创建者
        version_note: 版本说明
    """

    version_id: str = Field(default_factory=lambda: f"v-{uuid.uuid4().hex[:12]}")
    entity_id: str = Field(..., description="关联的实体 ID")
    revision_number: int = Field(default=1, ge=1, description="修订号")
    parent_version_id: str = Field(default="", description="父版本 ID")
    changeset: list[ChangeRecord] = Field(default_factory=list, description="变更记录列表")
    snapshot: dict[str, Any] = Field(default_factory=dict, description="实体快照")
    valid_from: float = Field(default_factory=time.time, description="此版本生效时间")
    valid_until: float = Field(default=0.0, description="此版本失效时间 (0=当前版本)")
    created_at: float = Field(default_factory=time.time, description="版本创建时间")
    created_by: str = Field(default="system", description="创建者")
    version_note: str = Field(default="", description="版本说明")

    def is_current(self) -> bool:
        """是否为当前版本."""
        return self.valid_until == 0.0

    def is_valid_at(self, timestamp: float) -> bool:
        """在指定时间点是否有效."""
        if timestamp < self.valid_from:
            return False
        if self.valid_until > 0.0 and timestamp >= self.valid_until:
            return False
        return True

    def change_count(self) -> int:
        """变更数量."""
        return len(self.changeset)

    def has_parent(self) -> bool:
        """是否有父版本."""
        return bool(self.parent_version_id)


# ============================================================
# 证据记录 (借鉴 ProVe + 证据分层)
# ============================================================


class EvidenceRecord(BaseModel):
    """知识证据记录 (借鉴 ProVe 自动溯源验证 + 证据分层体系).

    记录支持某条知识声明的证据信息，用于质量评估和可验证性。

    Attributes:
        evidence_id: 证据唯一标识
        entity_id: 关联的实体 ID
        triple_id: 关联的三元组 ID (可选)
        source_type: 证据来源类型
        source_reference: 来源引用 (DOI/URL/文档ID)
        source_content: 来源原文片段
        confidence: 证据置信度
        verified_by: 验证方式
        verified_at: 验证时间
        verifier: 验证者
    """

    evidence_id: str = Field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:12]}")
    entity_id: str = Field(..., description="关联的实体 ID")
    triple_id: str = Field(default="", description="关联的三元组 ID")
    source_type: str = Field(default="document", description="证据来源类型")
    source_reference: str = Field(default="", description="来源引用 (DOI/URL/文档ID)")
    source_content: str = Field(default="", description="来源原文片段")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="证据置信度")
    verified_by: str = Field(default="automated", description="验证方式")
    verified_at: float = Field(default=0.0, description="验证时间 (0=未验证)")
    verifier: str = Field(default="system", description="验证者")

    def is_verified(self) -> bool:
        """是否已验证."""
        return self.verified_at > 0.0

    def is_strong(self) -> bool:
        """是否为强证据 (置信度 >= 0.8)."""
        return self.confidence >= 0.8


# ============================================================
# 知识冲突 (借鉴 MACR + Detect-Then-Resolve)
# ============================================================


class KnowledgeConflict(BaseModel):
    """知识冲突记录 (借鉴 MACR 多智能体冲突解决 + Detect-Then-Resolve).

    当不同来源的知识声明相互矛盾时，创建冲突记录并执行解决策略。

    Attributes:
        conflict_id: 冲突唯一标识
        conflict_type: 冲突类型
        entity_id: 冲突涉及的实体 ID
        field_path: 冲突字段路径
        conflicting_values: 冲突值列表 (含来源信息)
        detection_method: 冲突检测方法
        resolution_strategy: 解决策略
        resolved_value: 解决后的值
        resolved_claim_id: 被采纳的声明 ID
        resolution_explanation: 解决说明 (可解释性)
        status: 冲突状态
        detected_at: 检测时间
        resolved_at: 解决时间
        resolved_by: 解决者
    """

    conflict_id: str = Field(default_factory=lambda: f"cf-{uuid.uuid4().hex[:12]}")
    conflict_type: ConflictType = Field(..., description="冲突类型")
    entity_id: str = Field(..., description="冲突涉及的实体 ID")
    field_path: str = Field(default="", description="冲突字段路径")
    conflicting_values: list[dict[str, Any]] = Field(
        default_factory=list,
        description="冲突值列表 (含来源信息)",
    )
    detection_method: str = Field(default="manual", description="冲突检测方法")
    resolution_strategy: ConflictResolutionStrategy = Field(
        default=ConflictResolutionStrategy.KEEP_BOTH,
        description="解决策略",
    )
    resolved_value: Any = Field(default=None, description="解决后的值")
    resolved_claim_id: str = Field(default="", description="被采纳的声明 ID")
    resolution_explanation: str = Field(default="", description="解决说明")
    status: str = Field(default="detected", description="冲突状态 (detected/resolved/ignored)")
    detected_at: float = Field(default_factory=time.time, description="检测时间")
    resolved_at: float = Field(default=0.0, description="解决时间 (0=未解决)")
    resolved_by: str = Field(default="", description="解决者")

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        allowed = {"detected", "resolved", "ignored"}
        if v not in allowed:
            raise ValueError(f"status 必须是 {allowed} 之一")
        return v

    def is_resolved(self) -> bool:
        """是否已解决."""
        return self.status == "resolved"

    def is_ignored(self) -> bool:
        """是否已忽略."""
        return self.status == "ignored"

    def needs_manual_review(self) -> bool:
        """是否需要人工审核."""
        return self.resolution_strategy == ConflictResolutionStrategy.MANUAL_REVIEW and not self.is_resolved()

    def resolve(
        self,
        value: Any,
        claim_id: str = "",
        explanation: str = "",
        resolved_by: str = "system",
    ) -> None:
        """解决冲突."""
        self.resolved_value = value
        self.resolved_claim_id = claim_id
        self.resolution_explanation = explanation
        self.resolved_by = resolved_by
        self.status = "resolved"
        self.resolved_at = time.time()

    def ignore(self, reason: str = "", by: str = "system") -> None:
        """忽略冲突."""
        self.resolution_explanation = reason
        self.resolved_by = by
        self.status = "ignored"
        self.resolved_at = time.time()

    def conflicting_value_count(self) -> int:
        """冲突值数量."""
        return len(self.conflicting_values)


# ============================================================
# 知识图谱容器 (借鉴 Neo4j 属性图 + Microsoft Graph)
# ============================================================


class KnowledgeGraph(BaseModel):
    """知识图谱容器 (借鉴 Neo4j 属性图模型 + GraphRAG 子图管理).

    管理一组知识实体及其关系三元组，提供图级别的统计和管理能力。

    Attributes:
        graph_id: 图谱唯一标识
        domain: 所属领域
        name: 图谱名称
        description: 图谱描述
        entities: 实体字典 {entity_id: KnowledgeEntity}
        triples: 三元组列表 (跨实体关系)
        versions: 版本历史
        conflicts: 冲突记录
        created_at: 创建时间
        updated_at: 最后更新时间
        metadata: 扩展元数据
    """

    graph_id: str = Field(default_factory=lambda: f"kg-{uuid.uuid4().hex[:12]}")
    domain: str = Field(default="general", description="所属领域")
    name: str = Field(default="", description="图谱名称")
    description: str = Field(default="", description="图谱描述")
    entities: dict[str, KnowledgeEntity] = Field(default_factory=dict, description="实体字典")
    triples: list[KnowledgeTriple] = Field(default_factory=list, description="跨实体三元组")
    versions: list[KnowledgeVersion] = Field(default_factory=list, description="版本历史")
    conflicts: list[KnowledgeConflict] = Field(default_factory=list, description="冲突记录")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="最后更新时间")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    def touch(self) -> None:
        """更新 updated_at 时间戳."""
        self.updated_at = time.time()

    def add_entity(self, entity: KnowledgeEntity) -> None:
        """添加实体."""
        self.entities[entity.entity_id] = entity
        self.touch()

    def get_entity(self, entity_id: str) -> KnowledgeEntity | None:
        """获取实体."""
        return self.entities.get(entity_id)

    def remove_entity(self, entity_id: str) -> bool:
        """移除实体."""
        if entity_id in self.entities:
            del self.entities[entity_id]
            self.touch()
            return True
        return False

    def add_triple(self, triple: KnowledgeTriple) -> None:
        """添加跨实体三元组."""
        self.triples.append(triple)
        self.touch()

    def entity_count(self) -> int:
        """实体数量."""
        return len(self.entities)

    def triple_count(self) -> int:
        """三元组总数 (含实体内部)."""
        internal = sum(len(e.triples) for e in self.entities.values())
        return internal + len(self.triples)

    def active_entity_count(self) -> int:
        """活跃实体数量."""
        return sum(1 for e in self.entities.values() if e.is_active())

    def get_entities_by_type(self, entity_type: EntityType) -> list[KnowledgeEntity]:
        """按类型获取实体."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_entities_by_domain(self, domain: str) -> list[KnowledgeEntity]:
        """按领域获取实体."""
        return [e for e in self.entities.values() if e.domain == domain]

    def find_entity_by_name(self, name: str) -> KnowledgeEntity | None:
        """按名称或别名查找实体."""
        for e in self.entities.values():
            if e.match_name_or_alias(name):
                return e
        return None

    def find_entities_by_tag(self, tag: str) -> list[KnowledgeEntity]:
        """按标签查找实体."""
        return [e for e in self.entities.values() if e.has_tag(tag)]

    def get_stats(self) -> KnowledgeBaseStats:
        """获取图谱统计."""
        entities_by_type: dict[str, int] = {}
        for e in self.entities.values():
            key = e.entity_type.value
            entities_by_type[key] = entities_by_type.get(key, 0) + 1

        quality_scores = [
            e.quality.overall() for e in self.entities.values() if e.quality is not None
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return KnowledgeBaseStats(
            total_entities=self.entity_count(),
            total_triples=self.triple_count(),
            total_sources=len({e.source.source_id for e in self.entities.values() if e.source}),
            entities_by_type=entities_by_type,
            avg_quality=round(avg_quality, 4),
            last_updated=self.updated_at,
        )

    def unresolved_conflict_count(self) -> int:
        """未解决冲突数量."""
        return sum(1 for c in self.conflicts if not c.is_resolved())

    def is_empty(self) -> bool:
        """图谱是否为空."""
        return self.entity_count() == 0 and len(self.triples) == 0


__all__ = [
    # 枚举
    "ContentModality",
    "EntityType",
    "RelationType",
    "SourceTier",
    "QualityDimension",
    "AccessLevel",
    "ChunkingStrategy",
    "StatementRank",
    "ChunkRelationshipType",
    "ProvenanceRole",
    "KnowledgeStatus",
    "ConflictType",
    "ConflictResolutionStrategy",
    "QueryOperator",
    "PropertyDataType",
    "InferenceRuleType",
    "VerificationStatus",
    # 模型
    "KnowledgeSource",
    "KnowledgeQualifier",
    "KnowledgeTriple",
    "KnowledgeEntity",
    "ChunkRelationship",
    "DocumentChunk",
    "QualityScore",
    "ProvenanceInfo",
    "EmbeddingVector",
    "RetrievalResult",
    "KnowledgeBaseStats",
    "IngestResult",
    "RetrievalFilter",
    "SubgraphConfig",
    "QueryCondition",
    "KnowledgeQuery",
    "ChangeRecord",
    "KnowledgeVersion",
    "EvidenceRecord",
    "KnowledgeConflict",
    "KnowledgeGraph",
]
