"""L3 领域知识层.

提供领域知识的核心数据模型、本体定义、异常体系、
索引引擎、存储引擎、检索引擎、持久化层、缓存层、重排器、
意图路由、事实校验、跨库对齐、知识连接器、知识摄入管道和层间接口模型。

融合 W3C OWL/RDF、Wikidata、Schema.org、LlamaIndex、PROV-O、
Dublin Core、DBpedia 质量框架、RAG-Anything 多模态、
GraphRAG 子图提取、MACR 冲突解决、ConVer-G 版本管理、
ProVe 证据验证、SHACL 约束验证、Redis RDB/AOF、SQLite WAL、
Caffeine W-TinyLFU、Elasticsearch request cache、Cohere Rerank、
HNSW (Malkov & Yashunin 2016)、LangChain Router、LlamaIndex RouterQueryEngine、
FActScore、ProVe、SAFE、GraphRAG 社区感知检索、RRF 融合、
Wikidata 实体对齐、Haystack Pipeline 等世界先进方案。

子模块:
    models:              核心数据模型 (实体/三元组/切片/质量/溯源/向量/查询/版本/冲突/图谱)
    ontology:            领域本体定义 (化学/材料/教育/通用 + 注册中心 + 推理引擎)
    exceptions:          异常体系 (-32400 ~ -32416)
    index:               索引引擎 (HashIndex/TypeIndex/InvertedIndex/VectorIndex/NameIndex)
    hnsw_index:          HNSW 向量索引 (近似最近邻搜索)
    store:               存储引擎 (EntityStore/TripleStore/ChunkStore/KnowledgeStore)
    retrieval:           检索引擎 (VectorRetriever/KeywordRetriever/GraphRetriever/HybridRetriever)
    persistence:         持久化层 (PersistenceManager/TransactionManager)
    cache:               缓存层 (LRUCache/QueryCache/CachedKnowledgeStore)
    reranker:            重排器 (MMR/MetadataBoost/QualityBoost/RecencyBoost/GraphCentrality/Composite)
    intent_router:       意图驱动路由 (IntentType/EntityExtractor/IntentClassifier/IntentRouter)
    fact_check:          事实校验 (StandardValue/StandardValueStore/AssertionExtractor/FactChecker)
    cross_db:            跨库对齐融合 (CrossDBAligner/QualityWeightedFuser/AlignmentResult)
    connector:           知识连接器 (KnowledgeConnector/ConnectorRegistry/CircuitBreaker)
    ingestion:           知识摄入管道 (ChunkingEngine/ClassificationEngine/IngestionPipeline)
    api_models:          层间接口模型 (KnowledgeHit/LearnerProfile/ProvenanceMetadata/MCPToolDescriptor)
    query_rewriter:      查询重写引擎 (QueryRewriter/RewriteStrategy/RewrittenQuery, 融合 MultiQueryRetriever+SubQuestion+HyDE)
    embedding:           嵌入管理器 (EmbeddingManager/EmbeddingCache/EmbeddingBackend/EmbeddingResult, LRU+TTL 缓存)
    metrics:             指标监控 (MetricsCollector/Counter/Histogram/Timer, 融合 Prometheus+LangSmith)
    community:           社区检测 (CommunityDetector/Community/CommunityDetectionResult, 融合 GraphRAG+Leiden)
    response_synthesizer: 响应合成器 (ResponseSynthesizer, 融合 LlamaIndex ResponseSynthesizer+GraphRAG map-reduce)
    access_control:      访问控制 (AccessControlManager/AccessControlledStore, RBAC 引擎, 融合 Neo4j RBAC)
    audit_trail:         审计轨迹 (AuditTrail, 不可变 append-only 日志, 融合 Neo4j 事务日志+Wikidata 编辑历史)
    graph_reasoner:      图推理器 (GraphReasoner, Dijkstra+Yen+前向链式+链接预测, 融合 Neo4j Cypher+GraphRAG)
    graph_reasoner_v2:   增强图推理器 (TransE嵌入+后向链式+置信度加权遍历+子图推理)
    kg_builder:          知识图谱构建引擎 (实体/关系抽取+实体消解+增量构建, 融合 REBEL+SpaCy+OpenIE)
    graphrag_retriever:  GraphRAG双通道检索 (局部搜索+全局搜索+RRF融合, 融合 Microsoft GraphRAG+OMD-GraphRAG)
    neo4j_adapter:       Neo4j图数据库适配器 (Cypher构建+实体映射+批量导入, neo4j可选依赖)
    schema_evolution:    模式演进 (SchemaEvolutionManager, 版本追踪+迁移计划, 融合 Neo4j schema 迁移+Django migrations)
    kb_manager:          知识库管理器 (KnowledgeBaseManager, 生命周期+备份+GC, 融合 Neo4j DBMS+ES ILM)
    quality_manager:     知识质量管理与评估 (QualityManager, 六维评估+冲突消解+溯源追踪+监控仪表板,
                          融合 OQuaRE-KG+ISO 25012+MACR+CRDL+PROV-O+Great Expectations)

Usage::

    from dy3_polaris.l3 import (
        KnowledgeEntity, EntityType, KnowledgeTriple,
        DocumentChunk, QualityScore, ProvenanceInfo,
        DomainOntology, OntologyRegistry,
        KnowledgeGraph, KnowledgeConflict, KnowledgeVersion,
        KnowledgeStore, RetrievalEngine,
        PersistenceManager, TransactionManager,
        QueryCache, CachedKnowledgeStore,
        HNSWIndex, MMRReranker, CompositeReranker,
        IntentRouter, FactChecker, CrossDBAligner,
        IngestionPipeline, KnowledgeHit, LearnerProfile,
    )

    # 创建知识实体
    entity = KnowledgeEntity(
        entity_type=EntityType.CHEMICAL_COMPOUND,
        name="水",
        identifiers={"cas": "7732-18-5"},
        properties={"formula": "H2O", "molecular_weight": 18.015},
    )

    # 使用本体验证
    registry = OntologyRegistry()
    violations = registry.validate_full(
        "chemistry", EntityType.CHEMICAL_COMPOUND, entity.properties
    )

    # 知识存储
    store = KnowledgeStore()
    store.add_entity(entity)

    # 知识检索
    engine = RetrievalEngine(store)
    result = engine.keyword_search("水")

    # 持久化
    pm = PersistenceManager(store, "/data/kb")
    pm.save_snapshot()

    # 事务
    tx_mgr = TransactionManager(store)
    with tx_mgr.begin() as tx:
        tx.add_entity(entity)
        tx.commit()

    # 缓存
    cached = CachedKnowledgeStore(store)
    entity = cached.get_entity(entity.entity_id)

    # HNSW 索引
    hnsw = HNSWIndex(dim=128, M=16, ef_construction=200)
    hnsw.add("vec1", [0.1] * 128)
    results = hnsw.search([0.1] * 128, top_k=10)

    # 意图路由检索
    router = IntentRouter(store)
    routed = router.route("Dy3+离子的发射波长是多少nm?")

    # 事实校验
    checker = FactChecker()
    report = checker.check("Dy3+离子的发射波长为580nm")

    # 跨库对齐融合
    aligner = CrossDBAligner()
    aligner.add_source("vector", results_list, scores_list)
    fused = aligner.fuse(query="Dy3+跃迁波长")

    # 知识摄入
    pipeline = IngestionPipeline(store)
    ingest_result = pipeline.ingest("稀土离子能级...", "doc-001")
"""

from .cache import (
    CacheStats,
    CachedKnowledgeStore,
    LRUCache,
    QueryCache,
)
from .exceptions import (
    ChunkingError,
    ConflictError,
    DuplicateEntityError,
    EmbeddingError,
    EntityMergeError,
    EntityNotFoundError,
    InferenceError,
    IngestError,
    L3Error,
    OntologyValidationError,
    ProvenanceError,
    QualityAssessmentError,
    QueryError,
    RetrievalError,
    VersionConflictError,
)
from .hnsw_index import HNSWIndex
from .index import (
    HashIndex,
    InvertedIndex,
    NameIndex,
    TypeIndex,
    VectorIndex,
)
from .models import (
    AccessLevel,
    ChangeRecord,
    ChunkRelationship,
    ChunkRelationshipType,
    ChunkingStrategy,
    ConflictResolutionStrategy,
    ConflictType,
    ContentModality,
    DocumentChunk,
    EmbeddingVector,
    EntityType,
    EvidenceRecord,
    InferenceRuleType,
    IngestResult,
    KnowledgeBaseStats,
    KnowledgeConflict,
    KnowledgeEntity,
    KnowledgeGraph,
    KnowledgeQualifier,
    KnowledgeQuery,
    KnowledgeSource,
    KnowledgeStatus,
    KnowledgeTriple,
    KnowledgeVersion,
    ProvenanceInfo,
    ProvenanceRole,
    PropertyDataType,
    QueryCondition,
    QueryOperator,
    QualityDimension,
    QualityScore,
    RelationType,
    RetrievalFilter,
    RetrievalResult,
    SourceTier,
    StatementRank,
    SubgraphConfig,
    VerificationStatus,
)
from .ontology import (
    DomainOntology,
    DomainType,
    OntologyAxiom,
    OntologyClass,
    OntologyMapping,
    OntologyProperty,
    OntologyRegistry,
    OntologyRelation,
    OntologyRule,
)
from .persistence import (
    PersistenceManager,
    Transaction,
    TransactionManager,
    TransactionState,
)
from .reranker import (
    BaseReranker,
    CompositeReranker,
    GraphCentralityReranker,
    MMRReranker,
    MetadataBoostReranker,
    QualityBoostReranker,
    RecencyBoostReranker,
    RerankStrategy,
)
from .retrieval import (
    BaseRetriever,
    GraphRetriever,
    HybridRetriever,
    KeywordRetriever,
    RetrievalEngine,
    VectorRetriever,
)
from .store import (
    ChunkStore,
    EntityStore,
    KnowledgeStore,
    TripleStore,
)
from .intent_router import (
    EntityExtractor,
    ExtractedEntity,
    IntentClassifier,
    IntentResult,
    IntentRouter,
    IntentType,
    RoutedResult,
)
from .fact_check import (
    AssertionExtractor,
    CheckResult,
    CheckStatus,
    FactCheckReport,
    FactChecker,
    NumericAssertion,
    StandardValue,
    StandardValueStore,
    ToleranceType,
)
from .cross_db import (
    AlignmentResult,
    AlignedItem,
    CrossDBAligner,
    FusionConfig,
    QualityWeightedFuser,
    SourceType,
)
from .connector import (
    CircuitBreaker,
    CircuitState,
    ConnectorConfig,
    ConnectorHealth,
    ConnectorProtocol,
    ConnectorRegistry,
    ConnectorResponse,
    ConnectorStatus,
    ConnectorTier,
    KnowledgeConnector,
)
from .ingestion import (
    AuthorityTier,
    ChunkingConfig,
    ChunkingEngine,
    ChunkMetadata,
    ClassificationEngine,
    ClassificationResult,
    ContentType,
    IngestionPipeline,
    IngestionResult,
    KnowledgeDomain,
    KnowledgeLevel,
)
from .api_models import (
    BloomLevel,
    FactCheckSummary,
    KPMastery,
    KnowledgeHit,
    KnowledgeRetrievalResult,
    LearnerProfile,
    LearningStyle,
    MCPToolCall,
    MCPToolDescriptor,
    MCPToolResult,
    ProvenanceEvent,
    ProvenanceEventType,
    ProvenanceMetadata,
    apply_learner_filter,
    to_knowledge_hit,
    to_provenance_event,
    to_retrieval_result,
)
from .query_rewriter import (
    QueryRewriter,
    RewriteStrategy,
    RewrittenQuery,
)
from .context_builder import (
    ContextBudget,
    ContextBuilder,
    CoreferenceResolver,
    DialogTurn,
    HistoryCompressStrategy,
    HistoryCompressor,
    LearnerContextAdapter,
    LLMClassifier,
    QueryContext,
    RetrievalNeedAssessor,
    SchemaContextInjector,
)
from .embedding import (
    EmbeddingBackend,
    EmbeddingCache,
    EmbeddingManager,
    EmbeddingResult,
)
from .metrics import (
    Counter,
    Histogram,
    MetricSample,
    MetricType,
    MetricsCollector,
    Timer,
)
from .community import (
    Community,
    CommunityAlgorithm,
    CommunityDetectionResult,
    CommunityDetector,
    CommunityHierarchy,
)
from .access_control import (
    AccessControlledStore,
    AccessControlManager,
    AccessDecision,
    AccessDeniedError,
    AccessPolicy,
    AccessRequest,
    AccessResult,
    Permission,
    ResourceType,
    Role,
    User,
)
from .response_synthesizer import (
    Citation,
    EvidencePiece,
    EvidenceType,
    QueryType,
    ResponseSynthesizer,
    SynthesisConfig,
    SynthesisError,
    SynthesisMode,
    SynthesizedResponse,
)
from .audit_trail import (
    AuditEntry,
    AuditError,
    AuditQuery,
    AuditStats,
    AuditTrail,
    ChangeDiff,
    OperationType,
)
from .audit_trail import ResourceType as AuditResourceType
from .graph_reasoner import (
    GraphReasoner,
    InferenceRule,
    PathResult,
    ReasoningError,
    ReasoningMode,
    ReasoningResult,
)
from .schema_evolution import (
    ChangeType,
    CompatibilityLevel,
    MigrationPlan,
    MigrationStep,
    SchemaChange,
    SchemaDiff,
    SchemaEvolutionError,
    SchemaEvolutionManager,
    SchemaVersion,
)
from .kb_manager import (
    KBConfig,
    KBManagerError,
    KBLifecycleState,
    KBSnapshot,
    HealthStatus,
    KnowledgeBaseManager,
    RetentionPolicy,
)
from .data_source_adapter import (
    AdapterCapability,
    AdapterError,
    AdapterSpec,
    AuthenticationError,
    DataAdapterBase,
    DataAdapterRegistry,
    DataSourceSchema,
    DataSourceType,
    DefaultRecoverer,
    DiscoverResult,
    FieldMapping,
    LifecyclePhase,
    ReadResult,
    RecoveryAction,
    RecoveryExhaustedError,
    Recoverer,
    SchemaDiscoveryError,
    SchemaField,
    SchemaMapper,
    SyncCheckpoint,
    SyncCoordinator,
    SyncError,
    SyncMode,
)
from .adapter_bases import (
    DatabaseAdapter,
    FileAdapter,
    GraphQLAdapter,
    MCPAdapter,
    RESTAdapter,
)
from .adapters_tier1_public import (
    ArxivAdapter,
    ChemSpiderAdapter,
    CrossRefAdapter,
    DOAJAdapter,
    NISTWebBookAdapter,
    OpenAlexAdapter,
    PubChemAdapter,
    SemanticScholarAdapter,
    UniProtAdapter,
    WikipediaAdapter,
)
from .adapters_tier2_industry import (
    CASAdapter,
    EngineeringVillageAdapter,
    GooglePatentsAdapter,
    ReaxysAdapter,
    SciFinderAdapter,
    WebOfScienceAdapter,
)
from .adapters_tier3_private import (
    AcademicAffairsAdapter,
    InternalDocRepositoryAdapter,
    LibraryOPACAdapter,
    LIMSAdapter,
)
from .graph_reasoner_v2 import (
    BackwardChainingReasoner,
    ConfidenceWeightedTraversal,
    Goal,
    LinkPredictionResult,
    SubgraphReasoner,
    SubgraphReasoningResult,
    TrainingReport,
    TransEEmbedder,
    WeightedTraversalResult,
)
from .kg_builder import (
    BatchBuildResult,
    BuildResult,
    EntityCluster,
    KGEntityExtractor,
    KGEntityResolver,
    ExtractionStrategy,
    KGExtractedEntity,
    KGExtractedRelation,
    KnowledgeGraphBuilder,
    RelationExtractor,
    RelationPattern,
    UnionFind,
)
from .graphrag_retriever import (  # noqa: F401
    CommunitySummarizer,
    CommunitySummary,
    ExtractedSubgraph,
    FusionStrategy,
    GlobalSearchResult,
    GraphRAGResult,
    GraphRAGRetriever,
    LocalSearchResult,
    SubgraphExtractor,
    SubgraphStrategy,
    _simplified_pagerank,
)
from .neo4j_adapter import (
    BatchSyncResult,
    CypherQueryBuilder,
    CypherQuery,
    GraphStats,
    Neo4jAdapter,
    Neo4jEntityMapper,
    SubgraphResult,
    SyncResult,
)
from .quality_manager import (
    AccuracyAssessor,
    AssessmentLevel,
    BaseQualityAssessor,
    CompletenessAssessor,
    ConflictDetector,
    ConflictDetectionMethod,
    ConflictResolver,
    ConsistencyAssessor,
    MetricResult,
    ProvenanceTracker,
    ProvenanceVerificationResult,
    QualityAssessmentResult,
    QualityDashboard,
    QualityDashboardData,
    QualityGrade,
    QualityManager,
    RelevancyAssessor,
    TimelinessAssessor,
    TrustworthinessAssessor,
)

__all__ = [
    # 异常 (15 个)
    "L3Error",
    "EntityNotFoundError",
    "DuplicateEntityError",
    "OntologyValidationError",
    "QualityAssessmentError",
    "ProvenanceError",
    "ChunkingError",
    "EmbeddingError",
    "RetrievalError",
    "IngestError",
    "ConflictError",
    "VersionConflictError",
    "QueryError",
    "InferenceError",
    "EntityMergeError",
    # 枚举 (16 个)
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
    # 核心数据模型 (21 个)
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
    # 本体 (9 个)
    "OntologyProperty",
    "OntologyRelation",
    "OntologyClass",
    "OntologyAxiom",
    "OntologyRule",
    "OntologyMapping",
    "DomainType",
    "DomainOntology",
    "OntologyRegistry",
    # 索引引擎 (6 个)
    "HashIndex",
    "TypeIndex",
    "InvertedIndex",
    "VectorIndex",
    "NameIndex",
    "HNSWIndex",
    # 存储引擎 (4 个)
    "EntityStore",
    "TripleStore",
    "ChunkStore",
    "KnowledgeStore",
    # 检索引擎 (6 个)
    "BaseRetriever",
    "VectorRetriever",
    "KeywordRetriever",
    "GraphRetriever",
    "HybridRetriever",
    "RetrievalEngine",
    # 持久化层 (4 个)
    "PersistenceManager",
    "Transaction",
    "TransactionManager",
    "TransactionState",
    # 缓存层 (4 个)
    "CacheStats",
    "LRUCache",
    "QueryCache",
    "CachedKnowledgeStore",
    # 重排器 (8 个)
    "RerankStrategy",
    "BaseReranker",
    "MMRReranker",
    "MetadataBoostReranker",
    "QualityBoostReranker",
    "RecencyBoostReranker",
    "GraphCentralityReranker",
    "CompositeReranker",
    # 意图路由 (7 个)
    "IntentType",
    "ExtractedEntity",
    "IntentResult",
    "EntityExtractor",
    "IntentClassifier",
    "IntentRouter",
    "RoutedResult",
    # 事实校验 (9 个)
    "ToleranceType",
    "CheckStatus",
    "StandardValue",
    "NumericAssertion",
    "CheckResult",
    "FactCheckReport",
    "StandardValueStore",
    "AssertionExtractor",
    "FactChecker",
    # 跨库对齐 (6 个)
    "SourceType",
    "AlignedItem",
    "FusionConfig",
    "AlignmentResult",
    "CrossDBAligner",
    "QualityWeightedFuser",
    # 知识连接器 (10 个)
    "ConnectorTier",
    "ConnectorStatus",
    "ConnectorProtocol",
    "CircuitState",
    "ConnectorConfig",
    "ConnectorHealth",
    "ConnectorResponse",
    "CircuitBreaker",
    "KnowledgeConnector",
    "ConnectorRegistry",
    # 知识摄入 (11 个)
    "KnowledgeDomain",
    "KnowledgeLevel",
    "ContentType",
    "AuthorityTier",
    "ChunkMetadata",
    "ChunkingConfig",
    "ClassificationResult",
    "IngestionResult",
    "ChunkingEngine",
    "ClassificationEngine",
    "IngestionPipeline",
    # 层间接口模型 (17 个)
    "LearningStyle",
    "BloomLevel",
    "KPMastery",
    "LearnerProfile",
    "KnowledgeHit",
    "FactCheckSummary",
    "KnowledgeRetrievalResult",
    "ProvenanceEventType",
    "ProvenanceEvent",
    "ProvenanceMetadata",
    "MCPToolDescriptor",
    "MCPToolCall",
    "MCPToolResult",
    "to_knowledge_hit",
    "to_retrieval_result",
    "to_provenance_event",
    "apply_learner_filter",
    # 查询重写 (3 个)
    "RewriteStrategy",
    "RewrittenQuery",
    "QueryRewriter",
    # 上下文构建 (11 个)
    "ContextBudget",
    "ContextBuilder",
    "CoreferenceResolver",
    "DialogTurn",
    "HistoryCompressStrategy",
    "HistoryCompressor",
    "LearnerContextAdapter",
    "LLMClassifier",
    "QueryContext",
    "RetrievalNeedAssessor",
    "SchemaContextInjector",
    # 嵌入管理 (4 个)
    "EmbeddingBackend",
    "EmbeddingCache",
    "EmbeddingManager",
    "EmbeddingResult",
    # 指标监控 (6 个)
    "MetricType",
    "MetricSample",
    "Counter",
    "Histogram",
    "Timer",
    "MetricsCollector",
    # 社区检测 (5 个)
    "CommunityAlgorithm",
    "Community",
    "CommunityDetectionResult",
    "CommunityDetector",
    "CommunityHierarchy",
    # 访问控制 (10 个)
    "Role",
    "Permission",
    "ResourceType",
    "AccessDecision",
    "AccessDeniedError",
    "User",
    "AccessPolicy",
    "AccessRequest",
    "AccessResult",
    "AccessControlManager",
    "AccessControlledStore",
    # 响应合成 (9 个)
    "SynthesisMode",
    "EvidenceType",
    "QueryType",
    "Citation",
    "EvidencePiece",
    "SynthesizedResponse",
    "SynthesisConfig",
    "SynthesisError",
    "ResponseSynthesizer",
    # 审计轨迹 (8 个)
    "OperationType",
    "AuditResourceType",
    "ChangeDiff",
    "AuditEntry",
    "AuditQuery",
    "AuditStats",
    "AuditError",
    "AuditTrail",
    # 图推理 (6 个)
    "ReasoningMode",
    "PathResult",
    "ReasoningResult",
    "InferenceRule",
    "ReasoningError",
    "GraphReasoner",
    # 模式演进 (9 个)
    "ChangeType",
    "CompatibilityLevel",
    "SchemaChange",
    "SchemaVersion",
    "MigrationStep",
    "MigrationPlan",
    "SchemaDiff",
    "SchemaEvolutionError",
    "SchemaEvolutionManager",
    # 知识库管理 (7 个)
    "KBLifecycleState",
    "RetentionPolicy",
    "HealthStatus",
    "KBSnapshot",
    "KBConfig",
    "KBManagerError",
    "KnowledgeBaseManager",
    # 数据源适配器框架 — 核心组件 (23 个)
    "DataSourceType",
    "SyncMode",
    "AdapterCapability",
    "LifecyclePhase",
    "RecoveryAction",
    "SchemaField",
    "DataSourceSchema",
    "FieldMapping",
    "SchemaMapper",
    "SyncCheckpoint",
    "ReadResult",
    "DiscoverResult",
    "AdapterSpec",
    "Recoverer",
    "DefaultRecoverer",
    "DataAdapterBase",
    "DataAdapterRegistry",
    "SyncCoordinator",
    "AdapterError",
    "AuthenticationError",
    "SchemaDiscoveryError",
    "SyncError",
    "RecoveryExhaustedError",
    # 数据源适配器框架 — 协议基类 (5 个)
    "RESTAdapter",
    "GraphQLAdapter",
    "DatabaseAdapter",
    "FileAdapter",
    "MCPAdapter",
    # 数据源适配器 — Tier-1 公共数据源 (10 个)
    "NISTWebBookAdapter",
    "PubChemAdapter",
    "ArxivAdapter",
    "WikipediaAdapter",
    "OpenAlexAdapter",
    "CrossRefAdapter",
    "DOAJAdapter",
    "UniProtAdapter",
    "ChemSpiderAdapter",
    "SemanticScholarAdapter",
    # 数据源适配器 — Tier-2 行业数据源 (6 个)
    "CASAdapter",
    "WebOfScienceAdapter",
    "SciFinderAdapter",
    "ReaxysAdapter",
    "GooglePatentsAdapter",
    "EngineeringVillageAdapter",
    # 数据源适配器 — Tier-3 校园/私有数据源 (4 个)
    "LibraryOPACAdapter",
    "LIMSAdapter",
    "AcademicAffairsAdapter",
    "InternalDocRepositoryAdapter",
    # 图推理增强 V2 (10 个)
    "TrainingReport",
    "LinkPredictionResult",
    "Goal",
    "WeightedTraversalResult",
    "SubgraphReasoningResult",
    "TransEEmbedder",
    "BackwardChainingReasoner",
    "ConfidenceWeightedTraversal",
    "SubgraphReasoner",
    # 知识图谱构建引擎 (12 个)
    "ExtractionStrategy",
    "KGExtractedEntity",
    "KGExtractedRelation",
    "EntityCluster",
    "BuildResult",
    "BatchBuildResult",
    "RelationPattern",
    "KGEntityExtractor",
    "RelationExtractor",
    "KGEntityResolver",
    "UnionFind",
    "KnowledgeGraphBuilder",
    # GraphRAG 双通道检索 (10 个)
    "SubgraphStrategy",
    "FusionStrategy",
    "ExtractedSubgraph",
    "CommunitySummary",
    "LocalSearchResult",
    "GlobalSearchResult",
    "GraphRAGResult",
    "SubgraphExtractor",
    "CommunitySummarizer",
    "GraphRAGRetriever",
    # Neo4j 图数据库适配器 (7 个)
    "SyncResult",
    "BatchSyncResult",
    "SubgraphResult",
    "GraphStats",
    "CypherQueryBuilder",
    "CypherQuery",
    "Neo4jEntityMapper",
    "Neo4jAdapter",
    # 知识质量管理与评估 (19 个)
    "AssessmentLevel",
    "QualityGrade",
    "ConflictDetectionMethod",
    "ProvenanceVerificationResult",
    "MetricResult",
    "QualityAssessmentResult",
    "QualityDashboardData",
    "BaseQualityAssessor",
    "AccuracyAssessor",
    "ConsistencyAssessor",
    "CompletenessAssessor",
    "TimelinessAssessor",
    "TrustworthinessAssessor",
    "RelevancyAssessor",
    "ConflictDetector",
    "ConflictResolver",
    "ProvenanceTracker",
    "QualityDashboard",
    "QualityManager",
]
