"""L3 领域知识层 — 层间接口模型.

定义 L3 知识层与其他层之间的接口数据结构，基于项目规划文档中的接口规范。

融合世界先进方案的接口设计:
- LlamaIndex QueryEngine output adapter: 检索结果适配器模式
- LangChain output parser: 结构化输出解析
- gRPC protobuf interface design: 跨层接口契约
- OpenAPI schema spec: 工具描述与 schema 定义
- MCP (Model Context Protocol) tool spec: 工具调用与结果协议
- PROV-O provenance model: 溯源事件与元数据

接口分四组:
1. L2 → L3 (学习者画像传入): LearningStyle, BloomLevel, KPMastery, LearnerProfile
2. L3 → L4 (知识检索结果传出): KnowledgeHit, FactCheckSummary, KnowledgeRetrievalResult
3. L3 → CC3 (溯源事件): ProvenanceEventType, ProvenanceEvent, ProvenanceMetadata
4. L3 → L6 (MCP 工具暴露): MCPToolDescriptor, MCPToolCall, MCPToolResult

适配器函数:
- to_knowledge_hit: 检索结果 → KnowledgeHit
- to_retrieval_result: 原始结果 → KnowledgeRetrievalResult (L4 接口格式)
- to_provenance_event: 原始事件 → ProvenanceEvent (CC3 接口格式)
- apply_learner_filter: 根据学习者画像过滤/调整检索结果
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# L2 → L3 接口模型 (L2 个性化层传入 L3)
# ============================================================


class LearningStyle(str, Enum):
    """学习风格 (借鉴 VARK 模型).

    VARK 将学习者偏好分为四类模态:
    - VISUAL: 视觉型 — 偏好图表、图像、思维导图等可视化呈现
    - AUDITORY: 听觉型 — 偏好讲解、音频、讨论等听觉通道
    - READING: 读写型 — 偏好文字、阅读、笔记等文本通道
    - KINESTHETIC: 动觉型 — 偏好实验、操作、模拟等实践通道
    """

    VISUAL = "visual"
    AUDITORY = "auditory"
    READING = "reading"
    KINESTHETIC = "kinesthetic"


class BloomLevel(str, Enum):
    """布鲁姆认知层级 (借鉴 Bloom's Taxonomy 修订版).

    六个由低到高的认知层次:
    - REMEMBER: 记忆 — 提取、识别、回忆相关知识
    - UNDERSTAND: 理解 — 从教学信息中建构意义
    - APPLY: 应用 — 在给定情境中执行或实施程序
    - ANALYZE: 分析 — 分解并确定各部分之间的关系
    - EVALUATE: 评价 — 基于准则和标准做出判断
    - CREATE: 创造 — 将要素组合为新的结构或模式
    """

    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class KPMastery(BaseModel):
    """知识点掌握度 (借鉴 BKT 输出模型).

    描述学习者对单个知识点(KP)的掌握状态，
    由 L2 个性化层通过贝叶斯知识追踪(BKT)计算后传入 L3。

    Attributes:
        kp_id: 知识点唯一标识 (如 "DOM-A-01")
        mastery_prob: 掌握概率 [0.0, 1.0], BKT 后验 P(known)
        attempts: 该知识点的总作答次数
        correct_count: 正确作答次数
        last_attempt_time: 最近一次作答时间戳 (Unix epoch 秒)
    """

    kp_id: str = Field(..., description="知识点唯一标识")
    mastery_prob: float = Field(
        ..., ge=0.0, le=1.0, description="掌握概率 [0.0, 1.0]"
    )
    attempts: int = Field(default=0, ge=0, description="总作答次数")
    correct_count: int = Field(default=0, ge=0, description="正确作答次数")
    last_attempt_time: float = Field(
        default=0.0, description="最近一次作答时间戳 (Unix epoch 秒)"
    )

    def accuracy(self) -> float:
        """计算正确率."""
        if self.attempts == 0:
            return 0.0
        return self.correct_count / self.attempts

    def is_weak(self, threshold: float = 0.5) -> bool:
        """是否为薄弱知识点 (掌握概率低于阈值)."""
        return self.mastery_prob < threshold


class LearnerProfile(BaseModel):
    """学习者画像 (L2 个性化层传入 L3 的核心数据结构).

    聚合学习者的身份、知识掌握状态、偏好和目标信息，
    供 L3 在检索时做个性化过滤与排序。

    Attributes:
        learner_id: 学习者唯一标识
        level: 学习者能力等级 (如 "beginner"/"intermediate"/"advanced")
        kp_mastery: 各知识点掌握状态, 键为 kp_id
        preferred_style: 偏好学习风格
        bloom_target: 布鲁姆认知目标层级
        weak_kps: 薄弱知识点 ID 列表
        interests: 兴趣领域列表 (如 ["材料科学", "发光器件"])
    """

    learner_id: str = Field(..., description="学习者唯一标识")
    level: str = Field(default="beginner", description="学习者能力等级")
    kp_mastery: dict[str, KPMastery] = Field(
        default_factory=dict, description="各知识点掌握状态, 键为 kp_id"
    )
    preferred_style: LearningStyle = Field(
        default=LearningStyle.READING, description="偏好学习风格"
    )
    bloom_target: BloomLevel = Field(
        default=BloomLevel.UNDERSTAND, description="布鲁姆认知目标层级"
    )
    weak_kps: list[str] = Field(
        default_factory=list, description="薄弱知识点 ID 列表"
    )
    interests: list[str] = Field(
        default_factory=list, description="兴趣领域列表"
    )

    def get_mastery(self, kp_id: str) -> KPMastery | None:
        """获取指定知识点的掌握状态."""
        return self.kp_mastery.get(kp_id)

    def is_weak_kp(self, kp_id: str) -> bool:
        """判断指定知识点是否为薄弱项."""
        return kp_id in self.weak_kps

    def weak_kp_count(self) -> int:
        """薄弱知识点数量."""
        return len(self.weak_kps)


# ============================================================
# L3 → L4 接口模型 (L3 传给 L4 决策引擎)
# ============================================================


class KnowledgeHit(BaseModel):
    """知识命中结果 (L3 检索后传给 L4 决策引擎的单条知识命中).

    封装单条检索命中的完整信息，包括内容、相关性分数、
    来源、子维度分数、瓶颈标记和置信度。

    Attributes:
        kp_id: 命中的知识点 ID
        content: 命中的知识内容文本
        score: 综合相关性分数 [0.0, 1.0]
        source: 检索来源类型 ("vector"/"keyword"/"graph"/"hybrid")
        source_doc_id: 来源文档 ID
        source_refs: 来源引用列表 (页码、章节、URI 等)
        sub_scores: 子维度分数 (如 {"visual": 0.8, "difficulty": 0.4})
        is_bottleneck: 是否为知识瓶颈 (阻碍后续学习的关键知识点)
        confidence: 知识置信度 [0.0, 1.0]
    """

    kp_id: str = Field(..., description="命中的知识点 ID")
    content: str = Field(..., min_length=1, description="命中的知识内容文本")
    score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="综合相关性分数 [0.0, 1.0]"
    )
    source: str = Field(default="vector", description="检索来源类型")
    source_doc_id: str = Field(default="", description="来源文档 ID")
    source_refs: list[str] = Field(
        default_factory=list, description="来源引用列表"
    )
    sub_scores: dict[str, float] = Field(
        default_factory=dict,
        description="子维度分数 (如 visual/audio/difficulty 等)",
    )
    is_bottleneck: bool = Field(
        default=False, description="是否为知识瓶颈"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="知识置信度 [0.0, 1.0]"
    )

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """是否高置信度."""
        return self.confidence >= threshold


class FactCheckSummary(BaseModel):
    """事实校验摘要 (L3 事实校验后传给 L4 的汇总结果).

    聚合检索结果中所有可校验项的事实校验统计，
    为 L4 决策提供可信度参考。

    Attributes:
        checked: 校验的条目总数
        passed: 通过校验的条目数
        failed: 未通过校验的条目数
        skipped: 跳过校验的条目数 (无标准值可匹配)
        overall_passed: 整体是否通过 (failed == 0)
        failed_items: 未通过条目的详细信息列表
    """

    checked: int = Field(default=0, ge=0, description="校验的条目总数")
    passed: int = Field(default=0, ge=0, description="通过校验的条目数")
    failed: int = Field(default=0, ge=0, description="未通过校验的条目数")
    skipped: int = Field(default=0, ge=0, description="跳过校验的条目数")
    overall_passed: bool = Field(
        default=True, description="整体是否通过 (failed == 0)"
    )
    failed_items: list[dict[str, Any]] = Field(
        default_factory=list, description="未通过条目的详细信息列表"
    )

    def pass_rate(self) -> float:
        """计算通过率."""
        if self.checked == 0:
            return 1.0
        return self.passed / self.checked


class KnowledgeRetrievalResult(BaseModel):
    """知识检索结果 (L3 传给 L4 决策引擎的完整检索响应).

    封装一次知识检索的完整输出，包括查询信息、命中列表、
    事实校验摘要、延迟、溯源和总数。

    Attributes:
        query_id: 查询唯一标识 (用于全链路追踪)
        query: 原始查询文本
        intent_type: 查询意图类型 ("concept"/"numeric"/"relational"/"composite")
        hits: 命中结果列表
        fact_check: 事实校验摘要 (若无校验则为 None)
        latency_ms: 检索耗时 (毫秒)
        provenance: 溯源元数据 (含来源链、模型版本等)
        total: 总匹配数 (可能大于 hits 长度)
    """

    query_id: str = Field(
        default_factory=lambda: f"q-{uuid.uuid4().hex[:12]}",
        description="查询唯一标识",
    )
    query: str = Field(..., description="原始查询文本")
    intent_type: str = Field(
        default="concept", description="查询意图类型"
    )
    hits: list[KnowledgeHit] = Field(
        default_factory=list, description="命中结果列表"
    )
    fact_check: FactCheckSummary | None = Field(
        default=None, description="事实校验摘要"
    )
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="检索耗时 (毫秒)"
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="溯源元数据"
    )
    total: int = Field(default=0, ge=0, description="总匹配数")

    def is_empty(self) -> bool:
        """结果是否为空."""
        return len(self.hits) == 0

    def top_k(self, k: int) -> list[KnowledgeHit]:
        """获取前 k 个命中 (按 score 降序)."""
        return sorted(self.hits, key=lambda h: h.score, reverse=True)[:k]

    def best_score(self) -> float:
        """最高命中分数."""
        return max((h.score for h in self.hits), default=0.0)


# ============================================================
# L3 → CC3 溯源接口模型 (L3 传给 CC3 Provenance)
# ============================================================


class ProvenanceEventType(str, Enum):
    """溯源事件类型 (借鉴 PROV-O Activity 类型).

    - QUERY: 知识检索事件 — 学习者发起查询
    - INGEST: 知识导入事件 — 新知识入库
    - UPDATE: 知识更新事件 — 已有知识修正
    - FACT_CHECK: 事实校验事件 — 检索结果校验
    - VERSION_RESTORE: 版本恢复事件 — 历史版本回滚
    """

    QUERY = "query"
    INGEST = "ingest"
    UPDATE = "update"
    FACT_CHECK = "fact_check"
    VERSION_RESTORE = "version_restore"


class ProvenanceEvent(BaseModel):
    """溯源事件 (L3 传给 CC3 Provenance 的事件记录).

    记录 L3 知识层的一次完整活动，包含事件类型、时间、
    学习者、查询、检索路径、结果和模型版本等信息，
    供 CC3 构建防篡改溯源链。

    Attributes:
        event_type: 事件类型
        timestamp: 事件时间戳 (Unix epoch 秒)
        learner_id: 学习者 ID (系统操作时为 "system")
        query: 查询文本 (非查询类事件可为空)
        intent: 查询意图 (concept/numeric/relational/composite)
        retrieval_path: 检索路径描述 (如 "vector→rerank→fact_check")
        results: 检索结果快照 (序列化的结果列表)
        latency_ms: 事件处理耗时 (毫秒)
        model_versions: 涉及的模型版本 (如 {"embedding": "v2", "reranker": "v1"})
    """

    event_type: ProvenanceEventType = Field(
        ..., description="事件类型"
    )
    timestamp: float = Field(
        default_factory=time.time, description="事件时间戳 (Unix epoch 秒)"
    )
    learner_id: str = Field(
        default="system", description="学习者 ID"
    )
    query: str = Field(default="", description="查询文本")
    intent: str = Field(default="", description="查询意图")
    retrieval_path: str = Field(
        default="", description="检索路径描述"
    )
    results: list[dict[str, Any]] = Field(
        default_factory=list, description="检索结果快照"
    )
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="事件处理耗时 (毫秒)"
    )
    model_versions: dict[str, str] = Field(
        default_factory=dict, description="涉及的模型版本"
    )

    def is_system_event(self) -> bool:
        """是否为系统事件 (非学习者发起)."""
        return self.learner_id == "system"


class ProvenanceMetadata(BaseModel):
    """溯源元数据 (L3 传给 CC3 的单条知识来源元信息).

    封装知识来源的文献级元数据，支持 DOI 引用、
    标准参考、教材页码等溯源信息。

    Attributes:
        source_doc_id: 来源文档 ID
        doi: 数字对象标识符 (DOI, 可为 None)
        standard_ref: 标准参考引用 (如 "GB/T 1234-2020", 可为 None)
        textbook_page: 教材页码 (可为 None)
        confidence: 来源置信度 [0.0, 1.0]
        fact_checked: 是否已通过事实校验
    """

    source_doc_id: str = Field(..., description="来源文档 ID")
    doi: str | None = Field(default=None, description="数字对象标识符 (DOI)")
    standard_ref: str | None = Field(
        default=None, description="标准参考引用"
    )
    textbook_page: int | None = Field(
        default=None, ge=0, description="教材页码"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="来源置信度 [0.0, 1.0]"
    )
    fact_checked: bool = Field(
        default=False, description="是否已通过事实校验"
    )

    def has_external_ref(self) -> bool:
        """是否有外部参考 (DOI 或标准引用)."""
        return self.doi is not None or self.standard_ref is not None


# ============================================================
# L3 → L6 MCP 接口模型 (L3 通过 MCP 暴露给 L6)
# ============================================================


class MCPToolDescriptor(BaseModel):
    """MCP 工具描述符 (L3 暴露给 L6 的工具元信息).

    借鉴 MCP (Model Context Protocol) Tool 规范和 OpenAPI schema spec,
    描述一个 L3 知识检索工具的名称、功能、输入输出 schema 和标签。

    Attributes:
        name: 工具名称 (如 "knowledge_search")
        description: 工具功能描述
        input_schema: 输入参数 JSON Schema
        output_schema: 输出结果 JSON Schema
        tags: 标签列表 (如 ["L3", "retrieval", "vector"])
    """

    name: str = Field(..., min_length=1, description="工具名称")
    description: str = Field(default="", description="工具功能描述")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="输入参数 JSON Schema"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="输出结果 JSON Schema"
    )
    tags: list[str] = Field(
        default_factory=list, description="标签列表"
    )

    def has_tag(self, tag: str) -> bool:
        """是否包含指定标签."""
        return tag in self.tags


class MCPToolCall(BaseModel):
    """MCP 工具调用请求 (L6 调用 L3 工具时的请求体).

    借鉴 MCP tool/call 请求规范，封装工具名称、参数、
    调用 ID 和超时设置。

    Attributes:
        tool_name: 目标工具名称
        arguments: 调用参数 (键值对)
        call_id: 调用唯一标识 (用于关联请求与响应)
        timeout_ms: 超时时间 (毫秒, 默认 30000)
    """

    tool_name: str = Field(..., min_length=1, description="目标工具名称")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="调用参数"
    )
    call_id: str = Field(
        default_factory=lambda: f"call-{uuid.uuid4().hex[:12]}",
        description="调用唯一标识",
    )
    timeout_ms: int = Field(
        default=30000, ge=1, description="超时时间 (毫秒)"
    )


class MCPToolResult(BaseModel):
    """MCP 工具调用结果 (L3 执行工具后返回给 L6 的响应体).

    借鉴 MCP tool/call 响应规范，封装调用结果、成功状态、
    错误信息和耗时。

    Attributes:
        call_id: 调用唯一标识 (与 MCPToolCall.call_id 对应)
        success: 是否成功执行
        result: 执行结果 (失败时为 None)
        error: 错误信息 (成功时为 None)
        latency_ms: 执行耗时 (毫秒)
    """

    call_id: str = Field(..., description="调用唯一标识")
    success: bool = Field(default=True, description="是否成功执行")
    result: Any | None = Field(default=None, description="执行结果")
    error: str | None = Field(default=None, description="错误信息")
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="执行耗时 (毫秒)"
    )

    @classmethod
    def ok(
        cls, call_id: str, result: Any, latency_ms: float = 0.0
    ) -> MCPToolResult:
        """构造成功结果."""
        return cls(
            call_id=call_id,
            success=True,
            result=result,
            error=None,
            latency_ms=latency_ms,
        )

    @classmethod
    def fail(
        cls, call_id: str, error: str, latency_ms: float = 0.0
    ) -> MCPToolResult:
        """构造失败结果."""
        return cls(
            call_id=call_id,
            success=False,
            result=None,
            error=error,
            latency_ms=latency_ms,
        )


# ============================================================
# 适配器函数 (将内部模型转换为接口模型)
# ============================================================


def to_knowledge_hit(chunk_dict: dict, score: float, source: str) -> KnowledgeHit:
    """将检索结果转换为 KnowledgeHit.

    适配 L3 内部检索结果 (DocumentChunk 序列化字典或 RetrievalResult.results 元素)
    为 L4 接口模型 KnowledgeHit。

    字段映射策略 (优先级从高到低):
        - kp_id: chunk_dict["kp_id"] → ["entity_id"] → ["chunk_id"] → ""
        - content: chunk_dict["content"] → ["text"] → ""
        - source_doc_id: chunk_dict["source_doc_id"] → ["document_id"] → ""
        - source_refs: chunk_dict["source_refs"] → provenance 引用 → []
        - sub_scores: chunk_dict["sub_scores"] → quality 子维度 → {}
        - is_bottleneck: chunk_dict["is_bottleneck"] (默认 False)
        - confidence: chunk_dict["confidence"] → quality.overall() → score

    Args:
        chunk_dict: 检索结果字典 (DocumentChunk.model_dump() 或类似结构)
        score: 综合相关性分数 [0.0, 1.0]
        source: 检索来源类型 ("vector"/"keyword"/"graph"/"hybrid")

    Returns:
        转换后的 KnowledgeHit 对象
    """
    # 提取 kp_id (兼容多种内部字段名)
    kp_id: str = (
        chunk_dict.get("kp_id")
        or chunk_dict.get("entity_id")
        or chunk_dict.get("chunk_id")
        or ""
    )

    # 提取内容文本
    content: str = chunk_dict.get("content") or chunk_dict.get("text") or ""

    # 提取来源文档 ID
    source_doc_id: str = (
        chunk_dict.get("source_doc_id")
        or chunk_dict.get("document_id")
        or ""
    )

    # 提取来源引用 (兼容 provenance 结构)
    source_refs: list[str] = list(chunk_dict.get("source_refs", []))
    if not source_refs:
        provenance = chunk_dict.get("provenance")
        if isinstance(provenance, dict):
            primary = provenance.get("primary_source", "")
            if primary:
                source_refs.append(primary)

    # 提取子维度分数 (兼容 quality 结构)
    sub_scores: dict[str, float] = {}
    raw_sub = chunk_dict.get("sub_scores")
    if isinstance(raw_sub, dict):
        sub_scores = {
            k: float(v) for k, v in raw_sub.items() if isinstance(v, (int, float))
        }
    if not sub_scores:
        quality = chunk_dict.get("quality")
        if isinstance(quality, dict):
            for dim in ("accuracy", "trustworthiness", "consistency", "relevancy"):
                val = quality.get(dim)
                if isinstance(val, (int, float)):
                    sub_scores[dim] = float(val)

    # 提取瓶颈标记
    is_bottleneck: bool = bool(chunk_dict.get("is_bottleneck", False))

    # 提取置信度 (优先用显式字段，其次用 quality 综合分数，最后回退到 score)
    confidence: float = score
    raw_conf = chunk_dict.get("confidence")
    if isinstance(raw_conf, (int, float)):
        confidence = float(raw_conf)
    else:
        quality = chunk_dict.get("quality")
        if isinstance(quality, dict):
            overall = quality.get("overall")
            if isinstance(overall, (int, float)):
                confidence = float(overall)

    return KnowledgeHit(
        kp_id=kp_id,
        content=content,
        score=max(0.0, min(1.0, float(score))),
        source=source,
        source_doc_id=source_doc_id,
        source_refs=source_refs,
        sub_scores=sub_scores,
        is_bottleneck=is_bottleneck,
        confidence=max(0.0, min(1.0, confidence)),
    )


def to_retrieval_result(
    query: str,
    results: list[dict],
    scores: list[float],
    intent: str,
    latency: float,
) -> KnowledgeRetrievalResult:
    """转换为 L4 接口格式 KnowledgeRetrievalResult.

    将 L3 内部检索引擎的原始输出 (RetrievalResult 格式: results + scores)
    适配为 L4 决策引擎所需的 KnowledgeRetrievalResult 接口模型。

    自动将每条原始结果转换为 KnowledgeHit，并填充查询元信息。

    Args:
        query: 原始查询文本
        results: 检索结果字典列表
        scores: 对应的相关性分数列表 (与 results 等长)
        intent: 查询意图类型 ("concept"/"numeric"/"relational"/"composite")
        latency: 检索耗时 (毫秒)

    Returns:
        转换后的 KnowledgeRetrievalResult 对象
    """
    # 确保分数列表与结果列表等长
    safe_scores = list(scores)
    if len(safe_scores) < len(results):
        safe_scores.extend([0.0] * (len(results) - len(safe_scores)))
    safe_scores = safe_scores[: len(results)]

    # 推断每条结果的检索来源
    hits: list[KnowledgeHit] = []
    for chunk_dict, score in zip(results, safe_scores):
        # 从结果中提取来源类型，回退到 intent 推断
        source = chunk_dict.get("source_type", "")
        if not source:
            source_map = {
                "concept": "hybrid",
                "numeric": "keyword",
                "relational": "graph",
                "composite": "hybrid",
            }
            source = source_map.get(intent, "vector")
        hits.append(to_knowledge_hit(chunk_dict, float(score), source))

    return KnowledgeRetrievalResult(
        query=query,
        intent_type=intent,
        hits=hits,
        latency_ms=float(latency),
        total=len(hits),
    )


def to_provenance_event(
    event_type: str,
    query: str,
    results: list[dict],
    latency: float,
    **kwargs: Any,
) -> ProvenanceEvent:
    """转换为 CC3 事件 ProvenanceEvent.

    将 L3 内部活动的原始信息适配为 CC3 Provenance 所需的
    ProvenanceEvent 接口模型。

    支持通过关键字参数覆盖默认字段:
        - learner_id: 学习者 ID (默认 "system")
        - intent: 查询意图 (默认 "")
        - retrieval_path: 检索路径 (默认 "")
        - model_versions: 模型版本字典 (默认 {})
        - timestamp: 自定义时间戳 (默认当前时间)

    Args:
        event_type: 事件类型字符串 (需匹配 ProvenanceEventType 枚举值)
        query: 查询文本
        results: 检索结果快照列表
        latency: 事件处理耗时 (毫秒)
        **kwargs: 额外字段覆盖 (learner_id/intent/retrieval_path/model_versions/timestamp)

    Returns:
        转换后的 ProvenanceEvent 对象

    Raises:
        ValueError: event_type 不在 ProvenanceEventType 枚举值中
    """
    # 校验并转换事件类型
    valid_types = {e.value for e in ProvenanceEventType}
    if event_type not in valid_types:
        raise ValueError(
            f"无效的事件类型 '{event_type}', "
            f"有效值: {sorted(valid_types)}"
        )
    resolved_type = ProvenanceEventType(event_type)

    return ProvenanceEvent(
        event_type=resolved_type,
        timestamp=float(kwargs.get("timestamp", time.time())),
        learner_id=kwargs.get("learner_id", "system"),
        query=query,
        intent=kwargs.get("intent", ""),
        retrieval_path=kwargs.get("retrieval_path", ""),
        results=results,
        latency_ms=float(latency),
        model_versions=kwargs.get("model_versions", {}),
    )


def apply_learner_filter(
    profile: LearnerProfile, results: list[KnowledgeHit]
) -> list[KnowledgeHit]:
    """根据学习者画像过滤/调整检索结果.

    对检索命中结果执行三重个性化调整，提升与学习者
    当前状态最相关的知识命中权重，过滤或降权不匹配的内容。

    调整策略:
        1. 弱项 KP 加权 (1.5x): 对学习者薄弱知识点
           (kp_id in profile.weak_kps) 的命中结果，分数提升 1.5 倍，
           优先推送薄弱知识点相关内容以促进补救学习。

        2. 学习风格过滤: 根据 profile.preferred_style 调整分数:
           - VISUAL: 提升 sub_scores 中含 visual/image/table 的命中 (1.2x)，
             不匹配的施加 0.9x 惩罚
           - AUDITORY: 提升 audio/auditory 维度命中 (1.2x)
           - READING: 提升 text/reading 维度命中 (1.2x)
           - KINESTHETIC: 提升 kinesthetic/interactive/practice 维度命中 (1.2x)

        3. 知识层级过滤: 根据 profile.bloom_target 限制超出能力范围的内容:
           - 若命中 sub_scores 含 "difficulty" 字段且超过学习者目标层级，
             分数降低 50% (0.5x)
           - 结合 profile.level 微调: beginner 对高难度内容额外 0.8x

    最终按调整后分数降序排列，并截断到 [0.0, 1.0] 范围。

    Args:
        profile: 学习者画像
        results: 原始检索命中列表

    Returns:
        调整后的命中列表 (按分数降序)
    """
    if not results:
        return []

    # 布鲁姆层级 → 数值映射 (用于难度比较)
    bloom_rank: dict[str, int] = {
        BloomLevel.REMEMBER.value: 1,
        BloomLevel.UNDERSTAND.value: 2,
        BloomLevel.APPLY.value: 3,
        BloomLevel.ANALYZE.value: 4,
        BloomLevel.EVALUATE.value: 5,
        BloomLevel.CREATE.value: 6,
    }
    target_rank = bloom_rank.get(profile.bloom_target.value, 2)

    # 学习风格 → 匹配的 sub_scores 键
    style_keys: dict[str, list[str]] = {
        LearningStyle.VISUAL.value: ["visual", "image", "table", "diagram"],
        LearningStyle.AUDITORY.value: ["audio", "auditory", "voice", "speech"],
        LearningStyle.READING.value: ["text", "reading", "written", "note"],
        LearningStyle.KINESTHETIC.value: [
            "kinesthetic",
            "interactive",
            "practice",
            "hands_on",
        ],
    }
    matched_keys = style_keys.get(profile.preferred_style.value, [])

    # 学习者能力等级 → 难度容忍系数
    level_difficulty_factor: dict[str, float] = {
        "beginner": 0.8,
        "intermediate": 1.0,
        "advanced": 1.0,
    }
    level_factor = level_difficulty_factor.get(profile.level, 1.0)

    adjusted: list[KnowledgeHit] = []

    for hit in results:
        score = hit.score

        # --- 策略 1: 弱项 KP 加权 1.5x ---
        if profile.is_weak_kp(hit.kp_id):
            score *= 1.5

        # --- 策略 2: 学习风格过滤 ---
        if matched_keys:
            # 检查 sub_scores 中是否有匹配风格维度的键
            has_style_match = any(
                key in hit.sub_scores and hit.sub_scores[key] > 0
                for key in matched_keys
            )
            if has_style_match:
                score *= 1.2  # 风格匹配，提升权重
            else:
                score *= 0.9  # 风格不匹配，轻微惩罚

        # --- 策略 3: 知识层级过滤 ---
        difficulty = hit.sub_scores.get("difficulty")
        if isinstance(difficulty, (int, float)) and difficulty > 0:
            # 将 difficulty (0.0~1.0) 映射到布鲁姆层级 (1~6)
            difficulty_rank = int(difficulty * 6) + 1
            if difficulty_rank > target_rank:
                # 难度超过学习者目标层级，降低权重
                score *= 0.5
                # 初学者对高难度内容额外惩罚
                score *= level_factor

        # 截断到 [0.0, 1.0]
        score = max(0.0, min(1.0, score))

        # 创建调整后的副本 (Pydantic v2 model_copy)
        adjusted_hit = hit.model_copy(update={"score": score})
        adjusted.append(adjusted_hit)

    # 按调整后分数降序排列
    adjusted.sort(key=lambda h: h.score, reverse=True)
    return adjusted


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    # L2 → L3 接口模型
    "LearningStyle",
    "BloomLevel",
    "KPMastery",
    "LearnerProfile",
    # L3 → L4 接口模型
    "KnowledgeHit",
    "FactCheckSummary",
    "KnowledgeRetrievalResult",
    # L3 → CC3 溯源接口模型
    "ProvenanceEventType",
    "ProvenanceEvent",
    "ProvenanceMetadata",
    # L3 → L6 MCP 接口模型
    "MCPToolDescriptor",
    "MCPToolCall",
    "MCPToolResult",
    # 适配器函数
    "to_knowledge_hit",
    "to_retrieval_result",
    "to_provenance_event",
    "apply_learner_filter",
]
