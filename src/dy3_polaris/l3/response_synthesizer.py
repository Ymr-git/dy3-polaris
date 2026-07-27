"""L3 领域知识层 — 响应合成器.

融合世界先进方案的响应合成设计:
- LlamaIndex ResponseSynthesizer: 响应合成抽象 + 多种合成模式
  (Compact / Refine / TreeSummarize)，将检索到的 Node 合并为自然语言答案。
- GraphRAG: map-reduce 答案生成 + 社区摘要聚合，通过分层摘要解决
  大规模证据的上下文窗口瓶颈。
- LangChain Stuff/MapReduce/Refine: 文档链合成策略的同构映射。
- FActScore / ProVe: 证据分层 (直接/推断/上下文) + 置信度评估。
- DBpedia/Wikidata 质量框架: 多维质量分数加权融入置信度。

五种合成模式:
1. COMPACT        — 紧凑模式: 拼接相关片段，生成结构化摘要答案
2. REFINE         — 精炼模式: 以首条结果为基线，逐条迭代精炼答案
3. TREE_SUMMARIZE — 树摘要模式: 分组摘要 (map) + 归约合并 (reduce)
4. TEMPLATE       — 模板模式: 按查询类型选择预定义模板填充
5. BULLET         — 要点模式: 抽取关键信息生成要点列表

设计理念:
- 不依赖外部 LLM: 全部使用模板与确定性规则合成，保证可复现、可测试、
  零额外延迟与零额外成本。
- 证据可追溯: 每条答案附带 EvidencePiece 与 Citation，支持溯源审计。
- 置信度可解释: 综合相关性分数均值、证据数量、分数方差与知识质量评分。
- 线程安全: 所有共享状态通过 threading.RLock 保护，支持并发合成。

Usage::

    from dy3_polaris.l3.response_synthesizer import (
        ResponseSynthesizer, SynthesisConfig, SynthesisMode,
    )

    synth = ResponseSynthesizer(SynthesisConfig(mode=SynthesisMode.TREE_SUMMARIZE))
    response = synth.synthesize(retrieval_result, query="Dy3+ 的发射波长是多少")
    print(response.answer, response.confidence)
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from enum import Enum
from threading import RLock
from typing import Any

from pydantic import BaseModel, Field

from .exceptions import L3Error
from .models import (
    AccessLevel,
    DocumentChunk,
    KnowledgeEntity,
    KnowledgeTriple,
    QualityScore,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class SynthesisError(L3Error):
    """响应合成错误.

    当响应合成过程中出现配置错误、不支持的合成模式或数据处理异常时触发。
    继承 L3Error 异常体系，JSON-RPC 错误码 -32415。

    Attributes:
        mode: 触发错误的合成模式名称
    """

    def __init__(
        self,
        mode: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.mode = mode
        super().__init__(
            "L3_SYNTHESIS",
            detail or f"mode={mode}",
            {"mode": mode, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32415


class SynthesisMode(str, Enum):
    """响应合成模式 (借鉴 LlamaIndex ResponseSynthesizer).

    Attributes:
        COMPACT: 紧凑模式 — 拼接所有相关片段，生成结构化摘要答案
        REFINE: 精炼模式 — 以首条结果为基线，逐条迭代精炼答案
        TREE_SUMMARIZE: 树摘要模式 — 分组摘要后归约合并 (GraphRAG map-reduce)
        TEMPLATE: 模板模式 — 按查询类型选择预定义模板填充 (纯模板，无精炼)
        BULLET: 要点模式 — 抽取关键信息生成要点列表
    """

    COMPACT = "compact"
    REFINE = "refine"
    TREE_SUMMARIZE = "tree_summarize"
    TEMPLATE = "template"
    BULLET = "bullet"


class EvidenceType(str, Enum):
    """证据类型 (借鉴 ProVe 证据分层 + FActScore 原子事实).

    Attributes:
        DIRECT: 直接证据 — 结果内容直接陈述查询答案 (高相关切片/实体)
        INFERRED: 推断证据 — 需要基于三元组/关系进行推断 (图结构证据)
        CONTEXTUAL: 上下文证据 — 提供背景上下文，需结合其他证据理解
    """

    DIRECT = "direct"
    INFERRED = "inferred"
    CONTEXTUAL = "contextual"


class QueryType(str, Enum):
    """查询类型 (借鉴 LangChain Router + 意图分类).

    Attributes:
        DEFINITION: 定义类 — "X 是什么 / 什么是 X"
        COMPARISON: 比较类 — "比较 / 区别 / vs"
        NUMERIC: 数值类 — "多少 / 几"
        RELATIONAL: 关系类 — "关系 / 关系是什么"
        PROCEDURAL: 流程类 — "如何 / 怎么 / 步骤"
        GENERAL: 通用类 — 未匹配到特定模式
    """

    DEFINITION = "definition"
    COMPARISON = "comparison"
    NUMERIC = "numeric"
    RELATIONAL = "relational"
    PROCEDURAL = "procedural"
    GENERAL = "general"


class Citation(BaseModel):
    """引用信息 (借鉴学术引用 + RAG 溯源).

    指向答案所依据的具体知识来源，支持溯源审计。

    Attributes:
        citation_id: 引用唯一标识
        source_type: 来源类型 ("entity" / "chunk" / "triple")
        source_id: 来源对象 ID (实体 ID / 切片 ID / 三元组 ID)
        title: 来源标题 (实体名称 / 切片章节 / 三元组描述)
        relevance_score: 相关性分数 (0.0~1.0)
        snippet: 相关文本片段
    """

    citation_id: str = Field(default_factory=lambda: f"cit-{uuid.uuid4().hex[:10]}")
    source_type: str = Field(..., description="来源类型 (entity/chunk/triple)")
    source_id: str = Field(default="", description="来源对象 ID")
    title: str = Field(default="", description="来源标题")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="相关性分数")
    snippet: str = Field(default="", description="相关文本片段")


class EvidencePiece(BaseModel):
    """证据片段 (借鉴 ProVe 证据分层 + FActScore 原子事实).

    从单条检索结果中抽取的原子证据单元，附带置信度与证据类型。

    Attributes:
        evidence_id: 证据唯一标识
        content: 证据文本内容
        source_type: 来源类型 ("entity" / "chunk" / "triple")
        source_id: 来源对象 ID
        confidence: 证据置信度 (0.0~1.0)，综合相关性、质量与访问级别
        evidence_type: 证据类型 ("direct" / "inferred" / "contextual")
    """

    evidence_id: str = Field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:10]}")
    content: str = Field(default="", description="证据文本内容")
    source_type: str = Field(default="", description="来源类型 (entity/chunk/triple)")
    source_id: str = Field(default="", description="来源对象 ID")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="证据置信度")
    evidence_type: str = Field(
        default=EvidenceType.CONTEXTUAL.value,
        description="证据类型 (direct/inferred/contextual)",
    )


class SynthesizedResponse(BaseModel):
    """合成响应 (借鉴 LlamaIndex RESPONSE_TYPE + GraphRAG 答案对象).

    响应合成器的最终输出，包含自然语言答案、引用、证据与置信度。

    Attributes:
        query: 原始查询
        answer: 合成的自然语言答案
        citations: 引用列表
        confidence: 整体置信度 (0.0~1.0)
        source_count: 来源结果总数
        synthesis_mode: 使用的合成模式
        synthesis_time_ms: 合成耗时 (毫秒)
        evidence_pieces: 证据片段列表
        metadata: 扩展元数据 (trace_id / source_type / query_type 等)
    """

    query: str = Field(..., description="原始查询")
    answer: str = Field(default="", description="合成的自然语言答案")
    citations: list[Citation] = Field(default_factory=list, description="引用列表")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="整体置信度")
    source_count: int = Field(default=0, ge=0, description="来源结果总数")
    synthesis_mode: SynthesisMode = Field(default=SynthesisMode.COMPACT, description="合成模式")
    synthesis_time_ms: float = Field(default=0.0, ge=0.0, description="合成耗时 (毫秒)")
    evidence_pieces: list[EvidencePiece] = Field(default_factory=list, description="证据片段列表")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    def is_empty(self) -> bool:
        """响应是否为空 (无答案且无证据)."""
        return not self.answer and not self.evidence_pieces


class SynthesisConfig(BaseModel):
    """合成配置.

    Attributes:
        mode: 合成模式 (默认 COMPACT)
        max_tokens: 答案最大 token 数 (近似估算截断)
        max_citations: 最大引用数量
        min_confidence: 最低置信度下限 (有证据时的地板值)
        include_citations: 是否在响应中包含引用
        language: 输出语言 (ISO 639-1)
    """

    mode: SynthesisMode = Field(default=SynthesisMode.COMPACT, description="合成模式")
    max_tokens: int = Field(default=2000, ge=64, description="答案最大 token 数")
    max_citations: int = Field(default=10, ge=0, description="最大引用数量")
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0, description="最低置信度下限")
    include_citations: bool = Field(default=True, description="是否包含引用")
    language: str = Field(default="zh", description="输出语言")


# 访问级别质量惩罚映射 (借鉴 RBAC 访问控制)
_ACCESS_FACTOR: dict[AccessLevel, float] = {
    AccessLevel.PUBLIC: 1.0,
    AccessLevel.INTERNAL: 0.95,
    AccessLevel.RESTRICTED: 0.85,
    AccessLevel.CONFIDENTIAL: 0.7,
}

# 查询类型关键词模式 (借鉴 LangChain Router 规则优先策略)
_QUERY_PATTERNS: list[tuple[QueryType, re.Pattern[str]]] = [
    (QueryType.DEFINITION, re.compile(r"是什么|什么是|定义|概念|含义|指什么")),
    (QueryType.COMPARISON, re.compile(r"比较|区别|对比|不同|差异|vs|VS| versus")),
    (QueryType.NUMERIC, re.compile(r"多少|几|数值|值是|等于|含量|比例|温度|波长|浓度")),
    (QueryType.RELATIONAL, re.compile(r"关系|联系|关联|作用|影响|之间")),
    (QueryType.PROCEDURAL, re.compile(r"如何|怎么|怎样|步骤|方法|流程|过程|做法")),
]


class ResponseSynthesizer:
    """响应合成器 (借鉴 LlamaIndex ResponseSynthesizer + GraphRAG map-reduce).

    将检索结果合成为自然语言答案，包含引用和置信度评估。
    不依赖外部 LLM，使用模板和确定性规则合成，保证可复现与可测试。

    设计理念:
        - 借鉴 LlamaIndex ResponseSynthesizer: 统一 synthesize 入口 +
          多模式分派 (Compact/Refine/TreeSummarize)，将检索 Node 合成为答案。
        - 借鉴 GraphRAG map-reduce: TREE_SUMMARIZE 模式将证据分组摘要 (map)
          再归约合并 (reduce)，缓解大规模证据的上下文窗口瓶颈。
        - 借鉴 FActScore/ProVe: 每条答案附带分层证据 (直接/推断/上下文)
          与可解释置信度。
        - 借鉴 DBpedia/Wikidata 质量框架: 多维质量分数加权融入证据置信度。

    线程安全:
        所有共享状态 (合成计数 _synthesis_count) 通过 threading.RLock 保护，
        RLock 支持同线程重入，适用于合成方法内部嵌套调用。

    Attributes:
        _config: 合成配置
        _lock: 线程安全锁 (RLock)
        _synthesis_count: 累计合成次数
    """

    def __init__(self, config: SynthesisConfig | None = None) -> None:
        """初始化响应合成器.

        Args:
            config: 合成配置，None 时使用默认配置 (COMPACT 模式)
        """
        self._config: SynthesisConfig = config or SynthesisConfig()
        self._lock: RLock = RLock()
        self._synthesis_count: int = 0

    def synthesize(
        self,
        retrieval_result: RetrievalResult,
        *,
        query: str | None = None,
    ) -> SynthesizedResponse:
        """合成响应 (主入口).

        将 RetrievalResult 合成为 SynthesizedResponse，根据配置的合成模式
        分派到对应的合成策略。线程安全。

        Args:
            retrieval_result: 检索结果
            query: 显式查询，None 时使用 retrieval_result.query

        Returns:
            合成响应

        Raises:
            SynthesisError: 当配置了不支持的合成模式时
        """
        with self._lock:
            self._synthesis_count += 1
            start_time = time.perf_counter()
            effective_query: str = (query or retrieval_result.query or "").strip()

            # 空结果快速返回
            if retrieval_result.is_empty():
                synthesis_time_ms = (time.perf_counter() - start_time) * 1000.0
                logger.debug("空检索结果，返回低置信度响应: query=%r", effective_query)
                return SynthesizedResponse(
                    query=effective_query,
                    answer=self._format_answer(
                        "未检索到与\u201c{query}\u201d相关的知识。",
                        query=effective_query or "该查询",
                    ),
                    citations=[],
                    confidence=0.0,
                    source_count=0,
                    synthesis_mode=self._config.mode,
                    synthesis_time_ms=round(synthesis_time_ms, 2),
                    evidence_pieces=[],
                    metadata={
                        "trace_id": retrieval_result.trace_id,
                        "source_type": retrieval_result.source_type,
                        "empty": True,
                        "query_type": self._detect_query_type(effective_query),
                    },
                )

            results: list[dict[str, Any]] = retrieval_result.results
            scores: list[float] = list(retrieval_result.scores)
            # 分数列表对齐: 缺失分数以中位数 0.5 兜底
            if len(scores) < len(results):
                scores = scores + [0.5] * (len(results) - len(scores))
            elif len(scores) > len(results):
                scores = scores[: len(results)]

            # 按合成模式分派
            mode = self._config.mode
            try:
                if mode == SynthesisMode.COMPACT:
                    answer, evidence_pieces, citations = self._synthesize_compact(
                        results, scores, effective_query
                    )
                elif mode == SynthesisMode.REFINE:
                    answer, evidence_pieces, citations = self._synthesize_refine(
                        results, scores, effective_query
                    )
                elif mode == SynthesisMode.TREE_SUMMARIZE:
                    answer, evidence_pieces, citations = self._synthesize_tree(
                        results, scores, effective_query
                    )
                elif mode == SynthesisMode.TEMPLATE:
                    answer, evidence_pieces, citations = self._synthesize_template(
                        results, scores, effective_query
                    )
                elif mode == SynthesisMode.BULLET:
                    answer, evidence_pieces, citations = self._synthesize_bullet(
                        results, scores, effective_query
                    )
                else:  # pragma: no cover - 枚举完备性兜底
                    raise SynthesisError(mode=str(mode), detail=f"不支持的合成模式: {mode}")
            except SynthesisError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("响应合成失败: mode=%s query=%r", mode, effective_query)
                raise SynthesisError(
                    mode=str(mode),
                    detail=f"合成过程异常: {exc}",
                    context={"query": effective_query},
                ) from exc

            # 计算整体置信度
            confidence = self._calculate_confidence(scores, len(evidence_pieces))
            if evidence_pieces:
                confidence = max(confidence, self._config.min_confidence)
            else:
                confidence = 0.0

            # 引用裁剪与开关
            if not self._config.include_citations:
                citations = []
            else:
                citations = citations[: self._config.max_citations]

            synthesis_time_ms = (time.perf_counter() - start_time) * 1000.0
            response = SynthesizedResponse(
                query=effective_query,
                answer=answer,
                citations=citations,
                confidence=round(confidence, 4),
                source_count=len(results),
                synthesis_mode=self._config.mode,
                synthesis_time_ms=round(synthesis_time_ms, 2),
                evidence_pieces=evidence_pieces,
                metadata={
                    "trace_id": retrieval_result.trace_id,
                    "source_type": retrieval_result.source_type,
                    "retrieval_time_ms": retrieval_result.retrieval_time_ms,
                    "query_type": self._detect_query_type(effective_query),
                },
            )
            logger.debug(
                "合成完成: mode=%s sources=%d evidence=%d confidence=%.3f time=%.2fms",
                mode.value, len(results), len(evidence_pieces), confidence, synthesis_time_ms,
            )
            return response

    # ---- 合成模式实现 ----

    def _synthesize_compact(
        self, results: list[dict[str, Any]], scores: list[float], query: str,
    ) -> tuple[str, list[EvidencePiece], list[Citation]]:
        """紧凑模式: 拼接相关片段，生成结构化摘要答案 (借鉴 LlamaIndex Compact).

        将所有结果的文本片段拼接为整体上下文，再生成结构化摘要答案。
        适合证据数量适中、需要全局视角的场景。
        """
        evidence_pieces: list[EvidencePiece] = []
        citations: list[Citation] = []
        snippets: list[str] = []

        for index, (result, score) in enumerate(zip(results, scores)):
            evidence_pieces.append(self._extract_evidence(result, score))
            if len(citations) < self._config.max_citations:
                citations.append(self._create_citation(result, score, index))
            snippet = self._get_result_text(result)
            if snippet:
                snippets.append(snippet)

        if not snippets:
            return (
                self._format_answer("未找到与\u201c{query}\u201d直接相关的知识内容。", query=query),
                evidence_pieces, citations,
            )

        combined = self._truncate_to_tokens("\n".join(snippets))
        answer = self._format_answer(
            "根据检索到的 {count} 条相关知识，针对\u201c{query}\u201d的解答如下：\n\n"
            "{content}\n\n（共综合 {count} 条证据）",
            count=len(snippets), query=query, content=combined,
        )
        return answer, evidence_pieces, citations

    def _synthesize_refine(
        self, results: list[dict[str, Any]], scores: list[float], query: str,
    ) -> tuple[str, list[EvidencePiece], list[Citation]]:
        """精炼模式: 以首条结果为基线，逐条迭代精炼 (借鉴 LlamaIndex Refine).

        按相关性分数降序排列，以最高分结果作为初始答案基线，
        随后逐条引入新证据对答案进行增量精炼。
        适合需要逐步构建完整答案、强调证据增量补充的场景。
        """
        evidence_pieces: list[EvidencePiece] = []
        citations: list[Citation] = []

        if not results:
            return (
                self._format_answer("未找到与\u201c{query}\u201d相关的知识。", query=query),
                evidence_pieces, citations,
            )

        # 按分数降序排列，优先处理最相关结果
        paired = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

        # 以第一条结果作为初始答案基线
        first_result, first_score = paired[0]
        evidence_pieces.append(self._extract_evidence(first_result, first_score))
        citations.append(self._create_citation(first_result, first_score, 0))
        current_answer = self._get_result_text(first_result)

        # 迭代精炼: 逐条引入新证据
        connectors = ["此外，", "另外，", "同时，", "进一步地，", "值得注意的是，"]
        for offset, (result, score) in enumerate(paired[1:], start=1):
            text = self._get_result_text(result)
            if not text:
                continue
            evidence_pieces.append(self._extract_evidence(result, score))
            if len(citations) < self._config.max_citations:
                citations.append(self._create_citation(result, score, offset))
            connector = connectors[(offset - 1) % len(connectors)]
            current_answer = f"{current_answer}\n{connector}{self._extract_key_sentence(text)}"

        current_answer = self._truncate_to_tokens(current_answer)
        answer = self._format_answer(
            "针对\u201c{query}\u201d，经迭代综合 {count} 条知识得出：\n\n{content}",
            count=len(evidence_pieces), query=query, content=current_answer,
        )
        return answer, evidence_pieces, citations

    def _synthesize_tree(
        self, results: list[dict[str, Any]], scores: list[float], query: str,
    ) -> tuple[str, list[EvidencePiece], list[Citation]]:
        """树摘要模式: 分组摘要 + 归约合并 (借鉴 GraphRAG map-reduce).

        将结果分组 (每组 group_size 条)，对每组生成局部摘要 (map 阶段)，
        再将所有组摘要归约合并为最终答案 (reduce 阶段)。
        适合证据规模较大、需要分层压缩的场景，缓解上下文窗口瓶颈。
        """
        evidence_pieces: list[EvidencePiece] = []
        citations: list[Citation] = []

        if not results:
            return (
                self._format_answer("未找到与\u201c{query}\u201d相关的知识。", query=query),
                evidence_pieces, citations,
            )

        paired = list(zip(results, scores))
        group_size = 3  # 每组证据数 (借鉴 GraphRAG 社区摘要粒度)
        groups = [paired[i : i + group_size] for i in range(0, len(paired), group_size)]

        # Map 阶段: 对每组生成局部摘要
        group_summaries: list[str] = []
        for group_idx, group in enumerate(groups):
            group_snippets: list[str] = []
            for result, score in group:
                evidence_pieces.append(self._extract_evidence(result, score))
                text = self._get_result_text(result)
                if text:
                    group_snippets.append(text)
            group_text = "；".join(s for s in group_snippets if s)
            if group_text:
                group_summaries.append(f"[摘要{group_idx + 1}] {group_text}")

        # Reduce 阶段: 合并所有组摘要
        for index, (result, score) in enumerate(paired):
            if len(citations) < self._config.max_citations:
                citations.append(self._create_citation(result, score, index))

        combined = self._truncate_to_tokens("\n".join(group_summaries))
        answer = self._format_answer(
            "针对\u201c{query}\u201d，采用 map-reduce 策略综合 {group_count} 组知识"
            "（{evidence_count} 条证据）：\n\n{content}",
            query=query, group_count=len(groups),
            evidence_count=len(evidence_pieces), content=combined,
        )
        return answer, evidence_pieces, citations

    def _synthesize_template(
        self, results: list[dict[str, Any]], scores: list[float], query: str,
    ) -> tuple[str, list[EvidencePiece], list[Citation]]:
        """模板模式: 按查询类型选择预定义模板填充 (借鉴 LangChain 模板链).

        根据查询意图 (定义/比较/数值/关系/流程) 选择对应模板，
        从最相关结果中抽取关键信息填充模板。纯模板合成，不做迭代精炼。
        适合查询意图明确、需要结构化输出的场景。
        """
        evidence_pieces: list[EvidencePiece] = []
        citations: list[Citation] = []

        for index, (result, score) in enumerate(zip(results, scores)):
            evidence_pieces.append(self._extract_evidence(result, score))
            if len(citations) < self._config.max_citations:
                citations.append(self._create_citation(result, score, index))

        if not results:
            return (
                self._format_answer("未找到与\u201c{query}\u201d相关的知识。", query=query),
                evidence_pieces, citations,
            )

        query_type = self._detect_query_type(query)
        best_result, _ = max(zip(results, scores), key=lambda x: x[1])

        # 按查询类型选择模板
        if query_type == QueryType.DEFINITION.value:
            name = self._get_title(best_result)
            description = best_result.get("description") or self._get_result_text(best_result)
            aliases = best_result.get("aliases")
            alias_str = ""
            if isinstance(aliases, list) and aliases:
                alias_str = f"（又称：{'、'.join(str(a) for a in aliases[:5])}）"
            answer = self._format_answer(
                "{name}{alias}：{description}",
                name=name or query, alias=alias_str,
                description=description or "暂无详细定义描述。",
            )
        elif query_type == QueryType.COMPARISON.value:
            paired = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
            items: list[str] = []
            for result, _score in paired[:4]:
                name = self._get_title(result)
                desc = self._extract_key_sentence(
                    result.get("description") or self._get_result_text(result)
                )
                items.append(f"- {name}：{desc}" if desc else f"- {name}")
            body = "\n".join(items) if items else "暂无可比较的知识点。"
            answer = self._format_answer(
                "关于\u201c{query}\u201d的比较分析：\n\n{body}", query=query, body=body,
            )
        elif query_type == QueryType.NUMERIC.value:
            findings: list[str] = []
            numeric_re = re.compile(r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*([a-zA-Z%°℃]+)?")
            for result in results:
                text = self._get_result_text(result)
                name = self._get_title(result)
                match = numeric_re.search(text)
                if match:
                    value, unit = match.group(1), (match.group(2) or "")
                    sentence = self._extract_key_sentence(text)
                    findings.append(f"- {name}：{value}{unit}（来源：{sentence[:60]}）")
            body = "\n".join(findings) if findings else "未检索到明确的数值信息。"
            answer = self._format_answer(
                "根据知识库，关于\u201c{query}\u201d的数值信息：\n\n{body}",
                query=query, body=body,
            )
        elif query_type == QueryType.RELATIONAL.value:
            relations: list[str] = []
            for result in results:
                if self._detect_source_type(result) == "triple":
                    relations.append(f"- {self._get_result_text(result)}")
                else:
                    text = self._get_result_text(result)
                    if text:
                        relations.append(f"- {self._extract_key_sentence(text)}")
            body = "\n".join(relations) if relations else "未检索到明确的关系信息。"
            answer = self._format_answer(
                "根据知识图谱，关于\u201c{query}\u201d的关系信息：\n\n{body}",
                query=query, body=body,
            )
        elif query_type == QueryType.PROCEDURAL.value:
            steps: list[str] = []
            for step_idx, result in enumerate(results, start=1):
                text = self._get_result_text(result)
                if text:
                    steps.append(f"{step_idx}. {self._extract_key_sentence(text)}")
            body = "\n".join(steps) if steps else "未检索到明确的方法或步骤信息。"
            answer = self._format_answer(
                "关于\u201c{query}\u201d的方法/步骤：\n\n{body}", query=query, body=body,
            )
        else:
            # 通用模板: 以最相关结果描述作答
            name = self._get_title(best_result)
            description = self._get_result_text(best_result)
            answer = self._format_answer(
                "关于\u201c{query}\u201d，相关知识点\u201c{name}\u201d的信息如下：\n\n{description}",
                query=query, name=name or "未知", description=description or "暂无详细描述",
            )

        return self._truncate_to_tokens(answer), evidence_pieces, citations

    def _synthesize_bullet(
        self, results: list[dict[str, Any]], scores: list[float], query: str,
    ) -> tuple[str, list[EvidencePiece], list[Citation]]:
        """要点模式: 抽取关键信息生成要点列表.

        从每条结果中抽取关键要点，组织为项目符号列表。
        适合需要快速浏览、信息密度高的场景。
        """
        evidence_pieces: list[EvidencePiece] = []
        citations: list[Citation] = []
        bullets: list[str] = []

        for index, (result, score) in enumerate(zip(results, scores)):
            evidence_pieces.append(self._extract_evidence(result, score))
            if len(citations) < self._config.max_citations:
                citations.append(self._create_citation(result, score, index))
            key_point = self._extract_key_point(result)
            if key_point:
                bullets.append(f"- {key_point}")

        if not bullets:
            return (
                self._format_answer("未找到与\u201c{query}\u201d相关的知识要点。", query=query),
                evidence_pieces, citations,
            )

        body = self._truncate_to_tokens("\n".join(bullets))
        answer = self._format_answer(
            "关于\u201c{query}\u201d的关键信息如下：\n\n{body}", query=query, body=body,
        )
        return answer, evidence_pieces, citations

    # ---- 证据与引用提取 ----

    def _extract_evidence(self, result: dict[str, Any], score: float) -> EvidencePiece:
        """从单条结果中抽取证据片段 (借鉴 ProVe 证据分层).

        根据结果来源类型提取最相关的文本片段，并综合相关性分数、
        知识质量评分与访问级别计算证据置信度，判定证据类型。

        证据置信度 = 相关性分数 × 质量因子 × 访问因子
        证据类型: triple→inferred, score≥0.7→direct, 其余→contextual
        """
        source_type = self._detect_source_type(result)
        source_id = self._get_source_id(result, source_type)
        content = self._get_result_text(result)

        # 证据置信度 = 相关性 × 质量因子 × 访问因子
        normalized_score = max(0.0, min(1.0, float(score)))
        quality_factor = self._get_quality_factor(result)
        access_factor = self._get_access_factor(result)
        confidence = max(0.0, min(1.0, normalized_score * quality_factor * access_factor))

        # 证据类型判定 (借鉴 ProVe 证据分层)
        if source_type == "triple":
            evidence_type = EvidenceType.INFERRED.value
        elif normalized_score >= 0.7:
            evidence_type = EvidenceType.DIRECT.value
        else:
            evidence_type = EvidenceType.CONTEXTUAL.value

        return EvidencePiece(
            content=content, source_type=source_type, source_id=source_id,
            confidence=round(confidence, 4), evidence_type=evidence_type,
        )

    def _create_citation(
        self, result: dict[str, Any], score: float, index: int,
    ) -> Citation:
        """从结果创建引用 (借鉴学术引用 + RAG 溯源).

        Args:
            result: 单条检索结果字典
            score: 该结果的相关性分数
            index: 结果序号 (用于生成稳定标识)
        """
        source_type = self._detect_source_type(result)
        source_id = self._get_source_id(result, source_type)
        title = self._get_title(result)
        snippet = self._extract_key_sentence(self._get_result_text(result), max_length=200)
        relevance = max(0.0, min(1.0, float(score)))
        return Citation(
            citation_id=f"cit-{index:04d}-{uuid.uuid4().hex[:8]}",
            source_type=source_type, source_id=source_id, title=title,
            relevance_score=round(relevance, 4), snippet=snippet,
        )

    # ---- 置信度计算 ----

    def _calculate_confidence(self, scores: list[float], evidence_count: int) -> float:
        """计算整体置信度 (借鉴 FActScore + 多因子加权).

        综合三个因子计算整体置信度:
        1. 相关性分数均值 (权重 0.5): 检索结果整体相关程度
        2. 证据数量因子 (权重 0.3): 证据越充分置信度越高 (对数增长，趋于饱和)
        3. 分数一致性因子 (权重 0.2): 分数方差越小越一致，置信度越高

        confidence = avg_score * 0.5 + evidence_factor * 0.3 + consistency * 0.2
        """
        if not scores or evidence_count == 0:
            return 0.0

        # 因子 1: 相关性分数均值 (归一化到 [0,1])
        avg_score = sum(float(s) for s in scores) / len(scores)
        avg_norm = max(0.0, min(1.0, avg_score))

        # 因子 2: 证据数量因子 (对数增长，约 9 条证据趋于 0.95)
        evidence_factor = 1.0 - math.exp(-evidence_count / 3.0)

        # 因子 3: 分数一致性因子 (方差越小越一致)
        if len(scores) > 1:
            variance = sum((float(s) - avg_score) ** 2 for s in scores) / len(scores)
            consistency = max(0.0, 1.0 - variance * 4.0)
        else:
            consistency = 0.8  # 单条证据缺乏交叉验证，一致性折扣

        confidence = avg_norm * 0.5 + evidence_factor * 0.3 + consistency * 0.2
        return max(0.0, min(1.0, confidence))

    # ---- 查询类型检测 ----

    def _detect_query_type(self, query: str) -> str:
        """检测查询类型 (借鉴 LangChain Router 规则优先策略).

        通过正则模式匹配识别查询意图:
        - definition:  "是什么 / 什么是 / 定义 / 概念 / 含义"
        - comparison:   "比较 / 区别 / 对比 / vs"
        - numeric:      "多少 / 几 / 数值 / 波长 / 浓度"
        - relational:   "关系 / 联系 / 关联 / 作用"
        - procedural:   "如何 / 怎么 / 步骤 / 方法 / 流程"
        - general:      未匹配到特定模式

        优先级按列表顺序 (definition > comparison > numeric > relational > procedural)，
        首个匹配的模式胜出。
        """
        if not query:
            return QueryType.GENERAL.value
        for query_type, pattern in _QUERY_PATTERNS:
            if pattern.search(query):
                return query_type.value
        return QueryType.GENERAL.value

    # ---- 答案格式化 ----

    def _format_answer(self, template: str, **kwargs: Any) -> str:
        """使用模板格式化答案 (安全格式化).

        使用 str.format_map 进行格式化，缺失的键以空字符串替代，
        避免因模板变量缺失而抛出 KeyError。格式化异常时回退返回原模板。
        """

        class _SafeDict(dict[str, Any]):
            """安全字典: 缺失键返回空字符串而非抛出 KeyError."""

            def __missing__(self, key: str) -> str:  # type: ignore[override]
                return ""

        try:
            return template.format_map(_SafeDict(kwargs))
        except (IndexError, ValueError):
            return template

    # ---- 统计信息 ----

    def get_stats(self) -> dict[str, Any]:
        """获取合成统计信息.

        返回合成器的运行时统计与当前配置快照，线程安全。

        Returns:
            统计信息字典 (synthesis_count / mode / max_tokens 等配置参数)
        """
        with self._lock:
            return {
                "synthesis_count": self._synthesis_count,
                "mode": self._config.mode.value,
                "max_tokens": self._config.max_tokens,
                "max_citations": self._config.max_citations,
                "min_confidence": self._config.min_confidence,
                "include_citations": self._config.include_citations,
                "language": self._config.language,
            }

    # ---- 内部工具方法 ----

    def _detect_source_type(self, result: dict[str, Any]) -> str:
        """检测结果来源类型 (entity / chunk / triple).

        根据结果字典的特征键推断来源类型:
        - entity: 含 entity_id 或 entity_type
        - triple: 含 triple_id 或 (subject_id + predicate)
        - chunk:  含 chunk_id 或 (document_id + content)
        """
        if "entity_id" in result or "entity_type" in result:
            return "entity"
        if "triple_id" in result or ("subject_id" in result and "predicate" in result):
            return "triple"
        if "chunk_id" in result or ("document_id" in result and "content" in result):
            return "chunk"
        return "chunk" if "content" in result else "entity"

    def _get_source_id(self, result: dict[str, Any], source_type: str) -> str:
        """获取来源对象 ID."""
        id_key_map = {"entity": "entity_id", "triple": "triple_id", "chunk": "chunk_id"}
        id_key = id_key_map.get(source_type, "id")
        return str(result.get(id_key) or result.get("id") or "")

    def _get_title(self, result: dict[str, Any]) -> str:
        """获取结果标题 (实体名称 / 切片章节 / 三元组描述)."""
        source_type = self._detect_source_type(result)
        if source_type == "entity":
            return str(result.get("name") or result.get("title") or "")
        if source_type == "chunk":
            section = result.get("section")
            if section:
                return str(section)
            return str(result.get("document_id") or result.get("title") or "")
        if source_type == "triple":
            subject = result.get("subject_name") or result.get("subject_id") or ""
            predicate = result.get("predicate") or ""
            obj = result.get("object_name") or result.get("object_id") or ""
            if subject and predicate and obj:
                return f"{subject} {predicate} {obj}"
            return str(result.get("triple_id") or "")
        return str(result.get("title") or result.get("name") or "")

    def _get_result_text(self, result: dict[str, Any]) -> str:
        """从结果字典提取主要文本内容.

        根据来源类型提取最相关的文本:
        - entity: 名称 + 描述 + 关键属性
        - chunk:  content 字段
        - triple: subject predicate object 组合
        """
        source_type = self._detect_source_type(result)

        if source_type == "entity":
            parts: list[str] = []
            name = result.get("name", "")
            description = result.get("description", "")
            if name:
                parts.append(str(name))
            if description:
                parts.append(str(description))
            properties = result.get("properties", {})
            if isinstance(properties, dict) and properties:
                prop_items = list(properties.items())[:5]
                parts.append("；".join(f"{k}={v}" for k, v in prop_items))
            return "：".join(parts) if parts else ""

        if source_type == "chunk":
            return str(result.get("content") or "")

        if source_type == "triple":
            subject = result.get("subject_name") or result.get("subject_id") or ""
            predicate = result.get("predicate") or ""
            obj = result.get("object_name") or result.get("object_id") or ""
            if result.get("object_is_literal") and result.get("object_value") is not None:
                obj = str(result.get("object_value"))
            if subject and predicate and obj:
                return f"{subject} {predicate} {obj}"
            return ""

        return str(result.get("content") or result.get("description") or result.get("name") or "")

    def _extract_key_sentence(self, text: str, max_length: int = 120) -> str:
        """提取文本的首个关键句.

        按中文标点 (。！？；) 与换行分割，返回首个非空句，超长则截断。
        用于精炼模式与引用片段生成。
        """
        if not text:
            return ""
        for sentence in re.split(r"[。！？\n；]", text):
            stripped = sentence.strip()
            if stripped:
                return stripped[:max_length]
        return text[:max_length]

    def _extract_key_point(self, result: dict[str, Any]) -> str:
        """从结果中抽取关键要点 (用于要点模式)."""
        source_type = self._detect_source_type(result)
        if source_type == "entity":
            name = result.get("name", "")
            description = result.get("description", "")
            if name and description:
                return f"{name}：{self._extract_key_sentence(description)}"
            return str(name or description or "")
        if source_type == "triple":
            return self._get_result_text(result)
        return self._extract_key_sentence(str(result.get("content", "")))

    def _get_quality_factor(self, result: dict[str, Any]) -> float:
        """从结果中提取质量因子 (借鉴 QualityScore 多维质量评估).

        优先解析序列化的 QualityScore 字典 (取多维均值)，
        其次取数值型 quality_score / confidence_score，默认 0.8。
        """
        quality = result.get("quality") or result.get("quality_score")
        if isinstance(quality, dict):
            dims = [
                "accuracy", "trustworthiness", "consistency",
                "timeliness", "completeness", "relevancy",
            ]
            values = [float(quality.get(d, 0.8)) for d in dims if d in quality]
            if values:
                return max(0.0, min(1.0, sum(values) / len(values)))
        if isinstance(quality, (int, float)):
            return max(0.0, min(1.0, float(quality)))
        confidence_score = result.get("confidence_score")
        if isinstance(confidence_score, (int, float)):
            return max(0.0, min(1.0, float(confidence_score)))
        return 0.8

    def _get_access_factor(self, result: dict[str, Any]) -> float:
        """从结果中提取访问级别因子 (借鉴 RBAC 访问控制).

        根据访问控制级别返回置信度惩罚因子:
        public=1.0, internal=0.95, restricted=0.85, confidential=0.7。
        """
        level = result.get("access_level", "internal")
        try:
            return _ACCESS_FACTOR.get(AccessLevel(str(level)), 0.95)
        except (ValueError, TypeError):
            return 0.95

    def _estimate_tokens(self, text: str) -> int:
        """估算文本 token 数.

        中文约 1.5 字符/token，英文/符号约 4 字符/token，取加权估算。
        """
        if not text:
            return 0
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_count = len(text) - chinese_count
        return max(1, int(chinese_count / 1.5 + other_count / 4))

    def _truncate_to_tokens(self, text: str) -> str:
        """按 max_tokens 截断文本.

        当文本估算 token 数超过配置上限时，按比例截断并追加省略号。
        用于控制合成答案长度，避免上下文窗口溢出。
        """
        max_tokens = self._config.max_tokens
        if max_tokens <= 0:
            return text
        estimated = self._estimate_tokens(text)
        if estimated <= max_tokens:
            return text
        ratio = max_tokens / max(1, estimated)
        cut_length = max(1, int(len(text) * ratio * 0.95))
        return text[:cut_length].rstrip() + "…"


__all__ = [
    "SynthesisError",
    "SynthesisMode",
    "EvidenceType",
    "QueryType",
    "Citation",
    "EvidencePiece",
    "SynthesizedResponse",
    "SynthesisConfig",
    "ResponseSynthesizer",
]
