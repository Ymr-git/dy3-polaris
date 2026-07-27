"""L3 领域知识层 — 查询重写引擎.

融合世界先进方案的查询重写设计:
- LangChain MultiQueryRetriever: 多查询生成 (同一意图生成多个变体提升召回)
- LlamaIndex SubQuestionQueryEngine: 子问题分解 (复合查询拆解为可并行的子问题)
- HyDE (Hypothetical Document Embeddings, Gao et al. 2022): 假设文档生成
  (先生成假设性答案文档再向量化, 缩小查询-文档语义鸿沟)
- LangChain ContextualCompressionRetriever: 上下文压缩 (提取查询核心意图)
- Elasticsearch synonym filter + query expansion: 同义词扩展与查询扩展

五类重写策略:
1. SYNONYM     — 同义词扩展: 基于领域词典扩展查询词 (借鉴 ES synonym filter)
2. DECOMPOSE   — 子问题分解: 将复合查询分解为子问题 (借鉴 LlamaIndex SubQuestionQueryEngine)
3. HYDE        — 假设文档: 生成假设文档用于向量检索 (借鉴 HyDE)
4. EXPAND      — 查询扩展: 添加相关术语 (借鉴 LangChain MultiQueryRetriever)
5. CONTEXTUAL  — 上下文压缩: 提取查询核心意图 (借鉴 ContextualCompressionRetriever)

设计理念:
- 查询重写在检索之前执行, 通过改写查询提升召回率与精确率。
- 不同策略适用于不同场景: 同义词扩展弥补词汇鸿沟, 子问题分解处理复合查询,
  HyDE 缩小查询-文档语义鸿沟, 查询扩展增加语义覆盖, 上下文压缩去除噪声。
- 内置稀土发光材料领域词典, 开箱即用; 支持自定义领域词典扩展。
- 不依赖外部 LLM, 采用规则+模板的确定性重写 (可复现、可测试);
  接口设计预留未来接入 LLM 重写的扩展点。

线程安全: 领域词典通过 threading.RLock 保护, 支持并发重写与词典更新。
所有重写操作均为只读查询 (对领域词典), 重写过程无共享可变状态。

Usage::

    from dy3_polaris.l3.query_rewriter import (
        QueryRewriter, RewriteStrategy,
    )

    rewriter = QueryRewriter()

    # 单策略重写
    rq = rewriter.rewrite("Dy3+离子的发射波长和能级跃迁", strategy=RewriteStrategy.EXPAND)
    print(rq.rewritten)
    print(rq.sub_queries)
    print(rq.confidence)

    # 多策略重写 (生成多个变体, 借鉴 MultiQueryRetriever)
    variants = rewriter.rewrite_multi("荧光猝灭与浓度关系")
    for v in variants:
        print(v.strategy, v.rewritten)

    # 直接使用子能力
    expanded = rewriter.expand_synonyms("波长效率")
    sub_qs = rewriter.decompose("波长和效率有什么关系?")
    hyde_doc = rewriter.generate_hyde("Dy3+的发射波长")
    keywords = rewriter.extract_keywords("稀土发光材料的浓度猝灭机理")
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 重写策略枚举
# ============================================================


class RewriteStrategy(str, Enum):
    """查询重写策略 (借鉴 LangChain MultiQueryRetriever + LlamaIndex + HyDE).

    Attributes:
        SYNONYM: 同义词扩展 — 基于领域词典扩展查询词, 弥补词汇鸿沟
        DECOMPOSE: 子问题分解 — 将复合查询分解为子问题, 支持并行检索
        HYDE: 假设文档 — 生成假设性答案文档用于向量检索, 缩小语义鸿沟
        EXPAND: 查询扩展 — 添加相关术语, 增加语义覆盖 (综合策略)
        CONTEXTUAL: 上下文压缩 — 提取查询核心意图, 去除噪声词
    """

    SYNONYM = "synonym"
    DECOMPOSE = "decompose"
    HYDE = "hyde"
    EXPAND = "expand"
    CONTEXTUAL = "contextual"


# ============================================================
# 重写结果数据结构
# ============================================================


@dataclass
class RewrittenQuery:
    """重写后的查询结果 (借鉴 LangChain RewrittenQuery + LlamaIndex 查询变体).

    封装原始查询、重写后的查询、所用策略、子查询列表、置信度与元信息。

    Attributes:
        original: 原始查询文本
        rewritten: 重写后的查询文本 (或假设文档)
        strategy: 使用的重写策略
        sub_queries: 子查询列表 (仅 DECOMPOSE 策略非空, 其他策略可填充相关变体)
        confidence: 重写置信度 [0.0, 1.0], 反映重写质量与匹配程度
        metadata: 附加元信息 (如匹配的领域术语、提取的关键词等)
    """

    original: str
    rewritten: str
    strategy: RewriteStrategy
    sub_queries: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RewrittenQuery(strategy={self.strategy.value}, "
            f"confidence={self.confidence:.3f}, "
            f"rewritten={self.rewritten[:50]!r}...)"
        )


# ============================================================
# 查询重写引擎
# ============================================================


class QueryRewriter:
    """查询重写引擎.

    融合 LangChain MultiQueryRetriever、LlamaIndex SubQuestionQueryEngine、
    HyDE、ContextualCompressionRetriever 等先进方案, 提供五类查询重写策略。

    功能:
    1. 同义词扩展: 基于领域词典扩展查询词 (expand_synonyms)
    2. 子问题分解: 将复合查询分解为子问题 (decompose)
    3. HyDE: 生成假设文档用于向量检索 (generate_hyde)
    4. 查询扩展: 添加相关术语 (综合同义词+关键词)
    5. 上下文压缩: 提取查询核心意图 (去除停用词与噪声)

    内置稀土发光材料领域词典 (8 个核心术语及其同义词), 支持自定义扩展。
    采用规则+模板的确定性重写, 不依赖外部 LLM, 保证可复现与可测试。

    Attributes:
        _domain_dict: 领域词典 {术语: [同义词...]}
        _domain_vocab: 领域词汇全集 (术语+同义词), 用于关键词匹配
        _lock: 线程安全锁 (保护领域词典的读写)
    """

    # --------------------------------------------------------
    # 内置领域词典 — 稀土发光材料领域
    # --------------------------------------------------------

    DEFAULT_DOMAIN_DICT: dict[str, list[str]] = {
        "波长": ["发射波长", "激发波长", "吸收波长", "emission wavelength"],
        "能级": ["能量级别", "electronic state", "energy level"],
        "跃迁": ["电子跃迁", "辐射跃迁", "无辐射跃迁", "transition"],
        "效率": ["量子效率", "发光效率", "quantum efficiency"],
        "浓度": ["掺杂浓度", "mole fraction", "doping concentration"],
        "基质": ["host material", "晶格", "lattice"],
        "荧光": ["发光", "luminescence", "photoluminescence"],
        "猝灭": ["浓度猝灭", "quenching", "荧光猝灭"],
    }

    # 中文停用词 (查询噪声词)
    _STOPWORDS: set[str] = {
        "的", "是", "了", "在", "和", "与", "及", "以及", "并", "同时",
        "还有", "而且", "并且", "有", "为", "对", "等", "个", "这", "那",
        "也", "都", "就", "还", "又", "很", "非常", "什么", "怎么", "如何",
        "为什么", "哪些", "哪个", "是否", "吗", "呢", "啊", "吧", "呀",
        "关于", "对于", "请问", "请", "能", "可以", "应该", "会", "被",
        "把", "给", "向", "从", "到", "于", "之", "其", "此", "该",
        "一种", "一个", "一些", "中", "上", "下", "里", "外", "间",
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and",
        "or", "is", "are", "was", "were", "be", "been", "what", "how",
        "why", "which", "does", "do", "can", "could", "would", "should",
    }

    # 复合查询分隔符 (子问题分解用)
    _CONJUNCTION_PATTERN: re.Pattern[str] = re.compile(
        r"以及|同时|还有|而且|并且|和|与|及|并"
    )
    _SENTENCE_DELIMITER_PATTERN: re.Pattern[str] = re.compile(r"[？?；;]")

    # 词元提取: CJK 连续段 + 字母数字 (含连字符/斜杠) 段
    _TOKEN_PATTERN: re.Pattern[str] = re.compile(
        r"[\u4e00-\u9fff]+|[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*"
    )

    # 标点与空白清理
    _PUNCT_PATTERN: re.Pattern[str] = re.compile(
        r"[，。、；：？！,.;:?!()\[\]{}\"'《》【】\s]+"
    )

    def __init__(
        self, *, domain_dict: dict[str, list[str]] | None = None
    ) -> None:
        """初始化查询重写引擎.

        Args:
            domain_dict: 自定义领域词典 {术语: [同义词...]}。
                None 时使用内置稀土发光材料领域词典。
        """
        self._lock = threading.RLock()
        self._domain_dict: dict[str, list[str]] = {}
        # 使用深拷贝避免外部修改内置字典
        base = domain_dict if domain_dict is not None else self.DEFAULT_DOMAIN_DICT
        for term, syns in base.items():
            self._domain_dict[term] = list(syns)
        self._rebuild_vocab()

    # --------------------------------------------------------
    # 内部辅助
    # --------------------------------------------------------

    def _rebuild_vocab(self) -> None:
        """重建领域词汇全集 (术语 + 同义词), 按长度降序排列便于最长匹配.

        调用方需持有 self._lock。
        """
        vocab: set[str] = set()
        for term, syns in self._domain_dict.items():
            vocab.add(term)
            for s in syns:
                vocab.add(s)
        # 按长度降序: 优先匹配长术语, 避免短术语截断长术语
        self._domain_vocab: list[str] = sorted(vocab, key=len, reverse=True)

    def _match_domain_terms(self, text: str) -> list[str]:
        """在文本中匹配领域词汇 (最长匹配, 去重保序).

        Args:
            text: 待匹配文本

        Returns:
            匹配到的领域词汇列表 (按出现顺序, 去重)
        """
        matched: list[str] = []
        seen: set[str] = set()
        remaining = text
        for term in self._domain_vocab:
            if term in remaining:
                if term not in seen:
                    matched.append(term)
                    seen.add(term)
                # 移除已匹配区间避免子串重复匹配
                remaining = remaining.replace(term, " ")
        return matched

    # --------------------------------------------------------
    # 关键词提取
    # --------------------------------------------------------

    def extract_keywords(self, query: str) -> list[str]:
        """提取查询关键词 (借鉴 spaCy 关键词提取 + 领域词典匹配).

        提取策略:
        1. 优先匹配领域词汇 (术语+同义词, 最长匹配)
        2. 提取剩余 CJK 连续段 (长度 >= 2) 与字母数字词元
        3. 过滤停用词与单字符噪声

        Args:
            query: 查询文本

        Returns:
            关键词列表 (按出现顺序, 去重)
        """
        if not query or not query.strip():
            return []

        with self._lock:
            vocab = list(self._domain_vocab)
            domain_dict = {k: list(v) for k, v in self._domain_dict.items()}

        # 清理标点 (替换为空格, 保留词元边界)
        text = self._PUNCT_PATTERN.sub(" ", query)

        keywords: list[str] = []
        seen: set[str] = set()
        remaining = text

        # 1. 优先匹配领域词汇 (最长匹配)
        for term in vocab:
            if term in remaining:
                if term.lower() not in seen:
                    keywords.append(term)
                    seen.add(term.lower())
                remaining = remaining.replace(term, " ")

        # 2. 提取剩余词元
        for tok in self._TOKEN_PATTERN.findall(remaining):
            if len(tok) < 2:
                continue
            low = tok.lower()
            if low in self._STOPWORDS or tok in self._STOPWORDS:
                continue
            if low not in seen:
                keywords.append(tok)
                seen.add(low)

        return keywords

    # --------------------------------------------------------
    # 同义词扩展
    # --------------------------------------------------------

    def expand_synonyms(self, query: str) -> str:
        """同义词扩展 (借鉴 Elasticsearch synonym filter).

        对查询中出现的领域术语, 追加其同义词作为扩展上下文,
        缩小查询与文档之间的词汇鸿沟。

        Args:
            query: 原始查询

        Returns:
            扩展后的查询文本 (原查询 + 同义词扩展标注)
        """
        if not query or not query.strip():
            return query

        with self._lock:
            domain_dict = {k: list(v) for k, v in self._domain_dict.items()}

        expansions: list[str] = []
        for term, syns in domain_dict.items():
            if term in query:
                # 仅追加尚未出现在原查询中的同义词
                new_syns = [s for s in syns if s not in query]
                if new_syns:
                    expansions.append(f"({term} OR {' OR '.join(new_syns)})")

        if not expansions:
            return query.strip()
        return f"{query.strip()} {' '.join(expansions)}"

    # --------------------------------------------------------
    # 子问题分解
    # --------------------------------------------------------

    def decompose(self, query: str) -> list[str]:
        """子问题分解 (借鉴 LlamaIndex SubQuestionQueryEngine).

        将复合查询按连词 (和/与/及/以及/并/同时/还有/而且/并且)
        与句子分隔符 (？?；;) 拆解为独立子问题, 支持并行检索与分别回答。

        Args:
            query: 复合查询文本

        Returns:
            子问题列表; 若查询不可分解则返回包含原查询的单元素列表
        """
        if not query or not query.strip():
            return []

        text = query.strip()

        # 第一级: 句子分隔符 (问号/分号)
        segments = self._SENTENCE_DELIMITER_PATTERN.split(text)

        # 第二级: 连词拆分
        sub_queries: list[str] = []
        for seg in segments:
            parts = self._CONJUNCTION_PATTERN.split(seg)
            for p in parts:
                p = p.strip()
                if p and len(p) >= 2:
                    sub_queries.append(p)

        # 不可分解: 返回原查询
        if not sub_queries:
            return [text] if text else []
        return sub_queries

    # --------------------------------------------------------
    # HyDE 假设文档生成
    # --------------------------------------------------------

    def generate_hyde(self, query: str) -> str:
        """生成假设文档 (借鉴 HyDE, Gao et al. 2022).

        HyDE 核心思想: 先为查询生成一个假设性答案文档, 再对该文档进行向量化
        用于检索。假设文档无需事实正确, 只需落入正确的语义空间, 从而缩小
        查询 (短问句) 与文档 (长段落) 之间的语义鸿沟。

        本实现采用领域模板 + 关键词填充的确定性生成 (不依赖外部 LLM):
        - 提取查询关键词与匹配的领域术语
        - 套用稀土发光材料领域的典型表述模板
        - 生成一段语义相关的假设性研究叙述

        Args:
            query: 原始查询

        Returns:
            假设性答案文档文本 (用于后续向量化检索)
        """
        if not query or not query.strip():
            return query

        keywords = self.extract_keywords(query)
        with self._lock:
            domain_dict = {k: list(v) for k, v in self._domain_dict.items()}

        # 收集匹配的领域术语及其前 2 个同义词 (构建语义上下文)
        context_terms: list[str] = []
        seen: set[str] = set()
        for term in domain_dict:
            if term in query:
                if term not in seen:
                    context_terms.append(term)
                    seen.add(term)
                for s in domain_dict[term][:2]:
                    if s not in seen:
                        context_terms.append(s)
                        seen.add(s)

        kw_str = "、".join(keywords) if keywords else query.strip()
        ctx_str = "、".join(context_terms) if context_terms else "发光材料"

        # 领域模板: 模拟稀土发光材料研究文献的典型表述
        doc = (
            f"关于{kw_str}的研究表明, 该体系在稀土发光材料领域具有重要意义。"
            f"相关研究通常围绕{ctx_str}等核心概念展开, "
            f"关注其光谱特性、能级结构与发光性能之间的内在联系。"
            f"通过调控掺杂浓度、优化基质组分以及分析跃迁机理, "
            f"可系统揭示该体系的发光效率与猝灭行为规律。"
        )
        return doc

    # --------------------------------------------------------
    # 查询扩展 (综合策略)
    # --------------------------------------------------------

    def _expand_query(self, query: str) -> str:
        """查询扩展综合策略 (借鉴 LangChain MultiQueryRetriever).

        融合同义词扩展与关键词补充:
        1. 同义词扩展 (expand_synonyms)
        2. 追加提取的关键词作为相关术语 (尚未出现在扩展查询中的)

        Args:
            query: 原始查询

        Returns:
            扩展后的查询文本
        """
        base = self.expand_synonyms(query)
        keywords = self.extract_keywords(query)
        extra = [kw for kw in keywords if kw not in base]
        if extra:
            return f"{base} 相关术语: {', '.join(extra)}"
        return base

    # --------------------------------------------------------
    # 上下文压缩
    # --------------------------------------------------------

    def _contextual_compress(self, query: str) -> str:
        """上下文压缩 (借鉴 LangChain ContextualCompressionRetriever).

        提取查询的核心意图: 仅保留关键词, 去除停用词与噪声词,
        生成精简查询以提升检索精确率。

        Args:
            query: 原始查询

        Returns:
            压缩后的核心查询
        """
        keywords = self.extract_keywords(query)
        if not keywords:
            return query.strip()
        return " ".join(keywords)

    # --------------------------------------------------------
    # 统一重写入口
    # --------------------------------------------------------

    def rewrite(
        self, query: str, *, strategy: RewriteStrategy = RewriteStrategy.EXPAND
    ) -> RewrittenQuery:
        """按指定策略重写查询.

        Args:
            query: 原始查询
            strategy: 重写策略, 默认 EXPAND (查询扩展)

        Returns:
            RewrittenQuery 重写结果 (含重写文本、子查询、置信度、元信息)
        """
        if not query or not query.strip():
            return RewrittenQuery(
                original=query or "",
                rewritten=query or "",
                strategy=strategy,
                sub_queries=[],
                confidence=0.0,
                metadata={"empty": True},
            )

        text = query.strip()

        if strategy == RewriteStrategy.SYNONYM:
            rewritten = self.expand_synonyms(text)
            with self._lock:
                matched = [t for t in self._domain_dict if t in text]
            # 置信度: 匹配术语越多, 同义词扩展越有效
            confidence = min(1.0, 0.4 + 0.15 * len(matched))
            metadata: dict[str, Any] = {
                "matched_terms": matched,
                "expansion_applied": rewritten != text,
            }
            return RewrittenQuery(
                original=text, rewritten=rewritten, strategy=strategy,
                sub_queries=[], confidence=confidence, metadata=metadata,
            )

        if strategy == RewriteStrategy.DECOMPOSE:
            sub_queries = self.decompose(text)
            # 置信度: 子问题越多, 分解越有意义; 单一问题置信度较低
            if len(sub_queries) > 1:
                confidence = min(1.0, 0.4 + 0.2 * len(sub_queries))
            else:
                confidence = 0.4
            metadata = {
                "sub_query_count": len(sub_queries),
                "decomposed": len(sub_queries) > 1,
            }
            # 重写文本: 用分隔符串联子问题
            rewritten = " || ".join(sub_queries) if sub_queries else text
            return RewrittenQuery(
                original=text, rewritten=rewritten, strategy=strategy,
                sub_queries=sub_queries, confidence=confidence, metadata=metadata,
            )

        if strategy == RewriteStrategy.HYDE:
            rewritten = self.generate_hyde(text)
            keywords = self.extract_keywords(text)
            # 置信度: 关键词越丰富, 假设文档语义越聚焦
            confidence = min(1.0, 0.5 + 0.08 * len(keywords))
            metadata = {
                "keywords": keywords,
                "doc_length": len(rewritten),
            }
            return RewrittenQuery(
                original=text, rewritten=rewritten, strategy=strategy,
                sub_queries=[], confidence=confidence, metadata=metadata,
            )

        if strategy == RewriteStrategy.CONTEXTUAL:
            rewritten = self._contextual_compress(text)
            keywords = self.extract_keywords(text)
            # 置信度: 提取的关键词越多, 压缩后意图越清晰
            confidence = min(1.0, 0.4 + 0.1 * len(keywords))
            metadata = {
                "keywords": keywords,
                "compressed": rewritten != text,
                "original_length": len(text),
                "compressed_length": len(rewritten),
            }
            return RewrittenQuery(
                original=text, rewritten=rewritten, strategy=strategy,
                sub_queries=[], confidence=confidence, metadata=metadata,
            )

        # 默认: EXPAND
        rewritten = self._expand_query(text)
        with self._lock:
            matched = [t for t in self._domain_dict if t in text]
        keywords = self.extract_keywords(text)
        # 置信度: 综合同义词匹配数与关键词数
        confidence = min(1.0, 0.4 + 0.1 * len(matched) + 0.05 * len(keywords))
        metadata = {
            "matched_terms": matched,
            "keywords": keywords,
            "expansion_applied": rewritten != text,
        }
        return RewrittenQuery(
            original=text, rewritten=rewritten, strategy=strategy,
            sub_queries=[], confidence=confidence, metadata=metadata,
        )

    def rewrite_multi(
        self,
        query: str,
        *,
        strategies: list[RewriteStrategy] | None = None,
    ) -> list[RewrittenQuery]:
        """多策略重写 (借鉴 LangChain MultiQueryRetriever 多查询生成).

        对同一查询应用多种重写策略, 生成多个查询变体。
        多变体可用于多路并行检索 + 结果融合, 提升整体召回率。

        Args:
            query: 原始查询
            strategies: 要应用的策略列表, None 时使用全部 5 种策略

        Returns:
            重写结果列表 (每个策略一个 RewrittenQuery)
        """
        if strategies is None:
            strategies = list(RewriteStrategy)
        return [self.rewrite(query, strategy=s) for s in strategies]

    # --------------------------------------------------------
    # 领域词典管理
    # --------------------------------------------------------

    def add_domain_term(self, term: str, synonyms: list[str]) -> None:
        """添加或更新领域术语及其同义词.

        Args:
            term: 领域术语
            synonyms: 同义词列表
        """
        with self._lock:
            self._domain_dict[term] = list(synonyms)
            self._rebuild_vocab()
        logger.debug("添加领域术语: %s (%d 个同义词)", term, len(synonyms))

    @property
    def domain_dict(self) -> dict[str, list[str]]:
        """返回领域词典的副本 (线程安全)."""
        with self._lock:
            return {k: list(v) for k, v in self._domain_dict.items()}

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._domain_dict)
        return f"QueryRewriter(domain_terms={n})"


__all__ = [
    "RewriteStrategy",
    "RewrittenQuery",
    "QueryRewriter",
]
