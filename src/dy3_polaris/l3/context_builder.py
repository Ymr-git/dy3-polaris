"""L3 领域知识层 -- 上下文构建器.

融合世界先进方案的上下文构建设计:
- ReCAP (2025): 递归上下文感知推理 -- 父任务再注入 + 滑动窗口记忆
- LlamaIndex QueryPipeline: 多阶段查询变换 + 上下文累积
- Agentic RAG: OODA 循环 -- Orient 阶段的上下文评估与按需补充
- Context Recycling (2026): 五层记忆架构 -- 工作记忆显式管理
- Self-RAG: [Retrieve] Token -- 自主判断是否需要额外检索
- Plan-and-Solve: 计划作为上下文骨架

核心职责:
  将用户查询、学习者画像、对话历史、知识图谱 Schema 等
  异构信息源组装成结构化 QueryContext, 供 IntentRouter 消费。

设计原则:
  1. 零依赖外部 LLM -- 规则 + 模板实现, 保证可测试与可复现
  2. 预算感知 -- Token 预算管理, 避免上下文爆炸
  3. 学习者感知 -- 画像驱动的上下文裁剪与优先级排序
  4. 对话感知 -- 基于历史轮次的上下文压缩与指代消解
  5. 领域感知 -- KG Schema 注入, 增强意图识别准确度

Usage::

    from dy3_polaris.l3.context_builder import ContextBuilder, QueryContext
    from dy3_polaris.l3.api_models import LearnerProfile

    builder = ContextBuilder()
    ctx = builder.build(
        query="Dy3+的4F9/2能级跃迁波长是多少?",
        learner_profile=profile,
        dialog_history=[...],
    )
    # ctx 被传入 IntentRouter.route()
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


# ============================================================
# 上下文预算管理
# ============================================================


@dataclass(frozen=True)
class ContextBudget:
    """上下文 Token 预算分配.

    借鉴 Kaman Research (2026) 自适应上下文管理:
    - 系统提示: 10-15%
    - 工具定义: 15-20%
    - 检索知识: 30-40%
    - 对话历史: 10-20%
    - 输出预留: 25-50%

    本模块控制的是 "检索知识 + 对话历史" 部分。

    Attributes:
        max_tokens: 总 Token 上限 (字符数近似, 1 中文 ≈ 2 token)
        query_ratio: 查询重写结果占比
        history_ratio: 对话历史占比
        learner_ratio: 学习者画像占比
        schema_ratio: KG Schema 占比
        retrieval_ratio: 检索结果预留占比
    """

    max_tokens: int = 4096
    query_ratio: float = 0.10
    history_ratio: float = 0.20
    learner_ratio: float = 0.05
    schema_ratio: float = 0.05
    retrieval_ratio: float = 0.60

    def budget_for(self, key: str) -> int:
        """获取指定部分的 Token 预算."""
        ratios: dict[str, float] = {
            "query": self.query_ratio,
            "history": self.history_ratio,
            "learner": self.learner_ratio,
            "schema": self.schema_ratio,
            "retrieval": self.retrieval_ratio,
        }
        ratio = ratios.get(key, 0.0)
        return int(self.max_tokens * ratio)


# ============================================================
# 对话历史压缩
# ============================================================


class HistoryCompressStrategy(str, Enum):
    """对话历史压缩策略.

    借鉴 SWE-agent Observation Masking + OpenHands LLM 摘要:
    - RECENT: 仅保留最近 N 轮 (Observation Masking)
    - SUMMARIZE: 压缩旧轮次为摘要 (LLM 摘要, 本实现用规则)
    - SLIDING_WINDOW: 滑动窗口, 保留最近 N 个字符
    """

    RECENT = "recent"
    SUMMARIZE = "summarize"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class DialogTurn:
    """单轮对话记录."""

    role: str  # "user" / "assistant" / "system"
    content: str
    timestamp: float = 0.0


class HistoryCompressor:
    """对话历史压缩器.

    借鉴 Context Recycling (2026) 的工作记忆管理:
    - 父任务再注入: 始终保留首轮对话 (原始意图不丢失)
    - 滑动窗口: 保留最近 N 轮完整内容
    - 旧轮次压缩: 超出窗口的历史压缩为摘要行
    """

    # 压缩后的摘要模板
    _SUMMARY_TEMPLATE = (
        '[历史摘要: {turn_count}轮对话, 主题涉及{topics}, '
        '最近查询"{last_query}"]'
    )

    def __init__(
        self,
        *,
        strategy: HistoryCompressStrategy = HistoryCompressStrategy.RECENT,
        max_recent_turns: int = 5,
        max_chars: int = 800,
    ) -> None:
        self._strategy = strategy
        self._max_recent = max_recent_turns
        self._max_chars = max_chars

    def compress(self, turns: list[DialogTurn]) -> list[DialogTurn]:
        """压缩对话历史.

        Returns:
            压缩后的对话列表 (可能包含摘要行)
        """
        if not turns:
            return []

        if self._strategy == HistoryCompressStrategy.RECENT:
            return self._compress_recent(turns)
        if self._strategy == HistoryCompressStrategy.SLIDING_WINDOW:
            return self._compress_sliding_window(turns)
        # SUMMARIZE
        return self._compress_summarize(turns)

    def _compress_recent(self, turns: list[DialogTurn]) -> list[DialogTurn]:
        """保留最近 N 轮 + 首轮 (父任务再注入)."""
        if len(turns) <= self._max_recent:
            return list(turns)
        # 始终保留首轮 (ReCAP: 父任务再注入)
        first = turns[:1]
        recent = turns[-self._max_recent:]
        return first + recent

    def _compress_sliding_window(self, turns: list[DialogTurn]) -> list[DialogTurn]:
        """滑动窗口: 保留最近 N 个字符内的内容."""
        budget = self._max_chars
        result: list[DialogTurn] = []
        # 从最新开始, 逐轮填充预算
        for turn in reversed(turns):
            turn_chars = len(turn.content)
            if budget - turn_chars < 0:
                break
            result.append(turn)
            budget -= turn_chars
        result.reverse()
        return result

    def _compress_summarize(self, turns: list[DialogTurn]) -> list[DialogTurn]:
        """压缩旧轮次为摘要行 (规则实现, 不依赖 LLM)."""
        if len(turns) <= self._max_recent:
            return list(turns)

        old_turns = turns[:-self._max_recent]
        recent_turns = turns[-self._max_recent:]

        # 提取主题词 (从用户轮次中提取关键词)
        user_queries = [t.content for t in old_turns if t.role == "user"]
        topics = self._extract_topics(user_queries)
        last_q = user_queries[-1][:30] if user_queries else ""

        summary_text = self._SUMMARY_TEMPLATE.format(
            turn_count=len(old_turns),
            topics=topics,
            last_query=last_q,
        )

        summary_turn = DialogTurn(
            role="system",
            content=summary_text,
            timestamp=old_turns[0].timestamp if old_turns else 0.0,
        )
        return [summary_turn] + recent_turns

    def _extract_topics(self, queries: list[str]) -> str:
        """从历史查询中提取主题词 (简单词频统计)."""
        if not queries:
            return ""
        # 合并所有查询
        all_text = " ".join(queries)
        # 提取 CJK 词和英文词
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_]+", all_text)
        # 去重, 取前 5 个高频词
        seen: set[str] = set()
        unique: list[str] = []
        for t in tokens:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                unique.append(t)
            if len(unique) >= 5:
                break
        return "、".join(unique) if unique else ""


# ============================================================
# 指代消解
# ============================================================


class CoreferenceResolver:
    """简单指代消解器.

    借鉴 ReCAP 的上下文感知推理:
    利用前一轮 assistant 回复中的实体, 替换当前查询中的代词。

    规则:
    - "它/这个/那个/其" → 前文最近提到的化学实体 (离子/化学式)
    - "该XX/这个XX" → 查找前文以 XX 开头的实体
    """

    # 代词模式
    _PRONOUN_PATTERN = re.compile(
        r"(?:它|这个|那个|其|该|此)"
    )

    # 化学实体提取 (复用 EntityExtractor 的正则)
    _ION_PATTERN = re.compile(r"\b([A-Z][a-z]?\d*[+-])\b")
    _FORMULA_PATTERN = re.compile(r"\b((?:[A-Z][a-z]?\d*){2,})\b")

    def resolve(
        self,
        query: str,
        dialog_history: list[DialogTurn],
    ) -> str:
        """消解查询中的指代.

        Args:
            query: 当前查询
            dialog_history: 对话历史 (用于提取前文实体)

        Returns:
            消解后的查询 (代词被替换为实际实体)
        """
        if not dialog_history or not self._PRONOUN_PATTERN.search(query):
            return query

        # 从历史中提取最近的化学实体 (assistant 回复优先)
        recent_entity = self._find_recent_entity(dialog_history)
        if not recent_entity:
            return query

        # 替换代词 (仅替换第一个匹配)
        resolved = self._PRONOUN_PATTERN.sub(recent_entity, query, count=1)
        if resolved != query:
            logger.debug("指代消解: %s → %s", query, resolved)
        return resolved

    def _find_recent_entity(
        self, turns: list[DialogTurn]
    ) -> str | None:
        """从历史中查找最近的化学实体."""
        for turn in reversed(turns):
            if turn.role != "assistant":
                continue
            # 优先查离子
            m = self._ION_PATTERN.search(turn.content)
            if m:
                return m.group()
            # 其次查化学式
            m = self._FORMULA_PATTERN.search(turn.content)
            if m:
                return m.group()
        # 最后查用户消息
        for turn in reversed(turns):
            if turn.role != "user":
                continue
            m = self._ION_PATTERN.search(turn.content)
            if m:
                return m.group()
            m = self._FORMULA_PATTERN.search(turn.content)
            if m:
                return m.group()
        return None


# ============================================================
# KG Schema 上下文注入
# ============================================================


class SchemaContextInjector:
    """KG Schema 上下文注入器.

    借鉴 SEAL (2025) 的 Agent 校准模块:
    将 KG Schema 信息注入查询上下文, 帮助意图分类器更准确地
    将自然语言意图映射到图查询。

    注入内容:
    - 领域本体中的实体类型列表
    - 关系类型列表
    - 属性约束 (用于数值意图检测)
    """

    # 实体类型 → 中文描述
    _ENTITY_TYPE_LABELS: dict[str, str] = {
        "concept": "概念/定义",
        "numeric": "数值/参数",
        "relational": "关系/路径",
    }

    def __init__(self) -> None:
        self._domain_info: dict[str, dict[str, Any]] = {}
        self._init_default_domains()

    def _init_default_domains(self) -> None:
        """初始化默认领域信息."""
        self._domain_info = {
            "chemistry": {
                "entity_types": [
                    "chemical_element", "chemical_compound",
                    "ion", "mineral", "material",
                ],
                "relation_types": [
                    "doped_in", "emits_at", "absorbs_at",
                    "excited_by", "quenched_by", "depends_on",
                ],
                "numeric_properties": [
                    "wavelength", "concentration", "temperature",
                    "quantum_efficiency", "lifetime", "energy",
                ],
            },
            "education": {
                "entity_types": [
                    "knowledge_point", "course", "prerequisite",
                    "learning_objective", "assessment",
                ],
                "relation_types": [
                    "prerequisite_of", "tested_by",
                    "belongs_to", "requires", "teaches",
                ],
                "numeric_properties": [
                    "difficulty", "mastery", "credit_hours",
                ],
            },
        }

    def inject(self, query: str, domain: str = "chemistry") -> str:
        """将领域 Schema 信息注入查询上下文.

        不直接修改查询文本, 而是返回 Schema 上下文片段,
        由 ContextBuilder 统一管理。

        Args:
            query: 原始查询 (用于检测相关领域)
            domain: 目标领域

        Returns:
            Schema 上下文文本 (可为空)
        """
        info = self._domain_info.get(domain)
        if info is None:
            return ""

        parts: list[str] = []
        if info.get("entity_types"):
            parts.append(
                f"领域实体类型: {', '.join(info['entity_types'][:8])}"
            )
        if info.get("relation_types"):
            parts.append(
                f"关系类型: {', '.join(info['relation_types'][:8])}"
            )
        if info.get("numeric_properties"):
            parts.append(
                f"可查询数值属性: {', '.join(info['numeric_properties'][:6])}"
            )

        return " | ".join(parts)

    def detect_domain(self, query: str) -> str:
        """基于查询内容检测最可能的领域.

        简单启发式: 基于领域关键词匹配。
        """
        domain_keywords: dict[str, list[str]] = {
            "chemistry": [
                "离子", "跃迁", "波长", "浓度", "掺杂", "猝灭",
                "能级", "激发", "发射", "吸收", "光谱",
                "ion", "transition", "wavelength", "doping",
            ],
            "education": [
                "学习", "考试", "知识点", "课程", "教学",
                "掌握", "难度", "练习", "作业",
                "learning", "exam", "course", "mastery",
            ],
        }
        query_lower = query.lower()
        best_domain = "chemistry"  # 默认领域
        best_score = 0
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_domain = domain
        return best_domain

    def register_domain(
        self, domain: str, info: dict[str, Any]
    ) -> None:
        """注册自定义领域."""
        self._domain_info[domain] = info


# ============================================================
# 学习者适配器
# ============================================================


class LearnerContextAdapter:
    """学习者上下文适配器.

    借鉴 LlamaIndex 的 Contextual Compression + Self-RAG 的按需检索:
    根据学习者画像生成上下文提示片段, 影响:
    1. 意图分类: 薄弱知识点相关查询提升优先级
    2. 查询重写: 根据认知层级选择重写策略
    3. 检索过滤: 注入 Bloom 层级过滤条件

    不依赖外部 LLM, 使用规则 + 模板生成上下文片段。
    """

    # Bloom 层级 → 查询重写策略映射
    _BLOOM_STRATEGY_HINT: dict[str, str] = {
        "remember": "contextual",   # 记忆层: 压缩查询, 精确匹配
        "understand": "expand",      # 理解层: 扩展查询, 增加覆盖
        "apply": "synonym",          # 应用层: 同义词扩展, 桥接术语
        "analyze": "decompose",      # 分析层: 子问题分解
        "evaluate": "decompose",     # 评价层: 子问题分解 + 对比
        "create": "expand",          # 创造层: 查询扩展, 探索关联
    }

    # Bloom 层级 → 检索深度建议
    _BLOOM_DEPTH_HINT: dict[str, int] = {
        "remember": 1,
        "understand": 1,
        "apply": 2,
        "analyze": 2,
        "evaluate": 3,
        "create": 3,
    }

    def adapt(self, learner_profile: LearnerProfile | None) -> dict[str, Any]:
        """根据学习者画像生成上下文适配信息.

        Returns:
            适配信息字典, 包含:
            - weak_kp_ids: 薄弱知识点列表
            - bloom_level: 布鲁姆层级
            - suggested_strategy: 建议的查询重写策略
            - suggested_depth: 建议的图检索深度
            - style_preference: 学习风格
            - context_hint: 上下文提示片段
        """
        if learner_profile is None:
            return {
                "weak_kp_ids": [],
                "bloom_level": "understand",
                "suggested_strategy": "expand",
                "suggested_depth": 1,
                "style_preference": "reading",
                "context_hint": "",
            }

        bloom = learner_profile.bloom_target.value
        strategy = self._BLOOM_STRATEGY_HINT.get(bloom, "expand")
        depth = self._BLOOM_DEPTH_HINT.get(bloom, 1)
        style = learner_profile.preferred_style.value

        # 构建上下文提示
        hints: list[str] = []
        if learner_profile.weak_kps:
            hints.append(
                f"学习者有 {len(learner_profile.weak_kps)} 个薄弱知识点, "
                f"优先关注相关知识"
            )
        if learner_profile.level == "beginner":
            hints.append("学习者为初学者, 优先基础概念")
        elif learner_profile.level == "advanced":
            hints.append("学习者为高级, 可深入前沿内容")

        return {
            "weak_kp_ids": list(learner_profile.weak_kps),
            "bloom_level": bloom,
            "suggested_strategy": strategy,
            "suggested_depth": depth,
            "style_preference": style,
            "context_hint": "; ".join(hints),
        }


# ============================================================
# 自我评估: 是否需要额外检索
# ============================================================


class RetrievalNeedAssessor:
    """检索需求评估器.

    借鉴 Self-RAG 的 [Retrieve] Token 机制:
    在实际检索前, 快速评估当前上下文是否足以回答查询,
    避免不必要的检索调用, 降低延迟。

    评估维度:
    1. 上下文覆盖率: 查询关键词在历史上下文中的覆盖率
    2. 实体匹配度: 提取的实体在知识库中是否有对应
    3. 确定性信号: 查询是否包含高确定性答案 (如 "是什么" 定义类)
    """

    # 高确定性查询模式 (可能不需要检索)
    _HIGH_CERTAINTY_PATTERN = re.compile(
        r"(?:你好|hi|hello|谢谢|thanks|再见|bye)"
        r"|(?:你叫什么|你是谁|你能做什么)",
        re.IGNORECASE,
    )

    def assess(
        self,
        query: str,
        context: QueryContext | None = None,
    ) -> bool:
        """评估是否需要执行知识检索.

        Returns:
            True 表示需要检索, False 表示当前上下文可能已足够
        """
        # 检查是否为闲聊/元问题
        if self._HIGH_CERTAINTY_PATTERN.search(query):
            return False

        # 检查查询长度 -- 极短查询可能是闲聊
        if len(query.strip()) < 4:
            return False

        # 如果有历史上下文, 检查覆盖率
        if context and context.dialog_history:
            last_assistant = self._get_last_assistant(context.dialog_history)
            if last_assistant and self._covers_query(query, last_assistant):
                return False

        # 默认: 需要检索
        return True

    def _get_last_assistant(
        self, turns: list[DialogTurn]
    ) -> str | None:
        """获取最后一轮 assistant 回复."""
        for turn in reversed(turns):
            if turn.role == "assistant":
                return turn.content
        return None

    def _covers_query(self, query: str, context_text: str) -> bool:
        """检查上下文是否已覆盖查询 (简单关键词重叠)."""
        # 提取查询中的关键词
        query_keywords = set(
            re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_]+", query)
        )
        if not query_keywords:
            return False
        # 检查覆盖率
        context_lower = context_text.lower()
        covered = sum(
            1 for kw in query_keywords if kw.lower() in context_lower
        )
        return covered / len(query_keywords) >= 0.8


# ============================================================
# 查询上下文 -- 核心数据结构
# ============================================================


@dataclass
class QueryContext:
    """查询上下文 -- ContextBuilder 的输出.

    封装一次查询所需的全部上下文信息, 是 L3 内部
    意图路由和检索的完整输入。

    Attributes:
        context_id: 上下文唯一标识
        original_query: 原始查询文本 (未修改)
        resolved_query: 指代消解后的查询
        rewritten_queries: 查询重写结果列表
        intent_hint: 意图提示 (由上下文推断, 辅助意图分类)
        entities: 提取的实体列表 (实体名称字符串)
        dialog_history: 压缩后的对话历史
        learner_adaptation: 学习者适配信息
        schema_context: KG Schema 上下文片段
        domain: 检测到的领域
        needs_retrieval: 是否需要执行检索
        suggested_top_k: 建议返回结果数
        suggested_depth: 建议的图检索深度
        metadata: 附加元信息
    """

    context_id: str = field(
        default_factory=lambda: f"ctx-{uuid.uuid4().hex[:12]}"
    )
    original_query: str = ""
    resolved_query: str = ""
    rewritten_queries: list[str] = field(default_factory=list)
    intent_hint: str = ""
    entities: list[str] = field(default_factory=list)
    dialog_history: list[DialogTurn] = field(default_factory=list)
    learner_adaptation: dict[str, Any] = field(default_factory=dict)
    schema_context: str = ""
    domain: str = "chemistry"
    needs_retrieval: bool = True
    suggested_top_k: int = 10
    suggested_depth: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def active_query(self) -> str:
        """当前活跃查询 (指代消解后的查询)."""
        return self.resolved_query or self.original_query

    @property
    def build_time_ms(self) -> float:
        """构建耗时 (毫秒)."""
        return self.metadata.get("build_time_ms", 0.0)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (用于调试和日志)."""
        return {
            "context_id": self.context_id,
            "original_query": self.original_query,
            "resolved_query": self.resolved_query,
            "rewritten_queries": self.rewritten_queries,
            "intent_hint": self.intent_hint,
            "entities": self.entities,
            "dialog_turns": len(self.dialog_history),
            "domain": self.domain,
            "needs_retrieval": self.needs_retrieval,
            "suggested_top_k": self.suggested_top_k,
            "suggested_depth": self.suggested_depth,
            "learner_level": self.learner_adaptation.get("bloom_level", ""),
            "build_time_ms": self.build_time_ms,
        }


# ============================================================
# 上下文构建器 -- 主入口
# ============================================================


# LLM 分类器协议 (预留接口, 不强制依赖)
class LLMClassifier(Protocol):
    """LLM 意图分类器协议 (预留接口)."""

    def classify(self, query: str, context: str = "") -> dict[str, Any]: ...


class ContextBuilder:
    """上下文构建器 -- L3 意图理解与上下文构建的主入口.

    融合 ReCAP + Agentic RAG + Self-RAG + Plan-and-Solve 的设计,
    将异构信息源组装为结构化 QueryContext。

    构建流程 (四阶段):
    1. 输入预处理: 指代消解 + 查询清洗
    2. 意图理解: 实体提取 + 领域检测 + Schema 注入
    3. 上下文组装: 对话压缩 + 学习者适配 + 查询重写
    4. 自我评估: 检索需求判断 + 参数建议

    线程安全: 所有组件均为无状态或线程安全的。

    Usage::

        builder = ContextBuilder()
        ctx = builder.build(
            query="Dy3+的4F9/2能级跃迁波长是多少?",
            learner_profile=profile,
            dialog_history=turns,
        )
        # 将 ctx 传入 IntentRouter
    """

    def __init__(
        self,
        *,
        budget: ContextBudget | None = None,
        history_strategy: HistoryCompressStrategy = HistoryCompressStrategy.RECENT,
        max_history_turns: int = 5,
        llm_classifier: LLMClassifier | None = None,
    ) -> None:
        """初始化上下文构建器.

        Args:
            budget: Token 预算配置
            history_strategy: 对话历史压缩策略
            max_history_turns: 最大保留历史轮次
            llm_classifier: 可选的外部 LLM 分类器
        """
        self._budget = budget or ContextBudget()
        self._compressor = HistoryCompressor(
            strategy=history_strategy,
            max_recent_turns=max_history_turns,
            max_chars=budget.budget_for("history") * 2 if budget else 800,
        )
        self._coref = CoreferenceResolver()
        self._schema_injector = SchemaContextInjector()
        self._learner_adapter = LearnerContextAdapter()
        self._need_assessor = RetrievalNeedAssessor()
        self._llm_classifier = llm_classifier

        # 延迟导入 (避免循环依赖)
        self._extractor: Any = None
        self._rewriter: Any = None

    def _get_extractor(self) -> Any:
        """延迟获取 EntityExtractor."""
        if self._extractor is None:
            from .intent_router import EntityExtractor
            self._extractor = EntityExtractor()
        return self._extractor

    def _get_rewriter(self) -> Any:
        """延迟获取 QueryRewriter."""
        if self._rewriter is None:
            from .query_rewriter import QueryRewriter
            self._rewriter = QueryRewriter()
        return self._rewriter

    def build(
        self,
        query: str,
        *,
        learner_profile: Any | None = None,
        dialog_history: list[DialogTurn] | None = None,
        rewrite_strategies: list[str] | None = None,
    ) -> QueryContext:
        """构建查询上下文.

        Args:
            query: 用户查询文本
            learner_profile: 学习者画像 (LearnerProfile 或 None)
            dialog_history: 对话历史 (DialogTurn 列表或 None)
            rewrite_strategies: 要应用的查询重写策略 (None = 自动选择)

        Returns:
            QueryContext 结构化上下文
        """
        start_time = time.time()
        turns = dialog_history or []
        metadata: dict[str, Any] = {}

        # --- Phase 1: 输入预处理 ---
        resolved_query = self._coref.resolve(query, turns)
        metadata["coreference_resolved"] = resolved_query != query

        # --- Phase 2: 意图理解 ---
        # 2a. 实体提取
        extractor = self._get_extractor()
        extracted = extractor.extract(resolved_query)
        entities = [e.text for e in extracted]

        # 2b. 领域检测
        domain = self._schema_injector.detect_domain(resolved_query)

        # 2c. Schema 注入
        schema_ctx = self._schema_injector.inject(resolved_query, domain)

        # 2d. LLM 意图提示 (如果可用)
        intent_hint = ""
        if self._llm_classifier is not None:
            try:
                llm_result = self._llm_classifier.classify(
                    resolved_query, schema_ctx
                )
                intent_hint = llm_result.get("intent_hint", "")
                metadata["llm_classification"] = llm_result
            except Exception:
                logger.warning("LLM 分类器调用失败, 使用规则兜底", exc_info=True)

        # --- Phase 3: 上下文组装 ---
        # 3a. 对话历史压缩
        compressed_history = self._compressor.compress(turns)

        # 3b. 学习者适配
        learner_adapt = self._learner_adapter.adapt(learner_profile)

        # 3c. 查询重写
        rewriter = self._get_rewriter()
        rewritten_queries = self._rewrite_query(
            resolved_query, rewriter, learner_adapt, rewrite_strategies
        )

        # --- Phase 4: 自我评估 ---
        # 先构建一个临时 context 用于评估
        temp_ctx = QueryContext(
            original_query=query,
            resolved_query=resolved_query,
            dialog_history=compressed_history,
            domain=domain,
        )
        needs_retrieval = self._need_assessor.assess(resolved_query, temp_ctx)

        # 参数建议
        suggested_top_k = self._suggest_top_k(learner_adapt, extracted)
        suggested_depth = learner_adapt.get("suggested_depth", 1)

        # 意图提示增强: 基于实体和领域推断
        if not intent_hint:
            intent_hint = self._infer_intent_hint(resolved_query, extracted, domain)

        elapsed = (time.time() - start_time) * 1000
        metadata["build_time_ms"] = round(elapsed, 2)
        metadata["history_compressed"] = len(turns) - len(compressed_history)
        metadata["entities_count"] = len(entities)
        metadata["rewrite_count"] = len(rewritten_queries)

        return QueryContext(
            original_query=query,
            resolved_query=resolved_query,
            rewritten_queries=rewritten_queries,
            intent_hint=intent_hint,
            entities=entities,
            dialog_history=compressed_history,
            learner_adaptation=learner_adapt,
            schema_context=schema_ctx,
            domain=domain,
            needs_retrieval=needs_retrieval,
            suggested_top_k=suggested_top_k,
            suggested_depth=suggested_depth,
            metadata=metadata,
        )

    def _rewrite_query(
        self,
        query: str,
        rewriter: Any,
        learner_adapt: dict[str, Any],
        strategies: list[str] | None,
    ) -> list[str]:
        """执行查询重写.

        如果未指定策略, 根据学习者适配信息自动选择。
        """
        from .query_rewriter import RewriteStrategy

        if strategies is not None:
            # 用户指定策略
            strategyEnums = []
            for s in strategies:
                try:
                    strategyEnums.append(RewriteStrategy(s))
                except ValueError:
                    pass
            if strategyEnums:
                results = rewriter.rewrite_multi(query, strategies=strategyEnums)
                return [r.rewritten for r in results if r.rewritten != query]

        # 自动选择策略
        suggested = learner_adapt.get("suggested_strategy", "expand")
        try:
            strategy = RewriteStrategy(suggested)
        except ValueError:
            strategy = RewriteStrategy.EXPAND

        result = rewriter.rewrite(query, strategy=strategy)
        rewritten = [result.rewritten] if result.rewritten != query else []

        # 对复合查询, 额外尝试分解策略
        if len(query) > 20 and ("及" in query or "和" in query):
            decomp = rewriter.rewrite(query, strategy=RewriteStrategy.DECOMPOSE)
            if decomp.sub_queries and decomp.sub_queries != [query]:
                rewritten.extend(decomp.sub_queries)

        return rewritten

    def _suggest_top_k(
        self,
        learner_adapt: dict[str, Any],
        entities: list[Any],
    ) -> int:
        """建议返回结果数."""
        base = 10
        # 初学者: 少而精
        if learner_adapt.get("bloom_level") in ("remember", "understand"):
            base = 5
        # 高级学习者: 多而全
        elif learner_adapt.get("bloom_level") in ("analyze", "evaluate", "create"):
            base = 15
        # 有薄弱知识点: 适当增加以覆盖更多相关知识
        if learner_adapt.get("weak_kp_ids"):
            base = min(base + 3, 20)
        return base

    def _infer_intent_hint(
        self,
        query: str,
        entities: list[Any],
        domain: str,
    ) -> str:
        """基于查询内容和实体推断意图提示."""
        hints: list[str] = []

        # 数值实体 → numeric 提示
        has_numeric = any(e.entity_type == "numeric" for e in entities)
        if has_numeric:
            hints.append("numeric")

        # 化学实体 → 可能是 relational
        has_chemical = any(
            e.entity_type in ("ion", "formula", "spectral_term")
            for e in entities
        )
        if has_chemical:
            hints.append("relational")

        # 长查询 → 可能是 composite
        if len(query) > 30:
            hints.append("composite")

        # 默认 concept
        if not hints:
            hints.append("concept")

        return "+".join(hints)


__all__ = [
    # 数据结构
    "ContextBudget",
    "DialogTurn",
    "HistoryCompressStrategy",
    "QueryContext",
    # 核心组件
    "HistoryCompressor",
    "CoreferenceResolver",
    "SchemaContextInjector",
    "LearnerContextAdapter",
    "RetrievalNeedAssessor",
    # 主入口
    "ContextBuilder",
    # 协议
    "LLMClassifier",
]
