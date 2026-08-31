"""L3 领域知识层 — 意图驱动路由检索引擎.

融合世界先进方案的意图识别与路由设计:
- LangChain Router: 意图分类 + 多路检索路由
- LlamaIndex Router Query Engine: 意图驱动查询引擎
- Weaviate Hybrid Search: 向量+关键词+图混合检索
- GraphRAG: 社区感知检索 + 子图提取
- Cohere Command-R: 意图分类 + 工具选择
- ReCAP (2025): 递归上下文感知推理
- Self-RAG: [Retrieve] Token 自主检索决策
- Plan-and-Solve: 计划驱动的意图分解

四类意图路由:
1. concept  (概念检索) → 向量检索 + 关键词检索 → RRF 融合
2. numeric   (数值检索) → 精确查询 + 结构化过滤 → 事实校验
3. relational(关系检索) → 图遍历 + 子图提取 → 路径推理
4. composite (复合检索) → 三路并行 + RRF 融合 + 事实校验

上下文增强路由 (v2):
- ContextBuilder 预构建: 指代消解 + 领域检测 + 学习者适配
- 意图提示融合: 上下文推断的 intent_hint 辅助分类
- 多查询变体路由: 利用重写结果多路并行检索
- 自适应参数: 根据学习者画像调整 top_k 和图深度
- 检索跳过: needs_retrieval=False 时直接返回空结果

意图识别策略:
- 规则优先 (高频模式 <10ms): 正则匹配 + 关键词匹配
- LLM 兜底 (模糊查询 <100ms): 外部 LLM 分类 (预留接口)
- 优先级: numeric > relational > concept > composite

实体提取:
- 离子符号: Dy3+, Eu3+, Tb3+ 等 (正则 [A-Z][a-z]?\\d*[+-])
- 化学式: YAG, Y2O3, YVO4 等 (正则 [A-Z][a-z]?\\d*(?:[A-Z][a-z]?\\d*)*)
- 光谱项: 4F9/2, 5D0 等 (正则 \\d[A-Z]\\d+/\\d+)
- 数值+单位: 580nm, 5.5e-19 J, 300K 等
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import RetrievalError
from .models import RetrievalFilter, RetrievalResult
from .store import KnowledgeStore

logger = logging.getLogger(__name__)


# ============================================================
# 意图类型定义
# ============================================================


class IntentType(str, Enum):
    """查询意图类型 (借鉴 LangChain Router + Cohere intent classification).

    CONCEPT: 概念检索 — 查询概念定义、原理、机理等
    NUMERIC: 数值检索 — 查询具体数值、参数、标准值等
    RELATIONAL: 关系检索 — 查询实体间关系、路径、影响等
    COMPOSITE: 复合检索 — 同时包含多种意图
    """

    CONCEPT = "concept"
    NUMERIC = "numeric"
    RELATIONAL = "relational"
    COMPOSITE = "composite"


@dataclass
class ExtractedEntity:
    """提取的实体信息 (借鉴 spaCy NER + 领域正则)."""

    text: str
    entity_type: str  # ion / formula / spectral_term / numeric / keyword
    value: str | float | None = None
    unit: str | None = None
    start: int = 0
    end: int = 0


@dataclass
class IntentResult:
    """意图识别结果 (借鉴 Cohere intent classification).

    Attributes:
        intent_type: 识别的意图类型
        confidence: 置信度 (0-1)
        matched_rules: 匹配的规则列表
        extracted_entities: 提取的实体列表
        suggested_path: 建议的检索路径
        classification_time_ms: 分类耗时 (毫秒)
    """

    intent_type: IntentType
    confidence: float
    matched_rules: list[str] = field(default_factory=list)
    extracted_entities: list[ExtractedEntity] = field(default_factory=list)
    suggested_path: str = ""
    classification_time_ms: float = 0.0


# ============================================================
# 实体提取器 — 领域命名实体识别
# ============================================================


class EntityExtractor:
    """领域实体提取器 (借鉴 spaCy NER + ChemDataExtractor + 正则规则).

    支持提取:
    - 离子符号: Dy3+, Eu3+, Tb3+, Sm3+ 等
    - 化学式: YAG, Y2O3, YVO4, BaMgAl10O17 等
    - 光谱项: 4F9/2, 5D0, 7F0 等
    - 数值+单位: 580nm, 5.5e-19 J, 300K, 1.5 mol% 等
    - 关键词: 激发态, 跃迁, 发光, 猝灭, 能量传递 等
    """

    # 离子符号正则: 元素符号 + 电荷 (Dy3+, Eu2+, Ce3+)
    ION_PATTERN = re.compile(
        r"\b([A-Z][a-z]?\d*[+-])\b"
    )

    # 化学式正则: 元素符号 + 数字 (Y2O3, BaMgAl10O17)
    FORMULA_PATTERN = re.compile(
        r"\b((?:[A-Z][a-z]?\d*){2,})\b"
    )

    # 光谱项正则: 数字+字母+数字/数字 (4F9/2, 5D0, 7F1)
    SPECTRAL_TERM_PATTERN = re.compile(
        r"\b(\d[A-Z]\d+\/\d+)\b|\b(\d[A-Z]\d+)\b"
    )

    # 数值+单位正则: 数字 + 可选科学计数法 + 单位
    NUMERIC_UNIT_PATTERN = re.compile(
        r"(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*"
        r"(nm|cm[-‐]?1|K|mol%|mol/L|eV|J|meV|wt%|at%|mol|g/cm[³3]|"
        r"lux|cd/m[²2]|lm/W|nm|μs|ms|s|Hz|kHz|MHz|GHz|"
        r"Å|pm|nm|μm|mm|cm|m)\b",
        re.IGNORECASE,
    )

    # 领域关键词
    DOMAIN_KEYWORDS = {
        "mechanism": [
            "机理", "机制", "原理", "机制", "过程", "途径",
            "mechanism", "process", "pathway", "principle",
        ],
        "relationship": [
            "关系", "关联", "影响", "作用", "连接", "依赖",
            "先修", "前置", "后续", "因果",
            "relation", "relationship", "affect", "influence",
            "cause", "depend", "prerequisite",
        ],
        "numeric": [
            "值", "参数", "浓度", "温度", "波长", "效率",
            "能量", "距离", "寿命", "量子效率",
            "value", "parameter", "concentration", "temperature",
            "wavelength", "efficiency", "energy", "distance",
            "lifetime", "quantum efficiency",
        ],
        "definition": [
            "是什么", "什么是", "定义", "概念", "含义",
            "what is", "define", "definition", "concept", "meaning",
        ],
        "comparison": [
            "比较", "对比", "区别", "差异", "优劣",
            "compare", "comparison", "difference", "versus", "vs",
        ],
    }

    def extract(self, query: str) -> list[ExtractedEntity]:
        """从查询文本中提取领域实体.

        Args:
            query: 查询文本

        Returns:
            提取的实体列表 (按位置排序)
        """
        entities: list[ExtractedEntity] = []

        # 提取离子符号
        for match in self.ION_PATTERN.finditer(query):
            entities.append(ExtractedEntity(
                text=match.group(),
                entity_type="ion",
                value=match.group(),
                start=match.start(),
                end=match.end(),
            ))

        # 提取化学式 (排除已被识别为离子的)
        ion_spans = {(e.start, e.end) for e in entities if e.entity_type == "ion"}
        for match in self.FORMULA_PATTERN.finditer(query):
            # 跳过与离子重叠的匹配
            if any(
                match.start() < end and match.end() > start
                for start, end in ion_spans
            ):
                continue
            entities.append(ExtractedEntity(
                text=match.group(),
                entity_type="formula",
                value=match.group(),
                start=match.start(),
                end=match.end(),
            ))

        # 提取光谱项
        for match in self.SPECTRAL_TERM_PATTERN.finditer(query):
            text = match.group(1) or match.group(2)
            if text:
                entities.append(ExtractedEntity(
                    text=text,
                    entity_type="spectral_term",
                    value=text,
                    start=match.start(),
                    end=match.end(),
                ))

        # 提取数值+单位
        for match in self.NUMERIC_UNIT_PATTERN.finditer(query):
            value_str = match.group(1)
            unit = match.group(2).lower().replace("‐", "-").replace("-", "")
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
            entities.append(ExtractedEntity(
                text=match.group(),
                entity_type="numeric",
                value=value,
                unit=unit,
                start=match.start(),
                end=match.end(),
            ))

        # 提取领域关键词
        query_lower = query.lower()
        for category, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                idx = query_lower.find(kw.lower())
                if idx >= 0:
                    entities.append(ExtractedEntity(
                        text=query[idx:idx + len(kw)],
                        entity_type="keyword",
                        value=kw,
                        start=idx,
                        end=idx + len(kw),
                    ))

        # 按位置排序并去重
        entities.sort(key=lambda e: e.start)
        return self._deduplicate(entities)

    def _deduplicate(
        self, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        """去重叠实体 (保留最长的匹配)."""
        if not entities:
            return []

        result: list[ExtractedEntity] = []
        used_spans: list[tuple[int, int]] = []

        for entity in entities:
            # 检查是否与已选实体重叠
            overlap = False
            for start, end in used_spans:
                if entity.start < end and entity.end > start:
                    overlap = True
                    break
            if not overlap:
                result.append(entity)
                used_spans.append((entity.start, entity.end))

        return result


# ============================================================
# 意图分类器 — 规则优先 + LLM 兜底
# ============================================================


class IntentClassifier:
    """意图分类器 (借鉴 LangChain Router + Cohere Command-R intent).

    两级分类策略:
    1. 规则引擎 (优先, <10ms): 基于关键词和模式匹配
    2. LLM 兜底 (备用, <100ms): 外部 LLM 分类 (预留接口)

    优先级: numeric > relational > concept > composite
    """

    def __init__(self) -> None:
        self._extractor = EntityExtractor()

        # 数值意图规则
        self._numeric_rules: list[tuple[str, re.Pattern[str]]] = [
            ("numeric_unit", re.compile(
                r"\d+\.?\d*\s*(?:nm|cm[-‐]?1|K|mol%|eV|J|meV|wt%|Å|"
                r"lux|cd/m[²2]|lm/W|μs|ms|Hz)\b",
                re.IGNORECASE,
            )),
            ("numeric_keyword", re.compile(
                r"(?:值|参数|浓度|温度|波长|效率|能量|距离|寿命|"
                r"value|parameter|concentration|temperature|"
                r"wavelength|efficiency|energy|distance|lifetime)",
                re.IGNORECASE,
            )),
            ("standard_ref", re.compile(
                r"(?:GB/T|IEC|CIE|ASTM|ISO)\s*\d+",
                re.IGNORECASE,
            )),
        ]

        # 关系意图规则
        self._relational_rules: list[tuple[str, re.Pattern[str]]] = [
            ("relationship_kw", re.compile(
                r"(?:关系|关联|影响|作用|连接|依赖|先修|前置|后续|因果|"
                r"relation|relationship|affect|influence|cause|depend|"
                r"prerequisite)",
                re.IGNORECASE,
            )),
            ("path_kw", re.compile(
                r"(?:路径|链路|传递|转移|跃迁|"
                r"path|chain|transfer|transition)",
                re.IGNORECASE,
            )),
            ("graph_kw", re.compile(
                r"(?:子图|邻居|连通|社区|"
                r"subgraph|neighbor|connected|community)",
                re.IGNORECASE,
            )),
        ]

        # 概念意图规则
        self._concept_rules: list[tuple[str, re.Pattern[str]]] = [
            ("definition_kw", re.compile(
                r"(?:是什么|什么是|是啥|为何物|指的是|何种|定义|概念|含义|解释|说明|"
                r"what is|what's|define|definition|concept|meaning|explain)",
                re.IGNORECASE,
            )),
            ("mechanism_kw", re.compile(
                r"(?:机理|机制|原理|过程|途径|"
                r"mechanism|process|pathway|principle)",
                re.IGNORECASE,
            )),
            # 方法/过程意图 ("怎么制备" / "如何合成" 等, 此前归入 fallback 导致误判)
            ("method_kw", re.compile(
                r"(?:怎么|如何|怎样|怎么弄|制备|合成|方法|步骤|工艺|流程|做法|"
                r"how|method|synthesize|synthesis|fabricate|procedure|steps)",
                re.IGNORECASE,
            )),
            # 原因/机理意图 ("为什么发光" / "为何猝灭" 等)
            ("reason_kw", re.compile(
                r"(?:为什么|为何|为啥|原因|因为|由于|"
                r"why|reason|because|cause)",
                re.IGNORECASE,
            )),
        ]

        # 复合意图规则 (多种意图同时出现)
        self._composite_rules: list[tuple[str, re.Pattern[str]]] = [
            ("comparison_kw", re.compile(
                r"(?:比较|对比|区别|差异|优劣|"
                r"compare|comparison|difference|versus|vs)",
                re.IGNORECASE,
            )),
            ("multi_intent", re.compile(
                r"(?:并且|同时|此外|另外|以及|"
                r"and also|moreover|in addition|as well as)",
                re.IGNORECASE,
            )),
        ]

    def classify(
        self,
        query: str,
        *,
        use_llm: bool = False,
        intent_hint: str = "",
        schema_context: str = "",
    ) -> IntentResult:
        """分类查询意图.

        Args:
            query: 查询文本
            use_llm: 是否使用 LLM 兜底 (规则无法确定时)
            intent_hint: 上下文推断的意图提示 (如 "numeric+relational")
            schema_context: KG Schema 上下文片段

        Returns:
            意图识别结果
        """
        start_time = time.time()

        # 提取实体
        entities = self._extractor.extract(query)

        # 规则匹配
        matched_rules: list[str] = []
        intent_scores: dict[IntentType, float] = {
            IntentType.NUMERIC: 0.0,
            IntentType.RELATIONAL: 0.0,
            IntentType.CONCEPT: 0.0,
            IntentType.COMPOSITE: 0.0,
        }

        # 检查数值意图
        for rule_name, pattern in self._numeric_rules:
            if pattern.search(query):
                matched_rules.append(f"numeric:{rule_name}")
                intent_scores[IntentType.NUMERIC] += 0.4

        # 数值实体加分
        numeric_entities = [
            e for e in entities if e.entity_type == "numeric"
        ]
        if numeric_entities:
            intent_scores[IntentType.NUMERIC] += 0.3
            matched_rules.append("numeric:entity_extracted")

        # 检查关系意图
        for rule_name, pattern in self._relational_rules:
            if pattern.search(query):
                matched_rules.append(f"relational:{rule_name}")
                intent_scores[IntentType.RELATIONAL] += 0.35

        # 离子/化学式 + 关系词 → 关系意图加分
        has_chemical_entity = any(
            e.entity_type in ("ion", "formula", "spectral_term")
            for e in entities
        )
        if has_chemical_entity and intent_scores[IntentType.RELATIONAL] > 0:
            intent_scores[IntentType.RELATIONAL] += 0.2

        # 检查概念意图
        for rule_name, pattern in self._concept_rules:
            if pattern.search(query):
                matched_rules.append(f"concept:{rule_name}")
                intent_scores[IntentType.CONCEPT] += 0.35

        # 检查复合意图
        for rule_name, pattern in self._composite_rules:
            if pattern.search(query):
                matched_rules.append(f"composite:{rule_name}")
                intent_scores[IntentType.COMPOSITE] += 0.4

        # 多意图检测: 如果有两个以上意图得分 > 0
        active_intents = [
            it for it, score in intent_scores.items()
            if score > 0.3 and it != IntentType.COMPOSITE
        ]
        if len(active_intents) >= 2:
            intent_scores[IntentType.COMPOSITE] += 0.5
            matched_rules.append("composite:multi_intent_detected")

        # --- v2: 意图提示融合 (Self-RAG 风格) ---
        if intent_hint:
            self._apply_intent_hint(
                intent_hint, intent_scores, matched_rules
            )

        # --- v2: Schema 上下文增强 ---
        if schema_context:
            self._apply_schema_boost(
                query, schema_context, intent_scores, matched_rules
            )

        # 选择最高分意图
        best_intent = max(intent_scores, key=lambda k: intent_scores[k])
        best_score = intent_scores[best_intent]

        # 如果所有得分都很低，默认为概念检索
        if best_score < 0.15:
            best_intent = IntentType.CONCEPT
            best_score = 0.3
            matched_rules.append("fallback:default_concept")

        # LLM 兜底 (预留接口)
        if use_llm and best_score < 0.5:
            llm_result = self._llm_classify(query)
            if llm_result is not None:
                best_intent = llm_result
                best_score = 0.7
                matched_rules.append("llm:fallback")

        # 确定建议路径
        path_map = {
            IntentType.CONCEPT: "vector+keyword→rrf",
            IntentType.NUMERIC: "exact+filter→fact_check",
            IntentType.RELATIONAL: "graph_traversal→subgraph",
            IntentType.COMPOSITE: "parallel(vector+keyword+graph)→rrf+fact_check",
        }

        elapsed = (time.time() - start_time) * 1000

        return IntentResult(
            intent_type=best_intent,
            confidence=min(best_score, 1.0),
            matched_rules=matched_rules,
            extracted_entities=entities,
            suggested_path=path_map.get(best_intent, "vector+keyword→rrf"),
            classification_time_ms=round(elapsed, 2),
        )

    def _apply_intent_hint(
        self,
        hint: str,
        scores: dict[IntentType, float],
        matched_rules: list[str],
    ) -> None:
        """融合上下文推断的意图提示.

        借鉴 Self-RAG 的反思 token: 外部上下文的信号作为弱监督
        对规则评分进行微调 (小幅加分, 不覆盖规则结论).
        """
        _HINT_MAP: dict[str, IntentType] = {
            "numeric": IntentType.NUMERIC,
            "relational": IntentType.RELATIONAL,
            "concept": IntentType.CONCEPT,
            "composite": IntentType.COMPOSITE,
        }
        for part in hint.split("+"):
            part = part.strip().lower()
            if part in _HINT_MAP:
                scores[_HINT_MAP[part]] += 0.15
                matched_rules.append(f"context_hint:{part}")

    def _apply_schema_boost(
        self,
        query: str,
        schema_context: str,
        scores: dict[IntentType, float],
        matched_rules: list[str],
    ) -> None:
        """基于 Schema 上下文增强意图分类.

        借鉴 SEAL (2025) Agent 校准:
        如果查询中的属性名匹配 Schema 中的数值属性,
        增强 numeric 信号; 匹配关系类型则增强 relational。
        """
        query_lower = query.lower()
        ctx_lower = schema_context.lower()

        # 提取 Schema 中声明的数值属性名
        if "可查询数值属性" in ctx_lower:
            # 匹配查询词与 Schema 属性
            for prop in ["波长", "浓度", "温度", "效率", "能量",
                          "距离", "寿命", "量子效率", "难度", "掌握"]:
                if prop in query_lower:
                    scores[IntentType.NUMERIC] += 0.1
                    matched_rules.append(f"schema_boost:numeric:{prop}")
                    break  # 一次加分即可

        # 匹配关系类型
        if "关系类型" in ctx_lower:
            for rel in ["依赖", "先修", "前置", "影响", "掺杂", "猝灭"]:
                if rel in query_lower:
                    scores[IntentType.RELATIONAL] += 0.1
                    matched_rules.append(f"schema_boost:relational:{rel}")
                    break

    def _llm_classify(self, query: str) -> IntentType | None:
        """LLM 兜底分类: 规则覆盖不了时用 flash 模型判意图 (None 表示无意图/不可用).

        借鉴 Cohere intent classification + Self-RAG 反思: 规则 <10ms, LLM <1s,
        仅规则 fallback 时才走 LLM (省 token). 返回 None 的语义是「无明确意图」,
        由调用方据此决定是否澄清。
        """
        try:
            from dy3_polaris.l3.llm_config import chat_completion
        except Exception:  # noqa: BLE001
            return None
        prompt = (
            "判断下面这个提问属于哪类意图，只回复一个词：\n"
            "- definition: 在问某个对象是什么/定义/含义 (含口语化表达, 如「是啥」「是干嘛的」「何方神圣」)\n"
            "- method: 问怎么做/制备/方法步骤\n"
            "- reason: 问为什么/原因/机理\n"
            "- numeric: 问数值/参数/多少\n"
            "- relational: 问关系/影响/联系\n"
            "- comparison: 问比较/区别/差异\n"
            "- none: 信息不足/纯寒暄/与发光材料无关\n"
            f"提问：{query}"
        )
        try:
            raw = chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=16, disable_thinking=True,
            )
        except Exception:  # noqa: BLE001
            return None
        raw = (raw or "").strip().lower()
        if not raw or "none" in raw:
            return None
        if "compar" in raw or "比较" in raw or "区别" in raw:
            return IntentType.COMPOSITE
        if "numeric" in raw or "数值" in raw or "多少" in raw:
            return IntentType.NUMERIC
        if "relation" in raw or "关系" in raw or "影响" in raw:
            return IntentType.RELATIONAL
        # definition / method / reason / concept 等 → CONCEPT (有明确意图)
        return IntentType.CONCEPT


# ============================================================
# 意图路由检索引擎
# ============================================================


class IntentRouter:
    """意图驱动路由检索引擎 (借鉴 LangChain Router + LlamaIndex RouterQueryEngine).

    根据查询意图自动选择最优检索路径:
    - concept  → 向量检索 + 关键词检索 → RRF 融合
    - numeric   → 精确查询 + 过滤 → 事实校验
    - relational → 图遍历 + 子图提取 → 路径推理
    - composite → 三路并行 + RRF 融合 + 事实校验

    v2 上下文增强:
    - 集成 ContextBuilder, 支持预构建上下文的路由
    - 意图提示融合: 利用上下文 intent_hint 和 schema_context 辅助分类
    - 多查询变体路由: 利用重写结果多路并行检索 + RRF 融合
    - 自适应参数: 根据 QueryContext 中的 suggested_top_k/depth 调整
    - 检索跳过: needs_retrieval=False 时直接返回空结果 (Self-RAG)

    Usage::

        from dy3_polaris.l3 import IntentRouter, KnowledgeStore
        from dy3_polaris.l3.context_builder import ContextBuilder

        store = KnowledgeStore()
        router = IntentRouter(store)

        # v1: 直接路由 (向后兼容)
        result = router.route("Dy3+离子的4F9/2能级跃迁波长是多少nm?")

        # v2: 上下文增强路由
        builder = ContextBuilder()
        ctx = builder.build(query, learner_profile=profile, dialog_history=turns)
        result = router.route_with_context(ctx)
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        classifier: IntentClassifier | None = None,
        use_llm_fallback: bool = False,
        top_k: int = 10,
        rrf_k: int = 60,
        context_builder: Any | None = None,
    ) -> None:
        """初始化意图路由引擎.

        Args:
            store: 知识存储
            classifier: 自定义意图分类器 (默认创建)
            use_llm_fallback: 是否启用 LLM 兜底
            top_k: 默认返回结果数
            rrf_k: RRF 融合参数
            context_builder: 可选的 ContextBuilder 实例
        """
        self.store = store
        self.classifier = classifier or IntentClassifier()
        self._use_llm = use_llm_fallback
        self._default_top_k = top_k
        self._rrf_k = rrf_k
        self._context_builder = context_builder

        # 延迟导入检索引擎 (避免循环导入)
        from .retrieval import RetrievalEngine
        self._engine = RetrievalEngine(store)

    def route(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: RetrievalFilter | None = None,
        query_vector: list[float] | None = None,
        entity_id: str | None = None,
    ) -> RoutedResult:
        """意图驱动路由检索.

        Args:
            query: 查询文本
            top_k: 返回结果数 (默认使用初始化值)
            filter: 过滤条件
            query_vector: 查询向量 (向量检索用)
            entity_id: 起始实体 ID (图检索用)

        Returns:
            路由检索结果 (包含意图识别 + 检索结果)

        Raises:
            RetrievalError: 检索失败
        """
        start_time = time.time()
        k = top_k or self._default_top_k

        # 1. 意图识别
        intent = self.classifier.classify(
            query, use_llm=self._use_llm
        )

        # 2. 根据意图路由
        try:
            if intent.intent_type == IntentType.CONCEPT:
                result = self._route_concept(
                    query, k, filter, query_vector
                )
            elif intent.intent_type == IntentType.NUMERIC:
                result = self._route_numeric(
                    query, k, filter, intent.extracted_entities
                )
            elif intent.intent_type == IntentType.RELATIONAL:
                result = self._route_relational(
                    query, k, filter, entity_id, intent.extracted_entities
                )
            else:  # COMPOSITE
                result = self._route_composite(
                    query, k, filter, query_vector, entity_id,
                    intent.extracted_entities,
                )
        except Exception as exc:
            raise RetrievalError(
                query=query,
                reason=f"意图路由检索失败: {exc}",
            ) from exc

        elapsed = (time.time() - start_time) * 1000

        return RoutedResult(
            intent=intent,
            retrieval_result=result,
            total_time_ms=round(elapsed, 2),
        )

    def _route_concept(
        self,
        query: str,
        top_k: int,
        filter: RetrievalFilter | None,
        query_vector: list[float] | None,
    ) -> RetrievalResult:
        """概念检索路径: 向量 + 关键词 → RRF 融合.

        借鉴 LlamaIndex RouterQueryEngine: 多检索器并行 + 结果融合。
        """
        if query_vector is not None:
            # 向量 + 关键词混合
            return self._engine.hybrid_search(
                query=query,
                top_k=top_k,
                filter=filter,
                query_vector=query_vector,
                retrievers=["vector", "keyword"],
            )
        else:
            # 仅关键词检索
            return self._engine.keyword_search(
                query=query, top_k=top_k, filter=filter
            )

    def _route_numeric(
        self,
        query: str,
        top_k: int,
        filter: RetrievalFilter | None,
        entities: list[ExtractedEntity],
    ) -> RetrievalResult:
        """数值检索路径: 精确查询 + 过滤 + 事实校验.

        借鉴 Weaviate BM25 + 精确过滤模式:
        1. 提取数值实体 (值 + 单位)
        2. 关键词检索获取候选集
        3. 精确过滤匹配数值
        """
        # 构建增强查询 (提取数值关键词)
        numeric_entities = [
            e for e in entities if e.entity_type == "numeric"
        ]

        # 先用关键词检索获取候选
        result = self._engine.keyword_search(
            query=query, top_k=top_k * 2, filter=filter
        )

        # 如果有数值实体，进行精确过滤 (提升含匹配数值的结果)
        if numeric_entities and result.results:
            boosted_results: list[dict[str, Any]] = []
            boosted_scores: list[float] = []

            for i, (res, score) in enumerate(
                zip(result.results, result.scores)
            ):
                boost = 1.0
                content = str(res.get("content", "")).lower()

                for ne in numeric_entities:
                    if ne.value is not None:
                        val_str = str(ne.value)
                        if val_str in content:
                            boost += 0.3
                        if ne.unit and ne.unit.lower() in content:
                            boost += 0.1

                boosted_results.append(res)
                boosted_scores.append(score * boost)

            # 重新排序
            sorted_pairs = sorted(
                zip(boosted_results, boosted_scores),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

            result.results = [p[0] for p in sorted_pairs]
            result.scores = [p[1] for p in sorted_pairs]
            result.total = len(result.results)

        return result

    def _route_relational(
        self,
        query: str,
        top_k: int,
        filter: RetrievalFilter | None,
        entity_id: str | None,
        entities: list[ExtractedEntity],
    ) -> RetrievalResult:
        """关系检索路径: 图遍历 + 子图提取.

        借鉴 GraphRAG: 子图提取 + 路径推理
        1. 从查询中提取实体 (离子/化学式)
        2. 在图中查找匹配实体
        3. 执行 BFS 遍历获取子图
        """
        # 如果有指定 entity_id，直接图检索
        if entity_id:
            return self._engine.graph_search(
                entity_id=entity_id,
                query=query,
                top_k=top_k,
                max_depth=2,
                filter=filter,
            )

        # 从查询中提取化学实体，在知识库中查找
        chemical_entities = [
            e for e in entities
            if e.entity_type in ("ion", "formula", "spectral_term")
        ]

        if chemical_entities:
            for ce in chemical_entities:
                # 尝试通过名称查找实体
                found_list = self.store.entity_store.find_by_name(ce.text)
                if found_list:
                    return self._engine.graph_search(
                        entity_id=found_list[0].entity_id,
                        query=query,
                        top_k=top_k,
                        max_depth=2,
                        filter=filter,
                    )

        # 兜底: 关键词检索
        return self._engine.keyword_search(
            query=query, top_k=top_k, filter=filter
        )

    def _route_composite(
        self,
        query: str,
        top_k: int,
        filter: RetrievalFilter | None,
        query_vector: list[float] | None,
        entity_id: str | None,
        entities: list[ExtractedEntity],
    ) -> RetrievalResult:
        """复合检索路径: 三路并行 + RRF 融合 + 事实校验.

        借鉴 GraphRAG + RRF: 三路并行检索 + 融合排序
        1. 向量检索 (语义相似)
        2. 关键词检索 (精确匹配)
        3. 图检索 (关系推理)
        4. RRF 融合三路结果
        """
        return self._engine.hybrid_search(
            query=query,
            top_k=top_k,
            filter=filter,
            query_vector=query_vector,
            entity_id=entity_id,
            retrievers=["vector", "keyword", "graph"],
        )

    def batch_route(
        self,
        queries: list[str],
        *,
        top_k: int | None = None,
        filter: RetrievalFilter | None = None,
    ) -> list[RoutedResult]:
        """批量意图路由检索.

        Args:
            queries: 查询列表
            top_k: 每个查询返回结果数
            filter: 过滤条件

        Returns:
            路由结果列表
        """
        return [
            self.route(q, top_k=top_k, filter=filter)
            for q in queries
        ]

    # ============================================================
    # v2: 上下文增强路由
    # ============================================================

    def route_with_context(
        self,
        ctx: Any,
        *,
        filter: RetrievalFilter | None = None,
        query_vector: list[float] | None = None,
    ) -> RoutedResult:
        """基于预构建 QueryContext 的上下文增强路由.

        借鉴 ReCAP + Self-RAG + Plan-and-Solve:
        1. Self-RAG: 检查 needs_retrieval, 跳过不必要的检索
        2. 意图融合: 将上下文 intent_hint 和 schema_context 传入分类器
        3. 自适应参数: 使用 suggested_top_k 和 suggested_depth
        4. 多查询变体: 利用 rewritten_queries 多路检索 + RRF 融合

        Args:
            ctx: QueryContext 实例 (由 ContextBuilder.build() 产出)
            filter: 额外过滤条件
            query_vector: 查询向量

        Returns:
            路由检索结果
        """
        start_time = time.time()

        # Self-RAG: 检索需求评估
        if not ctx.needs_retrieval:
            logger.debug(
                "Self-RAG: 跳过检索, context_id=%s",
                ctx.context_id,
            )
            empty_result = RetrievalResult(
                query=ctx.active_query,
                results=[],
                scores=[],
                total=0,
                retrieval_time_ms=0.0,
            )
            return RoutedResult(
                intent=IntentResult(
                    intent_type=IntentType.CONCEPT,
                    confidence=0.0,
                    matched_rules=["self_rag:skip_retrieval"],
                ),
                retrieval_result=empty_result,
                total_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        # 自适应参数
        k = ctx.suggested_top_k
        depth = ctx.suggested_depth

        # 意图分类 (融合上下文信号)
        intent = self.classifier.classify(
            ctx.active_query,
            use_llm=self._use_llm,
            intent_hint=ctx.intent_hint,
            schema_context=ctx.schema_context,
        )

        # 多查询变体路由 (Plan-and-Solve: 分步查询)
        rewritten = ctx.rewritten_queries
        if len(rewritten) >= 2:
            result = self._route_multi_query(
                ctx.active_query,
                rewritten,
                intent,
                top_k=k,
                depth=depth,
                filter=filter,
                query_vector=query_vector,
            )
        else:
            # 单查询路由 (使用指代消解后的查询)
            result = self._route_single(
                ctx.active_query, intent, k, filter, query_vector, depth)

        elapsed = (time.time() - start_time) * 1000
        return RoutedResult(
            intent=intent,
            retrieval_result=result,
            total_time_ms=round(elapsed, 2),
        )

    def route_auto(
        self,
        query: str,
        *,
        learner_profile: Any = None,
        dialog_history: Any = None,
        filter: RetrievalFilter | None = None,
        query_vector: list[float] | None = None,
    ) -> RoutedResult:
        """自动选择路由模式.

        如果已配置 ContextBuilder, 自动构建上下文并使用
        上下文增强路由; 否则回退到 v1 直接路由。

        这是面向 L4 决策引擎的推荐入口。
        """
        if self._context_builder is not None:
            ctx = self._context_builder.build(
                query,
                learner_profile=learner_profile,
                dialog_history=dialog_history,
            )
            return self.route_with_context(
                ctx, filter=filter, query_vector=query_vector
            )
        return self.route(
            query, filter=filter, query_vector=query_vector
        )

    def _route_single(
        self,
        query: str,
        intent: IntentResult,
        top_k: int,
        filter: RetrievalFilter | None,
        query_vector: list[float] | None,
        depth: int,
    ) -> RetrievalResult:
        """单查询路由 (复用 v1 逻辑, 增加 depth 参数)."""
        entity_id = None
        entities = intent.extracted_entities

        # 查找实体 ID (图检索用)
        chemical = [
            e for e in entities
            if e.entity_type in ("ion", "formula", "spectral_term")
        ]
        if chemical:
            found = self.store.entity_store.find_by_name(chemical[0].text)
            if found:
                entity_id = found[0].entity_id

        if intent.intent_type == IntentType.CONCEPT:
            return self._route_concept(query, top_k, filter, query_vector)
        elif intent.intent_type == IntentType.NUMERIC:
            return self._route_numeric(query, top_k, filter, entities)
        elif intent.intent_type == IntentType.RELATIONAL:
            result = self._route_relational(
                query, top_k, filter, entity_id, entities
            )
            # 使用上下文建议的深度
            if depth > 1:
                try:
                    deep_result = self._engine.graph_search(
                        entity_id=entity_id or "",
                        query=query,
                        top_k=top_k,
                        max_depth=depth,
                        filter=filter,
                    )
                    if deep_result.total > 0:
                        result = self._rrf_fuse([result, deep_result])
                except Exception:
                    logger.debug("深度图检索失败, 使用浅层结果", exc_info=True)
            return result
        else:  # COMPOSITE
            return self._route_composite(
                query, top_k, filter, query_vector, entity_id, entities
            )

    def _route_multi_query(
        self,
        primary_query: str,
        rewritten_queries: list[str],
        intent: IntentResult,
        *,
        top_k: int,
        depth: int,
        filter: RetrievalFilter | None,
        query_vector: list[float] | None,
    ) -> RetrievalResult:
        """多查询变体路由 (Plan-and-Solve: 分步检索 + RRF 融合).

        对主查询和每个重写变体分别检索, 然后用 RRF 融合结果。
        借鉴 LangChain MultiQueryRetriever + RRF。
        """
        all_queries = [primary_query] + [
            q for q in rewritten_queries
            if q != primary_query
        ]

        # 限制变体数量 (避免延迟过高)
        max_variants = 3
        if len(all_queries) > max_variants + 1:
            all_queries = all_queries[:max_variants + 1]

        sub_results: list[RetrievalResult] = []
        for q in all_queries:
            try:
                sub = self._route_single(
                    q, intent, top_k, filter, query_vector, depth
                )
                sub_results.append(sub)
            except Exception:
                logger.debug("变体查询检索失败: %s", q, exc_info=True)

        if len(sub_results) <= 1:
            return sub_results[0] if sub_results else RetrievalResult(
                query=primary_query, results=[], scores=[], total=0,
                retrieval_time_ms=0.0,
            )

        return self._rrf_fuse(sub_results)

    def _rrf_fuse(
        self, results: list[RetrievalResult],
    ) -> RetrievalResult:
        """RRF (Reciprocal Rank Fusion) 融合多路检索结果.

        RRF 公式: score(d) = Σ 1/(k + rank_i(d))
        k 默认为 60 (Cormack et al. 2009 经验值)。
        """
        k = self._rrf_k
        # 收集所有文档 ID 及其各路排名
        doc_scores: dict[str, float] = {}
        doc_data: dict[str, dict[str, Any]] = {}

        for result in results:
            for rank, (res, _score) in enumerate(
                zip(result.results, result.scores)
            ):
                doc_id = res.get("id", res.get("chunk_id", f"r-{rank}"))
                rrf_score = 1.0 / (k + rank + 1)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
                if doc_id not in doc_data:
                    doc_data[doc_id] = res

        # 按 RRF 分数排序
        sorted_docs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )

        fused_results = [doc_data[doc_id] for doc_id, _ in sorted_docs]
        fused_scores = [score for _, score in sorted_docs]

        query = results[0].query if results else ""
        total_time = sum(r.retrieval_time_ms for r in results)

        # 请求级 trace_id 接通 (contextvars, 见 l5/tracing.py)
        from dy3_polaris.l5.tracing import get_trace_id

        return RetrievalResult(
            query=query,
            results=fused_results,
            scores=fused_scores,
            total=len(fused_results),
            retrieval_time_ms=total_time,
            trace_id=get_trace_id(),
        )


# ============================================================
# 路由检索结果
# ============================================================


@dataclass
class RoutedResult:
    """意图路由检索结果.

    Attributes:
        intent: 意图识别结果
        retrieval_result: 检索结果
        total_time_ms: 总耗时 (意图识别 + 检索)
    """

    intent: IntentResult
    retrieval_result: RetrievalResult
    total_time_ms: float

    @property
    def results(self) -> list[dict[str, Any]]:
        """检索结果列表."""
        return self.retrieval_result.results

    @property
    def scores(self) -> list[float]:
        """检索结果分数."""
        return self.retrieval_result.scores

    @property
    def total(self) -> int:
        """结果总数."""
        return self.retrieval_result.total


__all__ = [
    "IntentType",
    "ExtractedEntity",
    "IntentResult",
    "EntityExtractor",
    "IntentClassifier",
    "IntentRouter",
    "RoutedResult",
]
