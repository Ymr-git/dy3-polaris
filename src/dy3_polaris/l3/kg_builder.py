"""L3 领域知识层 — 知识图谱构建引擎.

从非结构化/半结构化文本中自动构建知识图谱的完整流水线，融合世界先进
IE (Information Extraction) 与 KG 构建方案:

- REBEL (Huguet Cabot & Navigli, 2021): 端到端关系抽取范式，本引擎以
  "规则模板 + 模式匹配" 等价实现其 trigger-based 抽取思想。
- SpaCy NER + GPT-NER: 命名实体识别的工程化组合，本引擎以 "正则模式 +
  词典匹配 + 类型推测" 实现零外部依赖的等价能力。
- OpenIE (Banko et al., 2007): 开放信息抽取，从任意文本中抽取 (s, p, o)
  三元组，本引擎以可扩展的模式模板表实现等价能力。
- DARE (Culotta & McCallum, 2005): 基于依存句法的关系抽取，本引擎以
  "触发词 + 位置约束" 模板近似其依存路径模式。
- Entity Resolution / Coreference Resolution / Wikidata dedup: 实体消解
  三大范式，本引擎以 "精确/别名/标识符/模糊" 四级匹配 + Union-Find
  并查集实现等价类管理。
- GraphRAG (Edge et al., 2024): 增量图谱构建 + 社区组织，本引擎的
  KnowledgeGraphBuilder 实现增量构建与质量控制。
- ConVer-G / DBpedia-TKG: 版本化知识图谱，本引擎通过 KnowledgeStore
  委托实现版本追踪。
- MACR: 多智能体冲突解决，本引擎在质量控制阶段预留冲突检测钩子。

四大核心组件
------------
1. KGEntityExtractor    — 规则驱动实体抽取器 (PATTERN / DICTIONARY / HYBRID)
2. RelationExtractor  — 模式模板关系抽取器 (>=15 个材料科学领域模板)
3. KGEntityResolver     — 实体消解器 (Union-Find + 四级匹配)
4. KnowledgeGraphBuilder — 整合流水线 (抽取→关系→消解→校验→持久化)

设计原则
--------
- 零外部依赖: 仅使用 Python 标准库 (re / threading / time / uuid) + pydantic v2
- 线程安全: 所有可变状态通过 RLock 保护
- 增量构建: 新文本到达时只处理新增内容，基于 KGEntityResolver 去重
- 质量控制: 置信度阈值过滤 + 本体约束校验 (委托 OntologyRegistry)
- 与 KnowledgeStore 无缝集成: 持久化阶段直接调用 store.add_entity / add_triple

Usage::

    from dy3_polaris.l3.kg_builder import KnowledgeGraphBuilder
    from dy3_polaris.l3.store import KnowledgeStore

    store = KnowledgeStore()
    builder = KnowledgeGraphBuilder(store=store, domain="materials")

    # 单文本构建
    result = builder.build_from_text(
        "Dy3+ 掺杂 YAG 的发射波长为 580 nm，采用溶胶-凝胶法制备。",
        source_id="doc-001",
    )
    print(result.entities_created, result.triples_created)

    # 批量构建
    batch = builder.build_from_texts([
        ("Dy3+ 掺杂 NaYF4 ...", "doc-002"),
        ("CaF2 的晶系为立方晶系 ...", "doc-003"),
    ])
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .models import EntityType, KnowledgeEntity, KnowledgeTriple, RelationType
from .store import KnowledgeStore

logger = logging.getLogger(__name__)


# ============================================================
# 数据类 — 抽取结果容器
# ============================================================


class ExtractionStrategy(str, Enum):
    """实体抽取策略.

    借鉴 SpaCy NER pipeline 的可配置组合策略:

    - PATTERN: 基于预编译正则表达式的模式匹配
        (化学式 / CAS 号 / 波长 / 数值+单位 / 专有名词)
    - DICTIONARY: 基于领域词典的精确匹配
        (从已有实体名称库或同义词表匹配)
    - HYBRID: PATTERN + DICTIONARY 混合 (默认)
        先执行 PATTERN，再对未匹配文本执行 DICTIONARY，去重合并
    """

    PATTERN = "pattern"
    DICTIONARY = "dictionary"
    HYBRID = "hybrid"


class KGExtractedEntity(BaseModel):
    """抽取出的实体候选 (中间结果，尚未持久化).

    借鉴 REBEL span representation + GPT-NER 的实体输出格式，
    每个 Span 携带类型推测、置信度与原文片段，便于后续消解与校验。

    Attributes:
        entity_name: 规范化后的实体名称 (去除多余空白)
        entity_type: 推测的 EntityType (PATTERN 推测可能为 CONCEPT)
        span: 在原文中的字符区间 (start, end)
        confidence: 抽取置信度 [0.0, 1.0]
            PATTERN 命中: 0.7~0.9
            DICTIONARY 命中: 0.9~1.0
            专有名词启发式: 0.5~0.7
        source_text: 原文片段 (用于调试与溯源)
        extraction_method: 抽取方法 ("pattern" / "dictionary" / "heuristic")
        pattern_name: 命中的模式名称 (PATTERN 专属)
        identifiers: 从文本中识别出的外部标识符 (如 cas / doi)
        metadata: 扩展元数据
    """

    entity_name: str = Field(..., description="规范化实体名称")
    entity_type: EntityType = Field(
        default=EntityType.CONCEPT, description="推测的实体类型"
    )
    span: tuple[int, int] = Field(
        default=(0, 0), description="原文字符区间 (start, end)"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="抽取置信度"
    )
    source_text: str = Field(default="", description="原文片段")
    extraction_method: str = Field(
        default="pattern", description="抽取方法"
    )
    pattern_name: str = Field(default="", description="命中的模式名称")
    identifiers: dict[str, str] = Field(
        default_factory=dict, description="外部标识符"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="扩展元数据"
    )


class KGExtractedRelation(BaseModel):
    """抽取出的关系候选 (中间结果，尚未持久化).

    借鉴 REBEL triple output + OpenIE 的 (s, p, o) 表示，
    主语/宾语引用 KGExtractedEntity 的索引或名称，谓词为 RelationType 值。

    Attributes:
        relation_type: 关系类型 (对应 RelationType 值或自定义字符串)
        subject_name: 主语实体名称
        object_name: 宾语实体名称 (可为字面值)
        object_is_literal: 宾语是否为字面值 (数值/单位等)
        subject_span: 主语在原文的区间
        object_span: 宾语在原文的区间
        confidence: 抽取置信度
        trigger_text: 触发文本片段
        pattern_name: 命中的模式名称
        qualifiers: 限定符 (如条件、温度等)
        metadata: 扩展元数据
    """

    relation_type: str = Field(..., description="关系类型")
    subject_name: str = Field(..., description="主语实体名称")
    object_name: str = Field(..., description="宾语实体名称或字面值")
    object_is_literal: bool = Field(
        default=False, description="宾语是否为字面值"
    )
    subject_span: tuple[int, int] = Field(
        default=(0, 0), description="主语区间"
    )
    object_span: tuple[int, int] = Field(
        default=(0, 0), description="宾语区间"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="抽取置信度"
    )
    trigger_text: str = Field(default="", description="触发文本片段")
    pattern_name: str = Field(default="", description="命中的模式名称")
    qualifiers: dict[str, Any] = Field(
        default_factory=dict, description="限定符"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="扩展元数据"
    )


class EntityCluster(BaseModel):
    """实体消解后的等价类簇.

    借鉴 Wikidata 实体合并 + Entity Resolution cluster 概念，
    一个簇内的所有 source_entities 指向同一个 canonical_entity。

    Attributes:
        cluster_id: 簇唯一标识
        canonical_name: 规范名称 (最长或最高置信度的名称)
        canonical_type: 规范实体类型
        aliases: 别名列表 (含所有非规范名称)
        identifiers: 合并后的所有外部标识符
        source_entities: 簇内所有 KGExtractedEntity (含规范实体)
        merged_count: 合并的实体数量 (含规范)
        best_confidence: 簇内最高置信度
    """

    cluster_id: str = Field(
        default_factory=lambda: f"cl-{uuid.uuid4().hex[:8]}"
    )
    canonical_name: str = Field(..., description="规范名称")
    canonical_type: EntityType = Field(
        default=EntityType.CONCEPT, description="规范实体类型"
    )
    aliases: list[str] = Field(
        default_factory=list, description="别名列表"
    )
    identifiers: dict[str, str] = Field(
        default_factory=dict, description="合并后的外部标识符"
    )
    source_entities: list[KGExtractedEntity] = Field(
        default_factory=list, description="簇内所有抽取实体"
    )
    merged_count: int = Field(default=1, description="合并实体数量")
    best_confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="簇内最高置信度"
    )


class BuildResult(BaseModel):
    """单次构建结果统计.

    借鉴 ConVer-G 变更集 + GraphRAG 增量构建统计，
    记录一次 build_from_text 调用中实体/三元组/消解的增删改情况。

    Attributes:
        source_id: 数据源 ID
        entities_created: 新增实体数
        entities_updated: 更新实体数 (属性/别名补充)
        entities_skipped: 跳过实体数 (低于置信度阈值或重复)
        triples_created: 新增三元组数
        triples_skipped: 跳过三元组数 (重复或校验失败)
        resolution_merged: 消解合并的实体数
        build_time_ms: 构建耗时 (毫秒)
        warnings: 警告信息列表
        entity_ids: 新增/更新后的实体 ID 列表 (供调用方引用)
        triple_ids: 新增的三元组 ID 列表
    """

    source_id: str = Field(default="", description="数据源 ID")
    entities_created: int = Field(default=0, description="新增实体数")
    entities_updated: int = Field(default=0, description="更新实体数")
    entities_skipped: int = Field(default=0, description="跳过实体数")
    triples_created: int = Field(default=0, description="新增三元组数")
    triples_skipped: int = Field(default=0, description="跳过三元组数")
    resolution_merged: int = Field(
        default=0, description="消解合并的实体数"
    )
    build_time_ms: float = Field(
        default=0.0, description="构建耗时 (毫秒)"
    )
    warnings: list[str] = Field(
        default_factory=list, description="警告信息列表"
    )
    entity_ids: list[str] = Field(
        default_factory=list, description="新增/更新后的实体 ID"
    )
    triple_ids: list[str] = Field(
        default_factory=list, description="新增的三元组 ID"
    )


class BatchBuildResult(BaseModel):
    """批量构建结果统计.

    汇总多次 build_from_text 的结果，提供整体构建统计。

    Attributes:
        total_texts: 处理的文本总数
        success_count: 成功构建的文本数
        failure_count: 失败的文本数
        results: 每个文本的 BuildResult 列表
        total_entities_created: 累计新增实体数
        total_triples_created: 累计新增三元组数
        total_resolution_merged: 累计消解合并数
        total_build_time_ms: 累计耗时 (毫秒)
        warnings: 全局警告信息
    """

    total_texts: int = Field(default=0, description="文本总数")
    success_count: int = Field(default=0, description="成功数")
    failure_count: int = Field(default=0, description="失败数")
    results: list[BuildResult] = Field(
        default_factory=list, description="每个文本的构建结果"
    )
    total_entities_created: int = Field(
        default=0, description="累计新增实体数"
    )
    total_triples_created: int = Field(
        default=0, description="累计新增三元组数"
    )
    total_resolution_merged: int = Field(
        default=0, description="累计消解合并数"
    )
    total_build_time_ms: float = Field(
        default=0.0, description="累计耗时 (毫秒)"
    )
    warnings: list[str] = Field(
        default_factory=list, description="全局警告信息"
    )


# ============================================================
# 组件 1: KGEntityExtractor — 规则驱动实体抽取器
# ============================================================


class KGEntityExtractor:
    """规则驱动的实体抽取器 (零外部依赖).

    借鉴 SpaCy NER pipeline + GPT-NER 的分层抽取思想，
    通过预编译正则模式 + 领域词典实现等价的命名实体识别能力，
    无需任何外部 NLP 库即可在 Dy3+ Polaris 材料科学领域工作。

    支持三种抽取策略:
    - PATTERN: 基于正则表达式的模式匹配 (化学式 / CAS 号 / 波长 / 数值+单位 / 专有名词)
    - DICTIONARY: 基于词典的精确匹配 (从已有实体名称库匹配)
    - HYBRID: PATTERN + DICTIONARY 混合 (默认，先 PATTERN 再 DICTIONARY 补充)

    内置模式 (预编译正则):
    - chemical_formula: 化学式 (Dy3+, YAG, NaYF4, CaF2, H2O, Fe2O3 等)
    - cas_number: CAS 号 (\\d{2,7}-\\d{2}-\\d)
    - wavelength: 波长 (\\d+\\.?\\d*\\s*nm)
    - numeric_value: 数值+单位 (\\d+\\.?\\d*\\s*(K|℃|nm|μm|eV|mol|g|mg|kJ|nm|nm))
    - proper_noun: 专有名词 (首字母大写英文词 / 中文引号内容)
    - doi: DOI 标识符 (10\\.\\d{4,}/\\S+)
    - temperature: 温度 (\\d+\\.?\\d*\\s*(K|℃|°C))

    算法说明:
        1. PATTERN 模式: 对文本逐个应用预编译正则，记录所有非重叠匹配
        2. DICTIONARY 模式: 对文本执行多模式字符串搜索 (基于 Trie/哈希)
        3. HYBRID 模式: 先 PATTERN，再对 PATTERN 未覆盖区间执行 DICTIONARY
        4. 类型推测: 根据命中模式推测 EntityType
            - chemical_formula → CHEMICAL_COMPOUND
            - cas_number → CHEMICAL_COMPOUND
            - proper_noun (词典) → 词典指定类型
            - proper_noun (启发式) → CONCEPT
            - wavelength / numeric_value → CONCEPT (字面值)

    线程安全:
        - _lock (RLock) 保护 _dictionaries 的并发修改
        - 预编译模式为不可变对象，无需加锁

    Usage::

        extractor = KGEntityExtractor(strategy=ExtractionStrategy.HYBRID)
        extractor.add_dictionary(
            EntityType.MATERIAL,
            ["YAG", "NaYF4", "CaF2", "钇铝石榴石"],
        )
        entities = extractor.extract("Dy3+ 掺杂 YAG 的发射波长为 580 nm")
        # entities 包含 Dy3+ (chemical_compound), YAG (material), 580 nm (concept)
    """

    # ---- 预编译正则模式表 ----
    # 每项: (pattern_name, compiled_regex, entity_type, confidence, extraction_method)
    _PATTERNS: list[tuple[str, re.Pattern[str], EntityType, float, str]] = [
        # CAS 号: \d{2,7}-\d{2}-\d (优先匹配，避免被 chemical_formula 吞掉)
        (
            "cas_number",
            re.compile(r"\b(\d{2,7}-\d{2}-\d)\b"),
            EntityType.CHEMICAL_COMPOUND,
            0.95,
            "pattern",
        ),
        # DOI: 10.\d{4,}/\S+
        (
            "doi",
            re.compile(r"\b(10\.\d{4,}/[^\s,;\"'\)\]\}]+)"),
            EntityType.PAPER,
            0.95,
            "pattern",
        ),
        # 化学式: Dy3+, YAG, NaYF4, CaF2, H2O, Fe2O3, La2O3 等
        # 规则: 元素符号(首字母大写+可选小写) + 可选数字 + 可选电荷(+/-)
        #       或全大写缩写 (YAG, YLF, NIST) 或混合 (NaYF4)
        # 注意: 离子如 Dy3+, Na+, Cl- 的 +/- 不在 \b 边界内，
        #       因此使用 (?<![a-zA-Z]) 替代 \b 确保左边界
        (
            "chemical_formula",
            re.compile(
                r"(?<![a-zA-Z])("
                r"(?:[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)"  # 多元素化学式 H2O, NaYF4
                r"|(?:[A-Z]{2,5}\d*)"  # 全大写缩写 YAG, YLF, NIST
                r"|(?:[A-Z][a-z]?\d*[+-])"  # 离子 Dy3+, Na+, Cl-
                r")(?![a-zA-Z\d])"
            ),
            EntityType.CHEMICAL_COMPOUND,
            0.85,
            "pattern",
        ),
        # 波长: \d+\.?\d*\s*nm
        (
            "wavelength",
            re.compile(r"(\d+\.?\d*\s*nm)\b"),
            EntityType.CONCEPT,
            0.85,
            "pattern",
        ),
        # 温度: \d+\.?\d*\s*(K|℃|°C)
        (
            "temperature",
            re.compile(r"(\d+\.?\d*\s*(?:K|℃|°C))\b"),
            EntityType.CONCEPT,
            0.85,
            "pattern",
        ),
        # 数值+单位: 各种物理量 (eV, mol, g, mg, kJ, J, %, MPa, GPa 等)
        (
            "numeric_value",
            re.compile(
                r"(\d+\.?\d*\s*(?:eV|mol|g|mg|kg|kJ|J|%|MPa|GPa|Hz|kHz|MHz|GHz|"
                r"V|mV|kV|A|mA|W|mW|s|ms|μs|ns|ps|min|h|ppm|ppb|nm|μm|mm|cm|m)\b)"
            ),
            EntityType.CONCEPT,
            0.75,
            "pattern",
        ),
        # 中文引号内容: "..." 或 《...》 (书名/术语)
        (
            "chinese_quoted",
            re.compile(r"[\"“]([^\"”]{1,40})[\"”]"),
            EntityType.CONCEPT,
            0.65,
            "heuristic",
        ),
        # 书名号: 《...》
        (
            "book_title",
            re.compile(r"《([^》]{1,60})》"),
            EntityType.TEXTBOOK,
            0.75,
            "heuristic",
        ),
        # 英文专有名词: 连续 2+ 个首字母大写词 (YAG Crystal, New York)
        (
            "english_proper",
            re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"),
            EntityType.CONCEPT,
            0.6,
            "heuristic",
        ),
    ]

    def __init__(
        self,
        strategy: ExtractionStrategy = ExtractionStrategy.HYBRID,
        *,
        min_confidence: float = 0.4,
    ) -> None:
        """初始化实体抽取器.

        Args:
            strategy: 抽取策略 (默认 HYBRID)
            min_confidence: 最低置信度阈值，低于此值的候选将被过滤
        """
        self._strategy: ExtractionStrategy = strategy
        self._min_confidence: float = min_confidence
        # 领域词典: {entity_type: {name_lower: name_display}}
        self._dictionaries: dict[EntityType, dict[str, str]] = {}
        # 别名映射: {alias_lower: canonical_name} (供消解使用)
        self._alias_map: dict[str, str] = {}
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 词典管理
    # --------------------------------------------------------

    def add_dictionary(
        self,
        entity_type: EntityType,
        names: list[str],
        *,
        aliases: dict[str, str] | None = None,
    ) -> None:
        """添加领域词典.

        借鉴 GPT-NER 的 prompt-augmented dictionary 思想，
        词典提供高置信度的已知实体，DICTIONARY 模式优先匹配。

        Args:
            entity_type: 实体类型
            names: 实体名称列表 (规范名称)
            aliases: 可选别名映射 {alias: canonical_name}
                别名会被规范化为小写存储，匹配时忽略大小写

        Usage::

            extractor.add_dictionary(
                EntityType.MATERIAL,
                ["YAG", "NaYF4", "CaF2"],
                aliases={"钇铝石榴石": "YAG", "Y3Al5O12": "YAG"},
            )
        """
        with self._lock:
            if entity_type not in self._dictionaries:
                self._dictionaries[entity_type] = {}
            for name in names:
                if name and name.strip():
                    normalized = name.strip()
                    self._dictionaries[entity_type][normalized.lower()] = normalized
            if aliases:
                for alias, canonical in aliases.items():
                    if alias and canonical:
                        self._alias_map[alias.strip().lower()] = canonical.strip()
        logger.debug(
            "添加词典: type=%s, names=%d, aliases=%d",
            entity_type.value,
            len(names),
            len(aliases or {}),
        )

    def get_dictionary(self, entity_type: EntityType) -> list[str]:
        """获取指定类型的词典内容 (副本)."""
        with self._lock:
            return list(self._dictionaries.get(entity_type, {}).values())

    # --------------------------------------------------------
    # 抽取主流程
    # --------------------------------------------------------

    def extract(self, text: str) -> list[KGExtractedEntity]:
        """从文本中抽取实体.

        根据 _strategy 选择抽取方式:
        - PATTERN: 仅执行模式匹配
        - DICTIONARY: 仅执行词典匹配
        - HYBRID: 先 PATTERN 再对未覆盖区间执行 DICTIONARY，去重合并

        Args:
            text: 输入文本

        Returns:
            抽取的实体列表 (按 span 起始位置排序，已过滤低置信度)
        """
        if not text or not text.strip():
            return []

        if self._strategy == ExtractionStrategy.PATTERN:
            results = self._extract_by_pattern(text)
        elif self._strategy == ExtractionStrategy.DICTIONARY:
            results = self._extract_by_dictionary(text, excluded_spans=set())
        else:  # HYBRID
            pattern_results = self._extract_by_pattern(text)
            # 收集 PATTERN 已覆盖的区间
            covered: set[int] = set()
            for ent in pattern_results:
                start, end = ent.span
                covered.update(range(start, end))
            dict_results = self._extract_by_dictionary(
                text, excluded_spans=covered
            )
            results = pattern_results + dict_results

        # 过滤低置信度
        results = [
            e for e in results if e.confidence >= self._min_confidence
        ]
        # 按 span 起始位置排序
        results.sort(key=lambda e: (e.span[0], e.span[1]))
        return results

    # --------------------------------------------------------
    # PATTERN 抽取
    # --------------------------------------------------------

    def _extract_by_pattern(self, text: str) -> list[KGExtractedEntity]:
        """基于预编译正则模式抽取实体.

        算法: 遍历所有模式，记录所有匹配，对重叠匹配保留置信度最高者。
        """
        results: list[KGExtractedEntity] = []
        occupied: list[tuple[int, int]] = []

        for pattern_name, regex, entity_type, confidence, method in self._PATTERNS:
            for match in regex.finditer(text):
                start, end = match.span()
                # 跳过与已匹配区间重叠的部分 (保留高置信度)
                if self._overlaps(start, end, occupied):
                    continue
                # 对于带捕获组的模式，使用 group(1)
                source_text = match.group(1) if match.groups() else match.group(0)
                entity_name = source_text.strip()

                # 标识符识别
                identifiers: dict[str, str] = {}
                if pattern_name == "cas_number":
                    identifiers["cas"] = entity_name
                elif pattern_name == "doi":
                    identifiers["doi"] = entity_name

                entity = KGExtractedEntity(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    span=(start, end),
                    confidence=confidence,
                    source_text=source_text,
                    extraction_method=method,
                    pattern_name=pattern_name,
                    identifiers=identifiers,
                )
                results.append(entity)
                occupied.append((start, end))

        return results

    # --------------------------------------------------------
    # DICTIONARY 抽取
    # --------------------------------------------------------

    def _extract_by_dictionary(
        self,
        text: str,
        *,
        excluded_spans: set[int],
    ) -> list[KGExtractedEntity]:
        """基于领域词典抽取实体.

        算法: 对每个词典项执行不区分大小写的子串搜索，
        跳过 excluded_spans 中已覆盖的字符位置。

        Args:
            text: 输入文本
            excluded_spans: 已被 PATTERN 覆盖的字符位置集合
        """
        results: list[KGExtractedEntity] = []
        with self._lock:
            # 快照词典避免长时间持锁
            dict_snapshot: list[tuple[EntityType, str, str]] = []
            for etype, name_map in self._dictionaries.items():
                for name_lower, name_display in name_map.items():
                    dict_snapshot.append((etype, name_lower, name_display))
            alias_snapshot = dict(self._alias_map)

        text_lower = text.lower()

        # 词典实体匹配
        for etype, name_lower, name_display in dict_snapshot:
            start = 0
            while True:
                idx = text_lower.find(name_lower, start)
                if idx < 0:
                    break
                end = idx + len(name_lower)
                # 检查是否落在已占用区间
                if not self._span_excluded(idx, end, excluded_spans):
                    # 检查词边界 (避免匹配子串，如 "YAG" 在 "YAGate" 中)
                    if self._is_word_boundary(text, idx, end):
                        entity = KGExtractedEntity(
                            entity_name=name_display,
                            entity_type=etype,
                            span=(idx, end),
                            confidence=0.95,  # 词典匹配高置信度
                            source_text=text[idx:end],
                            extraction_method="dictionary",
                            pattern_name="dictionary",
                        )
                        results.append(entity)
                        # 标记占用
                        for i in range(idx, end):
                            excluded_spans.add(i)
                start = idx + 1

        # 别名匹配 (映射到规范名称)
        for alias_lower, canonical in alias_snapshot.items():
            start = 0
            while True:
                idx = text_lower.find(alias_lower, start)
                if idx < 0:
                    break
                end = idx + len(alias_lower)
                if not self._span_excluded(idx, end, excluded_spans):
                    if self._is_word_boundary(text, idx, end):
                        # 推测类型: 查找规范名称在哪个词典中
                        etype = self._lookup_type(canonical)
                        entity = KGExtractedEntity(
                            entity_name=canonical,  # 使用规范名称
                            entity_type=etype,
                            span=(idx, end),
                            confidence=0.9,
                            source_text=text[idx:end],
                            extraction_method="dictionary",
                            pattern_name="alias",
                            metadata={"matched_alias": text[idx:end]},
                        )
                        results.append(entity)
                        for i in range(idx, end):
                            excluded_spans.add(i)
                start = idx + 1

        return results

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    @staticmethod
    def _overlaps(
        start: int, end: int, occupied: list[tuple[int, int]]
    ) -> bool:
        """检查区间是否与已占用区间重叠."""
        for o_start, o_end in occupied:
            if not (end <= o_start or start >= o_end):
                return True
        return False

    @staticmethod
    def _span_excluded(
        start: int, end: int, excluded: set[int]
    ) -> bool:
        """检查区间是否完全落在已排除集合中."""
        if start >= end:
            return True
        # 只要有一个字符未被排除，就不算完全排除
        return all(i in excluded for i in range(start, end))

    @staticmethod
    def _is_word_boundary(
        text: str, start: int, end: int
    ) -> bool:
        """检查匹配区间是否在词边界上.

        对于英文: 前后字符应为非字母数字 (避免 "YAG" 匹配 "YAGate")
        对于中文: 总是返回 True (中文无空格分词)
        """
        if start > 0:
            prev_char = text[start - 1]
            if prev_char.isascii() and prev_char.isalnum():
                return False
        if end < len(text):
            next_char = text[end]
            if next_char.isascii() and next_char.isalnum():
                return False
        return True

    def _lookup_type(self, name: str) -> EntityType:
        """查找规范名称对应的实体类型."""
        name_lower = name.lower()
        with self._lock:
            for etype, name_map in self._dictionaries.items():
                if name_lower in name_map:
                    return etype
        return EntityType.CONCEPT


# ============================================================
# 组件 2: RelationExtractor — 模式模板关系抽取器
# ============================================================


class RelationPattern(BaseModel):
    """关系抽取模式模板.

    借鉴 REBEL 的 trigger-based 抽取 + DARE 的依存路径模式，
    每个模式定义一个触发正则、关系类型和主宾位置。

    Attributes:
        pattern_name: 模式名称 (唯一标识)
        trigger_regex: 触发正则表达式 (字符串形式，便于序列化)
        relation_type: 关系类型 (RelationType 值或自定义)
        subject_group: 主语捕获组序号 (1-based)
        object_group: 宾语捕获组序号 (1-based)
        object_is_literal: 宾语是否为字面值 (如波长、温度)
        confidence: 命中时的基础置信度
        description: 模式描述
    """

    pattern_name: str = Field(..., description="模式名称")
    trigger_regex: str = Field(..., description="触发正则表达式")
    relation_type: str = Field(..., description="关系类型")
    subject_group: int = Field(default=1, description="主语捕获组序号")
    object_group: int = Field(default=2, description="宾语捕获组序号")
    object_is_literal: bool = Field(
        default=False, description="宾语是否为字面值"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="基础置信度"
    )
    description: str = Field(default="", description="模式描述")


class RelationExtractor:
    """基于模式模板的关系抽取器.

    借鉴 REBEL (端到端关系抽取) + OpenIE (开放信息抽取) + DARE (依存句法
    关系抽取) 的 trigger-based 抽取思想，通过预定义的模式模板从文本中
    抽取实体间关系。

    每个模式是一个 (trigger_pattern, relation_type, subject_position,
    object_position) 元组，通过正则捕获组定位主语和宾语。

    内置模式 (>=15 个，覆盖 Dy3+ Polaris 材料科学领域):

    掺杂关系:
        - doped_in_zh: "X 掺杂 Y" → doped_in (X, Y)
        - doped_in_en: "X doped Y" / "Y doped with X" → doped_in
        - doped_colon: "X:Y" (如 Dy:YAG) → doped_in

    发射关系:
        - emits_at_zh: "X 发射 Ynm 光" / "X 的发射波长为 Y" → emits_at
        - emits_at_en: "X emits at Y" / "emission of X at Y" → emits_at
        - emission_peak: "X 的发射峰位于 Y" → emits_at

    组成关系:
        - composed_of_zh: "X 由 Y 组成" / "X 包含 Y" → part_of
        - composed_of_en: "X consists of Y" / "X contains Y" → part_of

    属性关系:
        - has_property_zh: "X 的 Y 为 Z" (如 YAG 的晶系为立方) → has_property
        - has_property_en: "X has Y of Z" / "Y of X is Z" → has_property

    引用关系:
        - cites_zh: "X 参考 Y" / "根据 X" → cites
        - cites_en: "X et al. [Y]" / "according to X" → cites

    方法关系:
        - prepared_by_zh: "采用 X 方法制备 Y" / "通过 X 合成 Y" → derived_from
        - prepared_by_en: "Y prepared by X" / "Y synthesized via X" → derived_from

    晶系关系:
        - crystal_system: "X 的晶系为 Y" → has_property

    算法说明:
        1. 对文本应用每个模式的预编译正则
        2. 提取主语/宾语捕获组
        3. 将主语/宾语与已抽取的实体列表对齐 (基于 span 重叠或名称匹配)
        4. 若宾语为字面值 (object_is_literal=True)，直接作为字面宾语
        5. 输出 KGExtractedRelation 列表

    线程安全:
        - _lock (RLock) 保护 _patterns 的并发修改
        - 预编译模式为不可变对象

    Usage::

        extractor = RelationExtractor()
        relations = extractor.extract(
            "Dy3+ 掺杂 YAG 的发射波长为 580 nm",
            entities=entity_list,
        )
        # relations 包含 (Dy3+, doped_in, YAG), (YAG, emits_at, "580 nm")
    """

    # 默认模式定义 (字符串形式，运行时编译)
    _DEFAULT_PATTERNS: list[RelationPattern] = [
        # ---- 掺杂关系 ----
        RelationPattern(
            pattern_name="doped_in_zh",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s*掺杂\s*([A-Za-z0-9]+)",
            relation_type=RelationType.DERIVED_FROM.value,
            subject_group=1,
            object_group=2,
            confidence=0.9,
            description="X 掺杂 Y → X 派生自 Y (X 作为掺杂剂进入 Y 基质)",
        ),
        RelationPattern(
            pattern_name="doped_in_en",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s*doped\s+([A-Za-z0-9]+)",
            relation_type=RelationType.DERIVED_FROM.value,
            subject_group=1,
            object_group=2,
            confidence=0.9,
            description="X doped Y → X 派生自 Y",
        ),
        RelationPattern(
            pattern_name="doped_with_en",
            trigger_regex=r"([A-Za-z0-9]+)\s+doped\s+with\s+([A-Za-z0-9+\-]+)",
            relation_type=RelationType.DERIVED_FROM.value,
            subject_group=2,
            object_group=1,
            confidence=0.9,
            description="Y doped with X → X 派生自 Y",
        ),
        RelationPattern(
            pattern_name="doped_colon",
            trigger_regex=r"\b([A-Za-z0-9+\-]+):([A-Za-z0-9]+)\b",
            relation_type=RelationType.DERIVED_FROM.value,
            subject_group=1,
            object_group=2,
            confidence=0.75,
            description="X:Y (如 Dy:YAG) → X 派生自 Y",
        ),
        # ---- 发射关系 ----
        RelationPattern(
            pattern_name="emits_at_zh_wavelength",
            trigger_regex=r"([A-Za-z0-9+\-]+).*?发射波长为\s*(\d+\.?\d*\s*nm)",
            relation_type=RelationType.HAS_PROPERTY.value,
            subject_group=1,
            object_group=2,
            object_is_literal=True,
            confidence=0.9,
            description="X 的发射波长为 Y → X has_property Y",
        ),
        RelationPattern(
            pattern_name="emission_peak_zh",
            trigger_regex=r"([A-Za-z0-9+\-]+).*?发射峰位于\s*(\d+\.?\d*\s*nm)",
            relation_type=RelationType.HAS_PROPERTY.value,
            subject_group=1,
            object_group=2,
            object_is_literal=True,
            confidence=0.9,
            description="X 的发射峰位于 Y → X has_property Y",
        ),
        RelationPattern(
            pattern_name="emits_at_zh_light",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s*发射\s*(\d+\.?\d*\s*nm)\s*光?",
            relation_type=RelationType.HAS_PROPERTY.value,
            subject_group=1,
            object_group=2,
            object_is_literal=True,
            confidence=0.85,
            description="X 发射 Ynm 光 → X has_property Y",
        ),
        RelationPattern(
            pattern_name="emits_at_en",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s+emits\s+at\s+(\d+\.?\d*\s*nm)",
            relation_type=RelationType.HAS_PROPERTY.value,
            subject_group=1,
            object_group=2,
            object_is_literal=True,
            confidence=0.9,
            description="X emits at Y → X has_property Y",
        ),
        # ---- 组成关系 ----
        RelationPattern(
            pattern_name="composed_of_zh",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s*由\s*([A-Za-z0-9+\-\s,、]+?)\s*组成",
            relation_type=RelationType.PART_OF.value,
            subject_group=2,
            object_group=1,
            confidence=0.85,
            description="X 由 Y 组成 → Y part_of X",
        ),
        RelationPattern(
            pattern_name="contains_zh",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s*包含\s*([A-Za-z0-9+\-]+)",
            relation_type=RelationType.PART_OF.value,
            subject_group=2,
            object_group=1,
            confidence=0.85,
            description="X 包含 Y → Y part_of X",
        ),
        RelationPattern(
            pattern_name="composed_of_en",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s+consists\s+of\s+([A-Za-z0-9+\-\s,]+)",
            relation_type=RelationType.PART_OF.value,
            subject_group=2,
            object_group=1,
            confidence=0.85,
            description="X consists of Y → Y part_of X",
        ),
        # ---- 属性关系 ----
        RelationPattern(
            pattern_name="has_property_zh",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s*的\s*([^\s,，。；]+?)\s*为\s*([^\s,，。；]+)",
            relation_type=RelationType.HAS_PROPERTY.value,
            subject_group=1,
            object_group=3,
            object_is_literal=True,
            confidence=0.8,
            description="X 的 Y 为 Z → X has_property Z (Y 作为限定符)",
        ),
        RelationPattern(
            pattern_name="crystal_system_zh",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s*的晶系为\s*(立方|四方|正交|三方|六方|单斜|三斜|等轴)晶?系?",
            relation_type=RelationType.HAS_PROPERTY.value,
            subject_group=1,
            object_group=2,
            object_is_literal=True,
            confidence=0.9,
            description="X 的晶系为 Y → X has_property Y",
        ),
        RelationPattern(
            pattern_name="has_property_en",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s+has\s+([^\s,;]+)\s+of\s+([^\s,;]+)",
            relation_type=RelationType.HAS_PROPERTY.value,
            subject_group=1,
            object_group=3,
            object_is_literal=True,
            confidence=0.8,
            description="X has Y of Z → X has_property Z",
        ),
        # ---- 引用关系 ----
        RelationPattern(
            pattern_name="cites_zh",
            trigger_regex=r"根据\s*([A-Za-z][A-Za-z\s]+?)(?:等人|et al\.?)",
            relation_type=RelationType.CITES.value,
            subject_group=1,
            object_group=1,
            confidence=0.7,
            description="根据 X 等人 → (文档) cites X",
        ),
        RelationPattern(
            pattern_name="cites_et_al",
            trigger_regex=r"([A-Za-z][A-Za-z]+)\s+et\s+al\.?",
            relation_type=RelationType.CITES.value,
            subject_group=1,
            object_group=1,
            confidence=0.7,
            description="X et al. → (文档) cites X",
        ),
        # ---- 方法关系 ----
        RelationPattern(
            pattern_name="prepared_by_zh",
            trigger_regex=r"采用\s*([^\s,，。；]+?)\s*(?:方法\s*)?制备\s*([A-Za-z0-9+\-]+)",
            relation_type=RelationType.DERIVED_FROM.value,
            subject_group=2,
            object_group=1,
            confidence=0.85,
            description="采用 X 方法制备 Y → Y derived_from X",
        ),
        RelationPattern(
            pattern_name="synthesized_via_zh",
            trigger_regex=r"通过\s*([^\s,，。；]+?)\s*合成\s*([A-Za-z0-9+\-]+)",
            relation_type=RelationType.DERIVED_FROM.value,
            subject_group=2,
            object_group=1,
            confidence=0.85,
            description="通过 X 合成 Y → Y derived_from X",
        ),
        RelationPattern(
            pattern_name="prepared_by_en",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s+prepared\s+by\s+([^\s,;]+)",
            relation_type=RelationType.DERIVED_FROM.value,
            subject_group=1,
            object_group=2,
            confidence=0.85,
            description="Y prepared by X → Y derived_from X",
        ),
        # ---- 支持关系 ----
        RelationPattern(
            pattern_name="supports_zh",
            trigger_regex=r"([A-Za-z0-9+\-]+)\s*支持\s*([A-Za-z0-9+\-]+)",
            relation_type=RelationType.SUPPORTS.value,
            subject_group=1,
            object_group=2,
            confidence=0.75,
            description="X 支持 Y → X supports Y",
        ),
    ]

    def __init__(
        self,
        *,
        min_confidence: float = 0.4,
    ) -> None:
        """初始化关系抽取器.

        Args:
            min_confidence: 最低置信度阈值
        """
        self._min_confidence: float = min_confidence
        self._patterns: list[RelationPattern] = list(self._DEFAULT_PATTERNS)
        # 预编译正则缓存: {pattern_name: compiled_regex}
        self._compiled: dict[str, re.Pattern[str]] = {}
        self._lock = threading.RLock()
        self._recompile_all()

    def _recompile_all(self) -> None:
        """重新编译所有模式 (已持有锁)."""
        self._compiled.clear()
        for pattern in self._patterns:
            try:
                self._compiled[pattern.pattern_name] = re.compile(
                    pattern.trigger_regex
                )
            except re.error as exc:
                logger.error(
                    "模式 %s 正则编译失败: %s",
                    pattern.pattern_name,
                    exc,
                )

    # --------------------------------------------------------
    # 模式管理
    # --------------------------------------------------------

    def add_pattern(
        self,
        pattern: str,
        relation_type: str | RelationType,
        *,
        subject_group: int = 1,
        object_group: int = 2,
        object_is_literal: bool = False,
        confidence: float = 0.8,
        name: str = "",
        description: str = "",
    ) -> None:
        """添加自定义关系抽取模式.

        借鉴 OpenIE 的可扩展模式表设计，允许运行时注入领域特定模式。

        Args:
            pattern: 正则表达式字符串 (含捕获组)
            relation_type: 关系类型 (RelationType 值或自定义字符串)
            subject_group: 主语捕获组序号 (1-based)
            object_group: 宾语捕获组序号 (1-based)
            object_is_literal: 宾语是否为字面值
            confidence: 基础置信度
            name: 模式名称 (为空时自动生成)
            description: 模式描述

        Usage::

            extractor.add_pattern(
                r"(\w+)\s*属于\s*(\w+)",
                RelationType.PART_OF,
                name="belongs_to_zh",
                description="X 属于 Y → X part_of Y",
            )
        """
        if isinstance(relation_type, RelationType):
            relation_type = relation_type.value
        if not name:
            name = f"custom-{uuid.uuid4().hex[:6]}"
        new_pattern = RelationPattern(
            pattern_name=name,
            trigger_regex=pattern,
            relation_type=relation_type,
            subject_group=subject_group,
            object_group=object_group,
            object_is_literal=object_is_literal,
            confidence=confidence,
            description=description,
        )
        with self._lock:
            # 避免重名
            existing_names = {p.pattern_name for p in self._patterns}
            if name in existing_names:
                name = f"{name}-{uuid.uuid4().hex[:4]}"
                new_pattern.pattern_name = name
            self._patterns.append(new_pattern)
            try:
                self._compiled[name] = re.compile(pattern)
            except re.error as exc:
                logger.error("自定义模式 %s 编译失败: %s", name, exc)
        logger.debug("添加关系模式: %s -> %s", name, relation_type)

    def list_patterns(self) -> list[RelationPattern]:
        """列出所有模式 (副本)."""
        with self._lock:
            return [p.model_copy() for p in self._patterns]

    # --------------------------------------------------------
    # 抽取主流程
    # --------------------------------------------------------

    def extract(
        self,
        text: str,
        entities: list[KGExtractedEntity] | None = None,
    ) -> list[KGExtractedRelation]:
        """从文本中抽取关系.

        算法:
            1. 对文本应用每个模式的预编译正则
            2. 提取主语/宾语捕获组文本
            3. 若提供 entities 列表，尝试将主语/宾语与已有实体对齐
               (基于 span 重叠或名称匹配)
            4. 若宾语为字面值，直接作为字面宾语
            5. 输出 KGExtractedRelation 列表

        Args:
            text: 输入文本
            entities: 已抽取的实体列表 (用于主语/宾语对齐)

        Returns:
            抽取的关系列表 (已过滤低置信度)
        """
        if not text or not text.strip():
            return []

        results: list[KGExtractedRelation] = []
        entities = entities or []

        with self._lock:
            patterns_snapshot = list(self._patterns)
            compiled_snapshot = dict(self._compiled)

        for pattern in patterns_snapshot:
            regex = compiled_snapshot.get(pattern.pattern_name)
            if regex is None:
                continue
            for match in regex.finditer(text):
                try:
                    subject_text = match.group(pattern.subject_group)
                    object_text = match.group(pattern.object_group)
                except IndexError:
                    continue
                if not subject_text or not object_text:
                    continue

                subject_text = subject_text.strip()
                object_text = object_text.strip()
                if not subject_text or not object_text:
                    continue

                # 尝试与已有实体对齐
                subject_name = self._align_entity(
                    subject_text, match.start(pattern.subject_group),
                    match.end(pattern.subject_group), entities,
                )
                object_name = self._align_entity(
                    object_text, match.start(pattern.object_group),
                    match.end(pattern.object_group), entities,
                )

                relation = KGExtractedRelation(
                    relation_type=pattern.relation_type,
                    subject_name=subject_name,
                    object_name=object_name,
                    object_is_literal=pattern.object_is_literal,
                    subject_span=(
                        match.start(pattern.subject_group),
                        match.end(pattern.subject_group),
                    ),
                    object_span=(
                        match.start(pattern.object_group),
                        match.end(pattern.object_group),
                    ),
                    confidence=pattern.confidence,
                    trigger_text=match.group(0),
                    pattern_name=pattern.pattern_name,
                    metadata={
                        "description": pattern.description,
                    },
                )
                results.append(relation)

        # 过滤低置信度
        results = [
            r for r in results if r.confidence >= self._min_confidence
        ]
        # 去重 (相同主谓宾+关系类型)
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[KGExtractedRelation] = []
        for rel in results:
            key = (
                rel.subject_name,
                rel.relation_type,
                rel.object_name,
                rel.pattern_name,
            )
            if key not in seen:
                seen.add(key)
                deduped.append(rel)
        return deduped

    @staticmethod
    def _align_entity(
        text: str,
        start: int,
        end: int,
        entities: list[KGExtractedEntity],
    ) -> str:
        """将匹配文本与已有实体对齐.

        优先级:
            1. span 完全包含某个实体的 span → 使用该实体名称
            2. span 与某个实体的 span 重叠 → 使用该实体名称
            3. 名称与某个实体名称匹配 (忽略大小写) → 使用规范名称
            4. 无匹配 → 返回原始文本

        Args:
            text: 匹配到的文本
            start: 匹配起始位置
            end: 匹配结束位置
            entities: 已抽取的实体列表

        Returns:
            对齐后的实体名称
        """
        # 1. span 重叠对齐
        for ent in entities:
            ent_start, ent_end = ent.span
            # 完全包含
            if start <= ent_start and end >= ent_end:
                return ent.entity_name
            # 重叠
            if not (end <= ent_start or start >= ent_end):
                return ent.entity_name
        # 2. 名称匹配
        text_lower = text.lower()
        for ent in entities:
            if ent.entity_name.lower() == text_lower:
                return ent.entity_name
        # 3. 无匹配
        return text


# ============================================================
# 组件 3: KGEntityResolver — 实体消解器
# ============================================================


class UnionFind:
    """Union-Find 并查集 (带路径压缩与按秩合并).

    借鉴 Entity Resolution 中的等价类管理，用于将多个抽取实体合并为
    同一个规范实体。路径压缩使查询接近 O(1)，按秩合并保证树高 O(log n)。

    Attributes:
        parent: 父节点映射 {item: parent}
        rank: 秩 (树高近似) {item: rank}
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def make_set(self, item: str) -> None:
        """创建单元素集合."""
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        """查找根节点 (带路径压缩)."""
        if item not in self.parent:
            self.make_set(item)
        # 路径压缩
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        # 压缩路径
        while self.parent[item] != root:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item
        return root

    def union(self, item1: str, item2: str) -> str:
        """合并两个集合 (按秩合并).

        Returns:
            合并后的根节点
        """
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 == root2:
            return root1
        # 按秩合并
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
            return root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
            return root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1
            return root1

    def groups(self) -> dict[str, list[str]]:
        """返回所有等价类 {root: [members]}."""
        result: dict[str, list[str]] = {}
        for item in self.parent:
            root = self.find(item)
            result.setdefault(root, []).append(item)
        return result


class KGEntityResolver:
    """实体消解器.

    借鉴 Entity Resolution (Köpcke & Rahm, 2010) + Coreference Resolution
    + Wikidata dedup 的多策略消解方案，将不同来源抽取的同一实体合并。

    支持四级匹配策略 (按优先级递减):
    1. 标识符匹配: CAS 号 / DOI 等唯一标识相同 → 必为同一实体
    2. 精确匹配: 名称完全相同 (忽略大小写与空白)
    3. 别名匹配: 通过预定义别名表 (alias → canonical)
    4. 模糊匹配: 编辑距离 <= 2 的候选 (可选，默认关闭)

    使用 Union-Find 并查集管理等价类，保证 O(n α(n)) 的合并效率。

    线程安全:
        - _lock (RLock) 保护 _aliases 和 _fuzzy_enabled 的并发访问
        - resolve 方法内部构建独立的 UnionFind，无需加锁

    Usage::

        resolver = KGEntityResolver()
        resolver.add_alias("Dy3+", "镝离子")
        resolver.add_alias("Dy3+", "Dy(III)")
        clusters = resolver.resolve(extracted_entities)
        # 同一实体的多个抽取结果合并为一个 EntityCluster
    """

    def __init__(
        self,
        *,
        enable_fuzzy: bool = False,
        fuzzy_max_distance: int = 2,
    ) -> None:
        """初始化实体消解器.

        Args:
            enable_fuzzy: 是否启用模糊匹配 (默认关闭，性能考虑)
            fuzzy_max_distance: 模糊匹配最大编辑距离
        """
        self._aliases: dict[str, str] = {}  # alias_lower -> canonical
        self._enable_fuzzy: bool = enable_fuzzy
        self._fuzzy_max_distance: int = fuzzy_max_distance
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 别名管理
    # --------------------------------------------------------

    def add_alias(self, canonical: str, alias: str) -> None:
        """添加别名映射.

        Args:
            canonical: 规范名称
            alias: 别名 (将映射到 canonical)
        """
        if not canonical or not alias:
            return
        with self._lock:
            self._aliases[alias.strip().lower()] = canonical.strip()

    def add_aliases(
        self, canonical: str, aliases: list[str]
    ) -> None:
        """批量添加别名."""
        for alias in aliases:
            self.add_alias(canonical, alias)

    def get_canonical(self, name: str) -> str:
        """获取规范名称 (若无别名映射则返回原名)."""
        with self._lock:
            return self._aliases.get(name.strip().lower(), name.strip())

    # --------------------------------------------------------
    # 消解主流程
    # --------------------------------------------------------

    def resolve(
        self, entities: list[KGExtractedEntity]
    ) -> list[EntityCluster]:
        """实体消解: 将多个抽取实体合并为等价类簇.

        算法:
            1. 为每个实体生成唯一 ID
            2. 应用四级匹配策略，对匹配对调用 UnionFind.union
            3. 按等价类聚合，每个簇选举 canonical_entity
               (最高置信度 + 最长名称 + 词典优先)
            4. 输出 EntityCluster 列表

        Args:
            entities: 抽取的实体列表

        Returns:
            消解后的实体簇列表
        """
        if not entities:
            return []

        uf = UnionFind()
        # 为每个实体分配唯一 ID
        entity_ids: list[str] = []
        for i, ent in enumerate(entities):
            eid = f"ext-{i}"
            uf.make_set(eid)
            entity_ids.append(eid)

        # ---- 1. 标识符匹配 ----
        id_to_indices: dict[str, list[int]] = {}
        for i, ent in enumerate(entities):
            for id_type, id_value in ent.identifiers.items():
                key = f"{id_type}:{id_value.lower()}"
                id_to_indices.setdefault(key, []).append(i)
        for indices in id_to_indices.values():
            for j in range(1, len(indices)):
                uf.union(entity_ids[indices[0]], entity_ids[indices[j]])

        # ---- 2. 精确匹配 (忽略大小写/空白) ----
        name_to_indices: dict[str, list[int]] = {}
        for i, ent in enumerate(entities):
            normalized = self._normalize_name(ent.entity_name)
            name_to_indices.setdefault(normalized, []).append(i)
        for indices in name_to_indices.values():
            for j in range(1, len(indices)):
                uf.union(entity_ids[indices[0]], entity_ids[indices[j]])

        # ---- 3. 别名匹配 ----
        with self._lock:
            aliases_snapshot = dict(self._aliases)
        # 构建规范名 -> 索引列表
        canonical_to_indices: dict[str, list[int]] = {}
        for i, ent in enumerate(entities):
            canonical = aliases_snapshot.get(
                ent.entity_name.strip().lower()
            )
            if canonical:
                canonical_to_indices.setdefault(canonical, []).append(i)
        # 同一规范名的实体合并
        for indices in canonical_to_indices.values():
            for j in range(1, len(indices)):
                uf.union(entity_ids[indices[0]], entity_ids[indices[j]])
        # 别名之间若映射到同一规范名，也已合并
        # 额外: 别名 canonical 与原实体名匹配
        for i, ent in enumerate(entities):
            canonical = aliases_snapshot.get(
                ent.entity_name.strip().lower()
            )
            if canonical:
                # 查找名为 canonical 的实体
                canonical_norm = self._normalize_name(canonical)
                if canonical_norm in name_to_indices:
                    for j in name_to_indices[canonical_norm]:
                        uf.union(entity_ids[i], entity_ids[j])

        # ---- 4. 模糊匹配 (可选) ----
        if self._enable_fuzzy and self._fuzzy_max_distance > 0:
            self._fuzzy_match(entities, entity_ids, uf)

        # ---- 聚合为簇 ----
        groups = uf.groups()
        clusters: list[EntityCluster] = []
        for root, members in groups.items():
            member_indices = [
                int(mid.split("-", 1)[1]) for mid in members
            ]
            cluster_entities = [entities[i] for i in member_indices]
            cluster = self._build_cluster(cluster_entities)
            clusters.append(cluster)

        return clusters

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    @staticmethod
    def _normalize_name(name: str) -> str:
        """规范化名称: 转小写 + 去除多余空白."""
        return re.sub(r"\s+", "", name.strip().lower())

    def _fuzzy_match(
        self,
        entities: list[KGExtractedEntity],
        entity_ids: list[str],
        uf: UnionFind,
    ) -> None:
        """模糊匹配 (编辑距离 <= 阈值).

        仅对同类型实体执行模糊匹配，避免误合并。
        复杂度 O(n^2 * L^2)，n 大时建议关闭。
        """
        # 按类型分组
        type_groups: dict[EntityType, list[int]] = {}
        for i, ent in enumerate(entities):
            type_groups.setdefault(ent.entity_type, []).append(i)

        for indices in type_groups.values():
            n = len(indices)
            if n < 2:
                continue
            for a in range(n):
                for b in range(a + 1, n):
                    name_a = entities[indices[a]].entity_name
                    name_b = entities[indices[b]].entity_name
                    dist = self._levenshtein(
                        name_a.lower(), name_b.lower()
                    )
                    if 0 < dist <= self._fuzzy_max_distance:
                        uf.union(
                            entity_ids[indices[a]],
                            entity_ids[indices[b]],
                        )

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """计算 Levenshtein 编辑距离 (标准 DP 实现)."""
        if len(s1) < len(s2):
            return KGEntityResolver._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(
                    min(insertions, deletions, substitutions)
                )
            previous_row = current_row
        return previous_row[-1]

    def _build_cluster(
        self, entities: list[KGExtractedEntity]
    ) -> EntityCluster:
        """从一组等价实体构建 EntityCluster.

        规范实体选举优先级:
            1. 置信度最高
            2. 词典匹配优先 (extraction_method == "dictionary")
            3. 名称最长 (信息量最大)
        """
        if not entities:
            # 不应该发生，但防御性处理
            return EntityCluster(
                canonical_name="",
                canonical_type=EntityType.CONCEPT,
            )

        # 选举规范实体
        canonical = max(
            entities,
            key=lambda e: (
                e.confidence,
                1 if e.extraction_method == "dictionary" else 0,
                len(e.entity_name),
            ),
        )

        # 收集别名 (非规范名称)
        aliases: list[str] = []
        seen = {canonical.entity_name.lower()}
        for ent in entities:
            name_lower = ent.entity_name.lower()
            if name_lower not in seen:
                aliases.append(ent.entity_name)
                seen.add(name_lower)

        # 合并标识符
        identifiers: dict[str, str] = {}
        for ent in entities:
            for k, v in ent.identifiers.items():
                if k not in identifiers:
                    identifiers[k] = v

        # 最高置信度
        best_conf = max(e.confidence for e in entities)

        return EntityCluster(
            canonical_name=canonical.entity_name,
            canonical_type=canonical.entity_type,
            aliases=aliases,
            identifiers=identifiers,
            source_entities=entities,
            merged_count=len(entities),
            best_confidence=best_conf,
        )


# ============================================================
# 组件 4: KnowledgeGraphBuilder — 知识图谱构建器
# ============================================================


class KnowledgeGraphBuilder:
    """知识图谱增量构建器.

    整合实体抽取 → 关系抽取 → 实体消解 → 本体校验 → 图谱写入的完整流水线，
    借鉴 GraphRAG (Edge et al., 2024) 的增量构建 + ConVer-G 的版本追踪 +
    MACR 的质量控制思想。

    特点:
        - 增量构建: 新文本到达时只处理新增内容，基于 KGEntityResolver 去重
        - 去重: 基于 KGEntityResolver 避免重复实体 (标识符/名称/别名/模糊)
        - 质量控制: 置信度阈值过滤 + 本体约束校验 (可选)
        - 统计追踪: 构建统计 (新增/更新/跳过实体和三元组数)
        - 与 KnowledgeStore 无缝集成: 直接调用 store.add_entity / add_triple

    流水线:
        1. extract_entities: KGEntityExtractor.extract(text)
        2. extract_relations: RelationExtractor.extract(text, entities)
        3. resolve_entities: KGEntityResolver.resolve(entities)
        4. validate: 本体约束校验 (可选，委托 OntologyRegistry)
        5. persist: 写入 KnowledgeStore
            - 实体: 检查标识符重复 → 新增或更新 (合并别名/标识符)
            - 三元组: 检查 (s, p, o) 重复 → 新增或跳过

    线程安全:
        - _lock (RLock) 保护 _entity_index 缓存的并发访问
        - KnowledgeStore 本身线程安全

    Usage::

        store = KnowledgeStore()
        builder = KnowledgeGraphBuilder(
            store=store,
            domain="materials",
            min_entity_confidence=0.5,
            min_relation_confidence=0.5,
        )
        result = builder.build_from_text(
            "Dy3+ 掺杂 YAG 的发射波长为 580 nm",
            source_id="doc-001",
        )
    """

    def __init__(
        self,
        *,
        store: KnowledgeStore,
        domain: str = "general",
        entity_extractor: KGEntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
        entity_resolver: KGEntityResolver | None = None,
        min_entity_confidence: float = 0.4,
        min_relation_confidence: float = 0.4,
        enable_ontology_validation: bool = False,
        ontology_registry: Any | None = None,
    ) -> None:
        """初始化知识图谱构建器.

        Args:
            store: 知识存储 (必需)
            domain: 领域标识 (用于实体 domain 字段)
            entity_extractor: 自定义实体抽取器 (为空则创建默认)
            relation_extractor: 自定义关系抽取器 (为空则创建默认)
            entity_resolver: 自定义实体消解器 (为空则创建默认)
            min_entity_confidence: 实体最低置信度阈值
            min_relation_confidence: 关系最低置信度阈值
            enable_ontology_validation: 是否启用本体校验
            ontology_registry: OntologyRegistry 实例 (启用校验时必需)
        """
        self._store: KnowledgeStore = store
        self._domain: str = domain
        self._entity_extractor: KGEntityExtractor = (
            entity_extractor or KGEntityExtractor()
        )
        self._relation_extractor: RelationExtractor = (
            relation_extractor or RelationExtractor()
        )
        self._entity_resolver: KGEntityResolver = (
            entity_resolver or KGEntityResolver()
        )
        self._min_entity_confidence: float = min_entity_confidence
        self._min_relation_confidence: float = min_relation_confidence
        self._enable_ontology_validation: bool = enable_ontology_validation
        self._ontology_registry: Any | None = ontology_registry
        # 名称 -> entity_id 缓存 (增量构建时避免重复查询 store)
        self._entity_index: dict[str, str] = {}
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 扩展接口
    # --------------------------------------------------------

    def add_extraction_pattern(
        self,
        pattern: str,
        relation_type: str | RelationType,
        *,
        subject_group: int = 1,
        object_group: int = 2,
        object_is_literal: bool = False,
        confidence: float = 0.8,
        name: str = "",
        description: str = "",
    ) -> None:
        """添加自定义关系抽取模式 (委托 RelationExtractor)."""
        self._relation_extractor.add_pattern(
            pattern,
            relation_type,
            subject_group=subject_group,
            object_group=object_group,
            object_is_literal=object_is_literal,
            confidence=confidence,
            name=name,
            description=description,
        )

    def add_entity_dictionary(
        self,
        entity_type: EntityType,
        names: list[str],
        *,
        aliases: dict[str, str] | None = None,
    ) -> None:
        """添加领域词典 (委托 KGEntityExtractor)."""
        self._entity_extractor.add_dictionary(
            entity_type, names, aliases=aliases
        )
        # 同时注册别名到 KGEntityResolver
        if aliases:
            for alias, canonical in aliases.items():
                self._entity_resolver.add_alias(canonical, alias)

    # --------------------------------------------------------
    # 单文本构建
    # --------------------------------------------------------

    def build_from_text(
        self,
        text: str,
        source_id: str,
        *,
        source_meta: dict[str, Any] | None = None,
    ) -> BuildResult:
        """从单段文本构建知识图谱 (完整流水线).

        流水线:
            1. extract: KGEntityExtractor.extract(text)
            2. extract relations: RelationExtractor.extract(text, entities)
            3. resolve: KGEntityResolver.resolve(entities)
            4. validate: 本体约束校验 (可选)
            5. persist: 写入 KnowledgeStore

        Args:
            text: 输入文本
            source_id: 数据源 ID (用于溯源)
            source_meta: 源元数据 (附加到实体 metadata)

        Returns:
            构建结果统计 BuildResult
        """
        start_time = time.time()
        result = BuildResult(source_id=source_id)
        source_meta = source_meta or {}

        if not text or not text.strip():
            result.warnings.append("输入文本为空")
            result.build_time_ms = (time.time() - start_time) * 1000
            return result

        try:
            # ---- 1. 实体抽取 ----
            extracted_entities = self._entity_extractor.extract(text)
            # 置信度过滤
            filtered_entities = [
                e for e in extracted_entities
                if e.confidence >= self._min_entity_confidence
            ]
            skipped = len(extracted_entities) - len(filtered_entities)
            result.entities_skipped += skipped

            # ---- 2. 关系抽取 ----
            extracted_relations = self._relation_extractor.extract(
                text, filtered_entities
            )
            filtered_relations = [
                r for r in extracted_relations
                if r.confidence >= self._min_relation_confidence
            ]
            skipped_rel = len(extracted_relations) - len(filtered_relations)
            result.triples_skipped += skipped_rel

            # ---- 3. 实体消解 ----
            clusters = self._entity_resolver.resolve(filtered_entities)
            result.resolution_merged = sum(
                1 for c in clusters if c.merged_count > 1
            )

            # ---- 4. 持久化实体 ----
            cluster_name_to_id: dict[str, str] = {}
            for cluster in clusters:
                entity_id, action, warning = self._persist_cluster(
                    cluster, source_id, source_meta
                )
                if entity_id:
                    cluster_name_to_id[cluster.canonical_name] = entity_id
                    if action == "created":
                        result.entities_created += 1
                    elif action == "updated":
                        result.entities_updated += 1
                    result.entity_ids.append(entity_id)
                if warning:
                    result.warnings.append(warning)

            # ---- 5. 持久化三元组 ----
            for rel in filtered_relations:
                triple_id, action, warning = self._persist_relation(
                    rel, cluster_name_to_id, source_id
                )
                if triple_id:
                    if action == "created":
                        result.triples_created += 1
                        result.triple_ids.append(triple_id)
                    elif action == "skipped":
                        result.triples_skipped += 1
                if warning:
                    result.warnings.append(warning)

        except Exception as exc:
            logger.exception("构建知识图谱失败: source=%s", source_id)
            result.warnings.append(f"构建异常: {exc}")

        result.build_time_ms = (time.time() - start_time) * 1000
        return result

    # --------------------------------------------------------
    # 批量构建
    # --------------------------------------------------------

    def build_from_texts(
        self,
        texts: list[tuple[str, str]],
        *,
        source_meta: dict[str, Any] | None = None,
    ) -> BatchBuildResult:
        """批量构建知识图谱.

        Args:
            texts: [(text, source_id), ...] 列表
            source_meta: 全局源元数据

        Returns:
            批量构建结果统计 BatchBuildResult
        """
        batch = BatchBuildResult(total_texts=len(texts))
        start_time = time.time()

        for text, source_id in texts:
            try:
                result = self.build_from_text(
                    text, source_id, source_meta=source_meta
                )
                batch.results.append(result)
                if result.warnings and any(
                    "构建异常" in w for w in result.warnings
                ):
                    batch.failure_count += 1
                else:
                    batch.success_count += 1
                batch.total_entities_created += result.entities_created
                batch.total_triples_created += result.triples_created
                batch.total_resolution_merged += result.resolution_merged
            except Exception as exc:
                batch.failure_count += 1
                batch.warnings.append(
                    f"source={source_id} 构建失败: {exc}"
                )
                logger.exception(
                    "批量构建失败: source=%s", source_id
                )

        batch.total_build_time_ms = (time.time() - start_time) * 1000
        return batch

    # --------------------------------------------------------
    # 持久化辅助
    # --------------------------------------------------------

    def _persist_cluster(
        self,
        cluster: EntityCluster,
        source_id: str,
        source_meta: dict[str, Any],
    ) -> tuple[str | None, str, str]:
        """持久化实体簇到 KnowledgeStore.

        策略:
            1. 检查标识符是否已存在 → 存在则更新 (合并别名/标识符)
            2. 检查名称是否已存在 (缓存 + store) → 存在则更新
            3. 否则新建实体

        Returns:
            (entity_id, action, warning) 三元组
            action: "created" / "updated" / "skipped"
        """
        canonical_name = cluster.canonical_name
        if not canonical_name:
            return None, "skipped", "规范名称为空，跳过"

        # 1. 标识符查找
        existing_id: str | None = None
        for id_type, id_value in cluster.identifiers.items():
            found = self._store.entity_store.find_by_identifier(
                id_type, id_value
            )
            if found:
                existing_id = found.entity_id
                break

        # 2. 名称查找 (缓存优先)
        if existing_id is None:
            with self._lock:
                existing_id = self._entity_index.get(canonical_name.lower())
            if existing_id:
                # 验证 store 中仍存在
                if not self._store.entity_store.exists(existing_id):
                    with self._lock:
                        self._entity_index.pop(
                            canonical_name.lower(), None
                        )
                    existing_id = None

        if existing_id is None:
            # 查询 store
            matches = self._store.entity_store.find_by_name(canonical_name)
            if matches:
                existing_id = matches[0].entity_id

        # 3. 新建或更新
        if existing_id is None:
            # ---- 新建实体 ----
            entity = KnowledgeEntity(
                entity_type=cluster.canonical_type,
                name=canonical_name,
                aliases=cluster.aliases,
                identifiers=cluster.identifiers,
                domain=self._domain,
                confidence_score=cluster.best_confidence,
                metadata={
                    "source_id": source_id,
                    "extraction_method": (
                        cluster.source_entities[0].extraction_method
                        if cluster.source_entities
                        else "unknown"
                    ),
                    **source_meta,
                },
            )
            # 本体校验 (可选)
            if self._enable_ontology_validation and self._ontology_registry:
                violations = self._ontology_registry.validate_full(
                    self._domain,
                    cluster.canonical_type,
                    entity.properties,
                )
                if violations:
                    warning = (
                        f"实体 {canonical_name} 本体校验失败: "
                        f"{'; '.join(violations)}"
                    )
                    # 仍创建，但记录警告
                    try:
                        self._store.add_entity(entity)
                        with self._lock:
                            self._entity_index[canonical_name.lower()] = (
                                entity.entity_id
                            )
                        return entity.entity_id, "created", warning
                    except Exception as exc:
                        return None, "skipped", f"创建实体失败: {exc}"
            try:
                self._store.add_entity(entity)
                with self._lock:
                    self._entity_index[canonical_name.lower()] = (
                        entity.entity_id
                    )
                return entity.entity_id, "created", ""
            except Exception as exc:
                return None, "skipped", f"创建实体失败: {exc}"
        else:
            # ---- 更新实体 (合并别名/标识符) ----
            try:
                existing = self._store.get_entity_or_raise(existing_id)
                # 合并别名
                new_aliases = set(existing.aliases)
                new_aliases.update(cluster.aliases)
                # 合并标识符
                new_identifiers = dict(existing.identifiers)
                for k, v in cluster.identifiers.items():
                    if k not in new_identifiers:
                        new_identifiers[k] = v
                # 更新置信度 (取较高者)
                new_confidence = max(
                    existing.confidence_score, cluster.best_confidence
                )
                self._store.update_entity(
                    existing_id,
                    aliases=list(new_aliases),
                    identifiers=new_identifiers,
                    confidence_score=new_confidence,
                    changed_by="kg_builder",
                    reason=f"增量合并: source={source_id}",
                )
                with self._lock:
                    self._entity_index[canonical_name.lower()] = existing_id
                return existing_id, "updated", ""
            except Exception as exc:
                return existing_id, "updated", f"更新实体失败: {exc}"

    def _persist_relation(
        self,
        relation: KGExtractedRelation,
        cluster_name_to_id: dict[str, str],
        source_id: str,
    ) -> tuple[str | None, str, str]:
        """持久化关系到 KnowledgeStore (三元组).

        策略:
            1. 解析主语 entity_id (从 cluster_name_to_id 或 store)
            2. 解析宾语:
               - 字面值: 直接作为 object_value
               - 实体: 从 cluster_name_to_id 或 store 解析
            3. 检查 (s, p, o) 是否已存在 → 存在则跳过
            4. 创建 KnowledgeTriple 并写入

        Returns:
            (triple_id, action, warning) 三元组
        """
        subject_name = relation.subject_name
        object_name = relation.object_name

        # 解析主语
        subject_id = cluster_name_to_id.get(subject_name)
        if subject_id is None:
            subject_id = self._lookup_entity_id(subject_name)
        if subject_id is None:
            return None, "skipped", f"主语实体未找到: {subject_name}"

        # 解析宾语
        object_id: str = ""
        object_value: Any = None
        if relation.object_is_literal:
            object_value = object_name
        else:
            object_id = cluster_name_to_id.get(object_name) or ""
            if not object_id:
                object_id = self._lookup_entity_id(object_name) or ""
            if not object_id:
                return None, "skipped", f"宾语实体未找到: {object_name}"

        # 检查重复 (s, p, o)
        existing_triples = self._store.triple_store.get_by_subject_predicate(
            subject_id, relation.relation_type
        )
        for t in existing_triples:
            if relation.object_is_literal:
                if t.object_is_literal and str(t.object_value) == str(
                    object_value
                ):
                    return t.triple_id, "skipped", ""
            else:
                if not t.object_is_literal and t.object_id == object_id:
                    return t.triple_id, "skipped", ""

        # 创建三元组
        triple = KnowledgeTriple(
            subject_id=subject_id,
            predicate=relation.relation_type,
            object_id=object_id,
            object_value=object_value,
            object_is_literal=relation.object_is_literal,
            confidence=relation.confidence,
            source_id=source_id,
        )
        try:
            self._store.add_triple(triple)
            # 同时附加到主语实体的 triples 列表
            subject = self._store.get_entity(subject_id)
            if subject:
                subject.add_triple(triple)
            return triple.triple_id, "created", ""
        except Exception as exc:
            return None, "skipped", f"创建三元组失败: {exc}"

    def _lookup_entity_id(self, name: str) -> str | None:
        """通过名称查找实体 ID (缓存优先)."""
        if not name:
            return None
        name_lower = name.lower()
        with self._lock:
            cached = self._entity_index.get(name_lower)
            if cached and self._store.entity_store.exists(cached):
                return cached
            if cached:
                self._entity_index.pop(name_lower, None)
        # 查询 store
        matches = self._store.entity_store.find_by_name(name)
        if matches:
            with self._lock:
                self._entity_index[name_lower] = matches[0].entity_id
            return matches[0].entity_id
        return None

    # --------------------------------------------------------
    # 统计与诊断
    # --------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取构建器统计信息."""
        return {
            "domain": self._domain,
            "entity_count": self._store.entity_count(),
            "triple_count": self._store.triple_count(),
            "cached_entities": len(self._entity_index),
            "entity_patterns": len(
                self._entity_extractor._PATTERNS
            ),
            "relation_patterns": len(
                self._relation_extractor.list_patterns()
            ),
            "dictionaries": {
                etype.value: len(names)
                for etype, names in self._entity_extractor._dictionaries.items()
            },
            "aliases": len(self._entity_resolver._aliases),
        }


# ============================================================
# 模块导出
# ============================================================


__all__ = [
    # 枚举
    "ExtractionStrategy",
    # 数据类
    "KGExtractedEntity",
    "KGExtractedRelation",
    "EntityCluster",
    "BuildResult",
    "BatchBuildResult",
    "RelationPattern",
    # 核心组件
    "KGEntityExtractor",
    "RelationExtractor",
    "KGEntityResolver",
    "UnionFind",
    "KnowledgeGraphBuilder",
]
