"""L3 领域知识层 — 知识摄入管道.

融合世界先进方案的知识摄入与处理管道设计:
- LangChain Document Loader: 多格式文档加载 (PDF/HTML/Markdown/Word)
- LlamaIndex Node Parser: 语义分块 + 层级节点关系保留
- Unstructured.io: 非结构化文档解析 (表格/图像/公式提取)
- LlamaIndex MetadataExtractor: 自动元数据提取 (标题/实体/关键词)
- GraphRAG entity extraction: 实体关系抽取 + 知识图谱构建
- RAG-Anything multimodal: 多模态内容识别 (文本/表格/公式/图像)
- Haystack PreProcessor: 文本清洗 + 分块 + 合并策略
- Spark NLP: 领域命名实体识别 (化学/材料/物理)

六维分类体系 (D1~D6):
    D1 知识域 (KnowledgeDomain): PHYSICS/CHEMISTRY/MATERIALS/DEVICE/APPLICATION/METHODOLOGY
    D2 材料体系 (material_system): 如 "钙钛矿"/"有机-无机杂化"/"硅基"
    D3 知识层级 (KnowledgeLevel): BASIC/INTERMEDIATE/ADVANCED/TOOL
    D4 内容类型 (ContentType): LITERATURE/TEXTBOOK/CONCEPT/EXPERIMENT_DATA/INTERACTION_HISTORY
    D5 KP 锚定 (kp_anchors): 知识点锚点列表
    D6 权威度 (AuthorityTier): T1(顶级期刊) ~ T4(用户交互)

三级分块策略 (借鉴 LlamaIndex hierarchical node parser):
    L1 章级 (2000~4000 字符): 保留完整章节上下文
    L2 节级 (800~1500 字符): 主题段落聚合
    L3 段落级 (200~500 字符): 原子知识单元

线程安全: IngestionPipeline 通过 threading.RLock 保护去重集合。
所有引擎均为内存实现，接口设计支持未来替换为分布式处理后端。
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .exceptions import ChunkingError, IngestError
from .models import (
    ChunkingStrategy,
    ContentModality,
    DocumentChunk,
    EntityType,
    KnowledgeEntity,
)
from .store import KnowledgeStore

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class KnowledgeDomain(str, Enum):
    """知识域分类 (D1 维度).

    覆盖光电材料与器件研究的六大知识域:
    - PHYSICS: 物理学 (光学/电磁学/半导体物理/量子力学)
    - CHEMISTRY: 化学 (合成/表征/反应机理/分子结构)
    - MATERIALS: 材料科学 (材料设计/性能/制备工艺)
    - DEVICE: 器件工程 (结构设计/工艺集成/性能测试)
    - APPLICATION: 应用场景 (太阳能电池/LED/传感器/探测器)
    - METHODOLOGY: 方法论 (计算方法/表征技术/实验设计)
    """

    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    MATERIALS = "materials"
    DEVICE = "device"
    APPLICATION = "application"
    METHODOLOGY = "methodology"


class KnowledgeLevel(str, Enum):
    """知识层级 (D3 维度).

    对应不同深度和受众的知识分级:
    - BASIC: 基础知识 (概念定义/基本原理/入门级)
    - INTERMEDIATE: 中级知识 (详细机制/定量分析/本科级)
    - ADVANCED: 高级知识 (前沿研究/理论推导/研究生级)
    - TOOL: 工具知识 (操作流程/计算工具/实验技能)
    """

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    TOOL = "tool"


class ContentType(str, Enum):
    """内容类型 (D4 维度).

    区分知识的来源形态和呈现形式:
    - LITERATURE: 文献 (期刊论文/会议论文/预印本)
    - TEXTBOOK: 教材 (教科书/讲义/专著)
    - CONCEPT: 概念 (定义/术语/百科条目)
    - EXPERIMENT_DATA: 实验数据 (测量数据/表征结果/工艺参数)
    - INTERACTION_HISTORY: 交互历史 (问答记录/讨论/推理过程)
    """

    LITERATURE = "literature"
    TEXTBOOK = "textbook"
    CONCEPT = "concept"
    EXPERIMENT_DATA = "experiment_data"
    INTERACTION_HISTORY = "interaction_history"


class AuthorityTier(int, Enum):
    """权威度分级 (D6 维度).

    对应知识来源的可信度等级 (数值越小权威度越高):
    - T1 (1): 顶级 — 顶级期刊 (Nature/Science) + 权威手册 (CRC/NIST)
    - T2 (2): 高级 — 知名期刊 + 权威教材 + 行业标准
    - T3 (3): 中级 — 一般期刊 + 课程讲义 + 技术报告
    - T4 (4): 基础 — 用户交互 + 草稿 + 未验证内容
    """

    T1 = 1
    T2 = 2
    T3 = 3
    T4 = 4


# ============================================================
# 数据模型 (Pydantic v2)
# ============================================================


class ChunkMetadata(BaseModel):
    """切片元数据 (借鉴 LlamaIndex Metadata + Dublin Core).

    每个文档切片的六维分类元数据，支持精准检索和知识图谱构建。

    Attributes:
        knowledge_domain: D1 知识域
        material_system: D2 材料体系 (如 "钙钛矿", "有机-无机杂化")
        knowledge_level: D3 知识层级
        content_type: D4 内容类型
        kp_anchors: D5 知识点锚点列表 (如 ["KP-C-001", "KP-M-042"])
        authority_tier: D6 权威度分级
        key_concepts: 关键概念列表 (自动提取的领域术语)
        formulas: 公式列表 (如 [{"latex": "E=mc^2", "context": "..."}])
        numerical_data: 数值数据列表 (如 [{"value": 1.5, "unit": "eV", "name": "带隙"}])
        standard_references: 标准引用列表 (如 ["DOI:10.xxx", "CAS:7732-18-5"])
        prerequisite_kps: 前置知识点列表 (学习依赖关系)
    """

    knowledge_domain: KnowledgeDomain = Field(
        default=KnowledgeDomain.MATERIALS, description="D1 知识域"
    )
    material_system: str = Field(default="", description="D2 材料体系")
    knowledge_level: KnowledgeLevel = Field(
        default=KnowledgeLevel.INTERMEDIATE, description="D3 知识层级"
    )
    content_type: ContentType = Field(
        default=ContentType.CONCEPT, description="D4 内容类型"
    )
    kp_anchors: list[str] = Field(default_factory=list, description="D5 知识点锚点列表")
    authority_tier: AuthorityTier = Field(
        default=AuthorityTier.T3, description="D6 权威度分级"
    )
    key_concepts: list[str] = Field(
        default_factory=list, description="关键概念列表"
    )
    formulas: list[dict[str, Any]] = Field(
        default_factory=list, description="公式列表"
    )
    numerical_data: list[dict[str, Any]] = Field(
        default_factory=list, description="数值数据列表"
    )
    standard_references: list[str] = Field(
        default_factory=list, description="标准引用列表"
    )
    prerequisite_kps: list[str] = Field(
        default_factory=list, description="前置知识点列表"
    )


class ChunkingConfig(BaseModel):
    """分块配置 (借鉴 LlamaIndex SentenceSplitter + Haystack PreProcessor).

    三级分块策略配置，支持章级/节级/段落级分块。

    Attributes:
        min_chunk_size: 最小分块大小 (字符数，默认 200)
        max_chunk_size: 最大分块大小 (字符数，默认 2000)
        overlap: 分块重叠大小 (字符数，默认 100)
        strategy: 分块策略
        level_config: 三级分块配置
            L1 章级: {"min": 2000, "max": 4000}
            L2 节级: {"min": 800, "max": 1500}
            L3 段落级: {"min": 200, "max": 500}
    """

    min_chunk_size: int = Field(default=200, ge=50, description="最小分块大小 (字符)")
    max_chunk_size: int = Field(default=2000, ge=500, description="最大分块大小 (字符)")
    overlap: int = Field(default=100, ge=0, description="分块重叠大小 (字符)")
    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.SEMANTIC_PARAGRAPH, description="分块策略"
    )
    level_config: dict[str, dict[str, int]] = Field(
        default_factory=lambda: {
            "L1": {"min": 2000, "max": 4000},
            "L2": {"min": 800, "max": 1500},
            "L3": {"min": 200, "max": 500},
        },
        description="三级分块配置",
    )


class ClassificationResult(BaseModel):
    """分类结果 (六维分类输出).

    ClassificationEngine.classify() 的返回结果，
    包含 D1~D6 六个维度的分类结论和整体置信度。

    Attributes:
        domain: D1 知识域
        material_system: D2 材料体系
        level: D3 知识层级
        content_type: D4 内容类型
        kp_anchors: D5 知识点锚点列表
        authority_tier: D6 权威度分级
        key_concepts: 关键概念列表
        confidence: 整体分类置信度 [0.0, 1.0]
    """

    domain: KnowledgeDomain = Field(
        default=KnowledgeDomain.MATERIALS, description="D1 知识域"
    )
    material_system: str = Field(default="", description="D2 材料体系")
    level: KnowledgeLevel = Field(
        default=KnowledgeLevel.INTERMEDIATE, description="D3 知识层级"
    )
    content_type: ContentType = Field(
        default=ContentType.CONCEPT, description="D4 内容类型"
    )
    kp_anchors: list[str] = Field(default_factory=list, description="D5 知识点锚点")
    authority_tier: AuthorityTier = Field(
        default=AuthorityTier.T3, description="D6 权威度分级"
    )
    key_concepts: list[str] = Field(
        default_factory=list, description="关键概念列表"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="整体分类置信度 [0,1]"
    )


class IngestionResult(BaseModel):
    """摄入结果.

    IngestionPipeline.ingest() 的返回结果，
    统计摄入过程中的成功/失败/跳过数量和耗时。

    Attributes:
        total_chunks: 总分块数
        successful: 成功摄入数
        failed: 失败数
        skipped: 跳过数 (验证失败/重复)
        processing_time_ms: 处理耗时 (毫秒)
        errors: 错误信息列表
        chunk_ids: 成功摄入的切片 ID 列表
    """

    total_chunks: int = Field(default=0, ge=0, description="总分块数")
    successful: int = Field(default=0, ge=0, description="成功摄入数")
    failed: int = Field(default=0, ge=0, description="失败数")
    skipped: int = Field(default=0, ge=0, description="跳过数")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="处理耗时 (毫秒)"
    )
    errors: list[str] = Field(default_factory=list, description="错误信息列表")
    chunk_ids: list[str] = Field(
        default_factory=list, description="成功摄入的切片 ID 列表"
    )


# ============================================================
# ChunkingEngine — 分块引擎
# ============================================================


class ChunkingEngine:
    """文档分块引擎 (借鉴 LlamaIndex NodeParser + Haystack PreProcessor).

    支持三级分块策略:
    - L1 章级 (2000~4000 字符): 按章节标题分割，保留完整章节上下文
    - L2 节级 (800~1500 字符): 按段落聚合，主题内聚
    - L3 段落级 (200~500 字符): 按句子/段落分割，原子知识单元

    分块流程:
    1. 按策略 (章节/段落/句子) 初步分割
    2. 合并过小块 (< min_chunk_size)
    3. 拆分过大块 (> max_chunk_size)
    4. 添加重叠区域 (overlap 字符)
    5. 生成 DocumentChunk 对象，保留层级关系

    Attributes:
        _config: 分块配置
    """

    # 章节标题正则 (匹配 Markdown/纯文本标题)
    _SECTION_PATTERN = re.compile(
        r"^(?:#+\s+|第[一二三四五六七八九十\d]+[章节部分][\s.、]|"
        r"Chapter\s+\d+|Section\s+\d+|"
        r"\d+[\.\)]\s+\S)",
        re.MULTILINE,
    )
    # 段落分隔正则 (两个以上换行)
    _PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")
    # 句子分隔正则 (中英文句号/问号/叹号)
    _SENTENCE_PATTERN = re.compile(r"(?<=[。！？.!?])\s+")
    # Markdown 表格行检测
    _TABLE_ROW = re.compile(r"^\|.*\|$", re.MULTILINE)
    # Markdown 表格分隔行
    _TABLE_SEP = re.compile(r"^\|[-:\s]+\|$", re.MULTILINE)
    # LaTeX 行间公式 ($$...$$ 或 \[...\])
    _FORMULA_DISPLAY = re.compile(r"\$\$[^$]+\$\$|\\\[[^\]]+\\\]", re.DOTALL)
    # LaTeX 行内公式 ($...$)
    _FORMULA_INLINE = re.compile(r"\$[^$\n]+\$")

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """初始化分块引擎.

        Args:
            config: 分块配置，None 时使用默认配置
        """
        self._config: ChunkingConfig = config or ChunkingConfig()

    @property
    def config(self) -> ChunkingConfig:
        """分块配置."""
        return self._config

    def chunk(self, text: str, document_id: str) -> list[DocumentChunk]:
        """将文档文本分块为 DocumentChunk 列表.

        根据配置的策略执行分块，自动合并过小块和拆分过大块，
        生成带有层级关系和重叠区域的 DocumentChunk 对象。
        增强版: 自动检测表格/公式并独立成块，设置对应的 content_type。

        Args:
            text: 待分块的文档文本
            document_id: 所属文档 ID

        Returns:
            DocumentChunk 列表，按文档顺序排列

        Raises:
            ChunkingError: 分块失败
        """
        if not text or not text.strip():
            raise ChunkingError(
                document_id=document_id,
                strategy=self._config.strategy.value,
                detail="待分块文本为空",
            )

        try:
            # 步骤 1: 按策略初步分割 (返回 _ChunkItem 列表，含类型和保护标记)
            strategy = self._config.strategy
            if strategy == ChunkingStrategy.STRUCTURED_HEADING:
                items = self._chunk_by_section(text)
            elif strategy == ChunkingStrategy.SEMANTIC_PARAGRAPH:
                items = self._chunk_by_paragraph(text)
            elif strategy == ChunkingStrategy.RECURSIVE_CHAR:
                items = self._chunk_by_sentence(text)
            else:
                # FIXED_LENGTH: 按固定长度分割
                items = self._chunk_fixed_length(text)

            # 步骤 2: 合并过小块 (保护表格/公式不被合并)
            items = self._merge_small_chunks(
                items, self._config.min_chunk_size
            )

            # 步骤 3: 拆分过大块 (保护表格/公式不被拆分)
            items = self._split_large_chunks(
                items, self._config.max_chunk_size
            )

            # 步骤 4: 生成 DocumentChunk 对象
            chunks: list[DocumentChunk] = []
            for index, item in enumerate(items):
                content = item.text.strip()
                if not content:
                    continue

                # 检测章节信息
                section = self._detect_section(content)
                # 检测标题层级
                heading_level = self._detect_heading_level(content)

                chunk = DocumentChunk(
                    document_id=document_id,
                    content=content,
                    content_type=item.content_type,
                    chunk_index=index,
                    strategy=strategy,
                    overlap_prev=self._config.overlap if index > 0 else 0,
                    section=section,
                    heading_level=heading_level,
                    metadata={
                        "chunk_level": self._determine_chunk_level(len(content)),
                    },
                )
                chunks.append(chunk)

            # 步骤 5: 添加重叠区域 (向前借用 overlap 字符，跳过表格/公式块)
            if self._config.overlap > 0 and len(chunks) > 1:
                for i in range(1, len(chunks)):
                    if chunks[i].content_type in (ContentModality.TABLE, ContentModality.EQUATION):
                        continue
                    prev_content = chunks[i - 1].content
                    overlap_text = prev_content[-self._config.overlap :]
                    chunks[i].content = overlap_text + chunks[i].content
                    chunks[i].char_count = len(chunks[i].content)
                    chunks[i].token_count = max(1, len(chunks[i].content) // 4)

            logger.info(
                "文档 %s 分块完成: %d 个切片 (策略=%s, 表格=%d, 公式=%d)",
                document_id,
                len(chunks),
                strategy.value,
                sum(1 for c in chunks if c.content_type == ContentModality.TABLE),
                sum(1 for c in chunks if c.content_type == ContentModality.EQUATION),
            )
            return chunks

        except ChunkingError:
            raise
        except Exception as exc:
            raise ChunkingError(
                document_id=document_id,
                strategy=self._config.strategy.value,
                detail=f"分块失败: {exc}",
            ) from exc

    # --------------------------------------------------------
    # _ChunkItem — 内部结构化切片项
    # --------------------------------------------------------

    class _ChunkItem:
        """内部切片项，携带内容和类型信息，支持保护标记防止合并/拆分。"""
        __slots__ = ('text', 'content_type', 'protected')

        def __init__(self, text: str, content_type: ContentModality = ContentModality.TEXT, protected: bool = False):
            self.text = text
            self.content_type = content_type
            self.protected = protected

    # --------------------------------------------------------
    # 内容类型检测
    # --------------------------------------------------------

    @staticmethod
    def _detect_content_type(content: str) -> ContentModality:
        """检测文本内容类型: 表格/公式/普通文本.

        Args:
            content: 待检测文本

        Returns:
            内容模态类型
        """
        text = content.strip()
        if not text:
            return ContentModality.TEXT
        # 检测表格: 至少包含一个表头行和一个分隔行
        lines = text.split('\n')
        if len(lines) >= 2:
            # 检查是否包含表格分隔行 (|---|)
            has_sep = any(ChunkingEngine._TABLE_SEP.match(l) for l in lines)
            # 检查是否至少有两行以 | 开头
            pipe_count = sum(1 for l in lines if l.strip().startswith('|') and l.strip().endswith('|'))
            if has_sep and pipe_count >= 2:
                return ContentModality.TABLE
        # 检测行间公式 ($$...$$ 或 \[...\])
        if ChunkingEngine._FORMULA_DISPLAY.search(text):
            return ContentModality.EQUATION
        # 检测行内公式密度 (含 $...$ 且有公式特征词)
        inline_matches = ChunkingEngine._FORMULA_INLINE.findall(text)
        if len(inline_matches) >= 2:
            return ContentModality.EQUATION
        if len(inline_matches) == 1 and len(text) < 200:
            return ContentModality.EQUATION
        return ContentModality.TEXT

    # --------------------------------------------------------
    # 分块策略实现
    # --------------------------------------------------------

    def _chunk_by_section(self, text: str) -> list['ChunkingEngine._ChunkItem']:
        """按章节分块 (L1 章级策略).

        使用章节标题正则识别章节边界，每个章节作为一个分块。

        Args:
            text: 待分块文本

        Returns:
            _ChunkItem 列表
        """
        # 查找所有章节标题位置
        matches = list(self._SECTION_PATTERN.finditer(text))
        if not matches:
            # 无章节标题，退化为段落分块
            return self._chunk_by_paragraph(text)

        raw_chunks: list[str] = []
        # 第一个标题之前的内容
        if matches[0].start() > 0:
            prefix = text[: matches[0].start()].strip()
            if prefix:
                raw_chunks.append(prefix)

        # 每个标题到下一个标题之间的内容
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                raw_chunks.append(section_text)

        return [self._ChunkItem(c, self._detect_content_type(c)) for c in raw_chunks]

    def _chunk_by_paragraph(self, text: str) -> list['ChunkingEngine._ChunkItem']:
        """按段落分块 (L2 节级策略)，增强版自动检测表格和公式.

        使用双换行符识别段落边界，同时检测并标记表格/公式块，
        使其在后续的合并/拆分步骤中保持独立。

        Args:
            text: 待分块文本

        Returns:
            _ChunkItem 列表 (含 content_type 和保护标记)
        """
        # 步骤 1: 提取行间公式块 ($$...$$ 或 \[...\])，用占位符替换
        formula_blocks: dict[str, str] = {}
        def _replace_formula(m):
            fid = f"\x00FORMULA_{len(formula_blocks)}\x00"
            formula_blocks[fid] = m.group(0)
            return fid
        text_clean = self._FORMULA_DISPLAY.sub(_replace_formula, text)

        # 步骤 2: 提取表格块 (连续 | 行，含分隔行)
        table_blocks: dict[str, str] = {}
        def _extract_tables(t: str) -> str:
            lines = t.split('\n')
            result = []
            i = 0
            while i < len(lines):
                line = lines[i]
                # 检测表格起始: 行以 | 开头
                if line.strip().startswith('|') and line.strip().endswith('|'):
                    table_lines = [line]
                    i += 1
                    # 收集后续表格行 (连续 | 行)
                    while i < len(lines):
                        l = lines[i].strip()
                        if l.startswith('|') and l.endswith('|'):
                            table_lines.append(lines[i])
                            i += 1
                        else:
                            break
                    # 判断是否包含分隔行 (|---|)
                    if any(self._TABLE_SEP.match(l) for l in table_lines):
                        tid = f"\x00TABLE_{len(table_blocks)}\x00"
                        table_blocks[tid] = '\n'.join(table_lines)
                        result.append(tid)
                    else:
                        result.extend(table_lines)
                else:
                    result.append(line)
                    i += 1
            return '\n'.join(result)
        text_clean = _extract_tables(text_clean)

        # 步骤 3: 按段落分割剩余文本
        paragraphs = self._PARAGRAPH_PATTERN.split(text_clean)

        # 步骤 4: 还原占位符并创建 _ChunkItem
        items: list[ChunkingEngine._ChunkItem] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 按位置依次替换占位符: 处理段落中的公式/表格占位符
            # 有些段落可能包含多个占位符 (如 "前言 \x00TABLE_0\x00 结论")
            while True:
                tidx = para.find('\x00TABLE_')
                fidx = para.find('\x00FORMULA_')
                if tidx == -1 and fidx == -1:
                    break
                # 找到最靠前的占位符
                earliest = min(
                    (tidx if tidx >= 0 else len(para)),
                    (fidx if fidx >= 0 else len(para))
                )
                # 占位符前的文本
                prefix = para[:earliest].strip()
                if prefix:
                    items.append(self._ChunkItem(prefix, self._detect_content_type(prefix)))
                # 占位符本身
                marker_end = para.index('\x00', earliest + 1) + 1
                marker = para[earliest:marker_end]
                if marker in table_blocks:
                    items.append(self._ChunkItem(table_blocks[marker], ContentModality.TABLE, protected=True))
                elif marker in formula_blocks:
                    items.append(self._ChunkItem(formula_blocks[marker], ContentModality.EQUATION, protected=True))
                para = para[marker_end:].strip()
            if para:
                items.append(self._ChunkItem(para, self._detect_content_type(para)))

        return items

    def _chunk_by_sentence(self, text: str) -> list['ChunkingEngine._ChunkItem']:
        """按句子分块 (L3 段落级策略).

        使用句号/问号/叹号识别句子边界，每个句子作为一个分块。
        适用于需要细粒度分割的场景。

        Args:
            text: 待分块文本

        Returns:
            _ChunkItem 列表
        """
        # 先按段落分割，再按句子分割
        paragraphs = self._PARAGRAPH_PATTERN.split(text)
        sentences: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # 按句子分割
            parts = self._SENTENCE_PATTERN.split(para)
            sentences.extend(s.strip() for s in parts if s.strip())
        return [self._ChunkItem(s, self._detect_content_type(s)) for s in sentences]

    def _chunk_fixed_length(self, text: str) -> list['ChunkingEngine._ChunkItem']:
        """按固定长度分块 (FIXED_LENGTH 策略).

        Args:
            text: 待分块文本

        Returns:
            _ChunkItem 列表
        """
        max_size = self._config.max_chunk_size
        chunks: list[str] = []
        for i in range(0, len(text), max_size):
            chunk = text[i : i + max_size]
            if chunk.strip():
                chunks.append(chunk)
        return [self._ChunkItem(c, self._detect_content_type(c)) for c in chunks]

    # --------------------------------------------------------
    # 块调整策略
    # --------------------------------------------------------

    def _merge_small_chunks(
        self, items: list['ChunkingEngine._ChunkItem'], min_size: int
    ) -> list['ChunkingEngine._ChunkItem']:
        """合并过小块 (借鉴 Haystack PreProcessor merge策略).

        将小于 min_size 的相邻块合并，直到达到最小尺寸要求。
        受保护块 (表格/公式) 独立保留，不参与合并。

        Args:
            items: 待合并的切片项列表
            min_size: 最小块大小 (字符)

        Returns:
            合并后的切片项列表
        """
        if not items:
            return []

        merged: list[ChunkingEngine._ChunkItem] = []
        buffer_text = ""
        buffer_types: list[ContentModality] = []

        for item in items:
            # 保护块: 直接输出，不清空缓冲区
            if item.protected:
                if buffer_text.strip():
                    merged.append(self._ChunkItem(
                        buffer_text.strip(),
                        ContentModality.MIXED if len(set(buffer_types)) > 1 else buffer_types[0],
                    ))
                    buffer_text = ""
                    buffer_types = []
                merged.append(item)
                continue

            if len(buffer_text) + len(item.text) < min_size:
                # 合并到缓冲区
                if buffer_text:
                    buffer_text += "\n\n" + item.text
                else:
                    buffer_text = item.text
                buffer_types.append(item.content_type)
            else:
                # 缓冲区已达最小尺寸
                if buffer_text:
                    merged.append(self._ChunkItem(
                        buffer_text.strip(),
                        ContentModality.MIXED if len(set(buffer_types)) > 1 else buffer_types[0],
                    ))
                buffer_text = item.text
                buffer_types = [item.content_type]

        # 处理最后剩余的缓冲区
        if buffer_text.strip():
            if merged and len(buffer_text) < min_size and not merged[-1].protected:
                # 最后一个小块合并到前一块 (仅当前一块非保护)
                merged[-1] = self._ChunkItem(
                    merged[-1].text + "\n\n" + buffer_text,
                    ContentModality.MIXED
                    if merged[-1].content_type != ContentModality.TEXT and len(buffer_types) > 0
                    else merged[-1].content_type,
                )
            else:
                merged.append(self._ChunkItem(
                    buffer_text.strip(),
                    ContentModality.MIXED if len(set(buffer_types)) > 1 else buffer_types[0],
                ))

        return merged

    def _split_large_chunks(
        self, items: list['ChunkingEngine._ChunkItem'], max_size: int
    ) -> list['ChunkingEngine._ChunkItem']:
        """拆分过大块 (借鉴 LlamaIndex SentenceSplitter).

        将大于 max_size 的块按句子边界拆分，直到每个块不超过最大尺寸。
        受保护块 (表格/公式) 独立保留，不参与拆分。

        Args:
            items: 待拆分的切片项列表
            max_size: 最大块大小 (字符)

        Returns:
            拆分后的切片项列表
        """
        result: list[ChunkingEngine._ChunkItem] = []
        for item in items:
            # 保护块或未超限: 原样保留
            if item.protected or len(item.text) <= max_size:
                result.append(item)
                continue

            # 按句子拆分
            sentences = self._SENTENCE_PATTERN.split(item.text)
            if len(sentences) <= 1:
                # 无法按句子拆分，强制按字符拆分
                for i in range(0, len(item.text), max_size):
                    seg = item.text[i: i + max_size].strip()
                    if seg:
                        result.append(self._ChunkItem(seg, item.content_type))
                continue

            # 贪心合并句子，不超过 max_size
            buffer = ""
            for sentence in sentences:
                if len(buffer) + len(sentence) > max_size and buffer:
                    result.append(self._ChunkItem(buffer.strip(), item.content_type))
                    buffer = sentence
                else:
                    buffer = buffer + sentence if buffer else sentence
            if buffer.strip():
                result.append(self._ChunkItem(buffer.strip(), item.content_type))

        return result

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    def _detect_section(self, content: str) -> str:
        """检测切片所属章节.

        Args:
            content: 切片文本内容

        Returns:
            章节标题 (无标题时返回空字符串)
        """
        first_line = content.split("\n", 1)[0].strip()
        if self._SECTION_PATTERN.match(first_line):
            return first_line.lstrip("#").strip()
        return ""

    def _detect_heading_level(self, content: str) -> int:
        """检测标题层级.

        Args:
            content: 切片文本内容

        Returns:
            标题层级 (0=非标题, 1=H1, 2=H2, ...)
        """
        first_line = content.split("\n", 1)[0].strip()
        if first_line.startswith("#"):
            return len(first_line) - len(first_line.lstrip("#"))
        if self._SECTION_PATTERN.match(first_line):
            return 1
        return 0

    def _determine_chunk_level(self, char_count: int) -> str:
        """根据字符数判定分块层级.

        Args:
            char_count: 字符数

        Returns:
            层级标识 ("L1"/"L2"/"L3")
        """
        level_config = self._config.level_config
        l1_max = level_config.get("L1", {}).get("max", 4000)
        l2_max = level_config.get("L2", {}).get("max", 1500)
        l3_max = level_config.get("L3", {}).get("max", 500)

        if char_count > l2_max:
            return "L1"
        if char_count > l3_max:
            return "L2"
        return "L3"


# ============================================================
# ClassificationEngine — 分类引擎
# ============================================================


class ClassificationEngine:
    """六维分类引擎 (借鉴 LlamaIndex MetadataExtractor + GraphRAG entity extraction).

    对文档内容执行六个维度的自动分类:
    - D1 知识域分类: 基于领域关键词匹配
    - D2 材料体系分类: 基于材料名称识别
    - D3 知识层级分类: 基于内容深度和术语密度
    - D4 内容类型分类: 基于内容结构特征
    - D5 KP 锚定: 基于知识点模式提取
    - D6 权威度评估: 基于来源元数据

    分类结果用于:
    - 切片元数据标注 (ChunkMetadata)
    - 检索过滤和排序
    - 知识图谱构建
    - 学习路径推荐
    """

    # D1 知识域关键词映射
    _DOMAIN_KEYWORDS: dict[KnowledgeDomain, list[str]] = {
        KnowledgeDomain.PHYSICS: [
            "能带", "带隙", "载流子", "激子", "光子", "量子效率",
            "费米能级", "态密度", "光学常数", "折射率", "吸收系数",
            "漂移", "扩散", "复合", "隧穿", "量子力学", "薛定谔",
        ],
        KnowledgeDomain.CHEMISTRY: [
            "合成", "反应", "分子", "化学键", "晶体结构", "溶剂",
            "配体", "官能团", "氧化还原", "聚合", "交联", "分解",
            "滴定", "色谱", "质谱", "核磁", "结晶", "溶解度",
        ],
        KnowledgeDomain.MATERIALS: [
            "薄膜", "晶体", "钙钛矿", "聚合物", "复合材料", "纳米材料",
            "掺杂", "退火", "溅射", "沉积", "旋涂", "蒸镀",
            "形貌", "晶界", "缺陷", "应力", "相变", "热稳定性",
        ],
        KnowledgeDomain.DEVICE: [
            "器件", "电池", "二极管", "晶体管", "电极", "界面",
            "异质结", "p-n结", "效率", "填充因子", "开路电压",
            "短路电流", "I-V曲线", "EQE", "暗电流", "串联电阻",
        ],
        KnowledgeDomain.APPLICATION: [
            "太阳能电池", "LED", "发光", "传感器", "探测器",
            "光伏", "照明", "显示", "成像", "产业化", "商业化",
            "成本", "稳定性", "寿命", "封装", "模块",
        ],
        KnowledgeDomain.METHODOLOGY: [
            "计算", "模拟", "仿真", "第一性原理", "DFT", "分子动力学",
            "有限元", "蒙特卡洛", "表征", "测试", "测量",
            "XRD", "SEM", "TEM", "AFM", "PL", "紫外可见",
        ],
    }

    # D2 材料体系关键词映射
    _MATERIAL_SYSTEMS: dict[str, list[str]] = {
        "钙钛矿": ["钙钛矿", "perovskite", "MAPbI", "FAPbI", "CsPb"],
        "有机-无机杂化": ["有机-无机", "杂化", "hybrid", "MOF"],
        "硅基": ["硅", "silicon", "Si", "单晶硅", "多晶硅"],
        "化合物半导体": ["GaAs", "InP", "GaN", "CdTe", "CIGS", "CZTS"],
        "有机半导体": ["PEDOT", "P3HT", "PCBM", "有机半导体", "共轭聚合物"],
        "量子点": ["量子点", "quantum dot", "QD", "纳米晶"],
        "二维材料": ["石墨烯", "graphene", "MoS2", "WSe2", "黑磷"],
        "氧化物": ["TiO2", "ZnO", "NiO", "IGZO", "氧化物"],
    }

    # D4 内容类型特征关键词
    _CONTENT_TYPE_KEYWORDS: dict[ContentType, list[str]] = {
        ContentType.LITERATURE: [
            "Abstract", "摘要", "引言", "Introduction", "参考文献",
            "References", "DOI", "et al.", "如图所示", "文献报道",
        ],
        ContentType.TEXTBOOK: [
            "例题", "习题", "思考题", "本章小结", "定义",
            "定理", "证明", "推论", "第\\d+章", "练习",
        ],
        ContentType.CONCEPT: [
            "是指", "定义为", "概念", "术语", "简称",
            "又称", "即", "是指", "表示",
        ],
        ContentType.EXPERIMENT_DATA: [
            "表\\d+", "Figure \\d+", "实验数据", "测量结果",
            "数据如下", "参数", "条件", "Tab\\.", "Fig\\.",
        ],
        ContentType.INTERACTION_HISTORY: [
            "请问", "回答", "用户", "助手", "交互",
            "对话", "问答", "建议", "我认为",
        ],
    }

    def __init__(self) -> None:
        """初始化分类引擎."""
        # 预编译正则模式
        self._kp_pattern = re.compile(r"KP-[A-Z]-\d{3,4}")
        self._formula_pattern = re.compile(r"\$[^$]+\$|\\\[[^\]]+\\")
        self._number_pattern = re.compile(
            r"(?<![\w.])(\d+\.?\d*)\s*(eV|nm|nm\^2|cm-1|K|V|A|mA|W|mW|%)"
        )
        self._reference_pattern = re.compile(
            r"(?:DOI:\s*[\d./\w-]+|CAS:\s*[\d-]+|arXiv:\s*[\d.]+)"
        )

    def classify(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> ClassificationResult:
        """对内容执行六维分类.

        Args:
            content: 待分类的文本内容
            metadata: 来源元数据 (用于权威度评估)

        Returns:
            分类结果 (含 D1~D6 六个维度和置信度)
        """
        metadata = metadata or {}

        # D1: 知识域分类
        domain = self._classify_domain(content)
        # D2: 材料体系分类
        material_system = self._classify_material(content)
        # D3: 知识层级分类
        level = self._classify_level(content)
        # D4: 内容类型分类
        content_type = self._classify_type(content)
        # D5: KP 锚定
        kp_anchors = self._extract_kp_anchors(content)
        # D6: 权威度评估
        authority_tier = self._assess_authority(metadata)
        # 关键概念提取
        key_concepts = self._extract_key_concepts(content)

        # 计算综合置信度
        confidence = self._calculate_confidence(
            domain, material_system, level, content_type, content
        )

        return ClassificationResult(
            domain=domain,
            material_system=material_system,
            level=level,
            content_type=content_type,
            kp_anchors=kp_anchors,
            authority_tier=authority_tier,
            key_concepts=key_concepts,
            confidence=confidence,
        )

    def _classify_domain(self, content: str) -> KnowledgeDomain:
        """D1 知识域分类 (基于关键词匹配).

        统计各知识域关键词在内容中的出现频率，取最高分域。

        Args:
            content: 文本内容

        Returns:
            匹配度最高的知识域
        """
        scores: dict[KnowledgeDomain, int] = {}
        content_lower = content.lower()

        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            score = sum(
                1 for kw in keywords if kw.lower() in content_lower
            )
            scores[domain] = score

        # 取最高分域，平局时取 MATERIALS (默认域)
        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return KnowledgeDomain.MATERIALS

        for domain in KnowledgeDomain:
            if scores.get(domain, 0) == max_score:
                return domain

        return KnowledgeDomain.MATERIALS

    def _classify_material(self, content: str) -> str:
        """D2 材料体系分类 (基于材料名称识别).

        匹配预定义的材料体系关键词，返回匹配到的材料体系名称。

        Args:
            content: 文本内容

        Returns:
            材料体系名称 (未匹配时返回空字符串)
        """
        content_lower = content.lower()
        matched_systems: list[str] = []

        for system, keywords in self._MATERIAL_SYSTEMS.items():
            for kw in keywords:
                if kw.lower() in content_lower:
                    matched_systems.append(system)
                    break

        if not matched_systems:
            return ""

        # 多个匹配时返回全部 (用 "/" 分隔)
        return "/".join(matched_systems)

    def _classify_level(self, content: str) -> KnowledgeLevel:
        """D3 知识层级分类 (基于内容深度和术语密度).

        根据术语密度、公式数量和内容复杂度判定知识层级:
        - 高术语密度 + 多公式 → ADVANCED
        - 中术语密度 → INTERMEDIATE
        - 低术语密度 + 简单表述 → BASIC
        - 操作流程/工具描述 → TOOL

        Args:
            content: 文本内容

        Returns:
            知识层级
        """
        # 检测操作流程特征
        tool_indicators = ["步骤", "操作", "流程", "方法如下", "Step", "Procedure"]
        if any(indicator in content for indicator in tool_indicators):
            return KnowledgeLevel.TOOL

        # 统计术语密度
        total_terms = 0
        content_lower = content.lower()
        for keywords in self._DOMAIN_KEYWORDS.values():
            total_terms += sum(
                1 for kw in keywords if kw.lower() in content_lower
            )

        # 统计公式数量
        formula_count = len(self._formula_pattern.findall(content))

        # 判定层级
        char_count = max(len(content), 1)
        term_density = total_terms / char_count * 1000  # 每千字符术语数

        if term_density > 10 or formula_count > 2:
            return KnowledgeLevel.ADVANCED
        if term_density > 3:
            return KnowledgeLevel.INTERMEDIATE
        return KnowledgeLevel.BASIC

    def _classify_type(self, content: str) -> ContentType:
        """D4 内容类型分类 (基于内容结构特征).

        匹配各内容类型的特征关键词，取匹配度最高的类型。

        Args:
            content: 文本内容

        Returns:
            内容类型
        """
        scores: dict[ContentType, int] = {}

        for content_type, keywords in self._CONTENT_TYPE_KEYWORDS.items():
            score = 0
            for kw in keywords:
                try:
                    if re.search(kw, content, re.IGNORECASE):
                        score += 1
                except re.error:
                    if kw.lower() in content.lower():
                        score += 1
            scores[content_type] = score

        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return ContentType.CONCEPT

        for ct in ContentType:
            if scores.get(ct, 0) == max_score:
                return ct

        return ContentType.CONCEPT

    def _extract_kp_anchors(self, content: str) -> list[str]:
        """D5 KP 锚点提取 (基于知识点 ID 模式).

        识别内容中引用的知识点锚点 (如 KP-C-001, KP-M-042)。

        Args:
            content: 文本内容

        Returns:
            知识点锚点列表 (去重)
        """
        matches = self._kp_pattern.findall(content)
        # 去重并保持顺序
        seen: set[str] = set()
        result: list[str] = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def _assess_authority(self, metadata: dict[str, Any]) -> AuthorityTier:
        """D6 权威度评估 (基于来源元数据).

        根据来源类型、期刊等级、是否经同行评审等元数据评估权威度。

        Args:
            metadata: 来源元数据

        Returns:
            权威度分级
        """
        source_type = metadata.get("source_type", "")
        journal = metadata.get("journal", "")
        is_peer_reviewed = metadata.get("peer_reviewed", False)
        is_verified = metadata.get("verified", False)

        # T1: 顶级期刊/权威手册
        top_journals = {"nature", "science", "nature energy", "nature materials"}
        if journal.lower() in top_journals:
            return AuthorityTier.T1

        # T1: 权威数据库
        if source_type in {"nist", "crc_handbook", "pubchem"}:
            return AuthorityTier.T1

        # T2: 知名期刊 + 同行评审
        if is_peer_reviewed or source_type in {"journal", "textbook", "standard"}:
            return AuthorityTier.T2

        # T3: 一般来源
        if source_type in {"conference", "lecture", "report", "thesis"}:
            return AuthorityTier.T3

        # T4: 用户交互/未验证
        if source_type in {"user_input", "draft", "interaction"} or not is_verified:
            return AuthorityTier.T4

        return AuthorityTier.T3

    def _extract_key_concepts(self, content: str) -> list[str]:
        """关键概念提取 (基于领域术语识别).

        从内容中提取领域关键术语，用于知识图谱构建和检索增强。

        Args:
            content: 文本内容

        Returns:
            关键概念列表 (去重，最多 20 个)
        """
        concepts: list[str] = []
        content_lower = content.lower()

        # 从各域关键词中提取匹配的术语
        for keywords in self._DOMAIN_KEYWORDS.values():
            for kw in keywords:
                if kw.lower() in content_lower and kw not in concepts:
                    concepts.append(kw)

        # 从材料体系关键词中提取
        for keywords in self._MATERIAL_SYSTEMS.values():
            for kw in keywords:
                if kw.lower() in content_lower and kw not in concepts:
                    concepts.append(kw)

        # 限制数量
        return concepts[:20]

    def _calculate_confidence(
        self,
        domain: KnowledgeDomain,
        material_system: str,
        level: KnowledgeLevel,
        content_type: ContentType,
        content: str,
    ) -> float:
        """计算分类置信度.

        根据各维度匹配强度计算综合置信度。

        Args:
            domain: D1 分类结果
            material_system: D2 分类结果
            level: D3 分类结果
            content_type: D4 分类结果
            content: 原始内容

        Returns:
            置信度 [0.0, 1.0]
        """
        score = 0.0

        # 知识域匹配强度
        content_lower = content.lower()
        domain_keywords = self._DOMAIN_KEYWORDS.get(domain, [])
        domain_hits = sum(1 for kw in domain_keywords if kw.lower() in content_lower)
        if domain_hits > 0:
            score += min(domain_hits / 5.0, 1.0) * 0.3

        # 材料体系匹配
        if material_system:
            score += 0.2

        # 内容类型匹配
        type_keywords = self._CONTENT_TYPE_KEYWORDS.get(content_type, [])
        type_hits = sum(
            1 for kw in type_keywords
            if kw.lower() in content_lower
        )
        if type_hits > 0:
            score += min(type_hits / 3.0, 1.0) * 0.2

        # 内容长度 (过短的内容置信度低)
        if len(content) > 100:
            score += 0.15
        elif len(content) > 50:
            score += 0.1

        # 层级检测置信度 (TOOL 和 ADVANCED 更可靠)
        if level in (KnowledgeLevel.TOOL, KnowledgeLevel.ADVANCED):
            score += 0.15
        else:
            score += 0.1

        return min(score, 1.0)


# ============================================================
# IngestionPipeline — 知识摄入管道
# ============================================================


class IngestionPipeline:
    """知识摄入管道 (借鉴 LangChain ingestion pipeline + LlamaIndex ingestion).

    端到端知识摄入流程:
    1. 文档分块 (ChunkingEngine)
    2. 六维分类 (ClassificationEngine)
    3. 块验证 (内容非空、长度合规)
    4. 去重检查 (内容哈希)
    5. 元数据标注 (ChunkMetadata)
    6. 存储写入 (KnowledgeStore.add_chunk)
    7. 实体抽取 (可选，生成 KnowledgeEntity)
    8. 结果统计 (IngestionResult)

    支持单文档摄入和批量摄入两种模式。

    Attributes:
        _store: 知识存储引擎
        _chunker: 分块引擎
        _classifier: 分类引擎
        _content_hashes: 已摄入内容哈希集合 (用于去重)
        _stats: 摄入统计
        _lock: 线程安全锁
    """

    def __init__(
        self,
        store: KnowledgeStore,
        chunker: ChunkingEngine | None = None,
        classifier: ClassificationEngine | None = None,
    ) -> None:
        """初始化知识摄入管道.

        Args:
            store: 知识存储引擎
            chunker: 分块引擎 (None 时使用默认配置)
            classifier: 分类引擎 (None 时创建默认实例)
        """
        self._store: KnowledgeStore = store
        self._chunker: ChunkingEngine = chunker or ChunkingEngine()
        self._classifier: ClassificationEngine = classifier or ClassificationEngine()
        self._content_hashes: set[str] = set()
        self._stats: dict[str, Any] = {
            "total_ingested": 0,
            "total_failed": 0,
            "total_skipped": 0,
            "total_documents": 0,
        }
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 摄入入口
    # --------------------------------------------------------

    def ingest(
        self,
        content: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """摄入单个文档.

        执行完整的摄入流程: 分块 → 分类 → 验证 → 去重 → 存储。

        Args:
            content: 文档文本内容
            document_id: 文档唯一标识
            metadata: 来源元数据 (用于分类和权威度评估)

        Returns:
            摄入结果统计

        Raises:
            IngestError: 摄入过程发生严重错误
        """
        start_time = time.time()
        metadata = metadata or {}
        errors: list[str] = []
        chunk_ids: list[str] = []
        successful = 0
        failed = 0
        skipped = 0

        try:
            # 步骤 1: 文档分块
            chunks = self._chunker.chunk(content, document_id)
        except ChunkingError as exc:
            raise IngestError(
                source=document_id,
                count=1,
                detail=f"分块失败: {exc}",
            ) from exc

        total_chunks = len(chunks)

        # 步骤 2: 逐块处理
        for chunk in chunks:
            try:
                # 步骤 2a: 验证块
                if not self._validate_chunk(chunk):
                    skipped += 1
                    continue

                # 步骤 2b: 去重检查
                content_hash = self._compute_hash(chunk.content)
                if self._deduplicate(content_hash):
                    skipped += 1
                    continue

                # 步骤 2c: 六维分类
                classification = self._classifier.classify(
                    chunk.content, metadata
                )

                # 步骤 2d: 处理并存储
                chunk_id = self._process_chunk(
                    chunk, document_id, classification
                )
                if chunk_id is not None:
                    chunk_ids.append(chunk_id)
                    successful += 1
                    with self._lock:
                        self._content_hashes.add(content_hash)
                else:
                    failed += 1
                    errors.append(f"块 {chunk.chunk_index} 存储失败")

            except Exception as exc:
                failed += 1
                errors.append(f"块 {chunk.chunk_index} 处理异常: {exc}")
                logger.exception("文档 %s 块 %d 处理异常",
                                 document_id, chunk.chunk_index)

        # 步骤 3: 更新统计
        elapsed_ms = (time.time() - start_time) * 1000
        with self._lock:
            self._stats["total_ingested"] += successful
            self._stats["total_failed"] += failed
            self._stats["total_skipped"] += skipped
            self._stats["total_documents"] += 1

        result = IngestionResult(
            total_chunks=total_chunks,
            successful=successful,
            failed=failed,
            skipped=skipped,
            processing_time_ms=round(elapsed_ms, 2),
            errors=errors,
            chunk_ids=chunk_ids,
        )

        logger.info(
            "文档 %s 摄入完成: %d/%d 成功, %d 跳过, %d 失败 (%.1fms)",
            document_id, successful, total_chunks, skipped, failed, elapsed_ms,
        )
        return result

    def ingest_batch(self, items: list[dict[str, Any]]) -> IngestionResult:
        """批量摄入多个文档.

        Args:
            items: 文档列表，每个元素为字典:
                - content: 文档文本 (必需)
                - document_id: 文档 ID (必需)
                - metadata: 来源元数据 (可选)

        Returns:
            汇总摄入结果统计

        Raises:
            IngestError: 批量摄入发生严重错误
        """
        start_time = time.time()
        total_successful = 0
        total_failed = 0
        total_skipped = 0
        total_chunks = 0
        all_errors: list[str] = []
        all_chunk_ids: list[str] = []

        for i, item in enumerate(items):
            content = item.get("content", "")
            document_id = item.get("document_id", f"batch-{i}")
            metadata = item.get("metadata")

            if not content:
                all_errors.append(f"文档 {document_id} 内容为空，跳过")
                total_failed += 1
                continue

            try:
                result = self.ingest(content, document_id, metadata)
                total_successful += result.successful
                total_failed += result.failed
                total_skipped += result.skipped
                total_chunks += result.total_chunks
                all_errors.extend(result.errors)
                all_chunk_ids.extend(result.chunk_ids)
            except IngestError as exc:
                all_errors.append(f"文档 {document_id} 摄入失败: {exc}")
                total_failed += 1

        elapsed_ms = (time.time() - start_time) * 1000
        return IngestionResult(
            total_chunks=total_chunks,
            successful=total_successful,
            failed=total_failed,
            skipped=total_skipped,
            processing_time_ms=round(elapsed_ms, 2),
            errors=all_errors,
            chunk_ids=all_chunk_ids,
        )

    # --------------------------------------------------------
    # 内部处理方法
    # --------------------------------------------------------

    def _process_chunk(
        self,
        chunk: DocumentChunk,
        document_id: str,
        classification: ClassificationResult,
    ) -> str | None:
        """处理单个块: 标注元数据并写入存储.

        将分类结果写入切片元数据，然后通过 KnowledgeStore.add_chunk 存储。

        Args:
            chunk: 文档切片
            document_id: 所属文档 ID
            classification: 分类结果

        Returns:
            成功时返回切片 ID，失败返回 None
        """
        try:
            # 构建切片元数据
            chunk_metadata = ChunkMetadata(
                knowledge_domain=classification.domain,
                material_system=classification.material_system,
                knowledge_level=classification.level,
                content_type=classification.content_type,
                kp_anchors=classification.kp_anchors,
                authority_tier=classification.authority_tier,
                key_concepts=classification.key_concepts,
            )

            # 将分类元数据写入切片的 metadata 字段
            chunk.metadata["classification"] = chunk_metadata.model_dump(mode="json")
            chunk.metadata["classification_confidence"] = classification.confidence

            # 写入存储
            self._store.add_chunk(chunk)

            # 可选: 为关键概念创建知识实体 (借鉴 GraphRAG entity extraction)
            if classification.key_concepts:
                self._create_entities_from_concepts(
                    classification, document_id, chunk.chunk_id
                )

            return chunk.chunk_id

        except Exception as exc:
            logger.exception("处理块 %s 失败: %s", chunk.chunk_id, exc)
            return None

    def _validate_chunk(self, chunk: DocumentChunk) -> bool:
        """验证切片有效性.

        检查切片内容是否非空、长度是否在合理范围内。

        Args:
            chunk: 待验证的切片

        Returns:
            True 如果验证通过
        """
        if not chunk.content or not chunk.content.strip():
            return False
        if len(chunk.content) < 10:
            # 过短的切片 (少于 10 字符) 跳过
            return False
        return True

    def _deduplicate(self, content_hash: str) -> bool:
        """去重检查.

        检查内容哈希是否已存在于已摄入集合中。

        Args:
            content_hash: 内容哈希值

        Returns:
            True 如果内容已存在 (重复)
        """
        with self._lock:
            return content_hash in self._content_hashes

    @staticmethod
    def _compute_hash(content: str) -> str:
        """计算内容哈希 (SHA-256).

        Args:
            content: 文本内容

        Returns:
            SHA-256 哈希十六进制字符串
        """
        return hashlib.sha256(content.encode()).hexdigest()

    def _create_entities_from_concepts(
        self,
        classification: ClassificationResult,
        document_id: str,
        chunk_id: str,
    ) -> None:
        """从分类结果中提取的关键概念创建知识实体 (借鉴 GraphRAG entity extraction).

        为每个关键概念创建一个 CONCEPT 类型的 KnowledgeEntity，
        并建立与源切片的关联。

        Args:
            classification: 分类结果
            document_id: 源文档 ID
            chunk_id: 源切片 ID
        """
        for concept in classification.key_concepts:
            try:
                # 检查是否已存在同名实体
                # (KnowledgeStore 不提供按名称查询，此处简化处理)
                entity = KnowledgeEntity(
                    entity_type=EntityType.CONCEPT,
                    name=concept,
                    description=f"从文档 {document_id} 切片 {chunk_id} 自动提取的概念",
                    domain=classification.domain.value,
                    tags=[classification.material_system]
                    if classification.material_system
                    else [],
                    metadata={
                        "source_document_id": document_id,
                        "source_chunk_id": chunk_id,
                        "authority_tier": classification.authority_tier.value,
                        "extraction_method": "auto_classify",
                    },
                )
                self._store.add_entity(entity, check_duplicate=False)
            except Exception as exc:
                # 实体创建失败不影响摄入流程
                logger.debug("概念 '%s' 实体创建失败: %s", concept, exc)

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取摄入管道统计信息.

        Returns:
            统计信息字典，包括:
            - total_ingested: 累计成功摄入切片数
            - total_failed: 累计失败数
            - total_skipped: 累计跳过数
            - total_documents: 累计处理文档数
            - unique_hashes: 去重集合大小
            - store_chunks: 存储中切片总数
            - store_entities: 存储中实体总数
        """
        with self._lock:
            stats = dict(self._stats)
            stats["unique_hashes"] = len(self._content_hashes)

        stats["store_chunks"] = self._store.chunk_count()
        stats["store_entities"] = self._store.entity_count()
        return stats


__all__ = [
    # 枚举
    "KnowledgeDomain",
    "KnowledgeLevel",
    "ContentType",
    "AuthorityTier",
    # 数据模型
    "ChunkMetadata",
    "ChunkingConfig",
    "ClassificationResult",
    "IngestionResult",
    # 核心引擎
    "ChunkingEngine",
    "ClassificationEngine",
    "IngestionPipeline",
]
