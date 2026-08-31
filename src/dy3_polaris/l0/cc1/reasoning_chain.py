"""推理链提取与 DAG 分析模块.

为 L2 LogicLayer 提供推理链提取、依赖关系构建、循环检测、断链检测、
冗余检测与拓扑排序能力。

本模块从自然语言文本中识别因果推理结构，构建推理有向无环图 (DAG)，
并检测推理过程中的逻辑缺陷，包括：

- **循环推理 (circular logic)**: 推理步骤形成环，A→B→C→A
- **断链 (broken chains)**: 结论缺乏前提支撑
- **冗余路径 (redundant paths)**: 不同推理路径得到相同结论
- **矛盾 (contradictions)**: 不同声明间的数值或定性冲突

核心组件：
- :class:`ReasoningStep` — 推理步骤数据结构
- :class:`ReasoningChainExtractor` — 从文本中提取推理步骤
- :class:`ReasoningDAG` — 推理有向无环图分析
- :class:`Contradiction` — 矛盾数据结构
- :class:`ContradictionDetector` — 声明间矛盾检测

自包含模块，仅依赖 Python 标准库。
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any


# ============================================================
# 异常定义
# ============================================================


class ReasoningChainError(Exception):
    """推理链分析基础异常.

    所有推理链模块中的异常均继承此类，便于上层统一捕获。

    Attributes:
        message: 错误消息
        detail: 详细信息 (英文，用于日志)
    """

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ============================================================
# 枚举与常量
# ============================================================


class ContradictionType(str, Enum):
    """矛盾类型."""

    NUMERIC = "numeric"
    QUALITATIVE = "qualitative"


# 正向因果标记: 前提在标记之前，结论在标记之后
# 格式: "前提 [标记] 结论"
FORWARD_CAUSAL_MARKERS: list[str] = [
    # 中文
    "因此",
    "所以",
    "从而",
    "进而",
    "导致",
    # 英文
    "consequently",
    "therefore",
    "thus",
    "hence",
    "as a result",
    "leading to",
]

# 反向因果标记: 结论在标记之前（或标记在句首），前提在标记之后
# 格式: "结论 [标记] 前提" 或 "[标记] 前提, 结论"
BACKWARD_CAUSAL_MARKERS: list[str] = [
    # 中文
    "因为",
    "由于",
    # 英文
    "because",
]

# 所有因果标记（正向 + 反向）
ALL_CAUSAL_MARKERS: list[str] = FORWARD_CAUSAL_MARKERS + BACKWARD_CAUSAL_MARKERS

# 定性反义词对（用于矛盾检测）
OPPOSITE_PAIRS: list[tuple[str, str]] = [
    # 中文反义词
    ("增加", "减少"),
    ("增大", "减小"),
    ("升高", "降低"),
    ("上升", "下降"),
    ("增强", "减弱"),
    ("提高", "降低"),
    ("促进", "抑制"),
    ("加速", "减速"),
    ("变长", "变短"),
    ("延长", "缩短"),
    ("变大", "变小"),
    ("变强", "变弱"),
    ("正比", "反比"),
    # 英文反义词
    ("increase", "decrease"),
    ("rise", "fall"),
    ("up", "down"),
    ("strong", "weak"),
    ("high", "low"),
    ("fast", "slow"),
    ("long", "short"),
    ("large", "small"),
    ("more", "less"),
    ("positive", "negative"),
    ("enhance", "reduce"),
    ("improve", "degrade"),
]

# 中文停用词（关键词提取时过滤）
_STOPWORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "那", "与", "或",
    "则", "而", "其", "此", "为", "以", "可", "能", "将", "被",
    "使", "让", "令", "因", "由", "从", "对", "于", "向", "给",
    # 英文停用词
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "and", "or", "but", "not", "no", "yes", "this", "that",
    "it", "he", "she", "we", "they", "his", "her", "its",
    "to", "of", "in", "on", "at", "for", "with", "from",
    "by", "as", "can", "could", "should", "would", "may", "might",
    "has", "have", "had", "do", "does", "did", "will", "shall",
}

# 单位字符集（数值提取时匹配科学单位）
_UNIT_CHARS = r"a-zA-Z%°℃μΩ²³·/\u4e00-\u9fff\d"


# ============================================================
# 数据结构
# ============================================================


@dataclass
class ReasoningStep:
    """推理步骤.

    表示推理链中的一个逻辑步骤，包含前提、结论和因果关系标记。

    Attributes:
        step_id: 步骤唯一标识符
        text: 步骤的完整陈述文本
        premises: 前提步骤 ID 列表（本步骤依赖的步骤）
        conclusion: 结论文本（从前提推导出的结论）
        causal_marker: 因果关系标记（如 "因此"、"because" 等）
        position: 在原文中的位置序号（从 0 开始）
    """

    step_id: str
    text: str
    premises: list[str] = field(default_factory=list)
    conclusion: str = ""
    causal_marker: str = ""
    position: int = 0

    def __repr__(self) -> str:
        marker = f"[{self.causal_marker}]" if self.causal_marker else ""
        return (
            f"ReasoningStep(id={self.step_id}, pos={self.position}, "
            f"marker={marker}, premises={self.premises})"
        )


@dataclass
class Contradiction:
    """矛盾.

    表示两个声明之间的矛盾关系。

    Attributes:
        claim_a: 第一个声明（原始对象或字符串）
        claim_b: 第二个声明（原始对象或字符串）
        contradiction_type: 矛盾类型（"numeric" 或 "qualitative"）
        description: 矛盾描述
    """

    claim_a: Any
    claim_b: Any
    contradiction_type: str = ""
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"Contradiction(type={self.contradiction_type}, "
            f"desc={self.description!r})"
        )


# ============================================================
# 推理链提取器
# ============================================================


class ReasoningChainExtractor:
    """推理链提取器.

    从自然语言文本中提取推理步骤，识别因果标记，
    提取前提-结论对，并构建步骤间的依赖关系。

    支持中英文因果标记，包括：
    - 正向标记: 因此, 所以, 从而, 进而, 导致, therefore, consequently 等
    - 反向标记: 因为, 由于, because 等

    工作流程::

        extractor = ReasoningChainExtractor()
        steps = extractor.extract(text)
        # steps 中每个步骤已包含 premises 依赖关系

    Note:
        依赖关系构建基于关键词重叠启发式匹配，可能存在误判。
        对于精确的语义依赖分析，建议结合领域知识图谱使用。
    """

    def __init__(self) -> None:
        self._forward_markers = FORWARD_CAUSAL_MARKERS
        self._backward_markers = BACKWARD_CAUSAL_MARKERS
        self._all_markers = ALL_CAUSAL_MARKERS

    def extract(self, text: str) -> list[ReasoningStep]:
        """从文本中提取推理步骤.

        解析文本中的因果推理结构，识别因果标记，提取前提-结论对，
        并通过关键词匹配构建步骤间的依赖关系。

        Args:
            text: 待分析的自然语言文本

        Returns:
            推理步骤列表，按原文顺序排列。每个步骤的 ``premises``
            字段已填充所依赖的前序步骤 ID。

        Raises:
            ReasoningChainError: 当输入文本无效时
        """
        if text is None:
            raise ReasoningChainError(
                "输入文本不能为 None", detail="text is None"
            )
        if not isinstance(text, str):
            raise ReasoningChainError(
                f"输入文本类型无效: {type(text).__name__}",
                detail=f"Expected str, got {type(text).__name__}",
            )
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        steps: list[ReasoningStep] = []
        premise_texts: dict[str, str] = {}

        for position, sentence in enumerate(sentences):
            step, premise_text = self._extract_step(sentence, position)
            steps.append(step)
            premise_texts[step.step_id] = premise_text

        # 构建步骤间依赖关系
        self._build_dependencies(steps, premise_texts)

        return steps

    def _split_sentences(self, text: str) -> list[str]:
        """将文本分割为句子/陈述.

        按句末标点（。！？；.!?\n）分割，保留完整的因果结构，
        不按逗号分割以保持前提-结论对的完整性。

        Args:
            text: 原始文本

        Returns:
            非空句子列表
        """
        pattern = re.compile(r"[。！？；.!?;]\s*|\n+")
        raw_sentences = pattern.split(text)
        return [s.strip() for s in raw_sentences if s.strip()]

    def _find_marker(
        self, sentence: str
    ) -> tuple[str, str, int] | None:
        """在句子中查找最早的因果标记.

        遍历所有因果标记，返回位置最早的一个。

        Args:
            sentence: 待查找的句子

        Returns:
            ``(marker_text, marker_type, position)`` 元组，
            其中 ``marker_type`` 为 "forward" 或 "backward"。
            未找到时返回 ``None``。
        """
        sentence_lower = sentence.lower()
        best: tuple[str, str, int] | None = None
        best_pos = len(sentence) + 1

        # 检查正向标记
        for marker in self._forward_markers:
            pos = sentence_lower.find(marker.lower())
            if 0 <= pos < best_pos:
                best_pos = pos
                best = (marker, "forward", pos)

        # 检查反向标记
        for marker in self._backward_markers:
            pos = sentence_lower.find(marker.lower())
            if 0 <= pos < best_pos:
                best_pos = pos
                best = (marker, "backward", pos)

        return best

    def _extract_step(
        self, sentence: str, position: int
    ) -> tuple[ReasoningStep, str]:
        """从单个句子中提取推理步骤.

        根据因果标记将句子拆分为前提和结论，构建 :class:`ReasoningStep`。

        Args:
            sentence: 待提取的句子
            position: 在原文中的位置序号

        Returns:
            ``(ReasoningStep, premise_text)`` 元组，
            其中 ``premise_text`` 为提取到的前提文本（用于后续依赖构建）。
        """
        step_id = f"step-{position:04d}-{uuid.uuid4().hex[:6]}"
        marker_info = self._find_marker(sentence)

        if marker_info is None:
            # 无因果标记 — 独立陈述
            clean = sentence.strip()
            step = ReasoningStep(
                step_id=step_id,
                text=clean,
                premises=[],
                conclusion=clean,
                causal_marker="",
                position=position,
            )
            return step, ""

        marker_text, marker_type, marker_pos = marker_info
        text_before = sentence[:marker_pos].strip()
        text_after = sentence[marker_pos + len(marker_text):].strip()

        premise_text = ""
        conclusion_text = ""

        if marker_type == "forward":
            # 正向标记: "前提 [标记] 结论"
            premise_text = text_before
            conclusion_text = text_after
        else:
            # 反向标记: "结论 [标记] 前提" 或 "[标记] 前提, 结论"
            if text_before:
                # 标记在句中: 结论在前，前提在后
                conclusion_text = text_before
                premise_text = text_after
            else:
                # 标记在句首: 分割后续文本为前提和结论
                premise_text, conclusion_text = self._split_premise_conclusion(
                    text_after
                )

        # 清理文本
        premise_text = self._clean_text(premise_text)
        conclusion_text = self._clean_text(conclusion_text)

        full_text = sentence.strip()
        if not conclusion_text:
            conclusion_text = full_text
        if not premise_text and marker_type == "forward":
            premise_text = text_before.strip()

        step = ReasoningStep(
            step_id=step_id,
            text=full_text,
            premises=[],
            conclusion=conclusion_text,
            causal_marker=marker_text,
            position=position,
        )
        return step, premise_text

    def _split_premise_conclusion(self, text: str) -> tuple[str, str]:
        """将 "前提, 结论" 格式的文本分割为前提和结论.

        处理 "因为A，所以B" 或 "由于A，B" 等句首标记格式，
        按第一个分隔符拆分，并清理结论中的残余标记。

        Args:
            text: 标记之后的文本

        Returns:
            ``(premise, conclusion)`` 元组
        """
        pattern = re.compile(r"[，,；;]")
        match = pattern.search(text)
        if match:
            premise = text[: match.start()].strip()
            conclusion = text[match.end():].strip()
            conclusion = self._remove_leading_marker(conclusion)
            return premise, conclusion
        # 无分隔符 — 整体作为前提
        return text.strip(), ""

    def _remove_leading_marker(self, text: str) -> str:
        """移除文本开头的因果标记.

        处理 "所以B" → "B" 等情况。

        Args:
            text: 待清理的文本

        Returns:
            移除开头标记后的文本
        """
        text = text.strip()
        text_lower = text.lower()
        for marker in self._all_markers:
            if text_lower.startswith(marker.lower()):
                return text[len(marker):].strip()
        return text

    @staticmethod
    def _clean_text(text: str) -> str:
        """清理文本: 去除首尾标点和空白.

        Args:
            text: 待清理的文本

        Returns:
            清理后的文本
        """
        text = text.strip()
        text = re.sub(r"^[，,；;：:、\s]+", "", text)
        text = re.sub(r"[，,；;：:、\s]+$", "", text)
        return text

    def _build_dependencies(
        self,
        steps: list[ReasoningStep],
        premise_texts: dict[str, str],
    ) -> None:
        """构建步骤间的依赖关系.

        通过匹配前提文本与前序步骤的结论，建立前提→结论依赖。
        采用关键词重叠启发式: 当前提文本与前序步骤结论的关键词
        重叠度超过阈值时，建立依赖边。

        Args:
            steps: 推理步骤列表（按位置排序）
            premise_texts: 步骤 ID → 前提文本的映射
        """
        for step in steps:
            premise_text = premise_texts.get(step.step_id, "")
            if not premise_text or len(premise_text) < 2:
                continue

            for prev_step in steps:
                if prev_step.position >= step.position:
                    break
                target = prev_step.conclusion or prev_step.text
                if self._is_related(premise_text, target):
                    if prev_step.step_id not in step.premises:
                        step.premises.append(prev_step.step_id)

    def _is_related(self, premise_text: str, conclusion: str) -> bool:
        """判断前提文本与结论是否相关.

        采用两级匹配策略：
        1. **直接包含**: 一方是另一方的子串
        2. **关键词重叠**: 提取双方关键词，重叠度超过阈值

        Args:
            premise_text: 前提文本
            conclusion: 结论文本

        Returns:
            相关返回 ``True``，否则 ``False``
        """
        if not premise_text or not conclusion:
            return False

        # 直接包含检查
        if premise_text in conclusion or conclusion in premise_text:
            return True

        # 关键词重叠检查
        premise_keywords = self._extract_keywords(premise_text)
        conclusion_keywords = self._extract_keywords(conclusion)

        if not premise_keywords or not conclusion_keywords:
            return False

        overlap = premise_keywords & conclusion_keywords
        threshold = max(1, len(premise_keywords) // 2)
        return len(overlap) >= threshold

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """从文本中提取关键词.

        提取中文二元组（2 字符滑动窗口）、英文单词（2+ 字母），
        过滤停用词。同时提取数值-单位组合。

        中文无词边界，使用 2 字符二元组 (bigram) 近似词匹配，
        保证不同文本间存在可比较的最小语义单元。

        Args:
            text: 待提取的文本

        Returns:
            关键词集合（小写形式）
        """
        keywords: set[str] = set()

        # 中文二元组 (2 字符滑动窗口)
        for match in re.finditer(r"[\u4e00-\u9fff]+", text):
            seq = match.group()
            for i in range(len(seq) - 1):
                bigram = seq[i : i + 2]
                if bigram not in _STOPWORDS:
                    keywords.add(bigram)

        # 英文单词 (2+ 字母)
        for match in re.finditer(r"[a-zA-Z]{2,}", text):
            word = match.group().lower()
            if word not in _STOPWORDS:
                keywords.add(word)

        # 数值-单位组合
        for match in re.finditer(r"\d+\.?\d*\s*[a-zA-Z%°℃μΩ]*", text):
            keywords.add(match.group().lower().strip())

        return keywords


# ============================================================
# 推理有向无环图
# ============================================================


class ReasoningDAG:
    """推理有向无环图.

    从推理步骤构建 DAG，支持循环检测、断链检测、
    冗余检测和拓扑排序。

    图的边方向: **前提步骤 → 结论步骤**
    (A → B 表示 A 是 B 的前提，B 依赖 A)

    使用示例::

        extractor = ReasoningChainExtractor()
        steps = extractor.extract(text)
        dag = ReasoningDAG.build(steps)

        cycles = dag.detect_cycles()       # 检测循环逻辑
        breaks = dag.detect_breaks()       # 检测断链
        redundant = dag.detect_redundancy() # 检测冗余路径
        order = dag.topological_sort()     # 拓扑排序

    Note:
        循环检测使用递归 DFS，对于超过 500 个节点的图
        可能触发 Python 递归深度限制。
    """

    def __init__(self) -> None:
        # adjacency: step_id -> [dependent step_ids] (premise -> conclusion)
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        # reverse: step_id -> [premise step_ids] (conclusion -> premises)
        self._reverse: dict[str, list[str]] = defaultdict(list)
        # steps map: step_id -> ReasoningStep
        self._steps: dict[str, ReasoningStep] = {}

    # ---- 构建 ----

    @classmethod
    def build(cls, steps: list[ReasoningStep]) -> ReasoningDAG:
        """从推理步骤列表构建 DAG.

        根据 ``ReasoningStep.premises`` 字段建立边，
        仅保留指向步骤列表中存在的前提步骤的边。

        Args:
            steps: 推理步骤列表

        Returns:
            构建好的 :class:`ReasoningDAG` 实例
        """
        dag = cls()
        step_ids: set[str] = set()

        # 注册所有步骤
        for step in steps:
            dag._steps[step.step_id] = step
            step_ids.add(step.step_id)

        # 建立边: premise -> step
        for step in steps:
            for premise_id in step.premises:
                if premise_id in step_ids:
                    if step.step_id not in dag._adjacency[premise_id]:
                        dag._adjacency[premise_id].append(step.step_id)
                    if premise_id not in dag._reverse[step.step_id]:
                        dag._reverse[step.step_id].append(premise_id)

        return dag

    # ---- 分析方法 ----

    def detect_cycles(self) -> list[list[str]]:
        """检测推理中的循环 (循环逻辑).

        使用 DFS 三色染色法（白/灰/黑）检测环，
        返回所有找到的环路径。每个环表示为步骤 ID 列表，
        首尾元素相同以表示环的闭合（如 ``[A, B, C, A]``）。

        Returns:
            环路径列表，每个环是步骤 ID 列表。
            无环时返回空列表。
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {sid: WHITE for sid in self._steps}
        cycles: list[list[str]] = []

        def _dfs(node: str, path: list[str]) -> None:
            color[node] = GRAY
            path.append(node)

            for neighbor in self._adjacency.get(node, []):
                neighbor_color = color.get(neighbor, WHITE)
                if neighbor_color == GRAY:
                    # 发现回边 — 提取环
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                elif neighbor_color == WHITE:
                    _dfs(neighbor, path)

            path.pop()
            color[node] = BLACK

        for node_id in self._steps:
            if color[node_id] == WHITE:
                _dfs(node_id, [])

        return self._deduplicate_cycles(cycles)

    def detect_breaks(self) -> list[str]:
        """检测断链 (缺乏前提的结论).

        返回有因果标记（即提出了结论）但在 DAG 中无有效前提步骤
        的步骤 ID 列表。这些步骤提出了结论但无法追溯到任何前提，
        构成推理链中的 "断裂"。

        Returns:
            断链步骤 ID 列表
        """
        breaks: list[str] = []
        for step_id, step in self._steps.items():
            valid_premises = self._reverse.get(step_id, [])
            if step.causal_marker and not valid_premises:
                breaks.append(step_id)
        return breaks

    def detect_redundancy(self) -> list[tuple[str, str]]:
        """检测冗余推理路径.

        返回结论相同或高度相似（相似度 >= 0.85）的步骤对列表。
        两条不同路径推导出相同结论，可能存在冗余推理。

        Returns:
            冗余步骤对列表，每对为 ``(step_id_a, step_id_b)``
        """
        redundant: list[tuple[str, str]] = []
        step_list = list(self._steps.values())

        for i in range(len(step_list)):
            for j in range(i + 1, len(step_list)):
                if self._conclusions_redundant(step_list[i], step_list[j]):
                    redundant.append((step_list[i].step_id, step_list[j].step_id))

        return redundant

    def topological_sort(self) -> list[str]:
        """对推理步骤进行拓扑排序.

        使用 Kahn 算法，返回步骤 ID 的拓扑顺序。
        前提步骤排在结论步骤之前。若图中存在环，
        则环内节点不会被包含在结果中（部分排序）。

        Returns:
            拓扑排序后的步骤 ID 列表。
            存在环时返回不含环内节点的部分排序。
        """
        # 计算入度
        in_degree: dict[str, int] = {sid: 0 for sid in self._steps}
        for step_id in self._steps:
            for dependent in self._adjacency.get(step_id, []):
                in_degree[dependent] = in_degree.get(dependent, 0) + 1

        # 初始化队列（入度为 0 的节点）
        queue: deque[str] = deque()
        for sid in self._steps:
            if in_degree.get(sid, 0) == 0:
                queue.append(sid)

        # 处理队列
        result: list[str] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self._adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    # ---- 辅助方法 ----

    @staticmethod
    def _deduplicate_cycles(cycles: list[list[str]]) -> list[list[str]]:
        """对环列表去重.

        通过规范化（旋转到最小元素开头）后比较去重。
        """
        seen: set[tuple[str, ...]] = set()
        unique: list[list[str]] = []

        for cycle in cycles:
            normalized = ReasoningDAG._normalize_cycle(cycle)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(cycle)

        return unique

    @staticmethod
    def _normalize_cycle(cycle: list[str]) -> tuple[str, ...]:
        """规范化环用于去重比较.

        移除闭合元素并旋转到最小元素开头。
        """
        # 移除闭合元素（首尾相同的最后一个）
        nodes = cycle[:-1] if len(cycle) > 1 and cycle[0] == cycle[-1] else list(cycle)
        if not nodes:
            return ()
        min_idx = nodes.index(min(nodes))
        rotated = nodes[min_idx:] + nodes[:min_idx]
        return tuple(rotated)

    def _conclusions_redundant(
        self, step_a: ReasoningStep, step_b: ReasoningStep
    ) -> bool:
        """判断两个步骤的结论是否冗余.

        判断依据（满足任一即冗余）：
        1. 结论文本完全相同
        2. 一方包含另一方
        3. 序列相似度 >= 0.85

        Args:
            step_a: 步骤 A
            step_b: 步骤 B

        Returns:
            冗余返回 ``True``
        """
        conc_a = step_a.conclusion.strip()
        conc_b = step_b.conclusion.strip()
        if not conc_a or not conc_b:
            return False
        if conc_a == conc_b:
            return True
        if conc_a in conc_b or conc_b in conc_a:
            return True
        ratio = SequenceMatcher(None, conc_a, conc_b).ratio()
        return ratio >= 0.85

    # ---- 查询方法 ----

    def get_step(self, step_id: str) -> ReasoningStep | None:
        """获取指定 ID 的步骤.

        Args:
            step_id: 步骤 ID

        Returns:
            步骤对象，不存在时返回 ``None``
        """
        return self._steps.get(step_id)

    def get_premises(self, step_id: str) -> list[str]:
        """获取步骤的前提步骤 ID 列表.

        Args:
            step_id: 步骤 ID

        Returns:
            前提步骤 ID 列表
        """
        return list(self._reverse.get(step_id, []))

    def get_dependents(self, step_id: str) -> list[str]:
        """获取依赖该步骤的后续步骤 ID 列表.

        Args:
            step_id: 步骤 ID

        Returns:
            依赖步骤 ID 列表
        """
        return list(self._adjacency.get(step_id, []))

    @property
    def node_count(self) -> int:
        """图中节点数."""
        return len(self._steps)

    @property
    def edge_count(self) -> int:
        """图中边数."""
        return sum(len(deps) for deps in self._adjacency.values())

    def __len__(self) -> int:
        return len(self._steps)

    def __contains__(self, step_id: str) -> bool:
        return step_id in self._steps

    def __repr__(self) -> str:
        return (
            f"ReasoningDAG(nodes={self.node_count}, "
            f"edges={self.edge_count})"
        )


# ============================================================
# 矛盾检测器
# ============================================================


class ContradictionDetector:
    """矛盾检测器.

    检测声明之间的矛盾，包括：

    - **数值冲突 (numeric)**: 同一属性对应不同数值
      (如 "浓度为3mol%" vs "浓度为5mol%")
    - **定性冲突 (qualitative)**: 相反的描述
      (如 "强度增加" vs "强度减少")

    支持多种声明类型：字符串、具有 ``text`` 属性的对象
    (如 :class:`Claim`)、具有 ``conclusion`` 属性的对象
    (如 :class:`ReasoningStep`)。

    使用示例::

        detector = ContradictionDetector()
        contradictions = detector.detect_contradictions(claims)
        for c in contradictions:
            print(c.contradiction_type, c.description)
    """

    def detect_contradictions(self, claims: list) -> list[Contradiction]:
        """检测声明间的矛盾.

        对所有声明两两组合，检测数值冲突和定性冲突。

        Args:
            claims: 声明列表 (字符串或具有 text/conclusion 属性的对象)

        Returns:
            矛盾列表
        """
        if not claims or len(claims) < 2:
            return []

        contradictions: list[Contradiction] = []

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                claim_a = claims[i]
                claim_b = claims[j]
                text_a = self._get_text(claim_a)
                text_b = self._get_text(claim_b)

                if not text_a or not text_b:
                    continue

                # 数值冲突检测
                numeric = self._check_numeric_conflict(
                    claim_a, claim_b, text_a, text_b
                )
                if numeric is not None:
                    contradictions.append(numeric)

                # 定性冲突检测
                qualitative = self._check_qualitative_conflict(
                    claim_a, claim_b, text_a, text_b
                )
                if qualitative is not None:
                    contradictions.append(qualitative)

        return contradictions

    # ---- 文本提取 ----

    @staticmethod
    def _get_text(claim: Any) -> str:
        """从声明对象中提取文本.

        按以下优先级提取：
        1. 字符串直接返回
        2. ``text`` 属性
        3. ``conclusion`` 属性
        4. ``str(claim)`` 兜底

        Args:
            claim: 声明对象

        Returns:
            声明文本
        """
        if isinstance(claim, str):
            return claim
        if hasattr(claim, "text") and claim.text:
            return str(claim.text)
        if hasattr(claim, "conclusion") and claim.conclusion:
            return str(claim.conclusion)
        return str(claim) if claim else ""

    # ---- 数值冲突检测 ----

    def _check_numeric_conflict(
        self,
        claim_a: Any,
        claim_b: Any,
        text_a: str,
        text_b: str,
    ) -> Contradiction | None:
        """检测数值冲突.

        提取两个声明中的属性-数值对，检查是否存在
        同一属性对应不同数值的情况。

        Args:
            claim_a: 声明 A
            claim_b: 声明 B
            text_a: 声明 A 文本
            text_b: 声明 B 文本

        Returns:
            检测到冲突返回 :class:`Contradiction`，否则 ``None``
        """
        nums_a = self._extract_property_numbers(text_a)
        nums_b = self._extract_property_numbers(text_b)

        if not nums_a or not nums_b:
            return None

        for prop_a, val_a, unit_a in nums_a:
            for prop_b, val_b, unit_b in nums_b:
                # 同一属性（不区分大小写）
                if prop_a.lower() == prop_b.lower():
                    if val_a != val_b:
                        return Contradiction(
                            claim_a=claim_a,
                            claim_b=claim_b,
                            contradiction_type=ContradictionType.NUMERIC.value,
                            description=(
                                f"数值冲突: 属性 '{prop_a}' "
                                f"在声明 A 中为 {val_a}{unit_a}, "
                                f"在声明 B 中为 {val_b}{unit_b}"
                            ),
                        )

        return None

    @staticmethod
    def _extract_property_numbers(
        text: str,
    ) -> list[tuple[str, float, str]]:
        """从文本中提取属性-数值对.

        识别以下模式：
        - 中文: "发射峰为575nm"、"浓度是3mol%"
        - 通用: "温度=300K"、"温度: 300K"
        - 英文: "temperature is 300K"

        Args:
            text: 待提取的文本

        Returns:
            ``(property, value, unit)`` 元组列表
        """
        results: list[tuple[str, float, str]] = []

        unit_pattern = f"[{_UNIT_CHARS}]*"
        patterns = [
            # 中文: 属性 [为是] 数值 [单位]
            re.compile(
                rf"([\u4e00-\u9fff]{{2,}})\s*[为是]\s*"
                rf"(\d+\.?\d*)\s*({unit_pattern})"
            ),
            # 通用: 属性 [=：:] 数值 [单位]
            re.compile(
                rf"([\u4e00-\u9fffa-zA-Z_]{{2,}})\s*[=：:]\s*"
                rf"(\d+\.?\d*)\s*({unit_pattern})"
            ),
            # 英文: property is/are/was/were value [unit]
            re.compile(
                rf"([a-zA-Z_]{{2,}})\s+(?:is|are|was|were)\s+"
                rf"(\d+\.?\d*)\s*({unit_pattern})",
                re.IGNORECASE,
            ),
        ]

        for pattern in patterns:
            for match in pattern.finditer(text):
                prop = match.group(1).strip()
                try:
                    val = float(match.group(2))
                except (ValueError, IndexError):
                    continue
                unit = ""
                if match.lastindex and match.lastindex >= 3:
                    unit = match.group(3).strip()
                results.append((prop, val, unit))

        return results

    # ---- 定性冲突检测 ----

    def _check_qualitative_conflict(
        self,
        claim_a: Any,
        claim_b: Any,
        text_a: str,
        text_b: str,
    ) -> Contradiction | None:
        """检测定性冲突.

        检查两个声明是否包含相反的定性描述（如 "增加" vs "减少"），
        且描述的是同一主体。

        Args:
            claim_a: 声明 A
            claim_b: 声明 B
            text_a: 声明 A 文本
            text_b: 声明 B 文本

        Returns:
            检测到冲突返回 :class:`Contradiction`，否则 ``None``
        """
        for word_a, word_b in OPPOSITE_PAIRS:
            a_has_first = self._contains_word(text_a, word_a)
            b_has_second = self._contains_word(text_b, word_b)
            a_has_second = self._contains_word(text_a, word_b)
            b_has_first = self._contains_word(text_b, word_a)

            if (a_has_first and b_has_second) or (a_has_second and b_has_first):
                if self._same_subject(text_a, text_b, word_a, word_b):
                    return Contradiction(
                        claim_a=claim_a,
                        claim_b=claim_b,
                        contradiction_type=ContradictionType.QUALITATIVE.value,
                        description=(
                            f"定性冲突: 声明 A 包含 '{word_a}', "
                            f"声明 B 包含 '{word_b}' (相反描述)"
                        ),
                    )

        return None

    @staticmethod
    def _contains_word(text: str, word: str) -> bool:
        """检查文本是否包含指定词语.

        英文使用词边界匹配，中文使用直接包含检查。

        Args:
            text: 待检查的文本
            word: 待查找的词语

        Returns:
            包含返回 ``True``
        """
        if not word:
            return False
        # 英文词使用词边界匹配
        if re.match(r"[a-zA-Z]", word):
            return bool(
                re.search(
                    r"\b" + re.escape(word) + r"\b",
                    text,
                    re.IGNORECASE,
                )
            )
        # 中文使用直接包含
        return word in text

    def _same_subject(
        self,
        text_a: str,
        text_b: str,
        word_a: str,
        word_b: str,
    ) -> bool:
        """判断两个文本是否描述同一主体.

        提取双方关键词（排除反义词本身和停用词），
        若存在至少一个公共关键词则视为同一主体。

        Args:
            text_a: 文本 A
            text_b: 文本 B
            word_a: 反义词 A
            word_b: 反义词 B

        Returns:
            同一主体返回 ``True``
        """
        exclude = {word_a.lower(), word_b.lower()}

        def _extract_tokens(text: str) -> set[str]:
            """提取中文二元组和英文单词作为比较单元."""
            tokens: set[str] = set()
            # 中文二元组 (2 字符滑动窗口)
            for match in re.finditer(r"[\u4e00-\u9fff]+", text):
                seq = match.group()
                for i in range(len(seq) - 1):
                    pair = seq[i : i + 2]
                    if pair.lower() not in _STOPWORDS and pair.lower() not in exclude:
                        tokens.add(pair)
            # 英文单词 (2+ 字母)
            for match in re.finditer(r"[a-zA-Z]{2,}", text):
                word = match.group().lower()
                if word not in _STOPWORDS and word not in exclude:
                    tokens.add(word)
            return tokens

        words_a = _extract_tokens(text_a)
        words_b = _extract_tokens(text_b)

        if not words_a or not words_b:
            # 无法确定主体，保守假设为同一主体
            return True

        overlap = words_a & words_b
        return len(overlap) > 0


# ============================================================
# 公开 API
# ============================================================


__all__ = [
    # 异常
    "ReasoningChainError",
    # 枚举
    "ContradictionType",
    # 常量
    "FORWARD_CAUSAL_MARKERS",
    "BACKWARD_CAUSAL_MARKERS",
    "ALL_CAUSAL_MARKERS",
    "OPPOSITE_PAIRS",
    # 数据结构
    "ReasoningStep",
    "Contradiction",
    # 核心类
    "ReasoningChainExtractor",
    "ReasoningDAG",
    "ContradictionDetector",
]
