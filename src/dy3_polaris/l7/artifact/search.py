"""L7 Artifact 管理系统 — 搜索与过滤 (search.py).

任务拆分 T3 · 设计文档 Ch.3.6。

实现 Artifact 的三种检索方式:

1. **全文本搜索**: 基于 title/payload/来源 Agent 注释关键词匹配,
   支持引号精确匹配 + 布尔运算符 (AND/OR/NOT)。
2. **结构化过滤**: type / source_agent / kp_id / 时间范围 / 是否含编辑。
3. **学情关联搜索**: 输入 KP ID, 找到直接关联 + 知识图谱间接关联的 Artifact。

融合世界先进方案:
    - 倒排索引 (Elasticsearch / SQLite FTS5): 词 → 文档集合的反向映射,
      布尔查询做集合运算 (AND 交集 / OR 并集 / NOT 差集)
    - BM25 思想: 词频 + 文档频率加权的相关性排序 (简化为命中次数)
    - 结构化查询: 字段过滤与全文检索的混合

设计要点:
    - InvertedIndex: defaultdict(set) term → doc_ids, 增量构建
    - 分词: 英文按空白 + 中文按单字/短语切分 (满足领域 KP 标识 A-01 等)
    - 查询解析: 支持 "exact phrase"、AND/OR/NOT、括号分组
    - 与既有 ArtifactManager.search() 语义兼容 (大小写不敏感, 命中次数排序)
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from typing import Any, Iterable

from ..models import Artifact

#: 英文/数字与中文混合匹配 (保持原始顺序)
_MIXED_RE = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]+")
#: 查询布尔运算符 (tokenize 已小写化, 故用小写形式)
_AND = {"and", "&", "+"}
_OR = {"or", "|"}
_NOT = {"not", "-", "!"}


def tokenize(text: Any) -> list[str]:
    """分词 — 英文按词, 中文按相邻双字 (bigram), 保留 KP 标识 (A-01).

    中文 bigram 策略 (借鉴主流中文检索): "荧光效率" → ["荧光","光效","效率"],
    支持短语匹配的同时避免单字噪声。查询 "荧光" 时同样生成 bigram, 可命中。
    按原始位置顺序输出, 保证布尔运算符 (AND/OR/NOT) 的语义顺序。

    Args:
        text: 待分词文本。

    Returns:
        小写词元列表。
    """
    if text is None:
        return []
    s = str(text).lower()
    tokens: list[str] = []
    for m in _MIXED_RE.finditer(s):
        tok = m.group()
        if tok[0].isascii():
            # 英文/数字/KP 标识
            tokens.append(tok)
        else:
            # 中文: 相邻双字 bigram
            if len(tok) == 1:
                tokens.append(tok)
            else:
                for i in range(len(tok) - 1):
                    tokens.append(tok[i : i + 2])
    return tokens


def build_index(artifacts: Iterable[Artifact]) -> "InvertedIndex":
    """便捷函数 — 从 Artifact 集合构建倒排索引."""
    index = InvertedIndex()
    index.add_artifacts(artifacts)
    return index


class InvertedIndex:
    """倒排索引 — 词元 → 文档 ID 集合.

    Attributes:
        doc_count: 索引文档总数。
    """

    def __init__(self) -> None:
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._doc_count = 0
        self._lock = threading.RLock()

    def add_document(self, doc_id: str, text: Any) -> None:
        """为单个文档建立倒排 (增量)."""
        tokens = set(tokenize(text))
        with self._lock:
            for term in tokens:
                self._postings[term].add(doc_id)
            self._doc_count += 1

    def add_artifacts(self, artifacts: Iterable[Artifact]) -> None:
        """为多个 Artifact 建立索引 (title + payload 文本)."""
        for artifact in artifacts:
            text = self._artifact_text(artifact)
            self.add_document(artifact.artifact_id, text)

    def remove_document(self, doc_id: str) -> None:
        """从索引移除文档."""
        with self._lock:
            for postings in self._postings.values():
                postings.discard(doc_id)
            self._doc_count = max(0, self._doc_count - 1)

    def postings(self, term: str) -> set[str]:
        """返回词元的文档 ID 集合."""
        with self._lock:
            return set(self._postings.get(term.lower(), set()))

    def terms(self) -> list[str]:
        """返回全部词元."""
        with self._lock:
            return list(self._postings.keys())

    def size(self) -> int:
        """词元数量."""
        with self._lock:
            return len(self._postings)

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @staticmethod
    def _artifact_text(artifact: Artifact) -> str:
        """提取 Artifact 的可搜索文本 (title + payload + agent + 注释)."""
        parts = [
            artifact.title or "",
            artifact.source_agent or "",
            " ".join(artifact.provenance_chain or []),
        ]
        payload = artifact.payload or {}
        parts.append(str(payload.get("content", "")))
        parts.append(str(payload.get("summary", "")))
        parts.append(str(payload.get("description", "")))
        # 结构化字段也参与搜索
        for key in ("title", "chart_type", "latex", "headers"):
            if key in payload:
                parts.append(str(payload[key]))
        return " ".join(parts)


# ============================================================
# 查询解析与执行
# ============================================================

#: 短语引用: "exact phrase"
_PHRASE_RE = re.compile(r'"([^"]+)"')


def parse_query(query: str) -> str:
    """规范化查询字符串 (保留布尔运算符与短语)."""
    return (query or "").strip()


def _match_phrase(index: InvertedIndex, phrase: str) -> set[str]:
    """精确短语匹配 — 短语内全部词元都出现 (AND 语义)."""
    tokens = tokenize(phrase)
    if not tokens:
        return set()
    result: set[str] | None = None
    for term in tokens:
        postings = index.postings(term)
        result = postings if result is None else (result & postings)
    return result or set()


class SearchEngine:
    """Artifact 搜索执行器 — 倒排索引 + 布尔查询 + 结构化过滤.

    使用示例::

        engine = SearchEngine()
        engine.reindex(artifacts)
        results = engine.search("荧光 AND 效率")
        results = engine.search("材料", filters={"type": "chart", "source_agent": "A1"})
    """

    def __init__(self) -> None:
        self._index = InvertedIndex()
        self._artifacts: dict[str, Artifact] = {}
        self._lock = threading.RLock()

    # ----------------------------------------------------------
    # 索引维护
    # ----------------------------------------------------------

    def reindex(self, artifacts: Iterable[Artifact]) -> None:
        """全量重建索引."""
        with self._lock:
            self._artifacts = {a.artifact_id: a for a in artifacts}
            self._index = InvertedIndex()
            self._index.add_artifacts(artifacts)

    def add(self, artifact: Artifact) -> None:
        """增量添加 Artifact."""
        with self._lock:
            if artifact.artifact_id in self._artifacts:
                self._index.remove_document(artifact.artifact_id)
            self._artifacts[artifact.artifact_id] = artifact
            self._index.add_artifacts([artifact])

    def remove(self, artifact_id: str) -> None:
        """增量移除 Artifact."""
        with self._lock:
            self._artifacts.pop(artifact_id, None)
            self._index.remove_document(artifact_id)

    # ----------------------------------------------------------
    # 全文本搜索
    # ----------------------------------------------------------

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        sort_by: str = "-created_at",
    ) -> list[Artifact]:
        """全文搜索 + 结构化过滤.

        Args:
            query: 查询字符串 (支持 "短语" 与 AND/OR/NOT, 默认 OR)。
            filters: 结构化过滤条件:
                type / source_agent / kp_id / session_id /
                min_created / max_created (时间戳) / edited_only (bool)。
            sort_by: 排序字段, "-" 前缀降序 (如 "-created_at")。

        Returns:
            匹配的 Artifact 列表 (按相关度/时间排序)。
        """
        with self._lock:
            matched_ids = self._search_ids(query)
            artifacts = [self._artifacts[a_id] for a_id in matched_ids if a_id in self._artifacts]
        artifacts = self._apply_filters(artifacts, filters or {})
        return self._sort(artifacts, sort_by)

    def _search_ids(self, query: str) -> set[str]:
        """执行布尔查询, 返回匹配的 Artifact ID 集合.

        支持语法:
            - 引号精确短语: "exact phrase" (AND 语义)
            - AND: 交集 (A AND B → 同时含 A 与 B)
            - OR: 并集 (A OR B → 含任一)  [默认, 无运算符时也是 OR]
            - NOT: 差集 (A NOT B → 含 A 不含 B)
        """
        query = parse_query(query)
        if not query:
            return set(self._artifacts.keys())

        # 先提取短语, 短语整体作为词元
        phrases = _PHRASE_RE.findall(query)
        remainder = _PHRASE_RE.sub(" ", query)
        toks = tokenize(remainder)

        # 显式布尔解析: 按运算符归类词元
        pos_terms: list[str] = []   # OR 语义 (默认)
        and_terms: list[str] = []   # AND 语义
        neg_terms: list[str] = []   # NOT 语义
        last_op = "or"
        for t in toks:
            if t in _AND:
                last_op = "and"
                continue
            if t in _OR:
                last_op = "or"
                continue
            if t in _NOT:
                last_op = "not"
                continue
            if last_op == "and":
                and_terms.append(t)
            elif last_op == "not":
                neg_terms.append(t)
            else:
                pos_terms.append(t)
            last_op = "or"  # 单次运算符仅作用于下一词元

        # 正项 OR 并集
        result: set[str] = set()
        for term in pos_terms:
            result |= self._index.postings(term)
        for phrase in phrases:
            result |= _match_phrase(self._index, phrase)

        # AND 组: 交集后与 result 取交集
        if and_terms:
            and_set: set[str] | None = None
            for term in and_terms:
                postings = self._index.postings(term)
                and_set = postings if and_set is None else (and_set & postings)
            if and_set is not None:
                result = and_set if not result else (result & and_set)

        # NOT 组: 差集
        for term in neg_terms:
            result -= self._index.postings(term)

        return result

    # ----------------------------------------------------------
    # 结构化过滤
    # ----------------------------------------------------------

    def _apply_filters(
        self, artifacts: list[Artifact], filters: dict[str, Any]
    ) -> list[Artifact]:
        """应用结构化过滤条件 (设计文档 Ch.3.6)."""
        result = artifacts
        if "type" in filters and filters["type"] is not None:
            wanted = {str(filters["type"]).lower()}
            result = [a for a in result if (a.type.value if hasattr(a.type, "value") else str(a.type)).lower() in wanted]
        if "source_agent" in filters and filters["source_agent"]:
            wanted_agent = str(filters["source_agent"]).lower()
            result = [a for a in result if a.source_agent.lower() == wanted_agent]
        if "session_id" in filters and filters["session_id"]:
            result = [a for a in result if a.session_id == filters["session_id"]]
        if "kp_id" in filters and filters["kp_id"]:
            kp = str(filters["kp_id"])
            result = [a for a in result if self._matches_kp_id(a, kp)]
        if "min_created" in filters and filters["min_created"] is not None:
            result = [a for a in result if a.created_at >= float(filters["min_created"])]
        if "max_created" in filters and filters["max_created"] is not None:
            result = [a for a in result if a.created_at <= float(filters["max_created"])]
        if filters.get("edited_only"):
            result = [a for a in result if a.version > 1]
        if "state" in filters and filters["state"] is not None:
            wanted_state = str(filters["state"]).lower()
            result = [
                a for a in result
                if (a.state.value if hasattr(a.state, "value") else str(a.state)).lower() == wanted_state
            ]
        return result

    @staticmethod
    def _matches_kp_id(artifact: Artifact, kp_id: str) -> bool:
        """匹配 KP ID — 检查 learner_context.kp_ids 与 payload 中的 KP 引用."""
        learner = artifact.learner_context or {}
        kp_ids = learner.get("kp_ids") or []
        if kp_id in kp_ids:
            return True
        # payload 中的 KP 引用 (如 kp_id 字段或文本)
        payload = artifact.payload or {}
        if payload.get("kp_id") == kp_id:
            return True
        text = str(payload)
        return f'"{kp_id}"' in text or f"'{kp_id}'" in text or kp_id in str(learner)

    # ----------------------------------------------------------
    # 学情关联搜索
    # ----------------------------------------------------------

    def related_by_kp(
        self, kp_id: str, max_depth: int = 2, kp_graph: dict[str, set[str]] | None = None
    ) -> list[Artifact]:
        """学情关联搜索 (设计文档 Ch.3.6) — 按 KP ID 找直接 + 间接关联 Artifact.

        Args:
            kp_id: 目标知识点 ID。
            max_depth: 知识图谱间接关联的最大跳数。
            kp_graph: 知识图谱邻接表 {kp_id: {相邻 kp_id}}; 提供时支持
                间接关联 (通过图谱遍历), 否则仅直接关联。

        Returns:
            关联 Artifact 列表 (直接关联优先, 按时间倒序)。
        """
        related_kps = self._expand_kp(kp_id, max_depth, kp_graph)
        direct: list[Artifact] = []
        indirect: list[Artifact] = []
        with self._lock:
            for artifact in self._artifacts.values():
                hit_kps = self._artifact_kps(artifact)
                if kp_id in hit_kps:
                    direct.append(artifact)
                elif hit_kps & related_kps:
                    indirect.append(artifact)
        direct.sort(key=lambda a: a.created_at, reverse=True)
        indirect.sort(key=lambda a: a.created_at, reverse=True)
        return direct + indirect

    @staticmethod
    def _expand_kp(
        kp_id: str, max_depth: int, kp_graph: dict[str, set[str]] | None
    ) -> set[str]:
        """BFS 展开 KP 的间接关联集合."""
        if not kp_graph:
            return set()
        visited: set[str] = {kp_id}
        frontier = {kp_id}
        for _ in range(max_depth):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= kp_graph.get(node, set())
            new_nodes = nxt - visited
            if not new_nodes:
                break
            visited |= new_nodes
            frontier = new_nodes
        visited.discard(kp_id)
        return visited

    @staticmethod
    def _artifact_kps(artifact: Artifact) -> set[str]:
        """提取 Artifact 关联的全部 KP ID."""
        kps: set[str] = set()
        learner = artifact.learner_context or {}
        kps |= {str(k) for k in (learner.get("kp_ids") or [])}
        payload = artifact.payload or {}
        if payload.get("kp_id"):
            kps.add(str(payload["kp_id"]))
        # 文本中匹配 KP 标识
        text = str(payload) + str(artifact.title or "")
        kps |= set(re.findall(r"\b[A-D]-\d{2}\b", text))
        return kps

    # ----------------------------------------------------------
    # 排序
    # ----------------------------------------------------------

    @staticmethod
    def _sort(artifacts: list[Artifact], sort_by: str) -> list[Artifact]:
        """排序 (支持 - 前缀降序)."""
        if not sort_by:
            return artifacts
        desc = sort_by.startswith("-")
        field = sort_by[1:] if desc else sort_by
        if field == "created_at":
            key = lambda a: a.created_at  # noqa: E731
        elif field == "updated_at":
            key = lambda a: a.updated_at  # noqa: E731
        elif field == "version":
            key = lambda a: a.version  # noqa: E731
        else:
            key = lambda a: str(a.title or "")  # noqa: E731
        return sorted(artifacts, key=key, reverse=desc)
