"""L3 领域知识层 — 索引引擎.

融合世界先进方案的索引设计:
- Neo4j: B-tree 标签索引 + 原生指针遍历
- Milvus: IVF/HNSW 向量索引 + 标量过滤
- Weaviate: 倒排索引 (BM25) + HNSW 向量索引 + 预过滤
- LlamaIndex: 多类型 Store 抽象 + 索引委托
- GraphRAG: 社区检测索引 + 层次化检索

提供四类索引:
1. HashIndex       — O(1) 精确查找 (借鉴 Neo4j B-tree 精确匹配)
2. InvertedIndex   — 全文倒排索引 + BM25 评分 (借鉴 Weaviate BM25)
3. VectorIndex     — 向量相似性索引 (暴力搜索 + HNSW 接口预留)
4. TypeIndex       — 实体类型多值索引 (借鉴 Neo4j 标签索引)

所有索引均为内存实现，接口设计支持未来替换为持久化后端。
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 哈希索引 — O(1) 精确查找
# ============================================================


class HashIndex:
    """哈希索引 (借鉴 Neo4j B-tree 精确匹配).

    提供基于字典的 O(1) 精确查找，支持单值和多值映射。

    内部使用 set 存储值集合: add/remove/contains 均为 O(1) 摊还,
    避免高频键 (如反复出现的实体名) 下 list 线性扫描导致的 O(n²) 退化。

    Attributes:
        _index: 键到值集合的映射
        _lock: 线程安全锁
    """

    def __init__(self) -> None:
        self._index: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def add(self, key: str, value: str) -> None:
        """添加键值对 (O(1) 摊还)."""
        with self._lock:
            self._index[key].add(value)

    def remove(self, key: str, value: str) -> bool:
        """移除键值对 (O(1) 摊还)."""
        with self._lock:
            if key in self._index:
                self._index[key].discard(value)
                if not self._index[key]:
                    del self._index[key]
                return True
            return False

    def get(self, key: str) -> list[str]:
        """获取键对应的所有值."""
        with self._lock:
            return list(self._index.get(key, ()))

    def contains(self, key: str) -> bool:
        """是否包含键."""
        with self._lock:
            return key in self._index

    def clear(self) -> None:
        """清空索引."""
        with self._lock:
            self._index.clear()

    def size(self) -> int:
        """索引大小."""
        with self._lock:
            return len(self._index)

    def keys(self) -> list[str]:
        """所有键."""
        with self._lock:
            return list(self._index.keys())


# ============================================================
# 类型索引 — 多值映射
# ============================================================


class TypeIndex:
    """实体类型索引 (借鉴 Neo4j 标签索引).

    按实体类型分组存储 ID，支持按类型快速检索。

    Attributes:
        _index: 类型到 ID 集合的映射
        _lock: 线程安全锁
    """

    def __init__(self) -> None:
        self._index: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def add(self, type_name: str, entity_id: str) -> None:
        """添加类型索引."""
        with self._lock:
            self._index[type_name].add(entity_id)

    def remove(self, type_name: str, entity_id: str) -> bool:
        """移除类型索引."""
        with self._lock:
            if type_name in self._index:
                self._index[type_name].discard(entity_id)
                if not self._index[type_name]:
                    del self._index[type_name]
                return True
            return False

    def get_by_type(self, type_name: str) -> list[str]:
        """按类型获取所有实体 ID."""
        with self._lock:
            return list(self._index.get(type_name, set()))

    def get_types(self) -> list[str]:
        """所有已索引的类型."""
        with self._lock:
            return list(self._index.keys())

    def type_count(self, type_name: str) -> int:
        """指定类型的实体数量."""
        with self._lock:
            return len(self._index.get(type_name, set()))

    def clear(self) -> None:
        """清空索引."""
        with self._lock:
            self._index.clear()

    def total(self) -> int:
        """所有类型的实体总数."""
        with self._lock:
            return sum(len(v) for v in self._index.values())


# ============================================================
# 倒排索引 — BM25 全文检索
# ============================================================


# 中文停用词 (最小集)
_STOP_WORDS: frozenset[str] = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "那", "它", "他",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "can",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "as", "and", "or", "not", "but", "if", "then", "else",
    "this", "that", "these", "those", "it", "its",
})


def _tokenize(text: str) -> list[str]:
    """分词 (简单中英文混合分词).

    英文按空格和标点分词，中文按字分词。
    过滤停用词和空白。
    """
    if not text:
        return []

    # 统一小写
    text = text.lower()

    # 清洗 HTML 标签: <sup>3+</sup> -> 3+, <sub>2</sub> -> 2 (化学式上下标还原)
    text = re.sub(r"<sup>([^<]*)</sup>", r"\1", text)
    text = re.sub(r"<sub>([^<]*)</sub>", r"\1", text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)

    # 提取英文 token: 允许字母+数字+电荷组合 (dy3+, na4, f9/2 等化学式)
    en_tokens = re.findall(r"[a-z][a-z0-9+\-/]*", text)

    # 提取中文字符序列，按字分词
    cn_chars = re.findall(r"[\u4e00-\u9fff]", text)

    tokens = en_tokens + cn_chars

    # 过滤停用词; 中文单字保留 (按字分词), 英文 token 已由正则保证 >=2 字符
    return [t for t in tokens if t not in _STOP_WORDS]


class InvertedIndex:
    """倒排索引 + BM25 评分 (借鉴 Weaviate BM25 + Elasticsearch 倒排索引).

    提供全文检索能力，支持 BM25 相关性评分。

    BM25 公式:
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) /
                      (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

    其中:
        IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        f(qi, D) = 词 qi 在文档 D 中的频率
        |D| = 文档 D 的长度
        avgdl = 平均文档长度
        k1 = 1.5 (词频饱和参数)
        b = 0.75 (长度归一化参数)

    Attributes:
        _postings: 词项到文档列表的映射 {term: {doc_id: term_freq}}
        _doc_lengths: 文档长度 {doc_id: length}
        _total_length: 所有文档总长度
        _doc_count: 文档总数
        _lock: 线程安全锁
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_lengths: dict[str, int] = {}
        self._total_length: int = 0
        self._doc_count: int = 0
        self._k1 = k1
        self._b = b
        self._lock = threading.RLock()

    def add_document(self, doc_id: str, text: str) -> None:
        """添加文档到倒排索引.

        Args:
            doc_id: 文档 ID
            text: 文档文本内容
        """
        tokens = _tokenize(text)
        if not tokens:
            return

        with self._lock:
            # 如果文档已存在，先移除旧数据
            if doc_id in self._doc_lengths:
                self._remove_document_locked(doc_id)

            # 统计词频
            term_freqs: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            # 更新倒排索引
            for term, freq in term_freqs.items():
                self._postings[term][doc_id] = freq

            # 更新文档长度统计
            doc_len = len(tokens)
            self._doc_lengths[doc_id] = doc_len
            self._total_length += doc_len
            self._doc_count += 1

    def remove_document(self, doc_id: str) -> bool:
        """从倒排索引中移除文档."""
        with self._lock:
            return self._remove_document_locked(doc_id)

    def _remove_document_locked(self, doc_id: str) -> bool:
        """移除文档 (已持有锁)."""
        if doc_id not in self._doc_lengths:
            return False

        doc_len = self._doc_lengths.pop(doc_id)
        self._total_length -= doc_len
        self._doc_count -= 1

        # 从所有倒排链中移除
        empty_terms: list[str] = []
        for term, postings in self._postings.items():
            if doc_id in postings:
                del postings[doc_id]
                if not postings:
                    empty_terms.append(term)

        for term in empty_terms:
            del self._postings[term]

        return True

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """BM25 全文检索.

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            [(doc_id, score)] 按分数降序排列
        """
        query_tokens = _tokenize(query)
        if not query_tokens or self._doc_count == 0:
            return []

        with self._lock:
            avgdl = self._total_length / self._doc_count if self._doc_count > 0 else 0.0
            scores: dict[str, float] = defaultdict(float)

            for term in query_tokens:
                postings = self._postings.get(term)
                if not postings:
                    continue

                # 计算 IDF
                n_qi = len(postings)
                idf = math.log(
                    (self._doc_count - n_qi + 0.5) / (n_qi + 0.5) + 1.0
                )

                # 对每个包含该词的文档计算 BM25 分数
                for doc_id, freq in postings.items():
                    doc_len = self._doc_lengths.get(doc_id, 0)
                    if avgdl > 0:
                        norm = 1 - self._b + self._b * (doc_len / avgdl)
                    else:
                        norm = 1.0

                    score = idf * (freq * (self._k1 + 1)) / (freq + self._k1 * norm)
                    scores[doc_id] += score

            # 排序并取 top_k
            result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return result[:top_k]

    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """获取词项在文档中的频率."""
        with self._lock:
            return self._postings.get(term, {}).get(doc_id, 0)

    def get_document_frequency(self, term: str) -> int:
        """获取词项的文档频率 (包含该词的文档数)."""
        with self._lock:
            return len(self._postings.get(term, {}))

    def vocabulary_size(self) -> int:
        """词表大小."""
        with self._lock:
            return len(self._postings)

    def doc_count(self) -> int:
        """已索引文档数."""
        with self._lock:
            return self._doc_count

    def clear(self) -> None:
        """清空索引."""
        with self._lock:
            self._postings.clear()
            self._doc_lengths.clear()
            self._total_length = 0
            self._doc_count = 0


# ============================================================
# 向量索引 — 相似性搜索
# ============================================================


class VectorIndex:
    """向量相似性索引 (借鉴 Milvus IVF/FLAT + HNSW 接口预留).

    当前实现为暴力搜索 (FLAT)，接口设计兼容未来 HNSW 替换。

    支持余弦相似度和欧氏距离两种度量。
    预过滤模式：先按元数据过滤，再进行向量搜索 (借鉴 Weaviate 预过滤).

    Attributes:
        _vectors: 向量存储 {id: (vector, metadata)}
        _dim: 向量维度
        _metric: 相似度度量 ("cosine" 或 "euclidean")
        _lock: 线程安全锁
    """

    def __init__(
        self,
        dim: int = 0,
        metric: str = "cosine",
    ) -> None:
        self._vectors: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self._dim = dim
        self._metric = metric
        self._lock = threading.RLock()

    def add(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加向量.

        Args:
            vector_id: 向量唯一标识
            vector: 密集向量
            metadata: 元数据 (用于预过滤)
        """
        with self._lock:
            if self._dim == 0 and vector:
                self._dim = len(vector)
            self._vectors[vector_id] = (list(vector), metadata or {})

    def remove(self, vector_id: str) -> bool:
        """移除向量."""
        with self._lock:
            if vector_id in self._vectors:
                del self._vectors[vector_id]
                return True
            return False

    def get(self, vector_id: str) -> tuple[list[float], dict[str, Any]] | None:
        """获取向量."""
        with self._lock:
            return self._vectors.get(vector_id)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_fn: Any = None,
    ) -> list[tuple[str, float]]:
        """向量相似性搜索.

        Args:
            query_vector: 查询向量
            top_k: 返回前 k 个结果
            filter_fn: 预过滤函数 (metadata -> bool)，借鉴 Weaviate 预过滤

        Returns:
            [(vector_id, score)] 按分数降序排列
        """
        with self._lock:
            if not self._vectors:
                return []

            results: list[tuple[str, float]] = []
            for vid, (vec, meta) in self._vectors.items():
                # 预过滤 (借鉴 Weaviate 预过滤模式)
                if filter_fn is not None and not filter_fn(meta):
                    continue

                if self._metric == "cosine":
                    score = self._cosine_similarity(query_vector, vec)
                else:
                    score = -self._euclidean_distance(query_vector, vec)

                results.append((vid, score))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _euclidean_distance(a: list[float], b: list[float]) -> float:
        """欧氏距离."""
        if not a or not b or len(a) != len(b):
            return float("inf")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def size(self) -> int:
        """向量数量."""
        with self._lock:
            return len(self._vectors)

    @property
    def dim(self) -> int:
        """向量维度."""
        return self._dim

    def clear(self) -> None:
        """清空索引."""
        with self._lock:
            self._vectors.clear()


# ============================================================
# 名称索引 — 名称/别名到实体ID映射
# ============================================================


class NameIndex:
    """名称索引 (借鉴 Neo4f 全文索引 + 别名映射).

    支持按名称和别名查找实体，忽略大小写。

    Attributes:
        _name_index: 规范化名称到实体ID集合
        _alias_index: 规范化别名到实体ID集合
        _lock: 线程安全锁
    """

    def __init__(self) -> None:
        self._name_index: dict[str, set[str]] = defaultdict(set)
        self._alias_index: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def add_name(self, name: str, entity_id: str) -> None:
        """添加名称索引."""
        with self._lock:
            self._name_index[name.lower()].add(entity_id)

    def add_alias(self, alias: str, entity_id: str) -> None:
        """添加别名索引."""
        with self._lock:
            self._alias_index[alias.lower()].add(entity_id)

    def remove_name(self, name: str, entity_id: str) -> bool:
        """移除名称索引."""
        with self._lock:
            key = name.lower()
            if key in self._name_index:
                self._name_index[key].discard(entity_id)
                if not self._name_index[key]:
                    del self._name_index[key]
                return True
            return False

    def remove_alias(self, alias: str, entity_id: str) -> bool:
        """移除别名索引."""
        with self._lock:
            key = alias.lower()
            if key in self._alias_index:
                self._alias_index[key].discard(entity_id)
                if not self._alias_index[key]:
                    del self._alias_index[key]
                return True
            return False

    def lookup(self, query: str) -> list[str]:
        """按名称或别名查找 (忽略大小写)."""
        with self._lock:
            key = query.lower()
            result: set[str] = set()
            result.update(self._name_index.get(key, set()))
            result.update(self._alias_index.get(key, set()))
            return list(result)

    def prefix_search(self, prefix: str, limit: int = 10) -> list[str]:
        """前缀搜索 (返回匹配的名称列表)."""
        with self._lock:
            prefix_lower = prefix.lower()
            matches: list[str] = []
            for name in self._name_index:
                if name.startswith(prefix_lower):
                    matches.append(name)
                    if len(matches) >= limit:
                        break
            for alias in self._alias_index:
                if alias.startswith(prefix_lower):
                    matches.append(alias)
                    if len(matches) >= limit:
                        break
            return matches

    def clear(self) -> None:
        """清空索引."""
        with self._lock:
            self._name_index.clear()
            self._alias_index.clear()


__all__ = [
    "HashIndex",
    "TypeIndex",
    "InvertedIndex",
    "VectorIndex",
    "NameIndex",
]
