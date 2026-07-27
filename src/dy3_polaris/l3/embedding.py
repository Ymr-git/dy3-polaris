"""L3 领域知识层 — 嵌入管理器.

融合世界先进方案的嵌入管理设计:
- LlamaIndex EmbeddingCache: 嵌入缓存 (避免重复编码, 降低延迟与成本)
- OpenAI Embedding API: 批量嵌入接口 (batch embedding, 减少 API 往返)
- Sentence-Transformers: 多模型支持 (本地/远程模型可插拔)
- Redis Vector: 向量存储优化 (缓存键 + TTL 过期 + LRU 淘汰)
- Pinecone/Milvus: 维度验证与归一化 (向量入库前预处理)

提供四类组件:
1. EmbeddingBackend  — 嵌入后端类型枚举 (openai/sentence_transformers/cohere/custom)
2. EmbeddingResult   — 嵌入结果数据结构 (文本/向量/模型/维度/缓存命中/延迟)
3. EmbeddingCache    — 嵌入缓存 (LRU + TTL, 借鉴 Redis allkeys-lru + 惰性过期)
4. EmbeddingManager  — 嵌入管理器 (多后端 + 缓存 + 批量 + 归一化 + 统计)

设计理念:
- 缓存优先: 每次嵌入先查缓存, 命中则直接返回 (零计算开销), 未命中才计算并回填。
  借鉴 LlamaIndex EmbeddingCache: 相同文本+模型产生相同向量, 缓存可显著降低
  重复编码的延迟与 API 成本。
- LRU + TTL 双重淘汰: 容量超限淘汰最久未使用条目 (借鉴 Redis allkeys-lru),
  超时自动过期 (借鉴 Redis TTL 惰性删除), 兼顾内存占用与新鲜度。
- 维度验证: 编码后校验向量维度与配置一致, 避免维度不匹配导致检索异常。
- L2 归一化: 归一化后向量余弦相似度等价于点积, 加速检索 (借鉴 Pinecone 归一化)。
- 自定义后端: 使用 hashlib 生成确定性伪嵌入向量 (相同文本→相同向量),
  适用于测试与开发环境, 无需外部依赖即可运行。

线程安全: 缓存与统计计数器通过 threading.RLock 保护, 支持并发嵌入。
不依赖外部 AI 库 (openai, sentence-transformers, cohere 等);
OPENAI/SENTENCE_TRANSFORMERS/COHERE 后端接口已预留, 接入时需实现对应 _compute_embedding。

Usage::

    from dy3_polaris.l3.embedding import (
        EmbeddingManager, EmbeddingCache, EmbeddingBackend,
    )

    # 自定义后端 (确定性伪嵌入, 用于测试/开发)
    manager = EmbeddingManager(
        backend=EmbeddingBackend.CUSTOM,
        model_name="pseudo-768",
        dim=768,
        normalize=True,
    )

    # 单文本嵌入
    result = manager.embed("Dy3+离子的发射波长")
    print(result.dim, result.cached, result.latency_ms)

    # 批量嵌入
    results = manager.embed_batch(["波长", "能级", "跃迁"])
    print(manager.stats)

    # 缓存命中 (第二次嵌入相同文本)
    cached = manager.embed("Dy3+离子的发射波长")
    assert cached.cached is True

    # 独立缓存使用
    cache = EmbeddingCache(max_size=5000, ttl_seconds=1800)
    cache.set("text", "model", [0.1, 0.2, 0.3])
    vec = cache.get("text", "model")
    print(cache.stats)
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 嵌入后端枚举
# ============================================================


class EmbeddingBackend(str, Enum):
    """嵌入后端类型 (借鉴 Sentence-Transformers 多模型 + OpenAI API 设计).

    Attributes:
        OPENAI: OpenAI Embedding API (text-embedding-3-small 等)
        SENTENCE_TRANSFORMERS: Sentence-Transformers 本地模型 (BGE/E5 等)
        COHERE: Cohere Embed API
        CUSTOM: 自定义后端 (hashlib 确定性伪嵌入, 用于测试/开发)
    """

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    COHERE = "cohere"
    CUSTOM = "custom"


# ============================================================
# 嵌入结果数据结构
# ============================================================


@dataclass
class EmbeddingResult:
    """嵌入结果 (借鉴 OpenAI Embedding 响应 + LlamaIndex 嵌入元信息).

    Attributes:
        text: 原始文本
        vector: 嵌入向量
        model: 编码模型名称
        dim: 向量维度
        cached: 是否来自缓存 (True 表示缓存命中, 未触发实际编码)
        latency_ms: 本次嵌入耗时 (毫秒), 缓存命中时为查缓存耗时
    """

    text: str
    vector: list[float]
    model: str
    dim: int
    cached: bool
    latency_ms: float

    def __repr__(self) -> str:
        return (
            f"EmbeddingResult(model={self.model!r}, dim={self.dim}, "
            f"cached={self.cached}, latency_ms={self.latency_ms:.3f})"
        )


# ============================================================
# 嵌入缓存 — LRU + TTL
# ============================================================


class EmbeddingCache:
    """嵌入缓存 (借鉴 LlamaIndex EmbeddingCache + Redis allkeys-lru + TTL).

    基于 (text, model) 复合键缓存嵌入向量, 避免重复编码。
    采用 OrderedDict 实现 O(1) get/set + LRU 容量淘汰 + TTL 惰性过期。

    设计要点:
    - 缓存键: SHA-256(text || model) 哈希, 避免长文本作为键的内存开销
      (借鉴 Elasticsearch request cache key 哈希策略)
    - TTL 惰性过期: get 时检查过期, 避免后台轮询开销 (借鉴 Redis 惰性删除)
    - LRU 容量淘汰: 容量超限时淘汰最久未使用条目 (借鉴 Redis allkeys-lru)
    - 统计追踪: 命中/未命中/淘汰计数, 支持命中率监控

    Attributes:
        _max_size: 最大缓存条目数 (<=0 禁用缓存)
        _ttl: 生存时间 (秒), <=0 永不过期
        _cache: OrderedDict {hash_key: (vector, expiry_ts)}
        _index: hash_key -> (text, model) 反向索引 (便于统计与清理)
        _lock: 线程安全锁
        _hits/_misses/_evictions: 统计计数器
    """

    def __init__(
        self, *, max_size: int = 10000, ttl_seconds: int = 3600
    ) -> None:
        """初始化嵌入缓存.

        Args:
            max_size: 最大缓存条目数, <=0 表示禁用缓存
            ttl_seconds: 生存时间 (秒), <=0 表示永不过期
        """
        self._max_size: int = max_size
        self._ttl: float = float(ttl_seconds)
        self._cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._index: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()
        # 统计
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    # --------------------------------------------------------
    # 内部辅助
    # --------------------------------------------------------

    @staticmethod
    def _make_key(text: str, model: str) -> str:
        """生成缓存键 (SHA-256(text || model), 借鉴 ES request cache key).

        Args:
            text: 文本
            model: 模型名称

        Returns:
            SHA-256 十六进制哈希字符串
        """
        payload = f"{text}\x00{model}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _is_expired(self, expiry: float) -> bool:
        """判断条目是否已过期 (expiry=0 表示永不过期)."""
        return expiry > 0.0 and time.time() >= expiry

    def _compute_expiry(self) -> float:
        """计算过期时间戳 (0.0 表示永不过期)."""
        if self._ttl <= 0.0:
            return 0.0
        return time.time() + self._ttl

    def _evict_if_needed(self) -> None:
        """容量超限时淘汰 LRU 端条目 (已持有锁)."""
        while self._max_size > 0 and len(self._cache) > self._max_size:
            key, _ = self._cache.popitem(last=False)
            self._index.pop(key, None)
            self._evictions += 1

    # --------------------------------------------------------
    # 核心操作
    # --------------------------------------------------------

    def get(self, text: str, model: str) -> list[float] | None:
        """获取缓存的嵌入向量 (命中时更新 LRU 顺序).

        Args:
            text: 文本
            model: 模型名称

        Returns:
            缓存的向量, 未命中或已过期返回 None
        """
        with self._lock:
            if self._max_size <= 0:
                self._misses += 1
                return None

            key = self._make_key(text, model)
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            vector, expiry = entry
            # 惰性过期检查
            if self._is_expired(expiry):
                del self._cache[key]
                self._index.pop(key, None)
                self._misses += 1
                logger.debug("嵌入缓存条目过期: %s", key[:12])
                return None

            # 命中: 移动到 MRU 端
            self._cache.move_to_end(key)
            self._hits += 1
            return vector

    def set(self, text: str, model: str, vector: list[float]) -> None:
        """写入嵌入向量到缓存 (覆盖已存在的键).

        Args:
            text: 文本
            model: 模型名称
            vector: 嵌入向量
        """
        with self._lock:
            if self._max_size <= 0:
                return

            key = self._make_key(text, model)
            expiry = self._compute_expiry()
            if key in self._cache:
                self._cache[key] = (vector, expiry)
                self._cache.move_to_end(key)
                self._index[key] = (text, model)
                return

            self._cache[key] = (vector, expiry)
            self._index[key] = (text, model)
            self._evict_if_needed()

    def clear(self) -> None:
        """清空所有缓存条目 (不影响统计计数)."""
        with self._lock:
            self._cache.clear()
            self._index.clear()

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """返回缓存统计信息.

        Returns:
            包含 size/max_size/ttl/hits/misses/evictions/hit_rate 的字典
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "total_requests": total,
                "hit_rate": (self._hits / total) if total > 0 else 0.0,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        with self._lock:
            total = self._hits + self._misses
            rate = (self._hits / total) if total > 0 else 0.0
        return (
            f"EmbeddingCache(size={len(self)}, max_size={self._max_size}, "
            f"hit_rate={rate:.4f})"
        )


# ============================================================
# 嵌入管理器
# ============================================================


class EmbeddingManager:
    """嵌入管理器 (借鉴 LlamaIndex Embedding + OpenAI batch API + Pinecone 归一化).

    统一管理文本嵌入的生成、缓存、归一化与统计。

    功能:
    1. 多后端支持 (OpenAI/Sentence-Transformers/Cohere/Custom)
       - CUSTOM 后端: hashlib 确定性伪嵌入 (测试/开发用, 无外部依赖)
       - 其他后端: 接口已预留, 需实现对应 _compute_embedding
    2. LRU+TTL 缓存: 避免重复编码, 降低延迟与成本
    3. 批量嵌入接口: 批量编码减少单次开销 (借鉴 OpenAI batch embedding)
    4. 维度验证: 编码后校验向量维度与配置一致
    5. L2 归一化: 归一化后余弦相似度等价于点积, 加速检索
    6. 嵌入统计: 总编码数/缓存命中/未命中/平均延迟/缓存统计

    设计理念:
    - 缓存优先 (cache-aside): embed() 先查缓存, 命中直接返回, 未命中才计算并回填。
    - 确定性伪嵌入 (CUSTOM): 基于 SHA-256 重复哈希生成 dim 维向量,
      保证相同文本生成相同向量, 适用于无外部模型的测试与开发环境。
    - 归一化幂等: 已归一化向量再次归一化保持不变 (数值稳定)。

    Attributes:
        _backend: 嵌入后端
        _model_name: 模型名称
        _dim: 期望向量维度
        _cache: 嵌入缓存 (None 表示禁用缓存)
        _normalize: 是否 L2 归一化
        _lock: 统计计数器锁
        _total_embeds: 总嵌入次数 (含缓存命中)
        _cache_hits: 缓存命中次数
        _cache_misses: 缓存未命中次数
        _compute_total: 实际计算次数 (不含缓存命中)
        _total_latency_ms: 累计延迟 (毫秒)
    """

    def __init__(
        self,
        *,
        backend: EmbeddingBackend = EmbeddingBackend.CUSTOM,
        model_name: str = "default",
        dim: int = 768,
        cache: EmbeddingCache | None = None,
        normalize: bool = True,
    ) -> None:
        """初始化嵌入管理器.

        Args:
            backend: 嵌入后端, 默认 CUSTOM (确定性伪嵌入)
            model_name: 模型名称 (用于缓存键隔离与结果标注)
            dim: 期望向量维度 (必须 > 0)
            cache: 嵌入缓存, None 时自动创建默认缓存 (max_size=10000, ttl=3600)
            normalize: 是否对向量进行 L2 归一化
        """
        if dim <= 0:
            raise ValueError(f"嵌入维度必须为正整数, 得到: {dim}")

        self._backend: EmbeddingBackend = backend
        self._model_name: str = model_name
        self._dim: int = dim
        self._normalize: bool = normalize
        # None 表示禁用缓存; 否则使用传入或自动创建的缓存
        self._cache: EmbeddingCache | None = (
            cache if cache is not None else EmbeddingCache()
        )

        # 统计计数器
        self._lock = threading.RLock()
        self._total_embeds: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._compute_total: int = 0
        self._total_latency_ms: float = 0.0

    # --------------------------------------------------------
    # 透明属性
    # --------------------------------------------------------

    @property
    def backend(self) -> EmbeddingBackend:
        """返回嵌入后端类型."""
        return self._backend

    @property
    def model_name(self) -> str:
        """返回模型名称."""
        return self._model_name

    @property
    def dim(self) -> int:
        """返回期望向量维度."""
        return self._dim

    @property
    def normalize(self) -> bool:
        """返回是否启用 L2 归一化."""
        return self._normalize

    @property
    def cache(self) -> EmbeddingCache | None:
        """返回嵌入缓存对象 (None 表示禁用缓存)."""
        return self._cache

    # --------------------------------------------------------
    # 嵌入计算 (后端分发)
    # --------------------------------------------------------

    def _compute_embedding(self, text: str) -> list[float]:
        """计算文本嵌入向量 (后端分发, 不查缓存).

        根据后端类型调用对应的编码逻辑:
        - CUSTOM: hashlib 确定性伪嵌入 (默认, 无外部依赖)
        - OPENAI/SENTENCE_TRANSFORMERS/COHERE: 需接入对应 SDK,
          当前抛出 NotImplementedError 提示需实现外部后端。

        Args:
            text: 待编码文本

        Returns:
            嵌入向量 (长度等于 dim)

        Raises:
            NotImplementedError: 后端需要外部 AI 库但尚未实现
        """
        if self._backend == EmbeddingBackend.CUSTOM:
            return self._compute_pseudo_embedding(text)

        # 外部后端: 接口已预留, 接入时实现对应 SDK 调用
        raise NotImplementedError(
            f"嵌入后端 {self._backend.value} 需要外部 AI 库支持, "
            f"请在子类中实现 _compute_embedding。当前仅 CUSTOM 后端可用。"
        )

    def _compute_pseudo_embedding(self, text: str) -> list[float]:
        """生成确定性伪嵌入向量 (CUSTOM 后端, 借鉴 hashing trick).

        使用 SHA-256 重复哈希生成 dim 维伪随机向量, 保证:
        - 确定性: 相同文本 → 相同向量 (可复现)
        - 分布均匀: 哈希值映射到 [-1, 1] 区间, 近似均匀分布
        - 无外部依赖: 仅用 hashlib 标准库

        适用于测试与开发环境, 模拟真实嵌入的向量空间行为
        (相同文本相似度为 1, 不同文本相似度趋近于 0)。

        Args:
            text: 待编码文本

        Returns:
            dim 维伪嵌入向量 (未归一化, 值域 [-1, 1])
        """
        vector: list[float] = []
        seed = text.encode("utf-8")
        counter = 0
        while len(vector) < self._dim:
            # 重复哈希: seed + counter → 32 字节 → 多个浮点数
            digest = hashlib.sha256(
                seed + counter.to_bytes(4, "big")
            ).digest()
            # 每 4 字节生成一个浮点数 (32 字节 → 8 个浮点)
            for i in range(0, len(digest) - 3, 4):
                if len(vector) >= self._dim:
                    break
                chunk = digest[i : i + 4]
                # 映射到 [0, 1] 再变换到 [-1, 1]
                val = int.from_bytes(chunk, "big") / 0xFFFFFFFF
                vector.append(val * 2.0 - 1.0)
            counter += 1
        return vector[: self._dim]

    # --------------------------------------------------------
    # 归一化
    # --------------------------------------------------------

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """L2 归一化向量 (借鉴 Pinecone 归一化预处理).

        归一化后 ||v|| = 1, 余弦相似度等价于点积, 加速检索计算。
        零向量保持不变 (避免除零)。

        Args:
            vector: 待归一化向量

        Returns:
            L2 归一化后的向量
        """
        norm = math.sqrt(sum(x * x for x in vector))
        if norm < 1e-12:
            return list(vector)
        return [x / norm for x in vector]

    # --------------------------------------------------------
    # 嵌入接口
    # --------------------------------------------------------

    def embed(self, text: str) -> EmbeddingResult:
        """嵌入单条文本 (缓存优先).

        流程:
        1. 查缓存 → 命中则直接返回 (cached=True)
        2. 未命中 → 计算嵌入 → 维度验证 → 归一化 → 回填缓存 → 返回 (cached=False)

        Args:
            text: 待嵌入文本

        Returns:
            EmbeddingResult 嵌入结果
        """
        start = time.perf_counter()

        # 1. 查缓存
        cached_vector: list[float] | None = None
        if self._cache is not None:
            cached_vector = self._cache.get(text, self._model_name)

        if cached_vector is not None:
            # 缓存命中
            latency_ms = (time.perf_counter() - start) * 1000.0
            with self._lock:
                self._total_embeds += 1
                self._cache_hits += 1
                self._total_latency_ms += latency_ms
            return EmbeddingResult(
                text=text,
                vector=cached_vector,
                model=self._model_name,
                dim=len(cached_vector),
                cached=True,
                latency_ms=latency_ms,
            )

        # 2. 未命中: 计算嵌入
        vector = self._compute_embedding(text)

        # 维度验证
        if len(vector) != self._dim:
            raise ValueError(
                f"嵌入维度不匹配: 期望 {self._dim}, 实际 {len(vector)} "
                f"(model={self._model_name}, text={text[:50]!r})"
            )

        # 归一化
        if self._normalize:
            vector = self._normalize_vector(vector)

        # 回填缓存
        if self._cache is not None:
            self._cache.set(text, self._model_name, vector)

        latency_ms = (time.perf_counter() - start) * 1000.0
        with self._lock:
            self._total_embeds += 1
            self._cache_misses += 1
            self._compute_total += 1
            self._total_latency_ms += latency_ms

        return EmbeddingResult(
            text=text,
            vector=vector,
            model=self._model_name,
            dim=len(vector),
            cached=False,
            latency_ms=latency_ms,
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """批量嵌入文本 (借鉴 OpenAI batch embedding API).

        逐条调用 embed(), 自动复用缓存。批量接口为未来接入支持批量编码的
        外部后端 (如 OpenAI batch API) 预留扩展点。

        Args:
            texts: 待嵌入文本列表

        Returns:
            嵌入结果列表 (与输入文本顺序一一对应)
        """
        return [self.embed(t) for t in texts]

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """返回嵌入管理器统计信息.

        Returns:
            包含 backend/model/dim/normalize/total_embeds/cache_hits/
            cache_misses/compute_count/avg_latency_ms/cache_stats 的字典
        """
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._total_embeds
                if self._total_embeds > 0
                else 0.0
            )
            cache_hit_rate = (
                self._cache_hits / self._total_embeds
                if self._total_embeds > 0
                else 0.0
            )
            return {
                "backend": self._backend.value,
                "model": self._model_name,
                "dim": self._dim,
                "normalize": self._normalize,
                "total_embeds": self._total_embeds,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "compute_count": self._compute_total,
                "cache_hit_rate": cache_hit_rate,
                "avg_latency_ms": avg_latency,
                "total_latency_ms": self._total_latency_ms,
                "cache_stats": (
                    self._cache.stats if self._cache is not None else None
                ),
            }

    def reset_stats(self) -> None:
        """重置嵌入统计计数器 (不影响缓存)."""
        with self._lock:
            self._total_embeds = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._compute_total = 0
            self._total_latency_ms = 0.0

    def __repr__(self) -> str:
        return (
            f"EmbeddingManager(backend={self._backend.value}, "
            f"model={self._model_name!r}, dim={self._dim}, "
            f"normalize={self._normalize})"
        )


__all__ = [
    "EmbeddingBackend",
    "EmbeddingResult",
    "EmbeddingCache",
    "EmbeddingManager",
]
