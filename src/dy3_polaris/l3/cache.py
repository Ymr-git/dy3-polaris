"""L3 领域知识层 — 知识存储引擎缓存层.

融合世界先进方案的缓存设计:
- Redis: maxmemory-policy (allkeys-lru) 淘汰策略 + TTL 过期 + 查询缓存
- Caffeine: W-TinyLFU 频率+新近度混合淘汰 + 装饰器包装模式 + 透明缓存
- Elasticsearch: request cache (查询结果缓存 + 写时失效)
- Vercel ISR: stale-while-revalidate 理念 (时间驱动的自动失效)
- Python functools.lru_cache: 透明函数级缓存 + 装饰器模式
- Neo4j: query cache + 计划缓存 (图遍历结果复用)

提供四类组件:
1. CacheStats          — 缓存统计信息 (命中/未命中/淘汰/命中率)
2. LRUCache[K, V]      — 通用 LRU 缓存 (O(1) get/put + 容量淘汰 + TTL 过期)
3. QueryCache          — 知识查询缓存 (实体/检索/遍历结果分命名空间缓存)
4. CachedKnowledgeStore — KnowledgeStore 装饰器 (读缓存 + 写失效 + 级联失效)

失效策略 (三层):
1. 写失效 (write-through): 实体/三元组/切片变更时自动失效相关缓存
2. TTL 失效: 基于生存时间的自动过期 (借鉴 Redis TTL + Vercel ISR)
3. 容量淘汰: LRU 淘汰最久未使用的条目 (借鉴 Redis allkeys-lru)

缓存键策略:
- 查询参数序列化为 JSON, 取 SHA-256 哈希作为键 (借鉴 Elasticsearch request cache key)
- 支持命名空间隔离 (entity/triple/chunk/query/retrieval)

线程安全: 所有共享状态通过 threading.RLock 保护。
所有缓存均为内存实现，接口设计支持未来替换为分布式缓存后端 (如 Redis/Memcached)。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict, defaultdict
from typing import Any, Generic, TypeVar

from .models import DocumentChunk, KnowledgeEntity, KnowledgeTriple, RetrievalResult
from .store import KnowledgeStore

logger = logging.getLogger(__name__)

# 泛型类型变量
K = TypeVar("K")
V = TypeVar("V")


# ============================================================
# 缓存统计 — 命中率与淘汰追踪
# ============================================================


class CacheStats:
    """缓存统计信息 (借鉴 Caffeine CacheStats + Micrometer 指标).

    追踪缓存命中、未命中、淘汰次数，计算命中率。
    统计计数器在 LRUCache 的锁保护下递增 (CPython GIL 保证单条 int 自增原子)，
    reset/to_dict 通过自身锁提供一致性快照。

    Attributes:
        hits: 缓存命中次数
        misses: 缓存未命中次数
        evictions: 容量淘汰次数 (不含 TTL 过期)
        total_requests: 总请求次数 (hits + misses)
    """

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.total_requests: int = 0
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 内部记录接口 (由 LRUCache 在其锁内调用)
    # --------------------------------------------------------

    def record_hit(self) -> None:
        """记录一次命中."""
        self.hits += 1
        self.total_requests += 1

    def record_miss(self) -> None:
        """记录一次未命中."""
        self.misses += 1
        self.total_requests += 1

    def record_eviction(self) -> None:
        """记录一次容量淘汰."""
        self.evictions += 1

    # --------------------------------------------------------
    # 派生指标
    # --------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """缓存命中率 (hits / total_requests).

        Returns:
            命中率 [0.0, 1.0]，无请求时返回 0.0
        """
        with self._lock:
            if self.total_requests == 0:
                return 0.0
            return self.hits / self.total_requests

    # --------------------------------------------------------
    # 操作接口
    # --------------------------------------------------------

    def reset(self) -> None:
        """重置所有统计计数器为零."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.total_requests = 0

    def to_dict(self) -> dict[str, Any]:
        """导出统计信息为字典 (便于序列化与监控上报).

        Returns:
            包含 hits/misses/evictions/total_requests/hit_rate 的字典
        """
        with self._lock:
            total = self.total_requests
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "total_requests": total,
                "hit_rate": (self.hits / total) if total > 0 else 0.0,
            }

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, hit_rate={self.hit_rate:.4f})"
        )


# ============================================================
# LRU 缓存 — O(1) get/put + 容量淘汰 + TTL 过期
# ============================================================


class LRUCache(Generic[K, V]):
    """LRU 缓存 (借鉴 Redis maxmemory-policy + Caffeine W-TinyLFU).

    提供 O(1) 的 get/put 操作，支持容量上限、TTL 过期、统计信息。
    使用 OrderedDict 实现 LRU 淘汰策略: 访问/更新时移动到末尾 (MRU 端)，
    容量超限时从头部 (LRU 端) 弹出最久未使用的条目。

    设计要点:
    - TTL 采用惰性过期 (lazy expiration): get 时检查并清除过期条目，
      避免后台轮询线程的开销 (借鉴 Redis 惰性删除)。
    - 容量淘汰采用近似 LRU (借鉴 Redis approximated LRU 采样思想，
      此处使用精确 LRU 基于 OrderedDict，O(1) 复杂度)。
    - 统计信息独立追踪，可重置与导出。

    Attributes:
        max_size: 最大缓存条目数 (<=0 表示禁用缓存)
        ttl: 生存时间 (秒), 0 表示永不过期
        _cache: OrderedDict 存储 {key: (value, expiry_timestamp)}
        _lock: 线程安全锁 (RLock 支持重入)
        _stats: 缓存统计
    """

    def __init__(self, max_size: int = 1024, ttl: float = 0.0) -> None:
        """初始化 LRU 缓存.

        Args:
            max_size: 最大缓存条目数，<=0 表示禁用缓存 (put 为空操作)
            ttl: 默认生存时间 (秒)，<=0 表示永不过期
        """
        self.max_size: int = max_size
        self.ttl: float = ttl
        self._cache: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()

    # --------------------------------------------------------
    # 内部辅助
    # --------------------------------------------------------

    def _is_expired(self, expiry: float) -> bool:
        """判断条目是否已过期 (expiry=0 表示永不过期)."""
        return expiry > 0.0 and time.time() >= expiry

    def _compute_expiry(self, ttl: float | None) -> float:
        """计算过期时间戳.

        Args:
            ttl: 生存时间，None 表示使用默认 ttl，<=0 表示永不过期

        Returns:
            过期时间戳，0.0 表示永不过期
        """
        effective_ttl = self.ttl if ttl is None else ttl
        if effective_ttl <= 0.0:
            return 0.0
        return time.time() + effective_ttl

    def _evict_if_needed(self) -> None:
        """容量超限时淘汰 LRU 端条目 (已持有锁)."""
        while self.max_size > 0 and len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # 弹出最旧 (LRU 端)
            self._stats.record_eviction()

    # --------------------------------------------------------
    # 核心操作
    # --------------------------------------------------------

    def get(self, key: K) -> V | None:
        """获取缓存值 (命中时更新 LRU 顺序).

        Args:
            key: 缓存键

        Returns:
            缓存值，未命中或已过期返回 None
        """
        with self._lock:
            # 禁用缓存模式
            if self.max_size <= 0:
                self._stats.record_miss()
                return None

            entry = self._cache.get(key)
            if entry is None:
                self._stats.record_miss()
                return None

            value, expiry = entry
            # 惰性过期检查
            if self._is_expired(expiry):
                del self._cache[key]
                self._stats.record_miss()
                logger.debug("LRU 缓存条目过期: %s", key)
                return None

            # 命中: 移动到 MRU 端
            self._cache.move_to_end(key)
            self._stats.record_hit()
            return value

    def put(self, key: K, value: V, ttl: float | None = None) -> None:
        """写入缓存值 (覆盖已存在的键).

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 本次写入的生存时间 (秒)，None 使用默认 ttl，<=0 永不过期
        """
        with self._lock:
            # 禁用缓存模式
            if self.max_size <= 0:
                return

            expiry = self._compute_expiry(ttl)
            if key in self._cache:
                # 更新已存在条目并提升到 MRU 端
                self._cache[key] = (value, expiry)
                self._cache.move_to_end(key)
                return

            self._cache[key] = (value, expiry)
            self._evict_if_needed()

    def remove(self, key: K) -> bool:
        """移除指定键的缓存条目.

        Args:
            key: 缓存键

        Returns:
            键存在并已移除返回 True，否则返回 False
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存条目 (不影响统计计数)."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """返回当前缓存条目数 (含尚未惰性清除的过期条目)."""
        with self._lock:
            return len(self._cache)

    def keys(self) -> list[K]:
        """返回所有缓存键 (按 LRU 顺序，LRU 端在前)."""
        with self._lock:
            return list(self._cache.keys())

    def stats(self) -> CacheStats:
        """返回缓存统计对象 (返回活动对象，可直接 reset 或读取).

        Returns:
            CacheStats 实例
        """
        return self._stats

    def contains(self, key: K) -> bool:
        """判断键是否存在且未过期 (不影响 LRU 顺序与统计).

        Args:
            key: 缓存键

        Returns:
            存在且未过期返回 True，否则返回 False
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            _, expiry = entry
            if self._is_expired(expiry):
                del self._cache[key]
                return False
            return True

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, key: object) -> bool:
        # 委托 contains 实现，仅当键类型兼容时才检查
        return self.contains(key)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return (
            f"LRUCache(max_size={self.max_size}, ttl={self.ttl}, "
            f"size={self.size()})"
        )


# ============================================================
# 知识查询缓存 — 分命名空间的查询结果缓存
# ============================================================


class QueryCache:
    """知识查询缓存 (借鉴 Elasticsearch request cache + Redis query cache).

    缓存检索结果、实体查询、图遍历结果。
    支持基于内容哈希的缓存键、自动失效、手动失效。

    失效策略:
    1. 写失效 (write-through): 实体/三元组/切片变更时自动失效相关缓存
    2. TTL 失效: 基于时间的自动过期 (委托 LRUCache 惰性过期)
    3. 容量淘汰: LRU 淘汰最久未使用的条目 (委托 LRUCache)

    缓存键策略:
    - 查询参数序列化为 JSON, 取 SHA-256 哈希作为键
    - 支持命名空间隔离 (entity/triple/chunk/query/retrieval)

    依赖追踪 (用于精确级联失效):
    - _entity_to_retrievals: 实体 ID -> 依赖该实体的检索缓存键集合
    - _retrieval_to_entities: 检索缓存键 -> 其依赖的实体 ID 集合
    - _traversal_index: 遍历源实体 ID -> 遍历缓存键集合
    - _entity_domain: 实体 ID -> 所属领域 (用于按领域失效)

    注意: 依赖追踪为尽力而为 (best-effort)，仅追踪检索结果中显式包含的
    实体 ID。LRU 容量淘汰可能产生少量悬挂的依赖记录 (指向已淘汰的缓存键)，
    这些记录在下次失效时会被安全清理，不影响正确性。

    Attributes:
        _entity_cache: 实体缓存
        _retrieval_cache: 检索结果缓存
        _traversal_cache: 图遍历结果缓存
        _entity_domain: 实体 -> 领域映射
        _entity_to_retrievals: 实体 -> 依赖检索键集合
        _retrieval_to_entities: 检索键 -> 依赖实体集合
        _traversal_index: 遍历源 -> 遍历键集合
        _lock: 线程安全锁
    """

    def __init__(self, max_size: int = 2048, ttl: float = 300.0) -> None:
        """初始化查询缓存.

        Args:
            max_size: 每个命名空间的最大条目数
            ttl: 默认生存时间 (秒)，默认 5 分钟
        """
        self._entity_cache: LRUCache[str, KnowledgeEntity] = LRUCache(
            max_size=max_size, ttl=ttl
        )
        self._retrieval_cache: LRUCache[str, RetrievalResult] = LRUCache(
            max_size=max_size, ttl=ttl
        )
        self._traversal_cache: LRUCache[str, list[dict]] = LRUCache(
            max_size=max_size, ttl=ttl
        )
        # 依赖追踪结构
        self._entity_domain: dict[str, str] = {}
        self._entity_to_retrievals: dict[str, set[str]] = defaultdict(set)
        self._retrieval_to_entities: dict[str, set[str]] = {}
        self._traversal_index: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 缓存键工具
    # --------------------------------------------------------

    @staticmethod
    def make_query_hash(query: str, **kwargs: Any) -> str:
        """生成查询缓存键 (借鉴 Elasticsearch request cache key).

        将查询文本与参数序列化为规范 JSON (键排序)，取 SHA-256 哈希。
        确保相同查询参数产生相同缓存键。

        Args:
            query: 查询文本
            **kwargs: 查询参数

        Returns:
            SHA-256 十六进制哈希字符串
        """
        payload = {"query": query, "kwargs": kwargs}
        serialized = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, default=str
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _traversal_key(source_id: str, max_depth: int) -> str:
        """构造图遍历缓存键."""
        return f"traversal:{source_id}:{max_depth}"

    # --------------------------------------------------------
    # 依赖提取
    # --------------------------------------------------------

    @staticmethod
    def _extract_entity_ids(result: RetrievalResult) -> set[str]:
        """从检索结果中提取涉及的实体 ID (尽力而为).

        扫描结果字典中的 entity_id / subject_id / object_id 字段，
        用于建立检索结果到实体的依赖关系，支持精确级联失效。

        Args:
            result: 检索结果

        Returns:
            涉及的实体 ID 集合
        """
        ids: set[str] = set()
        for item in result.results:
            if isinstance(item, dict):
                for field in ("entity_id", "subject_id", "object_id"):
                    val = item.get(field)
                    if isinstance(val, str) and val:
                        ids.add(val)
        return ids

    # --------------------------------------------------------
    # 实体缓存
    # --------------------------------------------------------

    def get_entity(self, entity_id: str) -> KnowledgeEntity | None:
        """获取缓存的实体.

        Args:
            entity_id: 实体 ID

        Returns:
            缓存的实体，未命中返回 None
        """
        return self._entity_cache.get(entity_id)

    def put_entity(self, entity_id: str, entity: KnowledgeEntity) -> None:
        """缓存实体并记录其领域映射.

        Args:
            entity_id: 实体 ID
            entity: 实体对象
        """
        with self._lock:
            self._entity_cache.put(entity_id, entity)
            self._entity_domain[entity_id] = entity.domain

    # --------------------------------------------------------
    # 检索结果缓存
    # --------------------------------------------------------

    def get_retrieval(self, query_hash: str) -> RetrievalResult | None:
        """获取缓存的检索结果.

        Args:
            query_hash: 查询哈希 (由 make_query_hash 生成)

        Returns:
            缓存的检索结果，未命中返回 None
        """
        return self._retrieval_cache.get(query_hash)

    def put_retrieval(self, query_hash: str, result: RetrievalResult) -> None:
        """缓存检索结果并建立实体依赖关系.

        Args:
            query_hash: 查询哈希
            result: 检索结果
        """
        with self._lock:
            # 刷新场景: 先清理旧的依赖关系
            old_deps = self._retrieval_to_entities.get(query_hash)
            if old_deps is not None:
                for eid in old_deps:
                    self._entity_to_retrievals[eid].discard(query_hash)

            # 建立新的依赖关系
            new_deps = self._extract_entity_ids(result)
            self._retrieval_to_entities[query_hash] = new_deps
            for eid in new_deps:
                self._entity_to_retrievals[eid].add(query_hash)

            self._retrieval_cache.put(query_hash, result)
            logger.debug(
                "缓存检索结果: %s (依赖 %d 个实体)", query_hash, len(new_deps)
            )

    # --------------------------------------------------------
    # 图遍历结果缓存
    # --------------------------------------------------------

    def get_traversal(
        self, source_id: str, max_depth: int
    ) -> list[dict] | None:
        """获取缓存的图遍历结果.

        Args:
            source_id: 遍历起始实体 ID
            max_depth: 最大遍历深度

        Returns:
            缓存的遍历结果列表，未命中返回 None
        """
        return self._traversal_cache.get(self._traversal_key(source_id, max_depth))

    def put_traversal(
        self, source_id: str, max_depth: int, result: list[dict]
    ) -> None:
        """缓存图遍历结果.

        Args:
            source_id: 遍历起始实体 ID
            max_depth: 最大遍历深度
            result: 遍历结果列表
        """
        with self._lock:
            key = self._traversal_key(source_id, max_depth)
            self._traversal_index[source_id].add(key)
            self._traversal_cache.put(key, result)
            logger.debug("缓存图遍历结果: %s (depth=%d)", source_id, max_depth)

    # --------------------------------------------------------
    # 失效操作
    # --------------------------------------------------------

    def invalidate_entity(self, entity_id: str) -> int:
        """失效与指定实体相关的所有缓存 (级联失效).

        失效范围:
        1. 实体缓存中的该实体条目
        2. 以该实体为源的图遍历结果
        3. 依赖该实体的检索结果 (基于依赖追踪)

        Args:
            entity_id: 实体 ID

        Returns:
            实际清除的缓存条目数
        """
        with self._lock:
            count = 0

            # 1. 实体缓存条目
            if self._entity_cache.remove(entity_id):
                count += 1

            # 2. 以该实体为源的图遍历结果
            traversal_keys = self._traversal_index.pop(entity_id, set())
            for tk in traversal_keys:
                if self._traversal_cache.remove(tk):
                    count += 1

            # 3. 依赖该实体的检索结果
            retrieval_keys = self._entity_to_retrievals.pop(entity_id, set())
            for rk in retrieval_keys:
                if self._retrieval_cache.remove(rk):
                    count += 1
                self._retrieval_to_entities.pop(rk, None)

            # 4. 领域映射
            self._entity_domain.pop(entity_id, None)

            logger.debug("失效实体相关缓存: %s (清除 %d 条)", entity_id, count)
            return count

    def invalidate_domain(self, domain: str) -> int:
        """失效指定领域内所有实体的相关缓存.

        Args:
            domain: 领域标识

        Returns:
            实际清除的缓存条目数
        """
        with self._lock:
            # 先收集实体 ID (invalidate_entity 会修改 _entity_domain)
            entity_ids = [
                eid for eid, dom in self._entity_domain.items() if dom == domain
            ]
            total = 0
            for eid in entity_ids:
                total += self.invalidate_entity(eid)
            logger.debug("失效领域相关缓存: %s (清除 %d 条)", domain, total)
            return total

    def invalidate_retrievals(self) -> int:
        """失效全部检索结果缓存 (保留实体与遍历缓存).

        用于切片/三元组变更等难以精确确定受影响实体的场景，
        保守地清除所有检索结果 (借鉴 Elasticsearch shard refresh 清空 request cache)。

        Returns:
            清除的检索结果条目数
        """
        with self._lock:
            count = self._retrieval_cache.size()

            # 清理检索依赖追踪 (正向 + 反向)
            for qh, eids in self._retrieval_to_entities.items():
                for eid in eids:
                    self._entity_to_retrievals[eid].discard(qh)
            self._retrieval_to_entities.clear()

            # 清理 _entity_to_retrievals 中的空集合
            empty_eids = [
                eid for eid, s in self._entity_to_retrievals.items() if not s
            ]
            for eid in empty_eids:
                del self._entity_to_retrievals[eid]

            self._retrieval_cache.clear()
            logger.debug("失效全部检索缓存 (%d 条)", count)
            return count

    def invalidate_all(self) -> int:
        """失效所有缓存 (实体 + 检索 + 遍历).

        Returns:
            清除的缓存条目总数
        """
        with self._lock:
            count = (
                self._entity_cache.size()
                + self._retrieval_cache.size()
                + self._traversal_cache.size()
            )
            self._entity_cache.clear()
            self._retrieval_cache.clear()
            self._traversal_cache.clear()
            self._entity_domain.clear()
            self._entity_to_retrievals.clear()
            self._retrieval_to_entities.clear()
            self._traversal_index.clear()
            logger.debug("失效全部缓存 (%d 条)", count)
            return count

    def clear(self) -> None:
        """清空所有缓存与依赖追踪结构 (不影响统计计数)."""
        with self._lock:
            self._entity_cache.clear()
            self._retrieval_cache.clear()
            self._traversal_cache.clear()
            self._entity_domain.clear()
            self._entity_to_retrievals.clear()
            self._retrieval_to_entities.clear()
            self._traversal_index.clear()

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def stats(self) -> dict[str, CacheStats]:
        """返回各命名空间的缓存统计.

        Returns:
            命名空间名 -> CacheStats 的映射 (entity/retrieval/traversal)
        """
        return {
            "entity": self._entity_cache.stats(),
            "retrieval": self._retrieval_cache.stats(),
            "traversal": self._traversal_cache.stats(),
        }

    def __repr__(self) -> str:
        return (
            f"QueryCache(entity={self._entity_cache.size()}, "
            f"retrieval={self._retrieval_cache.size()}, "
            f"traversal={self._traversal_cache.size()})"
        )


# ============================================================
# 带缓存的 KnowledgeStore 装饰器
# ============================================================


class CachedKnowledgeStore:
    """带缓存的 KnowledgeStore 装饰器 (借鉴装饰器模式 + Caffeine).

    透明包装 KnowledgeStore，对读操作自动缓存，对写操作自动失效。
    支持 cascade invalidation: 修改实体时自动失效相关三元组和检索缓存。

    缓存分层:
    - 实体查询 -> QueryCache._entity_cache (精确失效)
    - 检索结果 -> QueryCache._retrieval_cache (写时失效)
    - 图遍历   -> QueryCache._traversal_cache (实体变更时失效)
    - 三元组   -> 本地 _triple_cache (精确失效)
    - 切片     -> 本地 _chunk_cache (精确失效)

    失效规则:
    - get_* : 缓存未命中时回源并回填缓存
    - add/update/remove entity : 失效该实体 + 相关检索 + 相关遍历
    - add/remove triple : 失效关联实体 (主语/宾语) + 检索缓存
    - add/remove chunk : 失效检索缓存 (切片影响全文/向量检索)

    注意: 检索结果缓存采用保守失效策略 (任何写入清除全部检索缓存)，
    因为切片/三元组变更难以精确确定受影响的检索查询。
    此权衡适用于读多写少的知识库场景 (借鉴 Elasticsearch request cache
    在 shard refresh 时整体失效的策略)。

    Attributes:
        _store: 被包装的内部 KnowledgeStore
        _cache: 查询缓存
        _triple_cache: 三元组点查缓存
        _chunk_cache: 切片点查缓存
        _lock: 线程安全锁
    """

    def __init__(
        self, store: KnowledgeStore, cache: QueryCache | None = None
    ) -> None:
        """初始化带缓存的 KnowledgeStore.

        Args:
            store: 被包装的知识存储
            cache: 查询缓存，None 时自动创建默认 QueryCache
        """
        self._store: KnowledgeStore = store
        self._cache: QueryCache = cache if cache is not None else QueryCache()
        # 三元组与切片的点查缓存 (QueryCache 不含此命名空间，本地维护)
        self._triple_cache: LRUCache[str, KnowledgeTriple] = LRUCache(
            max_size=2048, ttl=300.0
        )
        self._chunk_cache: LRUCache[str, DocumentChunk] = LRUCache(
            max_size=2048, ttl=300.0
        )
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 透明属性
    # --------------------------------------------------------

    @property
    def inner_store(self) -> KnowledgeStore:
        """返回被包装的内部 KnowledgeStore."""
        return self._store

    @property
    def cache(self) -> QueryCache:
        """返回查询缓存对象."""
        return self._cache

    def cache_stats(self) -> dict[str, CacheStats]:
        """返回所有缓存的统计信息.

        Returns:
            命名空间名 -> CacheStats 的映射 (entity/retrieval/traversal/triple/chunk)
        """
        stats = dict(self._cache.stats())
        stats["triple"] = self._triple_cache.stats()
        stats["chunk"] = self._chunk_cache.stats()
        return stats

    # --------------------------------------------------------
    # 实体操作 (读缓存 + 写失效)
    # --------------------------------------------------------

    def get_entity(self, entity_id: str) -> KnowledgeEntity | None:
        """获取实体 (缓存优先，未命中回源并回填).

        Args:
            entity_id: 实体 ID

        Returns:
            实体对象，不存在返回 None
        """
        # 缓存命中
        cached = self._cache.get_entity(entity_id)
        if cached is not None:
            return cached
        # 回源
        entity = self._store.get_entity(entity_id)
        if entity is not None:
            self._cache.put_entity(entity_id, entity)
        return entity

    def add_entity(self, entity: KnowledgeEntity) -> KnowledgeEntity:
        """添加实体并缓存结果.

        新增实体可能使已有检索结果不再完整，故失效检索缓存。

        Args:
            entity: 要添加的实体

        Returns:
            存储后的实体
        """
        result = self._store.add_entity(entity)
        # 回填实体缓存
        self._cache.put_entity(result.entity_id, result)
        # 新数据可能改变检索结果，失效检索缓存
        self._cache.invalidate_retrievals()
        logger.debug("添加实体并刷新缓存: %s", result.entity_id)
        return result

    def update_entity(self, entity_id: str, **updates: Any) -> KnowledgeEntity:
        """更新实体并级联失效相关缓存.

        失效该实体的缓存条目、相关图遍历结果、依赖该实体的检索结果，
        然后回填最新实体到缓存。

        Args:
            entity_id: 实体 ID
            **updates: 要更新的字段

        Returns:
            更新后的实体
        """
        result = self._store.update_entity(entity_id, **updates)
        # 级联失效: 实体缓存 + 遍历 + 相关检索
        self._cache.invalidate_entity(entity_id)
        # 回填最新值
        self._cache.put_entity(entity_id, result)
        logger.debug("更新实体并级联失效缓存: %s", entity_id)
        return result

    def remove_entity(self, entity_id: str) -> KnowledgeEntity | None:
        """移除实体并级联失效相关缓存.

        Args:
            entity_id: 实体 ID

        Returns:
            被移除的实体，不存在返回 None
        """
        result = self._store.remove_entity(entity_id)
        # 级联失效: 实体缓存 + 遍历 + 相关检索 + 领域映射
        self._cache.invalidate_entity(entity_id)
        logger.debug("移除实体并级联失效缓存: %s", entity_id)
        return result

    # --------------------------------------------------------
    # 三元组操作 (本地缓存 + 级联失效)
    # --------------------------------------------------------

    def get_triple(self, triple_id: str) -> KnowledgeTriple | None:
        """获取三元组 (缓存优先，未命中回源并回填).

        Args:
            triple_id: 三元组 ID

        Returns:
            三元组对象，不存在返回 None
        """
        cached = self._triple_cache.get(triple_id)
        if cached is not None:
            return cached
        triple = self._store.get_triple(triple_id)
        if triple is not None:
            self._triple_cache.put(triple_id, triple)
        return triple

    def add_triple(self, triple: KnowledgeTriple) -> KnowledgeTriple:
        """添加三元组并级联失效相关缓存.

        三元组变更影响图遍历与检索结果，故失效关联实体与检索缓存。

        Args:
            triple: 要添加的三元组

        Returns:
            存储后的三元组
        """
        result = self._store.add_triple(triple)
        # 失效主语/宾语相关缓存 (遍历 + 检索)
        self._cache.invalidate_entity(result.subject_id)
        if result.object_id:
            self._cache.invalidate_entity(result.object_id)
        # 三元组变更影响检索结果
        self._cache.invalidate_retrievals()
        logger.debug("添加三元组并级联失效缓存: %s", result.triple_id)
        return result

    def remove_triple(self, triple_id: str) -> KnowledgeTriple | None:
        """移除三元组并级联失效相关缓存.

        Args:
            triple_id: 三元组 ID

        Returns:
            被移除的三元组，不存在返回 None
        """
        # 先回源获取以确定关联实体 (用于级联失效)
        triple = self._store.get_triple(triple_id)
        result = self._store.remove_triple(triple_id)
        # 移除三元组点查缓存
        self._triple_cache.remove(triple_id)
        # 级联失效关联实体与检索缓存
        if triple is not None:
            self._cache.invalidate_entity(triple.subject_id)
            if triple.object_id:
                self._cache.invalidate_entity(triple.object_id)
        self._cache.invalidate_retrievals()
        logger.debug("移除三元组并级联失效缓存: %s", triple_id)
        return result

    # --------------------------------------------------------
    # 切片操作 (本地缓存 + 检索失效)
    # --------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """获取切片 (缓存优先，未命中回源并回填).

        Args:
            chunk_id: 切片 ID

        Returns:
            切片对象，不存在返回 None
        """
        cached = self._chunk_cache.get(chunk_id)
        if cached is not None:
            return cached
        chunk = self._store.get_chunk(chunk_id)
        if chunk is not None:
            self._chunk_cache.put(chunk_id, chunk)
        return chunk

    def add_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """添加切片并失效检索缓存.

        切片变更影响全文检索与向量检索结果。

        Args:
            chunk: 要添加的切片

        Returns:
            存储后的切片
        """
        result = self._store.add_chunk(chunk)
        self._cache.invalidate_retrievals()
        logger.debug("添加切片并失效检索缓存: %s", result.chunk_id)
        return result

    def remove_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """移除切片并失效检索缓存.

        Args:
            chunk_id: 切片 ID

        Returns:
            被移除的切片，不存在返回 None
        """
        result = self._store.remove_chunk(chunk_id)
        self._chunk_cache.remove(chunk_id)
        self._cache.invalidate_retrievals()
        logger.debug("移除切片并失效检索缓存: %s", chunk_id)
        return result

    def __repr__(self) -> str:
        return f"CachedKnowledgeStore(store={self._store!r}, cache={self._cache!r})"
