"""L2 个性化层缓存 — 分层 TTL + write-through 语义.

设计依据:
- 参考 L1 ContextCache 的 TTL 分层缓存模式 (会话级热数据 + 持久层冷数据)
- 借鉴 Redis Agent Memory 两层记忆模型:
  - Session Memory (缓存层): 热数据, 分层 TTL 控制 (profile/bkt/memory)
  - Long-term Memory (backing_store): 冷数据, write-through 持久化

分层 TTL:
- profile : 学情画像热缓存, 默认 300s (5 分钟)
- bkt     : BKT 追踪状态, 默认 30s (高频更新, 短 TTL)
- memory  : 记忆层数据, 默认 600s (10 分钟)

write-through 语义:
- set() 同时写入缓存层与 backing_store (若提供)
- clear() 只清缓存层, 保留 backing_store (持久性)
- 缓存未命中或过期时, get() 自动从 backing_store 恢复

线程安全: threading.RLock 保护缓存层.
"""

from __future__ import annotations

import threading
import time
from typing import Any, MutableMapping


# ============================================================
# 1. 常量定义
# ============================================================

# 默认分层 TTL (秒)
DEFAULT_PROFILE_TTL: int = 300   # 学情画像: 5 分钟
DEFAULT_BKT_TTL: int = 30        # BKT 追踪状态: 30 秒
DEFAULT_MEMORY_TTL: int = 600    # 记忆层: 10 分钟

# 支持的缓存层名称 (get 遍历顺序)
_CACHE_LAYERS: tuple[str, ...] = ("profile", "bkt", "memory")


# ============================================================
# 2. L2Cache 分层 TTL 缓存
# ============================================================


class L2Cache:
    """L2 分层 TTL 缓存 (profile / bkt / memory) + write-through backing store.

    内部存储结构: ``dict[str, dict[str, tuple[Any, float]]]``
    即 ``layer -> key -> (value, expire_ts)``.

    Args:
        profile_ttl: profile 层 TTL (秒), 默认 300
        bkt_ttl: bkt 层 TTL (秒), 默认 30
        memory_ttl: memory 层 TTL (秒), 默认 600
        backing_store: 可选的写穿后端 (任意 dict-like 对象).
            set() 时同步写入; 缓存未命中/过期时 get() 从中恢复.

    Attributes:
        profile_ttl: profile 层 TTL
        bkt_ttl: bkt 层 TTL
        memory_ttl: memory 层 TTL
        backing_store: 写穿后端 (可能为 None)

    线程安全: threading.RLock 保护缓存层 (backing_store 的线程安全由调用方保证).
    """

    def __init__(
        self,
        profile_ttl: int = DEFAULT_PROFILE_TTL,
        bkt_ttl: int = DEFAULT_BKT_TTL,
        memory_ttl: int = DEFAULT_MEMORY_TTL,
        backing_store: MutableMapping[str, Any] | None = None,
    ) -> None:
        self.profile_ttl = profile_ttl
        self.bkt_ttl = bkt_ttl
        self.memory_ttl = memory_ttl
        self.backing_store = backing_store

        # layer -> {key -> (value, expire_ts)}
        self._cache: dict[str, dict[str, tuple[Any, float]]] = {}
        self._lock = threading.RLock()

    # --- TTL 解析 ---

    def _ttl_for_layer(self, layer: str) -> float:
        """获取指定层的 TTL (秒).

        Args:
            layer: 缓存层名称 ("profile" / "bkt" / "memory")

        Returns:
            该层 TTL; 未知层回退到 profile_ttl
        """
        if layer == "profile":
            return float(self.profile_ttl)
        if layer == "bkt":
            return float(self.bkt_ttl)
        if layer == "memory":
            return float(self.memory_ttl)
        return float(self.profile_ttl)

    # --- 读取 ---

    def get(self, key: str) -> Any | None:
        """从缓存层读取 key (跨层查找, 命中即返回, 过期则清理).

        查找顺序: profile -> bkt -> memory.
        - 命中且未过期: 返回 value
        - 命中但已过期: 清理该条目, 继续检查下一层 (而非直接跳到 backing_store)
        - 全层未命中: 转向 backing_store
        - backing_store 为 None 或无此 key: 返回 None

        Args:
            key: 缓存键

        Returns:
            缓存值, 未命中返回 None
        """
        now = time.time()
        with self._lock:
            for layer_name in _CACHE_LAYERS:
                layer = self._cache.get(layer_name)
                if not layer:
                    continue
                entry = layer.get(key)
                if entry is None:
                    continue
                value, expire_ts = entry
                if now <= expire_ts:
                    # 命中且未过期
                    return value
                # 已过期: 清理该条目, 继续检查下一层 (而非 break 跳过其余层)
                layer.pop(key, None)
                continue
            # 缓存层未命中/过期: 从 backing_store 恢复
            if self.backing_store is not None:
                return self.backing_store.get(key)
            return None

    # --- 写入 (write-through) ---

    def set(self, key: str, value: Any, layer: str = "profile") -> None:
        """写入缓存 (write-through: 同时写入 backing_store).

        Args:
            key: 缓存键
            value: 缓存值
            layer: 缓存层 ("profile" / "bkt" / "memory"), 默认 "profile"
        """
        expire_ts = time.time() + self._ttl_for_layer(layer)
        with self._lock:
            self._cache.setdefault(layer, {})[key] = (value, expire_ts)
            if self.backing_store is not None:
                self.backing_store[key] = value

    # --- 清理 ---

    def clear(self) -> None:
        """清空所有缓存层 (保留 backing_store).

        clear 后, get() 将从 backing_store 恢复数据 (write-through 持久性).
        """
        with self._lock:
            for layer in self._cache.values():
                layer.clear()

    # --- 辅助方法 ---

    def invalidate(self, key: str) -> None:
        """从所有缓存层移除指定 key (不影响 backing_store)."""
        with self._lock:
            for layer in self._cache.values():
                layer.pop(key, None)

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计信息 (各层条目数)."""
        with self._lock:
            return {
                layer_name: len(self._cache.get(layer_name, {}))
                for layer_name in _CACHE_LAYERS
            }


# ============================================================
# __all__
# ============================================================

__all__ = [
    "L2Cache",
    "DEFAULT_PROFILE_TTL",
    "DEFAULT_BKT_TTL",
    "DEFAULT_MEMORY_TTL",
]
