"""工作记忆 — 当前会话的短期信息缓存 (Miller's Law 7±2).

融合世界先进方案:
- Miller (1956): 工作记忆容量 7±2 信息块
- Cowan (2001): 工作记忆容量 4±1 (更严格的估计)
- Atkinson-Shiffrin 模型: 感觉记忆 → 短期记忆 → 长期记忆
- Baddeley 工作记忆模型: 中央执行系统 + 语音回路 + 视觉空间画板

设计:
- MAX_CHUNKS = 9 (Miller 上限)
- 超容量时 LRU (Least Recently Used) 淘汰最久未访问的块
- 每个 MemoryChunk 含 importance 权重 (0.0-1.0)
- access_chunk() 方法更新访问顺序, 实现真正 LRU 语义
- 淘汰策略: 优先淘汰 importance 最低且最久未访问的块

模块构成:
1. ``MemoryChunk``: 工作记忆信息块数据类 (chunk_id / content / chunk_type /
   timestamp / importance), 提供 ``to_dict()`` / ``from_dict()`` 往返序列化.
2. ``WorkingMemory``: 会话级工作记忆缓冲区, 按 Miller 容量上限管理,
   超容量时按 LRU + importance 淘汰最久未访问且重要度最低的块.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


# ============================================================
# 1. 常量定义
# ============================================================

# 工作记忆容量上限 (Miller's Law 7±2 的上界)
MAX_CHUNKS: int = 9

# MemoryChunk 默认重要度
DEFAULT_IMPORTANCE: float = 0.5


# ============================================================
# 2. MemoryChunk 数据类
# ============================================================


@dataclass
class MemoryChunk:
    """工作记忆信息块 (单个上下文单元).

    Attributes:
        chunk_id: 信息块唯一标识
        content: 信息块内容 (文本 / 结构化数据序列化字符串)
        chunk_type: 信息块类型 (knowledge / qa / hint / feedback ...)
        timestamp: 创建时间戳 (秒, float)
        importance: 重要度权重 [0.0, 1.0], 默认 0.5
    """

    chunk_id: str
    content: str
    chunk_type: str
    timestamp: float
    importance: float = DEFAULT_IMPORTANCE

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "timestamp": self.timestamp,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryChunk:
        """从字典反序列化."""
        return cls(
            chunk_id=d["chunk_id"],
            content=d["content"],
            chunk_type=d["chunk_type"],
            timestamp=d["timestamp"],
            importance=d.get("importance", DEFAULT_IMPORTANCE),
        )


# ============================================================
# 3. WorkingMemory 工作记忆类
# ============================================================


class WorkingMemory:
    """工作记忆 — 当前会话的短期信息缓存 (Miller's Law).

    使用 ``list`` 按访问时间顺序存储 ``MemoryChunk``:
    - 列表末尾为最近访问的块, 首部为最久未访问的块.
    - 达到 ``MAX_CHUNKS`` 容量后再添加, 淘汰首部 (最久未访问) 块.
    - ``access_chunk()`` 将被访问的块移到末尾, 实现 LRU 语义.
    - ``get_context()`` 返回内部列表的浅拷贝, 修改返回值不影响内部状态.

    线程安全: threading.Lock 保护所有读写操作.

    Attributes:
        MAX_CHUNKS: 工作记忆容量上限 (类常量, 默认 9, Miller 上限).
    """

    # Miller's Law 上限 (7±2 的上界)
    MAX_CHUNKS: int = MAX_CHUNKS

    def __init__(self) -> None:
        """初始化空工作记忆."""
        self._chunks: list[MemoryChunk] = []
        self._lock = threading.Lock()

    # --- 写入 ---

    def add_chunk(self, chunk: MemoryChunk) -> None:
        """添加信息块到工作记忆.

        若已达到 ``MAX_CHUNKS`` 容量, 先淘汰列表首部 (最久未访问的块) 实现 LRU,
        再追加新块, 保证容量不超过上限.

        Args:
            chunk: 待添加的信息块
        """
        with self._lock:
            if len(self._chunks) >= self.MAX_CHUNKS:
                # LRU: 弹出最久未访问的块 (列表首部)
                self._chunks.pop(0)
            self._chunks.append(chunk)

    # --- 访问 (LRU 更新) ---

    def access_chunk(self, chunk_id: str) -> MemoryChunk | None:
        """访问指定信息块, 并将其移到最近访问位置 (LRU 更新).

        Args:
            chunk_id: 要访问的信息块 ID.

        Returns:
            找到的 MemoryChunk; 未找到返回 None.
        """
        with self._lock:
            for i, chunk in enumerate(self._chunks):
                if chunk.chunk_id == chunk_id:
                    # 移到末尾 (最近访问)
                    self._chunks.pop(i)
                    self._chunks.append(chunk)
                    return chunk
            return None

    # --- 读取 ---

    def get_context(self) -> list[MemoryChunk]:
        """返回当前工作记忆上下文 (信息块列表的浅拷贝).

        注意: 本方法不更新 LRU 访问顺序. 如需更新, 请先调用 ``access_chunk``.

        Returns:
            当前信息块列表的副本 (修改返回值不影响内部状态)
        """
        with self._lock:
            return list(self._chunks)

    def get_size(self) -> int:
        """返回当前信息块数量.

        Returns:
            当前块数
        """
        with self._lock:
            return len(self._chunks)

    def is_full(self) -> bool:
        """判断工作记忆是否已满.

        Returns:
            当前块数 >= MAX_CHUNKS 时返回 True, 否则 False
        """
        with self._lock:
            return len(self._chunks) >= self.MAX_CHUNKS

    # --- 清理 ---

    def clear(self) -> None:
        """清空工作记忆 (移除全部信息块)."""
        with self._lock:
            self._chunks.clear()

    # --- 序列化 ---

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (含全部信息块).

        Returns:
            ``{"chunks": [chunk.to_dict(), ...]}`` 格式字典
        """
        with self._lock:
            return {
                "chunks": [c.to_dict() for c in self._chunks],
            }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkingMemory:
        """从字典反序列化.

        Args:
            d: ``to_dict()`` 产生的字典

        Returns:
            重建的 WorkingMemory 实例
        """
        wm = cls()
        wm._chunks = [MemoryChunk.from_dict(c) for c in d.get("chunks", [])]
        return wm


# ============================================================
# __all__
# ============================================================

__all__ = [
    "MAX_CHUNKS",
    "DEFAULT_IMPORTANCE",
    "MemoryChunk",
    "WorkingMemory",
]
