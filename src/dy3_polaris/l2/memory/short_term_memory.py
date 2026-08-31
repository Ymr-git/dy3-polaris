"""短期记忆 — 7天保留窗口的时间衰减记忆.

融合世界先进方案:
- Atkinson-Shiffrin 模型: 短期记忆 → 长期记忆固化
- Ebbinghaus 遗忘曲线: 时间衰减
- Jupyter Kernel: 会话结束归档至短期记忆

设计:
- RETENTION_HOURS = 168 (7天)
- 超期条目自动清理
- 按 learner_id 隔离

存储结构: ``dict[learner_id, list[entry]]``, 每个 entry 为字典并含 ``timestamp``
(由 ``add()`` 自动写入). 条目按 ``timestamp`` 判定是否超 ``RETENTION_HOURS``
保留窗口, 超期条目在 ``get_entries()`` 中被过滤, 在 ``cleanup()`` 中被移除.
"""

from __future__ import annotations

import threading
import time
from typing import Any


# ============================================================
# 1. 常量定义
# ============================================================

# 短期记忆保留窗口 (小时): 7 天
RETENTION_HOURS: float = 168.0

# 秒 -> 小时换算系数
_SECONDS_PER_HOUR: float = 3600.0


# ============================================================
# 2. ShortTermMemory 短期记忆类
# ============================================================


class ShortTermMemory:
    """短期记忆 — 7 天保留窗口的时间衰减记忆.

    按 ``learner_id`` 隔离存储条目列表, 每个条目为字典并自动附加 ``timestamp``.
    条目超 ``RETENTION_HOURS`` (168 小时 = 7 天) 视为过期:
    - ``get_entries()`` 过滤过期条目, 仅返回保留窗口内的条目.
    - ``cleanup()`` 物理移除全部过期条目并返回清理数量.
    - ``expire_all()`` 清空全部条目 (强制全部过期).

    Attributes:
        RETENTION_HOURS: 保留窗口 (类常量, 默认 168 小时 = 7 天).
    """

    # 保留窗口 (小时): 7 天
    RETENTION_HOURS: float = RETENTION_HOURS

    def __init__(self) -> None:
        """初始化空短期记忆."""
        # learner_id -> list[entry(dict)]
        self._entries: dict[str, list[dict[str, Any]]] = {}
        # 保护 _entries 读写 (线程安全), 避免 add/cleanup/expire_all 并发竞态
        self._lock = threading.RLock()

    # --- 内部辅助 ---

    def _is_expired(self, entry: dict[str, Any], now: float) -> bool:
        """判定单个条目是否过期.

        Args:
            entry: 条目字典 (含 timestamp)
            now: 当前时间戳 (秒)

        Returns:
            距上次写入时间超过 RETENTION_HOURS 返回 True, 否则 False
        """
        ts = entry.get("timestamp", now)
        age_hours = (now - ts) / _SECONDS_PER_HOUR
        return age_hours > self.RETENTION_HOURS

    # --- 写入 ---

    def add(self, entry: dict[str, Any]) -> None:
        """添加条目到短期记忆 (自动附加 timestamp).

        ``entry`` 必须包含 ``learner_id`` 字段以定位存储桶. 本方法对入参做浅拷贝
        后写入, 避免外部修改影响内部状态, 并自动写入当前时间戳.

        Args:
            entry: 条目字典, 须含 ``learner_id`` 字段

        Raises:
            KeyError: entry 缺少 ``learner_id`` 字段
        """
        stored = dict(entry)
        stored["timestamp"] = time.time()
        learner_id = stored["learner_id"]
        with self._lock:
            self._entries.setdefault(learner_id, []).append(stored)

    # --- 读取 ---

    def get_entries(self, learner_id: str) -> list[dict[str, Any]]:
        """获取指定学习者的未过期条目.

        仅返回保留窗口内 (距写入时间 <= RETENTION_HOURS) 的条目,
        不存在的 learner 返回空列表. 返回值为新列表 (浅拷贝), 修改不影响内部状态.

        Args:
            learner_id: 学习者 ID

        Returns:
            未过期条目列表 (按写入顺序)
        """
        now = time.time()
        with self._lock:
            bucket = self._entries.get(learner_id)
            if not bucket:
                return []
            return [e for e in bucket if not self._is_expired(e, now)]

    # --- 清理 ---

    def cleanup(self) -> int:
        """清理全部过期条目, 返回清理数量.

        遍历所有学习者的条目列表, 物理移除过期条目; 若某学习者条目被清空,
        则移除该学习者的存储桶键.

        Returns:
            被清理的过期条目总数
        """
        now = time.time()
        removed = 0
        empty_learners: list[str] = []
        for learner_id, bucket in self._entries.items():
            kept: list[dict[str, Any]] = []
            for entry in bucket:
                if self._is_expired(entry, now):
                    removed += 1
                else:
                    kept.append(entry)
            self._entries[learner_id] = kept
            if not kept:
                empty_learners.append(learner_id)
        # 移除空存储桶, 保持存储整洁
        for learner_id in empty_learners:
            del self._entries[learner_id]
        return removed

    def expire_all(self) -> None:
        """全部过期 — 清空全部学习者的全部条目."""
        self._entries.clear()

    # --- 序列化 ---

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (含全部学习者的条目).

        Returns:
            ``{"entries": {learner_id: [entry, ...], ...}}`` 格式字典
        """
        return {
            "entries": {
                lid: [dict(e) for e in bucket]
                for lid, bucket in self._entries.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShortTermMemory:
        """从字典反序列化.

        Args:
            d: ``to_dict()`` 产生的字典

        Returns:
            重建的 ShortTermMemory 实例
        """
        stm = cls()
        for lid, bucket in d.get("entries", {}).items():
            stm._entries[lid] = [dict(e) for e in bucket]
        return stm


# ============================================================
# __all__
# ============================================================

__all__ = [
    "RETENTION_HOURS",
    "ShortTermMemory",
]
