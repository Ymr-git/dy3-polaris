"""长期记忆 — 委托 L2Store 的持久化记忆层.

融合世界先进方案:
- Atkinson-Shiffrin 模型: 长期记忆持久存储
- Redis Agent Memory: 两层记忆模型 (Session + Long-term)
- Claude Science: Persistent Kernels + Session Fork

设计:
- 委托 L2Store 进行持久化 (依赖注入)
- 存储维度: 画像快照 / 答题历史 / 追踪状态 / IRT 状态
- 提供检索接口供 ProfileBuilder 调用

本类为 ``L2Store`` 的语义化门面 (facade): 不自己维护状态, 全部读写委托给
注入的 store. 通过依赖注入可灵活切换内存 (``InMemoryL2Store``) / Redis /
关系型数据库后端.
"""

from __future__ import annotations

from typing import Any

from dy3_polaris.l2.models import AnswerRecord, LearnerSnapshot, TracingState
from dy3_polaris.l2.store import InMemoryL2Store, L2Store


# ============================================================
# 1. LongTermMemory 长期记忆类
# ============================================================


class LongTermMemory:
    """长期记忆 — 委托 L2Store 的持久化记忆层 (依赖注入).

    作为 ``L2Store`` 的语义化门面, 将画像 / 答题历史 / 追踪状态的读写
    映射到 store 接口, 自身不持有可变状态, 保证多实例共享同一 store 时
    数据一致.

    Args:
        store: L2Store 实现 (依赖注入); 为 None 时默认使用 ``InMemoryL2Store``.

    Attributes:
        store: 注入的持久化后端 (只读属性).
    """

    def __init__(self, store: L2Store | None = None) -> None:
        """初始化长期记忆.

        Args:
            store: L2Store 实现; 为 None 时默认 ``InMemoryL2Store``.
        """
        self._store: L2Store = store if store is not None else InMemoryL2Store()

    @property
    def store(self) -> L2Store:
        """返回注入的持久化后端."""
        return self._store

    # --- 画像快照 ---

    def save_snapshot(self, learner_id: str, snapshot: LearnerSnapshot) -> None:
        """保存学习者画像快照 (覆盖写).

        Args:
            learner_id: 学习者 ID
            snapshot: 画像快照
        """
        self._store.save_profile(learner_id, snapshot)

    def get_snapshot(self, learner_id: str) -> LearnerSnapshot | None:
        """获取学习者最新画像快照, 不存在返回 None.

        Args:
            learner_id: 学习者 ID

        Returns:
            画像快照, 不存在返回 None
        """
        return self._store.get_profile(learner_id)

    # --- 答题历史 ---

    def save_answer_history(
        self, learner_id: str, records: list[AnswerRecord]
    ) -> None:
        """保存学习者答题历史 (覆盖写).

        Args:
            learner_id: 学习者 ID
            records: 答题记录列表
        """
        self._store.save_answer_history(learner_id, records)

    def get_answer_history(self, learner_id: str) -> list[AnswerRecord] | None:
        """获取学习者答题历史, 不存在返回 None.

        Args:
            learner_id: 学习者 ID

        Returns:
            答题记录列表, 不存在返回 None
        """
        return self._store.get_answer_history(learner_id)

    # --- 知识追踪状态 ---

    def save_tracing_state(
        self, learner_id: str, kp_id: str, state: TracingState
    ) -> None:
        """保存单个知识点的 BKT 追踪状态 (按 learner_id + kp_id 覆盖写).

        Args:
            learner_id: 学习者 ID
            kp_id: 知识点 ID
            state: 追踪状态
        """
        self._store.save_tracing_state(learner_id, kp_id, state)

    def get_tracing_state(
        self, learner_id: str, kp_id: str
    ) -> TracingState | None:
        """获取单个知识点的追踪状态, 不存在返回 None.

        Args:
            learner_id: 学习者 ID
            kp_id: 知识点 ID

        Returns:
            追踪状态, 不存在返回 None
        """
        return self._store.get_tracing_state(learner_id, kp_id)

    # --- 序列化 (等价方法) ---

    def to_dict(self) -> dict[str, Any]:
        """返回长期记忆元数据 (持久化由 store 委托, 不内联快照).

        本类为 store 门面, 自身无可变状态; ``to_dict`` 返回 store 类型与统计
        信息作为等价的可观测接口 (若 store 提供 ``get_stats``).

        Returns:
            ``{"store_type": ..., "store_stats": ...}`` 格式字典
        """
        stats_fn = getattr(self._store, "get_stats", None)
        stats = stats_fn() if callable(stats_fn) else {}
        return {
            "store_type": type(self._store).__name__,
            "store_stats": stats,
        }


# ============================================================
# __all__
# ============================================================

__all__ = [
    "LongTermMemory",
]
