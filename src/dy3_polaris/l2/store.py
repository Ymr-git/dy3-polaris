"""L2 个性化层存储层 — 抽象基类 + 内存实现.

设计依据:
- 参考 L0 cc1/store.py 与 L1 ContextCache 的线程安全存储模式
- L2Store 定义统一存储接口, 便于切换内存 / Redis / 持久化后端
- InMemoryL2Store 提供测试友好的内存实现, threading.RLock 保护所有读写

存储维度:
- profile        : learner_id           -> LearnerSnapshot
- answer_history : learner_id           -> list[AnswerRecord]
- tracing_state  : (learner_id, kp_id)  -> TracingState
- irt_state      : learner_id           -> IRTState
- session        : session_id           -> SessionRecord
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any

from dy3_polaris.l2.models import (
    AnswerRecord,
    IRTState,
    LearnerSnapshot,
    ProfileConflictError,
    SessionRecord,
    TracingState,
)


# ============================================================
# 1. 抽象基类 L2Store
# ============================================================


class L2Store(ABC):
    """L2 个性化层存储抽象基类.

    定义画像 / 答题历史 / 追踪状态 / IRT 状态 / 会话 的统一读写接口.
    具体实现可对接内存 (InMemoryL2Store)、Redis 或关系型数据库.
    """

    # --- 学情画像 ---

    @abstractmethod
    def save_profile(self, learner_id: str, snapshot: LearnerSnapshot) -> None:
        """保存学习者画像快照 (覆盖写)."""
        ...

    @abstractmethod
    def get_profile(self, learner_id: str) -> LearnerSnapshot | None:
        """获取学习者画像快照, 不存在返回 None."""
        ...

    # --- 答题历史 ---

    @abstractmethod
    def save_answer_history(
        self, learner_id: str, records: list[AnswerRecord]
    ) -> None:
        """保存学习者答题历史 (覆盖写)."""
        ...

    @abstractmethod
    def get_answer_history(self, learner_id: str) -> list[AnswerRecord] | None:
        """获取学习者答题历史, 不存在返回 None."""
        ...

    # --- 知识追踪状态 ---

    @abstractmethod
    def save_tracing_state(
        self, learner_id: str, kp_id: str, state: TracingState
    ) -> None:
        """保存单个知识点的 BKT 追踪状态 (按 learner_id + kp_id 覆盖写)."""
        ...

    @abstractmethod
    def get_tracing_state(
        self, learner_id: str, kp_id: str
    ) -> TracingState | None:
        """获取单个知识点的追踪状态, 不存在返回 None."""
        ...

    def get_all_tracing_states(
        self, learner_id: str
    ) -> dict[str, TracingState]:
        """获取学习者所有知识点的追踪状态 {kp_id: TracingState}.

        非抽象默认实现返回空字典 (具体实现可覆盖以提供实际数据).
        供 UpdatePipeline 的遗忘衰减集成遍历学习者全部知识点使用.
        """
        return {}

    # --- IRT 能力状态 ---

    @abstractmethod
    def save_irt_state(self, learner_id: str, state: IRTState) -> None:
        """保存学习者 IRT 能力状态 (覆盖写)."""
        ...

    @abstractmethod
    def get_irt_state(self, learner_id: str) -> IRTState | None:
        """获取学习者 IRT 能力状态, 不存在返回 None."""
        ...

    # --- 会话记录 ---

    @abstractmethod
    def save_session(self, session_id: str, session: SessionRecord) -> None:
        """保存会话记录 (按 session_id 覆盖写).

        Args:
            session_id: 会话 ID (存储键)
            session: 会话记录
        """
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> SessionRecord | None:
        """获取会话记录, 不存在返回 None."""
        ...


# ============================================================
# 2. 内存实现 InMemoryL2Store
# ============================================================


def _merge_snapshots(
    existing: LearnerSnapshot, incoming: LearnerSnapshot
) -> LearnerSnapshot:
    """合并两代画像快照 (并发写保护, 避免丢失更新).

    - kp_mastery: 并集 (incoming 覆盖同 key, existing 其余保留)
    - weak_kps: 基于合并后 kp_mastery 按 0.6 阈值重算
    - extras: 深层字典合并 (各自日志保留)
    - version: max + 1
    """
    import copy

    km = dict(existing.kp_mastery)
    km.update(incoming.kp_mastery)

    ex = dict(existing.extras or {})
    for k, v in (incoming.extras or {}).items():
        if k in ex and isinstance(ex[k], list) and isinstance(v, list):
            # 日志列表合并 (按 repr 去重)
            merged_list = list(ex[k])
            seen = {repr(item) for item in merged_list}
            for item in v:
                key = repr(item)
                if key not in seen:
                    merged_list.append(item)
                    seen.add(key)
            ex[k] = merged_list[-200:]
        else:
            ex[k] = v

    merged = copy.deepcopy(incoming)
    merged.kp_mastery = km
    merged.extras = ex
    merged.weak_kps = sorted(k for k, m in km.items() if m < 0.6)
    merged.version = max(existing.version, incoming.version) + 1
    return merged


class InMemoryL2Store(L2Store):
    """L2Store 的内存实现 (线程安全, threading.RLock 保护).

    适用于单进程测试与开发场景. 所有读写操作均在 RLock 保护下进行,
    支持同一 learner 不同 kp 的追踪状态相互隔离.

    线程安全: threading.RLock 保护全部共享字典.
    """

    def __init__(self, persist_dir: str | os.PathLike | None = None) -> None:
        self._lock = threading.RLock()
        # 学情画像: learner_id -> LearnerSnapshot
        self._profiles: dict[str, LearnerSnapshot] = {}
        # 答题历史: learner_id -> list[AnswerRecord]
        self._answer_history: dict[str, list[AnswerRecord]] = {}
        # 追踪状态: learner_id -> {kp_id -> TracingState}
        self._tracing_states: dict[str, dict[str, TracingState]] = {}
        # IRT 状态: learner_id -> IRTState
        self._irt_states: dict[str, IRTState] = {}
        # 会话记录: session_id -> SessionRecord
        self._sessions: dict[str, SessionRecord] = {}
        # 画像 JSON 持久化目录 (借鉴 dy-agent-system learner JSON 模式)
        self._persist_dir = str(persist_dir) if persist_dir else ""
        if self._persist_dir:
            os.makedirs(self._persist_dir, exist_ok=True)
            self._load_profiles_from_disk()

    # --- 画像 JSON 持久化 ---

    def _profile_path(self, learner_id: str) -> str:
        return os.path.join(self._persist_dir, f"{learner_id}.json")

    def _load_profiles_from_disk(self) -> None:
        """启动时从磁盘恢复画像 (重启不丢失学习数据)."""
        try:
            for name in os.listdir(self._persist_dir):
                if not name.endswith(".json"):
                    continue
                p = os.path.join(self._persist_dir, name)
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                snap = LearnerSnapshot.from_dict(d)
                self._profiles[snap.learner_id] = snap
        except Exception:
            logging.getLogger("dy3_polaris.l2.store").warning(
                "画像持久化加载失败", exc_info=True)
        if self._profiles:
            logging.getLogger("dy3_polaris.l2.store").info(
                "从磁盘恢复 %d 个学习者画像", len(self._profiles))

    def save_profile(
        self,
        learner_id: str,
        snapshot: LearnerSnapshot,
        expected_version: int | None = None,
    ) -> LearnerSnapshot:
        """保存学习者画像快照 (合并写 + 乐观锁).

        合并规则:
        - kp_mastery: 并集 (新快照 key 覆盖, 旧快照其余 key 保留)
        - weak_kps: 基于合并后 kp_mastery 重算
        - extras: 字典合并 (日志追加, 时间戳去重)
        - version: 递增 (乐观并发控制)

        乐观锁:
        - expected_version 非 None 时执行 CAS: 若与存储中版本不匹配则抛
          ProfileConflictError, 调用方需重新拉取最新画像后重试。
        """
        with self._lock:
            existing = self._profiles.get(learner_id)
            if expected_version is not None:
                current = existing.version if existing is not None else 0
                if current != expected_version:
                    raise ProfileConflictError(
                        learner_id, expected_version, current
                    )
            merged = snapshot
            if existing is not None:
                merged = _merge_snapshots(existing, snapshot)
            else:
                merged.version = max(merged.version, 0) + 1
            self._profiles[learner_id] = merged
            snapshot.version = merged.version
        if self._persist_dir:
            try:
                with open(self._profile_path(learner_id), "w", encoding="utf-8") as f:
                    json.dump(merged.to_dict(), f, ensure_ascii=False, indent=2)
            except Exception:
                logging.getLogger("dy3_polaris.l2.store").warning(
                    "画像持久化写入失败: %s", learner_id, exc_info=True)
        return merged

    def get_profile(self, learner_id: str) -> LearnerSnapshot | None:
        """获取学习者画像快照, 不存在返回 None."""
        with self._lock:
            snap = self._profiles.get(learner_id)
            if snap is None and self._persist_dir:
                try:
                    with open(self._profile_path(learner_id), encoding="utf-8") as f:
                        d = json.load(f)
                    snap = LearnerSnapshot.from_dict(d)
                    self._profiles[learner_id] = snap
                except FileNotFoundError:
                    return None
                except Exception:
                    logging.getLogger("dy3_polaris.l2.store").warning(
                        "画像持久化读取失败: %s", learner_id, exc_info=True)
            return snap

    # --- 答题历史 ---

    def save_answer_history(
        self, learner_id: str, records: list[AnswerRecord]
    ) -> None:
        """保存学习者答题历史 (覆盖写, 浅拷贝列表避免外部修改影响)."""
        with self._lock:
            self._answer_history[learner_id] = list(records)

    def get_answer_history(self, learner_id: str) -> list[AnswerRecord] | None:
        """获取学习者答题历史, 不存在返回 None.

        返回防御性副本 (新列表), 修改返回值不影响内部状态.
        """
        with self._lock:
            history = self._answer_history.get(learner_id)
            if history is None:
                return None
            return list(history)  # defensive copy

    # --- 知识追踪状态 ---

    def save_tracing_state(
        self, learner_id: str, kp_id: str, state: TracingState
    ) -> None:
        """保存单个知识点的 BKT 追踪状态 (按 learner_id + kp_id 覆盖写)."""
        with self._lock:
            self._tracing_states.setdefault(learner_id, {})[kp_id] = state

    def get_tracing_state(
        self, learner_id: str, kp_id: str
    ) -> TracingState | None:
        """获取单个知识点的追踪状态, 不存在返回 None.

        注意: 返回的 TracingState 为内部存储对象的引用. 调用方不应直接修改
        返回对象 (如需修改请构造新对象后调用 save_tracing_store 持久化).
        """
        with self._lock:
            return self._tracing_states.get(learner_id, {}).get(kp_id)

    def get_all_tracing_states(
        self, learner_id: str
    ) -> dict[str, TracingState]:
        """获取学习者所有知识点的追踪状态 {kp_id: TracingState}.

        返回字典的浅拷贝 (新 dict), 但 value 仍为内部 TracingState 引用;
        调用方不应直接修改返回对象. 不存在该学习者时返回空字典.

        供 UpdatePipeline 的遗忘衰减集成遍历学习者全部知识点使用.
        """
        with self._lock:
            return dict(self._tracing_states.get(learner_id, {}))

    # --- IRT 能力状态 ---

    def save_irt_state(self, learner_id: str, state: IRTState) -> None:
        """保存学习者 IRT 能力状态 (覆盖写)."""
        with self._lock:
            self._irt_states[learner_id] = state

    def get_irt_state(self, learner_id: str) -> IRTState | None:
        """获取学习者 IRT 能力状态, 不存在返回 None."""
        with self._lock:
            return self._irt_states.get(learner_id)

    # --- 会话记录 ---

    def save_session(self, session_id: str, session: SessionRecord) -> None:
        """保存会话记录 (按 session_id 覆盖写).

        Args:
            session_id: 会话 ID (存储键)
            session: 会话记录
        """
        with self._lock:
            self._sessions[session_id] = session

    def get_session(self, session_id: str) -> SessionRecord | None:
        """获取会话记录, 不存在返回 None."""
        with self._lock:
            return self._sessions.get(session_id)

    # --- 辅助方法 ---

    def get_stats(self) -> dict[str, Any]:
        """获取存储统计信息 (用于监控/调试)."""
        with self._lock:
            return {
                "profiles": len(self._profiles),
                "answer_histories": len(self._answer_history),
                "tracing_states": sum(
                    len(kps) for kps in self._tracing_states.values()
                ),
                "irt_states": len(self._irt_states),
                "sessions": len(self._sessions),
            }

    def clear(self) -> None:
        """清空所有内存数据."""
        with self._lock:
            self._profiles.clear()
            self._answer_history.clear()
            self._tracing_states.clear()
            self._irt_states.clear()
            self._sessions.clear()


# ============================================================
# __all__
# ============================================================

__all__ = [
    "L2Store",
    "InMemoryL2Store",
]
