"""L2 memory 子模块测试 — 工作记忆 / 短期记忆 / 长期记忆三层体系.

测试覆盖 (TDD):
1. MemoryChunk: 数据类字段 / 默认值 / to_dict()/from_dict() 往返序列化
2. WorkingMemory:
   - MAX_CHUNKS = 9 (Miller's Law 上限)
   - add_chunk() 超容量时 LRU 淘汰最旧块
   - get_context() / clear() / get_size() / is_full()
3. ShortTermMemory:
   - RETENTION_HOURS = 168 (7 天)
   - add() 自动加 timestamp
   - get_entries() 过滤过期条目 / cleanup() 清理过期 / expire_all() 全部过期
4. LongTermMemory:
   - 依赖注入 (默认 InMemoryL2Store)
   - save/get 画像快照 / 答题历史 / 追踪状态 (委托 L2Store)
"""

import time

import pytest

from dy3_polaris.l2.memory import (
    LongTermMemory,
    MemoryChunk,
    ShortTermMemory,
    WorkingMemory,
)
from dy3_polaris.l2.models import AnswerRecord, LearnerSnapshot, TracingState
from dy3_polaris.l2.store import InMemoryL2Store


# ============================================================
# 1. MemoryChunk 数据类测试
# ============================================================


class TestMemoryChunk:
    """MemoryChunk 数据类测试 — 字段 / 默认值 / 序列化往返."""

    def test_chunk_creation(self):
        """MemoryChunk 全字段创建."""
        chunk = MemoryChunk(
            chunk_id="chunk-001",
            content="二次函数的图像是抛物线",
            chunk_type="knowledge",
            timestamp=1000.0,
            importance=0.9,
        )
        assert chunk.chunk_id == "chunk-001"
        assert chunk.content == "二次函数的图像是抛物线"
        assert chunk.chunk_type == "knowledge"
        assert chunk.timestamp == 1000.0
        assert chunk.importance == 0.9

    def test_chunk_default_importance(self):
        """MemoryChunk 默认 importance=0.5."""
        chunk = MemoryChunk(
            chunk_id="chunk-002",
            content="勾股定理",
            chunk_type="knowledge",
            timestamp=2000.0,
        )
        assert chunk.importance == 0.5

    def test_chunk_to_dict(self):
        """MemoryChunk to_dict() 返回包含全部字段的字典."""
        chunk = MemoryChunk(
            chunk_id="c1",
            content="content-1",
            chunk_type="text",
            timestamp=1.0,
            importance=0.7,
        )
        d = chunk.to_dict()
        assert isinstance(d, dict)
        assert d["chunk_id"] == "c1"
        assert d["content"] == "content-1"
        assert d["chunk_type"] == "text"
        assert d["timestamp"] == 1.0
        assert d["importance"] == 0.7

    def test_chunk_from_dict(self):
        """MemoryChunk from_dict() 反序列化."""
        d = {
            "chunk_id": "c2",
            "content": "content-2",
            "chunk_type": "qa",
            "timestamp": 2.0,
            "importance": 0.3,
        }
        chunk = MemoryChunk.from_dict(d)
        assert chunk.chunk_id == "c2"
        assert chunk.content == "content-2"
        assert chunk.chunk_type == "qa"
        assert chunk.timestamp == 2.0
        assert chunk.importance == 0.3

    def test_chunk_roundtrip(self):
        """MemoryChunk to_dict()/from_dict() 往返一致."""
        original = MemoryChunk(
            chunk_id="chunk-rt",
            content="往返测试内容",
            chunk_type="hint",
            timestamp=9999.0,
            importance=0.88,
        )
        restored = MemoryChunk.from_dict(original.to_dict())
        assert restored.chunk_id == original.chunk_id
        assert restored.content == original.content
        assert restored.chunk_type == original.chunk_type
        assert restored.timestamp == original.timestamp
        assert restored.importance == original.importance

    def test_chunk_roundtrip_preserves_default_importance(self):
        """MemoryChunk 默认 importance 往返后保持 0.5."""
        original = MemoryChunk(
            chunk_id="c3",
            content="默认重要度",
            chunk_type="text",
            timestamp=3.0,
        )
        restored = MemoryChunk.from_dict(original.to_dict())
        assert restored.importance == 0.5


# ============================================================
# 2. WorkingMemory 工作记忆测试
# ============================================================


class TestWorkingMemory:
    """WorkingMemory 工作记忆测试 — Miller 容量 / LRU 淘汰 / 上下文管理."""

    def _make_chunk(self, idx: int) -> MemoryChunk:
        return MemoryChunk(
            chunk_id=f"chunk-{idx}",
            content=f"content-{idx}",
            chunk_type="text",
            timestamp=float(idx),
            importance=0.5,
        )

    # --- 常量 ---

    def test_max_chunks_constant(self):
        """WorkingMemory.MAX_CHUNKS == 9 (Miller's Law 上限)."""
        assert WorkingMemory.MAX_CHUNKS == 9

    # --- 初始状态 ---

    def test_initial_size_is_zero(self):
        """新建工作记忆大小为 0."""
        wm = WorkingMemory()
        assert wm.get_size() == 0

    def test_initial_context_empty(self):
        """新建工作记忆上下文为空列表."""
        wm = WorkingMemory()
        assert wm.get_context() == []

    def test_initial_not_full(self):
        """新建工作记忆未满."""
        wm = WorkingMemory()
        assert wm.is_full() is False

    # --- add_chunk / get_context / get_size ---

    def test_add_chunk_increases_size(self):
        """add_chunk 后 size 增加."""
        wm = WorkingMemory()
        wm.add_chunk(self._make_chunk(1))
        assert wm.get_size() == 1

    def test_get_context_returns_list(self):
        """get_context() 返回列表."""
        wm = WorkingMemory()
        wm.add_chunk(self._make_chunk(1))
        ctx = wm.get_context()
        assert isinstance(ctx, list)
        assert len(ctx) == 1

    def test_get_context_returns_chunks(self):
        """get_context() 返回 MemoryChunk 列表."""
        wm = WorkingMemory()
        wm.add_chunk(self._make_chunk(1))
        wm.add_chunk(self._make_chunk(2))
        ctx = wm.get_context()
        assert all(isinstance(c, MemoryChunk) for c in ctx)
        assert ctx[0].chunk_id == "chunk-1"
        assert ctx[1].chunk_id == "chunk-2"

    def test_get_context_returns_copy(self):
        """get_context() 返回副本, 修改不影响内部状态."""
        wm = WorkingMemory()
        wm.add_chunk(self._make_chunk(1))
        ctx = wm.get_context()
        ctx.clear()
        assert wm.get_size() == 1

    def test_add_chunks_under_capacity(self):
        """添加 9 个块 (未超容量) 全部保留."""
        wm = WorkingMemory()
        for i in range(9):
            wm.add_chunk(self._make_chunk(i))
        assert wm.get_size() == 9
        assert wm.is_full() is True
        ctx = wm.get_context()
        assert ctx[0].chunk_id == "chunk-0"
        assert ctx[8].chunk_id == "chunk-8"

    # --- LRU 淘汰 ---

    def test_add_chunk_lru_eviction(self):
        """超容量时 LRU 淘汰, size 保持 MAX_CHUNKS."""
        wm = WorkingMemory()
        for i in range(10):
            wm.add_chunk(self._make_chunk(i))
        assert wm.get_size() == 9

    def test_lru_evicts_oldest(self):
        """LRU 淘汰最旧块 (第 0 个), 保留最新."""
        wm = WorkingMemory()
        for i in range(10):
            wm.add_chunk(self._make_chunk(i))
        ctx = wm.get_context()
        # chunk-0 (最旧) 被淘汰, chunk-1 成为最旧
        assert ctx[0].chunk_id == "chunk-1"
        # chunk-9 (最新) 在末尾
        assert ctx[-1].chunk_id == "chunk-9"

    def test_lru_eviction_order_preserved(self):
        """连续超容量添加, 列表保持按时间顺序."""
        wm = WorkingMemory()
        for i in range(12):
            wm.add_chunk(self._make_chunk(i))
        assert wm.get_size() == 9
        ctx = wm.get_context()
        # 淘汰 chunk-0..chunk-2, 保留 chunk-3..chunk-11
        assert ctx[0].chunk_id == "chunk-3"
        assert ctx[-1].chunk_id == "chunk-11"

    # --- clear ---

    def test_clear(self):
        """clear() 清空工作记忆."""
        wm = WorkingMemory()
        for i in range(5):
            wm.add_chunk(self._make_chunk(i))
        wm.clear()
        assert wm.get_size() == 0
        assert wm.get_context() == []
        assert wm.is_full() is False

    def test_clear_on_empty(self):
        """对空工作记忆 clear() 不报错."""
        wm = WorkingMemory()
        wm.clear()
        assert wm.get_size() == 0

    # --- is_full ---

    def test_is_full_false_under_capacity(self):
        """未达容量时 is_full() 为 False."""
        wm = WorkingMemory()
        wm.add_chunk(self._make_chunk(1))
        assert wm.is_full() is False

    def test_is_full_true_at_capacity(self):
        """达到容量时 is_full() 为 True."""
        wm = WorkingMemory()
        for i in range(9):
            wm.add_chunk(self._make_chunk(i))
        assert wm.is_full() is True

    # --- 序列化往返 ---

    def test_to_dict(self):
        """WorkingMemory to_dict() 返回含 chunks 的字典."""
        wm = WorkingMemory()
        wm.add_chunk(self._make_chunk(1))
        wm.add_chunk(self._make_chunk(2))
        d = wm.to_dict()
        assert isinstance(d, dict)
        assert "chunks" in d
        assert len(d["chunks"]) == 2

    def test_roundtrip(self):
        """WorkingMemory to_dict()/from_dict() 往返一致."""
        wm = WorkingMemory()
        wm.add_chunk(self._make_chunk(1))
        wm.add_chunk(self._make_chunk(2))
        restored = WorkingMemory.from_dict(wm.to_dict())
        assert restored.get_size() == 2
        ctx = restored.get_context()
        assert ctx[0].chunk_id == "chunk-1"
        assert ctx[1].chunk_id == "chunk-2"

    def test_roundtrip_preserves_chunk_fields(self):
        """WorkingMemory 往返后 MemoryChunk 字段保持."""
        wm = WorkingMemory()
        wm.add_chunk(
            MemoryChunk(
                chunk_id="rt-1",
                content="往返内容",
                chunk_type="hint",
                timestamp=42.0,
                importance=0.9,
            )
        )
        restored = WorkingMemory.from_dict(wm.to_dict())
        chunk = restored.get_context()[0]
        assert chunk.content == "往返内容"
        assert chunk.importance == 0.9
        assert chunk.timestamp == 42.0


# ============================================================
# 3. ShortTermMemory 短期记忆测试
# ============================================================


class TestShortTermMemory:
    """ShortTermMemory 短期记忆测试 — 7 天保留 / 时间衰减 / 清理."""

    # --- 常量 ---

    def test_retention_hours_constant(self):
        """ShortTermMemory.RETENTION_HOURS == 168 (7 天)."""
        assert ShortTermMemory.RETENTION_HOURS == 168

    # --- add / get_entries ---

    def test_add_entry(self):
        """add() 添加条目后 get_entries() 可取回."""
        stm = ShortTermMemory()
        stm.add({"learner_id": "learner-001", "event": "答题"})
        entries = stm.get_entries("learner-001")
        assert len(entries) == 1
        assert entries[0]["event"] == "答题"

    def test_add_entry_auto_timestamp(self):
        """add() 自动为条目添加 timestamp."""
        stm = ShortTermMemory()
        before = time.time()
        stm.add({"learner_id": "learner-001", "event": "答题"})
        after = time.time()
        entries = stm.get_entries("learner-001")
        assert "timestamp" in entries[0]
        assert before <= entries[0]["timestamp"] <= after

    def test_get_entries_missing_learner_returns_empty(self):
        """get_entries() 不存在的 learner 返回空列表."""
        stm = ShortTermMemory()
        assert stm.get_entries("nonexistent") == []

    def test_get_entries_preserves_learner_id(self):
        """get_entries() 返回的条目保留 learner_id."""
        stm = ShortTermMemory()
        stm.add({"learner_id": "learner-001", "event": "答题"})
        entries = stm.get_entries("learner-001")
        assert entries[0]["learner_id"] == "learner-001"

    # --- 过期过滤 ---

    def test_get_entries_filters_expired(self):
        """get_entries() 过滤过期条目 (超 168 小时)."""
        stm = ShortTermMemory()
        now = time.time()
        stm.add({"learner_id": "l1", "event": "old"})
        stm.add({"learner_id": "l1", "event": "new"})
        # 将第一条设为 200 小时前 (已过期)
        stm._entries["l1"][0]["timestamp"] = now - 200 * 3600
        entries = stm.get_entries("l1")
        assert len(entries) == 1
        assert entries[0]["event"] == "new"

    def test_get_entries_keeps_within_retention(self):
        """get_entries() 保留保留窗口内的条目 (100 小时未过期)."""
        stm = ShortTermMemory()
        now = time.time()
        stm.add({"learner_id": "l1", "event": "recent"})
        # 100 小时前 (< 168 小时, 未过期)
        stm._entries["l1"][0]["timestamp"] = now - 100 * 3600
        entries = stm.get_entries("l1")
        assert len(entries) == 1
        assert entries[0]["event"] == "recent"

    # --- cleanup ---

    def test_cleanup_removes_expired(self):
        """cleanup() 清理过期条目并返回清理数量."""
        stm = ShortTermMemory()
        now = time.time()
        stm.add({"learner_id": "l1", "event": "old"})
        stm.add({"learner_id": "l1", "event": "new"})
        stm.add({"learner_id": "l2", "event": "old"})
        stm._entries["l1"][0]["timestamp"] = now - 200 * 3600
        stm._entries["l2"][0]["timestamp"] = now - 200 * 3600
        removed = stm.cleanup()
        assert removed == 2
        assert len(stm.get_entries("l1")) == 1
        assert len(stm.get_entries("l2")) == 0

    def test_cleanup_no_expired_returns_zero(self):
        """cleanup() 无过期条目时返回 0."""
        stm = ShortTermMemory()
        stm.add({"learner_id": "l1", "event": "a"})
        stm.add({"learner_id": "l2", "event": "b"})
        removed = stm.cleanup()
        assert removed == 0

    def test_cleanup_empty_returns_zero(self):
        """cleanup() 空记忆时返回 0."""
        stm = ShortTermMemory()
        removed = stm.cleanup()
        assert removed == 0

    # --- expire_all ---

    def test_expire_all(self):
        """expire_all() 全部过期, get_entries 返回空."""
        stm = ShortTermMemory()
        stm.add({"learner_id": "l1", "event": "a"})
        stm.add({"learner_id": "l2", "event": "b"})
        stm.expire_all()
        assert stm.get_entries("l1") == []
        assert stm.get_entries("l2") == []

    def test_expire_all_empty(self):
        """expire_all() 空记忆不报错."""
        stm = ShortTermMemory()
        stm.expire_all()
        assert stm.get_entries("l1") == []

    # --- 学习者隔离 ---

    def test_learner_isolation(self):
        """不同 learner 的短期记忆相互隔离."""
        stm = ShortTermMemory()
        stm.add({"learner_id": "l1", "event": "a1"})
        stm.add({"learner_id": "l2", "event": "a2"})
        assert len(stm.get_entries("l1")) == 1
        assert len(stm.get_entries("l2")) == 1
        assert stm.get_entries("l1")[0]["event"] == "a1"
        assert stm.get_entries("l2")[0]["event"] == "a2"

    # --- 序列化往返 ---

    def test_to_dict(self):
        """ShortTermMemory to_dict() 返回含 entries 的字典."""
        stm = ShortTermMemory()
        stm.add({"learner_id": "l1", "event": "a"})
        d = stm.to_dict()
        assert isinstance(d, dict)
        assert "entries" in d

    def test_roundtrip(self):
        """ShortTermMemory to_dict()/from_dict() 往返一致."""
        stm = ShortTermMemory()
        stm.add({"learner_id": "l1", "event": "a"})
        stm.add({"learner_id": "l2", "event": "b"})
        restored = ShortTermMemory.from_dict(stm.to_dict())
        assert len(restored.get_entries("l1")) == 1
        assert len(restored.get_entries("l2")) == 1
        assert restored.get_entries("l1")[0]["event"] == "a"


# ============================================================
# 4. LongTermMemory 长期记忆测试
# ============================================================


class TestLongTermMemory:
    """LongTermMemory 长期记忆测试 — 依赖注入 / 委托 L2Store."""

    def _make_snapshot(self, learner_id="learner-001") -> LearnerSnapshot:
        return LearnerSnapshot(
            learner_id=learner_id,
            snapshot_ts=1000.0,
            kp_mastery={"kp-1": 0.8},
            theta=0.7,
            level="intermediate",
        )

    def _make_answer_history(self) -> list:
        return [
            AnswerRecord(
                learner_id="learner-001", kp_id="kp-1", correct=True, timestamp=1.0
            ),
            AnswerRecord(
                learner_id="learner-001", kp_id="kp-1", correct=False, timestamp=2.0
            ),
        ]

    def _make_tracing_state(self, kp_id="kp-001") -> TracingState:
        return TracingState(
            kp_id=kp_id,
            mastery_prob=0.75,
            attempts=3,
            correct_count=2,
            last_attempt_time=1000.0,
            bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1},
        )

    # --- 依赖注入 ---

    def test_default_store(self):
        """无参构造时默认使用 InMemoryL2Store."""
        ltm = LongTermMemory()
        assert isinstance(ltm.store, InMemoryL2Store)

    def test_dependency_injection(self):
        """自定义 store 被注入使用."""
        store = InMemoryL2Store()
        ltm = LongTermMemory(store=store)
        assert ltm.store is store

    def test_accepts_custom_store_subclass(self):
        """接受任意 L2Store 子类."""
        class CustomStore(InMemoryL2Store):
            pass

        custom = CustomStore()
        ltm = LongTermMemory(store=custom)
        assert ltm.store is custom

    # --- 画像快照 ---

    def test_save_and_get_snapshot(self):
        """save_snapshot 后 get_snapshot 取回正确数据."""
        ltm = LongTermMemory()
        snap = self._make_snapshot()
        ltm.save_snapshot("learner-001", snap)
        got = ltm.get_snapshot("learner-001")
        assert got is not None
        assert got.learner_id == "learner-001"
        assert got.theta == 0.7
        assert got.level == "intermediate"

    def test_get_snapshot_missing_returns_none(self):
        """get_snapshot 不存在时返回 None."""
        ltm = LongTermMemory()
        assert ltm.get_snapshot("nonexistent") is None

    def test_save_snapshot_overwrites(self):
        """重复 save_snapshot 覆盖旧数据."""
        ltm = LongTermMemory()
        ltm.save_snapshot("learner-001", self._make_snapshot())
        updated = LearnerSnapshot(
            learner_id="learner-001",
            snapshot_ts=2000.0,
            kp_mastery={"kp-1": 0.95},
            theta=0.9,
            level="advanced",
        )
        ltm.save_snapshot("learner-001", updated)
        got = ltm.get_snapshot("learner-001")
        assert got.level == "advanced"
        assert got.theta == 0.9

    # --- 答题历史 ---

    def test_save_and_get_answer_history(self):
        """save_answer_history 后 get_answer_history 取回正确数据."""
        ltm = LongTermMemory()
        history = self._make_answer_history()
        ltm.save_answer_history("learner-001", history)
        got = ltm.get_answer_history("learner-001")
        assert got is not None
        assert len(got) == 2
        assert got[0].correct is True
        assert got[1].correct is False

    def test_get_answer_history_missing_returns_none(self):
        """get_answer_history 不存在时返回 None."""
        ltm = LongTermMemory()
        assert ltm.get_answer_history("nonexistent") is None

    # --- 追踪状态 ---

    def test_save_and_get_tracing_state(self):
        """save_tracing_state 后 get_tracing_state 取回正确数据."""
        ltm = LongTermMemory()
        state = self._make_tracing_state()
        ltm.save_tracing_state("learner-001", "kp-001", state)
        got = ltm.get_tracing_state("learner-001", "kp-001")
        assert got is not None
        assert got.kp_id == "kp-001"
        assert got.mastery_prob == 0.75
        assert got.attempts == 3

    def test_get_tracing_state_missing_returns_none(self):
        """get_tracing_state 不存在时返回 None."""
        ltm = LongTermMemory()
        assert ltm.get_tracing_state("nonexistent", "kp-x") is None

    def test_tracing_state_different_kp_isolated(self):
        """同一 learner 不同 kp 的 tracing_state 互不干扰."""
        ltm = LongTermMemory()
        ltm.save_tracing_state(
            "learner-001", "kp-1", self._make_tracing_state("kp-1")
        )
        ltm.save_tracing_state(
            "learner-001",
            "kp-2",
            TracingState(
                kp_id="kp-2",
                mastery_prob=0.9,
                bkt_params={"p_l0": 0.5, "p_t": 0.1, "p_g": 0.2, "p_s": 0.1},
            ),
        )
        assert ltm.get_tracing_state("learner-001", "kp-1").mastery_prob == 0.75
        assert ltm.get_tracing_state("learner-001", "kp-2").mastery_prob == 0.9

    # --- 委托验证 ---

    def test_delegates_to_injected_store(self):
        """LongTermMemory 委托注入的 store (store 自身持有数据)."""
        store = InMemoryL2Store()
        ltm = LongTermMemory(store=store)
        snap = self._make_snapshot()
        ltm.save_snapshot("learner-001", snap)
        # store 自身应持有该画像
        assert store.get_profile("learner-001") is snap

    def test_delegates_answer_history_to_store(self):
        """答题历史委托给 store."""
        store = InMemoryL2Store()
        ltm = LongTermMemory(store=store)
        history = self._make_answer_history()
        ltm.save_answer_history("learner-001", history)
        assert store.get_answer_history("learner-001") is not None
        assert len(store.get_answer_history("learner-001")) == 2

    def test_delegates_tracing_state_to_store(self):
        """追踪状态委托给 store."""
        store = InMemoryL2Store()
        ltm = LongTermMemory(store=store)
        state = self._make_tracing_state()
        ltm.save_tracing_state("learner-001", "kp-001", state)
        assert store.get_tracing_state("learner-001", "kp-001") is state

    # --- 序列化 (等价方法) ---

    def test_to_dict_returns_metadata(self):
        """LongTermMemory to_dict() 返回含 store 信息的元数据."""
        ltm = LongTermMemory()
        d = ltm.to_dict()
        assert isinstance(d, dict)
        assert "store_type" in d
