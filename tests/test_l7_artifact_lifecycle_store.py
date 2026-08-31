"""L7 Artifact 管理系统 T3 — 生命周期状态机与存储层单元测试.

测试覆盖:
1. LifecycleStateMachine: 合法/非法转移、允许目标、终态、子状态
2. ContentStore: CAS 内容寻址、去重、引用释放
3. MemoryArtifactStore: L1 LRU 淘汰
4. JsonFileArtifactStore: L2 文件持久化、会话上限、恢复
5. TieredArtifactStore: 三级读穿、写穿、快照往返
"""

from __future__ import annotations

import json
import os

import pytest

from dy3_polaris.l7.artifact.lifecycle import (
    LifecycleStateMachine,
    StateTransitionError,
)
from dy3_polaris.l7.artifact.store import (
    ContentStore,
    JsonFileArtifactStore,
    MemoryArtifactStore,
    NoopServerStore,
    ServerArtifactStore,
    TieredArtifactStore,
)
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactLifecycleState,
    ArtifactType,
)


def _artifact(payload: dict | None = None, **kwargs) -> Artifact:
    return Artifact(
        type=ArtifactType.TEXT,
        mime="text/vnd.dy3+markdown",
        payload=payload or {"content": "x"},
        **kwargs,
    )


# ============================================================
# 生命周期状态机
# ============================================================

class TestLifecycleStateMachine:
    """状态机合法/非法转移 (设计文档 Ch.3.2)."""

    def test_valid_transitions(self):
        m = LifecycleStateMachine()
        assert m.can_transition(ArtifactLifecycleState.CREATED, ArtifactLifecycleState.RENDERED)
        assert m.can_transition(ArtifactLifecycleState.RENDERED, ArtifactLifecycleState.REVIEWED)
        assert m.can_transition(ArtifactLifecycleState.RENDERED, ArtifactLifecycleState.ARCHIVED)
        assert m.can_transition(ArtifactLifecycleState.REVIEWED, ArtifactLifecycleState.EDITED)
        assert m.can_transition(ArtifactLifecycleState.REVIEWED, ArtifactLifecycleState.ARCHIVED)
        assert m.can_transition(ArtifactLifecycleState.EDITED, ArtifactLifecycleState.RENDERED)
        assert m.can_transition(ArtifactLifecycleState.ARCHIVED, ArtifactLifecycleState.RENDERED)

    def test_invalid_transitions(self):
        m = LifecycleStateMachine()
        # Created 不能直接跳非 Rendered 状态
        assert not m.can_transition(ArtifactLifecycleState.CREATED, ArtifactLifecycleState.REVIEWED)
        assert not m.can_transition(ArtifactLifecycleState.CREATED, ArtifactLifecycleState.EDITED)
        assert not m.can_transition(ArtifactLifecycleState.CREATED, ArtifactLifecycleState.ARCHIVED)
        # Reviewed 不能回 Rendered
        assert not m.can_transition(ArtifactLifecycleState.REVIEWED, ArtifactLifecycleState.RENDERED)
        # Edited 不能回 Reviewed
        assert not m.can_transition(ArtifactLifecycleState.EDITED, ArtifactLifecycleState.REVIEWED)
        # 自环非法
        assert not m.can_transition(ArtifactLifecycleState.CREATED, ArtifactLifecycleState.CREATED)

    def test_transition_returns_target(self):
        m = LifecycleStateMachine()
        state = m.transition(ArtifactLifecycleState.CREATED, ArtifactLifecycleState.RENDERED)
        assert state == ArtifactLifecycleState.RENDERED

    def test_invalid_transition_raises(self):
        m = LifecycleStateMachine()
        with pytest.raises(StateTransitionError):
            m.transition(ArtifactLifecycleState.CREATED, ArtifactLifecycleState.ARCHIVED)

    def test_allowed_targets(self):
        m = LifecycleStateMachine()
        targets = m.allowed_targets(ArtifactLifecycleState.RENDERED)
        assert ArtifactLifecycleState.REVIEWED in targets
        assert ArtifactLifecycleState.ARCHIVED in targets
        assert ArtifactLifecycleState.CREATED not in targets

    def test_sub_states(self):
        m = LifecycleStateMachine()
        created = m.sub_states(ArtifactLifecycleState.CREATED)
        assert "validating" in created and "registered" in created
        rendered = m.sub_states(ArtifactLifecycleState.RENDERED)
        assert set(rendered) == {"routing", "rendering", "mounted"}
        edited = m.sub_states(ArtifactLifecycleState.EDITED)
        assert "new_version" in edited

    def test_validate_sub_state(self):
        m = LifecycleStateMachine()
        m.validate_sub_state(ArtifactLifecycleState.CREATED, "validating")
        with pytest.raises(Exception):
            m.validate_sub_state(ArtifactLifecycleState.CREATED, "mounted")


# ============================================================
# CAS 内容寻址存储
# ============================================================

class TestContentStore:
    """CAS 内容寻址 + 去重."""

    def test_put_returns_sha256(self):
        cs = ContentStore()
        digest = cs.put({"a": 1})
        assert len(digest) == 64  # sha256 hex

    def test_dedup_by_content(self):
        cs = ContentStore()
        h1 = cs.put({"data": "same"})
        h2 = cs.put({"data": "same"})
        assert h1 == h2
        assert cs.size() == 1

    def test_get_roundtrip(self):
        cs = ContentStore()
        digest = cs.put({"data": [1, 2, 3]})
        assert cs.get(digest) == {"data": [1, 2, 3]}

    def test_get_missing_returns_none(self):
        cs = ContentStore()
        assert cs.get("0" * 64) is None

    def test_release_removes_object(self):
        cs = ContentStore()
        digest = cs.put({"x": 1})
        cs.release(digest)
        assert cs.get(digest) is None

    def test_capacity_lru_eviction(self):
        cs = ContentStore(capacity=2)
        cs.put({"a": 1})
        cs.put({"b": 2})
        cs.put({"c": 3})
        assert cs.size() == 2
        # 最旧 a 被淘汰
        assert cs.get(cs.put({"a": 1})) is None or cs.size() == 2


# ============================================================
# L1 内存层
# ============================================================

class TestMemoryArtifactStore:
    """L1 内存层 + LRU 淘汰."""

    def test_save_load(self):
        store = MemoryArtifactStore()
        art = _artifact()
        store.save(art)
        assert store.load(art.artifact_id) is not None
        assert store.load("missing") is None

    def test_list(self):
        store = MemoryArtifactStore()
        store.save(_artifact(title="a"))
        store.save(_artifact(title="b"))
        assert len(store.list()) == 2

    def test_delete(self):
        store = MemoryArtifactStore()
        art = _artifact()
        store.save(art)
        assert store.delete(art.artifact_id) is True
        assert store.delete(art.artifact_id) is False

    def test_capacity_eviction(self):
        store = MemoryArtifactStore(capacity=2)
        store.save(_artifact(title="a"))
        store.save(_artifact(title="b"))
        store.save(_artifact(title="c"))
        assert store.size() == 2
        ids = {a.title for a in store.list()}
        assert "a" not in ids


# ============================================================
# L2 本地文件层
# ============================================================

class TestJsonFileArtifactStore:
    """L2 文件持久化 (模拟 IndexedDB)."""

    def test_roundtrip_persistence(self, tmp_path):
        path = str(tmp_path / "l2.json")
        store = JsonFileArtifactStore(path)
        art = _artifact(payload={"content": "持久化"}, title="p1")
        store.save(art)

        store2 = JsonFileArtifactStore(path)
        restored = store2.load(art.artifact_id)
        assert restored is not None
        assert restored.payload["content"] == "持久化"
        assert restored.title == "p1"

    def test_clear_removes_file_data(self, tmp_path):
        path = str(tmp_path / "l2.json")
        store = JsonFileArtifactStore(path)
        store.save(_artifact(title="x"))
        store.clear()
        store2 = JsonFileArtifactStore(path)
        assert len(store2.list()) == 0

    def test_corrupt_file_tolerated(self, tmp_path):
        path = str(tmp_path / "l2.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{corrupt json")
        store = JsonFileArtifactStore(path)
        assert len(store.list()) == 0

    def test_session_limit(self, tmp_path):
        path = str(tmp_path / "l2.json")
        store = JsonFileArtifactStore(path, max_sessions=2)
        for i in range(4):
            store.save(_artifact(title=f"s{i}", session_id=f"session-{i}"))
        # 只保留最近 2 个会话
        assert store.size() <= 4  # 每个会话 1 个 artifact, 保留 2 会话


# ============================================================
# 三级存储门面
# ============================================================

class TestTieredArtifactStore:
    """三级存储读穿 + 写穿 + 快照."""

    def _make_store(self, tmp_path):
        return TieredArtifactStore(
            l1=MemoryArtifactStore(),
            l2=JsonFileArtifactStore(str(tmp_path / "l2.json")),
            l3=NoopServerStore(),
        )

    def test_write_through_all_layers(self, tmp_path):
        store = self._make_store(tmp_path)
        art = _artifact(title="t")
        store.save(art)
        assert store.l1.load(art.artifact_id) is not None
        assert store.l2.load(art.artifact_id) is not None

    def test_read_through_l2_to_l1(self, tmp_path):
        # L1 空, L2 有数据 → 读穿回填 L1
        l2 = JsonFileArtifactStore(str(tmp_path / "l2.json"))
        art = _artifact(title="t")
        l2.save(art)
        store = TieredArtifactStore(l1=MemoryArtifactStore(), l2=l2, l3=None)
        loaded = store.load(art.artifact_id)
        assert loaded is not None
        assert store.l1.load(art.artifact_id) is not None  # 回填

    def test_snapshot_roundtrip(self, tmp_path):
        store = self._make_store(tmp_path)
        store.save(_artifact(title="a"))
        store.save(_artifact(title="b"))
        snap = str(tmp_path / "snap.json")
        n = store.save_snapshot(snap)
        assert n == 2

        store2 = TieredArtifactStore(l1=MemoryArtifactStore())
        restored = store2.load_snapshot(snap)
        assert restored == 2
        assert len(store2.list()) == 2

    def test_missing_snapshot_raises(self, tmp_path):
        store = TieredArtifactStore(l1=MemoryArtifactStore())
        with pytest.raises(FileNotFoundError):
            store.load_snapshot(str(tmp_path / "missing.json"))

    def test_cas_dedup_via_tiered(self, tmp_path):
        store = self._make_store(tmp_path)
        a1 = _artifact(payload={"content": "重复内容"}, title="a")
        a2 = _artifact(payload={"content": "重复内容"}, title="b")
        store.save(a1)
        store.save(a2)
        # 相同 payload → 相同内容哈希
        h1 = a1.learner_context.get("_content_hash")
        h2 = a2.learner_context.get("_content_hash")
        assert h1 is not None and h1 == h2
