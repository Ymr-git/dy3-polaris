"""L7 Artifact 管理系统 T3 — ArtifactManager 增强集成测试.

测试覆盖:
1. review / unarchive / remove (软删/硬删)
2. merge (分支合并, 三种策略)
3. 归档状态防护 (update/apply_edit/fork 拒绝)
4. 生命周期事件发射 (registered/reviewed/archived/removed)
5. 快照持久化往返 (save_snapshot/load_snapshot)
6. fulltext_search / related_by_kp 集成
7. 注册幂等性
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.artifact.lifecycle import LifecycleStateMachine
from dy3_polaris.l7.artifact_manager import ArtifactManager
from dy3_polaris.l7.events import (
    ARTIFACT_ARCHIVED,
    ARTIFACT_MERGED,
    ARTIFACT_REGISTERED,
    ARTIFACT_REMOVED,
    ARTIFACT_RESTORED,
    ARTIFACT_REVIEWED,
    get_global_emitter,
    reset_global_emitter,
)
from dy3_polaris.l7.exceptions import ArtifactNotFoundError
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactDiff,
    ArtifactLifecycleState,
    ArtifactType,
)


def _art(payload: dict | None = None, **kwargs) -> Artifact:
    mime = kwargs.pop("mime", "text/vnd.dy3+markdown")
    return Artifact(
        type=ArtifactType.CHART if "chart" in mime else ArtifactType.TEXT,
        mime=mime,
        payload=payload or {"content": "x"},
        **kwargs,
    )


class _EventRecorder:
    """事件捕获器."""

    def __init__(self):
        self.events: list[str] = []
        self._types = [
            ARTIFACT_REGISTERED, ARTIFACT_REVIEWED, ARTIFACT_ARCHIVED,
            ARTIFACT_RESTORED, ARTIFACT_REMOVED, ARTIFACT_MERGED,
        ]

    def attach(self):
        for t in self._types:
            get_global_emitter().on(t, self._capture)

    def _capture(self, event):
        self.events.append(event.event_type)

    def detach(self):
        for t in self._types:
            get_global_emitter().off(t, self._capture)


class TestReviewUnarchive:
    """审核与恢复."""

    @pytest.fixture(autouse=True)
    def _clean_emitter(self):
        reset_global_emitter()
        yield
        reset_global_emitter()

    def test_review_transitions_state(self):
        manager = ArtifactManager(state_machine=LifecycleStateMachine())
        art = _art()
        manager.register(art)
        art.state = ArtifactLifecycleState.RENDERED  # 渲染后进入可审核状态
        reviewed = manager.review(art.artifact_id, reviewer="cc1")
        assert reviewed.state == ArtifactLifecycleState.REVIEWED
        assert "review" in reviewed.learner_context
        assert reviewed.learner_context["review"]["reviewer"] == "cc1"

    def test_review_invalid_state_raises(self):
        manager = ArtifactManager(state_machine=LifecycleStateMachine())
        art = _art()
        manager.register(art)  # CREATED 不能直接审核
        with pytest.raises(Exception):
            manager.review(art.artifact_id)

    def test_unarchive_restores(self):
        manager = ArtifactManager()
        art = _art()
        manager.register(art)
        manager.archive(art.artifact_id)
        assert manager.get(art.artifact_id).state == ArtifactLifecycleState.ARCHIVED
        restored = manager.unarchive(art.artifact_id)
        assert restored.state == ArtifactLifecycleState.RENDERED

    def test_unarchive_non_archived_raises(self):
        manager = ArtifactManager(state_machine=LifecycleStateMachine())
        art = _art()
        manager.register(art)
        with pytest.raises(Exception):
            manager.unarchive(art.artifact_id)  # CREATED→RENDERED 非法


class TestRemove:
    """移除 (软删/硬删)."""

    def test_soft_remove_keeps_version_tree(self):
        manager = ArtifactManager()
        art = _art()
        manager.register(art)
        manager.update(art.artifact_id, {"content": "v2"})
        assert manager.remove(art.artifact_id) is True
        with pytest.raises(ArtifactNotFoundError):
            manager.get(art.artifact_id)
        # 软删: 版本树仍在
        assert art.artifact_id in manager._version_trees  # noqa: SLF001

    def test_hard_remove_clears_tree(self):
        manager = ArtifactManager()
        art = _art()
        manager.register(art)
        manager.remove(art.artifact_id, hard=True)
        assert art.artifact_id not in manager._version_trees  # noqa: SLF001

    def test_remove_unknown_raises(self):
        manager = ArtifactManager()
        with pytest.raises(ArtifactNotFoundError):
            manager.remove("art-missing")


class TestMergeManager:
    """ArtifactManager.merge 集成."""

    def test_merge_ours(self):
        manager = ArtifactManager()
        art = _art({"content": "v1"})
        manager.register(art)
        manager.update(art.artifact_id, {"content": "v2"})
        forked = manager.fork(art.artifact_id, fork_reason="branch-A")
        merged = manager.merge(forked.artifact_id, branch_version=3, strategy="ours")
        # 合并成为新 head
        assert manager.get(art.artifact_id).version == merged.version
        assert "merge" in (merged.fork_origin or "")

    def test_merge_missing_branch_raises(self):
        manager = ArtifactManager()
        art = _art()
        manager.register(art)
        with pytest.raises(Exception):
            manager.merge(art.artifact_id, branch_version=99, strategy="auto")


class TestArchivedGuard:
    """归档状态防护."""

    def test_update_archived_rejected(self):
        manager = ArtifactManager()
        art = _art()
        manager.register(art)
        manager.archive(art.artifact_id)
        with pytest.raises(Exception, match="已归档"):
            manager.update(art.artifact_id, {"content": "y"})

    def test_apply_edit_archived_rejected(self):
        manager = ArtifactManager()
        art = _art()
        manager.register(art)
        manager.archive(art.artifact_id)
        diff = ArtifactDiff(artifact_id=art.artifact_id,
                            ops=[{"op": "replace", "path": "content", "value": "y"}])
        with pytest.raises(Exception, match="已归档"):
            manager.apply_edit(art.artifact_id, diff)

    def test_fork_archived_rejected(self):
        manager = ArtifactManager()
        art = _art()
        manager.register(art)
        manager.archive(art.artifact_id)
        with pytest.raises(Exception, match="已归档"):
            manager.fork(art.artifact_id)

    def test_guard_disabled_allows(self):
        manager = ArtifactManager(guard_archived=False)
        art = _art()
        manager.register(art)
        manager.archive(art.artifact_id)
        updated = manager.update(art.artifact_id, {"content": "y"})
        assert updated.payload["content"] == "y"


class TestManagerEvents:
    """生命周期事件发射."""

    def test_events_emitted(self):
        reset_global_emitter()
        recorder = _EventRecorder()
        recorder.attach()
        try:
            manager = ArtifactManager()
            art = _art()
            manager.register(art)
            art.state = ArtifactLifecycleState.RENDERED
            manager.review(art.artifact_id)
            manager.archive(art.artifact_id)
            manager.unarchive(art.artifact_id)
            manager.remove(art.artifact_id)
        finally:
            recorder.detach()
            reset_global_emitter()
        assert ARTIFACT_REGISTERED in recorder.events
        assert ARTIFACT_REVIEWED in recorder.events
        assert ARTIFACT_ARCHIVED in recorder.events
        assert ARTIFACT_RESTORED in recorder.events
        assert ARTIFACT_REMOVED in recorder.events

    def test_events_disabled(self):
        reset_global_emitter()
        recorder = _EventRecorder()
        recorder.attach()
        try:
            manager = ArtifactManager(emit_events=False)
            art = _art()
            manager.register(art)
        finally:
            recorder.detach()
            reset_global_emitter()
        assert ARTIFACT_REGISTERED not in recorder.events


class TestManagerSnapshot:
    """快照持久化往返."""

    def test_snapshot_roundtrip(self, tmp_path):
        path = str(tmp_path / "snap.json")
        m1 = ArtifactManager()
        a1 = _art({"content": "持久化数据"}, title="snapshot-art")
        m1.register(a1)
        m1.update(a1.artifact_id, {"content": "v2"})
        assert m1.save_snapshot(path) == 1

        m2 = ArtifactManager()
        assert m2.load_snapshot(path) == 1
        restored = m2.get(a1.artifact_id)
        assert restored is not None
        assert restored.payload["content"] == "v2"

    def test_load_missing_snapshot_raises(self, tmp_path):
        m = ArtifactManager()
        with pytest.raises(FileNotFoundError):
            m.load_snapshot(str(tmp_path / "missing.json"))


class TestManagerSearchIntegration:
    """fulltext_search / related_by_kp 集成."""

    def test_fulltext_search(self):
        manager = ArtifactManager()
        a1 = _art({"content": "荧光效率分析"}, title="效率报告", learner_context={"kp_ids": ["A-01"]})
        a2 = _art({"content": "材料性能对比"}, title="性能图", learner_context={"kp_ids": ["A-02"]})
        manager.register(a1)
        manager.register(a2)
        results = manager.fulltext_search("荧光", sort_by="-created_at")
        assert len(results) == 1
        assert results[0].title == "效率报告"

    def test_fulltext_search_filters(self):
        manager = ArtifactManager()
        a1 = _art({"content": "图表内容"}, mime="application/vnd.dy3.chart+json")
        a2 = _art({"content": "文本内容"})
        manager.register(a1)
        manager.register(a2)
        results = manager.fulltext_search("", filters={"type": "chart"})
        assert len(results) == 1

    def test_related_by_kp(self):
        manager = ArtifactManager()
        a1 = _art({"content": "A 域内容"}, learner_context={"kp_ids": ["A-01"]})
        a2 = _art({"content": "B 域内容"}, learner_context={"kp_ids": ["B-01"]})
        manager.register(a1)
        manager.register(a2)
        related = manager.related_by_kp("A-01")
        assert len(related) == 1
        assert related[0].title == ""

    def test_register_idempotent(self):
        manager = ArtifactManager()
        art = _art()
        manager.register(art)
        manager.register(art)
        assert manager.get(art.artifact_id).version == 1
