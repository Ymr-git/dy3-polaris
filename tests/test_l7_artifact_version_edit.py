"""L7 Artifact 管理系统 T3 — 版本 DAG 合并与编辑通道单元测试.

测试覆盖:
1. ArtifactVersionGraph: 多 parent 合并、公共祖先、谱系、拓扑序
2. MergeConflictError: auto 三方合并冲突检测
3. EditChannel: 编辑提交、订阅者广播、历史记录、渲染更新回调
4. EditPermission: CC2 只读 / Agent 默认可编辑
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.artifact.edit_channel import EditChannel
from dy3_polaris.l7.artifact.models import EditPermission
from dy3_polaris.l7.artifact.version_manager import (
    ArtifactVersionGraph,
    MergeConflictError,
    MergeResult,
)
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactDiff,
    ArtifactType,
)
from dy3_polaris.l7.artifact_manager import ArtifactManager


def _art(payload: dict, **kwargs) -> Artifact:
    return Artifact(
        type=ArtifactType.TEXT,
        mime="text/vnd.dy3+markdown",
        payload=payload,
        **kwargs,
    )


# ============================================================
# 版本 DAG
# ============================================================

class TestArtifactVersionGraph:
    """DAG 版本管理 (设计文档 Ch.3.3)."""

    def _build_forked_graph(self):
        """v1 → v2 → v3a (fork A) / v3b (fork B)."""
        g = ArtifactVersionGraph("art-1")
        g.add_version(_art({"a": 1}))
        g.add_version(_art({"a": 2}), parents=(1,))
        g.add_version(
            _art({"a": 2, "b": 1}), parents=(2,), fork_origin="session-fork-A"
        )
        g.add_version(
            _art({"a": 9, "c": 2}), parents=(2,), fork_origin="session-fork-B"
        )
        return g

    def test_linear_versions(self):
        g = ArtifactVersionGraph("art")
        n1 = g.add_version(_art({"a": 1}))
        n2 = g.add_version(_art({"a": 2}), parents=(1,))
        assert n1.version == 1 and n2.version == 2
        assert g.latest_version() == 2
        assert g.get_lineage(2) == [1, 2]

    def test_fork_creates_branch(self):
        g = self._build_forked_graph()
        # v3a/v3b 的 parent 均为 v2
        assert g.parent_versions(3) == [2]
        assert g.parent_versions(4) == [2]
        assert g.get_node(3).fork_origin == "session-fork-A"
        assert g.get_node(4).fork_origin == "session-fork-B"

    def test_common_ancestor(self):
        g = self._build_forked_graph()
        assert g.common_ancestor(3, 4) == 2
        assert g.common_ancestor(2, 3) == 2
        assert g.common_ancestor(4, 1) == 1

    def test_is_descendant(self):
        g = self._build_forked_graph()
        assert g.is_descendant(2, 3) is True
        assert g.is_descendant(1, 4) is True
        assert g.is_descendant(3, 4) is False

    def test_topological_order(self):
        g = self._build_forked_graph()
        order = g.all_versions()
        assert order == [1, 2, 3, 4]  # 父先于子

    def test_merge_creates_multi_parent(self):
        g = self._build_forked_graph()
        result = g.merge("art-1", branch_version=3, main_version=4, strategy="auto")
        assert isinstance(result, MergeResult)
        assert result.merge_version == 5
        assert set(result.parents) == {3, 4}  # 多 parent (Git merge commit)

    def test_merge_auto_no_conflict(self):
        g = self._build_forked_graph()
        result = g.merge("art-1", branch_version=3, main_version=4, strategy="auto")
        assert result.conflicts == []
        # 合并 payload = 分支内容 (theirs 分支)
        assert result.merged_payload == {"a": 2, "b": 1}

    def test_merge_ours_strategy(self):
        g = self._build_forked_graph()
        result = g.merge("art-1", branch_version=3, main_version=4, strategy="ours")
        assert result.merged_payload == {"a": 9, "c": 2}

    def test_merge_theirs_strategy(self):
        g = self._build_forked_graph()
        result = g.merge("art-1", branch_version=3, main_version=4, strategy="theirs")
        assert result.merged_payload == {"a": 2, "b": 1}

    def test_merge_conflict_raises(self):
        # v3a 与 v3b 都改 a 字段 → auto 冲突
        g = ArtifactVersionGraph("art-2")
        g.add_version(_art({"a": 1}))
        g.add_version(_art({"a": 2}), parents=(1,))
        g.add_version(_art({"a": 3}), parents=(2,), fork_origin="fA")
        g.add_version(_art({"a": 9}), parents=(2,), fork_origin="fB")
        with pytest.raises(MergeConflictError) as exc:
            g.merge("art-2", branch_version=3, main_version=4, strategy="auto")
        assert "a" in exc.value.conflicts

    def test_merge_missing_version_raises(self):
        g = ArtifactVersionGraph("art")
        g.add_version(_art({"a": 1}))
        with pytest.raises(ValueError):
            g.merge("art", branch_version=5, main_version=1, strategy="auto")

    def test_snapshot_access(self):
        g = self._build_forked_graph()
        snap = g.get_snapshot(3)
        assert snap is not None
        assert snap.payload == {"a": 2, "b": 1}


# ============================================================
# 编辑通道
# ============================================================

class _FakeSubscriber:
    """模拟 L5 Agent Runtime 的编辑订阅者."""

    def __init__(self):
        self.edits: list[tuple[str, str]] = []

    def on_artifact_edit(self, artifact: Artifact, diff: ArtifactDiff) -> None:
        self.edits.append((artifact.artifact_id, diff.edit_reason))


class TestEditChannel:
    """Artifact-Edit 通道 (设计文档 Ch.3.4)."""

    def _make_channel(self, manager: ArtifactManager, updater=None):
        return EditChannel(
            diff_applier=manager.apply_edit,
            render_updater=updater,
        )

    def test_submit_applies_diff_and_creates_version(self):
        manager = ArtifactManager()
        art = _art({"content": "原文"})
        manager.register(art)
        channel = self._make_channel(manager)
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[{"op": "replace", "path": "content", "value": "编辑后"}],
            edit_reason="用户修改",
        )
        new_art = channel.submit(art.artifact_id, diff, art)
        assert new_art.version == 2
        assert new_art.payload["content"] == "编辑后"

    def test_submit_broadcasts_to_subscribers(self):
        manager = ArtifactManager()
        art = _art({"content": "x"})
        manager.register(art)
        channel = self._make_channel(manager)
        sub = _FakeSubscriber()
        channel.subscribe("agent-a1", sub)
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[{"op": "add", "path": "note", "value": "n"}],
            edit_reason="补充说明",
        )
        channel.submit(art.artifact_id, diff, art)
        assert len(sub.edits) == 1
        assert sub.edits[0][1] == "补充说明"

    def test_unsubscribe(self):
        manager = ArtifactManager()
        art = _art({"content": "x"})
        manager.register(art)
        channel = self._make_channel(manager)
        sub = _FakeSubscriber()
        channel.subscribe("agent-a1", sub)
        assert channel.unsubscribe("agent-a1") is True
        assert channel.subscriber_names() == []

    def test_render_updater_called(self):
        manager = ArtifactManager()
        art = _art({"content": "x"})
        manager.register(art)
        calls = []
        updater = lambda diff: calls.append(diff.artifact_id)  # noqa: E731
        channel = self._make_channel(manager, updater=updater)
        diff = ArtifactDiff(artifact_id=art.artifact_id, ops=[], edit_reason="r")
        channel.submit(art.artifact_id, diff, art)
        assert calls == [art.artifact_id]

    def test_not_editable_raises(self):
        manager = ArtifactManager()
        art = _art({"content": "x"}, editable=False)
        manager.register(art)
        channel = self._make_channel(manager)
        diff = ArtifactDiff(artifact_id=art.artifact_id, ops=[])
        with pytest.raises(Exception):
            channel.submit(art.artifact_id, diff, art)

    def test_history_recorded(self):
        manager = ArtifactManager()
        art = _art({"content": "x"})
        manager.register(art)
        channel = self._make_channel(manager)
        diff = ArtifactDiff(artifact_id=art.artifact_id, ops=[], edit_reason="r1")
        channel.submit(art.artifact_id, diff, art)
        diff2 = ArtifactDiff(artifact_id=art.artifact_id, ops=[], edit_reason="r2")
        channel.submit(art.artifact_id, diff2, art)
        assert channel.count() == 2
        assert len(channel.history()) == 2


# ============================================================
# 编辑权限
# ============================================================

class TestEditPermission:
    """editable 双源决定 (设计文档 Ch.3.4)."""

    def test_cc2_approved_readonly(self):
        perm = EditPermission.cc2_approved()
        assert perm.editable is False
        assert perm.source == "cc2_approved"

    def test_agent_editable_default(self):
        perm = EditPermission.agent_editable()
        assert perm.editable is True
        assert perm.source == "source_agent"

    def test_to_dict(self):
        perm = EditPermission.cc2_approved()
        d = perm.to_dict()
        assert d["editable"] is False
        assert d["source"] == "cc2_approved"
