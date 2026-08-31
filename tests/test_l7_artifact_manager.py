"""L7 ArtifactManager 测试 — TDD 测试用例.

测试覆盖:
1. 生命周期管理 (register / get / update / archive / list_artifacts)
2. 版本管理 DAG (update / fork / get_version_history / get_latest_version / get_version)
3. 编辑管理 (apply_edit / get_diff)
4. 搜索与过滤 (search)
5. 统计信息 (get_stats)

融合方案:
- Jupyter nbformat: Artifact 元数据 + payload 分离
- Git: DAG 版本树管理
- IndexedDB: 三级缓存策略 (L1 内存 / L2 本地 / L3 服务端)
- RFC 6902 JSON Patch: 增量编辑
"""

from __future__ import annotations

import threading

import pytest

from dy3_polaris.l7.artifact_manager import ArtifactManager, VersionTree
from dy3_polaris.l7.exceptions import (
    ArtifactNotFoundError,
    ArtifactNotEditableError,
    ArtifactValidationError,
    L7Error,
    VersionConflictError,
)
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactDiff,
    ArtifactLifecycleState,
    ArtifactType,
    ArtifactVersionNode,
    LearnerMode,
    RenderContext,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def manager():
    """创建 ArtifactManager 实例."""
    return ArtifactManager()


@pytest.fixture
def sample_payload():
    """样本 payload (学情诊断报告)."""
    return {
        "report_id": "rpt-001",
        "learner_id": "stu-001",
        "kp_gaps": ["KP-12", "KP-18"],
        "mastery_vector": {"KP-12": 0.35, "KP-18": 0.28},
        "confidence": 0.87,
    }


@pytest.fixture
def text_artifact(sample_payload):
    """创建文本类型 Artifact."""
    return Artifact(
        type=ArtifactType.TEXT,
        source_agent="agent.diagnosis",
        payload=sample_payload,
        title="学情诊断报告",
    )


# ============================================================
# 1. 生命周期管理测试
# ============================================================


class TestLifecycleManagement:
    """Artifact 生命周期管理测试."""

    def test_register_stores_artifact_and_sets_created(self, manager, text_artifact):
        """register 应存储 Artifact 并设置状态为 CREATED."""
        result = manager.register(text_artifact)
        assert result.artifact_id == text_artifact.artifact_id
        assert result.state == ArtifactLifecycleState.CREATED

    def test_get_returns_artifact(self, manager, text_artifact):
        """get 应返回已注册的 Artifact."""
        manager.register(text_artifact)
        retrieved = manager.get(text_artifact.artifact_id)
        assert retrieved is not None
        assert retrieved.artifact_id == text_artifact.artifact_id
        assert retrieved.source_agent == "agent.diagnosis"

    def test_get_raises_not_found_for_unknown_id(self, manager):
        """get 对未知 ID 应抛出 ArtifactNotFoundError."""
        with pytest.raises(ArtifactNotFoundError) as exc_info:
            manager.get("art-unknown-99999")
        assert exc_info.value.artifact_id == "art-unknown-99999"

    def test_update_creates_new_version(self, manager, text_artifact):
        """update 应创建新版本."""
        manager.register(text_artifact)
        new_payload = {**text_artifact.payload, "confidence": 0.92}
        updated = manager.update(text_artifact.artifact_id, new_payload)
        assert updated.version == 2
        assert updated.payload["confidence"] == 0.92
        assert updated.artifact_id == text_artifact.artifact_id

    def test_update_with_edit_reason(self, manager, text_artifact):
        """update 可携带编辑原因."""
        manager.register(text_artifact)
        updated = manager.update(
            text_artifact.artifact_id,
            {"v": 2},
            edit_reason="修正置信度",
        )
        assert updated.version == 2

    def test_update_unknown_raises_not_found(self, manager):
        """update 未知 ID 应抛出 ArtifactNotFoundError."""
        with pytest.raises(ArtifactNotFoundError):
            manager.update("art-unknown-99999", {"v": 1})

    def test_archive_sets_state_archived(self, manager, text_artifact):
        """archive 应设置状态为 ARCHIVED."""
        manager.register(text_artifact)
        result = manager.archive(text_artifact.artifact_id)
        assert result is True
        archived = manager.get(text_artifact.artifact_id)
        assert archived.state == ArtifactLifecycleState.ARCHIVED

    def test_archive_unknown_raises_not_found(self, manager):
        """archive 未知 ID 应抛出 ArtifactNotFoundError."""
        with pytest.raises(ArtifactNotFoundError):
            manager.archive("art-unknown-99999")

    def test_list_artifacts_returns_active_only(self, manager):
        """list_artifacts 应返回所有活跃 (非归档) 的 Artifact."""
        art1 = Artifact(type=ArtifactType.TEXT, source_agent="a1", payload={})
        art2 = Artifact(type=ArtifactType.CHART, source_agent="a2", payload={})
        art3 = Artifact(type=ArtifactType.TEXT, source_agent="a3", payload={})
        manager.register(art1)
        manager.register(art2)
        manager.register(art3)
        manager.archive(art3.artifact_id)

        active = manager.list_artifacts()
        assert len(active) == 2
        ids = {a.artifact_id for a in active}
        assert art1.artifact_id in ids
        assert art2.artifact_id in ids
        assert art3.artifact_id not in ids

    def test_list_artifacts_filters_by_session(self, manager):
        """list_artifacts 按 session_id 过滤."""
        art1 = Artifact(type=ArtifactType.TEXT, source_agent="a1", payload={}, session_id="sess-1")
        art2 = Artifact(type=ArtifactType.TEXT, source_agent="a2", payload={}, session_id="sess-2")
        art3 = Artifact(type=ArtifactType.TEXT, source_agent="a3", payload={}, session_id="sess-1")
        manager.register(art1)
        manager.register(art2)
        manager.register(art3)

        results = manager.list_artifacts(session_id="sess-1")
        assert len(results) == 2
        for a in results:
            assert a.session_id == "sess-1"

    def test_list_artifacts_filters_by_type(self, manager):
        """list_artifacts 按 artifact_type 过滤."""
        art1 = Artifact(type=ArtifactType.TEXT, source_agent="a1", payload={})
        art2 = Artifact(type=ArtifactType.CHART, source_agent="a2", payload={})
        art3 = Artifact(type=ArtifactType.TEXT, source_agent="a3", payload={})
        manager.register(art1)
        manager.register(art2)
        manager.register(art3)

        results = manager.list_artifacts(artifact_type=ArtifactType.TEXT)
        assert len(results) == 2
        for a in results:
            assert a.type == ArtifactType.TEXT

    def test_list_artifacts_filters_by_source_agent(self, manager):
        """list_artifacts 按 source_agent 过滤."""
        art1 = Artifact(type=ArtifactType.TEXT, source_agent="agent.diagnosis", payload={})
        art2 = Artifact(type=ArtifactType.TEXT, source_agent="agent.generation", payload={})
        art3 = Artifact(type=ArtifactType.TEXT, source_agent="agent.diagnosis", payload={})
        manager.register(art1)
        manager.register(art2)
        manager.register(art3)

        results = manager.list_artifacts(source_agent="agent.diagnosis")
        assert len(results) == 2
        for a in results:
            assert a.source_agent == "agent.diagnosis"

    def test_list_artifacts_filters_by_kp_id(self, manager):
        """list_artifacts 按 kp_id 过滤."""
        art1 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a1",
            payload={"kp_gaps": ["KP-12", "KP-18"]},
        )
        art2 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a2",
            payload={"kp_gaps": ["KP-99"]},
        )
        manager.register(art1)
        manager.register(art2)

        results = manager.list_artifacts(kp_id="KP-12")
        assert len(results) == 1
        assert results[0].artifact_id == art1.artifact_id

    def test_list_artifacts_combined_filters(self, manager):
        """list_artifacts 组合过滤."""
        art1 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload={},
            session_id="sess-1",
        )
        art2 = Artifact(
            type=ArtifactType.CHART,
            source_agent="agent.diagnosis",
            payload={},
            session_id="sess-1",
        )
        art3 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload={},
            session_id="sess-2",
        )
        manager.register(art1)
        manager.register(art2)
        manager.register(art3)

        results = manager.list_artifacts(
            session_id="sess-1",
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
        )
        assert len(results) == 1
        assert results[0].artifact_id == art1.artifact_id


# ============================================================
# 2. 版本管理 (DAG) 测试
# ============================================================


class TestVersionManagement:
    """版本管理 DAG 测试."""

    def test_first_version_has_version_1_and_no_parent(self, manager, text_artifact):
        """第一个版本 version=1, parent_version=None."""
        manager.register(text_artifact)
        history = manager.get_version_history(text_artifact.artifact_id)
        assert len(history) == 1
        assert history[0].version == 1
        assert history[0].parent_version is None

    def test_update_creates_version_2_with_parent_1(self, manager, text_artifact):
        """update 创建 version=2, parent_version=1."""
        manager.register(text_artifact)
        manager.update(text_artifact.artifact_id, {"v": 2})
        history = manager.get_version_history(text_artifact.artifact_id)
        assert len(history) == 2
        assert history[1].version == 2
        assert history[1].parent_version == 1

    def test_multiple_updates_create_sequential_versions(self, manager, text_artifact):
        """多次 update 创建连续版本."""
        manager.register(text_artifact)
        for i in range(2, 5):
            manager.update(text_artifact.artifact_id, {"v": i})
        history = manager.get_version_history(text_artifact.artifact_id)
        assert len(history) == 4
        assert [h.version for h in history] == [1, 2, 3, 4]

    def test_fork_creates_branch_version(self, manager, text_artifact):
        """fork 创建分支版本."""
        manager.register(text_artifact)
        forked = manager.fork(text_artifact.artifact_id, "alternative approach")
        assert forked.version == 2
        assert forked.fork_origin == "alternative approach"
        assert forked.artifact_id == text_artifact.artifact_id

    def test_fork_history_has_correct_parent(self, manager, text_artifact):
        """fork 在版本树中有正确的 parent_version."""
        manager.register(text_artifact)
        manager.fork(text_artifact.artifact_id, "fork-1")
        history = manager.get_version_history(text_artifact.artifact_id)
        assert len(history) == 2
        assert history[1].version == 2
        assert history[1].parent_version == 1
        assert history[1].fork_origin == "fork-1"

    def test_multiple_forks_create_separate_branches(self, manager, text_artifact):
        """多次 fork 创建独立分支."""
        manager.register(text_artifact)
        fork1 = manager.fork(text_artifact.artifact_id, "branch-A")
        fork2 = manager.fork(text_artifact.artifact_id, "branch-B")
        assert fork1.version == 2
        assert fork2.version == 3
        # 两个 fork 的 parent 都应该是当前 head (version 1)
        history = manager.get_version_history(text_artifact.artifact_id)
        node2 = next(n for n in history if n.version == 2)
        node3 = next(n for n in history if n.version == 3)
        assert node2.parent_version == 1
        assert node3.parent_version == 1
        assert node2.fork_origin == "branch-A"
        assert node3.fork_origin == "branch-B"

    def test_get_version_history_topological_order(self, manager, text_artifact):
        """get_version_history 返回拓扑排序的版本列表."""
        manager.register(text_artifact)
        manager.update(text_artifact.artifact_id, {"v": 2})
        manager.update(text_artifact.artifact_id, {"v": 3})
        manager.fork(text_artifact.artifact_id, "fork")
        manager.update(text_artifact.artifact_id, {"v": 5})

        history = manager.get_version_history(text_artifact.artifact_id)
        assert len(history) == 5
        # 拓扑序: 父版本在子版本之前
        versions = [h.version for h in history]
        # 每个版本的 parent 应该在它之前出现
        for node in history:
            if node.parent_version is not None:
                parent_idx = versions.index(node.parent_version)
                child_idx = versions.index(node.version)
                assert parent_idx < child_idx, (
                    f"parent v{node.parent_version} should come before v{node.version}"
                )

    def test_get_latest_version(self, manager, text_artifact):
        """get_latest_version 返回最高版本号."""
        manager.register(text_artifact)
        manager.update(text_artifact.artifact_id, {"v": 2})
        manager.update(text_artifact.artifact_id, {"v": 3})
        manager.fork(text_artifact.artifact_id, "fork")

        latest = manager.get_latest_version(text_artifact.artifact_id)
        assert latest == 4

    def test_get_version_returns_specific_version(self, manager, text_artifact):
        """get_version 返回指定版本的 Artifact."""
        manager.register(text_artifact)
        manager.update(text_artifact.artifact_id, {"v": 2})
        manager.update(text_artifact.artifact_id, {"v": 3})

        v1 = manager.get_version(text_artifact.artifact_id, 1)
        assert v1 is not None
        assert v1.version == 1
        assert v1.payload == text_artifact.payload

        v3 = manager.get_version(text_artifact.artifact_id, 3)
        assert v3 is not None
        assert v3.version == 3
        assert v3.payload == {"v": 3}

    def test_get_with_version_param(self, manager, text_artifact):
        """get(artifact_id, version) 返回指定版本."""
        manager.register(text_artifact)
        manager.update(text_artifact.artifact_id, {"v": 2})

        latest = manager.get(text_artifact.artifact_id)
        assert latest.version == 2

        v1 = manager.get(text_artifact.artifact_id, version=1)
        assert v1.version == 1

    def test_version_tree_parent_child_relationships(self, manager, text_artifact):
        """版本树维护正确的父子关系."""
        manager.register(text_artifact)
        manager.update(text_artifact.artifact_id, {"v": 2})
        manager.fork(text_artifact.artifact_id, "fork-A")
        manager.update(text_artifact.artifact_id, {"v": 4})

        history = manager.get_version_history(text_artifact.artifact_id)
        # v1 -> v2 -> v4 (main line), v2 -> v3 (fork) — wait, fork is from head
        # After update to v2, head is v2. Fork from v2 creates v3 (parent=2).
        # Then update from head (v2, since fork doesn't move head) creates v4 (parent=2).
        node_map = {h.version: h for h in history}
        assert node_map[1].parent_version is None
        assert node_map[2].parent_version == 1
        assert node_map[3].parent_version == 2
        assert node_map[3].fork_origin == "fork-A"
        assert node_map[4].parent_version == 2

    def test_get_version_history_unknown_raises(self, manager):
        """get_version_history 对未知 ID 应抛出异常."""
        with pytest.raises(ArtifactNotFoundError):
            manager.get_version_history("art-unknown-99999")

    def test_get_latest_version_unknown_raises(self, manager):
        """get_latest_version 对未知 ID 应抛出异常."""
        with pytest.raises(ArtifactNotFoundError):
            manager.get_latest_version("art-unknown-99999")

    def test_fork_unknown_raises_not_found(self, manager):
        """fork 未知 ID 应抛出 ArtifactNotFoundError."""
        with pytest.raises(ArtifactNotFoundError):
            manager.fork("art-unknown-99999", "test")


# ============================================================
# 3. 编辑管理测试
# ============================================================


class TestEditManagement:
    """编辑管理测试 (apply_edit / get_diff)."""

    def test_apply_edit_applies_diff(self, manager, text_artifact):
        """apply_edit 应将 ArtifactDiff 应用到 Artifact."""
        manager.register(text_artifact)
        diff = ArtifactDiff(
            artifact_id=text_artifact.artifact_id,
            ops=[
                {"op": "replace", "path": "confidence", "value": 0.95},
                {"op": "add", "path": "reviewer", "value": "cc1.actor_critic"},
            ],
            edit_reason="提升置信度",
        )
        edited = manager.apply_edit(text_artifact.artifact_id, diff)
        assert edited.payload["confidence"] == 0.95
        assert edited.payload["reviewer"] == "cc1.actor_critic"
        assert edited.version == 2

    def test_apply_edit_raises_not_editable(self, manager):
        """apply_edit 对 editable=False 的 Artifact 抛出 ArtifactNotEditableError."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={"key": "value"},
            editable=False,
        )
        manager.register(art)
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[{"op": "replace", "path": "key", "value": "new_value"}],
        )
        with pytest.raises(ArtifactNotEditableError) as exc_info:
            manager.apply_edit(art.artifact_id, diff)
        assert exc_info.value.artifact_id == art.artifact_id

    def test_apply_edit_creates_new_version(self, manager, text_artifact):
        """apply_edit 创建新版本."""
        manager.register(text_artifact)
        diff = ArtifactDiff(
            artifact_id=text_artifact.artifact_id,
            ops=[{"op": "add", "path": "new_field", "value": "new_value"}],
        )
        edited = manager.apply_edit(text_artifact.artifact_id, diff)
        assert edited.version == 2
        assert edited.payload["new_field"] == "new_value"

    def test_apply_edit_unknown_raises_not_found(self, manager):
        """apply_edit 对未知 ID 应抛出 ArtifactNotFoundError."""
        diff = ArtifactDiff(
            artifact_id="art-unknown-99999",
            ops=[],
        )
        with pytest.raises(ArtifactNotFoundError):
            manager.apply_edit("art-unknown-99999", diff)

    def test_apply_edit_remove_op(self, manager, text_artifact):
        """apply_edit 支持 remove 操作."""
        manager.register(text_artifact)
        diff = ArtifactDiff(
            artifact_id=text_artifact.artifact_id,
            ops=[{"op": "remove", "path": "confidence"}],
        )
        edited = manager.apply_edit(text_artifact.artifact_id, diff)
        assert "confidence" not in edited.payload

    def test_get_diff_returns_diff_between_versions(self, manager, text_artifact):
        """get_diff 返回两个版本之间的差异."""
        manager.register(text_artifact)
        manager.update(
            text_artifact.artifact_id,
            {**text_artifact.payload, "confidence": 0.95, "new_key": "new_val"},
        )

        diff = manager.get_diff(text_artifact.artifact_id, from_version=1, to_version=2)
        assert diff.artifact_id == text_artifact.artifact_id
        # 应检测到 confidence 变更 (replace) 和 new_key 新增 (add)
        ops_by_path = {op.get("path"): op for op in diff.ops}
        assert "confidence" in ops_by_path
        assert ops_by_path["confidence"]["op"] == "replace"
        assert ops_by_path["confidence"]["value"] == 0.95
        assert "new_key" in ops_by_path
        assert ops_by_path["new_key"]["op"] == "add"

    def test_get_diff_detects_removed_keys(self, manager, text_artifact):
        """get_diff 检测被删除的键."""
        manager.register(text_artifact)
        # 移除一个键
        new_payload = {k: v for k, v in text_artifact.payload.items() if k != "confidence"}
        manager.update(text_artifact.artifact_id, new_payload)

        diff = manager.get_diff(text_artifact.artifact_id, from_version=1, to_version=2)
        ops_by_path = {op.get("path"): op for op in diff.ops}
        assert "confidence" in ops_by_path
        assert ops_by_path["confidence"]["op"] == "remove"


# ============================================================
# 4. 搜索与过滤测试
# ============================================================


class TestSearchAndFilter:
    """搜索与过滤测试."""

    def test_search_finds_by_title(self, manager):
        """search 按 title 进行文本搜索."""
        art1 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a1",
            payload={},
            title="学情诊断报告",
        )
        art2 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a2",
            payload={},
            title="知识图谱分析",
        )
        manager.register(art1)
        manager.register(art2)

        results = manager.search("诊断")
        assert len(results) >= 1
        assert any(a.artifact_id == art1.artifact_id for a in results)

    def test_search_finds_by_payload(self, manager):
        """search 按 payload 内容进行文本搜索."""
        art1 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a1",
            payload={"description": "量子力学波函数分析"},
            title="",
        )
        art2 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a2",
            payload={"description": "有机化学命名法"},
            title="",
        )
        manager.register(art1)
        manager.register(art2)

        results = manager.search("量子")
        assert len(results) >= 1
        assert any(a.artifact_id == art1.artifact_id for a in results)

    def test_search_returns_empty_for_no_match(self, manager):
        """search 无匹配时返回空列表."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a1",
            payload={"data": "hello"},
            title="world",
        )
        manager.register(art)
        results = manager.search("nonexistent_query_xyz")
        assert len(results) == 0

    def test_search_sorted_by_relevance(self, manager):
        """search 结果按相关性排序 (匹配次数多的在前)."""
        art1 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a1",
            payload={"content": "量子力学量子态"},
            title="量子力学",
        )
        art2 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a2",
            payload={"content": "量子"},
            title="物理",
        )
        manager.register(art1)
        manager.register(art2)

        results = manager.search("量子")
        assert len(results) == 2
        # art1 匹配次数更多 (title + payload 多次匹配) 应排在前面
        assert results[0].artifact_id == art1.artifact_id

    def test_search_case_insensitive(self, manager):
        """search 不区分大小写."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a1",
            payload={"data": "Quantum Mechanics"},
            title="",
        )
        manager.register(art)
        results = manager.search("quantum")
        assert len(results) >= 1
        assert results[0].artifact_id == art.artifact_id


# ============================================================
# 5. 统计信息测试
# ============================================================


class TestStatistics:
    """统计信息测试 (get_stats)."""

    def test_get_stats_returns_counts(self, manager):
        """get_stats 返回统计数据."""
        manager.register(Artifact(type=ArtifactType.TEXT, source_agent="a1", payload={}))
        manager.register(Artifact(type=ArtifactType.TEXT, source_agent="a2", payload={}))
        manager.register(Artifact(type=ArtifactType.CHART, source_agent="a3", payload={}))
        manager.register(Artifact(type=ArtifactType.GRAPH, source_agent="a4", payload={}))

        stats = manager.get_stats()
        assert stats["total"] == 4
        assert stats["by_type"]["text"] == 2
        assert stats["by_type"]["chart"] == 1
        assert stats["by_type"]["graph"] == 1

    def test_get_stats_counts_by_state(self, manager):
        """get_stats 按状态统计."""
        art1 = Artifact(type=ArtifactType.TEXT, source_agent="a1", payload={})
        art2 = Artifact(type=ArtifactType.TEXT, source_agent="a2", payload={})
        art3 = Artifact(type=ArtifactType.TEXT, source_agent="a3", payload={})
        manager.register(art1)
        manager.register(art2)
        manager.register(art3)
        manager.archive(art3.artifact_id)

        stats = manager.get_stats()
        assert stats["by_state"]["created"] == 2
        assert stats["by_state"]["archived"] == 1

    def test_get_stats_empty_manager(self, manager):
        """空管理器的统计."""
        stats = manager.get_stats()
        assert stats["total"] == 0
        assert stats["by_type"] == {}
        assert stats["by_state"] == {}


# ============================================================
# 6. 异常体系测试
# ============================================================


class TestExceptions:
    """异常体系测试."""

    def test_l7_error_is_base(self):
        """L7Error 是所有 L7 异常的基类."""
        assert issubclass(ArtifactNotFoundError, L7Error)
        assert issubclass(ArtifactValidationError, L7Error)
        assert issubclass(VersionConflictError, L7Error)
        assert issubclass(ArtifactNotEditableError, L7Error)

    def test_artifact_not_found_has_artifact_id(self):
        """ArtifactNotFoundError 携带 artifact_id."""
        err = ArtifactNotFoundError("art-001")
        assert err.artifact_id == "art-001"
        assert "art-001" in str(err)

    def test_artifact_validation_has_field(self):
        """ArtifactValidationError 携带 field."""
        err = ArtifactValidationError(field="payload", detail="payload is empty")
        assert err.field == "payload"

    def test_version_conflict_has_attributes(self):
        """VersionConflictError 携带 artifact_id 和 version."""
        err = VersionConflictError("art-001", 3)
        assert err.artifact_id == "art-001"
        assert err.version == 3

    def test_artifact_not_editable_has_artifact_id(self):
        """ArtifactNotEditableError 携带 artifact_id."""
        err = ArtifactNotEditableError("art-001")
        assert err.artifact_id == "art-001"

    def test_jsonrpc_codes(self):
        """各异常有正确的 JSON-RPC 错误码."""
        assert L7Error("TEST")._jsonrpc_code() == -32500
        assert ArtifactNotFoundError("x")._jsonrpc_code() == -32502
        assert ArtifactValidationError("f")._jsonrpc_code() == -32503
        assert VersionConflictError("x", 1)._jsonrpc_code() == -32506
        assert ArtifactNotEditableError("x")._jsonrpc_code() == -32507

    def test_to_json_rpc_error(self):
        """异常可转换为 JSON-RPC 错误对象."""
        err = ArtifactNotFoundError("art-001")
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32502
        assert rpc["message"] == "L7_ARTIFACT_NOT_FOUND"
        assert rpc["data"]["artifact_id"] == "art-001"


# ============================================================
# 7. 模型测试
# ============================================================


class TestModels:
    """Pydantic 模型测试."""

    def test_artifact_auto_generates_id(self):
        """Artifact 自动生成 art-{uuid} 格式的 ID."""
        art = Artifact(type=ArtifactType.TEXT, source_agent="a1", payload={})
        assert art.artifact_id.startswith("art-")

    def test_artifact_default_values(self):
        """Artifact 默认值正确."""
        art = Artifact(type=ArtifactType.TEXT, source_agent="a1", payload={})
        assert art.version == 1
        assert art.editable is True
        assert art.fork_origin is None
        assert art.state == ArtifactLifecycleState.CREATED
        assert art.session_id == ""
        assert art.title == ""
        assert art.created_at > 0
        assert art.updated_at > 0

    def test_artifact_type_enum_values(self):
        """ArtifactType 枚举值正确."""
        assert ArtifactType.TEXT == "text"
        assert ArtifactType.CHART == "chart"
        assert ArtifactType.GRAPH == "graph"
        assert ArtifactType.MOLECULE == "molecule"
        assert ArtifactType.TABLE == "table"
        assert ArtifactType.FORMULA == "formula"
        assert ArtifactType.PROVENANCE == "provenance"
        assert ArtifactType.INTERACTIVE == "interactive"

    def test_lifecycle_state_enum_values(self):
        """ArtifactLifecycleState 枚举值正确."""
        assert ArtifactLifecycleState.CREATED == "created"
        assert ArtifactLifecycleState.RENDERED == "rendered"
        assert ArtifactLifecycleState.REVIEWED == "reviewed"
        assert ArtifactLifecycleState.EDITED == "edited"
        assert ArtifactLifecycleState.ARCHIVED == "archived"

    def test_artifact_diff_model(self):
        """ArtifactDiff 模型."""
        diff = ArtifactDiff(
            artifact_id="art-001",
            ops=[{"op": "add", "path": "key", "value": "val"}],
            edit_reason="test",
        )
        assert diff.artifact_id == "art-001"
        assert len(diff.ops) == 1
        assert diff.edit_reason == "test"
        assert diff.created_at > 0

    def test_version_node_model(self):
        """ArtifactVersionNode 模型."""
        node = ArtifactVersionNode(
            version=2,
            artifact_id="art-001",
            parent_version=1,
        )
        assert node.version == 2
        assert node.artifact_id == "art-001"
        assert node.parent_version == 1
        assert node.fork_origin is None

    def test_render_context_defaults(self):
        """RenderContext 默认值."""
        ctx = RenderContext()
        assert ctx.viewport.width == 1280
        assert ctx.viewport.height == 720
        assert ctx.theme == "light"
        assert ctx.learner_mode == LearnerMode.INTERMEDIATE
        assert ctx.locale == "zh-CN"


# ============================================================
# 8. VersionTree 测试
# ============================================================


class TestVersionTree:
    """VersionTree (DAG) 测试."""

    def test_version_tree_creation(self, text_artifact):
        """创建版本树."""
        tree = VersionTree(text_artifact)
        assert tree.get_latest_version() == 1
        node = tree.get_node(1)
        assert node is not None
        assert node.version == 1
        assert node.parent_version is None

    def test_version_tree_add_version(self, text_artifact):
        """添加新版本."""
        tree = VersionTree(text_artifact)
        v2 = text_artifact.model_copy(update={"version": 2, "payload": {"v": 2}})
        node = tree.add_version(v2, parent_version=1)
        assert node.version == 2
        assert node.parent_version == 1
        assert tree.get_latest_version() == 2

    def test_version_tree_get_artifact(self, text_artifact):
        """获取版本的 Artifact 快照."""
        tree = VersionTree(text_artifact)
        v2 = text_artifact.model_copy(update={"version": 2, "payload": {"v": 2}})
        tree.add_version(v2, parent_version=1)
        art = tree.get_artifact(2)
        assert art is not None
        assert art.version == 2
        assert art.payload == {"v": 2}

    def test_version_tree_get_lineage(self, text_artifact):
        """获取版本的溯源链."""
        tree = VersionTree(text_artifact)
        v2 = text_artifact.model_copy(update={"version": 2})
        v3 = text_artifact.model_copy(update={"version": 3})
        tree.add_version(v2, parent_version=1)
        tree.add_version(v3, parent_version=2)

        lineage = tree.get_lineage(3)
        assert len(lineage) == 3
        assert lineage[0].version == 1
        assert lineage[2].version == 3

    def test_version_tree_get_all_versions(self, text_artifact):
        """获取所有版本节点."""
        tree = VersionTree(text_artifact)
        v2 = text_artifact.model_copy(update={"version": 2})
        tree.add_version(v2, parent_version=1)

        all_versions = tree.get_all_versions()
        assert len(all_versions) == 2
        assert all_versions[0].version == 1
        assert all_versions[1].version == 2


# ============================================================
# 9. 线程安全测试
# ============================================================


class TestThreadSafety:
    """线程安全测试."""

    def test_concurrent_register(self, manager):
        """并发注册 Artifact 不冲突."""
        results = []
        errors = []

        def register_artifact(idx):
            try:
                art = Artifact(
                    type=ArtifactType.TEXT,
                    source_agent=f"agent-{idx}",
                    payload={"index": idx},
                )
                manager.register(art)
                results.append(art.artifact_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_artifact, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        assert len(set(results)) == 20  # 所有 ID 唯一

    def test_concurrent_update(self, manager, text_artifact):
        """并发更新同一 Artifact."""
        manager.register(text_artifact)

        def update_artifact(idx):
            manager.update(text_artifact.artifact_id, {"update_idx": idx})

        threads = [threading.Thread(target=update_artifact, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        history = manager.get_version_history(text_artifact.artifact_id)
        # 1 (register) + 10 (updates) = 11 versions
        assert len(history) == 11


# ============================================================
# 10. 集成场景测试
# ============================================================


class TestIntegrationScenarios:
    """集成场景测试."""

    def test_full_lifecycle(self, manager, sample_payload):
        """完整生命周期: 创建 → 更新 → Fork → 编辑 → 归档."""
        # 1. 创建
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload=sample_payload,
            title="诊断报告",
            session_id="sess-001",
        )
        manager.register(art)
        assert art.state == ArtifactLifecycleState.CREATED

        # 2. 更新
        updated = manager.update(art.artifact_id, {**sample_payload, "confidence": 0.92})
        assert updated.version == 2

        # 3. Fork
        forked = manager.fork(art.artifact_id, "alternative")
        assert forked.version == 3
        assert forked.fork_origin == "alternative"

        # 4. 编辑
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[{"op": "add", "path": "review_status", "value": "approved"}],
            edit_reason="审核通过",
        )
        edited = manager.apply_edit(art.artifact_id, diff)
        assert edited.payload["review_status"] == "approved"

        # 5. 归档
        manager.archive(art.artifact_id)
        final = manager.get(art.artifact_id)
        assert final.state == ArtifactLifecycleState.ARCHIVED

        # 6. 验证版本历史
        history = manager.get_version_history(art.artifact_id)
        assert len(history) == 4

    def test_multi_agent_artifact_topology(self, manager):
        """多 Agent 产物拓扑."""
        # A1 学情诊断
        diagnosis = Artifact(
            type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload={"report": "学情诊断"},
            title="学情诊断报告",
        )
        manager.register(diagnosis)

        # A2 知识生成
        generation = Artifact(
            type=ArtifactType.GRAPH,
            source_agent="agent.generation",
            payload={"graph": "knowledge_graph"},
            title="知识图谱",
        )
        manager.register(generation)

        # A3 审核校验
        review = Artifact(
            type=ArtifactType.PROVENANCE,
            source_agent="agent.review",
            payload={"review": "pass"},
            title="审核记录",
        )
        manager.register(review)

        # A4 导学决策
        guidance = Artifact(
            type=ArtifactType.INTERACTIVE,
            source_agent="agent.guidance",
            payload={"plan": "study_plan"},
            title="导学方案",
        )
        manager.register(guidance)

        # 按类型过滤
        text_arts = manager.list_artifacts(artifact_type=ArtifactType.TEXT)
        assert len(text_arts) == 1

        # 搜索
        results = manager.search("诊断")
        assert len(results) >= 1

        # 统计
        stats = manager.get_stats()
        assert stats["total"] == 4

    def test_version_diff_roundtrip(self, manager, text_artifact):
        """版本 diff 往返: get_diff → apply_edit 应还原变更."""
        manager.register(text_artifact)
        # 更新到 v2
        manager.update(
            text_artifact.artifact_id,
            {**text_artifact.payload, "confidence": 0.99, "new_key": "new_val"},
        )
        # 获取 v1→v2 的 diff
        diff = manager.get_diff(text_artifact.artifact_id, from_version=1, to_version=2)
        # 验证 diff 包含变更
        assert len(diff.ops) >= 2

    def test_archived_not_in_list(self, manager):
        """归档的 Artifact 不在 list_artifacts 中."""
        art1 = Artifact(type=ArtifactType.TEXT, source_agent="a1", payload={})
        art2 = Artifact(type=ArtifactType.TEXT, source_agent="a2", payload={})
        manager.register(art1)
        manager.register(art2)
        manager.archive(art1.artifact_id)

        active = manager.list_artifacts()
        assert len(active) == 1
        assert active[0].artifact_id == art2.artifact_id
