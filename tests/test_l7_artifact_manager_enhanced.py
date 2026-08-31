"""L7 ArtifactManager 增强测试 — TDD 测试用例.

增强覆盖:
1. 完整 RFC 6902 JSON Patch 操作 (move / copy / test + JSON Pointer 路径)
2. 嵌套 diff (递归比较 dict / list, 生成 JSON Pointer 路径)
3. 乐观版本锁 (update 新增 expected_version 参数)

设计约束:
- 向后兼容: 扁平 key (如 "content") 继续工作, JSON Pointer (如 "/content") 新增支持
- 现有 350 个测试不受影响
"""

from __future__ import annotations

import pytest

from dy3_polaris.l7.artifact_manager import ArtifactManager
from dy3_polaris.l7.exceptions import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    L7Error,
    VersionConflictError,
)
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactDiff,
    ArtifactType,
    DiffOp,
    DiffOpType,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def manager():
    """创建 ArtifactManager 实例."""
    return ArtifactManager()


@pytest.fixture
def nested_payload():
    """嵌套 payload (含 dict / list 多层结构)."""
    return {
        "content": "诊断报告",
        "data": [
            {"name": "Alice", "score": 90},
            {"name": "Bob", "score": 75},
        ],
        "config": {
            "enabled": True,
            "level": "intermediate",
            "thresholds": {"min": 0.5, "max": 0.9},
        },
        "confidence": 0.87,
    }


@pytest.fixture
def nested_artifact(nested_payload):
    """创建带嵌套 payload 的 Artifact."""
    return Artifact(
        type=ArtifactType.TEXT,
        source_agent="agent.diagnosis",
        payload=nested_payload,
        title="嵌套诊断报告",
    )


def _register_and_update(manager, artifact, new_payload):
    """注册 Artifact 并更新到 v2, 返回 artifact_id."""
    manager.register(artifact)
    manager.update(artifact.artifact_id, new_payload)
    return artifact.artifact_id


# ============================================================
# 1. 完整 RFC 6902 操作测试
# ============================================================


class TestRFC6902MoveOperation:
    """RFC 6902 MOVE 操作测试."""

    def test_move_operation_moves_nested_value(self, manager, nested_artifact):
        """move 操作移动嵌套值: source 被删除, target 被赋值."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "move", "path": "/config/level_renamed", "from": "/config/level"},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        # source 被删除
        assert "level" not in edited.payload["config"]
        # target 被赋值
        assert edited.payload["config"]["level_renamed"] == "intermediate"

    def test_move_operation_top_level_flat_key(self, manager, nested_artifact):
        """move 操作支持扁平 key (向后兼容)."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "move", "path": "confidence_renamed", "from": "confidence"},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        assert "confidence" not in edited.payload
        assert edited.payload["confidence_renamed"] == 0.87

    def test_move_operation_source_not_found_raises(self, manager, nested_artifact):
        """move 操作 source 不存在时抛出异常."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "move", "path": "/target", "from": "/config/missing_key"},
            ],
        )
        with pytest.raises(L7Error):
            manager.apply_edit(nested_artifact.artifact_id, diff)

    def test_move_operation_creates_new_version(self, manager, nested_artifact):
        """move 操作创建新版本."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "move", "path": "/new_confidence", "from": "/confidence"},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        assert edited.version == 2
        assert edited.payload["new_confidence"] == 0.87


class TestRFC6902CopyOperation:
    """RFC 6902 COPY 操作测试."""

    def test_copy_operation_copies_nested_value(self, manager, nested_artifact):
        """copy 操作复制嵌套值: source 保持不变, target 被赋值."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "copy", "path": "/config/level_copy", "from": "/config/level"},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        # source 保持不变
        assert edited.payload["config"]["level"] == "intermediate"
        # target 被赋值
        assert edited.payload["config"]["level_copy"] == "intermediate"

    def test_copy_operation_source_not_found_raises(self, manager, nested_artifact):
        """copy 操作 source 不存在时抛出异常."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "copy", "path": "/target", "from": "/config/missing_key"},
            ],
        )
        with pytest.raises(L7Error):
            manager.apply_edit(nested_artifact.artifact_id, diff)

    def test_copy_operation_independent_deep_copy(self, manager):
        """copy 操作对嵌套对象进行深拷贝, 修改 target 不影响 source."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"src": {"nested": {"value": 1}}},
        )
        manager.register(art)
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[{"op": "copy", "path": "/dst", "from": "/src"}],
        )
        edited = manager.apply_edit(art.artifact_id, diff)
        assert edited.payload["dst"] == {"nested": {"value": 1}}
        # 修改 dst 不应影响 src (深拷贝独立性)
        edited.payload["dst"]["nested"]["value"] = 999
        assert edited.payload["src"]["nested"]["value"] == 1


class TestRFC6902TestOperation:
    """RFC 6902 TEST 操作测试."""

    def test_test_operation_passes_when_value_matches(self, manager, nested_artifact):
        """test 操作值匹配时通过 (不抛异常, 创建新版本)."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "test", "path": "/config/level", "value": "intermediate"},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        # test 通过后应正常创建新版本
        assert edited.version == 2
        # payload 内容不变
        assert edited.payload["config"]["level"] == "intermediate"

    def test_test_operation_raises_when_value_mismatches(
        self, manager, nested_artifact
    ):
        """test 操作值不匹配时抛出 ArtifactValidationError."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "test", "path": "/config/level", "value": "advanced"},
            ],
        )
        with pytest.raises(ArtifactValidationError):
            manager.apply_edit(nested_artifact.artifact_id, diff)

    def test_test_operation_flat_key_backward_compatible(
        self, manager, nested_artifact
    ):
        """test 操作支持扁平 key (向后兼容)."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "test", "path": "confidence", "value": 0.87},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        assert edited.version == 2

    def test_test_operation_flat_key_mismatch_raises(
        self, manager, nested_artifact
    ):
        """test 操作扁平 key 值不匹配时抛出异常."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "test", "path": "confidence", "value": 0.5},
            ],
        )
        with pytest.raises(ArtifactValidationError):
            manager.apply_edit(nested_artifact.artifact_id, diff)


class TestJSONPointerPathSupport:
    """JSON Pointer 路径支持测试."""

    def test_json_pointer_path_resolves_nested_dict(self, manager, nested_artifact):
        """JSON Pointer 路径 /config/thresholds/max 正确解析 (嵌套 dict)."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "replace", "path": "/config/thresholds/max", "value": 0.95},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        assert edited.payload["config"]["thresholds"]["max"] == 0.95
        # 其他字段不变
        assert edited.payload["config"]["thresholds"]["min"] == 0.5

    def test_json_pointer_path_resolves_list_index(self, manager, nested_artifact):
        """JSON Pointer 路径 /data/0/name 正确解析 (list 索引)."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "replace", "path": "/data/0/name", "value": "Alicia"},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        assert edited.payload["data"][0]["name"] == "Alicia"
        assert edited.payload["data"][1]["name"] == "Bob"

    def test_json_pointer_add_to_nested_dict(self, manager, nested_artifact):
        """JSON Pointer add 操作向嵌套 dict 添加新键."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "add", "path": "/config/new_key", "value": "new_value"},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        assert edited.payload["config"]["new_key"] == "new_value"

    def test_json_pointer_remove_from_nested_dict(self, manager, nested_artifact):
        """JSON Pointer remove 操作删除嵌套 dict 中的键."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "remove", "path": "/config/thresholds/min"},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        assert "min" not in edited.payload["config"]["thresholds"]
        assert edited.payload["config"]["thresholds"]["max"] == 0.9

    def test_flat_key_continues_to_work(self, manager, nested_artifact):
        """扁平 key 'content' 继续工作 (向后兼容)."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "replace", "path": "content", "value": "更新后的报告"},
                {"op": "add", "path": "reviewer", "value": "cc1.actor_critic"},
                {"op": "remove", "path": "confidence"},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        assert edited.payload["content"] == "更新后的报告"
        assert edited.payload["reviewer"] == "cc1.actor_critic"
        assert "confidence" not in edited.payload

    def test_json_pointer_does_not_mutate_version_history(
        self, manager, nested_artifact
    ):
        """JSON Pointer 嵌套编辑不应破坏版本树中的历史快照."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "replace", "path": "/config/thresholds/max", "value": 0.95},
            ],
        )
        manager.apply_edit(nested_artifact.artifact_id, diff)
        # v1 的历史快照应保持原值
        v1 = manager.get_version(nested_artifact.artifact_id, 1)
        assert v1.payload["config"]["thresholds"]["max"] == 0.9


class TestMixedOperations:
    """混合操作测试 (单个 diff 中多种 op 协同工作)."""

    def test_mixed_add_move_test_in_single_diff(self, manager):
        """混合操作 (add + move + test) 在单个 diff 中工作."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"a": 1, "b": 2, "c": 3},
        )
        manager.register(art)
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[
                {"op": "add", "path": "/d", "value": 4},
                {"op": "move", "path": "/e", "from": "/a"},
                {"op": "test", "path": "/b", "value": 2},
            ],
        )
        edited = manager.apply_edit(art.artifact_id, diff)
        # add /d = 4
        assert edited.payload["d"] == 4
        # move /a -> /e (a 被删除, e = 1)
        assert "a" not in edited.payload
        assert edited.payload["e"] == 1
        # test /b == 2 通过, b 保持不变
        assert edited.payload["b"] == 2
        assert edited.payload["c"] == 3

    def test_mixed_replace_copy_remove(self, manager, nested_artifact):
        """混合 replace + copy + remove 操作."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[
                {"op": "replace", "path": "/confidence", "value": 0.99},
                {"op": "copy", "path": "/config/level_backup", "from": "/config/level"},
                {"op": "remove", "path": "/data/1"},
            ],
        )
        edited = manager.apply_edit(nested_artifact.artifact_id, diff)
        assert edited.payload["confidence"] == 0.99
        assert edited.payload["config"]["level"] == "intermediate"
        assert edited.payload["config"]["level_backup"] == "intermediate"
        # data[1] 被删除
        assert len(edited.payload["data"]) == 1
        assert edited.payload["data"][0]["name"] == "Alice"

    def test_test_failure_aborts_before_subsequent_ops(self, manager):
        """test 失败时抛出异常, 后续操作不执行."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"x": 10},
        )
        manager.register(art)
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[
                {"op": "test", "path": "/x", "value": 999},  # 失败
                {"op": "add", "path": "/y", "value": 20},
            ],
        )
        with pytest.raises(ArtifactValidationError):
            manager.apply_edit(art.artifact_id, diff)
        # 由于 test 失败, 应不产生新版本 (head 仍是 v1)
        latest = manager.get(art.artifact_id)
        assert latest.version == 1
        assert "y" not in latest.payload


# ============================================================
# 2. 嵌套 diff 测试
# ============================================================


class TestNestedDiff:
    """嵌套 diff 测试 (get_diff 递归比较)."""

    def test_top_level_key_change_backward_compatible(self, manager):
        """顶层 key 变更生成 add/remove/replace (向后兼容, 路径为扁平 key)."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"a": 1, "b": 2},
        )
        manager.register(art)
        manager.update(art.artifact_id, {"a": 1, "b": 3, "c": 4})
        diff = manager.get_diff(art.artifact_id, from_version=1, to_version=2)
        ops_by_path = {op.get("path"): op for op in diff.ops}
        # b 变更 (replace), c 新增 (add) — 扁平 key 路径
        assert "b" in ops_by_path
        assert ops_by_path["b"]["op"] == "replace"
        assert ops_by_path["b"]["value"] == 3
        assert "c" in ops_by_path
        assert ops_by_path["c"]["op"] == "add"
        assert ops_by_path["c"]["value"] == 4

    def test_top_level_key_removed(self, manager):
        """顶层 key 删除生成 remove (扁平 key 路径)."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"a": 1, "b": 2, "c": 3},
        )
        manager.register(art)
        manager.update(art.artifact_id, {"a": 1, "c": 3})
        diff = manager.get_diff(art.artifact_id, from_version=1, to_version=2)
        ops_by_path = {op.get("path"): op for op in diff.ops}
        assert "b" in ops_by_path
        assert ops_by_path["b"]["op"] == "remove"

    def test_nested_dict_change_generates_pointer_path(self, manager):
        """嵌套 dict 变更生成正确路径 (如 /data/config/enabled)."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"data": {"config": {"enabled": True, "name": "x"}}},
        )
        manager.register(art)
        manager.update(
            art.artifact_id,
            {"data": {"config": {"enabled": False, "name": "x"}}},
        )
        diff = manager.get_diff(art.artifact_id, from_version=1, to_version=2)
        ops_by_path = {op.get("path"): op for op in diff.ops}
        assert "/data/config/enabled" in ops_by_path
        assert ops_by_path["/data/config/enabled"]["op"] == "replace"
        assert ops_by_path["/data/config/enabled"]["value"] is False
        # name 未变更, 不应出现
        assert "/data/config/name" not in ops_by_path

    def test_list_element_change_generates_index_path(self, manager):
        """list 元素变更生成正确路径 (如 /items/0/name)."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"items": [{"name": "a", "v": 1}, {"name": "b", "v": 2}]},
        )
        manager.register(art)
        manager.update(
            art.artifact_id,
            {"items": [{"name": "a", "v": 1}, {"name": "B", "v": 2}]},
        )
        diff = manager.get_diff(art.artifact_id, from_version=1, to_version=2)
        ops_by_path = {op.get("path"): op for op in diff.ops}
        assert "/items/1/name" in ops_by_path
        assert ops_by_path["/items/1/name"]["op"] == "replace"
        assert ops_by_path["/items/1/name"]["value"] == "B"

    def test_list_add_element_generates_add(self, manager):
        """list 新增元素生成 add 操作."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"items": [1, 2]},
        )
        manager.register(art)
        manager.update(art.artifact_id, {"items": [1, 2, 3]})
        diff = manager.get_diff(art.artifact_id, from_version=1, to_version=2)
        ops_by_path = {op.get("path"): op for op in diff.ops}
        assert "/items/2" in ops_by_path
        assert ops_by_path["/items/2"]["op"] == "add"
        assert ops_by_path["/items/2"]["value"] == 3

    def test_list_remove_element_generates_remove(self, manager):
        """list 删除元素生成 remove 操作."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"items": [1, 2, 3]},
        )
        manager.register(art)
        manager.update(art.artifact_id, {"items": [1, 3]})
        diff = manager.get_diff(art.artifact_id, from_version=1, to_version=2)
        ops_by_path = {op.get("path"): op for op in diff.ops}
        # 索引 1: 2 -> 3 (replace), 索引 2: 3 被删除 (remove)
        assert "/items/1" in ops_by_path
        assert ops_by_path["/items/1"]["op"] == "replace"
        assert ops_by_path["/items/1"]["value"] == 3
        assert "/items/2" in ops_by_path
        assert ops_by_path["/items/2"]["op"] == "remove"

    def test_multi_level_nested_change_tracked(self, manager):
        """多层嵌套变更正确追踪 (如 /a/b/c/d)."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"a": {"b": {"c": {"d": 1, "e": 2}}}},
        )
        manager.register(art)
        manager.update(
            art.artifact_id,
            {"a": {"b": {"c": {"d": 999, "e": 2}}}},
        )
        diff = manager.get_diff(art.artifact_id, from_version=1, to_version=2)
        ops_by_path = {op.get("path"): op for op in diff.ops}
        assert "/a/b/c/d" in ops_by_path
        assert ops_by_path["/a/b/c/d"]["op"] == "replace"
        assert ops_by_path["/a/b/c/d"]["value"] == 999
        # e 未变更
        assert "/a/b/c/e" not in ops_by_path

    def test_no_change_returns_empty_ops(self, manager):
        """空变更返回空 ops 列表."""
        payload = {"a": 1, "b": {"c": 2}, "items": [1, 2, 3]}
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload=payload,
        )
        manager.register(art)
        # 用相同的 payload 更新 (深拷贝以避免共享引用干扰)
        import copy as _copy

        manager.update(art.artifact_id, _copy.deepcopy(payload))
        diff = manager.get_diff(art.artifact_id, from_version=1, to_version=2)
        assert diff.ops == []

    def test_type_change_generates_replace(self, manager):
        """类型变更 (dict -> str) 生成 replace."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"field": {"x": 1}},
        )
        manager.register(art)
        manager.update(art.artifact_id, {"field": "hello"})
        diff = manager.get_diff(art.artifact_id, from_version=1, to_version=2)
        ops_by_path = {op.get("path"): op for op in diff.ops}
        assert "field" in ops_by_path
        assert ops_by_path["field"]["op"] == "replace"
        assert ops_by_path["field"]["value"] == "hello"

    def test_nested_diff_roundtrip_applies_correctly(self, manager):
        """嵌套 diff 生成后可被 apply_edit 正确应用 (JSON Pointer)."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"config": {"enabled": True, "level": "beginner"}},
        )
        manager.register(art)
        manager.update(
            art.artifact_id,
            {"config": {"enabled": False, "level": "beginner"}},
        )
        # 取 v1 -> v2 的 diff (应含 /config/enabled replace)
        diff = manager.get_diff(art.artifact_id, from_version=1, to_version=2)
        # 应用到 v1 的副本 (新 artifact) 验证 JSON Pointer 可被 apply_edit 解析
        art2 = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"config": {"enabled": True, "level": "beginner"}},
        )
        manager.register(art2)
        edited = manager.apply_edit(art2.artifact_id, diff)
        assert edited.payload["config"]["enabled"] is False
        assert edited.payload["config"]["level"] == "beginner"


# ============================================================
# 3. 乐观版本锁测试
# ============================================================


class TestOptimisticVersionLock:
    """乐观版本锁测试 (update 新增 expected_version 参数)."""

    def test_update_with_matching_expected_version(self, manager, nested_artifact):
        """expected_version 匹配时正常更新."""
        manager.register(nested_artifact)
        assert nested_artifact.version == 1
        updated = manager.update(
            nested_artifact.artifact_id,
            {"content": "new"},
            expected_version=1,
        )
        assert updated.version == 2
        assert updated.payload["content"] == "new"

    def test_update_with_mismatched_expected_version_raises(
        self, manager, nested_artifact
    ):
        """expected_version 不匹配时抛出 VersionConflictError."""
        manager.register(nested_artifact)
        with pytest.raises(VersionConflictError) as exc_info:
            manager.update(
                nested_artifact.artifact_id,
                {"content": "new"},
                expected_version=99,
            )
        # 异常应携带 artifact_id 和实际版本号
        assert exc_info.value.artifact_id == nested_artifact.artifact_id
        assert exc_info.value.version == 1

    def test_update_without_expected_version_no_check(self, manager, nested_artifact):
        """expected_version=None 时不检查 (默认行为, 向后兼容)."""
        manager.register(nested_artifact)
        # 不传 expected_version
        updated = manager.update(nested_artifact.artifact_id, {"content": "v2"})
        assert updated.version == 2
        # 再次更新, 仍不检查
        updated2 = manager.update(nested_artifact.artifact_id, {"content": "v3"})
        assert updated2.version == 3

    def test_expected_version_none_explicit_no_check(self, manager, nested_artifact):
        """显式传 expected_version=None 时不检查."""
        manager.register(nested_artifact)
        manager.update(nested_artifact.artifact_id, {"content": "v2"})
        # 当前版本是 2, 但传 None 不检查, 应正常更新
        updated = manager.update(
            nested_artifact.artifact_id,
            {"content": "v3"},
            expected_version=None,
        )
        assert updated.version == 3

    def test_version_conflict_error_has_attributes(self):
        """VersionConflictError 包含 artifact_id 和 version 属性."""
        err = VersionConflictError("art-001", 5)
        assert err.artifact_id == "art-001"
        assert err.version == 5
        assert "art-001" in str(err)

    def test_version_lock_after_multiple_updates(self, manager, nested_artifact):
        """多次更新后, expected_version 必须匹配最新版本."""
        manager.register(nested_artifact)  # v1
        manager.update(nested_artifact.artifact_id, {"content": "v2"})  # v2
        manager.update(nested_artifact.artifact_id, {"content": "v3"})  # v3

        # 用过期的 expected_version=2 应失败
        with pytest.raises(VersionConflictError) as exc_info:
            manager.update(
                nested_artifact.artifact_id,
                {"content": "v4"},
                expected_version=2,
            )
        assert exc_info.value.version == 3

        # 用正确的 expected_version=3 应成功
        updated = manager.update(
            nested_artifact.artifact_id,
            {"content": "v4"},
            expected_version=3,
        )
        assert updated.version == 4

    def test_version_conflict_does_not_create_version(self, manager, nested_artifact):
        """版本冲突时不创建新版本."""
        manager.register(nested_artifact)  # v1
        with pytest.raises(VersionConflictError):
            manager.update(
                nested_artifact.artifact_id,
                {"content": "x"},
                expected_version=99,
            )
        # 最新版本仍是 1
        assert manager.get_latest_version(nested_artifact.artifact_id) == 1


# ============================================================
# 4. DiffOp / DiffOpType 模型集成测试
# ============================================================


class TestDiffOpIntegration:
    """DiffOp / DiffOpType 模型与增强操作的集成测试."""

    def test_diffop_type_enum_has_all_rfc6902_ops(self):
        """DiffOpType 枚举包含全部 RFC 6902 操作类型."""
        assert DiffOpType.ADD.value == "add"
        assert DiffOpType.REPLACE.value == "replace"
        assert DiffOpType.REMOVE.value == "remove"
        assert DiffOpType.MOVE.value == "move"
        assert DiffOpType.COPY.value == "copy"
        assert DiffOpType.TEST.value == "test"

    def test_diffop_move_object_accepted(self, manager):
        """apply_edit 接受 DiffOp 对象形式的 move 操作."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"src": "hello", "keep": 1},
        )
        manager.register(art)
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[
                DiffOp(op=DiffOpType.MOVE, path="/dst", value=None),
            ],
        )
        # DiffOp 模型没有 from 字段, 此处验证 move 用 dict 形式更合适;
        # 但确认 DiffOp 对象至少能被 apply_edit 接受不崩溃 (move 缺 from 会抛异常)
        with pytest.raises((L7Error, KeyError, TypeError)):
            manager.apply_edit(art.artifact_id, diff)

    def test_diffop_test_object_accepted(self, manager):
        """apply_edit 接受 dict 形式的 test 操作."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"x": 42},
        )
        manager.register(art)
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[
                {"op": "test", "path": "/x", "value": 42},
                {"op": "add", "path": "/y", "value": 100},
            ],
        )
        edited = manager.apply_edit(art.artifact_id, diff)
        assert edited.payload["y"] == 100


# ============================================================
# 5. 异常与边界测试
# ============================================================


class TestEdgeCases:
    """异常与边界情况测试."""

    def test_unknown_op_raises(self, manager, nested_artifact):
        """未知 op 类型应抛出异常."""
        manager.register(nested_artifact)
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[{"op": "foobar", "path": "/x", "value": 1}],
        )
        with pytest.raises(L7Error):
            manager.apply_edit(nested_artifact.artifact_id, diff)

    def test_apply_edit_preserves_other_versions(self, manager, nested_artifact):
        """apply_edit 嵌套编辑不破坏其他版本的快照."""
        manager.register(nested_artifact)  # v1
        manager.update(nested_artifact.artifact_id, dict(nested_artifact.payload))  # v2
        # 对 v2 做嵌套编辑
        diff = ArtifactDiff(
            artifact_id=nested_artifact.artifact_id,
            ops=[{"op": "replace", "path": "/config/level", "value": "advanced"}],
        )
        manager.apply_edit(nested_artifact.artifact_id, diff)  # v3
        # v1, v2 的 config.level 应保持原值
        assert manager.get_version(nested_artifact.artifact_id, 1).payload["config"]["level"] == "intermediate"
        assert manager.get_version(nested_artifact.artifact_id, 2).payload["config"]["level"] == "intermediate"
        assert manager.get_version(nested_artifact.artifact_id, 3).payload["config"]["level"] == "advanced"

    def test_json_pointer_escape_tilde(self, manager):
        """JSON Pointer 转义: ~1 -> /, ~0 -> ~."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"a/b": {"c": 1}, "x~y": 2},
        )
        manager.register(art)
        # /a~1b/c 应解析为 payload["a/b"]["c"]
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[{"op": "replace", "path": "/a~1b/c", "value": 99}],
        )
        edited = manager.apply_edit(art.artifact_id, diff)
        assert edited.payload["a/b"]["c"] == 99

    def test_move_within_list(self, manager):
        """move 操作在 list 元素间移动."""
        art = Artifact(
            type=ArtifactType.TEXT,
            source_agent="a",
            payload={"items": [1, 2, 3]},
        )
        manager.register(art)
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[{"op": "move", "path": "/moved", "from": "/items/1"}],
        )
        edited = manager.apply_edit(art.artifact_id, diff)
        assert edited.payload["items"] == [1, 3]
        assert edited.payload["moved"] == 2

    def test_get_diff_unknown_artifact_raises(self, manager):
        """get_diff 对未知 artifact 抛出 ArtifactNotFoundError."""
        with pytest.raises(ArtifactNotFoundError):
            manager.get_diff("art-missing", from_version=1, to_version=2)
