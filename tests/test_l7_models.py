"""L7 体验呈现层 — 核心数据模型测试.

覆盖 Artifact / DiffOp / ArtifactDiff / RenderContext / RenderDescriptor /
ArtifactVersionNode / VersionTree 全部模型。

测试领域: Dy3+ 发光材料 (YAG 基质, 4f-4f 跃迁, 480/574/660nm 发射)
"""
from __future__ import annotations

import time
import uuid

import pytest

from dy3_polaris.l7.exceptions import ArtifactValidationError
from dy3_polaris.l7.models import (
    ArtifactType,
    ArtifactLifecycleState,
    LearnerMode,
    DiffOpType,
    Artifact,
    DiffOp,
    ArtifactDiff,
    RenderContext,
    RenderDescriptor,
    ArtifactVersionNode,
    VersionTree,
)


# ============================================================
# ArtifactType 枚举
# ============================================================


class TestArtifactType:
    """ArtifactType 枚举测试."""

    def test_has_all_types(self):
        assert ArtifactType.TEXT == "text"
        assert ArtifactType.CHART == "chart"
        assert ArtifactType.GRAPH == "graph"
        assert ArtifactType.MOLECULE == "molecule"
        assert ArtifactType.TABLE == "table"
        assert ArtifactType.FORMULA == "formula"
        assert ArtifactType.PROVENANCE == "provenance"
        assert ArtifactType.INTERACTIVE == "interactive"

    def test_enum_count(self):
        assert len(ArtifactType) == 8

    def test_is_str_enum(self):
        assert isinstance(ArtifactType.TEXT, str)


# ============================================================
# ArtifactLifecycleState 枚举
# ============================================================


class TestArtifactLifecycleState:
    """ArtifactLifecycleState 枚举测试."""

    def test_has_all_states(self):
        assert ArtifactLifecycleState.CREATED == "created"
        assert ArtifactLifecycleState.RENDERED == "rendered"
        assert ArtifactLifecycleState.REVIEWED == "reviewed"
        assert ArtifactLifecycleState.EDITED == "edited"
        assert ArtifactLifecycleState.ARCHIVED == "archived"

    def test_enum_count(self):
        assert len(ArtifactLifecycleState) == 5

    def test_is_str_enum(self):
        assert isinstance(ArtifactLifecycleState.CREATED, str)


# ============================================================
# LearnerMode 枚举
# ============================================================


class TestLearnerMode:
    """LearnerMode 枚举测试."""

    def test_has_all_modes(self):
        assert LearnerMode.BEGINNER == "beginner"
        assert LearnerMode.INTERMEDIATE == "intermediate"
        assert LearnerMode.ADVANCED == "advanced"

    def test_is_str_enum(self):
        assert isinstance(LearnerMode.BEGINNER, str)


# ============================================================
# DiffOpType 枚举
# ============================================================


class TestDiffOpType:
    """DiffOpType 枚举测试 (JSON Patch RFC 6902)."""

    def test_has_all_ops(self):
        assert DiffOpType.ADD == "add"
        assert DiffOpType.REPLACE == "replace"
        assert DiffOpType.REMOVE == "remove"
        assert DiffOpType.MOVE == "move"
        assert DiffOpType.COPY == "copy"
        assert DiffOpType.TEST == "test"

    def test_enum_count(self):
        assert len(DiffOpType) == 6


# ============================================================
# Artifact 模型
# ============================================================


class TestArtifact:
    """Artifact 模型测试."""

    def test_auto_generates_artifact_id(self):
        """artifact_id 应自动生成，格式为 art-{uuid}."""
        art = Artifact(type=ArtifactType.TEXT)
        assert art.artifact_id.startswith("art-")
        # art- 后面应跟随 hex 字符
        suffix = art.artifact_id[4:]
        assert len(suffix) > 0
        # 两个 Artifact 的 ID 应不同
        art2 = Artifact(type=ArtifactType.TEXT)
        assert art.artifact_id != art2.artifact_id

    def test_has_all_required_fields(self):
        """Artifact 应包含所有必需字段."""
        art = Artifact(type=ArtifactType.CHART, mime="application/vnd.dy3.chart+json")
        assert hasattr(art, "artifact_id")
        assert hasattr(art, "type")
        assert hasattr(art, "mime")
        assert hasattr(art, "source_agent")
        assert hasattr(art, "provenance_chain")
        assert hasattr(art, "learner_context")
        assert hasattr(art, "version")
        assert hasattr(art, "editable")
        assert hasattr(art, "fork_origin")
        assert hasattr(art, "payload")
        assert hasattr(art, "created_at")
        assert hasattr(art, "updated_at")

    def test_default_field_values(self):
        """默认值测试."""
        art = Artifact(type=ArtifactType.TEXT)
        assert art.type == ArtifactType.TEXT
        assert art.source_agent == ""
        assert art.provenance_chain == []
        assert art.learner_context == {}
        assert art.version == 1
        assert art.editable is True
        assert art.fork_origin is None
        assert art.payload == {}
        assert art.created_at > 0
        assert art.updated_at > 0

    def test_custom_field_values(self):
        """自定义字段值测试."""
        now = time.time()
        art = Artifact(
            type=ArtifactType.GRAPH,
            mime="application/vnd.dy3.graph+json",
            source_agent="agent-reasoner-01",
            provenance_chain=["kpa-001", "kpa-002"],
            learner_context={"level": "intermediate", "kp": ["kp-1", "kp-2"]},
            version=3,
            editable=False,
            fork_origin="art-original123",
            payload={"nodes": [], "edges": []},
            created_at=now,
            updated_at=now,
        )
        assert art.type == ArtifactType.GRAPH
        assert art.mime == "application/vnd.dy3.graph+json"
        assert art.source_agent == "agent-reasoner-01"
        assert art.provenance_chain == ["kpa-001", "kpa-002"]
        assert art.learner_context == {"level": "intermediate", "kp": ["kp-1", "kp-2"]}
        assert art.version == 3
        assert art.editable is False
        assert art.fork_origin == "art-original123"
        assert art.payload == {"nodes": [], "edges": []}

    # --- mime_to_type classmethod ---

    def test_mime_to_type_text(self):
        assert Artifact.mime_to_type("text/vnd.dy3+markdown") == ArtifactType.TEXT

    def test_mime_to_type_chart(self):
        assert Artifact.mime_to_type("application/vnd.dy3.chart+json") == ArtifactType.CHART

    def test_mime_to_type_graph(self):
        assert Artifact.mime_to_type("application/vnd.dy3.graph+json") == ArtifactType.GRAPH

    def test_mime_to_type_molecule(self):
        assert Artifact.mime_to_type("chemical/x-mdl-molfile") == ArtifactType.MOLECULE

    def test_mime_to_type_table(self):
        assert Artifact.mime_to_type("application/vnd.dy3.table+json") == ArtifactType.TABLE

    def test_mime_to_type_formula(self):
        assert Artifact.mime_to_type("application/vnd.dy3.formula+json") == ArtifactType.FORMULA

    def test_mime_to_type_provenance(self):
        assert Artifact.mime_to_type("application/vnd.dy3.provenance+json") == ArtifactType.PROVENANCE

    def test_mime_to_type_interactive(self):
        assert Artifact.mime_to_type("application/vnd.dy3.interactive+json") == ArtifactType.INTERACTIVE

    def test_mime_to_type_unknown_returns_none(self):
        """未知 MIME 类型应返回 None."""
        assert Artifact.mime_to_type("application/pdf") is None
        assert Artifact.mime_to_type("unknown/type") is None

    # --- validate method ---

    def test_validate_text_valid(self):
        art = Artifact(
            type=ArtifactType.TEXT,
            mime="text/vnd.dy3+markdown",
            payload={"content": "# Dy3+ 发光机制\n4f-4f 跃迁"},
        )
        assert art.validate() is True

    def test_validate_text_invalid(self):
        art = Artifact(
            type=ArtifactType.TEXT,
            mime="text/vnd.dy3+markdown",
            payload={"wrong_key": "missing content"},
        )
        with pytest.raises(ArtifactValidationError) as exc_info:
            art.validate()
        assert "content" in exc_info.value.missing_fields

    def test_validate_chart_valid(self):
        art = Artifact(
            type=ArtifactType.CHART,
            mime="application/vnd.dy3.chart+json",
            payload={"chart_type": "bar", "data": [{"x": "480nm", "y": 0.8}]},
        )
        assert art.validate() is True

    def test_validate_chart_invalid(self):
        art = Artifact(
            type=ArtifactType.CHART,
            mime="application/vnd.dy3.chart+json",
            payload={"chart_type": "bar"},  # missing "data"
        )
        with pytest.raises(ArtifactValidationError):
            art.validate()

    def test_validate_graph_valid(self):
        art = Artifact(
            type=ArtifactType.GRAPH,
            mime="application/vnd.dy3.graph+json",
            payload={"nodes": [{"id": "n1"}], "edges": [{"source": "n1", "target": "n2"}]},
        )
        assert art.validate() is True

    def test_validate_graph_invalid(self):
        art = Artifact(
            type=ArtifactType.GRAPH,
            mime="application/vnd.dy3.graph+json",
            payload={"nodes": []},  # missing "edges"
        )
        with pytest.raises(ArtifactValidationError):
            art.validate()

    def test_validate_molecule_valid(self):
        art = Artifact(
            type=ArtifactType.MOLECULE,
            mime="chemical/x-mdl-molfile",
            payload={"molfile": "Dy3+ YAG structure"},
        )
        assert art.validate() is True

    def test_validate_molecule_valid_smiles(self):
        art = Artifact(
            type=ArtifactType.MOLECULE,
            mime="chemical/x-mdl-molfile",
            payload={"smiles": "[Dy]"},
        )
        assert art.validate() is True

    def test_validate_table_valid(self):
        art = Artifact(
            type=ArtifactType.TABLE,
            mime="application/vnd.dy3.table+json",
            payload={"headers": ["波长", "强度"], "rows": [["480nm", "0.8"]]},
        )
        assert art.validate() is True

    def test_validate_table_invalid(self):
        art = Artifact(
            type=ArtifactType.TABLE,
            mime="application/vnd.dy3.table+json",
            payload={"headers": ["a"]},  # missing "rows"
        )
        with pytest.raises(ArtifactValidationError):
            art.validate()

    def test_validate_formula_valid(self):
        art = Artifact(
            type=ArtifactType.FORMULA,
            mime="application/vnd.dy3.formula+json",
            payload={"latex": r"E = h\nu"},
        )
        assert art.validate() is True

    def test_validate_provenance_valid(self):
        art = Artifact(
            type=ArtifactType.PROVENANCE,
            mime="application/vnd.dy3.provenance+json",
            payload={"chain": [{"kpa_id": "kpa-001"}]},
        )
        assert art.validate() is True

    def test_validate_interactive_valid(self):
        art = Artifact(
            type=ArtifactType.INTERACTIVE,
            mime="application/vnd.dy3.interactive+json",
            payload={"widget_type": "simulator"},
        )
        assert art.validate() is True

    def test_validate_interactive_invalid(self):
        art = Artifact(
            type=ArtifactType.INTERACTIVE,
            mime="application/vnd.dy3.interactive+json",
            payload={"wrong": "no widget_type"},
        )
        with pytest.raises(ArtifactValidationError):
            art.validate()

    # --- to_dict method ---

    def test_to_dict_returns_dict(self):
        art = Artifact(
            type=ArtifactType.CHART,
            mime="application/vnd.dy3.chart+json",
            source_agent="agent-1",
            payload={"chart_type": "bar", "data": []},
        )
        d = art.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_serializes_type_as_string(self):
        art = Artifact(type=ArtifactType.GRAPH)
        d = art.to_dict()
        assert d["type"] == "graph"

    def test_to_dict_includes_all_fields(self):
        art = Artifact(
            type=ArtifactType.TEXT,
            mime="text/vnd.dy3+markdown",
            source_agent="agent-1",
            provenance_chain=["kpa-1"],
            learner_context={"level": "beginner"},
            version=2,
            editable=False,
            fork_origin="art-orig",
            payload={"content": "hello"},
        )
        d = art.to_dict()
        assert d["artifact_id"] == art.artifact_id
        assert d["type"] == "text"
        assert d["mime"] == "text/vnd.dy3+markdown"
        assert d["source_agent"] == "agent-1"
        assert d["provenance_chain"] == ["kpa-1"]
        assert d["learner_context"] == {"level": "beginner"}
        assert d["version"] == 2
        assert d["editable"] is False
        assert d["fork_origin"] == "art-orig"
        assert d["payload"] == {"content": "hello"}
        assert "created_at" in d
        assert "updated_at" in d


# ============================================================
# DiffOp 模型
# ============================================================


class TestDiffOp:
    """DiffOp 模型测试 (JSON Patch RFC 6902)."""

    def test_has_op_path_value_fields(self):
        op = DiffOp(op=DiffOpType.ADD, path="/payload/content", value="new content")
        assert op.op == DiffOpType.ADD
        assert op.path == "/payload/content"
        assert op.value == "new content"

    def test_op_accepts_string(self):
        op = DiffOp(op="replace", path="/payload/title", value="new title")
        assert op.op == DiffOpType.REPLACE

    def test_value_optional_for_remove(self):
        op = DiffOp(op=DiffOpType.REMOVE, path="/payload/old_field")
        assert op.op == DiffOpType.REMOVE
        assert op.path == "/payload/old_field"

    def test_all_op_types(self):
        for op_type in DiffOpType:
            op = DiffOp(op=op_type, path="/test", value=None)
            assert op.op == op_type

    def test_to_dict(self):
        op = DiffOp(op=DiffOpType.REPLACE, path="/payload/content", value="updated")
        d = op.to_dict()
        assert d["op"] == "replace"
        assert d["path"] == "/payload/content"
        assert d["value"] == "updated"


# ============================================================
# ArtifactDiff 模型
# ============================================================


class TestArtifactDiff:
    """ArtifactDiff 模型测试."""

    def test_has_required_fields(self):
        diff = ArtifactDiff(artifact_id="art-abc123")
        assert diff.artifact_id == "art-abc123"
        assert diff.ops == []
        assert diff.edit_reason == ""
        assert diff.created_at > 0

    def test_ops_list_of_diffop(self):
        ops = [
            DiffOp(op=DiffOpType.ADD, path="/payload/content", value="new"),
            DiffOp(op=DiffOpType.REMOVE, path="/payload/old"),
        ]
        diff = ArtifactDiff(artifact_id="art-1", ops=ops, edit_reason="fix typo")
        assert len(diff.ops) == 2
        assert diff.ops[0].op == DiffOpType.ADD
        assert diff.ops[1].op == DiffOpType.REMOVE
        assert diff.edit_reason == "fix typo"

    def test_auto_created_at(self):
        diff = ArtifactDiff(artifact_id="art-1")
        assert isinstance(diff.created_at, float)
        assert diff.created_at > 0

    def test_to_dict(self):
        ops = [DiffOp(op=DiffOpType.REPLACE, path="/payload/content", value="updated")]
        diff = ArtifactDiff(artifact_id="art-1", ops=ops, edit_reason="update")
        d = diff.to_dict()
        assert d["artifact_id"] == "art-1"
        assert d["edit_reason"] == "update"
        assert len(d["ops"]) == 1
        assert d["ops"][0]["op"] == "replace"


# ============================================================
# RenderContext 模型
# ============================================================


class TestRenderContext:
    """RenderContext 模型测试."""

    def test_has_all_fields(self):
        ctx = RenderContext()
        assert hasattr(ctx, "viewport")
        assert hasattr(ctx, "theme")
        assert hasattr(ctx, "learner_mode")
        assert hasattr(ctx, "bkt_state")
        assert hasattr(ctx, "kp_ids")
        assert hasattr(ctx, "locale")

    def test_default_values(self):
        ctx = RenderContext()
        assert ctx.viewport.width > 0
        assert ctx.viewport.height > 0
        assert ctx.theme == "light"
        assert ctx.learner_mode == LearnerMode.INTERMEDIATE
        assert ctx.bkt_state == {}
        assert ctx.kp_ids == []
        assert ctx.locale == "zh-CN"

    def test_custom_values(self):
        ctx = RenderContext(
            viewport={"width": 1920, "height": 1080},
            theme="dark",
            learner_mode=LearnerMode.ADVANCED,
            bkt_state={"kp-1": 0.85},
            kp_ids=["kp-1", "kp-2"],
            locale="en-US",
        )
        assert ctx.viewport.width == 1920
        assert ctx.viewport.height == 1080
        assert ctx.theme == "dark"
        assert ctx.learner_mode == LearnerMode.ADVANCED
        assert ctx.bkt_state == {"kp-1": 0.85}
        assert ctx.kp_ids == ["kp-1", "kp-2"]
        assert ctx.locale == "en-US"

    def test_learner_mode_accepts_string(self):
        ctx = RenderContext(learner_mode="beginner")
        assert ctx.learner_mode == LearnerMode.BEGINNER


# ============================================================
# RenderDescriptor 模型
# ============================================================


class TestRenderDescriptor:
    """RenderDescriptor 模型测试."""

    def test_has_all_fields(self):
        rd = RenderDescriptor(artifact_id="art-1", mime="text/vnd.dy3+markdown")
        assert hasattr(rd, "render_id")
        assert hasattr(rd, "artifact_id")
        assert hasattr(rd, "mime")
        assert hasattr(rd, "html")
        assert hasattr(rd, "config")
        assert hasattr(rd, "assets")
        assert hasattr(rd, "metadata")
        assert hasattr(rd, "rendered_at")
        assert hasattr(rd, "render_time_ms")

    def test_auto_generates_render_id(self):
        rd = RenderDescriptor(artifact_id="art-1", mime="text/vnd.dy3+markdown")
        assert rd.render_id.startswith("rd-")

    def test_html_optional(self):
        """html 字段可选，默认为 None."""
        rd = RenderDescriptor(artifact_id="art-1", mime="text/vnd.dy3+markdown")
        assert rd.html is None

    def test_custom_values(self):
        now = time.time()
        rd = RenderDescriptor(
            artifact_id="art-1",
            mime="application/vnd.dy3.chart+json",
            html="<div id='chart'></div>",
            config={"renderer": "echarts"},
            assets=["https://cdn.example.com/echarts.min.js"],
            metadata={"dpr": 2},
            rendered_at=now,
            render_time_ms=42.5,
        )
        assert rd.artifact_id == "art-1"
        assert rd.mime == "application/vnd.dy3.chart+json"
        assert rd.html == "<div id='chart'></div>"
        assert rd.config == {"renderer": "echarts"}
        assert rd.assets == ["https://cdn.example.com/echarts.min.js"]
        assert rd.metadata == {"dpr": 2}
        assert rd.rendered_at == now
        assert rd.render_time_ms == 42.5

    def test_default_config_and_assets(self):
        rd = RenderDescriptor(artifact_id="art-1", mime="text/plain")
        assert rd.config == {}
        assert rd.assets == []
        assert rd.metadata == {}
        assert rd.render_time_ms == 0.0

    def test_to_dict(self):
        rd = RenderDescriptor(
            artifact_id="art-1",
            mime="text/vnd.dy3+markdown",
            html="<p>hello</p>",
            render_time_ms=10.0,
        )
        d = rd.to_dict()
        assert d["artifact_id"] == "art-1"
        assert d["mime"] == "text/vnd.dy3+markdown"
        assert d["html"] == "<p>hello</p>"
        assert d["render_time_ms"] == 10.0


# ============================================================
# ArtifactVersionNode 模型
# ============================================================


class TestArtifactVersionNode:
    """ArtifactVersionNode 模型测试."""

    def test_has_all_fields(self):
        node = ArtifactVersionNode(version=1, artifact_id="art-1")
        assert hasattr(node, "version")
        assert hasattr(node, "artifact_id")
        assert hasattr(node, "parent_version")
        assert hasattr(node, "fork_origin")
        assert hasattr(node, "created_at")

    def test_default_optional_fields(self):
        node = ArtifactVersionNode(version=1, artifact_id="art-1")
        assert node.version == 1
        assert node.artifact_id == "art-1"
        assert node.parent_version is None
        assert node.fork_origin is None
        assert node.created_at > 0

    def test_custom_fields(self):
        node = ArtifactVersionNode(
            version=2,
            artifact_id="art-1",
            parent_version=1,
            fork_origin="art-orig-v3",
        )
        assert node.version == 2
        assert node.parent_version == 1
        assert node.fork_origin == "art-orig-v3"

    def test_to_dict(self):
        node = ArtifactVersionNode(version=3, artifact_id="art-1", parent_version=2)
        d = node.to_dict()
        assert d["version"] == 3
        assert d["artifact_id"] == "art-1"
        assert d["parent_version"] == 2
        assert d["fork_origin"] is None


# ============================================================
# VersionTree 模型
# ============================================================


class TestVersionTree:
    """VersionTree 版本树 (DAG) 测试."""

    def test_empty_tree(self):
        tree = VersionTree(artifact_id="art-1")
        assert tree.artifact_id == "art-1"
        assert tree.get_latest_version() is None
        assert tree.get_lineage(1) == []

    def test_add_single_version(self):
        tree = VersionTree(artifact_id="art-1")
        node = ArtifactVersionNode(version=1, artifact_id="art-1")
        tree.add_version(node)
        assert tree.get_latest_version() == 1
        assert tree.get_lineage(1) == [1]

    def test_add_linear_chain(self):
        """线性版本链: v1 → v2 → v3."""
        tree = VersionTree(artifact_id="art-1")
        tree.add_version(ArtifactVersionNode(version=1, artifact_id="art-1"))
        tree.add_version(ArtifactVersionNode(version=2, artifact_id="art-1", parent_version=1))
        tree.add_version(ArtifactVersionNode(version=3, artifact_id="art-1", parent_version=2))

        assert tree.get_latest_version() == 3
        assert tree.get_lineage(3) == [1, 2, 3]
        assert tree.get_lineage(2) == [1, 2]
        assert tree.get_lineage(1) == [1]

    def test_add_forked_version(self):
        """分叉版本: v1 → v2 → v3, v2 → v4 (fork)."""
        tree = VersionTree(artifact_id="art-1")
        tree.add_version(ArtifactVersionNode(version=1, artifact_id="art-1"))
        tree.add_version(ArtifactVersionNode(version=2, artifact_id="art-1", parent_version=1))
        tree.add_version(ArtifactVersionNode(version=3, artifact_id="art-1", parent_version=2))
        tree.add_version(ArtifactVersionNode(version=4, artifact_id="art-1", parent_version=2, fork_origin="art-orig-v3"))

        assert tree.get_latest_version() == 4
        assert tree.get_lineage(4) == [1, 2, 4]
        assert tree.get_lineage(3) == [1, 2, 3]

    def test_get_lineage_nonexistent_version(self):
        tree = VersionTree(artifact_id="art-1")
        tree.add_version(ArtifactVersionNode(version=1, artifact_id="art-1"))
        assert tree.get_lineage(99) == []

    def test_get_all_versions(self):
        tree = VersionTree(artifact_id="art-1")
        tree.add_version(ArtifactVersionNode(version=1, artifact_id="art-1"))
        tree.add_version(ArtifactVersionNode(version=2, artifact_id="art-1", parent_version=1))
        all_versions = tree.get_all_versions()
        assert sorted(all_versions) == [1, 2]

    def test_get_version_node(self):
        tree = VersionTree(artifact_id="art-1")
        node = ArtifactVersionNode(version=1, artifact_id="art-1")
        tree.add_version(node)
        retrieved = tree.get_version_node(1)
        assert retrieved is not None
        assert retrieved.version == 1
        assert tree.get_version_node(99) is None

    def test_add_duplicate_version_raises(self):
        tree = VersionTree(artifact_id="art-1")
        tree.add_version(ArtifactVersionNode(version=1, artifact_id="art-1"))
        with pytest.raises(ValueError):
            tree.add_version(ArtifactVersionNode(version=1, artifact_id="art-1"))

    def test_get_children(self):
        """获取某版本的所有直接子版本."""
        tree = VersionTree(artifact_id="art-1")
        tree.add_version(ArtifactVersionNode(version=1, artifact_id="art-1"))
        tree.add_version(ArtifactVersionNode(version=2, artifact_id="art-1", parent_version=1))
        tree.add_version(ArtifactVersionNode(version=3, artifact_id="art-1", parent_version=1))
        children = tree.get_children(1)
        assert sorted(children) == [2, 3]
        assert tree.get_children(2) == []

    def test_version_count(self):
        tree = VersionTree(artifact_id="art-1")
        for v in range(1, 6):
            tree.add_version(ArtifactVersionNode(version=v, artifact_id="art-1", parent_version=v - 1 if v > 1 else None))
        assert tree.version_count() == 5
