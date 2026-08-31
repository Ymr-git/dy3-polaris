"""产物管理模块测试 — TDD 测试用例.

测试覆盖:
1. Artifact — 不可变产物载体 (元数据 + payload + 溯源链)
2. ArtifactType — 产物类型枚举 (8 种类型)
3. ArtifactState — 产物生命周期状态 (5 阶段状态机)
4. ArtifactVersion — 产物版本记录 (版本号 + 内容哈希 + CC1 状态)
5. ArtifactStore — 抽象存储接口 (save/load/list/delete/versions)
6. InMemoryArtifactStore — 内存存储实现
7. ArtifactManager — 产物管理器 (创建/更新/版本/搜索/归档/溯源)
8. ArtifactEdit — 编辑操作记录 (编辑意图/状态/审核)
9. ArtifactProvenance — 产物溯源记录 (actor_chain + code_hash)
10. 集成测试 — 与 SessionManager/OrchestrationEngine/Communication 联动
11. 错误处理 — 产物异常与恢复

融合世界先进方案:
- LangGraph: Store API + 层次命名空间 + 向量搜索 + TTL
- OpenAI Agents SDK: RunResult 多面结果对象 + to_state() 快照
- Google ADK: BaseArtifactService + 版本自动递增 + MIME 类型 + session/user 作用域
- CrewAI: Task Output 声明式配置 + 任务链传递
- AutoGen: State 序列化 + version 兼容字段
- Temporal: Event Sourcing + Query/Signal/Update 三态分离
- Claude Science: 五维度溯源 + Execution Log 权威优先 + 版本 diff
- L5 设计文档: Artifact-Edit Channel + 五阶段生命周期 + DAG 版本树 + CC1 审核
- L7 设计文档: 三级缓存存储 + 搜索过滤 + 编辑权限控制
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from dy3_polaris.l5.artifact_manager import (
    Artifact,
    ArtifactEdit,
    ArtifactEditState,
    ArtifactError,
    ArtifactManager,
    ArtifactNotFoundError,
    ArtifactProvenance,
    ArtifactState,
    ArtifactStore,
    ArtifactType,
    ArtifactVersion,
    InMemoryArtifactStore,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def artifact_store():
    """创建内存产物存储实例."""
    return InMemoryArtifactStore()


@pytest.fixture
def artifact_manager(artifact_store):
    """创建产物管理器实例."""
    return ArtifactManager(store=artifact_store)


@pytest.fixture
def sample_artifact_data():
    """样本产物数据 (学情诊断报告)."""
    return {
        "report_id": "rpt-001",
        "learner_id": "stu-001",
        "kp_gaps": ["KP-12", "KP-18"],
        "mastery_vector": {"KP-12": 0.35, "KP-18": 0.28},
        "confidence": 0.87,
    }


# ============================================================
# 1. Artifact 测试
# ============================================================


class TestArtifact:
    """不可变产物载体测试 (统一元数据格式)."""

    def test_artifact_creation(self, sample_artifact_data):
        """创建产物应自动生成 ID 和时间戳."""
        art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload=sample_artifact_data,
        )
        assert art.artifact_id.startswith("art-")
        assert art.artifact_type == ArtifactType.TEXT
        assert art.source_agent == "agent.diagnosis"
        assert art.payload["report_id"] == "rpt-001"
        assert art.created_at > 0
        assert art.version == 1
        assert art.state == ArtifactState.CREATED
        assert art.editable is True
        assert art.fork_origin is None
        assert art.mime == "application/json"

    def test_artifact_with_custom_id(self, sample_artifact_data):
        """产物可指定自定义 ID."""
        art = Artifact(
            artifact_id="art-custom-001",
            artifact_type=ArtifactType.CHART,
            source_agent="agent.generation",
            payload={"chart_type": "bar"},
        )
        assert art.artifact_id == "art-custom-001"

    def test_artifact_immutability(self, sample_artifact_data):
        """产物应是不可变的 (frozen)."""
        art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload=sample_artifact_data,
        )
        with pytest.raises((AttributeError, TypeError)):
            art.artifact_type = ArtifactType.CHART

    def test_artifact_with_mime(self):
        """产物可指定 MIME 类型 (ADK 模式)."""
        art = Artifact(
            artifact_type=ArtifactType.MOLECULE,
            source_agent="agent.generation",
            payload={"smiles": "c1ccccc1"},
            mime="chemical/x-mdl-molfile",
        )
        assert art.mime == "chemical/x-mdl-molfile"

    def test_artifact_default_mime_by_type(self):
        """不同类型产物应有默认 MIME (ADK 模式)."""
        text_art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        assert text_art.mime == "application/json"

        chart_art = Artifact(
            artifact_type=ArtifactType.CHART,
            source_agent="agent.test",
            payload={},
        )
        assert chart_art.mime == "application/json"

    def test_artifact_with_provenance_chain(self, sample_artifact_data):
        """产物应携带溯源链 (L5/L7 设计文档)."""
        art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload=sample_artifact_data,
            provenance_chain=["kpa-001", "kpa-002"],
        )
        assert len(art.provenance_chain) == 2
        assert art.provenance_chain[0] == "kpa-001"

    def test_artifact_with_learner_context(self, sample_artifact_data):
        """产物应携带学习上下文 (L7 设计文档)."""
        art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload=sample_artifact_data,
            learner_context={
                "learner_id": "stu-001",
                "kp_id": "KP-12",
                "bkt_mastery": 0.35,
            },
        )
        assert art.learner_context["learner_id"] == "stu-001"
        assert art.learner_context["bkt_mastery"] == 0.35

    def test_artifact_with_fork_origin(self, sample_artifact_data):
        """产物可标记 Fork 来源 (L5 Session Fork 集成)."""
        art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation",
            payload=sample_artifact_data,
            fork_origin="sess-fork-001",
        )
        assert art.fork_origin == "sess-fork-001"

    def test_artifact_not_editable(self):
        """CC2 审批通过的产物应为只读 (L7 设计文档)."""
        art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation",
            payload={},
            editable=False,
        )
        assert art.editable is False

    def test_artifact_to_dict(self, sample_artifact_data):
        """产物应可序列化为字典."""
        art = Artifact(
            artifact_type=ArtifactType.CHART,
            source_agent="agent.generation",
            payload=sample_artifact_data,
        )
        d = art.to_dict()
        assert d["artifact_type"] == "chart"
        assert d["source_agent"] == "agent.generation"
        assert d["payload"]["report_id"] == "rpt-001"
        assert "artifact_id" in d
        assert "created_at" in d
        assert "version" in d


# ============================================================
# 2. ArtifactType 测试
# ============================================================


class TestArtifactType:
    """产物类型枚举测试 (L7 设计文档 8 种类型)."""

    def test_all_types_defined(self):
        """8 种产物类型应全部定义."""
        assert ArtifactType.TEXT == "text"
        assert ArtifactType.CHART == "chart"
        assert ArtifactType.GRAPH == "graph"
        assert ArtifactType.MOLECULE == "molecule"
        assert ArtifactType.TABLE == "table"
        assert ArtifactType.FORMULA == "formula"
        assert ArtifactType.PROVENANCE == "provenance"
        assert ArtifactType.INTERACTIVE == "interactive"

    def test_type_from_string(self):
        """应支持从字符串创建枚举."""
        assert ArtifactType("text") == ArtifactType.TEXT
        assert ArtifactType("chart") == ArtifactType.CHART


# ============================================================
# 3. ArtifactState 测试
# ============================================================


class TestArtifactState:
    """产物生命周期状态测试 (L7 设计文档 5 阶段状态机)."""

    def test_all_states_defined(self):
        """5 种生命周期状态应全部定义."""
        assert ArtifactState.CREATED == "created"
        assert ArtifactState.RENDERED == "rendered"
        assert ArtifactState.REVIEWED == "reviewed"
        assert ArtifactState.EDITED == "edited"
        assert ArtifactState.ARCHIVED == "archived"

    def test_state_transitions(self):
        """状态转换: CREATED → RENDERED → REVIEWED → EDITED → RENDERED."""
        art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        assert art.state == ArtifactState.CREATED

        art.transition_to(ArtifactState.RENDERED)
        assert art.state == ArtifactState.RENDERED

        art.transition_to(ArtifactState.REVIEWED)
        assert art.state == ArtifactState.REVIEWED

        art.transition_to(ArtifactState.EDITED)
        assert art.state == ArtifactState.EDITED

        art.transition_to(ArtifactState.RENDERED)
        assert art.state == ArtifactState.RENDERED

    def test_invalid_transition_raises(self):
        """非法状态转换应抛异常 (ARCHIVED → CREATED)."""
        art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        art.transition_to(ArtifactState.ARCHIVED)
        with pytest.raises(ArtifactError, match="invalid.*transition"):
            art.transition_to(ArtifactState.CREATED)

    def test_archive_is_terminal(self):
        """ARCHIVED 是终态,不能再转换."""
        art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        art.transition_to(ArtifactState.ARCHIVED)
        with pytest.raises(ArtifactError, match="invalid.*transition"):
            art.transition_to(ArtifactState.RENDERED)


# ============================================================
# 4. ArtifactVersion 测试
# ============================================================


class TestArtifactVersion:
    """产物版本记录测试 (ADK 版本自动递增 + Claude Science 内容哈希)."""

    def test_version_creation(self):
        """创建版本记录."""
        ver = ArtifactVersion(
            artifact_id="art-001",
            version=1,
            content_hash="sha256:abc123",
            created_by="agent.diagnosis",
        )
        assert ver.artifact_id == "art-001"
        assert ver.version == 1
        assert ver.content_hash == "sha256:abc123"
        assert ver.created_by == "agent.diagnosis"
        assert ver.cc1_status == "pending"
        assert ver.created_at > 0

    def test_version_with_edit_operation(self):
        """版本可携带编辑操作记录 (L5 设计文档)."""
        ver = ArtifactVersion(
            artifact_id="art-001",
            version=2,
            content_hash="sha256:def456",
            created_by="agent.generation",
            edit_operation={"type": "axis_change", "axis": "x", "scale": "log"},
        )
        assert ver.edit_operation["type"] == "axis_change"
        assert ver.edit_operation["scale"] == "log"

    def test_version_cc1_status(self):
        """版本可记录 CC1 审核状态 (L5 设计文档)."""
        ver = ArtifactVersion(
            artifact_id="art-001",
            version=1,
            content_hash="sha256:abc123",
            created_by="agent.diagnosis",
            cc1_status="pass",
        )
        assert ver.cc1_status == "pass"

    def test_version_with_output_ref(self):
        """版本可携带产物存储引用 (L5 DDL output_ref)."""
        ver = ArtifactVersion(
            artifact_id="art-001",
            version=1,
            content_hash="sha256:abc123",
            created_by="agent.diagnosis",
            output_ref="s3://bucket/art-001/v1",
        )
        assert ver.output_ref == "s3://bucket/art-001/v1"

    def test_version_with_data_hash(self):
        """版本可携带数据哈希 (L5 DDL data_hash)."""
        ver = ArtifactVersion(
            artifact_id="art-001",
            version=1,
            content_hash="sha256:abc123",
            data_hash="sha256:data789",
            created_by="agent.diagnosis",
        )
        assert ver.data_hash == "sha256:data789"


# ============================================================
# 5. ArtifactStore 测试 (抽象接口)
# ============================================================


class TestArtifactStore:
    """产物存储抽象接口测试 (ADK BaseArtifactService 模式)."""

    def test_store_is_abstract(self):
        """ArtifactStore 应是抽象类."""
        with pytest.raises(TypeError):
            ArtifactStore()  # type: ignore[abstract]

    def test_abstract_methods_defined(self):
        """抽象方法应包含 save/load/list/delete/list_versions."""
        abstract_methods = ArtifactStore.__abstractmethods__
        assert "save" in abstract_methods
        assert "load" in abstract_methods
        assert "list_artifacts" in abstract_methods
        assert "delete" in abstract_methods
        assert "list_versions" in abstract_methods


# ============================================================
# 6. InMemoryArtifactStore 测试
# ============================================================


class TestInMemoryArtifactStore:
    """内存产物存储实现测试."""

    def test_save_artifact(self, artifact_store, sample_artifact_data):
        """保存产物应返回版本号 (ADK 模式)."""
        art = Artifact(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload=sample_artifact_data,
        )
        version = artifact_store.save(art)
        assert version == 1

    def test_save_artifact_increments_version(self, artifact_store):
        """同一产物多次保存应递增版本号 (ADK 模式)."""
        art = Artifact(
            artifact_id="art-test-001",
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={"v": 1},
        )
        v1 = artifact_store.save(art)
        assert v1 == 1

        art2 = Artifact(
            artifact_id="art-test-001",
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={"v": 2},
            version=2,
        )
        v2 = artifact_store.save(art2)
        assert v2 == 2

    def test_load_artifact(self, artifact_store, sample_artifact_data):
        """加载产物."""
        art = Artifact(
            artifact_id="art-load-001",
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload=sample_artifact_data,
        )
        artifact_store.save(art)

        loaded = artifact_store.load("art-load-001")
        assert loaded is not None
        assert loaded.artifact_id == "art-load-001"
        assert loaded.payload["report_id"] == "rpt-001"

    def test_load_specific_version(self, artifact_store):
        """加载特定版本的产物 (ADK 模式)."""
        for i in range(3):
            art = Artifact(
                artifact_id="art-ver-001",
                artifact_type=ArtifactType.TEXT,
                source_agent="agent.test",
                payload={"version": i + 1},
                version=i + 1,
            )
            artifact_store.save(art)

        v2 = artifact_store.load("art-ver-001", version=2)
        assert v2 is not None
        assert v2.payload["version"] == 2

    def test_load_nonexistent_returns_none(self, artifact_store):
        """加载不存在的产物返回 None."""
        assert artifact_store.load("nonexistent") is None

    def test_load_nonexistent_version_returns_none(self, artifact_store):
        """加载不存在的版本返回 None."""
        art = Artifact(
            artifact_id="art-test-002",
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        artifact_store.save(art)
        assert artifact_store.load("art-test-002", version=99) is None

    def test_list_artifacts(self, artifact_store):
        """列出所有产物 ID."""
        for i in range(3):
            art = Artifact(
                artifact_id=f"art-list-{i:03d}",
                artifact_type=ArtifactType.TEXT,
                source_agent="agent.test",
                payload={},
            )
            artifact_store.save(art)

        keys = artifact_store.list_artifacts()
        assert len(keys) == 3
        assert "art-list-000" in keys
        assert "art-list-002" in keys

    def test_list_versions(self, artifact_store):
        """列出产物的所有版本号 (ADK 模式)."""
        for i in range(3):
            art = Artifact(
                artifact_id="art-ver-002",
                artifact_type=ArtifactType.TEXT,
                source_agent="agent.test",
                payload={"v": i + 1},
                version=i + 1,
            )
            artifact_store.save(art)

        versions = artifact_store.list_versions("art-ver-002")
        assert len(versions) == 3
        assert versions == [1, 2, 3]

    def test_delete_artifact(self, artifact_store):
        """删除产物 (含所有版本)."""
        art = Artifact(
            artifact_id="art-del-001",
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        artifact_store.save(art)
        artifact_store.delete("art-del-001")
        assert artifact_store.load("art-del-001") is None

    def test_delete_nonexistent_raises(self, artifact_store):
        """删除不存在的产物应抛异常."""
        with pytest.raises(ArtifactNotFoundError):
            artifact_store.delete("nonexistent")


# ============================================================
# 7. ArtifactManager 测试
# ============================================================


class TestArtifactManager:
    """产物管理器测试 (融合 LangGraph Store + ADK + L5/L7 设计)."""

    def test_manager_creation(self, artifact_manager):
        """创建产物管理器."""
        assert artifact_manager is not None

    def test_create_artifact(self, artifact_manager, sample_artifact_data):
        """创建产物 (Agent 产出)."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload=sample_artifact_data,
        )
        assert art.artifact_id.startswith("art-")
        assert art.state == ArtifactState.CREATED
        assert art.version == 1

    def test_create_artifact_with_learner_context(self, artifact_manager):
        """创建产物带学习上下文."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.CHART,
            source_agent="agent.generation",
            payload={"chart_type": "spectrum"},
            learner_context={"learner_id": "stu-001", "kp_id": "KP-12"},
        )
        assert art.learner_context["learner_id"] == "stu-001"

    def test_get_artifact(self, artifact_manager, sample_artifact_data):
        """获取产物."""
        created = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload=sample_artifact_data,
        )
        retrieved = artifact_manager.get(created.artifact_id)
        assert retrieved is not None
        assert retrieved.artifact_id == created.artifact_id

    def test_get_nonexistent_raises(self, artifact_manager):
        """获取不存在的产物应抛异常."""
        with pytest.raises(ArtifactNotFoundError):
            artifact_manager.get("art-nonexistent")

    def test_update_artifact(self, artifact_manager, sample_artifact_data):
        """更新产物 (创建新版本, ADK 模式)."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload=sample_artifact_data,
        )
        new_payload = {**sample_artifact_data, "confidence": 0.92}
        updated = artifact_manager.update(
            art.artifact_id,
            payload=new_payload,
            source_agent="agent.generation",
        )
        assert updated.version == 2
        assert updated.payload["confidence"] == 0.92

    def test_update_nonexistent_raises(self, artifact_manager):
        """更新不存在的产物应抛异常."""
        with pytest.raises(ArtifactNotFoundError):
            artifact_manager.update("art-nonexistent", payload={})

    def test_update_not_editable_raises(self, artifact_manager):
        """更新只读产物应抛异常 (L7 设计文档)."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation",
            payload={},
            editable=False,
        )
        with pytest.raises(ArtifactError, match="not editable"):
            artifact_manager.update(art.artifact_id, payload={"v": 2})

    def test_get_version_history(self, artifact_manager):
        """获取版本历史 (Claude Science 版本链)."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={"v": 1},
        )
        for i in range(2, 5):
            artifact_manager.update(
                art.artifact_id,
                payload={"v": i},
                source_agent="agent.test",
            )

        history = artifact_manager.get_version_history(art.artifact_id)
        assert len(history) == 4
        assert history[0].version == 1
        assert history[3].version == 4

    def test_get_specific_version(self, artifact_manager):
        """获取特定版本的产物 (ADK 模式)."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={"v": 1},
        )
        artifact_manager.update(
            art.artifact_id, payload={"v": 2}, source_agent="agent.test"
        )

        v1 = artifact_manager.get_version(art.artifact_id, 1)
        assert v1 is not None
        assert v1.payload["v"] == 1

        v2 = artifact_manager.get_version(art.artifact_id, 2)
        assert v2 is not None
        assert v2.payload["v"] == 2

    def test_search_by_type(self, artifact_manager):
        """按类型搜索产物 (L7 结构化过滤)."""
        artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={"name": "report1"},
        )
        artifact_manager.create(
            artifact_type=ArtifactType.CHART,
            source_agent="agent.test",
            payload={"name": "chart1"},
        )
        artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={"name": "report2"},
        )

        text_arts = artifact_manager.search(artifact_type=ArtifactType.TEXT)
        assert len(text_arts) == 2

        chart_arts = artifact_manager.search(artifact_type=ArtifactType.CHART)
        assert len(chart_arts) == 1

    def test_search_by_source_agent(self, artifact_manager):
        """按来源 Agent 搜索 (L7 结构化过滤)."""
        artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload={},
        )
        artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation",
            payload={},
        )

        results = artifact_manager.search(source_agent="agent.diagnosis")
        assert len(results) == 1
        assert results[0].source_agent == "agent.diagnosis"

    def test_search_by_state(self, artifact_manager):
        """按状态搜索 (L7 结构化过滤)."""
        art1 = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        art2 = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        art2.transition_to(ArtifactState.RENDERED)

        created = artifact_manager.search(state=ArtifactState.CREATED)
        assert len(created) == 1
        assert created[0].artifact_id == art1.artifact_id

    def test_search_combined_filter(self, artifact_manager):
        """组合过滤 (L7 多维度组合过滤)."""
        artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload={"topic": "math"},
        )
        artifact_manager.create(
            artifact_type=ArtifactType.CHART,
            source_agent="agent.diagnosis",
            payload={"topic": "math"},
        )
        artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation",
            payload={"topic": "physics"},
        )

        results = artifact_manager.search(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
        )
        assert len(results) == 1

    def test_archive_artifact(self, artifact_manager):
        """归档产物 (L7 生命周期终态)."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        artifact_manager.archive(art.artifact_id)
        archived = artifact_manager.get(art.artifact_id)
        assert archived.state == ArtifactState.ARCHIVED

    def test_delete_artifact(self, artifact_manager):
        """删除产物."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        artifact_manager.delete(art.artifact_id)
        with pytest.raises(ArtifactNotFoundError):
            artifact_manager.get(art.artifact_id)

    def test_list_all(self, artifact_manager):
        """列出所有产物."""
        for i in range(5):
            artifact_manager.create(
                artifact_type=ArtifactType.TEXT,
                source_agent="agent.test",
                payload={"index": i},
            )
        all_arts = artifact_manager.list_all()
        assert len(all_arts) == 5


# ============================================================
# 8. ArtifactEdit 测试
# ============================================================


class TestArtifactEdit:
    """产物编辑操作测试 (L5 Artifact-Edit Channel)."""

    def test_edit_creation(self):
        """创建编辑操作记录."""
        edit = ArtifactEdit(
            artifact_id="art-001",
            learner_id="stu-001",
            edit_content={"type": "text_change", "old": "发光", "new": "荧光"},
        )
        assert edit.artifact_id == "art-001"
        assert edit.learner_id == "stu-001"
        assert edit.edit_content["type"] == "text_change"
        assert edit.state == ArtifactEditState.PENDING
        assert edit.edit_id.startswith("edit-")

    def test_edit_state_transitions(self):
        """编辑状态转换: PENDING → APPLIED / REJECTED (L5 设计文档)."""
        edit = ArtifactEdit(
            artifact_id="art-001",
            learner_id="stu-001",
            edit_content={},
        )
        assert edit.state == ArtifactEditState.PENDING

        edit.state = ArtifactEditState.APPLIED
        assert edit.state == ArtifactEditState.APPLIED

    def test_edit_rejected(self):
        """编辑可被拒绝 (CC1 审核不通过)."""
        edit = ArtifactEdit(
            artifact_id="art-001",
            learner_id="stu-001",
            edit_content={},
        )
        edit.state = ArtifactEditState.REJECTED
        assert edit.state == ArtifactEditState.REJECTED

    def test_edit_with_edit_operation(self):
        """编辑可携带结构化编辑操作 (L5 设计文档)."""
        edit = ArtifactEdit(
            artifact_id="art-001",
            learner_id="stu-001",
            edit_content={
                "type": "axis_change",
                "axis": "x",
                "from": "linear",
                "to": "log",
            },
        )
        assert edit.edit_content["type"] == "axis_change"
        assert edit.edit_content["to"] == "log"


# ============================================================
# 9. ArtifactProvenance 测试
# ============================================================


class TestArtifactProvenance:
    """产物溯源记录测试 (Claude Science 五维度溯源 + L5 Provenance Ledger)."""

    def test_provenance_creation(self):
        """创建溯源记录."""
        prov = ArtifactProvenance(
            artifact_id="art-001",
            actor_chain=["agent.diagnosis", "cc1.actor_critic"],
            edit_summary="生成学情诊断报告",
            code_hash="sha256:a3f8b2c1...",
        )
        assert prov.artifact_id == "art-001"
        assert len(prov.actor_chain) == 2
        assert prov.edit_summary == "生成学情诊断报告"
        assert prov.code_hash == "sha256:a3f8b2c1..."
        assert prov.timestamp > 0

    def test_provenance_with_from_to_version(self):
        """溯源记录可标记版本变更 (L5 Provenance Ledger)."""
        prov = ArtifactProvenance(
            artifact_id="art-001",
            actor_chain=["agent.generation"],
            edit_summary="切换X轴为对数刻度",
            code_hash="sha256:xyz789",
            from_version=3,
            to_version=4,
        )
        assert prov.from_version == 3
        assert prov.to_version == 4

    def test_provenance_with_data_hash(self):
        """溯源记录可携带数据哈希."""
        prov = ArtifactProvenance(
            artifact_id="art-001",
            actor_chain=["agent.test"],
            edit_summary="test",
            code_hash="sha256:abc",
            data_hash="sha256:data456",
        )
        assert prov.data_hash == "sha256:data456"

    def test_provenance_to_dict(self):
        """溯源记录应可序列化为字典."""
        prov = ArtifactProvenance(
            artifact_id="art-001",
            actor_chain=["agent.a", "agent.b"],
            edit_summary="test edit",
            code_hash="sha256:abc",
        )
        d = prov.to_dict()
        assert d["artifact_id"] == "art-001"
        assert d["actor_chain"] == ["agent.a", "agent.b"]
        assert d["edit_summary"] == "test edit"
        assert "timestamp" in d


# ============================================================
# 10. 集成测试
# ============================================================


class TestArtifactIntegration:
    """产物管理集成测试 (与现有系统联动)."""

    def test_artifact_with_orchestration_result(self, artifact_manager):
        """产物 + 编排结果联动 (OrchestrationResult.outputs → Artifact)."""
        # 模拟编排结果
        orchestration_output = {
            "task_id": "t1",
            "agent_id": "agent.diagnosis",
            "output": {"report_id": "rpt-001", "kp_gaps": ["KP-12"]},
        }

        # 将编排结果注册为产物
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent=orchestration_output["agent_id"],
            payload=orchestration_output["output"],
            provenance_chain=[orchestration_output["task_id"]],
        )
        assert art.payload["report_id"] == "rpt-001"
        assert art.provenance_chain[0] == "t1"

    def test_artifact_with_communication_message(self, artifact_manager):
        """产物 + 通信消息联动 (Message.payload → Artifact)."""
        # 模拟通信消息中的产物数据
        msg_payload = {
            "report_id": "rpt-002",
            "learner_id": "stu-001",
            "mastery_vector": {"KP-12": 0.35},
        }

        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload=msg_payload,
        )
        assert art.payload["learner_id"] == "stu-001"

    def test_artifact_edit_lifecycle(self, artifact_manager):
        """完整产物编辑生命周期 (L5 设计文档五阶段)."""
        # 1. Agent 产出
        art = artifact_manager.create(
            artifact_type=ArtifactType.CHART,
            source_agent="agent.generation",
            payload={"chart_type": "bar", "x_axis": "linear"},
        )
        assert art.state == ArtifactState.CREATED

        # 2. 渲染
        art.transition_to(ArtifactState.RENDERED)

        # 3. 用户编辑 → Agent 产出新版本
        updated = artifact_manager.update(
            art.artifact_id,
            payload={"chart_type": "bar", "x_axis": "log"},
            source_agent="agent.generation",
            edit_operation={"type": "axis_change", "axis": "x", "to": "log"},
        )
        assert updated.version == 2
        assert updated.payload["x_axis"] == "log"

        # 4. 审核通过
        updated.transition_to(ArtifactState.REVIEWED)

        # 5. 归档
        artifact_manager.archive(art.artifact_id)
        final = artifact_manager.get(art.artifact_id)
        assert final.state == ArtifactState.ARCHIVED

    def test_artifact_provenance_chain(self, artifact_manager):
        """产物溯源链 (多版本溯源记录)."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload={"v": 1},
        )

        # 记录溯源
        artifact_manager.add_provenance(
            artifact_id=art.artifact_id,
            actor_chain=["agent.diagnosis"],
            edit_summary="初始诊断报告",
            code_hash="sha256:v1",
            from_version=1,
            to_version=1,
        )

        # 更新产物
        artifact_manager.update(
            art.artifact_id,
            payload={"v": 2},
            source_agent="agent.generation",
        )

        artifact_manager.add_provenance(
            artifact_id=art.artifact_id,
            actor_chain=["agent.generation", "cc1.actor_critic"],
            edit_summary="增强诊断报告内容",
            code_hash="sha256:v2",
            from_version=1,
            to_version=2,
        )

        chain = artifact_manager.get_provenance_chain(art.artifact_id)
        assert len(chain) == 2
        assert chain[0].from_version == 1
        assert chain[1].to_version == 2
        assert "cc1.actor_critic" in chain[1].actor_chain

    def test_multi_agent_artifact_topology(self, artifact_manager):
        """多 Agent 产物拓扑 (4 个核心 Agent 产出)."""
        # A1 学情诊断
        diagnosis = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.diagnosis",
            payload={"report": "diagnosis"},
        )

        # A2 知识生成
        generation = artifact_manager.create(
            artifact_type=ArtifactType.GRAPH,
            source_agent="agent.generation",
            payload={"graph": "knowledge_graph"},
        )

        # A3 审核校验
        review = artifact_manager.create(
            artifact_type=ArtifactType.PROVENANCE,
            source_agent="agent.review",
            payload={"review": "pass"},
        )

        # A4 导学决策
        guidance = artifact_manager.create(
            artifact_type=ArtifactType.INTERACTIVE,
            source_agent="agent.guidance",
            payload={"plan": "study_plan"},
        )

        all_arts = artifact_manager.list_all()
        assert len(all_arts) == 4

        # 按来源搜索
        diag = artifact_manager.search(source_agent="agent.diagnosis")
        assert len(diag) == 1
        assert diag[0].artifact_id == diagnosis.artifact_id

    def test_fork_artifact_scenario(self, artifact_manager):
        """Fork 产物场景 (L5 Session Fork 产物合并)."""
        # 主会话产物
        main_art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation",
            payload={"content": "main version"},
        )

        # Fork 产物 (带 fork_origin)
        fork_art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.generation",
            payload={"content": "fork version"},
            fork_origin="sess-fork-001",
        )

        # 搜索 Fork 产物
        fork_arts = artifact_manager.search(fork_origin="sess-fork-001")
        assert len(fork_arts) == 1
        assert fork_arts[0].payload["content"] == "fork version"

        # 搜索主会话产物 (fork_origin 为 None)
        main_arts = artifact_manager.search(fork_origin=None)
        assert len(main_arts) == 1
        assert main_arts[0].payload["content"] == "main version"


# ============================================================
# 11. 错误处理测试
# ============================================================


class TestArtifactErrorHandling:
    """产物管理错误处理与恢复测试."""

    def test_artifact_error_creation(self):
        """创建产物错误."""
        err = ArtifactError("Test error")
        assert str(err) == "Test error"

    def test_artifact_not_found_error(self):
        """产物不存在错误."""
        err = ArtifactNotFoundError("art-001")
        assert "art-001" in str(err)
        assert isinstance(err, ArtifactError)

    def test_update_archived_raises(self, artifact_manager):
        """归档后的产物不能更新."""
        art = artifact_manager.create(
            artifact_type=ArtifactType.TEXT,
            source_agent="agent.test",
            payload={},
        )
        artifact_manager.archive(art.artifact_id)
        with pytest.raises(ArtifactError, match="archived"):
            artifact_manager.update(art.artifact_id, payload={"v": 2})

    def test_get_version_history_nonexistent_raises(self, artifact_manager):
        """获取不存在产物的版本历史应抛异常."""
        with pytest.raises(ArtifactNotFoundError):
            artifact_manager.get_version_history("art-nonexistent")

    def test_add_provenance_nonexistent_raises(self, artifact_manager):
        """为不存在的产物添加溯源应抛异常."""
        with pytest.raises(ArtifactNotFoundError):
            artifact_manager.add_provenance(
                artifact_id="art-nonexistent",
                actor_chain=["agent.test"],
                edit_summary="test",
                code_hash="sha256:abc",
            )

    def test_invalid_artifact_type_raises(self):
        """无效产物类型应抛异常."""
        with pytest.raises(ValueError):
            Artifact(
                artifact_type="invalid_type",  # type: ignore[arg-type]
                source_agent="agent.test",
                payload={},
            )

    def test_create_artifact_with_empty_source_agent_raises(self, artifact_manager):
        """空 source_agent 应抛异常."""
        with pytest.raises((ValueError, ArtifactError)):
            artifact_manager.create(
                artifact_type=ArtifactType.TEXT,
                source_agent="",
                payload={},
            )
