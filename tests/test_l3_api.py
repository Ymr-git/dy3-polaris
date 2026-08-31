"""L3 知识层 REST API 全链路集成测试.

基于 httpx.AsyncClient + ASGITransport 进行无端口测试,
覆盖 L3Router 暴露的全部 40+ REST 端点。

测试覆盖:
  1.  响应辅助函数 (_ok / _err / _safe_model_dump)
  2.  健康检查端点
  3.  知识实体 CRUD (创建/列表/获取/更新/删除)
  4.  三元组管理 (创建/查询/删除)
  5.  知识检索 (关键词/向量/混合/意图驱动)
  6.  知识摄入管道 (单文档/批量)
  7.  事实校验 (校验/标准值管理)
  8.  质量管理 (评估/批量/全局/冲突/溯源/审计)
  9.  图推理 (路径/多跳/规则/链接预测/模式匹配/类比)
  10. 本体管理 (领域列表/获取/验证)
  11. 持久化 (快照/恢复)
  12. 知识库统计
  13. 错误处理与边界条件
  14. 全链路端到端工作流
  15. L3Router 元信息
"""

from __future__ import annotations

import logging
import os
import tempfile

logging.disable(logging.CRITICAL)

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette

from dy3_polaris.l3 import KnowledgeStore
from dy3_polaris.l3.api.router import L3Router, _ok, _err, _safe_model_dump
from dy3_polaris.l3.fact_check import FactChecker, StandardValue, StandardValueStore
from dy3_polaris.l3.graph_reasoner import GraphReasoner, ReasoningMode
from dy3_polaris.l3.ingestion import IngestionPipeline
from dy3_polaris.l3.intent_router import IntentRouter
from dy3_polaris.l3.models import (
    EntityType,
    KnowledgeEntity,
    KnowledgeTriple,
    RelationType,
)
from dy3_polaris.l3.ontology import OntologyRegistry
from dy3_polaris.l3.persistence import PersistenceManager
from dy3_polaris.l3.quality_manager import QualityManager
from dy3_polaris.l3.retrieval import RetrievalEngine


# ============================================================
# 测试辅助
# ============================================================

def _make_store_with_data() -> KnowledgeStore:
    """创建并填充测试数据的 KnowledgeStore.

    预置:
    - 3 个实体: Dy3+ (MATERIAL), YAG (CHEMICAL_COMPOUND), Eu3+ (MATERIAL)
    - 2 个三元组: Dy3+ -doped_in-> YAG, Eu3+ -doped_in-> YAG
    """
    store = KnowledgeStore()

    # 创建实体
    dy3 = KnowledgeEntity(
        name="Dy3+",
        entity_type=EntityType.MATERIAL,
        description="镝离子,稀土发光材料激活剂",
        domain="luminescence",
        properties={"emission_wavelength": 580.0, "quantum_efficiency": 0.85},
        identifiers={"cas": "7429-90-5"},
        tags=["rare_earth", "activator"],
        aliases=["Dy3plus", "dysprosium_ion"],
    )
    yag = KnowledgeEntity(
        name="YAG",
        entity_type=EntityType.CHEMICAL_COMPOUND,
        description="钇铝石榴石,常用发光基质材料",
        domain="luminescence",
        properties={"formula": "Y3Al5O12", "crystal_system": "cubic"},
        identifiers={"cas": "12005-37-3"},
        tags=["host_material", "garnet"],
        aliases=["Y3Al5O12"],
    )
    eu3 = KnowledgeEntity(
        name="Eu3+",
        entity_type=EntityType.MATERIAL,
        description="铕离子,红色发光激活剂",
        domain="luminescence",
        properties={"emission_wavelength": 615.0, "quantum_efficiency": 0.78},
        tags=["rare_earth", "activator"],
        aliases=["Eu3plus"],
    )

    store.add_entity(dy3)
    store.add_entity(yag)
    store.add_entity(eu3)

    # 创建三元组
    t1 = KnowledgeTriple(
        subject_id=dy3.entity_id,
        predicate=RelationType.RELATED_TO.value,
        object_id=yag.entity_id,
        confidence=0.95,
    )
    t2 = KnowledgeTriple(
        subject_id=eu3.entity_id,
        predicate=RelationType.RELATED_TO.value,
        object_id=yag.entity_id,
        confidence=0.90,
    )
    store.add_triple(t1)
    store.add_triple(t2)

    return store


def _make_router(store: KnowledgeStore | None = None) -> L3Router:
    """创建配置好的 L3Router."""
    store = store or _make_store_with_data()
    tmpdir = tempfile.mkdtemp(prefix="l3_test_")
    return L3Router(
        store,
        persistence_manager=PersistenceManager(store, base_path=tmpdir),
    )


async def _get(client: AsyncClient, path: str) -> dict:
    r = await client.get(path)
    return r.json()


async def _post(client: AsyncClient, path: str, json: dict | None = None) -> dict:
    r = await client.post(path, json=json or {})
    return r.json()


async def _put(client: AsyncClient, path: str, json: dict | None = None) -> dict:
    r = await client.put(path, json=json or {})
    return r.json()


async def _delete(client: AsyncClient, path: str) -> dict:
    r = await client.delete(path)
    return r.json()


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def store() -> KnowledgeStore:
    return _make_store_with_data()


@pytest.fixture
def router(store: KnowledgeStore) -> L3Router:
    tmpdir = tempfile.mkdtemp(prefix="l3_test_")
    return L3Router(
        store,
        persistence_manager=PersistenceManager(store, base_path=tmpdir),
    )


@pytest.fixture
async def client(router: L3Router) -> AsyncClient:
    app = router.create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def empty_client() -> AsyncClient:
    """空知识库客户端 (无预置数据)."""
    s = KnowledgeStore()
    tmpdir = tempfile.mkdtemp(prefix="l3_test_")
    r = L3Router(s, persistence_manager=PersistenceManager(s, base_path=tmpdir))
    app = r.create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ============================================================
# 1. 响应辅助函数
# ============================================================

class TestResponseHelpers:
    """测试 _ok / _err / _safe_model_dump 辅助函数."""

    def test_ok_默认值(self) -> None:
        r = _ok()
        assert r["code"] == 0
        assert r["data"] is None
        assert r["message"] == ""

    def test_ok_带数据(self) -> None:
        r = _ok({"key": "value"}, "成功")
        assert r["code"] == 0
        assert r["data"] == {"key": "value"}
        assert r["message"] == "成功"

    def test_err_基本格式(self) -> None:
        r = _err(-32700, "解析失败")
        assert r["code"] == -32700
        assert r["message"] == "解析失败"
        assert "detail" not in r

    def test_err_带详情(self) -> None:
        r = _err(-32700, "解析失败", "JSON 格式错误")
        assert r["detail"] == "JSON 格式错误"

    def test_safe_model_dump_pydantic模型(self) -> None:
        entity = KnowledgeEntity(name="测试", entity_type=EntityType.CONCEPT)
        result = _safe_model_dump(entity)
        assert isinstance(result, dict)
        assert result["name"] == "测试"
        assert result["entity_type"] == "concept"

    def test_safe_model_dump_列表(self) -> None:
        entities = [
            KnowledgeEntity(name="A", entity_type=EntityType.CONCEPT),
            KnowledgeEntity(name="B", entity_type=EntityType.MATERIAL),
        ]
        result = _safe_model_dump(entities)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "A"

    def test_safe_model_dump_None(self) -> None:
        assert _safe_model_dump(None) is None

    def test_safe_model_dump_原始类型(self) -> None:
        assert _safe_model_dump(42) == 42
        assert _safe_model_dump("hello") == "hello"
        assert _safe_model_dump(True) is True


# ============================================================
# 2. 健康检查
# ============================================================

class TestHealthEndpoint:
    """GET /health 健康检查."""

    async def test_health_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/health")
        assert data["code"] == 0
        assert data["data"]["status"] == "healthy"
        assert data["data"]["layer"] == "L3"

    async def test_health_包含实体计数(self, client: AsyncClient) -> None:
        data = await _get(client, "/health")
        assert data["data"]["entity_count"] == 3
        assert data["data"]["triple_count"] == 2

    async def test_health_包含时间戳(self, client: AsyncClient) -> None:
        data = await _get(client, "/health")
        assert "timestamp" in data["data"]
        assert isinstance(data["data"]["timestamp"], float)

    async def test_health_空库(self, empty_client: AsyncClient) -> None:
        data = await _get(empty_client, "/health")
        assert data["code"] == 0
        assert data["data"]["entity_count"] == 0
        assert data["data"]["triple_count"] == 0


# ============================================================
# 3. 知识实体 CRUD
# ============================================================

class TestEntityCRUD:
    """知识实体管理端点."""

    async def test_create_entity_201(self, empty_client: AsyncClient) -> None:
        data = await _post(empty_client, "/entities", {
            "name": "Tb3+",
            "entity_type": "material",
            "description": "铽离子,绿色发光激活剂",
            "domain": "luminescence",
            "properties": {"emission_wavelength": 545.0},
            "tags": ["rare_earth"],
            "aliases": ["Tb3plus"],
        })
        assert data["code"] == 0
        assert data["data"]["name"] == "Tb3+"
        assert data["data"]["entity_type"] == "material"
        assert data["data"]["domain"] == "luminescence"
        assert "entity_id" in data["data"]

    async def test_create_entity_默认类型(self, empty_client: AsyncClient) -> None:
        data = await _post(empty_client, "/entities", {"name": "概念A"})
        assert data["code"] == 0
        assert data["data"]["entity_type"] == "concept"

    async def test_create_entity_400_缺name(self, empty_client: AsyncClient) -> None:
        data = await _post(empty_client, "/entities", {"entity_type": "material"})
        assert data["code"] != 0

    async def test_create_entity_400_无效类型(self, empty_client: AsyncClient) -> None:
        data = await _post(empty_client, "/entities", {
            "name": "测试",
            "entity_type": "INVALID_TYPE",
        })
        assert data["code"] != 0

    async def test_list_entities_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/entities")
        assert data["code"] == 0
        assert data["data"]["total"] == 3
        assert len(data["data"]["items"]) == 3

    async def test_list_entities_分页(self, client: AsyncClient) -> None:
        data = await _get(client, "/entities?limit=2&offset=0")
        assert data["code"] == 0
        assert len(data["data"]["items"]) == 2
        assert data["data"]["has_more"] is True

    async def test_list_entities_分页第二页(self, client: AsyncClient) -> None:
        data = await _get(client, "/entities?limit=2&offset=2")
        assert data["code"] == 0
        assert len(data["data"]["items"]) == 1
        assert data["data"]["has_more"] is False

    async def test_list_entities_按类型过滤(self, client: AsyncClient) -> None:
        data = await _get(client, "/entities?entity_type=material")
        assert data["code"] == 0
        assert len(data["data"]["items"]) == 2
        for item in data["data"]["items"]:
            assert item["entity_type"] == "material"

    async def test_list_entities_按领域过滤(self, client: AsyncClient) -> None:
        data = await _get(client, "/entities?domain=luminescence")
        assert data["code"] == 0
        assert len(data["data"]["items"]) == 3

    async def test_get_entity_200(self, client: AsyncClient, store: KnowledgeStore) -> None:
        eid = list(store.entity_store._entities.keys())[0]
        data = await _get(client, f"/entities/{eid}")
        assert data["code"] == 0
        assert data["data"]["entity_id"] == eid

    async def test_get_entity_404(self, client: AsyncClient) -> None:
        data = await _get(client, "/entities/nonexistent")
        assert data["code"] == -32601

    async def test_update_entity_200(self, client: AsyncClient, store: KnowledgeStore) -> None:
        eid = list(store.entity_store._entities.keys())[0]
        data = await _put(client, f"/entities/{eid}", {
            "description": "更新后的描述",
            "changed_by": "test_user",
            "reason": "测试更新",
        })
        assert data["code"] == 0
        assert data["data"]["description"] == "更新后的描述"

    async def test_update_entity_404(self, client: AsyncClient) -> None:
        data = await _put(client, "/entities/ghost", {"description": "test"})
        assert data["code"] == -32601

    async def test_delete_entity_200(self, client: AsyncClient, store: KnowledgeStore) -> None:
        eid = list(store.entity_store._entities.keys())[0]
        data = await _delete(client, f"/entities/{eid}")
        assert data["code"] == 0
        assert data["data"]["removed"] == eid

        # 验证已删除
        data2 = await _get(client, f"/entities/{eid}")
        assert data2["code"] == -32601

    async def test_delete_entity_404(self, client: AsyncClient) -> None:
        data = await _delete(client, "/entities/ghost")
        assert data["code"] == -32601

    async def test_delete_entity_级联删除三元组(
        self, client: AsyncClient, store: KnowledgeStore
    ) -> None:
        """删除实体后, 关联三元组也应被清理."""
        eid = list(store.entity_store._entities.keys())[0]
        await _delete(client, f"/entities/{eid}")

        # 检查三元组数量减少
        stats = await _get(client, "/stats")
        assert stats["data"]["triple_count"] < 2


# ============================================================
# 4. 三元组管理
# ============================================================

class TestTripleManagement:
    """三元组管理端点."""

    async def test_create_triple_201(self, client: AsyncClient, store: KnowledgeStore) -> None:
        ids = list(store.entity_store._entities.keys())
        data = await _post(client, "/triples", {
            "subject_id": ids[0],
            "predicate": "supports",
            "object_id": ids[1],
            "confidence": 0.88,
        })
        assert data["code"] == 0
        assert data["data"]["predicate"] == "supports"
        assert data["data"]["confidence"] == 0.88

    async def test_create_triple_400_缺subject_id(self, client: AsyncClient) -> None:
        data = await _post(client, "/triples", {"predicate": "related_to"})
        assert data["code"] != 0

    async def test_list_triples_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/triples")
        assert data["code"] == 0
        assert data["data"]["total"] == 2
        assert len(data["data"]["items"]) == 2

    async def test_list_triples_按主语过滤(self, client: AsyncClient, store: KnowledgeStore) -> None:
        eid = list(store.entity_store._entities.keys())[0]
        data = await _get(client, f"/triples?subject_id={eid}")
        assert data["code"] == 0
        for item in data["data"]["items"]:
            assert item["subject_id"] == eid

    async def test_list_triples_按谓词过滤(self, client: AsyncClient) -> None:
        data = await _get(client, "/triples?predicate=related_to")
        assert data["code"] == 0
        for item in data["data"]["items"]:
            assert item["predicate"] == "related_to"

    async def test_list_triples_无结果(self, client: AsyncClient) -> None:
        data = await _get(client, "/triples?predicate=nonexistent_predicate")
        assert data["code"] == 0
        assert len(data["data"]["items"]) == 0

    async def test_delete_triple_200(self, client: AsyncClient, store: KnowledgeStore) -> None:
        tid = list(store.triple_store._triples.keys())[0]
        data = await _delete(client, f"/triples/{tid}")
        assert data["code"] == 0
        assert data["data"]["removed"] == tid

    async def test_delete_triple_404(self, client: AsyncClient) -> None:
        data = await _delete(client, "/triples/ghost")
        assert data["code"] == -32601


# ============================================================
# 5. 知识检索
# ============================================================

class TestRetrievalEndpoints:
    """知识检索端点."""

    async def test_retrieve_keyword_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/retrieve/keyword", {
            "query": "Dy3+ 发光",
            "top_k": 5,
        })
        assert data["code"] == 0
        assert "results" in data["data"] or "items" in data["data"]

    async def test_retrieve_keyword_400_空query(self, client: AsyncClient) -> None:
        data = await _post(client, "/retrieve/keyword", {"query": ""})
        assert data["code"] == -32700

    async def test_retrieve_vector_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/retrieve/vector", {
            "query_vector": [0.1] * 10,
            "query": "发光材料",
            "top_k": 5,
        })
        assert data["code"] == 0

    async def test_retrieve_vector_400_空向量(self, client: AsyncClient) -> None:
        data = await _post(client, "/retrieve/vector", {"query_vector": []})
        assert data["code"] == -32700

    async def test_retrieve_hybrid_200_带向量(self, client: AsyncClient) -> None:
        data = await _post(client, "/retrieve/hybrid", {
            "query": "YAG 基质材料",
            "query_vector": [0.2] * 10,
            "top_k": 5,
        })
        assert data["code"] == 0

    async def test_retrieve_hybrid_200_无向量(self, client: AsyncClient) -> None:
        data = await _post(client, "/retrieve/hybrid", {
            "query": "YAG 基质材料",
            "top_k": 5,
        })
        assert data["code"] == 0

    async def test_retrieve_hybrid_400_空query(self, client: AsyncClient) -> None:
        data = await _post(client, "/retrieve/hybrid", {"query": ""})
        assert data["code"] == -32700

    async def test_retrieve_intent_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/retrieve/intent", {
            "query": "Dy3+的发射波长是多少nm",
            "top_k": 5,
        })
        assert data["code"] == 0
        assert "intent" in data["data"]
        assert "retrieval_result" in data["data"]

    async def test_retrieve_intent_400_空query(self, client: AsyncClient) -> None:
        data = await _post(client, "/retrieve/intent", {"query": ""})
        assert data["code"] == -32700


# ============================================================
# 6. 知识摄入
# ============================================================

class TestIngestionEndpoints:
    """知识摄入端点."""

    async def test_ingest_201(self, empty_client: AsyncClient) -> None:
        data = await _post(empty_client, "/ingest", {
            "content": "Dy3+离子在YAG基质中的发射波长为580nm,量子效率为85%。",
            "document_id": "doc-001",
            "metadata": {"source": "textbook", "title": "稀土发光材料"},
        })
        assert data["code"] == 0

    async def test_ingest_400_缺content(self, empty_client: AsyncClient) -> None:
        data = await _post(empty_client, "/ingest", {"document_id": "doc-001"})
        assert data["code"] == -32700

    async def test_ingest_400_缺document_id(self, empty_client: AsyncClient) -> None:
        data = await _post(empty_client, "/ingest", {"content": "测试内容"})
        assert data["code"] == -32700

    async def test_ingest_batch_201(self, empty_client: AsyncClient) -> None:
        data = await _post(empty_client, "/ingest/batch", {
            "items": [
                {
                    "content": "Eu3+离子的发射波长为615nm。",
                    "document_id": "doc-001",
                },
                {
                    "content": "YAG的化学式为Y3Al5O12,属于立方晶系。",
                    "document_id": "doc-002",
                },
            ],
        })
        assert data["code"] == 0

    async def test_ingest_batch_400_空列表(self, empty_client: AsyncClient) -> None:
        data = await _post(empty_client, "/ingest/batch", {"items": []})
        assert data["code"] == -32700


# ============================================================
# 7. 事实校验
# ============================================================

class TestFactCheckEndpoints:
    """事实校验端点."""

    async def test_fact_check_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/fact-check", {
            "content": "发射波长为580nm,温度为300K",
        })
        assert data["code"] == 0
        assert "total_assertions" in data["data"]

    async def test_fact_check_400_空内容(self, client: AsyncClient) -> None:
        data = await _post(client, "/fact-check", {"content": ""})
        assert data["code"] == -32700

    async def test_list_standards_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/standards")
        assert data["code"] == 0
        assert "items" in data["data"]
        assert "total" in data["data"]

    async def test_add_standard_201(self, client: AsyncClient) -> None:
        data = await _post(client, "/standards", {
            "kp_id": "KP-001",
            "param_name": "emission_wavelength",
            "standard_value": 580.0,
            "unit": "nm",
            "source_ref": "GB/T 1-2020",
        })
        assert data["code"] == 0
        assert data["data"]["kp_id"] == "KP-001"
        assert data["data"]["standard_value"] == 580.0
        assert data["data"]["tolerance"] == 2.0  # 默认容差

    async def test_add_standard_201_自定义容差(self, client: AsyncClient) -> None:
        data = await _post(client, "/standards", {
            "kp_id": "KP-002",
            "param_name": "custom_param",
            "standard_value": 100.0,
            "tolerance": 5.0,
            "tolerance_type": "relative",
            "unit": "%",
        })
        assert data["code"] == 0
        assert data["data"]["tolerance"] == 5.0
        assert data["data"]["tolerance_type"] == "relative"

    async def test_add_standard_400_缺参数(self, client: AsyncClient) -> None:
        data = await _post(client, "/standards", {"kp_id": "KP-001"})
        assert data["code"] != 0

    async def test_add_standard_then_list(self, client: AsyncClient) -> None:
        """添加标准值后列表应包含新标准."""
        await _post(client, "/standards", {
            "kp_id": "KP-TEST",
            "param_name": "test_param",
            "standard_value": 42.0,
            "unit": "test",
        })
        data = await _get(client, "/standards")
        kp_ids = [s["kp_id"] for s in data["data"]["items"]]
        assert "KP-TEST" in kp_ids


# ============================================================
# 8. 质量管理
# ============================================================

class TestQualityEndpoints:
    """质量管理端点."""

    async def test_quality_assess_200(self, client: AsyncClient, store: KnowledgeStore) -> None:
        eid = list(store.entity_store._entities.keys())[0]
        data = await _post(client, "/quality/assess", {"entity_id": eid})
        assert data["code"] == 0

    async def test_quality_assess_404(self, client: AsyncClient) -> None:
        data = await _post(client, "/quality/assess", {"entity_id": "ghost"})
        assert data["code"] == -32601

    async def test_quality_assess_batch_200(self, client: AsyncClient, store: KnowledgeStore) -> None:
        ids = list(store.entity_store._entities.keys())
        data = await _post(client, "/quality/assess/batch", {"entity_ids": ids})
        assert data["code"] == 0
        assert isinstance(data["data"], list)

    async def test_quality_assess_batch_400_空列表(self, client: AsyncClient) -> None:
        data = await _post(client, "/quality/assess/batch", {"entity_ids": []})
        assert data["code"] == -32700

    async def test_quality_assess_global_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/quality/assess/global")
        assert data["code"] == 0

    async def test_quality_detect_conflicts_200(self, client: AsyncClient, store: KnowledgeStore) -> None:
        eid = list(store.entity_store._entities.keys())[0]
        data = await _post(client, "/quality/conflicts/detect", {
            "entity_id": eid,
            "external_claims": [{"property": "emission_wavelength", "value": 999.0}],
        })
        assert data["code"] == 0

    async def test_quality_detect_conflicts_404(self, client: AsyncClient) -> None:
        data = await _post(client, "/quality/conflicts/detect", {"entity_id": "ghost"})
        assert data["code"] == -32601

    async def test_quality_dashboard_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/quality/dashboard")
        assert data["code"] == 0

    async def test_quality_record_provenance_201(self, client: AsyncClient, store: KnowledgeStore) -> None:
        eid = list(store.entity_store._entities.keys())[0]
        data = await _post(client, "/quality/provenance", {
            "entity_id": eid,
            "activity_type": "create",
            "agent_id": "test_agent",
            "description": "测试溯源记录",
        })
        assert data["code"] == 0

    async def test_quality_record_provenance_400_缺参数(self, client: AsyncClient) -> None:
        data = await _post(client, "/quality/provenance", {"entity_id": "e1"})
        assert data["code"] != 0

    async def test_quality_get_provenance_200(self, client: AsyncClient, store: KnowledgeStore) -> None:
        eid = list(store.entity_store._entities.keys())[0]
        # 先记录溯源
        await _post(client, "/quality/provenance", {
            "entity_id": eid,
            "activity_type": "create",
            "description": "测试",
        })
        # 再查询
        data = await _get(client, f"/quality/provenance/{eid}")
        assert data["code"] == 0

    async def test_quality_get_provenance_404(self, client: AsyncClient) -> None:
        data = await _get(client, "/quality/provenance/ghost")
        assert data["code"] == -32601

    async def test_quality_audit_log_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/quality/audit-log")
        assert data["code"] == 0

    async def test_quality_audit_log_按实体过滤(self, client: AsyncClient, store: KnowledgeStore) -> None:
        eid = list(store.entity_store._entities.keys())[0]
        # 先记录溯源
        await _post(client, "/quality/provenance", {
            "entity_id": eid,
            "activity_type": "create",
            "description": "测试",
        })
        data = await _get(client, f"/quality/audit-log?entity_id={eid}")
        assert data["code"] == 0


# ============================================================
# 9. 图推理
# ============================================================

class TestGraphReasoningEndpoints:
    """图推理端点."""

    async def test_graph_reason_path_finding(self, client: AsyncClient, store: KnowledgeStore) -> None:
        ids = list(store.entity_store._entities.keys())
        data = await _post(client, "/graph/reason", {
            "query": "Dy3+ 到 YAG 的路径",
            "mode": "path_finding",
            "start_id": ids[0],
            "end_id": ids[1],
        })
        assert data["code"] == 0
        assert data["data"]["mode"] == "path_finding"

    async def test_graph_reason_multi_hop(self, client: AsyncClient, store: KnowledgeStore) -> None:
        ids = list(store.entity_store._entities.keys())
        data = await _post(client, "/graph/reason", {
            "query": "从 Dy3+ 出发的多跳推理",
            "mode": "multi_hop",
            "start_id": ids[0],
            "relations": ["related_to"],
        })
        assert data["code"] == 0

    async def test_graph_reason_rule_inference(self, client: AsyncClient) -> None:
        data = await _post(client, "/graph/reason", {
            "query": "前向链式规则推理",
            "mode": "rule_inference",
        })
        assert data["code"] == 0

    async def test_graph_reason_link_prediction(self, client: AsyncClient, store: KnowledgeStore) -> None:
        ids = list(store.entity_store._entities.keys())
        data = await _post(client, "/graph/reason", {
            "query": "Dy3+ 的链接预测",
            "mode": "link_prediction",
            "entity_id": ids[0],
        })
        assert data["code"] == 0

    async def test_graph_reason_pattern_match(self, client: AsyncClient) -> None:
        data = await _post(client, "/graph/reason", {
            "query": "匹配材料-化合物模式",
            "mode": "pattern_match",
            "pattern": {
                "nodes": [
                    {"var": "x", "type": "material"},
                    {"var": "y", "type": "chemical_compound"},
                ],
                "edges": [{"from": "x", "to": "y", "predicate": "related_to"}],
            },
        })
        assert data["code"] == 0

    async def test_graph_reason_analogy(self, client: AsyncClient, store: KnowledgeStore) -> None:
        ids = list(store.entity_store._entities.keys())
        data = await _post(client, "/graph/reason", {
            "query": "类比推理",
            "mode": "analogy",
            "source_pair": [ids[0], ids[1]],
            "target_entity": ids[2],
        })
        assert data["code"] == 0

    async def test_graph_reason_400_无效模式(self, client: AsyncClient) -> None:
        data = await _post(client, "/graph/reason", {
            "query": "test",
            "mode": "invalid_mode",
        })
        assert data["code"] == -32700

    async def test_graph_stats_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/graph/stats")
        assert data["code"] == 0
        assert "rules_count" in data["data"]
        assert "triples_count" in data["data"]


# ============================================================
# 10. 本体管理
# ============================================================

class TestOntologyEndpoints:
    """本体管理端点."""

    async def test_ontology_domains_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/ontology/domains")
        assert data["code"] == 0
        assert "domains" in data["data"]
        assert "count" in data["data"]

    async def test_get_ontology_404(self, client: AsyncClient) -> None:
        data = await _get(client, "/ontology/nonexistent_domain")
        assert data["code"] == -32601

    async def test_ontology_validate_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/ontology/validate", {
            "domain": "luminescence",
            "entity_type": "material",
            "properties": {"emission_wavelength": 580.0},
        })
        assert data["code"] == 0
        assert "valid" in data["data"]
        assert "violations" in data["data"]


# ============================================================
# 11. 持久化
# ============================================================

class TestPersistenceEndpoints:
    """持久化端点."""

    async def test_persistence_snapshot_200(self, client: AsyncClient) -> None:
        data = await _post(client, "/persistence/snapshot")
        assert data["code"] == 0
        assert "path" in data["data"]

    async def test_persistence_snapshot_带路径(self, client: AsyncClient) -> None:
        tmpdir = tempfile.mkdtemp(prefix="l3_snapshot_")
        data = await _post(client, "/persistence/snapshot", {"path": tmpdir})
        assert data["code"] == 0
        assert tmpdir in data["data"]["path"]

    async def test_persistence_restore_200(self, client: AsyncClient) -> None:
        # 先保存
        snap = await _post(client, "/persistence/snapshot")
        path = snap["data"]["path"]
        # 再恢复
        data = await _post(client, "/persistence/restore", {"path": path})
        assert data["code"] == 0
        assert data["data"]["restored"] == path

    async def test_persistence_restore_400_空路径(self, client: AsyncClient) -> None:
        data = await _post(client, "/persistence/restore", {"path": ""})
        assert data["code"] == -32700

    async def test_persistence_restore_500_不存在路径(self, client: AsyncClient) -> None:
        data = await _post(client, "/persistence/restore", {"path": "/nonexistent/path"})
        assert data["code"] != 0


# ============================================================
# 12. 知识库统计
# ============================================================

class TestStatsEndpoint:
    """GET /stats 知识库统计."""

    async def test_stats_200(self, client: AsyncClient) -> None:
        data = await _get(client, "/stats")
        assert data["code"] == 0
        assert data["data"]["entity_count"] == 3
        assert data["data"]["triple_count"] == 2

    async def test_stats_包含类型分布(self, client: AsyncClient) -> None:
        data = await _get(client, "/stats")
        types = data["data"]["entity_types"]
        assert "material" in types
        assert "chemical_compound" in types

    async def test_stats_包含领域分布(self, client: AsyncClient) -> None:
        data = await _get(client, "/stats")
        domains = data["data"]["domains"]
        assert "luminescence" in domains

    async def test_stats_空库(self, empty_client: AsyncClient) -> None:
        data = await _get(empty_client, "/stats")
        assert data["code"] == 0
        assert data["data"]["entity_count"] == 0
        assert data["data"]["triple_count"] == 0


# ============================================================
# 13. 错误处理与边界条件
# ============================================================

class TestErrorHandling:
    """错误处理与边界条件."""

    async def test_请求体解析失败(self, client: AsyncClient) -> None:
        """发送非 JSON body."""
        r = await client.post(
            "/entities",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        data = r.json()
        assert data["code"] == -32700

    async def test_404_路径不存在(self, client: AsyncClient) -> None:
        r = await client.get("/nonexistent/path")
        assert r.status_code == 404

    async def test_create_entity_重复名称(self, empty_client: AsyncClient) -> None:
        """创建同名实体 (允许, 因 entity_id 不同)."""
        await _post(empty_client, "/entities", {"name": "重复", "entity_type": "concept"})
        data = await _post(empty_client, "/entities", {"name": "重复", "entity_type": "concept"})
        assert data["code"] == 0

    async def test_limit_上限(self, client: AsyncClient) -> None:
        """limit 超过 100 应被截断为 100."""
        data = await _get(client, "/entities?limit=999")
        assert data["code"] == 0
        assert data["data"]["limit"] == 100

    async def test_negative_offset(self, client: AsyncClient) -> None:
        """负 offset 应被修正为 0."""
        data = await _get(client, "/entities?offset=-5")
        assert data["code"] == 0
        assert data["data"]["offset"] == 0


# ============================================================
# 14. 全链路端到端工作流
# ============================================================

class TestFullLinkWorkflow:
    """全链路端到端工作流测试.

    模拟完整的知识管理生命周期:
    创建 → 关联 → 检索 → 摄入 → 校验 → 评估 → 推理 → 持久化
    """

    async def test_知识管理全链路(self, empty_client: AsyncClient) -> None:
        """端到端: 创建实体→建立关系→检索→摄入→校验→评估→推理→快照→恢复."""

        # Step 1: 创建实体
        entity1 = await _post(empty_client, "/entities", {
            "name": "Ce3+",
            "entity_type": "material",
            "description": "铈离子,蓝色发光激活剂",
            "domain": "luminescence",
            "properties": {"emission_wavelength": 460.0},
        })
        assert entity1["code"] == 0
        eid1 = entity1["data"]["entity_id"]

        entity2 = await _post(empty_client, "/entities", {
            "name": "LuAG",
            "entity_type": "chemical_compound",
            "description": "硅酸镥,蓝色发光基质",
            "domain": "luminescence",
        })
        assert entity2["code"] == 0
        eid2 = entity2["data"]["entity_id"]

        # Step 2: 建立关系
        triple = await _post(empty_client, "/triples", {
            "subject_id": eid1,
            "predicate": "related_to",
            "object_id": eid2,
            "confidence": 0.92,
        })
        assert triple["code"] == 0

        # Step 3: 验证统计
        stats = await _get(empty_client, "/stats")
        assert stats["data"]["entity_count"] == 2
        assert stats["data"]["triple_count"] == 1

        # Step 4: 检索
        kw_result = await _post(empty_client, "/retrieve/keyword", {
            "query": "Ce3+ 发光",
            "top_k": 5,
        })
        assert kw_result["code"] == 0

        # Step 5: 摄入知识
        ingest = await _post(empty_client, "/ingest", {
            "content": "Ce3+在LuAG基质中的发射波长为460nm。",
            "document_id": "doc-ce3-luag",
        })
        assert ingest["code"] == 0

        # Step 6: 事实校验
        check = await _post(empty_client, "/fact-check", {
            "content": "发射波长为460nm",
        })
        assert check["code"] == 0

        # Step 7: 质量评估
        assess = await _post(empty_client, "/quality/assess", {"entity_id": eid1})
        assert assess["code"] == 0

        # Step 8: 图推理
        reason = await _post(empty_client, "/graph/reason", {
            "query": "Ce3+ 到 LuAG 的路径",
            "mode": "path_finding",
            "start_id": eid1,
            "end_id": eid2,
        })
        assert reason["code"] == 0

        # Step 9: 持久化
        snap = await _post(empty_client, "/persistence/snapshot")
        assert snap["code"] == 0
        path = snap["data"]["path"]

        # Step 10: 恢复
        restore = await _post(empty_client, "/persistence/restore", {"path": path})
        assert restore["code"] == 0

    async def test_多实体批量操作工作流(self, empty_client: AsyncClient) -> None:
        """批量创建→批量评估→全局统计."""

        # 批量创建
        for i in range(5):
            await _post(empty_client, "/entities", {
                "name": f"材料_{i}",
                "entity_type": "material",
                "domain": "test",
            })

        # 验证创建
        stats = await _get(empty_client, "/stats")
        assert stats["data"]["entity_count"] == 5

        # 获取所有 ID
        listing = await _get(empty_client, "/entities?limit=100")
        ids = [e["entity_id"] for e in listing["data"]["items"]]

        # 批量评估
        batch = await _post(empty_client, "/quality/assess/batch", {"entity_ids": ids})
        assert batch["code"] == 0
        assert len(batch["data"]) == 5

        # 全局评估
        global_assess = await _post(empty_client, "/quality/assess/global")
        assert global_assess["code"] == 0

    async def test_冲突检测与消解工作流(self, client: AsyncClient, store: KnowledgeStore) -> None:
        """创建冲突→检测→消解."""

        eid = list(store.entity_store._entities.keys())[0]

        # 记录溯源
        prov = await _post(client, "/quality/provenance", {
            "entity_id": eid,
            "activity_type": "create",
            "agent_id": "test",
            "description": "初始创建",
        })
        assert prov["code"] == 0

        # 检测冲突
        conflicts = await _post(client, "/quality/conflicts/detect", {
            "entity_id": eid,
            "external_claims": [
                {"property": "emission_wavelength", "value": 999.0},
            ],
        })
        assert conflicts["code"] == 0

        # 查询溯源
        prov_get = await _get(client, f"/quality/provenance/{eid}")
        assert prov_get["code"] == 0

        # 查询审计日志
        audit = await _get(client, "/quality/audit-log")
        assert audit["code"] == 0


# ============================================================
# 15. L3Router 元信息
# ============================================================

class TestL3RouterMeta:
    """L3Router 元信息测试."""

    def test_create_app_返回_starlette实例(self, router: L3Router) -> None:
        app = router.create_app()
        assert isinstance(app, Starlette)

    def test_get_routes_summary_39条路由(self, router: L3Router) -> None:
        routes = router.get_routes_summary()
        assert len(routes) == 39

    def test_routes_summary_每条有必需字段(self, router: L3Router) -> None:
        for r in router.get_routes_summary():
            assert "path" in r
            assert "methods" in r
            assert "description" in r

    def test_cors_默认配置(self, store: KnowledgeStore) -> None:
        router = L3Router(store)
        app = router.create_app()
        assert app is not None

    def test_cors_自定义配置(self, store: KnowledgeStore) -> None:
        router = L3Router(store, cors_origins=["https://example.com"])
        app = router.create_app()
        assert app is not None

    def test_自定义组件注入(self, store: KnowledgeStore) -> None:
        """验证自定义组件可以注入到路由器."""
        custom_retrieval = RetrievalEngine(store)
        custom_fact_checker = FactChecker()
        custom_quality = QualityManager()
        custom_graph = GraphReasoner(store)
        custom_ontology = OntologyRegistry()
        custom_intent = IntentRouter(store)
        custom_ingestion = IngestionPipeline(store)
        tmpdir = tempfile.mkdtemp()
        custom_persistence = PersistenceManager(store, base_path=tmpdir)

        router = L3Router(
            store,
            retrieval_engine=custom_retrieval,
            intent_router=custom_intent,
            ingestion_pipeline=custom_ingestion,
            fact_checker=custom_fact_checker,
            quality_manager=custom_quality,
            graph_reasoner=custom_graph,
            ontology_registry=custom_ontology,
            persistence_manager=custom_persistence,
        )
        app = router.create_app()
        assert isinstance(app, Starlette)

    def test_empty_store_路由器(self) -> None:
        """空知识库也能创建路由器."""
        store = KnowledgeStore()
        tmpdir = tempfile.mkdtemp()
        router = L3Router(store, persistence_manager=PersistenceManager(store, base_path=tmpdir))
        app = router.create_app()
        assert isinstance(app, Starlette)
