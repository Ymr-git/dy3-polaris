"""L3 领域知识层 — REST API 路由层.

基于 Starlette 构建, 将 L3 知识层的完整功能暴露为 RESTful JSON API。
遵循与 L6 API 一致的设计模式: 统一响应格式、CORS 中间件、异常统一处理。

融合世界先进方案的 API 设计:
- Strapi / Directus: 资源导向 CRUD + 分页 + 过滤
- LlamaIndex QueryEngine API: 检索即服务
- Haystack Pipeline REST API: 摄入管道端点化
- Wikidata API: 实体/三元组 REST 操作
- Neo4j REST API: 图推理 Cypher 端点
- Elasticsearch API: 混合检索 + 重排
- OpenAPI 3.0: 资源描述与 schema
- JSON:API spec: 统一响应结构
- PostgREST: 数据库直接映射 REST
- Supabase API: 实时 CRUD + RPC

设计原则:
- 资源导向 URL 设计 (RESTful 语义)
- 统一响应格式: {"code": 0, "data": ..., "message": ""}
- 异常统一处理, L3Error 自动映射为 HTTP 响应
- CORS 中间件支持
- 分页 / 过滤 / 排序
- 全链路追踪 ID

使用示例::

    from dy3_polaris.l3 import KnowledgeStore
    from dy3_polaris.l3.api import L3Router

    store = KnowledgeStore()
    router = L3Router(store)
    app = router.create_app()

    # 独立运行
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

    # 或嵌入到 Starlette/FastAPI
    # from starlette.applications import Starlette
    # from starlette.routing import Mount
    # main_app = Starlette(routes=[
    #     Mount("/l3", app=app),
    # ])
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any
from urllib.parse import parse_qs

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from dy3_polaris.l3.exceptions import L3Error
from dy3_polaris.l3.fact_check import FactChecker, StandardValue, StandardValueStore
from dy3_polaris.l3.graph_reasoner import GraphReasoner, ReasoningMode
from dy3_polaris.l3.ingestion import IngestionPipeline
from dy3_polaris.l3.intent_router import IntentRouter
from dy3_polaris.l3.models import (
    EntityType,
    KnowledgeEntity,
    KnowledgeTriple,
    QualityScore,
    RelationType,
    RetrievalFilter,
)
from dy3_polaris.l3.ontology import OntologyRegistry
from dy3_polaris.l3.persistence import PersistenceManager
from dy3_polaris.l3.quality_manager import QualityManager
from dy3_polaris.l3.retrieval import RetrievalEngine
from dy3_polaris.l3.store import KnowledgeStore

_logger = logging.getLogger("dy3_polaris.l3.api.router")


# ============================================================
# 统一响应
# ============================================================

# 响应信封单点 (SSOT: shared/contract.py)
from dy3_polaris.shared.contract import err as _err, ok as _ok


def _l3_error_to_dict(err: L3Error) -> dict[str, Any]:
    """将 L3Error 转为响应字典."""
    return _err(-32400, err.__class__.__name__, str(err))


def _safe_model_dump(obj: Any) -> Any:
    """安全地将 Pydantic 模型或 dataclass 转为可 JSON 序列化的字典.

    处理:
    - Pydantic BaseModel: model_dump(mode="json")
    - dataclass: __dict__ 深度转换
    - list/tuple: 递归处理
    - dict: 递归处理值
    - 其他: 原样返回
    """
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _safe_model_dump(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_model_dump(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _safe_model_dump(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # 枚举或其他
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


# ============================================================
# 路由处理器
# ============================================================

class _RouteHandlers:
    """将 L3 知识层方法适配为 Starlette Request→Response 处理器.

    每个处理器方法:
    1. 解析请求参数 (path/query/body)
    2. 调用 L3 组件方法
    3. 将异常转为统一错误响应
    4. 返回 JSONResponse
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        retrieval_engine: RetrievalEngine | None = None,
        intent_router: IntentRouter | None = None,
        ingestion_pipeline: IngestionPipeline | None = None,
        fact_checker: FactChecker | None = None,
        quality_manager: QualityManager | None = None,
        graph_reasoner: GraphReasoner | None = None,
        ontology_registry: OntologyRegistry | None = None,
        persistence_manager: PersistenceManager | None = None,
        embedding_manager: Any | None = None,
    ) -> None:
        self._store = store
        self._retrieval = retrieval_engine or RetrievalEngine(store)
        self._intent_router = intent_router or IntentRouter(store)
        self._ingestion = ingestion_pipeline or IngestionPipeline(store)
        self._fact_checker = fact_checker or FactChecker()
        self._quality_mgr = quality_manager or QualityManager()
        self._graph_reasoner = graph_reasoner or GraphReasoner(store)
        self._ontology = ontology_registry or OntologyRegistry()
        self._persistence = persistence_manager or PersistenceManager(
            store, base_path="/tmp/dy3_polaris_l3_snapshots",
        )
        # 嵌入管理器 (可选): 提供后检索端点自动生成 query_vector, 启用语义检索
        self._embedding = embedding_manager

    # ---- 健康检查 ----

    async def health(self, request: Request) -> JSONResponse:
        """GET /l3/health — L3 知识层健康检查."""
        return JSONResponse(_ok({
            "status": "healthy",
            "layer": "L3",
            "entity_count": self._store.entity_count(),
            "triple_count": self._store.triple_count(),
            "timestamp": time.time(),
        }))

    # ---- 知识实体管理 (CRUD) ----

    async def create_entity(self, request: Request) -> JSONResponse:
        """POST /l3/entities — 创建知识实体.

        请求体:
            name: 实体名称 (必填)
            entity_type: 实体类型 (必填, 如 "CHEMICAL_COMPOUND")
            description: 描述
            domain: 领域
            properties: 属性字典
            identifiers: 标识符映射
            tags: 标签列表
            aliases: 别名列表
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        try:
            entity_type_str = body.get("entity_type", "concept")
            entity_type = EntityType(entity_type_str)
            entity = KnowledgeEntity(
                name=body["name"],
                entity_type=entity_type,
                description=body.get("description", ""),
                domain=body.get("domain", "general"),
                properties=body.get("properties", {}),
                identifiers=body.get("identifiers", {}),
                tags=body.get("tags", []),
                aliases=body.get("aliases", []),
            )
            created = self._store.add_entity(entity)
            return JSONResponse(_ok(_safe_model_dump(created)), status_code=201)
        except L3Error as e:
            return JSONResponse(_l3_error_to_dict(e), status_code=400)
        except (KeyError, ValueError) as e:
            return JSONResponse(_err(-32700, f"参数错误: {e}"), status_code=400)
        except Exception as e:
            _logger.exception("创建实体失败")
            return JSONResponse(_err(-32400, "创建实体失败", str(e)), status_code=500)

    async def list_entities(self, request: Request) -> JSONResponse:
        """GET /l3/entities — 列出知识实体.

        查询参数:
            entity_type: 按类型过滤
            domain: 按领域过滤
            limit: 分页大小 (默认 20, 最大 100)
            offset: 偏移量 (默认 0)
        """
        try:
            qp = request.query_params
            entity_type = qp.get("entity_type")
            domain = qp.get("domain")
            limit = min(int(qp.get("limit", 20)), 100)
            offset = max(int(qp.get("offset", 0)), 0)

            et = EntityType(entity_type) if entity_type else None

            # EntityStore.list_entities 不支持 entity_type/domain 过滤,
            # 需在取出后手动过滤
            if et:
                entities = self._store.entity_store.find_by_type(et)
            elif domain:
                entities = self._store.entity_store.find_by_domain(domain)
            else:
                # 无过滤: 全部实体, 按"有价值类型优先"排序
                # (chemical_compound > material > method > paper > 其他 > concept),
                # 让知识库列表优先展示化学式/材料等核心实体, 而非 concept 碎片.
                all_entities = list(self._store.entity_store._entities.values())
                type_rank = {
                    "chemical_compound": 0,
                    "material": 1,
                    "method": 2,
                    "paper": 3,
                    "textbook": 4,
                    "experiment": 5,
                    "concept": 6,
                }
                all_entities.sort(key=lambda e: (type_rank.get(e.entity_type.value, 7), e.name or ""))
                entities = all_entities

            # 若按类型/领域过滤了, 手动分页
            if et or domain:
                has_more = len(entities) > offset + limit
                items = entities[offset : offset + limit]
            else:
                has_more = len(entities) > offset + limit
                items = entities[offset : offset + limit]

            return JSONResponse(_ok({
                "items": [_safe_model_dump(e) for e in items],
                # total 应为过滤后的实体数 (而非全库总数), 让前端分页正确
                "total": len(entities),
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
            }))
        except Exception as e:
            _logger.exception("列出实体失败")
            return JSONResponse(_err(-32400, "列出实体失败", str(e)), status_code=500)

    async def get_entity(self, request: Request) -> JSONResponse:
        """GET /l3/entities/{id} — 获取单个实体."""
        eid = request.path_params["id"]
        entity = self._store.get_entity(eid)
        if entity is None:
            return JSONResponse(_err(-32601, f"实体未找到: {eid}"), status_code=404)
        return JSONResponse(_ok(_safe_model_dump(entity)))

    async def update_entity(self, request: Request) -> JSONResponse:
        """PUT /l3/entities/{id} — 更新实体.

        请求体: 要更新的字段 (name, description, properties, tags, ...)
        """
        eid = request.path_params["id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        existing = self._store.get_entity(eid)
        if existing is None:
            return JSONResponse(_err(-32601, f"实体未找到: {eid}"), status_code=404)

        try:
            updated = self._store.update_entity(
                eid,
                changed_by=body.pop("changed_by", "api"),
                reason=body.pop("reason", "REST API 更新"),
                **body,
            )
            return JSONResponse(_ok(_safe_model_dump(updated)))
        except L3Error as e:
            return JSONResponse(_l3_error_to_dict(e), status_code=400)
        except Exception as e:
            _logger.exception("更新实体失败")
            return JSONResponse(_err(-32400, "更新实体失败", str(e)), status_code=500)

    async def delete_entity(self, request: Request) -> JSONResponse:
        """DELETE /l3/entities/{id} — 删除实体."""
        eid = request.path_params["id"]
        removed = self._store.remove_entity(eid)
        if removed is None:
            return JSONResponse(_err(-32601, f"实体未找到: {eid}"), status_code=404)
        return JSONResponse(_ok({"removed": eid}))

    # ---- 三元组管理 ----

    async def create_triple(self, request: Request) -> JSONResponse:
        """POST /l3/triples — 创建三元组.

        请求体:
            subject_id: 主语实体 ID (必填)
            predicate: 谓词 (必填, 如 "RELATED_TO")
            object_id: 宾语实体 ID 或文本
            confidence: 置信度 [0,1]
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        try:
            triple = KnowledgeTriple(
                subject_id=body["subject_id"],
                predicate=body.get("predicate", RelationType.RELATED_TO.value),
                object_id=body.get("object_id", ""),
                confidence=float(body.get("confidence", 1.0)),
            )
            created = self._store.add_triple(triple)
            return JSONResponse(_ok(_safe_model_dump(created)), status_code=201)
        except L3Error as e:
            return JSONResponse(_l3_error_to_dict(e), status_code=400)
        except (KeyError, ValueError) as e:
            return JSONResponse(_err(-32700, f"参数错误: {e}"), status_code=400)

    async def list_triples(self, request: Request) -> JSONResponse:
        """GET /l3/triples — 查询三元组.

        查询参数:
            subject_id: 按主语过滤
            predicate: 按谓词过滤
            object_id: 按宾语过滤
            limit: 分页大小
        """
        qp = request.query_params
        subject_id = qp.get("subject_id")
        limit = min(int(qp.get("limit", 50)), 200)

        triples: list[KnowledgeTriple] = []
        if subject_id:
            # 仅返回出边三元组 (subject_id 为该实体的三元组)
            triples = self._store.triple_store.get_outgoing(subject_id)
        else:
            # 获取所有三元组
            ts = self._store.triple_store
            triples = list(ts._triples.values())

        # 谓词/宾语过滤
        predicate = qp.get("predicate")
        if predicate:
            triples = [t for t in triples if t.predicate == predicate]
        object_id = qp.get("object_id")
        if object_id:
            triples = [t for t in triples if t.object_id == object_id]

        # 分页
        triples = triples[:limit]

        return JSONResponse(_ok({
            "items": [_safe_model_dump(t) for t in triples],
            "total": self._store.triple_count(),
        }))

    async def delete_triple(self, request: Request) -> JSONResponse:
        """DELETE /l3/triples/{id} — 删除三元组."""
        tid = request.path_params["id"]
        removed = self._store.triple_store.remove_triple(tid)
        if removed is None:
            return JSONResponse(_err(-32601, f"三元组未找到: {tid}"), status_code=404)
        return JSONResponse(_ok({"removed": tid}))

    # ---- 知识检索 ----

    async def retrieve_keyword(self, request: Request) -> JSONResponse:
        """POST /l3/retrieve/keyword — 关键词检索.

        请求体:
            query: 查询文本 (必填)
            top_k: 返回结果数 (默认 10)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        query = body.get("query", "")
        if not query:
            return JSONResponse(_err(-32700, "query 不能为空"), status_code=400)

        top_k = min(int(body.get("top_k", 10)), 100)
        result = self._retrieval.keyword_search(query, top_k=top_k)
        return JSONResponse(_ok(_safe_model_dump(result)))

    async def retrieve_vector(self, request: Request) -> JSONResponse:
        """POST /l3/retrieve/vector — 向量检索.

        请求体:
            query_vector: 查询向量 (必填)
            query: 查询文本 (用于重排)
            top_k: 返回结果数
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        query_vector = body.get("query_vector", [])
        if not query_vector:
            return JSONResponse(_err(-32700, "query_vector 不能为空"), status_code=400)

        top_k = min(int(body.get("top_k", 10)), 100)
        result = self._retrieval.vector_search(
            query_vector,
            query=body.get("query", ""),
            top_k=top_k,
        )
        return JSONResponse(_ok(_safe_model_dump(result)))

    async def retrieve_hybrid(self, request: Request) -> JSONResponse:
        """POST /l3/retrieve/hybrid — 混合检索.

        请求体:
            query: 查询文本 (必填)
            query_vector: 查询向量 (可选)
            top_k: 返回结果数
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        query = body.get("query", "")
        if not query:
            return JSONResponse(_err(-32700, "query 不能为空"), status_code=400)

        top_k = min(int(body.get("top_k", 10)), 100)
        query_vector = body.get("query_vector")

        if query_vector:
            result = self._retrieval.hybrid_search(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
            )
        else:
            result = self._retrieval.hybrid_search(
                query=query,
                top_k=top_k,
            )
        return JSONResponse(_ok(_safe_model_dump(result)))

    async def retrieve_intent(self, request: Request) -> JSONResponse:
        """POST /l3/retrieve/intent — 意图驱动路由检索.

        请求体:
            query: 查询文本 (必填)
            top_k: 返回结果数
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        query = body.get("query", "")
        if not query:
            return JSONResponse(_err(-32700, "query 不能为空"), status_code=400)

        top_k = min(int(body.get("top_k", 10)), 100)
        # 语义检索: 若有嵌入管理器则自动生成 query_vector (未配置时降级关键词/图检索)
        query_vector = None
        if self._embedding is not None:
            try:
                query_vector = self._embedding.embed(query).vector
            except Exception as exc:  # noqa: BLE001 (模型未就绪等, 降级检索)
                _logger.warning("query 向量生成失败, 降级检索: %s", exc)
        routed = self._intent_router.route(
            query, top_k=top_k, query_vector=query_vector
        )

        return JSONResponse(_ok({
            "intent": _safe_model_dump(routed.intent),
            "retrieval_result": _safe_model_dump(routed.retrieval_result),
            "total_time_ms": routed.total_time_ms,
        }))

    # ---- 知识摄入 ----

    async def ingest(self, request: Request) -> JSONResponse:
        """POST /l3/ingest — 知识摄入管道.

        请求体:
            content: 文档内容 (必填)
            document_id: 文档 ID (必填)
            metadata: 来源元数据
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        content = body.get("content", "")
        doc_id = body.get("document_id", "")
        if not content or not doc_id:
            return JSONResponse(
                _err(-32700, "content 和 document_id 不能为空"),
                status_code=400,
            )

        try:
            result = self._ingestion.ingest(
                content=content,
                document_id=doc_id,
                metadata=body.get("metadata"),
            )
            return JSONResponse(_ok(_safe_model_dump(result)), status_code=201)
        except L3Error as e:
            return JSONResponse(_l3_error_to_dict(e), status_code=400)
        except Exception as e:
            _logger.exception("知识摄入失败")
            return JSONResponse(_err(-32400, "知识摄入失败", str(e)), status_code=500)

    async def ingest_batch(self, request: Request) -> JSONResponse:
        """POST /l3/ingest/batch — 批量摄入.

        请求体:
            items: [{content, document_id, metadata}, ...]
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        items = body.get("items", [])
        if not items:
            return JSONResponse(_err(-32700, "items 不能为空"), status_code=400)

        try:
            result = self._ingestion.ingest_batch(items)
            return JSONResponse(_ok(_safe_model_dump(result)), status_code=201)
        except L3Error as e:
            return JSONResponse(_l3_error_to_dict(e), status_code=400)

    # ---- 事实校验 ----

    async def fact_check(self, request: Request) -> JSONResponse:
        """POST /l3/fact-check — 事实校验.

        请求体:
            content: 待校验内容 (必填)
            kp_ids: 限定知识点 ID 列表 (可选)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        content = body.get("content", "")
        if not content:
            return JSONResponse(_err(-32700, "content 不能为空"), status_code=400)

        report = self._fact_checker.check(
            content,
            kp_ids=body.get("kp_ids"),
        )
        return JSONResponse(_ok(_safe_model_dump(report)))

    async def list_standards(self, request: Request) -> JSONResponse:
        """GET /l3/standards — 获取标准值列表."""
        store = self._fact_checker.standard_store
        standards = store.list_all()
        return JSONResponse(_ok({
            "items": [_safe_model_dump(s) for s in standards],
            "total": len(standards),
        }))

    async def add_standard(self, request: Request) -> JSONResponse:
        """POST /l3/standards — 添加标准值.

        请求体:
            kp_id: 知识点 ID (必填)
            param_name: 参数名 (必填, 如 "emission_wavelength")
            standard_value: 标准值 (必填)
            tolerance: 容差 (可选, 默认取参数默认容差)
            tolerance_type: 容差类型 (absolute/relative/threshold)
            unit: 单位
            source_ref: 来源引用
            source_type: 来源类型 (standard/literature/calculated)
            confidence: 置信度
            notes: 备注
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        try:
            param_name = body["param_name"]
            # 如果未提供 tolerance, 尝试取默认值
            default_tol = self._fact_checker.standard_store.get_default_tolerance(param_name)

            standard = StandardValue(
                kp_id=body["kp_id"],
                param_name=param_name,
                standard_value=float(body["standard_value"]),
                tolerance=float(body.get("tolerance", default_tol["tolerance"] if default_tol else 0.05)),
                tolerance_type=body.get("tolerance_type", default_tol["tolerance_type"] if default_tol else "absolute"),
                unit=body.get("unit", default_tol["unit"] if default_tol else ""),
                source_type=body.get("source_type", "standard"),
                source_ref=body.get("source_ref", ""),
                confidence=float(body.get("confidence", 1.0)),
                notes=body.get("notes", ""),
            )
            self._fact_checker.standard_store.add(standard)
            return JSONResponse(_ok(_safe_model_dump(standard)), status_code=201)
        except (KeyError, ValueError) as e:
            return JSONResponse(_err(-32700, f"参数错误: {e}"), status_code=400)

    # ---- 质量管理 ----

    async def quality_assess(self, request: Request) -> JSONResponse:
        """POST /l3/quality/assess — 单实体质量评估.

        请求体:
            entity_id: 实体 ID (必填)
            context: 评估上下文 (可选)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        eid = body.get("entity_id", "")
        entity = self._store.get_entity(eid)
        if entity is None:
            return JSONResponse(_err(-32601, f"实体未找到: {eid}"), status_code=404)

        context = body.get("context", {})
        context.setdefault("store", self._store)

        result = self._quality_mgr.assess_entity(entity, context=context)
        return JSONResponse(_ok(_safe_model_dump(result)))

    async def quality_assess_batch(self, request: Request) -> JSONResponse:
        """POST /l3/quality/assess/batch — 批量质量评估.

        请求体:
            entity_ids: 实体 ID 列表 (必填)
            context: 评估上下文 (可选)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        ids = body.get("entity_ids", [])
        if not ids:
            return JSONResponse(_err(-32700, "entity_ids 不能为空"), status_code=400)

        entities: list[KnowledgeEntity] = []
        for eid in ids:
            e = self._store.get_entity(eid)
            if e is not None:
                entities.append(e)

        context = body.get("context", {})
        context.setdefault("store", self._store)

        results = self._quality_mgr.assess_batch(entities, context=context)
        return JSONResponse(_ok([_safe_model_dump(r) for r in results]))

    async def quality_assess_global(self, request: Request) -> JSONResponse:
        """POST /l3/quality/assess/global — 全库质量评估."""
        dashboard = self._quality_mgr.assess_global(self._store)
        return JSONResponse(_ok(_safe_model_dump(dashboard)))

    async def quality_detect_conflicts(self, request: Request) -> JSONResponse:
        """POST /l3/quality/conflicts/detect — 冲突检测.

        请求体:
            entity_id: 实体 ID (必填)
            external_claims: 外部声明列表
            history: 历史版本列表
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        eid = body.get("entity_id", "")
        entity = self._store.get_entity(eid)
        if entity is None:
            return JSONResponse(_err(-32601, f"实体未找到: {eid}"), status_code=404)

        conflicts = self._quality_mgr.detect_conflicts(
            entity,
            external_claims=body.get("external_claims"),
            history=body.get("history"),
        )
        return JSONResponse(_ok([_safe_model_dump(c) for c in conflicts]))

    async def quality_resolve_conflict(self, request: Request) -> JSONResponse:
        """POST /l3/quality/conflicts/resolve — 冲突消解.

        请求体:
            conflict: 冲突对象 (必填)
            strategy: 消解策略 (可选)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        from dy3_polaris.l3.models import KnowledgeConflict, ConflictResolutionStrategy

        conflict_data = body.get("conflict", {})
        strategy_str = body.get("strategy")

        try:
            conflict = KnowledgeConflict(**conflict_data)
            strategy = (
                ConflictResolutionStrategy(strategy_str)
                if strategy_str
                else None
            )
            resolved = self._quality_mgr.resolve_conflict(conflict, strategy)
            return JSONResponse(_ok(_safe_model_dump(resolved)))
        except (KeyError, ValueError) as e:
            return JSONResponse(_err(-32700, f"参数错误: {e}"), status_code=400)

    async def quality_dashboard(self, request: Request) -> JSONResponse:
        """GET /l3/quality/dashboard — 质量仪表板."""
        total = self._store.entity_count()
        dashboard = self._quality_mgr.get_dashboard(total_entities=total)
        return JSONResponse(_ok(_safe_model_dump(dashboard)))

    async def quality_get_provenance(self, request: Request) -> JSONResponse:
        """GET /l3/quality/provenance/{id} — 溯源查询."""
        eid = request.path_params["id"]
        prov = self._quality_mgr.get_provenance(eid)
        if prov is None:
            return JSONResponse(_err(-32601, f"溯源未找到: {eid}"), status_code=404)
        return JSONResponse(_ok(_safe_model_dump(prov)))

    async def quality_record_provenance(self, request: Request) -> JSONResponse:
        """POST /l3/quality/provenance — 记录溯源.

        请求体:
            entity_id: 实体 ID (必填)
            activity_type: 活动类型 (必填)
            agent_id: 智能体 ID
            description: 描述
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        try:
            prov = self._quality_mgr.record_provenance(
                entity_id=body["entity_id"],
                activity_type=body["activity_type"],
                agent_id=body.get("agent_id", "api"),
                description=body.get("description", ""),
            )
            return JSONResponse(_ok(_safe_model_dump(prov)), status_code=201)
        except (KeyError, ValueError) as e:
            return JSONResponse(_err(-32700, f"参数错误: {e}"), status_code=400)

    async def quality_audit_log(self, request: Request) -> JSONResponse:
        """GET /l3/quality/audit-log — 审计日志.

        查询参数:
            entity_id: 按实体过滤
            activity_type: 按活动类型过滤
            limit: 返回条数
        """
        qp = request.query_params
        entity_id = qp.get("entity_id")
        activity_type = qp.get("activity_type")
        limit = min(int(qp.get("limit", 50)), 200)

        log = self._quality_mgr.get_audit_log(
            entity_id=entity_id,
            activity_type=activity_type,
            limit=limit,
        )
        return JSONResponse(_ok(log))

    # ---- 溯源链验证 (M-F7 缺口补齐) ----

    async def quality_provenance_chain(self, request: Request) -> JSONResponse:
        """GET /l3/quality/provenance/{id}/chain — 溯源链追踪 + 完整性验证.

        返回:
            {entity_id, verified, chain: [{activity_type, agent_id, integrity_hash, timestamp, description}]}

        语义:
            - 实体不存在 → 404 (查询对象错误)
            - 实体存在但无溯源记录 → 200 + unverifiable (合法状态, 前端据此显示"未验证")
        """
        eid = request.path_params["id"]
        # 实体不存在 → 404, 区别于「实体存在但无溯源」的合法 unverifiable 状态
        if self._store.get_entity(eid) is None:
            return JSONResponse(_err(-32601, f"实体未找到: {eid}"), status_code=404)

        prov = self._quality_mgr.get_provenance(eid)
        if prov is None:
            # 无溯源记录是合法状态 (非错误): 返回 200 + unverifiable, 前端据此显示"未验证"。
            # 避免返回 404 被浏览器记为资源加载失败, 产生控制台错误刷屏。
            return JSONResponse(_ok({
                "entity_id": eid,
                "verified": "unverifiable",
                "chain": [],
            }))

        chain = self._quality_mgr.trace_provenance_chain(eid, max_depth=10)
        try:
            verified = self._quality_mgr.verify_provenance_chain(eid)
        except Exception:
            verified = "unverified"

        chain_payload = []
        for item in chain:
            try:
                d = _safe_model_dump(item)
            except Exception:
                d = item.to_dict() if hasattr(item, "to_dict") else dict(item)
            chain_payload.append({
                "entity_id": d.get("entity_id"),
                "activity_type": d.get("activity_type"),
                "agent_id": d.get("agent_id") or d.get("generated_by_agent"),
                "integrity_hash": d.get("integrity_hash", ""),
                "timestamp": d.get("timestamp", d.get("generated_at", d.get("ts"))),
                "description": d.get("description", d.get("activity_description", "")),
            })
        return JSONResponse(_ok({
            "entity_id": eid,
            "verified": verified.value if hasattr(verified, 'value') else str(verified),
            "chain": chain_payload,
        }))

    # ---- 图推理 ----

    async def graph_reason(self, request: Request) -> JSONResponse:
        """POST /l3/graph/reason — 图推理.

        请求体:
            query: 查询文本 (必填)
            mode: 推理模式 (path_finding/multi_hop/rule_inference/
                  link_prediction/pattern_match/analogy)
            **kwargs: 模式特定参数
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        query = body.get("query", "")
        mode_str = body.get("mode", "path_finding")

        try:
            mode = ReasoningMode(mode_str)
        except ValueError:
            return JSONResponse(
                _err(-32700, f"未知推理模式: {mode_str}"),
                status_code=400,
            )

        kwargs = {k: v for k, v in body.items() if k not in ("query", "mode")}
        result = self._graph_reasoner.reason(query, mode, **kwargs)
        return JSONResponse(_ok(_safe_model_dump(result)))

    async def graph_stats(self, request: Request) -> JSONResponse:
        """GET /l3/graph/stats — 图统计."""
        stats = self._graph_reasoner.get_stats()
        return JSONResponse(_ok(stats))

    async def graph_subgraph(self, request: Request) -> JSONResponse:
        """GET /l3/graph/subgraph — 图谱子图 (节点+边, 供前端力导向图可视化).

        返回核心实体 (chemical_compound/material) 及其关系边, 聚焦 Dy 相关实体,
        避免返回全部 9668 实体导致前端卡顿. 查询参数:
            limit: 最大节点数 (默认 80)
            focus: 聚焦关键词 (默认 Dy, 返回含该词的实体子图)
        """
        qp = request.query_params
        limit = min(int(qp.get("limit", 80)), 200)
        focus = qp.get("focus", "Dy").strip() or "Dy"

        store = self._store
        # 1. 聚焦实体: chemical_compound/material 类型且名称含 focus
        focus_entities = [
            e for e in store.entity_store._entities.values()
            if e.entity_type.value in ("chemical_compound", "material")
            and focus.lower() in (e.name or "").lower()
        ]
        # 若 focus 命中太少, 放宽到全部 chemical_compound/material
        if len(focus_entities) < 5:
            focus_entities = [
                e for e in store.entity_store._entities.values()
                if e.entity_type.value in ("chemical_compound", "material")
            ]

        # 限制初始节点数
        focus_entities = focus_entities[:limit]

        # 2. 收集焦点节点 ID + 找相邻节点 (与焦点有边的实体)
        focus_ids = {e.entity_id for e in focus_entities}
        all_entity = {e.entity_id: e for e in store.entity_store._entities.values()}

        # 遍历三元组, 收集与焦点相连的邻居节点
        neighbor_ids: set[str] = set()
        relevant_edges: list[Any] = []
        for t in store.triple_store._triples.values():
            if t.object_is_literal:
                continue
            s_in = t.subject_id in focus_ids
            o_in = t.object_id in focus_ids
            if s_in or o_in:
                relevant_edges.append(t)
                if s_in and t.object_id in all_entity:
                    neighbor_ids.add(t.object_id)
                if o_in and t.subject_id in all_entity:
                    neighbor_ids.add(t.subject_id)

        # 3. 构建节点 (焦点 + 邻居)
        nodes: list[dict[str, Any]] = []
        entity_id_to_name: dict[str, str] = {}
        seen_nodes: set[str] = set()
        for e in focus_entities:
            if e.entity_id not in seen_nodes:
                seen_nodes.add(e.entity_id)
                nodes.append({"id": e.entity_id, "label": e.name or e.entity_id,
                              "type": e.entity_type.value, "domain": e.domain, "degree": 0})
                entity_id_to_name[e.entity_id] = e.name or e.entity_id
        for nid in neighbor_ids:
            if nid in seen_nodes:
                continue
            e = all_entity.get(nid)
            if e is None:
                continue
            seen_nodes.add(nid)
            nodes.append({"id": nid, "label": e.name or nid,
                          "type": e.entity_type.value, "domain": e.domain, "degree": 0})
            entity_id_to_name[nid] = e.name or nid
            if len(nodes) >= limit * 2:
                break

        # 4. 构建边 (只保留两端都在 nodes 里)
        node_ids = set(entity_id_to_name.keys())
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for t in relevant_edges:
            if t.subject_id in node_ids and t.object_id in node_ids:
                key = (t.subject_id, t.predicate, t.object_id)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append({
                    "source": t.subject_id,
                    "target": t.object_id,
                    "label": t.predicate,
                })
                for n in nodes:
                    if n["id"] == t.subject_id or n["id"] == t.object_id:
                        n["degree"] += 1

        return JSONResponse(_ok({
            "nodes": nodes,
            "edges": edges,
            "focus": focus,
            "total_nodes": store.entity_count(),
            "total_edges": store.triple_count(),
        }))

    async def graph_hierarchy(self, request: Request) -> JSONResponse:
        """GET /l3/graph/hierarchy — 分层知识图谱 (L1-L4 层级实体 + 关系边).

        直接返回 domain 命中层级前缀的实体 (绕过 list_entities 的 type_rank 排序,
        否则 concept 层级实体排最后, 前端 offset 分页拉不到), 供前端分层径向图可视化.
        """
        store = self._store
        prefixes = ("L1", "L2:", "L3:", "L4:", "activator", "property", "application")
        nodes: list[dict[str, Any]] = []
        node_ids: set[str] = set()
        for e in store.entity_store._entities.values():
            d = e.domain or ""
            if not any(d == p or d.startswith(p) for p in prefixes):
                continue
            node_ids.add(e.entity_id)
            nodes.append({
                "id": e.entity_id,
                "label": e.name or e.entity_id,
                "type": e.entity_type.value,
                "domain": d,
                "desc": e.description or "",
                "degree": 0,
            })

        degree: dict[str, int] = {}
        edges: list[dict[str, Any]] = []
        for t in store.triple_store._triples.values():
            if t.subject_id in node_ids and t.object_id in node_ids:
                edges.append({
                    "source": t.subject_id,
                    "target": t.object_id,
                    "label": t.predicate,
                })
                degree[t.subject_id] = degree.get(t.subject_id, 0) + 1
                degree[t.object_id] = degree.get(t.object_id, 0) + 1
        for n in nodes:
            n["degree"] = degree.get(n["id"], 0)

        return JSONResponse(_ok({
            "nodes": nodes,
            "edges": edges,
            "total": len(nodes),
        }))

    async def graph_kp_relations(self, request: Request) -> JSONResponse:
        """GET /l3/graph/kp-relations — 知识点关系图 (48 KP, 迁移后章.节.序号 + 教学关系边).

        从已播种的知识图谱 (self._store) 读取 48 个知识点 (42 重编号 + 6 第 6 章新增),
        按新 ID (章.节.序号) 输出, 附章/节归属; 边覆盖全部教学关系 (前提/类比/因果/
        表征/上下位/应用), 并带来源标记 source_id (rule=规则边 / llm=LLM 补边 / ""=手工)。
        deepens (深化) 是 prerequisite_of 的反向, 由前端按箭头方向体现, 不单独成边。
        """
        from dy3_polaris.l2.kp_catalog import (
            CHAPTER_LABELS,
            NEW_KP_NAMES,
            NEW_KP_TO_CHAPTER,
            NEW_KP_TO_SECTION,
            SECTION_LABELS,
            to_new_id,
        )

        store = self._store
        TEACHING = {"prerequisite_of", "analogous_to", "affects",
                    "characterized_by", "subconcept_of", "applies_to"}

        def _new_id(eid: str) -> str:
            raw = eid.split(":", 1)[1] if ":" in eid else eid
            if "." in raw and raw[0].isdigit():
                return raw
            return to_new_id(raw)

        # 知识点节点 (48)
        kp_entities = [
            e for e in store.entity_store._entities.values()
            if e.entity_id.startswith("kp:")
        ]
        nodes: list[dict[str, Any]] = []
        for e in kp_entities:
            nid = _new_id(e.entity_id)
            ch = NEW_KP_TO_CHAPTER.get(nid, "")
            sec = NEW_KP_TO_SECTION.get(nid, "")
            nodes.append({
                "id": f"kp:{nid}",
                "kp_id": nid,
                "label": NEW_KP_NAMES.get(nid, e.name),
                "chapter": ch,
                "chapter_label": CHAPTER_LABELS.get(ch, ""),
                "section": sec,
                "section_label": SECTION_LABELS.get(sec, ""),
                "entity_id": e.entity_id,
            })

        # 教学关系边 (KP → KP)
        kp_ids = {e.entity_id for e in kp_entities}
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for t in store.triple_store._triples.values():
            if t.predicate not in TEACHING:
                continue
            if t.subject_id not in kp_ids or t.object_id not in kp_ids:
                continue
            sig = (t.subject_id, t.predicate, t.object_id)
            if sig in seen:
                continue
            seen.add(sig)
            edges.append({
                "source": f"kp:{_new_id(t.subject_id)}",
                "target": f"kp:{_new_id(t.object_id)}",
                "label": t.predicate,
                "source_id": getattr(t, "source_id", "") or "rule",
                "confidence": float(t.confidence),
            })

        return JSONResponse(_ok({
            "nodes": nodes,
            "edges": edges,
            "total": len(nodes),
            "relation_types": sorted(TEACHING),
        }))

    async def graph_role_kp(self, request: Request) -> JSONResponse:
        """GET /l3/graph/role-kp — 职业角色 + 角色-知识点关联 (多职业维度).

        返回 7 种职业角色 (学生/教师/材料工程师/照明设计师/研究员/健康专家/
        质量工程师) 各自关注的知识点子集 (含权重), 供按角色的个性化学习路径推荐。
        """
        from dy3_polaris.l2.kp_catalog import KP_NAMES
        from dy3_polaris.l2.kp_roles import role_kps, role_list

        roles: list[dict[str, Any]] = []
        for r in role_list():
            kps = [
                {"kp_id": k, "name": KP_NAMES.get(k, k), "weight": w}
                for k, w in sorted(role_kps(r["role_id"]).items())
            ]
            roles.append({**r, "kps": kps})
        return JSONResponse(_ok({"roles": roles, "total": len(roles)}))

    # ---- 图消费层 (P3) ----

    async def graph_learning_path(self, request: Request) -> JSONResponse:
        """GET /l3/graph/learning-path — 学习路径 (Dijkstra 加权最短路径 + 可读解释).

        查询参数:
            start: 起点实体 ID (如 kp:2.1.1 或 ion:Dy3+)
            goal : 终点实体 ID
            max_depth: 最大深度 (默认 10)
        """
        qp = request.query_params
        start = qp.get("start", "").strip()
        goal = qp.get("goal", "").strip()
        if not start or not goal:
            return JSONResponse(_err(-32700, "start 与 goal 参数必填"), status_code=400)
        try:
            max_depth = min(int(qp.get("max_depth", 10)), 20)
        except ValueError:
            max_depth = 10
        try:
            from dy3_polaris.l3.graph_consume import learning_path
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(_err(-32603, f"图消费层不可用: {exc}"), status_code=500)
        result = learning_path(self._store, start, goal, max_depth=max_depth)
        if result is None:
            return JSONResponse(
                _err(-32602, f"未找到 {start} → {goal} 的路径"),
                status_code=404,
            )
        return JSONResponse(_ok(result))

    async def graph_analogy(self, request: Request) -> JSONResponse:
        """POST /l3/graph/analogy — 类比推理 (关系模式迁移).

        请求体:
            source_a / source_b: 源关系对 (A, B)
            target: 目标实体 ID (把 A→B 的关系模式迁移到 target)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)
        src_a = str(body.get("source_a") or body.get("source_pair", [None])[0] or "")
        src_b = str(body.get("source_b") or (body.get("source_pair") or [None, None])[1] or "")
        target = str(body.get("target") or "")
        if not src_a or not src_b or not target:
            return JSONResponse(_err(-32700, "source_a/source_b/target 必填"), status_code=400)
        try:
            from dy3_polaris.l3.graph_consume import analogy
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(_err(-32603, f"图消费层不可用: {exc}"), status_code=500)
        return JSONResponse(_ok({"source_pair": [src_a, src_b], "target": target,
                                 "results": analogy(self._store, (src_a, src_b), target)}))

    async def graph_recall(self, request: Request) -> JSONResponse:
        """POST /l3/graph/recall — 多跳召回 (多类型图 → 事实 + 概念实体证据).

        请求体:
            query: 查询文本 (必填)
            max_hop: 最大跳数 (默认 2)
            max_facts / max_concepts: 事实/概念实体上限
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)
        query = str(body.get("query") or "").strip()
        if not query:
            return JSONResponse(_err(-32700, "query 必填"), status_code=400)
        try:
            from dy3_polaris.l3.graph_consume import recall, resolve_seeds
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(_err(-32603, f"图消费层不可用: {exc}"), status_code=500)
        seeds = resolve_seeds(query, self._store)
        evidence = recall(
            query, self._store,
            max_hop=int(body.get("max_hop", 2)),
            max_facts=int(body.get("max_facts", 8)),
            max_concepts=int(body.get("max_concepts", 8)),
        )
        return JSONResponse(_ok({"query": query, "seeds": seeds,
                                 "evidence": evidence, "count": len(evidence)}))

    async def graph_provenance(self, request: Request) -> JSONResponse:
        """GET /l3/graph/provenance/{id} — 实体溯源 (入/出边 + 邻接实体 + 来源标记)."""
        eid = request.path_params["id"]
        try:
            from dy3_polaris.l3.graph_consume import provenance
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(_err(-32603, f"图消费层不可用: {exc}"), status_code=500)
        result = provenance(self._store, eid)
        if not result.get("exists"):
            return JSONResponse(_err(-32602, f"实体不存在: {eid}"), status_code=404)
        return JSONResponse(_ok(result))

    # ---- 本体管理 ----

    async def ontology_domains(self, request: Request) -> JSONResponse:
        """GET /l3/ontology/domains — 列出所有领域."""
        domains = self._ontology.list_domains()
        return JSONResponse(_ok({"domains": domains, "count": len(domains)}))

    async def get_ontology(self, request: Request) -> JSONResponse:
        """GET /l3/ontology/{domain} — 获取领域本体."""
        domain = request.path_params["domain"]
        onto = self._ontology.get_ontology(domain)
        if onto is None:
            return JSONResponse(_err(-32601, f"本体未找到: {domain}"), status_code=404)
        return JSONResponse(_ok(_safe_model_dump(onto)))

    async def ontology_validate(self, request: Request) -> JSONResponse:
        """POST /l3/ontology/validate — 本体验证.

        请求体:
            domain: 领域 (必填)
            entity_type: 实体类型 (必填)
            properties: 属性字典 (必填)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        domain = body.get("domain", "")
        et_str = body.get("entity_type", "CONCEPT")
        properties = body.get("properties", {})

        try:
            et = EntityType(et_str)
            violations = self._ontology.validate_full(domain, et, properties)
            return JSONResponse(_ok({
                "valid": len(violations) == 0,
                "violations": violations,
                "violation_count": len(violations),
            }))
        except Exception as e:
            return JSONResponse(_err(-32400, "本体验证失败", str(e)), status_code=500)

    # ---- 持久化 ----

    async def persistence_snapshot(self, request: Request) -> JSONResponse:
        """POST /l3/persistence/snapshot — 保存快照.

        请求体:
            path: 快照路径 (可选, 默认自动生成)
        """
        try:
            body = await request.json() if await request.body() else {}
        except Exception:
            body = {}

        path = body.get("path")
        saved = self._persistence.save_snapshot(path)
        return JSONResponse(_ok({"path": str(saved)}))

    async def persistence_restore(self, request: Request) -> JSONResponse:
        """POST /l3/persistence/restore — 恢复快照.

        请求体:
            path: 快照路径 (必填)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        path = body.get("path", "")
        if not path:
            return JSONResponse(_err(-32700, "path 不能为空"), status_code=400)

        try:
            self._persistence.load_snapshot(path)
            return JSONResponse(_ok({"restored": path}))
        except Exception as e:
            return JSONResponse(_err(-32400, "恢复快照失败", str(e)), status_code=500)

    # ---- 知识库统计 ----

    async def stats(self, request: Request) -> JSONResponse:
        """GET /l3/stats — 知识库统计信息."""
        store = self._store
        entity_types: dict[str, int] = {}
        domains: dict[str, int] = {}
        for e in store.entity_store._entities.values():
            et = e.entity_type.value
            entity_types[et] = entity_types.get(et, 0) + 1
            d = e.domain
            domains[d] = domains.get(d, 0) + 1

        return JSONResponse(_ok({
            "entity_count": store.entity_count(),
            "triple_count": store.triple_count(),
            "chunk_count": store.chunk_store.count() if hasattr(store, "chunk_store") else 0,
            "entity_types": entity_types,
            "domains": domains,
            "ontology_domains": self._ontology.list_domains(),
            "quality_assessment_count": self._quality_mgr.assessment_count,
            "provenance_count": self._quality_mgr.provenance_count,
            "timestamp": time.time(),
        }))


# ============================================================
# L3 路由器
# ============================================================

class L3Router:
    """L3 知识层 REST API 路由器.

    将 KnowledgeStore 及相关组件暴露为 RESTful API。
    遵循与 L6Router 一致的设计模式。

    使用示例::

        from dy3_polaris.l3 import KnowledgeStore
        from dy3_polaris.l3.api import L3Router

        store = KnowledgeStore()
        router = L3Router(store)
        app = router.create_app()

        # 独立运行
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)

        # 或嵌入到主应用
        from starlette.routing import Mount
        main_routes = [
            Mount("/l3", app=router.create_app()),
        ]
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        retrieval_engine: RetrievalEngine | None = None,
        intent_router: IntentRouter | None = None,
        ingestion_pipeline: IngestionPipeline | None = None,
        fact_checker: FactChecker | None = None,
        quality_manager: QualityManager | None = None,
        graph_reasoner: GraphReasoner | None = None,
        ontology_registry: OntologyRegistry | None = None,
        persistence_manager: PersistenceManager | None = None,
        embedding_manager: Any | None = None,
        cors_origins: list[str] | None = None,
    ) -> None:
        """初始化 L3 路由器.

        Args:
            store: 知识存储 (必填)
            retrieval_engine: 检索引擎 (None 自动创建)
            intent_router: 意图路由 (None 自动创建)
            ingestion_pipeline: 摄入管道 (None 自动创建)
            fact_checker: 事实校验器 (None 自动创建)
            quality_manager: 质量管理器 (None 自动创建)
            graph_reasoner: 图推理器 (None 自动创建)
            ontology_registry: 本体注册中心 (None 自动创建)
            persistence_manager: 持久化管理器 (None 自动创建)
            embedding_manager: 嵌入管理器 (可选, 提供后启用语义检索)
            cors_origins: CORS 允许的源 (默认 ["*"])
        """
        self._store = store
        self._cors_origins = cors_origins or ["*"]
        self._handlers = _RouteHandlers(
            store,
            retrieval_engine=retrieval_engine,
            intent_router=intent_router,
            ingestion_pipeline=ingestion_pipeline,
            fact_checker=fact_checker,
            quality_manager=quality_manager,
            graph_reasoner=graph_reasoner,
            ontology_registry=ontology_registry,
            persistence_manager=persistence_manager,
            embedding_manager=embedding_manager,
        )

    @property
    def store(self) -> KnowledgeStore:
        """获取关联的知识存储 (供统一应用集成使用)."""
        return self._store

    def create_app(self) -> Starlette:
        """创建 Starlette 应用实例.

        Returns:
            配置好的 Starlette 应用, 可直接传给 uvicorn.run()
            或通过 Mount 嵌入到主应用。
        """
        h = self._handlers

        routes = [
            # 健康检查
            Route("/health", h.health, methods=["GET"]),

            # 知识实体管理 (CRUD) — 静态路由在动态路由之前
            Route("/entities", h.create_entity, methods=["POST"]),
            Route("/entities", h.list_entities, methods=["GET"]),
            Route("/entities/{id}", h.get_entity, methods=["GET"]),
            Route("/entities/{id}", h.update_entity, methods=["PUT"]),
            Route("/entities/{id}", h.delete_entity, methods=["DELETE"]),

            # 三元组管理
            Route("/triples", h.create_triple, methods=["POST"]),
            Route("/triples", h.list_triples, methods=["GET"]),
            Route("/triples/{id}", h.delete_triple, methods=["DELETE"]),

            # 知识检索
            Route("/retrieve/keyword", h.retrieve_keyword, methods=["POST"]),
            Route("/retrieve/vector", h.retrieve_vector, methods=["POST"]),
            Route("/retrieve/hybrid", h.retrieve_hybrid, methods=["POST"]),
            Route("/retrieve/intent", h.retrieve_intent, methods=["POST"]),

            # 知识摄入
            Route("/ingest", h.ingest, methods=["POST"]),
            Route("/ingest/batch", h.ingest_batch, methods=["POST"]),

            # 事实校验
            Route("/fact-check", h.fact_check, methods=["POST"]),
            Route("/standards", h.list_standards, methods=["GET"]),
            Route("/standards", h.add_standard, methods=["POST"]),

            # 质量管理
            Route("/quality/assess", h.quality_assess, methods=["POST"]),
            Route("/quality/assess/batch", h.quality_assess_batch, methods=["POST"]),
            Route("/quality/assess/global", h.quality_assess_global, methods=["POST"]),
            Route("/quality/conflicts/detect", h.quality_detect_conflicts, methods=["POST"]),
            Route("/quality/conflicts/resolve", h.quality_resolve_conflict, methods=["POST"]),
            Route("/quality/dashboard", h.quality_dashboard, methods=["GET"]),
            Route("/quality/provenance", h.quality_record_provenance, methods=["POST"]),
            Route("/quality/provenance/{id}/chain", h.quality_provenance_chain, methods=["GET"]),
            Route("/quality/provenance/{id}", h.quality_get_provenance, methods=["GET"]),
            Route("/quality/audit-log", h.quality_audit_log, methods=["GET"]),

            # 图推理
            Route("/graph/reason", h.graph_reason, methods=["POST"]),
            Route("/graph/stats", h.graph_stats, methods=["GET"]),
            Route("/graph/subgraph", h.graph_subgraph, methods=["GET"]),
            Route("/graph/hierarchy", h.graph_hierarchy, methods=["GET"]),
            Route("/graph/kp-relations", h.graph_kp_relations, methods=["GET"]),
            Route("/graph/role-kp", h.graph_role_kp, methods=["GET"]),
            Route("/graph/learning-path", h.graph_learning_path, methods=["GET"]),
            Route("/graph/analogy", h.graph_analogy, methods=["POST"]),
            Route("/graph/recall", h.graph_recall, methods=["POST"]),
            Route("/graph/provenance/{id}", h.graph_provenance, methods=["GET"]),

            # 本体管理
            Route("/ontology/domains", h.ontology_domains, methods=["GET"]),
            Route("/ontology/validate", h.ontology_validate, methods=["POST"]),
            Route("/ontology/{domain}", h.get_ontology, methods=["GET"]),

            # 持久化
            Route("/persistence/snapshot", h.persistence_snapshot, methods=["POST"]),
            Route("/persistence/restore", h.persistence_restore, methods=["POST"]),

            # 知识库统计
            Route("/stats", h.stats, methods=["GET"]),
        ]

        middleware = []
        if self._cors_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=self._cors_origins,
                    allow_methods=["*"] if "*" in self._cors_origins
                                   else ["GET", "POST", "PUT", "DELETE"],
                    allow_headers=["*"],
                )
            )

        app = Starlette(routes=routes, middleware=middleware)
        return app

    def get_routes_summary(self) -> list[dict[str, str]]:
        """获取所有路由摘要 (用于文档/发现).

        Returns:
            [{"path": ..., "methods": [...], "description": ...}]
        """
        return [
            {"path": "/health", "methods": ["GET"], "description": "L3 知识层健康检查"},
            {"path": "/entities", "methods": ["POST"], "description": "创建知识实体"},
            {"path": "/entities", "methods": ["GET"], "description": "列出知识实体"},
            {"path": "/entities/{id}", "methods": ["GET"], "description": "获取单个实体"},
            {"path": "/entities/{id}", "methods": ["PUT"], "description": "更新实体"},
            {"path": "/entities/{id}", "methods": ["DELETE"], "description": "删除实体"},
            {"path": "/triples", "methods": ["POST"], "description": "创建三元组"},
            {"path": "/triples", "methods": ["GET"], "description": "查询三元组"},
            {"path": "/triples/{id}", "methods": ["DELETE"], "description": "删除三元组"},
            {"path": "/retrieve/keyword", "methods": ["POST"], "description": "关键词检索"},
            {"path": "/retrieve/vector", "methods": ["POST"], "description": "向量检索"},
            {"path": "/retrieve/hybrid", "methods": ["POST"], "description": "混合检索"},
            {"path": "/retrieve/intent", "methods": ["POST"], "description": "意图驱动检索"},
            {"path": "/ingest", "methods": ["POST"], "description": "知识摄入"},
            {"path": "/ingest/batch", "methods": ["POST"], "description": "批量摄入"},
            {"path": "/fact-check", "methods": ["POST"], "description": "事实校验"},
            {"path": "/standards", "methods": ["GET"], "description": "获取标准值列表"},
            {"path": "/standards", "methods": ["POST"], "description": "添加标准值"},
            {"path": "/quality/assess", "methods": ["POST"], "description": "单实体质量评估"},
            {"path": "/quality/assess/batch", "methods": ["POST"], "description": "批量质量评估"},
            {"path": "/quality/assess/global", "methods": ["POST"], "description": "全库质量评估"},
            {"path": "/quality/conflicts/detect", "methods": ["POST"], "description": "冲突检测"},
            {"path": "/quality/conflicts/resolve", "methods": ["POST"], "description": "冲突消解"},
            {"path": "/quality/dashboard", "methods": ["GET"], "description": "质量仪表板"},
            {"path": "/quality/provenance", "methods": ["POST"], "description": "记录溯源"},
            {"path": "/quality/provenance/{id}", "methods": ["GET"], "description": "溯源查询"},
            {"path": "/quality/audit-log", "methods": ["GET"], "description": "审计日志"},
            {"path": "/graph/reason", "methods": ["POST"], "description": "图推理"},
            {"path": "/graph/stats", "methods": ["GET"], "description": "图统计"},
            {"path": "/graph/learning-path", "methods": ["GET"], "description": "学习路径 (Dijkstra)"},
            {"path": "/graph/analogy", "methods": ["POST"], "description": "类比推理"},
            {"path": "/graph/recall", "methods": ["POST"], "description": "多跳召回 (多类型图)"},
            {"path": "/graph/provenance/{id}", "methods": ["GET"], "description": "实体溯源"},
            {"path": "/ontology/domains", "methods": ["GET"], "description": "列出所有领域"},
            {"path": "/ontology/validate", "methods": ["POST"], "description": "本体验证"},
            {"path": "/ontology/{domain}", "methods": ["GET"], "description": "获取领域本体"},
            {"path": "/persistence/snapshot", "methods": ["POST"], "description": "保存快照"},
            {"path": "/persistence/restore", "methods": ["POST"], "description": "恢复快照"},
            {"path": "/stats", "methods": ["GET"], "description": "知识库统计"},
        ]


__all__ = ["L3Router"]
