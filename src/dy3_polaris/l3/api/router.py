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

def _ok(data: Any = None, message: str = "") -> dict[str, Any]:
    """构造成功响应."""
    return {"code": 0, "data": data, "message": message}


def _err(code: int, message: str, detail: str = "") -> dict[str, Any]:
    """构造错误响应."""
    resp: dict[str, Any] = {"code": code, "message": message}
    if detail:
        resp["detail"] = detail
    return resp


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
                entities = self._store.entity_store.list_entities(
                    limit=limit + 1,
                    offset=offset,
                )

            # 若按类型/领域过滤了, 手动分页
            if et or domain:
                has_more = len(entities) > offset + limit
                items = entities[offset : offset + limit]
            else:
                has_more = len(entities) > limit
                items = entities[:limit]

            return JSONResponse(_ok({
                "items": [_safe_model_dump(e) for e in items],
                "total": self._store.entity_count(),
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
        routed = self._intent_router.route(query, top_k=top_k)

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
        )

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
            Route("/quality/provenance/{id}", h.quality_get_provenance, methods=["GET"]),
            Route("/quality/audit-log", h.quality_audit_log, methods=["GET"]),

            # 图推理
            Route("/graph/reason", h.graph_reason, methods=["POST"]),
            Route("/graph/stats", h.graph_stats, methods=["GET"]),

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
            {"path": "/ontology/domains", "methods": ["GET"], "description": "列出所有领域"},
            {"path": "/ontology/validate", "methods": ["POST"], "description": "本体验证"},
            {"path": "/ontology/{domain}", "methods": ["GET"], "description": "获取领域本体"},
            {"path": "/persistence/snapshot", "methods": ["POST"], "description": "保存快照"},
            {"path": "/persistence/restore", "methods": ["POST"], "description": "恢复快照"},
            {"path": "/stats", "methods": ["GET"], "description": "知识库统计"},
        ]


__all__ = ["L3Router"]
