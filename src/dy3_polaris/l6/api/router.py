"""L6 REST API 路由层.

基于 Starlette 构建, 将所有 L6 子系统暴露为 RESTful JSON API。
同时支持 JSON-RPC 2.0 批量调用接口。

设计原则:
- 资源导向 URL 设计 (符合 RESTful 语义)
- 统一响应格式: {"code": 0, "data": ..., "message": ""}
- 异常统一处理, L6Error 自动映射为 HTTP 响应
- JSON-RPC 2.0 兼容端点, 支持批量调用
- CORS 中间件支持
- 健康检查就绪探针

端点概览:
    GET  /health                    — 健康检查 + 各模块状态
    POST /jsonrpc                   — JSON-RPC 2.0 批量调用入口

    # 工具注册中心
    GET    /tools                     — 列出所有工具
    GET    /tools/{name}              — 查询单个工具
    POST   /tools/{name}/call         — 调用工具
    GET    /tools/stats               — 调用统计

    # 算力资源
    GET    /compute/resources         — 列出所有算力资源
    POST   /compute/resources         — 注册算力资源
    GET    /compute/resources/{id}    — 查询单个资源
    DELETE /compute/resources/{id}    — 注销资源
    POST   /compute/allocate          — 分配算力
    POST   /compute/tasks/{id}/start  — 启动任务
    POST   /compute/tasks/{id}/complete — 完成任务
    POST   /compute/tasks/{id}/fail   — 标记任务失败
    GET    /compute/metrics           — 算力度量

    # A2A 协议
    GET    /a2a/agents                 — 列出所有 Agent
    POST   /a2a/agents                 — 注册 Agent
    GET    /a2a/agents/{id}            — 查询 Agent
    DELETE /a2a/agents/{id}            — 注销 Agent

    # 溯源
    GET    /provenance/chains          — 列出所有链
    GET    /provenance/chains/{id}     — 查询单条链
    POST   /provenance/chains          — 创建链
    GET    /provenance/query           — 查询 KPA

    # 广播
    GET    /broadcast/topics           — 列出所有主题
    POST   /broadcast/publish          — 发布事件
    GET    /broadcast/subscribers      — 查询订阅者
    GET    /broadcast/metrics          — 广播度量
    GET    /broadcast/events           — 事件日志

    # 记忆图谱
    POST   /memory/nodes               — 添加节点
    GET    /memory/nodes/{id}          — 查询节点
    DELETE /memory/nodes/{id}          — 删除节点
    POST   /memory/edges               — 添加边
    DELETE /memory/edges               — 删除边
    GET    /memory/search              — 搜索节点
    GET    /memory/metrics             — 图谱度量
    GET    /memory/export              — 导出图谱
    POST   /memory/decay               — 触发衰减
    POST   /memory/spread              — 扩散激活
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from dy3_polaris.l6.core.config import L6Config
from dy3_polaris.l6.core.engine import L6CoreEngine
from dy3_polaris.l6.core.exceptions import L6Error

_logger = logging.getLogger("dy3_polaris.l6.api.router")


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


def _l6_error_to_dict(err: L6Error) -> dict[str, Any]:
    """将 L6Error 转为响应字典."""
    return _err(-32000, err.code, err.detail)


# ============================================================
# 路由处理器
# ============================================================

class _RouteHandlers:
    """将 L6CoreEngine 方法适配为 Starlette Request→Response 处理器.

    每个处理器方法:
    1. 从 engine 获取所需子系统
    2. 调用子系统方法
    3. 将异常转为统一错误响应
    4. 返回 JSONResponse
    """

    def __init__(self, engine: L6CoreEngine) -> None:
        self._engine = engine

    # ---- 健康检查 ----

    async def health(self, request: Request) -> JSONResponse:
        """GET /health — 健康检查 + 各模块状态."""
        status = self._engine.get_status()
        return JSONResponse(_ok(status))

    # ---- JSON-RPC 2.0 ----

    async def jsonrpc(self, request: Request) -> JSONResponse:
        """POST /jsonrpc — JSON-RPC 2.0 调用入口.

        支持单请求和批量请求 (数组)。
        仅代理 call_tool 方法到 engine.call_tool。
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "Parse error"), status_code=200)

        is_batch = isinstance(body, list)
        requests_list = body if is_batch else [body]
        responses = []

        for req in requests_list:
            resp = self._handle_jsonrpc_single(req)
            if resp is not None:
                responses.append(resp)

        if is_batch:
            return JSONResponse(responses if responses else [])
        return JSONResponse(responses[0] if responses else {"jsonrpc": "2.0", "result": None, "id": None})

    def _handle_jsonrpc_single(self, req: dict[str, Any]) -> dict[str, Any] | None:
        """处理单个 JSON-RPC 请求."""
        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        # 通知 (无 id) 不返回
        if req_id is None:
            try:
                self._dispatch_jsonrpc(method, params)
            except Exception:
                pass
            return None

        try:
            result = self._dispatch_jsonrpc(method, params)
            return {"jsonrpc": "2.0", "result": result, "id": req_id}
        except L6Error as e:
            return {"jsonrpc": "2.0", "error": e.to_json_rpc_error(), "id": req_id}
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": req_id,
            }

    def _dispatch_jsonrpc(self, method: str, params: dict[str, Any]) -> Any:
        """分发 JSON-RPC 方法."""
        if method == "call_tool":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            return self._engine.call_tool(tool_name, arguments)
        if method == "get_status":
            return self._engine.get_status()
        raise Exception(f"Method not found: {method}")

    # ---- 工具注册中心 ----

    async def list_tools(self, request: Request) -> JSONResponse:
        """GET /tools — 列出所有工具."""
        reg = self._engine.tool_registry
        if reg is None:
            return JSONResponse(_err(-32000, "工具注册中心未初始化"), status_code=503)
        tools = reg.export_all_entries()
        return JSONResponse(_ok(tools))

    async def get_tool(self, request: Request) -> JSONResponse:
        """GET /tools/{name} — 查询单个工具."""
        reg = self._engine.tool_registry
        if reg is None:
            return JSONResponse(_err(-32000, "工具注册中心未初始化"), status_code=503)
        name = request.path_params["name"]
        entry = reg.get(name)
        if entry is None:
            return JSONResponse(_err(-32601, f"工具未找到: {name}"), status_code=404)
        return JSONResponse(_ok(entry.to_dict()))

    async def call_tool_rest(self, request: Request) -> JSONResponse:
        """POST /tools/{name}/call — 调用工具."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)
        name = request.path_params["name"]
        result = self._engine.call_tool(name, body.get("arguments", {}))
        if "error" in result:
            return JSONResponse(_err(-32000, "工具调用失败", result["error"]), status_code=500)
        return JSONResponse(_ok(result))

    async def tool_stats(self, request: Request) -> JSONResponse:
        """GET /tools/stats — 调用统计."""
        reg = self._engine.tool_registry
        if reg is None:
            return JSONResponse(_err(-32000, "工具注册中心未初始化"), status_code=503)
        return JSONResponse(_ok(reg.export_registry_summary()))

    # ---- 算力资源 ----

    async def list_compute_resources(self, request: Request) -> JSONResponse:
        """GET /compute/resources — 列出所有算力资源."""
        sched = self._engine.compute_scheduler
        if sched is None:
            return JSONResponse(_err(-32000, "算力调度器未初始化"), status_code=503)
        resources = []
        for rid in sched._resources:
            r = sched._resources[rid]
            resources.append(r.model_dump(mode="json"))
        return JSONResponse(_ok(resources))

    async def register_compute_resource(self, request: Request) -> JSONResponse:
        """POST /compute/resources — 注册算力资源."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)
        sched = self._engine.compute_scheduler
        if sched is None:
            return JSONResponse(_err(-32000, "算力调度器未初始化"), status_code=503)
        try:
            from dy3_polaris.l6.core.models import ComputeResourceDescriptor
            desc = ComputeResourceDescriptor(**body)
            rid = sched.register(desc)
            return JSONResponse(_ok({"resource_id": rid}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def get_compute_resource(self, request: Request) -> JSONResponse:
        """GET /compute/resources/{id} — 查询单个资源."""
        sched = self._engine.compute_scheduler
        if sched is None:
            return JSONResponse(_err(-32000, "算力调度器未初始化"), status_code=503)
        rid = request.path_params["id"]
        r = sched.get_resource(rid)
        if r is None:
            return JSONResponse(_err(-32000, f"资源未找到: {rid}"), status_code=404)
        return JSONResponse(_ok(r.model_dump(mode="json")))

    async def delete_compute_resource(self, request: Request) -> JSONResponse:
        """DELETE /compute/resources/{id} — 注销资源."""
        sched = self._engine.compute_scheduler
        if sched is None:
            return JSONResponse(_err(-32000, "算力调度器未初始化"), status_code=503)
        rid = request.path_params["id"]
        ok = sched.unregister(rid)
        if not ok:
            return JSONResponse(_err(-32000, f"资源注销失败: {rid}"), status_code=404)
        return JSONResponse(_ok({"unregistered": rid}))

    async def compute_metrics(self, request: Request) -> JSONResponse:
        """GET /compute/metrics — 算力度量."""
        sched = self._engine.compute_scheduler
        if sched is None:
            return JSONResponse(_err(-32000, "算力调度器未初始化"), status_code=503)
        return JSONResponse(_ok(sched.export_summary()))

    # ---- A2A 协议 ----

    async def list_agents(self, request: Request) -> JSONResponse:
        """GET /a2a/agents — 列出所有 Agent."""
        bus = self._engine.a2a_bus
        if bus is None:
            return JSONResponse(_err(-32000, "A2A 总线未初始化"), status_code=503)
        agent_list = []
        for aid, cap in bus._agents.items():
            d = cap.model_dump(mode="json")
            d["agent_id"] = aid
            agent_list.append(d)
        return JSONResponse(_ok(agent_list))

    async def register_agent(self, request: Request) -> JSONResponse:
        """POST /a2a/agents — 注册 Agent."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)
        bus = self._engine.a2a_bus
        if bus is None:
            return JSONResponse(_err(-32000, "A2A 总线未初始化"), status_code=503)
        try:
            from dy3_polaris.l6.core.models import A2ACapability
            cap = A2ACapability(**body)
            aid = cap.agent_id
            bus.register_agent(aid, cap)
            return JSONResponse(_ok({"agent_id": aid}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def get_agent(self, request: Request) -> JSONResponse:
        """GET /a2a/agents/{id} — 查询 Agent."""
        bus = self._engine.a2a_bus
        if bus is None:
            return JSONResponse(_err(-32000, "A2A 总线未初始化"), status_code=503)
        aid = request.path_params["id"]
        agent = bus.get_agent(aid)
        if agent is None:
            return JSONResponse(_err(-32000, f"Agent 未找到: {aid}"), status_code=404)
        return JSONResponse(_ok(agent.model_dump(mode="json")))

    async def delete_agent(self, request: Request) -> JSONResponse:
        """DELETE /a2a/agents/{id} — 注销 Agent."""
        bus = self._engine.a2a_bus
        if bus is None:
            return JSONResponse(_err(-32000, "A2A 总线未初始化"), status_code=503)
        aid = request.path_params["id"]
        cap = bus.unregister_agent(aid)
        if cap is None:
            return JSONResponse(_err(-32000, f"Agent 未找到: {aid}"), status_code=404)
        return JSONResponse(_ok({"unregistered": aid}))

    # ---- 溯源 ----

    async def list_chains(self, request: Request) -> JSONResponse:
        """GET /provenance/chains — 列出所有溯源链."""
        store = self._engine.provenance_store
        if store is None:
            return JSONResponse(_err(-32000, "溯源存储未初始化"), status_code=503)
        return JSONResponse(_ok({
            "chain_ids": store.all_chain_ids(),
            "chain_count": store.chain_count,
            "total_kpa_count": store.total_kpa_count,
        }))

    async def get_chain(self, request: Request) -> JSONResponse:
        """GET /provenance/chains/{id} — 查询单条链."""
        store = self._engine.provenance_store
        if store is None:
            return JSONResponse(_err(-32000, "溯源存储未初始化"), status_code=503)
        cid = request.path_params["id"]
        chain = store.get_chain(cid)
        if chain is None:
            return JSONResponse(_err(-32000, f"链未找到: {cid}"), status_code=404)
        return JSONResponse(_ok(chain.to_dict()))

    async def create_chain(self, request: Request) -> JSONResponse:
        """POST /provenance/chains — 创建溯源链."""
        store = self._engine.provenance_store
        if store is None:
            return JSONResponse(_err(-32000, "溯源存储未初始化"), status_code=503)
        chain = store.create_chain()
        return JSONResponse(_ok({"chain_id": chain.chain_id}))

    # ---- 广播 ----

    async def list_topics(self, request: Request) -> JSONResponse:
        """GET /broadcast/topics — 列出所有主题."""
        bus = self._engine.broadcast_bus
        if bus is None:
            return JSONResponse(_err(-32000, "广播总线未初始化"), status_code=503)
        return JSONResponse(_ok(bus.list_topics()))

    async def publish_event(self, request: Request) -> JSONResponse:
        """POST /broadcast/publish — 发布事件.

        注意: REST 端点发布事件不携带回调订阅者,
        仅记录事件日志和度量。"""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)
        bus = self._engine.broadcast_bus
        if bus is None:
            return JSONResponse(_err(-32000, "广播总线未初始化"), status_code=503)
        try:
            event = bus.publish(
                topic=body.get("topic", ""),
                payload=body.get("payload", {}),
                source=body.get("source", "rest-api"),
                metadata=body.get("metadata", {}),
            )
            return JSONResponse(_ok(event.to_dict()))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def broadcast_metrics(self, request: Request) -> JSONResponse:
        """GET /broadcast/metrics — 广播度量."""
        bus = self._engine.broadcast_bus
        if bus is None:
            return JSONResponse(_err(-32000, "广播总线未初始化"), status_code=503)
        return JSONResponse(_ok(bus.get_metrics()))

    async def event_log(self, request: Request) -> JSONResponse:
        """GET /broadcast/events — 事件日志."""
        bus = self._engine.broadcast_bus
        if bus is None:
            return JSONResponse(_err(-32000, "广播总线未初始化"), status_code=503)
        events = [e.to_dict() for e in bus.get_event_log()]
        return JSONResponse(_ok(events))

    # ---- 记忆图谱 ----

    async def add_memory_node(self, request: Request) -> JSONResponse:
        """POST /memory/nodes — 添加节点."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)
        graph = self._engine.memory_graph
        if graph is None:
            return JSONResponse(_err(-32000, "记忆图谱未初始化"), status_code=503)
        try:
            from dy3_polaris.l6.broadcast.memory_graph import NodeType
            node = graph.add_node(
                node_id=body.get("node_id"),
                node_type=NodeType(body.get("node_type", "knowledge")),
                content=body.get("content", {}),
                metadata=body.get("metadata", {}),
                strength=body.get("strength", 1.0),
            )
            return JSONResponse(_ok(node.to_dict()))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def get_memory_node(self, request: Request) -> JSONResponse:
        """GET /memory/nodes/{id} — 查询节点."""
        graph = self._engine.memory_graph
        if graph is None:
            return JSONResponse(_err(-32000, "记忆图谱未初始化"), status_code=503)
        nid = request.path_params["id"]
        try:
            node = graph.get_node(nid)
            return JSONResponse(_ok(node.to_dict()))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=404)

    async def delete_memory_node(self, request: Request) -> JSONResponse:
        """DELETE /memory/nodes/{id} — 删除节点."""
        graph = self._engine.memory_graph
        if graph is None:
            return JSONResponse(_err(-32000, "记忆图谱未初始化"), status_code=503)
        nid = request.path_params["id"]
        try:
            graph.remove_node(nid)
            return JSONResponse(_ok({"removed": nid}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=404)

    async def add_memory_edge(self, request: Request) -> JSONResponse:
        """POST /memory/edges — 添加边."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)
        graph = self._engine.memory_graph
        if graph is None:
            return JSONResponse(_err(-32000, "记忆图谱未初始化"), status_code=503)
        try:
            from dy3_polaris.l6.broadcast.memory_graph import EdgeType
            edge = graph.add_edge(
                source_id=body["source_id"],
                target_id=body["target_id"],
                edge_type=EdgeType(body.get("edge_type", "related")),
                weight=body.get("weight", 1.0),
            )
            return JSONResponse(_ok(edge.to_dict()))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def delete_memory_edge(self, request: Request) -> JSONResponse:
        """DELETE /memory/edges — 删除边."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)
        graph = self._engine.memory_graph
        if graph is None:
            return JSONResponse(_err(-32000, "记忆图谱未初始化"), status_code=503)
        try:
            graph.remove_edge(body["source_id"], body["target_id"])
            return JSONResponse(_ok({"removed": {"source_id": body["source_id"], "target_id": body["target_id"]}}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=404)

    async def search_memory(self, request: Request) -> JSONResponse:
        """GET /memory/search — 搜索节点."""
        graph = self._engine.memory_graph
        if graph is None:
            return JSONResponse(_err(-32000, "记忆图谱未初始化"), status_code=503)
        from dy3_polaris.l6.broadcast.memory_graph import NodeType
        params = request.query_params
        node_type = None
        if params.get("node_type"):
            try:
                node_type = NodeType(params["node_type"])
            except ValueError:
                pass
        results = graph.search(
            node_type=node_type,
            min_strength=float(params["min_strength"]) if "min_strength" in params else None,
            metadata_key=params.get("metadata_key"),
            metadata_value=params.get("metadata_value"),
            limit=int(params.get("limit", 100)),
        )
        return JSONResponse(_ok([n.to_dict() for n in results]))

    async def memory_metrics(self, request: Request) -> JSONResponse:
        """GET /memory/metrics — 图谱度量."""
        graph = self._engine.memory_graph
        if graph is None:
            return JSONResponse(_err(-32000, "记忆图谱未初始化"), status_code=503)
        return JSONResponse(_ok(graph.get_metrics()))

    async def export_memory(self, request: Request) -> JSONResponse:
        """GET /memory/export — 导出图谱."""
        graph = self._engine.memory_graph
        if graph is None:
            return JSONResponse(_err(-32000, "记忆图谱未初始化"), status_code=503)
        return JSONResponse(_ok(graph.export()))

    async def decay_memory(self, request: Request) -> JSONResponse:
        """POST /memory/decay — 触发衰减."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        graph = self._engine.memory_graph
        if graph is None:
            return JSONResponse(_err(-32000, "记忆图谱未初始化"), status_code=503)
        factor = body.get("factor")
        pruned = graph.decay(factor=factor if factor is not None else None)
        return JSONResponse(_ok({"pruned": pruned}))

    async def spread_activation(self, request: Request) -> JSONResponse:
        """POST /memory/spread — 扩散激活."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)
        graph = self._engine.memory_graph
        if graph is None:
            return JSONResponse(_err(-32000, "记忆图谱未初始化"), status_code=503)
        try:
            node_id = body.get("node_id", "")
            depth = body.get("depth")
            decay = body.get("decay")
            activations = graph.spreading_activation(
                node_id=node_id,
                depth=depth,
                decay=decay,
            )
            return JSONResponse(_ok(activations))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)


# ============================================================
# L6Router
# ============================================================

class L6Router:
    """L6 REST API 路由器.

    将 L6CoreEngine 的所有子系统暴露为 RESTful API。
    基于 Starlette 构建, 支持挂载到 uvicorn 或嵌入到 FastAPI 应用。

    Usage::

        engine = L6CoreEngine()
        engine.initialize()

        router = L6Router(engine, config)
        app = router.create_app()

        # 独立运行
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

        # 或嵌入到 FastAPI
        # from fastapi import FastAPI
        # fa = FastAPI()
        # fa.mount("/l6", app)
    """

    def __init__(self, engine: L6CoreEngine, config: L6Config | None = None) -> None:
        self._engine = engine
        self._config = config or engine.config
        self._handlers = _RouteHandlers(engine)

    def create_app(self) -> Starlette:
        """创建 Starlette 应用实例.

        Returns:
            配置好的 Starlette 应用, 可直接传给 uvicorn.run()
        """
        h = self._handlers
        rest_cfg = self._config.get_rest_config()

        routes = [
            # 健康检查 & JSON-RPC
            Route("/health", h.health, methods=["GET"]),
            Route("/jsonrpc", h.jsonrpc, methods=["POST"]),

            # 工具 (静态路由必须在动态 {name} 之前)
            Route("/tools", h.list_tools, methods=["GET"]),
            Route("/tools/stats", h.tool_stats, methods=["GET"]),
            Route("/tools/{name}", h.get_tool, methods=["GET"]),
            Route("/tools/{name}/call", h.call_tool_rest, methods=["POST"]),

            # 算力
            Route("/compute/resources", h.list_compute_resources, methods=["GET"]),
            Route("/compute/resources", h.register_compute_resource, methods=["POST"]),
            Route("/compute/resources/{id}", h.get_compute_resource, methods=["GET"]),
            Route("/compute/resources/{id}", h.delete_compute_resource, methods=["DELETE"]),
            Route("/compute/metrics", h.compute_metrics, methods=["GET"]),

            # A2A
            Route("/a2a/agents", h.list_agents, methods=["GET"]),
            Route("/a2a/agents", h.register_agent, methods=["POST"]),
            Route("/a2a/agents/{id}", h.get_agent, methods=["GET"]),
            Route("/a2a/agents/{id}", h.delete_agent, methods=["DELETE"]),

            # 溯源
            Route("/provenance/chains", h.list_chains, methods=["GET"]),
            Route("/provenance/chains", h.create_chain, methods=["POST"]),
            Route("/provenance/chains/{id}", h.get_chain, methods=["GET"]),

            # 广播
            Route("/broadcast/topics", h.list_topics, methods=["GET"]),
            Route("/broadcast/publish", h.publish_event, methods=["POST"]),
            Route("/broadcast/metrics", h.broadcast_metrics, methods=["GET"]),
            Route("/broadcast/events", h.event_log, methods=["GET"]),

            # 记忆图谱
            Route("/memory/nodes", h.add_memory_node, methods=["POST"]),
            Route("/memory/nodes/{id}", h.get_memory_node, methods=["GET"]),
            Route("/memory/nodes/{id}", h.delete_memory_node, methods=["DELETE"]),
            Route("/memory/edges", h.add_memory_edge, methods=["POST"]),
            Route("/memory/edges", h.delete_memory_edge, methods=["DELETE"]),
            Route("/memory/search", h.search_memory, methods=["GET"]),
            Route("/memory/metrics", h.memory_metrics, methods=["GET"]),
            Route("/memory/export", h.export_memory, methods=["GET"]),
            Route("/memory/decay", h.decay_memory, methods=["POST"]),
            Route("/memory/spread", h.spread_activation, methods=["POST"]),
        ]

        middleware = []
        cors_origins = rest_cfg.get("cors_origins", ["*"])
        if cors_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=cors_origins,
                    allow_methods=["*"] if "*" in cors_origins else ["GET", "POST", "DELETE"],
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
            {"path": "/health", "methods": ["GET"], "description": "健康检查 + 各模块状态"},
            {"path": "/jsonrpc", "methods": ["POST"], "description": "JSON-RPC 2.0 调用入口"},
            {"path": "/tools", "methods": ["GET"], "description": "列出所有工具"},
            {"path": "/tools/{name}", "methods": ["GET"], "description": "查询单个工具"},
            {"path": "/tools/{name}/call", "methods": ["POST"], "description": "调用工具"},
            {"path": "/tools/stats", "methods": ["GET"], "description": "调用统计"},
            {"path": "/compute/resources", "methods": ["GET", "POST"], "description": "列出/注册算力资源"},
            {"path": "/compute/resources/{id}", "methods": ["GET", "DELETE"], "description": "查询/注销算力资源"},
            {"path": "/compute/metrics", "methods": ["GET"], "description": "算力度量"},
            {"path": "/a2a/agents", "methods": ["GET", "POST"], "description": "列出/注册 Agent"},
            {"path": "/a2a/agents/{id}", "methods": ["GET", "DELETE"], "description": "查询/注销 Agent"},
            {"path": "/provenance/chains", "methods": ["GET", "POST"], "description": "列出/创建溯源链"},
            {"path": "/provenance/chains/{id}", "methods": ["GET"], "description": "查询溯源链"},
            {"path": "/broadcast/topics", "methods": ["GET"], "description": "列出广播主题"},
            {"path": "/broadcast/publish", "methods": ["POST"], "description": "发布广播事件"},
            {"path": "/broadcast/metrics", "methods": ["GET"], "description": "广播度量"},
            {"path": "/broadcast/events", "methods": ["GET"], "description": "广播事件日志"},
            {"path": "/memory/nodes", "methods": ["POST"], "description": "添加记忆节点"},
            {"path": "/memory/nodes/{id}", "methods": ["GET", "DELETE"], "description": "查询/删除记忆节点"},
            {"path": "/memory/edges", "methods": ["POST", "DELETE"], "description": "添加/删除记忆边"},
            {"path": "/memory/search", "methods": ["GET"], "description": "搜索记忆节点"},
            {"path": "/memory/metrics", "methods": ["GET"], "description": "图谱度量"},
            {"path": "/memory/export", "methods": ["GET"], "description": "导出图谱"},
            {"path": "/memory/decay", "methods": ["POST"], "description": "触发衰减"},
            {"path": "/memory/spread", "methods": ["POST"], "description": "扩散激活"},
        ]
