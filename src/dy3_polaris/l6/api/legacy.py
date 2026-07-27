"""Legacy 遗留系统集成适配器.

将旧版 API (扁平 JSON / v1 前缀) 的请求转换为 L6 现代内部调用,
再将 L6 响应转换回 Legacy 格式。保证新旧系统平滑迁移。

设计原则:
- 适配器模式: 不修改 L6 内部代码, 仅做格式转换
- 字段映射: legacy_field → l6_field 的双向映射表
- 版本标记: 所有响应携带 "api_version": "legacy/v1"
- 空值兼容: 旧版可能缺失新字段, 适配器填充默认值
- 可挂载到 L6Router 或独立 Starlette 应用

Legacy 格式约定:
- 统一响应: {"status": "ok"|"error", "data": ..., "error_msg": "", "api_version": "legacy/v1"}
- 旧版工具调用: POST /api/v1/tool/call {"tool": "name", "args": {}}
- 旧版学情查询: GET  /api/v1/learner/{student_id}
- 旧版知识查询: GET  /api/v1/knowledge/{kp_id}
- 旧版评估结果: GET  /api/v1/assessment/{assessment_id}
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from dy3_polaris.l6.core.engine import L6CoreEngine

_logger = logging.getLogger("dy3_polaris.l6.api.legacy")


# ============================================================
# Legacy 响应格式
# ============================================================

def _legacy_ok(data: Any = None) -> dict[str, Any]:
    """构造 Legacy 成功响应."""
    return {
        "status": "ok",
        "data": data,
        "error_msg": "",
        "api_version": "legacy/v1",
    }


def _legacy_err(msg: str) -> dict[str, Any]:
    """构造 Legacy 错误响应."""
    return {
        "status": "error",
        "data": None,
        "error_msg": msg,
        "api_version": "legacy/v1",
    }


# ============================================================
# Legacy 路由处理器
# ============================================================

class _LegacyHandlers:
    """Legacy API 处理器. 将旧版请求适配到 L6 内部调用."""

    def __init__(self, engine: L6CoreEngine) -> None:
        self._engine = engine

    # ---- 旧版工具调用 ----

    async def tool_call(self, request: Request) -> JSONResponse:
        """POST {prefix}/tool/call — 旧版工具调用.

        旧版格式: {"tool": "name", "args": {...}}
        L6 格式: call_tool(name, arguments)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_legacy_err("请求体解析失败"), status_code=400)

        tool_name = body.get("tool", "")
        args = body.get("args", {})

        if not tool_name:
            return JSONResponse(_legacy_err("缺少 tool 字段"), status_code=400)

        result = self._engine.call_tool(tool_name, args)

        if "error" in result:
            err = result["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return JSONResponse(_legacy_err(msg), status_code=500)

        return JSONResponse(_legacy_ok(result))

    # ---- 旧版学情查询 ----

    async def get_learner(self, request: Request) -> JSONResponse:
        """GET {prefix}/learner/{student_id} — 旧版学情查询.

        适配到记忆图谱查询学习者节点。
        """
        student_id = request.path_params["student_id"]
        graph = self._engine.memory_graph

        if graph is None:
            return JSONResponse(_legacy_err("记忆图谱未启用"), status_code=503)

        results = graph.search(
            metadata_key="student_id",
            metadata_value=student_id,
            limit=1,
        )

        if not results:
            return JSONResponse(_legacy_err(f"学习者未找到: {student_id}"), status_code=404)

        node = results[0]
        return JSONResponse(_legacy_ok(self._node_to_legacy_learner(node)))

    # ---- 旧版知识查询 ----

    async def get_knowledge(self, request: Request) -> JSONResponse:
        """GET {prefix}/knowledge/{kp_id} — 旧版知识点查询.

        适配到记忆图谱查询知识点节点。
        """
        kp_id = request.path_params["kp_id"]
        graph = self._engine.memory_graph

        if graph is None:
            return JSONResponse(_legacy_err("记忆图谱未启用"), status_code=503)

        try:
            node = graph.get_node(kp_id)
        except Exception:
            return JSONResponse(_legacy_err(f"知识点未找到: {kp_id}"), status_code=404)

        return JSONResponse(_legacy_ok(self._node_to_legacy_knowledge(node)))

    # ---- 旧版评估结果查询 ----

    async def get_assessment(self, request: Request) -> JSONResponse:
        """GET {prefix}/assessment/{assessment_id} — 旧版评估结果查询.

        适配到记忆图谱查询评估节点。
        """
        assessment_id = request.path_params["assessment_id"]
        graph = self._engine.memory_graph

        if graph is None:
            return JSONResponse(_legacy_err("记忆图谱未启用"), status_code=503)

        try:
            node = graph.get_node(assessment_id)
        except Exception:
            return JSONResponse(_legacy_err(f"评估结果未找到: {assessment_id}"), status_code=404)

        return JSONResponse(_legacy_ok(self._node_to_legacy_assessment(node)))

    # ---- 旧版系统状态 ----

    async def system_status(self, request: Request) -> JSONResponse:
        """GET {prefix}/system/status — 旧版系统状态.

        适配到 L6 engine.get_status()，转换字段名。
        """
        status = self._engine.get_status()
        legacy_data = {
            "online": status["initialized"],
            "modules": {
                "tool_registry": status["modules"].get("tool_registry", False),
                "a2a": status["modules"].get("a2a_bus", False),
                "compute": status["modules"].get("compute_scheduler", False),
                "provenance": status["modules"].get("provenance_store", False),
            },
            "version": "legacy/v1",
        }
        return JSONResponse(_legacy_ok(legacy_data))

    # ---- 旧版工具列表 ----

    async def tool_list(self, request: Request) -> JSONResponse:
        """GET {prefix}/tools — 旧版工具列表.

        旧版格式: [{"name": ..., "description": ..., "category": ...}]
        """
        reg = self._engine.tool_registry
        if reg is None:
            return JSONResponse(_legacy_err("工具注册中心未初始化"), status_code=503)

        entries = reg.export_all_entries()
        tools = []
        for t in entries:
            tools.append({
                "name": t["name"],
                "description": t["description"],
                "category": t.get("category", ""),
            })
        return JSONResponse(_legacy_ok(tools))

    # ============================================================
    # 字段映射
    # ============================================================

    @staticmethod
    def _node_to_legacy_learner(node: Any) -> dict[str, Any]:
        """记忆图谱节点 → Legacy 学情格式."""
        d = node.to_dict()
        return {
            "student_id": d.get("node_id", ""),
            "name": d.get("content", {}).get("name", ""),
            "grade": d.get("metadata", {}).get("grade", ""),
            "major": d.get("metadata", {}).get("major", ""),
            "strength": d.get("strength", 1.0),
            "last_accessed": d.get("last_accessed_at"),
            "access_count": d.get("access_count", 0),
        }

    @staticmethod
    def _node_to_legacy_knowledge(node: Any) -> dict[str, Any]:
        """记忆图谱节点 → Legacy 知识点格式."""
        d = node.to_dict()
        return {
            "kp_id": d.get("node_id", ""),
            "title": d.get("content", {}).get("title", ""),
            "description": d.get("content", {}).get("description", ""),
            "strength": d.get("strength", 1.0),
            "access_count": d.get("access_count", 0),
        }

    @staticmethod
    def _node_to_legacy_assessment(node: Any) -> dict[str, Any]:
        """记忆图谱节点 → Legacy 评估结果格式."""
        d = node.to_dict()
        return {
            "assessment_id": d.get("node_id", ""),
            "student_id": d.get("metadata", {}).get("student_id", ""),
            "score": d.get("content", {}).get("score"),
            "max_score": d.get("content", {}).get("max_score"),
            "passed": d.get("content", {}).get("passed", False),
            "timestamp": d.get("created_at"),
        }


# ============================================================
# LegacyAdapter
# ============================================================

class LegacyAdapter:
    """Legacy 遗留系统集成适配器.

    将旧版扁平 JSON API 请求转换为 L6 内部调用,
    响应转换回 Legacy 格式。

    Usage::

        engine = L6CoreEngine()
        engine.initialize()

        adapter = LegacyAdapter(engine, prefix="/api/v1")
        app = adapter.create_app()

        # 独立运行
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8080)

        # 或挂载到 L6Router
        # l6_app = L6Router(engine).create_app()
        # l6_app.mount("/api/v1", adapter.create_app())
    """

    def __init__(
        self,
        engine: L6CoreEngine,
        prefix: str = "/api/v1",
    ) -> None:
        self._engine = engine
        self._prefix = prefix.rstrip("/")
        self._handlers = _LegacyHandlers(engine)

    def create_app(self) -> Starlette:
        """创建 Legacy Starlette 应用.

        Returns:
            配置好的 Starlette 应用
        """
        from starlette.applications import Starlette

        h = self._handlers
        p = self._prefix

        routes = [
            Route(f"{p}/tool/call", h.tool_call, methods=["POST"]),
            Route(f"{p}/tools", h.tool_list, methods=["GET"]),
            Route(f"{p}/learner/{{student_id}}", h.get_learner, methods=["GET"]),
            Route(f"{p}/knowledge/{{kp_id}}", h.get_knowledge, methods=["GET"]),
            Route(f"{p}/assessment/{{assessment_id}}", h.get_assessment, methods=["GET"]),
            Route(f"{p}/system/status", h.system_status, methods=["GET"]),
        ]

        return Starlette(routes=routes)

    def get_routes_summary(self) -> list[dict[str, str]]:
        """获取所有 Legacy 路由摘要."""
        p = self._prefix
        return [
            {"path": f"{p}/tool/call", "methods": ["POST"], "description": "旧版工具调用"},
            {"path": f"{p}/tools", "methods": ["GET"], "description": "旧版工具列表"},
            {"path": f"{p}/learner/{{student_id}}", "methods": ["GET"], "description": "旧版学情查询"},
            {"path": f"{p}/knowledge/{{kp_id}}", "methods": ["GET"], "description": "旧版知识点查询"},
            {"path": f"{p}/assessment/{{assessment_id}}", "methods": ["GET"], "description": "旧版评估结果查询"},
            {"path": f"{p}/system/status", "methods": ["GET"], "description": "旧版系统状态"},
        ]

    def mount_to(self, parent_app: Starlette) -> None:
        """将 Legacy 路由挂载到父 Starlette 应用.

        Args:
            parent_app: 父 Starlette 应用 (通常是 L6Router.create_app() 的返回值)
        """
        from starlette.routing import Mount

        legacy_app = self.create_app()
        # 将子应用的路由挂载到父应用
        parent_app.routes.append(
            Mount(self._prefix, app=legacy_app)
        )
