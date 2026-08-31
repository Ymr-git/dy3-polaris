"""L7 体验呈现层 — OpenAPI 3.0.3 Schema 自动生成.

从 L7Router 的路由摘要自动生成 OpenAPI 3.0.3 规范, 并提供
Swagger UI 交互式文档。

融合世界先进方案:
- FastAPI auto-docs: 从路由定义自动生成 OpenAPI schema
- Swagger UI: 交互式 API 文档浏览器
- OpenAPI 3.0.3: 标准 API 描述规范

核心功能:
1. ``generate_openapi_spec(router)`` — 从 L7Router 生成完整 OpenAPI 3.0.3 spec
2. ``openapi_handler(router)`` — 返回 Starlette handler, 在 GET /openapi.json 提供 spec
3. ``swagger_ui_html(router)`` — 返回 Swagger UI HTML 字符串
4. ``swagger_ui_handler(router)`` — 返回 Starlette handler, 在 GET /docs 提供 Swagger UI

使用示例::

    from dy3_polaris.l7.api.router import L7Router
    from dy3_polaris.l7.api.openapi import generate_openapi_spec

    router = L7Router(...)
    spec = generate_openapi_spec(router)
    # spec 是一个完整的 OpenAPI 3.0.3 字典

    app = router.create_app()
    # app 已内置 GET /openapi.json 和 GET /docs 端点
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

if TYPE_CHECKING:
    from .router import L7Router


# ============================================================
# 常量
# ============================================================

OPENAPI_VERSION = "3.0.3"
API_TITLE = "DY3+ Polaris L7 体验呈现层 API"
API_DESCRIPTION = (
    "L7 体验呈现层 RESTful API — 提供渲染流水线、Artifact 管理、"
    "渲染器注册中心等核心功能的 HTTP 接口。"
    "统一响应格式: {code, data, message}。"
)
API_VERSION = "1.0.0"

#: Swagger UI CDN 基础 URL
_SWAGGER_UI_CDN = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5"


# ============================================================
# Schema 定义 (components/schemas)
# ============================================================


def _artifact_schema() -> dict[str, Any]:
    """Artifact 模型的 OpenAPI schema."""
    return {
        "type": "object",
        "description": "L5 智能体产出的可渲染制品",
        "properties": {
            "artifact_id": {"type": "string", "description": "Artifact 唯一标识"},
            "type": {
                "type": "string",
                "enum": [
                    "text", "chart", "graph", "molecule",
                    "table", "formula", "provenance", "interactive",
                ],
                "description": "Artifact 类型",
            },
            "mime": {"type": "string", "description": "MIME 类型，用于路由到渲染器"},
            "source_agent": {"type": "string", "description": "产出该 Artifact 的 Agent ID"},
            "provenance_chain": {
                "type": "array",
                "items": {"type": "string"},
                "description": "溯源链 (KPA ID 列表)",
            },
            "learner_context": {
                "type": "object",
                "description": "学习者上下文",
            },
            "version": {"type": "integer", "minimum": 1, "description": "版本号"},
            "editable": {"type": "boolean", "description": "是否可编辑"},
            "fork_origin": {
                "type": "string",
                "nullable": True,
                "description": "分叉来源 Artifact ID",
            },
            "payload": {"type": "object", "description": "渲染载荷数据"},
            "session_id": {"type": "string", "description": "会话 ID"},
            "title": {"type": "string", "description": "标题"},
            "created_at": {"type": "number", "format": "float", "description": "创建时间戳"},
            "updated_at": {"type": "number", "format": "float", "description": "更新时间戳"},
            "state": {
                "type": "string",
                "enum": ["created", "rendered", "reviewed", "edited", "archived"],
                "description": "生命周期状态",
            },
        },
    }


def _render_context_schema() -> dict[str, Any]:
    """RenderContext 模型的 OpenAPI schema."""
    return {
        "type": "object",
        "description": "渲染上下文 — 渲染时的环境与学习者状态",
        "properties": {
            "viewport": {
                "type": "object",
                "properties": {
                    "width": {"type": "integer", "minimum": 1, "default": 1280},
                    "height": {"type": "integer", "minimum": 1, "default": 720},
                },
                "description": "视口尺寸",
            },
            "theme": {
                "type": "string",
                "enum": ["light", "dark", "auto"],
                "default": "light",
                "description": "主题",
            },
            "learner_mode": {
                "type": "string",
                "enum": ["beginner", "intermediate", "advanced"],
                "default": "intermediate",
                "description": "学习者模式",
            },
            "bkt_state": {"type": "object", "description": "BKT 知识掌握状态"},
            "kp_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "关联知识点 ID 列表",
            },
            "locale": {"type": "string", "default": "zh-CN", "description": "语言区域"},
        },
    }


def _render_descriptor_schema() -> dict[str, Any]:
    """RenderDescriptor 模型的 OpenAPI schema."""
    return {
        "type": "object",
        "description": "渲染描述符 — 渲染器产出，前端可直接消费",
        "properties": {
            "render_id": {"type": "string", "description": "渲染实例唯一标识"},
            "artifact_id": {"type": "string", "description": "关联的 Artifact ID"},
            "mime": {"type": "string", "description": "渲染输出的 MIME 类型"},
            "html": {
                "type": "string",
                "nullable": True,
                "description": "可嵌入的 HTML 片段",
            },
            "config": {"type": "object", "description": "前端渲染配置"},
            "assets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "依赖的静态资源 URL 列表",
            },
            "metadata": {"type": "object", "description": "渲染元数据"},
            "rendered_at": {"type": "number", "format": "float", "description": "渲染完成时间戳"},
            "render_time_ms": {
                "type": "number",
                "format": "float",
                "minimum": 0,
                "description": "渲染耗时 (毫秒)",
            },
        },
    }


def _artifact_diff_schema() -> dict[str, Any]:
    """ArtifactDiff 模型的 OpenAPI schema (RFC 6902 JSON Patch)."""
    return {
        "type": "object",
        "description": "Artifact 增量差异 — 用于渲染器的增量更新 (RFC 6902 JSON Patch)",
        "properties": {
            "artifact_id": {"type": "string", "description": "关联的 Artifact ID"},
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["add", "replace", "remove", "move", "copy", "test"],
                            "description": "操作类型 (RFC 6902)",
                        },
                        "path": {
                            "type": "string",
                            "description": "JSON Pointer 路径",
                        },
                        "value": {
                            "description": "操作值 (add/replace/test 需要)",
                        },
                    },
                    "required": ["op", "path"],
                },
                "description": "差异操作列表 (JSON Patch 风格)",
            },
            "edit_reason": {"type": "string", "description": "编辑原因"},
            "created_at": {"type": "number", "format": "float", "description": "创建时间戳"},
        },
        "required": ["artifact_id"],
    }


def _error_response_schema() -> dict[str, Any]:
    """错误响应的 OpenAPI schema."""
    return {
        "type": "object",
        "description": "统一错误响应",
        "properties": {
            "code": {"type": "integer", "description": "错误码 (非 0)"},
            "message": {"type": "string", "description": "错误消息"},
            "detail": {"type": "string", "description": "错误详情 (可选)"},
        },
        "required": ["code", "message"],
    }


def _success_response_schema() -> dict[str, Any]:
    """成功响应的 OpenAPI schema."""
    return {
        "type": "object",
        "description": "统一成功响应",
        "properties": {
            "code": {"type": "integer", "description": "状态码 (0 表示成功)", "example": 0},
            "data": {"description": "响应数据"},
            "message": {"type": "string", "description": "消息", "example": ""},
        },
        "required": ["code", "data", "message"],
    }


def _build_schemas() -> dict[str, Any]:
    """构建 components/schemas 字典."""
    return {
        "Artifact": _artifact_schema(),
        "RenderContext": _render_context_schema(),
        "RenderDescriptor": _render_descriptor_schema(),
        "ArtifactDiff": _artifact_diff_schema(),
        "ErrorResponse": _error_response_schema(),
        "SuccessResponse": _success_response_schema(),
    }


# ============================================================
# 参数定义
# ============================================================

#: GET /artifacts 的查询参数
_QUERY_PARAMS_ARTIFACTS: list[dict[str, Any]] = [
    {
        "name": "type",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
        "description": "Artifact 类型过滤 (text/chart/graph/molecule/table/formula/provenance/interactive)",
    },
    {
        "name": "source_agent",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
        "description": "来源 Agent ID 过滤",
    },
    {
        "name": "session_id",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
        "description": "会话 ID 过滤",
    },
    {
        "name": "kp_id",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
        "description": "知识点 ID 过滤",
    },
    {
        "name": "page",
        "in": "query",
        "required": False,
        "schema": {"type": "integer", "default": 1, "minimum": 1},
        "description": "页码 (默认 1)",
    },
    {
        "name": "size",
        "in": "query",
        "required": False,
        "schema": {"type": "integer", "default": 20, "minimum": 1},
        "description": "每页数量 (默认 20)",
    },
    {
        "name": "sort",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
        "description": "排序字段 (前缀 - 表示降序, 如 -created_at)",
    },
]

#: POST /render 的查询参数 (渲染上下文)
_QUERY_PARAMS_RENDER_CONTEXT: list[dict[str, Any]] = [
    {
        "name": "theme",
        "in": "query",
        "required": False,
        "schema": {"type": "string", "enum": ["light", "dark", "auto"], "default": "light"},
        "description": "主题 (light/dark/auto)",
    },
    {
        "name": "learner_mode",
        "in": "query",
        "required": False,
        "schema": {
            "type": "string",
            "enum": ["beginner", "intermediate", "advanced"],
            "default": "intermediate",
        },
        "description": "学习者模式 (beginner/intermediate/advanced)",
    },
    {
        "name": "viewport_width",
        "in": "query",
        "required": False,
        "schema": {"type": "integer", "default": 1280, "minimum": 1},
        "description": "视口宽度 (像素)",
    },
    {
        "name": "viewport_height",
        "in": "query",
        "required": False,
        "schema": {"type": "integer", "default": 720, "minimum": 1},
        "description": "视口高度 (像素)",
    },
    {
        "name": "locale",
        "in": "query",
        "required": False,
        "schema": {"type": "string", "default": "zh-CN"},
        "description": "语言区域",
    },
]


# ============================================================
# 辅助函数
# ============================================================


def _extract_path_params(path: str) -> list[str]:
    """从路径中提取路径参数名.

    例如 "/render/{artifact_id}" → ["artifact_id"]
         "/registry/supports/{mime_type}" → ["mime_type"]
    """
    # 匹配 {param} 或 {param:path} 格式
    raw_params = re.findall(r"\{([^}]+)\}", path)
    # 去除 :path 等后缀
    return [p.split(":")[0] for p in raw_params]


def _generate_operation_id(method: str, path: str, api_prefix: str = "") -> str:
    """从 HTTP 方法和路径生成 operationId.

    例如:
        GET /health → get_health
        POST /render → post_render
        GET /render/{artifact_id} → get_render_artifact_id
        GET /registry/supports/{mime_type} → get_registry_supports_mime_type
    """
    # 去除 API 前缀
    clean_path = path
    if api_prefix and clean_path.startswith(api_prefix):
        clean_path = clean_path[len(api_prefix):]

    # 去除花括号
    cleaned = re.sub(r"[{}]", "", clean_path)
    # 按 / 分割, 过滤空段
    parts = [p for p in cleaned.split("/") if p]
    # 连接
    name = "_".join(parts) if parts else "root"
    return f"{method.lower()}_{name}"


def _build_path_params(path: str) -> list[dict[str, Any]]:
    """为路径中的参数生成 OpenAPI path parameter 定义."""
    params = []
    for param_name in _extract_path_params(path):
        params.append({
            "name": param_name,
            "in": "path",
            "required": True,
            "schema": {"type": "string"},
            "description": f"路径参数: {param_name}",
        })
    return params


def _build_query_params(path: str, method: str) -> list[dict[str, Any]]:
    """根据路径和方法生成查询参数定义."""
    # 去除 API 前缀以匹配
    if path.endswith("/artifacts") and method == "get":
        return list(_QUERY_PARAMS_ARTIFACTS)
    if path.endswith("/render") and method == "post":
        return list(_QUERY_PARAMS_RENDER_CONTEXT)
    return []


def _build_request_body(path: str, method: str) -> dict[str, Any] | None:
    """根据路径和方法生成 requestBody 定义."""
    if method == "post" and path.endswith("/render"):
        return {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Artifact"},
                },
            },
            "description": "待渲染的 Artifact",
        }
    if method == "put" and "{artifact_id}" in path and "/render/" in path:
        return {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ArtifactDiff"},
                },
            },
            "description": "Artifact 增量差异 (RFC 6902 JSON Patch)",
        }
    if method == "post" and path.endswith("/edit"):
        return {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ArtifactDiff"},
                },
            },
            "description": "编辑差异操作 (RFC 6902 JSON Patch)",
        }
    return None


def _build_responses(path: str, method: str) -> dict[str, Any]:
    """根据路径和方法生成 responses 定义.

    所有端点: 200 (成功) + 500 (服务器错误)
    含 {artifact_id} 的端点: + 404 (未找到)
    POST/PUT 端点: + 422 (校验错误)
    """
    responses: dict[str, Any] = {
        "200": {
            "description": "成功响应",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/SuccessResponse"},
                },
            },
        },
        "500": {
            "description": "服务器内部错误",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                },
            },
        },
    }

    # 含 {artifact_id} 的端点添加 404
    if "{artifact_id}" in path:
        responses["404"] = {
            "description": "Artifact 未找到",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                },
            },
        }

    # POST/PUT 端点添加 422
    if method in ("post", "put"):
        responses["422"] = {
            "description": "请求体验证失败",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                },
            },
        }

    return responses


# ============================================================
# 核心: generate_openapi_spec
# ============================================================


def generate_openapi_spec(router: "L7Router") -> dict[str, Any]:
    """从 L7Router 实例生成完整的 OpenAPI 3.0.3 规范.

    遍历 ``router.get_routes_summary()`` 返回的所有路由,
    自动生成路径参数、查询参数、请求体和响应定义。

    Args:
        router: L7Router 实例。

    Returns:
        完整的 OpenAPI 3.0.3 spec 字典, 包含:
        - ``openapi``: 版本号 "3.0.3"
        - ``info``: API 元信息 (title, description, version)
        - ``paths``: 从路由摘要自动生成的路径定义
        - ``components.schemas``: Artifact, RenderContext, RenderDescriptor,
          ArtifactDiff, ErrorResponse, SuccessResponse 的 schema 定义
    """
    api_prefix = getattr(router, "_api_prefix", "")
    routes_summary = router.get_routes_summary()

    paths: dict[str, Any] = {}

    for route in routes_summary:
        path = route["path"]
        methods = route.get("methods", [])
        description = route.get("description", "")

        if path not in paths:
            paths[path] = {}

        for method_raw in methods:
            method = method_raw.lower()

            operation: dict[str, Any] = {
                "summary": description,
                "operationId": _generate_operation_id(method, path, api_prefix),
                "responses": _build_responses(path, method),
            }

            # 参数: 路径参数 + 查询参数
            params = _build_path_params(path)
            params.extend(_build_query_params(path, method))
            if params:
                operation["parameters"] = params

            # 请求体 (POST/PUT)
            body = _build_request_body(path, method)
            if body is not None:
                operation["requestBody"] = body

            paths[path][method] = operation

    spec: dict[str, Any] = {
        "openapi": OPENAPI_VERSION,
        "info": {
            "title": API_TITLE,
            "description": API_DESCRIPTION,
            "version": API_VERSION,
        },
        "paths": paths,
        "components": {
            "schemas": _build_schemas(),
        },
    }

    return spec


# ============================================================
# openapi_handler — GET /openapi.json
# ============================================================


def openapi_handler(router: "L7Router"):
    """返回一个 Starlette handler, 在 GET /openapi.json 提供 OpenAPI spec.

    Args:
        router: L7Router 实例。

    Returns:
        async handler 函数, 接受 Request, 返回 JSONResponse。
    """
    async def _handler(request: Request) -> JSONResponse:
        spec = generate_openapi_spec(router)
        return JSONResponse(spec)

    return _handler


# ============================================================
# swagger_ui_html — Swagger UI HTML
# ============================================================


def swagger_ui_html(router: "L7Router") -> str:
    """返回 Swagger UI HTML 字符串.

    HTML 加载 Swagger UI CDN 资源, 并从 ``/openapi.json`` (或带前缀的路径)
    获取 OpenAPI spec。

    Args:
        router: L7Router 实例 (用于确定 API 前缀)。

    Returns:
        Swagger UI HTML 字符串。
    """
    api_prefix = getattr(router, "_api_prefix", "")
    spec_url = f"{api_prefix}/openapi.json" if api_prefix else "/openapi.json"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{API_TITLE} - Swagger UI</title>
    <link rel="stylesheet" type="text/css" href="{_SWAGGER_UI_CDN}/swagger-ui.css">
    <link rel="icon" type="image/png" href="{_SWAGGER_UI_CDN}/favicon-32x32.png" sizes="32x32">
    <link rel="icon" type="image/png" href="{_SWAGGER_UI_CDN}/favicon-16x16.png" sizes="16x16">
    <style>
        html {{ box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }}
        *, *:before, *:after {{ box-sizing: inherit; }}
        body {{ margin: 0; background: #fafafa; }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="{_SWAGGER_UI_CDN}/swagger-ui-bundle.js" charset="UTF-8"></script>
    <script src="{_SWAGGER_UI_CDN}/swagger-ui-standalone-preset.js" charset="UTF-8"></script>
    <script>
        window.onload = function() {{
            window.ui = SwaggerUIBundle({{
                url: "{spec_url}",
                dom_id: "#swagger-ui",
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            }});
        }};
    </script>
</body>
</html>"""


# ============================================================
# swagger_ui_handler — GET /docs
# ============================================================


def swagger_ui_handler(router: "L7Router"):
    """返回一个 Starlette handler, 在 GET /docs 提供 Swagger UI.

    Args:
        router: L7Router 实例。

    Returns:
        async handler 函数, 接受 Request, 返回 HTMLResponse。
    """
    async def _handler(request: Request) -> HTMLResponse:
        return HTMLResponse(swagger_ui_html(router))

    return _handler


__all__ = [
    "generate_openapi_spec",
    "openapi_handler",
    "swagger_ui_html",
    "swagger_ui_handler",
]
