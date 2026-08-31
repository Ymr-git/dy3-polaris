"""L7 体验呈现层 — OpenAPI Schema 自动生成测试 (TDD).

遵循严格 TDD: 先写测试 → 观察失败 → 实现 → 验证通过.

测试覆盖:
1. generate_openapi_spec 返回有效的 OpenAPI 3.0.3 字典
2. info 部分包含 title, description, version
3. 所有路由从 get_routes_summary 出现在 paths 中
4. 路径参数 (artifact_id, mime_type) 正确定义
5. GET /artifacts 的查询参数 (type, source_agent, page, size, sort) 定义
6. POST /render 有 requestBody 引用 Artifact schema
7. POST /artifacts/{id}/edit 有 requestBody 引用 JSON Patch 操作
8. 所有端点有 200 响应
9. 错误响应 (404, 422, 500) 在适当位置定义
10. components/schemas 包含 Artifact, RenderContext, RenderDescriptor, ErrorResponse
11. Swagger UI HTML 包含预期元素
12. OpenAPI 端点返回 200 和正确 content type
13. /docs 端点返回 HTML
14. API 前缀在 OpenAPI paths 中被尊重

设计参考:
- FastAPI auto-docs: 从路由定义生成 OpenAPI
- Swagger UI: 交互式 API 文档
"""

from __future__ import annotations

import logging
from typing import Any

logging.disable(logging.CRITICAL)

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l7.artifact_manager import ArtifactManager
from dy3_polaris.l7.irenderer import IRenderer
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactDiff,
    ArtifactType,
    RenderContext,
    RenderDescriptor,
)
from dy3_polaris.l7.registry import RendererRegistry
from dy3_polaris.l7.api.router import L7Router

# TDD RED phase: openapi 模块可能尚未实现
try:
    from dy3_polaris.l7.api.openapi import (
        generate_openapi_spec,
        openapi_handler,
        swagger_ui_html,
    )
    _OPENAPI_AVAILABLE = True
except ImportError:
    _OPENAPI_AVAILABLE = False
    generate_openapi_spec = None  # type: ignore[assignment]
    openapi_handler = None  # type: ignore[assignment]
    swagger_ui_html = None  # type: ignore[assignment]


# ============================================================
# 测试用渲染器
# ============================================================


class _TestRenderer(IRenderer):
    """测试用 Markdown 渲染器."""

    __test__ = False

    _SUPPORTED_MIMES = ["text/vnd.dy3+markdown", "text/plain"]

    def render(self, artifact: Artifact, context: RenderContext) -> RenderDescriptor:
        return RenderDescriptor(
            artifact_id=artifact.artifact_id,
            mime=artifact.mime,
            html=f"<div class='markdown'>{artifact.title}</div>",
            config={"theme": context.theme, "mode": context.learner_mode.value},
        )

    def update(self, diff: ArtifactDiff) -> RenderDescriptor:
        return RenderDescriptor(
            artifact_id=diff.artifact_id,
            mime="text/vnd.dy3+markdown",
            html="<div class='markdown'>updated</div>",
        )

    def destroy(self) -> None:
        pass

    def supported_mime_types(self) -> list[str]:
        return list(self._SUPPORTED_MIMES)


# ============================================================
# 测试辅助
# ============================================================


def _make_router(api_prefix: str = "") -> L7Router:
    """创建带 _TestRenderer 的 L7Router."""
    reg = RendererRegistry()
    reg.register(_TestRenderer())
    return L7Router(
        artifact_manager=ArtifactManager(),
        registry=reg,
        api_prefix=api_prefix,
    )


def _make_spec(api_prefix: str = "") -> tuple[dict[str, Any], L7Router]:
    """生成 OpenAPI spec 并返回 (spec, router)."""
    router = _make_router(api_prefix=api_prefix)
    spec = generate_openapi_spec(router)
    return spec, router


def _query_param_names(spec: dict, path: str, method: str) -> list[str]:
    """获取指定操作的查询参数名列表."""
    op = spec["paths"][path][method]
    return [p["name"] for p in op.get("parameters", []) if p.get("in") == "query"]


def _path_param_names(spec: dict, path: str, method: str) -> list[str]:
    """获取指定操作的路径参数名列表."""
    op = spec["paths"][path][method]
    return [p["name"] for p in op.get("parameters", []) if p.get("in") == "path"]


# ============================================================
# 1. generate_openapi_spec 基本结构
# ============================================================


class TestGenerateOpenAPISpec:
    """generate_openapi_spec 返回有效的 OpenAPI 3.0.3 字典."""

    def test_spec_returns_dict(self):
        """generate_openapi_spec 返回字典."""
        spec, _ = _make_spec()
        assert isinstance(spec, dict)

    def test_openapi_version_is_3_0_3(self):
        """openapi 版本为 3.0.3."""
        spec, _ = _make_spec()
        assert spec["openapi"] == "3.0.3"

    def test_info_has_title(self):
        """info 包含 title."""
        spec, _ = _make_spec()
        assert "title" in spec["info"]
        assert isinstance(spec["info"]["title"], str)
        assert len(spec["info"]["title"]) > 0

    def test_info_has_description(self):
        """info 包含 description."""
        spec, _ = _make_spec()
        assert "description" in spec["info"]
        assert isinstance(spec["info"]["description"], str)
        assert len(spec["info"]["description"]) > 0

    def test_info_has_version(self):
        """info 包含 version."""
        spec, _ = _make_spec()
        assert "version" in spec["info"]
        assert isinstance(spec["info"]["version"], str)
        assert len(spec["info"]["version"]) > 0

    def test_has_paths_key(self):
        """spec 包含 paths 键."""
        spec, _ = _make_spec()
        assert "paths" in spec
        assert isinstance(spec["paths"], dict)
        assert len(spec["paths"]) > 0

    def test_has_components_key(self):
        """spec 包含 components 键."""
        spec, _ = _make_spec()
        assert "components" in spec

    def test_components_has_schemas(self):
        """components 包含 schemas."""
        spec, _ = _make_spec()
        assert "schemas" in spec["components"]
        assert isinstance(spec["components"]["schemas"], dict)

    def test_all_routes_in_paths(self):
        """所有 get_routes_summary 的路由出现在 paths 中."""
        router = _make_router()
        spec = generate_openapi_spec(router)
        summary = router.get_routes_summary()
        for route in summary:
            assert route["path"] in spec["paths"], f"Path {route['path']} not in spec"

    def test_health_path_in_spec(self):
        """/health 路径在 spec 中."""
        spec, _ = _make_spec()
        assert "/health" in spec["paths"]

    def test_render_post_in_spec(self):
        """/render POST 在 spec 中."""
        spec, _ = _make_spec()
        assert "/render" in spec["paths"]
        assert "post" in spec["paths"]["/render"]

    def test_render_by_id_in_spec(self):
        """/render/{artifact_id} 在 spec 中."""
        spec, _ = _make_spec()
        assert "/render/{artifact_id}" in spec["paths"]

    def test_artifacts_in_spec(self):
        """/artifacts 在 spec 中."""
        spec, _ = _make_spec()
        assert "/artifacts" in spec["paths"]

    def test_stats_in_spec(self):
        """/stats 在 spec 中."""
        spec, _ = _make_spec()
        assert "/stats" in spec["paths"]

    def test_schemas_has_artifact(self):
        """schemas 包含 Artifact."""
        spec, _ = _make_spec()
        assert "Artifact" in spec["components"]["schemas"]

    def test_schemas_has_render_context(self):
        """schemas 包含 RenderContext."""
        spec, _ = _make_spec()
        assert "RenderContext" in spec["components"]["schemas"]

    def test_schemas_has_render_descriptor(self):
        """schemas 包含 RenderDescriptor."""
        spec, _ = _make_spec()
        assert "RenderDescriptor" in spec["components"]["schemas"]

    def test_schemas_has_error_response(self):
        """schemas 包含 ErrorResponse."""
        spec, _ = _make_spec()
        assert "ErrorResponse" in spec["components"]["schemas"]

    def test_artifact_schema_has_properties(self):
        """Artifact schema 有 properties 字段."""
        spec, _ = _make_spec()
        artifact_schema = spec["components"]["schemas"]["Artifact"]
        assert "properties" in artifact_schema
        props = artifact_schema["properties"]
        assert "artifact_id" in props
        assert "type" in props
        assert "mime" in props

    def test_render_descriptor_schema_has_properties(self):
        """RenderDescriptor schema 有 properties 字段."""
        spec, _ = _make_spec()
        desc_schema = spec["components"]["schemas"]["RenderDescriptor"]
        assert "properties" in desc_schema
        props = desc_schema["properties"]
        assert "render_id" in props
        assert "artifact_id" in props
        assert "html" in props


# ============================================================
# 2. 路径参数
# ============================================================


class TestPathParameters:
    """路径参数 (artifact_id, mime_type) 正确定义."""

    def test_render_by_id_has_artifact_id_param(self):
        """GET /render/{artifact_id} 有 artifact_id 路径参数."""
        spec, _ = _make_spec()
        params = _path_param_names(spec, "/render/{artifact_id}", "get")
        assert "artifact_id" in params

    def test_artifacts_by_id_has_artifact_id_param(self):
        """GET /artifacts/{artifact_id} 有 artifact_id 路径参数."""
        spec, _ = _make_spec()
        params = _path_param_names(spec, "/artifacts/{artifact_id}", "get")
        assert "artifact_id" in params

    def test_versions_has_artifact_id_param(self):
        """GET /artifacts/{artifact_id}/versions 有 artifact_id 路径参数."""
        spec, _ = _make_spec()
        path = "/artifacts/{artifact_id}/versions"
        params = _path_param_names(spec, path, "get")
        assert "artifact_id" in params

    def test_edit_has_artifact_id_param(self):
        """POST /artifacts/{artifact_id}/edit 有 artifact_id 路径参数."""
        spec, _ = _make_spec()
        path = "/artifacts/{artifact_id}/edit"
        params = _path_param_names(spec, path, "post")
        assert "artifact_id" in params

    def test_supports_has_mime_type_param(self):
        """GET /registry/supports/{mime_type} 有 mime_type 路径参数."""
        spec, _ = _make_spec()
        path = "/registry/supports/{mime_type}"
        params = _path_param_names(spec, path, "get")
        assert "mime_type" in params

    def test_path_params_are_required(self):
        """所有路径参数标记为 required=True."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                for param in op.get("parameters", []):
                    if param.get("in") == "path":
                        assert param.get("required") is True, (
                            f"Path param {param['name']} in {method.upper()} {path} "
                            f"should be required"
                        )

    def test_path_params_have_string_schema(self):
        """路径参数的 schema 类型为 string."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                for param in op.get("parameters", []):
                    if param.get("in") == "path":
                        assert param.get("schema", {}).get("type") == "string"


# ============================================================
# 3. 查询参数
# ============================================================


class TestQueryParameters:
    """GET /artifacts 和 POST /render 的查询参数."""

    def test_list_artifacts_has_type_query(self):
        """GET /artifacts 有 type 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/artifacts", "get")
        assert "type" in params

    def test_list_artifacts_has_source_agent_query(self):
        """GET /artifacts 有 source_agent 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/artifacts", "get")
        assert "source_agent" in params

    def test_list_artifacts_has_page_query(self):
        """GET /artifacts 有 page 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/artifacts", "get")
        assert "page" in params

    def test_list_artifacts_has_size_query(self):
        """GET /artifacts 有 size 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/artifacts", "get")
        assert "size" in params

    def test_list_artifacts_has_sort_query(self):
        """GET /artifacts 有 sort 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/artifacts", "get")
        assert "sort" in params

    def test_list_artifacts_has_session_id_query(self):
        """GET /artifacts 有 session_id 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/artifacts", "get")
        assert "session_id" in params

    def test_list_artifacts_has_kp_id_query(self):
        """GET /artifacts 有 kp_id 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/artifacts", "get")
        assert "kp_id" in params

    def test_render_has_theme_query(self):
        """POST /render 有 theme 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/render", "post")
        assert "theme" in params

    def test_render_has_learner_mode_query(self):
        """POST /render 有 learner_mode 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/render", "post")
        assert "learner_mode" in params

    def test_render_has_viewport_width_query(self):
        """POST /render 有 viewport_width 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/render", "post")
        assert "viewport_width" in params

    def test_render_has_viewport_height_query(self):
        """POST /render 有 viewport_height 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/render", "post")
        assert "viewport_height" in params

    def test_render_has_locale_query(self):
        """POST /render 有 locale 查询参数."""
        spec, _ = _make_spec()
        params = _query_param_names(spec, "/render", "post")
        assert "locale" in params

    def test_query_params_not_required(self):
        """查询参数标记为 required=False."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                for param in op.get("parameters", []):
                    if param.get("in") == "query":
                        assert param.get("required") is False, (
                            f"Query param {param['name']} in {method.upper()} {path} "
                            f"should not be required"
                        )


# ============================================================
# 4. 请求体
# ============================================================


class TestRequestBody:
    """POST/PUT 端点的 requestBody."""

    def test_render_post_has_request_body(self):
        """POST /render 有 requestBody."""
        spec, _ = _make_spec()
        op = spec["paths"]["/render"]["post"]
        assert "requestBody" in op

    def test_render_post_request_body_ref_artifact(self):
        """POST /render requestBody 引用 Artifact schema."""
        spec, _ = _make_spec()
        op = spec["paths"]["/render"]["post"]
        body = op["requestBody"]
        schema_ref = body["content"]["application/json"]["schema"]
        assert "$ref" in schema_ref
        assert "Artifact" in schema_ref["$ref"]

    def test_render_post_request_body_required(self):
        """POST /render requestBody 标记为 required."""
        spec, _ = _make_spec()
        op = spec["paths"]["/render"]["post"]
        assert op["requestBody"].get("required") is True

    def test_render_put_has_request_body(self):
        """PUT /render/{artifact_id} 有 requestBody."""
        spec, _ = _make_spec()
        op = spec["paths"]["/render/{artifact_id}"]["put"]
        assert "requestBody" in op

    def test_render_put_request_body_has_json_content(self):
        """PUT /render/{artifact_id} requestBody 有 application/json content."""
        spec, _ = _make_spec()
        op = spec["paths"]["/render/{artifact_id}"]["put"]
        body = op["requestBody"]
        assert "application/json" in body["content"]

    def test_edit_has_request_body(self):
        """POST /artifacts/{artifact_id}/edit 有 requestBody."""
        spec, _ = _make_spec()
        path = "/artifacts/{artifact_id}/edit"
        op = spec["paths"][path]["post"]
        assert "requestBody" in op

    def test_edit_request_body_ref_artifact_diff(self):
        """POST /artifacts/{artifact_id}/edit requestBody 引用 ArtifactDiff schema."""
        spec, _ = _make_spec()
        path = "/artifacts/{artifact_id}/edit"
        op = spec["paths"][path]["post"]
        body = op["requestBody"]
        schema_ref = body["content"]["application/json"]["schema"]
        assert "$ref" in schema_ref
        assert "ArtifactDiff" in schema_ref["$ref"]

    def test_schemas_has_artifact_diff(self):
        """schemas 包含 ArtifactDiff (用于 edit 端点)."""
        spec, _ = _make_spec()
        assert "ArtifactDiff" in spec["components"]["schemas"]

    def test_get_endpoints_no_request_body(self):
        """GET 端点没有 requestBody."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            if "get" in methods:
                assert "requestBody" not in methods["get"], (
                    f"GET {path} should not have requestBody"
                )


# ============================================================
# 5. 响应
# ============================================================


class TestResponses:
    """端点响应定义."""

    def test_all_endpoints_have_200(self):
        """所有端点有 200 响应."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                assert "200" in op["responses"], (
                    f"{method.upper()} {path} missing 200 response"
                )

    def test_response_200_has_description(self):
        """200 响应有 description."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                resp_200 = op["responses"]["200"]
                assert "description" in resp_200

    def test_artifact_id_endpoints_have_404(self):
        """包含 {artifact_id} 的端点有 404 响应."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            if "{artifact_id}" in path:
                for method, op in methods.items():
                    assert "404" in op["responses"], (
                        f"{method.upper()} {path} missing 404 response"
                    )

    def test_post_put_endpoints_have_422(self):
        """POST/PUT 端点有 422 响应."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                if method in ("post", "put"):
                    assert "422" in op["responses"], (
                        f"{method.upper()} {path} missing 422 response"
                    )

    def test_all_endpoints_have_500(self):
        """所有端点有 500 响应."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                assert "500" in op["responses"], (
                    f"{method.upper()} {path} missing 500 response"
                )

    def test_error_response_ref_error_schema(self):
        """错误响应引用 ErrorResponse schema."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                for code, resp in op["responses"].items():
                    if code != "200":
                        content = resp.get("content", {})
                        if "application/json" in content:
                            schema = content["application/json"].get("schema", {})
                            if "$ref" in schema:
                                assert "ErrorResponse" in schema["$ref"], (
                                    f"{method.upper()} {path} {code} response "
                                    f"should reference ErrorResponse"
                                )


# ============================================================
# 6. operationId
# ============================================================


class TestOperationId:
    """operationId 生成."""

    def test_operation_id_present(self):
        """每个操作有 operationId."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                assert "operationId" in op, (
                    f"{method.upper()} {path} missing operationId"
                )
                assert isinstance(op["operationId"], str)
                assert len(op["operationId"]) > 0

    def test_operation_id_starts_with_method(self):
        """operationId 以 HTTP 方法开头."""
        spec, _ = _make_spec()
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                assert op["operationId"].startswith(method), (
                    f"operationId {op['operationId']} should start with '{method}'"
                )

    def test_operation_ids_unique(self):
        """所有 operationId 唯一."""
        spec, _ = _make_spec()
        ids = []
        for path, methods in spec["paths"].items():
            for method, op in methods.items():
                ids.append(op["operationId"])
        assert len(ids) == len(set(ids)), f"Duplicate operationIds: {ids}"

    def test_summary_from_description(self):
        """操作的 summary 来自路由描述."""
        router = _make_router()
        spec = generate_openapi_spec(router)
        summary = router.get_routes_summary()
        for route in summary:
            path = route["path"]
            method = route["methods"][0].lower()
            op = spec["paths"][path][method]
            assert op["summary"] == route["description"]


# ============================================================
# 7. Swagger UI HTML
# ============================================================


class TestSwaggerUI:
    """swagger_ui_html 函数."""

    def test_swagger_ui_returns_string(self):
        """swagger_ui_html 返回字符串."""
        router = _make_router()
        html = swagger_ui_html(router)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_swagger_ui_contains_cdn(self):
        """HTML 包含 Swagger UI CDN 引用."""
        router = _make_router()
        html = swagger_ui_html(router)
        assert "swagger-ui" in html.lower()

    def test_swagger_ui_contains_js_bundle(self):
        """HTML 包含 Swagger UI JS bundle."""
        router = _make_router()
        html = swagger_ui_html(router)
        assert "swagger-ui-bundle" in html.lower() or "swagger-ui.js" in html.lower()

    def test_swagger_ui_contains_css(self):
        """HTML 包含 Swagger UI CSS."""
        router = _make_router()
        html = swagger_ui_html(router)
        assert "swagger-ui.css" in html.lower()

    def test_swagger_ui_contains_openapi_url(self):
        """HTML 包含 /openapi.json 引用."""
        router = _make_router()
        html = swagger_ui_html(router)
        assert "/openapi.json" in html

    def test_swagger_ui_contains_swagger_ui_div(self):
        """HTML 包含 swagger-ui div."""
        router = _make_router()
        html = swagger_ui_html(router)
        assert "swagger-ui" in html

    def test_swagger_ui_with_prefix(self):
        """带前缀时 HTML 引用 {prefix}/openapi.json."""
        router = _make_router(api_prefix="/api/v1")
        html = swagger_ui_html(router)
        assert "/api/v1/openapi.json" in html

    def test_swagger_ui_is_valid_html(self):
        """HTML 是有效的 HTML 文档."""
        router = _make_router()
        html = swagger_ui_html(router)
        assert html.strip().lower().startswith("<!doctype html>") or \
               html.strip().lower().startswith("<html")


# ============================================================
# 8. OpenAPI HTTP 端点
# ============================================================


class TestOpenAPIEndpoint:
    """GET /openapi.json 和 GET /docs 端点."""

    def test_openapi_endpoint_returns_200(self):
        """GET /openapi.json 返回 200."""
        router = _make_router()
        client = TestClient(router.create_app())
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_endpoint_content_type(self):
        """GET /openapi.json content-type 为 application/json."""
        router = _make_router()
        client = TestClient(router.create_app())
        resp = client.get("/openapi.json")
        assert "application/json" in resp.headers.get("content-type", "")

    def test_openapi_endpoint_returns_valid_spec(self):
        """GET /openapi.json 返回有效的 OpenAPI spec."""
        router = _make_router()
        client = TestClient(router.create_app())
        resp = client.get("/openapi.json")
        spec = resp.json()
        assert spec["openapi"] == "3.0.3"
        assert "paths" in spec
        assert "components" in spec

    def test_openapi_endpoint_includes_all_paths(self):
        """GET /openapi.json 包含所有路由."""
        router = _make_router()
        client = TestClient(router.create_app())
        resp = client.get("/openapi.json")
        spec = resp.json()
        summary = router.get_routes_summary()
        for route in summary:
            assert route["path"] in spec["paths"]

    def test_docs_endpoint_returns_200(self):
        """GET /docs 返回 200."""
        router = _make_router()
        client = TestClient(router.create_app())
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_docs_endpoint_content_type(self):
        """GET /docs content-type 为 text/html."""
        router = _make_router()
        client = TestClient(router.create_app())
        resp = client.get("/docs")
        assert "text/html" in resp.headers.get("content-type", "")

    def test_docs_endpoint_contains_swagger(self):
        """GET /docs 包含 Swagger UI 元素."""
        router = _make_router()
        client = TestClient(router.create_app())
        resp = client.get("/docs")
        assert "swagger-ui" in resp.text.lower()


# ============================================================
# 9. API 前缀
# ============================================================


class TestAPIPrefix:
    """API 前缀在 OpenAPI 中被尊重."""

    def test_prefix_respected_in_paths(self):
        """带前缀时, paths 包含前缀路径."""
        spec, router = _make_spec(api_prefix="/api/v1")
        summary = router.get_routes_summary()
        for route in summary:
            assert route["path"] in spec["paths"], (
                f"Prefixed path {route['path']} not in spec"
            )

    def test_prefix_health_path(self):
        """带前缀时, /api/v1/health 在 paths 中."""
        spec, _ = _make_spec(api_prefix="/api/v1")
        assert "/api/v1/health" in spec["paths"]

    def test_prefix_openapi_available_without_prefix(self):
        """带前缀时, /openapi.json 仍可在根路径访问."""
        router = _make_router(api_prefix="/api/v1")
        client = TestClient(router.create_app())
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_prefix_openapi_available_with_prefix(self):
        """带前缀时, /api/v1/openapi.json 也可访问."""
        router = _make_router(api_prefix="/api/v1")
        client = TestClient(router.create_app())
        resp = client.get("/api/v1/openapi.json")
        assert resp.status_code == 200

    def test_prefix_docs_available_without_prefix(self):
        """带前缀时, /docs 仍可在根路径访问."""
        router = _make_router(api_prefix="/api/v1")
        client = TestClient(router.create_app())
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_prefix_docs_available_with_prefix(self):
        """带前缀时, /api/v1/docs 也可访问."""
        router = _make_router(api_prefix="/api/v1")
        client = TestClient(router.create_app())
        resp = client.get("/api/v1/docs")
        assert resp.status_code == 200

    def test_prefixed_spec_has_prefixed_paths(self):
        """带前缀的 spec 不包含无前缀路径."""
        spec, _ = _make_spec(api_prefix="/api/v1")
        # 所有路径应包含 /api/v1 前缀
        for path in spec["paths"]:
            assert path.startswith("/api/v1"), f"Path {path} should start with /api/v1"


# ============================================================
# 10. openapi_handler 函数
# ============================================================


class TestOpenAPIHandler:
    """openapi_handler 函数."""

    def test_handler_returns_callable(self):
        """openapi_handler 返回可调用对象."""
        router = _make_router()
        handler = openapi_handler(router)
        assert callable(handler)

    def test_handler_serves_spec(self):
        """handler 返回的响应体是 OpenAPI spec."""
        import asyncio
        router = _make_router()
        handler = openapi_handler(router)

        async def _call():
            return await handler(None)  # type: ignore[arg-type]

        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(_call())
        finally:
            loop.close()

        assert response.status_code == 200
        body = response.body
        import json
        spec = json.loads(body)
        assert spec["openapi"] == "3.0.3"
