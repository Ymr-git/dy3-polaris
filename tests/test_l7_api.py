"""L7 体验呈现层 — REST API 路由器测试 (TDD).

基于 Starlette TestClient 进行同步无端口 API 测试,
遵循项目统一响应格式 (与 L5/L6 API 一致): ``{"code": 0, "data": ..., "message": ""}``.

测试覆盖:
1. 健康检查端点 GET /health
2. 渲染端点 POST /render, GET /render/{id}, PUT /render/{id}, DELETE /render/{id}
3. Artifact 端点 GET /artifacts, GET /artifacts/{id}, GET /artifacts/{id}/versions, POST /artifacts/{id}/edit
4. 注册中心端点 GET /registry/mime-types, GET /registry/renderers, GET /registry/supports/{mime}
5. 统计端点 GET /stats
6. L7Router 元信息 (get_routes_summary)
7. 错误处理与边界情况

设计参考:
- L5 Router 模式: _ok / _err / _safe_dump / _RouteHandlers / L7Router
- Jupyter MIME Bundle: 多模态渲染路由
- VS Code CustomEditor: 渲染实例生命周期 (render/update/destroy)
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

# L7Router 在实现前不存在, 这是 RED phase
try:
    from dy3_polaris.l7.api import L7Router
    _L7_ROUTER_AVAILABLE = True
except ImportError:
    _L7_ROUTER_AVAILABLE = False
    L7Router = None  # type: ignore[assignment,misc]


# ============================================================
# 测试用渲染器
# ============================================================


class TestRenderer(IRenderer):
    """测试用 Markdown 渲染器 — 实现 IRenderer 接口."""

    # 防止 pytest 将其误识别为测试类
    __test__ = False

    _SUPPORTED_MIMES = [
        "text/vnd.dy3+markdown",
        "text/plain",
    ]

    def __init__(self) -> None:
        self._destroyed = False
        self._render_count = 0
        self._update_count = 0

    def render(
        self, artifact: Artifact, context: RenderContext
    ) -> RenderDescriptor:
        self._render_count += 1
        return RenderDescriptor(
            artifact_id=artifact.artifact_id,
            mime=artifact.mime,
            html=f"<div class='markdown'>{artifact.title}</div>",
            config={"theme": context.theme, "mode": context.learner_mode.value},
            metadata={"renderer": "TestRenderer", "render_count": self._render_count},
        )

    def update(self, diff: ArtifactDiff) -> RenderDescriptor:
        self._update_count += 1
        return RenderDescriptor(
            artifact_id=diff.artifact_id,
            mime="text/vnd.dy3+markdown",
            html="<div class='markdown'>updated</div>",
            metadata={"renderer": "TestRenderer", "update_count": self._update_count},
        )

    def destroy(self) -> None:
        self._destroyed = True

    def supported_mime_types(self) -> list[str]:
        return list(self._SUPPORTED_MIMES)


# ============================================================
# 测试辅助
# ============================================================


def _make_registry() -> RendererRegistry:
    """创建带 TestRenderer 的注册表."""
    reg = RendererRegistry()
    reg.register(TestRenderer())
    return reg


def _make_artifact(
    *,
    artifact_id: str = "art-test-001",
    artifact_type: ArtifactType = ArtifactType.TEXT,
    mime: str = "text/vnd.dy3+markdown",
    source_agent: str = "agent.alpha",
    title: str = "学情诊断报告",
    payload: dict[str, Any] | None = None,
) -> Artifact:
    """构造测试用 Artifact."""
    if payload is None:
        payload = {"content": "# Hello\n学情诊断报告正文"}
    return Artifact(
        artifact_id=artifact_id,
        type=artifact_type,
        mime=mime,
        source_agent=source_agent,
        payload=payload,
        title=title,
        session_id="sess-001",
    )


def _make_router(
    *,
    registry: RendererRegistry | None = None,
    artifact_manager: ArtifactManager | None = None,
) -> Any:
    """构造 L7Router 实例 (注入可控的 registry / artifact_manager)."""
    reg = registry if registry is not None else _make_registry()
    am = artifact_manager if artifact_manager is not None else ArtifactManager()
    return L7Router(
        artifact_manager=am,
        registry=reg,
        pipeline=None,
    )


def _make_client(
    *,
    registry: RendererRegistry | None = None,
    artifact_manager: ArtifactManager | None = None,
) -> TestClient:
    """构造 TestClient."""
    router = _make_router(registry=registry, artifact_manager=artifact_manager)
    return TestClient(router.create_app())


# ============================================================
# 1. 健康检查端点
# ============================================================


@pytest.mark.skipif(not _L7_ROUTER_AVAILABLE, reason="L7Router 未实现")
class TestHealthEndpoint:
    """GET /health — L7 体验呈现层健康检查."""

    def test_health_returns_200(self):
        """健康检查返回 200 和统一响应格式."""
        client = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "healthy"
        assert data["data"]["layer"] == "L7"

    def test_health_includes_service_info(self):
        """健康检查包含 timestamp 与 services 字段."""
        client = _make_client()
        resp = client.get("/health")
        data = resp.json()["data"]
        assert "timestamp" in data
        assert "services" in data
        services = data["services"]
        # 至少应包含 registry / artifact_manager / pipeline 三个服务状态
        assert "registry" in services
        assert "artifact_manager" in services

    def test_health_message_empty(self):
        """成功响应 message 字段为空字符串."""
        client = _make_client()
        resp = client.get("/health")
        assert resp.json()["message"] == ""


# ============================================================
# 2. 渲染端点
# ============================================================


@pytest.mark.skipif(not _L7_ROUTER_AVAILABLE, reason="L7Router 未实现")
class TestRenderEndpoints:
    """渲染端点: POST /render, GET/PUT/DELETE /render/{artifact_id}."""

    def test_render_returns_200_with_descriptor(self):
        """POST /render 返回 200 和 RenderDescriptor."""
        client = _make_client()
        artifact = _make_artifact()
        resp = client.post("/render", json=artifact.to_dict())
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        desc = data["data"]
        assert desc["artifact_id"] == artifact.artifact_id
        assert desc["mime"] == artifact.mime
        assert "html" in desc
        assert "render_id" in desc

    def test_render_with_invalid_mime_returns_422(self):
        """POST /render 未注册的 MIME 返回 422."""
        client = _make_client()
        # 使用 TEXT 类型 + 合法 payload, 确保 validate() 通过,
        # 让 422 由 MIME 未注册触发 (而非 payload 校验失败)
        artifact = _make_artifact(
            mime="application/x-totally-unknown",
            artifact_type=ArtifactType.TEXT,
            payload={"content": "hi"},
        )
        resp = client.post("/render", json=artifact.to_dict())
        assert resp.status_code == 422
        data = resp.json()
        assert data["code"] != 0
        assert "message" in data

    def test_render_with_missing_body_returns_400(self):
        """POST /render 缺少请求体返回 400."""
        client = _make_client()
        resp = client.post("/render")
        assert resp.status_code == 400
        data = resp.json()
        assert data["code"] != 0

    def test_render_with_invalid_json_returns_400(self):
        """POST /render 非法 JSON 返回 400."""
        client = _make_client()
        resp = client.post(
            "/render",
            content="not-json{{",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_get_cached_render_returns_descriptor(self):
        """GET /render/{id} 返回缓存的渲染描述."""
        client = _make_client()
        artifact = _make_artifact()
        # 先渲染
        client.post("/render", json=artifact.to_dict())
        # 再获取缓存
        resp = client.get(f"/render/{artifact.artifact_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["artifact_id"] == artifact.artifact_id

    def test_get_cached_render_not_found_returns_404(self):
        """GET /render/{id} 未渲染过的 ID 返回 404."""
        client = _make_client()
        resp = client.get("/render/art-nonexistent")
        assert resp.status_code == 404

    def test_put_render_updates_descriptor(self):
        """PUT /render/{id} 增量更新渲染."""
        am = ArtifactManager()
        artifact = _make_artifact(artifact_id="art-put-001", title="PUT")
        # 真实 RenderPipeline.update 要求 Artifact 已在管理器中注册
        am.register(artifact)
        client = _make_client(artifact_manager=am)
        client.post("/render", json=artifact.to_dict())
        diff = ArtifactDiff(
            artifact_id=artifact.artifact_id,
            ops=[
                {"op": "replace", "path": "content", "value": "# Updated"},
            ],
            edit_reason="learner edit",
        )
        resp = client.put(
            f"/render/{artifact.artifact_id}", json=diff.to_dict()
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert data["data"]["artifact_id"] == artifact.artifact_id

    def test_put_render_not_cached_returns_404(self):
        """PUT /render/{id} 未渲染过的 ID 返回 404."""
        client = _make_client()
        diff = ArtifactDiff(
            artifact_id="art-nonexistent",
            ops=[],
        )
        resp = client.put("/render/art-nonexistent", json=diff.to_dict())
        assert resp.status_code == 404

    def test_delete_render_destroys_instance(self):
        """DELETE /render/{id} 销毁渲染实例."""
        client = _make_client()
        artifact = _make_artifact()
        client.post("/render", json=artifact.to_dict())
        resp = client.delete(f"/render/{artifact.artifact_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        # 删除后获取应 404
        resp2 = client.get(f"/render/{artifact.artifact_id}")
        assert resp2.status_code == 404

    def test_delete_render_not_cached_is_idempotent(self):
        """DELETE /render/{id} 未渲染过的 ID 幂等返回 200 (与 RenderPipeline 一致)."""
        client = _make_client()
        resp = client.delete("/render/art-nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0

    def test_render_with_query_context(self):
        """POST /render 支持从 query 参数构造 RenderContext."""
        client = _make_client()
        artifact = _make_artifact()
        resp = client.post(
            "/render",
            json=artifact.to_dict(),
            params={"theme": "dark", "learner_mode": "beginner"},
        )
        assert resp.status_code == 200
        desc = resp.json()["data"]
        # TestRenderer 把 theme / mode 写入 config
        assert desc["config"]["theme"] == "dark"
        assert desc["config"]["mode"] == "beginner"


# ============================================================
# 3. Artifact 端点
# ============================================================


@pytest.mark.skipif(not _L7_ROUTER_AVAILABLE, reason="L7Router 未实现")
class TestArtifactEndpoints:
    """Artifact 端点."""

    def test_list_artifacts_returns_list(self):
        """GET /artifacts 返回 Artifact 列表 (分页格式)."""
        am = ArtifactManager()
        am.register(_make_artifact(artifact_id="art-1", title="报告一"))
        am.register(_make_artifact(artifact_id="art-2", title="报告二"))
        client = _make_client(artifact_manager=am)
        resp = client.get("/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert isinstance(data["data"], dict)
        assert data["data"]["total"] == 2
        assert len(data["data"]["items"]) == 2

    def test_list_artifacts_empty(self):
        """GET /artifacts 空管理器返回空列表 (分页格式)."""
        client = _make_client()
        resp = client.get("/artifacts")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_artifacts_filter_by_type(self):
        """GET /artifacts?type=chart 按类型过滤."""
        am = ArtifactManager()
        am.register(
            _make_artifact(
                artifact_id="art-text",
                artifact_type=ArtifactType.TEXT,
                mime="text/vnd.dy3+markdown",
                payload={"content": "hi"},
                title="文本",
            )
        )
        am.register(
            _make_artifact(
                artifact_id="art-chart",
                artifact_type=ArtifactType.CHART,
                mime="application/vnd.dy3.chart+json",
                payload={"chart_type": "bar", "data": [1, 2, 3]},
                title="图表",
            )
        )
        client = _make_client(artifact_manager=am)
        resp = client.get("/artifacts", params={"type": "chart"})
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["type"] == "chart"

    def test_list_artifacts_filter_by_source_agent(self):
        """GET /artifacts?source_agent=A1 按来源 Agent 过滤."""
        am = ArtifactManager()
        am.register(
            _make_artifact(artifact_id="art-a1", source_agent="A1", title="A1")
        )
        am.register(
            _make_artifact(artifact_id="art-a2", source_agent="A2", title="A2")
        )
        client = _make_client(artifact_manager=am)
        resp = client.get("/artifacts", params={"source_agent": "A1"})
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["source_agent"] == "A1"

    def test_get_artifact_detail(self):
        """GET /artifacts/{id} 返回 Artifact 详情."""
        am = ArtifactManager()
        art = _make_artifact(artifact_id="art-detail", title="详情")
        am.register(art)
        client = _make_client(artifact_manager=am)
        resp = client.get(f"/artifacts/{art.artifact_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["artifact_id"] == art.artifact_id
        assert data["title"] == "详情"

    def test_get_artifact_detail_not_found(self):
        """GET /artifacts/{id} 不存在返回 404."""
        client = _make_client()
        resp = client.get("/artifacts/art-missing")
        assert resp.status_code == 404

    def test_get_version_history(self):
        """GET /artifacts/{id}/versions 返回版本历史."""
        am = ArtifactManager()
        art = _make_artifact(artifact_id="art-ver", title="版本")
        am.register(art)
        am.update(art.artifact_id, {"content": "v2"}, edit_reason="update")
        client = _make_client(artifact_manager=am)
        resp = client.get(f"/artifacts/{art.artifact_id}/versions")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, list)
        assert len(data) >= 2

    def test_get_version_history_not_found(self):
        """GET /artifacts/{id}/versions 不存在返回 404."""
        client = _make_client()
        resp = client.get("/artifacts/art-missing/versions")
        assert resp.status_code == 404

    def test_edit_artifact_applies_diff(self):
        """POST /artifacts/{id}/edit 应用编辑差异."""
        am = ArtifactManager()
        art = _make_artifact(artifact_id="art-edit", title="编辑")
        am.register(art)
        client = _make_client(artifact_manager=am)
        diff = ArtifactDiff(
            artifact_id=art.artifact_id,
            ops=[
                {"op": "replace", "path": "content", "value": "# 编辑后"},
            ],
            edit_reason="learner edit",
        )
        resp = client.post(
            f"/artifacts/{art.artifact_id}/edit", json=diff.to_dict()
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        # 版本号应递增
        assert data["version"] >= 2

    def test_edit_artifact_not_found(self):
        """POST /artifacts/{id}/edit 不存在返回 404."""
        client = _make_client()
        diff = ArtifactDiff(artifact_id="art-missing", ops=[])
        resp = client.post("/artifacts/art-missing/edit", json=diff.to_dict())
        assert resp.status_code == 404


# ============================================================
# 4. 注册中心端点
# ============================================================


@pytest.mark.skipif(not _L7_ROUTER_AVAILABLE, reason="L7Router 未实现")
class TestRegistryEndpoints:
    """注册中心端点."""

    def test_get_mime_types(self):
        """GET /registry/mime-types 返回已注册 MIME 列表."""
        client = _make_client()
        resp = client.get("/registry/mime-types")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, list)
        assert "text/vnd.dy3+markdown" in data

    def test_get_renderers(self):
        """GET /registry/renderers 返回渲染器类名列表."""
        client = _make_client()
        resp = client.get("/registry/renderers")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, list)
        assert "TestRenderer" in data

    def test_supports_registered_mime(self):
        """GET /registry/supports/{mime} 已注册返回 supported=True."""
        client = _make_client()
        resp = client.get("/registry/supports/text/vnd.dy3+markdown")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["supported"] is True

    def test_supports_unregistered_mime(self):
        """GET /registry/supports/{mime} 未注册返回 supported=False."""
        client = _make_client()
        resp = client.get("/registry/supports/application/x-unknown")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["supported"] is False


# ============================================================
# 5. 统计端点
# ============================================================


@pytest.mark.skipif(not _L7_ROUTER_AVAILABLE, reason="L7Router 未实现")
class TestStatsEndpoint:
    """GET /stats — 统计信息."""

    def test_stats_returns_200(self):
        """GET /stats 返回 200 和统计信息."""
        am = ArtifactManager()
        am.register(_make_artifact(artifact_id="art-s1", title="s1"))
        client = _make_client(artifact_manager=am)
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "artifacts" in data
        assert data["artifacts"]["total"] >= 1

    def test_stats_includes_pipeline_stats(self):
        """GET /stats 包含 pipeline 统计."""
        client = _make_client()
        resp = client.get("/stats")
        data = resp.json()["data"]
        assert "pipeline" in data


# ============================================================
# 6. L7Router 元信息
# ============================================================


@pytest.mark.skipif(not _L7_ROUTER_AVAILABLE, reason="L7Router 未实现")
class TestRouterMetadata:
    """L7Router 元信息."""

    def test_get_routes_summary_returns_list(self):
        """get_routes_summary 返回路由摘要列表."""
        router = _make_router()
        summary = router.get_routes_summary()
        assert isinstance(summary, list)
        assert len(summary) >= 13
        paths = {item["path"] for item in summary}
        assert "/health" in paths
        assert "/render" in paths
        assert "/artifacts" in paths
        assert "/stats" in paths

    def test_routes_summary_has_methods_and_description(self):
        """每条路由摘要包含 methods 与 description."""
        router = _make_router()
        for item in router.get_routes_summary():
            assert "path" in item
            assert "methods" in item
            assert "description" in item

    def test_create_app_returns_starlette(self):
        """create_app 返回 Starlette 应用实例."""
        from starlette.applications import Starlette

        router = _make_router()
        app = router.create_app()
        assert isinstance(app, Starlette)

    def test_router_with_defaults(self):
        """不传任何依赖时 L7Router 仍可创建应用."""
        router = L7Router()
        app = router.create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_router_with_custom_cors(self):
        """自定义 CORS 源不报错."""
        router = L7Router(cors_origins=["https://example.com"])
        app = router.create_app()
        client = TestClient(app)
        resp = client.get(
            "/health", headers={"Origin": "https://example.com"}
        )
        assert resp.status_code == 200


# ============================================================
# 7. 包 __init__.py 导出
# ============================================================


class TestPackageExports:
    """验证 l7/__init__.py 导出."""

    def test_exceptions_importable(self):
        """L7 异常类可从包顶层导入."""
        from dy3_polaris.l7 import (
            L7Error,
            RendererNotFoundError,
            ArtifactNotFoundError,
            UnsupportedMimeError,
        )
        assert issubclass(RendererNotFoundError, L7Error)
        assert issubclass(ArtifactNotFoundError, L7Error)
        assert issubclass(UnsupportedMimeError, L7Error)

    def test_models_importable(self):
        """模型类可从包顶层导入."""
        from dy3_polaris.l7 import (
            Artifact,
            ArtifactDiff,
            RenderContext,
            RenderDescriptor,
            ArtifactType,
            MIME_TO_TYPE,
            TYPE_TO_MIME,
        )
        assert ArtifactType.TEXT.value == "text"
        assert "text/vnd.dy3+markdown" in MIME_TO_TYPE
        assert ArtifactType.TEXT in TYPE_TO_MIME

    def test_core_importable(self):
        """核心组件可从包顶层导入."""
        from dy3_polaris.l7 import (
            IRenderer,
            RendererRegistry,
            get_registry,
            reset_registry,
            ArtifactManager,
        )
        assert IRenderer is not None
        assert RendererRegistry is not None
        assert ArtifactManager is not None

    def test_l7router_importable(self):
        """L7Router 可从包顶层导入 (条件导出)."""
        if not _L7_ROUTER_AVAILABLE:
            pytest.skip("L7Router 未实现")
        from dy3_polaris.l7 import L7Router as TopL7Router
        assert TopL7Router is L7Router
