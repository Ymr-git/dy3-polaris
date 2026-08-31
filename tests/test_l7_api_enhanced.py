"""L7 API 增强测试 — 自动注册 + 分页排序 + API 版本前缀.

TDD: 先写测试 → 确认失败 → 实现 → 确认通过.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l7.api.router import L7Router
from dy3_polaris.l7.artifact_manager import ArtifactManager
from dy3_polaris.l7.models import Artifact, ArtifactType, RenderContext, Viewport
from dy3_polaris.l7.registry import RendererRegistry, reset_registry
from dy3_polaris.l7.irenderer import IRenderer


# ============================================================
# 测试用渲染器
# ============================================================

class _SimpleRenderer(IRenderer):
    """简单测试渲染器."""

    _MIME_TYPES = ["text/vnd.dy3+markdown"]

    def render(self, artifact: Artifact, context: RenderContext):
        from dy3_polaris.l7.models import RenderDescriptor
        import time, uuid
        return RenderDescriptor(
            render_id=f"rd-{uuid.uuid4().hex[:12]}",
            artifact_id=artifact.artifact_id,
            mime=artifact.mime,
            html=f"<p>{artifact.payload.get('content', '')}</p>",
            config={},
            assets=[],
            metadata={},
            rendered_at=time.time(),
            render_time_ms=1.0,
        )

    def update(self, diff):
        from dy3_polaris.l7.models import RenderDescriptor
        import time, uuid
        return RenderDescriptor(
            render_id=f"rd-{uuid.uuid4().hex[:12]}",
            artifact_id=diff.artifact_id,
            mime="text/vnd.dy3+markdown",
            html="<p>updated</p>",
            config={},
            assets=[],
            metadata={},
            rendered_at=time.time(),
            render_time_ms=1.0,
        )

    def destroy(self) -> None:
        pass

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)


def _make_router(
    artifact_manager: ArtifactManager | None = None,
    registry: RendererRegistry | None = None,
    api_prefix: str = "",
) -> L7Router:
    """创建测试用 L7Router."""
    am = artifact_manager or ArtifactManager()
    reg = registry or RendererRegistry()
    reg.register(_SimpleRenderer())
    return L7Router(
        artifact_manager=am,
        registry=reg,
        api_prefix=api_prefix,
    )


def _make_text_artifact(title: str = "测试", content: str = "内容") -> dict:
    """创建测试用 Artifact 字典."""
    return {
        "type": "text",
        "mime": "text/vnd.dy3+markdown",
        "title": title,
        "payload": {"content": content},
        "source_agent": "test-agent",
        "session_id": "sess-001",
    }


# ============================================================
# 增强 1: POST /render 自动注册 Artifact
# ============================================================

class TestAutoRegisterOnRender:
    """POST /render 自动将 Artifact 注册到 ArtifactManager."""

    def test_render_auto_registers_artifact(self):
        """POST /render 后 artifact 出现在 ArtifactManager 中."""
        router = _make_router()
        am = router._am
        client = TestClient(router.create_app())

        body = _make_text_artifact(title="自动注册", content="hello")
        resp = client.post("/render", json=body)

        assert resp.status_code == 200
        data = resp.json()["data"]
        artifact_id = data["artifact_id"]

        # Artifact 应该在 ArtifactManager 中
        retrieved = am.get(artifact_id)
        assert retrieved is not None
        assert retrieved.title == "自动注册"

    def test_render_auto_register_then_put_works(self):
        """POST /render 后 PUT /render/{id} 能正常工作（不需要外部预注册）."""
        router = _make_router()
        client = TestClient(router.create_app())

        body = _make_text_artifact(content="初始内容")
        resp = client.post("/render", json=body)
        assert resp.status_code == 200
        artifact_id = resp.json()["data"]["artifact_id"]

        # PUT 更新（依赖 ArtifactManager 中有此 artifact）
        diff_body = {
            "artifact_id": artifact_id,
            "ops": [{"op": "replace", "path": "content", "value": "更新内容"}],
        }
        resp = client.put(f"/render/{artifact_id}", json=diff_body)
        assert resp.status_code == 200

    def test_render_auto_register_idempotent(self):
        """对同一 artifact_id 多次 POST /render 不重复创建版本."""
        router = _make_router()
        am = router._am
        client = TestClient(router.create_app())

        body = _make_text_artifact(content="内容")
        # 使用固定 artifact_id
        body["artifact_id"] = "art-fixed-001"
        client.post("/render", json=body)
        client.post("/render", json=body)

        # ArtifactManager 中只有一个 artifact
        artifacts = am.list_artifacts()
        matching = [a for a in artifacts if a.artifact_id == "art-fixed-001"]
        assert len(matching) == 1


# ============================================================
# 增强 2: GET /artifacts 分页与排序
# ============================================================

class TestPaginationAndSorting:
    """GET /artifacts 支持分页和排序."""

    def _setup_artifacts(self, router: L7Router, count: int = 5):
        """在 ArtifactManager 中创建多个 artifact."""
        am = router._am
        for i in range(count):
            art = Artifact(
                type=ArtifactType.TEXT,
                mime="text/vnd.dy3+markdown",
                title=f"制品-{i}",
                payload={"content": f"内容-{i}"},
                source_agent="test-agent",
                session_id="sess-001",
            )
            am.register(art)
        return am

    def test_default_pagination(self):
        """默认返回第 1 页，每页 20 条."""
        router = _make_router()
        self._setup_artifacts(router, count=3)
        client = TestClient(router.create_app())

        resp = client.get("/artifacts")
        assert resp.status_code == 200
        data = resp.json()["data"]
        # data 应该是包含 items 和分页信息的字典
        assert "items" in data
        assert len(data["items"]) == 3
        assert data["total"] == 3
        assert data["page"] == 1
        assert data["size"] == 20
        assert data["total_pages"] == 1

    def test_custom_page_size(self):
        """自定义 page 和 size 参数."""
        router = _make_router()
        self._setup_artifacts(router, count=5)
        client = TestClient(router.create_app())

        resp = client.get("/artifacts?page=1&size=2")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["size"] == 2
        assert data["total_pages"] == 3

    def test_second_page(self):
        """获取第二页."""
        router = _make_router()
        self._setup_artifacts(router, count=5)
        client = TestClient(router.create_app())

        resp = client.get("/artifacts?page=2&size=2")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["items"]) == 2
        assert data["page"] == 2

    def test_last_page_partial(self):
        """最后一页可能不满."""
        router = _make_router()
        self._setup_artifacts(router, count=5)
        client = TestClient(router.create_app())

        resp = client.get("/artifacts?page=3&size=2")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["items"]) == 1
        assert data["page"] == 3

    def test_empty_page(self):
        """超出范围的页返回空列表."""
        router = _make_router()
        self._setup_artifacts(router, count=3)
        client = TestClient(router.create_app())

        resp = client.get("/artifacts?page=10&size=20")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["items"]) == 0
        assert data["total"] == 3

    def test_sort_by_created_at_desc(self):
        """按创建时间降序排列."""
        router = _make_router()
        self._setup_artifacts(router, count=3)
        client = TestClient(router.create_app())

        resp = client.get("/artifacts?sort=-created_at")
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        # 第一个应该是最新的
        timestamps = [item["created_at"] for item in items]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_sort_by_created_at_asc(self):
        """按创建时间升序排列."""
        router = _make_router()
        self._setup_artifacts(router, count=3)
        client = TestClient(router.create_app())

        resp = client.get("/artifacts?sort=created_at")
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        timestamps = [item["created_at"] for item in items]
        assert timestamps == sorted(timestamps)

    def test_sort_by_title_asc(self):
        """按标题升序排列."""
        router = _make_router()
        self._setup_artifacts(router, count=3)
        client = TestClient(router.create_app())

        resp = client.get("/artifacts?sort=title")
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        titles = [item["title"] for item in items]
        assert titles == sorted(titles)

    def test_pagination_with_filter(self):
        """分页和过滤同时使用."""
        router = _make_router()
        am = router._am
        # 创建不同 source_agent 的 artifact
        for i in range(4):
            art = Artifact(
                type=ArtifactType.TEXT,
                mime="text/vnd.dy3+markdown",
                title=f"制品-{i}",
                payload={"content": f"内容-{i}"},
                source_agent="agent-a",
            )
            am.register(art)
        for i in range(2):
            art = Artifact(
                type=ArtifactType.TEXT,
                mime="text/vnd.dy3+markdown",
                title=f"其他-{i}",
                payload={"content": f"其他内容-{i}"},
                source_agent="agent-b",
            )
            am.register(art)

        client = TestClient(router.create_app())
        resp = client.get("/artifacts?source_agent=agent-a&page=1&size=2")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["items"]) == 2
        assert data["total"] == 4  # agent-a 有 4 个
        assert data["total_pages"] == 2

    def test_no_artifacts_returns_empty_page(self):
        """无 artifact 时返回空分页."""
        router = _make_router()
        client = TestClient(router.create_app())

        resp = client.get("/artifacts")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["items"]) == 0
        assert data["total"] == 0
        assert data["total_pages"] == 0


# ============================================================
# 增强 3: API 版本前缀
# ============================================================

class TestAPIVersioning:
    """所有路由支持 /api/v1/ 版本前缀."""

    def test_default_no_prefix(self):
        """默认无前缀，路由在根路径."""
        router = _make_router()
        client = TestClient(router.create_app())

        resp = client.get("/health")
        assert resp.status_code == 200

    def test_with_api_prefix(self):
        """设置 api_prefix 后路由在 /api/v1/ 下."""
        router = _make_router(api_prefix="/api/v1")
        client = TestClient(router.create_app())

        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["data"]["layer"] == "L7"

    def test_with_api_prefix_health_also_at_root(self):
        """设置 api_prefix 后 /health 仍可在根路径访问."""
        router = _make_router(api_prefix="/api/v1")
        client = TestClient(router.create_app())

        resp = client.get("/health")
        assert resp.status_code == 200

    def test_with_api_prefix_artifacts(self):
        """版本前缀下的 /artifacts 端点."""
        router = _make_router(api_prefix="/api/v1")
        am = router._am
        art = Artifact(
            type=ArtifactType.TEXT,
            mime="text/vnd.dy3+markdown",
            title="测试",
            payload={"content": "内容"},
        )
        am.register(art)

        client = TestClient(router.create_app())
        resp = client.get("/api/v1/artifacts")
        assert resp.status_code == 200

    def test_with_api_prefix_render(self):
        """版本前缀下的 /render 端点."""
        router = _make_router(api_prefix="/api/v1")
        client = TestClient(router.create_app())

        body = _make_text_artifact(content="版本化渲染")
        resp = client.post("/api/v1/render", json=body)
        assert resp.status_code == 200

    def test_with_api_prefix_stats(self):
        """版本前缀下的 /stats 端点."""
        router = _make_router(api_prefix="/api/v1")
        client = TestClient(router.create_app())

        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200

    def test_routes_summary_includes_prefix(self):
        """get_routes_summary 包含前缀信息."""
        router = _make_router(api_prefix="/api/v1")
        summary = router.get_routes_summary()
        assert len(summary) > 0
        # 所有路径应包含 /api/v1 前缀
        for route in summary:
            if route["path"] != "/health":  # health 可能同时在根路径
                assert "/api/v1" in route["path"] or route["path"] == "/health"

    def test_custom_prefix(self):
        """自定义前缀."""
        router = _make_router(api_prefix="/v2")
        client = TestClient(router.create_app())

        resp = client.get("/v2/health")
        assert resp.status_code == 200

    def test_empty_prefix_uses_root(self):
        """空前缀等同无前缀."""
        router = _make_router(api_prefix="")
        client = TestClient(router.create_app())

        resp = client.get("/health")
        assert resp.status_code == 200
