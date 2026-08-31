"""L7 体验呈现层 — RendererRegistry 单元测试 (TDD).

测试覆盖:
1. register(renderer) 自动探测渲染器支持的 MIME 类型并注册
2. get_renderer(mime_type) 返回正确的渲染器
3. get_renderer() 对未知 MIME 抛出 RendererNotFoundError
4. unregister(mime_type) 移除渲染器
5. list_mime_types() 返回所有已注册 MIME 类型
6. is_supported(mime_type) 返回布尔值
7. 支持延迟初始化 (工厂模式)
8. get_renderer_for_artifact(artifact) 按 artifact.mime 路由
9. 多个渲染器可注册不同 MIME 类型
10. 单个渲染器可支持多个 MIME 类型
11. list_renderers() 返回渲染器类名
12. clear() 清空注册表
13. 线程安全 (threading.Lock)
"""

from __future__ import annotations

import threading

import pytest

from dy3_polaris.l7.exceptions import L7Error, RendererNotFoundError
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactDiff,
    ArtifactType,
    RenderContext,
    RenderDescriptor,
)


# ============================================================
# 延迟导入被测对象
# ============================================================

def _import_irenderer():
    from dy3_polaris.l7.irenderer import IRenderer

    return IRenderer


def _import_registry():
    from dy3_polaris.l7.registry import RendererRegistry

    return RendererRegistry


# ============================================================
# 测试用渲染器构造器
# ============================================================

def _make_renderer(base, mime_types, *, name="TestRenderer"):
    """基于 IRenderer 构造一个可配置 MIME 类型的具体渲染器."""

    class _Renderer(base):  # type: ignore[misc, valid-type]
        _MIME_TYPES = list(mime_types)

        def __init__(self) -> None:
            self._destroyed = False
            self._render_count = 0
            self._update_count = 0
            self._factory_call_count = 0

        def render(self, artifact: Artifact, context: RenderContext) -> RenderDescriptor:
            self._render_count += 1
            return RenderDescriptor(
                artifact_id=artifact.artifact_id,
                mime=artifact.mime,
                html=f"<div class='{self.__class__.__name__}'>{artifact.title}</div>",
                metadata={"renderer": self.__class__.__name__},
            )

        def update(self, diff: ArtifactDiff) -> RenderDescriptor:
            self._update_count += 1
            return RenderDescriptor(
                artifact_id=diff.artifact_id,
                html="<div>updated</div>",
            )

        def destroy(self) -> None:
            self._destroyed = True

        def supported_mime_types(self) -> list[str]:
            return list(self._MIME_TYPES)

    _Renderer.__name__ = name
    return _Renderer


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def IRenderer():
    return _import_irenderer()


@pytest.fixture
def RendererRegistry():
    return _import_registry()


@pytest.fixture
def registry(RendererRegistry):
    """每个测试使用全新的注册表."""
    return RendererRegistry()


@pytest.fixture
def MarkdownRenderer(IRenderer):
    return _make_renderer(
        IRenderer, ["text/markdown", "text/plain"], name="MarkdownRenderer"
    )


@pytest.fixture
def ChartRenderer(IRenderer):
    return _make_renderer(
        IRenderer, ["application/vnd.echarts+json"], name="ChartRenderer"
    )


@pytest.fixture
def GraphRenderer(IRenderer):
    return _make_renderer(
        IRenderer, ["application/vnd.vis+json"], name="GraphRenderer"
    )


# ============================================================
# 1. 注册
# ============================================================

class TestRegister:
    """验证 register() 行为."""

    def test_register_adds_renderer(self, registry, MarkdownRenderer):
        """register(renderer) 应将渲染器加入注册表."""
        renderer = MarkdownRenderer()
        registry.register(renderer)
        assert registry.is_supported("text/markdown")
        assert registry.is_supported("text/plain")

    def test_register_auto_detects_mime_types(self, registry, MarkdownRenderer):
        """register 应自动调用 supported_mime_types() 探测 MIME."""
        renderer = MarkdownRenderer()
        registry.register(renderer)
        mimes = registry.list_mime_types()
        assert "text/markdown" in mimes
        assert "text/plain" in mimes

    def test_register_multiple_renderers(self, registry, MarkdownRenderer, ChartRenderer, GraphRenderer):
        """多个渲染器可注册不同 MIME 类型."""
        registry.register(MarkdownRenderer())
        registry.register(ChartRenderer())
        registry.register(GraphRenderer())
        assert len(registry.list_mime_types()) >= 3
        assert registry.is_supported("application/vnd.echarts+json")
        assert registry.is_supported("application/vnd.vis+json")

    def test_register_returns_none_or_renderer(self, registry, MarkdownRenderer):
        """register 返回值不抛异常 (允许 None 或渲染器)."""
        renderer = MarkdownRenderer()
        result = registry.register(renderer)
        # 允许返回 None 或渲染器本身，关键是不抛异常
        assert result is None or result is renderer

    def test_register_single_renderer_multiple_mimes(self, registry, IRenderer):
        """一个渲染器支持多个 MIME 类型时全部注册."""
        MultiRenderer = _make_renderer(
            IRenderer, ["text/html", "text/plain", "text/markdown"], name="MultiRenderer"
        )
        registry.register(MultiRenderer())
        assert registry.is_supported("text/html")
        assert registry.is_supported("text/plain")
        assert registry.is_supported("text/markdown")
        # 同一渲染器实例应能通过任一 MIME 取回
        r1 = registry.get_renderer("text/html")
        r2 = registry.get_renderer("text/plain")
        assert r1 is r2


# ============================================================
# 2. 获取渲染器
# ============================================================

class TestGetRenderer:
    """验证 get_renderer() 行为."""

    def test_get_renderer_returns_correct_instance(self, registry, MarkdownRenderer, ChartRenderer):
        """get_renderer(mime_type) 返回正确的渲染器."""
        md = MarkdownRenderer()
        chart = ChartRenderer()
        registry.register(md)
        registry.register(chart)
        assert registry.get_renderer("text/markdown") is md
        assert registry.get_renderer("application/vnd.echarts+json") is chart

    def test_get_renderer_unknown_raises(self, registry):
        """get_renderer 对未知 MIME 抛出 RendererNotFoundError."""
        with pytest.raises(RendererNotFoundError):
            registry.get_renderer("application/x-unknown")

    def test_get_renderer_unknown_error_carries_mime(self, registry):
        """RendererNotFoundError 应携带 mime_type 属性."""
        with pytest.raises(RendererNotFoundError) as exc_info:
            registry.get_renderer("application/x-unknown")
        assert exc_info.value.mime_type == "application/x-unknown"

    def test_get_renderer_returns_irrenderer_subclass(
        self, registry, MarkdownRenderer, IRenderer
    ):
        """返回的对象应为 IRenderer 子类实例."""
        registry.register(MarkdownRenderer())
        renderer = registry.get_renderer("text/markdown")
        assert isinstance(renderer, IRenderer)

    def test_get_renderer_for_multiple_mimes_same_instance(
        self, registry, MarkdownRenderer
    ):
        """同一渲染器的多个 MIME 返回同一实例."""
        md = MarkdownRenderer()
        registry.register(md)
        assert registry.get_renderer("text/markdown") is md
        assert registry.get_renderer("text/plain") is md


# ============================================================
# 3. 注销
# ============================================================

class TestUnregister:
    """验证 unregister() 行为."""

    def test_unregister_removes_renderer(self, registry, MarkdownRenderer):
        """unregister(mime_type) 移除渲染器."""
        registry.register(MarkdownRenderer())
        registry.unregister("text/markdown")
        assert not registry.is_supported("text/markdown")

    def test_unregister_all_mimes_for_renderer(self, registry, MarkdownRenderer):
        """注销一个 MIME 后，该渲染器的其他 MIME 仍保留 (各自独立注销)."""
        registry.register(MarkdownRenderer())
        registry.unregister("text/markdown")
        # text/plain 仍应可用
        assert registry.is_supported("text/plain")

    def test_get_after_unregister_raises(self, registry, MarkdownRenderer):
        """注销后再 get_renderer 应抛异常."""
        registry.register(MarkdownRenderer())
        registry.unregister("text/markdown")
        with pytest.raises(RendererNotFoundError):
            registry.get_renderer("text/markdown")

    def test_unregister_unknown_is_safe(self, registry):
        """注销不存在的 MIME 不应抛异常 (幂等)."""
        # 应静默返回或返回 None，不抛异常
        result = registry.unregister("application/x-nonexistent")
        assert result is None or result is not None  # 关键是不抛异常


# ============================================================
# 4. 查询方法
# ============================================================

class TestQueryMethods:
    """验证 list_mime_types / is_supported / list_renderers 行为."""

    def test_list_mime_types_empty(self, registry):
        """空注册表 list_mime_types 返回空列表."""
        assert registry.list_mime_types() == []

    def test_list_mime_types_returns_all(self, registry, MarkdownRenderer, ChartRenderer):
        """list_mime_types 返回所有已注册 MIME."""
        registry.register(MarkdownRenderer())
        registry.register(ChartRenderer())
        mimes = registry.list_mime_types()
        assert set(mimes) >= {"text/markdown", "text/plain", "application/vnd.echarts+json"}

    def test_is_supported_true(self, registry, MarkdownRenderer):
        """已注册 MIME 返回 True."""
        registry.register(MarkdownRenderer())
        assert registry.is_supported("text/markdown") is True

    def test_is_supported_false(self, registry):
        """未注册 MIME 返回 False."""
        assert registry.is_supported("application/x-unknown") is False

    def test_list_renderers_returns_class_names(
        self, registry, MarkdownRenderer, ChartRenderer
    ):
        """list_renderers 返回渲染器类名列表."""
        registry.register(MarkdownRenderer())
        registry.register(ChartRenderer())
        names = registry.list_renderers()
        assert isinstance(names, list)
        assert "MarkdownRenderer" in names
        assert "ChartRenderer" in names

    def test_list_renderers_empty(self, registry):
        """空注册表 list_renderers 返回空列表."""
        assert registry.list_renderers() == []


# ============================================================
# 5. 延迟初始化 (工厂模式)
# ============================================================

class TestLazyFactory:
    """验证 register_factory / 延迟初始化行为."""

    def test_register_factory_lazy(self, registry, MarkdownRenderer):
        """工厂注册后不应立即创建实例."""
        created = []

        def factory():
            instance = MarkdownRenderer()
            created.append(instance)
            return instance

        registry.register_factory("text/markdown", factory)
        # 注册工厂时尚未实例化
        assert len(created) == 0

    def test_factory_creates_on_get(self, registry, MarkdownRenderer):
        """首次 get_renderer 触发工厂创建."""
        created = []

        def factory():
            instance = MarkdownRenderer()
            created.append(instance)
            return instance

        registry.register_factory("text/markdown", factory)
        renderer = registry.get_renderer("text/markdown")
        assert len(created) == 1
        assert renderer is created[0]

    def test_factory_caches_instance(self, registry, MarkdownRenderer):
        """工厂创建的实例应被缓存 (再次获取返回同一实例)."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return MarkdownRenderer()

        registry.register_factory("text/markdown", factory)
        r1 = registry.get_renderer("text/markdown")
        r2 = registry.get_renderer("text/markdown")
        assert r1 is r2
        assert call_count == 1

    def test_factory_is_supported_before_instantiation(self, registry, MarkdownRenderer):
        """注册工厂后 is_supported 应返回 True (即便尚未实例化)."""
        registry.register_factory("text/markdown", MarkdownRenderer)
        assert registry.is_supported("text/markdown") is True

    def test_factory_supported_via_list_mime_types(self, registry, MarkdownRenderer):
        """注册工厂后 list_mime_types 应包含该 MIME."""
        registry.register_factory("text/markdown", MarkdownRenderer)
        assert "text/markdown" in registry.list_mime_types()

    def test_factory_unknown_mime_raises(self, registry):
        """仅注册了工厂的 MIME 未注册时 get 仍抛异常."""
        with pytest.raises(RendererNotFoundError):
            registry.get_renderer("application/x-not-registered")


# ============================================================
# 6. 按 Artifact 路由
# ============================================================

class TestGetRendererForArtifact:
    """验证 get_renderer_for_artifact() 行为."""

    def test_routes_by_artifact_mime(self, registry, MarkdownRenderer, ChartRenderer):
        """get_renderer_for_artifact 按 artifact.mime 路由."""
        md = MarkdownRenderer()
        chart = ChartRenderer()
        registry.register(md)
        registry.register(chart)

        md_artifact = Artifact(mime="text/markdown", type=ArtifactType.TEXT)
        chart_artifact = Artifact(
            mime="application/vnd.echarts+json", type=ArtifactType.CHART
        )

        assert registry.get_renderer_for_artifact(md_artifact) is md
        assert registry.get_renderer_for_artifact(chart_artifact) is chart

    def test_routes_unknown_artifact_raises(self, registry):
        """未知 MIME 的 artifact 抛出 RendererNotFoundError."""
        artifact = Artifact(mime="application/x-unknown", type=ArtifactType.TEXT)
        with pytest.raises(RendererNotFoundError):
            registry.get_renderer_for_artifact(artifact)

    def test_get_renderer_for_artifact_returns_irrenderer(
        self, registry, MarkdownRenderer, IRenderer
    ):
        """返回对象应为 IRenderer 子类实例."""
        registry.register(MarkdownRenderer())
        artifact = Artifact(mime="text/markdown", type=ArtifactType.TEXT)
        renderer = registry.get_renderer_for_artifact(artifact)
        assert isinstance(renderer, IRenderer)


# ============================================================
# 7. clear()
# ============================================================

class TestClear:
    """验证 clear() 行为."""

    def test_clear_removes_all(self, registry, MarkdownRenderer, ChartRenderer):
        """clear() 移除所有渲染器."""
        registry.register(MarkdownRenderer())
        registry.register(ChartRenderer())
        assert len(registry.list_mime_types()) > 0
        registry.clear()
        assert registry.list_mime_types() == []
        assert registry.list_renderers() == []

    def test_clear_idempotent(self, registry):
        """clear() 可多次调用."""
        registry.clear()
        registry.clear()
        assert registry.list_mime_types() == []

    def test_get_after_clear_raises(self, registry, MarkdownRenderer):
        """clear 后 get_renderer 抛异常."""
        registry.register(MarkdownRenderer())
        registry.clear()
        with pytest.raises(RendererNotFoundError):
            registry.get_renderer("text/markdown")


# ============================================================
# 8. 线程安全
# ============================================================

class TestThreadSafety:
    """验证线程安全 (threading.Lock)."""

    def test_concurrent_register(self, registry, IRenderer):
        """并发注册不丢失数据."""
        # 为每个线程构造一个支持唯一 MIME 的渲染器类
        def make_class(i):
            return _make_renderer(
                IRenderer, [f"application/x-type-{i}"], name=f"Renderer{i}"
            )

        classes = [make_class(i) for i in range(20)]
        threads = []

        def worker(cls):
            registry.register(cls())

        for cls in classes:
            t = threading.Thread(target=worker, args=(cls,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 20 个唯一 MIME 全部应已注册
        assert len(registry.list_mime_types()) == 20

    def test_concurrent_get_after_register(self, registry, MarkdownRenderer):
        """并发读取已注册渲染器应稳定返回同一实例."""
        md = MarkdownRenderer()
        registry.register(md)
        results: list[object] = []
        results_lock = threading.Lock()

        def worker():
            r = registry.get_renderer("text/markdown")
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(r is md for r in results)

    def test_lock_attribute_exists(self, registry):
        """注册表应持有 threading.Lock 实例."""
        assert hasattr(registry, "_lock")
        assert isinstance(registry._lock, type(threading.Lock()))


# ============================================================
# 9. 异常体系一致性
# ============================================================

class TestExceptionIntegration:
    """验证注册表异常体系与 L7 一致."""

    def test_renderer_not_found_is_l7_error(self, registry):
        """RendererNotFoundError 应为 L7Error 子类."""
        with pytest.raises(L7Error):
            registry.get_renderer("application/x-unknown")

    def test_renderer_not_found_is_renderer_not_found(self, registry):
        """应精确为 RendererNotFoundError."""
        with pytest.raises(RendererNotFoundError):
            registry.get_renderer("application/x-unknown")
