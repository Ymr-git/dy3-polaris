"""L7 体验呈现层 — RenderPipeline 单元测试 (TDD).

测试覆盖:
1. 基础渲染: render() 路由、返回值、校验、异常、状态更新、耗时记录
2. 缓存: render() 缓存、get_cached()、force=True 强制重渲染、clear_cache()
3. 增量更新: update() 调用 renderer.update()、异常、缓存更新
4. 生命周期管理: destroy() 幂等、安全销毁
5. 批量渲染: render_batch() 多制品、错误收集、部分结果
6. 视口懒加载 (IntersectionObserver): render_if_visible()、mark_visible/hidden
7. 超时控制: render() timeout_ms 参数、RenderTimeoutError
8. 统计: get_stats() 指标追踪

融合方案:
- React Server Components: 增量更新 + 资源生命周期管理
- IntersectionObserver: 视口可见性驱动的懒加载
- Jupyter nbconvert: 批量渲染管道
- Grafana plugin: 渲染缓存 + TTL 过期
"""

from __future__ import annotations

import time

import pytest

from dy3_polaris.l7.artifact_manager import ArtifactManager
from dy3_polaris.l7.exceptions import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    L7Error,
    RenderContextError,
    RenderTimeoutError,
    UnsupportedMimeError,
)
from dy3_polaris.l7.irenderer import IRenderer
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactDiff,
    ArtifactLifecycleState,
    ArtifactType,
    RenderContext,
    RenderDescriptor,
)
from dy3_polaris.l7.registry import RendererRegistry


# ============================================================
# 延迟导入被测对象
# ============================================================

def _import_pipeline():
    from dy3_polaris.l7.pipeline import RenderPipeline

    return RenderPipeline


# ============================================================
# 测试用具体渲染器
# ============================================================

class TestRenderer(IRenderer):
    """最小可用的具体渲染器 — 跟踪 render/update/destroy 调用."""

    __test__ = False  # pytest 不应将此类作为测试类收集

    _MIME_TYPES = ["text/markdown", "text/plain"]

    def __init__(self) -> None:
        self.render_count = 0
        self.update_count = 0
        self.destroy_count = 0
        self._destroyed = False
        self._last_artifact: Artifact | None = None
        self._last_context: RenderContext | None = None
        self._last_diff: ArtifactDiff | None = None

    def render(self, artifact: Artifact, context: RenderContext) -> RenderDescriptor:
        self.render_count += 1
        self._last_artifact = artifact
        self._last_context = context
        self._destroyed = False  # 重新渲染时重置销毁状态
        return RenderDescriptor(
            artifact_id=artifact.artifact_id,
            mime=artifact.mime,
            html=f"<div class='test-renderer'>{artifact.payload.get('content', '')}</div>",
            config={"theme": context.theme},
            metadata={"renderer": "TestRenderer", "render_count": self.render_count},
        )

    def update(self, diff: ArtifactDiff) -> RenderDescriptor:
        self.update_count += 1
        self._last_diff = diff
        return RenderDescriptor(
            artifact_id=diff.artifact_id,
            mime="text/markdown",
            html=f"<div data-ops='{len(diff.ops)}'>updated</div>",
            metadata={"renderer": "TestRenderer", "update_count": self.update_count},
        )

    def destroy(self) -> None:
        self.destroy_count += 1
        self._destroyed = True

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)


class SlowTestRenderer(IRenderer):
    """模拟慢速渲染器 — 用于超时测试."""

    _MIME_TYPES = ["text/markdown"]
    _DELAY = 0.15  # 150ms

    def render(self, artifact: Artifact, context: RenderContext) -> RenderDescriptor:
        time.sleep(self._DELAY)
        return RenderDescriptor(
            artifact_id=artifact.artifact_id,
            mime=artifact.mime,
            html="<div>slow</div>",
        )

    def update(self, diff: ArtifactDiff) -> RenderDescriptor:
        time.sleep(self._DELAY)
        return RenderDescriptor(
            artifact_id=diff.artifact_id,
            mime="text/markdown",
            html="<div>slow-updated</div>",
        )

    def destroy(self) -> None:
        pass

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)


class ChartTestRenderer(IRenderer):
    """图表渲染器 — 用于多渲染器测试."""

    _MIME_TYPES = ["application/vnd.dy3.chart+json"]

    def __init__(self) -> None:
        self.render_count = 0
        self.destroy_count = 0

    def render(self, artifact: Artifact, context: RenderContext) -> RenderDescriptor:
        self.render_count += 1
        return RenderDescriptor(
            artifact_id=artifact.artifact_id,
            mime=artifact.mime,
            html="<div class='chart'>chart</div>",
            config={"chart_type": artifact.payload.get("chart_type", "bar")},
        )

    def update(self, diff: ArtifactDiff) -> RenderDescriptor:
        return RenderDescriptor(
            artifact_id=diff.artifact_id,
            mime="application/vnd.dy3.chart+json",
            html="<div>chart-updated</div>",
        )

    def destroy(self) -> None:
        self.destroy_count += 1

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def RenderPipeline():
    return _import_pipeline()


@pytest.fixture
def registry():
    """每个测试使用全新的注册表."""
    r = RendererRegistry()
    r.register(TestRenderer())
    r.register(ChartTestRenderer())
    return r


@pytest.fixture
def artifact_manager():
    return ArtifactManager()


@pytest.fixture
def pipeline(RenderPipeline, registry, artifact_manager):
    """创建带 TestRenderer 的 RenderPipeline."""
    return RenderPipeline(registry=registry, artifact_manager=artifact_manager)


@pytest.fixture
def text_artifact() -> Artifact:
    """有效的文本类型 Artifact."""
    return Artifact(
        artifact_id="art-text-001",
        type=ArtifactType.TEXT,
        mime="text/markdown",
        source_agent="agent.explainer",
        payload={"content": "Hello, Polaris!"},
        title="测试文本",
    )


@pytest.fixture
def chart_artifact() -> Artifact:
    """有效的图表类型 Artifact."""
    return Artifact(
        artifact_id="art-chart-001",
        type=ArtifactType.CHART,
        mime="application/vnd.dy3.chart+json",
        source_agent="agent.analyzer",
        payload={"chart_type": "bar", "data": [{"x": 1, "y": 2}]},
        title="测试图表",
    )


@pytest.fixture
def invalid_artifact() -> Artifact:
    """无效的文本类型 Artifact (缺少 content 字段)."""
    return Artifact(
        artifact_id="art-invalid-001",
        type=ArtifactType.TEXT,
        mime="text/markdown",
        payload={"wrong_key": "no content"},
    )


@pytest.fixture
def context() -> RenderContext:
    return RenderContext(theme="dark", learner_mode="beginner")


@pytest.fixture
def sample_diff() -> ArtifactDiff:
    return ArtifactDiff(
        artifact_id="art-text-001",
        ops=[{"op": "replace", "path": "/content", "value": "Updated!"}],
        edit_reason="refinement",
    )


# ============================================================
# 1. 基础渲染
# ============================================================

class TestBasicRendering:
    """验证 render() 基础行为."""

    def test_render_routes_to_correct_renderer(self, pipeline, text_artifact, context):
        """render() 应通过 registry 路由到正确的渲染器."""
        descriptor = pipeline.render(text_artifact, context)
        assert isinstance(descriptor, RenderDescriptor)
        assert descriptor.artifact_id == text_artifact.artifact_id

    def test_render_returns_descriptor_with_correct_artifact_id(
        self, pipeline, text_artifact, context
    ):
        """返回的 RenderDescriptor 应携带正确的 artifact_id."""
        descriptor = pipeline.render(text_artifact, context)
        assert descriptor.artifact_id == "art-text-001"

    def test_render_returns_descriptor_with_correct_mime(
        self, pipeline, text_artifact, context
    ):
        """返回的 RenderDescriptor 应携带正确的 mime."""
        descriptor = pipeline.render(text_artifact, context)
        assert descriptor.mime == "text/markdown"

    def test_render_raises_on_validation_error(
        self, pipeline, invalid_artifact, context
    ):
        """artifact.validate() 失败时应抛出 ArtifactValidationError."""
        with pytest.raises(ArtifactValidationError):
            pipeline.render(invalid_artifact, context)

    def test_render_raises_on_unsupported_mime(self, RenderPipeline, artifact_manager):
        """没有注册对应 MIME 的渲染器时应抛出 UnsupportedMimeError."""
        empty_registry = RendererRegistry()
        pipe = RenderPipeline(registry=empty_registry, artifact_manager=artifact_manager)
        artifact = Artifact(
            artifact_id="art-unknown-001",
            type=ArtifactType.TEXT,
            mime="application/x-unknown",
            payload={"content": "test"},
        )
        with pytest.raises(UnsupportedMimeError):
            pipe.render(artifact, RenderContext())

    def test_render_updates_artifact_state_to_rendered(
        self, pipeline, text_artifact, context, artifact_manager
    ):
        """render() 应通过 ArtifactManager 将 artifact 状态更新为 RENDERED."""
        artifact_manager.register(text_artifact)
        pipeline.render(text_artifact, context)
        assert text_artifact.state == ArtifactLifecycleState.RENDERED
        # 验证 artifact_manager 中的副本也被更新
        managed = artifact_manager.get(text_artifact.artifact_id)
        assert managed.state == ArtifactLifecycleState.RENDERED

    def test_render_records_render_time_ms(self, pipeline, text_artifact, context):
        """render() 应在 descriptor 中记录 render_time_ms."""
        descriptor = pipeline.render(text_artifact, context)
        assert descriptor.render_time_ms >= 0.0
        # 渲染应该非常快 (小于 1 秒)
        assert descriptor.render_time_ms < 1000.0

    def test_render_calls_renderer_render(self, pipeline, text_artifact, context):
        """render() 应实际调用渲染器的 render() 方法."""
        descriptor = pipeline.render(text_artifact, context)
        assert descriptor is not None
        # 通过 metadata 验证渲染器被调用
        assert descriptor.metadata.get("renderer") == "TestRenderer"

    def test_render_returns_render_descriptor_instance(
        self, pipeline, text_artifact, context
    ):
        """render() 返回值应为 RenderDescriptor 实例."""
        descriptor = pipeline.render(text_artifact, context)
        assert isinstance(descriptor, RenderDescriptor)

    def test_render_produces_html(self, pipeline, text_artifact, context):
        """render() 结果应包含 HTML 片段."""
        descriptor = pipeline.render(text_artifact, context)
        assert descriptor.html is not None
        assert len(descriptor.html) > 0

    def test_render_unsupported_mime_is_l7_error(self, RenderPipeline, artifact_manager):
        """UnsupportedMimeError 应为 L7Error 子类."""
        empty_registry = RendererRegistry()
        pipe = RenderPipeline(registry=empty_registry, artifact_manager=artifact_manager)
        artifact = Artifact(
            artifact_id="art-unknown-002",
            type=ArtifactType.TEXT,
            mime="application/x-unknown",
            payload={"content": "test"},
        )
        with pytest.raises(L7Error):
            pipe.render(artifact, RenderContext())


# ============================================================
# 2. 缓存
# ============================================================

class TestCaching:
    """验证缓存行为."""

    def test_render_caches_descriptor(self, pipeline, text_artifact, context):
        """render() 应缓存 RenderDescriptor."""
        descriptor = pipeline.render(text_artifact, context)
        cached = pipeline.get_cached(text_artifact.artifact_id)
        assert cached is not None
        assert cached is descriptor  # 同一对象

    def test_get_cached_returns_cached_descriptor(
        self, pipeline, text_artifact, context
    ):
        """get_cached() 返回缓存的描述符."""
        descriptor = pipeline.render(text_artifact, context)
        cached = pipeline.get_cached(text_artifact.artifact_id)
        assert cached is descriptor

    def test_get_cached_returns_none_for_unknown(self, pipeline):
        """get_cached() 对未知 artifact_id 返回 None."""
        assert pipeline.get_cached("art-nonexistent") is None

    def test_render_returns_cached_without_rerendering(
        self, pipeline, text_artifact, context
    ):
        """第二次 render() 应返回缓存 (不重新渲染)."""
        first = pipeline.render(text_artifact, context)
        second = pipeline.render(text_artifact, context)
        assert first is second  # 同一缓存对象

    def test_render_with_force_rerenders(self, pipeline, text_artifact, context):
        """force=True 应强制重新渲染即使已缓存."""
        first = pipeline.render(text_artifact, context)
        second = pipeline.render(text_artifact, context, force=True)
        assert first is not second  # 不同对象
        assert second.artifact_id == text_artifact.artifact_id

    def test_clear_cache_removes_all(self, pipeline, text_artifact, context):
        """clear_cache() 移除所有缓存."""
        pipeline.render(text_artifact, context)
        assert pipeline.get_cached(text_artifact.artifact_id) is not None
        pipeline.clear_cache()
        assert pipeline.get_cached(text_artifact.artifact_id) is None

    def test_clear_cache_idempotent(self, pipeline):
        """clear_cache() 可多次调用."""
        pipeline.clear_cache()
        pipeline.clear_cache()
        assert pipeline.get_cached("any") is None

    def test_cache_separate_per_artifact(
        self, pipeline, text_artifact, context, chart_artifact
    ):
        """不同 artifact 的缓存独立."""
        pipeline.render(text_artifact, context)
        pipeline.render(chart_artifact, context)
        assert pipeline.get_cached(text_artifact.artifact_id) is not None
        assert pipeline.get_cached(chart_artifact.artifact_id) is not None
        assert (
            pipeline.get_cached(text_artifact.artifact_id)
            is not pipeline.get_cached(chart_artifact.artifact_id)
        )


# ============================================================
# 3. 增量更新
# ============================================================

class TestIncrementalUpdate:
    """验证 update() 增量更新行为."""

    def test_update_calls_renderer_update(
        self, pipeline, text_artifact, context, sample_diff
    ):
        """update() 应调用 renderer.update(diff) 并返回新 descriptor."""
        artifact_manager = pipeline._artifact_manager
        artifact_manager.register(text_artifact)
        pipeline.render(text_artifact, context)

        new_descriptor = pipeline.update(text_artifact.artifact_id, sample_diff)
        assert isinstance(new_descriptor, RenderDescriptor)
        assert new_descriptor.artifact_id == text_artifact.artifact_id

    def test_update_raises_artifact_not_found(self, pipeline, sample_diff):
        """update() 对未注册的 artifact 抛出 ArtifactNotFoundError."""
        with pytest.raises(ArtifactNotFoundError):
            pipeline.update("art-nonexistent", sample_diff)

    def test_update_updates_cache(self, pipeline, text_artifact, context, sample_diff):
        """update() 应用新 descriptor 更新缓存."""
        artifact_manager = pipeline._artifact_manager
        artifact_manager.register(text_artifact)
        pipeline.render(text_artifact, context)

        new_descriptor = pipeline.update(text_artifact.artifact_id, sample_diff)
        cached = pipeline.get_cached(text_artifact.artifact_id)
        assert cached is new_descriptor

    def test_update_raises_render_context_error_without_prior_render(
        self, pipeline, text_artifact, sample_diff
    ):
        """没有先前渲染 (无渲染器实例) 时 update() 抛出 RenderContextError."""
        artifact_manager = pipeline._artifact_manager
        artifact_manager.register(text_artifact)
        # 不先 render, 直接 update
        with pytest.raises(RenderContextError):
            pipeline.update(text_artifact.artifact_id, sample_diff)

    def test_update_returns_new_descriptor_different_from_cached(
        self, pipeline, text_artifact, context, sample_diff
    ):
        """update() 返回的 descriptor 应不同于缓存中的旧 descriptor."""
        artifact_manager = pipeline._artifact_manager
        artifact_manager.register(text_artifact)
        original = pipeline.render(text_artifact, context)
        updated = pipeline.update(text_artifact.artifact_id, sample_diff)
        assert updated is not original
        assert updated.artifact_id == original.artifact_id

    def test_update_records_render_time_ms(
        self, pipeline, text_artifact, context, sample_diff
    ):
        """update() 也应记录 render_time_ms."""
        artifact_manager = pipeline._artifact_manager
        artifact_manager.register(text_artifact)
        pipeline.render(text_artifact, context)
        updated = pipeline.update(text_artifact.artifact_id, sample_diff)
        assert updated.render_time_ms >= 0.0

    def test_update_multiple_times(self, pipeline, text_artifact, context, sample_diff):
        """多次 update() 应持续工作."""
        artifact_manager = pipeline._artifact_manager
        artifact_manager.register(text_artifact)
        pipeline.render(text_artifact, context)
        d1 = pipeline.update(text_artifact.artifact_id, sample_diff)
        d2 = pipeline.update(text_artifact.artifact_id, sample_diff)
        assert d1 is not d2
        assert d2.artifact_id == text_artifact.artifact_id


# ============================================================
# 4. 生命周期管理
# ============================================================

class TestLifecycleManagement:
    """验证 destroy() 生命周期管理."""

    def test_destroy_calls_renderer_destroy(
        self, pipeline, text_artifact, context
    ):
        """destroy() 应调用 renderer.destroy()."""
        pipeline.render(text_artifact, context)
        # 获取存储的渲染器实例
        renderer = pipeline._renderer_instances.get(text_artifact.artifact_id)
        assert renderer is not None
        destroy_count_before = renderer.destroy_count
        pipeline.destroy(text_artifact.artifact_id)
        assert renderer.destroy_count == destroy_count_before + 1

    def test_destroy_removes_cache(self, pipeline, text_artifact, context):
        """destroy() 应移除缓存."""
        pipeline.render(text_artifact, context)
        assert pipeline.get_cached(text_artifact.artifact_id) is not None
        pipeline.destroy(text_artifact.artifact_id)
        assert pipeline.get_cached(text_artifact.artifact_id) is None

    def test_destroy_is_idempotent(self, pipeline, text_artifact, context):
        """destroy() 多次调用不应抛出异常."""
        pipeline.render(text_artifact, context)
        pipeline.destroy(text_artifact.artifact_id)
        pipeline.destroy(text_artifact.artifact_id)  # 不应抛异常

    def test_destroy_unknown_id_is_safe(self, pipeline):
        """destroy() 对未知 ID 不应抛出异常."""
        pipeline.destroy("art-nonexistent")  # 不应抛异常

    def test_destroy_removes_renderer_instance(
        self, pipeline, text_artifact, context
    ):
        """destroy() 应移除渲染器实例."""
        pipeline.render(text_artifact, context)
        assert text_artifact.artifact_id in pipeline._renderer_instances
        pipeline.destroy(text_artifact.artifact_id)
        assert text_artifact.artifact_id not in pipeline._renderer_instances

    def test_render_after_destroy_works(self, pipeline, text_artifact, context):
        """destroy() 后重新 render() 应正常工作."""
        pipeline.render(text_artifact, context)
        pipeline.destroy(text_artifact.artifact_id)
        # 重新渲染
        new_descriptor = pipeline.render(text_artifact, context)
        assert isinstance(new_descriptor, RenderDescriptor)
        assert new_descriptor.artifact_id == text_artifact.artifact_id


# ============================================================
# 5. 批量渲染
# ============================================================

class TestBatchRendering:
    """验证 render_batch() 批量渲染行为."""

    def test_render_batch_renders_multiple(
        self, pipeline, text_artifact, context, chart_artifact
    ):
        """render_batch() 应渲染多个 artifact."""
        results, errors = pipeline.render_batch(
            [text_artifact, chart_artifact], context
        )
        assert len(results) == 2
        assert text_artifact.artifact_id in results
        assert chart_artifact.artifact_id in results
        assert len(errors) == 0

    def test_render_batch_returns_dict_mapping(
        self, pipeline, text_artifact, context, chart_artifact
    ):
        """render_batch() 返回 dict 映射 artifact_id 到 RenderDescriptor."""
        results, errors = pipeline.render_batch(
            [text_artifact, chart_artifact], context
        )
        assert isinstance(results, dict)
        assert isinstance(results[text_artifact.artifact_id], RenderDescriptor)
        assert isinstance(results[chart_artifact.artifact_id], RenderDescriptor)

    def test_render_batch_continues_on_failure(
        self, pipeline, text_artifact, invalid_artifact, context, chart_artifact
    ):
        """render_batch() 在单个失败时继续渲染其他."""
        artifacts = [text_artifact, invalid_artifact, chart_artifact]
        results, errors = pipeline.render_batch(artifacts, context)
        # 有效的两个应成功
        assert len(results) == 2
        assert text_artifact.artifact_id in results
        assert chart_artifact.artifact_id in results
        # 无效的一个应在 errors 中
        assert len(errors) == 1
        assert invalid_artifact.artifact_id in errors

    def test_render_batch_returns_partial_results_with_errors(
        self, pipeline, text_artifact, invalid_artifact, context
    ):
        """render_batch() 返回部分结果和错误字典."""
        results, errors = pipeline.render_batch(
            [text_artifact, invalid_artifact], context
        )
        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[invalid_artifact.artifact_id], str)

    def test_render_batch_empty_list(self, pipeline, context):
        """render_batch() 空列表返回空结果."""
        results, errors = pipeline.render_batch([], context)
        assert len(results) == 0
        assert len(errors) == 0

    def test_render_batch_returns_tuple(self, pipeline, text_artifact, context):
        """render_batch() 返回值为 tuple."""
        result = pipeline.render_batch([text_artifact], context)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_render_batch_error_message_is_string(
        self, pipeline, invalid_artifact, context
    ):
        """render_batch() 错误字典的值为字符串."""
        _, errors = pipeline.render_batch([invalid_artifact], context)
        assert invalid_artifact.artifact_id in errors
        assert isinstance(errors[invalid_artifact.artifact_id], str)


# ============================================================
# 6. 视口懒加载 (IntersectionObserver)
# ============================================================

class TestViewportLazyLoading:
    """验证视口可见性驱动的懒加载."""

    def test_render_if_visible_true_renders(
        self, pipeline, text_artifact, context
    ):
        """is_visible=True 时 render_if_visible() 正常渲染."""
        descriptor = pipeline.render_if_visible(
            text_artifact, context, is_visible=True
        )
        assert descriptor is not None
        assert isinstance(descriptor, RenderDescriptor)
        assert descriptor.artifact_id == text_artifact.artifact_id

    def test_render_if_visible_false_skips(self, pipeline, text_artifact, context):
        """is_visible=False 时 render_if_visible() 跳过渲染返回 None."""
        descriptor = pipeline.render_if_visible(
            text_artifact, context, is_visible=False
        )
        assert descriptor is None

    def test_render_if_visible_false_does_not_cache(
        self, pipeline, text_artifact, context
    ):
        """is_visible=False 时不应缓存."""
        pipeline.render_if_visible(text_artifact, context, is_visible=False)
        assert pipeline.get_cached(text_artifact.artifact_id) is None

    def test_mark_visible_tracks_visibility(self, pipeline):
        """mark_visible() 应将 artifact_id 加入可见集合."""
        pipeline.mark_visible("art-001")
        assert "art-001" in pipeline._visible

    def test_mark_hidden_tracks_visibility(self, pipeline):
        """mark_hidden() 应将 artifact_id 从可见集合移除."""
        pipeline.mark_visible("art-001")
        assert "art-001" in pipeline._visible
        pipeline.mark_hidden("art-001")
        assert "art-001" not in pipeline._visible

    def test_mark_hidden_destroys_renderer_instance(
        self, pipeline, text_artifact, context
    ):
        """mark_hidden() 应销毁渲染器实例 (GPU 资源释放)."""
        pipeline.render(text_artifact, context)
        renderer = pipeline._renderer_instances.get(text_artifact.artifact_id)
        assert renderer is not None
        destroy_count_before = renderer.destroy_count

        pipeline.mark_hidden(text_artifact.artifact_id)

        assert renderer.destroy_count == destroy_count_before + 1
        assert text_artifact.artifact_id not in pipeline._renderer_instances

    def test_mark_hidden_clears_cache(self, pipeline, text_artifact, context):
        """mark_hidden() 应清除缓存."""
        pipeline.render(text_artifact, context)
        assert pipeline.get_cached(text_artifact.artifact_id) is not None
        pipeline.mark_hidden(text_artifact.artifact_id)
        assert pipeline.get_cached(text_artifact.artifact_id) is None

    def test_mark_visible_then_rerender_on_demand(
        self, pipeline, text_artifact, context
    ):
        """mark_visible() 后重新 render_if_visible 应按需重渲染."""
        # 初始渲染
        pipeline.render_if_visible(text_artifact, context, is_visible=True)
        original = pipeline.get_cached(text_artifact.artifact_id)
        assert original is not None

        # 标记隐藏 — 销毁渲染器, 清除缓存
        pipeline.mark_hidden(text_artifact.artifact_id)
        assert pipeline.get_cached(text_artifact.artifact_id) is None

        # 标记可见
        pipeline.mark_visible(text_artifact.artifact_id)

        # 重新渲染 — 按需
        new_descriptor = pipeline.render_if_visible(
            text_artifact, context, is_visible=True
        )
        assert new_descriptor is not None
        assert new_descriptor is not original
        assert new_descriptor.artifact_id == text_artifact.artifact_id

    def test_mark_hidden_idempotent(self, pipeline):
        """mark_hidden() 对未渲染的 artifact 不应抛异常."""
        pipeline.mark_hidden("art-nonexistent")  # 不应抛异常

    def test_mark_visible_idempotent(self, pipeline):
        """mark_visible() 多次调用不应产生重复."""
        pipeline.mark_visible("art-001")
        pipeline.mark_visible("art-001")
        assert len(pipeline._visible) == 1


# ============================================================
# 7. 超时控制
# ============================================================

class TestTimeoutHandling:
    """验证超时控制行为."""

    def test_render_raises_timeout_on_slow_renderer(
        self, RenderPipeline, artifact_manager
    ):
        """慢速渲染器超过 timeout_ms 时应抛出 RenderTimeoutError."""
        registry = RendererRegistry()
        registry.register(SlowTestRenderer())
        pipe = RenderPipeline(registry=registry, artifact_manager=artifact_manager)
        artifact = Artifact(
            artifact_id="art-slow-001",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            payload={"content": "slow"},
        )
        # SlowTestRenderer 延迟 150ms, 设 timeout=50ms
        with pytest.raises(RenderTimeoutError):
            pipe.render(artifact, RenderContext(), timeout_ms=50)

    def test_render_succeeds_within_timeout(
        self, RenderPipeline, artifact_manager
    ):
        """渲染在超时时间内完成时应正常返回."""
        registry = RendererRegistry()
        registry.register(SlowTestRenderer())
        pipe = RenderPipeline(registry=registry, artifact_manager=artifact_manager)
        artifact = Artifact(
            artifact_id="art-slow-002",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            payload={"content": "slow"},
        )
        # 150ms 延迟, 设 timeout=500ms (足够)
        descriptor = pipe.render(artifact, RenderContext(), timeout_ms=500)
        assert isinstance(descriptor, RenderDescriptor)

    def test_render_default_timeout_is_30s(self, RenderPipeline, registry, artifact_manager):
        """render() 默认 timeout_ms 为 30000."""
        pipe = RenderPipeline(registry=registry, artifact_manager=artifact_manager)
        import inspect

        sig = inspect.signature(pipe.render)
        assert sig.parameters["timeout_ms"].default == 30000

    def test_render_timeout_does_not_cache(self, RenderPipeline, artifact_manager):
        """超时后不应缓存结果."""
        registry = RendererRegistry()
        registry.register(SlowTestRenderer())
        pipe = RenderPipeline(registry=registry, artifact_manager=artifact_manager)
        artifact = Artifact(
            artifact_id="art-slow-003",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            payload={"content": "slow"},
        )
        try:
            pipe.render(artifact, RenderContext(), timeout_ms=50)
        except RenderTimeoutError:
            pass
        assert pipe.get_cached(artifact.artifact_id) is None

    def test_render_timeout_is_l7_error(self, RenderPipeline, artifact_manager):
        """RenderTimeoutError 应为 L7Error 子类."""
        registry = RendererRegistry()
        registry.register(SlowTestRenderer())
        pipe = RenderPipeline(registry=registry, artifact_manager=artifact_manager)
        artifact = Artifact(
            artifact_id="art-slow-004",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            payload={"content": "slow"},
        )
        with pytest.raises(L7Error):
            pipe.render(artifact, RenderContext(), timeout_ms=50)


# ============================================================
# 8. 统计
# ============================================================

class TestStatistics:
    """验证 get_stats() 统计行为."""

    def test_get_stats_returns_dict(self, pipeline):
        """get_stats() 应返回字典."""
        stats = pipeline.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_has_required_keys(self, pipeline):
        """get_stats() 应包含必需的键."""
        stats = pipeline.get_stats()
        assert "total_renders" in stats
        assert "cache_size" in stats
        assert "cache_hit_rate" in stats
        assert "avg_render_time_ms" in stats

    def test_total_renders_increments(self, pipeline, text_artifact, context):
        """每次成功渲染应递增 total_renders."""
        pipeline.render(text_artifact, context)
        stats = pipeline.get_stats()
        assert stats["total_renders"] == 1

    def test_total_renders_not_incremented_on_cache_hit(
        self, pipeline, text_artifact, context
    ):
        """缓存命中时 total_renders 不应递增."""
        pipeline.render(text_artifact, context)
        pipeline.render(text_artifact, context)  # cache hit
        stats = pipeline.get_stats()
        assert stats["total_renders"] == 1

    def test_cache_size_reflects_cache(self, pipeline, text_artifact, context):
        """cache_size 应反映当前缓存大小."""
        assert pipeline.get_stats()["cache_size"] == 0
        pipeline.render(text_artifact, context)
        assert pipeline.get_stats()["cache_size"] == 1

    def test_cache_hit_rate_after_hit(self, pipeline, text_artifact, context):
        """缓存命中后 cache_hit_rate 应正确计算."""
        pipeline.render(text_artifact, context)  # miss
        pipeline.render(text_artifact, context)  # hit
        stats = pipeline.get_stats()
        # 1 hit / 2 requests = 0.5
        assert stats["cache_hit_rate"] == pytest.approx(0.5)

    def test_cache_hit_rate_zero_without_hits(self, pipeline, text_artifact, context):
        """无缓存命中时 cache_hit_rate 应为 0."""
        pipeline.render(text_artifact, context)
        stats = pipeline.get_stats()
        assert stats["cache_hit_rate"] == 0.0

    def test_avg_render_time_ms(self, pipeline, text_artifact, context):
        """avg_render_time_ms 应为平均渲染时间."""
        pipeline.render(text_artifact, context)
        stats = pipeline.get_stats()
        assert stats["avg_render_time_ms"] >= 0.0

    def test_stats_tracked_across_multiple_renders(
        self, pipeline, text_artifact, context, chart_artifact
    ):
        """多次渲染后统计应正确追踪."""
        # 渲染两个不同的 artifact
        pipeline.render(text_artifact, context)
        pipeline.render(chart_artifact, context)
        # 第二次渲染 text_artifact (cache hit)
        pipeline.render(text_artifact, context)

        stats = pipeline.get_stats()
        assert stats["total_renders"] == 2  # 2 次实际渲染
        assert stats["cache_size"] == 2
        assert stats["cache_hit_rate"] == pytest.approx(1.0 / 3.0)  # 1 hit / 3 requests
        assert stats["avg_render_time_ms"] >= 0.0

    def test_stats_initial_state(self, pipeline):
        """初始状态统计应为零值."""
        stats = pipeline.get_stats()
        assert stats["total_renders"] == 0
        assert stats["cache_size"] == 0
        assert stats["cache_hit_rate"] == 0.0
        assert stats["avg_render_time_ms"] == 0.0

    def test_clear_cache_updates_cache_size(
        self, pipeline, text_artifact, context
    ):
        """clear_cache() 后 cache_size 应为 0."""
        pipeline.render(text_artifact, context)
        assert pipeline.get_stats()["cache_size"] == 1
        pipeline.clear_cache()
        assert pipeline.get_stats()["cache_size"] == 0


# ============================================================
# 9. 构造函数与默认值
# ============================================================

class TestConstructor:
    """验证构造函数行为."""

    def test_default_constructor(self, RenderPipeline):
        """无参构造应使用全局 registry 和新 ArtifactManager."""
        pipe = RenderPipeline()
        assert pipe._registry is not None
        assert pipe._artifact_manager is not None

    def test_custom_registry(self, RenderPipeline, registry, artifact_manager):
        """自定义 registry 应被使用."""
        pipe = RenderPipeline(registry=registry, artifact_manager=artifact_manager)
        assert pipe._registry is registry

    def test_custom_artifact_manager(self, RenderPipeline, registry, artifact_manager):
        """自定义 artifact_manager 应被使用."""
        pipe = RenderPipeline(registry=registry, artifact_manager=artifact_manager)
        assert pipe._artifact_manager is artifact_manager

    def test_has_lock(self, RenderPipeline):
        """pipeline 应持有线程锁."""
        pipe = RenderPipeline()
        assert hasattr(pipe, "_lock")
        import threading
        assert isinstance(pipe._lock, type(threading.RLock()))


# ============================================================
# 10. 线程安全
# ============================================================

class TestThreadSafety:
    """验证线程安全."""

    def test_concurrent_render(self, pipeline, context):
        """并发渲染不同 artifact 不应出错."""
        import threading

        artifacts = [
            Artifact(
                artifact_id=f"art-concurrent-{i}",
                type=ArtifactType.TEXT,
                mime="text/markdown",
                payload={"content": f"content-{i}"},
            )
            for i in range(10)
        ]

        results: list[RenderDescriptor] = []
        results_lock = threading.Lock()

        def worker(art):
            desc = pipeline.render(art, context)
            with results_lock:
                results.append(desc)

        threads = [threading.Thread(target=worker, args=(a,)) for a in artifacts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(isinstance(r, RenderDescriptor) for r in results)

    def test_concurrent_cache_access(self, pipeline, text_artifact, context):
        """并发缓存访问应稳定."""
        import threading

        pipeline.render(text_artifact, context)

        results: list[RenderDescriptor | None] = []
        results_lock = threading.Lock()

        def worker():
            r = pipeline.get_cached(text_artifact.artifact_id)
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(r is not None for r in results)
