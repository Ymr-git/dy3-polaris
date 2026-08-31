"""L7 体验呈现层 — RenderPipeline 增强测试 (TDD).

针对三项向后兼容的增强编写测试:
1. 缓存 TTL — cache_ttl_seconds / get_cache_age / evict_expired
2. 上下文感知缓存键 — _cache_key(artifact_id, version, context)
3. 真超时 — concurrent.futures 实现真正的渲染超时中断

约束:
- 现有 render() 签名保持不变，仅新增可选参数 (version / cache_ttl_seconds 等).
- 默认行为 (TTL=0 / version=0 / context 不参与键) 与原实现完全一致.
"""

from __future__ import annotations

import inspect
import threading
import time

import pytest

from dy3_polaris.l7.artifact_manager import ArtifactManager
from dy3_polaris.l7.exceptions import L7Error, RenderTimeoutError
from dy3_polaris.l7.irenderer import IRenderer
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactType,
    RenderContext,
    RenderDescriptor,
    Viewport,
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

class CountingRenderer(IRenderer):
    """计数渲染器 — 跟踪 render 调用次数，输出携带上下文信息."""

    __test__ = False

    _MIME_TYPES = ["text/markdown"]

    def __init__(self) -> None:
        self.render_count = 0

    def render(self, artifact: Artifact, context: RenderContext) -> RenderDescriptor:
        self.render_count += 1
        mode = context.learner_mode
        mode_val = mode.value if hasattr(mode, "value") else str(mode)
        return RenderDescriptor(
            artifact_id=artifact.artifact_id,
            mime=artifact.mime,
            html=f"<div>{self.render_count}</div>",
            config={
                "theme": context.theme,
                "learner_mode": mode_val,
                "viewport": (context.viewport.width, context.viewport.height),
                "locale": context.locale,
            },
            metadata={"render_count": self.render_count},
        )

    def update(self, diff) -> RenderDescriptor:  # type: ignore[override]
        return RenderDescriptor(
            artifact_id=diff.artifact_id,
            mime="text/markdown",
            html="<div>updated</div>",
        )

    def destroy(self) -> None:
        pass

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)


class BlockingRenderer(IRenderer):
    """阻塞渲染器 — 在事件上阻塞直到被释放，用于真超时测试.

    使用类级 threading.Event，测试可在 finally 中调用 release() 唤醒
    被遗弃的工作线程，避免线程在会话退出时滞留。
    """

    __test__ = False

    _MIME_TYPES = ["text/markdown"]
    _release: threading.Event = threading.Event()

    def __init__(self) -> None:
        self.render_count = 0

    @classmethod
    def reset(cls) -> None:
        cls._release.clear()

    @classmethod
    def release(cls) -> None:
        cls._release.set()

    def render(self, artifact: Artifact, context: RenderContext) -> RenderDescriptor:
        # 阻塞直到 release 或 5 秒兜底超时
        self._release.wait(timeout=5.0)
        self.render_count += 1
        return RenderDescriptor(
            artifact_id=artifact.artifact_id,
            mime=artifact.mime,
            html="<div>blocked</div>",
        )

    def update(self, diff) -> RenderDescriptor:  # type: ignore[override]
        return RenderDescriptor(
            artifact_id=diff.artifact_id,
            mime="text/markdown",
            html="<div>blocked-updated</div>",
        )

    def destroy(self) -> None:
        pass

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
    """每个测试使用全新的注册表 (注册 CountingRenderer)."""
    r = RendererRegistry()
    r.register(CountingRenderer())
    return r


@pytest.fixture
def pipeline(RenderPipeline, registry):
    return RenderPipeline(registry=registry, artifact_manager=ArtifactManager())


@pytest.fixture
def text_artifact() -> Artifact:
    return Artifact(
        artifact_id="art-enh-001",
        type=ArtifactType.TEXT,
        mime="text/markdown",
        source_agent="agent.explainer",
        payload={"content": "Hello, Polaris!"},
        title="增强测试文本",
    )


@pytest.fixture
def context() -> RenderContext:
    return RenderContext(theme="dark", learner_mode="beginner")


@pytest.fixture(autouse=True)
def _release_blocking_renderer():
    """每个测试结束后释放 BlockingRenderer，避免工作线程滞留."""
    yield
    BlockingRenderer.release()


# ============================================================
# 增强 1: 缓存 TTL
# ============================================================

class TestCacheTTL:
    """验证缓存 TTL 过期行为."""

    def test_ttl_zero_never_expires(self, RenderPipeline, registry, text_artifact, context):
        """TTL=0 时缓存永不过期 (默认行为)."""
        pipe = RenderPipeline(
            registry=registry,
            artifact_manager=ArtifactManager(),
            cache_ttl_seconds=0,
        )
        d1 = pipe.render(text_artifact, context)
        d2 = pipe.render(text_artifact, context)
        assert d1 is d2  # 缓存命中，同一对象

    def test_ttl_expires_after_ttl(self, RenderPipeline, registry, text_artifact, context):
        """TTL=0.3 时缓存 0.4 秒后过期，重新渲染."""
        pipe = RenderPipeline(
            registry=registry,
            artifact_manager=ArtifactManager(),
            cache_ttl_seconds=0.3,
        )
        d1 = pipe.render(text_artifact, context)
        time.sleep(0.4)
        d2 = pipe.render(text_artifact, context)
        assert d1 is not d2  # 过期 -> 重新渲染

    def test_get_cached_returns_none_for_expired(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """get_cached() 对过期缓存返回 None."""
        pipe = RenderPipeline(
            registry=registry,
            artifact_manager=ArtifactManager(),
            cache_ttl_seconds=0.3,
        )
        pipe.render(text_artifact, context)
        assert pipe.get_cached(text_artifact.artifact_id) is not None
        time.sleep(0.4)
        assert pipe.get_cached(text_artifact.artifact_id) is None

    def test_render_rerenders_on_expired(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """render() 对过期缓存重新渲染 (render_count 递增)."""
        pipe = RenderPipeline(
            registry=registry,
            artifact_manager=ArtifactManager(),
            cache_ttl_seconds=0.3,
        )
        pipe.render(text_artifact, context)
        renderer = pipe._renderer_instances[text_artifact.artifact_id]
        before = renderer.render_count
        time.sleep(0.4)
        pipe.render(text_artifact, context)
        assert renderer.render_count == before + 1

    def test_evict_expired_returns_count(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """evict_expired() 清除过期项并返回数量."""
        pipe = RenderPipeline(
            registry=registry,
            artifact_manager=ArtifactManager(),
            cache_ttl_seconds=0.3,
        )
        pipe.render(text_artifact, context)
        time.sleep(0.4)
        n = pipe.evict_expired()
        assert n == 1
        assert pipe.get_cached(text_artifact.artifact_id) is None

    def test_evict_expired_keeps_fresh(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """evict_expired() 保留未过期项."""
        pipe = RenderPipeline(
            registry=registry,
            artifact_manager=ArtifactManager(),
            cache_ttl_seconds=5.0,
        )
        pipe.render(text_artifact, context)
        n = pipe.evict_expired()
        assert n == 0
        assert pipe.get_cached(text_artifact.artifact_id) is not None

    def test_evict_expired_zero_ttl_returns_zero(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """TTL=0 时 evict_expired() 返回 0 (永不过期)."""
        pipe = RenderPipeline(
            registry=registry,
            artifact_manager=ArtifactManager(),
            cache_ttl_seconds=0,
        )
        pipe.render(text_artifact, context)
        assert pipe.evict_expired() == 0
        assert pipe.get_cached(text_artifact.artifact_id) is not None

    def test_get_cache_age_returns_age(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """get_cache_age() 返回缓存年龄 (秒)."""
        pipe = RenderPipeline(
            registry=registry,
            artifact_manager=ArtifactManager(),
            cache_ttl_seconds=5.0,
        )
        pipe.render(text_artifact, context)
        age = pipe.get_cache_age(text_artifact.artifact_id)
        assert age is not None
        assert 0.0 <= age < 1.0

    def test_get_cache_age_none_for_missing(self, RenderPipeline, registry):
        """get_cache_age() 对不存在的缓存返回 None."""
        pipe = RenderPipeline(registry=registry, artifact_manager=ArtifactManager())
        assert pipe.get_cache_age("art-missing") is None

    def test_cache_ttl_seconds_default_is_zero(self, RenderPipeline):
        """__init__ cache_ttl_seconds 默认值为 0."""
        sig = inspect.signature(RenderPipeline.__init__)
        assert sig.parameters["cache_ttl_seconds"].default == 0

    def test_expired_cache_does_not_count_as_hit(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """过期缓存视为未命中，不计入缓存命中率."""
        pipe = RenderPipeline(
            registry=registry,
            artifact_manager=ArtifactManager(),
            cache_ttl_seconds=0.3,
        )
        pipe.render(text_artifact, context)  # miss
        time.sleep(0.4)
        pipe.render(text_artifact, context)  # expired -> miss (re-render)
        stats = pipe.get_stats()
        # 两次均未命中 (第二次因过期而重新渲染)，命中率为 0
        assert stats["total_renders"] == 2
        assert stats["cache_hit_rate"] == 0.0


# ============================================================
# 增强 2: 上下文感知缓存键
# ============================================================

class TestContextAwareCacheKey:
    """验证上下文感知缓存键行为."""

    def test_cache_key_degenerates_with_id_only(self, pipeline):
        """_cache_key 仅 artifact_id 时退化为简单格式."""
        assert pipeline._cache_key("art-1") == "art-1"

    def test_cache_key_includes_version(self, pipeline):
        """_cache_key 含 version 时包含版本号."""
        key = pipeline._cache_key("art-1", version=3)
        assert key.startswith("art-1")
        assert "v3" in key

    def test_cache_key_includes_context_hash(self, pipeline):
        """_cache_key 含 context 时包含上下文哈希 (h 前缀)."""
        ctx = RenderContext(theme="dark")
        key = pipeline._cache_key("art-1", context=ctx)
        assert key.startswith("art-1")
        assert ":h" in key

    def test_different_contexts_different_keys(self, pipeline):
        """不同 context 产生不同缓存键."""
        ctx1 = RenderContext(theme="light")
        ctx2 = RenderContext(theme="dark")
        k1 = pipeline._cache_key("art-1", context=ctx1)
        k2 = pipeline._cache_key("art-1", context=ctx2)
        assert k1 != k2

    def test_same_context_same_key(self, pipeline):
        """等价 context 产生相同缓存键."""
        ctx1 = RenderContext(theme="dark", learner_mode="beginner")
        ctx2 = RenderContext(theme="dark", learner_mode="beginner")
        assert pipeline._cache_key("art-1", context=ctx1) == pipeline._cache_key(
            "art-1", context=ctx2
        )

    def test_context_hash_considers_theme(self, pipeline):
        """context_hash 考虑 theme."""
        ctx1 = RenderContext(theme="light")
        ctx2 = RenderContext(theme="dark")
        assert pipeline._cache_key("art-1", context=ctx1) != pipeline._cache_key(
            "art-1", context=ctx2
        )

    def test_context_hash_considers_learner_mode(self, pipeline):
        """context_hash 考虑 learner_mode."""
        ctx1 = RenderContext(learner_mode="beginner")
        ctx2 = RenderContext(learner_mode="advanced")
        assert pipeline._cache_key("art-1", context=ctx1) != pipeline._cache_key(
            "art-1", context=ctx2
        )

    def test_context_hash_considers_viewport(self, pipeline):
        """context_hash 考虑 viewport 尺寸."""
        ctx1 = RenderContext(viewport=Viewport(width=1280, height=720))
        ctx2 = RenderContext(viewport=Viewport(width=1920, height=1080))
        assert pipeline._cache_key("art-1", context=ctx1) != pipeline._cache_key(
            "art-1", context=ctx2
        )

    def test_context_hash_considers_locale(self, pipeline):
        """context_hash 考虑 locale."""
        ctx1 = RenderContext(locale="zh-CN")
        ctx2 = RenderContext(locale="en-US")
        assert pipeline._cache_key("art-1", context=ctx1) != pipeline._cache_key(
            "art-1", context=ctx2
        )

    def test_render_different_contexts_not_overwrite(
        self, RenderPipeline, registry, text_artifact
    ):
        """不同 context 的同一 artifact 不互相覆盖."""
        pipe = RenderPipeline(registry=registry, artifact_manager=ArtifactManager())
        ctx1 = RenderContext(theme="light")
        ctx2 = RenderContext(theme="dark")
        d1 = pipe.render(text_artifact, ctx1, version=1)
        d2 = pipe.render(text_artifact, ctx2, version=1)
        assert d1 is not d2
        assert pipe.get_cached(text_artifact.artifact_id, version=1, context=ctx1) is d1
        assert pipe.get_cached(text_artifact.artifact_id, version=1, context=ctx2) is d2

    def test_render_different_versions_not_overwrite(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """不同 version 的同一 artifact 不互相覆盖."""
        pipe = RenderPipeline(registry=registry, artifact_manager=ArtifactManager())
        d1 = pipe.render(text_artifact, context, version=1)
        d2 = pipe.render(text_artifact, context, version=2)
        assert d1 is not d2
        assert pipe.get_cached(text_artifact.artifact_id, version=1, context=context) is d1
        assert pipe.get_cached(text_artifact.artifact_id, version=2, context=context) is d2

    def test_default_render_uses_simple_key(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """不传 version/context 时仍使用简单键 (向后兼容)."""
        pipe = RenderPipeline(registry=registry, artifact_manager=ArtifactManager())
        d = pipe.render(text_artifact, context)
        # get_cached 仅传 artifact_id 即可命中
        assert pipe.get_cached(text_artifact.artifact_id) is d
        # 内部缓存键退化为简单 artifact_id
        assert text_artifact.artifact_id in pipe._cache

    def test_cache_key_version_zero_context_none_is_simple(self, pipeline):
        """version=0 且 context=None 退化为简单 artifact_id."""
        assert pipeline._cache_key("art-x", version=0, context=None) == "art-x"

    def test_get_cached_supports_enhanced_key(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """get_cached() 支持 version/context 参数检索增强键缓存."""
        pipe = RenderPipeline(registry=registry, artifact_manager=ArtifactManager())
        d = pipe.render(text_artifact, context, version=1)
        assert pipe.get_cached(text_artifact.artifact_id, version=1, context=context) is d
        # 简单键检索应返回 None (缓存存储在增强键下)
        assert pipe.get_cached(text_artifact.artifact_id) is None


# ============================================================
# 增强 3: 真超时
# ============================================================

class TestRealTimeout:
    """验证 concurrent.futures 实现的真超时中断."""

    def test_render_completes_within_timeout(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """渲染在超时前完成时正常返回."""
        pipe = RenderPipeline(registry=registry, artifact_manager=ArtifactManager())
        d = pipe.render(text_artifact, context, timeout_ms=5000)
        assert isinstance(d, RenderDescriptor)
        assert d.artifact_id == text_artifact.artifact_id

    def test_render_raises_timeout_error(self, RenderPipeline, text_artifact, context):
        """渲染超时时抛出 RenderTimeoutError."""
        BlockingRenderer.reset()
        reg = RendererRegistry()
        reg.register(BlockingRenderer())
        pipe = RenderPipeline(registry=reg, artifact_manager=ArtifactManager())
        try:
            with pytest.raises(RenderTimeoutError):
                pipe.render(text_artifact, context, timeout_ms=100)
        finally:
            BlockingRenderer.release()

    def test_real_timeout_does_not_wait_full_block(
        self, RenderPipeline, text_artifact, context
    ):
        """真超时: 超时后立即返回，不等阻塞渲染完成 (5s)."""
        BlockingRenderer.reset()
        reg = RendererRegistry()
        reg.register(BlockingRenderer())
        pipe = RenderPipeline(registry=reg, artifact_manager=ArtifactManager())
        start = time.time()
        try:
            with pytest.raises(RenderTimeoutError):
                pipe.render(text_artifact, context, timeout_ms=100)
        finally:
            BlockingRenderer.release()
        elapsed = time.time() - start
        # 真超时应在 ~0.1s 返回，远小于阻塞时长 5s
        assert elapsed < 1.0

    def test_timeout_does_not_cache(self, RenderPipeline, text_artifact, context):
        """超时后不缓存结果."""
        BlockingRenderer.reset()
        reg = RendererRegistry()
        reg.register(BlockingRenderer())
        pipe = RenderPipeline(registry=reg, artifact_manager=ArtifactManager())
        try:
            with pytest.raises(RenderTimeoutError):
                pipe.render(text_artifact, context, timeout_ms=100)
        finally:
            BlockingRenderer.release()
        assert pipe.get_cached(text_artifact.artifact_id) is None

    def test_timeout_error_has_timeout_seconds(
        self, RenderPipeline, text_artifact, context
    ):
        """超时异常包含 timeout_seconds 属性."""
        BlockingRenderer.reset()
        reg = RendererRegistry()
        reg.register(BlockingRenderer())
        pipe = RenderPipeline(registry=reg, artifact_manager=ArtifactManager())
        try:
            with pytest.raises(RenderTimeoutError) as ei:
                pipe.render(text_artifact, context, timeout_ms=250)
        finally:
            BlockingRenderer.release()
        assert ei.value.timeout_seconds == pytest.approx(0.25)

    def test_timeout_error_is_l7_error(self, RenderPipeline, text_artifact, context):
        """RenderTimeoutError 仍是 L7Error 子类."""
        BlockingRenderer.reset()
        reg = RendererRegistry()
        reg.register(BlockingRenderer())
        pipe = RenderPipeline(registry=reg, artifact_manager=ArtifactManager())
        try:
            with pytest.raises(L7Error):
                pipe.render(text_artifact, context, timeout_ms=100)
        finally:
            BlockingRenderer.release()

    def test_default_timeout_is_30s(self, RenderPipeline, registry):
        """render() 默认 timeout_ms 仍为 30000."""
        pipe = RenderPipeline(registry=registry, artifact_manager=ArtifactManager())
        sig = inspect.signature(pipe.render)
        assert sig.parameters["timeout_ms"].default == 30000

    def test_fast_render_returns_descriptor(
        self, RenderPipeline, registry, text_artifact, context
    ):
        """快速渲染在默认 30s 超时内正常返回."""
        pipe = RenderPipeline(registry=registry, artifact_manager=ArtifactManager())
        d = pipe.render(text_artifact, context)
        assert isinstance(d, RenderDescriptor)
        assert d.artifact_id == text_artifact.artifact_id

    def test_timeout_does_not_increment_total_renders(
        self, RenderPipeline, text_artifact, context
    ):
        """超时不计入成功渲染次数."""
        BlockingRenderer.reset()
        reg = RendererRegistry()
        reg.register(BlockingRenderer())
        pipe = RenderPipeline(registry=reg, artifact_manager=ArtifactManager())
        try:
            with pytest.raises(RenderTimeoutError):
                pipe.render(text_artifact, context, timeout_ms=100)
        finally:
            BlockingRenderer.release()
        stats = pipe.get_stats()
        assert stats["total_renders"] == 0
