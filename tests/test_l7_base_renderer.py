"""L7 体验呈现层 — BaseRenderer / FallbackRenderer 单元测试 (TDD).

测试覆盖:
1. BaseRenderer 是抽象类，不能直接实例化
2. BaseRenderer 子类必须实现 do_render
3. BaseRenderer.render() 执行模板方法流程
4. BaseRenderer._validate_artifact() 对 None/空 payload 抛 ArtifactValidationError
5. BaseRenderer._preprocess / _postprocess 默认行为
6. BaseRenderer._build_descriptor 返回正确的 RenderDescriptor
7. BaseRenderer.update() 默认抛 NotImplementedError
8. BaseRenderer.destroy() 默认空操作
9. BaseRenderer.supported_mime_types() 返回 _MIME_TYPES 副本
10. FallbackRenderer 可以实例化
11. FallbackRenderer.do_render() 返回 HTML 字符串
12. FallbackRenderer.render() 完整流程
13. FallbackRenderer 处理各种类型 Artifact
14. FallbackRenderer.supported_mime_types() 返回空列表
15. FallbackRenderer 的 HTML 输出包含 title 和 payload 内容
"""

from __future__ import annotations

import inspect
from abc import ABC

import pytest

from dy3_polaris.l7.exceptions import ArtifactValidationError
from dy3_polaris.l7.irenderer import IRenderer
from dy3_polaris.l7.models import (
    Artifact,
    ArtifactDiff,
    ArtifactType,
    RenderContext,
    RenderDescriptor,
)


# ============================================================
# 被测对象延迟导入 — 让模块缺失时报错更清晰
# ============================================================

def _import_base_renderer():
    from dy3_polaris.l7.base_renderer import BaseRenderer

    return BaseRenderer


def _import_fallback_renderer():
    from dy3_polaris.l7.base_renderer import FallbackRenderer

    return FallbackRenderer


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def BaseRenderer():
    return _import_base_renderer()


@pytest.fixture
def FallbackRenderer():
    return _import_fallback_renderer()


@pytest.fixture
def sample_artifact() -> Artifact:
    return Artifact(
        artifact_id="art-base-001",
        type=ArtifactType.TEXT,
        mime="text/markdown",
        source_agent="agent-explainer",
        title="示例文本制品",
        payload={"content": "Hello, Polaris!"},
    )


@pytest.fixture
def sample_context() -> RenderContext:
    return RenderContext(theme="dark", learner_mode="beginner")


@pytest.fixture
def sample_diff() -> ArtifactDiff:
    return ArtifactDiff(
        artifact_id="art-base-001",
        ops=[{"op": "replace", "path": "/content", "value": "Updated!"}],
        edit_reason="refinement",
    )


def _make_concrete_base_renderer(BaseRenderer):
    """基于 BaseRenderer 动态构造一个完整实现的具体渲染器."""

    class ConcreteBaseRenderer(BaseRenderer):
        """完整实现 do_render 的具体渲染器 (继承 BaseRenderer)."""

        _MIME_TYPES = ["text/markdown", "text/plain"]

        def __init__(self) -> None:
            self.preprocess_calls = 0
            self.postprocess_calls = 0
            self.do_render_calls = 0
            self.validate_calls = 0

        def do_render(self, artifact: Artifact, context: RenderContext) -> str:
            self.do_render_calls += 1
            content = artifact.payload.get("content", "")
            return f"<div>{content}</div>"

        # 覆盖钩子以追踪调用
        def _preprocess(self, artifact: Artifact, context: RenderContext) -> Artifact:
            self.preprocess_calls += 1
            return artifact

        def _postprocess(self, html: str, context: RenderContext) -> str:
            self.postprocess_calls += 1
            return html

        def _validate_artifact(self, artifact: Artifact) -> None:
            self.validate_calls += 1
            super()._validate_artifact(artifact)

    return ConcreteBaseRenderer


def _make_incomplete_base_renderer(BaseRenderer):
    """构造一个缺少 do_render() 的不完整渲染器 (用于验证抽象强制)."""

    class IncompleteBaseRenderer(BaseRenderer):
        # 故意不实现 do_render()
        pass

    return IncompleteBaseRenderer


# ============================================================
# 1. BaseRenderer 抽象类结构
# ============================================================

class TestBaseRendererAbstraction:
    """验证 BaseRenderer 的抽象基类结构."""

    def test_is_abc_subclass(self, BaseRenderer):
        """BaseRenderer 应继承自 abc.ABC."""
        assert issubclass(BaseRenderer, ABC)

    def test_inherits_irenderer(self, BaseRenderer):
        """BaseRenderer 应继承 IRenderer."""
        assert issubclass(BaseRenderer, IRenderer)

    def test_cannot_instantiate_directly(self, BaseRenderer):
        """BaseRenderer 是抽象类，直接实例化应抛出 TypeError."""
        with pytest.raises(TypeError):
            BaseRenderer()

    def test_do_render_is_abstract(self, BaseRenderer):
        """do_render 应为抽象方法 (BaseRenderer 唯一抽象方法)."""
        assert "do_render" in BaseRenderer.__abstractmethods__

    def test_do_render_is_decorated_abstract(self, BaseRenderer):
        """do_render 方法应被 @abstractmethod 装饰."""
        method = getattr(BaseRenderer, "do_render")
        assert getattr(method, "__isabstractmethod__", False), (
            "do_render 应为抽象方法"
        )

    def test_render_is_not_abstract(self, BaseRenderer):
        """render 应已由 BaseRenderer 实现 (不再是抽象方法)."""
        assert "render" not in BaseRenderer.__abstractmethods__

    def test_update_is_not_abstract(self, BaseRenderer):
        """update 应已由 BaseRenderer 实现 (不再是抽象方法)."""
        assert "update" not in BaseRenderer.__abstractmethods__

    def test_destroy_is_not_abstract(self, BaseRenderer):
        """destroy 应已由 BaseRenderer 实现 (不再是抽象方法)."""
        assert "destroy" not in BaseRenderer.__abstractmethods__

    def test_supported_mime_types_is_not_abstract(self, BaseRenderer):
        """supported_mime_types 应已由 BaseRenderer 实现 (不再是抽象方法)."""
        assert "supported_mime_types" not in BaseRenderer.__abstractmethods__

    def test_render_signature(self, BaseRenderer):
        """render 签名: (artifact, context) -> RenderDescriptor."""
        sig = inspect.signature(BaseRenderer.render)
        params = list(sig.parameters)
        assert "artifact" in params
        assert "context" in params

    def test_do_render_signature(self, BaseRenderer):
        """do_render 签名: (artifact, context) -> str."""
        sig = inspect.signature(BaseRenderer.do_render)
        params = list(sig.parameters)
        assert "artifact" in params
        assert "context" in params


class TestBaseRendererSubclassRequirement:
    """验证 BaseRenderer 子类必须实现 do_render."""

    def test_incomplete_subclass_cannot_instantiate(self, BaseRenderer):
        """缺少 do_render 的子类仍无法实例化 (TypeError)."""
        Incomplete = _make_incomplete_base_renderer(BaseRenderer)
        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_can_instantiate(self, BaseRenderer):
        """完整实现 do_render 的子类可实例化."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        assert renderer is not None
        assert isinstance(renderer, Concrete)


# ============================================================
# 2. BaseRenderer.render() 模板方法流程
# ============================================================

class TestBaseRendererTemplateMethod:
    """验证 BaseRenderer.render() 执行模板方法流程."""

    def test_render_returns_render_descriptor(
        self, BaseRenderer, sample_artifact, sample_context
    ):
        """render() 应返回 RenderDescriptor 实例."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        result = renderer.render(sample_artifact, sample_context)
        assert isinstance(result, RenderDescriptor)

    def test_render_invokes_validate_preprocess_do_render_postprocess(
        self, BaseRenderer, sample_artifact, sample_context
    ):
        """render() 应按序调用: _validate_artifact → _preprocess → do_render → _postprocess."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        renderer.render(sample_artifact, sample_context)
        assert renderer.validate_calls == 1
        assert renderer.preprocess_calls == 1
        assert renderer.do_render_calls == 1
        assert renderer.postprocess_calls == 1

    def test_render_carries_artifact_id(
        self, BaseRenderer, sample_artifact, sample_context
    ):
        """渲染结果应携带来源 artifact_id."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        result = renderer.render(sample_artifact, sample_context)
        assert result.artifact_id == sample_artifact.artifact_id

    def test_render_carries_mime(
        self, BaseRenderer, sample_artifact, sample_context
    ):
        """渲染结果应携带 mime 类型."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        result = renderer.render(sample_artifact, sample_context)
        assert result.mime == sample_artifact.mime

    def test_render_produces_html(
        self, BaseRenderer, sample_artifact, sample_context
    ):
        """渲染结果应包含 HTML 片段 (do_render 的输出)."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        result = renderer.render(sample_artifact, sample_context)
        assert isinstance(result.html, str)
        assert "Hello, Polaris!" in result.html

    def test_render_validation_short_circuits(
        self, BaseRenderer, sample_context
    ):
        """校验失败时，应跳过后续 _preprocess / do_render / _postprocess."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        # 空 payload 触发校验失败
        bad_artifact = Artifact(
            artifact_id="art-empty",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            payload={},
        )
        with pytest.raises(ArtifactValidationError):
            renderer.render(bad_artifact, sample_context)
        # 校验后立即抛出，后续步骤不应执行
        assert renderer.do_render_calls == 0
        assert renderer.preprocess_calls == 0
        assert renderer.postprocess_calls == 0

    def test_render_passes_preprocessed_artifact_to_do_render(
        self, BaseRenderer, sample_artifact, sample_context
    ):
        """_preprocess 的返回值应被传入 do_render."""
        # 构造一个 _preprocess 修改 artifact 的渲染器
        class PreprocessRenderer(BaseRenderer):
            _MIME_TYPES = ["text/markdown"]

            def __init__(self) -> None:
                self.received_artifact = None

            def do_render(self, artifact: Artifact, context: RenderContext) -> str:
                self.received_artifact = artifact
                return f"<div>{artifact.payload.get('content', '')}</div>"

            def _preprocess(self, artifact: Artifact, context: RenderContext) -> Artifact:
                # 返回一个新的 artifact，payload 中 content 被改写
                return artifact.model_copy(
                    update={"payload": {"content": "PREPROCESSED"}}
                )

        renderer = PreprocessRenderer()
        result = renderer.render(sample_artifact, sample_context)
        # do_render 收到的是预处理后的 artifact
        assert renderer.received_artifact.payload.get("content") == "PREPROCESSED"
        assert "PREPROCESSED" in result.html

    def test_render_applies_postprocess(
        self, BaseRenderer, sample_artifact, sample_context
    ):
        """_postprocess 的返回值应作为最终 HTML."""
        class PostprocessRenderer(BaseRenderer):
            _MIME_TYPES = ["text/markdown"]

            def do_render(self, artifact: Artifact, context: RenderContext) -> str:
                return "<div>RAW</div>"

            def _postprocess(self, html: str, context: RenderContext) -> str:
                return f"<wrapper>{html}</wrapper>"

        renderer = PostprocessRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert result.html == "<wrapper><div>RAW</div></wrapper>"


# ============================================================
# 3. BaseRenderer._validate_artifact()
# ============================================================

class TestBaseRendererValidateArtifact:
    """验证 BaseRenderer._validate_artifact() 校验逻辑."""

    def test_validate_raises_on_none_artifact(self, BaseRenderer):
        """artifact 为 None 时应抛出 ArtifactValidationError."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        with pytest.raises(ArtifactValidationError) as exc_info:
            renderer._validate_artifact(None)
        assert exc_info.value.field == "artifact"

    def test_validate_raises_on_empty_payload(self, BaseRenderer):
        """payload 为空字典时应抛出 ArtifactValidationError."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        empty_artifact = Artifact(
            artifact_id="art-empty",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            payload={},
        )
        with pytest.raises(ArtifactValidationError) as exc_info:
            renderer._validate_artifact(empty_artifact)
        assert exc_info.value.field == "payload"

    def test_validate_passes_on_valid_artifact(self, BaseRenderer, sample_artifact):
        """有效 artifact (payload 非空) 校验通过 (不抛异常)."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        # 不应抛出异常
        renderer._validate_artifact(sample_artifact)

    def test_validation_error_propagates_from_render(
        self, BaseRenderer, sample_context
    ):
        """render() 中校验失败时应抛出 ArtifactValidationError."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        empty_artifact = Artifact(
            artifact_id="art-empty",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            payload={},
        )
        with pytest.raises(ArtifactValidationError):
            renderer.render(empty_artifact, sample_context)


# ============================================================
# 4. BaseRenderer._preprocess / _postprocess 默认行为
# ============================================================

class TestBaseRendererHooks:
    """验证 _preprocess / _postprocess 钩子默认行为."""

    def test_preprocess_default_returns_same_artifact(self, BaseRenderer, sample_artifact, sample_context):
        """默认 _preprocess 返回原 artifact (不修改)."""
        # 使用一个不覆盖钩子的最小子类
        class MinimalRenderer(BaseRenderer):
            _MIME_TYPES = ["text/markdown"]

            def do_render(self, artifact: Artifact, context: RenderContext) -> str:
                return "<div></div>"

        renderer = MinimalRenderer()
        result = renderer._preprocess(sample_artifact, sample_context)
        assert result is sample_artifact

    def test_postprocess_default_returns_same_html(self, BaseRenderer, sample_context):
        """默认 _postprocess 返回原 html 字符串."""
        class MinimalRenderer(BaseRenderer):
            _MIME_TYPES = ["text/markdown"]

            def do_render(self, artifact: Artifact, context: RenderContext) -> str:
                return "<div></div>"

        renderer = MinimalRenderer()
        original_html = "<div>ORIGINAL</div>"
        result = renderer._postprocess(original_html, sample_context)
        assert result is original_html


# ============================================================
# 5. BaseRenderer._build_descriptor()
# ============================================================

class TestBaseRendererBuildDescriptor:
    """验证 BaseRenderer._build_descriptor() 返回正确的 RenderDescriptor."""

    def test_build_descriptor_returns_render_descriptor(
        self, BaseRenderer, sample_artifact
    ):
        """_build_descriptor 返回 RenderDescriptor 实例."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        html = "<div>test</div>"
        descriptor = renderer._build_descriptor(sample_artifact, html)
        assert isinstance(descriptor, RenderDescriptor)

    def test_build_descriptor_carries_artifact_id(
        self, BaseRenderer, sample_artifact
    ):
        """描述符应携带 artifact_id."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        descriptor = renderer._build_descriptor(sample_artifact, "<div></div>")
        assert descriptor.artifact_id == sample_artifact.artifact_id

    def test_build_descriptor_carries_mime(self, BaseRenderer, sample_artifact):
        """描述符应携带 mime 类型."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        descriptor = renderer._build_descriptor(sample_artifact, "<div></div>")
        assert descriptor.mime == sample_artifact.mime

    def test_build_descriptor_carries_html(self, BaseRenderer, sample_artifact):
        """描述符应携带 html 内容."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        html = "<div>CONTENT</div>"
        descriptor = renderer._build_descriptor(sample_artifact, html)
        assert descriptor.html == html

    def test_build_descriptor_render_id_prefix(self, BaseRenderer, sample_artifact):
        """render_id 应以 'rd-' 前缀开头."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        descriptor = renderer._build_descriptor(sample_artifact, "<div></div>")
        assert descriptor.render_id.startswith("rd-")

    def test_build_descriptor_render_id_unique(self, BaseRenderer, sample_artifact):
        """每次构建的 render_id 应唯一."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        d1 = renderer._build_descriptor(sample_artifact, "<div></div>")
        d2 = renderer._build_descriptor(sample_artifact, "<div></div>")
        assert d1.render_id != d2.render_id

    def test_build_descriptor_metadata_contains_renderer_name(
        self, BaseRenderer, sample_artifact
    ):
        """metadata 应包含 'renderer' 字段，值为类名."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        descriptor = renderer._build_descriptor(sample_artifact, "<div></div>")
        assert "renderer" in descriptor.metadata
        assert descriptor.metadata["renderer"] == "ConcreteBaseRenderer"

    def test_build_descriptor_rendered_at_is_float(
        self, BaseRenderer, sample_artifact
    ):
        """rendered_at 应为浮点时间戳."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        descriptor = renderer._build_descriptor(sample_artifact, "<div></div>")
        assert isinstance(descriptor.rendered_at, float)
        assert descriptor.rendered_at > 0

    def test_build_descriptor_render_time_ms(self, BaseRenderer, sample_artifact):
        """render_time_ms 应为非负浮点数 (默认 0.0)."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        descriptor = renderer._build_descriptor(sample_artifact, "<div></div>")
        assert isinstance(descriptor.render_time_ms, float)
        assert descriptor.render_time_ms >= 0.0


# ============================================================
# 6. BaseRenderer.update() 默认行为
# ============================================================

class TestBaseRendererUpdate:
    """验证 BaseRenderer.update() 默认抛出 NotImplementedError."""

    def test_update_raises_not_implemented(
        self, BaseRenderer, sample_diff
    ):
        """默认 update() 应抛出 NotImplementedError."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        with pytest.raises(NotImplementedError):
            renderer.update(sample_diff)

    def test_update_error_message_contains_class_name(
        self, BaseRenderer, sample_diff
    ):
        """NotImplementedError 消息应包含类名."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        with pytest.raises(NotImplementedError) as exc_info:
            renderer.update(sample_diff)
        assert "ConcreteBaseRenderer" in str(exc_info.value)

    def test_subclass_can_override_update(self, BaseRenderer, sample_diff):
        """子类可覆盖 update() 提供增量更新实现."""
        class UpdatableRenderer(BaseRenderer):
            _MIME_TYPES = ["text/markdown"]

            def do_render(self, artifact: Artifact, context: RenderContext) -> str:
                return "<div></div>"

            def update(self, diff: ArtifactDiff) -> RenderDescriptor:
                return RenderDescriptor(
                    artifact_id=diff.artifact_id,
                    mime="text/markdown",
                    html="<div>UPDATED</div>",
                )

        renderer = UpdatableRenderer()
        result = renderer.update(sample_diff)
        assert isinstance(result, RenderDescriptor)
        assert result.artifact_id == sample_diff.artifact_id


# ============================================================
# 7. BaseRenderer.destroy() 默认行为
# ============================================================

class TestBaseRendererDestroy:
    """验证 BaseRenderer.destroy() 默认空操作."""

    def test_destroy_returns_none(self, BaseRenderer):
        """默认 destroy() 返回 None."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        result = renderer.destroy()
        assert result is None

    def test_destroy_does_not_raise(self, BaseRenderer):
        """默认 destroy() 不抛异常."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        # 不应抛出异常
        renderer.destroy()

    def test_destroy_idempotent(self, BaseRenderer):
        """destroy() 可被多次调用 (幂等)."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        renderer.destroy()
        renderer.destroy()  # 不应抛异常

    def test_subclass_can_override_destroy(self, BaseRenderer):
        """子类可覆盖 destroy() 提供资源清理."""
        class CleanupRenderer(BaseRenderer):
            _MIME_TYPES = ["text/markdown"]

            def __init__(self) -> None:
                self._destroyed = False

            def do_render(self, artifact: Artifact, context: RenderContext) -> str:
                return "<div></div>"

            def destroy(self) -> None:
                self._destroyed = True

        renderer = CleanupRenderer()
        assert renderer._destroyed is False
        renderer.destroy()
        assert renderer._destroyed is True


# ============================================================
# 8. BaseRenderer.supported_mime_types()
# ============================================================

class TestBaseRendererSupportedMimeTypes:
    """验证 BaseRenderer.supported_mime_types() 返回 _MIME_TYPES 副本."""

    def test_returns_list(self, BaseRenderer):
        """supported_mime_types() 返回 list."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        mimes = renderer.supported_mime_types()
        assert isinstance(mimes, list)

    def test_returns_expected_mimes(self, BaseRenderer):
        """返回预期的 MIME 类型 (来自 _MIME_TYPES)."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        mimes = renderer.supported_mime_types()
        assert "text/markdown" in mimes
        assert "text/plain" in mimes

    def test_returns_independent_copy(self, BaseRenderer):
        """返回值应为独立副本 (修改不影响 _MIME_TYPES)."""
        Concrete = _make_concrete_base_renderer(BaseRenderer)
        renderer = Concrete()
        mimes1 = renderer.supported_mime_types()
        mimes1.append("application/x-evil")
        mimes2 = renderer.supported_mime_types()
        assert "application/x-evil" not in mimes2

    def test_default_mime_types_empty(self, BaseRenderer):
        """_MIME_TYPES 默认值为空列表."""
        # 直接检查类属性
        assert BaseRenderer._MIME_TYPES == []


# ============================================================
# 9. FallbackRenderer 实例化与结构
# ============================================================

class TestFallbackRendererInstantiation:
    """验证 FallbackRenderer 可以实例化."""

    def test_can_instantiate(self, FallbackRenderer):
        """FallbackRenderer 可实例化."""
        renderer = FallbackRenderer()
        assert renderer is not None

    def test_is_base_renderer_subclass(self, FallbackRenderer, BaseRenderer):
        """FallbackRenderer 应继承 BaseRenderer."""
        assert issubclass(FallbackRenderer, BaseRenderer)

    def test_is_irenderer_subclass(self, FallbackRenderer):
        """FallbackRenderer 应继承 IRenderer (间接)."""
        assert issubclass(FallbackRenderer, IRenderer)

    def test_do_render_is_not_abstract(self, FallbackRenderer):
        """FallbackRenderer 已实现 do_render (不再是抽象方法)."""
        assert "do_render" not in FallbackRenderer.__abstractmethods__


# ============================================================
# 10. FallbackRenderer.do_render()
# ============================================================

class TestFallbackRendererDoRender:
    """验证 FallbackRenderer.do_render() 返回 HTML 字符串."""

    def test_returns_html_string(self, FallbackRenderer, sample_artifact, sample_context):
        """do_render() 返回字符串 (HTML)."""
        renderer = FallbackRenderer()
        html = renderer.do_render(sample_artifact, sample_context)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_html_contains_fallback_class(self, FallbackRenderer, sample_artifact, sample_context):
        """HTML 应包含 fallback-renderer CSS 类."""
        renderer = FallbackRenderer()
        html = renderer.do_render(sample_artifact, sample_context)
        assert "fallback-renderer" in html

    def test_html_contains_pre_tag(self, FallbackRenderer, sample_artifact, sample_context):
        """HTML 应包含 <pre> 标签包裹 payload."""
        renderer = FallbackRenderer()
        html = renderer.do_render(sample_artifact, sample_context)
        assert "<pre>" in html
        assert "</pre>" in html

    def test_html_contains_title(self, FallbackRenderer, sample_context):
        """HTML 应包含 artifact 的 title."""
        renderer = FallbackRenderer()
        artifact = Artifact(
            artifact_id="art-title-001",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            title="我的标题",
            payload={"content": "内容"},
        )
        html = renderer.do_render(artifact, sample_context)
        assert "我的标题" in html

    def test_html_contains_default_title_when_empty(self, FallbackRenderer, sample_context):
        """title 为空时，HTML 应包含默认标题 '未命名制品'."""
        renderer = FallbackRenderer()
        artifact = Artifact(
            artifact_id="art-notitle-001",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            title="",
            payload={"content": "内容"},
        )
        html = renderer.do_render(artifact, sample_context)
        assert "未命名制品" in html

    def test_html_contains_payload_content(self, FallbackRenderer, sample_artifact, sample_context):
        """HTML 应包含 payload 的 JSON 序列化内容."""
        renderer = FallbackRenderer()
        html = renderer.do_render(sample_artifact, sample_context)
        # payload 中包含 content 字段
        assert "content" in html
        assert "Hello, Polaris!" in html


# ============================================================
# 11. FallbackRenderer.render() 完整流程
# ============================================================

class TestFallbackRendererRender:
    """验证 FallbackRenderer.render() 完整流程."""

    def test_render_returns_render_descriptor(
        self, FallbackRenderer, sample_artifact, sample_context
    ):
        """render() 应返回 RenderDescriptor 实例."""
        renderer = FallbackRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert isinstance(result, RenderDescriptor)

    def test_render_carries_artifact_id(
        self, FallbackRenderer, sample_artifact, sample_context
    ):
        """渲染结果应携带 artifact_id."""
        renderer = FallbackRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert result.artifact_id == sample_artifact.artifact_id

    def test_render_carries_mime(
        self, FallbackRenderer, sample_artifact, sample_context
    ):
        """渲染结果应携带 mime 类型."""
        renderer = FallbackRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert result.mime == sample_artifact.mime

    def test_render_produces_html(
        self, FallbackRenderer, sample_artifact, sample_context
    ):
        """渲染结果应包含 HTML 片段."""
        renderer = FallbackRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert isinstance(result.html, str)
        assert "fallback-renderer" in result.html

    def test_render_metadata_contains_renderer_name(
        self, FallbackRenderer, sample_artifact, sample_context
    ):
        """metadata 应包含 'renderer' 字段，值为 'FallbackRenderer'."""
        renderer = FallbackRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert result.metadata.get("renderer") == "FallbackRenderer"

    def test_render_render_id_prefix(
        self, FallbackRenderer, sample_artifact, sample_context
    ):
        """render_id 应以 'rd-' 前缀开头."""
        renderer = FallbackRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert result.render_id.startswith("rd-")

    def test_render_validates_payload(
        self, FallbackRenderer, sample_context
    ):
        """render() 对空 payload 应抛出 ArtifactValidationError."""
        renderer = FallbackRenderer()
        empty_artifact = Artifact(
            artifact_id="art-empty",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            payload={},
        )
        with pytest.raises(ArtifactValidationError):
            renderer.render(empty_artifact, sample_context)


# ============================================================
# 12. FallbackRenderer 处理各种类型 Artifact
# ============================================================

class TestFallbackRendererMultiType:
    """验证 FallbackRenderer 处理各种类型的 Artifact."""

    @pytest.mark.parametrize(
        "artifact_type,mime,payload",
        [
            (ArtifactType.TEXT, "text/markdown", {"content": "文本内容"}),
            (ArtifactType.CHART, "application/vnd.dy3.chart+json", {
                "chart_type": "bar",
                "data": {"labels": ["A", "B"], "values": [1, 2]},
            }),
            (ArtifactType.GRAPH, "application/vnd.dy3.graph+json", {
                "nodes": [{"id": "n1"}],
                "edges": [{"source": "n1", "target": "n1"}],
            }),
            (ArtifactType.TABLE, "application/vnd.dy3.table+json", {
                "headers": ["col1"],
                "rows": [["v1"]],
            }),
            (ArtifactType.FORMULA, "application/vnd.dy3.formula+json", {
                "latex": "E=mc^2",
            }),
        ],
        ids=["text", "chart", "graph", "table", "formula"],
    )
    def test_render_various_artifact_types(
        self, FallbackRenderer, artifact_type, mime, payload, sample_context
    ):
        """FallbackRenderer 应能渲染各种类型的 Artifact."""
        renderer = FallbackRenderer()
        artifact = Artifact(
            artifact_id=f"art-{artifact_type.value}",
            type=artifact_type,
            mime=mime,
            title=f"{artifact_type.value} 制品",
            payload=payload,
        )
        result = renderer.render(artifact, sample_context)
        assert isinstance(result, RenderDescriptor)
        assert result.artifact_id == artifact.artifact_id
        assert result.mime == mime
        assert isinstance(result.html, str)
        assert len(result.html) > 0

    def test_render_unknown_mime_type(self, FallbackRenderer, sample_context):
        """FallbackRenderer 应能处理未知 MIME 类型的 Artifact."""
        renderer = FallbackRenderer()
        artifact = Artifact(
            artifact_id="art-unknown-mime",
            type=ArtifactType.TEXT,
            mime="application/x-totally-unknown",
            title="未知类型",
            payload={"mystery": "data", "count": 42},
        )
        result = renderer.render(artifact, sample_context)
        assert isinstance(result, RenderDescriptor)
        assert result.mime == "application/x-totally-unknown"
        assert "mystery" in result.html
        assert "42" in result.html

    def test_render_complex_nested_payload(self, FallbackRenderer, sample_context):
        """FallbackRenderer 应能处理嵌套复杂的 payload."""
        renderer = FallbackRenderer()
        artifact = Artifact(
            artifact_id="art-nested",
            type=ArtifactType.INTERACTIVE,
            mime="application/vnd.dy3.interactive+json",
            title="嵌套制品",
            payload={
                "widget_type": "slider",
                "config": {
                    "min": 0,
                    "max": 100,
                    "step": 5,
                },
                "metadata": ["tag1", "tag2"],
            },
        )
        result = renderer.render(artifact, sample_context)
        assert isinstance(result, RenderDescriptor)
        assert "widget_type" in result.html
        assert "slider" in result.html

    def test_render_unicode_content(self, FallbackRenderer, sample_context):
        """FallbackRenderer 应正确渲染 Unicode 内容 (中文/emoji)."""
        renderer = FallbackRenderer()
        artifact = Artifact(
            artifact_id="art-unicode",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            title="Unicode 测试 🎉",
            payload={"content": "你好世界 🌍 Привет"},
        )
        result = renderer.render(artifact, sample_context)
        assert "你好世界" in result.html
        assert "🌍" in result.html


# ============================================================
# 13. FallbackRenderer.supported_mime_types()
# ============================================================

class TestFallbackRendererSupportedMimeTypes:
    """验证 FallbackRenderer.supported_mime_types() 返回空列表."""

    def test_returns_empty_list(self, FallbackRenderer):
        """supported_mime_types() 应返回空列表."""
        renderer = FallbackRenderer()
        mimes = renderer.supported_mime_types()
        assert isinstance(mimes, list)
        assert len(mimes) == 0

    def test_returns_independent_copy(self, FallbackRenderer):
        """返回值应为独立副本 (即使为空)."""
        renderer = FallbackRenderer()
        mimes1 = renderer.supported_mime_types()
        mimes1.append("application/x-evil")
        mimes2 = renderer.supported_mime_types()
        assert len(mimes2) == 0

    def test_mime_types_class_attribute_empty(self, FallbackRenderer):
        """_MIME_TYPES 类属性应为空列表."""
        assert FallbackRenderer._MIME_TYPES == []


# ============================================================
# 14. FallbackRenderer HTML 输出内容
# ============================================================

class TestFallbackRendererHtmlOutput:
    """验证 FallbackRenderer 的 HTML 输出包含 title 和 payload 内容."""

    def test_html_contains_h3_title(self, FallbackRenderer, sample_context):
        """HTML 应包含 <h3> 标签包裹的 title."""
        renderer = FallbackRenderer()
        artifact = Artifact(
            artifact_id="art-h3",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            title="标题文本",
            payload={"content": "内容"},
        )
        html = renderer.do_render(artifact, sample_context)
        assert "<h3>" in html
        assert "</h3>" in html
        assert "标题文本" in html

    def test_html_contains_json_indented_payload(self, FallbackRenderer, sample_context):
        """HTML 中 payload 应为 JSON 缩进格式 (indent=2)."""
        renderer = FallbackRenderer()
        artifact = Artifact(
            artifact_id="art-json",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            title="JSON 格式测试",
            payload={"key1": "value1", "key2": "value2"},
        )
        html = renderer.do_render(artifact, sample_context)
        # JSON dumps with indent=2 应包含换行符和缩进
        assert "key1" in html
        assert "value1" in html
        # indent=2 会产生换行
        assert "\n" in html

    def test_html_wrapped_in_div(self, FallbackRenderer, sample_artifact, sample_context):
        """HTML 应被 <div class='fallback-renderer'> 包裹."""
        renderer = FallbackRenderer()
        html = renderer.do_render(sample_artifact, sample_context)
        assert html.startswith('<div class="fallback-renderer">')
        assert html.endswith("</div>")

    def test_html_with_none_payload_value(self, FallbackRenderer, sample_context):
        """payload 中包含 None 值时应正确序列化."""
        renderer = FallbackRenderer()
        artifact = Artifact(
            artifact_id="art-none",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            title="None 测试",
            payload={"key": None, "count": 0},
        )
        html = renderer.do_render(artifact, sample_context)
        assert "null" in html  # JSON 中的 None

    def test_html_with_non_serializable_payload(self, FallbackRenderer, sample_context):
        """payload 中包含不可序列化对象时应通过 default=str 处理."""
        renderer = FallbackRenderer()

        class CustomObject:
            def __str__(self) -> str:
                return "CUSTOM_OBJECT_STRING"

        artifact = Artifact(
            artifact_id="art-custom",
            type=ArtifactType.TEXT,
            mime="text/markdown",
            title="自定义对象测试",
            payload={"obj": CustomObject()},
        )
        html = renderer.do_render(artifact, sample_context)
        assert "CUSTOM_OBJECT_STRING" in html


# ============================================================
# 15. FallbackRenderer.update() / destroy() 继承行为
# ============================================================

class TestFallbackRendererInheritedBehavior:
    """验证 FallbackRenderer 继承 BaseRenderer 的 update/destroy 行为."""

    def test_update_raises_not_implemented(self, FallbackRenderer, sample_diff):
        """FallbackRenderer.update() 应继承默认行为 (抛 NotImplementedError)."""
        renderer = FallbackRenderer()
        with pytest.raises(NotImplementedError):
            renderer.update(sample_diff)

    def test_destroy_returns_none(self, FallbackRenderer):
        """FallbackRenderer.destroy() 应继承默认行为 (返回 None)."""
        renderer = FallbackRenderer()
        result = renderer.destroy()
        assert result is None

    def test_destroy_idempotent(self, FallbackRenderer):
        """FallbackRenderer.destroy() 可被多次调用."""
        renderer = FallbackRenderer()
        renderer.destroy()
        renderer.destroy()  # 不应抛异常


# ============================================================
# 16. FallbackRenderer 集成 — 与 Registry 配合
# ============================================================

class TestFallbackRendererRegistryIntegration:
    """验证 FallbackRenderer 可与 RendererRegistry 配合使用."""

    def test_fallback_as_universal_handler(self, FallbackRenderer, sample_artifact, sample_context):
        """FallbackRenderer 可作为通用降级渲染器处理任意 Artifact."""
        renderer = FallbackRenderer()
        # 模拟 Registry 找不到专用渲染器时使用 FallbackRenderer
        result = renderer.render(sample_artifact, sample_context)
        assert isinstance(result, RenderDescriptor)
        assert result.html is not None
        assert len(result.html) > 0

    def test_fallback_handles_multiple_artifacts(
        self, FallbackRenderer, sample_context
    ):
        """同一 FallbackRenderer 实例可处理多个不同 Artifact."""
        renderer = FallbackRenderer()
        artifacts = [
            Artifact(
                artifact_id=f"art-multi-{i}",
                type=ArtifactType.TEXT,
                mime="text/markdown",
                title=f"制品 {i}",
                payload={"content": f"内容 {i}"},
            )
            for i in range(3)
        ]
        results = [renderer.render(a, sample_context) for a in artifacts]
        assert len(results) == 3
        # 每个 render_id 应唯一
        render_ids = [r.render_id for r in results]
        assert len(set(render_ids)) == 3
        # 每个 artifact_id 应正确携带
        for i, result in enumerate(results):
            assert result.artifact_id == f"art-multi-{i}"
            assert f"制品 {i}" in result.html
