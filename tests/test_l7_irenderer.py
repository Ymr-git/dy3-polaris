"""L7 体验呈现层 — IRenderer 接口单元测试 (TDD).

测试覆盖:
1. IRenderer 是抽象基类 (ABC)，含 4 个抽象方法
   (render / update / destroy / supported_mime_types)
2. 无法直接实例化 IRenderer (抛出 TypeError)
3. 缺少任一抽象方法的子类仍无法实例化
4. 完整实现全部 4 个方法的具体子类可正常工作
5. TestRenderer.render() 返回 RenderDescriptor
6. TestRenderer.update() 接受 ArtifactDiff 并返回 RenderDescriptor
7. TestRenderer.destroy() 执行清理
8. TestRenderer.supported_mime_types() 返回 MIME 字符串列表
9. 抽象方法在 IRenderer 上以占位形式存在 (协议一致性)
"""

from __future__ import annotations

import inspect
from abc import ABC

import pytest

from dy3_polaris.l7.exceptions import L7Error
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

def _import_irenderer():
    from dy3_polaris.l7.irenderer import IRenderer

    return IRenderer


# ============================================================
# 测试用具体渲染器
# ============================================================

class _TestRenderer:
    """最小可用的具体渲染器 (在导入 IRenderer 后动态继承)."""

    # 真正的基类在 fixture 中通过 __class_getitem__ / 动态基类注入，
    # 但为简化，这里在测试函数内部定义带继承的版本。


def _make_concrete_renderer(base):
    """基于传入的 IRenderer 基类动态构造一个完整实现的具体渲染器."""

    class ConcreteTestRenderer(base):
        """完整实现 IRenderer 全部抽象方法的具体渲染器."""

        _MIME_TYPES = ["text/markdown", "text/plain"]

        def __init__(self) -> None:
            self._destroyed = False
            self._render_count = 0
            self._update_count = 0

        def render(self, artifact: Artifact, context: RenderContext) -> RenderDescriptor:
            self._render_count += 1
            return RenderDescriptor(
                artifact_id=artifact.artifact_id,
                mime=artifact.mime,
                html=f"<div>{artifact.payload.get('content', '')}</div>",
                config={"theme": context.theme},
                metadata={"render_count": self._render_count},
            )

        def update(self, diff: ArtifactDiff) -> RenderDescriptor:
            self._update_count += 1
            return RenderDescriptor(
                artifact_id=diff.artifact_id,
                mime="text/markdown",
                html=f"<div data-ops='{len(diff.ops)}'>updated</div>",
                metadata={"update_count": self._update_count},
            )

        def destroy(self) -> None:
            self._destroyed = True

        def supported_mime_types(self) -> list[str]:
            return list(self._MIME_TYPES)

    return ConcreteTestRenderer


def _make_incomplete_renderer(base):
    """构造一个缺少 destroy() 的不完整渲染器 (用于验证抽象强制)."""

    class IncompleteRenderer(base):
        def render(self, artifact, context):  # noqa: ANN001
            return RenderDescriptor()

        def update(self, diff):  # noqa: ANN001
            return RenderDescriptor()

        def supported_mime_types(self) -> list[str]:
            return ["text/markdown"]

        # 故意不实现 destroy()

    return IncompleteRenderer


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def IRenderer():
    return _import_irenderer()


@pytest.fixture
def ConcreteRenderer(IRenderer):
    return _make_concrete_renderer(IRenderer)


@pytest.fixture
def sample_artifact() -> Artifact:
    return Artifact(
        artifact_id="art-test-001",
        type=ArtifactType.TEXT,
        mime="text/markdown",
        source_agent="agent-explainer",
        payload={"content": "Hello, Polaris!"},
    )


@pytest.fixture
def sample_context() -> RenderContext:
    # learner_mode 为 LearnerMode 枚举, 使用合法值 "beginner"
    return RenderContext(theme="dark", learner_mode="beginner")


@pytest.fixture
def sample_diff() -> ArtifactDiff:
    # ArtifactDiff.ops 为 list[DiffOp], Pydantic 会将 dict 强制转换为 DiffOp
    return ArtifactDiff(
        artifact_id="art-test-001",
        ops=[{"op": "replace", "path": "/content", "value": "Updated!"}],
        edit_reason="refinement",
    )


# ============================================================
# 1. 抽象基类结构
# ============================================================

class TestIRendererAbstraction:
    """验证 IRenderer 的抽象基类结构."""

    def test_is_abc_subclass(self, IRenderer):
        """IRenderer 应继承自 abc.ABC."""
        assert issubclass(IRenderer, ABC)

    def test_cannot_instantiate_directly(self, IRenderer):
        """IRenderer 是抽象类，直接实例化应抛出 TypeError."""
        with pytest.raises(TypeError):
            IRenderer()

    def test_has_four_abstract_methods(self, IRenderer):
        """IRenderer 应声明 4 个抽象方法."""
        expected = {"render", "update", "destroy", "supported_mime_types"}
        actual = set(IRenderer.__abstractmethods__)
        assert actual == expected, f"期望抽象方法 {expected}，实际 {actual}"

    def test_abstract_methods_are_decorated(self, IRenderer):
        """四个方法均应被 @abstractmethod 装饰."""
        for name in ("render", "update", "destroy", "supported_mime_types"):
            method = getattr(IRenderer, name)
            assert getattr(method, "__isabstractmethod__", False), (
                f"{name} 应为抽象方法"
            )

    def test_render_signature(self, IRenderer):
        """render 签名: (artifact: Artifact, context: RenderContext) -> RenderDescriptor."""
        sig = inspect.signature(IRenderer.render)
        params = list(sig.parameters)
        assert "artifact" in params
        assert "context" in params

    def test_update_signature(self, IRenderer):
        """update 签名: (diff: ArtifactDiff) -> RenderDescriptor."""
        sig = inspect.signature(IRenderer.update)
        params = list(sig.parameters)
        assert "diff" in params

    def test_destroy_signature(self, IRenderer):
        """destroy 签名: () -> None (仅 self, 无额外参数)."""
        sig = inspect.signature(IRenderer.destroy)
        # 类上检视未绑定方法时包含 self, 故应仅有 self 一个参数
        assert set(sig.parameters) == {"self"}

    def test_supported_mime_types_signature(self, IRenderer):
        """supported_mime_types 签名: () -> list[str] (仅 self, 无额外参数)."""
        sig = inspect.signature(IRenderer.supported_mime_types)
        assert set(sig.parameters) == {"self"}


# ============================================================
# 2. 具体子类实例化
# ============================================================

class TestConcreteRendererInstantiation:
    """验证完整实现的具体渲染器可正常实例化."""

    def test_can_instantiate_concrete(self, ConcreteRenderer):
        """完整实现全部抽象方法的具体子类可实例化."""
        renderer = ConcreteRenderer()
        assert renderer is not None
        assert isinstance(renderer, ConcreteRenderer)

    def test_incomplete_renderer_cannot_instantiate(self, IRenderer):
        """缺少任一抽象方法的子类仍无法实例化 (TypeError)."""
        Incomplete = _make_incomplete_renderer(IRenderer)
        with pytest.raises(TypeError):
            Incomplete()


# ============================================================
# 3. render() 行为
# ============================================================

class TestRenderMethod:
    """验证 TestRenderer.render() 行为."""

    def test_render_returns_render_descriptor(
        self, ConcreteRenderer, sample_artifact, sample_context
    ):
        """render() 应返回 RenderDescriptor 实例."""
        renderer = ConcreteRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert isinstance(result, RenderDescriptor)

    def test_render_carries_artifact_id(
        self, ConcreteRenderer, sample_artifact, sample_context
    ):
        """渲染结果应携带来源 artifact_id."""
        renderer = ConcreteRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert result.artifact_id == sample_artifact.artifact_id

    def test_render_carries_mime(
        self, ConcreteRenderer, sample_artifact, sample_context
    ):
        """渲染结果应携带 mime 类型."""
        renderer = ConcreteRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert result.mime == sample_artifact.mime

    def test_render_produces_html(
        self, ConcreteRenderer, sample_artifact, sample_context
    ):
        """渲染结果应包含 HTML 片段."""
        renderer = ConcreteRenderer()
        result = renderer.render(sample_artifact, sample_context)
        assert isinstance(result.html, str)
        assert len(result.html) > 0


# ============================================================
# 4. update() 行为
# ============================================================

class TestUpdateMethod:
    """验证 TestRenderer.update() 行为."""

    def test_update_accepts_artifact_diff(self, ConcreteRenderer, sample_diff):
        """update() 接受 ArtifactDiff 参数."""
        renderer = ConcreteRenderer()
        result = renderer.update(sample_diff)
        assert isinstance(result, RenderDescriptor)

    def test_update_returns_render_descriptor(
        self, ConcreteRenderer, sample_diff
    ):
        """update() 返回 RenderDescriptor 实例."""
        renderer = ConcreteRenderer()
        result = renderer.update(sample_diff)
        assert isinstance(result, RenderDescriptor)

    def test_update_carries_artifact_id(self, ConcreteRenderer, sample_diff):
        """增量更新结果应携带 diff 中的 artifact_id."""
        renderer = ConcreteRenderer()
        result = renderer.update(sample_diff)
        assert result.artifact_id == sample_diff.artifact_id


# ============================================================
# 5. destroy() 行为
# ============================================================

class TestDestroyMethod:
    """验证 TestRenderer.destroy() 行为."""

    def test_destroy_cleans_up(self, ConcreteRenderer):
        """destroy() 应执行资源清理."""
        renderer = ConcreteRenderer()
        assert renderer._destroyed is False
        renderer.destroy()
        assert renderer._destroyed is True

    def test_destroy_returns_none(self, ConcreteRenderer):
        """destroy() 返回 None."""
        renderer = ConcreteRenderer()
        result = renderer.destroy()
        assert result is None

    def test_destroy_idempotent(self, ConcreteRenderer):
        """destroy() 可被多次调用 (幂等)."""
        renderer = ConcreteRenderer()
        renderer.destroy()
        renderer.destroy()  # 不应抛异常
        assert renderer._destroyed is True


# ============================================================
# 6. supported_mime_types() 行为
# ============================================================

class TestSupportedMimeTypes:
    """验证 TestRenderer.supported_mime_types() 行为."""

    def test_returns_list_of_strings(self, ConcreteRenderer):
        """supported_mime_types() 返回字符串列表."""
        renderer = ConcreteRenderer()
        mimes = renderer.supported_mime_types()
        assert isinstance(mimes, list)
        assert len(mimes) > 0
        assert all(isinstance(m, str) for m in mimes)

    def test_returns_expected_mimes(self, ConcreteRenderer):
        """返回预期的 MIME 类型."""
        renderer = ConcreteRenderer()
        mimes = renderer.supported_mime_types()
        assert "text/markdown" in mimes

    def test_returns_independent_copy(self, ConcreteRenderer):
        """返回值应为独立副本 (修改不影响内部状态)."""
        renderer = ConcreteRenderer()
        mimes1 = renderer.supported_mime_types()
        mimes1.append("application/x-evil")
        mimes2 = renderer.supported_mime_types()
        assert "application/x-evil" not in mimes2


# ============================================================
# 7. 异常体系一致性
# ============================================================

class TestExceptionConsistency:
    """验证 L7 异常体系与渲染器接口一致."""

    def test_l7_error_is_exception(self):
        """L7Error 应为 Exception 子类."""
        assert issubclass(L7Error, Exception)

    def test_l7_error_inherits_l6_error(self):
        """L7Error 应继承自 L6Error."""
        from dy3_polaris.l6.core.exceptions import L6Error

        assert issubclass(L7Error, L6Error)

    def test_renderer_not_found_exists(self):
        """RendererNotFoundError 异常应存在且为 L7Error 子类."""
        from dy3_polaris.l7.exceptions import RendererNotFoundError

        assert issubclass(RendererNotFoundError, L7Error)
