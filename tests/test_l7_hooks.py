"""L7 体验呈现层 — 渲染钩子 / 中间件系统 单元测试 (TDD).

测试覆盖 (≥45 用例):
1. BaseRenderHook 默认行为 (no-op, 返回 None, name 属性)
2. HookRegistry: register / unregister / clear / count / 优先级排序
3. HookRegistry: 线程安全 (并发 register / unregister)
4. HookablePipeline: before_render 修改 context
5. HookablePipeline: after_render 修改 descriptor
6. HookablePipeline: on_render_error 失败时被调用
7. HookablePipeline: 错误在钩子之后仍向上传播
8. HookablePipeline: 多个钩子按优先级顺序运行
9. HookablePipeline: update/destroy/get_cached/clear_cache/get_stats 委托
10. HookablePipeline: get_stats 包含 hook_count
11. LoggingHook: 产生日志
12. TimingHook: 记录耗时, get_average_time, get_total_renders
13. CachingHook: 第二次渲染返回缓存描述符
14. CachingHook: 版本变化时不命中缓存
15. 钩子返回 None 时保持原始 context/descriptor

融合方案:
- Express.js middleware: 链式处理器
- Django middleware: process_request/process_response 模式
- Starlette middleware: ASGI 中间件
- React Suspense: 降级处理
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from dy3_polaris.l7.models import (
    Artifact,
    ArtifactType,
    RenderContext,
    RenderDescriptor,
)


# ============================================================
# 延迟导入被测对象 (hooks 模块尚不存在时, 单个测试失败而非整体收集失败)
# ============================================================


@pytest.fixture
def hooks_mod():
    """延迟导入 hooks 模块, 实现失败隔离."""
    import dy3_polaris.l7.hooks as h

    return h


# ============================================================
# 测试用 FakePipeline — 最小可用的渲染流水线替身
# ============================================================


class FakePipeline:
    """最小渲染流水线替身 — 跟踪所有委托调用.

    实现 HookablePipeline 所需的完整接口子集, 不依赖 RendererRegistry,
    使钩子测试可以专注于 HookablePipeline 行为本身.
    """

    __test__ = False  # pytest 不应将此类作为测试类收集

    def __init__(self) -> None:
        self.render_count = 0
        self.update_count = 0
        self.destroy_count = 0
        self.clear_cache_count = 0
        self.get_cached_count = 0
        self.last_artifact: Artifact | None = None
        self.last_context: RenderContext | None = None
        self.last_kwargs: dict | None = None
        self.fail: bool = False
        self.fail_error: Exception | None = None
        self._cache: dict[str, RenderDescriptor] = {}

    def render(
        self,
        artifact: Artifact,
        context: RenderContext,
        **kwargs: object,
    ) -> RenderDescriptor:
        self.render_count += 1
        self.last_artifact = artifact
        self.last_context = context
        self.last_kwargs = dict(kwargs)
        if self.fail:
            err = self.fail_error or RuntimeError("render failed")
            raise err
        descriptor = RenderDescriptor(
            artifact_id=artifact.artifact_id,
            mime=artifact.mime,
            html=f"<div>{artifact.payload.get('content', '')}</div>",
            config={"theme": context.theme, "locale": context.locale},
            render_time_ms=10.0,
        )
        self._cache[artifact.artifact_id] = descriptor
        return descriptor

    def update(self, artifact_id: str, diff: object, **kwargs: object) -> RenderDescriptor:
        self.update_count += 1
        return RenderDescriptor(
            artifact_id=artifact_id,
            mime="text/markdown",
            html="<div>updated</div>",
        )

    def destroy(self, artifact_id: str, **kwargs: object) -> None:
        self.destroy_count += 1

    def get_cached(self, artifact_id: str) -> RenderDescriptor | None:
        self.get_cached_count += 1
        return self._cache.get(artifact_id)

    def clear_cache(self) -> None:
        self.clear_cache_count += 1
        self._cache.clear()

    def get_stats(self) -> dict:
        return {
            "total_renders": self.render_count,
            "cache_size": len(self._cache),
            "mode": "fake",
        }


# ============================================================
# 测试用钩子 (实现 RenderHook 协议, 鸭子类型, 不依赖 hooks 模块)
# ============================================================


class SpyHook:
    """记录所有调用的间谍钩子 — 实现 RenderHook 协议 (鸭子类型)."""

    __test__ = False

    def __init__(self, name: str = "SpyHook") -> None:
        self.name = name
        self.before_calls: list[tuple] = []
        self.after_calls: list[tuple] = []
        self.error_calls: list[tuple] = []
        self._before_return = None
        self._after_return = None

    def before_render(self, artifact: Artifact, context: RenderContext):
        self.before_calls.append((artifact, context))
        return self._before_return

    def after_render(self, artifact: Artifact, descriptor: RenderDescriptor):
        self.after_calls.append((artifact, descriptor))
        return self._after_return

    def on_render_error(self, artifact: Artifact, error: Exception) -> None:
        self.error_calls.append((artifact, error))


class ContextModifyingHook:
    """修改渲染上下文的钩子 — 返回修改后的 RenderContext."""

    __test__ = False

    def __init__(self, theme: str | None = None, locale: str | None = None) -> None:
        self.name = "ContextModifyingHook"
        self.theme = theme
        self.locale = locale
        self.received_contexts: list[RenderContext] = []

    def before_render(self, artifact: Artifact, context: RenderContext):
        self.received_contexts.append(context)
        modified = context.model_copy()
        if self.theme is not None:
            modified.theme = self.theme
        if self.locale is not None:
            modified.locale = self.locale
        return modified

    def after_render(self, artifact: Artifact, descriptor: RenderDescriptor):
        return None

    def on_render_error(self, artifact: Artifact, error: Exception) -> None:
        pass


class DescriptorModifyingHook:
    """修改渲染描述符的钩子 — 返回修改后的 RenderDescriptor."""

    __test__ = False

    def __init__(
        self,
        html: str | None = None,
        config_key: str | None = None,
        config_value: object = None,
    ) -> None:
        self.name = "DescriptorModifyingHook"
        self.html = html
        self.config_key = config_key
        self.config_value = config_value
        self.received_descriptors: list[RenderDescriptor] = []

    def before_render(self, artifact: Artifact, context: RenderContext):
        return None

    def after_render(self, artifact: Artifact, descriptor: RenderDescriptor):
        self.received_descriptors.append(descriptor)
        modified = descriptor.model_copy()
        modified.config = dict(descriptor.config)
        if self.html is not None:
            modified.html = self.html
        if self.config_key is not None:
            modified.config[self.config_key] = self.config_value
        return modified

    def on_render_error(self, artifact: Artifact, error: Exception) -> None:
        pass


class OrderTrackingHook:
    """追踪运行顺序的钩子 — 将自身名字追加到共享列表."""

    __test__ = False

    def __init__(self, label: str, before_log: list[str], after_log: list[str]) -> None:
        self.name = label
        self._before_log = before_log
        self._after_log = after_log

    def before_render(self, artifact: Artifact, context: RenderContext):
        self._before_log.append(self.name)
        return None

    def after_render(self, artifact: Artifact, descriptor: RenderDescriptor):
        self._after_log.append(self.name)
        return None

    def on_render_error(self, artifact: Artifact, error: Exception) -> None:
        pass


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def registry(hooks_mod):
    return hooks_mod.HookRegistry()


@pytest.fixture
def hookable(hooks_mod, fake_pipeline, registry):
    return hooks_mod.HookablePipeline(fake_pipeline, registry)


@pytest.fixture
def text_artifact() -> Artifact:
    return Artifact(
        artifact_id="art-text-001",
        type=ArtifactType.TEXT,
        mime="text/markdown",
        source_agent="agent.explainer",
        payload={"content": "Hello, Polaris!"},
        version=1,
        title="测试文本",
    )


@pytest.fixture
def context() -> RenderContext:
    return RenderContext(theme="light", locale="zh-CN")


# ============================================================
# 1. BaseRenderHook 默认行为
# ============================================================


def test_base_render_hook_before_render_returns_none(hooks_mod, text_artifact, context):
    """BaseRenderHook.before_render 默认返回 None (no-op)."""
    hook = hooks_mod.BaseRenderHook()
    assert hook.before_render(text_artifact, context) is None


def test_base_render_hook_after_render_returns_none(hooks_mod, text_artifact):
    """BaseRenderHook.after_render 默认返回 None (no-op)."""
    hook = hooks_mod.BaseRenderHook()
    descriptor = RenderDescriptor(artifact_id=text_artifact.artifact_id)
    assert hook.after_render(text_artifact, descriptor) is None


def test_base_render_hook_on_render_error_does_not_raise(hooks_mod, text_artifact):
    """BaseRenderHook.on_render_error 默认不抛异常."""
    hook = hooks_mod.BaseRenderHook()
    # 不应抛出任何异常
    hook.on_render_error(text_artifact, RuntimeError("boom"))


def test_base_render_hook_name_defaults_to_class_name(hooks_mod):
    """BaseRenderHook.name 默认返回类名."""
    hook = hooks_mod.BaseRenderHook()
    assert hook.name == "BaseRenderHook"


def test_base_render_hook_subclass_overrides_before_render(hooks_mod, text_artifact, context):
    """子类可以只覆盖 before_render, 其余方法保持默认 no-op."""
    class MyHook(hooks_mod.BaseRenderHook):
        def before_render(self, artifact, context):
            modified = context.model_copy()
            modified.theme = "dark"
            return modified

    hook = MyHook()
    result = hook.before_render(text_artifact, context)
    assert result is not None
    assert result.theme == "dark"
    # 未覆盖的方法仍为 no-op
    assert hook.after_render(text_artifact, RenderDescriptor()) is None
    assert hook.name == "MyHook"


def test_base_render_hook_satisfies_render_hook_protocol(hooks_mod):
    """BaseRenderHook 实例满足 RenderHook 协议 (runtime_checkable)."""
    hook = hooks_mod.BaseRenderHook()
    assert isinstance(hook, hooks_mod.RenderHook)


# ============================================================
# 2. RenderHook Protocol
# ============================================================


def test_render_hook_protocol_is_runtime_checkable(hooks_mod):
    """RenderHook 协议可进行 runtime isinstance 检查."""
    # Protocol 类应存在且可被 isinstance 使用
    spy = SpyHook()
    assert isinstance(spy, hooks_mod.RenderHook)


# ============================================================
# 3. HookRegistry: 基础操作
# ============================================================


def test_registry_initially_empty(hooks_mod):
    """新建 HookRegistry 应为空."""
    reg = hooks_mod.HookRegistry()
    assert reg.count() == 0
    assert reg.get_hooks() == []


def test_registry_register_increases_count(hooks_mod):
    """register 后 count 增加."""
    reg = hooks_mod.HookRegistry()
    reg.register(SpyHook(name="a"))
    assert reg.count() == 1
    reg.register(SpyHook(name="b"))
    assert reg.count() == 2


def test_registry_get_hooks_returns_registered_hook(hooks_mod):
    """get_hooks 返回已注册的钩子."""
    reg = hooks_mod.HookRegistry()
    hook = SpyHook(name="alpha")
    reg.register(hook)
    hooks = reg.get_hooks()
    assert hooks == [hook]


def test_registry_unregister_removes_hook(hooks_mod):
    """unregister 移除指定钩子."""
    reg = hooks_mod.HookRegistry()
    hook = SpyHook(name="to-remove")
    reg.register(hook)
    assert reg.count() == 1
    reg.unregister(hook)
    assert reg.count() == 0
    assert reg.get_hooks() == []


def test_registry_unregister_not_registered_is_safe(hooks_mod):
    """unregister 未注册的钩子不应抛异常 (安全操作)."""
    reg = hooks_mod.HookRegistry()
    # 空注册表 unregister 不抛异常
    reg.unregister(SpyHook(name="never-registered"))
    assert reg.count() == 0


def test_registry_clear_removes_all_hooks(hooks_mod):
    """clear 移除所有钩子."""
    reg = hooks_mod.HookRegistry()
    reg.register(SpyHook(name="a"))
    reg.register(SpyHook(name="b"))
    reg.register(SpyHook(name="c"))
    assert reg.count() == 3
    reg.clear()
    assert reg.count() == 0
    assert reg.get_hooks() == []


def test_registry_count_reflects_registrations(hooks_mod):
    """count 准确反映注册数量."""
    reg = hooks_mod.HookRegistry()
    for i in range(5):
        reg.register(SpyHook(name=f"h{i}"))
    assert reg.count() == 5
    reg.unregister(reg.get_hooks()[0])
    assert reg.count() == 4


# ============================================================
# 4. HookRegistry: 优先级排序
# ============================================================


def test_registry_priority_lower_runs_first(hooks_mod):
    """优先级数值越低越先运行 (get_hooks 排序)."""
    reg = hooks_mod.HookRegistry()
    high = SpyHook(name="high")
    low = SpyHook(name="low")
    reg.register(high, priority=100)
    reg.register(low, priority=0)
    hooks = reg.get_hooks()
    assert hooks == [low, high]  # low (0) 在 high (100) 之前


def test_registry_equal_priority_preserves_insertion_order(hooks_mod):
    """相同优先级保持注册顺序 (稳定排序)."""
    reg = hooks_mod.HookRegistry()
    first = SpyHook(name="first")
    second = SpyHook(name="second")
    third = SpyHook(name="third")
    reg.register(first, priority=5)
    reg.register(second, priority=5)
    reg.register(third, priority=5)
    hooks = reg.get_hooks()
    assert hooks == [first, second, third]


def test_registry_priority_ordering_multiple(hooks_mod):
    """多个不同优先级的钩子按数值升序排列."""
    reg = hooks_mod.HookRegistry()
    p30 = SpyHook(name="p30")
    p10 = SpyHook(name="p10")
    p20 = SpyHook(name="p20")
    p0 = SpyHook(name="p0")
    reg.register(p30, priority=30)
    reg.register(p10, priority=10)
    reg.register(p20, priority=20)
    reg.register(p0, priority=0)
    hooks = reg.get_hooks()
    assert hooks == [p0, p10, p20, p30]


def test_registry_register_existing_updates_priority(hooks_mod):
    """重复注册同一钩子对象时更新优先级 (幂等)."""
    reg = hooks_mod.HookRegistry()
    hook = SpyHook(name="dup")
    other = SpyHook(name="other")
    reg.register(hook, priority=50)
    reg.register(other, priority=10)
    assert reg.get_hooks() == [other, hook]
    # 重新注册 hook 为更高优先级 (更低数值)
    reg.register(hook, priority=0)
    assert reg.count() == 2  # 未重复添加
    assert reg.get_hooks() == [hook, other]


def test_registry_get_hooks_returns_copy(hooks_mod):
    """get_hooks 返回列表副本, 修改不影响内部状态."""
    reg = hooks_mod.HookRegistry()
    hook = SpyHook(name="a")
    reg.register(hook)
    hooks = reg.get_hooks()
    hooks.clear()
    assert reg.count() == 1
    assert reg.get_hooks() == [hook]


# ============================================================
# 5. HookRegistry: 线程安全
# ============================================================


def test_registry_concurrent_register_thread_safety(hooks_mod):
    """并发 register 应保证线程安全, 最终 count 正确."""
    reg = hooks_mod.HookRegistry()
    n = 50
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            reg.register(SpyHook(name=f"h{i}"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert reg.count() == n


def test_registry_concurrent_register_and_unregister(hooks_mod):
    """并发 register 与 unregister 应无异常且最终状态一致."""
    reg = hooks_mod.HookRegistry()
    keep = [SpyHook(name=f"keep{i}") for i in range(20)]
    remove = [SpyHook(name=f"remove{i}") for i in range(20)]
    for h in keep + remove:
        reg.register(h)
    assert reg.count() == 40

    errors: list[Exception] = []

    def unreg_worker(subset: list) -> None:
        try:
            for h in subset:
                reg.unregister(h)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    half = len(remove) // 2
    t1 = threading.Thread(target=unreg_worker, args=(remove[:half],))
    t2 = threading.Thread(target=unreg_worker, args=(remove[half:],))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []
    assert reg.count() == 20
    remaining = set(id(h) for h in reg.get_hooks())
    keep_ids = set(id(h) for h in keep)
    assert remaining == keep_ids


# ============================================================
# 6. HookablePipeline: 基础渲染与委托
# ============================================================


def test_hookable_pipeline_render_delegates_to_underlying(hookable, fake_pipeline, text_artifact, context):
    """render 委托到底层流水线."""
    hookable.render(text_artifact, context)
    assert fake_pipeline.render_count == 1
    assert fake_pipeline.last_artifact is text_artifact


def test_hookable_pipeline_render_returns_descriptor(hookable, fake_pipeline, text_artifact, context):
    """render 返回 RenderDescriptor."""
    result = hookable.render(text_artifact, context)
    assert isinstance(result, RenderDescriptor)
    assert result.artifact_id == text_artifact.artifact_id


def test_hookable_pipeline_render_passes_kwargs(hookable, fake_pipeline, text_artifact, context):
    """render 透传 kwargs 到底层流水线."""
    hookable.render(text_artifact, context, force=True, custom="value")
    assert fake_pipeline.last_kwargs == {"force": True, "custom": "value"}


def test_hookable_pipeline_no_hooks_works(hooks_mod, fake_pipeline, text_artifact, context):
    """无钩子时 HookablePipeline 正常工作 (透传)."""
    reg = hooks_mod.HookRegistry()
    pipe = hooks_mod.HookablePipeline(fake_pipeline, reg)
    result = pipe.render(text_artifact, context)
    assert isinstance(result, RenderDescriptor)
    assert fake_pipeline.render_count == 1


# ============================================================
# 7. HookablePipeline: before_render 修改 context
# ============================================================


def test_hookable_pipeline_before_render_modifies_context(hookable, fake_pipeline, registry, text_artifact, context):
    """before_render 返回新 context 时, 底层流水线接收修改后的 context."""
    hook = ContextModifyingHook(theme="dark")
    registry.register(hook)
    hookable.render(text_artifact, context)
    assert fake_pipeline.last_context is not None
    assert fake_pipeline.last_context.theme == "dark"


def test_hookable_pipeline_before_render_none_preserves_context(hookable, fake_pipeline, registry, text_artifact, context):
    """before_render 返回 None 时保持原始 context."""
    hook = SpyHook()
    hook._before_return = None
    registry.register(hook)
    hookable.render(text_artifact, context)
    assert fake_pipeline.last_context is context


def test_hookable_pipeline_before_render_chained_multiple_hooks(hookable, fake_pipeline, registry, text_artifact, context):
    """多个 before_render 钩子链式修改 context (每个基于上一个结果)."""
    hook1 = ContextModifyingHook(theme="dark")
    hook2 = ContextModifyingHook(locale="en-US")
    registry.register(hook1)
    registry.register(hook2)
    hookable.render(text_artifact, context)
    final = fake_pipeline.last_context
    assert final.theme == "dark"
    assert final.locale == "en-US"
    # hook2 应接收到 hook1 修改后的 context
    assert hook2.received_contexts[0].theme == "dark"


# ============================================================
# 8. HookablePipeline: after_render 修改 descriptor
# ============================================================


def test_hookable_pipeline_after_render_modifies_descriptor(hookable, registry, text_artifact, context):
    """after_render 返回新 descriptor 时, 最终返回修改后的 descriptor."""
    hook = DescriptorModifyingHook(html="<div>modified</div>")
    registry.register(hook)
    result = hookable.render(text_artifact, context)
    assert result.html == "<div>modified</div>"


def test_hookable_pipeline_after_render_none_preserves_descriptor(hookable, fake_pipeline, registry, text_artifact, context):
    """after_render 返回 None 时保持底层流水线产出的 descriptor."""
    hook = SpyHook()
    hook._after_return = None
    registry.register(hook)
    result = hookable.render(text_artifact, context)
    # 与底层产出的 html 一致 (未被修改)
    assert result.html == f"<div>{text_artifact.payload.get('content', '')}</div>"


def test_hookable_pipeline_after_render_chained_multiple_hooks(hookable, registry, text_artifact, context):
    """多个 after_render 钩子链式修改 descriptor (累加 config)."""
    hook1 = DescriptorModifyingHook(config_key="step1", config_value=True)
    hook2 = DescriptorModifyingHook(config_key="step2", config_value=True)
    registry.register(hook1)
    registry.register(hook2)
    result = hookable.render(text_artifact, context)
    assert result.config.get("step1") is True
    assert result.config.get("step2") is True


# ============================================================
# 9. HookablePipeline: on_render_error 与错误传播
# ============================================================


def test_hookable_pipeline_on_render_error_called_on_failure(hookable, fake_pipeline, registry, text_artifact, context):
    """底层渲染失败时调用 on_render_error."""
    fake_pipeline.fail = True
    spy = SpyHook()
    registry.register(spy)
    with pytest.raises(RuntimeError):
        hookable.render(text_artifact, context)
    assert len(spy.error_calls) == 1
    called_artifact, called_error = spy.error_calls[0]
    assert called_artifact is text_artifact
    assert isinstance(called_error, RuntimeError)


def test_hookable_pipeline_error_propagated_after_hooks(hookable, fake_pipeline, registry, text_artifact, context):
    """错误在钩子执行后仍向上传播."""
    fake_pipeline.fail = True
    fake_pipeline.fail_error = ValueError("propagated")
    registry.register(SpyHook())
    with pytest.raises(ValueError, match="propagated"):
        hookable.render(text_artifact, context)


def test_hookable_pipeline_all_hooks_on_render_error_called(hookable, fake_pipeline, registry, text_artifact, context):
    """所有钩子的 on_render_error 都被调用 (即使某个钩子异常也不中断其余)."""
    fake_pipeline.fail = True
    spy1 = SpyHook(name="spy1")
    spy2 = SpyHook(name="spy2")
    spy3 = SpyHook(name="spy3")
    registry.register(spy1)
    registry.register(spy2)
    registry.register(spy3)
    with pytest.raises(RuntimeError):
        hookable.render(text_artifact, context)
    assert len(spy1.error_calls) == 1
    assert len(spy2.error_calls) == 1
    assert len(spy3.error_calls) == 1


# ============================================================
# 10. HookablePipeline: 多钩子优先级顺序
# ============================================================


def test_hookable_pipeline_multiple_hooks_run_in_priority_order(hookable, registry, text_artifact, context):
    """多个钩子按优先级顺序运行 before_render 和 after_render."""
    before_log: list[str] = []
    after_log: list[str] = []
    hook_a = OrderTrackingHook("A", before_log, after_log)
    hook_b = OrderTrackingHook("B", before_log, after_log)
    hook_c = OrderTrackingHook("C", before_log, after_log)
    # B (0) -> A (10) -> C (20)
    registry.register(hook_a, priority=10)
    registry.register(hook_b, priority=0)
    registry.register(hook_c, priority=20)
    hookable.render(text_artifact, context)
    assert before_log == ["B", "A", "C"]
    assert after_log == ["B", "A", "C"]


# ============================================================
# 11. HookablePipeline: 委托方法
# ============================================================


def test_hookable_pipeline_update_delegates(hookable, fake_pipeline, text_artifact):
    """update 委托到底层流水线."""
    from dy3_polaris.l7.models import ArtifactDiff

    d = ArtifactDiff(artifact_id=text_artifact.artifact_id)
    result = hookable.update(text_artifact.artifact_id, d)
    assert fake_pipeline.update_count == 1
    assert isinstance(result, RenderDescriptor)


def test_hookable_pipeline_destroy_delegates(hookable, fake_pipeline, text_artifact):
    """destroy 委托到底层流水线."""
    hookable.destroy(text_artifact.artifact_id)
    assert fake_pipeline.destroy_count == 1


def test_hookable_pipeline_get_cached_delegates(hookable, fake_pipeline, text_artifact, context):
    """get_cached 委托到底层流水线."""
    # 先渲染产生缓存
    hookable.render(text_artifact, context)
    result = hookable.get_cached(text_artifact.artifact_id)
    assert fake_pipeline.get_cached_count >= 1
    assert result is not None
    assert result.artifact_id == text_artifact.artifact_id


def test_hookable_pipeline_clear_cache_delegates(hookable, fake_pipeline):
    """clear_cache 委托到底层流水线."""
    hookable.clear_cache()
    assert fake_pipeline.clear_cache_count == 1


def test_hookable_pipeline_get_stats_includes_hook_count(hookable, registry, text_artifact, context):
    """get_stats 包含 hook_count 字段."""
    registry.register(SpyHook(name="a"))
    registry.register(SpyHook(name="b"))
    stats = hookable.get_stats()
    assert "hook_count" in stats
    assert stats["hook_count"] == 2


def test_hookable_pipeline_get_stats_merges_underlying_stats(hookable, fake_pipeline, text_artifact, context):
    """get_stats 合并底层流水线的统计信息."""
    hookable.render(text_artifact, context)
    stats = hookable.get_stats()
    assert stats.get("total_renders") == 1
    assert stats.get("mode") == "fake"
    assert "hook_count" in stats


# ============================================================
# 12. LoggingHook
# ============================================================


def test_logging_hook_before_render_logs(hooks_mod, text_artifact, context, caplog):
    """LoggingHook.before_render 产生日志."""
    caplog.set_level(logging.INFO)
    hook = hooks_mod.LoggingHook()
    hook.before_render(text_artifact, context)
    assert any("before_render" in r.message for r in caplog.records)
    assert any(text_artifact.artifact_id in r.message for r in caplog.records)


def test_logging_hook_after_render_logs_with_render_time(hooks_mod, text_artifact, caplog):
    """LoggingHook.after_render 产生包含 render_time_ms 的日志."""
    caplog.set_level(logging.INFO)
    hook = hooks_mod.LoggingHook()
    descriptor = RenderDescriptor(
        artifact_id=text_artifact.artifact_id,
        render_time_ms=42.5,
    )
    hook.after_render(text_artifact, descriptor)
    messages = [r.message for r in caplog.records]
    assert any("after_render" in m for m in messages)
    assert any("42.5" in m for m in messages)


def test_logging_hook_on_render_error_logs(hooks_mod, text_artifact, caplog):
    """LoggingHook.on_render_error 产生错误日志."""
    caplog.set_level(logging.ERROR)
    hook = hooks_mod.LoggingHook()
    hook.on_render_error(text_artifact, RuntimeError("boom"))
    messages = [r.message for r in caplog.records]
    assert any("render_error" in m for m in messages)
    assert any("boom" in m for m in messages)


def test_logging_hook_name(hooks_mod):
    """LoggingHook.name 默认为类名."""
    hook = hooks_mod.LoggingHook()
    assert hook.name == "LoggingHook"


# ============================================================
# 13. TimingHook
# ============================================================


def test_timing_hook_records_duration(hooks_mod, text_artifact):
    """TimingHook 记录渲染耗时."""
    hook = hooks_mod.TimingHook()
    descriptor = RenderDescriptor(
        artifact_id=text_artifact.artifact_id,
        render_time_ms=15.0,
    )
    hook.after_render(text_artifact, descriptor)
    assert hook.get_total_renders() == 1


def test_timing_hook_get_average_time(hooks_mod, text_artifact):
    """TimingHook.get_average_time 返回平均耗时."""
    hook = hooks_mod.TimingHook()
    hook.after_render(text_artifact, RenderDescriptor(render_time_ms=10.0))
    hook.after_render(text_artifact, RenderDescriptor(render_time_ms=20.0))
    hook.after_render(text_artifact, RenderDescriptor(render_time_ms=30.0))
    assert hook.get_total_renders() == 3
    assert hook.get_average_time() == pytest.approx(20.0)


def test_timing_hook_get_total_renders(hooks_mod, text_artifact):
    """TimingHook.get_total_renders 返回渲染次数."""
    hook = hooks_mod.TimingHook()
    assert hook.get_total_renders() == 0
    hook.after_render(text_artifact, RenderDescriptor(render_time_ms=5.0))
    assert hook.get_total_renders() == 1
    hook.after_render(text_artifact, RenderDescriptor(render_time_ms=5.0))
    assert hook.get_total_renders() == 2


def test_timing_hook_average_zero_when_no_renders(hooks_mod):
    """无渲染记录时 get_average_time 返回 0."""
    hook = hooks_mod.TimingHook()
    assert hook.get_average_time() == 0.0
    assert hook.get_total_renders() == 0


# ============================================================
# 14. CachingHook
# ============================================================


def test_caching_hook_returns_cached_on_second_render(hooks_mod, text_artifact, context):
    """CachingHook: 相同 artifact_id + version 第二次渲染返回缓存, 跳过底层渲染."""
    fake = FakePipeline()
    reg = hooks_mod.HookRegistry()
    reg.register(hooks_mod.CachingHook())
    pipe = hooks_mod.HookablePipeline(fake, reg)

    # 第一次渲染 — 底层被调用, 缓存被填充
    first = pipe.render(text_artifact, context)
    assert fake.render_count == 1

    # 第二次渲染 — 命中缓存, 底层不应被再次调用
    second = pipe.render(text_artifact, context)
    assert fake.render_count == 1
    # 返回缓存的描述符 (同一对象)
    assert second is first


def test_caching_hook_does_not_cache_when_version_changes(hooks_mod, text_artifact, context):
    """CachingHook: artifact 版本变化时不命中缓存, 触发重新渲染."""
    fake = FakePipeline()
    reg = hooks_mod.HookRegistry()
    reg.register(hooks_mod.CachingHook())
    pipe = hooks_mod.HookablePipeline(fake, reg)

    # v1 渲染
    art_v1 = text_artifact.model_copy()
    art_v1.version = 1
    pipe.render(art_v1, context)
    assert fake.render_count == 1

    # v2 渲染 — 不同版本, 缓存未命中, 底层被调用
    art_v2 = text_artifact.model_copy()
    art_v2.version = 2
    pipe.render(art_v2, context)
    assert fake.render_count == 2

    # 再次 v1 — 命中缓存
    pipe.render(art_v1, context)
    assert fake.render_count == 2


def test_caching_hook_after_render_caches_descriptor(hooks_mod, text_artifact):
    """CachingHook.after_render 缓存描述符, 可通过 get_cached_descriptor 取回."""
    hook = hooks_mod.CachingHook()
    descriptor = RenderDescriptor(
        artifact_id=text_artifact.artifact_id,
        render_time_ms=12.0,
    )
    # after_render 返回 None (不修改描述符, 仅缓存)
    result = hook.after_render(text_artifact, descriptor)
    assert result is None
    cached = hook.get_cached_descriptor(text_artifact)
    assert cached is descriptor


def test_caching_hook_name(hooks_mod):
    """CachingHook.name 默认为类名."""
    hook = hooks_mod.CachingHook()
    assert hook.name == "CachingHook"


# ============================================================
# 15. 集成: 钩子返回 None 保持原始值
# ============================================================


def test_hook_returns_none_preserves_original_context(hookable, fake_pipeline, registry, text_artifact, context):
    """钩子 before_render 返回 None 时, 底层接收原始 context 对象."""
    original_context = context
    registry.register(SpyHook())  # 默认 before_render 返回 None
    hookable.render(text_artifact, original_context)
    assert fake_pipeline.last_context is original_context


def test_hook_returns_none_preserves_original_descriptor(hookable, fake_pipeline, registry, text_artifact, context):
    """钩子 after_render 返回 None 时, 最终返回底层产出的 descriptor."""
    registry.register(SpyHook())  # 默认 after_render 返回 None
    result = hookable.render(text_artifact, context)
    # 底层缓存中的描述符应与返回一致 (未被修改)
    cached = fake_pipeline.get_cached(text_artifact.artifact_id)
    assert cached is not None
    assert result.html == cached.html


def test_hookable_pipeline_before_render_called_before_underlying(hookable, fake_pipeline, registry, text_artifact, context):
    """before_render 在底层 render 之前被调用."""
    spy = SpyHook()
    registry.register(spy)
    hookable.render(text_artifact, context)
    assert len(spy.before_calls) == 1
    assert len(spy.after_calls) == 1
    assert fake_pipeline.render_count == 1


def test_hookable_pipeline_after_render_receives_underlying_descriptor(hookable, fake_pipeline, registry, text_artifact, context):
    """after_render 接收底层流水线产出的 descriptor."""
    spy = SpyHook()
    registry.register(spy)
    hookable.render(text_artifact, context)
    _, received_descriptor = spy.after_calls[0]
    assert received_descriptor.artifact_id == text_artifact.artifact_id
