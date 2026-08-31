"""L7 体验呈现层 — 渲染钩子 / 中间件系统 (Render Hooks / Middleware).

为渲染流水线 (RenderPipeline / _FallbackPipeline) 提供可插拔的前置/后置处理器链,
允许在不修改渲染器核心逻辑的前提下注入横切关注点: 日志、计时、缓存、降级、监控等。

融合世界先进方案:
- **Express.js middleware** — 链式处理器, 按注册顺序依次执行
- **Django middleware** — process_request / process_response 配对模式
- **Starlette middleware** — ASGI 中间件, 包装底层应用
- **React Suspense** — 降级 (fallback) 处理 (on_render_error)

核心组件:
1. ``RenderHook`` (Protocol) — 渲染钩子协议, 定义三个扩展点:
   - ``before_render``: 渲染前, 可修改并返回新的 RenderContext (返回 None 保持原样)
   - ``after_render``:  渲染后, 可修改并返回新的 RenderDescriptor (返回 None 保持原样)
   - ``on_render_error``: 渲染失败时被调用 (仅记录/降级, 不吞异常)

2. ``BaseRenderHook`` — 提供 no-op 默认实现的基类, 子类按需覆盖;
   ``name`` 属性默认返回类名。

3. ``HookRegistry`` — 线程安全的钩子注册中心, 支持优先级排序
   (数值越小越先执行), register 幂等 (重复注册同一对象时更新优先级)。

4. ``HookablePipeline`` — 包装任意渲染流水线 (RenderPipeline /
   _FallbackPipeline), 在 render 前后织入钩子链, update/destroy 等方法透传委托。

5. 内置钩子:
   - ``LoggingHook``  — 渲染前后/错误日志
   - ``TimingHook``   — 渲染耗时统计 (get_average_time / get_total_renders)
   - ``CachingHook``  — 基于 (artifact_id, version) 的内存缓存, 命中时短路渲染

渲染流程 (HookablePipeline.render):
    before_render(hook₁ → hook₂ → ...)   # 链式修改 context
        ↓
    [缓存命中检查]                          # CachingHook 等可短路
        ↓
    pipeline.render(artifact, context)    # 底层渲染 (或使用缓存描述符)
        ├─ 成功 → after_render(hook₁ → hook₂ → ...)  # 链式修改 descriptor
        └─ 失败 → on_render_error(所有钩子) → 重新抛出原始异常

线程安全: ``HookRegistry`` 使用 ``threading.Lock`` 保护; ``HookablePipeline``
本身无内部可变状态, 渲染期间多次读取注册表快照。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol, runtime_checkable

from .models import Artifact, RenderContext, RenderDescriptor

_logger = logging.getLogger("dy3_polaris.l7.hooks")


# ============================================================
# RenderHook 协议
# ============================================================


@runtime_checkable
class RenderHook(Protocol):
    """渲染钩子协议 — 渲染流程的可插拔扩展点.

    实现者可选择性地覆盖三个方法; 通过鸭子类型 (Protocol) 即可被
    ``HookRegistry`` 接受, 不强制继承 ``BaseRenderHook``。

    方法语义:
        - ``before_render``: 渲染前调用。返回新的 RenderContext 以替换渲染上下文,
          返回 None 表示保持原始上下文不变。
        - ``after_render``: 渲染成功后调用。返回新的 RenderDescriptor 以替换渲染结果,
          返回 None 表示保持原始描述符不变。
        - ``on_render_error``: 渲染抛出异常时调用。仅用于记录/降级,
          异常仍会向上传播 (由 HookablePipeline 负责重新抛出)。
    """

    def before_render(
        self, artifact: Artifact, context: RenderContext
    ) -> RenderContext | None:
        """渲染前调用 — 可修改并返回新的 RenderContext, 或返回 None 保持原样.

        Args:
            artifact: 待渲染的制品。
            context: 当前渲染上下文。

        Returns:
            新的 RenderContext (替换), 或 None (保持原始 context)。
        """
        ...

    def after_render(
        self, artifact: Artifact, descriptor: RenderDescriptor
    ) -> RenderDescriptor | None:
        """渲染成功后调用 — 可修改并返回新的 RenderDescriptor, 或返回 None 保持原样.

        Args:
            artifact: 已渲染的制品。
            descriptor: 底层产出的渲染描述符。

        Returns:
            新的 RenderDescriptor (替换), 或 None (保持原始 descriptor)。
        """
        ...

    def on_render_error(self, artifact: Artifact, error: Exception) -> None:
        """渲染失败时调用 — 用于记录日志 / 降级处理 / 指标上报.

        异常本身不会被吞没; HookablePipeline 在调用所有钩子的
        on_render_error 后会重新抛出原始异常。

        Args:
            artifact: 渲染失败的制品。
            error: 底层抛出的异常。
        """
        ...


# ============================================================
# BaseRenderHook — no-op 默认基类
# ============================================================


class BaseRenderHook:
    """渲染钩子基类 — 提供 no-op 默认实现.

    子类只需覆盖感兴趣的方法, 其余保持默认空操作。
    ``name`` 属性默认返回类名, 子类可覆盖以提供自定义名称。

    通过鸭子类型满足 ``RenderHook`` 协议 (无需显式继承 Protocol)。
    """

    @property
    def name(self) -> str:
        """钩子名称 — 默认返回类名."""
        return type(self).__name__

    def before_render(
        self, artifact: Artifact, context: RenderContext
    ) -> RenderContext | None:
        """默认 no-op — 返回 None, 保持原始 context."""
        return None

    def after_render(
        self, artifact: Artifact, descriptor: RenderDescriptor
    ) -> RenderDescriptor | None:
        """默认 no-op — 返回 None, 保持原始 descriptor."""
        return None

    def on_render_error(self, artifact: Artifact, error: Exception) -> None:
        """默认 no-op — 不抛异常, 不做任何处理."""
        return None


# ============================================================
# HookRegistry — 线程安全的钩子注册中心
# ============================================================


class HookRegistry:
    """线程安全的渲染钩子注册中心.

    维护钩子列表并按优先级排序。优先级数值越小越先执行;
    相同优先级保持注册顺序 (稳定排序)。

    ``register`` 对同一钩子对象幂等: 重复注册时更新其优先级而非新增副本,
    避免同一钩子被多次执行。

    线程安全: 所有公开方法使用 ``threading.Lock`` 保护, 可安全用于
    并发注册 / 注销场景。

    Attributes:
        _hooks: 内部钩子存储, 元组列表 ``[(priority, seq, hook), ...]``。
            ``seq`` 为单调递增的注册序号, 用于稳定排序。
        _lock: 线程锁, 保护并发访问。
        _seq: 单调递增序号计数器。
    """

    def __init__(self) -> None:
        """初始化空的钩子注册中心."""
        # (priority, seq, hook) — seq 保证同优先级时稳定排序
        self._hooks: list[tuple[int, int, RenderHook]] = []
        self._lock = threading.Lock()
        self._seq = 0

    def register(self, hook: RenderHook, priority: int = 0) -> None:
        """注册钩子, 指定优先级 (数值越小越先执行).

        对同一钩子对象幂等: 若已注册, 则更新其优先级, 不新增副本。

        Args:
            hook: 待注册的钩子 (实现 RenderHook 协议)。
            priority: 优先级, 默认 0。数值越小越先执行。
        """
        with self._lock:
            # 幂等: 已存在则更新优先级 (保留原注册序号)
            for idx, (_p, seq, h) in enumerate(self._hooks):
                if h is hook:
                    self._hooks[idx] = (priority, seq, hook)
                    return
            # 新增
            self._seq += 1
            self._hooks.append((priority, self._seq, hook))

    def unregister(self, hook: RenderHook) -> None:
        """注销钩子。若未注册则安全无操作 (不抛异常).

        Args:
            hook: 待注销的钩子。
        """
        with self._lock:
            self._hooks = [
                (p, seq, h) for (p, seq, h) in self._hooks if h is not hook
            ]

    def clear(self) -> None:
        """移除所有已注册的钩子."""
        with self._lock:
            self._hooks.clear()

    def get_hooks(self) -> list[RenderHook]:
        """返回按优先级排序的钩子列表 (副本).

        排序规则: 优先级数值升序; 相同优先级按注册顺序 (seq 升序)。

        Returns:
            排序后的钩子列表。返回的是新列表, 修改不影响注册中心内部状态。
        """
        with self._lock:
            sorted_entries = sorted(self._hooks, key=lambda entry: (entry[0], entry[1]))
            return [hook for (_p, _seq, hook) in sorted_entries]

    def count(self) -> int:
        """返回已注册的钩子数量.

        Returns:
            钩子数量。
        """
        with self._lock:
            return len(self._hooks)


# ============================================================
# HookablePipeline — 可钩子渲染流水线包装器
# ============================================================


class HookablePipeline:
    """渲染流水线包装器 — 在 render 前后织入钩子链.

    包装任意实现了渲染流水线接口的对象 (RenderPipeline 或 _FallbackPipeline),
    在 ``render`` 操作前后执行注册的钩子, 其余方法 (update / destroy /
    get_cached / clear_cache / get_stats) 直接透传委托。

    借鉴方案:
        - Starlette middleware: ASGI 中间件包装底层应用
        - Django middleware: process_request / process_response 配对
        - Express.js middleware: 链式 next() 调用

    渲染流程 (``render``):
        1. 依次执行所有钩子的 ``before_render``:
           - 若返回非 None, 用返回值替换当前 context (链式累积修改)。
        2. 缓存命中检查 (鸭子类型):
           - 若某钩子暴露 ``get_cached_descriptor(artifact)`` 且返回非 None,
             则使用该缓存描述符, 跳过底层渲染 (短路)。
        3. 若未命中缓存, 调用底层 ``pipeline.render(artifact, context, **kwargs)``。
        4. 渲染成功: 依次执行所有钩子的 ``after_render``:
           - 若返回非 None, 用返回值替换当前 descriptor (链式累积修改)。
        5. 渲染失败: 执行所有钩子的 ``on_render_error`` (单个钩子异常不中断
           其余), 然后重新抛出原始异常。
        6. 返回最终 descriptor。

    Attributes:
        _pipeline: 被包装的底层渲染流水线。
        _registry: 钩子注册中心。
    """

    def __init__(self, pipeline: Any, hook_registry: HookRegistry) -> None:
        """初始化 HookablePipeline.

        Args:
            pipeline: 被包装的渲染流水线 (RenderPipeline / _FallbackPipeline 等)。
            hook_registry: 钩子注册中心。
        """
        self._pipeline = pipeline
        self._registry = hook_registry

    # ------------------------------------------------------------
    # 渲染 (织入钩子链)
    # ------------------------------------------------------------

    def render(
        self,
        artifact: Artifact,
        context: RenderContext,
        **kwargs: Any,
    ) -> RenderDescriptor:
        """渲染 Artifact, 在前后织入钩子链.

        Args:
            artifact: 待渲染的制品。
            context: 渲染上下文。
            **kwargs: 透传给底层流水线 render 的额外参数。

        Returns:
            经过 after_render 钩子链处理后的最终 RenderDescriptor。

        Raises:
            Exception: 底层渲染抛出的原始异常 (在调用 on_render_error 后重新抛出)。
        """
        hooks = self._registry.get_hooks()

        # 1. before_render — 链式修改 context
        current_context: RenderContext = context
        for hook in hooks:
            result = hook.before_render(artifact, current_context)
            if result is not None:
                current_context = result

        # 2. 缓存命中检查 (鸭子类型: CachingHook 等可短路)
        descriptor: RenderDescriptor | None = None
        for hook in hooks:
            getter = getattr(hook, "get_cached_descriptor", None)
            if callable(getter):
                cached = getter(artifact)
                if cached is not None:
                    descriptor = cached
                    break

        # 3. 渲染 (或使用缓存描述符)
        try:
            if descriptor is None:
                descriptor = self._pipeline.render(artifact, current_context, **kwargs)
        except Exception as exc:
            # 5. on_render_error — 所有钩子都执行, 单个钩子异常不中断其余
            for hook in hooks:
                try:
                    hook.on_render_error(artifact, exc)
                except Exception:  # pragma: no cover
                    hook_name = getattr(hook, "name", repr(hook))
                    _logger.warning(
                        "on_render_error hook %s raised an exception",
                        hook_name,
                        exc_info=True,
                    )
            # 重新抛出原始异常 (不吞没)
            raise

        # 4. after_render — 链式修改 descriptor
        #    descriptor 此时一定非 None (渲染成功或缓存命中)
        for hook in hooks:
            result = hook.after_render(artifact, descriptor)
            if result is not None:
                descriptor = result

        # 6. 返回最终 descriptor
        assert descriptor is not None  # 渲染成功必定产出描述符
        return descriptor

    # ------------------------------------------------------------
    # 委托方法 (不织入钩子)
    # ------------------------------------------------------------

    def update(self, artifact_id: str, diff: Any, **kwargs: Any) -> RenderDescriptor:
        """委托底层流水线的 update (增量更新, 暂不织入钩子).

        Args:
            artifact_id: Artifact ID。
            diff: Artifact 增量差异。
            **kwargs: 透传给底层流水线的额外参数。

        Returns:
            更新后的 RenderDescriptor。
        """
        return self._pipeline.update(artifact_id, diff, **kwargs)

    def destroy(self, artifact_id: str, **kwargs: Any) -> None:
        """委托底层流水线的 destroy (释放资源, 幂等).

        Args:
            artifact_id: Artifact ID。
            **kwargs: 透传给底层流水线的额外参数。
        """
        return self._pipeline.destroy(artifact_id, **kwargs)

    def get_cached(self, artifact_id: str) -> RenderDescriptor | None:
        """委托底层流水线的 get_cached.

        Args:
            artifact_id: Artifact ID。

        Returns:
            缓存的 RenderDescriptor, 未缓存返回 None。
        """
        return self._pipeline.get_cached(artifact_id)

    def clear_cache(self) -> None:
        """委托底层流水线的 clear_cache (清空缓存)."""
        return self._pipeline.clear_cache()

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息 — 合并底层流水线统计并附加 hook_count.

        Returns:
            包含底层统计字段及 ``hook_count`` (已注册钩子数量) 的字典。
        """
        stats: dict[str, Any] = dict(self._pipeline.get_stats())
        stats["hook_count"] = self._registry.count()
        return stats


# ============================================================
# 内置钩子: LoggingHook
# ============================================================


class LoggingHook(BaseRenderHook):
    """日志钩子 — 在渲染前后及错误时输出结构化日志.

    借鉴 Express.js middleware 的请求/响应日志模式,
    记录 artifact_id、mime 与 render_time_ms, 便于追踪与调试。

    Args:
        logger: 自定义 logger; 默认使用 ``dy3_polaris.l7.hooks.logging``。
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger if logger is not None else logging.getLogger(
            "dy3_polaris.l7.hooks.logging"
        )

    def before_render(
        self, artifact: Artifact, context: RenderContext
    ) -> RenderContext | None:
        """渲染前输出 INFO 日志 (artifact_id, mime)."""
        self._logger.info(
            "before_render: artifact_id=%s, mime=%s",
            artifact.artifact_id,
            artifact.mime,
        )
        return None

    def after_render(
        self, artifact: Artifact, descriptor: RenderDescriptor
    ) -> RenderDescriptor | None:
        """渲染后输出 INFO 日志 (artifact_id, mime, render_time_ms)."""
        self._logger.info(
            "after_render: artifact_id=%s, mime=%s, render_time_ms=%.2f",
            artifact.artifact_id,
            descriptor.mime,
            descriptor.render_time_ms,
        )
        return None

    def on_render_error(self, artifact: Artifact, error: Exception) -> None:
        """渲染失败时输出 ERROR 日志 (artifact_id, error)."""
        self._logger.error(
            "render_error: artifact_id=%s, error=%s",
            artifact.artifact_id,
            error,
        )


# ============================================================
# 内置钩子: TimingHook
# ============================================================


class TimingHook(BaseRenderHook):
    """计时钩子 — 记录每次渲染的 render_time_ms, 暴露统计接口.

    在 ``after_render`` 中收集 ``descriptor.render_time_ms``,
    通过 ``get_average_time()`` / ``get_total_renders()`` 查询。

    借鉴 Grafana plugin 的渲染耗时指标采集。
    """

    def __init__(self) -> None:
        self._durations: list[float] = []

    def after_render(
        self, artifact: Artifact, descriptor: RenderDescriptor
    ) -> RenderDescriptor | None:
        """记录本次渲染耗时."""
        self._durations.append(descriptor.render_time_ms)
        return None

    def get_average_time(self) -> float:
        """返回平均渲染耗时 (毫秒).

        无记录时返回 0.0。
        """
        if not self._durations:
            return 0.0
        return sum(self._durations) / len(self._durations)

    def get_total_renders(self) -> int:
        """返回已记录的渲染次数."""
        return len(self._durations)


# ============================================================
# 内置钩子: CachingHook
# ============================================================


class CachingHook(BaseRenderHook):
    """缓存钩子 — 基于 (artifact_id, version) 的内存缓存, 命中时短路渲染.

    工作原理:
        - ``after_render``: 将渲染产出的 descriptor 缓存, 键为
          ``(artifact_id, version)``。返回 None (不修改描述符)。
        - ``get_cached_descriptor(artifact)``: 查询缓存, 命中返回描述符,
          未命中返回 None。``HookablePipeline.render`` 会在调用底层渲染前
          通过鸭子类型查询此方法实现短路。

    版本感知: 缓存键包含 artifact.version, 版本变化时缓存未命中触发重新渲染。

    注意: 此钩子仅在 ``HookablePipeline`` 内生效; 直接调用底层流水线
    不经过此缓存。
    """

    def __init__(self) -> None:
        # (artifact_id, version) -> RenderDescriptor
        self._cache: dict[tuple[str, int], RenderDescriptor] = {}

    @staticmethod
    def _key(artifact: Artifact) -> tuple[str, int]:
        """生成缓存键 — (artifact_id, version)."""
        return (artifact.artifact_id, artifact.version)

    def before_render(
        self, artifact: Artifact, context: RenderContext
    ) -> RenderContext | None:
        """默认 no-op — 缓存命中由 ``get_cached_descriptor`` 负责."""
        return None

    def after_render(
        self, artifact: Artifact, descriptor: RenderDescriptor
    ) -> RenderDescriptor | None:
        """缓存渲染产出的描述符 (键为 artifact_id + version), 返回 None."""
        self._cache[self._key(artifact)] = descriptor
        return None

    def get_cached_descriptor(self, artifact: Artifact) -> RenderDescriptor | None:
        """查询缓存, 命中返回描述符, 未命中返回 None.

        被 ``HookablePipeline.render`` 通过鸭子类型调用以实现短路。

        Args:
            artifact: 待渲染的制品。

        Returns:
            缓存的 RenderDescriptor, 或 None。
        """
        return self._cache.get(self._key(artifact))

    def clear(self) -> None:
        """清空缓存."""
        self._cache.clear()

    def cache_size(self) -> int:
        """返回当前缓存条目数."""
        return len(self._cache)


__all__ = [
    "RenderHook",
    "BaseRenderHook",
    "HookRegistry",
    "HookablePipeline",
    "LoggingHook",
    "TimingHook",
    "CachingHook",
]
