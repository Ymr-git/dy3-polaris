"""L7 体验呈现层 — 渲染器注册中心 (RendererRegistry).

按 MIME type 将 Artifact 路由到对应的原生渲染器。支持两种注册方式:

1. **即时注册** (eager): ``register(renderer)`` — 直接注册渲染器实例，
   自动通过 ``renderer.supported_mime_types()`` 探测其支持的全部 MIME。
2. **延迟注册** (lazy / factory): ``register_factory(mime_type, factory)`` —
   注册一个工厂函数，实例在首次 ``get_renderer()`` 时才创建并缓存
   (借鉴 Service Locator / Inversion of Control 模式)。

设计要点:
    - 一个渲染器可声明支持多个 MIME 类型 (MIME Bundle 模型)。
    - 同一渲染器实例通过其任一支持的 MIME 取回时返回同一实例。
    - 所有公开方法线程安全 (``threading.Lock``)。
    - 未匹配到渲染器时抛出 ``RendererNotFoundError``。

架构:
    RendererRegistry
    ├── _renderers: dict[str, IRenderer]              # MIME -> 渲染器实例
    └── _factories: dict[str, Callable[[], IRenderer]] # MIME -> 延迟工厂
"""

from __future__ import annotations

import threading
from typing import Callable

from .exceptions import RendererNotFoundError
from .irenderer import IRenderer
from .models import Artifact


class RendererRegistry:
    """渲染器注册中心 — 按 MIME type 路由 Artifact 到对应渲染器.

    使用示例::

        registry = RendererRegistry()

        # 即时注册
        registry.register(MarkdownRenderer())

        # 延迟注册 (工厂模式)
        registry.register_factory("application/vnd.echarts+json", ChartRenderer)

        # 路由
        renderer = registry.get_renderer(artifact.mime)
        descriptor = renderer.render(artifact, context)
    """

    def __init__(self) -> None:
        # MIME -> 渲染器实例 (已实例化)
        self._renderers: dict[str, IRenderer] = {}
        # MIME -> 延迟工厂 (尚未实例化)
        self._factories: dict[str, Callable[[], IRenderer]] = {}
        # 并发控制
        self._lock = threading.Lock()

    # ============================================================
    # 注册
    # ============================================================

    def register(self, renderer: IRenderer) -> None:
        """注册一个渲染器实例，自动探测其支持的全部 MIME 类型.

        Args:
            renderer: 实现 IRenderer 接口的渲染器实例。

        Note:
            - 通过 ``renderer.supported_mime_types()`` 自动获取支持的 MIME 列表。
            - 该渲染器支持的每个 MIME 都会被映射到此实例。
            - 若某 MIME 此前注册过延迟工厂，将被覆盖 (即时实例优先)。
        """
        mimes = renderer.supported_mime_types()
        with self._lock:
            for mime in mimes:
                self._renderers[mime] = renderer
                # 即时实例优先，移除可能存在的延迟工厂
                self._factories.pop(mime, None)

    def register_factory(
        self,
        mime_type: str,
        factory: Callable[[], IRenderer],
    ) -> None:
        """注册一个延迟工厂 — 渲染器实例在首次 ``get_renderer()`` 时才创建.

        Args:
            mime_type: 工厂负责的 MIME 类型。
            factory: 无参可调用对象，调用返回一个 IRenderer 实例。

        Note:
            - 注册工厂时尚不会调用 factory，实例化推迟到首次获取。
            - 实例化后，渲染器声明支持的全部 MIME 都会被缓存到该实例。
        """
        with self._lock:
            self._factories[mime_type] = factory

    # ============================================================
    # 注销
    # ============================================================

    def unregister(self, mime_type: str) -> None:
        """移除指定 MIME 对应的渲染器 (即时实例或延迟工厂).

        Args:
            mime_type: 要注销的 MIME 类型。

        Note:
            - 仅移除该单一 MIME 的映射，同一渲染器的其他 MIME 不受影响。
            - 注销不存在的 MIME 是安全的 (幂等，不抛异常)。
        """
        with self._lock:
            self._renderers.pop(mime_type, None)
            self._factories.pop(mime_type, None)

    # ============================================================
    # 获取
    # ============================================================

    def get_renderer(self, mime_type: str) -> IRenderer:
        """获取指定 MIME 对应的渲染器.

        若该 MIME 仅注册了延迟工厂，则在首次调用时实例化并缓存。

        Args:
            mime_type: 目标 MIME 类型。

        Returns:
            对应的 IRenderer 实例。

        Raises:
            RendererNotFoundError: 没有任何渲染器或工厂处理该 MIME 类型。
        """
        with self._lock:
            # 1. 已有实例 — 直接返回
            instance = self._renderers.get(mime_type)
            if instance is not None:
                return instance

            # 2. 延迟工厂 — 首次实例化并缓存
            factory = self._factories.pop(mime_type, None)
            if factory is not None:
                renderer = factory()
                # 缓存到该渲染器声明支持的全部 MIME
                for mt in renderer.supported_mime_types():
                    self._renderers[mt] = renderer
                    self._factories.pop(mt, None)
                # 确保请求的 MIME 一定命中 (即便渲染器未显式声明)
                self._renderers.setdefault(mime_type, renderer)
                return self._renderers[mime_type]

            # 3. 未注册
            raise RendererNotFoundError(mime_type)

    def get_renderer_for_artifact(self, artifact: Artifact) -> IRenderer:
        """按 Artifact 的 mime 字段路由到对应渲染器 (便捷方法).

        Args:
            artifact: 待渲染的制品。

        Returns:
            对应的 IRenderer 实例。

        Raises:
            RendererNotFoundError: 没有渲染器处理 artifact.mime。
        """
        return self.get_renderer(artifact.mime)

    # ============================================================
    # 查询
    # ============================================================

    def is_supported(self, mime_type: str) -> bool:
        """检查指定 MIME 是否被任何渲染器 (实例或工厂) 支持."""
        with self._lock:
            return mime_type in self._renderers or mime_type in self._factories

    def list_mime_types(self) -> list[str]:
        """返回所有已注册的 MIME 类型 (含延迟工厂注册的)."""
        with self._lock:
            return list({*self._renderers.keys(), *self._factories.keys()})

    def list_renderers(self) -> list[str]:
        """返回所有已实例化渲染器的类名列表 (去重，保持首次注册顺序)."""
        with self._lock:
            seen: set[str] = set()
            names: list[str] = []
            for renderer in self._renderers.values():
                name = type(renderer).__name__
                if name not in seen:
                    seen.add(name)
                    names.append(name)
            return names

    # ============================================================
    # 维护
    # ============================================================

    def clear(self) -> None:
        """清空所有渲染器与延迟工厂 (幂等)."""
        with self._lock:
            self._renderers.clear()
            self._factories.clear()

    # ============================================================
    # 属性
    # ============================================================

    @property
    def size(self) -> int:
        """已注册的 MIME 类型总数 (含延迟工厂)."""
        with self._lock:
            return len({*self._renderers.keys(), *self._factories.keys()})


# ============================================================
# 全局注册中心单例
# ============================================================

_global_registry: RendererRegistry | None = None
_global_lock = threading.Lock()


def get_registry() -> RendererRegistry:
    """获取全局 RendererRegistry 单例."""
    global _global_registry
    if _global_registry is None:
        with _global_lock:
            if _global_registry is None:
                _global_registry = RendererRegistry()
    return _global_registry


def reset_registry() -> None:
    """重置全局注册中心 (仅用于测试)."""
    global _global_registry
    with _global_lock:
        _global_registry = None
