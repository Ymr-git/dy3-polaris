"""L7 体验呈现层 — 渲染器接口 (IRenderer).

定义所有原生渲染器的统一抽象基类。IRenderer 是 L7 层的核心契约：
RendererRegistry 通过 ``supported_mime_types()`` 将 Artifact 路由到对应渲染器，
渲染器负责将 Artifact 转换为前端可消费的 RenderDescriptor，并支持增量更新与资源回收。

融合世界先进方案:
- Jupyter nbformat: MIME Bundle 渲染模型 (一个渲染器声明自己支持的 MIME 集合)
- VS Code CustomEditor: 统一编辑器接口 (render / update / destroy 生命周期)
- React Server Components: 增量更新 + 资源管理 (update 接收 ArtifactDiff)

生命周期:
    register → render(artifact, context) → update(diff)* → destroy()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Artifact, ArtifactDiff, RenderContext, RenderDescriptor


class IRenderer(ABC):
    """渲染器接口 — 所有原生渲染器的统一抽象基类.

    子类必须实现以下四个抽象方法:

    - ``render(artifact, context) -> RenderDescriptor``:
        将 Artifact 渲染为前端可消费的渲染描述。
    - ``update(diff) -> RenderDescriptor``:
        基于增量差异 (ArtifactDiff) 更新已有渲染。
    - ``destroy() -> None``:
        销毁渲染实例，释放资源 (应幂等)。
    - ``supported_mime_types() -> list[str]``:
        返回该 Renderer 支持的 MIME type 列表，供 RendererRegistry 路由使用。

    融合方案:
        - Jupyter nbformat: MIME Bundle 渲染模型
        - VS Code CustomEditor: 统一编辑器接口
        - React Server Components: 增量更新 + 资源管理
    """

    @abstractmethod
    def render(self, artifact: Artifact, context: RenderContext) -> RenderDescriptor:
        """将 Artifact 渲染为前端可消费的渲染描述.

        Args:
            artifact: 待渲染的制品 (携带类型、MIME、载荷数据)。
            context: 渲染上下文 (视口、主题、学习者状态等)。

        Returns:
            RenderDescriptor: 前端可直接消费的渲染描述符。
        """
        ...

    @abstractmethod
    def update(self, diff: ArtifactDiff) -> RenderDescriptor:
        """基于增量差异更新已有渲染.

        借鉴 React Server Components 的增量更新模型，避免全量重渲染。

        Args:
            diff: Artifact 增量差异 (JSON-Patch 风格操作列表)。

        Returns:
            RenderDescriptor: 更新后的渲染描述符。
        """
        ...

    @abstractmethod
    def destroy(self) -> None:
        """销毁渲染实例，释放资源.

        应当幂等 — 多次调用不应抛出异常。
        """
        ...

    @abstractmethod
    def supported_mime_types(self) -> list[str]:
        """返回该 Renderer 支持的 MIME type 列表.

        RendererRegistry 据此将 Artifact 按 MIME 路由到对应渲染器。
        一个渲染器可声明支持多个 MIME 类型。

        Returns:
            支持的 MIME type 字符串列表 (应为独立副本)。
        """
        ...
