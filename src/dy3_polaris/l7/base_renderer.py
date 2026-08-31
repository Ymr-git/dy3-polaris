"""L7 体验呈现层 — 渲染器基类与降级渲染器.

本模块提供两个核心组件:

1. **BaseRenderer** — 模板方法模式的抽象基类，继承 IRenderer。
   将渲染流程固化为 ``校验 → 预处理 → do_render → 后处理 → 构建描述符``，
   子类只需实现 ``do_render()`` 即可，其余通用逻辑由基类提供。
   同时提供 ``_validate_artifact()`` / ``_build_descriptor()`` 等通用方法，
   以及 ``_preprocess()`` / ``_postprocess()`` 钩子方法 (默认空实现)。

2. **FallbackRenderer** — 未知 MIME 类型的降级渲染器。
   当 RendererRegistry 中找不到对应 MIME 的渲染器时，使用此渲染器进行降级处理。
   将任意 Artifact 的 payload 转换为可读的 ``<pre>`` 文本块。

设计模式:
    - Template Method (模板方法): BaseRenderer.render() 定义算法骨架，
      do_render() 为抽象原语操作，由子类实现。
    - Hook Methods (钩子方法): _preprocess / _postprocess 提供扩展点，
      子类可选择性覆盖以插入自定义逻辑。
    - Null Object / Fallback (降级对象): FallbackRenderer 作为"找不到渲染器"
      时的兜底实现，保证系统永不因未知 MIME 而崩溃。

融合世界先进方案:
    - Jupyter nbformat: MIME Bundle 渲染模型
    - VS Code CustomEditor: 统一编辑器生命周期 (render / update / destroy)
    - React Server Components: 增量更新 + 资源管理
    - Django CBV / Spring Template Method: 钩子方法扩展点

生命周期:
    register → render(artifact, context) → update(diff)* → destroy()
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod

from .exceptions import ArtifactValidationError
from .irenderer import IRenderer
from .models import Artifact, ArtifactDiff, RenderContext, RenderDescriptor


class BaseRenderer(IRenderer, ABC):
    """渲染器基类 — 模板方法模式.

    继承 IRenderer 并提供通用渲染流程的骨架实现。子类只需实现
    ``do_render()`` 抽象方法即可获得完整的渲染能力。

    提供的能力:
        - ``render()`` 模板方法: 校验 → 预处理 → do_render → 后处理 → 构建描述符
        - ``do_render()`` 抽象方法: 子类实现具体渲染逻辑，返回 HTML 字符串
        - ``_build_descriptor()`` 通用描述符构建
        - ``_validate_artifact()`` 通用校验 (非空 + payload 非空)
        - ``_preprocess()`` / ``_postprocess()`` 钩子方法 (默认空实现)
        - ``supported_mime_types()`` 返回类属性 ``_MIME_TYPES`` 的副本
        - ``update()`` 默认实现: 不支持增量更新 (抛 NotImplementedError)
        - ``destroy()`` 默认实现: 空操作

    Attributes:
        _MIME_TYPES: 该渲染器支持的 MIME 类型列表，子类应覆盖此属性。
                     默认为空列表。
    """

    _MIME_TYPES: list[str] = []

    def render(self, artifact: Artifact, context: RenderContext) -> RenderDescriptor:
        """模板方法 — 渲染 Artifact 为 RenderDescriptor.

        执行流程 (模板方法模式):
            1. ``_validate_artifact(artifact)`` — 校验 artifact 非空、payload 非空
            2. ``_preprocess(artifact, context)`` — 预处理钩子 (子类可覆盖)
            3. ``do_render(preprocessed, context)`` — 子类实现的具体渲染逻辑
            4. ``_postprocess(html, context)`` — 后处理钩子 (子类可覆盖)
            5. ``_build_descriptor(artifact, html)`` — 构建最终 RenderDescriptor

        Args:
            artifact: 待渲染的制品 (携带类型、MIME、载荷数据)。
            context: 渲染上下文 (视口、主题、学习者状态等)。

        Returns:
            RenderDescriptor: 前端可直接消费的渲染描述符。

        Raises:
            ArtifactValidationError: artifact 为 None 或 payload 为空时抛出。
        """
        # 1. 校验
        self._validate_artifact(artifact)
        # 2. 预处理 (钩子，默认返回原 artifact)
        preprocessed = self._preprocess(artifact, context)
        # 3. 具体渲染 (子类实现)
        html = self.do_render(preprocessed, context)
        # 4. 后处理 (钩子，默认返回原 html)
        html = self._postprocess(html, context)
        # 5. 构建描述符
        return self._build_descriptor(artifact, html)

    @abstractmethod
    def do_render(self, artifact: Artifact, context: RenderContext) -> str:
        """子类实现具体渲染逻辑，返回 HTML 字符串.

        这是模板方法 ``render()`` 中的抽象原语操作，子类必须实现。
        接收的 artifact 已经过 ``_preprocess()`` 预处理。

        Args:
            artifact: 预处理后的制品 (已通过校验)。
            context: 渲染上下文。

        Returns:
            渲染产生的 HTML 字符串 (将作为 RenderDescriptor.html)。
        """
        ...

    def update(self, diff: ArtifactDiff) -> RenderDescriptor:
        """基于增量差异更新已有渲染 — 默认不支持增量更新.

        默认实现抛出 ``NotImplementedError``，表示该渲染器不支持增量更新，
        调用方应回退到全量重渲染 (``render()``)。

        子类如需支持增量更新，应覆盖此方法。

        Args:
            diff: Artifact 增量差异 (JSON-Patch 风格操作列表)。

        Returns:
            RenderDescriptor: 更新后的渲染描述符。

        Raises:
            NotImplementedError: 默认始终抛出 (子类可覆盖)。
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support incremental update"
        )

    def destroy(self) -> None:
        """销毁渲染实例，释放资源 — 默认空操作.

        默认实现为空操作 (幂等)。子类如需释放资源 (如关闭连接、清理缓存)，
        应覆盖此方法。覆盖实现应保持幂等性 — 多次调用不应抛出异常。
        """
        pass

    def supported_mime_types(self) -> list[str]:
        """返回该 Renderer 支持的 MIME type 列表.

        返回类属性 ``_MIME_TYPES`` 的独立副本，修改返回值不影响内部状态。

        Returns:
            支持的 MIME type 字符串列表 (独立副本)。
        """
        return list(self._MIME_TYPES)

    # ============================================================
    # 内部方法 (子类可覆盖)
    # ============================================================

    def _validate_artifact(self, artifact: Artifact) -> None:
        """通用校验 — 检查 artifact 非空、payload 非空.

        Args:
            artifact: 待校验的制品。

        Raises:
            ArtifactValidationError: artifact 为 None 或 payload 为空字典时抛出。
        """
        if artifact is None:
            raise ArtifactValidationError(
                field="artifact",
                detail="Artifact is None",
            )
        if not artifact.payload:
            raise ArtifactValidationError(
                field="payload",
                detail="Payload is empty",
            )

    def _preprocess(self, artifact: Artifact, context: RenderContext) -> Artifact:
        """预处理钩子 — 默认返回原 artifact，不做任何修改.

        子类可覆盖此方法以在渲染前对 artifact 进行转换 (如数据清洗、
        字段补全、根据 context 适配等)。

        Args:
            artifact: 原始制品 (已通过校验)。
            context: 渲染上下文。

        Returns:
            预处理后的 Artifact (默认为原对象)。
        """
        return artifact

    def _postprocess(self, html: str, context: RenderContext) -> str:
        """后处理钩子 — 默认返回原 html，不做任何修改.

        子类可覆盖此方法以对渲染产生的 HTML 进行后处理 (如注入资源引用、
        添加 wrapper、根据 context 调整样式等)。

        Args:
            html: do_render 产生的 HTML 字符串。
            context: 渲染上下文。

        Returns:
            后处理后的 HTML 字符串 (默认为原字符串)。
        """
        return html

    def _build_descriptor(
        self, artifact: Artifact, html: str
    ) -> RenderDescriptor:
        """构建 RenderDescriptor — 通用描述符构建.

        将渲染产生的 HTML 与 artifact 信息组装为最终的 RenderDescriptor。

        Args:
            artifact: 原始制品 (用于提取 artifact_id 和 mime)。
            html: 最终的 HTML 字符串 (已过后处理)。

        Returns:
            RenderDescriptor: 包含 HTML、元数据和时间戳的渲染描述符。
        """
        return RenderDescriptor(
            render_id=f"rd-{uuid.uuid4().hex[:12]}",
            artifact_id=artifact.artifact_id,
            mime=artifact.mime,
            html=html,
            config={},
            assets=[],
            metadata={"renderer": type(self).__name__},
            rendered_at=time.time(),
            render_time_ms=0.0,
        )


class FallbackRenderer(BaseRenderer):
    """降级渲染器 — 将任意 Artifact 渲染为纯文本.

    当 RendererRegistry 中找不到对应 MIME 的渲染器时，使用此渲染器进行降级处理。
    将 artifact 的 payload 转换为可读的 HTML ``<pre>`` 文本块 (JSON 缩进格式)，
    保证系统在面对未知 MIME 类型时仍能给出可读的输出，而非崩溃。

    特性:
        - ``_MIME_TYPES`` 为空列表: 不主动注册任何 MIME，仅作为兜底使用
        - ``do_render()`` 将 payload 序列化为 JSON 并包裹在 ``<pre>`` 标签中
        - ``supports_any()`` 返回 True，标识为通用降级渲染器

    使用示例::

        registry = RendererRegistry()
        # ... 注册专用渲染器 ...
        # 当 get_renderer() 抛出 RendererNotFoundError 时:
        fallback = FallbackRenderer()
        descriptor = fallback.render(artifact, context)
    """

    _MIME_TYPES: list[str] = []  # 空列表，不主动注册任何 MIME

    def do_render(self, artifact: Artifact, context: RenderContext) -> str:
        """将 payload 渲染为 HTML 格式的 ``<pre>`` 块.

        将 artifact.payload 序列化为 JSON 字符串 (缩进 2 空格)，
        并与 title 一起包裹在 ``<div class="fallback-renderer">`` 中。

        Args:
            artifact: 待渲染的制品 (已通过校验，payload 非空)。
            context: 渲染上下文 (当前实现未使用，保留以符合接口契约)。

        Returns:
            HTML 字符串，格式为::

                <div class="fallback-renderer">
                  <h3>{title or "未命名制品"}</h3>
                  <pre>{json_payload}</pre>
                </div>
        """
        payload_str = json.dumps(
            artifact.payload, indent=2, ensure_ascii=False, default=str
        )
        title = artifact.title or "未命名制品"
        return (
            f'<div class="fallback-renderer">'
            f"<h3>{title}</h3>"
            f"<pre>{payload_str}</pre>"
            f"</div>"
        )

    def supports_any(self) -> bool:
        """标识为通用降级渲染器.

        Returns:
            True — FallbackRenderer 可处理任意 MIME 类型的 Artifact。
        """
        return True
