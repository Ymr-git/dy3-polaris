"""L7 体验呈现层 — 异常定义.

继承 L6Error 异常体系，集成 JSON-RPC 错误码。
错误码范围 -32500 ~ -32508。

融合方案:
- RFC 7807 Problem Details: 结构化错误响应
- OpenAPI 3.1: 标准化错误码规范
- JSON-RPC 2.0: 错误码保留区间 -32000 ~ -32768

分层异常层级:
    L6Error (L6 协议层基类)
      └── L7Error (L7 体验呈现层基类, -32500)
            ├── RendererNotFoundError       渲染器未找到 (-32501)
            ├── ArtifactNotFoundError       Artifact 未找到 (-32502)
            ├── ArtifactValidationError     Artifact 校验失败 (-32503)
            ├── RenderTimeoutError          渲染超时 (-32504)
            ├── UnsupportedMimeError        不支持的 MIME 类型 (-32505)
            ├── VersionConflictError        版本冲突 (-32506)
            ├── ArtifactNotEditableError    Artifact 不可编辑 (-32507)
            └── RenderContextError          渲染上下文错误 (-32508)
"""

from __future__ import annotations

from typing import Any

from ..l6.core.exceptions import L6Error


# ============================================================
# 基础异常
# ============================================================

class L7Error(L6Error):
    """L7 体验呈现层基础异常.

    继承 L6Error 的结构化错误信息 (code / detail / context)，
    为 L7 层渲染器、Artifact 管理器等组件提供统一异常契约。

    Attributes:
        code: 机器可读的错误码 (e.g. "L7_ERROR")
        detail: 人类可读的错误详情
        context: 结构化上下文信息，用于日志和调试
    """

    def __init__(
        self,
        code: str = "L7_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32500


# ============================================================
# 渲染器异常
# ============================================================

class RendererNotFoundError(L7Error):
    """渲染器未找到 — 注册表中不存在能处理该 MIME 类型的渲染器.

    Attributes:
        mime_type: 未匹配到渲染器的 MIME 类型
    """

    def __init__(
        self,
        mime_type: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.mime_type = mime_type
        super().__init__(
            "L7_RENDERER_NOT_FOUND",
            detail or f"No renderer registered for MIME type: {mime_type}",
            {"mime_type": mime_type, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32501


class UnsupportedMimeError(L7Error):
    """不支持的 MIME 类型 — 渲染器声明不支持该 MIME 类型.

    Attributes:
        mime_type: 不被支持的 MIME 类型
    """

    def __init__(
        self,
        mime_type: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.mime_type = mime_type
        super().__init__(
            "L7_UNSUPPORTED_MIME",
            detail or f"Unsupported MIME type: {mime_type}",
            {"mime_type": mime_type, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32505


# ============================================================
# Artifact 异常
# ============================================================

class ArtifactNotFoundError(L7Error):
    """Artifact 未找到 — 指定 ID 的 Artifact 不存在.

    Attributes:
        artifact_id: 未找到的 Artifact ID
    """

    def __init__(
        self,
        artifact_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.artifact_id = artifact_id
        super().__init__(
            "L7_ARTIFACT_NOT_FOUND",
            detail or f"Artifact not found: {artifact_id}",
            {"artifact_id": artifact_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32502


class ArtifactValidationError(L7Error):
    """Artifact 校验失败 — payload 结构与类型不匹配.

    Attributes:
        field: 校验失败的字段路径
        missing_fields: 缺失的字段列表
    """

    def __init__(
        self,
        field: str = "",
        missing_fields: list[str] | None = None,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.field = field
        self.missing_fields = missing_fields or []
        super().__init__(
            "L7_ARTIFACT_VALIDATION_FAILED",
            detail or f"field={field}, missing={missing_fields}",
            {"field": field, "missing_fields": self.missing_fields, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32503


class ArtifactNotEditableError(L7Error):
    """Artifact 不可编辑 — 尝试修改已封存/锁定的 Artifact.

    Attributes:
        artifact_id: 不可编辑的 Artifact ID
    """

    def __init__(
        self,
        artifact_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.artifact_id = artifact_id
        super().__init__(
            "L7_ARTIFACT_NOT_EDITABLE",
            detail or f"Artifact {artifact_id} is not editable",
            {"artifact_id": artifact_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32507


# ============================================================
# 渲染异常
# ============================================================

class RenderTimeoutError(L7Error):
    """渲染超时 — 渲染过程超过指定时间限制.

    Attributes:
        timeout_seconds: 超时时间（秒）
    """

    def __init__(
        self,
        timeout_seconds: float,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(
            "L7_RENDER_TIMEOUT",
            detail or f"Render timed out after {timeout_seconds}s",
            {"timeout_seconds": timeout_seconds, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32504


class RenderContextError(L7Error):
    """渲染上下文错误 — 渲染上下文中缺失或非法的键值.

    Attributes:
        context_key: 出错的上下文键
    """

    def __init__(
        self,
        context_key: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.context_key = context_key
        super().__init__(
            "L7_RENDER_CONTEXT_ERROR",
            detail or f"Render context error: {context_key}",
            {"context_key": context_key, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32508


# ============================================================
# 版本异常
# ============================================================

class VersionConflictError(L7Error):
    """版本冲突 — 并发编辑导致版本号冲突.

    Attributes:
        artifact_id: 发生冲突的 Artifact ID
        version: 冲突的版本号
    """

    def __init__(
        self,
        artifact_id: str,
        version: int,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.artifact_id = artifact_id
        self.version = version
        super().__init__(
            "L7_VERSION_CONFLICT",
            detail or f"Version conflict: artifact_id={artifact_id}, version={version}",
            {"artifact_id": artifact_id, "version": version, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32506
