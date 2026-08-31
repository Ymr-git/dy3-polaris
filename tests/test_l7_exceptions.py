"""L7 体验呈现层 — 异常体系测试.

覆盖 L7 异常继承层级、业务属性、JSON-RPC 错误码映射。
错误码范围 -32500 ~ -32508。

测试领域: Dy3+ 发光材料 (YAG 基质, 4f-4f 跃迁, 480/574/660nm 发射)
"""
from __future__ import annotations

import pytest

from dy3_polaris.l6.core.exceptions import L6Error
from dy3_polaris.l7.exceptions import (
    L7Error,
    RendererNotFoundError,
    ArtifactNotFoundError,
    ArtifactValidationError,
    RenderTimeoutError,
    UnsupportedMimeError,
    VersionConflictError,
    ArtifactNotEditableError,
    RenderContextError,
)


# ============================================================
# L7Error 基类
# ============================================================


class TestL7ErrorBase:
    """L7Error 基类测试."""

    def test_inherits_from_l6_error(self):
        """L7Error 应继承自 L6Error."""
        err = L7Error()
        assert isinstance(err, L6Error)
        assert isinstance(err, Exception)

    def test_default_attributes(self):
        """默认 code/detail/context 属性."""
        err = L7Error()
        assert err.code == "L7_ERROR"
        assert err.detail == ""
        assert err.context == {}

    def test_custom_attributes(self):
        """自定义 code/detail/context."""
        err = L7Error(
            code="L7_CUSTOM",
            detail="something went wrong",
            context={"artifact_id": "art-abc123"},
        )
        assert err.code == "L7_CUSTOM"
        assert err.detail == "something went wrong"
        assert err.context == {"artifact_id": "art-abc123"}

    def test_jsonrpc_code(self):
        """L7Error 基类 _jsonrpc_code 返回 -32500."""
        err = L7Error()
        assert err._jsonrpc_code() == -32500

    def test_to_json_rpc_error_structure(self):
        """to_json_rpc_error 返回正确的 JSON-RPC 错误对象."""
        err = L7Error(detail="render failed", context={"mime": "text/plain"})
        obj = err.to_json_rpc_error()
        assert isinstance(obj, dict)
        assert obj["code"] == -32500
        assert obj["message"] == "L7_ERROR"
        assert obj["data"]["detail"] == "render failed"
        assert obj["data"]["mime"] == "text/plain"

    def test_to_json_rpc_error_no_data_when_empty(self):
        """无 detail 且无 context 时 data 为 None."""
        err = L7Error()
        obj = err.to_json_rpc_error()
        assert obj["code"] == -32500
        assert obj["message"] == "L7_ERROR"
        assert obj["data"] is None

    def test_all_subclasses_inherit_l7_error(self):
        """所有 L7 子异常都应继承自 L7Error 和 L6Error."""
        subclasses = [
            RendererNotFoundError,
            ArtifactNotFoundError,
            ArtifactValidationError,
            RenderTimeoutError,
            UnsupportedMimeError,
            VersionConflictError,
            ArtifactNotEditableError,
            RenderContextError,
        ]
        for cls in subclasses:
            assert issubclass(cls, L7Error), f"{cls.__name__} should inherit L7Error"
            assert issubclass(cls, L6Error), f"{cls.__name__} should inherit L6Error"


# ============================================================
# RendererNotFoundError (-32501)
# ============================================================


class TestRendererNotFoundError:
    """渲染器未找到异常测试."""

    def test_has_mime_type_attribute(self):
        err = RendererNotFoundError(mime_type="application/vnd.dy3.chart+json")
        assert err.mime_type == "application/vnd.dy3.chart+json"

    def test_jsonrpc_code(self):
        err = RendererNotFoundError(mime_type="text/plain")
        assert err._jsonrpc_code() == -32501

    def test_to_json_rpc_error_includes_mime_type(self):
        mime = "application/vnd.dy3.molecule+json"
        err = RendererNotFoundError(mime_type=mime)
        obj = err.to_json_rpc_error()
        assert obj["code"] == -32501
        assert obj["data"]["mime_type"] == mime

    def test_is_l7_error(self):
        err = RendererNotFoundError(mime_type="text/plain")
        assert isinstance(err, L7Error)


# ============================================================
# ArtifactNotFoundError (-32502)
# ============================================================


class TestArtifactNotFoundError:
    """Artifact 未找到异常测试."""

    def test_has_artifact_id_attribute(self):
        err = ArtifactNotFoundError(artifact_id="art-abc123def456")
        assert err.artifact_id == "art-abc123def456"

    def test_jsonrpc_code(self):
        err = ArtifactNotFoundError(artifact_id="art-test")
        assert err._jsonrpc_code() == -32502

    def test_to_json_rpc_error_includes_artifact_id(self):
        aid = "art-deadbeef"
        err = ArtifactNotFoundError(artifact_id=aid)
        obj = err.to_json_rpc_error()
        assert obj["code"] == -32502
        assert obj["data"]["artifact_id"] == aid


# ============================================================
# ArtifactValidationError (-32503)
# ============================================================


class TestArtifactValidationError:
    """Artifact 校验失败异常测试."""

    def test_has_field_attribute(self):
        err = ArtifactValidationError(field="payload.content")
        assert err.field == "payload.content"

    def test_has_missing_fields_attribute(self):
        err = ArtifactValidationError(field="payload", missing_fields=["data", "chart_type"])
        assert err.missing_fields == ["data", "chart_type"]

    def test_missing_fields_defaults_to_empty(self):
        err = ArtifactValidationError(field="payload")
        assert err.missing_fields == []

    def test_jsonrpc_code(self):
        err = ArtifactValidationError(field="payload")
        assert err._jsonrpc_code() == -32503

    def test_to_json_rpc_error_includes_field_and_missing(self):
        err = ArtifactValidationError(field="payload", missing_fields=["nodes", "edges"])
        obj = err.to_json_rpc_error()
        assert obj["code"] == -32503
        assert obj["data"]["field"] == "payload"
        assert obj["data"]["missing_fields"] == ["nodes", "edges"]


# ============================================================
# RenderTimeoutError (-32504)
# ============================================================


class TestRenderTimeoutError:
    """渲染超时异常测试."""

    def test_has_timeout_seconds_attribute(self):
        err = RenderTimeoutError(timeout_seconds=30.0)
        assert err.timeout_seconds == 30.0

    def test_jsonrpc_code(self):
        err = RenderTimeoutError(timeout_seconds=10)
        assert err._jsonrpc_code() == -32504

    def test_to_json_rpc_error_includes_timeout(self):
        err = RenderTimeoutError(timeout_seconds=45.5)
        obj = err.to_json_rpc_error()
        assert obj["code"] == -32504
        assert obj["data"]["timeout_seconds"] == 45.5


# ============================================================
# UnsupportedMimeError (-32505)
# ============================================================


class TestUnsupportedMimeError:
    """不支持的 MIME 类型异常测试."""

    def test_has_mime_type_attribute(self):
        err = UnsupportedMimeError(mime_type="application/pdf")
        assert err.mime_type == "application/pdf"

    def test_jsonrpc_code(self):
        err = UnsupportedMimeError(mime_type="video/mp4")
        assert err._jsonrpc_code() == -32505

    def test_to_json_rpc_error_includes_mime_type(self):
        mime = "image/png"
        err = UnsupportedMimeError(mime_type=mime)
        obj = err.to_json_rpc_error()
        assert obj["code"] == -32505
        assert obj["data"]["mime_type"] == mime


# ============================================================
# VersionConflictError (-32506)
# ============================================================


class TestVersionConflictError:
    """版本冲突异常测试."""

    def test_has_artifact_id_attribute(self):
        err = VersionConflictError(artifact_id="art-abc", version=3)
        assert err.artifact_id == "art-abc"

    def test_has_version_attribute(self):
        err = VersionConflictError(artifact_id="art-abc", version=5)
        assert err.version == 5

    def test_jsonrpc_code(self):
        err = VersionConflictError(artifact_id="art-x", version=1)
        assert err._jsonrpc_code() == -32506

    def test_to_json_rpc_error_includes_artifact_id_and_version(self):
        err = VersionConflictError(artifact_id="art-y", version=7)
        obj = err.to_json_rpc_error()
        assert obj["code"] == -32506
        assert obj["data"]["artifact_id"] == "art-y"
        assert obj["data"]["version"] == 7


# ============================================================
# ArtifactNotEditableError (-32507)
# ============================================================


class TestArtifactNotEditableError:
    """Artifact 不可编辑异常测试."""

    def test_has_artifact_id_attribute(self):
        err = ArtifactNotEditableError(artifact_id="art-frozen")
        assert err.artifact_id == "art-frozen"

    def test_jsonrpc_code(self):
        err = ArtifactNotEditableError(artifact_id="art-locked")
        assert err._jsonrpc_code() == -32507

    def test_to_json_rpc_error_includes_artifact_id(self):
        err = ArtifactNotEditableError(artifact_id="art-sealed")
        obj = err.to_json_rpc_error()
        assert obj["code"] == -32507
        assert obj["data"]["artifact_id"] == "art-sealed"


# ============================================================
# RenderContextError (-32508)
# ============================================================


class TestRenderContextError:
    """渲染上下文错误异常测试."""

    def test_has_context_key_attribute(self):
        err = RenderContextError(context_key="viewport.width")
        assert err.context_key == "viewport.width"

    def test_jsonrpc_code(self):
        err = RenderContextError(context_key="theme")
        assert err._jsonrpc_code() == -32508

    def test_to_json_rpc_error_includes_context_key(self):
        err = RenderContextError(context_key="learner_mode")
        obj = err.to_json_rpc_error()
        assert obj["code"] == -32508
        assert obj["data"]["context_key"] == "learner_mode"


# ============================================================
# 错误码全局唯一性
# ============================================================


class TestErrorCodeUniqueness:
    """确保所有 L7 异常的 JSON-RPC 错误码唯一且在 -32500~-32508 范围内."""

    def test_all_codes_unique_and_in_range(self):
        codes = [
            L7Error()._jsonrpc_code(),
            RendererNotFoundError(mime_type="x")._jsonrpc_code(),
            ArtifactNotFoundError(artifact_id="x")._jsonrpc_code(),
            ArtifactValidationError(field="x")._jsonrpc_code(),
            RenderTimeoutError(timeout_seconds=1)._jsonrpc_code(),
            UnsupportedMimeError(mime_type="x")._jsonrpc_code(),
            VersionConflictError(artifact_id="x", version=1)._jsonrpc_code(),
            ArtifactNotEditableError(artifact_id="x")._jsonrpc_code(),
            RenderContextError(context_key="x")._jsonrpc_code(),
        ]
        # 全部在 -32500 ~ -32508 范围内
        for c in codes:
            assert -32508 <= c <= -32500, f"code {c} out of range"
        # 全部唯一
        assert len(set(codes)) == len(codes), "duplicate JSON-RPC codes detected"
