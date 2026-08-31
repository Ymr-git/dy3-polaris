"""CC3 溯源捕获层 — 异常定义.

继承 L6Error 异常体系，集成 JSON-RPC 错误码。
错误码范围 -32400 ~ -32406。

融合方案:
- RFC 7807 Problem Details: 结构化错误响应
- OpenAPI 3.1: 标准化错误码规范
"""

from __future__ import annotations

from typing import Any

from ...l6.core.exceptions import L6Error


class CC3Error(L6Error):
    """CC3 溯源捕获层基础异常."""

    def __init__(
        self,
        code: str = "CC3_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32400


class HashMismatchError(CC3Error):
    """哈希校验失败 — 溯源链可能被篡改."""

    def __init__(
        self,
        expected_hash: str,
        actual_hash: str,
        record_id: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        self.record_id = record_id
        super().__init__(
            "CC3_HASH_MISMATCH",
            detail or f"expected={expected_hash[:16]}..., actual={actual_hash[:16]}..., record={record_id}",
            {"expected_hash": expected_hash, "actual_hash": actual_hash, "record_id": record_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32401


class AnnotationNotFoundError(CC3Error):
    """KPA 标注未找到."""

    def __init__(
        self,
        annotation_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.annotation_id = annotation_id
        super().__init__(
            "CC3_ANNOTATION_NOT_FOUND",
            detail or f"annotation_id={annotation_id}",
            {"annotation_id": annotation_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32402


class DebateLogNotFoundError(CC3Error):
    """辩论日志未找到."""

    def __init__(
        self,
        debate_log_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.debate_log_id = debate_log_id
        super().__init__(
            "CC3_DEBATE_LOG_NOT_FOUND",
            detail or f"debate_log_id={debate_log_id}",
            {"debate_log_id": debate_log_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32403


class SchemaValidationError(CC3Error):
    """七维标注数据 Schema 校验失败."""

    def __init__(
        self,
        dimension: str,
        missing_fields: list[str] | None = None,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.dimension = dimension
        self.missing_fields = missing_fields or []
        super().__init__(
            "CC3_SCHEMA_VALIDATION_FAILED",
            detail or f"dimension={dimension}, missing={missing_fields}",
            {"dimension": dimension, "missing_fields": missing_fields or [], **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32404


class ChainBrokenError(CC3Error):
    """溯源链断裂 — prev_hash 不匹配."""

    def __init__(
        self,
        chain_id: str,
        broken_at_index: int,
        expected_prev: str,
        actual_prev: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.chain_id = chain_id
        self.broken_at_index = broken_at_index
        self.expected_prev = expected_prev
        self.actual_prev = actual_prev
        super().__init__(
            "CC3_CHAIN_BROKEN",
            detail or f"chain={chain_id}, index={broken_at_index}, expected_prev={expected_prev[:16]}..., actual_prev={actual_prev[:16]}...",
            {"chain_id": chain_id, "broken_at_index": broken_at_index, "expected_prev": expected_prev, "actual_prev": actual_prev, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32405


class StorageUnavailableError(CC3Error):
    """L0 Ledger 存储服务不可用."""

    def __init__(
        self,
        reason: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "CC3_STORAGE_UNAVAILABLE",
            detail or reason or "L0 Ledger storage unavailable",
            {"reason": reason, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32406
