"""CC1 防幻觉层 — 异常定义.

继承 L6Error 异常体系，集成 JSON-RPC 错误码。
错误码范围 -32200 ~ -32206。
"""

from __future__ import annotations

from typing import Any

from ...l6.core.exceptions import L6Error


class CC1Error(L6Error):
    """CC1 防幻觉层基础异常."""

    def __init__(self, message: str, *, detail: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__(message, detail=detail, context=context or {})

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32200


class VerificationError(CC1Error):
    """验证过程错误."""

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32201


class ClaimExtractionError(CC1Error):
    """声明提取错误."""

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32202


class VerifierNotFoundError(CC1Error):
    """验证器未找到错误.

    当尝试使用未注册的验证器类型时抛出。
    """

    def __init__(self, verifier_type: str, **kwargs: Any) -> None:
        super().__init__(
            f"验证器类型 '{verifier_type}' 未注册",
            detail=f"Requested verifier type: {verifier_type}",
            context={"verifier_type": verifier_type},
            **kwargs,
        )

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32203


class HallucinationDetectedError(CC1Error):
    """检测到严重幻觉错误.

    当幻觉分数低于拒绝阈值时抛出，输出被拒绝。
    """

    def __init__(self, score: float, threshold: float, **kwargs: Any) -> None:
        super().__init__(
            f"检测到严重幻觉: 分数 {score:.3f} 低于拒绝阈值 {threshold:.3f}",
            detail=f"Score={score}, RefuseThreshold={threshold}",
            context={"score": score, "threshold": threshold},
            **kwargs,
        )

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32204


class PipelineConfigError(CC1Error):
    """管道配置错误."""

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32205


class EvidenceInsufficientError(CC1Error):
    """证据不足错误.

    当验证需要证据但未提供任何证据时抛出。
    """

    def __init__(self, claim_id: str, verifier_type: str = "", **kwargs: Any) -> None:
        super().__init__(
            f"声明 '{claim_id}' 缺少验证证据",
            detail=f"Claim={claim_id}, Verifier={verifier_type}",
            context={"claim_id": claim_id, "verifier_type": verifier_type},
            **kwargs,
        )

    @classmethod
    def _jsonrpc_code(cls) -> int:
        return -32206
