"""L0 治理异常体系.

与 L6 L6Error 体系对齐，遵循相同的设计模式：
- 继承 L6Error 基类
- 标准化错误码 (GOVERNANCE_*)
- to_json_rpc_error() 自动映射
- 子类保存治理专属上下文字段
"""

from __future__ import annotations

from typing import Any

from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 治理异常层级
# ============================================================


class GovernanceError(L6Error):
    """L0 治理层异常基类.

    所有治理相关异常的父类。
    默认错误码 GOVERNANCE_ERROR，JSON-RPC 码 -32100。
    """

    def __init__(
        self,
        code: str = "GOVERNANCE_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32100


class PolicyNotFoundError(GovernanceError):
    """策略未找到."""

    def __init__(
        self,
        policy_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.policy_id = policy_id
        super().__init__(
            "GOVERNANCE_POLICY_NOT_FOUND",
            detail or f"策略未找到: {policy_id}",
            {"policy_id": policy_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32101


class PolicyConflictError(GovernanceError):
    """策略冲突.

    当新策略与已有策略在优先级和匹配范围上产生冲突时抛出。
    """

    def __init__(
        self,
        message: str,
        policy_ids: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.policy_ids = policy_ids or []
        super().__init__(
            "GOVERNANCE_POLICY_CONFLICT",
            message,
            {"conflicting_policies": self.policy_ids, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32102


class ViolationError(GovernanceError):
    """违规错误.

    当策略评估触发违规（deny/escalate）时抛出。
    """

    def __init__(
        self,
        violation_id: str,
        policy_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.violation_id = violation_id
        self.policy_id = policy_id
        super().__init__(
            "GOVERNANCE_VIOLATION",
            detail or f"违规: {violation_id} (策略: {policy_id})",
            {"violation_id": violation_id, "policy_id": policy_id, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32103


class ComplianceCheckFailedError(GovernanceError):
    """合规检查失败.

    合规评分低于阈值时抛出。
    """

    def __init__(
        self,
        score: float,
        threshold: float,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.score = score
        self.threshold = threshold
        super().__init__(
            "GOVERNANCE_COMPLIANCE_FAILED",
            detail or f"合规评分 {score} 低于阈值 {threshold}",
            {"score": score, "threshold": threshold, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32104


class PolicyValidationError(GovernanceError):
    """策略校验错误.

    策略创建/更新时校验失败（如无效的匹配规则语法）。
    """

    def __init__(
        self,
        detail: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "GOVERNANCE_POLICY_VALIDATION_ERROR",
            detail,
            context,
        )

    def _jsonrpc_code(self) -> int:
        return -32105


class RuleSyntaxError(GovernanceError):
    """规则语法错误.

    匹配规则的正则表达式或通配符语法无效时抛出。
    """

    def __init__(
        self,
        field: str,
        operator: str,
        value: str,
        reason: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.field = field
        self.operator = operator
        self.value = value
        super().__init__(
            "GOVERNANCE_RULE_SYNTAX_ERROR",
            reason or f"规则语法错误: {field} {operator} {value}",
            {"field": field, "operator": operator, "value": value, **(context or {})},
        )

    def _jsonrpc_code(self) -> int:
        return -32106
