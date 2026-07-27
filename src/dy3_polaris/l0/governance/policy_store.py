"""L0 策略存储引擎.

提供治理策略的 CRUD 操作和多维度查询能力，
遵循与 ProvenanceStore 相同的设计模式：
- 内存存储（生产环境可替换为持久化后端）
- 多维度查询：按 ID、作用域、域、标签、启用状态
- 按优先级排序的评估用接口
- 违规记录管理
- 事件日志（有容量上限）

所有查询均返回副本或序列化字典，不暴露内部引用。
"""

from __future__ import annotations

import fnmatch
import logging
import re
import threading
import time
from collections import defaultdict
from typing import Any

from .exceptions import PolicyNotFoundError, PolicyValidationError, RuleSyntaxError
from .models import (
    EvalRequest,
    GovernanceEvent,
    GovernanceEventType,
    GovernancePolicy,
    MatchOperator,
    PolicyScope,
    SeverityLevel,
    ViolationRecord,
    ViolationStatus,
)

logger = logging.getLogger(__name__)


# 事件日志容量上限
_EVENT_LOG_CAP = 500
# 违规记录容量上限
_VIOLATION_CAP = 2000


class PolicyStore:
    """策略存储引擎.

    管理治理策略的全生命周期和多维度查询。
    同时维护违规记录和治理事件日志。

    线程安全：所有写操作通过 RLock 保护。
    """

    def __init__(self) -> None:
        # 策略存储：policy_id -> GovernancePolicy
        self._policies: dict[str, GovernancePolicy] = {}

        # 违规记录：violation_id -> ViolationRecord
        self._violations: dict[str, ViolationRecord] = {}

        # 治理事件日志（有容量上限）
        self._event_log: list[GovernanceEvent] = []

        # 评估计数器
        self._eval_count: int = 0
        self._deny_count: int = 0
        self._escalate_count: int = 0

        # 读写锁
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    def _append_event(self, event: GovernanceEvent) -> None:
        """追加事件到日志，超过容量时截断."""
        self._event_log.append(event)
        if len(self._event_log) > _EVENT_LOG_CAP:
            self._event_log = self._event_log[-_EVENT_LOG_CAP:]

    def _make_event(
        self,
        event_type: GovernanceEventType,
        detail: str = "",
        references: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> GovernanceEvent:
        """快捷构造治理事件."""
        return GovernanceEvent(
            event_type=event_type,
            detail=detail,
            references=references or [],
            payload=payload or {},
        )

    # --------------------------------------------------------
    # 规则匹配（供 G2 引擎使用）
    # --------------------------------------------------------

    @staticmethod
    def match_rule(
        rule_field: str,
        rule_operator: MatchOperator,
        rule_value: str,
        rule_negate: bool,
        request: EvalRequest,
    ) -> bool:
        """匹配单条规则.

        从 request 中提取 rule_field 对应的值（支持嵌套上下文），
        按指定操作符进行匹配。

        Args:
            rule_field: 匹配字段名
            rule_operator: 匹配操作符
            rule_value: 期望值
            rule_negate: 是否取反
            request: 评估请求

        Returns:
            是否命中
        """
        # 构建待匹配的值字典
        request_dict: dict[str, Any] = {
            "actor": request.actor,
            "action": request.action,
            "resource": request.resource,
            "layer": request.layer,
            "domain": request.domain,
        }
        # 合并额外上下文（上下文中的同名键覆盖基本字段）
        request_dict.update(request.context)

        # 提取字段值
        actual = str(request_dict.get(rule_field, ""))

        # 执行匹配
        matched = False
        if rule_operator == MatchOperator.EXACT:
            matched = actual == rule_value
        elif rule_operator == MatchOperator.GLOB:
            matched = fnmatch.fnmatch(actual, rule_value)
        elif rule_operator == MatchOperator.REGEX:
            try:
                matched = bool(re.search(rule_value, actual))
            except re.error:
                matched = False

        # 取反
        return (not matched) if rule_negate else matched

    @staticmethod
    def match_condition(
        condition: "PolicyCondition",
        request: EvalRequest,
    ) -> bool:
        """匹配条件组合.

        空规则列表视为全匹配。

        Args:
            condition: 策略条件
            request: 评估请求

        Returns:
            是否命中
        """
        if not condition.rules:
            return True

        results = [
            PolicyStore.match_rule(
                r.field, r.operator, r.value, r.negate, request
            )
            for r in condition.rules
        ]

        if condition.logic == "and":
            return all(results)
        else:
            return any(results)

    # --------------------------------------------------------
    # 策略 CRUD
    # --------------------------------------------------------

    def add_policy(self, policy: GovernancePolicy) -> GovernancePolicy:
        """添加策略.

        校验匹配规则的语法（正则/通配符），然后存储。

        Args:
            policy: 要添加的策略

        Returns:
            存储后的策略（含事件记录）

        Raises:
            PolicyValidationError: 策略语法无效
        """
        self._validate_rules(policy)

        with self._lock:
            self._policies[policy.policy_id] = policy
            self._append_event(self._make_event(
                GovernanceEventType.POLICY_CREATED,
                f"创建策略: {policy.name}",
                [policy.policy_id],
            ))
        return policy

    def get_policy(self, policy_id: str) -> GovernancePolicy | None:
        """获取策略."""
        return self._policies.get(policy_id)

    def get_policy_or_raise(self, policy_id: str) -> GovernancePolicy:
        """获取策略，不存在时抛异常."""
        policy = self._policies.get(policy_id)
        if policy is None:
            raise PolicyNotFoundError(policy_id)
        return policy

    def update_policy(self, policy_id: str, **updates: Any) -> GovernancePolicy:
        """更新策略字段.

        Args:
            policy_id: 策略 ID
            **updates: 要更新的字段

        Returns:
            更新后的策略

        Raises:
            PolicyNotFoundError: 策略不存在
            PolicyValidationError: 更新后校验失败
        """
        with self._lock:
            policy = self._policies.get(policy_id)
            if policy is None:
                raise PolicyNotFoundError(policy_id)

            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            policy.touch()

            # 更新后重新校验规则
            self._validate_rules(policy)

            self._append_event(self._make_event(
                GovernanceEventType.POLICY_UPDATED,
                f"更新策略: {policy.name}",
                [policy_id],
            ))
            return policy

    def remove_policy(self, policy_id: str) -> GovernancePolicy | None:
        """移除策略.

        Returns:
            被移除的策略，若不存在则返回 None
        """
        with self._lock:
            policy = self._policies.pop(policy_id, None)
            if policy is not None:
                self._append_event(self._make_event(
                    GovernanceEventType.POLICY_DELETED,
                    f"删除策略: {policy.name}",
                    [policy_id],
                ))
            return policy

    def enable_policy(self, policy_id: str) -> GovernancePolicy:
        """启用策略."""
        return self.update_policy(policy_id, enabled=True)

    def disable_policy(self, policy_id: str) -> GovernancePolicy:
        """禁用策略."""
        return self.update_policy(policy_id, enabled=False)

    # --------------------------------------------------------
    # 策略查询
    # --------------------------------------------------------

    def list_policies(
        self,
        *,
        scope: PolicyScope | None = None,
        domain: str | None = None,
        enabled_only: bool = False,
        tag: str | None = None,
    ) -> list[GovernancePolicy]:
        """按条件查询策略.

        多个条件为 AND 关系。

        Args:
            scope: 按作用域筛选
            domain: 按治理域筛选
            enabled_only: 仅返回启用的策略
            tag: 按标签筛选

        Returns:
            匹配的策略列表（按优先级降序）
        """
        results: list[GovernancePolicy] = []
        for policy in self._policies.values():
            if scope is not None and policy.scope != scope:
                continue
            if domain is not None and policy.domain != domain:
                continue
            if enabled_only and not policy.enabled:
                continue
            if tag is not None and tag not in policy.tags:
                continue
            results.append(policy)

        # 按优先级降序排序
        results.sort(key=lambda p: p.priority, reverse=True)
        return results

    def get_evaluable_policies(self) -> list[GovernancePolicy]:
        """获取所有可评估的策略.

        返回已启用且按优先级降序排列的策略列表。
        G2 策略引擎的核心输入。
        """
        return self.list_policies(enabled_only=True)

    def get_policies_for_scope(self, scope: PolicyScope) -> list[GovernancePolicy]:
        """获取指定作用域和 global 的所有已启用策略.

        策略评估时，global 策略总是参与匹配，
        然后叠加指定作用域的策略。
        """
        policies = self.list_policies(enabled_only=True)
        return [
            p for p in policies
            if p.scope == PolicyScope.GLOBAL or p.scope == scope
        ]

    # --------------------------------------------------------
    # 违规记录管理
    # --------------------------------------------------------

    def add_violation(self, violation: ViolationRecord) -> ViolationRecord:
        """添加违规记录.

        超过容量上限时移除最早的违规记录。
        """
        with self._lock:
            # 容量控制
            if len(self._violations) >= _VIOLATION_CAP:
                oldest_id = min(
                    self._violations,
                    key=lambda k: self._violations[k].created_at,
                )
                del self._violations[oldest_id]

            self._violations[violation.violation_id] = violation
            self._append_event(self._make_event(
                GovernanceEventType.VIOLATION_DETECTED,
                f"违规: {violation.detail}",
                [violation.violation_id, violation.policy_id],
            ))
        return violation

    def get_violation(self, violation_id: str) -> ViolationRecord | None:
        """获取违规记录."""
        return self._violations.get(violation_id)

    def query_violations(
        self,
        *,
        policy_id: str | None = None,
        actor: str | None = None,
        severity: SeverityLevel | None = None,
        status: ViolationStatus | None = None,
        limit: int = 100,
    ) -> list[ViolationRecord]:
        """查询违规记录.

        多条件 AND 关系，按创建时间倒序，limit 控制返回上限。
        """
        results: list[ViolationRecord] = []
        for v in self._violations.values():
            if policy_id is not None and v.policy_id != policy_id:
                continue
            if actor is not None and v.actor != actor:
                continue
            if severity is not None and v.severity != severity:
                continue
            if status is not None and v.status != status:
                continue
            results.append(v)

        results.sort(key=lambda v: v.created_at, reverse=True)
        return results[:limit]

    @property
    def violation_count(self) -> int:
        """违规记录总数."""
        return len(self._violations)

    @property
    def active_violation_count(self) -> int:
        """活跃违规数（detected + confirmed）."""
        return sum(
            1 for v in self._violations.values()
            if v.status in (ViolationStatus.DETECTED, ViolationStatus.CONFIRMED)
        )

    # --------------------------------------------------------
    # 事件日志
    # --------------------------------------------------------

    def get_event_log(self, limit: int = 100) -> list[GovernanceEvent]:
        """获取最近的事件日志."""
        return list(self._event_log[-limit:])

    # --------------------------------------------------------
    # 统计信息
    # --------------------------------------------------------

    @property
    def policy_count(self) -> int:
        """策略总数."""
        return len(self._policies)

    @property
    def enabled_policy_count(self) -> int:
        """已启用策略数."""
        return sum(1 for p in self._policies.values() if p.enabled)

    def get_stats(self) -> dict[str, Any]:
        """获取存储统计信息."""
        by_scope: dict[str, int] = defaultdict(int)
        by_action: dict[str, int] = defaultdict(int)
        for p in self._policies.values():
            by_scope[p.scope] += 1
            by_action[p.action] += 1

        return {
            "policies": {
                "total": self.policy_count,
                "enabled": self.enabled_policy_count,
                "disabled": self.policy_count - self.enabled_policy_count,
                "by_scope": dict(by_scope),
                "by_action": dict(by_action),
            },
            "violations": {
                "total": self.violation_count,
                "active": self.active_violation_count,
            },
            "evaluations": {
                "total": self._eval_count,
                "denied": self._deny_count,
                "escalated": self._escalate_count,
            },
            "event_log_size": len(self._event_log),
        }

    def record_eval(self, denied: bool = False, escalated: bool = False) -> None:
        """记录一次评估事件（由 G2 引擎调用）."""
        self._eval_count += 1
        if denied:
            self._deny_count += 1
        if escalated:
            self._escalate_count += 1

    # --------------------------------------------------------
    # 导出
    # --------------------------------------------------------

    def export_all(self) -> dict[str, Any]:
        """导出完整存储内容（含策略和违规明细）."""
        return {
            "policies": {
                pid: p.model_dump(mode="json")
                for pid, p in self._policies.items()
            },
            "violations": {
                vid: v.model_dump(mode="json")
                for vid, v in self._violations.items()
            },
            "stats": self.get_stats(),
        }

    def export_summary(self) -> dict[str, Any]:
        """导出存储摘要（不含明细）."""
        return {
            "stats": self.get_stats(),
            "policy_ids": sorted(self._policies.keys()),
            "violation_ids": sorted(self._violations.keys()),
        }

    # --------------------------------------------------------
    # G6 路由层兼容别名
    # --------------------------------------------------------

    def add(self, policy: GovernancePolicy) -> str:
        """添加策略并返回 policy_id（G6 路由层别名）."""
        self.add_policy(policy)
        return policy.policy_id

    def get(self, policy_id: str) -> GovernancePolicy | None:
        """获取策略（G6 路由层别名）."""
        return self.get_policy(policy_id)

    def remove(self, policy_id: str) -> bool:
        """移除策略，返回是否成功（G6 路由层别名）."""
        return self.remove_policy(policy_id) is not None

    def count(self) -> int:
        """策略总数（G6 路由层别名）."""
        return self.policy_count

    def list_all(self) -> list[GovernancePolicy]:
        """列出所有策略（G6 路由层别名）."""
        return self.list_policies()

    # --------------------------------------------------------
    # 内部校验
    # --------------------------------------------------------

    @staticmethod
    def _validate_rules(policy: GovernancePolicy) -> None:
        """校验策略中所有匹配规则的语法.

        Args:
            policy: 要校验的策略

        Raises:
            RuleSyntaxError: 正则表达式语法无效
            PolicyValidationError: 其他校验失败
        """
        for rule in policy.condition.rules:
            if rule.operator == MatchOperator.REGEX:
                try:
                    re.compile(rule.value)
                except re.error as e:
                    raise RuleSyntaxError(
                        field=rule.field,
                        operator=rule.operator.value,
                        value=rule.value,
                        reason=f"正则编译失败: {e}",
                    )
            # glob 语法由 fnmatch 处理，不需要预校验

        # 校验 escalate 策略必须有 target
        from .models import PolicyAction
        if policy.action == PolicyAction.ESCALATE and policy.escalation is not None:
            if not policy.escalation.target:
                raise PolicyValidationError(
                    detail="escalate 策略必须指定升级目标 (target)",
                    context={"policy_id": policy.policy_id},
                )
