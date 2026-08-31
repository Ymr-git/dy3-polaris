"""G1 治理基础模型与存储测试.

覆盖范围:
1. 枚举定义（值、类型、数量）
2. GovernancePolicy 模型（创建、校验、默认值、model_validator）
3. PolicyMatchRule / PolicyCondition 模型
4. TransformSpec / EscalationSpec 模型
5. EvalRequest / EvalResult 模型
6. ViolationRecord 模型（生命周期：detect → confirm → resolve/ignore）
7. ComplianceReport / DimensionScore 模型
8. ReputationSnapshot 模型
9. GovernanceEvent 模型
10. PolicyStore CRUD
11. PolicyStore 查询（list_policies 多维度筛选）
12. PolicyStore 规则匹配（exact/glob/regex/negate/and/or）
13. PolicyStore 违规记录管理
14. PolicyStore 事件日志
15. PolicyStore 统计与导出
16. PolicyStore 容量控制
17. 治理异常体系（继承、错误码、JSON-RPC 映射）
18. 规则语法校验
19. Config 治理配置字段
"""

from __future__ import annotations

import logging
import time

logging.disable(logging.CRITICAL)

import pytest
from pydantic import ValidationError

from dy3_polaris.l0.governance.models import (
    ComplianceReport,
    ComplianceTemplate,
    DimensionScore,
    EscalationSpec,
    EvalRequest,
    EvalResult,
    GovernanceDomain,
    GovernanceEvent,
    GovernanceEventType,
    GovernancePolicy,
    MatchOperator,
    PolicyAction,
    PolicyCondition,
    PolicyMatchRule,
    PolicyScope,
    ReputationSnapshot,
    SeverityLevel,
    TransformSpec,
    ViolationRecord,
    ViolationStatus,
)
from dy3_polaris.l0.governance.policy_store import PolicyStore
from dy3_polaris.l0.governance.exceptions import (
    ComplianceCheckFailedError,
    GovernanceError,
    PolicyConflictError,
    PolicyNotFoundError,
    PolicyValidationError,
    RuleSyntaxError,
    ViolationError,
)
from dy3_polaris.l6.core.config import L6Config, reset_config


# ============================================================
# 1. 枚举定义
# ============================================================

class TestEnums:
    """验证所有治理层枚举的定义."""

    def test_policy_action_枚举值(self) -> None:
        assert PolicyAction.ALLOW == "allow"
        assert PolicyAction.DENY == "deny"
        assert PolicyAction.LOG == "log"
        assert PolicyAction.ESCALATE == "escalate"
        assert PolicyAction.TRANSFORM == "transform"
        assert len(PolicyAction) == 5

    def test_policy_scope_枚举值(self) -> None:
        assert PolicyScope.GLOBAL == "global"
        assert PolicyScope.AGENT == "agent"
        assert PolicyScope.TOOL == "tool"
        assert PolicyScope.LAYER == "layer"
        assert PolicyScope.DOMAIN == "domain"
        assert len(PolicyScope) == 5

    def test_match_operator_枚举值(self) -> None:
        assert MatchOperator.EXACT == "exact"
        assert MatchOperator.GLOB == "glob"
        assert MatchOperator.REGEX == "regex"
        assert len(MatchOperator) == 3

    def test_severity_level_枚举值(self) -> None:
        assert SeverityLevel.LOW == "low"
        assert SeverityLevel.MEDIUM == "medium"
        assert SeverityLevel.HIGH == "high"
        assert SeverityLevel.CRITICAL == "critical"
        assert len(SeverityLevel) == 4

    def test_governance_event_type_枚举值(self) -> None:
        assert GovernanceEventType.POLICY_CREATED == "policy_created"
        assert GovernanceEventType.VIOLATION_DETECTED == "violation_detected"
        assert GovernanceEventType.AUDIT_REPORT_GENERATED == "audit_report_generated"
        assert len(GovernanceEventType) == 11

    def test_violation_status_枚举值(self) -> None:
        assert ViolationStatus.DETECTED == "detected"
        assert ViolationStatus.CONFIRMED == "confirmed"
        assert ViolationStatus.RESOLVED == "resolved"
        assert ViolationStatus.IGNORED == "ignored"
        assert len(ViolationStatus) == 4

    def test_compliance_template_枚举值(self) -> None:
        assert ComplianceTemplate.ACADEMIC_INTEGRITY == "academic_integrity"
        assert ComplianceTemplate.DATA_PRIVACY == "data_privacy"
        assert ComplianceTemplate.CONTENT_SAFETY == "content_safety"
        assert ComplianceTemplate.PLATFORM_OPS == "platform_ops"
        assert ComplianceTemplate.ETHICAL_COMPLIANCE == "ethical_compliance"
        assert ComplianceTemplate.COPYRIGHT_PROTECTION == "copyright_protection"
        assert len(ComplianceTemplate) == 6

    def test_governance_domain_枚举值(self) -> None:
        assert GovernanceDomain.PROVENANCE == "provenance"
        assert GovernanceDomain.POLICY == "policy"
        assert GovernanceDomain.REPUTATION == "reputation"
        assert GovernanceDomain.AUDIT == "audit"
        assert len(GovernanceDomain) == 4

    def test_枚举是_str_子类(self) -> None:
        """(str, Enum) 模式: 枚举值可直接当字符串使用."""
        assert isinstance(PolicyAction.ALLOW, str)
        assert PolicyAction.DENY == "deny"


# ============================================================
# 2. GovernancePolicy 模型
# ============================================================

class TestGovernancePolicy:
    """GovernancePolicy 创建与校验."""

    def test_默认值创建(self) -> None:
        p = GovernancePolicy(name="测试策略")
        assert p.name == "测试策略"
        assert p.policy_id.startswith("pol-")
        assert p.action == PolicyAction.ALLOW
        assert p.scope == PolicyScope.GLOBAL
        assert p.priority == 0
        assert p.enabled is True
        assert p.domain == GovernanceDomain.POLICY
        assert p.condition.rules == []
        assert p.tags == []
        assert p.compliance_templates == []
        assert p.created_at > 0
        assert p.updated_at > 0
        assert p.created_by == "system"

    def test_完整字段创建(self) -> None:
        p = GovernancePolicy(
            name="禁止删除",
            description="禁止所有删除操作",
            domain=GovernanceDomain.AUDIT,
            scope=PolicyScope.GLOBAL,
            priority=100,
            action=PolicyAction.DENY,
            tags=["安全", "删除"],
            created_by="admin",
            compliance_templates=[ComplianceTemplate.CONTENT_SAFETY],
        )
        assert p.action == PolicyAction.DENY
        assert p.priority == 100
        assert len(p.tags) == 2
        assert p.compliance_templates[0] == ComplianceTemplate.CONTENT_SAFETY

    def test_transform_策略自动填充(self) -> None:
        p = GovernancePolicy(
            name="脱敏策略",
            action=PolicyAction.TRANSFORM,
        )
        assert p.transform is not None
        assert isinstance(p.transform, TransformSpec)

    def test_escalate_策略自动填充(self) -> None:
        p = GovernancePolicy(
            name="升级策略",
            action=PolicyAction.ESCALATE,
        )
        assert p.escalation is not None
        assert isinstance(p.escalation, EscalationSpec)
        assert p.escalation.target == "human_review"

    def test_自定义_escalation_spec(self) -> None:
        esc = EscalationSpec(target="admin", timeout_seconds=600.0)
        p = GovernancePolicy(
            name="自定义升级",
            action=PolicyAction.ESCALATE,
            escalation=esc,
        )
        assert p.escalation.target == "admin"
        assert p.escalation.timeout_seconds == 600.0

    def test_touch_更新时间戳(self) -> None:
        p = GovernancePolicy(name="t")
        old_ts = p.updated_at
        time.sleep(0.01)
        p.touch()
        assert p.updated_at > old_ts

    def test_model_dump_序列化(self) -> None:
        p = GovernancePolicy(name="序列化测试")
        d = p.model_dump(mode="json")
        assert "policy_id" in d
        assert "name" in d
        assert d["action"] == "allow"

    def test_priority_边界值(self) -> None:
        p_min = GovernancePolicy(name="min", priority=-1000)
        assert p_min.priority == -1000
        p_max = GovernancePolicy(name="max", priority=1000)
        assert p_max.priority == 1000
        with pytest.raises(ValidationError):
            GovernancePolicy(name="bad", priority=1001)
        with pytest.raises(ValidationError):
            GovernancePolicy(name="bad", priority=-1001)


# ============================================================
# 3. PolicyMatchRule / PolicyCondition
# ============================================================

class TestMatchRule:
    """匹配规则模型."""

    def test_默认值(self) -> None:
        r = PolicyMatchRule(field="agent_id", value="tutor-1")
        assert r.operator == MatchOperator.EXACT
        assert r.negate is False

    def test_glob_规则(self) -> None:
        r = PolicyMatchRule(
            field="agent_id", operator=MatchOperator.GLOB, value="tutor-*"
        )
        assert r.operator == MatchOperator.GLOB

    def test_regex_规则(self) -> None:
        r = PolicyMatchRule(
            field="tool_name", operator=MatchOperator.REGEX, value="bkt_.+"
        )
        assert r.operator == MatchOperator.REGEX

    def test_negate_规则(self) -> None:
        r = PolicyMatchRule(field="layer", value="L0", negate=True)
        assert r.negate is True


class TestPolicyCondition:
    """条件组合."""

    def test_默认_and_逻辑(self) -> None:
        c = PolicyCondition()
        assert c.logic == "and"
        assert c.rules == []

    def test_or_逻辑(self) -> None:
        c = PolicyCondition(logic="or", rules=[
            PolicyMatchRule(field="a", value="1"),
        ])
        assert c.logic == "or"
        assert len(c.rules) == 1


# ============================================================
# 4. TransformSpec / EscalationSpec
# ============================================================

class TestTransformSpec:
    def test_默认值(self) -> None:
        t = TransformSpec()
        assert t.mask_fields == []
        assert t.strip_fields == []
        assert t.inject_fields == {}
        assert t.max_value == {}

    def test_完整配置(self) -> None:
        t = TransformSpec(
            mask_fields=["phone", "email"],
            strip_fields=["debug_info"],
            inject_fields={"source": "governance"},
            max_value={"score": 100.0},
        )
        assert len(t.mask_fields) == 2
        assert t.inject_fields["source"] == "governance"


class TestEscalationSpec:
    def test_默认值(self) -> None:
        e = EscalationSpec(target="human_review")
        assert e.timeout_seconds == 300.0
        assert e.auto_resolve is False
        assert e.reason_template == "策略触发升级: {policy_id}"

    def test_超时下界(self) -> None:
        with pytest.raises(ValidationError):
            EscalationSpec(target="x", timeout_seconds=0.5)


# ============================================================
# 5. EvalRequest / EvalResult
# ============================================================

class TestEvalRequest:
    def test_默认值(self) -> None:
        req = EvalRequest()
        assert req.actor == ""
        assert req.action == ""
        assert req.context == {}

    def test_完整请求(self) -> None:
        req = EvalRequest(
            actor="agent-1",
            action="tool_call",
            resource="bkt_compute",
            layer="L6",
            domain="chemistry",
            context={"ip": "10.0.0.1"},
        )
        assert req.context["ip"] == "10.0.0.1"


class TestEvalResult:
    def test_默认值(self) -> None:
        r = EvalResult(decision=PolicyAction.ALLOW)
        assert r.decision == "allow"
        assert r.matched_policy_id is None
        assert r.evaluated_at > 0

    def test_带策略命中(self) -> None:
        r = EvalResult(
            decision=PolicyAction.DENY,
            matched_policy_id="pol-abc",
            matched_policy_name="禁止删除",
            reason="匹配拒绝策略",
        )
        assert r.decision == "deny"
        assert r.matched_policy_id == "pol-abc"

    def test_序列化(self) -> None:
        r = EvalResult(decision=PolicyAction.ESCALATE)
        d = r.model_dump(mode="json")
        assert d["decision"] == "escalate"


# ============================================================
# 6. ViolationRecord
# ============================================================

class TestViolationRecord:
    def test_默认值(self) -> None:
        v = ViolationRecord(policy_id="pol-x")
        assert v.violation_id.startswith("vio-")
        assert v.severity == SeverityLevel.MEDIUM
        assert v.status == ViolationStatus.DETECTED
        assert v.policy_id == "pol-x"

    def test_生命周期_确认_解决(self) -> None:
        v = ViolationRecord(policy_id="p1", actor="a1")
        assert v.status == ViolationStatus.DETECTED
        v.confirm()
        assert v.status == ViolationStatus.CONFIRMED
        v.resolve(by="admin", note="已修复")
        assert v.status == ViolationStatus.RESOLVED
        assert v.resolved_by == "admin"
        assert v.resolution_note == "已修复"
        assert v.resolved_at is not None

    def test_生命周期_忽略(self) -> None:
        v = ViolationRecord(policy_id="p1")
        v.ignore(note="误报")
        assert v.status == ViolationStatus.IGNORED
        assert v.resolution_note == "误报"

    def test_confirm_只能从_detected_转换(self) -> None:
        v = ViolationRecord(policy_id="p1")
        v.resolve(by="x")
        # resolved 后再 confirm 不变
        v.confirm()
        assert v.status == ViolationStatus.RESOLVED


# ============================================================
# 7. ComplianceReport / DimensionScore
# ============================================================

class TestDimensionScore:
    def test_默认值(self) -> None:
        ds = DimensionScore(dimension="策略合规", score=85.0)
        assert ds.weight == 1.0
        assert ds.details == ""

    def test_评分边界(self) -> None:
        ds = DimensionScore(dimension="x", score=100.0)
        assert ds.score == 100.0
        ds2 = DimensionScore(dimension="x", score=0.0)
        assert ds2.score == 0.0
        with pytest.raises(ValidationError):
            DimensionScore(dimension="x", score=101.0)


class TestComplianceReport:
    def test_默认值(self) -> None:
        r = ComplianceReport(period_start=0.0, period_end=100.0)
        assert r.overall_score == 100.0
        assert r.dimensions == []
        assert r.violation_summary["total"] == 0
        assert r.policy_summary["total"] == 0

    def test_compute_overall_score_加权(self) -> None:
        r = ComplianceReport(period_start=0.0, period_end=100.0, dimensions=[
            DimensionScore(dimension="策略", score=80.0, weight=0.6),
            DimensionScore(dimension="幻觉", score=90.0, weight=0.4),
        ])
        score = r.compute_overall_score()
        assert score == 84.0  # (80*0.6 + 90*0.4) / 1.0
        assert r.overall_score == 84.0

    def test_compute_overall_score_空维度(self) -> None:
        r = ComplianceReport(period_start=0.0, period_end=100.0)
        score = r.compute_overall_score()
        assert score == 100.0  # 保持默认

    def test_compute_overall_score_零权重(self) -> None:
        r = ComplianceReport(period_start=0.0, period_end=100.0, dimensions=[
            DimensionScore(dimension="x", score=50.0, weight=0.0),
        ])
        score = r.compute_overall_score()
        # 总权重为 0，返回默认值
        assert score == 100.0


# ============================================================
# 8. ReputationSnapshot
# ============================================================

class TestReputationSnapshot:
    def test_默认值(self) -> None:
        r = ReputationSnapshot(agent_id="a1")
        assert r.overall_trust == 0.5
        assert r.accuracy == 0.5
        assert r.reliability == 0.5
        assert r.compliance_rate == 1.0
        assert r.contribution == 0.0
        assert r.sample_count == 0

    def test_边界值(self) -> None:
        r = ReputationSnapshot(agent_id="a1", overall_trust=0.0, compliance_rate=0.0)
        assert r.overall_trust == 0.0
        with pytest.raises(ValidationError):
            ReputationSnapshot(agent_id="a1", overall_trust=1.1)


# ============================================================
# 9. GovernanceEvent
# ============================================================

class TestGovernanceEvent:
    def test_默认值(self) -> None:
        e = GovernanceEvent(event_type=GovernanceEventType.POLICY_CREATED)
        assert e.event_id.startswith("gevt-")
        assert e.actor == "system"
        assert e.domain == GovernanceDomain.POLICY
        assert e.references == []
        assert e.timestamp > 0

    def test_完整事件(self) -> None:
        e = GovernanceEvent(
            event_type=GovernanceEventType.VIOLATION_DETECTED,
            actor="engine",
            domain=GovernanceDomain.AUDIT,
            detail="检测到违规",
            references=["vio-abc", "pol-xyz"],
            payload={"severity": "high"},
        )
        assert len(e.references) == 2
        assert e.payload["severity"] == "high"


# ============================================================
# 10. PolicyStore CRUD
# ============================================================

class TestPolicyStoreCRUD:
    def setup_method(self) -> None:
        self.store = PolicyStore()

    def test_add_and_get(self) -> None:
        p = GovernancePolicy(name="测试")
        self.store.add_policy(p)
        assert self.store.get_policy(p.policy_id) is p

    def test_add_生成事件日志(self) -> None:
        p = GovernancePolicy(name="测试")
        self.store.add_policy(p)
        log = self.store.get_event_log()
        assert any(e.event_type == GovernanceEventType.POLICY_CREATED for e in log)

    def test_get_不存在返回_None(self) -> None:
        assert self.store.get_policy("ghost") is None

    def test_get_or_raise_不存在抛异常(self) -> None:
        with pytest.raises(PolicyNotFoundError):
            self.store.get_policy_or_raise("ghost")

    def test_update_policy(self) -> None:
        p = GovernancePolicy(name="原始", priority=10)
        self.store.add_policy(p)
        updated = self.store.update_policy(p.policy_id, name="更新后", priority=50)
        assert updated.name == "更新后"
        assert updated.priority == 50

    def test_update_不存在抛异常(self) -> None:
        with pytest.raises(PolicyNotFoundError):
            self.store.update_policy("ghost", name="x")

    def test_remove_policy(self) -> None:
        p = GovernancePolicy(name="删除")
        self.store.add_policy(p)
        removed = self.store.remove_policy(p.policy_id)
        assert removed is p
        assert self.store.get_policy(p.policy_id) is None

    def test_remove_不存在返回_None(self) -> None:
        assert self.store.remove_policy("ghost") is None

    def test_remove_生成事件(self) -> None:
        p = GovernancePolicy(name="x")
        self.store.add_policy(p)
        self.store.remove_policy(p.policy_id)
        log = self.store.get_event_log()
        assert any(e.event_type == GovernanceEventType.POLICY_DELETED for e in log)

    def test_enable_disable(self) -> None:
        p = GovernancePolicy(name="x")
        self.store.add_policy(p)
        self.store.disable_policy(p.policy_id)
        assert self.store.get_policy(p.policy_id).enabled is False
        self.store.enable_policy(p.policy_id)
        assert self.store.get_policy(p.policy_id).enabled is True

    def test_policy_count(self) -> None:
        self.store.add_policy(GovernancePolicy(name="a"))
        self.store.add_policy(GovernancePolicy(name="b"))
        self.store.add_policy(GovernancePolicy(name="c"))
        assert self.store.policy_count == 3
        assert self.store.enabled_policy_count == 3


# ============================================================
# 11. PolicyStore 查询
# ============================================================

class TestPolicyStoreQuery:
    def setup_method(self) -> None:
        self.store = PolicyStore()
        self.p1 = GovernancePolicy(name="global", scope=PolicyScope.GLOBAL, priority=100)
        self.p2 = GovernancePolicy(name="agent-scope", scope=PolicyScope.AGENT, priority=50, tags=["test"])
        self.p3 = GovernancePolicy(name="disabled", enabled=False, priority=200)
        self.p4 = GovernancePolicy(name="tool-scope", scope=PolicyScope.TOOL, priority=75)
        for p in [self.p1, self.p2, self.p3, self.p4]:
            self.store.add_policy(p)

    def test_list_all(self) -> None:
        all_p = self.store.list_policies()
        assert len(all_p) == 4

    def test_list_按作用域(self) -> None:
        result = self.store.list_policies(scope=PolicyScope.AGENT)
        assert len(result) == 1
        assert result[0].name == "agent-scope"

    def test_list_enabled_only(self) -> None:
        result = self.store.list_policies(enabled_only=True)
        assert len(result) == 3

    def test_list_按标签(self) -> None:
        result = self.store.list_policies(tag="test")
        assert len(result) == 1

    def test_list_多条件组合(self) -> None:
        result = self.store.list_policies(scope=PolicyScope.GLOBAL, enabled_only=True)
        assert len(result) == 1

    def test_优先级降序排列(self) -> None:
        result = self.store.list_policies()
        priorities = [p.priority for p in result]
        assert priorities == sorted(priorities, reverse=True)

    def test_get_evaluable_policies(self) -> None:
        result = self.store.get_evaluable_policies()
        assert len(result) == 3
        assert all(p.enabled for p in result)

    def test_get_policies_for_scope(self) -> None:
        result = self.store.get_policies_for_scope(PolicyScope.AGENT)
        # 应包含 global + agent
        scopes = {p.scope for p in result}
        assert PolicyScope.GLOBAL in scopes
        assert PolicyScope.AGENT in scopes


# ============================================================
# 12. 规则匹配
# ============================================================

class TestRuleMatching:
    def setup_method(self) -> None:
        self.store = PolicyStore()

    def _req(self, **kwargs) -> EvalRequest:
        return EvalRequest(
            actor=kwargs.get("actor", ""),
            action=kwargs.get("action", ""),
            resource=kwargs.get("resource", ""),
            layer=kwargs.get("layer", ""),
            domain=kwargs.get("domain", ""),
            context=kwargs.get("context", {}),
        )

    def test_exact_匹配(self) -> None:
        hit = PolicyStore.match_rule(
            "actor", MatchOperator.EXACT, "agent-1", False,
            self._req(actor="agent-1"),
        )
        assert hit is True

    def test_exact_不匹配(self) -> None:
        hit = PolicyStore.match_rule(
            "actor", MatchOperator.EXACT, "agent-1", False,
            self._req(actor="agent-2"),
        )
        assert hit is False

    def test_glob_匹配(self) -> None:
        hit = PolicyStore.match_rule(
            "actor", MatchOperator.GLOB, "tutor-*", False,
            self._req(actor="tutor-01"),
        )
        assert hit is True

    def test_glob_不匹配(self) -> None:
        hit = PolicyStore.match_rule(
            "actor", MatchOperator.GLOB, "tutor-*", False,
            self._req(actor="admin-01"),
        )
        assert hit is False

    def test_regex_匹配(self) -> None:
        hit = PolicyStore.match_rule(
            "resource", MatchOperator.REGEX, r"bkt_.+", False,
            self._req(resource="bkt_compute"),
        )
        assert hit is True

    def test_regex_不匹配(self) -> None:
        hit = PolicyStore.match_rule(
            "resource", MatchOperator.REGEX, r"bkt_.+", False,
            self._req(resource="irt_eval"),
        )
        assert hit is False

    def test_negate_取反(self) -> None:
        hit = PolicyStore.match_rule(
            "layer", MatchOperator.EXACT, "L0", True,
            self._req(layer="L0"),
        )
        assert hit is False  # L0 negate → 不匹配

    def test_negate_取反_不匹配时命中(self) -> None:
        hit = PolicyStore.match_rule(
            "layer", MatchOperator.EXACT, "L0", True,
            self._req(layer="L6"),
        )
        assert hit is True  # L6 != L0, negate → 命中

    def test_上下文字段匹配(self) -> None:
        hit = PolicyStore.match_rule(
            "ip", MatchOperator.EXACT, "10.0.0.1", False,
            self._req(context={"ip": "10.0.0.1"}),
        )
        assert hit is True

    def test_context_覆盖基本字段(self) -> None:
        # context 中的 actor 覆盖基本字段
        hit = PolicyStore.match_rule(
            "actor", MatchOperator.EXACT, "override", False,
            self._req(actor="original", context={"actor": "override"}),
        )
        assert hit is True

    def test_condition_and_全匹配(self) -> None:
        c = PolicyCondition(logic="and", rules=[
            PolicyMatchRule(field="actor", value="a1"),
            PolicyMatchRule(field="layer", value="L6"),
        ])
        assert PolicyStore.match_condition(c, self._req(actor="a1", layer="L6")) is True
        assert PolicyStore.match_condition(c, self._req(actor="a1", layer="L0")) is False

    def test_condition_or_任一匹配(self) -> None:
        c = PolicyCondition(logic="or", rules=[
            PolicyMatchRule(field="actor", value="a1"),
            PolicyMatchRule(field="actor", value="a2"),
        ])
        assert PolicyStore.match_condition(c, self._req(actor="a2")) is True
        assert PolicyStore.match_condition(c, self._req(actor="a3")) is False

    def test_空条件_全匹配(self) -> None:
        c = PolicyCondition()
        assert PolicyStore.match_condition(c, self._req(actor="anyone")) is True

    def test_regex_无效语法_返回_False(self) -> None:
        hit = PolicyStore.match_rule(
            "actor", MatchOperator.REGEX, r"[invalid", False,
            self._req(actor="anything"),
        )
        assert hit is False


# ============================================================
# 13. 违规记录管理
# ============================================================

class TestViolationManagement:
    def setup_method(self) -> None:
        self.store = PolicyStore()

    def test_add_and_get(self) -> None:
        v = ViolationRecord(policy_id="p1", actor="a1", detail="违规")
        self.store.add_violation(v)
        assert self.store.get_violation(v.violation_id) is v

    def test_add_生成事件(self) -> None:
        v = ViolationRecord(policy_id="p1")
        self.store.add_violation(v)
        log = self.store.get_event_log()
        assert any(e.event_type == GovernanceEventType.VIOLATION_DETECTED for e in log)

    def test_query_按_policy_id(self) -> None:
        self.store.add_violation(ViolationRecord(policy_id="p1"))
        self.store.add_violation(ViolationRecord(policy_id="p2"))
        result = self.store.query_violations(policy_id="p1")
        assert len(result) == 1

    def test_query_按_actor(self) -> None:
        self.store.add_violation(ViolationRecord(policy_id="p1", actor="a1"))
        self.store.add_violation(ViolationRecord(policy_id="p2", actor="a2"))
        result = self.store.query_violations(actor="a1")
        assert len(result) == 1

    def test_query_按_severity(self) -> None:
        self.store.add_violation(ViolationRecord(policy_id="p1", severity=SeverityLevel.HIGH))
        self.store.add_violation(ViolationRecord(policy_id="p2", severity=SeverityLevel.LOW))
        result = self.store.query_violations(severity=SeverityLevel.HIGH)
        assert len(result) == 1

    def test_query_按_status(self) -> None:
        v = ViolationRecord(policy_id="p1")
        self.store.add_violation(v)
        v.resolve(by="admin")
        result = self.store.query_violations(status=ViolationStatus.RESOLVED)
        assert len(result) == 1
        result2 = self.store.query_violations(status=ViolationStatus.DETECTED)
        assert len(result2) == 0

    def test_query_limit(self) -> None:
        for i in range(10):
            self.store.add_violation(ViolationRecord(policy_id=f"p{i}"))
        result = self.store.query_violations(limit=3)
        assert len(result) == 3

    def test_query_按时间倒序(self) -> None:
        import time as _t
        self.store.add_violation(ViolationRecord(policy_id="p1"))
        _t.sleep(0.002)  # 保证时间戳可区分 (同毫秒创建排序不稳定)
        self.store.add_violation(ViolationRecord(policy_id="p2"))
        result = self.store.query_violations()
        assert result[0].policy_id == "p2"  # 最新在前

    def test_violation_count(self) -> None:
        self.store.add_violation(ViolationRecord(policy_id="p1"))
        self.store.add_violation(ViolationRecord(policy_id="p2"))
        assert self.store.violation_count == 2
        assert self.store.active_violation_count == 2

    def test_active_violation_count_排除_resolved(self) -> None:
        v1 = ViolationRecord(policy_id="p1")
        v2 = ViolationRecord(policy_id="p2")
        self.store.add_violation(v1)
        self.store.add_violation(v2)
        v1.resolve(by="admin")
        assert self.store.active_violation_count == 1


# ============================================================
# 14. 事件日志
# ============================================================

class TestEventLog:
    def setup_method(self) -> None:
        self.store = PolicyStore()

    def test_事件日志_自动记录(self) -> None:
        p = GovernancePolicy(name="x")
        self.store.add_policy(p)
        log = self.store.get_event_log()
        assert len(log) >= 1

    def test_事件日志_limit(self) -> None:
        for i in range(10):
            self.store.add_policy(GovernancePolicy(name=f"p{i}"))
        log = self.store.get_event_log(limit=3)
        assert len(log) == 3


# ============================================================
# 15. 统计与导出
# ============================================================

class TestStatsAndExport:
    def setup_method(self) -> None:
        self.store = PolicyStore()
        self.store.add_policy(GovernancePolicy(name="a", scope=PolicyScope.GLOBAL))
        self.store.add_policy(GovernancePolicy(name="b", scope=PolicyScope.AGENT, enabled=False))
        self.store.add_violation(ViolationRecord(policy_id="p1", severity=SeverityLevel.HIGH))

    def test_get_stats(self) -> None:
        stats = self.store.get_stats()
        assert stats["policies"]["total"] == 2
        assert stats["policies"]["enabled"] == 1
        assert stats["policies"]["disabled"] == 1
        assert stats["violations"]["total"] == 1

    def test_stats_by_scope(self) -> None:
        stats = self.store.get_stats()
        by_scope = stats["policies"]["by_scope"]
        assert by_scope["global"] == 1
        assert by_scope["agent"] == 1

    def test_stats_by_action(self) -> None:
        stats = self.store.get_stats()
        by_action = stats["policies"]["by_action"]
        assert by_action["allow"] == 2

    def test_record_eval(self) -> None:
        self.store.record_eval()
        self.store.record_eval(denied=True)
        self.store.record_eval(escalated=True)
        stats = self.store.get_stats()
        assert stats["evaluations"]["total"] == 3
        assert stats["evaluations"]["denied"] == 1
        assert stats["evaluations"]["escalated"] == 1

    def test_export_all(self) -> None:
        data = self.store.export_all()
        assert "policies" in data
        assert "violations" in data
        assert "stats" in data
        assert len(data["policies"]) == 2

    def test_export_summary(self) -> None:
        data = self.store.export_summary()
        assert "stats" in data
        assert "policy_ids" in data
        assert len(data["policy_ids"]) == 2


# ============================================================
# 16. 容量控制
# ============================================================

class TestCapacityControl:
    def test_违规记录容量淘汰(self) -> None:
        """超过 _VIOLATION_CAP 时淘汰最早的."""
        from dy3_polaris.l0.governance import policy_store
        original_cap = policy_store._VIOLATION_CAP
        try:
            policy_store._VIOLATION_CAP = 5
            store = PolicyStore()
            violations = []
            for i in range(7):
                v = ViolationRecord(policy_id=f"p{i}", detail=f"违规{i}")
                store.add_violation(v)
                violations.append(v)
                time.sleep(0.001)  # 确保时间戳不同
            # 容量为 5，添加了 7 个，应淘汰最早的 2 个
            assert store.violation_count == 5
            # 最早的应被淘汰
            assert store.get_violation(violations[0].violation_id) is None
            assert store.get_violation(violations[1].violation_id) is None
            # 最新的应保留
            assert store.get_violation(violations[6].violation_id) is not None
        finally:
            policy_store._VIOLATION_CAP = original_cap

    def test_事件日志容量截断(self) -> None:
        from dy3_polaris.l0.governance import policy_store
        original_cap = policy_store._EVENT_LOG_CAP
        try:
            policy_store._EVENT_LOG_CAP = 3
            store = PolicyStore()
            for i in range(5):
                store.add_policy(GovernancePolicy(name=f"p{i}"))
            log = store.get_event_log()
            assert len(log) == 3
        finally:
            policy_store._EVENT_LOG_CAP = original_cap


# ============================================================
# 17. 治理异常体系
# ============================================================

class TestGovernanceExceptions:
    def test_继承_L6Error(self) -> None:
        assert issubclass(GovernanceError, Exception)
        err = GovernanceError()
        assert err.code == "GOVERNANCE_ERROR"

    def test_错误码格式(self) -> None:
        assert PolicyNotFoundError("x").code == "GOVERNANCE_POLICY_NOT_FOUND"
        assert PolicyConflictError("x").code == "GOVERNANCE_POLICY_CONFLICT"
        assert ViolationError("v", "p").code == "GOVERNANCE_VIOLATION"
        assert ComplianceCheckFailedError(50, 60).code == "GOVERNANCE_COMPLIANCE_FAILED"
        assert PolicyValidationError("x").code == "GOVERNANCE_POLICY_VALIDATION_ERROR"
        assert RuleSyntaxError("f", "op", "v").code == "GOVERNANCE_RULE_SYNTAX_ERROR"

    def test_jsonrpc_code_映射(self) -> None:
        assert GovernanceError()._jsonrpc_code() == -32100
        assert PolicyNotFoundError("x")._jsonrpc_code() == -32101
        assert PolicyConflictError("x")._jsonrpc_code() == -32102
        assert ViolationError("v", "p")._jsonrpc_code() == -32103
        assert ComplianceCheckFailedError(50, 60)._jsonrpc_code() == -32104
        assert PolicyValidationError("x")._jsonrpc_code() == -32105
        assert RuleSyntaxError("f", "op", "v")._jsonrpc_code() == -32106

    def test_to_json_rpc_error(self) -> None:
        err = PolicyNotFoundError("pol-123")
        j = err.to_json_rpc_error()
        assert j["code"] == -32101
        assert j["message"] == "GOVERNANCE_POLICY_NOT_FOUND"
        assert j["data"]["detail"] == "策略未找到: pol-123"
        assert j["data"]["policy_id"] == "pol-123"

    def test_上下文字段(self) -> None:
        err = ViolationError("vio-x", "pol-y", context={"extra": 42})
        j = err.to_json_rpc_error()
        assert j["data"]["violation_id"] == "vio-x"
        assert j["data"]["policy_id"] == "pol-y"
        assert j["data"]["extra"] == 42

    def test_默认_detail_自动生成(self) -> None:
        err = ComplianceCheckFailedError(45.5, 60.0)
        assert "45.5" in err.detail
        assert "60.0" in err.detail


# ============================================================
# 18. 规则语法校验
# ============================================================

class TestRuleValidation:
    def setup_method(self) -> None:
        self.store = PolicyStore()

    def test_合法_regex_通过(self) -> None:
        p = GovernancePolicy(
            name="合法",
            condition=PolicyCondition(rules=[
                PolicyMatchRule(field="x", operator=MatchOperator.REGEX, value=r"\d+")
            ]),
        )
        self.store.add_policy(p)  # 不应抛异常

    def test_非法_regex_拒绝(self) -> None:
        p = GovernancePolicy(
            name="非法",
            condition=PolicyCondition(rules=[
                PolicyMatchRule(field="x", operator=MatchOperator.REGEX, value=r"[invalid")
            ]),
        )
        with pytest.raises(RuleSyntaxError):
            self.store.add_policy(p)

    def test_glob_不预校验(self) -> None:
        p = GovernancePolicy(
            name="glob",
            condition=PolicyCondition(rules=[
                PolicyMatchRule(field="x", operator=MatchOperator.GLOB, value="*?[invalid")
            ]),
        )
        # glob 不预校验，直接通过
        self.store.add_policy(p)

    def test_escalate_无_target_拒绝(self) -> None:
        p = GovernancePolicy(
            name="无目标",
            action=PolicyAction.ESCALATE,
            escalation=EscalationSpec(target=""),
        )
        with pytest.raises(PolicyValidationError):
            self.store.add_policy(p)

    def test_update_时重新校验(self) -> None:
        p = GovernancePolicy(name="x")
        self.store.add_policy(p)
        # 更新为非法正则应失败
        with pytest.raises(RuleSyntaxError):
            self.store.update_policy(p.policy_id, condition=PolicyCondition(rules=[
                PolicyMatchRule(field="x", operator=MatchOperator.REGEX, value=r"[bad")
            ]))


# ============================================================
# 19. Config 治理配置
# ============================================================

class TestGovernanceConfig:
    def teardown_method(self) -> None:
        reset_config()

    def test_默认配置值(self) -> None:
        reset_config()
        config = L6Config()
        assert config.governance_enabled is True
        assert config.governance_default_action == "allow"
        assert config.governance_violation_log_max == 2000
        assert config.governance_event_log_max == 500
        assert config.governance_compliance_threshold == 60.0

    def test_get_governance_config(self) -> None:
        reset_config()
        config = L6Config()
        gc = config.get_governance_config()
        assert gc["enabled"] is True
        assert gc["default_action"] == "allow"
        assert gc["violation_log_max"] == 2000
        assert gc["compliance_threshold"] == 60.0

    def test_环境变量覆盖(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_config()
        monkeypatch.setenv("DY3_L6_GOVERNANCE_ENABLED", "false")
        monkeypatch.setenv("DY3_L6_GOVERNANCE_COMPLIANCE_THRESHOLD", "80.0")
        config = L6Config()
        assert config.governance_enabled is False
        assert config.governance_compliance_threshold == 80.0
        reset_config()
