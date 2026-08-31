"""G2 策略评估引擎测试.

覆盖 PolicyEvaluator 的全部能力：
- 基础评估、优先级排序、deny-override
- 钩子管道（Pre/Post）、缓存机制
- 转换应用、冲突检测、度量收集
- 批量评估、快速合规检查
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from dy3_polaris.l0.governance.models import (
    EvalRequest,
    EvalResult,
    GovernancePolicy,
    MatchOperator,
    PolicyAction,
    PolicyCondition,
    PolicyMatchRule,
    PolicyScope,
    SeverityLevel,
    TransformSpec,
    ViolationStatus,
)
from dy3_polaris.l0.governance.policy_store import PolicyStore
from dy3_polaris.l0.governance.evaluator import (
    EvaluatorMetrics,
    PolicyEvaluator,
    _DecisionCache,
)


# ============================================================
# 辅助工具
# ============================================================


def _make_policy(
    name: str = "测试策略",
    action: PolicyAction = PolicyAction.ALLOW,
    scope: PolicyScope = PolicyScope.GLOBAL,
    priority: int = 0,
    rules: list[PolicyMatchRule] | None = None,
    enabled: bool = True,
) -> GovernancePolicy:
    """快捷构建策略."""
    return GovernancePolicy(
        name=name,
        action=action,
        scope=scope,
        priority=priority,
        condition=PolicyCondition(rules=rules or []),
        enabled=enabled,
    )


def _make_request(
    actor: str = "user-001",
    action: str = "tool_call",
    resource: str = "bkt_compute",
    layer: str = "L5",
    domain: str = "",
    context: dict[str, Any] | None = None,
) -> EvalRequest:
    """快捷构建评估请求."""
    return EvalRequest(
        actor=actor,
        action=action,
        resource=resource,
        layer=layer,
        domain=domain,
        context=context or {},
    )


class _CountingPreHook:
    """计数前置钩子."""

    def __init__(self, result: EvalResult | None = None) -> None:
        self.call_count = 0
        self._result = result

    def __call__(self, request: EvalRequest) -> EvalResult | None:
        self.call_count += 1
        return self._result


class _AppendReasonPostHook:
    """追加原因后置钩子."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, request: EvalRequest, result: EvalResult) -> EvalResult:
        self.call_count += 1
        result.reason += " [post-hook]"
        return result


# ============================================================
# 测试类
# ============================================================


class TestBasicEvaluation:
    """基础评估测试."""

    def test_无策略_默认允许(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        req = _make_request()
        result = ev.evaluate(req)
        assert result.decision == PolicyAction.ALLOW
        assert result.matched_policy_id is None

    def test_自定义默认动作(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store, default_action=PolicyAction.DENY)
        req = _make_request()
        result = ev.evaluate(req)
        assert result.decision == PolicyAction.DENY

    def test_全局策略_空条件_命中(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy("日志策略", PolicyAction.LOG, priority=10)
        store.add_policy(p)
        req = _make_request()
        result = ev.evaluate(req)
        assert result.decision == PolicyAction.LOG
        assert result.matched_policy_id == p.policy_id
        assert result.matched_policy_name == "日志策略"

    def test_禁用策略_不参与评估(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy("禁用策略", PolicyAction.DENY, priority=100)
        p.enabled = False
        store.add_policy(p)
        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.ALLOW

    def test_条件不匹配_跳过(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy(
            "仅限tutor",
            PolicyAction.DENY,
            rules=[PolicyMatchRule(field="actor", operator="exact", value="tutor-001")],
        )
        store.add_policy(p)
        result = ev.evaluate(_make_request(actor="lab-001"))
        assert result.decision == PolicyAction.ALLOW

    def test_条件匹配_拒绝(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy(
            "仅限tutor",
            PolicyAction.DENY,
            rules=[PolicyMatchRule(field="actor", operator="exact", value="tutor-001")],
        )
        store.add_policy(p)
        result = ev.evaluate(_make_request(actor="tutor-001"))
        assert result.decision == PolicyAction.DENY


class TestDenyOverride:
    """Cedar 式 deny-override 测试."""

    def test_allow_和_deny_同匹配_deny优先(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        allow_p = _make_policy("允许", PolicyAction.ALLOW, priority=100)
        deny_p = _make_policy(
            "拒绝",
            PolicyAction.DENY,
            priority=50,
            rules=[PolicyMatchRule(field="resource", operator="glob", value="bkt_*")],
        )
        store.add_policy(allow_p)
        store.add_policy(deny_p)
        result = ev.evaluate(_make_request(resource="bkt_compute"))
        assert result.decision == PolicyAction.DENY
        assert result.matched_policy_id == deny_p.policy_id

    def test_deny优先级低于_allow_仍然deny(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        allow_p = _make_policy("高优先允许", PolicyAction.ALLOW, priority=200)
        deny_p = _make_policy(
            "低优先拒绝",
            PolicyAction.DENY,
            priority=1,
            rules=[PolicyMatchRule(field="resource", operator="glob", value="bkt_*")],
        )
        store.add_policy(allow_p)
        store.add_policy(deny_p)
        result = ev.evaluate(_make_request(resource="bkt_compute"))
        assert result.decision == PolicyAction.DENY

    def test_仅allow匹配_无deny(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy("允许", PolicyAction.ALLOW, priority=10)
        store.add_policy(p)
        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.ALLOW
        assert result.matched_policy_id == p.policy_id

    def test_仅deny匹配(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy("拒绝", PolicyAction.DENY, priority=10)
        store.add_policy(p)
        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.DENY


class TestPriorityOrdering:
    """优先级排序测试."""

    def test_最高优先级策略决定结果(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p_low = _make_policy("低优先", PolicyAction.LOG, priority=0)
        p_high = _make_policy("高优先", PolicyAction.ESCALATE, priority=500)
        store.add_policy(p_low)
        store.add_policy(p_high)
        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.ESCALATE
        assert result.matched_policy_id == p_high.policy_id

    def test_同优先级_按添加顺序(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p1 = _make_policy("策略A", PolicyAction.LOG, priority=100)
        p2 = _make_policy("策略B", PolicyAction.TRANSFORM, priority=100)
        store.add_policy(p1)
        store.add_policy(p2)
        result = ev.evaluate(_make_request())
        assert result.decision in (PolicyAction.LOG, PolicyAction.TRANSFORM)


class TestScopeFiltering:
    """作用域过滤测试."""

    def test_tool作用域策略_仅在工具请求中匹配(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        tool_policy = _make_policy(
            "工具策略",
            PolicyAction.DENY,
            scope=PolicyScope.TOOL,
            rules=[PolicyMatchRule(field="resource", operator="exact", value="ext_search")],
        )
        store.add_policy(tool_policy)

        # 工具请求 → 命中
        req_tool = _make_request(resource="ext_search", context={"tool_name": "ext_search"})
        r1 = ev.evaluate(req_tool, dry_run=True)
        assert r1.decision == PolicyAction.DENY

        # 非工具请求 → 不命中
        ev.invalidate_cache()
        req_agent = _make_request(actor="agent-001", resource="decision")
        r2 = ev.evaluate(req_agent, dry_run=True)
        assert r2.decision == PolicyAction.ALLOW

    def test_global策略_始终参与(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        global_p = _make_policy("全局日志", PolicyAction.LOG, scope=PolicyScope.GLOBAL, priority=1)
        store.add_policy(global_p)
        req = _make_request(actor="agent-001")
        result = ev.evaluate(req)
        assert result.decision == PolicyAction.LOG


class TestDryRun:
    """干运行模式测试."""

    def test_deny_dry_run_不创建违规(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy("拒绝", PolicyAction.DENY, priority=10)
        store.add_policy(p)
        result = ev.evaluate(_make_request(), dry_run=True)
        assert result.decision == PolicyAction.DENY
        assert store.violation_count == 0

    def test_deny_非dry_run_创建违规(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy("拒绝", PolicyAction.DENY, priority=10)
        store.add_policy(p)
        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.DENY
        assert store.violation_count == 1

    def test_escalate_dry_run_不创建违规(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = GovernancePolicy(
            name="升级策略",
            action=PolicyAction.ESCALATE,
            escalation={"target": "human_review"},
        )
        store.add_policy(p)
        result = ev.evaluate(_make_request(), dry_run=True)
        assert result.decision == PolicyAction.ESCALATE
        assert store.violation_count == 0

    def test_escalate_非dry_run_创建违规(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = GovernancePolicy(
            name="升级策略",
            action=PolicyAction.ESCALATE,
            escalation={"target": "human_review"},
        )
        store.add_policy(p)
        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.ESCALATE
        assert store.violation_count == 1

    def test_实例级_dry_run(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store, dry_run=True)
        p = _make_policy("拒绝", PolicyAction.DENY, priority=10)
        store.add_policy(p)
        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.DENY
        assert store.violation_count == 0


class TestCacheMechanism:
    """缓存机制测试."""

    def test_缓存命中(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store, cache_ttl_seconds=300)
        p = _make_policy("日志", PolicyAction.LOG, priority=10)
        store.add_policy(p)
        req = _make_request()

        r1 = ev.evaluate(req)
        assert ev.metrics.cache_misses.value == 1
        assert ev.metrics.cache_hits.value == 0

        r2 = ev.evaluate(req)
        assert ev.metrics.cache_hits.value == 1
        assert r2.decision == r1.decision

    def test_禁用缓存(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy("日志", PolicyAction.LOG, priority=10)
        store.add_policy(p)
        req = _make_request()

        ev.evaluate(req, use_cache=False)
        ev.evaluate(req, use_cache=False)
        assert ev.metrics.cache_hits.value == 0
        assert ev.metrics.cache_misses.value == 0

    def test_缓存失效_策略变更后(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store, cache_ttl_seconds=300)
        p = _make_policy("允许", PolicyAction.ALLOW, priority=10)
        store.add_policy(p)
        req = _make_request()

        r1 = ev.evaluate(req)
        assert r1.decision == PolicyAction.ALLOW

        # 更新策略为 deny
        store.update_policy(p.policy_id, action=PolicyAction.DENY)
        ev.invalidate_cache()

        r2 = ev.evaluate(req)
        assert r2.decision == PolicyAction.DENY

    def test_缓存容量限制(self) -> None:
        cache = _DecisionCache(max_size=3, ttl_seconds=300)
        req = _make_request()
        for i in range(5):
            r = _make_request(actor=f"agent-{i:03d}")
            cache.put(r, EvalResult(decision=PolicyAction.ALLOW))
        assert cache.size == 3

    def test_缓存TTL过期(self) -> None:
        cache = _DecisionCache(max_size=100, ttl_seconds=0.01)
        req = _make_request()
        cache.put(req, EvalResult(decision=PolicyAction.ALLOW))
        time.sleep(0.05)
        assert cache.get(req) is None


class TestPreHook:
    """前置钩子测试."""

    def test_pre_hook_短路(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        hook_result = EvalResult(
            decision=PolicyAction.DENY,
            reason="pre-hook deny",
        )
        hook = _CountingPreHook(result=hook_result)
        ev.add_pre_hook(hook)

        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.DENY
        assert result.reason == "pre-hook deny"
        assert hook.call_count == 1
        assert ev.metrics.pre_hook_shortcircuits.value == 1

    def test_pre_hook_不短路(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        hook = _CountingPreHook(result=None)
        ev.add_pre_hook(hook)

        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.ALLOW
        assert hook.call_count == 1
        assert ev.metrics.pre_hook_shortcircuits.value == 0

    def test_多pre_hook_第一个短路(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        hook1 = _CountingPreHook(result=EvalResult(decision=PolicyAction.DENY))
        hook2 = _CountingPreHook(result=None)
        ev.add_pre_hook(hook1)
        ev.add_pre_hook(hook2)

        ev.evaluate(_make_request())
        assert hook1.call_count == 1
        assert hook2.call_count == 0

    def test_pre_hook_移除(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        hook = _CountingPreHook(result=EvalResult(decision=PolicyAction.DENY))
        ev.add_pre_hook(hook)
        ev.remove_pre_hook(hook)
        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.ALLOW


class TestPostHook:
    """后置钩子测试."""

    def test_post_hook_修改结果(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        hook = _AppendReasonPostHook()
        ev.add_post_hook(hook)

        result = ev.evaluate(_make_request())
        assert "[post-hook]" in result.reason
        assert hook.call_count == 1

    def test_多post_hook_链式执行(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        h1 = _AppendReasonPostHook()
        h2 = _AppendReasonPostHook()
        ev.add_post_hook(h1)
        ev.add_post_hook(h2)

        result = ev.evaluate(_make_request())
        assert result.reason.count("[post-hook]") == 2

    def test_post_hook_移除(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        hook = _AppendReasonPostHook()
        ev.add_post_hook(hook)
        ev.remove_post_hook(hook)
        result = ev.evaluate(_make_request())
        assert "[post-hook]" not in result.reason


class TestTransformApplication:
    """转换应用测试."""

    def test_strip_fields(self) -> None:
        data = {"a": 1, "b": 2, "c": 3}
        spec = TransformSpec(strip_fields=["b", "c"])
        result = PolicyEvaluator.apply_transform(data, spec)
        assert result == {"a": 1}
        assert data == {"a": 1, "b": 2, "c": 3}  # 原数据不变

    def test_mask_fields_长文本(self) -> None:
        data = {"phone": "13812345678", "name": "张三"}
        spec = TransformSpec(mask_fields=["phone"])
        result = PolicyEvaluator.apply_transform(data, spec)
        assert result["phone"] == "13***78"
        assert result["name"] == "张三"

    def test_mask_fields_短文本(self) -> None:
        data = {"token": "abc"}
        spec = TransformSpec(mask_fields=["token"])
        result = PolicyEvaluator.apply_transform(data, spec)
        assert result["token"] == "***"

    def test_inject_fields(self) -> None:
        data = {"x": 1}
        spec = TransformSpec(inject_fields={"y": 2, "z": 3})
        result = PolicyEvaluator.apply_transform(data, spec)
        assert result == {"x": 1, "y": 2, "z": 3}

    def test_max_value(self) -> None:
        data = {"score": 150.0, "count": 5}
        spec = TransformSpec(max_value={"score": 100.0, "count": 10})
        result = PolicyEvaluator.apply_transform(data, spec)
        assert result["score"] == 100.0
        assert result["count"] == 5

    def test_组合转换(self) -> None:
        data = {"a": 1, "b": "13812345678", "c": 200}
        spec = TransformSpec(
            strip_fields=["a"],
            mask_fields=["b"],
            inject_fields={"d": "injected"},
            max_value={"c": 100},
        )
        result = PolicyEvaluator.apply_transform(data, spec)
        assert "a" not in result
        assert result["b"] == "13***78"
        assert result["c"] == 100
        assert result["d"] == "injected"

    def test_transform结果携带规范(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store, dry_run=True)
        p = GovernancePolicy(
            name="转换策略",
            action=PolicyAction.TRANSFORM,
            transform={"mask_fields": ["phone"], "inject_fields": {"masked": True}},
        )
        store.add_policy(p)
        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.TRANSFORM
        assert result.transform is not None
        assert "phone" in result.transform.mask_fields


class TestEscalationResult:
    """升级结果测试."""

    def test_escalate结果携带规范(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store, dry_run=True)
        p = GovernancePolicy(
            name="升级策略",
            action=PolicyAction.ESCALATE,
            escalation={"target": "human_review", "timeout_seconds": 600},
        )
        store.add_policy(p)
        result = ev.evaluate(_make_request())
        assert result.decision == PolicyAction.ESCALATE
        assert result.escalation is not None
        assert result.escalation.target == "human_review"
        assert result.escalation.timeout_seconds == 600


class TestViolationCreation:
    """违规记录创建测试."""

    def test_deny创建违规_severity_high(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy("拒绝策略", PolicyAction.DENY, priority=10)
        store.add_policy(p)
        ev.evaluate(_make_request())
        assert store.violation_count == 1
        vio = store.query_violations(limit=1)[0]
        assert vio.policy_id == p.policy_id
        assert vio.severity == SeverityLevel.HIGH
        assert vio.status == ViolationStatus.DETECTED

    def test_escalate创建违规_severity_medium(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = GovernancePolicy(
            name="升级策略",
            action=PolicyAction.ESCALATE,
            escalation={"target": "admin"},
        )
        store.add_policy(p)
        ev.evaluate(_make_request())
        assert store.violation_count == 1
        vio = store.query_violations(limit=1)[0]
        assert vio.severity == SeverityLevel.MEDIUM

    def test_allow不创建违规(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        store.add_policy(_make_policy("允许策略", PolicyAction.ALLOW, priority=10))
        ev.evaluate(_make_request())
        assert store.violation_count == 0


class TestEventGeneration:
    """评估事件生成测试."""

    def test_评估生成事件(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        store.add_policy(_make_policy("日志", PolicyAction.LOG, priority=10))
        ev.evaluate(_make_request())
        events = store.get_event_log()
        eval_events = [e for e in events if e.event_type.value == "policy_evaluated"]
        assert len(eval_events) >= 1


class TestMetrics:
    """度量收集测试."""

    def test_评估计数(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        ev.evaluate(_make_request())
        ev.evaluate(_make_request(actor="user-002"), use_cache=False)
        assert ev.metrics.evaluations_total.value == 2
        assert ev.metrics.allows.value == 2

    def test_deny计数(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        store.add_policy(_make_policy("拒绝", PolicyAction.DENY, priority=10))
        ev.evaluate(_make_request())
        assert ev.metrics.denials.value == 1

    def test_escalate计数(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store, dry_run=True)
        p = GovernancePolicy(
            name="升级",
            action=PolicyAction.ESCALATE,
            escalation={"target": "admin"},
        )
        store.add_policy(p)
        ev.evaluate(_make_request())
        assert ev.metrics.escalations.value == 1

    def test_缓存命中率(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        req = _make_request()
        ev.evaluate(req)
        ev.evaluate(req)
        assert ev.metrics.cache_hit_rate > 0

    def test_延迟记录(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        ev.evaluate(_make_request())
        assert ev.metrics.avg_latency_ms >= 0  # 环境计时精度: 快速操作可能为 0

    def test_export_metrics(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        ev.evaluate(_make_request())
        m = ev.export_metrics()
        assert "evaluations" in m
        assert "cache" in m
        assert "latency_ms" in m
        assert m["evaluations"]["total"] == 1


class TestConflictDetection:
    """冲突检测测试."""

    def test_同优先级同作用域_allow_deny冲突(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p1 = _make_policy("允许A", PolicyAction.ALLOW, scope=PolicyScope.GLOBAL, priority=0)
        p2 = _make_policy("拒绝B", PolicyAction.DENY, scope=PolicyScope.GLOBAL, priority=0)
        store.add_policy(p1)
        store.add_policy(p2)
        conflicts = ev.detect_conflicts()
        assert len(conflicts) >= 1
        assert conflicts[0]["type"] == "allow_deny_conflict"

    def test_不同作用域_不算冲突(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p1 = _make_policy("允许", PolicyAction.ALLOW, scope=PolicyScope.GLOBAL, priority=0)
        p2 = _make_policy("拒绝", PolicyAction.DENY, scope=PolicyScope.TOOL, priority=0)
        store.add_policy(p1)
        store.add_policy(p2)
        conflicts = ev.detect_conflicts()
        assert len(conflicts) == 0

    def test_无冲突(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        store.add_policy(_make_policy("允许", PolicyAction.ALLOW, priority=10))
        store.add_policy(_make_policy("日志", PolicyAction.LOG, priority=5))
        assert ev.detect_conflicts() == []


class TestBatchEvaluation:
    """批量评估测试."""

    def test_批量评估(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store, dry_run=True)
        store.add_policy(_make_policy("拒绝ext", PolicyAction.DENY, scope=PolicyScope.TOOL, priority=10,
                               rules=[PolicyMatchRule(field="resource", operator="glob", value="ext_*")]))
        requests = [
            _make_request(resource="ext_search", context={"tool_name": "ext_search"}),
            _make_request(resource="bkt_compute", context={"tool_name": "bkt_compute"}),
            _make_request(resource="ext_download", context={"tool_name": "ext_download"}),
        ]
        results = ev.evaluate_batch(requests)
        assert len(results) == 3
        assert results[0].decision == PolicyAction.DENY
        assert results[1].decision == PolicyAction.ALLOW
        assert results[2].decision == PolicyAction.DENY


class TestQuickComplianceCheck:
    """快速合规检查测试."""

    def test_基础合规检查(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        store.add_policy(_make_policy("P1", PolicyAction.ALLOW))
        check = ev.quick_compliance_check()
        assert check["policies_total"] == 1
        assert check["policies_enabled"] == 1
        assert check["violations_total"] == 0
        assert "evaluator_metrics" in check


class TestScopeInference:
    """作用域推断测试."""

    def test_tool_context推断(self) -> None:
        req = _make_request(context={"tool_name": "bkt_compute"})
        assert PolicyEvaluator._infer_scope(req) == PolicyScope.TOOL

    def test_tool_resource前缀推断(self) -> None:
        req = _make_request(resource="tool:bkt_compute")
        assert PolicyEvaluator._infer_scope(req) == PolicyScope.TOOL

    def test_agent_context推断(self) -> None:
        req = _make_request(context={"agent_id": "tutor-001"}, layer="", domain="")
        assert PolicyEvaluator._infer_scope(req) == PolicyScope.AGENT

    def test_agent_actor前缀推断(self) -> None:
        req = _make_request(actor="agent-001", layer="", domain="")
        assert PolicyEvaluator._infer_scope(req) == PolicyScope.AGENT

    def test_layer推断(self) -> None:
        req = _make_request(layer="L4", domain="")
        assert PolicyEvaluator._infer_scope(req) == PolicyScope.LAYER

    def test_domain推断(self) -> None:
        req = _make_request(domain="chemistry", layer="")
        assert PolicyEvaluator._infer_scope(req) == PolicyScope.DOMAIN

    def test_global兜底(self) -> None:
        req = _make_request(layer="", domain="")
        assert PolicyEvaluator._infer_scope(req) == PolicyScope.GLOBAL


class TestEvaluatorMetricsUnit:
    """EvaluatorMetrics 单元测试."""

    def test_cache_hit_rate_无数据(self) -> None:
        m = EvaluatorMetrics()
        assert m.cache_hit_rate == 0.0

    def test_cache_hit_rate_计算正确(self) -> None:
        m = EvaluatorMetrics()
        m.cache_hits.inc(3)
        m.cache_misses.inc(1)
        assert abs(m.cache_hit_rate - 0.75) < 0.001

    def test_avg_latency_无数据(self) -> None:
        m = EvaluatorMetrics()
        assert m.avg_latency_ms == 0.0

    def test_p99_无数据(self) -> None:
        m = EvaluatorMetrics()
        assert m._latency.p99 == 0.0


class TestDecisionCacheUnit:
    """_DecisionCache 单元测试."""

    def test_invalidate(self) -> None:
        cache = _DecisionCache(max_size=100, ttl_seconds=300)
        req = _make_request()
        cache.put(req, EvalResult(decision=PolicyAction.ALLOW))
        assert cache.size == 1
        cache.invalidate()
        assert cache.size == 0

    def test_不同请求不同键(self) -> None:
        cache = _DecisionCache(max_size=100, ttl_seconds=300)
        r1 = _make_request(actor="a")
        r2 = _make_request(actor="b")
        cache.put(r1, EvalResult(decision=PolicyAction.ALLOW))
        cache.put(r2, EvalResult(decision=PolicyAction.DENY))
        assert cache.size == 2
        assert cache.get(r1).decision == PolicyAction.ALLOW
        assert cache.get(r2).decision == PolicyAction.DENY


class TestHookProperties:
    """钩子列表属性测试."""

    def test_pre_hooks_返回副本(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        hooks = ev.pre_hooks
        assert hooks == []
        hooks.append(None)  # 修改副本不影响内部
        assert len(ev.pre_hooks) == 0

    def test_post_hooks_返回副本(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        hooks = ev.post_hooks
        assert hooks == []
        hooks.append(None)
        assert len(ev.post_hooks) == 0


class TestLogAction:
    """LOG 动作测试."""

    def test_log_不创建违规(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        store.add_policy(_make_policy("日志", PolicyAction.LOG, priority=10))
        ev.evaluate(_make_request())
        assert store.violation_count == 0

    def test_log_记录评估(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        store.add_policy(_make_policy("日志", PolicyAction.LOG, priority=10))
        ev.evaluate(_make_request())
        assert ev.metrics.logs.value == 1


class TestReasonConstruction:
    """决策原因构建测试."""

    def test_匹配策略_包含名称和ID(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy("安全策略", PolicyAction.DENY, priority=42)
        store.add_policy(p)
        result = ev.evaluate(_make_request(), dry_run=True)
        assert "安全策略" in result.reason
        assert p.policy_id in result.reason
        assert "priority=42" in result.reason


class TestContextBasedMatching:
    """上下文字段匹配测试."""

    def test_上下文字段匹配(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        p = _make_policy(
            "限速",
            PolicyAction.ESCALATE,
            rules=[PolicyMatchRule(field="rate_limit_exceeded", operator="exact", value="true")],
        )
        store.add_policy(p)
        req = _make_request(context={"rate_limit_exceeded": "true"})
        result = ev.evaluate(req, dry_run=True)
        assert result.decision == PolicyAction.ESCALATE

    def test_上下文覆盖_basic字段(self) -> None:
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        # 规则匹配 context 中的 actor
        p = _make_policy(
            "代理检查",
            PolicyAction.DENY,
            rules=[PolicyMatchRule(field="actor", operator="exact", value="proxy-agent")],
        )
        store.add_policy(p)
        # context.actor 覆盖 request.actor
        req = _make_request(actor="real-user", context={"actor": "proxy-agent"})
        result = ev.evaluate(req, dry_run=True)
        assert result.decision == PolicyAction.DENY
