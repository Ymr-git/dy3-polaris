"""L0 策略评估引擎 (G2).

提供声明式策略的核心评估能力，融合四大开源引擎精华：
- **Cedar 式 deny-override**: 任何匹配的 deny 策略都否决允许（安全侧兜底）
- **Casbin 式 LRU 缓存**: 决策缓存 + TTL，减少重复匹配开销
- **OPA 式钩子管道**: Pre/Post 评估钩子，支持外部逻辑注入
- **Kyverno 式三阶段**: match → evaluate → mutate 管道处理

评估决策算法：
1. 作用域过滤：global + 请求作用域的策略（Kyverno match 阶段）
2. 优先级降序遍历，收集所有匹配策略（validate 阶段）
3. deny-override：任意匹配 deny → 最终决策为 deny（Cedar 原则）
4. 否则取最高优先级匹配策略的动作作为决策
5. 附加 transform/escalation 规范 + 违规记录 + 事件（mutate 阶段）
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Protocol, runtime_checkable

from .exceptions import PolicyConflictError
from .models import (
    EvalRequest,
    EvalResult,
    GovernanceEvent,
    GovernanceEventType,
    GovernancePolicy,
    PolicyAction,
    PolicyCondition,
    PolicyScope,
    SeverityLevel,
    TransformSpec,
    ViolationRecord,
)
from .policy_store import PolicyStore

logger = logging.getLogger(__name__)


# ============================================================
# 钩子协议（OPA 插件接口 + Kyverno Webhook 启发）
# ============================================================


@runtime_checkable
class PreEvalHook(Protocol):
    """前置评估钩子协议.

    在策略匹配之前执行。返回非 None 的 EvalResult 可短路后续评估。
    典型用途：白名单放行、认证前置检查、请求上下文增强。
    """

    def __call__(self, request: EvalRequest) -> EvalResult | None:  # pragma: no cover
        ...


@runtime_checkable
class PostEvalHook(Protocol):
    """后置评估钩子协议.

    在策略决策之后执行，可修改评估结果。
    典型用途：审计日志、结果增强、告警触发。
    """

    def __call__(self, request: EvalRequest, result: EvalResult) -> EvalResult:  # pragma: no cover
        ...


# ============================================================
# 评估度量（L6 ComputeMetrics 模式）
# ============================================================


@dataclass
class _Counter:
    """线程安全计数器."""

    _value: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    @property
    def value(self) -> int:
        return self._value


@dataclass
class _LatencyTracker:
    """延迟样本收集."""

    max_samples: int = 200
    _samples: list[float] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def record(self, ms: float) -> None:
        with self._lock:
            self._samples.append(ms)
            if len(self._samples) > self.max_samples:
                self._samples = self._samples[-self.max_samples:]

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def avg(self) -> float:
        with self._lock:
            return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def p99(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            s = sorted(self._samples)
            idx = int(len(s) * 0.99)
            return s[min(idx, len(s) - 1)]


class EvaluatorMetrics:
    """策略评估度量收集器.

    跟踪评估引擎运行时指标，与 L6 ComputeMetrics 风格保持一致。
    """

    def __init__(self) -> None:
        self._started_at = time.time()
        self.evaluations_total = _Counter()
        self.cache_hits = _Counter()
        self.cache_misses = _Counter()
        self.allows = _Counter()
        self.denials = _Counter()
        self.escalations = _Counter()
        self.transforms = _Counter()
        self.logs = _Counter()
        self._latency = _LatencyTracker()
        self.pre_hook_shortcircuits = _Counter()

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits.value + self.cache_misses.value
        return self.cache_hits.value / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self._latency.avg

    def record_eval(self, decision: PolicyAction, latency_ms: float) -> None:
        """记录一次评估结果."""
        self.evaluations_total.inc()
        self._latency.record(latency_ms)
        counter_map = {
            PolicyAction.ALLOW: self.allows,
            PolicyAction.DENY: self.denials,
            PolicyAction.ESCALATE: self.escalations,
            PolicyAction.TRANSFORM: self.transforms,
            PolicyAction.LOG: self.logs,
        }
        counter_map.get(decision, self.allows).inc()

    def export(self) -> dict[str, Any]:
        """导出度量数据."""
        return {
            "uptime_seconds": round(time.time() - self._started_at, 2),
            "evaluations": {
                "total": self.evaluations_total.value,
                "allows": self.allows.value,
                "denials": self.denials.value,
                "escalations": self.escalations.value,
                "transforms": self.transforms.value,
                "logs": self.logs.value,
            },
            "cache": {
                "hits": self.cache_hits.value,
                "misses": self.cache_misses.value,
                "hit_rate": round(self.cache_hit_rate, 4),
            },
            "latency_ms": {
                "avg": round(self._latency.avg, 3),
                "p99": round(self._latency.p99, 3),
                "samples": self._latency.count,
            },
            "pre_hook_shortcircuits": self.pre_hook_shortcircuits.value,
        }


# ============================================================
# LRU 决策缓存（Casbin CachedEnforcer 启发）
# ============================================================


class _DecisionCache:
    """基于 LRU 的决策缓存.

    缓存键为请求核心字段的 SHA256 哈希，支持可配置容量和 TTL。
    线程安全。
    """

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 60.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[EvalResult, float]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _make_key(request: EvalRequest) -> str:
        """生成请求的缓存键."""
        key_data = {
            "actor": request.actor,
            "action": request.action,
            "resource": request.resource,
            "layer": request.layer,
            "domain": request.domain,
            "ctx_hash": hashlib.sha256(
                json.dumps(request.context, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()

    def get(self, request: EvalRequest) -> EvalResult | None:
        """查询缓存."""
        key = self._make_key(request)
        with self._lock:
            if key in self._cache:
                result, ts = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._cache.move_to_end(key)
                    return result
                else:
                    del self._cache[key]
            return None

    def put(self, request: EvalRequest, result: EvalResult) -> None:
        """写入缓存."""
        key = self._make_key(request)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = (result, time.time())
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = (result, time.time())

    def invalidate(self) -> None:
        """清空缓存."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ============================================================
# 策略评估引擎
# ============================================================


class PolicyEvaluator:
    """策略评估引擎.

    G2 核心组件，将声明式策略转化为运行时决策。

    评估决策算法（Cedar deny-override + 优先级排序）：
    1. 作用域过滤：global + 请求作用域的策略
    2. 优先级降序遍历，收集所有匹配策略
    3. deny-override：任意匹配 deny → 最终决策为 deny
    4. 否则取最高优先级匹配策略的动作作为决策
    5. 附加对应的 transform/escalation 规范
    """

    def __init__(
        self,
        store: PolicyStore,
        *,
        default_action: PolicyAction = PolicyAction.ALLOW,
        cache_max_size: int = 1024,
        cache_ttl_seconds: float = 60.0,
        dry_run: bool = False,
    ) -> None:
        self._store = store
        self._default_action = default_action
        self._dry_run = dry_run
        self._cache = _DecisionCache(max_size=cache_max_size, ttl_seconds=cache_ttl_seconds)
        self._metrics = EvaluatorMetrics()
        self._pre_hooks: list[PreEvalHook] = []
        self._post_hooks: list[PostEvalHook] = []
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 核心评估
    # --------------------------------------------------------

    def evaluate(
        self,
        request: EvalRequest,
        *,
        dry_run: bool | None = None,
        use_cache: bool = True,
    ) -> EvalResult:
        """评估单个请求."""
        is_dry = dry_run if dry_run is not None else self._dry_run
        t0 = time.monotonic()

        # 阶段 0: Pre-evaluate 钩子
        for hook in self._pre_hooks:
            hook_result = hook(request)
            if hook_result is not None:
                self._metrics.pre_hook_shortcircuits.inc()
                self._record_metrics(hook_result.decision, t0, cached=False)
                return hook_result

        # 阶段 1: 缓存查询
        if use_cache:
            cached = self._cache.get(request)
            if cached is not None:
                self._metrics.cache_hits.inc()
                self._record_metrics(cached.decision, t0, cached=True)
                return cached
            self._metrics.cache_misses.inc()

        # 阶段 2: 策略评估
        result = self._evaluate_policies(request, dry_run=is_dry)

        # 阶段 3: Post-evaluate 钩子
        for hook in self._post_hooks:
            result = hook(request, result)

        # 阶段 4: 缓存写入
        if use_cache:
            self._cache.put(request, result)

        # 阶段 5: 度量记录
        self._record_metrics(result.decision, t0, cached=False)

        return result

    def evaluate_batch(
        self,
        requests: list[EvalRequest],
        *,
        dry_run: bool | None = None,
        use_cache: bool = True,
    ) -> list[EvalResult]:
        """批量评估多个请求."""
        return [
            self.evaluate(r, dry_run=dry_run, use_cache=use_cache)
            for r in requests
        ]

    # --------------------------------------------------------
    # 内部评估逻辑
    # --------------------------------------------------------

    def _evaluate_policies(
        self,
        request: EvalRequest,
        *,
        dry_run: bool,
    ) -> EvalResult:
        """核心策略评估（Kyverno 三阶段启发）."""
        # Match: 作用域过滤
        scope = self._infer_scope(request)
        candidates = self._store.get_policies_for_scope(scope)

        # Validate: 收集所有匹配策略
        matches: list[GovernancePolicy] = []
        for policy in candidates:
            if PolicyStore.match_condition(policy.condition, request):
                matches.append(policy)

        if not matches:
            self._store.record_eval()
            self._emit_eval_event(request, None, self._default_action)
            return EvalResult(decision=self._default_action)

        # 按优先级降序
        matches.sort(key=lambda p: p.priority, reverse=True)

        # Deny-override: 扫描是否存在 deny（Cedar 原则）
        deny_policy: GovernancePolicy | None = None
        for policy in matches:
            if policy.action == PolicyAction.DENY:
                deny_policy = policy
                break

        if deny_policy is not None:
            result = self._build_result(deny_policy, request)
            if not dry_run:
                self._create_violation(deny_policy, request, result)
            self._store.record_eval(denied=True)
            self._emit_eval_event(request, deny_policy, PolicyAction.DENY)
            return result

        # 无 deny → 最高优先级匹配策略的动作决定结果
        top_policy = matches[0]
        result = self._build_result(top_policy, request)

        # Mutate: 副作用
        if not dry_run:
            if top_policy.action == PolicyAction.ESCALATE:
                self._create_violation(top_policy, request, result)
                self._store.record_eval(escalated=True)
            else:
                self._store.record_eval()

        self._emit_eval_event(request, top_policy, top_policy.action)
        return result

    def _build_result(
        self,
        policy: GovernancePolicy,
        request: EvalRequest,
    ) -> EvalResult:
        """从匹配策略构建评估结果."""
        reason = (
            f"策略 [{policy.name}] (id={policy.policy_id}, "
            f"priority={policy.priority}) 命中"
        )
        return EvalResult(
            decision=policy.action,
            matched_policy_id=policy.policy_id,
            matched_policy_name=policy.name,
            reason=reason,
            transform=policy.transform if policy.action == PolicyAction.TRANSFORM else None,
            escalation=policy.escalation if policy.action == PolicyAction.ESCALATE else None,
        )

    def _create_violation(
        self,
        policy: GovernancePolicy,
        request: EvalRequest,
        result: EvalResult,
    ) -> ViolationRecord:
        """创建违规记录并存储."""
        severity = (
            SeverityLevel.HIGH
            if policy.action == PolicyAction.DENY
            else SeverityLevel.MEDIUM
        )
        violation = ViolationRecord(
            policy_id=policy.policy_id,
            policy_name=policy.name,
            severity=severity,
            actor=request.actor,
            action=request.action,
            resource=request.resource,
            layer=request.layer,
            detail=f"策略 [{policy.name}] 触发 {policy.action.value}",
            eval_request=request.model_dump(mode="json"),
            eval_result=result.model_dump(mode="json"),
        )
        self._store.add_violation(violation)
        return violation

    def _emit_eval_event(
        self,
        request: EvalRequest,
        policy: GovernancePolicy | None,
        decision: PolicyAction,
    ) -> None:
        """记录策略评估事件."""
        self._store._append_event(GovernanceEvent(
            event_type=GovernanceEventType.POLICY_EVALUATED,
            detail=f"{decision.value}: actor={request.actor} action={request.action}",
            references=[policy.policy_id] if policy else [],
            payload={
                "actor": request.actor,
                "action": request.action,
                "resource": request.resource,
                "decision": decision.value,
            },
        ))

    @staticmethod
    def _infer_scope(request: EvalRequest) -> PolicyScope:
        """从请求推断作用域.

        优先级：context 显式标记 > resource 前缀 > domain > layer > actor 前缀 > GLOBAL。
        """
        # context 中的显式标记优先
        if request.context.get("tool_name"):
            return PolicyScope.TOOL
        if request.context.get("agent_id"):
            return PolicyScope.AGENT
        # resource 前缀
        if request.resource.startswith("tool:"):
            return PolicyScope.TOOL
        # 基本字段
        if request.domain:
            return PolicyScope.DOMAIN
        if request.layer:
            return PolicyScope.LAYER
        # actor 前缀（最低优先级）
        if request.actor.startswith("agent-"):
            return PolicyScope.AGENT
        return PolicyScope.GLOBAL

    def _record_metrics(
        self,
        decision: PolicyAction,
        t0: float,
        *,
        cached: bool,
    ) -> None:
        """记录度量."""
        latency_ms = (time.monotonic() - t0) * 1000
        if not cached:
            self._metrics.record_eval(decision, latency_ms)

    # --------------------------------------------------------
    # 转换应用（Kyverno mutate 阶段启发）
    # --------------------------------------------------------

    @staticmethod
    def apply_transform(
        data: dict[str, Any],
        transform: TransformSpec,
    ) -> dict[str, Any]:
        """应用转换规范到数据字典."""
        result = dict(data)

        for f in transform.strip_fields:
            result.pop(f, None)

        for f in transform.mask_fields:
            if f in result:
                val = result[f]
                if isinstance(val, str) and len(val) > 4:
                    result[f] = val[:2] + "***" + val[-2:]
                else:
                    result[f] = "***"

        result.update(transform.inject_fields)

        for f, max_v in transform.max_value.items():
            if f in result and isinstance(result[f], (int, float)):
                result[f] = min(result[f], max_v)

        return result

    # --------------------------------------------------------
    # 冲突检测
    # --------------------------------------------------------

    def detect_conflicts(self) -> list[dict[str, Any]]:
        """检测策略冲突."""
        policies = self._store.list_policies(enabled_only=True)
        conflicts: list[dict[str, Any]] = []

        by_priority: dict[int, list[GovernancePolicy]] = {}
        for p in policies:
            by_priority.setdefault(p.priority, []).append(p)

        for priority, group in by_priority.items():
            if len(group) < 2:
                continue
            for p1, p2 in combinations(group, 2):
                if p1.scope != p2.scope:
                    continue
                actions = {p1.action, p2.action}
                if PolicyAction.DENY in actions and PolicyAction.ALLOW in actions:
                    conflicts.append({
                        "type": "allow_deny_conflict",
                        "priority": priority,
                        "scope": p1.scope.value,
                        "policies": [
                            {"id": p1.policy_id, "name": p1.name, "action": p1.action.value},
                            {"id": p2.policy_id, "name": p2.name, "action": p2.action.value},
                        ],
                        "resolution": "deny-override: deny 策略将优先生效",
                    })

        return conflicts

    # --------------------------------------------------------
    # 钩子管理
    # --------------------------------------------------------

    def add_pre_hook(self, hook: PreEvalHook) -> None:
        self._pre_hooks.append(hook)

    def remove_pre_hook(self, hook: PreEvalHook) -> None:
        self._pre_hooks.remove(hook)

    def add_post_hook(self, hook: PostEvalHook) -> None:
        self._post_hooks.append(hook)

    def remove_post_hook(self, hook: PostEvalHook) -> None:
        self._post_hooks.remove(hook)

    @property
    def pre_hooks(self) -> list[PreEvalHook]:
        return list(self._pre_hooks)

    @property
    def post_hooks(self) -> list[PostEvalHook]:
        return list(self._post_hooks)

    # --------------------------------------------------------
    # 缓存管理
    # --------------------------------------------------------

    def invalidate_cache(self) -> None:
        self._cache.invalidate()

    @property
    def cache_size(self) -> int:
        return self._cache.size

    # --------------------------------------------------------
    # 度量
    # --------------------------------------------------------

    @property
    def metrics(self) -> EvaluatorMetrics:
        return self._metrics

    def export_metrics(self) -> dict[str, Any]:
        return self._metrics.export()

    # --------------------------------------------------------
    # 快速合规检查
    # --------------------------------------------------------

    def quick_compliance_check(
        self,
        scope: PolicyScope = PolicyScope.GLOBAL,
    ) -> dict[str, Any]:
        """快速合规状态检查."""
        stats = self._store.get_stats()
        return {
            "scope": scope.value,
            "policies_total": stats["policies"]["total"],
            "policies_enabled": stats["policies"]["enabled"],
            "violations_total": stats["violations"]["total"],
            "violations_active": stats["violations"]["active"],
            "evaluations_total": stats["evaluations"]["total"],
            "evaluations_denied": stats["evaluations"]["denied"],
            "evaluations_escalated": stats["evaluations"]["escalated"],
            "evaluator_metrics": self.export_metrics(),
        }
