"""G5 治理审计引擎 — 不可变审计日志与查询.

融合 OPA Decision Log + CloudTrail 不可变存储 + BeyondCorp UEBA 基线检测：
- 每条治理决策生成唯一 decision_id，支持跨 Agent 追踪
- 审计日志不可变存储，容量控制 FIFO 淘汰
- UEBA 风格行为基线检测，发现异常模式
- 合规框架自动映射（SOC2 / NIST AI RMF）
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
_logger = logger


# ============================================================
# 审计日志模型
# ============================================================


class DecisionLog(BaseModel):
    """治理决策审计日志 (OPA Decision Log 启发).

    记录每次治理决策的完整上下文，支持端到端回溯。

    Attributes:
        decision_id: 全局唯一决策 ID
        trace_id: 分布式追踪 ID（关联跨 Agent 调用链）
        timestamp: 决策时间戳
        actor: 执行者（Agent ID 或用户 ID）
        action: 操作类型（tool_call, policy_eval, mode_switch 等）
        layer: 所属架构层（L0-L7, CC1-CC3）
        input_context: 决策输入上下文
        output_result: 决策输出结果
        policy_version: 策略版本或配置哈希
        outcome: 决策结果（allow/deny/escalate/transform/error）
        latency_ms: 决策延迟（毫秒）
        agent_id: 关联的 Agent ID
        session_id: 会话 ID
        metadata: 额外元数据
    """

    decision_id: str = Field(
        default_factory=lambda: f"dec-{int(time.time()*1000)}-{threading.current_thread().ident}",
    )
    trace_id: str = Field(default="", description="分布式追踪 ID")
    timestamp: float = Field(default_factory=time.time)
    actor: str = Field(default="", description="执行者")
    action: str = Field(default="", description="操作类型")
    layer: str = Field(default="", description="所属层")
    input_context: dict[str, Any] = Field(default_factory=dict)
    output_result: dict[str, Any] = Field(default_factory=dict)
    policy_version: str = Field(default="", description="策略版本")
    outcome: str = Field(default="", description="决策结果")
    latency_ms: float = Field(default=0.0, ge=0.0)
    agent_id: str = Field(default="", description="Agent ID")
    session_id: str = Field(default="", description="会话 ID")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditQueryFilter(BaseModel):
    """审计日志查询过滤器."""

    actor: str | None = None
    action: str | None = None
    layer: str | None = None
    outcome: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    trace_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    limit: int = Field(default=100, ge=1, le=10000)


class BehaviorBaseline(BaseModel):
    """行为基线 (BeyondCorp UEBA 启发).

    Agent 或用户的行为统计基线，用于异常检测。
    """

    entity_id: str = Field(description="实体 ID（Agent 或用户）")
    entity_type: str = Field(default="agent", description="实体类型")
    action_counts: dict[str, int] = Field(default_factory=dict)
    outcome_counts: dict[str, int] = Field(default_factory=dict)
    avg_latency_ms: float = Field(default=0.0)
    total_decisions: int = Field(default=0)
    window_start: float = Field(default_factory=time.time)
    window_end: float = Field(default_factory=time.time)


class AnomalyAlert(BaseModel):
    """异常告警."""

    alert_id: str = Field(
        default_factory=lambda: f"alrt-{int(time.time()*1000)}",
    )
    entity_id: str = Field(description="异常实体 ID")
    alert_type: str = Field(description="异常类型")
    severity: str = Field(default="medium")
    description: str = Field(default="")
    expected_value: float | None = None
    actual_value: float | None = None
    related_decisions: list[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


# ============================================================
# 审计引擎
# ============================================================


class AuditEngine:
    """治理审计引擎.

    提供不可变审计日志存储、查询、聚合与异常检测。

    使用示例::

        engine = AuditEngine()
        log = engine.record(
            actor="agent-tutor",
            action="policy_eval",
            layer="L0",
            outcome="allow",
            input_context={"tool": "grade"},
            output_result={"decision": "allow"},
        )

        # 查询
        logs = engine.query(actor="agent-tutor", limit=10)

        # 异常检测
        alerts = engine.detect_anomalies("agent-tutor")
    """

    def __init__(self, max_logs: int = 10000, persist_path: str | None = None) -> None:
        self._max_logs = max_logs
        self._logs: list[DecisionLog] = []
        self._index_by_decision: dict[str, DecisionLog] = {}
        self._index_by_trace: dict[str, list[DecisionLog]] = defaultdict(list)
        self._index_by_agent: dict[str, list[DecisionLog]] = defaultdict(list)
        self._baselines: dict[str, BehaviorBaseline] = {}
        self._alerts: list[AnomalyAlert] = []
        self._lock = threading.RLock()
        self._persist_path: str | None = persist_path

        # 统计
        self._total_recorded = 0
        self._total_anomalies = 0

        if self._persist_path:
            self._load_persisted()

    def _load_persisted(self) -> None:
        """启动加载上次会话的审计日志 (JSON Lines, 根治重启丢失轨迹)."""
        try:
            path = Path(self._persist_path)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                return
            loaded = 0
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        log = DecisionLog.model_validate(data)
                        self._logs.append(log)
                        self._index_by_decision[log.decision_id] = log
                        if log.trace_id:
                            self._index_by_trace[log.trace_id].append(log)
                        if log.agent_id:
                            self._index_by_agent[log.agent_id].append(log)
                        loaded += 1
                    except Exception:  # noqa: BLE001 - 跳过损坏行
                        continue
            self._logs = self._logs[-self._max_logs :]
            self._total_recorded = len(self._logs)
            _logger.info("审计日志加载 %d 条 (persist=%s)", loaded, path)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("审计日志加载失败: %s", exc)

    def _persist(self, log: DecisionLog) -> None:
        """追加写一条审计日志 (JSON Lines)."""
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(log.model_dump(mode="json"), ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            _logger.warning("审计日志写盘失败: %s", exc)

    def record(
        self,
        *,
        actor: str = "",
        action: str = "",
        layer: str = "",
        outcome: str = "",
        input_context: dict[str, Any] | None = None,
        output_result: dict[str, Any] | None = None,
        policy_version: str = "",
        latency_ms: float = 0.0,
        agent_id: str = "",
        session_id: str = "",
        trace_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DecisionLog:
        """记录治理决策日志."""
        log = DecisionLog(
            trace_id=trace_id,
            actor=actor,
            action=action,
            layer=layer,
            input_context=input_context or {},
            output_result=output_result or {},
            policy_version=policy_version,
            outcome=outcome,
            latency_ms=latency_ms,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._logs.append(log)
            self._index_by_decision[log.decision_id] = log
            if log.trace_id:
                self._index_by_trace[log.trace_id].append(log)
            if log.agent_id:
                self._index_by_agent[log.agent_id].append(log)
            self._total_recorded += 1

            # FIFO 淘汰
            while len(self._logs) > self._max_logs:
                removed = self._logs.pop(0)
                self._index_by_decision.pop(removed.decision_id, None)

        # 持久化 (JSON Lines 追加, 失败不影响内存)
        self._persist(log)

        return log

    def get(self, decision_id: str) -> DecisionLog | None:
        """通过 decision_id 获取单条日志."""
        with self._lock:
            return self._index_by_decision.get(decision_id)

    def query(
        self,
        *,
        actor: str | None = None,
        action: str | None = None,
        layer: str | None = None,
        outcome: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
    ) -> list[DecisionLog]:
        """多条件查询审计日志."""
        with self._lock:
            # 优先使用 agent_id 索引加速
            if agent_id:
                candidates = list(self._index_by_agent.get(agent_id, []))
            elif trace_id:
                candidates = list(self._index_by_trace.get(trace_id, []))
            else:
                candidates = list(self._logs)

        results: list[DecisionLog] = []
        for log in candidates:
            if actor is not None and log.actor != actor:
                continue
            if action is not None and log.action != action:
                continue
            if layer is not None and log.layer != layer:
                continue
            if outcome is not None and log.outcome != outcome:
                continue
            if session_id is not None and log.session_id != session_id:
                continue
            if trace_id is not None and log.trace_id != trace_id:
                continue
            if start_time is not None and log.timestamp < start_time:
                continue
            if end_time is not None and log.timestamp > end_time:
                continue
            results.append(log)

        # 按时间倒序
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]

    def get_trace(self, trace_id: str) -> list[DecisionLog]:
        """获取完整追踪链（按时间排序）."""
        with self._lock:
            logs = list(self._index_by_trace.get(trace_id, []))
        logs.sort(key=lambda x: x.timestamp)
        return logs

    def aggregate_by_action(self, agent_id: str | None = None) -> dict[str, int]:
        """按 action 聚合计数."""
        with self._lock:
            if agent_id:
                logs = self._index_by_agent.get(agent_id, [])
            else:
                logs = self._logs

        counts: dict[str, int] = {}
        for log in logs:
            key = log.action or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def aggregate_by_outcome(self, agent_id: str | None = None) -> dict[str, int]:
        """按 outcome 聚合计数."""
        with self._lock:
            if agent_id:
                logs = self._index_by_agent.get(agent_id, [])
            else:
                logs = self._logs

        counts: dict[str, int] = {}
        for log in logs:
            key = log.outcome or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def latency_stats(self, agent_id: str | None = None) -> dict[str, float]:
        """延迟统计."""
        with self._lock:
            if agent_id:
                logs = self._index_by_agent.get(agent_id, [])
            else:
                logs = self._logs

        latencies = [log.latency_ms for log in logs if log.latency_ms > 0]
        if not latencies:
            return {"count": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0, "p99": 0.0}

        latencies.sort()
        n = len(latencies)
        p99_idx = int(n * 0.99) - 1
        p99_idx = max(0, min(p99_idx, n - 1))
        return {
            "count": float(n),
            "avg": round(sum(latencies) / n, 3),
            "min": round(latencies[0], 3),
            "max": round(latencies[-1], 3),
            "p99": round(latencies[p99_idx], 3),
        }

    # ==========================================================
    # UEBA 基线检测 (BeyondCorp 启发)
    # ==========================================================

    def build_baseline(
        self,
        entity_id: str,
        entity_type: str = "agent",
        window_seconds: float = 3600.0,
    ) -> BehaviorBaseline:
        """为实体构建行为基线."""
        end_time = time.time()
        start_time = end_time - window_seconds

        with self._lock:
            if entity_type == "agent":
                logs = [
                    log for log in self._index_by_agent.get(entity_id, [])
                    if start_time <= log.timestamp <= end_time
                ]
            else:
                logs = [
                    log for log in self._logs
                    if log.actor == entity_id and start_time <= log.timestamp <= end_time
                ]

        action_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {}
        latencies: list[float] = []

        for log in logs:
            action_counts[log.action] = action_counts.get(log.action, 0) + 1
            outcome_counts[log.outcome] = outcome_counts.get(log.outcome, 0) + 1
            if log.latency_ms > 0:
                latencies.append(log.latency_ms)

        baseline = BehaviorBaseline(
            entity_id=entity_id,
            entity_type=entity_type,
            action_counts=action_counts,
            outcome_counts=outcome_counts,
            avg_latency_ms=round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            total_decisions=len(logs),
            window_start=start_time,
            window_end=end_time,
        )

        with self._lock:
            self._baselines[entity_id] = baseline

        return baseline

    def detect_anomalies(
        self,
        entity_id: str,
        recent_window_seconds: float = 300.0,
    ) -> list[AnomalyAlert]:
        """检测实体行为异常.

        检测规则：
        1. 延迟异常：近期平均延迟 > 基线 3 倍
        2. 失败率异常：近期失败率 > 基线 2 倍
        3. 行为模式异常：出现基线中不存在的 action
        """
        with self._lock:
            baseline = self._baselines.get(entity_id)

        if baseline is None or baseline.total_decisions == 0:
            return []

        end_time = time.time()
        start_time = end_time - recent_window_seconds

        with self._lock:
            recent_logs = [
                log for log in self._index_by_agent.get(entity_id, [])
                if start_time <= log.timestamp <= end_time
            ]

        if len(recent_logs) < 3:
            return []

        alerts: list[AnomalyAlert] = []

        # 延迟异常检测
        recent_latencies = [log.latency_ms for log in recent_logs if log.latency_ms > 0]
        if recent_latencies and baseline.avg_latency_ms > 0:
            recent_avg = sum(recent_latencies) / len(recent_latencies)
            if recent_avg > baseline.avg_latency_ms * 3:
                alerts.append(AnomalyAlert(
                    entity_id=entity_id,
                    alert_type="latency_spike",
                    severity="high",
                    description=f"延迟异常: 近期平均 {recent_avg:.1f}ms > 基线 {baseline.avg_latency_ms:.1f}ms 的 3 倍",
                    expected_value=baseline.avg_latency_ms,
                    actual_value=recent_avg,
                ))

        # 失败率异常检测
        recent_failures = sum(1 for log in recent_logs if log.outcome in ("deny", "error", "reject"))
        recent_fail_rate = recent_failures / len(recent_logs)

        baseline_total = baseline.total_decisions
        baseline_failures = baseline.outcome_counts.get("deny", 0) + baseline.outcome_counts.get("error", 0)
        baseline_fail_rate = baseline_failures / baseline_total if baseline_total > 0 else 0

        if baseline_fail_rate > 0 and recent_fail_rate > baseline_fail_rate * 2:
            alerts.append(AnomalyAlert(
                entity_id=entity_id,
                alert_type="failure_rate_spike",
                severity="critical",
                description=f"失败率异常: 近期 {recent_fail_rate:.1%} > 基线 {baseline_fail_rate:.1%} 的 2 倍",
                expected_value=baseline_fail_rate,
                actual_value=recent_fail_rate,
            ))
        elif baseline_fail_rate == 0 and recent_fail_rate > 0.3:
            alerts.append(AnomalyAlert(
                entity_id=entity_id,
                alert_type="failure_rate_spike",
                severity="high",
                description=f"失败率异常: 基线零失败，近期 {recent_fail_rate:.1%}",
                expected_value=0.0,
                actual_value=recent_fail_rate,
            ))

        # 行为模式异常（新 action 类型）
        recent_actions = {log.action for log in recent_logs}
        baseline_actions = set(baseline.action_counts.keys())
        new_actions = recent_actions - baseline_actions
        for action in new_actions:
            alerts.append(AnomalyAlert(
                entity_id=entity_id,
                alert_type="new_action_pattern",
                severity="medium",
                description=f"新行为模式: 出现基线中不存在的 action '{action}'",
            ))

        with self._lock:
            self._alerts.extend(alerts)
            self._total_anomalies += len(alerts)

        return alerts

    def get_alerts(
        self,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[AnomalyAlert]:
        """获取异常告警."""
        with self._lock:
            alerts = list(self._alerts)
        if entity_id:
            alerts = [a for a in alerts if a.entity_id == entity_id]
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]

    # ==========================================================
    # 统计与导出
    # ==========================================================

    def get_stats(self) -> dict[str, Any]:
        """获取审计引擎统计."""
        with self._lock:
            outcome_counts: dict[str, int] = {}
            for log in self._logs:
                key = log.outcome or "unknown"
                outcome_counts[key] = outcome_counts.get(key, 0) + 1

            return {
                "total_recorded": self._total_recorded,
                "current_logs": len(self._logs),
                "max_logs": self._max_logs,
                "unique_agents": len(self._index_by_agent),
                "unique_traces": len(self._index_by_trace),
                "baselines": len(self._baselines),
                "total_anomalies": self._total_anomalies,
                "outcome_distribution": outcome_counts,
            }

    def export_summary(self) -> dict[str, Any]:
        """导出审计摘要."""
        with self._lock:
            return {
                "log_count": len(self._logs),
                "time_range": {
                    "start": min((log.timestamp for log in self._logs), default=None),
                    "end": max((log.timestamp for log in self._logs), default=None),
                } if self._logs else None,
                "action_distribution": self.aggregate_by_action(),
                "outcome_distribution": self.aggregate_by_outcome(),
                "latency_stats": self.latency_stats(),
                "alert_count": len(self._alerts),
            }

    def clear(self) -> None:
        """清空所有数据."""
        with self._lock:
            self._logs.clear()
            self._index_by_decision.clear()
            self._index_by_trace.clear()
            self._index_by_agent.clear()
            self._baselines.clear()
            self._alerts.clear()
            self._total_recorded = 0
            self._total_anomalies = 0
