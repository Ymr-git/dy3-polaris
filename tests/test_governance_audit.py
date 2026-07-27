"""G5 治理审计与度量层完整测试.

覆盖范围:
1. DecisionLog 模型 — 创建、默认值、字段约束
2. BehaviorBaseline 模型 — 创建、字段
3. AnomalyAlert 模型 — 创建、字段
4. AuditEngine 记录与查询 — record, get, query 多条件, get_trace, 时间范围, limit
5. AuditEngine 聚合 — aggregate_by_action, aggregate_by_outcome, latency_stats
6. AuditEngine 基线与异常检测 — build_baseline, detect_anomalies
7. AuditEngine 告警与统计 — get_alerts, get_stats, export_summary, clear
8. AuditEngine FIFO 淘汰 — 超容量淘汰, 索引同步清理
9. MetricType 枚举 — 值和 str 继承
10. MetricDefinition 模型 — 创建、字段
11. SLODefinition 模型 — 创建、默认值
12. MetricsEngine 指标记录 — record, get_values, get_latest, labels 过滤
13. MetricsEngine 聚合 — aggregate avg/sum/min/max/count
14. MetricsEngine SLO — register, get, evaluate
15. MetricsEngine BurnRate 告警 — 触发/不触发/多 SLO
16. MetricsEngine DORA — record_dora_deployment, get_dora_metrics
17. MetricsEngine 统计与清空 — get_stats, clear, FIFO
18. NISTFunction 枚举
19. ComplianceControl 模型
20. ComplianceDomain 模型 — compute_score
21. GovernanceComplianceReport 模型 — compute_overall
22. ComplianceReporter 生成 — generate_from_audit
23. ComplianceReporter NIST 摘要 — generate_nist_summary
24. ComplianceReporter 控制点评估 — _evaluate_control
25. 端到端集成 — 审计+度量+合规报告完整流程
"""

from __future__ import annotations

import logging
import time

logging.disable(logging.CRITICAL)

import pytest
from pydantic import ValidationError

from dy3_polaris.l0.governance.audit_engine import (
    AnomalyAlert,
    AuditEngine,
    AuditQueryFilter,
    BehaviorBaseline,
    DecisionLog,
)
from dy3_polaris.l0.governance.compliance import (
    ComplianceControl,
    ComplianceDomain,
    ComplianceReporter,
    GovernanceComplianceReport,
    NISTFunction,
)
from dy3_polaris.l0.governance.metrics_engine import (
    BurnRateAlert,
    MetricDefinition,
    MetricType,
    MetricValue,
    MetricsEngine,
    SLODefinition,
    SLOSnapshot,
)


# ============================================================
# 1. DecisionLog 模型
# ============================================================


class TestDecisionLog模型:
    """验证 DecisionLog 数据模型的创建与默认值."""

    def test_创建最小日志_所有字段使用默认值(self) -> None:
        """不传参数创建 DecisionLog，全部字段应使用默认值."""
        log = DecisionLog()
        assert log.decision_id != ""
        assert log.trace_id == ""
        assert log.actor == ""
        assert log.action == ""
        assert log.layer == ""
        assert log.input_context == {}
        assert log.output_result == {}
        assert log.policy_version == ""
        assert log.outcome == ""
        assert log.latency_ms == 0.0
        assert log.agent_id == ""
        assert log.session_id == ""
        assert log.metadata == {}

    def test_decision_id_自动生成唯一值(self) -> None:
        """decision_id 应由工厂函数自动生成，连续两次不同."""
        log1 = DecisionLog()
        time.sleep(0.001)  # 确保毫秒级时间戳不同
        log2 = DecisionLog()
        assert log1.decision_id != log2.decision_id

    def test_timestamp_自动生成当前时间(self) -> None:
        """timestamp 默认值应接近当前时间."""
        before = time.time()
        log = DecisionLog()
        after = time.time()
        assert before <= log.timestamp <= after

    def test_完整参数创建_字段正确赋值(self) -> None:
        """传入全部参数时，所有字段应正确赋值."""
        now = time.time()
        log = DecisionLog(
            decision_id="dec-001",
            trace_id="trace-abc",
            timestamp=now,
            actor="agent-tutor",
            action="policy_eval",
            layer="L0",
            input_context={"tool": "grade"},
            output_result={"decision": "allow"},
            policy_version="v2.1",
            outcome="allow",
            latency_ms=15.5,
            agent_id="agent-001",
            session_id="sess-xyz",
            metadata={"env": "test"},
        )
        assert log.decision_id == "dec-001"
        assert log.trace_id == "trace-abc"
        assert log.timestamp == now
        assert log.actor == "agent-tutor"
        assert log.action == "policy_eval"
        assert log.layer == "L0"
        assert log.input_context == {"tool": "grade"}
        assert log.output_result == {"decision": "allow"}
        assert log.policy_version == "v2.1"
        assert log.outcome == "allow"
        assert log.latency_ms == 15.5
        assert log.agent_id == "agent-001"
        assert log.session_id == "sess-xyz"
        assert log.metadata == {"env": "test"}

    def test_latency_ms_不接受负值(self) -> None:
        """latency_ms 字段设置了 ge=0.0，负值应触发 ValidationError."""
        with pytest.raises(ValidationError):
            DecisionLog(latency_ms=-1.0)

    def test_latency_ms_接受零值(self) -> None:
        """latency_ms 为 0.0 应通过校验."""
        log = DecisionLog(latency_ms=0.0)
        assert log.latency_ms == 0.0

    def test_input_context_接受任意字典(self) -> None:
        """input_context 应接受任意结构的字典."""
        log = DecisionLog(
            input_context={"nested": {"deep": [1, 2, 3]}, "key": "value"},
        )
        assert log.input_context["nested"]["deep"] == [1, 2, 3]

    def test_output_result_接受任意字典(self) -> None:
        """output_result 应接受任意结构的字典."""
        log = DecisionLog(
            output_result={"scores": {"math": 95, "english": 88}},
        )
        assert log.output_result["scores"]["math"] == 95

    def test_metadata_接受任意字典(self) -> None:
        """metadata 应接受任意结构的字典."""
        log = DecisionLog(metadata={"custom_tag": "test-run-42"})
        assert log.metadata["custom_tag"] == "test-run-42"

    def test_部分参数创建_其余使用默认值(self) -> None:
        """只传部分参数，未传字段使用默认值."""
        log = DecisionLog(actor="agent-a", action="tool_call")
        assert log.actor == "agent-a"
        assert log.action == "tool_call"
        assert log.trace_id == ""
        assert log.layer == ""
        assert log.outcome == ""
        assert log.agent_id == ""


# ============================================================
# 2. BehaviorBaseline 模型
# ============================================================


class TestBehaviorBaseline模型:
    """验证 BehaviorBaseline 数据模型."""

    def test_最小创建_只有entity_id(self) -> None:
        """只传 entity_id 创建基线."""
        bl = BehaviorBaseline(entity_id="agent-001")
        assert bl.entity_id == "agent-001"
        assert bl.entity_type == "agent"
        assert bl.action_counts == {}
        assert bl.outcome_counts == {}
        assert bl.avg_latency_ms == 0.0
        assert bl.total_decisions == 0

    def test_完整创建_所有字段(self) -> None:
        """传入所有字段创建基线."""
        now = time.time()
        bl = BehaviorBaseline(
            entity_id="agent-001",
            entity_type="user",
            action_counts={"policy_eval": 10, "tool_call": 5},
            outcome_counts={"allow": 12, "deny": 3},
            avg_latency_ms=25.5,
            total_decisions=15,
            window_start=now - 3600,
            window_end=now,
        )
        assert bl.entity_type == "user"
        assert bl.action_counts == {"policy_eval": 10, "tool_call": 5}
        assert bl.outcome_counts == {"allow": 12, "deny": 3}
        assert bl.avg_latency_ms == 25.5
        assert bl.total_decisions == 15
        assert bl.window_end == now

    def test_entity_type_默认值为agent(self) -> None:
        """entity_type 不传时应默认为 agent."""
        bl = BehaviorBaseline(entity_id="x")
        assert bl.entity_type == "agent"

    def test_action_counts_默认空字典(self) -> None:
        """action_counts 默认为空字典."""
        bl = BehaviorBaseline(entity_id="x")
        assert bl.action_counts == {}

    def test_outcome_counts_默认空字典(self) -> None:
        """outcome_counts 默认为空字典."""
        bl = BehaviorBaseline(entity_id="x")
        assert bl.outcome_counts == {}

    def test_avg_latency_ms_默认零(self) -> None:
        """avg_latency_ms 默认为 0.0."""
        bl = BehaviorBaseline(entity_id="x")
        assert bl.avg_latency_ms == 0.0

    def test_total_decisions_默认零(self) -> None:
        """total_decisions 默认为 0."""
        bl = BehaviorBaseline(entity_id="x")
        assert bl.total_decisions == 0

    def test_window时间戳_自动生成(self) -> None:
        """window_start 和 window_end 应自动生成当前时间."""
        before = time.time()
        bl = BehaviorBaseline(entity_id="x")
        assert before <= bl.window_start
        assert before <= bl.window_end


# ============================================================
# 3. AnomalyAlert 模型
# ============================================================


class TestAnomalyAlert模型:
    """验证 AnomalyAlert 数据模型."""

    def test_最小创建_只有必要字段(self) -> None:
        """只传 entity_id 和 alert_type 创建告警."""
        alert = AnomalyAlert(entity_id="agent-001", alert_type="latency_spike")
        assert alert.entity_id == "agent-001"
        assert alert.severity == "medium"
        assert alert.description == ""
        assert alert.expected_value is None
        assert alert.actual_value is None
        assert alert.related_decisions == []

    def test_完整创建_所有字段(self) -> None:
        """传入所有字段创建告警."""
        alert = AnomalyAlert(
            alert_id="alrt-001",
            entity_id="agent-001",
            alert_type="latency_spike",
            severity="high",
            description="延迟异常检测",
            expected_value=10.0,
            actual_value=50.0,
            related_decisions=["dec-1", "dec-2"],
            timestamp=time.time(),
        )
        assert alert.alert_id == "alrt-001"
        assert alert.alert_type == "latency_spike"
        assert alert.severity == "high"
        assert alert.description == "延迟异常检测"
        assert alert.expected_value == 10.0
        assert alert.actual_value == 50.0
        assert alert.related_decisions == ["dec-1", "dec-2"]

    def test_alert_id_自动生成(self) -> None:
        """alert_id 应由工厂函数自动生成."""
        alert = AnomalyAlert(entity_id="x", alert_type="test")
        assert alert.alert_id.startswith("alrt-")

    def test_severity_默认medium(self) -> None:
        """severity 默认值应为 medium."""
        alert = AnomalyAlert(entity_id="x", alert_type="test")
        assert alert.severity == "medium"

    def test_timestamp_自动生成(self) -> None:
        """timestamp 应自动生成当前时间."""
        before = time.time()
        alert = AnomalyAlert(entity_id="x", alert_type="test")
        assert before <= alert.timestamp


# ============================================================
# 4. AuditEngine 记录与查询
# ============================================================


class TestAuditEngine记录与查询:
    """验证 AuditEngine 的记录、获取、多条件查询功能."""

    def setup_method(self) -> None:
        """每个测试前创建新的审计引擎."""
        self.engine = AuditEngine()

    def test_record_返回DecisionLog(self) -> None:
        """record 方法应返回 DecisionLog 实例."""
        log = self.engine.record(actor="agent-a", action="tool_call")
        assert isinstance(log, DecisionLog)

    def test_record_生成唯一decision_id(self) -> None:
        """连续记录的日志应有不同的 decision_id."""
        log1 = self.engine.record()
        time.sleep(0.001)  # 确保毫秒级时间戳不同
        log2 = self.engine.record()
        assert log1.decision_id != log2.decision_id

    def test_get_按decision_id查找存在的日志(self) -> None:
        """get 应能通过 decision_id 找到已记录的日志."""
        log = self.engine.record(actor="agent-a")
        found = self.engine.get(log.decision_id)
        assert found is not None
        assert found.actor == "agent-a"

    def test_get_查找不存在的id返回None(self) -> None:
        """get 传入不存在的 decision_id 应返回 None."""
        result = self.engine.get("non-existent-id")
        assert result is None

    def test_record_存储所有字段(self) -> None:
        """record 存储后，通过 get 获取应保留全部字段."""
        log = self.engine.record(
            actor="agent-a",
            action="policy_eval",
            layer="L0",
            outcome="allow",
            input_context={"k": "v"},
            output_result={"r": 1},
            policy_version="v1",
            latency_ms=10.0,
            agent_id="agent-001",
            session_id="sess-1",
            trace_id="trace-1",
            metadata={"tag": "test"},
        )
        found = self.engine.get(log.decision_id)
        assert found is not None
        assert found.actor == "agent-a"
        assert found.action == "policy_eval"
        assert found.layer == "L0"
        assert found.outcome == "allow"
        assert found.input_context == {"k": "v"}
        assert found.output_result == {"r": 1}
        assert found.policy_version == "v1"
        assert found.latency_ms == 10.0
        assert found.agent_id == "agent-001"
        assert found.session_id == "sess-1"
        assert found.trace_id == "trace-1"
        assert found.metadata == {"tag": "test"}

    def test_query_无过滤_返回所有日志按时间倒序(self) -> None:
        """query 不传过滤条件应返回所有日志，按时间倒序排列."""
        for i in range(5):
            self.engine.record(actor=f"agent-{i}")
        logs = self.engine.query()
        assert len(logs) == 5
        timestamps = [l.timestamp for l in logs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_query_按actor过滤(self) -> None:
        """query 按 actor 过滤应只返回匹配的日志."""
        self.engine.record(actor="agent-a", action="act-1")
        self.engine.record(actor="agent-b", action="act-2")
        self.engine.record(actor="agent-a", action="act-3")
        logs = self.engine.query(actor="agent-a")
        assert len(logs) == 2
        assert all(l.actor == "agent-a" for l in logs)

    def test_query_按action过滤(self) -> None:
        """query 按 action 过滤应只返回匹配的日志."""
        self.engine.record(action="policy_eval")
        self.engine.record(action="tool_call")
        self.engine.record(action="policy_eval")
        logs = self.engine.query(action="policy_eval")
        assert len(logs) == 2

    def test_query_按layer过滤(self) -> None:
        """query 按 layer 过滤应只返回匹配的日志."""
        self.engine.record(layer="L0")
        self.engine.record(layer="L1")
        self.engine.record(layer="L0")
        logs = self.engine.query(layer="L0")
        assert len(logs) == 2

    def test_query_按outcome过滤(self) -> None:
        """query 按 outcome 过滤应只返回匹配的日志."""
        self.engine.record(outcome="allow")
        self.engine.record(outcome="deny")
        self.engine.record(outcome="allow")
        logs = self.engine.query(outcome="allow")
        assert len(logs) == 2

    def test_query_按agent_id过滤(self) -> None:
        """query 按 agent_id 过滤应只返回匹配的日志."""
        self.engine.record(agent_id="agent-001")
        self.engine.record(agent_id="agent-002")
        self.engine.record(agent_id="agent-001")
        logs = self.engine.query(agent_id="agent-001")
        assert len(logs) == 2

    def test_query_按session_id过滤(self) -> None:
        """query 按 session_id 过滤应只返回匹配的日志."""
        self.engine.record(session_id="sess-1")
        self.engine.record(session_id="sess-2")
        logs = self.engine.query(session_id="sess-1")
        assert len(logs) == 1

    def test_query_按trace_id过滤(self) -> None:
        """query 按 trace_id 过滤应只返回匹配的日志."""
        self.engine.record(trace_id="trace-a")
        self.engine.record(trace_id="trace-b")
        self.engine.record(trace_id="trace-a")
        logs = self.engine.query(trace_id="trace-a")
        assert len(logs) == 2

    def test_query_多条件组合过滤(self) -> None:
        """query 多条件组合应返回同时满足所有条件的日志."""
        self.engine.record(actor="a", action="x", layer="L0", outcome="allow")
        self.engine.record(actor="a", action="x", layer="L1", outcome="deny")
        self.engine.record(actor="b", action="x", layer="L0", outcome="allow")
        logs = self.engine.query(actor="a", action="x", layer="L0", outcome="allow")
        assert len(logs) == 1
        assert logs[0].actor == "a"

    def test_query_按start_time过滤(self) -> None:
        """query 按 start_time 过滤应只返回时间之后的日志."""
        now = time.time()
        self.engine._logs.clear()
        self.engine._index_by_decision.clear()
        self.engine._total_recorded = 0
        # 手动添加带特定时间戳的日志
        log1 = DecisionLog(actor="a", timestamp=now - 10)
        log2 = DecisionLog(actor="a", timestamp=now + 10)
        self.engine._logs.append(log1)
        self.engine._index_by_decision[log1.decision_id] = log1
        self.engine._total_recorded += 1
        self.engine._logs.append(log2)
        self.engine._index_by_decision[log2.decision_id] = log2
        self.engine._total_recorded += 1
        logs = self.engine.query(start_time=now)
        assert len(logs) == 1
        assert logs[0].timestamp == now + 10

    def test_query_按end_time过滤(self) -> None:
        """query 按 end_time 过滤应只返回时间之前的日志."""
        now = time.time()
        self.engine._logs.clear()
        self.engine._index_by_decision.clear()
        self.engine._total_recorded = 0
        log1 = DecisionLog(actor="a", timestamp=now - 10)
        log2 = DecisionLog(actor="a", timestamp=now + 10)
        self.engine._logs.append(log1)
        self.engine._index_by_decision[log1.decision_id] = log1
        self.engine._total_recorded += 1
        self.engine._logs.append(log2)
        self.engine._index_by_decision[log2.decision_id] = log2
        self.engine._total_recorded += 1
        logs = self.engine.query(end_time=now)
        assert len(logs) == 1
        assert logs[0].timestamp == now - 10

    def test_query_limit限制返回数量(self) -> None:
        """query 的 limit 参数应限制返回条数."""
        for _ in range(10):
            self.engine.record(actor="a")
        logs = self.engine.query(limit=3)
        assert len(logs) == 3

    def test_get_trace_按时间正序返回(self) -> None:
        """get_trace 应返回指定 trace_id 的所有日志，按时间正序."""
        trace_id = "trace-123"
        log1 = self.engine.record(trace_id=trace_id, actor="a1")
        log2 = self.engine.record(trace_id=trace_id, actor="a2")
        log3 = self.engine.record(trace_id=trace_id, actor="a3")
        trace = self.engine.get_trace(trace_id)
        assert len(trace) == 3
        timestamps = [l.timestamp for l in trace]
        assert timestamps == sorted(timestamps)

    def test_get_trace_不存在的trace_id返回空列表(self) -> None:
        """get_trace 传入不存在的 trace_id 应返回空列表."""
        trace = self.engine.get_trace("non-existent")
        assert trace == []

    def test_get_trace_混合trace_id互不干扰(self) -> None:
        """不同 trace_id 的日志应互不影响."""
        self.engine.record(trace_id="trace-a", actor="a1")
        self.engine.record(trace_id="trace-b", actor="b1")
        self.engine.record(trace_id="trace-a", actor="a2")
        trace_a = self.engine.get_trace("trace-a")
        trace_b = self.engine.get_trace("trace-b")
        assert len(trace_a) == 2
        assert len(trace_b) == 1


# ============================================================
# 5. AuditEngine 聚合
# ============================================================


class TestAuditEngine聚合:
    """验证 AuditEngine 的聚合统计功能."""

    def setup_method(self) -> None:
        self.engine = AuditEngine()

    def test_aggregate_by_action_全局聚合(self) -> None:
        """aggregate_by_action 不指定 agent_id 应聚合所有日志."""
        self.engine.record(action="policy_eval")
        self.engine.record(action="tool_call")
        self.engine.record(action="policy_eval")
        result = self.engine.aggregate_by_action()
        assert result["policy_eval"] == 2
        assert result["tool_call"] == 1

    def test_aggregate_by_action_按agent_id(self) -> None:
        """aggregate_by_action 指定 agent_id 应只聚合该 Agent."""
        self.engine.record(agent_id="a1", action="policy_eval")
        self.engine.record(agent_id="a2", action="tool_call")
        self.engine.record(agent_id="a1", action="policy_eval")
        result = self.engine.aggregate_by_action(agent_id="a1")
        assert result["policy_eval"] == 2
        assert "tool_call" not in result

    def test_aggregate_by_action_空日志返回空字典(self) -> None:
        """无日志时 aggregate_by_action 应返回空字典."""
        result = self.engine.aggregate_by_action()
        assert result == {}

    def test_aggregate_by_action_空action记为unknown(self) -> None:
        """action 为空字符串时应记为 unknown."""
        self.engine.record(action="")
        self.engine.record(action="")
        result = self.engine.aggregate_by_action()
        assert result["unknown"] == 2

    def test_aggregate_by_outcome_全局聚合(self) -> None:
        """aggregate_by_outcome 不指定 agent_id 应聚合所有日志."""
        self.engine.record(outcome="allow")
        self.engine.record(outcome="deny")
        self.engine.record(outcome="allow")
        result = self.engine.aggregate_by_outcome()
        assert result["allow"] == 2
        assert result["deny"] == 1

    def test_aggregate_by_outcome_按agent_id(self) -> None:
        """aggregate_by_outcome 指定 agent_id 应只聚合该 Agent."""
        self.engine.record(agent_id="a1", outcome="allow")
        self.engine.record(agent_id="a2", outcome="deny")
        self.engine.record(agent_id="a1", outcome="deny")
        result = self.engine.aggregate_by_outcome(agent_id="a1")
        assert result["allow"] == 1
        assert result["deny"] == 1

    def test_aggregate_by_outcome_空日志返回空字典(self) -> None:
        """无日志时 aggregate_by_outcome 应返回空字典."""
        result = self.engine.aggregate_by_outcome()
        assert result == {}

    def test_latency_stats_基本统计(self) -> None:
        """latency_stats 应返回正确的 count/avg/min/max/p99."""
        for latency in [10.0, 20.0, 30.0, 40.0, 50.0]:
            self.engine.record(latency_ms=latency)
        stats = self.engine.latency_stats()
        assert stats["count"] == 5.0
        assert stats["avg"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0

    def test_latency_stats_忽略零延迟(self) -> None:
        """latency_stats 应忽略 latency_ms 为 0 的日志."""
        self.engine.record(latency_ms=0.0)
        self.engine.record(latency_ms=10.0)
        self.engine.record(latency_ms=20.0)
        stats = self.engine.latency_stats()
        assert stats["count"] == 2.0
        assert stats["avg"] == 15.0

    def test_latency_stats_空日志返回全零(self) -> None:
        """无日志时 latency_stats 应返回全零统计."""
        stats = self.engine.latency_stats()
        assert stats == {"count": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0, "p99": 0.0}

    def test_latency_stats_按agent_id(self) -> None:
        """latency_stats 指定 agent_id 应只统计该 Agent."""
        self.engine.record(agent_id="a1", latency_ms=10.0)
        self.engine.record(agent_id="a2", latency_ms=100.0)
        self.engine.record(agent_id="a1", latency_ms=20.0)
        stats = self.engine.latency_stats(agent_id="a1")
        assert stats["count"] == 2.0
        assert stats["avg"] == 15.0


# ============================================================
# 6. AuditEngine 基线与异常检测
# ============================================================


class TestAuditEngine基线与异常检测:
    """验证 UEBA 行为基线构建与异常检测."""

    def setup_method(self) -> None:
        self.engine = AuditEngine()

    def test_build_baseline_返回BehaviorBaseline(self) -> None:
        """build_baseline 应返回 BehaviorBaseline 实例."""
        self.engine.record(agent_id="agent-001", action="eval", latency_ms=10.0)
        bl = self.engine.build_baseline("agent-001")
        assert isinstance(bl, BehaviorBaseline)
        assert bl.entity_id == "agent-001"

    def test_build_baseline_统计action_counts(self) -> None:
        """build_baseline 应正确统计 action 分布."""
        self.engine.record(agent_id="agent-001", action="eval")
        self.engine.record(agent_id="agent-001", action="eval")
        self.engine.record(agent_id="agent-001", action="call")
        bl = self.engine.build_baseline("agent-001")
        assert bl.action_counts["eval"] == 2
        assert bl.action_counts["call"] == 1

    def test_build_baseline_统计outcome_counts(self) -> None:
        """build_baseline 应正确统计 outcome 分布."""
        self.engine.record(agent_id="agent-001", outcome="allow")
        self.engine.record(agent_id="agent-001", outcome="deny")
        self.engine.record(agent_id="agent-001", outcome="allow")
        bl = self.engine.build_baseline("agent-001")
        assert bl.outcome_counts["allow"] == 2
        assert bl.outcome_counts["deny"] == 1

    def test_build_baseline_计算平均延迟(self) -> None:
        """build_baseline 应正确计算平均延迟."""
        self.engine.record(agent_id="agent-001", latency_ms=10.0)
        self.engine.record(agent_id="agent-001", latency_ms=20.0)
        bl = self.engine.build_baseline("agent-001")
        assert bl.avg_latency_ms == 15.0

    def test_build_baseline_统计total_decisions(self) -> None:
        """build_baseline 应正确统计总决策数."""
        self.engine.record(agent_id="agent-001", action="a")
        self.engine.record(agent_id="agent-001", action="b")
        self.engine.record(agent_id="agent-001", action="c")
        bl = self.engine.build_baseline("agent-001")
        assert bl.total_decisions == 3

    def test_build_baseline_空数据返回零值基线(self) -> None:
        """build_baseline 对无数据的实体应返回零值基线."""
        bl = self.engine.build_baseline("nonexistent")
        assert bl.total_decisions == 0
        assert bl.avg_latency_ms == 0.0
        assert bl.action_counts == {}

    def test_build_baseline_存储到引擎内部(self) -> None:
        """build_baseline 应将基线存储到引擎内部."""
        self.engine.record(agent_id="agent-001", action="eval")
        self.engine.build_baseline("agent-001")
        assert "agent-001" in self.engine._baselines

    def test_detect_anomalies_延迟异常_超过基线3倍(self) -> None:
        """延迟超过基线 3 倍时应触发延迟异常告警."""
        # 构建基线：正常延迟 10ms
        for _ in range(10):
            self.engine.record(agent_id="agent-001", action="eval", latency_ms=10.0)
        self.engine.build_baseline("agent-001")

        # 模拟近期高延迟
        now = time.time()
        for _ in range(5):
            log = DecisionLog(
                agent_id="agent-001",
                action="eval",
                latency_ms=100.0,
                timestamp=now,
            )
            self.engine._logs.append(log)
            self.engine._index_by_decision[log.decision_id] = log
            self.engine._index_by_agent["agent-001"].append(log)

        alerts = self.engine.detect_anomalies("agent-001")
        latency_alerts = [a for a in alerts if a.alert_type == "latency_spike"]
        assert len(latency_alerts) >= 1
        assert latency_alerts[0].severity == "high"

    def test_detect_anomalies_延迟正常_不触发告警(self) -> None:
        """延迟在基线范围内时不应触发延迟告警."""
        for _ in range(10):
            self.engine.record(agent_id="agent-001", action="eval", latency_ms=10.0)
        self.engine.build_baseline("agent-001")

        # 正常延迟
        now = time.time()
        for _ in range(5):
            log = DecisionLog(
                agent_id="agent-001",
                action="eval",
                latency_ms=12.0,
                timestamp=now,
            )
            self.engine._logs.append(log)
            self.engine._index_by_decision[log.decision_id] = log
            self.engine._index_by_agent["agent-001"].append(log)

        alerts = self.engine.detect_anomalies("agent-001")
        latency_alerts = [a for a in alerts if a.alert_type == "latency_spike"]
        assert len(latency_alerts) == 0

    def test_detect_anomalies_失败率异常_超过基线2倍(self) -> None:
        """失败率超过基线 2 倍时应触发失败率异常告警."""
        # 基线：10% 失败率
        for i in range(9):
            self.engine.record(agent_id="agent-001", action="eval", outcome="allow")
        self.engine.record(agent_id="agent-001", action="eval", outcome="deny")
        self.engine.build_baseline("agent-001")

        # 近期 80% 失败率
        now = time.time()
        for _ in range(4):
            log = DecisionLog(
                agent_id="agent-001",
                action="eval",
                outcome="deny",
                timestamp=now,
            )
            self.engine._logs.append(log)
            self.engine._index_by_decision[log.decision_id] = log
            self.engine._index_by_agent["agent-001"].append(log)

        alerts = self.engine.detect_anomalies("agent-001")
        failure_alerts = [a for a in alerts if a.alert_type == "failure_rate_spike"]
        assert len(failure_alerts) >= 1
        assert failure_alerts[0].severity == "critical"

    def test_detect_anomalies_失败率异常_基线零失败时超过30pct(self) -> None:
        """基线零失败率时，近期失败率超过 30% 应触发告警."""
        # 用较旧的时间戳记录基线数据，确保不在近期窗口内
        old_time = time.time() - 600  # 10 分钟前
        for _ in range(10):
            log = DecisionLog(
                agent_id="agent-001", action="eval", outcome="allow",
                timestamp=old_time,
            )
            self.engine._logs.append(log)
            self.engine._index_by_decision[log.decision_id] = log
            self.engine._index_by_agent["agent-001"].append(log)
            self.engine._total_recorded += 1
        self.engine.build_baseline("agent-001")

        # 近期 50% 失败率 — 使用与基线相同的 action 避免触发 new_action_pattern
        now = time.time()
        log1 = DecisionLog(agent_id="agent-001", action="eval", outcome="deny", timestamp=now)
        log2 = DecisionLog(agent_id="agent-001", action="eval", outcome="deny", timestamp=now)
        log3 = DecisionLog(agent_id="agent-001", action="eval", outcome="allow", timestamp=now)
        for log in [log1, log2, log3]:
            self.engine._logs.append(log)
            self.engine._index_by_decision[log.decision_id] = log
            self.engine._index_by_agent["agent-001"].append(log)

        alerts = self.engine.detect_anomalies("agent-001")
        failure_alerts = [a for a in alerts if a.alert_type == "failure_rate_spike"]
        assert len(failure_alerts) >= 1

    def test_detect_anomalies_新行为模式检测(self) -> None:
        """出现基线中不存在的 action 时应触发新行为模式告警."""
        self.engine.record(agent_id="agent-001", action="eval")
        self.engine.record(agent_id="agent-001", action="eval")
        self.engine.build_baseline("agent-001")

        # 近期出现新 action
        now = time.time()
        for _ in range(5):
            log = DecisionLog(
                agent_id="agent-001",
                action="new_action",
                timestamp=now,
            )
            self.engine._logs.append(log)
            self.engine._index_by_decision[log.decision_id] = log
            self.engine._index_by_agent["agent-001"].append(log)

        alerts = self.engine.detect_anomalies("agent-001")
        new_action_alerts = [a for a in alerts if a.alert_type == "new_action_pattern"]
        assert len(new_action_alerts) >= 1
        assert new_action_alerts[0].severity == "medium"

    def test_detect_anomalies_无基线返回空列表(self) -> None:
        """未构建基线时 detect_anomalies 应返回空列表."""
        self.engine.record(agent_id="agent-001", action="eval")
        alerts = self.engine.detect_anomalies("agent-001")
        assert alerts == []

    def test_detect_anomalies_不足3条近期数据返回空列表(self) -> None:
        """近期数据不足 3 条时不应触发异常检测."""
        for _ in range(10):
            self.engine.record(agent_id="agent-001", action="eval", latency_ms=10.0)
        self.engine.build_baseline("agent-001")

        # 只添加 2 条近期记录（不足 3 条阈值），不触发异常
        now = time.time()
        for latency in [12.0, 11.0]:
            log = DecisionLog(
                agent_id="agent-001",
                action="eval",
                latency_ms=latency,
                timestamp=now,
            )
            self.engine._logs.append(log)
            self.engine._index_by_decision[log.decision_id] = log
            self.engine._index_by_agent["agent-001"].append(log)

        alerts = self.engine.detect_anomalies("agent-001")
        assert alerts == []


# ============================================================
# 7. AuditEngine 告警与统计
# ============================================================


class TestAuditEngine告警与统计:
    """验证告警管理、统计导出与清空功能."""

    def setup_method(self) -> None:
        self.engine = AuditEngine()

    def test_get_alerts_无告警返回空列表(self) -> None:
        """无告警时 get_alerts 应返回空列表."""
        alerts = self.engine.get_alerts()
        assert alerts == []

    def test_get_alerts_按entity_id过滤(self) -> None:
        """get_alerts 按 entity_id 过滤应只返回对应实体的告警."""
        alert1 = AnomalyAlert(entity_id="agent-a", alert_type="test")
        alert2 = AnomalyAlert(entity_id="agent-b", alert_type="test")
        self.engine._alerts.extend([alert1, alert2])
        alerts = self.engine.get_alerts(entity_id="agent-a")
        assert len(alerts) == 1
        assert alerts[0].entity_id == "agent-a"

    def test_get_alerts_limit限制返回数量(self) -> None:
        """get_alerts 的 limit 应限制返回条数."""
        for i in range(10):
            self.engine._alerts.append(
                AnomalyAlert(entity_id=f"agent-{i}", alert_type="test"),
            )
        alerts = self.engine.get_alerts(limit=3)
        assert len(alerts) == 3

    def test_get_alerts_按时间倒序(self) -> None:
        """get_alerts 应按时间倒序排列."""
        now = time.time()
        old_alert = AnomalyAlert(entity_id="a", alert_type="test", timestamp=now - 10)
        new_alert = AnomalyAlert(entity_id="a", alert_type="test", timestamp=now + 10)
        self.engine._alerts.extend([old_alert, new_alert])
        alerts = self.engine.get_alerts()
        assert alerts[0].timestamp > alerts[1].timestamp

    def test_get_stats_初始状态(self) -> None:
        """初始状态统计应为零值."""
        stats = self.engine.get_stats()
        assert stats["total_recorded"] == 0
        assert stats["current_logs"] == 0
        assert stats["unique_agents"] == 0
        assert stats["unique_traces"] == 0
        assert stats["baselines"] == 0
        assert stats["total_anomalies"] == 0

    def test_get_stats_记录后统计正确(self) -> None:
        """记录日志后统计应正确更新."""
        self.engine.record(agent_id="a1", outcome="allow", trace_id="t1")
        self.engine.record(agent_id="a1", outcome="deny", trace_id="t1")
        self.engine.record(agent_id="a2", outcome="allow", trace_id="t2")
        stats = self.engine.get_stats()
        assert stats["total_recorded"] == 3
        assert stats["current_logs"] == 3
        assert stats["unique_agents"] == 2
        assert stats["unique_traces"] == 2
        assert stats["outcome_distribution"]["allow"] == 2
        assert stats["outcome_distribution"]["deny"] == 1

    def test_get_stats_包含max_logs(self) -> None:
        """统计应包含 max_logs 配置."""
        engine = AuditEngine(max_logs=500)
        stats = engine.get_stats()
        assert stats["max_logs"] == 500

    def test_export_summary_初始状态(self) -> None:
        """初始状态摘要应为空."""
        summary = self.engine.export_summary()
        assert summary["log_count"] == 0
        assert summary["time_range"] is None
        assert summary["action_distribution"] == {}
        assert summary["outcome_distribution"] == {}
        assert summary["alert_count"] == 0

    def test_export_summary_有数据时包含完整信息(self) -> None:
        """有数据时摘要应包含完整信息."""
        self.engine.record(action="eval", outcome="allow", latency_ms=10.0)
        self.engine.record(action="call", outcome="deny", latency_ms=20.0)
        summary = self.engine.export_summary()
        assert summary["log_count"] == 2
        assert summary["time_range"] is not None
        assert "start" in summary["time_range"]
        assert "end" in summary["time_range"]
        assert summary["action_distribution"]["eval"] == 1
        assert summary["action_distribution"]["call"] == 1
        assert summary["outcome_distribution"]["allow"] == 1
        assert summary["outcome_distribution"]["deny"] == 1
        assert summary["latency_stats"]["count"] == 2.0

    def test_clear_清空所有数据(self) -> None:
        """clear 应清空所有内部状态."""
        self.engine.record(agent_id="a", outcome="allow", trace_id="t1")
        self.engine.build_baseline("a")
        self.engine._alerts.append(AnomalyAlert(entity_id="a", alert_type="test"))
        self.engine.clear()
        stats = self.engine.get_stats()
        assert stats["total_recorded"] == 0
        assert stats["current_logs"] == 0
        assert stats["unique_agents"] == 0
        assert stats["baselines"] == 0
        assert len(self.engine._alerts) == 0


# ============================================================
# 8. AuditEngine FIFO 淘汰
# ============================================================


class TestAuditEngine_FIFO淘汰:
    """验证容量控制与 FIFO 淘汰机制."""

    def test_FIFO_超容量时淘汰最早的日志(self) -> None:
        """当日志数量超过 max_logs 时，应淘汰最早的日志."""
        engine = AuditEngine(max_logs=5)
        logs = []
        for i in range(8):
            time.sleep(0.001)  # 确保毫秒级时间戳唯一
            log = engine.record(actor=f"agent-{i}")
            logs.append(log)

        # 容器中只有 5 条
        assert len(engine._logs) == 5
        # 最老的 3 条已被淘汰
        for i in range(3):
            assert engine.get(logs[i].decision_id) is None
        # 最新的 5 条仍在
        for i in range(3, 8):
            assert engine.get(logs[i].decision_id) is not None

    def test_FIFO_索引同步清理(self) -> None:
        """FIFO 淘汰时 decision_id 索引应同步清理."""
        engine = AuditEngine(max_logs=3)
        log1 = engine.record(trace_id="trace-a")
        time.sleep(0.001)
        log2 = engine.record(trace_id="trace-a")
        time.sleep(0.001)
        log3 = engine.record(trace_id="trace-a")
        time.sleep(0.001)
        log4 = engine.record(trace_id="trace-b")  # 触发 log1 被淘汰

        assert engine.get(log1.decision_id) is None
        assert engine.get(log4.decision_id) is not None

    def test_FIFO_精确等于容量不淘汰(self) -> None:
        """日志数量等于 max_logs 时不应淘汰."""
        engine = AuditEngine(max_logs=5)
        logs = []
        for i in range(5):
            logs.append(engine.record(actor=f"agent-{i}"))
        assert len(engine._logs) == 5
        for log in logs:
            assert engine.get(log.decision_id) is not None

    def test_total_recorded_不受FIFO影响(self) -> None:
        """total_recorded 计数不受 FIFO 淘汰影响."""
        engine = AuditEngine(max_logs=2)
        engine.record()
        engine.record()
        engine.record()  # 淘汰第一条
        assert engine.get_stats()["total_recorded"] == 3


# ============================================================
# 9. MetricType 枚举
# ============================================================


class TestMetricType枚举:
    """验证 MetricType 枚举的定义与类型."""

    def test_COUNTER值(self) -> None:
        """COUNTER 枚举值应为 counter."""
        assert MetricType.COUNTER == "counter"

    def test_GAUGE值(self) -> None:
        """GAUGE 枚举值应为 gauge."""
        assert MetricType.GAUGE == "gauge"

    def test_HISTOGRAM值(self) -> None:
        """HISTOGRAM 枚举值应为 histogram."""
        assert MetricType.HISTOGRAM == "histogram"

    def test_枚举成员数量(self) -> None:
        """MetricType 应有 3 个成员."""
        assert len(MetricType) == 3

    def test_继承str类型(self) -> None:
        """MetricType 应继承 str，可直接作为字符串使用."""
        assert isinstance(MetricType.COUNTER, str)
        assert MetricType.COUNTER.upper() == "COUNTER"

    def test_枚举迭代(self) -> None:
        """应能遍历所有枚举成员."""
        values = [m.value for m in MetricType]
        assert "counter" in values
        assert "gauge" in values
        assert "histogram" in values


# ============================================================
# 10. MetricDefinition 模型
# ============================================================


class TestMetricDefinition模型:
    """验证 MetricDefinition 数据模型."""

    def test_最小创建_只有name(self) -> None:
        """只传 name 创建指标定义."""
        md = MetricDefinition(name="task_success")
        assert md.name == "task_success"
        assert md.description == ""
        assert md.metric_type == MetricType.GAUGE
        assert md.unit == ""
        assert md.labels == []

    def test_完整创建_所有字段(self) -> None:
        """传入所有字段创建指标定义."""
        md = MetricDefinition(
            name="api_latency",
            description="API 请求延迟",
            metric_type=MetricType.HISTOGRAM,
            unit="ms",
            labels=["agent", "endpoint"],
        )
        assert md.name == "api_latency"
        assert md.description == "API 请求延迟"
        assert md.metric_type == MetricType.HISTOGRAM
        assert md.unit == "ms"
        assert md.labels == ["agent", "endpoint"]

    def test_metric_type_默认GAUGE(self) -> None:
        """metric_type 默认应为 GAUGE."""
        md = MetricDefinition(name="x")
        assert md.metric_type == MetricType.GAUGE

    def test_labels_默认空列表(self) -> None:
        """labels 默认应为空列表."""
        md = MetricDefinition(name="x")
        assert md.labels == []


# ============================================================
# 11. SLODefinition 模型
# ============================================================


class TestSLODefinition模型:
    """验证 SLODefinition 数据模型."""

    def test_最小创建_只有name和metric_name(self) -> None:
        """只传 name 和 metric_name 创建 SLO 定义."""
        slo = SLODefinition(name="api_slo", metric_name="api_latency")
        assert slo.name == "api_slo"
        assert slo.metric_name == "api_latency"

    def test_target_percentage_默认99_5(self) -> None:
        """target_percentage 默认值应为 99.5."""
        slo = SLODefinition(name="x", metric_name="y")
        assert slo.target_percentage == 99.5

    def test_evaluation_window_seconds_默认3600(self) -> None:
        """evaluation_window_seconds 默认值应为 3600."""
        slo = SLODefinition(name="x", metric_name="y")
        assert slo.evaluation_window_seconds == 3600.0

    def test_burn_rate_alerts_默认值(self) -> None:
        """burn_rate_alerts 默认值应为 [14.4, 6.0, 3.0]."""
        slo = SLODefinition(name="x", metric_name="y")
        assert slo.burn_rate_alerts == [14.4, 6.0, 3.0]

    def test_description_默认空字符串(self) -> None:
        """description 默认应为空字符串."""
        slo = SLODefinition(name="x", metric_name="y")
        assert slo.description == ""

    def test_target_percentage_接受0到100(self) -> None:
        """target_percentage 应接受 0 到 100 的值."""
        slo = SLODefinition(name="x", metric_name="y", target_percentage=99.9)
        assert slo.target_percentage == 99.9

    def test_target_percentage_不接受负值(self) -> None:
        """target_percentage 为负值应触发 ValidationError."""
        with pytest.raises(ValidationError):
            SLODefinition(name="x", metric_name="y", target_percentage=-1.0)

    def test_target_percentage_不接受超过100(self) -> None:
        """target_percentage 超过 100 应触发 ValidationError."""
        with pytest.raises(ValidationError):
            SLODefinition(name="x", metric_name="y", target_percentage=101.0)

    def test_evaluation_window_seconds_最小值60(self) -> None:
        """evaluation_window_seconds 小于 60 应触发 ValidationError."""
        with pytest.raises(ValidationError):
            SLODefinition(name="x", metric_name="y", evaluation_window_seconds=30.0)

    def test_完整创建_所有字段(self) -> None:
        """传入所有字段创建 SLO 定义."""
        slo = SLODefinition(
            name="agent_task_slo",
            metric_name="task_success",
            target_percentage=99.9,
            evaluation_window_seconds=7200.0,
            burn_rate_alerts=[20.0, 10.0, 5.0],
            description="Agent 任务成功率 SLO",
        )
        assert slo.name == "agent_task_slo"
        assert slo.target_percentage == 99.9
        assert slo.evaluation_window_seconds == 7200.0
        assert slo.burn_rate_alerts == [20.0, 10.0, 5.0]
        assert slo.description == "Agent 任务成功率 SLO"


# ============================================================
# 12. MetricsEngine 指标记录
# ============================================================


class TestMetricsEngine指标记录:
    """验证 MetricsEngine 的指标记录与查询功能."""

    def setup_method(self) -> None:
        self.engine = MetricsEngine()

    def test_record_返回MetricValue(self) -> None:
        """record 应返回 MetricValue 实例."""
        mv = self.engine.record("metric_a", 42.0)
        assert isinstance(mv, MetricValue)

    def test_record_字段正确赋值(self) -> None:
        """record 后 MetricValue 字段应正确赋值."""
        mv = self.engine.record("metric_a", 42.0, labels={"k": "v"}, timestamp=1000.0)
        assert mv.metric_name == "metric_a"
        assert mv.value == 42.0
        assert mv.labels == {"k": "v"}
        assert mv.timestamp == 1000.0

    def test_record_自动生成时间戳(self) -> None:
        """record 不传 timestamp 时应自动生成."""
        before = time.time()
        mv = self.engine.record("metric_a", 1.0)
        after = time.time()
        assert before <= mv.timestamp <= after

    def test_record_labels默认空字典(self) -> None:
        """record 不传 labels 时应为空字典."""
        mv = self.engine.record("metric_a", 1.0)
        assert mv.labels == {}

    def test_get_values_查询存在的指标(self) -> None:
        """get_values 应返回指定指标的值列表."""
        self.engine.record("metric_a", 1.0)
        self.engine.record("metric_a", 2.0)
        values = self.engine.get_values("metric_a")
        assert len(values) == 2

    def test_get_values_不存在的指标返回空列表(self) -> None:
        """get_values 查询不存在的指标应返回空列表."""
        values = self.engine.get_values("nonexistent")
        assert values == []

    def test_get_values_按时间范围过滤(self) -> None:
        """get_values 应按 start_time 和 end_time 过滤."""
        self.engine.record("m", 1.0, timestamp=100.0)
        self.engine.record("m", 2.0, timestamp=200.0)
        self.engine.record("m", 3.0, timestamp=300.0)
        values = self.engine.get_values("m", start_time=150.0, end_time=250.0)
        assert len(values) == 1
        assert values[0].value == 2.0

    def test_get_values_按labels过滤(self) -> None:
        """get_values 应按 labels 精确匹配过滤."""
        self.engine.record("m", 1.0, labels={"agent": "a"})
        self.engine.record("m", 2.0, labels={"agent": "b"})
        self.engine.record("m", 3.0, labels={"agent": "a", "env": "prod"})
        values = self.engine.get_values("m", labels_filter={"agent": "a"})
        # 两条 agent=a 的记录（第三条也匹配 agent=a）
        assert len(values) == 2

    def test_get_values_多labels同时匹配(self) -> None:
        """get_values 多个 label 应同时匹配."""
        self.engine.record("m", 1.0, labels={"agent": "a", "env": "prod"})
        self.engine.record("m", 2.0, labels={"agent": "a", "env": "test"})
        values = self.engine.get_values("m", labels_filter={"agent": "a", "env": "prod"})
        assert len(values) == 1
        assert values[0].value == 1.0

    def test_get_values_按时间倒序(self) -> None:
        """get_values 应按时间倒序排列."""
        self.engine.record("m", 1.0, timestamp=100.0)
        self.engine.record("m", 2.0, timestamp=300.0)
        self.engine.record("m", 3.0, timestamp=200.0)
        values = self.engine.get_values("m")
        timestamps = [v.timestamp for v in values]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_get_values_limit限制返回数量(self) -> None:
        """get_values 的 limit 应限制返回条数."""
        for i in range(10):
            self.engine.record("m", float(i), timestamp=float(i))
        values = self.engine.get_values("m", limit=3)
        assert len(values) == 3

    def test_get_latest_获取最新值(self) -> None:
        """get_latest 应返回最新记录的指标值."""
        self.engine.record("m", 1.0, timestamp=100.0)
        self.engine.record("m", 2.0, timestamp=200.0)
        self.engine.record("m", 3.0, timestamp=300.0)
        latest = self.engine.get_latest("m")
        assert latest is not None
        assert latest.value == 3.0

    def test_get_latest_无数据返回None(self) -> None:
        """get_latest 无数据时应返回 None."""
        latest = self.engine.get_latest("nonexistent")
        assert latest is None

    def test_define_metric_注册指标定义(self) -> None:
        """define_metric 应注册指标定义."""
        md = MetricDefinition(name="m1", description="测试指标")
        self.engine.define_metric(md)
        assert "m1" in self.engine._definitions

    def test_define_metric_覆盖同名定义(self) -> None:
        """define_metric 同名应覆盖旧定义."""
        self.engine.define_metric(MetricDefinition(name="m1", description="旧"))
        self.engine.define_metric(MetricDefinition(name="m1", description="新"))
        assert self.engine._definitions["m1"].description == "新"


# ============================================================
# 13. MetricsEngine 聚合
# ============================================================


class TestMetricsEngine聚合:
    """验证 MetricsEngine 的聚合函数."""

    def setup_method(self) -> None:
        self.engine = MetricsEngine()

    def test_aggregate_avg_计算平均值(self) -> None:
        """aggregate avg 应返回平均值."""
        self.engine.record("m", 10.0)
        self.engine.record("m", 20.0)
        self.engine.record("m", 30.0)
        result = self.engine.aggregate("m", "avg")
        assert result == 20.0

    def test_aggregate_sum_计算总和(self) -> None:
        """aggregate sum 应返回总和."""
        self.engine.record("m", 10.0)
        self.engine.record("m", 20.0)
        result = self.engine.aggregate("m", "sum")
        assert result == 30.0

    def test_aggregate_min_计算最小值(self) -> None:
        """aggregate min 应返回最小值."""
        self.engine.record("m", 10.0)
        self.engine.record("m", 5.0)
        self.engine.record("m", 20.0)
        result = self.engine.aggregate("m", "min")
        assert result == 5.0

    def test_aggregate_max_计算最大值(self) -> None:
        """aggregate max 应返回最大值."""
        self.engine.record("m", 10.0)
        self.engine.record("m", 5.0)
        self.engine.record("m", 20.0)
        result = self.engine.aggregate("m", "max")
        assert result == 20.0

    def test_aggregate_count_计算数量(self) -> None:
        """aggregate count 应返回记录数量."""
        self.engine.record("m", 1.0)
        self.engine.record("m", 2.0)
        self.engine.record("m", 3.0)
        result = self.engine.aggregate("m", "count")
        assert result == 3.0

    def test_aggregate_未知函数返回零(self) -> None:
        """aggregate 传入未知函数应返回 0.0."""
        self.engine.record("m", 1.0)
        result = self.engine.aggregate("m", "unknown_func")
        assert result == 0.0

    def test_aggregate_无数据返回零(self) -> None:
        """aggregate 无数据时应返回 0.0."""
        result = self.engine.aggregate("nonexistent", "avg")
        assert result == 0.0

    def test_aggregate_按时间范围(self) -> None:
        """aggregate 应支持按时间范围过滤."""
        self.engine.record("m", 10.0, timestamp=100.0)
        self.engine.record("m", 20.0, timestamp=200.0)
        self.engine.record("m", 30.0, timestamp=300.0)
        result = self.engine.aggregate("m", "sum", start_time=150.0, end_time=250.0)
        assert result == 20.0


# ============================================================
# 14. MetricsEngine SLO
# ============================================================


class TestMetricsEngine_SLO:
    """验证 SLO 注册、获取与评估."""

    def setup_method(self) -> None:
        self.engine = MetricsEngine()

    def test_register_slo_注册SLO定义(self) -> None:
        """register_slo 应注册 SLO 定义."""
        slo = SLODefinition(name="test_slo", metric_name="task_success")
        self.engine.register_slo(slo)
        assert self.engine.get_slo("test_slo") is not None

    def test_get_slo_获取已注册的SLO(self) -> None:
        """get_slo 应返回已注册的 SLO 定义."""
        slo = SLODefinition(name="test_slo", metric_name="task_success")
        self.engine.register_slo(slo)
        found = self.engine.get_slo("test_slo")
        assert found is not None
        assert found.name == "test_slo"
        assert found.metric_name == "task_success"

    def test_get_slo_不存在的SLO返回None(self) -> None:
        """get_slo 传入不存在的名称应返回 None."""
        result = self.engine.get_slo("nonexistent")
        assert result is None

    def test_evaluate_slo_全成功_合规率100(self) -> None:
        """全成功数据评估应得到 100% 合规率."""
        slo = SLODefinition(
            name="test_slo",
            metric_name="task_success",
            target_percentage=99.5,
            evaluation_window_seconds=3600.0,
        )
        self.engine.register_slo(slo)
        now = time.time()
        for _ in range(10):
            self.engine.record("task_success", 1.0, timestamp=now)

        snapshot = self.engine.evaluate_slo("test_slo")
        assert snapshot.compliance_percentage == 100.0
        assert snapshot.error_budget_remaining == 0.5
        assert snapshot.burn_rate == 0.0

    def test_evaluate_slo_全失败_合规率0(self) -> None:
        """全失败数据评估应得到 0% 合规率."""
        slo = SLODefinition(
            name="test_slo",
            metric_name="task_success",
            target_percentage=99.5,
            evaluation_window_seconds=3600.0,
        )
        self.engine.register_slo(slo)
        now = time.time()
        for _ in range(10):
            self.engine.record("task_success", 0.0, timestamp=now)

        snapshot = self.engine.evaluate_slo("test_slo")
        assert snapshot.compliance_percentage == 0.0
        assert snapshot.error_budget_remaining == 0.0

    def test_evaluate_slo_混合结果_合规率正确(self) -> None:
        """混合成功与失败应正确计算合规率."""
        slo = SLODefinition(
            name="test_slo",
            metric_name="task_success",
            target_percentage=99.0,
            evaluation_window_seconds=3600.0,
        )
        self.engine.register_slo(slo)
        now = time.time()
        for _ in range(8):
            self.engine.record("task_success", 1.0, timestamp=now)
        for _ in range(2):
            self.engine.record("task_success", 0.0, timestamp=now)

        snapshot = self.engine.evaluate_slo("test_slo")
        assert snapshot.compliance_percentage == 80.0

    def test_evaluate_slo_无数据_合规率100(self) -> None:
        """无数据时 SLO 评估应默认合规率 100%."""
        slo = SLODefinition(
            name="test_slo",
            metric_name="task_success",
            target_percentage=99.5,
        )
        self.engine.register_slo(slo)
        snapshot = self.engine.evaluate_slo("test_slo")
        assert snapshot.compliance_percentage == 100.0
        assert snapshot.burn_rate == 0.0

    def test_evaluate_slo_超出窗口的数据不计入(self) -> None:
        """超出评估窗口的数据不应计入 SLO 评估."""
        slo = SLODefinition(
            name="test_slo",
            metric_name="task_success",
            evaluation_window_seconds=3600.0,
        )
        self.engine.register_slo(slo)
        # 窗口外的旧数据
        old_time = time.time() - 4000.0
        self.engine.record("task_success", 0.0, timestamp=old_time)
        # 窗口内的新数据
        now = time.time()
        self.engine.record("task_success", 1.0, timestamp=now)

        snapshot = self.engine.evaluate_slo("test_slo")
        assert snapshot.compliance_percentage == 100.0

    def test_evaluate_slo_不存在的SLO抛出异常(self) -> None:
        """评估未注册的 SLO 应抛出 ValueError."""
        with pytest.raises(ValueError, match="未注册"):
            self.engine.evaluate_slo("nonexistent_slo")

    def test_evaluate_slo_返回SLOSnapshot(self) -> None:
        """evaluate_slo 应返回 SLOSnapshot 实例."""
        slo = SLODefinition(name="test_slo", metric_name="m")
        self.engine.register_slo(slo)
        snapshot = self.engine.evaluate_slo("test_slo")
        assert isinstance(snapshot, SLOSnapshot)
        assert snapshot.slo_name == "test_slo"

    def test_evaluate_all_slos_评估所有已注册的SLO(self) -> None:
        """evaluate_all_slos 应评估所有已注册 SLO."""
        self.engine.register_slo(SLODefinition(name="slo_a", metric_name="m_a"))
        self.engine.register_slo(SLODefinition(name="slo_b", metric_name="m_b"))
        snapshots = self.engine.evaluate_all_slos()
        assert len(snapshots) == 2
        names = {s.slo_name for s in snapshots}
        assert "slo_a" in names
        assert "slo_b" in names


# ============================================================
# 15. MetricsEngine BurnRate 告警
# ============================================================


class TestMetricsEngine_BurnRate告警:
    """验证 Burn Rate 告警触发机制."""

    def setup_method(self) -> None:
        self.engine = MetricsEngine()

    def test_BurnRate_不触发告警_合规率达标(self) -> None:
        """合规率高于目标时不应触发 Burn Rate 告警."""
        slo = SLODefinition(
            name="test_slo",
            metric_name="task_success",
            target_percentage=99.0,
            evaluation_window_seconds=3600.0,
        )
        self.engine.register_slo(slo)
        now = time.time()
        for _ in range(10):
            self.engine.record("task_success", 1.0, timestamp=now)

        snapshot = self.engine.evaluate_slo("test_slo")
        assert snapshot.alert_fired is False
        assert snapshot.alert_threshold is None

    def test_BurnRate_触发最高阈值告警(self) -> None:
        """Burn Rate 超过最高阈值时应触发告警."""
        slo = SLODefinition(
            name="test_slo",
            metric_name="task_success",
            target_percentage=99.9,
            evaluation_window_seconds=3600.0,
            burn_rate_alerts=[14.4, 6.0, 3.0],
        )
        self.engine.register_slo(slo)
        now = time.time()
        # 全部失败，触发高 burn rate
        for _ in range(100):
            self.engine.record("task_success", 0.0, timestamp=now)

        snapshot = self.engine.evaluate_slo("test_slo")
        assert snapshot.alert_fired is True
        assert snapshot.alert_threshold == 14.4

    def test_BurnRate_alerts列表存储(self) -> None:
        """触发的告警应存储到引擎内部."""
        slo = SLODefinition(
            name="test_slo",
            metric_name="task_success",
            target_percentage=99.9,
            evaluation_window_seconds=3600.0,
        )
        self.engine.register_slo(slo)
        now = time.time()
        for _ in range(50):
            self.engine.record("task_success", 0.0, timestamp=now)

        self.engine.evaluate_slo("test_slo")
        alerts = self.engine.get_burn_rate_alerts("test_slo")
        assert len(alerts) >= 1
        assert alerts[0].slo_name == "test_slo"

    def test_get_burn_rate_alerts_按SLO过滤(self) -> None:
        """get_burn_rate_alerts 应按 slo_name 过滤."""
        alert_a = BurnRateAlert(
            slo_name="slo_a", burn_rate=10.0, threshold=14.4,
            error_budget_remaining=0.0,
        )
        alert_b = BurnRateAlert(
            slo_name="slo_b", burn_rate=5.0, threshold=6.0,
            error_budget_remaining=1.0,
        )
        self.engine._alerts.extend([alert_a, alert_b])
        alerts_a = self.engine.get_burn_rate_alerts("slo_a")
        assert len(alerts_a) == 1
        assert alerts_a[0].slo_name == "slo_a"

    def test_get_burn_rate_alerts_不传slo_name返回全部(self) -> None:
        """get_burn_rate_alerts 不传 slo_name 应返回所有告警."""
        self.engine._alerts.append(
            BurnRateAlert(slo_name="a", burn_rate=1.0, threshold=3.0, error_budget_remaining=0.5),
        )
        alerts = self.engine.get_burn_rate_alerts()
        assert len(alerts) == 1

    def test_get_burn_rate_alerts_limit限制返回数量(self) -> None:
        """get_burn_rate_alerts 的 limit 应限制返回条数."""
        for i in range(10):
            self.engine._alerts.append(
                BurnRateAlert(
                    slo_name=f"slo-{i}", burn_rate=1.0, threshold=3.0,
                    error_budget_remaining=0.5,
                ),
            )
        alerts = self.engine.get_burn_rate_alerts(limit=3)
        assert len(alerts) == 3

    def test_BurnRateAlert_默认severity为warning(self) -> None:
        """BurnRateAlert 的 severity 默认应为 warning."""
        alert = BurnRateAlert(
            slo_name="x", burn_rate=1.0, threshold=3.0, error_budget_remaining=0.5,
        )
        assert alert.severity == "warning"


# ============================================================
# 16. MetricsEngine DORA
# ============================================================


class TestMetricsEngine_DORA:
    """验证 DORA 指标记录与查询."""

    def setup_method(self) -> None:
        self.engine = MetricsEngine()

    def test_record_dora_deployment_记录成功部署(self) -> None:
        """record_dora_deployment 成功时应记录 dora_deployment=1.0."""
        self.engine.record_dora_deployment("agent-001", success=True)
        values = self.engine.get_values("dora_deployment", labels_filter={"agent": "agent-001"})
        assert len(values) == 1
        assert values[0].value == 1.0

    def test_record_dora_deployment_记录失败部署(self) -> None:
        """record_dora_deployment 失败时应记录 dora_deployment=0.0."""
        self.engine.record_dora_deployment("agent-001", success=False)
        values = self.engine.get_values("dora_deployment", labels_filter={"agent": "agent-001"})
        assert len(values) == 1
        assert values[0].value == 0.0

    def test_record_dora_deployment_记录延迟(self) -> None:
        """record_dora_deployment 传入 latency_ms 时应记录延迟指标."""
        self.engine.record_dora_deployment("agent-001", success=True, latency_ms=150.0)
        values = self.engine.get_values(
            "dora_deployment_latency_ms",
            labels_filter={"agent": "agent-001"},
        )
        assert len(values) == 1
        assert values[0].value == 150.0

    def test_record_dora_deployment_零延迟不记录延迟指标(self) -> None:
        """record_dora_deployment latency_ms=0 时不记录延迟指标."""
        self.engine.record_dora_deployment("agent-001", success=True, latency_ms=0.0)
        values = self.engine.get_values("dora_deployment_latency_ms")
        assert len(values) == 0

    def test_get_dora_metrics_部署频率24h(self) -> None:
        """get_dora_metrics 应返回 24 小时内的部署次数."""
        now = time.time()
        for _ in range(5):
            self.engine.record("dora_deployment", 1.0, labels={"agent": "agent-001"}, timestamp=now)
        metrics = self.engine.get_dora_metrics("agent-001")
        assert metrics["deployment_frequency_24h"] == 5

    def test_get_dora_metrics_变更失败率(self) -> None:
        """get_dora_metrics 应正确计算变更失败率."""
        now = time.time()
        for _ in range(8):
            self.engine.record("dora_deployment", 1.0, labels={"agent": "agent-001"}, timestamp=now)
        for _ in range(2):
            self.engine.record("dora_deployment", 0.0, labels={"agent": "agent-001"}, timestamp=now)
        metrics = self.engine.get_dora_metrics("agent-001")
        assert metrics["change_failure_rate"] == 0.2

    def test_get_dora_metrics_平均前置时间(self) -> None:
        """get_dora_metrics 应正确计算平均前置时间."""
        now = time.time()
        self.engine.record("dora_deployment_latency_ms", 100.0, labels={"agent": "agent-001"}, timestamp=now)
        self.engine.record("dora_deployment_latency_ms", 200.0, labels={"agent": "agent-001"}, timestamp=now)
        metrics = self.engine.get_dora_metrics("agent-001")
        assert metrics["avg_lead_time_ms"] == 150.0

    def test_get_dora_metrics_无部署数据返回零值(self) -> None:
        """get_dora_metrics 无数据时应返回零值."""
        metrics = self.engine.get_dora_metrics("agent-001")
        assert metrics["deployment_frequency_24h"] == 0
        assert metrics["change_failure_rate"] == 0.0
        assert metrics["avg_lead_time_ms"] == 0.0

    def test_get_dora_metrics_不传agent_id返回全局数据(self) -> None:
        """get_dora_metrics 不传 agent_id 应返回全局数据."""
        now = time.time()
        self.engine.record("dora_deployment", 1.0, labels={"agent": "a"}, timestamp=now)
        self.engine.record("dora_deployment", 1.0, labels={"agent": "b"}, timestamp=now)
        metrics = self.engine.get_dora_metrics()
        assert metrics["deployment_frequency_24h"] == 2


# ============================================================
# 17. MetricsEngine 统计与清空
# ============================================================


class TestMetricsEngine统计与清空:
    """验证统计信息获取、清空与 FIFO 机制."""

    def setup_method(self) -> None:
        self.engine = MetricsEngine()

    def test_get_stats_初始状态(self) -> None:
        """初始状态统计应为零值."""
        stats = self.engine.get_stats()
        assert stats["defined_metrics"] == 0
        assert stats["metric_names"] == []
        assert stats["total_values"] == 0
        assert stats["registered_slos"] == 0
        assert stats["slo_names"] == []
        assert stats["burn_rate_alerts"] == 0

    def test_get_stats_记录后更新(self) -> None:
        """记录数据后统计应正确更新."""
        self.engine.define_metric(MetricDefinition(name="m1"))
        self.engine.record("m1", 1.0)
        self.engine.record("m1", 2.0)
        self.engine.register_slo(SLODefinition(name="slo1", metric_name="m1"))
        stats = self.engine.get_stats()
        assert stats["defined_metrics"] == 1
        assert "m1" in stats["metric_names"]
        assert stats["total_values"] == 2
        assert stats["registered_slos"] == 1
        assert "slo1" in stats["slo_names"]

    def test_clear_清空所有数据(self) -> None:
        """clear 应清空所有内部状态."""
        self.engine.define_metric(MetricDefinition(name="m1"))
        self.engine.record("m1", 1.0)
        self.engine.register_slo(SLODefinition(name="slo1", metric_name="m1"))
        self.engine.clear()
        stats = self.engine.get_stats()
        assert stats["defined_metrics"] == 0
        assert stats["total_values"] == 0
        assert stats["registered_slos"] == 0

    def test_FIFO_指标值超容量淘汰(self) -> None:
        """指标值超过 max_values_per_metric 时应淘汰旧值."""
        engine = MetricsEngine(max_values_per_metric=5)
        for i in range(8):
            engine.record("m", float(i), timestamp=float(i))
        values = engine.get_values("m")
        assert len(values) == 5
        # 最早的 3 条已被淘汰，保留后 5 条
        vals = [v.value for v in values]
        assert 3.0 in vals
        assert 4.0 in vals
        assert 7.0 in vals
        assert 0.0 not in vals

    def test_FIFO_不影响其他指标(self) -> None:
        """一个指标的 FIFO 淘汰不应影响其他指标."""
        engine = MetricsEngine(max_values_per_metric=3)
        for i in range(5):
            engine.record("m1", float(i), timestamp=float(i))
            engine.record("m2", float(i), timestamp=float(i))
        # m1 应只剩 3 条
        assert len(engine._values["m1"]) == 3
        # m2 也只剩 3 条
        assert len(engine._values["m2"]) == 3


# ============================================================
# 18. NISTFunction 枚举
# ============================================================


class TestNISTFunction枚举:
    """验证 NISTFunction 枚举的定义."""

    def test_GOVERN值(self) -> None:
        """GOVERN 枚举值应为 govern."""
        assert NISTFunction.GOVERN == "govern"

    def test_MAP值(self) -> None:
        """MAP 枚举值应为 map."""
        assert NISTFunction.MAP == "map"

    def test_MEASURE值(self) -> None:
        """MEASURE 枚举值应为 measure."""
        assert NISTFunction.MEASURE == "measure"

    def test_MANAGE值(self) -> None:
        """MANAGE 枚举值应为 manage."""
        assert NISTFunction.MANAGE == "manage"

    def test_枚举成员数量(self) -> None:
        """NISTFunction 应有 4 个成员."""
        assert len(NISTFunction) == 4

    def test_继承str类型(self) -> None:
        """NISTFunction 应继承 str，可直接作为字符串使用."""
        assert isinstance(NISTFunction.GOVERN, str)
        assert NISTFunction.GOVERN.upper() == "GOVERN"


# ============================================================
# 19. ComplianceControl 模型
# ============================================================


class TestComplianceControl模型:
    """验证 ComplianceControl 数据模型."""

    def test_最小创建_只有必要字段(self) -> None:
        """只传必要字段创建控制点."""
        ctrl = ComplianceControl(
            control_id="CC6.1",
            name="逻辑与物理访问控制",
            framework="SOC2",
            nist_function=NISTFunction.GOVERN,
        )
        assert ctrl.control_id == "CC6.1"
        assert ctrl.name == "逻辑与物理访问控制"
        assert ctrl.framework == "SOC2"
        assert ctrl.status == "compliant"
        assert ctrl.evidence == []
        assert ctrl.score == 1.0
        assert ctrl.findings == []
        assert ctrl.recommendations == []

    def test_完整创建_所有字段(self) -> None:
        """传入所有字段创建控制点."""
        ctrl = ComplianceControl(
            control_id="GOVERN-1",
            name="AI治理结构",
            framework="NIST_AI_RMF",
            nist_function=NISTFunction.GOVERN,
            description="建立AI治理",
            status="partial",
            evidence=["审计日志总数: 100"],
            score=0.7,
            findings=["缺少角色定义"],
            recommendations=["建议优化"],
        )
        assert ctrl.score == 0.7
        assert ctrl.status == "partial"
        assert len(ctrl.evidence) == 1
        assert ctrl.findings == ["缺少角色定义"]

    def test_score_不接受负值(self) -> None:
        """score 为负值应触发 ValidationError."""
        with pytest.raises(ValidationError):
            ComplianceControl(
                control_id="x", name="y", framework="z",
                nist_function=NISTFunction.GOVERN, score=-0.1,
            )

    def test_score_不接受超过1(self) -> None:
        """score 超过 1.0 应触发 ValidationError."""
        with pytest.raises(ValidationError):
            ComplianceControl(
                control_id="x", name="y", framework="z",
                nist_function=NISTFunction.GOVERN, score=1.1,
            )


# ============================================================
# 20. ComplianceDomain 模型
# ============================================================


class TestComplianceDomain模型:
    """验证 ComplianceDomain 数据模型与 compute_score."""

    def test_最小创建_只有必要字段(self) -> None:
        """只传必要字段创建合规域."""
        domain = ComplianceDomain(domain_id="soc2", name="SOC2")
        assert domain.domain_id == "soc2"
        assert domain.name == "SOC2"
        assert domain.controls == []
        assert domain.overall_score == 1.0

    def test_compute_score_空控制点返回1(self) -> None:
        """无控制点时 compute_score 应返回 1.0."""
        domain = ComplianceDomain(domain_id="x", name="y")
        assert domain.compute_score() == 1.0

    def test_compute_score_所有满分返回1(self) -> None:
        """所有控制点满分时 compute_score 应返回 1.0."""
        controls = [
            ComplianceControl(
                control_id=f"c{i}", name=f"控制{i}", framework="z",
                nist_function=NISTFunction.GOVERN, score=1.0,
            )
            for i in range(5)
        ]
        domain = ComplianceDomain(domain_id="x", name="y", controls=controls)
        assert domain.compute_score() == 1.0

    def test_compute_score_混合分数正确平均(self) -> None:
        """混合分数时 compute_score 应返回平均值."""
        controls = [
            ComplianceControl(
                control_id="c1", name="a", framework="z",
                nist_function=NISTFunction.GOVERN, score=0.8,
            ),
            ComplianceControl(
                control_id="c2", name="b", framework="z",
                nist_function=NISTFunction.GOVERN, score=0.6,
            ),
            ComplianceControl(
                control_id="c3", name="c", framework="z",
                nist_function=NISTFunction.GOVERN, score=1.0,
            ),
        ]
        domain = ComplianceDomain(domain_id="x", name="y", controls=controls)
        score = domain.compute_score()
        assert score == pytest.approx(0.8, abs=0.001)

    def test_compute_score_所有零分返回0(self) -> None:
        """所有控制点零分时 compute_score 应返回 0.0."""
        controls = [
            ComplianceControl(
                control_id=f"c{i}", name=f"控制{i}", framework="z",
                nist_function=NISTFunction.GOVERN, score=0.0,
            )
            for i in range(3)
        ]
        domain = ComplianceDomain(domain_id="x", name="y", controls=controls)
        assert domain.compute_score() == 0.0


# ============================================================
# 21. GovernanceComplianceReport 模型
# ============================================================


class TestGovernanceComplianceReport模型:
    """验证 GovernanceComplianceReport 数据模型与 compute_overall."""

    def test_最小创建_全部默认值(self) -> None:
        """不传参数创建报告，应使用默认值."""
        report = GovernanceComplianceReport()
        assert report.title == "治理合规报告"
        assert report.frameworks == []
        assert report.domains == []
        assert report.overall_score == 1.0
        assert report.summary == ""
        assert report.risks == []

    def test_compute_overall_空域返回1(self) -> None:
        """无域时 compute_overall 应返回 1.0."""
        report = GovernanceComplianceReport()
        assert report.compute_overall() == 1.0

    def test_compute_overall_多域平均(self) -> None:
        """多域时 compute_overall 应返回各域评分的平均值."""
        domain1 = ComplianceDomain(
            domain_id="d1", name="域1",
            controls=[
                ComplianceControl(
                    control_id="c1", name="x", framework="z",
                    nist_function=NISTFunction.GOVERN, score=0.8,
                ),
            ],
        )
        domain2 = ComplianceDomain(
            domain_id="d2", name="域2",
            controls=[
                ComplianceControl(
                    control_id="c2", name="x", framework="z",
                    nist_function=NISTFunction.GOVERN, score=0.6,
                ),
            ],
        )
        report = GovernanceComplianceReport(domains=[domain1, domain2])
        overall = report.compute_overall()
        assert overall == pytest.approx(0.7, abs=0.001)

    def test_report_id_自动生成(self) -> None:
        """report_id 应自动生成."""
        report = GovernanceComplianceReport()
        assert report.report_id.startswith("comp-")

    def test_generated_at_自动生成当前时间(self) -> None:
        """generated_at 应自动生成当前时间."""
        before = time.time()
        report = GovernanceComplianceReport()
        assert before <= report.generated_at


# ============================================================
# 22. ComplianceReporter 生成
# ============================================================


class TestComplianceReporter生成:
    """验证 ComplianceReporter 报告生成功能."""

    def setup_method(self) -> None:
        self.reporter = ComplianceReporter()

    def test_内置17个控制点(self) -> None:
        """ComplianceReporter 应内置 17 个控制点（SOC2:5 + NIST:8 + 学术:4）."""
        total = (
            len(self.reporter.SOC2_CONTROLS)
            + len(self.reporter.NIST_CONTROLS)
            + len(self.reporter.ACADEMIC_CONTROLS)
        )
        assert total == 17

    def test_generate_from_audit_空数据生成报告(self) -> None:
        """空审计数据应生成报告，但评分较低."""
        report = self.reporter.generate_from_audit(
            audit_stats={},
            metrics_stats={},
        )
        assert isinstance(report, GovernanceComplianceReport)
        assert len(report.domains) == 3
        assert report.overall_score < 1.0  # 空数据应有扣分

    def test_generate_from_audit_有完整数据(self) -> None:
        """完整审计数据应生成高分报告."""
        audit_stats = {
            "total_recorded": 1000,
            "unique_agents": 5,
            "unique_traces": 20,
            "outcome_distribution": {"allow": 900, "deny": 80, "error": 20},
            "total_anomalies": 3,
            "baselines": 5,
        }
        metrics_stats = {
            "registered_slos": 3,
        }
        report = self.reporter.generate_from_audit(
            audit_stats=audit_stats,
            metrics_stats=metrics_stats,
        )
        assert report.overall_score > 0.8

    def test_generate_from_audit_指定框架(self) -> None:
        """指定框架列表应只评估对应框架."""
        report = self.reporter.generate_from_audit(
            audit_stats={},
            frameworks=["SOC2"],
        )
        assert len(report.domains) == 1
        assert report.domains[0].domain_id == "soc2"
        assert report.frameworks == ["SOC2"]

    def test_generate_from_audit_报告标题正确(self) -> None:
        """生成报告的标题应为"治理合规评估报告"."""
        report = self.reporter.generate_from_audit(audit_stats={})
        assert report.title == "治理合规评估报告"

    def test_generate_from_audit_摘要非空(self) -> None:
        """生成报告的摘要应非空."""
        report = self.reporter.generate_from_audit(audit_stats={})
        assert report.summary != ""
        assert "整体合规评分" in report.summary

    def test_generate_from_audit_框架包含在报告中(self) -> None:
        """报告应包含评估的框架列表."""
        report = self.reporter.generate_from_audit(
            audit_stats={},
            frameworks=["SOC2", "NIST_AI_RMF"],
        )
        assert "SOC2" in report.frameworks
        assert "NIST_AI_RMF" in report.frameworks


# ============================================================
# 23. ComplianceReporter NIST 摘要
# ============================================================


class TestComplianceReporter_NIST摘要:
    """验证 NIST 四函数摘要生成."""

    def setup_method(self) -> None:
        self.reporter = ComplianceReporter()

    def test_generate_nist_summary_包含四个函数(self) -> None:
        """NIST 摘要应包含四个函数的键."""
        report = self.reporter.generate_from_audit(audit_stats={})
        summary = self.reporter.generate_nist_summary(report)
        assert "govern" in summary
        assert "map" in summary
        assert "measure" in summary
        assert "manage" in summary

    def test_generate_nist_summary_每个函数有controls和avg_score(self) -> None:
        """每个函数应有 controls 列表和 avg_score."""
        report = self.reporter.generate_from_audit(audit_stats={})
        summary = self.reporter.generate_nist_summary(report)
        for func_key in ["govern", "map", "measure", "manage"]:
            data = summary[func_key]
            assert "controls" in data
            assert "avg_score" in data
            assert isinstance(data["controls"], list)

    def test_generate_nist_summary_control_count正确(self) -> None:
        """control_count 应反映实际控制点数量."""
        report = self.reporter.generate_from_audit(audit_stats={})
        summary = self.reporter.generate_nist_summary(report)
        total = sum(data["control_count"] for data in summary.values())
        assert total == 17

    def test_generate_nist_summary_avg_score范围0到1(self) -> None:
        """avg_score 应在 0 到 1 之间."""
        report = self.reporter.generate_from_audit(audit_stats={})
        summary = self.reporter.generate_nist_summary(report)
        for func_key in summary:
            assert 0.0 <= summary[func_key]["avg_score"] <= 1.0

    def test_generate_nist_summary_controls包含id和score(self) -> None:
        """每个 control 应包含 id, name, score, status."""
        report = self.reporter.generate_from_audit(audit_stats={})
        summary = self.reporter.generate_nist_summary(report)
        govern_controls = summary["govern"]["controls"]
        if govern_controls:
            ctrl = govern_controls[0]
            assert "id" in ctrl
            assert "name" in ctrl
            assert "score" in ctrl
            assert "status" in ctrl


# ============================================================
# 24. ComplianceReporter 控制点评估
# ============================================================


class TestComplianceReporter控制点评估:
    """验证 _evaluate_control 对各 NIST 函数的评估逻辑."""

    def setup_method(self) -> None:
        self.reporter = ComplianceReporter()

    def test_GOVERN_有Agent_满分(self) -> None:
        """GOVERN 函数有 Agent 时不应扣分."""
        audit_stats = {"total_recorded": 100, "unique_agents": 5}
        ctrl_def = {
            "control_id": "GOVERN-1",
            "name": "AI治理",
            "nist_function": NISTFunction.GOVERN,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert ctrl.score == 1.0
        assert ctrl.status == "compliant"

    def test_GOVERN_无Agent_扣0_3分(self) -> None:
        """GOVERN 函数无 Agent 应扣 0.3 分."""
        audit_stats = {"total_recorded": 0, "unique_agents": 0}
        ctrl_def = {
            "control_id": "GOVERN-1",
            "name": "AI治理",
            "nist_function": NISTFunction.GOVERN,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert ctrl.score == 0.7
        assert "未检测到 Agent 治理结构" in ctrl.findings

    def test_GOVERN_添加审计日志证据(self) -> None:
        """GOVERN 函数应添加审计日志总数作为证据."""
        audit_stats = {"total_recorded": 500, "unique_agents": 3}
        ctrl_def = {
            "control_id": "GOVERN-1",
            "name": "AI治理",
            "nist_function": NISTFunction.GOVERN,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert any("审计日志总数: 500" in e for e in ctrl.evidence)

    def test_MAP_有traces_不扣分(self) -> None:
        """MAP 函数有追踪链时不应扣分."""
        audit_stats = {"unique_traces": 10}
        ctrl_def = {
            "control_id": "MAP-1",
            "name": "上下文识别",
            "nist_function": NISTFunction.MAP,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert ctrl.score == 1.0

    def test_MAP_无traces_扣0_2分(self) -> None:
        """MAP 函数无追踪链应扣 0.2 分."""
        audit_stats = {"unique_traces": 0}
        ctrl_def = {
            "control_id": "MAP-1",
            "name": "上下文识别",
            "nist_function": NISTFunction.MAP,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert ctrl.score == 0.8
        assert "缺少分布式追踪上下文映射" in ctrl.findings

    def test_MEASURE_有SLO_不扣分(self) -> None:
        """MEASURE 函数有 SLO 时不应因缺少 SLO 扣分."""
        audit_stats = {"baselines": 5, "total_anomalies": 0}
        metrics_stats = {"registered_slos": 3}
        ctrl_def = {
            "control_id": "MEASURE-1",
            "name": "性能评估",
            "nist_function": NISTFunction.MEASURE,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, metrics_stats)
        assert ctrl.score == 1.0

    def test_MEASURE_无SLO_扣0_4分(self) -> None:
        """MEASURE 函数无 SLO 应扣 0.4 分."""
        audit_stats = {"baselines": 5}
        metrics_stats = {"registered_slos": 0}
        ctrl_def = {
            "control_id": "MEASURE-1",
            "name": "性能评估",
            "nist_function": NISTFunction.MEASURE,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, metrics_stats)
        assert ctrl.score == 0.6
        assert any("未定义 SLO 度量指标" in f for f in ctrl.findings)

    def test_MEASURE_无baselines_额外扣0_2分(self) -> None:
        """MEASURE 函数无 baselines 应额外扣 0.2 分."""
        audit_stats = {"baselines": 0}
        metrics_stats = {"registered_slos": 0}
        ctrl_def = {
            "control_id": "MEASURE-1",
            "name": "性能评估",
            "nist_function": NISTFunction.MEASURE,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, metrics_stats)
        assert ctrl.score == 0.4  # 1.0 - 0.4 - 0.2
        assert any("未建立行为基线" in f for f in ctrl.findings)

    def test_MANAGE_拒绝率正常_不扣分(self) -> None:
        """MANAGE 函数拒绝率正常（<=30%）时不应扣分."""
        audit_stats = {"outcome_distribution": {"allow": 90, "deny": 10}}
        ctrl_def = {
            "control_id": "MANAGE-1",
            "name": "风险处置",
            "nist_function": NISTFunction.MANAGE,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert ctrl.score == 1.0

    def test_MANAGE_拒绝率过高_扣0_3分(self) -> None:
        """MANAGE 函数拒绝率超过 30% 应扣 0.3 分."""
        audit_stats = {"outcome_distribution": {"allow": 60, "deny": 30, "error": 10}}
        ctrl_def = {
            "control_id": "MANAGE-1",
            "name": "风险处置",
            "nist_function": NISTFunction.MANAGE,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert ctrl.score == 0.7
        assert any("拒绝率过高" in f for f in ctrl.findings)

    def test_MANAGE_无决策记录_扣0_3分(self) -> None:
        """MANAGE 函数无决策记录应扣 0.3 分."""
        audit_stats = {"outcome_distribution": {}}
        ctrl_def = {
            "control_id": "MANAGE-1",
            "name": "风险处置",
            "nist_function": NISTFunction.MANAGE,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert ctrl.score == 0.7
        assert any("无决策记录" in f for f in ctrl.findings)

    def test_score低于0_5状态为non_compliant(self) -> None:
        """score < 0.5 时 status 应为 non_compliant."""
        audit_stats = {"baselines": 0}
        metrics_stats = {"registered_slos": 0}
        ctrl_def = {
            "control_id": "MEASURE-1",
            "name": "性能评估",
            "nist_function": NISTFunction.MEASURE,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, metrics_stats)
        assert ctrl.status == "non_compliant"

    def test_score介于0_5到0_8状态为partial(self) -> None:
        """0.5 <= score < 0.8 时 status 应为 partial."""
        audit_stats = {"unique_agents": 0}
        ctrl_def = {
            "control_id": "GOVERN-1",
            "name": "治理",
            "nist_function": NISTFunction.GOVERN,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert ctrl.status == "partial"

    def test_score高于0_8状态为compliant(self) -> None:
        """score >= 0.8 时 status 应为 compliant."""
        audit_stats = {"unique_agents": 5, "total_recorded": 100}
        ctrl_def = {
            "control_id": "GOVERN-1",
            "name": "治理",
            "nist_function": NISTFunction.GOVERN,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert ctrl.status == "compliant"

    def test_score低于0_5添加紧急改进建议(self) -> None:
        """score < 0.5 时应添加紧急改进建议."""
        audit_stats = {"baselines": 0}
        metrics_stats = {"registered_slos": 0}
        ctrl_def = {
            "control_id": "MEASURE-1",
            "name": "性能评估",
            "nist_function": NISTFunction.MEASURE,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, metrics_stats)
        assert any("紧急改进" in r for r in ctrl.recommendations)

    def test_score介于0_5到0_8添加建议优化建议(self) -> None:
        """0.5 <= score < 0.8 时应添加建议优化建议."""
        audit_stats = {"unique_agents": 0}
        ctrl_def = {
            "control_id": "GOVERN-1",
            "name": "治理",
            "nist_function": NISTFunction.GOVERN,
        }
        ctrl = self.reporter._evaluate_control(ctrl_def, audit_stats, {})
        assert any("建议优化" in r for r in ctrl.recommendations)

    def test_risk_收集_score低于0_7(self) -> None:
        """score < 0.7 的控制点应被收集到风险列表."""
        report = self.reporter.generate_from_audit(audit_stats={})
        # 空数据应产生风险项
        assert len(report.risks) > 0

    def test_risk_severity_high当score低于0_5(self) -> None:
        """score < 0.5 的风险项 severity 应为 high."""
        report = self.reporter.generate_from_audit(audit_stats={})
        high_risks = [r for r in report.risks if r["severity"] == "high"]
        # 空数据时 MEASURE 控制点 score=0.4，应产生 high 风险
        assert len(high_risks) >= 1

    def test_risk_severity_medium当score介于0_5到0_7(self) -> None:
        """0.5 <= score < 0.7 的风险项 severity 应为 medium."""
        # MEASURE: baselines=3 有基线不扣分，无 SLO 扣 0.4，score=0.6
        report = self.reporter.generate_from_audit(
            audit_stats={"baselines": 3, "total_anomalies": 0},
            metrics_stats={"registered_slos": 0},
        )
        medium_risks = [r for r in report.risks if r["severity"] == "medium"]
        assert len(medium_risks) >= 1


# ============================================================
# 25. 端到端集成
# ============================================================


class Test端到端集成:
    """验证审计+度量+合规报告的完整工作流."""

    def test_完整流程_审计记录到合规报告(self) -> None:
        """从审计记录到合规报告的完整流程."""
        # 1. 创建引擎
        audit = AuditEngine()
        metrics = MetricsEngine()
        reporter = ComplianceReporter()

        # 2. 记录审计日志
        for i in range(20):
            audit.record(
                actor=f"agent-{i % 3}",
                action="policy_eval" if i % 2 == 0 else "tool_call",
                layer="L0",
                outcome="allow" if i % 5 != 0 else "deny",
                latency_ms=10.0 + i,
                agent_id=f"agent-{i % 3}",
                trace_id=f"trace-{i % 4}",
            )

        # 3. 注册指标和 SLO
        metrics.define_metric(MetricDefinition(name="task_success", description="任务成功率"))
        metrics.register_slo(SLODefinition(
            name="agent_task_slo",
            metric_name="task_success",
            target_percentage=99.0,
        ))

        # 4. 记录指标
        now = time.time()
        for i in range(50):
            metrics.record("task_success", 1.0 if i % 10 != 0 else 0.0, timestamp=now)

        # 5. 记录 DORA
        metrics.record_dora_deployment("agent-0", success=True, latency_ms=120.0)
        metrics.record_dora_deployment("agent-1", success=False, latency_ms=200.0)

        # 6. 构建基线与异常检测
        baseline = audit.build_baseline("agent-0")
        assert isinstance(baseline, BehaviorBaseline)

        # 7. 获取统计数据
        audit_stats = audit.get_stats()
        metrics_stats = metrics.get_stats()
        assert audit_stats["total_recorded"] == 20
        assert metrics_stats["registered_slos"] == 1

        # 8. 评估 SLO
        snapshot = metrics.evaluate_slo("agent_task_slo")
        assert isinstance(snapshot, SLOSnapshot)

        # 9. 获取 DORA 指标
        dora = metrics.get_dora_metrics()
        assert dora["deployment_frequency_24h"] == 2

        # 10. 生成合规报告
        report = reporter.generate_from_audit(
            audit_stats=audit_stats,
            metrics_stats=metrics_stats,
        )
        assert isinstance(report, GovernanceComplianceReport)
        assert len(report.domains) == 3
        assert report.overall_score > 0
        assert report.summary != ""

    def test_完整流程_导出审计摘要(self) -> None:
        """审计引擎导出摘要应包含完整统计."""
        audit = AuditEngine()
        audit.record(action="eval", outcome="allow", latency_ms=5.0)
        audit.record(action="call", outcome="deny", latency_ms=10.0)
        summary = audit.export_summary()
        assert summary["log_count"] == 2
        assert summary["latency_stats"]["count"] == 2.0

    def test_完整流程_NIST摘要与报告一致性(self) -> None:
        """NIST 摘要的控制点总数应与报告控制点总数一致."""
        reporter = ComplianceReporter()
        report = reporter.generate_from_audit(
            audit_stats={"unique_agents": 3, "unique_traces": 5, "baselines": 2},
            metrics_stats={"registered_slos": 2},
        )
        nist_summary = reporter.generate_nist_summary(report)
        total_controls = sum(d["control_count"] for d in nist_summary.values())
        report_total = sum(len(d.controls) for d in report.domains)
        assert total_controls == report_total

    def test_完整流程_多Agent隔离(self) -> None:
        """不同 Agent 的审计数据应互不干扰."""
        audit = AuditEngine()
        audit.record(agent_id="agent-a", action="eval", outcome="allow", latency_ms=10.0)
        audit.record(agent_id="agent-b", action="call", outcome="deny", latency_ms=20.0)

        # 按 agent_id 查询应隔离
        logs_a = audit.query(agent_id="agent-a")
        logs_b = audit.query(agent_id="agent-b")
        assert len(logs_a) == 1
        assert len(logs_b) == 1
        assert logs_a[0].agent_id == "agent-a"
        assert logs_b[0].agent_id == "agent-b"

        # 聚合也应隔离
        agg_a = audit.aggregate_by_action(agent_id="agent-a")
        agg_b = audit.aggregate_by_action(agent_id="agent-b")
        assert "eval" in agg_a
        assert "call" in agg_b
        assert "eval" not in agg_b

    def test_完整流程_合规报告生成后引擎状态不变(self) -> None:
        """生成合规报告后审计和度量引擎内部状态不应改变."""
        audit = AuditEngine()
        metrics = MetricsEngine()
        audit.record(actor="agent-a", outcome="allow")
        metrics.record("m", 1.0)

        audit_stats_before = audit.get_stats()["total_recorded"]
        metrics_stats_before = metrics.get_stats()["total_values"]

        reporter = ComplianceReporter()
        reporter.generate_from_audit(audit_stats=audit.get_stats(), metrics_stats=metrics.get_stats())

        assert audit.get_stats()["total_recorded"] == audit_stats_before
        assert metrics.get_stats()["total_values"] == metrics_stats_before

    def test_完整流程_SLO评估多次结果一致(self) -> None:
        """同一时间窗口内多次评估 SLO 结果应一致."""
        metrics = MetricsEngine()
        metrics.register_slo(SLODefinition(
            name="test_slo", metric_name="m", target_percentage=99.0,
        ))
        now = time.time()
        for i in range(10):
            metrics.record("m", 1.0 if i < 9 else 0.0, timestamp=now)

        snap1 = metrics.evaluate_slo("test_slo")
        snap2 = metrics.evaluate_slo("test_slo")
        assert snap1.compliance_percentage == snap2.compliance_percentage
