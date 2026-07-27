"""T3 A2A 增强模块 - 单元测试.

测试覆盖:
1. A2AMetrics（指标收集、导出、重置）
2. AgentHealthTracker（健康评分、智能路由、导出）
3. TokenStore + 指纹（Token 生命周期、指纹生成与验证）

所有测试为同步测试，无需 pytest.mark.asyncio。
"""

from __future__ import annotations

import time

import pytest

from dy3_polaris.l6.a2a.metrics import A2AMetrics, _Counter, _LatencyTracker
from dy3_polaris.l6.a2a.health import AgentHealthTracker
from dy3_polaris.l6.a2a.auth import TokenStore, agent_fingerprint, verify_fingerprint


# ============================================================
# 1. A2AMetrics 测试
# ============================================================

class TestCounterBasic:
    """_Counter 基础计数测试."""

    def test_counter_basic(self):
        """创建 _Counter 实例，inc() 多次，验证 value."""
        counter = _Counter()
        assert counter.value == 0

        counter.inc()
        assert counter.value == 1

        counter.inc(5)
        assert counter.value == 6

        counter.inc(3)
        assert counter.value == 9


class TestLatencyTrackerBasic:
    """_LatencyTracker 基础延迟追踪测试."""

    def test_latency_tracker_basic(self):
        """创建 _LatencyTracker，record 多个值，验证 count/avg_ms/max_ms/p99_ms."""
        tracker = _LatencyTracker(max_samples=100)
        samples = [100.0, 200.0, 300.0, 400.0, 500.0]

        for s in samples:
            tracker.record(s)

        assert tracker.count == 5
        assert tracker.avg_ms == 300.0  # (100+200+300+400+500)/5
        assert tracker.max_ms == 500.0
        # p99: 5 个样本排序后索引 = min(int(5*0.99), 4) = min(4, 4) = 4 -> 500.0
        assert tracker.p99_ms == 500.0

    def test_latency_tracker_empty(self):
        """空 _LatencyTracker，所有属性返回 0."""
        tracker = _LatencyTracker()
        assert tracker.count == 0
        assert tracker.avg_ms == 0.0
        assert tracker.max_ms == 0.0
        assert tracker.p99_ms == 0.0

    def test_latency_tracker_single_sample(self):
        """单个样本的 p99."""
        tracker = _LatencyTracker()
        tracker.record(42.0)
        assert tracker.count == 1
        assert tracker.p99_ms == 42.0

    def test_latency_tracker_capped(self):
        """记录超过 max_samples 个值，验证只保留最近 max_samples 个."""
        max_samples = 5
        tracker = _LatencyTracker(max_samples=max_samples)

        # 记录 10 个值
        for i in range(10):
            tracker.record(float(i * 100))

        assert tracker.count == max_samples  # 只保留 5 个
        # 最近的 5 个值: 500, 600, 700, 800, 900
        assert tracker.avg_ms == 700.0  # (500+600+700+800+900)/5
        assert tracker.max_ms == 900.0


class TestMetricsInitial:
    """A2AMetrics 初始状态测试."""

    def test_metrics_initial_state(self):
        """创建 A2AMetrics，export() 验证所有初始值为 0."""
        metrics = A2AMetrics()
        result = metrics.export()

        # uptime > 0（刚创建）
        assert result["uptime_seconds"] >= 0

        # 消息指标初始值
        assert result["messages"]["total_sent"] == 0
        assert result["messages"]["total_received"] == 0
        assert result["messages"]["by_type"] == {}

        # 任务指标初始值
        assert result["tasks"]["created"] == 0
        assert result["tasks"]["completed"] == 0
        assert result["tasks"]["failed"] == 0
        assert result["tasks"]["cancelled"] == 0
        assert result["tasks"]["timeout"] == 0
        assert result["tasks"]["success_rate"] == 0.0

        # 延迟初始值
        assert result["tasks"]["latency_ms"]["avg"] == 0.0
        assert result["tasks"]["latency_ms"]["max"] == 0.0
        assert result["tasks"]["latency_ms"]["p99"] == 0.0
        assert result["tasks"]["latency_ms"]["samples"] == 0

        # 会话初始值
        assert result["sessions"]["created"] == 0
        assert result["sessions"]["closed"] == 0
        assert result["sessions"]["expired"] == 0
        assert result["sessions"]["active"] == 0

        # Agent 与错误初始值
        assert result["agents"] == {}
        assert result["errors"] == {}
        assert result["capabilities"] == {}


class TestMetricsMessageEvents:
    """A2AMetrics 消息事件测试."""

    def test_metrics_message_events(self):
        """on_message_sent / on_message_received 多次后 export 验证 messages.by_type 和 totals."""
        metrics = A2AMetrics()

        # 发送 3 条 TASK_REQUEST
        metrics.on_message_sent("TASK_REQUEST", "agent-A", "agent-B")
        metrics.on_message_sent("TASK_REQUEST", "agent-A", "agent-C")
        metrics.on_message_sent("TASK_REQUEST", "agent-D", "agent-B")

        # 发送 2 条 TASK_RESULT
        metrics.on_message_sent("TASK_RESULT", "agent-B", "agent-A")
        metrics.on_message_sent("TASK_RESULT", "agent-C", "agent-A")

        # 接收 4 条消息
        metrics.on_message_received("TASK_REQUEST", "agent-B")
        metrics.on_message_received("TASK_REQUEST", "agent-C")
        metrics.on_message_received("TASK_REQUEST", "agent-B")
        metrics.on_message_received("TASK_RESULT", "agent-A")

        result = metrics.export()

        # 按类型统计: by_type 只统计 sent 方向
        assert result["messages"]["by_type"]["TASK_REQUEST"] == 3
        assert result["messages"]["by_type"]["TASK_RESULT"] == 2

        # 总发送 = 5
        assert result["messages"]["total_sent"] == 5
        # 总接收 = 4
        assert result["messages"]["total_received"] == 4


class TestMetricsTaskEvents:
    """A2AMetrics 任务事件测试."""

    def test_metrics_task_events(self):
        """on_task_created/completed/failed/cancelled/timeout 后 export 验证 tasks 块."""
        metrics = A2AMetrics()

        # 创建 5 个任务
        for _ in range(5):
            metrics.on_task_created("task-1", "agent-A", "agent-B")

        # 完成 2 个
        metrics.on_task_completed()
        metrics.on_task_completed()

        # 失败 1 个
        metrics.on_task_failed()

        # 取消 1 个
        metrics.on_task_cancelled()

        # 超时 1 个
        metrics.on_task_timeout()

        result = metrics.export()
        tasks = result["tasks"]

        assert tasks["created"] == 5
        assert tasks["completed"] == 2
        assert tasks["failed"] == 1
        assert tasks["cancelled"] == 1
        assert tasks["timeout"] == 1

        # 成功率 = completed / (completed+failed+cancelled+timeout) = 2/5 = 0.4
        assert tasks["success_rate"] == 0.4


class TestMetricsSessionEvents:
    """A2AMetrics 会话事件测试."""

    def test_metrics_session_events(self):
        """on_session_created/closed/expired 后验证 sessions 块."""
        metrics = A2AMetrics()

        # 创建 10 个会话
        for _ in range(10):
            metrics.on_session_created()

        # 关闭 3 个
        for _ in range(3):
            metrics.on_session_closed()

        # 过期 2 个
        for _ in range(2):
            metrics.on_session_expired()

        result = metrics.export()
        sessions = result["sessions"]

        assert sessions["created"] == 10
        assert sessions["closed"] == 3
        assert sessions["expired"] == 2
        # active = created - closed - expired = 10 - 3 - 2 = 5
        assert sessions["active"] == 5


class TestMetricsErrorEvents:
    """A2AMetrics 错误事件测试."""

    def test_metrics_error_events(self):
        """on_error 多次后验证 errors 块."""
        metrics = A2AMetrics()

        metrics.on_error("TIMEOUT")
        metrics.on_error("TIMEOUT")
        metrics.on_error("AUTH_FAILED")
        metrics.on_error("AUTH_FAILED")
        metrics.on_error("AUTH_FAILED")
        metrics.on_error("PROTOCOL_ERROR")

        result = metrics.export()
        errors = result["errors"]

        assert errors["TIMEOUT"] == 2
        assert errors["AUTH_FAILED"] == 3
        assert errors["PROTOCOL_ERROR"] == 1


class TestMetricsCapabilityUsage:
    """A2AMetrics 能力使用测试."""

    def test_metrics_capability_usage(self):
        """on_capability_requested 后验证 capabilities 块."""
        metrics = A2AMetrics()

        metrics.on_capability_requested("knowledge_assessment")
        metrics.on_capability_requested("knowledge_assessment")
        metrics.on_capability_requested("adaptive_tutoring")
        metrics.on_capability_requested("knowledge_assessment")

        result = metrics.export()
        caps = result["capabilities"]

        assert caps["knowledge_assessment"] == 3
        assert caps["adaptive_tutoring"] == 1


class TestMetricsLatencyTracking:
    """A2AMetrics 延迟追踪测试."""

    def test_metrics_latency_tracking(self):
        """on_task_completed 传不同 latency_ms，验证 latency_ms 的 avg/max/p99."""
        metrics = A2AMetrics()

        # latency_ms=0 的调用不记录延迟样本
        metrics.on_task_completed(latency_ms=0)

        # 记录不同延迟
        latencies = [50.0, 100.0, 150.0, 200.0, 500.0, 1000.0, 80.0, 120.0]
        for lat in latencies:
            metrics.on_task_completed(latency_ms=lat)

        result = metrics.export()
        lat_info = result["tasks"]["latency_ms"]

        # 8 个样本
        assert lat_info["samples"] == 8
        # avg = (50+100+150+200+500+1000+80+120) / 8 = 275.0
        assert lat_info["avg"] == 275.0
        # max = 1000.0
        assert lat_info["max"] == 1000.0


class TestMetricsSuccessRate:
    """A2AMetrics 成功率测试."""

    def test_metrics_success_rate(self):
        """创建 3 completed + 1 failed，验证 success_rate = 0.75."""
        metrics = A2AMetrics()

        for _ in range(3):
            metrics.on_task_completed()
        metrics.on_task_failed()

        result = metrics.export()
        tasks = result["tasks"]

        assert tasks["completed"] == 3
        assert tasks["failed"] == 1
        # success_rate = 3 / (3+1) = 0.75
        assert tasks["success_rate"] == 0.75

    def test_metrics_success_rate_no_finished(self):
        """无已完成任务时 success_rate = 0.0."""
        metrics = A2AMetrics()
        metrics.on_task_created("t1", "a1", "a2")

        result = metrics.export()
        assert result["tasks"]["success_rate"] == 0.0


class TestMetricsReset:
    """A2AMetrics 重置测试."""

    def test_metrics_reset(self):
        """添加数据后 reset()，export() 验证全部归零."""
        metrics = A2AMetrics()

        # 填充数据
        metrics.on_message_sent("TASK_REQUEST", "a1", "a2")
        metrics.on_message_received("TASK_REQUEST", "a2")
        metrics.on_task_created("t1", "a1", "a2")
        metrics.on_task_completed(latency_ms=100)
        metrics.on_task_failed()
        metrics.on_session_created()
        metrics.on_session_closed()
        metrics.on_error("TIMEOUT")
        metrics.on_capability_requested("knowledge_assessment")

        # 确认有数据
        result_before = metrics.export()
        assert result_before["messages"]["total_sent"] == 1
        assert result_before["tasks"]["created"] == 1

        # 重置
        metrics.reset()
        result_after = metrics.export()

        # 验证全部归零
        assert result_after["messages"]["total_sent"] == 0
        assert result_after["messages"]["total_received"] == 0
        assert result_after["messages"]["by_type"] == {}
        assert result_after["tasks"]["created"] == 0
        assert result_after["tasks"]["completed"] == 0
        assert result_after["tasks"]["failed"] == 0
        assert result_after["tasks"]["cancelled"] == 0
        assert result_after["tasks"]["timeout"] == 0
        assert result_after["tasks"]["success_rate"] == 0.0
        assert result_after["tasks"]["latency_ms"]["samples"] == 0
        assert result_after["sessions"]["created"] == 0
        assert result_after["sessions"]["active"] == 0
        assert result_after["agents"] == {}
        assert result_after["errors"] == {}
        assert result_after["capabilities"] == {}


# ============================================================
# 2. AgentHealthTracker 测试
# ============================================================

class TestAgentHealthUnknown:
    """未知 Agent 健康评分测试."""

    def test_unknown_agent_score(self):
        """未记录的 Agent，health_score 应为 50.0."""
        tracker = AgentHealthTracker()
        score = tracker.health_score("nonexistent-agent")
        assert score == 50.0


class TestAgentHealthPerfect:
    """完美 Agent 健康评分测试."""

    def test_perfect_agent_score(self):
        """记录 10 次 completed + 最近心跳 + 无负载，验证高分 (接近 100)."""
        tracker = AgentHealthTracker()

        agent_id = "perfect-agent"

        # 记录 10 次成功完成，延迟低 (100ms)
        for _ in range(10):
            tracker.record_task_result(agent_id, "completed", latency_ms=100)

        # 记录最近心跳
        tracker.record_heartbeat(agent_id)

        # 确保低负载 (并发 0/5)
        tracker.update_concurrency(agent_id, current=0, max_concurrent=5)

        score = tracker.health_score(agent_id)

        # 成功率 100 分 * 0.35 = 35
        # 延迟 100ms <= 200ms -> 100 分 * 0.25 = 25
        # 稳定性：CV 应该很低（同延迟），100 分 * 0.20 = 20
        # 活跃度：最近心跳 -> 100 分 * 0.10 = 10
        # 负载：0/5 -> 100 分 * 0.10 = 10
        # 总分 = 35 + 25 + 20 + 10 + 10 = 100.0
        assert score >= 95.0


class TestAgentHealthFailed:
    """失败 Agent 健康评分测试."""

    def test_failed_agent_low_score(self):
        """记录 10 次 failed，验证低分."""
        tracker = AgentHealthTracker()

        agent_id = "failing-agent"
        for _ in range(10):
            tracker.record_task_result(agent_id, "failed")

        score = tracker.health_score(agent_id)

        # 成功率 = 0/10 = 0，成功率分 = 0 * 100 = 0 -> 0 * 0.35 = 0
        # 延迟无数据 -> 100 分 -> 25
        # 稳定性无数据 -> 100 分 -> 20
        # 活跃度无心跳 -> 50 分 -> 5
        # 负载默认 -> 100 分 -> 10
        # 总分 = 0 + 25 + 20 + 5 + 10 = 60
        # 但成功率维度为 0，整体应较低
        assert score < 70.0


class TestAgentHealthLatencyPenalty:
    """延迟惩罚测试."""

    def test_latency_penalty(self):
        """记录高延迟 (avg 1500ms) 的 completed 任务，验证延迟维度扣分."""
        tracker = AgentHealthTracker()

        agent_id = "slow-agent"

        # 10 次完成，延迟 1500ms
        for _ in range(10):
            tracker.record_task_result(agent_id, "completed", latency_ms=1500)

        # 心跳 + 低负载
        tracker.record_heartbeat(agent_id)
        tracker.update_concurrency(agent_id, current=0, max_concurrent=5)

        score = tracker.health_score(agent_id)

        # 延迟 1500ms: score_latency = 100 * (1 - (1500-200)/1800) = 100 * (1 - 1300/1800) ≈ 27.78
        # 延迟维度分 ≈ 27.78 * 0.25 ≈ 6.94
        # 即使其他维度满分 (35 + 20 + 10 + 10 = 75)，总分 ≈ 81.94
        assert score < 85.0


class TestAgentHealthLoadPenalty:
    """负载惩罚测试."""

    def test_load_penalty(self):
        """update_concurrency 设置高负载 (4/5)，验证负载维度扣分."""
        tracker = AgentHealthTracker()

        agent_id = "loaded-agent"

        # 完美记录
        for _ in range(10):
            tracker.record_task_result(agent_id, "completed", latency_ms=100)
        tracker.record_heartbeat(agent_id)

        # 设置高负载 4/5
        tracker.update_concurrency(agent_id, current=4, max_concurrent=5)

        score = tracker.health_score(agent_id)

        # 负载 4/5 = 0.8 -> score_load = 100 * (1 - 0.8) = 20
        # 负载维度分 = 20 * 0.10 = 2
        # 其他维度满分: 35 + 25 + 20 + 10 = 90
        # 总分 ≈ 92
        assert score < 95.0


class TestAgentHealthSelectBest:
    """智能路由选择测试."""

    def test_select_best_agent(self):
        """两个 Agent 一个健康一个不健康，select_best_agent 返回健康的."""
        tracker = AgentHealthTracker()

        # 健康的 Agent: 全部成功
        healthy_id = "healthy-agent"
        for _ in range(10):
            tracker.record_task_result(healthy_id, "completed", latency_ms=100)
        tracker.record_heartbeat(healthy_id)
        tracker.update_concurrency(healthy_id, current=0, max_concurrent=5)

        # 不健康的 Agent: 全部失败
        unhealthy_id = "unhealthy-agent"
        for _ in range(10):
            tracker.record_task_result(unhealthy_id, "failed")

        best = tracker.select_best_agent([healthy_id, unhealthy_id])
        assert best == healthy_id

    def test_select_best_empty(self):
        """空列表返回 None."""
        tracker = AgentHealthTracker()
        best = tracker.select_best_agent([])
        assert best is None

    def test_select_best_single(self):
        """单个候选 Agent 返回该 Agent."""
        tracker = AgentHealthTracker()
        best = tracker.select_best_agent(["only-agent"])
        assert best == "only-agent"


class TestAgentHealthReport:
    """健康报告测试."""

    def test_health_report(self):
        """验证 health_report 包含 agent_id, health_score, success_rate 等字段."""
        tracker = AgentHealthTracker()

        agent_id = "report-agent"
        tracker.record_task_result(agent_id, "completed", latency_ms=100)
        tracker.record_task_result(agent_id, "failed")
        tracker.record_task_result(agent_id, "completed", latency_ms=200)

        report = tracker.health_report(agent_id)

        assert report["agent_id"] == agent_id
        assert "health_score" in report
        assert "success_rate" in report
        assert "avg_latency_ms" in report
        assert "latency_cv" in report
        assert "load_ratio" in report
        assert "total_tasks" in report

        # success_rate = 2/3 ≈ 0.6667（to_dict 四舍五入到 4 位小数）
        assert report["success_rate"] == round(2 / 3, 4)

    def test_health_report_unknown_agent(self):
        """未知 Agent 的报告只包含 agent_id 和 health_score."""
        tracker = AgentHealthTracker()
        report = tracker.health_report("ghost")

        assert report["agent_id"] == "ghost"
        assert report["health_score"] == 50.0


class TestAgentHealthExport:
    """AgentHealthTracker 导出测试."""

    def test_export(self):
        """验证 export 包含 total_agents, health_scores, reports."""
        tracker = AgentHealthTracker()

        tracker.record_task_result("agent-A", "completed", latency_ms=100)
        tracker.record_task_result("agent-B", "failed")

        export = tracker.export()

        assert export["total_agents"] == 2
        assert "agent-A" in export["health_scores"]
        assert "agent-B" in export["health_scores"]
        assert isinstance(export["reports"], list)
        assert len(export["reports"]) == 2

        # 报告应包含 health_score
        for report in export["reports"]:
            assert "health_score" in report


class TestAgentHealthReset:
    """AgentHealthTracker 重置测试."""

    def test_reset(self):
        """记录数据后 reset，验证清空."""
        tracker = AgentHealthTracker()

        tracker.record_task_result("agent-A", "completed", latency_ms=100)
        tracker.record_task_result("agent-A", "failed")
        tracker.record_heartbeat("agent-A")
        tracker.update_concurrency("agent-A", current=2, max_concurrent=5)

        # 确认有数据
        assert tracker.health_score("agent-A") != 50.0

        # 重置
        tracker.reset()

        # 重置后应返回未知 Agent 的默认分
        assert tracker.health_score("agent-A") == 50.0

        # 导出应为空
        export = tracker.export()
        assert export["total_agents"] == 0
        assert export["health_scores"] == {}
        assert export["reports"] == []


# ============================================================
# 3. TokenStore + 指纹测试
# ============================================================

class TestTokenGenerateValidate:
    """Token 生成与验证测试."""

    def test_token_generate_validate(self):
        """生成 token，validate 返回非 None."""
        store = TokenStore(secret_key="test-secret")
        token = store.generate(agent_id="agent-001", session_id="sess-001")

        # token 应为非空字符串
        assert isinstance(token, str)
        assert len(token) > 0

        info = store.validate(token)
        assert info is not None
        assert info["agent_id"] == "agent-001"
        assert info["session_id"] == "sess-001"

    def test_token_expired(self):
        """生成 ttl_seconds=0.01 的 token，sleep 0.02，validate 返回 None."""
        store = TokenStore(secret_key="test-secret")
        token = store.generate(agent_id="agent-001", ttl_seconds=0.01)

        # 等待 token 过期
        time.sleep(0.02)

        info = store.validate(token)
        assert info is None

    def test_token_revoke(self):
        """生成后 revoke，validate 返回 None."""
        store = TokenStore(secret_key="test-secret")
        token = store.generate(agent_id="agent-001")

        # 验证生效
        assert store.validate(token) is not None

        # 吊销
        revoked = store.revoke(token)
        assert revoked is True

        # 吊销后验证应返回 None
        assert store.validate(token) is None

    def test_token_revoke_nonexistent(self):
        """吊销不存在的 token 返回 False."""
        store = TokenStore(secret_key="test-secret")
        revoked = store.revoke("nonexistent-token")
        assert revoked is False

    def test_token_invalid(self):
        """validate 随机字符串返回 None."""
        store = TokenStore(secret_key="test-secret")
        info = store.validate("random-invalid-token-string")
        assert info is None

    def test_token_multiple_tokens(self):
        """生成多个 token，各自独立验证."""
        store = TokenStore(secret_key="test-secret")

        token_a = store.generate(agent_id="agent-A", session_id="sess-A")
        token_b = store.generate(agent_id="agent-B", session_id="sess-B")

        info_a = store.validate(token_a)
        assert info_a["agent_id"] == "agent-A"

        info_b = store.validate(token_b)
        assert info_b["agent_id"] == "agent-B"


class TestTokenCleanup:
    """Token 过期清理测试."""

    def test_token_cleanup(self):
        """生成多个过期 token，cleanup_expired 返回正确数量."""
        store = TokenStore(secret_key="test-secret")

        # 生成 3 个短命 token
        token_1 = store.generate(agent_id="agent-1", ttl_seconds=0.01)
        token_2 = store.generate(agent_id="agent-2", ttl_seconds=0.01)
        token_3 = store.generate(agent_id="agent-3", ttl_seconds=0.01)

        # 生成 2 个长命 token
        token_4 = store.generate(agent_id="agent-4", ttl_seconds=3600)
        token_5 = store.generate(agent_id="agent-5", ttl_seconds=3600)

        # 等待短命 token 过期
        time.sleep(0.02)

        # 清理
        cleaned = store.cleanup_expired()

        # 应清理 3 个
        assert cleaned == 3

        # 长命 token 仍然有效
        assert store.validate(token_4) is not None
        assert store.validate(token_5) is not None

        # 过期 token 已被移除
        assert store.validate(token_1) is None
        assert store.validate(token_2) is None
        assert store.validate(token_3) is None

        # 存储中剩余 2 个
        assert store.size == 2


class TestFingerprintDeterministic:
    """指纹确定性测试."""

    def test_fingerprint_deterministic(self):
        """相同输入两次调用 agent_fingerprint 结果一致."""
        fp1 = agent_fingerprint("agent-001", ["capability-A", "capability-B"], secret="my-secret")
        fp2 = agent_fingerprint("agent-001", ["capability-B", "capability-A"], secret="my-secret")
        # capabilities 会被排序，顺序不影响结果
        assert fp1 == fp2
        assert len(fp1) == 16  # 16 字符十六进制

    def test_fingerprint_different_secret(self):
        """相同输入不同 secret 结果不同."""
        fp_a = agent_fingerprint("agent-001", ["cap-A"], secret="secret-A")
        fp_b = agent_fingerprint("agent-001", ["cap-A"], secret="secret-B")
        assert fp_a != fp_b

    def test_fingerprint_different_inputs(self):
        """不同输入产生不同指纹."""
        fp_1 = agent_fingerprint("agent-001", ["cap-A"], secret="secret")
        fp_2 = agent_fingerprint("agent-002", ["cap-A"], secret="secret")
        fp_3 = agent_fingerprint("agent-001", ["cap-B"], secret="secret")

        assert fp_1 != fp_2  # 不同 agent_id
        assert fp_1 != fp_3  # 不同 capabilities

    def test_fingerprint_default_secret(self):
        """不传 secret 时也能正常生成指纹."""
        fp = agent_fingerprint("agent-001", ["cap-A"])
        assert isinstance(fp, str)
        assert len(fp) == 16


class TestVerifyFingerprint:
    """指纹验证测试."""

    def test_verify_fingerprint_match(self):
        """verify_fingerprint 正确返回 True."""
        fp = agent_fingerprint("agent-001", ["cap-A", "cap-B"], secret="secret")
        result = verify_fingerprint("agent-001", ["cap-A", "cap-B"], fp, secret="secret")
        assert result is True

    def test_verify_fingerprint_mismatch(self):
        """修改 fingerprint 返回 False."""
        fp = agent_fingerprint("agent-001", ["cap-A"], secret="secret")
        # 篡改指纹
        tampered = "a" * 16
        result = verify_fingerprint("agent-001", ["cap-A"], tampered, secret="secret")
        assert result is False

    def test_verify_fingerprint_wrong_agent(self):
        """使用错误 agent_id 验证返回 False."""
        fp = agent_fingerprint("agent-001", ["cap-A"], secret="secret")
        result = verify_fingerprint("agent-002", ["cap-A"], fp, secret="secret")
        assert result is False

    def test_verify_fingerprint_wrong_secret(self):
        """使用错误 secret 验证返回 False."""
        fp = agent_fingerprint("agent-001", ["cap-A"], secret="secret-A")
        result = verify_fingerprint("agent-001", ["cap-A"], fp, secret="secret-B")
        assert result is False
