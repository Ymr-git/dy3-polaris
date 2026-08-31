"""CC4 三横切集成 — 完整测试.

覆盖 CC4 治理闭环系统的全部模块:
1. 数据模型 (models) — 枚举/桥接事件/治理上下文/反馈信号/健康检查/断路器配置
2. 异常体系 (exceptions) — CC4Error 层级 + JSON-RPC 错误码
3. 断路器 (CircuitBreaker) — 三态模型/失败计数/自动恢复/事件日志
4. CC1→CC2 桥接器 — 评审结果注入路由决策/审批创建/置信度转换
5. CC1→CC3 桥接器 — 评审结果标注到 KPA 校验维度
6. CC2→CC3 桥接器 — 审批记录写入 KPA 决策维度
7. 反馈飞轮 (FeedbackLoop) — 溯源完整性评估/信号生成/动作执行
8. 统一网关 (UnifiedGateway) — 治理闭环编排/指标采集/事件记录
9. 健康聚合器 (HealthAggregator) — 三模块探针/告警生成/总体状态
10. REST API (CC4APIRouter) — 22个端点全覆盖

测试领域: Dy3+ 发光材料 (YAG 基质, 4f-4f 跃迁, 480/574/660nm 发射)
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from dy3_polaris.l0.cc_integration import (
    # 枚举
    BridgeDirection,
    GovernancePhase,
    FeedbackSignalType,
    AlertSeverity,
    HealthStatus,
    CircuitState,
    # 数据模型
    BridgeEvent,
    GovernanceContext,
    GovernanceDecision,
    FeedbackSignal,
    FeedbackAction,
    HealthCheck,
    SystemHealthReport,
    CircuitBreakerConfig,
    GovernanceMetrics,
    # 异常
    CC4Error,
    BridgeConnectionError,
    FeedbackLoopError,
    GatewayRoutingError,
    HealthCheckError,
    CircuitBreakerOpenError,
    GovernancePolicyError,
    # 核心组件
    CircuitBreaker,
    CC1CC2Bridge,
    CC1CC3Bridge,
    CC2CC3Bridge,
    FeedbackLoop,
    UnifiedGateway,
    HealthAggregator,
    # REST API
    CC4APIRouter,
)


# ============================================================
# 辅助工厂
# ============================================================


def _make_review_result(
    verdict: str = "pass",
    score: float = 85.0,
) -> Any:
    """创建模拟 CC1 ReviewResult."""
    from dataclasses import asdict

    from dy3_polaris.l0.cc1.review_pipeline import ReviewResult
    from dy3_polaris.l0.cc1.layers import ReviewLayerType
    from dy3_polaris.l0.cc1.state_machine import ReviewVerdict

    verdict_map = {
        "pass": ReviewVerdict.PASS,
        "flag": ReviewVerdict.FLAG,
        "block": ReviewVerdict.BLOCK,
    }
    layer_scores = {
        ReviewLayerType.L1_FACT: 85.0,
        ReviewLayerType.L2_LOGIC: 82.0,
        ReviewLayerType.L3_NUMERICAL: 88.0,
        ReviewLayerType.L4_PROVENANCE: 80.0,
    }
    rr = ReviewResult(
        verdict=verdict_map.get(verdict, ReviewVerdict.PASS),
        composite_score=score,
        layer_scores=layer_scores,
        issues=[],
        metadata={"source": "test"},
    )
    # 为 dataclass 添加 model_dump 兼容方法
    if not hasattr(rr, "model_dump"):
        def _model_dump(mode: str = "python"):
            d = asdict(rr)
            # 枚举转字符串
            if hasattr(d.get("verdict"), "value"):
                d["verdict"] = d["verdict"].value
            new_scores = {}
            for k, v in d.get("layer_scores", {}).items():
                key = k.value if hasattr(k, "value") else str(k)
                new_scores[key] = v
            d["layer_scores"] = new_scores
            return d
        rr.model_dump = _model_dump  # type: ignore[attr-defined]
    return rr


# ============================================================
# 1. 数据模型测试
# ============================================================


class TestModels:
    """数据模型测试."""

    def test_bridge_direction_六向数据流(self):
        """测试桥接方向枚举包含六个方向."""
        assert len(BridgeDirection) == 6
        assert BridgeDirection.CC1_TO_CC2.value == "cc1_to_cc2"
        assert BridgeDirection.CC3_TO_CC1.value == "cc3_to_cc1"

    def test_governance_phase_四阶段(self):
        """测试治理闭环四阶段."""
        assert len(GovernancePhase) == 4
        assert GovernancePhase.RECONCILE.value == "reconcile"
        assert GovernancePhase.VERIFY.value == "verify"

    def test_feedback_signal_type_七种信号(self):
        """测试反馈信号类型包含七种."""
        assert len(FeedbackSignalType) == 7
        assert (
            FeedbackSignalType.PROVENANCE_COMPLETENESS.value
            == "provenance_completeness"
        )

    def test_alert_severity_四级(self):
        """测试告警严重级别."""
        assert len(AlertSeverity) == 4
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_health_status_四态(self):
        """测试健康状态四态."""
        assert len(HealthStatus) == 4
        assert HealthStatus.HEALTHY.value == "healthy"

    def test_circuit_state_三态(self):
        """测试断路器三态模型."""
        assert len(CircuitState) == 3
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_bridge_event_默认值(self):
        """测试桥接事件默认值生成."""
        event = BridgeEvent(
            source="cc1",
            target="cc2",
            direction=BridgeDirection.CC1_TO_CC2,
        )
        assert event.event_id.startswith("be-")
        assert event.source == "cc1"
        assert event.target == "cc2"
        assert event.direction == BridgeDirection.CC1_TO_CC2
        assert event.payload == {}
        assert event.timestamp > 0

    def test_governance_context_默认值(self):
        """测试治理上下文默认值."""
        ctx = GovernanceContext()
        assert ctx.context_id.startswith("gc-")
        assert ctx.phase == GovernancePhase.RECONCILE
        assert ctx.cc1_verdict == ""
        assert ctx.cc1_score == 0.0

    def test_governance_decision_默认值(self):
        """测试治理决策默认值."""
        decision = GovernanceDecision()
        assert decision.decision_id.startswith("gd-")
        assert decision.phase == GovernancePhase.ACT
        assert decision.affected_modules == []

    def test_feedback_signal_默认值(self):
        """测试反馈信号默认值."""
        signal = FeedbackSignal()
        assert signal.signal_id.startswith("fs-")
        assert signal.source == "cc3"
        assert signal.target == "cc1"
        assert signal.triggered is False

    def test_feedback_action_默认值(self):
        """测试反馈动作默认值."""
        action = FeedbackAction()
        assert action.action_id.startswith("fa-")
        assert action.executed is False
        assert action.result == {}

    def test_health_check_默认值(self):
        """测试健康检查默认值."""
        check = HealthCheck(module="cc1")
        assert check.module == "cc1"
        assert check.status == HealthStatus.UNKNOWN
        assert check.latency_ms == 0.0

    def test_system_health_report_默认值(self):
        """测试系统健康报告默认值."""
        report = SystemHealthReport()
        assert report.overall_status == HealthStatus.UNKNOWN
        assert report.modules == {}
        assert report.active_alerts == []

    def test_circuit_breaker_config_默认值(self):
        """测试断路器配置默认值."""
        config = CircuitBreakerConfig(module="test")
        assert config.module == "test"
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 3

    def test_governance_metrics_默认值(self):
        """测试治理指标默认值."""
        metrics = GovernanceMetrics()
        assert metrics.total_bridges == 0
        assert metrics.bridge_success_rate == 0.0
        assert metrics.collected_at > 0


# ============================================================
# 2. 异常体系测试
# ============================================================


class TestExceptions:
    """异常体系测试."""

    def test_cc4_error_基础异常(self):
        """测试 CC4Error 基础异常."""
        exc = CC4Error("测试错误")
        assert "测试错误" in str(exc)
        assert exc._jsonrpc_code() == -32400

    def test_bridge_connection_error(self):
        """测试桥接连接错误."""
        exc = BridgeConnectionError("cc1", "cc2", "超时")
        assert "cc1" in str(exc)
        assert "cc2" in str(exc)
        assert exc._jsonrpc_code() == -32401

    def test_feedback_loop_error(self):
        """测试反馈飞轮错误."""
        exc = FeedbackLoopError("provenance_completeness", "阈值超限")
        assert "provenance_completeness" in str(exc)
        assert exc._jsonrpc_code() == -32402

    def test_gateway_routing_error(self):
        """测试网关路由错误."""
        exc = GatewayRoutingError("/cc4/gateway/govern", "无效参数")
        assert "/cc4/gateway/govern" in str(exc)
        assert exc._jsonrpc_code() == -32403

    def test_health_check_error(self):
        """测试健康检查错误."""
        exc = HealthCheckError("cc1", "连接超时")
        assert "cc1" in str(exc)
        assert exc._jsonrpc_code() == -32404

    def test_circuit_breaker_open_error(self):
        """测试断路器开启错误."""
        exc = CircuitBreakerOpenError("cc2", 30.0)
        assert "cc2" in str(exc)
        assert exc._jsonrpc_code() == -32405

    def test_governance_policy_error(self):
        """测试治理策略错误."""
        exc = GovernancePolicyError("max_retry", "超过最大重试次数")
        assert "max_retry" in str(exc)
        assert exc._jsonrpc_code() == -32406

    def test_异常继承链(self):
        """测试所有 CC4 异常都继承 CC4Error."""
        assert issubclass(BridgeConnectionError, CC4Error)
        assert issubclass(FeedbackLoopError, CC4Error)
        assert issubclass(GatewayRoutingError, CC4Error)
        assert issubclass(HealthCheckError, CC4Error)
        assert issubclass(CircuitBreakerOpenError, CC4Error)
        assert issubclass(GovernancePolicyError, CC4Error)


# ============================================================
# 3. 断路器测试
# ============================================================


class TestCircuitBreaker:
    """断路器测试."""

    def test_初始状态为关闭(self):
        """测试断路器初始状态为 CLOSED."""
        cb = CircuitBreaker("test_module")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_open is False

    def test_正常调用通过(self):
        """测试正常调用通过断路器."""
        cb = CircuitBreaker("test_module")
        result = cb.call(lambda x: x * 2, 21)
        assert result == 42

    def test_连续失败触发跳闸(self):
        """测试连续失败达到阈值后断路器跳闸."""
        cb = CircuitBreaker(
            "test_module",
            CircuitBreakerConfig(
                module="test_module",
                failure_threshold=3,
                recovery_timeout=1.0,
            ),
        )

        def fail():
            raise ValueError("失败")

        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(fail)

        assert cb.state == CircuitState.OPEN
        assert cb.is_open is True

    def test_跳闸后拒绝调用(self):
        """测试断路器跳闸后拒绝调用并抛出 CircuitBreakerOpenError."""
        cb = CircuitBreaker(
            "test_module",
            CircuitBreakerConfig(
                module="test_module",
                failure_threshold=2,
                recovery_timeout=10.0,
            ),
        )

        def fail():
            raise RuntimeError("失败")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(fail)

        assert cb.is_open is True
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: "不应执行")

    def test_恢复超时后进入半开状态(self):
        """测试恢复超时后断路器进入 HALF_OPEN 状态."""
        cb = CircuitBreaker(
            "test_module",
            CircuitBreakerConfig(
                module="test_module",
                failure_threshold=1,
                recovery_timeout=0.1,
                success_threshold=1,
            ),
        )

        def fail():
            raise ValueError("失败")

        with pytest.raises(ValueError):
            cb.call(fail)
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_半开状态成功后关闭(self):
        """测试半开状态成功达到阈值后断路器关闭."""
        cb = CircuitBreaker(
            "test_module",
            CircuitBreakerConfig(
                module="test_module",
                failure_threshold=1,
                recovery_timeout=0.1,
                success_threshold=1,
            ),
        )

        def fail():
            raise ValueError("失败")

        with pytest.raises(ValueError):
            cb.call(fail)

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        result = cb.call(lambda: "成功")
        assert result == "成功"
        assert cb.state == CircuitState.CLOSED

    def test_重置恢复初始状态(self):
        """测试重置后断路器恢复初始状态."""
        cb = CircuitBreaker(
            "test_module",
            CircuitBreakerConfig(
                module="test_module",
                failure_threshold=1,
            ),
        )

        def fail():
            raise ValueError("失败")

        with pytest.raises(ValueError):
            cb.call(fail)
        assert cb.is_open is True

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_open is False

    def test_get_status_返回状态字典(self):
        """测试 get_status 返回状态字典."""
        cb = CircuitBreaker("test_module")
        status = cb.get_status()
        assert isinstance(status, dict)
        assert status["state"] == "closed"
        assert "failure_count" in status
        assert "success_count" in status
        assert "total_trips" in status

    def test_get_events_返回事件日志(self):
        """测试 get_events 返回事件日志 (仅状态转换记录事件)."""
        cb = CircuitBreaker(
            "test_module",
            CircuitBreakerConfig(
                module="test_module",
                failure_threshold=1,
                recovery_timeout=0.1,
                success_threshold=1,
            ),
        )
        # 触发一次跳闸以产生事件
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        events = cb.get_events(limit=10)
        assert isinstance(events, list)
        assert len(events) > 0


# ============================================================
# 4. CC1→CC2 桥接器测试
# ============================================================


class TestCC1CC2Bridge:
    """CC1→CC2 桥接器测试."""

    def test_桥接PASS评审结果(self):
        """测试桥接 PASS 评审结果到 CC2 路由."""
        bridge = CC1CC2Bridge()
        review = _make_review_result("pass", 90.0)
        result = bridge.bridge(
            review_result=review,
            operation_type="content_generation",
            user_id="student-001",
            session_id="sess-001",
            trace_id="trace-001",
        )
        assert result["success"] is True
        assert "routing_result" in result
        assert result["routing_result"] is not None

    def test_桥接FLAG评审结果(self):
        """测试桥接 FLAG 评审结果."""
        bridge = CC1CC2Bridge()
        review = _make_review_result("flag", 60.0)
        result = bridge.bridge(review_result=review)
        assert result["success"] is True

    def test_桥接BLOCK评审结果(self):
        """测试桥接 BLOCK 评审结果."""
        bridge = CC1CC2Bridge()
        review = _make_review_result("block", 30.0)
        result = bridge.bridge(review_result=review)
        assert result["success"] is True

    def test_桥接统计(self):
        """测试桥接器统计."""
        bridge = CC1CC2Bridge()
        review = _make_review_result("pass", 85.0)
        bridge.bridge(review_result=review)
        stats = bridge.get_statistics()
        assert isinstance(stats, dict)
        # 兼容 total 和 total_bridges 两种 key
        total = stats.get("total", stats.get("total_bridges", 0))
        assert total >= 1

    def test_桥接事件列表(self):
        """测试桥接器事件列表."""
        bridge = CC1CC2Bridge()
        review = _make_review_result("pass", 85.0)
        bridge.bridge(review_result=review)
        events = bridge.get_events(limit=10)
        assert isinstance(events, list)
        assert len(events) >= 1

    def test_桥接器重置(self):
        """测试桥接器重置."""
        bridge = CC1CC2Bridge()
        review = _make_review_result("pass", 85.0)
        bridge.bridge(review_result=review)
        bridge.reset()
        stats = bridge.get_statistics()
        total = stats.get("total", stats.get("total_bridges", 0))
        assert total == 0


# ============================================================
# 5. CC1→CC3 桥接器测试
# ============================================================


class TestCC1CC3Bridge:
    """CC1→CC3 桥接器测试."""

    def test_桥接评审结果到KPA校验维度(self):
        """测试评审结果标注到 KPA 校验维度."""
        bridge = CC1CC3Bridge()
        review = _make_review_result("pass", 85.0)
        result = bridge.bridge(
            review_result=review,
            target_id="kp-dy3-yag-4f",
            trace_id="trace-001",
            session_id="sess-001",
        )
        assert isinstance(result, dict)
        assert "success" in result

    def test_桥接统计(self):
        """测试 CC1→CC3 桥接器统计."""
        bridge = CC1CC3Bridge()
        stats = bridge.get_statistics()
        assert isinstance(stats, dict)

    def test_桥接事件列表(self):
        """测试 CC1→CC3 桥接器事件列表."""
        bridge = CC1CC3Bridge()
        events = bridge.get_events(limit=10)
        assert isinstance(events, list)

    def test_桥接器重置(self):
        """测试 CC1→CC3 桥接器重置."""
        bridge = CC1CC3Bridge()
        bridge.reset()
        stats = bridge.get_statistics()
        assert stats.get("total", 0) == 0


# ============================================================
# 6. CC2→CC3 桥接器测试
# ============================================================


class TestCC2CC3Bridge:
    """CC2→CC3 桥接器测试."""

    def test_桥接统计(self):
        """测试 CC2→CC3 桥接器统计."""
        bridge = CC2CC3Bridge()
        stats = bridge.get_statistics()
        assert isinstance(stats, dict)

    def test_桥接事件列表(self):
        """测试 CC2→CC3 桥接器事件列表."""
        bridge = CC2CC3Bridge()
        events = bridge.get_events(limit=10)
        assert isinstance(events, list)

    def test_桥接器重置(self):
        """测试 CC2→CC3 桥接器重置."""
        bridge = CC2CC3Bridge()
        bridge.reset()
        stats = bridge.get_statistics()
        assert stats.get("total", 0) == 0


# ============================================================
# 7. 反馈飞轮测试
# ============================================================


class TestFeedbackLoop:
    """反馈飞轮测试."""

    def test_反馈飞轮初始化(self):
        """测试反馈飞轮初始化."""
        loop = FeedbackLoop()
        assert loop is not None

    def test_反馈统计(self):
        """测试反馈飞轮统计."""
        loop = FeedbackLoop()
        stats = loop.get_statistics()
        assert isinstance(stats, dict)

    def test_反馈事件列表(self):
        """测试反馈飞轮事件列表."""
        loop = FeedbackLoop()
        events = loop.get_events(limit=10)
        assert isinstance(events, list)

    def test_反馈飞轮重置(self):
        """测试反馈飞轮重置."""
        loop = FeedbackLoop()
        loop.reset()
        stats = loop.get_statistics()
        assert isinstance(stats, dict)

    def test_评审质量仅使用真实_cc1_统计(self):
        """没有真实样本时不得由溯源完整度伪造通过率."""
        prov = {"completeness_score": 0.9, "chain_verified": True}
        escalation = {"needs_escalation": False}

        unknown = FeedbackLoop().generate_signals(
            "kpa-unknown", prov, escalation, trace_id="trace-real-cc1"
        )
        assert not any(
            signal.signal_type == FeedbackSignalType.REVIEW_QUALITY
            for signal in unknown
        )

        loop = FeedbackLoop(
            cc1_statistics_provider=lambda: {
                "total": 10,
                "pass": 6,
                "pass_rate": 60.0,
            }
        )
        signals = loop.generate_signals(
            "kpa-real", prov, escalation, trace_id="trace-real-cc1"
        )
        review_signal = next(
            signal
            for signal in signals
            if signal.signal_type == FeedbackSignalType.REVIEW_QUALITY
        )
        assert review_signal.value == 0.6
        assert review_signal.trace_id == "trace-real-cc1"
        assert "mock" not in review_signal.message.lower()

    def test_治理检查故障不会触发阈值动作(self):
        """CC3 工具故障是 unknown，不是低完整度。"""

        class _BrokenIntegration:
            def check_provenance_for_cc1(self, annotation_id):
                raise RuntimeError("provenance unavailable")

            def check_escalation_for_cc2(self, annotation_id):
                raise RuntimeError("escalation unavailable")

        result = FeedbackLoop(cc_integration=_BrokenIntegration()).evaluate(
            "kpa-broken",
            trace_id="trace-broken",
            session_id="session-broken",
        )
        assert result["success"] is False
        assert result["signals"] == []
        assert result["actions"] == []
        assert result["trace_id"] == "trace-broken"


# ============================================================
# 8. 统一网关测试
# ============================================================


class TestUnifiedGateway:
    """统一网关测试."""

    def test_网关初始化(self):
        """测试统一网关初始化."""
        gw = UnifiedGateway()
        assert gw.cc1_cc2_bridge is not None
        assert gw.cc1_cc3_bridge is not None
        assert gw.cc2_cc3_bridge is not None
        assert gw.circuit_breakers is not None

    def test_网关治理无评审结果(self):
        """测试网关治理 — 无评审结果时返回失败."""
        gw = UnifiedGateway()
        result = gw.govern(review_result=None)
        assert result["success"] is False
        assert "未提供" in result["error"]

    def test_网关治理完整闭环(self):
        """测试网关治理完整闭环 — CC1→CC2→CC3→反馈."""
        gw = UnifiedGateway()
        review = _make_review_result("pass", 90.0)
        result = gw.govern(
            review_result=review,
            operation_type="content_generation",
            user_id="student-001",
            session_id="sess-001",
            trace_id="trace-001",
        )
        assert isinstance(result, dict)
        assert "success" in result
        assert result["trace_id"] == "trace-001"
        assert result["session_id"] == "sess-001"
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    def test_网关统计(self):
        """测试网关统计."""
        gw = UnifiedGateway()
        review = _make_review_result("pass", 85.0)
        gw.govern(review_result=review)
        stats = gw.get_statistics()
        assert isinstance(stats, dict)
        assert stats["total_governances"] >= 1

    def test_网关治理指标(self):
        """测试网关治理指标."""
        gw = UnifiedGateway()
        review = _make_review_result("pass", 85.0)
        gw.govern(review_result=review)
        metrics = gw.get_governance_metrics()
        assert isinstance(metrics, GovernanceMetrics)
        assert metrics.total_bridges >= 0

    def test_网关事件列表(self):
        """测试网关事件列表."""
        gw = UnifiedGateway()
        review = _make_review_result("pass", 85.0)
        gw.govern(review_result=review)
        events = gw.get_events(limit=10)
        assert isinstance(events, list)
        assert len(events) >= 1

    def test_网关重置(self):
        """测试网关重置."""
        gw = UnifiedGateway()
        review = _make_review_result("pass", 85.0)
        gw.govern(review_result=review)
        gw.reset()
        stats = gw.get_statistics()
        assert stats["total_governances"] == 0


# ============================================================
# 9. 健康聚合器测试
# ============================================================


class TestHealthAggregator:
    """健康聚合器测试."""

    def test_聚合器初始化(self):
        """测试健康聚合器初始化."""
        agg = HealthAggregator()
        assert agg.circuit_breakers is not None
        assert agg.last_report is None

    def test_健康检查返回报告(self):
        """测试健康检查返回 SystemHealthReport."""
        agg = HealthAggregator()
        report = agg.check_health()
        assert isinstance(report, SystemHealthReport)
        assert report.overall_status in HealthStatus
        assert "cc1" in report.modules
        assert "cc2" in report.modules
        assert "cc3" in report.modules

    def test_未配置模块返回UNKNOWN(self):
        """测试未配置模块返回 UNKNOWN 状态."""
        agg = HealthAggregator()
        report = agg.check_health()
        assert report.modules["cc1"].status == HealthStatus.UNKNOWN
        assert report.modules["cc2"].status == HealthStatus.UNKNOWN
        assert report.modules["cc3"].status == HealthStatus.UNKNOWN

    def test_聚合指标(self):
        """测试聚合指标."""
        agg = HealthAggregator()
        metrics = agg.get_metrics()
        assert isinstance(metrics, dict)
        assert "modules" in metrics
        assert "circuit_breakers" in metrics
        assert "collected_at" in metrics

    def test_全UNKNOWN总体为UNKNOWN(self):
        """测试全部模块 UNKNOWN 时总体状态为 UNKNOWN."""
        agg = HealthAggregator()
        report = agg.check_health()
        assert report.overall_status == HealthStatus.UNKNOWN


# ============================================================
# 10. REST API 测试
# ============================================================


class TestCC4APIRouter:
    """REST API 路由器测试."""

    def test_API路由器初始化(self):
        """测试 API 路由器初始化."""
        router = CC4APIRouter()
        assert router is not None
        assert router._gateway is not None
        assert router._cc1_cc2_bridge is not None
        assert router._health_aggregator is not None

    def test_API_govern_无评审结果(self):
        """测试 govern 端点 — 无评审结果."""
        router = CC4APIRouter()
        result = router.govern({})
        assert result["code"] == 200
        data = result["data"]
        assert data["success"] is False

    def test_API_govern_完整闭环(self):
        """测试 govern 端点 — 完整治理闭环."""
        router = CC4APIRouter()
        review = _make_review_result("pass", 90.0)
        result = router.govern({
            "review_result": review.model_dump(),
            "operation_type": "content_generation",
            "user_id": "student-001",
            "session_id": "sess-001",
            "trace_id": "trace-001",
        })
        assert result["code"] == 200
        data = result["data"]
        assert "success" in data
        assert data["trace_id"] == "trace-001"

    def test_API_gateway_statistics(self):
        """测试网关统计端点."""
        router = CC4APIRouter()
        result = router.gateway_statistics()
        assert result["code"] == 200
        assert "total_governances" in result["data"]

    def test_API_gateway_metrics(self):
        """测试治理指标端点."""
        router = CC4APIRouter()
        result = router.gateway_metrics()
        assert result["code"] == 200
        assert "total_bridges" in result["data"]

    def test_API_gateway_events(self):
        """测试治理事件端点."""
        router = CC4APIRouter()
        result = router.gateway_events(limit=5)
        assert result["code"] == 200
        assert isinstance(result["data"], list)

    def test_API_gateway_reset(self):
        """测试网关重置端点."""
        router = CC4APIRouter()
        result = router.gateway_reset()
        assert result["code"] == 200
        assert result["data"]["reset"] is True

    def test_API_bridge_cc1_cc2_无评审结果(self):
        """测试 CC1→CC2 桥接端点 — 无评审结果."""
        router = CC4APIRouter()
        result = router.bridge_cc1_cc2({})
        assert result["code"] == 400

    def test_API_bridge_cc1_cc2_有评审结果(self):
        """测试 CC1→CC2 桥接端点 — 有评审结果."""
        router = CC4APIRouter()
        review = _make_review_result("pass", 90.0)
        result = router.bridge_cc1_cc2({
            "review_result": review.model_dump(),
            "operation_type": "content_generation",
        })
        assert result["code"] == 200

    def test_API_bridge_cc1_cc3_有评审结果(self):
        """测试 CC1→CC3 桥接端点."""
        router = CC4APIRouter()
        review = _make_review_result("pass", 90.0)
        result = router.bridge_cc1_cc3({
            "review_result": review.model_dump(),
            "target_id": "kp-dy3-yag-4f",
        })
        assert result["code"] == 200

    def test_API_bridge_cc2_cc3_无审批记录(self):
        """测试 CC2→CC3 桥接端点 — 无审批记录."""
        router = CC4APIRouter()
        result = router.bridge_cc2_cc3({})
        assert result["code"] == 400

    def test_API_bridge_statistics_存在(self):
        """测试桥接器统计端点 — 存在的桥接器."""
        router = CC4APIRouter()
        result = router.bridge_statistics("cc1-cc2")
        assert result["code"] == 200

    def test_API_bridge_statistics_不存在(self):
        """测试桥接器统计端点 — 不存在的桥接器."""
        router = CC4APIRouter()
        result = router.bridge_statistics("invalid")
        assert result["code"] == 404

    def test_API_bridge_events_存在(self):
        """测试桥接器事件端点."""
        router = CC4APIRouter()
        result = router.bridge_events("cc1-cc2", limit=5)
        assert result["code"] == 200

    def test_API_bridge_reset_存在(self):
        """测试桥接器重置端点."""
        router = CC4APIRouter()
        result = router.bridge_reset("cc1-cc2")
        assert result["code"] == 200

    def test_API_feedback_evaluate_无标注ID(self):
        """测试反馈评估端点 — 无标注 ID."""
        router = CC4APIRouter()
        result = router.feedback_evaluate({})
        assert result["code"] == 400

    def test_API_feedback_statistics(self):
        """测试反馈统计端点."""
        router = CC4APIRouter()
        result = router.feedback_statistics()
        assert result["code"] == 200

    def test_API_feedback_events(self):
        """测试反馈事件端点."""
        router = CC4APIRouter()
        result = router.feedback_events(limit=5)
        assert result["code"] == 200

    def test_API_feedback_reset(self):
        """测试反馈重置端点."""
        router = CC4APIRouter()
        result = router.feedback_reset()
        assert result["code"] == 200

    def test_API_health(self):
        """测试健康检查端点."""
        router = CC4APIRouter()
        result = router.health()
        assert result["code"] == 200
        assert "overall_status" in result["data"]

    def test_API_health_metrics(self):
        """测试聚合指标端点."""
        router = CC4APIRouter()
        result = router.health_metrics()
        assert result["code"] == 200
        assert "modules" in result["data"]

    def test_API_circuit_list(self):
        """测试断路器列表端点."""
        router = CC4APIRouter()
        result = router.circuit_list()
        assert result["code"] == 200
        assert isinstance(result["data"], list)

    def test_API_circuit_status_存在(self):
        """测试断路器状态端点 — 存在的断路器."""
        router = CC4APIRouter()
        # 获取第一个断路器名称
        circuits = router.circuit_list()["data"]
        if circuits:
            name = circuits[0]["name"]
            result = router.circuit_status(name)
            assert result["code"] == 200

    def test_API_circuit_status_不存在(self):
        """测试断路器状态端点 — 不存在的断路器."""
        router = CC4APIRouter()
        result = router.circuit_status("nonexistent")
        assert result["code"] == 404

    def test_API_circuit_reset_存在(self):
        """测试断路器重置端点."""
        router = CC4APIRouter()
        circuits = router.circuit_list()["data"]
        if circuits:
            name = circuits[0]["name"]
            result = router.circuit_reset(name)
            assert result["code"] == 200

    def test_API_overview(self):
        """测试系统概览端点."""
        router = CC4APIRouter()
        result = router.overview()
        assert result["code"] == 200
        data = result["data"]
        assert "uptime_seconds" in data
        assert "gateway_statistics" in data
        assert "governance_metrics" in data
        assert "health" in data
        assert "circuit_breakers" in data
        assert "feedback_loop_configured" in data

    def test_API_health_check_轻量级(self):
        """测试轻量级健康检查端点."""
        router = CC4APIRouter()
        result = router.health_check()
        assert result["code"] == 200
        assert result["data"]["status"] == "ok"
        assert "uptime_seconds" in result["data"]


# ============================================================
# 11. 端到端治理闭环测试
# ============================================================


class TestEndToEndGovernance:
    """端到端治理闭环测试."""

    def test_完整治理闭环_PASS评审(self):
        """测试 PASS 评审的完整治理闭环."""
        gw = UnifiedGateway()
        review = _make_review_result("pass", 92.0)
        result = gw.govern(
            review_result=review,
            operation_type="knowledge_generation",
            user_id="student-001",
            session_id="sess-001",
            trace_id="trace-e2e-001",
        )
        assert isinstance(result, dict)
        assert result["trace_id"] == "trace-e2e-001"
        # CC1→CC2 桥接应该执行
        assert result["cc1_to_cc2"] is not None
        # 检查事件已记录
        events = gw.get_events(limit=5)
        assert len(events) >= 1

    def test_完整治理闭环_BLOCK评审(self):
        """测试 BLOCK 评审的完整治理闭环."""
        gw = UnifiedGateway()
        review = _make_review_result("block", 25.0)
        result = gw.govern(
            review_result=review,
            operation_type="content_generation",
            trace_id="trace-e2e-002",
        )
        assert isinstance(result, dict)
        # BLOCK 评审应该降低置信度
        assert result["cc1_to_cc2"] is not None

    def test_多次治理后指标递增(self):
        """测试多次治理后统计指标递增."""
        gw = UnifiedGateway()
        for i in range(3):
            review = _make_review_result("pass", 80.0 + i * 5)
            gw.govern(review_result=review, trace_id=f"trace-batch-{i}")

        stats = gw.get_statistics()
        assert stats["total_governances"] == 3
        assert stats["cc1_to_cc2_runs"] == 3

    def test_治理指标包含CC1通过率(self):
        """测试治理指标包含 CC1 通过率."""
        gw = UnifiedGateway()
        metrics = gw.get_governance_metrics()
        assert hasattr(metrics, "cc1_pass_rate")
        assert isinstance(metrics.cc1_pass_rate, float)

    def test_治理指标包含断路器跳闸数(self):
        """测试治理指标包含断路器跳闸数."""
        gw = UnifiedGateway()
        metrics = gw.get_governance_metrics()
        assert hasattr(metrics, "circuit_breaker_trips")
        assert isinstance(metrics.circuit_breaker_trips, int)
