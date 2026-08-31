"""T4 人机协同 (HiTL) — 测试套件.

遵循 TDD Red-Green-Refactor:
1. 先写测试 (RED): 每个测试描述期望行为
2. 验证测试失败 (feature missing)
3. 最小实现 (GREEN)
4. 重构 (保持绿色)

测试覆盖:
- 异常体系: JSON-RPC -32400 范围
- ConfidenceGate: 置信度门控 (PASS/WARNING/BLOCK)
- EmergencyDetector: 紧急干预检测 (认知负荷/连续错误/异常速度)
- FeedbackLoop: 反馈回路 (提交/分类/路由/历史)
- HiTLManager: 四类协同场景 (确认/纠错/创造/紧急)
- InteractionMode: 四种交互模式 (被动/主动/强制/可选)
- 线程安全: 并发访问
- 边界情况: 空值/非法输入/超时
- 集成测试: 全生命周期
"""

from __future__ import annotations

import threading
import time

import pytest

from dy3_polaris.l1.models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    ConfidenceGateResult,
    EmergencyAlert,
    FeedbackCategory,
    FeedbackReport,
    FeedbackType,
    HiTLPriority,
    HiTLType,
    AlertType,
    BLOCK_THRESHOLD,
    WARNING_THRESHOLD,
    EMERGENCY_THRESHOLD,
    CONSECUTIVE_ERROR_THRESHOLD,
    FAST_ANSWER_THRESHOLD_MS,
    BKT_DEVIATION_THRESHOLD,
)
from dy3_polaris.l1.hitl_manager import (
    # 异常
    L1HiTLError,
    ConfidenceGateError,
    ApprovalError,
    FeedbackError,
    EmergencyError,
    # 置信度门控
    ConfidenceGate,
    GateDecision,
    # 紧急检测
    EmergencyDetector,
    # 反馈回路
    FeedbackLoop,
    FeedbackRoutingResult,
    # 交互模式
    InteractionMode,
    # 核心管理器
    HiTLManager,
    # 常量
    APPROVAL_TIMEOUT_SECONDS,
    MAX_CORRECTION_RETRIES,
    EMERGENCY_RESPONSE_MS,
    FEEDBACK_HISTORY_LIMIT,
)


# ============================================================
# 辅助函数
# ============================================================


def make_approval_request(
    user_id: str = "user-001",
    session_id: str = "sess-001",
    hitl_type: HiTLType = HiTLType.CONFIRMATION,
    content: str = "Dy3+ 能级跃迁知识卡片",
    priority: HiTLPriority = HiTLPriority.P2,
    confidence: float = 0.9,
) -> ApprovalRequest:
    """创建 ApprovalRequest 测试辅助."""
    return ApprovalRequest(
        user_id=user_id,
        session_id=session_id,
        hitl_type=hitl_type,
        content=content,
        priority=priority,
        confidence=confidence,
    )


def make_feedback_report(
    user_id: str = "user-001",
    session_id: str = "sess-001",
    feedback_type: FeedbackType = FeedbackType.INCORRECT,
    content: str = "波长标注有误, 应为 480nm 而非 485nm",
    severity: float = 0.8,
) -> FeedbackReport:
    """创建 FeedbackReport 测试辅助."""
    return FeedbackReport(
        user_id=user_id,
        session_id=session_id,
        feedback_type=feedback_type,
        content=content,
        severity=severity,
    )


# ============================================================
# 1. 异常体系测试
# ============================================================


class TestExceptionHierarchy:
    """异常继承与 JSON-RPC 错误码测试."""

    def test_base_error_inherits_l6(self):
        """L1HiTLError 继承 L6Error."""
        from dy3_polaris.l6.core.exceptions import L6Error
        assert issubclass(L1HiTLError, L6Error)

    def test_base_error_jsonrpc_code(self):
        """L1HiTLError JSON-RPC 码为 -32400."""
        err = L1HiTLError("test")
        assert err._jsonrpc_code() == -32400

    def test_confidence_gate_error_inherits_base(self):
        """ConfidenceGateError 继承 L1HiTLError."""
        assert issubclass(ConfidenceGateError, L1HiTLError)

    def test_confidence_gate_error_jsonrpc_code(self):
        """ConfidenceGateError JSON-RPC 码为 -32401."""
        err = ConfidenceGateError("test")
        assert err._jsonrpc_code() == -32401

    def test_approval_error_inherits_base(self):
        """ApprovalError 继承 L1HiTLError."""
        assert issubclass(ApprovalError, L1HiTLError)

    def test_approval_error_jsonrpc_code(self):
        """ApprovalError JSON-RPC 码为 -32402."""
        err = ApprovalError("test")
        assert err._jsonrpc_code() == -32402

    def test_feedback_error_inherits_base(self):
        """FeedbackError 继承 L1HiTLError."""
        assert issubclass(FeedbackError, L1HiTLError)

    def test_feedback_error_jsonrpc_code(self):
        """FeedbackError JSON-RPC 码为 -32403."""
        err = FeedbackError("test")
        assert err._jsonrpc_code() == -32403

    def test_emergency_error_inherits_base(self):
        """EmergencyError 继承 L1HiTLError."""
        assert issubclass(EmergencyError, L1HiTLError)

    def test_emergency_error_jsonrpc_code(self):
        """EmergencyError JSON-RPC 码为 -32404."""
        err = EmergencyError("test")
        assert err._jsonrpc_code() == -32404

    def test_approval_error_contains_request_id(self):
        """ApprovalError 包含 request_id 上下文."""
        err = ApprovalError("审批超时", request_id="hitl-abc123")
        assert err.context.get("request_id") == "hitl-abc123"


# ============================================================
# 2. 置信度门控测试
# ============================================================


class TestConfidenceGate:
    """置信度门控测试 (设计文档 4.4)."""

    def test_evaluate_pass(self):
        """置信度 >= 0.85 → PASS."""
        gate = ConfidenceGate()
        result = gate.evaluate(0.9)
        assert result.gate_result == ConfidenceGateResult.PASS
        assert result.decision == GateDecision.PRESENT

    def test_evaluate_warning(self):
        """0.4 <= 置信度 < 0.85 → WARNING."""
        gate = ConfidenceGate()
        result = gate.evaluate(0.6)
        assert result.gate_result == ConfidenceGateResult.WARNING
        assert result.decision == GateDecision.PRESENT_WITH_LABEL

    def test_evaluate_block(self):
        """置信度 < 0.4 → BLOCK."""
        gate = ConfidenceGate()
        result = gate.evaluate(0.3)
        assert result.gate_result == ConfidenceGateResult.BLOCK
        assert result.decision == GateDecision.HOLD_FOR_REVIEW

    def test_evaluate_boundary_pass(self):
        """边界值: 置信度 = 0.85 → PASS."""
        gate = ConfidenceGate()
        result = gate.evaluate(WARNING_THRESHOLD)
        assert result.gate_result == ConfidenceGateResult.PASS

    def test_evaluate_boundary_block(self):
        """边界值: 置信度 = 0.4 → WARNING (不是 BLOCK)."""
        gate = ConfidenceGate()
        result = gate.evaluate(BLOCK_THRESHOLD)
        assert result.gate_result == ConfidenceGateResult.WARNING

    def test_evaluate_invalid_confidence(self):
        """非法置信度 (< 0 或 > 1) 抛异常."""
        gate = ConfidenceGate()
        with pytest.raises(ConfidenceGateError):
            gate.evaluate(-0.1)
        with pytest.raises(ConfidenceGateError):
            gate.evaluate(1.5)

    def test_evaluate_records_provenance(self):
        """门控结果记录 Provenance (来源追踪)."""
        gate = ConfidenceGate()
        result = gate.evaluate(0.6, artifact_id="art-001", agent_id="agent-001")
        assert result.provenance is not None
        assert result.provenance["artifact_id"] == "art-001"
        assert result.provenance["agent_id"] == "agent-001"
        assert "timestamp" in result.provenance

    def test_evaluate_recommends_interaction_mode(self):
        """门控结果推荐交互模式."""
        gate = ConfidenceGate()
        # PASS → 被动确认
        pass_result = gate.evaluate(0.9)
        assert pass_result.recommended_mode == InteractionMode.PASSIVE_CONFIRMATION
        # WARNING → 主动建议
        warn_result = gate.evaluate(0.6)
        assert warn_result.recommended_mode == InteractionMode.PROACTIVE_SUGGESTION
        # BLOCK → 强制阻断
        block_result = gate.evaluate(0.3)
        assert block_result.recommended_mode == InteractionMode.MANDATORY_BLOCK


# ============================================================
# 3. 紧急干预检测测试
# ============================================================


class TestEmergencyDetector:
    """紧急干预检测测试 (设计文档 4.2, 4.3)."""

    def test_check_normal_no_alert(self):
        """正常状态不触发紧急警报."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.5,
            consecutive_errors=3,
            avg_answer_time_ms=8000,
        )
        assert alert is None

    def test_check_high_cognitive_load(self):
        """认知负荷 >= 0.95 触发紧急警报."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.97,
            consecutive_errors=2,
            avg_answer_time_ms=8000,
        )
        assert alert is not None
        assert alert.alert_type == AlertType.HIGH_COGNITIVE_LOAD
        assert alert.trigger_value == pytest.approx(0.97, abs=0.01)

    def test_check_consecutive_errors(self):
        """连续错误 >= 10 次触发紧急警报."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.5,
            consecutive_errors=CONSECUTIVE_ERROR_THRESHOLD,
            avg_answer_time_ms=8000,
        )
        assert alert is not None
        assert alert.alert_type == AlertType.CONSECUTIVE_ERRORS
        assert alert.error_count == CONSECUTIVE_ERROR_THRESHOLD

    def test_check_fast_answering(self):
        """异常答题速度 (< 5秒) 触发紧急警报."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.5,
            consecutive_errors=2,
            avg_answer_time_ms=3000,  # < 5000ms
        )
        assert alert is not None
        assert alert.alert_type == AlertType.FAST_ANSWERING

    def test_check_bkt_deviation(self):
        """BKT 预测偏差 > 30% 触发纠错型警报 (非紧急)."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.5,
            consecutive_errors=2,
            avg_answer_time_ms=8000,
            bkt_deviation=0.35,  # > 0.3
        )
        assert alert is not None
        assert alert.alert_type == AlertType.BKT_DEVIATION

    def test_check_priority_p0(self):
        """紧急警报优先级为 P0."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.99,
            consecutive_errors=2,
            avg_answer_time_ms=8000,
        )
        assert alert is not None

    def test_check_includes_cognitive_load(self):
        """紧急警报包含认知负荷值."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.96,
            consecutive_errors=2,
            avg_answer_time_ms=8000,
        )
        assert alert is not None
        assert alert.cognitive_load == pytest.approx(0.96, abs=0.01)

    def test_check_multiple_triggers(self):
        """多个触发条件同时满足, 优先返回最严重的."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.98,
            consecutive_errors=15,
            avg_answer_time_ms=2000,
        )
        assert alert is not None
        # 认知负荷是最严重的 (P0)
        assert alert.alert_type == AlertType.HIGH_COGNITIVE_LOAD

    def test_check_invalid_cognitive_load(self):
        """非法认知负荷值抛异常."""
        detector = EmergencyDetector()
        with pytest.raises(EmergencyError):
            detector.check(
                session_id="sess-001",
                user_id="user-001",
                cognitive_load=1.5,
                consecutive_errors=0,
                avg_answer_time_ms=8000,
            )

    def test_resolve_alert(self):
        """解决紧急警报."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.97,
            consecutive_errors=2,
            avg_answer_time_ms=8000,
        )
        assert alert is not None
        assert not alert.is_resolved
        detector.resolve_alert(alert.alert_id)
        assert alert.is_resolved

    def test_get_active_alerts(self):
        """获取活跃警报列表."""
        detector = EmergencyDetector()
        detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.97,
            consecutive_errors=2,
            avg_answer_time_ms=8000,
        )
        active = detector.get_active_alerts()
        assert len(active) == 1
        assert not active[0].is_resolved


# ============================================================
# 4. 反馈回路测试
# ============================================================


class TestFeedbackLoop:
    """反馈回路测试 (设计文档 4.5)."""

    def test_submit_feedback(self):
        """提交反馈."""
        loop = FeedbackLoop()
        report = make_feedback_report()
        result = loop.submit_feedback(report)
        assert result is not None
        assert result.report_id == report.report_id

    def test_submit_feedback_factual_classification(self):
        """事实性反馈分类 (内容有误)."""
        loop = FeedbackLoop()
        report = make_feedback_report(
            feedback_type=FeedbackType.INCORRECT,
            content="波长标注有误",
        )
        result = loop.submit_feedback(report)
        assert result.category == FeedbackCategory.FACTUAL
        assert result.routing_target == "knowledge_base"

    def test_submit_feedback_adaptive_classification(self):
        """适应性反馈分类 (需要更多)."""
        loop = FeedbackLoop()
        report = make_feedback_report(
            feedback_type=FeedbackType.NEED_MORE,
            content="内容太难, 不适合我的水平",
        )
        result = loop.submit_feedback(report)
        assert result.category == FeedbackCategory.ADAPTIVE
        assert result.routing_target == "abac_policy"

    def test_submit_feedback_safety_classification(self):
        """安全性反馈分类 (举报)."""
        loop = FeedbackLoop()
        report = make_feedback_report(
            feedback_type=FeedbackType.REPORT,
            content="包含不当内容",
        )
        result = loop.submit_feedback(report)
        assert result.category == FeedbackCategory.SAFETY
        assert result.routing_target == "governance"

    def test_submit_feedback_understood_no_routing(self):
        """已理解反馈不触发路由."""
        loop = FeedbackLoop()
        report = make_feedback_report(
            feedback_type=FeedbackType.UNDERSTOOD,
            content="内容清晰易懂",
        )
        result = loop.submit_feedback(report)
        assert result.category is None
        assert result.routing_target is None

    def test_get_feedback_history(self):
        """获取反馈历史."""
        loop = FeedbackLoop()
        for i in range(5):
            loop.submit_feedback(make_feedback_report(
                content=f"反馈 {i}",
            ))
        history = loop.get_feedback_history("sess-001")
        assert len(history) == 5

    def test_get_feedback_history_filtered_by_session(self):
        """按会话过滤反馈历史."""
        loop = FeedbackLoop()
        loop.submit_feedback(make_feedback_report(session_id="sess-001"))
        loop.submit_feedback(make_feedback_report(session_id="sess-002"))
        history = loop.get_feedback_history("sess-001")
        assert len(history) == 1

    def test_get_feedback_history_limit(self):
        """反馈历史限制数量."""
        loop = FeedbackLoop()
        for i in range(FEEDBACK_HISTORY_LIMIT + 10):
            loop.submit_feedback(make_feedback_report(
                content=f"反馈 {i}",
            ))
        history = loop.get_feedback_history("sess-001")
        assert len(history) <= FEEDBACK_HISTORY_LIMIT

    def test_submit_feedback_invalid_report(self):
        """非法反馈报告抛异常."""
        loop = FeedbackLoop()
        with pytest.raises(FeedbackError):
            loop.submit_feedback(None)


# ============================================================
# 5. HiTLManager 核心测试
# ============================================================


class TestHiTLManager:
    """HiTL 协同管理器核心测试 (设计文档 4.1)."""

    def test_create_manager(self):
        """创建 HiTL 管理器."""
        manager = HiTLManager()
        assert manager is not None

    # --- 确认型场景 ---

    def test_handle_confirmation(self):
        """确认型: 学生确认"已理解"."""
        manager = HiTLManager()
        request = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="Dy3+ 能级跃迁知识卡片",
            confidence=0.9,
        )
        assert request.hitl_type == HiTLType.CONFIRMATION
        assert request.status == "pending"

        response = ApprovalResponse(
            request_id=request.request_id,
            responder_id="user-001",
            decision=ApprovalDecision.APPROVE,
        )
        result = manager.handle_confirmation(request, response)
        assert result.is_approved()
        assert request.status == "approved"

    def test_handle_confirmation_reject(self):
        """确认型: 学生标记"不理解"."""
        manager = HiTLManager()
        request = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="晶体场分裂理论",
            confidence=0.6,
        )
        response = ApprovalResponse(
            request_id=request.request_id,
            responder_id="user-001",
            decision=ApprovalDecision.REJECT,
            comment="仍不理解",
        )
        result = manager.handle_confirmation(request, response)
        assert not result.is_approved()
        assert request.status == "rejected"

    # --- 纠错型场景 ---

    def test_handle_correction(self):
        """纠错型: 学生标记"不理解" → Agent 自纠."""
        manager = HiTLManager()
        request = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CORRECTION,
            content="Dy3+ 黄蓝比调控",
            confidence=0.5,
        )
        response = ApprovalResponse(
            request_id=request.request_id,
            responder_id="user-001",
            decision=ApprovalDecision.REJECT,
            comment="解释不够清楚",
        )
        result = manager.handle_correction(request, response)
        assert result is not None
        assert result.retry_count == 1

    def test_handle_correction_max_retries_escalate(self):
        """纠错型: 3 次未解决 → 教师介入."""
        manager = HiTLManager()
        request = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CORRECTION,
            content="复杂概念解释",
            confidence=0.45,
        )
        # 模拟 3 次拒绝
        for i in range(MAX_CORRECTION_RETRIES):
            response = ApprovalResponse(
                request_id=request.request_id,
                responder_id="user-001",
                decision=ApprovalDecision.REJECT,
                comment=f"第 {i+1} 次不理解",
            )
            result = manager.handle_correction(request, response)

        # 应已升级至教师
        assert result.escalated is True
        assert result.escalation_target == "teacher"

    def test_handle_correction_resolved_before_max(self):
        """纠错型: 在 3 次内解决 → 不升级."""
        manager = HiTLManager()
        request = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CORRECTION,
            content="概念解释",
            confidence=0.5,
        )
        # 第 1 次拒绝
        resp1 = ApprovalResponse(
            request_id=request.request_id,
            responder_id="user-001",
            decision=ApprovalDecision.REJECT,
        )
        manager.handle_correction(request, resp1)

        # 第 2 次通过
        resp2 = ApprovalResponse(
            request_id=request.request_id,
            responder_id="user-001",
            decision=ApprovalDecision.APPROVE,
        )
        result = manager.handle_correction(request, resp2)
        assert result.escalated is False
        assert result.retry_count == 2

    # --- 创造型场景 ---

    def test_handle_creative(self):
        """创造型: 教师创建内容 → 审核校验."""
        manager = HiTLManager()
        request = manager.create_approval_request(
            user_id="teacher-001",
            session_id="sess-001",
            hitl_type=HiTLType.CREATIVE,
            content="Dy3+ 黄蓝比调控综合实验指导书",
            confidence=0.8,
            priority=HiTLPriority.P2,
        )
        response = ApprovalResponse(
            request_id=request.request_id,
            responder_id="reviewer-001",
            decision=ApprovalDecision.APPROVE,
            comment="内容审核通过",
        )
        result = manager.handle_creative(request, response)
        assert result.is_approved()
        assert request.status == "approved"

    def test_handle_creative_reject_with_review(self):
        """创造型: 审核不通过 → 退回修改."""
        manager = HiTLManager()
        request = manager.create_approval_request(
            user_id="teacher-001",
            session_id="sess-001",
            hitl_type=HiTLType.CREATIVE,
            content="实验指导书草稿",
            confidence=0.3,
        )
        response = ApprovalResponse(
            request_id=request.request_id,
            responder_id="reviewer-001",
            decision=ApprovalDecision.REJECT,
            comment="数据不准确, 需修改",
        )
        result = manager.handle_creative(request, response)
        assert not result.is_approved()
        assert request.status == "rejected"

    def test_handle_creative_modify(self):
        """创造型: 修改后放行."""
        manager = HiTLManager()
        request = manager.create_approval_request(
            user_id="teacher-001",
            session_id="sess-001",
            hitl_type=HiTLType.CREATIVE,
            content="实验指导书",
            confidence=0.7,
        )
        response = ApprovalResponse(
            request_id=request.request_id,
            responder_id="reviewer-001",
            decision=ApprovalDecision.MODIFY,
            modifications=[{"field": "difficulty", "old_value": "hard", "new_value": "medium"}],
        )
        result = manager.handle_creative(request, response)
        assert result.decision == ApprovalDecision.MODIFY
        assert len(result.modifications) > 0

    # --- 紧急干预场景 ---

    def test_handle_emergency(self):
        """紧急干预: 自动暂停 + 通知教师."""
        manager = HiTLManager()
        alert = manager.handle_emergency(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.97,
            consecutive_errors=12,
            avg_answer_time_ms=8000,
        )
        assert alert is not None
        assert alert.alert_type == AlertType.HIGH_COGNITIVE_LOAD
        assert alert.is_resolved is False

    def test_handle_emergency_consecutive_errors(self):
        """紧急干预: 连续错误触发."""
        manager = HiTLManager()
        alert = manager.handle_emergency(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.5,
            consecutive_errors=15,
            avg_answer_time_ms=8000,
        )
        assert alert is not None
        assert alert.alert_type == AlertType.CONSECUTIVE_ERRORS

    def test_handle_emergency_no_trigger(self):
        """无触发条件时不产生紧急干预."""
        manager = HiTLManager()
        alert = manager.handle_emergency(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.5,
            consecutive_errors=3,
            avg_answer_time_ms=8000,
        )
        assert alert is None

    def test_handle_emergency_resolves_alert(self):
        """紧急干预后可解决警报."""
        manager = HiTLManager()
        alert = manager.handle_emergency(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.98,
            consecutive_errors=2,
            avg_answer_time_ms=8000,
        )
        assert alert is not None
        manager.resolve_emergency(alert.alert_id)
        assert alert.is_resolved

    # --- 审批请求管理 ---

    def test_create_approval_request(self):
        """创建审批请求."""
        manager = HiTLManager()
        request = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="测试内容",
            confidence=0.85,
        )
        assert request.request_id.startswith("hitl-")
        assert request.status == "pending"

    def test_get_pending_requests(self):
        """获取待处理请求."""
        manager = HiTLManager()
        manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="内容1",
        )
        manager.create_approval_request(
            user_id="user-002",
            session_id="sess-002",
            hitl_type=HiTLType.CREATIVE,
            content="内容2",
        )
        pending = manager.get_pending_requests()
        assert len(pending) == 2

    def test_get_pending_requests_by_user(self):
        """按用户过滤待处理请求."""
        manager = HiTLManager()
        manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="内容1",
        )
        manager.create_approval_request(
            user_id="user-002",
            session_id="sess-002",
            hitl_type=HiTLType.CONFIRMATION,
            content="内容2",
        )
        pending = manager.get_pending_requests(user_id="user-001")
        assert len(pending) == 1
        assert pending[0].user_id == "user-001"

    def test_get_request_by_id(self):
        """按 ID 获取请求."""
        manager = HiTLManager()
        created = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="测试",
        )
        found = manager.get_request(created.request_id)
        assert found is not None
        assert found.request_id == created.request_id

    def test_get_request_not_found(self):
        """获取不存在的请求返回 None."""
        manager = HiTLManager()
        assert manager.get_request("hitl-xxx") is None

    def test_expired_request_auto_status(self):
        """过期请求自动标记为 expired."""
        manager = HiTLManager()
        request = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="测试",
            deadline=int(time.time() * 1000) - 10000,  # 已过期
        )
        # 获取待处理时, 过期请求不应出现
        pending = manager.get_pending_requests()
        assert len(pending) == 0


# ============================================================
# 6. 交互模式测试
# ============================================================


class TestInteractionMode:
    """交互模式测试 (设计文档 4.3)."""

    def test_passive_confirmation_mode(self):
        """被动确认模式: 内容底部附加确认控件."""
        manager = HiTLManager()
        mode = manager.get_interaction_mode(HiTLType.CONFIRMATION, ConfidenceGateResult.PASS)
        assert mode == InteractionMode.PASSIVE_CONFIRMATION

    def test_proactive_suggestion_mode(self):
        """主动建议模式: 检测学习瓶颈时弹出建议."""
        manager = HiTLManager()
        mode = manager.get_interaction_mode(HiTLType.CORRECTION, ConfidenceGateResult.WARNING)
        assert mode == InteractionMode.PROACTIVE_SUGGESTION

    def test_mandatory_block_mode(self):
        """强制阻断模式: 紧急暂停."""
        manager = HiTLManager()
        mode = manager.get_interaction_mode(HiTLType.EMERGENCY, ConfidenceGateResult.BLOCK)
        assert mode == InteractionMode.MANDATORY_BLOCK

    def test_optional_negotiation_mode(self):
        """可选协商模式: 教师与系统双向协商."""
        manager = HiTLManager()
        mode = manager.get_interaction_mode(HiTLType.CREATIVE, ConfidenceGateResult.WARNING)
        assert mode == InteractionMode.OPTIONAL_NEGOTIATION

    def test_interaction_mode_enum_values(self):
        """交互模式枚举值."""
        assert InteractionMode.PASSIVE_CONFIRMATION.value == "passive_confirmation"
        assert InteractionMode.PROACTIVE_SUGGESTION.value == "proactive_suggestion"
        assert InteractionMode.MANDATORY_BLOCK.value == "mandatory_block"
        assert InteractionMode.OPTIONAL_NEGOTIATION.value == "optional_negotiation"


# ============================================================
# 7. 反馈路由测试
# ============================================================


class TestFeedbackRouting:
    """反馈路由测试 (设计文档 4.5)."""

    def test_factual_routes_to_knowledge_base(self):
        """事实性错误 → 知识库修正 (L3)."""
        loop = FeedbackLoop()
        report = make_feedback_report(
            feedback_type=FeedbackType.INCORRECT,
            content="Dy3+ 特征波长标注错误",
        )
        result = loop.submit_feedback(report)
        assert result.routing_target == "knowledge_base"
        assert result.category == FeedbackCategory.FACTUAL

    def test_adaptive_routes_to_abac(self):
        """适应性不足 → ABAC 策略调整."""
        loop = FeedbackLoop()
        report = make_feedback_report(
            feedback_type=FeedbackType.NEED_MORE,
            content="内容难度过高",
        )
        result = loop.submit_feedback(report)
        assert result.routing_target == "abac_policy"
        assert result.category == FeedbackCategory.ADAPTIVE

    def test_safety_routes_to_governance(self):
        """安全问题 → 升级至 L0 治理."""
        loop = FeedbackLoop()
        report = make_feedback_report(
            feedback_type=FeedbackType.REPORT,
            content="包含不当内容",
            severity=1.0,
        )
        result = loop.submit_feedback(report)
        assert result.routing_target == "governance"
        assert result.category == FeedbackCategory.SAFETY

    def test_feedback_includes_severity(self):
        """反馈路由结果包含严重度."""
        loop = FeedbackLoop()
        report = make_feedback_report(severity=0.9)
        result = loop.submit_feedback(report)
        assert result.severity == pytest.approx(0.9, abs=0.01)

    def test_feedback_includes_envelope_id(self):
        """反馈记录包含信封 ID (可溯源)."""
        loop = FeedbackLoop()
        report = make_feedback_report()
        report.source_envelope_id = "env-001"
        result = loop.submit_feedback(report)
        assert result.source_envelope_id == "env-001"


# ============================================================
# 8. 线程安全测试
# ============================================================


class TestThreadSafety:
    """线程安全测试."""

    def test_concurrent_submit_feedback(self):
        """并发提交反馈."""
        loop = FeedbackLoop()
        results: list[Exception | None] = [None] * 50

        def submit(idx: int) -> None:
            try:
                loop.submit_feedback(make_feedback_report(
                    content=f"并发反馈 {idx}",
                ))
                results[idx] = None
            except Exception as e:
                results[idx] = e

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is None for r in results)
        history = loop.get_feedback_history("sess-001")
        assert len(history) == 50

    def test_concurrent_create_approval(self):
        """并发创建审批请求."""
        manager = HiTLManager()
        request_ids: list[str] = ["" for _ in range(20)]
        errors: list[Exception | None] = [None] * 20

        def create(idx: int) -> None:
            try:
                req = manager.create_approval_request(
                    user_id=f"user-{idx}",
                    session_id=f"sess-{idx}",
                    hitl_type=HiTLType.CONFIRMATION,
                    content=f"内容 {idx}",
                )
                request_ids[idx] = req.request_id
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=create, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(e is None for e in errors)
        assert len(set(request_ids)) == 20  # 全部唯一

    def test_concurrent_emergency_detection(self):
        """并发紧急检测."""
        detector = EmergencyDetector()
        alerts: list[EmergencyAlert | None] = [None] * 10
        errors: list[Exception | None] = [None] * 10

        def detect(idx: int) -> None:
            try:
                alerts[idx] = detector.check(
                    session_id=f"sess-{idx}",
                    user_id=f"user-{idx}",
                    cognitive_load=0.97,
                    consecutive_errors=2,
                    avg_answer_time_ms=8000,
                )
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=detect, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(e is None for e in errors)
        assert all(a is not None for a in alerts)


# ============================================================
# 9. 边界情况测试
# ============================================================


class TestEdgeCases:
    """边界情况测试."""

    def test_confidence_exactly_zero(self):
        """置信度为 0 → BLOCK."""
        gate = ConfidenceGate()
        result = gate.evaluate(0.0)
        assert result.gate_result == ConfidenceGateResult.BLOCK

    def test_confidence_exactly_one(self):
        """置信度为 1 → PASS."""
        gate = ConfidenceGate()
        result = gate.evaluate(1.0)
        assert result.gate_result == ConfidenceGateResult.PASS

    def test_emergency_detector_zero_errors(self):
        """0 次连续错误不触发."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.5,
            consecutive_errors=0,
            avg_answer_time_ms=8000,
        )
        assert alert is None

    def test_emergency_detector_boundary_cognitive_load(self):
        """认知负荷 = 0.95 → 触发 (边界值)."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=EMERGENCY_THRESHOLD,
            consecutive_errors=0,
            avg_answer_time_ms=8000,
        )
        assert alert is not None

    def test_emergency_detector_boundary_fast_answer(self):
        """答题时间 = 5000ms → 不触发 (边界值, < 5000 才触发)."""
        detector = EmergencyDetector()
        alert = detector.check(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.5,
            consecutive_errors=0,
            avg_answer_time_ms=FAST_ANSWER_THRESHOLD_MS,
        )
        assert alert is None

    def test_empty_feedback_history(self):
        """空反馈历史."""
        loop = FeedbackLoop()
        history = loop.get_feedback_history("sess-001")
        assert len(history) == 0

    def test_approval_request_with_deadline(self):
        """带截止时间的审批请求."""
        manager = HiTLManager()
        future_ts = int(time.time() * 1000) + 60000  # 1 分钟后
        request = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="测试",
            deadline=future_ts,
        )
        assert not request.is_expired()

    def test_handle_confirmation_nonexistent_request(self):
        """处理不存在的请求抛异常."""
        manager = HiTLManager()
        response = ApprovalResponse(
            request_id="hitl-xxx",
            responder_id="user-001",
            decision=ApprovalDecision.APPROVE,
        )
        with pytest.raises(ApprovalError):
            manager.handle_confirmation(
                ApprovalRequest(
                    user_id="user-001",
                    session_id="sess-001",
                    hitl_type=HiTLType.CONFIRMATION,
                    content="test",
                    request_id="hitl-xxx",
                ),
                response,
            )


# ============================================================
# 10. 集成测试: 全生命周期
# ============================================================


class TestIntegrationLifecycle:
    """全生命周期集成测试."""

    def test_full_hitl_lifecycle(self):
        """完整 HiTL 生命周期: 生成→门控→确认→反馈→改进."""
        manager = HiTLManager()
        gate = ConfidenceGate()

        # 1. Agent 生成内容, 置信度门控评估
        gate_result = gate.evaluate(0.6, artifact_id="art-001", agent_id="agent-001")
        assert gate_result.gate_result == ConfidenceGateResult.WARNING

        # 2. 根据 WARNING → 创建确认请求
        request = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="Dy3+ 能级跃迁",
            confidence=0.6,
        )

        # 3. 学生确认"已理解"
        response = ApprovalResponse(
            request_id=request.request_id,
            responder_id="user-001",
            decision=ApprovalDecision.APPROVE,
        )
        manager.handle_confirmation(request, response)
        assert request.status == "approved"

        # 4. 学生后续提交反馈
        loop = manager.feedback_loop
        report = make_feedback_report(
            feedback_type=FeedbackType.INCORRECT,
            content="细节有误",
        )
        result = loop.submit_feedback(report)
        assert result.category == FeedbackCategory.FACTUAL

        # 5. 验证反馈历史
        history = loop.get_feedback_history("sess-001")
        assert len(history) == 1

    def test_emergency_intervention_flow(self):
        """紧急干预流程: 检测→暂停→通知→解决."""
        manager = HiTLManager()

        # 1. 检测紧急情况
        alert = manager.handle_emergency(
            session_id="sess-001",
            user_id="user-001",
            cognitive_load=0.98,
            consecutive_errors=12,
            avg_answer_time_ms=3000,
        )
        assert alert is not None

        # 2. 验证警报已记录
        active = manager.emergency_detector.get_active_alerts()
        assert len(active) == 1

        # 3. 解决紧急情况
        manager.resolve_emergency(alert.alert_id)
        assert alert.is_resolved

        # 4. 验证无活跃警报
        active = manager.emergency_detector.get_active_alerts()
        assert len(active) == 0

    def test_correction_escalation_flow(self):
        """纠错升级流程: 拒绝→自纠→拒绝→自纠→拒绝→升级教师."""
        manager = HiTLManager()

        request = manager.create_approval_request(
            user_id="user-001",
            session_id="sess-001",
            hitl_type=HiTLType.CORRECTION,
            content="复杂概念",
            confidence=0.45,
        )

        # 模拟 3 次拒绝后升级
        for i in range(MAX_CORRECTION_RETRIES):
            resp = ApprovalResponse(
                request_id=request.request_id,
                responder_id="user-001",
                decision=ApprovalDecision.REJECT,
                comment=f"第 {i+1} 次拒绝",
            )
            result = manager.handle_correction(request, resp)

        assert result.escalated is True
        assert result.escalation_target == "teacher"

    def test_confidence_gate_to_interaction_mode_mapping(self):
        """置信度门控 → 交互模式映射."""
        manager = HiTLManager()
        gate = ConfidenceGate()

        # PASS → 被动确认
        pass_result = gate.evaluate(0.9)
        mode = manager.get_interaction_mode(HiTLType.CONFIRMATION, pass_result.gate_result)
        assert mode == InteractionMode.PASSIVE_CONFIRMATION

        # WARNING + 纠错 → 主动建议
        warn_result = gate.evaluate(0.6)
        mode = manager.get_interaction_mode(HiTLType.CORRECTION, warn_result.gate_result)
        assert mode == InteractionMode.PROACTIVE_SUGGESTION

        # BLOCK + 紧急 → 强制阻断
        block_result = gate.evaluate(0.3)
        mode = manager.get_interaction_mode(HiTLType.EMERGENCY, block_result.gate_result)
        assert mode == InteractionMode.MANDATORY_BLOCK

    def test_feedback_closed_loop(self):
        """反馈闭环: 提交→分类→路由→历史追踪."""
        manager = HiTLManager()
        loop = manager.feedback_loop

        # 提交多种反馈
        loop.submit_feedback(make_feedback_report(
            feedback_type=FeedbackType.INCORRECT,
            content="事实错误1",
        ))
        loop.submit_feedback(make_feedback_report(
            feedback_type=FeedbackType.NEED_MORE,
            content="需要更多细节",
        ))
        loop.submit_feedback(make_feedback_report(
            feedback_type=FeedbackType.UNDERSTOOD,
            content="已理解",
        ))

        # 验证历史
        history = loop.get_feedback_history("sess-001")
        assert len(history) == 3

        # 验证分类
        factual_count = sum(1 for h in history if h.feedback_type == FeedbackType.INCORRECT)
        adaptive_count = sum(1 for h in history if h.feedback_type == FeedbackType.NEED_MORE)
        understood_count = sum(1 for h in history if h.feedback_type == FeedbackType.UNDERSTOOD)
        assert factual_count == 1
        assert adaptive_count == 1
        assert understood_count == 1
