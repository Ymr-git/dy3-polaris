"""G3 CC2 人机协作层完整测试.

覆盖 CC2 人机协作层的全部能力，共计 18 个维度 120+ 测试用例：
1.  Test枚举值 — 7 个枚举的字符串值与 (str, Enum) 继承
2.  TestREACTScore — 均分计算、四级映射与边界值
3.  TestAgentCollaborationProfile — 创建、步数管理、覆盖记录
4.  Test干预请求模型 — InterventionRequest / HumanResponse / InterventionRecord
5.  Test协商模型 — NegotiationRound / NegotiationSession
6.  TestModeSwitchEvent — 创建与 is_upgrade 属性
7.  TestCollaborationConfig — 默认配置与字段约束
8.  Test异常体系 — 7 个异常类、JSON-RPC 码、继承层级
9.  TestCollaborationEngine_配置管理 — register / get / update / list
10. TestCollaborationEngine_REACT评分 — evaluate_react
11. TestCollaborationEngine_模式切换 — switch_mode 相邻/跳级/同模式
12. TestCollaborationEngine_干预请求 — create / respond / expire / cancel
13. TestCollaborationEngine_升级 — escalate_to_human
14. TestCollaborationEngine_自主步数 — check_auto_step
15. TestCollaborationEngine_协商 — start / add_round / finalize
16. TestCollaborationEngine_持续质量 — evaluate_sustained_quality
17. TestCollaborationEngine_查询与统计 — query / get_stats / clear
18. Test端到端集成 — 完整教学场景流程
"""

from __future__ import annotations

import enum
import time
from typing import Any

import pytest

from dy3_polaris.l0.cc2 import (
    # 枚举
    CollaborationMode,
    HumanDecision,
    InterventionStatus,
    InterventionType,
    NegotiationPhase,
    ReviewOutcome,
    SwitchTrigger,
    # 模型
    AgentCollaborationProfile,
    CollaborationConfig,
    HumanResponse,
    InterventionRecord,
    InterventionRequest,
    ModeSwitchEvent,
    NegotiationRound,
    NegotiationSession,
    REACTScore,
    # 异常
    CC2Error,
    EscalationTargetError,
    InterventionConflictError,
    InterventionTimeoutError,
    ModeSwitchError,
    NegotiationExhaustedError,
    ProfileNotFoundError,
    # 引擎
    CollaborationEngine,
)
from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 辅助函数
# ============================================================


def _make_profile(
    agent_id: str = "agent-001",
    mode: CollaborationMode = CollaborationMode.CONDITIONAL,
    **kwargs: Any,
) -> AgentCollaborationProfile:
    """创建测试用 Agent 协作配置."""
    return AgentCollaborationProfile(agent_id=agent_id, mode=mode, **kwargs)


def _make_engine(
    config: CollaborationConfig | None = None,
    register_agent: str | None = "agent-001",
) -> CollaborationEngine:
    """创建测试引擎，可选择性注册一个默认 Agent."""
    engine = CollaborationEngine(config=config)
    if register_agent:
        engine.register_profile(_make_profile(register_agent))
    return engine


def _make_react_score(**kwargs: float) -> REACTScore:
    """创建 REACT 评分对象."""
    return REACTScore(**kwargs)


# ============================================================
# 1. 测试枚举值
# ============================================================


class Test枚举值:
    """验证 7 个枚举的字符串值与 (str, Enum) 继承."""

    # --- CollaborationMode ---

    def test_协作模式_四个值(self) -> None:
        assert len(CollaborationMode) == 4

    def test_协作模式_字符串值(self) -> None:
        assert CollaborationMode.SUPERVISED.value == "supervised"
        assert CollaborationMode.CONDITIONAL.value == "conditional"
        assert CollaborationMode.MONITORED.value == "monitored"
        assert CollaborationMode.AUTONOMOUS.value == "autonomous"

    def test_协作模式_继承str和Enum(self) -> None:
        assert issubclass(CollaborationMode, str)
        assert issubclass(CollaborationMode, enum.Enum)

    def test_协作模式_字符串比较(self) -> None:
        assert CollaborationMode.SUPERVISED == "supervised"
        assert CollaborationMode.AUTONOMOUS != "monitored"

    def test_协作模式_遍历(self) -> None:
        modes = list(CollaborationMode)
        assert set(modes) == {
            CollaborationMode.SUPERVISED,
            CollaborationMode.CONDITIONAL,
            CollaborationMode.MONITORED,
            CollaborationMode.AUTONOMOUS,
        }

    # --- InterventionType ---

    def test_干预类型_五个值(self) -> None:
        assert len(InterventionType) == 5

    def test_干预类型_字符串值(self) -> None:
        assert InterventionType.CHECKPOINT.value == "checkpoint"
        assert InterventionType.ESCALATION.value == "escalation"
        assert InterventionType.NEGOTIATION.value == "negotiation"
        assert InterventionType.OVERRIDE.value == "override"
        assert InterventionType.REVIEW.value == "review"

    def test_干预类型_继承str和Enum(self) -> None:
        assert issubclass(InterventionType, str)
        assert issubclass(InterventionType, enum.Enum)

    def test_干预类型_字符串比较(self) -> None:
        assert InterventionType.CHECKPOINT == "checkpoint"
        assert InterventionType.ESCALATION != "override"

    # --- InterventionStatus ---

    def test_干预状态_五个值(self) -> None:
        assert len(InterventionStatus) == 5

    def test_干预状态_完整生命周期值(self) -> None:
        expected = {"pending", "active", "resolved", "expired", "cancelled"}
        assert {s.value for s in InterventionStatus} == expected

    def test_干预状态_继承str和Enum(self) -> None:
        assert issubclass(InterventionStatus, str)
        assert issubclass(InterventionStatus, enum.Enum)

    # --- HumanDecision ---

    def test_人类决策_七个值(self) -> None:
        assert len(HumanDecision) == 7

    def test_人类决策_字符串值(self) -> None:
        assert HumanDecision.APPROVE.value == "approve"
        assert HumanDecision.REJECT.value == "reject"
        assert HumanDecision.MODIFY.value == "modify"
        assert HumanDecision.COUNTEROFFER.value == "counteroffer"
        assert HumanDecision.SKIP.value == "skip"
        assert HumanDecision.DELEGATE.value == "delegate"
        assert HumanDecision.TERMINATE.value == "terminate"

    def test_人类决策_继承str和Enum(self) -> None:
        assert issubclass(HumanDecision, str)
        assert issubclass(HumanDecision, enum.Enum)

    # --- NegotiationPhase ---

    def test_协商阶段_四个值(self) -> None:
        assert len(NegotiationPhase) == 4

    def test_协商阶段_三阶段生命周期(self) -> None:
        expected = {"screening", "negotiation", "execution", "abandonment"}
        assert {p.value for p in NegotiationPhase} == expected

    def test_协商阶段_继承str和Enum(self) -> None:
        assert issubclass(NegotiationPhase, str)
        assert issubclass(NegotiationPhase, enum.Enum)

    # --- SwitchTrigger ---

    def test_切换触发_十个值(self) -> None:
        assert len(SwitchTrigger) == 10

    def test_切换触发_字符串值(self) -> None:
        assert SwitchTrigger.LOW_CONFIDENCE.value == "low_confidence"
        assert SwitchTrigger.ANOMALY_DETECTED.value == "anomaly_detected"
        assert SwitchTrigger.SUSTAINED_QUALITY.value == "sustained_quality"
        assert SwitchTrigger.SCHEDULED.value == "scheduled"

    def test_切换触发_继承str和Enum(self) -> None:
        assert issubclass(SwitchTrigger, str)
        assert issubclass(SwitchTrigger, enum.Enum)

    # --- ReviewOutcome ---

    def test_审核结论_四个值(self) -> None:
        assert len(ReviewOutcome) == 4

    def test_审核结论_字符串值(self) -> None:
        assert ReviewOutcome.ACCEPTED.value == "accepted"
        assert ReviewOutcome.REVISED.value == "revised"
        assert ReviewOutcome.REJECTED.value == "rejected"
        assert ReviewOutcome.ESCALATED.value == "escalated"

    def test_审核结论_继承str和Enum(self) -> None:
        assert issubclass(ReviewOutcome, str)
        assert issubclass(ReviewOutcome, enum.Enum)


# ============================================================
# 2. 测试 REACTScore
# ============================================================


class TestREACTScore:
    """验证 REACT 五维评分的均分计算和四级模式映射."""

    def test_默认值_各维度均为3(self) -> None:
        score = REACTScore()
        assert score.risk == 3.0
        assert score.explainability == 3.0
        assert score.accuracy == 3.0
        assert score.consequence == 3.0
        assert score.time_sensitivity == 3.0

    def test_均分计算_满分(self) -> None:
        score = REACTScore(
            risk=5, explainability=5, accuracy=5, consequence=5, time_sensitivity=5,
        )
        assert score.average() == 5.0

    def test_均分计算_零分(self) -> None:
        score = REACTScore(
            risk=0, explainability=0, accuracy=0, consequence=0, time_sensitivity=0,
        )
        assert score.average() == 0.0

    def test_均分计算_精度保留三位(self) -> None:
        score = REACTScore(
            risk=1, explainability=2, accuracy=3, consequence=4, time_sensitivity=5,
        )
        assert score.average() == 3.0

    def test_均分计算_非整数(self) -> None:
        score = REACTScore(
            risk=1.5, explainability=2.5, accuracy=3.0, consequence=3.5, time_sensitivity=4.0,
        )
        assert score.average() == pytest.approx(2.9, abs=0.001)

    # --- 四级映射 ---

    def test_映射_均分低于1_5返回AUTONOMOUS(self) -> None:
        score = REACTScore(
            risk=0, explainability=0, accuracy=1, consequence=1, time_sensitivity=2,
        )
        assert score.average() < 1.5
        assert score.to_mode() == CollaborationMode.AUTONOMOUS

    def test_映射_均分1_5至2_5返回MONITORED(self) -> None:
        score = REACTScore(
            risk=2, explainability=2, accuracy=2, consequence=2, time_sensitivity=2,
        )
        assert 1.5 <= score.average() < 2.5
        assert score.to_mode() == CollaborationMode.MONITORED

    def test_映射_均分2_5至3_5返回CONDITIONAL(self) -> None:
        score = REACTScore(
            risk=3, explainability=3, accuracy=3, consequence=3, time_sensitivity=3,
        )
        assert 2.5 <= score.average() < 3.5
        assert score.to_mode() == CollaborationMode.CONDITIONAL

    def test_映射_均分大于等于3_5返回SUPERVISED(self) -> None:
        score = REACTScore(
            risk=4, explainability=4, accuracy=4, consequence=4, time_sensitivity=4,
        )
        assert score.average() >= 3.5
        assert score.to_mode() == CollaborationMode.SUPERVISED

    def test_映射_边界值恰好1_5(self) -> None:
        """均分恰好 1.5 时应返回 MONITORED（>=1.5 不满足 <1.5）."""
        score = REACTScore(
            risk=1.5, explainability=1.5, accuracy=1.5, consequence=1.5, time_sensitivity=1.5,
        )
        assert score.average() == 1.5
        assert score.to_mode() == CollaborationMode.MONITORED

    def test_映射_边界值恰好2_5(self) -> None:
        score = REACTScore(
            risk=2.5, explainability=2.5, accuracy=2.5, consequence=2.5, time_sensitivity=2.5,
        )
        assert score.average() == 2.5
        assert score.to_mode() == CollaborationMode.CONDITIONAL

    def test_映射_边界值恰好3_5(self) -> None:
        score = REACTScore(
            risk=3.5, explainability=3.5, accuracy=3.5, consequence=3.5, time_sensitivity=3.5,
        )
        assert score.average() == 3.5
        assert score.to_mode() == CollaborationMode.SUPERVISED

    # --- 验证约束 ---

    def test_维度值超出范围_负值抛异常(self) -> None:
        with pytest.raises(Exception):
            REACTScore(risk=-1)

    def test_维度值超出范围_超过5抛异常(self) -> None:
        with pytest.raises(Exception):
            REACTScore(risk=6)


# ============================================================
# 3. 测试 AgentCollaborationProfile
# ============================================================


class TestAgentCollaborationProfile:
    """验证 Agent 协作配置的创建、步数管理和覆盖记录."""

    def test_默认配置_CONDITIONAL模式(self) -> None:
        p = _make_profile()
        assert p.mode == CollaborationMode.CONDITIONAL
        assert p.default_mode == CollaborationMode.CONDITIONAL

    def test_默认配置_最大自主步数10(self) -> None:
        p = _make_profile()
        assert p.max_auto_steps == 10

    def test_默认配置_置信度阈值0_7(self) -> None:
        p = _make_profile()
        assert p.confidence_threshold == 0.7

    def test_默认配置_超时300秒(self) -> None:
        p = _make_profile()
        assert p.timeout_seconds == 300.0

    def test_默认配置_启用状态(self) -> None:
        p = _make_profile()
        assert p.enabled is True

    def test_创建指定AUTONOMOUS模式(self) -> None:
        p = _make_profile(mode=CollaborationMode.AUTONOMOUS)
        assert p.mode == CollaborationMode.AUTONOMOUS

    def test_创建指定SUPERVISED模式(self) -> None:
        p = _make_profile(mode=CollaborationMode.SUPERVISED)
        assert p.mode == CollaborationMode.SUPERVISED

    # --- 步数管理 ---

    def test_初始步数为零(self) -> None:
        p = _make_profile()
        assert p.auto_step_count == 0

    def test_重置自主步数(self) -> None:
        p = _make_profile()
        p.auto_step_count = 5
        p.reset_auto_steps()
        assert p.auto_step_count == 0

    def test_递增自主步数_未达上限返回False(self) -> None:
        p = _make_profile(max_auto_steps=5)
        assert p.increment_auto_step() is False
        assert p.auto_step_count == 1

    def test_递增自主步数_恰好达上限返回True(self) -> None:
        p = _make_profile(max_auto_steps=3)
        p.increment_auto_step()
        p.increment_auto_step()
        result = p.increment_auto_step()
        assert result is True
        assert p.auto_step_count == 3

    def test_递增自主步数_超过上限也返回True(self) -> None:
        p = _make_profile(max_auto_steps=2)
        p.increment_auto_step()
        p.increment_auto_step()
        result = p.increment_auto_step()
        assert result is True
        assert p.auto_step_count == 3

    def test_最大步数为0时永不达上限(self) -> None:
        """max_auto_steps=0 表示无限自主步数."""
        p = _make_profile(max_auto_steps=0)
        for _ in range(100):
            assert p.increment_auto_step() is False

    # --- 覆盖记录 ---

    def test_初始覆盖次数为零(self) -> None:
        p = _make_profile()
        assert p.override_count == 0

    def test_记录覆盖_递增(self) -> None:
        p = _make_profile()
        p.record_override()
        assert p.override_count == 1
        p.record_override()
        assert p.override_count == 2

    def test_升级目标列表默认为空(self) -> None:
        p = _make_profile()
        assert p.escalation_targets == []

    def test_标签默认为空(self) -> None:
        p = _make_profile()
        assert p.tags == []

    def test_创建时指定升级目标(self) -> None:
        p = _make_profile(escalation_targets=["supervisor-001", "admin-001"])
        assert len(p.escalation_targets) == 2


# ============================================================
# 4. 测试干预请求模型
# ============================================================


class Test干预请求模型:
    """验证 InterventionRequest、HumanResponse、InterventionRecord 模型."""

    # --- InterventionRequest ---

    def test_创建干预请求_默认值(self) -> None:
        req = InterventionRequest(agent_id="tutor-001")
        assert req.agent_id == "tutor-001"
        assert req.intervention_type == InterventionType.CHECKPOINT
        assert req.confidence == 0.5
        assert req.priority == 50
        assert req.payload == {}

    def test_创建干预请求_指定类型(self) -> None:
        req = InterventionRequest(
            agent_id="tutor-001",
            intervention_type=InterventionType.ESCALATION,
        )
        assert req.intervention_type == InterventionType.ESCALATION

    def test_创建干预请求_带载荷(self) -> None:
        payload = {"student_answer": "A", "proposed_score": 85}
        req = InterventionRequest(
            agent_id="tutor-001",
            payload=payload,
            proposed_action="提交评分",
        )
        assert req.payload["student_answer"] == "A"
        assert req.proposed_action == "提交评分"

    def test_创建干预请求_自动生成ID(self) -> None:
        req = InterventionRequest(agent_id="tutor-001")
        assert req.request_id.startswith("intv-")

    def test_创建干预请求_指定置信度和优先级(self) -> None:
        req = InterventionRequest(
            agent_id="tutor-001",
            confidence=0.95,
            priority=90,
        )
        assert req.confidence == 0.95
        assert req.priority == 90

    # --- HumanResponse ---

    def test_创建人类响应_批准决策(self) -> None:
        resp = HumanResponse(
            request_id="req-001",
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
        )
        assert resp.decision == HumanDecision.APPROVE
        assert resp.feedback == ""

    def test_创建人类响应_拒绝带反馈(self) -> None:
        resp = HumanResponse(
            request_id="req-001",
            human_id="teacher-001",
            decision=HumanDecision.REJECT,
            feedback="评分过高",
        )
        assert resp.decision == HumanDecision.REJECT
        assert resp.feedback == "评分过高"

    def test_创建人类响应_修改动作(self) -> None:
        resp = HumanResponse(
            request_id="req-001",
            human_id="teacher-001",
            decision=HumanDecision.MODIFY,
            modified_action="调整评分为 80",
        )
        assert resp.modified_action == "调整评分为 80"

    def test_创建人类响应_反提案(self) -> None:
        resp = HumanResponse(
            request_id="req-001",
            human_id="teacher-001",
            decision=HumanDecision.COUNTEROFFER,
            counteroffer={"score": 82, "reason": "综合考虑"},
        )
        assert resp.counteroffer["score"] == 82

    def test_创建人类响应_委托(self) -> None:
        resp = HumanResponse(
            request_id="req-001",
            human_id="teacher-001",
            decision=HumanDecision.DELEGATE,
            delegate_target="senior-teacher-001",
        )
        assert resp.delegate_target == "senior-teacher-001"

    def test_创建人类响应_自动生成ID(self) -> None:
        resp = HumanResponse(
            request_id="req-001",
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
        )
        assert resp.request_id == "req-001"
        assert resp.response_id.startswith("hresp-")

    # --- InterventionRecord ---

    def test_创建干预记录_默认PENDING状态(self) -> None:
        req = InterventionRequest(agent_id="tutor-001")
        record = InterventionRecord(request=req)
        assert record.status == InterventionStatus.PENDING
        assert record.response is None

    def test_解决干预记录(self) -> None:
        req = InterventionRequest(agent_id="tutor-001")
        record = InterventionRecord(request=req)
        resp = HumanResponse(
            request_id=req.request_id,
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
        )
        record.resolve(resp, "教师批准")
        assert record.status == InterventionStatus.RESOLVED
        assert record.response is not None
        assert record.resolution_summary == "教师批准"
        assert record.resolved_at is not None
        assert record.duration_seconds >= 0

    def test_解决干预记录_自动生成摘要(self) -> None:
        req = InterventionRequest(agent_id="tutor-001")
        record = InterventionRecord(request=req)
        resp = HumanResponse(
            request_id=req.request_id,
            human_id="teacher-001",
            decision=HumanDecision.REJECT,
            feedback="理由充分",
        )
        record.resolve(resp)
        assert record.resolution_summary == "reject: 理由充分"

    def test_过期干预记录(self) -> None:
        req = InterventionRequest(agent_id="tutor-001")
        record = InterventionRecord(request=req)
        record.expire()
        assert record.status == InterventionStatus.EXPIRED
        assert record.resolved_at is not None

    def test_取消干预记录(self) -> None:
        req = InterventionRequest(agent_id="tutor-001")
        record = InterventionRecord(request=req)
        record.cancel()
        assert record.status == InterventionStatus.CANCELLED
        assert record.resolved_at is not None


# ============================================================
# 5. 测试协商模型
# ============================================================


class Test协商模型:
    """验证 NegotiationRound 和 NegotiationSession 模型."""

    # --- NegotiationRound ---

    def test_创建协商回合_默认值(self) -> None:
        r = NegotiationRound(round_number=1, proposer="agent")
        assert r.round_number == 1
        assert r.proposer == "agent"
        assert r.confidence == 0.5
        assert r.reasoning == ""
        assert r.proposal == {}

    def test_创建协商回合_指定内容(self) -> None:
        r = NegotiationRound(
            round_number=2,
            proposer="human",
            proposal={"adjusted_score": 80},
            confidence=0.85,
            reasoning="考虑到学生进步趋势",
        )
        assert r.proposer == "human"
        assert r.proposal["adjusted_score"] == 80
        assert r.confidence == 0.85

    def test_协商回合_proposer只能是agent或human(self) -> None:
        NegotiationRound(round_number=1, proposer="agent")
        NegotiationRound(round_number=1, proposer="human")
        # 非法值由 Pydantic 校验
        with pytest.raises(Exception):
            NegotiationRound(round_number=1, proposer="system")  # type: ignore[arg-type]

    # --- NegotiationSession ---

    def test_创建协商会话_默认SCREENING阶段(self) -> None:
        session = NegotiationSession(agent_id="tutor-001", human_id="teacher-001")
        assert session.phase == NegotiationPhase.SCREENING
        assert session.status == InterventionStatus.ACTIVE
        assert session.current_round == 0
        assert session.rounds == []

    def test_添加第一轮协商_自动推进到NEGOTIATION(self) -> None:
        session = NegotiationSession(agent_id="tutor-001", human_id="teacher-001")
        session.add_round(
            proposer="agent",
            proposal={"score": 85},
            confidence=0.8,
            reasoning="基于答案分析",
        )
        assert session.current_round == 1
        assert session.phase == NegotiationPhase.NEGOTIATION

    def test_添加多轮协商(self) -> None:
        session = NegotiationSession(agent_id="tutor-001", human_id="teacher-001", max_rounds=5)
        session.add_round("agent", {"score": 85}, 0.8)
        session.add_round("human", {"score": 80}, 0.9, "调低一些")
        session.add_round("agent", {"score": 82}, 0.85, "折中方案")
        assert session.current_round == 3

    def test_协商是否耗尽_未达上限(self) -> None:
        session = NegotiationSession(agent_id="tutor-001", human_id="teacher-001", max_rounds=5)
        session.add_round("agent", {"score": 85})
        assert session.is_exhausted is False

    def test_协商是否耗尽_恰好达上限(self) -> None:
        session = NegotiationSession(agent_id="tutor-001", human_id="teacher-001", max_rounds=3)
        session.add_round("agent", {"score": 85})
        session.add_round("human", {"score": 80})
        session.add_round("agent", {"score": 82})
        assert session.is_exhausted is True

    def test_结束协商_批准进入EXECUTION(self) -> None:
        session = NegotiationSession(agent_id="tutor-001", human_id="teacher-001")
        session.add_round("agent", {"score": 85})
        session.finalize(HumanDecision.APPROVE)
        assert session.phase == NegotiationPhase.EXECUTION
        assert session.status == InterventionStatus.RESOLVED
        assert session.final_decision == HumanDecision.APPROVE

    def test_结束协商_跳过也进入EXECUTION(self) -> None:
        session = NegotiationSession(agent_id="tutor-001", human_id="teacher-001")
        session.finalize(HumanDecision.SKIP)
        assert session.phase == NegotiationPhase.EXECUTION

    def test_结束协商_拒绝进入ABANDONMENT(self) -> None:
        session = NegotiationSession(agent_id="tutor-001", human_id="teacher-001")
        session.add_round("agent", {"score": 85})
        session.finalize(HumanDecision.REJECT)
        assert session.phase == NegotiationPhase.ABANDONMENT
        assert session.status == InterventionStatus.RESOLVED

    def test_结束协商_终止也进入ABANDONMENT(self) -> None:
        session = NegotiationSession(agent_id="tutor-001", human_id="teacher-001")
        session.finalize(HumanDecision.TERMINATE)
        assert session.phase == NegotiationPhase.ABANDONMENT

    def test_协商会话_自动生成ID(self) -> None:
        session = NegotiationSession(agent_id="tutor-001", human_id="teacher-001")
        assert session.session_id.startswith("nego-")


# ============================================================
# 6. 测试 ModeSwitchEvent
# ============================================================


class TestModeSwitchEvent:
    """验证模式切换事件的创建和 is_upgrade 属性."""

    def test_创建切换事件_基本信息(self) -> None:
        event = ModeSwitchEvent(
            agent_id="tutor-001",
            from_mode=CollaborationMode.MONITORED,
            to_mode=CollaborationMode.CONDITIONAL,
            trigger=SwitchTrigger.SUSTAINED_QUALITY,
        )
        assert event.from_mode == CollaborationMode.MONITORED
        assert event.to_mode == CollaborationMode.CONDITIONAL
        assert event.trigger == SwitchTrigger.SUSTAINED_QUALITY

    def test_创建切换事件_带REACT评分(self) -> None:
        score = _make_react_score(risk=4, explainability=4, accuracy=4, consequence=4, time_sensitivity=4)
        event = ModeSwitchEvent(
            agent_id="tutor-001",
            from_mode=CollaborationMode.CONDITIONAL,
            to_mode=CollaborationMode.SUPERVISED,
            trigger=SwitchTrigger.LOW_CONFIDENCE,
            react_score=score,
        )
        assert event.react_score is not None
        assert event.react_score.average() == 4.0

    def test_创建切换事件_带置信度(self) -> None:
        event = ModeSwitchEvent(
            agent_id="tutor-001",
            from_mode=CollaborationMode.CONDITIONAL,
            to_mode=CollaborationMode.AUTONOMOUS,
            trigger=SwitchTrigger.PERMISSION_GRANTED,
            confidence_at_time=0.98,
        )
        assert event.confidence_at_time == 0.98

    def test_创建切换事件_自动生成ID(self) -> None:
        event = ModeSwitchEvent(
            agent_id="tutor-001",
            from_mode=CollaborationMode.CONDITIONAL,
            to_mode=CollaborationMode.SUPERVISED,
            trigger=SwitchTrigger.MANUAL_REQUEST,
        )
        assert event.event_id.startswith("sw-")

    # --- is_upgrade 属性 ---

    def test_is_upgrade_降级返回False(self) -> None:
        """SUPERVISED → CONDITIONAL 是降级（更多自主权）."""
        event = ModeSwitchEvent(
            agent_id="tutor-001",
            from_mode=CollaborationMode.SUPERVISED,
            to_mode=CollaborationMode.CONDITIONAL,
            trigger=SwitchTrigger.SUSTAINED_QUALITY,
        )
        assert event.is_upgrade is False

    def test_is_upgrade_升级返回True(self) -> None:
        """CONDITIONAL → SUPERVISED 是升级（更多人类控制）."""
        event = ModeSwitchEvent(
            agent_id="tutor-001",
            from_mode=CollaborationMode.CONDITIONAL,
            to_mode=CollaborationMode.SUPERVISED,
            trigger=SwitchTrigger.LOW_CONFIDENCE,
        )
        assert event.is_upgrade is True

    def test_is_upgrade_同模式返回False(self) -> None:
        event = ModeSwitchEvent(
            agent_id="tutor-001",
            from_mode=CollaborationMode.MONITORED,
            to_mode=CollaborationMode.MONITORED,
            trigger=SwitchTrigger.MANUAL_REQUEST,
        )
        assert event.is_upgrade is False

    def test_is_upgrade_从AUTONOMOUS到SUPERVISED是升级(self) -> None:
        event = ModeSwitchEvent(
            agent_id="tutor-001",
            from_mode=CollaborationMode.AUTONOMOUS,
            to_mode=CollaborationMode.SUPERVISED,
            trigger=SwitchTrigger.CHAOS_DETECTED,
        )
        assert event.is_upgrade is True

    def test_is_upgrade_从CONDITIONAL到MONITORED是降级(self) -> None:
        event = ModeSwitchEvent(
            agent_id="tutor-001",
            from_mode=CollaborationMode.CONDITIONAL,
            to_mode=CollaborationMode.MONITORED,
            trigger=SwitchTrigger.NO_OVERRIDE,
        )
        assert event.is_upgrade is False


# ============================================================
# 7. 测试 CollaborationConfig
# ============================================================


class TestCollaborationConfig:
    """验证全局协作配置的默认值与字段约束."""

    def test_默认模式为CONDITIONAL(self) -> None:
        cfg = CollaborationConfig()
        assert cfg.default_mode == CollaborationMode.CONDITIONAL

    def test_最大待处理干预数默认10(self) -> None:
        cfg = CollaborationConfig()
        assert cfg.max_pending_interventions == 10

    def test_干预超时默认300秒(self) -> None:
        cfg = CollaborationConfig()
        assert cfg.intervention_timeout_seconds == 300.0

    def test_最大协商轮次默认5(self) -> None:
        cfg = CollaborationConfig()
        assert cfg.max_negotiation_rounds == 5

    def test_协商默认启用(self) -> None:
        cfg = CollaborationConfig()
        assert cfg.enable_negotiation is True

    def test_自动升级默认启用(self) -> None:
        cfg = CollaborationConfig()
        assert cfg.enable_auto_escalation is True

    def test_连续覆盖阈值默认5(self) -> None:
        cfg = CollaborationConfig()
        assert cfg.consecutive_override_threshold == 5

    def test_持续质量窗口默认3600秒(self) -> None:
        cfg = CollaborationConfig()
        assert cfg.sustained_quality_window == 3600.0

    def test_持续质量阈值默认0_95(self) -> None:
        cfg = CollaborationConfig()
        assert cfg.sustained_quality_threshold == 0.95

    def test_自定义配置(self) -> None:
        cfg = CollaborationConfig(
            default_mode=CollaborationMode.SUPERVISED,
            max_pending_interventions=20,
            enable_negotiation=False,
        )
        assert cfg.default_mode == CollaborationMode.SUPERVISED
        assert cfg.max_pending_interventions == 20
        assert cfg.enable_negotiation is False


# ============================================================
# 8. 测试异常体系
# ============================================================


class Test异常体系:
    """验证 7 个异常类的 JSON-RPC 码、继承层级和上下文."""

    def test_CC2Error_基础属性(self) -> None:
        err = CC2Error(detail="基础错误测试")
        assert err.code == "CC2_ERROR"
        assert err.detail == "基础错误测试"

    def test_CC2Error_jsonrpc码为负32300(self) -> None:
        err = CC2Error()
        assert err._jsonrpc_code() == -32300

    def test_CC2Error_继承L6Error(self) -> None:
        assert issubclass(CC2Error, L6Error)
        assert issubclass(CC2Error, Exception)

    def test_CC2Error_上下文信息(self) -> None:
        err = CC2Error(detail="测试", context={"key": "value"})
        assert err.context == {"key": "value"}

    def test_CC2Error_to_json_rpc_error(self) -> None:
        err = CC2Error(detail="测试详情")
        rpc = err.to_json_rpc_error()
        assert rpc["code"] == -32300
        assert rpc["message"] == "CC2_ERROR"

    # --- 子类异常 ---

    def test_InterventionTimeoutError_jsonrpc码负32301(self) -> None:
        err = InterventionTimeoutError("req-001", 300)
        assert err._jsonrpc_code() == -32301
        assert err.code == "CC2_INTERVENTION_TIMEOUT"
        assert err.request_id == "req-001"
        assert err.timeout_seconds == 300

    def test_NegotiationExhaustedError_jsonrpc码负32302(self) -> None:
        err = NegotiationExhaustedError("nego-001", 5, 5)
        assert err._jsonrpc_code() == -32302
        assert err.rounds == 5
        assert err.max_rounds == 5

    def test_ProfileNotFoundError_jsonrpc码负32303(self) -> None:
        err = ProfileNotFoundError("agent-999")
        assert err._jsonrpc_code() == -32303
        assert err.agent_id == "agent-999"

    def test_ModeSwitchError_jsonrpc码负32304(self) -> None:
        err = ModeSwitchError("agent-001", "monitored", "supervised")
        assert err._jsonrpc_code() == -32304
        assert err.from_mode == "monitored"
        assert err.to_mode == "supervised"

    def test_InterventionConflictError_jsonrpc码负32305(self) -> None:
        err = InterventionConflictError("req-001", "已解决")
        assert err._jsonrpc_code() == -32305
        assert err.request_id == "req-001"

    def test_EscalationTargetError_jsonrpc码负32306(self) -> None:
        err = EscalationTargetError("unknown-target", "agent-001")
        assert err._jsonrpc_code() == -32306
        assert err.target == "unknown-target"

    # --- 继承关系 ---

    def test_所有CC2异常继承CC2Error(self) -> None:
        assert issubclass(InterventionTimeoutError, CC2Error)
        assert issubclass(NegotiationExhaustedError, CC2Error)
        assert issubclass(ProfileNotFoundError, CC2Error)
        assert issubclass(ModeSwitchError, CC2Error)
        assert issubclass(InterventionConflictError, CC2Error)
        assert issubclass(EscalationTargetError, CC2Error)

    def test_所有CC2异常间接继承L6Error(self) -> None:
        for cls in [
            InterventionTimeoutError,
            NegotiationExhaustedError,
            ProfileNotFoundError,
            ModeSwitchError,
            InterventionConflictError,
            EscalationTargetError,
        ]:
            assert issubclass(cls, L6Error), f"{cls.__name__} 不继承 L6Error"

    def test_异常均可被捕获为L6Error(self) -> None:
        for cls, args in [
            (InterventionTimeoutError, ("req-001", 300)),
            (NegotiationExhaustedError, ("nego-001", 5, 5)),
            (ProfileNotFoundError, ("agent-001",)),
            (ModeSwitchError, ("agent-001", "m1", "m2")),
            (InterventionConflictError, ("req-001",)),
            (EscalationTargetError, ("target-001",)),
        ]:
            try:
                raise cls(*args)  # type: ignore[arg-type]
            except L6Error:
                pass  # 预期行为

    def test_异常字符串表示包含错误码(self) -> None:
        err = CC2Error(detail="详细信息")
        assert "CC2_ERROR" in str(err)


# ============================================================
# 9. 测试 CollaborationEngine_配置管理
# ============================================================


class TestCollaborationEngine_配置管理:
    """验证引擎的 register/get/update/list 配置管理."""

    def test_注册配置(self) -> None:
        engine = _make_engine(register_agent=None)
        profile = _make_profile("tutor-001")
        engine.register_profile(profile)
        assert engine.get_profile("tutor-001").agent_id == "tutor-001"

    def test_注册多个配置(self) -> None:
        engine = _make_engine(register_agent=None)
        engine.register_profile(_make_profile("agent-a"))
        engine.register_profile(_make_profile("agent-b"))
        engine.register_profile(_make_profile("agent-c"))
        profiles = engine.list_profiles()
        assert len(profiles) == 3

    def test_获取不存在的配置抛异常(self) -> None:
        engine = _make_engine(register_agent=None)
        with pytest.raises(ProfileNotFoundError):
            engine.get_profile("nonexistent")

    def test_更新配置_mode字段(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.MONITORED)
        assert engine.get_profile("agent-001").mode == CollaborationMode.MONITORED

    def test_更新配置_置信度阈值(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", confidence_threshold=0.9)
        assert engine.get_profile("agent-001").confidence_threshold == 0.9

    def test_更新配置_最大自主步数(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", max_auto_steps=20)
        assert engine.get_profile("agent-001").max_auto_steps == 20

    def test_更新不存在的配置抛异常(self) -> None:
        engine = _make_engine()
        with pytest.raises(ProfileNotFoundError):
            engine.update_profile("nonexistent", mode=CollaborationMode.AUTONOMOUS)

    def test_列出配置_空引擎返回空列表(self) -> None:
        engine = _make_engine(register_agent=None)
        assert engine.list_profiles() == []

    def test_重复注册覆盖旧配置(self) -> None:
        engine = _make_engine(register_agent=None)
        engine.register_profile(_make_profile("agent-001", mode=CollaborationMode.AUTONOMOUS))
        engine.register_profile(_make_profile("agent-001", mode=CollaborationMode.SUPERVISED))
        assert engine.get_profile("agent-001").mode == CollaborationMode.SUPERVISED


# ============================================================
# 10. 测试 CollaborationEngine_REACT评分
# ============================================================


class TestCollaborationEngine_REACT评分:
    """验证引擎的 evaluate_react 方法."""

    def test_evaluate_react_低分返回AUTONOMOUS(self) -> None:
        engine = _make_engine()
        score = _make_react_score(risk=0, explainability=0, accuracy=0, consequence=0, time_sensitivity=1)
        mode = engine.evaluate_react("agent-001", score)
        assert mode == CollaborationMode.AUTONOMOUS

    def test_evaluate_react_中等分返回MONITORED(self) -> None:
        engine = _make_engine()
        score = _make_react_score(risk=2, explainability=2, accuracy=2, consequence=2, time_sensitivity=2)
        mode = engine.evaluate_react("agent-001", score)
        assert mode == CollaborationMode.MONITORED

    def test_evaluate_react_中高分返回CONDITIONAL(self) -> None:
        engine = _make_engine()
        score = _make_react_score(risk=3, explainability=3, accuracy=3, consequence=3, time_sensitivity=3)
        mode = engine.evaluate_react("agent-001", score)
        assert mode == CollaborationMode.CONDITIONAL

    def test_evaluate_react_高分返回SUPERVISED(self) -> None:
        engine = _make_engine()
        score = _make_react_score(risk=5, explainability=5, accuracy=5, consequence=5, time_sensitivity=5)
        mode = engine.evaluate_react("agent-001", score)
        assert mode == CollaborationMode.SUPERVISED

    def test_evaluate_react_不依赖已注册Agent(self) -> None:
        """evaluate_react 仅依赖评分本身，不要求 Agent 已注册."""
        engine = _make_engine(register_agent=None)
        score = _make_react_score(risk=1, explainability=1, accuracy=1, consequence=1, time_sensitivity=1)
        mode = engine.evaluate_react("nonexistent-agent", score)
        assert mode == CollaborationMode.AUTONOMOUS


# ============================================================
# 11. 测试 CollaborationEngine_模式切换
# ============================================================


class TestCollaborationEngine_模式切换:
    """验证引擎的 switch_mode 方法：相邻级、跳级、同模式."""

    def test_相邻级切换_MONITORED到CONDITIONAL(self) -> None:
        engine = _make_engine()
        profile = engine.get_profile("agent-001")
        engine.update_profile("agent-001", mode=CollaborationMode.MONITORED)
        event = engine.switch_mode(
            agent_id="agent-001",
            to_mode=CollaborationMode.CONDITIONAL,
            trigger=SwitchTrigger.SUSTAINED_QUALITY,
            reason="持续质量达标",
        )
        assert event.from_mode == CollaborationMode.MONITORED
        assert event.to_mode == CollaborationMode.CONDITIONAL
        assert engine.get_profile("agent-001").mode == CollaborationMode.CONDITIONAL

    def test_相邻级切换_CONDITIONAL到SUPERVISED(self) -> None:
        engine = _make_engine()
        event = engine.switch_mode(
            agent_id="agent-001",
            to_mode=CollaborationMode.SUPERVISED,
            trigger=SwitchTrigger.LOW_CONFIDENCE,
        )
        assert event.from_mode == CollaborationMode.CONDITIONAL
        assert event.to_mode == CollaborationMode.SUPERVISED
        assert engine.get_profile("agent-001").mode == CollaborationMode.SUPERVISED

    def test_相邻级切换_AUTONOMOUS到MONITORED(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.AUTONOMOUS)
        event = engine.switch_mode(
            agent_id="agent-001",
            to_mode=CollaborationMode.MONITORED,
            trigger=SwitchTrigger.ANOMALY_DETECTED,
        )
        assert event.to_mode == CollaborationMode.MONITORED

    def test_跳级切换_默认不允许抛异常(self) -> None:
        """AUTONOMOUS → SUPERVISED 跨两级，默认应抛 ModeSwitchError."""
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.AUTONOMOUS)
        with pytest.raises(ModeSwitchError):
            engine.switch_mode(
                agent_id="agent-001",
                to_mode=CollaborationMode.SUPERVISED,
                trigger=SwitchTrigger.CHAOS_DETECTED,
            )

    def test_跳级切换_allow_skip允许(self) -> None:
        """allow_skip=True 时允许跳级."""
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.AUTONOMOUS)
        event = engine.switch_mode(
            agent_id="agent-001",
            to_mode=CollaborationMode.SUPERVISED,
            trigger=SwitchTrigger.CHAOS_DETECTED,
            reason="混沌感知紧急升级",
            allow_skip=True,
        )
        assert event.from_mode == CollaborationMode.AUTONOMOUS
        assert event.to_mode == CollaborationMode.SUPERVISED
        assert engine.get_profile("agent-001").mode == CollaborationMode.SUPERVISED

    def test_同模式切换_返回事件但不改变profile(self) -> None:
        engine = _make_engine()
        event = engine.switch_mode(
            agent_id="agent-001",
            to_mode=CollaborationMode.CONDITIONAL,
            trigger=SwitchTrigger.MANUAL_REQUEST,
        )
        assert event.from_mode == CollaborationMode.CONDITIONAL
        assert event.to_mode == CollaborationMode.CONDITIONAL
        assert engine.get_profile("agent-001").mode == CollaborationMode.CONDITIONAL

    def test_未注册Agent切换抛异常(self) -> None:
        engine = _make_engine(register_agent=None)
        with pytest.raises(ProfileNotFoundError):
            engine.switch_mode(
                agent_id="nonexistent",
                to_mode=CollaborationMode.SUPERVISED,
                trigger=SwitchTrigger.LOW_CONFIDENCE,
            )

    def test_切换事件记录到引擎(self) -> None:
        engine = _make_engine()
        engine.switch_mode(
            agent_id="agent-001",
            to_mode=CollaborationMode.SUPERVISED,
            trigger=SwitchTrigger.MANUAL_REQUEST,
        )
        events = engine.get_switch_events("agent-001")
        assert len(events) == 1

    def test_连续切换事件顺序(self) -> None:
        engine = _make_engine()
        engine.switch_mode("agent-001", CollaborationMode.SUPERVISED, SwitchTrigger.LOW_CONFIDENCE)
        engine.switch_mode("agent-001", CollaborationMode.CONDITIONAL, SwitchTrigger.SUSTAINED_QUALITY)
        engine.switch_mode("agent-001", CollaborationMode.MONITORED, SwitchTrigger.NO_OVERRIDE)
        events = engine.get_switch_events("agent-001")
        assert len(events) == 3


# ============================================================
# 12. 测试 CollaborationEngine_干预请求
# ============================================================


class TestCollaborationEngine_干预请求:
    """验证引擎的 create/respond/expire/cancel 干预管理."""

    # --- create_intervention ---

    def test_创建干预_返回PENDING记录(self) -> None:
        engine = _make_engine()
        record = engine.create_intervention(
            agent_id="agent-001",
            intervention_type=InterventionType.CHECKPOINT,
            reason="关键决策需审批",
        )
        assert record.status == InterventionStatus.PENDING

    def test_创建干预_载荷和动作正确(self) -> None:
        engine = _make_engine()
        record = engine.create_intervention(
            agent_id="agent-001",
            payload={"score": 85},
            proposed_action="提交评分",
            confidence=0.8,
        )
        assert record.request.payload == {"score": 85}
        assert record.request.proposed_action == "提交评分"
        assert record.request.confidence == 0.8

    def test_创建干预_可通过ID查询(self) -> None:
        engine = _make_engine()
        record = engine.create_intervention(agent_id="agent-001")
        fetched = engine.get_intervention(record.request.request_id)
        assert fetched is not None
        assert fetched.request.request_id == record.request.request_id

    def test_创建干预_使用配置超时(self) -> None:
        engine = _make_engine(config=CollaborationConfig(intervention_timeout_seconds=600))
        record = engine.create_intervention(agent_id="agent-001")
        assert record.request.timeout_seconds == 600

    def test_创建干预_多种类型(self) -> None:
        engine = _make_engine()
        for itype in InterventionType:
            record = engine.create_intervention(
                agent_id="agent-001",
                intervention_type=itype,
            )
            assert record.request.intervention_type == itype

    # --- respond_to_intervention ---

    def test_响应干预_批准(self) -> None:
        engine = _make_engine()
        record = engine.create_intervention(agent_id="agent-001")
        result = engine.respond_to_intervention(
            request_id=record.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
            feedback="批准执行",
        )
        assert result.status == InterventionStatus.RESOLVED
        assert result.response is not None
        assert result.response.decision == HumanDecision.APPROVE

    def test_响应干预_批准后重置自主步数(self) -> None:
        engine = _make_engine()
        profile = engine.get_profile("agent-001")
        profile.auto_step_count = 5
        record = engine.create_intervention(agent_id="agent-001")
        engine.respond_to_intervention(
            request_id=record.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
        )
        assert profile.auto_step_count == 0

    def test_响应干预_拒绝记录覆盖并重置步数(self) -> None:
        engine = _make_engine()
        profile = engine.get_profile("agent-001")
        profile.auto_step_count = 3
        initial_override = profile.override_count
        record = engine.create_intervention(agent_id="agent-001")
        engine.respond_to_intervention(
            request_id=record.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.REJECT,
            feedback="不同意",
        )
        assert profile.override_count == initial_override + 1
        assert profile.auto_step_count == 0

    def test_响应干预_修改也记录覆盖(self) -> None:
        engine = _make_engine()
        profile = engine.get_profile("agent-001")
        record = engine.create_intervention(agent_id="agent-001")
        engine.respond_to_intervention(
            request_id=record.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.MODIFY,
            modified_action="调整为80分",
        )
        assert profile.override_count == 1

    def test_响应不存在的干预抛异常(self) -> None:
        engine = _make_engine()
        with pytest.raises(InterventionConflictError):
            engine.respond_to_intervention(
                request_id="nonexistent",
                human_id="teacher-001",
                decision=HumanDecision.APPROVE,
            )

    def test_重复响应抛异常(self) -> None:
        engine = _make_engine()
        record = engine.create_intervention(agent_id="agent-001")
        engine.respond_to_intervention(
            request_id=record.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
        )
        with pytest.raises(InterventionConflictError):
            engine.respond_to_intervention(
                request_id=record.request.request_id,
                human_id="teacher-001",
                decision=HumanDecision.APPROVE,
            )

    def test_响应已过期的干预抛异常(self) -> None:
        engine = _make_engine()
        record = engine.create_intervention(agent_id="agent-001")
        engine.expire_intervention(record.request.request_id)
        with pytest.raises(InterventionConflictError):
            engine.respond_to_intervention(
                request_id=record.request.request_id,
                human_id="teacher-001",
                decision=HumanDecision.APPROVE,
            )

    # --- expire / cancel ---

    def test_过期干预_状态变为EXPIRED(self) -> None:
        engine = _make_engine()
        record = engine.create_intervention(agent_id="agent-001")
        result = engine.expire_intervention(record.request.request_id)
        assert result is not None
        assert result.status == InterventionStatus.EXPIRED

    def test_取消干预_状态变为CANCELLED(self) -> None:
        engine = _make_engine()
        record = engine.create_intervention(agent_id="agent-001")
        result = engine.cancel_intervention(record.request.request_id)
        assert result is not None
        assert result.status == InterventionStatus.CANCELLED

    def test_过期不存在的干预返回None(self) -> None:
        engine = _make_engine()
        result = engine.expire_intervention("nonexistent")
        assert result is None

    def test_取消已解决的干预返回None(self) -> None:
        engine = _make_engine()
        record = engine.create_intervention(agent_id="agent-001")
        engine.respond_to_intervention(
            request_id=record.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
        )
        result = engine.cancel_intervention(record.request.request_id)
        assert result is None


# ============================================================
# 13. 测试 CollaborationEngine_升级
# ============================================================


class TestCollaborationEngine_升级:
    """验证引擎的 escalate_to_human 升级功能."""

    def test_默认目标_创建升级干预(self) -> None:
        engine = _make_engine()
        record = engine.escalate_to_human(
            agent_id="agent-001",
            reason="检测到异常输出",
        )
        assert record.request.intervention_type == InterventionType.ESCALATION
        assert record.status == InterventionStatus.PENDING
        assert record.request.context.get("escalation_target") == "human_operator"

    def test_默认目标_高优先级(self) -> None:
        engine = _make_engine()
        record = engine.escalate_to_human(
            agent_id="agent-001",
            reason="紧急异常",
            priority=95,
        )
        assert record.request.priority == 95

    def test_自动升级启用_自动切SUPERVISED(self) -> None:
        engine = _make_engine(config=CollaborationConfig(enable_auto_escalation=True))
        engine.update_profile("agent-001", mode=CollaborationMode.MONITORED)
        engine.escalate_to_human(agent_id="agent-001", reason="混沌检测")
        assert engine.get_profile("agent-001").mode == CollaborationMode.SUPERVISED

    def test_自动升级启用_已是SUPERVISED不再切换(self) -> None:
        engine = _make_engine(config=CollaborationConfig(enable_auto_escalation=True))
        engine.update_profile("agent-001", mode=CollaborationMode.SUPERVISED)
        engine.escalate_to_human(agent_id="agent-001", reason="测试")
        # 没有新增切换事件（同模式不记录）
        events = engine.get_switch_events("agent-001")
        assert len(events) == 0

    def test_自动升级禁用_不切换模式(self) -> None:
        engine = _make_engine(config=CollaborationConfig(enable_auto_escalation=False))
        engine.update_profile("agent-001", mode=CollaborationMode.MONITORED)
        engine.escalate_to_human(agent_id="agent-001", reason="测试")
        assert engine.get_profile("agent-001").mode == CollaborationMode.MONITORED

    def test_自动升级_跨多级切到SUPERVISED(self) -> None:
        """混沌感知允许从 AUTONOMOUS 直接跳到 SUPERVISED."""
        engine = _make_engine(config=CollaborationConfig(enable_auto_escalation=True))
        engine.update_profile("agent-001", mode=CollaborationMode.AUTONOMOUS)
        engine.escalate_to_human(agent_id="agent-001", reason="混沌注入")
        assert engine.get_profile("agent-001").mode == CollaborationMode.SUPERVISED

    def test_自定义目标_已注册Agent不报错(self) -> None:
        engine = _make_engine()
        engine.register_profile(_make_profile("supervisor-001"))
        record = engine.escalate_to_human(
            agent_id="agent-001",
            reason="升级到 supervisor",
            target="supervisor-001",
        )
        assert record.request.context.get("escalation_target") == "supervisor-001"

    def test_自定义目标_未注册Agent抛异常(self) -> None:
        engine = _make_engine()
        with pytest.raises(EscalationTargetError):
            engine.escalate_to_human(
                agent_id="agent-001",
                reason="升级到不存在的目标",
                target="nonexistent-supervisor",
            )

    def test_升级_创建干预且可响应(self) -> None:
        engine = _make_engine()
        record = engine.escalate_to_human(agent_id="agent-001", reason="测试升级")
        resolved = engine.respond_to_intervention(
            request_id=record.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
        )
        assert resolved.status == InterventionStatus.RESOLVED


# ============================================================
# 14. 测试 CollaborationEngine_自主步数
# ============================================================


class TestCollaborationEngine_自主步数:
    """验证引擎的 check_auto_step 自主步数检查."""

    def test_SUPERVISED模式_每步创建干预(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.SUPERVISED)
        r1 = engine.check_auto_step("agent-001", confidence=0.9)
        assert r1 is not None
        assert r1.request.intervention_type == InterventionType.CHECKPOINT
        assert "每步审批" in r1.request.reason

    def test_置信度低于阈值_创建升级干预(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.CONDITIONAL, confidence_threshold=0.8)
        result = engine.check_auto_step("agent-001", confidence=0.5)
        assert result is not None
        assert result.request.intervention_type == InterventionType.ESCALATION

    def test_置信度达标_未达上限_不创建干预(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.CONDITIONAL, max_auto_steps=10)
        result = engine.check_auto_step("agent-001", confidence=0.9)
        assert result is None

    def test_达步数上限_创建CHECKPOINT干预(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.CONDITIONAL, max_auto_steps=3)
        # 前 2 步不触发
        assert engine.check_auto_step("agent-001", confidence=0.9) is None
        assert engine.check_auto_step("agent-001", confidence=0.9) is None
        # 第 3 步触发
        result = engine.check_auto_step("agent-001", confidence=0.9)
        assert result is not None
        assert result.request.intervention_type == InterventionType.CHECKPOINT
        assert "上限" in result.request.reason

    def test_未注册Agent返回None(self) -> None:
        engine = _make_engine(register_agent=None)
        result = engine.check_auto_step("nonexistent", confidence=0.9)
        assert result is None

    def test_禁用状态的Agent返回None(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", enabled=False)
        result = engine.check_auto_step("agent-001", confidence=0.1)
        assert result is None

    def test_AUTONOMOUS模式_置信度达标_不触发(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.AUTONOMOUS, max_auto_steps=100)
        result = engine.check_auto_step("agent-001", confidence=0.95)
        assert result is None

    def test_AUTONOMOUS模式_置信度低_仍触发升级(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.AUTONOMOUS, confidence_threshold=0.7)
        result = engine.check_auto_step("agent-001", confidence=0.3)
        assert result is not None
        assert result.request.intervention_type == InterventionType.ESCALATION


# ============================================================
# 15. 测试 CollaborationEngine_协商
# ============================================================


class TestCollaborationEngine_协商:
    """验证引擎的 start/add_round/finalize 协商管理."""

    def test_启动协商_初始提案后进入NEGOTIATION(self) -> None:
        """start_negotiation 会自动添加初始提案回合，阶段从 SCREENING 推进到 NEGOTIATION."""
        engine = _make_engine()
        session = engine.start_negotiation(
            agent_id="agent-001",
            human_id="teacher-001",
            topic="评分标准协商",
            initial_proposal={"score": 85},
            initial_confidence=0.8,
        )
        assert session.phase == NegotiationPhase.NEGOTIATION
        assert len(session.rounds) == 1

    def test_启动协商_自动推进到NEGOTIATION(self) -> None:
        engine = _make_engine()
        session = engine.start_negotiation(
            agent_id="agent-001",
            human_id="teacher-001",
            topic="评分标准",
            initial_proposal={"score": 85},
        )
        # 初始提案添加后，从 SCREENING 进入 NEGOTIATION
        assert session.phase == NegotiationPhase.NEGOTIATION

    def test_添加协商回合(self) -> None:
        engine = _make_engine()
        session = engine.start_negotiation(
            agent_id="agent-001",
            human_id="teacher-001",
            topic="协商主题",
            initial_proposal={"v": 1},
        )
        round_data = engine.add_negotiation_round(
            session_id=session.session_id,
            proposer="human",
            proposal={"v": 2},
            confidence=0.9,
            reasoning="人类调整",
        )
        assert round_data.round_number == 2
        assert round_data.proposer == "human"

    def test_添加协商回合_不存在的会话抛异常(self) -> None:
        engine = _make_engine()
        with pytest.raises(CC2Error):
            engine.add_negotiation_round(
                session_id="nonexistent",
                proposer="agent",
                proposal={"v": 1},
            )

    def test_协商轮次耗尽抛异常(self) -> None:
        engine = _make_engine()
        session = engine.start_negotiation(
            agent_id="agent-001",
            human_id="teacher-001",
            topic="协商",
            initial_proposal={"v": 1},
            max_rounds=2,
        )
        # 第 2 轮
        engine.add_negotiation_round(
            session_id=session.session_id,
            proposer="human",
            proposal={"v": 2},
        )
        # 第 3 轮应抛异常（已耗尽）
        with pytest.raises(NegotiationExhaustedError):
            engine.add_negotiation_round(
                session_id=session.session_id,
                proposer="agent",
                proposal={"v": 3},
            )

    def test_结束协商_批准(self) -> None:
        engine = _make_engine()
        session = engine.start_negotiation(
            agent_id="agent-001",
            human_id="teacher-001",
            topic="协商",
            initial_proposal={"v": 1},
        )
        result = engine.finalize_negotiation(
            session_id=session.session_id,
            decision=HumanDecision.APPROVE,
        )
        assert result.final_decision == HumanDecision.APPROVE
        assert result.phase == NegotiationPhase.EXECUTION

    def test_结束协商_拒绝(self) -> None:
        engine = _make_engine()
        session = engine.start_negotiation(
            agent_id="agent-001",
            human_id="teacher-001",
            topic="协商",
            initial_proposal={"v": 1},
        )
        result = engine.finalize_negotiation(
            session_id=session.session_id,
            decision=HumanDecision.REJECT,
        )
        assert result.phase == NegotiationPhase.ABANDONMENT

    def test_结束不存在的协商抛异常(self) -> None:
        engine = _make_engine()
        with pytest.raises(CC2Error):
            engine.finalize_negotiation(
                session_id="nonexistent",
                decision=HumanDecision.APPROVE,
            )

    def test_协商禁用时启动抛异常(self) -> None:
        engine = _make_engine(config=CollaborationConfig(enable_negotiation=False))
        with pytest.raises(CC2Error):
            engine.start_negotiation(
                agent_id="agent-001",
                human_id="teacher-001",
                topic="测试",
                initial_proposal={"v": 1},
            )

    def test_获取协商会话(self) -> None:
        engine = _make_engine()
        session = engine.start_negotiation(
            agent_id="agent-001",
            human_id="teacher-001",
            topic="协商",
            initial_proposal={"v": 1},
        )
        fetched = engine.get_negotiation(session.session_id)
        assert fetched is not None
        assert fetched.topic == "协商"


# ============================================================
# 16. 测试 CollaborationEngine_持续质量
# ============================================================


class TestCollaborationEngine_持续质量:
    """验证引擎的 evaluate_sustained_quality 持续质量评估."""

    def test_质量达标_从CONDITIONAL降级到MONITORED(self) -> None:
        engine = _make_engine(
            config=CollaborationConfig(
                sustained_quality_threshold=0.95,
                sustained_quality_window=3600,
            ),
        )
        event = engine.evaluate_sustained_quality(
            agent_id="agent-001",
            accuracy=0.98,
        )
        assert event is not None
        assert event.from_mode == CollaborationMode.CONDITIONAL
        assert event.to_mode == CollaborationMode.MONITORED
        assert event.trigger == SwitchTrigger.SUSTAINED_QUALITY

    def test_质量达标_从SUPERVISED降级到CONDITIONAL(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.SUPERVISED)
        event = engine.evaluate_sustained_quality(
            agent_id="agent-001",
            accuracy=0.97,
        )
        assert event is not None
        assert event.from_mode == CollaborationMode.SUPERVISED
        assert event.to_mode == CollaborationMode.CONDITIONAL

    def test_质量达标_从MONITORED降级到AUTONOMOUS(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.MONITORED)
        event = engine.evaluate_sustained_quality(
            agent_id="agent-001",
            accuracy=0.99,
        )
        assert event is not None
        assert event.to_mode == CollaborationMode.AUTONOMOUS

    def test_质量不达标_不触发降级(self) -> None:
        engine = _make_engine()
        event = engine.evaluate_sustained_quality(
            agent_id="agent-001",
            accuracy=0.80,
        )
        assert event is None

    def test_已是AUTONOMOUS_不再降级(self) -> None:
        engine = _make_engine()
        engine.update_profile("agent-001", mode=CollaborationMode.AUTONOMOUS)
        event = engine.evaluate_sustained_quality(
            agent_id="agent-001",
            accuracy=0.99,
        )
        assert event is None

    def test_未注册Agent返回None(self) -> None:
        engine = _make_engine(register_agent=None)
        event = engine.evaluate_sustained_quality(
            agent_id="nonexistent",
            accuracy=0.99,
        )
        assert event is None

    def test_恰好达到阈值_触发降级(self) -> None:
        engine = _make_engine()
        event = engine.evaluate_sustained_quality(
            agent_id="agent-001",
            accuracy=0.95,
        )
        assert event is not None

    def test_自定义窗口参数(self) -> None:
        engine = _make_engine()
        event = engine.evaluate_sustained_quality(
            agent_id="agent-001",
            accuracy=0.99,
            window_seconds=7200,
        )
        assert event is not None
        # 降级成功
        assert engine.get_profile("agent-001").mode == CollaborationMode.MONITORED


# ============================================================
# 17. 测试 CollaborationEngine_查询与统计
# ============================================================


class TestCollaborationEngine_查询与统计:
    """验证引擎的查询、统计和清空功能."""

    def test_查询干预_按Agent过滤(self) -> None:
        engine = _make_engine(register_agent=None)
        engine.register_profile(_make_profile("agent-a"))
        engine.register_profile(_make_profile("agent-b"))
        engine.create_intervention(agent_id="agent-a")
        engine.create_intervention(agent_id="agent-a")
        engine.create_intervention(agent_id="agent-b")
        results = engine.query_interventions(agent_id="agent-a")
        assert len(results) == 2

    def test_查询干预_按状态过滤(self) -> None:
        engine = _make_engine()
        r1 = engine.create_intervention(agent_id="agent-001")
        engine.create_intervention(agent_id="agent-001")
        engine.respond_to_intervention(
            request_id=r1.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
        )
        pending = engine.query_interventions(agent_id="agent-001", status=InterventionStatus.PENDING)
        resolved = engine.query_interventions(agent_id="agent-001", status=InterventionStatus.RESOLVED)
        assert len(pending) == 1
        assert len(resolved) == 1

    def test_查询干预_按类型过滤(self) -> None:
        engine = _make_engine()
        engine.create_intervention(agent_id="agent-001", intervention_type=InterventionType.CHECKPOINT)
        engine.create_intervention(agent_id="agent-001", intervention_type=InterventionType.ESCALATION)
        engine.create_intervention(agent_id="agent-001", intervention_type=InterventionType.REVIEW)
        results = engine.query_interventions(
            agent_id="agent-001",
            intervention_type=InterventionType.ESCALATION,
        )
        assert len(results) == 1

    def test_查询干预_限制返回数量(self) -> None:
        engine = _make_engine()
        for _ in range(5):
            engine.create_intervention(agent_id="agent-001")
        results = engine.query_interventions(agent_id="agent-001", limit=3)
        assert len(results) == 3

    def test_查询干预_无过滤返回全部(self) -> None:
        engine = _make_engine()
        engine.create_intervention(agent_id="agent-001")
        engine.create_intervention(agent_id="agent-001")
        results = engine.query_interventions()
        assert len(results) == 2

    def test_统计_初始值为零(self) -> None:
        engine = _make_engine(register_agent=None)
        stats = engine.get_stats()
        assert stats["registered_agents"] == 0
        assert stats["total_interventions"] == 0
        assert stats["resolved"] == 0
        assert stats["mode_switches"] == 0

    def test_统计_注册Agent计数(self) -> None:
        engine = _make_engine()
        engine.register_profile(_make_profile("agent-b"))
        stats = engine.get_stats()
        assert stats["registered_agents"] == 2

    def test_统计_干预计数(self) -> None:
        engine = _make_engine()
        engine.create_intervention(agent_id="agent-001")
        engine.create_intervention(agent_id="agent-001")
        r = engine.create_intervention(agent_id="agent-001")
        engine.respond_to_intervention(
            request_id=r.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
        )
        stats = engine.get_stats()
        assert stats["total_interventions"] == 3
        assert stats["resolved"] == 1
        assert stats["pending_interventions"] == 2

    def test_统计_按决策分组(self) -> None:
        engine = _make_engine()
        r1 = engine.create_intervention(agent_id="agent-001")
        r2 = engine.create_intervention(agent_id="agent-001")
        engine.respond_to_intervention(
            request_id=r1.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.APPROVE,
        )
        engine.respond_to_intervention(
            request_id=r2.request.request_id,
            human_id="teacher-001",
            decision=HumanDecision.REJECT,
        )
        stats = engine.get_stats()
        assert stats["by_decision"]["approve"] == 1
        assert stats["by_decision"]["reject"] == 1

    def test_统计_按模式分组(self) -> None:
        engine = _make_engine()
        engine.register_profile(_make_profile("agent-b", mode=CollaborationMode.AUTONOMOUS))
        stats = engine.get_stats()
        assert stats["agents_by_mode"]["conditional"] == 1
        assert stats["agents_by_mode"]["autonomous"] == 1

    def test_统计_模式切换计数(self) -> None:
        engine = _make_engine()
        engine.switch_mode("agent-001", CollaborationMode.SUPERVISED, SwitchTrigger.LOW_CONFIDENCE)
        stats = engine.get_stats()
        assert stats["mode_switches"] == 1

    def test_统计_协商计数(self) -> None:
        engine = _make_engine()
        engine.start_negotiation(
            agent_id="agent-001",
            human_id="teacher-001",
            topic="测试",
            initial_proposal={"v": 1},
        )
        stats = engine.get_stats()
        assert stats["negotiations"] == 1

    def test_清空引擎_所有计数归零(self) -> None:
        engine = _make_engine()
        engine.create_intervention(agent_id="agent-001")
        engine.switch_mode("agent-001", CollaborationMode.SUPERVISED, SwitchTrigger.LOW_CONFIDENCE)
        engine.clear()
        stats = engine.get_stats()
        assert stats["total_interventions"] == 0
        assert stats["mode_switches"] == 0
        assert stats["negotiations"] == 0
        assert stats["resolved"] == 0

    def test_清空引擎_配置保留(self) -> None:
        engine = _make_engine()
        engine.clear()
        # 清空后 Agent 配置仍在（clear 只清干预、协商、事件）
        profile = engine.get_profile("agent-001")
        assert profile.agent_id == "agent-001"

    def test_获取切换事件_按Agent过滤(self) -> None:
        engine = _make_engine()
        engine.register_profile(_make_profile("agent-b"))
        engine.switch_mode("agent-001", CollaborationMode.SUPERVISED, SwitchTrigger.LOW_CONFIDENCE)
        engine.switch_mode("agent-b", CollaborationMode.SUPERVISED, SwitchTrigger.LOW_CONFIDENCE)
        events_a = engine.get_switch_events(agent_id="agent-001")
        assert len(events_a) == 1

    def test_获取切换事件_限制数量(self) -> None:
        engine = _make_engine()
        for _ in range(5):
            engine.switch_mode("agent-001", CollaborationMode.SUPERVISED, SwitchTrigger.LOW_CONFIDENCE)
            engine.switch_mode("agent-001", CollaborationMode.CONDITIONAL, SwitchTrigger.SUSTAINED_QUALITY)
        events = engine.get_switch_events(agent_id="agent-001", limit=3)
        assert len(events) == 3


# ============================================================
# 18. 测试端到端集成
# ============================================================


class Test端到端集成:
    """完整教学场景流程的端到端集成测试."""

    def test_完整教学场景_从注册到自主运行(self) -> None:
        """模拟教师 Agent 的完整协作生命周期."""
        engine = CollaborationEngine()

        # 1. 注册教学 Agent，初始 CONDITIONAL 模式
        engine.register_profile(AgentCollaborationProfile(
            agent_id="tutor-agent",
            mode=CollaborationMode.CONDITIONAL,
            max_auto_steps=5,
            confidence_threshold=0.7,
        ))

        # 2. 首次评分决策创建干预请求
        record = engine.create_intervention(
            agent_id="tutor-agent",
            intervention_type=InterventionType.CHECKPOINT,
            reason="首次评分需教师确认",
            payload={"student": "张三", "answer": "...", "score": 85},
            proposed_action="提交评分 85 分",
            confidence=0.75,
        )
        assert record.status == InterventionStatus.PENDING

        # 3. 教师批准
        resolved = engine.respond_to_intervention(
            request_id=record.request.request_id,
            human_id="teacher-wang",
            decision=HumanDecision.APPROVE,
            feedback="评分合理，批准",
        )
        assert resolved.status == InterventionStatus.RESOLVED

        # 4. Agent 连续自主运行多步
        for i in range(4):
            result = engine.check_auto_step("tutor-agent", confidence=0.85)
            assert result is None, f"第 {i+1} 步不应触发干预"

        # 5. 达到自主步数上限，触发检查点干预
        checkpoint = engine.check_auto_step("tutor-agent", confidence=0.9)
        assert checkpoint is not None
        assert checkpoint.request.intervention_type == InterventionType.CHECKPOINT

        # 6. 教师再次批准，重置步数
        engine.respond_to_intervention(
            request_id=checkpoint.request.request_id,
            human_id="teacher-wang",
            decision=HumanDecision.APPROVE,
        )
        profile = engine.get_profile("tutor-agent")
        assert profile.auto_step_count == 0

        # 7. 验证统计
        stats = engine.get_stats()
        assert stats["registered_agents"] == 1
        assert stats["total_interventions"] == 2
        assert stats["resolved"] == 2

    def test_完整场景_异常升级与模式切换(self) -> None:
        """模拟检测到异常后的升级流程."""
        engine = CollaborationEngine()
        engine.register_profile(AgentCollaborationProfile(
            agent_id="grading-agent",
            mode=CollaborationMode.MONITORED,
            escalation_targets=["senior-teacher"],
        ))

        # 1. Agent 在 MONITORED 模式运行时检测到异常
        escalation = engine.escalate_to_human(
            agent_id="grading-agent",
            reason="检测到评分异常：连续多个学生获得相同分数",
            payload={"anomaly_type": "score_clustering"},
            priority=90,
        )
        assert escalation.request.intervention_type == InterventionType.ESCALATION
        assert escalation.request.priority == 90

        # 2. 自动升级：MONITORED → SUPERVISED
        assert engine.get_profile("grading-agent").mode == CollaborationMode.SUPERVISED

        # 3. 验证切换事件
        events = engine.get_switch_events("grading-agent")
        assert len(events) == 1
        assert events[0].is_upgrade is True

        # 4. 高级教师响应
        engine.respond_to_intervention(
            request_id=escalation.request.request_id,
            human_id="senior-teacher",
            decision=HumanDecision.MODIFY,
            modified_action="调整评分算法后重新评分",
            feedback="评分算法存在问题，需修改",
        )

        # 5. 验证覆盖计数
        profile = engine.get_profile("grading-agent")
        assert profile.override_count == 1

    def test_完整场景_REACT评分驱动的模式分配(self) -> None:
        """REACT 评分驱动不同任务分配不同协作模式."""
        engine = CollaborationEngine()
        engine.register_profile(AgentCollaborationProfile(agent_id="tutor"))

        # 高风险任务 → SUPERVISED
        high_score = REACTScore(
            risk=5, explainability=4, accuracy=5, consequence=5, time_sensitivity=4,
        )
        assert high_score.to_mode() == CollaborationMode.SUPERVISED

        # 低风险任务 → AUTONOMOUS
        low_score = REACTScore(
            risk=0, explainability=0, accuracy=1, consequence=0, time_sensitivity=0,
        )
        assert low_score.to_mode() == CollaborationMode.AUTONOMOUS

        # 中等风险 → CONDITIONAL
        mid_score = REACTScore(
            risk=3, explainability=3, accuracy=3, consequence=3, time_sensitivity=3,
        )
        assert mid_score.to_mode() == CollaborationMode.CONDITIONAL

    def test_完整场景_协商达成一致(self) -> None:
        """Agent 与教师通过协商达成评分共识."""
        engine = CollaborationEngine()
        engine.register_profile(AgentCollaborationProfile(agent_id="tutor"))

        # 1. Agent 发起协商
        session = engine.start_negotiation(
            agent_id="tutor",
            human_id="teacher-li",
            topic="期末考试评分标准",
            initial_proposal={
                "criteria": "综合平时成绩与期末考试",
                "weight": {"homework": 0.3, "midterm": 0.3, "final": 0.4},
            },
            initial_confidence=0.75,
        )

        # 2. 教师提出反提案
        engine.add_negotiation_round(
            session_id=session.session_id,
            proposer="human",
            proposal={
                "criteria": "综合平时成绩与期末考试",
                "weight": {"homework": 0.2, "midterm": 0.3, "final": 0.5},
            },
            confidence=0.85,
            reasoning="期末考试权重应更高",
        )

        # 3. Agent 接受调整
        engine.add_negotiation_round(
            session_id=session.session_id,
            proposer="agent",
            proposal={
                "criteria": "综合平时成绩与期末考试",
                "weight": {"homework": 0.25, "midterm": 0.3, "final": 0.45},
            },
            confidence=0.9,
            reasoning="折中方案",
        )

        # 4. 教师批准
        result = engine.finalize_negotiation(
            session_id=session.session_id,
            decision=HumanDecision.APPROVE,
        )
        assert result.final_decision == HumanDecision.APPROVE
        assert result.phase == NegotiationPhase.EXECUTION
        assert len(result.rounds) == 3

    def test_完整场景_持续质量评估自动降级(self) -> None:
        """Agent 持续高质量输出后自动降低人类控制级别."""
        engine = CollaborationEngine(
            config=CollaborationConfig(
                sustained_quality_threshold=0.95,
            ),
        )
        engine.register_profile(AgentCollaborationProfile(
            agent_id="auto-tutor",
            mode=CollaborationMode.SUPERVISED,
        ))

        # 1. 持续高质量评估
        event = engine.evaluate_sustained_quality(
            agent_id="auto-tutor",
            accuracy=0.98,
        )
        assert event is not None
        assert event.from_mode == CollaborationMode.SUPERVISED
        assert event.to_mode == CollaborationMode.CONDITIONAL

        # 2. 继续高质量，继续降级
        event2 = engine.evaluate_sustained_quality(
            agent_id="auto-tutor",
            accuracy=0.97,
        )
        assert event2 is not None
        assert event2.to_mode == CollaborationMode.MONITORED

        # 3. 最终降到 AUTONOMOUS
        event3 = engine.evaluate_sustained_quality(
            agent_id="auto-tutor",
            accuracy=0.99,
        )
        assert event3 is not None
        assert event3.to_mode == CollaborationMode.AUTONOMOUS

        # 4. 已是 AUTONOMOUS，不再降级
        event4 = engine.evaluate_sustained_quality(
            agent_id="auto-tutor",
            accuracy=0.99,
        )
        assert event4 is None

    def test_完整场景_干预超时与取消(self) -> None:
        """测试干预的超时过期和主动取消."""
        engine = CollaborationEngine()
        engine.register_profile(AgentCollaborationProfile(agent_id="tutor"))

        # 1. 创建多个干预
        r1 = engine.create_intervention(agent_id="tutor", reason="干预1")
        r2 = engine.create_intervention(agent_id="tutor", reason="干预2")
        r3 = engine.create_intervention(agent_id="tutor", reason="干预3")

        # 2. 过期一个
        expired = engine.expire_intervention(r1.request.request_id)
        assert expired is not None
        assert expired.status == InterventionStatus.EXPIRED

        # 3. 取消一个
        cancelled = engine.cancel_intervention(r2.request.request_id)
        assert cancelled is not None
        assert cancelled.status == InterventionStatus.CANCELLED

        # 4. 响应剩余的一个
        resolved = engine.respond_to_intervention(
            request_id=r3.request.request_id,
            human_id="teacher",
            decision=HumanDecision.APPROVE,
        )
        assert resolved.status == InterventionStatus.RESOLVED

        # 5. 验证统计
        stats = engine.get_stats()
        assert stats["total_interventions"] == 3
        assert stats["resolved"] == 1
        assert stats["expired"] == 1

    def test_完整场景_多种人类决策处理(self) -> None:
        """验证所有人类决策类型的处理."""
        engine = CollaborationEngine()
        engine.register_profile(AgentCollaborationProfile(agent_id="tutor"))

        decision_results = []

        for decision in HumanDecision:
            record = engine.create_intervention(
                agent_id="tutor",
                reason=f"测试 {decision.value}",
            )
            resolved = engine.respond_to_intervention(
                request_id=record.request.request_id,
                human_id="teacher",
                decision=decision,
                feedback=f"{decision.value} 反馈",
            )
            decision_results.append(resolved)

        # 所有决策都应成功处理
        for result in decision_results:
            assert result.status == InterventionStatus.RESOLVED

        # APPROVE 不增加覆盖计数
        # REJECT, MODIFY 增加覆盖计数
        profile = engine.get_profile("tutor")
        expected_overrides = sum(
            1 for d in HumanDecision if d in (HumanDecision.REJECT, HumanDecision.MODIFY)
        )
        assert profile.override_count == expected_overrides

    def test_完整场景_引擎清空后重新使用(self) -> None:
        """引擎清空后可以重新注册和操作."""
        engine = CollaborationEngine()

        # 第一轮使用
        engine.register_profile(_make_profile("agent-001"))
        engine.create_intervention(agent_id="agent-001")
        assert engine.get_stats()["total_interventions"] == 1

        # 清空（清空干预/协商/事件，但保留已注册的 Agent 配置）
        engine.clear()
        assert engine.get_stats()["total_interventions"] == 0

        # 注册新 Agent 并重新使用
        engine.register_profile(_make_profile("agent-002"))
        engine.create_intervention(agent_id="agent-002")
        stats = engine.get_stats()
        assert stats["total_interventions"] == 1
        assert stats["registered_agents"] == 2
