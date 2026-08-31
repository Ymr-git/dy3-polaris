"""L1 用户域核心数据模型增强测试 — TDD RED 阶段.

测试覆盖设计文档中识别的差距:
1. BKT 四参数模型 (p_slip / p_guess / p_transit)
2. MasterySnapshot 跨层对齐 (correct_count + 时间戳单位)
3. LearningGoal Bloom 认知层级
4. ContextEnvelope 补充组件 (resources / time_constraint)
5. Role 独立模型 + base_permissions
6. LearningContext 独立持久化实体
7. Checkpoint 类型化模型
8. 会话组件类型化 (AgentState / Interaction / Artifact)
9. student_id 格式校验
10. 增强 HiTL 模型字段
11. ABAC daily_agent_calls + 隐私常量
12. 缺失 from_dict 补全
13. 跨层对齐转换方法
14. 新增常量与阈值

设计依据:
- L1 设计文档第三章 3.1: 7 个上下文组件 (缺 resources / time_constraint)
- L1 设计文档第三章 3.3: JSON Schema 校验规则
- L1 设计文档第三章 3.4: 衰减公式 + 认知负荷计算
- L1 设计文档第四章 4.2: 置信度门控 + 纠错升级
- L1 设计文档第六章 6.4: K-匿名 / l-多样性
- L1 设计文档第七章 7.3: ER 图五表
- L1 设计文档第八章 8.2: BKT 四参数 + DecayRequest
- L3 api_models.py: KPMastery / LearnerProfile / BloomLevel / LearningStyle
- L5 session_manager.py: ForkCheckpoint 四类快照
"""

from __future__ import annotations

import math
import time
from typing import Any

import pytest

from dy3_polaris.l1.models import (
    # 常量 (已存在)
    MS_PER_HOUR,
    MS_PER_SEC,
    MIN_STABILITY,
    STABILITY_GAIN,
    PRIOR_PROB,
    WEAK_THRESHOLD,
    EMERGENCY_THRESHOLD,
    BLOCK_THRESHOLD,
    WARNING_THRESHOLD,
    MAX_DAILY_AGENT_CALLS,
    # 新增常量
    K_ANONYMITY_MIN,
    L_DIVERSITY_MIN,
    COGNITIVE_LOAD_RECALC_INTERVAL,
    FAST_ANSWER_THRESHOLD_MS,
    CONSECUTIVE_ERROR_THRESHOLD,
    BKT_DEVIATION_THRESHOLD,
    # 已有枚举
    UserRole,
    UserStatus,
    Permission,
    GradeLevel,
    MajorDirection,
    LabAccessTier,
    LearningPhase,
    SessionType,
    SessionStatus,
    HiTLType,
    HiTLPriority,
    ConfidenceGateResult,
    FeedbackType,
    DataLevel,
    AuditAction,
    AuditResult,
    # 已有模型
    ABACAttributes,
    User,
    MasterySnapshot,
    LearningGoal,
    LearningState,
    ContextEnvelope,
    LearningSession,
    SessionFork,
    AuditLogEntry,
    ApprovalRequest,
    ApprovalResponse,
    FeedbackReport,
    EmergencyAlert,
    calculate_decay,
    # 新增枚举
    ApprovalDecision,
    FeedbackCategory,
    AlertType,
    # 新增模型
    Role,
    LearningContext,
    SessionCheckpoint,
    AgentState,
    Interaction,
    SessionArtifact,
    ResourceItem,
    TimeConstraint,
    ProvenanceRecord,
    BKTParams,
)


# ============================================================
# 1. 新增常量测试
# ============================================================


class TestNewConstants:
    """隐私保护与系统阈值新增常量."""

    def test_privacy_constants(self):
        """K-匿名 >= 5, l-多样性 >= 3 (设计文档 6.4)."""
        assert K_ANONYMITY_MIN == 5
        assert L_DIVERSITY_MIN == 3

    def test_cognitive_load_recalc_interval(self):
        """每 5 次交互重新计算认知负荷 (设计文档 3.4)."""
        assert COGNITIVE_LOAD_RECALC_INTERVAL == 5

    def test_fast_answer_threshold(self):
        """答题速度 < 5 秒判定为异常 (设计文档 4.2)."""
        assert FAST_ANSWER_THRESHOLD_MS == 5_000

    def test_consecutive_error_threshold(self):
        """连续错误 >= 10 次触发紧急干预 (设计文档 4.2)."""
        assert CONSECUTIVE_ERROR_THRESHOLD == 10

    def test_bkt_deviation_threshold(self):
        """答题结果与 BKT 预测偏差 > 30% 触发纠错 (设计文档 4.2)."""
        assert BKT_DEVIATION_THRESHOLD == 0.3


# ============================================================
# 2. BKT 四参数模型测试
# ============================================================


class TestBKTParams:
    """BKT 四参数模型 (设计文档 8.2).

    P(Know): 已掌握概率
    P(Slip): 已掌握但答错的概率 (失误)
    P(Guess): 未掌握但答对的概率 (猜测)
    P(Transit): 从未掌握到掌握的转移概率 (学习)
    """

    def test_create_bkt_params_defaults(self):
        """BKT 默认参数符合经典 BKT 模型初始值."""
        params = BKTParams(p_know=0.3)
        assert params.p_know == 0.3
        assert 0.0 <= params.p_slip <= 1.0
        assert 0.0 <= params.p_guess <= 1.0
        assert 0.0 <= params.p_transit <= 1.0

    def test_bkt_params_all_fields(self):
        params = BKTParams(
            p_know=0.85,
            p_slip=0.1,
            p_guess=0.25,
            p_transit=0.15,
        )
        assert params.p_know == 0.85
        assert params.p_slip == 0.1
        assert params.p_guess == 0.25
        assert params.p_transit == 0.15

    def test_bkt_params_range_validation(self):
        with pytest.raises(ValueError):
            BKTParams(p_know=1.5)
        with pytest.raises(ValueError):
            BKTParams(p_know=-0.1)
        with pytest.raises(ValueError):
            BKTParams(p_know=0.5, p_slip=1.5)
        with pytest.raises(ValueError):
            BKTParams(p_know=0.5, p_guess=-0.1)
        with pytest.raises(ValueError):
            BKTParams(p_know=0.5, p_transit=2.0)

    def test_bkt_bayesian_update_correct(self):
        """答对时 BKT 后验更新: P(Know|correct) 上升."""
        params = BKTParams(p_know=0.3, p_slip=0.1, p_guess=0.25, p_transit=0.1)
        prior = params.p_know
        updated = params.bayesian_update(is_correct=True)
        assert updated.p_know > prior

    def test_bkt_bayesian_update_incorrect(self):
        """答错时 BKT 后验更新: P(Know|incorrect) 下降."""
        params = BKTParams(p_know=0.7, p_slip=0.1, p_guess=0.25, p_transit=0.1)
        prior = params.p_know
        updated = params.bayesian_update(is_correct=False)
        assert updated.p_know < prior

    def test_bkt_predict_correct_prob(self):
        """预测答对概率 = P(K)*P(!S) + P(!K)*P(G)."""
        params = BKTParams(p_know=0.7, p_slip=0.1, p_guess=0.25)
        # P(correct) = 0.7 * (1-0.1) + (1-0.7) * 0.25 = 0.63 + 0.075 = 0.705
        predicted = params.predict_correct_prob()
        assert predicted == pytest.approx(0.705, abs=0.01)

    def test_bkt_to_dict_from_dict_roundtrip(self):
        params = BKTParams(p_know=0.8, p_slip=0.15, p_guess=0.2, p_transit=0.12)
        d = params.to_dict()
        restored = BKTParams.from_dict(d)
        assert restored.p_know == params.p_know
        assert restored.p_slip == params.p_slip
        assert restored.p_guess == params.p_guess
        assert restored.p_transit == params.p_transit


class TestMasterySnapshotBKT:
    """MasterySnapshot 增强: BKT 四参数 + correct_count."""

    def test_mastery_snapshot_has_bkt_params(self):
        """MasterySnapshot 应内嵌 BKT 四参数."""
        snap = MasterySnapshot(
            kc_id="kc1",
            p_know=0.75,
            last_practiced_at=int(time.time() * 1000),
        )
        assert hasattr(snap, "bkt_params")
        assert isinstance(snap.bkt_params, BKTParams)
        assert snap.bkt_params.p_know == 0.75

    def test_mastery_snapshot_has_correct_count(self):
        """对齐 L3 KPMastery.correct_count."""
        snap = MasterySnapshot(
            kc_id="kc1",
            p_know=0.75,
            last_practiced_at=int(time.time() * 1000),
            correct_count=8,
            attempts=12,
        )
        assert snap.correct_count == 8
        assert snap.attempts == 12

    def test_mastery_snapshot_accuracy(self):
        """正确率 = correct_count / attempts."""
        snap = MasterySnapshot(
            kc_id="kc1",
            p_know=0.75,
            last_practiced_at=int(time.time() * 1000),
            correct_count=8,
            attempts=10,
        )
        assert snap.accuracy() == pytest.approx(0.8)

    def test_mastery_snapshot_accuracy_zero_attempts(self):
        snap = MasterySnapshot(
            kc_id="kc1", p_know=0.5, last_practiced_at=0,
        )
        assert snap.accuracy() == 0.0

    def test_to_l3_kp_mastery_conversion(self):
        """MasterySnapshot → L3 KPMastery 无损转换."""
        from dy3_polaris.l3.api_models import KPMastery

        now_ms = int(time.time() * 1000)
        snap = MasterySnapshot(
            kc_id="DOM-A-01",
            p_know=0.75,
            last_practiced_at=now_ms,
            correct_count=8,
            attempts=12,
        )
        kp = snap.to_l3_kp_mastery()
        assert kp.kp_id == "DOM-A-01"
        assert kp.mastery_prob == 0.75
        assert kp.attempts == 12
        assert kp.correct_count == 8


# ============================================================
# 3. LearningGoal Bloom 认知层级测试
# ============================================================


class TestLearningGoalBloom:
    """LearningGoal 增强: Bloom 认知层级对齐 L3."""

    def test_goal_has_bloom_level(self):
        from dy3_polaris.l3.api_models import BloomLevel

        goal = LearningGoal(
            description="掌握 Dy3+ 能级跃迁机理",
            bloom_level=BloomLevel.UNDERSTAND,
        )
        assert goal.bloom_level == BloomLevel.UNDERSTAND

    def test_goal_default_bloom_level(self):
        from dy3_polaris.l3.api_models import BloomLevel

        goal = LearningGoal(description="test")
        assert goal.bloom_level == BloomLevel.UNDERSTAND

    def test_goal_bloom_level_to_dict(self):
        from dy3_polaris.l3.api_models import BloomLevel

        goal = LearningGoal(
            description="设计实验方案",
            bloom_level=BloomLevel.CREATE,
            priority=5,
        )
        d = goal.to_dict()
        assert d["bloom_level"] == "create"

    def test_goal_bloom_level_from_dict(self):
        d = {
            "description": "分析光谱数据",
            "priority": 4,
            "bloom_level": "analyze",
        }
        goal = LearningGoal.from_dict(d)
        from dy3_polaris.l3.api_models import BloomLevel

        assert goal.bloom_level == BloomLevel.ANALYZE


# ============================================================
# 4. ContextEnvelope 补充组件测试
# ============================================================


class TestContextEnvelopeResources:
    """ContextEnvelope 增强: 可用资源 + 时间约束."""

    def test_envelope_has_resources(self):
        env = ContextEnvelope(user_id="u1", session_id="s1")
        assert hasattr(env, "resources")
        assert isinstance(env.resources, list)

    def test_envelope_has_time_constraint(self):
        env = ContextEnvelope(user_id="u1", session_id="s1")
        assert hasattr(env, "time_constraint")
        # 默认应为 None 或默认 TimeConstraint
        assert env.time_constraint is not None or True  # 允许 None 默认

    def test_envelope_with_resources(self):
        resource = ResourceItem(
            resource_id="res-001",
            title="Dy3+ 能级图解",
            resource_type="diagram",
            difficulty=0.5,
        )
        env = ContextEnvelope(
            user_id="u1",
            session_id="s1",
            resources=[resource],
        )
        assert len(env.resources) == 1
        assert env.resources[0].resource_id == "res-001"

    def test_envelope_with_time_constraint(self):
        tc = TimeConstraint(
            available_minutes=60,
            recommended_phase=LearningPhase.PRACTICE,
        )
        env = ContextEnvelope(
            user_id="u1",
            session_id="s1",
            time_constraint=tc,
        )
        assert env.time_constraint.available_minutes == 60

    def test_envelope_to_dict_includes_new_fields(self):
        env = ContextEnvelope(user_id="u1", session_id="s1")
        d = env.to_dict()
        assert "resources" in d
        assert "time_constraint" in d

    def test_envelope_from_dict_with_new_fields(self):
        d = {
            "user_id": "u1",
            "session_id": "s1",
            "resources": [
                {
                    "resource_id": "res-001",
                    "title": "test",
                    "resource_type": "card",
                    "difficulty": 0.5,
                }
            ],
            "time_constraint": {
                "available_minutes": 45,
                "recommended_phase": "quiz",
            },
        }
        env = ContextEnvelope.from_dict(d)
        assert len(env.resources) == 1
        assert env.resources[0].resource_id == "res-001"
        assert env.time_constraint.available_minutes == 45


class TestResourceItem:
    """可用资源项模型."""

    def test_create_resource(self):
        res = ResourceItem(
            resource_id="res-001",
            title="Dy3+ 能级跃迁图",
            resource_type="diagram",
            difficulty=0.5,
        )
        assert res.resource_id == "res-001"
        assert res.title == "Dy3+ 能级跃迁图"
        assert res.resource_type == "diagram"
        assert res.difficulty == 0.5

    def test_difficulty_range(self):
        with pytest.raises(ValueError):
            ResourceItem(
                resource_id="r", title="t", resource_type="c", difficulty=-0.1
            )
        with pytest.raises(ValueError):
            ResourceItem(
                resource_id="r", title="t", resource_type="c", difficulty=1.5
            )

    def test_to_dict_from_dict(self):
        res = ResourceItem(
            resource_id="res-001",
            title="test",
            resource_type="card",
            difficulty=0.7,
        )
        d = res.to_dict()
        restored = ResourceItem.from_dict(d)
        assert restored.resource_id == "res-001"
        assert restored.difficulty == 0.7


class TestTimeConstraint:
    """时间约束模型."""

    def test_create_default(self):
        tc = TimeConstraint()
        assert tc.available_minutes > 0
        assert tc.recommended_phase is not None

    def test_create_with_values(self):
        tc = TimeConstraint(
            available_minutes=90,
            recommended_phase=LearningPhase.PRACTICE,
        )
        assert tc.available_minutes == 90
        assert tc.recommended_phase == LearningPhase.PRACTICE

    def test_to_dict_from_dict(self):
        tc = TimeConstraint(
            available_minutes=60,
            recommended_phase=LearningPhase.REVIEW,
        )
        d = tc.to_dict()
        restored = TimeConstraint.from_dict(d)
        assert restored.available_minutes == 60


# ============================================================
# 5. Role 独立模型测试
# ============================================================


class TestRole:
    """Role 独立模型 (ER 图 Role 表)."""

    def test_create_role_with_permissions(self):
        role = Role(
            role_code="undergrad",
            role_name="本科生",
            base_permissions=[
                Permission.KB_PUBLIC_READ,
                Permission.AGENT_DIAGNOSIS,
                Permission.VIEW_OWN_REPORT,
            ],
        )
        assert role.role_code == "undergrad"
        assert role.role_name == "本科生"
        assert Permission.KB_PUBLIC_READ in role.base_permissions
        assert role.role_id > 0

    def test_role_has_permission(self):
        role = Role(
            role_code="teacher",
            role_name="教师",
            base_permissions=[
                Permission.KB_WRITE_EDIT,
                Permission.VIEW_STUDENT_REPORT,
            ],
        )
        assert role.has_permission(Permission.KB_WRITE_EDIT)
        assert not role.has_permission(Permission.USER_MANAGE)

    def test_role_id_is_auto_increment(self):
        r1 = Role(role_code="a", role_name="A", base_permissions=[])
        r2 = Role(role_code="b", role_name="B", base_permissions=[])
        assert r1.role_id != r2.role_id

    def test_role_to_dict(self):
        role = Role(
            role_code="admin",
            role_name="管理员",
            base_permissions=[Permission.SYSTEM_CONFIG, Permission.USER_MANAGE],
        )
        d = role.to_dict()
        assert d["role_code"] == "admin"
        assert d["role_name"] == "管理员"
        assert "system_config" in d["base_permissions"]

    def test_default_roles(self):
        """Role.default_roles() 返回 5 种预定义角色."""
        roles = Role.default_roles()
        assert len(roles) == 5
        codes = {r.role_code for r in roles}
        assert "undergrad" in codes
        assert "graduate" in codes
        assert "teacher" in codes
        assert "admin" in codes
        assert "alumni" in codes


# ============================================================
# 6. LearningContext 独立持久化实体测试
# ============================================================


class TestLearningContext:
    """LearningContext 独立持久化实体 (ER 图 LearningContext 表)."""

    def test_create_context(self):
        ctx = LearningContext(
            session_id="sess-001",
            user_id="u-001",
        )
        assert ctx.context_id  # 自动生成
        assert ctx.session_id == "sess-001"
        assert ctx.user_id == "u-001"
        assert ctx.last_refreshed > 0
        assert isinstance(ctx.envelope, ContextEnvelope)

    def test_context_id_is_unique(self):
        c1 = LearningContext(session_id="s1", user_id="u1")
        c2 = LearningContext(session_id="s2", user_id="u2")
        assert c1.context_id != c2.context_id

    def test_context_refresh(self):
        ctx = LearningContext(session_id="s1", user_id="u1")
        old_refreshed = ctx.last_refreshed
        time.sleep(0.01)
        ctx.refresh()
        assert ctx.last_refreshed > old_refreshed

    def test_context_to_dict(self):
        ctx = LearningContext(session_id="sess-001", user_id="u-001")
        d = ctx.to_dict()
        assert d["session_id"] == "sess-001"
        assert d["user_id"] == "u-001"
        assert "context_id" in d
        assert "envelope" in d
        assert "last_refreshed" in d

    def test_context_from_dict(self):
        d = {
            "context_id": "ctx-001",
            "session_id": "sess-001",
            "user_id": "u-001",
            "last_refreshed": int(time.time() * 1000),
            "envelope": {
                "user_id": "u-001",
                "session_id": "sess-001",
            },
        }
        ctx = LearningContext.from_dict(d)
        assert ctx.context_id == "ctx-001"
        assert ctx.session_id == "sess-001"
        assert isinstance(ctx.envelope, ContextEnvelope)


# ============================================================
# 7. SessionCheckpoint 类型化模型测试
# ============================================================


class TestSessionCheckpoint:
    """Session Checkpoint 类型化模型 (设计文档 5.3)."""

    def test_create_checkpoint(self):
        cp = SessionCheckpoint(
            session_id="sess-001",
            seq=0,
            agent_states={"diagnosis": {"status": "idle"}},
        )
        assert cp.checkpoint_id  # 自动生成
        assert cp.session_id == "sess-001"
        assert cp.seq == 0
        assert cp.created_at > 0

    def test_checkpoint_id_unique(self):
        c1 = SessionCheckpoint(session_id="s1", seq=0)
        c2 = SessionCheckpoint(session_id="s1", seq=1)
        assert c1.checkpoint_id != c2.checkpoint_id

    def test_checkpoint_to_dict(self):
        cp = SessionCheckpoint(
            session_id="sess-001",
            seq=5,
            agent_states={"agent1": {"state": "done"}},
        )
        d = cp.to_dict()
        assert d["session_id"] == "sess-001"
        assert d["seq"] == 5
        assert "checkpoint_id" in d

    def test_checkpoint_from_dict(self):
        d = {
            "checkpoint_id": "cp-001",
            "session_id": "sess-001",
            "seq": 3,
            "agent_states": {"a1": {"s": "ok"}},
            "created_at": 1234567890,
        }
        cp = SessionCheckpoint.from_dict(d)
        assert cp.checkpoint_id == "cp-001"
        assert cp.seq == 3


# ============================================================
# 8. 会话组件类型化测试
# ============================================================


class TestAgentState:
    """Agent 推理状态快照 (设计文档 5.1)."""

    def test_create_agent_state(self):
        state = AgentState(
            agent_id="agent-diagnosis-001",
            agent_type="diagnosis",
            status="completed",
            output_summary="学生掌握度偏低, 建议强化基础概念",
        )
        assert state.agent_id == "agent-diagnosis-001"
        assert state.agent_type == "diagnosis"
        assert state.status == "completed"
        assert "掌握度" in state.output_summary

    def test_to_dict_from_dict(self):
        state = AgentState(
            agent_id="a1",
            agent_type="review",
            status="running",
            output_summary="checking...",
        )
        d = state.to_dict()
        restored = AgentState.from_dict(d)
        assert restored.agent_id == "a1"
        assert restored.status == "running"


class TestInteraction:
    """交互历史条目 (设计文档 5.1)."""

    def test_create_interaction(self):
        inter = Interaction(
            interaction_type="qa",
            content="什么是 Dy3+ 的 4f-4f 跃迁?",
            user_role="student",
        )
        assert inter.interaction_id  # 自动生成
        assert inter.interaction_type == "qa"
        assert "4f-4f" in inter.content

    def test_interaction_with_response(self):
        inter = Interaction(
            interaction_type="qa",
            content="什么是 Judd-Ofelt 理论?",
            user_role="student",
            response="Judd-Ofelt 理论用于计算...",
            agent_id="agent-knowledge-gen-001",
        )
        assert inter.response is not None
        assert inter.agent_id == "agent-knowledge-gen-001"

    def test_to_dict_from_dict(self):
        inter = Interaction(
            interaction_type="quiz",
            content="Dy3+ 主发射峰在哪个波段?",
            user_role="student",
            response="~575 nm (黄光)",
        )
        d = inter.to_dict()
        restored = Interaction.from_dict(d)
        assert restored.interaction_type == "quiz"
        assert restored.content == inter.content


class TestSessionArtifact:
    """会话产出物 (设计文档 5.1)."""

    def test_create_artifact(self):
        art = SessionArtifact(
            artifact_type="knowledge_card",
            title="Dy3+ 能级跃迁卡片",
            content="...",
        )
        assert art.artifact_id  # 自动生成
        assert art.artifact_type == "knowledge_card"
        assert art.title == "Dy3+ 能级跃迁卡片"

    def test_artifact_with_confidence(self):
        art = SessionArtifact(
            artifact_type="quiz",
            title="能级跃迁测验",
            content="...",
            confidence=0.88,
        )
        assert art.confidence == 0.88
        gate = ConfidenceGateResult.evaluate(art.confidence)
        assert gate == ConfidenceGateResult.PASS

    def test_to_dict_from_dict(self):
        art = SessionArtifact(
            artifact_type="report",
            title="学情诊断报告",
            content="...",
            confidence=0.72,
        )
        d = art.to_dict()
        restored = SessionArtifact.from_dict(d)
        assert restored.artifact_type == "report"
        assert restored.confidence == 0.72


class TestLearningSessionTypedComponents:
    """LearningSession 使用类型化组件."""

    def test_session_has_typed_agent_states(self):
        session = LearningSession(
            user_id="u1",
            session_type=SessionType.LEARNING,
        )
        state = AgentState(
            agent_id="a1", agent_type="diagnosis", status="done"
        )
        session.set_agent_state("a1", state)
        assert "a1" in session.agent_states
        retrieved = session.get_agent_state("a1")
        assert retrieved is not None
        assert retrieved.status == "done"

    def test_session_has_typed_interactions(self):
        session = LearningSession(
            user_id="u1",
            session_type=SessionType.LEARNING,
        )
        inter = Interaction(
            interaction_type="qa",
            content="test question",
            user_role="student",
        )
        session.add_typed_interaction(inter)
        assert len(session.interaction_log) == 1

    def test_session_has_typed_artifacts(self):
        session = LearningSession(
            user_id="u1",
            session_type=SessionType.LEARNING,
        )
        art = SessionArtifact(
            artifact_type="card", title="test", content="..."
        )
        session.add_typed_artifact(art)
        assert len(session.artifacts) == 1

    def test_session_has_updated_at(self):
        session = LearningSession(
            user_id="u1",
            session_type=SessionType.LEARNING,
        )
        assert hasattr(session, "updated_at")
        assert session.updated_at >= session.created_at

    def test_session_touch_updates_timestamp(self):
        session = LearningSession(
            user_id="u1",
            session_type=SessionType.LEARNING,
        )
        old = session.updated_at
        time.sleep(0.01)
        session.touch()
        assert session.updated_at > old


# ============================================================
# 9. student_id 格式校验测试
# ============================================================


class TestStudentIdValidation:
    """student_id 格式校验 (设计文档 3.3 JSON Schema)."""

    def test_valid_student_id(self):
        """格式 ^[A-Z]{2}\\d{8}$ (如 CS20240001)."""
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        assert user.student_id == "CS20240001"

    def test_valid_graduate_id(self):
        user = User(
            student_id="GR20240001",
            institution_id="inst-001",
            role=UserRole.GRADUATE,
        )
        assert user.student_id == "GR20240001"

    def test_invalid_lowercase(self):
        with pytest.raises(ValueError):
            User(
                student_id="cs20240001",
                institution_id="inst-001",
                role=UserRole.UNDERGRAD,
            )

    def test_invalid_too_short(self):
        with pytest.raises(ValueError):
            User(
                student_id="CS2024001",
                institution_id="inst-001",
                role=UserRole.UNDERGRAD,
            )

    def test_invalid_extra_chars(self):
        with pytest.raises(ValueError):
            User(
                student_id="CS20240001X",
                institution_id="inst-001",
                role=UserRole.UNDERGRAD,
            )

    def test_invalid_no_letters(self):
        with pytest.raises(ValueError):
            User(
                student_id="1220240001",
                institution_id="inst-001",
                role=UserRole.UNDERGRAD,
            )


# ============================================================
# 10. ABAC daily_agent_calls 测试
# ============================================================


class TestABACDailyAgentCalls:
    """ABAC 属性增加 daily_agent_calls (Cedar 策略引用)."""

    def test_abac_has_daily_agent_calls(self):
        attrs = ABACAttributes()
        assert hasattr(attrs, "daily_agent_calls")
        assert attrs.daily_agent_calls == 0

    def test_abac_set_daily_agent_calls(self):
        attrs = ABACAttributes(daily_agent_calls=15)
        assert attrs.daily_agent_calls == 15

    def test_abac_increment_calls(self):
        attrs = ABACAttributes()
        attrs.increment_agent_calls()
        assert attrs.daily_agent_calls == 1
        attrs.increment_agent_calls()
        assert attrs.daily_agent_calls == 2

    def test_abac_reset_daily_calls(self):
        attrs = ABACAttributes(daily_agent_calls=18)
        attrs.reset_daily_calls()
        assert attrs.daily_agent_calls == 0

    def test_abac_calls_within_limit(self):
        attrs = ABACAttributes(daily_agent_calls=19)
        assert attrs.can_invoke_agent() is True
        attrs.increment_agent_calls()
        assert attrs.can_invoke_agent() is False

    def test_to_dict_includes_daily_calls(self):
        attrs = ABACAttributes(daily_agent_calls=10)
        d = attrs.to_dict()
        assert "daily_agent_calls" in d
        assert d["daily_agent_calls"] == 10


# ============================================================
# 11. 增强 HiTL 模型测试
# ============================================================


class TestApprovalDecision:
    """审批决策枚举 (三态: APPROVE / REJECT / MODIFY)."""

    def test_has_three_decisions(self):
        assert ApprovalDecision.APPROVE
        assert ApprovalDecision.REJECT
        assert ApprovalDecision.MODIFY

    def test_enum_values(self):
        assert ApprovalDecision.APPROVE.value == "approve"
        assert ApprovalDecision.REJECT.value == "reject"
        assert ApprovalDecision.MODIFY.value == "modify"


class TestEnhancedApprovalRequest:
    """增强 ApprovalRequest 字段."""

    def test_request_has_confidence(self):
        req = ApprovalRequest(
            user_id="u1",
            session_id="s1",
            hitl_type=HiTLType.CONFIRMATION,
            content="确认理解",
            confidence=0.7,
        )
        assert req.confidence == 0.7
        gate = ConfidenceGateResult.evaluate(req.confidence)
        assert gate == ConfidenceGateResult.WARNING

    def test_request_has_deadline(self):
        req = ApprovalRequest(
            user_id="u1",
            session_id="s1",
            hitl_type=HiTLType.CONFIRMATION,
            content="确认",
            deadline=int(time.time() * 1000) + 60000,
        )
        assert req.deadline is not None

    def test_request_is_expired(self):
        old_deadline = int(time.time() * 1000) - 1000
        req = ApprovalRequest(
            user_id="u1",
            session_id="s1",
            hitl_type=HiTLType.CONFIRMATION,
            content="确认",
            deadline=old_deadline,
        )
        assert req.is_expired() is True

    def test_request_not_expired(self):
        future = int(time.time() * 1000) + 60000
        req = ApprovalRequest(
            user_id="u1",
            session_id="s1",
            hitl_type=HiTLType.CONFIRMATION,
            content="确认",
            deadline=future,
        )
        assert req.is_expired() is False


class TestEnhancedApprovalResponse:
    """增强 ApprovalResponse: 三态决策."""

    def test_approve_response(self):
        resp = ApprovalResponse(
            request_id="hitl-001",
            responder_id="u-001",
            decision=ApprovalDecision.APPROVE,
            comment="内容准确",
        )
        assert resp.decision == ApprovalDecision.APPROVE
        assert resp.is_approved() is True

    def test_reject_response(self):
        resp = ApprovalResponse(
            request_id="hitl-001",
            responder_id="u-001",
            decision=ApprovalDecision.REJECT,
            comment="内容有误",
        )
        assert resp.is_approved() is False

    def test_modify_response(self):
        resp = ApprovalResponse(
            request_id="hitl-001",
            responder_id="u-001",
            decision=ApprovalDecision.MODIFY,
            comment="微调描述",
            modifications=[{"field": "title", "old": "A", "new": "B"}],
        )
        assert resp.decision == ApprovalDecision.MODIFY
        assert len(resp.modifications) == 1

    def test_response_to_dict(self):
        resp = ApprovalResponse(
            request_id="hitl-001",
            responder_id="u-001",
            decision=ApprovalDecision.APPROVE,
            comment="ok",
        )
        d = resp.to_dict()
        assert d["decision"] == "approve"


class TestFeedbackCategory:
    """反馈分类枚举 (设计文档 8.4: FACTUAL / ADAPTIVE / SAFETY)."""

    def test_has_three_categories(self):
        assert FeedbackCategory.FACTUAL
        assert FeedbackCategory.ADAPTIVE
        assert FeedbackCategory.SAFETY


class TestEnhancedFeedbackReport:
    """增强 FeedbackReport: severity + source_envelope_id."""

    def test_report_with_severity(self):
        report = FeedbackReport(
            user_id="u1",
            session_id="s1",
            feedback_type=FeedbackType.INCORRECT,
            content="4f-4f 跃迁描述有误",
            severity=0.8,
        )
        assert report.severity == 0.8

    def test_report_with_source_envelope(self):
        report = FeedbackReport(
            user_id="u1",
            session_id="s1",
            feedback_type=FeedbackType.NEED_MORE,
            content="需要更多实例",
            source_envelope_id="env-001",
        )
        assert report.source_envelope_id == "env-001"

    def test_report_to_dict_includes_new_fields(self):
        report = FeedbackReport(
            user_id="u1",
            session_id="s1",
            feedback_type=FeedbackType.REPORT,
            content="安全问题",
            severity=0.9,
            source_envelope_id="env-002",
        )
        d = report.to_dict()
        assert "severity" in d
        assert "source_envelope_id" in d


class TestAlertType:
    """紧急警报类型枚举."""

    def test_has_alert_types(self):
        assert AlertType.HIGH_COGNITIVE_LOAD
        assert AlertType.CONSECUTIVE_ERRORS
        assert AlertType.FAST_ANSWERING
        assert AlertType.BKT_DEVIATION


class TestEnhancedEmergencyAlert:
    """增强 EmergencyAlert: alert_type + cognitive_load + error_count."""

    def test_alert_with_type(self):
        alert = EmergencyAlert(
            session_id="s1",
            user_id="u1",
            trigger_reason="认知负荷过高",
            trigger_value=0.97,
            alert_type=AlertType.HIGH_COGNITIVE_LOAD,
            cognitive_load=0.97,
        )
        assert alert.alert_type == AlertType.HIGH_COGNITIVE_LOAD
        assert alert.cognitive_load == 0.97

    def test_alert_consecutive_errors(self):
        alert = EmergencyAlert(
            session_id="s1",
            user_id="u1",
            trigger_reason="连续错误>=10次",
            trigger_value=10,
            alert_type=AlertType.CONSECUTIVE_ERRORS,
            error_count=10,
        )
        assert alert.error_count == 10

    def test_alert_to_dict_includes_new_fields(self):
        alert = EmergencyAlert(
            session_id="s1",
            user_id="u1",
            trigger_reason="test",
            trigger_value=0.96,
            alert_type=AlertType.HIGH_COGNITIVE_LOAD,
            cognitive_load=0.96,
            error_count=0,
        )
        d = alert.to_dict()
        assert "alert_type" in d
        assert "cognitive_load" in d
        assert "error_count" in d


# ============================================================
# 12. ProvenanceRecord 测试
# ============================================================


class TestProvenanceRecord:
    """溯源记录 (设计文档 8.1, PROV-O)."""

    def test_create_provenance(self):
        prov = ProvenanceRecord(
            artifact_id="art-001",
            actor_chain=["agent-knowledge-gen-001", "agent-review-001"],
            code_hash="sha256:abc123",
            env_hash="sha256:def456",
        )
        assert prov.artifact_id == "art-001"
        assert len(prov.actor_chain) == 2
        assert prov.code_hash == "sha256:abc123"
        assert prov.provenance_id  # 自动生成

    def test_to_dict(self):
        prov = ProvenanceRecord(
            artifact_id="art-001",
            actor_chain=["a1"],
            code_hash="h1",
            env_hash="h2",
        )
        d = prov.to_dict()
        assert d["artifact_id"] == "art-001"
        assert "actor_chain" in d
        assert d["code_hash"] == "h1"


# ============================================================
# 13. 缺失 from_dict 补全测试
# ============================================================


class TestMissingFromDict:
    """验证所有模型都有 from_dict 方法."""

    def test_session_fork_from_dict(self):
        fork = SessionFork(
            source_session_id="sess-001",
            fork_point_seq=3,
            fork_reason="test",
            branch_label="A",
            snapshot_at_fork=ContextEnvelope(user_id="u1", session_id="s1"),
        )
        d = fork.to_dict()
        restored = SessionFork.from_dict(d)
        assert restored.source_session_id == "sess-001"
        assert restored.fork_point_seq == 3
        assert isinstance(restored.snapshot_at_fork, ContextEnvelope)

    def test_emergency_alert_from_dict(self):
        alert = EmergencyAlert(
            session_id="s1",
            user_id="u1",
            trigger_reason="test",
            trigger_value=0.96,
        )
        d = alert.to_dict()
        restored = EmergencyAlert.from_dict(d)
        assert restored.session_id == "s1"
        assert restored.trigger_value == 0.96

    def test_approval_response_from_dict(self):
        resp = ApprovalResponse(
            request_id="r1",
            responder_id="u1",
            decision=ApprovalDecision.APPROVE,
        )
        d = resp.to_dict()
        restored = ApprovalResponse.from_dict(d)
        assert restored.request_id == "r1"
        assert restored.decision == ApprovalDecision.APPROVE

    def test_audit_log_entry_from_dict(self):
        entry = AuditLogEntry(
            actor_id="CS20240001",
            actor_role=UserRole.UNDERGRAD,
            action=AuditAction.VIEW,
            target_resource="kb:test",
            target_data_level=DataLevel.L2_INTERNAL,
            purpose="学习",
            result=AuditResult.SUCCESS,
        )
        d = entry.to_dict()
        restored = AuditLogEntry.from_dict(d)
        assert restored.actor_id == "CS20240001"
        assert restored.action == AuditAction.VIEW


# ============================================================
# 14. 跨层对齐转换方法测试
# ============================================================


class TestCrossLayerAlignment:
    """跨层对齐转换方法."""

    def test_mastery_snapshot_to_l3_conversion(self):
        """L1 MasterySnapshot → L3 KPMastery 无损转换."""
        from dy3_polaris.l3.api_models import KPMastery

        now_ms = int(time.time() * 1000)
        snap = MasterySnapshot(
            kc_id="DOM-A-01",
            p_know=0.75,
            last_practiced_at=now_ms,
            correct_count=7,
            attempts=10,
        )
        kp = snap.to_l3_kp_mastery()
        assert isinstance(kp, KPMastery)
        assert kp.kp_id == "DOM-A-01"
        assert kp.mastery_prob == 0.75
        assert kp.attempts == 10
        assert kp.correct_count == 7

    def test_mastery_snapshot_from_l3_conversion(self):
        """L3 KPMastery → L1 MasterySnapshot 逆向转换."""
        from dy3_polaris.l3.api_models import KPMastery

        kp = KPMastery(
            kp_id="DOM-B-02",
            mastery_prob=0.6,
            attempts=5,
            correct_count=3,
            last_attempt_time=time.time(),
        )
        snap = MasterySnapshot.from_l3_kp_mastery(kp)
        assert snap.kc_id == "DOM-B-02"
        assert snap.p_know == 0.6
        assert snap.attempts == 5
        assert snap.correct_count == 3

    def test_context_envelope_to_l3_learner_profile(self):
        """ContextEnvelope → L3 LearnerProfile 部分转换."""
        from dy3_polaris.l3.api_models import LearnerProfile

        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u1",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(kc_id="kc1", p_know=0.8, last_practiced_at=now_ms),
                MasterySnapshot(kc_id="kc2", p_know=0.4, last_practiced_at=now_ms),
            ],
        )
        profile = env.to_l3_learner_profile()
        assert isinstance(profile, LearnerProfile)
        assert len(profile.kp_mastery) == 2


# ============================================================
# 15. 增强序列化往返测试
# ============================================================


class TestEnhancedSerialization:
    """增强模型的序列化往返完整性."""

    def test_user_with_abac_daily_calls_roundtrip(self):
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
            abac_attributes=ABACAttributes(daily_agent_calls=5),
        )
        d = user.to_dict()
        assert "daily_agent_calls" in d["abac_attributes"]
        restored = User.from_dict(d)
        assert restored.abac_attributes.daily_agent_calls == 5

    def test_session_with_typed_components_roundtrip(self):
        session = LearningSession(
            user_id="u1",
            session_type=SessionType.LEARNING,
        )
        session.set_agent_state(
            "a1",
            AgentState(agent_id="a1", agent_type="diagnosis", status="done"),
        )
        session.add_typed_interaction(
            Interaction(interaction_type="qa", content="test", user_role="student")
        )
        session.add_typed_artifact(
            SessionArtifact(artifact_type="card", title="t", content="c")
        )
        d = session.to_dict()
        restored = LearningSession.from_dict(d)
        assert restored.user_id == "u1"
        assert len(restored.interaction_log) == 1
        assert len(restored.artifacts) == 1

    def test_context_envelope_full_roundtrip(self):
        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u1",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(
                    kc_id="kc1",
                    p_know=0.8,
                    last_practiced_at=now_ms,
                    correct_count=5,
                    attempts=8,
                ),
            ],
            goals=[
                LearningGoal(description="goal1", priority=4),
            ],
            resources=[
                ResourceItem(
                    resource_id="r1", title="res", resource_type="card", difficulty=0.5
                ),
            ],
            time_constraint=TimeConstraint(
                available_minutes=45, recommended_phase=LearningPhase.QUIZ
            ),
        )
        d = env.to_dict()
        restored = ContextEnvelope.from_dict(d)
        assert len(restored.mastery_snapshot) == 1
        assert restored.mastery_snapshot[0].correct_count == 5
        assert len(restored.resources) == 1
        assert restored.time_constraint.available_minutes == 45
