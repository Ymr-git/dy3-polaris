"""L1 用户域核心数据模型测试 — TDD 测试用例.

测试覆盖:
1. 角色与权限枚举 (UserRole / UserStatus / Permission)
2. ABAC 属性维度 (GradeLevel / MajorDirection / LabAccessTier / ABACAttributes)
3. User 模型 (字段 + 默认值 + ID 生成)
4. 学习上下文 (LearningPhase / MasterySnapshot / LearningGoal / LearningState / ContextEnvelope)
5. 会话与 Fork (SessionType / SessionStatus / LearningSession / SessionFork)
6. 审计与脱敏 (DataLevel / AuditAction / AuditResult / AuditLogEntry)
7. HiTL 协同 (HiTLType / HiTLPriority / ConfidenceGate / FeedbackType)
8. HiTL 数据模型 (ApprovalRequest / ApprovalResponse / FeedbackReport / EmergencyAlert)
9. ContextEnvelope 业务方法 (is_expired / refresh_decay / get_weak_kcs / to_summary)
10. 衰减公式常量与计算 (calculate_decay)
11. 序列化往返 (to_dict / from_dict)
12. 跨层对齐 (MasterySnapshot ↔ L3 KPMastery)
13. 模型校验与边界条件

设计依据:
- L1 设计文档第二章: 角色分级体系 (RBAC+ABAC)
- L1 设计文档第三章: 学习上下文经纪 (Context Envelope)
- L1 设计文档第五章: 学习会话管理 (Session Fork)
- L1 设计文档第七章: ER 图与数据模型
- L3 api_models.py: KPMastery / LearnerProfile 接口对齐
"""

from __future__ import annotations

import math
import time
from typing import Any

import pytest

from dy3_polaris.l1.models import (
    # 常量
    MS_PER_HOUR,
    MS_PER_SEC,
    MIN_STABILITY,
    STABILITY_GAIN,
    PRIOR_PROB,
    MIN_DECAY,
    DEFAULT_REPS,
    PRIORITY_NORMAL,
    DEFAULT_SESSION_MS,
    DEFAULT_INTERACTIONS,
    DEFAULT_COGNITIVE_LOAD,
    DEFAULT_TTL,
    WEAK_THRESHOLD,
    EMERGENCY_THRESHOLD,
    BLOCK_THRESHOLD,
    WARNING_THRESHOLD,
    MAX_DAILY_AGENT_CALLS,
    # 衰减函数
    calculate_decay,
    # 角色与权限枚举
    UserRole,
    UserStatus,
    Permission,
    # ABAC 枚举
    GradeLevel,
    MajorDirection,
    LabAccessTier,
    # ABAC 模型
    ABACAttributes,
    User,
    # 学习上下文枚举
    LearningPhase,
    # 学习上下文模型
    MasterySnapshot,
    LearningGoal,
    LearningState,
    ContextEnvelope,
    # 会话枚举
    SessionType,
    SessionStatus,
    # 会话模型
    LearningSession,
    SessionFork,
    # 审计枚举
    DataLevel,
    AuditAction,
    AuditResult,
    # 审计模型
    AuditLogEntry,
    # HiTL 枚举
    HiTLType,
    HiTLPriority,
    ConfidenceGateResult,
    FeedbackType,
    # HiTL 模型
    ApprovalRequest,
    ApprovalResponse,
    FeedbackReport,
    EmergencyAlert,
)


# ============================================================
# 1. 常量测试
# ============================================================


class TestConstants:
    """衰减公式与系统阈值常量."""

    def test_time_conversion_constants(self):
        assert MS_PER_HOUR == 3_600_000
        assert MS_PER_SEC == 1_000

    def test_decay_formula_constants(self):
        assert MIN_STABILITY > 0
        assert STABILITY_GAIN > 0
        assert 0.0 <= PRIOR_PROB <= 1.0

    def test_default_values(self):
        assert MIN_DECAY == 1.0
        assert DEFAULT_REPS == 0
        assert PRIORITY_NORMAL == 3
        assert DEFAULT_SESSION_MS == 0
        assert DEFAULT_INTERACTIONS == 0
        assert DEFAULT_COGNITIVE_LOAD == 0.5
        assert DEFAULT_TTL == 3600

    def test_thresholds(self):
        assert WEAK_THRESHOLD == 0.5
        assert EMERGENCY_THRESHOLD == 0.95
        assert BLOCK_THRESHOLD == 0.4
        assert WARNING_THRESHOLD == 0.85
        assert MAX_DAILY_AGENT_CALLS > 0

    def test_threshold_ordering(self):
        assert BLOCK_THRESHOLD < WARNING_THRESHOLD < EMERGENCY_THRESHOLD


# ============================================================
# 2. 角色与权限枚举测试
# ============================================================


class TestUserRole:
    """用户角色枚举 (设计文档第二章 2.1)."""

    def test_has_all_five_roles(self):
        assert UserRole.UNDERGRAD
        assert UserRole.GRADUATE
        assert UserRole.TEACHER
        assert UserRole.ADMIN
        assert UserRole.ALUMNI

    def test_enum_values_are_strings(self):
        assert UserRole.UNDERGRAD.value == "undergrad"
        assert UserRole.GRADUATE.value == "graduate"
        assert UserRole.TEACHER.value == "teacher"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.ALUMNI.value == "alumni"

    def test_str_enum_behavior(self):
        assert isinstance(UserRole.UNDERGRAD, str)
        assert UserRole("undergrad") is UserRole.UNDERGRAD

    def test_alumni_is_readonly_role(self):
        """校友角色为只读, 不参与教学活动."""
        assert UserRole.ALUMNI != UserRole.UNDERGRAD


class TestUserStatus:
    """用户状态枚举 (设计文档 ER 图 User.status)."""

    def test_has_three_statuses(self):
        assert UserStatus.ACTIVE
        assert UserStatus.SUSPENDED
        assert UserStatus.ALUMNI

    def test_enum_values(self):
        assert UserStatus.ACTIVE.value == "active"
        assert UserStatus.SUSPENDED.value == "suspended"
        assert UserStatus.ALUMNI.value == "alumni"


class TestPermission:
    """13 项权限枚举 = 12 项功能权限 (设计文档第二章 2.2) + 1 项 HiTL 确认权限."""

    def test_has_thirteen_permissions(self):
        permissions = list(Permission)
        assert len(permissions) == 13

    def test_knowledge_permissions(self):
        assert Permission.KB_PUBLIC_READ
        assert Permission.KB_INTERNAL_DATA_ACCESS
        assert Permission.KB_WRITE_EDIT

    def test_agent_permissions(self):
        assert Permission.AGENT_DIAGNOSIS
        assert Permission.AGENT_KNOWLEDGE_GEN
        assert Permission.AGENT_REVIEW
        assert Permission.AGENT_GUIDE

    def test_data_permissions(self):
        assert Permission.VIEW_OWN_REPORT
        assert Permission.VIEW_STUDENT_REPORT
        assert Permission.EXPORT_REPORT

    def test_system_permissions(self):
        assert Permission.SYSTEM_CONFIG
        assert Permission.USER_MANAGE

    def test_hitl_permission(self):
        assert Permission.HITL_CONFIRM

    def test_str_enum_behavior(self):
        assert isinstance(Permission.KB_PUBLIC_READ, str)


# ============================================================
# 3. ABAC 属性测试
# ============================================================


class TestGradeLevel:
    """年级枚举 (设计文档第二章 2.3 ABAC 属性)."""

    def test_has_all_six_levels(self):
        levels = list(GradeLevel)
        assert len(levels) == 6
        assert GradeLevel.FRESHMAN
        assert GradeLevel.SOPHOMORE
        assert GradeLevel.JUNIOR
        assert GradeLevel.SENIOR
        assert GradeLevel.MASTER
        assert GradeLevel.PHD


class TestMajorDirection:
    """专业方向枚举 (设计文档第二章 2.3)."""

    def test_has_all_five_directions(self):
        directions = list(MajorDirection)
        assert len(directions) == 5
        assert MajorDirection.PHYSICS
        assert MajorDirection.CHEMISTRY
        assert MajorDirection.MATERIALS_SCI
        assert MajorDirection.OPTICS
        assert MajorDirection.ENGINEERING


class TestLabAccessTier:
    """实验权限等级枚举 (设计文档第二章 2.3)."""

    def test_has_four_tiers(self):
        tiers = list(LabAccessTier)
        assert len(tiers) == 4

    def test_tier_ordering(self):
        """TIER0 < TIER1 < TIER2 < TIER3."""
        assert LabAccessTier.TIER0.value < LabAccessTier.TIER1.value
        assert LabAccessTier.TIER1.value < LabAccessTier.TIER2.value
        assert LabAccessTier.TIER2.value < LabAccessTier.TIER3.value

    def test_tier0_is_virtual(self):
        assert LabAccessTier.TIER0.value == "tier0"


class TestABACAttributes:
    """ABAC 属性模型 (设计文档第二章 2.3, 5 个属性维度)."""

    def test_default_attributes(self):
        attrs = ABACAttributes()
        assert attrs.grade_level == GradeLevel.FRESHMAN
        assert attrs.major_direction == MajorDirection.MATERIALS_SCI
        assert attrs.course_progress == 0.0
        assert attrs.lab_access_tier == LabAccessTier.TIER0
        assert attrs.supervisor_id is None

    def test_full_attributes(self):
        attrs = ABACAttributes(
            grade_level=GradeLevel.MASTER,
            major_direction=MajorDirection.PHYSICS,
            course_progress=0.65,
            lab_access_tier=LabAccessTier.TIER2,
            supervisor_id="u-teacher-001",
        )
        assert attrs.grade_level == GradeLevel.MASTER
        assert attrs.course_progress == 0.65

    def test_course_progress_range(self):
        """课程进度范围 [0.0, 1.0]."""
        with pytest.raises(Exception):
            ABACAttributes(course_progress=-0.1)
        with pytest.raises(Exception):
            ABACAttributes(course_progress=1.5)

    def test_course_progress_boundary(self):
        attrs = ABACAttributes(course_progress=0.0)
        assert attrs.course_progress == 0.0
        attrs = ABACAttributes(course_progress=1.0)
        assert attrs.course_progress == 1.0

    def test_to_dict(self):
        attrs = ABACAttributes(
            grade_level=GradeLevel.PHD,
            major_direction=MajorDirection.CHEMISTRY,
            course_progress=0.8,
            lab_access_tier=LabAccessTier.TIER3,
            supervisor_id="u-sup-001",
        )
        d = attrs.to_dict()
        assert d["grade_level"] == "phd"
        assert d["major_direction"] == "chemistry"
        assert d["course_progress"] == 0.8
        assert d["lab_access_tier"] == "tier3"
        assert d["supervisor_id"] == "u-sup-001"


# ============================================================
# 4. User 模型测试
# ============================================================


class TestUser:
    """用户模型 (设计文档 ER 图 User 表)."""

    def test_create_user_with_defaults(self):
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        assert user.user_id.startswith("u-")
        assert user.student_id == "CS20240001"
        assert user.role == UserRole.UNDERGRAD
        assert user.status == UserStatus.ACTIVE
        assert isinstance(user.abac_attributes, ABACAttributes)
        assert user.created_at > 0
        assert user.updated_at > 0

    def test_user_id_is_unique(self):
        u1 = User(student_id="CS20240001", institution_id="i", role=UserRole.UNDERGRAD)
        u2 = User(student_id="CS20240002", institution_id="i", role=UserRole.UNDERGRAD)
        assert u1.user_id != u2.user_id

    def test_create_graduate_user(self):
        attrs = ABACAttributes(
            grade_level=GradeLevel.MASTER,
            supervisor_id="u-teacher-001",
        )
        user = User(
            student_id="GR20240001",
            institution_id="inst-001",
            role=UserRole.GRADUATE,
            abac_attributes=attrs,
        )
        assert user.role == UserRole.GRADUATE
        assert user.abac_attributes.supervisor_id == "u-teacher-001"

    def test_create_alumni_user(self):
        user = User(
            student_id="AL20200001",
            institution_id="inst-001",
            role=UserRole.ALUMNI,
            status=UserStatus.ALUMNI,
        )
        assert user.role == UserRole.ALUMNI
        assert user.status == UserStatus.ALUMNI

    def test_to_dict(self):
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.TEACHER,
        )
        d = user.to_dict()
        assert d["student_id"] == "CS20240001"
        assert d["role"] == "teacher"
        assert d["status"] == "active"
        assert "abac_attributes" in d
        assert isinstance(d["abac_attributes"], dict)

    def test_from_dict_roundtrip(self):
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        d = user.to_dict()
        restored = User.from_dict(d)
        assert restored.student_id == user.student_id
        assert restored.role == user.role
        assert restored.status == user.status

    def test_student_id_required(self):
        with pytest.raises(Exception):
            User(institution_id="i", role=UserRole.UNDERGRAD)


# ============================================================
# 5. 学习上下文测试
# ============================================================


class TestLearningPhase:
    """学习阶段枚举 (设计文档第三章 3.3)."""

    def test_has_four_phases(self):
        phases = list(LearningPhase)
        assert len(phases) == 4
        assert LearningPhase.PREVIEW
        assert LearningPhase.PRACTICE
        assert LearningPhase.QUIZ
        assert LearningPhase.REVIEW

    def test_enum_values(self):
        assert LearningPhase.PREVIEW.value == "preview"
        assert LearningPhase.PRACTICE.value == "practice"
        assert LearningPhase.QUIZ.value == "quiz"
        assert LearningPhase.REVIEW.value == "review"


class TestMasterySnapshot:
    """知识掌握快照 (设计文档第三章 3.3, 3.6)."""

    def test_create_snapshot(self):
        snap = MasterySnapshot(
            kc_id="dy3_energy_level_4f",
            p_know=0.85,
            last_practiced_at=int(time.time() * 1000),
        )
        assert snap.kc_id == "dy3_energy_level_4f"
        assert snap.p_know == 0.85
        assert snap.decay_factor == MIN_DECAY
        assert snap.repetitions == DEFAULT_REPS

    def test_p_know_range(self):
        with pytest.raises(Exception):
            MasterySnapshot(kc_id="kc1", p_know=-0.1, last_practiced_at=0)
        with pytest.raises(Exception):
            MasterySnapshot(kc_id="kc1", p_know=1.5, last_practiced_at=0)

    def test_decay_factor_range(self):
        with pytest.raises(Exception):
            MasterySnapshot(
                kc_id="kc1", p_know=0.5, last_practiced_at=0, decay_factor=-0.1
            )
        with pytest.raises(Exception):
            MasterySnapshot(
                kc_id="kc1", p_know=0.5, last_practiced_at=0, decay_factor=1.5
            )

    def test_repetitions_non_negative(self):
        with pytest.raises(Exception):
            MasterySnapshot(kc_id="kc1", p_know=0.5, last_practiced_at=0, repetitions=-1)

    def test_effective_mastery(self):
        """有效掌握度 = p_know * decay_factor."""
        snap = MasterySnapshot(
            kc_id="kc1", p_know=0.8, last_practiced_at=0, decay_factor=0.5
        )
        assert snap.effective_mastery() == pytest.approx(0.4)

    def test_is_weak(self):
        snap = MasterySnapshot(
            kc_id="kc1", p_know=0.3, last_practiced_at=0, decay_factor=1.0
        )
        assert snap.is_weak() is True

        snap2 = MasterySnapshot(
            kc_id="kc2", p_know=0.8, last_practiced_at=0, decay_factor=1.0
        )
        assert snap2.is_weak() is False

    def test_to_dict(self):
        snap = MasterySnapshot(
            kc_id="kc1", p_know=0.7, last_practiced_at=1000, decay_factor=0.9, repetitions=3
        )
        d = snap.to_dict()
        assert d["kc_id"] == "kc1"
        assert d["p_know"] == 0.7
        assert d["decay_factor"] == 0.9
        assert d["repetitions"] == 3
        assert d["last_practiced_at"] == 1000

    def test_aligns_with_l3_kp_mastery(self):
        """MasterySnapshot 字段对齐 L3 KPMastery (kc_id ↔ kp_id, p_know ↔ mastery_prob)."""
        from dy3_polaris.l3.api_models import KPMastery

        snap = MasterySnapshot(
            kc_id="DOM-A-01",
            p_know=0.75,
            last_practiced_at=int(time.time() * 1000),
        )
        # 转换为 L3 KPMastery 应无损
        kp = KPMastery(
            kp_id=snap.kc_id,
            mastery_prob=snap.p_know,
        )
        assert kp.kp_id == snap.kc_id
        assert kp.mastery_prob == snap.p_know


class TestLearningGoal:
    """学习目标 (设计文档第三章 3.3)."""

    def test_create_goal(self):
        goal = LearningGoal(description="掌握 Dy3+ 能级跃迁机理")
        assert goal.description == "掌握 Dy3+ 能级跃迁机理"
        assert goal.priority == PRIORITY_NORMAL

    def test_priority_range(self):
        with pytest.raises(Exception):
            LearningGoal(description="test", priority=0)
        with pytest.raises(Exception):
            LearningGoal(description="test", priority=6)

    def test_priority_boundary(self):
        g1 = LearningGoal(description="test", priority=1)
        g2 = LearningGoal(description="test", priority=5)
        assert g1.priority == 1
        assert g2.priority == 5

    def test_to_dict(self):
        goal = LearningGoal(description="test goal", priority=5)
        d = goal.to_dict()
        assert d["description"] == "test goal"
        assert d["priority"] == 5


class TestLearningState:
    """学习状态 (设计文档第三章 3.3)."""

    def test_defaults(self):
        state = LearningState()
        assert state.phase == LearningPhase.PREVIEW
        assert state.session_duration_ms == DEFAULT_SESSION_MS
        assert state.interaction_count == DEFAULT_INTERACTIONS
        assert state.cognitive_load == DEFAULT_COGNITIVE_LOAD

    def test_cognitive_load_range(self):
        with pytest.raises(Exception):
            LearningState(cognitive_load=-0.1)
        with pytest.raises(Exception):
            LearningState(cognitive_load=1.5)

    def test_cognitive_load_boundary(self):
        s1 = LearningState(cognitive_load=0.0)
        s2 = LearningState(cognitive_load=1.0)
        assert s1.cognitive_load == 0.0
        assert s2.cognitive_load == 1.0

    def test_is_emergency(self):
        state = LearningState(cognitive_load=0.96)
        assert state.is_emergency() is True

    def test_not_emergency(self):
        state = LearningState(cognitive_load=0.5)
        assert state.is_emergency() is False

    def test_to_dict(self):
        state = LearningState(
            phase=LearningPhase.PRACTICE,
            session_duration_ms=60000,
            interaction_count=15,
            cognitive_load=0.7,
        )
        d = state.to_dict()
        assert d["phase"] == "practice"
        assert d["session_duration_ms"] == 60000
        assert d["interaction_count"] == 15
        assert d["cognitive_load"] == 0.7


class TestContextEnvelope:
    """上下文信封 — L1 向下层传递数据的唯一载体 (设计文档第三章 3.3, 3.6)."""

    def test_create_envelope_with_defaults(self):
        env = ContextEnvelope(user_id="CS20240001", session_id="sess-001")
        assert env.envelope_id  # 自动生成
        assert env.user_id == "CS20240001"
        assert env.session_id == "sess-001"
        assert env.timestamp > 0
        assert isinstance(env.learning_state, LearningState)
        assert env.mastery_snapshot == []
        assert env.goals == []
        assert env.ttl == DEFAULT_TTL

    def test_envelope_id_is_unique(self):
        e1 = ContextEnvelope(user_id="u1", session_id="s1")
        e2 = ContextEnvelope(user_id="u2", session_id="s2")
        assert e1.envelope_id != e2.envelope_id

    def test_is_expired_false_when_fresh(self):
        env = ContextEnvelope(
            user_id="u1", session_id="s1", timestamp=int(time.time() * 1000)
        )
        assert env.is_expired() is False

    def test_is_expired_true_when_stale(self):
        old_ts = int(time.time() * 1000) - (DEFAULT_TTL + 100) * MS_PER_SEC
        env = ContextEnvelope(
            user_id="u1", session_id="s1", timestamp=old_ts, ttl=DEFAULT_TTL
        )
        assert env.is_expired() is True

    def test_is_expired_boundary(self):
        """恰好 TTL 秒时不过期."""
        ts = int(time.time() * 1000) - DEFAULT_TTL * MS_PER_SEC
        env = ContextEnvelope(
            user_id="u1", session_id="s1", timestamp=ts, ttl=DEFAULT_TTL
        )
        assert env.is_expired() is False

    def test_refresh_decay_updates_snapshots(self):
        """刷新衰减系数应更新所有 MasterySnapshot."""
        now_ms = int(time.time() * 1000)
        old_ts = now_ms - 48 * MS_PER_HOUR  # 48 小时前
        snap = MasterySnapshot(
            kc_id="kc1", p_know=0.9, last_practiced_at=old_ts, repetitions=2
        )
        env = ContextEnvelope(
            user_id="u1",
            session_id="s1",
            mastery_snapshot=[snap],
            timestamp=now_ms,
        )
        original_decay = snap.decay_factor
        env.refresh_decay(now_ms)
        assert env.mastery_snapshot[0].decay_factor < original_decay
        assert env.mastery_snapshot[0].decay_factor > 0

    def test_refresh_decay_fresh_kc_stays_high(self):
        """刚练习的 KC 衰减系数应接近 1.0."""
        now_ms = int(time.time() * 1000)
        snap = MasterySnapshot(
            kc_id="kc1", p_know=0.9, last_practiced_at=now_ms, repetitions=5
        )
        env = ContextEnvelope(
            user_id="u1", session_id="s1", mastery_snapshot=[snap], timestamp=now_ms
        )
        env.refresh_decay(now_ms)
        assert env.mastery_snapshot[0].decay_factor > 0.95

    def test_get_weak_kcs(self):
        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u1",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(kc_id="weak1", p_know=0.2, last_practiced_at=now_ms),
                MasterySnapshot(kc_id="strong1", p_know=0.9, last_practiced_at=now_ms),
                MasterySnapshot(kc_id="weak2", p_know=0.3, last_practiced_at=now_ms),
            ],
            timestamp=now_ms,
        )
        weak = env.get_weak_kcs()
        assert "weak1" in weak
        assert "weak2" in weak
        assert "strong1" not in weak
        assert len(weak) == 2

    def test_get_weak_kcs_custom_threshold(self):
        now_ms = int(time.time() * 1000)
        env = ContextEnvelope(
            user_id="u1",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(kc_id="kc1", p_know=0.6, last_practiced_at=now_ms),
            ],
            timestamp=now_ms,
        )
        assert env.get_weak_kcs(threshold=0.7) == ["kc1"]
        assert env.get_weak_kcs(threshold=0.5) == []

    def test_get_weak_kcs_empty(self):
        env = ContextEnvelope(user_id="u1", session_id="s1")
        assert env.get_weak_kcs() == []

    def test_to_summary(self):
        """to_summary 返回脱敏后的上下文摘要."""
        env = ContextEnvelope(
            user_id="CS20240001",
            session_id="sess-001",
            learning_state=LearningState(phase=LearningPhase.PRACTICE, cognitive_load=0.6),
        )
        summary = env.to_summary()
        assert isinstance(summary, dict)
        assert "phase" in summary
        assert "cognitive_load" in summary
        assert "weak_kc_count" in summary
        # 摘要不应包含原始学号
        assert "CS20240001" not in str(summary)

    def test_to_dict(self):
        env = ContextEnvelope(
            user_id="u1",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(kc_id="kc1", p_know=0.8, last_practiced_at=1000),
            ],
        )
        d = env.to_dict()
        assert d["user_id"] == "u1"
        assert d["session_id"] == "s1"
        assert isinstance(d["learning_state"], dict)
        assert isinstance(d["mastery_snapshot"], list)
        assert d["mastery_snapshot"][0]["kc_id"] == "kc1"

    def test_from_dict_roundtrip(self):
        env = ContextEnvelope(
            user_id="u1",
            session_id="s1",
            mastery_snapshot=[
                MasterySnapshot(kc_id="kc1", p_know=0.8, last_practiced_at=1000),
            ],
            goals=[LearningGoal(description="goal1", priority=4)],
        )
        d = env.to_dict()
        restored = ContextEnvelope.from_dict(d)
        assert restored.user_id == env.user_id
        assert restored.session_id == env.session_id
        assert len(restored.mastery_snapshot) == 1
        assert restored.mastery_snapshot[0].kc_id == "kc1"
        assert len(restored.goals) == 1
        assert restored.goals[0].description == "goal1"


# ============================================================
# 6. 衰减公式测试
# ============================================================


class TestCalculateDecay:
    """Ebbinghaus 遗忘曲线 + 间隔重复修正 (设计文档第三章 3.4)."""

    def test_fresh_practice_no_decay(self):
        """刚练习的 KC 衰减应接近 1.0."""
        now_ms = int(time.time() * 1000)
        decay = calculate_decay(
            p_know=0.9,
            last_practiced=now_ms,
            repetitions=1,
            current_ts=now_ms,
        )
        assert decay == pytest.approx(0.9, abs=0.05)

    def test_decay_decreases_with_time(self):
        """衰减随时间增加而减小."""
        now_ms = int(time.time() * 1000)
        recent = calculate_decay(0.9, now_ms - 1000, 1, now_ms)
        old = calculate_decay(0.9, now_ms - 72 * MS_PER_HOUR, 1, now_ms)
        assert old < recent

    def test_repetitions_increase_stability(self):
        """重复次数越多, 衰减越慢 (稳定性增加)."""
        now_ms = int(time.time() * 1000)
        elapsed = 24 * MS_PER_HOUR
        low_rep = calculate_decay(0.9, now_ms - elapsed, 1, now_ms)
        high_rep = calculate_decay(0.9, now_ms - elapsed, 10, now_ms)
        assert high_rep > low_rep

    def test_floor_at_prior_prob(self):
        """衰减后不低于先验概率."""
        now_ms = int(time.time() * 1000)
        very_old = calculate_decay(
            p_know=0.9,
            last_practiced=now_ms - 365 * 24 * MS_PER_HOUR,  # 1 年前
            repetitions=0,
            current_ts=now_ms,
        )
        assert very_old >= PRIOR_PROB

    def test_zero_p_know_stays_zero(self):
        """p_know=0 时有效掌握度应为 0 (或先验)."""
        now_ms = int(time.time() * 1000)
        result = calculate_decay(0.0, now_ms, 0, now_ms)
        assert result == 0.0

    def test_p_know_range_validation(self):
        with pytest.raises((ValueError, Exception)):
            calculate_decay(p_know=-0.1, last_practiced=0, repetitions=0, current_ts=0)
        with pytest.raises((ValueError, Exception)):
            calculate_decay(p_know=1.5, last_practiced=0, repetitions=0, current_ts=0)

    def test_repetitions_non_negative(self):
        with pytest.raises((ValueError, Exception)):
            calculate_decay(p_know=0.5, last_practiced=0, repetitions=-1, current_ts=0)


# ============================================================
# 7. 会话与 Fork 测试
# ============================================================


class TestSessionType:
    """会话类型枚举 (设计文档第五章 5.2)."""

    def test_has_five_types(self):
        types = list(SessionType)
        assert len(types) == 5
        assert SessionType.DIAGNOSIS
        assert SessionType.LEARNING
        assert SessionType.LAB_GUIDE
        assert SessionType.ASSESSMENT
        assert SessionType.QUERY

    def test_enum_values(self):
        assert SessionType.DIAGNOSIS.value == "diagnosis"
        assert SessionType.LEARNING.value == "learning"
        assert SessionType.LAB_GUIDE.value == "lab_guide"
        assert SessionType.ASSESSMENT.value == "assessment"


class TestSessionStatus:
    """会话状态枚举 (设计文档第五章 5.1)."""

    def test_has_four_statuses(self):
        statuses = list(SessionStatus)
        assert len(statuses) == 4
        assert SessionStatus.ACTIVE
        assert SessionStatus.PAUSED
        assert SessionStatus.FORKED
        assert SessionStatus.COMPLETED


class TestLearningSession:
    """学习会话模型 (设计文档第五章 5.1)."""

    def test_create_session_with_defaults(self):
        session = LearningSession(
            user_id="u-001",
            session_type=SessionType.LEARNING,
        )
        assert session.session_id.startswith("sess-")
        assert session.user_id == "u-001"
        assert session.session_type == SessionType.LEARNING
        assert session.parent_session_id is None
        assert session.fork_point_seq is None
        assert isinstance(session.context, ContextEnvelope)
        assert session.agent_states == {}
        assert session.interaction_log == []
        assert session.artifacts == []
        assert session.status == SessionStatus.ACTIVE
        assert session.checkpoint_indices == []
        assert session.created_at > 0

    def test_session_id_is_unique(self):
        s1 = LearningSession(user_id="u1", session_type=SessionType.LEARNING)
        s2 = LearningSession(user_id="u2", session_type=SessionType.LEARNING)
        assert s1.session_id != s2.session_id

    def test_forked_session(self):
        session = LearningSession(
            user_id="u-001",
            session_type=SessionType.LEARNING,
            parent_session_id="sess-parent-001",
            fork_point_seq=5,
            status=SessionStatus.FORKED,
        )
        assert session.parent_session_id == "sess-parent-001"
        assert session.fork_point_seq == 5
        assert session.status == SessionStatus.FORKED

    def test_add_checkpoint(self):
        session = LearningSession(user_id="u1", session_type=SessionType.LEARNING)
        session.add_checkpoint(0)
        session.add_checkpoint(1)
        assert len(session.checkpoint_indices) == 2

    def test_add_interaction(self):
        session = LearningSession(user_id="u1", session_type=SessionType.LEARNING)
        session.add_interaction({"type": "qa", "content": "hello"})
        assert len(session.interaction_log) == 1

    def test_add_artifact(self):
        session = LearningSession(user_id="u1", session_type=SessionType.LEARNING)
        session.add_artifact({"type": "card", "id": "art-001"})
        assert len(session.artifacts) == 1

    def test_to_dict(self):
        session = LearningSession(
            user_id="u-001",
            session_type=SessionType.DIAGNOSIS,
            status=SessionStatus.COMPLETED,
        )
        d = session.to_dict()
        assert d["user_id"] == "u-001"
        assert d["session_type"] == "diagnosis"
        assert d["status"] == "completed"
        assert "context" in d
        assert isinstance(d["context"], dict)

    def test_from_dict_roundtrip(self):
        session = LearningSession(
            user_id="u-001",
            session_type=SessionType.LEARNING,
        )
        d = session.to_dict()
        restored = LearningSession.from_dict(d)
        assert restored.user_id == session.user_id
        assert restored.session_type == session.session_type
        assert restored.status == session.status


class TestSessionFork:
    """Session Fork 数据结构 (设计文档第五章 5.5)."""

    def test_create_fork(self):
        fork = SessionFork(
            source_session_id="sess-001",
            fork_point_seq=3,
            fork_reason="学生手动",
            branch_label="路径A-先理论",
            snapshot_at_fork=ContextEnvelope(user_id="u1", session_id="sess-001"),
        )
        assert fork.fork_id.startswith("fork-")
        assert fork.source_session_id == "sess-001"
        assert fork.fork_point_seq == 3
        assert fork.fork_reason == "学生手动"
        assert fork.branch_label == "路径A-先理论"
        assert isinstance(fork.snapshot_at_fork, ContextEnvelope)
        assert fork.merge_target is None
        assert fork.is_merged is False

    def test_fork_id_is_unique(self):
        f1 = SessionFork(
            source_session_id="s1", fork_point_seq=0, fork_reason="test",
            branch_label="A", snapshot_at_fork=ContextEnvelope(user_id="u", session_id="s"),
        )
        f2 = SessionFork(
            source_session_id="s2", fork_point_seq=0, fork_reason="test",
            branch_label="B", snapshot_at_fork=ContextEnvelope(user_id="u", session_id="s"),
        )
        assert f1.fork_id != f2.fork_id

    def test_merged_fork(self):
        fork = SessionFork(
            source_session_id="sess-001",
            fork_point_seq=3,
            fork_reason="A/B测试",
            branch_label="路径B",
            snapshot_at_fork=ContextEnvelope(user_id="u", session_id="s"),
            merge_target="sess-001",
            is_merged=True,
        )
        assert fork.is_merged is True
        assert fork.merge_target == "sess-001"

    def test_to_dict(self):
        fork = SessionFork(
            source_session_id="sess-001",
            fork_point_seq=3,
            fork_reason="教师建议",
            branch_label="路径A",
            snapshot_at_fork=ContextEnvelope(user_id="u", session_id="s"),
        )
        d = fork.to_dict()
        assert d["source_session_id"] == "sess-001"
        assert d["fork_point_seq"] == 3
        assert d["is_merged"] is False
        assert isinstance(d["snapshot_at_fork"], dict)


# ============================================================
# 8. 审计与脱敏测试
# ============================================================


class TestDataLevel:
    """数据分级枚举 (设计文档第六章 6.1)."""

    def test_has_four_levels(self):
        levels = list(DataLevel)
        assert len(levels) == 4
        assert DataLevel.L1_PUBLIC
        assert DataLevel.L2_INTERNAL
        assert DataLevel.L3_SENSITIVE
        assert DataLevel.L4_CONFIDENTIAL

    def test_level_ordering(self):
        """L1 < L2 < L3 < L4 (公开度递减, 敏感度递增)."""
        assert DataLevel.L1_PUBLIC.value < DataLevel.L2_INTERNAL.value
        assert DataLevel.L2_INTERNAL.value < DataLevel.L3_SENSITIVE.value
        assert DataLevel.L3_SENSITIVE.value < DataLevel.L4_CONFIDENTIAL.value


class TestAuditAction:
    """审计操作类型枚举 (设计文档第七章 7.4)."""

    def test_has_all_actions(self):
        actions = list(AuditAction)
        assert AuditAction.VIEW
        assert AuditAction.EXPORT
        assert AuditAction.MODIFY
        assert AuditAction.DELETE
        assert AuditAction.AGENT_INVOKE
        assert AuditAction.APPROVE
        assert AuditAction.REJECT
        assert AuditAction.LOGIN
        assert AuditAction.LOGOUT

    def test_str_enum(self):
        assert isinstance(AuditAction.VIEW, str)


class TestAuditResult:
    """审计结果枚举 (设计文档第七章 7.4)."""

    def test_has_three_results(self):
        assert AuditResult.SUCCESS
        assert AuditResult.DENIED
        assert AuditResult.ERROR


class TestAuditLogEntry:
    """审计日志条目 (设计文档第七章 7.4)."""

    def test_create_entry(self):
        entry = AuditLogEntry(
            actor_id="CS20240001",
            actor_role=UserRole.UNDERGRAD,
            action=AuditAction.VIEW,
            target_resource="kb:dy3_energy_level",
            target_data_level=DataLevel.L2_INTERNAL,
            purpose="学习查阅",
            result=AuditResult.SUCCESS,
        )
        assert entry.log_id.startswith("audit-")
        assert entry.actor_id == "CS20240001"
        assert entry.action == AuditAction.VIEW
        assert entry.result == AuditResult.SUCCESS
        assert entry.timestamp > 0
        assert entry.session_id is None
        assert entry.ip_hash is None

    def test_entry_with_session(self):
        entry = AuditLogEntry(
            actor_id="CS20240001",
            actor_role=UserRole.UNDERGRAD,
            action=AuditAction.AGENT_INVOKE,
            target_resource="agent:diagnosis",
            target_data_level=DataLevel.L3_SENSITIVE,
            purpose="学情诊断",
            result=AuditResult.SUCCESS,
            session_id="sess-001",
            ip_hash="a3f2b7c1",
        )
        assert entry.session_id == "sess-001"
        assert entry.ip_hash == "a3f2b7c1"

    def test_to_dict(self):
        entry = AuditLogEntry(
            actor_id="CS20240001",
            actor_role=UserRole.TEACHER,
            action=AuditAction.EXPORT,
            target_resource="report:student_001",
            target_data_level=DataLevel.L3_SENSITIVE,
            purpose="教学评估",
            result=AuditResult.SUCCESS,
        )
        d = entry.to_dict()
        assert d["actor_id"] == "CS20240001"
        assert d["actor_role"] == "teacher"
        assert d["action"] == "view" if False else d["action"] == "export"
        assert d["result"] == "success"


# ============================================================
# 9. HiTL 协同枚举测试
# ============================================================


class TestHiTLType:
    """HiTL 协同场景类型 (设计文档第四章 4.1)."""

    def test_has_four_types(self):
        types = list(HiTLType)
        assert len(types) == 4
        assert HiTLType.CONFIRMATION
        assert HiTLType.CORRECTION
        assert HiTLType.CREATIVE
        assert HiTLType.EMERGENCY


class TestHiTLPriority:
    """HiTL 优先级枚举 (设计文档第四章)."""

    def test_has_four_priorities(self):
        priorities = list(HiTLPriority)
        assert len(priorities) == 4
        assert HiTLPriority.P0
        assert HiTLPriority.P1
        assert HiTLPriority.P2
        assert HiTLPriority.P3

    def test_priority_ordering(self):
        assert HiTLPriority.P0.value < HiTLPriority.P1.value
        assert HiTLPriority.P1.value < HiTLPriority.P2.value
        assert HiTLPriority.P2.value < HiTLPriority.P3.value


class TestConfidenceGateResult:
    """置信度门控结果 (设计文档第四章 4.2)."""

    def test_has_three_results(self):
        results = list(ConfidenceGateResult)
        assert len(results) == 3
        assert ConfidenceGateResult.PASS
        assert ConfidenceGateResult.WARNING
        assert ConfidenceGateResult.BLOCK

    def test_evaluate_high_confidence(self):
        """confidence >= 0.85 → PASS."""
        result = ConfidenceGateResult.evaluate(0.9)
        assert result == ConfidenceGateResult.PASS

    def test_evaluate_medium_confidence(self):
        """0.4 <= confidence < 0.85 → WARNING."""
        result = ConfidenceGateResult.evaluate(0.5)
        assert result == ConfidenceGateResult.WARNING

    def test_evaluate_low_confidence(self):
        """confidence < 0.4 → BLOCK."""
        result = ConfidenceGateResult.evaluate(0.3)
        assert result == ConfidenceGateResult.BLOCK

    def test_evaluate_boundary_pass(self):
        result = ConfidenceGateResult.evaluate(0.85)
        assert result == ConfidenceGateResult.PASS

    def test_evaluate_boundary_warning(self):
        result = ConfidenceGateResult.evaluate(0.4)
        assert result == ConfidenceGateResult.WARNING

    def test_evaluate_boundary_block(self):
        result = ConfidenceGateResult.evaluate(0.39)
        assert result == ConfidenceGateResult.BLOCK


class TestFeedbackType:
    """反馈类型枚举 (设计文档第七章 7.5 feedback_options)."""

    def test_has_four_types(self):
        types = list(FeedbackType)
        assert len(types) == 4
        assert FeedbackType.UNDERSTOOD
        assert FeedbackType.NEED_MORE
        assert FeedbackType.INCORRECT
        assert FeedbackType.REPORT


# ============================================================
# 10. HiTL 数据模型测试
# ============================================================


class TestApprovalRequest:
    """HiTL 确认请求 (设计文档第四章 4.1, 第八章 8.4)."""

    def test_create_request(self):
        req = ApprovalRequest(
            user_id="u-001",
            session_id="sess-001",
            hitl_type=HiTLType.CONFIRMATION,
            content="请确认已理解 Dy3+ 能级跃迁",
            priority=HiTLPriority.P2,
        )
        assert req.request_id.startswith("hitl-")
        assert req.user_id == "u-001"
        assert req.hitl_type == HiTLType.CONFIRMATION
        assert req.status == "pending"

    def test_request_id_unique(self):
        r1 = ApprovalRequest(
            user_id="u", session_id="s", hitl_type=HiTLType.CONFIRMATION, content="c"
        )
        r2 = ApprovalRequest(
            user_id="u", session_id="s", hitl_type=HiTLType.CONFIRMATION, content="c"
        )
        assert r1.request_id != r2.request_id

    def test_to_dict(self):
        req = ApprovalRequest(
            user_id="u-001",
            session_id="sess-001",
            hitl_type=HiTLType.CORRECTION,
            content="内容有误",
            priority=HiTLPriority.P1,
        )
        d = req.to_dict()
        assert d["user_id"] == "u-001"
        assert d["hitl_type"] == "correction"
        assert d["priority"] == "p1"


class TestApprovalResponse:
    """HiTL 确认响应."""

    def test_create_response(self):
        resp = ApprovalResponse(
            request_id="hitl-001",
            responder_id="u-001",
            approved=True,
            comment="确认理解",
        )
        assert resp.request_id == "hitl-001"
        assert resp.approved is True
        assert resp.comment == "确认理解"

    def test_rejected_response(self):
        resp = ApprovalResponse(
            request_id="hitl-001",
            responder_id="u-001",
            approved=False,
            comment="内容不准确",
        )
        assert resp.approved is False


class TestFeedbackReport:
    """HiTL 反馈报告 (设计文档第四章 4.4)."""

    def test_create_report(self):
        report = FeedbackReport(
            user_id="u-001",
            session_id="sess-001",
            feedback_type=FeedbackType.INCORRECT,
            content="Dy3+ 的 4f-4f 跃迁是禁戒跃迁，描述有误",
            artifact_id="art-001",
        )
        assert report.report_id.startswith("fb-")
        assert report.feedback_type == FeedbackType.INCORRECT
        assert report.user_id == "u-001"

    def test_report_with_correction(self):
        report = FeedbackReport(
            user_id="u-001",
            session_id="sess-001",
            feedback_type=FeedbackType.NEED_MORE,
            content="需要更多关于黄蓝比调控的实例",
        )
        assert report.feedback_type == FeedbackType.NEED_MORE
        assert report.artifact_id is None

    def test_to_dict(self):
        report = FeedbackReport(
            user_id="u-001",
            session_id="sess-001",
            feedback_type=FeedbackType.REPORT,
            content="安全问题",
        )
        d = report.to_dict()
        assert d["feedback_type"] == "report"
        assert d["user_id"] == "u-001"


class TestEmergencyAlert:
    """紧急干预警报 (设计文档第四章 4.3)."""

    def test_create_alert(self):
        alert = EmergencyAlert(
            session_id="sess-001",
            user_id="u-001",
            trigger_reason="认知负荷过高",
            trigger_value=0.97,
        )
        assert alert.alert_id.startswith("emg-")
        assert alert.session_id == "sess-001"
        assert alert.trigger_reason == "认知负荷过高"
        assert alert.trigger_value == 0.97
        assert alert.is_resolved is False

    def test_alert_id_unique(self):
        a1 = EmergencyAlert(
            session_id="s", user_id="u", trigger_reason="r", trigger_value=0.96
        )
        a2 = EmergencyAlert(
            session_id="s", user_id="u", trigger_reason="r", trigger_value=0.96
        )
        assert a1.alert_id != a2.alert_id

    def test_to_dict(self):
        alert = EmergencyAlert(
            session_id="sess-001",
            user_id="u-001",
            trigger_reason="连续错误>=10次",
            trigger_value=10,
        )
        d = alert.to_dict()
        assert d["session_id"] == "sess-001"
        assert d["trigger_reason"] == "连续错误>=10次"
        assert d["is_resolved"] is False


# ============================================================
# 11. 模块导出测试
# ============================================================


class TestModuleExports:
    """验证 __init__.py 导出完整性."""

    def test_import_from_init(self):
        from dy3_polaris.l1 import models as l1_models

        assert hasattr(l1_models, "UserRole")
        assert hasattr(l1_models, "ContextEnvelope")
        assert hasattr(l1_models, "LearningSession")
        assert hasattr(l1_models, "calculate_decay")

    def test_all_exports_defined(self):
        from dy3_polaris.l1.models import __all__

        assert "UserRole" in __all__
        assert "ContextEnvelope" in __all__
        assert "LearningSession" in __all__
        assert "calculate_decay" in __all__
        assert "AuditLogEntry" in __all__
        assert "EmergencyAlert" in __all__
