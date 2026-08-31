"""L1 访问控制模块测试 — RBAC 权限矩阵 + ABAC 策略评估 + 综合访问控制管理.

测试覆盖:
1. RBACMatrix: 13 项权限 × 5 种角色的权限矩阵
2. ABACEvaluator: Cedar 策略子集评估引擎
3. AccessControlManager: RBAC + ABAC 混合访问控制
4. 权限条件标记 (限导师授权范围 / 仅课程范围 / 限数量)
5. 审计日志集成
6. 线程安全
7. 边界条件与异常处理
"""

import threading
import time

import pytest

from dy3_polaris.l1.access_control import (
    ABACEvaluator,
    ABACPolicy,
    AccessControlManager,
    AccessDecision,
    AccessDeniedError,
    AccessRequest,
    AccessResult,
    ActionType,
    RBACMatrix,
    ResourceType,
)
from dy3_polaris.l1.models import (
    ABACAttributes,
    AuditResult,
    GradeLevel,
    LabAccessTier,
    MajorDirection,
    Permission,
    User,
    UserRole,
    UserStatus,
)


# ============================================================
# 1. RBACMatrix 测试
# ============================================================


class TestRBACMatrix:
    """RBAC 权限矩阵测试."""

    def test_undergrad_permissions(self):
        """本科生: 6 项基础权限 (含知识生成Agent限数量)."""
        matrix = RBACMatrix()
        perms = matrix.get_permissions(UserRole.UNDERGRAD)
        assert Permission.KB_PUBLIC_READ in perms
        assert Permission.AGENT_DIAGNOSIS in perms
        assert Permission.AGENT_KNOWLEDGE_GEN in perms
        assert Permission.AGENT_GUIDE in perms
        assert Permission.VIEW_OWN_REPORT in perms
        assert Permission.HITL_CONFIRM in perms
        assert len(perms) == 6

    def test_graduate_permissions(self):
        """研究生: 8 项权限 (含内部数据访问和导出)."""
        matrix = RBACMatrix()
        perms = matrix.get_permissions(UserRole.GRADUATE)
        assert Permission.KB_PUBLIC_READ in perms
        assert Permission.KB_INTERNAL_DATA_ACCESS in perms
        assert Permission.AGENT_DIAGNOSIS in perms
        assert Permission.AGENT_KNOWLEDGE_GEN in perms
        assert Permission.AGENT_GUIDE in perms
        assert Permission.VIEW_OWN_REPORT in perms
        assert Permission.EXPORT_REPORT in perms
        assert Permission.HITL_CONFIRM in perms
        assert len(perms) == 8

    def test_teacher_permissions(self):
        """教师: 8 项权限 (含写入和学生报告查看)."""
        matrix = RBACMatrix()
        perms = matrix.get_permissions(UserRole.TEACHER)
        assert Permission.KB_PUBLIC_READ in perms
        assert Permission.KB_INTERNAL_DATA_ACCESS in perms
        assert Permission.KB_WRITE_EDIT in perms
        assert Permission.AGENT_REVIEW in perms
        assert Permission.AGENT_GUIDE in perms
        assert Permission.VIEW_STUDENT_REPORT in perms
        assert Permission.EXPORT_REPORT in perms
        assert Permission.HITL_CONFIRM in perms
        assert len(perms) == 8

    def test_admin_permissions(self):
        """管理员: 全部 13 项权限."""
        matrix = RBACMatrix()
        perms = matrix.get_permissions(UserRole.ADMIN)
        assert len(perms) == 13
        for perm in Permission:
            assert perm in perms

    def test_alumni_permissions(self):
        """校友: 2 项只读权限."""
        matrix = RBACMatrix()
        perms = matrix.get_permissions(UserRole.ALUMNI)
        assert Permission.KB_PUBLIC_READ in perms
        assert Permission.VIEW_OWN_REPORT in perms
        assert len(perms) == 2

    def test_check_permission_allowed(self):
        """本科生有公开读权限."""
        matrix = RBACMatrix()
        assert matrix.check_permission(UserRole.UNDERGRAD, Permission.KB_PUBLIC_READ) is True

    def test_check_permission_denied(self):
        """本科生无写入权限."""
        matrix = RBACMatrix()
        assert matrix.check_permission(UserRole.UNDERGRAD, Permission.KB_WRITE_EDIT) is False

    def test_admin_has_all_permissions(self):
        """管理员拥有所有权限."""
        matrix = RBACMatrix()
        for perm in Permission:
            assert matrix.check_permission(UserRole.ADMIN, perm) is True

    def test_alumni_cannot_invoke_agents(self):
        """校友不能调用 Agent."""
        matrix = RBACMatrix()
        assert matrix.check_permission(UserRole.ALUMNI, Permission.AGENT_DIAGNOSIS) is False
        assert matrix.check_permission(UserRole.ALUMNI, Permission.AGENT_GUIDE) is False

    def test_undergrad_cannot_access_internal_data(self):
        """本科生不能访问内部实验数据."""
        matrix = RBACMatrix()
        assert matrix.check_permission(UserRole.UNDERGRAD, Permission.KB_INTERNAL_DATA_ACCESS) is False

    def test_undergrad_cannot_view_student_reports(self):
        """本科生不能查看学生报告."""
        matrix = RBACMatrix()
        assert matrix.check_permission(UserRole.UNDERGRAD, Permission.VIEW_STUDENT_REPORT) is False

    def test_get_all_role_permission_matrix(self):
        """获取完整角色权限矩阵."""
        matrix = RBACMatrix()
        full_matrix = matrix.get_matrix()
        assert UserRole.UNDERGRAD in full_matrix
        assert UserRole.GRADUATE in full_matrix
        assert UserRole.TEACHER in full_matrix
        assert UserRole.ADMIN in full_matrix
        assert UserRole.ALUMNI in full_matrix

    def test_has_conditional_marker(self):
        """权限带条件标记 (限导师授权范围 / 仅课程范围)."""
        matrix = RBACMatrix()
        # 研究生内部数据访问 → 限导师授权范围
        assert matrix.has_conditional_marker(
            UserRole.GRADUATE, Permission.KB_INTERNAL_DATA_ACCESS
        )
        # 教师写入 → 仅课程范围
        assert matrix.has_conditional_marker(
            UserRole.TEACHER, Permission.KB_WRITE_EDIT
        )
        # 本科生公开读 → 无条件
        assert not matrix.has_conditional_marker(
            UserRole.UNDERGRAD, Permission.KB_PUBLIC_READ
        )

    def test_get_conditional_description(self):
        """获取条件描述文本."""
        matrix = RBACMatrix()
        desc = matrix.get_conditional_description(
            UserRole.GRADUATE, Permission.KB_INTERNAL_DATA_ACCESS
        )
        assert "导师" in desc

    def test_add_custom_permission(self):
        """动态添加自定义权限到角色."""
        matrix = RBACMatrix()
        matrix.add_permission(UserRole.UNDERGRAD, Permission.EXPORT_REPORT)
        assert matrix.check_permission(UserRole.UNDERGRAD, Permission.EXPORT_REPORT)

    def test_remove_permission(self):
        """动态移除角色权限."""
        matrix = RBACMatrix()
        matrix.remove_permission(UserRole.TEACHER, Permission.KB_WRITE_EDIT)
        assert not matrix.check_permission(UserRole.TEACHER, Permission.KB_WRITE_EDIT)


# ============================================================
# 2. ABACEvaluator 测试
# ============================================================


class TestABACEvaluator:
    """ABAC 策略评估引擎测试."""

    def _make_graduate(self, supervisor_id: str = "teacher-001") -> User:
        return User(
            student_id="CS20240002",
            institution_id="inst-001",
            role=UserRole.GRADUATE,
            abac_attributes=ABACAttributes(
                grade_level=GradeLevel.MASTER,
                lab_access_tier=LabAccessTier.TIER2,
                supervisor_id=supervisor_id,
            ),
        )

    def _make_undergrad(self, course_progress: float = 0.5) -> User:
        return User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
            abac_attributes=ABACAttributes(
                grade_level=GradeLevel.SOPHOMORE,
                course_progress=course_progress,
                lab_access_tier=LabAccessTier.TIER1,
            ),
        )

    def test_lab_data_access_allowed_same_supervisor(self):
        """研究生可访问导师授权范围内的实验数据."""
        user = self._make_graduate(supervisor_id="teacher-001")
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.ACCESS_LAB_DATA,
            resource_type=ResourceType.LAB_DATASET,
            context={
                "supervisor_id": "teacher-001",
                "required_tier": LabAccessTier.TIER1.value,
            },
        )
        assert result.allowed

    def test_lab_data_access_denied_different_supervisor(self):
        """研究生不能访问其他导师的实验数据."""
        user = self._make_graduate(supervisor_id="teacher-001")
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.ACCESS_LAB_DATA,
            resource_type=ResourceType.LAB_DATASET,
            context={
                "supervisor_id": "teacher-002",
                "required_tier": LabAccessTier.TIER1.value,
            },
        )
        assert not result.allowed

    def test_lab_data_access_denied_insufficient_tier(self):
        """研究生实验室权限等级不足."""
        user = self._make_graduate(supervisor_id="teacher-001")
        user.abac_attributes.lab_access_tier = LabAccessTier.TIER1
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.ACCESS_LAB_DATA,
            resource_type=ResourceType.LAB_DATASET,
            context={
                "supervisor_id": "teacher-001",
                "required_tier": LabAccessTier.TIER3.value,
            },
        )
        assert not result.allowed

    def test_undergrad_agent_call_within_limit(self):
        """本科生 Agent 调用次数未超限."""
        user = self._make_undergrad()
        user.abac_attributes.daily_agent_calls = 10
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.INVOKE_KNOWLEDGE_AGENT,
            resource_type=ResourceType.AGENT,
            context={},
        )
        assert result.allowed

    def test_undergrad_agent_call_exceeds_limit(self):
        """本科生 Agent 调用次数超限."""
        user = self._make_undergrad()
        user.abac_attributes.daily_agent_calls = 25  # 超过 MAX_DAILY_AGENT_CALLS=20
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.INVOKE_KNOWLEDGE_AGENT,
            resource_type=ResourceType.AGENT,
            context={},
        )
        assert not result.allowed

    def test_course_progress_blocks_lab_guide(self):
        """课程进度 < 0.3 时不推荐综合实验指导."""
        user = self._make_undergrad(course_progress=0.2)
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.ACCESS_LAB_GUIDE,
            resource_type=ResourceType.LAB_GUIDE,
            context={"guide_type": "comprehensive"},
        )
        assert not result.allowed

    def test_course_progress_allows_lab_guide(self):
        """课程进度 >= 0.3 时可以推荐实验指导."""
        user = self._make_undergrad(course_progress=0.5)
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.ACCESS_LAB_GUIDE,
            resource_type=ResourceType.LAB_GUIDE,
            context={"guide_type": "comprehensive"},
        )
        assert result.allowed

    def test_grade_level_blocks_advanced_module(self):
        """大二以下不可访问高级模块."""
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
            abac_attributes=ABACAttributes(
                grade_level=GradeLevel.FRESHMAN,
            ),
        )
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.ACCESS_ADVANCED_MODULE,
            resource_type=ResourceType.KNOWLEDGE_MODULE,
            context={"module_difficulty": "advanced"},
        )
        assert not result.allowed

    def test_grade_level_allows_advanced_module(self):
        """大三及以上可访问高级模块."""
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
            abac_attributes=ABACAttributes(
                grade_level=GradeLevel.JUNIOR,
            ),
        )
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.ACCESS_ADVANCED_MODULE,
            resource_type=ResourceType.KNOWLEDGE_MODULE,
            context={"module_difficulty": "advanced"},
        )
        assert result.allowed

    def test_no_matching_policy_allows_by_default(self):
        """无匹配 ABAC 策略时默认放行 (RBAC 已做粗粒度控制)."""
        user = self._make_undergrad()
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.VIEW_OWN_REPORT,
            resource_type=ResourceType.REPORT,
            context={},
        )
        assert result.allowed

    def test_custom_policy_add(self):
        """添加自定义 ABAC 策略."""
        evaluator = ABACEvaluator()
        policy = ABACPolicy(
            policy_id="custom-001",
            name="自定义策略",
            description="测试自定义策略",
            applicable_roles=[UserRole.UNDERGRAD],
            action=ActionType.VIEW_OWN_REPORT,
            resource_type=ResourceType.REPORT,
            condition=lambda user, ctx: user.abac_attributes.course_progress > 0.1,
            decision=AccessDecision.ALLOW,
        )
        evaluator.add_policy(policy)
        assert len(evaluator.list_policies()) >= 1

    def test_custom_policy_remove(self):
        """移除自定义 ABAC 策略."""
        evaluator = ABACEvaluator()
        policy = ABACPolicy(
            policy_id="custom-remove-001",
            name="待移除策略",
            description="测试移除",
            applicable_roles=[UserRole.UNDERGRAD],
            action=ActionType.VIEW_OWN_REPORT,
            resource_type=ResourceType.REPORT,
            condition=lambda user, ctx: True,
            decision=AccessDecision.DENY,
        )
        evaluator.add_policy(policy)
        assert len(evaluator.list_policies()) >= 1
        evaluator.remove_policy("custom-remove-001")
        # 移除后该策略不再生效

    def test_explicit_deny_overrides_allow(self):
        """显式 DENY 覆盖 ALLOW (同优先级)."""
        evaluator = ABACEvaluator()
        # 先添加 ALLOW 策略
        allow_policy = ABACPolicy(
            policy_id="allow-001",
            name="允许策略",
            description="允许",
            applicable_roles=[UserRole.UNDERGRAD],
            action=ActionType.VIEW_OWN_REPORT,
            resource_type=ResourceType.REPORT,
            condition=lambda user, ctx: True,
            decision=AccessDecision.ALLOW,
            priority=10,
        )
        # 再添加 DENY 策略 (同优先级)
        deny_policy = ABACPolicy(
            policy_id="deny-001",
            name="拒绝策略",
            description="拒绝",
            applicable_roles=[UserRole.UNDERGRAD],
            action=ActionType.VIEW_OWN_REPORT,
            resource_type=ResourceType.REPORT,
            condition=lambda user, ctx: True,
            decision=AccessDecision.DENY,
            priority=10,
        )
        evaluator.add_policy(allow_policy)
        evaluator.add_policy(deny_policy)

        user = self._make_undergrad()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.VIEW_OWN_REPORT,
            resource_type=ResourceType.REPORT,
            context={},
        )
        assert not result.allowed

    def test_higher_priority_allow_overrides_lower_deny(self):
        """高优先级 ALLOW 覆盖低优先级 DENY."""
        evaluator = ABACEvaluator()
        allow_policy = ABACPolicy(
            policy_id="high-allow-001",
            name="高优先级允许",
            description="高优先级允许",
            applicable_roles=[UserRole.UNDERGRAD],
            action=ActionType.VIEW_OWN_REPORT,
            resource_type=ResourceType.REPORT,
            condition=lambda user, ctx: True,
            decision=AccessDecision.ALLOW,
            priority=100,
        )
        deny_policy = ABACPolicy(
            policy_id="low-deny-001",
            name="低优先级拒绝",
            description="低优先级拒绝",
            applicable_roles=[UserRole.UNDERGRAD],
            action=ActionType.VIEW_OWN_REPORT,
            resource_type=ResourceType.REPORT,
            condition=lambda user, ctx: True,
            decision=AccessDecision.DENY,
            priority=10,
        )
        evaluator.add_policy(allow_policy)
        evaluator.add_policy(deny_policy)

        user = self._make_undergrad()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.VIEW_OWN_REPORT,
            resource_type=ResourceType.REPORT,
            context={},
        )
        assert result.allowed

    def test_alumni_readonly_enforcement(self):
        """校友只能只读访问."""
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.ALUMNI,
            status=UserStatus.ALUMNI,
            abac_attributes=ABACAttributes(lab_access_tier=LabAccessTier.TIER0),
        )
        evaluator = ABACEvaluator()
        result = evaluator.evaluate(
            user=user,
            action=ActionType.WRITE_KB,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context={},
        )
        assert not result.allowed


# ============================================================
# 3. AccessControlManager 测试
# ============================================================


class TestAccessControlManager:
    """RBAC + ABAC 混合访问控制管理器测试."""

    def _make_user(
        self,
        role: UserRole = UserRole.UNDERGRAD,
        status: UserStatus = UserStatus.ACTIVE,
        course_progress: float = 0.5,
        supervisor_id: str | None = None,
        lab_tier: LabAccessTier = LabAccessTier.TIER1,
    ) -> User:
        return User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=role,
            status=status,
            abac_attributes=ABACAttributes(
                grade_level=GradeLevel.SOPHOMORE,
                course_progress=course_progress,
                lab_access_tier=lab_tier,
                supervisor_id=supervisor_id,
            ),
        )

    def test_check_access_rbac_allowed(self):
        """RBAC 允许的权限, 无 ABAC 约束 → 放行."""
        acm = AccessControlManager()
        user = self._make_user()
        result = acm.check_access(
            user=user,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context={},
        )
        assert result.allowed

    def test_check_access_rbac_denied(self):
        """RBAC 拒绝的权限 → 直接拒绝."""
        acm = AccessControlManager()
        user = self._make_user(role=UserRole.UNDERGRAD)
        result = acm.check_access(
            user=user,
            permission=Permission.KB_WRITE_EDIT,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context={},
        )
        assert not result.allowed
        assert "RBAC" in result.reason or "权限" in result.reason

    def test_check_access_abac_denied(self):
        """RBAC 允许但 ABAC 拒绝 → 拒绝."""
        acm = AccessControlManager()
        user = self._make_user(course_progress=0.2)  # 进度过低
        result = acm.check_access(
            user=user,
            permission=Permission.AGENT_GUIDE,
            resource_type=ResourceType.LAB_GUIDE,
            context={"guide_type": "comprehensive"},
            action=ActionType.ACCESS_LAB_GUIDE,
        )
        assert not result.allowed

    def test_check_access_both_allowed(self):
        """RBAC 和 ABAC 都允许 → 放行."""
        acm = AccessControlManager()
        user = self._make_user(course_progress=0.5)
        result = acm.check_access(
            user=user,
            permission=Permission.AGENT_GUIDE,
            resource_type=ResourceType.LAB_GUIDE,
            context={"guide_type": "comprehensive"},
            action=ActionType.ACCESS_LAB_GUIDE,
        )
        assert result.allowed

    def test_enforce_allowed(self):
        """enforce() 在允许时返回 True."""
        acm = AccessControlManager()
        user = self._make_user()
        assert acm.enforce(
            user=user,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
        )

    def test_enforce_denied_raises(self):
        """enforce() 在拒绝时抛出 AccessDeniedError."""
        acm = AccessControlManager()
        user = self._make_user(role=UserRole.UNDERGRAD)
        with pytest.raises(AccessDeniedError):
            acm.enforce(
                user=user,
                permission=Permission.KB_WRITE_EDIT,
                resource_type=ResourceType.KNOWLEDGE_BASE,
            )

    def test_suspended_user_denied(self):
        """已停用用户所有操作被拒绝."""
        acm = AccessControlManager()
        user = self._make_user(status=UserStatus.SUSPENDED)
        result = acm.check_access(
            user=user,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context={},
        )
        assert not result.allowed
        assert "停用" in result.reason or "SUSPENDED" in result.reason

    def test_alumni_write_denied(self):
        """校友写入操作被拒绝."""
        acm = AccessControlManager()
        user = self._make_user(role=UserRole.ALUMNI, status=UserStatus.ALUMNI)
        result = acm.check_access(
            user=user,
            permission=Permission.KB_WRITE_EDIT,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context={},
        )
        assert not result.allowed

    def test_alumni_read_allowed(self):
        """校友只读操作允许."""
        acm = AccessControlManager()
        user = self._make_user(role=UserRole.ALUMNI, status=UserStatus.ALUMNI)
        result = acm.check_access(
            user=user,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context={},
        )
        assert result.allowed

    def test_admin_all_access(self):
        """管理员可访问一切."""
        acm = AccessControlManager()
        user = self._make_user(role=UserRole.ADMIN)
        for perm in Permission:
            result = acm.check_access(
                user=user,
                permission=perm,
                resource_type=ResourceType.KNOWLEDGE_BASE,
                context={},
            )
            assert result.allowed, f"Admin should have {perm.value}"

    def test_access_log_recorded(self):
        """访问决策记录到日志."""
        acm = AccessControlManager()
        user = self._make_user()
        acm.check_access(
            user=user,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context={},
        )
        logs = acm.get_access_log()
        assert len(logs) >= 1
        assert logs[-1].user_id == user.user_id

    def test_access_log_filter_by_user(self):
        """按用户过滤访问日志."""
        acm = AccessControlManager()
        user1 = self._make_user()
        user2 = User(
            student_id="CS20240002",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        acm.check_access(
            user=user1,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
        )
        acm.check_access(
            user=user2,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
        )
        logs = acm.get_access_log(user_id=user1.user_id)
        assert all(log.user_id == user1.user_id for log in logs)

    def test_get_stats(self):
        """获取访问控制统计信息."""
        acm = AccessControlManager()
        user = self._make_user()
        acm.check_access(
            user=user,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
        )
        acm.check_access(
            user=user,
            permission=Permission.KB_WRITE_EDIT,
            resource_type=ResourceType.KNOWLEDGE_BASE,
        )
        stats = acm.get_stats()
        assert stats["total_access_requests"] >= 2
        assert stats["allowed_requests"] >= 1
        assert stats["denied_requests"] >= 1

    def test_graduate_lab_data_access_with_supervisor(self):
        """研究生在导师范围内可访问实验数据."""
        acm = AccessControlManager()
        user = self._make_user(
            role=UserRole.GRADUATE,
            supervisor_id="teacher-001",
            lab_tier=LabAccessTier.TIER2,
        )
        result = acm.check_access(
            user=user,
            permission=Permission.KB_INTERNAL_DATA_ACCESS,
            resource_type=ResourceType.LAB_DATASET,
            context={
                "supervisor_id": "teacher-001",
                "required_tier": LabAccessTier.TIER1.value,
            },
            action=ActionType.ACCESS_LAB_DATA,
        )
        assert result.allowed

    def test_graduate_lab_data_access_wrong_supervisor(self):
        """研究生不能访问其他导师的数据."""
        acm = AccessControlManager()
        user = self._make_user(
            role=UserRole.GRADUATE,
            supervisor_id="teacher-001",
            lab_tier=LabAccessTier.TIER2,
        )
        result = acm.check_access(
            user=user,
            permission=Permission.KB_INTERNAL_DATA_ACCESS,
            resource_type=ResourceType.LAB_DATASET,
            context={
                "supervisor_id": "teacher-002",
                "required_tier": LabAccessTier.TIER1.value,
            },
            action=ActionType.ACCESS_LAB_DATA,
        )
        assert not result.allowed

    def test_undergrad_agent_frequency_limit(self):
        """本科生 Agent 调用频率限制."""
        acm = AccessControlManager()
        user = self._make_user()
        user.abac_attributes.daily_agent_calls = 25  # 超限
        result = acm.check_access(
            user=user,
            permission=Permission.AGENT_KNOWLEDGE_GEN,
            resource_type=ResourceType.AGENT,
            context={},
            action=ActionType.INVOKE_KNOWLEDGE_AGENT,
        )
        assert not result.allowed

    def test_audit_integration(self):
        """访问拒绝时记录审计日志."""
        acm = AccessControlManager()
        user = self._make_user(role=UserRole.UNDERGRAD)
        acm.check_access(
            user=user,
            permission=Permission.KB_WRITE_EDIT,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context={},
        )
        audit_logs = acm.get_audit_logs()
        assert len(audit_logs) >= 1
        denied_log = audit_logs[-1]
        assert denied_log.result == AuditResult.DENIED


# ============================================================
# 4. AccessDeniedError 测试
# ============================================================


class TestAccessDeniedError:
    """访问拒绝异常测试."""

    def test_error_contains_user_id(self):
        """异常包含用户 ID."""
        err = AccessDeniedError(
            user_id="u-001",
            permission=Permission.KB_WRITE_EDIT,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            reason="权限不足",
        )
        assert err.user_id == "u-001"

    def test_error_contains_permission(self):
        """异常包含权限信息."""
        err = AccessDeniedError(
            user_id="u-001",
            permission=Permission.KB_WRITE_EDIT,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            reason="权限不足",
        )
        assert err.permission == Permission.KB_WRITE_EDIT

    def test_error_jsonrpc_code(self):
        """异常 JSON-RPC 码正确."""
        err = AccessDeniedError(
            user_id="u-001",
            permission=Permission.KB_WRITE_EDIT,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            reason="权限不足",
        )
        assert err._jsonrpc_code() == -32206


# ============================================================
# 5. 线程安全测试
# ============================================================


class TestThreadSafety:
    """访问控制管理器线程安全测试."""

    def test_concurrent_check_access(self):
        """并发访问检查不抛异常."""
        acm = AccessControlManager()
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        errors = []

        def worker():
            try:
                for _ in range(100):
                    acm.check_access(
                        user=user,
                        permission=Permission.KB_PUBLIC_READ,
                        resource_type=ResourceType.KNOWLEDGE_BASE,
                        context={},
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(acm.get_access_log(limit=0)) == 1000

    def test_concurrent_policy_modification(self):
        """并发策略修改不抛异常."""
        acm = AccessControlManager()
        errors = []

        def adder():
            try:
                for i in range(20):
                    policy = ABACPolicy(
                        policy_id=f"concurrent-{threading.get_ident()}-{i}",
                        name=f"并发策略 {i}",
                        description="测试",
                        applicable_roles=[UserRole.UNDERGRAD],
                        action=ActionType.VIEW_OWN_REPORT,
                        resource_type=ResourceType.REPORT,
                        condition=lambda user, ctx: True,
                        decision=AccessDecision.ALLOW,
                    )
                    acm.abac_evaluator.add_policy(policy)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=adder) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================
# 6. 边界条件与异常处理
# ============================================================


class TestEdgeCases:
    """边界条件测试."""

    def test_unknown_role_denied(self):
        """未知角色被拒绝 (虽然枚举不会出现, 但测试防御性)."""
        acm = AccessControlManager()
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        result = acm.check_access(
            user=user,
            permission=Permission.SYSTEM_CONFIG,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context={},
        )
        assert not result.allowed

    def test_empty_context(self):
        """空 context 不抛异常."""
        acm = AccessControlManager()
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        result = acm.check_access(
            user=user,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context={},
        )
        assert result.allowed

    def test_none_context(self):
        """None context 不抛异常."""
        acm = AccessControlManager()
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        result = acm.check_access(
            user=user,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
            context=None,
        )
        assert result.allowed

    def test_access_result_serialization(self):
        """AccessResult 可序列化为 dict."""
        result = AccessResult(
            allowed=True,
            decision=AccessDecision.ALLOW,
            reason="测试通过",
            user_id="u-001",
        )
        d = result.to_dict()
        assert d["allowed"] is True
        assert d["decision"] == "allow"

    def test_access_request_construction(self):
        """AccessRequest 正确构造."""
        req = AccessRequest(
            user_id="u-001",
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
        )
        assert req.user_id == "u-001"
        assert req.permission == Permission.KB_PUBLIC_READ

    def test_clear_access_log(self):
        """清空访问日志."""
        acm = AccessControlManager()
        user = User(
            student_id="CS20240001",
            institution_id="inst-001",
            role=UserRole.UNDERGRAD,
        )
        acm.check_access(
            user=user,
            permission=Permission.KB_PUBLIC_READ,
            resource_type=ResourceType.KNOWLEDGE_BASE,
        )
        assert len(acm.get_access_log()) >= 1
        acm.clear_access_log()
        assert len(acm.get_access_log()) == 0
