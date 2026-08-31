"""L1 用户域访问控制模块 — RBAC 权限矩阵 + ABAC 策略评估 + 混合访问控制管理.

设计依据:
- L1 设计文档第二章 2.2: 角色权限矩阵 (13 项功能 × 5 种角色)
- L1 设计文档第二章 2.3: ABAC 属性策略 + Cedar 策略语言示例
- L1 设计文档第二章 2.4: 角色生命周期 (与 auth.py 联动)
- L1 设计文档第一章 1.2: 最小权限原则

融合世界先进方案:
- AWS IAM: 策略优先级 + 显式拒绝优先 (Explicit Deny Override)
- Amazon Cedar: 声明式 ABAC 策略, permit/forbid + when/unless 条件
- Google Zanzibar: 关系型访问控制 (ReBAC) 的理念融合到 ABAC
- OPA (Open Policy Agent): 策略即数据, 运行时热更新
- Neo4j RBAC: 角色-权限分离, 子图级权限控制
- Khan Academy: 教育场景角色权限精细化 (课程范围/学生范围)

模块组成:
1. 枚举: ActionType / ResourceType / AccessDecision
2. 异常: AccessDeniedError (JSON-RPC -32206)
3. RBAC 矩阵: RBACMatrix (13 权限 × 5 角色 + 条件标记)
4. ABAC 策略: ABACPolicy + ABACEvaluator (Cedar 语义子集)
5. 访问控制管理器: AccessControlManager (RBAC + ABAC + 审计)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from dy3_polaris.l1.models import (
    ABACAttributes,
    AuditAction,
    AuditLogEntry,
    AuditResult,
    DataLevel,
    GradeLevel,
    LabAccessTier,
    MAX_DAILY_AGENT_CALLS,
    Permission,
    User,
    UserRole,
    UserStatus,
)
from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 1. 枚举定义
# ============================================================


class ActionType(str, Enum):
    """操作类型 (ABAC 评估的动作维度).

    对应 Cedar 策略中的 action 字段, 将 Permission 枚举映射到
    更细粒度的操作场景, 用于 ABAC 策略匹配.
    """

    # 知识库操作
    READ_KB = "read_kb"
    ACCESS_INTERNAL_DATA = "access_internal_data"
    WRITE_KB = "write_kb"

    # Agent 调用
    INVOKE_DIAGNOSIS_AGENT = "invoke_diagnosis_agent"
    INVOKE_KNOWLEDGE_AGENT = "invoke_knowledge_agent"
    INVOKE_REVIEW_AGENT = "invoke_review_agent"
    INVOKE_GUIDE_AGENT = "invoke_guide_agent"

    # 学情数据
    VIEW_OWN_REPORT = "view_own_report"
    VIEW_STUDENT_REPORT = "view_student_report"
    EXPORT_REPORT = "export_report"

    # 系统管理
    SYSTEM_CONFIG = "system_config"
    USER_MANAGE = "user_manage"

    # ABAC 专有动作 (无直接 Permission 映射)
    ACCESS_LAB_DATA = "access_lab_data"
    ACCESS_LAB_GUIDE = "access_lab_guide"
    ACCESS_ADVANCED_MODULE = "access_advanced_module"
    HITL_CONFIRM = "hitl_confirm"


class ResourceType(str, Enum):
    """资源类型 (ABAC 评估的资源维度).

    对应 Cedar 策略中的 resource 字段, 标识被访问的资源类别.
    """

    KNOWLEDGE_BASE = "knowledge_base"
    LAB_DATASET = "lab_dataset"
    LAB_GUIDE = "lab_guide"
    KNOWLEDGE_MODULE = "knowledge_module"
    AGENT = "agent"
    REPORT = "report"
    USER_RESOURCE = "user_resource"
    SYSTEM_CONFIG_RESOURCE = "system_config_resource"


class AccessDecision(str, Enum):
    """访问决策 (借鉴 AWS IAM 决策模型).

    三种决策:
        ALLOW: 允许访问
        DENY: 拒绝访问
        DENY_AND_LOG: 拒绝访问并记录审计日志 (用于敏感操作)
    """

    ALLOW = "allow"
    DENY = "deny"
    DENY_AND_LOG = "deny_and_log"


# ============================================================
# 2. 异常定义 (JSON-RPC -32206)
# ============================================================


class AccessDeniedError(L6Error):
    """访问被拒绝异常 (JSON-RPC -32206).

    当 AccessControlManager 判定访问请求被拒绝并调用 enforce() 时抛出.
    继承 L6Error 异常体系, 集成 JSON-RPC 错误码.

    Attributes:
        user_id: 被拒绝的用户 ID
        permission: 被拒绝的权限
        resource_type: 被拒绝的资源类型
        reason: 拒绝原因
    """

    def __init__(
        self,
        user_id: str,
        permission: Permission,
        resource_type: ResourceType,
        resource_id: str = "",
        reason: str = "访问被拒绝",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.user_id = user_id
        self.permission = permission
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.reason = reason
        ctx: dict[str, Any] = {
            "user_id": user_id,
            "permission": permission.value,
            "resource_type": resource_type.value,
            "resource_id": resource_id,
            "reason": reason,
        }
        ctx.update(context or {})
        super().__init__(
            "L1_ACCESS_DENIED",
            detail
            or (
                f"user={user_id}, permission={permission.value}, "
                f"resource_type={resource_type.value}, "
                f"resource_id={resource_id}, reason={reason}"
            ),
            ctx,
        )

    def _jsonrpc_code(self) -> int:
        return -32206


# ============================================================
# 3. RBAC 权限矩阵
# ============================================================


class RBACMatrix:
    """RBAC 权限矩阵 — 13 项功能权限 × 5 种角色.

    设计依据:
    - L1 设计文档第二章 2.2: 角色权限矩阵表
    - 借鉴 Neo4j RBAC: 角色-权限分离, 基础权限 + 条件标记
    - 借鉴 AWS IAM: 权限可动态增删 (运行时热更新)

    权限矩阵 (✓=允许, ✗=拒绝, (条件)=有条件允许):

    | 权限                    | UNDERGRAD | GRADUATE       | TEACHER        | ADMIN | ALUMNI |
    |------------------------|-----------|----------------|----------------|-------|--------|
    | KB_PUBLIC_READ         | ✓         | ✓              | ✓              | ✓     | ✓      |
    | KB_INTERNAL_DATA_ACCESS| ✗         | ✓(限导师范围)  | ✓              | ✓     | ✗      |
    | KB_WRITE_EDIT          | ✗         | ✗              | ✓(仅课程范围)  | ✓     | ✗      |
    | AGENT_DIAGNOSIS        | ✓         | ✓              | ✓              | ✓     | ✗      |
    | AGENT_KNOWLEDGE_GEN    | ✓(限数量) | ✓              | ✓              | ✓     | ✗      |
    | AGENT_REVIEW           | ✗         | ✗              | ✓              | ✓     | ✗      |
    | AGENT_GUIDE            | ✓         | ✓              | ✓              | ✓     | ✗      |
    | VIEW_OWN_REPORT        | ✓         | ✓              | N/A            | N/A   | ✓      |
    | VIEW_STUDENT_REPORT    | ✗         | ✗              | ✓(仅所带学生)  | ✓     | ✗      |
    | EXPORT_REPORT          | ✗         | ✓(匿名聚合)    | ✓              | ✓     | ✗      |
    | SYSTEM_CONFIG          | ✗         | ✗              | ✓(课程级)      | ✓     | ✗      |
    | USER_MANAGE            | ✗         | ✗              | ✗              | ✓     | ✗      |
    | HITL_CONFIRM           | ✓         | ✓              | ✓              | ✗     | ✗      |

    条件标记由 ABAC 策略引擎进一步评估, RBAC 仅做粗粒度控制.

    线程安全: threading.RLock 保护权限矩阵.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 权限矩阵: role → set[Permission]
        self._matrix: dict[UserRole, set[Permission]] = self._build_default_matrix()
        # 条件标记: (role, permission) → 条件描述
        self._conditions: dict[tuple[UserRole, Permission], str] = (
            self._build_default_conditions()
        )

    def _build_default_matrix(self) -> dict[UserRole, set[Permission]]:
        """构建默认权限矩阵 (设计文档 2.2)."""
        return {
            UserRole.UNDERGRAD: {
                Permission.KB_PUBLIC_READ,
                Permission.AGENT_DIAGNOSIS,
                Permission.AGENT_KNOWLEDGE_GEN,
                Permission.AGENT_GUIDE,
                Permission.VIEW_OWN_REPORT,
                Permission.HITL_CONFIRM,
            },
            UserRole.GRADUATE: {
                Permission.KB_PUBLIC_READ,
                Permission.KB_INTERNAL_DATA_ACCESS,
                Permission.AGENT_DIAGNOSIS,
                Permission.AGENT_KNOWLEDGE_GEN,
                Permission.AGENT_GUIDE,
                Permission.VIEW_OWN_REPORT,
                Permission.EXPORT_REPORT,
                Permission.HITL_CONFIRM,
            },
            UserRole.RESEARCHER: {
                Permission.KB_PUBLIC_READ,
                Permission.KB_INTERNAL_DATA_ACCESS,
                Permission.AGENT_DIAGNOSIS,
                Permission.AGENT_KNOWLEDGE_GEN,
                Permission.AGENT_GUIDE,
                Permission.VIEW_OWN_REPORT,
                Permission.EXPORT_REPORT,
                Permission.HITL_CONFIRM,
            },
            UserRole.TEACHER: {
                Permission.KB_PUBLIC_READ,
                Permission.KB_INTERNAL_DATA_ACCESS,
                Permission.KB_WRITE_EDIT,
                Permission.AGENT_REVIEW,
                Permission.AGENT_GUIDE,
                Permission.VIEW_STUDENT_REPORT,
                Permission.EXPORT_REPORT,
                Permission.HITL_CONFIRM,
            },
            UserRole.ADMIN: set(Permission),
            UserRole.ALUMNI: {
                Permission.KB_PUBLIC_READ,
                Permission.VIEW_OWN_REPORT,
            },
        }

    def _build_default_conditions(
        self,
    ) -> dict[tuple[UserRole, Permission], str]:
        """构建默认条件标记 (设计文档 2.2 表格中的括号注释)."""
        return {
            # 研究生: 内部数据访问限导师授权范围
            (UserRole.GRADUATE, Permission.KB_INTERNAL_DATA_ACCESS): "限导师授权范围",
            # 研究生: 导出限匿名聚合
            (UserRole.GRADUATE, Permission.EXPORT_REPORT): "匿名聚合导出",
            # 教师: 写入仅课程范围
            (UserRole.TEACHER, Permission.KB_WRITE_EDIT): "仅课程范围",
            # 教师: 学情诊断限所带学生
            (UserRole.TEACHER, Permission.AGENT_DIAGNOSIS): "限所带学生",
            # 教师: 查看学生报告限所带学生
            (UserRole.TEACHER, Permission.VIEW_STUDENT_REPORT): "仅所带学生",
            # 教师: 系统配置限课程级
            (UserRole.TEACHER, Permission.SYSTEM_CONFIG): "课程级配置",
            # 本科生: 知识生成 Agent 限数量
            (UserRole.UNDERGRAD, Permission.AGENT_KNOWLEDGE_GEN): "限每日数量",
            # 本科生: 学情诊断限自身
            (UserRole.UNDERGRAD, Permission.AGENT_DIAGNOSIS): "限自身",
            # 本科生: 导学被动接收
            (UserRole.UNDERGRAD, Permission.AGENT_GUIDE): "被动接收",
        }

    def check_permission(
        self, role: UserRole, permission: Permission
    ) -> bool:
        """检查角色是否拥有指定权限 (粗粒度, 不考虑条件标记).

        Args:
            role: 用户角色
            permission: 要检查的权限

        Returns:
            True 如果角色拥有该权限 (可能附带条件标记)
        """
        with self._lock:
            return permission in self._matrix.get(role, set())

    def get_permissions(self, role: UserRole) -> set[Permission]:
        """获取角色的所有权限.

        Args:
            role: 用户角色

        Returns:
            该角色的权限集合 (副本, 修改不影响内部状态)
        """
        with self._lock:
            return set(self._matrix.get(role, set()))

    def get_matrix(self) -> dict[UserRole, set[Permission]]:
        """获取完整角色权限矩阵.

        Returns:
            角色 → 权限集合的映射 (副本)
        """
        with self._lock:
            return {role: set(perms) for role, perms in self._matrix.items()}

    def has_conditional_marker(
        self, role: UserRole, permission: Permission
    ) -> bool:
        """检查权限是否附带条件标记.

        条件标记表示该权限需要 ABAC 策略进一步评估,
        如"限导师授权范围"、"仅课程范围"等.

        Args:
            role: 用户角色
            permission: 要检查的权限

        Returns:
            True 如果该权限附带条件标记
        """
        with self._lock:
            return (role, permission) in self._conditions

    def get_conditional_description(
        self, role: UserRole, permission: Permission
    ) -> str:
        """获取条件标记的描述文本.

        Args:
            role: 用户角色
            permission: 要检查的权限

        Returns:
            条件描述文本, 无条件时返回空字符串
        """
        with self._lock:
            return self._conditions.get((role, permission), "")

    def add_permission(
        self, role: UserRole, permission: Permission
    ) -> None:
        """动态添加权限到角色 (运行时热更新).

        Args:
            role: 用户角色
            permission: 要添加的权限
        """
        with self._lock:
            if role not in self._matrix:
                self._matrix[role] = set()
            self._matrix[role].add(permission)

    def remove_permission(
        self, role: UserRole, permission: Permission
    ) -> None:
        """动态移除角色权限 (运行时热更新).

        Args:
            role: 用户角色
            permission: 要移除的权限
        """
        with self._lock:
            if role in self._matrix:
                self._matrix[role].discard(permission)
            # 同时移除条件标记
            self._conditions.pop((role, permission), None)

    def add_condition(
        self,
        role: UserRole,
        permission: Permission,
        description: str,
    ) -> None:
        """添加条件标记到权限.

        Args:
            role: 用户角色
            permission: 要标记的权限
            description: 条件描述
        """
        with self._lock:
            self._conditions[(role, permission)] = description


# ============================================================
# 4. ABAC 策略与评估引擎
# ============================================================


@dataclass
class ABACPolicy:
    """ABAC 策略定义 (Cedar 语义子集).

    对应 Cedar 策略语言中的 permit/forbid 语句:
    ```
    permit(
        principal in [roles],
        action == Action::"action_type",
        resource == ResourceType::"resource_type"
    ) when { condition(user, context) };
    ```

    Attributes:
        policy_id: 策略唯一标识
        name: 策略名称
        description: 策略描述
        applicable_roles: 适用的角色列表
        action: 匹配的操作类型
        resource_type: 匹配的资源类型
        condition: 条件函数 (user, context) → bool
        decision: 决策 (ALLOW / DENY)
        priority: 优先级 (数值越大优先级越高, 默认 0)
    """

    policy_id: str
    name: str
    description: str
    applicable_roles: list[UserRole]
    action: ActionType
    resource_type: ResourceType
    condition: Callable[[User, dict[str, Any]], bool]
    decision: AccessDecision = AccessDecision.ALLOW
    priority: int = 0

    def matches(
        self, user: User, action: ActionType, resource_type: ResourceType
    ) -> bool:
        """检查策略是否匹配当前请求 (角色 + 动作 + 资源类型).

        Args:
            user: 用户对象
            action: 操作类型
            resource_type: 资源类型

        Returns:
            True 如果角色、动作和资源类型均匹配
        """
        return (
            user.role in self.applicable_roles
            and self.action == action
            and self.resource_type == resource_type
        )


class ABACEvaluator:
    """ABAC 策略评估引擎 — Cedar 语义子集实现.

    设计依据:
    - L1 设计文档第二章 2.3: ABAC 属性策略 + Cedar 策略示例
    - 借鉴 Amazon Cedar: permit/forbid + when/unless 条件
    - 借鉴 AWS IAM: 策略优先级 + 显式拒绝优先
    - 借鉴 OPA: 策略运行时热更新

    内置策略 (设计文档 Cedar 示例):
    1. 研究生实验数据访问: 限导师授权范围 + 实验权限等级
    2. 本科生 Agent 调用频率: 限 MAX_DAILY_AGENT_CALLS
    3. 课程进度门槛: progress < 0.3 阻止综合实验指导
    4. 年级门槛: 大二以下不可访问高级模块
    5. 校友只读: 非读操作一律拒绝

    评估规则:
    1. 遍历所有匹配策略 (角色 + 动作 + 资源类型)
    2. 按优先级降序排列
    3. 同优先级: DENY 覆盖 ALLOW (显式拒绝优先)
    4. 不同优先级: 高优先级覆盖低优先级
    5. 无匹配策略: 默认放行 (RBAC 已做粗粒度控制)

    线程安全: threading.RLock 保护策略列表.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._policies: list[ABACPolicy] = []
        self._init_builtin_policies()

    def _init_builtin_policies(self) -> None:
        """初始化内置 ABAC 策略 (设计文档 Cedar 示例)."""

        # 策略 1: 研究生实验数据访问 → 限导师授权范围 + 权限等级
        self._policies.append(
            ABACPolicy(
                policy_id="builtin-grad-lab-data",
                name="研究生实验数据访问控制",
                description="研究生仅可访问导师授权范围且权限等级足够的实验数据",
                applicable_roles=[UserRole.GRADUATE],
                action=ActionType.ACCESS_LAB_DATA,
                resource_type=ResourceType.LAB_DATASET,
                condition=self._check_grad_lab_data_access,
                decision=AccessDecision.ALLOW,
                priority=50,
            )
        )

        # 策略 2: 本科生 Agent 调用频率限制
        self._policies.append(
            ABACPolicy(
                policy_id="builtin-undergrad-agent-freq",
                name="本科生Agent调用频率限制",
                description="本科生知识生成Agent每日调用次数不得超过上限",
                applicable_roles=[UserRole.UNDERGRAD],
                action=ActionType.INVOKE_KNOWLEDGE_AGENT,
                resource_type=ResourceType.AGENT,
                condition=self._check_undergrad_agent_frequency,
                decision=AccessDecision.ALLOW,
                priority=50,
            )
        )

        # 策略 3: 课程进度门槛 — 综合实验指导
        self._policies.append(
            ABACPolicy(
                policy_id="builtin-progress-lab-guide",
                name="课程进度实验指导门槛",
                description="课程进度 < 0.3 时不推荐综合实验指导",
                applicable_roles=[UserRole.UNDERGRAD, UserRole.GRADUATE],
                action=ActionType.ACCESS_LAB_GUIDE,
                resource_type=ResourceType.LAB_GUIDE,
                condition=self._check_course_progress_for_guide,
                decision=AccessDecision.ALLOW,
                priority=50,
            )
        )

        # 策略 4: 年级门槛 — 高级模块
        self._policies.append(
            ABACPolicy(
                policy_id="builtin-grade-advanced-module",
                name="年级高级模块门槛",
                description="大二以下学生不可访问高级模块",
                applicable_roles=[UserRole.UNDERGRAD],
                action=ActionType.ACCESS_ADVANCED_MODULE,
                resource_type=ResourceType.KNOWLEDGE_MODULE,
                condition=self._check_grade_for_advanced,
                decision=AccessDecision.ALLOW,
                priority=50,
            )
        )

        # 策略 5: 校友只读强制
        self._policies.append(
            ABACPolicy(
                policy_id="builtin-alumni-readonly",
                name="校友只读强制",
                description="校友不可执行任何写操作",
                applicable_roles=[UserRole.ALUMNI],
                action=ActionType.WRITE_KB,
                resource_type=ResourceType.KNOWLEDGE_BASE,
                condition=lambda user, ctx: True,  # 永远触发 DENY
                decision=AccessDecision.DENY,
                priority=100,  # 最高优先级, 不可被覆盖
            )
        )

    # --- 内置条件函数 ---

    @staticmethod
    def _check_grad_lab_data_access(
        user: User, context: dict[str, Any]
    ) -> bool:
        """研究生实验数据访问条件: 导师范围 + 权限等级.

        Cedar 等效:
        permit(...) when {
            resource.supervisor_id == principal.supervisor_id
            and principal.lab_access_tier >= resource.required_tier
        }
        """
        resource_supervisor = context.get("supervisor_id", "")
        required_tier_str = context.get("required_tier", LabAccessTier.TIER0.value)

        # 导师范围检查
        if user.abac_attributes.supervisor_id != resource_supervisor:
            return False

        # 权限等级检查
        try:
            required_tier = LabAccessTier(required_tier_str)
        except ValueError:
            required_tier = LabAccessTier.TIER0

        tier_order = {
            LabAccessTier.TIER0: 0,
            LabAccessTier.TIER1: 1,
            LabAccessTier.TIER2: 2,
            LabAccessTier.TIER3: 3,
        }
        user_tier = tier_order.get(user.abac_attributes.lab_access_tier, 0)
        needed_tier = tier_order.get(required_tier, 0)
        return user_tier >= needed_tier

    @staticmethod
    def _check_undergrad_agent_frequency(
        user: User, context: dict[str, Any]
    ) -> bool:
        """本科生 Agent 调用频率条件.

        Cedar 等效:
        permit(...) when {
            principal.daily_agent_calls < MAX_DAILY_CALLS
        }
        """
        return user.abac_attributes.daily_agent_calls < MAX_DAILY_AGENT_CALLS

    @staticmethod
    def _check_course_progress_for_guide(
        user: User, context: dict[str, Any]
    ) -> bool:
        """课程进度实验指导门槛.

        课程进度 < 0.3 时不推荐综合实验指导.
        """
        guide_type = context.get("guide_type", "")
        if guide_type == "comprehensive":
            return user.abac_attributes.course_progress >= 0.3
        return True

    @staticmethod
    def _check_grade_for_advanced(
        user: User, context: dict[str, Any]
    ) -> bool:
        """年级高级模块门槛.

        大二以下 (FRESHMAN, SOPHOMORE) 不可访问高级模块.
        """
        module_difficulty = context.get("module_difficulty", "")
        if module_difficulty == "advanced":
            blocked_grades = {GradeLevel.FRESHMAN, GradeLevel.SOPHOMORE}
            return user.abac_attributes.grade_level not in blocked_grades
        return True

    # --- 策略管理 ---

    def add_policy(self, policy: ABACPolicy) -> None:
        """添加 ABAC 策略 (运行时热更新).

        Args:
            policy: 要添加的策略
        """
        with self._lock:
            # 避免重复 ID
            self._policies = [
                p for p in self._policies if p.policy_id != policy.policy_id
            ]
            self._policies.append(policy)
            # 按优先级降序排列
            self._policies.sort(key=lambda p: p.priority, reverse=True)

    def remove_policy(self, policy_id: str) -> None:
        """移除 ABAC 策略.

        Args:
            policy_id: 策略 ID
        """
        with self._lock:
            self._policies = [
                p for p in self._policies if p.policy_id != policy_id
            ]

    def list_policies(self) -> list[ABACPolicy]:
        """列出所有策略 (按优先级降序)."""
        with self._lock:
            return list(self._policies)

    def get_policy(self, policy_id: str) -> ABACPolicy | None:
        """获取指定策略."""
        with self._lock:
            for p in self._policies:
                if p.policy_id == policy_id:
                    return p
            return None

    # --- 策略评估 ---

    def evaluate(
        self,
        user: User,
        action: ActionType,
        resource_type: ResourceType,
        context: dict[str, Any] | None = None,
    ) -> AccessResult:
        """评估 ABAC 策略.

        评估流程:
        1. 查找匹配策略 (角色 + 动作 + 资源类型)
        2. 评估匹配策略的条件函数
        3. 按优先级 + 显式拒绝优先规则决策
        4. 无匹配策略 → 默认放行

        Args:
            user: 用户对象
            action: 操作类型
            resource_type: 资源类型
            context: ABAC 上下文 (资源属性等)

        Returns:
            AccessResult 评估结果
        """
        ctx = context or {}
        now = time.time()

        with self._lock:
            matching: list[ABACPolicy] = []
            for policy in self._policies:
                if policy.matches(user, action, resource_type):
                    matching.append(policy)

        if not matching:
            # 无匹配策略 → 默认放行 (RBAC 已做粗粒度控制)
            return AccessResult(
                allowed=True,
                decision=AccessDecision.ALLOW,
                reason="无匹配 ABAC 策略, 默认放行",
                user_id=user.user_id,
                timestamp=now,
            )

        # 按优先级降序排列
        matching.sort(key=lambda p: p.priority, reverse=True)

        # 分组评估: 同优先级一组
        highest_priority = matching[0].priority
        highest_group = [
            p for p in matching if p.priority == highest_priority
        ]

        # 最高优先级层: 检查是否有显式拒绝 (条件满足且 decision=DENY)
        for policy in highest_group:
            if policy.decision == AccessDecision.DENY:
                if policy.condition(user, ctx):
                    return AccessResult(
                        allowed=False,
                        decision=AccessDecision.DENY,
                        reason=f"ABAC 策略拒绝: {policy.name} ({policy.policy_id})",
                        policy_id=policy.policy_id,
                        user_id=user.user_id,
                        timestamp=now,
                    )

        # 最高优先级层: 检查是否有允许 (条件满足且 decision=ALLOW)
        for policy in highest_group:
            if policy.decision == AccessDecision.ALLOW:
                if policy.condition(user, ctx):
                    return AccessResult(
                        allowed=True,
                        decision=AccessDecision.ALLOW,
                        reason=f"ABAC 策略允许: {policy.name} ({policy.policy_id})",
                        policy_id=policy.policy_id,
                        user_id=user.user_id,
                        timestamp=now,
                    )

        # 最高优先级层条件均不满足 → 检查下一层
        # 对于 ALLOW 策略, 条件不满足 → 拒绝
        # 对于 DENY 策略, 条件不满足 → 继续检查
        for policy in highest_group:
            if policy.decision == AccessDecision.ALLOW:
                # ALLOW 条件不满足 → 该策略不允许此操作
                return AccessResult(
                    allowed=False,
                    decision=AccessDecision.DENY,
                    reason=f"ABAC 条件不满足: {policy.name} ({policy.policy_id})",
                    policy_id=policy.policy_id,
                    user_id=user.user_id,
                    timestamp=now,
                )

        # 所有策略条件均不满足 → 默认放行
        return AccessResult(
            allowed=True,
            decision=AccessDecision.ALLOW,
            reason="ABAC 策略条件均不满足, 默认放行",
            user_id=user.user_id,
            timestamp=now,
        )


# ============================================================
# 5. 访问请求与结果
# ============================================================


@dataclass
class AccessRequest:
    """访问请求 (借鉴 AWS IAM AuthorizationRequest).

    Attributes:
        user_id: 请求用户 ID
        permission: 请求的功能权限 (RBAC 维度)
        resource_type: 资源类型 (ABAC 维度)
        resource_id: 资源 ID (可选)
        context: ABAC 上下文 (资源属性等)
        action: 操作类型 (ABAC 维度, 可选, 默认从 permission 推断)
    """

    user_id: str
    permission: Permission
    resource_type: ResourceType
    resource_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    action: ActionType | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "permission": self.permission.value,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "context": self.context,
            "action": self.action.value if self.action else None,
        }


@dataclass
class AccessResult:
    """访问评估结果 (借鉴 AWS IAM EvaluationResult).

    Attributes:
        allowed: 是否允许访问
        decision: 决策类型
        reason: 决策原因 (人类可读)
        policy_id: 命中的策略 ID (无匹配时为空)
        user_id: 请求用户 ID
        timestamp: 评估时间戳
    """

    allowed: bool
    decision: AccessDecision
    reason: str
    user_id: str
    policy_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "decision": self.decision.value,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
        }


# ============================================================
# 6. 访问控制管理器 (RBAC + ABAC 混合)
# ============================================================


# Permission → ActionType 默认映射
_PERMISSION_ACTION_MAP: dict[Permission, ActionType] = {
    Permission.KB_PUBLIC_READ: ActionType.READ_KB,
    Permission.KB_INTERNAL_DATA_ACCESS: ActionType.ACCESS_INTERNAL_DATA,
    Permission.KB_WRITE_EDIT: ActionType.WRITE_KB,
    Permission.AGENT_DIAGNOSIS: ActionType.INVOKE_DIAGNOSIS_AGENT,
    Permission.AGENT_KNOWLEDGE_GEN: ActionType.INVOKE_KNOWLEDGE_AGENT,
    Permission.AGENT_REVIEW: ActionType.INVOKE_REVIEW_AGENT,
    Permission.AGENT_GUIDE: ActionType.INVOKE_GUIDE_AGENT,
    Permission.VIEW_OWN_REPORT: ActionType.VIEW_OWN_REPORT,
    Permission.VIEW_STUDENT_REPORT: ActionType.VIEW_STUDENT_REPORT,
    Permission.EXPORT_REPORT: ActionType.EXPORT_REPORT,
    Permission.SYSTEM_CONFIG: ActionType.SYSTEM_CONFIG,
    Permission.USER_MANAGE: ActionType.USER_MANAGE,
    Permission.HITL_CONFIRM: ActionType.HITL_CONFIRM,
}


class AccessControlManager:
    """RBAC + ABAC 混合访问控制管理器.

    设计依据:
    - L1 设计文档第二章: RBAC+ABAC 混合模型
    - 借鉴 AWS IAM: 策略评估引擎 + 显式拒绝优先
    - 借鉴 Neo4j RBAC: 用户-角色-权限三级授权
    - 借鉴 OPA: 策略热更新

    评估流程:
    1. 状态检查: 用户必须为 ACTIVE 或 ALUMNI
    2. RBAC 粗粒度: 检查角色是否拥有该权限
    3. ABAC 细粒度: 评估 ABAC 策略 (条件标记 → 策略匹配)
    4. 审计日志: 记录所有访问决策
    5. 统计信息: 维护允许/拒绝计数

    线程安全: threading.RLock 保护共享状态.
    """

    def __init__(
        self,
        rbac_matrix: RBACMatrix | None = None,
        abac_evaluator: ABACEvaluator | None = None,
    ) -> None:
        self.rbac_matrix = rbac_matrix or RBACMatrix()
        self.abac_evaluator = abac_evaluator or ABACEvaluator()
        self._access_log: list[AccessResult] = []
        self._audit_logs: list[AuditLogEntry] = []
        self._lock = threading.RLock()

    def check_access(
        self,
        user: User,
        permission: Permission,
        resource_type: ResourceType,
        context: dict[str, Any] | None = None,
        action: ActionType | None = None,
    ) -> AccessResult:
        """评估访问请求 (RBAC + ABAC 混合).

        评估流程:
        1. 用户状态检查 (SUSPENDED → 拒绝)
        2. RBAC 权限检查 (角色无权限 → 拒绝)
        3. ABAC 策略评估 (条件不满足 → 拒绝)
        4. 记录审计日志

        Args:
            user: 用户对象
            permission: 请求的功能权限
            resource_type: 资源类型
            context: ABAC 上下文 (可选)
            action: 操作类型 (可选, 默认从 permission 推断)

        Returns:
            AccessResult 评估结果
        """
        ctx = context if context is not None else {}
        now = time.time()
        user_id = user.user_id

        # 1. 用户状态检查
        if user.status == UserStatus.SUSPENDED:
            result = AccessResult(
                allowed=False,
                decision=AccessDecision.DENY,
                reason=f"用户已停用 (SUSPENDED), 拒绝所有操作",
                user_id=user_id,
                timestamp=now,
            )
            self._record_result(result, user, permission, resource_type)
            return result

        # 2. RBAC 权限检查
        if not self.rbac_matrix.check_permission(user.role, permission):
            result = AccessResult(
                allowed=False,
                decision=AccessDecision.DENY,
                reason=f"RBAC 权限不足: 角色 {user.role.value} 无 {permission.value} 权限",
                user_id=user_id,
                timestamp=now,
            )
            self._record_result(result, user, permission, resource_type)
            return result

        # 3. ABAC 策略评估
        # 确定操作类型: 优先使用显式传入的 action, 否则从 permission 推断
        eval_action = action or _PERMISSION_ACTION_MAP.get(permission)
        if eval_action is None:
            # 无对应的 ActionType, RBAC 已通过 → 放行
            result = AccessResult(
                allowed=True,
                decision=AccessDecision.ALLOW,
                reason="RBAC 权限通过, 无 ABAC 策略匹配",
                user_id=user_id,
                timestamp=now,
            )
            self._record_result(result, user, permission, resource_type)
            return result

        abac_result = self.abac_evaluator.evaluate(
            user=user,
            action=eval_action,
            resource_type=resource_type,
            context=ctx,
        )

        # 合并结果
        if not abac_result.allowed:
            self._record_result(
                abac_result, user, permission, resource_type
            )
            return abac_result

        # RBAC + ABAC 均通过
        result = AccessResult(
            allowed=True,
            decision=AccessDecision.ALLOW,
            reason=f"RBAC + ABAC 均通过: {abac_result.reason}",
            policy_id=abac_result.policy_id,
            user_id=user_id,
            timestamp=now,
        )
        self._record_result(result, user, permission, resource_type)
        return result

    def enforce(
        self,
        user: User,
        permission: Permission,
        resource_type: ResourceType,
        context: dict[str, Any] | None = None,
        action: ActionType | None = None,
        resource_id: str = "",
    ) -> bool:
        """强制执行访问控制.

        评估访问请求, 如果被拒绝则抛出 AccessDeniedError.
        用于在业务操作前进行访问控制检查.

        Args:
            user: 用户对象
            permission: 请求的权限
            resource_type: 资源类型
            context: ABAC 上下文 (可选)
            action: 操作类型 (可选)
            resource_id: 资源 ID (可选)

        Returns:
            True (如果允许访问)

        Raises:
            AccessDeniedError: 访问被拒绝时抛出
        """
        result = self.check_access(
            user=user,
            permission=permission,
            resource_type=resource_type,
            context=context,
            action=action,
        )
        if not result.allowed:
            raise AccessDeniedError(
                user_id=user.user_id,
                permission=permission,
                resource_type=resource_type,
                resource_id=resource_id,
                reason=result.reason,
            )
        return True

    # --- 日志与统计 ---

    def get_access_log(
        self,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[AccessResult]:
        """查询访问日志.

        Args:
            user_id: 用户 ID 过滤 (可选)
            limit: 返回记录数上限

        Returns:
            访问结果列表 (按时间正序, 取最后 limit 条)
        """
        with self._lock:
            if user_id is not None:
                logs = [r for r in self._access_log if r.user_id == user_id]
            else:
                logs = list(self._access_log)
            if limit > 0:
                return logs[-limit:]
            return logs

    def get_audit_logs(self, limit: int = 100) -> list[AuditLogEntry]:
        """获取审计日志.

        Args:
            limit: 返回记录数上限

        Returns:
            审计日志列表
        """
        with self._lock:
            return list(self._audit_logs[-limit:])

    def clear_access_log(self) -> None:
        """清空访问日志."""
        with self._lock:
            self._access_log.clear()
            self._audit_logs.clear()

    def get_stats(self) -> dict[str, Any]:
        """获取访问控制统计信息.

        Returns:
            统计信息字典:
            - total_access_requests: 总访问请求数
            - allowed_requests: 允许的请求数
            - denied_requests: 拒绝的请求数
            - allow_rate: 允许率 (0.0 ~ 1.0)
        """
        with self._lock:
            total = len(self._access_log)
            allowed = sum(1 for r in self._access_log if r.allowed)
            denied = total - allowed
            return {
                "total_access_requests": total,
                "allowed_requests": allowed,
                "denied_requests": denied,
                "allow_rate": allowed / total if total > 0 else 0.0,
            }

    # --- 内部方法 ---

    def _record_result(
        self,
        result: AccessResult,
        user: User,
        permission: Permission,
        resource_type: ResourceType,
    ) -> None:
        """记录访问结果到日志和审计.

        Args:
            result: 访问结果
            user: 用户对象
            permission: 请求的权限
            resource_type: 资源类型
        """
        with self._lock:
            # 访问日志
            self._access_log.append(result)

            # 审计日志 (仅在拒绝或 DENY_AND_LOG 时记录)
            if not result.allowed or result.decision == AccessDecision.DENY_AND_LOG:
                audit_entry = AuditLogEntry(
                    actor_id=user.user_id,
                    actor_role=user.role,
                    action=AuditAction.AGENT_INVOKE,
                    target_resource=f"{resource_type.value}:{permission.value}",
                    target_data_level=DataLevel.L2_INTERNAL,
                    purpose=result.reason,
                    result=AuditResult.DENIED if not result.allowed else AuditResult.SUCCESS,
                )
                self._audit_logs.append(audit_entry)
            elif result.allowed:
                # 允许的操作也记录审计 (可选, 根据策略)
                audit_entry = AuditLogEntry(
                    actor_id=user.user_id,
                    actor_role=user.role,
                    action=AuditAction.AGENT_INVOKE,
                    target_resource=f"{resource_type.value}:{permission.value}",
                    target_data_level=DataLevel.L2_INTERNAL,
                    purpose=result.reason,
                    result=AuditResult.SUCCESS,
                )
                self._audit_logs.append(audit_entry)


__all__ = [
    # 枚举
    "ActionType",
    "ResourceType",
    "AccessDecision",
    # 异常
    "AccessDeniedError",
    # RBAC
    "RBACMatrix",
    # ABAC
    "ABACPolicy",
    "ABACEvaluator",
    # 请求与结果
    "AccessRequest",
    "AccessResult",
    # 管理器
    "AccessControlManager",
]
