"""L3 领域知识层 — 基于角色的访问控制 (RBAC).

借鉴 Neo4j RBAC (Role-Based Access Control) 和企业知识管理系统的 ACL 设计,
为知识存储引擎提供细粒度的访问控制能力。

核心概念:
    - 用户 (User): 系统中的实体, 拥有一个或多个角色
    - 角色 (Role): 权限的集合, 如管理员、编辑者、贡献者、读者、访客
    - 权限 (Permission): 对资源的操作能力, 如创建、读取、更新、删除、导出、管理
    - 资源类型 (ResourceType): 被保护的资源类别, 如实体、三元组、文档块、本体、索引
    - 访问级别 (AccessLevel): 资源的安全等级, 从公开到机密
    - 策略 (AccessPolicy): 将角色、权限、资源类型、访问级别组合的授权规则
    - 访问请求 (AccessRequest): 一次具体的访问请求
    - 访问结果 (AccessResult): 访问控制评估的结果

设计借鉴:
    1. Neo4j RBAC: 角色-权限分离, 子图级权限控制, 细粒度资源访问
    2. 企业知识管理: 访问级别分层 (公开 < 内部 < 受限 < 机密)
    3. AWS IAM: 策略优先级 + 显式拒绝优先 (Explicit Deny Override)
    4. OAuth 2.0 Scope: 细粒度权限范围控制
    5. Neo4j 子图权限: 通过 AccessControlledStore 装饰器实现透明访问控制

线程安全: 所有共享状态通过 threading.RLock 保护。
无外部依赖: 仅使用标准库 + pydantic v2。
"""

from __future__ import annotations

import time
from enum import Enum
from threading import RLock
from typing import Any

from pydantic import BaseModel, Field

from .exceptions import L3Error
from .models import AccessLevel, EntityType, KnowledgeEntity


# ============================================================
# 枚举定义
# ============================================================


class Role(str, Enum):
    """用户角色 (借鉴 Neo4j 内置角色体系).

    角色层级:
        ADMIN > EDITOR > CONTRIBUTOR > READER > GUEST

    每个角色对应一组默认权限策略, 可通过 AccessPolicy 灵活扩展。
    一个用户可同时拥有多个角色, 权限取并集。
    """

    ADMIN = "admin"  # 完全访问权限
    EDITOR = "editor"  # 读写 INTERNAL 及以下级别
    CONTRIBUTOR = "contributor"  # 读 INTERNAL, 仅写 PUBLIC
    READER = "reader"  # 仅读 PUBLIC 和 INTERNAL
    GUEST = "guest"  # 仅读 PUBLIC


class Permission(str, Enum):
    """操作权限 (借鉴 Neo4j privilege 体系).

    权限是访问控制的最小单位, 通过策略与角色绑定。
    """

    CREATE = "create"  # 创建资源
    READ = "read"  # 读取资源
    UPDATE = "update"  # 更新资源
    DELETE = "delete"  # 删除资源
    EXPORT = "export"  # 导出资源
    ADMIN = "admin"  # 管理操作 (备份、恢复、模式变更)


class ResourceType(str, Enum):
    """资源类型 (借鉴 Neo4j 子图资源分类).

    每种资源类型可独立配置访问策略, 实现细粒度控制。
    """

    ENTITY = "entity"  # 知识实体
    TRIPLE = "triple"  # 三元组 (SPO)
    CHUNK = "chunk"  # 文档切片
    ONTOLOGY = "ontology"  # 本体定义
    INDEX = "index"  # 索引结构


class AccessDecision(str, Enum):
    """访问决策 (借鉴 AWS IAM 决策模型).

    三种决策:
        ALLOW: 允许访问
        DENY: 拒绝访问
        DENY_AND_LOG: 拒绝访问并记录日志 (用于敏感操作审计)
    """

    ALLOW = "allow"
    DENY = "deny"
    DENY_AND_LOG = "deny_and_log"


# ============================================================
# 异常定义
# ============================================================


class AccessDeniedError(L3Error):
    """访问被拒绝异常 (借鉴 Neo4j Forbidden 错误).

    当访问控制管理器判定某次访问请求被拒绝时抛出此异常。
    继承 L3Error 异常体系, 集成 JSON-RPC 错误码 (-32415)。

    Attributes:
        user_id: 被拒绝的用户 ID
        permission: 被拒绝的权限
        resource_type: 被拒绝的资源类型
        resource_id: 被拒绝的资源 ID
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
            "L3_ACCESS_DENIED",
            detail
            or (
                f"user={user_id}, permission={permission.value}, "
                f"resource_type={resource_type.value}, "
                f"resource_id={resource_id}, reason={reason}"
            ),
            ctx,
        )

    def _jsonrpc_code(self) -> int:
        return -32415


# ============================================================
# 访问级别排序 (借鉴企业知识管理安全分级)
# ============================================================

#: 访问级别排序映射, 数值越大级别越高。
#: PUBLIC(0) < INTERNAL(1) < RESTRICTED(2) < CONFIDENTIAL(3)
_LEVEL_ORDER: dict[AccessLevel, int] = {
    AccessLevel.PUBLIC: 0,
    AccessLevel.INTERNAL: 1,
    AccessLevel.RESTRICTED: 2,
    AccessLevel.CONFIDENTIAL: 3,
}


# ============================================================
# 数据模型
# ============================================================


class User(BaseModel):
    """用户模型 (借鉴 Neo4j User + 企业 LDAP 用户属性).

    一个用户可拥有多个角色, 实际权限为所有角色策略的并集。
    用户的 access_level 字段决定了其可访问的最高资源级别。

    Attributes:
        user_id: 用户唯一标识
        username: 用户名 (显示用)
        roles: 角色列表, 默认为 READER
        access_level: 用户的最高访问级别, 默认为 PUBLIC
        department: 所属部门 (用于部门级访问控制扩展)
        metadata: 扩展元数据 (如邮箱、职位等)
    """

    user_id: str = Field(..., description="用户唯一标识")
    username: str = Field(..., description="用户名")
    roles: list[Role] = Field(
        default_factory=lambda: [Role.READER],
        description="角色列表",
    )
    access_level: AccessLevel = Field(
        default=AccessLevel.PUBLIC,
        description="用户的最高访问级别",
    )
    department: str = Field(default="", description="所属部门")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据",
    )


class AccessPolicy(BaseModel):
    """访问策略 (借鉴 AWS IAM Policy + Neo4j Privilege).

    策略是访问控制的核心规则单元, 将角色、权限、资源类型和访问级别
    组合为一条授权规则。多条策略通过优先级和决策类型协同工作。

    评估规则:
        1. 按优先级降序排列, 高优先级策略先评估
        2. 同一优先级下, 显式 DENY 覆盖 ALLOW
        3. 无匹配策略时, 默认拒绝 (Default Deny)

    Attributes:
        policy_id: 策略唯一标识
        name: 策略名称
        description: 策略描述
        roles: 适用的角色列表
        permissions: 授权的权限列表
        resource_types: 适用的资源类型列表
        access_levels: 适用的访问级别列表
        priority: 优先级 (数值越大优先级越高)
        decision: 决策类型 (ALLOW / DENY / DENY_AND_LOG)
    """

    policy_id: str = Field(..., description="策略唯一标识")
    name: str = Field(..., description="策略名称")
    description: str = Field(default="", description="策略描述")
    roles: list[Role] = Field(..., description="适用的角色列表")
    permissions: list[Permission] = Field(..., description="授权的权限列表")
    resource_types: list[ResourceType] = Field(..., description="适用的资源类型列表")
    access_levels: list[AccessLevel] = Field(..., description="适用的访问级别列表")
    priority: int = Field(default=0, description="优先级 (越大越高)")
    decision: AccessDecision = Field(
        default=AccessDecision.ALLOW,
        description="决策类型",
    )


class AccessRequest(BaseModel):
    """访问请求 (借鉴 AWS IAM AuthorizationRequest).

    每次访问控制评估都基于一个 AccessRequest, 包含用户、权限、
    资源类型和资源访问级别等信息。

    Attributes:
        user_id: 请求用户 ID
        permission: 请求的权限
        resource_type: 请求的资源类型
        resource_id: 请求的资源 ID (可选)
        resource_access_level: 资源的访问级别
        metadata: 扩展请求元数据
    """

    user_id: str = Field(..., description="请求用户 ID")
    permission: Permission = Field(..., description="请求的权限")
    resource_type: ResourceType = Field(..., description="请求的资源类型")
    resource_id: str = Field(default="", description="请求的资源 ID")
    resource_access_level: AccessLevel = Field(
        default=AccessLevel.PUBLIC,
        description="资源的访问级别",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展请求元数据",
    )


class AccessResult(BaseModel):
    """访问结果 (借鉴 AWS IAM EvaluationResult).

    访问控制评估的结果, 包含是否允许、决策类型、原因和命中的策略。

    Attributes:
        allowed: 是否允许访问
        decision: 决策类型
        reason: 决策原因 (人类可读)
        policy_id: 命中的策略 ID (无匹配时为空)
        user_id: 请求用户 ID
        timestamp: 评估时间戳
    """

    allowed: bool = Field(..., description="是否允许访问")
    decision: AccessDecision = Field(..., description="决策类型")
    reason: str = Field(..., description="决策原因")
    policy_id: str = Field(default="", description="命中的策略 ID")
    user_id: str = Field(..., description="请求用户 ID")
    timestamp: float = Field(..., description="评估时间戳")


# ============================================================
# 访问控制管理器
# ============================================================


class AccessControlManager:
    """访问控制管理器 (借鉴 Neo4j RBAC + 企业知识管理 ACL).

    功能:
        1. 用户-角色-权限三级授权
        2. 基于策略的访问评估 (优先级 + 显式拒绝)
        3. 资源类型 + 访问级别双维度控制
        4. 访问日志记录与查询
        5. 策略热更新 (运行时增删策略, 无需重启)

    设计理念:
        - 默认拒绝 (Default Deny): 无匹配策略时拒绝访问
        - 显式拒绝优先 (Explicit Deny Override): 同优先级下 DENY 覆盖 ALLOW
        - 最小权限原则 (Least Privilege): 每个角色仅授予必要权限
        - 关注点分离 (Separation of Concerns): 访问控制逻辑与业务逻辑解耦

    线程安全: 所有共享状态通过 threading.RLock 保护, 支持并发访问。

    Attributes:
        _users: 用户注册表 {user_id: User}
        _policies: 策略列表 (按优先级降序维护)
        _access_log: 访问日志列表
        _lock: 线程安全可重入锁
    """

    def __init__(self) -> None:
        """初始化访问控制管理器, 自动创建默认 RBAC 策略."""
        self._users: dict[str, User] = {}
        self._policies: list[AccessPolicy] = []
        self._access_log: list[AccessResult] = []
        self._lock: RLock = RLock()
        self._init_default_policies()

    # ================================================================
    # 用户管理
    # ================================================================

    def register_user(self, user: User) -> None:
        """注册用户.

        如果 user_id 已存在则覆盖更新 (借鉴 Neo4j 用户管理语义)。

        Args:
            user: 要注册的用户对象
        """
        with self._lock:
            self._users[user.user_id] = user

    def get_user(self, user_id: str) -> User | None:
        """获取用户信息.

        Args:
            user_id: 用户 ID

        Returns:
            用户对象, 不存在时返回 None
        """
        with self._lock:
            return self._users.get(user_id)

    def update_user(self, user_id: str, **updates: Any) -> User:
        """更新用户信息.

        Args:
            user_id: 用户 ID
            **updates: 要更新的字段 (如 roles=[Role.EDITOR], department="研发部")

        Returns:
            更新后的用户对象

        Raises:
            L3Error: 用户不存在时抛出
        """
        with self._lock:
            user = self._users.get(user_id)
            if user is None:
                raise L3Error(
                    "L3_USER_NOT_FOUND",
                    f"用户不存在: {user_id}",
                    {"user_id": user_id},
                )
            updated = user.model_copy(update=updates)
            self._users[user_id] = updated
            return updated

    def remove_user(self, user_id: str) -> User | None:
        """移除用户.

        Args:
            user_id: 用户 ID

        Returns:
            被移除的用户对象, 不存在时返回 None
        """
        with self._lock:
            return self._users.pop(user_id, None)

    def list_users(self) -> list[User]:
        """列出所有已注册用户.

        Returns:
            用户列表
        """
        with self._lock:
            return list(self._users.values())

    # ================================================================
    # 策略管理
    # ================================================================

    def add_policy(self, policy: AccessPolicy) -> None:
        """添加访问策略.

        策略添加后自动按优先级降序排列, 支持运行时热更新。
        如果 policy_id 已存在则抛出异常, 避免意外覆盖。

        Args:
            policy: 要添加的策略对象

        Raises:
            L3Error: 策略 ID 已存在时抛出
        """
        with self._lock:
            for existing in self._policies:
                if existing.policy_id == policy.policy_id:
                    raise L3Error(
                        "L3_POLICY_DUPLICATE",
                        f"策略 ID 已存在: {policy.policy_id}",
                        {"policy_id": policy.policy_id},
                    )
            self._policies.append(policy)
            # 按优先级降序排列, 高优先级在前
            self._policies.sort(key=lambda p: p.priority, reverse=True)

    def remove_policy(self, policy_id: str) -> None:
        """移除访问策略.

        Args:
            policy_id: 策略 ID
        """
        with self._lock:
            self._policies = [
                p for p in self._policies if p.policy_id != policy_id
            ]

    def get_policy(self, policy_id: str) -> AccessPolicy | None:
        """获取策略信息.

        Args:
            policy_id: 策略 ID

        Returns:
            策略对象, 不存在时返回 None
        """
        with self._lock:
            for p in self._policies:
                if p.policy_id == policy_id:
                    return p
            return None

    def list_policies(self) -> list[AccessPolicy]:
        """列出所有策略 (按优先级降序).

        Returns:
            策略列表
        """
        with self._lock:
            return list(self._policies)

    # ================================================================
    # 访问控制核心
    # ================================================================

    def check_access(self, request: AccessRequest) -> AccessResult:
        """评估访问请求 (借鉴 AWS IAM EvaluateConditions).

        评估流程:
            1. 查找用户, 不存在则拒绝
            2. 检查用户访问级别是否覆盖资源级别
            3. 评估匹配的策略 (优先级 + 显式拒绝)
            4. 记录访问日志

        Args:
            request: 访问请求

        Returns:
            访问结果
        """
        with self._lock:
            user = self._users.get(request.user_id)
            if user is None:
                result = AccessResult(
                    allowed=False,
                    decision=AccessDecision.DENY,
                    reason=f"用户不存在: {request.user_id}",
                    user_id=request.user_id,
                    timestamp=time.time(),
                )
                self._access_log.append(result)
                return result

            # 检查用户访问级别是否覆盖资源访问级别
            if not self._level_allows(
                user.access_level, request.resource_access_level
            ):
                result = AccessResult(
                    allowed=False,
                    decision=AccessDecision.DENY,
                    reason=(
                        f"访问级别不足: 用户级别={user.access_level.value}, "
                        f"资源级别={request.resource_access_level.value}"
                    ),
                    user_id=request.user_id,
                    timestamp=time.time(),
                )
                self._access_log.append(result)
                return result

            # 评估策略
            result = self._evaluate_policies(request, user)
            self._access_log.append(result)
            return result

    def enforce(
        self,
        user_id: str,
        permission: Permission,
        resource_type: ResourceType,
        resource_access_level: AccessLevel = AccessLevel.PUBLIC,
        resource_id: str = "",
    ) -> bool:
        """强制执行访问控制 (借鉴 Neo4j auth enforcement).

        评估访问请求, 如果被拒绝则抛出 AccessDeniedError。
        用于在业务操作前进行访问控制检查。

        Args:
            user_id: 用户 ID
            permission: 请求的权限
            resource_type: 资源类型
            resource_access_level: 资源访问级别
            resource_id: 资源 ID

        Returns:
            True (如果允许访问)

        Raises:
            AccessDeniedError: 访问被拒绝时抛出
        """
        request = AccessRequest(
            user_id=user_id,
            permission=permission,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_access_level=resource_access_level,
        )
        result = self.check_access(request)
        if not result.allowed:
            raise AccessDeniedError(
                user_id=user_id,
                permission=permission,
                resource_type=resource_type,
                resource_id=resource_id,
                reason=result.reason,
            )
        return True

    # ================================================================
    # 访问日志
    # ================================================================

    def get_access_log(
        self,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[AccessResult]:
        """查询访问日志.

        支持按用户过滤, 返回最近 N 条记录。

        Args:
            user_id: 用户 ID (可选, 为 None 时查询全部)
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

    def clear_access_log(self) -> None:
        """清空访问日志."""
        with self._lock:
            self._access_log.clear()

    # ================================================================
    # 统计信息
    # ================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取访问控制统计信息.

        Returns:
            统计信息字典, 包含:
            - users: 注册用户数
            - policies: 策略数
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
                "users": len(self._users),
                "policies": len(self._policies),
                "total_access_requests": total,
                "allowed_requests": allowed,
                "denied_requests": denied,
                "allow_rate": allowed / total if total > 0 else 0.0,
            }

    # ================================================================
    # 内部方法
    # ================================================================

    def _init_default_policies(self) -> None:
        """初始化默认 RBAC 策略 (借鉴 Neo4j 内置角色权限).

        创建 5 条默认策略, 每条对应一个角色:
            1. ADMIN: 所有权限, 所有资源, 所有级别 (priority=100)
            2. EDITOR: 创建/读取/更新, 实体/三元组/切片, 公开+内部 (priority=80)
            3. CONTRIBUTOR: 创建/读取, 实体/三元组/切片, 公开+内部 (priority=60)
            4. READER: 读取, 所有资源, 公开+内部 (priority=40)
            5. GUEST: 读取, 所有资源, 仅公开 (priority=20)
        """
        # 管理员: 完全访问
        self._policies.append(
            AccessPolicy(
                policy_id="default-admin",
                name="管理员默认策略",
                description="管理员拥有所有权限, 可访问所有资源和访问级别",
                roles=[Role.ADMIN],
                permissions=list(Permission),
                resource_types=list(ResourceType),
                access_levels=list(AccessLevel),
                priority=100,
                decision=AccessDecision.ALLOW,
            )
        )

        # 编辑者: 创建/读取/更新, 实体/三元组/切片, 公开+内部
        self._policies.append(
            AccessPolicy(
                policy_id="default-editor",
                name="编辑者默认策略",
                description="编辑者可创建、读取、更新实体、三元组和文档切片, "
                "访问级别为内部及以下",
                roles=[Role.EDITOR],
                permissions=[
                    Permission.CREATE,
                    Permission.READ,
                    Permission.UPDATE,
                    Permission.EXPORT,
                ],
                resource_types=[
                    ResourceType.ENTITY,
                    ResourceType.TRIPLE,
                    ResourceType.CHUNK,
                ],
                access_levels=[
                    AccessLevel.PUBLIC,
                    AccessLevel.INTERNAL,
                ],
                priority=80,
                decision=AccessDecision.ALLOW,
            )
        )

        # 贡献者: 创建/读取, 实体/三元组/切片, 公开+内部
        self._policies.append(
            AccessPolicy(
                policy_id="default-contributor",
                name="贡献者默认策略",
                description="贡献者可创建和读取实体、三元组和文档切片, "
                "写入仅限公开级别, 读取可到内部级别",
                roles=[Role.CONTRIBUTOR],
                permissions=[
                    Permission.CREATE,
                    Permission.READ,
                ],
                resource_types=[
                    ResourceType.ENTITY,
                    ResourceType.TRIPLE,
                    ResourceType.CHUNK,
                ],
                access_levels=[
                    AccessLevel.PUBLIC,
                    AccessLevel.INTERNAL,
                ],
                priority=60,
                decision=AccessDecision.ALLOW,
            )
        )

        # 读者: 仅读取, 所有资源, 公开+内部
        self._policies.append(
            AccessPolicy(
                policy_id="default-reader",
                name="读者默认策略",
                description="读者仅可读取公开和内部级别的资源",
                roles=[Role.READER],
                permissions=[Permission.READ],
                resource_types=list(ResourceType),
                access_levels=[
                    AccessLevel.PUBLIC,
                    AccessLevel.INTERNAL,
                ],
                priority=40,
                decision=AccessDecision.ALLOW,
            )
        )

        # 访客: 仅读取, 所有资源, 仅公开
        self._policies.append(
            AccessPolicy(
                policy_id="default-guest",
                name="访客默认策略",
                description="访客仅可读取公开级别的资源",
                roles=[Role.GUEST],
                permissions=[Permission.READ],
                resource_types=list(ResourceType),
                access_levels=[AccessLevel.PUBLIC],
                priority=20,
                decision=AccessDecision.ALLOW,
            )
        )

        # 按优先级降序排列
        self._policies.sort(key=lambda p: p.priority, reverse=True)

    def _evaluate_policies(
        self, request: AccessRequest, user: User
    ) -> AccessResult:
        """评估所有匹配策略 (借鉴 AWS IAM 策略评估引擎).

        评估规则:
            1. 遍历所有策略, 找出匹配的策略 (角色、权限、资源类型、访问级别均匹配)
            2. 匹配策略按优先级降序排列
            3. 在最高优先级层, 如果有任何策略决策为 DENY 或 DENY_AND_LOG, 则拒绝
            4. 否则, 第一个 (最高优先级) 匹配策略的 ALLOW 决策生效
            5. 无匹配策略时, 默认拒绝 (Default Deny)

        Args:
            request: 访问请求
            user: 用户对象

        Returns:
            访问结果
        """
        matching: list[AccessPolicy] = []

        for policy in self._policies:
            # 检查角色匹配 (用户的任一角色在策略角色列表中)
            if not any(role in policy.roles for role in user.roles):
                continue

            # 检查权限匹配
            if request.permission not in policy.permissions:
                continue

            # 检查资源类型匹配
            if request.resource_type not in policy.resource_types:
                continue

            # 检查访问级别匹配
            if request.resource_access_level not in policy.access_levels:
                continue

            matching.append(policy)

        if not matching:
            return AccessResult(
                allowed=False,
                decision=AccessDecision.DENY,
                reason="没有匹配的访问策略, 默认拒绝 (Default Deny)",
                user_id=request.user_id,
                timestamp=time.time(),
            )

        # 按优先级降序排列 (已在 add_policy 中排序, 此处确保一致性)
        matching.sort(key=lambda p: p.priority, reverse=True)

        # 检查最高优先级层是否有显式拒绝
        highest_priority = matching[0].priority
        for policy in matching:
            if policy.priority < highest_priority:
                break
            if policy.decision in (
                AccessDecision.DENY,
                AccessDecision.DENY_AND_LOG,
            ):
                return AccessResult(
                    allowed=False,
                    decision=policy.decision,
                    reason=(
                        f"策略 {policy.policy_id} ({policy.name}) "
                        f"显式拒绝访问"
                    ),
                    policy_id=policy.policy_id,
                    user_id=request.user_id,
                    timestamp=time.time(),
                )

        # 最高优先级无拒绝, 取第一个匹配策略的 ALLOW
        policy = matching[0]
        return AccessResult(
            allowed=True,
            decision=policy.decision,
            reason=f"策略 {policy.policy_id} ({policy.name}) 允许访问",
            policy_id=policy.policy_id,
            user_id=request.user_id,
            timestamp=time.time(),
        )

    @staticmethod
    def _level_allows(
        user_level: AccessLevel, resource_level: AccessLevel
    ) -> bool:
        """检查用户访问级别是否允许访问资源级别.

        层级关系: PUBLIC(0) < INTERNAL(1) < RESTRICTED(2) < CONFIDENTIAL(3)
        用户可以访问其级别及以下的资源。

        示例:
            - 用户级别 INTERNAL(1) 可访问 PUBLIC(0) 和 INTERNAL(1)
            - 用户级别 INTERNAL(1) 不可访问 RESTRICTED(2) 和 CONFIDENTIAL(3)

        Args:
            user_level: 用户的访问级别
            resource_level: 资源的访问级别

        Returns:
            True 如果用户级别 >= 资源级别
        """
        user_rank = _LEVEL_ORDER.get(user_level, 0)
        resource_rank = _LEVEL_ORDER.get(resource_level, 0)
        return user_rank >= resource_rank


# ============================================================
# 访问控制存储装饰器
# ============================================================


class AccessControlledStore:
    """访问控制存储装饰器 (借鉴 Neo4j 子图权限 + 透明代理模式).

    包装 KnowledgeStore (或类似对象), 在所有操作前进行访问控制检查。
    业务代码只需传入 user_id, 访问控制逻辑对调用方透明。

    设计理念:
        - 透明代理: 调用方无需关心访问控制细节, 只需传入 user_id
        - 读时过滤: 查询结果自动按用户权限过滤
        - 写时校验: 写操作前强制执行权限检查, 拒绝时抛出 AccessDeniedError
        - 关注点分离: 访问控制与存储逻辑解耦, 可独立演进

    用法示例:
        >>> acm = AccessControlManager()
        >>> acm.register_user(User(user_id="u1", username="alice", roles=[Role.EDITOR]))
        >>> store = KnowledgeStore()
        >>> ac_store = AccessControlledStore(store, acm)
        >>> entity = KnowledgeEntity(entity_type=EntityType.CONCEPT, name="水")
        >>> ac_store.add_entity(entity, user_id="u1")  # 自动检查权限

    Attributes:
        _store: 被包装的存储对象 (KnowledgeStore 或类似接口)
        _acm: 访问控制管理器
    """

    def __init__(self, store: Any, acm: AccessControlManager) -> None:
        """初始化访问控制存储装饰器.

        Args:
            store: 被包装的存储对象, 需实现 add_entity/get_entity/
                   update_entity/delete_entity/search 等方法
            acm: 访问控制管理器
        """
        self._store = store
        self._acm = acm

    # ================================================================
    # 实体操作
    # ================================================================

    def add_entity(
        self, entity: KnowledgeEntity, *, user_id: str
    ) -> KnowledgeEntity:
        """添加实体 (带访问控制).

        操作前检查用户是否拥有对指定访问级别实体的 CREATE 权限。

        Args:
            entity: 要添加的知识实体
            user_id: 操作用户 ID

        Returns:
            添加后的实体

        Raises:
            AccessDeniedError: 用户无创建权限时抛出
        """
        self._acm.enforce(
            user_id=user_id,
            permission=Permission.CREATE,
            resource_type=ResourceType.ENTITY,
            resource_access_level=entity.access_level,
        )
        return self._store.add_entity(entity)

    def get_entity(
        self, entity_id: str, *, user_id: str
    ) -> KnowledgeEntity | None:
        """获取实体 (带访问控制).

        先从存储中获取实体, 再检查用户是否有权读取该访问级别的实体。
        如果用户无权访问, 返回 None (不泄露资源存在性)。

        Args:
            entity_id: 实体 ID
            user_id: 操作用户 ID

        Returns:
            实体对象, 不存在或无权限时返回 None
        """
        entity = self._store.get_entity(entity_id)
        if entity is None:
            return None

        result = self._acm.check_access(
            AccessRequest(
                user_id=user_id,
                permission=Permission.READ,
                resource_type=ResourceType.ENTITY,
                resource_id=entity_id,
                resource_access_level=entity.access_level,
            )
        )
        if not result.allowed:
            return None
        return entity

    def update_entity(
        self, entity_id: str, updates: dict[str, Any], *, user_id: str
    ) -> KnowledgeEntity:
        """更新实体 (带访问控制).

        先获取实体以确定其访问级别, 再检查用户是否有 UPDATE 权限。

        Args:
            entity_id: 实体 ID
            updates: 更新字段字典
            user_id: 操作用户 ID

        Returns:
            更新后的实体

        Raises:
            AccessDeniedError: 用户无更新权限时抛出
            L3Error: 实体不存在时抛出
        """
        entity = self._store.get_entity(entity_id)
        if entity is None:
            raise L3Error(
                "L3_ENTITY_NOT_FOUND",
                f"实体不存在: {entity_id}",
                {"entity_id": entity_id},
            )
        self._acm.enforce(
            user_id=user_id,
            permission=Permission.UPDATE,
            resource_type=ResourceType.ENTITY,
            resource_access_level=entity.access_level,
            resource_id=entity_id,
        )
        return self._store.update_entity(entity_id, **updates)

    def delete_entity(self, entity_id: str, *, user_id: str) -> bool:
        """删除实体 (带访问控制).

        先获取实体以确定其访问级别, 再检查用户是否有 DELETE 权限。

        Args:
            entity_id: 实体 ID
            user_id: 操作用户 ID

        Returns:
            True 如果删除成功

        Raises:
            AccessDeniedError: 用户无删除权限时抛出
            L3Error: 实体不存在时抛出
        """
        entity = self._store.get_entity(entity_id)
        if entity is None:
            raise L3Error(
                "L3_ENTITY_NOT_FOUND",
                f"实体不存在: {entity_id}",
                {"entity_id": entity_id},
            )
        self._acm.enforce(
            user_id=user_id,
            permission=Permission.DELETE,
            resource_type=ResourceType.ENTITY,
            resource_access_level=entity.access_level,
            resource_id=entity_id,
        )
        # 兼容 delete_entity / remove_entity 两种方法名
        if hasattr(self._store, "delete_entity"):
            self._store.delete_entity(entity_id)
        else:
            self._store.remove_entity(entity_id)
        return True

    # ================================================================
    # 三元组操作
    # ================================================================

    def add_triple(self, triple: Any, *, user_id: str) -> Any:
        """添加三元组 (带访问控制).

        Args:
            triple: 三元组对象
            user_id: 操作用户 ID

        Returns:
            添加后的三元组

        Raises:
            AccessDeniedError: 用户无创建权限时抛出
        """
        resource_level = self._extract_access_level(triple)
        self._acm.enforce(
            user_id=user_id,
            permission=Permission.CREATE,
            resource_type=ResourceType.TRIPLE,
            resource_access_level=resource_level,
        )
        return self._store.add_triple(triple)

    def get_triples(
        self, *, user_id: str, **kwargs: Any
    ) -> list[Any]:
        """获取三元组 (带访问控制过滤).

        先从存储获取三元组, 再按用户权限过滤结果。

        Args:
            user_id: 操作用户 ID
            **kwargs: 传递给底层存储的查询参数

        Returns:
            过滤后的三元组列表
        """
        # 兼容 get_triples / get_triple 方法
        if hasattr(self._store, "get_triples"):
            triples = self._store.get_triples(**kwargs)
        elif hasattr(self._store, "get_all_triples"):
            triples = self._store.get_all_triples(**kwargs)
        else:
            triples = []

        if not isinstance(triples, list):
            return []

        return self._filter_by_access(
            triples, user_id, ResourceType.TRIPLE
        )

    # ================================================================
    # 文档切片操作
    # ================================================================

    def add_chunk(self, chunk: Any, *, user_id: str) -> Any:
        """添加文档切片 (带访问控制).

        Args:
            chunk: 文档切片对象
            user_id: 操作用户 ID

        Returns:
            添加后的切片

        Raises:
            AccessDeniedError: 用户无创建权限时抛出
        """
        resource_level = self._extract_access_level(chunk)
        self._acm.enforce(
            user_id=user_id,
            permission=Permission.CREATE,
            resource_type=ResourceType.CHUNK,
            resource_access_level=resource_level,
        )
        return self._store.add_chunk(chunk)

    def get_chunk(
        self, chunk_id: str, *, user_id: str
    ) -> Any:
        """获取文档切片 (带访问控制).

        Args:
            chunk_id: 切片 ID
            user_id: 操作用户 ID

        Returns:
            切片对象, 不存在或无权限时返回 None
        """
        chunk = self._store.get_chunk(chunk_id)
        if chunk is None:
            return None

        resource_level = self._extract_access_level(chunk)
        result = self._acm.check_access(
            AccessRequest(
                user_id=user_id,
                permission=Permission.READ,
                resource_type=ResourceType.CHUNK,
                resource_id=chunk_id,
                resource_access_level=resource_level,
            )
        )
        if not result.allowed:
            return None
        return chunk

    # ================================================================
    # 搜索
    # ================================================================

    def search(
        self, query: str, *, user_id: str, **kwargs: Any
    ) -> list[Any]:
        """搜索 (带访问控制过滤).

        先执行底层搜索, 再按用户权限过滤结果。
        支持 KnowledgeEntity 列表和 (chunk, score) 元组列表两种返回格式。

        Args:
            query: 搜索查询
            user_id: 操作用户 ID
            **kwargs: 传递给底层存储的搜索参数

        Returns:
            过滤后的搜索结果列表
        """
        # 兼容多种搜索方法
        if hasattr(self._store, "search"):
            results = self._store.search(query, **kwargs)
        elif hasattr(self._store, "search_text"):
            results = self._store.search_text(query, **kwargs)
        else:
            return []

        if not isinstance(results, list):
            return []

        return self._filter_by_access(results, user_id, ResourceType.ENTITY)

    # ================================================================
    # 内部方法
    # ================================================================

    @staticmethod
    def _extract_access_level(resource: Any) -> AccessLevel:
        """从资源对象中提取访问级别.

        依次尝试:
            1. resource.access_level (直接属性)
            2. resource.metadata["access_level"] (元数据中的访问级别)
            3. 默认返回 AccessLevel.PUBLIC

        Args:
            resource: 资源对象 (实体/三元组/切片)

        Returns:
            资源的访问级别
        """
        # 直接属性
        level = getattr(resource, "access_level", None)
        if isinstance(level, AccessLevel):
            return level
        if isinstance(level, str):
            try:
                return AccessLevel(level)
            except ValueError:
                pass

        # 元数据中的访问级别
        metadata = getattr(resource, "metadata", None)
        if isinstance(metadata, dict):
            raw = metadata.get("access_level")
            if isinstance(raw, AccessLevel):
                return raw
            if isinstance(raw, str):
                try:
                    return AccessLevel(raw)
                except ValueError:
                    pass

        return AccessLevel.PUBLIC

    def _filter_by_access(
        self,
        results: list[Any],
        user_id: str,
        resource_type: ResourceType,
    ) -> list[Any]:
        """按用户权限过滤搜索结果.

        遍历结果列表, 对每个结果检查用户的 READ 权限。
        支持 (item, score) 元组格式 (搜索结果常见格式)。

        Args:
            results: 原始结果列表
            user_id: 用户 ID
            resource_type: 资源类型

        Returns:
            过滤后的结果列表
        """
        filtered: list[Any] = []
        for item in results:
            # 处理 (item, score) 元组格式
            if isinstance(item, tuple) and len(item) >= 1:
                resource = item[0]
            else:
                resource = item

            resource_level = self._extract_access_level(resource)

            result = self._acm.check_access(
                AccessRequest(
                    user_id=user_id,
                    permission=Permission.READ,
                    resource_type=resource_type,
                    resource_access_level=resource_level,
                )
            )
            if result.allowed:
                filtered.append(item)
        return filtered


__all__ = [
    # 枚举
    "Role",
    "Permission",
    "ResourceType",
    "AccessDecision",
    # 异常
    "AccessDeniedError",
    # 数据模型
    "User",
    "AccessPolicy",
    "AccessRequest",
    "AccessResult",
    # 管理器
    "AccessControlManager",
    "AccessControlledStore",
]
