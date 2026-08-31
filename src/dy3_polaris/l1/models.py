"""L1 用户域核心数据模型 — 全系统用户层基础数据结构.

设计依据:
- L1 设计文档第二章: 角色分级体系 (RBAC+ABAC 混合模型)
- L1 设计文档第三章: 学习上下文经纪 (Context Envelope + Ebbinghaus 衰减)
- L1 设计文档第四章: 人机协同 (HiTL 四类场景 + 置信度门控)
- L1 设计文档第五章: 学习会话管理 (Session + Fork)
- L1 设计文档第六章: 隐私保护 (数据分级 + 脱敏)
- L1 设计文档第七章: ER 图与技术实现

融合世界先进方案:
- OpenAI Platform: RBAC + ABAC 混合权限模型 (粗粒度角色 + 细粒度属性)
- LangChain Memory: 上下文信封模式 (Context Envelope 作为跨层唯一载体)
- Anki/FSRS: 间隔重复稳定性模型 (stability = base + reps * gain)
- Khan Academy: BKT 后验概率 + 遗忘曲线衰减
- Temporal: Session Fork + Checkpoint 机制
- Cedar/OPA: ABAC 策略引擎属性维度设计
- PROV-O: 审计日志溯源模型
- Bloom's Taxonomy: 认知层级与学习阶段映射

模块组成:
1. 常量: 时间转换、衰减公式参数、系统阈值
2. 衰减函数: calculate_decay (Ebbinghaus 遗忘曲线 + 间隔重复修正)
3. 角色与权限枚举: UserRole / UserStatus / Permission (13 项)
4. ABAC 枚举与模型: GradeLevel / MajorDirection / LabAccessTier / ABACAttributes / User
5. 学习上下文: LearningPhase / MasterySnapshot / LearningGoal / LearningState / ContextEnvelope
6. 会话与 Fork: SessionType / SessionStatus / LearningSession / SessionFork
7. 审计与脱敏: DataLevel / AuditAction / AuditResult / AuditLogEntry
8. HiTL 协同: HiTLType / HiTLPriority / ConfidenceGateResult / FeedbackType
9. HiTL 数据模型: ApprovalRequest / ApprovalResponse / FeedbackReport / EmergencyAlert

跨层对齐:
- MasterySnapshot.kc_id ↔ L3 KPMastery.kp_id
- MasterySnapshot.p_know ↔ L3 KPMastery.mastery_prob
- ContextEnvelope 是 L1 → L2/L3/L4 的唯一数据载体
"""

from __future__ import annotations

import itertools
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# 统一 ID 命名空间 (单点: shared/ids.py)
from dy3_polaris.shared.ids import new_session_id as _new_session_id


# ============================================================
# 1. 常量定义
# ============================================================

# --- 时间转换 ---
MS_PER_HOUR: int = 3_600_000  # 1 小时 = 3,600,000 毫秒
MS_PER_SEC: int = 1_000       # 1 秒 = 1,000 毫秒

# --- Ebbinghaus 遗忘曲线参数 (设计文档 3.4) ---
MIN_STABILITY: float = 24.0   # 最小记忆稳定性 (小时), 新学知识的初始稳定性
STABILITY_GAIN: float = 24.0   # 每次练习增加的稳定性 (小时), 间隔重复效应 (1次练习≈1天稳定性)
PRIOR_PROB: float = 0.3       # BKT 先验概率 P(Know), 衰减下限

# --- 默认值 ---
MIN_DECAY: float = 1.0        # 初始衰减系数 (无衰减)
DEFAULT_REPS: int = 0          # 默认练习次数
PRIORITY_NORMAL: int = 3       # 默认目标优先级 (1-5, 3 为中等)
DEFAULT_SESSION_MS: int = 0    # 默认会话时长 (毫秒)
DEFAULT_INTERACTIONS: int = 0  # 默认交互次数
DEFAULT_COGNITIVE_LOAD: float = 0.5  # 默认认知负荷 (中等)
DEFAULT_TTL: int = 3600        # 默认上下文 TTL (秒, 1 小时)

# --- 系统阈值 ---
WEAK_THRESHOLD: float = 0.5          # 薄弱知识点阈值 (有效掌握度 < 0.5)
EMERGENCY_THRESHOLD: float = 0.95    # 紧急干预阈值 (认知负荷 >= 0.95)
BLOCK_THRESHOLD: float = 0.4         # 置信度阻断阈值 (< 0.4 阻止呈现)
WARNING_THRESHOLD: float = 0.85      # 置信度警告阈值 (>= 0.85 直接放行)
MAX_DAILY_AGENT_CALLS: int = 20      # 本科生每日 Agent 调用上限

# --- 隐私保护参数 (设计文档 6.4) ---
K_ANONYMITY_MIN: int = 5           # K-匿名最小组大小
L_DIVERSITY_MIN: int = 3           # l-多样性最小值

# --- 系统监控阈值 (设计文档 3.4, 4.2) ---
COGNITIVE_LOAD_RECALC_INTERVAL: int = 5   # 每5次交互重新计算认知负荷
FAST_ANSWER_THRESHOLD_MS: int = 5_000     # 答题速度<5秒判定异常
CONSECUTIVE_ERROR_THRESHOLD: int = 10     # 连续错误>=10次触发干预
BKT_DEVIATION_THRESHOLD: float = 0.3      # BKT预测偏差>30%触发纠错


# ============================================================
# 2. 衰减函数 (设计文档 3.4 Ebbinghaus 遗忘曲线)
# ============================================================


def calculate_decay(
    p_know: float,
    last_practiced: int,
    repetitions: int,
    current_ts: int,
) -> float:
    """计算知识掌握度的遗忘衰减.

    基于 Ebbinghaus 遗忘曲线 + 间隔重复修正:
    1. 计算自上次练习以来的经过时间 (小时)
    2. 根据练习次数计算记忆稳定性 (stability = MIN_STABILITY + reps * STABILITY_GAIN)
    3. 指数衰减: decay = exp(-elapsed_hours / stability)
    4. 有效掌握度: effective = p_know * decay
    5. 不低于先验概率 PRIOR_PROB (除非 p_know == 0)

    Args:
        p_know: BKT 后验掌握概率 [0.0, 1.0]
        last_practiced: 上次练习时间戳 (毫秒)
        repetitions: 该 KC 累计练习次数 (>= 0)
        current_ts: 当前时间戳 (毫秒)

    Returns:
        衰减后的有效掌握度 [PRIOR_PROB, p_know]

    Raises:
        ValueError: p_know 不在 [0.0, 1.0] 或 repetitions < 0
    """
    if not (0.0 <= p_know <= 1.0):
        raise ValueError(
            f"p_know must be in [0.0, 1.0], got {p_know}"
        )
    if repetitions < 0:
        raise ValueError(
            f"repetitions must be >= 0, got {repetitions}"
        )

    # p_know == 0 时, 学习者从未掌握该知识, 衰减后仍为 0
    if p_know == 0.0:
        return 0.0

    elapsed_hours = max(0, (current_ts - last_practiced) / MS_PER_HOUR)

    # 记忆稳定性: 练习次数越多, 遗忘越慢
    stability = MIN_STABILITY + repetitions * STABILITY_GAIN

    # Ebbinghaus 指数衰减
    decay = math.exp(-elapsed_hours / stability)

    # 有效掌握度 = 原始掌握度 × 衰减系数
    effective_mastery = p_know * decay

    # 不低于先验概率 (除非 p_know == 0, 已在上面处理)
    return max(effective_mastery, PRIOR_PROB)


# ============================================================
# 2.1 BKT 四参数模型 (设计文档 8.2)
# ============================================================


@dataclass
class BKTParams:
    """BKT 四参数模型 (设计文档 8.2).

    贝叶斯知识追踪 (Bayesian Knowledge Tracing) 经典四参数:
    - p_know: 已掌握概率 P(Know) [0.0, 1.0]
    - p_slip: 已掌握但答错的概率 (失误) P(Slip) [0.0, 1.0]
    - p_guess: 未掌握但答对的概率 (猜测) P(Guess) [0.0, 1.0]
    - p_transit: 从未掌握到掌握的转移概率 (学习) P(Transit) [0.0, 1.0]
    """

    p_know: float
    p_slip: float = 0.1
    p_guess: float = 0.25
    p_transit: float = 0.1

    def __post_init__(self) -> None:
        """校验所有参数在 [0.0, 1.0] 范围内."""
        for name, val in [
            ("p_know", self.p_know),
            ("p_slip", self.p_slip),
            ("p_guess", self.p_guess),
            ("p_transit", self.p_transit),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {val}"
                )

    def bayesian_update(self, is_correct: bool) -> BKTParams:
        """BKT 贝叶斯后验更新.

        根据观测结果 (答对/答错) 更新 P(Know):
        - 答对: P(Know|correct) = P(K)*P(!S) / [P(K)*P(!S) + P(!K)*P(G)]
        - 答错: P(Know|incorrect) = P(K)*P(S) / [P(K)*P(S) + P(!K)*P(!G)]
        然后应用学习转移: P(Know)' = P(Know) + (1-P(Know)) * P(Transit)

        Args:
            is_correct: 是否答对

        Returns:
            更新后的新 BKTParams 对象 (不可变风格)
        """
        if is_correct:
            p_know_post = (self.p_know * (1 - self.p_slip)) / (
                self.p_know * (1 - self.p_slip)
                + (1 - self.p_know) * self.p_guess
            )
        else:
            p_know_post = (self.p_know * self.p_slip) / (
                self.p_know * self.p_slip
                + (1 - self.p_know) * (1 - self.p_guess)
            )
        # 应用学习转移 (transit)
        p_know_post = p_know_post + (1 - p_know_post) * self.p_transit
        return BKTParams(
            p_know=p_know_post,
            p_slip=self.p_slip,
            p_guess=self.p_guess,
            p_transit=self.p_transit,
        )

    def predict_correct_prob(self) -> float:
        """预测答对概率 P(correct) = P(K)*P(!S) + P(!K)*P(G)."""
        return (
            self.p_know * (1 - self.p_slip)
            + (1 - self.p_know) * self.p_guess
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "p_know": self.p_know,
            "p_slip": self.p_slip,
            "p_guess": self.p_guess,
            "p_transit": self.p_transit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BKTParams:
        """从字典反序列化."""
        return cls(
            p_know=d["p_know"],
            p_slip=d.get("p_slip", 0.1),
            p_guess=d.get("p_guess", 0.25),
            p_transit=d.get("p_transit", 0.1),
        )


# ============================================================
# 3. 角色与权限枚举 (设计文档第二章)
# ============================================================


class UserRole(str, Enum):
    """用户角色 (设计文档 2.1, RBAC 粗粒度权限划分).

    六种角色覆盖完整教育生态:
    - UNDERGRAD: 本科生 — 知识学习与诊断的主体用户
    - GRADUATE: 研究生 — 拥有实验数据访问与高级 Agent 调用权限
    - RESEARCHER: 科研员 — 企业研发/工程技术人员, 权限与研究生同级
    - TEACHER: 教师 — 内容审核、学情查看、实验指导权限
    - ADMIN: 管理员 — 系统配置与用户管理全权限
    - ALUMNI: 校友 — 只读权限, 不参与教学活动
    """

    UNDERGRAD = "undergrad"
    GRADUATE = "graduate"
    RESEARCHER = "researcher"
    TEACHER = "teacher"
    ADMIN = "admin"
    ALUMNI = "alumni"


class UserStatus(str, Enum):
    """用户状态 (设计文档 ER 图 User.status).

    - ACTIVE: 活跃 — 正常使用系统
    - SUSPENDED: 停用 — 因违规或安全原因暂停
    - ALUMNI: 校友 — 毕业后转为只读状态
    """

    ACTIVE = "active"
    SUSPENDED = "suspended"
    ALUMNI = "alumni"


class Permission(str, Enum):
    """功能权限枚举 (设计文档 2.2 权限矩阵).

    12 项功能权限 + 1 项 HiTL 确认权限 = 13 项:
    - 知识库 (3): 公开读 / 内部数据访问 / 写入编辑
    - Agent 调用 (4): 诊断 / 知识生成 / 审核校验 / 导学决策
    - 学情数据 (3): 查看自身报告 / 查看学生报告 / 导出报告
    - 系统管理 (2): 系统配置 / 用户管理
    - HiTL (1): 人机协同确认
    """

    # 知识库权限
    KB_PUBLIC_READ = "kb_public_read"
    KB_INTERNAL_DATA_ACCESS = "kb_internal_data_access"
    KB_WRITE_EDIT = "kb_write_edit"

    # Agent 调用权限
    AGENT_DIAGNOSIS = "agent_diagnosis"
    AGENT_KNOWLEDGE_GEN = "agent_knowledge_gen"
    AGENT_REVIEW = "agent_review"
    AGENT_GUIDE = "agent_guide"

    # 学情数据权限
    VIEW_OWN_REPORT = "view_own_report"
    VIEW_STUDENT_REPORT = "view_student_report"
    EXPORT_REPORT = "export_report"

    # 系统管理权限
    SYSTEM_CONFIG = "system_config"
    USER_MANAGE = "user_manage"

    # HiTL 确认权限
    HITL_CONFIRM = "hitl_confirm"


# ============================================================
# 3.1 Role 独立模型 (设计文档 ER 图 Role 表)
# ============================================================


_role_counter = itertools.count(1)


@dataclass
class Role:
    """角色模型 (设计文档 ER 图 Role 表, 与 UserRole 枚举配合).

    独立的角色实体, 支持权限矩阵的持久化与动态配置:
    - role_id: 角色自增 ID (自动生成)
    - role_code: 角色代码 (与 UserRole 枚举值对齐)
    - role_name: 角色中文名称
    - base_permissions: 该角色的基础权限列表 (RBAC 粗粒度)
    """

    role_code: str
    role_name: str
    base_permissions: list[Permission]
    role_id: int = field(default_factory=lambda: next(_role_counter))

    def has_permission(self, perm: Permission) -> bool:
        """检查该角色是否拥有指定权限."""
        return perm in self.base_permissions

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "role_id": self.role_id,
            "role_code": self.role_code,
            "role_name": self.role_name,
            "base_permissions": [p.value for p in self.base_permissions],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Role:
        """从字典反序列化."""
        perms_raw = d.get("base_permissions", [])
        perms = [
            Permission(p) if isinstance(p, str) else p
            for p in perms_raw
        ]
        return cls(
            role_code=d["role_code"],
            role_name=d["role_name"],
            base_permissions=perms,
            role_id=d.get("role_id", next(_role_counter)),
        )

    @classmethod
    def default_roles(cls) -> list[Role]:
        """返回 5 种预定义角色 (对应 UserRole 枚举)."""
        return [
            cls(
                role_code="undergrad",
                role_name="本科生",
                base_permissions=[
                    Permission.KB_PUBLIC_READ,
                    Permission.AGENT_DIAGNOSIS,
                    Permission.AGENT_GUIDE,
                    Permission.VIEW_OWN_REPORT,
                    Permission.HITL_CONFIRM,
                ],
            ),
            cls(
                role_code="graduate",
                role_name="研究生",
                base_permissions=[
                    Permission.KB_PUBLIC_READ,
                    Permission.KB_INTERNAL_DATA_ACCESS,
                    Permission.AGENT_DIAGNOSIS,
                    Permission.AGENT_KNOWLEDGE_GEN,
                    Permission.AGENT_GUIDE,
                    Permission.VIEW_OWN_REPORT,
                    Permission.EXPORT_REPORT,
                    Permission.HITL_CONFIRM,
                ],
            ),
            cls(
                role_code="teacher",
                role_name="教师",
                base_permissions=[
                    Permission.KB_PUBLIC_READ,
                    Permission.KB_INTERNAL_DATA_ACCESS,
                    Permission.KB_WRITE_EDIT,
                    Permission.AGENT_REVIEW,
                    Permission.AGENT_GUIDE,
                    Permission.VIEW_STUDENT_REPORT,
                    Permission.EXPORT_REPORT,
                    Permission.HITL_CONFIRM,
                ],
            ),
            cls(
                role_code="admin",
                role_name="管理员",
                base_permissions=list(Permission),
            ),
            cls(
                role_code="alumni",
                role_name="校友",
                base_permissions=[
                    Permission.KB_PUBLIC_READ,
                    Permission.VIEW_OWN_REPORT,
                ],
            ),
        ]


# ============================================================
# 4. ABAC 枚举 (设计文档 2.3, 5 个属性维度)
# ============================================================


class GradeLevel(str, Enum):
    """年级层级 (ABAC 属性维度 1: grade_level).

    六个层级覆盖本科到博士:
    - FRESHMAN: 大一
    - SOPHOMORE: 大二
    - JUNIOR: 大三
    - SENIOR: 大四
    - MASTER: 硕士
    - PHD: 博士
    """

    FRESHMAN = "freshman"
    SOPHOMORE = "sophomore"
    JUNIOR = "junior"
    SENIOR = "senior"
    MASTER = "master"
    PHD = "phd"


class MajorDirection(str, Enum):
    """专业方向 (ABAC 属性维度 2: major_direction).

    五个方向对应 Dy3+ 材料科学领域:
    - PHYSICS: 物理学 (光谱学、能级理论)
    - CHEMISTRY: 化学 (配位化学、合成)
    - MATERIALS_SCI: 材料科学 (发光材料、器件)
    - OPTICS: 光学 (光学测量、发光性能)
    - ENGINEERING: 工程学 (器件设计、工艺)
    """

    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    MATERIALS_SCI = "materials_sci"
    OPTICS = "optics"
    ENGINEERING = "engineering"


class LabAccessTier(str, Enum):
    """实验权限等级 (ABAC 属性维度 4: lab_access_tier).

    四级递进权限:
    - TIER0: 虚拟实验 (仅模拟, 无实机操作)
    - TIER1: 基础实验 (光谱测量等低风险操作)
    - TIER2: 高级实验 (材料合成等中风险操作)
    - TIER3: 危险实验 (高温高压等高风险操作)
    """

    TIER0 = "tier0"
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"


# ============================================================
# 5. ABAC 属性与用户模型 (设计文档 2.3, ER 图 User 表)
# ============================================================


@dataclass
class ABACAttributes:
    """ABAC 属性模型 (设计文档 2.3, 5 个属性维度).

    用于细粒度条件化权限控制:
    - grade_level: 年级 → 控制内容难度范围
    - major_direction: 专业 → 控制知识库领域范围
    - course_progress: 课程进度 [0.0, 1.0] → 控制实验推荐门槛
    - lab_access_tier: 实验权限 → 控制实验指导 Agent 调用
    - supervisor_id: 导师 ID → 研究生数据访问范围限制
    """

    grade_level: GradeLevel = GradeLevel.FRESHMAN
    major_direction: MajorDirection = MajorDirection.MATERIALS_SCI
    course_progress: float = 0.0
    lab_access_tier: LabAccessTier = LabAccessTier.TIER0
    supervisor_id: str | None = None
    daily_agent_calls: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.course_progress <= 1.0):
            raise ValueError(
                f"course_progress must be in [0.0, 1.0], got {self.course_progress}"
            )

    def increment_agent_calls(self) -> None:
        """递增每日 Agent 调用计数."""
        self.daily_agent_calls += 1

    def reset_daily_calls(self) -> None:
        """重置每日 Agent 调用计数 (每日 0 点调用)."""
        self.daily_agent_calls = 0

    def can_invoke_agent(self) -> bool:
        """检查是否仍可调用 Agent (未超过每日上限)."""
        return self.daily_agent_calls < MAX_DAILY_AGENT_CALLS

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "grade_level": self.grade_level.value,
            "major_direction": self.major_direction.value,
            "course_progress": self.course_progress,
            "lab_access_tier": self.lab_access_tier.value,
            "supervisor_id": self.supervisor_id,
            "daily_agent_calls": self.daily_agent_calls,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ABACAttributes:
        """从字典反序列化."""
        return cls(
            grade_level=GradeLevel(d.get("grade_level", "freshman")),
            major_direction=MajorDirection(d.get("major_direction", "materials_sci")),
            course_progress=d.get("course_progress", 0.0),
            lab_access_tier=LabAccessTier(d.get("lab_access_tier", "tier0")),
            supervisor_id=d.get("supervisor_id"),
            daily_agent_calls=d.get("daily_agent_calls", 0),
        )


@dataclass
class User:
    """用户模型 (设计文档 ER 图 User 表).

    整合身份信息、角色与 ABAC 属性:
    - user_id: 系统内部唯一 ID (自动生成, 前缀 "u-")
    - student_id: 学号 (机构核发, 不可自主创建)
    - institution_id: 机构 ID (单机构多角色模型)
    - role: RBAC 角色 (粗粒度权限)
    - status: 账户状态
    - abac_attributes: ABAC 属性 (细粒度权限)
    - created_at / updated_at: 时间戳 (毫秒)
    """

    student_id: str
    institution_id: str
    role: UserRole
    user_id: str = field(default_factory=lambda: f"u-{uuid.uuid4().hex[:12]}")
    status: UserStatus = UserStatus.ACTIVE
    abac_attributes: ABACAttributes = field(default_factory=ABACAttributes)
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))
    learning_style: Any = field(default=None)  # type: ignore[assignment]

    # student_id 格式校验正则 (设计文档 3.3 JSON Schema): 2位大写字母 + 8位数字
    _STUDENT_ID_PATTERN = re.compile(r"^[A-Z]{2}\d{8}$")

    def __post_init__(self) -> None:
        """校验 student_id 格式 (2位大写字母 + 8位数字)."""
        if not self._STUDENT_ID_PATTERN.match(self.student_id):
            raise ValueError(
                f"student_id must match pattern ^[A-Z]{{2}}\\d{{8}}$, "
                f"got '{self.student_id}'"
            )

    def touch(self) -> None:
        """更新 updated_at 为当前时间."""
        self.updated_at = int(time.time() * 1000)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "user_id": self.user_id,
            "student_id": self.student_id,
            "institution_id": self.institution_id,
            "role": self.role.value,
            "status": self.status.value,
            "abac_attributes": self.abac_attributes.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "learning_style": (
                self.learning_style.to_dict()
                if self.learning_style is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> User:
        """从字典反序列化."""
        abac_raw = d.get("abac_attributes", {})
        abac = (
            ABACAttributes.from_dict(abac_raw)
            if isinstance(abac_raw, dict)
            else abac_raw
        )
        # 延迟导入 VARKProfile 避免循环依赖
        ls_raw = d.get("learning_style")
        learning_style = None
        if isinstance(ls_raw, dict):
            try:
                from dy3_polaris.l1.models import VARKProfile
                learning_style = VARKProfile.from_dict(ls_raw)
            except (ImportError, KeyError):
                learning_style = None
        return cls(
            student_id=d["student_id"],
            institution_id=d.get("institution_id", ""),
            role=UserRole(d.get("role", "undergrad")),
            user_id=d.get("user_id", f"u-{uuid.uuid4().hex[:12]}"),
            status=UserStatus(d.get("status", "active")),
            abac_attributes=abac,
            created_at=d.get("created_at", int(time.time() * 1000)),
            updated_at=d.get("updated_at", int(time.time() * 1000)),
            learning_style=learning_style,
        )


# ============================================================
# 6. 学习上下文枚举与模型 (设计文档第三章)
# ============================================================


class LearningPhase(str, Enum):
    """学习阶段 (设计文档 3.3).

    四个阶段对应完整学习周期:
    - PREVIEW: 预习 — 课前知识准备
    - PRACTICE: 练习 — 课中交互式学习
    - QUIZ: 测验 — 知识掌握度评估
    - REVIEW: 复习 — 课后巩固与遗忘补偿
    """

    PREVIEW = "preview"
    PRACTICE = "practice"
    QUIZ = "quiz"
    REVIEW = "review"


@dataclass
class MasterySnapshot:
    """知识掌握快照 (设计文档 3.3, 3.6).

    记录学习者对单个知识组件 (KC) 的掌握状态:
    - kc_id: 知识组件 ID (如 "dy3_energy_level_4f")
    - p_know: BKT 后验掌握概率 [0.0, 1.0]
    - last_practiced_at: 最近练习时间戳 (毫秒)
    - decay_factor: 遗忘衰减系数 [0.0, 1.0], 1.0 = 无衰减
    - repetitions: 累计练习次数

    跨层对齐: kc_id ↔ L3 KPMastery.kp_id, p_know ↔ L3 KPMastery.mastery_prob
    """

    kc_id: str
    p_know: float
    last_practiced_at: int
    decay_factor: float = MIN_DECAY
    repetitions: int = DEFAULT_REPS
    bkt_params: BKTParams | None = None
    correct_count: int = 0
    attempts: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_know <= 1.0):
            raise ValueError(
                f"p_know must be in [0.0, 1.0], got {self.p_know}"
            )
        if not (0.0 <= self.decay_factor <= 1.0):
            raise ValueError(
                f"decay_factor must be in [0.0, 1.0], got {self.decay_factor}"
            )
        if self.repetitions < 0:
            raise ValueError(
                f"repetitions must be >= 0, got {self.repetitions}"
            )
        # 自动从 p_know 创建 BKTParams (如果未显式传入)
        if self.bkt_params is None:
            self.bkt_params = BKTParams(p_know=self.p_know)

    def effective_mastery(self) -> float:
        """有效掌握度 = p_know × decay_factor."""
        return self.p_know * self.decay_factor

    def is_weak(self, threshold: float = WEAK_THRESHOLD) -> bool:
        """是否为薄弱知识点 (有效掌握度低于阈值)."""
        return self.effective_mastery() < threshold

    def accuracy(self) -> float:
        """正确率 = correct_count / attempts (attempts==0 时返回 0.0)."""
        if self.attempts == 0:
            return 0.0
        return self.correct_count / self.attempts

    def to_l3_kp_mastery(self) -> Any:
        """转换为 L3 KPMastery (跨层对齐, 延迟导入避免循环依赖).

        字段映射:
        - kc_id → kp_id
        - p_know → mastery_prob
        - correct_count → correct_count
        - attempts → attempts
        - last_practiced_at (ms) → last_attempt_time (秒, float)
        """
        from dy3_polaris.l3.api_models import KPMastery

        return KPMastery(
            kp_id=self.kc_id,
            mastery_prob=self.p_know,
            attempts=self.attempts,
            correct_count=self.correct_count,
            last_attempt_time=self.last_practiced_at / MS_PER_SEC,
        )

    @classmethod
    def from_l3_kp_mastery(cls, kp: Any) -> MasterySnapshot:
        """从 L3 KPMastery 逆向转换 (跨层对齐, 延迟导入避免循环依赖).

        字段映射:
        - kp_id → kc_id
        - mastery_prob → p_know
        - correct_count → correct_count
        - attempts → attempts
        - last_attempt_time (秒, float) → last_practiced_at (ms, int)
        """
        return cls(
            kc_id=kp.kp_id,
            p_know=kp.mastery_prob,
            last_practiced_at=int(kp.last_attempt_time * MS_PER_SEC),
            correct_count=kp.correct_count,
            attempts=kp.attempts,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "kc_id": self.kc_id,
            "p_know": self.p_know,
            "last_practiced_at": self.last_practiced_at,
            "decay_factor": self.decay_factor,
            "repetitions": self.repetitions,
            "bkt_params": self.bkt_params.to_dict() if self.bkt_params else None,
            "correct_count": self.correct_count,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MasterySnapshot:
        """从字典反序列化."""
        bkt_raw = d.get("bkt_params")
        bkt = (
            BKTParams.from_dict(bkt_raw)
            if isinstance(bkt_raw, dict)
            else bkt_raw
        )
        return cls(
            kc_id=d["kc_id"],
            p_know=d["p_know"],
            last_practiced_at=d["last_practiced_at"],
            decay_factor=d.get("decay_factor", MIN_DECAY),
            repetitions=d.get("repetitions", DEFAULT_REPS),
            bkt_params=bkt,
            correct_count=d.get("correct_count", 0),
            attempts=d.get("attempts", 0),
        )


@dataclass
class LearningGoal:
    """学习目标 (设计文档 3.3).

    - description: 目标描述 (如 "掌握 Dy3+ 能级跃迁机理")
    - priority: 优先级 1-5, 5 为最高
    - deadline: 截止时间 (可为 None)
    """

    description: str
    priority: int = PRIORITY_NORMAL
    deadline: float | None = None
    bloom_level: Any = None

    def __post_init__(self) -> None:
        if not (1 <= self.priority <= 5):
            raise ValueError(
                f"priority must be in [1, 5], got {self.priority}"
            )
        # 默认 Bloom 认知层级为 UNDERSTAND (延迟导入避免循环依赖)
        if self.bloom_level is None:
            from dy3_polaris.l3.api_models import BloomLevel

            self.bloom_level = BloomLevel.UNDERSTAND

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        bloom_val = (
            self.bloom_level.value
            if isinstance(self.bloom_level, Enum)
            else self.bloom_level
        )
        return {
            "description": self.description,
            "priority": self.priority,
            "deadline": self.deadline,
            "bloom_level": bloom_val,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningGoal:
        """从字典反序列化."""
        bloom_raw = d.get("bloom_level")
        if isinstance(bloom_raw, str):
            from dy3_polaris.l3.api_models import BloomLevel

            bloom_level: Any = BloomLevel(bloom_raw)
        else:
            bloom_level = bloom_raw
        return cls(
            description=d["description"],
            priority=d.get("priority", PRIORITY_NORMAL),
            deadline=d.get("deadline"),
            bloom_level=bloom_level,
        )


@dataclass
class LearningState:
    """当前学习状态 (设计文档 3.3).

    实时反映学习者当前会话中的状态:
    - phase: 当前学习阶段
    - session_duration_ms: 会话持续时长 (毫秒)
    - interaction_count: 交互次数 (问答/答题/反馈)
    - cognitive_load: 认知负荷 [0.0, 1.0], >= 0.95 触发紧急干预
    """

    phase: LearningPhase = LearningPhase.PREVIEW
    session_duration_ms: int = DEFAULT_SESSION_MS
    interaction_count: int = DEFAULT_INTERACTIONS
    cognitive_load: float = DEFAULT_COGNITIVE_LOAD

    def __post_init__(self) -> None:
        if not (0.0 <= self.cognitive_load <= 1.0):
            raise ValueError(
                f"cognitive_load must be in [0.0, 1.0], got {self.cognitive_load}"
            )

    def is_emergency(self) -> bool:
        """认知负荷是否达到紧急干预阈值."""
        return self.cognitive_load >= EMERGENCY_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "phase": self.phase.value,
            "session_duration_ms": self.session_duration_ms,
            "interaction_count": self.interaction_count,
            "cognitive_load": self.cognitive_load,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningState:
        """从字典反序列化."""
        return cls(
            phase=LearningPhase(d.get("phase", "preview")),
            session_duration_ms=d.get("session_duration_ms", DEFAULT_SESSION_MS),
            interaction_count=d.get("interaction_count", DEFAULT_INTERACTIONS),
            cognitive_load=d.get("cognitive_load", DEFAULT_COGNITIVE_LOAD),
        )


@dataclass
class ResourceItem:
    """可用资源项 (设计文档 3.1 上下文组件 resources).

    描述学习者当前可用的学习资源:
    - resource_id: 资源唯一 ID
    - title: 资源标题
    - resource_type: 资源类型 (如 "diagram", "card", "video")
    - difficulty: 难度 [0.0, 1.0]
    """

    resource_id: str
    title: str
    resource_type: str
    difficulty: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.difficulty <= 1.0):
            raise ValueError(
                f"difficulty must be in [0.0, 1.0], got {self.difficulty}"
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "resource_id": self.resource_id,
            "title": self.title,
            "resource_type": self.resource_type,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResourceItem:
        """从字典反序列化."""
        return cls(
            resource_id=d["resource_id"],
            title=d["title"],
            resource_type=d["resource_type"],
            difficulty=d.get("difficulty", 0.5),
        )


@dataclass
class TimeConstraint:
    """时间约束 (设计文档 3.1 上下文组件 time_constraint).

    描述学习者当前可用的时间预算与推荐学习阶段:
    - available_minutes: 可用时间 (分钟)
    - recommended_phase: 推荐学习阶段
    """

    available_minutes: int = 45
    recommended_phase: LearningPhase = LearningPhase.PRACTICE

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "available_minutes": self.available_minutes,
            "recommended_phase": self.recommended_phase.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimeConstraint:
        """从字典反序列化."""
        return cls(
            available_minutes=d.get("available_minutes", 45),
            recommended_phase=LearningPhase(
                d.get("recommended_phase", "practice")
            ),
        )


@dataclass
class ContextEnvelope:
    """上下文信封 — L1 向下层传递数据的唯一载体 (设计文档 3.3, 3.6).

    封装经过权限过滤和数据最小化处理的学习者上下文:
    - envelope_id: 信封唯一 ID (自动生成)
    - user_id: 用户 ID (脱敏后, 非原始学号)
    - session_id: 会话 ID
    - timestamp: 创建时间戳 (毫秒)
    - learning_state: 当前学习状态
    - mastery_snapshot: 知识掌握快照列表
    - goals: 学习目标列表
    - ttl: 生存时间 (秒), 超时后标记为过期

    设计原则:
    - 唯一出口: 所有下层模块仅通过此对象获取用户上下文
    - 权限过滤: 传递前已根据用户角色过滤敏感数据
    - 数据最小化: 仅包含下层推理所需的最小数据集
    - 时效性: TTL 机制确保上下文新鲜度, 衰减机制模拟遗忘
    """

    user_id: str
    session_id: str
    envelope_id: str = field(default_factory=lambda: f"env-{uuid.uuid4().hex[:12]}")
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    learning_state: LearningState = field(default_factory=LearningState)
    mastery_snapshot: list[MasterySnapshot] = field(default_factory=list)
    goals: list[LearningGoal] = field(default_factory=list)
    ttl: int = DEFAULT_TTL
    resources: list[ResourceItem] = field(default_factory=list)
    time_constraint: TimeConstraint | None = None
    cognitive_load_breakdown: Any = field(default=None)  # type: ignore[assignment]
    learning_style: Any = field(default=None)  # type: ignore[assignment]
    irt_ability: Any = field(default=None)  # type: ignore[assignment]
    engagement: Any = field(default=None)  # type: ignore[assignment]
    mastery_trajectories: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查上下文是否过期 (elapsed_seconds > ttl)."""
        current_ms = int(time.time() * 1000)
        elapsed_sec = int((current_ms - self.timestamp) / MS_PER_SEC)
        return elapsed_sec > self.ttl

    def refresh_decay(self, current_ts: int) -> None:
        """刷新所有 KC 的遗忘衰减系数.

        基于当前时间重新计算每个 MasterySnapshot 的 decay_factor:
        - stability = MIN_STABILITY + repetitions * STABILITY_GAIN
        - decay = exp(-elapsed_hours / stability)
        - decay_factor = decay (不含 p_know, 纯衰减乘数)
        """
        for snap in self.mastery_snapshot:
            elapsed_hours = max(
                0, (current_ts - snap.last_practiced_at) / MS_PER_HOUR
            )
            stability = MIN_STABILITY + snap.repetitions * STABILITY_GAIN
            snap.decay_factor = math.exp(-elapsed_hours / stability)

    def get_weak_kcs(
        self, threshold: float = WEAK_THRESHOLD
    ) -> list[str]:
        """获取有效掌握度低于阈值的知识点 ID 列表.

        有效掌握度 = p_know × decay_factor, 综合考虑原始掌握度和遗忘衰减.
        """
        return [
            s.kc_id
            for s in self.mastery_snapshot
            if s.effective_mastery() < threshold
        ]

    def get_zpd_recommendation(self) -> tuple[float, str]:
        """ZPD 便捷推荐接口 — 基于当前 IRT 能力推荐难度和调整方向.

        维果茨基最近发展区 (ZPD) 理论: 推荐难度应落在学习者能力
        上下的一段区间内, 既不过于简单也不过于困难.

        Returns:
            (recommended_difficulty, adjustment_direction):
            - recommended_difficulty: 推荐难度 (ZPD 中点)
            - adjustment_direction: "increase" / "decrease" / "optimal"
            - 当 irt_ability 为 None 时返回 (0.5, "optimal")
        """
        if self.irt_ability is None:
            return (0.5, "optimal")
        theta = getattr(self.irt_ability, "theta", None)
        if theta is None:
            return (0.5, "optimal")
        zpd = ZoneOfProximalDevelopment(learner_theta=float(theta))
        recommended = zpd.recommended_difficulty()
        direction = zpd.adjustment_direction(0.5)
        return (float(recommended), direction)

    def to_summary(self) -> dict[str, Any]:
        """生成脱敏后的上下文摘要 (不含原始学号等敏感信息).

        用于 API 响应和日志记录, 仅暴露非敏感的聚合信息.
        """
        return {
            "phase": self.learning_state.phase.value,
            "cognitive_load": self.learning_state.cognitive_load,
            "weak_kc_count": len(self.get_weak_kcs()),
            "resource_count": len(self.resources),
            "has_time_constraint": self.time_constraint is not None,
        }

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "envelope_id": self.envelope_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "learning_state": self.learning_state.to_dict(),
            "mastery_snapshot": [s.to_dict() for s in self.mastery_snapshot],
            "goals": [g.to_dict() for g in self.goals],
            "ttl": self.ttl,
            "resources": [r.to_dict() for r in self.resources],
            "time_constraint": (
                self.time_constraint.to_dict()
                if self.time_constraint
                else None
            ),
            "cognitive_load_breakdown": (
                self.cognitive_load_breakdown.to_dict()
                if self.cognitive_load_breakdown is not None
                else None
            ),
            "learning_style": (
                self.learning_style.to_dict()
                if self.learning_style is not None
                else None
            ),
            "irt_ability": (
                self.irt_ability.to_dict()
                if self.irt_ability is not None
                else None
            ),
            "engagement": (
                self.engagement.to_dict()
                if self.engagement is not None
                else None
            ),
            "mastery_trajectories": (
                {k: v.to_dict() for k, v in self.mastery_trajectories.items()}
                if self.mastery_trajectories
                else {}
            ),
        }

    def to_l3_learner_profile(self) -> Any:
        """转换为 L3 LearnerProfile (跨层对齐, 延迟导入避免循环依赖).

        将 MasterySnapshot 列表转换为 KPMastery 字典,
        并填充薄弱知识点列表供 L3 个性化检索使用.
        当推导值为 None 时不传该字段, 让 L3 使用默认值.
        """
        from dy3_polaris.l3.api_models import LearnerProfile

        kp_mastery = {
            snap.kc_id: snap.to_l3_kp_mastery()
            for snap in self.mastery_snapshot
        }
        kwargs: dict[str, Any] = {
            "learner_id": self.user_id,
            "kp_mastery": kp_mastery,
            "weak_kps": self.get_weak_kcs(),
        }
        level = self._derive_level()
        if level is not None:
            kwargs["level"] = level
        preferred_style = self._derive_preferred_style()
        if preferred_style is not None:
            kwargs["preferred_style"] = preferred_style
        bloom_target = self._derive_bloom_target()
        if bloom_target is not None:
            kwargs["bloom_target"] = bloom_target
        return LearnerProfile(**kwargs)

    def _derive_level(self) -> str | None:
        """根据平均掌握度推导学习者水平."""
        if not self.mastery_snapshot:
            return None
        avg = sum(s.p_know for s in self.mastery_snapshot) / len(
            self.mastery_snapshot
        )
        if avg >= 0.7:
            return "advanced"
        elif avg >= 0.4:
            return "intermediate"
        return "beginner"

    def _derive_preferred_style(self) -> Any:
        """从 learning_style 推导 L3 LearningStyle."""
        if self.learning_style is None:
            return None
        from dy3_polaris.l3.api_models import LearningStyle
        style_map = {
            "visual": LearningStyle.VISUAL,
            "aural": LearningStyle.AUDITORY,
            "read_write": LearningStyle.READING,
            "kinesthetic": LearningStyle.KINESTHETIC,
        }
        primary = self.learning_style.primary_style
        return style_map.get(primary.value if hasattr(primary, 'value') else str(primary))

    def _derive_bloom_target(self) -> Any:
        """从 goals 推导 Bloom 目标层级."""
        if not self.goals:
            return None
        # 取优先级最高的目标的 bloom_level
        sorted_goals = sorted(
            self.goals, key=lambda g: g.priority, reverse=True
        )
        for goal in sorted_goals:
            if hasattr(goal, "bloom_level") and goal.bloom_level is not None:
                return goal.bloom_level
        return None

    @classmethod
    def from_l3_learner_profile(
        cls, profile: Any, session_id: str
    ) -> ContextEnvelope:
        """从 L3 LearnerProfile 逆向构建 ContextEnvelope."""
        mastery = [
            MasterySnapshot.from_l3_kp_mastery(kp)
            for kp in profile.kp_mastery.values()
        ]
        return cls(
            user_id=profile.learner_id,
            session_id=session_id,
            mastery_snapshot=mastery,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContextEnvelope:
        """从字典反序列化."""
        tc_raw = d.get("time_constraint")
        tc = (
            TimeConstraint.from_dict(tc_raw)
            if isinstance(tc_raw, dict)
            else tc_raw
        )
        # 延迟导入新字段类型
        clb_raw = d.get("cognitive_load_breakdown")
        ls_raw = d.get("learning_style")
        irt_raw = d.get("irt_ability")
        eng_raw = d.get("engagement")

        cognitive_load_breakdown = None
        learning_style = None
        irt_ability = None
        engagement = None

        if isinstance(clb_raw, dict):
            try:
                from dy3_polaris.l1.models import CognitiveLoadBreakdown
                cognitive_load_breakdown = CognitiveLoadBreakdown.from_dict(clb_raw)
            except (ImportError, KeyError):
                pass
        if isinstance(ls_raw, dict):
            try:
                from dy3_polaris.l1.models import VARKProfile
                learning_style = VARKProfile.from_dict(ls_raw)
            except (ImportError, KeyError):
                pass
        if isinstance(irt_raw, dict):
            try:
                from dy3_polaris.l1.models import IRTAbility
                irt_ability = IRTAbility.from_dict(irt_raw)
            except (ImportError, KeyError):
                pass
        if isinstance(eng_raw, dict):
            try:
                from dy3_polaris.l1.models import EngagementMetrics
                engagement = EngagementMetrics.from_dict(eng_raw)
            except (ImportError, KeyError):
                pass

        # mastery_trajectories (向后兼容: 缺失时返回空 dict)
        mt_raw = d.get("mastery_trajectories", {})
        mastery_trajectories: dict[str, Any] = {}
        if isinstance(mt_raw, dict):
            for k, v in mt_raw.items():
                if isinstance(v, dict):
                    try:
                        from dy3_polaris.l1.models import MasteryTrajectory
                        mastery_trajectories[k] = MasteryTrajectory.from_dict(v)
                    except (ImportError, KeyError):
                        pass

        return cls(
            user_id=d["user_id"],
            session_id=d["session_id"],
            envelope_id=d.get("envelope_id", f"env-{uuid.uuid4().hex[:12]}"),
            timestamp=d.get("timestamp", int(time.time() * 1000)),
            learning_state=LearningState.from_dict(
                d.get("learning_state", {})
            ),
            mastery_snapshot=[
                MasterySnapshot.from_dict(s)
                for s in d.get("mastery_snapshot", [])
            ],
            goals=[
                LearningGoal.from_dict(g)
                for g in d.get("goals", [])
            ],
            ttl=d.get("ttl", DEFAULT_TTL),
            resources=[
                ResourceItem.from_dict(r)
                for r in d.get("resources", [])
            ],
            time_constraint=tc,
            cognitive_load_breakdown=cognitive_load_breakdown,
            learning_style=learning_style,
            irt_ability=irt_ability,
            engagement=engagement,
            mastery_trajectories=mastery_trajectories,
        )


@dataclass
class LearningContext:
    """学习上下文持久化实体 (设计文档 ER 图 LearningContext 表).

    独立于会话的上下文实体, 支持跨会话的上下文恢复与刷新:
    - context_id: 上下文唯一 ID (自动生成, 前缀 "ctx-")
    - session_id: 关联会话 ID
    - user_id: 用户 ID
    - envelope: 上下文信封 (延迟初始化)
    - last_refreshed: 最近刷新时间戳 (毫秒)
    """

    session_id: str
    user_id: str
    context_id: str = field(default_factory=lambda: f"ctx-{uuid.uuid4().hex[:12]}")
    envelope: ContextEnvelope | None = field(default=None)  # type: ignore[assignment]
    last_refreshed: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        """延迟初始化 envelope (需要 user_id 和 session_id)."""
        if self.envelope is None:
            self.envelope = ContextEnvelope(
                user_id=self.user_id, session_id=self.session_id
            )

    def refresh(self) -> None:
        """刷新上下文: 更新时间戳并重新计算衰减系数."""
        self.last_refreshed = int(time.time() * 1000)
        self.envelope.refresh_decay(self.last_refreshed)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "context_id": self.context_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "envelope": self.envelope.to_dict() if self.envelope else None,
            "last_refreshed": self.last_refreshed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningContext:
        """从字典反序列化."""
        env_raw = d.get("envelope")
        env = (
            ContextEnvelope.from_dict(env_raw)
            if isinstance(env_raw, dict)
            else env_raw
        )
        return cls(
            session_id=d["session_id"],
            user_id=d["user_id"],
            context_id=d.get("context_id", f"ctx-{uuid.uuid4().hex[:12]}"),
            envelope=env,
            last_refreshed=d.get(
                "last_refreshed", int(time.time() * 1000)
            ),
        )


# ============================================================
# 7. 会话与 Fork 枚举与模型 (设计文档第五章)
# ============================================================


class SessionType(str, Enum):
    """会话类型 (设计文档 5.2).

    五种会话类型对应不同教学场景:
    - DIAGNOSIS: 学情诊断 — 评估学习者知识掌握状态
    - LEARNING: 知识学习 — 交互式知识学习与练习
    - LAB_GUIDE: 实验指导 — 实验操作步骤指导与安全提示
    - ASSESSMENT: 测评考核 — 正式测评与成绩记录
    - QUERY: 实时答疑 — 多智能体问答会话 (统一会话入口, 关联 L5 执行会话)
    """

    DIAGNOSIS = "diagnosis"
    LEARNING = "learning"
    LAB_GUIDE = "lab_guide"
    ASSESSMENT = "assessment"
    QUERY = "query"


class SessionStatus(str, Enum):
    """会话状态 (设计文档 5.1).

    - ACTIVE: 活跃 — 正在进行
    - PAUSED: 暂停 — 用户主动暂停或系统触发
    - FORKED: 已分叉 — 从此会话创建了 Fork 分支
    - COMPLETED: 已完成 — 正常结束并归档
    """

    ACTIVE = "active"
    PAUSED = "paused"
    FORKED = "forked"
    COMPLETED = "completed"


@dataclass
class AgentState:
    """Agent 推理状态快照 (设计文档 5.1).

    记录单个 Agent 在某时刻的运行状态:
    - agent_id: Agent 唯一标识
    - agent_type: Agent 类型 (如 "diagnosis", "review")
    - status: 运行状态 (如 "idle", "running", "completed")
    - output_summary: 输出摘要
    """

    agent_id: str
    agent_type: str
    status: str
    output_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "output_summary": self.output_summary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentState:
        """从字典反序列化."""
        return cls(
            agent_id=d["agent_id"],
            agent_type=d["agent_type"],
            status=d["status"],
            output_summary=d.get("output_summary", ""),
        )


@dataclass
class Interaction:
    """交互历史条目 (设计文档 5.1).

    记录一次完整的用户-Agent 交互:
    - interaction_type: 交互类型 (如 "qa", "quiz", "feedback")
    - content: 用户输入内容
    - user_role: 用户角色 (如 "student", "teacher")
    - response: Agent 响应内容 (可为 None)
    - agent_id: 响应 Agent ID (可为 None)
    - interaction_id: 交互唯一 ID (自动生成)
    - timestamp: 时间戳 (毫秒)
    """

    interaction_type: str
    content: str
    user_role: str = "student"
    response: str | None = None
    agent_id: str | None = None
    interaction_id: str = field(default_factory=lambda: f"int-{uuid.uuid4().hex[:12]}")
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "interaction_id": self.interaction_id,
            "interaction_type": self.interaction_type,
            "content": self.content,
            "user_role": self.user_role,
            "response": self.response,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Interaction:
        """从字典反序列化."""
        return cls(
            interaction_type=d["interaction_type"],
            content=d["content"],
            user_role=d.get("user_role", "student"),
            response=d.get("response"),
            agent_id=d.get("agent_id"),
            interaction_id=d.get(
                "interaction_id", f"int-{uuid.uuid4().hex[:12]}"
            ),
            timestamp=d.get("timestamp", int(time.time() * 1000)),
        )


@dataclass
class SessionArtifact:
    """会话产出物 (设计文档 5.1).

    记录会话中产生的知识卡片、测验、报告等产出物:
    - artifact_type: 产出物类型 (如 "knowledge_card", "quiz", "report")
    - title: 产出物标题
    - content: 产出物内容
    - confidence: 置信度 [0.0, 1.0]
    - artifact_id: 产出物唯一 ID (自动生成)
    - created_at: 创建时间戳 (毫秒)
    """

    artifact_type: str
    title: str
    content: str = ""
    confidence: float = 1.0
    artifact_id: str = field(default_factory=lambda: f"art-{uuid.uuid4().hex[:12]}")
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        """校验 confidence 在 [0.0, 1.0] 范围内."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "title": self.title,
            "content": self.content,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionArtifact:
        """从字典反序列化."""
        return cls(
            artifact_type=d["artifact_type"],
            title=d["title"],
            content=d.get("content", ""),
            confidence=d.get("confidence", 1.0),
            artifact_id=d.get(
                "artifact_id", f"art-{uuid.uuid4().hex[:12]}"
            ),
            created_at=d.get("created_at", int(time.time() * 1000)),
        )


@dataclass
class LearningSession:
    """学习会话模型 (设计文档 5.1).

    管理一次完整学习交互的生命周期:
    - session_id: 会话唯一 ID (自动生成, 前缀 "sess-")
    - user_id: 用户 ID
    - session_type: 会话类型
    - parent_session_id: 父会话 ID (Fork 时设置)
    - fork_point_seq: Fork 点序列号 (Fork 时设置)
    - context: 当前上下文信封
    - agent_states: Agent 状态字典 (agent_id → state)
    - interaction_log: 交互历史日志
    - artifacts: 关联产出物列表
    - status: 会话状态
    - checkpoint_indices: 检查点序列号列表
    - created_at: 创建时间戳 (毫秒)
    """

    user_id: str
    session_type: SessionType
    # 统一命名空间: 前缀 sess- (L1 用户会话, 唯一对外入口) — 见 shared/ids.py
    session_id: str = field(default_factory=lambda: _new_session_id("l1"))
    parent_session_id: str | None = None
    fork_point_seq: int | None = None
    context: ContextEnvelope = field(default=None)  # type: ignore[assignment]
    agent_states: dict[str, Any] = field(default_factory=dict)
    interaction_log: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    checkpoint_indices: list[int] = field(default_factory=list)
    #: 关联的 L5 Agent 执行会话 ID 列表 (统一会话入口: L1 会话聚合 L5 执行记录)
    agent_sessions: list[str] = field(default_factory=list)
    #: 会话内提问/交互次数
    question_count: int = 0
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        # 延迟初始化 context (需要 user_id 和 session_id)
        if self.context is None:
            self.context = ContextEnvelope(
                user_id=self.user_id, session_id=self.session_id
            )

    def attach_agent_session(self, agent_session_id: str) -> None:
        """关联一次 L5 Agent 执行会话 (统一会话闭环).

        更新 agent_sessions 列表 (去重)、提问计数与活跃时间。
        """
        if agent_session_id and agent_session_id not in self.agent_sessions:
            self.agent_sessions.append(agent_session_id)
        self.question_count += 1
        self.updated_at = int(time.time() * 1000)

    def add_checkpoint(self, seq: int) -> None:
        """添加检查点序列号."""
        self.checkpoint_indices.append(seq)

    def add_interaction(self, interaction: dict[str, Any]) -> None:
        """记录一次交互."""
        self.interaction_log.append(interaction)

    def add_artifact(self, artifact: dict[str, Any]) -> None:
        """关联一个产出物."""
        self.artifacts.append(artifact)

    def touch(self) -> None:
        """更新 updated_at 为当前时间."""
        self.updated_at = int(time.time() * 1000)

    def set_agent_state(self, agent_id: str, state: AgentState) -> None:
        """设置类型化 Agent 状态 (存储为字典)."""
        self.agent_states[agent_id] = state.to_dict()

    def get_agent_state(self, agent_id: str) -> AgentState | None:
        """获取类型化 Agent 状态 (从字典还原)."""
        raw = self.agent_states.get(agent_id)
        if raw is None:
            return None
        if isinstance(raw, dict):
            return AgentState.from_dict(raw)
        return raw

    def add_typed_interaction(self, interaction: Interaction) -> None:
        """添加类型化交互 (转换为字典后追加到 interaction_log)."""
        self.interaction_log.append(interaction.to_dict())

    def add_typed_artifact(self, artifact: SessionArtifact) -> None:
        """添加类型化产出物 (转换为字典后追加到 artifacts)."""
        self.artifacts.append(artifact.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "session_type": self.session_type.value,
            "parent_session_id": self.parent_session_id,
            "fork_point_seq": self.fork_point_seq,
            "context": self.context.to_dict() if self.context else None,
            "agent_states": self.agent_states,
            "interaction_log": self.interaction_log,
            "artifacts": self.artifacts,
            "status": self.status.value,
            "checkpoint_indices": self.checkpoint_indices,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningSession:
        """从字典反序列化."""
        ctx_raw = d.get("context")
        ctx = (
            ContextEnvelope.from_dict(ctx_raw)
            if isinstance(ctx_raw, dict)
            else ctx_raw
        )
        return cls(
            user_id=d["user_id"],
            session_type=SessionType(d.get("session_type", "learning")),
            session_id=d.get("session_id") or _new_session_id("l1"),
            parent_session_id=d.get("parent_session_id"),
            fork_point_seq=d.get("fork_point_seq"),
            context=ctx,
            agent_states=d.get("agent_states", {}),
            interaction_log=d.get("interaction_log", []),
            artifacts=d.get("artifacts", []),
            status=SessionStatus(d.get("status", "active")),
            checkpoint_indices=d.get("checkpoint_indices", []),
            created_at=d.get("created_at", int(time.time() * 1000)),
            updated_at=d.get("updated_at", int(time.time() * 1000)),
        )


@dataclass
class SessionFork:
    """Session Fork 数据结构 (设计文档 5.5).

    支持学习路径分支与对比:
    - fork_id: Fork 唯一 ID (自动生成, 前缀 "fork-")
    - source_session_id: 源会话 ID
    - fork_point_seq: Fork 点检查点序列号
    - fork_reason: Fork 原因 (如 "学生手动", "A/B测试")
    - branch_label: 分支标签 (如 "路径A-先理论")
    - snapshot_at_fork: Fork 点的上下文快照
    - merge_target: 合并目标会话 ID (未合并时为 None)
    - is_merged: 是否已合并回主会话
    """

    source_session_id: str
    fork_point_seq: int
    fork_reason: str
    branch_label: str
    snapshot_at_fork: ContextEnvelope
    fork_id: str = field(default_factory=lambda: f"fork-{uuid.uuid4().hex[:12]}")
    merge_target: str | None = None
    is_merged: bool = False
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        """校验 fork_point_seq >= 0."""
        if self.fork_point_seq < 0:
            raise ValueError(
                f"fork_point_seq must be >= 0, got {self.fork_point_seq}"
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "fork_id": self.fork_id,
            "source_session_id": self.source_session_id,
            "fork_point_seq": self.fork_point_seq,
            "fork_reason": self.fork_reason,
            "branch_label": self.branch_label,
            "snapshot_at_fork": self.snapshot_at_fork.to_dict(),
            "merge_target": self.merge_target,
            "is_merged": self.is_merged,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionFork:
        """从字典反序列化."""
        snap_raw = d.get("snapshot_at_fork")
        snap = (
            ContextEnvelope.from_dict(snap_raw)
            if isinstance(snap_raw, dict)
            else snap_raw
        )
        return cls(
            source_session_id=d["source_session_id"],
            fork_point_seq=d["fork_point_seq"],
            fork_reason=d["fork_reason"],
            branch_label=d["branch_label"],
            snapshot_at_fork=snap,
            fork_id=d.get("fork_id", f"fork-{uuid.uuid4().hex[:12]}"),
            merge_target=d.get("merge_target"),
            is_merged=d.get("is_merged", False),
            created_at=d.get("created_at", int(time.time() * 1000)),
        )


@dataclass
class SessionCheckpoint:
    """Session Checkpoint 类型化模型 (设计文档 5.3).

    支持会话状态的快照与恢复:
    - checkpoint_id: 检查点唯一 ID (自动生成, 前缀 "cp-")
    - session_id: 关联会话 ID
    - seq: 检查点序列号
    - agent_states: Agent 状态快照字典 (agent_id → state)
    - created_at: 创建时间戳 (毫秒)
    """

    session_id: str
    seq: int
    agent_states: dict[str, Any] = field(default_factory=dict)
    checkpoint_id: str = field(default_factory=lambda: f"cp-{uuid.uuid4().hex[:12]}")
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        """校验 seq >= 0."""
        if self.seq < 0:
            raise ValueError(
                f"seq must be >= 0, got {self.seq}"
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "seq": self.seq,
            "agent_states": self.agent_states,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionCheckpoint:
        """从字典反序列化."""
        return cls(
            session_id=d["session_id"],
            seq=d["seq"],
            agent_states=d.get("agent_states", {}),
            checkpoint_id=d.get(
                "checkpoint_id", f"cp-{uuid.uuid4().hex[:12]}"
            ),
            created_at=d.get("created_at", int(time.time() * 1000)),
        )


# ============================================================
# 8. 审计与脱敏枚举与模型 (设计文档第六章, 第七章)
# ============================================================


class DataLevel(str, Enum):
    """数据分级 (设计文档 6.1, GB/T 35273).

    四级分类, 公开度递减、敏感度递增:
    - L1_PUBLIC: 公开 — 可对外发布 (如课程公告)
    - L2_INTERNAL: 内部 — 机构内部可见 (如知识库内容)
    - L3_SENSITIVE: 敏感 — 个人学习数据 (如学情报告)
    - L4_CONFIDENTIAL: 机密 — 核心隐私数据 (如学号、成绩)
    """

    L1_PUBLIC = "l1_public"
    L2_INTERNAL = "l2_internal"
    L3_SENSITIVE = "l3_sensitive"
    L4_CONFIDENTIAL = "l4_confidential"


class AuditAction(str, Enum):
    """审计操作类型 (设计文档 7.4).

    9 种操作覆盖全链路可审计场景:
    - VIEW: 查看 (知识库浏览、学情报告查看)
    - EXPORT: 导出 (报告导出、数据下载)
    - MODIFY: 修改 (知识库编辑、配置变更)
    - DELETE: 删除 (数据删除、账户注销)
    - AGENT_INVOKE: Agent 调用 (诊断/生成/审核/导学)
    - APPROVE: 审批通过 (HiTL 确认)
    - REJECT: 审批驳回 (HiTL 拒绝)
    - LOGIN: 登录
    - LOGOUT: 登出
    """

    VIEW = "view"
    EXPORT = "export"
    MODIFY = "modify"
    DELETE = "delete"
    AGENT_INVOKE = "agent_invoke"
    APPROVE = "approve"
    REJECT = "reject"
    LOGIN = "login"
    LOGOUT = "logout"


class AuditResult(str, Enum):
    """审计结果 (设计文档 7.4).

    - SUCCESS: 操作成功
    - DENIED: 权限拒绝
    - ERROR: 系统错误
    """

    SUCCESS = "success"
    DENIED = "denied"
    ERROR = "error"


@dataclass
class AuditLogEntry:
    """审计日志条目 (设计文档 7.4).

    append-only 不可篡改的审计记录:
    - log_id: 日志唯一 ID (自动生成, 前缀 "audit-")
    - actor_id: 操作者 ID (学号或系统 ID)
    - actor_role: 操作者角色
    - action: 操作类型
    - target_resource: 目标资源 (如 "kb:dy3_energy_level")
    - target_data_level: 目标数据分级
    - purpose: 操作目的 (如 "学习查阅", "教学评估")
    - result: 操作结果
    - session_id: 关联会话 ID (可为 None)
    - ip_hash: IP 哈希 (脱敏, 前4后4, 可为 None)
    - timestamp: 时间戳 (毫秒)
    """

    actor_id: str
    actor_role: UserRole
    action: AuditAction
    target_resource: str
    target_data_level: DataLevel
    purpose: str
    result: AuditResult
    log_id: str = field(default_factory=lambda: f"audit-{uuid.uuid4().hex[:12]}")
    session_id: str | None = None
    ip_hash: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    provenance_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "log_id": self.log_id,
            "actor_id": self.actor_id,
            "actor_role": self.actor_role.value,
            "action": self.action.value,
            "target_resource": self.target_resource,
            "target_data_level": self.target_data_level.value,
            "purpose": self.purpose,
            "result": self.result.value,
            "session_id": self.session_id,
            "ip_hash": self.ip_hash,
            "timestamp": self.timestamp,
            "provenance_chain": self.provenance_chain,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditLogEntry:
        """从字典反序列化."""
        return cls(
            actor_id=d["actor_id"],
            actor_role=UserRole(d.get("actor_role", "undergrad")),
            action=AuditAction(d.get("action", "view")),
            target_resource=d["target_resource"],
            target_data_level=DataLevel(
                d.get("target_data_level", "l2_internal")
            ),
            purpose=d["purpose"],
            result=AuditResult(d.get("result", "success")),
            log_id=d.get("log_id", f"audit-{uuid.uuid4().hex[:12]}"),
            session_id=d.get("session_id"),
            ip_hash=d.get("ip_hash"),
            timestamp=d.get("timestamp", int(time.time() * 1000)),
            provenance_chain=d.get("provenance_chain", []),
        )


# ============================================================
# 9. HiTL 协同枚举 (设计文档第四章)
# ============================================================


class HiTLType(str, Enum):
    """HiTL 协同场景类型 (设计文档 4.1).

    四类协同场景覆盖教学全流程:
    - CONFIRMATION: 确认型 — 学生确认"已理解", 知识校验
    - CORRECTION: 纠错型 — 学生标记"不理解", 触发 Agent 自纠
    - CREATIVE: 创造型 — 教师创建内容, 审核 Agent 校验
    - EMERGENCY: 紧急干预 — 安全阻断, 自动暂停 + 通知教师
    """

    CONFIRMATION = "confirmation"
    CORRECTION = "correction"
    CREATIVE = "creative"
    EMERGENCY = "emergency"


class HiTLPriority(str, Enum):
    """HiTL 优先级 (设计文档第四章).

    - P0: 最高 — 紧急干预, 即时处理
    - P1: 高 — 安全相关问题, 优先处理
    - P2: 中 — 常规确认/纠错, 正常处理
    - P3: 低 — 非紧急反馈, 空闲处理
    """

    P0 = "p0"
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"


class ConfidenceGateResult(str, Enum):
    """置信度门控结果 (设计文档 4.2).

    三级门控决定 Agent 输出的呈现策略:
    - PASS: 置信度 >= 0.85, 直接呈现
    - WARNING: 0.4 <= 置信度 < 0.85, 附标签 + 建议人工复核
    - BLOCK: 置信度 < 0.4, 阻止呈现 + 触发审核流程
    """

    PASS = "pass"
    WARNING = "warning"
    BLOCK = "block"

    @classmethod
    def evaluate(cls, confidence: float) -> ConfidenceGateResult:
        """根据置信度评估门控结果.

        Args:
            confidence: Agent 输出置信度 [0.0, 1.0]

        Returns:
            - PASS: confidence >= WARNING_THRESHOLD (0.85)
            - WARNING: BLOCK_THRESHOLD (0.4) <= confidence < WARNING_THRESHOLD
            - BLOCK: confidence < BLOCK_THRESHOLD
        """
        if confidence >= WARNING_THRESHOLD:
            return cls.PASS
        elif confidence >= BLOCK_THRESHOLD:
            return cls.WARNING
        else:
            return cls.BLOCK


class FeedbackType(str, Enum):
    """反馈类型 (设计文档 7.5 feedback_options).

    四种学生反馈类型:
    - UNDERSTOOD: 已理解 — 内容质量良好
    - NEED_MORE: 需要更多 — 内容不够深入或不够详细
    - INCORRECT: 内容有误 — 事实性错误, 触发纠错流程
    - REPORT: 安全问题 — 举报不当内容, 升级处理
    """

    UNDERSTOOD = "understood"
    NEED_MORE = "need_more"
    INCORRECT = "incorrect"
    REPORT = "report"


class ApprovalDecision(str, Enum):
    """审批决策枚举 (设计文档 8.4, 三态决策).

    - APPROVE: 批准 — 内容通过审核
    - REJECT: 驳回 — 内容被拒绝
    - MODIFY: 修改 — 内容需修改后重新提交
    """

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


class FeedbackCategory(str, Enum):
    """反馈分类枚举 (设计文档 8.4).

    - FACTUAL: 事实性反馈 — 内容存在事实性错误
    - ADAPTIVE: 适应性反馈 — 内容不匹配学习者水平
    - SAFETY: 安全性反馈 — 内容存在安全问题
    """

    FACTUAL = "factual"
    ADAPTIVE = "adaptive"
    SAFETY = "safety"


class AlertType(str, Enum):
    """紧急警报类型枚举 (设计文档 4.2, 4.3).

    - HIGH_COGNITIVE_LOAD: 认知负荷过高
    - CONSECUTIVE_ERRORS: 连续错误过多
    - FAST_ANSWERING: 异常答题速度
    - BKT_DEVIATION: BKT 预测偏差过大
    """

    HIGH_COGNITIVE_LOAD = "high_cognitive_load"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    FAST_ANSWERING = "fast_answering"
    BKT_DEVIATION = "bkt_deviation"


# ============================================================
# 10. HiTL 数据模型 (设计文档第四章, 第八章 8.4)
# ============================================================


@dataclass
class ApprovalRequest:
    """HiTL 确认请求 (设计文档 4.1, 8.4).

    用于确认型和创造型协同场景:
    - request_id: 请求唯一 ID (自动生成, 前缀 "hitl-")
    - user_id: 请求目标用户 ID
    - session_id: 关联会话 ID
    - hitl_type: 协同场景类型
    - content: 需确认的内容
    - priority: 优先级
    - status: 请求状态 (pending / approved / rejected / expired)
    - created_at: 创建时间戳 (毫秒)
    """

    user_id: str
    session_id: str
    hitl_type: HiTLType
    content: str
    priority: HiTLPriority = HiTLPriority.P2
    request_id: str = field(default_factory=lambda: f"hitl-{uuid.uuid4().hex[:12]}")
    status: str = "pending"
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    confidence: float = 1.0
    deadline: int | None = None

    def __post_init__(self) -> None:
        """校验 confidence 在 [0.0, 1.0] 范围内."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def is_expired(self) -> bool:
        """检查请求是否已过期.

        若 deadline 为 None 则永不过期; 否则与当前时间比较.
        """
        if self.deadline is None:
            return False
        return int(time.time() * 1000) > self.deadline

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "hitl_type": self.hitl_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "status": self.status,
            "created_at": self.created_at,
            "confidence": self.confidence,
            "deadline": self.deadline,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ApprovalRequest:
        """从字典反序列化."""
        hitl_raw = d.get("hitl_type")
        hitl_type = (
            HiTLType(hitl_raw) if isinstance(hitl_raw, str) else hitl_raw
        )
        priority_raw = d.get("priority")
        priority = (
            HiTLPriority(priority_raw)
            if isinstance(priority_raw, str)
            else (priority_raw or HiTLPriority.P2)
        )
        return cls(
            user_id=d["user_id"],
            session_id=d["session_id"],
            hitl_type=hitl_type,
            content=d["content"],
            priority=priority,
            request_id=d.get(
                "request_id", f"hitl-{uuid.uuid4().hex[:12]}"
            ),
            status=d.get("status", "pending"),
            created_at=d.get("created_at", int(time.time() * 1000)),
            confidence=d.get("confidence", 1.0),
            deadline=d.get("deadline"),
        )


@dataclass
class ApprovalResponse:
    """HiTL 确认响应 (设计文档 4.1, 8.4 三态决策).

    - request_id: 关联的请求 ID
    - responder_id: 响应者 ID
    - approved: 是否批准 (向后兼容, 与 decision 同步)
    - comment: 附加评论
    - modifications: 修改建议列表 (MODIFY 决策时使用)
    - decision: 三态决策 (APPROVE / REJECT / MODIFY)
    - responded_at: 响应时间戳 (毫秒)
    """

    request_id: str
    responder_id: str
    approved: bool = True
    comment: str = ""
    modifications: list[dict[str, Any]] = field(default_factory=list)
    decision: ApprovalDecision | None = None
    responded_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        """同步 approved 与 decision (支持向后兼容)."""
        if self.decision is None:
            self.decision = (
                ApprovalDecision.APPROVE
                if self.approved
                else ApprovalDecision.REJECT
            )
        # 以 decision 为准同步 approved
        self.approved = self.decision == ApprovalDecision.APPROVE

    def is_approved(self) -> bool:
        """是否批准 (基于 decision 判断)."""
        return self.decision == ApprovalDecision.APPROVE

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "request_id": self.request_id,
            "responder_id": self.responder_id,
            "approved": self.approved,
            "comment": self.comment,
            "modifications": self.modifications,
            "decision": self.decision.value if self.decision else None,
            "responded_at": self.responded_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ApprovalResponse:
        """从字典反序列化."""
        decision_raw = d.get("decision")
        decision = (
            ApprovalDecision(decision_raw)
            if isinstance(decision_raw, str)
            else decision_raw
        )
        return cls(
            request_id=d["request_id"],
            responder_id=d["responder_id"],
            approved=d.get("approved", True),
            comment=d.get("comment", ""),
            modifications=d.get("modifications", []),
            decision=decision,
            responded_at=d.get(
                "responded_at", int(time.time() * 1000)
            ),
        )


@dataclass
class FeedbackReport:
    """HiTL 反馈报告 (设计文档 4.4).

    学生或教师提交的内容反馈:
    - report_id: 报告唯一 ID (自动生成, 前缀 "fb-")
    - user_id: 反馈提交者 ID
    - session_id: 关联会话 ID
    - feedback_type: 反馈类型
    - content: 反馈内容
    - artifact_id: 关联产出物 ID (可为 None)
    - created_at: 创建时间戳 (毫秒)
    """

    user_id: str
    session_id: str
    feedback_type: FeedbackType
    content: str
    artifact_id: str | None = None
    report_id: str = field(default_factory=lambda: f"fb-{uuid.uuid4().hex[:12]}")
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    severity: float = 0.0
    source_envelope_id: str | None = None

    def __post_init__(self) -> None:
        """校验 severity 在 [0.0, 1.0] 范围内."""
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(
                f"severity must be in [0.0, 1.0], got {self.severity}"
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "report_id": self.report_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "feedback_type": self.feedback_type.value,
            "content": self.content,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "severity": self.severity,
            "source_envelope_id": self.source_envelope_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeedbackReport:
        """从字典反序列化."""
        ft_raw = d.get("feedback_type")
        feedback_type = (
            FeedbackType(ft_raw) if isinstance(ft_raw, str) else ft_raw
        )
        return cls(
            user_id=d["user_id"],
            session_id=d["session_id"],
            feedback_type=feedback_type,
            content=d["content"],
            artifact_id=d.get("artifact_id"),
            report_id=d.get("report_id", f"fb-{uuid.uuid4().hex[:12]}"),
            created_at=d.get("created_at", int(time.time() * 1000)),
            severity=d.get("severity", 0.0),
            source_envelope_id=d.get("source_envelope_id"),
        )


@dataclass
class EmergencyAlert:
    """紧急干预警报 (设计文档 4.3).

    当检测到紧急情况时自动触发:
    - alert_id: 警报唯一 ID (自动生成, 前缀 "emg-")
    - session_id: 关联会话 ID
    - user_id: 用户 ID
    - trigger_reason: 触发原因 (如 "认知负荷过高", "连续错误>=10次")
    - trigger_value: 触发值 (如 0.97 或 10)
    - is_resolved: 是否已解决
    - created_at: 创建时间戳 (毫秒)

    触发条件:
    - 连续错误 >= 10 次
    - 认知负荷 >= 0.95 (EMERGENCY_THRESHOLD)
    - 异常答题速度 (< 5 秒/题)
    """

    session_id: str
    user_id: str
    trigger_reason: str
    trigger_value: float
    alert_id: str = field(default_factory=lambda: f"emg-{uuid.uuid4().hex[:12]}")
    is_resolved: bool = False
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    alert_type: AlertType | None = None
    cognitive_load: float = 0.0
    error_count: int = 0

    def __post_init__(self) -> None:
        """校验 cognitive_load 在 [0.0, 1.0] 范围内."""
        if not (0.0 <= self.cognitive_load <= 1.0):
            raise ValueError(
                f"cognitive_load must be in [0.0, 1.0], got {self.cognitive_load}"
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "alert_id": self.alert_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "trigger_reason": self.trigger_reason,
            "trigger_value": self.trigger_value,
            "is_resolved": self.is_resolved,
            "created_at": self.created_at,
            "alert_type": (
                self.alert_type.value if self.alert_type else None
            ),
            "cognitive_load": self.cognitive_load,
            "error_count": self.error_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EmergencyAlert:
        """从字典反序列化."""
        at_raw = d.get("alert_type")
        alert_type = (
            AlertType(at_raw) if isinstance(at_raw, str) else at_raw
        )
        return cls(
            session_id=d["session_id"],
            user_id=d["user_id"],
            trigger_reason=d["trigger_reason"],
            trigger_value=d["trigger_value"],
            alert_id=d.get("alert_id", f"emg-{uuid.uuid4().hex[:12]}"),
            is_resolved=d.get("is_resolved", False),
            created_at=d.get("created_at", int(time.time() * 1000)),
            alert_type=alert_type,
            cognitive_load=d.get("cognitive_load", 0.0),
            error_count=d.get("error_count", 0),
        )


@dataclass
class ProvenanceRecord:
    """溯源记录 (设计文档 8.1, PROV-O 模型).

    记录产出物的完整溯源链, 支持防篡改审计:
    - provenance_id: 溯源记录唯一 ID (自动生成, 前缀 "prov-")
    - artifact_id: 关联产出物 ID
    - actor_chain: 参与者链 (Agent ID 列表, 按执行顺序)
    - code_hash: 代码版本哈希 (如 "sha256:abc123")
    - env_hash: 环境版本哈希 (如 "sha256:def456")
    - created_at: 创建时间戳 (毫秒)
    """

    artifact_id: str
    actor_chain: list[str]
    code_hash: str
    env_hash: str
    provenance_id: str = field(default_factory=lambda: f"prov-{uuid.uuid4().hex[:12]}")
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "provenance_id": self.provenance_id,
            "artifact_id": self.artifact_id,
            "actor_chain": self.actor_chain,
            "code_hash": self.code_hash,
            "env_hash": self.env_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProvenanceRecord:
        """从字典反序列化."""
        return cls(
            artifact_id=d["artifact_id"],
            actor_chain=d.get("actor_chain", []),
            code_hash=d["code_hash"],
            env_hash=d["env_hash"],
            provenance_id=d.get(
                "provenance_id", f"prov-{uuid.uuid4().hex[:12]}"
            ),
            created_at=d.get("created_at", int(time.time() * 1000)),
        )


# ============================================================
# 11. FSRS 间隔重复调度器 (设计文档 3.4, FSRS-6 算法)
# ============================================================


@dataclass
class FSRSParameters:
    """FSRS 参数模型 — 21 参数权重 (w0-w20, FSRS-6).

    参考: open-spaced-repetition/fsrs4 全局参数集 (FSRS-6 扩展).
    w0-w3: 初始稳定性 S0(G)
    w4-w5: 初始难度 D0(G) = w4 - exp(w5*(G-1)) + 1
    w6-w7: 难度更新与均值回归
    w8-w10: 成功回忆稳定性更新
    w11-w14: 遗忘稳定性更新
    w15-w16: hard_penalty / easy_bonus
    w17-w19: 短期记忆模型 (same-day)
    w20: 衰减参数 DECAY = -w20
    """

    weights: list[float] = field(default_factory=lambda: [
        0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345,
        1.4604, 0.0046, 1.54575, 0.1192, 1.01925, 1.9395,
        0.11, 0.29605, 2.2698, 0.2315, 2.9898, 0.51655,
        0.6621, 0.8285, 0.12,  # w19=0.8285, w20=0.12
    ])
    request_retention: float = 0.9
    maximum_interval: int = 36500

    def __post_init__(self) -> None:
        if len(self.weights) < 4:
            raise ValueError(
                f"weights must have at least 4 elements (for initial_stability), "
                f"got {len(self.weights)}"
            )

    @property
    def decay(self) -> float:
        """FSRS-6 衰减参数 = -w20 (回退 -0.5 for FSRS-5)."""
        if len(self.weights) > 20:
            return -self.weights[20]
        return -0.5

    @property
    def factor(self) -> float:
        """FSRS-6 因子 = 0.9^(1/decay) - 1 (回退 19/81 for FSRS-5)."""
        decay = self.decay
        return 0.9 ** (1.0 / decay) - 1.0

    def initial_stability(self, grade: int) -> float:
        """S0(G) = w[G-1]."""
        if not (1 <= grade <= 4):
            raise ValueError(f"grade must be in [1, 4], got {grade}")
        return self.weights[grade - 1]

    def initial_difficulty(self, grade: int) -> float:
        """D0(G) = w4 - exp(w5*(G-1)) + 1, clamped to [1, 10]."""
        if not (1 <= grade <= 4):
            raise ValueError(f"grade must be in [1, 4], got {grade}")
        d = self.weights[4] - math.exp(self.weights[5] * (grade - 1)) + 1
        return max(1.0, min(10.0, d))

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "request_retention": self.request_retention,
            "maximum_interval": self.maximum_interval,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FSRSParameters:
        return cls(
            weights=d.get("weights", []),
            request_retention=d.get("request_retention", 0.9),
            maximum_interval=d.get("maximum_interval", 36500),
        )


@dataclass
class FSRSCardState:
    """FSRS 卡片状态 — 记忆稳定性/难度/可提取性/状态.

    状态常量 (类级属性, 非字段):
    - NEW: 新卡片, 尚未学习
    - LEARNING: 学习中, 短间隔复习
    - REVIEW: 复习中, 长间隔巩固
    - RELEARNING: 重新学习, 遗忘后重学
    - SUSPENDED: 暂停, 不参与调度
    """

    # 类级状态常量 (无类型标注, 不作为 dataclass 字段)
    NEW = "new"
    LEARNING = "learning"
    REVIEW = "review"
    RELEARNING = "relearning"
    SUSPENDED = "suspended"

    kc_id: str
    stability: float = 0.0
    difficulty: float = 0.0
    state: str = "new"
    reps: int = 0
    lapses: int = 0
    last_review_ts: int = 0

    def retrievability(
        self,
        current_ts: int,
        decay: float = -0.5,
        factor: float = 19.0 / 81.0,
    ) -> float:
        """可提取性 R = (1 + factor * t/S)^(-decay).

        Args:
            current_ts: 当前时间戳 (毫秒).
            decay: 衰减指数 (FSRS-6: -w20, 默认 -0.5).
            factor: 因子 (FSRS-6: 0.9^(1/decay)-1, 默认 19/81).
        """
        if self.state == self.NEW or self.stability == 0:
            return 0.0
        elapsed_days = max(
            0.0, (current_ts - self.last_review_ts) / (MS_PER_SEC * 86400)
        )
        r = (1 + factor * elapsed_days / self.stability) ** decay
        return max(0.0, min(1.0, r))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kc_id": self.kc_id,
            "stability": self.stability,
            "difficulty": self.difficulty,
            "state": self.state,
            "reps": self.reps,
            "lapses": self.lapses,
            "last_review_ts": self.last_review_ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FSRSCardState:
        return cls(
            kc_id=d["kc_id"],
            stability=d.get("stability", 0.0),
            difficulty=d.get("difficulty", 0.0),
            state=d.get("state", "new"),
            reps=d.get("reps", 0),
            lapses=d.get("lapses", 0),
            last_review_ts=d.get("last_review_ts", 0),
        )


@dataclass
class FSRSReviewLog:
    """FSRS 复习日志 — 记录每次复习的评分与状态转换."""

    kc_id: str
    grade: int
    elapsed_days: float
    state_before: str = "new"
    state_after: str = "new"
    review_id: str = field(default_factory=lambda: f"rev-{uuid.uuid4().hex[:12]}")
    reviewed_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not (1 <= self.grade <= 4):
            raise ValueError(
                f"grade must be in [1, 4], got {self.grade}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kc_id": self.kc_id,
            "grade": self.grade,
            "elapsed_days": self.elapsed_days,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "review_id": self.review_id,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FSRSReviewLog:
        return cls(
            kc_id=d["kc_id"],
            grade=d["grade"],
            elapsed_days=d["elapsed_days"],
            state_before=d.get("state_before", "new"),
            state_after=d.get("state_after", "new"),
            review_id=d.get("review_id", ""),
            reviewed_at=d.get("reviewed_at", 0),
        )


# ============================================================
# 12. IRT 项目反应理论 (1PL/2PL/3PL)
# ============================================================


class IRTModel(str, Enum):
    """IRT 模型类型."""

    ONE_PL = "1pl"  # Rasch: 仅 difficulty
    TWO_PL = "2pl"  # difficulty + discrimination
    THREE_PL = "3pl"  # difficulty + discrimination + guessing
    FOUR_PL = "4pl"  # difficulty + discrimination + guessing + upper asymptote


@dataclass
class IRTItem:
    """IRT 题目参数.

    - difficulty_b: 难度参数 [-3, +3]
    - discrimination_a: 区分度参数 (> 0)
    - guessing_c: 猜测参数 [0, 0.5] (下渐近线)
    - upper_d: 上渐近线 (0, 1], 默认 1.0 (4PL 专用)
    """

    item_id: str
    model_type: IRTModel
    difficulty_b: float
    discrimination_a: float = 1.0
    guessing_c: float = 0.0
    upper_d: float = 1.0

    def __post_init__(self) -> None:
        if not (-3.0 <= self.difficulty_b <= 3.0):
            raise ValueError(
                f"difficulty_b must be in [-3.0, 3.0], got {self.difficulty_b}"
            )
        if self.discrimination_a <= 0:
            raise ValueError(
                f"discrimination_a must be > 0, got {self.discrimination_a}"
            )
        if not (0.0 <= self.guessing_c <= 0.5):
            raise ValueError(
                f"guessing_c must be in [0.0, 0.5], got {self.guessing_c}"
            )
        if not (0.0 < self.upper_d <= 1.0):
            raise ValueError(
                f"upper_d must be in (0.0, 1.0], got {self.upper_d}"
            )
        # 1PL 固定 discrimination=1.0, guessing=0.0
        if self.model_type == IRTModel.ONE_PL:
            self.discrimination_a = 1.0
            self.guessing_c = 0.0
        # 2PL 固定 guessing=0.0
        elif self.model_type == IRTModel.TWO_PL:
            self.guessing_c = 0.0
        # 3PL / 4PL: 无强制约束

    def probability(self, theta: float) -> float:
        """P(θ) = c + (d - c) / (1 + e^(-a*(θ - b))) (4PL) 或 c + (1-c)/(1+e^(-z)) (3PL)."""
        z = self.discrimination_a * (theta - self.difficulty_b)
        if self.model_type == IRTModel.FOUR_PL:
            c = self.guessing_c
            d = self.upper_d
            p = c + (d - c) / (1 + math.exp(-z))
        else:
            p = self.guessing_c + (1 - self.guessing_c) / (1 + math.exp(-z))
        return max(0.0, min(1.0, p))

    def information(self, theta: float) -> float:
        """信息函数.

        - 2PL: I(θ) = a² · P · (1 - P)
        - 3PL (c>0): I(θ) = a² · (P-c)² · (1-P) / ((1-c)² · P)
        - 4PL: I(θ) = a² · (P-c)² · (d-P)² / ((d-c)² · P · (1-P))
        """
        p = self.probability(theta)
        p = max(1e-10, min(1 - 1e-10, p))
        if self.model_type == IRTModel.FOUR_PL:
            c = self.guessing_c
            d = self.upper_d
            denom = (d - c) ** 2 * p * (1 - p)
            if denom <= 0:
                return 0.0
            return self.discrimination_a ** 2 * (p - c) ** 2 * (d - p) ** 2 / denom
        if self.model_type == IRTModel.THREE_PL and self.guessing_c > 0:
            c = self.guessing_c
            denom = (1 - c) ** 2 * p
            if denom <= 0:
                return 0.0
            return (
                self.discrimination_a ** 2
                * (p - c) ** 2
                * (1 - p)
                / denom
            )
        return self.discrimination_a ** 2 * p * (1 - p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "model_type": self.model_type.value,
            "difficulty_b": self.difficulty_b,
            "discrimination_a": self.discrimination_a,
            "guessing_c": self.guessing_c,
            "upper_d": self.upper_d,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IRTItem:
        mt_raw = d.get("model_type", "1pl")
        model_type = IRTModel(mt_raw) if isinstance(mt_raw, str) else mt_raw
        return cls(
            item_id=d["item_id"],
            model_type=model_type,
            difficulty_b=d["difficulty_b"],
            discrimination_a=d.get("discrimination_a", 1.0),
            guessing_c=d.get("guessing_c", 0.0),
            upper_d=d.get("upper_d", 1.0),
        )


@dataclass
class IRTAbility:
    """IRT 能力参数.

    - theta: 能力值 [-3, +3]
    - standard_error: 测量标准误 (> 0)
    """

    user_id: str
    theta: float
    standard_error: float = 0.3

    def __post_init__(self) -> None:
        if not (-3.0 <= self.theta <= 3.0):
            raise ValueError(
                f"theta must be in [-3.0, 3.0], got {self.theta}"
            )

    def confidence_interval_95(self) -> tuple[float, float]:
        """95% 置信区间 = θ ± 1.96 * SE."""
        margin = 1.96 * self.standard_error
        return (self.theta - margin, self.theta + margin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "theta": self.theta,
            "standard_error": self.standard_error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IRTAbility:
        return cls(
            user_id=d["user_id"],
            theta=d["theta"],
            standard_error=d.get("standard_error", 0.3),
        )


# ============================================================
# 13. VARK 学习风格模型
# ============================================================


class VARKStyle(str, Enum):
    """VARK 学习风格模态."""

    VISUAL = "visual"
    AURAL = "aural"
    READ_WRITE = "read_write"
    KINESTHETIC = "kinesthetic"
    MULTIMODAL = "multimodal"


@dataclass
class VARKProfile:
    """VARK 学习风格画像.

    四维分数 [0.0, 1.0], 自动推导主导风格.
    """

    user_id: str
    visual_score: float = 0.0
    aural_score: float = 0.0
    read_write_score: float = 0.0
    kinesthetic_score: float = 0.0
    confidence: float = 0.0
    primary_style: VARKStyle = field(default=VARKStyle.MULTIMODAL, init=False)

    def __post_init__(self) -> None:
        for name, val in [
            ("visual_score", self.visual_score),
            ("aural_score", self.aural_score),
            ("read_write_score", self.read_write_score),
            ("kinesthetic_score", self.kinesthetic_score),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {val}"
                )
        self.primary_style = self._detect_primary_style()

    def _detect_primary_style(self) -> VARKStyle:
        """检测主导风格: 最高分, 若多个接近则 MULTIMODAL."""
        scores = {
            VARKStyle.VISUAL: self.visual_score,
            VARKStyle.AURAL: self.aural_score,
            VARKStyle.READ_WRITE: self.read_write_score,
            VARKStyle.KINESTHETIC: self.kinesthetic_score,
        }
        max_score = max(scores.values())
        if max_score == 0:
            return VARKStyle.MULTIMODAL
        # 检查是否有多个维度接近最高分 (差值 < 0.05)
        near_max = [s for s in scores.values() if max_score - s < 0.05]
        if len(near_max) >= 2:
            return VARKStyle.MULTIMODAL
        return max(scores, key=scores.get)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "visual_score": self.visual_score,
            "aural_score": self.aural_score,
            "read_write_score": self.read_write_score,
            "kinesthetic_score": self.kinesthetic_score,
            "confidence": self.confidence,
            "primary_style": self.primary_style.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VARKProfile:
        ps_raw = d.get("primary_style")
        # primary_style is computed in __post_init__, so we ignore it on restore
        obj = cls(
            user_id=d["user_id"],
            visual_score=d.get("visual_score", 0.0),
            aural_score=d.get("aural_score", 0.0),
            read_write_score=d.get("read_write_score", 0.0),
            kinesthetic_score=d.get("kinesthetic_score", 0.0),
            confidence=d.get("confidence", 0.0),
        )
        # Override if explicitly provided
        if isinstance(ps_raw, str):
            obj.primary_style = VARKStyle(ps_raw)
        return obj


@dataclass
class ContentModality:
    """内容模态标签 — 标记学习资源支持的 VARK 模态."""

    content_id: str
    modality_tags: list[VARKStyle] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_id": self.content_id,
            "modality_tags": [t.value for t in self.modality_tags],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContentModality:
        tags_raw = d.get("modality_tags", [])
        modality_tags = [
            VARKStyle(t) if isinstance(t, str) else t
            for t in tags_raw
        ]
        return cls(
            content_id=d["content_id"],
            modality_tags=modality_tags,
        )


# ============================================================
# 14. 认知负荷三分模型 (Sweller Cognitive Load Theory)
# ============================================================


@dataclass
class CognitiveLoadBreakdown:
    """认知负荷三分模型: ICL + ECL + GCL.

    - intrinsic_load: 内在负荷 (任务本身复杂度)
    - extraneous_load: 外在负荷 (呈现方式造成的额外负荷)
    - germane_load: 关联负荷 (图式建构的有效负荷)
    """

    intrinsic_load: float = 0.0
    extraneous_load: float = 0.0
    germane_load: float = 0.0

    def __post_init__(self) -> None:
        for name, val in [
            ("intrinsic_load", self.intrinsic_load),
            ("extraneous_load", self.extraneous_load),
            ("germane_load", self.germane_load),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {val}"
                )

    @property
    def total_load(self) -> float:
        """总负荷 = ICL + ECL + GCL."""
        return self.intrinsic_load + self.extraneous_load + self.germane_load

    def is_overloaded(self) -> bool:
        """总负荷 >= EMERGENCY_THRESHOLD 时标记为过载."""
        return self.total_load >= EMERGENCY_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "intrinsic_load": self.intrinsic_load,
            "extraneous_load": self.extraneous_load,
            "germane_load": self.germane_load,
            "total_load": self.total_load,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CognitiveLoadBreakdown:
        return cls(
            intrinsic_load=d.get("intrinsic_load", 0.0),
            extraneous_load=d.get("extraneous_load", 0.0),
            germane_load=d.get("germane_load", 0.0),
        )


@dataclass
class ElementInteractivity:
    """元素交互度 — 衡量知识元素间的依赖关系.

    interactivity_ratio = interaction_count / element_count
    """

    element_count: int = 0
    interaction_count: int = 0

    @property
    def interactivity_ratio(self) -> float:
        """交互度比率 = 交互数 / 元素数."""
        if self.element_count == 0:
            return 0.0
        return self.interaction_count / self.element_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_count": self.element_count,
            "interaction_count": self.interaction_count,
            "interactivity_ratio": self.interactivity_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ElementInteractivity:
        return cls(
            element_count=d.get("element_count", 0),
            interaction_count=d.get("interaction_count", 0),
        )


# ============================================================
# 15. Bloom 2D 分类法 (Anderson & Krathwohl 修订版)
# ============================================================


class KnowledgeType(str, Enum):
    """知识类型 (Anderson & Krathwohl 四类)."""

    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    METACOGNITIVE = "metacognitive"


@dataclass
class BloomTag:
    """Bloom 2D 标签: 认知层级 × 知识类型."""

    cognitive_level: Any  # L3 BloomLevel
    knowledge_type: KnowledgeType

    def matrix_cell(self) -> str:
        """2D 矩阵单元格标识 = (cognitive_level, knowledge_type)."""
        cl = (
            self.cognitive_level.value
            if hasattr(self.cognitive_level, "value")
            else str(self.cognitive_level)
        )
        return f"{cl}×{self.knowledge_type.value}"

    def to_dict(self) -> dict[str, Any]:
        cl = (
            self.cognitive_level.value
            if hasattr(self.cognitive_level, "value")
            else str(self.cognitive_level)
        )
        return {
            "cognitive_level": cl,
            "knowledge_type": self.knowledge_type.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BloomTag:
        kt_raw = d.get("knowledge_type", "factual")
        knowledge_type = (
            KnowledgeType(kt_raw) if isinstance(kt_raw, str) else kt_raw
        )

        cl_raw = d.get("cognitive_level", "remember")
        if isinstance(cl_raw, str):
            try:
                from dy3_polaris.l3.api_models import BloomLevel
                cognitive_level: Any = BloomLevel(cl_raw)
            except Exception:
                cognitive_level = cl_raw
        else:
            cognitive_level = cl_raw

        return cls(
            cognitive_level=cognitive_level,
            knowledge_type=knowledge_type,
        )


# ============================================================
# 16. 跨层接口数据结构 (设计文档第八章)
# ============================================================


@dataclass
class BKTUpdate:
    """BKT 参数更新 (L2 → L1)."""

    kc_id: str
    p_know: float
    p_slip: float = 0.1
    p_guess: float = 0.25
    p_transit: float = 0.1

    def __post_init__(self) -> None:
        for name, val in [
            ("p_know", self.p_know),
            ("p_slip", self.p_slip),
            ("p_guess", self.p_guess),
            ("p_transit", self.p_transit),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {val}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kc_id": self.kc_id,
            "p_know": self.p_know,
            "p_slip": self.p_slip,
            "p_guess": self.p_guess,
            "p_transit": self.p_transit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BKTUpdate:
        return cls(
            kc_id=d["kc_id"],
            p_know=d["p_know"],
            p_slip=d.get("p_slip", 0.1),
            p_guess=d.get("p_guess", 0.25),
            p_transit=d.get("p_transit", 0.1),
        )


@dataclass
class MemoryEntry:
    """学习记忆写入 (L1 → L2)."""

    session_id: str
    interaction_summary: str
    key_insights: list[str] = field(default_factory=list)
    weak_areas: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "interaction_summary": self.interaction_summary,
            "key_insights": self.key_insights,
            "weak_areas": self.weak_areas,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        return cls(
            session_id=d["session_id"],
            interaction_summary=d["interaction_summary"],
            key_insights=d.get("key_insights", []),
            weak_areas=d.get("weak_areas", []),
        )


@dataclass
class DecayRequest:
    """遗忘调度请求 (L1 → L2)."""

    user_id: str
    kcs_to_review: list[str] = field(default_factory=list)
    urgency_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "kcs_to_review": self.kcs_to_review,
            "urgency_scores": self.urgency_scores,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DecayRequest:
        return cls(
            user_id=d["user_id"],
            kcs_to_review=d.get("kcs_to_review", []),
            urgency_scores=d.get("urgency_scores", {}),
        )


@dataclass
class AccessCheck:
    """知识访问检查 (L1 → L3)."""

    user_id: str
    resource_id: str
    access_level: str = "read"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "access_level": self.access_level,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AccessCheck:
        return cls(
            user_id=d["user_id"],
            resource_id=d["resource_id"],
            access_level=d.get("access_level", "read"),
        )


@dataclass
class ResourceRequest:
    """资源推荐请求 (L1 → L3)."""

    weak_kcs: list[str] = field(default_factory=list)
    difficulty_range: tuple[float, float] = (0.3, 0.7)
    resource_types: list[str] = field(default_factory=list)
    count_limit: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "weak_kcs": self.weak_kcs,
            "difficulty_range": list(self.difficulty_range),
            "resource_types": self.resource_types,
            "count_limit": self.count_limit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResourceRequest:
        dr = d.get("difficulty_range", (0.3, 0.7))
        if isinstance(dr, list):
            dr = tuple(dr)
        return cls(
            weak_kcs=d.get("weak_kcs", []),
            difficulty_range=dr,
            resource_types=d.get("resource_types", []),
            count_limit=d.get("count_limit", 5),
        )


@dataclass
class KnowledgeResult:
    """知识查询结果 (L3 → L1)."""

    resources: list[Any] = field(default_factory=list)  # list[ResourceItem]
    confidence_scores: dict[str, float] = field(default_factory=dict)
    source_trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": [
                r.to_dict() if hasattr(r, "to_dict") else r
                for r in self.resources
            ],
            "confidence_scores": self.confidence_scores,
            "source_trace": self.source_trace,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeResult:
        resources_raw = d.get("resources", [])
        resources = []
        for r in resources_raw:
            if isinstance(r, dict):
                try:
                    resources.append(ResourceItem.from_dict(r))
                except (KeyError, TypeError):
                    resources.append(r)
            else:
                resources.append(r)
        return cls(
            resources=resources,
            confidence_scores=d.get("confidence_scores", {}),
            source_trace=d.get("source_trace", []),
        )


@dataclass
class PrivacyEvent:
    """隐私事件通知 (L1 → L0)."""

    event_type: str
    user_id: str
    data_level: DataLevel = DataLevel.L3_SENSITIVE
    detail: str = ""
    event_id: str = field(default_factory=lambda: f"pevt-{uuid.uuid4().hex[:12]}")
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "user_id": self.user_id,
            "data_level": self.data_level.value,
            "detail": self.detail,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PrivacyEvent:
        dl_raw = d.get("data_level", "l3_sensitive")
        data_level = (
            DataLevel(dl_raw) if isinstance(dl_raw, str) else dl_raw
        )
        return cls(
            event_type=d["event_type"],
            user_id=d["user_id"],
            data_level=data_level,
            detail=d.get("detail", ""),
            event_id=d.get("event_id", f"pevt-{uuid.uuid4().hex[:12]}"),
            timestamp=d.get("timestamp", int(time.time() * 1000)),
        )


@dataclass
class PolicyUpdate:
    """策略更新通知 (L0 → L1)."""

    policy_id: str
    version: str
    diff: dict[str, Any] = field(default_factory=dict)
    effective_at: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "diff": self.diff,
            "effective_at": self.effective_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyUpdate:
        return cls(
            policy_id=d["policy_id"],
            version=d["version"],
            diff=d.get("diff", {}),
            effective_at=d.get("effective_at", 0),
        )


# ============================================================
# 17. 隐私保护执行模型 (设计文档第六章)
# ============================================================


class DesensitizationMethod(str, Enum):
    """脱敏方法枚举."""

    HASH = "hash"
    AGGREGATE = "aggregate"
    BUCKET = "bucket"
    DP_NOISE = "dp_noise"
    PSEUDO_ID = "pseudo_id"


class RetentionPhase(str, Enum):
    """数据保留阶段枚举."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    ANONYMIZED = "anonymized"
    DELETED = "deleted"


@dataclass
class PrivacyConfig:
    """隐私配置模型.

    - k_anonymity: K-匿名最小组大小 (>= 1)
    - l_diversity: l-多样性最小值 (>= 1)
    - epsilon: 差分隐私 ε (> 0)
    - delta: 差分隐私 δ (>= 0)
    """

    k_anonymity: int = K_ANONYMITY_MIN
    l_diversity: int = L_DIVERSITY_MIN
    epsilon: float = 1.0
    delta: float = 1e-5
    quasi_identifiers: list[str] = field(default_factory=list)
    sensitive_attributes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be > 0, got {self.epsilon}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_anonymity": self.k_anonymity,
            "l_diversity": self.l_diversity,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "quasi_identifiers": self.quasi_identifiers,
            "sensitive_attributes": self.sensitive_attributes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PrivacyConfig:
        return cls(
            k_anonymity=d.get("k_anonymity", K_ANONYMITY_MIN),
            l_diversity=d.get("l_diversity", L_DIVERSITY_MIN),
            epsilon=d.get("epsilon", 1.0),
            delta=d.get("delta", 1e-5),
            quasi_identifiers=d.get("quasi_identifiers", []),
            sensitive_attributes=d.get("sensitive_attributes", []),
        )


@dataclass
class RetentionPolicy:
    """数据保留策略 — 分阶段保留与销毁."""

    data_level: str
    phases: list[tuple[Any, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_level": self.data_level,
            "phases": [
                (p.value if hasattr(p, "value") else str(p), days)
                for p, days in self.phases
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetentionPolicy:
        phases_raw = d.get("phases", [])
        phases = []
        for item in phases_raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                phase_raw, days = item
                phase = (
                    RetentionPhase(phase_raw)
                    if isinstance(phase_raw, str)
                    else phase_raw
                )
                phases.append((phase, days))
        return cls(
            data_level=d["data_level"],
            phases=phases,
        )


def desensitize_student_id(student_id: str, salt: str = "") -> str:
    """学号脱敏 — 使用 HMAC-SHA256 + salt 生成不可逆哈希.

    Args:
        student_id: 原始学号
        salt: 随机盐值

    Returns:
        脱敏后的哈希字符串 (十六进制)
    """
    import hashlib
    import hmac
    data = f"{salt}:{student_id}".encode("utf-8")
    return hmac.new(data, b"", hashlib.sha256).hexdigest()


def bucket_response_time(response_ms: int) -> str:
    """答题时间分桶.

    - fast: < 5 秒 (FAST_ANSWER_THRESHOLD_MS)
    - normal: 5-60 秒
    - slow: > 60 秒
    """
    if response_ms < FAST_ANSWER_THRESHOLD_MS:
        return "fast"
    elif response_ms <= 60_000:
        return "normal"
    else:
        return "slow"


# ============================================================
# 18. 学习分析事件 (xAPI / Caliper 兼容)
# ============================================================


@dataclass
class EventResult:
    """学习事件结果."""

    score_scaled: float = 0.0
    score_raw: int = 0
    score_max: int = 0
    success: bool = False
    completion: bool = False
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.score_scaled <= 1.0):
            raise ValueError(
                f"score_scaled must be in [0.0, 1.0], got {self.score_scaled}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_scaled": self.score_scaled,
            "score_raw": self.score_raw,
            "score_max": self.score_max,
            "success": self.success,
            "completion": self.completion,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventResult:
        return cls(
            score_scaled=d.get("score_scaled", 0.0),
            score_raw=d.get("score_raw", 0),
            score_max=d.get("score_max", 0),
            success=d.get("success", False),
            completion=d.get("completion", False),
            duration_ms=d.get("duration_ms", 0),
        )


@dataclass
class LearningEvent:
    """统一学习事件模型 (xAPI Actor-Verb-Object / Caliper Event)."""

    actor_id: str
    action: str
    object_id: str
    object_type: str = "resource"
    result: EventResult | None = None
    event_id: str = field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "action": self.action,
            "object_id": self.object_id,
            "object_type": self.object_type,
            "result": self.result.to_dict() if self.result else None,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningEvent:
        result_raw = d.get("result")
        result = (
            EventResult.from_dict(result_raw)
            if isinstance(result_raw, dict)
            else result_raw
        )
        return cls(
            actor_id=d["actor_id"],
            action=d["action"],
            object_id=d["object_id"],
            object_type=d.get("object_type", "resource"),
            result=result,
            event_id=d.get("event_id", f"evt-{uuid.uuid4().hex[:12]}"),
            timestamp=d.get("timestamp", int(time.time() * 1000)),
        )


# ============================================================
# 19. 参与度指标与会话分析
# ============================================================


class EngagementLevel(str, Enum):
    """参与度等级."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DISENGAGED = "disengaged"


@dataclass
class EngagementMetrics:
    """三维参与度指标 (行为/认知/情感).

    行为维度: session_duration_ms, login_frequency, completion_rate
    认知维度: accuracy_rate, avg_response_time_ms
    情感维度: sentiment_score, hint_usage_count (反向)
    """

    session_duration_ms: int = 0
    login_frequency: int = 0
    completion_rate: float = 0.0
    accuracy_rate: float = 0.0
    avg_response_time_ms: int = 0
    sentiment_score: float = 0.0
    hint_usage_count: int = 0

    def composite_score(self) -> float:
        """综合参与度 = 加权平均(行为+认知+情感)."""
        # 行为维度 (权重 0.4)
        behavioral = (
            min(1.0, self.session_duration_ms / 3_600_000) * 0.3
            + min(1.0, self.login_frequency / 7) * 0.3
            + self.completion_rate * 0.4
        )
        # 认知维度 (权重 0.35)
        cognitive = (
            self.accuracy_rate * 0.6
            + max(0.0, 1.0 - self.avg_response_time_ms / 60_000) * 0.4
        )
        # 情感维度 (权重 0.25)
        emotional = (
            max(0.0, self.sentiment_score) * 0.7
            + max(0.0, 1.0 - self.hint_usage_count / 10) * 0.3
        )
        return max(0.0, min(1.0, behavioral * 0.4 + cognitive * 0.35 + emotional * 0.25))

    def classify_level(self) -> EngagementLevel:
        """根据综合得分分类参与度等级."""
        score = self.composite_score()
        if score >= 0.7:
            return EngagementLevel.HIGH
        elif score >= 0.4:
            return EngagementLevel.MEDIUM
        elif score >= 0.2:
            return EngagementLevel.LOW
        return EngagementLevel.DISENGAGED

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_duration_ms": self.session_duration_ms,
            "login_frequency": self.login_frequency,
            "completion_rate": self.completion_rate,
            "accuracy_rate": self.accuracy_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
            "sentiment_score": self.sentiment_score,
            "hint_usage_count": self.hint_usage_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngagementMetrics:
        return cls(
            session_duration_ms=d.get("session_duration_ms", 0),
            login_frequency=d.get("login_frequency", 0),
            completion_rate=d.get("completion_rate", 0.0),
            accuracy_rate=d.get("accuracy_rate", 0.0),
            avg_response_time_ms=d.get("avg_response_time_ms", 0),
            sentiment_score=d.get("sentiment_score", 0.0),
            hint_usage_count=d.get("hint_usage_count", 0),
        )


@dataclass
class SessionAnalytics:
    """会话聚合分析."""

    session_id: str
    total_duration_ms: int = 0
    total_interactions: int = 0
    total_questions: int = 0
    correct_answers: int = 0
    mastery_delta: float = 0.0
    engagement_score: float = 0.0

    @property
    def accuracy_rate(self) -> float:
        """正确率 = correct_answers / total_questions."""
        if self.total_questions == 0:
            return 0.0
        return self.correct_answers / self.total_questions

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_duration_ms": self.total_duration_ms,
            "total_interactions": self.total_interactions,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "mastery_delta": self.mastery_delta,
            "engagement_score": self.engagement_score,
            "accuracy_rate": self.accuracy_rate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionAnalytics:
        return cls(
            session_id=d["session_id"],
            total_duration_ms=d.get("total_duration_ms", 0),
            total_interactions=d.get("total_interactions", 0),
            total_questions=d.get("total_questions", 0),
            correct_answers=d.get("correct_answers", 0),
            mastery_delta=d.get("mastery_delta", 0.0),
            engagement_score=d.get("engagement_score", 0.0),
        )


# ============================================================
# 20. 学习路径数据结构
# ============================================================


@dataclass
class PathNode:
    """学习路径节点."""

    kc_id: str
    order: int
    estimated_difficulty: float = 0.5
    prerequisite_kcs: list[str] = field(default_factory=list)
    estimated_time_minutes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kc_id": self.kc_id,
            "order": self.order,
            "estimated_difficulty": self.estimated_difficulty,
            "prerequisite_kcs": self.prerequisite_kcs,
            "estimated_time_minutes": self.estimated_time_minutes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PathNode:
        return cls(
            kc_id=d["kc_id"],
            order=d["order"],
            estimated_difficulty=d.get("estimated_difficulty", 0.5),
            prerequisite_kcs=d.get("prerequisite_kcs", []),
            estimated_time_minutes=d.get("estimated_time_minutes", 0),
        )


@dataclass
class LearningPath:
    """学习路径."""

    user_id: str
    nodes: list[PathNode] = field(default_factory=list)
    path_id: str = field(default_factory=lambda: f"path-{uuid.uuid4().hex[:12]}")
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def total_estimated_time(self) -> int:
        """总预估时间 (分钟)."""
        return sum(n.estimated_time_minutes for n in self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "user_id": self.user_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningPath:
        return cls(
            user_id=d["user_id"],
            nodes=[PathNode.from_dict(n) for n in d.get("nodes", [])],
            path_id=d.get("path_id", f"path-{uuid.uuid4().hex[:12]}"),
            created_at=d.get("created_at", int(time.time() * 1000)),
        )


@dataclass
class PathRecommendation:
    """路径推荐."""

    user_id: str
    recommended_path_id: str
    rationale: str
    predicted_mastery_gain: float = 0.0
    confidence: float = 1.0
    recommendation_id: str = field(default_factory=lambda: f"rec-{uuid.uuid4().hex[:12]}")
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "user_id": self.user_id,
            "recommended_path_id": self.recommended_path_id,
            "rationale": self.rationale,
            "predicted_mastery_gain": self.predicted_mastery_gain,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PathRecommendation:
        return cls(
            user_id=d["user_id"],
            recommended_path_id=d["recommended_path_id"],
            rationale=d["rationale"],
            predicted_mastery_gain=d.get("predicted_mastery_gain", 0.0),
            confidence=d.get("confidence", 1.0),
            recommendation_id=d.get("recommendation_id", f"rec-{uuid.uuid4().hex[:12]}"),
            created_at=d.get("created_at", int(time.time() * 1000)),
        )

# ============================================================
# 21. 最近发展区 ZPD (Vygotsky Zone of Proximal Development)
# ============================================================


@dataclass
class ZoneOfProximalDevelopment:
    """维果茨基最近发展区 — 定义学习者最优难度范围.

    ZPD 是学习者当前能力上下的一段区间, 在此区间内的任务
    既不过于简单 (已掌握), 也不过于困难 (超出能力),
    是学习效率最高的区域.

    - learner_theta: 学习者 IRT 能力参数 θ [-3, +3]
    - zpd_lower: ZPD 下界 = θ - delta_lower
    - zpd_upper: ZPD 上界 = θ + delta_upper
    - delta_lower: 下界偏移量 (默认 0.5)
    - delta_upper: 上界偏移量 (默认 0.5)
    """

    learner_theta: float
    delta_lower: float = 0.5
    delta_upper: float = 0.5
    zpd_lower: float = field(default=0.0, init=False)
    zpd_upper: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not (-3.0 <= self.learner_theta <= 3.0):
            raise ValueError(
                f"learner_theta must be in [-3.0, 3.0], got {self.learner_theta}"
            )
        if self.delta_lower <= 0:
            raise ValueError(
                f"delta_lower must be > 0, got {self.delta_lower}"
            )
        if self.delta_upper <= 0:
            raise ValueError(
                f"delta_upper must be > 0, got {self.delta_upper}"
            )
        self.zpd_lower = max(-3.0, self.learner_theta - self.delta_lower)
        self.zpd_upper = min(3.0, self.learner_theta + self.delta_upper)

    def is_in_zpd(self, difficulty: float) -> bool:
        """检查题目难度是否落在 ZPD 内."""
        return self.zpd_lower <= difficulty <= self.zpd_upper

    def recommended_difficulty(self) -> float:
        """推荐难度 = ZPD 中点."""
        return (self.zpd_lower + self.zpd_upper) / 2.0

    def adjustment_direction(self, current_difficulty: float) -> str:
        """根据当前难度给出调整方向.

        Returns:
            "increase" / "decrease" / "optimal"
        """
        if current_difficulty < self.zpd_lower:
            return "increase"
        elif current_difficulty > self.zpd_upper:
            return "decrease"
        return "optimal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_theta": self.learner_theta,
            "delta_lower": self.delta_lower,
            "delta_upper": self.delta_upper,
            "zpd_lower": self.zpd_lower,
            "zpd_upper": self.zpd_upper,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ZoneOfProximalDevelopment:
        return cls(
            learner_theta=d["learner_theta"],
            delta_lower=d.get("delta_lower", 0.5),
            delta_upper=d.get("delta_upper", 0.5),
        )


# ============================================================
# 22. 知识点元数据 (Knowledge Component)
# ============================================================


@dataclass
class KnowledgeComponent:
    """知识点元数据 — L1 层轻量级 KC 描述.

    对齐 L3 KPMastery.kp_id, 补充 L1 所需的元数据:
    - 名称、Bloom 标签、预估难度、前置知识点、预估学习时间
    - 用于学习路径规划、资源推荐、ZPD 匹配
    """

    kc_id: str
    name: str
    bloom_tag: BloomTag | None = None
    estimated_difficulty: float = 0.5
    prerequisite_kcs: list[str] = field(default_factory=list)
    estimated_time_minutes: int = 0
    kc_description: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.estimated_difficulty <= 1.0):
            raise ValueError(
                f"estimated_difficulty must be in [0.0, 1.0], "
                f"got {self.estimated_difficulty}"
            )
        if self.estimated_time_minutes < 0:
            raise ValueError(
                f"estimated_time_minutes must be >= 0, "
                f"got {self.estimated_time_minutes}"
            )

    def has_prerequisites(self) -> bool:
        """是否存在前置知识点."""
        return len(self.prerequisite_kcs) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kc_id": self.kc_id,
            "name": self.name,
            "bloom_tag": self.bloom_tag.to_dict() if self.bloom_tag else None,
            "estimated_difficulty": self.estimated_difficulty,
            "prerequisite_kcs": self.prerequisite_kcs,
            "estimated_time_minutes": self.estimated_time_minutes,
            "kc_description": self.kc_description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeComponent:
        bt_raw = d.get("bloom_tag")
        bloom_tag = BloomTag.from_dict(bt_raw) if bt_raw else None
        return cls(
            kc_id=d["kc_id"],
            name=d["name"],
            bloom_tag=bloom_tag,
            estimated_difficulty=d.get("estimated_difficulty", 0.5),
            prerequisite_kcs=d.get("prerequisite_kcs", []),
            estimated_time_minutes=d.get("estimated_time_minutes", 0),
            kc_description=d.get("kc_description", ""),
        )


# ============================================================
# 23. 掌握度轨迹 (Mastery Trajectory)
# ============================================================


@dataclass
class MasteryTrajectoryPoint:
    """掌握度轨迹点 — 时间序列中的单个数据点.

    记录某一时刻的掌握状态, 用于趋势分析和预测.
    """

    kc_id: str
    timestamp: int
    p_know: float
    decay_factor: float = 1.0
    interaction_type: str = "practice"

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_know <= 1.0):
            raise ValueError(
                f"p_know must be in [0.0, 1.0], got {self.p_know}"
            )
        if not (0.0 <= self.decay_factor <= 1.0):
            raise ValueError(
                f"decay_factor must be in [0.0, 1.0], got {self.decay_factor}"
            )

    def effective_mastery(self) -> float:
        """有效掌握度 = p_know * decay_factor."""
        return self.p_know * self.decay_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "kc_id": self.kc_id,
            "timestamp": self.timestamp,
            "p_know": self.p_know,
            "decay_factor": self.decay_factor,
            "interaction_type": self.interaction_type,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MasteryTrajectoryPoint:
        return cls(
            kc_id=d["kc_id"],
            timestamp=d["timestamp"],
            p_know=d["p_know"],
            decay_factor=d.get("decay_factor", 1.0),
            interaction_type=d.get("interaction_type", "practice"),
        )


@dataclass
class MasteryTrajectory:
    """掌握度轨迹 — 按 KC 组织的时间序列.

    记录某个 KC 的掌握度变化历史, 支持:
    - 趋势检测 (improving / stable / declining)
    - 最近状态查询
    - 变化率计算
    """

    kc_id: str
    points: list[MasteryTrajectoryPoint] = field(default_factory=list)

    def add_point(self, point: MasteryTrajectoryPoint) -> None:
        """添加轨迹点 (自动按时间戳排序)."""
        self.points.append(point)
        self.points.sort(key=lambda p: p.timestamp)

    def latest(self) -> MasteryTrajectoryPoint | None:
        """获取最新的轨迹点."""
        if not self.points:
            return None
        return self.points[-1]

    def earliest(self) -> MasteryTrajectoryPoint | None:
        """获取最早的轨迹点."""
        if not self.points:
            return None
        return self.points[0]

    def trend(self) -> str:
        """检测掌握度趋势.

        Returns:
            "improving" / "stable" / "declining" / "insufficient_data"
        """
        if len(self.points) < 2:
            return "insufficient_data"
        recent = self.points[-1].p_know
        previous = self.points[-2].p_know
        delta = recent - previous
        if delta > 0.05:
            return "improving"
        elif delta < -0.05:
            return "declining"
        return "stable"

    def mastery_delta(self) -> float:
        """计算首末掌握度变化量.

        Returns:
            Δp_know = latest - earliest; 若不足 2 点则返回 0.0
        """
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].p_know - self.points[0].p_know

    def to_dict(self) -> dict[str, Any]:
        return {
            "kc_id": self.kc_id,
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MasteryTrajectory:
        points = [
            MasteryTrajectoryPoint.from_dict(p)
            for p in d.get("points", [])
        ]
        return cls(
            kc_id=d["kc_id"],
            points=points,
        )


# ============================================================
# 24. 学习计划 (Study Plan)
# ============================================================


@dataclass
class StudyBlock:
    """学习时间块 — 计划中的单个学习时段.

    将学习路径节点映射到具体的时间段.
    """

    kc_id: str
    start_time: int  # Unix 毫秒
    duration_minutes: int
    phase: LearningPhase = LearningPhase.PRACTICE
    block_id: str = field(default_factory=lambda: f"block-{uuid.uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if self.duration_minutes <= 0:
            raise ValueError(
                f"duration_minutes must be > 0, got {self.duration_minutes}"
            )

    def end_time(self) -> int:
        """结束时间 = 开始时间 + 时长."""
        return self.start_time + self.duration_minutes * MS_PER_SEC * 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "kc_id": self.kc_id,
            "start_time": self.start_time,
            "duration_minutes": self.duration_minutes,
            "phase": self.phase.value,
            "block_id": self.block_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StudyBlock:
        phase_raw = d.get("phase", "practice")
        phase = LearningPhase(phase_raw) if isinstance(phase_raw, str) else phase_raw
        return cls(
            kc_id=d["kc_id"],
            start_time=d["start_time"],
            duration_minutes=d["duration_minutes"],
            phase=phase,
            block_id=d.get("block_id", f"block-{uuid.uuid4().hex[:8]}"),
        )


@dataclass
class StudyPlan:
    """学习计划 — 结合学习路径与时间约束的具体安排.

    将 LearningPath + TimeConstraint + LearningGoals 聚合为
    可执行的学习日程, 包含一组 StudyBlock.
    """

    user_id: str
    blocks: list[StudyBlock] = field(default_factory=list)
    plan_id: str = field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:12]}")
    goals: list[LearningGoal] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    total_estimated_minutes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.total_estimated_minutes = sum(
            b.duration_minutes for b in self.blocks
        )

    def add_block(self, block: StudyBlock) -> None:
        """添加学习块并重新计算总时长."""
        self.blocks.append(block)
        self.total_estimated_minutes = sum(
            b.duration_minutes for b in self.blocks
        )

    def block_count(self) -> int:
        """学习块数量."""
        return len(self.blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "plan_id": self.plan_id,
            "blocks": [b.to_dict() for b in self.blocks],
            "goals": [g.to_dict() for g in self.goals],
            "created_at": self.created_at,
            "total_estimated_minutes": self.total_estimated_minutes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StudyPlan:
        blocks = [StudyBlock.from_dict(b) for b in d.get("blocks", [])]
        goals = [LearningGoal.from_dict(g) for g in d.get("goals", [])]
        return cls(
            user_id=d["user_id"],
            blocks=blocks,
            plan_id=d.get("plan_id", f"plan-{uuid.uuid4().hex[:12]}"),
            goals=goals,
            created_at=d.get("created_at", int(time.time() * 1000)),
        )


# ============================================================
# 25. 学习效率指标 (Learning Efficiency)
# ============================================================


@dataclass
class LearningEfficiency:
    """学习效率指标 — 衡量单位投入的掌握度提升.

    核心公式:
    - efficiency = mastery_gain / time_spent_hours
    - interaction_efficiency = mastery_gain / interactions

    用于:
    - 评估学习策略有效性
    - 比较不同学习阶段的效率
    - 为导学决策 Agent 提供数据支撑
    """

    mastery_gain: float = 0.0
    time_spent_ms: int = 0
    interactions: int = 0
    kc_id: str = ""
    session_id: str = ""
    measured_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if self.time_spent_ms < 0:
            raise ValueError(
                f"time_spent_ms must be >= 0, got {self.time_spent_ms}"
            )
        if self.interactions < 0:
            raise ValueError(
                f"interactions must be >= 0, got {self.interactions}"
            )

    def time_efficiency(self) -> float:
        """时间效率 = 掌握度提升 / 小时数.

        Returns:
            每小时掌握度提升量; 若时间为 0 则返回 0.0
        """
        if self.time_spent_ms == 0:
            return 0.0
        hours = self.time_spent_ms / MS_PER_HOUR
        return self.mastery_gain / hours

    def interaction_efficiency(self) -> float:
        """交互效率 = 掌握度提升 / 交互次数.

        Returns:
            每次交互的掌握度提升量; 若交互数为 0 则返回 0.0
        """
        if self.interactions == 0:
            return 0.0
        return self.mastery_gain / self.interactions

    def efficiency_rating(self) -> str:
        """效率等级评定.

        基于时间效率的综合评级:
        - "high": time_efficiency > 0.3 (每小时提升 > 30%)
        - "medium": 0.1 <= time_efficiency <= 0.3
        - "low": time_efficiency < 0.1
        """
        te = self.time_efficiency()
        if te > 0.3:
            return "high"
        elif te >= 0.1:
            return "medium"
        return "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mastery_gain": self.mastery_gain,
            "time_spent_ms": self.time_spent_ms,
            "interactions": self.interactions,
            "kc_id": self.kc_id,
            "session_id": self.session_id,
            "measured_at": self.measured_at,
            "time_efficiency": self.time_efficiency(),
            "interaction_efficiency": self.interaction_efficiency(),
            "efficiency_rating": self.efficiency_rating(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningEfficiency:
        return cls(
            mastery_gain=d.get("mastery_gain", 0.0),
            time_spent_ms=d.get("time_spent_ms", 0),
            interactions=d.get("interactions", 0),
            kc_id=d.get("kc_id", ""),
            session_id=d.get("session_id", ""),
            measured_at=d.get("measured_at", int(time.time() * 1000)),
        )


__all__ = [
    # 常量
    "MS_PER_HOUR",
    "MS_PER_SEC",
    "MIN_STABILITY",
    "STABILITY_GAIN",
    "PRIOR_PROB",
    "MIN_DECAY",
    "DEFAULT_REPS",
    "PRIORITY_NORMAL",
    "DEFAULT_SESSION_MS",
    "DEFAULT_INTERACTIONS",
    "DEFAULT_COGNITIVE_LOAD",
    "DEFAULT_TTL",
    "WEAK_THRESHOLD",
    "EMERGENCY_THRESHOLD",
    "BLOCK_THRESHOLD",
    "WARNING_THRESHOLD",
    "MAX_DAILY_AGENT_CALLS",
    "K_ANONYMITY_MIN",
    "L_DIVERSITY_MIN",
    "COGNITIVE_LOAD_RECALC_INTERVAL",
    "FAST_ANSWER_THRESHOLD_MS",
    "CONSECUTIVE_ERROR_THRESHOLD",
    "BKT_DEVIATION_THRESHOLD",
    # 衰减函数
    "calculate_decay",
    # BKT 模型
    "BKTParams",
    # 角色与权限枚举
    "UserRole",
    "UserStatus",
    "Permission",
    # 角色模型
    "Role",
    # ABAC 枚举
    "GradeLevel",
    "MajorDirection",
    "LabAccessTier",
    # ABAC 模型
    "ABACAttributes",
    "User",
    # 学习上下文枚举
    "LearningPhase",
    # 学习上下文模型
    "MasterySnapshot",
    "LearningGoal",
    "LearningState",
    "ResourceItem",
    "TimeConstraint",
    "ContextEnvelope",
    "LearningContext",
    # 会话枚举
    "SessionType",
    "SessionStatus",
    # 会话组件模型
    "AgentState",
    "Interaction",
    "SessionArtifact",
    # 会话模型
    "LearningSession",
    "SessionFork",
    "SessionCheckpoint",
    # 审计枚举
    "DataLevel",
    "AuditAction",
    "AuditResult",
    # 审计模型
    "AuditLogEntry",
    # HiTL 枚举
    "HiTLType",
    "HiTLPriority",
    "ConfidenceGateResult",
    "FeedbackType",
    "ApprovalDecision",
    "FeedbackCategory",
    "AlertType",
    # HiTL 模型
    "ApprovalRequest",
    "ApprovalResponse",
    "FeedbackReport",
    "EmergencyAlert",
    # 溯源模型
    "ProvenanceRecord",
    # FSRS 间隔重复调度器
    "FSRSParameters",
    "FSRSCardState",
    "FSRSReviewLog",
    # IRT 项目反应理论
    "IRTModel",
    "IRTItem",
    "IRTAbility",
    # VARK 学习风格
    "VARKStyle",
    "VARKProfile",
    "ContentModality",
    # 认知负荷三分模型
    "CognitiveLoadBreakdown",
    "ElementInteractivity",
    # Bloom 2D 分类法
    "KnowledgeType",
    "BloomTag",
    # 跨层接口数据结构
    "BKTUpdate",
    "MemoryEntry",
    "DecayRequest",
    "AccessCheck",
    "ResourceRequest",
    "KnowledgeResult",
    "PrivacyEvent",
    "PolicyUpdate",
    # 隐私保护执行模型
    "DesensitizationMethod",
    "RetentionPhase",
    "PrivacyConfig",
    "RetentionPolicy",
    "desensitize_student_id",
    "bucket_response_time",
    # 学习分析事件
    "EventResult",
    "LearningEvent",
    # 参与度指标与会话分析
    "EngagementLevel",
    "EngagementMetrics",
    "SessionAnalytics",
    # 学习路径数据结构
    "PathNode",
    "LearningPath",
    "PathRecommendation",
    # 最近发展区
    "ZoneOfProximalDevelopment",
    # 知识点元数据
    "KnowledgeComponent",
    # 掌握度轨迹
    "MasteryTrajectoryPoint",
    "MasteryTrajectory",
    # 学习计划
    "StudyBlock",
    "StudyPlan",
    # 学习效率指标
    "LearningEfficiency",
]
