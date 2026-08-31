"""L1 用户域隐私治理 (Privacy Governance) — 核心引擎.

设计依据:
- L1 设计文档第六章: 隐私保护与数据治理 (6.1-6.6)
- L1 设计文档第七章 7.2: API `/api/v1/audit/logs`, `/api/v1/export/learner-data`
- L1 设计文档第七章 7.4: 审计日志模型

融合世界先进方案:
- FERPA: 学生教育记录隐私保护 (Family Educational Rights and Privacy Act)
- GDPR: 数据主体权利 + 数据最小化 + 留存限制 + 被遗忘权
- PIPL: 个人信息保护法 — 敏感个人信息单独同意 + 跨境传输限制
- k-匿名 (Sweeney 2002): 准标识符泛化防重标识
- l-多样性 (Machanavajjhala 2006): 防止同质化攻击
- 差分隐私 (Dwork 2006): Laplace 机制 ε-差分隐私
- 审计日志哈希链: 类区块链防篡改 (SHA-256 链式哈希)
- Privacy by Design (Cavoukian 2009): 隐私嵌入式设计七原则
- NIST SP 800-53: 审计与问责 (AU) 控制族

模块组成:
1. 异常体系: L1PrivacyError 层级 (JSON-RPC -32650 范围)
2. DataClassifier: 数据分级 + RBAC 访问控制 + 数据最小化校验
3. DesensitizationEngine: 5 种脱敏方法 + K-匿名 + l-多样性 + 差分隐私
4. RetentionManager: 四阶段数据留存 (ACTIVE → ARCHIVED → ANONYMIZED → DELETED)
5. AuditLogger: append-only 审计日志 + 哈希链完整性验证
6. PrivacyEventNotifier: 隐私事件通知 (L1 → L0)
7. PrivacyGovernanceManager: 统一治理管理器 (Facade 模式)
"""

from __future__ import annotations

import hashlib
import hmac
import random
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from dy3_polaris.l1.models import (
    AuditAction,
    AuditLogEntry,
    AuditResult,
    DataLevel,
    DesensitizationMethod,
    K_ANONYMITY_MIN,
    L_DIVERSITY_MIN,
    PrivacyConfig,
    PrivacyEvent,
    RetentionPhase,
    RetentionPolicy,
    User,
    UserRole,
    bucket_response_time,
    desensitize_student_id,
)
from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 1. 常量定义
# ============================================================

# --- 留存时间阈值 (毫秒) ---
MS_PER_DAY: int = 24 * 60 * 60 * 1000          # 1 天毫秒数
RETENTION_ARCHIVE_DAYS: int = 365               # 毕业后 1 年 → 归档
RETENTION_ANONYMIZE_DAYS: int = 1095            # 毕业后 3 年 → 匿名化
RETENTION_DELETE_DAYS: int = 1095               # 毕业后 3 年 → 删除 (机密数据)

RETENTION_ARCHIVE_MS: int = RETENTION_ARCHIVE_DAYS * MS_PER_DAY
RETENTION_ANONYMIZE_MS: int = RETENTION_ANONYMIZE_DAYS * MS_PER_DAY

# --- 审计日志哈希链 ---
GENESIS_HASH: str = "0" * 64                    # 创世哈希 (SHA-256 零值)

# --- 差分隐私 ---
DP_CLAMP_MIN: float = 0.0                       # 差分隐私加噪后下限
DP_CLAMP_MAX: float = 1.0                       # 差分隐私加噪后上限

# --- 伪 ID 前缀 ---
PSEUDO_ID_PREFIX: str = "pseudo-"

# --- 审计日志默认分页 ---
DEFAULT_QUERY_LIMIT: int = 100


# ============================================================
# 2. 异常体系 (JSON-RPC -32650 范围)
# ============================================================


class L1PrivacyError(L6Error):
    """L1 隐私治理层基础异常 (JSON-RPC -32650).

    所有隐私治理相关异常的基类, 继承自 L6Error.
    """

    def __init__(
        self,
        code: str = "L1_PRIVACY_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        return -32650


class DataClassificationError(L1PrivacyError):
    """数据分级错误 (JSON-RPC -32651).

    数据类型无法分类、分级配置无效等.
    """

    def __init__(
        self,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "DATA_CLASSIFICATION_ERROR",
            detail or "数据分级错误",
            context,
        )

    def _jsonrpc_code(self) -> int:
        return -32651


class DesensitizationError(L1PrivacyError):
    """数据脱敏错误 (JSON-RPC -32652).

    不支持的脱敏方法、脱敏参数无效等.
    """

    def __init__(
        self,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "DESENSITIZATION_ERROR",
            detail or "数据脱敏错误",
            context,
        )

    def _jsonrpc_code(self) -> int:
        return -32652


class RetentionExecutionError(L1PrivacyError):
    """留存策略执行错误 (JSON-RPC -32653).

    留存阶段无效、执行失败等.
    """

    def __init__(
        self,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "RETENTION_EXECUTION_ERROR",
            detail or "留存策略执行错误",
            context,
        )

    def _jsonrpc_code(self) -> int:
        return -32653


class AuditLogError(L1PrivacyError):
    """审计日志错误 (JSON-RPC -32654).

    审计日志写入失败、查询参数无效等.
    """

    def __init__(
        self,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "AUDIT_LOG_ERROR",
            detail or "审计日志错误",
            context,
        )

    def _jsonrpc_code(self) -> int:
        return -32654


class PrivacyViolationError(L1PrivacyError):
    """隐私违规错误 (JSON-RPC -32655).

    越权访问、违反最小化原则、违反留存策略等.

    Attributes:
        user_id: 违规用户 ID
        violation_type: 违规类型 (如 "unauthorized_access")
    """

    def __init__(
        self,
        user_id: str = "",
        violation_type: str = "",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.user_id = user_id
        self.violation_type = violation_type
        ctx: dict[str, Any] = {}
        if user_id:
            ctx["user_id"] = user_id
        if violation_type:
            ctx["violation_type"] = violation_type
        if context:
            ctx.update(context)
        super().__init__(
            "PRIVACY_VIOLATION",
            detail or f"隐私违规: {violation_type}",
            ctx,
        )

    def _jsonrpc_code(self) -> int:
        return -32655


# ============================================================
# 3. 数据分级控制 (DataClassifier)
# ============================================================

# --- 数据类型 → 分级映射 (设计文档 6.1) ---
_DATA_TYPE_MAPPING: dict[str, DataLevel] = {
    # L1_PUBLIC: 公开数据
    "course_announcement": DataLevel.L1_PUBLIC,
    "public_notice": DataLevel.L1_PUBLIC,
    "syllabus": DataLevel.L1_PUBLIC,
    "course_catalog": DataLevel.L1_PUBLIC,
    "public_knowledge": DataLevel.L1_PUBLIC,

    # L2_INTERNAL: 内部数据
    "knowledge_base_content": DataLevel.L2_INTERNAL,
    "kb_content": DataLevel.L2_INTERNAL,
    "internal_notice": DataLevel.L2_INTERNAL,
    "agent_config": DataLevel.L2_INTERNAL,
    "system_log": DataLevel.L2_INTERNAL,

    # L3_SENSITIVE: 敏感数据 (个人学习数据)
    "learning_report": DataLevel.L3_SENSITIVE,
    "mastery_data": DataLevel.L3_SENSITIVE,
    "interaction_log": DataLevel.L3_SENSITIVE,
    "response_time": DataLevel.L3_SENSITIVE,
    "cognitive_load": DataLevel.L3_SENSITIVE,
    "learning_path": DataLevel.L3_SENSITIVE,

    # L4_CONFIDENTIAL: 机密数据 (核心隐私)
    "student_id": DataLevel.L4_CONFIDENTIAL,
    "student_name": DataLevel.L4_CONFIDENTIAL,
    "grade": DataLevel.L4_CONFIDENTIAL,
    "score": DataLevel.L4_CONFIDENTIAL,
    "contact_info": DataLevel.L4_CONFIDENTIAL,
    "exam_result": DataLevel.L4_CONFIDENTIAL,
}

# --- RBAC 访问矩阵: 角色 → 可访问数据级别集合 ---
_ACCESS_MATRIX: dict[UserRole, set[DataLevel]] = {
    UserRole.UNDERGRAD: {
        DataLevel.L1_PUBLIC,
        DataLevel.L2_INTERNAL,
    },
    UserRole.GRADUATE: {
        DataLevel.L1_PUBLIC,
        DataLevel.L2_INTERNAL,
        DataLevel.L3_SENSITIVE,
    },
    UserRole.TEACHER: {
        DataLevel.L1_PUBLIC,
        DataLevel.L2_INTERNAL,
        DataLevel.L3_SENSITIVE,
    },
    UserRole.ADMIN: {
        DataLevel.L1_PUBLIC,
        DataLevel.L2_INTERNAL,
        DataLevel.L3_SENSITIVE,
        DataLevel.L4_CONFIDENTIAL,
    },
    UserRole.ALUMNI: {
        DataLevel.L1_PUBLIC,
    },
}

# --- 数据最小化: 允许采集的字段 (白名单) ---
_ALLOWED_FIELDS: set[str] = {
    "student_id",
    "grade_level",
    "answer_correct",
    "response_time",
    "mastery",
    "learning_goal",
    "knowledge_point",
    "session_id",
    "interaction_count",
    "cognitive_load",
}

# --- 数据最小化: 禁止采集的字段 (黑名单) ---
_BLOCKED_FIELDS: set[str] = {
    "mouse_track",
    "device_fingerprint",
    "ip_address",
    "biometric_data",
    "geo_location",
    "keystroke_timing",
    "camera_feed",
    "microphone_data",
    "browser_history",
}


class DataClassifier:
    """数据分级控制器 (设计文档 6.1, 6.2).

    职责:
    1. 数据分类: 将数据类型映射到四级分类 (L1-L4)
    2. 访问控制: 基于 RBAC 矩阵检查角色访问权限
    3. 数据最小化: 校验采集字段是否符合最小化原则

    融合方案:
    - GB/T 35273: 个人信息分级 (四級分类)
    - GDPR Art.5(1)(c): 数据最小化原则
    - FERPA §1232g: 教育记录访问权限
    - NIST SP 800-53 AC-3: 访问执行
    """

    def classify(self, data_type: str) -> DataLevel:
        """将数据类型分类到安全级别.

        未知数据类型默认分类为 L2_INTERNAL (安全默认原则).

        Args:
            data_type: 数据类型标识符

        Returns:
            对应的数据分级
        """
        return _DATA_TYPE_MAPPING.get(data_type, DataLevel.L2_INTERNAL)

    def check_access(self, user: User, level: DataLevel) -> bool:
        """检查用户是否有权访问指定级别的数据.

        基于 RBAC 访问矩阵进行粗粒度权限检查.

        Args:
            user: 用户对象
            level: 目标数据分级

        Returns:
            True 如果用户有权访问
        """
        allowed_levels = _ACCESS_MATRIX.get(user.role, set())
        return level in allowed_levels

    def check_minimization(self, field_name: str) -> bool:
        """检查字段是否符合数据最小化原则.

        仅允许白名单中的字段被采集.
        黑名单字段一律禁止.

        Args:
            field_name: 字段名

        Returns:
            True 如果字段允许采集
        """
        if field_name in _BLOCKED_FIELDS:
            return False
        if field_name in _ALLOWED_FIELDS:
            return True
        # 未知字段默认不允许 (安全默认)
        return False

    def get_allowed_fields(self) -> list[str]:
        """获取允许采集的字段列表.

        Returns:
            允许采集的字段名列表
        """
        return sorted(_ALLOWED_FIELDS)

    def get_blocked_fields(self) -> list[str]:
        """获取禁止采集的字段列表.

        Returns:
            禁止采集的字段名列表
        """
        return sorted(_BLOCKED_FIELDS)


# ============================================================
# 4. 数据脱敏引擎 (DesensitizationEngine)
# ============================================================


class DesensitizationEngine:
    """数据脱敏引擎 (设计文档 6.3).

    支持五种脱敏方法:
    1. HASH: HMAC-SHA256 不可逆哈希 (学号等标识符)
    2. AGGREGATE: 聚合统计 (答题记录 → 正确率)
    3. BUCKET: 分桶泛化 (响应时间 → fast/normal/slow)
    4. DP_NOISE: 差分隐私加噪 (Laplace 机制, ε-差分隐私)
    5. PSEUDO_ID: 伪 ID 替换 (一致性映射, 可逆性断开)

    隐私模型检查:
    - K-匿名 (k-anonymity): 每个准标识符组至少 k 条记录
    - l-多样性 (l-diversity): 每个组内敏感属性至少 l 个不同值

    融合方案:
    - Dwork 2006: Laplace 机制 (差分隐私)
    - Sweeney 2002: k-匿名模型
    - Machanavajjhala 2006: l-多样性模型
    - NIST SP 800-188: 数据脱敏指南

    线程安全: PSEUDO_ID 映射表受锁保护.
    """

    def __init__(self, privacy_config: PrivacyConfig | None = None) -> None:
        """初始化脱敏引擎.

        Args:
            privacy_config: 隐私配置 (K-匿名, l-多样性, 差分隐私参数)
        """
        self._config = privacy_config or PrivacyConfig()
        self._pseudo_id_map: dict[str, str] = {}
        self._pseudo_counter: int = 0
        self._lock = threading.Lock()

    def desensitize(
        self,
        data: Any,
        method: DesensitizationMethod | str,
        salt: str = "",
    ) -> Any:
        """对数据执行脱敏操作.

        Args:
            data: 原始数据 (类型取决于脱敏方法)
            method: 脱敏方法
            salt: 盐值 (仅 HASH 方法使用)

        Returns:
            脱敏后的数据

        Raises:
            DesensitizationError: 不支持的脱敏方法
        """
        # 规范化方法参数
        if isinstance(method, str):
            try:
                method = DesensitizationMethod(method)
            except ValueError:
                raise DesensitizationError(
                    detail=f"不支持的脱敏方法: {method}",
                    context={"method": str(method)},
                )

        if method == DesensitizationMethod.HASH:
            return self._hash_desensitize(str(data), salt)
        elif method == DesensitizationMethod.AGGREGATE:
            return self._aggregate_desensitize(data)
        elif method == DesensitizationMethod.BUCKET:
            return self._bucket_desensitize(data)
        elif method == DesensitizationMethod.DP_NOISE:
            return self._dp_noise_desensitize(data)
        elif method == DesensitizationMethod.PSEUDO_ID:
            return self._pseudo_id_desensitize(str(data))
        else:
            raise DesensitizationError(
                detail=f"不支持的脱敏方法: {method}",
                context={"method": str(method)},
            )

    def _hash_desensitize(self, data: str, salt: str = "") -> str:
        """HMAC-SHA256 不可逆哈希脱敏.

        使用 salt + data 作为 HMAC 密钥, 生成 64 字符十六进制哈希.
        相同输入 + 盐值产生相同哈希 (确定性).

        Args:
            data: 原始字符串
            salt: 盐值

        Returns:
            64 字符 SHA-256 十六进制哈希
        """
        key_material = f"{salt}:{data}".encode("utf-8")
        return hmac.new(key_material, b"", hashlib.sha256).hexdigest()

    def _aggregate_desensitize(self, data: Any) -> float:
        """聚合统计脱敏 — 将明细数据聚合为统计值.

        支持的输入:
        - list[bool]: 布尔列表 → True 比例 (正确率)
        - list[int/float]: 数值列表 → 平均值

        Args:
            data: 原始数据列表

        Returns:
            聚合后的统计值
        """
        if not isinstance(data, list) or len(data) == 0:
            return 0.0

        # 布尔列表 → True 比例
        if all(isinstance(x, bool) for x in data):
            true_count = sum(1 for x in data if x)
            return true_count / len(data)

        # 数值列表 → 平均值
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in data):
            return sum(data) / len(data)

        # 默认: 尝试计算 True 比例
        true_count = sum(1 for x in data if x)
        return true_count / len(data)

    def _bucket_desensitize(self, data: Any) -> str:
        """分桶泛化脱敏 — 将连续值映射到离散桶.

        响应时间分桶 (复用 models.bucket_response_time):
        - fast: < 5 秒
        - normal: 5-60 秒
        - slow: > 60 秒

        Args:
            data: 原始数值 (毫秒)

        Returns:
            桶标签字符串
        """
        if isinstance(data, (int, float)):
            return bucket_response_time(int(data))
        return "unknown"

    def _dp_noise_desensitize(self, data: Any) -> float:
        """差分隐私加噪脱敏 — Laplace 机制.

        对数值添加 Laplace(0, 1/ε) 噪声, 实现 ε-差分隐私.
        结果裁剪到 [0.0, 1.0] 范围.

        数学保证:
        - ε 越小 → 噪声越大 → 隐私保护越强
        - ε 越大 → 噪声越小 → 数据可用性越高

        Laplace(0, b) 采样: |X| ~ Exp(1/b), 符号各 50% 概率.

        Args:
            data: 原始数值 (期望在 [0, 1] 范围)

        Returns:
            加噪后的值 (裁剪到 [0.0, 1.0])
        """
        value = float(data) if isinstance(data, (int, float)) else 0.0
        epsilon = self._config.epsilon
        # Laplace 机制: noise ~ Lap(0, 1/ε)
        scale = 1.0 / epsilon if epsilon > 0 else 1.0
        # Laplace 采样: 先采样指数分布 |X|, 再随机赋符号
        noise = random.expovariate(1.0 / scale)
        if random.random() < 0.5:
            noise = -noise

        result = value + noise
        # 裁剪到 [0, 1]
        return max(DP_CLAMP_MIN, min(DP_CLAMP_MAX, result))

    def _pseudo_id_desensitize(self, data: str) -> str:
        """伪 ID 替换脱敏 — 一致性映射.

        同一引擎实例内, 相同输入始终映射到相同伪 ID.
        伪 ID 格式: "pseudo-" + 序号 (如 "pseudo-001").

        线程安全: 映射操作受锁保护.

        Args:
            data: 原始标识符

        Returns:
            伪 ID 字符串
        """
        with self._lock:
            if data not in self._pseudo_id_map:
                self._pseudo_counter += 1
                pseudo_id = f"{PSEUDO_ID_PREFIX}{self._pseudo_counter:06d}"
                self._pseudo_id_map[data] = pseudo_id
            return self._pseudo_id_map[data]

    def check_k_anonymity(
        self,
        records: list[dict[str, Any]],
        qi_field: str,
        k: int = K_ANONYMITY_MIN,
    ) -> bool:
        """检查数据集是否满足 K-匿名.

        K-匿名要求: 每个准标识符 (QI) 组至少包含 k 条记录.
        空数据集视为满足 (无记录可重标识).

        Args:
            records: 数据记录列表
            qi_field: 准标识符字段名
            k: 最小组大小

        Returns:
            True 如果满足 K-匿名
        """
        if not records:
            return True

        # 按准标识符分组
        groups: dict[Any, int] = Counter()
        for record in records:
            qi_value = record.get(qi_field)
            groups[qi_value] += 1

        # 检查每个组是否至少有 k 条记录
        return all(count >= k for count in groups.values())

    def check_l_diversity(
        self,
        records: list[dict[str, Any]],
        qi_field: str,
        sensitive_field: str,
        l: int = L_DIVERSITY_MIN,
    ) -> bool:
        """检查数据集是否满足 l-多样性.

        l-多样性要求: 每个准标识符组内, 敏感属性至少有 l 个不同值.
        空数据集视为满足.

        Args:
            records: 数据记录列表
            qi_field: 准标识符字段名
            sensitive_field: 敏感属性字段名
            l: 最小多样性值

        Returns:
            True 如果满足 l-多样性
        """
        if not records:
            return True

        # 按准标识符分组
        groups: dict[Any, set[Any]] = {}
        for record in records:
            qi_value = record.get(qi_field)
            sensitive_value = record.get(sensitive_field)
            if qi_value not in groups:
                groups[qi_value] = set()
            groups[qi_value].add(sensitive_value)

        # 检查每个组是否有至少 l 个不同的敏感值
        return all(len(values) >= l for values in groups.values())

    def anonymize_for_research(
        self,
        records: list[dict[str, Any]],
        quasi_identifiers: list[str],
        sensitive_attributes: list[str],
        k: int = K_ANONYMITY_MIN,
        l: int = L_DIVERSITY_MIN,
    ) -> list[dict[str, Any]]:
        """数据集匿名化 — 用于研究导出.

        匿名化步骤:
        1. 标识符替换: student_id 等直接标识符替换为伪 ID
        2. K-匿名检查: 按 QI 分组, 每组至少 k 条
        3. l-多样性检查: 每组敏感属性至少 l 个不同值
        4. 不满足的组: 删除该组所有记录 (抑制)

        Args:
            records: 原始数据记录
            quasi_identifiers: 准标识符字段列表
            sensitive_attributes: 敏感属性字段列表
            k: K-匿名参数
            l: l-多样性参数

        Returns:
            匿名化后的数据记录 (可能少于原始记录数)
        """
        if not records:
            return []

        # 1. 标识符替换
        anonymized: list[dict[str, Any]] = []
        for record in records:
            anon_record = dict(record)
            # 替换直接标识符
            if "student_id" in anon_record:
                anon_record["student_id"] = self._pseudo_id_desensitize(
                    str(anon_record["student_id"])
                )
            if "student_name" in anon_record:
                del anon_record["student_name"]
            anonymized.append(anon_record)

        # 2. 按 QI 分组检查 K-匿名和 l-多样性
        # 使用第一个 QI 字段进行分组 (简化处理)
        primary_qi = quasi_identifiers[0] if quasi_identifiers else None
        primary_sensitive = sensitive_attributes[0] if sensitive_attributes else None

        if primary_qi is None:
            return anonymized

        # 分组
        groups: dict[Any, list[dict[str, Any]]] = {}
        for record in anonymized:
            qi_value = record.get(primary_qi)
            if qi_value not in groups:
                groups[qi_value] = []
            groups[qi_value].append(record)

        # 检查每组是否满足 k 和 l
        result: list[dict[str, Any]] = []
        for qi_value, group_records in groups.items():
            # K-匿名检查
            if len(group_records) < k:
                continue  # 抑制不满足的组

            # l-多样性检查
            if primary_sensitive:
                sensitive_values = {
                    r.get(primary_sensitive) for r in group_records
                }
                if len(sensitive_values) < l:
                    continue  # 抑制不满足的组

            result.extend(group_records)

        return result


# ============================================================
# 5. 数据留存管理 (RetentionManager)
# ============================================================


@dataclass
class RetentionAction:
    """留存策略检查结果 — 描述当前应执行的留存动作.

    Attributes:
        phase: 当前留存阶段
        actions: 具体动作列表 (如 "学号脱敏", "成绩匿名化")
        days_since_graduation: 毕业后天数 (未毕业为 None)
    """

    phase: RetentionPhase
    actions: list[str] = field(default_factory=list)
    days_since_graduation: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "actions": self.actions,
            "days_since_graduation": self.days_since_graduation,
        }


@dataclass
class RetentionResult:
    """留存策略执行结果.

    Attributes:
        phase: 执行的留存阶段
        success: 是否执行成功
        detail: 执行详情
    """

    phase: RetentionPhase
    success: bool = True
    detail: str = ""


# --- 各阶段留存动作描述 ---
_RETENTION_ACTIONS: dict[RetentionPhase, list[str]] = {
    RetentionPhase.ACTIVE: [
        "数据正常保留",
        "定期备份",
    ],
    RetentionPhase.ARCHIVED: [
        "数据归档至冷存储",
        "访问权限收紧 (仅管理员可访问)",
    ],
    RetentionPhase.ANONYMIZED: [
        "学号脱敏 (HMAC-SHA256)",
        "姓名字段删除",
        "成绩数据匿名化",
        "交互记录聚合化",
    ],
    RetentionPhase.DELETED: [
        "物理删除所有个人标识数据",
        "删除关联学习记录",
        "保留脱敏统计汇总",
    ],
}

# --- 各数据级别留存策略 ---
_RETENTION_POLICIES: dict[DataLevel, RetentionPolicy] = {
    DataLevel.L1_PUBLIC: RetentionPolicy(
        data_level=DataLevel.L1_PUBLIC.value,
        phases=[
            (RetentionPhase.ACTIVE, -1),     # 无限期保留
            (RetentionPhase.ARCHIVED, -1),   # 永久归档
        ],
    ),
    DataLevel.L2_INTERNAL: RetentionPolicy(
        data_level=DataLevel.L2_INTERNAL.value,
        phases=[
            (RetentionPhase.ACTIVE, 0),
            (RetentionPhase.ARCHIVED, 365),      # 1 年后归档
            (RetentionPhase.ANONYMIZED, 1095),   # 3 年后匿名化
        ],
    ),
    DataLevel.L3_SENSITIVE: RetentionPolicy(
        data_level=DataLevel.L3_SENSITIVE.value,
        phases=[
            (RetentionPhase.ACTIVE, 0),
            (RetentionPhase.ARCHIVED, 365),      # 1 年后归档
            (RetentionPhase.ANONYMIZED, 1095),   # 3 年后匿名化
            (RetentionPhase.DELETED, 1825),      # 5 年后删除
        ],
    ),
    DataLevel.L4_CONFIDENTIAL: RetentionPolicy(
        data_level=DataLevel.L4_CONFIDENTIAL.value,
        phases=[
            (RetentionPhase.ACTIVE, 0),
            (RetentionPhase.ARCHIVED, 180),      # 6 个月后归档
            (RetentionPhase.ANONYMIZED, 365),    # 1 年后匿名化
            (RetentionPhase.DELETED, 1095),      # 3 年后删除
        ],
    ),
}


class RetentionManager:
    """数据留存策略管理器 (设计文档 6.4).

    四阶段生命周期管理:
    1. ACTIVE: 活跃期 — 用户在学期间, 数据正常使用
    2. ARCHIVED: 归档期 — 毕业后 1 年内, 数据转冷存储
    3. ANONYMIZED: 匿名化期 — 毕业后 1-3 年, 个人标识脱敏
    4. DELETED: 删除期 — 毕业后 3 年以上, 物理删除

    融合方案:
    - GDPR Art.5(1)(e): 存储限制原则 (不得超期保留)
    - GDPR Art.17: 被遗忘权 (删除权)
    - PIPL 第47条: 自动删除条件
    - FERPA §1232g(b)(1): 教育记录保留期限
    - NIST SP 800-88: 数据销毁指南
    """

    def get_policy(self, data_level: DataLevel) -> RetentionPolicy | None:
        """获取指定数据级别的留存策略.

        Args:
            data_level: 数据分级

        Returns:
            留存策略, 如果未定义则返回 None
        """
        return _RETENTION_POLICIES.get(data_level)

    def check_retention(
        self,
        user_id: str,
        graduation_ts: int | None,
        current_ts: int,
    ) -> RetentionAction:
        """检查用户数据的当前留存阶段.

        基于 graduation_ts (毕业时间戳) 计算留存阶段:
        - graduation_ts 为 None: ACTIVE (未毕业)
        - 毕业后 < 1 年: ARCHIVED
        - 毕业后 1-3 年: ANONYMIZED
        - 毕业后 ≥ 3 年: DELETED

        Args:
            user_id: 用户 ID
            graduation_ts: 毕业时间戳 (毫秒), None 表示未毕业
            current_ts: 当前时间戳 (毫秒)

        Returns:
            留存动作描述
        """
        if graduation_ts is None:
            return RetentionAction(
                phase=RetentionPhase.ACTIVE,
                actions=_RETENTION_ACTIONS[RetentionPhase.ACTIVE],
                days_since_graduation=None,
            )

        days_since = (current_ts - graduation_ts) / MS_PER_DAY

        if days_since < RETENTION_ARCHIVE_DAYS:
            phase = RetentionPhase.ARCHIVED
        elif days_since < RETENTION_ANONYMIZE_DAYS:
            phase = RetentionPhase.ANONYMIZED
        else:
            phase = RetentionPhase.DELETED

        return RetentionAction(
            phase=phase,
            actions=_RETENTION_ACTIONS[phase],
            days_since_graduation=days_since,
        )

    def execute_retention(
        self,
        user_id: str,
        phase: RetentionPhase,
    ) -> RetentionResult:
        """执行留存操作.

        根据留存阶段执行对应的数据处理操作.
        实际实现中会调用存储层执行物理操作.

        Args:
            user_id: 用户 ID
            phase: 目标留存阶段

        Returns:
            执行结果

        Raises:
            RetentionExecutionError: 执行失败
        """
        if phase not in _RETENTION_ACTIONS:
            raise RetentionExecutionError(
                detail=f"未知留存阶段: {phase}",
                context={"phase": str(phase)},
            )

        # 模拟执行 (生产环境对接存储层)
        actions = _RETENTION_ACTIONS[phase]
        return RetentionResult(
            phase=phase,
            success=True,
            detail=f"执行 {len(actions)} 项操作: {', '.join(actions)}",
        )


# ============================================================
# 6. 审计日志管理 (AuditLogger)
# ============================================================


@dataclass
class _ChainedEntry:
    """带哈希链的审计日志内部存储条目.

    Attributes:
        entry: 原始审计日志条目
        prev_hash: 前一条目的哈希 (创世条目为 GENESIS_HASH)
        curr_hash: 当前条目的哈希
    """

    entry: AuditLogEntry
    prev_hash: str
    curr_hash: str


class AuditLogger:
    """审计日志管理器 (设计文档 6.5, 7.4).

    核心特性:
    1. Append-only: 日志只追加, 不可删除或修改
    2. 哈希链完整性: 每条日志的哈希依赖前一条, 形成链式结构
    3. 防篡改: 任何修改都会破坏哈希链, verify_chain() 可检测
    4. 多维查询: 支持按操作者、操作类型、数据级别、结果、时间范围查询
    5. 分页: 支持 limit/offset 分页

    哈希链算法:
    - 第 1 条: curr_hash = SHA256(GENESIS_HASH + entry_content)
    - 第 N 条: curr_hash = SHA256(prev.curr_hash + entry_content)
    - entry_content = log_id + actor_id + action + target_resource + timestamp

    融合方案:
    - PROV-O: 审计日志溯源模型 (W3C Provenance)
    - Blockchain: 哈希链防篡改 (类区块链结构)
    - NIST SP 800-92: 日志管理指南
    - FERPA §1232g(b)(3): 教育记录访问日志
    - GDPR Art.30: 处理活动记录

    线程安全: 所有操作受读写锁保护.
    """

    def __init__(self) -> None:
        self._entries: list[_ChainedEntry] = []
        self._lock = threading.RLock()

    def log(self, entry: AuditLogEntry) -> None:
        """追加一条审计日志.

        自动计算哈希链. 日志一旦写入不可修改或删除.

        Args:
            entry: 审计日志条目
        """
        with self._lock:
            prev_hash = (
                self._entries[-1].curr_hash
                if self._entries
                else GENESIS_HASH
            )
            curr_hash = self._compute_hash(entry, prev_hash)
            self._entries.append(
                _ChainedEntry(
                    entry=entry,
                    prev_hash=prev_hash,
                    curr_hash=curr_hash,
                )
            )

    def query(
        self,
        actor_id: str | None = None,
        action: AuditAction | None = None,
        data_level: DataLevel | None = None,
        result: AuditResult | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
        limit: int = DEFAULT_QUERY_LIMIT,
        offset: int = 0,
    ) -> list[AuditLogEntry]:
        """多维条件查询审计日志.

        支持按操作者、操作类型、数据级别、结果、时间范围过滤,
        并支持 limit/offset 分页.

        Args:
            actor_id: 操作者 ID (None 表示不过滤)
            action: 操作类型 (None 表示不过滤)
            data_level: 数据级别 (None 表示不过滤)
            result: 操作结果 (None 表示不过滤)
            start_ts: 起始时间戳 (None 表示不限)
            end_ts: 结束时间戳 (None 表示不限)
            limit: 返回最大条数
            offset: 跳过前 N 条

        Returns:
            匹配的审计日志条目列表
        """
        with self._lock:
            results: list[AuditLogEntry] = []
            for chained in self._entries:
                entry = chained.entry
                # 过滤条件
                if actor_id is not None and entry.actor_id != actor_id:
                    continue
                if action is not None and entry.action != action:
                    continue
                if data_level is not None and entry.target_data_level != data_level:
                    continue
                if result is not None and entry.result != result:
                    continue
                if start_ts is not None and entry.timestamp < start_ts:
                    continue
                if end_ts is not None and entry.timestamp > end_ts:
                    continue
                results.append(entry)

            # 分页
            return results[offset : offset + limit]

    def verify_chain(self) -> bool:
        """验证哈希链完整性.

        重新计算每条日志的哈希, 检查是否与存储的哈希一致.
        任何篡改都会导致哈希不匹配.

        Returns:
            True 如果哈希链完整
        """
        with self._lock:
            prev_hash = GENESIS_HASH
            for chained in self._entries:
                expected_hash = self._compute_hash(chained.entry, prev_hash)
                if chained.prev_hash != prev_hash:
                    return False
                if chained.curr_hash != expected_hash:
                    return False
                prev_hash = chained.curr_hash
            return True

    def get_stats(self) -> dict[str, int]:
        """获取审计日志统计信息.

        Returns:
            统计字典:
            - total: 总日志数
            - success: 成功操作数
            - denied: 拒绝操作数
            - error: 错误操作数
        """
        with self._lock:
            stats = {
                "total": len(self._entries),
                "success": 0,
                "denied": 0,
                "error": 0,
            }
            for chained in self._entries:
                if chained.entry.result == AuditResult.SUCCESS:
                    stats["success"] += 1
                elif chained.entry.result == AuditResult.DENIED:
                    stats["denied"] += 1
                elif chained.entry.result == AuditResult.ERROR:
                    stats["error"] += 1
            return stats

    def __len__(self) -> int:
        """返回审计日志总数."""
        with self._lock:
            return len(self._entries)

    @staticmethod
    def _compute_hash(entry: AuditLogEntry, prev_hash: str) -> str:
        """计算审计日志条目的哈希.

        哈希输入 = prev_hash + entry_content
        entry_content = log_id + actor_id + action + target_resource + timestamp

        Args:
            entry: 审计日志条目
            prev_hash: 前一条目的哈希

        Returns:
            SHA-256 十六进制哈希 (64 字符)
        """
        content = (
            f"{entry.log_id}"
            f"|{entry.actor_id}"
            f"|{entry.action.value}"
            f"|{entry.target_resource}"
            f"|{entry.target_data_level.value}"
            f"|{entry.result.value}"
            f"|{entry.timestamp}"
        )
        data = f"{prev_hash}:{content}".encode("utf-8")
        return hashlib.sha256(data).hexdigest()


# ============================================================
# 7. 隐私事件通知 (PrivacyEventNotifier)
# ============================================================


class PrivacyEventNotifier:
    """隐私事件通知器 (设计文档 6.6, 8.1).

    职责:
    - 接收隐私事件 (越权访问、留存触发、数据导出等)
    - 生成 PrivacyEvent 并入队
    - 通知 L0 全局治理层 (生产环境通过消息队列)

    事件类型:
    - unauthorized_access: 越权访问尝试
    - data_export: 数据导出操作
    - retention_anonymize: 留存策略触发匿名化
    - retention_delete: 留存策略触发删除
    - minimization_violation: 违反数据最小化原则

    融合方案:
    - GDPR Art.33: 数据泄露通知 (72 小时内)
    - NIST SP 800-53 AU-6: 审计监控与异常检测
    - SIEM: 安全事件管理 (事件关联分析)

    线程安全: 事件队列操作受锁保护.
    """

    def __init__(self) -> None:
        self._events: list[PrivacyEvent] = []
        self._lock = threading.Lock()

    def notify(
        self,
        event_type: str,
        user_id: str,
        data_level: DataLevel = DataLevel.L3_SENSITIVE,
        detail: str = "",
    ) -> PrivacyEvent:
        """发送隐私事件通知.

        Args:
            event_type: 事件类型
            user_id: 关联用户 ID
            data_level: 涉及的数据级别
            detail: 事件详情

        Returns:
            创建的隐私事件
        """
        event = PrivacyEvent(
            event_type=event_type,
            user_id=user_id,
            data_level=data_level,
            detail=detail,
        )
        with self._lock:
            self._events.append(event)
        return event

    def get_events(self) -> list[PrivacyEvent]:
        """获取所有隐私事件.

        Returns:
            隐私事件列表 (按时间顺序)
        """
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """清除所有隐私事件."""
        with self._lock:
            self._events.clear()


# ============================================================
# 8. 统一隐私治理管理器 (PrivacyGovernanceManager)
# ============================================================


class PrivacyGovernanceManager:
    """隐私治理统一管理器 (Facade 模式).

    整合所有隐私治理组件, 提供统一接口:
    - DataClassifier: 数据分级与访问控制
    - DesensitizationEngine: 数据脱敏
    - RetentionManager: 留存策略管理
    - AuditLogger: 审计日志记录
    - PrivacyEventNotifier: 隐私事件通知

    设计原则:
    - Privacy by Design: 隐私保护嵌入所有数据处理流程
    - 最小权限: 默认拒绝, 显式授权
    - 安全默认: 导出数据默认脱敏, 管理员也脱敏
    - 全链路审计: 所有隐私相关操作记录审计日志

    融合方案:
    - Gartner PEST: 隐私工程框架
    - Privacy by Design (Cavoukian): 七原则
    - NIST Privacy Framework: 识别-治理-控制-沟通
    """

    def __init__(self, privacy_config: PrivacyConfig | None = None) -> None:
        """初始化隐私治理管理器.

        Args:
            privacy_config: 隐私配置 (可选, 默认使用 PrivacyConfig 默认值)
        """
        self._config = privacy_config or PrivacyConfig()
        self._classifier = DataClassifier()
        self._desensitization_engine = DesensitizationEngine(
            privacy_config=self._config
        )
        self._retention_manager = RetentionManager()
        self._audit_logger = AuditLogger()
        self._event_notifier = PrivacyEventNotifier()
        self._lock = threading.RLock()

    # --- 组件属性 ---

    @property
    def classifier(self) -> DataClassifier:
        """数据分级控制器."""
        return self._classifier

    @property
    def desensitization_engine(self) -> DesensitizationEngine:
        """数据脱敏引擎."""
        return self._desensitization_engine

    @property
    def retention_manager(self) -> RetentionManager:
        """留存策略管理器."""
        return self._retention_manager

    @property
    def audit_logger(self) -> AuditLogger:
        """审计日志管理器."""
        return self._audit_logger

    @property
    def event_notifier(self) -> PrivacyEventNotifier:
        """隐私事件通知器."""
        return self._event_notifier

    # --- 数据分级与访问控制 ---

    def classify_data(self, data_type: str) -> DataLevel:
        """数据分级.

        Args:
            data_type: 数据类型标识符

        Returns:
            数据分级
        """
        return self._classifier.classify(data_type)

    def check_data_access(self, user: User, level: DataLevel) -> bool:
        """检查数据访问权限.

        Args:
            user: 用户对象
            level: 数据分级

        Returns:
            True 如果有权访问
        """
        return self._classifier.check_access(user, level)

    # --- 数据脱敏 ---

    def desensitize_student_id(self, student_id: str, salt: str = "") -> str:
        """学号脱敏.

        使用 HMAC-SHA256 + salt 生成不可逆哈希.

        Args:
            student_id: 原始学号
            salt: 盐值

        Returns:
            64 字符哈希字符串
        """
        return desensitize_student_id(student_id, salt)

    # --- 审计日志 ---

    def log_audit(self, entry: AuditLogEntry) -> None:
        """记录审计日志.

        Args:
            entry: 审计日志条目
        """
        self._audit_logger.log(entry)

    # --- 留存策略 ---

    def check_user_retention(
        self,
        user_id: str,
        graduation_ts: int | None,
        current_ts: int,
    ) -> RetentionAction:
        """检查用户留存阶段.

        Args:
            user_id: 用户 ID
            graduation_ts: 毕业时间戳 (毫秒)
            current_ts: 当前时间戳 (毫秒)

        Returns:
            留存动作
        """
        return self._retention_manager.check_retention(
            user_id=user_id,
            graduation_ts=graduation_ts,
            current_ts=current_ts,
        )

    def execute_user_retention(
        self,
        user_id: str,
        phase: RetentionPhase,
    ) -> RetentionResult:
        """执行用户留存操作.

        执行留存操作并自动发送隐私事件通知.

        Args:
            user_id: 用户 ID
            phase: 留存阶段

        Returns:
            执行结果
        """
        result = self._retention_manager.execute_retention(user_id, phase)

        # 根据阶段发送隐私事件通知
        event_map = {
            RetentionPhase.ARCHIVED: "retention_archive",
            RetentionPhase.ANONYMIZED: "retention_anonymize",
            RetentionPhase.DELETED: "retention_delete",
        }
        event_type = event_map.get(phase)
        if event_type:
            self._event_notifier.notify(
                event_type=event_type,
                user_id=user_id,
                data_level=DataLevel.L3_SENSITIVE,
                detail=f"留存策略执行: {phase.value}",
            )

        return result

    # --- 隐私事件通知 ---

    def notify_event(
        self,
        event_type: str,
        user_id: str,
        data_level: DataLevel = DataLevel.L3_SENSITIVE,
        detail: str = "",
    ) -> PrivacyEvent:
        """发送隐私事件通知.

        Args:
            event_type: 事件类型
            user_id: 用户 ID
            data_level: 数据级别
            detail: 事件详情

        Returns:
            创建的隐私事件
        """
        return self._event_notifier.notify(
            event_type=event_type,
            user_id=user_id,
            data_level=data_level,
            detail=detail,
        )

    # --- 数据导出 (Privacy by Design) ---

    def export_learner_data(
        self,
        data: dict[str, Any],
        requester_role: UserRole,
        requester_id: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        """导出学情数据 (Privacy by Design).

        根据请求者角色执行不同级别的脱敏:
        - 本科生查看自己的数据: 保留原始学号 (FERPA 学生权利)
        - 本科生查看他人数据: 拒绝 (PrivacyViolationError)
        - 教师/研究生: 学号哈希脱敏, 姓名删除, 答题聚合
        - 管理员: 学号脱敏 (安全默认), 保留其他数据

        所有导出操作自动记录审计日志.

        Args:
            data: 原始学情数据
            requester_role: 请求者角色
            requester_id: 请求者 ID
            owner_id: 数据所有者 ID

        Returns:
            脱敏后的学情数据

        Raises:
            PrivacyViolationError: 本科生尝试访问他人数据
        """
        with self._lock:
            # 权限检查: 本科生只能查看自己的数据
            if requester_role == UserRole.UNDERGRAD:
                if requester_id is not None and owner_id is not None:
                    if requester_id != owner_id:
                        # 记录违规审计日志
                        self._audit_logger.log(
                            AuditLogEntry(
                                actor_id=requester_id or "unknown",
                                actor_role=requester_role,
                                action=AuditAction.EXPORT,
                                target_resource=f"learner_data:{owner_id}",
                                target_data_level=DataLevel.L4_CONFIDENTIAL,
                                purpose="数据导出",
                                result=AuditResult.DENIED,
                            )
                        )
                        # 发送违规通知
                        self._event_notifier.notify(
                            event_type="unauthorized_access",
                            user_id=requester_id or "unknown",
                            data_level=DataLevel.L4_CONFIDENTIAL,
                            detail=f"尝试越权访问用户 {owner_id} 的数据",
                        )
                        raise PrivacyViolationError(
                            user_id=requester_id or "",
                            violation_type="unauthorized_access",
                            detail=f"本科生 {requester_id} 尝试访问用户 {owner_id} 的数据",
                        )
                # 本科生查看自己的数据: 保留原始
                exported = dict(data)
            else:
                # 教师/研究生/管理员: 执行脱敏
                exported = self._desensitize_for_export(data, requester_role)

            # 记录审计日志
            self._audit_logger.log(
                AuditLogEntry(
                    actor_id=requester_id or "system",
                    actor_role=requester_role,
                    action=AuditAction.EXPORT,
                    target_resource=f"learner_data:{owner_id or 'unknown'}",
                    target_data_level=DataLevel.L3_SENSITIVE,
                    purpose="数据导出",
                    result=AuditResult.SUCCESS,
                )
            )

            return exported

    def _desensitize_for_export(
        self,
        data: dict[str, Any],
        requester_role: UserRole,
    ) -> dict[str, Any]:
        """对导出数据执行脱敏.

        脱敏规则:
        - student_id → HMAC-SHA256 哈希
        - student_name → 删除
        - answers (list[bool]) → 正确率
        - response_times (list[int]) → 分桶标签列表
        - mastery → 保留 (聚合数据, 不含个人标识)
        - grade_level → 保留 (低敏感)

        Args:
            data: 原始数据
            requester_role: 请求者角色

        Returns:
            脱敏后的数据
        """
        exported = {}

        for key, value in data.items():
            if key == "student_id":
                # 学号哈希脱敏
                exported[key] = self._desensitization_engine.desensitize(
                    str(value),
                    DesensitizationMethod.HASH,
                    salt="export",
                )
            elif key == "student_name":
                # 姓名字段删除 (不导出)
                continue
            elif key == "answers" and isinstance(value, list):
                # 答题记录聚合为正确率
                exported["answer_correct_rate"] = (
                    self._desensitization_engine.desensitize(
                        value,
                        DesensitizationMethod.AGGREGATE,
                    )
                )
            elif key == "response_times" and isinstance(value, list):
                # 响应时间分桶
                exported["response_time_buckets"] = [
                    self._desensitization_engine.desensitize(
                        rt,
                        DesensitizationMethod.BUCKET,
                    )
                    for rt in value
                ]
            else:
                # 其他字段保留
                exported[key] = value

        return exported


# ============================================================
# 9. 模块导出
# ============================================================

__all__ = [
    # 异常
    "L1PrivacyError",
    "DataClassificationError",
    "DesensitizationError",
    "RetentionExecutionError",
    "AuditLogError",
    "PrivacyViolationError",
    # 数据分级
    "DataClassifier",
    # 脱敏引擎
    "DesensitizationEngine",
    # 留存管理
    "RetentionManager",
    "RetentionAction",
    "RetentionResult",
    # 审计日志
    "AuditLogger",
    # 事件通知
    "PrivacyEventNotifier",
    # 统一管理器
    "PrivacyGovernanceManager",
    # 常量
    "MS_PER_DAY",
    "RETENTION_ARCHIVE_DAYS",
    "RETENTION_ANONYMIZE_DAYS",
    "GENESIS_HASH",
    "PSEUDO_ID_PREFIX",
]
