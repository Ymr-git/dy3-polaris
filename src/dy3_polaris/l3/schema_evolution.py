"""L3 领域知识层 — 模式演进与迁移管理.

借鉴世界先进的模式迁移方案:
- Neo4j schema migration: 图数据库模式版本追踪 + 约束迁移 + 在线模式演进
- Django migrations: 线性迁移链 + 自动检测 + 可逆迁移 + 迁移依赖图
- Alembic: 数据库迁移版本控制 + 升级/降级 + 自动生成迁移脚本
- JSON Schema evolution: 向后兼容性检查 + 语义版本号 + 兼容性矩阵

提供以下核心能力:
1. 本体版本追踪 — 记录每次本体变更的语义版本快照
2. 模式差异计算 — 对比两个版本间的结构差异 (Schema diff)
3. 向后兼容性检查 — 评估变更对已有数据的影响等级
4. 自动迁移计划生成 — 根据差异自动生成有序迁移步骤
5. 迁移执行与回滚 — 按序执行迁移步骤，支持逆向回滚
6. 迁移历史记录 — 追踪所有迁移操作的执行历史与统计

兼容性等级 (借鉴 JSON Schema 兼容性矩阵):
- FULL: 完全向后兼容 (仅有新增操作)
- PARTIAL: 大部分兼容 (存在重命名等轻微不兼容)
- BREAKING: 破坏性变更 (存在删除或类型变更)，需要迁移
- INCOMPATIBLE: 无法自动迁移 (多处破坏性变更)

线程安全: 所有共享状态通过 threading.RLock 保护。
所有变更均为内存操作，接口设计支持未来对接持久化后端。
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from threading import RLock
from typing import Any

from pydantic import BaseModel, Field

from .exceptions import L3Error
from .ontology import (
    DomainOntology,
    OntologyClass,
    OntologyProperty,
    OntologyRelation,
    OntologyRegistry,
)

logger = logging.getLogger(__name__)


# ============================================================
# 异常定义
# ============================================================


class SchemaEvolutionError(L3Error):
    """模式演进异常 (本体版本迁移/兼容性检查失败时触发).

    当版本快照未找到、迁移步骤执行失败、版本冲突等情况时抛出。
    JSON-RPC 错误码: -32415。
    """

    def __init__(
        self,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("L3_SCHEMA_EVOLUTION", detail, context)

    def _jsonrpc_code(self) -> int:
        return -32415


# ============================================================
# 枚举定义
# ============================================================


class ChangeType(str, Enum):
    """模式变更类型 (借鉴 Django migration operations 分类).

    涵盖类、属性、关系、约束、规则五大维度的增删改操作。
    """

    # 类变更
    CLASS_ADDED = "class_added"
    CLASS_REMOVED = "class_removed"
    CLASS_RENAMED = "class_renamed"
    # 属性变更
    PROPERTY_ADDED = "property_added"
    PROPERTY_REMOVED = "property_removed"
    PROPERTY_RENAMED = "property_renamed"
    PROPERTY_TYPE_CHANGED = "property_type_changed"
    # 关系变更
    RELATION_ADDED = "relation_added"
    RELATION_REMOVED = "relation_removed"
    RELATION_MODIFIED = "relation_modified"
    # 约束变更
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    # 规则变更
    RULE_ADDED = "rule_added"
    RULE_REMOVED = "rule_removed"


class CompatibilityLevel(str, Enum):
    """向后兼容性等级 (借鉴 JSON Schema / Protobuf 兼容性矩阵).

    Levels:
        FULL: 完全向后兼容，旧数据无需迁移即可在新模式下使用
        PARTIAL: 大部分兼容，存在轻微不兼容 (如重命名)，建议迁移
        BREAKING: 破坏性变更，旧数据必须迁移才能在新模式下使用
        INCOMPATIBLE: 无法自动迁移，需要手动干预或数据重构
    """

    FULL = "full"
    PARTIAL = "partial"
    BREAKING = "breaking"
    INCOMPATIBLE = "incompatible"


# ============================================================
# 数据模型 (pydantic v2)
# ============================================================


class SchemaChange(BaseModel):
    """单项模式变更记录.

    描述两个本体版本之间的一项原子变更。

    Attributes:
        change_type: 变更类型 (增/删/改/重命名/类型变更)
        target: 变更目标名称 (类/属性/关系的标识)
        old_value: 变更前的值 (新增时为 None)
        new_value: 变更后的值 (删除时为 None)
        description: 变更描述
        is_breaking: 是否为破坏性变更
    """

    change_type: ChangeType
    target: str
    old_value: Any = None
    new_value: Any = None
    description: str = ""
    is_breaking: bool = False


class SchemaVersion(BaseModel):
    """模式版本快照.

    记录某个时间点的本体模式状态，包含与前一个版本的差异。

    Attributes:
        version: 语义版本号 (如 "1.0.0")
        domain: 所属领域标识
        timestamp: 记录时间戳 (Unix epoch)
        changes: 与前一版本的变更列表 (首个版本为空)
        compatibility: 与前一版本的兼容性等级
        description: 版本描述
    """

    version: str
    domain: str
    timestamp: float
    changes: list[SchemaChange] = Field(default_factory=list)
    compatibility: CompatibilityLevel = CompatibilityLevel.FULL
    description: str = ""


class MigrationStep(BaseModel):
    """单个迁移步骤.

    描述一项原子迁移操作及其回滚方式。

    Attributes:
        step_id: 步骤唯一标识
        description: 步骤描述
        action: 执行动作 (如 "rename_class", "add_property")
        parameters: 动作参数
        is_reversible: 是否可逆
        rollback_action: 回滚动作 (空字符串表示使用逆操作)
    """

    step_id: str
    description: str
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    is_reversible: bool = True
    rollback_action: str = ""


class MigrationPlan(BaseModel):
    """迁移计划.

    从一个版本迁移到另一个版本的完整步骤序列。

    Attributes:
        plan_id: 计划唯一标识
        from_version: 起始版本号
        to_version: 目标版本号
        domain: 所属领域标识
        steps: 有序迁移步骤列表
        compatibility: 迁移兼容性等级
        estimated_time_ms: 预估执行时间 (毫秒)
        requires_backup: 是否需要预先备份
    """

    plan_id: str
    from_version: str
    to_version: str
    domain: str
    steps: list[MigrationStep] = Field(default_factory=list)
    compatibility: CompatibilityLevel
    estimated_time_ms: float = 0.0
    requires_backup: bool = True


class SchemaDiff(BaseModel):
    """模式差异报告.

    描述两个版本之间的完整差异。

    Attributes:
        from_version: 起始版本号
        to_version: 目标版本号
        domain: 所属领域标识
        changes: 变更列表
        compatibility: 兼容性等级
        summary: 差异摘要文本
    """

    from_version: str
    to_version: str
    domain: str
    changes: list[SchemaChange] = Field(default_factory=list)
    compatibility: CompatibilityLevel
    summary: str = ""


# ============================================================
# 模式演进管理器
# ============================================================


class SchemaEvolutionManager:
    """模式演进管理器 (借鉴 Neo4j schema 迁移 + Django migrations + Alembic).

    管理领域本体的版本演进生命周期，包括版本追踪、差异计算、
    兼容性检查、迁移计划生成、迁移执行与回滚、历史记录。

    功能:
    1. 本体版本追踪 (语义版本号)
    2. 模式差异计算 (Schema diff)
    3. 向后兼容性检查
    4. 自动迁移计划生成
    5. 迁移执行与回滚
    6. 迁移历史记录

    Usage::

        manager = SchemaEvolutionManager(registry)
        manager.record_version("chemistry", ontology_v1, description="初始版本")
        manager.record_version("chemistry", ontology_v2, description="新增属性")
        diff = manager.compute_diff("chemistry", "1.0.0", "2.0.0")
        plan = manager.generate_migration_plan("chemistry", "1.0.0", "2.0.0")
        result = manager.execute_migration(plan)
    """

    # 变更类型 -> (执行动作, 回滚动作, 是否可逆) 映射表
    _CHANGE_ACTION_MAP: dict[ChangeType, tuple[str, str, bool]] = {
        ChangeType.CLASS_ADDED: ("add_class", "remove_class", True),
        ChangeType.CLASS_REMOVED: ("remove_class", "add_class", True),
        ChangeType.CLASS_RENAMED: ("rename_class", "rename_class", True),
        ChangeType.PROPERTY_ADDED: ("add_property", "remove_property", True),
        ChangeType.PROPERTY_REMOVED: ("remove_property", "add_property", True),
        ChangeType.PROPERTY_RENAMED: ("rename_property", "rename_property", True),
        ChangeType.PROPERTY_TYPE_CHANGED: ("transform_data", "transform_data", False),
        ChangeType.RELATION_ADDED: ("add_relation", "remove_relation", True),
        ChangeType.RELATION_REMOVED: ("remove_relation", "add_relation", True),
        ChangeType.RELATION_MODIFIED: ("modify_relation", "modify_relation", False),
        ChangeType.CONSTRAINT_ADDED: ("add_constraint", "remove_constraint", True),
        ChangeType.CONSTRAINT_REMOVED: ("remove_constraint", "add_constraint", True),
        ChangeType.RULE_ADDED: ("add_rule", "remove_rule", True),
        ChangeType.RULE_REMOVED: ("remove_rule", "add_rule", True),
    }

    # 合法的迁移动作集合
    _VALID_ACTIONS: set[str] = {a for a, _, _ in _CHANGE_ACTION_MAP.values()}

    # 动作预估耗时 (毫秒)
    _ACTION_TIME_MS: dict[str, float] = {
        "transform_data": 500.0,
        "remove_class": 300.0,
        "remove_property": 300.0,
        "remove_relation": 200.0,
        "modify_relation": 200.0,
    }
    _DEFAULT_ACTION_TIME_MS: float = 100.0

    def __init__(self, registry: OntologyRegistry | None = None) -> None:
        """初始化模式演进管理器.

        Args:
            registry: 本体注册中心 (为 None 时创建默认实例)
        """
        self._registry: OntologyRegistry = registry or OntologyRegistry()
        # 领域 -> 版本历史列表
        self._versions: dict[str, list[SchemaVersion]] = {}
        # 领域 -> {版本号 -> 本体快照}
        self._snapshots: dict[str, dict[str, DomainOntology]] = {}
        # 已执行的迁移计划历史
        self._migration_history: list[MigrationPlan] = []
        # 已执行但未回滚的计划 (plan_id -> plan)
        self._executed_plans: dict[str, MigrationPlan] = {}
        self._lock: RLock = RLock()

    # ============================================================
    # 公开接口: 版本管理
    # ============================================================

    def record_version(
        self,
        domain: str,
        ontology: DomainOntology,
        *,
        description: str = "",
    ) -> SchemaVersion:
        """记录当前本体状态为一个新的版本快照.

        若该领域已有历史版本，则自动计算与最新版本的差异。

        Args:
            domain: 领域标识
            ontology: 领域本体实例
            description: 版本描述

        Returns:
            新创建的模式版本记录

        Raises:
            SchemaEvolutionError: 版本号已存在
        """
        with self._lock:
            version_str = ontology.version
            domain_snapshots = self._snapshots.setdefault(domain, {})
            domain_versions = self._versions.setdefault(domain, [])

            # 检查版本号是否已存在
            if version_str in domain_snapshots:
                raise SchemaEvolutionError(
                    f"版本号已存在: domain={domain}, version={version_str}",
                    context={"domain": domain, "version": version_str},
                )

            # 与前一版本比较，计算变更
            changes: list[SchemaChange] = []
            compatibility = CompatibilityLevel.FULL
            if domain_versions:
                prev_version = domain_versions[-1]
                prev_ontology = domain_snapshots[prev_version.version]
                changes = self._compare_ontologies(prev_ontology, ontology)
                compatibility = self._determine_compatibility(changes)

            schema_version = SchemaVersion(
                version=version_str,
                domain=domain,
                timestamp=time.time(),
                changes=changes,
                compatibility=compatibility,
                description=description,
            )

            domain_versions.append(schema_version)
            domain_snapshots[version_str] = ontology.model_copy(deep=True)

            logger.info(
                "记录版本: domain=%s, version=%s, changes=%d, compat=%s",
                domain,
                version_str,
                len(changes),
                compatibility.value,
            )
            return schema_version

    def compute_diff(
        self,
        domain: str,
        from_version: str,
        to_version: str,
    ) -> SchemaDiff:
        """计算两个版本之间的模式差异.

        Args:
            domain: 领域标识
            from_version: 起始版本号
            to_version: 目标版本号

        Returns:
            模式差异报告

        Raises:
            SchemaEvolutionError: 版本快照未找到
        """
        with self._lock:
            old_ontology = self._get_ontology_snapshot(domain, from_version)
            new_ontology = self._get_ontology_snapshot(domain, to_version)
            changes = self._compare_ontologies(old_ontology, new_ontology)
            compatibility = self._determine_compatibility(changes)
            summary = self._build_summary(changes)

            return SchemaDiff(
                from_version=from_version,
                to_version=to_version,
                domain=domain,
                changes=changes,
                compatibility=compatibility,
                summary=summary,
            )

    def check_compatibility(
        self,
        domain: str,
        from_version: str,
        to_version: str,
    ) -> CompatibilityLevel:
        """检查两个版本之间的向后兼容性.

        Args:
            domain: 领域标识
            from_version: 起始版本号
            to_version: 目标版本号

        Returns:
            兼容性等级
        """
        diff = self.compute_diff(domain, from_version, to_version)
        return diff.compatibility

    # ============================================================
    # 公开接口: 迁移管理
    # ============================================================

    def generate_migration_plan(
        self,
        domain: str,
        from_version: str,
        to_version: str,
    ) -> MigrationPlan:
        """生成从起始版本到目标版本的迁移计划.

        根据版本差异自动生成有序迁移步骤。

        Args:
            domain: 领域标识
            from_version: 起始版本号
            to_version: 目标版本号

        Returns:
            迁移计划

        Raises:
            SchemaEvolutionError: 版本快照未找到
        """
        with self._lock:
            diff = self.compute_diff(domain, from_version, to_version)
            steps = self._generate_steps(diff.changes)

            # 估算执行时间
            estimated_time = sum(
                self._ACTION_TIME_MS.get(s.action, self._DEFAULT_ACTION_TIME_MS)
                for s in steps
            )

            # 破坏性变更需要备份
            requires_backup = diff.compatibility in (
                CompatibilityLevel.BREAKING,
                CompatibilityLevel.INCOMPATIBLE,
            )

            plan_id = f"migration-{domain}-{from_version}-to-{to_version}"

            plan = MigrationPlan(
                plan_id=plan_id,
                from_version=from_version,
                to_version=to_version,
                domain=domain,
                steps=steps,
                compatibility=diff.compatibility,
                estimated_time_ms=estimated_time,
                requires_backup=requires_backup,
            )

            logger.info(
                "生成迁移计划: %s, steps=%d, compat=%s, est=%.0fms",
                plan_id,
                len(steps),
                diff.compatibility.value,
                estimated_time,
            )
            return plan

    def execute_migration(
        self,
        plan: MigrationPlan,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """执行迁移计划.

        按顺序执行每个迁移步骤，支持模拟运行 (dry_run)。
        任一步骤失败时立即中止并返回失败信息。

        Args:
            plan: 迁移计划
            dry_run: 是否模拟运行 (不实际执行)

        Returns:
            执行结果字典，包含状态、进度、各步骤结果
        """
        with self._lock:
            mode = "模拟" if dry_run else "实际"
            result: dict[str, Any] = {
                "plan_id": plan.plan_id,
                "domain": plan.domain,
                "from_version": plan.from_version,
                "to_version": plan.to_version,
                "mode": mode,
                "total_steps": len(plan.steps),
                "completed_steps": 0,
                "failed_step": None,
                "status": "running",
                "step_results": [],
            }

            logger.info("开始%s执行迁移计划: %s (%d 步)", mode, plan.plan_id, len(plan.steps))

            for i, step in enumerate(plan.steps):
                step_result: dict[str, Any] = {
                    "step_id": step.step_id,
                    "action": step.action,
                    "description": step.description,
                    "status": "pending",
                }

                try:
                    if dry_run:
                        step_result["status"] = "simulated"
                        step_result["output"] = {
                            "action": step.action,
                            "target": step.parameters.get("target", ""),
                        }
                    else:
                        output = self._execute_step(step)
                        step_result["status"] = "success"
                        step_result["output"] = output

                    result["step_results"].append(step_result)
                    result["completed_steps"] = i + 1
                except Exception as exc:
                    step_result["status"] = "failed"
                    step_result["error"] = str(exc)
                    result["step_results"].append(step_result)
                    result["failed_step"] = step.step_id
                    result["status"] = "failed"
                    logger.error(
                        "迁移步骤失败: plan=%s, step=%s, error=%s",
                        plan.plan_id, step.step_id, exc,
                    )
                    return result

            result["status"] = "completed"

            # 记录到迁移历史 (非模拟运行)
            if not dry_run:
                self._migration_history.append(plan)
                self._executed_plans[plan.plan_id] = plan

            logger.info(
                "迁移计划%s执行完成: %s, completed=%d/%d",
                mode, plan.plan_id, result["completed_steps"], len(plan.steps),
            )
            return result

    def rollback_migration(self, plan: MigrationPlan) -> dict[str, Any]:
        """回滚先前执行的迁移计划.

        按逆序执行每个可逆步骤的回滚动作。
        不可逆步骤将被跳过并记录。

        Args:
            plan: 先前执行的迁移计划

        Returns:
            回滚结果字典，包含状态、进度、各步骤结果
        """
        with self._lock:
            result: dict[str, Any] = {
                "plan_id": plan.plan_id,
                "domain": plan.domain,
                "total_steps": len(plan.steps),
                "rolled_back_steps": 0,
                "skipped_steps": 0,
                "failed_step": None,
                "status": "running",
                "step_results": [],
            }

            logger.info("开始回滚迁移计划: %s (%d 步)", plan.plan_id, len(plan.steps))

            # 逆序回滚
            for i in range(len(plan.steps) - 1, -1, -1):
                step = plan.steps[i]
                step_result: dict[str, Any] = {
                    "step_id": step.step_id,
                    "action": step.rollback_action or f"reverse_{step.action}",
                    "description": step.description,
                    "status": "pending",
                }

                # 不可逆步骤跳过
                if not step.is_reversible:
                    step_result["status"] = "skipped"
                    step_result["reason"] = "步骤不可逆"
                    result["step_results"].append(step_result)
                    result["skipped_steps"] += 1
                    continue

                try:
                    output = self._execute_rollback(step)
                    step_result["status"] = "success"
                    step_result["output"] = output
                    result["step_results"].append(step_result)
                    result["rolled_back_steps"] += 1
                except Exception as exc:
                    step_result["status"] = "failed"
                    step_result["error"] = str(exc)
                    result["step_results"].append(step_result)
                    result["failed_step"] = step.step_id
                    result["status"] = "failed"
                    logger.error(
                        "回滚步骤失败: plan=%s, step=%s, error=%s",
                        plan.plan_id, step.step_id, exc,
                    )
                    return result

            result["status"] = "completed"

            # 从已执行计划中移除
            self._executed_plans.pop(plan.plan_id, None)

            logger.info(
                "迁移计划回滚完成: %s, rolled_back=%d, skipped=%d",
                plan.plan_id, result["rolled_back_steps"], result["skipped_steps"],
            )
            return result

    # ============================================================
    # 公开接口: 查询与统计
    # ============================================================

    def get_version_history(self, domain: str) -> list[SchemaVersion]:
        """获取指定领域的版本历史.

        Args:
            domain: 领域标识

        Returns:
            版本记录列表 (按记录时间排序)
        """
        with self._lock:
            return list(self._versions.get(domain, []))

    def get_latest_version(self, domain: str) -> SchemaVersion | None:
        """获取指定领域的最新版本.

        Args:
            domain: 领域标识

        Returns:
            最新版本记录 (无历史版本时返回 None)
        """
        with self._lock:
            versions = self._versions.get(domain, [])
            if not versions:
                return None
            return versions[-1]

    def get_stats(self) -> dict[str, Any]:
        """获取模式演进管理器的统计信息.

        Returns:
            统计信息字典，包含领域数、版本数、迁移数等
        """
        with self._lock:
            total_versions = sum(len(v) for v in self._versions.values())
            total_migrations = len(self._migration_history)
            active_migrations = len(self._executed_plans)

            domain_stats: dict[str, Any] = {}
            for domain, versions in self._versions.items():
                latest = versions[-1] if versions else None
                domain_stats[domain] = {
                    "version_count": len(versions),
                    "latest_version": latest.version if latest else None,
                    "latest_compatibility": (
                        latest.compatibility.value if latest else None
                    ),
                    "latest_timestamp": latest.timestamp if latest else None,
                }

            return {
                "total_domains": len(self._versions),
                "total_versions": total_versions,
                "total_migrations": total_migrations,
                "active_migrations": active_migrations,
                "domains": domain_stats,
            }

    # ============================================================
    # 内部方法: 本体比较
    # ============================================================

    def _compare_ontologies(
        self,
        old: DomainOntology,
        new: DomainOntology,
    ) -> list[SchemaChange]:
        """比较两个本体并返回变更列表.

        对比维度: 类 (增删改/重命名)、属性 (增删改/重命名/类型变更)、
        关系 (增删改)、约束 (增删)、规则 (增删)。

        Args:
            old: 旧版本本体
            new: 新版本本体

        Returns:
            变更列表 (按检测顺序: 类 -> 属性 -> 关系 -> 约束 -> 规则)
        """
        changes: list[SchemaChange] = []

        # --- 类变更检测 ---
        changes.extend(self._diff_classes(old, new))

        # --- 属性变更检测 ---
        changes.extend(self._diff_properties(old, new))

        # --- 关系变更检测 ---
        changes.extend(self._diff_relations(old, new))

        # --- 约束 (公理) 变更检测 ---
        changes.extend(self._diff_axioms(old, new))

        # --- 推理规则变更检测 ---
        changes.extend(self._diff_rules(old, new))

        return changes

    def _diff_classes(
        self,
        old: DomainOntology,
        new: DomainOntology,
    ) -> list[SchemaChange]:
        """检测类变更 (增删/重命名)."""
        changes: list[SchemaChange] = []
        old_classes = {c.entity_type: c for c in old.classes}
        new_classes = {c.entity_type: c for c in new.classes}

        old_keys = set(old_classes.keys())
        new_keys = set(new_classes.keys())
        added = new_keys - old_keys
        removed = old_keys - new_keys

        # 重命名检测: 结构相似的移除+新增视为重命名
        rename_pairs = self._detect_class_renames(old_classes, new_classes, removed, added)
        for old_key, new_key in rename_pairs:
            removed.discard(old_key)
            added.discard(new_key)
            changes.append(SchemaChange(
                change_type=ChangeType.CLASS_RENAMED,
                target=str(new_key.value),
                old_value=str(old_key.value),
                new_value=str(new_key.value),
                description=f"类重命名: {old_key.value} -> {new_key.value}",
                is_breaking=False,
            ))

        # 新增类
        for key in sorted(added, key=lambda k: str(k.value)):
            cls = new_classes[key]
            changes.append(SchemaChange(
                change_type=ChangeType.CLASS_ADDED,
                target=str(key.value),
                old_value=None,
                new_value=self._class_to_dict(cls),
                description=f"新增类: {cls.display_name or key.value}",
                is_breaking=False,
            ))

        # 删除类
        for key in sorted(removed, key=lambda k: str(k.value)):
            cls = old_classes[key]
            changes.append(SchemaChange(
                change_type=ChangeType.CLASS_REMOVED,
                target=str(key.value),
                old_value=self._class_to_dict(cls),
                new_value=None,
                description=f"删除类: {cls.display_name or key.value}",
                is_breaking=True,
            ))

        return changes

    def _diff_properties(
        self,
        old: DomainOntology,
        new: DomainOntology,
    ) -> list[SchemaChange]:
        """检测属性变更 (增删/重命名/类型变更)."""
        changes: list[SchemaChange] = []
        old_props = self._collect_all_properties(old)
        new_props = self._collect_all_properties(new)

        old_names = set(old_props.keys())
        new_names = set(new_props.keys())
        added = new_names - old_names
        removed = old_names - new_names
        common = old_names & new_names

        # 重命名检测: 同类型的移除+新增视为重命名
        rename_pairs = self._detect_property_renames(old_props, new_props, removed, added)
        for old_name, new_name in rename_pairs:
            removed.discard(old_name)
            added.discard(new_name)
            changes.append(SchemaChange(
                change_type=ChangeType.PROPERTY_RENAMED,
                target=new_name,
                old_value=old_name,
                new_value=new_name,
                description=f"属性重命名: {old_name} -> {new_name}",
                is_breaking=False,
            ))

        # 新增属性
        for name in sorted(added):
            prop = new_props[name]
            changes.append(SchemaChange(
                change_type=ChangeType.PROPERTY_ADDED,
                target=name,
                old_value=None,
                new_value=self._prop_to_dict(prop),
                description=f"新增属性: {name}",
                is_breaking=prop.required,  # 新增必需属性为破坏性变更
            ))

        # 删除属性
        for name in sorted(removed):
            prop = old_props[name]
            changes.append(SchemaChange(
                change_type=ChangeType.PROPERTY_REMOVED,
                target=name,
                old_value=self._prop_to_dict(prop),
                new_value=None,
                description=f"删除属性: {name}",
                is_breaking=True,
            ))

        # 类型变更检测
        for name in sorted(common):
            old_p = old_props[name]
            new_p = new_props[name]
            old_type = old_p.data_type.value if old_p.data_type else old_p.range
            new_type = new_p.data_type.value if new_p.data_type else new_p.range
            if old_type != new_type:
                changes.append(SchemaChange(
                    change_type=ChangeType.PROPERTY_TYPE_CHANGED,
                    target=name,
                    old_value=old_type,
                    new_value=new_type,
                    description=f"属性类型变更: {name} ({old_type} -> {new_type})",
                    is_breaking=True,
                ))

        return changes

    def _diff_relations(
        self,
        old: DomainOntology,
        new: DomainOntology,
    ) -> list[SchemaChange]:
        """检测关系变更 (增删/修改)."""
        changes: list[SchemaChange] = []
        old_rels = {r.name: r for r in old.relations}
        new_rels = {r.name: r for r in new.relations}

        old_names = set(old_rels.keys())
        new_names = set(new_rels.keys())
        added = new_names - old_names
        removed = old_names - new_names
        common = old_names & new_names

        # 新增关系
        for name in sorted(added):
            rel = new_rels[name]
            changes.append(SchemaChange(
                change_type=ChangeType.RELATION_ADDED,
                target=name,
                old_value=None,
                new_value=self._relation_to_dict(rel),
                description=f"新增关系: {name}",
                is_breaking=False,
            ))

        # 删除关系
        for name in sorted(removed):
            rel = old_rels[name]
            changes.append(SchemaChange(
                change_type=ChangeType.RELATION_REMOVED,
                target=name,
                old_value=self._relation_to_dict(rel),
                new_value=None,
                description=f"删除关系: {name}",
                is_breaking=True,
            ))

        # 修改检测
        for name in sorted(common):
            if self._relation_modified(old_rels[name], new_rels[name]):
                changes.append(SchemaChange(
                    change_type=ChangeType.RELATION_MODIFIED,
                    target=name,
                    old_value=self._relation_to_dict(old_rels[name]),
                    new_value=self._relation_to_dict(new_rels[name]),
                    description=f"关系修改: {name}",
                    is_breaking=True,
                ))

        return changes

    def _diff_axioms(
        self,
        old: DomainOntology,
        new: DomainOntology,
    ) -> list[SchemaChange]:
        """检测约束 (公理) 变更 (增删)."""
        changes: list[SchemaChange] = []
        old_axioms = {a.axiom_id: a for a in old.axioms}
        new_axioms = {a.axiom_id: a for a in new.axioms}

        old_ids = set(old_axioms.keys())
        new_ids = set(new_axioms.keys())
        added = new_ids - old_ids
        removed = old_ids - new_ids

        # 新增约束 (可能导致已有数据不符合新约束)
        for axiom_id in sorted(added):
            axiom = new_axioms[axiom_id]
            changes.append(SchemaChange(
                change_type=ChangeType.CONSTRAINT_ADDED,
                target=axiom_id,
                old_value=None,
                new_value={
                    "axiom_type": axiom.axiom_type,
                    "subject": axiom.subject,
                    "object": axiom.object,
                },
                description=f"新增约束: {axiom.description or axiom_id}",
                is_breaking=True,
            ))

        # 删除约束 (放宽约束，不破坏已有数据)
        for axiom_id in sorted(removed):
            axiom = old_axioms[axiom_id]
            changes.append(SchemaChange(
                change_type=ChangeType.CONSTRAINT_REMOVED,
                target=axiom_id,
                old_value={
                    "axiom_type": axiom.axiom_type,
                    "subject": axiom.subject,
                    "object": axiom.object,
                },
                new_value=None,
                description=f"删除约束: {axiom.description or axiom_id}",
                is_breaking=False,
            ))

        return changes

    def _diff_rules(
        self,
        old: DomainOntology,
        new: DomainOntology,
    ) -> list[SchemaChange]:
        """检测推理规则变更 (增删)."""
        changes: list[SchemaChange] = []
        old_rules = {r.rule_id: r for r in old.inference_rules}
        new_rules = {r.rule_id: r for r in new.inference_rules}

        old_ids = set(old_rules.keys())
        new_ids = set(new_rules.keys())
        added = new_ids - old_ids
        removed = old_ids - new_ids

        for rule_id in sorted(added):
            rule = new_rules[rule_id]
            changes.append(SchemaChange(
                change_type=ChangeType.RULE_ADDED,
                target=rule_id,
                old_value=None,
                new_value={
                    "rule_type": rule.rule_type.value,
                    "applies_to_relation": rule.applies_to_relation,
                },
                description=f"新增推理规则: {rule.description or rule_id}",
                is_breaking=False,
            ))

        for rule_id in sorted(removed):
            rule = old_rules[rule_id]
            changes.append(SchemaChange(
                change_type=ChangeType.RULE_REMOVED,
                target=rule_id,
                old_value={
                    "rule_type": rule.rule_type.value,
                    "applies_to_relation": rule.applies_to_relation,
                },
                new_value=None,
                description=f"删除推理规则: {rule.description or rule_id}",
                is_breaking=False,
            ))

        return changes

    # ============================================================
    # 内部方法: 重命名检测
    # ============================================================

    @staticmethod
    def _detect_class_renames(
        old_classes: dict[Any, OntologyClass],
        new_classes: dict[Any, OntologyClass],
        removed: set[Any],
        added: set[Any],
    ) -> list[tuple[Any, Any]]:
        """检测类重命名 (基于属性集合 Jaccard 相似度).

        当一个类被删除、另一个类被新增，且它们的属性集合高度重叠时，
        判定为重命名而非增删。

        Returns:
            (旧标识, 新标识) 重命名配对列表
        """
        pairs: list[tuple[Any, Any]] = []
        used_new: set[Any] = set()
        for old_key in sorted(removed, key=lambda k: str(k.value)):
            old_cls = old_classes[old_key]
            old_prop_names = {p.name for p in old_cls.properties}
            best_match: Any | None = None
            best_score = 0.0
            for new_key in added:
                if new_key in used_new:
                    continue
                new_cls = new_classes[new_key]
                new_prop_names = {p.name for p in new_cls.properties}
                # Jaccard 相似度
                union = old_prop_names | new_prop_names
                if union:
                    score = len(old_prop_names & new_prop_names) / len(union)
                else:
                    score = 0.0
                # 相似度超过 0.5 视为重命名
                if score > 0.5 and score > best_score:
                    best_match = new_key
                    best_score = score
            if best_match is not None:
                pairs.append((old_key, best_match))
                used_new.add(best_match)
        return pairs

    @staticmethod
    def _detect_property_renames(
        old_props: dict[str, OntologyProperty],
        new_props: dict[str, OntologyProperty],
        removed: set[str],
        added: set[str],
    ) -> list[tuple[str, str]]:
        """检测属性重命名 (基于数据类型匹配).

        当一个属性被删除、另一个属性被新增，且它们的数据类型相同时，
        判定为重命名。

        Returns:
            (旧名称, 新名称) 重命名配对列表
        """
        pairs: list[tuple[str, str]] = []
        used_new: set[str] = set()
        for old_name in sorted(removed):
            old_p = old_props[old_name]
            old_type = old_p.data_type.value if old_p.data_type else old_p.range
            best_match: str | None = None
            for new_name in sorted(added):
                if new_name in used_new:
                    continue
                new_p = new_props[new_name]
                new_type = new_p.data_type.value if new_p.data_type else new_p.range
                if old_type == new_type:
                    best_match = new_name
                    break  # 取第一个类型匹配的
            if best_match is not None:
                pairs.append((old_name, best_match))
                used_new.add(best_match)
        return pairs

    @staticmethod
    def _relation_modified(old_r: OntologyRelation, new_r: OntologyRelation) -> bool:
        """判断关系是否被修改 (定义域/值域/逆关系/传递性/对称性/函数性)."""
        old_domain = [d.value for d in old_r.domain]
        new_domain = [d.value for d in new_r.domain]
        old_range = [r.value for r in old_r.range]
        new_range = [r.value for r in new_r.range]
        return (
            old_r.inverse_of != new_r.inverse_of
            or old_r.transitive != new_r.transitive
            or old_r.symmetric != new_r.symmetric
            or old_r.functional != new_r.functional
            or old_domain != new_domain
            or old_range != new_range
        )

    # ============================================================
    # 内部方法: 兼容性判定与步骤生成
    # ============================================================

    def _determine_compatibility(
        self,
        changes: list[SchemaChange],
    ) -> CompatibilityLevel:
        """根据变更列表判定向后兼容性等级.

        判定规则:
        - 无变更 -> FULL
        - 无破坏性变更 + 仅有新增 -> FULL
        - 无破坏性变更 + 存在重命名 -> PARTIAL
        - 1 项破坏性变更 -> BREAKING
        - 2+ 项破坏性变更 -> INCOMPATIBLE

        Args:
            changes: 变更列表

        Returns:
            兼容性等级
        """
        if not changes:
            return CompatibilityLevel.FULL

        breaking_count = sum(1 for c in changes if c.is_breaking)

        if breaking_count == 0:
            # 检查是否存在重命名
            has_renames = any(
                c.change_type in (ChangeType.CLASS_RENAMED, ChangeType.PROPERTY_RENAMED)
                for c in changes
            )
            if has_renames:
                return CompatibilityLevel.PARTIAL
            return CompatibilityLevel.FULL

        if breaking_count == 1:
            return CompatibilityLevel.BREAKING

        # 多项破坏性变更，无法自动迁移
        return CompatibilityLevel.INCOMPATIBLE

    def _generate_steps(self, changes: list[SchemaChange]) -> list[MigrationStep]:
        """根据模式变更列表生成有序迁移步骤.

        Args:
            changes: 变更列表

        Returns:
            迁移步骤列表
        """
        steps: list[MigrationStep] = []
        for i, change in enumerate(changes):
            step = self._change_to_step(change, i)
            steps.append(step)
        return steps

    def _change_to_step(self, change: SchemaChange, index: int) -> MigrationStep:
        """将单项变更转换为迁移步骤.

        Args:
            change: 模式变更
            index: 步骤序号

        Returns:
            迁移步骤
        """
        action, rollback_action, reversible = self._CHANGE_ACTION_MAP.get(
            change.change_type,
            ("transform_data", "transform_data", False),
        )
        step_id = f"step-{index:04d}-{change.change_type.value}"

        # 构建参数
        params: dict[str, Any] = {"target": change.target}
        if change.old_value is not None:
            params["old_value"] = change.old_value
        if change.new_value is not None:
            params["new_value"] = change.new_value

        # 重命名类操作需要 old_name / new_name
        if change.change_type in (ChangeType.CLASS_RENAMED, ChangeType.PROPERTY_RENAMED):
            params["old_name"] = change.old_value
            params["new_name"] = change.new_value

        return MigrationStep(
            step_id=step_id,
            description=change.description or f"{action}: {change.target}",
            action=action,
            parameters=params,
            is_reversible=reversible,
            rollback_action=rollback_action,
        )

    def _build_summary(self, changes: list[SchemaChange]) -> str:
        """构建变更摘要文本.

        Args:
            changes: 变更列表

        Returns:
            摘要文本
        """
        if not changes:
            return "无变更"

        # 按变更类型统计
        counts: dict[str, int] = {}
        for c in changes:
            ct = c.change_type.value
            counts[ct] = counts.get(ct, 0) + 1

        parts = [f"{count} 项 {name}" for name, count in sorted(counts.items())]
        breaking = sum(1 for c in changes if c.is_breaking)

        summary = f"共 {len(changes)} 项变更 ({', '.join(parts)})"
        if breaking:
            summary += f"，其中 {breaking} 项为破坏性变更"
        return summary

    # ============================================================
    # 内部方法: 序列化辅助
    # ============================================================

    @staticmethod
    def _class_to_dict(cls: OntologyClass) -> dict[str, Any]:
        """将本体类序列化为字典 (用于变更记录)."""
        return {
            "class_id": cls.class_id,
            "entity_type": cls.entity_type.value,
            "display_name": cls.display_name,
            "description": cls.description,
            "parent_type": cls.parent_type.value if cls.parent_type else None,
            "properties": [p.name for p in cls.properties],
            "allowed_relations": [r.value for r in cls.allowed_relations],
        }

    @staticmethod
    def _prop_to_dict(prop: OntologyProperty) -> dict[str, Any]:
        """将本体属性序列化为字典 (用于变更记录)."""
        return {
            "name": prop.name,
            "property_type": prop.property_type,
            "range": prop.range,
            "required": prop.required,
            "cardinality": prop.cardinality,
            "data_type": prop.data_type.value if prop.data_type else None,
        }

    @staticmethod
    def _relation_to_dict(rel: OntologyRelation) -> dict[str, Any]:
        """将本体关系序列化为字典 (用于变更记录)."""
        return {
            "name": rel.name,
            "display_name": rel.display_name,
            "domain": [d.value for d in rel.domain],
            "range": [r.value for r in rel.range],
            "inverse_of": rel.inverse_of,
            "transitive": rel.transitive,
            "symmetric": rel.symmetric,
            "functional": rel.functional,
        }

    @staticmethod
    def _collect_all_properties(ontology: DomainOntology) -> dict[str, OntologyProperty]:
        """收集本体中所有属性 (全局属性 + 各类内属性).

        同名属性以首次出现为准 (类属性优先于全局属性)。

        Args:
            ontology: 领域本体

        Returns:
            属性名 -> 属性定义 的映射
        """
        props: dict[str, OntologyProperty] = {}
        # 先收集类内属性
        for cls in ontology.classes:
            for p in cls.properties:
                if p.name not in props:
                    props[p.name] = p
        # 再补充全局属性
        for p in ontology.global_properties:
            if p.name not in props:
                props[p.name] = p
        return props

    # ============================================================
    # 内部方法: 迁移执行
    # ============================================================

    def _execute_step(self, step: MigrationStep) -> dict[str, Any]:
        """执行单个迁移步骤 (模拟操作).

        在真实系统中，此处会对接存储引擎执行实际的模式变更。
        当前实现为模拟操作，记录日志并返回结果。

        Args:
            step: 迁移步骤

        Returns:
            步骤执行输出

        Raises:
            SchemaEvolutionError: 未知迁移动作
        """
        if step.action not in self._VALID_ACTIONS:
            raise SchemaEvolutionError(
                f"未知的迁移动作: {step.action}",
                context={"action": step.action, "step_id": step.step_id},
            )
        target = step.parameters.get("target", "")
        logger.debug("[迁移] 执行步骤 %s: %s (%s)", step.step_id, step.action, target)
        # 模拟操作: 真实系统中会调用存储引擎 API
        return {
            "action": step.action,
            "target": target,
            "executed": True,
        }

    def _execute_rollback(self, step: MigrationStep) -> dict[str, Any]:
        """执行单个步骤的回滚 (模拟操作).

        Args:
            step: 需要回滚的迁移步骤

        Returns:
            回滚执行输出
        """
        rollback_action = step.rollback_action or f"reverse_{step.action}"
        target = step.parameters.get("target", "")
        logger.debug("[回滚] 执行步骤 %s: %s (%s)", step.step_id, rollback_action, target)
        # 模拟操作: 真实系统中会调用存储引擎 API 执行逆向操作
        return {
            "action": rollback_action,
            "target": target,
            "executed": True,
        }

    def _get_ontology_snapshot(self, domain: str, version: str) -> DomainOntology:
        """获取指定版本的本体快照.

        Args:
            domain: 领域标识
            version: 版本号

        Returns:
            本体快照

        Raises:
            SchemaEvolutionError: 版本快照未找到
        """
        snapshots = self._snapshots.get(domain, {})
        ontology = snapshots.get(version)
        if ontology is None:
            raise SchemaEvolutionError(
                f"版本快照未找到: domain={domain}, version={version}",
                context={"domain": domain, "version": version},
            )
        return ontology


__all__ = [
    "ChangeType",
    "CompatibilityLevel",
    "SchemaChange",
    "SchemaDiff",
    "SchemaEvolutionError",
    "SchemaEvolutionManager",
    "SchemaVersion",
    "MigrationPlan",
    "MigrationStep",
]
