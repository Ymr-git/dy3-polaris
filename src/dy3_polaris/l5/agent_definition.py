"""Agent 定义与注册模块 — L5 Agent Runtime 核心组件.

融合世界先进方案:
- LangGraph: 有状态节点 + 条件边 + 检查点
- OpenAI Agents SDK: Agent Card + Handoff 机制
- Google ADK: DAG 任务分解 + Agent 注册
- CrewAI: 角色化 Agent 定义
- AutoGen: 消息传递 + Agent 注册表

本模块实现:
1. AgentDefinition — Agent 注册表定义模型 (Pydantic v2)
2. AgentRegistry — Agent 注册中心 (多维度索引)
3. PromptVersionManager — Prompt 版本管理 (A/B 测试 + 回滚 + 模板渲染)
4. AgentFactory — 六步实例化流水线 (创建 → Prompt → 工具 → 广播 → Session → Kernel)
5. AgentInstance — 运行时实例与生命周期管理 (READY → ACTIVE → PAUSED → TERMINATED)
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# 统一 ID 命名空间 (单点: shared/ids.py)
from dy3_polaris.shared.ids import new_session_id

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================


class BroadcastMode(str, Enum):
    """广播模式 (借鉴 LangGraph 事件总线 + Redis Pub/Sub)."""

    PUB = "pub"        # 发布
    SUB = "sub"        # 订阅
    PUBSUB = "pubsub"  # 发布+订阅


class AgentInstanceState(str, Enum):
    """Agent 实例运行时状态 (借鉴 LangGraph 状态机 + OpenAI Agents SDK 生命周期).

    READY:      就绪 — 实例化完成，等待激活
    ACTIVE:     活跃 — 正在执行任务
    PAUSED:     暂停 — 被暂停，可恢复
    TERMINATED: 已终止 — 资源已释放，不可恢复
    ERROR:      错误 — 运行时异常，需人工干预
    """

    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ERROR = "error"


class FactoryStep(str, Enum):
    """AgentFactory 六步流水线步骤 (对应 L5 设计文档 2.2 节)."""

    CREATE_INSTANCE = "create_instance"       # 1. 创建运行实例
    BIND_PROMPT = "bind_prompt"               # 2. 绑定 Prompt 版本
    BIND_TOOLS = "bind_tools"                 # 3. 绑定工具集
    BIND_BROADCAST = "bind_broadcast"         # 4. 绑定学情广播订阅
    INJECT_SESSION = "inject_session"         # 5. 注入 Working Session
    START_KERNEL = "start_kernel"             # 6. 启动 Persistent Kernel
    RECORD_PROVENANCE = "record_provenance"   # 附加: 写入 Provenance Ledger


# ============================================================
# 异常定义
# ============================================================


class AgentRegistryError(Exception):
    """Agent 注册中心错误基类."""

    def __init__(self, code: str, detail: str = "", context: dict[str, Any] | None = None) -> None:
        self.code = code
        self.detail = detail
        self.context = context or {}
        parts = [f"[{code}]"]
        if detail:
            parts.append(detail)
        super().__init__(" ".join(parts))


class AgentNotFoundError(AgentRegistryError):
    """Agent 不存在."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(
            "AGENT_NOT_FOUND",
            f"Agent not found: {agent_id}",
            {"agent_id": agent_id},
        )
        self.agent_id = agent_id


class AgentAlreadyExistsError(AgentRegistryError):
    """Agent 已存在."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(
            "AGENT_ALREADY_EXISTS",
            f"Agent already registered: {agent_id}",
            {"agent_id": agent_id},
        )
        self.agent_id = agent_id


class PromptVersionError(Exception):
    """Prompt 版本管理错误基类."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"[{code}] {detail}" if detail else f"[{code}]")


class FactoryError(Exception):
    """AgentFactory 实例化错误."""

    def __init__(
        self,
        step: FactoryStep,
        detail: str = "",
        agent_id: str = "",
    ) -> None:
        self.step = step
        self.agent_id = agent_id
        parts = [f"[FACTORY_ERROR:{step.value}]"]
        if detail:
            parts.append(detail)
        if agent_id:
            parts.append(f"(agent={agent_id})")
        super().__init__(" ".join(parts))


# ============================================================
# 数据模型 — Agent 定义
# ============================================================

_VALID_STORES = {"milvus", "neo4j", "postgresql"}
_VALID_KERNEL_TYPES = {"python", "r"}
_AGENT_ID_PATTERN = re.compile(r"^agent\.[a-z]+\.[a-z_]+$")
_VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")


class PromptReference(BaseModel):
    """版本化 Prompt 引用 (借鉴 OpenAI Agents SDK Agent Card)."""

    template_id: str = Field(..., min_length=1, description="模板 ID")
    version: str = Field(..., pattern=r"^v\d+\.\d+\.\d+$", description="语义化版本号")

    model_config = {"frozen": True}


class MemoryConfig(BaseModel):
    """记忆配置 — 读/写权限控制 (借鉴 LangGraph 状态存储)."""

    read_stores: list[str] = Field(
        default_factory=list,
        description="可读记忆存储列表",
    )
    write_stores: list[str] = Field(
        default_factory=list,
        description="可写记忆存储列表",
    )

    model_config = {"frozen": False}

    @field_validator("read_stores", "write_stores")
    @classmethod
    def validate_stores(cls, v: list[str]) -> list[str]:
        for store in v:
            if store not in _VALID_STORES:
                raise ValueError(
                    f"Invalid memory store '{store}'. Must be one of {_VALID_STORES}"
                )
        return v


class ReputationConfig(BaseModel):
    """声誉配置 (借鉴 AutoGen Agent 信誉体系)."""

    initial_score: float = Field(
        default=80.0, ge=0.0, le=100.0, description="初始声誉分数"
    )
    penalty_factor: float = Field(
        default=1.0, ge=0.0, le=5.0, description="惩罚系数"
    )
    reward_factor: float = Field(
        default=1.0, ge=0.0, le=5.0, description="奖励系数"
    )

    model_config = {"frozen": False}


class BroadcastChannel(BaseModel):
    """广播频道配置 (借鉴 LangGraph 事件总线)."""

    channel: str = Field(..., min_length=1, description="频道名称")
    mode: BroadcastMode = Field(default=BroadcastMode.SUB, description="广播模式")

    model_config = {"frozen": True}


class KernelBinding(BaseModel):
    """持久内核绑定配置 (借鉴 Jupyter Kernel + L5 Persistent Kernel 设计)."""

    kernel_type: str = Field(..., description="内核类型 (python/r)")
    purpose: str = Field(default="", description="内核用途描述")

    model_config = {"frozen": True}

    @field_validator("kernel_type")
    @classmethod
    def validate_kernel_type(cls, v: str) -> str:
        if v not in _VALID_KERNEL_TYPES:
            raise ValueError(
                f"Invalid kernel type '{v}'. Must be one of {_VALID_KERNEL_TYPES}"
            )
        return v


class DecisionAuthority(BaseModel):
    """决策权限配置 (借鉴 TDP Supervisor 权限模型).

    仅导学决策 Agent 拥有完整权限。
    """

    scheduling: bool = Field(default=False, description="调度权 — 可调度其他 Agent")
    intervention: bool = Field(default=False, description="干预权 — 可中断/重定向任务")
    adaptive: bool = Field(default=False, description="自适应权 — 可修改自身策略")

    model_config = {"frozen": False}


class SelfEvolutionConfig(BaseModel):
    """自演化配置 (借鉴 CrewAI Agent 自省 + OpenAI Agents SDK 自改进).

    仅导学决策 Agent 启用自演化。
    """

    enabled: bool = Field(default=False, description="是否启用自演化")
    prompt_template_management: bool = Field(
        default=False, description="Prompt 模板管理权"
    )
    strategy_revision: bool = Field(
        default=False, description="策略修订权"
    )
    reflection_integration: bool = Field(
        default=False, description="反思整合权"
    )

    model_config = {"frozen": False}


class AgentDefinition(BaseModel):
    """Agent 注册表定义模型 (L5 设计文档 2.1 节).

    融合世界先进方案:
    - OpenAI Agents SDK: Agent Card 规范化定义
    - Google ADK: Agent 注册 + 能力声明
    - CrewAI: 角色化 (role) 定义
    - AutoGen: Agent 注册表 + 能力描述

    每个 Agent 在注册表中包含完整元数据:
    - 基础信息 (id, name, role)
    - Prompt 引用 (template_id + version)
    - 工具集 (从 Tool Registry 发现)
    - 记忆配置 (读/写权限)
    - 声誉配置 (初始分、惩罚/奖励系数)
    - 广播频道 (发布/订阅)
    - 内核绑定 (Python/R 持久内核)
    - 决策权限 (仅决策中枢)
    - 自演化配置 (仅决策中枢)
    """

    id: str = Field(
        ...,
        pattern=r"^agent\.[a-z]+\.[a-z_]+$",
        description="Agent 唯一标识，格式 agent.{domain}.{name}",
    )
    name: str = Field(..., min_length=2, max_length=64, description="人类可读名称")
    role: str = Field(..., min_length=10, max_length=512, description="角色描述")
    system_prompt: PromptReference = Field(..., description="版本化 Prompt 引用")
    tools: list[str] = Field(
        ...,
        min_length=1,
        description="绑定的工具 ID 列表",
    )
    connectors: list[str] = Field(
        default_factory=list,
        description="L3 Connector 引用列表",
    )
    memory_config: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="记忆配置",
    )
    reputation_config: ReputationConfig = Field(
        default_factory=ReputationConfig,
        description="声誉配置",
    )
    broadcast_channels: list[BroadcastChannel] = Field(
        default_factory=list,
        description="学情广播总线订阅/发布频道列表",
    )
    kernel_bindings: list[KernelBinding] = Field(
        default_factory=list,
        max_length=2,
        description="持久内核绑定，最多 2 个 (Python + R)",
    )

    # 导学决策 Agent 专属
    decision_authority: DecisionAuthority = Field(
        default_factory=DecisionAuthority,
        description="决策权限配置",
    )
    self_evolution: SelfEvolutionConfig = Field(
        default_factory=SelfEvolutionConfig,
        description="自演化配置",
    )

    # 元数据
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    updated_at: float = Field(default_factory=time.time, description="更新时间戳")

    model_config = {"frozen": False}

    @model_validator(mode="after")
    def validate_self_evolution_requires_authority(self) -> AgentDefinition:
        """自演化启用时必须有自适应权限."""
        if self.self_evolution.enabled and not self.decision_authority.adaptive:
            raise ValueError(
                "Self-evolution requires decision_authority.adaptive=True"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (用于 JSON 持久化)."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "system_prompt": {
                "template_id": self.system_prompt.template_id,
                "version": self.system_prompt.version,
            },
            "tools": list(self.tools),
            "connectors": list(self.connectors),
            "memory_config": {
                "read_stores": list(self.memory_config.read_stores),
                "write_stores": list(self.memory_config.write_stores),
            },
            "reputation_config": {
                "initial_score": self.reputation_config.initial_score,
                "penalty_factor": self.reputation_config.penalty_factor,
                "reward_factor": self.reputation_config.reward_factor,
            },
            "broadcast_channels": [
                {"channel": bc.channel, "mode": bc.mode.value}
                for bc in self.broadcast_channels
            ],
            "kernel_bindings": [
                {"type": kb.kernel_type, "purpose": kb.purpose}
                for kb in self.kernel_bindings
            ],
            "decision_authority": {
                "scheduling": self.decision_authority.scheduling,
                "intervention": self.decision_authority.intervention,
                "adaptive": self.decision_authority.adaptive,
            },
            "self_evolution": {
                "enabled": self.self_evolution.enabled,
                "prompt_template_management": self.self_evolution.prompt_template_management,
                "strategy_revision": self.self_evolution.strategy_revision,
                "reflection_integration": self.self_evolution.reflection_integration,
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentDefinition:
        """从字典反序列化."""
        # 处理嵌套对象
        kwargs: dict[str, Any] = dict(data)
        if "system_prompt" in kwargs and isinstance(kwargs["system_prompt"], dict):
            kwargs["system_prompt"] = PromptReference(**kwargs["system_prompt"])
        if "memory_config" in kwargs and isinstance(kwargs["memory_config"], dict):
            kwargs["memory_config"] = MemoryConfig(**kwargs["memory_config"])
        if "reputation_config" in kwargs and isinstance(kwargs["reputation_config"], dict):
            kwargs["reputation_config"] = ReputationConfig(**kwargs["reputation_config"])
        if "broadcast_channels" in kwargs:
            bcs = kwargs["broadcast_channels"]
            if isinstance(bcs, list) and bcs and isinstance(bcs[0], dict):
                kwargs["broadcast_channels"] = [BroadcastChannel(**bc) for bc in bcs]
        if "kernel_bindings" in kwargs:
            kbs = kwargs["kernel_bindings"]
            if isinstance(kbs, list) and kbs and isinstance(kbs[0], dict):
                # 兼容 "type" 字段名
                processed = []
                for kb in kbs:
                    kb_data = dict(kb)
                    if "type" in kb_data and "kernel_type" not in kb_data:
                        kb_data["kernel_type"] = kb_data.pop("type")
                    processed.append(KernelBinding(**kb_data))
                kwargs["kernel_bindings"] = processed
        if "decision_authority" in kwargs and isinstance(kwargs["decision_authority"], dict):
            kwargs["decision_authority"] = DecisionAuthority(**kwargs["decision_authority"])
        if "self_evolution" in kwargs and isinstance(kwargs["self_evolution"], dict):
            kwargs["self_evolution"] = SelfEvolutionConfig(**kwargs["self_evolution"])
        return cls(**kwargs)


# ============================================================
# Prompt 版本管理
# ============================================================


class PromptVersion(BaseModel):
    """Prompt 版本记录 (L5 设计文档 2.3 节).

    融合世界先进方案:
    - OpenAI Agents SDK: Agent Card 的 instructions 版本管理
    - LangGraph: Prompt 模板 + 变量注入
    - CrewAI: 角色化 Prompt 模板

    支持:
    - 语义化版本号 (v{major}.{minor}.{patch})
    - A/B 测试分组 (ab_group: A/B/None)
    - 版本激活/停用
    - 模板变量渲染 (str.format 风格)
    """

    template_id: str = Field(..., min_length=1, description="模板 ID")
    version: str = Field(
        ...,
        pattern=r"^v\d+\.\d+\.\d+$",
        description="语义化版本号 v{major}.{minor}.{patch}",
    )
    content: str = Field(..., min_length=1, description="Prompt 模板内容")
    changelog: str = Field(default="", description="变更日志")
    created_by: str = Field(default="system", description="创建者")
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    is_active: bool = Field(default=True, description="是否活跃")
    ab_group: str | None = Field(
        default=None,
        description="A/B 测试分组 (A/B/None)",
    )

    model_config = {"frozen": False}

    @field_validator("ab_group")
    @classmethod
    def validate_ab_group(cls, v: str | None) -> str | None:
        if v is not None and v not in ("A", "B"):
            raise ValueError(f"ab_group must be 'A', 'B', or None, got '{v}'")
        return v

    @property
    def version_tuple(self) -> tuple[int, int, int]:
        """解析版本号为 (major, minor, patch) 元组."""
        match = _VERSION_PATTERN.match(self.version)
        if match:
            parts = self.version[1:].split(".")
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        return (0, 0, 0)


class _SafeFormatDict(dict):
    """安全格式化字典 — 缺失的键返回占位符而非报错.

    实现 __missing__ 使得 str.format_map 在遇到未提供的变量时
    保留 {var_name} 原文，支持延迟绑定。
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class PromptVersionManager:
    """Prompt 版本管理器 (L5 设计文档 2.3 节).

    核心能力:
    1. 版本注册与查询 (CRUD)
    2. 活跃版本管理 (自动激活最新版本)
    3. A/B 测试分组 (确定性哈希分配)
    4. 版本回滚 (停用新版本，激活旧版本)
    5. 模板渲染 (注入上下文变量)

    线程安全: 所有公共方法均受 threading.RLock 保护。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 存储: (template_id, version) -> PromptVersion
        self._versions: dict[tuple[str, str], PromptVersion] = {}
        # 活跃版本索引: template_id -> version
        self._active_version: dict[str, str] = {}
        # A/B 测试分组: template_id -> {"A": version, "B": version}
        self._ab_groups: dict[str, dict[str, str]] = {}

    def register(self, prompt_version: PromptVersion) -> PromptVersion:
        """注册新的 Prompt 版本.

        如果是同 template_id 的第一个版本，自动设为活跃。
        如果已有活跃版本，新版本自动设为活跃（替代旧版本）。
        """
        with self._lock:
            key = (prompt_version.template_id, prompt_version.version)
            if key in self._versions:
                logger.warning(
                    f"Overwriting existing prompt version: "
                    f"{prompt_version.template_id}@{prompt_version.version}"
                )

            self._versions[key] = prompt_version

            # 自动激活最新版本
            tid = prompt_version.template_id
            current_active = self._active_version.get(tid)
            if current_active is None:
                # 第一个版本，直接激活
                self._active_version[tid] = prompt_version.version
            else:
                # 比较版本号，如果新版本更高则激活
                current_pv = self._versions.get((tid, current_active))
                if current_pv and prompt_version.version_tuple > current_pv.version_tuple:
                    # 停用旧版本
                    current_pv.is_active = False
                    self._active_version[tid] = prompt_version.version

            # 注册 A/B 测试分组
            if prompt_version.ab_group is not None:
                tid = prompt_version.template_id
                if tid not in self._ab_groups:
                    self._ab_groups[tid] = {}
                self._ab_groups[tid][prompt_version.ab_group] = prompt_version.version

            logger.info(
                f"Registered prompt version: {prompt_version.template_id}@{prompt_version.version}"
            )
            return prompt_version

    def get(self, template_id: str, version: str) -> PromptVersion | None:
        """获取指定版本的 Prompt."""
        with self._lock:
            return self._versions.get((template_id, version))

    def get_active(self, template_id: str) -> PromptVersion | None:
        """获取当前活跃版本的 Prompt."""
        with self._lock:
            version = self._active_version.get(template_id)
            if version is None:
                return None
            return self._versions.get((template_id, version))

    def list_versions(self, template_id: str) -> list[PromptVersion]:
        """列出模板的所有版本 (按版本号降序排列)."""
        with self._lock:
            versions = [
                pv for (tid, _), pv in self._versions.items()
                if tid == template_id
            ]
            return sorted(versions, key=lambda pv: pv.version_tuple, reverse=True)

    def count(self, template_id: str) -> int:
        """返回模板的版本数."""
        with self._lock:
            return sum(1 for (tid, _) in self._versions if tid == template_id)

    def deactivate(self, template_id: str, version: str) -> None:
        """停用某个版本."""
        with self._lock:
            pv = self._versions.get((template_id, version))
            if pv is None:
                raise PromptVersionError(
                    "VERSION_NOT_FOUND",
                    f"Prompt version not found: {template_id}@{version}",
                )
            pv.is_active = False
            # 如果停用的是活跃版本，需要找下一个活跃版本
            if self._active_version.get(template_id) == version:
                all_versions = self.list_versions(template_id)
                # 找到下一个活跃版本
                for next_pv in all_versions:
                    if next_pv.version != version:
                        next_pv.is_active = True
                        self._active_version[template_id] = next_pv.version
                        break
                else:
                    # 没有其他版本了
                    self._active_version.pop(template_id, None)

    def activate(self, template_id: str, version: str) -> None:
        """激活某个版本 (停用其他版本)."""
        with self._lock:
            pv = self._versions.get((template_id, version))
            if pv is None:
                raise PromptVersionError(
                    "VERSION_NOT_FOUND",
                    f"Prompt version not found: {template_id}@{version}",
                )
            # 停用当前活跃版本
            current = self._active_version.get(template_id)
            if current and current != version:
                current_pv = self._versions.get((template_id, current))
                if current_pv:
                    current_pv.is_active = False
            # 激活目标版本
            pv.is_active = True
            self._active_version[template_id] = version

    def get_ab_version(self, template_id: str, learner_id: str) -> PromptVersion | None:
        """根据学习者 ID 确定性分配 A/B 测试版本.

        使用 MD5 哈希确保同一学习者总是分到同一组。
        """
        with self._lock:
            groups = self._ab_groups.get(template_id)
            if not groups:
                # 没有 A/B 测试，返回活跃版本
                return self.get_active(template_id)

            # 确定性哈希分配
            hash_val = int(hashlib.md5(learner_id.encode()).hexdigest(), 16)
            group = "A" if hash_val % 2 == 0 else "B"

            version = groups.get(group)
            if version is None:
                # 如果分配的组没有版本，回退到另一组
                version = groups.get("B" if group == "A" else "A")
            if version is None:
                return self.get_active(template_id)

            return self._versions.get((template_id, version))

    def rollback(
        self,
        template_id: str,
        target_version: str,
        reason: str = "",
        operator: str = "system",
    ) -> PromptVersion:
        """版本回滚.

        停用所有比 target_version 更新的版本，激活 target_version。
        记录回滚操作日志 (用于 Provenance)。
        """
        with self._lock:
            target_pv = self._versions.get((template_id, target_version))
            if target_pv is None:
                raise PromptVersionError(
                    "VERSION_NOT_FOUND",
                    f"Cannot rollback: target version not found: {template_id}@{target_version}",
                )

            # 停用所有比目标版本更新的版本
            target_tuple = target_pv.version_tuple
            for (tid, ver), pv in self._versions.items():
                if tid == template_id and pv.version_tuple > target_tuple:
                    pv.is_active = False
                    logger.info(
                        f"Deactivated {template_id}@{ver} during rollback to {target_version}"
                    )

            # 激活目标版本
            target_pv.is_active = True
            self._active_version[template_id] = target_version

            logger.info(
                f"Rolled back {template_id} to {target_version}. "
                f"Reason: {reason}. Operator: {operator}"
            )
            return target_pv

    def render(
        self,
        template_id: str,
        version: str,
        context: dict[str, Any],
    ) -> str:
        """渲染 Prompt 模板 (注入上下文变量).

        使用 str.format 风格的变量替换。
        缺失的变量会保留为 {var_name} 占位符，不报错。
        这允许 Agent 在没有完整上下文时也能实例化（延迟绑定）。
        """
        with self._lock:
            pv = self._versions.get((template_id, version))
            if pv is None:
                raise PromptVersionError(
                    "VERSION_NOT_FOUND",
                    f"Cannot render: version not found: {template_id}@{version}",
                )
            try:
                return pv.content.format_map(_SafeFormatDict(context))
            except Exception as e:
                raise PromptVersionError(
                    "RENDER_ERROR",
                    f"Failed to render prompt: {e}",
                )


# ============================================================
# Agent 注册中心
# ============================================================


class AgentRegistry:
    """Agent 注册中心 (L5 设计文档 2.1 节).

    融合世界先进方案:
    - OpenAI Agents SDK: Agent Card 注册与发现
    - Google ADK: Agent 注册表 + 能力索引
    - AutoGen: Agent 注册表 + 消息路由
    - LangGraph: 有状态节点注册

    核心能力:
    1. Agent 注册/注销 (带 ID 唯一性校验)
    2. 多维度查询 (按 ID/工具/频道/权限)
    3. 覆盖更新 (overwrite 模式)
    4. 注册摘要导出

    线程安全: 所有公共方法均受 threading.RLock 保护。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 主存储: agent_id -> AgentDefinition
        self._agents: dict[str, AgentDefinition] = {}
        # 工具索引: tool_name -> set[agent_id]
        self._tool_index: dict[str, set[str]] = {}
        # 频道索引: channel_name -> set[agent_id]
        self._channel_index: dict[str, set[str]] = {}
        # 决策权限索引: agent_id (拥有决策权限的)
        self._decision_authority_agents: set[str] = set()

    def register(
        self,
        agent_def: AgentDefinition,
        *,
        overwrite: bool = False,
    ) -> AgentDefinition:
        """注册 Agent 定义.

        Args:
            agent_def: Agent 定义
            overwrite: 是否覆盖已存在的同 ID Agent

        Returns:
            注册的 AgentDefinition

        Raises:
            AgentAlreadyExistsError: Agent 已存在且 overwrite=False
        """
        with self._lock:
            agent_id = agent_def.id
            if agent_id in self._agents and not overwrite:
                raise AgentAlreadyExistsError(agent_id)

            # 如果覆盖，先清理旧索引
            if agent_id in self._agents:
                self._remove_from_indices(agent_id)
                old = self._agents[agent_id]
                agent_def.created_at = old.created_at
                agent_def.updated_at = time.time()

            # 存储
            self._agents[agent_id] = agent_def

            # 更新索引
            self._add_to_indices(agent_def)

            logger.info(f"Registered agent: {agent_id} ({agent_def.name})")
            return agent_def

    def get(self, agent_id: str) -> AgentDefinition | None:
        """获取 Agent 定义 (不存在返回 None)."""
        with self._lock:
            return self._agents.get(agent_id)

    def get_or_raise(self, agent_id: str) -> AgentDefinition:
        """获取 Agent 定义 (不存在抛出异常)."""
        agent = self.get(agent_id)
        if agent is None:
            raise AgentNotFoundError(agent_id)
        return agent

    def contains(self, agent_id: str) -> bool:
        """检查 Agent 是否已注册."""
        with self._lock:
            return agent_id in self._agents

    def unregister(self, agent_id: str) -> AgentDefinition:
        """注销 Agent.

        Returns:
            被移除的 AgentDefinition

        Raises:
            AgentNotFoundError: Agent 不存在
        """
        with self._lock:
            if agent_id not in self._agents:
                raise AgentNotFoundError(agent_id)
            agent_def = self._agents.pop(agent_id)
            self._remove_from_indices(agent_id)
            logger.info(f"Unregistered agent: {agent_id}")
            return agent_def

    def list_all(self) -> list[AgentDefinition]:
        """列出所有已注册 Agent."""
        with self._lock:
            return list(self._agents.values())

    def find_by_tool(self, tool_name: str) -> list[AgentDefinition]:
        """按绑定的工具查找 Agent."""
        with self._lock:
            agent_ids = self._tool_index.get(tool_name, set())
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_by_channel(self, channel_name: str) -> list[AgentDefinition]:
        """按广播频道查找 Agent."""
        with self._lock:
            agent_ids = self._channel_index.get(channel_name, set())
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_decision_authority_agents(self) -> list[AgentDefinition]:
        """查找拥有决策权限的 Agent."""
        with self._lock:
            return [
                self._agents[aid]
                for aid in self._decision_authority_agents
                if aid in self._agents
            ]

    def export_summary(self) -> dict[str, Any]:
        """导出注册中心摘要统计."""
        with self._lock:
            return {
                "total_agents": len(self._agents),
                "agent_ids": list(self._agents.keys()),
                "decision_authority_count": len(self._decision_authority_agents),
                "total_tools_indexed": sum(len(ids) for ids in self._tool_index.values()),
                "total_channels_indexed": sum(len(ids) for ids in self._channel_index.values()),
            }

    @property
    def size(self) -> int:
        """已注册 Agent 数量."""
        with self._lock:
            return len(self._agents)

    def _add_to_indices(self, agent_def: AgentDefinition) -> None:
        """添加到索引."""
        aid = agent_def.id
        for tool in agent_def.tools:
            self._tool_index.setdefault(tool, set()).add(aid)
        for bc in agent_def.broadcast_channels:
            self._channel_index.setdefault(bc.channel, set()).add(aid)
        if (
            agent_def.decision_authority.scheduling
            or agent_def.decision_authority.intervention
            or agent_def.decision_authority.adaptive
        ):
            self._decision_authority_agents.add(aid)

    def _remove_from_indices(self, agent_id: str) -> None:
        """从索引中移除."""
        agent_def = self._agents.get(agent_id)
        if agent_def is None:
            return
        for tool in agent_def.tools:
            if tool in self._tool_index:
                self._tool_index[tool].discard(agent_id)
                if not self._tool_index[tool]:
                    del self._tool_index[tool]
        for bc in agent_def.broadcast_channels:
            if bc.channel in self._channel_index:
                self._channel_index[bc.channel].discard(agent_id)
                if not self._channel_index[bc.channel]:
                    del self._channel_index[bc.channel]
        self._decision_authority_agents.discard(agent_id)

    def clear(self) -> None:
        """清空注册中心 (仅用于测试)."""
        with self._lock:
            self._agents.clear()
            self._tool_index.clear()
            self._channel_index.clear()
            self._decision_authority_agents.clear()


# ============================================================
# 运行时辅助模型
# ============================================================


class KernelHandle:
    """持久内核句柄 (模拟 L5 Persistent Kernel 的运行时句柄).

    在实际部署中，这将包装一个真实的 Python/R 内核进程。
    在当前实现中，它记录内核元数据和模拟状态。
    """

    def __init__(
        self,
        kernel_type: str,
        purpose: str,
        instance_id: str,
    ) -> None:
        self.kernel_type = kernel_type
        self.purpose = purpose
        self.instance_id = instance_id
        self.kernel_id = f"kernel-{uuid.uuid4().hex[:12]}"
        self.started_at = time.time()
        self.status = "running"
        self._variables: dict[str, Any] = {}  # 模拟跨 task 变量保留

    @property
    def is_alive(self) -> bool:
        return self.status == "running"

    def set_variable(self, name: str, value: Any) -> None:
        """设置内核变量 (跨 sub-task 保留)."""
        self._variables[name] = value

    def get_variable(self, name: str) -> Any | None:
        """获取内核变量."""
        return self._variables.get(name)

    def shutdown(self) -> None:
        """关闭内核."""
        self.status = "shutdown"
        self._variables.clear()


class WorkingSession:
    """Working Session (L2 级别的工作会话).

    借鉴 LangGraph 检查点机制，支持:
    - 会话 ID 标识
    - Checkpoint 快照
    - Fork/Merge (预留)
    """

    def __init__(
        self,
        session_id: str,
        agent_id: str,
        instance_id: str,
        learner_context: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.agent_id = agent_id
        self.instance_id = instance_id
        self.learner_context = learner_context or {}
        self.created_at = time.time()
        self.last_active_at = self.created_at
        self._checkpoints: list[dict[str, Any]] = []
        self._state: dict[str, Any] = {}

    def checkpoint(self, label: str = "") -> str:
        """创建检查点快照."""
        cp_id = f"cp-{uuid.uuid4().hex[:8]}"
        cp = {
            "cp_id": cp_id,
            "label": label,
            "timestamp": time.time(),
            "state_snapshot": dict(self._state),
        }
        self._checkpoints.append(cp)
        self.last_active_at = time.time()
        return cp_id

    def restore(self, cp_id: str) -> bool:
        """从检查点恢复状态."""
        for cp in self._checkpoints:
            if cp["cp_id"] == cp_id:
                self._state = dict(cp["state_snapshot"])
                return True
        return False

    def set_state(self, key: str, value: Any) -> None:
        """设置会话状态."""
        self._state[key] = value
        self.last_active_at = time.time()

    def get_state(self, key: str) -> Any | None:
        """获取会话状态."""
        return self._state.get(key)

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)


class BroadcastSubscription:
    """广播订阅记录 (模拟 Redis Pub/Sub 订阅)."""

    def __init__(
        self,
        channel: str,
        mode: BroadcastMode,
        subscriber_id: str,
    ) -> None:
        self.channel = channel
        self.mode = mode
        self.subscriber_id = subscriber_id
        self.active = True
        self.created_at = time.time()

    def unsubscribe(self) -> None:
        """取消订阅."""
        self.active = False


# ============================================================
# AgentInstance 运行时实例
# ============================================================


class AgentInstance:
    """Agent 运行时实例 (L5 设计文档 2.2 节).

    融合世界先进方案:
    - LangGraph: 有状态节点 + 状态机
    - OpenAI Agents SDK: Agent 实例 + Handoff
    - AutoGen: Agent 实例 + 消息处理

    生命周期: READY → ACTIVE → PAUSED → TERMINATED
    (ERROR 为异常状态，可从 ACTIVE/PAUSED 进入)

    持有:
    - 唯一 instance_id 和 session_id
    - 渲染后的 system prompt
    - 绑定的工具集
    - 广播频道订阅
    - Working Session (检查点/Fork)
    - Persistent Kernel 句柄
    - Provenance 记录
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        instance_id: str,
        session_id: str,
        rendered_prompt: str,
        bound_tools: list[str],
        broadcast_subscriptions: list[BroadcastSubscription],
        working_session: WorkingSession,
        kernel_handles: list[KernelHandle],
        provenance_record: dict[str, Any] | None = None,
        learner_context: dict[str, Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.instance_id = instance_id
        self.session_id = session_id
        self.rendered_prompt = rendered_prompt
        self.bound_tools = bound_tools
        self.broadcast_subscriptions = broadcast_subscriptions
        self.working_session = working_session
        self.kernel_handles = kernel_handles
        self.provenance_record = provenance_record
        self.learner_context = learner_context or {}

        self.state = AgentInstanceState.READY
        self.created_at = time.time()
        self.activated_at: float | None = None
        self._last_state_change = self.created_at

    def activate(self) -> None:
        """激活实例 (READY/PAUSED → ACTIVE)."""
        if self.state == AgentInstanceState.TERMINATED:
            raise AgentRegistryError(
                "INSTANCE_TERMINATED",
                f"Cannot activate terminated instance: {self.instance_id}",
            )
        if self.state == AgentInstanceState.ERROR:
            raise AgentRegistryError(
                "INSTANCE_ERROR",
                f"Cannot activate error instance: {self.instance_id}",
            )
        self.state = AgentInstanceState.ACTIVE
        self.activated_at = time.time()
        self._last_state_change = time.time()
        logger.info(f"Agent instance activated: {self.instance_id} ({self.agent_id})")

    def pause(self) -> None:
        """暂停实例 (ACTIVE → PAUSED)."""
        if self.state != AgentInstanceState.ACTIVE:
            raise AgentRegistryError(
                "INVALID_STATE_TRANSITION",
                f"Cannot pause from state {self.state.value}",
            )
        self.state = AgentInstanceState.PAUSED
        self._last_state_change = time.time()
        logger.info(f"Agent instance paused: {self.instance_id}")

    def resume(self) -> None:
        """恢复实例 (PAUSED → ACTIVE)."""
        if self.state != AgentInstanceState.PAUSED:
            raise AgentRegistryError(
                "INVALID_STATE_TRANSITION",
                f"Cannot resume from state {self.state.value}",
            )
        self.state = AgentInstanceState.ACTIVE
        self._last_state_change = time.time()
        logger.info(f"Agent instance resumed: {self.instance_id}")

    def terminate(self) -> None:
        """终止实例 (任意状态 → TERMINATED).

        释放所有资源:
        - 关闭 Persistent Kernel
        - 取消广播订阅
        """
        if self.state == AgentInstanceState.TERMINATED:
            return  # 幂等

        # 关闭内核
        for kh in self.kernel_handles:
            kh.shutdown()
        self.kernel_handles.clear()

        # 取消广播订阅
        for sub in self.broadcast_subscriptions:
            sub.unsubscribe()
        self.broadcast_subscriptions.clear()

        self.state = AgentInstanceState.TERMINATED
        self._last_state_change = time.time()
        logger.info(f"Agent instance terminated: {self.instance_id}")

    def mark_error(self, error_msg: str = "") -> None:
        """标记为错误状态."""
        self.state = AgentInstanceState.ERROR
        self._last_state_change = time.time()
        logger.error(f"Agent instance error: {self.instance_id} - {error_msg}")

    def health_check(self) -> dict[str, Any]:
        """健康检查."""
        uptime = time.time() - (self.activated_at or self.created_at)
        kernel_alive = all(kh.is_alive for kh in self.kernel_handles) if self.kernel_handles else True
        subs_active = all(s.active for s in self.broadcast_subscriptions) if self.broadcast_subscriptions else True

        return {
            "instance_id": self.instance_id,
            "agent_id": self.agent_id,
            "state": self.state.value,
            "healthy": (
                self.state in (AgentInstanceState.ACTIVE, AgentInstanceState.READY)
                and kernel_alive
                and subs_active
            ),
            "uptime_s": round(uptime, 2),
            "kernel_count": len(self.kernel_handles),
            "kernels_alive": sum(1 for kh in self.kernel_handles if kh.is_alive),
            "active_subscriptions": sum(1 for s in self.broadcast_subscriptions if s.active),
            "bound_tools_count": len(self.bound_tools),
            "checkpoint_count": self.working_session.checkpoint_count,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """实例元数据."""
        result = {
            "instance_id": self.instance_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "activated_at": self.activated_at,
            "bound_tools": list(self.bound_tools),
            "kernel_count": len(self.kernel_handles),
            "learner_context": dict(self.learner_context),
        }
        # 扁平化 learner_context 中的键到顶层，便于直接访问
        for k, v in self.learner_context.items():
            result[k] = v
        return result


# ============================================================
# AgentFactory 六步实例化流水线
# ============================================================


class AgentFactory:
    """Agent 工厂 (L5 设计文档 2.2 节).

    实现 6 步实例化流水线:
    1. 创建运行实例 — 从注册表读取定义，分配 instance_id 和 session_id
    2. 绑定 Prompt 版本 — 加载模板，注入上下文变量
    3. 绑定工具集 — 从 Tool Registry 加载，校验权限
    4. 绑定学情广播订阅 — 连接广播频道
    5. 注入 Working Session — 创建/复用会话
    6. 启动 Persistent Kernel — 启动 Python/R 内核

    附加: 记录 Provenance Ledger

    融合世界先进方案:
    - OpenAI Agents SDK: Agent 实例化 + 工具绑定
    - LangGraph: 有状态节点初始化 + 检查点
    - Google ADK: Agent Factory 模式
    - CrewAI: Agent 初始化 + 工具分配
    """

    def __init__(
        self,
        tool_registry: Any | None = None,
        prompt_manager: PromptVersionManager | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._prompt_manager = prompt_manager or PromptVersionManager()
        self._instances: dict[str, AgentInstance] = {}
        self._lock = threading.RLock()

    async def instantiate(
        self,
        agent_id: str,
        registry: AgentRegistry,
        learner_context: dict[str, Any] | None = None,
    ) -> AgentInstance:
        """执行 6 步实例化流水线.

        Args:
            agent_id: Agent 注册 ID
            registry: Agent 注册中心
            learner_context: 学习者上下文变量 (用于 Prompt 渲染)

        Returns:
            就绪的 AgentInstance

        Raises:
            AgentNotFoundError: Agent 未注册
            FactoryError: 实例化某步失败
        """
        learner_context = learner_context or {}

        # Step 1: 创建运行实例
        try:
            agent_def = registry.get_or_raise(agent_id)
        except AgentNotFoundError:
            raise

        instance_id = f"inst-{uuid.uuid4().hex[:12]}"
        # 统一命名空间: ws- (Agent 实例工作会话)
        session_id = new_session_id("ws")

        logger.info(
            f"[Factory] Step 1: Creating instance {instance_id} for agent {agent_id}"
        )

        # Step 2: 绑定 Prompt 版本
        try:
            prompt_ref = agent_def.system_prompt
            prompt_version = self._prompt_manager.get(
                prompt_ref.template_id,
                prompt_ref.version,
            )
            if prompt_version is None:
                raise FactoryError(
                    FactoryStep.BIND_PROMPT,
                    f"Prompt version not found: {prompt_ref.template_id}@{prompt_ref.version}",
                    agent_id=agent_id,
                )
            # 渲染 Prompt (注入上下文变量)
            rendered_prompt = self._prompt_manager.render(
                prompt_ref.template_id,
                prompt_ref.version,
                learner_context,
            )
        except FactoryError:
            raise
        except Exception as e:
            raise FactoryError(
                FactoryStep.BIND_PROMPT,
                f"Failed to bind prompt: {e}",
                agent_id=agent_id,
            ) from e

        logger.info(
            f"[Factory] Step 2: Bound prompt {prompt_ref.template_id}@{prompt_ref.version}"
        )

        # Step 3: 绑定工具集
        bound_tools = list(agent_def.tools)
        # 如果有 ToolRegistry，校验工具是否存在
        if self._tool_registry is not None:
            for tool_name in bound_tools:
                entry = self._tool_registry.get(tool_name)
                if entry is None:
                    logger.warning(
                        f"[Factory] Tool '{tool_name}' not found in registry "
                        f"(agent={agent_id}). Binding as stub."
                    )

        logger.info(f"[Factory] Step 3: Bound {len(bound_tools)} tools")

        # Step 4: 绑定学情广播订阅
        broadcast_subs: list[BroadcastSubscription] = []
        for bc in agent_def.broadcast_channels:
            sub = BroadcastSubscription(
                channel=bc.channel,
                mode=bc.mode,
                subscriber_id=instance_id,
            )
            broadcast_subs.append(sub)

        logger.info(f"[Factory] Step 4: Bound {len(broadcast_subs)} broadcast channels")

        # Step 5: 注入 Working Session
        working_session = WorkingSession(
            session_id=session_id,
            agent_id=agent_id,
            instance_id=instance_id,
            learner_context=learner_context,
        )
        # 创建初始检查点
        working_session.checkpoint(label="initial")

        logger.info(f"[Factory] Step 5: Created working session {session_id}")

        # Step 6: 启动 Persistent Kernel
        kernel_handles: list[KernelHandle] = []
        for kb in agent_def.kernel_bindings:
            kh = KernelHandle(
                kernel_type=kb.kernel_type,
                purpose=kb.purpose,
                instance_id=instance_id,
            )
            kernel_handles.append(kh)

        logger.info(f"[Factory] Step 6: Started {len(kernel_handles)} kernels")

        # 附加: 记录 Provenance
        provenance_record = {
            "event_type": "agent_instantiated",
            "agent_id": agent_id,
            "instance_id": instance_id,
            "session_id": session_id,
            "timestamp": time.time(),
            "factory_steps": [step.value for step in FactoryStep],
            "tools_bound": bound_tools,
            "kernel_count": len(kernel_handles),
            "broadcast_channels": [bc.channel for bc in agent_def.broadcast_channels],
        }

        # 创建实例
        instance = AgentInstance(
            agent_id=agent_id,
            agent_name=agent_def.name,
            instance_id=instance_id,
            session_id=session_id,
            rendered_prompt=rendered_prompt,
            bound_tools=bound_tools,
            broadcast_subscriptions=broadcast_subs,
            working_session=working_session,
            kernel_handles=kernel_handles,
            provenance_record=provenance_record,
            learner_context=learner_context,
        )

        # 注册到实例池
        with self._lock:
            self._instances[instance_id] = instance

        logger.info(
            f"[Factory] Agent instance ready: {instance_id} "
            f"({agent_id}, state={instance.state.value})"
        )

        return instance

    def get_instance(self, instance_id: str) -> AgentInstance | None:
        """获取已创建的实例."""
        with self._lock:
            return self._instances.get(instance_id)

    def list_instances(self) -> list[AgentInstance]:
        """列出所有实例."""
        with self._lock:
            return list(self._instances.values())

    def terminate_instance(self, instance_id: str) -> bool:
        """终止实例."""
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                return False
            instance.terminate()
            return True

    @property
    def instance_count(self) -> int:
        """当前实例数."""
        with self._lock:
            return len(self._instances)
