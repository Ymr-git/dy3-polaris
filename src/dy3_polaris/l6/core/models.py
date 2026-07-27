"""L6 数据模型.

基于 Pydantic v2 定义 Dy3+ Polaris L6 层的核心数据结构。
复用 MCP SDK 的 mcp.types 模型，扩展 Dy3+ 特有字段。
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# 工具分类与层级标签
# ============================================================

class LayerTag(str, Enum):
    """架构层标签，用于工具/资源的 annotations.tags."""
    L0_GOVERNANCE = "L0"
    L1_USER_DOMAIN = "L1"
    L2_PERSONALIZATION = "L2"
    L3_DOMAIN_KNOWLEDGE = "L3"
    L4_DECISION_ENGINE = "L4"
    L5_AGENT_RUNTIME = "L5"
    L6_PROTOCOL = "L6"
    L7_EXPERIENCE = "L7"
    CC1_ANTI_HALLUCINATION = "CC1"
    CC2_HUMAN_COLLAB = "CC2"
    CC3_PROVENANCE = "CC3"


class ToolCategory(str, Enum):
    """工具分类."""
    INTERNAL = "internal"           # 内部计算工具
    CONNECTOR_TIER1 = "connector_tier1"  # 公共数据连接器
    CONNECTOR_TIER2 = "connector_tier2"  # 行业数据连接器
    CONNECTOR_TIER3 = "connector_tier3"  # 校园数据连接器
    SKILLBOOK = "skillbook"         # L2 教学技能
    EXTERNAL = "external"           # 外部工具


class ResourceType(str, Enum):
    """资源类型."""
    LEARNER_PROFILE = "learner_profile"       # 学情画像
    KNOWLEDGE_GRAPH = "knowledge_graph"       # 知识图谱节点
    LEARNING_PATH = "learning_path"           # 学习路径
    ASSESSMENT_RESULT = "assessment_result"   # 评估结果
    LEARNING_RESOURCE = "learning_resource"   # 学习资源
    PROVENANCE = "provenance"                 # 溯源数据


# ============================================================
# Dy3+ 扩展的 Tool Annotations
# ============================================================

class Dy3ToolAnnotations(BaseModel):
    """Dy3+ 扩展的 MCP Tool Annotations.

    在 MCP 标准 annotations 基础上增加 Dy3+ 特有字段。
    """

    tags: list[str] = Field(
        default_factory=list,
        description="标签数组，如 ['bkt', 'personalization', 'L2']",
    )
    mime_type: str = Field(
        default="application/json",
        description="输出 MIME 类型",
    )
    readonly_hint: bool = Field(
        default=True,
        description="是否只读提示",
    )

    # ---- Dy3+ 扩展字段 ----
    layer: LayerTag | None = Field(default=None, description="所属架构层")
    category: ToolCategory = Field(default=ToolCategory.INTERNAL, description="工具分类")
    estimated_latency_ms: int = Field(
        default=100,
        ge=1,
        description="预估延迟（毫秒）",
        json_schema_extra={"examples": [50, 100, 500, 5000, 30000]},
    )
    domain_scope: list[str] = Field(
        default_factory=list,
        description="领域范围，如 ['DOM-A', 'DOM-B']",
    )
    requires_compute: bool = Field(default=False, description="是否需要额外算力")
    rate_limit: int | None = Field(default=None, description="自定义限流（次/分钟），None 使用默认值")


# ============================================================
# 工具注册信息
# ============================================================

class ToolRegistration(BaseModel):
    """MCP 工具注册信息.

    封装 MCP 标准 Tool Schema + Dy3+ 扩展元数据。
    """

    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$", description="工具唯一标识")
    description: str = Field(..., min_length=1, max_length=2048, description="工具功能描述")
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description="JSON Schema 输入参数定义",
    )
    output_schema: dict[str, Any] | None = Field(default=None, description="JSON Schema 输出结构定义")
    annotations: Dy3ToolAnnotations = Field(default_factory=Dy3ToolAnnotations)
    server_name: str = Field(default="dy3-polaris", description="所属 MCP Server 名称")
    version: str = Field(default="1.0.0", description="工具版本")
    enabled: bool = Field(default=True, description="是否启用")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.startswith(("bkt_", "irt_", "skill_", "nist_", "mp_", "ss_", "cie_", "icdd_", "rule_", "cross_", "standard_", "fact_", "topology_", "path_", "resource_", "knowledge_", "forgetting_", "pubchem_", "crossref_", "arxiv_", "wiki_", "openalex_", "cas_", "wos_", "scifinder_", "reaxys_", "thermocalc_", "vasp_", "library_", "edu_", "lims_", "campus_")):
            pass  # 允许其他名称以保持灵活性
        return v

    def to_mcp_tool_dict(self) -> dict[str, Any]:
        """转换为 MCP SDK Tool 格式（用于 FastMCP 注册）."""
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        return result


# ============================================================
# 资源注册信息
# ============================================================

class ResourceRegistration(BaseModel):
    """MCP 资源注册信息."""

    uri: str = Field(..., min_length=1, description="资源 URI（RFC 3986）")
    name: str = Field(..., min_length=1, max_length=128, description="资源名称")
    description: str = Field(default="", max_length=2048, description="资源描述")
    mime_type: str = Field(default="application/json", description="资源 MIME 类型")
    resource_type: ResourceType = Field(default=ResourceType.LEARNER_PROFILE)
    layer: LayerTag | None = Field(default=None, description="所属架构层")
    server_name: str = Field(default="dy3-polaris", description="所属 MCP Server")


# ============================================================
# 溯源 KPA (Knowledge Provenance Artifact)
# ============================================================

class KPAEventType(str, Enum):
    """KPA 事件类型."""
    TOOL_INVOKED = "tool_invoked"
    RESOURCE_READ = "resource_read"
    AGENT_OUTPUT = "agent_output"
    DECISION_ROUTED = "decision_routed"
    HUMAN_OVERRIDE = "human_override"
    REVIEW_RESULT = "review_result"
    KNOWLEDGE_GENERATED = "knowledge_generated"


class KPA(BaseModel):
    """Knowledge Provenance Artifact - 溯源数据包.

    记录每一次知识处理节点的输入、处理逻辑、输出及环境信息。
    通过 Merkle 链串联形成完整溯源链。
    """

    kpa_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="KPA 唯一 ID")
    prev_hash: str | None = Field(default=None, description="前一个 KPA 的 Merkle 哈希")
    event_type: KPAEventType = Field(..., description="事件类型")
    actor: str = Field(..., min_length=1, description="执行者标识（agent_id / tool_name）")
    layer: LayerTag = Field(..., description="执行层")
    timestamp: float = Field(default_factory=time.time, description="时间戳（Unix epoch）")

    # ---- 7 维溯源信息 ----
    input_snapshot: dict[str, Any] = Field(default_factory=dict, description="输入快照（参数摘要）")
    processing_logic: str = Field(default="", description="处理逻辑标识（prompt 版本/算法名/规则集版本）")
    output_snapshot: dict[str, Any] = Field(default_factory=dict, description="输出快照（结果摘要）")
    context_refs: list[str] = Field(default_factory=list, description="上下文引用（文献 DOI / KP ID / 资源 URI）")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="置信度 [0,1]")
    code_hash: str | None = Field(default=None, description="执行代码的 Git commit hash")
    env_hash: str | None = Field(default=None, description="环境配置 hash")

    def compute_hash(self) -> str:
        """计算此 KPA 的 Merkle 哈希.

        用于构建链式结构：每个 KPA 的 hash = H(prev_hash + event_type + actor + timestamp + input + output)
        """
        import hashlib
        payload = f"{self.prev_hash or ''}|{self.event_type.value}|{self.actor}|{self.timestamp}"
        payload += f"|{self.input_snapshot}|{self.output_snapshot}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================
# A2A 消息模型
# ============================================================

class A2AMessageType(str, Enum):
    """A2A 消息类型."""
    DISCOVERY = "discovery"
    HANDSHAKE_REQUEST = "handshake_request"
    HANDSHAKE_RESPONSE = "handshake_response"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_ERROR = "task_error"
    CANCEL = "cancel"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"


class A2AMessage(BaseModel):
    """A2A 协议消息基类."""

    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16], description="消息 ID")
    message_type: A2AMessageType
    from_agent: str = Field(..., min_length=1, description="发送方 Agent ID")
    to_agent: str = Field(..., min_length=1, description="接收方 Agent ID")
    timestamp: float = Field(default_factory=time.time)
    session_id: str | None = Field(default=None, description="会话 ID")
    payload: dict[str, Any] = Field(default_factory=dict, description="消息载荷")
    provenance: KPA | None = Field(default=None, description="附带的溯源包")


class A2ACapability(BaseModel):
    """Agent 能力声明（用于 Discovery / Handshake）."""

    agent_id: str
    agent_name: str
    version: str = "0.1.0"
    supported_methods: list[str] = Field(default_factory=list, description="支持的 MCP methods")
    supported_tools: list[str] = Field(default_factory=list, description="可调用的 MCP tools")
    transport_types: list[str] = Field(default_factory=lambda: ["stdio", "websocket"], description="支持的传输方式")
    max_concurrent_tasks: int = Field(default=5, ge=1)
    domain_scope: list[str] = Field(default_factory=list, description="领域范围")


# ============================================================
# 算力资源描述
# ============================================================

class ComputeResourceType(str, Enum):
    """算力资源类型."""
    LOCAL_CPU = "local_cpu"
    GPU = "gpu"
    SSH_REMOTE = "ssh_remote"
    HPC_SLURM = "hpc_slurm"
    CLOUD_GPU = "cloud_gpu"


class ComputeResourceStatus(str, Enum):
    """算力资源状态."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    DRAINING = "draining"


class ComputeResourceDescriptor(BaseModel):
    """算力资源描述符."""

    resource_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    resource_type: ComputeResourceType
    name: str = Field(..., min_length=1)
    status: ComputeResourceStatus = ComputeResourceStatus.AVAILABLE

    # ---- 容量指标 ----
    cpu_cores: int | None = Field(default=None, ge=1)
    gpu_count: int | None = Field(default=None, ge=0)
    gpu_memory_gb: float | None = Field(default=None, ge=0)
    memory_gb: float | None = Field(default=None, ge=0)

    # ---- 调度属性 ----
    priority: int = Field(default=0, ge=-100, le=100, description="优先级（越高越优先）")
    max_queue_depth: int = Field(default=10, ge=1, description="最大队列深度")
    current_queue: list[str] = Field(default_factory=list, description="当前队列中的任务 ID")
    estimated_latency_ms: int = Field(default=100, ge=1, description="预估延迟")
    endpoint: str | None = Field(default=None, description="远程资源 endpoint")
    auth_config: dict[str, str] | None = Field(default=None, description="认证配置（引用环境变量）")

    @property
    def queue_depth(self) -> int:
        return len(self.current_queue)

    @property
    def is_available(self) -> bool:
        return self.status == ComputeResourceStatus.AVAILABLE and self.queue_depth < self.max_queue_depth