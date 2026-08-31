"""L7 体验呈现层 — 核心数据模型.

遵循 L4 层 Pydantic v2 建模风格，提供 Artifact 管理、多模态渲染、
版本树 (DAG) 和学习者上下文适配所需的全部契约模型。

融合世界先进方案:
- Jupyter MIME Bundle: 多模态 Artifact 类型 + MIME 路由
- React Server Components: 增量差异 (ArtifactDiff / DiffOp)
- VS Code CustomEditor: 渲染上下文 (RenderContext)
- Git DAG: 版本树 (VersionTree / ArtifactVersionNode)
- JSON Patch RFC 6902: DiffOp 操作语义

数据契约流转:
    Artifact (L5 产出) → IRenderer.render(Artifact, RenderContext)
                       → RenderDescriptor (前端可消费)
    ArtifactDiff (增量差异) → IRenderer.update(ArtifactDiff) → RenderDescriptor
    ArtifactVersionNode → VersionTree (DAG) → get_lineage / get_latest_version
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .exceptions import ArtifactValidationError


# ============================================================
# 枚举定义
# ============================================================


class ArtifactType(str, Enum):
    """Artifact 类型枚举 (借鉴 Jupyter MIME Bundle + OLIVIA 多模态输出).

    TEXT:        文本类 — Markdown / 纯文本 / 富文本
    CHART:       图表类 — ECharts / Plotly / Vega-Lite
    GRAPH:       图谱类 — 知识图谱 / 关系网络
    MOLECULE:    分子类 — 2D/3D 分子结构 (RDKit / 3Dmol.js)
    TABLE:       表格类 — 结构化数据表
    FORMULA:     公式类 — LaTeX / MathML
    PROVENANCE:  溯源类 — KPA 链 / 推理链可视化
    INTERACTIVE: 交互类 — 可交互组件 / 模拟器
    """

    TEXT = "text"
    CHART = "chart"
    GRAPH = "graph"
    MOLECULE = "molecule"
    TABLE = "table"
    FORMULA = "formula"
    PROVENANCE = "provenance"
    INTERACTIVE = "interactive"


class ArtifactLifecycleState(str, Enum):
    """Artifact 生命周期状态.

    CREATED:   已创建 — 刚由 Agent 产出，尚未渲染
    RENDERED:  已渲染 — 渲染器已生成 RenderDescriptor
    REVIEWED:  已审核 — 通过 HITL 审核
    EDITED:    已编辑 — 学习者/教师已修改
    ARCHIVED:  已归档 — 不再活跃，保留历史
    """

    CREATED = "created"
    RENDERED = "rendered"
    REVIEWED = "reviewed"
    EDITED = "edited"
    ARCHIVED = "archived"


class LearnerMode(str, Enum):
    """学习者模式 — 根据学习者水平适配渲染深度.

    BEGINNER:     初学者 — 最大量辅助信息、详细解释、提示
    INTERMEDIATE: 中级 — 适度解释、关键提示
    ADVANCED:     高级 — 简洁输出、仅核心信息
    """

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class DiffOpType(str, Enum):
    """差异操作类型 (JSON Patch RFC 6902).

    ADD:     添加 — 在指定路径添加值
    REPLACE: 替换 — 替换指定路径的值
    REMOVE:  删除 — 删除指定路径的值
    MOVE:    移动 — 从源路径移动到目标路径
    COPY:    复制 — 从源路径复制到目标路径
    TEST:    测试 — 验证指定路径的值是否匹配
    """

    ADD = "add"
    REPLACE = "replace"
    REMOVE = "remove"
    MOVE = "move"
    COPY = "copy"
    TEST = "test"


# ============================================================
# MIME 类型映射
# ============================================================

#: Dy3+ 自定义 MIME 类型 → ArtifactType 映射
MIME_TO_TYPE: dict[str, ArtifactType] = {
    "text/vnd.dy3+markdown": ArtifactType.TEXT,
    "application/vnd.dy3.chart+json": ArtifactType.CHART,
    "application/vnd.dy3.graph+json": ArtifactType.GRAPH,
    "chemical/x-mdl-molfile": ArtifactType.MOLECULE,
    "application/vnd.dy3.table+json": ArtifactType.TABLE,
    "application/vnd.dy3.formula+json": ArtifactType.FORMULA,
    "application/vnd.dy3.provenance+json": ArtifactType.PROVENANCE,
    "application/vnd.dy3.interactive+json": ArtifactType.INTERACTIVE,
}

#: ArtifactType → MIME 类型映射 (反向)
TYPE_TO_MIME: dict[ArtifactType, str] = {v: k for k, v in MIME_TO_TYPE.items()}

#: 各 ArtifactType 的 payload 必需字段 (MOLECULE 特殊处理: molfile 或 smiles)
_PAYLOAD_REQUIRED_FIELDS: dict[ArtifactType, list[str]] = {
    ArtifactType.TEXT: ["content"],
    ArtifactType.CHART: ["chart_type", "data"],
    ArtifactType.GRAPH: ["nodes", "edges"],
    ArtifactType.TABLE: ["headers", "rows"],
    ArtifactType.FORMULA: ["latex"],
    ArtifactType.PROVENANCE: ["chain"],
    ArtifactType.INTERACTIVE: ["widget_type"],
}


# ============================================================
# 核心数据模型
# ============================================================


class Viewport(BaseModel):
    """视口尺寸 — 渲染时的显示区域.

    Attributes:
        width: 视口宽度 (像素)
        height: 视口高度 (像素)
    """

    width: int = Field(default=1280, ge=1, description="视口宽度 (像素)")
    height: int = Field(default=720, ge=1, description="视口高度 (像素)")

    model_config = {"frozen": False}


class Artifact(BaseModel):
    """Artifact — L5 智能体产出的可渲染制品.

    是渲染器的输入单元，携带类型、MIME、来源 Agent、溯源链、
    学习者上下文、版本、可编辑性和载荷数据。

    Attributes:
        artifact_id: 唯一标识，自动生成 "art-{uuid}"
        type: Artifact 类型
        mime: MIME 类型，用于路由到对应渲染器
        source_agent: 产出该 Artifact 的 Agent ID
        provenance_chain: 溯源链 (KPA ID 列表)
        learner_context: 学习者上下文 (知识掌握状态等)
        version: 版本号 (从 1 开始)
        editable: 是否可编辑
        fork_origin: 分叉来源 Artifact ID (None 表示原始创建)
        payload: 渲染载荷数据
        session_id: 会话 ID
        title: 标题 (用于搜索)
        created_at: 创建时间戳
        updated_at: 更新时间戳
        state: 生命周期状态
    """

    artifact_id: str = Field(
        default_factory=lambda: f"art-{uuid.uuid4().hex[:12]}",
        description="Artifact 唯一标识",
    )
    type: ArtifactType = Field(default=ArtifactType.TEXT, description="Artifact 类型")
    mime: str = Field(
        default="text/vnd.dy3+markdown",
        description="MIME 类型，用于路由到对应渲染器",
    )
    source_agent: str = Field(default="", description="产出该 Artifact 的 Agent ID")
    provenance_chain: list[str] = Field(
        default_factory=list,
        description="溯源链 (KPA ID 列表)",
    )
    learner_context: dict[str, Any] = Field(
        default_factory=dict,
        description="学习者上下文 (知识掌握状态等)",
    )
    version: int = Field(default=1, ge=1, description="版本号")
    editable: bool = Field(default=True, description="是否可编辑")
    fork_origin: str | None = Field(default=None, description="分叉来源 Artifact ID")
    payload: dict[str, Any] = Field(default_factory=dict, description="渲染载荷数据")
    session_id: str = Field(default="", description="会话 ID")
    title: str = Field(default="", description="标题 (用于搜索)")
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    updated_at: float = Field(default_factory=time.time, description="更新时间戳")
    state: ArtifactLifecycleState = Field(
        default=ArtifactLifecycleState.CREATED, description="生命周期状态"
    )

    model_config = {"frozen": False}

    @classmethod
    def mime_to_type(cls, mime: str) -> ArtifactType | None:
        """将 MIME 字符串映射到 ArtifactType.

        Args:
            mime: MIME 类型字符串

        Returns:
            对应的 ArtifactType，未知 MIME 返回 None
        """
        return MIME_TO_TYPE.get(mime)

    def validate(self) -> bool:
        """校验 payload 结构是否与类型匹配.

        根据 ArtifactType 检查 payload 中是否包含必需字段。
        MOLECULE 类型要求 molfile 或 smiles 至少存在其一。

        Returns:
            True 表示校验通过

        Raises:
            ArtifactValidationError: payload 结构与类型不匹配时抛出
        """
        # MOLECULE 特殊处理: molfile 或 smiles 至少一个
        if self.type == ArtifactType.MOLECULE:
            if "molfile" not in self.payload and "smiles" not in self.payload:
                raise ArtifactValidationError(
                    field="payload",
                    missing_fields=["molfile", "smiles"],
                    detail=f"MOLECULE artifact requires 'molfile' or 'smiles' in payload",
                )
            return True

        required = _PAYLOAD_REQUIRED_FIELDS.get(self.type, [])
        missing = [f for f in required if f not in self.payload]
        if missing:
            raise ArtifactValidationError(
                field="payload",
                missing_fields=missing,
                detail=f"{self.type.value} artifact missing required fields: {missing}",
            )
        return True

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典.

        将枚举字段转换为字符串值，保留所有业务字段。
        """
        return {
            "artifact_id": self.artifact_id,
            "type": self.type.value,
            "mime": self.mime,
            "source_agent": self.source_agent,
            "provenance_chain": self.provenance_chain,
            "learner_context": self.learner_context,
            "version": self.version,
            "editable": self.editable,
            "fork_origin": self.fork_origin,
            "payload": self.payload,
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state.value,
        }


class DiffOp(BaseModel):
    """差异操作 — JSON Patch RFC 6902 单个操作.

    描述 Artifact payload 中单个路径的变化。

    Attributes:
        op: 操作类型 (add/replace/remove/move/copy/test)
        path: JSON Pointer 路径 (e.g. "/payload/content")
        value: 操作值 (add/replace/test 需要, remove/move/copy 可选)
    """

    op: DiffOpType = Field(description="操作类型")
    path: str = Field(description="JSON Pointer 路径 (e.g. /payload/content)")
    value: Any = Field(default=None, description="操作值")

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "op": self.op.value,
            "path": self.path,
            "value": self.value,
        }


class ArtifactDiff(BaseModel):
    """Artifact 增量差异 — 用于渲染器的增量更新 (借鉴 React Server Components).

    描述 Artifact 从一个版本到另一个版本的变化操作序列。
    ops 字段同时接受 DiffOp 对象和 dict (RFC 6902 JSON Patch 风格)。

    Attributes:
        artifact_id: 关联的 Artifact ID
        ops: 差异操作列表 (DiffOp 或 dict, RFC 6902 JSON Patch 风格)
        edit_reason: 编辑原因
        created_at: 创建时间戳
    """

    artifact_id: str = Field(description="关联的 Artifact ID")
    ops: list[Any] = Field(default_factory=list, description="差异操作列表 (DiffOp 或 dict)")
    edit_reason: str = Field(default="", description="编辑原因")
    created_at: float = Field(default_factory=time.time, description="创建时间戳")

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        serialized_ops: list[dict[str, Any]] = []
        for op in self.ops:
            if isinstance(op, DiffOp):
                serialized_ops.append(op.to_dict())
            elif hasattr(op, "to_dict"):
                serialized_ops.append(op.to_dict())
            elif isinstance(op, dict):
                op_copy = dict(op)
                if "op" in op_copy and hasattr(op_copy["op"], "value"):
                    op_copy["op"] = op_copy["op"].value
                serialized_ops.append(op_copy)
            else:
                serialized_ops.append({"op": str(op)})
        return {
            "artifact_id": self.artifact_id,
            "ops": serialized_ops,
            "edit_reason": self.edit_reason,
            "created_at": self.created_at,
        }


class RenderContext(BaseModel):
    """渲染上下文 — 渲染时的环境与学习者状态 (借鉴 VS Code CustomEditor 上下文).

    传递给渲染器以适配视口、主题、学习者模式、知识掌握状态等。

    Attributes:
        viewport: 视口尺寸 (width/height)
        theme: 主题 (light/dark/auto)
        learner_mode: 学习者模式 (beginner/intermediate/advanced)
        bkt_state: BKT 知识掌握状态
        kp_ids: 关联知识点 ID 列表
        locale: 语言区域
    """

    viewport: Viewport = Field(default_factory=Viewport, description="视口尺寸")
    theme: str = Field(default="light", description="主题 (light/dark/auto)")
    learner_mode: LearnerMode = Field(
        default=LearnerMode.INTERMEDIATE, description="学习者模式"
    )
    bkt_state: dict[str, Any] = Field(
        default_factory=dict, description="BKT 知识掌握状态"
    )
    kp_ids: list[str] = Field(default_factory=list, description="关联知识点 ID 列表")
    locale: str = Field(default="zh-CN", description="语言区域")

    model_config = {"frozen": False}


class RenderDescriptor(BaseModel):
    """渲染描述符 — 渲染器产出，前端可直接消费 (借鉴 Jupyter MIME Bundle).

    包含 HTML 片段、配置、静态资源引用和元数据。

    Attributes:
        render_id: 渲染实例唯一标识，自动生成 "rd-{uuid}"
        artifact_id: 关联的 Artifact ID
        mime: 渲染输出的 MIME 类型
        html: 可嵌入的 HTML 片段 (可选)
        config: 前端渲染配置
        assets: 依赖的静态资源 URL 列表
        metadata: 渲染元数据
        rendered_at: 渲染完成时间戳
        render_time_ms: 渲染耗时 (毫秒)
    """

    render_id: str = Field(
        default_factory=lambda: f"rd-{uuid.uuid4().hex[:12]}",
        description="渲染实例唯一标识",
    )
    artifact_id: str = Field(default="", description="关联的 Artifact ID")
    mime: str = Field(default="", description="渲染输出的 MIME 类型")
    html: str | None = Field(default=None, description="可嵌入的 HTML 片段")
    config: dict[str, Any] = Field(default_factory=dict, description="前端渲染配置")
    assets: list[str] = Field(
        default_factory=list, description="依赖的静态资源 URL 列表"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="渲染元数据")
    rendered_at: float = Field(default_factory=time.time, description="渲染完成时间戳")
    render_time_ms: float = Field(default=0.0, ge=0.0, description="渲染耗时 (毫秒)")

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "render_id": self.render_id,
            "artifact_id": self.artifact_id,
            "mime": self.mime,
            "html": self.html,
            "config": self.config,
            "assets": self.assets,
            "metadata": self.metadata,
            "rendered_at": self.rendered_at,
            "render_time_ms": self.render_time_ms,
        }


class ArtifactVersionNode(BaseModel):
    """Artifact 版本节点 — 版本树 (DAG) 中的单个节点.

    每个节点记录版本号、所属 Artifact、父版本和分叉来源，
    构成有向无环图 (DAG) 支持分支与合并。

    Attributes:
        version: 版本号
        artifact_id: 所属 Artifact ID
        parent_version: 父版本号 (None 表示根版本)
        fork_origin: 分叉来源 (None 表示非分叉)
        created_at: 创建时间戳
    """

    version: int = Field(ge=1, description="版本号")
    artifact_id: str = Field(description="所属 Artifact ID")
    parent_version: int | None = Field(default=None, description="父版本号")
    fork_origin: str | None = Field(default=None, description="分叉来源")
    created_at: float = Field(default_factory=time.time, description="创建时间戳")

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "version": self.version,
            "artifact_id": self.artifact_id,
            "parent_version": self.parent_version,
            "fork_origin": self.fork_origin,
            "created_at": self.created_at,
        }


class VersionTree:
    """版本树 — 维护 Artifact 版本的 DAG 结构 (借鉴 Git DAG).

    支持线性版本链和分叉分支:
    - add_version: 添加版本节点
    - get_lineage: 获取从根到指定版本的版本链
    - get_latest_version: 获取最新版本号
    - get_children: 获取直接子版本
    - get_all_versions: 获取所有版本号

    DAG 示例:
        v1 → v2 → v3
              ↘ v4 (fork)

    lineage(v3) = [1, 2, 3]
    lineage(v4) = [1, 2, 4]
    """

    def __init__(self, artifact_id: str) -> None:
        """初始化版本树.

        Args:
            artifact_id: 关联的 Artifact ID
        """
        self.artifact_id = artifact_id
        self._nodes: dict[int, ArtifactVersionNode] = {}

    def add_version(self, node: ArtifactVersionNode) -> None:
        """添加版本节点到版本树.

        Args:
            node: 版本节点

        Raises:
            ValueError: 版本号已存在
        """
        if node.version in self._nodes:
            raise ValueError(f"Version {node.version} already exists in version tree")
        self._nodes[node.version] = node

    def get_latest_version(self) -> int | None:
        """获取最新版本号.

        Returns:
            最新版本号，空树返回 None
        """
        if not self._nodes:
            return None
        return max(self._nodes.keys())

    def get_lineage(self, version: int) -> list[int]:
        """获取从根到指定版本的版本链.

        沿 parent_version 链回溯到根，返回有序版本号列表。

        Args:
            version: 目标版本号

        Returns:
            从根到目标版本的版本号列表，版本不存在返回空列表
        """
        if version not in self._nodes:
            return []
        lineage: list[int] = []
        current: int | None = version
        while current is not None:
            node = self._nodes.get(current)
            if node is None:
                break
            lineage.append(current)
            current = node.parent_version
        lineage.reverse()
        return lineage

    def get_version_node(self, version: int) -> ArtifactVersionNode | None:
        """获取指定版本的节点.

        Args:
            version: 版本号

        Returns:
            版本节点，不存在返回 None
        """
        return self._nodes.get(version)

    def get_all_versions(self) -> list[int]:
        """获取所有版本号.

        Returns:
            版本号列表 (无序)
        """
        return list(self._nodes.keys())

    def get_children(self, version: int) -> list[int]:
        """获取指定版本的直接子版本.

        Args:
            version: 父版本号

        Returns:
            直接子版本号列表 (无序)
        """
        return [v for v, n in self._nodes.items() if n.parent_version == version]

    def version_count(self) -> int:
        """获取版本节点总数.

        Returns:
            版本节点数量
        """
        return len(self._nodes)
