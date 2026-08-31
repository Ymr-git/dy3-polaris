"""L7 Artifact 管理系统 — 数据模型 (models.py).

任务拆分 T3 交付物之一。Artifact 核心模型已在 T1 的
``l7/models.py`` 中完整实现 (Artifact/ArtifactDiff/DiffOp/VersionTree 等)，
本模块重新导出并补充 Artifact 管理域专属模型，保持任务拆分文件结构完整。

补充模型:
    - ArtifactMetadata: Artifact 元数据视图 (不含 payload, 用于列表/搜索)
    - EditPermission: 编辑权限判定结果 (editable 来源说明)
"""

from __future__ import annotations

from typing import Any

from ..models import (  # noqa: F401 — 重新导出 T1 模型
    Artifact,
    ArtifactDiff,
    ArtifactLifecycleState,
    ArtifactType,
    ArtifactVersionNode,
    DiffOp,
    DiffOpType,
    MIME_TO_TYPE,
    RenderContext,
    RenderDescriptor,
    TYPE_TO_MIME,
    VersionTree,
)

__all__ = [
    # T1 重新导出
    "Artifact",
    "ArtifactDiff",
    "ArtifactLifecycleState",
    "ArtifactType",
    "ArtifactVersionNode",
    "DiffOp",
    "DiffOpType",
    "MIME_TO_TYPE",
    "TYPE_TO_MIME",
    "VersionTree",
    # 管理域专属
    "ArtifactMetadata",
    "EditPermission",
]


class ArtifactMetadata:
    """Artifact 元数据视图 — 不含 payload 的轻量摘要 (列表/搜索展示).

    Attributes:
        artifact_id: Artifact ID。
        type: 类型。
        mime: MIME。
        title: 标题。
        source_agent: 来源 Agent。
        version: 版本号。
        state: 生命周期状态。
        session_id: 会话 ID。
        created_at / updated_at: 时间戳。
        kp_ids: 关联知识点。
    """

    def __init__(self, artifact: "Artifact") -> None:
        self.artifact_id = artifact.artifact_id
        self.type = artifact.type.value if hasattr(artifact.type, "value") else str(artifact.type)
        self.mime = artifact.mime
        self.title = artifact.title
        self.source_agent = artifact.source_agent
        self.version = artifact.version
        self.state = artifact.state.value if hasattr(artifact.state, "value") else str(artifact.state)
        self.session_id = artifact.session_id
        self.created_at = artifact.created_at
        self.updated_at = artifact.updated_at
        learner = artifact.learner_context or {}
        self.kp_ids: list[str] = [str(k) for k in (learner.get("kp_ids") or [])]

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "artifact_id": self.artifact_id,
            "type": self.type,
            "mime": self.mime,
            "title": self.title,
            "source_agent": self.source_agent,
            "version": self.version,
            "state": self.state,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "kp_ids": self.kp_ids,
        }


class EditPermission:
    """编辑权限判定结果 (设计文档 Ch.3.4).

    Attributes:
        editable: 是否可编辑。
        source: 权限来源 ("cc2_approved" / "source_agent" / "default")。
        reason: 说明。
    """

    def __init__(self, editable: bool, source: str, reason: str = "") -> None:
        self.editable = editable
        self.source = source
        self.reason = reason

    @classmethod
    def cc2_approved(cls, reason: str = "CC2 正式批准的教学计划内容为只读") -> "EditPermission":
        """CC2 审批决定: 只读."""
        return cls(editable=False, source="cc2_approved", reason=reason)

    @classmethod
    def agent_editable(cls, reason: str = "Agent 产出的图表/公式默认可编辑") -> "EditPermission":
        """源 Agent 决定: 可编辑 (默认)."""
        return cls(editable=True, source="source_agent", reason=reason)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "editable": self.editable,
            "source": self.source,
            "reason": self.reason,
        }
