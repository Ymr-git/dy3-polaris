"""L7 API — Artifact API (artifact_api.py).

任务拆分 T6 · 设计文档 Ch.9.2。

Artifact 端点处理器 (复用 ArtifactManager):
- GET /api/v1/artifacts — 列表 (session_id/type/source_agent/kp_id/page/size/sort)
- GET /api/v1/artifacts/{id} — 详情 (含 versions[])
- POST /api/v1/artifacts/{id}/edit — 提交编辑 (ArtifactDiff JSON Pointer)
"""

from __future__ import annotations

from typing import Any

from ..artifact_manager import ArtifactManager
from ..exceptions import ArtifactNotFoundError, ArtifactNotEditableError, VersionConflictError
from ..models import Artifact, ArtifactDiff
from .error_codes import error_payload


def list_artifacts(
    manager: ArtifactManager,
    session_id: str | None = None,
    artifact_type: str | None = None,
    source_agent: str | None = None,
    kp_id: str | None = None,
    page: int = 1,
    size: int = 20,
    sort: str = "-created_at",
) -> dict[str, Any]:
    """GET /api/v1/artifacts — 列表 (分页 + 过滤 + 排序)."""
    items = manager.list_artifacts(
        session_id=session_id,
        artifact_type=artifact_type,
        source_agent=source_agent,
        kp_id=kp_id,
    )
    # 排序
    desc = sort.startswith("-")
    field = sort[1:] if desc else sort
    if field == "created_at":
        items = sorted(items, key=lambda a: a.created_at, reverse=desc)
    elif field == "updated_at":
        items = sorted(items, key=lambda a: a.updated_at, reverse=desc)
    elif field == "version":
        items = sorted(items, key=lambda a: a.version, reverse=desc)
    total = len(items)
    page = max(1, page)
    size = max(1, min(size, 100))
    start = (page - 1) * size
    paged = items[start : start + size]
    return {
        "items": [a.to_dict() for a in paged],
        "total": total,
        "page": page,
        "size": size,
        "total_pages": (total + size - 1) // size,
    }


def get_artifact(manager: ArtifactManager, artifact_id: str) -> dict[str, Any]:
    """GET /api/v1/artifacts/{id} — 详情 (含 versions[])."""
    try:
        artifact = manager.get(artifact_id)
    except ArtifactNotFoundError:
        return error_payload("ARTIFACT_NOT_FOUND")
    versions = [n.to_dict() for n in manager.get_version_history(artifact_id)]
    data = artifact.to_dict()
    data["versions"] = versions
    return data


def edit_artifact(manager: ArtifactManager, artifact_id: str, diff: ArtifactDiff) -> dict[str, Any]:
    """POST /api/v1/artifacts/{id}/edit — 提交编辑.

    Args:
        manager: ArtifactManager。
        artifact_id: 目标 Artifact。
        diff: ArtifactDiff (JSON Pointer 路径)。

    Returns:
        编辑后的 Artifact 或错误 payload。
    """
    try:
        edited = manager.apply_edit(artifact_id, diff)
        return edited.to_dict()
    except ArtifactNotFoundError:
        return error_payload("ARTIFACT_NOT_FOUND")
    except ArtifactNotEditableError:
        return error_payload("ARTIFACT_READONLY")
    except VersionConflictError:
        return error_payload("EDIT_REJECTED", details={"reason": "版本冲突"})
    except Exception as exc:  # noqa: BLE001
        return error_payload("RENDER_PAYLOAD_INVALID", details={"detail": str(exc)})
