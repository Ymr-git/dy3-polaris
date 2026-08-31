"""L7 Artifact 管理系统 — 编辑通道 (edit_channel.py).

任务拆分 T3 · 设计文档 Ch.3.4。

实现 Artifact-Edit 通道, 连接用户编辑 → L5 Agent 处理 → 新版本 → 渲染更新。

完整调用链 (设计文档 Ch.3.4 D.1):
    1. 用户修改 Artifact 内容
    2. L7 计算增量差异 (ArtifactDiff, JSON Pointer 路径)
    3. 通过 L6 MCP 封装, 发送 artifact_edit broadcast 到 L5
    4. L5 Agent Runtime 路由给 source_agent
    5. Agent 处理编辑, 产出更新版本
    6. 新版本推送回 L7
    7. L7 调用 renderer.update(diff) 增量更新 DOM

编辑权限 (设计文档 Ch.3.4 callout):
    - editable 由源 Agent 或 CC2 审批决定
    - 默认: Agent 产出的图表/公式默认可编辑
    - CC2 正式批准的教学计划中的 Artifact 为只读

融合世界先进方案:
    - 观察者模式 / 事件订阅: L5 Agent 通过回调订阅编辑
    - 管道模式: DiffCompute → EditSubmit → AgentProcess → NewVersion
    - React Server Components: 增量差异驱动局部更新

设计要点:
    - EditChannel 依赖注入: diff 应用函数与渲染更新回调由调用方提供,
      实现与 ArtifactManager / RenderPipeline 的解耦
    - 同步提交 (submit) 与异步广播 (broadcast) 双模式
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..models import Artifact, ArtifactDiff, RenderDescriptor

# ============================================================
# 订阅者协议
# ============================================================


class ArtifactEditSubscriber(Protocol):
    """Artifact 编辑订阅者协议 — 模拟 L5 Agent Runtime 的编辑处理."""

    def on_artifact_edit(self, artifact: Artifact, diff: ArtifactDiff) -> None:
        """处理 Artifact 编辑 (L5 Agent 接收 artifact_edit broadcast)."""


#: 渲染更新回调 — 编辑应用后增量更新渲染 (renderer.update(diff))
RenderUpdater = Callable[[ArtifactDiff], RenderDescriptor | None]

#: Diff 应用回调 — 将 diff 应用到 Artifact 并返回新版本
#: (与 ArtifactManager.apply_edit(artifact_id, diff) 签名一致)
DiffApplier = Callable[[str, ArtifactDiff], Artifact]


@dataclass
class EditRecord:
    """编辑记录 — 编辑通道的审计日志条目."""

    artifact_id: str
    diff: ArtifactDiff
    edit_reason: str
    timestamp: float = field(default_factory=lambda: __import__("time").time())
    status: str = "submitted"  # submitted / processed / rejected
    subscriber: str = ""


class EditChannel:
    """Artifact-Edit 通道 — 用户编辑 → L5 处理 → 新版本 → 渲染更新.

    使用示例::

        channel = EditChannel(
            diff_applier=manager.apply_edit,
            render_updater=renderer_update,
        )
        channel.subscribe("agent-a1", subscriber)
        new_artifact = channel.submit(artifact_id, diff)
    """

    def __init__(
        self,
        diff_applier: DiffApplier,
        render_updater: RenderUpdater | None = None,
        broadcast: bool = True,
    ) -> None:
        self._diff_applier = diff_applier
        self._render_updater = render_updater
        self._broadcast = broadcast
        self._subscribers: dict[str, ArtifactEditSubscriber] = {}
        self._history: list[EditRecord] = []

    # ----------------------------------------------------------
    # 订阅管理
    # ----------------------------------------------------------

    def subscribe(self, name: str, subscriber: ArtifactEditSubscriber) -> None:
        """注册编辑订阅者 (模拟 L5 Agent)."""
        self._subscribers[name] = subscriber

    def unsubscribe(self, name: str) -> bool:
        """注销订阅者."""
        return self._subscribers.pop(name, None) is not None

    def subscriber_names(self) -> list[str]:
        """返回订阅者名单."""
        return list(self._subscribers.keys())

    # ----------------------------------------------------------
    # 编辑提交
    # ----------------------------------------------------------

    def submit(
        self,
        artifact_id: str,
        diff: ArtifactDiff,
        artifact: Artifact,
    ) -> Artifact:
        """提交编辑 — DiffCompute → EditSubmit → AgentProcess → NewVersion.

        Args:
            artifact_id: Artifact ID。
            diff: 增量差异 (RFC 6902)。
            artifact: 当前 Artifact 快照 (用于校验 editable)。

        Returns:
            应用编辑后的新版本 Artifact。

        Raises:
            ArtifactNotEditableError: editable=False (由 diff_applier 抛出)。
            ArtifactValidationError: diff 无法应用。
        """
        if not artifact.editable:
            from ..exceptions import ArtifactNotEditableError

            raise ArtifactNotEditableError(
                artifact_id=artifact_id,
                detail="Artifact 不可编辑 (CC2 批准的教学内容为只读)",
            )

        # DiffCompute: 应用增量差异生成新版本
        new_artifact = self._diff_applier(artifact.artifact_id, diff)

        # EditSubmit + AgentProcess: 广播给订阅者 (模拟 L5 artifact_edit broadcast)
        if self._broadcast and self._subscribers:
            for name, subscriber in list(self._subscribers.items()):
                try:
                    subscriber.on_artifact_edit(new_artifact, diff)
                    status = "processed"
                except Exception:  # noqa: BLE001 — 单订阅者失败不阻断
                    status = "rejected"
                self._history.append(
                    EditRecord(
                        artifact_id=artifact_id,
                        diff=diff,
                        edit_reason=diff.edit_reason,
                        status=status,
                        subscriber=name,
                    )
                )
        else:
            self._history.append(
                EditRecord(
                    artifact_id=artifact_id,
                    diff=diff,
                    edit_reason=diff.edit_reason,
                    status="submitted",
                    subscriber="",
                )
            )

        # NewVersion: 增量更新渲染 (renderer.update(diff))
        if self._render_updater is not None:
            try:
                self._render_updater(diff)
            except Exception:  # noqa: BLE001 — 渲染更新失败不影响版本生成
                pass

        return new_artifact

    def history(self) -> list[EditRecord]:
        """返回编辑历史 (审计)."""
        return list(self._history)

    def count(self) -> int:
        """编辑提交次数."""
        return len(self._history)
