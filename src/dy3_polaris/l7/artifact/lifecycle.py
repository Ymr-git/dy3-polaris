"""L7 Artifact 管理系统 — 生命周期状态机 (lifecycle.py).

任务拆分 T3 · 设计文档 Ch.3.2。

实现 Artifact 五阶段生命周期的显式状态机:

    Created → Rendered → Reviewed → Edited → Archived

合法转移 (依据设计文档 Ch.3.2 Mermaid stateDiagram):
    [*] → Created                       Agent 产出
    Created → Rendered                  L7 Renderer 渲染
    Rendered → Reviewed                 CC1 Actor-Critic 审查 (可选)
    Reviewed → Edited                   用户/Agent 修改
    Reviewed → Archived                 会话结束归档
    Edited → Rendered                   编辑后重新渲染
    Rendered → Archived                 会话结束归档
    Archived → Rendered                 (扩展) 历史会话回顾重新加载

非法转移 (设计文档未定义即非法):
    Created → Reviewed / Edited / Archived   (必须先渲染)
    Reviewed → Rendered                      (审查后只能编辑或归档)
    Edited → Reviewed                        (编辑后必然重新渲染)
    Created → Created                        (自环非法)

每阶段子状态 (设计文档 A.2):
    Created:  Validating → Registered
    Rendered: Routing → Rendering → Mounted
    Edited:   DiffCompute → EditSubmit → AgentProcess → NewVersion

融合世界先进方案:
    - 领域驱动设计 (DDD): 聚合根状态机, 显式转移表代替自由赋值
    - UML 状态机 (state diagram): 合法/非法转移的完备定义
    - L5 kernel_persistence 六态状态机 (合法转换表模式)

设计要点:
    - 状态机本身无状态 (纯函数式), 由调用方持有当前状态
    - transition() 校验合法性, 非法转移抛 StateTransitionError
    - 子状态作为状态明细 (phase detail) 记录, 不参与主转移校验
"""

from __future__ import annotations

from typing import Any

from ..exceptions import ArtifactValidationError
from ..models import ArtifactLifecycleState

# ============================================================
# 状态机异常
# ============================================================


class StateTransitionError(ArtifactValidationError):
    """非法状态转移异常 — 状态机不允许的转移时抛出."""

    def __init__(self, current: Any, target: Any, detail: str = "") -> None:
        self.current = current
        self.target = target
        msg = (
            f"非法状态转移: {getattr(current, 'value', current)}"
            f" → {getattr(target, 'value', target)}"
        )
        if detail:
            msg = f"{msg} ({detail})"
        super().__init__(
            field="state",
            missing_fields=[],
            detail=msg,
        )


# ============================================================
# 合法转移表
# ============================================================

#: 合法顶层转移表 — {当前状态: {合法目标状态集合}}
_VALID_TRANSITIONS: dict[ArtifactLifecycleState, set[ArtifactLifecycleState]] = {
    ArtifactLifecycleState.CREATED: {ArtifactLifecycleState.RENDERED},
    ArtifactLifecycleState.RENDERED: {
        ArtifactLifecycleState.REVIEWED,
        ArtifactLifecycleState.EDITED,
        ArtifactLifecycleState.ARCHIVED,
    },
    ArtifactLifecycleState.REVIEWED: {
        ArtifactLifecycleState.EDITED,
        ArtifactLifecycleState.ARCHIVED,
    },
    ArtifactLifecycleState.EDITED: {ArtifactLifecycleState.RENDERED},
    # Archived 为"软终态": 允许重新加载 (历史会话回顾, 设计文档 Archive 阶段)
    ArtifactLifecycleState.ARCHIVED: {ArtifactLifecycleState.RENDERED},
}

#: 终态集合 (设计文档: Archived 为吸收态, 但支持重新加载扩展)
_TERMINAL_STATES: set[ArtifactLifecycleState] = set()

#: 子状态定义 — {阶段: {子状态名: 描述}}
_SUB_STATES: dict[ArtifactLifecycleState, dict[str, str]] = {
    ArtifactLifecycleState.CREATED: {
        "validating": "元数据校验 (MIME 支持 + payload 完整性)",
        "registered": "写入 Artifact Store",
    },
    ArtifactLifecycleState.RENDERED: {
        "routing": "Renderer Registry 路由 (按 MIME type)",
        "rendering": "调用 IRenderer.render() 生成 DOM",
        "mounted": "挂载到 DOM",
    },
    ArtifactLifecycleState.EDITED: {
        "diff_compute": "计算增量差异 (ArtifactDiff)",
        "edit_submit": "通过 Artifact-Edit 通道回传 L5",
        "agent_process": "L5 Agent 处理编辑",
        "new_version": "生成新版本 (版本号递增)",
    },
}


class LifecycleStateMachine:
    """Artifact 生命周期状态机 — 显式合法转移校验.

    使用示例::

        machine = LifecycleStateMachine()
        state = ArtifactLifecycleState.CREATED
        state = machine.transition(state, ArtifactLifecycleState.RENDERED)  # OK
        machine.transition(state, ArtifactLifecycleState.ARCHIVED)  # Rendered→Archived OK

    Attributes:
        valid_transitions: 只读合法转移表 (当前状态 → 合法目标集合)。
    """

    def __init__(self) -> None:
        self.valid_transitions: dict[ArtifactLifecycleState, set[ArtifactLifecycleState]] = {
            k: set(v) for k, v in _VALID_TRANSITIONS.items()
        }

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def can_transition(
        self,
        current: ArtifactLifecycleState,
        target: ArtifactLifecycleState,
    ) -> bool:
        """判断转移是否合法.

        Args:
            current: 当前状态。
            target: 目标状态。

        Returns:
            True 表示允许该转移。
        """
        targets = self.valid_transitions.get(current)
        return targets is not None and target in targets

    def allowed_targets(self, current: ArtifactLifecycleState) -> list[ArtifactLifecycleState]:
        """返回当前状态的全部合法目标状态 (有序)."""
        targets = self.valid_transitions.get(current, set())
        # 按枚举定义顺序稳定排序
        return [s for s in ArtifactLifecycleState if s in targets]

    def is_terminal(self, state: ArtifactLifecycleState) -> bool:
        """判断是否为终态 (无出边)."""
        return not self.valid_transitions.get(state)

    # ----------------------------------------------------------
    # 转移
    # ----------------------------------------------------------

    def transition(
        self,
        current: ArtifactLifecycleState,
        target: ArtifactLifecycleState,
    ) -> ArtifactLifecycleState:
        """执行状态转移 (校验合法性).

        Args:
            current: 当前状态。
            target: 目标状态。

        Returns:
            转移后的目标状态。

        Raises:
            StateTransitionError: 非法转移时抛出。
        """
        if not self.can_transition(current, target):
            raise StateTransitionError(
                current=current,
                target=target,
                detail=f"合法目标: {[s.value for s in self.allowed_targets(current)]}",
            )
        return target

    def validate_state(self, state: ArtifactLifecycleState) -> None:
        """校验状态是否为本状态机已知状态.

        Args:
            state: 待校验状态。

        Raises:
            ArtifactValidationError: 未知状态。
        """
        if state not in ArtifactLifecycleState:
            raise ArtifactValidationError(
                field="state",
                detail=f"未知生命周期状态: {state}",
            )

    # ----------------------------------------------------------
    # 子状态
    # ----------------------------------------------------------

    def sub_states(self, phase: ArtifactLifecycleState) -> dict[str, str]:
        """返回某阶段的子状态定义 (设计文档 A.2).

        Args:
            phase: 主阶段。

        Returns:
            {子状态名: 描述}，无子状态阶段返回空字典。
        """
        return dict(_SUB_STATES.get(phase, {}))

    def validate_sub_state(
        self, phase: ArtifactLifecycleState, sub_state: str
    ) -> None:
        """校验子状态是否属于指定阶段.

        Args:
            phase: 主阶段。
            sub_state: 子状态名。

        Raises:
            ArtifactValidationError: 子状态不属于该阶段。
        """
        if sub_state not in _SUB_STATES.get(phase, {}):
            raise ArtifactValidationError(
                field="sub_state",
                detail=(
                    f"子状态 {sub_state!r} 不属于阶段"
                    f" {phase.value!r} (可选: {list(_SUB_STATES.get(phase, {}).keys())})"
                ),
            )


# ============================================================
# 便捷单例与辅助
# ============================================================

_DEFAULT_MACHINE = LifecycleStateMachine()


def get_state_machine() -> LifecycleStateMachine:
    """返回默认状态机实例 (可复用)."""
    return _DEFAULT_MACHINE


def assert_transition(
    current: ArtifactLifecycleState, target: ArtifactLifecycleState
) -> ArtifactLifecycleState:
    """模块级便捷函数 — 校验并返回目标状态."""
    return _DEFAULT_MACHINE.transition(current, target)
