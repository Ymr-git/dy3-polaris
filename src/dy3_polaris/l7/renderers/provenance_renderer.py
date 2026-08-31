"""L7 渲染器 — ProvenanceRenderer (application/vnd.dy3.provenance+json).

将溯源链 Artifact 渲染为三模式可视化 HTML (设计文档 §2.8 + §6)。

实现能力 (对应 L7 设计文档 §6.1-§6.4):

1. **时间线模式** (§6.1): 纵向时间线展示 KP 交互历史。
   5 种事件类型: 教学 / 测试 / 状态变更 / 决策 / 编辑。
   支持按类型筛选 (config), 事件节点含 Agent/知识源/溯源链 ID。
   隐私规则: learner_context["full_access"] 为 False 时脱敏原文。
2. **决策树模式** (§6.2): 横向流程图 5 步链路 —
   复杂度评估 → 范式选择 → Agent 调度 → 执行过程 → 裁决结果。
   三级溯源深度: summary / standard / full。
3. **分支合并图模式** (§6.4): Git 风格分支图 —
   主线 + 分叉 + Artifact 编辑 + 合并节点; 分支原因标注。

融合世界先进方案:
    - 时间线语义: 纵向时间轴 + 事件类型筛选
    - Git 分支图语义: 主线/分叉/合并可视化
    - 隐私分级: 默认脱敏, 完整模式需授权

输出契约:
    RenderDescriptor.html   — 完整可视化 HTML
    RenderDescriptor.config — {mode, depth, filters, privacy}
    RenderDescriptor.assets — []
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ._common import build_descriptor, esc, wrap

#: 支持的 MIME 类型
_MIME_TYPES: list[str] = ["application/vnd.dy3.provenance+json"]

#: 事件类型 → (显示名, 图标, 颜色)
_EVENT_TYPES: dict[str, tuple[str, str, str]] = {
    "teaching": ("教学", "📖", "#4b3fe3"),
    "test": ("测试", "✏️", "#22a5f7"),
    "state_change": ("状态变更", "🔄", "#f59e0b"),
    "decision": ("决策", "🧭", "#8b5cf6"),
    "edit": ("编辑", "🛠", "#10b981"),
}

#: 溯源深度 → 可见范围
_DEPTH_LABELS: dict[str, str] = {
    "summary": "摘要",
    "standard": "标准",
    "full": "完整",
}

#: 分支原因 → 徽章颜色
_BRANCH_REASONS: dict[str, str] = {
    "用户追问": "#22a5f7",
    "方案探索": "#8b5cf6",
    "Agent冲突": "#f59e0b",
}


class ProvenanceRenderer:
    """溯源渲染器 — 溯源链 → 三模式可视化 HTML."""

    _MIME_TYPES: list[str] = list(_MIME_TYPES)

    def render(self, artifact: Artifact, context: RenderContext):
        started = time.monotonic()
        if artifact is None or not artifact.payload:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload", detail="Provenance artifact requires non-empty payload"
            )
        payload = artifact.payload
        mode = str(payload.get("mode", "timeline")).lower()
        if mode not in ("timeline", "decision", "branch_merge"):
            mode = "timeline"
        # 按模式校验必需字段
        required: list[str] = []
        if mode == "timeline":
            required = ["events", "chain"]
        elif mode == "decision":
            required = ["steps", "chain"]
        else:
            required = ["branches", "mainline"]
        has_required = any(k in payload for k in required)
        if not has_required:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload",
                missing_fields=required,
                detail=(
                    f"Provenance artifact (mode={mode}) requires one of "
                    f"{required} in payload"
                ),
            )
        depth = str(payload.get("depth", "standard")).lower()
        if depth not in _DEPTH_LABELS:
            depth = "standard"
        theme = (context.theme if context else "light") or "light"
        full_access = bool(
            (artifact.learner_context or {}).get("full_access")
            or (context.bkt_state or {}).get("full_access")
            or (payload.get("full_access") or False)
        )
        privacy = {"full_access": full_access, "masked": not full_access}

        if mode == "timeline":
            html_body = self._render_timeline(payload, theme, full_access)
        elif mode == "decision":
            html_body = self._render_decision(payload, depth, theme)
        else:
            html_body = self._render_branch_merge(payload, theme)

        html = wrap(html_body, "l7-provenance", theme)
        config = {
            "mode": mode,
            "depth": depth,
            "depth_label": _DEPTH_LABELS[depth],
            "filters": {"by_type": True, "by_time": True} if mode == "timeline" else {},
            "privacy": privacy,
        }
        descriptor = build_descriptor(
            artifact,
            html=html,
            config=config,
            assets=[],
            metadata={
                "renderer": "ProvenanceRenderer",
                "mode": mode,
                "depth": depth,
                "privacy_masked": not full_access,
            },
        )
        descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
        return descriptor

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)

    # ----------------------------------------------------------
    # 时间线模式 (§6.1)
    # ----------------------------------------------------------

    def _render_timeline(
        self, payload: dict[str, Any], theme: str, full_access: bool
    ) -> str:
        events = payload.get("events") or payload.get("chain") or []
        parts = ['<div class="prov-timeline" data-mode="timeline">']
        parts.append('<div class="prov-filters"><span class="prov-filter-label">筛选:</span>')
        for key, (label, icon, _color) in _EVENT_TYPES.items():
            parts.append(
                f'<button class="prov-filter-chip" data-type="{key}">{icon} {label}</button>'
            )
        parts.append('<button class="prov-filter-chip active" data-type="all">全部</button></div>')
        parts.append('<div class="prov-timeline-track">')

        for ev in events:
            etype = str(ev.get("type", "teaching")).lower()
            label, icon, color = _EVENT_TYPES.get(etype, _EVENT_TYPES["teaching"])
            timestamp = ev.get("timestamp", ev.get("time", ""))
            kp_id = str(ev.get("kp_id", ev.get("target_kp", "")))
            chain_id = str(ev.get("chain_id", ev.get("provenance_chain_id", "")))
            agent = str(ev.get("agent", ""))
            summary = str(ev.get("summary", ev.get("content", "")))

            # 隐私脱敏: 无完整授权时不展示原始输入
            raw = ev.get("raw", "")
            if raw and not full_access:
                raw_display = '<span class="prov-masked">（原文已脱敏）</span>'
            else:
                raw_display = f"<div class=\"prov-raw\">{esc(raw)}</div>" if raw else ""

            parts.append(
                f'<div class="prov-event" data-type="{etype}">'
                f'<div class="prov-event-marker" style="background:{color}"></div>'
                f'<div class="prov-event-card">'
                f'<div class="prov-event-head">'
                f'<span class="prov-event-icon" aria-hidden="true">{icon}</span>'
                f'<span class="prov-event-type" style="color:{color}">{label}</span>'
                f'<span class="prov-event-time">{esc(timestamp)}</span>'
                f'</div>'
                f'<div class="prov-event-body">{esc(summary)}{raw_display}</div>'
                f'<div class="prov-event-meta">'
                f'{f"<span class=prov-tag>KP {esc(kp_id)}</span>" if kp_id else ""}'
                f'{f"<span class=prov-tag>Agent {esc(agent)}</span>" if agent else ""}'
                f'{f"<span class=prov-tag>{esc(chain_id)}</span>" if chain_id else ""}'
                f'</div></div></div>'
            )
        parts.append("</div></div>")
        return "\n".join(parts)

    # ----------------------------------------------------------
    # 决策树模式 (§6.2)
    # ----------------------------------------------------------

    def _render_decision(self, payload: dict[str, Any], depth: str, theme: str) -> str:
        steps = payload.get("steps") or payload.get("chain") or []
        parts = ['<div class="prov-decision" data-mode="decision" data-depth="' + depth + '">']
        parts.append(
            '<div class="prov-depth-switch"><span class="prov-depth-label">溯源深度:</span>'
            + "".join(
                f'<button class="prov-depth-chip{" active" if depth == d else ""}" data-depth="{d}">{_DEPTH_LABELS[d]}</button>'
                for d in ("summary", "standard", "full")
            )
            + "</div>"
        )
        parts.append('<div class="prov-decision-flow">')

        icons = ["🔍", "🧭", "🤖", "⚙️", "⚖️"]
        for idx, step in enumerate(steps):
            step_type = str(step.get("step", step.get("type", f"step-{idx}"))).lower()
            title = str(step.get("title", step.get("name", f"步骤 {idx+1}")))
            detail = str(step.get("detail", step.get("description", "")))
            score = step.get("score")
            icon = icons[idx] if idx < len(icons) else "•"
            visible = self._step_visible(step_type, depth)

            if not visible:
                continue
            parts.append(
                f'<div class="prov-step" data-step="{step_type}">'
                f'<div class="prov-step-icon" aria-hidden="true">{icon}</div>'
                f'<div class="prov-step-card">'
                f'<div class="prov-step-title">{esc(title)}</div>'
                f'<div class="prov-step-detail">{esc(detail)}</div>'
                + (
                    f'<div class="prov-step-score">评分: {esc(score)}</div>'
                    if score is not None
                    else ""
                )
                + f'</div></div>'
            )
            if idx < len(steps) - 1:
                parts.append('<div class="prov-arrow" aria-hidden="true">→</div>')
        parts.append("</div></div>")
        return "\n".join(parts)

    @staticmethod
    def _step_visible(step_type: str, depth: str) -> bool:
        """按溯源深度过滤步骤可见性.

        summary: 仅保留 决策 + 裁决 (关键节点)
        standard: 全部中间步骤 (默认)
        full: 全部 (含 Agent 内部推理, 已在 payload 层控制)
        """
        if depth == "summary":
            return step_type in ("decision", "adjudication", "verdict", "裁决", "决策")
        return True

    # ----------------------------------------------------------
    # 分支合并图模式 (§6.4)
    # ----------------------------------------------------------

    def _render_branch_merge(self, payload: dict[str, Any], theme: str) -> str:
        branches = payload.get("branches") or []
        merges = payload.get("merges") or []
        parts = ['<div class="prov-branch" data-mode="branch_merge">']
        parts.append('<div class="prov-branch-legend">'
                     '<span class="prov-legend-item"><span class="prov-legend-line main"></span>主线</span>'
                     '<span class="prov-legend-item"><span class="prov-legend-line branch"></span>分支</span>'
                     '<span class="prov-legend-item"><span class="prov-legend-dot merge"></span>合并</span>'
                     '</div>')
        parts.append('<div class="prov-branch-track">')

        # 主线
        main_nodes = payload.get("mainline") or [{"title": "初始版本", "id": "v1"}]
        all_rows: list[dict[str, Any]] = []
        for idx, node in enumerate(main_nodes):
            all_rows.append(
                {
                    "kind": "main",
                    "id": str(node.get("id", f"main-{idx}")),
                    "title": str(node.get("title", f"v{idx+1}")),
                    "reason": node.get("reason", ""),
                }
            )
        for branch in branches:
            all_rows.append(
                {
                    "kind": "branch",
                    "id": str(branch.get("id", "branch")),
                    "title": str(branch.get("title", "分支")),
                    "reason": str(branch.get("reason", "")),
                    "color": _BRANCH_REASONS.get(str(branch.get("reason", "")), "#94a3b8"),
                }
            )
        for merge in merges:
            all_rows.append(
                {
                    "kind": "merge",
                    "id": str(merge.get("id", "merge")),
                    "title": str(merge.get("title", "合并")),
                    "result": str(merge.get("result", "")),
                }
            )

        for row in all_rows:
            kind = row["kind"]
            if kind == "main":
                parts.append(
                    f'<div class="prov-branch-row main">'
                    f'<span class="prov-branch-dot main" aria-hidden="true"></span>'
                    f'<span class="prov-branch-title">{esc(row["title"])}</span></div>'
                )
            elif kind == "branch":
                reason = row.get("reason", "")
                color = _BRANCH_REASONS.get(reason, "#94a3b8")
                parts.append(
                    f'<div class="prov-branch-row branch">'
                    f'<span class="prov-branch-line" style="background:{color}" aria-hidden="true"></span>'
                    f'<span class="prov-branch-dot branch" style="background:{color}" aria-hidden="true"></span>'
                    f'<span class="prov-branch-title">{esc(row["title"])}</span>'
                    + (
                        f'<span class="prov-branch-reason" style="color:{color}">{esc(reason)}</span>'
                        if reason
                        else ""
                    )
                    + "</div>"
                )
            else:
                parts.append(
                    f'<div class="prov-branch-row merge">'
                    f'<span class="prov-branch-dot merge" aria-hidden="true"></span>'
                    f'<span class="prov-branch-title">{esc(row["title"])}</span>'
                    + (
                        f'<span class="prov-branch-result">{esc(row.get("result", ""))}</span>'
                        if row.get("result")
                        else ""
                    )
                    + "</div>"
                )
        parts.append("</div></div>")
        return "\n".join(parts)
