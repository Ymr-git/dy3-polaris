"""L7 溯源可视化 — 分支合并可视化 (branch_merge.py).

任务拆分 T5 · 设计文档 Ch.6.4。

Git 风格分支图: 主线中央垂直延伸 → 分支分叉 → 各自 Artifact 编辑 →
合并回主线以合并节点标记。分支原因标注 ("用户追问"/"方案探索"/"Agent 冲突")。

融合世界先进方案:
- Git 分支模型: 主线/分支/合并节点语义
- 可视化: 左侧时间轴 + 右侧内容布局 (类似 git log --graph)
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import build_descriptor, esc
from ._common import BRANCH_REASONS, build_panel_descriptor


def render_branch_merge(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    mainline: list[dict[str, Any]] | None = None,
    branches: list[dict[str, Any]] | None = None,
    merges: list[dict[str, Any]] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染分支合并图 (Ch.6.4).

    Args:
        mainline: 主线节点 [{id, title, time}].
        branches: 分支 [{id, title, reason, nodes: [节点], from_node}].
        merges: 合并 [{title, result, from_branch}].

    Returns:
        RenderDescriptor。
    """
    started = time.monotonic()
    mainline = mainline or []
    branches = branches or []
    merges = merges or []

    # 主线
    main_html = "".join(
        f'<div class="prov-branch-main-node">'
        f'<span class="prov-commit-dot main"></span>'
        f'<span class="prov-commit-title">{esc(n.get("title", ""))}</span>'
        f'<span class="prov-commit-time">{esc(str(n.get("time", "")))}</span>'
        f"</div>"
        for n in mainline
    )

    # 分支
    branch_html = ""
    for b in branches:
        reason = str(b.get("reason", ""))
        color = BRANCH_REASONS.get(reason, "#94a3b8")
        nodes = b.get("nodes") or []
        branch_nodes = "".join(
            f'<div class="prov-branch-artifact"><span class="prov-commit-dot branch" style="--bc:{color}"></span>'
            f'<span>{esc(str(n.get("title", "")))}</span></div>'
            for n in nodes
        )
        branch_html += (
            f'<div class="prov-branch" style="--bc:{color}">'
            f'<div class="prov-branch-head">'
            f'<span class="prov-branch-reason" style="background:{color}22;color:{color}">{esc(reason or "分支")}</span>'
            f'<span class="prov-branch-title">{esc(str(b.get("title", "")))}</span>'
            f"</div>{branch_nodes}</div>"
        )

    # 合并
    merge_html = "".join(
        f'<div class="prov-merge">'
        f'<span class="prov-merge-icon">🔀</span>'
        f'<span class="prov-merge-title">{esc(str(m.get("title", "合并")))}</span>'
        f'<span class="prov-merge-result">{esc(str(m.get("result", "")))}</span>'
        f"</div>"
        for m in merges
    )

    html_content = (
        f'<div class="prov-branch-panel">'
        f'<div class="prov-header"><h3>分支合并溯源</h3>'
        f'<span class="prov-branch-count">{len(branches)} 分支 · {len(merges)} 合并</span></div>'
        f'<div class="prov-branch-graph">'
        f'<div class="prov-mainline">{main_html or "<p class=prov-empty>暂无主线</p>"}</div>'
        f'<div class="prov-branches">{branch_html}</div>'
        f'<div class="prov-merges">{merge_html}</div>'
        f"</div></div>"
    )

    config = {
        "type": "branch_merge",
        "mainline_count": len(mainline),
        "branch_count": len(branches),
        "merge_count": len(merges),
        "reason_colors": BRANCH_REASONS,
    }
    descriptor = build_panel_descriptor(
        artifact, html_content, config,
        [],
        {"renderer": "BranchMerge", "branches": len(branches), "merges": len(merges)},
        "BranchMerge",
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor
