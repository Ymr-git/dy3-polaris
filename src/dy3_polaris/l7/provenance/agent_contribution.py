"""L7 溯源可视化 — Agent 交互链时间线 (agent_contribution.py).

任务拆分 T5 · 设计文档 Ch.6.3。

将 4-Agent 协同过程重塑为「逐步时间线」(第1步 → 第2步 → …)，
取代此前的「每个 Agent 交互次数」聚合条形图。每一步展示:

- 第 N 步编号 + 参与 Agent (图标 + 名称)
- 该 Agent 做了什么 (action)
- 传给了谁 (广播频道 + 接收 Agent, 或交接给下一环节)
- 阶段 / 交互类型 / 耗时 / 状态

数据来源: L5 InteractionRecorder 的 InteractionRecord 列表
(agent_id / agent_name / action / related_agents / channel /
 phase / phase_order / duration_ms / status / interaction_type)。
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ..renderers._common import esc
from ._common import build_panel_descriptor

#: Agent ID → 展示元数据 (图标/颜色), 与前端 dp-collab.js 对齐
_AGENT_META: dict[str, dict[str, str]] = {
    "agent.learning.diagnosis": {"name": "学情诊断", "icon": "🔍", "color": "#3b82f6"},
    "agent.knowledge.generation": {"name": "知识生成", "icon": "🧠", "color": "#8b5cf6"},
    "agent.quality.review": {"name": "审核校验", "icon": "🛡️", "color": "#10b981"},
    "agent.guidance.decision": {"name": "导学决策", "icon": "🎯", "color": "#f59e0b"},
    "cross.validate": {"name": "交叉验证", "icon": "⚖️", "color": "#6366f1"},
    "debate.pro": {"name": "正方辩论", "icon": "💬", "color": "#0ea5e9"},
    "debate.con": {"name": "反方辩论", "icon": "💬", "color": "#f43f5e"},
    "debate.vote": {"name": "投票裁决", "icon": "🗳️", "color": "#14b8a6"},
    "agent.adjudicator": {"name": "仲裁者", "icon": "⚖️", "color": "#eab308"},
}

#: 阶段 → 展示标签/颜色
_PHASE_META: dict[str, dict[str, str]] = {
    "diagnosis": {"label": "诊断阶段", "color": "#3b82f6"},
    "generation": {"label": "生成阶段", "color": "#8b5cf6"},
    "review": {"label": "审核阶段", "color": "#10b981"},
    "decision": {"label": "决策阶段", "color": "#f59e0b"},
    "feedback": {"label": "反馈阶段", "color": "#ec4899"},
    "orchestration": {"label": "编排阶段", "color": "#6366f1"},
    "system": {"label": "系统事件", "color": "#6b7280"},
}

#: 交互类型 → 图标/标签
_TYPE_META: dict[str, dict[str, str]] = {
    "agent_execution": {"icon": "⚡", "label": "执行"},
    "broadcast_send": {"icon": "📡", "label": "广播"},
    "broadcast_receive": {"icon": "📥", "label": "接收"},
    "pipeline_step": {"icon": "🔗", "label": "流水线"},
    "debate_round": {"icon": "💬", "label": "辩论"},
    "voting": {"icon": "🗳️", "label": "投票"},
    "feedback_loop": {"icon": "🔄", "label": "反馈"},
    "error": {"icon": "❌", "label": "错误"},
}


def _agent_meta(agent_id: str, agent_name: str = "") -> dict[str, str]:
    """Agent ID → 展示元数据 (已知 Agent 用映射, 未知回退到名称/ID)."""
    known = _AGENT_META.get(agent_id)
    if known:
        return known
    fallback = agent_name or (agent_id.split(".")[-1] if "." in agent_id else agent_id)
    return {"name": fallback or agent_id or "未知", "icon": "🤖", "color": "#6b7280"}


def _short_id(agent_id: str) -> str:
    """agent.learning.diagnosis → diagnosis (简短显示)."""
    return agent_id.split(".")[-1] if "." in agent_id else agent_id


def _str_val(value: Any) -> str:
    """把值归一化为字符串 (兼容 str 枚举 / 普通字符串)."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _as_float(value: Any) -> float:
    """安全转 float (非法值回退 0.0, 不抛异常)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_interactions(records: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """把 InteractionRecord 字典列表归一化为「逐步时间线」步骤.

    按 phase_order 升序 (缺失则按 timestamp 兜底), 每步计算:
        step / agent / color / action / passed_to / channel /
        phase_label / phase_color / type_icon / duration / status /
        next_agent (交接给下一环节, 供连续流展示) / is_last
    """
    clean = [r for r in (records or []) if isinstance(r, dict)]
    ordered = sorted(
        clean,
        key=lambda r: (
            r.get("phase_order") if isinstance(r.get("phase_order"), int) else 1 << 30,
            _as_float(r.get("timestamp")),
        ),
    )

    steps: list[dict[str, Any]] = []
    for i, r in enumerate(ordered):
        aid = str(r.get("agent_id") or "")
        am = _agent_meta(aid, str(r.get("agent_name") or ""))

        # 传给谁: related_agents + output_summary.receivers (广播接收者)
        related = r.get("related_agents") or []
        if not isinstance(related, list):
            related = []
        out = r.get("output_summary") or {}
        receivers = out.get("receivers") if isinstance(out, dict) else []
        if isinstance(receivers, list):
            related = list(dict.fromkeys([*related, *receivers]))
        channel = str(r.get("channel") or (out.get("channel") if isinstance(out, dict) else "") or "")

        phase = _str_val(r.get("phase")) or "system"
        pm = _PHASE_META.get(phase, {"label": phase, "color": "#6b7280"})
        itype = _str_val(r.get("interaction_type")) or "agent_execution"
        tm = _TYPE_META.get(itype, {"icon": "⚡", "label": itype})

        # 交接给下一环节 (连续流)
        next_r = ordered[i + 1] if i + 1 < len(ordered) else None
        next_aid = str(next_r.get("agent_id") or "") if next_r else ""
        next_am = _agent_meta(next_aid, str(next_r.get("agent_name") or "")) if next_r else None

        steps.append({
            "step": i + 1,
            "agent_id": aid,
            "agent_name": am["name"],
            "agent_icon": am["icon"],
            "color": am["color"],
            "action": str(r.get("action") or ""),
            "passed_to": [str(x) for x in related],
            "channel": channel,
            "phase": phase,
            "phase_label": pm["label"],
            "phase_color": pm["color"],
            "type_icon": tm["icon"],
            "type_label": tm["label"],
            "duration_ms": _as_float(r.get("duration_ms")),
            "status": str(r.get("status") or "completed"),
            "timestamp": _as_float(r.get("timestamp")),
            "next_agent": next_am["name"] if next_am else "",
            "is_last": next_r is None,
        })
    return steps


def _fmt_duration(ms: float) -> str:
    """毫秒 → 可读耗时 (1234ms / 2.3s)."""
    if ms <= 0:
        return ""
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def _fmt_time(ts: float) -> str:
    if not ts:
        return "--"
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _render_step(s: dict[str, Any]) -> str:
    """渲染单个步骤为 HTML (内联样式, 自包含)."""
    color = s["color"]
    # 顶部: 步骤编号 + Agent
    header = (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<span style="flex:none;background:{color};color:#fff;font-size:11px;font-weight:700;'
        f'padding:2px 9px;border-radius:999px">第{s["step"]}步</span>'
        f'<span style="font-size:15px">{s["agent_icon"]}</span>'
        f'<span style="font-weight:600;font-size:14px;color:var(--ink,#1f2937)">{esc(s["agent_name"])}</span>'
        f'</div>'
    )
    # 动作
    action = (
        f'<div style="margin-top:5px;font-size:13px;color:var(--ink,#1f2937);line-height:1.5">'
        f'{esc(s["action"]) or "（未记录动作）"}</div>'
    )
    # 传给谁
    pass_html = ""
    if s["passed_to"]:
        names = "、".join(_agent_meta(x)["name"] for x in s["passed_to"])
        chan = f' <code style="font-size:10.5px;background:var(--surface2,#f4f4f5);padding:1px 6px;border-radius:4px;color:#7c3aed">{esc(s["channel"])}</code>' if s["channel"] else ""
        pass_html = (
            f'<div style="margin-top:6px;font-size:12px;color:#4b5563">'
            f'📡 传给 <strong>{esc(names)}</strong>{chan}</div>'
        )
    elif s["next_agent"] and s["next_agent"] != s["agent_name"]:
        pass_html = (
            f'<div style="margin-top:6px;font-size:12px;color:#4b5563">'
            f'→ 交接给 <strong>{esc(s["next_agent"])}</strong></div>'
        )
    elif s["is_last"]:
        pass_html = (
            f'<div style="margin-top:6px;font-size:12px;color:#16a34a">✅ 产出最终结果</div>'
        )
    # 元信息: 阶段 / 类型 / 耗时 / 状态
    status_color = {"completed": "#16a34a", "failed": "#dc2626", "timeout": "#d97706", "pending": "#6b7280"}.get(s["status"], "#6b7280")
    dur = _fmt_duration(s["duration_ms"])
    meta_bits = [
        f'<span style="background:{s["phase_color"]}1a;color:{s["phase_color"]};padding:1px 8px;border-radius:999px;font-size:11px">{esc(s["phase_label"])}</span>',
        f'<span style="font-size:11px;color:#6b7280">{s["type_icon"]} {esc(s["type_label"])}</span>',
    ]
    if dur:
        meta_bits.append(f'<span style="font-size:11px;color:#6b7280">⏱ {esc(dur)}</span>')
    meta_bits.append(f'<span style="color:{status_color};font-size:11px">● {esc(s["status"])}</span>')
    meta = (
        f'<div style="margin-top:7px;display:flex;flex-wrap:wrap;gap:8px;align-items:center">'
        f'{"".join(meta_bits)}</div>'
    )
    # 连接线 (非最后一步)
    connector = "" if s["is_last"] else (
        f'<div style="position:absolute;left:24px;bottom:-13px;top:auto;width:2px;height:13px;background:#e5e7eb"></div>'
    )
    return (
        f'<div style="position:relative;margin-bottom:12px;padding:12px 14px 12px 16px;'
        f'border:1px solid var(--rule,#e5e7eb);border-left:3px solid {color};border-radius:10px;'
        f'background:var(--surface,#fff)">{connector}{header}{action}{pass_html}{meta}</div>'
    )


def render_agent_contribution(
    artifact: Artifact | None = None,
    context: RenderContext | None = None,
    agents: list[dict[str, Any]] | None = None,
    interactions: list[dict[str, Any]] | None = None,
    theme: str = "light",
) -> dict[str, Any]:
    """渲染 Agent 交互链时间线 (逐步: 谁 → 做了什么 → 传给谁).

    Args:
        agents: 可选, Agent 聚合统计 (向后兼容, 仅用于摘要; 不再渲染条形图)。
        interactions: InteractionRecord 字典列表 (逐步时间线主数据源)。

    Returns:
        RenderDescriptor (html + config)。
    """
    started = time.monotonic()
    agents = agents or []
    steps = normalize_interactions(interactions)

    # 参与 Agent 去重 (优先来自时间线, 兜底用 agents 参数)
    seen: dict[str, str] = {}
    for s in steps:
        seen.setdefault(s["agent_id"], s["agent_name"])
    agent_ids = list(seen.keys()) or [str(a.get("id") or a.get("name") or "?") for a in agents]
    agent_count = len(agent_ids)

    # 头部摘要
    summary_bits = [
        f'<div style="font-size:12px;color:var(--muted,#6b7280);line-height:1.7">'
        f'共 <strong style="color:var(--ink,#1f2937)">{len(steps)}</strong> 步 · '
        f'参与 <strong style="color:var(--ink,#1f2937)">{agent_count}</strong> 个 Agent · '
        f'按时间顺序呈现「谁 → 做了什么 → 传给谁」的完整协作链。</div>'
    ]
    if steps:
        first, last = steps[0], steps[-1]
        summary_bits.append(
            f'<div style="margin-top:4px;font-size:12px;color:var(--muted,#6b7280)">'
            f'{first["agent_icon"]} {esc(first["agent_name"])} → ⋯ → {last["agent_icon"]} {esc(last["agent_name"])}</div>'
        )

    # 步骤正文
    body = "\n".join(_render_step(s) for s in steps) if steps else '<p class="prov-empty">暂无交互记录 — 发起一次多智能体问答后即可查看逐步协作链路。</p>'

    html_content = (
        f'<div class="prov-contribution-panel">'
        f'<div class="prov-header"><h3>Agent 交互链时间线</h3></div>'
        f'{"".join(summary_bits)}'
        f'<div class="prov-list" role="list" aria-live="polite" style="margin-top:12px">{body}</div>'
        f"</div>"
    )

    config: dict[str, Any] = {
        "type": "agent_contribution",
        "step_count": len(steps),
        "agent_count": agent_count,
        "agent_ids": agent_ids,
        "steps": steps,
        "agents": agents,
    }
    descriptor = build_panel_descriptor(
        artifact, html_content, config,
        None,
        {"renderer": "AgentContribution", "steps": len(steps), "agents": agent_count},
        "AgentContribution",
    )
    descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
    return descriptor


__all__ = ["render_agent_contribution", "normalize_interactions"]
