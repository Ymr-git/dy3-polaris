"""L7 渲染器 — GraphRenderer (application/vnd.dy3.graph+json).

将知识图谱 Artifact 渲染为 vis.js Network 可消费的配置。
服务端完成节点/边归一化、BKT 学情着色与布局模式选择。

实现能力 (对应 L7 设计文档 §2.4 + §4.2.4):

1. **图谱数据**: payload["nodes"] (kp_id/name/domain/...) + payload["edges"]
   (source/target/label 关系类型: 基础/进阶/关联)。
2. **BKT 学情着色** (§2.4.2):
   - P(L)>0.8 绿 (透明度降低) / 0.5-0.8 靛蓝 / <0.5 琥珀 (加粗)
   - 瓶颈 KP (P(L)>0.7 且 P(K|L)<0.3) 红色脉冲动画
3. **布局模式** (§2.4.1): 力导向 (默认, 斥力/引力可调) / 层级
   (按前置依赖自上而下, A/B/C/D 四域着色)。
4. **交互** (§2.4.3): 点击展开 / 右键菜单 / 拖拽 / 路径查询 / 节点过滤
   (前端注册, 服务端生成配置骨架)。
5. **学习路径推荐 DAG** (§5.2.4): payload["learning_path"] 渲染
   当前应学 (琥珀高亮) / 后续待学 (灰色虚线) / 推荐路径 (粗实线)。

融合世界先进方案:
    - vis.js Network options: 声明式物理/交互配置
    - Neo4j 图谱语义: 节点/边/关系类型
    - WCAG: 色盲友好配色

输出契约:
    RenderDescriptor.html   — 挂载壳 (data-graph-id)
    RenderDescriptor.config — {nodes, edges, options, layout, mastery}
    RenderDescriptor.assets — [vis-network.min.js]
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ._common import (
    get_bkt_state,
    get_kp_state,
    is_bottleneck,
    kp_name,
    mastery_color,
    build_descriptor,
    wrap,
)

#: 支持的 MIME 类型
_MIME_TYPES: list[str] = ["application/vnd.dy3.graph+json"]

#: 域颜色 (层级布局时四域区分; 色盲友好时切换蓝橙系)
_DOMAIN_COLORS: dict[str, str] = {
    "A": "#4b3fe3",
    "B": "#22a5f7",
    "C": "#f59e0b",
    "D": "#10b981",
}
_DOMAIN_COLORS_CB: dict[str, str] = {
    "A": "#2563eb",
    "B": "#0ea5e9",
    "C": "#f97316",
    "D": "#16a34a",
}

#: 关系类型 → 边样式
_EDGE_LABELS: dict[str, dict[str, Any]] = {
    "基础": {"color": {"color": "#94a3b8"}, "width": 1.5, "dashes": False},
    "进阶": {"color": {"color": "#818cf8"}, "width": 2, "dashes": False},
    "关联": {"color": {"color": "#94a3b8"}, "width": 1, "dashes": True},
}


class GraphRenderer:
    """图谱渲染器 — 知识图谱 → vis.js Network 配置 (服务端构建)."""

    _MIME_TYPES: list[str] = list(_MIME_TYPES)

    def render(self, artifact: Artifact, context: RenderContext):
        started = time.monotonic()
        if artifact is None or not artifact.payload:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload", detail="Graph artifact requires non-empty payload"
            )
        payload = artifact.payload
        if "nodes" not in payload or "edges" not in payload:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload",
                missing_fields=["nodes", "edges"],
                detail="Graph artifact requires 'nodes' and 'edges' in payload",
            )

        theme = (context.theme if context else "light") or "light"
        colorblind = bool(payload.get("colorblind"))
        layout_mode = str(payload.get("layout", "force")).lower()
        if layout_mode not in ("force", "hierarchical"):
            layout_mode = "force"

        bkt_state = get_bkt_state(artifact, context)
        nodes = self._build_nodes(payload.get("nodes") or [], bkt_state, theme, colorblind, layout_mode)
        edges = self._build_edges(payload.get("edges") or [], payload.get("learning_path"))
        options = self._build_options(theme, colorblind, layout_mode)

        html = wrap(
            f'<div class="l7-graph" data-graph-id="{artifact.artifact_id}" '
            f'style="width:100%;height:{payload.get("height", 420)}px"></div>',
            "l7-graph-wrap",
            theme,
        )
        config = {
            "nodes": nodes,
            "edges": edges,
            "options": options,
            "layout": layout_mode,
            "mastery": self._mastery_summary(bkt_state),
            "interactions": {
                "click_expand": True,
                "context_menu": True,
                "drag": True,
                "path_query": True,
                "filter": {"by_domain": True, "by_mastery": True},
            },
        }
        descriptor = build_descriptor(
            artifact,
            html=html,
            config=config,
            assets=[
                "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js",
            ],
            metadata={
                "renderer": "GraphRenderer",
                "node_count": len(nodes),
                "edge_count": len(edges),
                "layout": layout_mode,
            },
        )
        descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
        return descriptor

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)

    # ----------------------------------------------------------
    # 内部实现
    # ----------------------------------------------------------

    def _build_nodes(
        self,
        raw_nodes: list[dict[str, Any]],
        bkt_state: dict[str, Any],
        theme: str,
        colorblind: bool,
        layout_mode: str,
    ) -> list[dict[str, Any]]:
        """构建 vis.js 节点 (含 BKT 学情着色)."""
        domain_colors = _DOMAIN_COLORS_CB if colorblind else _DOMAIN_COLORS
        nodes: list[dict[str, Any]] = []
        for raw in raw_nodes:
            kp_id = str(raw.get("id") or raw.get("kp_id") or "")
            name = str(raw.get("name") or kp_name(kp_id) or kp_id)
            domain = str(raw.get("domain") or (kp_id.split("-")[0] if "-" in kp_id else ""))
            state = get_kp_state(bkt_state, kp_id)

            p_l = state.get("p_l") if state else None
            bn = is_bottleneck(state)
            color = mastery_color(p_l, theme, colorblind) if p_l is not None else (
                domain_colors.get(domain, "#94a3b8")
            )

            node: dict[str, Any] = {
                "id": kp_id,
                "label": f"{kp_id}\n{name}" if len(name) <= 8 else kp_id,
                "title": self._node_tooltip(kp_id, name, state),
                "color": {"background": color, "border": color, "highlight": {"background": color, "border": "#ffffff"}},
                "font": {"color": theme == "dark" and "#e5e5e5" or "#171717", "size": 13},
                "shape": "dot",
                "size": self._node_size(raw, p_l),
                "data": {"kp_id": kp_id, "domain": domain, "mastery": p_l},
            }
            # 掌握节点降透明度 / 薄弱节点加粗边框
            if p_l is not None:
                if p_l > 0.8:
                    node["opacity"] = 0.85
                elif p_l < 0.5:
                    node["borderWidth"] = 2.5
            if bn:
                node["borderWidth"] = 3
                node["shadow"] = {"color": "#ef4444", "size": 12}
                node["data"]["bottleneck"] = True
            nodes.append(node)
        return nodes

    @staticmethod
    def _node_size(raw: dict[str, Any], p_l: float | None) -> int:
        """节点大小 = 被依赖度 (raw["dependents"]) 或默认 14."""
        deps = raw.get("dependents")
        if isinstance(deps, (int, float)):
            return int(10 + min(float(deps), 24))
        return 14

    @staticmethod
    def _node_tooltip(kp_id: str, name: str, state: dict[str, float] | None) -> str:
        """节点悬浮详情."""
        parts = [f"<b>{kp_id}</b> {name}"]
        if state:
            parts.append(
                f"P(L)={state['p_l']:.2f} · P(K|L)={state['p_k_l']:.2f}"
                f" · P(G)={state['p_g']:.2f} · P(S)={state['p_s']:.2f}"
            )
        else:
            parts.append("未纳入 BKT 追踪")
        return "<br>".join(parts)

    def _build_edges(
        self,
        raw_edges: list[dict[str, Any]],
        learning_path: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """构建 vis.js 边 (关系类型样式 + 学习路径高亮)."""
        edges: list[dict[str, Any]] = []
        path_ids: set[str] = set()
        current_ids: set[str] = set()
        if learning_path:
            path_ids = {str(n) for n in (learning_path.get("recommended") or [])}
            current_ids = {str(n) for n in (learning_path.get("current") or [])}

        for raw in raw_edges:
            source = str(raw.get("source") or raw.get("from") or "")
            target = str(raw.get("target") or raw.get("to") or "")
            label = str(raw.get("label") or raw.get("relation") or "")
            style = _EDGE_LABELS.get(label, _EDGE_LABELS["关联"])

            edge: dict[str, Any] = {
                "from": source,
                "to": target,
                "label": label or None,
                "arrows": "to",
                "color": dict(style["color"]),
                "width": style["width"],
                "dashes": style["dashes"],
                "smooth": {"type": "dynamic"},
                "font": {"size": 11, "strokeWidth": 3, "color": "#64748b"},
            }
            # 学习路径推荐: 推荐路径粗实线 (brand), 当前应学高亮
            if source in path_ids and target in path_ids:
                edge["width"] = 3.5
                edge["color"]["color"] = "#4b3fe3"
            if source in current_ids or target in current_ids:
                edge["color"]["color"] = "#d97706"
            edges.append(edge)
        return edges

    @staticmethod
    def _build_options(theme: str, colorblind: bool, layout_mode: str) -> dict[str, Any]:
        """构建 vis.js Network options."""
        text_color = "#e5e5e5" if theme == "dark" else "#171717"
        edge_color = "rgba(148,163,184,0.6)" if theme == "dark" else "rgba(100,116,139,0.5)"

        options: dict[str, Any] = {
            "autoResize": True,
            "interaction": {
                "hover": True,
                "tooltipDelay": 200,
                "navigationButtons": True,
                "keyboard": {"enabled": True, "bindToWindow": False},
            },
            "physics": {"enabled": True, "stabilization": {"iterations": 200}},
            "nodes": {"font": {"color": text_color}, "borderWidth": 1.5},
            "edges": {
                "color": {"inherit": False, "color": edge_color},
                "smooth": {"type": "dynamic"},
            },
        }

        if layout_mode == "hierarchical":
            options["physics"] = {"enabled": False}
            options["layout"] = {
                "hierarchical": {
                    "direction": "UD",
                    "sortMethod": "directed",
                    "nodeSpacing": 140,
                    "levelSeparation": 120,
                    "treeSpacing": 180,
                }
            }
        else:
            # 力导向: 斥力/引力可调 (设计文档 §2.4.1)
            options["physics"] = {
                "enabled": True,
                "forceAtlas2Based": {
                    "gravitationalConstant": -60,
                    "centralGravity": 0.01,
                    "springLength": 140,
                    "springConstant": 0.06,
                    "damping": 0.4,
                },
                "maxVelocity": 40,
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 250, "updateInterval": 25},
            }
        return options

    @staticmethod
    def _mastery_summary(bkt_state: dict[str, Any]) -> dict[str, Any]:
        """掌握度统计摘要 (供前端图例/筛选)."""
        if not bkt_state:
            return {"tracked": 0, "mastered": 0, "learning": 0, "weak": 0, "bottlenecks": 0}
        mastered = learning = weak = bottlenecks = 0
        for kp_id, state in bkt_state.items():
            if not isinstance(state, dict) or state.get("p_l") is None:
                continue
            p_l = float(state["p_l"])
            if p_l > 0.8:
                mastered += 1
            elif p_l >= 0.5:
                learning += 1
            else:
                weak += 1
            if is_bottleneck(state):
                bottlenecks += 1
        return {
            "tracked": len(bkt_state),
            "mastered": mastered,
            "learning": learning,
            "weak": weak,
            "bottlenecks": bottlenecks,
        }
