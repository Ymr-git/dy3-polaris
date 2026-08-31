"""CC3 溯源捕获层 — 可视化适配器 (Provenance Visualizer Adapter).

将 CC3 双核心数据 (KPA 标注 / DL 辩论日志 / 溯源链) 转换为
前端可视化库可直接消费的结构化数据, 适配 L7 ProvenanceRenderer 渲染需求。

支持的四种可视化格式:
1. Cytoscape.js — 溯源网络图 (节点: KPA 标注/来源/Agent; 边: wasDerivedFrom/wasGeneratedBy/annotates)
2. D3.js hierarchy — 溯源链层级树 (按架构层分组的链式拓扑)
3. Mermaid — 流程图 (溯源链流向 / 单标注溯源关系)
4. ECharts — 辩论时间线 (轮次交锋) 与 七维完整度雷达图

核心能力:
- to_cytoscape(): 生成 Cytoscape.js elements 图数据 (含来源等级着色、Agent 角色分类)
- to_d3_hierarchy(): 生成 D3.hierarchy 嵌套树 (按 layer 分组, 保留链序与哈希链接)
- to_mermaid(): 生成 Mermaid flowchart 文本 (chain 或 annotation 视角)
- to_echarts_timeline(): 生成辩论时间线 ECharts option (Generator/Reviewer 双轨 + 分歧度曲线)
- to_echarts_radar(): 生成七维完整度雷达图 ECharts option (含 CC1 四层评分叠加)
- export_all(): 一键导出全部格式, 适配 ProvenanceRenderSchema v2.0

设计要点:
- 仅依赖 Python 标准库, 不引入可视化库运行时依赖
- 数据访问失败时优雅降级 (返回空结构 + error 字段), 不中断渲染
- 节点 ID 做去重与转义, 兼容各前端库的标识符约束
- 所有时间戳以 Unix epoch (秒) 输出, 由前端负责时区/格式化

融合方案:
- W3C PROV: Entity-Activity-Agent 关系映射为图节点与边
- Cytoscape.js: 力导向网络图标准格式 (data.id / data.source / data.target)
- ECharts: 雷达图 indicator + series 标准结构
- Mermaid: flowchart LR/TD 文本图表语法
- L7 ProvenanceRenderer: ProvenanceRenderSchema v2.0 数据契约
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .kpa_engine import KPAEngine
from .debate_logger import DebateLogger
from .provenance_chain_builder import ProvenanceChainBuilder

logger = logging.getLogger(__name__)


# ============================================================
# 七维完整度维度定义 (顺序固定, 用于雷达图)
# ============================================================

#: 七维标注维度名称 (中文标签 / 英文 key / 完整度方法名后缀)
SEVEN_DIMENSIONS: list[dict[str, str]] = [
    {"key": "source", "label": "来源"},
    {"key": "generation", "label": "生成"},
    {"key": "validation", "label": "校验"},
    {"key": "decision", "label": "决策"},
    {"key": "evolution", "label": "演化"},
    {"key": "propagation", "label": "传播"},
    {"key": "relation", "label": "关联"},
]

#: CC1 四层评分维度 (用于雷达图第二组数据)
FOUR_LAYER_DIMENSIONS: list[dict[str, str]] = [
    {"key": "factual", "label": "事实性"},
    {"key": "logical", "label": "逻辑性"},
    {"key": "numerical", "label": "数值性"},
    {"key": "provenance", "label": "溯源性"},
]

#: 来源等级 -> 着色 (Cytoscape 节点样式)
TIER_COLORS: dict[str, str] = {
    "tier_1": "#1a237e",   # 深靛蓝 (顶级期刊/标准)
    "tier_2": "#6a1b9a",   # 紫罗兰 (权威教材/综述)
    "tier_3": "#757575",   # 灰色 (预印本/会议)
    "tier_4": "#bdbdbd",   # 浅灰 (内部文档)
    "tier_5": "#e0e0e0",   # 极浅灰 (未验证)
}

#: Agent 角色 -> 节点形状 (Cytoscape 节点样式)
AGENT_ROLE_SHAPES: dict[str, str] = {
    "generator": "triangle",
    "reviewer": "rectangle",
    "adjudicator": "diamond",
    "annotator": "ellipse",
}


def _sanitize_id(raw: str, prefix: str = "n") -> str:
    """生成可视化安全的节点 ID.

    Mermaid/Cytoscape 节点 ID 不能含空格与部分特殊字符,
    此函数将任意字符串转换为稳定的短哈希式 ID。

    Args:
        raw: 原始字符串 (可能含空格/斜杠/冒号等)
        prefix: ID 前缀 (区分节点类型)

    Returns:
        形如 ``prefix-a1b2c3d4`` 的安全 ID
    """
    if not raw:
        return f"{prefix}-empty"
    # 取字符的 ord 拼接后做简单折叠, 产生稳定的 8 位十六进制风格短码
    digest = 0
    for ch in raw:
        digest = (digest * 131 + ord(ch)) & 0xFFFFFFFF
    return f"{prefix}-{digest & 0xFFFFFFFF:08x}"


def _enum_value(val: Any) -> Any:
    """提取枚举的 value, 非枚举原样返回."""
    return val.value if hasattr(val, "value") else val


class ProvenanceVisualizer:
    """溯源可视化适配器 — 多格式可视化数据生成.

    将 KPA 标注引擎、辩论日志引擎、溯源链构建器的内部数据
    转换为 Cytoscape.js / D3.js / Mermaid / ECharts 四种前端格式。

    设计原则:
    - 只读访问: 不修改引擎内部状态, 仅消费公开查询接口
    - 优雅降级: 单条数据缺失或异常不影响整体导出, 错误记入 error 字段
    - 去重幂等: 重复调用产生一致结果 (基于当前引擎快照)

    使用示例::

        kpa = KPAEngine()
        dl = DebateLogger()
        chain = ProvenanceChainBuilder()

        viz = ProvenanceVisualizer(kpa, dl, chain)

        # 溯源网络图 (Cytoscape.js)
        graph = viz.to_cytoscape(target_id="kp-dy3-yag-4f")

        # 辩论时间线 (ECharts)
        timeline = viz.to_echarts_timeline(debate_log_id="dl-xxxx")

        # 七维完整度雷达图 (ECharts)
        radar = viz.to_echarts_radar(annotation_id="kpa-xxxx")

        # 一键导出全部格式
        bundle = viz.export_all(target_id="kp-dy3-yag-4f")
    """

    def __init__(
        self,
        kpa_engine: KPAEngine,
        debate_logger: DebateLogger,
        chain_builder: ProvenanceChainBuilder,
    ) -> None:
        """初始化可视化适配器.

        Args:
            kpa_engine: KPA 七维标注引擎实例
            debate_logger: 辩论日志引擎实例
            chain_builder: 溯源链构建器实例
        """
        self._kpa = kpa_engine
        self._dl = debate_logger
        self._chain = chain_builder

    # ==========================================================
    # 1. Cytoscape.js — 溯源网络图
    # ==========================================================

    def to_cytoscape(self, target_id: str = "") -> dict[str, Any]:
        """生成 Cytoscape.js 格式的溯源网络图数据.

        构建 W3C PROV 风格的溯源图:
        - 节点类型: annotation (KPA 标注) / source (来源) / agent (生成者) / knowledge (知识点)
        - 边类型: wasDerivedFrom (来源->标注) / wasGeneratedBy (agent->标注) /
                  annotates (标注->知识点) / prerequisite (前置知识) / same_domain (同域关联)

        节点 data 包含着色与形状提示:
        - source 节点: ``trust_tier`` 字段, 前端可按 TIER_COLORS 着色
        - agent 节点: ``role`` 字段, 前端可按 AGENT_ROLE_SHAPES 选形状
        - annotation 节点: ``completeness`` 字段 (0.0-1.0)

        Args:
            target_id: 目标对象 ID (空=全量标注, 非空=仅该目标的标注子图)

        Returns:
            Cytoscape.js elements 结构::

                {
                    "format": "cytoscape.js",
                    "elements": {
                        "nodes": [{"data": {"id", "label", "type", ...}}],
                        "edges": [{"data": {"id", "source", "target", "label"}}],
                    },
                    "metadata": {"target_id", "node_count", "edge_count", "generated_at"},
                    "errors": [str, ...],
                }
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        node_ids: set[str] = set()
        edge_ids: set[str] = set()
        errors: list[str] = []

        def add_node(node_id: str, data: dict[str, Any]) -> None:
            if node_id and node_id not in node_ids:
                nodes.append({"data": {"id": node_id, **data}})
                node_ids.add(node_id)

        def add_edge(src: str, tgt: str, label: str, extra: dict[str, Any] | None = None) -> None:
            if not src or not tgt or src == tgt:
                return
            edge_id = f"{src}->{tgt}:{label}"
            if edge_id in edge_ids:
                return
            edge_data: dict[str, Any] = {
                "id": edge_id,
                "source": src,
                "target": tgt,
                "label": label,
            }
            if extra:
                edge_data.update(extra)
            edges.append({"data": edge_data})
            edge_ids.add(edge_id)

        try:
            annotations = self._kpa.list_annotations(limit=10000)
        except Exception as exc:  # noqa: BLE001 - 可视化层需容错
            logger.warning("Cytoscape: 拉取标注列表失败: %s", exc)
            errors.append(f"list_annotations: {exc}")
            annotations = []

        for ann in annotations:
            if target_id and getattr(ann, "target_id", "") != target_id:
                continue

            ann_id = getattr(ann, "annotation_id", "")

            # 标注节点
            try:
                completeness = ann.completeness_score()
            except Exception:  # noqa: BLE001
                completeness = 0.0
            add_node(ann_id, {
                "label": f"KPA: {_truncate(getattr(ann, 'target_id', '') or ann_id, 20)}",
                "type": "annotation",
                "target_type": _enum_value(getattr(ann, "target_type", "")),
                "completeness": round(completeness, 4),
                "immutable_hash": getattr(ann, "immutable_hash", ""),
            })

            # 来源节点 + wasDerivedFrom 边
            source = getattr(ann, "source", None)
            primary_source = getattr(source, "primary_source", "") if source else ""
            if primary_source:
                src_node_id = _sanitize_id(primary_source, prefix="src")
                tier = _enum_value(getattr(source, "trust_tier", "")) if source else ""
                add_node(src_node_id, {
                    "label": _truncate(primary_source, 30),
                    "type": "source",
                    "trust_tier": tier,
                    "color": TIER_COLORS.get(tier, "#757575"),
                    "source_type": getattr(source, "source_type", "") if source else "",
                })
                add_edge(src_node_id, ann_id, "wasDerivedFrom")

                # 次要来源 (虚线边)
                secondary = getattr(source, "secondary_sources", []) if source else []
                for sec in secondary:
                    sec_node_id = _sanitize_id(sec, prefix="src")
                    add_node(sec_node_id, {
                        "label": _truncate(sec, 30),
                        "type": "source",
                        "trust_tier": "tier_3",
                        "color": TIER_COLORS["tier_3"],
                        "secondary": True,
                    })
                    add_edge(sec_node_id, ann_id, "wasDerivedFrom", {"dashed": True})

            # 生成 Agent 节点 + wasGeneratedBy 边
            generation = getattr(ann, "generation", None)
            agent_id = getattr(generation, "agent_id", "") if generation else ""
            if agent_id:
                role = getattr(generation, "agent_role", "annotator") or "annotator"
                add_node(agent_id, {
                    "label": f"Agent: {_truncate(agent_id, 20)}",
                    "type": "agent",
                    "role": role,
                    "shape": AGENT_ROLE_SHAPES.get(role, "ellipse"),
                    "version": getattr(generation, "agent_version", "") if generation else "",
                })
                add_edge(agent_id, ann_id, "wasGeneratedBy")

            # 知识点节点 + annotates 边
            tgt_id = getattr(ann, "target_id", "")
            if tgt_id:
                add_node(tgt_id, {
                    "label": f"KP: {_truncate(tgt_id, 20)}",
                    "type": "knowledge",
                    "metadata": getattr(ann, "target_metadata", {}),
                })
                add_edge(ann_id, tgt_id, "annotates")

            # 关联维度: 前置 / 后继 / 同域关联
            relation = getattr(ann, "relation", None)
            if relation is not None:
                for pre in getattr(relation, "prerequisites", []) or []:
                    add_node(pre, {"label": f"KP: {_truncate(pre, 20)}", "type": "knowledge"})
                    add_edge(pre, tgt_id or ann_id, "prerequisite")
                for suc in getattr(relation, "successors", []) or []:
                    add_node(suc, {"label": f"KP: {_truncate(suc, 20)}", "type": "knowledge"})
                    add_edge(tgt_id or ann_id, suc, "successor")
                for rel in getattr(relation, "same_domain_relations", []) or []:
                    rel_target = rel.get("target_id", "") if isinstance(rel, dict) else ""
                    strength = rel.get("strength", 0.0) if isinstance(rel, dict) else 0.0
                    if rel_target:
                        add_node(rel_target, {
                            "label": f"KP: {_truncate(rel_target, 20)}",
                            "type": "knowledge",
                        })
                        add_edge(
                            tgt_id or ann_id, rel_target, "same_domain",
                            {"strength": round(strength, 4)},
                        )

        return {
            "format": "cytoscape.js",
            "elements": {
                "nodes": nodes,
                "edges": edges,
            },
            "metadata": {
                "target_id": target_id,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "generated_at": time.time(),
            },
            "errors": errors,
        }

    # ==========================================================
    # 2. D3.js hierarchy — 溯源链层级树
    # ==========================================================

    def to_d3_hierarchy(self, chain_id: str) -> dict[str, Any]:
        """从溯源链生成 D3.js hierarchy 嵌套树.

        将线性溯源链按架构层 (layer) 分组, 构建两级嵌套树:
        - 根: 链元数据 (chain_id / node_count / merkle_root)
        - 一级子节点: layer 分组 (按节点首次出现顺序)
        - 二级子节点: 该层下的链节点 (保留链序)

        每个叶子节点携带完整属性 (agent / role / direction / hash 链接),
        便于 D3.tree / D3.cluster 渲染跨层流向。

        Args:
            chain_id: 溯源链 ID

        Returns:
            D3.hierarchy 兼容结构::

                {
                    "name": str,
                    "attributes": {...},
                    "children": [{"name": "L2", "children": [{"name": "node-0", "attributes": {...}]}]}],
                    "metadata": {...},
                    "errors": [str, ...],
                }
        """
        errors: list[str] = []

        try:
            nodes = self._chain.get_chain(chain_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("D3 hierarchy: 获取链失败 chain=%s: %s", chain_id, exc)
            errors.append(f"get_chain: {exc}")
            nodes = []

        # 链元数据
        meta: dict[str, Any] = {"chain_id": chain_id, "node_count": len(nodes)}
        try:
            chain_list = self._chain.list_chains()
            for cm in chain_list:
                if cm.get("chain_id") == chain_id:
                    meta.update(cm)
                    break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"list_chains: {exc}")

        if not nodes:
            return {
                "name": f"chain:{chain_id}",
                "attributes": meta,
                "children": [],
                "metadata": {
                    "format": "d3-hierarchy",
                    "chain_id": chain_id,
                    "total_nodes": len(nodes),
                    "layer_count": 0,
                    "layers": [],
                    "generated_at": time.time(),
                    **meta,
                },
                "errors": errors,
            }

        # 按 layer 分组, 保留首次出现顺序
        layer_order: list[str] = []
        grouped: dict[str, list[Any]] = {}
        for node in nodes:
            layer = getattr(node, "layer", "") or "unknown"
            if layer not in grouped:
                grouped[layer] = []
                layer_order.append(layer)
            grouped[layer].append(node)

        layer_children: list[dict[str, Any]] = []
        for layer in layer_order:
            layer_nodes: list[dict[str, Any]] = []
            for node in grouped[layer]:
                direction = getattr(node, "direction", None)
                layer_nodes.append({
                    "name": (
                        f"node-{getattr(node, 'node_index', '?')}: "
                        f"{_truncate(getattr(node, 'agent_id', '') or 'unknown', 16)}"
                    ),
                    "attributes": {
                        "node_id": getattr(node, "node_id", ""),
                        "node_index": getattr(node, "node_index", 0),
                        "annotation_id": getattr(node, "annotation_id", ""),
                        "target_id": getattr(node, "target_id", ""),
                        "agent_id": getattr(node, "agent_id", ""),
                        "agent_role": getattr(node, "agent_role", ""),
                        "layer": layer,
                        "direction": _enum_value(direction) if direction else "",
                        "timestamp": getattr(node, "timestamp", 0.0),
                        "node_hash": getattr(node, "node_hash", ""),
                        "prev_hash": getattr(node, "prev_hash", ""),
                    },
                    "children": [],
                })
            layer_children.append({
                "name": layer,
                "attributes": {
                    "layer": layer,
                    "node_count": len(layer_nodes),
                },
                "children": layer_nodes,
            })

        return {
            "name": f"chain:{chain_id}",
            "attributes": meta,
            "children": layer_children,
            "metadata": {
                "format": "d3-hierarchy",
                "chain_id": chain_id,
                "total_nodes": len(nodes),
                "layer_count": len(layer_order),
                "layers": layer_order,
                "generated_at": time.time(),
            },
            "errors": errors,
        }

    # ==========================================================
    # 3. Mermaid — 流程图
    # ==========================================================

    def to_mermaid(self, chain_id: str = "", annotation_id: str = "") -> str:
        """生成 Mermaid flowchart 图表字符串.

        支持两种视角 (互斥, chain_id 优先):
        - 链视角 (chain_id 非空): 绘制溯源链节点流向, 边标注跨层方向
        - 标注视角 (annotation_id 非空): 绘制单标注的来源/生成/校验/决策关系

        Args:
            chain_id: 溯源链 ID (优先)
            annotation_id: KPA 标注 ID (chain_id 为空时生效)

        Returns:
            Mermaid flowchart 文本, 可直接嵌入 ``<pre class="mermaid">`` 渲染。
            无数据时返回空图占位。
        """
        if chain_id:
            return self._mermaid_for_chain(chain_id)
        if annotation_id:
            return self._mermaid_for_annotation(annotation_id)
        return "flowchart TD\n    empty[\"暂无溯源数据\"]\n"

    def _mermaid_for_chain(self, chain_id: str) -> str:
        """生成溯源链的 Mermaid 流程图."""
        lines: list[str] = ["flowchart TD"]

        try:
            nodes = self._chain.get_chain(chain_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mermaid: 获取链失败 chain=%s: %s", chain_id, exc)
            return (
                "flowchart TD\n"
                f'    error["链不存在或读取失败: {chain_id}"]\n'
            )

        if not nodes:
            lines.append(f'    empty["链 {chain_id} 无节点"]')
            return "\n".join(lines) + "\n"

        prev_alias = ""
        for node in nodes:
            idx = getattr(node, "node_index", 0)
            alias = f"n{idx}"
            agent = _truncate(getattr(node, "agent_id", "") or "unknown", 20)
            role = getattr(node, "agent_role", "")
            layer = getattr(node, "layer", "")
            label = f"#{idx} {agent}"
            if role:
                label += f" ({role})"
            if layer:
                label += f" @ {layer}"
            # 标签内特殊字符用引号包裹
            lines.append(f'    {alias}["{label}"]')

            if prev_alias:
                direction = getattr(node, "direction", None)
                dir_label = _enum_value(direction) if direction else "next"
                lines.append(f"    {prev_alias} -->|{dir_label}| {alias}")
            prev_alias = alias

        # 哈希链接说明 (注释行)
        if nodes:
            tail = nodes[-1]
            tail_hash = getattr(tail, "node_hash", "")
            if tail_hash:
                lines.append(f'    tail["尾节点哈希: {tail_hash[:12]}..."]')
                lines.append(f"    {prev_alias} -.-> tail")

        return "\n".join(lines) + "\n"

    def _mermaid_for_annotation(self, annotation_id: str) -> str:
        """生成单标注溯源关系的 Mermaid 流程图."""
        lines: list[str] = ["flowchart TD"]

        try:
            ann = self._kpa.get_annotation(annotation_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mermaid: 获取标注失败 id=%s: %s", annotation_id, exc)
            return (
                "flowchart TD\n"
                f'    error["标注不存在或读取失败: {annotation_id}"]\n'
            )

        ann_alias = "ann"
        tgt = getattr(ann, "target_id", "") or annotation_id
        lines.append(f'    {ann_alias}["KPA标注: {_truncate(tgt, 24)}"]')

        # 来源
        source = getattr(ann, "source", None)
        primary = getattr(source, "primary_source", "") if source else ""
        if primary:
            src_alias = "src"
            lines.append(f'    {src_alias}["来源: {_truncate(primary, 24)}"]')
            lines.append(f"    {src_alias} -->|wasDerivedFrom| {ann_alias}")

        # 生成 Agent
        generation = getattr(ann, "generation", None)
        agent = getattr(generation, "agent_id", "") if generation else ""
        if agent:
            gen_alias = "gen"
            lines.append(f'    {gen_alias}["Agent: {_truncate(agent, 20)}"]')
            lines.append(f"    {gen_alias} -->|wasGeneratedBy| {ann_alias}")

        # 标注对象
        if tgt:
            kp_alias = "kp"
            lines.append(f'    {kp_alias}["知识点: {_truncate(tgt, 24)}"]')
            lines.append(f"    {ann_alias} -->|annotates| {kp_alias}")

        # 校验 (CC1)
        validation = getattr(ann, "validation", None)
        cc1 = getattr(validation, "cc1_review_id", "") if validation else ""
        if cc1:
            val_alias = "val"
            lines.append(f'    {val_alias}["CC1评审: {_truncate(cc1, 20)}"]')
            lines.append(f"    {val_alias} -.->|validatedBy| {ann_alias}")

        # 决策 (辩论)
        decision = getattr(ann, "decision", None)
        debate = getattr(decision, "debate_id", "") if decision else ""
        if debate:
            dec_alias = "dec"
            lines.append(f'    {dec_alias}["辩论: {_truncate(debate, 20)}"]')
            lines.append(f"    {dec_alias} -.->|debatedBy| {ann_alias}")

        return "\n".join(lines) + "\n"

    # ==========================================================
    # 4. ECharts — 辩论时间线
    # ==========================================================

    def to_echarts_timeline(self, debate_log_id: str = "") -> dict[str, Any]:
        """生成辩论时间线的 ECharts option 数据.

        适配设计文档 7.2.1 辩论时间线:
        - X 轴: 轮次 (Round 1/2/3 ... + 裁决阶段)
        - 双轨: Generator 论点 (上轨) / Reviewer 反驳 (下轨)
        - 分歧度曲线: 折线 + 阈值 markLine
        - 裁决标记: 末端标注共识立场与三维评分

        Args:
            debate_log_id: 辩论日志 ID (空=聚合全部日志的收敛概览)

        Returns:
            ECharts option 字典, 含 xAxis/yAxis/series/markLine/tooltip。
            可直接作为 ``echarts.setOption()`` 参数。
        """
        if debate_log_id:
            return self._echarts_timeline_single(debate_log_id)
        return self._echarts_timeline_aggregated()

    def _echarts_timeline_single(self, debate_log_id: str) -> dict[str, Any]:
        """单条辩论日志的详细时间线."""
        try:
            log = self._dl.get_log(debate_log_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ECharts timeline: 获取日志失败 id=%s: %s", debate_log_id, exc)
            return self._empty_echarts_option(
                f"辩论日志不存在: {debate_log_id}", error=str(exc)
            )

        rounds = getattr(log, "rounds", []) or []
        divergence_curve = getattr(log, "divergence_curve", []) or []
        threshold = getattr(log, "convergence_threshold", 0.1)

        # X 轴类别: Round 1..N + 裁决
        categories: list[str] = [f"Round {i + 1}" for i in range(len(rounds))]
        has_verdict = getattr(log, "adjudicator_verdict", None) is not None
        if has_verdict:
            categories.append("裁决")

        # Generator 论点数 / Reviewer 反驳数 (每轮)
        gen_counts: list[int] = []
        rev_counts: list[int] = []
        gen_confidences: list[Any] = []  # 每轮平均置信度 (无数据为 None)
        rev_confidences: list[Any] = []

        for rnd in rounds:
            gen_args = getattr(rnd, "generator_arguments", []) or []
            rev_counters = getattr(rnd, "reviewer_counterarguments", []) or []
            gen_counts.append(len(gen_args))
            rev_counts.append(len(rev_counters))
            gen_confidences.append(
                round(sum(a.confidence for a in gen_args) / len(gen_args), 4)
                if gen_args else None
            )
            rev_confidences.append(
                round(sum(c.confidence for c in rev_counters) / len(rev_counters), 4)
                if rev_counters else None
            )

        # 裁决阶段数据点
        if has_verdict:
            gen_counts.append(0)
            rev_counts.append(0)
            gen_confidences.append(None)
            rev_confidences.append(None)

        # 分歧度曲线 (对齐 X 轴, 末位补 None)
        divergence_series = list(divergence_curve)
        while len(divergence_series) < len(categories):
            divergence_series.append(None)

        # 分歧度阈值 markLine
        mark_line = {
            "symbol": "none",
            "silent": True,
            "lineStyle": {"type": "dashed", "color": "#e53935"},
            "data": [{"yAxis": threshold, "label": {"formatter": f"阈值 {threshold}"}}],
        }

        # 裁决信息 (tooltip 富文本)
        verdict_info = ""
        verdict = getattr(log, "adjudicator_verdict", None)
        if verdict is not None:
            score = getattr(verdict, "three_dimensional_score", {}) or {}
            consensus = getattr(verdict, "consensus_position", "")
            verdict_info = (
                f"共识: {consensus[:40]} | "
                f"三维评分: {score}"
            )

        option: dict[str, Any] = {
            "format": "echarts",
            "title": {
                "text": f"辩论时间线: {getattr(log, 'debate_id', debate_log_id)}",
                "subtext": (
                    f"状态: {_enum_value(getattr(log, 'convergence_status', ''))} | "
                    f"最终分歧度: {getattr(log, 'final_divergence', 0.0)}"
                ),
            },
            "tooltip": {"trigger": "axis"},
            "legend": {
                "data": ["Generator论点数", "Reviewer反驳数", "分歧度"],
            },
            "xAxis": {
                "type": "category",
                "data": categories,
                "axisLine": {"onZero": True},
            },
            "yAxis": [
                {
                    "type": "value",
                    "name": "论点/反驳数",
                    "position": "left",
                    "min": 0,
                    "minInterval": 1,
                },
                {
                    "type": "value",
                    "name": "分歧度",
                    "position": "right",
                    "min": 0.0,
                    "max": 1.0,
                },
            ],
            "series": [
                {
                    "name": "Generator论点数",
                    "type": "bar",
                    "yAxisIndex": 0,
                    "data": gen_counts,
                    "itemStyle": {"color": "#43a047"},
                    "label": {"show": True, "position": "top"},
                },
                {
                    "name": "Reviewer反驳数",
                    "type": "bar",
                    "yAxisIndex": 0,
                    "data": rev_counts,
                    "itemStyle": {"color": "#fb8c00"},
                    "label": {"show": True, "position": "bottom"},
                },
                {
                    "name": "分歧度",
                    "type": "line",
                    "yAxisIndex": 1,
                    "data": divergence_series,
                    "smooth": True,
                    "symbol": "circle",
                    "symbolSize": 8,
                    "lineStyle": {"width": 3, "color": "#1e88e5"},
                    "areaStyle": {"opacity": 0.15},
                    "markLine": mark_line,
                },
            ],
            "metadata": {
                "debate_log_id": debate_log_id,
                "debate_id": getattr(log, "debate_id", ""),
                "total_rounds": len(rounds),
                "converged": getattr(log, "convergence_reached", False),
                "convergence_round": getattr(log, "convergence_round", 0),
                "final_divergence": getattr(log, "final_divergence", 0.0),
                "verdict_info": verdict_info,
                "generated_at": time.time(),
            },
        }

        # 论点详情 (供前端展开)
        round_details: list[dict[str, Any]] = []
        for rnd in rounds:
            round_details.append({
                "round": getattr(rnd, "round_number", 0),
                "divergence": getattr(rnd, "round_divergence", 0.0),
                "duration_ms": getattr(rnd, "round_duration_ms", 0.0),
                "generator_confidence_avg": gen_confidences[len(round_details)],
                "reviewer_confidence_avg": rev_confidences[len(round_details)],
            })
        option["round_details"] = round_details
        return option

    def _echarts_timeline_aggregated(self) -> dict[str, Any]:
        """聚合全部辩论日志的收敛概览时间线."""
        try:
            logs = self._dl.list_logs(limit=10000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ECharts timeline: 拉取日志列表失败: %s", exc)
            return self._empty_echarts_option("无辩论日志数据", error=str(exc))

        if not logs:
            return self._empty_echarts_option("暂无辩论日志")

        # 按创建时间排序
        logs_sorted = sorted(logs, key=lambda l: getattr(l, "created_at", 0.0))

        labels: list[str] = []
        divergences: list[Any] = []
        rounds_counts: list[int] = []
        converged_flags: list[int] = []

        for lg in logs_sorted:
            labels.append(getattr(lg, "debate_id", getattr(lg, "debate_log_id", "")))
            divergences.append(round(getattr(lg, "final_divergence", 0.0), 4))
            rounds_counts.append(len(getattr(lg, "rounds", []) or []))
            converged_flags.append(1 if getattr(lg, "convergence_reached", False) else 0)

        converged_total = sum(converged_flags)
        convergence_rate = round(converged_total / len(logs_sorted), 4) if logs_sorted else 0.0

        return {
            "format": "echarts",
            "title": {
                "text": "辩论收敛概览",
                "subtext": f"共 {len(logs_sorted)} 场辩论, 收敛率 {convergence_rate:.1%}",
            },
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["最终分歧度", "轮次数", "收敛"]},
            "xAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {"rotate": 30, "interval": 0},
            },
            "yAxis": [
                {"type": "value", "name": "分歧度", "min": 0, "max": 1},
                {"type": "value", "name": "轮次", "position": "right", "minInterval": 1},
            ],
            "series": [
                {
                    "name": "最终分歧度",
                    "type": "line",
                    "yAxisIndex": 0,
                    "data": divergences,
                    "smooth": True,
                    "symbolSize": 6,
                    "lineStyle": {"color": "#1e88e5", "width": 2},
                    "areaStyle": {"opacity": 0.1},
                },
                {
                    "name": "轮次数",
                    "type": "bar",
                    "yAxisIndex": 1,
                    "data": rounds_counts,
                    "itemStyle": {"color": "#8e24aa"},
                },
                {
                    "name": "收敛",
                    "type": "bar",
                    "yAxisIndex": 1,
                    "data": converged_flags,
                    "itemStyle": {"color": "#43a047"},
                },
            ],
            "metadata": {
                "total_logs": len(logs_sorted),
                "convergence_rate": convergence_rate,
                "generated_at": time.time(),
            },
        }

    def _empty_echarts_option(
        self, message: str, error: str = ""
    ) -> dict[str, Any]:
        """生成空 ECharts option 占位."""
        opt: dict[str, Any] = {
            "format": "echarts",
            "title": {"text": message},
            "series": [],
            "metadata": {"generated_at": time.time(), "empty": True},
        }
        if error:
            opt["metadata"]["error"] = error
        return opt

    # ==========================================================
    # 5. ECharts — 七维完整度雷达图
    # ==========================================================

    def to_echarts_radar(self, annotation_id: str) -> dict[str, Any]:
        """生成七维完整度雷达图的 ECharts option.

        雷达图展示:
        - 主数据: KPA 七维标注完整度 (来源/生成/校验/决策/演化/传播/关联)
        - 叠加数据 (可选): CC1 四层评分 (事实/逻辑/数值/溯源) 重投影到对应维度

        Args:
            annotation_id: KPA 标注 ID

        Returns:
            ECharts option 字典, 含 radar.indicator 与 series。
            标注不存在时返回含 error 的占位 option。
        """
        try:
            ann = self._kpa.get_annotation(annotation_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ECharts radar: 获取标注失败 id=%s: %s", annotation_id, exc)
            return self._empty_echarts_option(
                f"标注不存在: {annotation_id}", error=str(exc)
            )

        # 七维完整度 (优先用引擎评估接口, 失败则直接读模型)
        dim_scores: dict[str, float] = {}
        try:
            report = self._kpa.evaluate_completeness(annotation_id)
            dim_scores = report.get("dimension_scores", {}) or {}
        except Exception as exc:  # noqa: BLE001 - 降级直接读模型
            logger.debug("radar: 引擎评估失败, 降级直读模型: %s", exc)

        if not dim_scores:
            source = getattr(ann, "source", None)
            generation = getattr(ann, "generation", None)
            validation = getattr(ann, "validation", None)
            decision = getattr(ann, "decision", None)
            evolution = getattr(ann, "evolution", None)
            propagation = getattr(ann, "propagation", None)
            relation = getattr(ann, "relation", None)
            dim_scores = {
                "source": source.completeness() if source else 0.0,
                "generation": generation.completeness() if generation else 0.0,
                "validation": validation.completeness() if validation else 0.0,
                "decision": decision.completeness() if decision else 0.0,
                "evolution": evolution.completeness() if evolution else 0.0,
                "propagation": propagation.completeness() if propagation else 0.0,
                "relation": relation.completeness() if relation else 0.0,
            }

        seven_values = [
            round(float(dim_scores.get(d["key"], 0.0)), 4) for d in SEVEN_DIMENSIONS
        ]
        overall = round(sum(seven_values) / len(seven_values), 4) if seven_values else 0.0

        # CC1 四层评分 (重投影到雷达: 归一化到 0-1)
        four_layer_values: list[float] = []
        validation = getattr(ann, "validation", None)
        four_layer_scores = (
            getattr(validation, "four_layer_scores", {}) or {} if validation else {}
        )
        has_four_layer = bool(four_layer_scores)
        if has_four_layer:
            for d in FOUR_LAYER_DIMENSIONS:
                raw = four_layer_scores.get(d["key"], 0.0)
                try:
                    four_layer_values.append(round(float(raw), 4))
                except (TypeError, ValueError):
                    four_layer_values.append(0.0)

        indicator = [
            {"name": d["label"], "max": 1.0} for d in SEVEN_DIMENSIONS
        ]

        series: list[dict[str, Any]] = [
            {
                "name": "七维完整度",
                "type": "radar",
                "data": [{"value": seven_values, "name": "完整度"}],
                "areaStyle": {"opacity": 0.2},
                "lineStyle": {"width": 2, "color": "#1e88e5"},
                "itemStyle": {"color": "#1e88e5"},
            }
        ]

        # 四层评分作为第二条雷达 (重投影: 用前四个维度承载, 后三位补 0)
        if has_four_layer:
            padded = four_layer_values + [0.0] * (len(SEVEN_DIMENSIONS) - len(four_layer_values))
            series.append({
                "name": "CC1四层评分",
                "type": "radar",
                "data": [{"value": padded[:len(SEVEN_DIMENSIONS)], "name": "CC1评分"}],
                "areaStyle": {"opacity": 0.1},
                "lineStyle": {"width": 2, "color": "#e53935", "type": "dashed"},
                "itemStyle": {"color": "#e53935"},
            })

        legend_data = ["七维完整度"]
        if has_four_layer:
            legend_data.append("CC1四层评分")

        return {
            "format": "echarts",
            "title": {
                "text": f"七维完整度: {_truncate(getattr(ann, 'target_id', '') or annotation_id, 24)}",
                "subtext": f"整体完整度: {overall:.1%}",
            },
            "tooltip": {"trigger": "item"},
            "legend": {"data": legend_data},
            "radar": {
                "indicator": indicator,
                "radius": "65%",
                "splitNumber": 5,
                "axisName": {"color": "#37474f"},
            },
            "series": series,
            "metadata": {
                "annotation_id": annotation_id,
                "target_id": getattr(ann, "target_id", ""),
                "overall_completeness": overall,
                "dimension_scores": {d["key"]: v for d, v in zip(SEVEN_DIMENSIONS, seven_values)},
                "four_layer_scores": four_layer_scores if has_four_layer else {},
                "filled_dimensions": getattr(ann, "filled_dimensions", lambda: [])(),
                "missing_dimensions": getattr(ann, "missing_dimensions", lambda: [])(),
                "generated_at": time.time(),
            },
        }

    # ==========================================================
    # 6. 一键导出全部格式
    # ==========================================================

    def export_all(self, target_id: str = "") -> dict[str, Any]:
        """导出全部可视化格式 (适配 ProvenanceRenderSchema v2.0).

        聚合以下数据:
        - cytoscape: 溯源网络图 (target_id 过滤)
        - mermaid_chains: 所有溯源链的 Mermaid 图 (链视角)
        - mermaid_annotations: target_id 关联标注的 Mermaid 图 (标注视角)
        - echarts_radar: 各标注的七维完整度雷达图
        - echarts_timeline: 辩论时间线 (聚合)
        - statistics: KPA / DL / Chain 汇总统计

        单项失败不中断整体导出, 错误汇总至 ``errors``。

        Args:
            target_id: 目标对象 ID (空=全量)

        Returns:
            全格式导出包::

                {
                    "schema": "ProvenanceRenderSchema/2.0",
                    "cytoscape": {...},
                    "mermaid_chains": [str, ...],
                    "mermaid_annotations": [str, ...],
                    "echarts_radars": [{...}],
                    "echarts_timeline": {...},
                    "statistics": {...},
                    "errors": [str, ...],
                    "generated_at": float,
                }
        """
        errors: list[str] = []

        # --- Cytoscape 图 ---
        try:
            cytoscape = self.to_cytoscape(target_id=target_id)
            errors.extend(cytoscape.get("errors", []))
        except Exception as exc:  # noqa: BLE001
            logger.warning("export_all: Cytoscape 生成失败: %s", exc)
            cytoscape = {"format": "cytoscape.js", "elements": {"nodes": [], "edges": []}, "errors": [str(exc)]}
            errors.append(f"cytoscape: {exc}")

        # --- 关联标注列表 ---
        annotations: list[Any] = []
        try:
            if target_id:
                annotations = self._kpa.get_by_target(target_id)
            else:
                annotations = self._kpa.list_annotations(limit=1000)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"annotations: {exc}")

        # --- Mermaid (标注视角) ---
        mermaid_annotations: list[str] = []
        for ann in annotations:
            ann_id = getattr(ann, "annotation_id", "")
            if ann_id:
                try:
                    mermaid_annotations.append(self._mermaid_for_annotation(ann_id))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"mermaid ann {ann_id}: {exc}")

        # --- Mermaid (链视角) ---
        mermaid_chains: list[str] = []
        try:
            chains = self._chain.list_chains()
        except Exception as exc:  # noqa: BLE001
            chains = []
            errors.append(f"list_chains: {exc}")

        for cm in chains:
            cid = cm.get("chain_id", "") if isinstance(cm, dict) else ""
            if cid:
                try:
                    mermaid_chains.append(self._mermaid_for_chain(cid))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"mermaid chain {cid}: {exc}")

        # --- ECharts 雷达图 ---
        echarts_radars: list[dict[str, Any]] = []
        for ann in annotations:
            ann_id = getattr(ann, "annotation_id", "")
            if ann_id:
                try:
                    echarts_radars.append(self.to_echarts_radar(ann_id))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"radar {ann_id}: {exc}")

        # --- ECharts 辩论时间线 (聚合) ---
        try:
            echarts_timeline = self._echarts_timeline_aggregated()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"timeline: {exc}")
            echarts_timeline = self._empty_echarts_option("辩论时间线生成失败", error=str(exc))

        # --- 统计汇总 ---
        statistics: dict[str, Any] = {}
        try:
            statistics["kpa"] = self._kpa.statistics()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"kpa stats: {exc}")
        try:
            statistics["debate_logs"] = self._dl.statistics()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"dl stats: {exc}")
        try:
            statistics["chains"] = {"total": len(chains), "items": chains[:10]}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"chain stats: {exc}")

        return {
            "schema": "ProvenanceRenderSchema/2.0",
            "target_id": target_id,
            "cytoscape": cytoscape,
            "mermaid_chains": mermaid_chains,
            "mermaid_annotations": mermaid_annotations,
            "echarts_radars": echarts_radars,
            "echarts_timeline": echarts_timeline,
            "statistics": statistics,
            "errors": errors,
            "generated_at": time.time(),
        }

    # ==========================================================
    # 辅助: 序列化导出 (JSON 字符串)
    # ==========================================================

    def export_all_json(self, target_id: str = "", indent: int = 2) -> str:
        """将 export_all 结果序列化为 JSON 字符串.

        便于直接写入文件或通过 HTTP/WebSocket 推送至 L7 前端。

        Args:
            target_id: 目标对象 ID
            indent: JSON 缩进 (0=紧凑)

        Returns:
            JSON 字符串
        """
        bundle = self.export_all(target_id=target_id)
        return json.dumps(bundle, ensure_ascii=False, indent=indent if indent > 0 else None)


def _truncate(text: str, max_len: int) -> str:
    """截断文本并添加省略号 (用于可视化标签)."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


__all__ = [
    "ProvenanceVisualizer",
    "SEVEN_DIMENSIONS",
    "FOUR_LAYER_DIMENSIONS",
    "TIER_COLORS",
    "AGENT_ROLE_SHAPES",
]
