"""L3 边补全 (P2 · LLM 生成部分) — DeepSeek 生成 + 规则校验 + 人工抽查后的教学关系边.

与 edge_enrich.py 的「规则生成」互补: 本模块边由 DeepSeek (deepseek-v4-flash)
生成候选, 经三重校验后固化:
  1. 规则校验 (端点 ∈ 48 新 ID / 关系 ∈ 教学关系枚举 / 无自环 / 无重复)
  2. 人工抽查 (剔除误用 subconcept_of 等语义错误的边)
  3. 确定性固化 (本表即最终结果, 不依赖运行时 LLM, 保证可复现)

关系语义 (见 l3/models.py RelationType):
  - applies_to      : 知识点/机理 → 应用场景 (第 6 章应用主线)
  - affects         : 跨域因果 (第 4 章合成工艺 → 第 2/3 章机理/性能)
  - prerequisite_of : 纵向学习前提

source_id="llm" 标记来源, confidence=0.85 (略低于规则边 1.0, 反映 LLM 溯源不确定性)。

端点用**新 ID** (章.节.序号) 作为规范编号; 入图时经 NEW_TO_OLD 映射到旧主键
(42 个重编号知识点) 或直接用新主键 (第 6 章 6 个新增知识点, 见 kp_graph_seed 的 P4 扩展)。
"""
from __future__ import annotations

import logging
from typing import Any

from dy3_polaris.l3.models import KnowledgeTriple

logger = logging.getLogger("dy3_polaris.l3.edge_llm")

# LLM 边来源标记 (区别于规则 "rule" 与手工默认 "")
_SRC = "llm"

# 固化置信度 (LLM 溯源, 略低于规则边)
_CONF = 0.85

#: 经校验 + 人工抽查后的 LLM 候选边 (新 ID 规范编号, 23 条)
LLM_EDGES: list[dict[str, str]] = [
    # ---- applies_to: 第 6 章「绿色健康照明应用」应用主线 ----
    {"src": "6.1.2", "rel": "applies_to", "dst": "6.3.2", "reason": "显色指数与色温知识应用于健康照明灯具的光学设计"},
    {"src": "6.2.1", "rel": "applies_to", "dst": "6.3.2", "reason": "蓝光危害机理指导健康照明灯具的蓝光防护设计"},
    {"src": "6.2.2", "rel": "applies_to", "dst": "6.3.2", "reason": "节律照明与光生物安全知识应用于健康照明灯具设计"},
    {"src": "3.3.2", "rel": "applies_to", "dst": "6.3.1", "reason": "色坐标与色纯度是单基质白光荧光粉设计的关键指标"},
    {"src": "2.1.1", "rel": "applies_to", "dst": "6.3.1", "reason": "Dy3+能级结构与4f-4f跃迁是设计单基质白光荧光粉的物理基础"},
    {"src": "2.2.2", "rel": "applies_to", "dst": "6.3.1", "reason": "荧光寿命与辐射跃迁速率影响单基质白光荧光粉的发光效率"},
    {"src": "2.3.3", "rel": "applies_to", "dst": "6.3.2", "reason": "热猝灭特性影响健康照明灯具的长期稳定性"},
    {"src": "3.3.1", "rel": "applies_to", "dst": "6.3.2", "reason": "量子效率是评估健康照明灯具发光性能的重要参数"},
    # ---- affects: 第 4 章合成工艺 → 第 2/3 章机理/性能 ----
    {"src": "4.2.1", "rel": "affects", "dst": "2.2.2", "reason": "焙烧温度影响结晶度，进而改变荧光寿命与辐射跃迁速率"},
    {"src": "4.2.1", "rel": "affects", "dst": "2.3.3", "reason": "焙烧温度影响晶格完整性，进而影响热猝灭行为"},
    {"src": "4.2.2", "rel": "affects", "dst": "2.1.1", "reason": "还原气氛控制Dy价态，直接影响4f-4f跃迁发光"},
    {"src": "4.2.3", "rel": "affects", "dst": "3.2.2", "reason": "助熔剂调控晶粒形貌，改变晶格对称性与局部配位环境"},
    {"src": "4.2.4", "rel": "affects", "dst": "2.3.1", "reason": "前驱体配比影响掺杂均匀性，进而影响能量传递效率"},
    {"src": "4.2.5", "rel": "affects", "dst": "2.3.3", "reason": "工艺缺陷引入非辐射复合中心，加剧热猝灭"},
    {"src": "4.2.6", "rel": "affects", "dst": "3.3.1", "reason": "规模放大与批次一致性影响最终产品的量子效率"},
    {"src": "4.1.1", "rel": "affects", "dst": "3.1.1", "reason": "固相烧结法影响氟化物基质的晶格形成与完整性"},
    {"src": "4.1.2", "rel": "affects", "dst": "3.1.2", "reason": "共沉淀法影响磷酸盐基质的晶相纯度与结晶度"},
    {"src": "4.1.3", "rel": "affects", "dst": "3.1.3", "reason": "溶胶-凝胶法影响铝酸盐基质的晶格结构与均匀性"},
    {"src": "4.1.4", "rel": "affects", "dst": "3.4.3", "reason": "水热法可制备纳米材料，影响表面效应与核壳结构"},
    # ---- prerequisite_of: 纵向学习前提 ----
    {"src": "6.2.1", "rel": "prerequisite_of", "dst": "6.2.2", "reason": "理解蓝光危害机理是学习节律照明与光生物安全的前提"},
    {"src": "6.1.1", "rel": "prerequisite_of", "dst": "6.1.2", "reason": "白光LED发光原理是理解显色指数与色温的基础"},
    {"src": "6.3.1", "rel": "prerequisite_of", "dst": "6.3.2", "reason": "单基质白光荧光粉设计是健康照明灯具封装的前提"},
    {"src": "3.3.1", "rel": "prerequisite_of", "dst": "5.2.3", "reason": "量子效率概念是绝对法测量的基础"},
]


def _kp_key(nid: str) -> str:
    """把新 ID 解析为图内知识点实体主键.

    42 个重编号知识点 → 旧主键 kp:{old} (图仍以旧主键为主);
    第 6 章 6 个新增知识点 → 新主键 kp:{new} (P4 扩展后入图)。
    """
    from dy3_polaris.l2.kp_catalog import NEW_TO_OLD

    old = NEW_TO_OLD.get(nid)
    return f"kp:{old}" if old else f"kp:{nid}"


def seed_llm_edges(store: Any) -> dict[str, int]:
    """幂等播种 LLM 生成的教学关系边, 返回 {llm_edges} 计数.

    端点实体不存在 (如第 6 章知识点尚未入图) 时跳过, 不报错。
    """
    from dy3_polaris.l2.kp_catalog import NEW_TO_OLD  # noqa: F401

    added = 0
    skipped = 0
    for e in LLM_EDGES:
        src_key = _kp_key(e["src"])
        dst_key = _kp_key(e["dst"])
        if store.get_entity(src_key) is None or store.get_entity(dst_key) is None:
            skipped += 1
            continue
        tid = f"llm:{e['src']}:{e['rel']}:{e['dst']}"
        if store.get_triple(tid) is None:
            store.add_triple(KnowledgeTriple(
                triple_id=tid,
                subject_id=src_key,
                predicate=e["rel"],
                object_id=dst_key,
                confidence=_CONF,
                source_id=_SRC,
            ))
            added += 1
    logger.info("LLM 边播种: added=%d skipped=%d", added, skipped)
    return {"llm_edges": added, "skipped": skipped}


__all__ = ["LLM_EDGES", "seed_llm_edges"]
