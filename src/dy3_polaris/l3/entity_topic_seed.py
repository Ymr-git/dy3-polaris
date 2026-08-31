"""L3 实体层 + Topic 层播种 (P1b).

把「多类型节点 + 纵向层级」落到内存图:
  1. Topic 层: 6 章 + 17 节 (topic:<code>) 作为主题节点,
     用 part_of 边把「节 → 章」和「知识点 → 节」串成三层树 (纵向层级)。
  2. 实体层: 材料/离子/能级/方法/参数 (28 个实体), 从 41 条 canonical facts
     与 _KG_NODES 落点抽取, 用 mentions 边 (知识点/事实 → 实体) 入图。

数据形态:
  - 主题实体    : entity_id="topic:1" ~ "topic:6", "topic:1.1" ~ "topic:6.3", type=TOPIC
  - 材料/离子/能级/方法/参数实体 : entity_id="mat:*/ion:*/el:*/mth:*/par:*"
  - part_of 边  : topic:{节} -part_of-> topic:{章};  kp:{旧ID} -part_of-> topic:{节}
  - mentions 边 : kp:{旧ID} -mentions-> {实体};      fact:{id} -mentions-> {实体}

说明: 知识点实体仍以旧 ID (kp:A-01~D-08) 为主键 (P4 再重编号为章.节.序号);
     第 6 章 (绿色健康照明应用) 的 6 个新增知识点在 P4 重编号时一起入图, 此处只建章/节骨架。

幂等: entity_id / triple_id 确定性命名, 重复播种自动跳过。
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from dy3_polaris.l3.models import EntityType, KnowledgeEntity, KnowledgeTriple

logger = logging.getLogger("dy3_polaris.l3.entity_topic_seed")

_DOMAIN = "dy-lighting"


# ============================================================
# 实体目录 (P1b) — 从 41 条 canonical facts 与 _KG_NODES 落点抽取
# ============================================================
# 每个实体: id (entity_id), type (EntityType), name, kp_ids (被哪些旧 ID 知识点提及),
#           aliases (检索别名)
ENTITIES: list[dict[str, Any]] = [
    # ---- 材料/基质 (MATERIAL) ----
    {"id": "mat:NaGdF4", "type": EntityType.MATERIAL, "name": "NaGdF4 (氟化物基质)",
     "kp_ids": ["B-01"], "aliases": ["NaGdF4", "氟化物", "基质"]},
    {"id": "mat:YPO4", "type": EntityType.MATERIAL, "name": "YPO4 (磷酸盐基质)",
     "kp_ids": ["B-02"], "aliases": ["YPO4", "磷酸盐", "锆石"]},
    {"id": "mat:BAM", "type": EntityType.MATERIAL, "name": "BaMgAl10O17 (BAM 铝酸盐基质)",
     "kp_ids": ["B-03"], "aliases": ["BAM", "BaMgAl10O17", "铝酸盐"]},
    # ---- 离子 (ION) ----
    {"id": "ion:Dy3+", "type": EntityType.ION, "name": "Dy³⁺ (激活剂)",
     "kp_ids": ["A-01", "A-05"], "aliases": ["Dy3+", "Dy³⁺", "镝离子", "dy3+"]},
    {"id": "ion:Yb3+", "type": EntityType.ION, "name": "Yb³⁺ (敏化剂)",
     "kp_ids": ["B-09"], "aliases": ["Yb3+", "Yb³⁺", "敏化剂"]},
    {"id": "ion:Y3+", "type": EntityType.ION, "name": "Y³⁺ (基质阳离子)",
     "kp_ids": ["B-02", "B-04"], "aliases": ["Y3+", "Y³⁺", "取代格位"]},
    # ---- 能级/跃迁 (ENERGY_LEVEL) ----
    {"id": "el:4F9_2", "type": EntityType.ENERGY_LEVEL, "name": "4F9/2 发射能级",
     "kp_ids": ["A-05"], "aliases": ["4F9/2", "F9_2", "发射能级"]},
    {"id": "el:6H15_2", "type": EntityType.ENERGY_LEVEL, "name": "6H15/2 基态",
     "kp_ids": ["A-03", "A-05"], "aliases": ["6H15/2", "H15_2", "基态"]},
    {"id": "el:6H13_2", "type": EntityType.ENERGY_LEVEL, "name": "6H13/2 激发态",
     "kp_ids": ["A-03", "A-05"], "aliases": ["6H13/2", "H13_2", "激发态"]},
    {"id": "el:6H11_2", "type": EntityType.ENERGY_LEVEL, "name": "6H11/2 激发态",
     "kp_ids": ["A-03"], "aliases": ["6H11/2", "H11_2"]},
    {"id": "el:4f-4f", "type": EntityType.ENERGY_LEVEL, "name": "4f-4f 跃迁通道",
     "kp_ids": ["A-04", "A-05"], "aliases": ["4f-4f", "f-f", "禁戒跃迁"]},
    {"id": "el:4f-5d", "type": EntityType.ENERGY_LEVEL, "name": "4f-5d 跃迁通道",
     "kp_ids": ["A-08"], "aliases": ["4f-5d", "5d", "允许跃迁"]},
    # ---- 表征方法 (METHOD) ----
    {"id": "mth:XRD", "type": EntityType.METHOD, "name": "XRD 物相分析",
     "kp_ids": ["D-01"], "aliases": ["XRD", "xrd", "衍射", "PDF 卡片"]},
    {"id": "mth:SEM", "type": EntityType.METHOD, "name": "SEM 形貌表征",
     "kp_ids": ["D-02"], "aliases": ["SEM", "扫描电镜", "形貌"]},
    {"id": "mth:TEM", "type": EntityType.METHOD, "name": "TEM 形貌/晶格表征",
     "kp_ids": ["D-02"], "aliases": ["TEM", "透射电镜", "HRTEM"]},
    {"id": "mth:ICP-OES", "type": EntityType.METHOD, "name": "ICP-OES 掺杂浓度定量",
     "kp_ids": ["D-07"], "aliases": ["ICP-OES", "ICP", "电感耦合等离子体"]},
    {"id": "mth:integrating-sphere", "type": EntityType.METHOD, "name": "积分球绝对法 (量子效率)",
     "kp_ids": ["D-05"], "aliases": ["积分球", "绝对法"]},
    {"id": "mth:CIE-1931", "type": EntityType.METHOD, "name": "CIE 1931 色度计算",
     "kp_ids": ["D-08"], "aliases": ["CIE", "CIE 1931", "三刺激值"]},
    {"id": "mth:PL", "type": EntityType.METHOD, "name": "PL/PLE 荧光光谱",
     "kp_ids": ["D-03"], "aliases": ["PL", "PLE", "发射光谱", "激发光谱", "荧光光谱"]},
    # ---- 性能参数 (PARAMETER) ----
    {"id": "par:QE", "type": EntityType.PARAMETER, "name": "量子效率 (IQE/EQE)",
     "kp_ids": ["B-06", "D-05"], "aliases": ["量子效率", "IQE", "EQE", "内量子效率"]},
    {"id": "par:chromaticity", "type": EntityType.PARAMETER, "name": "色坐标 (CIE x,y)",
     "kp_ids": ["B-07", "D-08"], "aliases": ["色坐标", "色纯度", "cie"]},
    {"id": "par:CCT", "type": EntityType.PARAMETER, "name": "色温 CCT",
     "kp_ids": ["B-07"], "aliases": ["CCT", "色温", "相关色温"]},
    {"id": "par:CRI", "type": EntityType.PARAMETER, "name": "显色指数 CRI",
     "kp_ids": ["B-07"], "aliases": ["CRI", "显色指数", "显色"]},
    {"id": "par:lifetime", "type": EntityType.PARAMETER, "name": "荧光寿命 τ",
     "kp_ids": ["A-10", "D-04"], "aliases": ["荧光寿命", "寿命", "衰减"]},
    {"id": "par:T50", "type": EntityType.PARAMETER, "name": "热稳定性 T50",
     "kp_ids": ["A-13", "D-06"], "aliases": ["T50", "热稳定性", "热猝灭"]},
    {"id": "par:YB-ratio", "type": EntityType.PARAMETER, "name": "黄蓝比 Y/B",
     "kp_ids": ["B-05", "B-07"], "aliases": ["Y/B", "黄蓝比", "YB_ratio"]},
    {"id": "par:JO-params", "type": EntityType.PARAMETER, "name": "J-O 强度参数 Ω2/Ω4/Ω6",
     "kp_ids": ["A-07"], "aliases": ["Ω2", "Ω4", "Ω6", "Judd-Ofelt"]},
    {"id": "par:Dq", "type": EntityType.PARAMETER, "name": "晶体场参数 Dq",
     "kp_ids": ["A-06"], "aliases": ["Dq", "晶体场参数", "晶场"]},
]


def _entity_entity(e: dict[str, Any]) -> KnowledgeEntity:
    """构造实体节点."""
    return KnowledgeEntity(
        entity_id=e["id"],
        entity_type=e["type"],
        name=e["name"],
        description=e["name"],
        domain=_DOMAIN,
        tags=[e["type"].value, "entity"],
        aliases=list(e.get("aliases", [])),
        properties={
            "entity_id": e["id"],
            "entity_type": e["type"].value,
            "kp_ids": list(e.get("kp_ids", [])),
        },
    )


def _topic_entity(code: str, name: str, kind: str) -> KnowledgeEntity:
    """构造主题节点 (章 kind="chapter" / 节 kind="section")."""
    return KnowledgeEntity(
        entity_id=f"topic:{code}",
        entity_type=EntityType.TOPIC,
        name=name,
        description=name,
        domain=_DOMAIN,
        tags=["topic", kind, "taxonomy"],
        aliases=[code],
        properties={"topic_code": code, "kind": kind},
    )


def seed_topics(store: Any) -> dict[str, int]:
    """幂等播种 Topic 层 (6 章 + 17 节) + part_of 纵向边.

    返回 {topics, part_of_edges}。part_of 方向 = 部分 → 整体:
      topic:{节} -part_of-> topic:{章}
      kp:{旧ID} -part_of-> topic:{节}
    """
    from dy3_polaris.l2.kp_catalog import (
        CHAPTERS,
        CHAPTER_LABELS,
        NEW_KP_TO_SECTION,
        SECTION_LABELS,
        to_new_id,
    )

    topics = part_of = 0

    # 1. 主题节点 (章 + 节)
    for chap in CHAPTERS:
        ccode = chap["code"]
        cname = chap["name"]
        if store.get_entity(f"topic:{ccode}") is None:
            store.add_entity(_topic_entity(ccode, cname, "chapter"), track_version=False)
            topics += 1
        for sec in chap["sections"]:
            scode = sec["code"]
            sname = sec["name"]
            if store.get_entity(f"topic:{scode}") is None:
                store.add_entity(_topic_entity(scode, sname, "section"), track_version=False)
                topics += 1

    # 2. part_of: 节 → 章
    for scode, sname in SECTION_LABELS.items():
        ccode = scode.split(".")[0]
        if ccode not in CHAPTER_LABELS:
            continue
        tid = f"po:{scode}:{ccode}"
        if store.get_triple(tid) is None:
            store.add_triple(KnowledgeTriple(
                triple_id=tid,
                subject_id=f"topic:{scode}",
                predicate="part_of",
                object_id=f"topic:{ccode}",
                confidence=1.0,
            ))
            part_of += 1

    # 3. part_of: 知识点(旧ID) → 节 (遍历旧知识点, 经 to_new_id 反查节)
    from dy3_polaris.l2.kp_catalog import ALL_KP_IDS
    for old_id in ALL_KP_IDS:
        sec = NEW_KP_TO_SECTION.get(to_new_id(old_id), "")
        if not sec or store.get_entity(f"kp:{old_id}") is None:
            continue
        tid = f"po:{old_id}:{sec}"
        if store.get_triple(tid) is None:
            store.add_triple(KnowledgeTriple(
                triple_id=tid,
                subject_id=f"kp:{old_id}",
                predicate="part_of",
                object_id=f"topic:{sec}",
                confidence=1.0,
            ))
            part_of += 1

    logger.info("Topic 层播种: topics=%d part_of_edges=%d", topics, part_of)
    return {"topics": topics, "part_of_edges": part_of}


def seed_entities(store: Any) -> dict[str, int]:
    """幂等播种实体层 (28 个) + mentions 边 (知识点/事实 → 实体).

    返回 {entities, kp_mentions, fact_mentions}。mentions 方向 = 提及方 → 被提及实体:
      kp:{旧ID}  -mentions-> {实体}
      fact:{id}  -mentions-> {实体}  (经事实 kp_ids 桥接推导)
    """
    from dy3_polaris.l3.textbook_fallback import CANONICAL_FACTS

    entities = kp_mentions = fact_mentions = 0

    # 1. 实体节点
    kp_to_entities: dict[str, list[str]] = defaultdict(list)
    for e in ENTITIES:
        if store.get_entity(e["id"]) is None:
            store.add_entity(_entity_entity(e), track_version=False)
            entities += 1
        for kp in e.get("kp_ids", []):
            kp_to_entities[kp].append(e["id"])

    # 2. 知识点 → 实体 mentions
    for kp, eids in kp_to_entities.items():
        if store.get_entity(f"kp:{kp}") is None:
            continue
        for eid in eids:
            tid = f"me:{kp}:{eid}"
            if store.get_triple(tid) is None:
                store.add_triple(KnowledgeTriple(
                    triple_id=tid,
                    subject_id=f"kp:{kp}",
                    predicate="mentions",
                    object_id=eid,
                    confidence=1.0,
                ))
                kp_mentions += 1

    # 3. 事实 → 实体 mentions (经事实 kp_ids 桥接)
    for fact in CANONICAL_FACTS:
        fid = f"fact:{fact['id']}"
        if store.get_entity(fid) is None:
            continue
        for kp in fact.get("kp_ids", []):
            for eid in kp_to_entities.get(kp, []):
                tid = f"me:{fact['id']}:{eid}"
                if store.get_triple(tid) is None:
                    store.add_triple(KnowledgeTriple(
                        triple_id=tid,
                        subject_id=fid,
                        predicate="mentions",
                        object_id=eid,
                        confidence=1.0,
                    ))
                    fact_mentions += 1

    logger.info(
        "实体层播种: entities=%d kp_mentions=%d fact_mentions=%d",
        entities, kp_mentions, fact_mentions,
    )
    return {"entities": entities, "kp_mentions": kp_mentions, "fact_mentions": fact_mentions}


def seed_all(store: Any) -> dict[str, Any]:
    """P1b 总入口: Topic 层 + 实体层."""
    t = seed_topics(store)
    e = seed_entities(store)
    return {"topics": t, "entities": e}


__all__ = ["ENTITIES", "seed_topics", "seed_entities", "seed_all"]
