"""L3 领域知识层 — 知识点关系图播种 (镝-绿色健康照明垂直领域).

把 L2 kp_catalog 的 42 个知识点 + 教学关系边 (前提/类比/因果/表征)
+ L3 textbook_fallback 的 41 条权威兜底事实, 统一入图到 KnowledgeStore,
使 GraphReasoner / GraphRAGRetriever 能在「知识点关系图」上跑多跳拓展。

数据形态:
  - 知识点实体    : entity_id="kp:A-01" ~ "kp:D-08", entity_type=CONCEPT
  - 事实实体      : entity_id="fact:tf-dy-01" ~ "fact:tf-dy-41", entity_type=CONCEPT
  - 知识点间边    : kp:src -{教学关系}-> kp:dst (来自 kp_catalog._KP_EDGES)
  - 事实→知识点边 : fact:tf-dy-XX -supports-> kp:YY (按 fact.kp_ids 关联)

幂等: entity_id / triple_id 确定性命名, 重复播种自动跳过。
"""
from __future__ import annotations

import logging
from typing import Any

from dy3_polaris.l3.models import EntityType, KnowledgeEntity, KnowledgeTriple

logger = logging.getLogger("dy3_polaris.l3.kp_graph_seed")

# 领域标识 (区别于机器抽取图的 "general")
_DOMAIN = "dy-lighting"


def _kp_entity(kp_id: str, name: str, level: str, domain_label: str) -> KnowledgeEntity:
    """构造知识点实体 (旧 ID 主键, 附新 ID 迁移桥接属性 new_id)."""
    from dy3_polaris.l2.kp_catalog import to_new_id

    return KnowledgeEntity(
        entity_id=f"kp:{kp_id}",
        entity_type=EntityType.CONCEPT,
        name=name,
        description=name,
        domain=_DOMAIN,
        tags=["kp", "knowledge_point", level.lower()],
        aliases=[kp_id, to_new_id(kp_id)],
        properties={
            "kp_id": kp_id,
            "new_id": to_new_id(kp_id),
            "level": level,
            "domain_label": domain_label,
        },
    )


def seed_kp_graph(
    store: Any,
    *,
    include_placeholder_facts: bool = False,
) -> dict[str, int]:
    """幂等播种知识点关系图, 返回 {kps, facts, edges, fact_kp_edges} 计数.

    Args:
        store: KnowledgeStore 实例

    Returns:
        本次实际新增的各项计数 (已存在则跳过不计入)。
    """
    from dy3_polaris.l2.kp_catalog import (
        DOMAIN_LABELS,
        KP_EDGES,
        KP_LEVELS,
        KP_NAMES,
        KP_TO_DOMAIN,
    )
    from dy3_polaris.l3.textbook_fallback import CANONICAL_FACTS

    kps = facts = edges = fact_edges = 0

    # 1. 知识点实体 (42 个)
    for kp_id, name in KP_NAMES.items():
        eid = f"kp:{kp_id}"
        if store.get_entity(eid) is None:
            store.add_entity(
                _kp_entity(
                    kp_id,
                    name,
                    KP_LEVELS.get(kp_id, "L1"),
                    DOMAIN_LABELS.get(KP_TO_DOMAIN.get(kp_id, ""), ""),
                ),
                track_version=False,
            )
            kps += 1

    # 2. 知识点间教学关系边 (44 条)
    for e in KP_EDGES:
        rel = e["rel"]
        tid = f"kpe:{e['src']}:{rel}:{e['dst']}"
        if store.get_triple(tid) is None:
            store.add_triple(
                KnowledgeTriple(
                    triple_id=tid,
                    subject_id=f"kp:{e['src']}",
                    predicate=rel,
                    object_id=f"kp:{e['dst']}",
                    confidence=1.0,
                )
            )
            edges += 1

    # 3. 占位事实不是可溯源科研证据，产品模式默认不入图。
    # 仅显式开发/测试场景可启用，并保留原有标签便于识别。
    source_facts = CANONICAL_FACTS if include_placeholder_facts else []
    for fact in source_facts:
        fid = f"fact:{fact['id']}"
        if store.get_entity(fid) is None:
            store.add_entity(
                KnowledgeEntity(
                    entity_id=fid,
                    entity_type=EntityType.CONCEPT,
                    name=fact["id"],
                    description=fact["content"],
                    domain=_DOMAIN,
                    tags=["fact", "textbook_fallback"],
                    aliases=[fact["id"]],
                    properties={
                        "chapter": fact.get("chapter", ""),
                        "kp_ids": fact.get("kp_ids", []),
                        "ions": fact.get("ions", []),
                    },
                ),
                track_version=False,
            )
            facts += 1

        for kp_id in fact.get("kp_ids", []):
            kp_entity = f"kp:{kp_id}"
            if store.get_entity(kp_entity) is None:
                continue
            tid = f"fe:{fact['id']}:{kp_id}"
            if store.get_triple(tid) is None:
                store.add_triple(
                    KnowledgeTriple(
                        triple_id=tid,
                        subject_id=fid,
                        predicate="supports",
                        object_id=kp_entity,
                        confidence=1.0,
                    )
                )
                fact_edges += 1

    logger.info(
        "KP 关系图播种完成: kps=%d facts=%d edges=%d fact_kp_edges=%d",
        kps, facts, edges, fact_edges,
    )
    return {
        "kps": kps,
        "facts": facts,
        "edges": edges,
        "fact_kp_edges": fact_edges,
    }


def seed_role_kp(store: Any) -> dict[str, int]:
    """幂等播种职业角色 + 角色-知识点关联, 返回 {roles, role_kp_edges} 计数.

    角色实体 entity_id="role:<id>", 角色-知识点边 role -depends_on-> kp,
    边 confidence 承载权重 (1.0 核心 / 0.6 相关)。支持按角色的个性化学习路径推荐。
    """
    from dy3_polaris.l2.kp_roles import ROLES, role_kps

    roles = edges = 0
    for rid, meta in ROLES.items():
        eid = f"role:{rid}"
        if store.get_entity(eid) is None:
            store.add_entity(
                KnowledgeEntity(
                    entity_id=eid,
                    entity_type=EntityType.CONCEPT,
                    name=meta["name"],
                    description=meta["desc"],
                    domain=_DOMAIN,
                    tags=["role", "occupation"],
                    aliases=[rid],
                    properties={"role_id": rid},
                ),
                track_version=False,
            )
            roles += 1
        for kp_id, weight in role_kps(rid).items():
            tid = f"rk:{rid}:{kp_id}"
            if store.get_triple(tid) is None:
                store.add_triple(
                    KnowledgeTriple(
                        triple_id=tid,
                        subject_id=eid,
                        predicate="depends_on",
                        object_id=f"kp:{kp_id}",
                        confidence=float(weight),
                    )
                )
                edges += 1
    logger.info("角色-知识点播种: roles=%d edges=%d", roles, edges)
    return {"roles": roles, "role_kp_edges": edges}


def seed_ch6_kps(store: Any) -> dict[str, int]:
    """幂等播种第 6 章「绿色健康照明应用」6 个新增知识点 (P4 扩展).

    主键 kp:{章.节.序号} (无旧 ID); 用 part_of 挂到 topic:{节}, 并把
    kp_catalog._KP_EDGES_NEW_CH6 的 6 条应用主线边入图。
    旧 42 知识点仍以旧主键 kp:{A-01~D-08} 为主 (迁移桥接见 _kp_entity.new_id)。

    返回 {kps, part_of_edges, edges} 计数。
    """
    from dy3_polaris.l2.kp_catalog import (
        _KP_EDGES_NEW_CH6,
        NEW_KP_NAMES,
        NEW_KP_TO_SECTION,
    )

    kps = part_of = edges = 0
    for nid, name in NEW_KP_NAMES.items():
        # 只播种第 6 章新增知识点 (无旧 ID 的那些)
        if nid.split(".")[0] != "6":
            continue
        eid = f"kp:{nid}"
        if store.get_entity(eid) is None:
            store.add_entity(
                KnowledgeEntity(
                    entity_id=eid,
                    entity_type=EntityType.CONCEPT,
                    name=name,
                    description=name,
                    domain=_DOMAIN,
                    tags=["kp", "knowledge_point", "l3", "ch6"],
                    aliases=[nid],
                    properties={
                        "kp_id": nid,
                        "new_id": nid,
                        "level": "L3",
                        "chapter": "6",
                    },
                ),
                track_version=False,
            )
            kps += 1
        # part_of: kp:{nid} -> topic:{节}
        sec = NEW_KP_TO_SECTION.get(nid, "")
        if sec:
            tid = f"po:{nid}:{sec}"
            if store.get_triple(tid) is None and store.get_entity(f"topic:{sec}") is not None:
                store.add_triple(KnowledgeTriple(
                    triple_id=tid,
                    subject_id=eid,
                    predicate="part_of",
                    object_id=f"topic:{sec}",
                    confidence=1.0,
                ))
                part_of += 1

    # 第 6 章应用主线边 (新旧 ID 混合端点, 需解析主键)
    for e in _KP_EDGES_NEW_CH6:
        src_key = _kp_key(e["src"])
        dst_key = _kp_key(e["dst"])
        if store.get_entity(src_key) is None or store.get_entity(dst_key) is None:
            continue
        tid = f"kpe:{e['src']}:{e['rel']}:{e['dst']}"
        if store.get_triple(tid) is None:
            store.add_triple(KnowledgeTriple(
                triple_id=tid,
                subject_id=src_key,
                predicate=e["rel"],
                object_id=dst_key,
                confidence=1.0,
            ))
            edges += 1

    logger.info("第 6 章知识点播种: kps=%d part_of=%d edges=%d", kps, part_of, edges)
    return {"kps": kps, "part_of_edges": part_of, "edges": edges}


def _kp_key(nid: str) -> str:
    """把新 ID 解析为图内知识点实体主键 (旧主键优先, 第 6 章用新主键)."""
    from dy3_polaris.l2.kp_catalog import NEW_TO_OLD

    old = NEW_TO_OLD.get(nid)
    return f"kp:{old}" if old else f"kp:{nid}"


__all__ = ["seed_kp_graph", "seed_role_kp", "seed_ch6_kps"]
