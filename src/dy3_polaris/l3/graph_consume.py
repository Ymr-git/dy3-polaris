"""L3 图消费层 (P3) — 让知识图谱真正驱动问答.

把多类型图 (知识点/实体/主题/事实/角色) 的推理能力暴露给 L5 问答与前端:

- ``recall()``        多跳召回 — 从查询命中的种子节点双向 BFS, 收集相关实体/事实证据,
                      并附溯源路径 (path/relation/hop), 补足静态 KP 兜底层未覆盖的
                      材料/离子/能级/方法/参数多类型实体。
- ``learning_path()`` 学习路径 — Dijkstra 加权最短路径 + 可读解释 (GraphReasoner)。
- ``analogy()``       类比推理 — 关系模式迁移 (GraphReasoner)。
- ``provenance()``    溯源 — 实体入/出边解释 (explain_path)。

产出的证据项与 L5 ``compose_items`` 兼容 (chunk_id/document_id/section/content/metadata),
``source_type`` 用 ``kg_graph`` / ``kg_concept`` 区分 (前者为可引用的权威事实, 后者为拓展上下文)。
本模块仅依赖 L3 (store / graph_reasoner), 不反向依赖 L5。
"""
from __future__ import annotations

import logging
from typing import Any

from dy3_polaris.l3.graph_reasoner import GraphReasoner

logger = logging.getLogger("dy3_polaris.l3.graph_consume")

#: 图证据来源标注 (与 l5 textbook_fallback / kp_expand 并列)
_GRAPH_SOURCE = "知识图谱 · 镝基绿色健康照明发光材料"

#: 值得作为「概念上下文」召回的实体类型 (事实/知识点单独处理)
_CONCEPT_TYPES = frozenset({
    "material", "ion", "energy_level", "method", "parameter",
    "chemical_compound", "knowledge_point", "topic", "role",
})

#: 稀土离子 → 图内离子实体主键 (查询侧元素符号归一后命中)
_ION_ENTITY = {
    "dy": "ion:Dy3+",
    "yb": "ion:Yb3+",
    "y": "ion:Y3+",
}


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def _matchable(alias: str) -> bool:
    """判断一个别名是否值得做子串匹配 (避免过短拉丁串误命中, 如 "pl" ∈ "sample").

    - 拉丁/数字串: 长度 >= 3 才参与匹配 (XRD/SEM/CCT/CRI/Dy3+ ...)
    - 含中文/非 ASCII: 长度 >= 2 即参与 (寿命/色温/显色 ...)
    """
    a = str(alias or "").strip()
    if len(a) < 2:
        return False
    if a.isascii():
        return len(a) >= 3
    return True


def _extract_query_ions(query: str) -> list[str]:
    """从查询里提取裸元素符号 (与 l5 _extract_ions 同语义的最小实现, 避免跨层依赖)."""
    q = _norm(query)
    ions: list[str] = []
    # 只认带价态的写法 (dy3+/dy³⁺) 或中文名, 避免裸 "y" 误命中英文词
    if "dy3" in q or "dy³" in q or "镝" in query:
        ions.append("dy")
    if "yb3" in q or "yb³" in q or "镱" in query:
        ions.append("yb")
    return ions


def _resolve_entity_key(raw: str, store: Any) -> str | None:
    """把「新旧 ID / 别名」解析为图内实体物理主键 (P4 迁移桥接).

    42 个重编号知识点图内以旧主键 kp:{A-01~D-08} 为主 (附 new_id 属性 + 新 ID 别名),
    第 6 章新增知识点以新主键 kp:{6.x.x} 为主。本函数统一接受:
      - 物理主键 (kp:A-05 / kp:6.1.1 / ion:Dy3+ / mth:XRD ...)
      - 新 ID 主键 (kp:2.1.1) → 经别名反查旧主键 kp:A-05
      - 裸 ID / 名称 / 别名 (2.1.1 / A-05 / Dy3+ ...)
    返回物理主键, 无匹配返回 None。
    """
    raw = str(raw or "").strip()
    if not raw:
        return None
    if store.get_entity(raw) is not None:
        return raw
    candidates = [raw]
    if raw.startswith("kp:"):
        candidates.append(raw[3:])
    if ":" not in raw:
        candidates.append(f"kp:{raw}")
    for cand in candidates:
        if store.get_entity(cand) is not None:
            return cand
    for cand in candidates:
        hits = store.entity_store.find_by_name(cand)
        if hits:
            return hits[0].entity_id
    return None


def resolve_seeds(query: str, store: Any, *, extra_seed_ids: list[str] | None = None) -> list[str]:
    """把查询解析为图内种子实体 ID (名称/别名子串匹配 + 离子映射 + 显式种子).

    子串匹配规则: 实体名或别名的「规范化形式」出现在查询中, 或查询包含该名/别名;
    拉丁别名要求 >=3 字符避免过短误命中。
    """
    q = _norm(query)
    seeds: list[str] = []
    seen: set[str] = set()

    def add(eid: str | None) -> None:
        if eid and eid not in seen and store.get_entity(eid) is not None:
            seen.add(eid)
            seeds.append(eid)

    for eid in extra_seed_ids or []:
        add(eid)

    # 离子实体 (Dy3+/Yb3+)
    for ion in _extract_query_ions(query):
        add(_ION_ENTITY.get(ion))

    # 名称/别名子串匹配
    for e in store.entity_store.list_entities(limit=100_000):
        for n in [e.name] + list(e.aliases):
            nn = _norm(n)
            if not nn or not _matchable(nn):
                continue
            if nn in q or (len(nn) >= 2 and nn in q):
                add(e.entity_id)
                break
    return seeds


def _path_names(path_ids: list[str] | None, store: Any) -> list[str]:
    """把实体 ID 路径转为名称路径 (溯源展示)."""
    if not path_ids:
        return []
    names: list[str] = []
    for eid in path_ids:
        e = store.get_entity(eid)
        names.append(e.name if e is not None else eid)
    return names


def _section_of(e: Any, store: Any) -> str:
    """实体所属章节/节 (经 part_of -> topic 上溯, 最佳努力)."""
    for t in store.triple_store.get_outgoing(e.entity_id, predicate="part_of"):
        if t.object_id and t.object_id.startswith("topic:"):
            parent = store.get_entity(t.object_id)
            if parent is not None:
                return parent.name or parent.entity_id
    return ""


def _evidence_item(e: Any, store: Any, *, path_ids: list[str], relation: str,
                   hop: int, source_type: str, content: str | None = None) -> dict[str, Any]:
    """构造与 L5 compose_items 兼容的证据项."""
    return {
        "chunk_id": f"kg:{e.entity_id}",
        "document_id": "kg-graph",
        "section": _section_of(e, store),
        "content": content if content is not None else (e.description or e.name),
        "metadata": {
            "entity": e.name,
            "entity_id": e.entity_id,
            "entity_type": e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type),
            "source_type": source_type,
            "source": _GRAPH_SOURCE,
            "path": _path_names(path_ids, store),
            "relation": relation,
            "hop": hop,
        },
    }


def _first_relation(src: str, dst: str, store: Any) -> str:
    """src -> dst 的首跳关系谓词 (沿最短路径方向)."""
    if src == dst:
        return "self"
    path = store.triple_store.get_path(src, dst, max_depth=2)
    if not path or len(path) < 2:
        return ""
    for t in store.triple_store.get_outgoing(path[0]):
        if t.object_id == path[1]:
            return t.predicate
    for t in store.triple_store.get_incoming(path[0]):
        if t.subject_id == path[1]:
            return t.predicate
    return ""


def recall(query: str, store: Any, *, max_hop: int = 2, max_facts: int = 8,
           max_concepts: int = 8, extra_seed_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """多跳召回 — 从种子节点双向 BFS, 收集事实 + 多类型概念实体证据.

    Returns:
        evidence 列表 (facts 优先, 概念实体次之, 均含溯源 path/relation/hop)。
    """
    seeds = resolve_seeds(query, store, extra_seed_ids=extra_seed_ids)
    if not seeds:
        return []

    facts: list[dict[str, Any]] = []
    concepts: list[dict[str, Any]] = []
    seen_fact: set[str] = set()
    seen_concept: set[str] = set()

    for seed in seeds:
        entity_ids, _triples = store.triple_store.traverse_bfs(
            seed, max_depth=max_hop, direction="both", max_entities=120,
        )
        for eid in entity_ids:
            if eid == seed:
                continue
            e = store.get_entity(eid)
            if e is None:
                continue
            tags = set(e.tags or [])
            path = store.triple_store.get_path(seed, eid, max_depth=max_hop) or []
            rel = _first_relation(seed, eid, store)
            hop = max(len(path) - 1, 0)
            if "fact" in tags:
                if eid in seen_fact or len(facts) >= max_facts:
                    continue
                seen_fact.add(eid)
                facts.append(_evidence_item(
                    e, store, path_ids=path, relation=rel, hop=hop,
                    source_type="kg_graph", content=e.description or e.name,
                ))
            elif e.entity_type.value in _CONCEPT_TYPES:
                if eid in seen_concept or len(concepts) >= max_concepts:
                    continue
                seen_concept.add(eid)
                concepts.append(_evidence_item(
                    e, store, path_ids=path, relation=rel, hop=hop,
                    source_type="kg_concept",
                ))

    # facts 优先 (权威可引用), 概念实体次之 (拓展上下文)
    return facts + concepts


def learning_path(store: Any, start_id: str, goal_id: str, *, max_depth: int = 10) -> dict[str, Any] | None:
    """学习路径 — Dijkstra 加权最短路径 + 可读解释 (GraphReasoner.find_shortest_path)."""
    start = _resolve_entity_key(start_id, store)
    goal = _resolve_entity_key(goal_id, store)
    if start is None or goal is None:
        return None
    reasoner = GraphReasoner(store)
    pr = reasoner.find_shortest_path(start, goal, max_depth=max_depth)
    if pr is None:
        return None
    return {
        "start_id": pr.start_id,
        "goal_id": pr.end_id,
        "path": pr.path,
        "path_names": _path_names(pr.path, store),
        "edges": pr.edges,
        "hop_count": pr.hop_count,
        "total_weight": pr.total_weight,
        "explanation": pr.explanation,
    }


def analogy(store: Any, source_pair: tuple[str, str], target_entity: str) -> list[dict[str, Any]]:
    """类比推理 — 关系模式迁移 (GraphReasoner.analogical_reasoning)."""
    reasoner = GraphReasoner(store)
    src_a = _resolve_entity_key(source_pair[0], store)
    src_b = _resolve_entity_key(source_pair[1], store)
    target = _resolve_entity_key(target_entity, store)
    if src_a is None or src_b is None or target is None:
        return []
    return reasoner.analogical_reasoning((src_a, src_b), target)


def provenance(store: Any, entity_id: str) -> dict[str, Any]:
    """溯源 — 实体入/出边 (predicate + 邻接实体名 + 置信度)."""
    eid = _resolve_entity_key(entity_id, store)
    if eid is None:
        return {"entity_id": entity_id, "exists": False, "outgoing": [], "incoming": []}
    e = store.get_entity(eid)
    if e is None:
        return {"entity_id": entity_id, "exists": False, "outgoing": [], "incoming": []}
    out: list[dict[str, Any]] = []
    for t in store.triple_store.get_outgoing(eid):
        if not t.object_id:
            continue
        n = store.get_entity(t.object_id)
        out.append({"predicate": t.predicate, "target": t.object_id,
                    "target_name": n.name if n else t.object_id,
                    "confidence": float(t.confidence), "source_id": getattr(t, "source_id", "")})
    inc: list[dict[str, Any]] = []
    for t in store.triple_store.get_incoming(eid):
        n = store.get_entity(t.subject_id)
        inc.append({"predicate": t.predicate, "source": t.subject_id,
                    "source_name": n.name if n else t.subject_id,
                    "confidence": float(t.confidence), "source_id": getattr(t, "source_id", "")})
    return {"entity_id": eid, "name": e.name, "exists": True,
            "outgoing": out, "incoming": inc}


__all__ = ["recall", "resolve_seeds", "_resolve_entity_key", "learning_path", "analogy", "provenance"]
