"""L3 领域知识层 — 图推理器.

融合世界先进方案的图推理引擎:
- Neo4j Cypher: 加权最短路径 / K-最短路径 / 子图模式匹配
- GraphRAG: 全局/局部搜索 + 多跳推理 + 证据聚合
- 知识图谱推理 (OWL Reasoner / KGAT / TransE): 前向链式规则推理 + 链接预测
- 类比推理 (Structure-Mapping Theory): 相似关系模式迁移

线程安全: 所有公开方法通过 threading.RLock 保护, 支持重入。
依赖: 仅标准库 + pydantic v2, 无外部依赖。
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from enum import Enum
from threading import RLock
from typing import Any

from pydantic import BaseModel, Field

from .exceptions import L3Error
from .models import EntityType, KnowledgeEntity, KnowledgeTriple, RelationType
from .store import KnowledgeStore


class ReasoningError(L3Error):
    """图推理错误 — 路径查找失败 / 规则推理矛盾 / 模式匹配异常等。"""

    def __init__(self, detail: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__("L3_REASONING", detail=detail or "图推理失败", context=context)


class ReasoningMode(str, Enum):
    """推理模式 (借鉴 Neo4j 查询类型 + GraphRAG 检索模式 + KG 推理任务分类)。"""

    PATH_FINDING = "path_finding"
    MULTI_HOP = "multi_hop"
    RULE_INFERENCE = "rule_inference"
    LINK_PREDICTION = "link_prediction"
    PATTERN_MATCH = "pattern_match"
    ANALOGY = "analogy"


class PathResult(BaseModel):
    """路径查找结果。"""

    start_id: str
    end_id: str
    path: list[str] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    total_weight: float = 0.0
    hop_count: int = 0
    explanation: str = ""


class ReasoningResult(BaseModel):
    """统一推理结果。"""

    mode: ReasoningMode
    query: str
    answers: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning_chain: list[str] = Field(default_factory=list)
    evidence_triples: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: float = 0.0


class InferenceRule(BaseModel):
    """推理规则 (借鉴 OWL 公理 + Datalog 规则).

    condition_pattern 描述触发条件, inference_pattern 描述推理结论。
    支持的规则类型 (condition_pattern["type"]):
    - "transitive": 传递性 (A -pred-> B, B -pred-> C => A -pred-> C)
    - "inverse": 逆关系 (A -pred-> B => B -inverse_pred-> A)
    - "match_infer": 通用匹配推理 (按谓词匹配并推断)
    """

    rule_id: str
    name: str
    condition_pattern: dict[str, Any] = Field(default_factory=dict)
    inference_pattern: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str = ""


class GraphReasoner:
    """图推理器 (借鉴 Neo4j Cypher 路径查询 + GraphRAG 全局/局部搜索 + KG 推理).

    功能:
    1. 加权最短路径 (Dijkstra 算法)
    2. K-最短路径 (Yen 算法简化版)
    3. 多跳推理 (带约束的图遍历)
    4. 前向链式规则推理
    5. 链接预测 (共同邻居 + Jaccard 相似度)
    6. 子图模式匹配
    7. 类比推理 (相似关系模式)
    8. 路径解释生成

    线程安全: 所有公开方法通过 RLock 保护, RLock 可重入以支持
    公开方法间的相互调用 (如 find_k_shortest_paths 调用 find_shortest_path)。
    """

    def __init__(self, store: KnowledgeStore) -> None:
        self._store: KnowledgeStore = store
        self._rules: list[InferenceRule] = []
        self._lock: RLock = RLock()
        self._init_default_rules()

    def find_shortest_path(
        self, start_id: str, end_id: str, *,
        max_depth: int = 10, directed: bool = True,
    ) -> PathResult | None:
        """查找加权最短路径 (Dijkstra 算法, 基于 heapq 优先队列).

        边权重由三元组置信度推导 (置信度越高权重越低)。返回 None 表示无路径。
        """
        with self._lock:
            core = self._dijkstra_core(start_id, end_id, max_depth=max_depth, directed=directed)
            if core is None:
                return None
            path, edges, total_weight = core
            return self._make_path_result(start_id, end_id, path, edges, total_weight)

    def find_k_shortest_paths(
        self, start_id: str, end_id: str, k: int = 3, max_depth: int = 10,
    ) -> list[PathResult]:
        """查找 K 条最短路径 (Yen 算法简化版).

        先找最短路径, 再逐边禁用已发现路径上的边, 在禁用约束下重新查找
        最短路径作为候选, 重复直至收集 K 条互异路径或无更多候选。
        """
        with self._lock:
            first = self._dijkstra_core(start_id, end_id, max_depth=max_depth, directed=True)
            if first is None:
                return []

            path, edges, total_weight = first
            results: list[PathResult] = [
                self._make_path_result(start_id, end_id, path, edges, total_weight)
            ]
            seen: set[tuple[str, ...]] = {tuple(path)}

            for _ in range(k - 1):
                best: tuple[list[str], list[dict[str, Any]], float] | None = None
                best_sig: tuple[str, ...] | None = None

                for base in results:
                    bp = base.path
                    for i in range(len(bp) - 1):
                        forbidden = {(bp[i], bp[i + 1])}
                        alt = self._dijkstra_core(
                            start_id, end_id, max_depth=max_depth,
                            directed=True, forbidden_edges=forbidden,
                        )
                        if alt is None:
                            continue
                        ap, ae, aw = alt
                        sig = tuple(ap)
                        if sig in seen:
                            continue
                        if best is None or aw < best[2]:
                            best = (ap, ae, aw)
                            best_sig = sig

                if best is None:
                    break
                ap, ae, aw = best
                results.append(self._make_path_result(start_id, end_id, ap, ae, aw))
                if best_sig is not None:
                    seen.add(best_sig)

            return results

    def multi_hop_reasoning(
        self, start_id: str, relations: list[str], *, max_depth: int = 5,
    ) -> list[dict[str, Any]]:
        """多跳推理 (按关系类型序列遍历图).

        从起点出发, 依次沿 relations 序列中的每个谓词扩展一跳,
        收集每跳可达的实体及其遍历路径。
        """
        with self._lock:
            rels: list[str] = list(relations)[:max_depth]
            frontier: dict[str, tuple[list[str], list[str]]] = {start_id: ([start_id], [])}
            all_answers: list[dict[str, Any]] = []

            for hop_idx, rel in enumerate(rels):
                next_frontier: dict[str, tuple[list[str], list[str]]] = {}
                for eid, (epath, erels) in frontier.items():
                    for t in self._store.triple_store.get_outgoing(eid, predicate=rel):
                        if not t.object_id:
                            continue
                        nid = t.object_id
                        npath = epath + [nid]
                        nrels = erels + [rel]
                        all_answers.append({
                            "entity_id": nid, "entity_name": self._entity_name(nid),
                            "hop": hop_idx + 1, "relation": rel, "path": npath,
                            "relations": list(nrels), "confidence": float(t.confidence),
                            "source_entity_id": eid,
                        })
                        if nid not in next_frontier:
                            next_frontier[nid] = (npath, nrels)
                frontier = next_frontier
                if not frontier:
                    break

            # 去重: 同一实体保留最短路径
            deduped: dict[str, dict[str, Any]] = {}
            for ans in all_answers:
                eid = ans["entity_id"]
                if eid not in deduped or len(ans["path"]) < len(deduped[eid]["path"]):
                    deduped[eid] = ans
            return list(deduped.values())

    def forward_chaining(self, max_iterations: int = 10) -> list[dict[str, Any]]:
        """前向链式规则推理.

        迭代应用所有推理规则, 将新推理出的三元组加入工作集继续推理,
        直至无新推理结果 (收敛) 或达到 max_iterations 上限。
        """
        with self._lock:
            working: set[tuple[str, str, str]] = set()
            for t in self._get_all_triples():
                if t.object_id:
                    working.add((t.subject_id, t.predicate, t.object_id))

            inferred_results: list[dict[str, Any]] = []
            for iteration in range(max_iterations):
                new_inferences: list[dict[str, Any]] = []
                for rule in self._rules:
                    for g in self._apply_rule(rule, working):
                        key = (g["subject_id"], g["predicate"], g["object_id"])
                        if key in working:
                            continue
                        working.add(key)
                        new_inferences.append(g)
                if not new_inferences:
                    break
                for g in new_inferences:
                    inferred_results.append({
                        "subject_id": g["subject_id"], "predicate": g["predicate"],
                        "object_id": g["object_id"], "rule_id": g.get("rule_id", ""),
                        "rule_name": g.get("rule_name", ""), "confidence": g.get("confidence", 1.0),
                        "iteration": iteration + 1, "inferred": True,
                    })
            return inferred_results

    def predict_links(self, entity_id: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        """链接预测 (共同邻居 + Jaccard 相似度).

        共同邻居越多、邻居集 Jaccard 相似度越高的实体对, 越可能存在直接连接。
        """
        with self._lock:
            target_nbrs = set(self._get_neighbor_ids(entity_id))
            target_nbrs.discard(entity_id)
            results: list[dict[str, Any]] = []

            for cand in self._get_all_entity_ids():
                if cand == entity_id or cand in target_nbrs:
                    continue
                cand_nbrs = set(self._get_neighbor_ids(cand))
                cand_nbrs.discard(cand)
                common = target_nbrs & cand_nbrs
                if not common:
                    continue
                union = target_nbrs | cand_nbrs
                jaccard = len(common) / len(union) if union else 0.0
                results.append({
                    "entity_id": cand, "entity_name": self._entity_name(cand),
                    "score": round(jaccard, 6), "common_neighbors": len(common),
                    "common_neighbor_ids": sorted(common),
                    "predicted_relation": self._predict_relation(entity_id, cand, common),
                })

            results.sort(key=lambda x: (x["score"], x["common_neighbors"]), reverse=True)
            return results[:top_k]

    def pattern_match(self, pattern: dict[str, Any]) -> list[dict[str, Any]]:
        """子图模式匹配 (借鉴 Cypher 图模式 + SPARQL 基本图模式).

        Pattern 示例::
            {"nodes": [{"var": "x", "type": "MATERIAL"}, {"var": "y", "type": "METHOD"}],
             "edges": [{"from": "x", "to": "y", "predicate": "derived_from"}]}

        节点约束支持 type 和 name, 边约束支持 from / to / predicate。
        采用回溯搜索, 在两个端点均绑定后即时校验边约束以剪枝。
        """
        with self._lock:
            nodes = pattern.get("nodes", [])
            edges = pattern.get("edges", [])
            var_order = [n["var"] for n in nodes]

            candidates: dict[str, list[str]] = {}
            for node in nodes:
                var, etype, name = node["var"], node.get("type"), node.get("name")
                if etype:
                    cand = [e.entity_id for e in self._store.entity_store.find_by_type(etype)]
                else:
                    cand = self._get_all_entity_ids()
                if name:
                    cand = [eid for eid in cand if self._entity_name(eid) == name]
                candidates[var] = cand

            results: list[dict[str, Any]] = []
            self._backtrack_match(var_order, 0, {}, candidates, edges, results)
            return results

    def analogical_reasoning(
        self, source_pair: tuple[str, str], target_entity: str,
    ) -> list[dict[str, Any]]:
        """类比推理 (关系模式迁移).

        找出 source_pair (A, B) 之间的关系谓词, 再找出与 target_entity
        以相同谓词关联的实体。例如 (Dy3+, YAG) 通过 "doped_in" 关联,
        则查找 (Eu3+, ?) 中同样通过 "doped_in" 关联的实体。
        """
        with self._lock:
            src_a, src_b = source_pair
            relations: list[tuple[str, float]] = [
                (t.predicate, float(t.confidence))
                for t in self._store.triple_store.get_by_subject(src_a)
                if t.object_id == src_b
            ]

            results: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for pred, conf in relations:
                for t in self._store.triple_store.get_outgoing(target_entity, predicate=pred):
                    if not t.object_id or (pred, t.object_id) in seen:
                        continue
                    seen.add((pred, t.object_id))
                    results.append({
                        "source_relation": pred, "source_pair": [src_a, src_b],
                        "source_confidence": conf, "target_entity": target_entity,
                        "predicted_entity": t.object_id,
                        "predicted_entity_name": self._entity_name(t.object_id),
                        "confidence": round(min(conf, float(t.confidence)) * 0.8, 6),
                        "reasoning": (
                            f"{self._entity_name(src_a)} 与 {self._entity_name(src_b)} "
                            f"通过 [{pred}] 关联, {self._entity_name(target_entity)} "
                            f"同样通过 [{pred}] 关联到 {self._entity_name(t.object_id)}"
                        ),
                    })
            results.sort(key=lambda x: x["confidence"], reverse=True)
            return results

    def explain_path(self, path: list[str], edges: list[dict]) -> str:
        """生成人类可读的路径解释 (格式: A →[predicate]→ B →[predicate]→ C)。"""
        if not path:
            return ""
        names = [self._entity_name(eid) for eid in path]
        if len(path) == 1:
            return names[0]
        parts: list[str] = [names[0]]
        for i in range(1, len(path)):
            pred, weight_str = "?", ""
            if i - 1 < len(edges):
                edge = edges[i - 1]
                pred = edge.get("predicate", "?")
                w = edge.get("weight")
                if w is not None:
                    weight_str = f" (w={round(float(w), 4)})"
            parts.append(f" →[{pred}]{weight_str}→ {names[i]}")
        return "".join(parts)

    def add_rule(self, rule: InferenceRule) -> None:
        """添加推理规则。"""
        with self._lock:
            self._rules.append(rule)

    def get_rules(self) -> list[InferenceRule]:
        """获取所有推理规则 (返回副本)。"""
        with self._lock:
            return list(self._rules)

    def get_stats(self) -> dict[str, Any]:
        """获取推理器统计信息。"""
        with self._lock:
            return {
                "rules_count": len(self._rules),
                "entities_count": self._store.entity_count(),
                "triples_count": self._store.triple_count(),
                "relation_types": len(RelationType),
                "reasoning_modes": [m.value for m in ReasoningMode],
            }

    def reason(self, query: str, mode: ReasoningMode, **kwargs: Any) -> ReasoningResult:
        """统一推理入口, 根据模式分发到具体推理方法。"""
        start_ts = time.perf_counter()
        answers: list[dict[str, Any]] = []
        chain: list[str] = []
        evidence: list[dict[str, Any]] = []
        confidence = 0.0

        try:
            if mode == ReasoningMode.PATH_FINDING:
                pr = self.find_shortest_path(
                    kwargs.get("start_id", ""), kwargs.get("end_id", ""),
                    max_depth=kwargs.get("max_depth", 10))
                if pr is not None:
                    answers = [pr.model_dump()]
                    chain = [pr.explanation]
                    evidence = list(pr.edges)
                    confidence = 1.0 / (1.0 + pr.total_weight)
            elif mode == ReasoningMode.MULTI_HOP:
                answers = self.multi_hop_reasoning(
                    kwargs.get("start_id", ""), kwargs.get("relations", []),
                    max_depth=kwargs.get("max_depth", 5))
                chain = [f"多跳推理: 找到 {len(answers)} 个可达实体"]
                if answers:
                    confidence = sum(a.get("confidence", 1.0) for a in answers) / len(answers)
            elif mode == ReasoningMode.RULE_INFERENCE:
                answers = self.forward_chaining(kwargs.get("max_iterations", 10))
                chain = [f"前向链式推理: 推理出 {len(answers)} 条新三元组"]
                if answers:
                    confidence = sum(a.get("confidence", 1.0) for a in answers) / len(answers)
            elif mode == ReasoningMode.LINK_PREDICTION:
                answers = self.predict_links(kwargs.get("entity_id", ""), top_k=kwargs.get("top_k", 10))
                chain = [f"链接预测: 预测 {len(answers)} 条潜在链接"]
                if answers:
                    confidence = answers[0].get("score", 0.0)
            elif mode == ReasoningMode.PATTERN_MATCH:
                answers = self.pattern_match(kwargs.get("pattern", {}))
                chain = [f"模式匹配: 找到 {len(answers)} 个匹配绑定"]
                confidence = 1.0 if answers else 0.0
            elif mode == ReasoningMode.ANALOGY:
                answers = self.analogical_reasoning(
                    kwargs.get("source_pair", ("", "")), kwargs.get("target_entity", ""))
                chain = [f"类比推理: 找到 {len(answers)} 个类比结果"]
                if answers:
                    confidence = answers[0].get("confidence", 0.0)
        except L3Error:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ReasoningError(
                detail=f"推理失败 [{mode.value}]: {exc}",
                context={"mode": mode.value, "query": query}) from exc

        return ReasoningResult(
            mode=mode, query=query, answers=answers,
            confidence=round(confidence, 6), reasoning_chain=chain,
            evidence_triples=evidence, elapsed_ms=round((time.perf_counter() - start_ts) * 1000.0, 4))

    # --- 内部辅助方法 (调用方需已持有锁) ---

    def _dijkstra_core(
        self, start_id: str, end_id: str, *,
        max_depth: int = 10, directed: bool = True,
        forbidden_edges: set[tuple[str, str]] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]], float] | None:
        """Dijkstra 核心实现 (基于 heapq). 返回 (path, edges, total_weight) 或 None.

        forbidden_edges 为禁用边集合 {(from_id, to_id)}, 用于 K-最短路径。
        """
        forbidden = forbidden_edges or set()
        if start_id == end_id:
            return ([start_id], [], 0.0)

        dist: dict[str, float] = {start_id: 0.0}
        prev: dict[str, tuple[str, dict[str, Any]]] = {}
        visited: set[str] = set()
        counter = 0
        # 堆元素: (距离, 跳数, 序号, 节点), 序号保证全序避免比较节点字符串
        heap: list[tuple[float, int, int, str]] = [(0.0, 0, counter, start_id)]

        while heap:
            d, hops, _, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == end_id:
                break
            if hops >= max_depth:
                continue
            for v, edge in self._get_neighbors(u, directed=directed):
                if (u, v) in forbidden:
                    continue
                w = float(edge.get("weight", 1.0))
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = (u, edge)
                    counter += 1
                    heapq.heappush(heap, (nd, hops + 1, counter, v))

        if end_id not in prev:
            return None

        # 重建路径
        path: list[str] = []
        edges: list[dict[str, Any]] = []
        node = end_id
        while node != start_id:
            path.append(node)
            p, edge = prev[node]
            edges.append(edge)
            node = p
        path.append(start_id)
        path.reverse()
        edges.reverse()
        return (path, edges, dist[end_id])

    def _make_path_result(
        self, start_id: str, end_id: str, path: list[str],
        edges: list[dict[str, Any]], total_weight: float,
    ) -> PathResult:
        """构造 PathResult。"""
        return PathResult(
            start_id=start_id, end_id=end_id, path=path, edges=edges,
            total_weight=round(total_weight, 6), hop_count=max(len(path) - 1, 0),
            explanation=self.explain_path(path, edges))

    def _get_neighbors(
        self, entity_id: str, *, directed: bool = True,
    ) -> list[tuple[str, dict[str, Any]]]:
        """获取实体的邻居节点及边信息. True=仅出边, False=出边+入边。"""
        neighbors: list[tuple[str, dict[str, Any]]] = []
        for t in self._store.triple_store.get_by_subject(entity_id):
            if t.object_id:
                neighbors.append((t.object_id, self._make_edge(t, "out")))
        if not directed:
            for t in self._store.triple_store.get_by_object(entity_id):
                neighbors.append((t.subject_id, self._make_edge(t, "in")))
        return neighbors

    def _build_adjacency(
        self, *, directed: bool = True,
    ) -> dict[str, list[tuple[str, dict[str, Any]]]]:
        """从三元组存储构建完整邻接表。"""
        adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for t in self._get_all_triples():
            if not t.object_id:
                continue
            adjacency[t.subject_id].append((t.object_id, self._make_edge(t, "out")))
            if not directed:
                adjacency[t.object_id].append((t.subject_id, self._make_edge(t, "in")))
        return dict(adjacency)

    def _init_default_rules(self) -> None:
        """初始化默认推理规则.

        规则1: is_a 传递性 (A is_a B, B is_a C => A is_a C)
        规则2: has_part / part_of 逆关系 (A has_part B => B part_of A)
        规则3: derived_from 传递性 (A derived_from B, B derived_from C => A derived_from C)
        """
        self._rules = [
            InferenceRule(
                rule_id="rule_transitive_is_a", name="is_a 传递性",
                condition_pattern={"type": "transitive", "predicate": "is_a"},
                inference_pattern={"predicate": "is_a"}, confidence=0.95,
                description="若 A is_a B 且 B is_a C, 则 A is_a C"),
            InferenceRule(
                rule_id="rule_inverse_has_part", name="has_part/part_of 逆关系",
                condition_pattern={"type": "inverse", "predicate": "has_part",
                                   "inverse_predicate": RelationType.PART_OF.value},
                inference_pattern={"predicate": RelationType.PART_OF.value}, confidence=0.9,
                description="若 A has_part B, 则 B part_of A"),
            InferenceRule(
                rule_id="rule_transitive_derived_from", name="derived_from 传递性",
                condition_pattern={"type": "transitive",
                                   "predicate": RelationType.DERIVED_FROM.value},
                inference_pattern={"predicate": RelationType.DERIVED_FROM.value}, confidence=0.85,
                description="若 A derived_from B 且 B derived_from C, 则 A derived_from C"),
        ]

    def _apply_rule(
        self, rule: InferenceRule, working: set[tuple[str, str, str]],
    ) -> list[dict[str, Any]]:
        """应用单条推理规则到工作集, 返回推理出的三元组 dict 列表。"""
        cond = rule.condition_pattern
        rtype = cond.get("type", "match_infer")
        results: list[dict[str, Any]] = []

        def mk(s: str, p: str, o: str) -> dict[str, Any]:
            return {"subject_id": s, "predicate": p, "object_id": o,
                    "rule_id": rule.rule_id, "rule_name": rule.name,
                    "confidence": rule.confidence}

        if rtype == "transitive":
            pred = cond.get("predicate", "")
            adj: dict[str, set[str]] = defaultdict(set)
            for s, p, o in working:
                if p == pred and o:
                    adj[s].add(o)
            for a, bs in adj.items():
                for b in bs:
                    for c in adj.get(b, ()):
                        if c != a:
                            results.append(mk(a, pred, c))
        elif rtype == "inverse":
            pred = cond.get("predicate", "")
            inv = cond.get("inverse_predicate", rule.inference_pattern.get("predicate", pred))
            for s, p, o in working:
                if p == pred and o:
                    results.append(mk(o, inv, s))
        else:
            cp = cond.get("predicate", "")
            ip = rule.inference_pattern.get("predicate", cp)
            for s, p, o in working:
                if p == cp and o:
                    results.append(mk(s, ip, o))
        return results

    def _predict_relation(
        self, entity_id: str, candidate_id: str, common_neighbors: set[str],
    ) -> str:
        """预测最可能的关系谓词 (统计到共同邻居的边谓词频次, 取最高频)。"""
        pred_counts: dict[str, int] = defaultdict(int)
        for nbr in common_neighbors:
            for t in self._store.triple_store.get_by_subject(entity_id):
                if t.object_id == nbr:
                    pred_counts[t.predicate] += 1
            for t in self._store.triple_store.get_by_subject(candidate_id):
                if t.object_id == nbr:
                    pred_counts[t.predicate] += 1
        if not pred_counts:
            return RelationType.RELATED_TO.value
        return max(pred_counts, key=pred_counts.get)

    def _backtrack_match(
        self, var_order: list[str], idx: int, binding: dict[str, str],
        candidates: dict[str, list[str]], edges: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> None:
        """回溯搜索模式匹配绑定 (逐变量绑定, 两端点绑定后即时校验边约束剪枝)。"""
        if idx == len(var_order):
            results.append({
                "bindings": dict(binding),
                "names": {v: self._entity_name(eid) for v, eid in binding.items()},
            })
            return
        var = var_order[idx]
        for cand in candidates.get(var, []):
            binding[var] = cand
            if self._edges_partial_ok(edges, binding):
                self._backtrack_match(var_order, idx + 1, binding, candidates, edges, results)
            binding.pop(var, None)

    def _edges_partial_ok(self, edges: list[dict[str, Any]], binding: dict[str, str]) -> bool:
        """检查当前绑定下所有可校验的边约束是否满足。"""
        for edge in edges:
            fv, tv = edge.get("from"), edge.get("to")
            if fv in binding and tv in binding:
                if not self._edge_exists(binding[fv], binding[tv], edge.get("predicate")):
                    return False
        return True

    def _edge_exists(self, subject_id: str, object_id: str, predicate: str | None) -> bool:
        """检查指定边是否存在 (predicate 为 None 表示任意谓词)。"""
        for t in self._store.triple_store.get_by_subject(subject_id):
            if t.object_id == object_id and (predicate is None or t.predicate == predicate):
                return True
        return False

    def _make_edge(self, triple: KnowledgeTriple, direction: str = "out") -> dict[str, Any]:
        """构造边信息 dict。"""
        return {
            "triple_id": triple.triple_id, "predicate": triple.predicate,
            "weight": self._edge_weight(triple), "confidence": float(triple.confidence),
            "subject_id": triple.subject_id, "object_id": triple.object_id,
            "direction": direction,
        }

    def _edge_weight(self, triple: KnowledgeTriple) -> float:
        """推导边权重.

        优先使用 weight 字段 (若存在), 否则基于置信度推导:
        weight = 1.0 / confidence (置信度越高, 权重越低, 路径越短)。
        """
        weight = getattr(triple, "weight", None)
        if isinstance(weight, (int, float)) and weight > 0:
            return float(weight)
        confidence = float(getattr(triple, "confidence", 1.0) or 0.0)
        if confidence <= 0:
            confidence = 1e-6
        return 1.0 / confidence

    def _entity_name(self, entity_id: str) -> str:
        """获取实体名称, 不存在时回退为 ID。"""
        entity = self._store.get_entity(entity_id)
        return entity.name if entity is not None else entity_id

    def _get_neighbor_ids(self, entity_id: str) -> list[str]:
        """获取实体邻居 ID 列表 (双向, 去重)。"""
        ids: set[str] = set()
        for t in self._store.triple_store.get_by_subject(entity_id):
            if t.object_id:
                ids.add(t.object_id)
        for t in self._store.triple_store.get_by_object(entity_id):
            ids.add(t.subject_id)
        ids.discard(entity_id)
        return list(ids)

    def _get_all_triples(self) -> list[KnowledgeTriple]:
        """获取存储中所有三元组.

        优先访问 TripleStore 内部字典, 回退到按谓词索引遍历。
        """
        internal = getattr(self._store.triple_store, "_triples", None)
        if isinstance(internal, dict):
            return list(internal.values())
        result: list[KnowledgeTriple] = []
        seen: set[str] = set()
        for rt in RelationType:
            for t in self._store.triple_store.get_by_predicate(rt.value):
                if t.triple_id not in seen:
                    seen.add(t.triple_id)
                    result.append(t)
        return result

    def _get_all_entity_ids(self) -> list[str]:
        """获取存储中所有实体 ID。"""
        internal = getattr(self._store.entity_store, "_entities", None)
        if isinstance(internal, dict):
            return list(internal.keys())
        return [e.entity_id for e in self._store.entity_store.list_entities(limit=10**9)]


__all__ = [
    "ReasoningMode", "PathResult", "ReasoningResult", "InferenceRule",
    "GraphReasoner", "ReasoningError",
]
