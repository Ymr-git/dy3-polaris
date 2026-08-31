"""Request-local Knowledge/Learner fusion for Diagnosis.

This module projects the frozen R-06B concept foundation and R-06C relation
network over aligned R-05 learner signals.  It is an internal teaching view:
it does not persist state, change scientific answers, or expose a public API.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from dy3_polaris.l2.kp_catalog import NEW_KP_NAMES, to_new_id
from dy3_polaris.l3.concept_foundation import (
    ConceptFoundation,
    KnowledgeConcept,
    MappingAssetType,
    build_concept_foundation,
)
from dy3_polaris.l3.concept_relations import (
    ConceptRelation,
    ConceptRelationNetwork,
    ConceptRelationType,
    RelationStatus,
    build_concept_relation_network,
)


_MASTERY_THRESHOLD = 0.7
_GENERIC_TERMS = {
    "材料", "问题", "分析", "评价", "研究", "应用", "机制", "性能",
    "影响", "方法", "基础", "知识", "概念", "发光", "照明", "dy3+",
}


@dataclass(frozen=True, slots=True)
class ConceptMasteryProjection:
    """Conservative Concept projection of existing aligned KP mastery."""

    concept_id: str
    kp_ids: tuple[str, ...]
    value: float | None
    source: str

    @property
    def mastered(self) -> bool:
        return self.value is not None and self.value >= _MASTERY_THRESHOLD


@dataclass(frozen=True, slots=True)
class ConceptLearningPath:
    """One explainable next-step decision, not a stored or fixed curriculum."""

    current_position: str
    next_concept: str
    reason: str
    prerequisite_gap: tuple[str, ...]
    expected_outcome: str
    confidence: float


@dataclass(frozen=True, slots=True)
class KnowledgeLearningContext:
    """Private Diagnosis input combining knowledge relations and learner state."""

    learner_id: str
    learning_goal: tuple[str, ...]
    target_concepts: tuple[str, ...]
    concept_mastery: Mapping[str, ConceptMasteryProjection]
    active_misconception_concepts: tuple[str, ...]
    evidence_available_concepts: tuple[str, ...]
    concept_to_kps: Mapping[str, tuple[str, ...]]
    concept_names: Mapping[str, str]
    learning_path: ConceptLearningPath
    trace: tuple[str, ...]


def _normalize(value: Any) -> str:
    text = str(value or "").casefold().replace("³⁺", "3+").replace("dy(iii)", "dy3+")
    return re.sub(r"[\s\-–—_:/,，。；;（）()\[\]]+", "", text)


def _terms(concept: KnowledgeConcept) -> tuple[str, ...]:
    values = (*concept.aliases, *(NEW_KP_NAMES.get(kp_id, "") for kp_id in concept.related_kps))
    terms: list[str] = []
    for value in values:
        normalized = _normalize(value)
        if len(normalized) >= 2:
            terms.append(normalized)
        for part in re.split(r"[\s、，,；;：:（）()\[\]/与和及]+", str(value or "")):
            normalized_part = _normalize(part)
            if len(normalized_part) >= 2 and normalized_part not in _GENERIC_TERMS:
                terms.append(normalized_part)
    return tuple(dict.fromkeys(terms))


def resolve_concepts(
    network: ConceptRelationNetwork,
    *values: Any,
    limit: int = 5,
) -> tuple[str, ...]:
    """Resolve concepts only from their canonical aliases and mapped KP names."""

    text = _normalize(" ".join(str(value or "") for value in values))
    if not text:
        return ()
    scored: list[tuple[int, str]] = []
    for concept in network.foundation.concepts.values():
        matched = [term for term in _terms(concept) if term in text]
        if not matched:
            continue
        score = sum(min(len(term), 12) for term in matched)
        exact = _normalize(concept.canonical_name) in text
        scored.append((score + (12 if exact else 0), concept.concept_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return ()
    cutoff = max(2, int(scored[0][0] * 0.55))
    return tuple(concept_id for score, concept_id in scored[:limit] if score >= cutoff)


def project_concept_mastery(
    foundation: ConceptFoundation,
    mastery: Mapping[str, float] | None,
) -> Mapping[str, ConceptMasteryProjection]:
    """Project KP model state without treating missing KPs as failed mastery."""

    values = {
        to_new_id(str(kp_id)): float(value)
        for kp_id, value in dict(mastery or {}).items()
        if isinstance(value, (int, float))
    }
    projected: dict[str, ConceptMasteryProjection] = {}
    for concept_id, concept in foundation.concepts.items():
        kp_ids = tuple(dict.fromkeys(to_new_id(item) for item in concept.related_kps))
        known = [values[kp_id] for kp_id in kp_ids if kp_id in values]
        # A Concept can span more than one KP.  The minimum avoids claiming
        # Concept mastery when only one learning projection is strong.
        value = min(known) if known else None
        projected[concept_id] = ConceptMasteryProjection(
            concept_id=concept_id,
            kp_ids=kp_ids,
            value=round(value, 4) if value is not None else None,
            source="aligned_kp_mastery" if known else "unknown",
        )
    return MappingProxyType(projected)


def _incoming_prerequisites(
    network: ConceptRelationNetwork,
) -> Mapping[str, tuple[ConceptRelation, ...]]:
    incoming: dict[str, list[ConceptRelation]] = {
        concept_id: [] for concept_id in network.foundation.concepts
    }
    for relation in network.relations:
        if (
            relation.status is RelationStatus.CURATED
            and relation.relation_type is ConceptRelationType.PREREQUISITE_OF
        ):
            incoming[relation.target_concept_id].append(relation)
    return MappingProxyType({key: tuple(value) for key, value in incoming.items()})


def _prerequisite_chain(
    network: ConceptRelationNetwork,
    targets: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    incoming = _incoming_prerequisites(network)
    ordered: list[str] = []
    relation_ids: list[str] = []
    visiting: set[str] = set()

    def visit(concept_id: str) -> None:
        if concept_id in visiting:
            return
        visiting.add(concept_id)
        for relation in incoming.get(concept_id, ()):
            visit(relation.source_concept_id)
            if relation.relation_id not in relation_ids:
                relation_ids.append(relation.relation_id)
        if concept_id not in ordered:
            ordered.append(concept_id)

    for target in targets:
        visit(target)
    return tuple(ordered), tuple(relation_ids)


def _active_misconception_concepts(
    network: ConceptRelationNetwork,
    misconceptions: Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    values: list[str] = []
    for item in misconceptions:
        if str(item.get("status") or "").upper() != "ACTIVE":
            continue
        values.extend(resolve_concepts(network, item.get("topic"), item.get("belief"), limit=3))
    return tuple(dict.fromkeys(values))


def _misconception_learning_candidates(
    network: ConceptRelationNetwork,
    concept_ids: Iterable[str],
) -> tuple[tuple[str, ConceptRelation], ...]:
    candidates: list[tuple[str, ConceptRelation]] = []
    preferred = {
        ConceptRelationType.EVALUATED_BY,
        ConceptRelationType.EXPLAINS,
        ConceptRelationType.PREREQUISITE_OF,
    }
    for concept_id in concept_ids:
        for relation in network.outgoing(concept_id):
            if relation.status is RelationStatus.CURATED and relation.relation_type in preferred:
                candidates.append((relation.target_concept_id, relation))
    return tuple(candidates)


def _has_candidate_evidence(store: Any, concept: KnowledgeConcept) -> bool:
    """Check real term mentions without claiming scientific support."""

    if store is None or not callable(getattr(store, "search_text", None)):
        return bool(concept.evidence_refs)
    for alias in sorted(concept.aliases, key=len, reverse=True):
        if len(_normalize(alias)) < 3:
            continue
        try:
            hits = store.search_text(alias, top_k=1)
        except Exception:  # noqa: BLE001 - unavailable evidence remains unavailable
            return False
        for hit in hits or ():
            chunk = hit[0] if isinstance(hit, tuple) else hit
            if _normalize(alias) in _normalize(getattr(chunk, "content", "")):
                return True
    return False


def build_knowledge_learning_context(
    *,
    learner_id: str,
    query: str,
    mastery: Mapping[str, float] | None = None,
    weak_kps: tuple[str, ...] = (),
    misconceptions: tuple[Mapping[str, Any], ...] = (),
    teaching_confidence: float | None = None,
    l3_store: Any | None = None,
    network: ConceptRelationNetwork | None = None,
) -> KnowledgeLearningContext:
    """Build one dynamic, traceable learning decision from current facts."""

    relation_network = network or build_concept_relation_network()
    foundation = relation_network.foundation
    concept_mastery = project_concept_mastery(foundation, mastery)
    concept_to_kps = MappingProxyType({
        concept_id: projection.kp_ids
        for concept_id, projection in concept_mastery.items()
    })
    targets = resolve_concepts(relation_network, query)
    misconception_concepts = _active_misconception_concepts(
        relation_network, misconceptions
    )
    weak_concepts = tuple(
        concept_id
        for concept_id, projection in concept_mastery.items()
        if set(projection.kp_ids).intersection({to_new_id(item) for item in weak_kps})
    )
    if not targets:
        targets = tuple(dict.fromkeys((*misconception_concepts, *weak_concepts)))[:5]

    chain, prerequisite_relation_ids = _prerequisite_chain(relation_network, targets)
    target_set = set(targets)
    prerequisite_gaps = tuple(
        concept_id
        for concept_id in chain
        if concept_id not in target_set and not concept_mastery[concept_id].mastered
    )
    current_position = next(
        (concept_id for concept_id in reversed(chain) if concept_mastery[concept_id].mastered),
        "unknown",
    )

    correction_candidates = _misconception_learning_candidates(
        relation_network, misconception_concepts
    )
    chain_set = set(chain)
    candidates: list[tuple[str, ConceptRelation | None, str]] = []
    for item in prerequisite_gaps:
        relation = next(
            (
                edge
                for edge in relation_network.outgoing(item)
                if edge.relation_type is ConceptRelationType.PREREQUISITE_OF
                and edge.target_concept_id in chain_set
            ),
            None,
        )
        candidates.append((item, relation, "prerequisite_gap"))
    candidates.extend((item, relation, "misconception_relation") for item, relation in correction_candidates)
    candidates.extend(
        (item, None, "learning_goal")
        for item in targets
        if not concept_mastery[item].mastered
    )
    if not candidates and targets:
        for target in targets:
            for relation in relation_network.outgoing(target):
                if relation.status is RelationStatus.CURATED:
                    candidates.append((relation.target_concept_id, relation, "relation_progression"))

    candidate_ids = tuple(dict.fromkeys(item[0] for item in candidates))
    evidence_available = tuple(
        concept_id
        for concept_id in candidate_ids
        if _has_candidate_evidence(l3_store, foundation.concepts[concept_id])
    )
    evidence_set = set(evidence_available)
    selected: tuple[str, ConceptRelation | None, str] | None = None
    for reason_type in ("prerequisite_gap", "misconception_relation", "learning_goal", "relation_progression"):
        group = [item for item in candidates if item[2] == reason_type]
        if group:
            # Evidence availability may rank equivalent choices, but it cannot
            # skip an earlier unmet prerequisite.
            selected = (
                group[0]
                if reason_type == "prerequisite_gap"
                else next((item for item in group if item[0] in evidence_set), group[0])
            )
            break

    trace: list[str] = [
        f"goal:{concept_id}" for concept_id in targets
    ]
    trace.extend(f"prerequisite:{relation_id}" for relation_id in prerequisite_relation_ids)
    trace.extend(f"misconception:{concept_id}" for concept_id in misconception_concepts)
    trace.extend(f"evidence_candidate:{concept_id}" for concept_id in evidence_available)

    if selected is None:
        next_concept = "unknown"
        reason = "no mapped Concept Relation supports a next learning decision"
        expected_outcome = "no teaching progression inferred"
        confidence = 0.0
    else:
        next_concept, relation, reason_type = selected
        concept = foundation.concepts[next_concept]
        if relation is not None:
            reason = (
                f"{reason_type}: {relation.relation_type.value} via "
                f"{relation.relation_id}; {relation.description}"
            )
            expected_outcome = f"理解{concept.canonical_name}在该科学关系中的作用"
            relation_confidence = relation.confidence
            trace.append(f"selected_relation:{relation.relation_id}")
        else:
            reason = f"{reason_type}: mapped learning goal"
            expected_outcome = (
                f"补齐{concept.canonical_name}前置知识"
                if reason_type == "prerequisite_gap"
                else f"掌握{concept.canonical_name}以推进当前学习目标"
            )
            relation_confidence = 0.7
        base = min(0.8, relation_confidence)
        if next_concept not in evidence_set:
            base = min(base, 0.55)
        if teaching_confidence is not None:
            base = min(base, max(0.0, float(teaching_confidence)))
        confidence = round(base, 4)
        trace.append(f"selected:{next_concept}:{reason_type}")

    return KnowledgeLearningContext(
        learner_id=learner_id,
        learning_goal=tuple(
            foundation.concepts[item].canonical_name for item in targets
        ),
        target_concepts=targets,
        concept_mastery=concept_mastery,
        active_misconception_concepts=misconception_concepts,
        evidence_available_concepts=evidence_available,
        concept_to_kps=concept_to_kps,
        concept_names=MappingProxyType({
            concept_id: concept.canonical_name
            for concept_id, concept in foundation.concepts.items()
        }),
        learning_path=ConceptLearningPath(
            current_position=current_position,
            next_concept=next_concept,
            reason=reason,
            prerequisite_gap=prerequisite_gaps,
            expected_outcome=expected_outcome,
            confidence=confidence,
        ),
        trace=tuple(trace),
    )


def public_knowledge_learning_projection(
    context: KnowledgeLearningContext,
) -> dict[str, Any]:
    """Expose a bounded, question-centred Concept graph without private state.

    The projection contains only canonical Concept identities, curated relation
    labels, aligned mastery classifications and KP links.  It intentionally
    excludes raw learner records, model internals and evidence text.
    """

    if not isinstance(context, KnowledgeLearningContext):
        return {}
    path = context.learning_path
    public_reason = str(path.reason)
    if public_reason.startswith("prerequisite_gap:"):
        public_reason = public_reason.replace(
            "prerequisite_gap:", "prerequisites_to_learn:", 1
        )
    concept_ids = tuple(dict.fromkeys((
        *context.target_concepts,
        *path.prerequisite_gap,
        *((path.current_position,) if path.current_position != "unknown" else ()),
        *((path.next_concept,) if path.next_concept != "unknown" else ()),
    )))[:12]
    concept_set = set(concept_ids)
    nodes: list[dict[str, Any]] = []
    for concept_id in concept_ids:
        projection = context.concept_mastery.get(concept_id)
        value = projection.value if projection is not None else None
        state = (
            "UNKNOWN"
            if value is None
            else "MASTERED"
            if projection.mastered
            else "LEARNING_GAP"
        )
        nodes.append({
            "concept_id": concept_id,
            "name": context.concept_names.get(concept_id, concept_id),
            "role": (
                "TARGET"
                if concept_id in context.target_concepts
                else "PREREQUISITE"
                if concept_id in path.prerequisite_gap
                else "NEXT"
            ),
            "learner_state": state,
            "mastery": value,
            "kp_ids": list(context.concept_to_kps.get(concept_id, ())),
            "evidence_available": concept_id in context.evidence_available_concepts,
        })

    network = build_concept_relation_network()
    edges = [
        {
            "relation_id": relation.relation_id,
            "source": relation.source_concept_id,
            "target": relation.target_concept_id,
            "relation_type": relation.relation_type.value,
            "status": relation.status.value,
        }
        for relation in network.relations
        if relation.status is RelationStatus.CURATED
        and relation.source_concept_id in concept_set
        and relation.target_concept_id in concept_set
    ]
    return {
        "learning_goal": list(context.learning_goal),
        "nodes": nodes,
        "edges": edges,
        "path": {
            "current_position": path.current_position,
            "next_concept": path.next_concept,
            # Public learner wording; the private ConceptLearningPath field
            # name remains inside the R06 fusion boundary.
            "prerequisites_to_learn": list(path.prerequisite_gap),
            "reason": public_reason,
            "expected_outcome": path.expected_outcome,
            "confidence": path.confidence,
        },
        "source_class": "DERIVED",
    }


__all__ = [
    "ConceptLearningPath",
    "ConceptMasteryProjection",
    "KnowledgeLearningContext",
    "build_knowledge_learning_context",
    "public_knowledge_learning_projection",
    "project_concept_mastery",
    "resolve_concepts",
]
