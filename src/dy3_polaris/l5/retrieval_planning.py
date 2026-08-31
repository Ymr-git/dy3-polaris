"""R-03D private subtask-aware retrieval planning and evidence organization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any

from dy3_polaris.l5.agent_contracts import AgentInput
from dy3_polaris.l5.collaboration_context import Subtask
from dy3_polaris.l5.task_understanding import TaskMode


@dataclass(frozen=True, slots=True)
class RetrievalNeed:
    required: bool
    reason: str
    information_gap: tuple[str, ...]
    target_claims: tuple[str, ...]
    target_entities: tuple[str, ...]
    target_parameters: tuple[str, ...]
    target_conditions: tuple[str, ...]
    source_preference: tuple[str, ...]
    coverage_requirement: str


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    task_id: str
    subtask_id: str
    purpose: str
    original_query: str
    rewritten_queries: tuple[str, ...]
    entities: tuple[str, ...]
    filters: tuple[str, ...]
    required_evidence_types: tuple[str, ...]
    top_k: int
    diversity_requirement: str
    rerank_profile: str
    reason: str
    expansion_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    chunk_reference: str
    source_reference: str
    entity: str
    material_system: str
    conditions: tuple[str, ...]
    supported_claim_types: tuple[str, ...]
    relevance_score: float
    source_quality: str
    provenance_reference: str
    conflict_flag: bool
    content: str
    rerank_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidencePack:
    task_id: str
    subtask_id: str
    items: tuple[EvidenceItem, ...]
    coverage: tuple[str, ...]
    conflicts: tuple[str, ...]
    missing_information: tuple[str, ...]
    version: int = 1
    parent_pack_ids: tuple[str, ...] = ()
    refresh_reason: str = ""
    requested_by: str = ""


_PARAMETER_TERMS = (
    "量子效率", "发射光谱", "色坐标", "色温", "CCT", "CRI", "寿命",
    "热稳定性", "激发", "掺杂浓度", "声子", "缺陷", "温度", "SPD",
    "蓝光风险", "暴露时间", "非辐射", "辐射跃迁",
)

_LUMINESCENT_ION_RE = re.compile(
    r"(?<![a-z])(dy|eu|er|yb|tb|ce|mn|cr|tm|ho|nd|sm|pr)"
    r"(?:2\+|3\+|4\+|²⁺|³⁺|⁴⁺)?(?![a-z])"
)

# These identifiers describe project-authored summaries or test fixtures, not
# independently traceable scientific sources.  They may remain in the local
# corpus for development/history, but they are not admissible as evidence for
# an answer released by the scientific review loop.
_NON_SCIENTIFIC_SOURCE_MARKERS = (
    "知识大纲",
    "persona-enrichment",
    "demo-seed",
    "mock",
    "placeholder",
    "textbook_fallback",
)


def _scientific_source_admissible(item: dict[str, Any]) -> bool:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    if metadata.get("source_type") == "curated_source_summary":
        # A project-maintained summary is only admissible when it preserves an
        # independently traceable source.  This prevents a convenient summary
        # field from becoming another ungrounded textbook fallback.
        if not str(metadata.get("source_uri") or "").strip():
            return False
        if metadata.get("evidence_status") != "reviewed":
            return False
    source = " ".join(
        str(value)
        for value in (
            item.get("document_id"),
            item.get("source"),
            metadata.get("document_id"),
            metadata.get("source"),
            metadata.get("source_name"),
        )
        if value
    ).casefold()
    if not source:
        # Missing provenance remains "unknown" and is handled by the later
        # grounding/release guard.  The source filter only rejects records
        # that are positively identified as project-authored placeholders.
        return True
    return not any(marker.casefold() in source for marker in _NON_SCIENTIFIC_SOURCE_MARKERS)


def _normalize_entity(value: str) -> str:
    return (
        str(value or "")
        .lower()
        .replace("²⁺", "2+")
        .replace("³⁺", "3+")
        .replace("⁴⁺", "4+")
        .replace(" ", "")
    )


def _entities(agent_input: AgentInput, subtask: Subtask) -> tuple[str, ...]:
    values = [entity.text for entity in agent_input.intent.domain_entities]
    values.extend(re.findall(r"[A-Z][A-Za-z0-9:+³⁺-]{1,24}", subtask.goal))
    return tuple(dict.fromkeys(value for value in values if value))


def _parameters(text: str) -> tuple[str, ...]:
    return tuple(term for term in _PARAMETER_TERMS if term.lower() in text.lower())


def build_retrieval_need(agent_input: AgentInput, subtask: Subtask) -> RetrievalNeed:
    required = subtask.evidence_need != "none"
    query = agent_input.user_query
    entities = _entities(agent_input, subtask)
    parameters = list(_parameters(f"{query} {subtask.goal}"))
    gaps: list[str] = []
    mode = agent_input.task_mode
    if "基质" in query:
        gaps.extend(("局域环境与对称性", "声子与非辐射损失", "缺陷占位与热稳定性"))
    if "发光效率" in query or "量子效率" in query:
        gaps.extend(("吸收与激发", "辐射与非辐射竞争", "浓度缺陷与热猝灭"))
    if mode is TaskMode.EVALUATE:
        gaps.extend(("spectral power distribution", "blue-light hazard relation", "exposure assumptions"))
        parameters.extend(("SPD", "蓝光风险", "暴露时间"))
    if mode is TaskMode.COMPARE:
        gaps.append("同条件评价指标")
    if not gaps:
        gaps.append(subtask.goal)
    return RetrievalNeed(
        required=required,
        reason=f"subtask evidence_need={subtask.evidence_need}",
        information_gap=tuple(dict.fromkeys(gaps)),
        target_claims=(subtask.goal,),
        target_entities=entities,
        target_parameters=tuple(dict.fromkeys(parameters)),
        target_conditions=tuple(agent_input.intent.ambiguity),
        source_preference=("domain_knowledge_base", "textbook_fallback"),
        coverage_requirement=("multi-concept" if len(gaps) > 1 else "direct"),
    )


def _rewrite_queries(
    agent_input: AgentInput, subtask: Subtask, need: RetrievalNeed
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    base = " ".join((agent_input.user_query, *need.target_entities)).strip()
    expansion: list[str] = []
    mode = agent_input.task_mode
    if mode is TaskMode.FACT_FIND:
        return ((f"{base} {subtask.goal}".strip(),), ())
    if mode is TaskMode.COMPARE:
        if "criteria" in subtask.type:
            expansion = ["量子效率", "发射光谱", "色度", "热稳定性", "测试条件"]
        elif "material_evidence" in subtask.type:
            expansion = ["发射", "色度", "效率", "热稳定性"]
        else:
            expansion = ["同条件比较", "证据差异"]
    elif mode is TaskMode.EVALUATE:
        expansion = ["spectral power distribution", "blue light hazard", "exposure", "CCT limitations"]
    elif "基质" in agent_input.user_query:
        expansion = ["局域晶场", "对称性", "声子能量", "非辐射", "缺陷占位", "热稳定性"]
    elif "发光效率" in agent_input.user_query:
        expansion = ["吸收激发", "辐射跃迁", "非辐射损失", "浓度猝灭", "热猝灭"]
    else:
        expansion = ["能级跃迁", "黄光蓝光", "发光机制"]
    if mode is TaskMode.EVALUATE:
        return ((
            f"{base} correlated color temperature spectral power distribution",
            f"{base} blue light hazard spectral weighting exposure",
            f"{base} healthy lighting CCT limitations",
        ), tuple(expansion))
    return ((f"{base} {subtask.goal} {' '.join(expansion)}".strip(),), tuple(expansion))


def build_retrieval_plan(agent_input: AgentInput, subtask: Subtask) -> RetrievalPlan:
    need = build_retrieval_need(agent_input, subtask)
    queries, expansions = _rewrite_queries(agent_input, subtask, need)
    return RetrievalPlan(
        task_id=agent_input.task_id,
        subtask_id=subtask.subtask_id,
        purpose=subtask.goal,
        original_query=agent_input.user_query,
        rewritten_queries=queries if need.required else (),
        entities=need.target_entities,
        filters=("explicit_entity_mismatch", "explicit_material_mismatch"),
        required_evidence_types=need.information_gap,
        top_k=6,
        diversity_requirement="source_and_claim",
        rerank_profile=agent_input.task_mode.value.lower(),
        reason=need.reason,
        expansion_terms=expansions,
    )


def build_retrieval_plans(agent_input: AgentInput) -> tuple[RetrievalPlan, ...]:
    return tuple(
        build_retrieval_plan(agent_input, subtask)
        for subtask in (agent_input.subtask, *agent_input.related_subtasks)
        if build_retrieval_need(agent_input, subtask).required
    )


def build_challenge_retrieval_plans(
    agent_input: AgentInput,
    missing_information: tuple[str, ...],
    *,
    reason: str,
) -> tuple[RetrievalPlan, ...]:
    """Derive targeted v2+ plans from an explicit Reviewer information gap."""
    missing = tuple(dict.fromkeys(str(item).strip() for item in missing_information if str(item).strip()))
    plans: list[RetrievalPlan] = []
    for plan in build_retrieval_plans(agent_input):
        targeted = tuple(
            f"{query} {' '.join(missing)}".strip()
            for query in plan.rewritten_queries
        )
        plans.append(
            replace(
                plan,
                rewritten_queries=targeted,
                required_evidence_types=tuple(
                    dict.fromkeys((*plan.required_evidence_types, *missing))
                ),
                reason=f"reviewer challenge: {reason}",
                expansion_terms=tuple(dict.fromkeys((*plan.expansion_terms, *missing))),
            )
        )
    return tuple(plans)


def _text(item: dict[str, Any]) -> str:
    return str(item.get("content") or item.get("text") or "")


def hard_filter(plan: RetrievalPlan, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target = _normalize_entity(" ".join(plan.entities))
    target_ions = set(_LUMINESCENT_ION_RE.findall(target))
    output: list[dict[str, Any]] = []
    for item in candidates:
        if not _scientific_source_admissible(item):
            continue
        text = _normalize_entity(_text(item))
        item_ion_values = _LUMINESCENT_ION_RE.findall(text)
        item_ions = set(item_ion_values)
        if target_ions and item_ions:
            target_count = sum(
                item_ion_values.count(symbol) for symbol in target_ions
            )
            other_count = sum(
                item_ion_values.count(symbol)
                for symbol in item_ions
                if symbol not in target_ions
            )
            # For a single-ion question, several incidental mentions of other
            # activators must be considered together.  Comparing only with
            # the most frequent other ion allowed mixed-ion survey passages
            # to outrank a direct Dy3+ mechanism paper.
            if target_count == 0 or target_count < other_count:
                continue
        material_targets = [entity.lower() for entity in plan.entities if ":" in entity]
        if material_targets and any(target in plan.purpose.lower() for target in material_targets):
            named = [value for value in re.findall(r"[a-z]{2,8}:dy", text)]
            if named and not any(target.split("³")[0] in text for target in material_targets):
                continue
        output.append(item)
    return output


def agent_aware_rerank(
    plan: RetrievalPlan,
    candidates: list[dict[str, Any]],
    base_scores: list[float] | None = None,
) -> list[tuple[dict[str, Any], float, tuple[str, ...]]]:
    base_scores = list(base_scores or ())
    # Chinese scientific questions are commonly written without spaces.  The
    # previous regex treated an entire sentence as one token, so direct
    # evidence and a generic paper both received semantic=0.  Use stable
    # Chinese bigrams from the original user question while preserving
    # chemical/ASCII tokens.  Expansion coverage is scored separately below.
    original = plan.original_query.lower()
    query_terms = set(re.findall(r"[A-Za-z0-9+³⁺/.-]+", original))
    for segment in re.findall(r"[一-鿿]+", original):
        query_terms.update(
            segment[index : index + 2]
            for index in range(max(0, len(segment) - 1))
        )
        if 2 <= len(segment) <= 6:
            query_terms.add(segment)
    normalized_purpose = _normalize_entity(plan.purpose)
    purpose_entities = [
        entity
        for entity in plan.entities
        if _normalize_entity(entity) in normalized_purpose
    ]
    if purpose_entities:
        longest = max(len(_normalize_entity(entity)) for entity in purpose_entities)
        purpose_entities = [
            entity for entity in purpose_entities
            if len(_normalize_entity(entity)) == longest
        ]
    seen_text: set[str] = set()
    seen_source: set[str] = set()
    ranked = []
    for index, item in enumerate(candidates):
        text = _text(item).lower()
        compact = re.sub(r"\s+", "", text)[:160]
        source = str(item.get("document_id") or item.get("source") or "")
        base = float(base_scores[index]) if index < len(base_scores) else 0.0
        overlap = sum(1 for term in query_terms if term and term in text)
        semantic = min(1.0, overlap / max(3, len(query_terms)))
        normalized_text = _normalize_entity(text)
        entity_hits = sum(
            1
            for entity in plan.entities
            if _normalize_entity(entity) in normalized_text
        )
        purpose_entity_hits = sum(
            1 for entity in purpose_entities
            if _normalize_entity(entity) in normalized_text
        )
        parameter_hits = sum(1 for value in plan.required_evidence_types if value.lower() in text)
        expansion_hits = sum(
            1
            for value in plan.expansion_terms
            if len(str(value).strip()) >= 3 and str(value).lower() in text
        )
        duplicate_penalty = 0.35 if compact in seen_text else 0.0
        source_penalty = 0.08 if source and source in seen_source else 0.0
        quality = 0.12 if item.get("document_id") or item.get("source") else 0.02
        score = base * 0.25 + semantic * 0.45 + min(0.25, entity_hits * 0.12) + min(0.4, purpose_entity_hits * 0.4) + min(0.25, parameter_hits * 0.08) + min(0.45, expansion_hits * 0.12) + quality - duplicate_penalty - source_penalty
        reasons = (
            f"base={base:.4f}", f"semantic={semantic:.4f}",
            f"entity_matches={entity_hits}", f"purpose_entity_matches={purpose_entity_hits}", f"coverage_matches={parameter_hits}",
            f"concept_matches={expansion_hits}",
            f"source_quality={quality:.2f}", f"duplicate_penalty={duplicate_penalty:.2f}",
            f"source_penalty={source_penalty:.2f}", f"final={score:.4f}",
        )
        ranked.append((item, score, reasons))
        seen_text.add(compact)
        if source:
            seen_source.add(source)
    ranked.sort(key=lambda value: value[1], reverse=True)
    return ranked[: plan.top_k]


def build_evidence_pack(
    plan: RetrievalPlan,
    ranked: list[tuple[dict[str, Any], float, tuple[str, ...]]],
    *,
    version: int = 1,
    parent_pack_ids: tuple[str, ...] = (),
    refresh_reason: str = "",
    requested_by: str = "",
) -> EvidencePack:
    items: list[EvidenceItem] = []
    covered: set[str] = set()
    for index, (item, score, reasons) in enumerate(ranked):
        text = _text(item)
        for need in plan.required_evidence_types:
            if need.lower() in text.lower():
                covered.add(need)
        chunk = str(item.get("chunk_id") or f"rank-{index}")
        source = str(item.get("document_id") or item.get("source") or "")
        entity = str((item.get("metadata") or {}).get("entity") or item.get("entity") or "")
        items.append(EvidenceItem(
            evidence_id=chunk, chunk_reference=chunk, source_reference=source,
            entity=entity, material_system=entity,
            conditions=_parameters(text), supported_claim_types=("FACT", "INFERENCE"),
            relevance_score=round(score, 6),
            source_quality="local_corpus" if source else "unknown",
            provenance_reference=str(item.get("provenance") or source or chunk),
            conflict_flag=False, content=text, rerank_reasons=reasons,
        ))
    missing = tuple(value for value in plan.required_evidence_types if value not in covered)
    return EvidencePack(
        task_id=plan.task_id, subtask_id=plan.subtask_id, items=tuple(items),
        coverage=tuple(sorted(covered)), conflicts=(), missing_information=missing,
        version=max(1, int(version)), parent_pack_ids=parent_pack_ids,
        refresh_reason=refresh_reason, requested_by=requested_by,
    )


def retrieval_metrics(pack: EvidencePack) -> dict[str, float]:
    items = pack.items
    if not items:
        return {"relevant_at_k": 0.0, "entity_match_at_k": 0.0, "evidence_coverage": 0.0, "duplicate_rate": 0.0, "source_diversity": 0.0}
    texts = [re.sub(r"\s+", "", item.content)[:160] for item in items]
    sources = {item.source_reference for item in items if item.source_reference}
    return {
        "relevant_at_k": sum(item.relevance_score > 0 for item in items) / len(items),
        "entity_match_at_k": sum(bool(item.entity) for item in items) / len(items),
        "evidence_coverage": len(pack.coverage) / max(1, len(pack.coverage) + len(pack.missing_information)),
        "duplicate_rate": 1.0 - len(set(texts)) / len(texts),
        "source_diversity": len(sources) / len(items),
    }


__all__ = ["RetrievalNeed", "RetrievalPlan", "EvidenceItem", "EvidencePack", "build_retrieval_need", "build_retrieval_plan", "build_retrieval_plans", "build_challenge_retrieval_plans", "hard_filter", "agent_aware_rerank", "build_evidence_pack", "retrieval_metrics"]
