"""Request-local learner intelligence view for the frozen R-03 Agent Core.

The view normalizes existing L2/profile/memory signals for Diagnosis.  It is
not a learner-state store, a public contract, or an additional model source.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
import time
from types import MappingProxyType
from typing import Any, Mapping

from dy3_polaris.l2.kp_catalog import (
    CHAPTER_LABELS,
    NEW_KP_EDGES,
    NEW_KP_KG_NODES,
    NEW_KP_LEVELS,
    NEW_KP_NAMES,
    NEW_KP_TO_CHAPTER,
    NEW_KP_TO_SECTION,
    SECTION_LABELS,
    to_new_id,
)
from dy3_polaris.l5.learner_model_alignment import (
    LearnerModelAlignment,
    align_learner_models,
)
from dy3_polaris.l5.knowledge_learning_fusion import (
    KnowledgeLearningContext,
    build_knowledge_learning_context,
    public_knowledge_learning_projection,
)
from dy3_polaris.l5.learner_foundation import (
    AdaptiveTeachingDecision,
    LearnerPersonaPrototype,
    PersonalLearnerModel,
    build_adaptive_teaching_decision,
    build_persona_prototype,
    build_personal_learner_model,
)
from dy3_polaris.l5.teaching_memory import (
    TeachingMemoryView,
    interpret_teaching_memory,
)


OBSERVED = "OBSERVED"
INFERRED = "INFERRED"
DERIVED = "DERIVED"

_GENERIC_TOPIC_TERMS = {
    "分析", "测试", "测量", "方法", "技术", "理论", "设计", "材料",
    "机理", "机制", "影响", "特征", "性能", "基础", "应用", "计算",
}


@dataclass(frozen=True, slots=True)
class LearningPathNode:
    """One catalog-backed node in a request-local learning path."""

    kp_id: str
    name: str
    stage: str
    status: str
    rationale: str


@dataclass(frozen=True, slots=True)
class LearningPath:
    """Dynamic teaching route; never a course store or public API schema."""

    learner_id: str
    target_domain: str
    current_stage: str
    milestones: tuple[LearningPathNode, ...]
    completed_nodes: tuple[str, ...]
    blocked_nodes: tuple[str, ...]
    recommended_nodes: tuple[str, ...]
    rationale: tuple[str, ...]
    confidence: float


def _normalize_topic_text(value: Any) -> str:
    text = str(value or "").lower().replace("³⁺", "3+").replace("dy(iii)", "dy3+")
    return re.sub(r"\s+", "", text)


def _topic_terms(kp_id: str) -> tuple[str, ...]:
    name = NEW_KP_NAMES.get(kp_id, "")
    chapter = CHAPTER_LABELS.get(NEW_KP_TO_CHAPTER.get(kp_id, ""), "")
    section = SECTION_LABELS.get(NEW_KP_TO_SECTION.get(kp_id, ""), "")
    values = (name, chapter, section, *NEW_KP_KG_NODES.get(kp_id, ()))
    terms: list[str] = []
    for value in values:
        raw = str(value or "").replace("³⁺", "3+")
        if raw.startswith("tf-"):
            continue
        for part in re.split(r"[\s、，,；;：:（）()\[\]/与和及]+", raw):
            normalized = _normalize_topic_text(part)
            if len(normalized) >= 2 and normalized not in _GENERIC_TOPIC_TERMS:
                terms.append(normalized)
                for suffix in _GENERIC_TOPIC_TERMS:
                    if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
                        terms.append(normalized[: -len(suffix)])
    return tuple(dict.fromkeys(terms))


def _name_terms(kp_id: str) -> tuple[str, ...]:
    raw = str(NEW_KP_NAMES.get(kp_id, "")).replace("³⁺", "3+")
    values: list[str] = []
    for part in re.split(r"[\s、，,；;：:（）()\[\]/与和及]+", raw):
        normalized = _normalize_topic_text(part)
        if len(normalized) >= 2:
            values.append(normalized)
            for suffix in _GENERIC_TOPIC_TERMS:
                if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
                    values.append(normalized[: -len(suffix)])
    return tuple(dict.fromkeys(values))


def resolve_learning_topics(*values: Any, limit: int = 5) -> tuple[str, ...]:
    """Resolve domain topics from catalog metadata, never from fixed answers.

    Matching uses all 48 catalog nodes, chapter/section labels and existing KG
    aliases.  It therefore generalizes across the represented Dy3+ material and
    healthy-lighting domain instead of routing named demo questions.
    """

    raw_text = " ".join(str(value or "") for value in values)
    semantic_text = raw_text
    # These are domain-unit/concept normalizations, not question-specific
    # routes or answer templates.
    if re.search(r"\b\d{3,5}\s*k\b", raw_text, flags=re.IGNORECASE):
        semantic_text += " 色温"
    if "发射" in raw_text:
        semantic_text += " 发光"
    elif "发光" in raw_text:
        semantic_text += " 发射"
    if "浓度" in raw_text and any(
        term in raw_text for term in ("猝灭", "降低", "下降", "减弱", "损失")
    ):
        semantic_text += " 浓度猝灭"
    text = _normalize_topic_text(semantic_text)
    if not text:
        return ()
    scored: list[tuple[float, str]] = []
    for kp_id in NEW_KP_NAMES:
        matched = [term for term in _topic_terms(kp_id) if term in text]
        if not matched:
            continue
        specific = [
            term for term in matched
            if term not in {"dy3+", "稀土", "发光", "健康", "照明"}
        ]
        if not specific and len(matched) < 2:
            continue
        score = sum(min(len(term), 8) for term in matched)
        score += 3 * len(specific)
        score += 8 * sum(1 for term in _name_terms(kp_id) if term in text)
        scored.append((float(score), kp_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return ()
    cutoff = scored[0][0] * 0.7
    return tuple(
        kp_id
        for score, kp_id in scored[: max(1, int(limit))]
        if score >= cutoff
    )


def _prerequisite_order(targets: tuple[str, ...]) -> tuple[str, ...]:
    parents: dict[str, list[str]] = {}
    for edge in NEW_KP_EDGES:
        if edge.get("rel") == "prerequisite_of":
            parents.setdefault(str(edge["dst"]), []).append(str(edge["src"]))
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(kp_id: str) -> None:
        if kp_id in visiting:
            return
        visiting.add(kp_id)
        for parent in parents.get(kp_id, ()):
            visit(parent)
        if kp_id not in ordered:
            ordered.append(kp_id)

    for target in targets:
        visit(target)
    return tuple(ordered)


def build_learning_path(
    *,
    learner_id: str,
    query: str,
    mastery: Mapping[str, float] | None = None,
    weak_kps: tuple[str, ...] = (),
    historical_exposure: tuple[str, ...] = (),
    misconceptions: tuple[Mapping[str, Any], ...] = (),
    teaching_confidence: float | None = None,
) -> LearningPath:
    """Derive a bounded route from the catalog graph and aligned learner data."""

    target_nodes = resolve_learning_topics(query)
    mastery_values = {
        to_new_id(str(kp_id)): float(value)
        for kp_id, value in dict(mastery or {}).items()
        if isinstance(value, (int, float))
    }
    weak_nodes = {to_new_id(str(item)) for item in weak_kps if str(item)}
    misconception_nodes: set[str] = set()
    for item in misconceptions:
        if str(item.get("status") or "") != "ACTIVE":
            continue
        topic = str(item.get("topic") or "")
        if topic in NEW_KP_NAMES:
            misconception_nodes.add(topic)
        else:
            misconception_nodes.update(
                resolve_learning_topics(topic, item.get("belief", ""), limit=2)
            )
    if not target_nodes:
        target_nodes = tuple(dict.fromkeys((*weak_nodes, *misconception_nodes)))[:5]

    route = _prerequisite_order(target_nodes)
    completed = tuple(
        kp_id for kp_id in route if mastery_values.get(kp_id, 0.0) >= 0.7
    )
    unresolved = tuple(kp_id for kp_id in route if kp_id not in completed)
    blocked_set = (weak_nodes | misconception_nodes) & set(route)
    blocked = tuple(kp_id for kp_id in route if kp_id in blocked_set)
    recommended = tuple(
        dict.fromkeys((*blocked, *unresolved))
    )[:6]
    exposure_nodes = {
        to_new_id(str(item)) if str(item) in NEW_KP_NAMES else str(item)
        for item in historical_exposure
    }

    milestones: list[LearningPathNode] = []
    for kp_id in route[:10]:
        if kp_id in completed:
            status = "COMPLETED"
            rationale = "aligned mastery is at or above the completion threshold"
        elif kp_id in misconception_nodes:
            status = "BLOCKED"
            rationale = "an active misconception requires verified correction"
        elif kp_id in weak_nodes:
            status = "BLOCKED"
            rationale = "aligned learner state marks this prerequisite as weak"
        elif kp_id in exposure_nodes:
            status = "RECOMMENDED"
            rationale = "previous exposure exists but does not prove mastery"
        else:
            status = "RECOMMENDED"
            rationale = "catalog prerequisite order for the current target"
        milestones.append(
            LearningPathNode(
                kp_id=kp_id,
                name=NEW_KP_NAMES.get(kp_id, kp_id),
                stage=NEW_KP_LEVELS.get(kp_id, "unknown"),
                status=status,
                rationale=rationale,
            )
        )

    first_recommended = next(
        (item for item in milestones if item.kp_id in recommended),
        None,
    )
    current_stage = (
        {"L1": "foundation", "L2": "intermediate", "L3": "advanced"}.get(
            first_recommended.stage if first_recommended else "", "unknown"
        )
    )
    chapters = tuple(
        dict.fromkeys(
            CHAPTER_LABELS.get(NEW_KP_TO_CHAPTER.get(kp_id, ""), "")
            for kp_id in target_nodes
            if CHAPTER_LABELS.get(NEW_KP_TO_CHAPTER.get(kp_id, ""), "")
        )
    )
    target_domain = chapters[0] if len(chapters) == 1 else (
        "cross-domain" if chapters else "unknown"
    )
    rationale = tuple(
        item for item in (
            "targets resolved from the current domain catalog" if target_nodes else "",
            "completion uses aligned mastery only" if completed else "no mastery-backed completion",
            "active misconceptions are blocking teaching signals" if misconception_nodes else "",
            "historical exposure does not count as mastery" if exposure_nodes else "",
        ) if item
    )
    base_confidence = 0.55 if target_nodes else 0.0
    if teaching_confidence is not None:
        base_confidence = min(base_confidence, max(0.0, float(teaching_confidence)))
    return LearningPath(
        learner_id=learner_id,
        target_domain=target_domain,
        current_stage=current_stage,
        milestones=tuple(milestones),
        completed_nodes=completed,
        blocked_nodes=blocked,
        recommended_nodes=recommended,
        rationale=rationale,
        confidence=round(base_confidence, 4),
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return a JSON-compatible copy of an internal read-only value."""
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class LearnerIntelligenceSignal:
    """One classified value with its real runtime source."""

    value: Any
    classification: str
    source_type: str
    confidence: float | None = None
    timestamp: float | None = None
    decision_eligible: bool = True


@dataclass(frozen=True, slots=True)
class LearnerIntelligenceView:
    """Read-only, request-local projection consumed only by Diagnosis."""

    learner_id: str
    facts: Mapping[str, LearnerIntelligenceSignal]
    models: Mapping[str, LearnerIntelligenceSignal]
    derived_context: Mapping[str, LearnerIntelligenceSignal]
    metadata: Mapping[str, Any]
    model_alignment: LearnerModelAlignment
    profile_projection: Mapping[str, Any]
    ability_projection: Mapping[str, Any]
    memory_projection: Mapping[str, Any]

    def value(self, section: str, key: str, default: Any = None) -> Any:
        values = getattr(self, section, {})
        signal = values.get(key) if isinstance(values, Mapping) else None
        return signal.value if isinstance(signal, LearnerIntelligenceSignal) else default

    def compatibility_projection(self, name: str) -> dict[str, Any]:
        """Copy an existing public projection without exposing the View."""
        value = getattr(self, name, {})
        projected = _thaw(value)
        return projected if isinstance(projected, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            return dict(result) if isinstance(result, Mapping) else {}
        except Exception:  # noqa: BLE001 - read-only compatibility projection
            return {}
    fields = (
        "learner_id",
        "snapshot_ts",
        "kp_mastery",
        "theta",
        "level",
        "learning_style",
        "bloom_target",
        "weak_kps",
        "confidence",
        "extras",
        "version",
    )
    return {
        field: getattr(value, field)
        for field in fields
        if hasattr(value, field)
    }


def _safe_call(target: Any, method: str, *args: Any) -> Any:
    callback = getattr(target, method, None)
    if not callable(callback):
        return None
    try:
        return callback(*args)
    except Exception:  # noqa: BLE001 - an unavailable source remains unknown
        return None


def _observed_records(profile_service: Any, learner_id: str) -> tuple[dict[str, Any], ...]:
    store = getattr(profile_service, "store", None)
    records = _safe_call(store, "get_answer_history", learner_id) or ()
    projected: list[dict[str, Any]] = []
    for item in list(records)[-20:]:
        record = _as_dict(item)
        projected.append(
            {
                key: record.get(key)
                for key in (
                    "kp_id",
                    "correct",
                    "timestamp",
                    "difficulty",
                    "question_id",
                    "response_time",
                )
                if key in record
            }
        )
    return tuple(projected)


def _normalized_stage(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"advanced", "graduate", "research", "高", "高级", "研究生"}:
        return "advanced"
    if text in {"beginner", "novice", "low", "低", "入门", "本科"}:
        return "beginner"
    if text in {"intermediate", "middle", "medium", "中", "中级"}:
        return "intermediate"
    return "unknown"


def build_learner_intelligence_view(
    input_data: Mapping[str, Any],
    deps: Any,
    *,
    learner_memory_view: Mapping[str, Any] | None = None,
    teaching_memory_view: TeachingMemoryView | None = None,
) -> LearnerIntelligenceView:
    """Read existing sources once and classify them without persisting data."""

    learner_id = str(
        input_data.get("learner_id")
        or input_data.get("student_id")
        or input_data.get("task_id")
        or "anonymous-request"
    )
    profile_service = getattr(deps, "profile_service", None)
    irt_service = getattr(deps, "irt_service", None)
    memory_service = getattr(deps, "memory_service", None)
    user_understanding_service = getattr(
        deps, "user_understanding_service", None
    )

    alignment = align_learner_models(
        learner_id,
        profile_service=profile_service,
        irt_service=irt_service,
    )
    profile_data = dict(alignment.profile_projection)
    ability_data = dict(alignment.ability_projection)
    memory = _safe_call(memory_service, "get_memory_snapshot", learner_id) or {}
    memory_data = dict(memory) if isinstance(memory, Mapping) else {}
    declared_profile = _as_dict(
        _safe_call(user_understanding_service, "get_profile", learner_id)
    )

    records = alignment.observed_records
    memory_view = (
        dict(learner_memory_view)
        if isinstance(learner_memory_view, Mapping)
        else {}
    )
    recent_tasks = tuple(str(item) for item in memory_view.get("recent_tasks") or ())
    observed_learning_events = tuple(
        item
        for item in memory_view.get("observed_learning_events") or ()
        if isinstance(item, Mapping)
    )
    inferred_learning_events = tuple(
        item
        for item in memory_view.get("inferred_learning_events") or ()
        if isinstance(item, Mapping)
    )
    derived_learning_events = tuple(
        item
        for item in memory_view.get("derived_learning_events") or ()
        if isinstance(item, Mapping)
    )
    prior_exposure_topics = tuple(
        str(item)
        for item in memory_view.get("prior_exposure_topics") or ()
        if str(item)
    )
    prior_exposure_labels = tuple(
        str(item)
        for item in memory_view.get("prior_exposure_labels") or ()
        if str(item)
    )
    historical_focus = tuple(
        str(item)
        for item in (
            memory_view.get("remaining_gap_labels")
            or memory_view.get("remaining_gaps")
            or ()
        )
        if str(item)
    )
    current_topics = tuple(
        str(item)
        for item in memory_view.get("current_topics") or ()
        if str(item)
    )
    misconceptions = tuple(
        dict(item)
        for item in memory_view.get("misconceptions") or ()
        if isinstance(item, Mapping)
    )

    profile_timestamp = alignment.profile_cache.get("timestamp")
    ability_timestamp = alignment.model_states["irt"].timestamp
    timestamps = [
        float(item)
        for item in (profile_timestamp, ability_timestamp, alignment.timestamp)
        if isinstance(item, (int, float)) and float(item) > 0
    ]

    mastery = dict(alignment.selected_mastery)
    theta = alignment.selected_theta
    theta_source = alignment.selected_theta_source
    mixed_confidence = alignment.confidence.profile_confidence

    if alignment.selected_mastery_source == "bkt_tracing_state":
        weak_kps = [
            kp_id for kp_id, value in mastery.items() if float(value) < 0.6
        ]
        weak_source = "bkt_model_state"
    else:
        weak_kps = list(profile_data.get("weak_kps") or ())
        if not weak_kps and mastery:
            weak_kps = [
                kp_id for kp_id, value in mastery.items() if float(value) < 0.6
            ]
        weak_source = (
            "profile_cache_fallback" if weak_kps else "unknown"
        )
    weak_kps = list(dict.fromkeys(str(item) for item in weak_kps if item))

    requested_stage = _normalized_stage(input_data.get("learner_level"))
    profile_stage = _normalized_stage(profile_data.get("level"))
    if theta is not None and theta_source == "irt_service":
        learning_stage = "advanced" if theta >= 1.0 else (
            "beginner" if theta < -1.0 else "intermediate"
        )
        stage_source = theta_source
    elif profile_stage != "unknown":
        learning_stage = profile_stage
        stage_source = "profile_cache_fallback"
    elif theta is not None:
        learning_stage = "advanced" if theta >= 1.0 else (
            "beginner" if theta < -1.0 else "intermediate"
        )
        stage_source = theta_source
    elif requested_stage != "unknown":
        learning_stage = requested_stage
        stage_source = "request_hint"
    else:
        learning_stage = "unknown"
        stage_source = "unknown"

    recommended_depth = learning_stage
    if weak_kps and learning_stage in {"unknown", "beginner"}:
        recommended_depth = "beginner"
    elif weak_kps and learning_stage == "advanced":
        recommended_depth = "intermediate"
    if recommended_depth == "unknown":
        recommended_depth = "foundation"
    can_advance_from_history = bool(
        memory_view.get("memory_available")
        and memory_view.get("related_to_query")
        and prior_exposure_topics
        and current_topics
        and set(current_topics).issubset(set(prior_exposure_topics))
    )
    adaptive_strategy = (
        "advance_from_prior_exposure"
        if can_advance_from_history
        else "establish_foundation"
    )
    # Prior exposure changes how Diagnosis frames the explanation, but it is
    # not assessment evidence and therefore cannot upgrade learner depth.
    # Only aligned BKT/IRT/AnswerRecord signals may justify that promotion.

    teaching_confidence = alignment.confidence.teaching_confidence
    view_confidence = float(teaching_confidence or 0.0)
    recent_records = tuple(records[-5:])
    recent_accuracy = (
        sum(1 for record in recent_records if bool(record.get("correct")))
        / len(recent_records)
        if recent_records
        else None
    )
    learning_path = build_learning_path(
        learner_id=learner_id,
        query=str(input_data.get("query") or ""),
        mastery=mastery,
        weak_kps=tuple(weak_kps),
        historical_exposure=prior_exposure_topics,
        misconceptions=misconceptions,
        teaching_confidence=teaching_confidence,
    )
    knowledge_learning_context = build_knowledge_learning_context(
        learner_id=learner_id,
        query=str(input_data.get("query") or ""),
        mastery=mastery,
        weak_kps=tuple(weak_kps),
        misconceptions=misconceptions,
        teaching_confidence=teaching_confidence,
        l3_store=getattr(deps, "l3_store", None),
    )
    concept_learning_path = knowledge_learning_context.learning_path
    teaching_memory_context = interpret_teaching_memory(
        teaching_memory_view,
        (
            *knowledge_learning_context.target_concepts,
            *concept_learning_path.prerequisite_gap,
            *(
                (concept_learning_path.next_concept,)
                if concept_learning_path.next_concept != "unknown"
                else ()
            ),
        ),
    )
    if teaching_memory_context.available:
        # Teaching Memory describes prior teaching experience, not mastery.
        # Diagnosis may adapt explanation depth, but BKT/IRT and scientific
        # content remain authoritative and untouched.
        adaptive_strategy = teaching_memory_context.strategy
        if adaptive_strategy in {
            "contrast_with_evidence", "repair_with_evidence"
        } and recommended_depth == "advanced":
            recommended_depth = "intermediate"
    mapped_next_kps = knowledge_learning_context.concept_to_kps.get(
        concept_learning_path.next_concept, ()
    )
    milestone_ids = {item.kp_id for item in learning_path.milestones}
    mapped_next_kp = next(
        (kp_id for kp_id in mapped_next_kps if kp_id in milestone_ids),
        "",
    )
    if mapped_next_kp:
        # R-05 LearningPath remains a compatibility projection, but it must not
        # contradict the R-06 relation-backed decision used by Diagnosis.
        learning_path = replace(
            learning_path,
            recommended_nodes=tuple(dict.fromkeys((
                mapped_next_kp,
                *learning_path.recommended_nodes,
            ))),
            rationale=tuple(dict.fromkeys((
                "ordered by the R-06 Concept Relation learning decision",
                *learning_path.rationale,
            ))),
        )
    if concept_learning_path.prerequisite_gap:
        # A relation-backed prerequisite gap limits teaching depth; it never
        # changes the scientific content or asserts that missing data is wrong.
        recommended_depth = (
            "intermediate" if learning_stage == "advanced" else "foundation"
        )

    persona_prototype = build_persona_prototype(
        declared_profile,
        request_stage_hint=(
            requested_stage if requested_stage != "unknown" else ""
        ),
    )
    personal_learner_model = build_personal_learner_model(
        learner_id=learner_id,
        persona_prior=persona_prototype,
        declared_profile=declared_profile,
        mastery=mastery,
        weak_knowledge=tuple(weak_kps),
        prerequisite_gaps=concept_learning_path.prerequisite_gap,
        target_concepts=knowledge_learning_context.target_concepts,
        misconceptions=misconceptions,
        observed_record_count=len(records),
        learning_event_count=(
            len(observed_learning_events)
            + len(inferred_learning_events)
            + len(derived_learning_events)
        ),
        model_confidence=view_confidence,
        teaching_memory_available=teaching_memory_context.available,
    )
    adaptive_teaching_decision = build_adaptive_teaching_decision(
        personal_learner_model,
        base_depth=recommended_depth,
        recent_accuracy=recent_accuracy,
        memory_strategy=adaptive_strategy,
        next_focus=(
            concept_learning_path.next_concept
            if concept_learning_path.next_concept != "unknown"
            else next(iter(learning_path.recommended_nodes), "")
        ),
        interaction_action=str(input_data.get("teaching_action") or ""),
    )
    recommended_depth = adaptive_teaching_decision.content_depth

    facts = {
        "declared_background": LearnerIntelligenceSignal(
            value=_freeze(declared_profile.get("declared_background") or {}),
            classification=OBSERVED,
            source_type=(
                "optional_user_declaration"
                if declared_profile.get("declared_background")
                else "unknown"
            ),
            confidence=persona_prototype.confidence or None,
        ),
        "requested_level": LearnerIntelligenceSignal(
            value=requested_stage,
            classification=OBSERVED,
            source_type="request_hint" if requested_stage != "unknown" else "unknown",
            confidence=1.0 if requested_stage != "unknown" else None,
        ),
        "known_history": LearnerIntelligenceSignal(
            value=_freeze(recent_tasks),
            classification=OBSERVED,
            source_type="learner_memory_task_refs" if recent_tasks else "unknown",
            confidence=1.0 if recent_tasks else None,
        ),
        "observed_records": LearnerIntelligenceSignal(
            value=_freeze(records),
            classification=OBSERVED,
            source_type="l2_answer_history" if records else "unknown",
            confidence=1.0 if records else None,
        ),
        "observed_learning_events": LearnerIntelligenceSignal(
            value=_freeze(observed_learning_events),
            classification=OBSERVED,
            source_type=(
                "learner_memory_observed_events"
                if observed_learning_events
                else "unknown"
            ),
            confidence=1.0 if observed_learning_events else None,
        ),
    }
    models = {
        "mastery": LearnerIntelligenceSignal(
            value=_freeze(mastery),
            classification=INFERRED,
            source_type=(
                "bkt_tracing_state"
                if alignment.selected_mastery_source == "bkt_tracing_state"
                else "profile_snapshot_bkt_projection"
                if mastery
                else "unknown"
            ),
            confidence=alignment.model_states["bkt"].confidence,
            timestamp=(
                alignment.model_states["bkt"].timestamp
                if alignment.selected_mastery_source == "bkt_tracing_state"
                else float(profile_timestamp)
                if isinstance(profile_timestamp, (int, float))
                else None
            ),
        ),
        "theta": LearnerIntelligenceSignal(
            value=theta,
            classification=INFERRED,
            source_type=theta_source,
            confidence=alignment.model_states["irt"].confidence,
            timestamp=float(ability_timestamp) if isinstance(ability_timestamp, (int, float)) else None,
        ),
        "confidence": LearnerIntelligenceSignal(
            value=mixed_confidence,
            classification=INFERRED,
            source_type="profile_snapshot_mixed_semantics" if mixed_confidence is not None else "unknown",
            confidence=None,
            timestamp=float(profile_timestamp) if isinstance(profile_timestamp, (int, float)) else None,
            decision_eligible=False,
        ),
        "historical_model_events": LearnerIntelligenceSignal(
            value=_freeze(inferred_learning_events),
            classification=INFERRED,
            source_type=(
                "learner_memory_inferred_events"
                if inferred_learning_events
                else "unknown"
            ),
            confidence=None,
            decision_eligible=False,
        ),
        "misconceptions": LearnerIntelligenceSignal(
            value=_freeze(misconceptions),
            classification=INFERRED,
            source_type=(
                "validated_learner_misconception_history"
                if misconceptions
                else "unknown"
            ),
            confidence=(
                max(float(item.get("confidence", 0.0) or 0.0) for item in misconceptions)
                if misconceptions
                else None
            ),
        ),
    }
    derived_context = {
        "weak_kps": LearnerIntelligenceSignal(
            value=_freeze(weak_kps),
            classification=DERIVED,
            source_type=weak_source if weak_kps else "unknown",
            confidence=view_confidence if weak_kps else None,
        ),
        "learning_stage": LearnerIntelligenceSignal(
            value=learning_stage,
            classification=DERIVED,
            source_type=stage_source,
            confidence=view_confidence if learning_stage != "unknown" else None,
        ),
        "recommended_depth": LearnerIntelligenceSignal(
            value=recommended_depth,
            classification=DERIVED,
            source_type=(
                "teaching_memory_diagnosis_interpretation"
                if teaching_memory_context.available
                else
                "learner_history_interpretation"
                if can_advance_from_history
                else "diagnosis_input_normalization"
            ),
            confidence=view_confidence,
        ),
        "recent_accuracy": LearnerIntelligenceSignal(
            value=recent_accuracy,
            classification=DERIVED,
            source_type="l2_answer_history" if recent_records else "unknown",
            confidence=alignment.confidence.data_confidence,
        ),
        "historical_exposure": LearnerIntelligenceSignal(
            value=_freeze(prior_exposure_labels or prior_exposure_topics),
            classification=DERIVED,
            source_type=(
                "reviewed_topic_exposure_history"
                if prior_exposure_topics
                else "unknown"
            ),
            confidence=1.0 if prior_exposure_topics else None,
        ),
        "prerequisite_focus": LearnerIntelligenceSignal(
            value=_freeze(historical_focus),
            classification=DERIVED,
            source_type=(
                "historical_learning_signal"
                if historical_focus
                else "unknown"
            ),
            confidence=1.0 if historical_focus else None,
        ),
        "adaptive_strategy": LearnerIntelligenceSignal(
            value=adaptive_strategy,
            classification=DERIVED,
            source_type=(
                "teaching_memory_diagnosis_interpretation"
                if teaching_memory_context.available
                else
                "learner_history_interpretation"
                if can_advance_from_history
                else "diagnosis_default"
            ),
            confidence=(
                teaching_memory_context.confidence
                if teaching_memory_context.available
                else 1.0 if can_advance_from_history else view_confidence
            ),
        ),
        "teaching_memory_context": LearnerIntelligenceSignal(
            value=teaching_memory_context,
            classification=DERIVED,
            source_type=(
                "r07b_teaching_memory_view"
                if teaching_memory_context.available
                else "unknown"
            ),
            confidence=(
                teaching_memory_context.confidence
                if teaching_memory_context.available
                else None
            ),
        ),
        "historical_learning_signals": LearnerIntelligenceSignal(
            value=_freeze(derived_learning_events),
            classification=DERIVED,
            source_type=(
                "learner_memory_derived_events"
                if derived_learning_events
                else "unknown"
            ),
            confidence=1.0 if derived_learning_events else None,
        ),
        "misconception_focus": LearnerIntelligenceSignal(
            value=_freeze(
                tuple(
                    {
                        "misconception_id": str(item.get("misconception_id") or ""),
                        "topic": str(item.get("topic") or ""),
                        "belief": str(item.get("belief") or ""),
                        "severity": str(item.get("severity") or ""),
                        "correction_strategy": str(item.get("correction_strategy") or ""),
                    }
                    for item in misconceptions
                    if str(item.get("status") or "") == "ACTIVE"
                )
            ),
            classification=DERIVED,
            source_type=(
                "diagnosis_misconception_interpretation"
                if any(str(item.get("status") or "") == "ACTIVE" for item in misconceptions)
                else "unknown"
            ),
            confidence=learning_path.confidence if misconceptions else None,
        ),
        "learning_path": LearnerIntelligenceSignal(
            value=learning_path,
            classification=DERIVED,
            source_type=(
                "catalog_graph_and_aligned_learner_state"
                if learning_path.milestones
                else "unknown"
            ),
            confidence=learning_path.confidence,
        ),
        "knowledge_learning_context": LearnerIntelligenceSignal(
            value=knowledge_learning_context,
            classification=DERIVED,
            source_type=(
                "r06_concept_relation_and_aligned_learner_state"
                if knowledge_learning_context.target_concepts
                else "unknown"
            ),
            confidence=concept_learning_path.confidence,
        ),
        "concept_learning_path": LearnerIntelligenceSignal(
            value=concept_learning_path,
            classification=DERIVED,
            source_type=(
                "r06_concept_relation_learning_decision"
                if concept_learning_path.next_concept != "unknown"
                else "unknown"
            ),
            confidence=concept_learning_path.confidence,
        ),
        "persona_prototype": LearnerIntelligenceSignal(
            value=persona_prototype,
            classification=DERIVED,
            source_type=(
                "evidence_weighted_cold_start_prior"
                if persona_prototype.source_refs
                else "unknown"
            ),
            confidence=persona_prototype.confidence or None,
        ),
        "personal_learner_model": LearnerIntelligenceSignal(
            value=personal_learner_model,
            classification=DERIVED,
            source_type="learner_intelligence_request_model",
            confidence=personal_learner_model.confidence,
        ),
        "adaptive_teaching_decision": LearnerIntelligenceSignal(
            value=adaptive_teaching_decision,
            classification=DERIVED,
            source_type="diagnosis_teaching_decision",
            confidence=adaptive_teaching_decision.confidence,
        ),
        "learner_lifecycle_stage": LearnerIntelligenceSignal(
            value=personal_learner_model.lifecycle_stage.value,
            classification=DERIVED,
            source_type="learner_evidence_maturity",
            confidence=personal_learner_model.confidence,
        ),
    }

    public_profile = dict(profile_data)
    if isinstance(public_profile.get("extras"), Mapping):
        extras = dict(public_profile["extras"])
        extras.pop("learner_memory", None)
        public_profile["extras"] = extras

    source_types = tuple(
        dict.fromkeys(
            signal.source_type
            for section in (facts, models, derived_context)
            for signal in section.values()
            if signal.source_type != "unknown"
        )
    )
    metadata = {
        "source_type": source_types,
        "confidence": view_confidence,
        "data_confidence": alignment.confidence.data_confidence,
        "model_confidence": dict(alignment.confidence.model_confidence),
        "teaching_confidence": alignment.confidence.teaching_confidence,
        "profile_confidence": alignment.confidence.profile_confidence,
        "alignment_status": alignment.alignment_status.value,
        "mastery_alignment_status": alignment.mastery_status.value,
        "theta_alignment_status": alignment.theta_status.value,
        "timestamp": max(timestamps) if timestamps else time.time(),
        "memory_available": bool(memory_view.get("memory_available")),
        "teaching_memory_available": teaching_memory_context.available,
        "teaching_memory_relevant_concepts": len(
            teaching_memory_context.relevant_concepts
        ),
        "adaptive_strategy": adaptive_strategy,
        "historical_signal_count": (
            len(observed_learning_events)
            + len(inferred_learning_events)
            + len(derived_learning_events)
        ),
        "active_misconception_count": sum(
            1 for item in misconceptions if str(item.get("status") or "") == "ACTIVE"
        ),
        "learning_path_available": bool(learning_path.milestones),
        "concept_learning_path_available": (
            concept_learning_path.next_concept != "unknown"
        ),
        "learner_lifecycle_stage": personal_learner_model.lifecycle_stage.value,
        "persona_prior_available": bool(persona_prototype.source_refs),
        "adaptive_diagnostic_needed": personal_learner_model.diagnostic.needed,
        "adaptive_teaching_strategy": adaptive_teaching_decision.explanation_strategy,
        "request_local": True,
    }
    return LearnerIntelligenceView(
        learner_id=learner_id,
        facts=MappingProxyType(facts),
        models=MappingProxyType(models),
        derived_context=MappingProxyType(derived_context),
        metadata=_freeze(metadata),
        model_alignment=alignment,
        profile_projection=_freeze(public_profile),
        ability_projection=_freeze(ability_data),
        memory_projection=_freeze(memory_data),
    )


def public_learner_intelligence_projection(
    view: LearnerIntelligenceView,
) -> dict[str, Any]:
    """Return evidence-labelled learner facts safe for the learner UI."""

    if not isinstance(view, LearnerIntelligenceView):
        return {}
    model = view.value("derived_context", "personal_learner_model")
    decision = view.value("derived_context", "adaptive_teaching_decision")
    knowledge = view.value("derived_context", "knowledge_learning_context")
    mastery = dict(view.value("models", "mastery", {}) or {})
    weak = list(view.value("derived_context", "weak_kps", ()) or ())
    declared = dict(view.value("facts", "declared_background", {}) or {})
    observed_count = len(view.value("facts", "observed_records", ()) or ())
    stage = str(view.metadata.get("learner_lifecycle_stage") or "UNKNOWN_LEARNER")
    projection = {
        "learner_id": view.learner_id,
        "lifecycle_stage": stage,
        "state": (
            "UNKNOWN" if stage == "UNKNOWN_LEARNER" else "EVIDENCE_BACKED"
        ),
        "declared_background": declared,
        "observed_record_count": observed_count,
        "model_state_available": bool(mastery),
        "mastery_summary": {
            "known_count": len(mastery),
            "weak_kps": weak[:8],
            "source_class": "MODEL_INFERRED" if mastery else "UNKNOWN",
        },
        "diagnostic": {
            "needed": bool(getattr(getattr(model, "diagnostic", None), "needed", True)),
            "target_concept": str(getattr(getattr(model, "diagnostic", None), "target_concept", "unknown")),
            "reason": str(getattr(getattr(model, "diagnostic", None), "reason", "insufficient learner evidence")),
        },
        "teaching_decision": {
            "content_depth": str(getattr(decision, "content_depth", "foundation")),
            "explanation_strategy": str(getattr(decision, "explanation_strategy", "baseline_explanation")),
            "representation_modes": list(getattr(decision, "representation_modes", ()) or ()),
            "difficulty_strategy": str(getattr(decision, "difficulty_strategy", "diagnose_then_maintain")),
            "next_focus": str(getattr(decision, "next_focus", "")),
            "confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
            "source_class": "DECISION",
        },
        "knowledge_context": (
            public_knowledge_learning_projection(knowledge)
            if isinstance(knowledge, KnowledgeLearningContext)
            else {}
        ),
        "source_classes": ["OBSERVED", "MODEL_INFERRED", "DERIVED", "DECISION"],
    }
    projection["report"] = build_public_learner_report(view)
    return projection


def build_public_learner_report(
    view: LearnerIntelligenceView,
) -> dict[str, Any]:
    """Build one authoritative, evidence-labelled public growth projection."""

    if not isinstance(view, LearnerIntelligenceView):
        return {}
    records = tuple(
        dict(item) for item in view.value("facts", "observed_records", ()) or ()
        if isinstance(item, Mapping)
    )
    mastery = {
        str(kp_id): float(value)
        for kp_id, value in dict(view.value("models", "mastery", {}) or {}).items()
    }
    by_kp: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        kp_id = str(record.get("kp_id") or "")
        if kp_id:
            by_kp.setdefault(kp_id, []).append(record)

    knowledge = view.value("derived_context", "knowledge_learning_context")
    knowledge_projection = (
        public_knowledge_learning_projection(knowledge)
        if isinstance(knowledge, KnowledgeLearningContext)
        else {}
    )
    path = dict(knowledge_projection.get("path") or {})
    prerequisites = tuple(str(item) for item in path.get("prerequisites_to_learn") or ())
    misconceptions = tuple(
        dict(item) for item in view.value("models", "misconceptions", ()) or ()
        if isinstance(item, Mapping)
    )

    findings: list[dict[str, Any]] = []
    for kp_id, kp_records in sorted(by_kp.items()):
        incorrect = sum(1 for item in kp_records if item.get("correct") is False)
        if len(kp_records) >= 2 and incorrect >= 2 and mastery.get(kp_id, 1.0) < 0.5:
            findings.append({
                "type": "VERIFIED_WEAKNESS",
                "reference": kp_id,
                "reason": f"{len(kp_records)} 次真实作答中有 {incorrect} 次错误",
                "evidence_count": len(kp_records),
                "source_class": "OBSERVED+MODEL_INFERRED",
            })
    for concept_id in prerequisites:
        findings.append({
            "type": "PREREQUISITE_GAP",
            "reference": concept_id,
            "reason": "R06 prerequisite relation indicates an unmet prior concept",
            "evidence_count": 1,
            "source_class": "DERIVED",
        })
    for item in misconceptions:
        status = str(item.get("status") or "")
        if status in {
            "CANDIDATE", "OBSERVED", "CONFIRMED", "ACTIVE", "INTERVENTION",
            "CHECK", "ADDRESSED", "RECURRENT", "REAPPEARED",
        }:
            findings.append({
                "type": "MISCONCEPTION",
                "reference": str(item.get("misconception_id") or item.get("topic") or "unknown"),
                "reason": f"source-backed misconception lifecycle status: {status}",
                "evidence_count": len(item.get("source_events") or ()),
                "source_class": "OBSERVED+INFERRED",
            })
    known_refs = set(mastery) | set(by_kp)
    for node in knowledge_projection.get("nodes") or ():
        kp_ids = tuple(str(item) for item in node.get("kp_ids") or ())
        if str(node.get("learner_state") or "") == "UNKNOWN" and not any(kp in known_refs for kp in kp_ids):
            findings.append({
                "type": "UNKNOWN",
                "reference": str(node.get("concept_id") or "unknown"),
                "reason": "no AnswerRecord or aligned model evidence exists",
                "evidence_count": 0,
                "source_class": "UNKNOWN",
            })

    recent = records[-3:]
    verified_weakness = any(item["type"] == "VERIFIED_WEAKNESS" for item in findings)
    prerequisite_gap = any(item["type"] == "PREREQUISITE_GAP" for item in findings)
    if not records:
        difficulty = "DIAGNOSE_FIRST"
        difficulty_reason = "没有真实作答，不能推定能力或调高难度。"
    elif verified_weakness or prerequisite_gap:
        difficulty = "LOW"
        difficulty_reason = "真实弱项或前置缺口需要先修复。"
    elif len(recent) >= 3 and all(item.get("correct") is True for item in recent):
        difficulty = "RAISE"
        difficulty_reason = "最近三次真实作答均正确，可进行有界进阶。"
    else:
        difficulty = "MAINTAIN"
        difficulty_reason = "现有作答证据不足以升降难度，保持当前层级。"

    ability = dict(view.ability_projection)
    response_count = int(ability.get("response_count", 0) or 0)
    theta = ability.get("theta") if response_count > 0 else None
    timeline: list[dict[str, Any]] = []
    for record in records:
        timeline.append({
            "event_type": "PRACTICE_RESULT",
            "timestamp": float(record.get("timestamp", 0.0) or 0.0),
            "reference": str(record.get("question_id") or record.get("kp_id") or ""),
            "outcome": "CORRECT" if record.get("correct") is True else "INCORRECT",
            "source_class": "OBSERVED",
        })
    for item in view.value("facts", "observed_learning_events", ()) or ():
        if isinstance(item, Mapping):
            timeline.append({
                "event_type": str(item.get("event_type") or item.get("type") or "LEARNING_EVENT"),
                "timestamp": float(item.get("timestamp", 0.0) or 0.0),
                "reference": str(item.get("task_id") or item.get("event_id") or ""),
                "outcome": str(item.get("outcome") or "RECORDED"),
                "source_class": "OBSERVED",
            })
    timeline.sort(key=lambda item: item["timestamp"])
    next_concept = str(path.get("next_concept") or "unknown")
    projected_nodes = list(knowledge_projection.get("nodes") or ())
    projected_edges = list(knowledge_projection.get("edges") or ())
    next_name = next(
        (
            str(item.get("name") or next_concept)
            for item in projected_nodes
            if str(item.get("concept_id") or "") == next_concept
        ),
        next_concept,
    )
    next_reason = difficulty_reason
    if next_concept != "unknown":
        if next_concept in prerequisites:
            next_reason = f"R06 先修关系显示，应先学习“{next_name}”，再进入目标概念。"
        else:
            next_reason = f"当前学习路径的下一节点是“{next_name}”。"
    return {
        "learner_id": view.learner_id,
        "status": (
            "EVIDENCE_BACKED"
            if records
            else "MODEL_ONLY"
            if mastery
            else "UNKNOWN"
        ),
        "evidence_sufficiency": {
            "answer_record_count": len(records),
            "modelled_kp_count": len(mastery),
            "irt_response_count": response_count,
            "source_class": (
                "OBSERVED+MODEL_INFERRED"
                if records and mastery
                else "OBSERVED"
                if records
                else "MODEL_INFERRED"
                if mastery
                else "UNKNOWN"
            ),
        },
        "ability": {
            "theta": theta,
            "standard_error": ability.get("se") if response_count > 0 else None,
            "status": "MODEL_INFERRED" if response_count > 0 else "UNKNOWN",
        },
        "findings": findings,
        "difficulty_decision": {
            "decision": difficulty,
            "reason": difficulty_reason,
            "source_class": "DECISION",
        },
        "learning_path": {
            **path,
            "nodes": projected_nodes,
            "edges": projected_edges,
            "source_class": "DERIVED_FROM_R06_RELATIONS",
        },
        "growth_timeline": timeline[-40:],
        "next_action": {
            "type": "PRACTICE" if difficulty == "DIAGNOSE_FIRST" else "LEARN_CONCEPT",
            "target": next_concept,
            "reason": next_reason,
            "source_class": "DECISION",
        },
        "updated_at": max((item["timestamp"] for item in timeline), default=float(view.metadata.get("timestamp", 0.0) or 0.0)),
    }


__all__ = [
    "DERIVED",
    "INFERRED",
    "OBSERVED",
    "LearnerIntelligenceSignal",
    "LearnerIntelligenceView",
    "LearnerPersonaPrototype",
    "PersonalLearnerModel",
    "AdaptiveTeachingDecision",
    "LearningPath",
    "LearningPathNode",
    "KnowledgeLearningContext",
    "build_learning_path",
    "build_public_learner_report",
    "build_learner_intelligence_view",
    "public_learner_intelligence_projection",
    "resolve_learning_topics",
]
