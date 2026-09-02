"""Request-local learner lifecycle, persona prior and teaching decision.

This module does not define a second learner state or a public API contract.
It combines already available declared, observed, inferred and memory-backed
signals into bounded objects that are interpreted by Diagnosis only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class LearnerLifecycleStage(str, Enum):
    """Evidence-backed maturity of the system's understanding of a learner."""

    UNKNOWN = "UNKNOWN_LEARNER"
    INITIAL_UNDERSTANDING = "INITIAL_UNDERSTANDING"
    INITIAL_MODEL = "INITIAL_LEARNER_MODEL"
    ADAPTIVE_UNDERSTANDING = "ADAPTIVE_UNDERSTANDING"
    PERSONALIZED_TEACHING = "PERSONALIZED_TEACHING"
    LONG_TERM_COMPANION = "LONG_TERM_COMPANION"


@dataclass(frozen=True, slots=True)
class LearnerPersonaPrototype:
    """Low-confidence prior for cold start; never a fixed user label."""

    prototype_id: str
    background: Mapping[str, str]
    knowledge_priors: Mapping[str, float]
    research_priors: Mapping[str, float]
    learning_preferences: tuple[str, ...]
    teaching_constraints: tuple[str, ...]
    source_refs: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class AdaptiveDiagnosticTarget:
    """The next valuable Concept to diagnose, not a generated test item."""

    needed: bool
    target_concept: str
    reason: str
    source_refs: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class PersonalLearnerModel:
    """Request-local interpretation of the learner; not persisted as state."""

    learner_id: str
    lifecycle_stage: LearnerLifecycleStage
    persona_prior: LearnerPersonaPrototype
    declared_goals: tuple[str, ...]
    target_concepts: tuple[str, ...]
    weak_knowledge: tuple[str, ...]
    prerequisite_gaps: tuple[str, ...]
    active_misconceptions: tuple[str, ...]
    observed_record_count: int
    learning_event_count: int
    model_state_available: bool
    teaching_memory_available: bool
    diagnostic: AdaptiveDiagnosticTarget
    source_refs: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class AdaptiveTeachingDecision:
    """Diagnosis-owned teaching strategy; it cannot change scientific facts."""

    content_depth: str
    explanation_strategy: str
    representation_modes: tuple[str, ...]
    difficulty_strategy: str
    resource_modes: tuple[str, ...]
    next_focus: str
    diagnostic_needed: bool
    rationale: tuple[str, ...]
    source_refs: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class InitialTeachingProfileCandidate:
    """One bounded cold-start teaching option, never a mastery label.

    The three candidates make the initial teaching choice inspectable.  Their
    scores describe relative fit for *presentation*, not learner ability.  No
    candidate is persisted and declared information cannot become mastery.
    """

    candidate_id: str
    label: str
    content_depth: str
    explanation_strategy: str
    representation_modes: tuple[str, ...]
    fit_score: float
    selected: bool
    evidence_basis: str
    diagnostic_required: bool
    rationale: tuple[str, ...]
    source_refs: tuple[str, ...]


_STAGE_PRIORS: dict[str, float] = {
    "secondary": 0.12,
    "high_school": 0.12,
    "foundation": 0.2,
    "beginner": 0.25,
    "undergraduate": 0.32,
    "freshman": 0.28,
    "intermediate": 0.48,
    "graduate": 0.58,
    "master": 0.58,
    "advanced": 0.68,
    "phd": 0.74,
    "researcher": 0.76,
    "professional": 0.65,
}

_MAJOR_ALIASES: dict[str, str] = {
    "materials": "materials",
    "materials_sci": "materials",
    "material_science": "materials",
    "材料": "materials",
    "physics": "physics",
    "optics": "physics",
    "物理": "physics",
    "光学": "physics",
    "chemistry": "chemistry",
    "化学": "chemistry",
    "optoelectronics": "optoelectronics",
    "lighting": "optoelectronics",
    "光电": "optoelectronics",
    "照明": "optoelectronics",
}

_EXPERIENCE_BONUS: dict[str, float] = {
    "none": 0.0,
    "unknown": 0.0,
    "introductory": 0.03,
    "coursework": 0.06,
    "lab": 0.12,
    "research": 0.2,
    "industry": 0.2,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return dict(result) if isinstance(result, Mapping) else {}
    return {}


def _known_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {
        "", "unknown", "unspecified", "skip", "skipped", "跳过"
    } else text


def _canonical_stage(value: Any) -> str:
    text = _known_text(value).lower()
    aliases = {
        "本科": "undergraduate",
        "本科生": "undergraduate",
        "本科阶段": "undergraduate",
        "研究生": "graduate",
        "研究生阶段": "graduate",
        "硕士": "graduate",
        "博士": "phd",
        "科研人员": "researcher",
        "从业者": "professional",
        "行业从业者": "professional",
        "高中": "high_school",
    }
    return aliases.get(text, text)


def _canonical_major(value: Any) -> str:
    text = _known_text(value).lower()
    for token, canonical in _MAJOR_ALIASES.items():
        if token in text:
            return canonical
    return text or "unknown"


def _canonical_experience(value: Any) -> str:
    text = _known_text(value).lower()
    aliases = {
        "刚开始了解": "introductory",
        "修过相关课程": "coursework",
        "有实验经历": "lab",
        "有科研经历": "research",
        "有行业经历": "industry",
    }
    return aliases.get(text, text or "unknown")


def _declared_goals(profile: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for item in profile.get("goals") or ():
        if isinstance(item, Mapping):
            text = _known_text(item.get("text"))
        else:
            text = _known_text(item)
        if text:
            values.append(text)
    declared = _mapping(profile.get("declared_background"))
    explicit = _known_text(declared.get("learning_goal"))
    if explicit:
        values.append(explicit)
    return tuple(dict.fromkeys(values))[-5:]


def build_persona_prototype(
    declared_profile: Mapping[str, Any] | None,
    *,
    request_stage_hint: str = "",
) -> LearnerPersonaPrototype:
    """Build an evidence-weighted cold-start prior from optional declarations."""

    profile = _mapping(declared_profile)
    declared = _mapping(profile.get("declared_background"))
    stage = _canonical_stage(
        declared.get("learning_stage") or request_stage_hint
    )
    major = _canonical_major(declared.get("professional_background"))
    experience = _canonical_experience(declared.get("domain_experience"))

    source_refs: list[str] = []
    if stage:
        source_refs.append("declared:learning_stage")
    if major != "unknown":
        source_refs.append("declared:professional_background")
    if experience != "unknown":
        source_refs.append("declared:domain_experience")
    goals = _declared_goals(profile)
    if goals:
        source_refs.append("declared:learning_goal")

    base = _STAGE_PRIORS.get(stage, 0.0)
    bonus = _EXPERIENCE_BONUS.get(experience, 0.0)
    knowledge = {
        "material_foundation": base,
        "optical_foundation": base,
        "luminescence_mechanism": base,
        "led_application": base,
    }
    if major == "materials":
        knowledge["material_foundation"] += 0.18
        knowledge["luminescence_mechanism"] += 0.08
    elif major == "physics":
        knowledge["optical_foundation"] += 0.18
        knowledge["luminescence_mechanism"] += 0.12
    elif major == "chemistry":
        knowledge["material_foundation"] += 0.12
    elif major == "optoelectronics":
        knowledge["led_application"] += 0.2
        knowledge["optical_foundation"] += 0.08
    knowledge = {key: round(_clamp(value + bonus), 4) for key, value in knowledge.items()}

    research_base = _clamp(base * 0.75 + bonus)
    research = {
        "scientific_reading": research_base,
        "problem_decomposition": _clamp(research_base - 0.05),
        "scientific_reasoning": research_base,
        "experimental_understanding": _clamp(
            research_base + (0.12 if experience in {"lab", "research", "industry"} else 0.0)
        ),
    }
    research = {key: round(value, 4) for key, value in research.items()}

    preferences: list[str] = []
    declared_preference = _known_text(declared.get("representation_preference")).lower()
    if declared_preference:
        preferences.append(declared_preference)
        source_refs.append("declared:representation_preference")
    expression = _known_text(profile.get("expression")).lower()
    if expression:
        preferences.append(expression)
        source_refs.append("observed:expression_preference")
    vark = _mapping(profile.get("vark_behavior"))
    if vark:
        strongest = max(vark, key=lambda key: float(vark.get(key, 0.0) or 0.0))
        mode = {"V": "visual", "A": "conceptual", "R": "evidence", "K": "practice"}.get(
            strongest.upper()
        )
        if mode:
            preferences.append(mode)
            source_refs.append("inferred:vark_behavior")

    constraints: list[str] = []
    if not source_refs:
        constraints.append("no_persona_evidence")
    if not goals:
        constraints.append("learning_goal_unknown")
    if stage and not any(key.startswith("observed:") for key in source_refs):
        constraints.append("declared_prior_requires_observation")

    declared_confidence = _clamp(float(profile.get("confidence", 0.0) or 0.0))
    confidence = min(
        0.65,
        0.12 * len([item for item in source_refs if item.startswith("declared:")])
        + 0.08 * len([item for item in source_refs if item.startswith("observed:")])
        + 0.25 * declared_confidence,
    )
    return LearnerPersonaPrototype(
        prototype_id="evidence-weighted-domain-prior",
        background=MappingProxyType({
            "learning_stage": stage or "unknown",
            "professional_background": major,
            "domain_experience": experience,
        }),
        knowledge_priors=MappingProxyType(knowledge),
        research_priors=MappingProxyType(research),
        learning_preferences=tuple(dict.fromkeys(preferences)),
        teaching_constraints=tuple(constraints),
        source_refs=tuple(dict.fromkeys(source_refs)),
        confidence=round(confidence, 4),
    )


def _diagnostic_target(
    *,
    lifecycle_stage: LearnerLifecycleStage,
    target_concepts: tuple[str, ...],
    prerequisite_gaps: tuple[str, ...],
    model_state_available: bool,
    observed_record_count: int,
    confidence: float,
) -> AdaptiveDiagnosticTarget:
    needed = bool(
        lifecycle_stage in {
            LearnerLifecycleStage.UNKNOWN,
            LearnerLifecycleStage.INITIAL_UNDERSTANDING,
        }
        or (not model_state_available and observed_record_count == 0)
    )
    target = next(iter(prerequisite_gaps or target_concepts), "unknown")
    if not needed:
        reason = "existing observed/model evidence is sufficient for a bounded teaching decision"
    elif target != "unknown":
        reason = "assess the earliest relation-backed Concept with the highest current uncertainty"
    else:
        reason = "collect an optional learning goal before selecting a domain diagnostic"
    refs = tuple(
        item for item in (
            target if target != "unknown" else "",
            "learner:model_state" if model_state_available else "",
            "learner:observed_records" if observed_record_count else "",
        ) if item
    )
    return AdaptiveDiagnosticTarget(
        needed=needed,
        target_concept=target,
        reason=reason,
        source_refs=refs,
        confidence=round(_clamp(confidence), 4),
    )


def build_personal_learner_model(
    *,
    learner_id: str,
    persona_prior: LearnerPersonaPrototype,
    declared_profile: Mapping[str, Any] | None,
    mastery: Mapping[str, float] | None,
    weak_knowledge: tuple[str, ...],
    prerequisite_gaps: tuple[str, ...],
    target_concepts: tuple[str, ...],
    misconceptions: tuple[Mapping[str, Any], ...],
    observed_record_count: int,
    learning_event_count: int,
    model_confidence: float,
    teaching_memory_available: bool,
) -> PersonalLearnerModel:
    """Combine the prior and current evidence without persisting a new model."""

    profile = _mapping(declared_profile)
    goals = _declared_goals(profile)
    model_available = bool(mastery) or model_confidence > 0.0
    active_misconceptions = tuple(
        str(item.get("misconception_id") or item.get("topic") or "")
        for item in misconceptions
        if str(item.get("status") or "") == "ACTIVE"
        and str(item.get("misconception_id") or item.get("topic") or "")
    )

    if (
        not persona_prior.source_refs
        and not model_available
        and observed_record_count == 0
        and learning_event_count == 0
        and not teaching_memory_available
    ):
        stage = LearnerLifecycleStage.UNKNOWN
    elif not model_available and observed_record_count == 0 and learning_event_count == 0:
        stage = (
            LearnerLifecycleStage.INITIAL_MODEL
            if len(persona_prior.source_refs) >= 2
            else LearnerLifecycleStage.INITIAL_UNDERSTANDING
        )
    elif teaching_memory_available and learning_event_count >= 3:
        stage = LearnerLifecycleStage.LONG_TERM_COMPANION
    elif teaching_memory_available:
        stage = LearnerLifecycleStage.PERSONALIZED_TEACHING
    elif observed_record_count > 0 or learning_event_count > 0:
        stage = LearnerLifecycleStage.ADAPTIVE_UNDERSTANDING
    else:
        stage = LearnerLifecycleStage.INITIAL_MODEL

    source_refs = list(persona_prior.source_refs)
    if observed_record_count:
        source_refs.append("observed:answer_records")
    if learning_event_count:
        source_refs.append("observed:learning_events")
    if mastery:
        source_refs.append("inferred:mastery_model")
    if model_confidence > 0.0:
        source_refs.append("inferred:model_alignment")
    if teaching_memory_available:
        source_refs.append("derived:teaching_memory_interpretation")
    if active_misconceptions:
        source_refs.append("inferred:source_backed_misconception")

    confidence = max(
        persona_prior.confidence * 0.6,
        _clamp(model_confidence),
        0.5 if observed_record_count else 0.0,
        0.45 if learning_event_count else 0.0,
    )
    diagnostic = _diagnostic_target(
        lifecycle_stage=stage,
        target_concepts=target_concepts,
        prerequisite_gaps=prerequisite_gaps,
        model_state_available=model_available,
        observed_record_count=observed_record_count,
        confidence=confidence,
    )
    return PersonalLearnerModel(
        learner_id=learner_id,
        lifecycle_stage=stage,
        persona_prior=persona_prior,
        declared_goals=goals,
        target_concepts=target_concepts,
        weak_knowledge=weak_knowledge,
        prerequisite_gaps=prerequisite_gaps,
        active_misconceptions=active_misconceptions,
        observed_record_count=int(observed_record_count),
        learning_event_count=int(learning_event_count),
        model_state_available=model_available,
        teaching_memory_available=bool(teaching_memory_available),
        diagnostic=diagnostic,
        source_refs=tuple(dict.fromkeys(source_refs)),
        confidence=round(_clamp(confidence), 4),
    )


def build_adaptive_teaching_decision(
    model: PersonalLearnerModel,
    *,
    base_depth: str,
    recent_accuracy: float | None,
    memory_strategy: str,
    next_focus: str,
    interaction_action: str = "",
) -> AdaptiveTeachingDecision:
    """Choose presentation strategy while preserving the reviewed fact set."""

    depth = str(base_depth or "foundation").lower()
    if depth not in {"foundation", "beginner", "intermediate", "advanced"}:
        depth = "foundation"
    rationale: list[str] = []

    if model.prerequisite_gaps:
        depth = (
            "intermediate"
            if depth in {"intermediate", "advanced"}
            else "foundation"
        )
        rationale.append("relation-backed prerequisite gap limits teaching depth")
    elif model.weak_knowledge:
        depth = (
            "intermediate"
            if depth in {"intermediate", "advanced"}
            else "beginner"
        )
        rationale.append("aligned weak knowledge requires additional scaffolding")
    elif depth in {"foundation", "beginner"} and model.persona_prior.confidence > 0.0:
        priors = tuple(model.persona_prior.knowledge_priors.values())
        prior_level = sum(priors) / len(priors) if priors else 0.0
        declared_stage = model.persona_prior.background.get("learning_stage", "unknown")
        if declared_stage in {"researcher", "phd", "professional"} and prior_level >= 0.55:
            depth = "advanced"
            rationale.append("declared research background supplies a low-confidence advanced prior")
        elif prior_level >= 0.42:
            depth = "intermediate"
            rationale.append("declared background supplies an intermediate cold-start prior")

    if model.active_misconceptions:
        explanation = "contrast_with_evidence"
        rationale.append("source-backed misconception requires explicit contrast")
    elif memory_strategy in {
        "repair_with_evidence",
        "contrast_with_evidence",
        "example_then_mechanism",
        "evidence_first_mechanism",
    }:
        explanation = memory_strategy
        rationale.append("observed prior resource feedback changes presentation strategy")
    elif model.prerequisite_gaps:
        explanation = "prerequisite_scaffolding"
    elif depth == "advanced":
        explanation = "evidence_first_mechanism"
    elif depth in {"foundation", "beginner"}:
        explanation = "foundation_conceptual"
    else:
        explanation = "mechanism_with_context"

    if recent_accuracy is None:
        difficulty = "diagnose_then_maintain" if model.diagnostic.needed else "maintain"
    elif recent_accuracy < 0.45:
        difficulty = "lower"
        rationale.append("recent observed accuracy is below the supported challenge band")
    elif recent_accuracy >= 0.8 and not model.prerequisite_gaps and not model.weak_knowledge:
        difficulty = "raise"
        rationale.append("recent observed accuracy supports a bounded challenge increase")
    else:
        difficulty = "maintain"

    representations = model.persona_prior.learning_preferences or ("structured_text",)
    if explanation in {"contrast_with_evidence", "repair_with_evidence", "evidence_first_mechanism"}:
        representations = tuple(dict.fromkeys((*representations, "evidence")))
    if explanation == "example_then_mechanism":
        representations = tuple(dict.fromkeys((*representations, "worked_example", "evidence")))
    if memory_strategy == "evidence_first_mechanism" and depth in {"foundation", "beginner"}:
        depth = "intermediate"
        rationale.append("observed request for deeper analysis raises bounded content depth")
    if model.prerequisite_gaps:
        representations = tuple(dict.fromkeys((*representations, "concept_path")))

    # Explicit resource feedback is an observed, request-scoped teaching
    # control.  It changes presentation strategy only; mastery and scientific
    # claims remain untouched until a real authored assessment is submitted.
    action = str(interaction_action or "").strip().lower()
    if action in {"still_confused", "change_explanation"}:
        explanation = "repair_with_evidence"
        representations = tuple(dict.fromkeys((*representations, "worked_example")))
        rationale.append("explicit learner feedback requests a different explanation")
    elif action == "request_example":
        explanation = "example_then_mechanism"
        representations = tuple(dict.fromkeys((*representations, "worked_example", "evidence")))
        rationale.append("explicit learner feedback requests a bounded example")
    elif action == "deepen":
        depth = "intermediate" if depth in {"foundation", "beginner"} else "advanced"
        explanation = "evidence_first_mechanism"
        representations = tuple(dict.fromkeys((*representations, "evidence")))
        rationale.append("explicit learner feedback requests deeper analysis")

    if depth == "advanced":
        resources = ("evidence_reading", "research_task", "staged_assessment")
    elif depth in {"foundation", "beginner"}:
        resources = ("concept_resource", "guided_practice", "staged_assessment")
    else:
        resources = ("concept_resource", "practical_guide", "staged_assessment")

    confidence = model.confidence
    if model.persona_prior.source_refs and not model.model_state_available:
        confidence = min(confidence, model.persona_prior.confidence)
    if not rationale:
        rationale.append("use the normalized learner depth from Diagnosis")
    return AdaptiveTeachingDecision(
        content_depth=depth,
        explanation_strategy=explanation,
        representation_modes=representations,
        difficulty_strategy=difficulty,
        resource_modes=resources,
        next_focus=next_focus or model.diagnostic.target_concept,
        diagnostic_needed=model.diagnostic.needed,
        rationale=tuple(rationale),
        source_refs=model.source_refs,
        confidence=round(_clamp(confidence), 4),
    )


def build_initial_teaching_profile_candidates(
    model: PersonalLearnerModel,
    decision: AdaptiveTeachingDecision,
) -> tuple[InitialTeachingProfileCandidate, ...]:
    """Compare three teaching plans against the same learner evidence.

    This is deliberately a request-local comparison layer over the existing
    PersonalLearnerModel and Diagnosis decision.  It does not create persona
    identities or claim that a cold-start learner has demonstrated knowledge.
    """

    if not isinstance(model, PersonalLearnerModel):
        raise TypeError("model must be PersonalLearnerModel")
    if not isinstance(decision, AdaptiveTeachingDecision):
        raise TypeError("decision must be AdaptiveTeachingDecision")

    if model.observed_record_count > 0 or model.model_state_available:
        evidence_basis = "OBSERVED_AND_MODEL_INFERRED"
    elif model.persona_prior.source_refs:
        evidence_basis = "DECLARED_PRIOR"
    else:
        evidence_basis = "UNKNOWN"

    selected_id = (
        "research_evidence"
        if decision.content_depth == "advanced"
        else "mechanism_application"
        if decision.content_depth == "intermediate"
        else "foundation_scaffold"
    )
    depth_index = {
        "foundation": 0,
        "beginner": 0,
        "intermediate": 1,
        "advanced": 2,
    }
    selected_depth = depth_index.get(decision.content_depth, 0)
    research_values = tuple(model.persona_prior.research_priors.values())
    research_prior = (
        sum(research_values) / len(research_values)
        if research_values else 0.0
    )
    observed_strength = min(
        1.0,
        0.2 * model.observed_record_count
        + (0.45 if model.model_state_available else 0.0),
    )

    definitions = (
        (
            "foundation_scaffold",
            "基础搭桥",
            "foundation",
            "prerequisite_scaffolding"
            if model.prerequisite_gaps else "foundation_conceptual",
            ("structured_text", "concept_path"),
            0,
        ),
        (
            "mechanism_application",
            "机制研习",
            "intermediate",
            "mechanism_with_context",
            ("mechanism_chain", "worked_example"),
            1,
        ),
        (
            "research_evidence",
            "科研证据",
            "advanced",
            "evidence_first_mechanism",
            ("evidence", "scientific_relation"),
            2,
        ),
    )
    candidates: list[InitialTeachingProfileCandidate] = []
    for candidate_id, label, depth, strategy, modes, candidate_depth in definitions:
        proximity = 1.0 - abs(candidate_depth - selected_depth) / 2.0
        score = 0.24 + 0.26 * proximity
        rationale: list[str] = []
        if candidate_id == "foundation_scaffold":
            if model.diagnostic.needed:
                score += 0.16
                rationale.append("真实诊断尚未完成，先保留基础搭桥方案")
            if model.prerequisite_gaps or model.weak_knowledge:
                score += 0.12
                rationale.append("现有先修缺口或薄弱点需要教学支架")
        elif candidate_id == "mechanism_application":
            if model.target_concepts:
                score += 0.08
                rationale.append("当前任务已有可解释的目标 Concept")
            if decision.content_depth == "intermediate":
                score += 0.1
                rationale.append("Diagnosis 当前选择机制层解释深度")
        else:
            score += 0.12 * research_prior
            if "evidence" in decision.representation_modes:
                score += 0.08
                rationale.append("当前呈现策略要求显式证据")
            if decision.content_depth == "advanced":
                score += 0.1
                rationale.append("Diagnosis 当前选择进阶科研深度")
        if candidate_id == selected_id:
            score += 0.12
            rationale.insert(0, "与本次 Diagnosis 教学决策一致")
        if evidence_basis == "OBSERVED_AND_MODEL_INFERRED":
            score += 0.08 * observed_strength
            rationale.append("包含真实作答或对齐模型状态")
        elif evidence_basis == "DECLARED_PRIOR":
            rationale.append("当前仅使用自愿声明作为低权重先验")
        else:
            rationale.append("当前没有学习者证据，匹配结果需经真实诊断修正")
        candidates.append(InitialTeachingProfileCandidate(
            candidate_id=candidate_id,
            label=label,
            content_depth=depth,
            explanation_strategy=strategy,
            representation_modes=modes,
            fit_score=round(_clamp(score, 0.0, 0.92), 4),
            selected=candidate_id == selected_id,
            evidence_basis=evidence_basis,
            diagnostic_required=model.diagnostic.needed,
            rationale=tuple(dict.fromkeys(rationale)),
            source_refs=tuple(model.source_refs),
        ))
    return tuple(candidates)


__all__ = [
    "AdaptiveDiagnosticTarget",
    "AdaptiveTeachingDecision",
    "InitialTeachingProfileCandidate",
    "LearnerLifecycleStage",
    "LearnerPersonaPrototype",
    "PersonalLearnerModel",
    "build_adaptive_teaching_decision",
    "build_initial_teaching_profile_candidates",
    "build_persona_prototype",
    "build_personal_learner_model",
]
