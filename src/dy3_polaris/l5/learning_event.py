"""Private request-level record of one completed teaching process.

This contract is deliberately distinct from task-state events, collaboration
traces, and the persisted ``agent_memory.LearningEvent`` history signals.  It
summarizes educational meaning from existing runtime facts without storing raw
prompts, generated answers, chain-of-thought, or a second learner state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import time
from typing import Any, Mapping

from dy3_polaris.l5.knowledge_learning_fusion import KnowledgeLearningContext


class TeachingActionType(str, Enum):
    EXPLANATION = "explanation"
    MISCONCEPTION_CORRECTION = "misconception_correction"
    CONCEPT_INTRODUCTION = "concept_introduction"
    PREREQUISITE_REPAIR = "prerequisite_repair"
    EVIDENCE_BASED_LEARNING = "evidence_based_learning"


@dataclass(frozen=True, slots=True)
class KnowledgeContextReference:
    source: str
    target_concepts: tuple[str, ...]
    relation_refs: tuple[str, ...]
    evidence_candidate_refs: tuple[str, ...]
    next_concept: str


@dataclass(frozen=True, slots=True)
class BeforeTeachingState:
    """Reference to the already-built request view, not a learner-state copy."""

    source_ref: str
    learner_level: str
    mastery_projection: Mapping[str, Any]
    misconception_state: tuple[str, ...]
    learning_gap: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TeachingAction:
    action_type: TeachingActionType
    producer: str
    status: str
    concept_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    summary: str


@dataclass(frozen=True, slots=True)
class AgentContributionSummary:
    agent_id: str
    contribution_id: str
    role_summary: str
    status: str
    claim_count: int
    evidence_ref_count: int
    uncertainty_count: int


@dataclass(frozen=True, slots=True)
class ReviewLearningSummary:
    producer: str
    status: str
    verdict: str
    real_reviewer_executed: bool
    challenge_types: tuple[str, ...]
    knowledge_gaps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GuidanceLearningSummary:
    decision_type: str
    next_action: str
    recommended_topics: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class LearningOutcome:
    concept_exposed: tuple[str, ...]
    misconception_addressed: tuple[str, ...]
    confidence_change: float | None
    next_learning_target: str
    completion_eligible: bool


@dataclass(frozen=True, slots=True)
class TeachingLearningEvent:
    """One private educational-process fact assembled after Guidance."""

    event_id: str
    learner_id: str
    task_id: str
    timestamp: float
    learning_goal: str
    task_mode: str
    related_concepts: tuple[str, ...]
    knowledge_context: KnowledgeContextReference
    before_state: BeforeTeachingState
    teaching_process: tuple[TeachingAction, ...]
    agent_contributions: tuple[AgentContributionSummary, ...]
    review: ReviewLearningSummary
    guidance: GuidanceLearningSummary
    outcome: LearningOutcome


_ROLE_SUMMARIES = {
    "agent.learning.diagnosis": "interpreted learner state for teaching",
    "agent.knowledge.generation": "produced the scientific explanation and evidence references",
    "agent.quality.review": "reviewed scientific quality and evidence limits",
    "agent.guidance.decision": "selected the next teaching action",
}


def _relation_refs(context: KnowledgeLearningContext) -> tuple[str, ...]:
    prefixes = ("prerequisite:", "selected_relation:")
    return tuple(
        item.split(":", 1)[1]
        for item in context.trace
        if item.startswith(prefixes) and ":" in item
    )


def _recommended_topics(decision: Any) -> tuple[str, ...]:
    topics: list[str] = []
    for item in getattr(decision, "recommended_path", ()) or ():
        if isinstance(item, Mapping):
            topic = str(item.get("topic") or "").strip()
            if topic:
                topics.append(topic)
    return tuple(dict.fromkeys(topics))


def _bounded_reference(value: Any) -> str:
    """Accept identifier-like provenance only, never sentence-like content."""

    reference = str(value or "").strip()
    if (
        not reference
        or len(reference) > 96
        or any(mark in reference for mark in ("\n", "\r", "。", "！", "？"))
    ):
        return ""
    return reference


def _agent_summaries(context: Any) -> tuple[AgentContributionSummary, ...]:
    values: list[AgentContributionSummary] = []
    for contribution in getattr(context, "contributions", ()) or ():
        agent_id = str(getattr(contribution, "agent_id", "") or "")
        values.append(AgentContributionSummary(
            agent_id=agent_id,
            contribution_id=str(
                getattr(contribution, "contribution_id", "") or ""
            ),
            role_summary=_ROLE_SUMMARIES.get(
                agent_id, "contributed a bounded runtime result"
            ),
            status=str(getattr(contribution, "status", "") or ""),
            claim_count=len(getattr(contribution, "claims", ()) or ()),
            evidence_ref_count=len(
                getattr(contribution, "evidence_refs", ()) or ()
            ),
            uncertainty_count=len(
                getattr(contribution, "uncertainty", ()) or ()
            ),
        ))
    return tuple(values)


def build_teaching_learning_event(
    *,
    context: Any,
    final_result: Any,
    review_candidate: Any,
    knowledge_learning_context: KnowledgeLearningContext,
    learner_view: Any,
) -> TeachingLearningEvent:
    """Assemble a bounded event only from facts already produced in runtime."""

    task_id = str(getattr(final_result, "task_id", "") or "")
    learner_id = str(
        getattr(learner_view, "learner_id", "")
        or getattr(context, "learner_context", {}).get("learner_id")
        or ""
    )
    decision = getattr(final_result, "decision", None)
    path = knowledge_learning_context.learning_path
    related_concepts = tuple(dict.fromkeys((
        *knowledge_learning_context.target_concepts,
        *path.prerequisite_gap,
        *(
            (path.next_concept,)
            if path.next_concept and path.next_concept != "unknown"
            else ()
        ),
    )))
    evidence_refs = tuple(dict.fromkeys(
        reference
        for item in getattr(final_result, "provenance_refs", ()) or ()
        if (reference := _bounded_reference(item))
    ))
    completion_eligible = bool(
        getattr(final_result, "completion_eligibility", False)
    )
    answer_delivered = bool(getattr(final_result, "answer", ""))
    process: list[TeachingAction] = []
    if answer_delivered:
        process.append(TeachingAction(
            action_type=TeachingActionType.EXPLANATION,
            producer="agent.knowledge.generation",
            status="delivered",
            concept_refs=knowledge_learning_context.target_concepts,
            evidence_refs=evidence_refs,
            summary="a reviewed explanation was delivered",
        ))
    if related_concepts and answer_delivered:
        process.append(TeachingAction(
            action_type=TeachingActionType.CONCEPT_INTRODUCTION,
            producer="agent.learning.diagnosis",
            status="delivered",
            concept_refs=related_concepts,
            evidence_refs=(),
            summary="mapped concepts entered the teaching context",
        ))
    if path.prerequisite_gap:
        process.append(TeachingAction(
            action_type=TeachingActionType.PREREQUISITE_REPAIR,
            producer="agent.guidance.decision",
            status="planned",
            concept_refs=path.prerequisite_gap,
            evidence_refs=(),
            summary="an unmet Concept Relation prerequisite was selected for repair",
        ))
    misconception_planned = bool(
        knowledge_learning_context.active_misconception_concepts
        and "misconception_relation" in path.reason
    )
    if misconception_planned:
        process.append(TeachingAction(
            action_type=TeachingActionType.MISCONCEPTION_CORRECTION,
            producer="agent.guidance.decision",
            status="planned",
            concept_refs=knowledge_learning_context.active_misconception_concepts,
            evidence_refs=evidence_refs,
            summary="an active misconception changed the relation-backed learning target",
        ))
    if evidence_refs and answer_delivered:
        process.append(TeachingAction(
            action_type=TeachingActionType.EVIDENCE_BASED_LEARNING,
            producer="agent.quality.review",
            status="delivered" if completion_eligible else "limited",
            concept_refs=knowledge_learning_context.target_concepts,
            evidence_refs=evidence_refs,
            summary="the teaching result retained source references",
        ))

    challenge_types = tuple(dict.fromkeys(
        str(getattr(getattr(item, "challenge_type", None), "value", "") or "")
        for item in getattr(context, "challenges", ()) or ()
        if str(getattr(getattr(item, "challenge_type", None), "value", "") or "")
    ))
    review = ReviewLearningSummary(
        producer=str(getattr(review_candidate, "producer", "") or ""),
        status=str(getattr(review_candidate, "raw_status", "") or ""),
        verdict=str(getattr(review_candidate, "raw_verdict", "") or ""),
        real_reviewer_executed=bool(
            getattr(review_candidate, "real_reviewer_executed", False)
        ),
        challenge_types=challenge_types,
        knowledge_gaps=tuple(
            str(item)
            for item in getattr(final_result, "knowledge_gaps", ()) or ()
            if str(item)
        ),
    )
    guidance = GuidanceLearningSummary(
        decision_type=str(
            getattr(getattr(decision, "decision_type", None), "value", "") or ""
        ),
        next_action=str(getattr(decision, "next_action", "") or ""),
        recommended_topics=_recommended_topics(decision),
        confidence=float(getattr(decision, "confidence", 0.0) or 0.0),
    )
    learner_level = str(
        learner_view.value("derived_context", "learning_stage", "unknown")
        if callable(getattr(learner_view, "value", None))
        else "unknown"
    )
    before_state = BeforeTeachingState(
        source_ref="LearnerIntelligenceView/KnowledgeLearningContext@request",
        learner_level=learner_level,
        mastery_projection=knowledge_learning_context.concept_mastery,
        misconception_state=(
            knowledge_learning_context.active_misconception_concepts
        ),
        learning_gap=path.prerequisite_gap,
    )
    outcome = LearningOutcome(
        concept_exposed=(related_concepts if answer_delivered else ()),
        misconception_addressed=(
            knowledge_learning_context.active_misconception_concepts
            if misconception_planned
            else ()
        ),
        confidence_change=None,
        next_learning_target=(
            path.next_concept if path.next_concept != "unknown" else ""
        ),
        completion_eligible=completion_eligible,
    )
    learning_goal = str(getattr(getattr(context, "task_plan", None), "goal", "") or "")
    task_mode = str(
        getattr(getattr(final_result, "task_mode", None), "value", "") or ""
    )
    seed = "|".join((
        learner_id,
        task_id,
        str(getattr(final_result, "answer_identity", "") or ""),
        task_mode,
    ))
    return TeachingLearningEvent(
        event_id=f"teaching-event-{sha256(seed.encode('utf-8')).hexdigest()[:20]}",
        learner_id=learner_id,
        task_id=task_id,
        timestamp=time.time(),
        learning_goal=learning_goal,
        task_mode=task_mode,
        related_concepts=related_concepts,
        knowledge_context=KnowledgeContextReference(
            source="R06D KnowledgeLearningContext",
            target_concepts=knowledge_learning_context.target_concepts,
            relation_refs=_relation_refs(knowledge_learning_context),
            evidence_candidate_refs=(
                knowledge_learning_context.evidence_available_concepts
            ),
            next_concept=(
                path.next_concept if path.next_concept != "unknown" else ""
            ),
        ),
        before_state=before_state,
        teaching_process=tuple(process),
        agent_contributions=_agent_summaries(context),
        review=review,
        guidance=guidance,
        outcome=outcome,
    )


__all__ = [
    "AgentContributionSummary",
    "BeforeTeachingState",
    "GuidanceLearningSummary",
    "KnowledgeContextReference",
    "LearningOutcome",
    "ReviewLearningSummary",
    "TeachingAction",
    "TeachingActionType",
    "TeachingLearningEvent",
    "build_teaching_learning_event",
]
