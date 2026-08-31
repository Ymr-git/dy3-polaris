"""Private teaching-experience memory derived from R-07A learning events.

The memory stores bounded teaching facts, never raw chat, prompts, generated
answers, chain-of-thought, or a second mastery model.  It reuses the existing
learner profile JSON envelope and can influence Agents only after a
``TeachingMemoryView`` has been interpreted by ``LearnerIntelligenceView`` and
Diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Iterable, Mapping

from dy3_polaris.l5.learning_event import TeachingLearningEvent
from dy3_polaris.l5.learning_resources import (
    ResourceInteractionAction,
    ResourceInteractionEvent,
)


logger = logging.getLogger("dy3_polaris.l5.teaching_memory")

_VERSION = "r07b-v1"
_CONCEPT_LIMIT = 160
_MISCONCEPTION_LIMIT = 80
_STRATEGY_LIMIT = 160
_EXPERIENCE_LIMIT = 120
_RESOURCE_INTERACTION_LIMIT = 120
_PRACTICE_VALIDATION_LIMIT = 160


class MisconceptionLifecycle(str, Enum):
    CANDIDATE = "CANDIDATE"
    OBSERVED = "OBSERVED"
    CONFIRMED = "CONFIRMED"
    ACTIVE = "ACTIVE"
    INTERVENTION = "INTERVENTION"
    CHECK = "CHECK"
    ADDRESSED = "ADDRESSED"
    RESOLVED = "RESOLVED"
    RECURRENT = "RECURRENT"
    REAPPEARED = "REAPPEARED"


class TeachingEffectStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    TRIED = "TRIED"
    POSITIVE_CANDIDATE = "POSITIVE_CANDIDATE"
    NEGATIVE_CANDIDATE = "NEGATIVE_CANDIDATE"
    VALIDATED_POSITIVE = "VALIDATED_POSITIVE"
    VALIDATED_NEGATIVE = "VALIDATED_NEGATIVE"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class ConceptLearningMemory:
    """Longitudinal teaching experience for one learner and Concept.

    ``teaching_effect`` describes the observed teaching process.  It is not a
    mastery value and cannot update BKT or IRT.
    """

    learner_id: str
    concept_id: str
    learning_attempts: int
    first_seen: float
    last_seen: float
    learning_event_ids: tuple[str, ...]
    teaching_effect: str
    confidence: float


@dataclass(frozen=True, slots=True)
class TeachingMisconceptionMemory:
    """Source-backed misconception lifecycle without storing belief text."""

    learner_id: str
    misconception_id: str
    concept_ids: tuple[str, ...]
    status: MisconceptionLifecycle
    source_event_ids: tuple[str, ...]
    intervention_event_ids: tuple[str, ...]
    confidence: float
    last_seen: float
    check_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TeachingStrategyMemory:
    """Recorded delivery/effect of a teaching action for one Concept."""

    learner_id: str
    concept_id: str
    strategy: str
    attempts: int
    learning_event_ids: tuple[str, ...]
    effect: str
    confidence: float
    last_used: float
    effect_status: TeachingEffectStatus = TeachingEffectStatus.UNKNOWN
    validated_outcomes: int = 0


@dataclass(frozen=True, slots=True)
class PracticeValidationEvent:
    """A server-correlated authored-question outcome, not a mastery model."""

    event_id: str
    learner_id: str
    task_id: str
    resource_id: str
    question_id: str
    kp_id: str
    concept_ids: tuple[str, ...]
    strategy: str
    correct: bool
    timestamp: float


@dataclass(frozen=True, slots=True)
class LearningExperienceMemory:
    """One bounded teaching-process reference, not a transcript."""

    experience_id: str
    learner_id: str
    event_id: str
    task_id: str
    concept_ids: tuple[str, ...]
    strategies: tuple[str, ...]
    outcome: str
    next_learning_target: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class TeachingMemoryView:
    """Read-only learner-scoped view; Agents must not consume it directly."""

    learner_id: str
    concept_learning: tuple[ConceptLearningMemory, ...] = ()
    misconceptions: tuple[TeachingMisconceptionMemory, ...] = ()
    strategies: tuple[TeachingStrategyMemory, ...] = ()
    experiences: tuple[LearningExperienceMemory, ...] = ()
    updated_at: float = 0.0

    @property
    def available(self) -> bool:
        return bool(
            self.concept_learning
            or self.misconceptions
            or self.strategies
            or self.experiences
        )


@dataclass(frozen=True, slots=True)
class TeachingMemoryInterpretation:
    """Diagnosis-eligible derivation from a TeachingMemoryView."""

    available: bool
    relevant_concepts: tuple[str, ...]
    prior_attempts: int
    strategy: str
    misconception_statuses: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    confidence: float


def _bounded_tuple(values: Iterable[Any], *, limit: int = 48) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(item).strip() for item in values if str(item).strip()
    ))[:limit]


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _enum_status(value: Any) -> MisconceptionLifecycle:
    try:
        return MisconceptionLifecycle(str(value or "ACTIVE"))
    except ValueError:
        return MisconceptionLifecycle.ACTIVE


def _effect_status(value: Any) -> TeachingEffectStatus:
    try:
        return TeachingEffectStatus(str(value or "UNKNOWN"))
    except ValueError:
        return TeachingEffectStatus.UNKNOWN


def _concept_from_mapping(value: Mapping[str, Any]) -> ConceptLearningMemory:
    return ConceptLearningMemory(
        learner_id=str(value.get("learner_id") or ""),
        concept_id=str(value.get("concept_id") or ""),
        learning_attempts=max(0, int(value.get("learning_attempts", 0) or 0)),
        first_seen=float(value.get("first_seen", 0.0) or 0.0),
        last_seen=float(value.get("last_seen", 0.0) or 0.0),
        learning_event_ids=_bounded_tuple(value.get("learning_event_ids") or ()),
        teaching_effect=str(value.get("teaching_effect") or "unknown"),
        confidence=_confidence(value.get("confidence")),
    )


def _misconception_from_mapping(
    value: Mapping[str, Any],
) -> TeachingMisconceptionMemory:
    return TeachingMisconceptionMemory(
        learner_id=str(value.get("learner_id") or ""),
        misconception_id=str(value.get("misconception_id") or ""),
        concept_ids=_bounded_tuple(value.get("concept_ids") or ()),
        status=_enum_status(value.get("status")),
        source_event_ids=_bounded_tuple(value.get("source_event_ids") or ()),
        intervention_event_ids=_bounded_tuple(
            value.get("intervention_event_ids") or ()
        ),
        confidence=_confidence(value.get("confidence")),
        last_seen=float(value.get("last_seen", 0.0) or 0.0),
        check_event_ids=_bounded_tuple(value.get("check_event_ids") or ()),
    )


def _strategy_from_mapping(value: Mapping[str, Any]) -> TeachingStrategyMemory:
    return TeachingStrategyMemory(
        learner_id=str(value.get("learner_id") or ""),
        concept_id=str(value.get("concept_id") or ""),
        strategy=str(value.get("strategy") or ""),
        attempts=max(0, int(value.get("attempts", 0) or 0)),
        learning_event_ids=_bounded_tuple(value.get("learning_event_ids") or ()),
        effect=str(value.get("effect") or "unknown"),
        confidence=_confidence(value.get("confidence")),
        last_used=float(value.get("last_used", 0.0) or 0.0),
        effect_status=_effect_status(value.get("effect_status")),
        validated_outcomes=max(0, int(value.get("validated_outcomes", 0) or 0)),
    )


def _experience_from_mapping(value: Mapping[str, Any]) -> LearningExperienceMemory:
    return LearningExperienceMemory(
        experience_id=str(value.get("experience_id") or ""),
        learner_id=str(value.get("learner_id") or ""),
        event_id=str(value.get("event_id") or ""),
        task_id=str(value.get("task_id") or ""),
        concept_ids=_bounded_tuple(value.get("concept_ids") or ()),
        strategies=_bounded_tuple(value.get("strategies") or ()),
        outcome=str(value.get("outcome") or "unknown"),
        next_learning_target=str(value.get("next_learning_target") or ""),
        timestamp=float(value.get("timestamp", 0.0) or 0.0),
    )


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, ConceptLearningMemory):
        return {
            "learner_id": value.learner_id,
            "concept_id": value.concept_id,
            "learning_attempts": value.learning_attempts,
            "first_seen": value.first_seen,
            "last_seen": value.last_seen,
            "learning_event_ids": list(value.learning_event_ids),
            "teaching_effect": value.teaching_effect,
            "confidence": value.confidence,
        }
    if isinstance(value, TeachingMisconceptionMemory):
        return {
            "learner_id": value.learner_id,
            "misconception_id": value.misconception_id,
            "concept_ids": list(value.concept_ids),
            "status": value.status.value,
            "source_event_ids": list(value.source_event_ids),
            "intervention_event_ids": list(value.intervention_event_ids),
            "confidence": value.confidence,
            "last_seen": value.last_seen,
            "check_event_ids": list(value.check_event_ids),
        }
    if isinstance(value, TeachingStrategyMemory):
        return {
            "learner_id": value.learner_id,
            "concept_id": value.concept_id,
            "strategy": value.strategy,
            "attempts": value.attempts,
            "learning_event_ids": list(value.learning_event_ids),
            "effect": value.effect,
            "confidence": value.confidence,
            "last_used": value.last_used,
            "effect_status": value.effect_status.value,
            "validated_outcomes": value.validated_outcomes,
        }
    if isinstance(value, LearningExperienceMemory):
        return {
            "experience_id": value.experience_id,
            "learner_id": value.learner_id,
            "event_id": value.event_id,
            "task_id": value.task_id,
            "concept_ids": list(value.concept_ids),
            "strategies": list(value.strategies),
            "outcome": value.outcome,
            "next_learning_target": value.next_learning_target,
            "timestamp": value.timestamp,
        }
    raise TypeError(f"unsupported teaching memory type: {type(value)!r}")


def _load_profile(profile_service: Any, learner_id: str) -> Any | None:
    if profile_service is None or not learner_id:
        return None
    try:
        return profile_service.get_profile_snapshot(str(learner_id))
    except Exception as exc:  # noqa: BLE001 - optional private compatibility path
        logger.warning("Teaching Memory profile read failed %s: %s", learner_id, exc)
        return None


def load_teaching_memory_view(
    profile_service: Any,
    learner_id: str,
) -> TeachingMemoryView:
    """Load a learner-scoped read-only projection from the existing profile."""

    learner_id = str(learner_id or "")
    profile = _load_profile(profile_service, learner_id)
    extras = dict(getattr(profile, "extras", {}) or {}) if profile is not None else {}
    learner_memory = extras.get("learner_memory")
    raw = (
        learner_memory.get("teaching_memory")
        if isinstance(learner_memory, Mapping)
        else None
    )
    raw = raw if isinstance(raw, Mapping) else {}

    def matching(values: Any, factory: Any) -> tuple[Any, ...]:
        parsed: list[Any] = []
        for item in values or ():
            if not isinstance(item, Mapping):
                continue
            try:
                value = factory(item)
            except (TypeError, ValueError):
                continue
            if value.learner_id == learner_id:
                parsed.append(value)
        return tuple(parsed)

    return TeachingMemoryView(
        learner_id=learner_id,
        concept_learning=matching(raw.get("concept_learning"), _concept_from_mapping),
        misconceptions=matching(raw.get("misconceptions"), _misconception_from_mapping),
        strategies=matching(raw.get("strategies"), _strategy_from_mapping),
        experiences=matching(raw.get("experiences"), _experience_from_mapping),
        updated_at=float(raw.get("updated_at", 0.0) or 0.0),
    )


def interpret_teaching_memory(
    view: TeachingMemoryView | None,
    concept_ids: Iterable[str],
) -> TeachingMemoryInterpretation:
    """Derive a bounded teaching signal for Diagnosis, never an Agent command."""

    if not isinstance(view, TeachingMemoryView) or not view.available:
        return TeachingMemoryInterpretation(
            available=False,
            relevant_concepts=(),
            prior_attempts=0,
            strategy="baseline_explanation",
            misconception_statuses=(),
            source_event_ids=(),
            confidence=0.0,
        )
    targets = set(_bounded_tuple(concept_ids))
    concept_memories = tuple(
        item for item in view.concept_learning if item.concept_id in targets
    )
    strategy_memories = tuple(
        item for item in view.strategies if item.concept_id in targets
    )
    misconception_memories = tuple(
        item for item in view.misconceptions if targets.intersection(item.concept_ids)
    )
    relevant = _bounded_tuple((
        *(item.concept_id for item in concept_memories),
        *(item.concept_id for item in strategy_memories),
        *(concept for item in misconception_memories for concept in item.concept_ids),
    ))
    if not relevant:
        return TeachingMemoryInterpretation(
            available=False,
            relevant_concepts=(),
            prior_attempts=0,
            strategy="baseline_explanation",
            misconception_statuses=(),
            source_event_ids=(),
            confidence=0.0,
        )
    statuses = _bounded_tuple(item.status.value for item in misconception_memories)
    effects = {item.teaching_effect for item in concept_memories}
    effects.update(item.effect for item in strategy_memories)
    if any(item.effect == "requested_example" for item in strategy_memories):
        strategy = "example_then_mechanism"
    elif any(
        item.effect == "requested_deeper_explanation"
        for item in strategy_memories
    ):
        strategy = "evidence_first_mechanism"
    elif statuses and any(
        status in {
            MisconceptionLifecycle.CANDIDATE.value,
            MisconceptionLifecycle.OBSERVED.value,
            MisconceptionLifecycle.CONFIRMED.value,
            MisconceptionLifecycle.ACTIVE.value,
            MisconceptionLifecycle.INTERVENTION.value,
            MisconceptionLifecycle.CHECK.value,
            MisconceptionLifecycle.ADDRESSED.value,
            MisconceptionLifecycle.RECURRENT.value,
            MisconceptionLifecycle.REAPPEARED.value,
        }
        for status in statuses
    ):
        strategy = "contrast_with_evidence"
    elif effects.intersection({"limited", "delivery_incomplete"}):
        strategy = "repair_with_evidence"
    else:
        strategy = "build_on_prior_exposure"
    event_ids = _bounded_tuple((
        *(event for item in concept_memories for event in item.learning_event_ids),
        *(event for item in strategy_memories for event in item.learning_event_ids),
        *(event for item in misconception_memories for event in item.source_event_ids),
        *(event for item in misconception_memories for event in item.intervention_event_ids),
    ))
    confidences = [
        item.confidence
        for item in (*concept_memories, *strategy_memories, *misconception_memories)
        if item.confidence > 0
    ]
    return TeachingMemoryInterpretation(
        available=True,
        relevant_concepts=relevant,
        prior_attempts=sum(item.learning_attempts for item in concept_memories),
        strategy=strategy,
        misconception_statuses=statuses,
        source_event_ids=event_ids,
        confidence=min(confidences) if confidences else 0.0,
    )


def _save_profile(profile_service: Any, profile: Any) -> bool:
    learner_id = str(getattr(profile, "learner_id", "") or "")
    if profile_service is None or not learner_id:
        return False
    updates = {"extras": dict(getattr(profile, "extras", {}) or {})}
    try:
        profile_service.apply_update(
            learner_id,
            updates=updates,
            expected_version=getattr(profile, "version", None),
        )
        return True
    except Exception:  # noqa: BLE001 - optimistic-lock retry once
        try:
            latest = profile_service.get_profile_snapshot(learner_id)
            if latest is None:
                return False
            profile_service.apply_update(
                learner_id,
                updates=updates,
                expected_version=getattr(latest, "version", None),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Teaching Memory profile write failed %s: %s", learner_id, exc)
            return False


def commit_resource_interaction(
    profile_service: Any,
    event: ResourceInteractionEvent,
) -> bool:
    """Persist an explicit resource-use fact without changing mastery.

    The event is retained as OBSERVED teaching feedback.  Only Diagnosis can
    later interpret its bounded strategy effect through TeachingMemoryView;
    this function never writes AnswerRecord, BKT, IRT, or LearnerSnapshot
    mastery fields.
    """

    if (
        profile_service is None
        or not isinstance(event, ResourceInteractionEvent)
        or not event.learner_id
        or not event.task_id
        or not event.resource_id
        or not event.event_id
    ):
        return False
    profile = _load_profile(profile_service, event.learner_id)
    if profile is None:
        return False
    extras = dict(getattr(profile, "extras", {}) or {})
    learner_memory = dict(extras.get("learner_memory") or {})
    raw = dict(learner_memory.get("teaching_memory") or {})
    interactions = [
        dict(item)
        for item in raw.get("resource_interactions") or ()
        if isinstance(item, Mapping)
    ]
    if any(str(item.get("event_id") or "") == event.event_id for item in interactions):
        return True

    effect = {
        ResourceInteractionAction.UNDERSTOOD: "self_reported_understood",
        ResourceInteractionAction.STILL_CONFUSED: "delivery_incomplete",
        ResourceInteractionAction.CHANGE_EXPLANATION: "delivery_incomplete",
        ResourceInteractionAction.REQUEST_EXAMPLE: "requested_example",
        ResourceInteractionAction.DEEPEN: "requested_deeper_explanation",
        ResourceInteractionAction.ASK_FOLLOW_UP: "guided_follow_up_started",
        ResourceInteractionAction.NEXT_CONCEPT: "requested_next_concept",
        ResourceInteractionAction.START_PRACTICE: "practice_started_not_scored",
        ResourceInteractionAction.OPEN: "resource_opened",
    }[event.action]
    effect_status = {
        ResourceInteractionAction.UNDERSTOOD: TeachingEffectStatus.POSITIVE_CANDIDATE,
        ResourceInteractionAction.STILL_CONFUSED: TeachingEffectStatus.NEGATIVE_CANDIDATE,
        ResourceInteractionAction.CHANGE_EXPLANATION: TeachingEffectStatus.NEGATIVE_CANDIDATE,
        ResourceInteractionAction.REQUEST_EXAMPLE: TeachingEffectStatus.NEGATIVE_CANDIDATE,
    }.get(event.action, TeachingEffectStatus.TRIED)
    confidence = 1.0 if event.action in {
        ResourceInteractionAction.UNDERSTOOD,
        ResourceInteractionAction.STILL_CONFUSED,
        ResourceInteractionAction.CHANGE_EXPLANATION,
        ResourceInteractionAction.REQUEST_EXAMPLE,
    } else 0.65

    current = load_teaching_memory_view(profile_service, event.learner_id)
    strategies = {
        (item.concept_id, item.strategy): item for item in current.strategies
    }
    strategy_name = f"resource:{event.resource_form or event.resource_family}"
    for concept_id in _bounded_tuple(event.concept_ids):
        key = (concept_id, strategy_name)
        previous = strategies.get(key)
        strategies[key] = TeachingStrategyMemory(
            learner_id=event.learner_id,
            concept_id=concept_id,
            strategy=strategy_name,
            attempts=(previous.attempts if previous else 0) + 1,
            learning_event_ids=_bounded_tuple((
                *((previous.learning_event_ids if previous else ())),
                event.event_id,
            )),
            effect=effect,
            confidence=max(previous.confidence if previous else 0.0, confidence),
            last_used=event.timestamp,
            effect_status=effect_status,
            validated_outcomes=previous.validated_outcomes if previous else 0,
        )

    interactions.append({
        "event_id": event.event_id,
        "learner_id": event.learner_id,
        "task_id": event.task_id,
        "resource_id": event.resource_id,
        "resource_family": event.resource_family,
        "resource_form": event.resource_form,
        "action": event.action.value,
        "concept_ids": list(event.concept_ids),
        "source_type": event.source_type,
        "source_class": "OBSERVED",
        "timestamp": event.timestamp,
    })
    raw["version"] = _VERSION
    raw["strategies"] = [
        _to_mapping(item)
        for item in sorted(strategies.values(), key=lambda item: item.last_used)[
            -_STRATEGY_LIMIT:
        ]
    ]
    raw["resource_interactions"] = interactions[-_RESOURCE_INTERACTION_LIMIT:]
    raw["updated_at"] = max(
        float(raw.get("updated_at", 0.0) or 0.0),
        event.timestamp,
    )
    learner_memory["teaching_memory"] = raw
    extras["learner_memory"] = learner_memory
    profile.extras = extras
    return _save_profile(profile_service, profile)


def commit_practice_validation(
    profile_service: Any,
    event: PracticeValidationEvent,
) -> bool:
    """Link one real AnswerRecord outcome to a server-issued teaching resource.

    The existing BKT/Profile path remains the sole mastery writer.  This only
    validates a contextual teaching-strategy observation and advances an
    already source-backed misconception lifecycle.
    """

    if (
        profile_service is None
        or not isinstance(event, PracticeValidationEvent)
        or not event.learner_id
        or not event.task_id
        or not event.resource_id
        or not event.question_id
        or not event.kp_id
    ):
        return False
    profile = _load_profile(profile_service, event.learner_id)
    if profile is None:
        return False
    extras = dict(getattr(profile, "extras", {}) or {})
    learner_memory = dict(extras.get("learner_memory") or {})
    raw = dict(learner_memory.get("teaching_memory") or {})
    validations = [
        dict(item) for item in raw.get("practice_validations") or ()
        if isinstance(item, Mapping)
    ]
    if any(str(item.get("event_id") or "") == event.event_id for item in validations):
        return True

    current = load_teaching_memory_view(profile_service, event.learner_id)
    strategies = {(item.concept_id, item.strategy): item for item in current.strategies}
    for concept_id in _bounded_tuple(event.concept_ids):
        key = (concept_id, event.strategy)
        previous = strategies.get(key)
        strategies[key] = TeachingStrategyMemory(
            learner_id=event.learner_id,
            concept_id=concept_id,
            strategy=event.strategy,
            attempts=previous.attempts if previous else 1,
            learning_event_ids=_bounded_tuple((
                *((previous.learning_event_ids if previous else ())),
                event.event_id,
            )),
            effect="practice_correct" if event.correct else "practice_incorrect",
            confidence=max(previous.confidence if previous else 0.0, 0.85),
            last_used=event.timestamp,
            effect_status=(
                TeachingEffectStatus.VALIDATED_POSITIVE
                if event.correct
                else TeachingEffectStatus.VALIDATED_NEGATIVE
            ),
            validated_outcomes=(previous.validated_outcomes if previous else 0) + 1,
        )

    misconceptions: dict[str, TeachingMisconceptionMemory] = {
        item.misconception_id: item for item in current.misconceptions
    }
    concept_set = set(event.concept_ids)
    for misconception_id, previous in tuple(misconceptions.items()):
        if not concept_set.intersection(previous.concept_ids):
            continue
        checks = _bounded_tuple((*previous.check_event_ids, event.event_id))
        if event.correct:
            status = (
                MisconceptionLifecycle.RESOLVED
                if previous.status is MisconceptionLifecycle.CHECK
                and len(previous.check_event_ids) >= 1
                else MisconceptionLifecycle.CHECK
            )
        else:
            status = (
                MisconceptionLifecycle.RECURRENT
                if previous.status is MisconceptionLifecycle.RESOLVED
                else MisconceptionLifecycle.CONFIRMED
            )
        misconceptions[misconception_id] = TeachingMisconceptionMemory(
            learner_id=previous.learner_id,
            misconception_id=previous.misconception_id,
            concept_ids=previous.concept_ids,
            status=status,
            source_event_ids=_bounded_tuple((*previous.source_event_ids, event.event_id)),
            intervention_event_ids=previous.intervention_event_ids,
            confidence=max(previous.confidence, 0.85),
            last_seen=event.timestamp,
            check_event_ids=checks,
        )

    validations.append({
        "event_id": event.event_id,
        "learner_id": event.learner_id,
        "task_id": event.task_id,
        "resource_id": event.resource_id,
        "question_id": event.question_id,
        "kp_id": event.kp_id,
        "concept_ids": list(event.concept_ids),
        "strategy": event.strategy,
        "correct": event.correct,
        "source_class": "OBSERVED",
        "timestamp": event.timestamp,
    })
    raw["version"] = "t5678-v2"
    raw["strategies"] = [
        _to_mapping(item) for item in sorted(
            strategies.values(), key=lambda item: item.last_used
        )[-_STRATEGY_LIMIT:]
    ]
    raw["misconceptions"] = [
        _to_mapping(item) for item in sorted(
            misconceptions.values(), key=lambda item: item.last_seen
        )[-_MISCONCEPTION_LIMIT:]
    ]
    raw["practice_validations"] = validations[-_PRACTICE_VALIDATION_LIMIT:]
    raw["updated_at"] = max(float(raw.get("updated_at", 0.0) or 0.0), event.timestamp)
    learner_memory["teaching_memory"] = raw
    extras["learner_memory"] = learner_memory
    profile.extras = extras
    return _save_profile(profile_service, profile)


def commit_teaching_memory(
    profile_service: Any,
    learner_id: str,
    event: TeachingLearningEvent,
    *,
    source_misconceptions: Iterable[Mapping[str, Any]] = (),
) -> bool:
    """Extract and persist bounded teaching facts from one real R-07A event."""

    learner_id = str(learner_id or "")
    if (
        profile_service is None
        or not learner_id
        or not isinstance(event, TeachingLearningEvent)
        or event.learner_id != learner_id
        or not event.event_id
        or not event.task_id
    ):
        return False
    profile = _load_profile(profile_service, learner_id)
    if profile is None:
        return False
    current = load_teaching_memory_view(profile_service, learner_id)
    if any(item.event_id == event.event_id for item in current.experiences):
        return True
    existing_extras = dict(getattr(profile, "extras", {}) or {})
    existing_learner_memory = dict(existing_extras.get("learner_memory") or {})
    existing_raw = dict(existing_learner_memory.get("teaching_memory") or {})

    concepts = {item.concept_id: item for item in current.concept_learning}
    strategies = {
        (item.concept_id, item.strategy): item for item in current.strategies
    }
    misconceptions = {
        item.misconception_id: item for item in current.misconceptions
    }
    completed = bool(
        event.outcome.completion_eligible and event.review.real_reviewer_executed
    )
    teaching_effect = (
        "delivered_reviewed_unverified"
        if completed
        else "delivered_unverified"
        if event.outcome.concept_exposed
        else "delivery_incomplete"
    )
    event_confidence = _confidence(event.guidance.confidence)

    for concept_id in _bounded_tuple(event.outcome.concept_exposed):
        previous = concepts.get(concept_id)
        concepts[concept_id] = ConceptLearningMemory(
            learner_id=learner_id,
            concept_id=concept_id,
            learning_attempts=(previous.learning_attempts if previous else 0) + 1,
            first_seen=previous.first_seen if previous else event.timestamp,
            last_seen=event.timestamp,
            learning_event_ids=_bounded_tuple((
                *((previous.learning_event_ids if previous else ())),
                event.event_id,
            )),
            teaching_effect=teaching_effect,
            confidence=(
                max(previous.confidence, event_confidence)
                if previous
                else event_confidence
            ),
        )

    event_strategies: list[str] = []
    for action in event.teaching_process:
        strategy = str(getattr(action.action_type, "value", action.action_type))
        event_strategies.append(strategy)
        action_concepts = _bounded_tuple(action.concept_refs or event.related_concepts)
        effect = (
            "delivered_reviewed_unverified"
            if action.status == "delivered" and completed
            else str(action.status or "unknown")
        )
        for concept_id in action_concepts:
            key = (concept_id, strategy)
            previous = strategies.get(key)
            strategies[key] = TeachingStrategyMemory(
                learner_id=learner_id,
                concept_id=concept_id,
                strategy=strategy,
                attempts=(previous.attempts if previous else 0) + 1,
                learning_event_ids=_bounded_tuple((
                    *((previous.learning_event_ids if previous else ())),
                    event.event_id,
                )),
                effect=effect,
                confidence=(
                    max(previous.confidence, event_confidence)
                    if previous
                    else event_confidence
                ),
                last_used=event.timestamp,
                effect_status=(
                    previous.effect_status
                    if previous and previous.effect_status in {
                        TeachingEffectStatus.VALIDATED_POSITIVE,
                        TeachingEffectStatus.VALIDATED_NEGATIVE,
                    }
                    else TeachingEffectStatus.TRIED
                ),
                validated_outcomes=previous.validated_outcomes if previous else 0,
            )

    addressed_concepts = set(event.outcome.misconception_addressed)
    current_misconception_concepts = _bounded_tuple(
        event.before_state.misconception_state
    )
    for source in source_misconceptions:
        misconception_id = str(source.get("misconception_id") or "").strip()
        source_refs = _bounded_tuple(source.get("source_events") or ())
        if not misconception_id or not source_refs:
            continue
        previous = misconceptions.get(misconception_id)
        source_status = str(source.get("status") or "ACTIVE")
        if previous and previous.status is MisconceptionLifecycle.RESOLVED and source_status in {
            "ACTIVE", "UNCERTAIN"
        }:
            status = MisconceptionLifecycle.REAPPEARED
        elif addressed_concepts:
            status = MisconceptionLifecycle.ADDRESSED
        elif source_status == "RESOLVED":
            status = MisconceptionLifecycle.RESOLVED
        else:
            status = MisconceptionLifecycle.ACTIVE
        misconceptions[misconception_id] = TeachingMisconceptionMemory(
            learner_id=learner_id,
            misconception_id=misconception_id,
            concept_ids=_bounded_tuple((
                *((previous.concept_ids if previous else ())),
                *current_misconception_concepts,
            )),
            status=status,
            source_event_ids=_bounded_tuple((
                *((previous.source_event_ids if previous else ())),
                *source_refs,
            )),
            intervention_event_ids=_bounded_tuple((
                *((previous.intervention_event_ids if previous else ())),
                *((event.event_id,) if addressed_concepts else ()),
            )),
            confidence=max(
                previous.confidence if previous else 0.0,
                _confidence(source.get("confidence")),
            ),
            last_seen=event.timestamp,
            check_event_ids=previous.check_event_ids if previous else (),
        )

    experience = LearningExperienceMemory(
        experience_id=f"experience-{event.event_id}",
        learner_id=learner_id,
        event_id=event.event_id,
        task_id=event.task_id,
        concept_ids=_bounded_tuple(event.related_concepts),
        strategies=_bounded_tuple(event_strategies),
        outcome=(
            "reviewed_delivery" if completed else "limited_delivery"
            if event.outcome.concept_exposed else "no_delivery"
        ),
        next_learning_target=str(event.outcome.next_learning_target or ""),
        timestamp=event.timestamp,
    )
    experiences = (*current.experiences, experience)
    raw = {
        "version": _VERSION,
        "concept_learning": [
            _to_mapping(item)
            for item in sorted(concepts.values(), key=lambda item: item.last_seen)[
                -_CONCEPT_LIMIT:
            ]
        ],
        "misconceptions": [
            _to_mapping(item)
            for item in sorted(
                misconceptions.values(), key=lambda item: item.last_seen
            )[-_MISCONCEPTION_LIMIT:]
        ],
        "strategies": [
            _to_mapping(item)
            for item in sorted(strategies.values(), key=lambda item: item.last_used)[
                -_STRATEGY_LIMIT:
            ]
        ],
        "experiences": [
            _to_mapping(item) for item in experiences[-_EXPERIENCE_LIMIT:]
        ],
        "resource_interactions": [
            dict(item) for item in existing_raw.get("resource_interactions") or ()
            if isinstance(item, Mapping)
        ][-_RESOURCE_INTERACTION_LIMIT:],
        "practice_validations": [
            dict(item) for item in existing_raw.get("practice_validations") or ()
            if isinstance(item, Mapping)
        ][-_PRACTICE_VALIDATION_LIMIT:],
        "updated_at": event.timestamp,
    }
    extras = dict(getattr(profile, "extras", {}) or {})
    learner_memory = dict(extras.get("learner_memory") or {})
    learner_memory["teaching_memory"] = raw
    extras["learner_memory"] = learner_memory
    profile.extras = extras
    return _save_profile(profile_service, profile)


__all__ = [
    "ConceptLearningMemory",
    "LearningExperienceMemory",
    "MisconceptionLifecycle",
    "PracticeValidationEvent",
    "TeachingEffectStatus",
    "TeachingMemoryInterpretation",
    "TeachingMemoryView",
    "TeachingMisconceptionMemory",
    "TeachingStrategyMemory",
    "commit_resource_interaction",
    "commit_practice_validation",
    "commit_teaching_memory",
    "interpret_teaching_memory",
    "load_teaching_memory_view",
]
