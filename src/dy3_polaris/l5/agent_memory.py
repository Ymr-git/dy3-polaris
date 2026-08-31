"""Agent 记忆与自提升 — AgentDefinition.memory_config/reputation_config 的运行时落地.

能力体现:
1. 问答记忆: 每次多 Agent 问答后, 把 {query, answer, verdict, confidence, ts}
   写入学习者画像 extras.agent_memory (跨 Agent 共享, 走 L2 唯一写方 apply_update);
2. 记忆召回: 生成前按查询词重叠召回该学习者历史问答, 用于关联上下文
   (跨问题积累领域偏好, 而非只针对单次反应);
3. 声誉自提升: 按审核裁决更新各 Agent 声誉分 (approved 奖励 / rejected 惩罚),
   存入画像 extras.agent_reputation, 声誉随表现动态调整;
4. 画像学习: 提问主题写入 extras.query_log, 学情画像 Agent 可据此累积
   学习者在领域内的兴趣/薄弱倾向.
"""

from __future__ import annotations

import logging
import hashlib
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from dy3_polaris.l2.kp_catalog import (
    CHAPTER_LABELS,
    NEW_KP_EDGES,
    NEW_KP_NAMES,
    NEW_KP_TO_CHAPTER,
)
from dy3_polaris.l5.learner_intelligence import resolve_learning_topics

logger = logging.getLogger("dy3_polaris.l5.agent_memory")

_MEMORY_LIMIT = 40
_REPUTATION_BASE = 80.0
_LEARNER_MEMORY_VERSION = 3
_LEARNING_HISTORY_LIMIT = 20
_ERROR_PATTERN_LIMIT = 12
_LEARNING_EVENT_LIMIT = 80
_MISCONCEPTION_LIMIT = 20


class LearningEventClassification(str, Enum):
    """Source semantics for one internal learning-history event."""

    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    DERIVED = "DERIVED"


class LearningEventType(str, Enum):
    """Closed event vocabulary; model values never belong in an event."""

    QUERY = "query"
    USER_ANSWER = "user_answer"
    PRACTICE_RESULT = "practice_result"
    FEEDBACK = "feedback"
    MODEL_INFERENCE = "model_inference"
    TOPIC_EXPLAINED = "topic_explained"
    REVIEWER_CHALLENGE = "reviewer_challenge"
    KNOWLEDGE_GAP = "knowledge_gap"
    CLARIFICATION = "clarification"
    LEARNING_RECOMMENDATION = "learning_recommendation"


class LearningEventOutcome(str, Enum):
    """Observed outcomes that may update, but never create, mastery state."""

    NONE = "NONE"
    CORRECTION_ACCEPTED = "CORRECTION_ACCEPTED"
    CORRECT_RESPONSE = "CORRECT_RESPONSE"
    REPEATED_ERROR = "REPEATED_ERROR"


class MisconceptionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    UNCERTAIN = "UNCERTAIN"


_EVENT_CLASSIFICATION = {
    LearningEventType.QUERY: LearningEventClassification.OBSERVED,
    LearningEventType.USER_ANSWER: LearningEventClassification.OBSERVED,
    LearningEventType.PRACTICE_RESULT: LearningEventClassification.OBSERVED,
    LearningEventType.FEEDBACK: LearningEventClassification.OBSERVED,
    LearningEventType.MODEL_INFERENCE: LearningEventClassification.INFERRED,
    LearningEventType.TOPIC_EXPLAINED: LearningEventClassification.DERIVED,
    LearningEventType.REVIEWER_CHALLENGE: LearningEventClassification.DERIVED,
    LearningEventType.KNOWLEDGE_GAP: LearningEventClassification.DERIVED,
    LearningEventType.CLARIFICATION: LearningEventClassification.DERIVED,
    LearningEventType.LEARNING_RECOMMENDATION: LearningEventClassification.DERIVED,
}


@dataclass(frozen=True, slots=True)
class LearningEvent:
    """Bounded historical event, not a learner model or public contract."""

    event_id: str
    learner_id: str
    task_id: str
    event_type: LearningEventType
    classification: LearningEventClassification
    source: str
    timestamp: float
    topics: tuple[str, ...] = ()
    content: str = ""
    reference: str = ""
    outcome: LearningEventOutcome = LearningEventOutcome.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "learner_id": self.learner_id,
            "task_id": self.task_id,
            "event_type": self.event_type.value,
            "classification": self.classification.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "topics": list(self.topics),
            "content": self.content,
            "reference": self.reference,
            "outcome": self.outcome.value,
        }


def create_learning_event(
    *,
    learner_id: str,
    task_id: str,
    event_type: LearningEventType | str,
    source: str,
    topics: tuple[str, ...] = (),
    content: str = "",
    reference: str = "",
    outcome: LearningEventOutcome | str = LearningEventOutcome.NONE,
    timestamp: float | None = None,
) -> LearningEvent:
    """Create one classified event without accepting model-state fields."""

    normalized_type = (
        event_type
        if isinstance(event_type, LearningEventType)
        else LearningEventType(str(event_type))
    )
    normalized_topics = tuple(
        dict.fromkeys(str(item).strip() for item in topics if str(item).strip())
    )
    normalized_content = str(content or "")[:200]
    normalized_reference = str(reference or "")[:128]
    normalized_outcome = (
        outcome
        if isinstance(outcome, LearningEventOutcome)
        else LearningEventOutcome(str(outcome))
    )
    seed = "|".join(
        (
            str(learner_id),
            str(task_id),
            normalized_type.value,
            str(source),
            ",".join(normalized_topics),
            normalized_content,
            normalized_reference,
            normalized_outcome.value,
        )
    )
    event_id = f"learning-event-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"
    return LearningEvent(
        event_id=event_id,
        learner_id=str(learner_id),
        task_id=str(task_id),
        event_type=normalized_type,
        classification=_EVENT_CLASSIFICATION[normalized_type],
        source=str(source),
        timestamp=float(timestamp if timestamp is not None else time.time()),
        topics=normalized_topics,
        content=normalized_content,
        reference=normalized_reference,
        outcome=normalized_outcome,
    )


@dataclass(frozen=True, slots=True)
class Misconception:
    """A bounded learner-model hypothesis backed by real source events."""

    misconception_id: str
    domain: str
    topic: str
    belief: str
    status: MisconceptionStatus
    confidence: float
    severity: str
    source_events: tuple[str, ...]
    evidence: tuple[str, ...]
    correction_strategy: str
    first_detected: float
    last_updated: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "misconception_id": self.misconception_id,
            "domain": self.domain,
            "topic": self.topic,
            "belief": self.belief,
            "status": self.status.value,
            "confidence": self.confidence,
            "severity": self.severity,
            "source_events": list(self.source_events),
            "evidence": list(self.evidence),
            "correction_strategy": self.correction_strategy,
            "first_detected": self.first_detected,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Misconception":
        return cls(
            misconception_id=str(value.get("misconception_id") or ""),
            domain=str(value.get("domain") or "unknown"),
            topic=str(value.get("topic") or "unknown"),
            belief=str(value.get("belief") or ""),
            status=MisconceptionStatus(str(value.get("status") or "UNCERTAIN")),
            confidence=float(value.get("confidence", 0.0) or 0.0),
            severity=str(value.get("severity") or "MEDIUM"),
            source_events=tuple(str(item) for item in value.get("source_events") or ()),
            evidence=tuple(str(item) for item in value.get("evidence") or ()),
            correction_strategy=str(value.get("correction_strategy") or ""),
            first_detected=float(value.get("first_detected", 0.0) or 0.0),
            last_updated=float(value.get("last_updated", 0.0) or 0.0),
        )


@dataclass(frozen=True, slots=True)
class LearnerMemoryCandidate:
    """Filtered task facts eligible for the shared learner memory.

    The candidate intentionally contains no generated answer, raw prompt,
    private Contract object, or model reasoning.
    """

    task_id: str
    question: str
    task_mode: str
    covered_topics: tuple[str, ...]
    remaining_gaps: tuple[str, ...]
    recommended_next_action: str
    evidence_sources: tuple[str, ...]
    error_patterns: tuple[dict[str, str], ...]
    valid: bool
    refusal_reason: str = ""
    learning_events: tuple[LearningEvent, ...] = ()
    misconceptions: tuple[Misconception, ...] = ()


def _empty_learner_memory() -> dict[str, Any]:
    return {
        "version": _LEARNER_MEMORY_VERSION,
        "knowledge_state": {},
        "learning_history": [],
        "learning_events": [],
        "error_patterns": [],
        "misconceptions": [],
        "evidence_sources": [],
        "updated_at": 0.0,
    }


def load_learner_memory(
    profile_service: Any,
    learner_id: str,
) -> dict[str, Any]:
    """Read the one shared learner memory from the existing L2 profile."""
    profile = _load_profile(profile_service, learner_id)
    if profile is None:
        return _empty_learner_memory()
    raw = _get_extras(profile).get("learner_memory")
    if not isinstance(raw, dict):
        return _empty_learner_memory()
    memory = _empty_learner_memory()
    memory.update(raw)
    memory["knowledge_state"] = dict(raw.get("knowledge_state") or {})
    memory["learning_history"] = list(raw.get("learning_history") or [])
    memory["learning_events"] = list(raw.get("learning_events") or [])
    memory["error_patterns"] = list(raw.get("error_patterns") or [])
    memory["misconceptions"] = list(raw.get("misconceptions") or [])
    memory["evidence_sources"] = list(raw.get("evidence_sources") or [])
    return memory


def _query_is_memory_related(query: str, memory: dict[str, Any]) -> bool:
    if not memory.get("learning_history") and not memory.get("learning_events"):
        return False
    current_topics, _ = _candidate_topics(query)
    historical_topics = {
        str(topic)
        for event in memory.get("learning_events", ())
        if isinstance(event, dict)
        for topic in event.get("topics", ())
    }
    historical_topics.update(
        str(topic)
        for item in memory.get("learning_history", ())
        if isinstance(item, dict)
        for topic in item.get("covered_topics", ())
    )
    if set(current_topics) & historical_topics:
        return True
    ancestor_map: dict[str, set[str]] = {}
    for edge in NEW_KP_EDGES:
        if edge.get("rel") == "prerequisite_of":
            ancestor_map.setdefault(str(edge["dst"]), set()).add(str(edge["src"]))

    def ancestors(topic: str) -> set[str]:
        found: set[str] = set()
        frontier = list(ancestor_map.get(topic, ()))
        while frontier:
            current = frontier.pop()
            if current in found:
                continue
            found.add(current)
            frontier.extend(ancestor_map.get(current, ()))
        return found

    if any(ancestors(topic) & historical_topics for topic in current_topics):
        return True
    historical_chapters = {
        NEW_KP_TO_CHAPTER.get(topic, "") for topic in historical_topics
    }
    if any(
        NEW_KP_TO_CHAPTER.get(topic, "") in historical_chapters
        and NEW_KP_TO_CHAPTER.get(topic, "")
        for topic in current_topics
    ):
        return True
    q = str(query or "").lower().replace("³⁺", "3+")
    current_entities = set(
        re.findall(r"[a-z]{1,3}\d*\+", q, flags=re.IGNORECASE)
    )
    if current_entities:
        for item in memory.get("learning_history", ()):
            if not isinstance(item, dict):
                continue
            previous = str(item.get("question") or "").lower().replace("³⁺", "3+")
            if current_entities & set(
                re.findall(r"[a-z]{1,3}\d*\+", previous, flags=re.IGNORECASE)
            ):
                return True
    return any(
        str(item.get("question") or "")[:24] in str(query or "")
        for item in memory.get("learning_history", [])[-5:]
        if str(item.get("question") or "")[:24]
    )


def build_memory_views(
    profile_service: Any,
    learner_id: str,
    query: str,
) -> dict[str, dict[str, Any]]:
    """Project one shared memory into four least-information Agent views."""
    memory = load_learner_memory(profile_service, learner_id)
    related = _query_is_memory_related(query, memory)
    events = tuple(
        dict(item)
        for item in memory.get("learning_events", ())
        if isinstance(item, dict)
    )
    observed_events = tuple(
        item for item in events if item.get("classification") == "OBSERVED"
    )
    inferred_events = tuple(
        item for item in events if item.get("classification") == "INFERRED"
    )
    derived_events = tuple(
        item for item in events if item.get("classification") == "DERIVED"
    )
    prior_exposure = tuple(
        dict.fromkeys(
            str(topic)
            for item in derived_events
            if item.get("event_type") == LearningEventType.TOPIC_EXPLAINED.value
            for topic in item.get("topics", ())
        )
    )
    if not prior_exposure:
        prior_exposure = tuple(
            dict.fromkeys(
                str(topic)
                for item in memory.get("learning_history", ())
                if isinstance(item, dict)
                for topic in item.get("covered_topics", ())
            )
        )
    gaps = tuple(
        dict.fromkeys(
            str(topic)
            for item in derived_events
            if item.get("event_type") == LearningEventType.KNOWLEDGE_GAP.value
            for topic in item.get("topics", ())
        )
    )
    if not gaps:
        gaps = tuple(
            dict.fromkeys(
                str(topic)
                for item in memory.get("learning_history", ())
                if isinstance(item, dict)
                for topic in item.get("remaining_gap", ())
            )
        )
    current_topics, _ = _candidate_topics(query)
    focus = tuple(topic for topic in current_topics if topic not in prior_exposure)
    if not focus and related:
        focus = tuple(topic for topic in gaps if topic not in prior_exposure)[:4]
    focus_labels = tuple(NEW_KP_NAMES.get(topic, topic) for topic in focus)
    exposure_labels = tuple(NEW_KP_NAMES.get(topic, topic) for topic in prior_exposure)
    available = bool(related and (prior_exposure or gaps or events))
    misconceptions = tuple(
        dict(item)
        for item in memory.get("misconceptions", ())
        if isinstance(item, dict)
        and str(item.get("status") or "") in {"ACTIVE", "UNCERTAIN"}
    )
    common = {
        "memory_available": available,
        "related_to_query": related,
        "legacy_projection": "isolated",
    }
    return {
        "agent.learning.diagnosis": {
            **common,
            "observed_learning_events": observed_events[-8:],
            "inferred_learning_events": inferred_events[-8:],
            "derived_learning_events": derived_events[-8:],
            "prior_exposure_topics": prior_exposure,
            "prior_exposure_labels": exposure_labels,
            "remaining_gaps": focus,
            "remaining_gap_labels": focus_labels,
            "current_topics": current_topics,
            "misconceptions": misconceptions,
            "recent_tasks": tuple(
                str(item.get("task_id") or "")
                for item in memory.get("learning_history", [])[-3:]
                if item.get("task_id")
            ),
        },
        "agent.knowledge.generation": {
            **common,
        },
        "agent.quality.review": {
            **common,
        },
        "agent.guidance.decision": {
            **common,
        },
    }


def _candidate_topics(question: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve topics and next graph nodes from the shared domain catalog."""
    covered = resolve_learning_topics(question)
    next_nodes = tuple(
        dict.fromkeys(
            str(edge["dst"])
            for topic in covered
            for edge in NEW_KP_EDGES
            if str(edge.get("src")) == topic
            and edge.get("rel") in {"prerequisite_of", "applies_to"}
        )
    )
    return covered, tuple(item for item in next_nodes if item not in covered)[:6]


def _candidate_error_patterns(
    misconceptions: tuple[Misconception, ...],
) -> tuple[dict[str, str], ...]:
    """Keep the old field as a projection of real misconception objects."""
    return tuple(
        {
            "misconception": item.belief,
            "source": "Reviewer Challenge",
            "severity": item.severity.lower(),
        }
        for item in misconceptions
    )


def extract_misconception_candidates(
    *,
    context: Any,
    question: str,
    topics: tuple[str, ...],
    timestamp: float | None = None,
) -> tuple[Misconception, ...]:
    """Map structured review facts to hypotheses without canned beliefs.

    The belief text comes from the concrete challenged Claim when available;
    the task statement is only a fallback for a semantic overclaim challenge.
    Evidence-insufficiency and ambiguous-requirement events are not silently
    attributed to the learner as misconceptions.
    """

    now = float(timestamp if timestamp is not None else time.time())
    contributions = {
        str(getattr(item, "contribution_id", "")): item
        for item in (getattr(context, "contributions", ()) or ())
    }
    allowed_types = {
        "CONDITION_MISMATCH",
        "ENTITY_MISMATCH",
        "OVERGENERALIZATION",
        "FACT_INFERENCE_CONFUSION",
        "SAFETY_OVERCLAIM",
        "UNSUPPORTED_CLAIM",
    }
    values: list[Misconception] = []
    for challenge in (getattr(context, "challenges", ()) or ()):
        challenge_type = str(
            getattr(getattr(challenge, "challenge_type", None), "value", "")
            or getattr(challenge, "challenge_type", "")
        )
        if challenge_type not in allowed_types:
            continue
        contribution = contributions.get(
            str(getattr(challenge, "target_contribution_id", ""))
        )
        target_ids = {
            str(item) for item in getattr(challenge, "target_claim_ids", ()) or ()
        }
        claims = tuple(getattr(contribution, "claims", ()) or ())
        belief = next(
            (
                str(getattr(claim, "statement", "")).strip()
                for claim in claims
                if str(getattr(claim, "claim_id", "")) in target_ids
                and str(getattr(claim, "statement", "")).strip()
            ),
            "",
        )
        if not belief and challenge_type in {
            "OVERGENERALIZATION", "FACT_INFERENCE_CONFUSION", "SAFETY_OVERCLAIM"
        }:
            belief = str(question or "").strip()
        if not belief:
            continue
        resolved_topics = resolve_learning_topics(
            belief,
            getattr(challenge, "reason", ""),
            *(getattr(challenge, "missing_information", ()) or ()),
            limit=3,
        )
        topic = next((item for item in resolved_topics if item in NEW_KP_NAMES), "")
        if not topic:
            topic = next((item for item in topics if item in NEW_KP_NAMES), "unknown")
        chapter = NEW_KP_TO_CHAPTER.get(topic, "")
        domain = CHAPTER_LABELS.get(chapter, "unknown")
        severity = str(
            getattr(getattr(challenge, "severity", None), "value", "")
            or getattr(challenge, "severity", "MEDIUM")
        ).upper()
        confidence = {
            "LOW": 0.4,
            "MEDIUM": 0.55,
            "HIGH": 0.7,
            "CRITICAL": 0.8,
        }.get(severity, 0.5)
        status = (
            MisconceptionStatus.ACTIVE
            if severity in {"HIGH", "CRITICAL"}
            else MisconceptionStatus.UNCERTAIN
        )
        requested_action = str(
            getattr(getattr(challenge, "requested_action", None), "value", "")
            or getattr(challenge, "requested_action", "")
        )
        missing = tuple(
            str(item) for item in getattr(challenge, "missing_information", ()) or ()
            if str(item)
        )
        correction_strategy = requested_action or "REVIEW"
        if missing:
            correction_strategy += ": " + "、".join(missing[:3])
        identity_seed = "|".join((domain, topic, " ".join(belief.lower().split())))
        misconception_id = (
            "misconception-"
            + hashlib.sha256(identity_seed.encode("utf-8")).hexdigest()[:16]
        )
        values.append(
            Misconception(
                misconception_id=misconception_id,
                domain=domain,
                topic=topic,
                belief=belief[:240],
                status=status,
                confidence=confidence,
                severity=severity,
                source_events=(str(getattr(challenge, "challenge_id", "")),),
                evidence=tuple(
                    str(item)
                    for item in getattr(challenge, "evidence_refs", ()) or ()
                    if str(item)
                )[:12],
                correction_strategy=correction_strategy[:240],
                first_detected=now,
                last_updated=now,
            )
        )
    unique: dict[str, Misconception] = {}
    for item in values:
        unique[item.misconception_id] = item
    return tuple(unique.values())


def update_misconceptions(
    existing: tuple[Misconception, ...],
    *,
    candidates: tuple[Misconception, ...] = (),
    events: tuple[LearningEvent, ...] = (),
) -> tuple[Misconception, ...]:
    """Apply repeated evidence to a misconception lifecycle deterministically."""

    values = {item.misconception_id: item for item in existing}
    for candidate in candidates:
        current = values.get(candidate.misconception_id)
        if current is None:
            values[candidate.misconception_id] = candidate
            continue
        confidence = min(0.95, max(current.confidence, candidate.confidence) + 0.08)
        values[candidate.misconception_id] = Misconception(
            misconception_id=current.misconception_id,
            domain=candidate.domain or current.domain,
            topic=candidate.topic or current.topic,
            belief=current.belief,
            status=(
                MisconceptionStatus.ACTIVE
                if confidence >= 0.6
                else MisconceptionStatus.UNCERTAIN
            ),
            confidence=round(confidence, 4),
            severity=candidate.severity or current.severity,
            source_events=tuple(
                dict.fromkeys((*current.source_events, *candidate.source_events))
            ),
            evidence=tuple(dict.fromkeys((*current.evidence, *candidate.evidence))),
            correction_strategy=(
                candidate.correction_strategy or current.correction_strategy
            ),
            first_detected=current.first_detected,
            last_updated=max(current.last_updated, candidate.last_updated),
        )

    for event in events:
        target = values.get(event.reference)
        if target is None or event.classification is not LearningEventClassification.OBSERVED:
            continue
        delta = {
            LearningEventOutcome.CORRECTION_ACCEPTED: -0.2,
            LearningEventOutcome.CORRECT_RESPONSE: -0.2,
            LearningEventOutcome.REPEATED_ERROR: 0.12,
        }.get(event.outcome, 0.0)
        if not delta:
            continue
        confidence = max(0.0, min(0.95, target.confidence + delta))
        if confidence <= 0.3:
            status = MisconceptionStatus.RESOLVED
        elif confidence >= 0.6:
            status = MisconceptionStatus.ACTIVE
        else:
            status = MisconceptionStatus.UNCERTAIN
        values[target.misconception_id] = Misconception(
            misconception_id=target.misconception_id,
            domain=target.domain,
            topic=target.topic,
            belief=target.belief,
            status=status,
            confidence=round(confidence, 4),
            severity=target.severity,
            source_events=tuple(
                dict.fromkeys((*target.source_events, event.event_id))
            ),
            evidence=target.evidence,
            correction_strategy=target.correction_strategy,
            first_detected=target.first_detected,
            last_updated=event.timestamp,
        )
    return tuple(sorted(values.values(), key=lambda item: item.last_updated))


def extract_memory_candidate(
    *,
    context: Any,
    final_result: Any,
    question: str,
    learner_id: str = "",
) -> LearnerMemoryCandidate:
    """Extract a bounded candidate from the reviewed final collaboration result."""
    task_id = str(getattr(final_result, "task_id", "") or "")
    task_mode = str(getattr(getattr(final_result, "task_mode", None), "value", "") or "")
    answer_identity = str(getattr(final_result, "answer_identity", "") or "")
    eligible = bool(getattr(final_result, "completion_eligibility", False))
    covered, gaps = _candidate_topics(question)
    provenance_values: list[str] = []
    for item in (getattr(final_result, "provenance_refs", ()) or ()):
        value = str(item or "").strip()
        # Evidence source memory stores references, never sentence-like content.
        if (
            value
            and len(value) <= 96
            and not any(mark in value for mark in ("\n", "。", "！", "？"))
        ):
            provenance_values.append(value)
    provenance = tuple(provenance_values)
    misconceptions = extract_misconception_candidates(
        context=context,
        question=question,
        topics=covered,
    )
    errors = _candidate_error_patterns(misconceptions)
    valid = bool(task_id and task_mode and answer_identity and eligible and covered)
    reason = "" if valid else "final reviewed task has no validated learning fact"
    events: list[LearningEvent] = []
    if valid:
        events.append(
            create_learning_event(
                learner_id=learner_id,
                task_id=task_id,
                event_type=LearningEventType.QUERY,
                source="user_interaction",
                topics=covered,
                content=question,
            )
        )
        events.append(
            create_learning_event(
                learner_id=learner_id,
                task_id=task_id,
                event_type=LearningEventType.TOPIC_EXPLAINED,
                source="reviewed_final_collaboration_result",
                topics=covered,
                reference=answer_identity,
            )
        )
        if gaps:
            events.append(
                create_learning_event(
                    learner_id=learner_id,
                    task_id=task_id,
                    event_type=LearningEventType.KNOWLEDGE_GAP,
                    source="teaching_interpretation",
                    topics=gaps,
                )
            )
        next_action = str(getattr(final_result, "next_action", "") or "")[:200]
        if next_action:
            events.append(
                create_learning_event(
                    learner_id=learner_id,
                    task_id=task_id,
                    event_type=LearningEventType.LEARNING_RECOMMENDATION,
                    source="guidance_decision",
                    topics=gaps or covered,
                    content=next_action,
                )
            )
        for misconception in misconceptions:
            events.append(
                create_learning_event(
                    learner_id=learner_id,
                    task_id=task_id,
                    event_type=LearningEventType.REVIEWER_CHALLENGE,
                    source="reviewer_challenge",
                    topics=(misconception.topic,),
                    content=misconception.belief,
                    reference=misconception.misconception_id,
                )
            )
    return LearnerMemoryCandidate(
        task_id=task_id,
        question=str(question or "")[:200],
        task_mode=task_mode,
        covered_topics=covered,
        remaining_gaps=gaps,
        recommended_next_action=str(getattr(final_result, "next_action", "") or "")[:200],
        evidence_sources=tuple(dict.fromkeys(provenance))[:12],
        error_patterns=errors,
        valid=valid,
        refusal_reason=reason,
        learning_events=tuple(events),
        misconceptions=misconceptions,
    )


def commit_learner_memory(
    profile_service: Any,
    learner_id: str,
    candidate: LearnerMemoryCandidate,
) -> bool:
    """Validate and persist selected decision facts through L2's sole writer."""
    if not candidate.valid:
        return False
    profile = _load_profile(profile_service, learner_id)
    if profile is None:
        return False
    extras = _get_extras(profile)
    memory = load_learner_memory(profile_service, learner_id)
    now = time.time()
    # Existing knowledge_state is retained only for backward-compatible reads.
    # A reviewed explanation is historical exposure, not observed mastery.
    knowledge = dict(memory.get("knowledge_state") or {})
    history = list(memory.get("learning_history") or [])
    history = [item for item in history if item.get("task_id") != candidate.task_id]
    history.append(
        {
            "task_id": candidate.task_id,
            "question": candidate.question,
            "task_mode": candidate.task_mode,
            "covered_topics": list(candidate.covered_topics),
            "remaining_gap": list(candidate.remaining_gaps),
            "recommended_next_action": candidate.recommended_next_action,
            "completed_at": now,
        }
    )
    patterns = list(memory.get("error_patterns") or [])
    known = {str(item.get("misconception") or "") for item in patterns}
    patterns.extend(
        dict(item)
        for item in candidate.error_patterns
        if item.get("misconception") not in known
    )
    sources = list(
        dict.fromkeys(
            [*memory.get("evidence_sources", []), *candidate.evidence_sources]
        )
    )[-20:]
    event_values = list(memory.get("learning_events") or [])
    known_event_ids = {
        str(item.get("event_id") or "")
        for item in event_values
        if isinstance(item, dict)
    }
    event_values.extend(
        event.to_dict()
        for event in candidate.learning_events
        if event.event_id not in known_event_ids
    )
    existing_misconceptions = tuple(
        Misconception.from_mapping(dict(item))
        for item in memory.get("misconceptions", ())
        if isinstance(item, dict) and item.get("misconception_id")
    )
    misconceptions = update_misconceptions(
        existing_misconceptions,
        candidates=candidate.misconceptions,
    )
    extras["learner_memory"] = {
        "version": _LEARNER_MEMORY_VERSION,
        "knowledge_state": knowledge,
        "learning_history": history[-_LEARNING_HISTORY_LIMIT:],
        "learning_events": event_values[-_LEARNING_EVENT_LIMIT:],
        "error_patterns": patterns[-_ERROR_PATTERN_LIMIT:],
        "misconceptions": [
            item.to_dict() for item in misconceptions[-_MISCONCEPTION_LIMIT:]
        ],
        "evidence_sources": sources,
        "updated_at": now,
    }
    profile.extras = extras
    return _save_profile(profile_service, profile)


def commit_learning_event(
    profile_service: Any,
    learner_id: str,
    event: LearningEvent,
) -> bool:
    """Persist one observed event and update only referenced misconception state."""

    if event.learner_id != str(learner_id):
        return False
    profile = _load_profile(profile_service, learner_id)
    if profile is None:
        return False
    extras = _get_extras(profile)
    memory = load_learner_memory(profile_service, learner_id)
    events = list(memory.get("learning_events") or ())
    if not any(
        isinstance(item, dict) and item.get("event_id") == event.event_id
        for item in events
    ):
        events.append(event.to_dict())
    existing = tuple(
        Misconception.from_mapping(dict(item))
        for item in memory.get("misconceptions", ())
        if isinstance(item, dict) and item.get("misconception_id")
    )
    misconceptions = update_misconceptions(existing, events=(event,))
    memory.update(
        {
            "version": _LEARNER_MEMORY_VERSION,
            "learning_events": events[-_LEARNING_EVENT_LIMIT:],
            "misconceptions": [
                item.to_dict() for item in misconceptions[-_MISCONCEPTION_LIMIT:]
            ],
            "updated_at": time.time(),
        }
    )
    extras["learner_memory"] = memory
    profile.extras = extras
    return _save_profile(profile_service, profile)


def _load_profile(profile_service: Any, learner_id: str) -> Any | None:
    if profile_service is None:
        return None
    try:
        return profile_service.get_profile_snapshot(learner_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("记忆读取画像失败 %s: %s", learner_id, exc)
        return None


def _save_profile(profile_service: Any, profile: Any) -> bool:
    if profile_service is None or profile is None:
        return False
    learner_id = getattr(profile, "learner_id", None)
    if not learner_id:
        return False
    try:
        updates: dict[str, Any] = {}
        extras = getattr(profile, "extras", None)
        if extras:
            updates["extras"] = dict(extras)
        confidence = getattr(profile, "confidence", None)
        if confidence is not None:
            updates["confidence"] = float(confidence)
        expected = getattr(profile, "version", None)
        try:
            profile_service.apply_update(
                learner_id, updates=updates, expected_version=expected
            )
            return True
        except Exception:  # noqa: BLE001 - 乐观锁冲突重试一次
            latest = profile_service.get_profile_snapshot(learner_id)
            if latest is not None:
                profile_service.apply_update(
                    learner_id, updates=updates, expected_version=latest.version
                )
                return True
            return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("记忆画像写回失败 %s: %s", learner_id, exc)
        return False


def _get_extras(profile: Any) -> dict[str, Any]:
    return dict(getattr(profile, "extras", {}) or {})


def remember_qa(
    profile_service: Any,
    learner_id: str,
    *,
    query: str,
    answer: str,
    verdict: str,
    confidence: float,
) -> bool:
    """记录一次多 Agent 问答到画像记忆 (跨 Agent 共享, 供召回与展示)."""
    profile = _load_profile(profile_service, learner_id)
    if profile is None:
        return False
    extras = _get_extras(profile)
    mem = list(extras.get("agent_memory", []) or [])
    mem.append({
        "ts": time.time(),
        "query": str(query)[:200],
        "answer": str(answer)[:300],
        "verdict": verdict,
        "confidence": round(float(confidence or 0.0), 4),
    })
    extras["agent_memory"] = mem[-_MEMORY_LIMIT:]
    profile.extras = extras
    return _save_profile(profile_service, profile)


def recall_memory(
    profile_service: Any,
    learner_id: str,
    query: str,
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """按查询词重叠召回该学习者历史问答 (跨问题记忆关联)."""
    profile = _load_profile(profile_service, learner_id)
    if profile is None:
        return []
    extras = _get_extras(profile)
    mem = list(extras.get("agent_memory", []) or [])
    if not mem:
        return []
    q = str(query).lower().replace(" ", "")
    q_tokens = {q[i : i + 2] for i in range(len(q) - 1)} or {q}
    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in mem:
        h = str(entry.get("query", "")).lower().replace(" ", "")
        if not h:
            continue
        h_tokens = {h[i : i + 2] for i in range(len(h) - 1)} or {h}
        inter = len(q_tokens & h_tokens)
        if inter:
            scored.append((inter / max(1, len(q_tokens)), entry))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]


def update_reputation(
    profile_service: Any,
    learner_id: str,
    agent_id: str,
    verdict: str,
) -> float | None:
    """按审核裁决更新 Agent 声誉分 (approved 奖励 / rejected 惩罚 / needs_review 中性偏负)."""
    profile = _load_profile(profile_service, learner_id)
    if profile is None:
        return None
    delta = {"approved": 1.0, "needs_review": -0.2, "rejected": -2.0}.get(verdict, 0.0)
    extras = _get_extras(profile)
    rep = dict(extras.get("agent_reputation", {}) or {})
    cur = float(rep.get(agent_id, _REPUTATION_BASE))
    rep[agent_id] = round(min(100.0, max(0.0, cur + delta)), 2)
    extras["agent_reputation"] = rep
    profile.extras = extras
    if not _save_profile(profile_service, profile):
        return None
    return rep[agent_id]


def record_query_log(
    profile_service: Any,
    learner_id: str,
    *,
    query: str,
    source: str = "query",
) -> bool:
    """画像学习: 提问主题记入 extras.query_log (累积领域兴趣/薄弱倾向)."""
    profile = _load_profile(profile_service, learner_id)
    if profile is None:
        return False
    extras = _get_extras(profile)
    qlog = list(extras.get("query_log", []) or [])
    qlog.append({
        "ts": time.time(),
        "query": str(query)[:200],
        "source": source,
    })
    extras["query_log"] = qlog[-_MEMORY_LIMIT:]
    profile.extras = extras
    return _save_profile(profile_service, profile)


# 声誉阈值: 低于此值触发 Prompt 回滚 (可回滚的自我进化)
_ROLLBACK_THRESHOLD = 40.0


def maybe_rollback_prompt(
    profile_service: Any,
    learner_id: str,
    agent_id: str,
    prompt_manager: Any | None = None,
) -> bool:
    """声誉驱动的 Prompt 回滚 (可回滚的自我进化).

    当某 Agent 的声誉持续走低、跌破阈值时, 自动回滚其 Prompt 到更早版本
    (若 prompt_manager 提供且存在可回滚的旧版本). 这使"进化"真正改变行为
    且可回滚, 而非仅记声誉分.

    Args:
        profile_service: 画像服务.
        learner_id: 学习者 ID.
        agent_id: Agent ID.
        prompt_manager: PromptVersionManager (可选, None 则跳过回滚).

    Returns:
        是否触发了回滚.
    """
    if prompt_manager is None:
        return False
    profile = _load_profile(profile_service, learner_id)
    if profile is None:
        return False
    extras = _get_extras(profile)
    rep = dict(extras.get("agent_reputation", {}) or {})
    score = float(rep.get(agent_id, _REPUTATION_BASE))
    if score >= _ROLLBACK_THRESHOLD:
        return False

    # 找到该 Agent 的 template_id (从 agent_id 映射, 如 agent.quality.review -> tpl.review)
    template_id = _agent_id_to_template(agent_id)
    if template_id is None:
        return False

    try:
        active = prompt_manager.get_active(template_id)
        if active is None:
            return False
        versions = prompt_manager.list_versions(template_id)
        # 找比当前活跃版本更早的版本
        older = [v for v in versions if v.version_tuple < active.version_tuple]
        if not older:
            return False
        target = older[0]  # 最近的一个旧版本
        prompt_manager.rollback(
            template_id,
            target.version,
            reason=f"声誉 {score:.1f} 低于阈值 {_ROLLBACK_THRESHOLD}, 自动回滚 {agent_id}",
            operator="self-evolution",
        )
        logger.warning(
            "自我进化: %s 声誉 %.1f 触发 Prompt 回滚 %s@%s -> %s@%s",
            agent_id, score, template_id, active.version, template_id, target.version,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prompt 回滚失败 %s: %s", agent_id, exc)
        return False


def _agent_id_to_template(agent_id: str) -> str | None:
    """Agent ID → Prompt template_id 映射."""
    mapping = {
        "agent.learning.diagnosis": "tpl.diagnosis",
        "agent.knowledge.generation": "tpl.generation",
        "agent.quality.review": "tpl.review",
        "agent.guidance.decision": "tpl.guidance",
    }
    return mapping.get(agent_id)
