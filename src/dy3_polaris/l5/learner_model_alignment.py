"""Read-only authority alignment for existing learner model sources.

This module does not create learner state.  It preserves observed records,
live model outputs, and profile cache values as separate runtime facts so the
Learner Intelligence layer can choose a source without silently overwriting
conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping


class AlignmentStatus(str, Enum):
    SYNCED = "SYNCED"
    STALE_PROFILE = "STALE_PROFILE"
    MODEL_MISSING = "MODEL_MISSING"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze(item) for item in value)
    return value


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
        except Exception:  # noqa: BLE001 - unavailable source remains missing
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
        "mastery_prob",
        "attempts",
        "correct_count",
        "last_attempt_time",
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
    except Exception:  # noqa: BLE001 - alignment is read-only and fail-closed
        return None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mapping_equal(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        math.isclose(float(left[key]), float(right[key]), abs_tol=1e-6)
        for key in left
    )


@dataclass(frozen=True, slots=True)
class AlignedModelState:
    model_name: str
    value: Any
    confidence: float | None
    timestamp: float | None
    source_type: str
    available: bool


@dataclass(frozen=True, slots=True)
class LearnerConfidenceAlignment:
    """Distinct confidence meanings; profile confidence is never reused."""

    data_confidence: float | None
    model_confidence: Mapping[str, float | None]
    teaching_confidence: float | None
    profile_confidence: float | None


@dataclass(frozen=True, slots=True)
class LearnerModelAlignment:
    """Request-local authority decision over existing learner data."""

    learner_id: str
    observed_records: tuple[Mapping[str, Any], ...]
    model_states: Mapping[str, AlignedModelState]
    profile_cache: Mapping[str, Any]
    bkt_mastery: Mapping[str, float]
    profile_mastery: Mapping[str, float]
    selected_mastery: Mapping[str, float]
    irt_theta: float | None
    profile_theta: float | None
    selected_theta: float | None
    selected_mastery_source: str
    selected_theta_source: str
    mastery_status: AlignmentStatus
    theta_status: AlignmentStatus
    alignment_status: AlignmentStatus
    confidence: LearnerConfidenceAlignment
    profile_projection: Mapping[str, Any]
    ability_projection: Mapping[str, Any]
    timestamp: float | None


def _observed_records(store: Any, learner_id: str) -> tuple[Mapping[str, Any], ...]:
    records = _safe_call(store, "get_answer_history", learner_id) or ()
    projected: list[Mapping[str, Any]] = []
    for item in list(records)[-20:]:
        record = _as_dict(item)
        projected.append(
            _freeze(
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
        )
    return tuple(projected)


def _live_bkt_states(store: Any, learner_id: str) -> dict[str, Any]:
    states = _safe_call(store, "get_all_tracing_states", learner_id) or {}
    return dict(states) if isinstance(states, Mapping) else {}


def _component_status(
    live_available: bool,
    cache_available: bool,
    equal: bool,
) -> AlignmentStatus:
    if live_available and cache_available:
        return AlignmentStatus.SYNCED if equal else AlignmentStatus.STALE_PROFILE
    if live_available:
        return AlignmentStatus.SYNCED
    if cache_available:
        return AlignmentStatus.MODEL_MISSING
    return AlignmentStatus.DATA_INSUFFICIENT


def align_learner_models(
    learner_id: str,
    *,
    profile_service: Any = None,
    irt_service: Any = None,
) -> LearnerModelAlignment:
    """Align observed facts, live model states, and profile cache values."""

    store = getattr(profile_service, "store", None)
    records = _observed_records(store, learner_id)
    profile = _safe_call(profile_service, "get_profile_snapshot", learner_id)
    profile_data = _as_dict(profile)
    ability = _safe_call(irt_service, "get_ability_snapshot", learner_id) or {}
    ability_data = dict(ability) if isinstance(ability, Mapping) else {}
    confidence_report = _safe_call(profile_service, "get_confidence", learner_id) or {}
    confidence_data = (
        dict(confidence_report)
        if isinstance(confidence_report, Mapping)
        else {}
    )

    bkt_states = _live_bkt_states(store, learner_id)
    bkt_mastery = {
        str(kp_id): float(getattr(state, "mastery_prob"))
        for kp_id, state in bkt_states.items()
        if _float(getattr(state, "mastery_prob", None)) is not None
    }
    profile_raw_mastery = profile_data.get("kp_mastery")
    profile_mastery = {
        str(kp_id): float(value)
        for kp_id, value in (
            profile_raw_mastery.items()
            if isinstance(profile_raw_mastery, Mapping)
            else ()
        )
        if _float(value) is not None
    }
    mastery_status = _component_status(
        bool(bkt_mastery),
        bool(profile_mastery),
        _mapping_equal(bkt_mastery, profile_mastery)
        if bkt_mastery and profile_mastery
        else False,
    )
    selected_mastery = bkt_mastery if bkt_mastery else profile_mastery
    selected_mastery_source = (
        "bkt_tracing_state" if bkt_mastery else
        "profile_cache_fallback" if profile_mastery else
        "unknown"
    )

    response_count = int(ability_data.get("response_count", 0) or 0)
    irt_theta = _float(ability_data.get("theta")) if response_count > 0 else None
    profile_theta = _float(profile_data.get("theta"))
    theta_status = _component_status(
        irt_theta is not None,
        profile_theta is not None,
        bool(
            irt_theta is not None
            and profile_theta is not None
            and math.isclose(irt_theta, profile_theta, abs_tol=1e-6)
        ),
    )
    selected_theta = irt_theta if irt_theta is not None else profile_theta
    selected_theta_source = (
        "irt_service" if irt_theta is not None else
        "profile_cache_fallback" if profile_theta is not None else
        "unknown"
    )

    bkt_timestamp_values = [
        _float(getattr(state, "last_attempt_time", None))
        for state in bkt_states.values()
    ]
    bkt_timestamp = max(
        (item for item in bkt_timestamp_values if item is not None),
        default=None,
    )
    irt_timestamp = _float(ability_data.get("last_update_time"))
    profile_timestamp = _float(profile_data.get("snapshot_ts"))

    kp_confidence = confidence_data.get("kp_confidence")
    kp_confidence_values = [
        float(value)
        for value in (
            kp_confidence.values() if isinstance(kp_confidence, Mapping) else ()
        )
        if _float(value) is not None
    ]
    bkt_confidence = (
        sum(kp_confidence_values) / len(kp_confidence_values)
        if kp_confidence_values
        else None
    )
    irt_se = _float(ability_data.get("se")) if irt_theta is not None else None
    irt_confidence = (
        round(min(1.0, 1.0 / (1.0 + max(irt_se, 0.01))), 4)
        if irt_se is not None
        else None
    )
    data_confidence = _float(confidence_data.get("data_sufficiency"))
    if data_confidence is None and not records:
        data_confidence = 0.0
    profile_confidence = _float(profile_data.get("confidence"))
    model_confidence_values = [
        value for value in (bkt_confidence, irt_confidence) if value is not None
    ]
    teaching_components = list(model_confidence_values)
    if data_confidence is not None:
        teaching_components.append(data_confidence)
    teaching_confidence = (
        round(min(teaching_components), 4)
        if model_confidence_values and teaching_components
        else None
    )
    confidence = LearnerConfidenceAlignment(
        data_confidence=data_confidence,
        model_confidence=MappingProxyType(
            {"bkt": bkt_confidence, "irt": irt_confidence}
        ),
        teaching_confidence=teaching_confidence,
        profile_confidence=profile_confidence,
    )

    model_states = MappingProxyType(
        {
            "bkt": AlignedModelState(
                model_name="BKT",
                value=_freeze(bkt_mastery),
                confidence=bkt_confidence,
                timestamp=bkt_timestamp,
                source_type="bkt_tracing_state",
                available=bool(bkt_mastery),
            ),
            "irt": AlignedModelState(
                model_name="IRT",
                value=irt_theta,
                confidence=irt_confidence,
                timestamp=irt_timestamp,
                source_type="irt_service",
                available=irt_theta is not None,
            ),
        }
    )
    profile_cache = _freeze(
        {
            "theta": profile_theta,
            "mastery": profile_mastery,
            "level": profile_data.get("level"),
            "weak_kps": tuple(profile_data.get("weak_kps") or ()),
            "dimensions": profile_data.get("dimensions") or {},
            "timestamp": profile_timestamp,
            "source_type": "profile_cache",
        }
    )

    component_statuses = (mastery_status, theta_status)
    if AlignmentStatus.STALE_PROFILE in component_statuses:
        alignment_status = AlignmentStatus.STALE_PROFILE
    elif not records and not bkt_mastery and irt_theta is None:
        alignment_status = AlignmentStatus.DATA_INSUFFICIENT
    elif AlignmentStatus.MODEL_MISSING in component_statuses:
        alignment_status = AlignmentStatus.MODEL_MISSING
    elif not records:
        alignment_status = AlignmentStatus.DATA_INSUFFICIENT
    else:
        alignment_status = AlignmentStatus.SYNCED

    public_profile = dict(profile_data)
    if isinstance(public_profile.get("extras"), Mapping):
        extras = dict(public_profile["extras"])
        extras.pop("learner_memory", None)
        public_profile["extras"] = extras
    timestamps = [
        item
        for item in (bkt_timestamp, irt_timestamp, profile_timestamp)
        if item is not None
    ]
    return LearnerModelAlignment(
        learner_id=learner_id,
        observed_records=records,
        model_states=model_states,
        profile_cache=profile_cache,
        bkt_mastery=_freeze(bkt_mastery),
        profile_mastery=_freeze(profile_mastery),
        selected_mastery=_freeze(selected_mastery),
        irt_theta=irt_theta,
        profile_theta=profile_theta,
        selected_theta=selected_theta,
        selected_mastery_source=selected_mastery_source,
        selected_theta_source=selected_theta_source,
        mastery_status=mastery_status,
        theta_status=theta_status,
        alignment_status=alignment_status,
        confidence=confidence,
        profile_projection=_freeze(public_profile),
        ability_projection=_freeze(ability_data),
        timestamp=max(timestamps) if timestamps else None,
    )


__all__ = [
    "AlignedModelState",
    "AlignmentStatus",
    "LearnerConfidenceAlignment",
    "LearnerModelAlignment",
    "align_learner_models",
]
