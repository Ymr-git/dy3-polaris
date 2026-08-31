"""R-05B learner fact/model/profile authority alignment tests."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from starlette.testclient import TestClient

from dy3_polaris.l2.models import AnswerRecord, TracingState
from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.learner_intelligence import (
    DERIVED,
    INFERRED,
    OBSERVED,
    LearnerIntelligenceView,
    build_learner_intelligence_view,
)
from dy3_polaris.l5.learner_model_alignment import (
    AlignmentStatus,
    align_learner_models,
)
from dy3_polaris.l5.unified_app import UnifiedApp


@dataclass
class _Profile:
    learner_id: str
    theta: float = -0.2
    level: str = "beginner"
    kp_mastery: dict[str, float] = field(
        default_factory=lambda: {"A-05": 0.2}
    )
    weak_kps: list[str] = field(default_factory=lambda: ["A-05"])
    confidence: float = 0.99
    snapshot_ts: float = 100.0
    extras: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "theta": self.theta,
            "level": self.level,
            "kp_mastery": dict(self.kp_mastery),
            "weak_kps": list(self.weak_kps),
            "confidence": self.confidence,
            "snapshot_ts": self.snapshot_ts,
            "extras": dict(self.extras),
            "version": self.version,
        }


class _Store:
    def __init__(self) -> None:
        self.records = [
            AnswerRecord(
                learner_id="learner-aligned",
                kp_id="A-05",
                correct=True,
                timestamp=180.0,
                difficulty=0.6,
                question_id="q-1",
            )
        ]
        self.states = {
            "A-05": TracingState(
                kp_id="A-05",
                mastery_prob=0.8,
                attempts=4,
                correct_count=3,
                last_attempt_time=180.0,
            )
        }

    def get_answer_history(self, _learner_id: str):
        return list(self.records)

    def get_all_tracing_states(self, _learner_id: str):
        return dict(self.states)


class _ProfileService:
    def __init__(self) -> None:
        self.profile = _Profile("learner-aligned")
        self.store = _Store()

    def get_profile_snapshot(self, _learner_id: str):
        return self.profile

    def get_weak_points(self, _learner_id: str):
        return {"weak_kps": list(self.profile.weak_kps)}

    def get_confidence(self, _learner_id: str):
        return {
            "overall_confidence": 0.95,
            "kp_confidence": {"A-05": 0.7},
            "data_sufficiency": 0.4,
        }


class _IRTService:
    def get_ability_snapshot(self, learner_id: str):
        return {
            "learner_id": learner_id,
            "theta": 1.2,
            "se": 0.25,
            "response_count": 6,
            "last_update_time": 200.0,
        }


def _deps() -> AgentDependencies:
    return AgentDependencies(
        profile_service=_ProfileService(),
        irt_service=_IRTService(),
    )


def test_observed_model_and_profile_cache_sources_are_distinct() -> None:
    alignment = align_learner_models(
        "learner-aligned",
        profile_service=_deps().profile_service,
        irt_service=_IRTService(),
    )
    view = build_learner_intelligence_view(
        {"learner_id": "learner-aligned"}, _deps()
    )

    assert alignment.observed_records[0]["question_id"] == "q-1"
    assert alignment.model_states["bkt"].model_name == "BKT"
    assert alignment.model_states["bkt"].source_type == "bkt_tracing_state"
    assert alignment.model_states["irt"].model_name == "IRT"
    assert alignment.profile_cache["source_type"] == "profile_cache"
    assert {signal.classification for signal in view.facts.values()} == {OBSERVED}
    assert {signal.classification for signal in view.models.values()} == {INFERRED}
    assert {
        signal.classification for signal in view.derived_context.values()
    } == {DERIVED}


def test_bkt_irt_profile_conflicts_preserve_both_values_and_mark_stale() -> None:
    deps = _deps()
    alignment = align_learner_models(
        "learner-aligned",
        profile_service=deps.profile_service,
        irt_service=deps.irt_service,
    )

    assert dict(alignment.bkt_mastery) == {"A-05": 0.8}
    assert dict(alignment.profile_mastery) == {"A-05": 0.2}
    assert dict(alignment.selected_mastery) == {"A-05": 0.8}
    assert alignment.irt_theta == 1.2
    assert alignment.profile_theta == -0.2
    assert alignment.selected_theta == 1.2
    assert alignment.mastery_status is AlignmentStatus.STALE_PROFILE
    assert alignment.theta_status is AlignmentStatus.STALE_PROFILE
    assert alignment.alignment_status is AlignmentStatus.STALE_PROFILE


def test_confidence_semantics_are_split_and_profile_confidence_is_not_reused() -> None:
    deps = _deps()
    alignment = align_learner_models(
        "learner-aligned",
        profile_service=deps.profile_service,
        irt_service=deps.irt_service,
    )
    confidence = alignment.confidence

    assert confidence.data_confidence == 0.4
    assert confidence.model_confidence["bkt"] == 0.7
    assert confidence.model_confidence["irt"] == 0.8
    assert confidence.teaching_confidence == 0.4
    assert confidence.profile_confidence == 0.99
    assert confidence.teaching_confidence != confidence.profile_confidence


def test_diagnosis_consumes_aligned_view_without_re_reading_sources() -> None:
    deps = _deps()
    payload = {
        "learner_id": "learner-aligned",
        "query": "解释Dy³⁺黄蓝双发射",
    }
    view = build_learner_intelligence_view(payload, deps)
    assert isinstance(view, LearnerIntelligenceView)
    assert view.metadata["theta_alignment_status"] == "STALE_PROFILE"
    assert view.models["theta"].source_type == "irt_service"
    assert view.value("derived_context", "learning_stage") == "advanced"
    # BKT live state (0.8) overrides stale profile mastery (0.2), so A-05 is
    # not propagated as a weak point.
    assert view.value("derived_context", "weak_kps") == ()

    deps.profile_service.get_profile_snapshot = lambda *_args: (_ for _ in ()).throw(
        AssertionError("Diagnosis must not re-read profile")
    )
    deps.irt_service.get_ability_snapshot = lambda *_args: (_ for _ in ()).throw(
        AssertionError("Diagnosis must not re-read IRT")
    )
    diagnosis = agent_workers.run_diagnosis(
        {**payload, "_learner_intelligence_view": view}, deps
    )

    assert diagnosis["level"] == "advanced"
    assert diagnosis["weak_kps"] == []
    assert diagnosis["confidence"] == 0.4


def test_api_response_keys_remain_exactly_current_and_alignment_does_not_leak(
    tmp_path,
) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )
    response = client.post(
        "/api/query",
        json={
            "query": "为什么Dy³⁺会产生黄蓝双发射？",
            "learner_id": "learner-r05b-api",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert set(data) == {
        "task_id", "task_state", "task_events", "action_type",
        "confidence", "recommended_path", "safety_level", "pipeline",
        "evidence", "sources", "session", "learner", "viz",
        "total_elapsed_ms", "requires_confirmation",
        "knowledge_unavailable", "review", "agent_trace", "collab_lines",
        "self_correction", "reasoning_loop", "flow_events",
        "broadcast_events", "consensus_score", "consensus_reached",
        "candidate_count", "candidates", "divergence_matrix", "debate",
        "needs_adjudication", "consensus_threshold", "question_type",
        "confirmation_questions", "confirmation_reason", "plan_id",
            "clarify", "answer",
            "quality_release", "teaching_strategy", "learning_resources",
            "learner_context", "knowledge_context",
        }
    serialized = json.dumps(data, ensure_ascii=False, default=str)
    for forbidden in (
        "LearnerModelAlignment",
        "AlignmentStatus",
        "data_confidence",
        "model_confidence",
        "teaching_confidence",
        "alignment_status",
        "profile_theta",
        "irt_theta",
    ):
        assert forbidden not in serialized
