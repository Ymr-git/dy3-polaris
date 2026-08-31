"""T6/T7 authority tests for long-term learning and public growth reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from starlette.testclient import TestClient

from dy3_polaris.l2.models import AnswerRecord
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.learner_intelligence import (
    build_learner_intelligence_view,
    build_public_learner_report,
)
from dy3_polaris.l5.teaching_memory import (
    PracticeValidationEvent,
    TeachingEffectStatus,
    commit_practice_validation,
    load_teaching_memory_view,
)
from dy3_polaris.l5.unified_app import UnifiedApp


@dataclass
class _Profile:
    learner_id: str
    kp_mastery: dict[str, float] = field(default_factory=dict)
    weak_kps: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    theta: float | None = None
    level: str = "beginner"
    confidence: float = 0.5
    snapshot_ts: float = 100.0
    version: int = 1

    def to_dict(self):
        return dict(self.__dict__)


class _Store:
    def __init__(self, records=()):
        self.records = list(records)

    def get_answer_history(self, _learner_id):
        return list(self.records)

    def get_all_tracing_states(self, _learner_id):
        return {}


class _ProfileService:
    def __init__(self, profile: _Profile, records=()):
        self.profile = profile
        self.store = _Store(records)

    def get_profile_snapshot(self, _learner_id):
        return self.profile

    def get_weak_points(self, _learner_id):
        return {"weak_kps": list(self.profile.weak_kps)}

    def apply_update(self, learner_id, *, updates, expected_version=None):
        assert learner_id == self.profile.learner_id
        assert expected_version in {None, self.profile.version}
        if "extras" in updates:
            self.profile.extras.update(dict(updates["extras"]))
        self.profile.version += 1
        return self.profile


class _IRT:
    def __init__(self, response_count=0):
        self.response_count = response_count

    def get_ability_snapshot(self, learner_id):
        return {
            "learner_id": learner_id,
            "theta": 1.1,
            "se": 0.2,
            "response_count": self.response_count,
            "last_update_time": 200.0 if self.response_count else 0.0,
        }


def _report(profile, records=(), *, response_count=0, query="Dy3+ concentration quenching"):
    service = _ProfileService(profile, records)
    view = build_learner_intelligence_view(
        {"learner_id": profile.learner_id, "query": query},
        AgentDependencies(profile_service=service, irt_service=_IRT(response_count)),
    )
    return build_public_learner_report(view)


def test_no_record_is_unknown_not_weakness_and_theta_is_not_faked() -> None:
    report = _report(_Profile("learner-unknown", theta=2.0))

    assert report["status"] == "UNKNOWN"
    assert report["ability"]["status"] == "UNKNOWN"
    assert report["ability"]["theta"] is None
    assert report["difficulty_decision"]["decision"] == "DIAGNOSE_FIRST"
    assert not any(item["type"] == "VERIFIED_WEAKNESS" for item in report["findings"])


def test_model_only_profile_is_not_labelled_as_observed_evidence() -> None:
    report = _report(
        _Profile("learner-model-only", kp_mastery={"2.1.1": 0.35})
    )

    assert report["status"] == "MODEL_ONLY"
    assert report["evidence_sufficiency"]["answer_record_count"] == 0
    assert report["evidence_sufficiency"]["source_class"] == "MODEL_INFERRED"


def test_repeated_real_errors_can_form_verified_weakness_and_real_timeline() -> None:
    records = [
        AnswerRecord("learner-weak", "2.3.2", False, 10.0, question_id="Q040"),
        AnswerRecord("learner-weak", "2.3.2", False, 20.0, question_id="Q040"),
    ]
    report = _report(_Profile("learner-weak", kp_mastery={"2.3.2": 0.3}), records)

    weakness = next(item for item in report["findings"] if item["type"] == "VERIFIED_WEAKNESS")
    assert weakness["reference"] == "2.3.2"
    assert weakness["evidence_count"] == 2
    assert report["difficulty_decision"]["decision"] == "LOW"
    assert [item["reference"] for item in report["growth_timeline"]] == ["Q040", "Q040"]
    assert all(item["source_class"] == "OBSERVED" for item in report["growth_timeline"])


def test_three_real_correct_answers_allow_bounded_raise_decision() -> None:
    records = [
        AnswerRecord("learner-raise", "2.1.1", True, float(i), question_id=f"Q{i}")
        for i in range(1, 4)
    ]
    report = _report(_Profile("learner-raise", kp_mastery={"2.1.1": 0.8}), records, response_count=3)

    assert report["difficulty_decision"]["decision"] == "RAISE"
    assert report["ability"]["status"] == "MODEL_INFERRED"
    assert report["ability"]["theta"] == 1.1


def test_practice_validates_contextual_strategy_but_does_not_write_mastery() -> None:
    profile = _Profile("learner-effect", kp_mastery={"2.3.2": 0.44})
    service = _ProfileService(profile)
    before = dict(profile.kp_mastery)
    event = PracticeValidationEvent(
        event_id="practice-validation-1",
        learner_id=profile.learner_id,
        task_id="task-1",
        resource_id="resource-1",
        question_id="Q040",
        kp_id="2.3.2",
        concept_ids=("concept:dy3:concentration-quenching",),
        strategy="mechanism_map",
        correct=True,
        timestamp=300.0,
    )

    assert commit_practice_validation(service, event) is True
    view = load_teaching_memory_view(service, profile.learner_id)
    strategy = next(item for item in view.strategies if item.strategy == "mechanism_map")
    assert strategy.effect_status is TeachingEffectStatus.VALIDATED_POSITIVE
    assert strategy.validated_outcomes == 1
    assert profile.kp_mastery == before


def test_match_report_exposes_authoritative_report_without_turning_unknown_into_zero(tmp_path) -> None:
    client = TestClient(UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app())
    response = client.get("/api/match-report/learner-no-real-record")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["source_class"] == "LEARNER_INTELLIGENCE_VIEW"
    assert data["report"]["difficulty_decision"]["decision"] == "DIAGNOSE_FIRST"
    assert data["overall_mastery"] is None
    assert data["theta"] is None
    assert all(item["type"] == "UNKNOWN" for item in data["blind_spots"])
    match = data["report"]["resource_difficulty_match"]
    assert match["learner_position"] is None
    assert match["learner_position_status"] == "UNKNOWN"
    assert match["authored_question_count"] > 0
    assert sum(item["question_count"] for item in match["bands"]) == match["authored_question_count"]
    assert "edges" in data["report"]["learning_path"]
