"""P0 learning-workspace product truth and safety acceptance tests."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from types import MappingProxyType

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l2.practice import PracticeBank
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.learner_intelligence import (
    build_learner_intelligence_view,
    build_public_learner_report,
)
from dy3_polaris.l5.learning_workspace import (
    build_learning_workspace_view,
    public_learning_workspace_projection,
)
from dy3_polaris.l5.unified_app import UnifiedApp, _assert_public_dto


def _workspace_for(query: str, *, practice_bank: object | None = None) -> dict:
    view = build_learner_intelligence_view(
        {"learner_id": "guest-workspace-unit", "query": query},
        AgentDependencies(),
    )
    workspace = build_learning_workspace_view(
        learner_view=view,
        learner_report=build_public_learner_report(view),
        practice_bank=practice_bank,
    )
    return public_learning_workspace_projection(workspace)


def test_unknown_learner_can_ask_without_fake_mastery_or_resume() -> None:
    data = _workspace_for("继续学习")

    ask = next(item for item in data["quick_actions"] if item["action_type"] == "ASK")
    assert ask["status"] == "AVAILABLE"
    assert data["lifecycle_stage"] == "UNKNOWN_LEARNER"
    assert data["learner_summary"]["observed_record_count"] == 0
    assert data["learner_summary"]["modelled_kp_count"] == 0
    assert data["resume_action"] is None
    analysis = data["initial_profile_analysis"]
    assert analysis["status"] == "DIAGNOSTIC_REQUIRED"
    assert analysis["evidence_basis"] == "UNKNOWN"
    assert len(analysis["candidates"]) == 3
    selected = next(item for item in analysis["candidates"] if item["selected"])
    assert selected["candidate_id"] == "foundation_scaffold"
    assert selected["diagnostic_required"] is True


def test_capability_eligibility_uses_real_authored_practice_coverage() -> None:
    with_bank = _workspace_for("解释Dy³⁺能级跃迁", practice_bank=PracticeBank())
    without_bank = _workspace_for(
        "解释Dy³⁺能级跃迁",
        practice_bank=SimpleNamespace(by_kp={}),
    )

    assert any(item["authored_practice_available"] for item in with_bank["capability_coverage"])
    assert all(not item["authored_practice_available"] for item in without_bank["capability_coverage"])
    practice_actions = [
        item for item in without_bank["quick_actions"]
        if item["action_type"] == "PRACTICE"
    ]
    assert practice_actions
    assert practice_actions[0]["status"] == "UNAVAILABLE"
    assert practice_actions[0]["route"] == ""


def test_all_relation_derived_prerequisite_blockers_are_visible() -> None:
    data = _workspace_for("4f-4f跃迁如何产生Dy³⁺可见发射？", practice_bank=PracticeBank())
    blocker_ids = [item["concept_id"] for item in data["blocking_prerequisites"]]
    required_ids = [
        item["concept_id"] for item in data["learning_sequence"]
        if item["status"] == "REQUIRED"
    ]

    assert blocker_ids
    assert required_ids == blocker_ids
    assert len(required_ids) == len(set(required_ids))


def test_evidence_mentions_are_not_promoted_to_released_support() -> None:
    data = _workspace_for("解释Dy³⁺能级跃迁", practice_bank=PracticeBank())
    statuses = {item["evidence_status"] for item in data["capability_coverage"]}

    assert "RELEASED_TASK_EVIDENCE" not in statuses
    assert statuses <= {"NONE", "MENTION_CANDIDATE_ONLY"}


def test_public_dto_guard_rejects_nested_runtime_objects() -> None:
    safe = {"safe": [1, "two", None, {"ok": True}]}
    _assert_public_dto(safe)
    _assert_public_dto(copy.deepcopy(safe))
    assert "_contract_candidate" not in repr(safe)

    with pytest.raises(RuntimeError, match="non-serializable runtime object"):
        _assert_public_dto({"unsafe": SimpleNamespace(secret="private")})
    with pytest.raises(RuntimeError, match="non-serializable runtime object"):
        _assert_public_dto(MappingProxyType({"unsafe": SimpleNamespace()}))

    class _ModelDumpTrap:
        called = False

        def model_dump(self):
            self.called = True
            return {"private": "leaked"}

    trap = _ModelDumpTrap()
    with pytest.raises(RuntimeError, match="non-serializable runtime object"):
        _assert_public_dto({"model": trap})
    assert trap.called is False


def test_workspace_endpoint_and_practice_feedback_are_real_and_private_safe(tmp_path) -> None:
    builder = UnifiedApp.create_full_app_builder(data_dir=str(tmp_path))
    client = TestClient(builder.create_app())
    learner_id = "guest-workspace-api"

    before = client.get(f"/api/learning-workspace/{learner_id}")
    assert before.status_code == 200
    before_data = before.json()["data"]
    assert before_data["continuity"] == "SAME_DEVICE"
    assert before_data["cross_device_continuity"] == "PARTIAL"
    assert before_data["recent_changes"] == []

    question_response = client.get(
        "/l2/practice/questions",
        params={"learner_id": learner_id, "count": 1},
    )
    assert question_response.status_code == 200
    question = question_response.json()["data"]["questions"][0]
    answer = client.post(
        "/l2/practice/answer",
        json={
            "learner_id": learner_id,
            "qid": question["qid"],
            "selected": -1,
            "attempt_purpose": "ROUTE_VERIFY",
        },
    )
    assert answer.status_code == 200
    answer_data = answer.json()["data"]
    assert answer_data["attempt_purpose"] == "ROUTE_VERIFY"
    assert answer_data["answer_saved"] is True
    assert answer_data["model_updated"] is False
    assert answer_data["model_update_status"] == "SKIPPED_BY_POLICY"
    assert "_runtime_metrics" not in answer_data

    after = client.get(f"/api/learning-workspace/{learner_id}")
    assert after.status_code == 200
    after_data = after.json()["data"]
    assert after_data["learner_summary"]["observed_record_count"] == 1
    assert any(item["event_type"] == "PRACTICE_RESULT" for item in after_data["recent_changes"])
    assert not any(
        "VERIFIED_WEAKNESS" in str(item)
        for item in after_data["recent_changes"]
    )

    serialized = after.text
    for forbidden in (
        "LearnerIntelligenceView",
        "KnowledgeLearningContext",
        "TeachingMemoryView",
        "_runtime_metrics",
        "_contract_candidate",
    ):
        assert forbidden not in serialized

    metrics = builder._handlers.runtime_measurement_summary()
    for metric in (
        "practice_submit_ms",
        "answer_record_write_ms",
        "bkt_update_ms",
        "learner_view_build_ms",
        "learner_report_build_ms",
        "workspace_projection_build_ms",
    ):
        assert metrics[metric]["status"] == "OBSERVED"
    assert metrics["profile_update_ms"]["status"] == "NOT_OBSERVED"
    practice_record = next(
        item for item in builder._handlers._runtime_measurements
        if item["operation"] == "practice_submit"
    )
    correlation = practice_record["correlation"]
    assert correlation["attempt_purpose"] == "ROUTE_VERIFY"
    assert correlation["answer_record_id"].startswith("answer-")
    assert learner_id not in str(correlation)


def test_practice_attempt_purpose_is_bounded(tmp_path) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )
    question = client.get(
        "/l2/practice/questions",
        params={"learner_id": "purpose-check", "count": 1},
    ).json()["data"]["questions"][0]

    invalid = client.post(
        "/l2/practice/answer",
        json={
            "learner_id": "purpose-check",
            "qid": question["qid"],
            "selected": -1,
            "attempt_purpose": "MAKE_UP_A_RESULT",
        },
    )
    assert invalid.status_code == 400


def test_runtime_measurements_are_content_free_and_honest(tmp_path) -> None:
    builder = UnifiedApp.create_full_app_builder(data_dir=str(tmp_path))
    client = TestClient(builder.create_app())
    client.get("/api/learning-workspace/metrics-check")

    summary = builder._handlers.runtime_measurement_summary()
    assert summary["learner_view_build_ms"]["status"] == "OBSERVED"
    assert summary["workspace_projection_build_ms"]["sample_count"] == 1
    assert summary["retrieval_ms"]["status"] == "NOT_OBSERVED"
    assert all(
        set(record) == {"task_id", "operation", "timestamp", "measurements", "correlation"}
        for record in builder._handlers._runtime_measurements
    )
    assert all(
        "metrics-check" not in str(record.get("correlation"))
        for record in builder._handlers._runtime_measurements
    )
