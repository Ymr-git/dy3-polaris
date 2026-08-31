"""R-05E catalog-backed learning path intelligence tests."""

from __future__ import annotations

import json

from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.learner_intelligence import (
    LearningPath,
    build_learning_path,
)
from dy3_polaris.l5.unified_app import UnifiedApp
from tests.l5.test_learner_memory import _ProfileService, _run_task


def test_new_learner_path_is_derived_from_catalog_prerequisites() -> None:
    path = build_learning_path(
        learner_id="learner-path-new",
        query="为什么浓度升高会发生浓度猝灭并降低发光？",
    )

    assert isinstance(path, LearningPath)
    assert path.current_stage == "foundation"
    assert "2.3.2" in {item.kp_id for item in path.milestones}
    assert "2.1.1" in {item.kp_id for item in path.milestones}
    assert path.completed_nodes == ()
    assert path.recommended_nodes
    assert len(path.milestones) < 10
    assert all(item.name for item in path.milestones)


def test_same_question_changes_path_from_aligned_mastery_not_query_content() -> None:
    query = "为什么浓度升高会发生浓度猝灭并降低发光？"
    new_path = build_learning_path(learner_id="new", query=query)
    experienced_path = build_learning_path(
        learner_id="experienced",
        query=query,
        mastery={
            "A-01": 0.9,
            "A-02": 0.9,
            "A-03": 0.85,
            "A-04": 0.8,
            "A-05": 0.9,
            "A-10": 0.8,
            "A-11": 0.75,
        },
        teaching_confidence=0.8,
    )

    assert experienced_path.completed_nodes
    assert experienced_path.recommended_nodes != new_path.recommended_nodes
    assert experienced_path.recommended_nodes[0] not in experienced_path.completed_nodes
    assert "2.3.2" in experienced_path.recommended_nodes
    assert "completion uses aligned mastery only" in experienced_path.rationale


def test_active_misconception_blocks_catalog_node_without_fixed_route() -> None:
    path = build_learning_path(
        learner_id="learner-path-misconception",
        query="健康照明灯具设计需要评价哪些指标？",
        misconceptions=(
            {
                "misconception_id": "m-any",
                "topic": "6.1.2",
                "belief": "只看一个色温指标即可",
                "status": "ACTIVE",
                "confidence": 0.7,
            },
        ),
    )

    assert "6.1.2" in path.blocked_nodes
    assert "6.1.2" in path.recommended_nodes
    node = next(item for item in path.milestones if item.kp_id == "6.1.2")
    assert node.status == "BLOCKED"
    assert "misconception" in node.rationale


def test_diagnosis_path_reaches_guidance_through_private_teaching_context(
    monkeypatch,
) -> None:
    service = _ProfileService()
    profile = service.ensure("learner-path-runtime")
    profile.kp_mastery = {
        "A-01": 0.9,
        "A-02": 0.9,
        "A-03": 0.9,
        "A-04": 0.9,
        "A-05": 0.9,
        "A-10": 0.8,
        "A-11": 0.8,
    }
    result, diagnosis_inputs, _generation_inputs = _run_task(
        monkeypatch,
        service=service,
        learner_id="learner-path-runtime",
        task_id="task-path-runtime",
        query="为什么浓度升高会发生浓度猝灭并降低发光？",
        answer="浓度升高会增强能量迁移，并可能打开非辐射损失通道。",
    )
    view = diagnosis_inputs[0]["_learner_intelligence_view"]
    path = view.value("derived_context", "learning_path")
    decision = result._contract_candidate.final_collaboration_result.decision

    assert isinstance(path, LearningPath)
    assert decision.recommended_path
    assert decision.recommended_path[0]["kp_id"] == path.recommended_nodes[0]
    assert "LearningPath" not in json.dumps(dict(result), ensure_ascii=False, default=str)


def test_api_does_not_leak_learning_path_or_adaptive_context(tmp_path) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )
    response = client.post(
        "/api/query",
        json={
            "query": "浓度猝灭为什么降低Dy³⁺发光？",
            "learner_id": "learner-r05e-api",
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    serialized = json.dumps(data, ensure_ascii=False, default=str)
    for forbidden in (
        "LearningPath", "completed_nodes", "blocked_nodes",
        "AdaptiveContext", "LearnerMemory", "misconception_focus",
    ):
        assert forbidden not in serialized
    assert {
        "answer", "evidence", "review", "confidence", "recommended_path",
        "task_id", "task_state",
    }.issubset(data)
