"""R-06D Knowledge/Learner/Agent fusion protection tests."""
from __future__ import annotations

import json

from starlette.testclient import TestClient

from dy3_polaris.l3.concept_foundation import build_concept_foundation
from dy3_polaris.l3.concept_relations import (
    ConceptRelationNetwork,
    build_concept_relation_network,
)
from dy3_polaris.l3.models import DocumentChunk
from dy3_polaris.l3.store import KnowledgeStore
from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5.agent_workers import AgentDependencies
from dy3_polaris.l5.collaboration_context import initialize_collaboration_context
from dy3_polaris.l5.knowledge_learning_fusion import (
    KnowledgeLearningContext,
    build_knowledge_learning_context,
    project_concept_mastery,
)
from dy3_polaris.l5.learner_intelligence import build_learner_intelligence_view
from dy3_polaris.l5.task_understanding import understand_task
from dy3_polaris.l5.unified_app import UnifiedApp
from tests.l5.test_learner_memory import _ProfileService, _run_task


_TRANSITION_QUERY = "4f-4f跃迁如何产生Dy³⁺可见发射？"


def test_kp_mapping_projects_mastery_to_a_distinct_concept() -> None:
    foundation = build_concept_foundation()
    projection = project_concept_mastery(
        foundation,
        {"2.3.2": 0.82},
    )["concept:dy3:concentration-quenching"]

    assert projection.concept_id != "2.3.2"
    assert projection.kp_ids == ("2.3.2",)
    assert projection.value == 0.82
    assert projection.source == "aligned_kp_mastery"


def test_missing_prerequisite_is_selected_from_concept_relation_order() -> None:
    context = build_knowledge_learning_context(
        learner_id="learner-foundation",
        query=_TRANSITION_QUERY,
    )
    path = context.learning_path

    assert path.next_concept == "concept:dy3:rare-earth-electron-configuration"
    assert path.prerequisite_gap == (
        "concept:dy3:rare-earth-electron-configuration",
        "concept:dy3:four-f-shell",
        "concept:dy3:dy3-energy-level-structure",
    )
    assert "relation:dy3:electron-config-prereq-four-f" in path.reason
    assert any(item.startswith("prerequisite:relation:dy3:") for item in context.trace)


def test_active_misconception_changes_path_through_scientific_relation() -> None:
    base = build_knowledge_learning_context(
        learner_id="learner-health-base",
        query="健康照明需要评价哪些指标？",
    )
    corrected = build_knowledge_learning_context(
        learner_id="learner-health-misconception",
        query="健康照明需要评价哪些指标？",
        misconceptions=({
            "status": "ACTIVE",
            "topic": "健康照明",
            "belief": "3000K一定健康",
        },),
    )

    assert base.learning_path.next_concept == "concept:dy3:green-healthy-lighting"
    assert corrected.learning_path.next_concept != base.learning_path.next_concept
    assert corrected.learning_path.next_concept in {
        "concept:dy3:correlated-color-temperature",
        "concept:dy3:color-rendering-index",
        "concept:dy3:blue-light-hazard",
    }
    assert "evaluated_by" in corrected.learning_path.reason
    assert "healthy-lighting-evaluated-by" in corrected.learning_path.reason


def test_different_mastery_changes_route_without_changing_the_question() -> None:
    foundation = build_knowledge_learning_context(
        learner_id="learner-weak",
        query=_TRANSITION_QUERY,
    )
    advanced = build_knowledge_learning_context(
        learner_id="learner-strong",
        query=_TRANSITION_QUERY,
        mastery={"1.1.1": 0.9, "1.1.2": 0.9, "2.1.1": 0.9},
    )

    assert foundation.learning_path.next_concept == (
        "concept:dy3:rare-earth-electron-configuration"
    )
    assert advanced.learning_path.current_position == (
        "concept:dy3:four-f-four-f-transition"
    )
    assert advanced.learning_path.next_concept == "concept:dy3:dy3-blue-emission"
    assert foundation.learning_path.next_concept != advanced.learning_path.next_concept


def test_same_question_changes_generation_depth_through_diagnosis(monkeypatch) -> None:
    weak_service = _ProfileService()
    _weak_result, _weak_diagnosis, weak_generation = _run_task(
        monkeypatch,
        service=weak_service,
        learner_id="learner-runtime-weak",
        task_id="task-r06d-weak",
        query=_TRANSITION_QUERY,
        answer="同一组经审核的科学事实。",
    )
    monkeypatch.undo()

    strong_service = _ProfileService()
    strong_profile = strong_service.ensure("learner-runtime-strong")
    strong_profile.level = "advanced"
    strong_profile.kp_mastery = {
        "1.1.1": 0.9,
        "1.1.2": 0.9,
        "2.1.1": 0.9,
    }
    _strong_result, _strong_diagnosis, strong_generation = _run_task(
        monkeypatch,
        service=strong_service,
        learner_id="learner-runtime-strong",
        task_id="task-r06d-strong",
        query=_TRANSITION_QUERY,
        answer="同一组经审核的科学事实。",
    )

    assert weak_generation[0]["learner_level"] == "foundation"
    assert strong_generation[0]["learner_level"] == "advanced"
    assert weak_generation[0]["query"] == strong_generation[0]["query"]


def test_relation_network_change_changes_route_without_fixed_learning_rule() -> None:
    default_network = build_concept_relation_network()
    relation_id = "relation:dy3:electron-config-prereq-four-f"
    changed_network = ConceptRelationNetwork(
        foundation=default_network.foundation,
        relations=tuple(
            item for item in default_network.relations
            if item.relation_id != relation_id
        ),
    )

    original = build_knowledge_learning_context(
        learner_id="learner-original-network",
        query=_TRANSITION_QUERY,
        network=default_network,
    )
    changed = build_knowledge_learning_context(
        learner_id="learner-changed-network",
        query=_TRANSITION_QUERY,
        network=changed_network,
    )

    assert original.learning_path.next_concept == (
        "concept:dy3:rare-earth-electron-configuration"
    )
    assert changed.learning_path.next_concept == "concept:dy3:four-f-shell"
    assert relation_id in " ".join(original.trace)
    assert relation_id not in " ".join(changed.trace)


def test_real_evidence_candidate_affects_confidence_but_not_prerequisite_order() -> None:
    store = KnowledgeStore()
    store.add_chunk(DocumentChunk(
        chunk_id="chunk-electron-configuration",
        document_id="paper-electron-configuration",
        content="稀土离子电子构型是理解4f壳层及其能级结构的基础。",
    ))
    without_evidence = build_knowledge_learning_context(
        learner_id="learner-no-evidence",
        query=_TRANSITION_QUERY,
    )
    with_evidence = build_knowledge_learning_context(
        learner_id="learner-with-evidence",
        query=_TRANSITION_QUERY,
        l3_store=store,
    )

    assert with_evidence.learning_path.next_concept == (
        without_evidence.learning_path.next_concept
    )
    assert "concept:dy3:rare-earth-electron-configuration" in (
        with_evidence.evidence_available_concepts
    )
    assert with_evidence.learning_path.confidence > (
        without_evidence.learning_path.confidence
    )


def test_diagnosis_consumes_fused_context_and_keeps_agent_contract_private() -> None:
    payload = {
        "task_id": "task-r06d-diagnosis",
        "learner_id": "learner-r06d-diagnosis",
        "query": _TRANSITION_QUERY,
    }
    deps = AgentDependencies()
    view = build_learner_intelligence_view(payload, deps)
    context = initialize_collaboration_context(
        payload,
        intent_resolver=lambda query, **_kwargs: understand_task(query, use_llm=False),
    )

    agent_workers._apply_diagnosis_teaching_context(context, view)
    diagnosis = agent_workers.run_diagnosis(
        {**payload, "_learner_intelligence_view": view}, deps
    )

    fused = context.learner_context["knowledge_learning_context"]
    assert isinstance(fused, KnowledgeLearningContext)
    assert context.learner_context["concept_learning_path"] is fused.learning_path
    assert diagnosis["level"] == "foundation"
    assert "Concept Relation" in diagnosis["summary"]
    assert "knowledge_learning_context" not in diagnosis
    assert "concept_learning_path" not in diagnosis


def test_current_api_and_frozen_agent_boundaries_do_not_leak_fusion(tmp_path) -> None:
    client = TestClient(
        UnifiedApp.create_full_app_builder(data_dir=str(tmp_path)).create_app()
    )
    response = client.post(
        "/api/query",
        json={
            "query": _TRANSITION_QUERY,
            "learner_id": "learner-r06d-api",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert {
        "answer", "evidence", "review", "confidence", "recommended_path",
        "task_id", "task_state",
    }.issubset(data)
    serialized = json.dumps(data, ensure_ascii=False, default=str)
    for forbidden in (
        "KnowledgeLearningContext",
        "ConceptLearningPath",
        "concept_mastery",
        "active_misconception_concepts",
        "prerequisite_gap",
        "selected_relation",
    ):
        assert forbidden not in serialized
    assert agent_workers.DIAGNOSIS_AGENT_ID == "agent.learning.diagnosis"
    assert agent_workers.GENERATION_AGENT_ID == "agent.knowledge.generation"
    assert agent_workers.REVIEW_AGENT_ID == "agent.quality.review"
    assert agent_workers.GUIDANCE_AGENT_ID == "agent.guidance.decision"
