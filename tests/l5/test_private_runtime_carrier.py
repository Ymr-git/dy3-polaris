"""P1 private runtime carrier type and serialization guard tests."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json

from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from dy3_polaris.l5 import agent_workers
from dy3_polaris.l5 import task_state as task_state_runtime
from dy3_polaris.l5.agent_workers import (
    AgentDependencies,
    _AnswerCorrelation,
    _EvidenceCandidate,
    _FinalPrivateCandidateSet,
    _PrivateRuntimeCarrier,
    _ReviewCandidate,
    _run_multi_candidate_generation,
)
from dy3_polaris.l5.api.router import _safe_dump
from dy3_polaris.l5.interaction_recorder import (
    InteractionPhase,
    InteractionRecorder,
)
from dy3_polaris.l5.unified_app import UnifiedApp
from dy3_polaris.shared.contract import ok


_PRIVATE_SENTINEL = "P1_PRIVATE_CANDIDATE_MUST_NOT_SERIALIZE"


def _carrier(public_data: dict[str, object]) -> _PrivateRuntimeCarrier:
    carrier = _PrivateRuntimeCarrier(public_data)
    carrier._contract_candidate = {
        "sentinel": _PRIVATE_SENTINEL,
        "private": True,
    }
    return carrier


def _selected_generation(
    answer: str,
    task_id: str,
    *,
    stage: str = "selected",
    knowledge_unavailable: bool = False,
    honest_unavailable: bool = False,
) -> _PrivateRuntimeCarrier:
    public = {
        "agent_id": agent_workers.GENERATION_AGENT_ID,
        "status": "completed",
        "query": "Dy3+ 浓度猝灭机理",
        "answer": answer,
        "confidence": 0.8,
        "context_chunks": [f"Dy3+ evidence for {answer}"],
        "citations": [f"citation for {answer}"],
        "sources": [{"chunk_id": f"source-{answer}", "entity": "Dy3+"}],
        "knowledge_unavailable": knowledge_unavailable,
        "honest_unavailable": honest_unavailable,
        "question_type": "mechanism",
        "candidates": [],
        "consensus_score": 1.0,
        "consensus_reached": True,
        "consensus_threshold": 0.5,
        "divergence_matrix": [],
        "agree_pairs": 0,
        "total_pairs": 0,
        "debate": None,
        "selected_candidate": "A",
        "needs_adjudication": False,
    }
    return agent_workers._attach_selected_evidence_candidate(
        public,
        {"task_id": task_id},
        stage=stage,
    )


def _review(verdict: str) -> dict[str, object]:
    return {
        "agent_id": agent_workers.REVIEW_AGENT_ID,
        "status": "completed",
        "verdict": verdict,
        "reason": f"review {verdict}",
        "confidence": 0.9,
    }


def _review_candidate(
    task_id: str,
    content: str,
    verdict: str = "approved",
) -> _PrivateRuntimeCarrier:
    return agent_workers._attach_review_candidate(
        _review(verdict),
        {"task_id": task_id},
        content=content,
        producer="agent.quality.review/run_review",
        real_reviewer_executed=True,
    )


def _no_critic_adoption(_input, _deps, generation):
    return {
        "adopted": False,
        "generation": generation,
        "rounds": [],
        "final_verdict": "pass",
        "final_score": 1.0,
        "reason": "",
    }


def _answer_correlation(result) -> _AnswerCorrelation:
    private_set = result._contract_candidate
    assert isinstance(private_set, _FinalPrivateCandidateSet)
    assert isinstance(private_set.answer_correlation, _AnswerCorrelation)
    return private_set.answer_correlation


def _guidance_input(task_id: str) -> dict[str, object]:
    task_context = task_state_runtime.create_task_context(task_id)
    task_state_runtime.set_task_state(task_context, "UNDERSTANDING")
    return {
        "query": "Dy3+ 浓度猝灭机理",
        "task_id": task_id,
        "task_context": task_context,
    }


def _stub_guidance_edges(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_workers,
        "run_diagnosis",
        lambda _input, _deps: {
            "agent_id": agent_workers.DIAGNOSIS_AGENT_ID,
            "status": "completed",
            "weak_kps": [],
            "confidence": 0.9,
        },
    )
    monkeypatch.setattr(
        agent_workers,
        "get_recorder",
        lambda: InteractionRecorder(),
    )


def test_carrier_preserves_plain_dict_contract() -> None:
    public = {
        "answer": "public answer",
        "confidence": 0.75,
        "review": {"verdict": "approved"},
    }
    carrier = _carrier(public)

    assert isinstance(carrier, dict)
    assert not hasattr(carrier, "__dict__")
    assert set(carrier) == set(public)
    assert dict(carrier) == public
    assert carrier == public
    assert carrier.get("answer") == "public answer"
    assert carrier["confidence"] == 0.75
    assert "_contract_candidate" not in carrier
    assert _PRIVATE_SENTINEL not in repr(carrier)
    assert _PRIVATE_SENTINEL not in str(carrier)


def test_standard_json_and_response_envelope_hide_private_slot() -> None:
    generation = _carrier({
        "answer": "generated answer",
        "context_chunks": ["chunk-1"],
    })
    review = _carrier({
        "verdict": "approved",
        "reason": "public reason",
    })
    guidance = _carrier({
        "answer": "generated answer",
        "pipeline": [generation, review],
        "review": review,
    })

    encoded = json.dumps(guidance, ensure_ascii=False)
    response_body = JSONResponse(ok(guidance)).body.decode("utf-8")

    assert _PRIVATE_SENTINEL not in encoded
    assert _PRIVATE_SENTINEL not in response_body
    assert json.loads(encoded) == dict(guidance)
    assert json.loads(response_body) == {
        "code": 0,
        "data": dict(guidance),
        "message": "",
    }


def test_safe_dump_preserves_public_items_and_hides_private_slot() -> None:
    generation = _carrier({
        "answer": "generated answer",
        "sources": [{"chunk_id": "chunk-1"}],
    })
    review = _carrier({
        "verdict": "needs_review",
        "confidence": 0.6,
    })
    guidance = _carrier({
        "answer": "generated answer",
        "pipeline": [generation, review],
        "review": review,
    })

    dumped = _safe_dump({"outputs": {"guidance": guidance}})
    serialized = json.dumps(dumped, ensure_ascii=False)

    assert dumped == {"outputs": {"guidance": dict(guidance)}}
    assert dumped["outputs"]["guidance"] != {}
    assert _PRIVATE_SENTINEL not in serialized


def test_interaction_recorder_only_records_public_mapping_items() -> None:
    generation = _carrier({
        "answer": "generated answer",
        "sources": [{"chunk_id": "chunk-1"}],
    })
    review = _carrier({
        "verdict": "approved",
        "reason": "public reason",
    })
    guidance = _carrier({
        "answer": "generated answer",
        "pipeline": [generation, review],
        "review": review,
    })
    recorder = InteractionRecorder()
    chain_id = recorder.start_chain(query="carrier serialization test")

    recorder.record_agent_execution(
        agent_id="agent.guidance.decision",
        agent_name="guidance",
        action="carrier serialization",
        output_data=guidance,
        phase=InteractionPhase.DECISION,
        chain_id=chain_id,
    )

    records = recorder.get_records_by_chain(chain_id)
    assert len(records) == 1
    serialized = json.dumps(records, ensure_ascii=False, default=str)
    assert "generated answer" in serialized
    assert "approved" in serialized
    assert _PRIVATE_SENTINEL not in serialized


def test_plain_dict_reconstruction_drops_private_slot_by_design() -> None:
    carrier = _carrier({"answer": "public answer"})

    reconstructed = dict(carrier)
    copied = carrier.copy()

    assert reconstructed == {"answer": "public answer"}
    assert copied == {"answer": "public answer"}
    assert not hasattr(reconstructed, "_contract_candidate")
    assert not hasattr(copied, "_contract_candidate")


def test_selected_generation_attaches_private_evidence_candidate(
    monkeypatch,
) -> None:
    def fake_run_generation(input_data, _deps):
        candidate_id = input_data["strategy_id"]
        return {
            "agent_id": "agent.knowledge.generation",
            "status": "completed",
            "answer": "selected public answer",
            "confidence": 0.8,
            "context_chunks": [f"chunk-{candidate_id}"],
            "citations": [f"citation-{candidate_id}"],
            "sources": [{"chunk_id": f"source-{candidate_id}"}],
            "knowledge_unavailable": False,
        }

    monkeypatch.setattr(agent_workers, "_detect_ambiguity", lambda _query: None)
    monkeypatch.setattr(agent_workers, "run_generation", fake_run_generation)

    result = _run_multi_candidate_generation(
        {"query": "Dy3+ 浓度猝灭机理", "task_id": "task-carrier-selected"},
        AgentDependencies(),
    )

    assert isinstance(result, _PrivateRuntimeCarrier)
    assert "_contract_candidate" not in result
    assert set(result) == {
        "agent_id",
        "status",
        "query",
        "answer",
        "confidence",
        "context_chunks",
        "citations",
        "sources",
        "knowledge_unavailable",
        "question_type",
        "candidates",
        "consensus_score",
        "consensus_reached",
        "consensus_threshold",
        "divergence_matrix",
        "agree_pairs",
        "total_pairs",
        "debate",
        "selected_candidate",
        "needs_adjudication",
    }
    candidate = result._contract_candidate
    assert isinstance(candidate, _EvidenceCandidate)
    assert {field.name for field in fields(candidate)} == {
        "task_id",
        "producer",
        "stage",
        "answer_identity",
        "context_chunks",
        "citations",
        "sources",
        "knowledge_unavailable",
            "honest_unavailable",
            "evidence_versions",
        }
    assert candidate.task_id == "task-carrier-selected"
    assert candidate.stage == "selected"
    assert candidate.answer_identity == hashlib.sha256(
        b"task-carrier-selectedselected public answer"
    ).hexdigest()
    assert candidate.context_chunks == ("chunk-A",)
    assert candidate.citations == ("citation-A",)
    assert candidate.sources == ({"chunk_id": "source-A"},)
    assert candidate.knowledge_unavailable is False
    assert candidate.honest_unavailable is False
    assert "_contract_candidate" not in dict(result)


def test_clarify_selected_return_attaches_private_evidence_candidate(
    monkeypatch,
) -> None:
    clarify = {"reason": "query is ambiguous", "options": ["A", "B"]}
    monkeypatch.setattr(agent_workers, "_detect_ambiguity", lambda _query: clarify)

    result = _run_multi_candidate_generation(
        {"query": "请详细说明", "task_id": "task-carrier-clarify"},
        AgentDependencies(),
    )

    assert isinstance(result, _PrivateRuntimeCarrier)
    assert result["clarify"] == clarify
    assert "_contract_candidate" not in result
    candidate = result._contract_candidate
    assert isinstance(candidate, _EvidenceCandidate)
    assert candidate.task_id == "task-carrier-clarify"
    assert candidate.stage == "clarify"
    assert candidate.context_chunks == ()
    assert candidate.citations == ()
    assert candidate.sources == ()
    assert candidate.knowledge_unavailable is False
    assert candidate.honest_unavailable is False


def test_terminal_selected_return_attaches_private_evidence_candidate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(agent_workers, "_detect_ambiguity", lambda _query: None)
    monkeypatch.setattr(
        agent_workers,
        "run_generation",
        lambda _input, _deps: {
            "agent_id": "agent.knowledge.generation",
            "status": "completed",
            "answer": "plain terminal answer",
            "confidence": 0.8,
            "context_chunks": [],
            "citations": [],
            "honest_unavailable": True,
        },
    )

    result = _run_multi_candidate_generation(
        {"query": "用大白话解释", "task_id": "task-carrier-terminal"},
        AgentDependencies(),
    )

    assert isinstance(result, _PrivateRuntimeCarrier)
    assert result["answer"] == "plain terminal answer"
    assert "_contract_candidate" not in result
    candidate = result._contract_candidate
    assert isinstance(candidate, _EvidenceCandidate)
    assert candidate.task_id == "task-carrier-terminal"
    assert candidate.stage == "terminal"
    assert candidate.context_chunks == ()
    assert candidate.citations == ()
    assert candidate.sources == ()
    assert candidate.knowledge_unavailable is False
    assert candidate.honest_unavailable is True


def test_plain_projection_with_real_evidence_remains_a_selected_candidate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(agent_workers, "_detect_ambiguity", lambda _query: None)
    monkeypatch.setattr(
        agent_workers,
        "run_generation",
        lambda _input, _deps: {
            "agent_id": "agent.knowledge.generation",
            "status": "completed",
            "answer": "evidence-backed plain explanation",
            "confidence": 0.8,
            "context_chunks": ["real supporting chunk"],
            "citations": ["document-real"],
            "sources": [{"document_id": "document-real"}],
            "plain_language": True,
        },
    )

    result = _run_multi_candidate_generation(
        {"query": "用通俗语言解释浓度猝灭", "task_id": "task-plain-evidence"},
        AgentDependencies(),
    )

    candidate = result._contract_candidate
    assert candidate.stage == "selected"
    assert candidate.context_chunks == ("real supporting chunk",)
    assert candidate.citations == ("document-real",)


def test_normal_run_review_attaches_private_review_candidate() -> None:
    public_keys = {
        "agent_id",
        "status",
        "verdict",
        "reason",
        "fact_check",
        "anti_hallucination",
        "confidence",
    }
    result = agent_workers.run_review(
        {
            "task_id": "task-review-normal",
            "content": "reviewed answer",
            "context_chunks": ["supporting chunk"],
        },
        AgentDependencies(),
    )

    assert isinstance(result, _PrivateRuntimeCarrier)
    assert set(result) == public_keys
    assert "_contract_candidate" not in result
    candidate = result._contract_candidate
    assert isinstance(candidate, _ReviewCandidate)
    assert candidate.task_id == "task-review-normal"
    assert candidate.producer == "agent.quality.review/run_review"
    assert candidate.reviewed_answer_identity == hashlib.sha256(
        b"task-review-normalreviewed answer"
    ).hexdigest()
    assert candidate.raw_status == result["status"]
    assert candidate.raw_verdict == result["verdict"]
    assert candidate.raw_reason == result["reason"]
    assert candidate.raw_fact_check == result["fact_check"]
    assert (
        candidate.raw_anti_hallucination
        == result["anti_hallucination"]
    )
    assert candidate.raw_confidence == result["confidence"]
    assert candidate.real_reviewer_executed is True
    assert candidate.mapping_refused_reason == ""


def test_skipped_run_review_keeps_public_result_and_refuses_mapping() -> None:
    result = agent_workers.run_review(
        {"task_id": "task-review-skipped", "content": ""},
        AgentDependencies(),
    )

    assert isinstance(result, _PrivateRuntimeCarrier)
    assert dict(result) == {
        "agent_id": agent_workers.REVIEW_AGENT_ID,
        "status": "skipped",
        "verdict": "skipped",
        "reason": "内容为空，跳过审核",
        "confidence": 1.0,
    }
    candidate = result._contract_candidate
    assert isinstance(candidate, _ReviewCandidate)
    assert candidate.producer == "skipped"
    assert candidate.reviewed_answer_identity == ""
    assert candidate.raw_status == "skipped"
    assert candidate.raw_verdict == "skipped"
    assert candidate.real_reviewer_executed is False
    assert candidate.mapping_refused_reason == "no reviewed content"


def test_terminal_synthetic_review_is_private_and_not_real(
    monkeypatch,
) -> None:
    _stub_guidance_edges(monkeypatch)
    terminal = _selected_generation(
        "honest terminal answer",
        "task-review-synthetic",
        stage="terminal",
        honest_unavailable=True,
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: terminal,
    )

    result = agent_workers.run_guidance(
        _guidance_input("task-review-synthetic"),
        AgentDependencies(),
    )

    review = result["pipeline"][2]
    assert review == {
        "agent_id": agent_workers.REVIEW_AGENT_ID,
        "status": "not_completed",
        "verdict": "skipped",
        "contribution": "scientific_quality_decision",
    }
    candidate = result._contract_candidate.review_candidate
    assert isinstance(candidate, _ReviewCandidate)
    assert candidate.producer == "synthetic_review"
    assert candidate.reviewed_answer_identity == ""
    assert candidate.raw_verdict == "skipped"
    assert candidate.real_reviewer_executed is False
    assert candidate.mapping_refused_reason == "real reviewer not executed"
    correlation = _answer_correlation(result)
    assert correlation.correlation is False


def test_initial_answer_evidence_and_review_identities_correlate(
    monkeypatch,
) -> None:
    _stub_guidance_edges(monkeypatch)
    task_id = "task-correlation-initial"
    generation = _selected_generation("initial answer", task_id)
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: generation,
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda input_data, _deps: _review_candidate(
            task_id,
            str(input_data.get("content") or ""),
        ),
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        _no_critic_adoption,
    )

    result = agent_workers.run_guidance(
        _guidance_input(task_id),
        AgentDependencies(),
    )

    private_set = result._contract_candidate
    assert isinstance(private_set, _FinalPrivateCandidateSet)
    assert private_set.evidence_candidate is generation._contract_candidate
    assert isinstance(private_set.review_candidate, _ReviewCandidate)
    correlation = _answer_correlation(result)
    assert correlation.correlation is True
    assert (
        correlation.final_answer_identity
        == generation._contract_candidate.answer_identity
        == private_set.review_candidate.reviewed_answer_identity
    )
    assert correlation.refusal_reasons == ()


def test_self_correction_adopts_generation2_candidate(monkeypatch) -> None:
    _stub_guidance_edges(monkeypatch)
    initial = _selected_generation("initial answer", "task-self-adopt")
    revised = _selected_generation("revised answer", "task-self-adopt")
    generations = iter((initial, revised))
    reviews = iter((
        _review_candidate(
            "task-self-adopt",
            "initial answer",
            "needs_review",
        ),
        _review_candidate("task-self-adopt", "revised answer"),
    ))
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: next(generations),
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda *_args, **_kwargs: next(reviews),
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        _no_critic_adoption,
    )

    result = agent_workers.run_guidance(
        _guidance_input("task-self-adopt"),
        AgentDependencies(),
    )

    private_set = result._contract_candidate
    assert private_set.evidence_candidate is revised._contract_candidate
    correlation = _answer_correlation(result)
    assert correlation.correlation is True
    assert (
        correlation.final_answer_identity
        == revised._contract_candidate.answer_identity
        == private_set.review_candidate.reviewed_answer_identity
    )


def test_self_correction_rejection_keeps_initial_candidate(monkeypatch) -> None:
    _stub_guidance_edges(monkeypatch)
    initial = _selected_generation("initial answer", "task-self-keep")
    revised = _selected_generation("rejected revision", "task-self-keep")
    generations = iter((initial, revised))
    reviews = iter((
        _review_candidate(
            "task-self-keep",
            "initial answer",
            "needs_review",
        ),
        _review_candidate(
            "task-self-keep",
            "rejected revision",
            "rejected",
        ),
    ))
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: next(generations),
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda *_args, **_kwargs: next(reviews),
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        _no_critic_adoption,
    )

    result = agent_workers.run_guidance(
        _guidance_input("task-self-keep"),
        AgentDependencies(),
    )

    private_set = result._contract_candidate
    assert private_set.evidence_candidate is initial._contract_candidate
    assert result["pipeline"][2]["verdict"] == "needs_review"
    correlation = _answer_correlation(result)
    assert correlation.correlation is False
    assert (
        initial._contract_candidate.answer_identity
        == private_set.review_candidate.reviewed_answer_identity
    )
    assert result["answer"] == ""
    assert "evidence answer identity mismatch" in correlation.refusal_reasons


def test_legacy_critic_cannot_become_second_correction_source(monkeypatch) -> None:
    _stub_guidance_edges(monkeypatch)
    initial = _selected_generation("initial answer", "task-critic-adopt")
    critic = _selected_generation("critic answer", "task-critic-adopt")
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: initial,
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda input_data, _deps: _review_candidate(
            "task-critic-adopt",
            str(input_data.get("content") or ""),
        ),
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        lambda *_args, **_kwargs: {
            "adopted": True,
            "generation": critic,
            "rounds": [{"verdict": "pass"}],
            "final_verdict": "pass",
            "final_score": 1.0,
            "reason": "",
        },
    )

    result = agent_workers.run_guidance(
        _guidance_input("task-critic-adopt"),
        AgentDependencies(),
    )

    private_set = result._contract_candidate
    assert private_set.evidence_candidate is initial._contract_candidate
    correlation = _answer_correlation(result)
    assert correlation.correlation is True
    assert (
        correlation.final_answer_identity
        == initial._contract_candidate.answer_identity
        == private_set.review_candidate.reviewed_answer_identity
    )


def test_critic_non_adoption_keeps_current_candidate(monkeypatch) -> None:
    _stub_guidance_edges(monkeypatch)
    initial = _selected_generation("initial answer", "task-critic-keep")
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: initial,
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda input_data, _deps: _review_candidate(
            "task-critic-keep",
            str(input_data.get("content") or ""),
        ),
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        _no_critic_adoption,
    )

    result = agent_workers.run_guidance(
        _guidance_input("task-critic-keep"),
        AgentDependencies(),
    )

    private_set = result._contract_candidate
    assert private_set.evidence_candidate is initial._contract_candidate
    assert _answer_correlation(result).correlation is True


def test_guidance_does_not_modify_answer_after_review(monkeypatch) -> None:
    _stub_guidance_edges(monkeypatch)
    task_id = "task-correlation-modified"
    generation = _selected_generation("reviewed answer", task_id)
    generation["needs_adjudication"] = True
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: generation,
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda input_data, _deps: _review_candidate(
            task_id,
            str(input_data.get("content") or ""),
        ),
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        _no_critic_adoption,
    )

    result = agent_workers.run_guidance(
        _guidance_input(task_id),
        AgentDependencies(),
    )

    assert result["answer"] == generation["answer"]
    correlation = _answer_correlation(result)
    assert correlation.correlation is True


def test_skipped_review_does_not_correlate(monkeypatch) -> None:
    _stub_guidance_edges(monkeypatch)
    task_id = "task-correlation-skipped"
    generation = _selected_generation("", task_id)
    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        lambda *_args, **_kwargs: generation,
    )

    result = agent_workers.run_guidance(
        _guidance_input(task_id),
        AgentDependencies(),
    )

    review = result["pipeline"][2]
    assert review["verdict"] == "skipped"
    correlation = _answer_correlation(result)
    assert correlation.correlation is False
    assert "real reviewer not executed" in correlation.refusal_reasons
    assert "no reviewed content" in correlation.refusal_reasons


def test_query_agent_and_orchestration_apis_do_not_leak_candidates(
    monkeypatch,
) -> None:
    _stub_guidance_edges(monkeypatch)
    real_run_review = agent_workers.run_review

    def selected(input_data, *_args, **_kwargs):
        return _selected_generation(
            "private carrier API answer",
            str(input_data.get("task_id") or ""),
        )

    monkeypatch.setattr(
        agent_workers,
        "_run_multi_candidate_generation",
        selected,
    )
    monkeypatch.setattr(
        agent_workers,
        "run_review",
        lambda input_data, _deps: real_run_review(
            input_data,
            AgentDependencies(),
        ),
    )
    monkeypatch.setattr(
        agent_workers,
        "_run_critic_loop",
        _no_critic_adoption,
    )
    builder = UnifiedApp.create_full_app_builder()
    client = TestClient(builder.create_app())

    query_response = client.post(
        "/api/query",
        json={"query": "Dy3+ 浓度猝灭机理"},
    )
    assert query_response.status_code == 200

    login = client.post(
        "/l1/api/v1/auth/login",
        json={"student_id": "DY20248888", "password": "admin888"},
    )
    headers = {
        "Authorization": "Bearer "
        + login.json()["data"]["access_token"]
    }
    agent_response = client.post(
        f"/l5/agents/{agent_workers.GUIDANCE_AGENT_ID}/run",
        json={"query": "Dy3+ 浓度猝灭机理"},
        headers=headers,
    )
    assert agent_response.status_code == 200

    review_response = client.post(
        f"/l5/agents/{agent_workers.REVIEW_AGENT_ID}/run",
        json={
            "task_id": "task-review-api",
            "content": "review API answer",
        },
        headers=headers,
    )
    assert review_response.status_code == 200

    orchestration_response = client.post(
        "/l5/orchestrate",
        json={
            "paradigm": "pipeline",
            "tasks": [
                {
                    "task_id": "guidance-task",
                    "agent_id": agent_workers.GUIDANCE_AGENT_ID,
                    "input": {
                        "task_id": "task-guidance-orchestrate",
                        "query": "Dy3+ 浓度猝灭机理",
                    },
                }
            ],
        },
        headers=headers,
    )
    assert orchestration_response.status_code == 200

    for response in (
        query_response,
        agent_response,
        review_response,
        orchestration_response,
    ):
        serialized = response.text
        assert "EvidenceCandidate" not in serialized
        assert "ReviewCandidate" not in serialized
        assert "_contract_candidate" not in serialized
        assert "answer_identity" not in serialized
        assert "reviewed_answer_identity" not in serialized
        assert '"correlation"' not in serialized
        assert '"readiness"' not in serialized
