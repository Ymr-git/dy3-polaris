"""T5 scientific Claim-Evidence truth and provenance tests."""
from __future__ import annotations

from dataclasses import replace

from dy3_polaris.l5.agent_contracts import Claim, ClaimType, EvidenceSupportLevel
from dy3_polaris.l5.retrieval_planning import EvidenceItem, EvidencePack
from dy3_polaris.l5.scientific_grounding import (
    build_scientific_grounding,
    clean_mineru_evidence,
    public_scientific_grounding_projection,
)


def _item(content: str, *, conflict: bool = False, evidence_id: str = "ev-1") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        chunk_reference=f"chunk-{evidence_id}",
        source_reference=f"paper-{evidence_id}",
        entity="Dy3+",
        material_system="NaYF4:Dy3+",
        conditions=(),
        supported_claim_types=(),
        relevance_score=0.99,
        source_quality="local_corpus",
        provenance_reference=f"prov-{evidence_id}",
        conflict_flag=conflict,
        content=content,
        rerank_reasons=("query_overlap",),
    )


def _claim(statement: str, *, identity: str = "answer-v1") -> Claim:
    return Claim(
        claim_id="claim-1",
        statement=statement,
        claim_type=ClaimType.FACT,
        confidence=0.8,
        answer_identity=identity,
    )


def _ground(claim: Claim, *items: EvidenceItem, version: int = 1, review_identity: str = "answer-v1"):
    return build_scientific_grounding(
        task_id="task-t5",
        answer_identity="answer-v1",
        claims=(claim,),
        evidence_packs=(EvidencePack(
            task_id="task-t5", subtask_id="sub-1", items=items,
            coverage=(), conflicts=(), missing_information=(), version=version,
        ),),
        review_identity=review_identity,
        reviewer_status="approved",
    )


def test_exact_atomic_claim_has_direct_support_and_source_provenance() -> None:
    statement = "Dy3+ concentration quenching reduces emission intensity at high concentration."
    grounding = _ground(_claim(statement), _item(statement))

    assert grounding.claims[0].support_status is EvidenceSupportLevel.SUPPORTS
    assert grounding.claims[0].evidence_refs == ("ev-1",)
    public = public_scientific_grounding_projection(grounding, release_eligible=True)
    assert public["claims"][0]["evidence"][0]["source"] == "paper-ev-1"
    assert public["claims"][0]["evidence"][0]["chunk_id"] == "chunk-ev-1"


def test_repeated_candidate_packs_do_not_duplicate_the_same_physical_evidence() -> None:
    statement = "Dy3+ concentration quenching reduces emission intensity at high concentration."
    first = _item(statement, evidence_id="candidate-1")
    repeated = replace(
        _item(statement, evidence_id="candidate-2"),
        source_reference=first.source_reference,
        chunk_reference=first.chunk_reference,
    )
    grounding = build_scientific_grounding(
        task_id="task-t5",
        answer_identity="answer-v1",
        claims=(_claim(statement),),
        evidence_packs=(
            EvidencePack(
                task_id="task-t5", subtask_id="candidate-a", items=(first,),
                coverage=(), conflicts=(), missing_information=(), version=1,
            ),
            EvidencePack(
                task_id="task-t5", subtask_id="candidate-b", items=(repeated,),
                coverage=(), conflicts=(), missing_information=(), version=2,
            ),
        ),
        review_identity="answer-v1",
        reviewer_status="approved",
    )

    assert len(grounding.links) == 1
    assert grounding.claims[0].evidence_refs == ("candidate-1",)
    public = public_scientific_grounding_projection(grounding, release_eligible=True)
    assert len(public["claims"][0]["evidence"]) == 1


def test_retrieval_relevance_or_single_mention_never_becomes_support() -> None:
    grounding = _ground(
        _claim("Dy3+ concentration quenching reduces emission intensity."),
        _item("This index contains the term Dy3+ only."),
    )

    assert grounding.claims[0].support_status in {
        EvidenceSupportLevel.MENTION, EvidenceSupportLevel.CANDIDATE,
        EvidenceSupportLevel.INSUFFICIENT,
    }
    assert grounding.claims[0].support_status is not EvidenceSupportLevel.SUPPORTS


def test_numeric_condition_mismatch_is_insufficient() -> None:
    grounding = _ground(
        _claim("At 2 mol% Dy3+, emission intensity reaches its maximum."),
        _item("At 8 mol% Dy3+, emission intensity reaches its maximum."),
    )

    assert grounding.claims[0].support_status is EvidenceSupportLevel.INSUFFICIENT
    assert "condition_mismatch" in grounding.issue_codes


def test_unrelated_numeric_passage_does_not_create_condition_conflict() -> None:
    statement = "At 2 mol% Dy3+, emission intensity reaches its maximum."
    grounding = _ground(
        _claim(statement),
        _item(statement, evidence_id="direct"),
        _item("A separate XRD scan used 8 mol% Eu3+.", evidence_id="unrelated"),
    )

    assert grounding.claims[0].support_status is EvidenceSupportLevel.SUPPORTS
    assert "condition_mismatch" not in grounding.issue_codes


def test_study_scoped_all_samples_is_not_a_universal_material_claim() -> None:
    statement = "The yellow peak dominated in all samples in this study."
    grounding = _ground(_claim(statement), _item(statement))

    assert grounding.claims[0].support_status is EvidenceSupportLevel.SUPPORTS
    assert "unsupported_universalization" not in grounding.issue_codes


def test_host_specific_evidence_cannot_support_universal_claim() -> None:
    grounding = _ground(
        _claim("All Dy3+ materials always show concentration quenching."),
        _item("NaYF4:Dy3+ at 4 mol% showed concentration quenching."),
    )

    assert grounding.claims[0].support_status is EvidenceSupportLevel.INSUFFICIENT
    assert "unsupported_universalization" in grounding.issue_codes


def test_conflicting_evidence_remains_conflicting_in_public_projection() -> None:
    grounding = _ground(
        _claim("Dy3+ concentration increase reduces emission intensity."),
        _item("Under the reported condition intensity increased.", conflict=True),
    )

    assert grounding.claims[0].support_status is EvidenceSupportLevel.CONFLICTS
    public = public_scientific_grounding_projection(grounding, release_eligible=True)
    assert public["claims"][0]["support_status"] == "CONFLICTS"
    assert public["claims"][0]["evidence"][0]["level"] == "CONFLICTS"


def test_answer_review_identity_mismatch_is_explicit_and_withheld_projection_hides_claims() -> None:
    grounding = _ground(_claim("Dy3+ emits visible light."), _item("Dy3+ emits visible light."), review_identity="answer-v2")

    assert grounding.identity_consistent is False
    assert "claim_evidence_review_identity_mismatch" in grounding.issue_codes
    public = public_scientific_grounding_projection(grounding, release_eligible=False)
    assert public["status"] == "WITHHELD"
    assert public["claims"] == []


def test_cleanup_preserves_original_chunk_and_source_identity() -> None:
    cleaned = clean_mineru_evidence(
        "Page 3\n12 | Dy3+ emission evidence.\nDy3+ emission evidence.\n[citation needed]",
        chunk_reference="chunk-real-7",
        source_reference="paper-real-2",
    )

    assert cleaned.chunk_reference == "chunk-real-7"
    assert cleaned.source_reference == "paper-real-2"
    assert "Page 3" in cleaned.original_text
    assert "Page 3" not in cleaned.cleaned_text
    assert "line_number_removed" in cleaned.transformations
