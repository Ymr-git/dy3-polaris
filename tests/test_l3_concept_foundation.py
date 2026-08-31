from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dy3_polaris.l3.concept_foundation import (
    ConceptType,
    EvidenceRole,
    MappingAssetType,
    MappingStatus,
    build_concept_foundation,
    canonical_concepts,
    load_curated_concept_evidence,
)
from dy3_polaris.l3.models import DocumentChunk, EntityType, KnowledgeEntity
from dy3_polaris.l3.store import KnowledgeStore


def test_canonical_concepts_have_stable_identity_and_closed_types() -> None:
    first = canonical_concepts()
    second = canonical_concepts()

    # The catalog remains a deliberately small canonical layer while covering
    # the current 48-KP curriculum.  Do not freeze an obsolete exact count.
    assert 48 <= len(first) <= 60
    assert [item.concept_id for item in first] == [item.concept_id for item in second]
    assert len({item.concept_id for item in first}) == len(first)
    assert all(item.concept_id.startswith("concept:dy3:") for item in first)
    assert all(not item.concept_id.startswith("e-") for item in first)
    assert {item.concept_type for item in first} <= set(ConceptType)
    with pytest.raises(FrozenInstanceError):
        first[0].canonical_name = "changed"  # type: ignore[misc]


def test_kp_is_a_learning_projection_not_the_concept_object() -> None:
    foundation = build_concept_foundation()
    concept = foundation.get_concept("concept:dy3:concentration-quenching")

    assert concept is not None
    assert "2.3.2" in concept.related_kps
    kp_mapping = next(
        item
        for item in foundation.mappings_for(concept.concept_id)
        if item.asset_type is MappingAssetType.KP
    )
    assert kp_mapping.asset_id == "kp-catalog:2.3.2"
    assert kp_mapping.asset_id != concept.concept_id
    assert kp_mapping.status is MappingStatus.CURATED

    shared = foundation.get_concept("concept:dy3:doping-concentration")
    assert shared is not None
    assert "2.3.2" in shared.related_kps


def test_existing_entity_maps_without_being_replaced() -> None:
    store = KnowledgeStore()
    entity = KnowledgeEntity(
        entity_id="entity-existing-cq",
        entity_type=EntityType.CONCEPT,
        name="浓度猝灭",
        aliases=["concentration quenching"],
        domain="materials",
    )
    store.add_entity(entity, track_version=False)

    foundation = build_concept_foundation(store, max_evidence_per_concept=0)
    mapping = next(
        item
        for item in foundation.mappings_for("concept:dy3:concentration-quenching")
        if item.asset_type is MappingAssetType.ENTITY
    )

    assert mapping.asset_id == entity.entity_id
    assert store.get_entity(entity.entity_id) is entity
    assert mapping.status is MappingStatus.CANDIDATE
    assert mapping.confidence == 0.0


def test_chunk_evidence_candidate_preserves_real_source_and_does_not_overclaim() -> None:
    store = KnowledgeStore()
    chunk = DocumentChunk(
        chunk_id="chunk-cq-001",
        document_id="paper-real-001",
        content="实验结果讨论了 Dy3+ 浓度猝灭，并比较了不同掺杂浓度。",
        section="Results",
        page=7,
    )
    store.add_chunk(chunk)

    foundation = build_concept_foundation(store, max_evidence_per_concept=2)
    evidence = foundation.evidence_for("concept:dy3:concentration-quenching")

    assert len(evidence) == 1
    item = evidence[0]
    assert item.chunk_id == chunk.chunk_id
    assert item.document_id == chunk.document_id
    assert item.section == "Results"
    assert item.page == 7
    assert item.evidence_role is EvidenceRole.MENTIONS
    assert item.status is MappingStatus.CANDIDATE
    assert item.confidence == 0.0
    assert item.claim_scope.startswith("term mention only:")
    assert "Figure" not in item.claim_scope


def test_missing_evidence_is_left_empty_without_fabrication() -> None:
    store = KnowledgeStore()
    store.add_chunk(DocumentChunk(
        chunk_id="chunk-unrelated",
        document_id="paper-unrelated",
        content="这段内容只讨论普通机械加工流程。",
    ))

    foundation = build_concept_foundation(store)
    concept = foundation.get_concept("concept:dy3:blue-light-hazard")

    assert concept is not None
    assert foundation.evidence_for(concept.concept_id) == ()
    assert concept.evidence_refs == ()


def test_unknown_page_remains_unknown() -> None:
    store = KnowledgeStore()
    store.add_chunk(DocumentChunk(
        chunk_id="chunk-cie-unknown-page",
        document_id="paper-cie",
        content="CIE色坐标用于描述样品的色度位置。",
        page=0,
    ))

    foundation = build_concept_foundation(store)
    evidence = foundation.evidence_for("concept:dy3:cie-chromaticity")

    assert evidence
    assert evidence[0].page is None


def test_builder_does_not_mutate_store_or_public_runtime_contracts() -> None:
    store = KnowledgeStore()
    store.add_entity(KnowledgeEntity(
        entity_id="entity-cie",
        entity_type=EntityType.CONCEPT,
        name="CIE色坐标",
    ), track_version=False)
    store.add_chunk(DocumentChunk(
        chunk_id="chunk-cie",
        document_id="paper-cie",
        content="本文测量CIE色坐标。",
    ))
    before = (store.entity_count(), store.triple_count(), store.chunk_count())

    foundation = build_concept_foundation(store)

    assert foundation.stats()["concepts"] == len(canonical_concepts())
    assert (store.entity_count(), store.triple_count(), store.chunk_count()) == before
    assert not hasattr(store, "concept_foundation")


def test_reviewed_concept_evidence_loads_with_stable_private_identity() -> None:
    store = KnowledgeStore()

    first = load_curated_concept_evidence(store)
    second = load_curated_concept_evidence(store)

    assert len(first) >= 10
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert store.chunk_count() == len(first)
    assert all(item.chunk_id.startswith("c-curated:") for item in first)
    assert all(
        item.metadata.get("source_type") == "curated_source_summary"
        for item in first
    )
    assert all(item.metadata.get("source_uri") for item in first)
    assert all(item.metadata.get("evidence_status") == "reviewed" for item in first)
    assert all(item.metadata.get("concept_ids") for item in first)
    assert all("answer" not in item.metadata for item in first)


def test_reviewed_concept_evidence_is_searchable_and_source_bounded() -> None:
    store = KnowledgeStore()
    load_curated_concept_evidence(store)

    thermal = store.search_text("热猝灭 非辐射", top_k=3)
    quantum = store.search_text("量子效率 积分球", top_k=3)
    hazard = store.search_text("蓝光危害 色温", top_k=3)

    assert any("https://doi.org/10.1039/D2TC04439K" == item.metadata["source_uri"] for item, _ in thermal)
    assert any("积分球" in item.content for item, _ in quantum)
    assert any("www.cie.co.at" in item.metadata["source_uri"] for item, _ in hazard)


def test_concept_evidence_can_be_retrieved_by_stable_mapping() -> None:
    store = KnowledgeStore()
    loaded = load_curated_concept_evidence(store)

    matches = store.find_chunks_by_metadata(
        "concept_ids",
        {"concept:dy3:host-lattice"},
    )

    assert matches
    assert {item.chunk_id for item in matches}.issubset(
        {item.chunk_id for item in loaded}
    )
    assert all(
        item.metadata.get("evidence_status") == "reviewed"
        for item in matches
    )
