from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dy3_polaris.l2.kp_catalog import NEW_KP_EDGES
from dy3_polaris.l3.concept_foundation import build_concept_foundation
from dy3_polaris.l3.concept_relations import (
    ConceptRelation,
    ConceptRelationNetwork,
    ConceptRelationType,
    RelationSource,
    RelationStatus,
    build_concept_relation_network,
    core_concept_relations,
)


def test_relation_can_be_created_and_is_read_only() -> None:
    relation = ConceptRelation(
        relation_id="relation:dy3:test-prerequisite",
        source_concept_id="concept:dy3:rare-earth-electron-configuration",
        target_concept_id="concept:dy3:four-f-shell",
        relation_type=ConceptRelationType.PREREQUISITE_OF,
        description="test relation",
        source=RelationSource.CURATED,
        confidence=1.0,
        status=RelationStatus.CURATED,
        source_reference="test:curated",
    )

    assert relation.relation_type is ConceptRelationType.PREREQUISITE_OF
    with pytest.raises(FrozenInstanceError):
        relation.description = "changed"  # type: ignore[misc]


def test_network_validates_all_source_and_target_concepts() -> None:
    network = build_concept_relation_network()

    assert network.stats()["concepts"] == len(network.foundation.concepts)
    # Isolated concepts are allowed when no verified relation is available.
    assert 37 <= network.stats()["participating_concepts"] <= network.stats()["concepts"]
    for relation in network.relations:
        assert network.foundation.get_concept(relation.source_concept_id) is not None
        assert network.foundation.get_concept(relation.target_concept_id) is not None


def test_unknown_concept_relation_is_rejected() -> None:
    foundation = build_concept_foundation()
    invalid = ConceptRelation(
        relation_id="relation:dy3:invalid-target",
        source_concept_id="concept:dy3:four-f-shell",
        target_concept_id="concept:dy3:not-a-real-concept",
        relation_type=ConceptRelationType.AFFECTS,
        description="invalid target",
        source=RelationSource.CURATED,
        confidence=1.0,
        status=RelationStatus.CURATED,
        source_reference="test:invalid",
    )

    with pytest.raises(ValueError, match="unknown target concept"):
        ConceptRelationNetwork(foundation=foundation, relations=(invalid,))
    with pytest.raises(KeyError, match="unknown concept"):
        build_concept_relation_network().find_path(
            "concept:dy3:not-a-real-concept",
            "concept:dy3:white-led",
        )


def test_relation_type_is_a_closed_runtime_control() -> None:
    assert {item.value for item in ConceptRelationType} == {
        "prerequisite_of",
        "explains",
        "causes",
        "affects",
        "evaluated_by",
        "applied_in",
    }

    with pytest.raises(ValueError, match="ConceptRelationType"):
        ConceptRelation(
            relation_id="relation:dy3:uncontrolled-type",
            source_concept_id="concept:dy3:four-f-shell",
            target_concept_id="concept:dy3:dy3-energy-level-structure",
            relation_type="similar_to",  # type: ignore[arg-type]
            description="uncontrolled",
            source=RelationSource.CURATED,
            confidence=1.0,
            status=RelationStatus.CURATED,
            source_reference="test:invalid",
        )


def test_foundational_to_application_concept_path_is_generated() -> None:
    network = build_concept_relation_network()
    path = network.find_path(
        "concept:dy3:rare-earth-electron-configuration",
        "concept:dy3:white-led",
    )

    assert path is not None
    assert path.source_concept_id == "concept:dy3:rare-earth-electron-configuration"
    assert path.target_concept_id == "concept:dy3:white-led"
    assert "concept:dy3:dy3-energy-level-structure" in path.concept_ids
    assert "concept:dy3:four-f-four-f-transition" in path.concept_ids
    assert "concept:dy3:single-phase-white-phosphor" in path.concept_ids
    assert path.hops <= 8
    assert network.find_path(
        path.source_concept_id, path.target_concept_id
    ) == path


def test_crystal_field_and_quenching_paths_preserve_real_edges() -> None:
    network = build_concept_relation_network()
    crystal_path = network.find_path(
        "concept:dy3:host-lattice",
        "concept:dy3:emission-spectrum",
    )
    quenching_path = network.find_path(
        "concept:dy3:doping-concentration",
        "concept:dy3:quantum-efficiency",
    )

    assert crystal_path is not None
    assert crystal_path.concept_ids == (
        "concept:dy3:host-lattice",
        "concept:dy3:local-coordination",
        "concept:dy3:crystal-field",
        "concept:dy3:stark-splitting",
        "concept:dy3:emission-spectrum",
    )
    assert quenching_path is not None
    assert quenching_path.concept_ids == (
        "concept:dy3:doping-concentration",
        "concept:dy3:concentration-quenching",
        "concept:dy3:quantum-efficiency",
    )


def test_ls_coupling_does_not_invent_a_direct_prerequisite_edge() -> None:
    network = build_concept_relation_network()
    relation = network.get_relation("relation:dy3:ls-rules-explains-energy-level")

    assert relation is not None
    assert relation.relation_type is ConceptRelationType.EXPLAINS
    assert relation.source is RelationSource.CURATED


def test_curriculum_expansion_relations_preserve_real_sources() -> None:
    network = build_concept_relation_network()

    process_path = network.find_path(
        "concept:dy3:reducing-atmosphere-valence",
        "concept:dy3:defects-traps",
    )
    thermal_path = network.find_path(
        "concept:dy3:thermal-quenching",
        "concept:dy3:thermal-stability-measurement",
    )

    assert process_path is not None
    assert process_path.concept_ids == (
        "concept:dy3:reducing-atmosphere-valence",
        "concept:dy3:process-defect-control",
        "concept:dy3:defects-traps",
    )
    assert thermal_path is not None
    assert thermal_path.hops == 1


def test_relations_are_sourced_and_not_created_from_cooccurrence() -> None:
    relations = core_concept_relations()

    assert relations
    assert all(item.status is RelationStatus.CURATED for item in relations)
    assert all(item.source in {RelationSource.CURATED, RelationSource.EXISTING_GRAPH} for item in relations)
    assert all(item.source_reference for item in relations)
    assert all(item.confidence == 1.0 for item in relations)
    assert all(item.source is not RelationSource.DERIVED for item in relations)


def test_existing_graph_sources_resolve_to_real_kp_edges() -> None:
    edge_keys = {
        (item["src"], item["rel"], item["dst"])
        for item in NEW_KP_EDGES
    }

    for relation in core_concept_relations():
        if relation.source is not RelationSource.EXISTING_GRAPH:
            continue
        encoded = relation.source_reference.split("NEW_KP_EDGES:", 1)[1]
        sources, predicate, target = encoded.split(":")
        assert all(
            (source, predicate, target) in edge_keys
            for source in sources.split("|")
        )


def test_relation_network_does_not_modify_foundation_or_public_runtime() -> None:
    foundation = build_concept_foundation()
    before = (
        tuple(foundation.concepts),
        foundation.mappings,
        foundation.evidence_mappings,
    )

    network = build_concept_relation_network(foundation)

    assert network.foundation is foundation
    assert (
        tuple(foundation.concepts),
        foundation.mappings,
        foundation.evidence_mappings,
    ) == before
    assert not hasattr(foundation, "relations")
