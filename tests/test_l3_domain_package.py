"""Real knowledge-domain package boundary and migration checks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dy3_polaris.l3.domain_package import (
    DomainPackageError,
    active_domain_package,
    load_domain_package,
)
from dy3_polaris.l5.unified_app import _l3_snapshot_dir, _l3_snapshot_path


def _write_test_package(root: Path, *, package_id: str = "alternate-domain") -> Path:
    snapshot_root = root / "snapshots"
    snapshot = snapshot_root / "snapshot_release"
    snapshot.mkdir(parents=True)
    for filename in ("entities.jsonl", "triples.jsonl", "chunks.jsonl"):
        (snapshot / filename).write_text("{}\n", encoding="utf-8")
    (snapshot / "manifest.json").write_text(
        json.dumps({
            "format_version": "1.0",
            "counts": {"entities": 2, "triples": 1, "chunks": 3},
        }),
        encoding="utf-8",
    )
    manifest = root / "domain.json"
    manifest.write_text(
        json.dumps({
            "schema_version": "1.0",
            "package_id": package_id,
            "display_name": "Alternate reviewed domain",
            "domain": "alternate-scientific-domain",
            "version": "1.0.0",
            "language": "en",
            "snapshot_root": "snapshots",
            "active_snapshot": "snapshot_release",
            "concept_catalog": {
                "catalog_id": "alternate-concepts",
                "provider": "alternate.concepts:load",
            },
            "curriculum_catalog": {
                "catalog_id": "alternate-curriculum",
                "provider": "alternate.curriculum:load",
            },
            "source_policy": "source_required",
        }),
        encoding="utf-8",
    )
    return manifest


def test_shipped_dy3_package_validates_real_corpus() -> None:
    package = active_domain_package()
    validation = package.validate()

    assert package.package_id == "dy3-green-healthy-lighting"
    assert validation.valid is True
    assert validation.entity_count > 0
    assert validation.chunk_count > 0
    assert validation.graph_ready is True
    assert package.snapshot_path.name == "snapshot_final"


def test_alternate_package_resolves_without_dy3_paths(tmp_path: Path) -> None:
    package = load_domain_package(_write_test_package(tmp_path))

    assert package.package_id == "alternate-domain"
    assert package.snapshot_root == (tmp_path / "snapshots").resolve()
    assert package.snapshot_path == (tmp_path / "snapshots" / "snapshot_release").resolve()
    assert package.public_summary()["chunk_count"] == 3


def test_runtime_snapshot_resolution_uses_selected_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_test_package(tmp_path)
    monkeypatch.setenv("DY3_KNOWLEDGE_DOMAIN_MANIFEST", str(manifest))

    assert _l3_snapshot_dir() == (tmp_path / "snapshots").resolve()
    assert _l3_snapshot_path().name == "snapshot_release"


def test_invalid_or_empty_package_is_rejected(tmp_path: Path) -> None:
    manifest = _write_test_package(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["active_snapshot"] = "missing"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DomainPackageError, match="invalid domain package"):
        load_domain_package(manifest)


def test_public_summary_never_exposes_local_paths() -> None:
    summary = active_domain_package().public_summary()

    assert "snapshot_root" not in summary
    assert "snapshot_path" not in summary
    assert summary["source_policy"] == "reviewed_snapshot"
