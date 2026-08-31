"""Validated, movable knowledge-domain package boundary.

The package does not replace retrieval or copy the L3 corpus.  It identifies
the reviewed corpus snapshot and the domain adapters that give that corpus its
teaching meaning.  A deployment can therefore point at another package without
changing the Agent or API execution chain.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


_PACKAGE_ENV = "DY3_KNOWLEDGE_DOMAIN_MANIFEST"
_SCHEMA_VERSION = "1.0"
_PACKAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
_REQUIRED_SNAPSHOT_FILES = ("entities.jsonl", "triples.jsonl", "chunks.jsonl")


class DomainPackageError(ValueError):
    """Raised when a knowledge-domain package cannot be trusted or loaded."""


@dataclass(frozen=True, slots=True)
class DomainPackageValidation:
    valid: bool
    errors: tuple[str, ...]
    entity_count: int
    triple_count: int
    chunk_count: int
    graph_ready: bool
    evidence_corpus_ready: bool


@dataclass(frozen=True, slots=True)
class KnowledgeDomainPackage:
    """Immutable description of one installable knowledge domain."""

    schema_version: str
    package_id: str
    display_name: str
    domain: str
    version: str
    language: str
    snapshot_root: Path
    active_snapshot: str
    concept_catalog: Mapping[str, str]
    curriculum_catalog: Mapping[str, str]
    source_policy: str
    manifest_path: Path

    @property
    def snapshot_path(self) -> Path:
        return (self.snapshot_root / self.active_snapshot).resolve()

    def validate(self) -> DomainPackageValidation:
        errors: list[str] = []
        if self.schema_version != _SCHEMA_VERSION:
            errors.append("unsupported_schema_version")
        if not _PACKAGE_ID_RE.fullmatch(self.package_id):
            errors.append("invalid_package_id")
        if not self.display_name or not self.domain or not self.version:
            errors.append("missing_identity_metadata")
        if not self.concept_catalog.get("provider"):
            errors.append("missing_concept_catalog_provider")
        if not self.curriculum_catalog.get("provider"):
            errors.append("missing_curriculum_catalog_provider")
        if self.source_policy not in {"source_required", "reviewed_snapshot"}:
            errors.append("invalid_source_policy")

        snapshot_path = self.snapshot_path
        try:
            snapshot_path.relative_to(self.snapshot_root.resolve())
        except ValueError:
            errors.append("active_snapshot_outside_root")

        snapshot_manifest = snapshot_path / "manifest.json"
        snapshot_payload: dict[str, Any] = {}
        if not snapshot_manifest.is_file():
            errors.append("missing_snapshot_manifest")
        else:
            try:
                parsed = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    snapshot_payload = parsed
                else:
                    errors.append("invalid_snapshot_manifest")
            except (OSError, UnicodeError, json.JSONDecodeError):
                errors.append("invalid_snapshot_manifest")

        for filename in _REQUIRED_SNAPSHOT_FILES:
            if not (snapshot_path / filename).is_file():
                errors.append(f"missing_snapshot_file:{filename}")

        counts = snapshot_payload.get("counts") or {}
        entity_count = _safe_count(counts.get("entities"))
        triple_count = _safe_count(counts.get("triples"))
        chunk_count = _safe_count(counts.get("chunks"))
        if entity_count <= 0:
            errors.append("empty_entity_corpus")
        if chunk_count <= 0:
            errors.append("empty_chunk_corpus")

        return DomainPackageValidation(
            valid=not errors,
            errors=tuple(dict.fromkeys(errors)),
            entity_count=entity_count,
            triple_count=triple_count,
            chunk_count=chunk_count,
            graph_ready=entity_count > 0 and triple_count > 0,
            evidence_corpus_ready=chunk_count > 0,
        )

    def public_summary(self) -> dict[str, Any]:
        """Return non-secret product facts; local filesystem paths stay private."""

        validation = self.validate()
        return {
            "package_id": self.package_id,
            "display_name": self.display_name,
            "domain": self.domain,
            "version": self.version,
            "language": self.language,
            "source_policy": self.source_policy,
            "concept_catalog_id": self.concept_catalog.get("catalog_id", ""),
            "curriculum_catalog_id": self.curriculum_catalog.get("catalog_id", ""),
            "entity_count": validation.entity_count,
            "triple_count": validation.triple_count,
            "chunk_count": validation.chunk_count,
            "graph_ready": validation.graph_ready,
            "evidence_corpus_ready": validation.evidence_corpus_ready,
            "valid": validation.valid,
            "validation_errors": list(validation.errors),
        }


def _safe_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def default_domain_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "domain_packages"
        / "dy3-green-healthy-lighting.json"
    )


def load_domain_package(path: str | Path) -> KnowledgeDomainPackage:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise DomainPackageError(f"domain package manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DomainPackageError("domain package manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise DomainPackageError("domain package manifest must be an object")

    snapshot_value = str(payload.get("snapshot_root") or "").strip()
    if not snapshot_value:
        raise DomainPackageError("domain package snapshot_root is required")
    snapshot_root = Path(snapshot_value)
    if not snapshot_root.is_absolute():
        snapshot_root = manifest_path.parent / snapshot_root

    concept_catalog = payload.get("concept_catalog") or {}
    curriculum_catalog = payload.get("curriculum_catalog") or {}
    if not isinstance(concept_catalog, dict) or not isinstance(curriculum_catalog, dict):
        raise DomainPackageError("domain package catalog adapters must be objects")

    package = KnowledgeDomainPackage(
        schema_version=str(payload.get("schema_version") or ""),
        package_id=str(payload.get("package_id") or ""),
        display_name=str(payload.get("display_name") or ""),
        domain=str(payload.get("domain") or ""),
        version=str(payload.get("version") or ""),
        language=str(payload.get("language") or "und"),
        snapshot_root=snapshot_root.resolve(),
        active_snapshot=str(payload.get("active_snapshot") or ""),
        concept_catalog=MappingProxyType(
            {str(key): str(value) for key, value in concept_catalog.items()}
        ),
        curriculum_catalog=MappingProxyType(
            {str(key): str(value) for key, value in curriculum_catalog.items()}
        ),
        source_policy=str(payload.get("source_policy") or ""),
        manifest_path=manifest_path,
    )
    validation = package.validate()
    if not validation.valid:
        raise DomainPackageError(
            "invalid domain package: " + ", ".join(validation.errors)
        )
    return package


def active_domain_package() -> KnowledgeDomainPackage:
    configured = os.environ.get(_PACKAGE_ENV, "").strip()
    return load_domain_package(configured or default_domain_manifest_path())


__all__ = [
    "DomainPackageError",
    "DomainPackageValidation",
    "KnowledgeDomainPackage",
    "active_domain_package",
    "default_domain_manifest_path",
    "load_domain_package",
]
