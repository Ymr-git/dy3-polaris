"""Conservative Claim-to-Evidence adapter over existing R03/R06 facts.

This module does not retrieve, review, or publish scientific content.  It
splits a selected Generation contribution into atomic R03 Claims and records
what the selected EvidencePack can actually establish.  A retrieved chunk is
never treated as support merely because its reranker score is high.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
from typing import Any, Iterable, Mapping

from dy3_polaris.l3.concept_foundation import build_concept_foundation
from dy3_polaris.l5.agent_contracts import Claim, ClaimType, EvidenceSupportLevel
from dy3_polaris.l5.retrieval_planning import EvidenceItem, EvidencePack


_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s+|\n+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]{2,}|[\u4e00-\u9fff]{2,6}")
_NUMBER_UNIT_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mol%|at%|wt%|%|nm|K|°C|℃|ms|μs|us|ns)",
    re.IGNORECASE,
)
_HOST_RE = re.compile(
    r"\b(?:Na|K|Li|Ca|Sr|Ba|Y|La|Gd|Lu|Al|Ga|Si|Ge|P|B|O|F|Cl|Br)"
    r"[A-Za-z0-9()\-]{2,24}\b"
)
_UNIVERSAL_TERMS = (
    "任何材料", "所有材料", "所有基质", "必然", "一定", "总是", "完全",
    "always", "all materials", "all hosts", "in all cases",
)
_CAUTIOUS_TERMS = ("可能", "通常", "在某些", "取决于", "可", "may", "can", "often")
_RECOMMENDATION_TERMS = ("建议", "可以进一步", "下一步", "应当", "推荐")
_INFERENCE_TERMS = ("可能", "推测", "表明", "说明", "因此", "归因", "可解释")
_STOPWORDS = {
    "因此", "由于", "以及", "其中", "可以", "可能", "通常", "当前", "这个", "一种",
    "the", "and", "that", "with", "from", "into", "this", "have", "has",
}


@dataclass(frozen=True, slots=True)
class EvidenceCondition:
    name: str
    value: str
    source: str


@dataclass(frozen=True, slots=True)
class ClaimEvidenceLink:
    claim_id: str
    evidence_id: str
    level: EvidenceSupportLevel
    reason: str
    source_reference: str
    chunk_reference: str
    conditions: tuple[EvidenceCondition, ...]
    evidence_version: int


@dataclass(frozen=True, slots=True)
class ScientificGrounding:
    task_id: str
    answer_identity: str
    review_identity: str
    claims: tuple[Claim, ...]
    links: tuple[ClaimEvidenceLink, ...]
    evidence_versions: tuple[int, ...]
    reviewer_status: str
    issue_codes: tuple[str, ...]
    identity_consistent: bool

    @property
    def support_counts(self) -> dict[str, int]:
        return {
            level.value: sum(1 for item in self.links if item.level is level)
            for level in EvidenceSupportLevel
        }


@dataclass(frozen=True, slots=True)
class CleanedEvidenceView:
    chunk_reference: str
    source_reference: str
    original_text: str
    cleaned_text: str
    transformations: tuple[str, ...]


def clean_mineru_evidence(
    text: str,
    *,
    chunk_reference: str,
    source_reference: str,
) -> CleanedEvidenceView:
    """Create a conservative display/review view while retaining raw text."""

    original = str(text or "")
    lines = [line.strip() for line in original.replace("\r\n", "\n").split("\n")]
    transformations: list[str] = []
    cleaned: list[str] = []
    previous = ""
    for line in lines:
        if not line:
            continue
        if re.fullmatch(r"(?:page|页)\s*\d+", line, re.IGNORECASE):
            transformations.append("page_marker_removed")
            continue
        line2 = re.sub(r"^\s*\d{1,4}\s*[|:]\s*", "", line)
        if line2 != line:
            transformations.append("line_number_removed")
        line2 = re.sub(r"\[(?:citation needed|\?+)\]", "", line2, flags=re.IGNORECASE)
        line2 = re.sub(r"\s+", " ", line2).strip()
        if not line2:
            continue
        if line2 == previous:
            transformations.append("duplicate_line_removed")
            continue
        cleaned.append(line2)
        previous = line2
    return CleanedEvidenceView(
        chunk_reference=str(chunk_reference or ""),
        source_reference=str(source_reference or ""),
        original_text=original,
        cleaned_text="\n".join(cleaned),
        transformations=tuple(dict.fromkeys(transformations)),
    )


def _claim_type(statement: str) -> ClaimType:
    lowered = statement.lower()
    if any(term in lowered for term in _RECOMMENDATION_TERMS):
        return ClaimType.RECOMMENDATION
    if any(term in lowered for term in _INFERENCE_TERMS):
        return ClaimType.INFERENCE
    if any(term in lowered for term in ("不确定", "尚不能", "证据不足", "unknown")):
        return ClaimType.UNCERTAIN
    return ClaimType.FACT


def _sentences(answer: str) -> tuple[str, ...]:
    values: list[str] = []
    for raw in _SENTENCE_RE.split(str(answer or "")):
        value = re.sub(r"^\s*(?:#{1,6}|[-*]|\d+[.)、])\s*", "", raw).strip()
        value = re.sub(r"\*\*|__|`", "", value).strip()
        if len(value) < 8 or value.endswith(":") or value.endswith("："):
            continue
        if value not in values:
            values.append(value[:600])
    return tuple(values[:12])


def _concept_ids(statement: str) -> tuple[str, ...]:
    lowered = statement.lower().replace("³⁺", "3+")
    foundation = build_concept_foundation()
    matches: list[str] = []
    for concept in foundation.concepts.values():
        terms = (concept.canonical_name, *concept.aliases)
        if any(str(term).lower().replace("³⁺", "3+") in lowered for term in terms if len(str(term)) >= 2):
            matches.append(concept.concept_id)
    return tuple(dict.fromkeys(matches))[:8]


def _conditions(text: str, source: str) -> tuple[EvidenceCondition, ...]:
    values: list[EvidenceCondition] = []
    for match in _NUMBER_UNIT_RE.finditer(text):
        unit = match.group("unit")
        name = "concentration" if "%" in unit else "wavelength" if unit.lower() == "nm" else "temperature" if unit in {"K", "°C", "℃"} else "lifetime"
        values.append(EvidenceCondition(name, f"{match.group('value')} {unit}", source))
    for match in _HOST_RE.finditer(text):
        value = match.group(0)
        if any(ch.isdigit() for ch in value):
            values.append(EvidenceCondition("host", value, source))
    return tuple(dict.fromkeys(values))[:16]


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(str(text or ""))
        if token.lower() not in _STOPWORDS and len(token) >= 2
    }


def _condition_mismatch(claim_text: str, evidence_text: str) -> bool:
    claim_values = {(m.group("value"), m.group("unit").lower()) for m in _NUMBER_UNIT_RE.finditer(claim_text)}
    evidence_values = {(m.group("value"), m.group("unit").lower()) for m in _NUMBER_UNIT_RE.finditer(evidence_text)}
    if not claim_values or not evidence_values:
        return False
    claim_units = {unit for _, unit in claim_values}
    evidence_units = {unit for _, unit in evidence_values}
    return bool(claim_units & evidence_units and not claim_values & evidence_values)


def _link_level(claim: Claim, item: EvidenceItem) -> tuple[EvidenceSupportLevel, str]:
    evidence = clean_mineru_evidence(
        item.content,
        chunk_reference=item.chunk_reference,
        source_reference=item.source_reference,
    ).cleaned_text
    claim_tokens = _tokens(claim.statement)
    evidence_tokens = _tokens(evidence)
    overlap = claim_tokens & evidence_tokens
    concept_hit = bool(set(claim.concept_ids) & set(_concept_ids(evidence)))
    if item.conflict_flag:
        return EvidenceSupportLevel.CONFLICTS, "evidence pack marks a conflicting result"
    exact = bool(claim.statement and claim.statement.lower() in evidence.lower())
    ratio = len(overlap) / max(1, len(claim_tokens))
    scientifically_related = exact or len(overlap) >= 2 or concept_hit
    # A different number in an unrelated retrieved passage is not a
    # condition conflict.  The old all-to-all check marked any numerical
    # source as contradictory before establishing that it addressed the same
    # claim, which made broader evidence packs less trustworthy than narrow
    # ones.
    if scientifically_related and _condition_mismatch(claim.statement, evidence):
        return EvidenceSupportLevel.INSUFFICIENT, "claim and evidence use different numeric conditions"
    if scientifically_related and any(term in claim.statement.lower() for term in _UNIVERSAL_TERMS) and (
        _conditions(evidence, item.evidence_id) or item.material_system
    ):
        return EvidenceSupportLevel.INSUFFICIENT, "condition-specific evidence cannot support a universal claim"
    if exact:
        return EvidenceSupportLevel.SUPPORTS, "the source contains the atomic claim in the same scope"
    if len(overlap) >= 3 and ratio >= 0.45:
        return EvidenceSupportLevel.SUPPORTS, "substantial claim terms occur together in the source"
    if len(overlap) >= 2 or concept_hit:
        return EvidenceSupportLevel.CANDIDATE, "scientifically related but direct support is not established"
    if len(overlap) == 1:
        return EvidenceSupportLevel.MENTION, "the source only mentions one relevant term"
    return EvidenceSupportLevel.INSUFFICIENT, "no claim-specific support found"


def atomic_claims(
    answer: str,
    *,
    contribution_id: str,
    answer_identity: str,
    confidence: float,
) -> tuple[Claim, ...]:
    claims: list[Claim] = []
    for index, statement in enumerate(_sentences(answer), start=1):
        digest = hashlib.sha256(f"{contribution_id}|{statement}".encode("utf-8")).hexdigest()[:14]
        conditions = _conditions(statement, f"claim:{digest}")
        claims.append(Claim(
            claim_id=f"claim-{digest}",
            statement=statement,
            claim_type=_claim_type(statement),
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            concept_ids=_concept_ids(statement),
            scope="condition_limited" if conditions or any(term in statement.lower() for term in _CAUTIOUS_TERMS) else "unspecified",
            conditions=tuple((item.name, item.value) for item in conditions),
            answer_identity=answer_identity,
        ))
    return tuple(claims)


def build_scientific_grounding(
    *,
    task_id: str,
    answer_identity: str,
    claims: Iterable[Claim],
    evidence_packs: Iterable[EvidencePack],
    review_identity: str = "",
    reviewer_status: str = "not_reviewed",
) -> ScientificGrounding:
    packs = tuple(pack for pack in evidence_packs if isinstance(pack, EvidencePack))
    versions = tuple(sorted({pack.version for pack in packs}))
    links: list[ClaimEvidenceLink] = []
    updated: list[Claim] = []
    issues: list[str] = []
    for claim in claims:
        claim_links: list[ClaimEvidenceLink] = []
        seen_evidence: set[tuple[str, str, str]] = set()
        for pack in packs:
            for item in pack.items:
                level, reason = _link_level(claim, item)
                if level is EvidenceSupportLevel.INSUFFICIENT and reason == "no claim-specific support found":
                    continue
                # Multi-candidate generation can carry the same retrieved chunk
                # through more than one EvidencePack.  A physical source chunk is
                # still one piece of evidence, not one item per candidate.  Keep
                # evidence with different support semantics separate, while
                # collapsing repeated references to the same source/chunk.
                evidence_key = (
                    item.source_reference or "",
                    item.chunk_reference or "",
                    level.value,
                )
                if not evidence_key[0] and not evidence_key[1]:
                    evidence_key = (item.evidence_id, "", level.value)
                if evidence_key in seen_evidence:
                    continue
                seen_evidence.add(evidence_key)
                claim_links.append(ClaimEvidenceLink(
                    claim_id=claim.claim_id,
                    evidence_id=item.evidence_id,
                    level=level,
                    reason=reason,
                    source_reference=item.source_reference,
                    chunk_reference=item.chunk_reference,
                    conditions=_conditions(item.content, item.evidence_id),
                    evidence_version=pack.version,
                ))
        links.extend(claim_links)
        levels = {link.level for link in claim_links}
        status = (
            EvidenceSupportLevel.CONFLICTS if EvidenceSupportLevel.CONFLICTS in levels
            else EvidenceSupportLevel.SUPPORTS if EvidenceSupportLevel.SUPPORTS in levels
            else EvidenceSupportLevel.CANDIDATE if EvidenceSupportLevel.CANDIDATE in levels
            else EvidenceSupportLevel.MENTION if EvidenceSupportLevel.MENTION in levels
            else EvidenceSupportLevel.INSUFFICIENT
        )
        if status is EvidenceSupportLevel.CONFLICTS:
            issues.append("conflicting_evidence")
        if (
            EvidenceSupportLevel.SUPPORTS not in levels
            and any("different numeric conditions" in link.reason for link in claim_links)
        ):
            issues.append("condition_mismatch")
        if (
            EvidenceSupportLevel.SUPPORTS not in levels
            and any("universal claim" in link.reason for link in claim_links)
        ):
            issues.append("unsupported_universalization")
        if status in {EvidenceSupportLevel.MENTION, EvidenceSupportLevel.INSUFFICIENT} and claim.claim_type is ClaimType.FACT:
            issues.append("fact_not_directly_supported")
        supported_refs = tuple(dict.fromkeys(
            link.evidence_id for link in claim_links if link.level is EvidenceSupportLevel.SUPPORTS
        ))
        source_refs = tuple(dict.fromkeys(link.source_reference for link in claim_links if link.source_reference))
        provenance_refs = tuple(dict.fromkeys(link.chunk_reference for link in claim_links if link.chunk_reference))
        updated.append(replace(
            claim,
            evidence_refs=supported_refs,
            support_status=status,
            source_refs=source_refs,
            provenance_refs=provenance_refs,
            reviewer_status=reviewer_status,
            evidence_version=max(versions, default=0),
        ))
    identity_consistent = bool(answer_identity) and all(
        claim.answer_identity == answer_identity for claim in updated
    )
    if review_identity and review_identity != answer_identity:
        identity_consistent = False
    if not identity_consistent:
        issues.append("claim_evidence_review_identity_mismatch")
    return ScientificGrounding(
        task_id=task_id,
        answer_identity=answer_identity,
        review_identity=review_identity,
        claims=tuple(updated),
        links=tuple(links),
        evidence_versions=versions,
        reviewer_status=reviewer_status,
        issue_codes=tuple(dict.fromkeys(issues)),
        identity_consistent=identity_consistent,
    )


def public_scientific_grounding_projection(
    grounding: ScientificGrounding,
    *,
    release_eligible: bool,
) -> dict[str, Any]:
    if not isinstance(grounding, ScientificGrounding):
        return {}
    if not release_eligible:
        return {
            "status": "WITHHELD",
            "claims": [],
            "support_counts": grounding.support_counts,
            "issues": list(grounding.issue_codes),
            "identity_consistent": grounding.identity_consistent,
            "source_class": "DERIVED_FROM_LOCAL_EVIDENCE",
        }
    claims = []
    for claim in grounding.claims:
        links = [link for link in grounding.links if link.claim_id == claim.claim_id]
        claims.append({
            "claim_id": claim.claim_id,
            "statement": claim.statement if release_eligible else "",
            "claim_type": claim.claim_type.value,
            "concept_ids": list(claim.concept_ids),
            "scope": claim.scope or "UNKNOWN",
            "conditions": [{"name": name, "value": value} for name, value in claim.conditions],
            "support_status": claim.support_status.value,
            "reviewer_status": claim.reviewer_status,
            "evidence": [{
                "evidence_id": link.evidence_id,
                "level": link.level.value,
                "reason": link.reason,
                "source": link.source_reference,
                "chunk_id": link.chunk_reference,
                "conditions": [{"name": item.name, "value": item.value} for item in link.conditions],
            } for link in links if link.level is not EvidenceSupportLevel.INSUFFICIENT],
        })
    return {
        "status": "AVAILABLE" if claims else "UNAVAILABLE",
        "claims": claims,
        "support_counts": grounding.support_counts,
        "issues": list(grounding.issue_codes),
        "identity_consistent": grounding.identity_consistent,
        "source_class": "DERIVED_FROM_LOCAL_EVIDENCE",
    }


__all__ = [
    "ClaimEvidenceLink",
    "CleanedEvidenceView",
    "EvidenceCondition",
    "ScientificGrounding",
    "atomic_claims",
    "build_scientific_grounding",
    "clean_mineru_evidence",
    "public_scientific_grounding_projection",
]
